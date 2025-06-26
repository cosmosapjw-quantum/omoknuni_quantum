"""UV Cutoff for Quantum MCTS

This module implements the UV cutoff mechanism from the v4.0 theoretical framework:

N_UV = N_parent^α_UV

Where empirically α_UV ∈ [0.3, 0.7] balances exploration and exploitation.
The theoretical value is: α_UV = 1/(1 + ε_coh * ΔE_N)

This provides principled thresholding for quantum corrections and controls
the balance between quantum exploration and classical exploitation.
"""

import torch
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CutoffMethod(Enum):
    """Methods for computing UV cutoff"""
    EMPIRICAL = "empirical"          # Use empirical α_UV value
    THEORETICAL = "theoretical"       # Compute from coherence and energy
    ADAPTIVE = "adaptive"            # Adapt based on performance
    HYBRID = "hybrid"               # Combine multiple methods


@dataclass 
class UVCutoffConfig:
    """Configuration for UV cutoff mechanism"""
    
    # Cutoff parameters
    alpha_UV: float = 0.5               # Default UV cutoff exponent
    alpha_UV_range: Tuple[float, float] = (0.3, 0.7)  # Empirical range
    
    # Theoretical parameters  
    epsilon_coherence: float = 0.1      # Coherence scale parameter
    beta_energy: float = 1.0           # Energy scale parameter
    
    # Cutoff method
    cutoff_method: CutoffMethod = CutoffMethod.EMPIRICAL
    
    # Adaptive parameters
    adaptation_rate: float = 0.01       # Rate of α_UV adaptation
    performance_window: int = 100       # Window for performance evaluation
    target_exploration_ratio: float = 0.4  # Target exploration/exploitation ratio
    
    # Numerical parameters
    min_cutoff: int = 1                # Minimum cutoff value
    max_cutoff: int = 10000            # Maximum cutoff value
    
    # Performance tracking
    track_exploration_stats: bool = True   # Track exploration statistics
    track_exploitation_stats: bool = True  # Track exploitation statistics
    
    # Device configuration
    device: str = 'cuda'


class UVCutoffMechanism:
    """Implements UV cutoff for quantum MCTS"""
    
    def __init__(self, config: UVCutoffConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Current cutoff parameters
        self.current_alpha_UV = config.alpha_UV
        
        # Performance tracking
        self.performance_history = {
            'exploration_scores': [],
            'exploitation_scores': [], 
            'balance_scores': [],
            'alpha_UV_values': [],
            'cutoff_values': [],
            'parent_visits': []
        }
        
        # Statistics
        self.stats = {
            'cutoffs_computed': 0,
            'adaptations_performed': 0,
            'average_cutoff_value': 0.0,
            'average_alpha_UV': 0.0
        }
    
    def compute_uv_cutoff(self, 
                         N_parent: int,
                         coherence_data: Optional[Dict] = None,
                         energy_data: Optional[Dict] = None) -> int:
        """Compute UV cutoff N_UV = N_parent^α_UV
        
        Args:
            N_parent: Visit count of parent node
            coherence_data: Coherence information for theoretical calculation
            energy_data: Energy scale information for theoretical calculation
            
        Returns:
            UV cutoff threshold
        """
        if N_parent <= 0:
            return self.config.min_cutoff
        
        # Determine α_UV based on method
        if self.config.cutoff_method == CutoffMethod.EMPIRICAL:
            alpha_UV = self.current_alpha_UV
            
        elif self.config.cutoff_method == CutoffMethod.THEORETICAL:
            alpha_UV = self._compute_theoretical_alpha_UV(coherence_data, energy_data)
            
        elif self.config.cutoff_method == CutoffMethod.ADAPTIVE:
            alpha_UV = self._compute_adaptive_alpha_UV()
            
        elif self.config.cutoff_method == CutoffMethod.HYBRID:
            alpha_UV = self._compute_hybrid_alpha_UV(coherence_data, energy_data)
            
        else:
            alpha_UV = self.current_alpha_UV
        
        # Clamp to empirical range
        alpha_UV = np.clip(alpha_UV, 
                          self.config.alpha_UV_range[0], 
                          self.config.alpha_UV_range[1])
        
        # Compute cutoff: N_UV = N_parent^α_UV
        N_UV_raw = N_parent ** alpha_UV
        N_UV = int(np.clip(N_UV_raw, self.config.min_cutoff, self.config.max_cutoff))
        
        # Update statistics
        self.stats['cutoffs_computed'] += 1
        self.stats['average_cutoff_value'] = (
            (self.stats['average_cutoff_value'] * (self.stats['cutoffs_computed'] - 1) + N_UV) 
            / self.stats['cutoffs_computed']
        )
        self.stats['average_alpha_UV'] = (
            (self.stats['average_alpha_UV'] * (self.stats['cutoffs_computed'] - 1) + alpha_UV)
            / self.stats['cutoffs_computed']
        )
        
        # Record for adaptation
        self.performance_history['alpha_UV_values'].append(alpha_UV)
        self.performance_history['cutoff_values'].append(N_UV)
        self.performance_history['parent_visits'].append(N_parent)
        
        return N_UV
    
    def _compute_theoretical_alpha_UV(self, 
                                    coherence_data: Optional[Dict],
                                    energy_data: Optional[Dict]) -> float:
        """Compute theoretical α_UV = 1/(1 + ε_coh * ΔE_N)"""
        if coherence_data is None or energy_data is None:
            logger.warning("Missing data for theoretical α_UV, using empirical value")
            return self.current_alpha_UV
        
        try:
            # Extract coherence and energy scales
            epsilon_coh = coherence_data.get('coherence_scale', self.config.epsilon_coherence)
            delta_E_N = energy_data.get('energy_difference', 
                                      self.config.beta_energy * energy_data.get('mean_q_value', 1.0))
            
            # Theoretical formula
            alpha_UV_theory = 1.0 / (1.0 + epsilon_coh * delta_E_N)
            
            return alpha_UV_theory
            
        except Exception as e:
            logger.warning(f"Theoretical α_UV computation failed: {e}")
            return self.current_alpha_UV
    
    def _compute_adaptive_alpha_UV(self) -> float:
        """Adapt α_UV based on exploration/exploitation performance"""
        if len(self.performance_history['balance_scores']) < 2:
            return self.current_alpha_UV
        
        # Get recent performance
        recent_scores = self.performance_history['balance_scores'][-self.config.performance_window:]
        recent_alphas = self.performance_history['alpha_UV_values'][-self.config.performance_window:]
        
        if len(recent_scores) < 5:
            return self.current_alpha_UV
        
        # Compute performance gradient
        try:
            scores_tensor = torch.tensor(recent_scores, device=self.device)
            alphas_tensor = torch.tensor(recent_alphas, device=self.device)
            
            # Linear regression: score = a * alpha + b
            X = torch.stack([alphas_tensor, torch.ones_like(alphas_tensor)], dim=1)
            coeffs = torch.linalg.lstsq(X, scores_tensor).solution
            gradient = coeffs[0].item()
            
            # Adapt in direction of gradient
            adaptation = self.config.adaptation_rate * gradient
            new_alpha = self.current_alpha_UV + adaptation
            
            return new_alpha
            
        except Exception as e:
            logger.debug(f"Adaptive α_UV computation failed: {e}")
            return self.current_alpha_UV
    
    def _compute_hybrid_alpha_UV(self,
                               coherence_data: Optional[Dict],
                               energy_data: Optional[Dict]) -> float:
        """Combine empirical, theoretical, and adaptive methods"""
        weights = {'empirical': 0.4, 'theoretical': 0.3, 'adaptive': 0.3}
        
        # Empirical component
        alpha_empirical = self.current_alpha_UV
        
        # Theoretical component
        alpha_theoretical = self._compute_theoretical_alpha_UV(coherence_data, energy_data)
        
        # Adaptive component
        alpha_adaptive = self._compute_adaptive_alpha_UV()
        
        # Weighted combination
        alpha_hybrid = (weights['empirical'] * alpha_empirical +
                       weights['theoretical'] * alpha_theoretical +
                       weights['adaptive'] * alpha_adaptive)
        
        return alpha_hybrid
    
    def update_performance_metrics(self,
                                 exploration_score: float,
                                 exploitation_score: float):
        """Update performance metrics for adaptation"""
        # Compute balance score (harmonic mean)
        if exploration_score > 0 and exploitation_score > 0:
            balance_score = 2 * exploration_score * exploitation_score / (
                exploration_score + exploitation_score
            )
        else:
            balance_score = 0.0
        
        # Record metrics
        self.performance_history['exploration_scores'].append(exploration_score)
        self.performance_history['exploitation_scores'].append(exploitation_score)
        self.performance_history['balance_scores'].append(balance_score)
        
        # Adaptation if enabled
        if self.config.cutoff_method in [CutoffMethod.ADAPTIVE, CutoffMethod.HYBRID]:
            self._perform_adaptation()
    
    def _perform_adaptation(self):
        """Perform adaptation of α_UV based on performance"""
        if len(self.performance_history['balance_scores']) < self.config.performance_window:
            return
        
        # Check if adaptation should be performed
        recent_balance = np.mean(self.performance_history['balance_scores'][-10:])
        
        # Get current exploration/exploitation ratio
        recent_exploration = np.mean(self.performance_history['exploration_scores'][-10:])
        recent_exploitation = np.mean(self.performance_history['exploitation_scores'][-10:])
        
        if recent_exploitation > 0:
            current_ratio = recent_exploration / recent_exploitation
            target_ratio = self.config.target_exploration_ratio
            
            # Adjust α_UV to move toward target ratio
            if current_ratio < target_ratio * 0.8:  # Too much exploitation
                self.current_alpha_UV = min(self.current_alpha_UV + 0.01, 
                                          self.config.alpha_UV_range[1])
                self.stats['adaptations_performed'] += 1
                
            elif current_ratio > target_ratio * 1.2:  # Too much exploration
                self.current_alpha_UV = max(self.current_alpha_UV - 0.01, 
                                          self.config.alpha_UV_range[0])
                self.stats['adaptations_performed'] += 1
    
    def validate_cutoff_performance(self) -> Dict[str, Any]:
        """Validate UV cutoff performance against theoretical predictions"""
        if len(self.performance_history['balance_scores']) < 10:
            return {'insufficient_data': True}
        
        # Analyze performance vs α_UV
        alphas = np.array(self.performance_history['alpha_UV_values'])
        scores = np.array(self.performance_history['balance_scores'])
        
        # Find optimal α_UV empirically
        if len(alphas) > 0:
            unique_alphas = np.unique(alphas)
            alpha_performance = {}
            
            for alpha in unique_alphas:
                mask = np.abs(alphas - alpha) < 0.05  # Tolerance for grouping
                if np.sum(mask) > 0:
                    alpha_performance[alpha] = np.mean(scores[mask])
            
            if alpha_performance:
                optimal_alpha = max(alpha_performance.keys(), 
                                  key=lambda a: alpha_performance[a])
                max_performance = alpha_performance[optimal_alpha]
                
                # Check if optimal is in theoretical range
                in_range = (self.config.alpha_UV_range[0] <= optimal_alpha <= 
                           self.config.alpha_UV_range[1])
                
                return {
                    'optimal_alpha_UV': optimal_alpha,
                    'max_performance': max_performance,
                    'in_theoretical_range': in_range,
                    'alpha_performance_map': alpha_performance,
                    'num_data_points': len(alphas)
                }
        
        return {'analysis_failed': True}
    
    def get_cutoff_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cutoff statistics"""
        stats = self.stats.copy()
        
        if self.performance_history['cutoff_values']:
            stats['cutoff_range'] = {
                'min': min(self.performance_history['cutoff_values']),
                'max': max(self.performance_history['cutoff_values']),
                'mean': np.mean(self.performance_history['cutoff_values']),
                'std': np.std(self.performance_history['cutoff_values'])
            }
        
        if self.performance_history['alpha_UV_values']:
            stats['alpha_UV_range'] = {
                'min': min(self.performance_history['alpha_UV_values']),
                'max': max(self.performance_history['alpha_UV_values']),
                'mean': np.mean(self.performance_history['alpha_UV_values']),
                'std': np.std(self.performance_history['alpha_UV_values'])
            }
        
        if self.performance_history['balance_scores']:
            stats['performance_stats'] = {
                'mean_balance': np.mean(self.performance_history['balance_scores']),
                'mean_exploration': np.mean(self.performance_history['exploration_scores']),
                'mean_exploitation': np.mean(self.performance_history['exploitation_scores'])
            }
        
        return stats
    
    def reset_adaptation(self):
        """Reset adaptation history"""
        self.current_alpha_UV = self.config.alpha_UV
        self.performance_history = {
            'exploration_scores': [],
            'exploitation_scores': [],
            'balance_scores': [],
            'alpha_UV_values': [],
            'cutoff_values': [],
            'parent_visits': []
        }


def create_uv_cutoff_mechanism(config: Optional[UVCutoffConfig] = None) -> UVCutoffMechanism:
    """Factory function to create UVCutoffMechanism with default configuration"""
    if config is None:
        config = UVCutoffConfig()
    
    return UVCutoffMechanism(config)


# Export main classes and functions
__all__ = [
    'UVCutoffMechanism',
    'UVCutoffConfig',
    'CutoffMethod',
    'create_uv_cutoff_mechanism'
]