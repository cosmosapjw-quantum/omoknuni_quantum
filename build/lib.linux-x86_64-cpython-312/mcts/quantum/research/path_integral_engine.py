"""Path Integral Engine for Quantum-Inspired MCTS

This module implements the rigorous path integral formulation for MCTS based on 
the quantum field theory framework. Each path in the MCTS tree represents a 
quantum particle trajectory, and the collection of paths (wave) forms a 
quantum superposition.

Key theoretical foundations:
- Lagrangian density: L(u_k; N^pre) = log[N^pre(u_k) + ε_N] + λ log P(u_k) - β Q(u_k)
- Action functional: S[γ] = Σ_k L(u_k) for path γ = (u_0, u_1, ..., u_{L-1})
- Information time: τ(N) = log(N + 2)
- Effective Planck constant: ℏ_eff(N) = ℏ[1 + Γ_N/2]
- Path probability: P(γ) ∝ exp(-S[γ]/ℏ_eff)

Performance target: < 2x overhead compared to classical MCTS
"""

import torch
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class InformationTimeMode(Enum):
    """Information time formulation modes"""
    DISCRETE = "discrete"    # τ(N) = log(N + 2) - v4.0 formulation
    CONTINUOUS = "continuous"  # τ(N) = N - classical formulation


@dataclass
class PathIntegralConfig:
    """Configuration for path integral computation"""
    
    # Fundamental constants
    hbar: float = 1.0  # Fundamental Planck constant
    temperature: float = 1.0  # System temperature
    
    # Lagrangian parameters (from v4.0 theory)
    lambda_prior: float = 1.4  # Prior coupling strength (λ in theory)
    beta_value: float = 1.0    # Value coupling strength (β in theory)
    epsilon_N: float = 1e-8    # Visit count regularization
    
    # Information time formulation
    time_mode: InformationTimeMode = InformationTimeMode.DISCRETE
    
    # Quantum parameters
    enable_quantum_corrections: bool = True
    quantum_interference: bool = True
    decoherence_rate: float = 0.01  # γ_0 in theory
    
    # Performance optimization
    use_lookup_tables: bool = True
    max_table_size: int = 10000
    batch_size: int = 3072  # Match wave size
    use_mixed_precision: bool = True
    cache_actions: bool = True
    
    # Numerical stability
    min_log_arg: float = 1e-12
    max_action_value: float = 100.0
    
    # Device configuration
    device: str = 'cuda'


class PathIntegralEngine:
    """Core engine for path integral computation in quantum MCTS"""
    
    def __init__(self, config: PathIntegralConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Pre-computed lookup tables for efficiency
        self._lookup_tables: Dict[str, torch.Tensor] = {}
        self._table_initialized = False
        
        # Cache for frequently accessed values
        self._cache: Dict[str, torch.Tensor] = {}
        
        # Performance statistics
        self.stats = {
            'total_paths_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'computation_time': 0.0
        }
        
        # Initialize lookup tables if enabled
        if config.use_lookup_tables:
            self._initialize_lookup_tables()
    
    def _initialize_lookup_tables(self):
        """Initialize pre-computed lookup tables for efficiency"""
        logger.info("Initializing path integral lookup tables...")
        
        # Information time lookup: τ(N) = log(N + 2)
        N_values = torch.arange(1, self.config.max_table_size + 1, 
                               dtype=torch.float32, device=self.device)
        
        if self.config.time_mode == InformationTimeMode.DISCRETE:
            self._lookup_tables['info_time'] = torch.log(N_values + 2)
        else:
            self._lookup_tables['info_time'] = N_values
        
        # Effective Planck constant: ℏ_eff(N) = ℏ[1 + Γ_N/2]
        # Simplified: ℏ_eff ≈ ℏ[1 + γ_0/(2√N)]
        gamma_N = self.config.decoherence_rate / (2 * torch.sqrt(N_values))
        self._lookup_tables['hbar_eff'] = self.config.hbar * (1 + gamma_N)
        
        # Regularized log for visit counts: log(N + ε_N)
        self._lookup_tables['log_visits'] = torch.log(N_values + self.config.epsilon_N)
        
        # Temperature factors for numerical stability
        self._lookup_tables['temp_factors'] = torch.exp(-torch.arange(0, 10, 0.1, device=self.device))
        
        self._table_initialized = True
        logger.info(f"Lookup tables initialized with {self.config.max_table_size} entries")
    
    def information_time(self, visit_counts: torch.Tensor) -> torch.Tensor:
        """Compute information time τ(N) = log(N + 2)
        
        Args:
            visit_counts: Tensor of visit counts [batch_size, ...]
            
        Returns:
            Information time values [batch_size, ...]
        """
        if self.config.use_lookup_tables and self._table_initialized:
            # Use lookup table for small values
            small_mask = visit_counts < self.config.max_table_size
            result = torch.zeros_like(visit_counts, dtype=torch.float32)
            
            # Lookup for small values
            if small_mask.any():
                small_indices = torch.clamp(visit_counts[small_mask].long() - 1, 0, 
                                          self.config.max_table_size - 1)
                result[small_mask] = self._lookup_tables['info_time'][small_indices]
            
            # Direct computation for large values
            large_mask = ~small_mask
            if large_mask.any():
                if self.config.time_mode == InformationTimeMode.DISCRETE:
                    result[large_mask] = torch.log(visit_counts[large_mask] + 2)
                else:
                    result[large_mask] = visit_counts[large_mask]
            
            return result
        else:
            # Direct computation
            if self.config.time_mode == InformationTimeMode.DISCRETE:
                return torch.log(visit_counts + 2)
            else:
                return visit_counts
    
    def effective_planck_constant(self, visit_counts: torch.Tensor) -> torch.Tensor:
        """Compute effective Planck constant using exact ℏ_eff derivation
        
        This method integrates with the exact ℏ_eff computation from exact_hbar_effective.py
        which derives ℏ_eff from the fundamental Lindblad equation using observable-matching:
        ℏ_eff(N) = ℏ_base / arccos(exp(-Γ_N/2))
        
        Args:
            visit_counts: Tensor of visit counts [batch_size, ...]
            
        Returns:
            Effective Planck constants [batch_size, ...]
        """
        try:
            # Try to use exact ℏ_eff computation if available
            from .exact_hbar_effective import create_exact_hbar_computer
            
            exact_computer = create_exact_hbar_computer()
            
            if visit_counts.numel() == 1:
                # Single value
                total_visits = int(visit_counts.item())
                hbar_eff = exact_computer.compute_exact_hbar_eff(total_visits)
                return torch.tensor(hbar_eff, device=visit_counts.device, dtype=torch.float32)
            else:
                # Batch computation
                result = torch.zeros_like(visit_counts, dtype=torch.float32)
                for i, count in enumerate(visit_counts.flatten()):
                    hbar_eff = exact_computer.compute_exact_hbar_eff(int(count.item()))
                    result.flatten()[i] = hbar_eff
                return result
                
        except ImportError:
            # Fallback to legacy approximation if exact computation not available
            gamma_N = self.config.decoherence_rate / (2 * torch.sqrt(visit_counts + 1))
            return self.config.hbar * (1 + gamma_N)
    
    def lagrangian_density(self, 
                          visit_counts: torch.Tensor,
                          priors: torch.Tensor,
                          q_values: torch.Tensor) -> torch.Tensor:
        """Compute Lagrangian density L(u_k; N^pre) for path elements
        
        From v4.0 theory:
        L(u_k; N^pre) = log[N^pre(u_k) + ε_N] + λ log P(u_k) - β Q(u_k)
        
        Args:
            visit_counts: Pre-rollout visit counts N^pre(u_k) [batch_size, path_length]
            priors: Neural network priors P(u_k) [batch_size, path_length]
            q_values: Q-values Q(u_k) [batch_size, path_length]
            
        Returns:
            Lagrangian density values [batch_size, path_length]
        """
        # Ensure numerical stability
        safe_visits = torch.clamp(visit_counts, min=0)
        safe_priors = torch.clamp(priors, min=self.config.min_log_arg, max=1.0)
        
        # Compute each term
        # Term 1: log[N^pre(u_k) + ε_N]
        log_visits_term = torch.log(safe_visits + self.config.epsilon_N)
        
        # Term 2: λ log P(u_k)
        prior_term = self.config.lambda_prior * torch.log(safe_priors)
        
        # Term 3: -β Q(u_k)
        value_term = -self.config.beta_value * q_values
        
        # Combine terms
        lagrangian = log_visits_term + prior_term + value_term
        
        # Numerical stability check
        lagrangian = torch.clamp(lagrangian, 
                               min=-self.config.max_action_value,
                               max=self.config.max_action_value)
        
        return lagrangian
    
    def path_action(self, 
                   path_visits: torch.Tensor,
                   path_priors: torch.Tensor,
                   path_qvalues: torch.Tensor,
                   path_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute path action S[γ] = Σ_k L(u_k) for multiple paths
        
        Args:
            path_visits: Visit counts for each path element [batch_size, max_path_length]
            path_priors: Prior probabilities [batch_size, max_path_length]
            path_qvalues: Q-values [batch_size, max_path_length]
            path_mask: Valid path elements mask [batch_size, max_path_length]
            
        Returns:
            Path actions S[γ] [batch_size]
        """
        # Compute Lagrangian density for all path elements
        lagrangian = self.lagrangian_density(path_visits, path_priors, path_qvalues)
        
        # Apply mask if provided
        if path_mask is not None:
            lagrangian = lagrangian * path_mask.float()
        
        # Sum over path elements to get total action
        path_actions = lagrangian.sum(dim=1)
        
        return path_actions
    
    def transition_amplitude(self, 
                           path_actions: torch.Tensor,
                           visit_counts: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute transition amplitude P(γ) ∝ exp(-S[γ]/ℏ_eff)
        
        Args:
            path_actions: Path actions S[γ] [batch_size]
            visit_counts: Representative visit counts for ℏ_eff computation [batch_size]
                         If None, uses mean visit count for each path
            
        Returns:
            Unnormalized transition amplitudes [batch_size]
        """
        if visit_counts is None:
            # Use a representative visit count
            visit_counts = torch.ones_like(path_actions) * 100  # Reasonable default
        
        # Compute effective Planck constant
        hbar_eff = self.effective_planck_constant(visit_counts)
        
        # Compute temperature (can be annealed)
        temperature = self.config.temperature
        
        # Compute amplitude: exp(-S[γ]/(ℏ_eff * T))
        exponent = -path_actions / (hbar_eff * temperature)
        
        # Numerical stability
        exponent = torch.clamp(exponent, min=-50, max=50)
        
        amplitudes = torch.exp(exponent)
        
        return amplitudes
    
    def normalize_amplitudes(self, amplitudes: torch.Tensor) -> torch.Tensor:
        """Normalize transition amplitudes to form probabilities
        
        Args:
            amplitudes: Unnormalized amplitudes [batch_size]
            
        Returns:
            Normalized probabilities [batch_size]
        """
        # Add small epsilon for numerical stability
        epsilon = 1e-12
        amplitudes = amplitudes + epsilon
        
        # Normalize
        total = amplitudes.sum()
        if total > epsilon:
            probabilities = amplitudes / total
        else:
            # Uniform distribution as fallback
            probabilities = torch.ones_like(amplitudes) / len(amplitudes)
        
        return probabilities
    
    def compute_path_integral_batch(self,
                                  path_visits: torch.Tensor,
                                  path_priors: torch.Tensor,
                                  path_qvalues: torch.Tensor,
                                  path_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute full path integral for a batch of paths
        
        This is the main interface for computing path integrals in the quantum MCTS.
        
        Args:
            path_visits: Visit counts [batch_size, max_path_length]
            path_priors: Prior probabilities [batch_size, max_path_length]
            path_qvalues: Q-values [batch_size, max_path_length]
            path_mask: Valid elements mask [batch_size, max_path_length]
            
        Returns:
            Dictionary with:
                - 'actions': Path actions S[γ] [batch_size]
                - 'amplitudes': Unnormalized amplitudes [batch_size]
                - 'probabilities': Normalized probabilities [batch_size]
                - 'info_time': Information time values [batch_size]
                - 'hbar_eff': Effective Planck constants [batch_size]
        """
        batch_size = path_visits.shape[0]
        
        # Compute path actions
        actions = self.path_action(path_visits, path_priors, path_qvalues, path_mask)
        
        # Use mean visit count per path for ℏ_eff computation
        if path_mask is not None:
            mean_visits = (path_visits * path_mask.float()).sum(dim=1) / path_mask.sum(dim=1).float()
        else:
            mean_visits = path_visits.mean(dim=1)
        
        # Compute information time
        info_time = self.information_time(mean_visits)
        
        # Compute effective Planck constant
        hbar_eff = self.effective_planck_constant(mean_visits)
        
        # Compute transition amplitudes
        amplitudes = self.transition_amplitude(actions, mean_visits)
        
        # Normalize to probabilities
        probabilities = self.normalize_amplitudes(amplitudes)
        
        # Update statistics
        self.stats['total_paths_processed'] += batch_size
        
        return {
            'actions': actions,
            'amplitudes': amplitudes,
            'probabilities': probabilities,
            'info_time': info_time,
            'hbar_eff': hbar_eff
        }
    
    def compute_path_integral_action(self, 
                                   visit_counts: torch.Tensor,
                                   priors: Optional[torch.Tensor] = None,
                                   q_values: Optional[torch.Tensor] = None,
                                   states: Optional[torch.Tensor] = None,
                                   actions: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute path integral action for plotting and analysis
        
        This method provides a simplified interface for computing actions
        used by visualization and analysis modules. Supports two call signatures:
        1. Original: (visit_counts, priors, q_values)
        2. Statistical physics: (states, actions) or (visit_counts=states, priors=actions)
        
        Args:
            visit_counts: Visit counts OR states tensor [batch_size] or [batch_size, path_length]
            priors: Prior probabilities OR actions tensor [batch_size] or [batch_size, path_length]
            q_values: Q-values [batch_size] or [batch_size, path_length] (optional)
            states: State tensors (alternative argument)
            actions: Action indices (alternative argument)
            
        Returns:
            Action values [batch_size] or Dict with effective_action for stats physics
        """
        # Handle statistical physics call signature (states, actions)
        if (priors is not None and q_values is None and 
            isinstance(visit_counts, torch.Tensor) and isinstance(priors, torch.Tensor)):
            # This is likely the (states, actions) call signature
            states_tensor = visit_counts
            actions_tensor = priors
            
            # Call the states-specific method
            return self.compute_path_integral_action_states(states_tensor, actions_tensor)
        
        # Handle keyword arguments for statistical physics
        if states is not None and actions is not None:
            return self.compute_path_integral_action_states(states, actions)
        
        # Original signature handling
        if priors is None or q_values is None:
            raise ValueError("For standard call, priors and q_values must be provided")
            
        # Ensure tensors are 2D for batch processing
        if visit_counts.dim() == 1:
            visit_counts = visit_counts.unsqueeze(0)
        if priors.dim() == 1:
            priors = priors.unsqueeze(0)
        if q_values.dim() == 1:
            q_values = q_values.unsqueeze(0)
        
        # Compute Lagrangian density
        lagrangian = self.lagrangian_density(visit_counts, priors, q_values)
        
        # Sum over path elements to get action
        actions = lagrangian.sum(dim=1)
        
        # Return scalar if input was 1D
        if actions.shape[0] == 1:
            return actions.squeeze(0)
        
        return actions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.stats.copy()
    
    def compute_path_integral_action_states(self, states: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute path integral action for statistical physics interface
        
        This method provides compatibility with statistical physics visualization
        that expects states and actions as inputs.
        
        Args:
            states: State tensors [batch_size, state_dim] (not directly used)
            actions: Action indices [batch_size]
            
        Returns:
            Dictionary with:
                - 'effective_action': Action values [batch_size]
                - 'partition_function': Partition function estimate
                - 'free_energy': Free energy estimate
        """
        batch_size = actions.shape[0]
        device = actions.device
        
        # Generate mock MCTS-like data from action distribution
        # In a real implementation, this would use actual MCTS tree statistics
        
        # Create mock visit counts based on action frequency
        unique_actions, counts = torch.unique(actions, return_counts=True)
        visit_counts = torch.ones(batch_size, device=device)
        for i, action in enumerate(actions):
            mask = unique_actions == action
            if mask.any():
                visit_counts[i] = counts[mask].float()
        
        # Create uniform priors
        num_actions = int(actions.max().item()) + 1
        priors = torch.ones(batch_size, device=device) / num_actions
        
        # Generate Q-values with some structure
        q_values = torch.randn(batch_size, device=device) * 0.5
        # Add preference for frequently selected actions
        q_values += visit_counts.log() * 0.1
        
        # Compute temperature from config
        T = self.config.temperature if hasattr(self.config, 'temperature') else 1.0
        
        # Compute effective action using the standard method
        action_result = self.compute_path_integral_action(visit_counts, priors, q_values)
        
        # Compute partition function estimate
        Z = torch.exp(-action_result / T).mean()
        
        # Compute free energy
        F = -T * torch.log(Z + 1e-10)
        
        return {
            'effective_action': action_result,
            'partition_function': Z,
            'free_energy': F
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        for key in self.stats:
            if isinstance(self.stats[key], (int, float)):
                self.stats[key] = 0
    
    def clear_cache(self):
        """Clear internal caches"""
        self._cache.clear()


def create_path_integral_engine(config: Optional[PathIntegralConfig] = None) -> PathIntegralEngine:
    """Factory function to create a PathIntegralEngine with default configuration"""
    if config is None:
        config = PathIntegralConfig()
    
    return PathIntegralEngine(config)


# Export main classes and functions
__all__ = [
    'PathIntegralEngine',
    'PathIntegralConfig', 
    'InformationTimeMode',
    'create_path_integral_engine'
]