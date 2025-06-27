"""Renormalization Group Flow for Quantum MCTS

This module implements the complete RG flow equations from the v4.0 theoretical framework:

Discrete RG recursion relations:
λ' = λ - (ℏ_eff * b) / N_p
β' = β * (1 + b / (2 * N_p))
ℏ_eff' = ℏ_eff + (γ₀ * b) / (2 * N_p)

Beta functions in continuum limit:
β_λ = -ℏ_eff
β_β = β/2
β_ℏ = γ₀/2

This leads to the key prediction: c_PUCT ~ N^(-1/2)
"""

import torch
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


@dataclass
class RGFlowConfig:
    """Configuration for Renormalization Group flow"""
    
    # Initial parameters
    lambda_0: float = 1.4                # Initial prior weight
    beta_0: float = 1.0                  # Initial value weight
    hbar_eff_0: float = 1.0             # Initial effective Planck constant
    gamma_0: float = 0.01               # Bare decoherence strength
    
    # RG flow parameters
    blocking_factor: int = 5             # Number of edges to integrate out per step
    min_parent_visits: int = 10          # Minimum visits for RG transformation
    max_rg_steps: int = 20               # Maximum RG steps
    
    # Continuum limit parameters
    enable_continuum_beta: bool = True   # Use continuum beta functions
    ell_step: float = 0.1               # Step size in log(b) for continuum limit
    
    # Convergence criteria
    parameter_tolerance: float = 1e-6    # Convergence tolerance for parameters
    max_iterations: int = 1000           # Maximum iterations for fixed point search
    
    # Performance optimization
    cache_rg_flows: bool = True          # Cache RG flow computations
    use_adaptive_blocking: bool = True   # Adaptive blocking factor
    
    # Device configuration
    device: str = 'cuda'


class RGFlowEquations:
    """Implements RG flow equations and parameter evolution"""
    
    def __init__(self, config: RGFlowConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Current parameters
        self.lambda_current = config.lambda_0
        self.beta_current = config.beta_0
        self.hbar_eff_current = config.hbar_eff_0
        
        # RG flow history
        self.flow_history = {
            'lambda': [config.lambda_0],
            'beta': [config.beta_0], 
            'hbar_eff': [config.hbar_eff_0],
            'ell': [0.0],
            'N_p': [0],
            'b': [0]
        }
        
        # Cache for RG transformations
        self._rg_cache: Dict[str, Dict[str, float]] = {}
        
        # Statistics
        self.stats = {
            'rg_steps_performed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_computation_time': 0.0
        }
    
    def perform_rg_step(self, 
                       N_parent: int,
                       b: Optional[int] = None) -> Dict[str, float]:
        """Perform one RG step: integrate out b low-visit edges
        
        Args:
            N_parent: Visit count of parent node
            b: Number of edges to integrate out (uses config default if None)
            
        Returns:
            Dictionary of new parameters after RG step
        """
        import time
        start_time = time.time()
        
        if b is None:
            b = self.config.blocking_factor
        
        # Adaptive blocking if enabled
        if self.config.use_adaptive_blocking:
            b = self._compute_adaptive_blocking_factor(N_parent, b)
        
        # Check cache
        cache_key = f"{N_parent}_{b}_{self.lambda_current:.6f}_{self.beta_current:.6f}_{self.hbar_eff_current:.6f}"
        
        if self.config.cache_rg_flows and cache_key in self._rg_cache:
            self.stats['cache_hits'] += 1
            return self._rg_cache[cache_key]
        
        self.stats['cache_misses'] += 1
        
        # Apply RG recursion relations
        lambda_new = self._evolve_lambda(N_parent, b)
        beta_new = self._evolve_beta(N_parent, b)
        hbar_eff_new = self._evolve_hbar_eff(N_parent, b)
        
        new_params = {
            'lambda': lambda_new,
            'beta': beta_new,
            'hbar_eff': hbar_eff_new
        }
        
        # Update current parameters
        self.lambda_current = lambda_new
        self.beta_current = beta_new
        self.hbar_eff_current = hbar_eff_new
        
        # Update history
        ell_new = self.flow_history['ell'][-1] + math.log(b)
        self.flow_history['lambda'].append(lambda_new)
        self.flow_history['beta'].append(beta_new)
        self.flow_history['hbar_eff'].append(hbar_eff_new)
        self.flow_history['ell'].append(ell_new)
        self.flow_history['N_p'].append(N_parent)
        self.flow_history['b'].append(b)
        
        # Cache result
        if self.config.cache_rg_flows:
            self._rg_cache[cache_key] = new_params
        
        # Update statistics
        computation_time = time.time() - start_time
        self.stats['rg_steps_performed'] += 1
        self.stats['average_computation_time'] = (
            (self.stats['average_computation_time'] * (self.stats['rg_steps_performed'] - 1) + 
             computation_time) / self.stats['rg_steps_performed']
        )
        
        return new_params
    
    def _evolve_lambda(self, N_parent: int, b: int) -> float:
        """Apply RG evolution to λ: λ' = λ - (ℏ_eff * b) / N_p"""
        if N_parent == 0:
            return self.lambda_current
        
        delta_lambda = -(self.hbar_eff_current * b) / N_parent
        return self.lambda_current + delta_lambda
    
    def _evolve_beta(self, N_parent: int, b: int) -> float:
        """Apply RG evolution to β: β' = β * (1 + b / (2 * N_p))"""
        if N_parent == 0:
            return self.beta_current
        
        factor = 1 + b / (2 * N_parent)
        return self.beta_current * factor
    
    def _evolve_hbar_eff(self, N_parent: int, b: int) -> float:
        """Apply RG evolution to ℏ_eff: ℏ_eff' = ℏ_eff + (γ₀ * b) / (2 * N_p)"""
        if N_parent == 0:
            return self.hbar_eff_current
        
        delta_hbar = (self.config.gamma_0 * b) / (2 * N_parent)
        return self.hbar_eff_current + delta_hbar
    
    def _compute_adaptive_blocking_factor(self, N_parent: int, default_b: int) -> int:
        """Compute adaptive blocking factor based on parent visits"""
        if N_parent < self.config.min_parent_visits:
            return max(1, default_b // 2)  # Smaller blocking for low visits
        elif N_parent > 1000:
            return min(default_b * 2, 20)  # Larger blocking for high visits
        else:
            return default_b
    
    def compute_beta_functions(self) -> Dict[str, float]:
        """Compute beta functions in continuum limit
        
        Returns:
            Dictionary containing β_λ, β_β, β_ℏ
        """
        # Theoretical beta functions
        beta_lambda = -self.hbar_eff_current
        beta_beta = self.beta_current / 2
        beta_hbar = self.config.gamma_0 / 2
        
        return {
            'beta_lambda': beta_lambda,
            'beta_beta': beta_beta,
            'beta_hbar': beta_hbar
        }
    
    def evolve_to_fixed_point(self, 
                             target_N: int,
                             start_N: int = 10) -> Dict[str, Any]:
        """Evolve parameters using RG flow to target visit count
        
        Args:
            target_N: Target total visit count
            start_N: Starting visit count
            
        Returns:
            Final parameters and evolution data
        """
        if target_N <= start_N:
            return {
                'final_params': self.get_current_parameters(),
                'evolution_data': self.flow_history,
                'converged': True
            }
        
        # Reset to initial conditions
        self.reset_to_initial_conditions()
        
        # Evolve in steps
        current_N = start_N
        step = 0
        
        while current_N < target_N and step < self.config.max_rg_steps:
            # Determine step size
            remaining = target_N - current_N
            b = min(self.config.blocking_factor, max(1, remaining // 10))
            
            # Take RG step
            self.perform_rg_step(current_N, b)
            
            # Update current N (this is approximate - in reality it's more complex)
            current_N += b * 10  # Heuristic scaling
            step += 1
        
        converged = current_N >= target_N
        
        return {
            'final_params': self.get_current_parameters(),
            'evolution_data': self.flow_history,
            'converged': converged,
            'final_N': current_N,
            'steps_taken': step
        }
    
    def compute_c_puct_evolution(self) -> torch.Tensor:
        """Compute c_PUCT evolution: c_PUCT = λ/β
        
        Returns:
            Tensor of c_PUCT values along RG flow
        """
        lambda_history = torch.tensor(self.flow_history['lambda'], device=self.device)
        beta_history = torch.tensor(self.flow_history['beta'], device=self.device)
        
        # Avoid division by zero
        safe_beta = torch.clamp(beta_history, min=1e-12)
        c_puct_history = lambda_history / safe_beta
        
        return c_puct_history
    
    def validate_power_law_decay(self) -> Dict[str, Any]:
        """Validate c_PUCT ~ N^(-1/2) decay prediction
        
        Returns:
            Validation results including fitted exponent
        """
        if len(self.flow_history['N_p']) < 5:
            return {'insufficient_data': True}
        
        # Get N values and c_PUCT values
        N_values = torch.tensor([N for N in self.flow_history['N_p'] if N > 0], 
                               dtype=torch.float32, device=self.device)
        c_puct_values = self.compute_c_puct_evolution()
        
        # Only use data points with valid N
        if len(N_values) != len(c_puct_values):
            valid_indices = [i for i, N in enumerate(self.flow_history['N_p']) if N > 0]
            c_puct_values = c_puct_values[valid_indices]
        
        if len(N_values) < 3:
            return {'insufficient_valid_data': True}
        
        # Fit power law: c = c₀ * N^α
        log_N = torch.log(N_values)
        log_c = torch.log(torch.clamp(c_puct_values, min=1e-12))
        
        # Linear regression in log space
        # log(c) = log(c₀) + α * log(N)
        X = torch.stack([torch.ones_like(log_N), log_N], dim=1)
        y = log_c
        
        # Solve: X @ [log(c₀), α] = y
        try:
            coeffs = torch.linalg.lstsq(X, y).solution
            log_c0, alpha = coeffs[0].item(), coeffs[1].item()
            
            # Theoretical prediction: α = -1/2
            theoretical_alpha = -0.5
            relative_error = abs(alpha - theoretical_alpha) / abs(theoretical_alpha)
            
            return {
                'fitted_alpha': alpha,
                'theoretical_alpha': theoretical_alpha,
                'relative_error': relative_error,
                'log_c0': log_c0,
                'validation_passed': relative_error < 0.2,
                'num_data_points': len(N_values)
            }
            
        except Exception as e:
            logger.warning(f"Power law fitting failed: {e}")
            return {'fitting_failed': True, 'error': str(e)}
    
    def get_current_parameters(self) -> Dict[str, float]:
        """Get current RG flow parameters"""
        return {
            'lambda': self.lambda_current,
            'beta': self.beta_current,
            'hbar_eff': self.hbar_eff_current
        }
    
    def reset_to_initial_conditions(self):
        """Reset RG flow to initial conditions"""
        self.lambda_current = self.config.lambda_0
        self.beta_current = self.config.beta_0
        self.hbar_eff_current = self.config.hbar_eff_0
        
        # Reset history
        self.flow_history = {
            'lambda': [self.config.lambda_0],
            'beta': [self.config.beta_0],
            'hbar_eff': [self.config.hbar_eff_0],
            'ell': [0.0],
            'N_p': [0],
            'b': [0]
        }
    
    def get_rg_flow_data(self) -> Dict[str, torch.Tensor]:
        """Get complete RG flow data as tensors"""
        return {
            'lambda': torch.tensor(self.flow_history['lambda'], device=self.device),
            'beta': torch.tensor(self.flow_history['beta'], device=self.device),
            'hbar_eff': torch.tensor(self.flow_history['hbar_eff'], device=self.device),
            'ell': torch.tensor(self.flow_history['ell'], device=self.device),
            'N_p': torch.tensor(self.flow_history['N_p'], device=self.device),
            'c_puct': self.compute_c_puct_evolution()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RG flow computation statistics"""
        stats = self.stats.copy()
        stats['cache_hit_rate'] = (
            self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
            if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0.0
        )
        stats['cache_size'] = len(self._rg_cache)
        stats['flow_history_length'] = len(self.flow_history['lambda'])
        return stats


def create_rg_flow_equations(config: Optional[RGFlowConfig] = None) -> RGFlowEquations:
    """Factory function to create RGFlowEquations with default configuration"""
    if config is None:
        config = RGFlowConfig()
    
    return RGFlowEquations(config)


# Export main classes and functions
__all__ = [
    'RGFlowEquations',
    'RGFlowConfig',
    'create_rg_flow_equations'
]