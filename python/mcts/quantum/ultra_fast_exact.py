"""
Ultra-Fast Exact ℏ_eff for Production MCTS
==========================================

Implements exact Lindblad-derived ℏ_eff with aggressive optimizations:
- Precomputed lookup table for common visit counts
- Single ℏ_eff computation per PUCT call (not per action)
- JIT-compiled vectorized operations

Target: <1.05x overhead vs ultra-fast mode.
"""

import torch
import math
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class UltraFastExactHbar:
    """Ultra-optimized exact ℏ_eff with aggressive performance optimization"""
    
    def __init__(self, 
                 hbar_base: float = 1.0,
                 gamma_0: float = 0.1,
                 alpha: float = 1.0,
                 device: str = 'cuda',
                 max_precompute: int = 2000):
        """
        Initialize ultra-fast exact ℏ_eff calculator.
        
        Args:
            hbar_base: Base Planck constant 
            gamma_0: Base decay rate coefficient
            alpha: Decay rate exponent (1.0 for linear scaling)
            device: Computation device
            max_precompute: Maximum visits to precompute
        """
        self.hbar_base = hbar_base
        self.gamma_0 = gamma_0
        self.alpha = alpha
        self.device = torch.device(device)
        self.max_precompute = max_precompute
        
        # Numerical safety
        self.arccos_epsilon = 1e-12
        self.hbar_min = 0.001
        
        # Precompute exact table for ultra-fast lookup
        self._precompute_hbar_table()
        
        logger.debug(f"Ultra-fast exact ℏ_eff table precomputed for {max_precompute} visits")
    
    def _precompute_hbar_table(self):
        """Precompute exact ℏ_eff lookup table"""
        # Compute exact values for 0 to max_precompute
        visit_counts = torch.arange(0, self.max_precompute + 1, dtype=torch.float32)
        
        # Vectorized exact computation
        gamma_values = self.gamma_0 * torch.pow(1.0 + visit_counts, self.alpha)
        exp_half_gamma = torch.exp(-gamma_values / 2.0)
        exp_half_gamma = torch.clamp(exp_half_gamma, 
                                   min=self.arccos_epsilon, 
                                   max=1.0 - self.arccos_epsilon)
        
        arccos_values = torch.arccos(exp_half_gamma)
        hbar_values = self.hbar_base / arccos_values
        hbar_values = torch.clamp(hbar_values, min=self.hbar_min)
        
        # Store on device for ultra-fast access
        self.hbar_table = hbar_values.to(self.device)
    
    def compute_hbar_eff(self, total_visits: int) -> float:
        """
        Ultra-fast exact ℏ_eff computation with table lookup.
        
        Args:
            total_visits: Total parent visit count
            
        Returns:
            Exact ℏ_eff value
        """
        # Ultra-fast table lookup for common cases
        if 0 <= total_visits <= self.max_precompute:
            return self.hbar_table[total_visits].item()
        
        # Fallback computation for very large visit counts
        gamma_n = self.gamma_0 * (1.0 + total_visits) ** self.alpha
        exp_half = math.exp(-gamma_n / 2.0)
        exp_half = max(self.arccos_epsilon, min(1.0 - self.arccos_epsilon, exp_half))
        
        arccos_val = math.acos(exp_half)
        hbar_eff = self.hbar_base / arccos_val
        
        return max(hbar_eff, self.hbar_min)
    
    @staticmethod
    @torch.jit.script
    def compute_exact_puct_jit(q_values: torch.Tensor,
                             visit_counts: torch.Tensor,
                             priors: torch.Tensor,
                             total_visits: int,
                             c_puct: float,
                             hbar_eff: float) -> torch.Tensor:
        """
        JIT-compiled exact PUCT computation for maximum performance.
        
        Score(k) = c_puct * p_k * (N_k/N_tot) + Q_k + (4 * ℏ_eff) / (3 * N_k)
        """
        # Classical terms
        exploration = c_puct * priors * (visit_counts / max(total_visits, 1))
        exploitation = q_values
        
        # Exact quantum one-loop term
        quantum_factor = 4.0 / 3.0
        quantum_bonus = (quantum_factor * hbar_eff) / (visit_counts + 1e-8)
        
        return exploration + exploitation + quantum_bonus


def create_ultra_fast_exact_hbar(config: dict) -> UltraFastExactHbar:
    """Factory function for ultra-fast exact ℏ_eff calculator"""
    return UltraFastExactHbar(
        hbar_base=config.get('hbar_base', 1.0),
        gamma_0=config.get('gamma_0', 0.1),
        alpha=config.get('alpha', 1.0),
        device=config.get('device', 'cuda'),
        max_precompute=config.get('max_precompute', 2000)
    )