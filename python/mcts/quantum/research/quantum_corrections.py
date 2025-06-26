"""Quantum Corrections for Path Integral MCTS

This module implements one-loop quantum corrections based on the rigorous QFT formulation.
Key theoretical results from v4.0:

1. Diagonal Hessian: H_kk = 1/(N_k^pre + ε_N), H_kk' = 0 for k ≠ k' (trees only)
2. One-loop effective action: Γ_1loop = S_cl + (ℏ_eff/2) Σ_k log h_k
3. Gaussian approximation valid for N_k ≥ 5 (Stirling's approximation)
4. UV cutoff: N_UV = N_parent^α_UV with α_UV ∈ [0.3, 0.7]

The implementation directly measures these corrections from the actual classical MCTS
visit statistics, enabling numerical computation of quantum effects.
"""

import torch
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class QuantumCorrectionConfig:
    """Configuration for quantum corrections computation"""
    
    # Hessian computation
    use_diagonal_approximation: bool = True  # Valid for trees
    min_visit_count: int = 1  # Minimum N_k for Gaussian approximation
    epsilon_N: float = 1e-8   # Regularization parameter
    
    # UV cutoff parameters
    enable_uv_cutoff: bool = True
    alpha_uv: float = 0.5     # Cutoff exponent: N_UV = N_parent^α_UV
    min_uv_threshold: int = 10
    max_uv_threshold: int = 1000
    
    # One-loop parameters
    hbar_eff: float = 1.0     # Effective Planck constant
    include_stirling_correction: bool = True
    
    # Numerical stability
    min_hessian_element: float = 1e-12
    max_hessian_element: float = 1e6
    log_cutoff: float = 50.0   # Maximum argument for log/exp
    
    # Performance optimization
    use_vectorized_computation: bool = True
    batch_size: int = 3072
    cache_hessian_elements: bool = True
    
    # Device configuration
    device: str = 'cuda'


class QuantumCorrections:
    """Quantum corrections computation for path integral MCTS"""
    
    def __init__(self, config: QuantumCorrectionConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Cache for computed values
        self._hessian_cache: Dict[int, float] = {}
        self._stirling_cache: Dict[int, float] = {}
        
        # Performance statistics
        self.stats = {
            'hessian_computations': 0,
            'cache_hits': 0,
            'stirling_corrections': 0
        }
    
    def diagonal_hessian_element(self, visit_count: Union[int, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Compute diagonal Hessian element h_k = 1/(N_k^pre + ε_N)
        
        From Theorem 6.1: For tree structures, the action Hessian is diagonal
        with H_kk = 1/(N_k^pre + ε_N).
        
        Args:
            visit_count: Visit count N_k (scalar or tensor)
            
        Returns:
            Diagonal Hessian element(s)
        """
        if isinstance(visit_count, (int, float)):
            # Scalar computation with caching
            if self.config.cache_hessian_elements and visit_count in self._hessian_cache:
                self.stats['cache_hits'] += 1
                return self._hessian_cache[visit_count]
            
            h_k = 1.0 / (visit_count + self.config.epsilon_N)
            h_k = max(self.config.min_hessian_element, 
                     min(h_k, self.config.max_hessian_element))
            
            if self.config.cache_hessian_elements:
                self._hessian_cache[visit_count] = h_k
            
            self.stats['hessian_computations'] += 1
            return h_k
        else:
            # Tensor computation
            safe_visits = torch.clamp(visit_count, min=0)
            h_k = 1.0 / (safe_visits + self.config.epsilon_N)
            h_k = torch.clamp(h_k, 
                            min=self.config.min_hessian_element,
                            max=self.config.max_hessian_element)
            
            self.stats['hessian_computations'] += len(visit_count)
            return h_k
    
    def stirling_correction(self, visit_count: Union[int, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Compute Stirling's approximation correction for discrete sums
        
        From Lemma 6.1: For N_k ≥ 5, Stirling's approximation justifies treating
        discrete δu_k as continuous. For smaller N_k, we include corrections.
        
        Args:
            visit_count: Visit count N_k
            
        Returns:
            Stirling correction term
        """
        if not self.config.include_stirling_correction:
            return 0.0 if isinstance(visit_count, (int, float)) else torch.zeros_like(visit_count)
        
        if isinstance(visit_count, (int, float)):
            # Scalar computation with caching
            if visit_count in self._stirling_cache:
                return self._stirling_cache[visit_count]
            
            if visit_count >= self.config.min_visit_count:
                # Stirling correction: 1/(2N) - 1/(12N²)
                correction = 1.0 / (2 * visit_count) - 1.0 / (12 * visit_count**2)
            else:
                # Small N correction - use exact discrete sum difference
                correction = self._small_n_correction(visit_count)
            
            self._stirling_cache[visit_count] = correction
            self.stats['stirling_corrections'] += 1
            return correction
        else:
            # Tensor computation
            large_mask = visit_count >= self.config.min_visit_count
            correction = torch.zeros_like(visit_count, dtype=torch.float32)
            
            # Stirling correction for large N
            if large_mask.any():
                large_counts = visit_count[large_mask]
                stirling_term = (1.0 / (2 * large_counts) - 
                               1.0 / (12 * large_counts**2))
                correction[large_mask] = stirling_term
            
            # Small N correction
            small_mask = ~large_mask
            if small_mask.any():
                small_counts = visit_count[small_mask]
                small_corrections = self._small_n_correction_tensor(small_counts)
                correction[small_mask] = small_corrections
            
            self.stats['stirling_corrections'] += len(visit_count)
            return correction
    
    def _small_n_correction(self, n: int) -> float:
        """Exact correction for small visit counts"""
        if n <= 0:
            return 0.0
        
        # Exact discrete sum: Σ(k=1 to n) 1/k
        discrete_sum = sum(1.0/k for k in range(1, n+1))
        
        # Continuous approximation: log(n + ε_N)
        continuous_approx = math.log(n + self.config.epsilon_N)
        
        return discrete_sum - continuous_approx
    
    def _small_n_correction_tensor(self, visit_counts: torch.Tensor) -> torch.Tensor:
        """Tensor version of small N correction"""
        corrections = torch.zeros_like(visit_counts, dtype=torch.float32)
        
        for i, n in enumerate(visit_counts):
            n_val = int(n.item())
            if n_val > 0:
                discrete_sum = sum(1.0/k for k in range(1, n_val+1))
                continuous_approx = math.log(n_val + self.config.epsilon_N)
                corrections[i] = discrete_sum - continuous_approx
        
        return corrections
    
    def uv_cutoff_threshold(self, parent_visit_count: Union[int, torch.Tensor]) -> Union[int, torch.Tensor]:
        """Compute UV cutoff threshold N_UV = N_parent^α_UV
        
        From Definition 7.1: The UV cutoff balances exploration and exploitation.
        
        Args:
            parent_visit_count: Visit count of parent node
            
        Returns:
            UV cutoff threshold
        """
        if not self.config.enable_uv_cutoff:
            return float('inf') if isinstance(parent_visit_count, (int, float)) else torch.full_like(parent_visit_count, float('inf'))
        
        if isinstance(parent_visit_count, (int, float)):
            threshold = int(parent_visit_count ** self.config.alpha_uv)
            return max(self.config.min_uv_threshold, 
                      min(threshold, self.config.max_uv_threshold))
        else:
            threshold = parent_visit_count.float() ** self.config.alpha_uv
            threshold = torch.clamp(threshold.int(),
                                  min=self.config.min_uv_threshold,
                                  max=self.config.max_uv_threshold)
            return threshold
    
    def one_loop_effective_action(self, 
                                visit_counts: torch.Tensor,
                                classical_action: torch.Tensor) -> torch.Tensor:
        """Compute one-loop effective action Γ_1loop = S_cl + (ℏ_eff/2) Σ_k log h_k
        
        From Theorem 6.2: The one-loop correction includes the determinant of the
        diagonal Hessian matrix.
        
        Args:
            visit_counts: Visit counts for path elements [batch_size, path_length]
            classical_action: Classical action S_cl [batch_size]
            
        Returns:
            One-loop effective action [batch_size]
        """
        batch_size, path_length = visit_counts.shape
        
        # Compute diagonal Hessian elements
        hessian_elements = self.diagonal_hessian_element(visit_counts)  # [batch_size, path_length]
        
        # Add Stirling corrections if enabled
        if self.config.include_stirling_correction:
            stirling_corrections = self.stirling_correction(visit_counts)
            hessian_elements = hessian_elements + stirling_corrections
        
        # Compute log determinant: Σ_k log h_k
        # Numerical stability for log
        safe_hessian = torch.clamp(hessian_elements, min=self.config.min_hessian_element)
        log_hessian = torch.log(safe_hessian)
        log_hessian = torch.clamp(log_hessian, 
                                min=-self.config.log_cutoff,
                                max=self.config.log_cutoff)
        
        # Sum over path elements
        log_det_hessian = log_hessian.sum(dim=1)  # [batch_size]
        
        # One-loop correction: (ℏ_eff/2) * log_det
        quantum_correction = (self.config.hbar_eff / 2.0) * log_det_hessian
        
        # Combine with classical action
        one_loop_action = classical_action + quantum_correction
        
        return one_loop_action
    
    def path_quantum_weight(self,
                           visit_counts: torch.Tensor,
                           path_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute quantum weight factors for paths based on visit statistics
        
        This provides quantum corrections to path probabilities based on
        the one-loop effective action.
        
        Args:
            visit_counts: Visit counts [batch_size, path_length]
            path_mask: Valid path elements [batch_size, path_length]
            
        Returns:
            Quantum weight factors [batch_size]
        """
        # Apply mask if provided
        if path_mask is not None:
            masked_visits = torch.where(path_mask, visit_counts, 
                                      torch.ones_like(visit_counts))
        else:
            masked_visits = visit_counts
        
        # Compute Hessian elements
        hessian_elements = self.diagonal_hessian_element(masked_visits)
        
        # Apply mask again to zero out invalid elements
        if path_mask is not None:
            hessian_elements = hessian_elements * path_mask.float()
        
        # Compute quantum weight: exp(-(ℏ_eff/2) Σ log h_k)
        log_hessian = torch.log(torch.clamp(hessian_elements, min=self.config.min_hessian_element))
        if path_mask is not None:
            log_sum = (log_hessian * path_mask.float()).sum(dim=1)
        else:
            log_sum = log_hessian.sum(dim=1)
        
        quantum_exponent = -(self.config.hbar_eff / 2.0) * log_sum
        quantum_exponent = torch.clamp(quantum_exponent,
                                     min=-self.config.log_cutoff,
                                     max=self.config.log_cutoff)
        
        quantum_weights = torch.exp(quantum_exponent)
        
        return quantum_weights
    
    def validate_gaussian_approximation(self, visit_counts: torch.Tensor) -> Dict[str, Any]:
        """Validate the Gaussian approximation for given visit counts
        
        From Lemma 6.1: Gaussian approximation is valid for N_k ≥ 5.
        
        Args:
            visit_counts: Visit counts to validate
            
        Returns:
            Validation statistics
        """
        total_counts = visit_counts.numel()
        valid_gaussian = (visit_counts >= 5).sum().item()
        small_counts = (visit_counts < 5).sum().item()
        zero_counts = (visit_counts == 0).sum().item()
        
        gaussian_fraction = valid_gaussian / total_counts if total_counts > 0 else 0
        
        return {
            'total_counts': total_counts,
            'valid_gaussian': valid_gaussian,
            'small_counts': small_counts,
            'zero_counts': zero_counts,
            'gaussian_fraction': gaussian_fraction,
            'is_valid': gaussian_fraction > 0.8  # At least 80% should be valid
        }
    
    def compute_batch_corrections(self,
                                visit_counts_batch: torch.Tensor,
                                classical_actions: torch.Tensor,
                                path_masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute quantum corrections for a batch of paths
        
        Main interface for computing all quantum corrections efficiently.
        
        Args:
            visit_counts_batch: Visit counts [batch_size, path_length]
            classical_actions: Classical actions [batch_size]
            path_masks: Valid path elements [batch_size, path_length]
            
        Returns:
            Dictionary with correction results
        """
        # One-loop effective action
        one_loop_actions = self.one_loop_effective_action(visit_counts_batch, classical_actions)
        
        # Quantum weights
        quantum_weights = self.path_quantum_weight(visit_counts_batch, path_masks)
        
        # Hessian elements for analysis
        hessian_elements = self.diagonal_hessian_element(visit_counts_batch)
        
        # UV cutoff thresholds (using mean visit count as representative)
        if path_masks is not None:
            mean_visits = (visit_counts_batch.float() * path_masks.float()).sum(dim=1) / path_masks.sum(dim=1).float()
        else:
            mean_visits = visit_counts_batch.float().mean(dim=1)
        
        uv_thresholds = self.uv_cutoff_threshold(mean_visits)
        
        # Validation statistics
        validation = self.validate_gaussian_approximation(visit_counts_batch)
        
        return {
            'one_loop_actions': one_loop_actions,
            'quantum_weights': quantum_weights,
            'hessian_elements': hessian_elements,
            'uv_thresholds': uv_thresholds,
            'validation': validation
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get computation statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset computation statistics"""
        for key in self.stats:
            self.stats[key] = 0
    
    def clear_cache(self):
        """Clear computation caches"""
        self._hessian_cache.clear()
        self._stirling_cache.clear()


def create_quantum_corrections(config: Optional[QuantumCorrectionConfig] = None) -> QuantumCorrections:
    """Factory function to create QuantumCorrections with default configuration"""
    if config is None:
        config = QuantumCorrectionConfig()
    
    return QuantumCorrections(config)


# Export main classes and functions
__all__ = [
    'QuantumCorrections',
    'QuantumCorrectionConfig',
    'create_quantum_corrections'
]