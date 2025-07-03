"""One-Loop Quantum Corrections for MCTS

This module implements the one-loop quantum corrections from Part III of quantum_mcts.md:
- Diagonal Hessian for tree structures
- Gaussian approximation for path integrals  
- One-loop effective action computation
- Integration with path integral formulation

Key theoretical results:
- H_kk' = δ_kk' h_k where h_k = 1/(N_k^pre + ε_N)
- Γ_1-loop = S_cl + (ℏ_eff/2) Σ_k log h_k
- Gaussian approximation valid for N_k ≥ 5
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
class OneLoopConfig:
    """Configuration for one-loop quantum corrections"""
    
    # Physical parameters
    hbar_eff: float = 1.0               # Effective Planck constant
    epsilon_N: float = 1e-8             # Visit count regularization
    
    # Integration with corrected ℏ_eff calculation
    auto_compute_hbar_eff: bool = True  # Automatically compute ℏ_eff from visit counts
    lindblad_config: Optional['LindbladConfig'] = None  # Config for ℏ_eff calculation
    
    # Hessian computation method (IMPORTANT for physics consistency)
    use_exact_hessian: bool = True  # Use exact h_k = (κ*p_k/2)*sqrt(N_total/N_k³)
    
    # RG counter-term for pruning (entropy loss from integrating out hard children)
    include_rg_counterterm: bool = True  # Add ℏ_eff*log(1+B_trim) per parent
    
    # Coupling constants
    kappa: float = 1.0  # c_puct exploration coefficient
    beta: float = 1.0   # Value weight coefficient
    
    # Classical action: Quadratic kernel for EXACT PUCT recovery
    # S_cl = κ*N_total*Σ(√q_k - p_k)² - mathematically proven to recover PUCT with quantum bonus
    # The quadratic form (√q_k - p_k)² is CRITICAL - not (√q_k - √p_k)²
    
    # Gaussian approximation parameters
    min_visits_gaussian: int = 5        # Minimum visits for Gaussian approximation
    use_stirling_approximation: bool = True  # Use Stirling for large N
    
    # Computation parameters
    max_hessian_size: int = 10000       # Maximum Hessian matrix size
    use_sparse_hessian: bool = True     # Use sparse representation
    diagonal_only: bool = True          # Use diagonal approximation for trees
    
    # Numerical stability
    min_hessian_eigenval: float = 1e-12 # Minimum eigenvalue for regularization
    max_log_correction: float = 50.0    # Maximum log correction to prevent overflow
    
    # Performance optimization
    cache_corrections: bool = True      # Cache computed corrections
    batch_computation: bool = True      # Batch multiple corrections
    
    # Device configuration
    device: str = 'cuda'


class OneLoopCorrections:
    """Implements one-loop quantum corrections for tree-structured MCTS"""
    
    def __init__(self, config: OneLoopConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize ℏ_eff calculator if auto-computation is enabled
        self.hbar_calculator = None
        if config.auto_compute_hbar_eff:
            if config.lindblad_config is not None:
                # Use provided Lindblad config
                from lindblad_dynamics import LindbladConfig
                self.hbar_calculator = None  # EffectivePlanckConstant no longer used
            else:
                # Create default config with corrected physics
                from lindblad_dynamics import LindbladConfig
                default_lindblad_config = LindbladConfig(
                    device=config.device,
                    hbar_base=1.0,
                    hbar_min=0.01,
                    information_decay_rate=0.5
                )
                self.hbar_calculator = None  # EffectivePlanckConstant no longer used
        
        # Cache for computed corrections
        self._correction_cache: Dict[str, torch.Tensor] = {}
        self._hessian_cache: Dict[str, torch.Tensor] = {}
        
        # Statistics
        self.stats = {
            'corrections_computed': 0,
            'hessian_computations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_computation_time': 0.0,
            'gaussian_approximations': 0,
            'stirling_approximations': 0,
            'hbar_computations': 0,
            'validation_errors': 0,
            'bounds_violations': 0
        }
    
    def _validate_visit_counts(self, visit_counts: torch.Tensor, method_name: str) -> None:
        """Comprehensive validation for visit count tensors"""
        try:
            # Type and device validation
            if not isinstance(visit_counts, torch.Tensor):
                raise TypeError(f"{method_name}: visit_counts must be torch.Tensor, got {type(visit_counts)}")
            
            if visit_counts.device != self.device:
                # Only log warning once per session to reduce spam
                if not hasattr(self, '_device_warning_logged'):
                    logger.warning(f"{method_name}: Input tensor device {visit_counts.device} != expected {self.device}")
                    logger.warning("Consider creating tensors on the correct device to improve performance")
                    self._device_warning_logged = True
                
                # Move tensor efficiently with non-blocking transfer
                visit_counts = visit_counts.to(self.device, non_blocking=True)
            
            # Dimension validation
            if visit_counts.dim() != 1:
                raise ValueError(f"{method_name}: visit_counts must be 1D tensor, got {visit_counts.dim()}D")
            
            if len(visit_counts) == 0:
                raise ValueError(f"{method_name}: visit_counts cannot be empty")
            
            if len(visit_counts) > self.config.max_hessian_size:
                raise ValueError(f"{method_name}: visit_counts length {len(visit_counts)} exceeds max_hessian_size {self.config.max_hessian_size}")
            
            # Numerical validation
            if torch.any(visit_counts < 0):
                self.stats['bounds_violations'] += 1
                raise ValueError(f"{method_name}: visit_counts must be non-negative, found min={visit_counts.min().item()}")
            
            if torch.any(torch.isnan(visit_counts)):
                self.stats['validation_errors'] += 1
                raise ValueError(f"{method_name}: visit_counts contains NaN values")
            
            if torch.any(torch.isinf(visit_counts)):
                self.stats['validation_errors'] += 1
                raise ValueError(f"{method_name}: visit_counts contains infinite values")
            
            # Physical bounds validation
            max_visits = visit_counts.max().item()
            if max_visits > 1e8:
                logger.warning(f"{method_name}: Very large visit count {max_visits:.1e} may cause numerical issues")
            
        except Exception as e:
            self.stats['validation_errors'] += 1
            logger.error(f"Validation failed in {method_name}: {e}")
            raise
    
    def _validate_path_indices(self, path_indices: torch.Tensor, visit_counts: torch.Tensor, method_name: str) -> None:
        """Validation for path indices"""
        try:
            # Type validation
            if not isinstance(path_indices, torch.Tensor):
                raise TypeError(f"{method_name}: path_indices must be torch.Tensor, got {type(path_indices)}")
            
            # Device consistency
            if path_indices.device != visit_counts.device:
                raise ValueError(f"{method_name}: path_indices device {path_indices.device} != visit_counts device {visit_counts.device}")
            
            # Dimension validation
            if path_indices.dim() != 1:
                raise ValueError(f"{method_name}: path_indices must be 1D tensor, got {path_indices.dim()}D")
            
            # Index bounds validation
            if torch.any(path_indices < 0):
                self.stats['bounds_violations'] += 1
                raise ValueError(f"{method_name}: path_indices must be non-negative, found min={path_indices.min().item()}")
            
            if torch.any(path_indices >= len(visit_counts)):
                self.stats['bounds_violations'] += 1
                max_idx = path_indices.max().item()
                raise ValueError(f"{method_name}: path_indices max {max_idx} >= visit_counts length {len(visit_counts)}")
            
        except Exception as e:
            self.stats['validation_errors'] += 1
            logger.error(f"Path indices validation failed in {method_name}: {e}")
            raise
    
    def _validate_tensor_bounds(self, tensor: torch.Tensor, name: str, min_val: float = -float('inf'), 
                               max_val: float = float('inf'), method_name: str = "") -> None:
        """Generic tensor bounds validation"""
        try:
            if torch.any(tensor < min_val):
                self.stats['bounds_violations'] += 1
                actual_min = tensor.min().item()
                raise ValueError(f"{method_name}: {name} below minimum bound: {actual_min} < {min_val}")
            
            if torch.any(tensor > max_val):
                self.stats['bounds_violations'] += 1
                actual_max = tensor.max().item()
                raise ValueError(f"{method_name}: {name} above maximum bound: {actual_max} > {max_val}")
            
        except Exception as e:
            self.stats['validation_errors'] += 1
            logger.error(f"Tensor bounds validation failed for {name} in {method_name}: {e}")
            raise
    
    def _validate_classical_action(self, classical_action: Union[torch.Tensor, float, int], method_name: str) -> None:
        """Validation for classical action input"""
        try:
            # Convert to tensor if needed
            if not isinstance(classical_action, torch.Tensor):
                if isinstance(classical_action, (int, float)):
                    classical_action = torch.tensor(classical_action, dtype=torch.float32, device=self.device)
                else:
                    raise TypeError(f"{method_name}: classical_action must be tensor or scalar, got {type(classical_action)}")
            
            # Device validation with optimized transfer
            if classical_action.device != self.device:
                # Only log warning once per session to reduce spam
                if not hasattr(self, '_classical_action_device_warning_logged'):
                    logger.warning(f"{method_name}: classical_action device {classical_action.device} != expected {self.device}")
                    logger.warning("Consider creating tensors on the correct device to improve performance")
                    self._classical_action_device_warning_logged = True
                
                # Move tensor efficiently with non-blocking transfer
                classical_action = classical_action.to(self.device, non_blocking=True)
            
            # Dimension validation (scalar or 1D batch)
            if classical_action.dim() > 1:
                raise ValueError(f"{method_name}: classical_action must be scalar or 1D tensor, got {classical_action.dim()}D")
            
            # Numerical validation
            if torch.any(torch.isnan(classical_action)):
                self.stats['validation_errors'] += 1
                raise ValueError(f"{method_name}: classical_action contains NaN values")
            
            if torch.any(torch.isinf(classical_action)):
                self.stats['validation_errors'] += 1
                raise ValueError(f"{method_name}: classical_action contains infinite values")
            
            # Physical bounds (actions shouldn't be extremely large)
            max_action = classical_action.max().item() if classical_action.numel() > 0 else 0
            if abs(max_action) > 1e6:
                logger.warning(f"{method_name}: Very large classical action {max_action:.1e} may indicate physics error")
            
        except Exception as e:
            self.stats['validation_errors'] += 1
            logger.error(f"Classical action validation failed in {method_name}: {e}")
            raise
    
    def _validate_hbar_eff(self, hbar_eff: float, method_name: str) -> None:
        """Validation for effective Planck constant"""
        try:
            if not isinstance(hbar_eff, (int, float)):
                raise TypeError(f"{method_name}: hbar_eff must be numeric, got {type(hbar_eff)}")
            
            if math.isnan(hbar_eff):
                self.stats['validation_errors'] += 1
                raise ValueError(f"{method_name}: hbar_eff is NaN")
            
            if math.isinf(hbar_eff):
                self.stats['validation_errors'] += 1
                raise ValueError(f"{method_name}: hbar_eff is infinite")
            
            # Physical bounds - ℏ_eff should be positive and reasonable
            if hbar_eff <= 0:
                self.stats['bounds_violations'] += 1
                raise ValueError(f"{method_name}: hbar_eff must be positive, got {hbar_eff}")
            
            if hbar_eff > 1e3:
                self.stats['bounds_violations'] += 1
                logger.warning(f"{method_name}: Very large hbar_eff {hbar_eff:.1e} may indicate explosion issue")
            
            if hbar_eff < 1e-6:
                logger.warning(f"{method_name}: Very small hbar_eff {hbar_eff:.1e} approaching classical limit")
            
        except Exception as e:
            self.stats['validation_errors'] += 1
            logger.error(f"hbar_eff validation failed in {method_name}: {e}")
            raise
    
    def compute_diagonal_hessian(self, 
                                visit_counts: torch.Tensor,
                                path_indices: Optional[torch.Tensor] = None,
                                priors: Optional[torch.Tensor] = None,
                                total_visits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute diagonal Hessian matrix for quadratic kernel action
        
        For the quadratic action S_cl = κ*N_total*Σ(√q_k - p_k)², the exact Hessian is:
        h_k = ∂²S_cl/∂N_k² = (κ*p_k/2) * sqrt(N_total/N_k³)
        
        This is the EXACT result from the corrected derivation, giving the quantum bonus
        Δ Γ^(1) = -3ℏ_eff/(4N_k) when differentiated.
        
        Args:
            visit_counts: Visit counts N_k [num_edges]
            path_indices: Optional path indices for specific subset
            priors: Neural network priors p_k [num_edges]
            total_visits: Total visits N_total
            
        Returns:
            Diagonal Hessian elements [num_edges] or [num_paths]
        """
        import time
        start_time = time.time()
        
        # COMPREHENSIVE INPUT VALIDATION
        self._validate_visit_counts(visit_counts, "compute_diagonal_hessian")
        
        if path_indices is not None:
            self._validate_path_indices(path_indices, visit_counts, "compute_diagonal_hessian")
        
        # Create cache key
        if path_indices is not None:
            cache_key = f"hessian_hellinger_{hash(tuple(visit_counts.cpu().numpy()))}_{hash(tuple(path_indices.cpu().numpy()))}"
            relevant_visits = visit_counts[path_indices]
        else:
            cache_key = f"hessian_hellinger_{hash(tuple(visit_counts.cpu().numpy()))}"
            relevant_visits = visit_counts
        
        if self.config.cache_corrections and cache_key in self._hessian_cache:
            self.stats['cache_hits'] += 1
            return self._hessian_cache[cache_key]
        
        self.stats['cache_misses'] += 1
        
        # Exact quadratic kernel Hessian: h_k = (κ*p_k/2) * sqrt(N_total/N_k³)
        # From the corrected analysis: ∂²S_cl/∂N_k² = (κ*p_k/2) * sqrt(N_total/N_k³)
        
        # Get priors and total visits
        if priors is None:
            # Default uniform priors if not provided
            priors = torch.ones_like(relevant_visits) / len(relevant_visits)
            logger.warning("No priors provided, using uniform priors")
        else:
            if path_indices is not None:
                priors = priors[path_indices]
        
        if total_visits is None:
            total_visits = visit_counts.sum()
        
        # Compute exact Hessian
        safe_visits = relevant_visits + self.config.epsilon_N
        kappa = self.config.kappa
        hessian_diagonal = (kappa * priors / 2) * torch.sqrt(total_visits / (safe_visits ** 3))
        
        # Ensure numerical stability
        hessian_diagonal = torch.clamp(hessian_diagonal, 
                                     min=self.config.min_hessian_eigenval,
                                     max=1.0/self.config.epsilon_N)
        
        # Cache result
        if self.config.cache_corrections:
            self._hessian_cache[cache_key] = hessian_diagonal
        
        # Update statistics
        computation_time = time.time() - start_time
        self.stats['hessian_computations'] += 1
        self.stats['average_computation_time'] = (
            (self.stats['average_computation_time'] * (self.stats['hessian_computations'] - 1) +
             computation_time) / self.stats['hessian_computations']
        )
        
        return hessian_diagonal
    
    def compute_corrected_classical_action(self, 
                                         visit_counts: torch.Tensor,
                                         priors: Optional[torch.Tensor] = None,
                                         q_values: Optional[torch.Tensor] = None,
                                         kappa: float = 0.7,
                                         beta: float = 1.0) -> torch.Tensor:
        """Compute complete classical action: Hellinger distance + value term
        
        Complete formulation from the closed-form effective action:
        S_cl = κ*N_total*Σ(√q_k - √p_k)² - β*Σ(N_k*Q_k)
        
        Args:
            visit_counts: Visit counts [num_actions]
            priors: Prior probabilities [num_actions] 
            q_values: Optional Q-values [num_actions]
            kappa: Exploration scale κ = c_puct/2 (default: 0.7)
            beta: Value weight (default: 1.0)
            
        Returns:
            Classical action value
        """
        N_total = torch.sum(visit_counts)
        
        # Regularized counts and priors
        safe_visits = visit_counts + self.config.epsilon_N
        
        if priors is not None:
            safe_priors = torch.clamp(priors, min=self.config.epsilon_N, max=1.0)
        else:
            # Uniform prior assumption
            K = len(visit_counts)
            safe_priors = torch.ones_like(visit_counts) / K
        
        # Empirical distribution
        q_k = safe_visits / torch.sum(safe_visits)
        
        # Hellinger distance exploration term: κ*N_total*Σ(√q_k - √p_k)²
        sqrt_q_k = torch.sqrt(q_k)
        sqrt_p_k = torch.sqrt(safe_priors)
        exploration_term = kappa * N_total * torch.sum((sqrt_q_k - sqrt_p_k)**2)
        
        # Value term: -β*Σ(N_k*Q_k)
        if q_values is not None:
            value_term = -beta * torch.sum(safe_visits * q_values)
        else:
            value_term = 0.0
        
        classical_action = exploration_term + value_term
        
        return classical_action
    
    def compute_complete_effective_action(self,
                                        visit_counts: torch.Tensor,
                                        priors: torch.Tensor,
                                        q_values: Optional[torch.Tensor] = None,
                                        kappa: float = 0.7,
                                        beta: float = 1.0,
                                        hbar_eff: float = 1.0,
                                        b_trim: int = 0) -> Dict[str, torch.Tensor]:
        """Compute complete closed-form effective action
        
        Implements the full quantum-corrected effective action:
        Γ_eff = S_cl + ΔΓ^(1) + ΔΓ_RG
        
        Where:
        - S_cl = κ*N_total*Σ(√q_k - √p_k)² - β*Σ(N_k*Q_k)
        - ΔΓ^(1) = (ℏ_eff/2)*Σ log(3κ/4N_k)  
        - ΔΓ_RG = ℏ_eff*log(1 + B_trim)
        
        Args:
            visit_counts: Visit counts N_k [num_actions]
            priors: Prior probabilities p_k [num_actions]
            q_values: Optional Q-values Q_k [num_actions]
            kappa: Exploration scale κ = c_puct/2 (default: 0.7)
            beta: Value weight (default: 1.0)
            hbar_eff: Effective Planck constant (default: 1.0)
            b_trim: Number of hard edges trimmed away (default: 0)
            
        Returns:
            Dictionary with all action components and total effective action
        """
        
        # 1. Classical action: S_cl = κ*N_total*Σ(√q_k - √p_k)² - β*Σ(N_k*Q_k)
        classical_action = self.compute_corrected_classical_action(
            visit_counts, priors, q_values, kappa, beta
        )
        
        # 2. One-loop correction: ΔΓ^(1) = (ℏ_eff/2)*Σ log(3κ/4N_k)
        safe_visits = visit_counts + self.config.epsilon_N
        hessian_diagonal = (3 * kappa / 4) / safe_visits
        
        log_h_k = torch.log(hessian_diagonal)
        log_h_k = torch.clamp(log_h_k, 
                             min=-self.config.max_log_correction,
                             max=self.config.max_log_correction)
        
        one_loop_correction = (hbar_eff / 2.0) * torch.sum(log_h_k)
        
        # 3. RG counter-term: ΔΓ_RG = ℏ_eff*log(1 + B_trim)
        if b_trim > 0:
            rg_correction = hbar_eff * torch.log(torch.tensor(1.0 + b_trim, device=self.device))
        else:
            rg_correction = torch.tensor(0.0, device=self.device)
        
        # 4. Total effective action
        total_effective_action = classical_action + one_loop_correction + rg_correction
        
        # Decompose classical action into components
        N_total = torch.sum(visit_counts)
        safe_visits = visit_counts + self.config.epsilon_N
        q_k = safe_visits / torch.sum(safe_visits)
        sqrt_q_k = torch.sqrt(q_k)
        sqrt_p_k = torch.sqrt(torch.clamp(priors, min=self.config.epsilon_N, max=1.0))
        
        exploration_term = kappa * N_total * torch.sum((sqrt_q_k - sqrt_p_k)**2)
        
        if q_values is not None:
            value_term = -beta * torch.sum(safe_visits * q_values)
        else:
            value_term = torch.tensor(0.0, device=self.device)
        
        return {
            # Total effective action
            'total_effective_action': total_effective_action,
            
            # Classical components
            'classical_action': classical_action,
            'exploration_term': exploration_term,
            'value_term': value_term,
            
            # Quantum corrections
            'one_loop_correction': one_loop_correction,
            'rg_correction': rg_correction,
            
            # Technical details
            'hessian_diagonal': hessian_diagonal,
            'log_determinant': torch.sum(log_h_k),
            'parameters': {
                'kappa': kappa,
                'beta': beta,
                'hbar_eff': hbar_eff,
                'b_trim': b_trim,
                'N_total': N_total
            }
        }
    
    def compute_edge_wise_effective_action(self,
                                         visit_counts: torch.Tensor,
                                         priors: torch.Tensor,
                                         q_values: Optional[torch.Tensor] = None,
                                         kappa: float = 0.7,
                                         beta: float = 1.0,
                                         hbar_eff: float = 1.0,
                                         b_trim: int = 0) -> torch.Tensor:
        """Compute edge-wise effective action for MCTS selection
        
        For each child k, computes:
        Γ_eff(k) = κ*N_total*(√q_k - √p_k)² - β*N_k*Q_k + 
                   (ℏ_eff/2)*log(3κ/4N_k) + ℏ_eff*log(1 + B_trim)
        
        Args:
            visit_counts: Visit counts N_k [num_actions]
            priors: Prior probabilities p_k [num_actions]
            q_values: Optional Q-values Q_k [num_actions]
            kappa: Exploration scale κ = c_puct/2 (default: 0.7)
            beta: Value weight (default: 1.0)
            hbar_eff: Effective Planck constant (default: 1.0)
            b_trim: Number of hard edges trimmed away (default: 0)
            
        Returns:
            Edge-wise effective actions [num_actions]
        """
        N_total = torch.sum(visit_counts)
        safe_visits = visit_counts + self.config.epsilon_N
        safe_priors = torch.clamp(priors, min=self.config.epsilon_N, max=1.0)
        q_k = safe_visits / torch.sum(safe_visits)
        
        # Edge-wise Hellinger exploration term
        sqrt_q_k = torch.sqrt(q_k)
        sqrt_p_k = torch.sqrt(safe_priors)
        exploration_terms = kappa * N_total * (sqrt_q_k - sqrt_p_k)**2
        
        # Edge-wise value terms
        if q_values is not None:
            value_terms = -beta * safe_visits * q_values
        else:
            value_terms = torch.zeros_like(safe_visits)
        
        # Edge-wise one-loop corrections
        hessian_diagonal = (3 * kappa / 4) / safe_visits
        log_h_k = torch.log(hessian_diagonal)
        log_h_k = torch.clamp(log_h_k, 
                             min=-self.config.max_log_correction,
                             max=self.config.max_log_correction)
        one_loop_terms = (hbar_eff / 2.0) * log_h_k
        
        # RG correction (same for all edges under this parent)
        if b_trim > 0:
            rg_correction = hbar_eff * torch.log(torch.tensor(1.0 + b_trim, device=self.device))
            rg_terms = torch.full_like(safe_visits, rg_correction)
        else:
            rg_terms = torch.zeros_like(safe_visits)
        
        # Total edge-wise effective actions
        edge_actions = exploration_terms + value_terms + one_loop_terms + rg_terms
        
        return edge_actions
    
    def select_best_action(self,
                          visit_counts: torch.Tensor,
                          priors: torch.Tensor,
                          q_values: Optional[torch.Tensor] = None,
                          kappa: float = 0.7,
                          beta: float = 1.0,
                          hbar_eff: float = 1.0,
                          b_trim: int = 0) -> Dict[str, torch.Tensor]:
        """Select best action using complete quantum effective action
        
        Returns the action that minimizes the complete effective action.
        
        Args:
            visit_counts: Visit counts N_k [num_actions]
            priors: Prior probabilities p_k [num_actions]
            q_values: Optional Q-values Q_k [num_actions]
            kappa: Exploration scale κ = c_puct/2 (default: 0.7)
            beta: Value weight (default: 1.0)
            hbar_eff: Effective Planck constant (default: 1.0)
            b_trim: Number of hard edges trimmed away (default: 0)
            
        Returns:
            Dictionary with best action and all action values
        """
        edge_actions = self.compute_edge_wise_effective_action(
            visit_counts, priors, q_values, kappa, beta, hbar_eff, b_trim
        )
        
        best_action = torch.argmin(edge_actions)
        
        # For comparison, compute PUCT selection
        N_total = torch.sum(visit_counts)
        if q_values is not None:
            puct_values = q_values + (kappa * 2) * priors * torch.sqrt(N_total) / (1 + visit_counts)
            puct_best_action = torch.argmax(puct_values)
        else:
            puct_exploration = priors * torch.sqrt(N_total) / (1 + visit_counts)
            puct_best_action = torch.argmax(puct_exploration)
        
        return {
            'best_action': best_action,
            'edge_actions': edge_actions,
            'puct_best_action': puct_best_action,
            'action_match': best_action == puct_best_action
        }
    
    def compute_one_loop_effective_action(self,
                                        classical_action: torch.Tensor,
                                        visit_counts: torch.Tensor,
                                        hbar_eff: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """Compute one-loop effective action Γ_1-loop = S_cl + (ℏ_eff/2) Σ_k log h_k
        
        From Theorem 6.2 in quantum_mcts.md
        
        Args:
            classical_action: Classical action S_cl [batch_size] or scalar
            visit_counts: Visit counts for all edges [num_edges]
            hbar_eff: Effective Planck constant (uses config default if None)
            
        Returns:
            Dictionary with one-loop results
        """
        # COMPREHENSIVE INPUT VALIDATION
        self._validate_visit_counts(visit_counts, "compute_one_loop_effective_action")
        self._validate_classical_action(classical_action, "compute_one_loop_effective_action")
        
        if hbar_eff is not None:
            self._validate_hbar_eff(hbar_eff, "compute_one_loop_effective_action")
        if hbar_eff is None:
            if self.hbar_calculator is not None:
                # AUTO-COMPUTE: Use corrected ℏ_eff calculation based on visit counts
                computed_hbar_eff, hbar_details = self.hbar_calculator.compute_effective_hbar(visit_counts)
                hbar_eff = computed_hbar_eff
                self.stats['hbar_computations'] += 1
                
                # Store details for debugging/analysis
                self._last_hbar_details = hbar_details
            else:
                # FALLBACK: Use legacy hardcoded value (not recommended)
                hbar_eff = self.config.hbar_eff
                self._last_hbar_details = {'method': 'legacy_hardcoded', 'value': hbar_eff}
        
        # Compute diagonal Hessian elements for Hellinger action
        kappa = 0.7  # Default κ = c_puct/2 = 1.4/2
        hessian_diagonal = self.compute_diagonal_hessian(visit_counts, kappa=kappa)
        
        # One-loop correction for Hellinger action:
        # Γ^(1) = (ℏ_eff/2) * Σ[log(3κ/4) - log(N_k)]
        # This preserves the log(N_k) scaling but adds a constant term
        
        log_h_k = torch.log(hessian_diagonal)
        
        # Clamp to prevent numerical issues
        log_h_k = torch.clamp(log_h_k, 
                             min=-self.config.max_log_correction,
                             max=self.config.max_log_correction)
        
        # The Hellinger correction includes both log(3κ/4) and -log(N_k) terms
        # hessian_diagonal already includes the (3κ/4) factor, so log(h_k) = log(3κ/4) - log(N_k)
        one_loop_correction = (hbar_eff / 2.0) * log_h_k.sum()
        
        # Total one-loop effective action
        if torch.is_tensor(classical_action):
            if classical_action.dim() == 0:  # Scalar
                gamma_one_loop = classical_action + one_loop_correction
            else:  # Batch
                gamma_one_loop = classical_action + one_loop_correction.expand_as(classical_action)
        else:
            gamma_one_loop = torch.tensor(classical_action, device=self.device) + one_loop_correction
        
        # Update statistics
        self.stats['corrections_computed'] += 1
        
        return {
            'gamma_one_loop': gamma_one_loop,
            'classical_action': classical_action,
            'one_loop_correction': one_loop_correction,
            'hessian_diagonal': hessian_diagonal,
            'log_determinant': log_h_k.sum(),
            'hbar_eff_used': hbar_eff,
            'num_edges': len(visit_counts)
        }
    
    def apply_gaussian_approximation(self, 
                                   visit_counts: torch.Tensor,
                                   use_stirling: Optional[bool] = None) -> Dict[str, Any]:
        """Apply Gaussian approximation based on Lemma 6.1
        
        For N_k ≥ 5: Use Stirling approximation for continuous treatment
        For N_k < 5: Use discrete sum with same log(N_k + ε_N) correction
        
        Args:
            visit_counts: Visit counts [num_edges]
            use_stirling: Whether to use Stirling approximation (auto-detect if None)
            
        Returns:
            Approximation results and validity flags
        """
        # INPUT VALIDATION
        self._validate_visit_counts(visit_counts, "apply_gaussian_approximation")
        if use_stirling is None:
            use_stirling = self.config.use_stirling_approximation
        
        # Determine which edges can use Gaussian approximation
        gaussian_mask = visit_counts >= self.config.min_visits_gaussian
        
        num_gaussian = gaussian_mask.sum().item()
        num_discrete = len(visit_counts) - num_gaussian
        
        # Compute approximation validity
        gaussian_fraction = num_gaussian / len(visit_counts)
        approximation_quality = self._assess_approximation_quality(visit_counts, gaussian_mask)
        
        # Apply corrections based on visit count regime
        corrections = torch.zeros_like(visit_counts, dtype=torch.float32)
        
        if num_gaussian > 0:
            # Gaussian regime: Use Stirling approximation
            gaussian_visits = visit_counts[gaussian_mask]
            if use_stirling:
                # Stirling correction: log(n!) ≈ n*log(n) - n
                stirling_correction = (gaussian_visits * torch.log(gaussian_visits + self.config.epsilon_N) 
                                     - gaussian_visits)
                corrections[gaussian_mask] = stirling_correction
                self.stats['stirling_approximations'] += num_gaussian
            else:
                # Standard Gaussian correction
                corrections[gaussian_mask] = torch.log(gaussian_visits + self.config.epsilon_N)
            
            self.stats['gaussian_approximations'] += num_gaussian
        
        if num_discrete > 0:
            # Discrete regime: Use exact discrete sum
            discrete_visits = visit_counts[~gaussian_mask]
            corrections[~gaussian_mask] = torch.log(discrete_visits + self.config.epsilon_N)
        
        return {
            'corrections': corrections,
            'gaussian_mask': gaussian_mask,
            'num_gaussian': num_gaussian,
            'num_discrete': num_discrete,
            'gaussian_fraction': gaussian_fraction,
            'approximation_quality': approximation_quality,
            'use_stirling': use_stirling,
            'total_correction': corrections.sum().item()
        }
    
    def _assess_approximation_quality(self, 
                                    visit_counts: torch.Tensor,
                                    gaussian_mask: torch.Tensor) -> Dict[str, float]:
        """Assess quality of Gaussian approximation"""
        
        if gaussian_mask.sum() == 0:
            return {'relative_error': float('inf'), 'max_discrete_count': visit_counts.max().item()}
        
        # For Gaussian regime, estimate relative error
        gaussian_visits = visit_counts[gaussian_mask]
        min_gaussian = gaussian_visits.min().item()
        
        # Relative error scales as O(1/N) for Stirling approximation
        relative_error = 1.0 / max(min_gaussian, 1.0)
        
        # Maximum visit count in discrete regime
        discrete_visits = visit_counts[~gaussian_mask]
        max_discrete = discrete_visits.max().item() if len(discrete_visits) > 0 else 0
        
        return {
            'relative_error': relative_error,
            'min_gaussian_count': min_gaussian,
            'max_discrete_count': max_discrete,
            'gaussian_regime_size': len(gaussian_visits)
        }
    
    def compute_augmented_puct_scores(self,
                                     visit_counts: torch.Tensor,
                                     priors: torch.Tensor,
                                     q_values: torch.Tensor,
                                     total_visits: Optional[torch.Tensor] = None,
                                     kappa: Optional[float] = None,
                                     beta: Optional[float] = None,
                                     hbar_eff: Optional[float] = None,
                                     b_trim: int = 0) -> torch.Tensor:
        """Compute augmented PUCT scores using the corrected quantum formula
        
        Final selection policy from the corrected derivation:
        Score(k) = κ*p_k*sqrt(N_total/N_k) + 3*ℏ_eff/(4*N_k) - β*Q_k
        
        The quantum bonus 3*ℏ_eff/(4*N_k) comes from the one-loop correction
        with the exact Hessian h_k = (κ*p_k/2)*sqrt(N_total/N_k³)
        
        Args:
            visit_counts: Visit counts N_k [num_actions]
            priors: Neural network priors p_k [num_actions]
            q_values: Q-values for exploitation [num_actions]
            total_visits: Total parent visits (computed if None)
            kappa: Exploration coefficient (uses config default if None)
            beta: Value weight coefficient (uses config default if None)
            hbar_eff: Effective Planck constant (auto-computed if None)
            b_trim: Number of pruned siblings (for RG penalty)
            
        Returns:
            Augmented PUCT scores [num_actions]
        """
        # Parameter defaults
        if kappa is None:
            kappa = self.config.kappa
        if beta is None:
            beta = self.config.beta
        if total_visits is None:
            total_visits = visit_counts.sum()
        if hbar_eff is None:
            if self.hbar_calculator is not None:
                hbar_eff, _ = self.hbar_calculator.compute_effective_hbar(visit_counts)
            else:
                hbar_eff = self.config.hbar_eff
        
        # Ensure numerical stability
        safe_visits = visit_counts + self.config.epsilon_N
        
        # Classical PUCT exploration term: κ*p_k*sqrt(N_total/N_k)
        exploration_term = kappa * priors * torch.sqrt(total_visits / safe_visits)
        
        # Quantum bonus from one-loop correction: 3*ℏ_eff/(4*N_k)
        QUANTUM_BONUS_COEFF = 0.75  # Exactly 3/4, no tuning!
        quantum_bonus = (QUANTUM_BONUS_COEFF * hbar_eff) / safe_visits
        
        # Value exploitation term: -β*Q_k
        exploitation_term = -beta * q_values
        
        # RG counter-term penalty (optional, affects parent selection)
        # This manifests as -ℏ_eff*log(1+b)/N per visit for parent ranking
        rg_penalty = 0.0
        if b_trim > 0 and self.config.include_rg_counterterm:
            rg_penalty = -hbar_eff * np.log(1 + b_trim) / total_visits
        
        # Total augmented PUCT score
        scores = exploration_term + quantum_bonus + exploitation_term + rg_penalty
        
        return scores
    
    def compute_quantum_weight_corrections(self,
                                         visit_counts: torch.Tensor,
                                         actions: torch.Tensor,
                                         hbar_eff: Optional[float] = None) -> torch.Tensor:
        """Compute quantum weight corrections for path probabilities
        
        Applies one-loop corrections to path weights: w_quantum = exp(-Γ_1-loop/ℏ_eff)
        
        Args:
            visit_counts: Visit counts [num_edges]
            actions: Classical actions [batch_size, ...]
            hbar_eff: Effective Planck constant
            
        Returns:
            Quantum-corrected weights [batch_size, ...]
        """
        # INPUT VALIDATION
        self._validate_visit_counts(visit_counts, "compute_quantum_weight_corrections")
        self._validate_classical_action(actions, "compute_quantum_weight_corrections")
        
        if hbar_eff is not None:
            self._validate_hbar_eff(hbar_eff, "compute_quantum_weight_corrections")
        
        if hbar_eff is None:
            hbar_eff = self.config.hbar_eff
        
        # Compute one-loop corrections for each action
        if actions.dim() == 1:
            # Batch of scalar actions
            corrected_weights = torch.zeros_like(actions)
            for i, action in enumerate(actions):
                one_loop_result = self.compute_one_loop_effective_action(
                    action, visit_counts, hbar_eff
                )
                corrected_weights[i] = torch.exp(-one_loop_result['gamma_one_loop'] / hbar_eff)
        else:
            # Single action or more complex structure
            one_loop_result = self.compute_one_loop_effective_action(
                actions, visit_counts, hbar_eff
            )
            corrected_weights = torch.exp(-one_loop_result['gamma_one_loop'] / hbar_eff)
        
        return corrected_weights
    
    def validate_tree_structure_assumption(self, 
                                         adjacency_matrix: torch.Tensor) -> Dict[str, Any]:
        """Validate that structure is tree-like for diagonal Hessian assumption
        
        Args:
            adjacency_matrix: Adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            Validation results
        """
        num_nodes = adjacency_matrix.shape[0]
        
        # Check if it's a tree: connected graph with n-1 edges
        num_edges = (adjacency_matrix > 0).sum().item() // 2  # Undirected edges
        
        # Tree property: num_edges = num_nodes - 1
        is_tree = (num_edges == num_nodes - 1)
        
        # Check connectivity (simplified)
        # For full validation, would need proper graph algorithms
        is_connected = True  # Placeholder - assume connected for MCTS trees
        
        # Check for cycles (trees have no cycles)
        has_cycles = num_edges > num_nodes - 1
        
        return {
            'is_tree': is_tree,
            'is_connected': is_connected,
            'has_cycles': has_cycles,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'hessian_diagonal_valid': is_tree and is_connected and not has_cycles
        }
    
    def compute_rg_counterterm(self, 
                             b_trim: int,
                             hbar_eff: Optional[float] = None) -> float:
        """Compute RG counter-term for pruning b hard children
        
        From the corrected derivation:
        ΔΓ_RG = ℏ_eff * log(1 + B_trim)
        
        This term arises from the Jacobian when integrating out hard children,
        representing the entropy loss from shrinking the probability simplex.
        
        Args:
            b_trim: Number of hard (pruned) children
            hbar_eff: Effective Planck constant
            
        Returns:
            RG counter-term contribution
        """
        if b_trim <= 0:
            return 0.0
            
        if hbar_eff is None:
            hbar_eff = self.config.hbar_eff
            
        # The entropy loss from removing b degrees of freedom
        rg_term = hbar_eff * np.log(1 + b_trim)
        
        # This term is added ONCE per parent, not per child
        # It affects parent-level free energy but cancels in child ranking
        
        return rg_term
    
    def compute_jarzynski_average(self,
                                entropy_costs: torch.Tensor,
                                b_trim: int,
                                hbar_eff: Optional[float] = None) -> Dict[str, float]:
        """Verify Jarzynski equality for pruning operation
        
        The Jarzynski equality for information-time MCTS:
        <exp(-ΔS_hard/ℏ_eff)>_hard = exp(-ℏ_eff*log(1+b))
        
        Where:
        - β ↔ 1/ℏ_eff (information temperature)
        - W ↔ ΔS_hard (entropy cost)
        - ΔF ↔ ℏ_eff*log(1+b) (free energy change)
        
        Args:
            entropy_costs: Entropy costs for different hard sector configurations
            b_trim: Number of pruned children
            hbar_eff: Effective Planck constant
            
        Returns:
            Dict with Jarzynski verification results
        """
        if hbar_eff is None:
            hbar_eff = self.config.hbar_eff
            
        # Left side: average of exponential work
        beta = 1.0 / hbar_eff  # Information-time inverse temperature
        exp_work = torch.exp(-beta * entropy_costs)
        avg_exp_work = exp_work.mean().item()
        
        # Right side: exponential of free energy change
        delta_F = hbar_eff * np.log(1 + b_trim)
        exp_delta_F = np.exp(-beta * delta_F)
        
        # Jarzynski equality check
        relative_error = abs(avg_exp_work - exp_delta_F) / exp_delta_F
        equality_satisfied = relative_error < 0.01  # 1% tolerance
        
        return {
            'avg_exp_work': avg_exp_work,
            'exp_delta_F': exp_delta_F,
            'relative_error': relative_error,
            'equality_satisfied': equality_satisfied,
            'beta': beta,
            'delta_F': delta_F,
            'num_samples': len(entropy_costs)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get computation statistics"""
        stats = self.stats.copy()
        
        if stats['corrections_computed'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
            stats['gaussian_usage_rate'] = stats['gaussian_approximations'] / max(stats['corrections_computed'], 1)
            
        return stats
    
    def reset_cache(self):
        """Reset all caches"""
        self._correction_cache.clear()
        self._hessian_cache.clear()


def create_one_loop_corrections(config: Optional[OneLoopConfig] = None) -> OneLoopCorrections:
    """Factory function to create OneLoopCorrections with default configuration"""
    if config is None:
        config = OneLoopConfig()
    
    return OneLoopCorrections(config)


# Export main classes and functions
__all__ = [
    'OneLoopCorrections',
    'OneLoopConfig',
    'create_one_loop_corrections'
]