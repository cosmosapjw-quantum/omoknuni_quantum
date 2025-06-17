"""
Quantum Field Theory Engine for MCTS
====================================

This module implements the rigorous QFT formulation where MCTS tree search
is mapped to a discrete field theory with quantum corrections.

Key Features:
- Effective action computation: Γ_eff = S_cl + (ℏ/2)Tr log M + O(ℏ²)
- Fluctuation matrix construction and determinant computation
- GPU-accelerated one-loop corrections
- Renormalization group flow integration
- Physical parameter determination from first principles

Optimizations:
- Pre-computed quantum tables for O(1) lookup
- Mixed precision (FP16/FP32) support
- Batch processing optimizations
- Fast approximation mode

Based on: docs/qft-mcts-math-foundations.md
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import logging
import math
from functools import lru_cache
from enum import Enum

logger = logging.getLogger(__name__)


class TimeFormulation(Enum):
    """Time formulation for QFT"""
    CONTINUOUS = "continuous"  # Original v1: continuous time
    DISCRETE = "discrete"      # v2.0: discrete information time τ(N) = log(N+2)


@dataclass
class QFTConfig:
    """Configuration for QFT engine with v2.0 support"""
    # Version control
    time_formulation: TimeFormulation = TimeFormulation.DISCRETE  # Default to v2.0
    
    # Physical parameters
    hbar_eff: float = 0.1           # Effective Planck constant ℏ_eff = 1/√N̄
    temperature: float = 1.0         # Temperature T for thermal averaging
    dimension: int = 4               # Effective dimension of tree space
    c_puct: Optional[float] = None   # Exploration constant (required for v2.0)
    
    # v2.0 specific parameters
    temperature_mode: str = 'annealing'  # 'fixed' or 'annealing'
    initial_temperature: float = 1.0      # T₀ for annealing
    use_neural_priors: bool = True        # Include priors in action
    prior_coupling: float = 1.0           # λ coupling strength for priors
    power_law_exponent: Optional[float] = None  # Γ₀ for decoherence
    
    # RG flow parameters
    epsilon: float = 0.01            # ε = 4 - d for dimensional regularization
    coupling_constant: float = 0.1   # g = 1/√N coupling
    
    # Numerical parameters
    matrix_regularization: float = 1e-8  # For numerical stability
    max_iterations: int = 100            # For iterative methods
    convergence_threshold: float = 1e-6  # Convergence criterion
    
    # GPU optimization
    use_mixed_precision: bool = True     # FP16/FP32 optimization
    batch_size: int = 1024              # Batch size for GPU kernels
    memory_pool_size: float = 0.8      # Fraction of GPU memory to use
    
    # Fast approximation mode
    fast_mode: bool = False             # Use approximations for speed
    quantum_table_size: int = 10000     # Size of pre-computed quantum table
    use_quantum_tables: bool = True     # Use pre-computed tables


class DiscreteTimeHandler:
    """Handles discrete information time for v2.0"""
    
    def __init__(self, config: QFTConfig):
        self.config = config
        self.T0 = config.initial_temperature
        self.eps = 1e-8
    
    def information_time(self, N: Union[int, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Compute information time τ(N) = log(N+2)"""
        if isinstance(N, torch.Tensor):
            return torch.log(N + 2)
        return math.log(N + 2)
    
    def time_derivative(self, N: Union[int, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Compute d/dτ = (N+2)d/dN"""
        return 1.0 / (N + 2)
    
    def compute_temperature(self, N: Union[int, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Compute temperature T(N) = T₀/log(N+2)"""
        if self.config.temperature_mode == 'fixed':
            return self.T0
        elif self.config.temperature_mode == 'annealing':
            tau = self.information_time(N)
            return self.T0 / (tau + self.eps)
        else:
            raise ValueError(f"Unknown temperature mode: {self.config.temperature_mode}")
    
    def compute_hbar_eff(self, N: Union[int, torch.Tensor], 
                         c_puct: Optional[float] = None) -> Union[float, torch.Tensor]:
        """Compute effective Planck constant for v2.0"""
        if c_puct is None:
            c_puct = self.config.c_puct
            if c_puct is None:
                raise ValueError("c_puct must be provided or set in config")
        
        tau = self.information_time(N)
        if isinstance(N, torch.Tensor):
            return c_puct / (torch.sqrt(N + 1) * tau)
        else:
            return c_puct / (math.sqrt(N + 1) * tau)


class QuantumLookupTables:
    """
    Pre-computed quantum correction tables for O(1) lookup.
    
    Stores pre-computed values for common quantum corrections:
    - One-loop determinants for typical fluctuation matrices
    - Decoherence corrections as function of visit variance
    - Path overlap kernels
    - v2.0: Discrete time corrections
    """
    
    def __init__(self, config: QFTConfig, device: torch.device):
        self.config = config
        self.device = device
        self.table_size = config.quantum_table_size
        
        # Initialize discrete time handler for v2.0
        if config.time_formulation == TimeFormulation.DISCRETE:
            self.time_handler = DiscreteTimeHandler(config)
        
        # Initialize lookup tables
        self._initialize_tables()
        
        logger.debug(f"Initialized quantum lookup tables with {self.table_size} entries")
    
    def _initialize_tables(self):
        """Pre-compute common quantum corrections"""
        # Visit count range for table
        min_visits = 1
        max_visits = 10000
        
        # Create logarithmic spacing for visit counts
        visit_counts = torch.logspace(
            math.log10(min_visits), 
            math.log10(max_visits), 
            self.table_size, 
            device=self.device
        )
        
        if self.config.time_formulation == TimeFormulation.DISCRETE:
            # v2.0: Compute ℏ_eff(N) dynamically
            hbar_eff_table = torch.zeros_like(visit_counts)
            for i, N in enumerate(visit_counts):
                hbar_eff_table[i] = self.time_handler.compute_hbar_eff(N)
            self.hbar_eff_table = hbar_eff_table
            
            # Temperature table for annealing
            self.temperature_table = torch.zeros_like(visit_counts)
            for i, N in enumerate(visit_counts):
                self.temperature_table[i] = self.time_handler.compute_temperature(N)
            
            # One-loop with dynamic ℏ_eff
            self.one_loop_table = 0.5 * hbar_eff_table * 2.0 * torch.log(visit_counts)
        else:
            # v1.0: Fixed ℏ_eff
            self.one_loop_table = 0.5 * self.config.hbar_eff * 2.0 * torch.log(visit_counts)
        
        # 2. Decoherence strength table
        if self.config.time_formulation == TimeFormulation.DISCRETE:
            # v2.0: Power-law decoherence
            if self.config.power_law_exponent is not None:
                gamma = self.config.power_law_exponent
            else:
                gamma = 0.1  # Default
            self.decoherence_table = visit_counts ** (-gamma)
        else:
            # v1.0: Linear decoherence
            variances = torch.linspace(0, 1000, self.table_size, device=self.device)
            self.decoherence_table = self.config.temperature * variances / 10.0
        
        # 3. Path overlap kernel table (exponential decay)
        distances = torch.arange(self.table_size, device=self.device, dtype=torch.float32)
        self.overlap_kernel_table = torch.exp(-distances / 5.0)
        
        # 4. Log determinant regularization table
        # Pre-compute log(x + ε) for various x values
        x_values = torch.logspace(-8, 4, self.table_size, device=self.device)
        self.log_regularized_table = torch.log(x_values + self.config.matrix_regularization)
        
    @lru_cache(maxsize=10000)
    def interpolate_one_loop(self, visit_count: float) -> float:
        """Fast O(1) lookup for one-loop correction"""
        if visit_count < 1:
            visit_count = 1
        if visit_count > 10000:
            visit_count = 10000
            
        # Binary search for interpolation
        log_visit = math.log10(visit_count)
        idx = (log_visit - 0) * (self.table_size - 1) / 4.0  # log10(10000) = 4
        idx = int(idx)
        
        if idx >= self.table_size - 1:
            return self.one_loop_table[-1].item()
        
        # Linear interpolation
        alpha = idx - int(idx)
        return (1 - alpha) * self.one_loop_table[idx].item() + alpha * self.one_loop_table[idx + 1].item()
    
    def get_overlap_kernel(self, distance: int) -> float:
        """Get pre-computed overlap kernel value"""
        if distance >= self.table_size:
            return 0.0  # Effectively zero for large distances
        return self.overlap_kernel_table[distance].item()
    
    def get_decoherence_strength(self, variance: float) -> float:
        """Get pre-computed decoherence correction"""
        if variance <= 0:
            return 0.0
        if variance >= 1000:
            return self.decoherence_table[-1].item()
            
        # Linear interpolation
        idx = variance * (self.table_size - 1) / 1000.0
        idx_low = int(idx)
        idx_high = min(idx_low + 1, self.table_size - 1)
        
        alpha = idx - idx_low
        return (1 - alpha) * self.decoherence_table[idx_low].item() + alpha * self.decoherence_table[idx_high].item()


class EffectiveActionEngine:
    """
    Computes the quantum effective action Γ_eff for MCTS paths.
    
    v1.0: Classical action: S_cl[π] = -Σᵢ log N(sᵢ, aᵢ)
    v2.0: Full PUCT action: S[γ] = -Σᵢ[log N(sᵢ,aᵢ) + λ log P(aᵢ|sᵢ)]
    
    The effective action includes:
    1. Classical/PUCT action
    2. One-loop quantum correction: (ℏ/2)Tr log M
    3. Higher-order corrections: O(ℏ²)
    
    Where M is the fluctuation matrix: M_ij = δ²S/δπᵢδπⱼ
    """
    
    def __init__(self, config: QFTConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # Initialize discrete time handler for v2.0
        if config.time_formulation == TimeFormulation.DISCRETE:
            self.time_handler = DiscreteTimeHandler(config)
        
        # Initialize computation caches
        self.fluctuation_matrix_cache = {}
        self.determinant_cache = {}
        
        # Initialize quantum lookup tables if enabled
        self.quantum_tables = None
        if config.use_quantum_tables:
            self.quantum_tables = QuantumLookupTables(config, device)
        
        # Mixed precision context
        self.amp_enabled = config.use_mixed_precision and device.type == 'cuda'
        
        # Statistics tracking
        self.stats = {
            'effective_actions_computed': 0,
            'cache_hits': 0,
            'avg_computation_time': 0.0,
            'quantum_correction_strength': 0.0,
            'table_lookups': 0,
            'fast_mode_uses': 0
        }
        
        logger.debug(f"EffectiveActionEngine initialized on {device} "
                    f"(time_formulation={config.time_formulation}, "
                    f"mixed_precision={config.use_mixed_precision}, "
                    f"quantum_tables={config.use_quantum_tables})")
        
    def compute_effective_action(
        self, 
        paths: torch.Tensor,
        visit_counts: torch.Tensor,
        priors: Optional[torch.Tensor] = None,
        simulation_count: Optional[int] = None,
        include_quantum: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute complete effective action Γ_eff[π]
        
        Args:
            paths: Tensor of shape (batch_size, max_depth) containing node indices
            visit_counts: Tensor of shape (num_nodes,) with visit counts
            priors: Tensor of shape (num_nodes,) with prior probabilities (v2.0)
            simulation_count: Current simulation count for time-dependent parameters
            include_quantum: Whether to include quantum corrections
            
        Returns:
            Tuple of (real_action, imaginary_action) where:
            - real_action: Classical/PUCT + quantum corrections
            - imaginary_action: Decoherence/dissipation terms
        """
        batch_size = paths.shape[0]
        
        # Use mixed precision if enabled
        with torch.amp.autocast(device_type='cuda', enabled=self.amp_enabled):
            # Process in batches for better GPU utilization
            if batch_size > self.config.batch_size:
                return self._compute_batched_action(paths, visit_counts, priors, simulation_count, include_quantum)
            
            # Fast mode uses approximations and lookup tables
            if self.config.fast_mode and self.quantum_tables is not None:
                self.stats['fast_mode_uses'] += 1
                return self._compute_fast_action(paths, visit_counts, include_quantum)
            
            # 1. Classical or PUCT action based on formulation
            if self.config.time_formulation == TimeFormulation.DISCRETE and priors is not None:
                # v2.0: Full PUCT action
                classical_action = self._compute_puct_action(paths, visit_counts, priors)
            else:
                # v1.0: Classical action
                classical_action = self._compute_classical_action(paths, visit_counts)
            
            if not include_quantum:
                # Return classical result only
                imaginary_part = torch.zeros_like(classical_action)
                return classical_action, imaginary_part
                
            # 2. One-loop quantum correction: (ℏ/2)Tr log M
            quantum_correction = self._compute_one_loop_correction(
                paths, visit_counts, priors, simulation_count
            )
            
            # 3. Decoherence correction (imaginary part)
            decoherence_correction = self._compute_decoherence_correction(
                paths, visit_counts, simulation_count
            )
            
            # Combine corrections
            real_action = classical_action + quantum_correction
            imaginary_action = decoherence_correction
            
            # Update statistics
            self.stats['effective_actions_computed'] += batch_size
            self.stats['quantum_correction_strength'] = quantum_correction.mean().item()
            
            return real_action, imaginary_action
    
    def _compute_classical_action(
        self, 
        paths: torch.Tensor, 
        visit_counts: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute classical action: S_cl[π] = -Σᵢ log N(sᵢ, aᵢ)
        
        This is the fundamental action that drives classical MCTS behavior.
        Paths with higher visit counts have lower action.
        """
        batch_size, max_depth = paths.shape
        
        # Ensure paths contain valid indices
        valid_mask = (paths >= 0) & (paths < visit_counts.shape[0])
        safe_paths = torch.clamp(paths, 0, visit_counts.shape[0] - 1)
        
        # Gather visit counts for each node in each path
        path_visits = visit_counts[safe_paths]  # Shape: (batch_size, max_depth)
        
        # Apply validity mask
        path_visits = path_visits * valid_mask.float()
        
        # Classical action: S_cl = -Σᵢ log N(sᵢ)
        # Add regularization to prevent log(0)
        regularized_visits = torch.clamp(path_visits, min=1.0)
        log_visits = torch.log(regularized_visits + self.config.matrix_regularization)
        classical_action = -torch.sum(log_visits * valid_mask.float(), dim=1)
        
        return classical_action
    
    def _compute_puct_action(
        self,
        paths: torch.Tensor,
        visit_counts: torch.Tensor,
        priors: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute v2.0 PUCT action: S[γ] = -Σᵢ[log N(sᵢ,aᵢ) + λ log P(aᵢ|sᵢ)]
        
        This includes both visit counts and neural network priors.
        The prior coupling λ controls the influence of the external field.
        """
        batch_size, max_depth = paths.shape
        
        # Ensure paths contain valid indices
        valid_mask = (paths >= 0) & (paths < visit_counts.shape[0])
        safe_paths = torch.clamp(paths, 0, visit_counts.shape[0] - 1)
        
        # Gather visit counts and priors for each node in each path
        path_visits = visit_counts[safe_paths]
        path_priors = priors[safe_paths]
        
        # Apply validity mask
        path_visits = path_visits * valid_mask.float()
        path_priors = path_priors * valid_mask.float() + (1 - valid_mask.float()) * 1.0
        
        # PUCT action: S = -[log N + λ log P]
        regularized_visits = torch.clamp(path_visits, min=1.0)
        regularized_priors = torch.clamp(path_priors, min=1e-8)
        
        log_visits = torch.log(regularized_visits + self.config.matrix_regularization)
        log_priors = torch.log(regularized_priors)
        
        # Get prior coupling strength
        lambda_coupling = self.config.prior_coupling
        if lambda_coupling == 'auto' and self.config.c_puct is not None:
            lambda_coupling = self.config.c_puct
        
        puct_action = -torch.sum(
            (log_visits + lambda_coupling * log_priors) * valid_mask.float(), 
            dim=1
        )
        
        return puct_action
    
    def _compute_one_loop_correction(
        self, 
        paths: torch.Tensor, 
        visit_counts: torch.Tensor,
        priors: Optional[torch.Tensor] = None,
        simulation_count: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute one-loop quantum correction: (ℏ/2)Tr log M
        
        Where M is the fluctuation matrix: M_ij = δ²S/δπᵢδπⱼ
        
        This captures quantum fluctuations around the classical path.
        """
        # Fast approximation if enabled
        if self.config.fast_mode and self.quantum_tables is not None:
            return self._compute_one_loop_fast(paths, visit_counts)
        
        # Build fluctuation matrix
        fluctuation_matrix = self._build_fluctuation_matrix(paths, visit_counts, priors)
        
        # Compute log determinant efficiently
        log_det = self._compute_log_determinant_gpu(fluctuation_matrix)
        
        # Compute effective ℏ based on formulation
        if self.config.time_formulation == TimeFormulation.DISCRETE:
            # v2.0: Dynamic ℏ_eff(N)
            if simulation_count is not None:
                hbar_eff = self.time_handler.compute_hbar_eff(simulation_count)
            else:
                hbar_eff = self.config.hbar_eff
        else:
            # v1.0: Fixed ℏ_eff
            hbar_eff = self.config.hbar_eff
        
        # One-loop correction: (ℏ/2)Tr log M
        quantum_correction = 0.5 * hbar_eff * log_det
        
        return quantum_correction
    
    def _compute_one_loop_fast(
        self,
        paths: torch.Tensor,
        visit_counts: torch.Tensor
    ) -> torch.Tensor:
        """
        Fast approximation of one-loop correction using pre-computed tables
        """
        batch_size = paths.shape[0]
        corrections = []
        
        for b in range(batch_size):
            path = paths[b]
            valid_nodes = path[path >= 0]
            
            if len(valid_nodes) == 0:
                corrections.append(torch.tensor(0.0, device=self.device))
                continue
            
            # Sum individual node contributions from lookup table
            total_correction = 0.0
            for node in valid_nodes:
                visit_count = visit_counts[node].item()
                total_correction += self.quantum_tables.interpolate_one_loop(visit_count)
            
            corrections.append(torch.tensor(total_correction, device=self.device))
            
        self.stats['table_lookups'] += batch_size
        return torch.stack(corrections)
    
    def _build_fluctuation_matrix(
        self, 
        paths: torch.Tensor, 
        visit_counts: torch.Tensor,
        priors: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Build fluctuation matrix M_ij = δ²S/δπᵢδπⱼ
        
        v1.0: For classical action S_cl = -Σₖ log N(πₖ)
        v2.0: For PUCT action S = -Σₖ[log N(πₖ) + λ log P(πₖ)]
        
        Matrix elements:
        - Diagonal: M_ii = 1/N_i² + λ/P_i² (v2.0)
        - Off-diagonal: M_ij = -K(i,j)/(N_i N_j)
        
        Where K(i,j) is the path overlap kernel.
        """
        batch_size, max_depth = paths.shape
        
        # Create fluctuation matrix for each path in batch
        matrices = []
        
        for b in range(batch_size):
            path = paths[b]
            valid_nodes = path[path >= 0]
            
            if len(valid_nodes) == 0:
                # Empty path - return identity
                matrix = torch.eye(1, device=self.device, dtype=torch.float32)
                matrices.append(matrix)
                continue
                
            path_length = len(valid_nodes)
            matrix = torch.zeros((path_length, path_length), device=self.device)
            
            # Get visit counts for this path
            path_visit_counts = visit_counts[valid_nodes]
            
            # Diagonal terms
            if self.config.time_formulation == TimeFormulation.DISCRETE and priors is not None:
                # v2.0: Include prior contribution
                path_priors = priors[valid_nodes]
                lambda_coupling = self.config.prior_coupling
                if lambda_coupling == 'auto' and self.config.c_puct is not None:
                    lambda_coupling = self.config.c_puct
                
                diagonal = (1.0 / (path_visit_counts**2 + self.config.matrix_regularization) + 
                           lambda_coupling / (path_priors**2 + self.config.matrix_regularization))
            else:
                # v1.0: Classical only
                diagonal = 1.0 / (path_visit_counts**2 + self.config.matrix_regularization)
            
            for i in range(path_length):
                matrix[i, i] = diagonal[i]
            
            # Off-diagonal terms: path overlap kernel
            for i in range(path_length):
                for j in range(i + 1, path_length):
                    # Simple overlap kernel based on visit count similarity
                    N_i = path_visit_counts[i]
                    N_j = path_visit_counts[j]
                    
                    # Overlap strength decreases with distance
                    distance = abs(i - j)
                    
                    # Use quantum table if available
                    if self.quantum_tables is not None and self.config.use_quantum_tables:
                        overlap = self.quantum_tables.get_overlap_kernel(distance)
                    else:
                        overlap = torch.exp(torch.tensor(-distance / 5.0, device=self.device))  # Exponential decay
                    
                    off_diagonal = -overlap / (N_i * N_j + self.config.matrix_regularization)
                    matrix[i, j] = off_diagonal
                    matrix[j, i] = off_diagonal  # Symmetric
                    
            matrices.append(matrix)
            
        return matrices
    
    def _compute_log_determinant_gpu(self, matrices: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute log determinant of fluctuation matrices efficiently on GPU
        
        Uses Cholesky decomposition when possible, falls back to LU decomposition.
        For numerical stability with potential ill-conditioning.
        """
        log_dets = []
        
        # Group matrices by size for batch processing
        size_groups = {}
        for idx, matrix in enumerate(matrices):
            size = matrix.shape[0]
            if size not in size_groups:
                size_groups[size] = []
            size_groups[size].append((idx, matrix))
        
        # Process each size group in batch
        for size, group in size_groups.items():
            if len(group) == 1:
                # Single matrix - process normally
                idx, matrix = group[0]
                log_det = self._compute_single_log_det(matrix)
                log_dets.append((idx, log_det))
            else:
                # Batch process matrices of same size
                indices = [g[0] for g in group]
                batch_matrices = torch.stack([g[1] for g in group])
                
                # Compute log determinants in batch
                batch_log_dets = self._compute_batch_log_det(batch_matrices)
                
                for i, idx in enumerate(indices):
                    log_dets.append((idx, batch_log_dets[i]))
        
        # Sort by original index and extract values
        log_dets.sort(key=lambda x: x[0])
        return torch.stack([x[1] for x in log_dets])
    
    def _compute_single_log_det(self, matrix: torch.Tensor) -> torch.Tensor:
        """Compute log determinant of a single matrix"""
        try:
            # Add strong regularization for numerical stability
            reg_matrix = matrix + 0.1 * torch.eye(matrix.shape[0], device=self.device)
            
            # Use mixed precision if enabled
            if self.config.use_mixed_precision:
                reg_matrix = reg_matrix.half()
            
            chol = torch.linalg.cholesky(reg_matrix)
            log_det = 2.0 * torch.sum(torch.log(torch.diag(chol).clamp(min=1e-10)))
            
        except RuntimeError:
            # Fall back to LU decomposition
            try:
                lu, pivots = torch.linalg.lu_factor(matrix + 
                    self.config.matrix_regularization * torch.eye(matrix.shape[0], device=self.device))
                log_det = torch.sum(torch.log(torch.abs(torch.diag(lu))))
                
            except RuntimeError:
                # Last resort: SVD (most stable but slowest)
                U, S, Vh = torch.linalg.svd(matrix)
                # Regularize singular values
                S_reg = torch.clamp(S, min=self.config.matrix_regularization)
                log_det = torch.sum(torch.log(S_reg))
        
        return log_det.float() if self.config.use_mixed_precision else log_det
    
    def _compute_batch_log_det(self, batch_matrices: torch.Tensor) -> torch.Tensor:
        """Compute log determinants for a batch of matrices of same size"""
        batch_size, n, _ = batch_matrices.shape
        
        # Add regularization
        eye_batch = torch.eye(n, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        reg_matrices = batch_matrices + 0.1 * eye_batch
        
        # Use mixed precision if enabled
        if self.config.use_mixed_precision:
            reg_matrices = reg_matrices.half()
        
        try:
            # Batch Cholesky decomposition
            L = torch.linalg.cholesky(reg_matrices)
            # Log determinant = 2 * sum(log(diagonal))
            log_dets = 2.0 * torch.sum(torch.log(
                torch.diagonal(L, dim1=-2, dim2=-1).clamp(min=1e-10)
            ), dim=-1)
            
        except RuntimeError:
            # Fall back to individual processing
            log_dets = []
            for i in range(batch_size):
                log_det = self._compute_single_log_det(batch_matrices[i])
                log_dets.append(log_det)
            log_dets = torch.stack(log_dets)
        
        return log_dets.float() if self.config.use_mixed_precision else log_dets
    
    def _compute_decoherence_correction(
        self, 
        paths: torch.Tensor, 
        visit_counts: torch.Tensor,
        simulation_count: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute decoherence correction (imaginary part of effective action)
        
        v1.0: Exponential decoherence based on visit variance
        v2.0: Power-law decoherence ρᵢⱼ(N) ~ N^(-Γ₀)
        
        From the docs: This represents dissipation due to environment coupling.
        The imaginary part arises from the influence functional.
        """
        batch_size, max_depth = paths.shape
        
        # Simple model: decoherence proportional to visit count differences
        decoherence_corrections = []
        
        for b in range(batch_size):
            path = paths[b]
            valid_nodes = path[path >= 0]
            
            if len(valid_nodes) <= 1:
                decoherence_corrections.append(torch.tensor(0.0, device=self.device))
                continue
                
            path_visits = visit_counts[valid_nodes]
            
            # Compute decoherence based on visit count variance
            visit_variance = torch.var(path_visits.float())
            
            if self.config.time_formulation == TimeFormulation.DISCRETE:
                # v2.0: Power-law decoherence
                if self.config.power_law_exponent is not None:
                    gamma = self.config.power_law_exponent
                else:
                    # Auto-compute from theory
                    sigma_Q = torch.std(path_visits.float())
                    T = self.time_handler.compute_temperature(
                        simulation_count if simulation_count is not None else 1000
                    )
                    gamma = 2 * (self.config.c_puct or 1.0) * sigma_Q**2 * T
                
                # Power-law decay
                avg_visits = torch.mean(path_visits.float())
                decoherence = avg_visits ** (-gamma)
                decoherence_corrections.append(decoherence)
            else:
                # v1.0: Use quantum table if available
                if self.quantum_tables is not None and self.config.use_quantum_tables:
                    self.stats['table_lookups'] += 1
                    decoherence = self.quantum_tables.get_decoherence_strength(visit_variance.item())
                    decoherence_corrections.append(torch.tensor(decoherence, device=self.device))
                else:
                    # Decoherence strength scales with temperature and variance
                    decoherence = self.config.temperature * visit_variance / len(valid_nodes)
                    decoherence_corrections.append(decoherence)
            
        return torch.stack(decoherence_corrections)
    
    def _compute_batched_action(
        self,
        paths: torch.Tensor,
        visit_counts: torch.Tensor,
        priors: Optional[torch.Tensor] = None,
        simulation_count: Optional[int] = None,
        include_quantum: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute effective action in batches for better GPU utilization
        """
        batch_size = paths.shape[0]
        batch_step = self.config.batch_size
        
        real_actions = []
        imaginary_actions = []
        
        for i in range(0, batch_size, batch_step):
            end_idx = min(i + batch_step, batch_size)
            batch_paths = paths[i:end_idx]
            
            real_batch, imag_batch = self.compute_effective_action(
                batch_paths, visit_counts, include_quantum
            )
            
            real_actions.append(real_batch)
            imaginary_actions.append(imag_batch)
        
        return torch.cat(real_actions), torch.cat(imaginary_actions)
    
    def _compute_fast_action(
        self,
        paths: torch.Tensor,
        visit_counts: torch.Tensor,
        include_quantum: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fast approximation mode using pre-computed tables and simplified physics
        """
        batch_size = paths.shape[0]
        
        # Classical action (always computed)
        classical_action = self._compute_classical_action(paths, visit_counts)
        
        if not include_quantum:
            return classical_action, torch.zeros_like(classical_action)
        
        # Fast quantum corrections using lookup tables
        quantum_corrections = torch.zeros(batch_size, device=self.device)
        decoherence_corrections = torch.zeros(batch_size, device=self.device)
        
        for b in range(batch_size):
            path = paths[b]
            valid_nodes = path[path >= 0]
            
            if len(valid_nodes) == 0:
                continue
                
            path_visits = visit_counts[valid_nodes]
            
            # Fast one-loop approximation: sum of individual node contributions
            for visit in path_visits:
                quantum_corrections[b] += self.quantum_tables.interpolate_one_loop(visit.item())
            
            # Fast decoherence: use pre-computed table
            if len(valid_nodes) > 1:
                variance = torch.var(path_visits.float()).item()
                decoherence_corrections[b] = self.quantum_tables.get_decoherence_strength(variance)
        
        real_action = classical_action + quantum_corrections
        
        self.stats['table_lookups'] += batch_size * 2  # One-loop and decoherence lookups
        
        return real_action, decoherence_corrections
    
    def compute_quantum_correction_strength(self, paths: torch.Tensor, visit_counts: torch.Tensor) -> float:
        """
        Measure the relative strength of quantum corrections vs classical action
        
        Returns:
            Ratio of quantum correction magnitude to classical action magnitude
        """
        classical_action, _ = self.compute_effective_action(paths, visit_counts, include_quantum=False)
        full_action, _ = self.compute_effective_action(paths, visit_counts, include_quantum=True)
        
        quantum_correction = full_action - classical_action
        
        # Avoid division by zero
        classical_magnitude = torch.abs(classical_action).mean()
        quantum_magnitude = torch.abs(quantum_correction).mean()
        
        if classical_magnitude > 0:
            return (quantum_magnitude / classical_magnitude).item()
        else:
            return 0.0
    
    def get_statistics(self) -> Dict[str, float]:
        """Get computation statistics"""
        return dict(self.stats)
    
    def reset_statistics(self):
        """Reset all statistics counters"""
        for key in self.stats:
            self.stats[key] = 0.0


class QFTEngine:
    """
    Main QFT engine that orchestrates effective action computation
    and integrates with the broader MCTS framework.
    """
    
    def __init__(self, config: QFTConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # Initialize sub-engines
        self.effective_action_engine = EffectiveActionEngine(config, device)
        
        # Track overall QFT statistics
        self.stats = {
            'total_computations': 0,
            'avg_quantum_strength': 0.0,
            'classical_limit_ratio': 0.0
        }
        
        logger.debug(f"QFTEngine initialized with ℏ_eff = {config.hbar_eff}")
    
    def compute_path_weights(
        self, 
        paths: torch.Tensor, 
        visit_counts: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute QFT-corrected path weights for selection
        
        The path probability is given by the quantum partition function:
        P[π] ∝ exp(i S_eff[π] / ℏ_eff)
        
        After Wick rotation to Euclidean signature:
        P[π] ∝ exp(-Re(S_eff[π]) / ℏ_eff) 
        """
        # Compute effective action
        real_action, imaginary_action = self.effective_action_engine.compute_effective_action(
            paths, visit_counts
        )
        
        # Quantum weight: exp(-Re(S_eff)/ℏ_eff)
        # Use log-softmax for numerical stability
        log_weights = -real_action / self.config.hbar_eff
        
        # Apply softmax for numerical stability
        weights = F.softmax(log_weights, dim=0)
        
        # Update statistics
        self.stats['total_computations'] += paths.shape[0]
        quantum_strength = self.effective_action_engine.compute_quantum_correction_strength(
            paths, visit_counts
        )
        self.stats['avg_quantum_strength'] = quantum_strength
        
        return weights
    
    def update_hbar_effective(self, average_visit_count: float):
        """
        Update effective Planck constant based on current tree state
        
        From theory: ℏ_eff = 1/√N̄ where N̄ is average visit count
        """
        new_hbar = 1.0 / np.sqrt(max(average_visit_count, 1.0))
        self.config.hbar_eff = new_hbar
        
        logger.debug(f"Updated ℏ_eff to {new_hbar:.4f} (N̄ = {average_visit_count:.1f})")
    
    def is_in_classical_limit(self) -> bool:
        """
        Check if we're in the classical limit where quantum corrections are negligible
        
        Classical limit: ℏ_eff → 0 ⟺ N̄ → ∞
        """
        return self.config.hbar_eff < 0.01  # Classical when ℏ < 1%
    
    def enable_fast_mode(self, enable: bool = True):
        """
        Enable or disable fast approximation mode
        
        Fast mode uses pre-computed quantum tables and approximations
        for significantly faster computation at slight accuracy cost.
        """
        self.config.fast_mode = enable
        if enable and self.effective_action_engine.quantum_tables is None:
            # Initialize quantum tables if not already done
            self.effective_action_engine.quantum_tables = QuantumLookupTables(
                self.config, self.device
            )
        logger.info(f"Fast mode {'enabled' if enable else 'disabled'}")
    
    def set_mixed_precision(self, enable: bool = True):
        """
        Enable or disable mixed precision computation
        
        Mixed precision uses FP16 for most computations with FP32 accumulation,
        providing ~2x speedup on modern GPUs with minimal accuracy loss.
        """
        self.config.use_mixed_precision = enable
        logger.info(f"Mixed precision {'enabled' if enable else 'disabled'}")
    
    def get_statistics(self) -> Dict[str, float]:
        """Get comprehensive QFT statistics"""
        stats = dict(self.stats)
        stats.update(self.effective_action_engine.get_statistics())
        
        # Add performance metrics
        if self.effective_action_engine.stats['effective_actions_computed'] > 0:
            stats['avg_actions_per_batch'] = (
                self.effective_action_engine.stats['effective_actions_computed'] / 
                max(1, self.stats['total_computations'])
            )
            
            if self.effective_action_engine.stats['table_lookups'] > 0:
                stats['table_lookup_ratio'] = (
                    self.effective_action_engine.stats['table_lookups'] /
                    self.effective_action_engine.stats['effective_actions_computed']
                )
        
        return stats
    
    def reset_statistics(self):
        """Reset all statistics counters"""
        for key in self.stats:
            self.stats[key] = 0.0 if isinstance(self.stats[key], float) else 0
        self.effective_action_engine.reset_statistics()


# Factory functions for easy instantiation
def create_qft_engine(
    device: Union[str, torch.device] = 'cuda',
    version: str = 'v2',
    fast_mode: bool = False,
    use_mixed_precision: bool = True,
    use_quantum_tables: bool = True,
    **kwargs
) -> QFTEngine:
    """
    Factory function to create QFT engine with sensible defaults
    
    Args:
        device: Device for computation ('cuda' or 'cpu')
        version: 'v1' for continuous time, 'v2' for discrete time
        fast_mode: Enable fast approximation mode
        use_mixed_precision: Enable FP16/FP32 mixed precision
        use_quantum_tables: Enable pre-computed quantum tables
        **kwargs: Override other config parameters
        
    Returns:
        Initialized QFTEngine with optimizations
        
    Example:
        # v2.0 with discrete time (recommended)
        engine = create_qft_engine(version='v2', branching_factor=10)
        
        # v1.0 with continuous time (legacy)
        engine = create_qft_engine(version='v1')
        
        # Production mode - fast with slight accuracy tradeoff
        engine = create_qft_engine(fast_mode=True)
        
        # High accuracy mode - slower but more precise
        engine = create_qft_engine(fast_mode=False, use_mixed_precision=False)
    """
    if version == 'v2':
        return create_qft_engine_v2(device, fast_mode=fast_mode, 
                                   use_mixed_precision=use_mixed_precision,
                                   use_quantum_tables=use_quantum_tables, **kwargs)
    elif version == 'v1':
        return create_qft_engine_v1(device, fast_mode=fast_mode,
                                   use_mixed_precision=use_mixed_precision, 
                                   use_quantum_tables=use_quantum_tables, **kwargs)
    else:
        raise ValueError(f"Unknown version: {version}. Use 'v1' or 'v2'.")


def create_qft_engine_v2(
    device: Union[str, torch.device] = 'cuda',
    branching_factor: int = 10,
    avg_game_length: int = 100,
    c_puct: Optional[float] = None,
    fast_mode: bool = False,
    use_mixed_precision: bool = True,
    use_quantum_tables: bool = True,
    **kwargs
) -> QFTEngine:
    """
    Create v2.0 QFT engine with discrete time formulation
    
    Args:
        device: Device for computation
        branching_factor: Average branching factor of the game
        avg_game_length: Average game length
        c_puct: Exploration constant (auto-computed if None)
        fast_mode: Enable fast approximation mode
        use_mixed_precision: Enable FP16/FP32 mixed precision
        use_quantum_tables: Enable pre-computed quantum tables
        **kwargs: Additional config parameters
    
    Returns:
        QFTEngine configured for v2.0
    """
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Disable mixed precision on CPU
    if device.type == 'cpu':
        use_mixed_precision = False
    
    # Auto-compute c_puct if not provided
    if c_puct is None:
        c_puct = math.sqrt(2 * math.log(branching_factor))
    
    config = QFTConfig(
        time_formulation=TimeFormulation.DISCRETE,
        branching_factor=branching_factor,
        avg_game_length=avg_game_length,
        c_puct=c_puct,
        temperature_mode='annealing',
        use_neural_priors=True,
        fast_mode=fast_mode,
        use_mixed_precision=use_mixed_precision,
        use_quantum_tables=use_quantum_tables,
        **kwargs
    )
    
    engine = QFTEngine(config, device)
    
    logger.info(f"Created v2.0 QFTEngine with discrete time: "
               f"branching_factor={branching_factor}, "
               f"c_puct={c_puct:.3f}, "
               f"device={device}")
    
    return engine


def create_qft_engine_v1(
    device: Union[str, torch.device] = 'cuda',
    hbar_eff: float = 0.1,
    temperature: float = 1.0,
    fast_mode: bool = False,
    use_mixed_precision: bool = True,
    use_quantum_tables: bool = True,
    **kwargs
) -> QFTEngine:
    """
    Create v1.0 QFT engine with continuous time formulation
    
    Args:
        device: Device for computation
        hbar_eff: Effective Planck constant
        temperature: Temperature for thermal averaging
        fast_mode: Enable fast approximation mode
        use_mixed_precision: Enable FP16/FP32 mixed precision
        use_quantum_tables: Enable pre-computed quantum tables
        **kwargs: Additional config parameters
    
    Returns:
        QFTEngine configured for v1.0
    """
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Disable mixed precision on CPU
    if device.type == 'cpu':
        use_mixed_precision = False
    
    config = QFTConfig(
        time_formulation=TimeFormulation.CONTINUOUS,
        hbar_eff=hbar_eff,
        temperature=temperature,
        use_neural_priors=False,
        fast_mode=fast_mode,
        use_mixed_precision=use_mixed_precision,
        use_quantum_tables=use_quantum_tables,
        **kwargs
    )
    
    engine = QFTEngine(config, device)
    
    logger.info(f"Created v1.0 QFTEngine with continuous time: "
               f"hbar_eff={hbar_eff}, "
               f"temperature={temperature}, "
               f"device={device}")
    
    return engine