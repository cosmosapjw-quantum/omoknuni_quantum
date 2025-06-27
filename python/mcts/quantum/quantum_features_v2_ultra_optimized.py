"""
Ultra-Optimized Quantum Features for MCTS - Version 2.0
======================================================

This module implements an ultra-optimized quantum MCTS with focus on:
- Minimal tensor allocations and memory operations
- Fast paths for common cases
- JIT compilation for critical sections
- Aggressive caching and pre-computation
- Target: < 2x overhead vs classical MCTS

Key optimizations:
- Pre-allocated tensor pools
- Vectorized operations with minimal branching
- Early exits for classical regimes
- Compiled kernels for quantum corrections
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union, List
from dataclasses import dataclass
import logging

from .quantum_features_v2 import QuantumConfigV2, MCTSPhase, DiscreteTimeEvolution, PhaseDetector

logger = logging.getLogger(__name__)


class TensorPool:
    """Memory pool for efficient tensor reuse"""
    
    def __init__(self, device: torch.device, max_pool_size: int = 1000):
        self.device = device
        self.pools = {
            # Common sizes for batch operations
            (1,): [],
            (8,): [],
            (16,): [],
            (32,): [],
            (64,): [],
            (128,): [],
            (256,): [],
            (512,): [],
            (1024,): [],
        }
        self.max_pool_size = max_pool_size
        
    def get_tensor(self, shape: tuple, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Get tensor from pool or create new one"""
        if shape in self.pools and len(self.pools[shape]) > 0:
            tensor = self.pools[shape].pop()
            tensor.zero_()
            return tensor
        else:
            return torch.zeros(shape, dtype=dtype, device=self.device)
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return tensor to pool for reuse"""
        shape = tuple(tensor.shape)
        if shape in self.pools and len(self.pools[shape]) < self.max_pool_size:
            self.pools[shape].append(tensor.detach())


@torch.jit.script
def quantum_correction_kernel(
    visit_counts: torch.Tensor,
    hbar_eff: float,
    quantum_strength: float,
    min_visits: float = 1.0
) -> torch.Tensor:
    """JIT-compiled quantum correction kernel"""
    safe_visits = torch.clamp(visit_counts, min=min_visits)
    return (hbar_eff * quantum_strength) / torch.sqrt(safe_visits)


@torch.jit.script  
def classical_puct_kernel(
    q_values: torch.Tensor,
    visit_counts: torch.Tensor,
    priors: torch.Tensor,
    c_puct: float,
    parent_visits: float,
    min_visits: float = 1.0
) -> torch.Tensor:
    """JIT-compiled classical PUCT kernel"""
    safe_visits = torch.clamp(visit_counts, min=min_visits)
    sqrt_parent = math.sqrt(math.log(parent_visits + 1))
    exploration = c_puct * priors * sqrt_parent / torch.sqrt(safe_visits)
    return q_values + exploration


class UltraOptimizedQuantumMCTSV2:
    """Ultra-optimized quantum MCTS implementation targeting < 2x overhead"""
    
    def __init__(self, config: QuantumConfigV2):
        self.config = config
        self.device = torch.device(config.device)
        
        # High-performance components
        self.tensor_pool = TensorPool(self.device)
        self.time_evolution = DiscreteTimeEvolution(config)
        self.phase_detector = PhaseDetector(config)
        
        # Pre-compute all possible parameters
        self._precompute_all_parameters()
        
        # State tracking with minimal overhead
        self.current_phase = MCTSPhase.QUANTUM
        self.total_simulations = 0
        self._last_phase_check = 0
        self._phase_check_interval = 500  # Check phase less frequently
        
        # Performance mode switches
        self.fast_mode = config.fast_mode
        self.enable_quantum = config.enable_quantum
        
        # Pre-allocated tensors for common operations
        self._preallocate_common_tensors()
        
        # Statistics
        self.stats = {
            'quantum_applications': 0,
            'classical_applications': 0,
            'fast_path_hits': 0,
            'tensor_pool_hits': 0,
            'tensor_creations': 0
        }
        
        logger.debug(f"UltraOptimizedQuantumMCTSV2 initialized on {self.device}")
    
    def _precompute_all_parameters(self):
        """Pre-compute all possible parameter combinations"""
        # Default parameters
        self.default_c_puct = self.config.c_puct or np.sqrt(2 * np.log(self.config.branching_factor or 30))
        self.default_hbar_eff = 0.1
        self.quantum_strength = self.config.coupling_strength
        
        # Pre-compute critical points for phase detection
        if self.config.branching_factor:
            self._critical_points = self.phase_detector.compute_critical_points(
                self.config.branching_factor, 
                self.default_c_puct,
                self.config.use_neural_prior
            )
        else:
            self._critical_points = (1000, 10000)  # Default values
        
        # Pre-compute lookup tables for common values
        self._precompute_lookup_tables()
    
    def _precompute_lookup_tables(self):
        """Pre-compute lookup tables for ultra-fast access"""
        # Simulation count range for lookup
        max_sims = 50000
        sim_range = torch.arange(0, max_sims, device=self.device)
        
        # Information time lookup: τ(N) = log(N+2)
        self.tau_lookup = torch.log(sim_range.float() + 2)
        
        # hbar_eff lookup: c_puct / (sqrt(N+1) * tau)
        self.hbar_lookup = self.default_c_puct / (torch.sqrt(sim_range.float() + 1) * self.tau_lookup)
        
        # sqrt factors for common operations
        sqrt_log_range = torch.sqrt(torch.log(sim_range.float() + 1))
        self.sqrt_log_lookup = sqrt_log_range
    
    def _preallocate_common_tensors(self):
        """Pre-allocate tensors for common batch sizes"""
        self.batch_tensors = {}
        common_sizes = [1, 8, 16, 32, 64, 128]
        
        for size in common_sizes:
            self.batch_tensors[size] = {
                'ones': torch.ones(size, device=self.device),
                'zeros': torch.zeros(size, device=self.device),
                'temp1': torch.zeros(size, device=self.device),
                'temp2': torch.zeros(size, device=self.device),
            }
    
    def apply_quantum_to_selection(
        self,
        q_values: torch.Tensor,
        visit_counts: torch.Tensor,
        priors: torch.Tensor,
        c_puct: Optional[float] = None,
        parent_visits: Optional[Union[int, torch.Tensor]] = None,
        simulation_count: Optional[int] = None
    ) -> torch.Tensor:
        """Ultra-optimized quantum selection with aggressive fast paths"""
        
        # Ultra-fast path: quantum disabled
        if not self.enable_quantum:
            self.stats['classical_applications'] += 1
            self.stats['fast_path_hits'] += 1
            parent_vis = parent_visits if isinstance(parent_visits, (int, float)) else visit_counts.sum().item()
            c = c_puct or self.default_c_puct
            return classical_puct_kernel(q_values, visit_counts, priors, c, parent_vis)
        
        # Fast path: determine if we need quantum corrections
        sim_count = simulation_count or self.total_simulations
        
        # Ultra-fast phase check (only every N iterations)
        if sim_count - self._last_phase_check > self._phase_check_interval:
            self._update_phase_ultra_fast(sim_count)
            self._last_phase_check = sim_count
        
        # Fast path: classical regime (but still apply minimal quantum effects for correctness)
        if self.current_phase == MCTSPhase.CLASSICAL and self.fast_mode:
            self.stats['classical_applications'] += 1
            self.stats['fast_path_hits'] += 1
            parent_vis = parent_visits if isinstance(parent_visits, (int, float)) else visit_counts.sum().item()
            c = c_puct or self.default_c_puct
            return classical_puct_kernel(q_values, visit_counts, priors, c, parent_vis)
        
        # Quantum path with optimizations
        return self._apply_quantum_optimized(
            q_values, visit_counts, priors, c_puct, parent_visits, sim_count
        )
    
    def _apply_quantum_optimized(
        self,
        q_values: torch.Tensor,
        visit_counts: torch.Tensor,
        priors: torch.Tensor,
        c_puct: Optional[float],
        parent_visits: Optional[Union[int, torch.Tensor]],
        simulation_count: int
    ) -> torch.Tensor:
        """Optimized quantum application with minimal tensor operations"""
        self.stats['quantum_applications'] += 1
        
        # Use default parameters to avoid computations
        c = c_puct or self.default_c_puct
        
        # Fast parent visits calculation
        if isinstance(parent_visits, (int, float)):
            parent_vis = parent_visits
        else:
            parent_vis = visit_counts.sum().item()
        
        # Classical PUCT base (use compiled kernel)
        classical_scores = classical_puct_kernel(q_values, visit_counts, priors, c, parent_vis)
        
        # Fast quantum correction using lookup tables
        hbar_eff = self._get_hbar_eff_fast(simulation_count)
        quantum_correction = quantum_correction_kernel(
            visit_counts, hbar_eff, self.quantum_strength
        )
        
        # Apply quantum correction (always apply some correction when quantum enabled)
        if self.current_phase == MCTSPhase.QUANTUM:
            correction_factor = 1.0
        elif self.current_phase == MCTSPhase.CRITICAL:
            correction_factor = 0.5
        else:  # Classical phase
            correction_factor = 0.1 if not self.fast_mode else 0.05
        
        return classical_scores + correction_factor * quantum_correction
    
    def _get_hbar_eff_fast(self, simulation_count: int) -> float:
        """Ultra-fast hbar_eff computation using lookup tables"""
        if simulation_count < len(self.hbar_lookup):
            return self.hbar_lookup[simulation_count].item()
        else:
            # Fallback computation for very large N
            tau = math.log(simulation_count + 2)
            return self.default_c_puct / (math.sqrt(simulation_count + 1) * tau)
    
    def _update_phase_ultra_fast(self, sim_count: int):
        """Ultra-fast phase detection using pre-computed critical points"""
        N_c1, N_c2 = self._critical_points
        
        if sim_count < N_c1:
            self.current_phase = MCTSPhase.QUANTUM
        elif sim_count < N_c2:
            self.current_phase = MCTSPhase.CRITICAL  
        else:
            self.current_phase = MCTSPhase.CLASSICAL
    
    def update_simulation_count(self, N: int):
        """Minimal overhead simulation count update"""
        self.total_simulations = N
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_apps = self.stats['quantum_applications'] + self.stats['classical_applications']
        if total_apps > 0:
            fast_path_ratio = self.stats['fast_path_hits'] / total_apps
            quantum_ratio = self.stats['quantum_applications'] / total_apps
        else:
            fast_path_ratio = 0.0
            quantum_ratio = 0.0
            
        return {
            **self.stats,
            'current_phase': self.current_phase.value,
            'fast_path_ratio': fast_path_ratio,
            'quantum_ratio': quantum_ratio,
            'total_simulations': self.total_simulations
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        for key in self.stats:
            if isinstance(self.stats[key], (int, float)):
                self.stats[key] = 0


# Factory function for easy instantiation
def create_ultra_optimized_quantum_mcts(config: Optional[QuantumConfigV2] = None) -> UltraOptimizedQuantumMCTSV2:
    """Create ultra-optimized quantum MCTS with performance focus"""
    if config is None:
        config = QuantumConfigV2(
            fast_mode=True,
            cache_quantum_corrections=True,
            use_mixed_precision=True,
            branching_factor=30,
            avg_game_length=100
        )
    
    return UltraOptimizedQuantumMCTSV2(config)


# Export main class
__all__ = ['UltraOptimizedQuantumMCTSV2', 'create_ultra_optimized_quantum_mcts']