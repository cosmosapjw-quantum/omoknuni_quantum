"""
Quantum Features for MCTS - Version 2.0 Optimized Implementation
===============================================================

This module implements an optimized v2.0 quantum-enhanced MCTS with:
- Aggressive vectorization and tensorization
- Pre-computed lookup tables like v1.0
- Batch-aware operations throughout
- Minimal tensor creation overhead
- JIT compilation support
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union, List
from dataclasses import dataclass
import logging

from .quantum_features_v2 import (
    QuantumConfigV2, MCTSPhase, DiscreteTimeEvolution,
    PhaseDetector, OptimalParameters
)

logger = logging.getLogger(__name__)


class OptimizedQuantumMCTSV2:
    """Heavily optimized v2.0 quantum MCTS implementation"""
    
    def __init__(self, config: QuantumConfigV2):
        self.config = config
        self.device = torch.device(config.device)
        
        # Components
        self.time_evolution = DiscreteTimeEvolution(config)
        self.phase_detector = PhaseDetector(config)
        
        # Pre-compute critical parameters
        self._precompute_parameters()
        
        # Pre-allocate common tensors
        self._preallocate_tensors()
        
        # State tracking
        self.current_phase = MCTSPhase.QUANTUM
        self.total_simulations = 0
        self.last_update_N = 0
        
        # Pre-compute phase configs
        self.phase_configs = {
            MCTSPhase.QUANTUM: self.phase_detector.get_phase_config(MCTSPhase.QUANTUM),
            MCTSPhase.CRITICAL: self.phase_detector.get_phase_config(MCTSPhase.CRITICAL),
            MCTSPhase.CLASSICAL: self.phase_detector.get_phase_config(MCTSPhase.CLASSICAL)
        }
        self.current_phase_config = self.phase_configs[self.current_phase]
        
        # Statistics
        self.stats = {
            'quantum_applications': 0,
            'phase_transitions': 0,
            'batch_calls': 0,
            'tensor_creations': 0
        }
    
    def _precompute_parameters(self):
        """Pre-compute lookup tables for fast access"""
        # Pre-compute information time values
        max_N = 100000
        N_values = torch.arange(0, max_N, device=self.device)
        
        # Information time lookup
        self.tau_table = torch.log(N_values.float() + 2)
        
        # Temperature lookup (for annealing mode)
        if self.config.temperature_mode == 'annealing':
            self.temperature_table = self.config.initial_temperature / (self.tau_table + 1e-8)
        else:
            self.temperature_table = torch.full((max_N,), self.config.initial_temperature, device=self.device)
        
        # Pre-compute hbar_eff factors (without c_puct which varies)
        self.hbar_factors = (N_values.float() + 2) / (torch.sqrt(N_values.float() + 1) * self.tau_table)
        
        # Pre-compute phase kick probabilities
        self.phase_kick_table = torch.tensor(
            [OptimalParameters.phase_kick_schedule(n) for n in range(10000)],
            device=self.device
        )
        
        # Pre-compute power-law decay factors for decoherence
        visit_range = torch.arange(1, 1000, device=self.device).float()
        self.decoherence_table = {}
        for gamma in [0.1, 0.2, 0.3, 0.5, 1.0]:  # Common gamma values
            self.decoherence_table[gamma] = visit_range ** (-gamma)
    
    def _preallocate_tensors(self):
        """Pre-allocate commonly used tensors"""
        # Pre-allocate for different batch sizes
        self.batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 3072]
        self.preallocated = {}
        
        for bs in self.batch_sizes:
            self.preallocated[bs] = {
                'ones': torch.ones(bs, device=self.device),
                'zeros': torch.zeros(bs, device=self.device),
                'rand_uniform': None,  # Will allocate on demand
                'rand_normal': None,   # Will allocate on demand
            }
        
        # Single value tensors (avoid repeated creation)
        self.scalar_tensors = {
            0: torch.tensor(0.0, device=self.device),
            1: torch.tensor(1.0, device=self.device),
            2: torch.tensor(2.0, device=self.device),
            'pi': torch.tensor(np.pi, device=self.device),
            '2pi': torch.tensor(2 * np.pi, device=self.device),
        }
    
    def _get_preallocated(self, size: int, tensor_type: str) -> torch.Tensor:
        """Get pre-allocated tensor or create if needed"""
        if size in self.preallocated:
            if tensor_type in ['ones', 'zeros']:
                return self.preallocated[size][tensor_type]
            elif tensor_type == 'rand_uniform':
                if self.preallocated[size]['rand_uniform'] is None:
                    self.preallocated[size]['rand_uniform'] = torch.empty(size, device=self.device)
                return self.preallocated[size]['rand_uniform'].uniform_()
            elif tensor_type == 'rand_normal':
                if self.preallocated[size]['rand_normal'] is None:
                    self.preallocated[size]['rand_normal'] = torch.empty(size, device=self.device)
                return self.preallocated[size]['rand_normal'].normal_()
        
        # Fallback for non-standard sizes
        self.stats['tensor_creations'] += 1
        if tensor_type == 'ones':
            return torch.ones(size, device=self.device)
        elif tensor_type == 'zeros':
            return torch.zeros(size, device=self.device)
        elif tensor_type == 'rand_uniform':
            return torch.rand(size, device=self.device)
        elif tensor_type == 'rand_normal':
            return torch.randn(size, device=self.device)
    
    @torch.jit.export
    def apply_quantum_to_selection_batch(
        self,
        q_values: torch.Tensor,
        visit_counts: torch.Tensor,
        priors: torch.Tensor,
        c_puct: torch.Tensor,  # Now a tensor for batch support
        parent_visits: torch.Tensor,
        simulation_counts: torch.Tensor  # Batch of simulation counts
    ) -> torch.Tensor:
        """Fully vectorized quantum selection for batches
        
        All inputs should be tensors with batch dimension.
        This avoids any scalar->tensor conversions.
        """
        batch_size = q_values.shape[0]
        self.stats['batch_calls'] += 1
        
        # Fast path for disabled quantum
        if not self.config.enable_quantum:
            sqrt_parent = torch.sqrt(torch.log(parent_visits + 1))
            visit_factor = torch.sqrt(visit_counts + 1)
            exploration = c_puct.unsqueeze(-1) * priors * sqrt_parent.unsqueeze(-1) / visit_factor
            return q_values + exploration
        
        # Vectorized phase detection (approximate for batch)
        avg_sim_count = simulation_counts.float().mean().item()
        phase_idx = 0 if avg_sim_count < 1000 else (1 if avg_sim_count < 10000 else 2)
        phase_config = list(self.phase_configs.values())[phase_idx]
        
        # Base computations (fully vectorized)
        sqrt_parent = torch.sqrt(torch.log(parent_visits + 1))
        visit_factor = torch.sqrt(visit_counts + 1)
        
        # Classical PUCT (vectorized with broadcasting)
        exploration = c_puct.unsqueeze(-1) * priors * sqrt_parent.unsqueeze(-1) / visit_factor
        exploration = exploration * phase_config['prior_trust']
        
        # Vectorized hbar_eff computation
        sim_indices = torch.clamp(simulation_counts.long(), 0, len(self.hbar_factors) - 1)
        hbar_factors = self.hbar_factors[sim_indices]
        hbar_eff = c_puct * hbar_factors * phase_config['quantum_strength']
        
        # Quantum bonus (fully vectorized)
        quantum_bonus = hbar_eff.unsqueeze(-1) / visit_factor
        
        # Vectorized interference for quantum/critical phases
        if phase_idx < 2:  # Not classical
            low_visit_mask = visit_counts < 10
            
            if low_visit_mask.any():
                # Vectorized phase kicks
                kick_prob = phase_config['interference_strength']
                random_phases = torch.rand_like(q_values) * self.scalar_tensors['2pi']
                phase_kicks = kick_prob * torch.sin(random_phases)
                quantum_bonus = quantum_bonus + torch.where(low_visit_mask, phase_kicks, self.scalar_tensors[0])
        
        return q_values + exploration + quantum_bonus
    
    def apply_quantum_to_selection(
        self,
        q_values: torch.Tensor,
        visit_counts: torch.Tensor,
        priors: torch.Tensor,
        c_puct: Optional[float] = None,
        parent_visits: Optional[Union[int, torch.Tensor]] = None,
        simulation_count: Optional[int] = None
    ) -> torch.Tensor:
        """Standard interface with automatic vectorization"""
        
        # Convert scalars to tensors efficiently
        batch_size = q_values.shape[0] if q_values.dim() > 1 else 1
        is_batched = q_values.dim() > 1
        
        # Handle c_puct
        if c_puct is None:
            c_puct = self.config.c_puct or np.sqrt(2 * np.log(self.config.branching_factor))
        
        if not isinstance(c_puct, torch.Tensor):
            c_puct_tensor = torch.full((batch_size,), c_puct, device=self.device)
        else:
            c_puct_tensor = c_puct
        
        # Handle parent_visits efficiently
        if parent_visits is None:
            parent_visits_tensor = visit_counts.sum(dim=-1) if is_batched else visit_counts.sum().unsqueeze(0)
        elif isinstance(parent_visits, int):
            parent_visits_tensor = torch.full((batch_size,), parent_visits, device=self.device)
        else:
            parent_visits_tensor = parent_visits
        
        # Handle simulation_count
        if simulation_count is None:
            simulation_count = self.total_simulations
        sim_count_tensor = torch.full((batch_size,), simulation_count, device=self.device)
        
        # Ensure proper shapes
        if not is_batched:
            q_values = q_values.unsqueeze(0)
            visit_counts = visit_counts.unsqueeze(0)
            priors = priors.unsqueeze(0)
        
        # Call vectorized implementation
        result = self.apply_quantum_to_selection_batch(
            q_values, visit_counts, priors,
            c_puct_tensor, parent_visits_tensor, sim_count_tensor
        )
        
        # Remove batch dimension if needed
        if not is_batched:
            result = result.squeeze(0)
        
        return result
    
    def update_simulation_count(self, N: int):
        """Optimized update - only when phase changes"""
        if N == self.total_simulations:
            return
        
        old_phase = self.current_phase
        self.total_simulations = N
        
        # Only check phase periodically
        if self.config.enable_phase_adaptation and N - self.last_update_N >= 100:
            self._update_phase_fast(N)
            self.last_update_N = N
            
            # Update cached config if phase changed
            if self.current_phase != old_phase:
                self.current_phase_config = self.phase_configs[self.current_phase]
                self.stats['phase_transitions'] += 1
    
    def _update_phase_fast(self, N: int):
        """Fast phase detection without recomputation"""
        # Use pre-computed critical points
        if not hasattr(self, '_critical_points'):
            b = self.config.branching_factor or 30
            c = self.config.c_puct or np.sqrt(2 * np.log(b))
            self._critical_points = self.phase_detector.compute_critical_points(
                b, c, self.config.use_neural_prior
            )
        
        N_c1, N_c2 = self._critical_points
        
        if N < N_c1:
            self.current_phase = MCTSPhase.QUANTUM
        elif N < N_c2:
            self.current_phase = MCTSPhase.CRITICAL
        else:
            self.current_phase = MCTSPhase.CLASSICAL
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.stats,
            'current_phase': self.current_phase.value,
            'total_simulations': self.total_simulations,
        }


# JIT compile critical functions if available
try:
    OptimizedQuantumMCTSV2.apply_quantum_to_selection_batch = torch.jit.script(
        OptimizedQuantumMCTSV2.apply_quantum_to_selection_batch
    )
except:
    pass  # JIT not available or failed