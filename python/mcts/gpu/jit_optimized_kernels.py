"""
JIT-Optimized MCTS Kernels
===========================

Fast PyTorch JIT Script implementations that avoid CUDA compilation overhead
while still providing significant performance improvements.

JIT compilation is:
- Much faster to compile (seconds vs minutes)
- Still provides GPU acceleration 
- Easier to debug and maintain
- Cross-platform compatible
"""

import torch
import torch.jit
from typing import Tuple, Optional
import math

@torch.jit.script
def jit_batched_ucb_selection_classical(
    node_indices: torch.Tensor,
    row_ptr: torch.Tensor,
    col_indices: torch.Tensor,
    edge_actions: torch.Tensor,
    edge_priors: torch.Tensor,
    visit_counts: torch.Tensor,
    value_sums: torch.Tensor,
    c_puct: float = 1.414
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    JIT-compiled classical UCB selection - fast compilation, good performance
    
    This function is compiled once and cached, providing near-CUDA performance
    without the compilation overhead.
    """
    batch_size = node_indices.shape[0]
    device = node_indices.device
    
    # Pre-allocate outputs
    selected_actions = torch.full((batch_size,), -1, dtype=torch.int32, device=device)
    selected_scores = torch.zeros(batch_size, device=device)
    
    # Get parent visits with minimum of 1
    parent_visits = visit_counts[node_indices].float()
    parent_visits = torch.maximum(parent_visits, torch.ones_like(parent_visits))
    
    # Process each node
    for i in range(batch_size):
        node_idx = node_indices[i]
        
        # Check bounds
        if node_idx >= row_ptr.shape[0] - 1:
            continue
            
        start_idx = row_ptr[node_idx]
        end_idx = row_ptr[node_idx + 1]
        
        # Skip nodes with no children
        if start_idx == end_idx:
            continue
        
        # Get child data (vectorized)
        child_nodes = col_indices[start_idx:end_idx]
        child_visits = visit_counts[child_nodes].float()
        child_values = value_sums[child_nodes]
        child_priors = edge_priors[start_idx:end_idx]
        
        # Compute Q-values (vectorized)
        q_values = torch.where(
            child_visits > 0.0,
            child_values / child_visits,
            torch.zeros_like(child_values)
        )
        
        # UCB computation (vectorized)
        sqrt_parent = torch.sqrt(parent_visits[i])
        exploration = c_puct * child_priors * sqrt_parent / (1.0 + child_visits)
        ucb_scores = q_values + exploration
        
        # Select best action (vectorized)
        best_idx = torch.argmax(ucb_scores)
        selected_actions[i] = edge_actions[start_idx + best_idx]
        selected_scores[i] = ucb_scores[best_idx]
    
    return selected_actions, selected_scores

@torch.jit.script
def jit_batched_ucb_selection_quantum(
    node_indices: torch.Tensor,
    row_ptr: torch.Tensor,
    col_indices: torch.Tensor,
    edge_actions: torch.Tensor,
    edge_priors: torch.Tensor,
    visit_counts: torch.Tensor,
    value_sums: torch.Tensor,
    c_puct: float = 1.414,
    # Quantum parameters
    quantum_phases: Optional[torch.Tensor] = None,
    uncertainty_table: Optional[torch.Tensor] = None,
    hbar_eff: float = 0.05,
    phase_kick_strength: float = 0.1,
    interference_alpha: float = 0.05
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    JIT-compiled quantum UCB selection implementing the same algorithm as CUDA
    
    This provides the exact same quantum algorithm as the CUDA version but with
    fast JIT compilation instead of slow CUDA compilation.
    """
    batch_size = node_indices.shape[0]
    device = node_indices.device
    
    # Pre-allocate outputs
    selected_actions = torch.full((batch_size,), -1, dtype=torch.int32, device=device)
    selected_scores = torch.zeros(batch_size, device=device)
    
    # Get parent visits with minimum of 1
    parent_visits = visit_counts[node_indices].float()
    parent_visits = torch.maximum(parent_visits, torch.ones_like(parent_visits))
    
    # Process each node
    for i in range(batch_size):
        node_idx = node_indices[i]
        
        # Check bounds
        if node_idx >= row_ptr.shape[0] - 1:
            continue
            
        start_idx = row_ptr[node_idx]
        end_idx = row_ptr[node_idx + 1]
        
        # Skip nodes with no children
        if start_idx == end_idx:
            continue
        
        # Get child data (vectorized)
        child_nodes = col_indices[start_idx:end_idx]
        child_visits = visit_counts[child_nodes].float()
        child_values = value_sums[child_nodes]
        child_priors = edge_priors[start_idx:end_idx]
        
        # Compute Q-values (vectorized)
        q_values = torch.where(
            child_visits > 0.0,
            child_values / child_visits,
            torch.zeros_like(child_values)
        )
        
        # Standard UCB computation (vectorized)
        sqrt_parent = torch.sqrt(parent_visits[i])
        exploration = c_puct * child_priors * sqrt_parent / (1.0 + child_visits)
        ucb_scores = q_values + exploration
        
        # Apply quantum corrections (same as CUDA implementation)
        if uncertainty_table is not None and uncertainty_table.shape[0] > 0:
            # 1. Quantum uncertainty boost for low-visit nodes
            max_table_size = uncertainty_table.shape[0]
            for j in range(child_visits.shape[0]):
                visit_idx = min(int(child_visits[j]), max_table_size - 1)
                quantum_boost = uncertainty_table[visit_idx]
                ucb_scores[j] = ucb_scores[j] + quantum_boost
        
        if quantum_phases is not None and quantum_phases.shape[0] > 0:
            # 2. Phase kick for exploration enhancement
            for j in range(child_visits.shape[0]):
                if child_visits[j] < 10.0:  # Apply to low-visit nodes
                    # Generate pseudo-random phase kick
                    seed = (node_idx * 1337 + int(parent_visits[i])) % 10000
                    rand_val = (seed * 1664525 + 1013904223) % 4294967296 / 4294967296.0
                    phase_kick = phase_kick_strength * torch.sin(torch.tensor(2.0 * math.pi * rand_val))
                    ucb_scores[j] = ucb_scores[j] + phase_kick
            
            # 3. Interference based on pre-computed phases
            if start_idx + child_visits.shape[0] <= quantum_phases.shape[0]:
                for j in range(child_visits.shape[0]):
                    phase = quantum_phases[start_idx + j]
                    interference = interference_alpha * torch.cos(phase)
                    ucb_scores[j] = ucb_scores[j] + interference
        
        # Select best action (vectorized)
        best_idx = torch.argmax(ucb_scores)
        selected_actions[i] = edge_actions[start_idx + best_idx]
        selected_scores[i] = ucb_scores[best_idx]
    
    return selected_actions, selected_scores

@torch.jit.script
def jit_parallel_backup(
    paths: torch.Tensor,
    leaf_values: torch.Tensor,
    path_lengths: torch.Tensor,
    value_sums: torch.Tensor,
    visit_counts: torch.Tensor
) -> torch.Tensor:
    """
    JIT-compiled parallel backup operation
    
    Much faster compilation than CUDA while maintaining good performance.
    """
    batch_size = paths.shape[0]
    max_depth = paths.shape[1]
    
    # Process each path
    for i in range(batch_size):
        value = leaf_values[i]
        path_length = path_lengths[i]
        
        # Traverse path from leaf to root
        for depth in range(min(path_length + 1, max_depth)):
            node_idx = paths[i, depth]
            if node_idx < 0 or node_idx >= value_sums.shape[0]:
                break
            
            # Update statistics
            value_sums[node_idx] = value_sums[node_idx] + value
            visit_counts[node_idx] = visit_counts[node_idx] + 1
            
            # Negate value for opponent's perspective
            value = -value
    
    return value_sums

class JITOptimizedKernels:
    """
    Manager for JIT-optimized MCTS kernels
    
    Provides the same interface as CUDA kernels but with much faster compilation.
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.stats = {
            'classical_calls': 0,
            'quantum_calls': 0,
            'backup_calls': 0
        }
        
        # Pre-compile JIT functions on first import
        self._warmup_jit_functions()
    
    def _warmup_jit_functions(self):
        """Warm up JIT functions with dummy data to trigger compilation"""
        try:
            # Create minimal test data
            device = self.device
            batch_size = 2
            
            node_indices = torch.arange(batch_size, device=device)
            row_ptr = torch.tensor([0, 2, 4], device=device)
            col_indices = torch.tensor([0, 1, 2, 3], device=device)
            edge_actions = torch.tensor([0, 1, 0, 1], device=device)
            edge_priors = torch.tensor([0.5, 0.5, 0.5, 0.5], device=device)
            visit_counts = torch.ones(4, device=device)
            value_sums = torch.rand(4, device=device)
            
            # Trigger JIT compilation
            _ = jit_batched_ucb_selection_classical(
                node_indices, row_ptr, col_indices, edge_actions, edge_priors,
                visit_counts, value_sums
            )
            
            # Test with quantum parameters
            quantum_phases = torch.rand(4, device=device) * 2 * math.pi
            uncertainty_table = torch.linspace(0.1, 0.01, 100, device=device)
            
            _ = jit_batched_ucb_selection_quantum(
                node_indices, row_ptr, col_indices, edge_actions, edge_priors,
                visit_counts, value_sums, quantum_phases=quantum_phases,
                uncertainty_table=uncertainty_table
            )
            
        except Exception:
            # JIT warmup failed, but that's OK - will compile on first use
            pass
    
    def batch_ucb_selection_classical(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Classical UCB selection using JIT compilation"""
        self.stats['classical_calls'] += 1
        return jit_batched_ucb_selection_classical(*args, **kwargs)
    
    def batch_ucb_selection_quantum(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantum UCB selection using JIT compilation"""
        self.stats['quantum_calls'] += 1
        return jit_batched_ucb_selection_quantum(*args, **kwargs)
    
    def parallel_backup(self, *args, **kwargs) -> torch.Tensor:
        """Parallel backup using JIT compilation"""
        self.stats['backup_calls'] += 1
        return jit_parallel_backup(*args, **kwargs)
    
    def get_stats(self):
        """Get usage statistics"""
        return self.stats.copy()

# Global instance for easy access
_jit_kernels = None

def get_jit_kernels(device: torch.device) -> JITOptimizedKernels:
    """Get global JIT kernel instance"""
    global _jit_kernels
    if _jit_kernels is None or _jit_kernels.device != device:
        _jit_kernels = JITOptimizedKernels(device)
    return _jit_kernels