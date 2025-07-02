"""
Classical Triton Kernels for Optimized MCTS
==========================================

This module provides Triton-based GPU kernels for classical MCTS computation,
ensuring optimization parity with quantum-inspired MCTS while maintaining
mutability safety that torch.compile might not provide.

Triton kernels offer:
- Python-based kernel development
- Automatic optimization and tuning  
- Better PyTorch integration
- Safe mutability handling
- Performance equivalent to CUDA kernels
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@triton.jit
def classical_ucb_selection_kernel(
    # Input tensors
    q_values_ptr,
    visit_counts_ptr,
    parent_visits_ptr,
    priors_ptr,
    
    # Lookup tables (classical optimization)
    sqrt_table_ptr,
    exploration_table_ptr,
    
    # CSR structure
    row_ptr_ptr,
    col_indices_ptr,
    
    # Output tensors
    selected_actions_ptr,
    ucb_scores_ptr,
    
    # Constants
    c_puct: tl.constexpr,
    max_table_size: tl.constexpr,
    num_nodes: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for classical UCB selection with lookup table optimization
    
    This provides equivalent optimization to quantum kernels while using
    classical UCB computation with precomputed lookup tables.
    """
    node_id = tl.program_id(0)
    
    if node_id >= num_nodes:
        return
    
    # Get children range for this node using CSR structure
    start_idx = tl.load(row_ptr_ptr + node_id)
    end_idx = tl.load(row_ptr_ptr + node_id + 1)
    num_children = end_idx - start_idx
    
    if num_children == 0:
        tl.store(selected_actions_ptr + node_id, -1)
        tl.store(ucb_scores_ptr + node_id, 0.0)
        return
    
    # Get parent visit count and compute sqrt using lookup table
    parent_visits = tl.load(parent_visits_ptr + node_id)
    
    # Use lookup table for sqrt computation if within range
    sqrt_parent = 0.0
    if parent_visits < max_table_size:
        sqrt_parent = tl.load(sqrt_table_ptr + parent_visits)
    else:
        # Fallback for out-of-range values
        sqrt_parent = tl.sqrt(parent_visits.to(tl.float32) + 1.0)
    
    # Find best UCB action using optimized computation
    best_score = -1e10
    best_action = -1
    
    # Process children in blocks for efficiency
    for child_offset in range(0, num_children, BLOCK_SIZE):
        child_mask = child_offset + tl.arange(0, BLOCK_SIZE) < num_children
        child_indices = start_idx + child_offset + tl.arange(0, BLOCK_SIZE)
        
        # Load child data
        child_nodes = tl.load(col_indices_ptr + child_indices, mask=child_mask, other=0)
        child_visits = tl.load(visit_counts_ptr + child_nodes, mask=child_mask, other=0)
        child_priors = tl.load(priors_ptr + child_indices, mask=child_mask, other=0.0)
        
        # Compute Q-values
        q_vals = tl.where(
            child_visits > 0,
            tl.load(q_values_ptr + child_nodes, mask=child_mask, other=0.0) / child_visits.to(tl.float32),
            0.0
        )
        
        # Compute exploration term using lookup table optimization
        exploration_factors = tl.zeros_like(child_visits, dtype=tl.float32)
        
        # Use lookup table for in-range values
        in_range_mask = child_visits < max_table_size
        if tl.sum(in_range_mask) > 0:
            table_indices = tl.where(in_range_mask, child_visits, 0)
            exploration_factors = tl.where(
                in_range_mask,
                tl.load(exploration_table_ptr + table_indices, mask=in_range_mask, other=0.0),
                exploration_factors
            )
        
        # Direct computation for out-of-range values
        out_range_mask = child_visits >= max_table_size
        if tl.sum(out_range_mask) > 0:
            direct_factors = c_puct / (1.0 + child_visits.to(tl.float32))
            exploration_factors = tl.where(out_range_mask, direct_factors, exploration_factors)
        
        # Compute classical UCB scores
        # UCB = Q + c_puct * prior * sqrt(parent_visits) / (1 + child_visits)
        # Using lookup tables: UCB = Q + prior * sqrt_parent * exploration_factor
        exploration = child_priors * sqrt_parent * exploration_factors
        ucb_scores = q_vals + exploration
        
        # Find local maximum
        for i in range(BLOCK_SIZE):
            if child_mask[i] and ucb_scores[i] > best_score:
                best_score = ucb_scores[i]
                best_action = child_offset + i
    
    # Store results
    tl.store(selected_actions_ptr + node_id, best_action)
    tl.store(ucb_scores_ptr + node_id, best_score)


@triton.jit
def classical_batch_ucb_kernel(
    # Input tensors - batch format
    batch_q_values_ptr,
    batch_child_visits_ptr,
    batch_parent_visits_ptr,
    batch_priors_ptr,
    
    # Lookup tables
    sqrt_table_ptr,
    exploration_table_ptr,
    
    # Output tensors
    batch_ucb_scores_ptr,
    
    # Constants
    batch_size: tl.constexpr,
    num_actions: tl.constexpr,
    max_table_size: tl.constexpr,
    c_puct: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for batch UCB computation with classical optimization
    
    Processes multiple nodes in parallel using vectorized operations
    and lookup table optimization.
    """
    batch_id = tl.program_id(0)
    action_block = tl.program_id(1)
    
    if batch_id >= batch_size:
        return
    
    # Calculate action indices for this block
    action_start = action_block * BLOCK_SIZE
    action_end = min(action_start + BLOCK_SIZE, num_actions)
    action_mask = action_start + tl.arange(0, BLOCK_SIZE) < action_end
    
    if action_start >= num_actions:
        return
    
    # Load parent visit count and compute sqrt using lookup table
    parent_visits = tl.load(batch_parent_visits_ptr + batch_id)
    
    sqrt_parent = 0.0
    if parent_visits < max_table_size:
        sqrt_parent = tl.load(sqrt_table_ptr + parent_visits)
    else:
        sqrt_parent = tl.sqrt(parent_visits.to(tl.float32) + 1.0)
    
    # Process actions in this block
    action_indices = action_start + tl.arange(0, BLOCK_SIZE)
    batch_action_indices = batch_id * num_actions + action_indices
    
    # Load child data for this batch and action block
    child_visits = tl.load(batch_child_visits_ptr + batch_action_indices, mask=action_mask, other=0)
    child_priors = tl.load(batch_priors_ptr + batch_action_indices, mask=action_mask, other=0.0)
    
    # Compute Q-values
    q_values = tl.load(batch_q_values_ptr + batch_action_indices, mask=action_mask, other=0.0)
    
    # Handle division by zero for unvisited nodes
    q_vals = tl.where(
        child_visits > 0,
        q_values / child_visits.to(tl.float32),
        0.0
    )
    
    # Compute exploration factors using lookup table
    exploration_factors = tl.zeros_like(child_visits, dtype=tl.float32)
    
    # Use lookup table for in-range values
    in_range_mask = (child_visits < max_table_size) & action_mask
    if tl.sum(in_range_mask) > 0:
        table_indices = tl.where(in_range_mask, child_visits, 0)
        exploration_factors = tl.where(
            in_range_mask,
            tl.load(exploration_table_ptr + table_indices, mask=in_range_mask, other=0.0),
            exploration_factors
        )
    
    # Direct computation for out-of-range values
    out_range_mask = (child_visits >= max_table_size) & action_mask
    if tl.sum(out_range_mask) > 0:
        direct_factors = c_puct / (1.0 + child_visits.to(tl.float32))
        exploration_factors = tl.where(out_range_mask, direct_factors, exploration_factors)
    
    # Compute classical UCB scores
    exploration = child_priors * sqrt_parent * exploration_factors
    ucb_scores = q_vals + exploration
    
    # Mask invalid actions
    ucb_scores = tl.where(action_mask, ucb_scores, -1e10)
    
    # Store results
    tl.store(batch_ucb_scores_ptr + batch_action_indices, ucb_scores, mask=action_mask)


@triton.jit
def classical_vectorized_backup_kernel(
    # Input tensors
    paths_ptr,
    path_lengths_ptr,
    values_ptr,
    
    # Output tensors  
    visit_counts_ptr,
    value_sums_ptr,
    
    # Constants
    batch_size: tl.constexpr,
    max_path_length: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for vectorized backup operation
    
    This provides the same backup optimization that quantum MCTS gets,
    ensuring classical MCTS has equivalent performance.
    """
    batch_id = tl.program_id(0)
    
    if batch_id >= batch_size:
        return
    
    # Load path length for this batch element
    path_length = tl.load(path_lengths_ptr + batch_id)
    value = tl.load(values_ptr + batch_id)
    
    if path_length <= 0:
        return
    
    # Process path nodes in blocks
    for depth_start in range(0, path_length, BLOCK_SIZE):
        depth_end = min(depth_start + BLOCK_SIZE, path_length)
        depth_mask = depth_start + tl.arange(0, BLOCK_SIZE) < depth_end
        
        if depth_start >= path_length:
            break
        
        # Load node indices for this depth block
        depth_indices = depth_start + tl.arange(0, BLOCK_SIZE)
        path_indices = batch_id * max_path_length + depth_indices
        node_indices = tl.load(paths_ptr + path_indices, mask=depth_mask, other=-1)
        
        # Update valid nodes
        valid_mask = (node_indices >= 0) & depth_mask
        
        if tl.sum(valid_mask) > 0:
            # Atomic updates for visit counts and value sums
            for i in range(BLOCK_SIZE):
                if valid_mask[i]:
                    node_idx = node_indices[i]
                    # Atomic increment visit count
                    tl.atomic_add(visit_counts_ptr + node_idx, 1)
                    # Atomic add value sum
                    tl.atomic_add(value_sums_ptr + node_idx, value)


class ClassicalTritonKernels:
    """
    Triton-based GPU kernels for classical MCTS computation
    
    This class provides the same kernel optimization that quantum MCTS receives,
    ensuring fair performance comparison while maintaining mutability safety.
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.use_triton = device.type == 'cuda'
        
        # Performance statistics
        self.stats = {
            'ucb_selections': 0,
            'batch_ucb_calls': 0,
            'backup_operations': 0,
            'kernel_compile_time': 0.0
        }
        
        logger.debug(f"Classical Triton kernels initialized on {device}")
        
    def batch_ucb_selection_classical(
        self,
        node_indices: torch.Tensor,
        q_values: torch.Tensor,
        visit_counts: torch.Tensor,
        parent_visits: torch.Tensor,
        priors: torch.Tensor,
        sqrt_table: torch.Tensor,
        exploration_table: torch.Tensor,
        row_ptr: torch.Tensor,
        col_indices: torch.Tensor,
        c_puct: float = 1.4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classical UCB selection using optimized Triton kernel
        
        Args:
            node_indices: Nodes to select actions for
            q_values: Q-values per node
            visit_counts: Visit counts per node  
            parent_visits: Parent visit counts
            priors: Prior probabilities
            sqrt_table: Precomputed sqrt lookup table
            exploration_table: Precomputed exploration factors
            row_ptr: CSR row pointers
            col_indices: CSR column indices
            c_puct: UCB exploration constant
            
        Returns:
            Tuple of (selected_actions, ucb_scores)
        """
        if not self.use_triton:
            raise RuntimeError("Triton kernels require CUDA device")
            
        batch_size = len(node_indices)
        max_table_size = len(sqrt_table)
        
        # Output tensors
        selected_actions = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        ucb_scores = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        
        # Grid and block configuration
        grid = (batch_size,)
        BLOCK_SIZE = 32
        
        # Launch Triton kernel
        classical_ucb_selection_kernel[grid](
            q_values, visit_counts, parent_visits, priors,
            sqrt_table, exploration_table,
            row_ptr, col_indices,
            selected_actions, ucb_scores,
            c_puct, max_table_size, batch_size, BLOCK_SIZE
        )
        
        self.stats['ucb_selections'] += batch_size
        return selected_actions, ucb_scores
    
    def batch_ucb_computation_classical(
        self,
        batch_q_values: torch.Tensor,
        batch_child_visits: torch.Tensor,
        batch_parent_visits: torch.Tensor,
        batch_priors: torch.Tensor,
        sqrt_table: torch.Tensor,
        exploration_table: torch.Tensor,
        c_puct: float = 1.4
    ) -> torch.Tensor:
        """
        Batch UCB computation using optimized Triton kernel
        
        Args:
            batch_q_values: [batch_size, num_actions] Q-values
            batch_child_visits: [batch_size, num_actions] child visit counts
            batch_parent_visits: [batch_size] parent visit counts
            batch_priors: [batch_size, num_actions] prior probabilities
            sqrt_table: Precomputed sqrt lookup table
            exploration_table: Precomputed exploration factors
            c_puct: UCB exploration constant
            
        Returns:
            [batch_size, num_actions] UCB scores
        """
        if not self.use_triton:
            raise RuntimeError("Triton kernels require CUDA device")
            
        batch_size, num_actions = batch_q_values.shape
        max_table_size = len(sqrt_table)
        
        # Output tensor
        batch_ucb_scores = torch.zeros_like(batch_q_values)
        
        # Grid configuration
        BLOCK_SIZE = 32
        grid = (batch_size, triton.cdiv(num_actions, BLOCK_SIZE))
        
        # Launch Triton kernel
        classical_batch_ucb_kernel[grid](
            batch_q_values, batch_child_visits, batch_parent_visits, batch_priors,
            sqrt_table, exploration_table,
            batch_ucb_scores,
            batch_size, num_actions, max_table_size, c_puct, BLOCK_SIZE
        )
        
        self.stats['batch_ucb_calls'] += 1
        return batch_ucb_scores
    
    def vectorized_backup_classical(
        self,
        paths: torch.Tensor,
        path_lengths: torch.Tensor,
        values: torch.Tensor,
        visit_counts: torch.Tensor,
        value_sums: torch.Tensor
    ):
        """
        Vectorized backup operation using Triton kernel
        
        Args:
            paths: [batch_size, max_path_length] path node indices
            path_lengths: [batch_size] length of each path
            values: [batch_size] values to backup
            visit_counts: [num_nodes] visit counts (modified in-place)
            value_sums: [num_nodes] value sums (modified in-place)
        """
        if not self.use_triton:
            raise RuntimeError("Triton kernels require CUDA device")
            
        batch_size, max_path_length = paths.shape
        
        # Grid configuration
        grid = (batch_size,)
        BLOCK_SIZE = 32
        
        # Launch Triton kernel
        classical_vectorized_backup_kernel[grid](
            paths, path_lengths, values,
            visit_counts, value_sums,
            batch_size, max_path_length, BLOCK_SIZE
        )
        
        self.stats['backup_operations'] += batch_size
    
    def get_stats(self):
        """Get performance statistics"""
        return self.stats.copy()


def get_classical_triton_kernels(device: torch.device) -> ClassicalTritonKernels:
    """Factory function for creating classical Triton kernels"""
    return ClassicalTritonKernels(device)


# Export main classes and functions
__all__ = [
    'ClassicalTritonKernels',
    'get_classical_triton_kernels',
    'classical_ucb_selection_kernel',
    'classical_batch_ucb_kernel',
    'classical_vectorized_backup_kernel'
]