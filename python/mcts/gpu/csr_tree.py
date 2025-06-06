"""CSR (Compressed Sparse Row) Tree Format for GPU-Optimized MCTS

This module implements CSR format storage for tree structures to enable:
- Contiguous memory access patterns for GPU kernels
- Coalesced memory reads in parallel processing
- Better cache utilization vs sparse child storage
- Improved bandwidth efficiency for large trees

The CSR format stores tree children using three arrays:
- row_ptr: Starting index for each node's children  
- col_indices: Child node indices (flattened)
- edge_data: Associated edge data (actions, priors, etc.)
"""

from __future__ import annotations

import torch
from typing import Optional, Tuple, Dict, List, Union, Any
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class CSRTreeConfig:
    """Configuration for CSR tree format"""
    max_nodes: int = 1_000_000
    max_edges: int = 5_000_000  # Estimate: ~5 edges per node on average
    device: str = 'cuda'
    dtype_indices: 'torch.dtype' = None  # Will be set to torch.int32 in __post_init__
    dtype_actions: 'torch.dtype' = None  # Will be set to torch.int16 in __post_init__
    dtype_values: 'torch.dtype' = None   # Will be set to torch.float16 in __post_init__
    
    # Performance tuning
    initial_capacity_factor: float = 0.1  # Start with 10% of max capacity
    growth_factor: float = 1.5
    enable_memory_pooling: bool = True
    
    def __post_init__(self):
        """Set default dtypes after initialization"""
        if self.dtype_indices is None:
            self.dtype_indices = torch.int32
        if self.dtype_actions is None:
            self.dtype_actions = torch.int16
        if self.dtype_values is None:
            self.dtype_values = torch.float16


class CSRTree:
    """GPU-optimized tree storage using Compressed Sparse Row format
    
    This class provides a memory-efficient, GPU-friendly representation
    of tree structures that enables vectorized operations on children.
    
    Memory layout:
    - Node data: Same as OptimizedTensorTree (visits, values, priors, etc.)
    - CSR structure: row_ptr[n+1], col_indices[nnz], edge_data[nnz]
    - Total memory: ~20 bytes/node + ~6 bytes/edge (vs ~100 bytes/node for sparse)
    """
    
    def __init__(self, config: CSRTreeConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Tree size tracking
        self.num_nodes = 0
        self.num_edges = 0
        self.max_nodes = config.max_nodes
        self.max_edges = config.max_edges
        
        # Initialize storage
        self._init_node_storage()
        self._init_csr_storage()
        
        # Performance tracking
        self.stats = {
            'memory_reallocations': 0,
            'batch_operations': 0,
            'cache_hits': 0
        }
        
        # Initialize optimized kernels if available
        try:
            from .csr_gpu_kernels_optimized import get_csr_batch_operations
            self.batch_ops = get_csr_batch_operations(self.device)
            logger.info("Initialized optimized CSR GPU kernels")
        except ImportError:
            self.batch_ops = None
            logger.info("Optimized CSR kernels not available, using fallback")
        
        # Flag for deferred row pointer updates
        self._needs_row_ptr_update = False
        
    def _init_node_storage(self):
        """Initialize node data storage (same as OptimizedTensorTree)"""
        n = self.max_nodes
        device = self.device
        
        # Node statistics
        self.visit_counts = torch.zeros(n, device=device, dtype=torch.int32)
        self.value_sums = torch.zeros(n, device=device, dtype=self.config.dtype_values)
        self.node_priors = torch.zeros(n, device=device, dtype=self.config.dtype_values)
        
        # Tree structure  
        self.parent_indices = torch.full((n,), -1, device=device, dtype=torch.int32)
        self.parent_actions = torch.full((n,), -1, device=device, dtype=self.config.dtype_actions)
        self.flags = torch.zeros(n, device=device, dtype=torch.uint8)
        
        # Quantum features
        self.phases = torch.zeros(n, device=device, dtype=self.config.dtype_values)
        
        # Game state tracking
        self.node_states = {}  # Dict mapping node index to game state
        
        # For compatibility with wave engine - children lookup table
        self.children = torch.full((n, 361), -1, device=device, dtype=torch.int32)  # Max 361 for Go
        
    def _init_csr_storage(self):
        """Initialize CSR format storage"""
        # Start with reduced capacity for memory efficiency
        initial_edges = int(self.max_edges * self.config.initial_capacity_factor)
        
        # CSR row pointers (one per node + 1)
        self.row_ptr = torch.zeros(self.max_nodes + 1, device=self.device, 
                                  dtype=self.config.dtype_indices)
        
        # CSR column indices (child node IDs)
        self.col_indices = torch.zeros(initial_edges, device=self.device,
                                      dtype=self.config.dtype_indices)
        
        # Edge data
        self.edge_actions = torch.zeros(initial_edges, device=self.device,
                                       dtype=self.config.dtype_actions)
        self.edge_priors = torch.zeros(initial_edges, device=self.device,
                                      dtype=self.config.dtype_values)
        
        # Current capacity
        self.edge_capacity = initial_edges
        
    def _grow_edge_storage(self, min_required: int):
        """Grow edge storage when capacity is exceeded"""
        new_capacity = max(min_required, 
                          int(self.edge_capacity * self.config.growth_factor))
        new_capacity = min(new_capacity, self.max_edges)
        
        if new_capacity <= self.edge_capacity:
            raise RuntimeError(f"Cannot grow edge storage beyond {self.max_edges}")
            
        # Allocate new tensors
        new_col_indices = torch.zeros(new_capacity, device=self.device,
                                     dtype=self.config.dtype_indices)
        new_edge_actions = torch.zeros(new_capacity, device=self.device,
                                      dtype=self.config.dtype_actions)
        new_edge_priors = torch.zeros(new_capacity, device=self.device,
                                     dtype=self.config.dtype_values)
        
        # Copy existing data
        new_col_indices[:self.num_edges] = self.col_indices[:self.num_edges]
        new_edge_actions[:self.num_edges] = self.edge_actions[:self.num_edges]
        new_edge_priors[:self.num_edges] = self.edge_priors[:self.num_edges]
        
        # Replace old tensors
        self.col_indices = new_col_indices
        self.edge_actions = new_edge_actions
        self.edge_priors = new_edge_priors
        self.edge_capacity = new_capacity
        
        self.stats['memory_reallocations'] += 1
        
    def add_root(self, prior: float = 1.0, state: Optional[Any] = None) -> int:
        """Add root node and return its index"""
        if self.num_nodes >= self.max_nodes:
            raise RuntimeError(f"Tree full: {self.num_nodes} nodes")
            
        idx = self.num_nodes
        self.num_nodes += 1
        
        # Initialize node data
        self.visit_counts[idx] = 0
        self.value_sums[idx] = 0.0
        self.node_priors[idx] = prior
        self.parent_indices[idx] = -1
        self.parent_actions[idx] = -1
        self.flags[idx] = 0
        self.phases[idx] = 0.0
        
        # Store game state
        if state is not None:
            self.node_states[idx] = state
        
        # Initialize CSR row pointer
        self.row_ptr[idx] = self.num_edges
        
        return idx
        
    def add_child(self, parent_idx: int, action: int, child_prior: float, 
                  child_state: Optional[Any] = None) -> int:
        """Add a child node and update CSR structure"""
        if self.num_nodes >= self.max_nodes:
            raise RuntimeError(f"Tree full: {self.num_nodes} nodes")
        if self.num_edges >= self.edge_capacity:
            self._grow_edge_storage(self.num_edges + 1)
            
        # Create child node
        child_idx = self.num_nodes
        self.num_nodes += 1
        
        # Initialize child node data
        self.visit_counts[child_idx] = 0
        self.value_sums[child_idx] = 0.0
        self.node_priors[child_idx] = child_prior
        self.parent_indices[child_idx] = parent_idx
        self.parent_actions[child_idx] = action
        self.flags[child_idx] = 0
        self.phases[child_idx] = 0.0
        
        # Store game state
        if child_state is not None:
            self.node_states[child_idx] = child_state
        
        # Add edge to CSR structure
        edge_idx = self.num_edges
        self.col_indices[edge_idx] = child_idx
        self.edge_actions[edge_idx] = action
        self.edge_priors[edge_idx] = child_prior
        self.num_edges += 1
        
        # DEFER row pointer updates for batch efficiency
        # We'll need to call _rebuild_row_pointers() periodically
        # For now, just mark that we need an update
        self._needs_row_ptr_update = True
        
        # Update children lookup table for vectorized operations
        # Find first empty slot in parent's children
        for i in range(self.children.shape[1]):
            if self.children[parent_idx, i] == -1:
                self.children[parent_idx, i] = child_idx
                break
        
        return child_idx
        
    def get_children(self, node_idx: int) -> Tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor']:
        """Get children indices, actions, and priors for a node
        
        Returns:
            child_indices: Tensor of child node indices
            child_actions: Tensor of actions leading to children  
            child_priors: Tensor of child prior probabilities
        """
        # Use the children lookup table instead of CSR
        children_slice = self.children[node_idx]
        valid_mask = children_slice >= 0
        valid_children = children_slice[valid_mask]
        
        if len(valid_children) == 0:
            # No children
            empty = torch.empty(0, device=self.device)
            return (empty.to(self.config.dtype_indices),
                   empty.to(self.config.dtype_actions),
                   empty.to(self.config.dtype_values))
        
        # Get actions and priors for valid children
        actions = self.parent_actions[valid_children]
        priors = self.node_priors[valid_children]
        
        return valid_children, actions, priors
        
    def batch_get_children(self, node_indices: torch.Tensor, 
                          max_children: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get children for multiple nodes efficiently (GPU-optimized)
        
        This is the key optimization for GPU performance - processes multiple
        nodes in parallel with coalesced memory access.
        
        Args:
            node_indices: Tensor of node indices to process
            max_children: Maximum number of children to return per node
            
        Returns:
            batch_children: (batch_size, max_children) child indices (-1 for padding)
            batch_actions: (batch_size, max_children) actions
            batch_priors: (batch_size, max_children) priors
        """
        self.stats['batch_operations'] += 1
        
        batch_size = len(node_indices)
        
        # Find maximum children count if not provided
        if max_children is None:
            row_starts = self.row_ptr[node_indices]
            row_ends = self.row_ptr[node_indices + 1]
            max_children = (row_ends - row_starts).max().item()
            
        if max_children == 0:
            empty_shape = (batch_size, 0)
            return (torch.empty(empty_shape, dtype=self.config.dtype_indices, device=self.device),
                   torch.empty(empty_shape, dtype=self.config.dtype_actions, device=self.device),
                   torch.empty(empty_shape, dtype=self.config.dtype_values, device=self.device))
            
        # Pre-allocate output tensors
        batch_children = torch.full((batch_size, max_children), -1,
                                   dtype=self.config.dtype_indices, device=self.device)
        batch_actions = torch.full((batch_size, max_children), -1,
                                  dtype=self.config.dtype_actions, device=self.device)
        batch_priors = torch.full((batch_size, max_children), 0.0,
                                 dtype=self.config.dtype_values, device=self.device)
        
        # Vectorized gathering using CSR indexing
        for i in range(batch_size):
            node_idx = node_indices[i].item()
            start = self.row_ptr[node_idx].item()
            end = self.row_ptr[node_idx + 1].item()
            num_children = end - start
            
            if num_children > 0:
                actual_children = min(num_children, max_children)
                batch_children[i, :actual_children] = self.col_indices[start:start + actual_children]
                batch_actions[i, :actual_children] = self.edge_actions[start:start + actual_children]  
                batch_priors[i, :actual_children] = self.edge_priors[start:start + actual_children]
                
        return batch_children, batch_actions, batch_priors
    
    def batch_select_ucb_optimized(
        self,
        node_indices: torch.Tensor,
        c_puct: float = 1.4,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select best actions using optimized GPU kernels
        
        Args:
            node_indices: Nodes to select actions for
            c_puct: UCB exploration constant
            temperature: Temperature for exploration
            
        Returns:
            Tuple of (selected_actions, ucb_scores)
        """
        if self.batch_ops is not None:
            # Use the new optimized interface
            actions = self.batch_ops.batch_select_ucb(
                node_indices=node_indices,
                row_ptr=self.row_ptr,
                col_indices=self.col_indices,
                edge_actions=self.edge_actions,
                edge_priors=self.edge_priors,
                visit_counts=self.visit_counts,
                value_sums=self.value_sums,
                c_puct=c_puct,
                temperature=temperature
            )
            # Return actions and dummy scores for compatibility
            return actions, torch.zeros_like(actions, dtype=torch.float32)
        else:
            # Fallback to standard implementation
            from .csr_gpu_kernels import csr_batch_ucb_torch
            actions = csr_batch_ucb_torch(
                node_indices, self.row_ptr, self.col_indices,
                self.edge_actions, self.edge_priors,
                self.visit_counts, self.value_sums,
                self.visit_counts[node_indices],
                c_puct, temperature
            )
            return actions, torch.zeros_like(actions, dtype=torch.float32)
    
    def batch_get_children_vectorized(self, node_indices: torch.Tensor,
                                     max_children: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fully vectorized version using advanced indexing (experimental)
        
        This version attempts to eliminate the Python loop using tensor operations.
        May be faster for large batch sizes on GPU.
        """
        batch_size = len(node_indices)
        
        # Get row start/end positions
        row_starts = self.row_ptr[node_indices]  # [batch_size]
        row_ends = self.row_ptr[node_indices + 1]  # [batch_size]
        num_children_per_node = row_ends - row_starts  # [batch_size]
        
        if max_children is None:
            max_children = num_children_per_node.max().item()
            
        if max_children == 0:
            empty_shape = (batch_size, 0)
            return (torch.empty(empty_shape, dtype=self.config.dtype_indices, device=self.device),
                   torch.empty(empty_shape, dtype=self.config.dtype_actions, device=self.device),
                   torch.empty(empty_shape, dtype=self.config.dtype_values, device=self.device))
        
        # Create index tensors for gathering
        batch_idx = torch.arange(batch_size, device=self.device).unsqueeze(1)  # [batch_size, 1]
        child_idx = torch.arange(max_children, device=self.device).unsqueeze(0)  # [1, max_children]
        
        # Calculate absolute indices in CSR arrays
        abs_indices = row_starts.unsqueeze(1) + child_idx  # [batch_size, max_children]
        
        # Create mask for valid indices
        valid_mask = child_idx < num_children_per_node.unsqueeze(1)  # [batch_size, max_children]
        
        # Clamp indices to valid range
        abs_indices = torch.clamp(abs_indices, 0, self.num_edges - 1)
        
        # Gather data
        batch_children = torch.where(valid_mask, 
                                    self.col_indices[abs_indices],
                                    torch.full_like(abs_indices, -1))
        batch_actions = torch.where(valid_mask,
                                   self.edge_actions[abs_indices], 
                                   torch.full_like(abs_indices, -1))
        batch_priors = torch.where(valid_mask,
                                  self.edge_priors[abs_indices],
                                  torch.zeros_like(abs_indices, dtype=self.config.dtype_values))
        
        return batch_children, batch_actions, batch_priors
        
    def update_visit_count(self, node_idx: int, delta: int = 1):
        """Update visit count for a node"""
        self.visit_counts[node_idx] += delta
        
    def update_value_sum(self, node_idx: int, value: float):
        """Update value sum for a node"""
        self.value_sums[node_idx] += value
        
    def get_q_value(self, node_idx: int) -> float:
        """Get Q-value (average value) for a node"""
        visits = self.visit_counts[node_idx].item()
        if visits == 0:
            return 0.0
        return (self.value_sums[node_idx] / visits).item()
        
    def batch_update_visits(self, node_indices: torch.Tensor, deltas: torch.Tensor):
        """Update visit counts for multiple nodes"""
        self.visit_counts[node_indices] += deltas
        
    def batch_update_values(self, node_indices: torch.Tensor, values: torch.Tensor):
        """Update value sums for multiple nodes"""
        self.value_sums[node_indices] += values
    
    def batch_backup_optimized(self, paths: torch.Tensor, values: torch.Tensor):
        """Optimized backup using coalesced memory access
        
        Args:
            paths: (batch_size, max_depth) tensor of node indices
            values: (batch_size,) tensor of values to backup
        """
        if self.batch_ops is not None:
            # Calculate path lengths for the new interface
            batch_size, max_depth = paths.shape
            # Path length is the number of valid (non-negative) nodes in each path
            path_lengths = (paths >= 0).sum(dim=1)
            
            # Use the new optimized backup interface
            updated_visits, updated_values = self.batch_ops.coalesced_backup(
                paths=paths,
                values=values,
                path_lengths=path_lengths,
                visit_counts=self.visit_counts,
                value_sums=self.value_sums
            )
            
            # Update the tree's tensors with the results
            self.visit_counts = updated_visits
            self.value_sums = updated_values
        else:
            # Fallback implementation
            batch_size, max_depth = paths.shape
            for i in range(batch_size):
                path = paths[i]
                value = values[i]
                valid_nodes = path[path >= 0]
                self.batch_update_visits(valid_nodes, torch.ones_like(valid_nodes))
                self.batch_update_values(valid_nodes, torch.full_like(valid_nodes, value, dtype=self.config.dtype_values))
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics in MB"""
        def tensor_mb(tensor):
            return tensor.element_size() * tensor.numel() / (1024 * 1024)
            
        node_memory = (tensor_mb(self.visit_counts) + tensor_mb(self.value_sums) + 
                      tensor_mb(self.node_priors) + tensor_mb(self.parent_indices) +
                      tensor_mb(self.parent_actions) + tensor_mb(self.flags) + 
                      tensor_mb(self.phases))
        
        csr_memory = (tensor_mb(self.row_ptr) + tensor_mb(self.col_indices) +
                     tensor_mb(self.edge_actions) + tensor_mb(self.edge_priors))
        
        return {
            'node_data_mb': node_memory,
            'csr_structure_mb': csr_memory,  
            'total_mb': node_memory + csr_memory,
            'nodes': self.num_nodes,
            'edges': self.num_edges,
            'bytes_per_node': (node_memory + csr_memory) * 1024 * 1024 / max(1, self.num_nodes)
        }
        
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            **self.stats,
            **self.get_memory_usage(),
            'edge_capacity': self.edge_capacity,
            'edge_utilization': self.num_edges / self.edge_capacity if self.edge_capacity > 0 else 0.0
        }
        
    @property 
    def is_terminal(self) -> torch.Tensor:
        """Get terminal flags as tensor"""
        return (self.flags & 2).bool()
        
    @property
    def is_expanded(self) -> torch.Tensor:
        """Get expanded flags as tensor"""
        return (self.flags & 1).bool()
        
    def set_terminal(self, node_idx: int, value: bool = True):
        """Set terminal flag for a node"""
        if value:
            self.flags[node_idx] |= 2
        else:
            self.flags[node_idx] &= ~2
            
    def set_expanded(self, node_idx: int, value: bool = True):
        """Set expanded flag for a node"""
        if value:
            self.flags[node_idx] |= 1
        else:
            self.flags[node_idx] &= ~1
    
    def _rebuild_row_pointers(self):
        """Rebuild row pointers from edge data for CSR format consistency"""
        if not self._needs_row_ptr_update:
            return
            
        # Reset row pointers
        self.row_ptr.zero_()
        
        # Count children for each node
        for edge_idx in range(self.num_edges):
            parent_idx = None
            child_idx = self.col_indices[edge_idx].item()
            
            # Find parent by looking at parent_indices
            if child_idx < self.num_nodes:
                parent_idx = self.parent_indices[child_idx].item()
                if parent_idx >= 0:
                    self.row_ptr[parent_idx + 1] += 1
        
        # Convert counts to pointers (cumulative sum)
        for i in range(1, self.row_ptr.shape[0]):
            self.row_ptr[i] += self.row_ptr[i - 1]
        
        self._needs_row_ptr_update = False
    
    def ensure_consistent(self):
        """Ensure CSR structure is consistent - call this before batch operations"""
        if self._needs_row_ptr_update:
            self._rebuild_row_pointers()
