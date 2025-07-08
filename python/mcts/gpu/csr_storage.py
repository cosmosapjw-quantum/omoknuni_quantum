"""CSR (Compressed Sparse Row) storage management

This module handles the CSR format storage for tree edges,
separating edge storage from tree logic.
"""

import torch
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass 
class CSRStorageConfig:
    """Configuration for CSR storage"""
    max_edges: int = 500000
    device: str = 'cuda'
    dtype_indices: torch.dtype = torch.int32
    dtype_actions: torch.dtype = torch.int32
    dtype_values: torch.dtype = torch.float32
    initial_capacity_factor: float = 0.1
    growth_factor: float = 1.5


class CSRStorage:
    """Manages CSR format storage for tree edges
    
    CSR format stores tree children using three arrays:
    - row_ptr: Starting index for each node's children
    - col_indices: Child node indices (flattened)
    - edge_data: Associated edge data (actions, priors)
    
    This provides:
    - Contiguous memory access for GPU kernels
    - Efficient iteration over node children
    - Better cache utilization vs pointer-based trees
    """
    
    def __init__(self, config: CSRStorageConfig, initial_nodes: int = 100000):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Initialize storage
        self._initialize_storage(initial_nodes)
        
        # Track edge count
        self.num_edges = 0
        
        # Flag for deferred row pointer updates
        self._needs_row_ptr_update = False
        
    def _initialize_storage(self, initial_nodes: int):
        """Initialize CSR storage arrays"""
        # Determine initial edge capacity
        if self.config.max_edges > 0:
            initial_edges = int(self.config.max_edges * self.config.initial_capacity_factor)
        else:
            initial_edges = 50000
            
        # Row pointers (one per node + 1)
        self.row_ptr = torch.zeros(initial_nodes + 1, device=self.device, 
                                  dtype=self.config.dtype_indices)
        
        # Column indices (child node IDs)
        self.col_indices = torch.zeros(initial_edges, device=self.device,
                                      dtype=self.config.dtype_indices)
        
        # Edge data
        self.edge_actions = torch.zeros(initial_edges, device=self.device,
                                       dtype=self.config.dtype_actions)
        self.edge_priors = torch.zeros(initial_edges, device=self.device,
                                      dtype=self.config.dtype_values)
        
        # Current capacity
        self.edge_capacity = initial_edges
        
    def add_edge(self, parent_idx: int, child_idx: int, action: int, prior: float) -> int:
        """Add a single edge and return its index"""
        if self.num_edges >= self.edge_capacity:
            self._grow_edge_storage(self.num_edges + 1)
            
        edge_idx = self.num_edges
        self.col_indices[edge_idx] = child_idx
        self.edge_actions[edge_idx] = action
        self.edge_priors[edge_idx] = prior
        self.num_edges += 1
        
        # Mark for row pointer update
        self._needs_row_ptr_update = True
        
        return edge_idx
        
    def add_edges_batch(self, parent_idx: int, child_indices: torch.Tensor,
                       actions: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        """Add multiple edges in batch"""
        num_new_edges = len(child_indices)
        
        if self.num_edges + num_new_edges > self.edge_capacity:
            self._grow_edge_storage(self.num_edges + num_new_edges)
            
        start_edge = self.num_edges
        edge_indices = torch.arange(start_edge, start_edge + num_new_edges, device=self.device)
        
        # Vectorized assignment
        self.col_indices[edge_indices] = child_indices
        self.edge_actions[edge_indices] = actions
        self.edge_priors[edge_indices] = priors
        
        self.num_edges += num_new_edges
        
        # Mark for row pointer update
        self._needs_row_ptr_update = True
        
        return edge_indices
        
    def _grow_edge_storage(self, min_required: int):
        """Grow edge storage when capacity is exceeded"""
        # Check if minimum required exceeds max edges
        if self.config.max_edges > 0 and min_required > self.config.max_edges:
            raise RuntimeError(f"Cannot grow edge storage beyond {self.config.max_edges}")
            
        new_capacity = max(min_required, int(self.edge_capacity * self.config.growth_factor))
        
        if self.config.max_edges > 0:
            new_capacity = min(new_capacity, self.config.max_edges)
            if new_capacity <= self.edge_capacity:
                raise RuntimeError(f"Cannot grow edge storage beyond {self.config.max_edges}")
                
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
        
        # Replace tensors
        self.col_indices = new_col_indices
        self.edge_actions = new_edge_actions
        self.edge_priors = new_edge_priors
        self.edge_capacity = new_capacity
        
        # Track reallocation
        self.memory_reallocations = getattr(self, 'memory_reallocations', 0) + 1
        
    def grow_row_ptr_if_needed(self, num_nodes: int):
        """Ensure row_ptr array is large enough for num_nodes"""
        required_size = num_nodes + 1
        if required_size > len(self.row_ptr):
            new_size = max(required_size, int(len(self.row_ptr) * self.config.growth_factor))
            new_row_ptr = torch.zeros(new_size, device=self.device, 
                                    dtype=self.config.dtype_indices)
            new_row_ptr[:len(self.row_ptr)] = self.row_ptr
            self.row_ptr = new_row_ptr
            
    def get_children_edges(self, node_idx: int) -> Tuple[int, int]:
        """Get start and end indices for a node's edges"""
        if self._needs_row_ptr_update:
            raise RuntimeError("Row pointers need update before querying children")
            
        start = self.row_ptr[node_idx].item()
        end = self.row_ptr[node_idx + 1].item()
        return start, end
        
    def get_node_children(self, node_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get children data for a node
        
        Returns:
            child_indices: Tensor of child node indices
            actions: Tensor of actions leading to children
            priors: Tensor of child prior probabilities
        """
        start, end = self.get_children_edges(node_idx)
        
        if start == end:
            # No children
            empty = torch.empty(0, device=self.device)
            return (empty.to(self.config.dtype_indices),
                   empty.to(self.config.dtype_actions),
                   empty.to(self.config.dtype_values))
                   
        edge_slice = slice(start, end)
        return (self.col_indices[edge_slice],
               self.edge_actions[edge_slice],
               self.edge_priors[edge_slice])
               
    def rebuild_row_pointers(self, children_table: torch.Tensor):
        """Rebuild row pointers from children table
        
        Args:
            children_table: [num_nodes, max_children] tensor where -1 indicates no child
        """
        if not self._needs_row_ptr_update:
            return
            
        with torch.no_grad():
            # Reset row pointers
            self.row_ptr.zero_()
            
            # Count valid children for each node
            valid_children_mask = children_table >= 0
            children_counts = valid_children_mask.sum(dim=1)
            
            # Ensure we don't exceed row_ptr bounds
            max_nodes = min(len(children_counts), self.row_ptr.shape[0] - 1)
            if max_nodes > 0:
                # Set counts in row_ptr (offset by 1 for CSR format)
                self.row_ptr[1:max_nodes+1] = children_counts[:max_nodes].to(self.row_ptr.dtype)
                
            # Convert counts to pointers (cumulative sum)
            self.row_ptr = torch.cumsum(self.row_ptr, dim=0)
            
        self._needs_row_ptr_update = False
        
    def needs_row_ptr_update(self) -> bool:
        """Check if row pointers need updating"""
        return self._needs_row_ptr_update
        
    def reset(self):
        """Reset CSR storage"""
        if self.num_edges > 0:
            # Clear used portion
            self.col_indices[:self.num_edges].zero_()
            self.edge_actions[:self.num_edges].zero_()
            self.edge_priors[:self.num_edges].zero_()
            
        self.row_ptr.zero_()
        self.num_edges = 0
        self._needs_row_ptr_update = False
        
    def get_memory_usage_mb(self) -> float:
        """Get total memory usage in MB"""
        def tensor_mb(tensor):
            return tensor.element_size() * tensor.numel() / (1024 * 1024)
            
        total = (tensor_mb(self.row_ptr) + tensor_mb(self.col_indices) +
                tensor_mb(self.edge_actions) + tensor_mb(self.edge_priors))
        return total
        
    def get_edge_utilization(self) -> float:
        """Get edge capacity utilization percentage"""
        if self.edge_capacity > 0:
            return self.num_edges / self.edge_capacity
        return 0.0