"""Refactored CSR Tree implementation using modular components

This is a cleaner, more maintainable version of CSRTree that separates
concerns into different modules while maintaining all functionality.
"""

import torch
from typing import Optional, Tuple, Dict, List, Union, Any
from dataclasses import dataclass
import numpy as np
import logging

from .node_data_manager import NodeDataManager, NodeDataConfig
from .csr_storage import CSRStorage, CSRStorageConfig
from .ucb_selector import UCBSelector, UCBConfig
from .mcts_gpu_accelerator import get_mcts_gpu_accelerator

logger = logging.getLogger(__name__)


@dataclass
class CSRTreeConfig:
    """Configuration for CSR tree"""
    max_nodes: int = 0  # 0 means no limit
    max_edges: int = 0  # 0 means no limit
    max_actions: int = 512  # Maximum actions per node
    device: str = 'cuda'
    dtype_indices: torch.dtype = None
    dtype_actions: torch.dtype = None
    dtype_values: torch.dtype = None
    
    # Performance tuning
    initial_capacity_factor: float = 0.1
    growth_factor: float = 1.5
    enable_memory_pooling: bool = True
    
    # Batching configuration
    batch_size: int = 256
    enable_batched_ops: bool = True
    
    # Virtual loss for parallelization
    virtual_loss_value: float = -1.0
    enable_virtual_loss: bool = True
    
    def __post_init__(self):
        """Set default dtypes after initialization"""
        if self.dtype_indices is None:
            self.dtype_indices = torch.int32
        if self.dtype_actions is None:
            self.dtype_actions = torch.int32
        if self.dtype_values is None:
            self.dtype_values = torch.float32


class CSRTree:
    """Refactored GPU-optimized tree using modular components
    
    This cleaner implementation separates concerns:
    - Node data management (NodeDataManager)
    - CSR edge storage (CSRStorage)
    - UCB selection logic (UCBSelector)
    - Tree operations (this class)
    
    All original functionality is preserved while improving maintainability.
    """
    
    def __init__(self, config: CSRTreeConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self._init_components()
        
        # Initialize storage
        self._init_storage()
        
        # Tree state
        self.num_nodes = 0
        self.num_edges = 0
        self.max_nodes = config.max_nodes if config.max_nodes > 0 else float('inf')
        self.max_edges = config.max_edges if config.max_edges > 0 else float('inf')
        
        # Atomic counters for thread safety
        self.node_counter = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.edge_counter = torch.zeros(1, dtype=torch.int32, device=self.device)
        
        # Performance tracking
        self.stats = {
            'memory_reallocations': 0,
            'batch_operations': 0,
            'cache_hits': 0,
            'batched_additions': 0
        }
        
        # Initialize batch operations if enabled
        if config.enable_batched_ops:
            self._init_batch_buffers()
            
        # Initialize GPU kernels if available
        self._init_gpu_kernels()
        
        # Add root node
        root_idx = self.add_root(prior=1.0)
        
        # Update counters
        self.node_counter[0] = self.num_nodes
        self.edge_counter[0] = self.num_edges
        
    def _init_components(self):
        """Initialize modular components"""
        # Node data manager
        node_config = NodeDataConfig(
            max_nodes=self.config.max_nodes,
            device=self.config.device,
            dtype_values=self.config.dtype_values,
            dtype_indices=self.config.dtype_indices,
            initial_capacity_factor=self.config.initial_capacity_factor,
            growth_factor=self.config.growth_factor,
            enable_virtual_loss=self.config.enable_virtual_loss,
            virtual_loss_value=self.config.virtual_loss_value
        )
        self.node_data = NodeDataManager(node_config)
        
        # CSR storage
        csr_config = CSRStorageConfig(
            max_edges=self.config.max_edges,
            device=self.config.device,
            dtype_indices=self.config.dtype_indices,
            dtype_actions=self.config.dtype_actions,
            dtype_values=self.config.dtype_values,
            initial_capacity_factor=self.config.initial_capacity_factor,
            growth_factor=self.config.growth_factor
        )
        initial_nodes = self.config.max_nodes if self.config.max_nodes > 0 else 100000
        self.csr_storage = CSRStorage(csr_config, initial_nodes)
        
        # UCB selector
        ucb_config = UCBConfig(
            c_puct=1.4,  # Default, will be overridden in select calls
            temperature=1.0,
            enable_virtual_loss=self.config.enable_virtual_loss,
            virtual_loss_value=self.config.virtual_loss_value,
            device=self.config.device
        )
        self.ucb_selector = UCBSelector(ucb_config)
        
    def _init_storage(self):
        """Initialize additional storage"""
        # Game state tracking
        self.node_states = {}
        
        # Children lookup table for compatibility
        max_children = self.config.max_actions
        if self.config.max_nodes > 0:
            if self.config.max_nodes <= 100:
                initial_nodes = self.config.max_nodes
            else:
                initial_nodes = max(10, int(self.config.max_nodes * self.config.initial_capacity_factor))
        else:
            initial_nodes = 100000
        self.children = torch.full((initial_nodes, max_children), -1, 
                                  device=self.device, dtype=torch.int32)
                                  
    def _init_batch_buffers(self):
        """Initialize buffers for batched operations"""
        batch_size = self.config.batch_size
        max_children = self.config.max_actions
        
        self.batch_parent_indices = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        self.batch_actions = torch.zeros((batch_size, max_children), dtype=torch.int32, device=self.device)
        self.batch_priors = torch.zeros((batch_size, max_children), dtype=torch.float32, device=self.device)
        self.batch_num_children = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        self.batch_states = [None] * batch_size
        self.batch_position = 0
        
    def _init_gpu_kernels(self):
        """Initialize GPU kernels if available"""
        try:
            self.batch_ops = get_mcts_gpu_accelerator(self.device)
        except Exception as e:
            self.batch_ops = None
            logger.warning(f"Failed to initialize GPU kernels: {e}")
            
    def _resize_children_array(self, min_size: int):
        """Resize children array to accommodate more nodes"""
        old_size = self.children.shape[0]
        new_size = max(min_size, int(old_size * self.config.growth_factor))
        max_children = self.children.shape[1]
        
        # Create new array
        new_children = torch.full((new_size, max_children), -1, 
                                 device=self.device, dtype=torch.int32)
        
        # Copy old data
        new_children[:old_size] = self.children
        
        # Replace old array
        self.children = new_children
        self.stats['memory_reallocations'] += 1
        logger.debug(f"Resized children array from {old_size} to {new_size}")
            
    # Properties for backward compatibility
    
    @property
    def children_start_indices(self):
        """Backward compatibility property for accessing CSR row pointers"""
        return self.csr_storage.row_ptr
        
    @property
    def children_end_indices(self):
        """Backward compatibility property for end indices (computed from row_ptr)"""
        # In CSR format, end index for node i is start index for node i+1
        return self.csr_storage.row_ptr[1:]
            
    # Tree operations
    
    def add_root(self, prior: float = 1.0, state: Optional[Any] = None) -> int:
        """Add root node and return its index"""
        if self.num_nodes > 0:
            raise RuntimeError("Root already exists")
            
        idx = self.node_data.allocate_node(prior, parent_idx=-1, parent_action=-1)
        self.num_nodes = self.node_data.num_nodes
        
        if state is not None:
            self.node_states[idx] = state
            
        # Initialize CSR row pointer
        self.csr_storage.row_ptr[idx] = 0
        
        return idx
        
    def add_child(self, parent_idx: int, action: int, child_prior: float,
                  child_state: Optional[Any] = None) -> int:
        """Add a single child node"""
        if self.config.enable_batched_ops:
            return self.add_children_batch(parent_idx, [action], [child_prior],
                                         [child_state] if child_state else None)[0]
        else:
            return self._add_child_direct(parent_idx, action, child_prior, child_state)
            
    def _add_child_direct(self, parent_idx: int, action: int, child_prior: float,
                         child_state: Optional[Any] = None) -> int:
        """Add a child node directly (non-batched)"""
        if self.max_nodes != float('inf') and self.num_nodes >= self.max_nodes:
            raise RuntimeError(f"Tree full: {self.num_nodes} nodes")
            
        # Allocate node
        try:
            child_idx = self.node_data.allocate_node(child_prior, parent_idx, action)
            self.num_nodes = self.node_data.num_nodes
        except RuntimeError:
            # Re-raise with consistent message
            raise RuntimeError(f"Tree full: {self.num_nodes} nodes")
        
        if child_state is not None:
            self.node_states[child_idx] = child_state
            
        # Add edge to CSR
        self.csr_storage.add_edge(parent_idx, child_idx, action, child_prior)
        self.num_edges = self.csr_storage.num_edges
        
        # Update counters
        self.node_counter[0] = self.num_nodes
        self.edge_counter[0] = self.num_edges
        
        # Update children lookup table
        self._update_children_table(parent_idx, child_idx)
        
        return child_idx
        
    def add_children_batch(self, parent_idx: int, actions: List[int], priors: List[float],
                          states: Optional[List[Any]] = None) -> List[int]:
        """Add multiple children to a parent node"""
        if not actions:
            return []
            
        # Check for duplicate actions
        actions, priors, states = self._filter_duplicate_actions(parent_idx, actions, priors, states)
        if not actions:
            return []
            
        num_children = len(actions)
        
        # Check capacity
        if self.max_nodes != float('inf') and self.num_nodes + num_children > self.max_nodes:
            raise RuntimeError(f"Tree full: {self.num_nodes + num_children} > {self.max_nodes}")
            
        # Convert to tensors
        actions_tensor = torch.tensor(actions, device=self.device, dtype=self.config.dtype_actions)
        priors_tensor = torch.tensor(priors, device=self.device, dtype=self.config.dtype_values)
        
        # Allocate nodes
        try:
            child_indices = self.node_data.allocate_nodes_batch(
                num_children, priors_tensor, parent_idx, actions_tensor
            )
            self.num_nodes = self.node_data.num_nodes
        except RuntimeError:
            # Re-raise with consistent message
            raise RuntimeError(f"Tree full: {self.num_nodes} nodes")
        
        # Add edges
        self.csr_storage.add_edges_batch(parent_idx, child_indices, actions_tensor, priors_tensor)
        self.num_edges = self.csr_storage.num_edges
        
        # Update children table by action index
        for i, (action, child_idx) in enumerate(zip(actions, child_indices)):
            self.children[parent_idx, action] = child_idx
        
        # Update row pointers for CSR format
        # This is needed to properly query children later
        if hasattr(self.csr_storage, 'rebuild_row_pointers'):
            # OPTIMIZATION: Only process nodes that actually exist
            self.csr_storage.rebuild_row_pointers(self.children, num_active_nodes=self.num_nodes)
        
        # Update counters
        self.node_counter[0] = self.num_nodes
        self.edge_counter[0] = self.num_edges
        
        # Store game states
        if states:
            for child_idx, state in zip(child_indices.cpu().numpy(), states):
                if state is not None:
                    self.node_states[child_idx] = state
                    
        # NOTE: Removed duplicate call to _update_children_table_batch
        # Children are already added by action index above
        
        return child_indices.cpu().tolist()
        
    def _filter_duplicate_actions(self, parent_idx: int, actions: List[int], 
                                 priors: List[float], states: Optional[List[Any]]) -> Tuple:
        """Filter out duplicate actions"""
        existing_children, existing_actions, _ = self.get_children(parent_idx)
        
        if len(existing_actions) == 0:
            return actions, priors, states
            
        existing_set = set(existing_actions.cpu().numpy().tolist())
        
        filtered_actions = []
        filtered_priors = []
        filtered_states = [] if states else None
        
        for i, action in enumerate(actions):
            if action not in existing_set:
                filtered_actions.append(action)
                filtered_priors.append(priors[i])
                if states:
                    filtered_states.append(states[i])
                    
        return filtered_actions, filtered_priors, filtered_states
        
    def _update_children_table(self, parent_idx: int, child_idx: int):
        """Update children lookup table for single child"""
        # Ensure children table is large enough
        self._ensure_children_capacity(parent_idx + 1)
        
        for i in range(self.children.shape[1]):
            if self.children[parent_idx, i] == -1:
                self.children[parent_idx, i] = child_idx
                break
                
    def _update_children_table_batch(self, parent_idx: int, child_indices: torch.Tensor):
        """Update children lookup table for multiple children"""
        # Ensure children table is large enough
        self._ensure_children_capacity(parent_idx + 1)
        
        parent_children = self.children[parent_idx]
        empty_mask = parent_children == -1
        empty_indices = torch.where(empty_mask)[0]
        
        num_to_add = min(len(child_indices), len(empty_indices))
        if num_to_add > 0:
            self.children[parent_idx, empty_indices[:num_to_add]] = child_indices[:num_to_add]
    def _ensure_children_capacity(self, min_nodes: int):
        """Ensure children table has enough rows for min_nodes"""
        if min_nodes > self.children.shape[0]:
            # Need to grow the children table
            old_size = self.children.shape[0]
            new_size = max(min_nodes, int(old_size * self.config.growth_factor))
            
            if self.config.max_nodes > 0:
                new_size = min(new_size, self.config.max_nodes)
                
            # Create new tensor
            new_children = torch.full((new_size, self.children.shape[1]), -1,
                                    device=self.device, dtype=torch.int32)
            
            # Copy old data
            new_children[:old_size] = self.children
            
            # Replace
            self.children = new_children
            
            # Track reallocation
            self.stats['memory_reallocations'] += 1
            
    def get_children(self, node_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get children indices, actions, and priors for a node"""
        # Bounds check
        if node_idx < 0 or node_idx >= self.children.shape[0]:
            empty = torch.empty(0, device=self.device)
            return (empty.to(self.config.dtype_indices),
                   empty.to(self.config.dtype_actions),
                   empty.to(self.config.dtype_values))
        
        # Use children lookup table for O(1) access
        children_slice = self.children[node_idx]
        valid_mask = children_slice >= 0
        
        if not valid_mask.any():
            empty = torch.empty(0, device=self.device)
            return (empty.to(self.config.dtype_indices),
                   empty.to(self.config.dtype_actions),
                   empty.to(self.config.dtype_values))
                   
        valid_children = children_slice[valid_mask]
        return (valid_children,
               self.node_data.parent_actions[valid_children],
               self.node_data.node_priors[valid_children])
               
    def batch_get_children(self, node_indices: torch.Tensor,
                          max_children: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get children for multiple nodes efficiently"""
        # Bounds check
        max_idx = node_indices.max().item() if node_indices.numel() > 0 else -1
        if max_idx >= self.children.shape[0]:
            # Need to resize children array
            self._resize_children_array(max_idx + 1)
        
        batch_children = self.children[node_indices]
        
        if max_children is not None and max_children < batch_children.shape[1]:
            batch_children = batch_children[:, :max_children]
            
        valid_mask = batch_children >= 0
        
        # Pre-allocate output tensors
        batch_actions = torch.full_like(batch_children, -1, dtype=self.config.dtype_actions)
        batch_priors = torch.zeros_like(batch_children, dtype=self.config.dtype_values)
        
        # Vectorized gathering
        valid_children = batch_children[valid_mask]
        if valid_children.numel() > 0:
            batch_actions[valid_mask] = self.node_data.parent_actions[valid_children]
            batch_priors[valid_mask] = self.node_data.node_priors[valid_children]
            
        return batch_children, batch_actions, batch_priors
        
    def batch_select_ucb_optimized(self, node_indices: torch.Tensor,
                                  c_puct: float = 1.4,
                                  temperature: float = 1.0,
                                  **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select best actions using UCB formula"""
        batch_size = len(node_indices)
        
        # Get children for all nodes
        batch_children, batch_actions, batch_priors = self.batch_get_children(node_indices)
        
        # Create valid mask
        valid_mask = batch_children >= 0
        
        # Get parent visits
        parent_visits = self.node_data.visit_counts[node_indices]
        
        # Get child statistics
        flat_children = batch_children[valid_mask]
        if flat_children.numel() == 0:
            return (torch.full((batch_size,), -1, dtype=torch.int32, device=self.device),
                   torch.zeros(batch_size, device=self.device))
                   
        # Prepare child data
        child_visits = torch.zeros_like(batch_children, dtype=torch.float32)
        child_values = torch.zeros_like(batch_children, dtype=torch.float32)
        
        if valid_mask.any():
            if self.config.enable_virtual_loss:
                effective_visits = self.node_data.get_effective_visits(flat_children)
                effective_values = self.node_data.get_effective_values(flat_children)
                child_visits[valid_mask] = effective_visits.float()
                child_values[valid_mask] = effective_values
            else:
                child_visits[valid_mask] = self.node_data.visit_counts[flat_children].float()
                child_values[valid_mask] = self.node_data.value_sums[flat_children]
                
        # Use UCB selector
        position_indices, ucb_scores = self.ucb_selector.select_batch(
            parent_visits, child_visits, child_values, batch_priors,
            valid_mask, c_puct, temperature
        )
        
        return position_indices, ucb_scores
        
    def batch_action_to_child(self, node_indices: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Convert position indices to child node indices"""
        batch_size = len(node_indices)
        
        if batch_size == 0:
            return torch.empty(0, dtype=torch.int32, device=self.device)
            
        with torch.no_grad():
            # Bounds checking
            node_bounds_ok = (node_indices >= 0) & (node_indices < self.num_nodes)
            action_bounds_ok = (actions >= 0) & (actions < self.children.shape[1])
            valid_mask = node_bounds_ok & action_bounds_ok
            
            # Initialize result
            child_indices = torch.full((batch_size,), -1, dtype=torch.int32, device=self.device)
            
            if not valid_mask.any():
                return child_indices
                
            # Get children using position indices
            valid_positions = torch.where(valid_mask)[0]
            valid_nodes = node_indices[valid_mask]
            valid_position_indices = actions[valid_mask]
            
            node_children_batch = self.children[valid_nodes]
            row_indices = torch.arange(len(valid_nodes), device=self.device)
            selected_children = node_children_batch[row_indices, valid_position_indices]
            
            # Check validity and assign
            valid_children = selected_children >= 0
            result_positions = valid_positions[valid_children]
            child_indices[result_positions] = selected_children[valid_children]
            
            return child_indices
            
    # Node data operations (delegate to NodeDataManager)
    
    @property
    def visit_counts(self):
        return self.node_data.visit_counts
        
    @property
    def value_sums(self):
        return self.node_data.value_sums
        
    @property
    def node_priors(self):
        return self.node_data.node_priors
        
    @property
    def parent_indices(self):
        return self.node_data.parent_indices
        
    @property
    def parent_actions(self):
        return self.node_data.parent_actions
        
    @property
    def flags(self):
        return self.node_data.flags
        
    @property
    def phases(self):
        return self.node_data.phases
        
    @property
    def virtual_loss_counts(self):
        return self.node_data.virtual_loss_counts
        
    def update_visit_count(self, node_idx: int, delta: int = 1):
        self.node_data.update_visit_count(node_idx, delta)
        
    def update_value_sum(self, node_idx: int, value: float):
        self.node_data.update_value_sum(node_idx, value)
        
    def batch_update_visits(self, node_indices: torch.Tensor, deltas: torch.Tensor):
        self.node_data.batch_update_visits(node_indices, deltas)
        
    def batch_update_values(self, node_indices: torch.Tensor, values: torch.Tensor):
        self.node_data.batch_update_values(node_indices, values)
        
    def get_q_value(self, node_idx: int) -> float:
        return self.node_data.get_q_value(node_idx)
        
    def apply_virtual_loss(self, node_indices: torch.Tensor):
        self.node_data.apply_virtual_loss(node_indices)
        
    def remove_virtual_loss(self, node_indices: torch.Tensor):
        self.node_data.remove_virtual_loss(node_indices)
        
    def set_terminal(self, node_idx: int, value: bool = True):
        self.node_data.set_terminal(node_idx, value)
        
    def set_expanded(self, node_idx: int, value: bool = True):
        self.node_data.set_expanded(node_idx, value)
        
    # CSR operations (delegate to CSRStorage)
    
    @property
    def row_ptr(self):
        return self.csr_storage.row_ptr
        
    @property
    def col_indices(self):
        return self.csr_storage.col_indices
        
    @property
    def edge_actions(self):
        return self.csr_storage.edge_actions
        
    @property
    def edge_priors(self):
        return self.csr_storage.edge_priors
        
    def ensure_consistent(self, force: bool = False):
        """Ensure CSR structure is consistent"""
        if force or self.csr_storage.needs_row_ptr_update():
            self.flush_batch()
            # OPTIMIZATION: Only process nodes that actually exist
            self.csr_storage.rebuild_row_pointers(self.children, num_active_nodes=self.num_nodes)
            
    def flush_batch(self):
        """Flush any pending batched operations"""
        # Placeholder - implement if batch operations are added
        pass
        
    # Tree operations
    
    def reset(self):
        """Reset tree to initial state with only root node"""
        # Reset components
        self.node_data.reset()
        self.csr_storage.reset()
        
        # Reset tree state
        self.num_nodes = 0
        self.num_edges = 0
        
        # Clear children table completely to avoid stale data
        self.children.fill_(-1)
        
        # Clear states
        self.node_states.clear()
        
        # Reset counters
        self.node_counter.zero_()
        self.edge_counter.zero_()
        
        # Clear stats
        self.stats = {
            'memory_reallocations': 0,
            'batch_operations': 0,
            'cache_hits': 0,
            'batched_additions': 0
        }
        
        # Add root node
        root_idx = self.add_root(prior=1.0)
        
        # Update counters
        self.node_counter[0] = self.num_nodes
        self.edge_counter[0] = self.num_edges
        
    def backup_path(self, path: List[int], value: float):
        """Backup value along a path (for compatibility)"""
        current_value = value
        for node_idx in reversed(path):
            self.node_data.update_visit_count(node_idx, 1)
            self.node_data.update_value_sum(node_idx, current_value)
            current_value = -current_value  # Minimax alternation
            
    def select_child(self, node_idx: int, c_puct: float = 1.414) -> Optional[int]:
        """Select best child using UCB formula (single node version)"""
        child_indices, child_actions, child_priors = self.get_children(node_idx)
        
        if len(child_indices) == 0:
            return None
            
        parent_visits = max(1, self.node_data.visit_counts[node_idx].item())
        child_visits = self.node_data.visit_counts[child_indices]
        child_values = self.node_data.value_sums[child_indices]
        
        best_idx = self.ucb_selector.select_single(
            parent_visits, child_visits, child_values, child_priors, c_puct
        )
        
        if best_idx >= 0:
            return child_indices[best_idx].item()
        return None
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics in MB"""
        node_memory = self.node_data.get_memory_usage_mb()
        csr_memory = self.csr_storage.get_memory_usage_mb()
        
        # Children table memory
        children_memory = (self.children.element_size() * self.children.numel()) / (1024 * 1024)
        
        # Update stats with memory reallocations from components
        self.stats['memory_reallocations'] = (
            getattr(self.node_data, 'memory_reallocations', 0) +
            getattr(self.csr_storage, 'memory_reallocations', 0)
        )
        
        return {
            'node_data_mb': node_memory,
            'csr_structure_mb': csr_memory,
            'children_table_mb': children_memory,
            'total_mb': node_memory + csr_memory + children_memory,
            'nodes': self.num_nodes,
            'edges': self.num_edges,
            'bytes_per_node': (node_memory + csr_memory) * 1024 * 1024 / max(1, self.num_nodes)
        }
        
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        memory_stats = self.get_memory_usage()
        
        # Update stats with memory reallocations before returning
        self.stats['memory_reallocations'] = (
            getattr(self.node_data, 'memory_reallocations', 0) +
            getattr(self.csr_storage, 'memory_reallocations', 0)
        )
        
        return {
            **self.stats,
            **memory_stats,
            'edge_utilization': self.csr_storage.get_edge_utilization(),
            'batch_enabled': self.config.enable_batched_ops
        }
        
    # Additional methods for full compatibility
    
    def batch_check_has_children(self, node_indices: torch.Tensor) -> torch.Tensor:
        """Check which nodes have children"""
        if node_indices.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=node_indices.device)
            
        # Bounds checking
        max_valid_index = self.children.shape[0] - 1
        valid_mask = (node_indices >= 0) & (node_indices <= max_valid_index)
        
        if not valid_mask.all():
            # Handle invalid indices
            result = torch.zeros(node_indices.shape[0], dtype=torch.bool, device=node_indices.device)
            valid_indices = node_indices[valid_mask]
            if valid_indices.numel() > 0:
                children_rows = self.children[valid_indices]
                valid_has_children = (children_rows >= 0).any(dim=1)
                result[valid_mask] = valid_has_children
            return result
            
        # All indices valid
        children_rows = self.children[node_indices]
        has_children_mask = (children_rows >= 0).any(dim=1)
        return has_children_mask
        
    def get_node_data(self, node_idx: int, fields: List[str]) -> Dict[str, torch.Tensor]:
        """Get node data for specified fields"""
        result = {}
        for field in fields:
            if field == 'visits':
                visits = self.node_data.visit_counts[node_idx]
                if self.config.enable_virtual_loss:
                    visits = visits + self.node_data.virtual_loss_counts[node_idx]
                result['visits'] = visits
            elif field == 'value':
                result['value'] = torch.tensor(self.node_data.get_q_value(node_idx), device=self.device)
            elif field == 'prior':
                result['prior'] = self.node_data.node_priors[node_idx]
            elif field == 'expanded':
                # Check if node has children
                children = self.children[node_idx]
                has_children = (children >= 0).any()
                result['expanded'] = torch.tensor([has_children], device=self.device)
            else:
                raise ValueError(f"Unknown field: {field}")
        return result
        
    def get_children_batch(self, node_indices: torch.Tensor) -> torch.Tensor:
        """Compatibility method that returns only child indices"""
        children, _, _ = self.batch_get_children(node_indices)
        return children
        
    def add_children(self, parent_idx: int, actions: List[int], priors: List[float]) -> torch.Tensor:
        """Add children to a node (compatibility wrapper)"""
        children = self.add_children_batch(parent_idx, actions, priors)
        return torch.tensor(children, device=self.device, dtype=torch.int32)
        
    def shift_root(self, new_root_idx: int) -> Dict[int, int]:
        """Shift root to a child node, preserving the subtree
        
        This operation:
        1. Makes the specified node the new root (index 0)
        2. Preserves all nodes in the subtree rooted at new_root_idx
        3. Discards all other nodes
        4. Remaps indices to be contiguous starting from 0
        5. Updates all internal references (parent/child relationships)
        
        Args:
            new_root_idx: Index of the node to become the new root
            
        Returns:
            Dictionary mapping old node indices to new indices
        """
        if new_root_idx == 0:
            # Already at root, no shift needed
            return {i: i for i in range(self.num_nodes)}
            
        if new_root_idx < 0 or new_root_idx >= self.num_nodes:
            raise ValueError(f"Invalid new root index: {new_root_idx}")
            
        # Phase 1: Identify all nodes in the subtree using BFS
        # Use torch tensors for better performance
        # We need to handle the case where children indices might be larger than num_nodes
        # This can happen if the tree structure is inconsistent
        max_child_idx = 0
        if self.num_nodes > 0:
            # Find the maximum child index to ensure our arrays are large enough
            valid_children_mask = self.children[:self.num_nodes] >= 0
            if valid_children_mask.any():
                max_child_idx = self.children[:self.num_nodes][valid_children_mask].max().item()
        
        max_possible_nodes = max(self.num_nodes, max_child_idx + 1)
        nodes_to_keep = torch.zeros(max_possible_nodes, dtype=torch.int32, device=self.device)
        visited = torch.zeros(max_possible_nodes, dtype=torch.bool, device=self.device)
        old_to_new_tensor = torch.full((max_possible_nodes,), -1, dtype=torch.int32, device=self.device)
        
        # BFS using tensors
        nodes_to_keep[0] = new_root_idx
        visited[new_root_idx] = True
        old_to_new_tensor[new_root_idx] = 0
        num_kept = 1
        
        read_ptr = 0
        while read_ptr < num_kept:
            current_node = nodes_to_keep[read_ptr].item()
            read_ptr += 1
            
            # Get children efficiently
            children_slice = self.children[current_node]
            valid_children = children_slice[children_slice >= 0]
            
            # Add unvisited children
            for child in valid_children:
                child_idx = child.item()
                if not visited[child_idx]:
                    visited[child_idx] = True
                    nodes_to_keep[num_kept] = child_idx
                    old_to_new_tensor[child_idx] = num_kept
                    num_kept += 1
                    
        # Trim to actual size
        nodes_to_keep = nodes_to_keep[:num_kept]
        
        # Convert to dictionary for compatibility
        old_to_new = {}
        for i in range(num_kept):
            old_idx = nodes_to_keep[i].item()
            old_to_new[old_idx] = i
                    
        num_kept = len(nodes_to_keep)
        
        # Phase 2: Create temporary storage for remapped data
        # We need to copy data to avoid overwriting during remapping
        device = self.device
        
        # Node data
        new_visit_counts = torch.zeros(num_kept, device=device, dtype=torch.int32)
        new_value_sums = torch.zeros(num_kept, device=device, dtype=self.config.dtype_values)
        new_node_priors = torch.zeros(num_kept, device=device, dtype=self.config.dtype_values)
        new_parent_indices = torch.full((num_kept,), -2, device=device, dtype=torch.int32)
        new_parent_actions = torch.full((num_kept,), -1, device=device, dtype=self.config.dtype_indices)
        new_flags = torch.zeros(num_kept, device=device, dtype=torch.uint8)
        new_phases = torch.zeros(num_kept, device=device, dtype=self.config.dtype_values)
        new_virtual_loss_counts = torch.zeros(num_kept, device=device, dtype=torch.int32)
        
        # Vectorized copy of node data
        new_indices = torch.arange(num_kept, device=device, dtype=torch.int32)
        
        # Direct copy using tensor indexing
        new_visit_counts[new_indices] = self.node_data.visit_counts[nodes_to_keep]
        new_value_sums[new_indices] = self.node_data.value_sums[nodes_to_keep]
        new_node_priors[new_indices] = self.node_data.node_priors[nodes_to_keep]
        new_flags[new_indices] = self.node_data.flags[nodes_to_keep]
        new_phases[new_indices] = self.node_data.phases[nodes_to_keep]
        new_virtual_loss_counts[new_indices] = self.node_data.virtual_loss_counts[nodes_to_keep]
        new_parent_actions[new_indices] = self.node_data.parent_actions[nodes_to_keep]
        
        # Remap parent indices using vectorized operations
        old_parents = self.node_data.parent_indices[nodes_to_keep]
        # Map parent indices, handling -1/-2 values
        remapped_parents = torch.where(
            old_parents >= 0,
            old_to_new_tensor[old_parents],
            torch.full_like(old_parents, -1)
        )
        new_parent_indices[:] = remapped_parents
        # Ensure root's parent is -1
        new_parent_indices[0] = -1
            
        # Phase 3: Update children lookup table
        new_children = torch.full((num_kept, self.config.max_actions), -1, 
                                 device=device, dtype=torch.int32)
        
        # Vectorized children table update
        for i in range(num_kept):
            old_idx = nodes_to_keep[i].item()
            new_idx = i
            
            # Get children for this node
            old_children = self.children[old_idx]
            valid_mask = old_children >= 0
            valid_children = old_children[valid_mask]
            
            if valid_children.numel() > 0:
                # Remap child indices
                remapped_children = old_to_new_tensor[valid_children]
                # Filter out children not in subtree (-1 values)
                kept_mask = remapped_children >= 0
                kept_children = remapped_children[kept_mask]
                
                # Update new children table
                num_kept_children = kept_children.numel()
                if num_kept_children > 0:
                    new_children[new_idx, :num_kept_children] = kept_children
                    
        # Phase 4: Rebuild CSR structure
        # Count edges
        new_num_edges = 0
        for i in range(num_kept):
            new_num_edges += (new_children[i] >= 0).sum().item()
            
        # Build new CSR storage
        new_row_ptr = torch.zeros(num_kept + 1, device=device, dtype=torch.int32)
        new_col_indices = torch.zeros(new_num_edges, device=device, dtype=torch.int32)
        new_edge_actions = torch.zeros(new_num_edges, device=device, dtype=self.config.dtype_actions)
        new_edge_priors = torch.zeros(new_num_edges, device=device, dtype=self.config.dtype_values)
        
        edge_idx = 0
        for node_idx in range(num_kept):
            new_row_ptr[node_idx] = edge_idx
            valid_children = new_children[node_idx][new_children[node_idx] >= 0]
            
            for child_idx in valid_children:
                new_col_indices[edge_idx] = child_idx
                new_edge_actions[edge_idx] = new_parent_actions[child_idx]
                new_edge_priors[edge_idx] = new_node_priors[child_idx]
                edge_idx += 1
                
        new_row_ptr[num_kept] = edge_idx
        
        # Phase 5: Update game states
        new_node_states = {}
        for old_idx, new_idx in old_to_new.items():
            if old_idx in self.node_states:
                new_node_states[new_idx] = self.node_states[old_idx]
                
        # Phase 6: Replace all data structures
        # Reset tree first to clear old data
        self.reset()
        
        # Update tree metadata
        self.num_nodes = num_kept
        self.num_edges = new_num_edges
        
        # Update node data
        self.node_data.num_nodes = num_kept
        self.node_data.visit_counts[:num_kept] = new_visit_counts
        self.node_data.value_sums[:num_kept] = new_value_sums
        self.node_data.node_priors[:num_kept] = new_node_priors
        self.node_data.parent_indices[:num_kept] = new_parent_indices
        self.node_data.parent_actions[:num_kept] = new_parent_actions
        self.node_data.flags[:num_kept] = new_flags
        self.node_data.phases[:num_kept] = new_phases
        self.node_data.virtual_loss_counts[:num_kept] = new_virtual_loss_counts
        
        # Update CSR storage
        self.csr_storage.num_edges = new_num_edges
        self.csr_storage.row_ptr[:num_kept + 1] = new_row_ptr
        self.csr_storage.col_indices[:new_num_edges] = new_col_indices
        self.csr_storage.edge_actions[:new_num_edges] = new_edge_actions
        self.csr_storage.edge_priors[:new_num_edges] = new_edge_priors
        
        # Update children table
        self.children[:num_kept] = new_children
        
        # Update game states
        self.node_states = new_node_states
        
        # Update counters
        self.node_counter[0] = self.num_nodes
        self.edge_counter[0] = self.num_edges
        
        # Clear the root's parent info (should already be -1)
        self.node_data.parent_indices[0] = -1
        self.node_data.parent_actions[0] = -1
        
        return old_to_new
        
    def get_child_by_action(self, node_idx: int, action: int) -> Optional[int]:
        """Get child node index for a given action"""
        children, actions, _ = self.get_children(node_idx)
        for i, child_action in enumerate(actions):
            if child_action.item() == action:
                return children[i].item()
        return None
        
    def remove_children(self, parent_idx: int, child_indices_to_remove: List[int]) -> None:
        """Remove specific children from a parent node
        
        This method removes children from the CSR tree structure by:
        1. Filtering out invalid children from the children table
        2. Updating CSR storage to exclude them
        3. Preserving tree consistency
        
        Args:
            parent_idx: Parent node index
            child_indices_to_remove: List of child node indices to remove
        """
        if not child_indices_to_remove:
            return
            
        # Convert to set for efficient lookup
        remove_set = set()
        for child_idx in child_indices_to_remove:
            if hasattr(child_idx, 'item'):
                remove_set.add(child_idx.item())
            else:
                remove_set.add(child_idx)
        
        # Update children table if it exists (action-indexed table)
        if hasattr(self, 'children'):
            # For each action, check if the child should be removed
            for action in range(self.children.shape[1]):
                child_idx = self.children[parent_idx, action]
                if child_idx.item() != -1 and child_idx.item() in remove_set:
                    # Remove this child by setting it to -1
                    self.children[parent_idx, action] = -1
        
        # Update CSR storage by rebuilding the parent's edges
        if hasattr(self.csr_storage, 'row_ptr'):
            # Get CSR range for parent
            start = self.csr_storage.row_ptr[parent_idx].item()
            end = self.csr_storage.row_ptr[parent_idx + 1].item()
            
            if start < end:
                # Get current edges
                edge_children = self.csr_storage.col_indices[start:end]
                edge_actions = self.csr_storage.edge_actions[start:end]
                edge_priors = self.csr_storage.edge_priors[start:end]
                
                # Filter out edges to remove
                valid_edges = []
                for i in range(end - start):
                    child_idx = edge_children[i].item()
                    if child_idx not in remove_set:
                        valid_edges.append(i)
                
                # Update the parent's edges in place
                for i, valid_idx in enumerate(valid_edges):
                    if start + i < self.csr_storage.col_indices.shape[0]:
                        self.csr_storage.col_indices[start + i] = edge_children[valid_idx]
                        self.csr_storage.edge_actions[start + i] = edge_actions[valid_idx]
                        self.csr_storage.edge_priors[start + i] = edge_priors[valid_idx]
                
                # Mark remaining slots as invalid
                for i in range(len(valid_edges), end - start):
                    if start + i < self.csr_storage.col_indices.shape[0]:
                        self.csr_storage.col_indices[start + i] = -1
                        self.csr_storage.edge_actions[start + i] = -1
                        self.csr_storage.edge_priors[start + i] = 0.0
                
                # Update the row pointer for this parent
                self.csr_storage.row_ptr[parent_idx + 1] = start + len(valid_edges)
                
                # Shift row pointers for subsequent nodes
                removed_count = (end - start) - len(valid_edges)
                if removed_count > 0:
                    self.csr_storage.row_ptr[parent_idx + 2:] -= removed_count
                    self.num_edges -= removed_count
                    self.edge_counter[0] = self.num_edges
        
        # Clear parent references for removed children  
        for child_idx in child_indices_to_remove:
            if child_idx < self.node_data.parent_indices.shape[0]:
                self.node_data.parent_indices[child_idx] = -1
                # Note: We don't clear parent_actions to -1 because get_children() 
                # uses parent_actions to return the action values
        
        logger.debug(f"Removed {len(child_indices_to_remove)} children from node {parent_idx}")
        
    def validate_statistics(self, level=None, check_interval: int = 1000):
        """Validate tree statistics (placeholder)"""
        try:
            from ..utils.validation import validate_mcts_tree, ValidationLevel
            if level is None:
                level = ValidationLevel.STANDARD
            return validate_mcts_tree(self, level, check_interval)
        except ImportError:
            class MockResult:
                def __init__(self):
                    self.passed = True
                    self.issues = []
                    self.details = {}
            return MockResult()
            
    @property
    def node_visits(self):
        """Alias for visit_counts for compatibility"""
        return self.visit_counts
        
    @node_visits.setter
    def node_visits(self, value):
        """Setter for compatibility"""
        self.node_data.visit_counts = value
        
    @property
    def is_terminal(self) -> torch.Tensor:
        """Get terminal flags as tensor"""
        return (self.node_data.flags & 2).bool()
        
    @property
    def is_expanded(self) -> torch.Tensor:
        """Get expanded flags as tensor"""
        return (self.node_data.flags & 1).bool()
        
    def batch_backup_optimized(self, paths: torch.Tensor, values: torch.Tensor):
        """Optimized backup using GPU kernels if available"""
        if self.batch_ops is not None and hasattr(self.batch_ops, 'batch_backup'):
            # Use GPU kernel
            self.batch_ops.batch_backup(paths, values, self.node_data.visit_counts, 
                                       self.node_data.value_sums)
        else:
            # Fallback implementation
            batch_size, max_depth = paths.shape
            for i in range(batch_size):
                path = paths[i]
                value = values[i]
                valid_nodes = path[path >= 0]
                self.node_data.batch_update_visits(valid_nodes, torch.ones_like(valid_nodes))
                self.node_data.batch_update_values(valid_nodes, 
                                                 torch.full_like(valid_nodes, value, 
                                                               dtype=self.config.dtype_values))
                                                               
    def _pytorch_ucb_selection_fast(self, node_indices: torch.Tensor, c_puct: float, 
                                   temperature: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fast PyTorch UCB selection for small batches"""
        batch_size = len(node_indices)
        selected_actions = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        ucb_scores = torch.zeros(batch_size, device=self.device)
        
        for i, node_idx in enumerate(node_indices):
            node_idx = node_idx.item()
            
            # Use CSR storage if row pointers are consistent
            if not self.csr_storage.needs_row_ptr_update():
                start = self.csr_storage.row_ptr[node_idx].item()
                end = self.csr_storage.row_ptr[node_idx + 1].item()
                
                if start == end:
                    selected_actions[i] = -1
                    ucb_scores[i] = -float('inf')
                    continue
                    
                # Get children data from CSR
                child_indices = self.csr_storage.col_indices[start:end]
                child_visits = self.node_data.visit_counts[child_indices].float()
                child_values = self.node_data.value_sums[child_indices]
                child_priors = self.csr_storage.edge_priors[start:end]
                
                # Compute Q values
                q_values = torch.where(
                    child_visits > 0,
                    child_values / child_visits,
                    torch.zeros_like(child_values)
                )
                
                # Compute UCB scores
                parent_visits = self.node_data.visit_counts[node_idx].float()
                exploration = c_puct * child_priors * torch.sqrt(parent_visits) / (1 + child_visits)
                child_ucb_scores = q_values + exploration
                
                # Select best action
                best_idx = child_ucb_scores.argmax()
                selected_actions[i] = self.csr_storage.edge_actions[start + best_idx]
                ucb_scores[i] = child_ucb_scores[best_idx]
            else:
                # Fallback to children table
                selected_actions[i] = -1
                ucb_scores[i] = -float('inf')
                
        return selected_actions, ucb_scores