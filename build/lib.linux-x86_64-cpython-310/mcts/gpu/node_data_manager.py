"""Node data management for CSR Tree

This module handles node-level data storage and operations,
separating node statistics from tree structure management.
"""

import torch
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class NodeDataConfig:
    """Configuration for node data storage"""
    max_nodes: int = 100000
    device: str = 'cuda'
    dtype_values: torch.dtype = torch.float32
    dtype_indices: torch.dtype = torch.int32
    initial_capacity_factor: float = 0.1
    growth_factor: float = 1.5
    enable_virtual_loss: bool = True
    virtual_loss_value: float = -1.0


class NodeDataManager:
    """Manages node-level data for MCTS tree
    
    Responsibilities:
    - Node statistics (visits, values, priors)
    - Virtual loss tracking
    - Node flags and metadata
    - Memory management for node data
    """
    
    def __init__(self, config: NodeDataConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Initialize storage
        self._initialize_storage()
        
        # Track current number of nodes
        self.num_nodes = 0
        
    def _initialize_storage(self):
        """Initialize node data tensors"""
        # Determine initial capacity
        if self.config.max_nodes > 0:
            # Ensure at least the full capacity is available if small
            if self.config.max_nodes <= 100:
                initial_size = self.config.max_nodes
            else:
                initial_size = max(10, int(self.config.max_nodes * self.config.initial_capacity_factor))
        else:
            # For unlimited growth, check if we have a very small initial capacity factor
            if self.config.initial_capacity_factor < 0.01:
                initial_size = max(10, int(1000 * self.config.initial_capacity_factor))
            else:
                initial_size = self._calculate_initial_size()
            
        # Node statistics
        self.visit_counts = torch.zeros(initial_size, device=self.device, dtype=torch.int32)
        self.value_sums = torch.zeros(initial_size, device=self.device, dtype=self.config.dtype_values)
        self.node_priors = torch.zeros(initial_size, device=self.device, dtype=self.config.dtype_values)
        
        # Virtual loss for parallelization
        self.virtual_loss_counts = torch.zeros(initial_size, device=self.device, dtype=torch.int32)
        
        # RAVE statistics (for each node, we need stats for all possible actions)
        # We'll use a dictionary to store RAVE data efficiently
        self.rave_visits = {}  # node_idx -> tensor of action visit counts
        self.rave_values = {}  # node_idx -> tensor of action value sums
        
        # Node metadata
        self.flags = torch.zeros(initial_size, device=self.device, dtype=torch.uint8)
        self.phases = torch.zeros(initial_size, device=self.device, dtype=self.config.dtype_values)
        
        # Parent information
        self.parent_indices = torch.full((initial_size,), -2, device=self.device, dtype=torch.int32)
        self.parent_actions = torch.full((initial_size,), -1, device=self.device, dtype=self.config.dtype_indices)
        
    def _calculate_initial_size(self) -> int:
        """Calculate initial size based on available memory"""
        if self.device.type == 'cuda':
            try:
                free_memory = torch.cuda.get_device_properties(self.device).total_memory - torch.cuda.memory_allocated(self.device)
                memory_budget = free_memory * 0.3
                bytes_per_node = 60  # Rough estimate
                max_affordable = int(memory_budget / bytes_per_node)
                return min(75000, max(20000, max_affordable))
            except:
                return 75000
        return 75000
        
    def allocate_node(self, prior: float, parent_idx: int = -1, parent_action: int = -1) -> int:
        """Allocate a new node and return its index"""
        if self.num_nodes >= len(self.visit_counts):
            self._grow_storage()
            
        idx = self.num_nodes
        self.num_nodes += 1
        
        # Initialize node data
        self.visit_counts[idx] = 0
        self.value_sums[idx] = 0.0
        self.node_priors[idx] = prior
        self.parent_indices[idx] = parent_idx
        self.parent_actions[idx] = parent_action
        self.flags[idx] = 0
        self.phases[idx] = 0.0
        self.virtual_loss_counts[idx] = 0
        
        return idx
    
    def initialize_rave_for_node(self, node_idx: int, action_space_size: int):
        """Initialize RAVE statistics for a node"""
        if node_idx not in self.rave_visits:
            self.rave_visits[node_idx] = torch.zeros(
                action_space_size, device=self.device, dtype=torch.int32
            )
            self.rave_values[node_idx] = torch.zeros(
                action_space_size, device=self.device, dtype=self.config.dtype_values
            )
    
    def update_rave(self, node_idx: int, action: int, value: float):
        """Update RAVE statistics for a node-action pair"""
        if node_idx in self.rave_visits:
            self.rave_visits[node_idx][action] += 1
            self.rave_values[node_idx][action] += value
    
    def get_rave_stats(self, node_idx: int, action: int) -> tuple:
        """Get RAVE statistics for a node-action pair"""
        if node_idx not in self.rave_visits:
            return 0, 0.0
        visits = self.rave_visits[node_idx][action].item()
        value_sum = self.rave_values[node_idx][action].item()
        return visits, value_sum
        
    def allocate_nodes_batch(self, count: int, priors: torch.Tensor, 
                           parent_idx: int, parent_actions: torch.Tensor) -> torch.Tensor:
        """Allocate multiple nodes in batch"""
        while self.num_nodes + count > len(self.visit_counts):
            self._grow_storage()
            
        start_idx = self.num_nodes
        end_idx = start_idx + count
        indices = torch.arange(start_idx, end_idx, device=self.device, dtype=torch.int32)
        
        # Vectorized initialization
        self.visit_counts[indices] = 0
        self.value_sums[indices] = 0.0
        self.node_priors[indices] = priors
        self.parent_indices[indices] = parent_idx
        self.parent_actions[indices] = parent_actions
        self.flags[indices] = 0
        self.phases[indices] = 0.0
        self.virtual_loss_counts[indices] = 0
        
        self.num_nodes = end_idx
        return indices
        
    def _grow_storage(self):
        """Grow storage capacity when needed"""
        current_size = len(self.visit_counts)
        new_size = int(current_size * self.config.growth_factor)
        
        if self.config.max_nodes > 0:
            new_size = min(new_size, self.config.max_nodes)
            if new_size <= current_size:
                raise RuntimeError(f"Cannot grow beyond max_nodes={self.config.max_nodes}")
                
        # Create new tensors
        new_visit_counts = torch.zeros(new_size, device=self.device, dtype=torch.int32)
        new_value_sums = torch.zeros(new_size, device=self.device, dtype=self.config.dtype_values)
        new_node_priors = torch.zeros(new_size, device=self.device, dtype=self.config.dtype_values)
        new_virtual_loss_counts = torch.zeros(new_size, device=self.device, dtype=torch.int32)
        new_parent_indices = torch.full((new_size,), -2, device=self.device, dtype=torch.int32)
        new_parent_actions = torch.full((new_size,), -1, device=self.device, dtype=self.config.dtype_indices)
        new_flags = torch.zeros(new_size, device=self.device, dtype=torch.uint8)
        new_phases = torch.zeros(new_size, device=self.device, dtype=self.config.dtype_values)
        
        # Copy existing data
        new_visit_counts[:current_size] = self.visit_counts
        new_value_sums[:current_size] = self.value_sums
        new_node_priors[:current_size] = self.node_priors
        new_virtual_loss_counts[:current_size] = self.virtual_loss_counts
        new_parent_indices[:current_size] = self.parent_indices
        new_parent_actions[:current_size] = self.parent_actions
        new_flags[:current_size] = self.flags
        new_phases[:current_size] = self.phases
        
        # Replace tensors
        self.visit_counts = new_visit_counts
        self.value_sums = new_value_sums
        self.node_priors = new_node_priors
        self.virtual_loss_counts = new_virtual_loss_counts
        self.parent_indices = new_parent_indices
        self.parent_actions = new_parent_actions
        self.flags = new_flags
        self.phases = new_phases
        
        # Track reallocation
        self.memory_reallocations = getattr(self, 'memory_reallocations', 0) + 1
        
    def update_visit_count(self, node_idx: int, delta: int = 1):
        """Update visit count for a node"""
        self.visit_counts[node_idx] += delta
        
    def update_value_sum(self, node_idx: int, value: float):
        """Update value sum for a node"""
        self.value_sums[node_idx] += value
        
    def batch_update_visits(self, node_indices: torch.Tensor, deltas: torch.Tensor):
        """Update visit counts for multiple nodes"""
        self.visit_counts[node_indices] += deltas
        
    def batch_update_values(self, node_indices: torch.Tensor, values: torch.Tensor):
        """Update value sums for multiple nodes"""
        self.value_sums[node_indices] += values
        
    def get_q_value(self, node_idx: int) -> float:
        """Get Q-value (average value) for a node"""
        visits = self.visit_counts[node_idx].item()
        if visits == 0:
            return 0.0
        return (self.value_sums[node_idx] / visits).item()
        
    def get_q_values_batch(self, node_indices: torch.Tensor) -> torch.Tensor:
        """Get Q-values for multiple nodes"""
        visits = self.visit_counts[node_indices].float()
        return torch.where(visits > 0, self.value_sums[node_indices] / visits, torch.zeros_like(visits))
        
    def apply_virtual_loss(self, node_indices: torch.Tensor):
        """Apply virtual loss to nodes"""
        if self.config.enable_virtual_loss:
            self.virtual_loss_counts[node_indices] += 1
            
    def remove_virtual_loss(self, node_indices: torch.Tensor):
        """Remove virtual loss from nodes"""
        if self.config.enable_virtual_loss:
            self.virtual_loss_counts[node_indices] = torch.maximum(
                self.virtual_loss_counts[node_indices] - 1,
                torch.zeros_like(self.virtual_loss_counts[node_indices])
            )
            
    def get_effective_visits(self, node_indices: torch.Tensor) -> torch.Tensor:
        """Get effective visit counts including virtual loss"""
        if self.config.enable_virtual_loss:
            return self.visit_counts[node_indices] + self.virtual_loss_counts[node_indices]
        return self.visit_counts[node_indices]
        
    def get_effective_values(self, node_indices: torch.Tensor) -> torch.Tensor:
        """Get effective value sums including virtual loss"""
        if self.config.enable_virtual_loss:
            virtual_value = self.virtual_loss_counts[node_indices].float() * self.config.virtual_loss_value
            return self.value_sums[node_indices] + virtual_value
        return self.value_sums[node_indices]
        
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
            
    def is_terminal(self, node_idx: int) -> bool:
        """Check if node is terminal"""
        return bool(self.flags[node_idx] & 2)
        
    def is_expanded(self, node_idx: int) -> bool:
        """Check if node is expanded"""
        return bool(self.flags[node_idx] & 1)
        
    def reset(self):
        """Reset all node data"""
        if self.num_nodes > 0:
            # Only clear used portion
            self.visit_counts[:self.num_nodes] = 0
            self.value_sums[:self.num_nodes] = 0.0
            self.node_priors[:self.num_nodes] = 0.0
            self.parent_indices[:self.num_nodes] = -2
            self.parent_actions[:self.num_nodes] = -1
            self.flags[:self.num_nodes] = 0
            self.phases[:self.num_nodes] = 0.0
            self.virtual_loss_counts[:self.num_nodes] = 0
        else:
            self.parent_indices.fill_(-2)
            
        self.num_nodes = 0
        
    def get_memory_usage_mb(self) -> float:
        """Get total memory usage in MB"""
        def tensor_mb(tensor):
            return tensor.element_size() * tensor.numel() / (1024 * 1024)
            
        total = (tensor_mb(self.visit_counts) + tensor_mb(self.value_sums) + 
                tensor_mb(self.node_priors) + tensor_mb(self.virtual_loss_counts) +
                tensor_mb(self.parent_indices) + tensor_mb(self.parent_actions) +
                tensor_mb(self.flags) + tensor_mb(self.phases))
        return total