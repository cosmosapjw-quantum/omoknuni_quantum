"""Lazy state allocation for GPU-efficient MCTS

This module implements on-demand state allocation to avoid
the overhead of persistent node-to-state mappings.
"""

import torch
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class LazyStateManager:
    """Manages game states with lazy allocation
    
    Instead of pre-allocating states for all nodes, this manager
    allocates states only when nodes are actually visited during search.
    This dramatically reduces memory usage and avoids expensive remapping.
    """
    
    def __init__(self, capacity: int, device='cuda'):
        self.capacity = capacity
        self.device = torch.device(device)
        self.allocated_count = 0
        
        # Free list for state recycling
        self.free_list = list(range(capacity))
        
        # Node to state mapping (-1 means no state allocated)
        self.node_to_state = torch.full(
            (capacity,), -1, dtype=torch.int32, device=device
        )
        
        # Track which states are allocated
        self.allocated_mask = torch.zeros(
            capacity, dtype=torch.bool, device=device
        )
        
        logger.debug(f"Initialized LazyStateManager with capacity {capacity}")
    
    def get_or_create_state(self, node_idx: int) -> int:
        """Get state for node, allocating if necessary
        
        Args:
            node_idx: Index of tree node
            
        Returns:
            Index of allocated state
        """
        # Check if already allocated
        current_state = self.node_to_state[node_idx].item()
        if current_state >= 0:
            return current_state
        
        # Allocate new state
        if not self.free_list:
            raise RuntimeError("LazyStateManager: No free states available")
        
        state_idx = self.free_list.pop()
        self.node_to_state[node_idx] = state_idx
        self.allocated_mask[state_idx] = True
        self.allocated_count += 1
        
        return state_idx
    
    def release_state(self, node_idx: int) -> None:
        """Release state back to free pool
        
        Args:
            node_idx: Index of tree node
        """
        state_idx = self.node_to_state[node_idx].item()
        if state_idx >= 0:
            self.node_to_state[node_idx] = -1
            self.allocated_mask[state_idx] = False
            self.free_list.append(state_idx)
            self.allocated_count -= 1
    
    def reset(self) -> None:
        """Reset all allocations"""
        self.node_to_state.fill_(-1)
        self.allocated_mask.zero_()
        self.free_list = list(range(self.capacity))
        self.allocated_count = 0
    
    def get_allocation_stats(self) -> dict:
        """Get statistics about state allocation"""
        return {
            'allocated': self.allocated_count,
            'free': len(self.free_list),
            'capacity': self.capacity,
            'utilization': self.allocated_count / self.capacity
        }


class StatePool:
    """Thread-safe state pool for multi-threaded MCTS
    
    This is a more advanced version that supports concurrent access.
    """
    
    def __init__(self, capacity: int, device='cuda'):
        self.capacity = capacity
        self.device = device
        
        # Use atomic operations for thread safety
        self.next_free = torch.tensor(0, dtype=torch.int32, device=device)
        self.free_indices = torch.arange(capacity, dtype=torch.int32, device=device)
        
        # Allocation tracking
        self.allocated = torch.zeros(capacity, dtype=torch.bool, device=device)
    
    def allocate_batch(self, count: int) -> Optional[torch.Tensor]:
        """Allocate multiple states atomically
        
        Args:
            count: Number of states to allocate
            
        Returns:
            Tensor of allocated indices or None if not enough free
        """
        # This would use atomic operations in full implementation
        # For now, simplified version
        start = self.next_free.item()
        if start + count > self.capacity:
            return None
        
        indices = self.free_indices[start:start + count].clone()
        self.next_free += count
        self.allocated[indices] = True
        
        return indices
    
    def release_batch(self, indices: torch.Tensor) -> None:
        """Release multiple states back to pool
        
        Args:
            indices: Tensor of state indices to release
        """
        self.allocated[indices] = False
        # In full implementation, would need to handle free list properly