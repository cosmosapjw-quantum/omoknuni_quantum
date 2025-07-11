"""Node Memory Pool for efficient memory management in MCTS

This module provides a memory pool for tree nodes to reduce allocation
overhead and improve cache locality.
"""

import torch
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MemoryPoolConfig:
    """Configuration for node memory pool"""
    capacity: int = 500000
    device: str = 'cuda'
    dtype_indices: torch.dtype = torch.int32
    initial_allocation_fraction: float = 0.5  # Pre-allocate 50%
    growth_factor: float = 1.5
    enable_defragmentation: bool = True
    defrag_threshold: float = 0.3  # Defrag when 30% fragmented


class NodeMemoryPool:
    """Efficient node allocation with recycling and defragmentation
    
    Key features:
    1. Pre-allocated memory blocks
    2. Fast O(1) allocation and deallocation
    3. Automatic defragmentation
    4. Cache-friendly memory layout
    """
    
    def __init__(self, config: MemoryPoolConfig):
        """Initialize memory pool
        
        Args:
            config: Memory pool configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        self.capacity = config.capacity
        
        # Pre-allocate memory
        self._init_memory()
        
        # Statistics
        self.stats = {
            'allocations': 0,
            'deallocations': 0,
            'defragmentations': 0,
            'peak_usage': 0,
            'fragmentation_ratio': 0.0
        }
    
    def _init_memory(self):
        """Initialize pre-allocated memory structures"""
        # Free list management (stack-based for O(1) operations)
        self.free_indices = torch.arange(
            self.capacity, device=self.device, dtype=self.config.dtype_indices
        )
        self.free_count = self.capacity
        self.free_top = self.capacity - 1  # Top of free stack
        
        # Allocation tracking
        self.allocated_mask = torch.zeros(
            self.capacity, device=self.device, dtype=torch.bool
        )
        
        # Memory blocks for contiguous allocation
        self.allocation_blocks = []  # List of (start, size) for each allocation
        self.block_map = {}  # Map from allocation ID to block info
        
        # Pre-allocate initial fraction
        initial_count = int(self.capacity * self.config.initial_allocation_fraction)
        if initial_count > 0:
            # Mark initial nodes as "pre-allocated" but still free
            logger.info(f"Pre-allocating {initial_count} nodes in memory pool")
    
    def allocate(self, count: int) -> torch.Tensor:
        """Allocate node indices from pool
        
        Args:
            count: Number of nodes to allocate
            
        Returns:
            Tensor of allocated node indices
            
        Raises:
            RuntimeError: If insufficient nodes available
        """
        if count > self.free_count:
            # Try defragmentation first
            if self.config.enable_defragmentation:
                self._defragment()
                
            if count > self.free_count:
                raise RuntimeError(
                    f"Insufficient nodes: requested {count}, available {self.free_count}"
                )
        
        # Allocate from top of free stack
        start_idx = self.free_top - count + 1
        indices = self.free_indices[start_idx:self.free_top + 1].clone()
        
        # Update free stack
        self.free_top -= count
        self.free_count -= count
        
        # Mark as allocated
        self.allocated_mask[indices] = True
        
        # Track allocation
        self.allocation_blocks.append((indices[0].item(), count))
        
        # Update statistics
        self.stats['allocations'] += 1
        current_usage = self.capacity - self.free_count
        if current_usage > self.stats['peak_usage']:
            self.stats['peak_usage'] = current_usage
        
        return indices
    
    def allocate_contiguous(self, count: int) -> Tuple[torch.Tensor, int]:
        """Allocate contiguous block of nodes for better cache locality
        
        Args:
            count: Number of nodes to allocate
            
        Returns:
            Tuple of (indices tensor, start index)
        """
        if count > self.free_count:
            raise RuntimeError(
                f"Insufficient nodes: requested {count}, available {self.free_count}"
            )
        
        # Try to find contiguous block in free list
        contiguous_start = self._find_contiguous_block(count)
        
        if contiguous_start is not None:
            # Found contiguous block
            indices = torch.arange(
                contiguous_start, contiguous_start + count,
                device=self.device, dtype=self.config.dtype_indices
            )
        else:
            # Fallback to regular allocation
            indices = self.allocate(count)
            contiguous_start = indices[0].item()
        
        return indices, contiguous_start
    
    def free(self, indices: torch.Tensor):
        """Return nodes to pool
        
        Args:
            indices: Node indices to free
        """
        if len(indices) == 0:
            return
        
        # Validate indices
        if not self.allocated_mask[indices].all():
            logger.warning("Attempting to free unallocated nodes")
            # Filter to only allocated nodes
            indices = indices[self.allocated_mask[indices]]
            if len(indices) == 0:
                return
        
        # Mark as free
        self.allocated_mask[indices] = False
        
        # Add back to free stack
        count = len(indices)
        self.free_indices[self.free_top + 1:self.free_top + 1 + count] = indices
        self.free_top += count
        self.free_count += count
        
        # Update statistics
        self.stats['deallocations'] += 1
        
        # Check if defragmentation needed
        self._check_fragmentation()
    
    def bulk_free(self, indices_list: List[torch.Tensor]):
        """Free multiple sets of indices efficiently
        
        Args:
            indices_list: List of index tensors to free
        """
        if not indices_list:
            return
        
        # Concatenate all indices
        all_indices = torch.cat(indices_list)
        
        # Free in one operation
        self.free(all_indices)
    
    def reset(self):
        """Reset pool to initial state"""
        # Reset free list
        self.free_indices = torch.arange(
            self.capacity, device=self.device, dtype=self.config.dtype_indices
        )
        self.free_count = self.capacity
        self.free_top = self.capacity - 1
        
        # Clear allocation tracking
        self.allocated_mask.zero_()
        self.allocation_blocks.clear()
        self.block_map.clear()
        
        # Reset statistics (keep historical data)
        self.stats['fragmentation_ratio'] = 0.0
    
    def _find_contiguous_block(self, count: int) -> Optional[int]:
        """Find contiguous block in free list
        
        Args:
            count: Size of block needed
            
        Returns:
            Start index of contiguous block, or None if not found
        """
        # Simple heuristic: check if top of free stack is contiguous
        if self.free_top >= count - 1:
            top_indices = self.free_indices[self.free_top - count + 1:self.free_top + 1]
            if self._is_contiguous(top_indices):
                return top_indices[0].item()
        
        # TODO: Implement more sophisticated search for larger pools
        return None
    
    def _is_contiguous(self, indices: torch.Tensor) -> bool:
        """Check if indices form a contiguous block"""
        if len(indices) <= 1:
            return True
        
        # Check if sorted indices have step size of 1
        sorted_indices = indices.sort()[0]
        diffs = sorted_indices[1:] - sorted_indices[:-1]
        return (diffs == 1).all()
    
    def _check_fragmentation(self):
        """Check fragmentation level and trigger defrag if needed"""
        if not self.config.enable_defragmentation:
            return
        
        # Calculate fragmentation ratio
        if self.free_count > 0 and self.free_count < self.capacity:
            # Simple metric: how scattered are free indices?
            free_mask = ~self.allocated_mask
            if free_mask.any():
                # Find runs of free nodes
                free_runs = self._count_free_runs(free_mask)
                fragmentation = 1.0 - (1.0 / free_runs) if free_runs > 0 else 0.0
                
                self.stats['fragmentation_ratio'] = fragmentation
                
                if fragmentation > self.config.defrag_threshold:
                    self._defragment()
    
    def _count_free_runs(self, free_mask: torch.Tensor) -> int:
        """Count number of contiguous free runs"""
        # Find transitions in the mask
        padded = torch.cat([
            torch.tensor([False], device=self.device),
            free_mask,
            torch.tensor([False], device=self.device)
        ])
        
        # Count rising edges (False -> True transitions)
        diffs = padded[1:].int() - padded[:-1].int()
        runs = (diffs == 1).sum().item()
        
        return runs
    
    def _defragment(self):
        """Defragment memory pool for better locality"""
        logger.info(f"Defragmenting memory pool (fragmentation: {self.stats['fragmentation_ratio']:.2f})")
        
        # Get all free indices
        free_mask = ~self.allocated_mask
        free_indices = torch.where(free_mask)[0]
        
        if len(free_indices) == 0:
            return
        
        # Sort free indices for better locality
        sorted_free = free_indices.sort()[0]
        
        # Update free list with sorted indices
        self.free_indices[:len(sorted_free)] = sorted_free
        self.free_top = len(sorted_free) - 1
        
        self.stats['defragmentations'] += 1
        
        # Update fragmentation ratio after defragmentation
        # Don't call _check_fragmentation to avoid recursion
        free_mask = ~self.allocated_mask
        if free_mask.any():
            free_runs = self._count_free_runs(free_mask)
            fragmentation = 1.0 - (1.0 / free_runs) if free_runs > 0 else 0.0
            self.stats['fragmentation_ratio'] = fragmentation
    
    def get_statistics(self) -> dict:
        """Get pool statistics"""
        stats = self.stats.copy()
        stats['current_usage'] = self.capacity - self.free_count
        stats['usage_ratio'] = (self.capacity - self.free_count) / self.capacity
        stats['free_count'] = self.free_count
        
        return stats
    
    def compact(self):
        """Compact allocated nodes for better cache performance
        
        This is an advanced operation that remaps allocated nodes
        to be more contiguous in memory.
        """
        # TODO: Implement node remapping for cache optimization
        # This would require updating all references in the tree
        logger.info("Node compaction not yet implemented")


def create_node_memory_pool(
    capacity: int = 500000,
    device: str = 'cuda',
    initial_allocation: float = 0.5
) -> NodeMemoryPool:
    """Factory function to create optimized memory pool
    
    Args:
        capacity: Maximum number of nodes
        device: Device for allocation
        initial_allocation: Fraction to pre-allocate
        
    Returns:
        Configured NodeMemoryPool instance
    """
    config = MemoryPoolConfig(
        capacity=capacity,
        device=device,
        initial_allocation_fraction=initial_allocation,
        enable_defragmentation=True
    )
    
    return NodeMemoryPool(config)