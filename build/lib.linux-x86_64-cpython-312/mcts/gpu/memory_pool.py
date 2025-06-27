"""Memory Pool Manager for CSRTree to avoid fragmentation

This module provides a memory pool implementation that pre-allocates
large chunks of memory and manages their reuse to avoid fragmentation
caused by frequent allocation/deallocation cycles.

Features:
- Pre-allocated tensor pools for different data types
- Thread-safe allocation/deallocation
- Automatic growth when pool is exhausted
- Memory usage tracking and statistics
"""

import torch
import threading
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryPoolConfig:
    """Configuration for memory pool"""
    # Initial pool sizes
    initial_node_capacity: int = 1_000_000  # 1M nodes
    initial_edge_capacity: int = 10_000_000  # 10M edges
    
    # Growth factors
    node_growth_factor: float = 1.5
    edge_growth_factor: float = 1.5
    
    # Device configuration
    device: str = 'cuda'
    
    # Pre-allocation settings
    num_pre_allocated_trees: int = 4  # Number of trees to pre-allocate
    
    # Memory limits
    max_memory_mb: int = 4096  # Maximum memory usage in MB
    
    # Defragmentation settings
    defrag_threshold: float = 0.3  # Defragment when 30% fragmented
    enable_auto_defrag: bool = True


class TensorPool:
    """Pool of pre-allocated tensors for a specific data type"""
    
    def __init__(self, dtype: torch.dtype, device: torch.device):
        self.dtype = dtype
        self.device = device
        self.free_tensors: List[torch.Tensor] = []
        self.allocated_tensors: Dict[int, torch.Tensor] = {}
        self.next_id = 0
        self.lock = threading.Lock()
        
        # Statistics
        self.total_allocated = 0
        self.total_deallocated = 0
        self.peak_usage = 0
        
    def allocate(self, size: int) -> Tuple[int, torch.Tensor]:
        """Allocate a tensor of given size from the pool"""
        with self.lock:
            # Try to find a suitable free tensor
            for i, tensor in enumerate(self.free_tensors):
                if tensor.numel() >= size:
                    # Remove from free list
                    self.free_tensors.pop(i)
                    
                    # Resize if needed (view to exact size)
                    if tensor.numel() > size:
                        tensor = tensor[:size]
                    
                    # Allocate
                    tensor_id = self.next_id
                    self.next_id += 1
                    self.allocated_tensors[tensor_id] = tensor
                    
                    # Update statistics
                    self.total_allocated += 1
                    self.peak_usage = max(self.peak_usage, len(self.allocated_tensors))
                    
                    # Zero out the tensor for clean state
                    tensor.zero_()
                    
                    return tensor_id, tensor
            
            # No suitable tensor found, allocate new one
            tensor = torch.zeros(size, dtype=self.dtype, device=self.device)
            tensor_id = self.next_id
            self.next_id += 1
            self.allocated_tensors[tensor_id] = tensor
            
            # Update statistics
            self.total_allocated += 1
            self.peak_usage = max(self.peak_usage, len(self.allocated_tensors))
            
            return tensor_id, tensor
    
    def deallocate(self, tensor_id: int):
        """Return a tensor to the pool"""
        with self.lock:
            if tensor_id in self.allocated_tensors:
                tensor = self.allocated_tensors.pop(tensor_id)
                self.free_tensors.append(tensor)
                self.total_deallocated += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self.lock:
            total_memory = sum(t.element_size() * t.numel() for t in self.free_tensors)
            total_memory += sum(t.element_size() * t.numel() for t in self.allocated_tensors.values())
            
            return {
                'dtype': str(self.dtype),
                'allocated_count': len(self.allocated_tensors),
                'free_count': len(self.free_tensors),
                'total_allocated': self.total_allocated,
                'total_deallocated': self.total_deallocated,
                'peak_usage': self.peak_usage,
                'memory_mb': total_memory / (1024 * 1024)
            }
    
    def defragment(self):
        """Defragment the pool by consolidating free tensors"""
        with self.lock:
            if not self.free_tensors:
                return
            
            # Sort free tensors by size
            self.free_tensors.sort(key=lambda t: t.numel())
            
            # Merge adjacent small tensors if possible
            # For now, just remove very small fragments
            min_size = 1000  # Minimum tensor size to keep
            self.free_tensors = [t for t in self.free_tensors if t.numel() >= min_size]


class CSRTreeMemoryPool:
    """Memory pool manager for CSRTree allocations"""
    
    def __init__(self, config: MemoryPoolConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Create tensor pools for different data types
        self.pools = {
            'int32': TensorPool(torch.int32, self.device),
            'float32': TensorPool(torch.float32, self.device),
            'uint8': TensorPool(torch.uint8, self.device),
        }
        
        # Pre-allocated tree structures
        self.free_trees: List[Dict[str, Any]] = []
        self.allocated_trees: Dict[int, Dict[str, Any]] = {}
        self.next_tree_id = 0
        self.tree_lock = threading.Lock()
        
        # Pre-allocate initial trees
        self._pre_allocate_trees()
        
        # Statistics
        self.stats = {
            'trees_allocated': 0,
            'trees_deallocated': 0,
            'peak_trees': 0,
            'defrag_count': 0
        }
        
        logger.debug(f"Initialized CSRTree memory pool with {config.num_pre_allocated_trees} pre-allocated trees")
    
    def _pre_allocate_trees(self):
        """Pre-allocate tree structures"""
        for _ in range(self.config.num_pre_allocated_trees):
            tree_buffers = self._allocate_tree_buffers(
                self.config.initial_node_capacity,
                self.config.initial_edge_capacity
            )
            self.free_trees.append(tree_buffers)
    
    def _allocate_tree_buffers(self, node_capacity: int, edge_capacity: int) -> Dict[str, Any]:
        """Allocate all buffers needed for a CSRTree"""
        buffers = {}
        
        # Node data buffers
        _, buffers['visit_counts'] = self.pools['int32'].allocate(node_capacity)
        _, buffers['value_sums'] = self.pools['float32'].allocate(node_capacity)
        _, buffers['node_priors'] = self.pools['float32'].allocate(node_capacity)
        _, buffers['virtual_loss_counts'] = self.pools['int32'].allocate(node_capacity)
        _, buffers['parent_indices'] = self.pools['int32'].allocate(node_capacity)
        _, buffers['parent_actions'] = self.pools['int32'].allocate(node_capacity)
        _, buffers['flags'] = self.pools['uint8'].allocate(node_capacity)
        _, buffers['phases'] = self.pools['float32'].allocate(node_capacity)
        
        # Children lookup table (max 512 children per node)
        _, buffers['children'] = self.pools['int32'].allocate(node_capacity * 512)
        buffers['children'] = buffers['children'].view(node_capacity, 512)
        buffers['children'].fill_(-1)
        
        # CSR structure
        _, buffers['row_ptr'] = self.pools['int32'].allocate(node_capacity + 1)
        _, buffers['col_indices'] = self.pools['int32'].allocate(edge_capacity)
        _, buffers['edge_actions'] = self.pools['int32'].allocate(edge_capacity)
        _, buffers['edge_priors'] = self.pools['float32'].allocate(edge_capacity)
        
        # Metadata
        buffers['node_capacity'] = node_capacity
        buffers['edge_capacity'] = edge_capacity
        buffers['num_nodes'] = 0
        buffers['num_edges'] = 0
        
        return buffers
    
    def allocate_tree(self) -> Tuple[int, Dict[str, Any]]:
        """Allocate a tree structure from the pool"""
        with self.tree_lock:
            # Try to get a free tree
            if self.free_trees:
                tree_buffers = self.free_trees.pop()
            else:
                # Allocate new tree
                logger.debug("Memory pool exhausted, allocating new tree buffers")
                tree_buffers = self._allocate_tree_buffers(
                    self.config.initial_node_capacity,
                    self.config.initial_edge_capacity
                )
            
            # Assign ID and track
            tree_id = self.next_tree_id
            self.next_tree_id += 1
            self.allocated_trees[tree_id] = tree_buffers
            
            # Update statistics
            self.stats['trees_allocated'] += 1
            self.stats['peak_trees'] = max(self.stats['peak_trees'], len(self.allocated_trees))
            
            # Check if we should defragment
            if self.config.enable_auto_defrag:
                self._check_defragmentation()
            
            return tree_id, tree_buffers
    
    def deallocate_tree(self, tree_id: int):
        """Return a tree structure to the pool"""
        with self.tree_lock:
            if tree_id in self.allocated_trees:
                tree_buffers = self.allocated_trees.pop(tree_id)
                
                # Clear the buffers for reuse
                self._clear_tree_buffers(tree_buffers)
                
                # Add to free list
                self.free_trees.append(tree_buffers)
                
                # Update statistics
                self.stats['trees_deallocated'] += 1
    
    def _clear_tree_buffers(self, buffers: Dict[str, Any]):
        """Clear tree buffers for reuse"""
        # Zero out all tensors
        for key, value in buffers.items():
            if isinstance(value, torch.Tensor):
                if key == 'parent_indices' or key == 'parent_actions':
                    value.fill_(-1)
                elif key == 'children':
                    value.fill_(-1)
                else:
                    value.zero_()
        
        # Reset metadata
        buffers['num_nodes'] = 0
        buffers['num_edges'] = 0
    
    def grow_tree_buffers(self, tree_id: int, new_node_capacity: int = None, 
                         new_edge_capacity: int = None) -> Dict[str, Any]:
        """Grow the buffers for a specific tree"""
        with self.tree_lock:
            if tree_id not in self.allocated_trees:
                raise ValueError(f"Tree {tree_id} not found in allocated trees")
            
            old_buffers = self.allocated_trees[tree_id]
            
            # Calculate new capacities
            if new_node_capacity is None:
                new_node_capacity = int(old_buffers['node_capacity'] * self.config.node_growth_factor)
            if new_edge_capacity is None:
                new_edge_capacity = int(old_buffers['edge_capacity'] * self.config.edge_growth_factor)
            
            # Allocate new buffers
            new_buffers = self._allocate_tree_buffers(new_node_capacity, new_edge_capacity)
            
            # Copy existing data
            self._copy_tree_data(old_buffers, new_buffers)
            
            # Replace in allocated trees
            self.allocated_trees[tree_id] = new_buffers
            
            # Return old buffers to pools
            self._return_buffers_to_pools(old_buffers)
            
            logger.debug(f"Grew tree {tree_id} buffers: nodes {old_buffers['node_capacity']} -> {new_node_capacity}, "
                        f"edges {old_buffers['edge_capacity']} -> {new_edge_capacity}")
            
            return new_buffers
    
    def _copy_tree_data(self, src: Dict[str, Any], dst: Dict[str, Any]):
        """Copy data from source buffers to destination buffers"""
        num_nodes = src['num_nodes']
        num_edges = src['num_edges']
        
        # Copy node data
        dst['visit_counts'][:num_nodes] = src['visit_counts'][:num_nodes]
        dst['value_sums'][:num_nodes] = src['value_sums'][:num_nodes]
        dst['node_priors'][:num_nodes] = src['node_priors'][:num_nodes]
        dst['virtual_loss_counts'][:num_nodes] = src['virtual_loss_counts'][:num_nodes]
        dst['parent_indices'][:num_nodes] = src['parent_indices'][:num_nodes]
        dst['parent_actions'][:num_nodes] = src['parent_actions'][:num_nodes]
        dst['flags'][:num_nodes] = src['flags'][:num_nodes]
        dst['phases'][:num_nodes] = src['phases'][:num_nodes]
        
        # Copy children table
        if num_nodes > 0:
            dst['children'][:num_nodes] = src['children'][:num_nodes]
        
        # Copy CSR data
        dst['row_ptr'][:num_nodes + 1] = src['row_ptr'][:num_nodes + 1]
        if num_edges > 0:
            dst['col_indices'][:num_edges] = src['col_indices'][:num_edges]
            dst['edge_actions'][:num_edges] = src['edge_actions'][:num_edges]
            dst['edge_priors'][:num_edges] = src['edge_priors'][:num_edges]
        
        # Copy metadata
        dst['num_nodes'] = num_nodes
        dst['num_edges'] = num_edges
    
    def _return_buffers_to_pools(self, buffers: Dict[str, Any]):
        """Return individual buffers to their respective pools"""
        # This is a simplified version - in production, we'd track tensor IDs
        # For now, we just let them be garbage collected
        pass
    
    def _check_defragmentation(self):
        """Check if defragmentation is needed"""
        # Calculate fragmentation ratio
        total_free = len(self.free_trees)
        total_allocated = len(self.allocated_trees)
        total = total_free + total_allocated
        
        if total > 0 and total_free / total > self.config.defrag_threshold:
            self.defragment()
    
    def defragment(self):
        """Defragment all pools"""
        logger.info("Starting memory pool defragmentation")
        
        # Defragment individual tensor pools
        for pool in self.pools.values():
            pool.defragment()
        
        # Consolidate free trees if needed
        # For now, just limit the number of free trees
        max_free_trees = self.config.num_pre_allocated_trees * 2
        if len(self.free_trees) > max_free_trees:
            # Keep only the most recently used trees
            self.free_trees = self.free_trees[-max_free_trees:]
        
        self.stats['defrag_count'] += 1
        logger.info("Memory pool defragmentation completed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory pool statistics"""
        with self.tree_lock:
            pool_stats = {name: pool.get_stats() for name, pool in self.pools.items()}
            
            # Calculate total memory usage
            total_memory_mb = sum(stats['memory_mb'] for stats in pool_stats.values())
            
            # Add tree-level statistics
            tree_stats = {
                'allocated_trees': len(self.allocated_trees),
                'free_trees': len(self.free_trees),
                'total_trees': len(self.allocated_trees) + len(self.free_trees),
                **self.stats
            }
            
            return {
                'pools': pool_stats,
                'trees': tree_stats,
                'total_memory_mb': total_memory_mb,
                'memory_limit_mb': self.config.max_memory_mb,
                'memory_usage_pct': (total_memory_mb / self.config.max_memory_mb) * 100
            }


# Global memory pool instance (singleton)
_global_memory_pool: Optional[CSRTreeMemoryPool] = None
_pool_lock = threading.Lock()


def get_memory_pool(config: Optional[MemoryPoolConfig] = None) -> CSRTreeMemoryPool:
    """Get or create the global memory pool instance"""
    global _global_memory_pool
    
    with _pool_lock:
        if _global_memory_pool is None:
            if config is None:
                # Automatically configure based on available GPU memory and context
                import os
                if os.environ.get('PYTEST_CURRENT_TEST'):
                    # Use smaller pool for testing
                    config = MemoryPoolConfig(
                        initial_node_capacity=50_000,  # 50K nodes for testing
                        initial_edge_capacity=500_000,  # 500K edges for testing
                        num_pre_allocated_trees=2,
                        max_memory_mb=1024  # 1GB limit for testing
                    )
                else:
                    # Smart configuration based on GPU memory
                    config = _get_auto_config()
            _global_memory_pool = CSRTreeMemoryPool(config)
        
        return _global_memory_pool


def reset_memory_pool():
    """Reset the global memory pool (mainly for testing)"""
    global _global_memory_pool
    
    with _pool_lock:
        _global_memory_pool = None


def _get_auto_config() -> MemoryPoolConfig:
    """Automatically configure memory pool based on GPU memory and context"""
    import os
    
    # Get GPU memory info
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        available_memory = total_memory - reserved_memory
        
        # Convert to MB
        available_mb = available_memory // (1024 * 1024)
        
        logger.debug(f"GPU {device}: Total memory: {total_memory // (1024**3):.1f}GB, "
                   f"Available: {available_mb}MB")
    else:
        # CPU fallback
        available_mb = 4096  # 4GB default for CPU
    
    # Check if we're in a worker process
    worker_id = os.environ.get('SELFPLAY_WORKER_ID')
    num_workers = int(os.environ.get('SELFPLAY_NUM_WORKERS', '1'))
    
    if worker_id is not None:
        # Worker process: allocate fraction of memory
        # Reserve some memory for PyTorch operations
        usable_memory = int(available_mb * 0.7)  # Use 70% of available
        per_worker_memory = usable_memory // num_workers
        
        # Scale down allocations for workers
        node_capacity = min(200_000, 50_000 * (per_worker_memory // 256))
        edge_capacity = node_capacity * 10
        
        config = MemoryPoolConfig(
            initial_node_capacity=node_capacity,
            initial_edge_capacity=edge_capacity,
            num_pre_allocated_trees=1,  # One tree per worker
            max_memory_mb=per_worker_memory,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        logger.debug(f"Worker {worker_id}/{num_workers}: Allocated {per_worker_memory}MB, "
                   f"nodes={node_capacity}, edges={edge_capacity}")
    else:
        # Main process: use more memory but leave room for workers
        if num_workers > 1:
            # Leave memory for workers
            main_memory = int(available_mb * 0.3)
        else:
            # Single process, use most memory
            main_memory = int(available_mb * 0.7)
        
        # Determine capacity based on memory
        node_capacity = min(1_000_000, 100_000 * (main_memory // 512))
        edge_capacity = node_capacity * 10
        
        config = MemoryPoolConfig(
            initial_node_capacity=node_capacity,
            initial_edge_capacity=edge_capacity,
            num_pre_allocated_trees=4,
            max_memory_mb=main_memory,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        logger.debug(f"Main process: Allocated {main_memory}MB, "
                   f"nodes={node_capacity}, edges={edge_capacity}")
    
    return config