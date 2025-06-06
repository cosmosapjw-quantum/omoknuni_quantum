"""Memory pool for zero-allocation MCTS

This implementation provides pre-allocated memory pools to eliminate
allocation overhead during MCTS search.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from collections import deque
import threading

logger = logging.getLogger(__name__)

@dataclass
class MemoryPoolConfig:
    """Configuration for memory pooling"""
    # Pool sizes
    tensor_pool_size_mb: int = 512
    numpy_pool_size_mb: int = 256
    object_pool_size: int = 10000
    
    # Tensor specifications
    common_tensor_shapes: List[Tuple[int, ...]] = None
    tensor_dtype: torch.dtype = torch.float32
    device: str = 'cuda'
    
    # Management
    enable_statistics: bool = True
    enable_defragmentation: bool = True
    defrag_threshold: float = 0.8  # Defrag when 80% fragmented
    
    def __post_init__(self):
        if self.common_tensor_shapes is None:
            # Common shapes for MCTS
            self.common_tensor_shapes = [
                (256,),           # Scalar batches
                (256, 225),       # Policy outputs
                (256, 15, 15),    # Gomoku boards
                (256, 100),       # Paths
                (1024,),          # Large scalar batches
                (1024, 225),      # Large policy batches
                (2048, 100),      # Wave paths
            ]

class TensorPool:
    """Pre-allocated tensor pool with recycling"""
    
    def __init__(self, config: MemoryPoolConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Pools organized by shape
        self.pools: Dict[Tuple[int, ...], deque] = {}
        self.allocated: Dict[Tuple[int, ...], List[torch.Tensor]] = {}
        
        # Statistics
        self.stats = {
            'allocations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'current_allocated_mb': 0,
            'peak_allocated_mb': 0
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Pre-allocate common shapes
        self._preallocate_common_shapes()
        
    def _preallocate_common_shapes(self):
        """Pre-allocate tensors for common shapes"""
        total_bytes = 0
        target_bytes = self.config.tensor_pool_size_mb * 1024 * 1024
        
        logger.info(f"Pre-allocating {self.config.tensor_pool_size_mb}MB tensor pool")
        
        for shape in self.config.common_tensor_shapes:
            # Estimate how many tensors of this shape to allocate
            tensor_size = np.prod(shape) * 4  # 4 bytes for float32
            num_tensors = min(100, target_bytes // (len(self.config.common_tensor_shapes) * tensor_size))
            
            self.pools[shape] = deque()
            self.allocated[shape] = []
            
            for _ in range(num_tensors):
                tensor = torch.empty(shape, dtype=self.config.tensor_dtype, device=self.device)
                self.pools[shape].append(tensor)
                total_bytes += tensor.numel() * tensor.element_size()
                
        self.stats['current_allocated_mb'] = total_bytes / (1024 * 1024)
        self.stats['peak_allocated_mb'] = self.stats['current_allocated_mb']
        
        logger.info(f"Pre-allocated {self.stats['current_allocated_mb']:.1f}MB in tensor pools")
        
    def acquire(self, shape: Tuple[int, ...], dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Acquire a tensor from the pool"""
        if dtype is None:
            dtype = self.config.tensor_dtype
            
        with self.lock:
            self.stats['allocations'] += 1
            
            # Check if we have this shape in pool
            if shape in self.pools and len(self.pools[shape]) > 0:
                tensor = self.pools[shape].popleft()
                self.stats['cache_hits'] += 1
                
                # Clear tensor (important for correctness)
                tensor.zero_()
                
                # Track allocation
                if shape not in self.allocated:
                    self.allocated[shape] = []
                self.allocated[shape].append(tensor)
                
                return tensor
            else:
                # Cache miss - allocate new tensor
                self.stats['cache_misses'] += 1
                
                tensor = torch.empty(shape, dtype=dtype, device=self.device)
                
                # Track allocation
                if shape not in self.allocated:
                    self.allocated[shape] = []
                    self.pools[shape] = deque()
                self.allocated[shape].append(tensor)
                
                # Update memory stats
                tensor_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
                self.stats['current_allocated_mb'] += tensor_mb
                self.stats['peak_allocated_mb'] = max(
                    self.stats['peak_allocated_mb'],
                    self.stats['current_allocated_mb']
                )
                
                return tensor
                
    def release(self, tensor: torch.Tensor):
        """Release tensor back to pool"""
        shape = tuple(tensor.shape)
        
        with self.lock:
            # Remove from allocated list
            if shape in self.allocated and tensor in self.allocated[shape]:
                self.allocated[shape].remove(tensor)
                
            # Add back to pool
            if shape in self.pools:
                self.pools[shape].append(tensor)
            else:
                # New shape - create pool
                self.pools[shape] = deque([tensor])
                
    def release_all(self):
        """Release all allocated tensors back to pools"""
        with self.lock:
            for shape, tensors in self.allocated.items():
                self.pools[shape].extend(tensors)
            self.allocated.clear()
            
    def clear(self):
        """Clear all pools (free memory)"""
        with self.lock:
            self.pools.clear()
            self.allocated.clear()
            self.stats['current_allocated_mb'] = 0
            
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self.lock:
            total_pooled = sum(len(pool) for pool in self.pools.values())
            total_allocated = sum(len(tensors) for tensors in self.allocated.values())
            
            cache_rate = (self.stats['cache_hits'] / self.stats['allocations'] 
                         if self.stats['allocations'] > 0 else 0)
            
            return {
                'cache_hit_rate': cache_rate,
                'total_pooled_tensors': total_pooled,
                'total_allocated_tensors': total_allocated,
                'current_allocated_mb': self.stats['current_allocated_mb'],
                'peak_allocated_mb': self.stats['peak_allocated_mb'],
                'unique_shapes': len(self.pools)
            }

class NumpyPool:
    """Pre-allocated numpy array pool"""
    
    def __init__(self, config: MemoryPoolConfig):
        self.config = config
        self.pools: Dict[Tuple[int, ...], deque] = {}
        self.allocated: Dict[Tuple[int, ...], List[np.ndarray]] = {}
        self.lock = threading.Lock()
        
        # Pre-allocate common shapes
        self._preallocate_common_shapes()
        
    def _preallocate_common_shapes(self):
        """Pre-allocate arrays for common shapes"""
        # Common numpy shapes for MCTS
        common_shapes = [
            (225,),         # Move probabilities
            (15, 15),       # Gomoku board
            (3, 15, 15),    # Board features
            (100,),         # Path indices
        ]
        
        target_bytes = self.config.numpy_pool_size_mb * 1024 * 1024
        
        for shape in common_shapes:
            array_size = np.prod(shape) * 4  # 4 bytes for float32
            num_arrays = min(50, target_bytes // (len(common_shapes) * array_size))
            
            self.pools[shape] = deque()
            
            for _ in range(num_arrays):
                array = np.empty(shape, dtype=np.float32)
                self.pools[shape].append(array)
                
    def acquire(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """Acquire array from pool"""
        with self.lock:
            if shape in self.pools and len(self.pools[shape]) > 0:
                array = self.pools[shape].popleft()
                array.fill(0)  # Clear array
                
                if shape not in self.allocated:
                    self.allocated[shape] = []
                self.allocated[shape].append(array)
                
                return array
            else:
                # Allocate new array
                array = np.zeros(shape, dtype=dtype)
                
                if shape not in self.allocated:
                    self.allocated[shape] = []
                    self.pools[shape] = deque()
                self.allocated[shape].append(array)
                
                return array
                
    def release(self, array: np.ndarray):
        """Release array back to pool"""
        shape = array.shape
        
        with self.lock:
            if shape in self.allocated and array in self.allocated[shape]:
                self.allocated[shape].remove(array)
                
            if shape in self.pools:
                self.pools[shape].append(array)

class ObjectPool:
    """Generic object pool for Python objects"""
    
    def __init__(self, factory_func, max_size: int = 1000):
        self.factory_func = factory_func
        self.max_size = max_size
        self.pool = deque()
        self.allocated = set()
        self.lock = threading.Lock()
        
    def acquire(self) -> Any:
        """Acquire object from pool"""
        with self.lock:
            if len(self.pool) > 0:
                obj = self.pool.popleft()
                self.allocated.add(id(obj))
                return obj
            else:
                obj = self.factory_func()
                self.allocated.add(id(obj))
                return obj
                
    def release(self, obj: Any):
        """Release object back to pool"""
        with self.lock:
            obj_id = id(obj)
            if obj_id in self.allocated:
                self.allocated.remove(obj_id)
                
                if len(self.pool) < self.max_size:
                    # Reset object if it has a reset method
                    if hasattr(obj, 'reset'):
                        obj.reset()
                    self.pool.append(obj)

class MemoryPoolManager:
    """Central memory pool manager for MCTS"""
    
    def __init__(self, config: MemoryPoolConfig):
        self.config = config
        
        # Initialize pools
        self.tensor_pool = TensorPool(config)
        self.numpy_pool = NumpyPool(config)
        
        # Object pools for common MCTS objects
        self.object_pools = {
            'node_list': ObjectPool(lambda: [], config.object_pool_size),
            'action_list': ObjectPool(lambda: [], config.object_pool_size),
            'path_list': ObjectPool(lambda: [], config.object_pool_size),
        }
        
        logger.info("Memory pool manager initialized")
        
    def acquire_tensor(self, shape: Tuple[int, ...], 
                      dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Acquire tensor from pool"""
        return self.tensor_pool.acquire(shape, dtype)
        
    def release_tensor(self, tensor: torch.Tensor):
        """Release tensor back to pool"""
        self.tensor_pool.release(tensor)
        
    def acquire_numpy(self, shape: Tuple[int, ...], 
                     dtype: np.dtype = np.float32) -> np.ndarray:
        """Acquire numpy array from pool"""
        return self.numpy_pool.acquire(shape, dtype)
        
    def release_numpy(self, array: np.ndarray):
        """Release numpy array back to pool"""
        self.numpy_pool.release(array)
        
    def acquire_list(self, list_type: str = 'node_list') -> List:
        """Acquire list from object pool"""
        if list_type in self.object_pools:
            lst = self.object_pools[list_type].acquire()
            lst.clear()  # Ensure it's empty
            return lst
        else:
            return []
            
    def release_list(self, lst: List, list_type: str = 'node_list'):
        """Release list back to pool"""
        if list_type in self.object_pools:
            self.object_pools[list_type].release(lst)
            
    def reset_frame(self):
        """Reset pools for new MCTS frame/iteration"""
        # Release all allocated tensors back to pools
        self.tensor_pool.release_all()
        
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics"""
        stats = {
            'tensor_pool': self.tensor_pool.get_stats(),
            'numpy_pool': {
                'total_shapes': len(self.numpy_pool.pools),
                'total_pooled': sum(len(pool) for pool in self.numpy_pool.pools.values())
            },
            'object_pools': {
                name: {
                    'pooled': len(pool.pool),
                    'allocated': len(pool.allocated)
                }
                for name, pool in self.object_pools.items()
            }
        }
        
        return stats
        
    def optimize(self):
        """Optimize memory pools based on usage patterns"""
        # This could analyze usage patterns and adjust pool sizes
        # For now, just log statistics
        stats = self.get_stats()
        logger.info(f"Memory pool stats: {stats}")
        
        # Could implement defragmentation here if needed
        pass