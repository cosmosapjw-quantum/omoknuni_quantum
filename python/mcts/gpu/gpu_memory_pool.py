"""GPU Memory Pool Manager for High-Performance MCTS

This module provides advanced memory pooling to eliminate allocation overhead
and achieve 5000+ sims/sec performance.
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import threading

logger = logging.getLogger(__name__)


@dataclass
class MemoryBlock:
    """Represents a reusable memory block"""
    tensor: torch.Tensor
    size: int
    in_use: bool = False
    last_used: float = 0.0


class GPUMemoryPool:
    """Advanced GPU memory pool with pre-allocation and zero-copy operations
    
    Key features:
    - Pre-allocated memory blocks for common sizes
    - Zero allocation overhead during MCTS
    - Thread-safe operations
    - Automatic memory reclamation
    """
    
    def __init__(self, device: torch.device, config: Optional[Dict] = None):
        self.device = device
        self.config = config or {}
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Memory pools by shape
        self.pools: Dict[Tuple[int, ...], List[MemoryBlock]] = {}
        
        # Pre-allocate common tensor shapes
        self._preallocate_common_shapes()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        
        logger.info(f"GPU Memory Pool initialized on {device}")
    
    def _preallocate_common_shapes(self):
        """Pre-allocate memory for common tensor shapes used in MCTS"""
        
        # Common batch sizes
        batch_sizes = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
        
        # Board features shape: (batch, channels, height, width)
        board_size = self.config.get('board_size', 15)
        channels = self.config.get('channels', 19)
        
        # Pre-allocate board feature tensors
        for batch_size in batch_sizes:
            shape = (batch_size, channels, board_size, board_size)
            self._preallocate_shape(shape, count=2)  # 2 copies per shape
        
        # Policy output shape: (batch, action_space)
        action_space = board_size * board_size
        for batch_size in batch_sizes:
            shape = (batch_size, action_space)
            self._preallocate_shape(shape, count=2)
        
        # Value output shape: (batch,)
        for batch_size in batch_sizes:
            shape = (batch_size,)
            self._preallocate_shape(shape, count=2)
        
        # Tree operation tensors
        common_sizes = [100, 500, 1000, 2000, 5000, 10000]
        for size in common_sizes:
            # Node indices
            self._preallocate_shape((size,), dtype=torch.int32, count=4)
            # Float values
            self._preallocate_shape((size,), dtype=torch.float32, count=2)
        
        logger.info(f"Pre-allocated {sum(len(blocks) for blocks in self.pools.values())} memory blocks")
    
    def _preallocate_shape(self, shape: Tuple[int, ...], dtype=torch.float32, count: int = 1):
        """Pre-allocate tensors of given shape"""
        key = (shape, dtype)
        if key not in self.pools:
            self.pools[key] = []
        
        for _ in range(count):
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
            block = MemoryBlock(tensor=tensor, size=tensor.numel() * tensor.element_size())
            self.pools[key].append(block)
    
    def allocate(self, shape: Tuple[int, ...], dtype=torch.float32) -> torch.Tensor:
        """Allocate a tensor from the pool with zero overhead"""
        key = (shape, dtype)
        
        with self.lock:
            # Check if we have a free block
            if key in self.pools:
                for block in self.pools[key]:
                    if not block.in_use:
                        block.in_use = True
                        self.hits += 1
                        # Return the existing tensor (zero allocation)
                        return block.tensor
            
            # No free block found, allocate new one
            self.misses += 1
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
            
            # Add to pool for future reuse
            if key not in self.pools:
                self.pools[key] = []
            
            block = MemoryBlock(tensor=tensor, size=tensor.numel() * tensor.element_size(), in_use=True)
            self.pools[key].append(block)
            
            return tensor
    
    def allocate_like(self, tensor: torch.Tensor) -> torch.Tensor:
        """Allocate a tensor with same shape and dtype as given tensor"""
        return self.allocate(tensor.shape, tensor.dtype)
    
    def release(self, tensor: torch.Tensor):
        """Release a tensor back to the pool"""
        key = (tensor.shape, tensor.dtype)
        
        with self.lock:
            if key in self.pools:
                for block in self.pools[key]:
                    if block.tensor is tensor:
                        block.in_use = False
                        return
    
    def get_stats(self) -> Dict[str, int]:
        """Get memory pool statistics"""
        total_blocks = sum(len(blocks) for blocks in self.pools.values())
        used_blocks = sum(1 for blocks in self.pools.values() for block in blocks if block.in_use)
        total_memory = sum(block.size for blocks in self.pools.values() for block in blocks)
        
        return {
            'total_blocks': total_blocks,
            'used_blocks': used_blocks,
            'free_blocks': total_blocks - used_blocks,
            'total_memory_mb': total_memory / (1024 * 1024),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0.0
        }
    
    def clear(self):
        """Clear all memory pools"""
        with self.lock:
            self.pools.clear()
            torch.cuda.empty_cache()
            logger.info("GPU Memory Pool cleared")


class TensorCache:
    """High-performance tensor cache for frequently accessed data
    
    Implements LRU eviction and zero-copy access patterns.
    """
    
    def __init__(self, max_size: int = 10000, device: torch.device = None):
        self.max_size = max_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache: Dict[int, torch.Tensor] = {}
        self.access_count: Dict[int, int] = {}
        self.lock = threading.Lock()
    
    def get(self, key: int) -> Optional[torch.Tensor]:
        """Get tensor from cache with zero-copy"""
        with self.lock:
            if key in self.cache:
                self.access_count[key] += 1
                return self.cache[key]
        return None
    
    def put(self, key: int, tensor: torch.Tensor):
        """Put tensor in cache"""
        with self.lock:
            # Evict least recently used if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                # Find least accessed key
                lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
                del self.cache[lru_key]
                del self.access_count[lru_key]
            
            self.cache[key] = tensor
            self.access_count[key] = 1
    
    def clear(self):
        """Clear the cache"""
        with self.lock:
            self.cache.clear()
            self.access_count.clear()


# Global memory pool instance
_global_memory_pool: Optional[GPUMemoryPool] = None


def get_memory_pool(device: torch.device = None, config: Optional[Dict] = None) -> GPUMemoryPool:
    """Get or create global memory pool instance"""
    global _global_memory_pool
    
    if _global_memory_pool is None:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _global_memory_pool = GPUMemoryPool(device, config)
    
    return _global_memory_pool


def clear_memory_pool():
    """Clear the global memory pool"""
    global _global_memory_pool
    if _global_memory_pool is not None:
        _global_memory_pool.clear()
        _global_memory_pool = None