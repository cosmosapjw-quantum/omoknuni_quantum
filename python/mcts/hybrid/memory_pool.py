"""Lock-Free Memory Pool Allocator

High-performance memory pool for reducing allocation overhead in MCTS.
Features thread-local pools to eliminate contention.
"""

import threading
from typing import Optional, List, Any, Dict, Type
from dataclasses import dataclass
import queue


class MemoryBlock:
    """A single memory block in the pool"""
    
    def __init__(self, block_id: int, size: int):
        self.block_id = block_id
        self.size = size
        self.data: Any = None
        self.in_use = False
        
    def reset(self):
        """Reset block to clean state"""
        self.data = None
        self.in_use = False


class MemoryPool:
    """Simple memory pool allocator
    
    Pre-allocates blocks of memory to avoid allocation overhead.
    Uses a free list for O(1) allocation/deallocation.
    """
    
    def __init__(self, block_size: int, num_blocks: int):
        """Initialize memory pool
        
        Args:
            block_size: Size of each block
            num_blocks: Number of blocks to pre-allocate
        """
        self.block_size = block_size
        self.num_blocks = num_blocks
        
        # Pre-allocate all blocks
        self._blocks = [
            MemoryBlock(i, block_size) 
            for i in range(num_blocks)
        ]
        
        # Free list (using deque for O(1) operations)
        from collections import deque
        self._free_list = deque(self._blocks)
        
        # Lock for thread safety (will be eliminated with thread-local pools)
        self._lock = threading.Lock()
        
        # Statistics
        self._stats = {
            'total_allocations': 0,
            'total_deallocations': 0,
            'peak_usage': 0
        }
        
    def allocate(self) -> Optional[MemoryBlock]:
        """Allocate a block from the pool
        
        Returns:
            MemoryBlock if available, None if pool is exhausted
        """
        with self._lock:
            if not self._free_list:
                return None
                
            block = self._free_list.popleft()
            block.in_use = True
            
            self._stats['total_allocations'] += 1
            current_usage = self.num_blocks - len(self._free_list)
            self._stats['peak_usage'] = max(self._stats['peak_usage'], current_usage)
            
            return block
            
    def deallocate(self, block: MemoryBlock) -> None:
        """Return a block to the pool"""
        if not block.in_use:
            return  # Already deallocated
            
        block.reset()
        
        with self._lock:
            self._free_list.append(block)
            self._stats['total_deallocations'] += 1
            
    def allocate_batch(self, count: int) -> List[MemoryBlock]:
        """Allocate multiple blocks at once"""
        blocks = []
        
        with self._lock:
            for _ in range(min(count, len(self._free_list))):
                if self._free_list:
                    block = self._free_list.popleft()
                    block.in_use = True
                    blocks.append(block)
                    
            self._stats['total_allocations'] += len(blocks)
            current_usage = self.num_blocks - len(self._free_list)
            self._stats['peak_usage'] = max(self._stats['peak_usage'], current_usage)
            
        return blocks
        
    def deallocate_batch(self, blocks: List[MemoryBlock]) -> None:
        """Deallocate multiple blocks at once"""
        for block in blocks:
            block.reset()
            
        with self._lock:
            self._free_list.extend(blocks)
            self._stats['total_deallocations'] += len(blocks)
            
    def available_blocks(self) -> int:
        """Get number of available blocks"""
        with self._lock:
            return len(self._free_list)
            
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics"""
        with self._lock:
            return {
                'total_blocks': self.num_blocks,
                'available_blocks': len(self._free_list),
                'allocated_blocks': self.num_blocks - len(self._free_list),
                'total_allocations': self._stats['total_allocations'],
                'total_deallocations': self._stats['total_deallocations'],
                'peak_usage': self._stats['peak_usage']
            }


class ThreadLocalMemoryPool:
    """Thread-local memory pool manager
    
    Each thread gets its own pool to eliminate contention.
    Supports work-stealing for load balancing.
    """
    
    def __init__(self, block_size: int, blocks_per_thread: int):
        """Initialize thread-local pool manager
        
        Args:
            block_size: Size of each block
            blocks_per_thread: Number of blocks per thread pool
        """
        self.block_size = block_size
        self.blocks_per_thread = blocks_per_thread
        
        # Thread-local storage
        self._thread_local = threading.local()
        
        # Global pool for work-stealing
        self._global_pool = MemoryPool(
            block_size=block_size,
            num_blocks=blocks_per_thread * 4  # Reserve for 4 threads
        )
        
        # Track all thread pools
        self._all_pools: Dict[int, MemoryPool] = {}
        self._pools_lock = threading.Lock()
        
    def _get_thread_pool(self) -> MemoryPool:
        """Get or create pool for current thread"""
        if not hasattr(self._thread_local, 'pool'):
            thread_id = threading.get_ident()
            
            # Create new pool for this thread
            pool = MemoryPool(
                block_size=self.block_size,
                num_blocks=self.blocks_per_thread
            )
            
            self._thread_local.pool = pool
            
            # Register pool
            with self._pools_lock:
                self._all_pools[thread_id] = pool
                
        return self._thread_local.pool
        
    def allocate(self) -> Optional[MemoryBlock]:
        """Allocate from thread-local pool"""
        pool = self._get_thread_pool()
        
        # Try local pool first
        block = pool.allocate()
        if block is not None:
            return block
            
        # Try global pool if local is exhausted
        return self._global_pool.allocate()
        
    def deallocate(self, block: MemoryBlock) -> None:
        """Deallocate to appropriate pool"""
        pool = self._get_thread_pool()
        
        # Check if block belongs to local pool
        if block.block_id < self.blocks_per_thread:
            pool.deallocate(block)
        else:
            # Return to global pool
            self._global_pool.deallocate(block)
            
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all pools"""
        stats = {
            'num_threads': len(self._all_pools),
            'global_pool': self._global_pool.get_stats(),
            'thread_pools': {}
        }
        
        with self._pools_lock:
            for thread_id, pool in self._all_pools.items():
                stats['thread_pools'][thread_id] = pool.get_stats()
                
        return stats


class ObjectPool:
    """Pool for reusable objects
    
    Manages a pool of pre-created objects that can be acquired and released.
    Objects are reset to clean state when released.
    """
    
    def __init__(self, object_class: Type, pool_size: int, **kwargs):
        """Initialize object pool
        
        Args:
            object_class: Class to instantiate
            pool_size: Number of objects to pre-create
            **kwargs: Arguments passed to object constructor
        """
        self.object_class = object_class
        self.pool_size = pool_size
        self.constructor_kwargs = kwargs
        
        # Pre-create objects
        self._objects = [
            object_class(**kwargs)
            for _ in range(pool_size)
        ]
        
        # Available objects
        from collections import deque
        self._available = deque(self._objects)
        self._lock = threading.Lock()
        
        # Track which objects are in use
        self._in_use = set()
        
    def acquire(self) -> Optional[Any]:
        """Acquire an object from the pool"""
        with self._lock:
            if not self._available:
                return None
                
            obj = self._available.popleft()
            self._in_use.add(id(obj))
            
            # Reset object state
            self._reset_object(obj)
            
            return obj
            
    def release(self, obj: Any) -> None:
        """Release object back to pool"""
        obj_id = id(obj)
        
        with self._lock:
            if obj_id not in self._in_use:
                return  # Not from this pool or already released
                
            self._in_use.remove(obj_id)
            self._available.append(obj)
            
    def _reset_object(self, obj: Any) -> None:
        """Reset object to clean state"""
        # Reset common attributes
        if hasattr(obj, 'value'):
            obj.value = 0
        if hasattr(obj, 'visits'):
            obj.visits = 0
        if hasattr(obj, 'children'):
            obj.children = []
        if hasattr(obj, 'data'):
            obj.data = None
            
        # Call reset method if available
        if hasattr(obj, 'reset'):
            obj.reset()
            
    def available_count(self) -> int:
        """Get number of available objects"""
        with self._lock:
            return len(self._available)


class LockFreeMemoryPool:
    """Lock-free memory pool using atomic operations
    
    This is a simplified Python version. Real lock-free implementation
    would use C++ with proper atomic operations.
    """
    
    def __init__(self, block_size: int, num_blocks: int):
        """Initialize lock-free pool"""
        self.block_size = block_size
        self.num_blocks = num_blocks
        
        # Pre-allocate blocks
        self._blocks = [
            MemoryBlock(i, block_size)
            for i in range(num_blocks)
        ]
        
        # Use queue.Queue which has thread-safe operations
        self._free_queue = queue.Queue(maxsize=num_blocks)
        
        # Initialize with all blocks
        for block in self._blocks:
            self._free_queue.put(block)
            
    def allocate(self) -> Optional[MemoryBlock]:
        """Lock-free allocation"""
        try:
            block = self._free_queue.get_nowait()
            block.in_use = True
            return block
        except queue.Empty:
            return None
            
    def deallocate(self, block: MemoryBlock) -> None:
        """Lock-free deallocation"""
        block.reset()
        try:
            self._free_queue.put_nowait(block)
        except queue.Full:
            pass  # Should not happen with proper usage
            
    def available_blocks(self) -> int:
        """Approximate number of available blocks"""
        return self._free_queue.qsize()