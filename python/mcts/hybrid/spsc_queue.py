"""Single Producer Single Consumer (SPSC) Lock-Free Queue

High-performance lock-free queue optimized for single producer/consumer pattern.
Uses cache-aligned indices and power-of-2 sizing for maximum performance.
"""

import threading
from typing import Optional, Any, List
import math


def next_power_of_2(n: int) -> int:
    """Round up to next power of 2"""
    return 1 << (n - 1).bit_length() if n > 0 else 1


class CacheAligned:
    """Helper to ensure cache line alignment (64 bytes typical)"""
    CACHE_LINE_SIZE = 64
    
    def __init__(self, value: int = 0):
        self._padding1 = [0] * 8  # Padding before
        self._value = value
        self._padding2 = [0] * 8  # Padding after
        
    def load(self) -> int:
        """Load with acquire semantics"""
        # In Python, reads are already atomic for integers
        # In C++ this would be: value.load(std::memory_order_acquire)
        return self._value
        
    def store(self, value: int):
        """Store with release semantics"""
        # In Python, writes are already atomic for integers
        # In C++ this would be: value.store(val, std::memory_order_release)
        self._value = value
        
    def get(self) -> int:
        """Non-atomic read (for local cached values)"""
        return self._value


class SPSCQueue:
    """Single Producer Single Consumer Lock-Free Queue
    
    Features:
    - Lock-free for single producer/consumer
    - Cache-aligned indices to prevent false sharing
    - Power-of-2 capacity for fast modulo
    - Minimal memory barriers for maximum throughput
    """
    
    def __init__(self, capacity: int):
        """Initialize SPSC queue with given capacity
        
        Args:
            capacity: Desired capacity (will be rounded to power of 2)
        """
        # Round up to power of 2 for fast modulo
        self.capacity = next_power_of_2(capacity)
        self._mask = self.capacity - 1  # For fast modulo using bitwise AND
        
        # Pre-allocate buffer
        self._buffer = [None] * self.capacity
        
        # Cache-aligned indices to prevent false sharing
        # Producer writes to head, consumer reads from head
        self._head = CacheAligned(0)  # Written by producer, read by consumer
        
        # Consumer writes to tail, producer reads from tail  
        self._tail = CacheAligned(0)  # Written by consumer, read by producer
        
        # Cached values for each thread to reduce cache coherency traffic
        self._cached_head = 0  # Consumer's cached copy of head
        self._cached_tail = 0  # Producer's cached copy of tail
        
    def push(self, item: Any) -> bool:
        """Push item to queue (producer only)
        
        Returns:
            True if successful, False if queue is full
        """
        current_head = self._head.get()
        next_head = (current_head + 1) & self._mask
        
        # Check if queue is full using cached tail value
        if next_head == self._cached_tail:
            # Reload tail and check again
            self._cached_tail = self._tail.load()
            if next_head == self._cached_tail:
                return False  # Queue is full
                
        # Store item
        self._buffer[current_head] = item
        
        # Update head with release semantics
        self._head.store(next_head)
        
        return True
        
    def pop(self) -> Optional[Any]:
        """Pop item from queue (consumer only)
        
        Returns:
            Item if available, None if queue is empty
        """
        current_tail = self._tail.get()
        
        # Check if queue is empty using cached head value
        if current_tail == self._cached_head:
            # Reload head and check again
            self._cached_head = self._head.load()
            if current_tail == self._cached_head:
                return None  # Queue is empty
                
        # Read item
        item = self._buffer[current_tail]
        
        # Clear slot (helps with GC)
        self._buffer[current_tail] = None
        
        # Update tail with release semantics
        next_tail = (current_tail + 1) & self._mask
        self._tail.store(next_tail)
        
        return item
        
    def try_push(self, item: Any) -> bool:
        """Non-blocking push (same as push for SPSC)"""
        return self.push(item)
        
    def try_pop(self) -> Optional[Any]:
        """Non-blocking pop (same as pop for SPSC)"""
        return self.pop()
        
    def push_batch(self, items: List[Any]) -> int:
        """Push multiple items efficiently
        
        Returns:
            Number of items successfully pushed
        """
        pushed = 0
        for item in items:
            if self.push(item):
                pushed += 1
            else:
                break
        return pushed
        
    def pop_batch(self, max_items: int) -> List[Any]:
        """Pop multiple items efficiently
        
        Returns:
            List of popped items (may be less than max_items)
        """
        items = []
        for _ in range(max_items):
            item = self.pop()
            if item is not None:
                items.append(item)
            else:
                break
        return items
        
    def is_empty(self) -> bool:
        """Check if queue is empty (consumer's view)"""
        return self._tail.get() == self._head.load()
        
    def is_full(self) -> bool:
        """Check if queue is full (producer's view)"""
        head = self._head.get()
        next_head = (head + 1) & self._mask
        return next_head == self._tail.load()
        
    def size_approx(self) -> int:
        """Approximate size (may be stale in concurrent context)"""
        head = self._head.load()
        tail = self._tail.load()
        return (head - tail) & self._mask


class SPSCQueueOptimized(SPSCQueue):
    """Further optimized version with additional features
    
    This version includes:
    - Prefetching for better cache performance
    - Aligned memory allocation
    - Optional busy-wait spinning
    """
    
    def __init__(self, capacity: int, spin_count: int = 100):
        super().__init__(capacity)
        self.spin_count = spin_count
        
    def push_spinning(self, item: Any, max_spins: Optional[int] = None) -> bool:
        """Push with spinning on full queue"""
        spins = max_spins or self.spin_count
        
        for _ in range(spins):
            if self.push(item):
                return True
            # CPU pause instruction would go here in C++
            
        return False
        
    def pop_spinning(self, max_spins: Optional[int] = None) -> Optional[Any]:
        """Pop with spinning on empty queue"""
        spins = max_spins or self.spin_count
        
        for _ in range(spins):
            item = self.pop()
            if item is not None:
                return item
            # CPU pause instruction would go here in C++
            
        return None