"""Thread-Local Selection Buffer System

Reduces synchronization overhead by buffering selections locally
before sending through SPSC queues.
"""

import threading
from typing import List, Any, Callable, Optional, Dict
from dataclasses import dataclass


class ThreadLocalBuffer:
    """Buffer for collecting items locally before batch processing
    
    Features:
    - Zero contention (thread-local)
    - Automatic flush when full
    - Batch operations for efficiency
    """
    
    def __init__(self, capacity: int, flush_callback: Optional[Callable[[List[Any]], None]] = None):
        """Initialize thread-local buffer
        
        Args:
            capacity: Buffer capacity before auto-flush
            flush_callback: Optional callback when buffer is flushed
        """
        self.capacity = capacity
        self.flush_callback = flush_callback
        self._items: List[Any] = []
        
    def add(self, item: Any) -> None:
        """Add item to buffer, auto-flush if full"""
        self._items.append(item)
        
        if len(self._items) >= self.capacity:
            self._auto_flush()
            
    def flush(self) -> List[Any]:
        """Manually flush buffer and return items"""
        items = self._items
        self._items = []
        return items
        
    def _auto_flush(self) -> None:
        """Internal auto-flush when buffer is full"""
        if self.flush_callback and self._items:
            items = self.flush()
            self.flush_callback(items)
            
    def size(self) -> int:
        """Get current number of items in buffer"""
        return len(self._items)
        
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return len(self._items) == 0
        
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return len(self._items) >= self.capacity
        
    def clear(self) -> None:
        """Clear buffer without flushing"""
        self._items = []


class ThreadLocalBufferManager:
    """Manages thread-local buffers for multiple threads
    
    Each thread gets its own buffer instance to avoid contention.
    Provides global operations like flush_all.
    """
    
    def __init__(
        self, 
        buffer_capacity: int = 64,
        global_flush_callback: Optional[Callable[[List[Any]], None]] = None
    ):
        """Initialize buffer manager
        
        Args:
            buffer_capacity: Capacity for each thread's buffer
            global_flush_callback: Callback for all flush operations
        """
        self.buffer_capacity = buffer_capacity
        self.global_flush_callback = global_flush_callback
        
        # Thread-local storage for buffers
        self._thread_local = threading.local()
        
        # Track all buffers for global operations
        self._all_buffers: Dict[int, ThreadLocalBuffer] = {}
        self._buffers_lock = threading.Lock()
        
    def get_buffer(self) -> ThreadLocalBuffer:
        """Get thread-local buffer for current thread"""
        # Check if buffer exists for this thread
        if not hasattr(self._thread_local, 'buffer'):
            # Create new buffer for this thread
            thread_id = threading.get_ident()
            
            # Create buffer with flush callback
            def thread_flush_callback(items):
                if self.global_flush_callback:
                    self.global_flush_callback(items)
                    
            buffer = ThreadLocalBuffer(
                capacity=self.buffer_capacity,
                flush_callback=thread_flush_callback
            )
            
            # Store in thread-local storage
            self._thread_local.buffer = buffer
            
            # Register buffer for global operations
            with self._buffers_lock:
                self._all_buffers[thread_id] = buffer
                
        return self._thread_local.buffer
        
    def flush_all(self) -> None:
        """Flush all thread buffers"""
        with self._buffers_lock:
            for thread_id, buffer in list(self._all_buffers.items()):
                items = buffer.flush()
                if items and self.global_flush_callback:
                    self.global_flush_callback(items)
                    
    def clear_all(self) -> None:
        """Clear all buffers without flushing"""
        with self._buffers_lock:
            for buffer in self._all_buffers.values():
                buffer.clear()
                
    def remove_thread_buffer(self, thread_id: Optional[int] = None) -> None:
        """Remove buffer for thread (cleanup)"""
        if thread_id is None:
            thread_id = threading.get_ident()
            
        with self._buffers_lock:
            if thread_id in self._all_buffers:
                # Flush before removing
                buffer = self._all_buffers[thread_id]
                items = buffer.flush()
                if items and self.global_flush_callback:
                    self.global_flush_callback(items)
                    
                del self._all_buffers[thread_id]
                
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about buffer usage"""
        with self._buffers_lock:
            total_items = sum(b.size() for b in self._all_buffers.values())
            return {
                'num_threads': len(self._all_buffers),
                'total_items_buffered': total_items,
                'buffer_capacity': self.buffer_capacity,
                'thread_ids': list(self._all_buffers.keys())
            }


class BufferedSPSCPipeline:
    """Complete pipeline with thread-local buffers feeding SPSC queues
    
    Architecture:
    Thread 1 → Buffer 1 → SPSC Queue 1 → Aggregator
    Thread 2 → Buffer 2 → SPSC Queue 2 → Aggregator
    ...
    Thread N → Buffer N → SPSC Queue N → Aggregator
    """
    
    def __init__(
        self,
        num_threads: int,
        buffer_capacity: int = 64,
        queue_capacity: int = 1024
    ):
        """Initialize buffered pipeline
        
        Args:
            num_threads: Number of producer threads
            buffer_capacity: Capacity for each thread buffer
            queue_capacity: Capacity for each SPSC queue
        """
        from .spsc_queue import SPSCQueue
        
        self.num_threads = num_threads
        self.buffer_capacity = buffer_capacity
        
        # Create SPSC queue for each thread
        self.queues = [SPSCQueue(capacity=queue_capacity) for _ in range(num_threads)]
        
        # Thread ID to queue index mapping
        self.thread_to_queue: Dict[int, int] = {}
        self.next_queue_idx = 0
        self.queue_assignment_lock = threading.Lock()
        
        # Create buffer manager with custom flush
        def flush_to_queue(items):
            thread_id = threading.get_ident()
            queue_idx = self._get_queue_for_thread(thread_id)
            queue = self.queues[queue_idx]
            
            # Push items to queue
            for item in items:
                while not queue.push(item):
                    # In real implementation, might use backpressure
                    pass
                    
        self.buffer_manager = ThreadLocalBufferManager(
            buffer_capacity=buffer_capacity,
            global_flush_callback=flush_to_queue
        )
        
    def _get_queue_for_thread(self, thread_id: int) -> int:
        """Get queue index for thread"""
        if thread_id not in self.thread_to_queue:
            with self.queue_assignment_lock:
                if thread_id not in self.thread_to_queue:
                    self.thread_to_queue[thread_id] = self.next_queue_idx
                    self.next_queue_idx = (self.next_queue_idx + 1) % self.num_threads
                    
        return self.thread_to_queue[thread_id]
        
    def submit(self, item: Any) -> None:
        """Submit item through buffered pipeline"""
        buffer = self.buffer_manager.get_buffer()
        buffer.add(item)
        
    def collect_all(self) -> List[Any]:
        """Collect all items from all queues"""
        # First flush all buffers
        self.buffer_manager.flush_all()
        
        # Then collect from all queues
        all_items = []
        for queue in self.queues:
            while not queue.is_empty():
                item = queue.pop()
                if item is not None:
                    all_items.append(item)
                    
        return all_items
        
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        buffer_stats = self.buffer_manager.get_stats()
        
        queue_sizes = [q.size_approx() for q in self.queues]
        
        return {
            'buffer_stats': buffer_stats,
            'queue_sizes': queue_sizes,
            'total_in_queues': sum(queue_sizes),
            'num_queues': len(self.queues)
        }