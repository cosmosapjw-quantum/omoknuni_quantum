"""Batch Evaluation Coordinator for Phase 1 Optimization

This module implements request batching coordination to eliminate the 14.7x slowdown
from individual request overhead in the RemoteEvaluator architecture.

Key features:
1. Proper condition variable synchronization instead of polling
2. Adaptive timeout handling for different load conditions
3. Cross-worker batch coordination for maximum efficiency
4. Graceful fallback under heavy load
"""

import torch
import numpy as np
import time
import threading
import queue
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class BatchEvaluationRequest:
    """Optimized batch request that bundles multiple evaluations"""
    request_id: int
    states: np.ndarray  # Shape: (batch_size, channels, height, width)
    legal_masks: Optional[np.ndarray]  # Shape: (batch_size, action_space)
    temperatures: np.ndarray  # Shape: (batch_size,)
    worker_id: int
    timestamp: float
    individual_request_ids: List[int]  # Track original request IDs for response routing


@dataclass 
class BatchEvaluationResponse:
    """Optimized batch response"""
    request_id: int
    policies: np.ndarray  # Shape: (batch_size, action_space)
    values: np.ndarray    # Shape: (batch_size,)
    worker_id: int
    individual_request_ids: List[int]


class RequestBatchingCoordinator:
    """Coordinates request batching across workers to reduce communication overhead
    
    This coordinator sits between workers and the GPU service, collecting individual
    evaluation requests and bundling them into efficient batches before sending
    to the GPU service.
    
    Automatically detects threading vs multiprocessing context and adapts accordingly.
    """
    
    def __init__(self, 
                 max_batch_size: int = 64,
                 batch_timeout_ms: float = 100.0,
                 enable_cross_worker_batching: bool = True):
        """Initialize the batching coordinator
        
        Args:
            max_batch_size: Maximum number of evaluations to batch together
            batch_timeout_ms: Maximum time to wait for batch fill (milliseconds)
            enable_cross_worker_batching: Allow batching requests from different workers
        """
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout_ms / 1000.0  # Convert to seconds
        
        # CRITICAL: Detect multiprocessing context and disable coordination if needed
        self._is_multiprocessing_context = self._detect_multiprocessing_context()
        
        if self._is_multiprocessing_context and enable_cross_worker_batching:
            logger.debug("Multiprocessing context detected - disabling cross-worker batching for stability")
            self.enable_cross_worker_batching = False
        else:
            self.enable_cross_worker_batching = enable_cross_worker_batching
        
        # Request coordination with proper synchronization
        self.pending_requests = []  # List of (request, response_event, response_data, queues)
        self.pending_lock = threading.Lock()
        self.pending_condition = threading.Condition(self.pending_lock)  # FIX: Use condition variable
        
        # Batch processing thread
        self.batch_thread = None
        self.stop_event = threading.Event()
        self.request_counter = 0
        self.request_counter_lock = threading.Lock()
        
        # Performance tracking
        self.stats = {
            'requests_processed': 0,
            'batches_created': 0,
            'avg_batch_size': 0.0,
            'total_latency_saved': 0.0,
            'cross_worker_batches': 0,
            'coordination_timeouts': 0,
            'coordination_successes': 0,
            'fallback_to_direct': 0
        }
        self.stats_lock = threading.Lock()
        
    
    def _detect_multiprocessing_context(self) -> bool:
        """Detect if we're running in a multiprocessing context
        
        Returns True if this appears to be a child process created by multiprocessing.
        This is a heuristic check that looks for common indicators.
        """
        import os
        import sys
        
        # Check for multiprocessing-specific environment variables
        if any(var in os.environ for var in ['MULTIPROCESSING_FORKED', 'MP_MAIN_FILE']):
            return True
        
        # Check if we're in a spawned process (common indicator)
        if hasattr(sys, '_getframe'):
            try:
                frame = sys._getframe()
                while frame:
                    # Look for multiprocessing in the call stack
                    if 'multiprocessing' in frame.f_code.co_filename:
                        return True
                    frame = frame.f_back
            except:
                pass
        
        # Check for process name patterns (heuristic)
        try:
            import multiprocessing as mp
            current_process = mp.current_process()
            # Main process is usually named 'MainProcess'
            if current_process.name != 'MainProcess' and 'Process-' in current_process.name:
                return True
        except:
            pass
        
        return False
    
    def start(self):
        """Start the batch coordination thread"""
        if self.batch_thread is not None:
            return
        
        self.stop_event.clear()
        self.batch_thread = threading.Thread(target=self._batch_coordination_loop, daemon=True)
        self.batch_thread.start()
    
    def stop(self):
        """Stop the batch coordination thread"""
        if self.batch_thread is None:
            return
        
        logger.info("Stopping RequestBatchingCoordinator...")
        self.stop_event.set()
        
        # Wake up the coordination thread
        with self.pending_condition:
            self.pending_condition.notify_all()
        
        self.batch_thread.join(timeout=2.0)
        if self.batch_thread.is_alive():
            logger.warning("RequestBatchingCoordinator thread did not stop cleanly")
        
        self.batch_thread = None
        logger.info("RequestBatchingCoordinator stopped")
    
    def coordinate_evaluation_batch(self, 
                                   states: np.ndarray,
                                   legal_masks: Optional[np.ndarray] = None,
                                   temperatures: Optional[np.ndarray] = None,
                                   worker_id: int = 0,
                                   gpu_service_request_queue = None,
                                   response_queue = None) -> Tuple[np.ndarray, np.ndarray]:
        """Coordinate a batch evaluation with optimized batching"""
        
        batch_size = states.shape[0]
        
        if temperatures is None:
            temperatures = np.ones(batch_size, dtype=np.float32)
        elif isinstance(temperatures, (int, float)):
            temperatures = np.full(batch_size, float(temperatures), dtype=np.float32)
        
        # Create coordination event for response synchronization
        response_event = threading.Event()
        response_data = {'policies': None, 'values': None, 'error': None}
        
        # Create batch request with unique ID
        with self.request_counter_lock:
            self.request_counter += 1
            request_id = self.request_counter
            
        batch_request = BatchEvaluationRequest(
            request_id=request_id,
            states=states,
            legal_masks=legal_masks,
            temperatures=temperatures,
            worker_id=worker_id,
            timestamp=time.time(),
            individual_request_ids=list(range(batch_size))
        )
        
        # Decision: immediate processing or coordination
        if batch_size >= self.max_batch_size or not self.enable_cross_worker_batching:
            # Large batch or coordination disabled - process immediately
            with self.stats_lock:
                self.stats['fallback_to_direct'] += 1
            return self._process_immediate_batch(batch_request, gpu_service_request_queue, response_queue)
        
        # Add to pending requests for coordination
        request_entry = (batch_request, response_event, response_data, gpu_service_request_queue, response_queue)
        
        with self.pending_condition:
            self.pending_requests.append(request_entry)
            self.pending_condition.notify()  # Wake up coordination thread
        
        # Wait for response with adaptive timeout based on load
        # Under heavy load, increase timeout to reduce contention
        pending_count = len(self.pending_requests)
        if pending_count > 10:
            # Heavy load - increase timeout
            wait_timeout = self.batch_timeout * 2 + 0.1
        elif pending_count > 5:
            # Medium load - moderate increase
            wait_timeout = self.batch_timeout * 1.5 + 0.05
        else:
            # Light load - normal timeout
            wait_timeout = self.batch_timeout + 0.05
        
        if not response_event.wait(timeout=wait_timeout):
            # Timeout - try to remove from pending
            with self.pending_condition:
                try:
                    self.pending_requests.remove(request_entry)
                    removed = True
                except ValueError:
                    removed = False  # Already being processed
            
            with self.stats_lock:
                self.stats['coordination_timeouts'] += 1
            
            if removed:
                # Successfully removed - process directly
                logger.debug(f"Coordination timeout for worker {worker_id} (load={pending_count}), falling back to direct")
                return self._process_immediate_batch(batch_request, gpu_service_request_queue, response_queue)
            else:
                # Already being processed - wait reasonable time for completion
                # FIX: Reduce critical timeout to avoid 1+ second delays
                if not response_event.wait(timeout=0.5):
                    logger.warning(f"Extended timeout for worker {worker_id}, falling back to direct processing")
                    # Don't raise error - just fallback to direct processing
                    return self._process_immediate_batch(batch_request, gpu_service_request_queue, response_queue)
        
        # Check for errors
        if response_data.get('error'):
            logger.warning(f"Coordination error for worker {worker_id}: {response_data['error']}")
            # Don't raise error - fallback to direct processing
            return self._process_immediate_batch(batch_request, gpu_service_request_queue, response_queue)
        
        with self.stats_lock:
            self.stats['coordination_successes'] += 1
        
        return response_data['policies'], response_data['values']
    
    def _process_immediate_batch(self, 
                                batch_request: BatchEvaluationRequest,
                                gpu_service_request_queue,
                                response_queue) -> Tuple[np.ndarray, np.ndarray]:
        """Process a batch immediately without coordination"""
        
        if gpu_service_request_queue is None or response_queue is None:
            raise ValueError("GPU service queues not provided")
        
        # Send directly to GPU service
        gpu_service_request_queue.put(batch_request)
        
        # Wait for response
        start_time = time.time()
        try:
            # Use reasonable timeout
            response = response_queue.get(timeout=5.0)
            
            if isinstance(response, BatchEvaluationResponse):
                processing_time = time.time() - start_time
                logger.debug(f"Immediate batch processed: size={batch_request.states.shape[0]}, "
                           f"time={processing_time*1000:.1f}ms")
                return response.policies, response.values
            else:
                logger.error(f"Unexpected response type: {type(response)}")
                raise RuntimeError("Invalid response type")
                
        except queue.Empty:
            logger.error(f"Timeout waiting for immediate batch response")
            # Return fallback
            batch_size = batch_request.states.shape[0]
            action_size = 225  # Gomoku action space
            return (np.random.rand(batch_size, action_size).astype(np.float32),
                   np.zeros(batch_size, dtype=np.float32))
    
    def _batch_coordination_loop(self):
        """Main coordination loop with proper synchronization"""
        
        while not self.stop_event.is_set():
            try:
                requests_to_process = []
                
                with self.pending_condition:
                    # Wait for requests or timeout
                    if not self.pending_requests:
                        # FIX: Use condition variable wait instead of polling
                        self.pending_condition.wait(timeout=self.batch_timeout)
                    
                    if not self.pending_requests:
                        continue
                    
                    # Check oldest request age and adapt processing strategy
                    current_time = time.time()
                    oldest_timestamp = self.pending_requests[0][0].timestamp
                    age = current_time - oldest_timestamp
                    pending_count = len(self.pending_requests)
                    
                    # Adaptive batching: under heavy load, be more aggressive
                    should_process = False
                    if pending_count >= self.max_batch_size:
                        # Full batch - process immediately
                        should_process = True
                    elif pending_count >= self.max_batch_size // 2 and age >= self.batch_timeout * 0.5:
                        # Half-full batch and moderate age - process to reduce latency
                        should_process = True
                    elif age >= self.batch_timeout:
                        # Timeout reached - process whatever we have
                        should_process = True
                    
                    if should_process:
                        # Take all pending requests up to max_batch_size
                        batch_count = min(len(self.pending_requests), self.max_batch_size)
                        requests_to_process = self.pending_requests[:batch_count]
                        self.pending_requests = self.pending_requests[batch_count:]
                
                # Process outside the lock
                if requests_to_process:
                    self._process_request_batch(requests_to_process)
                    
            except Exception as e:
                logger.error(f"Error in batch coordination loop: {e}", exc_info=True)
                # Don't let thread die on error
                time.sleep(0.01)
        
        logger.info("Batch coordination loop stopped")
    
    def _process_request_batch(self, requests: List[Tuple]):
        """Process a batch of requests together"""
        
        if not requests:
            return
        
        batch_start_time = time.time()
        
        # Unpack request components
        batch_requests = []
        response_events = []
        response_datas = []
        gpu_queues = []
        response_queues = []
        
        for req, event, data, gpu_q, resp_q in requests:
            batch_requests.append(req)
            response_events.append(event)
            response_datas.append(data)
            gpu_queues.append(gpu_q)
            response_queues.append(resp_q)
        
        # Use first worker's queues (they should all be the same in practice)
        gpu_queue = gpu_queues[0]
        response_queue = response_queues[0]
        
        if gpu_queue is None or response_queue is None:
            # Handle error case
            error_msg = "GPU service queues not available"
            for i in range(len(requests)):
                response_datas[i]['error'] = error_msg
                response_events[i].set()
            return
        
        # Combine all states into mega-batch
        try:
            all_states = []
            all_legal_masks = []
            all_temperatures = []
            request_boundaries = []
            
            current_idx = 0
            for req in batch_requests:
                batch_size = req.states.shape[0]
                all_states.append(req.states)
                all_temperatures.append(req.temperatures)
                
                if req.legal_masks is not None:
                    all_legal_masks.append(req.legal_masks)
                else:
                    # Create dummy mask
                    all_legal_masks.append(np.ones((batch_size, 225), dtype=bool))
                
                request_boundaries.append((current_idx, current_idx + batch_size))
                current_idx += batch_size
            
            # Convert to numpy arrays (handling both tensor and numpy inputs)
            if isinstance(all_states[0], torch.Tensor):
                mega_states = torch.cat([s if isinstance(s, torch.Tensor) else torch.from_numpy(s) 
                                        for s in all_states], dim=0).cpu().numpy()
            else:
                mega_states = np.concatenate(all_states, axis=0)
            
            if isinstance(all_temperatures[0], torch.Tensor):
                mega_temperatures = torch.cat([t if isinstance(t, torch.Tensor) else torch.from_numpy(t)
                                             for t in all_temperatures], dim=0).cpu().numpy()
            else:
                mega_temperatures = np.concatenate(all_temperatures, axis=0)
            
            if isinstance(all_legal_masks[0], torch.Tensor):
                mega_legal_masks = torch.cat([m if isinstance(m, torch.Tensor) else torch.from_numpy(m)
                                            for m in all_legal_masks], dim=0).cpu().numpy()
            else:
                mega_legal_masks = np.concatenate(all_legal_masks, axis=0)
            
            # Create mega-batch request
            with self.request_counter_lock:
                self.request_counter += 1
                mega_request_id = self.request_counter
                
            mega_request = BatchEvaluationRequest(
                request_id=mega_request_id,
                states=mega_states,
                legal_masks=mega_legal_masks if np.any(mega_legal_masks) else None,
                temperatures=mega_temperatures,
                worker_id=-1,  # Special ID for coordinated batch
                timestamp=time.time(),
                individual_request_ids=list(range(mega_states.shape[0]))
            )
            
            # Send to GPU service
            gpu_queue.put(mega_request)
            
            # Wait for response with timeout
            response = response_queue.get(timeout=5.0)
            
            if not isinstance(response, BatchEvaluationResponse):
                raise RuntimeError(f"Invalid response type: {type(response)}")
            
            # Distribute results back
            for i, (start_idx, end_idx) in enumerate(request_boundaries):
                response_datas[i]['policies'] = response.policies[start_idx:end_idx]
                response_datas[i]['values'] = response.values[start_idx:end_idx]
                response_events[i].set()
            
            # Update statistics
            batch_time = time.time() - batch_start_time
            with self.stats_lock:
                self.stats['batches_created'] += 1
                self.stats['requests_processed'] += len(batch_requests)
                if len(set(req.worker_id for req in batch_requests)) > 1:
                    self.stats['cross_worker_batches'] += 1
                
                total_size = sum(req.states.shape[0] for req in batch_requests)
                old_avg = self.stats['avg_batch_size']
                old_count = self.stats['batches_created'] - 1
                self.stats['avg_batch_size'] = (old_avg * old_count + total_size) / self.stats['batches_created']
            
            
        except Exception as e:
            logger.error(f"Error processing coordinated batch: {e}", exc_info=True)
            # Set error for all requests
            for i in range(len(requests)):
                response_datas[i]['error'] = str(e)
                response_events[i].set()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self.stats_lock:
            stats = self.stats.copy()
        
        total = stats['coordination_successes'] + stats['coordination_timeouts'] + stats['fallback_to_direct']
        if total > 0:
            stats['coordination_success_rate'] = stats['coordination_successes'] / total
            stats['timeout_rate'] = stats['coordination_timeouts'] / total
        else:
            stats['coordination_success_rate'] = 0.0
            stats['timeout_rate'] = 0.0
        
        return stats


# Global coordinator instance
_global_coordinator: Optional[RequestBatchingCoordinator] = None
_coordinator_lock = threading.Lock()
_coordinator_process_id: Optional[int] = None


def get_global_batching_coordinator(max_batch_size: int = 64,
                                   batch_timeout_ms: float = 100.0,
                                   enable_cross_worker_batching: bool = True) -> RequestBatchingCoordinator:
    """Get or create the global batching coordinator"""
    global _global_coordinator, _coordinator_process_id
    import os
    
    current_pid = os.getpid()
    
    with _coordinator_lock:
        if _global_coordinator is None or _coordinator_process_id != current_pid:
            if _global_coordinator is not None:
                try:
                    _global_coordinator.stop()
                except:
                    pass
            
            _global_coordinator = RequestBatchingCoordinator(
                max_batch_size=max_batch_size,
                batch_timeout_ms=batch_timeout_ms,
                enable_cross_worker_batching=enable_cross_worker_batching
            )
            _global_coordinator.start()
            _coordinator_process_id = current_pid
            
        
        return _global_coordinator


def cleanup_global_coordinator():
    """Clean up the global coordinator"""
    global _global_coordinator, _coordinator_process_id
    
    with _coordinator_lock:
        if _global_coordinator is not None:
            _global_coordinator.stop()
            _global_coordinator = None
            _coordinator_process_id = None