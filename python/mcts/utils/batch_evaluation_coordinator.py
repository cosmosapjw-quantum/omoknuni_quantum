"""Batch Evaluation Coordinator for Phase 1 Optimization

This module implements request batching coordination to eliminate the 14.7x slowdown
from individual request overhead in the RemoteEvaluator architecture.
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

# Note: Adaptive parameter tuning was removed from production code

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
    """
    
    def __init__(self, 
                 max_batch_size: int = 64,
                 batch_timeout_ms: float = 100.0,  # 100ms timeout to prevent warnings
                 enable_cross_worker_batching: bool = True):
        """Initialize the batching coordinator
        
        Args:
            max_batch_size: Maximum number of evaluations to batch together
            batch_timeout_ms: Maximum time to wait for batch fill (milliseconds)
            enable_cross_worker_batching: Allow batching requests from different workers
        """
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout_ms / 1000.0  # Convert to seconds
        self.enable_cross_worker_batching = enable_cross_worker_batching
        
        # Request coordination
        self.pending_requests: List[Tuple[Any, threading.Event, int]] = []  # (request, response_event, timestamp)
        self.pending_lock = threading.Lock()
        
        # Batch processing
        self.batch_thread = None
        self.stop_event = threading.Event()
        self.request_counter = 0
        
        # Performance tracking
        self.stats = {
            'requests_processed': 0,
            'batches_created': 0,
            'avg_batch_size': 0.0,
            'total_latency_saved': 0.0,
            'cross_worker_batches': 0,
            'coordination_timeouts': 0,
            'avg_processing_time': 0.0
        }
        
        # Phase 2.2: Adaptive timeout management
        self.adaptive_timeout = True
        self.base_timeout = self.batch_timeout
        self.timeout_history = []
        self.last_timeout_adjustment = time.time()
        
        # Adaptive parameter tuning disabled in production build
        self.adaptive_tuning_enabled = False
        self.last_metrics_report = time.time()
        self.simulation_count = 100  # Default simulation count
        
        if self.adaptive_tuning_enabled:
            try:
                self.parameter_tuner = get_global_parameter_tuner()
                # Register callback to receive parameter updates
                self.parameter_tuner.register_parameter_callback(
                    'batch_coordinator', 
                    self._update_adaptive_parameters
                )
                logger.debug("Adaptive parameter tuning enabled for batch coordinator")
            except Exception as e:
                logger.warning(f"Failed to enable adaptive tuning: {e}")
                self.adaptive_tuning_enabled = False
        
        logger.debug(f"Initialized BatchingCoordinator: max_batch={max_batch_size}, "
                    f"timeout={batch_timeout_ms}ms, cross_worker={enable_cross_worker_batching}, "
                    f"adaptive_tuning={self.adaptive_tuning_enabled}")
    
    def start(self):
        """Start the batch coordination thread"""
        if self.batch_thread is not None:
            return
        
        self.stop_event.clear()
        self.batch_thread = threading.Thread(target=self._batch_coordination_loop, daemon=True)
        self.batch_thread.start()
        logger.debug("BatchingCoordinator started")
    
    def stop(self):
        """Stop the batch coordination thread"""
        if self.batch_thread is None:
            return
        
        self.stop_event.set()
        self.batch_thread.join(timeout=2.0)
        self.batch_thread = None
        logger.debug("BatchingCoordinator stopped")
    
    def coordinate_evaluation_batch(self, 
                                   states: np.ndarray,
                                   legal_masks: Optional[np.ndarray] = None,
                                   temperatures: Optional[np.ndarray] = None,
                                   worker_id: int = 0,
                                   gpu_service_request_queue = None,
                                   response_queue = None) -> Tuple[np.ndarray, np.ndarray]:
        """Coordinate a batch evaluation with optimized batching
        
        This method takes a batch from a worker and potentially combines it with
        other pending requests to create larger, more efficient batches.
        
        Args:
            states: Batch of game states (batch_size, channels, height, width)
            legal_masks: Legal action masks (batch_size, action_space)
            temperatures: Temperature values (batch_size,)
            worker_id: ID of requesting worker
            gpu_service_request_queue: Queue to send requests to GPU service
            response_queue: Queue to receive responses from GPU service
        
        Returns:
            Tuple of (policies, values)
        """
        batch_size = states.shape[0]
        
        if temperatures is None:
            temperatures = np.ones(batch_size, dtype=np.float32)
        
        # Create coordination event for response synchronization
        response_event = threading.Event()
        response_data = {'policies': None, 'values': None}
        
        # Create batch request
        self.request_counter += 1
        batch_request = BatchEvaluationRequest(
            request_id=self.request_counter,
            states=states,
            legal_masks=legal_masks,
            temperatures=temperatures,
            worker_id=worker_id,
            timestamp=time.time(),
            individual_request_ids=list(range(batch_size))
        )
        
        # Check if we can immediately process or need to batch with others
        if batch_size >= self.max_batch_size or not self.enable_cross_worker_batching:
            # Large enough batch or cross-worker batching disabled - process immediately
            return self._process_immediate_batch(batch_request, gpu_service_request_queue, response_queue)
        else:
            # Add to pending requests for potential cross-worker batching
            return self._process_coordinated_batch(batch_request, response_event, response_data,
                                                  gpu_service_request_queue, response_queue)
    
    def _process_immediate_batch(self, 
                                batch_request: BatchEvaluationRequest,
                                gpu_service_request_queue,
                                response_queue) -> Tuple[np.ndarray, np.ndarray]:
        """Process a batch immediately without additional coordination"""
        
        # Send directly to GPU service
        gpu_service_request_queue.put(batch_request)
        
        # Wait for response
        start_time = time.time()
        try:
            response = response_queue.get(timeout=30.0)  # 30s timeout
            
            if isinstance(response, BatchEvaluationResponse):
                processing_time = time.time() - start_time
                logger.debug(f"Immediate batch processed: size={batch_request.states.shape[0]}, "
                           f"time={processing_time*1000:.1f}ms")
                return response.policies, response.values
            else:
                logger.error(f"Unexpected response type: {type(response)}")
                raise RuntimeError("Invalid response type")
                
        except queue.Empty:
            # Count timeouts and only log periodically to avoid spam
            if not hasattr(self, '_timeout_count'):
                self._timeout_count = 0
            self._timeout_count += 1
            if self._timeout_count <= 1 or self._timeout_count % 1000 == 0:
                logger.debug(f"Batch evaluation timeout (count: {self._timeout_count})")
            # Return fallback responses
            batch_size = batch_request.states.shape[0]
            action_size = 225  # Gomoku action space
            return (np.random.rand(batch_size, action_size).astype(np.float32),
                   np.zeros(batch_size, dtype=np.float32))
    
    def _process_coordinated_batch(self,
                                  batch_request: BatchEvaluationRequest,
                                  response_event: threading.Event,
                                  response_data: Dict,
                                  gpu_service_request_queue,
                                  response_queue) -> Tuple[np.ndarray, np.ndarray]:
        """Process a batch with cross-worker coordination for optimal batching"""
        
        # Add to pending requests
        request_entry = (batch_request, response_event, response_data, gpu_service_request_queue, response_queue)
        
        with self.pending_lock:
            self.pending_requests.append(request_entry)
        
        # Wait for coordination thread to process the batch
        timeout_exceeded = not response_event.wait(timeout=self.batch_timeout * 2)
        
        if timeout_exceeded:
            # Phase 2.2: Adaptive timeout handling
            self.stats['coordination_timeouts'] += 1
            self._adjust_adaptive_timeout()
            
            logger.debug(f"Batch coordination timeout for worker {batch_request.worker_id} "
                        f"(timeout={self.batch_timeout*1000:.1f}ms)")
            
            # Remove from pending and process immediately as fallback
            with self.pending_lock:
                try:
                    self.pending_requests.remove(request_entry)
                except ValueError:
                    pass  # Already processed
            
            return self._process_immediate_batch(batch_request, gpu_service_request_queue, response_queue)
        
        # Return coordinated results
        return response_data['policies'], response_data['values']
    
    def _batch_coordination_loop(self):
        """Main coordination loop that creates optimized batches from pending requests"""
        logger.debug("Batch coordination loop started")
        
        while not self.stop_event.is_set():
            try:
                # Check for pending requests
                current_time = time.time()
                
                with self.pending_lock:
                    if not self.pending_requests:
                        # No pending requests - sleep briefly
                        time.sleep(0.001)  # 1ms sleep
                        continue
                    
                    # Check if oldest request has exceeded timeout
                    oldest_request = self.pending_requests[0]
                    oldest_timestamp = oldest_request[0].timestamp
                    
                    if current_time - oldest_timestamp >= self.batch_timeout:
                        # Timeout reached - process current batch
                        batch_requests = []
                        response_events = []
                        response_datas = []
                        gpu_queues = []
                        response_queues = []
                        
                        # Collect all pending requests (or up to max_batch_size)
                        batch_count = min(len(self.pending_requests), self.max_batch_size)
                        
                        for _ in range(batch_count):
                            req, event, data, gpu_q, resp_q = self.pending_requests.pop(0)
                            batch_requests.append(req)
                            response_events.append(event)
                            response_datas.append(data)
                            gpu_queues.append(gpu_q)
                            response_queues.append(resp_q)
                    else:
                        # Not ready for timeout processing
                        batch_requests = []
                
                if not batch_requests:
                    continue
                
                # Create mega-batch from collected requests
                self._process_mega_batch(batch_requests, response_events, response_datas, 
                                       gpu_queues, response_queues)
                
                # Report metrics to adaptive tuner
                self._report_metrics_to_tuner()
                
            except Exception as e:
                logger.error(f"Error in batch coordination loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.01)  # Prevent tight error loop
        
        logger.debug("Batch coordination loop stopped")
    
    def _process_mega_batch(self,
                           batch_requests: List[BatchEvaluationRequest],
                           response_events: List[threading.Event],
                           response_datas: List[Dict],
                           gpu_queues: List,
                           response_queues: List):
        """Process a mega-batch created from multiple worker requests"""
        
        if not batch_requests:
            return
        
        # Combine all requests into a single mega-batch
        all_states = []
        all_legal_masks = []
        all_temperatures = []
        request_boundaries = []  # Track where each original request starts/ends
        
        current_idx = 0
        for req in batch_requests:
            batch_size = req.states.shape[0]
            all_states.append(req.states)
            all_temperatures.append(req.temperatures)
            
            if req.legal_masks is not None:
                all_legal_masks.append(req.legal_masks)
            
            request_boundaries.append((current_idx, current_idx + batch_size, req.worker_id))
            current_idx += batch_size
        
        # Create mega-batch arrays
        mega_states = np.concatenate(all_states, axis=0)
        mega_temperatures = np.concatenate(all_temperatures, axis=0)
        mega_legal_masks = np.concatenate(all_legal_masks, axis=0) if all_legal_masks else None
        
        # Create mega-batch request
        self.request_counter += 1
        mega_request = BatchEvaluationRequest(
            request_id=self.request_counter,
            states=mega_states,
            legal_masks=mega_legal_masks,
            temperatures=mega_temperatures,
            worker_id=-1,  # Special ID for mega-batch
            timestamp=time.time(),
            individual_request_ids=list(range(mega_states.shape[0]))
        )
        
        # Process mega-batch (use first worker's queues)
        start_time = time.time()
        gpu_queues[0].put(mega_request)
        
        try:
            mega_response = response_queues[0].get(timeout=30.0)
            processing_time = time.time() - start_time
            
            if isinstance(mega_response, BatchEvaluationResponse):
                # Distribute results back to original requests
                mega_policies = mega_response.policies
                mega_values = mega_response.values
                
                for i, (start_idx, end_idx, worker_id) in enumerate(request_boundaries):
                    # Extract results for this original request
                    policies_slice = mega_policies[start_idx:end_idx]
                    values_slice = mega_values[start_idx:end_idx]
                    
                    # Store results and signal completion
                    response_datas[i]['policies'] = policies_slice
                    response_datas[i]['values'] = values_slice
                    response_events[i].set()
                
                # Update statistics
                self.stats['batches_created'] += 1
                self.stats['requests_processed'] += len(batch_requests)
                self.stats['cross_worker_batches'] += 1
                total_original_size = sum(req.states.shape[0] for req in batch_requests)
                self.stats['avg_batch_size'] = (
                    self.stats['avg_batch_size'] * (self.stats['batches_created'] - 1) + total_original_size
                ) / self.stats['batches_created']
                
                logger.debug(f"Mega-batch processed: {len(batch_requests)} requests, "
                           f"{total_original_size} evaluations, {processing_time*1000:.1f}ms")
                
            else:
                logger.error(f"Unexpected mega-batch response type: {type(mega_response)}")
                self._handle_mega_batch_error(batch_requests, response_events, response_datas)
        
        except queue.Empty:
            logger.error("Timeout waiting for mega-batch response")
            self._handle_mega_batch_error(batch_requests, response_events, response_datas)
    
    def _handle_mega_batch_error(self,
                                batch_requests: List[BatchEvaluationRequest],
                                response_events: List[threading.Event],
                                response_datas: List[Dict]):
        """Handle errors in mega-batch processing by providing fallback responses"""
        
        for i, req in enumerate(batch_requests):
            batch_size = req.states.shape[0]
            action_size = 225  # Gomoku action space
            
            # Provide fallback responses
            response_datas[i]['policies'] = np.random.rand(batch_size, action_size).astype(np.float32)
            response_datas[i]['values'] = np.zeros(batch_size, dtype=np.float32)
            response_events[i].set()
    
    def _adjust_adaptive_timeout(self):
        """Phase 2.2: Adjust timeout based on system performance"""
        if not self.adaptive_timeout:
            return
        
        current_time = time.time()
        if current_time - self.last_timeout_adjustment < 5.0:  # Only adjust every 5 seconds
            return
        
        # Calculate timeout rate over recent period
        total_requests = self.stats['requests_processed']
        timeout_rate = self.stats['coordination_timeouts'] / max(total_requests, 1)
        
        # Adjust timeout based on timeout rate
        if timeout_rate > 0.1:  # More than 10% timeouts
            # Increase timeout to reduce timeouts
            self.batch_timeout = min(self.batch_timeout * 1.2, self.base_timeout * 3)
            logger.debug(f"Increased coordination timeout to {self.batch_timeout*1000:.1f}ms "
                        f"(timeout rate: {timeout_rate:.1%})")
        elif timeout_rate < 0.02 and self.batch_timeout > self.base_timeout:  # Less than 2% timeouts
            # Decrease timeout to improve latency
            self.batch_timeout = max(self.batch_timeout * 0.9, self.base_timeout)
            logger.debug(f"Decreased coordination timeout to {self.batch_timeout*1000:.1f}ms "
                        f"(timeout rate: {timeout_rate:.1%})")
        
        self.last_timeout_adjustment = current_time
    
    def _update_adaptive_parameters(self, params: 'AdaptiveParameters'):
        """Update coordination parameters based on adaptive tuning
        
        Args:
            params: New adaptive parameters from the tuner
        """
        try:
            # Update batch coordination parameters
            old_batch_size = self.max_batch_size
            old_timeout = self.batch_timeout * 1000
            
            self.max_batch_size = params.batch_size
            self.batch_timeout = params.coordination_timeout_ms / 1000.0
            
            # Log significant changes
            if abs(params.batch_size - old_batch_size) > 16 or abs(params.coordination_timeout_ms - old_timeout) > 20:
                logger.debug(f"Adaptive coordinator: batch_size {old_batch_size}->{params.batch_size}, "
                            f"timeout {old_timeout:.1f}->{params.coordination_timeout_ms:.1f}ms")
                
        except Exception as e:
            logger.error(f"Failed to update adaptive parameters: {e}")
    
    def update_simulation_count(self, simulation_count: int):
        """Update the current simulation count for adaptive tuning
        
        Args:
            simulation_count: Current simulation count per move
        """
        self.simulation_count = simulation_count
        
        # Force parameter update if simulation count changed significantly
        if self.adaptive_tuning_enabled and hasattr(self, 'parameter_tuner'):
            try:
                self.parameter_tuner.force_parameter_update(simulation_count)
            except Exception as e:
                logger.error(f"Failed to force parameter update: {e}")
    
    def _report_metrics_to_tuner(self):
        """Report current performance metrics to the adaptive tuner"""
        if not self.adaptive_tuning_enabled or not hasattr(self, 'parameter_tuner'):
            return
        
        current_time = time.time()
        
        # Report metrics every 2 seconds
        if current_time - self.last_metrics_report < 2.0:
            return
        
        try:
            # Calculate recent performance metrics
            total_requests = self.stats['requests_processed']
            total_time = current_time - self.last_metrics_report
            
            if total_time > 0 and total_requests > 0:
                sims_per_second = total_requests / total_time
                avg_batch_size = self.stats['avg_batch_size']
                
                # Report to adaptive tuner
                self.parameter_tuner.record_metrics(
                    simulations_per_second=sims_per_second,
                    simulation_count=self.simulation_count,
                    avg_batch_size=avg_batch_size,
                    avg_latency_ms=self.batch_timeout * 1000,
                    queue_depth=len(self.pending_requests),
                    active_workers=len(set(req[0].worker_id for req in self.pending_requests))
                )
                
            self.last_metrics_report = current_time
            
        except Exception as e:
            logger.error(f"Failed to report metrics to tuner: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.stats.copy()
        stats['current_timeout_ms'] = self.batch_timeout * 1000
        stats['timeout_rate'] = self.stats['coordination_timeouts'] / max(self.stats['requests_processed'], 1)
        return stats


# Global coordinator instance for shared use across workers (process-aware)
_global_coordinator: Optional[RequestBatchingCoordinator] = None
_coordinator_lock = threading.Lock()
_coordinator_process_id: Optional[int] = None


def get_global_batching_coordinator(max_batch_size: int = 64,
                                   batch_timeout_ms: float = 5.0,
                                   enable_cross_worker_batching: bool = True) -> RequestBatchingCoordinator:
    """Get or create the global batching coordinator instance (process-aware singleton)"""
    global _global_coordinator, _coordinator_process_id
    import os
    
    current_pid = os.getpid()
    
    with _coordinator_lock:
        # Check if we need a new instance for this process
        if _global_coordinator is None or _coordinator_process_id != current_pid:
            if _global_coordinator is not None:
                # Clean up old instance from different process
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