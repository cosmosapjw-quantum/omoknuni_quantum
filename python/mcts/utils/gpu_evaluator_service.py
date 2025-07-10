"""GPU Evaluator Service for efficient batch neural network evaluation

This service runs in the main process and handles all neural network evaluations
for self-play workers, maximizing GPU utilization through batching.
"""

import torch
import torch.nn as nn
import numpy as np
import queue
import threading
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import multiprocessing as mp
from collections import defaultdict
import psutil
import platform
import math

# Import batch coordinator
try:
    from .batch_evaluation_coordinator import BatchEvaluationRequest, BatchEvaluationResponse
    HAS_BATCH_COORDINATOR = True
except ImportError:
    HAS_BATCH_COORDINATOR = False

# Import adaptive parameter tuner for dynamic optimization
try:
    from .adaptive_parameter_tuner import get_global_parameter_tuner, AdaptiveParameters
    HAS_ADAPTIVE_TUNER = True
except ImportError:
    HAS_ADAPTIVE_TUNER = False

logger = logging.getLogger(__name__)


@dataclass
class EvaluationRequest:
    """Request for neural network evaluation"""
    request_id: int
    state: np.ndarray
    legal_mask: Optional[np.ndarray]
    temperature: float
    worker_id: int
    timestamp: float


@dataclass
class EvaluationResponse:
    """Response from neural network evaluation"""
    request_id: int
    policy: np.ndarray
    value: float
    worker_id: int


class GPUEvaluatorService:
    """Service that runs neural network evaluations on GPU for multiple workers
    
    This service runs in the main process and handles evaluation requests from
    self-play workers, batching them together for efficient GPU utilization.
    """
    
    # Class-level flag to track TensorRT initialization messages
    _tensorrt_logged = False
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda',
                 batch_size: Optional[int] = None,
                 batch_timeout: Optional[float] = None,
                 max_queue_size: Optional[int] = None,
                 workload_type: str = "balanced",
                 use_tensorrt: bool = True,
                 tensorrt_fp16: bool = True,
                 tensorrt_engine_path: Optional[str] = None):
        """Initialize GPU evaluator service
        
        Args:
            model: Neural network model to use for evaluation
            device: Device to run evaluations on
            batch_size: Maximum batch size for GPU evaluation (default: 256)
            batch_timeout: Maximum time to wait for a full batch in seconds (default: 0.01)
            max_queue_size: Maximum size of request/response queues (default: 10000)
            workload_type: Type of workload - "latency", "throughput", or "balanced"
            use_tensorrt: Enable TensorRT acceleration
            tensorrt_fp16: Use FP16 precision for TensorRT
            tensorrt_engine_path: Path to pre-compiled TensorRT engine
        """
        self.model = model
        self.device = torch.device(device)
        self.workload_type = workload_type
        
        # Use optimized defaults
        self.batch_size = batch_size if batch_size is not None else 256
        self.batch_timeout = batch_timeout if batch_timeout is not None else 0.01
        self.max_queue_size = max_queue_size if max_queue_size is not None else 10000
        
        # Check CUDA availability
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Please check your GPU setup.")
        
        # TensorRT setup
        self.use_tensorrt = use_tensorrt
        self.tensorrt_model = None
        
        if use_tensorrt:
            try:
                # Suppress TensorRT warnings at environment level
                import os
                old_trt_log_level = os.environ.get('TRT_LOGGER_VERBOSITY', '')
                os.environ['TRT_LOGGER_VERBOSITY'] = 'ERROR'
                
                from ..neural_networks.tensorrt_converter import optimize_for_hardware, HAS_TENSORRT
                if not HAS_TENSORRT:
                    if not GPUEvaluatorService._tensorrt_logged:
                        logger.warning("TensorRT requested but not available, falling back to PyTorch")
                        GPUEvaluatorService._tensorrt_logged = True
                    self.use_tensorrt = False
                else:
                    # Check if pre-compiled engine path is provided
                    if tensorrt_engine_path and os.path.exists(tensorrt_engine_path):
                        # Load pre-compiled TensorRT engine
                        if not GPUEvaluatorService._tensorrt_logged:
                            logger.info(f"Loading pre-compiled TensorRT engine from: {tensorrt_engine_path}")
                        else:
                            logger.debug(f"Loading pre-compiled TensorRT engine from: {tensorrt_engine_path}")
                        
                        # Use TensorRT evaluator to load the pre-compiled engine
                        from ..neural_networks.tensorrt_evaluator import TensorRTEvaluator
                        
                        try:
                            # Need to provide the model for proper initialization
                            self.tensorrt_model = TensorRTEvaluator(
                                model=model,  # Provide the PyTorch model
                                tensorrt_engine_path=tensorrt_engine_path,
                                device=str(self.device),
                                fp16_mode=tensorrt_fp16,
                                fallback_to_pytorch=False,  # We already have fallback logic here
                                network_config=None  # Engine already exists, no config needed
                            )
                        except Exception as e:
                            logger.warning(f"Failed to load TensorRT engine: {e}")
                            logger.info("Using PyTorch model instead of TensorRT")
                            self.use_tensorrt = False
                            self.tensorrt_model = None
                        
                        if not GPUEvaluatorService._tensorrt_logged:
                            logger.info("Successfully loaded pre-compiled TensorRT engine")
                            GPUEvaluatorService._tensorrt_logged = True
                    else:
                        # No pre-compiled engine, use centralized TensorRT manager
                        from .tensorrt_manager import get_tensorrt_manager
                        
                        # Get input shape from model metadata or assume Gomoku defaults
                        if hasattr(model, 'metadata') and hasattr(model.metadata, 'input_channels'):
                            input_channels = model.metadata.input_channels
                            board_size = model.metadata.board_size
                        else:
                            input_channels = 18  # Default for Gomoku
                            board_size = 15
                        
                        input_shape = (input_channels, board_size, board_size)
                        
                        # Get worker ID for logging
                        worker_id = os.getpid()
                        
                        # Get or convert model using centralized manager
                        tensorrt_manager = get_tensorrt_manager()
                        self.tensorrt_model = tensorrt_manager.get_or_convert_model(
                            pytorch_model=model,
                            input_shape=input_shape,
                            batch_size=batch_size,
                            fp16=True,
                            workspace_size=2 * (1 << 30),  # 2GB
                            worker_id=worker_id
                        )
                        
                        if self.tensorrt_model is None:
                            logger.warning(f"Worker {worker_id}: TensorRT conversion failed, falling back to PyTorch")
                            self.use_tensorrt = False
                        else:
                            if not GPUEvaluatorService._tensorrt_logged:
                                logger.info(f"Worker {worker_id}: Using TensorRT acceleration")
                                GPUEvaluatorService._tensorrt_logged = True
                    
                # Restore original TRT log level
                if old_trt_log_level:
                    os.environ['TRT_LOGGER_VERBOSITY'] = old_trt_log_level
                else:
                    os.environ.pop('TRT_LOGGER_VERBOSITY', None)
                    
                # Set CUDA stream for TensorRT model if available
                if self.tensorrt_model is not None:
                    if hasattr(self.tensorrt_model, 'cuda_stream') and hasattr(self, 'cuda_stream') and self.cuda_stream:
                        self.tensorrt_model.cuda_stream = self.cuda_stream
                    
                    # Use TensorRT model for inference
                    self.model = self.tensorrt_model
                    if not GPUEvaluatorService._tensorrt_logged:
                        logger.info("GPU service using TensorRT acceleration")
                    
            except Exception as e:
                logger.error(f"TensorRT conversion failed: {e}")
                logger.warning("Falling back to PyTorch model")
                self.use_tensorrt = False
        
        if not self.use_tensorrt:            
            # Move model to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
        
        # Create multiprocessing queues
        self.request_queue = mp.Queue(maxsize=self.max_queue_size)
        # Create separate response queues for each worker
        self.response_queues = {}
        
        # Phase 2.2 Ultra: CUDA stream for maximum parallelism
        if device == 'cuda' and torch.cuda.is_available():
            self.cuda_stream = torch.cuda.Stream()
        else:
            self.cuda_stream = None
        
        # Service thread
        self.service_thread = None
        self.stop_event = threading.Event()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'total_batches': 0,
            'total_time': 0.0,
            'avg_batch_size': 0.0,
            'avg_latency': 0.0
        }
        
        # Response routing
        self.pending_responses = defaultdict(dict)
        self.response_lock = threading.Lock()
        self.active_workers = set()  # Track active workers
        
        # Phase 3: Adaptive parameter tuning integration
        self.adaptive_tuning_enabled = HAS_ADAPTIVE_TUNER
        self.last_metrics_report = time.time()
        self.simulation_count = 100  # Default simulation count
        self.base_batch_size = self.batch_size  # Store original batch size
        self.base_batch_timeout = self.batch_timeout  # Store original timeout
        
        if self.adaptive_tuning_enabled:
            try:
                self.parameter_tuner = get_global_parameter_tuner()
                # Register callback to receive parameter updates
                self.parameter_tuner.register_parameter_callback(
                    'gpu_service', 
                    self._update_adaptive_parameters
                )
                logger.debug("Adaptive parameter tuning enabled for GPU evaluator service")
            except Exception as e:
                logger.warning(f"Failed to enable adaptive tuning: {e}")
                self.adaptive_tuning_enabled = False
        
    
    def start(self):
        """Start the evaluation service"""
        if self.service_thread is not None:
            raise RuntimeError("Service already started")
        
        self.stop_event.clear()
        self.service_thread = threading.Thread(target=self._service_loop, daemon=True)
        self.service_thread.start()
        
    
    def stop(self):
        """Stop the evaluation service"""
        if self.service_thread is None:
            return
        
        logger.debug("Stopping GPUEvaluatorService...")
        self.stop_event.set()
        
        # Send sentinel to unblock the service thread
        try:
            self.request_queue.put(None, block=False)
        except queue.Full:
            pass
        
        # Wait for thread to finish
        self.service_thread.join(timeout=5.0)
        if self.service_thread.is_alive():
            logger.warning("GPUEvaluatorService thread did not stop cleanly")
        
        self.service_thread = None
        logger.debug("GPUEvaluatorService stopped")
    
    def get_request_queue(self) -> mp.Queue:
        """Get the request queue for workers"""
        return self.request_queue
    
    def create_worker_queue(self, worker_id: int) -> mp.Queue:
        """Create a response queue for a specific worker"""
        if worker_id not in self.response_queues:
            self.response_queues[worker_id] = mp.Queue(maxsize=self.max_queue_size)
        self.active_workers.add(worker_id)
        return self.response_queues[worker_id]
    
    def cleanup_worker_queue(self, worker_id: int):
        """Clean up resources for a specific worker"""
        if worker_id in self.response_queues:
            # Clear any remaining messages
            q = self.response_queues[worker_id]
            try:
                while not q.empty():
                    q.get_nowait()
            except:
                pass
            del self.response_queues[worker_id]
        
        if worker_id in self.active_workers:
            self.active_workers.remove(worker_id)
    
    def _update_adaptive_parameters(self, params: 'AdaptiveParameters'):
        """Update GPU service parameters based on adaptive tuning
        
        Args:
            params: New adaptive parameters from the tuner
        """
        try:
            # Update GPU batch parameters
            old_batch_size = self.batch_size
            old_timeout = self.batch_timeout * 1000
            
            self.batch_size = params.batch_size
            self.batch_timeout = params.gpu_batch_timeout_ms / 1000.0
            
            # Log significant changes
            if abs(params.batch_size - old_batch_size) > 32 or abs(params.gpu_batch_timeout_ms - old_timeout) > 10:
                logger.debug(f"Adaptive GPU: batch_size {old_batch_size}->{params.batch_size}, "
                            f"timeout {old_timeout:.1f}->{params.gpu_batch_timeout_ms:.1f}ms")
                
        except Exception as e:
            logger.error(f"Failed to update adaptive GPU parameters: {e}")
    
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
                logger.error(f"Failed to force GPU parameter update: {e}")
    
    def _report_metrics_to_tuner(self):
        """Report current GPU performance metrics to the adaptive tuner"""
        if not self.adaptive_tuning_enabled or not hasattr(self, 'parameter_tuner'):
            return
        
        current_time = time.time()
        
        # Report metrics every 3 seconds
        if current_time - self.last_metrics_report < 3.0:
            return
        
        try:
            # Calculate recent performance metrics
            total_requests = self.stats['total_requests']
            total_time = self.stats['total_time']
            
            if total_time > 0 and total_requests > 0:
                sims_per_second = total_requests / total_time
                avg_latency = self.stats['avg_latency'] * 1000  # Convert to ms
                
                # Report to adaptive tuner
                self.parameter_tuner.record_metrics(
                    simulations_per_second=sims_per_second,
                    simulation_count=self.simulation_count,
                    avg_batch_size=self.stats['avg_batch_size'],
                    avg_latency_ms=avg_latency,
                    queue_depth=self.request_queue.qsize() if hasattr(self.request_queue, 'qsize') else 0,
                    active_workers=len(self.active_workers)
                )
                
            self.last_metrics_report = current_time
            
        except Exception as e:
            logger.error(f"Failed to report GPU metrics to tuner: {e}")
    
    def _service_loop(self):
        """Main service loop that processes evaluation requests"""
        
        while not self.stop_event.is_set():
            try:
                # Collect batch of requests
                batch_requests = self._collect_batch()
                
                if not batch_requests:
                    continue
                
                # Process legacy batch (optimized batches are processed immediately in _collect_batch)
                if batch_requests:
                    self._process_legacy_batch(batch_requests)
                
                # Report metrics to adaptive tuner
                self._report_metrics_to_tuner()
                
            except Exception as e:
                logger.error(f"Error in GPUEvaluatorService loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.01)  # Prevent tight error loop (reduced from 0.1s)
        
        logger.debug("GPUEvaluatorService loop stopped")
    
    def _collect_batch(self) -> List[Union[EvaluationRequest, 'BatchEvaluationRequest']]:
        """Collect a batch of requests from the queue (supports both legacy and optimized batch requests)"""
        batch = []
        deadline = time.time() + self.batch_timeout
        current_time = time.time()
        
        while len(batch) < self.batch_size and time.time() < deadline:
            timeout = max(0.001, deadline - time.time())
            
            try:
                # Get request from queue
                request = self.request_queue.get(timeout=timeout)
                
                # Check for sentinel
                if request is None:
                    if batch:
                        # Process current batch before stopping
                        break
                    else:
                        # No pending requests, can stop
                        return []
                
                # Handle both old EvaluationRequest and new BatchEvaluationRequest
                if HAS_BATCH_COORDINATOR and isinstance(request, BatchEvaluationRequest):
                    # New optimized batch request - process immediately for efficiency
                    if batch:
                        # Process current legacy batch first
                        self._process_legacy_batch(batch)
                        batch = []
                    
                    # Process optimized batch request immediately
                    self._process_optimized_batch_request(request)
                    continue
                
                # Legacy individual request
                # Filter out stale requests (older than 10 seconds)
                if current_time - request.timestamp > 10.0:
                    logger.warning(f"Dropping stale request from worker {request.worker_id}, age: {current_time - request.timestamp:.1f}s")
                    continue
                
                batch.append(request)
                
            except queue.Empty:
                # Timeout reached
                if batch:
                    break
        
        return batch
    
    def _process_optimized_batch_request(self, batch_request):
        """Process an optimized BatchEvaluationRequest with pre-batched states
        
        This method handles the new optimized batch format that reduces communication
        overhead by sending pre-batched arrays instead of individual requests.
        """
        if not HAS_BATCH_COORDINATOR:
            logger.error("Received BatchEvaluationRequest but coordinator not available")
            return
        
        start_time = time.time()
        batch_size = batch_request.states.shape[0]
        
        try:
            # Handle both numpy arrays and PyTorch tensors
            tensor_start = time.time()
            if isinstance(batch_request.states, torch.Tensor):
                state_tensor = batch_request.states.float().to(self.device)
            else:
                state_tensor = torch.from_numpy(batch_request.states).float().to(self.device)
            
            if batch_request.legal_masks is not None:
                if isinstance(batch_request.legal_masks, torch.Tensor):
                    legal_mask_tensor = batch_request.legal_masks.bool().to(self.device)
                else:
                    legal_mask_tensor = torch.from_numpy(batch_request.legal_masks).bool().to(self.device)
            else:
                legal_mask_tensor = None
            
            temperatures = batch_request.temperatures
            tensor_time = time.time() - tensor_start
            
            # ULTRA-OPTIMIZED GPU inference with ResNet forward_batch
            inference_start = time.time()
            
            # Check if we need temperature scaling and pre-process
            unique_temps = np.unique(temperatures)
            same_temperature = len(unique_temps) == 1
            temp_value = unique_temps[0] if same_temperature else 1.0
            
            if self.cuda_stream is not None:
                with torch.cuda.stream(self.cuda_stream):
                    # Use GPU-native forward_batch (no CPU-GPU transfers!)
                    if hasattr(self.model, 'forward_batch'):
                        # Let ResNet handle temperature internally (most efficient)
                        if same_temperature:
                            policies, values = self.model.forward_batch(state_tensor, legal_mask_tensor, temp_value)
                        else:
                            # For mixed temperatures, handle externally
                            policies, values = self.model.forward_batch(state_tensor, legal_mask_tensor, 1.0)
                            # Apply per-sample temperature scaling
                            temp_tensor = torch.from_numpy(temperatures).float().to(self.device, non_blocking=True)
                            # Re-apply temperature to log space equivalent
                            policies.log_().div_(temp_tensor.unsqueeze(1)).exp_()
                            # Renormalize
                            policies.div_(policies.sum(dim=1, keepdim=True))
                    else:
                        # Fallback to model.__call__
                        with torch.no_grad():
                            policy_logits, values = self.model(state_tensor)
                            # Apply temperature
                            if temp_value != 1.0:
                                policy_logits.div_(temp_value)
                            # Softmax
                            torch.softmax(policy_logits, dim=1, out=policy_logits)
                            policies = policy_logits
                
                # Single synchronization point
                self.cuda_stream.synchronize()
            else:
                # CPU fallback
                if hasattr(self.model, 'forward_batch'):
                    policies, values = self.model.forward_batch(state_tensor, legal_mask_tensor, temp_value)
                else:
                    with torch.no_grad():
                        policy_logits, values = self.model(state_tensor)
                        if temp_value != 1.0:
                            policy_logits = policy_logits / temp_value
                        policies = torch.softmax(policy_logits, dim=1)
            
            total_gpu_time = time.time() - inference_start
            
            # Single GPU->CPU transfer with pinned memory for speed
            convert_start = time.time()
            
            # Dynamic pinned memory allocation for any batch size
            max_buffer_size = max(1024, batch_size)  # At least 1024 for efficiency
            if not hasattr(self, '_cpu_policy_buffer') or self._cpu_policy_buffer.shape[0] < batch_size:
                # Create pinned memory buffers for faster transfers (dynamically sized)
                self._cpu_policy_buffer = torch.empty((max_buffer_size, 225), dtype=torch.float32, pin_memory=True)
                self._cpu_value_buffer = torch.empty((max_buffer_size,), dtype=torch.float32, pin_memory=True)
            
            # Always use pinned memory for maximum performance
            policy_slice = self._cpu_policy_buffer[:batch_size]
            value_slice = self._cpu_value_buffer[:batch_size]
            
            # Ultra-fast non-blocking transfer with proper error handling
            try:
                policy_slice.copy_(policies, non_blocking=True)
                # Ensure values tensor is 1D for copying
                values_flat = values.view(-1) if values.dim() > 1 else values
                value_slice.copy_(values_flat, non_blocking=True)
                
                # Synchronize only once at the end for maximum efficiency
                torch.cuda.synchronize()
                
                # Convert to numpy (no copy needed, just view)
                policies_np = policy_slice.numpy()
                values_np = value_slice.numpy()
                
            except RuntimeError as e:
                logger.warning(f"Pinned memory transfer failed: {e}, falling back to direct conversion")
                # Emergency fallback for shape mismatches
                policies_np = policy_logits.cpu().numpy()
                values_flat = values.view(-1) if values.dim() > 1 else values
                values_np = values_flat.cpu().numpy()
            
            convert_time = time.time() - convert_start
            
            # Create optimized batch response
            response = BatchEvaluationResponse(
                request_id=batch_request.request_id,
                policies=policies_np,
                values=values_np,
                worker_id=batch_request.worker_id,
                individual_request_ids=batch_request.individual_request_ids
            )
            
            # Send response to appropriate worker queue
            if batch_request.worker_id in self.response_queues:
                self.response_queues[batch_request.worker_id].put(response)
            elif batch_request.worker_id != -1:  # -1 indicates coordination requests without response queue
                logger.warning(f"No response queue for worker {batch_request.worker_id}")
            
            total_time = time.time() - start_time
            
            
        except Exception as e:
            logger.error(f"Error processing optimized batch {batch_request.request_id}: {e}")
            import traceback
            traceback.print_exc()
            
            # Send error response
            error_response = BatchEvaluationResponse(
                request_id=batch_request.request_id,
                policies=np.random.rand(batch_size, 225).astype(np.float32),
                values=np.zeros(batch_size, dtype=np.float32),
                worker_id=batch_request.worker_id,
                individual_request_ids=batch_request.individual_request_ids
            )
            
            if batch_request.worker_id in self.response_queues:
                self.response_queues[batch_request.worker_id].put(error_response)
    
    def _process_legacy_batch(self, requests: List[EvaluationRequest]):
        """Process a batch of legacy individual evaluation requests (backward compatibility)"""
        if not requests:
            return
        
        start_time = time.time()
        batch_size = len(requests)
        
        # Log that we're using legacy processing (for performance monitoring)
        logger.debug(f"Processing legacy batch: {batch_size} individual requests")
        
        try:
            # Prepare batch tensors with validation
            states = []
            legal_masks = []
            valid_requests = []
            
            expected_shape = (18, 15, 15)  # For Gomoku with 18 channels
            
            for i, req in enumerate(requests):
                try:
                    state = torch.from_numpy(req.state).float()
                    
                    # Validate state shape
                    if state.shape != expected_shape:
                        logger.warning(f"Invalid state shape at index {i}: {state.shape}, expected {expected_shape}")
                        # Send error response for invalid state
                        response = EvaluationResponse(
                            request_id=req.request_id,
                            policy=torch.zeros(225).numpy(),  # Default policy for 15x15 board
                            value=0.0,
                            worker_id=req.worker_id
                        )
                        if req.worker_id in self.response_queues:
                            self.response_queues[req.worker_id].put(response)
                        continue
                    
                    states.append(state)
                    valid_requests.append(req)
                    
                    if req.legal_mask is not None:
                        legal_masks.append(torch.from_numpy(req.legal_mask).bool())
                    else:
                        legal_masks.append(None)
                        
                except Exception as e:
                    logger.error(f"Error processing request {req.request_id}: {e}")
                    # Send error response
                    response = EvaluationResponse(
                        request_id=req.request_id,
                        policy=torch.zeros(225).numpy(),
                        value=0.0,
                        worker_id=req.worker_id
                    )
                    if req.worker_id in self.response_queues:
                        self.response_queues[req.worker_id].put(response)
                    continue
            
            # If no valid states, return early
            if not states:
                return
            
            # Track timing for performance analysis
            batch_start_time = time.time()
            
            # Stack states and move to device
            tensor_start = time.time()
            state_tensor = torch.stack(states).to(self.device)
            requests = valid_requests  # Update to only valid requests
            tensor_time = time.time() - tensor_start
            
            # Handle legal masks
            mask_start = time.time()
            if all(mask is not None for mask in legal_masks):
                legal_mask_tensor = torch.stack(legal_masks).to(self.device)
            else:
                legal_mask_tensor = None
            mask_time = time.time() - mask_start
            
            # Run evaluation - THE CRITICAL BOTTLENECK
            inference_start = time.time()
            with torch.no_grad():
                if hasattr(self.model, 'forward_batch'):
                    # Model supports batch forward with legal masks
                    policy_logits, values = self.model.forward_batch(state_tensor, legal_mask_tensor)
                else:
                    # Standard forward
                    policy_logits, values = self.model(state_tensor)
            inference_time = time.time() - inference_start
            
            # ULTRA-VECTORIZED: Apply temperature and softmax in pure tensor operations
            softmax_start = time.time()
            
            # Optimize for common case where all temperatures are the same
            temps = [req.temperature if req.temperature > 0 else 1.0 for req in requests]
            if len(set(temps)) == 1:
                # All temperatures are the same - use scalar division (fastest)
                if temps[0] != 1.0:
                    policy_logits = policy_logits / temps[0]
            else:
                # Different temperatures - use vectorized division
                temperatures = torch.tensor(temps, device=self.device, dtype=policy_logits.dtype)
                policy_logits = policy_logits / temperatures.unsqueeze(1)
            
            # Apply legal mask ONCE if provided (after temperature)
            if legal_mask_tensor is not None:
                policy_logits = policy_logits.masked_fill(~legal_mask_tensor, float('-inf'))
            
            # Single vectorized softmax for entire batch
            policies = torch.softmax(policy_logits, dim=1)
            
            softmax_time = time.time() - softmax_start
            
            # OPTIMIZED: Batch convert all tensors at once to avoid individual conversions
            response_start = time.time()
            policies_numpy = policies.cpu().numpy()  # Single batch conversion
            values_numpy = values.cpu().numpy()      # Single batch conversion
            conversion_time = time.time() - response_start
            
            # ULTRA-VECTORIZED: Minimize object creation and queue operations
            queue_start = time.time()
            
            # Group by worker_id using list comprehension (faster than defaultdict)
            worker_indices = {}
            for i, req in enumerate(requests):
                worker_id = req.worker_id
                if worker_id not in worker_indices:
                    worker_indices[worker_id] = []
                worker_indices[worker_id].append(i)
            
            # Send responses grouped by worker (reduces queue contention)
            for worker_id, indices in worker_indices.items():
                if worker_id not in self.response_queues:
                    if worker_id != -1:  # -1 indicates coordination requests
                        logger.error(f"No response queue for worker {worker_id}")
                    continue
                    
                try:
                    # Batch create responses for this worker
                    for i in indices:
                        req = requests[i]
                        response = EvaluationResponse(
                            request_id=req.request_id,
                            policy=policies_numpy[i],  # Already converted
                            value=float(values_numpy[i][0]),     # Extract scalar value  
                            worker_id=worker_id
                        )
                        self.response_queues[worker_id].put(response, timeout=1.0)
                        
                except queue.Full:
                    logger.error(f"Response queue full for worker {worker_id}")
                    # Skip this worker as it's likely dead
                    self.cleanup_worker_queue(worker_id)
            
            queue_time = time.time() - queue_start
            
            # Update statistics and log detailed timing
            elapsed = time.time() - start_time
            self.stats['total_requests'] += batch_size
            self.stats['total_batches'] += 1
            self.stats['total_time'] += elapsed
            self.stats['avg_batch_size'] = self.stats['total_requests'] / self.stats['total_batches']
            
            # Log detailed timing for performance analysis
            if not hasattr(self, '_batch_count'):
                self._batch_count = 0
            self._batch_count += 1
            
            # Update batch statistics
            self.stats['avg_latency'] = elapsed / batch_size
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            
            # Send error responses to unblock workers
            for req in requests:
                # Send dummy response to unblock workers
                # Use appropriate action size based on game type
                action_size = 225  # Default to Gomoku
                if hasattr(req, 'game_type'):
                    if req.game_type == 'chess':
                        action_size = 4096
                    elif req.game_type == 'go':
                        action_size = 361
                    elif req.game_type == 'gomoku':
                        action_size = 225
                
                response = EvaluationResponse(
                    request_id=req.request_id,
                    policy=np.ones(action_size) / action_size,  # Uniform policy
                    value=0.0,
                    worker_id=req.worker_id
                )
                worker_id = req.worker_id
                if worker_id in self.response_queues:
                    try:
                        self.response_queues[worker_id].put(response, timeout=1.0)
                    except queue.Full:
                        logger.error(f"Error response queue full for worker {worker_id}")
                        # Skip this worker as it's likely dead
                        self.cleanup_worker_queue(worker_id)
                elif worker_id != -1:  # -1 indicates coordination requests
                    logger.error(f"No response queue for worker {worker_id}")
    
    def get_stats(self) -> Dict[str, float]:
        """Get service statistics"""
        return self.stats.copy()


class RemoteEvaluator:
    """Evaluator that runs in worker process and communicates with GPU service"""
    
    def __init__(self, 
                 request_queue: mp.Queue,
                 response_queue: mp.Queue,
                 action_size: int,
                 worker_id: int = 0,
                 batch_timeout: float = 0.02):
        """Initialize remote evaluator
        
        Args:
            request_queue: Queue to send evaluation requests
            response_queue: Queue to receive evaluation responses
            action_size: Size of action space
            worker_id: ID of this worker
            batch_timeout: Timeout for response polling (optimized for performance)
        """
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.action_size = action_size
        self.worker_id = worker_id
        self.request_counter = 0
        self.batch_timeout = batch_timeout
        
        # Cache for responses
        self.response_cache = {}
        
        logger.debug(f"RemoteEvaluator created for worker {worker_id}")
    
    def evaluate(self, state: np.ndarray, 
                 legal_mask: Optional[np.ndarray] = None,
                 temperature: float = 1.0) -> Tuple[np.ndarray, float]:
        """Evaluate a state by sending request to GPU service
        
        Args:
            state: Game state to evaluate
            legal_mask: Mask of legal actions
            temperature: Temperature for policy (0 = deterministic)
            
        Returns:
            Tuple of (policy, value)
        """
        # Create request
        request_id = self.request_counter
        self.request_counter += 1
        
        request = EvaluationRequest(
            request_id=request_id,
            state=state,
            legal_mask=legal_mask,
            temperature=temperature,
            worker_id=self.worker_id,
            timestamp=time.time()
        )
        
        # Send request with error handling
        try:
            self.request_queue.put(request, timeout=1.0)
        except queue.Full:
            logger.error(f"Request queue full for worker {self.worker_id}")
            return np.ones(self.action_size) / self.action_size, 0.0
        
        # Wait for response with improved timeout handling
        timeout = 30.0  # Increased timeout for complex evaluations
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check for response
                response = self.response_queue.get(timeout=0.05)  # Shorter poll interval
                
                # Check if this is our response
                if response.worker_id == self.worker_id and response.request_id == request_id:
                    return response.policy, response.value
                else:
                    # Not our response, this shouldn't happen with separate queues
                    logger.warning(f"Worker {self.worker_id} received response for worker {response.worker_id}")
                    
            except queue.Empty:
                # Check if we should give up early
                if time.time() - request.timestamp > timeout * 2:
                    logger.warning(f"Request {request_id} is too old, giving up")
                    break
                continue
        
        # Timeout - return uniform policy (reduce logging verbosity)
        if hasattr(self, '_timeout_count'):
            self._timeout_count += 1
        else:
            self._timeout_count = 1
        
        # Only log first few timeouts to avoid spam
        if self._timeout_count <= 3:
            logger.warning(f"RemoteEvaluator timeout for worker {self.worker_id} (#{self._timeout_count})")
        elif self._timeout_count == 4:
            logger.warning(f"RemoteEvaluator worker {self.worker_id}: suppressing further timeout messages")
        
        return np.ones(self.action_size) / self.action_size, 0.0
    
    def evaluate_batch(self, states: Union[List[np.ndarray], np.ndarray], 
                       legal_masks: Optional[Union[List[np.ndarray], np.ndarray]] = None,
                       temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch of states by sending requests to GPU service
        
        Args:
            states: Batch of game states to evaluate
            legal_masks: Batch of masks of legal actions
            temperature: Temperature for policy (0 = deterministic)
            
        Returns:
            Tuple of (policies, values) as numpy arrays
        """
        batch_size = len(states) if hasattr(states, '__len__') else states.shape[0]
        
        # Send all requests
        request_ids = []
        for i in range(batch_size):
            state = states[i]
            legal_mask = legal_masks[i] if legal_masks is not None else None
            
            request_id = self.request_counter
            self.request_counter += 1
            request_ids.append(request_id)
            
            request = EvaluationRequest(
                request_id=request_id,
                state=state,
                legal_mask=legal_mask,
                temperature=temperature,
                worker_id=self.worker_id,
                timestamp=time.time()
            )
            
            self.request_queue.put(request)
        
        # Collect responses
        responses = {}
        timeout = 10.0  # 10 second timeout
        start_time = time.time()
        
        while len(responses) < batch_size and time.time() - start_time < timeout:
            try:
                # Use optimized timeout from hardware configuration, not hardcoded 0.1s
                # This dramatically improves performance by reducing polling overhead
                response = self.response_queue.get(timeout=self.batch_timeout)
                
                # Check if this is one of our responses
                if response.worker_id == self.worker_id and response.request_id in request_ids:
                    responses[response.request_id] = response
                else:
                    # Not our response
                    logger.warning(f"Worker {self.worker_id} received response for worker {response.worker_id}")
                    
            except queue.Empty:
                continue
        
        # Assemble results in order
        policies = []
        values = []
        
        for request_id in request_ids:
            if request_id in responses:
                response = responses[request_id]
                policies.append(response.policy)
                values.append(response.value)
            else:
                # Timeout - use uniform policy (reduce verbosity)
                policies.append(np.ones(self.action_size) / self.action_size)
                values.append(0.0)
        
        return np.stack(policies), np.array(values)