"""GPU Evaluator Service for efficient batch neural network evaluation

This service runs in the main process and handles all neural network evaluations
for self-play workers, maximizing GPU utilization through batching.
"""

import os
# Set CUDA initialization mode for multiprocessing
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda',
                 batch_size: int = 256,
                 batch_timeout: float = 0.01,
                 max_queue_size: int = 10000):
        """Initialize GPU evaluator service
        
        Args:
            model: Neural network model to use for evaluation
            device: Device to run evaluations on
            batch_size: Maximum batch size for GPU evaluation
            batch_timeout: Maximum time to wait for a full batch (seconds)
            max_queue_size: Maximum size of request/response queues
        """
        self.model = model
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.max_queue_size = max_queue_size
        
        # Check CUDA availability and move model to device
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = torch.device('cpu')
            device = 'cpu'
            
        # Initialize CUDA if needed
        if device == 'cuda':
            try:
                # Test CUDA availability
                torch.cuda.init()
                torch.cuda.synchronize()
                # Test tensor creation
                test_tensor = torch.zeros(1, device=self.device)
                del test_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"CUDA initialization failed: {e}")
                logger.warning("Falling back to CPU")
                self.device = torch.device('cpu')
                device = 'cpu'
        
        # Move model to device and set to eval mode
        try:
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to move model to {device}: {e}")
            if device == 'cuda':
                logger.warning("Falling back to CPU")
                self.device = torch.device('cpu')
                self.model.to(self.device)
                self.model.eval()
        
        # Create multiprocessing queues
        self.request_queue = mp.Queue(maxsize=max_queue_size)
        # Create separate response queues for each worker
        self.response_queues = {}
        
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
        
        logger.debug(f"GPUEvaluatorService initialized on {device} with batch_size={batch_size}")
    
    def start(self):
        """Start the evaluation service"""
        if self.service_thread is not None:
            raise RuntimeError("Service already started")
        
        self.stop_event.clear()
        self.service_thread = threading.Thread(target=self._service_loop, daemon=True)
        self.service_thread.start()
        
        logger.debug("GPUEvaluatorService started")
    
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
        return self.response_queues[worker_id]
    
    def _service_loop(self):
        """Main service loop that processes evaluation requests"""
        logger.debug("GPUEvaluatorService loop started")
        
        while not self.stop_event.is_set():
            try:
                # Collect batch of requests
                batch_requests = self._collect_batch()
                
                if not batch_requests:
                    continue
                
                # Process batch
                self._process_batch(batch_requests)
                
            except Exception as e:
                logger.error(f"Error in GPUEvaluatorService loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)  # Prevent tight error loop
        
        logger.debug("GPUEvaluatorService loop stopped")
    
    def _collect_batch(self) -> List[EvaluationRequest]:
        """Collect a batch of requests from the queue"""
        batch = []
        deadline = time.time() + self.batch_timeout
        
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
                
                batch.append(request)
                
            except queue.Empty:
                # Timeout reached
                if batch:
                    break
        
        return batch
    
    def _process_batch(self, requests: List[EvaluationRequest]):
        """Process a batch of evaluation requests"""
        if not requests:
            return
        
        start_time = time.time()
        batch_size = len(requests)
        
        try:
            # Prepare batch tensors
            states = []
            legal_masks = []
            
            for req in requests:
                states.append(torch.from_numpy(req.state).float())
                if req.legal_mask is not None:
                    legal_masks.append(torch.from_numpy(req.legal_mask).bool())
                else:
                    legal_masks.append(None)
            
            # Stack states with error handling
            try:
                state_tensor = torch.stack(states).to(self.device)
                
                # Handle legal masks
                if all(mask is not None for mask in legal_masks):
                    legal_mask_tensor = torch.stack(legal_masks).to(self.device)
                else:
                    legal_mask_tensor = None
            except RuntimeError as e:
                if "CUDA" in str(e) and self.device.type == 'cuda':
                    logger.error(f"CUDA error during tensor transfer: {e}")
                    logger.warning("Attempting to recover by clearing cache and retrying on CPU")
                    
                    # Clear CUDA cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Fall back to CPU for this batch
                    cpu_device = torch.device('cpu')
                    state_tensor = torch.stack(states).to(cpu_device)
                    
                    if all(mask is not None for mask in legal_masks):
                        legal_mask_tensor = torch.stack(legal_masks).to(cpu_device)
                    else:
                        legal_mask_tensor = None
                    
                    # Move model to CPU temporarily
                    self.model.to(cpu_device)
                    
                    # Process on CPU
                    with torch.no_grad():
                        if hasattr(self.model, 'forward_batch'):
                            policy_logits, values = self.model.forward_batch(state_tensor, legal_mask_tensor)
                        else:
                            policy_logits, values = self.model(state_tensor)
                            if legal_mask_tensor is not None:
                                policy_logits = policy_logits.masked_fill(~legal_mask_tensor, float('-inf'))
                    
                    # Try to move model back to GPU
                    try:
                        self.model.to(self.device)
                    except:
                        logger.error("Failed to move model back to GPU, continuing on CPU")
                        self.device = cpu_device
                    
                    # Continue processing responses
                    policies = []
                    for i, req in enumerate(requests):
                        logits = policy_logits[i]
                        
                        if req.temperature > 0:
                            probs = torch.softmax(logits / req.temperature, dim=-1)
                        else:
                            probs = torch.zeros_like(logits)
                            probs[logits.argmax()] = 1.0
                        
                        policies.append(probs.cpu().numpy())
                    
                    values_np = values.squeeze(-1).cpu().numpy()
                    
                    # Send responses
                    for i, req in enumerate(requests):
                        response = EvaluationResponse(
                            request_id=req.request_id,
                            policy=policies[i],
                            value=float(values_np[i]),
                            worker_id=req.worker_id
                        )
                        
                        if req.worker_id in self.response_queues:
                            try:
                                self.response_queues[req.worker_id].put(response)
                            except:
                                logger.error(f"Failed to send response to worker {req.worker_id}")
                    
                    return
                else:
                    raise
            
            # Run evaluation
            with torch.no_grad():
                if hasattr(self.model, 'forward_batch'):
                    # Model supports batch forward with legal masks
                    policy_logits, values = self.model.forward_batch(state_tensor, legal_mask_tensor)
                else:
                    # Standard forward
                    policy_logits, values = self.model(state_tensor)
                    
                    # Apply legal mask if provided
                    if legal_mask_tensor is not None:
                        policy_logits = policy_logits.masked_fill(~legal_mask_tensor, float('-inf'))
                
                # Apply temperature and softmax
                policies = []
                for i, req in enumerate(requests):
                    logits = policy_logits[i]
                    
                    # Apply temperature
                    if req.temperature > 0:
                        logits = logits / req.temperature
                    
                    # Softmax to get probabilities
                    if legal_mask_tensor is not None:
                        # Masked softmax
                        mask = legal_mask_tensor[i]
                        exp_logits = torch.exp(logits - logits[mask].max())
                        exp_logits[~mask] = 0
                        policy = exp_logits / exp_logits.sum()
                    else:
                        policy = torch.softmax(logits, dim=0)
                    
                    policies.append(policy)
                
                policies = torch.stack(policies)
            
            # Send responses
            for i, req in enumerate(requests):
                response = EvaluationResponse(
                    request_id=req.request_id,
                    policy=policies[i].cpu().numpy(),
                    value=values[i].item(),
                    worker_id=req.worker_id
                )
                
                # Put response in the worker's specific queue
                worker_id = req.worker_id
                if worker_id in self.response_queues:
                    self.response_queues[worker_id].put(response)
                else:
                    logger.error(f"No response queue for worker {worker_id}")
            
            # Update statistics
            elapsed = time.time() - start_time
            self.stats['total_requests'] += batch_size
            self.stats['total_batches'] += 1
            self.stats['total_time'] += elapsed
            self.stats['avg_batch_size'] = self.stats['total_requests'] / self.stats['total_batches']
            self.stats['avg_latency'] = elapsed / batch_size
            
            if self.stats['total_batches'] % 100 == 0:
                logger.debug(f"GPUEvaluatorService stats: {self.stats}")
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            
            # Check if this is a CUDA error
            if "CUDA" in str(e) and self.device.type == 'cuda':
                logger.warning("CUDA error detected, attempting recovery")
                
                # Clear CUDA cache and reset
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    except:
                        pass
                
                # Check if we should permanently switch to CPU
                self.cuda_error_count = getattr(self, 'cuda_error_count', 0) + 1
                if self.cuda_error_count >= 3:
                    logger.error("Multiple CUDA errors detected, permanently switching to CPU")
                    self.device = torch.device('cpu')
                    try:
                        self.model.to(self.device)
                    except:
                        logger.error("Failed to move model to CPU")
            
            # Send error responses
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
                    self.response_queues[worker_id].put(response)
                else:
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
                 worker_id: int = 0):
        """Initialize remote evaluator
        
        Args:
            request_queue: Queue to send evaluation requests
            response_queue: Queue to receive evaluation responses
            action_size: Size of action space
            worker_id: ID of this worker
        """
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.action_size = action_size
        self.worker_id = worker_id
        self.request_counter = 0
        
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
        
        # Send request
        self.request_queue.put(request)
        
        # Wait for response
        timeout = 10.0  # 10 second timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check for response
                response = self.response_queue.get(timeout=0.1)
                
                # Check if this is our response
                if response.worker_id == self.worker_id and response.request_id == request_id:
                    return response.policy, response.value
                else:
                    # Not our response, cache it for the right worker
                    # In practice, workers should only get their own responses
                    logger.warning(f"Worker {self.worker_id} received response for worker {response.worker_id}")
                    
            except queue.Empty:
                continue
        
        # Timeout - return uniform policy
        logger.error(f"RemoteEvaluator timeout for worker {self.worker_id}, request {request_id}")
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
                # Check for response
                response = self.response_queue.get(timeout=0.1)
                
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
                # Timeout - use uniform policy
                logger.error(f"RemoteEvaluator timeout for worker {self.worker_id}, request {request_id}")
                policies.append(np.ones(self.action_size) / self.action_size)
                values.append(0.0)
        
        return np.stack(policies), np.array(values)