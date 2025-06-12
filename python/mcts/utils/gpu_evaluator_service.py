"""GPU Evaluator Service using producer-consumer pattern

This module provides a service that runs in the main process and handles
all neural network evaluations on GPU, while worker processes handle game logic.
Uses a clean producer-consumer pattern for communication.
"""

import logging
import multiprocessing as mp
from multiprocessing import Process, Queue, Event
import queue
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import torch
import torch.nn as nn
import uuid

logger = logging.getLogger(__name__)


@dataclass
class EvaluationRequest:
    """Request for neural network evaluation"""
    request_id: str
    state: np.ndarray
    legal_mask: Optional[np.ndarray]
    temperature: float
    

@dataclass
class EvaluationResponse:
    """Response from neural network evaluation"""
    request_id: str
    policy: np.ndarray
    value: float
    error: Optional[str] = None


class GPUEvaluatorService:
    """Producer-consumer pattern GPU evaluation service
    
    Runs in a separate process and consumes evaluation requests from workers,
    processes them on GPU, and returns results.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda', 
                 batch_size: int = 256, batch_timeout: float = 0.01,
                 num_consumers: int = 1):
        """Initialize GPU evaluator service
        
        Args:
            model: Neural network model
            device: Device to run on ('cuda' or 'cpu')
            batch_size: Maximum batch size for evaluation
            batch_timeout: Maximum time to wait before evaluating partial batch
            num_consumers: Number of consumer processes (usually 1 for single GPU)
        """
        self.model = model
        self.device = device  # Store as string, will convert in process
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.num_consumers = num_consumers
        
        # Communication queues
        self.request_queue = mp.Queue(maxsize=1000)
        self.response_queue = mp.Queue(maxsize=1000)
        
        # Control
        self.shutdown_event = mp.Event()
        self.consumers = []
        
        logger.info(f"[GPU Service] Initialized with device: {device}, batch_size: {batch_size}")
    
    def start(self):
        """Start the consumer processes"""
        # CRITICAL: Move model to CPU before getting state dict
        self.model.cpu()
        cpu_state_dict = self.model.state_dict()
        
        model_class = type(self.model)
        
        # Try to get initialization kwargs
        model_kwargs = {}
        if hasattr(self.model, '_init_kwargs'):
            model_kwargs = self.model._init_kwargs
        elif hasattr(self.model, 'config'):
            # For models that take a config object
            model_kwargs = {'config': self.model.config}
        
        # Get model dimensions for fallback
        input_channels = 20  # Enhanced representation
        board_size = 15  # Gomoku default
        num_actions = 225  # 15x15
        
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'input_channels'):
                input_channels = self.model.config.input_channels
            if hasattr(self.model.config, 'input_height'):
                board_size = self.model.config.input_height
            if hasattr(self.model.config, 'num_actions'):
                num_actions = self.model.config.num_actions
        
        for i in range(self.num_consumers):
            consumer = Process(
                target=_gpu_consumer_process,
                args=(i, cpu_state_dict, model_class, model_kwargs,
                      self.device, self.batch_size, self.batch_timeout,
                      self.request_queue, self.response_queue, self.shutdown_event,
                      input_channels, board_size, num_actions)
            )
            consumer.start()
            self.consumers.append(consumer)
            
        logger.info(f"[GPU Service] Started {self.num_consumers} consumer processes")
    
    def stop(self):
        """Stop all consumer processes"""
        logger.info("[GPU Service] Stopping consumers...")
        self.shutdown_event.set()
        
        # Send poison pills to unblock consumers
        for _ in range(self.num_consumers):
            try:
                self.request_queue.put(None, timeout=0.1)
            except:
                pass
                
        # Wait for consumers to finish
        for consumer in self.consumers:
            consumer.join(timeout=5.0)
            if consumer.is_alive():
                logger.warning(f"[GPU Service] Force terminating consumer")
                consumer.terminate()
                
        logger.info("[GPU Service] All consumers stopped")
    
    def get_queues(self) -> Tuple[Queue, Queue]:
        """Get request and response queues for workers"""
        return self.request_queue, self.response_queue


def _gpu_consumer_process(consumer_id: int, model_state_dict: Dict, 
                         model_class: type, model_kwargs: Dict,
                         device: str, batch_size: int, batch_timeout: float,
                         request_queue: Queue, response_queue: Queue, 
                         shutdown_event: Event,
                         input_channels: int, board_size: int, num_actions: int):
    """GPU consumer process that handles evaluation requests"""
    import os
    import sys
    
    # Set up process
    print(f"[GPU Consumer {consumer_id}] Starting, PID: {os.getpid()}", file=sys.stderr)
    print(f"[GPU Consumer {consumer_id}] CUDA available: {torch.cuda.is_available()}", file=sys.stderr)
    
    try:
        # Recreate model with error handling
        try:
            if model_kwargs:
                model = model_class(**model_kwargs)
            else:
                model = model_class()
            model.load_state_dict(model_state_dict)
        except Exception as e:
            # Fallback: create model with explicit dimensions
            print(f"[GPU Consumer {consumer_id}] Model creation failed: {e}, using fallback", file=sys.stderr)
            from mcts.neural_networks.nn_model import AlphaZeroNetwork, ModelConfig
            config = ModelConfig(
                input_channels=input_channels,
                input_height=board_size,
                input_width=board_size,
                num_actions=num_actions,
                num_res_blocks=10,
                num_filters=128
            )
            model = AlphaZeroNetwork(config)
            model.load_state_dict(model_state_dict)
        
        # Move to device
        device_obj = torch.device(device)
        model.to(device_obj)
        model.eval()
        
        logger.info(f"[GPU Consumer {consumer_id}] Model loaded on {device}")
        
        # Batch collection
        batch_requests: List[Tuple[EvaluationRequest, float]] = []  # (request, timestamp)
        
        while not shutdown_event.is_set():
            try:
                # Try to get request with timeout
                remaining_timeout = batch_timeout
                if batch_requests:
                    # Adjust timeout based on oldest request
                    elapsed = time.time() - batch_requests[0][1]
                    remaining_timeout = max(0.001, batch_timeout - elapsed)
                
                request = request_queue.get(timeout=remaining_timeout)
                
                # Check for poison pill
                if request is None:
                    break
                
                # Ensure request state is numpy array (not tensor)
                if hasattr(request, 'state') and torch.is_tensor(request.state):
                    request.state = request.state.cpu().numpy()
                    
                batch_requests.append((request, time.time()))
                
                # Check if we should process
                should_process = (
                    len(batch_requests) >= batch_size or
                    (time.time() - batch_requests[0][1]) >= batch_timeout
                )
                
                if should_process:
                    _process_batch(model, device_obj, batch_requests, response_queue)
                    batch_requests = []
                    
            except queue.Empty:
                # Timeout - process any pending requests
                if batch_requests:
                    _process_batch(model, device_obj, batch_requests, response_queue)
                    batch_requests = []
            except Exception as e:
                logger.error(f"[GPU Consumer {consumer_id}] Error: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        logger.error(f"[GPU Consumer {consumer_id}] Fatal error: {e}")
        import traceback
        traceback.print_exc()


def _process_batch(model: nn.Module, device: torch.device, 
                   batch_requests: List[Tuple[EvaluationRequest, float]], 
                   response_queue: Queue):
    """Process a batch of evaluation requests"""
    if not batch_requests:
        return
        
    requests = [req for req, _ in batch_requests]
    
    try:
        # Stack states
        states = np.stack([req.state for req in requests])
        states_tensor = torch.from_numpy(states).float().to(device)
        
        # Run model
        with torch.no_grad():
            policy_logits, values = model(states_tensor)
        
        # Process each request
        for i, req in enumerate(requests):
            logits = policy_logits[i]
            
            # Apply temperature
            if req.temperature != 1.0:
                logits = logits / req.temperature
            
            # Apply legal mask
            if req.legal_mask is not None:
                mask = torch.from_numpy(req.legal_mask).bool().to(device)
                logits = logits.masked_fill(~mask, -float('inf'))
            
            # Softmax
            policy = torch.softmax(logits, dim=-1)
            
            # Create response
            response = EvaluationResponse(
                request_id=req.request_id,
                policy=policy.cpu().numpy(),
                value=values[i].item()
            )
            
            # Put response with timeout to avoid blocking
            try:
                response_queue.put(response, timeout=1.0)
            except queue.Full:
                logger.warning(f"[GPU Batch] Response queue full for request {req.request_id}")
            
    except Exception as e:
        logger.error(f"[GPU Batch] Processing error: {e}")
        # Send error responses
        for req in requests:
            response = EvaluationResponse(
                request_id=req.request_id,
                policy=np.ones(361) / 361,  # Uniform
                value=0.0,
                error=str(e)
            )
            # Put response with timeout to avoid blocking
            try:
                response_queue.put(response, timeout=1.0)
            except queue.Full:
                logger.warning(f"[GPU Batch] Response queue full for request {req.request_id}")


class RemoteEvaluator:
    """Evaluator for workers using producer-consumer pattern"""
    
    def __init__(self, request_queue: Queue, response_queue: Queue, 
                 action_size: int, worker_id: int = 0):
        """Initialize remote evaluator
        
        Args:
            request_queue: Queue to send requests to
            response_queue: Queue to receive responses from
            action_size: Size of action space
            worker_id: ID of this worker
        """
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.action_size = action_size
        self.worker_id = worker_id
        
        # Track pending requests
        self.pending_requests: Dict[str, float] = {}
        
    def evaluate(self, state: np.ndarray, legal_mask: Optional[np.ndarray] = None,
                temperature: float = 1.0) -> Tuple[np.ndarray, float]:
        """Evaluate a single position"""
        request_id = f"{self.worker_id}_{uuid.uuid4().hex[:8]}"
        
        # Ensure state is numpy array
        if torch.is_tensor(state):
            state = state.cpu().numpy()
        
        request = EvaluationRequest(
            request_id=request_id,
            state=state.astype(np.float32),  # Ensure float32
            legal_mask=legal_mask,
            temperature=temperature
        )
        
        # Send request
        self.request_queue.put(request)
        self.pending_requests[request_id] = time.time()
        
        # Wait for response
        max_wait = 10.0
        start_time = time.time()
        
        while True:
            try:
                response = self.response_queue.get(timeout=0.1)
                
                if response.request_id == request_id:
                    # Our response
                    del self.pending_requests[request_id]
                    if response.error:
                        raise RuntimeError(f"GPU evaluation error: {response.error}")
                    return response.policy, response.value
                else:
                    # Not our response, put it back
                    self.response_queue.put(response)
                    
            except queue.Empty:
                if time.time() - start_time > max_wait:
                    raise TimeoutError(f"Evaluation timeout for request {request_id}")
                    
    def evaluate_batch(self, states: np.ndarray, 
                      legal_masks: Optional[np.ndarray] = None,
                      temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch of positions"""
        # Ensure states is numpy array
        if torch.is_tensor(states):
            states = states.cpu().numpy()
            
        batch_size = states.shape[0]
        request_ids = []
        
        # Send all requests
        for i in range(batch_size):
            request_id = f"{self.worker_id}_{uuid.uuid4().hex[:8]}"
            
            # Ensure individual state is numpy
            state = states[i]
            if torch.is_tensor(state):
                state = state.cpu().numpy()
                
            request = EvaluationRequest(
                request_id=request_id,
                state=state.astype(np.float32),  # Ensure float32
                legal_mask=legal_masks[i] if legal_masks is not None else None,
                temperature=temperature
            )
            self.request_queue.put(request)
            request_ids.append(request_id)
            self.pending_requests[request_id] = time.time()
        
        # Collect responses
        policies = np.zeros((batch_size, self.action_size))
        values = np.zeros(batch_size)
        responses_received = {}
        
        max_wait = 10.0
        start_time = time.time()
        
        while len(responses_received) < batch_size:
            try:
                response = self.response_queue.get(timeout=0.1)
                
                if response.request_id in request_ids:
                    # One of our responses
                    responses_received[response.request_id] = response
                    del self.pending_requests[response.request_id]
                else:
                    # Not ours, put it back
                    self.response_queue.put(response)
                    
            except queue.Empty:
                if time.time() - start_time > max_wait:
                    raise TimeoutError(f"Batch evaluation timeout")
        
        # Assemble results in correct order
        for i, request_id in enumerate(request_ids):
            response = responses_received[request_id]
            if response.error:
                raise RuntimeError(f"GPU evaluation error: {response.error}")
            policies[i] = response.policy
            values[i] = response.value
            
        return policies, values