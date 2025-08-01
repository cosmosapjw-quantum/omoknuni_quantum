"""Single GPU Evaluator - Optimized neural network evaluation for single-GPU environments

This module consolidates the best features from direct_gpu_evaluator.py and
optimized_direct_gpu_evaluator.py into a single, clean implementation.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Tuple, Optional, Union, Dict, Any
import time
from contextlib import nullcontext

logger = logging.getLogger(__name__)


class SingleGPUEvaluator:
    """Optimized evaluator for single-GPU environments with zero CPU transfers
    
    This evaluator combines all optimizations for maximum performance:
    1. Zero CPU-GPU transfers (all operations stay on GPU)
    2. Direct tensor operations with optional numpy compatibility
    3. Mixed precision (FP16) support for 2x throughput
    4. Vectorized policy processing
    5. CUDA stream support for async operations
    6. TensorRT support (if available)
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda',
                 action_size: int = 225,
                 batch_size: int = 512,
                 use_mixed_precision: bool = True,
                 use_tensorrt: bool = False,
                 enable_timing: bool = False):
        """Initialize single-GPU evaluator
        
        Args:
            model: PyTorch neural network model or TensorRT model
            device: Device to run on ('cuda' or 'cpu')
            action_size: Size of action space
            batch_size: Maximum batch size for GPU processing
            use_mixed_precision: Whether to use FP16 inference
            use_tensorrt: Whether model is TensorRT optimized
        """
        self.model = model
        self.device = torch.device(device)
        self.action_size = action_size
        self.batch_size = batch_size
        self.use_mixed_precision = use_mixed_precision and device == 'cuda'
        self.use_tensorrt = use_tensorrt
        self.enable_timing = enable_timing
        
        # Flag to control output format (tensors vs numpy)
        self._return_torch_tensors = True
        
        # Move model to device if needed
        if not use_tensorrt and hasattr(model, 'to'):
            self.model.to(self.device)
            self.model.eval()
        
        # Create CUDA stream for async operations
        self.cuda_stream = torch.cuda.Stream() if device == 'cuda' else None
        
        # CUDA graph support
        self.use_cuda_graph = device == 'cuda' and torch.cuda.is_available()
        self.cuda_graph = None
        self.graph_captured = False
        self.graph_batch_size = batch_size
            
        # Pre-allocate constants for efficiency
        self.neg_inf = torch.tensor(-1e9, device=self.device, dtype=torch.float32)
        
        # Performance statistics
        self.stats = {
            'total_evaluations': 0,
            'total_batches': 0,
            'total_time': 0.0,
            'gpu_time': 0.0,
            'avg_batch_size': 0.0
        }
    
    def evaluate(self, 
                state: Union[np.ndarray, torch.Tensor], 
                legal_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
                temperature: float = 1.0) -> Tuple[Union[np.ndarray, torch.Tensor], Union[float, torch.Tensor]]:
        """Evaluate a single state
        
        Args:
            state: Game state array or tensor
            legal_mask: Boolean mask of legal actions
            temperature: Temperature for policy sampling
            
        Returns:
            Tuple of (policy, value) as tensors or numpy based on _return_torch_tensors
        """
        # Handle single state by adding batch dimension
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device, dtype=torch.float32)
        
        if state.dim() == 3:  # Add batch dimension
            state = state.unsqueeze(0)
            
        if legal_mask is not None:
            if isinstance(legal_mask, np.ndarray):
                legal_mask = torch.from_numpy(legal_mask).to(self.device, dtype=torch.bool)
            if legal_mask.dim() == 1:
                legal_mask = legal_mask.unsqueeze(0)
        
        # Use batch evaluation
        policies, values = self.evaluate_batch(state, legal_mask)
        
        # Apply temperature if needed
        if temperature != 1.0 and temperature > 0:
            policies = self._apply_temperature_vectorized(policies, temperature)
        
        # Extract single result
        policy = policies[0]
        value = values[0]
        
        if not self._return_torch_tensors:
            if isinstance(policy, torch.Tensor):
                return policy.cpu().numpy(), value.cpu().item()
            else:
                # Already numpy arrays
                return policy, value.item() if hasattr(value, 'item') else value
        return policy, value
    
    def evaluate_batch(self,
                      states: Union[np.ndarray, torch.Tensor],
                      legal_masks: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate a batch of states with zero CPU transfers
        
        Args:
            states: Batch of game states
            legal_masks: Batch of legal action masks
            
        Returns:
            Tuple of (policies, values) as GPU tensors
        """
        start_time = time.time()
        
        # Use CUDA stream for async execution if available
        stream_context = torch.cuda.stream(self.cuda_stream) if self.cuda_stream else nullcontext()
        
        with stream_context:
            with torch.no_grad():
                # Convert inputs to GPU tensors if needed
                if isinstance(states, np.ndarray):
                    state_tensor = torch.from_numpy(states).to(self.device, non_blocking=True, dtype=torch.float32)
                else:
                    state_tensor = states.to(self.device, non_blocking=True, dtype=torch.float32)
                
                # Record GPU time only if timing is enabled
                if self.enable_timing and self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    gpu_start = time.time()
                
                # Forward pass with mixed precision
                if self.use_mixed_precision and not self.use_tensorrt:
                    with torch.amp.autocast('cuda'):
                        policy_logits, values = self.model(state_tensor)
                        # Ensure outputs are FP32 for stability
                        policy_logits = policy_logits.float()
                        values = values.float()
                else:
                    policy_logits, values = self.model(state_tensor)
                
                # Process values shape
                if values.dim() > 1:
                    values = values.squeeze(-1)
                
                # Vectorized policy processing (all on GPU)
                policies = self._process_policies_vectorized(policy_logits, legal_masks)
                
                if self.enable_timing and self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    self.stats['gpu_time'] += time.time() - gpu_start
        
        # Update statistics
        batch_size = len(states) if isinstance(states, np.ndarray) else states.shape[0]
        self.stats['total_evaluations'] += batch_size
        self.stats['total_batches'] += 1
        self.stats['total_time'] += time.time() - start_time
        self.stats['avg_batch_size'] = self.stats['total_evaluations'] / self.stats['total_batches']
        
        # Return based on flag
        if self._return_torch_tensors:
            return policies, values
        else:
            # Only convert to numpy if explicitly requested
            return policies.cpu().numpy(), values.cpu().numpy()
    
    def _process_policies_vectorized(self, 
                                   policy_logits: torch.Tensor,
                                   legal_masks: Optional[Union[np.ndarray, torch.Tensor]]) -> torch.Tensor:
        """Vectorized policy processing entirely on GPU"""
        # Apply legal masks if provided
        if legal_masks is not None:
            if isinstance(legal_masks, np.ndarray):
                legal_masks = torch.from_numpy(legal_masks).to(self.device, non_blocking=True, dtype=torch.bool)
            elif not isinstance(legal_masks, torch.Tensor):
                legal_masks = legal_masks.to(self.device, non_blocking=True, dtype=torch.bool)
            
            # Mask illegal actions with -inf
            policy_logits = torch.where(legal_masks, policy_logits, self.neg_inf)
        
        # Stable softmax (all operations on GPU)
        policies = torch.softmax(policy_logits, dim=-1)
        
        return policies
    
    def _apply_temperature_vectorized(self, policies: torch.Tensor, temperature: float) -> torch.Tensor:
        """Apply temperature to policy distribution (vectorized)"""
        if temperature == 0:
            # Deterministic: choose max probability
            max_indices = policies.argmax(dim=-1)
            one_hot = torch.zeros_like(policies)
            one_hot.scatter_(-1, max_indices.unsqueeze(-1), 1.0)
            return one_hot
        else:
            # Apply temperature using log-space for stability
            log_policies = torch.log(policies + 1e-10)
            log_policies = log_policies / temperature
            return torch.softmax(log_policies, dim=-1)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.stats.copy()
        if stats['total_batches'] > 0:
            stats['avg_time_per_batch'] = stats['total_time'] / stats['total_batches']
            stats['avg_time_per_eval'] = stats['total_time'] / stats['total_evaluations']
            stats['gpu_utilization'] = stats['gpu_time'] / stats['total_time'] if stats['total_time'] > 0 else 0
        return stats
    
    def reset_statistics(self):
        """Reset performance statistics"""
        self.stats = {
            'total_evaluations': 0,
            'total_batches': 0,
            'total_time': 0.0,
            'gpu_time': 0.0,
            'avg_batch_size': 0.0
        }
    
    def warmup(self, warmup_steps: int = 10):
        """Warmup GPU with dummy evaluations"""
        if self.device.type != 'cuda':
            return
            
        logger.info(f"Warming up GPU with {warmup_steps} dummy evaluations...")
        # Use 18 channels as expected by feature extraction
        dummy_states = torch.randn(self.batch_size, 18, 15, 15, device=self.device)
        
        for _ in range(warmup_steps):
            with torch.no_grad():
                if self.use_mixed_precision and not self.use_tensorrt:
                    with torch.amp.autocast('cuda'):
                        self.model(dummy_states)
                else:
                    self.model(dummy_states)
        
        if self.enable_timing and self.device.type == 'cuda':
            torch.cuda.synchronize()
        logger.info("GPU warmup complete")