"""Direct GPU Evaluator for single-GPU environment

This module provides a direct GPU evaluator that eliminates multiprocessing
and queue-based communication for optimal single-GPU performance.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Tuple, Optional, Union, Dict, Any
import time

logger = logging.getLogger(__name__)


class DirectGPUEvaluator:
    """Direct GPU evaluator for single-GPU environments
    
    This evaluator directly interfaces with the neural network model
    without any multiprocessing or queue-based communication overhead.
    Optimized for maximum GPU utilization in single-GPU setups.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda',
                 action_size: int = 225,
                 batch_size: int = 512,
                 use_mixed_precision: bool = True):
        """Initialize direct GPU evaluator
        
        Args:
            model: PyTorch neural network model
            device: Device to run on ('cuda' or 'cpu')
            action_size: Size of action space (e.g., 225 for 15x15 Gomoku)
            batch_size: Maximum batch size for GPU processing
            use_mixed_precision: Whether to use mixed precision inference
        """
        self.model = model
        self.device = torch.device(device)
        self.action_size = action_size
        self.batch_size = batch_size
        self.use_mixed_precision = use_mixed_precision
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Mixed precision setup
        self.amp_enabled = use_mixed_precision and device == 'cuda'
        
        # Performance statistics
        self.stats = {
            'total_evaluations': 0,
            'total_batches': 0,
            'total_time': 0.0,
            'avg_batch_size': 0.0
        }
    
    def evaluate(self, 
                state: np.ndarray, 
                legal_mask: Optional[np.ndarray] = None,
                temperature: float = 1.0) -> Tuple[np.ndarray, float]:
        """Evaluate a single state
        
        Args:
            state: Game state array
            legal_mask: Boolean mask of legal actions
            temperature: Temperature for policy sampling
            
        Returns:
            Tuple of (policy, value)
        """
        # Convert single state to batch
        states = np.expand_dims(state, axis=0)
        legal_masks = np.expand_dims(legal_mask, axis=0) if legal_mask is not None else None
        
        # Use batch evaluation
        policies, values = self.evaluate_batch(states, legal_masks)
        
        # Apply temperature if needed
        if temperature != 1.0 and temperature > 0:
            policies[0] = self._apply_temperature(policies[0], temperature)
        
        return policies[0], values[0]
    
    def evaluate_batch(self,
                      states: np.ndarray,
                      legal_masks: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch of states
        
        Args:
            states: Batch of game states
            legal_masks: Batch of legal action masks
            
        Returns:
            Tuple of (policies, values) arrays
        """
        start_time = time.time()
        
        with torch.no_grad():
            # Convert to tensors
            state_tensor = torch.from_numpy(states).to(self.device, dtype=torch.float32)
            
            # Forward pass with mixed precision if enabled
            if self.amp_enabled:
                with torch.amp.autocast('cuda'):
                    policy_logits, values = self.model(state_tensor)
            else:
                policy_logits, values = self.model(state_tensor)
            
            # Convert to numpy
            policy_logits = policy_logits.cpu().numpy()
            values = values.cpu().numpy()
            
            # Handle different shapes of values
            if values.ndim > 1:
                values = values.squeeze()
            
            # Ensure values is 1D for batch processing
            if values.ndim == 0:
                values = np.array([values.item()])
            
            # Apply legal masks and softmax
            policies = self._process_policies(policy_logits, legal_masks)
        
        # Update statistics
        self.stats['total_evaluations'] += len(states)
        self.stats['total_batches'] += 1
        self.stats['total_time'] += time.time() - start_time
        self.stats['avg_batch_size'] = self.stats['total_evaluations'] / self.stats['total_batches']
        
        return policies, values
    
    def _process_policies(self, 
                         policy_logits: np.ndarray,
                         legal_masks: Optional[np.ndarray]) -> np.ndarray:
        """Process policy logits with legal masks and softmax"""
        batch_size = len(policy_logits)
        policies = np.zeros_like(policy_logits)
        
        for i in range(batch_size):
            logits = policy_logits[i]
            
            # Apply legal mask if provided
            if legal_masks is not None:
                legal_mask = legal_masks[i]
                # Set illegal moves to very negative value
                logits = np.where(legal_mask, logits, -1e9)
            
            # Stable softmax
            max_logit = np.max(logits)
            exp_logits = np.exp(logits - max_logit)
            policies[i] = exp_logits / np.sum(exp_logits)
        
        return policies
    
    def _apply_temperature(self, policy: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature to policy distribution"""
        if temperature == 0:
            # Deterministic: choose max probability
            best_action = np.argmax(policy)
            one_hot = np.zeros_like(policy)
            one_hot[best_action] = 1.0
            return one_hot
        else:
            # Apply temperature
            log_policy = np.log(policy + 1e-10)
            log_policy = log_policy / temperature
            
            # Stable softmax
            max_log = np.max(log_policy)
            exp_policy = np.exp(log_policy - max_log)
            return exp_policy / np.sum(exp_policy)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.stats.copy()
        if stats['total_batches'] > 0:
            stats['avg_time_per_batch'] = stats['total_time'] / stats['total_batches']
            stats['avg_time_per_eval'] = stats['total_time'] / stats['total_evaluations']
        return stats
    
    def reset_statistics(self):
        """Reset performance statistics"""
        self.stats = {
            'total_evaluations': 0,
            'total_batches': 0,
            'total_time': 0.0,
            'avg_batch_size': 0.0
        }