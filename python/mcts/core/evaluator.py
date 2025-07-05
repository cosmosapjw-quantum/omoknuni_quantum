"""Neural network evaluator interface for MCTS

This module provides the interface for neural network evaluation and
a mock implementation for testing without a trained model.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import numpy as np
import time
from collections import OrderedDict

import torch
HAS_TORCH = True  # torch is a core dependency


@dataclass
class EvaluatorConfig:
    """Configuration for neural network evaluator
    
    Attributes:
        batch_size: Maximum batch size for evaluation
        device: Device to run on ('cpu' or 'cuda')
        timeout: Timeout in seconds for evaluation
        enable_caching: Whether to cache evaluations
        cache_size: Maximum cache entries
    """
    batch_size: int = 64
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    timeout: float = 1.0
    enable_caching: bool = False
    cache_size: int = 10000


class Evaluator(ABC):
    """Abstract base class for neural network evaluators
    
    Provides the interface that MCTS expects for position evaluation.
    """
    
    def __init__(self, config: EvaluatorConfig, action_size: int):
        """Initialize evaluator with configuration
        
        Args:
            config: Evaluator configuration
            action_size: Number of possible actions in the game
        """
        self.config = config
        self.action_size = action_size
        self.stats = {
            'evaluations': 0,
            'batch_evaluations': 0,
            'total_time': 0.0,
            'avg_time': 0.0
        }
        
        if config.enable_caching:
            self.cache = OrderedDict()
        else:
            self.cache = None
    
    @abstractmethod  
    def evaluate(
        self, 
        state: np.ndarray,
        legal_mask: Optional[np.ndarray] = None,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """Evaluate a single game state
        
        Args:
            state: Game state as numpy array
            legal_mask: Boolean mask of legal moves
            temperature: Temperature for policy output
            
        Returns:
            Tuple of (policy, value):
                policy: Probability distribution over actions
                value: Value estimate in [-1, 1]
        """
        pass
    
    @abstractmethod
    def evaluate_batch(
        self,
        states: np.ndarray,
        legal_masks: Optional[np.ndarray] = None,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch of game states
        
        Args:
            states: Batch of game states [batch_size, ...]
            legal_masks: Batch of legal move masks [batch_size, action_size]
            temperature: Temperature for policy output
            
        Returns:
            Tuple of (policies, values):
                policies: Batch of probability distributions [batch_size, action_size]
                values: Batch of value estimates [batch_size]
        """
        pass
    
    def get_stats(self) -> Dict:
        """Get evaluation statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset evaluation statistics"""
        self.stats = {
            'evaluations': 0,
            'batch_evaluations': 0,
            'total_time': 0.0,
            'avg_time': 0.0
        }
    
    def _update_stats(self, batch_size: int, eval_time: float):
        """Update evaluation statistics"""
        self.stats['evaluations'] += batch_size
        self.stats['batch_evaluations'] += 1
        self.stats['total_time'] += eval_time
        self.stats['avg_time'] = self.stats['total_time'] / self.stats['evaluations']
    
    def _cache_get(self, state_key: str) -> Optional[Tuple[np.ndarray, float]]:
        """Get cached evaluation"""
        if self.cache is None:
            return None
        return self.cache.get(state_key)
    
    def _cache_put(self, state_key: str, policy: np.ndarray, value: float):
        """Cache evaluation result"""
        if self.cache is None:
            return
        
        if len(self.cache) >= self.config.cache_size:
            # Remove oldest entry
            self.cache.popitem(last=False)
        
        self.cache[state_key] = (policy.copy(), value)
    
    def warmup(self, dummy_state: np.ndarray, num_iterations: int = 10):
        """Warm up the evaluator with dummy evaluations"""
        dummy_batch = np.expand_dims(dummy_state, axis=0)
        
        for _ in range(num_iterations):
            self.evaluate(dummy_state)
            self.evaluate_batch(dummy_batch)



class AlphaZeroEvaluator(Evaluator):
    """Evaluator for AlphaZeroNetwork models that works with MCTS"""
    
    def __init__(self, model, config: Optional[EvaluatorConfig] = None,
                 device: Optional[str] = None, action_size: Optional[int] = None):
        """Initialize AlphaZero evaluator
        
        Args:
            model: AlphaZeroNetwork model (torch.nn.Module)
            config: Evaluator configuration
            device: Device to run on ('cpu' or 'cuda')
            action_size: Number of possible actions
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for AlphaZeroEvaluator")
        
        if config is None:
            config = EvaluatorConfig()
        
        if device is not None:
            config.device = device
        
        # Determine action size from model if not provided
        if action_size is None:
            action_size = self._infer_action_size(model)
        
        super().__init__(config, action_size)
        
        self.model = model
        self.device = torch.device(config.device)
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Pre-allocate tensors for efficiency
        self._preallocate_tensors()
    
    def _infer_action_size(self, model) -> int:
        """Infer action size from model architecture"""
        try:
            # Try to get from model attributes
            if hasattr(model, 'action_size'):
                return model.action_size
            
            # Try to infer from output layers
            for module in model.modules():
                if hasattr(module, 'out_features'):
                    # Assume the largest output layer is the policy head
                    return module.out_features
            
            # Default fallback
            return 225  # 15x15 Gomoku
            
        except Exception:
            return 225  # Safe default
    
    def _preallocate_tensors(self):
        """Pre-allocate tensors for common batch sizes"""
        self._tensor_cache = {}
        
        # Pre-allocate for common batch sizes
        for batch_size in [1, 8, 16, 32, 64]:
            try:
                # Dummy tensor - shape will be updated based on actual input
                dummy = torch.zeros((batch_size, 1), device=self.device)
                self._tensor_cache[batch_size] = dummy
            except Exception:
                # Skip if not enough memory
                break
    
    def evaluate(
        self, 
        state: np.ndarray,
        legal_mask: Optional[np.ndarray] = None,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """Evaluate single state using neural network"""
        start_time = time.time()
        
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            if isinstance(state, np.ndarray):
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            else:
                state_tensor = state.float().unsqueeze(0).to(self.device)
            
            # Forward pass
            policy_logits, value = self.model(state_tensor)
            
            # Apply temperature to policy
            if temperature != 1.0:
                policy_logits = policy_logits / temperature
            
            # Convert to numpy
            policy = torch.softmax(policy_logits, dim=1).cpu().numpy().squeeze(0)
            value = value.cpu().numpy().squeeze(0)
            
            # Apply legal mask if provided
            if legal_mask is not None:
                policy[~legal_mask] = 0
                policy_sum = policy.sum()
                if policy_sum > 0:
                    policy = policy / policy_sum
                else:
                    # Fallback to uniform over legal moves
                    legal_actions = np.where(legal_mask)[0]
                    policy = np.zeros_like(policy)
                    if len(legal_actions) > 0:
                        policy[legal_actions] = 1.0 / len(legal_actions)
        
        eval_time = time.time() - start_time
        self._update_stats(1, eval_time)
        
        return policy, float(value)
    
    def evaluate_batch(
        self,
        states: np.ndarray,
        legal_masks: Optional[np.ndarray] = None,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate batch of states using neural network"""
        start_time = time.time()
        batch_size = states.shape[0]
        
        with torch.no_grad():
            # Convert to tensor
            if isinstance(states, np.ndarray):
                state_tensor = torch.from_numpy(states).float().to(self.device)
            else:
                state_tensor = states.float().to(self.device)
            
            # Forward pass
            policy_logits, values = self.model(state_tensor)
            
            # Apply temperature to policies
            if temperature != 1.0:
                policy_logits = policy_logits / temperature
            
            # Convert to numpy
            policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
            values = values.cpu().numpy().squeeze()
            
            # Ensure values is 1D array even for single batch
            if values.ndim == 0:
                values = np.array([values])
            
            # Apply legal masks if provided
            if legal_masks is not None:
                for i in range(batch_size):
                    policies[i][~legal_masks[i]] = 0
                    policy_sum = policies[i].sum()
                    if policy_sum > 0:
                        policies[i] = policies[i] / policy_sum
                    else:
                        # Fallback to uniform over legal moves
                        legal_actions = np.where(legal_masks[i])[0]
                        policies[i] = np.zeros_like(policies[i])
                        if len(legal_actions) > 0:
                            policies[i][legal_actions] = 1.0 / len(legal_actions)
        
        eval_time = time.time() - start_time
        self._update_stats(batch_size, eval_time)
        
        return policies, values


class RandomEvaluator(Evaluator):
    """Random evaluator for baseline comparisons
    
    Returns random policies and neutral values for all positions.
    Useful for testing and as a baseline opponent.
    """
    
    def __init__(self, action_size: int = 225, config: Optional[EvaluatorConfig] = None):
        """Initialize random evaluator
        
        Args:
            action_size: Number of possible actions
            config: Evaluator configuration
        """
        if config is None:
            config = EvaluatorConfig()
        
        super().__init__(config, action_size)
        
        # Initialize random seed for reproducibility
        self.rng = np.random.RandomState(42)
    
    def evaluate(
        self, 
        state: np.ndarray,
        legal_mask: Optional[np.ndarray] = None,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """Return random policy and neutral value"""
        start_time = time.time()
        
        # Create random policy
        policy = self.rng.random(self.action_size).astype(np.float32)
        
        # Apply legal mask if provided
        if legal_mask is not None:
            policy[~legal_mask] = 0
            policy_sum = policy.sum()
            if policy_sum > 0:
                policy = policy / policy_sum
            else:
                # Fallback to uniform over legal moves
                legal_actions = np.where(legal_mask)[0]
                policy = np.zeros(self.action_size, dtype=np.float32)
                if len(legal_actions) > 0:
                    policy[legal_actions] = 1.0 / len(legal_actions)
        else:
            # Normalize
            policy = policy / policy.sum()
        
        # Random value between -0.1 and 0.1 (slightly biased around 0)
        value = (self.rng.random() - 0.5) * 0.2
        
        eval_time = time.time() - start_time
        self._update_stats(1, eval_time)
        
        return policy, float(value)
    
    def evaluate_batch(
        self,
        states: np.ndarray,
        legal_masks: Optional[np.ndarray] = None,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return random policies and neutral values for batch"""
        start_time = time.time()
        batch_size = states.shape[0]
        
        # Create random policies
        policies = self.rng.random((batch_size, self.action_size)).astype(np.float32)
        
        # Apply legal masks if provided
        if legal_masks is not None:
            for i in range(batch_size):
                policies[i][~legal_masks[i]] = 0
                policy_sum = policies[i].sum()
                if policy_sum > 0:
                    policies[i] = policies[i] / policy_sum
                else:
                    # Fallback to uniform over legal moves
                    legal_actions = np.where(legal_masks[i])[0]
                    policies[i] = np.zeros(self.action_size, dtype=np.float32)
                    if len(legal_actions) > 0:
                        policies[i][legal_actions] = 1.0 / len(legal_actions)
        else:
            # Normalize each policy
            for i in range(batch_size):
                policies[i] = policies[i] / policies[i].sum()
        
        # Random values between -0.1 and 0.1
        values = (self.rng.random(batch_size) - 0.5) * 0.2
        
        eval_time = time.time() - start_time
        self._update_stats(batch_size, eval_time)
        
        return policies, values.astype(np.float32)