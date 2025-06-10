"""Lightweight neural network evaluator for CPU workers

This module provides a fast, lightweight evaluator optimized for CPU execution
in hybrid mode. It trades accuracy for speed to maximize CPU throughput.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class LightweightNet(nn.Module):
    """Lightweight neural network for CPU evaluation
    
    Uses minimal layers and operations optimized for CPU.
    """
    
    def __init__(self, input_channels: int = 20, board_size: int = 15,
                 hidden_size: int = 128, num_actions: int = 225):
        super().__init__()
        
        self.input_channels = input_channels
        self.board_size = board_size
        self.num_actions = num_actions
        
        # Simple convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Global pooling to reduce computation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Small fully connected layers
        self.fc1 = nn.Linear(128, hidden_size)
        
        # Heads
        self.policy_head = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass optimized for CPU
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Tuple of (policy_logits, value)
        """
        # Convolutional layers with ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Shared layer
        x = F.relu(self.fc1(x))
        
        # Output heads
        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        
        return policy, value


class LightweightEvaluator:
    """Fast evaluator for CPU workers in hybrid mode"""
    
    def __init__(self, model_path: Optional[str] = None,
                 input_channels: int = 20, board_size: int = 15,
                 device: str = 'cpu'):
        """Initialize lightweight evaluator
        
        Args:
            model_path: Path to pre-trained model (optional)
            input_channels: Number of input channels
            board_size: Board size
            device: Device to run on (should be 'cpu' for CPU workers)
        """
        self.device = torch.device(device)
        self.input_channels = input_channels
        self.board_size = board_size
        
        # Create model
        self.model = LightweightNet(
            input_channels=input_channels,
            board_size=board_size,
            num_actions=board_size * board_size
        ).to(self.device)
        
        # Load weights if provided
        if model_path:
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded lightweight model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}. Using random initialization.")
        else:
            logger.debug("Using randomly initialized lightweight model")
        
        # Set to evaluation mode
        self.model.eval()
        
        # Disable gradients for inference
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Cache for repeated positions
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def evaluate(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Evaluate a single state
        
        Args:
            state: State array [channels, height, width]
            
        Returns:
            Tuple of (policy, value)
        """
        # Check cache
        state_key = state.tobytes()
        if state_key in self.cache:
            self.cache_hits += 1
            return self.cache[state_key]
        
        self.cache_misses += 1
        
        # Convert to tensor
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)
        
        # Convert to numpy
        policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        value = value.item()
        
        # Cache result
        if len(self.cache) < 10000:  # Limit cache size
            self.cache[state_key] = (policy, value)
        
        return policy, value
    
    def evaluate_batch(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch of states
        
        Args:
            states: Batch of states [batch_size, channels, height, width]
            
        Returns:
            Tuple of (policies, values)
        """
        batch_size = states.shape[0]
        
        # Convert to tensor
        states_tensor = torch.from_numpy(states).float().to(self.device)
        
        # Forward pass
        with torch.no_grad():
            policy_logits, values = self.model(states_tensor)
        
        # Convert to numpy
        policies = F.softmax(policy_logits, dim=1).cpu().numpy()
        values = values.squeeze(1).cpu().numpy()
        
        return policies, values
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate
        }
    
    def clear_cache(self):
        """Clear the position cache"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


class RandomEvaluator:
    """Ultra-fast random evaluator for CPU testing
    
    This can be used for pure exploration without neural network overhead.
    """
    
    def __init__(self, board_size: int = 15, exploration_bias: float = 0.1):
        self.board_size = board_size
        self.num_actions = board_size * board_size
        self.exploration_bias = exploration_bias
        
    def evaluate(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Random evaluation with slight center bias"""
        # Create policy with center bias
        policy = np.ones(self.num_actions) / self.num_actions
        
        # Add slight center bias for more realistic play
        center = self.board_size // 2
        for i in range(self.board_size):
            for j in range(self.board_size):
                dist = abs(i - center) + abs(j - center)
                idx = i * self.board_size + j
                policy[idx] += self.exploration_bias * np.exp(-dist / 5)
        
        # Normalize
        policy = policy / policy.sum()
        
        # Random value
        value = np.random.randn() * 0.1
        
        return policy, value
    
    def evaluate_batch(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Batch random evaluation"""
        batch_size = states.shape[0]
        
        policies = []
        values = []
        
        for _ in range(batch_size):
            policy, value = self.evaluate(None)  # State not used
            policies.append(policy)
            values.append(value)
        
        return np.array(policies), np.array(values)


def create_cpu_evaluator(evaluator_type: str = 'lightweight',
                        model_path: Optional[str] = None,
                        **kwargs) -> Any:
    """Factory function to create CPU evaluator
    
    Args:
        evaluator_type: Type of evaluator ('lightweight', 'random')
        model_path: Path to model weights (for lightweight)
        **kwargs: Additional arguments for evaluator
        
    Returns:
        Evaluator instance
    """
    if evaluator_type == 'lightweight':
        return LightweightEvaluator(model_path=model_path, **kwargs)
    elif evaluator_type == 'random':
        return RandomEvaluator(**kwargs)
    else:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")