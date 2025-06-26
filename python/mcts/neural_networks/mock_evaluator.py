"""
Mock evaluator for testing MCTS without a real neural network

This module provides a lightweight evaluator that returns random or fixed
values for testing MCTS functionality without the overhead of a real
neural network.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class MockEvaluator:
    """Mock neural network evaluator for testing"""
    
    def __init__(self, 
                 game_type: str = 'gomoku',
                 device: str = 'cpu',
                 deterministic: bool = False,
                 fixed_value: float = 0.0,
                 policy_temperature: float = 1.0):
        """
        Initialize mock evaluator
        
        Args:
            game_type: Type of game (gomoku, chess, go)
            device: Device to use (cpu, cuda)
            deterministic: If True, return fixed values
            fixed_value: Fixed value to return when deterministic
            policy_temperature: Temperature for policy generation
        """
        self.game_type = game_type
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.fixed_value = fixed_value
        self.policy_temperature = policy_temperature
        
        # Action space sizes
        self.action_spaces = {
            'gomoku': 225,  # 15x15
            'chess': 4096,  # 64x64 (from-to)
            'go': 361,      # 19x19
        }
        
        self.action_space = self.action_spaces.get(game_type, 225)
        
        # Track evaluation count for debugging
        self.eval_count = 0
        
        logger.info(f"MockEvaluator initialized for {game_type} with action space {self.action_space}")
        
        # Create board size for more intelligent policies
        if game_type == 'gomoku':
            self.board_size = 15
        elif game_type == 'go':
            self.board_size = 19
        else:
            self.board_size = 8  # chess
    
    def evaluate(self, states: Union[torch.Tensor, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate game states
        
        Args:
            states: Batch of game states
            
        Returns:
            values: Value estimates for each state
            policies: Policy distributions for each state
        """
        # Convert to tensor if needed
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()
        
        states = states.to(self.device)
        batch_size = states.shape[0]
        
        self.eval_count += batch_size
        
        if self.deterministic:
            # Return fixed values
            values = torch.full((batch_size, 1), self.fixed_value, device=self.device)
            
            # Center-biased policy for deterministic mode (better for gomoku)
            if self.game_type == 'gomoku' and hasattr(self, 'board_size'):
                policies = self._create_center_biased_policy(batch_size)
            else:
                # Uniform policy fallback
                policies = torch.ones(batch_size, self.action_space, device=self.device)
                policies = policies / self.action_space
        else:
            # Random values between -1 and 1
            values = torch.rand(batch_size, 1, device=self.device) * 2 - 1
            
            # Center-biased random policies for better MCTS behavior
            if self.game_type == 'gomoku' and hasattr(self, 'board_size'):
                policies = self._create_center_biased_policy(batch_size, add_noise=True)
            else:
                # Random policies with temperature fallback
                logits = torch.randn(batch_size, self.action_space, device=self.device)
                logits = logits / self.policy_temperature
                policies = torch.softmax(logits, dim=1)
        
        return policies, values
    
    def _create_center_biased_policy(self, batch_size: int, add_noise: bool = False) -> torch.Tensor:
        """Create center-biased policies for better MCTS behavior"""
        if self.game_type == 'gomoku':
            # Create policies biased towards center of board
            center = self.board_size // 2
            policies = torch.zeros(batch_size, self.action_space, device=self.device)
            
            for i in range(self.board_size):
                for j in range(self.board_size):
                    action = i * self.board_size + j
                    # Distance from center (Manhattan distance)
                    dist = abs(i - center) + abs(j - center)
                    # Higher probability for center squares
                    prob = 1.0 / (1.0 + dist * 0.3)
                    policies[:, action] = prob
            
            if add_noise:
                # Add some random noise to make it less deterministic
                noise = torch.randn_like(policies) * 0.1
                policies = policies + noise
                policies = torch.relu(policies)  # Ensure non-negative
            
            # Normalize to valid probability distribution
            policies = policies / (policies.sum(dim=1, keepdim=True) + 1e-8)
            return policies
        else:
            # Fallback for other games
            policies = torch.ones(batch_size, self.action_space, device=self.device)
            return policies / self.action_space
    
    def evaluate_batch(self, states: List[Union[np.ndarray, torch.Tensor]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate a batch of states (numpy interface)
        
        Args:
            states: List of game state arrays or tensors
            
        Returns:
            values: Array of value estimates
            policies: Array of policy distributions
        """
        # Stack states, handling both numpy and tensor inputs
        if isinstance(states[0], torch.Tensor):
            states_tensor = torch.stack([s.float() for s in states])
        else:
            states_tensor = torch.stack([torch.from_numpy(s).float() for s in states])
        
        # Evaluate
        policies, values = self.evaluate(states_tensor)
        
        # Convert back to numpy
        return policies.cpu().numpy(), values.cpu().numpy()
    
    def __call__(self, states: Union[torch.Tensor, np.ndarray, List[np.ndarray]]) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
        """
        Make evaluator callable
        
        Args:
            states: Game states in various formats
            
        Returns:
            values: Value estimates
            policies: Policy distributions
        """
        if isinstance(states, list):
            return self.evaluate_batch(states)
        else:
            return self.evaluate(states)
    
    def reset_stats(self):
        """Reset evaluation statistics"""
        self.eval_count = 0
    
    def get_stats(self) -> dict:
        """Get evaluation statistics"""
        return {
            'eval_count': self.eval_count,
            'game_type': self.game_type,
            'device': str(self.device),
            'deterministic': self.deterministic
        }
    
    def to(self, device: Union[str, torch.device]) -> 'MockEvaluator':
        """
        Move evaluator to device (for compatibility with real models)
        
        Args:
            device: Target device
            
        Returns:
            Self
        """
        self.device = torch.device(device)
        return self
    
    def eval(self) -> 'MockEvaluator':
        """Set to evaluation mode (no-op for mock)"""
        return self
    
    def train(self) -> 'MockEvaluator':
        """Set to training mode (no-op for mock)"""
        return self
    
    def state_dict(self) -> dict:
        """Get state dict (for compatibility)"""
        return {
            'game_type': self.game_type,
            'deterministic': self.deterministic,
            'fixed_value': self.fixed_value,
            'policy_temperature': self.policy_temperature
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load state dict (for compatibility)"""
        self.game_type = state_dict.get('game_type', self.game_type)
        self.deterministic = state_dict.get('deterministic', self.deterministic)
        self.fixed_value = state_dict.get('fixed_value', self.fixed_value)
        self.policy_temperature = state_dict.get('policy_temperature', self.policy_temperature)
    
    def parameters(self):
        """Return empty parameters (for compatibility)"""
        return []
    
    def named_parameters(self):
        """Return empty named parameters (for compatibility)"""
        return []


class DeterministicMockEvaluator(MockEvaluator):
    """Deterministic version for reproducible tests"""
    
    def __init__(self, game_type: str = 'gomoku', device: str = 'cpu'):
        super().__init__(
            game_type=game_type,
            device=device,
            deterministic=True,
            fixed_value=0.0,
            policy_temperature=1.0
        )


class BiasedMockEvaluator(MockEvaluator):
    """Mock evaluator with bias towards certain moves"""
    
    def __init__(self, 
                 game_type: str = 'gomoku',
                 device: str = 'cpu',
                 center_bias: float = 2.0):
        """
        Initialize biased evaluator
        
        Args:
            game_type: Type of game
            device: Device to use
            center_bias: Bias strength towards center moves
        """
        super().__init__(game_type=game_type, device=device)
        self.center_bias = center_bias
        
        # Create center bias mask
        self._create_center_bias()
    
    def _create_center_bias(self):
        """Create bias mask favoring center positions"""
        if self.game_type == 'gomoku':
            size = 15
            center = size // 2
            
            # Create distance from center
            bias = torch.zeros(size, size)
            for i in range(size):
                for j in range(size):
                    dist = max(abs(i - center), abs(j - center))
                    bias[i, j] = 1.0 / (1.0 + dist * 0.5)
            
            self.bias_mask = bias.flatten() * self.center_bias
        else:
            # Uniform for other games
            self.bias_mask = torch.ones(self.action_space)
    
    def evaluate(self, states: Union[torch.Tensor, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate with center bias"""
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()
        
        states = states.to(self.device)
        batch_size = states.shape[0]
        
        # Random values
        values = torch.rand(batch_size, 1, device=self.device) * 2 - 1
        
        # Biased policies
        bias_mask = self.bias_mask.to(self.device)
        logits = torch.randn(batch_size, self.action_space, device=self.device)
        logits = logits + bias_mask.unsqueeze(0)
        policies = torch.softmax(logits, dim=1)
        
        return values, policies


class SequentialMockEvaluator(MockEvaluator):
    """Mock evaluator that returns sequential values for testing"""
    
    def __init__(self, game_type: str = 'gomoku', device: str = 'cpu'):
        super().__init__(game_type=game_type, device=device)
        self.sequence_counter = 0
    
    def evaluate(self, states: Union[torch.Tensor, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return sequential values for testing patterns"""
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()
        
        states = states.to(self.device)
        batch_size = states.shape[0]
        
        # Sequential values
        values = []
        for i in range(batch_size):
            value = (self.sequence_counter % 20 - 10) / 10.0  # -1 to 1
            values.append(value)
            self.sequence_counter += 1
        
        values = torch.tensor(values, device=self.device).unsqueeze(1)
        
        # Uniform policies
        policies = torch.ones(batch_size, self.action_space, device=self.device)
        policies = policies / self.action_space
        
        return values, policies