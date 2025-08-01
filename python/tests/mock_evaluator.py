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
                 device: Union[str, torch.device] = 'cpu',
                 deterministic: bool = False,
                 fixed_value: float = 0.0,
                 policy_temperature: float = 1.0,
                 board_size: Optional[int] = None,
                 batch_size: int = 32,
                 use_amp: bool = False):
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
        self.device = torch.device(device) if isinstance(device, str) else device
        self.deterministic = deterministic
        self.fixed_value = fixed_value
        self.policy_temperature = policy_temperature
        self.batch_size = batch_size
        self.use_amp = use_amp
        self._return_torch_tensors = True
        
        # Handle board size parameter
        if board_size is not None:
            self.board_size = board_size
            # Calculate action space based on board size
            if game_type == 'go':
                self.action_space = board_size * board_size + 1  # +1 for pass
            elif game_type == 'chess':
                self.action_space = 20480  # 64*64*5 (with promotions)
            else:  # gomoku and others
                self.action_space = board_size * board_size
        else:
            # Default sizes
            self.action_spaces = {
                'gomoku': 225,  # 15x15
                'chess': 20480,  # 64x64x5 (from-to with promotions)
                'go': 362,      # 19x19 + pass (default)
            }
            
            self.action_space = self.action_spaces.get(game_type, 225)
            
            # Default board sizes
            if game_type == 'gomoku':
                self.board_size = 15
            elif game_type == 'go':
                self.board_size = 19
            elif game_type == 'chess':
                self.board_size = 8
            else:
                self.board_size = 15  # default
        
        # Track evaluation count for debugging
        self.eval_count = 0
        
        logger.info(f"MockEvaluator initialized for {game_type} with action space {self.action_space}")
    
    def set_board_size(self, board_size: int):
        """Set board size and update action space accordingly"""
        self.board_size = board_size
        if self.game_type == 'go':
            self.action_space = board_size * board_size + 1  # +1 for pass
        elif self.game_type == 'gomoku':
            self.action_space = board_size * board_size
    
    def evaluate(self, states: Union[torch.Tensor, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate game states
        
        Args:
            states: Batch of game states
            
        Returns:
            policies: Policy distributions for each state
            values: Value estimates for each state
        """
        # Convert to tensor if needed
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()
        
        states = states.to(self.device)
        
        # Handle single state without batch dimension
        if states.ndim == 2:  # Single state (height, width)
            states = states.unsqueeze(0)  # Add batch dimension
        
        batch_size = states.shape[0]
        
        self.eval_count += batch_size
        
        if self.deterministic:
            # Return fixed values - shape should be (batch_size,) not (batch_size, 1)
            values = torch.full((batch_size,), self.fixed_value, device=self.device)
            
            # Center-biased policy for deterministic mode (better for gomoku)
            if self.game_type == 'gomoku' and hasattr(self, 'board_size'):
                policies = self._create_center_biased_policy(batch_size)
            else:
                # Uniform policy fallback
                policies = torch.ones(batch_size, self.action_space, device=self.device)
                policies = policies / self.action_space
        else:
            # Random values between -1 and 1 - shape should be (batch_size,) not (batch_size, 1)
            values = torch.rand(batch_size, device=self.device) * 2 - 1
            
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
                    # Euclidean distance from center for smoother gradient
                    dist = ((i - center) ** 2 + (j - center) ** 2) ** 0.5
                    # Use much stronger exponential decay for higher variance
                    # This creates a sharp peak at center
                    prob = np.exp(-dist * 0.5)  # Stronger exponential decay
                    # Add base probability to avoid zeros at edges
                    prob = prob + 0.01
                    policies[:, action] = prob
            
            if add_noise:
                # Add stronger random noise for more variation
                noise = torch.randn_like(policies) * 0.5  # Increased noise
                policies = policies + noise
                policies = torch.relu(policies)  # Ensure non-negative
                
                # Add some random peaks to create more variation
                for b in range(batch_size):
                    # Add 3-5 random peaks
                    num_peaks = torch.randint(3, 6, (1,)).item()
                    for _ in range(num_peaks):
                        peak_pos = torch.randint(0, self.action_space, (1,)).item()
                        policies[b, peak_pos] += torch.rand(1).item() * 2.0
            
            # Normalize to valid probability distribution
            policies = policies / (policies.sum(dim=1, keepdim=True) + 1e-8)
            return policies
        else:
            # Fallback for other games - also add variation
            policies = torch.ones(batch_size, self.action_space, device=self.device)
            if add_noise:
                noise = torch.randn_like(policies) * 0.3
                policies = policies + noise
                policies = torch.relu(policies)
            return policies / (policies.sum(dim=1, keepdim=True) + 1e-8)
    
    def evaluate_batch(self, states_or_game_states, state_indices=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate a batch of states (numpy interface)
        
        Args:
            states_or_game_states: List of game state arrays/tensors OR GameStates object
            state_indices: Optional tensor of state indices (for GameStates interface)
            
        Returns:
            policies: Array of policy distributions
            values: Array of value estimates
        """
        # Handle hybrid mode interface: evaluator.evaluate_batch(game_states, state_indices)
        if state_indices is not None:
            # Extract features from game states using indices
            if hasattr(states_or_game_states, 'get_features_batch'):
                features = states_or_game_states.get_features_batch(state_indices)
                states_tensor = features.float()
            else:
                # Fallback: create mock states based on batch size
                batch_size = len(state_indices) if hasattr(state_indices, '__len__') else state_indices.shape[0]
                states_tensor = torch.randn(batch_size, 3, self.board_size or 15, self.board_size or 15)
        else:
            # Original interface: just states
            states = states_or_game_states
            
            # Handle numpy array directly (from get_nn_features_batch)
            if isinstance(states, np.ndarray):
                states_tensor = torch.from_numpy(states).float()
            elif isinstance(states, torch.Tensor):
                states_tensor = states.float()
            else:
                # Stack states, handling both numpy and tensor inputs
                if isinstance(states[0], torch.Tensor):
                    states_tensor = torch.stack([s.float() for s in states])
                else:
                    states_tensor = torch.stack([torch.from_numpy(s).float() for s in states])
        
        # Evaluate
        policies, values = self.evaluate(states_tensor)
        
        # For hybrid mode (when state_indices is provided), return tensors
        # For normal mode, return numpy arrays unless _return_torch_tensors is set
        if state_indices is not None or getattr(self, '_return_torch_tensors', False):
            return policies, values
        else:
            # Convert back to numpy for original interface
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
        
        # Handle single state without batch dimension
        if states.ndim == 2:  # Single state (height, width)
            states = states.unsqueeze(0)  # Add batch dimension
        
        batch_size = states.shape[0]
        
        # Random values - shape should be (batch_size,) not (batch_size, 1)
        values = torch.rand(batch_size, device=self.device) * 2 - 1
        
        # Biased policies
        bias_mask = self.bias_mask.to(self.device)
        logits = torch.randn(batch_size, self.action_space, device=self.device)
        logits = logits + bias_mask.unsqueeze(0)
        policies = torch.softmax(logits, dim=1)
        
        return policies, values


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
        
        values = torch.tensor(values, device=self.device)  # No unsqueeze - keep shape (batch_size,)
        
        # Uniform policies
        policies = torch.ones(batch_size, self.action_space, device=self.device)
        policies = policies / self.action_space
        
        return policies, values  # Fixed: return (policies, values) not (values, policies)


def create_mock_model(board_size: int = 9, device: str = 'cpu') -> torch.nn.Module:
    """Create a mock PyTorch model for testing
    
    Args:
        board_size: Board size for the game
        device: Device to place model on
        
    Returns:
        A mock torch.nn.Module that can be used as a model
    """
    class MockModel(torch.nn.Module):
        def __init__(self, board_size: int, device: str):
            super().__init__()
            self.board_size = board_size
            self.device = torch.device(device)
            self.action_space = board_size * board_size
            
            # Dummy layers
            self.conv1 = torch.nn.Conv2d(4, 32, 3, padding=1)
            self.fc_value = torch.nn.Linear(32 * board_size * board_size, 1)
            self.fc_policy = torch.nn.Linear(32 * board_size * board_size, self.action_space)
            
        def forward(self, x):
            # Simple forward pass
            x = torch.relu(self.conv1(x))
            x_flat = x.view(x.size(0), -1)
            
            value = torch.tanh(self.fc_value(x_flat))
            policy = torch.softmax(self.fc_policy(x_flat), dim=1)
            
            return policy, value
    
    model = MockModel(board_size, device)
    return model.to(device)