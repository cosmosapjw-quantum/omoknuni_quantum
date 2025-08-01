"""Mock evaluator for testing MCTS without a real neural network"""

import torch
import numpy as np
from typing import Tuple


class MockEvaluator:
    """Mock neural network evaluator for testing
    
    Returns random but consistent policies and values for testing
    MCTS implementations without requiring a trained model.
    """
    
    def __init__(self, action_space_size: int = 225):
        """Initialize mock evaluator
        
        Args:
            action_space_size: Size of action space (e.g., 225 for 15x15 board)
        """
        self.action_space_size = action_space_size
        self.device = torch.device('cpu')
    
    def evaluate_batch(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate a batch of positions
        
        Args:
            features: Batch of input features
            
        Returns:
            Tuple of (policies, values)
            - policies: (batch_size, action_space_size)
            - values: (batch_size,)
        """
        batch_size = features.shape[0]
        
        # Generate consistent pseudo-random policies based on feature sum
        # This ensures the same position always gets the same evaluation
        policies = []
        values = []
        
        for i in range(batch_size):
            # Use feature sum as seed for consistency
            seed = int(features[i].sum().item() * 1000) % 2147483647
            rng = np.random.RandomState(seed)
            
            # Generate policy (random but concentrated around center)
            policy = rng.randn(self.action_space_size) + 1.0
            
            # Add some preference for center moves
            center = self.action_space_size // 2
            for j in range(self.action_space_size):
                row = j // 15
                col = j % 15
                dist_to_center = abs(row - 7) + abs(col - 7)
                policy[j] += (14 - dist_to_center) * 0.1
            
            # Softmax normalization
            policy = np.exp(policy - np.max(policy))
            policy = policy / policy.sum()
            
            policies.append(policy)
            
            # Generate value (slightly random around 0)
            value = rng.randn() * 0.2
            values.append(value)
        
        # Convert to tensors
        policies_tensor = torch.tensor(np.array(policies), dtype=torch.float32)
        values_tensor = torch.tensor(np.array(values), dtype=torch.float32)
        
        return policies_tensor, values_tensor
    
    def to(self, device):
        """Move evaluator to device (no-op for mock)"""
        self.device = device
        return self
    
    def eval(self):
        """Set to evaluation mode (no-op for mock)"""
        return self