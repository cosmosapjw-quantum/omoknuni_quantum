"""Mock evaluator for testing MCTS without a real neural network"""

import torch
import numpy as np
from typing import Tuple

# PROFILING: Import comprehensive profiler
try:
    from ..profiling.gpu_profiler import get_profiler, profile, profile_gpu
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False
    def profile(name, sync=False):
        def decorator(func):
            return func
        return decorator
    def profile_gpu(name):
        def decorator(func):
            return func
        return decorator


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
    
    @profile("MockEvaluator.evaluate_batch")
    def evaluate_batch(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate a batch of positions
        
        Args:
            features: Batch of input features (numpy array or torch tensor)
            
        Returns:
            Tuple of (policies, values)
            - policies: (batch_size, action_space_size)
            - values: (batch_size,)
        """
        # Convert numpy array to tensor if needed
        import numpy as np
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
            if torch.cuda.is_available():
                features = features.cuda()
        
        batch_size = features.shape[0]
        
        # Optimized version: use vectorized operations instead of loops
        # Generate deterministic policies based on feature hashes
        feature_sums = features.view(batch_size, -1).sum(dim=1)
        seeds = (feature_sums * 1000).long() % 2147483647
        
        # Create base policies with some randomness
        # Don't set seed here - allow natural randomness
        base_policies = torch.randn(batch_size, self.action_space_size, device=features.device) + 1.0
        
        # Add center preference using vectorized operations
        # Create distance matrix once
        if not hasattr(self, '_center_weights'):
            weights = torch.zeros(15, 15)
            for i in range(15):
                for j in range(15):
                    weights[i, j] = (14 - abs(i - 7) - abs(j - 7)) * 0.1
            self._center_weights = weights.flatten()
        
        # Ensure center weights are on the same device as features
        if self._center_weights.device != features.device:
            self._center_weights = self._center_weights.to(features.device)
        
        # Add center weights to all policies at once
        policies = base_policies + self._center_weights.unsqueeze(0)
        
        # Add position-specific variation based on seeds - VECTORIZED
        # Instead of looping with .item(), use deterministic hash-based variation
        seed_variation = torch.randn_like(policies) * 0.1
        # Use seeds as indices for deterministic variation without .item()
        seed_factors = (seeds.float() / 1000.0).unsqueeze(1)
        policies = policies + seed_variation * seed_factors
        
        # Vectorized softmax
        policies = torch.softmax(policies, dim=1)
        
        # Generate values with slight variation - VECTORIZED
        # Don't set seed here - allow natural randomness
        base_values = torch.randn(batch_size, device=features.device) * 0.2
        
        # Add seed-based variation without .item()
        value_variation = torch.randn(batch_size, device=features.device) * 0.05
        value_seed_factors = ((seeds + 1000).float() / 10000.0)
        values = base_values + value_variation * value_seed_factors
        
        return policies, values
    
    def to(self, device):
        """Move evaluator to device (no-op for mock)"""
        self.device = device
        return self
    
    def eval(self):
        """Set to evaluation mode (no-op for mock)"""
        return self