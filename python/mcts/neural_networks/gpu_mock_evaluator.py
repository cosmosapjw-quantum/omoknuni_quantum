"""GPU-optimized mock evaluator for testing MCTS performance"""

import torch
from typing import Tuple


class GPUMockEvaluator:
    """GPU-native mock evaluator that avoids CPU transfers
    
    This evaluator keeps everything on GPU to test true MCTS performance
    without neural network overhead.
    """
    
    def __init__(self, action_space_size: int = 225, device: str = 'cuda'):
        """Initialize GPU mock evaluator
        
        Args:
            action_space_size: Size of action space
            device: Device to use
        """
        self.action_space_size = action_space_size
        self.device = torch.device(device)
        self._return_torch_tensors = True
        
        # Pre-generate some random weights on GPU for consistent policies
        self.policy_weights = torch.randn(
            100, action_space_size, device=self.device
        )
        self.value_weights = torch.randn(100, device=self.device)
        
    def evaluate_batch(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate a batch of positions entirely on GPU
        
        Args:
            features: Batch of input features (can be numpy or tensor)
            
        Returns:
            Tuple of (policies, values) as GPU tensors
        """
        # Ensure features are on GPU
        if not isinstance(features, torch.Tensor):
            features = torch.from_numpy(features).to(self.device, non_blocking=True)
        elif features.device != self.device:
            features = features.to(self.device, non_blocking=True)
            
        batch_size = features.shape[0]
        
        # Generate deterministic indices based on feature sum
        # This ensures same position gets same evaluation
        feature_sums = features.view(batch_size, -1).sum(dim=1)
        indices = (feature_sums * 1000).long() % 100
        
        # Look up pre-generated policies (no CPU operations)
        policies = self.policy_weights[indices]
        
        # Add some center bias
        center = self.action_space_size // 2
        center_bonus = torch.zeros_like(policies)
        if self.action_space_size == 225:  # 15x15 board
            # Add bonus to center area
            center_indices = [
                7*15+7, 7*15+6, 7*15+8, 6*15+7, 8*15+7,
                6*15+6, 6*15+8, 8*15+6, 8*15+8
            ]
            for idx in center_indices:
                if idx < self.action_space_size:
                    center_bonus[:, idx] = 2.0
        
        policies = policies + center_bonus
        
        # Apply softmax to get valid probabilities
        policies = torch.softmax(policies, dim=-1)
        
        # Generate values
        values = torch.tanh(self.value_weights[indices]).unsqueeze(-1)
        
        return policies, values
    
    def evaluate(self, features: torch.Tensor, 
                 legal_mask: torch.Tensor = None,
                 temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate single position
        
        Args:
            features: Input features
            legal_mask: Legal moves mask
            temperature: Temperature for policy
            
        Returns:
            Tuple of (policy, value)
        """
        # Add batch dimension if needed
        if features.dim() == 3:
            features = features.unsqueeze(0)
            
        policies, values = self.evaluate_batch(features)
        
        # Apply legal mask if provided
        if legal_mask is not None:
            if not isinstance(legal_mask, torch.Tensor):
                legal_mask = torch.from_numpy(legal_mask).to(self.device)
            # Mask illegal moves
            policies = policies * legal_mask
            # Renormalize
            policies = policies / (policies.sum(dim=-1, keepdim=True) + 1e-8)
            
        # Apply temperature
        if temperature != 1.0 and temperature > 0:
            policies = torch.pow(policies, 1.0 / temperature)
            policies = policies / (policies.sum(dim=-1, keepdim=True) + 1e-8)
            
        return policies[0], values[0]