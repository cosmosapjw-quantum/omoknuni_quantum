"""
Random Neural Network Evaluator for MCTS Testing

This module provides a random evaluator that mimics the interface of the real
neural network evaluator but returns random values. This is useful for:
- Testing the physics analysis pipeline without model loading overhead
- Debugging MCTS behavior without neural network dependencies
- Fast iteration during development
"""

import torch
import numpy as np
from typing import Tuple, List, Any
import logging

logger = logging.getLogger(__name__)

class RandomEvaluator:
    """
    Random evaluator that mimics SingleGPUEvaluator interface.
    
    Returns random policy and value predictions for MCTS testing.
    This allows testing the physics analysis pipeline without requiring
    trained neural network models.
    """
    
    def __init__(self, action_space_size: int = 225, device: str = 'cpu'):
        """
        Initialize random evaluator.
        
        Args:
            action_space_size: Size of action space (e.g., 225 for 15x15 board)
            device: Device to use (ignored for random evaluator)
        """
        self.action_space_size = action_space_size
        self.device = device
        self.evaluation_count = 0
        
        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        logger.info(f"Random evaluator initialized with action space size: {action_space_size}")
    
    def evaluate_batch(self, states: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate a batch of states with random values.
        
        Args:
            states: List of game states to evaluate
            
        Returns:
            Tuple of (policy_logits, values)
            - policy_logits: [batch_size, action_space_size] random logits as numpy array
            - values: [batch_size] random values in [-1, 1] as numpy array
        """
        batch_size = len(states)
        self.evaluation_count += batch_size
        
        # Generate random policy logits as numpy arrays
        policy_logits = np.random.randn(batch_size, self.action_space_size).astype(np.float32)
        
        # Generate random values in [-1, 1] as numpy arrays
        values = (np.random.rand(batch_size) * 2 - 1).astype(np.float32)
        
        return policy_logits, values
    
    def evaluate_single(self, state: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate a single state with random values.
        
        Args:
            state: Game state to evaluate
            
        Returns:
            Tuple of (policy_logits, value)
            - policy_logits: [action_space_size] random logits as numpy array
            - value: scalar random value in [-1, 1] as numpy array
        """
        policy_logits, values = self.evaluate_batch([state])
        return policy_logits[0], values[0]
    
    def __call__(self, states: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Make the evaluator callable (matches SingleGPUEvaluator interface)"""
        return self.evaluate_batch(states)
    
    def get_statistics(self) -> dict:
        """Get evaluator statistics"""
        return {
            'evaluation_count': self.evaluation_count,
            'action_space_size': self.action_space_size,
            'evaluator_type': 'random'
        }
    
    def reset_statistics(self):
        """Reset evaluation statistics"""
        self.evaluation_count = 0

class FastRandomEvaluator:
    """
    Even faster random evaluator using fixed patterns.
    
    Uses pre-generated random patterns for maximum speed during testing.
    """
    
    def __init__(self, action_space_size: int = 225, device: str = 'cpu'):
        """
        Initialize fast random evaluator.
        
        Args:
            action_space_size: Size of action space
            device: Device to use
        """
        self.action_space_size = action_space_size
        self.device = device
        self.evaluation_count = 0
        
        # Pre-generate patterns optimized for VRAM usage
        # Adjust based on simulation count to leave room for MCTS tree
        if action_space_size >= 225:  # Large board games
            self.pattern_size = 15000  # ~1GB VRAM, leaving more room for larger MCTS trees
        self.policy_patterns = torch.randn(self.pattern_size, action_space_size, device=self.device)
        self.value_patterns = torch.rand(self.pattern_size, device=self.device) * 2 - 1
        self.pattern_index = 0
        
        vram_usage_gb = (self.pattern_size * action_space_size * 4 * 2) / (1024**3)  # float32 * 2 tensors
        logger.info(f"Fast random evaluator initialized with {self.pattern_size} pre-generated patterns (~{vram_usage_gb:.1f}GB VRAM)")
    
    def evaluate_batch(self, states: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate batch using pre-generated patterns.
        
        Args:
            states: List of game states
            
        Returns:
            Tuple of (policy_logits, values) as numpy arrays
        """
        batch_size = len(states)
        self.evaluation_count += batch_size
        
        # Use pre-generated patterns cyclically
        indices = [(self.pattern_index + i) % self.pattern_size for i in range(batch_size)]
        self.pattern_index = (self.pattern_index + batch_size) % self.pattern_size
        
        policy_logits = self.policy_patterns[indices].cpu().numpy()
        values = self.value_patterns[indices].cpu().numpy()
        
        return policy_logits, values
    
    def evaluate_single(self, state: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate single state"""
        policy_logits, values = self.evaluate_batch([state])
        return policy_logits[0], values[0]
    
    def __call__(self, states: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Make the evaluator callable"""
        return self.evaluate_batch(states)
    
    def get_statistics(self) -> dict:
        """Get evaluator statistics"""
        return {
            'evaluation_count': self.evaluation_count,
            'action_space_size': self.action_space_size,
            'pattern_size': self.pattern_size,
            'evaluator_type': 'fast_random'
        }
    
    def reset_statistics(self):
        """Reset evaluation statistics"""
        self.evaluation_count = 0

def create_random_evaluator(evaluator_type: str = 'random', 
                          action_space_size: int = 225, 
                          device: str = 'cpu') -> Any:
    """
    Factory function to create random evaluators.
    
    Args:
        evaluator_type: Type of random evaluator ('random' or 'fast_random')
        action_space_size: Size of action space
        device: Device to use
        
    Returns:
        Random evaluator instance
    """
    if evaluator_type == 'fast_random':
        return FastRandomEvaluator(action_space_size, device)
    else:
        return RandomEvaluator(action_space_size, device)

# Example usage and testing
if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test random evaluator
    print("Testing RandomEvaluator...")
    evaluator = RandomEvaluator(action_space_size=225, device='cpu')
    
    # Mock states
    mock_states = [{'board': np.zeros((15, 15))} for _ in range(5)]
    
    # Test batch evaluation
    policy_logits, values = evaluator.evaluate_batch(mock_states)
    print(f"Batch evaluation: policy shape {policy_logits.shape}, values shape {values.shape}")
    print(f"Value range: {values.min():.3f} to {values.max():.3f}")
    
    # Test single evaluation
    single_policy, single_value = evaluator.evaluate_single(mock_states[0])
    print(f"Single evaluation: policy shape {single_policy.shape}, value {single_value:.3f}")
    
    # Test statistics
    stats = evaluator.get_statistics()
    print(f"Statistics: {stats}")
    
    # Test fast random evaluator
    print("\nTesting FastRandomEvaluator...")
    fast_evaluator = FastRandomEvaluator(action_space_size=225, device='cpu')
    
    # Speed test
    import time
    start_time = time.time()
    for _ in range(100):
        policy_logits, values = fast_evaluator.evaluate_batch(mock_states)
    end_time = time.time()
    
    print(f"FastRandomEvaluator: 100 batch evaluations in {end_time - start_time:.3f} seconds")
    print(f"Fast evaluator statistics: {fast_evaluator.get_statistics()}")