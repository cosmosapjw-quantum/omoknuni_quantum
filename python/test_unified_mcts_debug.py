#!/usr/bin/env python3
"""Debug version of test script with logging"""

import torch
import time
import numpy as np
import logging
from mcts.core.mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType
import alphazero_py

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Mock evaluator for testing
class MockEvaluator:
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        
    def evaluate_batch(self, features, legal_masks=None):
        """Mock evaluation - returns random values and policies"""
        batch_size = features.shape[0]
        board_size = features.shape[-1]
        action_size = board_size * board_size
        
        # Random values between -1 and 1
        values = torch.rand(batch_size, 1, device=self.device) * 2 - 1
        
        # Random policy logits
        policies = torch.rand(batch_size, action_size, device=self.device)
        
        return policies, values  # Note: policies first, then values


def test_minimal():
    """Minimal test to debug the issue"""
    print("Testing minimal MCTS search...")
    
    # Create config
    config = MCTSConfig(
        num_simulations=10,  # Very small
        c_puct=1.414,
        temperature=1.0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        game_type=GameType.GOMOKU,
        board_size=15,
        wave_size=16  # Very small for debugging
    )
    
    # Create evaluator
    evaluator = MockEvaluator(config.device)
    
    # Create MCTS
    print("Creating MCTS...")
    mcts = MCTS(config, evaluator)
    
    # Create game state
    game_state = alphazero_py.GomokuState()
    
    # Run search
    print("Running search...")
    start = time.time()
    policy = mcts.search(game_state, num_simulations=10)
    elapsed = time.time() - start
    
    print(f"Search completed in {elapsed:.3f}s")
    print(f"Policy shape: {policy.shape}")
    print(f"Policy sum: {policy.sum():.6f}")
    
    return True


if __name__ == "__main__":
    test_minimal()