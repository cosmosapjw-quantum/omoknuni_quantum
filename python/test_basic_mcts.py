#!/usr/bin/env python3
"""Basic functionality test for unified MCTS"""

import torch
import numpy as np
from mcts.core.mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType
import alphazero_py

# Simple test evaluator
class TestEvaluator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def evaluate_batch(self, features, legal_masks=None):
        batch_size = features.shape[0]
        action_size = 225  # 15x15 for Gomoku
        
        # Return uniform policy and zero values
        values = torch.zeros(batch_size, 1, device=self.device)
        policies = torch.ones(batch_size, action_size, device=self.device) / action_size
        
        return policies, values


def main():
    print("Testing basic MCTS functionality...")
    
    # Simple config
    config = MCTSConfig(
        num_simulations=100,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        game_type=GameType.GOMOKU,
        wave_size=32
    )
    
    # Create components
    evaluator = TestEvaluator()
    mcts = MCTS(config, evaluator)
    
    # Test 1: Basic search
    print("\nTest 1: Basic search")
    game_state = alphazero_py.GomokuState()
    policy = mcts.search(game_state, num_simulations=100)
    
    print(f"Policy shape: {policy.shape}")
    print(f"Policy sum: {policy.sum():.6f}")
    print(f"Non-zero actions: {(policy > 0).sum()}")
    
    assert policy.shape == (225,), f"Expected shape (225,), got {policy.shape}"
    assert abs(policy.sum() - 1.0) < 1e-5, f"Policy should sum to 1, got {policy.sum()}"
    print("✓ Basic search passed")
    
    # Test 2: Make a move and search again
    print("\nTest 2: Search after move")
    best_action = np.argmax(policy)
    game_state.make_move(best_action)
    
    policy2 = mcts.search(game_state, num_simulations=50)
    print(f"Policy after move shape: {policy2.shape}")
    print(f"Policy after move sum: {policy2.sum():.6f}")
    print("✓ Search after move passed")
    
    # Test 3: Get statistics
    print("\nTest 3: Statistics")
    stats = mcts.get_statistics()
    print(f"Total simulations: {stats['total_simulations']}")
    print(f"Avg sims/second: {stats['avg_sims_per_second']:.0f}")
    print(f"Tree nodes: {stats.get('tree_nodes', 'N/A')}")
    print("✓ Statistics passed")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    main()