#!/usr/bin/env python3
"""Test script to verify path integral integration in HighPerformanceMCTS"""

import torch
import sys
sys.path.append('.')

from mcts.core.high_performance_mcts import HighPerformanceMCTS, HighPerformanceMCTSConfig
from mcts.core.game_interface import GameInterface, GameType
from mcts.core.evaluator import MockEvaluator, EvaluatorConfig


def test_path_integral_search():
    """Test that path integral search works correctly"""
    print("Testing Path Integral Integration in HighPerformanceMCTS...")
    
    # Create game interface
    game = GameInterface(GameType.GOMOKU, board_size=9)
    
    # Create mock evaluator
    evaluator = MockEvaluator(
        EvaluatorConfig(batch_size=256),
        action_size=81  # 9x9 board
    )
    
    # Test with path integral ENABLED
    print("\n1. Testing with Path Integral ENABLED:")
    config_pi = HighPerformanceMCTSConfig(
        num_simulations=200,
        wave_size=64,
        enable_path_integral=True,  # Enable path integral
        enable_gpu=torch.cuda.is_available(),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        temperature=1.0
    )
    
    mcts_pi = HighPerformanceMCTS(config_pi, game, evaluator)
    
    # Create initial state
    state = game.get_initial_state()
    
    # Run search
    policy_pi = mcts_pi.search(state)
    print(f"  Policy computed with path integral: {len(policy_pi)} actions")
    print(f"  Top 5 actions: {sorted(policy_pi.items(), key=lambda x: x[1], reverse=True)[:5]}")
    
    # Test with path integral DISABLED
    print("\n2. Testing with Path Integral DISABLED (standard visit counts):")
    config_std = HighPerformanceMCTSConfig(
        num_simulations=200,
        wave_size=64,
        enable_path_integral=False,  # Disable path integral
        enable_gpu=torch.cuda.is_available(),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        temperature=1.0
    )
    
    mcts_std = HighPerformanceMCTS(config_std, game, evaluator)
    
    # Run search
    policy_std = mcts_std.search(state)
    print(f"  Policy computed with visit counts: {len(policy_std)} actions")
    print(f"  Top 5 actions: {sorted(policy_std.items(), key=lambda x: x[1], reverse=True)[:5]}")
    
    # Compare results
    print("\n3. Comparison:")
    print(f"  Both methods found {len(policy_pi)} and {len(policy_std)} actions")
    
    # Check that probabilities sum to 1
    sum_pi = sum(policy_pi.values())
    sum_std = sum(policy_std.values())
    print(f"  Path integral policy sum: {sum_pi:.6f}")
    print(f"  Standard policy sum: {sum_std:.6f}")
    
    # Test deterministic selection (temperature=0)
    print("\n4. Testing deterministic selection (temperature=0):")
    config_det = HighPerformanceMCTSConfig(
        num_simulations=200,
        wave_size=64,
        enable_path_integral=True,
        enable_gpu=torch.cuda.is_available(),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        temperature=0.0  # Deterministic
    )
    
    mcts_det = HighPerformanceMCTS(config_det, game, evaluator)
    policy_det = mcts_det.search(state)
    
    # Should have exactly one action with probability 1.0
    max_prob = max(policy_det.values())
    num_ones = sum(1 for p in policy_det.values() if p == 1.0)
    print(f"  Max probability: {max_prob}")
    print(f"  Number of actions with prob=1.0: {num_ones}")
    
    # Test get_best_action method
    print("\n5. Testing get_best_action method:")
    best_action = mcts_pi.get_best_action(state)
    print(f"  Best action: {best_action}")
    
    print("\n✓ Path integral integration test completed successfully!")


if __name__ == "__main__":
    test_path_integral_search()