#!/usr/bin/env python3
"""Basic MCTS test to verify functionality"""

import torch
import numpy as np
import time
from mcts.core.optimized_mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType
import alphazero_py

def test_basic_mcts():
    """Test basic MCTS functionality"""
    print("Basic MCTS Test")
    print("=" * 60)
    
    # Use real game state from alphazero_py
    print("Creating real Gomoku game state...")
    game_state = alphazero_py.GomokuState()
    
    # Create simple evaluator
    class SimpleEvaluator:
        def __init__(self, board_size=15, device='cuda'):
            self.board_size = board_size
            self.device = torch.device(device)
            self.num_actions = board_size * board_size
        
        def evaluate_batch(self, features):
            batch_size = features.shape[0]
            # Return uniform policy and neutral values
            policies = torch.ones((batch_size, self.num_actions), device=self.device) / self.num_actions
            values = torch.zeros((batch_size, 1), device=self.device)
            return policies, values
    
    # Create config with small wave size for debugging
    config = MCTSConfig(
        num_simulations=100,
        min_wave_size=32,  # Small for debugging
        max_wave_size=32,
        adaptive_wave_sizing=False,
        device='cuda',
        game_type=GameType.GOMOKU,
        board_size=15,
        enable_quantum=False,  # Disable quantum for basic test
        enable_debug_logging=True
    )
    
    print(f"\nConfiguration:")
    print(f"  Wave size: {config.max_wave_size}")
    print(f"  Simulations: {config.num_simulations}")
    print(f"  Device: {config.device}")
    
    # Create MCTS
    evaluator = SimpleEvaluator(board_size=15, device='cuda')
    mcts = MCTS(config, evaluator)
    
    # Run search
    print("\nRunning search...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    policy = mcts.search(game_state, num_simulations=100)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    print(f"\nSearch completed:")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Speed: {100/elapsed:.1f} sims/s")
    
    # Check results
    stats = mcts.get_statistics()
    print(f"\nTree statistics:")
    print(f"  Tree nodes: {stats.get('tree_nodes', 0)}")
    print(f"  Tree edges: {stats.get('tree_edges', 0)}")
    print(f"  State pool usage: {stats.get('state_pool_usage', 0):.2%}")
    
    # Check policy
    print(f"\nPolicy statistics:")
    print(f"  Policy shape: {policy.shape}")
    print(f"  Policy sum: {policy.sum():.3f}")
    print(f"  Max probability: {policy.max():.3f}")
    print(f"  Non-zero actions: {(policy > 0).sum()}")
    
    # Check tree structure
    print(f"\nTree structure:")
    print(f"  Root visits: {mcts.tree.visit_counts[0].item()}")
    print(f"  Root value: {mcts.tree.value_sums[0].item():.3f}")
    
    # Get root children
    children, actions, priors = mcts.tree.get_children(0)
    print(f"  Root children: {len(children)}")
    if len(children) > 0:
        print(f"  Child visits: {mcts.tree.visit_counts[children].cpu().numpy()}")
    
    # Test with larger wave size
    print("\n" + "=" * 60)
    print("Testing with optimal wave size...")
    
    config2 = MCTSConfig(
        num_simulations=10000,
        min_wave_size=3072,
        max_wave_size=3072,
        adaptive_wave_sizing=False,
        device='cuda',
        game_type=GameType.GOMOKU,
        board_size=15,
        enable_quantum=False,
        enable_debug_logging=False
    )
    
    mcts2 = MCTS(config2, evaluator)
    mcts2.optimize_for_hardware()
    
    # Warm up
    for _ in range(3):
        mcts2.search(game_state, num_simulations=100)
        mcts2.reset_tree()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    policy2 = mcts2.search(game_state, num_simulations=10000)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    print(f"\nOptimal wave size results:")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Speed: {10000/elapsed:.0f} sims/s")
    
    stats2 = mcts2.get_statistics()
    print(f"  Tree nodes: {stats2.get('tree_nodes', 0)}")
    print(f"  Tree edges: {stats2.get('tree_edges', 0)}")

if __name__ == "__main__":
    test_basic_mcts()