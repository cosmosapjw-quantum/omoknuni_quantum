#!/usr/bin/env python3
"""Simple MCTS test to verify it's working"""

import torch
import numpy as np
import time
from mcts.core.optimized_mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType

# Simple dummy evaluator
class DummyEvaluator:
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

# Dummy state
class DummyState:
    def __init__(self):
        self.board = np.zeros((15, 15), dtype=np.int8)
        self.current_player = 1
        self.move_count = 0
    def to_tensor(self):
        return torch.tensor(self.board, dtype=torch.int8)

def test_simple():
    print("Simple MCTS Test")
    print("=" * 60)
    
    # Create config
    config = MCTSConfig(
        num_simulations=100,
        min_wave_size=32,
        max_wave_size=32,
        adaptive_wave_sizing=False,
        device='cuda',
        game_type=GameType.GOMOKU,
        board_size=15,
        enable_quantum=False,
        enable_debug_logging=False
    )
    
    # Create MCTS
    evaluator = DummyEvaluator(board_size=15, device='cuda')
    mcts = MCTS(config, evaluator)
    
    # Create state
    state = DummyState()
    
    # Run search
    print("\nRunning search...")
    start = time.perf_counter()
    
    try:
        policy = mcts.search(state, num_simulations=100)
        elapsed = time.perf_counter() - start
        
        print(f"Success! Time: {elapsed:.3f}s")
        print(f"Speed: {100/elapsed:.0f} sims/s")
        
        # Check tree
        stats = mcts.get_statistics()
        print(f"\nTree stats:")
        print(f"  Nodes: {stats.get('tree_nodes', 0)}")
        print(f"  Edges: {stats.get('tree_edges', 0)}")
        
        # Check if tree is growing
        if stats.get('tree_nodes', 0) > 10:
            print("✓ Tree is growing properly!")
        else:
            print("✗ Tree is not growing (stuck at root)")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple()