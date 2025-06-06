#!/usr/bin/env python3
"""Minimal MCTS test to isolate hang issue"""

import os
import torch
import time

# Set CUDA arch to avoid warnings  
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'

from mcts.core.high_performance_mcts import HighPerformanceMCTS, HighPerformanceMCTSConfig
from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
from alphazero_py import GomokuState
import numpy as np


class SimpleGameInterface:
    def __init__(self):
        self.board_size = 15
        self.action_space = self.board_size * self.board_size
        
    def get_legal_moves(self, state):
        if hasattr(state, 'get_legal_moves'):
            moves = state.get_legal_moves()
            return moves[:5]  # Limit to 5 moves for speed
        return list(range(5))
    
    def apply_move(self, state, action):
        if hasattr(state, 'apply_move'):
            new_state = state.clone()
            new_state.apply_move(action)
            return new_state
        return state
    
    def state_to_numpy(self, state, use_enhanced=True):
        if hasattr(state, 'to_numpy'):
            return state.to_numpy()
        return np.zeros((20, 15, 15), dtype=np.float32)


def test_minimal():
    """Test with absolutely minimal configuration"""
    
    print("Testing minimal MCTS...")
    
    device = torch.device('cuda')
    game = SimpleGameInterface()
    evaluator = ResNetEvaluator(game_type='gomoku', device=device)
    
    # Absolutely minimal config
    config = HighPerformanceMCTSConfig(
        num_simulations=10,      # Tiny number
        wave_size=4,             # Tiny wave
        device=str(device),
        enable_gpu=True,         # Re-enable GPU (CUDA graphs disabled)
        enable_interference=False,    # Disable all complex features
        enable_path_integral=False,
        enable_transposition_table=False,
        mixed_precision=False
    )
    
    print("Creating MCTS...")
    mcts = HighPerformanceMCTS(config, game, evaluator)
    
    # Test just creating the search - don't even run it yet
    print("Testing tree initialization...")
    mcts._reset_tree()
    print("✅ Tree reset successful")
    
    # Try adding just the root
    root_state = GomokuState()
    root_idx = mcts.tree.add_root(state=root_state)
    print(f"✅ Root added: {root_idx}")
    
    # Try single wave
    print("Testing single wave...")
    start = time.perf_counter()
    try:
        result = mcts.wave_engine.run_wave(root_state, wave_size=1)
        elapsed = time.perf_counter() - start
        print(f"✅ Single wave completed in {elapsed:.2f}s")
        return True
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"❌ Single wave failed after {elapsed:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_minimal()
    print(f"\nResult: {'SUCCESS' if success else 'FAILURE'}")