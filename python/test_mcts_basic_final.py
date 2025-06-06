#!/usr/bin/env python3
"""Basic MCTS test to verify everything works"""

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
    """Simple game interface for testing"""
    
    def __init__(self):
        self.board_size = 15
        self.action_space = self.board_size * self.board_size
        
    def get_legal_moves(self, state):
        """Get legal moves from state"""
        if hasattr(state, 'get_legal_moves'):
            return state.get_legal_moves()
        return list(range(self.action_space))
    
    def apply_move(self, state, action):
        """Apply move to state"""
        if hasattr(state, 'apply_move'):
            new_state = state.clone()
            new_state.apply_move(action)
            return new_state
        return state
    
    def state_to_numpy(self, state, use_enhanced=True):
        """Convert state to numpy array"""
        if hasattr(state, 'to_numpy'):
            return state.to_numpy()
        return np.zeros((20, self.board_size, self.board_size), dtype=np.float32)


def test_basic_mcts():
    """Test basic MCTS functionality"""
    
    print("=" * 50)
    print("BASIC MCTS TEST")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create components
    game_interface = SimpleGameInterface()
    evaluator = ResNetEvaluator(game_type='gomoku', device=device)
    
    config = HighPerformanceMCTSConfig(
        num_simulations=200,
        wave_size=64,
        device=str(device),
        enable_gpu=True,
        enable_interference=False,  # Disable for simpler test
        enable_path_integral=False
    )
    
    print("Creating MCTS...")
    mcts = HighPerformanceMCTS(config, game_interface, evaluator)
    
    # Test search
    root_state = GomokuState()
    print("Running search...")
    
    start = time.perf_counter()
    policy = mcts.search(root_state)
    elapsed = time.perf_counter() - start
    
    print(f"✓ Search completed in {elapsed*1000:.1f} ms")
    print(f"✓ Policy has {len(policy)} moves")
    print(f"✓ Simulations/second: {config.num_simulations/elapsed:.0f}")
    
    # Test stats
    stats = mcts.get_search_statistics()
    print(f"✓ Tree size: {stats['tree_size']} nodes")
    
    if device.type == 'cuda':
        print(f"✓ GPU memory used: {stats.get('gpu_memory_mb', 0):.1f} MB")
    
    print("\n✅ Basic MCTS test PASSED!")
    return True


if __name__ == "__main__":
    success = test_basic_mcts()
    exit(0 if success else 1)