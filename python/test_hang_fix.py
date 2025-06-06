#!/usr/bin/env python3
"""Test to verify hang issue is fixed"""

import os
import torch
import time
import signal

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


def timeout_handler(signum, frame):
    print("❌ Test timed out - hang issue still exists!")
    exit(1)


def test_no_hang():
    """Test that MCTS doesn't hang"""
    
    print("🔧 Testing hang fix...")
    
    # Set timeout for 20 seconds
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(20)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create components
        game_interface = SimpleGameInterface()
        evaluator = ResNetEvaluator(game_type='gomoku', device=device)
        
        # Use smaller configuration for faster test
        config = HighPerformanceMCTSConfig(
            num_simulations=50,  # Small number
            wave_size=16,        # Small wave
            device=str(device),
            enable_gpu=True,
            enable_interference=False,  # Disable complex features
            enable_path_integral=False
        )
        
        print(f"Creating MCTS on {device}...")
        mcts = HighPerformanceMCTS(config, game_interface, evaluator)
        
        # Test single search
        root_state = GomokuState()
        print("Running single search...")
        
        start = time.perf_counter()
        policy = mcts.search(root_state)
        elapsed = time.perf_counter() - start
        
        # Cancel timeout - we succeeded!
        signal.alarm(0)
        
        print(f"✅ Search completed in {elapsed:.2f}s")
        print(f"✅ Policy has {len(policy)} moves")
        print(f"✅ No hang detected!")
        
        return True
        
    except TimeoutError:
        print("❌ Test timed out!")
        return False
    except Exception as e:
        signal.alarm(0)
        print(f"❌ Test failed with error: {e}")
        return False


if __name__ == "__main__":
    success = test_no_hang()
    if success:
        print("\n🎉 Hang issue fixed!")
    else:
        print("\n💥 Hang issue still exists!")
    exit(0 if success else 1)