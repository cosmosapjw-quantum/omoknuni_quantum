#!/usr/bin/env python3
"""Simple MCTS test to isolate the hang issue"""

import os
import torch
import time
import signal

# Set CUDA arch to avoid warnings  
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'

from mcts.core.evaluator import Evaluator
from mcts.gpu.csr_tree import CSRTree, CSRTreeConfig
from alphazero_py import GomokuState
import numpy as np


class MockEvaluator(Evaluator):
    """Simple mock evaluator to bypass neural network complexity"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.action_size = 225  # 15x15 board
        
    def evaluate(self, state_array):
        """Return random policy and value"""
        policy = np.random.dirichlet([1.0] * self.action_size)
        value = np.random.randn() * 0.1
        return policy, value
    
    def evaluate_batch(self, state_arrays):
        """Batch evaluation"""
        batch_size = len(state_arrays)
        policies = np.random.dirichlet([1.0] * self.action_size, size=batch_size)
        values = np.random.randn(batch_size) * 0.1
        return policies, values


class SimpleGameInterface:
    """Simple game interface"""
    
    def __init__(self):
        self.board_size = 15
        self.action_space = self.board_size * self.board_size
        
    def get_legal_moves(self, state):
        if hasattr(state, 'get_legal_moves'):
            moves = state.get_legal_moves()
            return moves[:10]  # Limit to 10 moves to speed up
        return list(range(min(10, self.action_space)))
    
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


def timeout_handler(signum, frame):
    print("❌ Simple test timed out!")
    exit(1)


def test_simple_mcts():
    """Test basic tree operations without complex wave engine"""
    
    print("🔧 Testing simple MCTS operations...")
    
    # Set timeout for 10 seconds
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create simple components
        game = SimpleGameInterface()
        evaluator = MockEvaluator(device)
        
        # Create CSR tree
        tree_config = CSRTreeConfig(max_nodes=1000, max_edges=5000, device=str(device))
        tree = CSRTree(tree_config)
        
        print("✓ Created tree")
        
        # Add root
        root_state = GomokuState()
        root_idx = tree.add_root(state=root_state)
        print(f"✓ Added root: {root_idx}")
        
        # Simple expansion test
        legal_moves = game.get_legal_moves(root_state)
        print(f"✓ Got {len(legal_moves)} legal moves")
        
        # Add a few children
        for i, move in enumerate(legal_moves[:3]):  # Just add 3 children
            child_state = game.apply_move(root_state, move)
            child_idx = tree.add_child(root_idx, move, 0.1, child_state)
            print(f"✓ Added child {i}: {child_idx}")
        
        # Ensure consistency
        tree.ensure_consistent()
        print("✓ Tree consistency ensured")
        
        # Test getting children
        children, actions, priors = tree.get_children(root_idx)
        print(f"✓ Root has {len(children)} children")
        
        # Cancel timeout - we succeeded!
        signal.alarm(0)
        
        print("✅ Simple MCTS operations completed successfully!")
        return True
        
    except TimeoutError:
        print("❌ Simple test timed out!")
        return False
    except Exception as e:
        signal.alarm(0)
        print(f"❌ Simple test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_simple_mcts()
    if success:
        print("\n🎉 Basic operations work - issue is in wave engine!")
    else:
        print("\n💥 Basic operations failed!")
    exit(0 if success else 1)