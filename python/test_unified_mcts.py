#!/usr/bin/env python3
"""Test script for unified MCTS refactoring

This script tests the performance and correctness of the refactored MCTS implementation.
"""

import torch
import time
import numpy as np
from mcts.core.mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType
import alphazero_py

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


def test_basic_search():
    """Test basic MCTS search functionality"""
    print("Testing basic MCTS search...")
    
    # Create config
    config = MCTSConfig(
        num_simulations=1000,
        c_puct=1.414,
        temperature=1.0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        game_type=GameType.GOMOKU,
        board_size=15,
        wave_size=256  # Small for testing
    )
    
    # Create evaluator
    evaluator = MockEvaluator(config.device)
    
    # Create MCTS
    mcts = MCTS(config, evaluator)
    
    # Create game state
    game_state = alphazero_py.GomokuState()
    
    # Run search
    start = time.time()
    policy = mcts.search(game_state, num_simulations=1000)
    elapsed = time.time() - start
    
    print(f"Search completed in {elapsed:.3f}s")
    print(f"Simulations per second: {1000/elapsed:.0f}")
    print(f"Policy shape: {policy.shape}")
    print(f"Policy sum: {policy.sum():.6f}")
    
    # Verify policy is valid probability distribution
    assert abs(policy.sum() - 1.0) < 1e-5, "Policy should sum to 1"
    assert (policy >= 0).all(), "Policy should be non-negative"
    
    print("✓ Basic search test passed!")
    return True


def test_performance():
    """Test MCTS performance with optimized settings"""
    print("\nTesting MCTS performance...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping performance test")
        return True
        
    # Create optimized config
    config = MCTSConfig(
        num_simulations=50000,
        c_puct=1.414,
        temperature=0.0,  # Deterministic
        device='cuda',
        game_type=GameType.GOMOKU,
        board_size=15,
        wave_size=3072,  # Optimal for RTX 3060 Ti
        enable_virtual_loss=True
    )
    
    # Create evaluator
    evaluator = MockEvaluator('cuda')
    
    # Create MCTS
    mcts = MCTS(config, evaluator)
    mcts.optimize_for_hardware()
    
    # Create game state
    game_state = alphazero_py.GomokuState()
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        mcts.search(game_state, num_simulations=1000)
    
    # Performance test
    print("Running performance test...")
    num_searches = 5
    total_sims = 0
    total_time = 0
    
    for i in range(num_searches):
        start = time.time()
        policy = mcts.search(game_state, num_simulations=10000)
        elapsed = time.time() - start
        
        sims_per_sec = 10000 / elapsed
        print(f"  Search {i+1}: {sims_per_sec:.0f} sims/s")
        
        total_sims += 10000
        total_time += elapsed
        
    avg_sims_per_sec = total_sims / total_time
    print(f"\nAverage performance: {avg_sims_per_sec:.0f} sims/s")
    
    # Get statistics
    stats = mcts.get_statistics()
    print(f"Tree nodes: {stats.get('tree_nodes', 0)}")
    print(f"GPU memory: {stats.get('gpu_memory_mb', 0):.1f} MB")
    
    print("✓ Performance test completed!")
    return True


def test_memory_management():
    """Test memory management and tree growth"""
    print("\nTesting memory management...")
    
    config = MCTSConfig(
        num_simulations=1000,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        game_type=GameType.GOMOKU,
        wave_size=256
    )
    
    evaluator = MockEvaluator(config.device)
    mcts = MCTS(config, evaluator)
    
    game_state = alphazero_py.GomokuState()
    
    # Run multiple searches to test tree growth
    for i in range(10):
        policy = mcts.search(game_state, num_simulations=500)
        
        stats = mcts.get_statistics()
        nodes = stats.get('tree_nodes', 0)
        memory = stats.get('tree_memory_mb', 0)
        
        print(f"  After {(i+1)*500} sims: {nodes} nodes, {memory:.1f} MB")
        
        # Make a move
        action = np.argmax(policy)
        game_state.apply_move(action)
        
    print("✓ Memory management test passed!")
    return True


def test_game_state_integration():
    """Test GPU game state integration"""
    print("\nTesting GPU game state integration...")
    
    config = MCTSConfig(
        num_simulations=100,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        game_type=GameType.GOMOKU,
        wave_size=64  # Small for testing
    )
    
    evaluator = MockEvaluator(config.device)
    mcts = MCTS(config, evaluator)
    
    # Test with different game states
    for game_type in [GameType.GOMOKU]:  # Add CHESS, GO when implemented
        print(f"  Testing {game_type.name}...")
        
        config.game_type = game_type
        mcts = MCTS(config, evaluator)
        
        if game_type == GameType.GOMOKU:
            game_state = alphazero_py.GomokuState()
        # elif game_type == GameType.CHESS:
        #     game_state = alphazero_py.ChessState()
        # elif game_type == GameType.GO:
        #     game_state = alphazero_py.GoState()
        
        policy = mcts.search(game_state, num_simulations=100)
        
        assert policy.shape[0] == config.board_size ** 2
        assert abs(policy.sum() - 1.0) < 1e-5
        
    print("✓ Game state integration test passed!")
    return True


def main():
    """Run all tests"""
    print("=== Testing Unified MCTS Implementation ===\n")
    
    tests = [
        test_basic_search,
        test_performance,
        test_memory_management,
        test_game_state_integration
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            
    print(f"\n=== Test Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n✓ All tests passed! The unified MCTS implementation is working correctly.")
    else:
        print(f"\n✗ {failed} tests failed. Please check the implementation.")
        
    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)