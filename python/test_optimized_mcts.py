#!/usr/bin/env python3
"""Test the optimized MCTS implementation"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import numpy as np
from mcts.core.optimized_mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType
import alphazero_py

# Activate virtual environment
activate_this = '/home/cosmo/venv/bin/activate_this.py'
if os.path.exists(activate_this):
    exec(open(activate_this).read(), {'__file__': activate_this})


class DummyEvaluator:
    """Fast dummy evaluator for testing"""
    def __init__(self, board_size=15):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.board_size = board_size
        self.action_size = board_size * board_size
        
    def evaluate_batch(self, features):
        batch_size = features.shape[0]
        # Random policy and value
        policies = torch.rand(batch_size, self.action_size, device=self.device)
        policies = policies / policies.sum(dim=1, keepdim=True)
        values = torch.rand(batch_size, 1, device=self.device) * 2 - 1  # [-1, 1]
        return policies, values


def test_basic_functionality():
    """Test basic MCTS functionality"""
    print("Testing basic MCTS functionality...")
    
    config = MCTSConfig(
        num_simulations=1000,
        max_wave_size=256,  # Small for quick test
        min_wave_size=256,
        adaptive_wave_sizing=False,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        game_type=GameType.GOMOKU,
        board_size=15,
        enable_debug_logging=True
    )
    
    evaluator = DummyEvaluator(board_size=15)
    mcts = MCTS(config, evaluator)
    
    # Create game state
    state = alphazero_py.GomokuState()
    
    # Run search
    policy = mcts.search(state, 1000)
    
    print(f"Policy shape: {policy.shape}")
    print(f"Policy sum: {policy.sum():.6f}")
    print(f"Max probability: {policy.max():.6f}")
    print(f"Non-zero actions: {(policy > 0).sum()}")
    
    # Get statistics
    stats = mcts.get_statistics()
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    assert policy.shape == (225,), f"Expected shape (225,), got {policy.shape}"
    assert abs(policy.sum() - 1.0) < 1e-5, f"Policy doesn't sum to 1: {policy.sum()}"
    
    print("✓ Basic functionality test passed!")


def test_performance():
    """Test performance with optimized settings"""
    print("\nTesting MCTS performance...")
    
    # Optimal configuration for performance
    config = MCTSConfig(
        num_simulations=10000,
        min_wave_size=3072,
        max_wave_size=3072,
        adaptive_wave_sizing=False,  # Critical!
        device='cuda',
        game_type=GameType.GOMOKU,
        board_size=15,
        memory_pool_size_mb=2048,
        max_tree_nodes=500000,
        use_mixed_precision=True,
        use_cuda_graphs=True,
        use_tensor_cores=True,
        enable_debug_logging=False  # Disable for performance test
    )
    
    evaluator = DummyEvaluator(board_size=15)
    mcts = MCTS(config, evaluator)
    mcts.optimize_for_hardware()
    
    # Create game state
    state = alphazero_py.GomokuState()
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        mcts.search(state, 100)
    
    # Performance test
    num_simulations = 100000
    print(f"\nRunning {num_simulations} simulations...")
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    policy = mcts.search(state, num_simulations)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    sims_per_sec = num_simulations / elapsed
    
    print(f"\nPerformance Results:")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Simulations/second: {sims_per_sec:,.0f}")
    print(f"  Tree nodes: {mcts.get_statistics()['tree_nodes']}")
    print(f"  Memory usage: {mcts.get_statistics()['tree_memory_mb']:.1f} MB")
    
    # Performance check
    if sims_per_sec >= 80000:
        print(f"✓ EXCELLENT! Achieved {sims_per_sec:,.0f} sims/s (target: 80k+)")
    elif sims_per_sec >= 50000:
        print(f"✓ Good performance: {sims_per_sec:,.0f} sims/s")
    else:
        print(f"⚠ Below target: {sims_per_sec:,.0f} sims/s (target: 80k+)")
    
    return sims_per_sec


def test_vectorization():
    """Test that vectorization is working correctly"""
    print("\nTesting vectorization...")
    
    config = MCTSConfig(
        num_simulations=3072,
        min_wave_size=3072,
        max_wave_size=3072,
        adaptive_wave_sizing=False,
        device='cuda',
        game_type=GameType.GOMOKU,
        profile_gpu_kernels=True
    )
    
    evaluator = DummyEvaluator()
    mcts = MCTS(config, evaluator)
    
    state = alphazero_py.GomokuState()
    
    # Run search
    policy = mcts.search(state, 3072)
    
    # Check kernel timings
    stats = mcts.get_statistics()
    if 'kernel_wave_total' in stats:
        wave_time = stats['kernel_wave_total']
        print(f"Wave processing time: {wave_time:.3f}s")
        print(f"Time per simulation: {wave_time / 3072 * 1000:.2f}ms")
    
    print("✓ Vectorization test passed!")


if __name__ == "__main__":
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    try:
        test_basic_functionality()
        test_vectorization()
        sims_per_sec = test_performance()
        
        print("\n" + "="*50)
        print("All tests passed!")
        print(f"Final performance: {sims_per_sec:,.0f} simulations/second")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()