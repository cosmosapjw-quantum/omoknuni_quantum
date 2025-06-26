#!/usr/bin/env python3
"""
Minimal MCTS Test - Test with minimal configuration to isolate performance issues
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path

# Add MCTS modules to path
mcts_root = Path(__file__).parent
sys.path.insert(0, str(mcts_root))

from mcts.core.mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType as GPUGameType
from mcts.neural_networks.mock_evaluator import MockEvaluator
import alphazero_py


def test_minimal_config():
    """Test with absolutely minimal config to isolate issues"""
    
    print("Testing with minimal configuration...")
    
    # Ultra-minimal config
    config = MCTSConfig(
        num_simulations=100,  # Much smaller to isolate
        wave_size=32,         # Much smaller wave
        min_wave_size=32,
        max_wave_size=32,
        device='cuda',
        game_type=GPUGameType.GOMOKU,
        board_size=15,
        enable_quantum=False,
        max_tree_nodes=1000,  # Much smaller tree
        use_mixed_precision=False,
        use_cuda_graphs=False,  # Disable CUDA graphs
        adaptive_wave_sizing=False,
        enable_debug_logging=False,
        profile_gpu_kernels=False,
    )
    
    evaluator = MockEvaluator(game_type='gomoku', device='cuda', deterministic=True)
    mcts = MCTS(config, evaluator)
    
    game_state = alphazero_py.GomokuState()
    
    # Time the search
    print("Running minimal search...")
    start_time = time.perf_counter()
    
    policy = mcts.search(game_state, 100)
    
    search_time = time.perf_counter() - start_time
    sims_per_sec = 100 / search_time
    
    print(f"Minimal config: {search_time:.3f}s, {sims_per_sec:.0f} sims/s")
    
    return search_time, sims_per_sec


def test_wave_sizes():
    """Test different wave sizes to see impact"""
    
    print("\nTesting different wave sizes...")
    
    wave_sizes = [16, 32, 64, 128, 256, 512]
    
    for wave_size in wave_sizes:
        config = MCTSConfig(
            num_simulations=100,
            wave_size=wave_size,
            min_wave_size=wave_size,
            max_wave_size=wave_size,
            device='cuda',
            game_type=GPUGameType.GOMOKU,
            board_size=15,
            enable_quantum=False,
            max_tree_nodes=1000,
            use_mixed_precision=False,
            use_cuda_graphs=False,
            adaptive_wave_sizing=False,
            enable_debug_logging=False,
            profile_gpu_kernels=False,
        )
        
        evaluator = MockEvaluator(game_type='gomoku', device='cuda', deterministic=True)
        mcts = MCTS(config, evaluator)
        game_state = alphazero_py.GomokuState()
        
        start_time = time.perf_counter()
        policy = mcts.search(game_state, 100)
        search_time = time.perf_counter() - start_time
        sims_per_sec = 100 / search_time
        
        print(f"Wave size {wave_size:3d}: {search_time:.3f}s, {sims_per_sec:.0f} sims/s")


def test_simulation_counts():
    """Test different simulation counts to see scaling"""
    
    print("\nTesting different simulation counts...")
    
    sim_counts = [10, 50, 100, 500, 1000]
    
    for sim_count in sim_counts:
        config = MCTSConfig(
            num_simulations=sim_count,
            wave_size=32,
            min_wave_size=32,
            max_wave_size=32,
            device='cuda',
            game_type=GPUGameType.GOMOKU,
            board_size=15,
            enable_quantum=False,
            max_tree_nodes=5000,
            use_mixed_precision=False,
            use_cuda_graphs=False,
            adaptive_wave_sizing=False,
            enable_debug_logging=False,
            profile_gpu_kernels=False,
        )
        
        evaluator = MockEvaluator(game_type='gomoku', device='cuda', deterministic=True)
        mcts = MCTS(config, evaluator)
        game_state = alphazero_py.GomokuState()
        
        start_time = time.perf_counter()
        policy = mcts.search(game_state, sim_count)
        search_time = time.perf_counter() - start_time
        sims_per_sec = sim_count / search_time
        
        print(f"{sim_count:4d} sims: {search_time:.3f}s, {sims_per_sec:.0f} sims/s")


def main():
    """Main testing function"""
    print("=" * 60)
    print("MINIMAL MCTS PERFORMANCE TEST")
    print("=" * 60)
    
    try:
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Test minimal config
        test_minimal_config()
        
        # Test wave sizes
        test_wave_sizes()
        
        # Test simulation scaling
        test_simulation_counts()
        
        print("\n" + "=" * 60)
        print("If all tests are slow, the issue is fundamental.")
        print("If small configs are fast, the issue is with scaling.")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()