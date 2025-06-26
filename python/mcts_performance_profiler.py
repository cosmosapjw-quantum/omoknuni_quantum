#!/usr/bin/env python3
"""
Dedicated MCTS Performance Profiler

This tool uses cProfile to identify the exact bottleneck causing the 
performance regression from 50,000+ sims/s to 178 sims/s.
"""

import sys
import os
import cProfile
import pstats
import io
import time
import torch
import numpy as np
from pathlib import Path

# Add MCTS modules to path
mcts_root = Path(__file__).parent
sys.path.insert(0, str(mcts_root))

from mcts.core.mcts import MCTS, MCTSConfig
from mcts.core.game_interface import GameInterface, GameType
from mcts.gpu.gpu_game_states import GameType as GPUGameType
from mcts.neural_networks.mock_evaluator import MockEvaluator
import alphazero_py


def create_test_mcts():
    """Create MCTS instance for testing"""
    # Use the same configuration as the benchmark
    config = MCTSConfig(
        num_simulations=1000,
        wave_size=3072,
        min_wave_size=3072,
        max_wave_size=3072,
        device='cuda',
        game_type=GPUGameType.GOMOKU,
        board_size=15,
        enable_quantum=False,
        max_tree_nodes=50000,
        use_mixed_precision=False,
    )
    
    evaluator = MockEvaluator(game_type='gomoku', device='cuda')
    mcts = MCTS(config, evaluator)
    mcts.optimize_for_hardware()
    
    return mcts, config


def run_mcts_search_test():
    """Run a single MCTS search for profiling"""
    print("🔧 Creating MCTS instance...")
    mcts, config = create_test_mcts()
    
    print("🎮 Creating game state...")
    game_state = alphazero_py.GomokuState()
    
    print("🚀 Starting MCTS search...")
    start_time = time.perf_counter()
    
    # Run the search that's causing the slowdown
    policy = mcts.search(game_state, config.num_simulations)
    
    end_time = time.perf_counter()
    search_time = end_time - start_time
    sims_per_second = config.num_simulations / search_time
    
    print(f"⏱️  Search completed: {search_time:.3f}s")
    print(f"🎯 Performance: {sims_per_second:.0f} sims/s")
    
    return policy, search_time, sims_per_second


def profile_mcts_performance():
    """Profile MCTS performance with cProfile"""
    print("=" * 60)
    print("MCTS PERFORMANCE PROFILER")
    print("=" * 60)
    
    # Create profiler
    profiler = cProfile.Profile()
    
    print("🔍 Starting profiling...")
    profiler.enable()
    
    # Run the test
    policy, search_time, sims_per_second = run_mcts_search_test()
    
    profiler.disable()
    print("✅ Profiling completed!")
    
    # Analyze results
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS")
    print("=" * 60)
    print(f"Search time: {search_time:.3f}s")
    print(f"Performance: {sims_per_second:.0f} sims/s")
    print(f"Expected: 50,000+ sims/s")
    print(f"Slowdown factor: {50000 / sims_per_second:.1f}x")
    
    # Get profiling stats
    stats_buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=stats_buffer)
    
    print("\n" + "=" * 60)
    print("TOP FUNCTIONS BY CUMULATIVE TIME")
    print("=" * 60)
    stats.sort_stats('cumulative')
    stats.print_stats(30)
    
    print("\n" + "=" * 60)
    print("TOP FUNCTIONS BY SELF TIME")
    print("=" * 60)
    stats.sort_stats('tottime')
    stats.print_stats(30)
    
    print("\n" + "=" * 60)
    print("FUNCTIONS WITH MOST CALLS")
    print("=" * 60)
    stats.sort_stats('ncalls')
    stats.print_stats(20)
    
    # Look for specific bottlenecks
    print("\n" + "=" * 60)
    print("MCTS-SPECIFIC BOTTLENECKS")
    print("=" * 60)
    
    # Filter for MCTS-related functions
    print("🔍 MCTS-related functions:")
    stats.sort_stats('cumulative')
    stats.print_stats('mcts')
    
    print("\n🔍 CSR-related functions:")
    stats.sort_stats('cumulative')
    stats.print_stats('csr')
    
    print("\n🔍 Selection-related functions:")
    stats.sort_stats('cumulative')
    stats.print_stats('select')
    
    print("\n🔍 GPU-related functions:")
    stats.sort_stats('cumulative')
    stats.print_stats('gpu')
    
    print("\n🔍 Kernel-related functions:")
    stats.sort_stats('cumulative')
    stats.print_stats('kernel')
    
    print("\n🔍 Torch-related functions:")
    stats.sort_stats('cumulative')
    stats.print_stats('torch')
    
    # Save detailed stats to file
    with open('mcts_profile_detailed.txt', 'w') as f:
        stats_obj = pstats.Stats(profiler, stream=f)
        stats_obj.sort_stats('cumulative').print_stats()
    
    print("📊 Detailed profiling results saved to 'mcts_profile_detailed.txt'")
    
    return stats


def analyze_specific_functions(stats):
    """Analyze specific functions that might be causing the slowdown"""
    print("\n" + "=" * 60)
    print("DETAILED FUNCTION ANALYSIS")
    print("=" * 60)
    
    # Functions to check specifically
    suspect_functions = [
        'search',
        'select_action', 
        'expand_node',
        'evaluate_batch',
        'backup_values',
        'ensure_consistent',
        'get_children',
        'batch_ucb_selection',
        '_evaluate_batch_vectorized',
        'synchronize',
        'empty_cache'
    ]
    
    for func_name in suspect_functions:
        print(f"\n🔍 Analyzing functions containing '{func_name}':")
        try:
            stats.sort_stats('cumulative')
            stats.print_stats(func_name)
        except Exception as e:
            print(f"  Error analyzing '{func_name}': {e}")


def main():
    """Main profiling function"""
    try:
        # Clear GPU cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Run profiling
        stats = profile_mcts_performance()
        
        # Additional analysis
        analyze_specific_functions(stats)
        
        print("\n" + "=" * 60)
        print("PROFILING COMPLETED")
        print("=" * 60)
        print("Check 'mcts_profile_detailed.txt' for complete results")
        
    except Exception as e:
        print(f"❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()