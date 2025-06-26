#!/usr/bin/env python3
"""
Focused Search Profiler - Only profiles the search() method

This tool isolates the search performance from initialization overhead.
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


def run_focused_search_benchmark():
    """Run MCTS search benchmark with focused profiling"""
    print("🔧 Creating MCTS instance (not profiled)...")
    
    # Create MCTS instance WITHOUT profiling (to exclude initialization)
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
    
    print("🎮 Creating game state (not profiled)...")
    game_state = alphazero_py.GomokuState()
    
    print("🔍 Starting FOCUSED profiling of search() method only...")
    
    # Profile ONLY the search method
    profiler = cProfile.Profile()
    profiler.enable()
    
    # TIME ONLY THE SEARCH
    search_start = time.perf_counter()
    policy = mcts.search(game_state, config.num_simulations)
    search_time = time.perf_counter() - search_start
    
    profiler.disable()
    
    # Calculate performance
    sims_per_second = config.num_simulations / search_time
    
    print("✅ Focused profiling completed!")
    print(f"⏱️  Pure search time: {search_time:.3f}s")
    print(f"🎯 Pure search performance: {sims_per_second:.0f} sims/s")
    
    # Analyze profiling results
    print("\n" + "=" * 60)
    print("FOCUSED SEARCH PERFORMANCE ANALYSIS")
    print("=" * 60)
    print(f"Search time: {search_time:.3f}s")
    print(f"Performance: {sims_per_second:.0f} sims/s")
    print(f"Expected: 50,000+ sims/s")
    print(f"Slowdown factor: {50000 / sims_per_second:.1f}x")
    
    # Print top functions by cumulative time
    print("\n" + "=" * 60)
    print("TOP FUNCTIONS IN SEARCH (BY CUMULATIVE TIME)")
    print("=" * 60)
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(30)
    
    print("\n" + "=" * 60)
    print("TOP FUNCTIONS IN SEARCH (BY SELF TIME)")
    print("=" * 60)
    
    stats.sort_stats('tottime')
    stats.print_stats(30)
    
    print("\n" + "=" * 60)
    print("MCTS SEARCH BOTTLENECKS")
    print("=" * 60)
    
    print("🔍 Selection functions:")
    stats.sort_stats('cumulative')
    stats.print_stats('select')
    
    print("\n🔍 UCB functions:")
    stats.sort_stats('cumulative')
    stats.print_stats('ucb')
    
    print("\n🔍 Expansion functions:")
    stats.sort_stats('cumulative')
    stats.print_stats('expand')
    
    print("\n🔍 Backup functions:")
    stats.sort_stats('cumulative')
    stats.print_stats('backup')
    
    print("\n🔍 Evaluation functions:")
    stats.sort_stats('cumulative')
    stats.print_stats('evaluate')
    
    # Save results
    with open('focused_search_profile.txt', 'w') as f:
        stats_obj = pstats.Stats(profiler, stream=f)
        stats_obj.sort_stats('cumulative')
        stats_obj.print_stats()
    
    print(f"\n📊 Detailed results saved to 'focused_search_profile.txt'")
    
    return search_time, sims_per_second


def main():
    """Main function"""
    try:
        print("=" * 60)
        print("FOCUSED MCTS SEARCH PROFILER")
        print("=" * 60)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Run focused profiling
        search_time, sims_per_second = run_focused_search_benchmark()
        
        print("\n" + "=" * 60)
        print("FOCUSED PROFILING COMPLETED")
        print("=" * 60)
        print(f"Pure search time: {search_time:.3f}s")
        print(f"Pure search performance: {sims_per_second:.0f} sims/s")
        
        if sims_per_second < 1000:
            print("❌ Performance is severely degraded")
        elif sims_per_second < 10000:
            print("⚠️  Performance is below expectations")
        else:
            print("✅ Performance is acceptable")
        
    except Exception as e:
        print(f"❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()