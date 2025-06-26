#!/usr/bin/env python3
"""
Debug Timing Profiler - Times each component separately

This tool measures exactly where the 5+ seconds are being spent.
"""

import sys
import os
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


def run_detailed_timing_analysis():
    """Time each component of MCTS execution separately"""
    
    print("=" * 60)
    print("DETAILED MCTS TIMING ANALYSIS")
    print("=" * 60)
    
    total_start = time.perf_counter()
    
    print("🔧 Step 1: Creating config...")
    config_start = time.perf_counter()
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
    config_time = time.perf_counter() - config_start
    print(f"   Config creation: {config_time:.3f}s")
    
    print("🔧 Step 2: Creating evaluator...")
    evaluator_start = time.perf_counter()
    evaluator = MockEvaluator(game_type='gomoku', device='cuda')
    evaluator_time = time.perf_counter() - evaluator_start
    print(f"   Evaluator creation: {evaluator_time:.3f}s")
    
    print("🔧 Step 3: Creating MCTS...")
    mcts_start = time.perf_counter()
    mcts = MCTS(config, evaluator)
    mcts_time = time.perf_counter() - mcts_start
    print(f"   MCTS creation: {mcts_time:.3f}s")
    
    print("🔧 Step 4: Hardware optimization...")
    opt_start = time.perf_counter()
    mcts.optimize_for_hardware()
    opt_time = time.perf_counter() - opt_start
    print(f"   Hardware optimization: {opt_time:.3f}s")
    
    print("🎮 Step 5: Creating game state...")
    game_start = time.perf_counter()
    game_state = alphazero_py.GomokuState()
    game_time = time.perf_counter() - game_start
    print(f"   Game state creation: {game_time:.3f}s")
    
    # Clear any CUDA caches
    print("🧹 Step 6: Clearing GPU cache...")
    cache_start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    cache_time = time.perf_counter() - cache_start
    print(f"   GPU cache clear: {cache_time:.3f}s")
    
    # Time just the search method
    print("🚀 Step 7: Running MCTS search...")
    search_start = time.perf_counter()
    policy = mcts.search(game_state, config.num_simulations)
    search_time = time.perf_counter() - search_start
    print(f"   Pure search time: {search_time:.3f}s")
    
    total_time = time.perf_counter() - total_start
    
    print("\n" + "=" * 60)
    print("TIMING BREAKDOWN")
    print("=" * 60)
    print(f"Config creation:       {config_time:.3f}s ({100*config_time/total_time:.1f}%)")
    print(f"Evaluator creation:    {evaluator_time:.3f}s ({100*evaluator_time/total_time:.1f}%)")
    print(f"MCTS creation:         {mcts_time:.3f}s ({100*mcts_time/total_time:.1f}%)")
    print(f"Hardware optimization: {opt_time:.3f}s ({100*opt_time/total_time:.1f}%)")
    print(f"Game state creation:   {game_time:.3f}s ({100*game_time/total_time:.1f}%)")
    print(f"GPU cache clear:       {cache_time:.3f}s ({100*cache_time/total_time:.1f}%)")
    print(f"Pure search:           {search_time:.3f}s ({100*search_time/total_time:.1f}%)")
    print(f"TOTAL TIME:            {total_time:.3f}s")
    
    overhead = total_time - search_time
    print(f"\nNon-search overhead:   {overhead:.3f}s ({100*overhead/total_time:.1f}%)")
    
    # Calculate performance
    sims_per_second = config.num_simulations / search_time
    total_sims_per_second = config.num_simulations / total_time
    
    print(f"\nPure search performance: {sims_per_second:.0f} sims/s")
    print(f"Total performance:       {total_sims_per_second:.0f} sims/s")
    print(f"Overhead factor:         {total_time/search_time:.1f}x")
    
    if search_time < 1.0:
        print("✅ Core search performance is good!")
    else:
        print("❌ Core search performance is slow")
    
    if overhead > search_time:
        print("⚠️  Overhead is larger than search time - this is the main issue")
    else:
        print("✅ Overhead is reasonable")
    
    return {
        'config_time': config_time,
        'evaluator_time': evaluator_time,
        'mcts_time': mcts_time,
        'opt_time': opt_time,
        'game_time': game_time,
        'cache_time': cache_time,
        'search_time': search_time,
        'total_time': total_time,
        'overhead': overhead,
        'search_sims_per_sec': sims_per_second,
        'total_sims_per_sec': total_sims_per_second
    }


def main():
    """Main timing analysis"""
    try:
        results = run_detailed_timing_analysis()
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        
        # Identify the main bottleneck
        times = {
            'MCTS creation': results['mcts_time'],
            'Search execution': results['search_time'],
            'Other overhead': results['overhead'] - results['mcts_time']
        }
        
        bottleneck = max(times.items(), key=lambda x: x[1])
        print(f"Main bottleneck: {bottleneck[0]} ({bottleneck[1]:.3f}s)")
        
        if results['search_sims_per_sec'] > 1000:
            print("🎯 Core MCTS performance is good - the issue is initialization overhead")
        else:
            print("🎯 Core MCTS search itself needs optimization")
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()