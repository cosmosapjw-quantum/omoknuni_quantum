#!/usr/bin/env python3
"""Test parallel MCTS speedup with CPU-only instances"""

import time
import psutil
import numpy as np
import multiprocessing as mp

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts import (
    HighPerformanceMCTS, HighPerformanceMCTSConfig,
    GameInterface, GameType,
    MockEvaluator, EvaluatorConfig
)
from mcts.parallel_mcts import ParallelMCTS, ParallelMCTSConfig, HybridParallelMCTS


def test_sequential_baseline():
    """Test sequential CPU-only performance"""
    print("=== SEQUENTIAL BASELINE (CPU-only) ===")
    
    # Create CPU-only configuration
    game = GameInterface(GameType.GOMOKU)
    
    eval_config = EvaluatorConfig(
        batch_size=64,
        device='cpu',
        use_fp16=False
    )
    evaluator = MockEvaluator(eval_config, 225)
    
    mcts_config = HighPerformanceMCTSConfig(
        num_simulations=400,
        wave_size=128,
        enable_gpu=False,
        device='cpu',
        mixed_precision=False,
        max_tree_size=50_000
    )
    
    mcts = HighPerformanceMCTS(mcts_config, game, evaluator)
    state = game.create_initial_state()
    
    # Run sequential searches
    num_searches = 10
    start_time = time.time()
    
    for i in range(num_searches):
        policy_dict = mcts.search(state)
        print(f".", end="", flush=True)
        
    elapsed = time.time() - start_time
    print()
    
    total_sims = num_searches * mcts_config.num_simulations
    sims_per_sec = total_sims / elapsed
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Total simulations: {total_sims:,}")
    print(f"Throughput: {sims_per_sec:,.0f} sims/s")
    
    return sims_per_sec


def test_parallel_cpu():
    """Test parallel CPU performance"""
    print("\n=== PARALLEL CPU MCTS ===")
    
    # Number of CPU instances
    num_cores = psutil.cpu_count(logical=False)
    num_instances = min(num_cores - 2, 10)  # Leave some cores for system
    print(f"Using {num_instances} CPU instances on {num_cores} physical cores")
    
    # Create configurations
    base_config = HighPerformanceMCTSConfig(
        num_simulations=400,  # Will be adjusted by ParallelMCTS
        wave_size=128,
        enable_gpu=False,
        device='cpu',
        mixed_precision=False,
        max_tree_size=100_000
    )
    
    eval_config = EvaluatorConfig(
        batch_size=64,
        device='cpu',
        use_fp16=False
    )
    
    parallel_config = ParallelMCTSConfig(
        num_instances=num_instances,
        simulations_per_instance=400,
        use_gpu_per_instance=False,
        aggregate_method='average'
    )
    
    # Create parallel MCTS
    parallel_mcts = ParallelMCTS(
        base_config,
        GameType.GOMOKU,
        eval_config,
        parallel_config
    )
    
    # Create test state
    game = GameInterface(GameType.GOMOKU)
    state = game.create_initial_state()
    
    # Run parallel searches
    num_searches = 10
    start_time = time.time()
    
    for i in range(num_searches):
        result = parallel_mcts.search(state)
        print(f".", end="", flush=True)
        
    elapsed = time.time() - start_time
    print()
    
    # Get statistics
    stats = parallel_mcts.get_statistics()
    total_sims = num_searches * stats['total_simulations']
    sims_per_sec = total_sims / elapsed
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Total simulations: {total_sims:,} ({stats['simulations_per_instance']} x {stats['num_instances']} instances)")
    print(f"Throughput: {sims_per_sec:,.0f} sims/s")
    
    # Clean up
    parallel_mcts.close()
    
    return sims_per_sec


def test_hybrid_approach():
    """Test hybrid GPU + CPU approach"""
    print("\n=== HYBRID GPU + CPU MCTS ===")
    
    import torch
    if not torch.cuda.is_available():
        print("GPU not available, skipping hybrid test")
        return 0
        
    # GPU configuration (primary)
    gpu_config = HighPerformanceMCTSConfig(
        num_simulations=1600,
        wave_size=2048,
        enable_gpu=True,
        device='cuda',
        mixed_precision=True,
        max_tree_size=200_000
    )
    
    # CPU configuration (diversity)
    cpu_config = HighPerformanceMCTSConfig(
        num_simulations=200,
        wave_size=64,
        enable_gpu=False,
        device='cpu',
        mixed_precision=False,
        max_tree_size=20_000
    )
    
    eval_config = EvaluatorConfig(
        batch_size=1024,
        device='cuda',
        use_fp16=True
    )
    
    # Create hybrid MCTS
    num_cpu_instances = min(psutil.cpu_count(logical=False) - 4, 8)
    
    hybrid_mcts = HybridParallelMCTS(
        gpu_config,
        cpu_config,
        GameType.GOMOKU,
        eval_config,
        num_cpu_instances=num_cpu_instances
    )
    
    print(f"GPU: {gpu_config.num_simulations} simulations")
    print(f"CPU: {num_cpu_instances} instances x {cpu_config.num_simulations} simulations")
    
    # Create test state
    game = GameInterface(GameType.GOMOKU)
    state = game.create_initial_state()
    
    # Run searches
    num_searches = 5
    start_time = time.time()
    
    for i in range(num_searches):
        result = hybrid_mcts.search(state)
        print(f".", end="", flush=True)
        
    elapsed = time.time() - start_time
    print()
    
    # Get statistics
    stats = hybrid_mcts.get_statistics()
    total_sims = num_searches * stats['total_simulations']
    sims_per_sec = total_sims / elapsed
    
    print(f"Time: {elapsed:.2f}s")
    print(f"GPU simulations: {stats['gpu_simulations'] * num_searches:,}")
    print(f"CPU simulations: {stats['cpu_simulations'] * num_searches:,}")
    print(f"Total simulations: {total_sims:,}")
    print(f"Throughput: {sims_per_sec:,.0f} sims/s")
    
    return sims_per_sec


def main():
    """Run all tests and show speedup"""
    print("="*60)
    print("PARALLEL MCTS SPEEDUP TEST")
    print("="*60)
    
    # Run tests
    seq_throughput = test_sequential_baseline()
    par_throughput = test_parallel_cpu()
    hybrid_throughput = test_hybrid_approach()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"\nSequential (1 CPU): {seq_throughput:,.0f} sims/s")
    print(f"Parallel CPU: {par_throughput:,.0f} sims/s ({par_throughput/seq_throughput:.1f}x speedup)")
    
    if hybrid_throughput > 0:
        print(f"Hybrid GPU+CPU: {hybrid_throughput:,.0f} sims/s ({hybrid_throughput/seq_throughput:.1f}x speedup)")
        
    # Recommendations
    print("\n=== RECOMMENDATIONS ===")
    
    if par_throughput > seq_throughput * 5:
        print("✅ CPU parallelization is highly effective!")
        print("   Use ParallelMCTS for CPU-heavy workloads")
    else:
        print("⚠️  CPU parallelization speedup is limited")
        print("   Consider reducing per-instance overhead")
        
    if hybrid_throughput > par_throughput * 2:
        print("✅ Hybrid approach is best for GPU systems!")
        print("   GPU handles main search, CPUs add diversity")
    
    # Show how to reach target
    target = 80_000
    best_throughput = max(seq_throughput, par_throughput, hybrid_throughput)
    
    print(f"\nTarget: {target:,} sims/s")
    print(f"Current best: {best_throughput:,.0f} sims/s ({best_throughput/target*100:.1f}% of target)")
    
    if best_throughput < target:
        print(f"\nTo reach target, need {target/best_throughput:.1f}x more performance:")
        print("- Integrate optimized tensor tree (2-3x expected)")
        print("- Fix simulation counting bug")
        print("- Further GPU optimization")
        print("- Use all 24 CPU threads")


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()