#!/usr/bin/env python3
"""Benchmark script to compare MCTS performance before and after refactoring

This script measures the performance improvement from the unified MCTS implementation.
"""

import torch
import time
import numpy as np
import gc
from mcts.core.mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType
import alphazero_py


class SimpleEvaluator:
    """Simple evaluator that returns uniform random policy"""
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        
    def evaluate_batch(self, features, legal_masks=None):
        batch_size = features.shape[0]
        board_size = features.shape[-1]
        action_size = board_size * board_size
        
        # Simple evaluation - uniform policy with small random noise
        values = torch.zeros(batch_size, 1, device=self.device)
        policies = torch.ones(batch_size, action_size, device=self.device) / action_size
        policies += torch.rand_like(policies) * 0.1
        policies = policies / policies.sum(dim=1, keepdim=True)
        
        return policies, values  # Note: policies first, then values


def benchmark_configuration(config_name, config, num_runs=5, warmup_runs=2):
    """Benchmark a specific configuration"""
    print(f"\n=== Benchmarking {config_name} ===")
    
    # Create evaluator and MCTS
    evaluator = SimpleEvaluator(config.device)
    mcts = MCTS(config, evaluator)
    
    # Optimize for hardware
    mcts.optimize_for_hardware()
    
    # Create game state
    game_state = alphazero_py.GomokuState()
    
    # Warmup
    print("Warming up...")
    for _ in range(warmup_runs):
        mcts.search(game_state, num_simulations=1000)
        
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Benchmark runs
    print(f"Running {num_runs} benchmark iterations...")
    results = []
    
    for i in range(num_runs):
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Run search
        start = time.perf_counter()
        policy = mcts.search(game_state, num_simulations=config.num_simulations)
        end = time.perf_counter()
        
        elapsed = end - start
        sims_per_sec = config.num_simulations / elapsed
        
        results.append(sims_per_sec)
        print(f"  Run {i+1}: {sims_per_sec:,.0f} sims/s ({elapsed:.3f}s)")
        
    # Calculate statistics
    results = np.array(results)
    mean_sps = results.mean()
    std_sps = results.std()
    max_sps = results.max()
    min_sps = results.min()
    
    print(f"\nResults for {config_name}:")
    print(f"  Mean: {mean_sps:,.0f} sims/s")
    print(f"  Std:  {std_sps:,.0f} sims/s")
    print(f"  Max:  {max_sps:,.0f} sims/s")
    print(f"  Min:  {min_sps:,.0f} sims/s")
    
    # Get final statistics
    stats = mcts.get_statistics()
    print(f"  Tree nodes: {stats.get('tree_nodes', 0):,}")
    print(f"  GPU memory: {stats.get('gpu_memory_mb', 0):.1f} MB")
    
    return {
        'mean': mean_sps,
        'std': std_sps,
        'max': max_sps,
        'min': min_sps,
        'stats': stats
    }


def main():
    """Run benchmarks with different configurations"""
    print("=== MCTS Performance Benchmark ===")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Define configurations to test
    base_config = {
        'num_simulations': 10000,
        'c_puct': 1.414,
        'temperature': 1.0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'game_type': GameType.GOMOKU,
        'board_size': 15,
        'enable_virtual_loss': True
    }
    
    configurations = [
        ("Small Wave (256)", {**base_config, 'wave_size': 256}),
        ("Medium Wave (1024)", {**base_config, 'wave_size': 1024}),
        ("Large Wave (2048)", {**base_config, 'wave_size': 2048}),
        ("Optimal Wave (3072)", {**base_config, 'wave_size': 3072}),
        ("Max Wave (4096)", {**base_config, 'wave_size': 4096}),
    ]
    
    # Add quantum configuration if GPU available
    if torch.cuda.is_available():
        configurations.append(
            ("Quantum Enhanced (3072)", {
                **base_config, 
                'wave_size': 3072,
                'enable_quantum': True
            })
        )
    
    # Run benchmarks
    results = {}
    for name, config_dict in configurations:
        config = MCTSConfig(**config_dict)
        results[name] = benchmark_configuration(name, config)
        
    # Summary
    print("\n=== Performance Summary ===")
    print(f"{'Configuration':<25} {'Mean (sims/s)':<15} {'Max (sims/s)':<15} {'Speedup':<10}")
    print("-" * 65)
    
    baseline = results[configurations[0][0]]['mean']
    for name, result in results.items():
        speedup = result['mean'] / baseline
        print(f"{name:<25} {result['mean']:>14,.0f} {result['max']:>14,.0f} {speedup:>9.2f}x")
        
    # Find best configuration
    best_config = max(results.items(), key=lambda x: x[1]['mean'])
    print(f"\nBest configuration: {best_config[0]}")
    print(f"Peak performance: {best_config[1]['max']:,.0f} simulations/second")
    
    # Performance vs target
    target = 300000  # Target from refactoring plan
    best_mean = best_config[1]['mean']
    print(f"\nPerformance vs target:")
    print(f"  Target: {target:,} sims/s")
    print(f"  Achieved: {best_mean:,.0f} sims/s")
    print(f"  Ratio: {best_mean/target:.2%}")


if __name__ == "__main__":
    main()