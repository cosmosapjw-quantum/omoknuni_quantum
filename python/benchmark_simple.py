#!/usr/bin/env python3
"""Simple performance benchmark for unified MCTS"""

import torch
import time
import numpy as np
from mcts.core.mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType
import alphazero_py


class FastEvaluator:
    """Fast evaluator for benchmarking"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Pre-allocate tensors for speed
        self.uniform_policy = torch.ones(1000, 225, device=self.device) / 225
        self.zero_values = torch.zeros(1000, 1, device=self.device)
        
    def evaluate_batch(self, features, legal_masks=None):
        batch_size = features.shape[0]
        return self.uniform_policy[:batch_size], self.zero_values[:batch_size]


def benchmark_wave_size(wave_size, num_sims=5000):
    """Benchmark a specific wave size"""
    print(f"\nBenchmarking wave_size={wave_size}")
    
    config = MCTSConfig(
        num_simulations=num_sims,
        device='cuda',
        game_type=GameType.GOMOKU,
        wave_size=wave_size
    )
    
    evaluator = FastEvaluator()
    mcts = MCTS(config, evaluator)
    
    # Warmup
    game_state = alphazero_py.GomokuState()
    mcts.search(game_state, num_simulations=100)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    policy = mcts.search(game_state, num_simulations=num_sims)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    sims_per_sec = num_sims / elapsed
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Sims/sec: {sims_per_sec:,.0f}")
    
    # Get stats
    stats = mcts.get_statistics()
    print(f"  Tree nodes: {stats.get('tree_nodes', 'N/A')}")
    print(f"  GPU memory: {stats.get('gpu_memory_mb', 0):.1f} MB")
    
    return sims_per_sec


def main():
    print("=== Simple MCTS Performance Benchmark ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
        
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Test different wave sizes
    wave_sizes = [256, 512, 1024, 2048, 3072]
    results = {}
    
    for wave_size in wave_sizes:
        try:
            sps = benchmark_wave_size(wave_size, num_sims=2000)
            results[wave_size] = sps
        except Exception as e:
            print(f"  Failed: {e}")
            results[wave_size] = 0
            
    # Summary
    print("\n=== Summary ===")
    print(f"{'Wave Size':<12} {'Sims/sec':<15}")
    print("-" * 27)
    
    best_wave = 0
    best_sps = 0
    
    for wave_size, sps in results.items():
        print(f"{wave_size:<12} {sps:>14,.0f}")
        if sps > best_sps:
            best_sps = sps
            best_wave = wave_size
            
    print(f"\nBest configuration: wave_size={best_wave}")
    print(f"Peak performance: {best_sps:,.0f} simulations/second")


if __name__ == "__main__":
    main()