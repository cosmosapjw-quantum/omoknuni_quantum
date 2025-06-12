#!/usr/bin/env python3
"""Final MCTS performance benchmark"""

import torch
import time
import numpy as np
from mcts.core.mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType
import alphazero_py


class FastEvaluator:
    """Fast evaluator for benchmarking"""
    def __init__(self):
        self.device = torch.device('cuda')
        # Pre-allocate large tensors
        self.policies = torch.ones(10000, 225, device=self.device) / 225
        self.values = torch.zeros(10000, 1, device=self.device)
        
    def evaluate_batch(self, features, legal_masks=None):
        batch_size = features.shape[0]
        return self.policies[:batch_size], self.values[:batch_size]


def benchmark():
    """Final performance benchmark"""
    print("=== Final MCTS Performance Benchmark ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    # Optimal configuration from testing
    config = MCTSConfig(
        num_simulations=100000,
        device='cuda',
        game_type=GameType.GOMOKU,
        wave_size=3072,  # Optimal for RTX 3060 Ti
        board_size=15,
        dirichlet_epsilon=0.0  # Disable for pure performance test
    )
    
    evaluator = FastEvaluator()
    mcts = MCTS(config, evaluator)
    game_state = alphazero_py.GomokuState()
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        mcts.search(game_state, num_simulations=1000)
    
    print("\nRunning benchmark...")
    
    # Run multiple trials
    trials = 5
    sim_counts = [10000, 50000, 100000]
    
    for num_sims in sim_counts:
        print(f"\n{num_sims} simulations:")
        times = []
        
        for trial in range(trials):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            policy = mcts.search(game_state, num_simulations=num_sims)
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            
            sims_per_sec = num_sims / elapsed
            print(f"  Trial {trial+1}: {elapsed:.3f}s ({sims_per_sec:,.0f} sims/s)")
        
        # Calculate average
        avg_time = np.mean(times)
        avg_sps = num_sims / avg_time
        std_sps = np.std([num_sims/t for t in times])
        
        print(f"  Average: {avg_sps:,.0f} ± {std_sps:,.0f} sims/s")
        
        # Check target
        if avg_sps >= 300000:
            print(f"  ✓ TARGET ACHIEVED! {avg_sps/300000:.1f}x target")
        elif avg_sps >= 100000:
            print(f"  ✓ Good performance: {avg_sps/1000:.0f}k sims/s")
        else:
            print(f"  → Need {300000/avg_sps:.1f}x speedup for target")
    
    # Final stats
    print("\nFinal statistics:")
    stats = mcts.get_statistics()
    print(f"  Total simulations: {stats.get('total_simulations', 0):,}")
    print(f"  Tree nodes: {stats.get('tree_nodes', 0):,}")
    print(f"  Tree memory: {stats.get('tree_memory_mb', 0):.1f} MB")


if __name__ == "__main__":
    benchmark()