#!/usr/bin/env python3
"""Test MCTS performance with larger simulations"""

import torch
import time
from mcts.core.mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType
import alphazero_py


class FastEvaluator:
    """Fast evaluator for benchmarking"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Pre-allocate for speed
        self.policies = torch.ones(1000, 225, device=self.device) / 225
        self.values = torch.zeros(1000, 1, device=self.device)
        
    def evaluate_batch(self, features, legal_masks=None):
        batch_size = features.shape[0]
        return self.policies[:batch_size], self.values[:batch_size]


def test_performance():
    """Test MCTS performance at different scales"""
    print("=== MCTS Performance Test ===\n")
    
    evaluator = FastEvaluator()
    
    # Test different configurations
    configs = [
        (100, 256),    # 100 sims, wave_size 256
        (1000, 256),   # 1000 sims, wave_size 256
        (1000, 1024),  # 1000 sims, wave_size 1024
        (5000, 3072),  # 5000 sims, wave_size 3072
    ]
    
    for num_sims, wave_size in configs:
        print(f"\nTesting {num_sims} simulations with wave_size={wave_size}")
        
        config = MCTSConfig(
            num_simulations=num_sims,
            device='cuda',
            game_type=GameType.GOMOKU,
            wave_size=wave_size,
            board_size=15
        )
        
        mcts = MCTS(config, evaluator)
        game_state = alphazero_py.GomokuState()
        
        # Warmup
        mcts.search(game_state, num_simulations=10)
        
        # Time the search
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        policy = mcts.search(game_state, num_simulations=num_sims)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        sims_per_sec = num_sims / elapsed
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Simulations/second: {sims_per_sec:,.0f}")
        print(f"  Policy sum: {policy.sum():.6f}")
        
        # Get tree stats
        stats = mcts.get_statistics()
        print(f"  Tree nodes: {stats.get('tree_nodes', 'N/A')}")
        
        # Check if we're meeting performance targets
        if sims_per_sec > 100000:
            print(f"  ✓ Performance target met!")
        else:
            print(f"  ✗ Need {100000/sims_per_sec:.1f}x speedup")


if __name__ == "__main__":
    test_performance()