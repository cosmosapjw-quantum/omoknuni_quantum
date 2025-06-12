#!/usr/bin/env python3
"""Simple performance test"""

import torch
import time
from mcts.core.mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType
import alphazero_py


class DummyEvaluator:
    def __init__(self):
        self.device = torch.device('cuda')
    def evaluate_batch(self, features, legal_masks=None):
        b = features.shape[0]
        return torch.ones(b, 225, device=self.device)/225, torch.zeros(b, 1, device=self.device)


# Test
config = MCTSConfig(
    device='cuda',
    game_type=GameType.GOMOKU,
    wave_size=3072,
    board_size=15
)

evaluator = DummyEvaluator()
mcts = MCTS(config, evaluator)
state = alphazero_py.GomokuState()

# Warmup
mcts.search(state, 100)

# Benchmark
torch.cuda.synchronize()
start = time.perf_counter()

policy = mcts.search(state, 10000)

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print(f"10000 simulations in {elapsed:.3f}s")
print(f"Simulations/second: {10000/elapsed:,.0f}")
print(f"Policy sum: {policy.sum()}")

# Check what's slow
stats = mcts.get_statistics()
print(f"\nTree nodes: {stats.get('tree_nodes', 0)}")
print(f"Sims/sec from stats: {stats.get('sims_per_second', 0):,.0f}")