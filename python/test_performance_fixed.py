#!/usr/bin/env python3
"""Test MCTS performance after fixing the issue"""

import torch
import numpy as np
import time
from mcts.core.optimized_mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType

# Simple evaluator
class DummyEvaluator:
    def __init__(self, board_size=15, device='cuda'):
        self.board_size = board_size
        self.device = torch.device(device)
        self.num_actions = board_size * board_size
    
    def evaluate_batch(self, features):
        batch_size = features.shape[0]
        policies = torch.ones((batch_size, self.num_actions), device=self.device) / self.num_actions
        values = torch.zeros((batch_size, 1), device=self.device)
        return policies, values

# Dummy state
class DummyState:
    def __init__(self):
        self.board = np.zeros((15, 15), dtype=np.int8)
        self.current_player = 1
        self.move_count = 0
    def to_tensor(self):
        return torch.tensor(self.board, dtype=torch.int8)

print("MCTS Performance Test (Fixed)")
print("=" * 60)

# Test with optimal configuration
config = MCTSConfig(
    num_simulations=10000,
    min_wave_size=3072,
    max_wave_size=3072,
    adaptive_wave_sizing=False,
    device='cuda',
    game_type=GameType.GOMOKU,
    board_size=15,
    enable_quantum=False,  # Test base performance first
    enable_debug_logging=False
)

evaluator = DummyEvaluator(board_size=15, device='cuda')
mcts = MCTS(config, evaluator)
mcts.optimize_for_hardware()

state = DummyState()

# Warm up
print("Warming up...")
for _ in range(3):
    mcts.search(state, num_simulations=100)
    mcts.reset_tree()

# Run multiple tests
times = []
for i in range(5):
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    policy = mcts.search(state, num_simulations=10000)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    times.append(elapsed)
    
    sims_per_sec = 10000 / elapsed
    print(f"Run {i+1}: {elapsed:.3f}s ({sims_per_sec:.0f} sims/s)")
    
    mcts.reset_tree()

avg_time = np.mean(times)
avg_speed = 10000 / avg_time
print(f"\nAverage: {avg_time:.3f}s ({avg_speed:.0f} sims/s)")

# Check if we're back to high performance
if avg_speed > 50000:
    print("✓ Performance restored! Achieving 50k+ sims/s")
elif avg_speed > 20000:
    print("△ Partial recovery, but still below optimal")
else:
    print("✗ Performance still degraded")