#!/usr/bin/env python3
"""Test specifically to verify quantum CUDA kernels are being called"""

import torch
import numpy as np
import logging
from mcts.core.optimized_mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType

# Enable detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

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

print("Testing Quantum CUDA Kernel Usage")
print("=" * 60)

# Test with quantum enabled
config = MCTSConfig(
    num_simulations=1000,
    min_wave_size=100,  # Smaller wave for debugging
    max_wave_size=100,
    adaptive_wave_sizing=False,
    device='cuda',
    game_type=GameType.GOMOKU,
    board_size=15,
    enable_quantum=True,
    enable_debug_logging=True
)

evaluator = DummyEvaluator(board_size=15, device='cuda')
mcts = MCTS(config, evaluator)
state = DummyState()

print("\nRunning search with quantum enabled...")
policy = mcts.search(state, num_simulations=1000)

# Check statistics
stats = mcts.get_statistics()
print(f"\nSearch complete:")
print(f"- Total simulations: {stats.get('total_simulations', 0)}")
print(f"- Simulations/second: {stats.get('sims_per_second', 0):.0f}")

# Check if quantum kernels were used
if hasattr(mcts.tree.batch_ops, 'stats'):
    kernel_stats = mcts.tree.batch_ops.stats
    print(f"\nKernel statistics:")
    print(f"- Total UCB calls: {kernel_stats.get('ucb_calls', 0)}")
    print(f"- Quantum kernel calls: {kernel_stats.get('quantum_calls', 0)}")
    print(f"- Backup calls: {kernel_stats.get('backup_calls', 0)}")
    
    if kernel_stats.get('quantum_calls', 0) > 0:
        print("✓ Quantum CUDA kernels are being used!")
    else:
        print("✗ Quantum CUDA kernels are NOT being called")
else:
    print("\nNo kernel statistics available")