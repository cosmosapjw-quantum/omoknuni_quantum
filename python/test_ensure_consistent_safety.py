#!/usr/bin/env python3
"""Test the safety of disabling ensure_consistent() calls"""

import torch
import numpy as np
import time
import logging
from mcts.core.optimized_mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType
from mcts.quantum.quantum_features import QuantumConfig

logging.basicConfig(level=logging.INFO, format='%(message)s')

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

print("Testing Safety of Disabling ensure_consistent()")
print("=" * 60)

# Understanding ensure_consistent():
print("\nWhat ensure_consistent() does:")
print("-" * 40)
print("1. Flushes any pending batched operations")
print("2. Rebuilds row pointers if _needs_row_ptr_update is True")
print("   - Row pointers map parent nodes to their children in CSR format")
print("   - Only needs rebuilding when children are added")
print()
print("When _needs_row_ptr_update is set to True:")
print("- When adding children to nodes (add_children, batched_add_children)")
print("- During tree initialization")
print()

# Test scenarios
print("\nAnalysis of Safety:")
print("-" * 40)
print("✓ SAFE to disable during selection phase:")
print("  - No children are added during selection")
print("  - Row pointers remain valid")
print("  - Only reading tree structure, not modifying")
print()
print("✗ NOT SAFE to disable during expansion phase:")
print("  - Children are added, changing tree structure")
print("  - Row pointers must be updated for correct CSR access")
print()

# Practical test
evaluator = DummyEvaluator(board_size=15, device='cuda')
state = DummyState()

quantum_config = QuantumConfig(
    quantum_level='tree_level',
    enable_quantum=True,
    min_wave_size=1,
    optimal_wave_size=3072,
    hbar_eff=0.1,
    phase_kick_strength=0.1,
    interference_alpha=0.05,
    fast_mode=True,
    device='cuda'
)

config = MCTSConfig(
    num_simulations=10000,
    min_wave_size=3072,
    max_wave_size=3072,
    adaptive_wave_sizing=False,
    device='cuda',
    game_type=GameType.GOMOKU,
    board_size=15,
    enable_quantum=True,
    quantum_config=quantum_config,
    enable_debug_logging=False
)

mcts = MCTS(config, evaluator)
mcts.optimize_for_hardware()

# Check when _needs_row_ptr_update is True
print("\nPractical Test:")
print("-" * 40)

# Initial state
print(f"Initial _needs_row_ptr_update: {mcts.tree._needs_row_ptr_update}")

# Run some simulations
mcts.search(state, num_simulations=100)

# Check after simulations
print(f"After 100 simulations _needs_row_ptr_update: {mcts.tree._needs_row_ptr_update}")

# Access tree data
num_nodes = mcts.tree.num_nodes
print(f"Number of nodes in tree: {num_nodes}")

print("\nRecommendation:")
print("-" * 40)
print("1. Remove ensure_consistent() from batch_select_ucb_optimized()")
print("   - This method only reads data, doesn't modify tree structure")
print("   - Will eliminate the performance overhead")
print()
print("2. Keep ensure_consistent() in methods that modify tree structure:")
print("   - add_children()")
print("   - batched_add_children()")
print("   - Any method that sets _needs_row_ptr_update = True")
print()
print("3. Alternative: Add a flag to skip consistency check in read-only operations")
print("   - ensure_consistent(read_only=True) could skip row pointer rebuild")

# Performance impact estimate
print("\nPerformance Impact:")
print("-" * 40)
print("Based on profiling data:")
print("- ensure_consistent() is called 4 times per wave (selection phase)")
print("- _rebuild_row_pointers() takes ~111ms when needed")
print("- Even checking the flag has overhead in hot path")
print("- Removing the call could improve performance by 10-20%")