#!/usr/bin/env python3
"""Check which kernels are being used"""

import torch
from mcts.core.optimized_mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType

# Dummy evaluator
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
        self.board = torch.zeros((15, 15), dtype=torch.int8)
        self.current_player = 1
        self.move_count = 0
    def to_tensor(self):
        return self.board

print("Checking kernel usage")
print("=" * 60)

# Create MCTS
config = MCTSConfig(
    num_simulations=100,
    min_wave_size=3072,
    max_wave_size=3072,
    adaptive_wave_sizing=False,
    device='cuda',
    game_type=GameType.GOMOKU,
    board_size=15,
    enable_quantum=True,
    enable_debug_logging=False
)

evaluator = DummyEvaluator(board_size=15, device='cuda')
mcts = MCTS(config, evaluator)

# Check what's available
print(f"\n1. MCTS gpu_ops: {mcts.gpu_ops is not None}")
if mcts.gpu_ops:
    print(f"   - use_cuda: {mcts.gpu_ops.use_cuda}")
    print(f"   - Has batch_ucb_selection: {hasattr(mcts.gpu_ops, 'batch_ucb_selection')}")
    
print(f"\n2. Tree batch_ops: {mcts.tree.batch_ops is not None}")
if mcts.tree.batch_ops:
    print(f"   - Has batch_ucb_selection: {hasattr(mcts.tree.batch_ops, 'batch_ucb_selection')}")
    
print(f"\n3. Quantum features: {mcts.quantum_features is not None}")
if mcts.quantum_features:
    print(f"   - Config: {mcts.quantum_features.config.enable_quantum}")

# Check the selection path
print("\n4. Selection path in _select_batch_vectorized:")
import inspect
source = inspect.getsource(mcts._select_batch_vectorized)
# Find the key lines
for line in source.split('\n'):
    if 'batch_select_ucb_optimized' in line or 'gpu_ops' in line:
        print(f"   {line.strip()}")

# Run a small search to check stats
state = DummyState()
mcts.search(state, num_simulations=100)

if hasattr(mcts.gpu_ops, 'stats'):
    print(f"\n5. GPU kernel stats after search:")
    print(f"   - UCB calls: {mcts.gpu_ops.stats.get('ucb_calls', 0)}")
    print(f"   - Quantum calls: {mcts.gpu_ops.stats.get('quantum_calls', 0)}")
    print(f"   - Backup calls: {mcts.gpu_ops.stats.get('backup_calls', 0)}")