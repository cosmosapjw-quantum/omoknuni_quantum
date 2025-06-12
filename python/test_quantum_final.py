#!/usr/bin/env python3
"""Final test of quantum MCTS with proper kernel loading"""

# Force fresh imports
import sys
for mod in list(sys.modules.keys()):
    if 'mcts.gpu' in mod:
        del sys.modules[mod]

import torch
import numpy as np
import time
from mcts.core.optimized_mcts import MCTS, MCTSConfig
from mcts.quantum.quantum_features import QuantumConfig
from mcts.gpu.gpu_game_states import GameType

# Check kernel loading
from mcts.gpu.unified_kernels import _UNIFIED_KERNELS, _KERNELS_AVAILABLE

print("Quantum MCTS Final Test")
print("=" * 60)

print(f"\nKernel loading status:")
print(f"  _KERNELS_AVAILABLE: {_KERNELS_AVAILABLE}")
if _UNIFIED_KERNELS:
    print(f"  Module: {_UNIFIED_KERNELS}")
    has_quantum = hasattr(_UNIFIED_KERNELS, 'batched_ucb_selection_quantum')
    print(f"  Has quantum kernel: {has_quantum}")

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

# Test configurations
configs = [
    ("Classical MCTS", False),
    ("Quantum MCTS", True),
]

# Dummy state
class DummyState:
    def __init__(self):
        self.board = np.zeros((15, 15), dtype=np.int8)
        self.current_player = 1
        self.move_count = 0
    def to_tensor(self):
        return torch.tensor(self.board, dtype=torch.int8)

results = []

for name, enable_quantum in configs:
    print(f"\n{name}:")
    print("-" * 40)
    
    # Create config
    config = MCTSConfig(
        num_simulations=1000,
        min_wave_size=3072,
        max_wave_size=3072,
        adaptive_wave_sizing=False,
        device='cuda',
        game_type=GameType.GOMOKU,
        board_size=15,
        enable_quantum=enable_quantum,
        enable_debug_logging=False
    )
    
    if enable_quantum:
        quantum_config = QuantumConfig(
            enable_quantum=True,
            hbar_eff=0.05,
            phase_kick_strength=0.1,
            interference_alpha=0.05,
            min_wave_size=3072,
            optimal_wave_size=3072,
            device='cuda',
            fast_mode=True
        )
        config.quantum_config = quantum_config
    
    # Create MCTS
    evaluator = DummyEvaluator(board_size=15, device='cuda')
    mcts = MCTS(config, evaluator)
    mcts.optimize_for_hardware()
    
    state = DummyState()
    
    # Warm up
    for _ in range(2):
        mcts.search(state, num_simulations=100)
        mcts.reset_tree()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    policy = mcts.search(state, num_simulations=1000)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    sims_per_sec = 1000 / elapsed
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Speed: {sims_per_sec:.0f} sims/s")
    
    # Check kernel usage
    kernel_type = "Unknown"
    if hasattr(mcts.gpu_ops, 'stats'):
        quantum_calls = mcts.gpu_ops.stats.get('quantum_calls', 0)
        ucb_calls = mcts.gpu_ops.stats.get('ucb_calls', 0)
        if quantum_calls > 0:
            kernel_type = f"CUDA Quantum ({quantum_calls}/{ucb_calls} calls)"
        elif ucb_calls > 0:
            kernel_type = "CUDA Classical"
    
    print(f"  Kernel: {kernel_type}")
    
    results.append((name, elapsed, sims_per_sec, kernel_type))

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Configuration':<20} {'Time (s)':<10} {'Speed (sims/s)':<15} {'Kernel':<25}")
print("-" * 70)

for name, elapsed, speed, kernel in results:
    print(f"{name:<20} {elapsed:<10.3f} {speed:<15.0f} {kernel:<25}")