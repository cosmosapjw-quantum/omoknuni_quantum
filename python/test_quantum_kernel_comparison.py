#!/usr/bin/env python3
"""Direct comparison of classical vs quantum MCTS performance"""

import torch
import numpy as np
import time
import logging
from mcts.core.optimized_mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType
from mcts.quantum.quantum_features import QuantumConfig

# Enable logging to see kernel usage
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

print("Direct Quantum vs Classical Comparison")
print("=" * 60)

evaluator = DummyEvaluator(board_size=15, device='cuda')
state = DummyState()

# Test configurations
test_configs = [
    ("Classical", False, None),
    ("Tree-level Quantum", True, 'tree_level'),
    ("One-loop Quantum", True, 'one_loop'),
]

results = []

for name, enable_quantum, quantum_level in test_configs:
    print(f"\n{name}:")
    print("-" * 30)
    
    # Create quantum config if needed
    quantum_config = None
    if enable_quantum:
        quantum_config = QuantumConfig(
            quantum_level=quantum_level,
            enable_quantum=True,
            min_wave_size=3072,
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
        enable_quantum=enable_quantum,
        quantum_config=quantum_config,
        enable_debug_logging=False
    )
    
    mcts = MCTS(config, evaluator)
    mcts.optimize_for_hardware()
    
    # Warmup
    for _ in range(3):
        mcts.search(state, num_simulations=1000)
        mcts.reset_tree()
    
    # Measure performance
    times = []
    for i in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        policy = mcts.search(state, num_simulations=10000)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        mcts.reset_tree()
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    sims_per_sec = 10000 / avg_time
    
    # Check kernel statistics
    quantum_calls = 0
    total_calls = 0
    if hasattr(mcts.tree, 'batch_ops') and mcts.tree.batch_ops:
        if hasattr(mcts.tree.batch_ops, 'stats'):
            stats = mcts.tree.batch_ops.stats
            quantum_calls = stats.get('quantum_calls', 0)
            total_calls = stats.get('ucb_calls', 0)
    
    results.append({
        'name': name,
        'avg_time': avg_time,
        'std_time': std_time,
        'sims_per_sec': sims_per_sec,
        'quantum_calls': quantum_calls,
        'total_calls': total_calls
    })
    
    print(f"Average time: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"Performance: {sims_per_sec:,.0f} sims/s")
    if enable_quantum and total_calls > 0:
        print(f"Quantum kernel usage: {quantum_calls}/{total_calls} ({quantum_calls/total_calls*100:.1f}%)")

print("\n" + "=" * 60)
print("Summary:")
print("-" * 60)

classical_speed = results[0]['sims_per_sec']
for result in results:
    speedup = result['sims_per_sec'] / classical_speed
    overhead = (1 - speedup) * 100
    print(f"{result['name']:20s}: {result['sims_per_sec']:8,.0f} sims/s ({speedup:.2f}x, {overhead:+.1f}% overhead)")

# Check if quantum is being used properly
print("\nQuantum Kernel Analysis:")
for result in results[1:]:  # Skip classical
    if result['total_calls'] > 0:
        quantum_percent = result['quantum_calls'] / result['total_calls'] * 100
        print(f"{result['name']:20s}: {quantum_percent:.1f}% quantum kernel usage")
        if quantum_percent < 50:
            print("  ⚠️  Low quantum kernel usage - check implementation")
    else:
        print(f"{result['name']:20s}: No kernel calls recorded")