#!/usr/bin/env python3
"""Test to identify the ensure_consistent() performance issue"""

import torch
import numpy as np
import time
import logging
from mcts.core.optimized_mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType
from mcts.quantum.quantum_features import QuantumConfig
from mcts.gpu.csr_tree import CSRTree

logging.basicConfig(level=logging.WARNING)

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

# Monkey-patch CSRTree to remove ensure_consistent() calls
original_batch_select_ucb = CSRTree.batch_select_ucb_optimized

def batch_select_ucb_no_consistency_check(
    self,
    node_indices: torch.Tensor,
    c_puct: float = 1.4,
    temperature: float = 1.0,
    **quantum_params
):
    """batch_select_ucb_optimized without ensure_consistent() call"""
    # Comment out the ensure_consistent() call
    # self.ensure_consistent()
    
    # Call rest of original method - we need to copy the logic
    # since we can't call super() on a monkey-patched method
    
    # Use the batch_ops if available for true UCB selection
    if self.batch_ops is not None and hasattr(self.batch_ops, 'batch_ucb_selection'):
        selected_actions, selected_scores = self.batch_ops.batch_ucb_selection(
            node_indices, c_puct=c_puct, **quantum_params
        )
        return selected_actions, selected_scores
    
    # Fallback implementation
    batch_size = len(node_indices)
    selected_actions = torch.full((batch_size,), -1, dtype=torch.int32, device=self.device)
    selected_scores = torch.zeros(batch_size, device=self.device)
    
    for i, node_idx in enumerate(node_indices):
        if node_idx < 0 or node_idx >= self.num_nodes:
            continue
            
        start = self.row_ptr[node_idx]
        end = self.row_ptr[node_idx + 1]
        
        if start >= end:
            continue
            
        children = self.col_indices[start:end]
        actions = self.edge_actions[start:end]
        priors = self.edge_priors[start:end]
        
        valid_mask = children >= 0
        if not valid_mask.any():
            continue
            
        valid_children = children[valid_mask]
        valid_actions = actions[valid_mask]
        valid_priors = priors[valid_mask]
        
        visit_counts = self.visit_counts[valid_children]
        q_values = self.q_values[valid_children]
        parent_visits = self.visit_counts[node_idx]
        
        ucb_scores = q_values + c_puct * valid_priors * torch.sqrt(parent_visits) / (1 + visit_counts)
        
        best_idx = ucb_scores.argmax()
        selected_actions[i] = valid_actions[best_idx]
        selected_scores[i] = ucb_scores[best_idx]
    
    return selected_actions, selected_scores

print("Testing ensure_consistent() Performance Impact")
print("=" * 60)

evaluator = DummyEvaluator(board_size=15, device='cuda')
state = DummyState()

# Test configurations
test_configs = [
    ("With ensure_consistent()", False, False),
    ("Without ensure_consistent()", True, False),
    ("Quantum + with ensure_consistent()", False, True),
    ("Quantum + without ensure_consistent()", True, True),
]

results = []

for name, remove_consistency_check, enable_quantum in test_configs:
    print(f"\n{name}:")
    print("-" * 40)
    
    # Apply or remove monkey patch
    if remove_consistency_check:
        CSRTree.batch_select_ucb_optimized = batch_select_ucb_no_consistency_check
    else:
        CSRTree.batch_select_ucb_optimized = original_batch_select_ucb
    
    # Create quantum config if needed
    quantum_config = None
    if enable_quantum:
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
        num_simulations=5000,
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
    mcts.search(state, num_simulations=500)
    mcts.reset_tree()
    
    # Measure performance
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    policy = mcts.search(state, num_simulations=5000)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    sims_per_sec = 5000 / elapsed
    
    results.append({
        'name': name,
        'time': elapsed,
        'sims_per_sec': sims_per_sec,
    })
    
    print(f"Time: {elapsed:.3f}s")
    print(f"Performance: {sims_per_sec:,.0f} sims/s")

# Restore original method
CSRTree.batch_select_ucb_optimized = original_batch_select_ucb

print("\n" + "=" * 60)
print("Summary:")
print("-" * 60)

baseline_speed = results[0]['sims_per_sec']
for result in results:
    speedup = result['sims_per_sec'] / baseline_speed
    print(f"{result['name']:40s}: {result['sims_per_sec']:8,.0f} sims/s ({speedup:.2f}x)")

print("\nKey Finding:")
print("-" * 60)
if results[1]['sims_per_sec'] > results[0]['sims_per_sec'] * 1.5:
    print("✓ ensure_consistent() is causing significant performance overhead!")
    print(f"  Removing it improves performance by {(results[1]['sims_per_sec'] / results[0]['sims_per_sec'] - 1) * 100:.0f}%")
else:
    print("✗ ensure_consistent() is NOT the main performance bottleneck in this test")