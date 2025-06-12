#!/usr/bin/env python3
"""Test quantum MCTS with optimized kernel bypass pattern"""

import torch
import numpy as np
import time
import logging
from typing import Tuple
from mcts.core.optimized_mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType
from mcts.quantum.quantum_features import QuantumConfig, QuantumMCTS

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

# Modified MCTS that bypasses optimized kernels for quantum
class QuantumBypassMCTS(MCTS):
    """MCTS that uses manual UCB computation when quantum is enabled"""
    
    def _select_batch_vectorized(self, active_nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select actions with quantum bypass pattern"""
        # Get children
        children_data = self.tree.batch_get_children(active_nodes)
        
        if children_data[0].dim() > 1:
            children_tensor = children_data[0]
            actions_tensor = children_data[1]
            priors_tensor = children_data[2]
        else:
            children_tensor = children_data[0].unsqueeze(0)
            actions_tensor = children_data[1].unsqueeze(0)
            priors_tensor = children_data[2].unsqueeze(0)
        
        # QUANTUM BYPASS: Use manual computation instead of batch_select_ucb_optimized
        if self.quantum_features and hasattr(self.quantum_features, 'apply_quantum_to_selection'):
            # Manual UCB computation with quantum enhancement
            visit_counts = self.tree.visit_counts[children_tensor]
            q_values = self.tree.q_values[children_tensor]
            
            # Get parent visit counts
            parent_visits = self.tree.visit_counts[active_nodes].unsqueeze(1)
            
            # Apply quantum enhancement
            ucb_scores = self.quantum_features.apply_quantum_to_selection(
                q_values=q_values,
                visit_counts=visit_counts,
                priors=priors_tensor,
                c_puct=self.config.c_puct,
                parent_visits=parent_visits
            )
            
            # Select best actions
            best_indices = ucb_scores.argmax(dim=1)
            batch_indices = torch.arange(len(active_nodes), device=self.device)
            best_children = children_tensor[batch_indices, best_indices]
            
            # Get corresponding actions
            selected_actions = actions_tensor[batch_indices, best_indices]
            
            return best_children, selected_actions
        
        else:
            # Classical: use optimized kernels
            if hasattr(self.tree, 'batch_select_ucb_optimized'):
                selected_actions, _ = self.tree.batch_select_ucb_optimized(
                    active_nodes, self.config.c_puct, 0.0
                )
                best_children = self.tree.batch_action_to_child(active_nodes, selected_actions)
                return best_children, selected_actions
            else:
                # Fallback
                return super()._select_batch_vectorized(active_nodes)

print("Quantum Bypass Pattern Test")
print("=" * 60)

evaluator = DummyEvaluator(board_size=15, device='cuda')
state = DummyState()

# Test configurations
test_configs = [
    ("Classical (optimized kernels)", False, None, MCTS),
    ("Quantum (bypass kernels)", True, 'tree_level', QuantumBypassMCTS),
]

results = []

for name, enable_quantum, quantum_level, mcts_class in test_configs:
    print(f"\n{name}:")
    print("-" * 40)
    
    # Create quantum config if needed
    quantum_config = None
    if enable_quantum:
        quantum_config = QuantumConfig(
            quantum_level=quantum_level,
            enable_quantum=True,
            min_wave_size=1,  # Always apply quantum
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
    
    mcts = mcts_class(config, evaluator)
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
    
    results.append({
        'name': name,
        'avg_time': avg_time,
        'std_time': std_time,
        'sims_per_sec': sims_per_sec,
    })
    
    print(f"Average time: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"Performance: {sims_per_sec:,.0f} sims/s")

print("\n" + "=" * 60)
print("Summary:")
print("-" * 60)

classical_speed = results[0]['sims_per_sec']
for result in results:
    speedup = result['sims_per_sec'] / classical_speed
    overhead = (1 - speedup) * 100
    print(f"{result['name']:30s}: {result['sims_per_sec']:8,.0f} sims/s ({speedup:.2f}x)")

print("\nKey Finding:")
print("-" * 60)
print("The quantum version can be faster by bypassing the 'optimized' kernels")
print("which have overhead from ensure_consistent() calls.")