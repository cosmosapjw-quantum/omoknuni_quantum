#!/usr/bin/env python3
"""Basic MCTS performance test"""

import time
import torch
import numpy as np
from mcts.core.high_performance_mcts import HighPerformanceMCTS, HighPerformanceMCTSConfig
from mcts.core.game_interface import GameInterface, GameType
from mcts.core.evaluator import Evaluator, EvaluatorConfig


class FastEvaluator(Evaluator):
    def __init__(self):
        config = EvaluatorConfig(device='cpu', batch_size=512)
        super().__init__(config, action_size=225)
        # Pre-generate uniform policy
        self.uniform_policy = np.ones(self.action_size) / self.action_size
        
    def evaluate(self, state, legal_mask=None, temperature=1.0):
        return self.uniform_policy.copy(), 0.0
    
    def evaluate_batch(self, states, legal_masks=None, temperature=1.0):
        if isinstance(states, torch.Tensor):
            batch_size = states.shape[0]
        elif isinstance(states, np.ndarray):
            batch_size = states.shape[0]
        else:
            batch_size = len(states)
        
        policies = np.tile(self.uniform_policy, (batch_size, 1))
        values = np.zeros(batch_size)
        return policies, values


# Create game and evaluator
print("Setting up MCTS...")
game = GameInterface(GameType.GOMOKU)
evaluator = FastEvaluator()

# Very small config for debugging
config = HighPerformanceMCTSConfig(
    num_simulations=16,  # Very small
    wave_size=4,         # Tiny waves
    enable_gpu=False,
    device='cpu',
    max_tree_size=1000,
    # Disable all optional features
    enable_interference=False,
    enable_transposition_table=False,
    enable_phase_policy=False,
    enable_path_integral=False,
)

mcts = HighPerformanceMCTS(config, game, evaluator)
state = game.create_initial_state()

print(f"\nRunning search with {config.num_simulations} simulations...")
start = time.time()

try:
    policy = mcts.search(state)
    elapsed = time.time() - start
    
    print(f"\nSearch completed in {elapsed:.3f} seconds")
    print(f"Simulations per second: {config.num_simulations / elapsed:.1f}")
    
    if policy:
        print(f"\nPolicy has {len(policy)} moves")
        top_5 = sorted(policy.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"Top 5 moves: {top_5}")
        
    stats = mcts.get_search_statistics()
    print(f"\nTree size: {stats['tree_size']} nodes")
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    
print("\nDone!")