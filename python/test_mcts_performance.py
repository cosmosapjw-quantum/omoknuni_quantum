#!/usr/bin/env python3
"""Quick performance test for optimized MCTS"""

import torch
import numpy as np
import time
from mcts.core.high_performance_mcts import HighPerformanceMCTS, HighPerformanceMCTSConfig
from mcts.core.game_interface import GameInterface, GameType
from mcts.core.evaluator import Evaluator, EvaluatorConfig


class FastDummyEvaluator(Evaluator):
    """Fast dummy evaluator for testing"""
    def __init__(self):
        config = EvaluatorConfig(device='cpu', batch_size=512)
        super().__init__(config, action_size=225)
        
    def evaluate(self, state, legal_mask=None, temperature=1.0):
        # Uniform policy, neutral value
        policy = np.ones(self.action_size) / self.action_size
        value = 0.0
        return policy, value
    
    def evaluate_batch(self, states, legal_masks=None, temperature=1.0):
        # Fast batch evaluation
        if isinstance(states, torch.Tensor):
            batch_size = states.shape[0]
        elif isinstance(states, np.ndarray):
            batch_size = states.shape[0]
        else:
            batch_size = len(states)
            
        # Uniform policies
        policies = np.ones((batch_size, self.action_size)) / self.action_size
        values = np.zeros(batch_size)
        return policies, values


def test_mcts_performance():
    """Test MCTS performance with various configurations"""
    
    print("MCTS Performance Test")
    print("=" * 50)
    
    # Create game and evaluator
    game = GameInterface(GameType.GOMOKU)
    evaluator = FastDummyEvaluator()
    
    # Test configurations
    configs = [
        {"name": "Small (100 sims, wave=32)", "num_simulations": 100, "wave_size": 32},
        {"name": "Medium (400 sims, wave=64)", "num_simulations": 400, "wave_size": 64},
        {"name": "Large (800 sims, wave=128)", "num_simulations": 800, "wave_size": 128},
    ]
    
    for cfg in configs:
        print(f"\nTesting: {cfg['name']}")
        
        config = HighPerformanceMCTSConfig(
            num_simulations=cfg['num_simulations'],
            wave_size=cfg['wave_size'],
            enable_gpu=False,  # CPU only for now
            device='cpu',
            max_tree_size=50000,
            # Disable optional features for baseline performance
            enable_interference=False,
            enable_transposition_table=False,
            enable_phase_policy=False,
            enable_path_integral=False,
        )
        
        mcts = HighPerformanceMCTS(config, game, evaluator)
        
        # Warm up
        state = game.create_initial_state()
        mcts.search(state)
        
        # Time multiple searches
        num_searches = 5
        start = time.time()
        
        for _ in range(num_searches):
            state = game.create_initial_state()
            policy = mcts.search(state)
        
        elapsed = time.time() - start
        avg_time = elapsed / num_searches
        
        # Get stats
        stats = mcts.get_search_statistics()
        sims_per_search = cfg['num_simulations']
        sims_per_second = sims_per_search / avg_time
        
        print(f"  Average search time: {avg_time:.3f}s")
        print(f"  Simulations/second: {sims_per_second:.1f}")
        print(f"  Tree size: {stats['tree_size']} nodes")
        
        # Show policy sample
        if policy:
            top_moves = sorted(policy.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  Top 3 moves: {top_moves}")
    
    print("\n" + "=" * 50)
    print("Test complete!")


if __name__ == "__main__":
    test_mcts_performance()