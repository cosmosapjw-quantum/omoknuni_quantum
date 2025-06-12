#!/usr/bin/env python3
"""Debug MCTS issues with detailed logging"""

import torch
import logging
from mcts.core.mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType
import alphazero_py

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class DebugEvaluator:
    """Simple evaluator for debugging"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.call_count = 0
        
    def evaluate_batch(self, features, legal_masks=None):
        self.call_count += 1
        print(f"[EVALUATOR] Call {self.call_count}, batch_size={features.shape[0]}")
        
        batch_size = features.shape[0]
        values = torch.zeros(batch_size, 1, device=self.device)
        policies = torch.ones(batch_size, 225, device=self.device) / 225
        return policies, values


def test_minimal():
    """Test minimal MCTS functionality"""
    print("=== Testing minimal MCTS ===")
    
    config = MCTSConfig(
        num_simulations=5,  # Very small number
        device='cuda',
        game_type=GameType.GOMOKU,
        wave_size=2,  # Small wave size
        board_size=15
    )
    
    evaluator = DebugEvaluator()
    print("Creating MCTS...")
    mcts = MCTS(config, evaluator)
    
    print("Creating game state...")
    game_state = alphazero_py.GomokuState()
    
    print("Starting search...")
    try:
        policy = mcts.search(game_state, num_simulations=5)
        print(f"Search complete! Policy sum: {policy.sum()}")
        print(f"Evaluator calls: {evaluator.call_count}")
        
        # Get stats
        stats = mcts.get_statistics()
        print(f"Tree nodes: {stats.get('tree_nodes', 'N/A')}")
        print(f"Simulations/sec: {stats.get('sims_per_second', 'N/A')}")
        
    except Exception as e:
        print(f"Error during search: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_minimal()