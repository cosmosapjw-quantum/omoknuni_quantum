#!/usr/bin/env python3
"""Debug performance issues in MCTS"""

import torch
import time
import cProfile
import pstats
from mcts.core.mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType
import alphazero_py


class FastEvaluator:
    """Fast evaluator for benchmarking"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.call_count = 0
        
    def evaluate_batch(self, features, legal_masks=None):
        self.call_count += 1
        batch_size = features.shape[0]
        values = torch.zeros(batch_size, 1, device=self.device)
        policies = torch.ones(batch_size, 225, device=self.device) / 225
        return policies, values


def profile_mcts():
    """Profile MCTS to find bottlenecks"""
    config = MCTSConfig(
        num_simulations=100,
        device='cuda',
        game_type=GameType.GOMOKU,
        wave_size=256
    )
    
    evaluator = FastEvaluator()
    mcts = MCTS(config, evaluator)
    game_state = alphazero_py.GomokuState()
    
    # Profile the search
    profiler = cProfile.Profile()
    profiler.enable()
    
    start = time.time()
    policy = mcts.search(game_state, num_simulations=100)
    elapsed = time.time() - start
    
    profiler.disable()
    
    print(f"Time: {elapsed:.3f}s")
    print(f"Sims/sec: {100/elapsed:.0f}")
    print(f"Evaluator calls: {evaluator.call_count}")
    print(f"Policy sum: {policy.sum()}")
    
    # Print top functions by time
    print("\nTop 10 functions by cumulative time:")
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
    
    return policy


def test_components():
    """Test individual components"""
    print("Testing individual components...")
    
    # Test GPU state operations
    from mcts.gpu.gpu_game_states import GPUGameStates, GPUGameStatesConfig
    
    config = GPUGameStatesConfig(
        capacity=10000,
        game_type=GameType.GOMOKU,
        device='cuda'
    )
    
    states = GPUGameStates(config)
    
    # Time state allocation
    start = time.time()
    indices = states.allocate_states(1000)
    print(f"Allocate 1000 states: {(time.time()-start)*1000:.1f}ms")
    
    # Time legal move generation
    start = time.time()
    legal_masks = states.get_legal_moves_mask(indices[:100])
    print(f"Get legal moves for 100 states: {(time.time()-start)*1000:.1f}ms")
    
    # Test tree operations
    from mcts.gpu.csr_tree import CSRTree, CSRTreeConfig
    
    tree_config = CSRTreeConfig(device='cuda')
    tree = CSRTree(tree_config)
    
    # Time batch UCB selection
    node_indices = torch.arange(100, device='cuda', dtype=torch.int32)
    start = time.time()
    actions, scores = tree.batch_select_ucb_optimized(node_indices, c_puct=1.4)
    print(f"Batch UCB selection for 100 nodes: {(time.time()-start)*1000:.1f}ms")


if __name__ == "__main__":
    print("=== MCTS Performance Debug ===\n")
    
    # Test components first
    test_components()
    
    print("\n=== Full MCTS Profile ===\n")
    
    # Profile full MCTS
    policy = profile_mcts()
    
    # Check policy
    print(f"\nFinal policy sum: {policy.sum()}")
    print(f"Non-zero entries: {(policy > 0).sum()}")
    print(f"Max policy value: {policy.max()}")