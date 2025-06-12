#!/usr/bin/env python3
"""Detailed profiling of MCTS to identify bottlenecks"""

import torch
import time
import cProfile
import pstats
import io
from contextlib import contextmanager
from mcts.core.mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType
import alphazero_py


@contextmanager
def profile_section(name):
    """Profile a specific section of code"""
    start = time.perf_counter()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    yield
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed*1000:.2f}ms")


class InstrumentedEvaluator:
    """Evaluator that tracks timing"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.call_count = 0
        self.total_time = 0
        
    def evaluate_batch(self, features, legal_masks=None):
        start = time.perf_counter()
        batch_size = features.shape[0]
        values = torch.zeros(batch_size, 1, device=self.device)
        policies = torch.ones(batch_size, 225, device=self.device) / 225
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.total_time += time.perf_counter() - start
        self.call_count += 1
        
        return policies, values


def profile_tree_operations():
    """Profile individual tree operations"""
    print("\n=== Tree Operations Profile ===")
    
    from mcts.gpu.csr_tree import CSRTree, CSRTreeConfig
    
    config = CSRTreeConfig(device='cuda', max_nodes=0, max_edges=0)
    tree = CSRTree(config)
    
    # Add some nodes
    print("\nAdding nodes...")
    with profile_section("Add 1000 children"):
        for i in range(1000):
            tree.add_child(0, i, 1.0/1000)
    
    # Test batch operations
    node_indices = torch.arange(100, device='cuda', dtype=torch.int32)
    
    with profile_section("Batch get children (100 nodes)"):
        children, actions, priors = tree.batch_get_children(node_indices)
    
    with profile_section("Batch UCB selection (100 nodes)"):
        selected_actions, scores = tree.batch_select_ucb_optimized(node_indices[:10])
    
    print(f"\nTree stats: {tree.num_nodes} nodes, {tree.num_edges} edges")


def profile_expansion():
    """Profile the expansion process in detail"""
    print("\n=== Expansion Profile ===")
    
    config = MCTSConfig(
        num_simulations=10,
        device='cuda',
        game_type=GameType.GOMOKU,
        wave_size=256
    )
    
    evaluator = InstrumentedEvaluator()
    mcts = MCTS(config, evaluator)
    
    # Initialize with a game state
    game_state = alphazero_py.GomokuState()
    
    # Profile just the expansion
    prof = cProfile.Profile()
    prof.enable()
    
    # Do a small search to trigger expansion
    policy = mcts.search(game_state, num_simulations=10)
    
    prof.disable()
    
    # Print expansion-specific stats
    s = io.StringIO()
    ps = pstats.Stats(prof, stream=s).sort_stats('cumulative')
    ps.print_stats(lambda x: 'expand' in x or 'add_child' in x or 'item' in x)
    print(s.getvalue())
    
    print(f"\nEvaluator: {evaluator.call_count} calls, {evaluator.total_time*1000:.2f}ms total")


def profile_full_search():
    """Profile full MCTS search"""
    print("\n=== Full Search Profile ===")
    
    config = MCTSConfig(
        num_simulations=100,
        device='cuda',
        game_type=GameType.GOMOKU,
        wave_size=256
    )
    
    evaluator = InstrumentedEvaluator()
    mcts = MCTS(config, evaluator)
    game_state = alphazero_py.GomokuState()
    
    # Warm up
    mcts.search(game_state, num_simulations=10)
    
    # Profile
    prof = cProfile.Profile()
    prof.enable()
    
    start = time.perf_counter()
    policy = mcts.search(game_state, num_simulations=100)
    elapsed = time.perf_counter() - start
    
    prof.disable()
    
    print(f"\nTotal time: {elapsed:.3f}s")
    print(f"Sims/sec: {100/elapsed:.0f}")
    print(f"Policy sum: {policy.sum():.6f}")
    
    # Print top time consumers
    print("\nTop 20 functions by cumulative time:")
    s = io.StringIO()
    ps = pstats.Stats(prof, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())
    
    # Look for .item() calls specifically
    print("\n=== .item() calls analysis ===")
    s = io.StringIO()
    ps = pstats.Stats(prof, stream=s).sort_stats('cumulative')
    ps.print_stats(lambda x: 'item' in x)
    print(s.getvalue())


def main():
    print("=== Detailed MCTS Performance Profile ===")
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, running on CPU")
    
    # Profile tree operations
    profile_tree_operations()
    
    # Profile expansion in detail
    profile_expansion()
    
    # Profile full search
    profile_full_search()


if __name__ == "__main__":
    main()