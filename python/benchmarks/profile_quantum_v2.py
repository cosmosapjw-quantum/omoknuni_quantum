"""
Profile Quantum MCTS v2.0 to identify performance bottlenecks
"""

import torch
import numpy as np
import time
import cProfile
import pstats
from io import StringIO

from mcts.quantum import create_quantum_mcts, UnifiedQuantumConfig
from mcts.core.mcts import MCTS, MCTSConfig
from mcts.core.game_interface import GameType


class MockEvaluator:
    """Simple mock evaluator for profiling"""
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
    
    def evaluate_batch(self, features: torch.Tensor):
        batch_size = features.shape[0]
        board_size_sq = 225  # 15x15
        policies = torch.rand(batch_size, board_size_sq, device=self.device)
        policies = torch.softmax(policies, dim=-1)
        values = torch.rand(batch_size, 1, device=self.device) * 2 - 1
        return policies, values


def profile_quantum_selection():
    """Profile just the quantum selection"""
    print("=== Profiling Quantum Selection ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 256
    num_actions = 50
    num_calls = 1000
    
    # Create test data
    q_values = torch.randn(batch_size, num_actions, device=device)
    visit_counts = torch.randint(0, 100, (batch_size, num_actions), device=device)
    priors = torch.softmax(torch.randn(batch_size, num_actions, device=device), dim=-1)
    
    # Create v2 instance
    quantum_v2 = create_quantum_mcts(
        version='v2',
        branching_factor=num_actions,
        device=device,
        enable_quantum=True
    )
    
    # Warmup
    for _ in range(10):
        _ = quantum_v2.apply_quantum_to_selection(
            q_values, visit_counts, priors,
            c_puct=1.414, total_simulations=1000
        )
    
    # Profile
    pr = cProfile.Profile()
    pr.enable()
    
    start = time.perf_counter()
    for i in range(num_calls):
        _ = quantum_v2.apply_quantum_to_selection(
            q_values, visit_counts, priors,
            c_puct=1.414, total_simulations=1000 + i
        )
    end = time.perf_counter()
    
    pr.disable()
    
    # Print results
    print(f"\nTotal time: {end - start:.4f}s")
    print(f"Time per call: {(end - start) / num_calls * 1000:.2f}ms")
    print(f"Calls per second: {num_calls / (end - start):.0f}")
    
    # Show top functions
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print("\nTop 20 functions by cumulative time:")
    print(s.getvalue())


def profile_mcts_search():
    """Profile full MCTS search"""
    print("\n=== Profiling MCTS Search ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_simulations = 100
    
    try:
        import alphazero_py
        state = alphazero_py.GomokuState()
    except ImportError:
        print("alphazero_py not available")
        return
    
    evaluator = MockEvaluator(device=device)
    
    # Create v2 MCTS
    config = MCTSConfig(
        num_simulations=num_simulations,
        enable_quantum=True,
        quantum_version='v2',
        quantum_branching_factor=225,
        quantum_avg_game_length=100,
        min_wave_size=32,
        max_wave_size=64,
        adaptive_wave_sizing=False,
        device=device,
        game_type=GameType.GOMOKU
    )
    mcts = MCTS(config, evaluator)
    
    # Warmup
    mcts.search(state, num_simulations=10)
    
    # Profile
    pr = cProfile.Profile()
    pr.enable()
    
    start = time.perf_counter()
    policy = mcts.search(state, num_simulations=num_simulations)
    end = time.perf_counter()
    
    pr.disable()
    
    # Print results
    print(f"\nTotal time: {end - start:.4f}s")
    print(f"Simulations per second: {num_simulations / (end - start):.0f}")
    
    # Show top functions
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print("\nTop 30 functions by cumulative time:")
    print(s.getvalue())


def compare_phase_overhead():
    """Compare overhead in different phases"""
    print("\n=== Comparing Phase Overhead ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 256
    num_actions = 50
    num_calls = 100
    
    # Create test data
    q_values = torch.randn(batch_size, num_actions, device=device)
    visit_counts = torch.randint(0, 100, (batch_size, num_actions), device=device)
    priors = torch.softmax(torch.randn(batch_size, num_actions, device=device), dim=-1)
    
    # Create v2 instance
    quantum_v2 = create_quantum_mcts(
        version='v2',
        branching_factor=num_actions,
        device=device,
        enable_quantum=True
    )
    
    # Test different simulation counts (phases)
    test_counts = [10, 100, 1000, 10000, 50000]
    
    for N in test_counts:
        # Time it
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.perf_counter()
        
        for _ in range(num_calls):
            _ = quantum_v2.apply_quantum_to_selection(
                q_values, visit_counts, priors,
                c_puct=1.414, total_simulations=N
            )
        
        torch.cuda.synchronize() if device == 'cuda' else None
        elapsed = time.perf_counter() - start
        
        # Get phase info
        phase_info = quantum_v2.get_phase_info()
        
        print(f"\nN={N:6d}: {elapsed*1000/num_calls:6.2f}ms/call, "
              f"Phase: {phase_info['current_phase']:10s}")


if __name__ == "__main__":
    profile_quantum_selection()
    compare_phase_overhead()
    profile_mcts_search()