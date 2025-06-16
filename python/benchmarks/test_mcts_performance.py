"""
Simple test to isolate MCTS performance issue
"""

import torch
import time
import numpy as np

from mcts.core.mcts import MCTS, MCTSConfig
from mcts.core.game_interface import GameType

class MockEvaluator:
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
    
    def evaluate_batch(self, features):
        batch_size = features.shape[0]
        policies = torch.rand(batch_size, 225, device=self.device)
        policies = torch.softmax(policies, dim=-1)
        values = torch.rand(batch_size, 1, device=self.device) * 2 - 1
        return policies, values


def test_mcts_configs():
    """Test different MCTS configurations"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_simulations = 500
    
    try:
        import alphazero_py
        state = alphazero_py.GomokuState()
    except ImportError:
        print("alphazero_py not available")
        return
    
    evaluator = MockEvaluator(device=device)
    
    configs = [
        ("Classical", {
            'enable_quantum': False,
        }),
        ("Quantum v1", {
            'enable_quantum': True,
            'quantum_version': 'v1',
        }),
        ("Quantum v2 - Small batch", {
            'enable_quantum': True,
            'quantum_version': 'v2',
            'quantum_branching_factor': 225,
            'quantum_avg_game_length': 100,
            'min_wave_size': 16,
            'max_wave_size': 16,
        }),
        ("Quantum v2 - Large batch", {
            'enable_quantum': True,
            'quantum_version': 'v2',
            'quantum_branching_factor': 225,
            'quantum_avg_game_length': 100,
            'min_wave_size': 256,
            'max_wave_size': 256,
        }),
        ("Quantum v2 - Phase adaptation off", {
            'enable_quantum': True,
            'quantum_version': 'v2',
            'quantum_branching_factor': 225,
            'quantum_avg_game_length': 100,
            'enable_phase_adaptation': False,  # Disable phase tracking
        })
    ]
    
    print(f"Testing MCTS performance ({num_simulations} simulations)")
    print("=" * 60)
    
    for name, extra_config in configs:
        # Create config - handle wave_size in extra_config
        base_config = {
            'num_simulations': num_simulations,
            'adaptive_wave_sizing': False,
            'device': device,
            'game_type': GameType.GOMOKU,
            'enable_debug_logging': False,  # Disable debug logs
        }
        # Add default wave sizes if not in extra_config
        if 'min_wave_size' not in extra_config:
            base_config['min_wave_size'] = 32
        if 'max_wave_size' not in extra_config:
            base_config['max_wave_size'] = 64
        
        # Merge configs
        base_config.update(extra_config)
        config = MCTSConfig(**base_config)
        
        # Create MCTS
        mcts = MCTS(config, evaluator)
        
        # Warmup
        mcts.search(state, num_simulations=10)
        
        # Time search
        start = time.perf_counter()
        policy = mcts.search(state, num_simulations=num_simulations)
        elapsed = time.perf_counter() - start
        
        sims_per_sec = num_simulations / elapsed
        print(f"{name:30s}: {elapsed:6.3f}s ({sims_per_sec:6.0f} sims/sec)")


def test_quantum_overhead_directly():
    """Test quantum selection overhead directly"""
    
    print("\n\nDirect quantum selection overhead test")
    print("=" * 60)
    
    from mcts.quantum import create_quantum_mcts
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 64
    num_actions = 225
    
    # Create test data
    q_values = torch.randn(batch_size, num_actions, device=device)
    visit_counts = torch.randint(0, 100, (batch_size, num_actions), device=device)
    priors = torch.softmax(torch.randn(batch_size, num_actions), dim=-1)
    
    # Test v1
    quantum_v1 = create_quantum_mcts(version='v1', device=device)
    
    # Warmup
    for _ in range(10):
        _ = quantum_v1.apply_quantum_to_selection(q_values, visit_counts, priors)
    
    # Time v1
    start = time.perf_counter()
    for _ in range(100):
        _ = quantum_v1.apply_quantum_to_selection(q_values, visit_counts, priors)
    v1_time = time.perf_counter() - start
    
    # Test v2
    quantum_v2 = create_quantum_mcts(
        version='v2',
        branching_factor=num_actions,
        device=device
    )
    
    # Warmup
    for _ in range(10):
        _ = quantum_v2.apply_quantum_to_selection(
            q_values, visit_counts, priors,
            c_puct=1.414, total_simulations=1000
        )
    
    # Time v2
    start = time.perf_counter()
    for _ in range(100):
        _ = quantum_v2.apply_quantum_to_selection(
            q_values, visit_counts, priors,
            c_puct=1.414, total_simulations=1000
        )
    v2_time = time.perf_counter() - start
    
    print(f"v1: {v1_time:.4f}s")
    print(f"v2: {v2_time:.4f}s")
    print(f"v2/v1 ratio: {v2_time/v1_time:.2f}x")


if __name__ == "__main__":
    test_mcts_configs()
    test_quantum_overhead_directly()