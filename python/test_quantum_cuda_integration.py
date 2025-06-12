#!/usr/bin/env python3
"""Test quantum CUDA kernel integration with optimized MCTS"""

import torch
import numpy as np
import time
from mcts.core.optimized_mcts import MCTS, MCTSConfig
# Create a simple dummy evaluator
class DummyEvaluator:
    def __init__(self, board_size=15, device='cuda'):
        self.board_size = board_size
        self.device = torch.device(device)
        self.num_actions = board_size * board_size
    
    def evaluate_batch(self, features):
        batch_size = features.shape[0]
        # Return uniform policy and neutral values
        policies = torch.ones((batch_size, self.num_actions), device=self.device) / self.num_actions
        values = torch.zeros((batch_size, 1), device=self.device)
        return policies, values
from mcts.quantum.quantum_features import QuantumConfig

def test_quantum_cuda_integration():
    """Test that quantum CUDA kernels work correctly"""
    print("Testing Quantum CUDA Kernel Integration")
    print("=" * 60)
    
    # Create configurations
    config = MCTSConfig(
        num_simulations=1000,
        min_wave_size=512,
        max_wave_size=512,
        adaptive_wave_sizing=False,
        device='cuda',
        game_type='GOMOKU',
        board_size=15,
        enable_quantum=True,
        enable_debug_logging=True
    )
    
    # Create quantum config
    quantum_config = QuantumConfig(
        enable_quantum=True,
        hbar_eff=0.05,
        phase_kick_strength=0.1,
        interference_alpha=0.05,
        min_wave_size=512,
        optimal_wave_size=512,
        device='cuda',
        fast_mode=True
    )
    
    config.quantum_config = quantum_config
    
    # Create evaluator
    evaluator = DummyEvaluator(board_size=15, device='cuda')
    
    # Create MCTS instance
    print("Creating MCTS with quantum features enabled...")
    mcts = MCTS(config, evaluator)
    
    # Create a dummy game state
    class DummyState:
        def __init__(self):
            self.board = np.zeros((15, 15), dtype=np.int8)
            self.current_player = 1
            self.move_count = 0
            
        def to_tensor(self):
            return torch.tensor(self.board, dtype=torch.int8)
    
    state = DummyState()
    
    # Check if quantum CUDA kernels are available
    print("\nChecking quantum kernel availability:")
    
    # Try to load quantum CUDA extension
    try:
        from mcts.gpu.quantum_cuda_extension import load_quantum_cuda_kernels, _QUANTUM_CUDA_AVAILABLE
        loaded = load_quantum_cuda_kernels()
        print(f"  Quantum CUDA kernels loaded: {loaded}")
        print(f"  Quantum CUDA available: {_QUANTUM_CUDA_AVAILABLE}")
    except Exception as e:
        print(f"  Failed to load quantum CUDA extension: {e}")
    
    # Check unified kernels
    if hasattr(mcts, 'gpu_ops') and mcts.gpu_ops is not None:
        print(f"  GPU ops available: True")
        print(f"  CUDA kernels loaded: {mcts.gpu_ops.use_cuda}")
        
        # Check for quantum kernel function
        if hasattr(mcts.gpu_ops, '_UNIFIED_KERNELS'):
            kernels = mcts.gpu_ops._UNIFIED_KERNELS
            if kernels and hasattr(kernels, 'batched_ucb_selection_quantum'):
                print(f"  Quantum UCB kernel available: True")
            else:
                print(f"  Quantum UCB kernel available: False")
    
    # Run a small search to test integration
    print("\nRunning test search with quantum features...")
    start_time = time.perf_counter()
    
    try:
        policy = mcts.search(state, num_simulations=100)
        elapsed = time.perf_counter() - start_time
        
        print(f"\nSearch completed successfully!")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Simulations/second: {100/elapsed:.1f}")
        
        # Check statistics
        stats = mcts.get_statistics()
        print(f"\nStatistics:")
        print(f"  Total simulations: {stats.get('total_simulations', 0)}")
        print(f"  Tree nodes: {stats.get('tree_nodes', 0)}")
        
        # Check if quantum kernels were used
        if hasattr(mcts.gpu_ops, 'stats'):
            gpu_stats = mcts.gpu_ops.stats
            print(f"\nGPU kernel usage:")
            print(f"  UCB calls: {gpu_stats.get('ucb_calls', 0)}")
            print(f"  Quantum calls: {gpu_stats.get('quantum_calls', 0)}")
            
            if gpu_stats.get('quantum_calls', 0) > 0:
                print(f"\n✓ Quantum CUDA kernels successfully used!")
            else:
                print(f"\n✗ Quantum CUDA kernels not used (may need compilation)")
        
    except Exception as e:
        print(f"\nError during search: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_quantum_cuda_integration()