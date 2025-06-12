#!/usr/bin/env python3
"""Diagnose MCTS performance issues"""

import torch
import numpy as np
import time
from mcts.core.optimized_mcts import MCTS, MCTSConfig
from mcts.quantum.quantum_features import QuantumConfig
from mcts.gpu.gpu_game_states import GameType

# Simple dummy evaluator
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

def diagnose_mcts():
    """Diagnose MCTS performance issues"""
    print("MCTS Performance Diagnosis")
    print("=" * 60)
    
    # Create config with optimal settings
    config = MCTSConfig(
        num_simulations=1000,
        min_wave_size=3072,
        max_wave_size=3072,
        adaptive_wave_sizing=False,  # Critical for performance
        device='cuda',
        game_type=GameType.GOMOKU,
        board_size=15,
        enable_quantum=True,
        enable_debug_logging=True
    )
    
    quantum_config = QuantumConfig(
        enable_quantum=True,
        hbar_eff=0.05,
        phase_kick_strength=0.1,
        interference_alpha=0.05,
        min_wave_size=3072,
        optimal_wave_size=3072,
        device='cuda',
        fast_mode=True
    )
    config.quantum_config = quantum_config
    
    print("Configuration:")
    print(f"  Wave size: {config.max_wave_size}")
    print(f"  Adaptive wave sizing: {config.adaptive_wave_sizing}")
    print(f"  Device: {config.device}")
    print(f"  Quantum enabled: {config.enable_quantum}")
    
    # Create evaluator and MCTS
    evaluator = DummyEvaluator(board_size=15, device='cuda')
    mcts = MCTS(config, evaluator)
    mcts.optimize_for_hardware()
    
    # Check GPU ops
    print("\nGPU Operations Check:")
    if hasattr(mcts, 'gpu_ops') and mcts.gpu_ops is not None:
        print(f"  GPU ops available: True")
        print(f"  CUDA kernels loaded: {mcts.gpu_ops.use_cuda}")
        print(f"  Device: {mcts.gpu_ops.device}")
        
        # Check for quantum kernel
        if hasattr(mcts.gpu_ops, '_UNIFIED_KERNELS'):
            print(f"  _UNIFIED_KERNELS exists: True")
        else:
            print(f"  _UNIFIED_KERNELS exists: False")
            
        # Check batch_ucb_selection parameters
        import inspect
        sig = inspect.signature(mcts.gpu_ops.batch_ucb_selection)
        params = list(sig.parameters.keys())
        print(f"  batch_ucb_selection params: {params}")
        quantum_params = [p for p in params if 'quantum' in p.lower()]
        print(f"  Quantum parameters: {quantum_params}")
    else:
        print(f"  GPU ops available: False")
    
    # Check quantum features
    print("\nQuantum Features Check:")
    if mcts.quantum_features is not None:
        print(f"  Quantum features loaded: True")
        print(f"  Config: {mcts.quantum_features.config}")
        if hasattr(mcts.quantum_features, 'uncertainty_table'):
            print(f"  Uncertainty table shape: {mcts.quantum_features.uncertainty_table.shape}")
    else:
        print(f"  Quantum features loaded: False")
    
    # Create dummy state
    class DummyState:
        def __init__(self):
            self.board = np.zeros((15, 15), dtype=np.int8)
            self.current_player = 1
            self.move_count = 0
        def to_tensor(self):
            return torch.tensor(self.board, dtype=torch.int8)
    
    state = DummyState()
    
    # Run a small search
    print("\nRunning test search...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    policy = mcts.search(state, num_simulations=100)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    print(f"\nSearch completed:")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Speed: {100/elapsed:.1f} sims/s")
    
    # Check statistics
    stats = mcts.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total simulations: {stats.get('total_simulations', 0)}")
    print(f"  Tree nodes: {stats.get('tree_nodes', 0)}")
    print(f"  State pool usage: {stats.get('state_pool_usage', 0):.2%}")
    
    # Check GPU kernel usage
    if hasattr(mcts.gpu_ops, 'stats'):
        gpu_stats = mcts.gpu_ops.stats
        print(f"\nGPU Kernel Usage:")
        print(f"  UCB calls: {gpu_stats.get('ucb_calls', 0)}")
        print(f"  Quantum calls: {gpu_stats.get('quantum_calls', 0)}")
        print(f"  Backup calls: {gpu_stats.get('backup_calls', 0)}")
        print(f"  Total nodes processed: {gpu_stats.get('total_nodes_processed', 0)}")
        
        if gpu_stats.get('ucb_calls', 0) > 0:
            avg_nodes = gpu_stats['total_nodes_processed'] / gpu_stats['ucb_calls']
            print(f"  Average nodes per UCB call: {avg_nodes:.1f}")
    
    # Profile wave processing
    print("\nProfiling wave processing...")
    mcts.reset_tree()
    
    # Enable profiling
    original_profile = mcts.config.profile_gpu_kernels
    mcts.config.profile_gpu_kernels = True
    mcts.kernel_timings = {} if mcts.kernel_timings is None else mcts.kernel_timings
    
    # Run another search
    torch.cuda.synchronize()
    wave_times = []
    
    # Manually run a few waves
    mcts._initialize_root(state)
    
    for i in range(5):
        torch.cuda.synchronize()
        wave_start = time.perf_counter()
        
        mcts._run_search_wave_vectorized(config.max_wave_size)
        
        torch.cuda.synchronize()
        wave_time = time.perf_counter() - wave_start
        wave_times.append(wave_time)
        
        wave_sims_per_sec = config.max_wave_size / wave_time
        print(f"  Wave {i+1}: {wave_time:.3f}s ({wave_sims_per_sec:.0f} sims/s)")
    
    avg_wave_time = np.mean(wave_times)
    avg_wave_speed = config.max_wave_size / avg_wave_time
    print(f"\nAverage wave speed: {avg_wave_speed:.0f} sims/s")
    
    # Check tree operations
    print("\nTree operations check:")
    print(f"  Tree type: {type(mcts.tree)}")
    print(f"  Tree nodes: {mcts.tree.num_nodes}")
    print(f"  Tree edges: {mcts.tree.num_edges}")
    if hasattr(mcts.tree, 'batch_ops'):
        print(f"  Batch ops available: {mcts.tree.batch_ops is not None}")
    
    config.profile_gpu_kernels = original_profile

if __name__ == "__main__":
    diagnose_mcts()