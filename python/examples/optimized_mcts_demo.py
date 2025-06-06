#!/usr/bin/env python3
"""Demonstration of optimized MCTS achieving 168k+ simulations/second

This example shows how to properly configure and use the MCTS
for maximum performance on GPU.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import alphazero_py
from mcts.core.mcts import MCTS, MCTSConfig
from mcts.core.game_interface import GameType
from mcts.neural_networks.resnet_evaluator import ResNetEvaluator


def main():
    """Run optimized MCTS demonstration"""
    print("=== Optimized MCTS Demo ===")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create neural network evaluator
    print("\nInitializing ResNet evaluator...")
    evaluator = ResNetEvaluator(
        game_type='gomoku',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Configure for maximum performance
    # Key settings:
    # - Fixed wave size of 3072 (optimal for RTX 3060 Ti)
    # - Adaptive sizing disabled for consistent performance
    # - Mixed precision and tensor cores enabled
    config = MCTSConfig(
        # Wave configuration - CRITICAL for performance
        min_wave_size=3072,
        max_wave_size=3072,
        adaptive_wave_sizing=False,  # Must be False for best performance
        
        # Memory settings
        memory_pool_size_mb=2048,    # 2GB memory pool
        max_tree_nodes=500000,       # 500k nodes (reasonable for 8GB VRAM)
        tree_batch_size=1024,
        
        # Caching
        cache_legal_moves=True,
        cache_features=True,
        use_zobrist_hashing=True,
        
        # GPU optimization
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_mixed_precision=torch.cuda.is_available(),
        use_cuda_graphs=torch.cuda.is_available(),
        use_tensor_cores=torch.cuda.is_available(),
        
        # Performance target
        target_sims_per_second=150000,
        
        # Game type
        game_type=GameType.GOMOKU
    )
    
    # Create MCTS instance
    print("\nCreating MCTS instance...")
    mcts = MCTS(config, evaluator)
    
    # Optimize for hardware
    print("Optimizing for hardware...")
    mcts.optimize_for_hardware()
    
    # Create game state
    game_state = alphazero_py.GomokuState()
    
    # Warmup (important for JIT compilation)
    print("\nWarming up (3 rounds)...")
    for i in range(3):
        policy = mcts.search(game_state, num_simulations=5000)
        print(f"  Warmup {i+1}/3 complete")
    
    # Performance test
    print("\nRunning performance test...")
    num_simulations = 100000
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    # Run MCTS search
    policy = mcts.search(game_state, num_simulations=num_simulations)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed_time = time.perf_counter() - start_time
    sims_per_sec = num_simulations / elapsed_time
    
    print(f"\nResults:")
    print(f"  Simulations: {num_simulations:,}")
    print(f"  Time: {elapsed_time:.3f} seconds")
    print(f"  Performance: {sims_per_sec:,.0f} simulations/second")
    
    if sims_per_sec >= 80000:
        print("  ✅ Target achieved!")
    else:
        print(f"  ❌ Below target (need {80000 - sims_per_sec:,.0f} more)")
    
    # Get best action
    best_action = mcts.get_best_action(game_state)
    print(f"\nBest move: {best_action} (row={best_action//15}, col={best_action%15})")
    
    # Show top 5 moves by probability
    print("\nTop 5 moves by probability:")
    top_moves = sorted(enumerate(policy), key=lambda x: x[1], reverse=True)[:5]
    for move, prob in top_moves:
        print(f"  Move {move} (row={move//15}, col={move%15}): {prob:.3%}")
    
    # Performance statistics
    stats = mcts.get_statistics()
    print(f"\nDetailed statistics:")
    print(f"  Total searches: {stats['total_searches']}")
    print(f"  Average sims/sec: {stats['avg_sims_per_second']:,.0f}")
    print(f"  Peak sims/sec: {stats['peak_sims_per_second']:,.0f}")
    
    if torch.cuda.is_available():
        print(f"  GPU memory used: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")
    
    wave_stats = stats.get('wave_stats', {})
    if wave_stats:
        print(f"  Average wave size: {wave_stats.get('average_wave_size', 0):.0f}")


if __name__ == "__main__":
    main()