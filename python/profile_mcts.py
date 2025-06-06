#\!/usr/bin/env python3
"""Profile MCTS performance to identify bottlenecks"""

import torch
import numpy as np
import time
import cProfile
import pstats
from io import StringIO

# Add parent directory to path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import game implementations
import alphazero_py

# Import MCTS implementations
from mcts.core.high_performance_mcts import HighPerformanceMCTS, HighPerformanceMCTSConfig
from mcts.core.game_interface import GameInterface, GameType
from mcts.neural_networks.resnet_evaluator import ResNetEvaluator


def profile_mcts():
    """Profile MCTS performance"""
    print("Setting up MCTS...")
    
    # Create game
    game = alphazero_py.GomokuState()
    game_interface = GameInterface(GameType.GOMOKU)
    
    # Create evaluator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluator = ResNetEvaluator(
        game_type='gomoku',
        device=device
    )
    
    # Configure MCTS with large wave size
    config = HighPerformanceMCTSConfig(
        num_simulations=1000,
        c_puct=1.414,
        device=device,
        wave_size=1000,  # Full batch
        enable_interference=False,  # Disable quantum features for baseline
        enable_phase_kicks=False,
        enable_path_integral=False
    )
    
    # Create MCTS
    mcts = HighPerformanceMCTS(config, game_interface, evaluator)
    
    print("\nWarming up...")
    # Warmup
    for _ in range(3):
        mcts.search(game)
    
    print("\nProfiling MCTS search...")
    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.perf_counter()
    for i in range(5):
        policy = mcts.search(game)
        print(f"  Search {i+1}/5 completed")
    
    end_time = time.perf_counter()
    profiler.disable()
    
    # Print timing
    total_time = end_time - start_time
    total_sims = 5 * 1000
    print(f"\nTotal time: {total_time:.3f}s")
    print(f"Simulations per second: {total_sims / total_time:.1f}")
    
    # Print profile
    print("\nTop 20 time-consuming functions:")
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())
    
    # Also check GPU memory usage
    if torch.cuda.is_available():
        print(f"\nGPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


if __name__ == "__main__":
    profile_mcts()
EOF < /dev/null
