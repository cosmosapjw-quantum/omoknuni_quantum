#!/usr/bin/env python3
"""Test MCTS speed without neural network bottleneck"""

import os
import sys
import time
import torch
import numpy as np

# Disable CUDA compilation
os.environ['DISABLE_CUDA_COMPILE'] = '1'

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alphazero_py
from mcts.core.high_performance_mcts import HighPerformanceMCTS, HighPerformanceMCTSConfig
from mcts.core.game_interface import GameInterface, GameType
from mcts.core.evaluator import MockEvaluator

def test_pure_mcts_speed():
    """Test MCTS speed with mock evaluator"""
    print("Testing MCTS speed without neural network...")
    
    # Create game
    game = alphazero_py.GomokuState()
    game_interface = GameInterface(GameType.GOMOKU)
    
    # Create mock evaluator (no neural network)
    from mcts.core.evaluator import EvaluatorConfig
    eval_config = EvaluatorConfig(device='cpu')
    evaluator = MockEvaluator(config=eval_config, action_size=225)  # 15x15 board
    
    # Test different configurations
    configs = [
        # (num_sims, wave_size, description)
        (100, 8, "Small waves"),
        (100, 32, "Medium waves"),
        (100, 64, "Large waves"),
        (1000, 256, "Very large waves"),
        (10000, 1024, "Huge waves"),
    ]
    
    for num_sims, wave_size, desc in configs:
        print(f"\n{desc}: {num_sims} simulations, wave_size={wave_size}")
        
        config = HighPerformanceMCTSConfig(
            num_simulations=num_sims,
            wave_size=wave_size,
            c_puct=1.0,
            device='cpu',
            enable_interference=False,
            enable_phase_policy=False,
            enable_path_integral=False
        )
        
        mcts = HighPerformanceMCTS(config, game_interface, evaluator)
        
        # Warmup
        mcts.search(game)
        
        # Measure
        runs = 5 if num_sims <= 1000 else 1
        total_time = 0
        
        for _ in range(runs):
            start = time.time()
            policy = mcts.search(game)
            end = time.time()
            total_time += (end - start)
        
        avg_time = total_time / runs
        throughput = num_sims / avg_time
        
        print(f"  Average time: {avg_time:.3f}s")
        print(f"  Throughput: {throughput:.1f} sims/sec")
        
        # Estimate NN overhead
        if desc == "Medium waves":
            mock_throughput = throughput
            
    # Now test with ResNet
    print("\n" + "="*50)
    print("Comparing with ResNet evaluator...")
    
    from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
    evaluator = ResNetEvaluator(game_type='gomoku', device='cpu')
    
    config = HighPerformanceMCTSConfig(
        num_simulations=100,
        wave_size=32,
        c_puct=1.0,
        device='cpu',
        enable_interference=False,
        enable_phase_policy=False,
        enable_path_integral=False
    )
    
    mcts = HighPerformanceMCTS(config, game_interface, evaluator)
    
    start = time.time()
    policy = mcts.search(game)
    end = time.time()
    
    resnet_throughput = 100 / (end - start)
    
    print(f"\nMock evaluator: {mock_throughput:.1f} sims/sec")
    print(f"ResNet evaluator: {resnet_throughput:.1f} sims/sec")
    print(f"NN overhead: {mock_throughput/resnet_throughput:.1f}x slowdown")


if __name__ == "__main__":
    print("Successfully loaded C++ game implementations from alphazero_py module")
    test_pure_mcts_speed()