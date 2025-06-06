#!/usr/bin/env python3
"""Test path integral dtype issues"""

import torch
import numpy as np
import os

# Disable CUDA compilation for now
os.environ['DISABLE_CUDA_COMPILE'] = '1'

import alphazero_py
from mcts.core.high_performance_mcts import HighPerformanceMCTS, HighPerformanceMCTSConfig
from mcts.core.game_interface import GameInterface, GameType
from mcts.neural_networks.resnet_evaluator import ResNetEvaluator

print("Testing path integral with dtype debugging...")

# Create game
game = alphazero_py.GomokuState()
game_interface = GameInterface(GameType.GOMOKU)

# Create evaluator
device = 'cuda' if torch.cuda.is_available() else 'cpu'
evaluator = ResNetEvaluator(
    game_type='gomoku',
    device=device
)

# Configure MCTS with path integral enabled
config = HighPerformanceMCTSConfig(
    num_simulations=100,
    c_puct=1.414,
    device=device,
    wave_size=64,
    enable_path_integral=True,  # Enable path integral
    enable_interference=False,
    enable_phase_policy=False
)

# Create MCTS
mcts = HighPerformanceMCTS(config, game_interface, evaluator)

print("\nRunning MCTS search with path integral...")
try:
    policy = mcts.search(game)
    print("✓ Search completed successfully!")
    print(f"Policy: {list(policy.items())[:5]}...")  # Show first 5 actions
except Exception as e:
    print(f"✗ Error during search: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")