#!/usr/bin/env python3
"""Test to identify where MCTS is hanging"""

import os
import sys
import time
import torch

# Disable CUDA compilation
os.environ['DISABLE_CUDA_COMPILE'] = '1'

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=== MCTS HANGING TEST ===")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: cuda" if torch.cuda.is_available() else "cpu")

print("\n1. Importing modules...")
import alphazero_py
print("   ✓ alphazero_py imported")

from mcts.core.high_performance_mcts import HighPerformanceMCTS, HighPerformanceMCTSConfig
print("   ✓ HighPerformanceMCTS imported")

from mcts.core.game_interface import GameInterface, GameType
print("   ✓ GameInterface imported")

from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
print("   ✓ ResNetEvaluator imported")

print("\n2. Creating game...")
game = alphazero_py.GomokuState()
game_interface = GameInterface(GameType.GOMOKU)
print("   ✓ Game created")

print("\n3. Creating evaluator...")
evaluator = ResNetEvaluator(
    game_type='gomoku',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
print("   ✓ Evaluator created")

print("\n4. Creating MCTS config...")
config = HighPerformanceMCTSConfig(
    num_simulations=10,  # Very small
    wave_size=2,  # Tiny wave
    c_puct=1.0,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    enable_interference=False,
    enable_phase_policy=False,
    enable_path_integral=False
)
print("   ✓ Config created")

print("\n5. Creating MCTS instance...")
mcts = HighPerformanceMCTS(config, game_interface, evaluator)
print("   ✓ MCTS created")

print("\n6. Running search...")
start = time.time()
policy = mcts.search(game)
end = time.time()

print(f"\n   ✓ Search completed in {end-start:.3f}s")
print(f"   Policy has {len(policy)} moves")
print(f"   First few moves: {list(policy.items())[:3]}")

print("\n=== TEST COMPLETE ===")