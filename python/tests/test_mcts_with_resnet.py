#!/usr/bin/env python3
"""Test MCTS with actual ResNet model for realistic performance"""

import pytest
import torch
import numpy as np
import time
import logging
from mcts.core.mcts import MCTS
from mcts.core.mcts_config import MCTSConfig
from mcts.gpu.gpu_game_states import GameType
from mcts.neural_networks.resnet_model import create_resnet_for_game
from mcts.neural_networks.resnet_evaluator import ResNetEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_resnet_evaluator(board_size, device='cuda', game_type='gomoku'):
    """Create actual ResNet evaluator"""
    # Create ResNet model
    model = create_resnet_for_game(
        game_type=game_type,
        input_channels=19,  # Standard AlphaZero features
        num_blocks=5,       # Smaller for faster testing
        num_filters=64      # Smaller for faster testing
    )
    
    # Create ResNetEvaluator which wraps the model
    evaluator = ResNetEvaluator(
        model=model,
        game_type=game_type,
        device=device
    )
    
    return evaluator

@pytest.mark.parametrize('backend', ['cpu', 'gpu', 'hybrid'])
def test_realistic_performance(backend):
    """Test MCTS performance with real ResNet model"""
    
    # Skip GPU tests if CUDA not available
    if backend in ['gpu', 'hybrid'] and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = 'cuda' if backend in ['gpu', 'hybrid'] and torch.cuda.is_available() else 'cpu'
    
    config = MCTSConfig(
        num_simulations=100,
        c_puct=1.0,
        board_size=15,  # Standard gomoku board size
        game_type=GameType.GOMOKU,
        device=device,
        backend=backend,
        max_tree_nodes=50000,
        batch_size=32,  # Optimize for GPU
        use_mixed_precision=True  # Use FP16 for better GPU performance
    )
    
    # Create real ResNet evaluator
    evaluator = create_resnet_evaluator(config.board_size, device, 'gomoku')
    
    # Create MCTS
    mcts = MCTS(config, evaluator)
    
    # Initial state
    state = np.zeros((config.board_size, config.board_size), dtype=np.int8)
    
    # Warmup
    logger.info(f"Warming up {backend} backend...")
    for _ in range(3):
        mcts._reset_for_new_search()
        mcts.search(state, num_simulations=50)
    
    # Benchmark
    num_searches = 5
    num_simulations = 100
    
    logger.info(f"Benchmarking {backend} backend with ResNet...")
    start_time = time.perf_counter()
    
    for i in range(num_searches):
        mcts._reset_for_new_search()
        policy = mcts.search(state, num_simulations=num_simulations)
        
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    total_simulations = num_searches * num_simulations
    sims_per_second = total_simulations / total_time
    
    logger.info(f"Backend: {backend}")
    logger.info(f"Total simulations: {total_simulations}")
    logger.info(f"Total time: {total_time:.3f}s")
    logger.info(f"Simulations/second: {sims_per_second:.0f}")
    
    # More realistic thresholds with actual neural network
    min_sims_per_second = {
        'cpu': 50,      # CPU with neural network is much slower
        'gpu': 500,     # GPU should be faster but still limited by NN
        'hybrid': 200   # Hybrid balances CPU tree and GPU NN
    }
    
    if backend in min_sims_per_second:
        assert sims_per_second >= min_sims_per_second[backend], \
            f"{backend} backend too slow: {sims_per_second:.0f} < {min_sims_per_second[backend]}"

def test_full_game_with_resnet():
    """Test playing a full game with ResNet"""
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    config = MCTSConfig(
        num_simulations=100,
        c_puct=1.0,
        board_size=15,  # Standard gomoku board size
        game_type=GameType.GOMOKU,
        device='cuda',
        backend='hybrid',  # Best for real games
        max_tree_nodes=50000,
        enable_subtree_reuse=True,
        temperature=1.0
    )
    
    evaluator = create_resnet_evaluator(config.board_size, 'cuda')
    mcts = MCTS(config, evaluator)
    
    # Play a short game
    state = np.zeros((config.board_size, config.board_size), dtype=np.int8)
    current_player = 1
    move_count = 0
    
    logger.info("Playing game with ResNet...")
    
    while move_count < 10:  # Play 10 moves
        # Get policy
        policy = mcts.search(state, num_simulations=50)
        
        # Select move
        valid_moves = (state.flatten() == 0)
        policy_masked = policy * valid_moves
        
        if policy_masked.sum() > 0:
            policy_masked /= policy_masked.sum()
            move = np.random.choice(len(policy_masked), p=policy_masked)
            
            # Apply move
            row, col = move // config.board_size, move % config.board_size
            state[row, col] = current_player
            
            # Update MCTS root
            if config.enable_subtree_reuse:
                mcts.update_root(move)
            
            current_player = -current_player
            move_count += 1
        else:
            break
    
    logger.info(f"Game ended after {move_count} moves")
    assert move_count > 0, "No moves were made"

if __name__ == "__main__":
    # Run performance tests
    for backend in ['cpu', 'gpu', 'hybrid']:
        if backend in ['gpu', 'hybrid'] and not torch.cuda.is_available():
            print(f"Skipping {backend} - no CUDA")
            continue
        test_realistic_performance(backend)
    
    # Run game test
    if torch.cuda.is_available():
        test_full_game_with_resnet()
    else:
        print("Skipping game test - no CUDA")