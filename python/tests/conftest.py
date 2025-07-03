"""Pytest configuration and fixtures for MCTS testing"""

import pytest
import torch
import numpy as np
from typing import Dict, Any, List, Optional

# Import core components
from mcts.core import GameInterface, GameType, Evaluator, MockEvaluator, AlphaZeroEvaluator
from mcts.core import MCTS, MCTSConfig
from mcts.utils import ConfigManager, AlphaZeroConfig, create_default_config


@pytest.fixture(scope="session")
def device():
    """Get test device (CPU for CI/CD compatibility)"""
    return torch.device('cpu')


@pytest.fixture
def gomoku_config():
    """Standard Gomoku configuration for testing"""
    return MCTSConfig(
        num_simulations=100,
        c_puct=1.414,
        temperature=1.0,
        classical_only_mode=True,  # Disable quantum for unit tests
        game_type=GameType.GOMOKU,
        board_size=15,
        device='cpu'
    )


@pytest.fixture
def small_gomoku_config():
    """Small Gomoku configuration for fast testing"""
    return MCTSConfig(
        num_simulations=10,
        c_puct=1.0,
        temperature=1.0,
        classical_only_mode=True,
        game_type=GameType.GOMOKU,
        board_size=9,  # Smaller board for faster tests
        device='cpu'
    )


@pytest.fixture
def game_interface():
    """Create a standard game interface for testing"""
    return GameInterface(game_type=GameType.GOMOKU, board_size=15)


@pytest.fixture
def small_game_interface():
    """Create a small game interface for fast testing"""
    return GameInterface(game_type=GameType.GOMOKU, board_size=9)


@pytest.fixture
def mock_evaluator():
    """Create a mock evaluator for testing"""
    return MockEvaluator()


@pytest.fixture
def sample_game_state():
    """Create a sample game state for testing"""
    # 15x15 Gomoku board with some moves
    state = np.zeros((3, 15, 15), dtype=np.float32)
    
    # Add some sample moves
    state[0, 7, 7] = 1.0  # Player 1 move
    state[1, 7, 8] = 1.0  # Player 2 move
    state[0, 8, 7] = 1.0  # Player 1 move
    state[1, 8, 8] = 1.0  # Player 2 move
    
    # Turn indicator (Player 1's turn)
    state[2, :, :] = 1.0
    
    return state


@pytest.fixture
def sample_small_game_state():
    """Create a sample small game state for testing"""
    # 9x9 Gomoku board with some moves
    state = np.zeros((3, 9, 9), dtype=np.float32)
    
    # Add some sample moves
    state[0, 4, 4] = 1.0  # Player 1 move
    state[1, 4, 5] = 1.0  # Player 2 move
    state[0, 5, 4] = 1.0  # Player 1 move
    
    # Turn indicator (Player 2's turn)
    state[2, :, :] = 0.0
    
    return state


@pytest.fixture
def sample_legal_moves():
    """Create sample legal moves for testing"""
    # For 15x15 board, most moves are legal except occupied positions
    legal_moves = np.ones(225, dtype=bool)
    
    # Mark some positions as occupied
    legal_moves[7*15 + 7] = False  # (7,7)
    legal_moves[7*15 + 8] = False  # (7,8)
    legal_moves[8*15 + 7] = False  # (8,7)
    legal_moves[8*15 + 8] = False  # (8,8)
    
    return legal_moves


@pytest.fixture
def sample_small_legal_moves():
    """Create sample legal moves for small board testing"""
    # For 9x9 board
    legal_moves = np.ones(81, dtype=bool)
    
    # Mark some positions as occupied
    legal_moves[4*9 + 4] = False  # (4,4)
    legal_moves[4*9 + 5] = False  # (4,5)
    legal_moves[5*9 + 4] = False  # (5,4)
    
    return legal_moves


@pytest.fixture
def batch_game_states():
    """Create a batch of game states for testing"""
    batch_size = 4
    states = np.zeros((batch_size, 3, 15, 15), dtype=np.float32)
    
    for i in range(batch_size):
        # Add different patterns for each state
        states[i, 0, i+5, i+5] = 1.0  # Player 1 move
        states[i, 1, i+5, i+6] = 1.0  # Player 2 move
        states[i, 2, :, :] = i % 2     # Alternating turns
    
    return states


@pytest.fixture
def alphazero_config():
    """Create AlphaZero configuration for testing"""
    return AlphaZeroConfig(
        game_type='gomoku',
        board_size=15,
        num_simulations=100,
        batch_size=32,
        lr=0.001,
        device='cpu'
    )


@pytest.fixture
def config_manager():
    """Create a config manager for testing"""
    return ConfigManager()


# Test data fixtures
@pytest.fixture
def winning_state():
    """Create a winning game state"""
    state = np.zeros((3, 15, 15), dtype=np.float32)
    
    # Create 5 in a row for Player 1
    for i in range(5):
        state[0, 7, 7+i] = 1.0
    
    # Turn indicator
    state[2, :, :] = 1.0
    
    return state


@pytest.fixture
def near_winning_state():
    """Create a state where Player 1 can win in one move"""
    state = np.zeros((3, 15, 15), dtype=np.float32)
    
    # Create 4 in a row for Player 1, missing one
    for i in range(4):
        state[0, 7, 7+i] = 1.0
    # Position (7, 11) would complete the 5-in-a-row
    
    # Add some Player 2 moves
    state[1, 8, 7] = 1.0
    state[1, 8, 8] = 1.0
    
    # Turn indicator (Player 1's turn)
    state[2, :, :] = 1.0
    
    return state


# Performance test fixtures
@pytest.fixture
def performance_config():
    """Configuration for performance testing"""
    return MCTSConfig(
        num_simulations=1000,
        c_puct=1.414,
        temperature=1.0,
        classical_only_mode=True,
        device='cpu'
    )


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_cuda():
    """Cleanup CUDA context after each test"""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds for reproducible tests"""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)