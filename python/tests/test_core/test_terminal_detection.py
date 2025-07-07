"""Test terminal position detection in MCTS

This module tests that MCTS properly detects terminal positions and avoids 
excessive tree expansion, following TDD principles.
"""

import pytest
import torch
import numpy as np
from mcts.core.mcts import MCTS
from mcts.core.game_interface import GameInterface, GameType


def assert_valid_policy(policy):
    """Assert that a policy is valid"""
    assert isinstance(policy, np.ndarray)
    assert len(policy.shape) == 1
    assert np.isclose(policy.sum(), 1.0, atol=1e-6)
    assert np.all(policy >= 0)


class TestTerminalDetection:
    """Test terminal position detection issues"""
    
    def test_terminal_position_creates_minimal_tree(self, base_mcts_config, mock_evaluator):
        """Test that terminal positions result in minimal tree expansion
        
        This test reproduces the issue where a terminal position expands to 21 nodes
        when it should create very few nodes.
        """
        # Create game interface and terminal state
        game = GameInterface(GameType.GOMOKU, board_size=15)
        state = game.create_initial_state()
        
        # Play winning sequence for player 1 (horizontal line)
        moves = [112, 127, 113, 128, 114, 129, 115, 130, 116]  # Horizontal line
        for move in moves:
            state = game.apply_move(state, move)
            
        # Verify this is actually a terminal state
        assert state.is_terminal(), "State should be terminal after winning sequence"
        game_result = state.get_game_result()
        print(f"DEBUG: Game result: {game_result}, type: {type(game_result)}")
        # Game result should not be ONGOING (0)
        assert hasattr(game_result, 'value') and game_result.value != 0, f"Expected non-ongoing game result, got {game_result}"
        
        # Initialize MCTS
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Search on terminal position
        policy = mcts.search(state, num_simulations=10)
        assert_valid_policy(policy)
        
        # Debug: Print tree state
        print(f"DEBUG: Tree has {mcts.tree.num_nodes} nodes after search on terminal position")
        print(f"DEBUG: Root children count: {len(mcts.tree.get_children(0)[0])}")
        
        # Tree should not expand much from terminal - this is the failing assertion
        assert mcts.tree.num_nodes < 5, f"Expected <5 nodes for terminal position, got {mcts.tree.num_nodes}"
        
    def test_non_terminal_position_expands_normally(self, base_mcts_config, mock_evaluator):
        """Test that non-terminal positions expand normally for comparison"""
        # Create game interface and non-terminal state
        game = GameInterface(GameType.GOMOKU, board_size=15)
        state = game.create_initial_state()
        
        # Play a few moves but don't create terminal state
        moves = [112, 127, 113]  # Just a few moves
        for move in moves:
            state = game.apply_move(state, move)
            
        # Verify this is NOT terminal
        assert not state.is_terminal(), "State should not be terminal"
        
        # Initialize MCTS
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Search on non-terminal position
        policy = mcts.search(state, num_simulations=10)
        assert_valid_policy(policy)
        
        # Tree should expand normally
        print(f"DEBUG: Non-terminal tree has {mcts.tree.num_nodes} nodes")
        assert mcts.tree.num_nodes >= 5, f"Non-terminal should expand more, got {mcts.tree.num_nodes}"
        
    def test_terminal_state_detection_methods(self, base_mcts_config, mock_evaluator):
        """Test that terminal state detection methods work correctly"""
        # Create game interface and states
        game = GameInterface(GameType.GOMOKU, board_size=15)
        
        # Test initial state (not terminal)
        initial_state = game.create_initial_state()
        assert not initial_state.is_terminal(), "Initial state should not be terminal"
        initial_result = initial_state.get_game_result()
        assert hasattr(initial_result, 'value') and initial_result.value == 0, "Initial state should be ongoing"
        
        # Test terminal state
        terminal_state = initial_state
        moves = [112, 127, 113, 128, 114, 129, 115, 130, 116]  # Horizontal line
        for move in moves:
            terminal_state = game.apply_move(terminal_state, move)
            
        assert terminal_state.is_terminal(), "Final state should be terminal"
        game_result = terminal_state.get_game_result()
        assert hasattr(game_result, 'value') and game_result.value != 0, f"Terminal state should not be ongoing, got {game_result}"