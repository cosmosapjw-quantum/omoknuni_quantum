"""Test state synchronization between C++ game interface and GPU game states

This module tests the critical synchronization between C++ game states and GPU game states
to ensure legal move validation is consistent.
"""

import pytest
import torch
import numpy as np
from mcts.core.mcts import MCTS
from mcts.core.game_interface import GameInterface, GameType


class TestStateSynchronization:
    """Test state synchronization issues"""
    
    def test_position_128_legal_move_consistency(self, base_mcts_config, mock_evaluator):
        """Test that position 128 is correctly marked as illegal after being occupied
        
        This test reproduces the critical bug where position 128 is marked as legal
        when it should be occupied.
        """
        # Create game interface and initial state
        game = GameInterface(GameType.GOMOKU, board_size=15)
        initial_state = game.create_initial_state()
        
        # Apply move 128 (should occupy position 128)
        state_with_move = game.apply_move(initial_state, 128)
        
        # Initialize MCTS with this state
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Check that GPU game states have position 128 marked as occupied
        mcts._ensure_root_initialized(state_with_move)
        root_state_idx = mcts.node_to_state[0].item()
        
        # Get legal moves mask from GPU game states
        legal_mask = mcts.game_states.get_legal_moves_mask(
            torch.tensor([root_state_idx], device=mcts.device)
        )[0]
        legal_moves = torch.nonzero(legal_mask).squeeze(-1).cpu().numpy()
        
        # Position 128 should NOT be in legal moves since it's occupied
        assert 128 not in legal_moves, f"Position 128 should be occupied but is marked as legal"
        
        # Verify the board state directly
        row, col = 128 // 15, 128 % 15  # Position 128 -> (8, 8)
        board_value = mcts.game_states.boards[root_state_idx, row, col].item()
        assert board_value != 0, f"Position 128 should be occupied (non-zero) but board shows {board_value}"
        
    def test_multiple_moves_synchronization(self, base_mcts_config, mock_evaluator):
        """Test that multiple moves are properly synchronized"""
        # Create game interface and initial state
        game = GameInterface(GameType.GOMOKU, board_size=15)
        state = game.create_initial_state()
        
        # Apply sequence of moves like in the failing test
        moves = [112, 127, 113, 128, 114]
        for move in moves:
            state = game.apply_move(state, move)
        
        # Initialize MCTS with final state
        mcts = MCTS(base_mcts_config, mock_evaluator)
        mcts._ensure_root_initialized(state)
        root_state_idx = mcts.node_to_state[0].item()
        
        # Get legal moves mask
        legal_mask = mcts.game_states.get_legal_moves_mask(
            torch.tensor([root_state_idx], device=mcts.device)
        )[0]
        legal_moves = set(torch.nonzero(legal_mask).squeeze(-1).cpu().numpy())
        
        # None of the played moves should be legal anymore
        for move in moves:
            assert move not in legal_moves, f"Move {move} should be occupied but is marked as legal"
            
        # Verify board state directly
        for move in moves:
            row, col = move // 15, move % 15
            board_value = mcts.game_states.boards[root_state_idx, row, col].item()
            assert board_value != 0, f"Position {move} should be occupied but board shows {board_value}"