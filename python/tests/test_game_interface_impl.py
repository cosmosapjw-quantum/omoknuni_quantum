"""Tests for game_interface implementations"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from mcts.core.game_interface import GameInterface, GameType

# Mock the C++ bindings for testing
class MockGomokuState:
    """Mock Gomoku state for testing"""
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int32)
        self.current_player = 1
        self.move_history = []
        self.last_move = -1
        
    def clone(self):
        new_state = MockGomokuState(self.board_size)
        new_state.board = self.board.copy()
        new_state.current_player = self.current_player
        new_state.move_history = self.move_history.copy()
        new_state.last_move = self.last_move
        return new_state
        
    def make_move(self, move):
        row = move // self.board_size
        col = move % self.board_size
        self.board[row, col] = self.current_player
        self.move_history.append(move)
        self.last_move = move
        self.current_player = 3 - self.current_player  # Switch between 1 and 2
        
    def get_legal_moves(self):
        legal = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    legal.append(i * self.board_size + j)
        return legal
        
    def is_terminal(self):
        return len(self.get_legal_moves()) == 0
        
    def get_tensor_representation(self):
        # Standard 3-channel representation
        tensor = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        tensor[0] = (self.board == 1).astype(np.float32)  # Player 1 pieces
        tensor[1] = (self.board == 2).astype(np.float32)  # Player 2 pieces
        tensor[2] = self.current_player - 1  # Current player (0 or 1)
        return tensor
        
    def get_basic_tensor_representation(self):
        # 18-channel representation (without attack/defense)
        tensor = np.zeros((18, self.board_size, self.board_size), dtype=np.float32)
        # Channel 0: Current board
        board_sum = (self.board == 1).astype(np.float32) - (self.board == 2).astype(np.float32)
        tensor[0] = board_sum
        # Channel 1: Current player
        tensor[1] = np.full((self.board_size, self.board_size), self.current_player, dtype=np.float32)
        # Channels 2-17: Move history (dummy for now)
        return tensor
        
    def get_enhanced_tensor_representation(self):
        # 20-channel representation (with attack/defense)
        tensor = np.zeros((20, self.board_size, self.board_size), dtype=np.float32)
        # Copy basic channels
        basic = self.get_basic_tensor_representation()
        tensor[:18] = basic
        # Channels 18-19: Attack/defense (dummy for now)
        return tensor
        
    def get_current_player(self):
        return self.current_player
        
    def get_winner(self):
        # Simple winner check (dummy implementation)
        if self.is_terminal():
            return 1  # Dummy winner
        return -1  # No winner yet


class TestTensorToState:
    """Test tensor_to_state implementation"""
    
    @patch('mcts.core.game_interface.alphazero_py')
    def test_tensor_to_state_basic_reconstruction(self, mock_alphazero):
        """Test basic reconstruction of game state from tensor"""
        # Setup mock
        mock_alphazero.GomokuState = MockGomokuState
        mock_alphazero.GameType.GOMOKU = 2
        
        # Create interface
        interface = GameInterface(GameType.GOMOKU, board_size=15)
        
        # Create a state with some moves
        state = MockGomokuState(15)
        state.make_move(112)  # Center move (7, 7)
        state.make_move(113)  # Adjacent move (7, 8)
        state.make_move(127)  # Another move (8, 7)
        
        # Convert to tensor
        tensor = interface.state_to_tensor(state, representation_type='standard')
        
        # Convert back to state
        reconstructed = interface.tensor_to_state(tensor)
        
        # Verify board positions match
        original_tensor = state.get_tensor_representation()
        reconstructed_tensor = reconstructed.get_tensor_representation()
        
        np.testing.assert_array_equal(original_tensor[0], reconstructed_tensor[0])  # Player 1 pieces
        np.testing.assert_array_equal(original_tensor[1], reconstructed_tensor[1])  # Player 2 pieces
        assert reconstructed.get_current_player() == state.get_current_player()
        
    @patch('mcts.core.game_interface.alphazero_py')
    def test_tensor_to_state_preserves_game_type(self, mock_alphazero):
        """Test that tensor_to_state preserves game type information"""
        mock_alphazero.GomokuState = MockGomokuState
        mock_alphazero.GameType.GOMOKU = 2
        
        interface = GameInterface(GameType.GOMOKU, board_size=15)
        state = MockGomokuState(15)
        
        tensor = interface.state_to_tensor(state)
        reconstructed = interface.tensor_to_state(tensor)
        
        # Should be able to make moves on reconstructed state
        legal_moves = interface.get_legal_moves(reconstructed)
        assert len(legal_moves) > 0
        
        # Should be able to clone reconstructed state
        cloned = reconstructed.clone()
        assert cloned is not reconstructed
        
    @patch('mcts.core.game_interface.alphazero_py')
    def test_tensor_to_state_handles_empty_board(self, mock_alphazero):
        """Test tensor_to_state with empty board"""
        mock_alphazero.GomokuState = MockGomokuState
        mock_alphazero.GameType.GOMOKU = 2
        
        interface = GameInterface(GameType.GOMOKU, board_size=15)
        state = MockGomokuState(15)  # Empty board
        
        tensor = interface.state_to_tensor(state)
        reconstructed = interface.tensor_to_state(tensor)
        
        # Verify empty board
        assert len(interface.get_legal_moves(reconstructed)) == 15 * 15
        assert reconstructed.get_current_player() == 1


class TestMoveHistoryTracking:
    """Test proper move history tracking implementation"""
    
    @patch('mcts.core.game_interface.alphazero_py')
    def test_encode_for_nn_with_move_history(self, mock_alphazero):
        """Test that encode_for_nn properly tracks move history"""
        mock_alphazero.GomokuState = MockGomokuState
        mock_alphazero.GameType.GOMOKU = 2
        mock_alphazero.compute_attack_defense_planes = Mock(
            return_value=(np.zeros((15, 15)), np.zeros((15, 15)))
        )
        
        interface = GameInterface(GameType.GOMOKU, board_size=15)
        
        # Create states with move history
        states = []
        state = MockGomokuState(15)
        states.append(state.clone())
        
        # Make several moves
        moves = [112, 113, 127, 128, 142, 143]
        for move in moves:
            state.make_move(move)
            states.append(state.clone())
            
        # Encode the current state with history
        encoded = interface.encode_for_nn(states[-1], states[:-1])
        
        # Check shape
        assert encoded.shape == (20, 15, 15)
        
        # Channels 2-9 should contain move history for player 1
        # Channels 10-17 should contain move history for player 2
        
        # Player 1 made moves at indices 0, 2, 4 (moves 112, 127, 142)
        # Player 2 made moves at indices 1, 3, 5 (moves 113, 128, 143)
        
        # Most recent move by player 1 (move 142) should be in channel 2
        row, col = 142 // 15, 142 % 15
        assert encoded[2, row, col] != 0  # Should have non-zero value at move position
        
        # Most recent move by player 2 (move 143) should be in channel 10
        row, col = 143 // 15, 143 % 15
        assert encoded[10, row, col] != 0  # Should have non-zero value at move position
        
    @patch('mcts.core.game_interface.alphazero_py')
    def test_encode_for_nn_without_history(self, mock_alphazero):
        """Test encode_for_nn when no history is provided"""
        mock_alphazero.GomokuState = MockGomokuState
        mock_alphazero.GameType.GOMOKU = 2
        mock_alphazero.compute_attack_defense_planes = Mock(
            return_value=(np.zeros((15, 15)), np.zeros((15, 15)))
        )
        
        interface = GameInterface(GameType.GOMOKU, board_size=15)
        state = MockGomokuState(15)
        state.make_move(112)
        
        # Encode without history
        encoded = interface.encode_for_nn(state, [])
        
        # Should still produce 20 channels
        assert encoded.shape == (20, 15, 15)
        
        # Move history channels (2-17) should be all zeros
        for i in range(2, 18):
            assert np.all(encoded[i] == 0)
            
    @patch('mcts.core.game_interface.alphazero_py')
    def test_encode_for_nn_tracks_last_8_moves_only(self, mock_alphazero):
        """Test that encode_for_nn only tracks the last 8 moves per player"""
        mock_alphazero.GomokuState = MockGomokuState  
        mock_alphazero.GameType.GOMOKU = 2
        mock_alphazero.compute_attack_defense_planes = Mock(
            return_value=(np.zeros((15, 15)), np.zeros((15, 15)))
        )
        
        interface = GameInterface(GameType.GOMOKU, board_size=15)
        
        # Create many moves (more than 8 per player)
        states = []
        state = MockGomokuState(15)
        states.append(state.clone())
        
        # Make 20 moves (10 per player)
        for i in range(20):
            move = i * 10  # Spread out moves
            state.make_move(move)
            states.append(state.clone())
            
        encoded = interface.encode_for_nn(states[-1], states[:-1])
        
        # Only the last 8 moves per player should be tracked
        # Count non-zero move history planes
        p1_moves = 0
        p2_moves = 0
        
        for i in range(2, 10):  # Player 1 history channels
            if np.any(encoded[i] != 0):
                p1_moves += 1
                
        for i in range(10, 18):  # Player 2 history channels
            if np.any(encoded[i] != 0):
                p2_moves += 1
                
        assert p1_moves <= 8
        assert p2_moves <= 8