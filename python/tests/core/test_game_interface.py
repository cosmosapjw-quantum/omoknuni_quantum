"""Tests for game_interface module"""

import pytest
import numpy as np
from typing import List, Tuple

from mcts.core.game_interface import GameInterface, GameType


class TestGameType:
    """Test the GameType enum"""
    
    def test_game_type_values(self):
        """Test that GameType has expected values"""
        assert GameType.GOMOKU == "gomoku"
        assert GameType.GO == "go"
        assert GameType.CHESS == "chess"
    
    def test_game_type_string_conversion(self):
        """Test string conversion of GameType"""
        assert str(GameType.GOMOKU) == "gomoku"
        assert str(GameType.GO) == "go"
        assert str(GameType.CHESS) == "chess"


class TestGameInterface:
    """Test the GameInterface class"""
    
    def test_game_interface_creation_gomoku(self):
        """Test creating GameInterface for Gomoku"""
        gi = GameInterface(game_type=GameType.GOMOKU, board_size=15)
        
        assert gi.game_type == GameType.GOMOKU
        assert gi.board_size == 15
        assert gi.max_moves == 225  # 15x15
        assert gi.board_shape == (15, 15)
    
    def test_game_interface_creation_go(self):
        """Test creating GameInterface for Go"""
        gi = GameInterface(game_type=GameType.GO, board_size=19)
        
        assert gi.game_type == GameType.GO
        assert gi.board_size == 19
        assert gi.max_moves == 362  # 19x19 + 1 (pass move)
        assert gi.board_shape == (19, 19)
    
    def test_game_interface_creation_chess(self):
        """Test creating GameInterface for Chess"""
        gi = GameInterface(game_type=GameType.CHESS)
        
        assert gi.game_type == GameType.CHESS
        assert gi.board_size == 8
        assert gi.max_moves == 4096  # Chess action space
        assert gi.board_shape == (8, 8)
    
    def test_game_interface_invalid_game_type(self):
        """Test creating GameInterface with invalid game type"""
        with pytest.raises(ValueError):
            GameInterface(game_type="invalid_game")
    
    def test_initial_state_gomoku(self, game_interface):
        """Test getting initial state for Gomoku"""
        state = game_interface.get_initial_state()
        
        assert state.shape == (3, 15, 15)
        assert state.dtype == np.float32
        
        # Empty board - player positions should be zero
        assert np.all(state[0] == 0)  # Player 1 positions
        assert np.all(state[1] == 0)  # Player 2 positions
        assert np.all(state[2] == 1)  # Turn indicator (Player 1 starts)
    
    def test_initial_state_small_board(self, small_game_interface):
        """Test getting initial state for small board"""
        state = small_game_interface.get_initial_state()
        
        assert state.shape == (3, 9, 9)
        assert np.all(state[0] == 0)
        assert np.all(state[1] == 0)
        assert np.all(state[2] == 1)
    
    def test_legal_moves_initial_state(self, game_interface):
        """Test getting legal moves from initial state"""
        state = game_interface.get_initial_state()
        legal_moves = game_interface.get_legal_moves(state)
        
        assert len(legal_moves) == 225
        assert legal_moves.dtype == bool
        assert np.all(legal_moves)  # All moves should be legal initially
    
    def test_legal_moves_with_occupied_positions(self, game_interface, sample_game_state):
        """Test getting legal moves with some occupied positions"""
        legal_moves = game_interface.get_legal_moves(sample_game_state)
        
        # Positions (7,7), (7,8), (8,7), (8,8) should be occupied
        occupied_indices = [7*15 + 7, 7*15 + 8, 8*15 + 7, 8*15 + 8]
        
        for idx in occupied_indices:
            assert not legal_moves[idx], f"Position {idx} should be occupied"
        
        # Other positions should be legal
        total_legal = np.sum(legal_moves)
        assert total_legal == 225 - 4  # All except 4 occupied positions
    
    def test_make_move_valid(self, game_interface):
        """Test making a valid move"""
        state = game_interface.get_initial_state()
        
        # Make move at center position (7, 7) -> action 7*15 + 7 = 112
        action = 112
        new_state = game_interface.make_move(state, action)
        
        assert new_state.shape == state.shape
        assert new_state[0, 7, 7] == 1.0  # Player 1 move placed
        assert np.all(new_state[2] == 0.0)  # Turn switched to Player 2
        
        # Original state should be unchanged
        assert np.all(state[0] == 0)
    
    def test_make_move_invalid(self, game_interface, sample_game_state):
        """Test making an invalid move (occupied position)"""
        # Try to move at occupied position (7, 7)
        action = 7*15 + 7
        
        with pytest.raises(ValueError):
            game_interface.make_move(sample_game_state, action)
    
    def test_make_move_out_of_bounds(self, game_interface):
        """Test making an out-of-bounds move"""
        state = game_interface.get_initial_state()
        
        # Action out of bounds
        with pytest.raises(ValueError):
            game_interface.make_move(state, 225)  # Max valid action is 224
        
        with pytest.raises(ValueError):
            game_interface.make_move(state, -1)
    
    def test_get_winner_no_winner(self, game_interface, sample_game_state):
        """Test getting winner when there's no winner yet"""
        winner = game_interface.get_winner(sample_game_state)
        assert winner == 0  # No winner
    
    def test_get_winner_player1_wins(self, game_interface, winning_state):
        """Test getting winner when Player 1 wins"""
        winner = game_interface.get_winner(winning_state)
        assert winner == 1  # Player 1 wins
    
    def test_get_winner_draw(self, game_interface):
        """Test getting winner in a draw situation"""
        # Create a full board with no winner
        state = np.zeros((3, 15, 15), dtype=np.float32)
        
        # Fill board in checkerboard pattern (no 5-in-a-row)
        for i in range(15):
            for j in range(15):
                if (i + j) % 2 == 0:
                    state[0, i, j] = 1.0  # Player 1
                else:
                    state[1, i, j] = 1.0  # Player 2
        
        winner = game_interface.get_winner(state)
        assert winner == -1  # Draw
    
    def test_is_terminal_initial_state(self, game_interface):
        """Test is_terminal on initial state"""
        state = game_interface.get_initial_state()
        assert not game_interface.is_terminal(state)
    
    def test_is_terminal_winning_state(self, game_interface, winning_state):
        """Test is_terminal on winning state"""
        assert game_interface.is_terminal(winning_state)
    
    def test_is_terminal_ongoing_game(self, game_interface, sample_game_state):
        """Test is_terminal on ongoing game"""
        assert not game_interface.is_terminal(sample_game_state)
    
    def test_action_to_position(self, game_interface):
        """Test converting action to board position"""
        # Action 112 should be (7, 7) for 15x15 board
        row, col = game_interface.action_to_position(112)
        assert row == 7
        assert col == 7
        
        # Action 0 should be (0, 0)
        row, col = game_interface.action_to_position(0)
        assert row == 0
        assert col == 0
        
        # Action 224 should be (14, 14)
        row, col = game_interface.action_to_position(224)
        assert row == 14
        assert col == 14
    
    def test_position_to_action(self, game_interface):
        """Test converting board position to action"""
        # Position (7, 7) should be action 112
        action = game_interface.position_to_action(7, 7)
        assert action == 112
        
        # Position (0, 0) should be action 0
        action = game_interface.position_to_action(0, 0)
        assert action == 0
        
        # Position (14, 14) should be action 224
        action = game_interface.position_to_action(14, 14)
        assert action == 224
    
    def test_action_position_roundtrip(self, game_interface):
        """Test that action->position->action is consistent"""
        for action in [0, 50, 112, 200, 224]:
            row, col = game_interface.action_to_position(action)
            recovered_action = game_interface.position_to_action(row, col)
            assert recovered_action == action
    
    def test_get_canonical_state_player1_turn(self, game_interface, sample_game_state):
        """Test getting canonical state when it's Player 1's turn"""
        # sample_game_state has Player 1's turn (state[2] == 1)
        canonical = game_interface.get_canonical_state(sample_game_state, player=1)
        
        # Should be unchanged for Player 1
        np.testing.assert_array_equal(canonical, sample_game_state)
    
    def test_get_canonical_state_player2_turn(self, game_interface, sample_small_game_state):
        """Test getting canonical state when it's Player 2's turn"""
        # sample_small_game_state has Player 2's turn (state[2] == 0)
        canonical = game_interface.get_canonical_state(sample_small_game_state, player=2)
        
        # Should swap player positions and flip turn indicator
        assert np.all(canonical[0] == sample_small_game_state[1])  # Player 2 -> Player 1
        assert np.all(canonical[1] == sample_small_game_state[0])  # Player 1 -> Player 2
        assert np.all(canonical[2] == 1.0)  # Turn indicator flipped to 1
    
    def test_clone_state(self, game_interface, sample_game_state):
        """Test cloning game state"""
        cloned_state = game_interface.clone_state(sample_game_state)
        
        # Should be equal but different objects
        np.testing.assert_array_equal(cloned_state, sample_game_state)
        assert cloned_state is not sample_game_state
        
        # Modifying clone shouldn't affect original
        cloned_state[0, 0, 0] = 1.0
        assert sample_game_state[0, 0, 0] == 0.0
    
    def test_state_to_string(self, game_interface, sample_game_state):
        """Test converting state to string representation"""
        state_str = game_interface.state_to_string(sample_game_state)
        
        assert isinstance(state_str, str)
        assert len(state_str) > 0
        
        # Should contain some indication of the board state
        assert 'X' in state_str or 'O' in state_str or '1' in state_str or '2' in state_str
    
    def test_get_symmetries(self, game_interface, sample_game_state):
        """Test getting symmetric transformations of state"""
        symmetries = game_interface.get_symmetries(sample_game_state)
        
        # Should return list of (state, action_mapping) tuples
        assert isinstance(symmetries, list)
        assert len(symmetries) == 8  # 4 rotations + 4 reflections
        
        for sym_state, action_map in symmetries:
            assert sym_state.shape == sample_game_state.shape
            assert len(action_map) == 225  # Gomoku action space
            assert all(0 <= a < 225 for a in action_map)
    
    def test_get_state_hash(self, game_interface, sample_game_state):
        """Test getting hash of game state"""
        hash1 = game_interface.get_state_hash(sample_game_state)
        hash2 = game_interface.get_state_hash(sample_game_state)
        
        # Same state should have same hash
        assert hash1 == hash2
        
        # Different state should have different hash
        different_state = game_interface.clone_state(sample_game_state)
        different_state[0, 0, 0] = 1.0
        hash3 = game_interface.get_state_hash(different_state)
        assert hash1 != hash3


class TestGameIntegration:
    """Integration tests for game interface functionality"""
    
    def test_complete_game_simulation(self, small_game_interface):
        """Test simulating a complete game"""
        state = small_game_interface.get_initial_state()
        moves_made = 0
        max_moves = 81  # 9x9 board
        
        while not small_game_interface.is_terminal(state) and moves_made < max_moves:
            legal_moves = small_game_interface.get_legal_moves(state)
            legal_actions = np.where(legal_moves)[0]
            
            if len(legal_actions) == 0:
                break
            
            # Make random legal move
            action = np.random.choice(legal_actions)
            state = small_game_interface.make_move(state, action)
            moves_made += 1
        
        # Game should terminate within reasonable number of moves
        assert moves_made <= max_moves
        
        # Final state should be terminal
        if moves_made < max_moves:
            assert small_game_interface.is_terminal(state)
    
    def test_game_state_consistency(self, game_interface):
        """Test that game state remains consistent through moves"""
        state = game_interface.get_initial_state()
        
        # Make several moves and check consistency
        moves = [112, 113, 127, 128, 142]  # Some center moves
        
        for i, action in enumerate(moves):
            legal_moves = game_interface.get_legal_moves(state)
            
            if legal_moves[action]:
                new_state = game_interface.make_move(state, action)
                
                # Check turn alternation
                current_player = 1 if np.all(new_state[2] == 0) else 2
                expected_player = (i % 2) + 1
                
                # Verify the move was placed correctly
                row, col = game_interface.action_to_position(action)
                player_layer = 0 if expected_player == 1 else 1
                assert new_state[player_layer, row, col] == 1.0
                
                state = new_state
    
    def test_different_board_sizes(self):
        """Test GameInterface with different board sizes"""
        sizes = [9, 15, 19]
        
        for size in sizes:
            gi = GameInterface(game_type=GameType.GOMOKU, board_size=size)
            
            assert gi.board_size == size
            assert gi.action_size == size * size
            assert gi.state_shape == (3, size, size)
            
            # Test basic functionality
            state = gi.get_initial_state()
            assert state.shape == (3, size, size)
            
            legal_moves = gi.get_legal_moves(state)
            assert len(legal_moves) == size * size
            assert np.all(legal_moves)  # All moves should be legal initially