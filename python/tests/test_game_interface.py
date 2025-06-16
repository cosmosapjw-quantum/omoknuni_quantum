"""Tests for GameInterface wrapper"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from mcts.core.game_interface import GameInterface, GameType


class TestGameInterface:
    """Test suite for GameInterface wrapper"""
    
    def test_game_interface_initialization(self):
        """Test basic initialization of GameInterface"""
        # Test with chess
        chess_interface = GameInterface(GameType.CHESS)
        assert chess_interface.game_type == GameType.CHESS
        assert chess_interface.board_shape == (8, 8)
        assert chess_interface.max_moves == 4096  # Chess has many possible moves
        
        # Test with Go
        go_interface = GameInterface(GameType.GO, board_size=19)
        assert go_interface.game_type == GameType.GO
        assert go_interface.board_shape == (19, 19)
        assert go_interface.max_moves == 19 * 19 + 1  # All intersections + pass
        
        # Test with Gomoku
        gomoku_interface = GameInterface(GameType.GOMOKU, board_size=15)
        assert gomoku_interface.game_type == GameType.GOMOKU
        assert gomoku_interface.board_shape == (15, 15)
        assert gomoku_interface.max_moves == 15 * 15
        
    def test_create_initial_state(self):
        """Test creating initial game states"""
        # Chess initial state
        chess_interface = GameInterface(GameType.CHESS)
        initial_state = chess_interface.create_initial_state()
        assert initial_state is not None
        assert hasattr(initial_state, 'get_tensor_representation')
        assert hasattr(initial_state, 'get_legal_moves')
        
        # Go initial state
        go_interface = GameInterface(GameType.GO, board_size=9)
        initial_state = go_interface.create_initial_state()
        assert initial_state is not None
        assert hasattr(initial_state, 'get_legal_moves')
        
    def test_state_to_numpy(self):
        """Test converting game state to numpy array"""
        chess_interface = GameInterface(GameType.CHESS)
        state = chess_interface.create_initial_state()
        
        # Convert to numpy
        board_array = chess_interface.state_to_numpy(state)
        assert isinstance(board_array, np.ndarray)
        # Shape is (channels, height, width)
        assert board_array.shape[1:] == (8, 8)  # Chess board is 8x8
        assert board_array.shape[0] >= 12  # At least 12 channels for piece types
        
        # Check initial position has pieces
        assert board_array.sum() > 0  # Should have pieces on board
        
    def test_get_legal_moves(self):
        """Test getting legal moves from a state"""
        chess_interface = GameInterface(GameType.CHESS)
        state = chess_interface.create_initial_state()
        
        legal_moves = chess_interface.get_legal_moves(state)
        assert isinstance(legal_moves, list)
        assert len(legal_moves) > 0  # Should have legal moves in initial position
        assert all(isinstance(move, int) for move in legal_moves)
        
        # In chess initial position, should have 20 moves (16 pawn + 4 knight)
        assert len(legal_moves) == 20
        
    def test_apply_move(self):
        """Test applying a move to a state"""
        chess_interface = GameInterface(GameType.CHESS)
        state = chess_interface.create_initial_state()
        
        # Get legal moves
        legal_moves = chess_interface.get_legal_moves(state)
        assert len(legal_moves) > 0
        
        # Apply first legal move
        move = legal_moves[0]
        new_state = chess_interface.apply_move(state, move)
        
        # State should change
        assert new_state is not state  # Should be a new state
        assert new_state != state  # Should be different
        
        # Turn should change
        assert chess_interface.get_current_player(state) != chess_interface.get_current_player(new_state)
        
    def test_is_terminal(self):
        """Test checking if state is terminal"""
        chess_interface = GameInterface(GameType.CHESS)
        state = chess_interface.create_initial_state()
        
        # Initial state should not be terminal
        assert not chess_interface.is_terminal(state)
        
        # Mock a terminal state
        mock_terminal_state = Mock()
        mock_terminal_state.is_terminal.return_value = True
        assert chess_interface.is_terminal(mock_terminal_state)
        
    def test_get_winner(self):
        """Test getting winner from terminal state"""
        chess_interface = GameInterface(GameType.CHESS)
        
        # If C++ games are available, we need to mock get_game_result
        if hasattr(chess_interface, '_game_class') and chess_interface._game_class is not None:
            # Mock a winning state for white
            mock_state = Mock()
            mock_state.is_terminal.return_value = True
            
            # Import alphazero_py to get GameResult enum
            try:
                import alphazero_py
                mock_state.get_game_result.return_value = alphazero_py.GameResult.WIN_PLAYER1
                winner = chess_interface.get_winner(mock_state)
                assert winner == 1
                
                # Mock a draw
                mock_state.get_game_result.return_value = alphazero_py.GameResult.DRAW
                winner = chess_interface.get_winner(mock_state)
                assert winner == 0
            except ImportError:
                # Fallback for when alphazero_py is not available
                mock_state.get_winner.return_value = 1
                winner = chess_interface.get_winner(mock_state)
                assert winner == 1
        else:
            # Use simple mock
            mock_state = Mock()
            mock_state.get_winner.return_value = 1
            winner = chess_interface.get_winner(mock_state)
            assert winner == 1
        
    def test_move_to_action_index(self):
        """Test converting moves to action indices"""
        chess_interface = GameInterface(GameType.CHESS)
        state = chess_interface.create_initial_state()
        
        legal_moves = chess_interface.get_legal_moves(state)
        
        # Convert to action indices
        action_indices = [chess_interface.move_to_action_index(move) for move in legal_moves]
        assert all(0 <= idx < chess_interface.max_moves for idx in action_indices)
        
        # Should be unique
        assert len(set(action_indices)) == len(action_indices)
        
    def test_action_index_to_move(self):
        """Test converting action indices back to moves"""
        chess_interface = GameInterface(GameType.CHESS)
        state = chess_interface.create_initial_state()
        
        legal_moves = chess_interface.get_legal_moves(state)
        
        # Round-trip conversion
        for move in legal_moves:
            idx = chess_interface.move_to_action_index(move)
            recovered_move = chess_interface.action_index_to_move(idx)
            assert recovered_move == move
            
    def test_get_action_probabilities_mask(self):
        """Test getting action probability mask for legal moves"""
        chess_interface = GameInterface(GameType.CHESS)
        state = chess_interface.create_initial_state()
        
        mask = chess_interface.get_action_probabilities_mask(state)
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (chess_interface.max_moves,)
        assert mask.dtype == bool
        
        # Check that legal moves are marked True
        legal_moves = chess_interface.get_legal_moves(state)
        legal_indices = [chess_interface.move_to_action_index(m) for m in legal_moves]
        
        assert mask[legal_indices].all()  # All legal moves should be True
        assert mask.sum() == len(legal_moves)  # Only legal moves should be True
        
    def test_get_symmetries(self):
        """Test getting board symmetries for data augmentation"""
        # Go has 8 symmetries (4 rotations * 2 reflections)
        go_interface = GameInterface(GameType.GO, board_size=9)
        state = go_interface.create_initial_state()
        board = go_interface.state_to_numpy(state)
        
        symmetries = go_interface.get_symmetries(board, np.ones(go_interface.max_moves))
        assert len(symmetries) == 8  # Go has 8 symmetries
        
        # Each symmetry should have board and policy
        for sym_board, sym_policy in symmetries:
            assert sym_board.shape == board.shape
            assert sym_policy.shape == (go_interface.max_moves,)
            
        # Chess has no symmetries (due to castling rights)
        chess_interface = GameInterface(GameType.CHESS)
        chess_board = chess_interface.state_to_numpy(chess_interface.create_initial_state())
        chess_symmetries = chess_interface.get_symmetries(chess_board, np.ones(chess_interface.max_moves))
        assert len(chess_symmetries) == 1  # Only identity
        
    def test_encode_for_nn(self):
        """Test encoding state for neural network input"""
        chess_interface = GameInterface(GameType.CHESS)
        state = chess_interface.create_initial_state()
        
        # Encode with history
        history = [state] * 8  # 8 positions for history
        encoded = chess_interface.encode_for_nn(state, history)
        
        assert isinstance(encoded, np.ndarray)
        assert len(encoded.shape) == 3
        assert encoded.shape[1:] == (8, 8)
        # Should have exactly 20 channels as per documentation
        assert encoded.shape[0] == 20
        
    def test_batch_operations(self):
        """Test batch operations for vectorized MCTS"""
        chess_interface = GameInterface(GameType.CHESS)
        
        # Create batch of states
        states = [chess_interface.create_initial_state() for _ in range(32)]
        
        # Batch convert to numpy
        boards = chess_interface.batch_state_to_numpy(states)
        # batch_state_to_numpy returns (batch, channels, height, width)
        assert boards.shape[0] == 32  # Batch size
        assert boards.shape[2:] == (8, 8)  # Chess board size
        
        # Batch get legal moves
        all_legal_moves = chess_interface.batch_get_legal_moves(states)
        assert len(all_legal_moves) == 32
        assert all(len(moves) == 20 for moves in all_legal_moves)  # Initial chess position
        
    def test_zobrist_hash(self):
        """Test Zobrist hashing for transposition tables"""
        chess_interface = GameInterface(GameType.CHESS)
        state = chess_interface.create_initial_state()
        
        hash1 = chess_interface.get_hash(state)
        assert isinstance(hash1, int)
        
        # Same state should give same hash
        hash2 = chess_interface.get_hash(state)
        assert hash1 == hash2
        
        # Different state should give different hash
        legal_moves = chess_interface.get_legal_moves(state)
        new_state = chess_interface.apply_move(state, legal_moves[0])
        hash3 = chess_interface.get_hash(new_state)
        assert hash1 != hash3
        
    def test_error_handling(self):
        """Test error handling for invalid operations"""
        chess_interface = GameInterface(GameType.CHESS)
        state = chess_interface.create_initial_state()
        
        # Invalid move should raise error
        with pytest.raises(ValueError):
            chess_interface.apply_move(state, -1)  # Invalid move
            
        # Invalid game type
        with pytest.raises(ValueError):
            GameInterface("invalid_game")