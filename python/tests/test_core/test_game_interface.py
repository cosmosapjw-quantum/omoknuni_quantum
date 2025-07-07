"""
Comprehensive tests for game interface

This module tests the GameInterface class which provides:
- C++ game state wrapping
- State to tensor conversions
- Game-specific move handling
- Symmetry transformations
- Legal move generation
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock

from mcts.core.game_interface import GameInterface, GameType, HAS_CPP_GAMES
from conftest import assert_tensor_equal


@pytest.fixture
def gomoku_game():
    """Create Gomoku game interface"""
    return GameInterface(GameType.GOMOKU, board_size=15)


@pytest.fixture
def gomoku_interface():
    """Create Gomoku game interface (alias for compatibility)"""
    return GameInterface(GameType.GOMOKU, board_size=15)


@pytest.fixture
def chess_interface():
    """Create Chess game interface"""
    return GameInterface(GameType.CHESS)


@pytest.fixture
def go_interface():
    """Create Go game interface"""
    return GameInterface(GameType.GO, board_size=19)


class TestGameInterfaceInitialization:
    """Test GameInterface initialization"""
    
    def test_gomoku_initialization(self):
        """Test Gomoku game initialization"""
        game = GameInterface(GameType.GOMOKU, board_size=15, input_representation='basic')
        
        assert game.game_type == GameType.GOMOKU
        assert game.board_size == 15
        assert game.board_shape == (15, 15)
        assert game.max_moves == 225
        assert game.piece_planes == 2
        assert game.input_representation == 'basic'
        
    def test_go_initialization(self):
        """Test Go game initialization"""
        game = GameInterface(GameType.GO, board_size=19, input_representation='enhanced')
        
        assert game.game_type == GameType.GO
        assert game.board_size == 19
        assert game.board_shape == (19, 19)
        assert game.max_moves == 362  # 19*19 + 1 for pass
        assert game.piece_planes == 2
        assert game.input_representation == 'enhanced'
        
    def test_chess_initialization(self):
        """Test Chess game initialization"""
        game = GameInterface(GameType.CHESS, input_representation='standard')
        
        assert game.game_type == GameType.CHESS
        assert game.board_size == 8
        assert game.board_shape == (8, 8)
        assert game.max_moves == 4096
        assert game.piece_planes == 12  # 6 piece types * 2 colors
        assert game.input_representation == 'standard'
        
    def test_game_options(self):
        """Test game-specific options"""
        # Gomoku with Renju rules
        game = GameInterface(GameType.GOMOKU, board_size=15, use_renju=True)
        assert game.game_options.get('use_renju') == True
        
        # Go with specific rules
        game = GameInterface(GameType.GO, board_size=19, 
                           rule_set='japanese', komi=6.5)
        assert game.game_options.get('rule_set') == 'japanese'
        assert game.game_options.get('komi') == 6.5
        
        # Chess with Chess960
        game = GameInterface(GameType.CHESS, chess960=True)
        assert game.game_options.get('chess960') == True


class TestStateCreation:
    """Test game state creation"""
    
    def test_create_initial_state(self, gomoku_game):
        """Test creating initial game state"""
        state = gomoku_game.create_initial_state()
        
        assert state is not None
        assert hasattr(state, 'get_legal_moves')
        assert hasattr(state, 'make_move')
        assert hasattr(state, 'is_terminal')
        assert hasattr(state, 'get_current_player')
        
        # Should be empty board
        assert not gomoku_game.is_terminal(state)
        assert gomoku_game.get_current_player(state) in [0, 1]
        
    def test_create_state_with_options(self):
        """Test creating states with game-specific options"""
        # Go with specific komi
        go_game = GameInterface(GameType.GO, board_size=9, komi=7.5)
        state = go_game.create_initial_state()
        assert state is not None
        
        # Chess with FEN position
        chess_game = GameInterface(GameType.CHESS, 
                                 fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        state = chess_game.create_initial_state()
        assert state is not None


class TestStateConversions:
    """Test state to tensor conversions"""
    
    def test_state_to_numpy_basic(self, gomoku_game):
        """Test basic state to numpy conversion"""
        state = gomoku_game.create_initial_state()
        
        # Basic representation (18 channels)
        tensor = gomoku_game.state_to_numpy(state, representation_type='basic')
        
        assert isinstance(tensor, np.ndarray)
        assert tensor.shape == (18, 15, 15)
        assert tensor.dtype == np.float32 or tensor.dtype == np.float64
        
        # Should be mostly zeros for empty board
        assert np.sum(tensor[0:2]) == 0  # No stones
        
    def test_state_to_numpy_enhanced(self, gomoku_game):
        """Test enhanced state representation"""
        state = gomoku_game.create_initial_state()
        
        # Enhanced representation (20 channels)
        tensor = gomoku_game.state_to_numpy(state, representation_type='enhanced')
        
        assert tensor.shape == (20, 15, 15)
        
        # Last two channels are attack/defense scores
        assert tensor.shape[0] == 20
        
    def test_state_to_numpy_standard(self, gomoku_game):
        """Test standard state representation"""
        state = gomoku_game.create_initial_state()
        
        # Standard representation (3 channels)
        tensor = gomoku_game.state_to_numpy(state, representation_type='standard')
        
        assert tensor.shape == (3, 15, 15)
        
        # Channel 0: Player 1 pieces
        # Channel 1: Player 2 pieces  
        # Channel 2: Current player
        
    def test_state_to_tensor(self, gomoku_game):
        """Test state to PyTorch tensor conversion"""
        state = gomoku_game.create_initial_state()
        
        tensor = gomoku_game.state_to_tensor(state)
        
        assert isinstance(tensor, torch.Tensor)
        # Default representation may vary - just check it's a valid tensor
        assert len(tensor.shape) == 3
        assert tensor.shape[1] == 15  # Height
        assert tensor.shape[2] == 15  # Width
        assert tensor.dtype == torch.float32
        
    def test_batch_state_conversion(self, gomoku_game):
        """Test batch state conversion"""
        states = []
        for _ in range(4):
            state = gomoku_game.create_initial_state()
            states.append(state)
            
        batch_tensor = gomoku_game.batch_state_to_numpy(states)
        
        # Check batch dimension and spatial dimensions
        assert batch_tensor.shape[0] == 4  # Batch size
        assert batch_tensor.shape[2] == 15  # Height
        assert batch_tensor.shape[3] == 15  # Width
        assert isinstance(batch_tensor, np.ndarray)


class TestTensorToState:
    """Test tensor to state reconstruction"""
    
    def test_tensor_to_state_basic(self, gomoku_game):
        """Test basic tensor to state conversion"""
        # Create a simple position
        state = gomoku_game.create_initial_state()
        state = gomoku_game.apply_move(state, 112)  # Center
        state = gomoku_game.apply_move(state, 113)  # Next to center
        
        # Convert to tensor and back
        tensor = gomoku_game.state_to_tensor(state, representation_type='standard')
        reconstructed = gomoku_game.tensor_to_state(tensor)
        
        # Should have same board position
        tensor_reconstructed = gomoku_game.state_to_tensor(reconstructed, representation_type='standard')
        
        # Board channels should match (may differ in move history)
        assert torch.allclose(tensor[0:2], tensor_reconstructed[0:2])
        
    def test_tensor_to_state_empty(self, gomoku_game):
        """Test tensor to state for empty board"""
        state = gomoku_game.create_initial_state()
        tensor = gomoku_game.state_to_tensor(state)
        
        reconstructed = gomoku_game.tensor_to_state(tensor)
        
        # Should be empty
        assert not gomoku_game.is_terminal(reconstructed)
        legal_moves = gomoku_game.get_legal_moves(reconstructed, shuffle=False)
        assert len(legal_moves) == 225  # All moves legal
        
    def test_tensor_to_state_validation(self, gomoku_game):
        """Test tensor validation in conversion"""
        # Invalid tensor shapes
        with pytest.raises(ValueError, match="Expected 3D tensor"):
            gomoku_game.tensor_to_state(torch.zeros(15, 15))
            
        with pytest.raises(ValueError, match="spatial dimensions"):
            gomoku_game.tensor_to_state(torch.zeros(3, 19, 19))  # Wrong size
            
        with pytest.raises(ValueError, match="Invalid number of channels"):
            gomoku_game.tensor_to_state(torch.zeros(5, 15, 15))  # Wrong channels
            
        # NaN/Inf values
        bad_tensor = torch.zeros(3, 15, 15)
        bad_tensor[0, 0, 0] = float('nan')
        with pytest.raises(ValueError, match="NaN or Inf"):
            gomoku_game.tensor_to_state(bad_tensor)


class TestMoveHandling:
    """Test move application and validation"""
    
    def test_legal_moves_generation(self, gomoku_game):
        """Test legal move generation"""
        state = gomoku_game.create_initial_state()
        
        # Get legal moves
        legal_moves = gomoku_game.get_legal_moves(state, shuffle=False)
        
        assert len(legal_moves) == 225  # All moves legal initially
        assert all(0 <= move < 225 for move in legal_moves)
        
        # After a move
        state = gomoku_game.apply_move(state, 112)
        legal_moves = gomoku_game.get_legal_moves(state, shuffle=False)
        
        assert len(legal_moves) == 224  # One less
        assert 112 not in legal_moves
        
    def test_legal_moves_shuffling(self, gomoku_game):
        """Test legal move shuffling"""
        state = gomoku_game.create_initial_state()
        
        # Get moves with and without shuffle
        moves_shuffled = gomoku_game.get_legal_moves(state, shuffle=True)
        moves_ordered = gomoku_game.get_legal_moves(state, shuffle=False)
        
        # Should have same moves but potentially different order
        assert set(moves_shuffled) == set(moves_ordered)
        assert len(moves_shuffled) == len(moves_ordered)
        
    def test_apply_move(self, gomoku_game):
        """Test move application"""
        state = gomoku_game.create_initial_state()
        
        # Apply center move
        new_state = gomoku_game.apply_move(state, 112)
        
        assert new_state != state  # Should be different object
        assert gomoku_game.get_current_player(new_state) != gomoku_game.get_current_player(state)
        
        # Original state unchanged
        assert len(gomoku_game.get_legal_moves(state)) == 225
        assert len(gomoku_game.get_legal_moves(new_state)) == 224
        
    def test_invalid_move_handling(self, gomoku_game):
        """Test invalid move handling"""
        state = gomoku_game.create_initial_state()
        
        # Out of bounds
        with pytest.raises(ValueError, match="Invalid move"):
            gomoku_game.apply_move(state, 999)
            
        # Negative move
        with pytest.raises(ValueError, match="Invalid move"):
            gomoku_game.apply_move(state, -1)
            
        # Occupied square
        state = gomoku_game.apply_move(state, 112)
        with pytest.raises(ValueError, match="Illegal move"):
            gomoku_game.apply_move(state, 112)
            
    def test_go_pass_move(self):
        """Test Go pass move handling"""
        go_game = GameInterface(GameType.GO, board_size=9)
        state = go_game.create_initial_state()
        
        # Pass move is -1 in Go
        new_state = go_game.apply_move(state, -1)
        
        # Should be valid
        assert new_state is not None
        assert go_game.get_current_player(new_state) != go_game.get_current_player(state)


class TestGameLogic:
    """Test game-specific logic"""
    
    def test_terminal_detection(self, gomoku_game):
        """Test generic terminal position detection"""
        state = gomoku_game.create_initial_state()
        
        # Not terminal initially
        assert not gomoku_game.is_terminal(state)
        
        # Test that is_terminal method exists and works
        assert hasattr(gomoku_game, 'is_terminal')
        assert callable(gomoku_game.is_terminal)
        
    def test_winner_detection(self, gomoku_game):
        """Test generic winner detection interface"""
        state = gomoku_game.create_initial_state()
        
        # Test winner detection method exists
        assert hasattr(gomoku_game, 'get_winner')
        assert callable(gomoku_game.get_winner)
        
        # Winner should be valid for non-terminal state
        if not gomoku_game.is_terminal(state):
            winner = gomoku_game.get_winner(state)
            assert winner in [0, 1, 2]  # Valid winner or ongoing
        
    def test_draw_detection(self):
        """Test draw detection"""
        # This is game-specific
        # For Gomoku, draws are rare (full board)
        # For Go, draws can happen with equal territory
        # For Chess, many draw conditions exist
        pass
        
    def test_current_player_tracking(self, gomoku_game):
        """Test current player tracking"""
        state = gomoku_game.create_initial_state()
        
        # Track player alternation
        players = []
        for i in range(10):
            players.append(gomoku_game.get_current_player(state))
            state = gomoku_game.apply_move(state, i * 10)  # Arbitrary moves
            
        # Should alternate
        for i in range(len(players) - 1):
            assert players[i] != players[i + 1]


class TestSymmetries:
    """Test symmetry transformations"""
    
    def test_symmetries_interface(self, gomoku_game):
        """Test generic symmetries interface"""
        # Create simple board state
        board = np.zeros((3, 15, 15))
        board[0, 7, 7] = 1  # Center piece
        
        # Create policy
        policy = np.zeros(225)
        policy[112] = 1.0  # Center move
        
        # Get symmetries
        symmetries = gomoku_game.get_symmetries(board, policy)
        
        # Should return list of symmetries
        assert isinstance(symmetries, list)
        assert len(symmetries) > 0
        
        # Each should be valid
        for sym_board, sym_policy in symmetries:
            assert sym_board.shape == board.shape
            assert sym_policy.shape == policy.shape
            assert abs(sym_policy.sum() - 1.0) < 1e-6
        
    def test_go_symmetries_with_pass(self):
        """Test Go symmetries preserve pass move"""
        go_game = GameInterface(GameType.GO, board_size=9)
        
        board = np.zeros((3, 9, 9))
        board[0, 4, 4] = 1  # Center stone
        
        # Policy with pass move
        policy = np.zeros(82)  # 81 + 1 for pass
        policy[40] = 0.6  # Center
        policy[81] = 0.4  # Pass move
        
        symmetries = go_game.get_symmetries(board, policy)
        
        # All symmetries should preserve pass probability
        for _, sym_policy in symmetries:
            assert abs(sym_policy[81] - 0.4) < 1e-6
            
    def test_chess_no_symmetries(self):
        """Test Chess has no symmetries due to castling"""
        chess_game = GameInterface(GameType.CHESS)
        
        board = np.zeros((12, 8, 8))
        policy = np.zeros(4096)
        policy[0] = 1.0
        
        symmetries = chess_game.get_symmetries(board, policy)
        
        # Should only return original
        assert len(symmetries) == 1
        assert np.allclose(symmetries[0][0], board)
        assert np.allclose(symmetries[0][1], policy)


class TestUtilityMethods:
    """Test utility methods"""
    
    def test_get_hash(self, gomoku_game):
        """Test state hashing"""
        state1 = gomoku_game.create_initial_state()
        state2 = gomoku_game.create_initial_state()
        
        # Same position should have same hash
        hash1 = gomoku_game.get_hash(state1)
        hash2 = gomoku_game.get_hash(state2)
        assert hash1 == hash2
        
        # Different position should have different hash
        state3 = gomoku_game.apply_move(state1, 112)
        hash3 = gomoku_game.get_hash(state3)
        assert hash3 != hash1
        
    def test_action_to_string(self, gomoku_game):
        """Test action string conversion"""
        state = gomoku_game.create_initial_state()
        
        # Convert action to string
        action_str = gomoku_game.action_to_string(state, 112)
        assert isinstance(action_str, str)
        
        # Convert back
        action = gomoku_game.string_to_action(state, action_str)
        assert action == 112
        
    def test_clone_state(self, gomoku_game):
        """Test state cloning"""
        state = gomoku_game.create_initial_state()
        state = gomoku_game.apply_move(state, 112)
        
        # Clone
        cloned = gomoku_game.clone_state(state)
        
        # Should be equal but different objects
        assert cloned != state  # Different objects
        assert gomoku_game.get_hash(cloned) == gomoku_game.get_hash(state)
        
        # Modifying clone shouldn't affect original
        cloned = gomoku_game.apply_move(cloned, 113)
        assert gomoku_game.get_hash(cloned) != gomoku_game.get_hash(state)
        
    def test_get_state_hash(self, gomoku_game):
        """Test state hashing (alternative method name)"""
        state1 = gomoku_game.create_initial_state()
        state2 = gomoku_game.create_initial_state()
        
        # Same states should have same hash
        if hasattr(gomoku_game, 'get_state_hash'):
            hash1 = gomoku_game.get_state_hash(state1)
            hash2 = gomoku_game.get_state_hash(state2)
            assert hash1 == hash2
            
            # Different states should have different hashes
            new_state = gomoku_game.apply_move(state1, 0)
            hash3 = gomoku_game.get_state_hash(new_state)
            assert hash1 != hash3
        
    def test_get_canonical_form(self, gomoku_game):
        """Test canonical form for training"""
        state = gomoku_game.create_initial_state()
        
        # Get canonical form
        canonical = gomoku_game.get_canonical_form(state)
        
        assert isinstance(canonical, np.ndarray)
        assert canonical.shape[0] >= 3  # At least 3 channels
        
    def test_encode_for_nn(self, gomoku_game):
        """Test neural network input encoding"""
        state = gomoku_game.create_initial_state()
        
        # Make some moves to create history
        history = []
        for i in range(5):
            history.append(state)
            state = gomoku_game.apply_move(state, i * 20)
            
        # Encode with history
        encoded = gomoku_game.encode_for_nn(state, history)
        
        assert encoded.shape == (20, 15, 15)  # 20 channels
        
        # Channel breakdown:
        # 0: Current board
        # 1: Current player
        # 2-9: Player 1 move history
        # 10-17: Player 2 move history  
        # 18-19: Attack/defense scores


class TestActionProbabilityMask:
    """Test action probability masking"""
    
    def test_action_probability_mask(self, gomoku_game):
        """Test legal action mask generation"""
        state = gomoku_game.create_initial_state()
        
        # Get mask
        mask = gomoku_game.get_action_probabilities_mask(state)
        
        assert mask.shape == (225,)
        assert mask.dtype == bool
        assert mask.sum() == 225  # All legal initially
        
        # After moves
        state = gomoku_game.apply_move(state, 112)
        state = gomoku_game.apply_move(state, 113)
        
        mask = gomoku_game.get_action_probabilities_mask(state)
        assert mask.sum() == 223  # Two occupied
        assert not mask[112]
        assert not mask[113]
        
    def test_get_legal_mask(self, gomoku_game):
        """Test legal move mask generation (using action probabilities mask)"""
        state = gomoku_game.create_initial_state()
        mask = gomoku_game.get_action_probabilities_mask(state)
        
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (225,)
        assert mask.dtype == bool
        
        # Check consistency with get_legal_moves
        legal_moves = gomoku_game.get_legal_moves(state, shuffle=False)
        for i in range(225):
            assert (i in legal_moves) == mask[i]


class TestGameSpecificFeatures:
    """Test game-specific features"""
    
    def test_go_specific_features(self):
        """Test Go-specific features"""
        go_game = GameInterface(GameType.GO, board_size=9, 
                              rule_set='chinese', komi=7.5, enforce_superko=True)
        state = go_game.create_initial_state()
        
        # Test pass move
        assert go_game.is_legal_move(state, -1)  # Pass is legal
        
        # Test action space includes pass
        assert go_game.get_action_space_size(state) == 82  # 81 + pass
        
    def test_chess_specific_features(self):
        """Test Chess-specific features"""
        chess_game = GameInterface(GameType.CHESS, chess960=False)
        state = chess_game.create_initial_state()
        
        # Chess has large action space (includes underpromotions)
        assert chess_game.get_action_space_size(state) == 20480
        
        # Test FEN support
        fen_str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        chess_with_fen = GameInterface(GameType.CHESS, fen=fen_str)
        state_from_fen = chess_with_fen.create_initial_state()
        assert state_from_fen is not None
        
    def test_gomoku_specific_features(self):
        """Test Gomoku-specific interface features"""
        # Test board size variations
        small_gomoku = GameInterface(GameType.GOMOKU, board_size=9)
        assert small_gomoku.board_size == 9
        assert small_gomoku.max_moves == 81
        
        # Test Renju option interface
        renju_game = GameInterface(GameType.GOMOKU, board_size=15, use_renju=True)
        assert renju_game.game_options.get('use_renju') == True


class TestEdgeCases:
    """Test edge cases"""
    
    def test_legal_moves_behavior(self, gomoku_game):
        """Test generic legal moves behavior"""
        state = gomoku_game.create_initial_state()
        legal_moves = gomoku_game.get_legal_moves(state)
        
        # Should have some legal moves initially
        assert len(legal_moves) > 0
        
        # All moves should be valid integers
        assert all(isinstance(move, int) for move in legal_moves)
        assert all(move >= 0 for move in legal_moves)
        
    def test_invalid_action_space(self, gomoku_game):
        """Test handling of invalid action"""
        state = gomoku_game.create_initial_state()
        
        # Out of bounds action
        with pytest.raises((ValueError, Exception)):
            gomoku_game.apply_move(state, 1000)  # Way out of bounds


class TestErrorHandling:
    """Test error handling"""
    
    def test_invalid_game_type(self):
        """Test invalid game type handling"""
        with pytest.raises(ValueError):
            GameInterface("invalid_game", board_size=15)
            
    def test_missing_cpp_module(self):
        """Test handling when C++ module is missing"""
        with patch('mcts.core.game_interface.HAS_CPP_GAMES', False):
            # Should still create game with mock
            game = GameInterface(GameType.GOMOKU, board_size=15)
            state = game.create_initial_state()
            assert state is not None  # Mock state
            
    def test_state_validation(self, gomoku_game):
        """Test state validation in methods"""
        # Test with None state
        with pytest.raises(AttributeError):
            gomoku_game.get_legal_moves(None)
            
        # Test with invalid state type
        with pytest.raises(AttributeError):
            gomoku_game.apply_move("not a state", 112)
            
    def test_state_hashing(self, gomoku_game):
        """Test state hashing for transposition tables"""
        state1 = gomoku_game.create_initial_state()
        state2 = gomoku_game.create_initial_state()
        
        # Same states should have same hash
        hash1 = gomoku_game.get_hash(state1)
        hash2 = gomoku_game.get_hash(state2)
        assert hash1 == hash2
        
        # Different states should have different hashes
        new_state = gomoku_game.apply_move(state1, 0)
        hash3 = gomoku_game.get_hash(new_state)
        assert hash1 != hash3


class TestIntegration:
    """Integration tests with MCTS"""
    
    def test_mcts_compatibility(self, gomoku_game):
        """Test interface compatibility with MCTS"""
        # Test that interface provides all methods needed by MCTS
        required_methods = [
            'create_initial_state',
            'get_legal_moves',
            'apply_move',
            'is_terminal',
            'get_winner',
            'get_canonical_form',
            'get_action_space_size'
        ]
        
        for method in required_methods:
            assert hasattr(gomoku_game, method)
            assert callable(getattr(gomoku_game, method))
            
    def test_neural_network_compatibility(self, gomoku_game):
        """Test interface compatibility with neural networks"""
        state = gomoku_game.create_initial_state()
        
        # Convert to numpy for neural network
        tensor = gomoku_game.state_to_numpy(state)
        
        # Should be suitable for neural network input
        assert isinstance(tensor, np.ndarray)
        assert tensor.dtype in [np.float32, np.float64]
        assert len(tensor.shape) == 3  # (channels, height, width)
        
        # Action space size should match network output
        assert gomoku_game.get_action_space_size(state) > 0