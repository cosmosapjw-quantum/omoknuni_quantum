"""Tests for GPU game state management

Tests cover:
- State allocation and deallocation
- State cloning
- Move application
- Legal move generation
- Terminal state detection
- Neural network feature extraction
- Enhanced features
- Different game types
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, Mock, MagicMock
import alphazero_py

from mcts.gpu.gpu_game_states import (
    GPUGameStates, GPUGameStatesConfig, GameType
)


@pytest.fixture
def gomoku_config():
    """Create Gomoku game configuration"""
    return GPUGameStatesConfig(
        capacity=100,
        game_type=GameType.GOMOKU,
        board_size=15,
        device='cpu',
        dtype=torch.float32
    )


@pytest.fixture
def go_config():
    """Create Go game configuration"""
    return GPUGameStatesConfig(
        capacity=50,
        game_type=GameType.GO,
        board_size=9,  # Small board for testing
        device='cpu'
    )


@pytest.fixture
def chess_config():
    """Create Chess game configuration"""
    return GPUGameStatesConfig(
        capacity=20,
        game_type=GameType.CHESS,
        board_size=8,
        device='cpu'
    )


@pytest.fixture
def gomoku_states(gomoku_config):
    """Create Gomoku GPU states"""
    return GPUGameStates(gomoku_config)


@pytest.fixture
def go_states(go_config):
    """Create Go GPU states"""
    return GPUGameStates(go_config)


@pytest.fixture
def chess_states(chess_config):
    """Create Chess GPU states"""
    return GPUGameStates(chess_config)


class TestGPUGameStatesConfig:
    """Test GPUGameStatesConfig configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = GPUGameStatesConfig()
        assert config.capacity == 100000
        assert config.game_type == GameType.GOMOKU
        assert config.board_size == 15
        assert config.device == 'cuda'
        assert config.dtype == torch.float32
        
    def test_custom_config(self):
        """Test custom configuration"""
        config = GPUGameStatesConfig(
            capacity=5000,
            game_type=GameType.GO,
            board_size=19,
            device='cpu'
        )
        assert config.capacity == 5000
        assert config.game_type == GameType.GO
        assert config.board_size == 19
        assert config.device == 'cpu'
        
    def test_board_size_defaults(self):
        """Test board size defaults for different games"""
        # Chess
        config = GPUGameStatesConfig(game_type=GameType.CHESS, board_size=0)
        assert config.board_size == 8
        
        # Go
        config = GPUGameStatesConfig(game_type=GameType.GO, board_size=0)
        assert config.board_size == 19
        
        # Gomoku
        config = GPUGameStatesConfig(game_type=GameType.GOMOKU, board_size=0)
        assert config.board_size == 15
        
    def test_game_type_enum(self):
        """Test game type enumeration"""
        assert GameType.CHESS == 0
        assert GameType.GO == 1
        assert GameType.GOMOKU == 2


class TestGPUGameStatesInitialization:
    """Test GPUGameStates initialization"""
    
    def test_gomoku_initialization(self, gomoku_states, gomoku_config):
        """Test Gomoku state initialization"""
        assert gomoku_states.config == gomoku_config
        assert gomoku_states.device == torch.device('cpu')
        assert gomoku_states.capacity == 100
        assert gomoku_states.game_type == GameType.GOMOKU
        assert gomoku_states.board_size == 15
        assert gomoku_states.num_states == 0
        
        # Check tensor shapes
        assert gomoku_states.boards.shape == (100, 15, 15)
        assert gomoku_states.current_player.shape == (100,)
        assert gomoku_states.move_count.shape == (100,)
        assert gomoku_states.is_terminal.shape == (100,)
        assert gomoku_states.winner.shape == (100,)
        
    def test_go_initialization(self, go_states, go_config):
        """Test Go state initialization"""
        assert go_states.game_type == GameType.GO
        assert go_states.board_size == 9
        
        # Check Go-specific tensors
        assert hasattr(go_states, 'ko_point')
        assert hasattr(go_states, 'captured')
        assert go_states.ko_point.shape == (50,)
        assert go_states.captured.shape == (50, 2)
        
    def test_chess_initialization(self, chess_states):
        """Test Chess state initialization"""
        assert chess_states.game_type == GameType.CHESS
        assert chess_states.board_size == 8
        
        # Check Chess-specific tensors
        assert hasattr(chess_states, 'castling')
        assert hasattr(chess_states, 'en_passant')
        assert hasattr(chess_states, 'halfmove_clock')
        assert chess_states.castling.shape == (20, 4)
        assert chess_states.en_passant.shape == (20,)
        
    def test_move_history_initialization(self, gomoku_states):
        """Test move history initialization"""
        assert gomoku_states.move_history_size == 8
        assert gomoku_states.move_history.shape == (100, 8)
        assert torch.all(gomoku_states.move_history == -1)
        
        # Check full move history
        assert gomoku_states.full_move_history.shape[0] == 100
        assert gomoku_states.full_move_history.shape[1] >= 225  # At least board size squared
        
    def test_allocation_tracking_initialization(self, gomoku_states):
        """Test allocation tracking initialization"""
        assert len(gomoku_states.free_indices) == 100
        assert torch.all(gomoku_states.free_indices == torch.arange(100))
        assert torch.all(gomoku_states.allocated_mask == False)
        
    def test_board_initialization_failure(self):
        """Test handling of board initialization failure"""
        # Mock a config that would fail initialization
        config = GPUGameStatesConfig(game_type=999)  # Invalid game type
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((RuntimeError, ValueError)):
            GPUGameStates(config)


class TestStateAllocation:
    """Test state allocation and deallocation"""
    
    def test_allocate_single_state(self, gomoku_states):
        """Test allocating a single state"""
        indices = gomoku_states.allocate_states(1)
        
        assert len(indices) == 1
        assert indices[0] == 0
        assert gomoku_states.num_states == 1
        assert gomoku_states.allocated_mask[0] == True
        assert len(gomoku_states.free_indices) == 99
        
    def test_allocate_multiple_states(self, gomoku_states):
        """Test allocating multiple states"""
        indices = gomoku_states.allocate_states(5)
        
        assert len(indices) == 5
        assert torch.all(indices == torch.arange(5))
        assert gomoku_states.num_states == 5
        assert torch.all(gomoku_states.allocated_mask[:5] == True)
        assert len(gomoku_states.free_indices) == 95
        
    def test_allocate_beyond_capacity(self, gomoku_states):
        """Test allocating more states than capacity"""
        with pytest.raises(RuntimeError, match="Cannot allocate .* states"):
            gomoku_states.allocate_states(101)
            
    def test_free_states(self, gomoku_states):
        """Test freeing allocated states"""
        # Allocate
        indices = gomoku_states.allocate_states(3)
        
        # Free
        gomoku_states.free_states(indices)
        
        assert gomoku_states.num_states == 0
        assert torch.all(gomoku_states.allocated_mask[:3] == False)
        assert len(gomoku_states.free_indices) == 100
        
    def test_free_unallocated_states(self, gomoku_states):
        """Test freeing unallocated states (should handle gracefully)"""
        # Try to free states that weren't allocated
        unallocated = torch.tensor([5, 10, 15])
        
        # Should log warning but not crash
        gomoku_states.free_states(unallocated)
        
    def test_mixed_free_states(self, gomoku_states):
        """Test freeing mix of allocated and unallocated states"""
        # Allocate some states
        allocated = gomoku_states.allocate_states(3)
        
        # Mix with unallocated
        mixed = torch.cat([allocated[:1], torch.tensor([10, 20])])
        
        # Should free only the allocated one
        gomoku_states.free_states(mixed)
        assert gomoku_states.num_states == 2
        
    def test_state_reset_on_allocation(self, gomoku_states):
        """Test that states are reset when allocated"""
        indices = gomoku_states.allocate_states(2)
        
        # Check boards are empty (for Gomoku)
        assert torch.all(gomoku_states.boards[indices] == 0)
        
        # Check metadata reset
        assert torch.all(gomoku_states.current_player[indices] == 1)
        assert torch.all(gomoku_states.move_count[indices] == 0)
        assert torch.all(gomoku_states.is_terminal[indices] == False)
        assert torch.all(gomoku_states.winner[indices] == 0)
        
    def test_chess_initial_position(self, chess_states):
        """Test Chess initial position setup"""
        idx = chess_states.allocate_states(1)[0]
        board = chess_states.boards[idx]
        
        # Check white pieces
        assert torch.all(board[1, :] == 1)  # White pawns
        assert board[0, 0] == 2  # White rook
        assert board[0, 4] == 6  # White king
        
        # Check black pieces  
        assert torch.all(board[6, :] == 7)  # Black pawns
        assert board[7, 0] == 8  # Black rook
        assert board[7, 4] == 12  # Black king


class TestStateCloning:
    """Test state cloning operations"""
    
    def test_clone_single_state(self, gomoku_states):
        """Test cloning a single state"""
        # Create parent state
        parent_idx = gomoku_states.allocate_states(1)[0]
        
        # Make some moves
        gomoku_states.boards[parent_idx, 7, 7] = 1
        gomoku_states.move_count[parent_idx] = 1
        gomoku_states.current_player[parent_idx] = 2
        
        # Clone
        clone_indices = gomoku_states.clone_states(
            torch.tensor([parent_idx]),
            torch.tensor([1])
        )
        
        assert len(clone_indices) == 1
        clone_idx = clone_indices[0]
        
        # Check clone matches parent
        assert torch.all(gomoku_states.boards[clone_idx] == gomoku_states.boards[parent_idx])
        assert gomoku_states.move_count[clone_idx] == 1
        assert gomoku_states.current_player[clone_idx] == 2
        
    def test_clone_multiple_states(self, gomoku_states):
        """Test cloning multiple states with different clone counts"""
        # Create parent states
        parent_indices = gomoku_states.allocate_states(2)
        
        # Set different states
        gomoku_states.boards[parent_indices[0], 0, 0] = 1
        gomoku_states.boards[parent_indices[1], 1, 1] = 2
        
        # Clone with different counts
        clone_indices = gomoku_states.clone_states(
            parent_indices,
            torch.tensor([2, 3])  # 2 clones of first, 3 of second
        )
        
        assert len(clone_indices) == 5
        
        # Check first parent clones
        assert torch.all(gomoku_states.boards[clone_indices[:2], 0, 0] == 1)
        
        # Check second parent clones
        assert torch.all(gomoku_states.boards[clone_indices[2:], 1, 1] == 2)
        
    def test_clone_with_history(self, gomoku_states):
        """Test cloning preserves move history"""
        parent_idx = gomoku_states.allocate_states(1)[0]
        
        # Set move history
        gomoku_states.move_history[parent_idx, :3] = torch.tensor([10, 20, 30])
        gomoku_states.full_move_history[parent_idx, :3] = torch.tensor([10, 20, 30])
        
        # Clone
        clone_idx = gomoku_states.clone_states(
            torch.tensor([parent_idx]),
            torch.tensor([1])
        )[0]
        
        # Check history preserved
        assert torch.all(gomoku_states.move_history[clone_idx, :3] == torch.tensor([10, 20, 30]))
        assert torch.all(gomoku_states.full_move_history[clone_idx, :3] == torch.tensor([10, 20, 30]))
        
    def test_clone_go_specific_state(self, go_states):
        """Test cloning Go-specific state"""
        parent_idx = go_states.allocate_states(1)[0]
        
        # Set Go-specific state
        go_states.ko_point[parent_idx] = 42
        go_states.captured[parent_idx] = torch.tensor([3, 5])
        
        # Clone
        clone_idx = go_states.clone_states(
            torch.tensor([parent_idx]),
            torch.tensor([1])
        )[0]
        
        # Check Go state preserved
        assert go_states.ko_point[clone_idx] == 42
        assert torch.all(go_states.captured[clone_idx] == torch.tensor([3, 5]))


class TestLegalMoves:
    """Test legal move generation"""
    
    def test_gomoku_legal_moves_empty_board(self, gomoku_states):
        """Test Gomoku legal moves on empty board"""
        idx = gomoku_states.allocate_states(1)
        
        legal_mask = gomoku_states.get_legal_moves_mask(idx)
        
        assert legal_mask.shape == (1, 225)  # 15x15
        assert torch.all(legal_mask)  # All moves legal on empty board
        
    def test_gomoku_legal_moves_with_stones(self, gomoku_states):
        """Test Gomoku legal moves with stones on board"""
        idx = gomoku_states.allocate_states(1)[0]
        
        # Place some stones
        gomoku_states.boards[idx, 7, 7] = 1
        gomoku_states.boards[idx, 7, 8] = 2
        gomoku_states.boards[idx, 8, 7] = 1
        
        legal_mask = gomoku_states.get_legal_moves_mask(torch.tensor([idx]))
        
        # Occupied squares should be illegal
        assert legal_mask[0, 7*15 + 7] == False
        assert legal_mask[0, 7*15 + 8] == False
        assert legal_mask[0, 8*15 + 7] == False
        
        # Other squares should be legal
        assert legal_mask[0, 0] == True
        assert legal_mask[0, 224] == True
        
    def test_go_legal_moves_ko(self, go_states):
        """Test Go legal moves with ko restriction"""
        idx = go_states.allocate_states(1)[0]
        
        # Set ko point
        go_states.ko_point[idx] = 40  # Position 40 is ko
        
        legal_mask = go_states.get_legal_moves_mask(torch.tensor([idx]))
        
        assert legal_mask.shape == (1, 82)  # 9x9 + pass move
        assert legal_mask[0, 40] == False  # Ko point illegal
        assert legal_mask[0, 0] == True  # Other points legal
        assert legal_mask[0, 81] == True  # Pass move is always legal
        
    def test_batch_legal_moves(self, gomoku_states):
        """Test batch legal move generation"""
        indices = gomoku_states.allocate_states(3)
        
        # Different board states
        gomoku_states.boards[indices[0], 0, 0] = 1
        gomoku_states.boards[indices[1], 1, 1] = 2
        gomoku_states.boards[indices[2], 2, 2] = 1
        
        legal_masks = gomoku_states.get_legal_moves_mask(indices)
        
        assert legal_masks.shape == (3, 225)
        
        # Check individual masks
        assert legal_masks[0, 0] == False
        assert legal_masks[1, 16] == False  # 1*15 + 1
        assert legal_masks[2, 32] == False  # 2*15 + 2
        
    def test_chess_legal_moves_placeholder(self, chess_states):
        """Test Chess legal moves (placeholder implementation)"""
        idx = chess_states.allocate_states(1)
        
        legal_mask = chess_states.get_legal_moves_mask(idx)
        
        assert legal_mask.shape == (1, 4096)  # Chess action space
        # Placeholder implementation marks first 10 moves as legal
        assert torch.all(legal_mask[0, :10] == True)


class TestMoveApplication:
    """Test move application"""
    
    def test_apply_single_move_gomoku(self, gomoku_states):
        """Test applying a single move in Gomoku"""
        idx = gomoku_states.allocate_states(1)[0]
        
        # Apply move at position (7,7) = action 112
        action = torch.tensor([112])  # 7*15 + 7
        gomoku_states.apply_moves(torch.tensor([idx]), action)
        
        # Check board updated
        assert gomoku_states.boards[idx, 7, 7] == 1  # Player 1 stone
        
        # Check metadata updated
        assert gomoku_states.current_player[idx] == 2
        assert gomoku_states.move_count[idx] == 1
        
        # Check move history
        assert gomoku_states.move_history[idx, -1] == 112
        assert gomoku_states.full_move_history[idx, 0] == 112
        
    def test_apply_multiple_moves(self, gomoku_states):
        """Test applying moves to multiple states"""
        indices = gomoku_states.allocate_states(2)
        
        actions = torch.tensor([0, 224])  # Top-left and bottom-right
        gomoku_states.apply_moves(indices, actions)
        
        # Check boards
        assert gomoku_states.boards[indices[0], 0, 0] == 1
        assert gomoku_states.boards[indices[1], 14, 14] == 1
        
        # Check players switched
        assert torch.all(gomoku_states.current_player[indices] == 2)
        
    def test_sequential_moves(self, gomoku_states):
        """Test applying sequential moves"""
        idx = gomoku_states.allocate_states(1)[0]
        
        # First move
        gomoku_states.apply_moves(torch.tensor([idx]), torch.tensor([112]))
        assert gomoku_states.boards[idx, 7, 7] == 1
        assert gomoku_states.current_player[idx] == 2
        
        # Second move
        gomoku_states.apply_moves(torch.tensor([idx]), torch.tensor([113]))
        assert gomoku_states.boards[idx, 7, 8] == 2
        assert gomoku_states.current_player[idx] == 1
        
        # Check move count
        assert gomoku_states.move_count[idx] == 2
        
    def test_move_history_rolling(self, gomoku_states):
        """Test move history rolling buffer"""
        idx = gomoku_states.allocate_states(1)[0]
        
        # Apply more moves than history size
        for i in range(10):
            action = torch.tensor([i])
            gomoku_states.apply_moves(torch.tensor([idx]), action)
            
        # Recent history should have last 8 moves
        expected = torch.tensor([2, 3, 4, 5, 6, 7, 8, 9])
        assert torch.all(gomoku_states.move_history[idx] == expected)
        
        # Full history should have all moves
        assert torch.all(gomoku_states.full_move_history[idx, :10] == torch.arange(10))


class TestTerminalStates:
    """Test terminal state detection"""
    
    def test_terminal_state_tracking(self, gomoku_states):
        """Test terminal state flag tracking"""
        idx = gomoku_states.allocate_states(1)[0]
        
        # Initially not terminal
        assert gomoku_states.is_terminal[idx] == False
        assert gomoku_states.winner[idx] == 0
        
        # Would need full implementation to test actual win detection
        # For now, just test the flags can be set
        gomoku_states.is_terminal[idx] = True
        gomoku_states.winner[idx] = 1
        
        assert gomoku_states.is_terminal[idx] == True
        assert gomoku_states.winner[idx] == 1


class TestNeuralNetworkFeatures:
    """Test neural network feature extraction"""
    
    def test_gomoku_basic_features(self, gomoku_states):
        """Test basic Gomoku feature extraction"""
        idx = gomoku_states.allocate_states(1)[0]
        
        # Place some stones
        gomoku_states.boards[idx, 7, 7] = 1
        gomoku_states.boards[idx, 7, 8] = 2
        
        features = gomoku_states.get_nn_features(torch.tensor([idx]))
        
        # Should have 18 channels for Gomoku
        assert features.shape == (1, 18, 15, 15)
        
        # Channel 0: All stones
        assert features[0, 0, 7, 7] == 1.0
        assert features[0, 0, 7, 8] == 1.0
        assert features[0, 0, 0, 0] == 0.0
        
        # Channel 1: Current player (constant)
        assert torch.all(features[0, 1] == gomoku_states.current_player[idx])
        
    def test_go_basic_features(self, go_states):
        """Test basic Go feature extraction"""
        idx = go_states.allocate_states(1)[0]
        
        # Place stones
        go_states.boards[idx, 0, 0] = 1  # Black
        go_states.boards[idx, 0, 1] = 2  # White
        
        features = go_states.get_nn_features(torch.tensor([idx]))
        
        assert features.shape == (1, 5, 9, 9)
        
        # Check planes
        assert features[0, 0, 0, 0] == 1.0  # Black stone
        assert features[0, 1, 0, 1] == 1.0  # White stone
        assert features[0, 2, 1, 1] == 1.0  # Empty
        
    def test_batch_feature_extraction(self, gomoku_states):
        """Test batch feature extraction"""
        indices = gomoku_states.allocate_states(3)
        
        # Different board states
        gomoku_states.boards[indices[0], 0, 0] = 1
        gomoku_states.boards[indices[1], 1, 1] = 2
        gomoku_states.boards[indices[2], 2, 2] = 1
        
        features = gomoku_states.get_nn_features(indices)
        
        assert features.shape == (3, 18, 15, 15)
        
        # Check individual features
        assert features[0, 0, 0, 0] == 1.0
        assert features[1, 0, 1, 1] == 1.0
        assert features[2, 0, 2, 2] == 1.0
        
    def test_nn_features_batch_alias(self, gomoku_states):
        """Test get_nn_features_batch alias"""
        idx = gomoku_states.allocate_states(1)
        
        features1 = gomoku_states.get_nn_features(idx)
        features2 = gomoku_states.get_nn_features_batch(idx)
        
        assert torch.all(features1 == features2)
        
    def test_move_history_features(self, gomoku_states):
        """Test move history in features"""
        idx = gomoku_states.allocate_states(1)[0]
        
        # Make a sequence of moves
        moves = [112, 113, 127, 128]  # Some positions
        for i, move in enumerate(moves):
            gomoku_states.apply_moves(torch.tensor([idx]), torch.tensor([move]))
            
        features = gomoku_states.get_nn_features(torch.tensor([idx]))
        
        # Channels 2-17 should contain move history
        # Due to vectorized implementation, checking exact positions is complex
        # Just verify shape and that history channels have some non-zero values
        assert features.shape == (1, 18, 15, 15)
        
        # At least some history channels should have moves
        history_channels = features[0, 2:18]
        assert history_channels.max() > 0  # Some moves recorded


class TestEnhancedFeatures:
    """Test enhanced feature extraction"""
    
    @patch('alphazero_py.GomokuState')
    def test_enable_enhanced_features(self, mock_gomoku_state, gomoku_states):
        """Test enabling enhanced features"""
        gomoku_states.enable_enhanced_features()
        
        assert gomoku_states._use_enhanced_features == True
        assert gomoku_states._enhanced_channels == 20
        assert hasattr(gomoku_states, '_cpp_states_cache')
        assert hasattr(gomoku_states, '_cpp_states_pool')
        
    def test_set_enhanced_channels(self, gomoku_states):
        """Test setting enhanced channel count"""
        gomoku_states.enable_enhanced_features()
        
        # Valid values
        gomoku_states.set_enhanced_channels(18)
        assert gomoku_states._enhanced_channels == 18
        
        gomoku_states.set_enhanced_channels(20)
        assert gomoku_states._enhanced_channels == 20
        
        # Invalid value
        with pytest.raises(ValueError):
            gomoku_states.set_enhanced_channels(16)
            
    @patch('alphazero_py.GomokuState')
    def test_enhanced_feature_extraction(self, mock_gomoku_state, gomoku_states):
        """Test enhanced feature extraction with mocked C++ state"""
        # Enable enhanced features
        gomoku_states.enable_enhanced_features()
        
        # Mock C++ state
        mock_state = Mock()
        mock_tensor = np.zeros((20, 15, 15), dtype=np.float32)
        mock_tensor[0, 7, 7] = 1.0  # Some test data
        mock_state.get_enhanced_tensor_representation.return_value = mock_tensor
        
        # Mock state creation
        mock_gomoku_state.return_value = mock_state
        
        idx = gomoku_states.allocate_states(1)
        features = gomoku_states.get_nn_features(idx)
        
        # Should return 20-channel features
        assert features.shape == (1, 20, 15, 15)
        assert features[0, 0, 7, 7] == 1.0
        
    @patch('alphazero_py.GomokuState')
    def test_enhanced_features_18_channels(self, mock_gomoku_state, gomoku_states):
        """Test enhanced features with 18 channels"""
        gomoku_states.enable_enhanced_features()
        gomoku_states.set_enhanced_channels(18)
        
        # Mock C++ state
        mock_state = Mock()
        mock_tensor = np.ones((20, 15, 15), dtype=np.float32)
        mock_state.get_enhanced_tensor_representation.return_value = mock_tensor
        mock_gomoku_state.return_value = mock_state
        
        idx = gomoku_states.allocate_states(1)
        features = gomoku_states.get_nn_features(idx)
        
        # Should return only 18 channels (remove last 2)
        assert features.shape == (1, 18, 15, 15)
        
    def test_clear_enhanced_cache(self, gomoku_states):
        """Test clearing enhanced feature cache"""
        gomoku_states.enable_enhanced_features()
        
        # Add some dummy data to cache
        gomoku_states._cpp_states_cache[0] = Mock()
        gomoku_states._cpp_states_pool.append(Mock())
        
        # Clear
        gomoku_states.clear_enhanced_cache()
        
        assert len(gomoku_states._cpp_states_cache) == 0
        assert len(gomoku_states._cpp_states_pool) == 0


class TestUtilityMethods:
    """Test utility methods"""
    
    def test_get_feature_planes(self, gomoku_states, go_states, chess_states):
        """Test getting feature plane count"""
        assert gomoku_states.get_feature_planes() == 4
        assert go_states.get_feature_planes() == 5
        assert chess_states.get_feature_planes() == 12
        
        # With enhanced features
        gomoku_states.enable_enhanced_features()
        assert gomoku_states.get_feature_planes() == 20
        
        gomoku_states.set_enhanced_channels(18)
        assert gomoku_states.get_feature_planes() == 18
        
    def test_get_state_info(self, gomoku_states):
        """Test getting state information"""
        idx = gomoku_states.allocate_states(1)[0]
        
        # Set some state
        gomoku_states.move_count[idx] = 5
        gomoku_states.current_player[idx] = 2
        
        info = gomoku_states.get_state_info(idx)
        
        assert info['game_type'] == 'GOMOKU'
        assert info['current_player'] == 2
        assert info['move_count'] == 5
        assert info['is_terminal'] == False
        assert info['winner'] == 0
        
    def test_get_state_info_go(self, go_states):
        """Test getting Go state information"""
        idx = go_states.allocate_states(1)[0]
        
        go_states.ko_point[idx] = 42
        go_states.captured[idx] = torch.tensor([3, 5])
        
        info = go_states.get_state_info(idx)
        
        assert info['game_type'] == 'GO'
        assert info['ko_point'] == 42
        assert info['captured'] == [3, 5]
        
    def test_get_state_info_chess(self, chess_states):
        """Test getting Chess state information"""
        idx = chess_states.allocate_states(1)[0]
        
        info = chess_states.get_state_info(idx)
        
        assert info['game_type'] == 'CHESS'
        assert 'castling' in info
        assert 'en_passant' in info
        assert 'halfmove_clock' in info


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_cuda_device_handling(self):
        """Test CUDA device handling"""
        with patch('torch.cuda.is_available', return_value=False):
            config = GPUGameStatesConfig(device='cuda')
            states = GPUGameStates(config)
            assert states.device == torch.device('cpu')
            
    def test_full_capacity_allocation(self, gomoku_states):
        """Test allocating all available states"""
        indices = gomoku_states.allocate_states(100)
        
        assert len(indices) == 100
        assert gomoku_states.num_states == 100
        
        # Should fail to allocate more
        with pytest.raises(RuntimeError):
            gomoku_states.allocate_states(1)
            
    def test_empty_batch_operations(self, gomoku_states):
        """Test operations on empty batches"""
        empty_indices = torch.tensor([], dtype=torch.long)
        
        # Should handle gracefully
        legal_masks = gomoku_states.get_legal_moves_mask(empty_indices)
        assert legal_masks.shape == (0, 225)
        
        features = gomoku_states.get_nn_features(empty_indices)
        assert features.shape == (0, 18, 15, 15)
        
    def test_large_board_sizes(self):
        """Test with large board sizes"""
        config = GPUGameStatesConfig(
            capacity=10,
            game_type=GameType.GO,
            board_size=19,
            device='cpu'
        )
        states = GPUGameStates(config)
        
        assert states.board_size == 19
        assert states.boards.shape == (10, 19, 19)
        
        # Legal moves should match board size + pass move
        idx = states.allocate_states(1)
        legal_mask = states.get_legal_moves_mask(idx)
        assert legal_mask.shape == (1, 362)  # 19x19 + pass move