"""Test suite for MCTS refactoring to ensure functionality is preserved"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock
import sys
sys.path.append('/home/cosmosapjw/omoknuni_quantum/python')

from mcts.core.mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType
from mcts.core.game_interface import GameInterface, GameType as LegacyGameType


class MockEvaluator:
    """Mock evaluator for testing"""
    def __init__(self):
        self._return_torch_tensors = False
        
    def evaluate(self, states):
        """Return mock policy and value"""
        if isinstance(states, torch.Tensor):
            batch_size = states.shape[0]
        else:
            batch_size = len(states)
            
        # Mock policy (uniform distribution)
        policy = torch.ones((batch_size, 225)) / 225  # 15x15 board
        value = torch.zeros(batch_size)
        
        if self._return_torch_tensors:
            return policy, value
        else:
            return policy.numpy(), value.numpy()
            
    def evaluate_batch(self, states):
        """Alias for evaluate to support batch evaluation"""
        return self.evaluate(states)


class MockGameState:
    """Mock game state for testing"""
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1
        self.move_count = 0
        
    def get_tensor_representation(self):
        """Return tensor representation"""
        current = (self.board == self.current_player).astype(np.float32)
        opponent = (self.board == -self.current_player).astype(np.float32)
        return np.stack([current, opponent])
        
    def get_current_player(self):
        return self.current_player
        
    def get_legal_actions(self):
        """Return all empty positions as legal actions"""
        return np.where(self.board.flatten() == 0)[0].tolist()
        
    def is_terminal(self):
        return False
        
    def get_winner(self):
        return None
        
    def get_move_history(self):
        return []
        
    def apply_action(self, action):
        """Apply action and return new state"""
        new_state = MockGameState(self.board_size)
        new_state.board = self.board.copy()
        row, col = divmod(action, self.board_size)
        new_state.board[row, col] = self.current_player
        new_state.current_player = -self.current_player
        new_state.move_count = self.move_count + 1
        return new_state


class TestMCTSBasicFunctionality:
    """Test basic MCTS functionality"""
    
    @pytest.fixture
    def basic_config(self):
        """Create basic MCTS config"""
        return MCTSConfig(
            num_simulations=100,
            c_puct=1.4,
            device='cpu',  # Use CPU for testing
            game_type=GameType.GOMOKU,
            board_size=15,
            enable_quantum=False,
            classical_only_mode=True,
            max_wave_size=32,
            enable_debug_logging=False
        )
        
    @pytest.fixture
    def mcts_instance(self, basic_config):
        """Create MCTS instance"""
        evaluator = MockEvaluator()
        return MCTS(basic_config, evaluator)
        
    def test_initialization(self, mcts_instance):
        """Test MCTS initialization"""
        assert mcts_instance is not None
        assert mcts_instance.config is not None
        assert mcts_instance.tree is not None
        assert mcts_instance.game_states is not None
        
    def test_search_basic(self, mcts_instance):
        """Test basic search functionality"""
        state = MockGameState()
        policy = mcts_instance.search(state, num_simulations=10)
        
        assert policy is not None
        assert isinstance(policy, np.ndarray)
        assert policy.shape == (225,)  # 15x15 board
        assert np.allclose(policy.sum(), 1.0)  # Should sum to 1
        
    def test_select_action(self, mcts_instance):
        """Test action selection"""
        state = MockGameState()
        action = mcts_instance.select_action(state, temperature=1.0)
        
        assert isinstance(action, int)
        assert 0 <= action < 225  # Valid board position
        
    def test_get_action_probabilities(self, mcts_instance):
        """Test getting action probabilities"""
        state = MockGameState()
        probs, action = mcts_instance.get_action_probabilities(state, temperature=1.0)
        
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (225,)
        assert np.allclose(probs.sum(), 1.0)
        assert isinstance(action, int)
        assert 0 <= action < 225
        

class TestMCTSConfiguration:
    """Test MCTS configuration handling"""
    
    def test_config_defaults(self):
        """Test default configuration values"""
        config = MCTSConfig()
        assert config.num_simulations == 10000
        assert config.c_puct == 1.414
        assert config.temperature == 1.0
        assert config.device == 'cuda'
        assert config.game_type == GameType.GOMOKU
        assert config.board_size == 15
        
    def test_config_game_type_conversion(self):
        """Test legacy game type conversion"""
        config = MCTSConfig(game_type=LegacyGameType.CHESS)
        config.__post_init__()
        assert config.game_type == GameType.CHESS
        
    def test_config_board_size_defaults(self):
        """Test board size defaults for different games"""
        # Chess
        config = MCTSConfig(game_type=GameType.CHESS, board_size=None)
        config.__post_init__()
        assert config.board_size == 8
        
        # Go
        config = MCTSConfig(game_type=GameType.GO, board_size=None)
        config.__post_init__()
        assert config.board_size == 19
        
        # Gomoku
        config = MCTSConfig(game_type=GameType.GOMOKU, board_size=None)
        config.__post_init__()
        assert config.board_size == 15


class TestMCTSTreeOperations:
    """Test MCTS tree operations"""
    
    @pytest.fixture
    def mcts_with_tree(self, basic_config):
        """Create MCTS with initialized tree"""
        evaluator = MockEvaluator()
        mcts = MCTS(basic_config, evaluator)
        state = MockGameState()
        mcts._initialize_root(state)
        return mcts, state
        
    def test_tree_initialization(self, mcts_with_tree):
        """Test tree is properly initialized"""
        mcts, state = mcts_with_tree
        assert mcts.tree.num_nodes > 0
        assert mcts.tree.parent_indices[0] == -1  # Root has no parent
        
    def test_clear_operation(self, mcts_with_tree):
        """Test clearing the tree"""
        mcts, state = mcts_with_tree
        # Run a search to populate the tree
        mcts.search(state, num_simulations=10)
        nodes_before = mcts.tree.num_nodes
        
        # Clear the tree
        mcts.clear()
        
        # Verify tree is reset
        assert mcts.tree.num_nodes == 1  # Only root node
        
    def test_update_root(self, mcts_with_tree):
        """Test updating root with new action"""
        mcts, state = mcts_with_tree
        # Run initial search
        mcts.search(state, num_simulations=10)
        
        # Select an action and update root
        action = 0  # Top-left corner
        new_state = state.apply_action(action)
        mcts.update_root(action, new_state)
        
        # Verify root is updated
        assert mcts.node_to_state[0] >= 0  # Root has a state


class TestMCTSWaveParallelization:
    """Test wave-based parallelization"""
    
    @pytest.fixture
    def wave_config(self):
        """Config with wave parallelization"""
        return MCTSConfig(
            num_simulations=100,
            device='cpu',
            min_wave_size=32,
            max_wave_size=32,
            classical_only_mode=True
        )
        
    def test_wave_size_configuration(self, wave_config):
        """Test wave size is properly configured"""
        evaluator = MockEvaluator()
        mcts = MCTS(wave_config, evaluator)
        
        # Check buffers are allocated for wave size
        assert mcts.paths_buffer.shape[0] == wave_config.max_wave_size
        assert mcts.path_lengths.shape[0] == wave_config.max_wave_size
        
    def test_wave_search(self, wave_config):
        """Test search with wave parallelization"""
        evaluator = MockEvaluator()
        mcts = MCTS(wave_config, evaluator)
        state = MockGameState()
        
        # Run search with waves
        policy = mcts.search(state, num_simulations=64)  # 2 waves of 32
        
        assert policy is not None
        assert mcts.stats['total_simulations'] >= 64


class TestMCTSStatistics:
    """Test MCTS statistics tracking"""
    
    def test_statistics_tracking(self, basic_config):
        """Test statistics are properly tracked"""
        evaluator = MockEvaluator()
        mcts = MCTS(basic_config, evaluator)
        state = MockGameState()
        
        # Run search
        mcts.search(state, num_simulations=10)
        
        stats = mcts.get_statistics()
        assert 'total_searches' in stats
        assert 'total_simulations' in stats
        assert 'tree_nodes' in stats
        assert stats['total_searches'] == 1
        assert stats['total_simulations'] == 10
        

if __name__ == "__main__":
    pytest.main([__file__, "-v"])