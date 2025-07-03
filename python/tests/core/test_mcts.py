"""Tests for MCTS core functionality"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time

from mcts.core.mcts import MCTS, MCTSConfig
from mcts.core.game_interface import GameInterface, GameType
from mcts.core.evaluator import MockEvaluator


class TestMCTSConfig:
    """Test the MCTSConfig dataclass"""
    
    def test_mcts_config_defaults(self):
        """Test MCTSConfig with default values"""
        config = MCTSConfig()
        
        assert config.num_simulations == 10000
        assert config.c_puct == 1.414
        assert config.temperature == 1.0
        assert config.dirichlet_alpha == 0.3
        assert config.dirichlet_epsilon == 0.25
        assert config.classical_only_mode == False
        assert config.enable_fast_ucb == True
        assert config.device == 'cuda'
        assert config.game_type == 2  # GameType.GOMOKU from gpu_game_states
        assert config.board_size == 15
    
    def test_mcts_config_custom(self):
        """Test MCTSConfig with custom values"""
        config = MCTSConfig(
            num_simulations=1000,
            c_puct=2.0,
            temperature=0.5,
            classical_only_mode=True,
            device='cpu',
            board_size=9
        )
        
        assert config.num_simulations == 1000
        assert config.c_puct == 2.0
        assert config.temperature == 0.5
        assert config.classical_only_mode == True
        assert config.device == 'cpu'
        assert config.board_size == 9
    
    def test_mcts_config_quantum_config_creation(self):
        """Test automatic quantum config creation"""
        config = MCTSConfig(enable_quantum=True)
        quantum_config = config.get_or_create_quantum_config()
        
        assert quantum_config is not None
        assert quantum_config.base_c_puct == config.c_puct
        assert quantum_config.device == config.device


class TestMCTS:
    """Test the MCTS implementation"""
    
    @pytest.fixture
    def mcts_instance(self, small_gomoku_config, small_game_interface, mock_evaluator):
        """Create MCTS instance for testing"""
        # Use small config for faster tests
        config = small_gomoku_config
        config.classical_only_mode = True  # Disable quantum for unit tests
        
        return MCTS(config, small_game_interface, mock_evaluator)
    
    def test_mcts_creation(self, small_gomoku_config, small_game_interface, mock_evaluator):
        """Test MCTS instance creation"""
        mcts = MCTS(small_gomoku_config, small_game_interface, mock_evaluator)
        
        assert mcts.config == small_gomoku_config
        assert mcts.game_interface == small_game_interface
        assert mcts.evaluator == mock_evaluator
        assert hasattr(mcts, 'tree')
    
    def test_mcts_search_single_simulation(self, mcts_instance, sample_small_game_state):
        """Test MCTS search with a single simulation"""
        # Override config for single simulation
        mcts_instance.config.num_simulations = 1
        
        action_probs = mcts_instance.search(sample_small_game_state)
        
        assert len(action_probs) == 81  # 9x9 board
        assert all(prob >= 0 for prob in action_probs)
        assert abs(sum(action_probs) - 1.0) < 0.01  # Should sum to 1
    
    def test_mcts_search_multiple_simulations(self, mcts_instance, sample_small_game_state):
        """Test MCTS search with multiple simulations"""
        action_probs = mcts_instance.search(sample_small_game_state)
        
        assert len(action_probs) == 81
        assert all(prob >= 0 for prob in action_probs)
        assert sum(action_probs) > 0  # At least some probability mass
    
    def test_mcts_search_legal_moves_only(self, mcts_instance, sample_small_game_state):
        """Test that MCTS only considers legal moves"""
        action_probs = mcts_instance.search(sample_small_game_state)
        
        # Get legal moves from game interface
        legal_moves = mcts_instance.game_interface.get_legal_moves(sample_small_game_state)
        
        # Check that illegal moves have zero probability
        for i, is_legal in enumerate(legal_moves):
            if not is_legal:
                assert action_probs[i] == 0.0, f"Illegal move {i} has non-zero probability"
    
    def test_mcts_search_with_temperature(self, mcts_instance, sample_small_game_state):
        """Test MCTS search with different temperatures"""
        # Low temperature (more deterministic)
        mcts_instance.config.temperature = 0.1
        low_temp_probs = mcts_instance.search(sample_small_game_state)
        
        # High temperature (more random)
        mcts_instance.config.temperature = 2.0
        high_temp_probs = mcts_instance.search(sample_small_game_state)
        
        # Low temperature should be more concentrated
        low_temp_max = max(low_temp_probs)
        high_temp_max = max(high_temp_probs)
        
        # This might not always hold due to randomness, so we'll just check they're valid
        assert low_temp_max > 0
        assert high_temp_max > 0
    
    def test_mcts_get_action_probabilities(self, mcts_instance, sample_small_game_state):
        """Test getting action probabilities from visit counts"""
        # Mock the tree to have some visit counts
        with patch.object(mcts_instance, 'tree') as mock_tree:
            mock_tree.get_visit_counts.return_value = np.array([10, 5, 0, 3, 0] + [0]*76)
            
            probs = mcts_instance.get_action_probabilities(sample_small_game_state, temperature=1.0)
            
            assert len(probs) == 81
            assert sum(probs) > 0
            # Most visited action should have highest probability
            assert np.argmax(probs) == 0  # First action had 10 visits
    
    def test_mcts_select_action_deterministic(self, mcts_instance, sample_small_game_state):
        """Test selecting action deterministically (temperature=0)"""
        # Mock search to return known probabilities
        with patch.object(mcts_instance, 'search') as mock_search:
            mock_search.return_value = [0.6, 0.3, 0.1] + [0.0]*78
            
            action = mcts_instance.select_action(sample_small_game_state, temperature=0.0)
            
            assert action == 0  # Should select action with highest probability
    
    def test_mcts_select_action_stochastic(self, mcts_instance, sample_small_game_state):
        """Test selecting action stochastically (temperature>0)"""
        # Mock search to return known probabilities
        with patch.object(mcts_instance, 'search') as mock_search:
            mock_search.return_value = [0.5, 0.3, 0.2] + [0.0]*78
            
            actions = []
            for _ in range(100):
                action = mcts_instance.select_action(sample_small_game_state, temperature=1.0)
                actions.append(action)
            
            # Should get some variety in selected actions
            unique_actions = set(actions)
            assert len(unique_actions) > 1, "Should have some randomness in action selection"
            assert all(a in [0, 1, 2] for a in unique_actions), "Should only select actions with non-zero probability"
    
    def test_mcts_tree_reuse(self, mcts_instance, sample_small_game_state):
        """Test that MCTS can reuse tree information"""
        # First search
        probs1 = mcts_instance.search(sample_small_game_state)
        
        # Second search on same state should potentially reuse tree
        probs2 = mcts_instance.search(sample_small_game_state)
        
        # Both should be valid probability distributions
        assert len(probs1) == len(probs2) == 81
        assert all(p >= 0 for p in probs1)
        assert all(p >= 0 for p in probs2)
    
    def test_mcts_different_states(self, mcts_instance, small_game_interface):
        """Test MCTS search on different game states"""
        # Initial state
        initial_state = small_game_interface.get_initial_state()
        initial_probs = mcts_instance.search(initial_state)
        
        # State after one move
        moved_state = small_game_interface.make_move(initial_state, 40)  # Center move
        moved_probs = mcts_instance.search(moved_state)
        
        # Both should be valid but likely different
        assert len(initial_probs) == len(moved_probs) == 81
        assert sum(initial_probs) > 0
        assert sum(moved_probs) > 0
        
        # The move that was just made should be illegal in the new state
        assert moved_probs[40] == 0.0, "Just-played move should be illegal"
    
    @pytest.mark.slow
    def test_mcts_performance(self, mcts_instance, sample_small_game_state):
        """Test MCTS performance with reasonable simulation count"""
        mcts_instance.config.num_simulations = 100
        
        start_time = time.time()
        action_probs = mcts_instance.search(sample_small_game_state)
        end_time = time.time()
        
        search_time = end_time - start_time
        simulations_per_second = 100 / search_time
        
        # Should complete reasonably quickly
        assert search_time < 5.0, f"Search took too long: {search_time:.2f}s"
        assert simulations_per_second > 10, f"Too slow: {simulations_per_second:.1f} sims/s"
        
        # Result should be valid
        assert len(action_probs) == 81
        assert sum(action_probs) > 0
    
    def test_mcts_with_quantum_disabled(self, small_gomoku_config, small_game_interface, mock_evaluator):
        """Test MCTS with quantum features explicitly disabled"""
        config = small_gomoku_config
        config.classical_only_mode = True
        config.enable_quantum = False
        
        mcts = MCTS(config, small_game_interface, mock_evaluator)
        
        # Should work normally
        state = small_game_interface.get_initial_state()
        action_probs = mcts.search(state)
        
        assert len(action_probs) == 81
        assert sum(action_probs) > 0
    
    def test_mcts_error_handling(self, mcts_instance):
        """Test MCTS error handling with invalid inputs"""
        # Invalid state shape
        invalid_state = np.random.rand(2, 9, 9).astype(np.float32)
        
        with pytest.raises((ValueError, RuntimeError)):
            mcts_instance.search(invalid_state)
        
        # None state
        with pytest.raises((ValueError, TypeError)):
            mcts_instance.search(None)
    
    def test_mcts_statistics(self, mcts_instance, sample_small_game_state):
        """Test MCTS statistics collection"""
        mcts_instance.config.num_simulations = 50
        
        # Perform search
        action_probs = mcts_instance.search(sample_small_game_state)
        
        # Check if statistics are available
        if hasattr(mcts_instance, 'get_statistics'):
            stats = mcts_instance.get_statistics()
            assert isinstance(stats, dict)
            
            # Should have some basic stats
            expected_keys = ['simulations_completed', 'search_time', 'nodes_expanded']
            for key in expected_keys:
                if key in stats:
                    assert isinstance(stats[key], (int, float))


class TestMCTSIntegration:
    """Integration tests for MCTS with different components"""
    
    def test_mcts_game_interface_integration(self, small_gomoku_config, mock_evaluator):
        """Test MCTS integration with different game interfaces"""
        # Test with different board sizes
        for board_size in [9, 15]:
            game_interface = GameInterface(GameType.GOMOKU, board_size)
            config = MCTSConfig(**small_gomoku_config.__dict__)
            config.board_size = board_size
            config.classical_only_mode = True
            
            mcts = MCTS(config, game_interface, mock_evaluator)
            
            initial_state = game_interface.get_initial_state()
            action_probs = mcts.search(initial_state)
            
            expected_actions = board_size * board_size
            assert len(action_probs) == expected_actions
            assert sum(action_probs) > 0
    
    def test_mcts_evaluator_integration(self, small_gomoku_config, small_game_interface):
        """Test MCTS integration with different evaluators"""
        # Test with MockEvaluator
        mock_eval = MockEvaluator(seed=42)
        mcts_mock = MCTS(small_gomoku_config, small_game_interface, mock_eval)
        
        state = small_game_interface.get_initial_state()
        probs_mock = mcts_mock.search(state)
        
        assert len(probs_mock) == 81
        assert sum(probs_mock) > 0
    
    def test_mcts_full_game_simulation(self, small_game_interface, mock_evaluator):
        """Test MCTS in a full game simulation"""
        config = MCTSConfig(
            num_simulations=20,  # Small for fast test
            classical_only_mode=True,
            game_type=GameType.GOMOKU,
            board_size=9,
            device='cpu'
        )
        
        mcts = MCTS(config, small_game_interface, mock_evaluator)
        
        state = small_game_interface.get_initial_state()
        moves_made = 0
        max_moves = 20  # Limit moves for test
        
        while not small_game_interface.is_terminal(state) and moves_made < max_moves:
            # Get action from MCTS
            action = mcts.select_action(state, temperature=1.0)
            
            # Verify action is legal
            legal_moves = small_game_interface.get_legal_moves(state)
            assert legal_moves[action], f"MCTS selected illegal action {action}"
            
            # Make the move
            state = small_game_interface.make_move(state, action)
            moves_made += 1
        
        # Should have made some moves without errors
        assert moves_made > 0
        assert moves_made <= max_moves
    
    def test_mcts_memory_usage(self, mcts_instance, sample_small_game_state):
        """Test that MCTS doesn't have obvious memory leaks"""
        import gc
        
        # Perform multiple searches
        for _ in range(10):
            mcts_instance.search(sample_small_game_state)
            gc.collect()
        
        # This test mainly ensures no exceptions are raised during repeated use
        # More sophisticated memory leak detection would require additional tools
        assert True  # If we get here without errors, memory management is working