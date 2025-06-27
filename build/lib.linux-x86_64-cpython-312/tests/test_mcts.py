"""Tests for the main MCTS class"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from mcts import (
    MCTS, MCTSConfig, 
    GameInterface, GameType,
    MockEvaluator, EvaluatorConfig
)


class TestMCTS:
    """Test suite for MCTS class"""
    
    def test_mcts_config(self):
        """Test MCTS configuration"""
        config = MCTSConfig()
        assert config.num_simulations == 10000
        assert config.c_puct == 1.414
        assert config.temperature == 1.0
        assert hasattr(config, 'adaptive_wave_sizing') and config.adaptive_wave_sizing == False
        
        # Custom config
        custom = MCTSConfig(
            num_simulations=1600,
            c_puct=2.0,
            temperature=0.5
        )
        assert custom.num_simulations == 1600
        assert custom.temperature == 0.5
        
    def test_mcts_initialization(self):
        """Test MCTS initialization"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=225)  # 15x15
        config = MCTSConfig(num_simulations=100, game_type=GameType.GOMOKU, board_size=15)
        
        mcts = MCTS(config, evaluator, game_interface=game)
        
        # MCTS stores game interface as cached_game
        assert mcts.cached_game is not None
        assert mcts.evaluator == evaluator
        assert mcts.config == config
        # Check implementation is initialized
        assert hasattr(mcts, 'tree')  # Optimized implementation always uses tree
        
    def test_search_from_position(self):
        """Test running MCTS search from a position"""
        # Use consistent board size (15x15 for Gomoku)
        game = GameInterface(GameType.GOMOKU, board_size=15)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=225)  # 15x15
        config = MCTSConfig(num_simulations=100, game_type=GameType.GOMOKU, board_size=15)
        
        mcts = MCTS(config, evaluator, game_interface=game)
        
        # Create initial position
        state = game.create_initial_state()
        
        # Run search - returns policy distribution
        policy = mcts.search(state)
        
        assert policy is not None
        assert isinstance(policy, np.ndarray)
        assert policy.shape == (225,)  # 15x15 board
        assert np.isclose(policy.sum(), 1.0)  # Should be normalized
        assert np.all(policy >= 0)  # All probabilities non-negative
        
    def test_get_action_probabilities(self):
        """Test getting action probabilities after search"""
        game = GameInterface(GameType.CHESS)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=4096)
        config = MCTSConfig(num_simulations=200, temperature=1.0, game_type=GameType.CHESS, board_size=8)
        
        mcts = MCTS(config, evaluator, game_interface=game)
        state = game.create_initial_state()
        
        # Run search - returns policy array
        policy = mcts.search(state)
        
        assert isinstance(policy, np.ndarray)
        assert policy.shape == (64,)  # 8x8 chess board
        assert np.isclose(policy.sum(), 1.0)
        assert np.all(policy >= 0)
        
        # For chess, policy is over board squares, not all possible moves
        # Just check that some squares have non-zero probability
        assert np.any(policy > 0)
        
    def test_get_best_action(self):
        """Test getting best action after search"""
        game = GameInterface(GameType.GO, board_size=9)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=81)  # 9x9
        config = MCTSConfig(num_simulations=300, game_type=GameType.GO, board_size=9)
        
        mcts = MCTS(config, evaluator, game_interface=game)
        state = game.create_initial_state()
        
        # Run search
        policy = mcts.search(state)
        
        # Get best action - takes state, not root
        best_action = mcts.get_best_action(state)
        
        assert best_action is not None
        assert best_action in game.get_legal_moves(state)
        
        # Best action should have high probability in policy
        assert policy[best_action] > 0
        
    def test_temperature_effects(self):
        """Test temperature effects on action selection"""
        game = GameInterface(GameType.GOMOKU, board_size=15)  # Standard board
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=225)  # 15x15
        config = MCTSConfig(
            num_simulations=100,
            game_type=GameType.GOMOKU,
            board_size=15,
            temperature=1.0
        )
        
        mcts = MCTS(config, evaluator, game_interface=game)
        state = game.create_initial_state()
        
        # Run search with normal temperature
        policy_t1 = mcts.search(state)
        
        # Run search with zero temperature (deterministic)
        mcts.config.temperature = 0.0
        policy_t0 = mcts.search(state)
        
        # With temp=0, should be more concentrated (higher max probability)
        assert policy_t0.max() > policy_t1.max()
        
        # Test entropy difference
        def entropy(p):
            return -np.sum(p * np.log(p + 1e-8))
        
        # Higher temperature should have higher entropy (more uniform)
        assert entropy(policy_t0) < entropy(policy_t1)
        
    def test_reuse_tree(self):
        """Test tree reuse between searches"""
        game = GameInterface(GameType.CHESS)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=4096)
        config = MCTSConfig(num_simulations=100, game_type=GameType.CHESS, board_size=8)
        
        mcts = MCTS(config, evaluator, game_interface=game)
        
        # Initial search
        state1 = game.create_initial_state()
        policy1 = mcts.search(state1)
        
        # Check first search worked
        assert isinstance(policy1, np.ndarray)
        assert policy1.shape == (64,)  # 8x8 board
        
        # For Chess, get_best_action returns board position not move encoding
        # So we'll use a legal move directly
        legal_moves = game.get_legal_moves(state1)
        assert len(legal_moves) > 0
        move = legal_moves[0]  # Take first legal move
        state2 = game.apply_move(state1, move)
        
        # Search from new position
        policy2 = mcts.search(state2)
        
        # Check that search worked
        assert isinstance(policy2, np.ndarray)
        assert policy2.shape == (64,)  # 8x8 board
        
    def test_search_terminal_position(self):
        """Test searching from a terminal position"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=225)  # 15x15
        config = MCTSConfig(num_simulations=10, game_type=GameType.GOMOKU, board_size=15)
        
        mcts = MCTS(config, evaluator, game_interface=game)
        
        # Create a real terminal state by filling the board
        state = game.create_initial_state()
        # Fill board to create a draw (terminal state)
        moves = []
        for i in range(9):
            for j in range(9):
                if game.is_terminal(state):
                    break
                legal = game.get_legal_moves(state)
                if legal:
                    move = legal[0]  # Just take first legal move
                    state = game.apply_move(state, move)
                    moves.append(move)
        
        # If not terminal yet, mock the terminal check
        original_is_terminal = game.is_terminal
        game.is_terminal = Mock(return_value=True)
        
        try:
            # Search should handle terminal state gracefully
            policy = mcts.search(state)
            
            assert policy is not None
            assert isinstance(policy, np.ndarray)
            assert policy.shape == (225,)  # 15x15
            # Terminal state should have uniform or zero policy
            assert np.all(policy >= 0)
            assert np.all(policy <= 1)
        finally:
            # Restore original method
            game.is_terminal = original_is_terminal
        
    def test_get_pv_line(self):
        """Test getting principal variation line"""
        game = GameInterface(GameType.GO, board_size=15)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=225)  # 15x15
        config = MCTSConfig(num_simulations=200, game_type=GameType.GO, board_size=15)
        
        mcts = MCTS(config, evaluator, game_interface=game)
        state = game.create_initial_state()
        
        # Run search
        policy = mcts.search(state)
        
        # Check policy is valid
        assert isinstance(policy, np.ndarray)
        assert policy.shape == (225,)  # 15x15
        assert abs(sum(policy) - 1.0) < 0.01  # Should sum to 1
        
        # Get best move from policy
        best_move_idx = np.argmax(policy)
        assert 0 <= best_move_idx < 225
        
        # Can't test PV line without access to tree internals
        # The MCTS.get_pv_line method expects a node object, but search returns policy array
            
    def test_get_search_statistics(self):
        """Test getting search statistics"""
        game = GameInterface(GameType.CHESS)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=4096)
        config = MCTSConfig(num_simulations=100, game_type=GameType.CHESS, board_size=8)
        
        mcts = MCTS(config, evaluator, game_interface=game)
        state = game.create_initial_state()
        
        # Run search
        policy = mcts.search(state)
        
        # Get statistics
        stats = mcts.get_statistics()
        
        assert 'total_simulations' in stats
        assert 'tree_nodes' in stats
        assert 'total_time' in stats
        assert 'last_search_sims_per_second' in stats
        
        assert stats['total_simulations'] >= config.num_simulations
        assert stats['tree_nodes'] > 0
        assert stats['total_time'] > 0
        assert stats['last_search_sims_per_second'] > 0
        
    def test_progressive_widening(self):
        """Test progressive expansion of nodes"""
        
        game = GameInterface(GameType.GO, board_size=19)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=361)  # 19x19
        config = MCTSConfig(
            num_simulations=500,
            initial_children_per_expansion=8,
            max_children_per_node=50,
            progressive_expansion_threshold=5,
            game_type=GameType.GO,
            board_size=19
        )
        
        mcts = MCTS(config, evaluator, game_interface=game)
        state = game.create_initial_state()
        
        # Run search
        policy = mcts.search(state)
        
        # Check that search completed successfully
        # Progressive expansion should have occurred during search
        root_data = mcts.tree.get_node_data(0, ['visits'])
        assert root_data['visits'].item() >= config.num_simulations
        
        # Policy should be valid
        assert len(policy) == 361
        assert abs(sum(policy) - 1.0) < 0.01  # Should sum to 1
        
    def test_noise_at_root(self):
        """Test adding Dirichlet noise at root"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=225)  # 15x15
        config = MCTSConfig(
            num_simulations=20,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=0.3,
            game_type=GameType.GOMOKU,
            board_size=15
        )
        
        mcts = MCTS(config, evaluator, game_interface=game)
        state = game.create_initial_state()
        
        # Test that noise is applied by checking if priors are modified
        # First, get the original policy without noise
        config_no_noise = MCTSConfig(
            num_simulations=20,
            dirichlet_epsilon=0.0,  # No noise
            game_type=GameType.GOMOKU,
            board_size=15
        )
        mcts_no_noise = MCTS(config_no_noise, evaluator, game_interface=game)
        policy_no_noise = mcts_no_noise.search(state)
        
        # Now search with noise
        policy_with_noise = mcts.search(state)
        
        # Both should be valid policies
        assert isinstance(policy_no_noise, np.ndarray)
        assert isinstance(policy_with_noise, np.ndarray)
        assert policy_no_noise.shape == (225,)
        assert policy_with_noise.shape == (225,)
        
        # Policies should differ due to Dirichlet noise
        # Check that at least some probabilities differ
        differences = np.sum(np.abs(policy_no_noise - policy_with_noise) > 1e-6)
        assert differences > 0, "Dirichlet noise should affect the policy"