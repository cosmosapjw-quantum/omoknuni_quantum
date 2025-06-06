"""Tests for the main MCTS class"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from mcts import (
    MCTS, MCTSConfig, 
    GameInterface, GameType,
    MockEvaluator, EvaluatorConfig,
    MemoryConfig
)


class TestMCTS:
    """Test suite for MCTS class"""
    
    def test_mcts_config(self):
        """Test MCTS configuration"""
        config = MCTSConfig()
        assert config.num_simulations == 800
        assert config.c_puct == 1.0
        assert config.temperature == 1.0
        assert config.use_wave_engine is True
        
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
        game = GameInterface(GameType.GOMOKU, board_size=9)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=81)
        config = MCTSConfig(num_simulations=100)
        
        mcts = MCTS(config, game, evaluator)
        
        assert mcts.game == game
        assert mcts.evaluator == evaluator
        assert mcts.config == config
        assert mcts.tree is not None
        assert mcts.wave_engine is not None
        
    def test_search_from_position(self):
        """Test running MCTS search from a position"""
        game = GameInterface(GameType.GOMOKU, board_size=9)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=81)
        config = MCTSConfig(num_simulations=100)
        
        mcts = MCTS(config, game, evaluator)
        
        # Create initial position
        state = game.create_initial_state()
        
        # Run search
        root = mcts.search(state)
        
        assert root is not None
        assert root.visit_count >= config.num_simulations
        assert root.is_expanded
        assert len(root.children) > 0
        
    def test_get_action_probabilities(self):
        """Test getting action probabilities after search"""
        game = GameInterface(GameType.CHESS)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=4096)
        config = MCTSConfig(num_simulations=200, temperature=1.0)
        
        mcts = MCTS(config, game, evaluator)
        state = game.create_initial_state()
        
        # Run search
        root = mcts.search(state)
        
        # Get action probabilities
        probs = mcts.get_action_probabilities(root, temperature=1.0)
        
        assert isinstance(probs, dict)
        assert len(probs) > 0
        assert all(0 <= p <= 1 for p in probs.values())
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        
    def test_get_best_action(self):
        """Test getting best action after search"""
        game = GameInterface(GameType.GO, board_size=9)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=82)
        config = MCTSConfig(num_simulations=300)
        
        mcts = MCTS(config, game, evaluator)
        state = game.create_initial_state()
        
        # Run search
        root = mcts.search(state)
        
        # Get best action
        best_action = mcts.get_best_action(root)
        
        assert best_action is not None
        assert best_action in game.get_legal_moves(state)
        
        # Best action should have high visit count
        best_child = root.children[best_action]
        visit_counts = [child.visit_count for child in root.children.values()]
        assert best_child.visit_count == max(visit_counts)
        
    def test_temperature_effects(self):
        """Test temperature effects on action selection"""
        game = GameInterface(GameType.GOMOKU, board_size=9)  # Smaller board
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=81)
        config = MCTSConfig(
            num_simulations=10,  # Fewer simulations
            progressive_widening=True,  # Limit expansion
            pw_alpha=1.0,
            pw_beta=0.5,
            use_wave_engine=False  # Sequential for precise control
        )
        
        mcts = MCTS(config, game, evaluator)
        state = game.create_initial_state()
        root = mcts.search(state)
        
        # Clear all children and create just a few with specific visit counts
        root.children.clear()
        root.is_expanded = False
        
        # Manually create just 4 children with very different visit counts
        from mcts.node import Node
        actions = [10, 20, 30, 40]  # Arbitrary action indices
        visit_counts = [100, 50, 10, 1]  # Very different visit counts
        
        for i, action in enumerate(actions):
            child_state = game.apply_move(state, action)
            child = Node(child_state, root, action, 0.25)  # Equal priors
            child.visit_count = visit_counts[i]
            child.value_sum = child.visit_count * 0.5
            root.children[action] = child
            
        root.is_expanded = True
        
        # Get probabilities at different temperatures
        probs_t0 = mcts.get_action_probabilities(root, temperature=0)
        probs_t1 = mcts.get_action_probabilities(root, temperature=1)
        probs_t2 = mcts.get_action_probabilities(root, temperature=2)
        
        # Temperature 0 should be deterministic (one action gets all probability)
        assert max(probs_t0.values()) > 0.99
        
        # Higher temperature should be more uniform
        entropy_t1 = -sum(p * np.log(p + 1e-8) for p in probs_t1.values() if p > 0)
        entropy_t2 = -sum(p * np.log(p + 1e-8) for p in probs_t2.values() if p > 0)
        assert entropy_t2 > entropy_t1
        
    def test_reuse_tree(self):
        """Test tree reuse between searches"""
        game = GameInterface(GameType.CHESS)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=4096)
        config = MCTSConfig(num_simulations=100, reuse_tree=True)
        
        mcts = MCTS(config, game, evaluator)
        
        # Initial search
        state1 = game.create_initial_state()
        root1 = mcts.search(state1)
        
        # Make a move
        best_action = mcts.get_best_action(root1)
        state2 = game.apply_move(state1, best_action)
        
        # Search from new position (should reuse subtree)
        root2 = mcts.search(state2, parent_action=best_action)
        
        # Root2 should have some visits from previous search
        assert root2.visit_count > config.num_simulations
        
    def test_search_terminal_position(self):
        """Test searching from a terminal position"""
        game = GameInterface(GameType.GOMOKU, board_size=9)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=81)
        config = MCTSConfig(num_simulations=10, use_wave_engine=False)  # Use sequential for simpler test
        
        mcts = MCTS(config, game, evaluator)
        
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
            root = mcts.search(state)
            
            assert root is not None
            assert root.is_terminal
            assert len(root.children) == 0
        finally:
            # Restore original method
            game.is_terminal = original_is_terminal
        
    def test_get_pv_line(self):
        """Test getting principal variation line"""
        game = GameInterface(GameType.GO, board_size=9)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=82)
        config = MCTSConfig(num_simulations=200)
        
        mcts = MCTS(config, game, evaluator)
        state = game.create_initial_state()
        
        # Run search
        root = mcts.search(state)
        
        # Get PV line
        pv_line = mcts.get_pv_line(root, max_depth=5)
        
        assert isinstance(pv_line, list)
        assert len(pv_line) > 0
        assert len(pv_line) <= 5
        
        # Each move in PV should be the most visited child
        current = root
        for move in pv_line:
            assert move in current.children
            child = current.children[move]
            # Should be most visited among siblings
            sibling_visits = [c.visit_count for c in current.children.values()]
            assert child.visit_count == max(sibling_visits)
            current = child
            
    def test_get_search_statistics(self):
        """Test getting search statistics"""
        game = GameInterface(GameType.CHESS)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=4096)
        config = MCTSConfig(num_simulations=100)
        
        mcts = MCTS(config, game, evaluator)
        state = game.create_initial_state()
        
        # Run search
        root = mcts.search(state)
        
        # Get statistics
        stats = mcts.get_statistics()
        
        assert 'total_simulations' in stats
        assert 'nodes_created' in stats
        assert 'search_time' in stats
        assert 'simulations_per_second' in stats
        
        assert stats['total_simulations'] >= config.num_simulations
        assert stats['nodes_created'] > 0
        assert stats['search_time'] > 0
        assert stats['simulations_per_second'] > 0
        
    def test_progressive_widening(self):
        """Test progressive widening if enabled"""
        
        game = GameInterface(GameType.GO, board_size=19)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=362)
        config = MCTSConfig(
            num_simulations=500,
            progressive_widening=True,
            pw_alpha=1.0,
            pw_beta=0.5,
            use_wave_engine=False  # Disable wave engine to avoid interference dependency
        )
        
        mcts = MCTS(config, game, evaluator)
        state = game.create_initial_state()
        
        # Run search
        root = mcts.search(state)
        
        # With progressive widening, not all legal moves should be expanded initially
        legal_moves = game.get_legal_moves(state)
        assert len(root.children) < len(legal_moves)
        
        # But highly visited nodes should have more children
        # (This is a simplified test - actual PW is more complex)
        assert len(root.children) > 10  # Should have expanded some moves
        
    def test_noise_at_root(self):
        """Test adding Dirichlet noise at root"""
        game = GameInterface(GameType.GOMOKU, board_size=9)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=81)
        config = MCTSConfig(
            num_simulations=20,
            add_noise_at_root=True,
            noise_epsilon=0.25,
            noise_alpha=0.3,
            reuse_tree=False,  # Don't reuse tree so each search is independent
            use_wave_engine=False  # Use sequential for more predictable behavior
        )
        
        mcts = MCTS(config, game, evaluator)
        state = game.create_initial_state()
        
        # Test that noise is applied by checking if priors are modified
        # First, get the original policy without noise
        config_no_noise = MCTSConfig(
            num_simulations=20,
            add_noise_at_root=False,
            use_wave_engine=False
        )
        mcts_no_noise = MCTS(game, evaluator, config_no_noise)
        root_no_noise = mcts_no_noise.search(state)
        
        # Get original priors (without noise)
        original_priors = {}
        if root_no_noise.children:
            for action, child in list(root_no_noise.children.items())[:5]:  # Just check first 5
                original_priors[action] = child.prior
        
        # Now search with noise
        root_with_noise = mcts.search(state)
        
        # Get priors with noise applied
        noise_priors = {}
        if root_with_noise.children:
            for action, child in root_with_noise.children.items():
                if action in original_priors:  # Only check actions we have originals for
                    noise_priors[action] = child.prior
        
        # At least some priors should be different due to noise
        if original_priors and noise_priors:
            differences = []
            for action in original_priors:
                if action in noise_priors:
                    diff = abs(original_priors[action] - noise_priors[action])
                    differences.append(diff)
                    
            # At least one prior should be noticeably different
            assert any(diff > 1e-6 for diff in differences), f"No significant differences found in priors: {differences}"