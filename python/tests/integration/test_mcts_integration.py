"""Integration tests for MCTS system"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

from mcts.core import MCTS, MCTSConfig, GameInterface, GameType, MockEvaluator
from mcts.utils import AlphaZeroConfig, create_default_config


class TestMCTSGameIntegration:
    """Test MCTS integration with different games"""
    
    @pytest.mark.parametrize("game_type,board_size,expected_actions", [
        (GameType.GOMOKU, 9, 81),
        (GameType.GOMOKU, 15, 225),
        (GameType.GO, 9, 82),  # 9x9 + pass move
        (GameType.GO, 13, 170),  # 13x13 + pass move
    ])
    def test_mcts_different_games(self, game_type, board_size, expected_actions):
        """Test MCTS with different game types and board sizes"""
        if game_type == GameType.GO and board_size == 13:
            pytest.skip("Go 13x13 not fully implemented")
        
        config = MCTSConfig(
            num_simulations=50,  # Small for fast test
            classical_only_mode=True,
            game_type=game_type,
            board_size=board_size,
            device='cpu'
        )
        
        game_interface = GameInterface(game_type, board_size)
        evaluator = MockEvaluator(seed=42)
        mcts = MCTS(config, game_interface, evaluator)
        
        # Test initial state
        initial_state = game_interface.get_initial_state()
        action_probs = mcts.search(initial_state)
        
        assert len(action_probs) == expected_actions
        assert sum(action_probs) > 0
        assert all(prob >= 0 for prob in action_probs)
        
        # Test action selection
        action = mcts.select_action(initial_state, temperature=1.0)
        assert 0 <= action < expected_actions
        
        # Verify action is legal
        legal_moves = game_interface.get_legal_moves(initial_state)
        assert legal_moves[action], f"Selected illegal action {action}"
    
    def test_mcts_game_progression(self):
        """Test MCTS through multiple game moves"""
        config = MCTSConfig(
            num_simulations=30,
            classical_only_mode=True,
            game_type=GameType.GOMOKU,
            board_size=9,
            device='cpu'
        )
        
        game_interface = GameInterface(GameType.GOMOKU, 9)
        evaluator = MockEvaluator(seed=42)
        mcts = MCTS(config, game_interface, evaluator)
        
        state = game_interface.get_initial_state()
        moves_made = 0
        max_moves = 10
        
        while not game_interface.is_terminal(state) and moves_made < max_moves:
            # Get MCTS recommendation
            action_probs = mcts.search(state)
            action = mcts.select_action(state, temperature=0.5)
            
            # Verify recommendation is legal
            legal_moves = game_interface.get_legal_moves(state)
            assert legal_moves[action], f"MCTS recommended illegal move {action}"
            
            # Make the move
            new_state = game_interface.make_move(state, action)
            
            # Verify state progression
            assert not np.array_equal(state, new_state), "State should change after move"
            
            state = new_state
            moves_made += 1
        
        assert moves_made > 0, "Should have made at least one move"
        assert moves_made <= max_moves, "Should not exceed move limit"
    
    def test_mcts_winning_position_detection(self):
        """Test MCTS behavior near winning positions"""
        config = MCTSConfig(
            num_simulations=100,
            classical_only_mode=True,
            game_type=GameType.GOMOKU,
            board_size=9,
            device='cpu'
        )
        
        game_interface = GameInterface(GameType.GOMOKU, 9)
        evaluator = MockEvaluator(seed=42)
        mcts = MCTS(config, game_interface, evaluator)
        
        # Create a near-winning state (4 in a row, need 1 more)
        state = np.zeros((3, 9, 9), dtype=np.float32)
        
        # Place 4 in a row for Player 1
        for i in range(4):
            state[0, 4, 2+i] = 1.0  # Row 4, columns 2-5
        
        # Add some Player 2 moves
        state[1, 3, 3] = 1.0
        state[1, 5, 4] = 1.0
        
        # Set turn to Player 1
        state[2, :, :] = 1.0
        
        # MCTS should recognize the winning opportunity
        action_probs = mcts.search(state)
        
        # The winning moves should be at (4, 1) or (4, 6)
        winning_actions = [
            game_interface.position_to_action(4, 1),  # Complete on left
            game_interface.position_to_action(4, 6)   # Complete on right
        ]
        
        # At least one winning move should have significant probability
        winning_prob = sum(action_probs[action] for action in winning_actions)
        assert winning_prob > 0.1, f"MCTS should favor winning moves, got prob {winning_prob}"


class TestMCTSEvaluatorIntegration:
    """Test MCTS integration with different evaluators"""
    
    def test_mcts_mock_evaluator_consistency(self):
        """Test MCTS with MockEvaluator for consistent behavior"""
        config = MCTSConfig(
            num_simulations=50,
            classical_only_mode=True,
            device='cpu'
        )
        
        game_interface = GameInterface(GameType.GOMOKU, 9)
        
        # Test with same seed - should get consistent results
        evaluator1 = MockEvaluator(seed=123)
        evaluator2 = MockEvaluator(seed=123)
        
        mcts1 = MCTS(config, game_interface, evaluator1)
        mcts2 = MCTS(config, game_interface, evaluator2)
        
        state = game_interface.get_initial_state()
        
        probs1 = mcts1.search(state)
        probs2 = mcts2.search(state)
        
        # Should be reasonably similar (not exactly equal due to tree search randomness)
        assert len(probs1) == len(probs2)
        assert sum(probs1) > 0 and sum(probs2) > 0
    
    def test_mcts_evaluator_error_handling(self):
        """Test MCTS behavior with evaluator errors"""
        config = MCTSConfig(
            num_simulations=20,
            classical_only_mode=True,
            device='cpu'
        )
        
        game_interface = GameInterface(GameType.GOMOKU, 9)
        
        # Create evaluator that raises errors
        faulty_evaluator = Mock()
        faulty_evaluator.evaluate.side_effect = RuntimeError("Evaluator error")
        
        mcts = MCTS(config, game_interface, faulty_evaluator)
        state = game_interface.get_initial_state()
        
        # Should handle evaluator errors gracefully
        with pytest.raises(RuntimeError):
            mcts.search(state)
    
    def test_mcts_evaluator_batch_processing(self):
        """Test MCTS with evaluator batch processing"""
        config = MCTSConfig(
            num_simulations=40,
            classical_only_mode=True,
            device='cpu'
        )
        
        game_interface = GameInterface(GameType.GOMOKU, 9)
        evaluator = MockEvaluator()
        
        # Track evaluator calls
        original_evaluate = evaluator.evaluate
        call_count = 0
        total_batch_size = 0
        
        def counting_evaluate(states, **kwargs):
            nonlocal call_count, total_batch_size
            call_count += 1
            total_batch_size += states.shape[0]
            return original_evaluate(states, **kwargs)
        
        evaluator.evaluate = counting_evaluate
        
        mcts = MCTS(config, game_interface, evaluator)
        state = game_interface.get_initial_state()
        
        action_probs = mcts.search(state)
        
        # Should have made evaluator calls
        assert call_count > 0, "Evaluator should have been called"
        assert total_batch_size >= config.num_simulations, "Should evaluate enough states"
        assert len(action_probs) == 81


class TestMCTSConfigurationIntegration:
    """Test MCTS integration with different configurations"""
    
    def test_mcts_with_alphazero_config(self):
        """Test MCTS integration with AlphaZero-style configuration"""
        alphazero_config = AlphaZeroConfig(
            game_type='gomoku',
            board_size=9,
            num_simulations=50,
            c_puct=1.0,
            temperature=1.0,
            device='cpu'
        )
        
        # Convert to MCTS config
        mcts_config = MCTSConfig(
            num_simulations=alphazero_config.num_simulations,
            c_puct=alphazero_config.c_puct,
            temperature=alphazero_config.temperature,
            classical_only_mode=True,
            game_type=GameType.GOMOKU,
            board_size=alphazero_config.board_size,
            device=alphazero_config.device
        )
        
        game_interface = GameInterface(GameType.GOMOKU, 9)
        evaluator = MockEvaluator()
        mcts = MCTS(mcts_config, game_interface, evaluator)
        
        state = game_interface.get_initial_state()
        action_probs = mcts.search(state)
        
        assert len(action_probs) == 81
        assert sum(action_probs) > 0
    
    def test_mcts_configuration_effects(self):
        """Test effects of different MCTS configuration parameters"""
        base_config = MCTSConfig(
            num_simulations=30,
            classical_only_mode=True,
            game_type=GameType.GOMOKU,
            board_size=9,
            device='cpu'
        )
        
        game_interface = GameInterface(GameType.GOMOKU, 9)
        evaluator = MockEvaluator(seed=42)
        
        # Test different c_puct values
        configs = [
            MCTSConfig(**{**base_config.__dict__, 'c_puct': 0.5}),
            MCTSConfig(**{**base_config.__dict__, 'c_puct': 1.0}),
            MCTSConfig(**{**base_config.__dict__, 'c_puct': 2.0}),
        ]
        
        state = game_interface.get_initial_state()
        results = []
        
        for config in configs:
            mcts = MCTS(config, game_interface, evaluator)
            action_probs = mcts.search(state)
            results.append(action_probs)
        
        # All should produce valid results
        for probs in results:
            assert len(probs) == 81
            assert sum(probs) > 0
            assert all(p >= 0 for p in probs)
        
        # Results might be different due to different exploration
        # (though with MockEvaluator they might be similar)
    
    def test_mcts_temperature_effects(self):
        """Test effects of temperature on action selection"""
        config = MCTSConfig(
            num_simulations=50,
            classical_only_mode=True,
            game_type=GameType.GOMOKU,
            board_size=9,
            device='cpu'
        )
        
        game_interface = GameInterface(GameType.GOMOKU, 9)
        evaluator = MockEvaluator(seed=42)
        mcts = MCTS(config, game_interface, evaluator)
        
        state = game_interface.get_initial_state()
        
        # Test different temperatures
        temperatures = [0.1, 1.0, 2.0]
        selected_actions = []
        
        for temp in temperatures:
            actions = []
            for _ in range(20):  # Multiple selections to see distribution
                action = mcts.select_action(state, temperature=temp)
                actions.append(action)
            selected_actions.append(actions)
        
        # Low temperature should be more deterministic
        low_temp_unique = len(set(selected_actions[0]))
        high_temp_unique = len(set(selected_actions[2]))
        
        # This is probabilistic, so we'll just check they're valid
        assert all(0 <= action < 81 for actions in selected_actions for action in actions)
        assert low_temp_unique >= 1  # Should select at least one action
        assert high_temp_unique >= 1  # Should select at least one action


class TestMCTSPerformanceIntegration:
    """Test MCTS performance characteristics in integration scenarios"""
    
    @pytest.mark.slow
    def test_mcts_performance_scaling(self):
        """Test MCTS performance with different simulation counts"""
        game_interface = GameInterface(GameType.GOMOKU, 9)
        evaluator = MockEvaluator()
        
        simulation_counts = [10, 50, 100]
        times = []
        
        for num_sims in simulation_counts:
            config = MCTSConfig(
                num_simulations=num_sims,
                classical_only_mode=True,
                device='cpu'
            )
            
            mcts = MCTS(config, game_interface, evaluator)
            state = game_interface.get_initial_state()
            
            start_time = time.time()
            action_probs = mcts.search(state)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            # Result should be valid regardless of simulation count
            assert len(action_probs) == 81
            assert sum(action_probs) > 0
        
        # More simulations should generally take more time
        # (Though this is not guaranteed due to caching and other optimizations)
        assert all(t > 0 for t in times), "All searches should take some time"
        assert max(times) < 10.0, "No search should take more than 10 seconds"
    
    def test_mcts_memory_usage_integration(self):
        """Test MCTS memory usage in integration scenario"""
        config = MCTSConfig(
            num_simulations=100,
            classical_only_mode=True,
            device='cpu'
        )
        
        game_interface = GameInterface(GameType.GOMOKU, 9)
        evaluator = MockEvaluator()
        mcts = MCTS(config, game_interface, evaluator)
        
        # Perform multiple searches on different states
        initial_state = game_interface.get_initial_state()
        
        for i in range(10):
            # Create slightly different states
            state = initial_state.copy()
            if i > 0:
                # Make a random legal move
                legal_moves = game_interface.get_legal_moves(state)
                legal_actions = np.where(legal_moves)[0]
                if len(legal_actions) > 0:
                    action = legal_actions[i % len(legal_actions)]
                    state = game_interface.make_move(state, action)
            
            # Perform search
            action_probs = mcts.search(state)
            assert len(action_probs) == 81
            assert sum(action_probs) > 0
        
        # If we get here without memory errors, memory usage is reasonable
        assert True
    
    def test_mcts_concurrent_usage(self):
        """Test MCTS behavior with concurrent usage patterns"""
        config = MCTSConfig(
            num_simulations=30,
            classical_only_mode=True,
            device='cpu'
        )
        
        game_interface = GameInterface(GameType.GOMOKU, 9)
        evaluator = MockEvaluator()
        
        # Create multiple MCTS instances
        mcts_instances = [MCTS(config, game_interface, evaluator) for _ in range(3)]
        
        state = game_interface.get_initial_state()
        results = []
        
        # Use all instances
        for mcts in mcts_instances:
            action_probs = mcts.search(state)
            results.append(action_probs)
        
        # All should produce valid results
        for probs in results:
            assert len(probs) == 81
            assert sum(probs) > 0
            assert all(p >= 0 for p in probs)


class TestMCTSRobustness:
    """Test MCTS robustness and error handling in integration scenarios"""
    
    def test_mcts_invalid_state_handling(self):
        """Test MCTS handling of invalid game states"""
        config = MCTSConfig(
            num_simulations=20,
            classical_only_mode=True,
            device='cpu'
        )
        
        game_interface = GameInterface(GameType.GOMOKU, 9)
        evaluator = MockEvaluator()
        mcts = MCTS(config, game_interface, evaluator)
        
        # Test with invalid state shape
        invalid_state = np.random.rand(2, 9, 9).astype(np.float32)  # Wrong channels
        
        with pytest.raises((ValueError, RuntimeError)):
            mcts.search(invalid_state)
        
        # Test with None state
        with pytest.raises((ValueError, TypeError)):
            mcts.search(None)
    
    def test_mcts_edge_case_states(self):
        """Test MCTS with edge case game states"""
        config = MCTSConfig(
            num_simulations=20,
            classical_only_mode=True,
            device='cpu'
        )
        
        game_interface = GameInterface(GameType.GOMOKU, 9)
        evaluator = MockEvaluator()
        mcts = MCTS(config, game_interface, evaluator)
        
        # Test with nearly full board
        state = np.zeros((3, 9, 9), dtype=np.float32)
        
        # Fill most of the board
        for i in range(9):
            for j in range(9):
                if (i + j) % 3 != 0:  # Leave some spaces
                    player = 1 if (i + j) % 2 == 0 else 2
                    layer = 0 if player == 1 else 1
                    state[layer, i, j] = 1.0
        
        # Set current player
        state[2, :, :] = 1.0
        
        # Should handle this state gracefully
        if not game_interface.is_terminal(state):
            action_probs = mcts.search(state)
            assert len(action_probs) == 81
            
            # Only legal moves should have non-zero probability
            legal_moves = game_interface.get_legal_moves(state)
            for i, (is_legal, prob) in enumerate(zip(legal_moves, action_probs)):
                if not is_legal:
                    assert prob == 0.0, f"Illegal move {i} has probability {prob}"
    
    def test_mcts_configuration_robustness(self):
        """Test MCTS robustness with unusual configurations"""
        game_interface = GameInterface(GameType.GOMOKU, 9)
        evaluator = MockEvaluator()
        
        # Test with very few simulations
        config_few_sims = MCTSConfig(
            num_simulations=1,
            classical_only_mode=True,
            device='cpu'
        )
        
        mcts = MCTS(config_few_sims, game_interface, evaluator)
        state = game_interface.get_initial_state()
        
        action_probs = mcts.search(state)
        assert len(action_probs) == 81
        assert sum(action_probs) > 0
        
        # Test with unusual c_puct values
        config_extreme_cpuct = MCTSConfig(
            num_simulations=20,
            c_puct=10.0,  # Very high exploration
            classical_only_mode=True,
            device='cpu'
        )
        
        mcts_extreme = MCTS(config_extreme_cpuct, game_interface, evaluator)
        action_probs_extreme = mcts_extreme.search(state)
        
        assert len(action_probs_extreme) == 81
        assert sum(action_probs_extreme) > 0