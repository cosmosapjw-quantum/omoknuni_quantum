"""Tests for the self-play manager module"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from mcts.neural_networks.self_play_manager import (
    SelfPlayManager, SelfPlayConfig, SelfPlayGame
)
from mcts.neural_networks.replay_buffer import GameExample
from mcts.utils.config_system import create_default_config


class TestSelfPlayConfig:
    """Test the SelfPlayConfig dataclass"""
    
    def test_default_config(self):
        """Test default self-play configuration"""
        config = SelfPlayConfig()
        
        assert config.num_games_per_iteration == 100
        assert config.num_workers == 4
        assert config.mcts_simulations == 800
        assert config.temperature == 1.0
        assert config.temperature_threshold == 30
        assert config.resign_threshold == -0.98
        assert config.enable_resign == True
        assert config.dirichlet_alpha == 0.3
        assert config.dirichlet_epsilon == 0.25
        assert config.c_puct == 1.5
    
    def test_custom_config(self):
        """Test custom self-play configuration"""
        config = SelfPlayConfig(
            num_games_per_iteration=200,
            num_workers=8,
            mcts_simulations=1600
        )
        
        assert config.num_games_per_iteration == 200
        assert config.num_workers == 8
        assert config.mcts_simulations == 1600


class TestSelfPlayGame:
    """Test the SelfPlayGame class"""
    
    def test_game_initialization(self):
        """Test self-play game initialization"""
        mock_game = Mock()
        mock_evaluator = Mock()
        
        # Mock the GameInterface creation
        with patch('mcts.neural_networks.self_play_manager.GameInterface'):
            game = SelfPlayGame(
                game=mock_game,
                evaluator=mock_evaluator,
                config=SelfPlayConfig(),
                game_id="test_game_1"
            )
            
            assert game.game == mock_game
            assert game.evaluator == mock_evaluator
            assert game.game_id == "test_game_1"
            assert game.examples == []
            assert game.current_player == 1
    
    def test_play_single_move(self):
        """Test playing a single move"""
        # Setup mocks
        mock_game = Mock()
        mock_game.get_current_player.return_value = 1
        mock_game.get_state.return_value = np.zeros((15, 15))
        mock_game.get_valid_actions.return_value = np.ones(225)
        mock_game.is_terminal.return_value = False
        mock_game.get_action_size.return_value = 225
        
        mock_mcts = Mock()
        mock_mcts.search.return_value = np.ones(225) / 225
        
        mock_evaluator = Mock()
        
        config = SelfPlayConfig(temperature=0.0)
        
        with patch('mcts.neural_networks.self_play_manager.MCTS', return_value=mock_mcts), \
             patch('mcts.neural_networks.self_play_manager.GameInterface'):
            game = SelfPlayGame(
                game=mock_game,
                evaluator=mock_evaluator,
                config=config,
                game_id="test_game_1"
            )
            
            # Play one move
            game.play_single_move()
            
            # Verify MCTS was used
            assert mock_mcts.search.called
            
            # Verify game was updated
            assert mock_game.make_action.called
            
            # Verify example was stored
            assert len(game.examples) == 1
            example = game.examples[0]
            assert isinstance(example, dict)
            assert 'state' in example
            assert 'policy' in example
            assert 'player' in example
            assert example['move_number'] == 0
    
    def test_play_full_game(self):
        """Test playing a full game"""
        # Setup game that ends after 2 moves
        mock_game = Mock()
        mock_game.get_current_player.side_effect = [1, -1]
        mock_game.get_state.return_value = np.zeros((15, 15))
        mock_game.get_valid_actions.return_value = np.ones(225)
        mock_game.is_terminal.side_effect = [False, False, True]
        mock_game.get_winner.return_value = 1
        mock_game.get_action_size.return_value = 225
        
        mock_mcts = Mock()
        mock_mcts.search.return_value = np.ones(225) / 225
        
        mock_evaluator = Mock()
        
        config = SelfPlayConfig(temperature=0.0, enable_resign=False)
        
        with patch('mcts.neural_networks.self_play_manager.MCTS', return_value=mock_mcts), \
             patch('mcts.neural_networks.self_play_manager.GameInterface'):
            game = SelfPlayGame(
                game=mock_game,
                evaluator=mock_evaluator,
                config=config,
                game_id="test_game_1"
            )
            
            # Play full game
            examples = game.play_game()
            
            # Should have 2 examples
            assert len(examples) == 2
            
            # Check values are assigned correctly (player 1 won)
            assert examples[0].value == 1.0  # Player 1's move
            assert examples[1].value == -1.0  # Player -1's move


class TestSelfPlayManager:
    """Test the SelfPlayManager class"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        config = create_default_config('gomoku')
        config.training.self_play_games_per_iteration = 10
        config.training.self_play_workers = 2
        return config
    
    @pytest.fixture
    def mock_evaluator(self):
        """Create mock evaluator"""
        evaluator = Mock()
        evaluator.evaluate.return_value = (np.ones(225) / 225, 0.0)
        return evaluator
    
    def test_manager_initialization(self, config):
        """Test self-play manager initialization"""
        manager = SelfPlayManager(
            config=config,
            game_class=Mock,
            evaluator=Mock()
        )
        
        assert manager.config == config
        assert manager.self_play_config.num_games_per_iteration == 10
        assert manager.self_play_config.num_workers == 2
    
    def test_generate_self_play_data_single_worker(self, config, mock_evaluator):
        """Test generating self-play data with single worker"""
        config.training.self_play_workers = 1
        
        # Mock game class
        mock_game_class = Mock()
        mock_game_instance = Mock()
        mock_game_instance.get_current_player.return_value = 1
        mock_game_instance.get_state.return_value = np.zeros((15, 15))
        mock_game_instance.get_valid_actions.return_value = np.ones(225)
        mock_game_instance.is_terminal.side_effect = [False, True]
        mock_game_instance.get_winner.return_value = 1
        mock_game_instance.get_action_size.return_value = 225
        mock_game_class.return_value = mock_game_instance
        
        manager = SelfPlayManager(
            config=config,
            game_class=mock_game_class,
            evaluator=mock_evaluator
        )
        
        with patch('mcts.neural_networks.self_play_manager.MCTS'):
            examples = manager.generate_self_play_data()
            
            # Should have generated examples
            assert len(examples) > 0
            assert all(isinstance(ex, GameExample) for ex in examples)
    
    def test_collect_game_metrics(self, config):
        """Test collecting metrics from games"""
        manager = SelfPlayManager(
            config=config,
            game_class=Mock,
            evaluator=Mock()
        )
        
        # Create test examples
        examples = [
            GameExample(
                state=np.zeros((15, 15)),
                policy=np.ones(225) / 225,
                value=1.0,
                game_id="game_1",
                move_number=0
            ),
            GameExample(
                state=np.zeros((15, 15)),
                policy=np.ones(225) / 225,
                value=-1.0,
                game_id="game_1",
                move_number=1
            ),
            GameExample(
                state=np.zeros((15, 15)),
                policy=np.ones(225) / 225,
                value=1.0,
                game_id="game_2",
                move_number=0
            ),
        ]
        
        metrics = manager.collect_game_metrics(examples)
        
        assert metrics['total_games'] == 2
        assert metrics['total_moves'] == 3
        assert metrics['avg_game_length'] == 1.5
        assert 'win_rates' in metrics
    
    def test_parallel_self_play(self, config, mock_evaluator):
        """Test parallel self-play data generation"""
        config.training.self_play_workers = 2
        config.training.self_play_games_per_iteration = 4
        
        # Mock game class
        mock_game_class = Mock()
        
        manager = SelfPlayManager(
            config=config,
            game_class=mock_game_class,
            evaluator=mock_evaluator
        )
        
        # Mock the worker function
        with patch.object(manager, '_play_games_worker') as mock_worker:
            mock_worker.return_value = [
                GameExample(
                    state=np.zeros((15, 15)),
                    policy=np.ones(225) / 225,
                    value=0.0,
                    game_id=f"game_{i}",
                    move_number=0
                ) for i in range(2)
            ]
            
            examples = manager.generate_self_play_data()
            
            # Should have 4 examples (2 games × 2 examples per game)
            assert len(examples) == 4
            
            # Worker should be called twice (once per worker)
            assert mock_worker.call_count == 2