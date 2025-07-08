"""Tests for the arena manager module"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from mcts.neural_networks.arena_manager import (
    ArenaManager, ArenaConfig, ArenaMatch, EloRatingSystem
)
from mcts.utils.config_system import create_default_config


class TestArenaConfig:
    """Test the ArenaConfig dataclass"""
    
    def test_default_config(self):
        """Test default arena configuration"""
        config = ArenaConfig()
        
        assert config.num_games == 40
        assert config.win_threshold == 0.55
        assert config.num_workers == 4
        assert config.mcts_simulations == 200
        assert config.temperature == 0.0
        assert config.device == "cuda"
        assert config.initial_elo == 1500
        assert config.k_factor == 32
    
    def test_custom_config(self):
        """Test custom arena configuration"""
        config = ArenaConfig(
            num_games=100,
            win_threshold=0.6,
            mcts_simulations=400
        )
        
        assert config.num_games == 100
        assert config.win_threshold == 0.6
        assert config.mcts_simulations == 400


class TestEloRatingSystem:
    """Test the ELO rating system"""
    
    def test_initial_rating(self):
        """Test initial ELO rating"""
        elo = EloRatingSystem()
        
        assert elo.get_rating("player1") == 1500
        assert elo.get_rating("new_player") == 1500
    
    def test_expected_score(self):
        """Test expected score calculation"""
        elo = EloRatingSystem()
        
        # Equal ratings
        assert abs(elo.expected_score(1500, 1500) - 0.5) < 0.001
        
        # 100 point difference
        expected = elo.expected_score(1600, 1500)
        assert 0.6 < expected < 0.65
        
        # 400 point difference
        expected = elo.expected_score(1900, 1500)
        assert expected > 0.9
    
    def test_update_ratings(self):
        """Test ELO rating updates"""
        elo = EloRatingSystem()
        
        # Player 1 wins
        elo.update_ratings("player1", "player2", 1.0)
        
        assert elo.get_rating("player1") > 1500
        assert elo.get_rating("player2") < 1500
        assert abs(elo.get_rating("player1") + elo.get_rating("player2") - 3000) < 0.001
    
    def test_draw_update(self):
        """Test ELO update for draw"""
        elo = EloRatingSystem()
        
        # Draw between equal players
        elo.update_ratings("player1", "player2", 0.5)
        
        assert elo.get_rating("player1") == 1500
        assert elo.get_rating("player2") == 1500
    
    def test_rating_history(self):
        """Test rating history tracking"""
        elo = EloRatingSystem()
        
        elo.update_ratings("player1", "player2", 1.0)
        elo.update_ratings("player1", "player3", 1.0)
        
        history = elo.get_rating_history("player1")
        assert len(history) == 3  # Initial + 2 games
        assert history[0] == 1500
        assert all(history[i] < history[i+1] for i in range(len(history)-1))


class TestArenaMatch:
    """Test the ArenaMatch class"""
    
    def test_match_initialization(self):
        """Test arena match initialization"""
        mock_game = Mock()
        mock_evaluator1 = Mock()
        mock_evaluator2 = Mock()
        
        with patch('mcts.neural_networks.arena_manager.GameInterface'):
            match = ArenaMatch(
                game_class=Mock(return_value=mock_game),
                evaluator1=mock_evaluator1,
                evaluator2=mock_evaluator2,
                config=ArenaConfig(),
                match_id="match_1"
            )
            
            assert match.evaluator1 == mock_evaluator1
            assert match.evaluator2 == mock_evaluator2
            assert match.match_id == "match_1"
            assert match.game_results == []
    
    def test_play_single_game(self):
        """Test playing a single arena game"""
        # Setup mocks
        mock_game = Mock()
        mock_game.get_current_player.side_effect = [1, -1, 1]
        mock_game.get_state.return_value = np.zeros((15, 15))
        mock_game.is_terminal.side_effect = [False, False, True]
        mock_game.get_winner.return_value = 1
        
        mock_evaluator1 = Mock()
        mock_evaluator2 = Mock()
        
        mock_mcts = Mock()
        mock_mcts.search.return_value = np.ones(225) / 225
        
        with patch('mcts.neural_networks.arena_manager.GameInterface'), \
             patch('mcts.neural_networks.arena_manager.MCTS', return_value=mock_mcts):
            match = ArenaMatch(
                game_class=Mock(return_value=mock_game),
                evaluator1=mock_evaluator1,
                evaluator2=mock_evaluator2,
                config=ArenaConfig(),
                match_id="match_1"
            )
            
            winner = match.play_single_game(0)
            
            assert winner in [-1, 0, 1]
            assert len(match.game_results) == 1
    
    def test_play_match(self):
        """Test playing a full match"""
        mock_game = Mock()
        mock_evaluator1 = Mock()
        mock_evaluator2 = Mock()
        
        with patch('mcts.neural_networks.arena_manager.GameInterface'), \
             patch.object(ArenaMatch, 'play_single_game') as mock_play:
            # Model 1 wins 3 out of 4 games
            mock_play.side_effect = [1, 1, -1, 1]
            
            match = ArenaMatch(
                game_class=Mock(return_value=mock_game),
                evaluator1=mock_evaluator1,
                evaluator2=mock_evaluator2,
                config=ArenaConfig(num_games=4),
                match_id="match_1"
            )
            
            results = match.play_match()
            
            assert results['total_games'] == 4
            assert results['model1_wins'] == 3
            assert results['model2_wins'] == 1
            assert results['draws'] == 0
            assert results['win_rate'] == 0.75


class TestArenaManager:
    """Test the ArenaManager class"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        config = create_default_config('gomoku')
        config.arena.num_games = 10
        config.arena.num_workers = 2
        return config
    
    def test_manager_initialization(self, config):
        """Test arena manager initialization"""
        manager = ArenaManager(
            config=config,
            game_class=Mock()
        )
        
        assert manager.config == config
        assert manager.arena_config.num_games == 10
        assert manager.arena_config.num_workers == 2
        assert isinstance(manager.elo_system, EloRatingSystem)
    
    def test_evaluate_models(self, config):
        """Test model evaluation"""
        mock_game_class = Mock()
        mock_evaluator1 = Mock()
        mock_evaluator2 = Mock()
        
        manager = ArenaManager(
            config=config,
            game_class=mock_game_class
        )
        
        with patch.object(manager, '_run_parallel_matches') as mock_run:
            mock_run.return_value = {
                'total_games': 10,
                'model1_wins': 6,
                'model2_wins': 3,
                'draws': 1,
                'win_rate': 0.6
            }
            
            results = manager.evaluate_models(
                evaluator1=mock_evaluator1,
                evaluator2=mock_evaluator2,
                model1_name="model_v1",
                model2_name="model_v2"
            )
            
            assert results['win_rate'] == 0.6
            assert results['passed_threshold'] == True
            assert 'elo_ratings' in results
    
    def test_elo_update_after_match(self, config):
        """Test ELO ratings are updated after match"""
        manager = ArenaManager(
            config=config,
            game_class=Mock()
        )
        
        # Initial ratings
        assert manager.elo_system.get_rating("model_v1") == 1500
        assert manager.elo_system.get_rating("model_v2") == 1500
        
        with patch.object(manager, '_run_parallel_matches') as mock_run:
            mock_run.return_value = {
                'total_games': 10,
                'model1_wins': 7,
                'model2_wins': 3,
                'draws': 0,
                'win_rate': 0.7
            }
            
            results = manager.evaluate_models(
                evaluator1=Mock(),
                evaluator2=Mock(),
                model1_name="model_v1",
                model2_name="model_v2"
            )
            
            # Model 1 should have higher rating
            assert manager.elo_system.get_rating("model_v1") > 1500
            assert manager.elo_system.get_rating("model_v2") < 1500
    
    def test_get_leaderboard(self, config):
        """Test getting ELO leaderboard"""
        manager = ArenaManager(
            config=config,
            game_class=Mock()
        )
        
        # Set some ratings
        manager.elo_system.ratings = {
            "model_v3": 1600,
            "model_v1": 1400,
            "model_v2": 1550
        }
        
        leaderboard = manager.get_leaderboard()
        
        assert len(leaderboard) == 3
        assert leaderboard[0][0] == "model_v3"
        assert leaderboard[0][1] == 1600
        assert leaderboard[1][0] == "model_v2"
        assert leaderboard[2][0] == "model_v1"