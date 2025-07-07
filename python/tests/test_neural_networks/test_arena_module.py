"""Tests for arena model evaluation and ELO tracking module

Tests cover:
- ELO tracker functionality
- Adaptive K-factor calculation
- Rating deflation and inflation detection
- Arena manager model comparison
- Game outcome tracking
- Tournament organization
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
from pathlib import Path
from datetime import datetime

from mcts.neural_networks.arena_module import (
    ArenaConfig, ELOTracker, ArenaManager
)
from mcts.utils.config_system import AlphaZeroConfig
from mcts.core.evaluator import RandomEvaluator


@pytest.fixture
def arena_config():
    """Create arena configuration"""
    return ArenaConfig(
        num_games=10,
        win_threshold=0.55,
        num_workers=2,
        mcts_simulations=50,
        c_puct=1.0,
        temperature=0.0,
        device='cpu'
    )


@pytest.fixture
def elo_tracker():
    """Create ELO tracker instance"""
    return ELOTracker(k_factor=32.0, initial_rating=1500.0)


@pytest.fixture
def alphazero_config():
    """Create AlphaZero configuration"""
    config = AlphaZeroConfig()
    config.game.game_type = 'gomoku'
    config.game.board_size = 15
    config.network.input_representation = 'basic'
    config.arena.elo_k_factor = 32.0
    return config


@pytest.fixture
def arena_manager(alphazero_config, arena_config):
    """Create ArenaManager instance"""
    return ArenaManager(alphazero_config, arena_config)


@pytest.fixture
def mock_evaluator():
    """Create mock evaluator"""
    evaluator = Mock()
    evaluator.evaluate.return_value = (np.ones(225) / 225, 0.0)
    evaluator.evaluate_batch.return_value = (np.ones((4, 225)) / 225, np.zeros(4))
    return evaluator


class TestArenaConfig:
    """Test ArenaConfig configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ArenaConfig()
        assert config.num_games == 40
        assert config.win_threshold == 0.55
        assert config.num_workers == 4
        assert config.mcts_simulations == 200
        assert config.temperature == 0.0
        assert config.device == 'cuda'
        assert config.enable_tree_reuse == False
        assert config.gc_frequency == 5
        assert config.max_memory_gb == 6.0
        
    def test_custom_config(self):
        """Test custom configuration"""
        config = ArenaConfig(
            num_games=100,
            win_threshold=0.6,
            device='cpu',
            memory_monitoring=False
        )
        assert config.num_games == 100
        assert config.win_threshold == 0.6
        assert config.device == 'cpu'
        assert config.memory_monitoring == False


class TestELOTracker:
    """Test ELO tracker functionality"""
    
    def test_initialization(self, elo_tracker):
        """Test ELO tracker initialization"""
        assert elo_tracker.k_factor == 32.0
        assert elo_tracker.initial_rating == 1500.0
        assert len(elo_tracker.ratings) == 0
        assert 'random' in elo_tracker.anchor_players
        assert elo_tracker.anchor_players['random'] == 0.0
        
    def test_update_ratings_basic(self, elo_tracker):
        """Test basic rating update"""
        # Player 1 wins all games
        elo_tracker.update_ratings('player1', 'player2', wins=5, draws=0, losses=0)
        
        # Player 1 should have higher rating
        assert elo_tracker.ratings['player1'] > 1500.0
        assert elo_tracker.ratings['player2'] < 1500.0
        
        # Check sum approximately preserved (with some tolerance for rounding)
        total_rating = elo_tracker.ratings['player1'] + elo_tracker.ratings['player2']
        assert abs(total_rating - 3000.0) < 1.0  # Small tolerance
        
    def test_update_ratings_draws(self, elo_tracker):
        """Test rating update with draws"""
        # All draws between equal players
        elo_tracker.update_ratings('player1', 'player2', wins=0, draws=10, losses=0)
        
        # Ratings should stay close to initial
        assert abs(elo_tracker.ratings['player1'] - 1500.0) < 1.0
        assert abs(elo_tracker.ratings['player2'] - 1500.0) < 1.0
        
    def test_adaptive_k_factor(self, elo_tracker):
        """Test adaptive K-factor calculation"""
        # New player has high K-factor
        k_new = elo_tracker.get_adaptive_k_factor('new_player', 'opponent')
        assert k_new == 32.0  # Full K for new players
        
        # Add game history
        elo_tracker.ratings['experienced'] = 1800.0
        elo_tracker.game_counts['experienced'] = 50
        
        k_exp = elo_tracker.get_adaptive_k_factor('experienced', 'opponent')
        assert k_exp < 32.0  # Lower K for experienced players
        
    def test_k_factor_vs_random(self, elo_tracker):
        """Test K-factor adjustment against random player"""
        # Low rated player vs random
        elo_tracker.ratings['weak'] = 100.0
        k_low = elo_tracker.get_adaptive_k_factor('weak', 'random')
        
        # High rated player vs random
        elo_tracker.ratings['strong'] = 2000.0
        k_high = elo_tracker.get_adaptive_k_factor('strong', 'random')
        
        # Strong player should have much lower K against random
        assert k_high < k_low * 0.1
        
    def test_recent_performance_tracking(self, elo_tracker):
        """Test recent performance calculation"""
        # Add game history
        for i in range(10):
            elo_tracker.update_ratings('player1', 'player2', 
                                      wins=1 if i % 2 == 0 else 0,
                                      draws=0,
                                      losses=0 if i % 2 == 0 else 1)
            
        perf = elo_tracker.get_recent_performance('player1', 'player2', last_n=5)
        assert perf is not None
        # Should be around 0.5 (alternating wins)
        assert 0.4 <= perf <= 0.6
        
    def test_should_play_vs_random_logic(self, elo_tracker):
        """Test logic for determining random matches"""
        # Early iterations always play
        assert elo_tracker.should_play_vs_random(1, 1500.0) == True
        assert elo_tracker.should_play_vs_random(3, 1500.0) == True
        
        # Mid iterations depend on ELO
        assert elo_tracker.should_play_vs_random(15, 400.0) == True  # Low ELO
        result_high_elo = elo_tracker.should_play_vs_random(15, 1200.0)  # High ELO
        # Should play less frequently
        
        # Late iterations rare
        result_late = elo_tracker.should_play_vs_random(150, 2000.0)
        # Very rare for high ELO late game
        
    def test_elo_growth_rate(self, elo_tracker):
        """Test ELO growth rate calculation"""
        # Add iteration history
        elo_tracker.iteration_elos[1] = 1500.0
        elo_tracker.iteration_elos[2] = 1520.0
        elo_tracker.iteration_elos[3] = 1545.0
        elo_tracker.iteration_elos[4] = 1575.0
        elo_tracker.iteration_elos[5] = 1610.0
        
        growth = elo_tracker.get_elo_growth_rate(5, window=5)
        # Average growth should be (1610-1500)/5 = 22
        assert 20 <= growth <= 25
        
    def test_rating_deflation(self, elo_tracker):
        """Test rating deflation mechanism"""
        # Create inflated ratings
        elo_tracker.ratings['player1'] = 2000.0
        elo_tracker.ratings['player2'] = 2000.0
        elo_tracker.ratings['player3'] = 2000.0
        
        # Force deflation
        old_sum = sum(elo_tracker.ratings.values())
        elo_tracker._apply_rating_deflation(old_sum)
        
        # Ratings should be deflated towards initial
        for player, rating in elo_tracker.ratings.items():
            if player != 'random':  # Not anchor
                assert rating < 2000.0
                
    def test_protect_best_elo(self, elo_tracker):
        """Test best model ELO protection"""
        # Setup initial ratings
        elo_tracker.ratings['best_model'] = 2000.0
        elo_tracker.ratings['challenger'] = 1800.0
        
        # Best model wins but would lose ELO due to expected result
        # This simulates a case where K-factor adjustments might cause decrease
        elo_tracker.update_ratings('best_model', 'challenger',
                                 wins=6, draws=0, losses=4,
                                 protect_best_elo=True,
                                 best_player='best_model')
        
        # Best model's ELO should not decrease
        assert elo_tracker.ratings['best_model'] >= 2000.0
        
    def test_confidence_intervals(self, elo_tracker):
        """Test confidence interval calculation"""
        elo_tracker.ratings['player1'] = 1600.0
        elo_tracker.rating_uncertainty['player1'] = 100.0
        
        lower, upper = elo_tracker.get_confidence_interval('player1')
        
        # 95% CI should be rating ± 2*uncertainty
        assert lower == 1400.0
        assert upper == 1800.0
        
    def test_leaderboard(self, elo_tracker):
        """Test leaderboard generation"""
        elo_tracker.ratings['player1'] = 1800.0
        elo_tracker.ratings['player2'] = 1600.0
        elo_tracker.ratings['player3'] = 2000.0
        
        leaderboard = elo_tracker.get_leaderboard()
        
        # Should be sorted by rating descending
        assert leaderboard[0][0] == 'player3'
        assert leaderboard[1][0] == 'player1'
        assert leaderboard[2][0] == 'player2'
        
    def test_detailed_leaderboard(self, elo_tracker):
        """Test detailed leaderboard with filtering"""
        # Add many iterations
        for i in range(20):
            elo_tracker.ratings[f'iter_{i}'] = 1500.0 + i * 10
            elo_tracker.game_counts[f'iter_{i}'] = i + 1
            
        # Get recent only
        leaderboard = elo_tracker.get_detailed_leaderboard(max_iterations=5)
        
        # Should only include recent iterations
        iter_players = [entry['player'] for entry in leaderboard if entry['player'].startswith('iter_')]
        iter_numbers = [int(p.split('_')[1]) for p in iter_players]
        assert all(n >= 15 for n in iter_numbers)  # Only recent 5
        
    def test_validation_metrics(self, elo_tracker):
        """Test validation metrics calculation"""
        # Add validation history
        elo_tracker.validation_history = [
            {'expected_score1': 0.7, 'actual_score1': 0.8},  # Good prediction
            {'expected_score1': 0.3, 'actual_score1': 0.2},  # Good prediction
            {'expected_score1': 0.8, 'actual_score1': 0.3},  # Bad prediction
        ]
        
        metrics = elo_tracker.get_validation_metrics()
        
        assert 'prediction_accuracy' in metrics
        assert 'rmse' in metrics
        assert metrics['total_validations'] == 3
        
    def test_save_load(self, elo_tracker, tmp_path):
        """Test saving and loading ELO data"""
        # Add some data
        elo_tracker.ratings['player1'] = 1700.0
        elo_tracker.game_counts['player1'] = 20
        elo_tracker.iteration_elos[5] = 1700.0
        
        # Save
        filepath = tmp_path / "elo_data.json"
        elo_tracker.save_to_file(str(filepath))
        
        # Load into new tracker
        new_tracker = ELOTracker()
        new_tracker.load_from_file(str(filepath))
        
        assert new_tracker.ratings['player1'] == 1700.0
        assert new_tracker.game_counts['player1'] == 20
        assert new_tracker.iteration_elos[5] == 1700.0
        
    def test_health_report(self, elo_tracker):
        """Test health report generation"""
        # Add some data
        for i in range(5):
            elo_tracker.ratings[f'player{i}'] = 1500.0 + i * 100
            elo_tracker.game_counts[f'player{i}'] = 10 + i * 5
            
        report = elo_tracker.get_health_report()
        
        assert 'total_players' in report
        assert 'rating_statistics' in report
        assert 'validation_metrics' in report
        assert 'inflation_indicators' in report
        assert report['total_players'] == 5
        
    def test_cleanup_old_iterations(self, elo_tracker):
        """Test cleanup of old iteration entries"""
        # Add many iterations
        for i in range(30):
            elo_tracker.ratings[f'iter_{i}'] = 1500.0 + i
            elo_tracker.iteration_elos[i] = 1500.0 + i
            
        # Also add non-iteration players
        elo_tracker.ratings['persistent'] = 1800.0
        
        # Cleanup keeping only recent 10
        elo_tracker.cleanup_old_iterations(keep_recent=10)
        
        # Check old iterations removed
        assert 'iter_0' not in elo_tracker.ratings
        assert 'iter_19' not in elo_tracker.ratings
        assert 'iter_20' in elo_tracker.ratings
        assert 'iter_29' in elo_tracker.ratings
        
        # Non-iteration players preserved
        assert 'persistent' in elo_tracker.ratings


class TestArenaManager:
    """Test ArenaManager functionality"""
    
    def test_initialization(self, arena_manager, alphazero_config):
        """Test arena manager initialization"""
        assert arena_manager.config == alphazero_config
        assert arena_manager.arena_config is not None
        assert arena_manager.game_interface is not None
        assert arena_manager.elo_tracker is not None
        
    @patch('mcts.neural_networks.arena_module.ArenaManager._sequential_arena')
    def test_compare_models_sequential(self, mock_sequential, arena_manager):
        """Test sequential model comparison"""
        mock_sequential.return_value = (6, 2, 2)  # wins, draws, losses
        
        model1 = Mock()
        model2 = Mock()
        
        result = arena_manager.compare_models(
            model1, model2,
            model1_name='model1',
            model2_name='model2',
            num_games=10,
            silent=True
        )
        
        assert result == (6, 2, 2)
        mock_sequential.assert_called_once()
        
    @patch('mcts.neural_networks.arena_module.ArenaManager._parallel_arena')
    def test_compare_models_vs_random(self, mock_parallel, arena_manager):
        """Test parallel arena for random opponent"""
        mock_parallel.return_value = (8, 0, 2)
        
        model = Mock()
        random_eval = RandomEvaluator()
        
        result = arena_manager.compare_models(
            model, random_eval,
            num_games=10,
            silent=True
        )
        
        assert result == (8, 0, 2)
        mock_parallel.assert_called_once()
        
    def test_cuda_error_handling(self, arena_manager):
        """Test CUDA error handling"""
        with patch('mcts.neural_networks.arena_module.ArenaManager._sequential_arena') as mock_seq:
            mock_seq.side_effect = RuntimeError("CUDA error: out of memory")
            
            with patch('torch.cuda.is_available', return_value=True):
                with patch('torch.cuda.synchronize') as mock_sync:
                    with patch('torch.cuda.empty_cache') as mock_empty:
                        with pytest.raises(RuntimeError):
                            arena_manager.compare_models(Mock(), Mock())
                            
                        # Should attempt cleanup
                        mock_sync.assert_called()
                        mock_empty.assert_called()
                        
    def test_config_serialization(self, arena_manager, alphazero_config):
        """Test config serialization for multiprocessing"""
        serialized = arena_manager._serialize_config(alphazero_config)
        
        assert isinstance(serialized, dict)
        assert 'game' in serialized
        assert 'network' in serialized
        assert serialized['game']['game_type'] == 'gomoku'
        assert serialized['game']['board_size'] == 15


class TestIntegration:
    """Integration tests for arena and ELO tracking"""
    
    def test_tournament_simulation(self, elo_tracker):
        """Test a small tournament simulation"""
        players = ['player1', 'player2', 'player3', 'random']
        
        # Simulate round-robin tournament
        results = [
            ('player1', 'player2', 3, 1, 1),  # P1 slightly better
            ('player1', 'player3', 2, 1, 2),  # P3 slightly better
            ('player1', 'random', 5, 0, 0),   # P1 crushes random
            ('player2', 'player3', 2, 2, 1),  # P2 slightly better
            ('player2', 'random', 4, 1, 0),   # P2 beats random
            ('player3', 'random', 5, 0, 0),   # P3 crushes random
        ]
        
        for p1, p2, wins, draws, losses in results:
            elo_tracker.update_ratings(p1, p2, wins, draws, losses)
            
        # Check final rankings make sense
        leaderboard = elo_tracker.get_leaderboard()
        rankings = {player: rank for rank, (player, _) in enumerate(leaderboard)}
        
        # Random should be last
        assert rankings['random'] == len(players) - 1
        
        # Others should be ranked by performance
        assert elo_tracker.ratings['player1'] > elo_tracker.ratings['random']
        assert elo_tracker.ratings['player2'] > elo_tracker.ratings['random']
        assert elo_tracker.ratings['player3'] > elo_tracker.ratings['random']
        
    def test_iteration_progression(self, elo_tracker):
        """Test ELO progression over training iterations"""
        # Simulate training progression
        for i in range(10):
            model_name = f'iter_{i}'
            
            # Each iteration beats previous
            if i > 0:
                prev_model = f'iter_{i-1}'
                elo_tracker.update_ratings(model_name, prev_model,
                                         wins=6, draws=2, losses=2)
                
            # Also test against random
            win_rate = min(0.95, 0.5 + i * 0.05)  # Improving win rate
            wins = int(10 * win_rate)
            losses = 10 - wins
            
            elo_tracker.update_ratings(model_name, 'random',
                                     wins=wins, draws=0, losses=losses)
                                     
        # Check progression
        growth_rate = elo_tracker.get_elo_growth_rate(9)
        assert growth_rate > 0  # Should show improvement
        
        # Check inflation detection
        indicators = elo_tracker._get_inflation_indicators()
        assert 'total_elo_growth' in indicators