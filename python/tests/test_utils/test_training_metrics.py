"""Tests for training metrics recording system"""

import pytest
import json
import time
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch

from mcts.utils.training_metrics import (
    MetricSnapshot, TrainingMetricsRecorder, MetricsVisualizer
)


@pytest.fixture
def temp_metrics_dir(tmp_path):
    """Create temporary directory for metrics"""
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    return metrics_dir


@pytest.fixture
def metrics_recorder(temp_metrics_dir):
    """Create metrics recorder instance"""
    return TrainingMetricsRecorder(
        save_dir=temp_metrics_dir,
        window_size=10,
        auto_save_interval=5
    )


class TestMetricSnapshot:
    """Test MetricSnapshot dataclass"""
    
    def test_creation(self):
        """Test snapshot creation"""
        snapshot = MetricSnapshot(
            timestamp=time.time(),
            iteration=10,
            epoch=1,
            policy_loss=0.5,
            value_loss=0.3,
            total_loss=0.8
        )
        
        assert snapshot.iteration == 10
        assert snapshot.epoch == 1
        assert snapshot.policy_loss == 0.5
        assert snapshot.value_loss == 0.3
        assert snapshot.total_loss == 0.8
        
    def test_optional_fields(self):
        """Test optional fields default to None"""
        snapshot = MetricSnapshot(
            timestamp=time.time(),
            iteration=5,
            epoch=0
        )
        
        assert snapshot.policy_loss is None
        assert snapshot.win_rate is None
        assert snapshot.elo_rating is None
        assert snapshot.gradient_norm is None
        
    def test_custom_metrics(self):
        """Test custom metrics storage"""
        snapshot = MetricSnapshot(
            timestamp=time.time(),
            iteration=1,
            custom_metrics={'accuracy': 0.95, 'entropy': 2.3}
        )
        
        assert snapshot.custom_metrics['accuracy'] == 0.95
        assert snapshot.custom_metrics['entropy'] == 2.3


class TestTrainingMetricsRecorder:
    """Test TrainingMetricsRecorder functionality"""
    
    def test_initialization(self, metrics_recorder):
        """Test recorder initialization"""
        assert metrics_recorder.window_size == 10
        assert metrics_recorder.auto_save_interval == 5
        assert metrics_recorder.current_iteration == 0
        assert metrics_recorder.current_epoch == 0
        assert len(metrics_recorder.snapshots) == 0
        
    def test_record_training_step(self, metrics_recorder):
        """Test recording training step metrics"""
        metrics_recorder.record_training_step(
            iteration=1,
            epoch=0,
            policy_loss=0.5,
            value_loss=0.3,
            total_loss=0.8,
            learning_rate=0.001,
            gradient_norm=2.5
        )
        
        assert len(metrics_recorder.snapshots) == 1
        snapshot = metrics_recorder.snapshots[0]
        assert snapshot.iteration == 1
        assert snapshot.policy_loss == 0.5
        assert snapshot.value_loss == 0.3
        assert snapshot.total_loss == 0.8
        assert snapshot.learning_rate == 0.001
        assert snapshot.gradient_norm == 2.5
        
        # Check rolling windows
        assert len(metrics_recorder.rolling_windows['policy_loss']) == 1
        assert metrics_recorder.rolling_windows['policy_loss'][0] == 0.5
        
    def test_record_evaluation(self, metrics_recorder):
        """Test recording evaluation metrics"""
        # First record a training step
        metrics_recorder.record_training_step(
            iteration=5, epoch=1, policy_loss=0.4, 
            value_loss=0.2, total_loss=0.6, learning_rate=0.001
        )
        
        # Then add evaluation metrics
        metrics_recorder.record_evaluation(
            iteration=5,
            win_rate=0.55,
            elo_rating=1600,
            elo_change=50
        )
        
        # Should update existing snapshot
        assert len(metrics_recorder.snapshots) == 1
        snapshot = metrics_recorder.snapshots[0]
        assert snapshot.win_rate == 0.55
        assert snapshot.elo_rating == 1600
        assert snapshot.elo_change == 50
        
    def test_record_self_play_metrics(self, metrics_recorder):
        """Test recording self-play metrics"""
        metrics_recorder.record_self_play_metrics(
            iteration=10,
            games_per_second=2.5,
            avg_game_length=150,
            policy_entropy=2.1,
            mcts_value_accuracy=0.85
        )
        
        assert len(metrics_recorder.snapshots) == 1
        snapshot = metrics_recorder.snapshots[0]
        assert snapshot.games_per_second == 2.5
        assert snapshot.avg_game_length == 150
        assert snapshot.policy_entropy == 2.1
        assert snapshot.mcts_value_accuracy == 0.85
        
    def test_custom_metrics(self, metrics_recorder):
        """Test recording custom metrics"""
        metrics_recorder.record_training_step(
            iteration=1, epoch=0,
            policy_loss=0.5, value_loss=0.3, total_loss=0.8,
            learning_rate=0.001,
            custom_accuracy=0.95,
            custom_f1_score=0.88
        )
        
        snapshot = metrics_recorder.snapshots[0]
        assert snapshot.custom_metrics['custom_accuracy'] == 0.95
        assert snapshot.custom_metrics['custom_f1_score'] == 0.88
        
    def test_moving_average(self, metrics_recorder):
        """Test moving average calculation"""
        # Record multiple values
        for i in range(15):
            metrics_recorder.record_training_step(
                iteration=i,
                epoch=0,
                policy_loss=0.5 + i * 0.01,
                value_loss=0.3,
                total_loss=0.8 + i * 0.01,
                learning_rate=0.001
            )
        
        # Check moving average (window size is 10)
        avg_policy_loss = metrics_recorder.get_moving_average('policy_loss')
        expected = np.mean([0.5 + i * 0.01 for i in range(5, 15)])
        assert abs(avg_policy_loss - expected) < 1e-6
        
        # Test with custom window
        avg_5 = metrics_recorder.get_moving_average('policy_loss', window=5)
        expected_5 = np.mean([0.5 + i * 0.01 for i in range(10, 15)])
        assert abs(avg_5 - expected_5) < 1e-6
        
    def test_metric_trend(self, metrics_recorder):
        """Test trend detection"""
        # Record improving loss
        for i in range(20):
            metrics_recorder.record_training_step(
                iteration=i,
                epoch=0,
                policy_loss=1.0 - i * 0.05,  # Decreasing loss
                value_loss=0.3,
                total_loss=1.3 - i * 0.05,
                learning_rate=0.001
            )
        
        assert metrics_recorder.get_metric_trend('total_loss') == 'improving'
        
        # Record improving win rate
        for i in range(20):
            metrics_recorder.record_evaluation(
                iteration=20 + i,
                win_rate=0.5 + i * 0.02,  # Increasing win rate
                elo_rating=1500 + i * 10,
                elo_change=10
            )
        
        assert metrics_recorder.get_metric_trend('win_rate') == 'improving'
        assert metrics_recorder.get_metric_trend('elo_rating') == 'improving'
        
    def test_best_metrics_tracking(self, metrics_recorder):
        """Test tracking of best metrics"""
        # Record some metrics
        metrics_recorder.record_training_step(
            iteration=1, epoch=0,
            policy_loss=0.8, value_loss=0.4, total_loss=1.2,
            learning_rate=0.001
        )
        
        metrics_recorder.record_evaluation(
            iteration=1, win_rate=0.45, elo_rating=1400, elo_change=0
        )
        
        # Record better metrics
        metrics_recorder.record_training_step(
            iteration=5, epoch=1,
            policy_loss=0.4, value_loss=0.2, total_loss=0.6,
            learning_rate=0.001
        )
        
        metrics_recorder.record_evaluation(
            iteration=5, win_rate=0.65, elo_rating=1600, elo_change=200
        )
        
        # Check best metrics
        assert metrics_recorder.best_metrics['min_loss'] == 0.6
        assert metrics_recorder.best_metrics['min_loss_iteration'] == 5
        assert metrics_recorder.best_metrics['win_rate'] == 0.65
        assert metrics_recorder.best_metrics['best_win_rate_iteration'] == 5
        assert metrics_recorder.best_metrics['elo_rating'] == 1600
        assert metrics_recorder.best_metrics['best_elo_iteration'] == 5
        
    def test_auto_save(self, metrics_recorder, temp_metrics_dir):
        """Test automatic saving"""
        # Record metrics to trigger auto-save
        for i in range(6):  # Auto-save interval is 5
            metrics_recorder.record_training_step(
                iteration=i,
                epoch=0,
                policy_loss=0.5,
                value_loss=0.3,
                total_loss=0.8,
                learning_rate=0.001
            )
        
        # Check if file was saved
        saved_files = list(temp_metrics_dir.glob("*.json"))
        assert len(saved_files) == 1
        
    def test_save_and_load(self, metrics_recorder, temp_metrics_dir):
        """Test saving and loading metrics"""
        # Record some metrics
        for i in range(10):
            metrics_recorder.record_training_step(
                iteration=i,
                epoch=i // 5,
                policy_loss=0.5 - i * 0.01,
                value_loss=0.3,
                total_loss=0.8 - i * 0.01,
                learning_rate=0.001 * (0.9 ** i)
            )
            
            if i % 3 == 0:
                metrics_recorder.record_evaluation(
                    iteration=i,
                    win_rate=0.5 + i * 0.02,
                    elo_rating=1500 + i * 20,
                    elo_change=20
                )
        
        # Save
        save_path = temp_metrics_dir / "test_metrics.json"
        metrics_recorder.save(save_path)
        
        # Create new recorder and load
        new_recorder = TrainingMetricsRecorder()
        new_recorder.load(save_path)
        
        # Verify loaded data
        assert len(new_recorder.snapshots) == len(metrics_recorder.snapshots)
        assert new_recorder.best_metrics == metrics_recorder.best_metrics
        
        # Check a few snapshots
        for i in [0, 5, 9]:
            orig = metrics_recorder.snapshots[i]
            loaded = new_recorder.snapshots[i]
            assert orig.iteration == loaded.iteration
            assert orig.policy_loss == loaded.policy_loss
            assert orig.win_rate == loaded.win_rate
            
    def test_get_summary(self, metrics_recorder):
        """Test summary generation"""
        # Record comprehensive metrics
        for i in range(20):
            metrics_recorder.record_training_step(
                iteration=i,
                epoch=i // 10,
                policy_loss=0.8 - i * 0.02,
                value_loss=0.4 - i * 0.01,
                total_loss=1.2 - i * 0.03,
                learning_rate=0.001 * (0.95 ** i)
            )
            
            if i % 5 == 0:
                metrics_recorder.record_evaluation(
                    iteration=i,
                    win_rate=0.4 + i * 0.015,
                    elo_rating=1400 + i * 15,
                    elo_change=15
                )
                
            metrics_recorder.record_self_play_metrics(
                iteration=i,
                games_per_second=2.0 + i * 0.1,
                avg_game_length=100 + i * 2,
                policy_entropy=2.5 - i * 0.02
            )
        
        summary = metrics_recorder.get_summary()
        
        # Check structure
        assert 'current_iteration' in summary
        assert 'current_metrics' in summary
        assert 'moving_averages' in summary
        assert 'trends' in summary
        assert 'best_metrics' in summary
        assert 'performance' in summary
        
        # Check values
        assert summary['current_iteration'] == 19
        assert summary['current_epoch'] == 1
        assert summary['current_metrics']['policy_loss'] == 0.8 - 19 * 0.02
        assert summary['moving_averages']['policy_loss'] > 0
        assert summary['trends']['loss_trend'] == 'improving'
        assert summary['best_metrics']['min_loss'] < 1.2
        
    def test_export_for_plotting(self, metrics_recorder):
        """Test export format for plotting"""
        # Record metrics
        for i in range(5):
            metrics_recorder.record_training_step(
                iteration=i,
                epoch=0,
                policy_loss=0.5,
                value_loss=0.3,
                total_loss=0.8,
                learning_rate=0.001,
                custom_metric=i * 2
            )
        
        data = metrics_recorder.export_for_plotting()
        
        # Check structure
        assert 'policy_loss' in data
        assert 'value_loss' in data
        assert 'total_loss' in data
        assert 'custom_custom_metric' in data
        
        # Check data format (list of tuples)
        assert len(data['policy_loss']) == 5
        assert all(isinstance(item, tuple) and len(item) == 2 for item in data['policy_loss'])
        
        # Check custom metric values
        custom_values = [val for iter, val in data['custom_custom_metric']]
        assert custom_values == [0, 2, 4, 6, 8]
        
    def test_print_summary(self, metrics_recorder, caplog):
        """Test summary printing"""
        # Record some metrics
        for i in range(10):
            metrics_recorder.record_training_step(
                iteration=i, epoch=0,
                policy_loss=0.5, value_loss=0.3, total_loss=0.8,
                learning_rate=0.001
            )
            
        metrics_recorder.record_evaluation(
            iteration=9, win_rate=0.55, elo_rating=1500, elo_change=50
        )
        
        # Print summary
        metrics_recorder.print_summary(detailed=True)
        
        # Check output
        assert "TRAINING METRICS SUMMARY" in caplog.text
        assert "Current Metrics:" in caplog.text
        assert "Trends:" in caplog.text
        assert "Moving Averages" in caplog.text
        assert "Best Results:" in caplog.text


class TestMetricsVisualizer:
    """Test metrics visualization"""
    
    def test_visualizer_creation(self, metrics_recorder):
        """Test visualizer initialization"""
        visualizer = MetricsVisualizer(metrics_recorder)
        assert visualizer.recorder == metrics_recorder
        
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_training_curves(self, mock_savefig, mock_figure, 
                                 metrics_recorder, temp_metrics_dir):
        """Test plotting functionality"""
        # Record comprehensive metrics
        for i in range(20):
            metrics_recorder.record_training_step(
                iteration=i, epoch=0,
                policy_loss=0.5 - i * 0.01,
                value_loss=0.3 - i * 0.005,
                total_loss=0.8 - i * 0.015,
                learning_rate=0.001 * (0.9 ** i)
            )
            
            if i % 5 == 0:
                metrics_recorder.record_evaluation(
                    iteration=i,
                    win_rate=0.5 + i * 0.01,
                    elo_rating=1500 + i * 10,
                    elo_change=10
                )
                
            metrics_recorder.record_self_play_metrics(
                iteration=i,
                games_per_second=2.0,
                avg_game_length=100,
                policy_entropy=2.3
            )
        
        visualizer = MetricsVisualizer(metrics_recorder)
        save_path = temp_metrics_dir / "test_plot.png"
        
        # Should create plot without errors
        visualizer.plot_training_curves(save_path=save_path, show=False)
        
        # Check matplotlib was called
        mock_figure.assert_called_once()
        
    def test_plot_no_data(self, caplog):
        """Test plotting with no data"""
        recorder = TrainingMetricsRecorder()
        visualizer = MetricsVisualizer(recorder)
        
        with caplog.at_level("WARNING"):
            visualizer.plot_training_curves()
            
        assert "No data to plot" in caplog.text
        
    @patch('matplotlib.pyplot')
    def test_plot_no_matplotlib(self, mock_plt, metrics_recorder, caplog):
        """Test handling missing matplotlib"""
        # Make import fail
        mock_plt.side_effect = ImportError()
        
        # Record some data
        metrics_recorder.record_training_step(
            iteration=1, epoch=0,
            policy_loss=0.5, value_loss=0.3, total_loss=0.8,
            learning_rate=0.001
        )
        
        visualizer = MetricsVisualizer(metrics_recorder)
        
        with caplog.at_level("WARNING"):
            # Should handle import error gracefully
            with patch('builtins.__import__', side_effect=ImportError()):
                visualizer.plot_training_curves()
            
        assert "matplotlib not installed" in caplog.text