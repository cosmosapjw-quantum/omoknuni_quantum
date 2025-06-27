"""
Comprehensive tests for ELO convergence detection and automatic training stop

Tests cover:
- Convergence detection algorithms
- Statistical plateau detection
- Trend analysis
- Auto-stop functionality
- State persistence
- Integration with training pipeline
"""

import pytest
import numpy as np
import json
import tempfile
import os
from typing import List, Tuple
from unittest.mock import Mock, patch
import logging

from mcts.neural_networks.elo_convergence import ELOConvergenceDetector, ConvergenceConfig

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestConvergenceConfig:
    """Test cases for ConvergenceConfig"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ConvergenceConfig()
        
        assert config.window_size == 10
        assert config.elo_improvement_threshold == 5.0
        assert config.convergence_patience == 5
        assert config.use_statistical_test == True
        assert config.confidence_level == 0.95
        assert config.min_iterations == 20
        assert config.min_elo_above_baseline == 100.0
        assert config.enable_auto_stop == True
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = ConvergenceConfig(
            window_size=20,
            elo_improvement_threshold=10.0,
            convergence_patience=10,
            enable_auto_stop=False
        )
        
        assert config.window_size == 20
        assert config.elo_improvement_threshold == 10.0
        assert config.convergence_patience == 10
        assert config.enable_auto_stop == False


class TestELOConvergenceDetector:
    """Test cases for ELOConvergenceDetector"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return ConvergenceConfig(
            window_size=5,
            elo_improvement_threshold=10.0,
            convergence_patience=3,
            min_iterations=5,
            min_elo_above_baseline=50.0,
            plateau_threshold=2.0,
            plateau_window=5,
            trend_window=5,
            max_negative_trend=-1.0
        )
    
    @pytest.fixture
    def detector(self, config):
        """Create detector instance"""
        return ELOConvergenceDetector(config)
    
    def test_initialization(self, detector):
        """Test detector initialization"""
        assert len(detector.elo_history) == 0
        assert len(detector.iteration_history) == 0
        assert detector.convergence_detected == False
        assert detector.should_stop == False
        assert detector.best_elo == float('-inf')
    
    def test_basic_update(self, detector):
        """Test basic ELO update"""
        result = detector.update(1, 100.0, "checkpoint_1.pt")
        
        assert result == False  # No stop yet
        assert len(detector.elo_history) == 1
        assert detector.elo_history[0] == 100.0
        assert detector.best_elo == 100.0
        assert detector.best_iteration == 1
    
    def test_improvement_stagnation_detection(self, detector):
        """Test detection of improvement stagnation"""
        # Rapid improvement initially
        elos = [0, 50, 100, 150, 200]
        for i, elo in enumerate(elos):
            detector.update(i, elo)
        
        # Then stagnation
        for i in range(5, 10):
            detector.update(i, 202.0)  # Small improvement only
        
        # Should detect convergence
        assert detector.convergence_detected == True
        assert "improvement_stagnation" in detector.stats['convergence_triggers']
    
    def test_statistical_plateau_detection(self, detector):
        """Test statistical plateau detection"""
        # Initial improvement
        for i in range(5):
            detector.update(i, i * 50.0)
        
        # Then plateau with low variance
        plateau_elos = [250.0, 251.0, 249.0, 250.5, 250.0]
        for i, elo in enumerate(plateau_elos, start=5):
            detector.update(i, elo)
        
        # Should detect plateau
        assert detector._check_statistical_plateau() == True
    
    def test_negative_trend_detection(self, detector):
        """Test negative trend detection"""
        # Rising then falling ELO
        elos = [0, 100, 200, 300, 280, 250, 220, 190, 160]
        
        for i, elo in enumerate(elos):
            result = detector.update(i, elo)
        
        # Should detect negative trend
        assert detector._check_negative_trend() == True
        assert detector.convergence_detected == True
    
    def test_minimum_performance_requirement(self, detector):
        """Test minimum performance threshold"""
        # Below minimum threshold
        for i in range(10):
            detector.update(i, 40.0)  # Below 50.0 threshold
        
        # Should not converge due to low performance
        assert detector.convergence_detected == False
        assert detector._check_minimum_performance() == False
    
    def test_convergence_patience(self, detector):
        """Test patience mechanism"""
        # Quick rise then plateau
        elos = [0, 100, 200, 250, 255, 256, 257, 258, 259]
        
        stop_results = []
        for i, elo in enumerate(elos):
            should_stop = detector.update(i, elo)
            stop_results.append(should_stop)
        
        # Should detect convergence but wait for patience
        convergence_idx = next(i for i, _ in enumerate(elos) 
                              if detector.convergence_detected)
        
        # Should stop after patience iterations
        assert stop_results[convergence_idx + detector.config.convergence_patience - 1] == True
    
    def test_best_model_tracking(self, detector):
        """Test tracking of best model"""
        elos = [100, 200, 150, 250, 180, 240]
        checkpoints = [f"ckpt_{i}.pt" for i in range(len(elos))]
        
        for i, (elo, ckpt) in enumerate(zip(elos, checkpoints)):
            detector.update(i, elo, ckpt)
        
        assert detector.best_elo == 250.0
        assert detector.best_iteration == 3
        assert detector.best_checkpoint == "ckpt_3.pt"
    
    def test_convergence_report(self, detector):
        """Test convergence report generation"""
        # Add some data
        for i in range(10):
            detector.update(i, i * 20.0)
        
        report = detector.get_convergence_report()
        
        assert report['status'] == 'training'
        assert report['current_elo'] == 180.0
        assert report['best_elo'] == 180.0
        assert report['iterations_trained'] == 10
        assert 'recent_improvement' in report
        assert 'total_improvement' in report
    
    def test_state_save_load(self, detector):
        """Test saving and loading detector state"""
        # Add some data
        for i in range(10):
            detector.update(i, i * 15.0, f"ckpt_{i}.pt")
        
        # Force convergence
        detector.convergence_detected = True
        detector.patience_counter = 2
        
        # Save state
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            detector.save_state(f.name)
            temp_path = f.name
        
        try:
            # Create new detector and load state
            new_detector = ELOConvergenceDetector(detector.config)
            new_detector.load_state(temp_path)
            
            # Verify state restored
            assert new_detector.elo_history == detector.elo_history
            assert new_detector.convergence_detected == True
            assert new_detector.patience_counter == 2
            assert new_detector.best_elo == detector.best_elo
            assert len(new_detector.recent_elos) == len(detector.recent_elos)
        finally:
            os.unlink(temp_path)
    
    def test_plot_generation(self, detector):
        """Test plot generation (smoke test)"""
        # Add data with convergence
        for i in range(20):
            if i < 10:
                elo = i * 20.0
            else:
                elo = 180.0 + np.random.normal(0, 2)
            detector.update(i, elo)
        
        # Force convergence detection
        detector.convergence_detected = True
        detector.convergence_iteration = 12
        
        # Test plot generation (won't display)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        try:
            detector.plot_convergence(temp_path)
            # Verify file created
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestConvergenceScenarios:
    """Test various convergence scenarios"""
    
    @pytest.fixture
    def detector(self):
        """Create detector with standard config"""
        config = ConvergenceConfig(
            window_size=10,
            elo_improvement_threshold=5.0,
            convergence_patience=5,
            min_iterations=10
        )
        return ELOConvergenceDetector(config)
    
    def test_steady_improvement_no_convergence(self, detector):
        """Test steady improvement should not trigger convergence"""
        # Steady improvement
        for i in range(50):
            elo = i * 10.0  # Consistent improvement
            should_stop = detector.update(i, elo)
            assert should_stop == False
        
        assert detector.convergence_detected == False
    
    def test_early_plateau_then_breakthrough(self, detector):
        """Test early plateau followed by breakthrough"""
        # Initial rise
        for i in range(10):
            detector.update(i, i * 20.0)
        
        # Plateau
        for i in range(10, 20):
            detector.update(i, 180.0 + np.random.uniform(-2, 2))
        
        # Breakthrough
        for i in range(20, 30):
            detector.update(i, 200.0 + (i - 20) * 10.0)
        
        # Should reset convergence detection on breakthrough
        assert detector.patience_counter == 0
    
    def test_oscillating_performance(self, detector):
        """Test oscillating performance pattern"""
        # Oscillating pattern
        for i in range(30):
            if i % 10 < 5:
                elo = 100.0 + i * 2.0  # Upward
            else:
                elo = 150.0 - (i % 10 - 5) * 5.0  # Downward
            
            detector.update(i, elo)
        
        # Might detect convergence due to low average improvement
        # This tests the robustness of detection
        report = detector.get_convergence_report()
        logger.info(f"Oscillating pattern report: {report}")
    
    def test_rapid_convergence(self, detector):
        """Test rapid convergence scenario"""
        # Very fast initial improvement
        elos = [0, 200, 250, 260, 265, 267, 268, 269, 270, 270]
        
        for i, elo in enumerate(elos):
            detector.update(i, elo)
        
        # Continue with minimal improvement
        for i in range(10, 20):
            should_stop = detector.update(i, 270.0 + np.random.uniform(-1, 1))
        
        assert detector.convergence_detected == True
        assert detector.should_stop == True


class TestIntegrationWithTraining:
    """Test integration with training pipeline"""
    
    def test_training_pipeline_integration(self):
        """Test integration with UnifiedTrainingPipeline"""
        from mcts.neural_networks.unified_training_pipeline import UnifiedTrainingPipeline
        from mcts.utils.config_system import AlphaZeroConfig
        
        # Create minimal config
        config = AlphaZeroConfig(
            experiment_name="test_convergence",
            num_iterations=5,
            game_type="gomoku",
            convergence={
                'enable_auto_stop': True,
                'window_size': 3,
                'convergence_patience': 2,
                'min_iterations': 3
            }
        )
        
        # Mock components
        with patch('mcts.neural_networks.unified_training_pipeline.ArenaModule') as mock_arena:
            with patch('mcts.neural_networks.unified_training_pipeline.ModelManager'):
                # Set up mock arena to return increasing ELOs
                mock_arena_instance = Mock()
                mock_arena_instance.evaluate_model.return_value = 100.0
                mock_arena.return_value = mock_arena_instance
                
                # Create pipeline
                pipeline = UnifiedTrainingPipeline(config)
                
                # Verify convergence detector created
                assert pipeline.convergence_detector is not None
                assert pipeline.convergence_detector.config.enable_auto_stop == True
                assert pipeline.convergence_detector.config.window_size == 3
    
    def test_auto_stop_behavior(self):
        """Test automatic training stop behavior"""
        config = ConvergenceConfig(
            enable_auto_stop=True,
            window_size=3,
            convergence_patience=2,
            min_iterations=3,
            elo_improvement_threshold=5.0
        )
        
        detector = ELOConvergenceDetector(config)
        
        # Simulate training with convergence
        elos = [0, 50, 100, 102, 103, 104, 105]  # Plateau after iteration 2
        
        stop_signals = []
        for i, elo in enumerate(elos):
            should_stop = detector.update(i, elo)
            stop_signals.append(should_stop)
            
            if should_stop:
                logger.info(f"Training stopped at iteration {i} with ELO {elo}")
                break
        
        # Should stop after detecting plateau + patience
        assert any(stop_signals)
        assert detector.should_stop == True
    
    def test_disabled_auto_stop(self):
        """Test behavior when auto-stop is disabled"""
        config = ConvergenceConfig(
            enable_auto_stop=False,
            convergence_patience=2,
            min_iterations=3
        )
        
        detector = ELOConvergenceDetector(config)
        
        # Add converged data
        for i in range(10):
            elo = 100.0 if i < 5 else 101.0
            should_stop = detector.update(i, elo)
            assert should_stop == False  # Should never stop
        
        # Convergence might be detected but no stop
        assert detector.should_stop == True  # Internal flag
        assert detector.update(10, 101.0) == False  # But returns False


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_detector_operations(self):
        """Test operations on empty detector"""
        detector = ELOConvergenceDetector(ConvergenceConfig())
        
        # Should handle gracefully
        assert detector._check_improvement_stagnation() == False
        assert detector._check_statistical_plateau() == False
        assert detector._check_negative_trend() == False
        assert detector._check_minimum_performance() == False
        
        report = detector.get_convergence_report()
        assert report['status'] == 'no_data'
    
    def test_single_data_point(self):
        """Test with single data point"""
        detector = ELOConvergenceDetector(ConvergenceConfig())
        
        should_stop = detector.update(0, 100.0)
        assert should_stop == False
        assert detector.best_elo == 100.0
        
        report = detector.get_convergence_report()
        assert report['current_elo'] == 100.0
    
    def test_negative_elo_values(self):
        """Test with negative ELO values"""
        detector = ELOConvergenceDetector(
            ConvergenceConfig(min_elo_above_baseline=-50.0)
        )
        
        # Negative ELOs (can happen early in training)
        elos = [-100, -50, -20, 0, 10, 20]
        
        for i, elo in enumerate(elos):
            detector.update(i, elo)
        
        # Should track improvement correctly
        assert detector.best_elo == 20.0
        report = detector.get_convergence_report()
        assert report['total_improvement'] == 120.0
    
    def test_large_elo_jumps(self):
        """Test with large ELO jumps"""
        detector = ELOConvergenceDetector(ConvergenceConfig())
        
        # Large jumps (could indicate evaluation issues)
        elos = [0, 500, 100, 600, 200, 700]
        
        for i, elo in enumerate(elos):
            detector.update(i, elo)
        
        # Should still track best
        assert detector.best_elo == 700.0
    
    def test_identical_elos(self):
        """Test with identical ELO values"""
        config = ConvergenceConfig(
            plateau_threshold=0.1,  # Very low threshold
            plateau_window=5
        )
        detector = ELOConvergenceDetector(config)
        
        # Exactly identical ELOs
        for i in range(10):
            detector.update(i, 100.0)
        
        # Should detect as plateau
        assert detector._check_statistical_plateau() == True
        assert np.var(detector.elo_history[-5:]) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])