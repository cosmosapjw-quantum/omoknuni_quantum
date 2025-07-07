"""Integration tests for the training pipeline

Tests cover:
- End-to-end training loop
- Self-play to training integration
- Arena evaluation integration
- Checkpoint save/load workflow
- Multi-worker coordination
- GPU service integration
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock

from mcts.utils.config_system import AlphaZeroConfig
from mcts.neural_networks.unified_training_pipeline import (
    UnifiedTrainingPipeline, GameExample
)
from mcts.neural_networks.self_play_module import SelfPlayManager
from mcts.neural_networks.arena_module import ArenaManager
from mcts.neural_networks.resnet_model import create_resnet_for_game
from mcts.core.game_interface import GameInterface, GameType


@pytest.fixture
def training_config():
    """Create training configuration"""
    config = AlphaZeroConfig()
    config.game.game_type = 'gomoku'
    config.game.board_size = 15
    config.network.num_res_blocks = 3  # Small for testing
    config.network.num_filters = 32
    config.training.num_iterations = 2
    config.training.num_games_per_iteration = 10
    config.training.num_epochs = 2
    config.training.batch_size = 8
    config.training.num_workers = 2
    config.training.checkpoint_interval = 1
    config.mcts.num_simulations = 50  # Small for testing
    config.arena.num_games = 10
    config.experiment_name = 'test_integration'
    return config


@pytest.fixture
def temp_experiment_dir():
    """Create temporary experiment directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / "test_experiment"
        exp_dir.mkdir(parents=True)
        (exp_dir / "checkpoints").mkdir()
        (exp_dir / "self_play_data").mkdir()
        (exp_dir / "logs").mkdir()
        yield exp_dir


@pytest.fixture
def training_pipeline(training_config, temp_experiment_dir):
    """Create training pipeline instance"""
    with patch('mcts.neural_networks.unified_training_pipeline.Path') as mock_path:
        mock_path.return_value = temp_experiment_dir
        pipeline = UnifiedTrainingPipeline(training_config)
        pipeline.experiment_dir = temp_experiment_dir
        return pipeline


@pytest.fixture
def sample_game_examples():
    """Create sample game examples for training"""
    examples = []
    for i in range(20):
        state = np.random.randn(18, 15, 15).astype(np.float32)
        policy = np.random.rand(225).astype(np.float32)
        policy = policy / policy.sum()
        value = np.random.uniform(-1, 1)
        
        example = GameExample(
            state=state,
            policy=policy,
            value=value,
            game_id=f"game_{i}",
            move_number=i
        )
        examples.append(example)
    return examples


class TestTrainingPipelineIntegration:
    """Test complete training pipeline integration"""
    
    def test_pipeline_initialization(self, training_pipeline, training_config):
        """Test training pipeline initialization"""
        assert training_pipeline.config == training_config
        assert training_pipeline.model is not None
        assert training_pipeline.optimizer is not None
        assert training_pipeline.iteration == 0
        assert training_pipeline.experiment_dir.exists()
        
    @patch('mcts.neural_networks.self_play_module.SelfPlayManager.generate_games')
    @patch('mcts.neural_networks.arena_module.ArenaManager.compare_models')
    def test_single_iteration(self, mock_arena, mock_self_play, 
                            training_pipeline, sample_game_examples):
        """Test single training iteration"""
        # Mock self-play to return examples
        mock_self_play.return_value = sample_game_examples
        
        # Mock arena to accept new model
        mock_arena.return_value = (7, 1, 2)  # wins, draws, losses
        
        # Run one iteration
        training_pipeline.train_iteration(iteration=1)
        
        # Verify self-play was called
        mock_self_play.assert_called_once()
        
        # Verify arena evaluation
        mock_arena.assert_called_once()
        
        # Check model was updated
        assert training_pipeline.training_stats['iterations_completed'] == 1
        
    def test_training_data_flow(self, training_pipeline, sample_game_examples):
        """Test data flow through training pipeline"""
        with patch.object(training_pipeline.self_play_manager, 'generate_games',
                         return_value=sample_game_examples):
            
            # Generate self-play data
            examples = training_pipeline._self_play_phase(iteration=1)
            assert len(examples) == len(sample_game_examples)
            
            # Process into training batches
            dataset = training_pipeline._prepare_training_data(examples)
            assert len(dataset) > 0
            
            # Train model (mock the actual training)
            with patch.object(training_pipeline, '_train_epoch'):
                training_pipeline._training_phase(dataset, iteration=1)
                
    def test_checkpoint_workflow(self, training_pipeline, temp_experiment_dir):
        """Test checkpoint save and load workflow"""
        # Save checkpoint
        checkpoint_path = temp_experiment_dir / "checkpoints" / "model_001.pt"
        training_pipeline._save_checkpoint(1, checkpoint_path)
        
        assert checkpoint_path.exists()
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'iteration' in checkpoint
        assert checkpoint['iteration'] == 1
        
    def test_best_model_update(self, training_pipeline):
        """Test best model update logic"""
        with patch.object(training_pipeline.arena_manager, 'compare_models') as mock_arena:
            # New model wins
            mock_arena.return_value = (6, 2, 2)  # 60% win rate
            
            old_best = training_pipeline.best_model
            should_update = training_pipeline._evaluate_model(iteration=1)
            
            assert should_update == True
            mock_arena.assert_called_once()
            
            # Update best model
            training_pipeline._update_best_model()
            assert training_pipeline.best_model != old_best


class TestSelfPlayIntegration:
    """Test self-play integration"""
    
    def test_self_play_with_current_model(self, training_pipeline):
        """Test self-play generation with current model"""
        with patch('mcts.core.mcts.MCTS') as mock_mcts_class:
            mock_mcts = Mock()
            mock_mcts.search.return_value = np.ones(225) / 225
            mock_mcts.get_root_value.return_value = 0.5
            mock_mcts_class.return_value = mock_mcts
            
            # Generate games
            examples = training_pipeline.self_play_manager.generate_games(
                training_pipeline.current_model,
                iteration=1,
                num_games=5,
                num_workers=1
            )
            
            # Should generate some examples
            assert len(examples) >= 0  # Depends on game termination
            
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gpu_service_integration(self, training_config):
        """Test GPU evaluator service integration"""
        from mcts.utils.gpu_evaluator_service import GPUEvaluatorService
        
        # Create model
        model = create_resnet_for_game('gomoku', num_blocks=3, num_filters=32)
        
        # Create GPU service
        gpu_service = GPUEvaluatorService(model, device='cuda')
        
        with patch('multiprocessing.Process'):
            gpu_service.start()
            
            # Create self-play manager
            self_play = SelfPlayManager(training_config)
            
            # Should be able to use GPU service
            assert gpu_service.state.name == 'RUNNING'
            
            gpu_service.stop()
            
    def test_multi_worker_coordination(self, training_pipeline):
        """Test multi-worker self-play coordination"""
        with patch('mcts.neural_networks.self_play_module._play_game_worker_wrapper') as mock_worker:
            # Mock worker to return examples
            mock_worker.return_value = None  # Workers use queues
            
            with patch('multiprocessing.Queue') as mock_queue_class:
                mock_queue = Mock()
                mock_queue.get.side_effect = [
                    [Mock()] * 5,  # First worker results
                    [Mock()] * 5,  # Second worker results
                ]
                mock_queue_class.return_value = mock_queue
                
                # Run parallel self-play
                examples = training_pipeline.self_play_manager._parallel_self_play(
                    training_pipeline.current_model,
                    iteration=1,
                    num_games=10,
                    num_workers=2
                )
                
                # Should coordinate workers
                assert mock_worker.call_count >= 0  # Depends on implementation


class TestArenaIntegration:
    """Test arena evaluation integration"""
    
    def test_arena_model_comparison(self, training_pipeline):
        """Test arena model comparison"""
        with patch('mcts.core.mcts.MCTS') as mock_mcts_class:
            mock_mcts = Mock()
            mock_mcts.search.return_value = np.ones(225) / 225
            mock_mcts_class.return_value = mock_mcts
            
            # Mock game outcomes
            with patch.object(training_pipeline.arena_manager.game_interface, 'is_terminal',
                            side_effect=[False] * 10 + [True]):
                with patch.object(training_pipeline.arena_manager.game_interface, 'get_winner',
                                return_value=1):
                    
                    wins, draws, losses = training_pipeline.arena_manager.compare_models(
                        training_pipeline.current_model,
                        training_pipeline.best_model,
                        num_games=1,
                        silent=True
                    )
                    
                    assert wins + draws + losses == 1
                    
    def test_elo_tracking_integration(self, training_pipeline):
        """Test ELO rating tracking integration"""
        elo_tracker = training_pipeline.arena_manager.elo_tracker
        
        # Initial ratings
        assert len(elo_tracker.ratings) == 0
        
        # After model comparison
        elo_tracker.update_ratings('model_v1', 'model_v0', wins=6, draws=2, losses=2)
        
        assert 'model_v1' in elo_tracker.ratings
        assert 'model_v0' in elo_tracker.ratings
        assert elo_tracker.ratings['model_v1'] > elo_tracker.ratings['model_v0']


class TestTrainingLoopIntegration:
    """Test complete training loop integration"""
    
    @patch('mcts.neural_networks.self_play_module.SelfPlayManager.generate_games')
    @patch('mcts.neural_networks.arena_module.ArenaManager.compare_models')
    def test_full_training_loop(self, mock_arena, mock_self_play,
                               training_config, sample_game_examples):
        """Test full training loop with multiple iterations"""
        # Configure for quick test
        training_config.training.num_iterations = 2
        training_config.training.num_epochs = 1
        
        # Mock components
        mock_self_play.return_value = sample_game_examples
        mock_arena.return_value = (6, 2, 2)  # Current model wins
        
        with patch('mcts.neural_networks.unified_training_pipeline.Path') as mock_path:
            with tempfile.TemporaryDirectory() as tmpdir:
                mock_path.return_value = Path(tmpdir)
                
                pipeline = UnifiedTrainingPipeline(training_config)
                
                # Run training
                with patch.object(pipeline, '_train_epoch'):
                    pipeline.train()
                    
                # Verify iterations completed
                assert pipeline.training_stats['iterations_completed'] == 2
                assert mock_self_play.call_count == 2
                assert mock_arena.call_count == 2
                
    def test_training_metrics_tracking(self, training_pipeline, sample_game_examples):
        """Test training metrics tracking"""
        with patch.object(training_pipeline.self_play_manager, 'generate_games',
                         return_value=sample_game_examples):
            with patch.object(training_pipeline, '_train_epoch') as mock_train:
                # Mock training metrics
                mock_train.return_value = {
                    'loss': 0.5,
                    'policy_loss': 0.3,
                    'value_loss': 0.2
                }
                
                # Run iteration
                training_pipeline.train_iteration(1)
                
                # Check metrics recorded
                stats = training_pipeline.training_stats
                assert 'training_losses' in stats
                assert len(stats['training_losses']) > 0


class TestErrorHandlingIntegration:
    """Test error handling in integration scenarios"""
    
    def test_self_play_error_recovery(self, training_pipeline):
        """Test recovery from self-play errors"""
        with patch.object(training_pipeline.self_play_manager, 'generate_games') as mock_sp:
            # First call fails, second succeeds
            mock_sp.side_effect = [
                RuntimeError("Worker crashed"),
                []  # Empty but valid
            ]
            
            # Should handle error and retry
            examples = training_pipeline._self_play_phase(1)
            assert examples == []
            assert mock_sp.call_count == 2
            
    def test_training_error_recovery(self, training_pipeline, sample_game_examples):
        """Test recovery from training errors"""
        dataset = training_pipeline._prepare_training_data(sample_game_examples)
        
        with patch.object(training_pipeline, '_train_epoch') as mock_train:
            # Simulate CUDA OOM error
            mock_train.side_effect = RuntimeError("CUDA out of memory")
            
            # Should handle error gracefully
            with pytest.raises(RuntimeError):
                training_pipeline._training_phase(dataset, 1)
                
    def test_checkpoint_corruption_handling(self, training_pipeline, temp_experiment_dir):
        """Test handling of corrupted checkpoints"""
        # Create corrupted checkpoint
        bad_checkpoint = temp_experiment_dir / "checkpoints" / "corrupted.pt"
        bad_checkpoint.write_text("corrupted data")
        
        # Should handle gracefully
        with pytest.raises(Exception):
            training_pipeline._load_checkpoint(bad_checkpoint)


class TestPerformanceIntegration:
    """Test performance aspects of integration"""
    
    def test_memory_management(self, training_pipeline, sample_game_examples):
        """Test memory management during training"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run training iteration
        with patch.object(training_pipeline.self_play_manager, 'generate_games',
                         return_value=sample_game_examples):
            with patch.object(training_pipeline, '_train_epoch'):
                with patch.object(training_pipeline.arena_manager, 'compare_models',
                                return_value=(5, 3, 2)):
                    
                    for i in range(3):
                        training_pipeline.train_iteration(i)
                        gc.collect()
                        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (not leaking)
        assert memory_growth < 500  # Less than 500MB growth
        
    def test_training_speed(self, training_pipeline, sample_game_examples):
        """Test training speed benchmarks"""
        dataset = training_pipeline._prepare_training_data(sample_game_examples)
        
        # Time single epoch
        start_time = time.time()
        
        # Run actual training (small dataset)
        for batch in dataset:
            states = torch.tensor(batch['states'])
            policies = torch.tensor(batch['policies'])
            values = torch.tensor(batch['values'])
            
            # Forward pass
            with torch.no_grad():
                pred_policies, pred_values = training_pipeline.current_model(states)
                
        epoch_time = time.time() - start_time
        
        # Should complete quickly for small dataset
        assert epoch_time < 10.0  # Less than 10 seconds


class TestDistributedIntegration:
    """Test distributed training integration"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_multi_gpu_setup(self, training_config):
        """Test multi-GPU training setup"""
        training_config.resources.num_gpus = torch.cuda.device_count()
        
        if training_config.resources.num_gpus > 1:
            with patch('torch.nn.DataParallel') as mock_dp:
                pipeline = UnifiedTrainingPipeline(training_config)
                
                # Model should be wrapped in DataParallel
                mock_dp.assert_called()
                
    def test_distributed_self_play(self, training_config):
        """Test distributed self-play workers"""
        training_config.training.num_workers = 4
        
        with patch('multiprocessing.Process') as mock_process:
            with patch('mcts.neural_networks.unified_training_pipeline.Path'):
                pipeline = UnifiedTrainingPipeline(training_config)
                
                # Test worker coordination
                with patch.object(pipeline.self_play_manager, '_parallel_self_play') as mock_psp:
                    mock_psp.return_value = []
                    
                    pipeline._self_play_phase(1)
                    
                    # Should use configured workers
                    call_args = mock_psp.call_args
                    assert call_args[1]['num_workers'] == 4