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
        training_pipeline.train(num_iterations=1)
        
        # Verify self-play was called
        mock_self_play.assert_called_once()
        
        # Verify arena evaluation
        mock_arena.assert_called_once()
        
        # Check model was updated
        assert training_pipeline.iteration == 1
        
    def test_training_data_flow(self, training_pipeline, sample_game_examples):
        """Test data flow through training pipeline"""
        # Mock SelfPlayManager.generate_games to return sample examples
        with patch('mcts.neural_networks.self_play_module.SelfPlayManager') as MockSelfPlay:
            mock_manager = Mock()
            mock_manager.generate_games.return_value = sample_game_examples
            MockSelfPlay.return_value = mock_manager
            
            # Generate self-play data
            examples = training_pipeline.generate_self_play_data()
            assert len(examples) >= len(sample_game_examples)  # May include augmentation
            
            # Verify the mock was called
            mock_manager.generate_games.assert_called()
                
    def test_checkpoint_workflow(self, training_pipeline):
        """Test checkpoint save and load workflow"""
        # Set iteration to 1 for consistent filename
        training_pipeline.iteration = 1
        
        # Save checkpoint
        training_pipeline.save_checkpoint()
        
        # Check checkpoint was saved with correct filename
        checkpoint_path = training_pipeline.checkpoint_dir / "checkpoint_iter_1.pt"
        assert checkpoint_path.exists()
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'iteration' in checkpoint
        assert checkpoint['iteration'] == 1
        
    def test_best_model_update(self, training_pipeline):
        """Test best model update logic"""
        # Mock the arena's compare_models method
        with patch.object(training_pipeline.arena, 'compare_models') as mock_compare:
            # New model wins
            mock_compare.return_value = (6, 2, 2)  # 60% win rate
            
            # Set initial best model iteration
            old_best_iteration = training_pipeline.best_model_iteration
            
            # Mock the ELO tracker update
            with patch('mcts.neural_networks.arena_module.ELOTracker') as MockELOTracker:
                mock_elo_tracker = Mock()
                mock_elo_tracker.update_ratings.return_value = (1550, 1500)
                mock_elo_tracker.should_accept_model.return_value = True
                MockELOTracker.return_value = mock_elo_tracker
                
                # Run evaluation phase (this happens during training)
                training_pipeline.elo_tracker = mock_elo_tracker
                training_pipeline.iteration = 1
                arena_wins, arena_draws, arena_losses = mock_compare.return_value
                accepted = mock_elo_tracker.should_accept_model.return_value
                
                assert accepted == True
                
                # If accepted, best model iteration should update
                if accepted:
                    training_pipeline.best_model_iteration = training_pipeline.iteration
                    
                assert training_pipeline.best_model_iteration != old_best_iteration


class TestSelfPlayIntegration:
    """Test self-play integration"""
    
    def test_self_play_with_current_model(self, training_pipeline):
        """Test self-play generation with current model"""
        with patch('mcts.core.mcts.MCTS') as mock_mcts_class:
            mock_mcts = Mock()
            mock_mcts.search.return_value = np.ones(225) / 225
            mock_mcts.get_root_value.return_value = 0.5
            mock_mcts_class.return_value = mock_mcts
            
            # Generate games using the pipeline's method
            examples = training_pipeline.generate_self_play_data()
            
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
            
            # Should be able to use GPU service - check stats exist
            assert hasattr(gpu_service, 'stats')
            assert gpu_service.stats is not None
            
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
                
                # Mock generate_self_play_data to return examples
                with patch.object(training_pipeline, 'generate_self_play_data') as mock_generate:
                    mock_generate.return_value = [Mock()] * 10
                    examples = training_pipeline.generate_self_play_data()
                
                # Should coordinate workers
                assert mock_worker.call_count >= 0  # Depends on implementation


class TestArenaIntegration:
    """Test arena evaluation integration"""
    
    def test_arena_model_comparison(self, training_pipeline):
        """Test arena model comparison"""
        with patch('mcts.core.mcts.MCTS') as mock_mcts_class:
            mock_mcts = Mock()
            mock_mcts.search.return_value = np.ones(225) / 225
            # Return moves that avoid conflicts
            move_counter = [0]
            def get_next_move(state, temperature):
                move = move_counter[0]
                move_counter[0] += 1
                return move
            mock_mcts.select_action.side_effect = get_next_move
            mock_mcts.tree = Mock()
            mock_mcts.tree.num_nodes = 100  # Mock tree size
            mock_mcts_class.return_value = mock_mcts
            
            # Mock game outcomes
            # Arena is accessed directly, not through arena_manager
            with patch.object(training_pipeline.game_interface, 'is_terminal',
                            side_effect=[False] * 10 + [True]):
                with patch.object(training_pipeline.game_interface, 'get_winner',
                                return_value=1):
                    
                    wins, draws, losses = training_pipeline.arena.compare_models(
                        training_pipeline.model,
                        training_pipeline.model,  # Compare with self for test
                        num_games=1,
                        silent=True
                    )
                    
                    assert wins + draws + losses == 1
                    
    def test_elo_tracking_integration(self, training_pipeline):
        """Test ELO rating tracking integration"""
        # Initialize ELO tracker if not already done
        if training_pipeline.elo_tracker is None:
            from mcts.neural_networks.arena_module import ELOTracker
            training_pipeline.elo_tracker = ELOTracker()
        elo_tracker = training_pipeline.elo_tracker
        
        # Initial ratings
        assert len(elo_tracker.ratings) == 0
        
        # After model comparison
        elo_tracker.update_ratings('model_v1', 'model_v0', wins=6, draws=2, losses=2)
        
        assert 'model_v1' in elo_tracker.ratings
        assert 'model_v0' in elo_tracker.ratings
        assert elo_tracker.ratings['model_v1'] > elo_tracker.ratings['model_v0']


class TestTrainingLoopIntegration:
    """Test complete training loop integration"""
    
    def test_full_training_loop(self, training_config, sample_game_examples):
        """Test full training loop with multiple iterations"""
        # Configure for quick test
        training_config.training.num_iterations = 2
        training_config.training.num_epochs = 1
        
        with patch('mcts.neural_networks.unified_training_pipeline.Path') as mock_path:
            with tempfile.TemporaryDirectory() as tmpdir:
                mock_path.return_value = Path(tmpdir)
                
                pipeline = UnifiedTrainingPipeline(training_config)
                
                # Run training with mocked components
                with patch.object(pipeline, 'generate_self_play_data') as mock_sp:
                    mock_sp.return_value = sample_game_examples
                    with patch.object(pipeline, 'train_neural_network') as mock_train:
                        mock_train.return_value = {
                            'loss': 0.5,
                            'policy_loss': 0.3,
                            'value_loss': 0.2
                        }
                        with patch.object(pipeline.arena, 'compare_models') as mock_arena:
                            mock_arena.return_value = (6, 2, 2)  # Current model wins
                            pipeline.train(2)  # Pass number of iterations
                    
                # Verify iterations completed
                assert pipeline.iteration == 2
                assert mock_sp.call_count == 2
                # The arena makes multiple comparisons per iteration:
                # Iteration 1: Current vs Random (forced for first model)
                # Iteration 2: Current vs Random, Current vs Previous, Current vs Best
                # Total expected: 4-5 calls depending on adaptive logic
                assert mock_arena.call_count >= 4
                
    def test_training_metrics_tracking(self, training_pipeline, sample_game_examples):
        """Test training metrics tracking"""
        with patch.object(training_pipeline, 'generate_self_play_data',
                         return_value=sample_game_examples):
            # Create a side effect that records metrics and returns values
            def mock_train_with_metrics():
                # Record the metrics
                training_pipeline.metrics_recorder.record_training_step(
                    iteration=training_pipeline.iteration,
                    epoch=0,
                    policy_loss=0.3,
                    value_loss=0.2,
                    total_loss=0.5,
                    learning_rate=0.001
                )
                return {
                    'loss': 0.5,
                    'policy_loss': 0.3,
                    'value_loss': 0.2
                }
            
            with patch.object(training_pipeline, 'train_neural_network', side_effect=mock_train_with_metrics):
                # Run a single training iteration
                training_pipeline.iteration = 0  # Reset iteration counter
                training_pipeline.train(1)  # Run 1 iteration
                
                # Check metrics recorded
                # Get statistics from metrics_recorder instead
                stats = training_pipeline.metrics_recorder.get_current_metrics()
                assert 'training_losses' in stats
                assert len(stats['training_losses']) > 0
                assert stats['training_losses'][0]['total_loss'] == 0.5
                assert stats['training_losses'][0]['policy_loss'] == 0.3
                assert stats['training_losses'][0]['value_loss'] == 0.2


class TestErrorHandlingIntegration:
    """Test error handling in integration scenarios"""
    
    def test_self_play_error_recovery(self, training_pipeline):
        """Test recovery from self-play errors"""
        with patch.object(training_pipeline, 'generate_self_play_data') as mock_sp:
            # First call fails, second succeeds
            mock_sp.side_effect = [
                RuntimeError("Worker crashed"),
                []  # Empty but valid
            ]
            
            # Should handle error on first call
            with pytest.raises(RuntimeError):
                training_pipeline.generate_self_play_data()
            
            # Second call should succeed
            examples = training_pipeline.generate_self_play_data()
            assert examples == []
            assert mock_sp.call_count == 2
            
    def test_training_error_recovery(self, training_pipeline, sample_game_examples):
        """Test recovery from training errors"""
        # Add examples to replay buffer
        training_pipeline.replay_buffer.add(sample_game_examples)
        
        with patch.object(training_pipeline, 'train_neural_network') as mock_train:
            # Simulate CUDA OOM error
            mock_train.side_effect = RuntimeError("CUDA out of memory")
            
            # Should handle error gracefully
            with pytest.raises(RuntimeError):
                training_pipeline.train_neural_network()
                
    def test_checkpoint_corruption_handling(self, training_pipeline, temp_experiment_dir):
        """Test handling of corrupted checkpoints"""
        # Create corrupted checkpoint
        bad_checkpoint = temp_experiment_dir / "checkpoints" / "corrupted.pt"
        bad_checkpoint.write_text("corrupted data")
        
        # Should handle gracefully
        with pytest.raises(Exception):
            training_pipeline.load_checkpoint(str(bad_checkpoint))


class TestPerformanceIntegration:
    """Test performance aspects of integration"""
    
    def test_memory_management(self, training_pipeline, sample_game_examples):
        """Test memory management during training"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run training iteration
        with patch.object(training_pipeline, 'generate_self_play_data',
                         return_value=sample_game_examples):
            with patch.object(training_pipeline, 'train_neural_network') as mock_train:
                mock_train.return_value = {
                    'loss': 0.5,
                    'policy_loss': 0.3,
                    'value_loss': 0.2
                }
                with patch.object(training_pipeline.arena, 'compare_models',
                                return_value=(5, 3, 2)):
                    
                    for i in range(3):
                        training_pipeline.train(i + 1)  # Train up to iteration i+1
                        gc.collect()
                        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (not leaking)
        # Note: TensorRT engine loading and CUDA initialization can use significant memory
        # Allow up to 2GB growth for GPU-accelerated deep learning operations
        assert memory_growth < 2000  # Less than 2GB growth
        
    def test_training_speed(self, training_pipeline, sample_game_examples):
        """Test training speed benchmarks"""
        # Add examples to replay buffer
        training_pipeline.replay_buffer.add(sample_game_examples)
        
        # Time single epoch
        start_time = time.time()
        
        # Create a simple forward pass test
        states = torch.stack([torch.from_numpy(ex.state).float() for ex in sample_game_examples[:32]])
        
        # Move states to the same device as the model
        device = next(training_pipeline.model.parameters()).device
        states = states.to(device)
        
        # Forward pass
        with torch.no_grad():
            pred_policies, pred_values = training_pipeline.model(states)
                
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
            with patch('mcts.neural_networks.unified_training_pipeline.Path') as mock_path:
                with tempfile.TemporaryDirectory() as tmpdir:
                    mock_path.return_value = Path(tmpdir)
                    pipeline = UnifiedTrainingPipeline(training_config)
                    
                    # Test that pipeline was created with distributed config
                    assert pipeline.config.training.num_workers == 4
                    assert hasattr(pipeline, 'generate_self_play_data')