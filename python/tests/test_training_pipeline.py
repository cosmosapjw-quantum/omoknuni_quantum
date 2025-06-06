"""Tests for training pipeline"""

import pytest
import torch
import numpy as np
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
import multiprocessing as mp

from mcts.neural_networks.training_pipeline import (
    TrainingConfig, GameExample, ReplayBuffer, 
    play_self_play_game, TrainingPipeline,
    create_training_pipeline
)
from mcts.mcts import MCTS, MCTSConfig
from mcts.game_interface import GameInterface
from mcts.nn_model import AlphaZeroNetwork, ModelConfig


class TestTrainingConfig:
    """Test TrainingConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = TrainingConfig()
        assert config.game_type == 'gomoku'
        assert config.batch_size == 512
        assert config.learning_rate == 0.01
        assert config.num_workers == 4
        assert config.mcts_simulations == 800
        assert config.device in ['cuda', 'cpu']
        
    def test_custom_config(self):
        """Test custom configuration"""
        config = TrainingConfig(
            game_type='chess',
            batch_size=256,
            learning_rate=0.001,
            num_workers=8,
            device='cpu'
        )
        assert config.game_type == 'chess'
        assert config.batch_size == 256
        assert config.learning_rate == 0.001
        assert config.num_workers == 8
        assert config.device == 'cpu'


class TestGameExample:
    """Test GameExample dataclass"""
    
    def test_game_example(self):
        """Test game example creation"""
        state = np.random.randn(8, 8, 17)
        policy = np.random.rand(100)
        policy /= policy.sum()
        
        example = GameExample(
            state=state,
            policy=policy,
            value=0.5,
            game_id='test_game_1',
            move_number=10
        )
        
        assert np.array_equal(example.state, state)
        assert np.array_equal(example.policy, policy)
        assert example.value == 0.5
        assert example.game_id == 'test_game_1'
        assert example.move_number == 10


class TestReplayBuffer:
    """Test ReplayBuffer class"""
    
    def test_add_examples(self):
        """Test adding examples to buffer"""
        buffer = ReplayBuffer(max_size=100)
        
        examples = []
        for i in range(10):
            example = GameExample(
                state=np.random.randn(8, 8, 17),
                policy=np.random.rand(100),
                value=float(i) / 10,
                game_id=f'game_{i}',
                move_number=i
            )
            examples.append(example)
            
        buffer.add_game(examples)
        assert len(buffer) == 10
        
    def test_max_size(self):
        """Test buffer respects max size"""
        buffer = ReplayBuffer(max_size=5)
        
        examples = []
        for i in range(10):
            example = GameExample(
                state=np.random.randn(8, 8, 17),
                policy=np.random.rand(100),
                value=0.0,
                game_id=f'game_{i}',
                move_number=i
            )
            examples.append(example)
            
        buffer.add_game(examples)
        assert len(buffer) == 5  # Should keep only last 5
        
    def test_getitem(self):
        """Test getting items from buffer"""
        buffer = ReplayBuffer()
        
        example = GameExample(
            state=np.random.randn(8, 8, 17).astype(np.float32),
            policy=np.random.rand(100).astype(np.float32),
            value=0.5,
            game_id='test',
            move_number=0
        )
        
        buffer.add_game([example])
        
        state, policy, value = buffer[0]
        assert isinstance(state, torch.Tensor)
        assert isinstance(policy, torch.Tensor)
        assert isinstance(value, torch.Tensor)
        assert state.shape == (8, 8, 17)
        assert policy.shape == (100,)
        assert value.item() == 0.5
        
    def test_save_load(self):
        """Test saving and loading buffer"""
        buffer = ReplayBuffer()
        
        # Add some examples
        examples = []
        for i in range(5):
            example = GameExample(
                state=np.random.randn(8, 8, 17),
                policy=np.random.rand(100),
                value=float(i),
                game_id=f'game_{i}',
                move_number=i
            )
            examples.append(example)
            
        buffer.add_game(examples)
        
        # Save and load
        with tempfile.NamedTemporaryFile(suffix='.pkl') as f:
            buffer.save(f.name)
            
            new_buffer = ReplayBuffer()
            new_buffer.load(f.name)
            
            assert len(new_buffer) == len(buffer)
            
            # Check content
            for i in range(len(buffer)):
                old_example = buffer.buffer[i]
                new_example = new_buffer.buffer[i]
                assert np.array_equal(old_example.state, new_example.state)
                assert np.array_equal(old_example.policy, new_example.policy)
                assert old_example.value == new_example.value


class TestSelfPlay:
    """Test self-play functionality"""
    
    @pytest.fixture
    def mock_game(self):
        """Create mock game"""
        game = Mock()
        game.is_terminal.side_effect = [False, False, False, True]
        game.get_nn_input.return_value = np.random.randn(8, 8, 17)
        game.get_state.return_value = Mock()
        game.get_reward.return_value = 1.0
        game.reset.return_value = None
        game.make_move.return_value = None
        return game
        
    @pytest.fixture
    def mock_mcts(self):
        """Create mock MCTS"""
        mcts = Mock(spec=MCTS)
        # Return normalized probabilities
        probs = np.random.rand(100)
        probs /= probs.sum()
        mcts.search.return_value = probs
        return mcts
        
    def test_play_self_play_game(self, mock_game, mock_mcts):
        """Test playing a self-play game"""
        config = TrainingConfig(
            max_moves_per_game=10,
            temperature_threshold=2,
            mcts_simulations=100
        )
        
        examples = play_self_play_game(
            mock_game, mock_mcts, config, 'test_game_1'
        )
        
        # Should have 3 examples (game terminates on 4th check)
        assert len(examples) == 3
        
        # Check examples
        for i, example in enumerate(examples):
            assert isinstance(example, GameExample)
            assert example.game_id == 'test_game_1'
            assert example.move_number == i
            assert example.value != 0.0  # Values should be filled
            
        # Check MCTS was called correctly
        assert mock_mcts.search.call_count == 3
        
        # Check game methods were called
        assert mock_game.reset.called
        assert mock_game.make_move.call_count == 3
        
    def test_temperature_switching(self, mock_game, mock_mcts):
        """Test temperature switching during game"""
        config = TrainingConfig(
            temperature_threshold=2,
            mcts_simulations=100
        )
        
        # Make game last longer
        mock_game.is_terminal.side_effect = [False] * 5 + [True]
        
        examples = play_self_play_game(
            mock_game, mock_mcts, config, 'test_game_2'
        )
        
        assert len(examples) == 5
        
        # Check MCTS configs
        mcts_calls = mock_mcts.search.call_args_list
        
        # First two moves should have temperature 1.0
        for i in range(2):
            config_arg = mcts_calls[i][0][1]
            assert config_arg.temperature == 1.0
            
        # Remaining moves should have temperature 0.1
        for i in range(2, 5):
            config_arg = mcts_calls[i][0][1]
            assert config_arg.temperature == 0.1


class TestTrainingPipeline:
    """Test TrainingPipeline class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
            
    @pytest.fixture
    def pipeline(self, temp_dir):
        """Create training pipeline"""
        config = TrainingConfig(
            game_type='gomoku',
            batch_size=32,
            save_dir=temp_dir,
            device='cpu',
            num_epochs=2
        )
        return TrainingPipeline(config)
        
    def test_initialization(self, pipeline):
        """Test pipeline initialization"""
        assert pipeline.iteration == 0
        assert isinstance(pipeline.replay_buffer, ReplayBuffer)
        assert isinstance(pipeline.model, AlphaZeroNetwork)
        assert pipeline.model.training
        
    def test_save_checkpoint(self, pipeline, temp_dir):
        """Test saving checkpoint"""
        pipeline.iteration = 5
        
        # Add some fake data to replay buffer
        examples = []
        for i in range(10):
            example = GameExample(
                state=np.random.randn(15, 15, 4),
                policy=np.random.rand(225),
                value=0.0,
                game_id=f'game_{i}',
                move_number=i
            )
            examples.append(example)
        pipeline.replay_buffer.add_game(examples)
        
        pipeline.save_checkpoint()
        
        # Check files exist
        checkpoint_path = os.path.join(temp_dir, 'checkpoint_iter_5.pt')
        buffer_path = os.path.join(temp_dir, 'replay_buffer_iter_5.pkl')
        
        assert os.path.exists(checkpoint_path)
        assert os.path.exists(buffer_path)
        
    def test_load_checkpoint(self, pipeline, temp_dir):
        """Test loading checkpoint"""
        # Save checkpoint first
        pipeline.iteration = 10
        pipeline.save_checkpoint()
        
        # Create new pipeline and load
        new_pipeline = TrainingPipeline(pipeline.config)
        checkpoint_path = os.path.join(temp_dir, 'checkpoint_iter_10.pt')
        new_pipeline.load_checkpoint(checkpoint_path)
        
        assert new_pipeline.iteration == 10
        
    def test_train_network(self, pipeline):
        """Test network training"""
        # Add training data
        examples = []
        for i in range(100):
            state = np.random.randn(4, 15, 15).astype(np.float32)
            policy = np.random.rand(225).astype(np.float32)
            policy /= policy.sum()
            
            example = GameExample(
                state=state,
                policy=policy,
                value=np.random.uniform(-1, 1),
                game_id=f'game_{i}',
                move_number=i
            )
            examples.append(example)
            
        pipeline.replay_buffer.add_game(examples)
        
        # Train
        metrics = pipeline.train_network()
        
        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert 'total_loss' in metrics
        assert all(v > 0 for v in metrics.values())
        
    def test_train_network_insufficient_data(self, pipeline):
        """Test training with insufficient data"""
        # Add only a few examples
        examples = []
        for i in range(5):
            example = GameExample(
                state=np.random.randn(15, 15, 4),
                policy=np.random.rand(225),
                value=0.0,
                game_id=f'game_{i}',
                move_number=i
            )
            examples.append(example)
            
        pipeline.replay_buffer.add_game(examples)
        
        # Should return None without training
        result = pipeline.train_network()
        assert result is None


class TestCreateTrainingPipeline:
    """Test factory function"""
    
    def test_create_default(self):
        """Test creating pipeline with default config"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('mcts.training_pipeline.TrainingConfig') as MockConfig:
                MockConfig.return_value = TrainingConfig(save_dir=tmpdir)
                pipeline = create_training_pipeline()
                
        assert isinstance(pipeline, TrainingPipeline)
        
    def test_create_custom(self):
        """Test creating pipeline with custom config"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                game_type='chess',
                batch_size=128,
                save_dir=tmpdir
            )
            pipeline = create_training_pipeline(config)
            
        assert isinstance(pipeline, TrainingPipeline)
        assert pipeline.config.game_type == 'chess'
        assert pipeline.config.batch_size == 128


class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.slow
    def test_full_training_iteration(self):
        """Test a full training iteration (marked slow)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                game_type='gomoku',
                batch_size=32,
                num_games_per_iteration=4,
                num_workers=2,
                num_epochs=1,
                save_dir=tmpdir,
                device='cpu'
            )
            
            pipeline = TrainingPipeline(config)
            
            # Mock game class
            mock_game_class = Mock()
            mock_game_instance = Mock(spec=GameInterface)
            mock_game_class.return_value = mock_game_instance
            
            # Mock the worker function to avoid multiprocessing issues in tests
            def mock_worker(worker_id, game_class, model_path, config, num_games, result_queue):
                examples = []
                for i in range(num_games):
                    for j in range(10):  # 10 moves per game
                        example = GameExample(
                            state=np.random.randn(4, 15, 15),
                            policy=np.random.rand(225),
                            value=np.random.uniform(-1, 1),
                            game_id=f'worker_{worker_id}_game_{i}',
                            move_number=j
                        )
                        examples.append(example)
                result_queue.put((worker_id, examples))
                
            with patch('mcts.training_pipeline.run_self_play_worker', mock_worker):
                # Generate self-play data
                examples = pipeline.generate_self_play_data(mock_game_class)
                
                assert len(examples) > 0
                pipeline.replay_buffer.add_game(examples)
                
                # Train network
                metrics = pipeline.train_network()
                assert metrics is not None
                
                # Save checkpoint
                pipeline.save_checkpoint()
                
                # Check checkpoint exists
                checkpoint_path = os.path.join(tmpdir, 'checkpoint_iter_0.pt')
                assert os.path.exists(checkpoint_path)