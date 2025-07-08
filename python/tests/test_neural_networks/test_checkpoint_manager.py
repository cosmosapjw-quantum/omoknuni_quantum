"""Tests for the checkpoint manager module"""

import pytest
import torch
import numpy as np
from pathlib import Path
import json
import shutil

from mcts.neural_networks.checkpoint_manager import CheckpointManager
from mcts.neural_networks.resnet_model import ResNetModel, ResNetConfig
from mcts.neural_networks.replay_buffer import ReplayBuffer, GameExample
from mcts.utils.config_system import create_default_config


class TestCheckpointManager:
    """Test the CheckpointManager class"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        config = create_default_config('gomoku')
        config.training.checkpoint_every_n_iterations = 5
        config.training.keep_last_n_checkpoints = 3
        config.training.keep_last_n_replay_buffers = 2
        return config
    
    @pytest.fixture
    def checkpoint_dir(self, tmp_path):
        """Create temporary checkpoint directory"""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        return checkpoint_dir
    
    @pytest.fixture
    def model(self):
        """Create test model"""
        config = ResNetConfig(
            input_channels=2,
            num_blocks=5,
            num_filters=64,
            fc_value_hidden=256,
            fc_policy_hidden=256
        )
        return ResNetModel(
            config=config,
            board_size=15,
            num_actions=225,
            game_type='gomoku'
        )
    
    def test_checkpoint_manager_initialization(self, config, checkpoint_dir):
        """Test checkpoint manager initialization"""
        manager = CheckpointManager(
            checkpoint_dir=str(checkpoint_dir),
            config=config
        )
        
        assert manager.checkpoint_dir == checkpoint_dir
        assert manager.checkpoint_every_n_iterations == 5
        assert manager.keep_last_n_checkpoints == 3
        assert manager.keep_last_n_replay_buffers == 2
    
    def test_save_checkpoint(self, config, checkpoint_dir, model):
        """Test saving a checkpoint"""
        manager = CheckpointManager(
            checkpoint_dir=str(checkpoint_dir),
            config=config
        )
        
        # Create test data
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        replay_buffer = ReplayBuffer(max_size=100)
        replay_buffer.add([
            GameExample(
                state=np.zeros((15, 15)),
                policy=np.ones(225) / 225,
                value=0.0
            )
        ])
        
        metadata = {
            'iteration': 10,
            'best_model_iteration': 5,
            'elo_ratings': {'iteration_10': 1600},
            'training_losses': [0.5, 0.4, 0.3]
        }
        
        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            iteration=10,
            model=model,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            metadata=metadata
        )
        
        assert checkpoint_path.exists()
        assert checkpoint_path.name == "checkpoint_10.pt"
        
        # Check replay buffer was saved
        replay_buffer_path = checkpoint_dir / "replay_buffer_10.pkl"
        assert replay_buffer_path.exists()
    
    def test_load_checkpoint(self, config, checkpoint_dir, model):
        """Test loading a checkpoint"""
        manager = CheckpointManager(
            checkpoint_dir=str(checkpoint_dir),
            config=config
        )
        
        # Create and save checkpoint
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        replay_buffer = ReplayBuffer(max_size=100)
        replay_buffer.add([
            GameExample(
                state=np.ones((15, 15)),
                policy=np.ones(225) / 225,
                value=1.0
            )
        ])
        
        metadata = {
            'iteration': 20,
            'best_model_iteration': 15,
            'elo_ratings': {'iteration_20': 1700}
        }
        
        checkpoint_path = manager.save_checkpoint(
            iteration=20,
            model=model,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            metadata=metadata
        )
        
        # Load checkpoint
        loaded_data = manager.load_checkpoint(checkpoint_path)
        
        assert loaded_data['iteration'] == 20
        assert loaded_data['best_model_iteration'] == 15
        assert loaded_data['elo_ratings']['iteration_20'] == 1700
        assert 'model_state_dict' in loaded_data
        assert 'optimizer_state_dict' in loaded_data
        
        # Load replay buffer
        loaded_buffer = manager.load_replay_buffer(20)
        assert len(loaded_buffer) == 1
    
    def test_cleanup_old_checkpoints(self, config, checkpoint_dir):
        """Test cleanup of old checkpoints"""
        manager = CheckpointManager(
            checkpoint_dir=str(checkpoint_dir),
            config=config
        )
        
        # Create multiple checkpoints
        for i in range(10):
            checkpoint_path = checkpoint_dir / f"checkpoint_{i}.pt"
            torch.save({'iteration': i}, checkpoint_path)
        
        # Run cleanup
        manager.cleanup_old_checkpoints()
        
        # Should keep only the last 3
        remaining = list(checkpoint_dir.glob("checkpoint_*.pt"))
        assert len(remaining) == 3
        
        # Should be the most recent ones
        iterations = [int(p.stem.split('_')[1]) for p in remaining]
        assert sorted(iterations) == [7, 8, 9]
    
    def test_cleanup_old_replay_buffers(self, config, checkpoint_dir):
        """Test cleanup of old replay buffers"""
        manager = CheckpointManager(
            checkpoint_dir=str(checkpoint_dir),
            config=config
        )
        
        # Create multiple replay buffers
        for i in range(10):
            buffer_path = checkpoint_dir / f"replay_buffer_{i}.pkl"
            with open(buffer_path, 'w') as f:
                f.write("dummy")
        
        # Run cleanup
        manager.cleanup_old_replay_buffers()
        
        # Should keep only the last 2
        remaining = list(checkpoint_dir.glob("replay_buffer_*.pkl"))
        assert len(remaining) == 2
        
        # Should be the most recent ones
        iterations = [int(p.stem.split('_')[2]) for p in remaining]
        assert sorted(iterations) == [8, 9]
    
    def test_get_latest_checkpoint(self, config, checkpoint_dir):
        """Test getting the latest checkpoint"""
        manager = CheckpointManager(
            checkpoint_dir=str(checkpoint_dir),
            config=config
        )
        
        # No checkpoints initially
        assert manager.get_latest_checkpoint() is None
        
        # Create checkpoints
        for i in [5, 10, 15]:
            checkpoint_path = checkpoint_dir / f"checkpoint_{i}.pt"
            torch.save({'iteration': i}, checkpoint_path)
        
        # Should return the latest
        latest = manager.get_latest_checkpoint()
        assert latest is not None
        assert latest.name == "checkpoint_15.pt"
    
    def test_save_best_model(self, config, checkpoint_dir, model):
        """Test saving the best model"""
        manager = CheckpointManager(
            checkpoint_dir=str(checkpoint_dir),
            config=config
        )
        
        # Save best model
        manager.save_best_model(
            model=model,
            iteration=25,
            elo_rating=1800
        )
        
        best_model_path = checkpoint_dir / "best_model.pt"
        assert best_model_path.exists()
        
        # Check metadata
        metadata_path = checkpoint_dir / "best_model_metadata.json"
        assert metadata_path.exists()
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        assert metadata['iteration'] == 25
        assert metadata['elo_rating'] == 1800