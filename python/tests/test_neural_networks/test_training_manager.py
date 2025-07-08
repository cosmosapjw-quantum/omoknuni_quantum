"""Tests for the training manager module"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from mcts.neural_networks.training_manager import (
    TrainingManager, TrainingConfig, TrainingMetrics
)
from mcts.neural_networks.replay_buffer import ReplayBuffer, GameExample
from mcts.utils.config_system import create_default_config


class TestTrainingConfig:
    """Test the TrainingConfig dataclass"""
    
    def test_default_config(self):
        """Test default training configuration"""
        config = TrainingConfig()
        
        assert config.batch_size == 256
        assert config.num_epochs == 10
        assert config.learning_rate == 0.001
        assert config.weight_decay == 1e-4
        assert config.gradient_clip_norm == 1.0
        assert config.lr_scheduler_type == "cosine"
        assert config.warmup_steps == 100
        assert config.policy_loss_weight == 1.0
        assert config.value_loss_weight == 1.0
    
    def test_custom_config(self):
        """Test custom training configuration"""
        config = TrainingConfig(
            batch_size=512,
            num_epochs=20,
            learning_rate=0.0001
        )
        
        assert config.batch_size == 512
        assert config.num_epochs == 20
        assert config.learning_rate == 0.0001


class TestTrainingMetrics:
    """Test the TrainingMetrics class"""
    
    def test_metrics_initialization(self):
        """Test metrics initialization"""
        metrics = TrainingMetrics()
        
        assert metrics.total_loss == []
        assert metrics.policy_loss == []
        assert metrics.value_loss == []
        assert metrics.learning_rate == []
        assert metrics.epoch_times == []
    
    def test_update_metrics(self):
        """Test updating metrics"""
        metrics = TrainingMetrics()
        
        metrics.update(
            total_loss=0.5,
            policy_loss=0.3,
            value_loss=0.2,
            learning_rate=0.001
        )
        
        assert len(metrics.total_loss) == 1
        assert metrics.total_loss[0] == 0.5
        assert metrics.policy_loss[0] == 0.3
        assert metrics.value_loss[0] == 0.2
        assert metrics.learning_rate[0] == 0.001
    
    def test_get_latest_metrics(self):
        """Test getting latest metrics"""
        metrics = TrainingMetrics()
        
        # No metrics yet
        latest = metrics.get_latest()
        assert latest['total_loss'] is None
        
        # Add some metrics
        metrics.update(total_loss=0.5, policy_loss=0.3, value_loss=0.2)
        metrics.update(total_loss=0.4, policy_loss=0.25, value_loss=0.15)
        
        latest = metrics.get_latest()
        assert latest['total_loss'] == 0.4
        assert latest['policy_loss'] == 0.25
        assert latest['value_loss'] == 0.15
    
    def test_get_averages(self):
        """Test getting average metrics"""
        metrics = TrainingMetrics()
        
        # Add some metrics
        for i in range(5):
            metrics.update(
                total_loss=0.5 - i*0.05,
                policy_loss=0.3 - i*0.03,
                value_loss=0.2 - i*0.02
            )
        
        averages = metrics.get_averages(last_n=3)
        assert abs(averages['total_loss'] - 0.35) < 0.001  # Average of last 3
        assert abs(averages['policy_loss'] - 0.21) < 0.001
        assert abs(averages['value_loss'] - 0.14) < 0.001


class TestTrainingManager:
    """Test the TrainingManager class"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        config = create_default_config('gomoku')
        config.training.batch_size = 64
        config.training.num_epochs = 2
        return config
    
    @pytest.fixture
    def model(self):
        """Create mock model"""
        model = Mock(spec=nn.Module)
        model.parameters.return_value = [torch.randn(10, 10)]
        model.train.return_value = model
        model.eval.return_value = model
        
        # Mock forward pass
        def mock_forward(x):
            batch_size = x.shape[0]
            policy = torch.randn(batch_size, 225)
            value = torch.randn(batch_size, 1)
            return policy, value
        
        model.forward.side_effect = mock_forward
        return model
    
    @pytest.fixture
    def replay_buffer(self):
        """Create test replay buffer"""
        buffer = ReplayBuffer(max_size=1000)
        
        # Add some test examples
        for i in range(100):
            buffer.add([GameExample(
                state=np.random.randn(2, 15, 15),
                policy=np.random.dirichlet(np.ones(225)),
                value=np.random.choice([-1, 0, 1])
            )])
        
        return buffer
    
    def test_manager_initialization(self, config, model):
        """Test training manager initialization"""
        manager = TrainingManager(
            config=config,
            model=model
        )
        
        assert manager.config == config
        assert manager.model == model
        assert manager.training_config.batch_size == 64
        assert manager.training_config.num_epochs == 2
        assert manager.optimizer is not None
        assert manager.scheduler is not None
    
    def test_compute_loss(self, config, model):
        """Test loss computation"""
        manager = TrainingManager(
            config=config,
            model=model
        )
        
        # Create test batch
        batch_states = torch.randn(16, 2, 15, 15)
        batch_policies = torch.randn(16, 225)
        batch_policies = torch.softmax(batch_policies, dim=1)  # Make valid probabilities
        batch_values = torch.randn(16, 1)
        
        # Compute loss
        total_loss, policy_loss, value_loss = manager.compute_loss(
            batch_states, batch_policies, batch_values
        )
        
        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(policy_loss, torch.Tensor)
        assert isinstance(value_loss, torch.Tensor)
        assert total_loss.requires_grad
    
    def test_train_epoch(self, config, model, replay_buffer):
        """Test training for one epoch"""
        manager = TrainingManager(
            config=config,
            model=model
        )
        
        # Train one epoch
        metrics = manager.train_epoch(replay_buffer)
        
        assert 'avg_total_loss' in metrics
        assert 'avg_policy_loss' in metrics
        assert 'avg_value_loss' in metrics
        assert metrics['num_batches'] > 0
    
    def test_train_full(self, config, model, replay_buffer):
        """Test full training run"""
        manager = TrainingManager(
            config=config,
            model=model
        )
        
        # Train
        results = manager.train(replay_buffer)
        
        assert 'num_epochs' in results
        assert results['num_epochs'] == 2
        assert 'final_loss' in results
        assert 'training_time' in results
        assert len(manager.metrics.total_loss) > 0
    
    def test_learning_rate_scheduling(self, config, model):
        """Test learning rate scheduling"""
        config.training.lr_scheduler_type = "cosine"
        
        manager = TrainingManager(
            config=config,
            model=model
        )
        
        initial_lr = manager.optimizer.param_groups[0]['lr']
        
        # Step scheduler several times
        for _ in range(10):
            manager.scheduler.step()
        
        current_lr = manager.optimizer.param_groups[0]['lr']
        assert current_lr < initial_lr  # LR should decrease
    
    def test_gradient_clipping(self, config, model, replay_buffer):
        """Test gradient clipping"""
        config.training.gradient_clip_norm = 1.0
        
        manager = TrainingManager(
            config=config,
            model=model
        )
        
        # Create a batch that would produce large gradients
        batch_states = torch.randn(16, 2, 15, 15) * 100  # Large inputs
        batch_policies = torch.ones(16, 225) / 225
        batch_values = torch.ones(16, 1)
        
        # Forward and backward pass
        total_loss, _, _ = manager.compute_loss(
            batch_states, batch_policies, batch_values
        )
        total_loss.backward()
        
        # Check gradients before clipping
        total_norm_before = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm_before += p.grad.data.norm(2).item() ** 2
        total_norm_before = total_norm_before ** 0.5
        
        # Apply gradient clipping
        manager._clip_gradients()
        
        # Check gradients after clipping
        total_norm_after = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm_after += p.grad.data.norm(2).item() ** 2
        total_norm_after = total_norm_after ** 0.5
        
        assert total_norm_after <= 1.0 + 1e-6  # Allow small numerical error
    
    def test_mixed_precision_training(self, config, model, replay_buffer):
        """Test mixed precision training"""
        config.device.use_mixed_precision = True
        
        with patch('torch.cuda.is_available', return_value=True):
            manager = TrainingManager(
                config=config,
                model=model
            )
            
            # Should have scaler for mixed precision
            assert hasattr(manager, 'scaler')
            assert manager.use_mixed_precision == True