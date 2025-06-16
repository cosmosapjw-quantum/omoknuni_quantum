"""Tests for neural network model"""

import pytest
import torch
import numpy as np
from unittest.mock import patch

from mcts.neural_networks.nn_model import (
    ModelConfig, ResidualBlock, PolicyHead, ValueHead,
    AlphaZeroNetwork, EnsembleNetwork, create_model
)


class TestModelConfig:
    """Test ModelConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = ModelConfig()
        assert config.input_channels == 20
        assert config.input_height == 8
        assert config.input_width == 8
        assert config.num_actions == 4096
        assert config.num_res_blocks == 19
        assert config.num_filters == 256
        assert config.value_head_hidden_size == 256
        assert config.use_batch_norm is True
        assert config.activation == 'relu'
        assert config.dropout_rate == 0.0
        
    def test_custom_config(self):
        """Test custom configuration"""
        config = ModelConfig(
            input_channels=4,
            input_height=15,
            input_width=15,
            num_actions=225,
            num_res_blocks=10,
            num_filters=128,
            dropout_rate=0.1
        )
        assert config.input_channels == 4
        assert config.input_height == 15
        assert config.input_width == 15
        assert config.num_actions == 225
        assert config.num_res_blocks == 10
        assert config.num_filters == 128
        assert config.dropout_rate == 0.1


class TestResidualBlock:
    """Test ResidualBlock module"""
    
    def test_residual_block_forward(self):
        """Test forward pass through residual block"""
        block = ResidualBlock(num_filters=64)
        
        # Create dummy input
        batch_size = 4
        x = torch.randn(batch_size, 64, 8, 8)
        
        output = block(x)
        
        # Check output shape
        assert output.shape == x.shape
        
        # Check that it's not just identity (some transformation happened)
        assert not torch.allclose(output, x)
        
    def test_residual_block_no_batch_norm(self):
        """Test residual block without batch norm"""
        block = ResidualBlock(num_filters=32, use_batch_norm=False)
        
        x = torch.randn(2, 32, 4, 4)
        output = block(x)
        
        assert output.shape == x.shape
        
    def test_residual_block_with_dropout(self):
        """Test residual block with dropout"""
        block = ResidualBlock(num_filters=32, dropout_rate=0.5)
        
        # Set to training mode for dropout
        block.train()
        
        x = torch.randn(2, 32, 4, 4)
        output1 = block(x)
        output2 = block(x)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output1, output2)
        
        # Set to eval mode
        block.eval()
        output3 = block(x)
        output4 = block(x)
        
        # Outputs should be same in eval mode
        assert torch.allclose(output3, output4)
        
    def test_residual_block_elu_activation(self):
        """Test residual block with ELU activation"""
        block = ResidualBlock(num_filters=32, activation='elu')
        
        x = torch.randn(2, 32, 4, 4)
        output = block(x)
        
        assert output.shape == x.shape


class TestPolicyHead:
    """Test PolicyHead module"""
    
    def test_policy_head_forward(self):
        """Test forward pass through policy head"""
        num_filters = 128
        board_size = 64  # 8x8
        num_actions = 100
        
        head = PolicyHead(num_filters, board_size, num_actions)
        
        # Create dummy input
        batch_size = 4
        x = torch.randn(batch_size, num_filters, 8, 8)
        
        policy_logits = head(x)
        
        # Check output shape
        assert policy_logits.shape == (batch_size, num_actions)
        
    def test_policy_head_no_batch_norm(self):
        """Test policy head without batch norm"""
        head = PolicyHead(64, 36, 50, use_batch_norm=False)
        
        x = torch.randn(2, 64, 6, 6)
        output = head(x)
        
        assert output.shape == (2, 50)


class TestValueHead:
    """Test ValueHead module"""
    
    def test_value_head_forward(self):
        """Test forward pass through value head"""
        num_filters = 128
        board_size = 64  # 8x8
        hidden_size = 256
        
        head = ValueHead(num_filters, board_size, hidden_size)
        
        # Create dummy input
        batch_size = 4
        x = torch.randn(batch_size, num_filters, 8, 8)
        
        value = head(x)
        
        # Check output shape
        assert value.shape == (batch_size, 1)
        
        # Check value range (tanh activation)
        assert torch.all(value >= -1.0)
        assert torch.all(value <= 1.0)
        
    def test_value_head_different_sizes(self):
        """Test value head with different configurations"""
        head = ValueHead(32, 225, 128)  # 15x15 board
        
        x = torch.randn(3, 32, 15, 15)
        value = head(x)
        
        assert value.shape == (3, 1)
        assert torch.all(torch.abs(value) <= 1.0)


class TestAlphaZeroNetwork:
    """Test AlphaZeroNetwork"""
    
    @pytest.fixture
    def small_config(self):
        """Create small config for testing"""
        return ModelConfig(
            input_channels=4,
            input_height=4,
            input_width=4,
            num_actions=16,
            num_res_blocks=2,
            num_filters=32,
            value_head_hidden_size=64
        )
        
    def test_network_forward(self, small_config):
        """Test forward pass through network"""
        model = AlphaZeroNetwork(small_config)
        
        batch_size = 8
        x = torch.randn(batch_size, 4, 4, 4)
        
        policy_logits, value = model(x)
        
        # Check output shapes
        assert policy_logits.shape == (batch_size, 16)
        assert value.shape == (batch_size, 1)
        
        # Check value range
        assert torch.all(torch.abs(value) <= 1.0)
        
    def test_network_predict_single(self, small_config):
        """Test single position prediction"""
        model = AlphaZeroNetwork(small_config)
        model.eval()
        
        # Single position
        x = np.random.randn(4, 4, 4).astype(np.float32)
        
        policy_probs, value = model.predict(x)
        
        # Check outputs
        assert isinstance(policy_probs, np.ndarray)
        assert isinstance(value, float)
        assert policy_probs.shape == (16,)
        assert np.allclose(policy_probs.sum(), 1.0, rtol=1e-5)
        assert -1.0 <= value <= 1.0
        
    def test_network_predict_tensor(self, small_config):
        """Test prediction with tensor input"""
        model = AlphaZeroNetwork(small_config)
        model.eval()
        
        x = torch.randn(4, 4, 4)
        policy_probs, value = model.predict(x)
        
        assert policy_probs.shape == (16,)
        assert isinstance(value, float)
        
    def test_network_no_batch_norm(self):
        """Test network without batch normalization"""
        config = ModelConfig(
            input_channels=3,
            input_height=3,
            input_width=3,
            num_actions=9,
            num_res_blocks=1,
            num_filters=16,
            use_batch_norm=False
        )
        
        model = AlphaZeroNetwork(config)
        x = torch.randn(2, 3, 3, 3)
        
        policy, value = model(x)
        assert policy.shape == (2, 9)
        assert value.shape == (2, 1)
        
    def test_network_with_dropout(self):
        """Test network with dropout"""
        config = ModelConfig(
            input_channels=3,
            input_height=3,
            input_width=3,
            num_actions=9,
            num_res_blocks=1,
            num_filters=16,
            dropout_rate=0.5
        )
        
        model = AlphaZeroNetwork(config)
        model.train()
        
        x = torch.randn(2, 3, 3, 3)
        
        # Multiple forward passes should give different results in train mode
        policy1, value1 = model(x)
        policy2, value2 = model(x)
        
        assert not torch.allclose(policy1, policy2)
        
    def test_network_param_count(self, small_config):
        """Test parameter counting"""
        model = AlphaZeroNetwork(small_config)
        
        num_params = model.get_num_params()
        num_trainable = model.get_num_trainable_params()
        
        assert num_params > 0
        assert num_trainable == num_params  # All params trainable by default
        
        # Freeze some parameters
        for param in model.initial_conv.parameters():
            param.requires_grad = False
            
        num_trainable_after = model.get_num_trainable_params()
        assert num_trainable_after < num_trainable


class TestCreateModel:
    """Test model creation helper"""
    
    def test_create_chess_model(self):
        """Test creating chess model"""
        model = create_model('chess')
        
        assert isinstance(model, AlphaZeroNetwork)
        assert model.config.input_channels == 17
        assert model.config.input_height == 8
        assert model.config.input_width == 8
        assert model.config.num_actions == 4096
        
    def test_create_go_model(self):
        """Test creating go model"""
        model = create_model('go')
        
        assert model.config.input_channels == 17
        assert model.config.input_height == 19
        assert model.config.input_width == 19
        assert model.config.num_actions == 362
        
    def test_create_gomoku_model(self):
        """Test creating gomoku model"""
        model = create_model('gomoku')
        
        assert model.config.input_channels == 4
        assert model.config.input_height == 15
        assert model.config.input_width == 15
        assert model.config.num_actions == 225
        
    def test_create_model_with_overrides(self):
        """Test creating model with parameter overrides"""
        model = create_model(
            'chess',
            num_res_blocks=10,
            num_filters=128,
            dropout_rate=0.1
        )
        
        assert model.config.num_res_blocks == 10
        assert model.config.num_filters == 128
        assert model.config.dropout_rate == 0.1
        
    def test_create_unknown_game_model(self):
        """Test creating model for unknown game type"""
        model = create_model('unknown_game', input_channels=5, num_actions=50)
        
        # Should use default config with overrides
        assert model.config.input_channels == 5
        assert model.config.num_actions == 50


class TestEnsembleNetwork:
    """Test EnsembleNetwork"""
    
    @pytest.fixture
    def ensemble_configs(self):
        """Create configs for ensemble"""
        return [
            ModelConfig(
                input_channels=3,
                input_height=3,
                input_width=3,
                num_actions=9,
                num_res_blocks=1,
                num_filters=16
            )
            for _ in range(3)
        ]
        
    def test_ensemble_forward(self, ensemble_configs):
        """Test forward pass through ensemble"""
        ensemble = EnsembleNetwork(ensemble_configs)
        
        batch_size = 4
        x = torch.randn(batch_size, 3, 3, 3)
        
        mean_policy, mean_value, stats = ensemble(x)
        
        # Check output shapes
        assert mean_policy.shape == (batch_size, 9)
        assert mean_value.shape == (batch_size, 1)
        
        # Check statistics
        assert 'policy_std' in stats
        assert 'value_std' in stats
        assert 'policy_entropy' in stats
        
        assert stats['policy_std'].shape == (batch_size, 9)
        assert stats['value_std'].shape == (batch_size, 1)
        assert stats['policy_entropy'].shape == (batch_size,)
        
    def test_ensemble_with_weights(self, ensemble_configs):
        """Test ensemble with custom weights"""
        weights = [0.5, 0.3, 0.2]
        ensemble = EnsembleNetwork(ensemble_configs, weights)
        
        x = torch.randn(2, 3, 3, 3)
        mean_policy, mean_value, _ = ensemble(x)
        
        assert mean_policy.shape == (2, 9)
        assert mean_value.shape == (2, 1)
        
    def test_ensemble_single_model(self):
        """Test ensemble with single model"""
        configs = [ModelConfig(
            input_channels=2,
            input_height=2,
            input_width=2,
            num_actions=4,
            num_res_blocks=1,
            num_filters=8
        )]
        
        ensemble = EnsembleNetwork(configs)
        
        x = torch.randn(1, 2, 2, 2)
        mean_policy, mean_value, stats = ensemble(x)
        
        # With single model, std should be zero
        assert torch.allclose(stats['policy_std'], torch.zeros_like(stats['policy_std']))
        assert torch.allclose(stats['value_std'], torch.zeros_like(stats['value_std']))


class TestGPUCompatibility:
    """Test GPU compatibility (skipped if no GPU)"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    def test_model_on_gpu(self):
        """Test model on GPU"""
        config = ModelConfig(
            input_channels=3,
            input_height=4,
            input_width=4,
            num_actions=16,
            num_res_blocks=2,
            num_filters=32
        )
        
        model = AlphaZeroNetwork(config)
        model = model.cuda()
        
        x = torch.randn(2, 3, 4, 4).cuda()
        policy, value = model(x)
        
        assert policy.is_cuda
        assert value.is_cuda
        assert policy.shape == (2, 16)
        assert value.shape == (2, 1)
        
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    def test_ensemble_on_gpu(self):
        """Test ensemble on GPU"""
        configs = [ModelConfig(
            input_channels=2,
            input_height=2,
            input_width=2,
            num_actions=4,
            num_res_blocks=1,
            num_filters=8
        ) for _ in range(2)]
        
        ensemble = EnsembleNetwork(configs)
        ensemble = ensemble.cuda()
        
        x = torch.randn(2, 2, 2, 2).cuda()
        mean_policy, mean_value, stats = ensemble(x)
        
        assert mean_policy.is_cuda
        assert mean_value.is_cuda
        assert all(s.is_cuda for s in stats.values())