"""Tests for ResNet model architecture

Tests cover:
- Model initialization and configuration
- Forward pass
- Residual blocks
- Policy and value heads
- Weight initialization
- Game-specific model creation
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from mcts.neural_networks.resnet_model import (
    ResNetModel, ResNetConfig, ResidualBlock, PolicyHead, ValueHead,
    create_resnet_for_game
)
from mcts.neural_networks.nn_framework import ModelMetadata


@pytest.fixture
def resnet_config():
    """Create ResNet configuration"""
    return ResNetConfig(
        num_blocks=3,  # Small for testing
        num_filters=64,  # Small for testing
        input_channels=18,
        fc_value_hidden=128,
        fc_policy_hidden=128
    )


@pytest.fixture
def gomoku_model(resnet_config):
    """Create Gomoku ResNet model"""
    return ResNetModel(
        config=resnet_config,
        board_size=15,
        num_actions=225,
        game_type='gomoku'
    )


@pytest.fixture
def sample_input():
    """Create sample input tensor"""
    batch_size = 4
    channels = 18
    board_size = 15
    return torch.randn(batch_size, channels, board_size, board_size)


class TestResNetConfig:
    """Test ResNetConfig configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ResNetConfig()
        assert config.num_blocks == 19
        assert config.num_filters == 256
        assert config.input_channels == 18
        assert config.fc_value_hidden == 256
        assert config.fc_policy_hidden == 256
        
    def test_custom_config(self):
        """Test custom configuration"""
        config = ResNetConfig(
            num_blocks=10,
            num_filters=128,
            input_channels=18,
            fc_value_hidden=512
        )
        assert config.num_blocks == 10
        assert config.num_filters == 128
        assert config.input_channels == 18
        assert config.fc_value_hidden == 512


class TestResidualBlock:
    """Test ResidualBlock functionality"""
    
    def test_initialization(self):
        """Test residual block initialization"""
        block = ResidualBlock(num_filters=64)
        
        # Check layers
        assert isinstance(block.conv1, nn.Conv2d)
        assert isinstance(block.bn1, nn.BatchNorm2d)
        assert isinstance(block.conv2, nn.Conv2d)
        assert isinstance(block.bn2, nn.BatchNorm2d)
        
        # Check conv parameters
        assert block.conv1.in_channels == 64
        assert block.conv1.out_channels == 64
        assert block.conv1.kernel_size == (3, 3)
        assert block.conv1.padding == (1, 1)
        assert block.conv1.bias is None
        
    def test_forward_pass(self):
        """Test residual block forward pass"""
        block = ResidualBlock(num_filters=64)
        
        # Create input
        x = torch.randn(2, 64, 8, 8)
        
        # Forward pass
        output = block(x)
        
        # Check output shape matches input
        assert output.shape == x.shape
        
        # Check residual connection (output should be different from conv only)
        conv_only = block.conv2(block.bn1(torch.relu(block.conv1(x))))
        assert not torch.allclose(output, conv_only)
        
    def test_gradient_flow(self):
        """Test gradient flows through residual connection"""
        block = ResidualBlock(num_filters=32)
        
        x = torch.randn(1, 32, 4, 4, requires_grad=True)
        output = block(x)
        
        # Compute gradient
        loss = output.sum()
        loss.backward()
        
        # Gradient should flow through residual connection
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestPolicyHead:
    """Test PolicyHead functionality"""
    
    def test_initialization(self):
        """Test policy head initialization"""
        head = PolicyHead(num_filters=64, board_size=15, num_actions=225)
        
        assert head.board_size == 15
        assert head.num_actions == 225
        
        # Check layers
        assert head.conv.in_channels == 64
        assert head.conv.out_channels == 2
        assert head.fc1.in_features == 2 * 15 * 15
        assert head.fc1.out_features == 256  # Default
        assert head.fc2.in_features == 256
        assert head.fc2.out_features == 225
        
    def test_forward_pass(self):
        """Test policy head forward pass"""
        head = PolicyHead(num_filters=64, board_size=8, num_actions=64)
        
        # Input from residual blocks
        x = torch.randn(4, 64, 8, 8)
        
        # Forward pass
        output = head(x)
        
        # Check output shape
        assert output.shape == (4, 64)
        
        # Check log softmax applied (sum of exp should be ~1)
        probs = torch.exp(output)
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)
        
    def test_weight_initialization(self):
        """Test policy head weight initialization"""
        head = PolicyHead(num_filters=64, board_size=8, num_actions=64)
        
        # Check fc2 weights are not too large
        assert head.fc2.weight.abs().max() < 1.0
        
        # Check bias has small random values
        if head.fc2.bias is not None:
            assert head.fc2.bias.abs().max() < 0.1
            assert head.fc2.bias.std() > 0  # Not constant
            
    def test_custom_hidden_size(self):
        """Test policy head with custom hidden size"""
        head = PolicyHead(
            num_filters=64,
            board_size=8,
            num_actions=64,
            fc_hidden=512
        )
        
        assert head.fc1.out_features == 512
        assert head.fc2.in_features == 512


class TestValueHead:
    """Test ValueHead functionality"""
    
    def test_initialization(self):
        """Test value head initialization"""
        head = ValueHead(num_filters=64, board_size=15)
        
        assert head.board_size == 15
        
        # Check layers
        assert head.conv.in_channels == 64
        assert head.conv.out_channels == 1
        assert head.fc1.in_features == 15 * 15
        assert head.fc1.out_features == 256  # Default
        assert head.fc2.in_features == 256
        assert head.fc2.out_features == 1
        
    def test_forward_pass(self):
        """Test value head forward pass"""
        head = ValueHead(num_filters=64, board_size=8)
        
        # Input from residual blocks
        x = torch.randn(4, 64, 8, 8)
        
        # Forward pass
        output = head(x)
        
        # Check output shape
        assert output.shape == (4, 1)
        
        # Check tanh applied (values in [-1, 1])
        assert torch.all(output >= -1)
        assert torch.all(output <= 1)
        
    def test_weight_initialization(self):
        """Test value head weight initialization"""
        head = ValueHead(num_filters=64, board_size=8)
        
        # Check fc2 weights are small
        assert head.fc2.weight.abs().mean() < 0.2
        
        # Check bias is zero
        if head.fc2.bias is not None:
            assert head.fc2.bias.item() == 0
            
    def test_initial_output_near_zero(self):
        """Test value head produces values near zero initially"""
        head = ValueHead(num_filters=64, board_size=8)
        head.eval()  # Eval mode for deterministic output
        
        # Random input
        x = torch.randn(100, 64, 8, 8)
        
        with torch.no_grad():
            values = head(x)
            
        # Most values should be near zero
        assert values.abs().mean() < 0.3


class TestResNetModel:
    """Test complete ResNetModel"""
    
    def test_initialization(self, gomoku_model, resnet_config):
        """Test model initialization"""
        assert gomoku_model.config == resnet_config
        assert gomoku_model.board_size == 15
        assert gomoku_model.num_actions == 225
        
        # Check metadata
        assert gomoku_model.metadata.game_type == 'gomoku'
        assert gomoku_model.metadata.board_size == 15
        assert gomoku_model.metadata.num_actions == 225
        assert gomoku_model.metadata.input_channels == 18
        assert gomoku_model.metadata.num_blocks == 3
        
    def test_layer_structure(self, gomoku_model):
        """Test model layer structure"""
        # Input convolution
        assert hasattr(gomoku_model, 'conv_input')
        assert hasattr(gomoku_model, 'bn_input')
        
        # Residual blocks
        assert len(gomoku_model.residual_blocks) == 3
        for block in gomoku_model.residual_blocks:
            assert isinstance(block, ResidualBlock)
            
        # Heads
        assert isinstance(gomoku_model.policy_head, PolicyHead)
        assert isinstance(gomoku_model.value_head, ValueHead)
        
    def test_forward_pass(self, gomoku_model, sample_input):
        """Test model forward pass"""
        policy, value = gomoku_model(sample_input)
        
        # Check shapes
        assert policy.shape == (4, 225)
        assert value.shape == (4, 1)
        
        # Check policy is log probabilities
        assert torch.all(policy <= 0)  # Log probs are negative
        
        # Check value is in [-1, 1]
        assert torch.all(value >= -1)
        assert torch.all(value <= 1)
        
    def test_gradient_computation(self, gomoku_model, sample_input):
        """Test gradient computation through model"""
        # Forward pass
        policy, value = gomoku_model(sample_input)
        
        # Compute loss
        policy_target = torch.randint(0, 225, (4,))
        value_target = torch.rand(4, 1) * 2 - 1  # Random values in [-1, 1]
        
        policy_loss = torch.nn.functional.nll_loss(policy, policy_target)
        value_loss = torch.nn.functional.mse_loss(value, value_target)
        total_loss = policy_loss + value_loss
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients exist
        for name, param in gomoku_model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"
            
    def test_parameter_count(self, gomoku_model):
        """Test parameter count is reasonable"""
        total_params = sum(p.numel() for p in gomoku_model.parameters())
        
        # With 3 blocks and 64 filters, should be much smaller than full model
        assert total_params < 1_000_000  # Less than 1M parameters
        
        # But should still have substantial parameters
        assert total_params > 10_000
        
    def test_eval_mode(self, gomoku_model, sample_input):
        """Test model in evaluation mode"""
        gomoku_model.eval()
        
        # Multiple forward passes should give same result
        with torch.no_grad():
            policy1, value1 = gomoku_model(sample_input)
            policy2, value2 = gomoku_model(sample_input)
            
        assert torch.allclose(policy1, policy2)
        assert torch.allclose(value1, value2)
        
    def test_training_mode(self, gomoku_model, sample_input):
        """Test model in training mode"""
        gomoku_model.train()
        
        # Forward passes may differ due to batch norm
        policy1, value1 = gomoku_model(sample_input)
        policy2, value2 = gomoku_model(sample_input)
        
        # Outputs should be different due to batch norm in training mode
        # (unless by chance they're identical)
        # Just check they have correct shapes
        assert policy1.shape == policy2.shape
        assert value1.shape == value2.shape


class TestModelCreation:
    """Test model creation utilities"""
    
    def test_create_chess_model(self):
        """Test creating Chess model"""
        model = create_resnet_for_game('chess')
        
        assert model.board_size == 8
        assert model.num_actions == 4096
        assert model.metadata.game_type == 'chess'
        
        # Default Chess uses 19 blocks
        assert model.config.num_blocks == 19
        assert model.config.num_filters == 256
        
    def test_create_go_model(self):
        """Test creating Go model"""
        model = create_resnet_for_game('go')
        
        assert model.board_size == 19
        assert model.num_actions == 362
        assert model.metadata.game_type == 'go'
        
        # Go uses more blocks by default
        assert model.config.num_blocks == 39
        
    def test_create_gomoku_model(self):
        """Test creating Gomoku model"""
        model = create_resnet_for_game('gomoku')
        
        assert model.board_size == 15
        assert model.num_actions == 225
        assert model.metadata.game_type == 'gomoku'
        
        # Gomoku uses fewer blocks and filters
        assert model.config.num_blocks == 10
        assert model.config.num_filters == 128
        
    def test_create_with_custom_params(self):
        """Test creating model with custom parameters"""
        model = create_resnet_for_game(
            'gomoku',
            input_channels=18,
            num_blocks=5,
            num_filters=64
        )
        
        assert model.config.input_channels == 18
        assert model.config.num_blocks == 5
        assert model.config.num_filters == 64
        
    def test_invalid_game_type(self):
        """Test error on invalid game type"""
        with pytest.raises(ValueError, match="Unknown game type"):
            create_resnet_for_game('invalid_game')


class TestWeightInitialization:
    """Test weight initialization strategies"""
    
    def test_conv_initialization(self, gomoku_model):
        """Test convolutional layer initialization"""
        # Check conv layers use Kaiming initialization
        conv_layer = gomoku_model.conv_input
        
        # Weights should have reasonable variance
        std = conv_layer.weight.std().item()
        assert 0.01 < std < 0.5
        
        # No bias in conv layers
        assert conv_layer.bias is None
        
    def test_batchnorm_initialization(self, gomoku_model):
        """Test batch norm initialization"""
        bn_layer = gomoku_model.bn_input
        
        # Weight should be near 1 with small variance
        assert 0.9 < bn_layer.weight.mean().item() < 1.1
        assert bn_layer.weight.std().item() < 0.1
        
        # Bias should be zero
        assert torch.all(bn_layer.bias == 0)
        
    def test_spatial_symmetry_breaking(self, gomoku_model):
        """Test spatial symmetry breaking in input conv"""
        # Input conv should have slight noise to break symmetry
        conv_weight = gomoku_model.conv_input.weight
        
        # Check weights are not perfectly symmetric
        # Compare opposite corners of kernels
        corner_diff = (conv_weight[:, :, 0, 0] - conv_weight[:, :, -1, -1]).abs()
        assert corner_diff.max() > 1e-6  # Some asymmetry exists


class TestMemoryUsage:
    """Test memory efficiency"""
    
    def test_batch_processing(self, gomoku_model):
        """Test model can handle various batch sizes"""
        gomoku_model.eval()
        
        batch_sizes = [1, 8, 16, 32]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 18, 15, 15)
            
            with torch.no_grad():
                policy, value = gomoku_model(x)
                
            assert policy.shape == (batch_size, 225)
            assert value.shape == (batch_size, 1)
            
    def test_large_model_creation(self):
        """Test creating large model (production size)"""
        config = ResNetConfig(
            num_blocks=19,
            num_filters=256,
            input_channels=18
        )
        
        model = ResNetModel(
            config=config,
            board_size=19,
            num_actions=362,
            game_type='go'
        )
        
        # Should create successfully
        assert len(model.residual_blocks) == 19
        
        # Test forward pass with small batch
        x = torch.randn(2, 18, 19, 19)
        policy, value = model(x)
        
        assert policy.shape == (2, 362)
        assert value.shape == (2, 1)