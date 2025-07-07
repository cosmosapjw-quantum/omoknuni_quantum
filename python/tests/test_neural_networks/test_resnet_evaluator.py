"""Tests for ResNet evaluator

Tests cover:
- Evaluator initialization
- Model loading and checkpoint handling
- Forward pass evaluation
- GPU-native batch processing
- Statistics tracking
- Configuration loading
- Game-specific evaluator creation
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import yaml

from mcts.neural_networks.resnet_evaluator import (
    ResNetEvaluator, create_evaluator_for_game, load_config_for_model,
    create_chess_evaluator, create_go_evaluator, create_gomoku_evaluator
)
from mcts.neural_networks.resnet_model import create_resnet_for_game
from mcts.core.evaluator import EvaluatorConfig
from mcts.utils.config_system import AlphaZeroConfig, NeuralNetworkConfig


@pytest.fixture
def evaluator_config():
    """Create evaluator configuration"""
    return EvaluatorConfig(
        batch_size=32,
        device='cpu',
        timeout=1.0,
        enable_caching=False
    )


@pytest.fixture
def network_config():
    """Create neural network configuration"""
    return NeuralNetworkConfig(
        input_channels=18,
        num_res_blocks=3,
        num_filters=64,
        fc_hidden_size=128
    )


@pytest.fixture
def gomoku_model():
    """Create Gomoku ResNet model"""
    return create_resnet_for_game('gomoku', input_channels=18, num_blocks=3, num_filters=64)


@pytest.fixture
def gomoku_evaluator(gomoku_model, evaluator_config):
    """Create Gomoku ResNet evaluator"""
    return ResNetEvaluator(
        model=gomoku_model,
        config=evaluator_config,
        game_type='gomoku'
    )


@pytest.fixture
def sample_states():
    """Create sample state tensors"""
    batch_size = 4
    channels = 18  # Basic representation
    board_size = 15
    return torch.randn(batch_size, channels, board_size, board_size)


@pytest.fixture
def temp_checkpoint(gomoku_model):
    """Create temporary checkpoint file"""
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        checkpoint = {
            'model_state_dict': gomoku_model.state_dict(),
            'metadata': {
                'game_type': 'gomoku',
                'board_size': 15,
                'num_actions': 225,
                'input_channels': 20
            }
        }
        torch.save(checkpoint, tmp.name)
        yield tmp.name
    Path(tmp.name).unlink()


class TestResNetEvaluatorInitialization:
    """Test ResNetEvaluator initialization"""
    
    def test_init_with_model(self, gomoku_model, evaluator_config):
        """Test initialization with pre-loaded model"""
        evaluator = ResNetEvaluator(
            model=gomoku_model,
            config=evaluator_config,
            game_type='gomoku'
        )
        
        assert evaluator.model == gomoku_model
        assert evaluator.game_type == 'gomoku'
        # Device should be cuda if available, cpu otherwise
        expected_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert evaluator.device == expected_device
        assert evaluator.action_size == 225
        assert evaluator.use_amp == True  # Default
        
    def test_init_without_model(self, evaluator_config):
        """Test initialization creating new model"""
        evaluator = ResNetEvaluator(
            config=evaluator_config,
            game_type='gomoku'
        )
        
        assert evaluator.model is not None
        assert evaluator.game_type == 'gomoku'
        assert evaluator.model.board_size == 15
        assert evaluator.model.num_actions == 225
        
    def test_init_with_checkpoint(self, temp_checkpoint, evaluator_config):
        """Test initialization from checkpoint"""
        with patch('mcts.neural_networks.nn_framework.ModelLoader.load_checkpoint') as mock_load:
            mock_model = Mock()
            mock_model.to = Mock(return_value=mock_model)  # Make to() return self
            mock_metadata = {'game_type': 'gomoku', 'board_size': 15}
            mock_load.return_value = (mock_model, mock_metadata)
            
            evaluator = ResNetEvaluator(
                config=evaluator_config,
                checkpoint_path=temp_checkpoint,
                game_type='gomoku'
            )
            
            assert evaluator.model == mock_model
            mock_load.assert_called_once_with(temp_checkpoint, None)
            
    def test_init_with_network_config(self, evaluator_config, network_config):
        """Test initialization with network configuration"""
        evaluator = ResNetEvaluator(
            config=evaluator_config,
            game_type='gomoku',
            network_config=network_config
        )
        
        assert evaluator.model.config.input_channels == 18
        assert evaluator.model.config.num_blocks == 3
        assert evaluator.model.config.num_filters == 64
        
    def test_init_with_input_channels(self, evaluator_config):
        """Test initialization with explicit input channels"""
        evaluator = ResNetEvaluator(
            config=evaluator_config,
            game_type='gomoku',
            input_channels=18
        )
        
        assert evaluator.model.config.input_channels == 18
        
    def test_device_selection(self):
        """Test device selection logic"""
        # CPU device
        evaluator = ResNetEvaluator(game_type='gomoku', device='cpu')
        assert evaluator.device == torch.device('cpu')
        
        # CUDA device (mocked)
        with patch('torch.cuda.is_available', return_value=True):
            evaluator = ResNetEvaluator(game_type='gomoku', device='cuda')
            assert evaluator.device.type == 'cuda'


class TestConfigLoading:
    """Test configuration loading functionality"""
    
    def test_load_config_from_experiment_dir(self, tmp_path):
        """Test loading config from experiment directory"""
        # Create experiment structure
        exp_dir = tmp_path / "experiments" / "test_exp"
        exp_dir.mkdir(parents=True)
        
        config_path = exp_dir / "config.yaml"
        checkpoint_path = exp_dir / "best_models" / "model.pt"
        checkpoint_path.parent.mkdir(parents=True)
        
        # Create config
        config = AlphaZeroConfig()
        config.network.input_channels = 18
        config.network.num_res_blocks = 5
        config.save(str(config_path))
        
        # Load config
        loaded_config = load_config_for_model(str(checkpoint_path))
        
        assert loaded_config is not None
        assert loaded_config.input_channels == 18
        assert loaded_config.num_res_blocks == 5
        
    def test_load_config_fallback_to_default(self):
        """Test fallback to default config"""
        # Non-existent checkpoint
        config = load_config_for_model("/non/existent/path.pt")
        
        # Should return None or find default config
        # (depends on whether configs/gomoku_classical.yaml exists)
        assert config is None or isinstance(config, NeuralNetworkConfig)
        
    def test_load_config_handles_errors(self, tmp_path):
        """Test config loading handles errors gracefully"""
        # Create invalid config file
        config_path = tmp_path / "config.yaml"
        config_path.write_text("invalid: yaml: content:")
        
        checkpoint_path = tmp_path / "model.pt"
        
        # Should handle error and fall back to default config if available
        config = load_config_for_model(str(checkpoint_path))
        
        # The function will try the invalid config, fail, then fall back to
        # configs/gomoku_classical.yaml if it exists
        if (Path(__file__).parent.parent.parent.parent / "configs" / "gomoku_classical.yaml").exists():
            # Default config should be loaded
            assert config is not None
            assert isinstance(config, NeuralNetworkConfig)
        else:
            # No fallback available
            assert config is None


class TestForwardPass:
    """Test forward pass evaluation"""
    
    def test_forward_model(self, gomoku_evaluator, sample_states):
        """Test _forward_model method"""
        # Ensure sample states are on the same device as the model
        device = next(gomoku_evaluator.model.parameters()).device
        sample_states = sample_states.to(device)
        
        policies, values = gomoku_evaluator._forward_model(sample_states)
        
        # Check shapes
        assert policies.shape == (4, 225)
        assert values.shape == (4,)
        
        # Check policies are probabilities
        assert torch.all(policies >= 0)
        assert torch.all(policies <= 1)
        assert torch.allclose(policies.sum(dim=1), torch.ones(4, device=policies.device), atol=1e-5)
        
        # Check values are in [-1, 1]
        assert torch.all(values >= -1)
        assert torch.all(values <= 1)
        
        # Check eval counter updated
        assert gomoku_evaluator.eval_count == 4
        
    def test_forward_batch_gpu_native(self, gomoku_evaluator, sample_states):
        """Test GPU-native forward_batch method"""
        # Move tensors to the same device as the model
        device = gomoku_evaluator.device
        sample_states = sample_states.to(device)
        
        # Create legal mask
        legal_mask = torch.ones(4, 225, dtype=torch.bool, device=device)
        legal_mask[:, 200:] = False  # Some illegal moves
        
        policies, values = gomoku_evaluator.forward_batch(
            sample_states,
            legal_mask,
            temperature=1.0
        )
        
        # Check shapes
        assert policies.shape == (4, 225)
        assert values.shape == (4,)
        
        # Check legal mask applied
        assert torch.all(policies[:, 200:] == 0)
        
        # Check normalization
        assert torch.allclose(policies.sum(dim=1), torch.ones(4, device=device), atol=1e-5)
        
    def test_temperature_scaling(self, gomoku_evaluator, sample_states):
        """Test temperature scaling in forward_batch"""
        # Low temperature (more deterministic)
        policies_low, _ = gomoku_evaluator.forward_batch(
            sample_states,
            temperature=0.5
        )
        
        # High temperature (more uniform)
        policies_high, _ = gomoku_evaluator.forward_batch(
            sample_states,
            temperature=2.0
        )
        
        # Low temperature should have higher max probabilities
        assert policies_low.max(dim=1)[0].mean() > policies_high.max(dim=1)[0].mean()
        
    def test_evaluate_single(self, gomoku_evaluator):
        """Test single state evaluation (inherited method)"""
        state = np.random.randn(20, 15, 15).astype(np.float32)
        
        policy, value = gomoku_evaluator.evaluate(state)
        
        assert policy.shape == (225,)
        assert isinstance(value, float)
        assert -1 <= value <= 1
        
    def test_evaluate_batch(self, gomoku_evaluator):
        """Test batch evaluation (inherited method)"""
        states = np.random.randn(8, 20, 15, 15).astype(np.float32)
        
        policies, values = gomoku_evaluator.evaluate_batch(states)
        
        assert policies.shape == (8, 225)
        assert values.shape == (8,)
        
    def test_mixed_precision(self, gomoku_model, evaluator_config):
        """Test mixed precision evaluation"""
        evaluator_config.use_mixed_precision = True
        evaluator = ResNetEvaluator(
            model=gomoku_model,
            config=evaluator_config,
            game_type='gomoku'
        )
        
        states = torch.randn(4, 20, 15, 15)
        
        # Should work with AMP enabled
        policies, values = evaluator._forward_model(states)
        assert policies.dtype == torch.float32  # Output should be float32


class TestStatistics:
    """Test statistics tracking"""
    
    def test_get_stats(self, gomoku_evaluator):
        """Test getting evaluator statistics"""
        # Perform some evaluations
        states = torch.randn(4, 20, 15, 15)
        gomoku_evaluator._forward_model(states)
        
        stats = gomoku_evaluator.get_stats()
        
        # Check base stats
        assert 'evaluations' in stats
        assert 'avg_time' in stats
        
        # Check ResNet-specific stats
        assert stats['model_params'] > 0
        assert stats['use_amp'] == True
        assert stats['cache_hits'] == 0
        assert stats['cache_misses'] == 0
        assert stats['cache_hit_rate'] == 0.0
        assert stats['eval_count_legacy'] == 4
        
    def test_reset_statistics(self, gomoku_evaluator):
        """Test resetting statistics"""
        # Add some stats
        gomoku_evaluator.eval_count = 100
        gomoku_evaluator._cache_hits = 50
        gomoku_evaluator._cache_misses = 50
        gomoku_evaluator._batch_cache['test'] = 'data'
        
        # Reset
        gomoku_evaluator.reset_statistics()
        
        assert gomoku_evaluator.eval_count == 0
        assert gomoku_evaluator._cache_hits == 0
        assert gomoku_evaluator._cache_misses == 0
        assert len(gomoku_evaluator._batch_cache) == 0


class TestCheckpointing:
    """Test checkpoint save/load functionality"""
    
    def test_save_checkpoint(self, gomoku_evaluator, tmp_path):
        """Test saving checkpoint"""
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        
        gomoku_evaluator.save_checkpoint(
            str(checkpoint_path),
            additional_data={'epoch': 10, 'loss': 0.5}
        )
        
        assert checkpoint_path.exists()
        
        # Load and verify
        checkpoint = torch.load(checkpoint_path)
        assert 'model_state_dict' in checkpoint
        assert 'evaluator_config' in checkpoint
        assert checkpoint['epoch'] == 10
        assert checkpoint['loss'] == 0.5
        
    def test_save_checkpoint_with_metadata(self, gomoku_evaluator, tmp_path):
        """Test saving checkpoint with model metadata"""
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        
        # Ensure model has metadata
        gomoku_evaluator.model.metadata.training_steps = 1000
        gomoku_evaluator.model.metadata.elo_rating = 1500.0
        
        gomoku_evaluator.save_checkpoint(str(checkpoint_path))
        
        # Check metadata file created
        metadata_path = tmp_path / "metadata.json"
        assert metadata_path.exists()
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        assert metadata['training_steps'] == 1000
        assert metadata['elo_rating'] == 1500.0
        
    def test_from_checkpoint(self, temp_checkpoint):
        """Test creating evaluator from checkpoint"""
        with patch('mcts.neural_networks.nn_framework.ModelLoader.load_checkpoint') as mock_load:
            mock_model = Mock()
            mock_model.board_size = 15
            mock_model.num_actions = 225
            mock_metadata = {'game_type': 'gomoku'}
            mock_load.return_value = (mock_model, mock_metadata)
            
            evaluator = ResNetEvaluator.from_checkpoint(temp_checkpoint)
            
            assert evaluator.model == mock_model
            mock_load.assert_called_once()


class TestGameSpecificCreation:
    """Test game-specific evaluator creation"""
    
    def test_create_evaluator_for_game(self):
        """Test creating evaluator for specific game"""
        evaluator = create_evaluator_for_game('gomoku')
        
        assert evaluator.game_type == 'gomoku'
        assert evaluator.model.board_size == 15
        assert evaluator.model.num_actions == 225
        
    def test_create_with_custom_architecture(self):
        """Test creating evaluator with custom architecture"""
        evaluator = create_evaluator_for_game(
            'go',
            num_blocks=5,
            num_filters=128
        )
        
        assert evaluator.model.config.num_blocks == 5
        assert evaluator.model.config.num_filters == 128
        
    def test_create_with_config_path(self, tmp_path):
        """Test creating evaluator with config file"""
        # Create config file
        config = AlphaZeroConfig()
        config.network.num_res_blocks = 7
        config.network.num_filters = 192
        config_path = tmp_path / "test_config.yaml"
        config.save(str(config_path))
        
        evaluator = create_evaluator_for_game(
            'chess',
            config_path=str(config_path)
        )
        
        assert evaluator.model.config.num_blocks == 7
        assert evaluator.model.config.num_filters == 192
        
    def test_create_chess_evaluator(self):
        """Test creating Chess evaluator"""
        evaluator = create_chess_evaluator(num_blocks=5)
        
        assert evaluator.game_type == 'chess'
        assert evaluator.model.board_size == 8
        assert evaluator.model.num_actions == 4096
        assert evaluator.model.config.num_blocks == 5
        
    def test_create_go_evaluator(self):
        """Test creating Go evaluator"""
        evaluator = create_go_evaluator(num_filters=192)
        
        assert evaluator.game_type == 'go'
        assert evaluator.model.board_size == 19
        assert evaluator.model.num_actions == 362
        assert evaluator.model.config.num_filters == 192
        
    def test_create_gomoku_evaluator(self):
        """Test creating Gomoku evaluator"""
        evaluator = create_gomoku_evaluator()
        
        assert evaluator.game_type == 'gomoku'
        assert evaluator.model.board_size == 15
        assert evaluator.model.num_actions == 225


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_batch_evaluation(self, gomoku_evaluator):
        """Test evaluation with empty batch"""
        empty_states = torch.randn(0, 20, 15, 15)
        
        policies, values = gomoku_evaluator._forward_model(empty_states)
        
        assert policies.shape == (0, 225)
        assert values.shape == (0,)
        
    def test_single_sample_batch(self, gomoku_evaluator):
        """Test evaluation with single sample"""
        single_state = torch.randn(1, 20, 15, 15)
        
        policies, values = gomoku_evaluator.forward_batch(single_state)
        
        assert policies.shape == (1, 225)
        assert values.shape == (1,)
        
    def test_no_legal_moves(self, gomoku_evaluator):
        """Test evaluation with no legal moves"""
        states = torch.randn(2, 20, 15, 15)
        legal_mask = torch.zeros(2, 225, dtype=torch.bool)
        
        policies, values = gomoku_evaluator.forward_batch(states, legal_mask)
        
        # Should handle gracefully (all zeros)
        assert torch.all(policies == 0)
        
    def test_device_mismatch_handling(self, gomoku_evaluator):
        """Test handling of device mismatches"""
        # Create states on different device (if available)
        if torch.cuda.is_available():
            cuda_states = torch.randn(4, 20, 15, 15).cuda()
            
            # Evaluator on CPU should handle this
            policies, values = gomoku_evaluator.evaluate_batch(cuda_states.cpu().numpy())
            
            assert policies.shape == (4, 225)
            assert values.shape == (4,)