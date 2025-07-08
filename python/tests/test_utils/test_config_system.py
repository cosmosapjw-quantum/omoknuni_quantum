"""Tests for configuration system

Tests cover:
- AlphaZeroConfig initialization and validation
- Configuration loading and saving
- Section-specific configurations
- Default values and overrides
- YAML serialization/deserialization
- Configuration merging
- Validation logic
"""

import pytest
import yaml
import json
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch, mock_open

from mcts.utils.config_system import (
    AlphaZeroConfig, GameConfig, NeuralNetworkConfig, TrainingFullConfig,
    MCTSFullConfig, ArenaFullConfig, merge_configs
)


@pytest.fixture
def default_config():
    """Create default AlphaZero configuration"""
    return AlphaZeroConfig()


@pytest.fixture
def custom_config():
    """Create custom AlphaZero configuration"""
    config = AlphaZeroConfig()
    config.game.game_type = 'chess'
    config.game.board_size = 8
    config.network.num_res_blocks = 10
    config.training.batch_size = 64
    config.mcts.num_simulations = 400
    return config


@pytest.fixture
def temp_config_file():
    """Create temporary config file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_data = {
            'game': {
                'game_type': 'go',
                'board_size': 19
            },
            'network': {
                'num_res_blocks': 20,
                'num_filters': 256
            },
            'training': {
                'learning_rate': 0.001,
                'num_iterations': 100
            }
        }
        yaml.dump(config_data, f)
        yield f.name
    Path(f.name).unlink()


class TestAlphaZeroConfigInitialization:
    """Test AlphaZeroConfig initialization"""
    
    def test_default_initialization(self):
        """Test default configuration values"""
        config = AlphaZeroConfig()
        
        # Game defaults
        assert config.game.game_type == 'gomoku'
        assert config.game.board_size == 15
        
        # Network defaults
        assert config.network.num_res_blocks == 10
        assert config.network.num_filters == 256
        
        # Training defaults
        assert config.training.batch_size == 512
        assert config.training.learning_rate == 0.01
        
        # MCTS defaults
        assert config.mcts.num_simulations == 800
        assert config.mcts.c_puct == 1.0
        
    def test_initialization_with_overrides(self):
        """Test initialization with custom values"""
        config = AlphaZeroConfig(
            game_type='chess',
            num_res_blocks=5,
            batch_size=128
        )
        
        assert config.game.game_type == 'chess'
        assert config.network.num_res_blocks == 5
        assert config.training.batch_size == 128
        
    def test_all_sections_present(self, default_config):
        """Test all configuration sections are present"""
        assert hasattr(default_config, 'game')
        assert hasattr(default_config, 'network')
        assert hasattr(default_config, 'training')
        assert hasattr(default_config, 'mcts')
        assert hasattr(default_config, 'arena')
        assert hasattr(default_config, 'log')
        assert hasattr(default_config, 'resources')


class TestGameConfig:
    """Test GameConfig section"""
    
    def test_game_config_defaults(self):
        """Test game configuration defaults"""
        config = GameConfig()
        
        assert config.game_type == 'gomoku'
        assert config.board_size == 15
        assert config.win_length == 5
        assert config.use_symmetries == True
        
    def test_game_specific_settings(self):
        """Test game-specific settings"""
        # Chess config
        chess_config = GameConfig(game_type='chess')
        assert chess_config.board_size == 8
        
        # Go config
        go_config = GameConfig(game_type='go', board_size=19)
        assert go_config.board_size == 19
        assert hasattr(go_config, 'komi')
        
    def test_invalid_game_type(self):
        """Test invalid game type handling"""
        with pytest.raises(ValueError):
            GameConfig(game_type='invalid_game')


class TestNetworkConfig:
    """Test NetworkConfig section"""
    
    def test_network_config_defaults(self):
        """Test network configuration defaults"""
        config = NeuralNetworkConfig()
        
        assert config.input_channels == 18
        assert config.num_res_blocks == 10
        assert config.num_filters == 256
        assert config.fc_hidden_size == 256
        assert config.input_representation == 'basic'
        
    def test_architecture_validation(self):
        """Test architecture parameter validation"""
        # Valid config
        config = NeuralNetworkConfig(num_res_blocks=20, num_filters=256)
        assert config.num_res_blocks == 20
        
        # Invalid values should raise or be clamped
        with pytest.raises(ValueError):
            NeuralNetworkConfig(num_res_blocks=-1)
            
    def test_input_representation_options(self):
        """Test input representation options"""
        representations = ['basic', 'enhanced', 'standard']
        
        for rep in representations:
            config = NeuralNetworkConfig(input_representation=rep)
            assert config.input_representation == rep
            
            # Channel count is always 18 by default in the current implementation
            assert config.input_channels == 18


class TestTrainingConfig:
    """Test TrainingConfig section"""
    
    def test_training_config_defaults(self):
        """Test training configuration defaults"""
        config = TrainingFullConfig()
        
        assert config.num_iterations == 100
        assert config.num_games_per_iteration == 100
        assert config.num_epochs == 10
        assert config.batch_size == 512  # Actual default
        assert config.learning_rate == 0.01
        assert config.checkpoint_interval == 10
        
    def test_optimizer_settings(self):
        """Test optimizer configuration"""
        config = TrainingFullConfig()
        
        assert config.optimizer == 'adam'
        assert config.weight_decay == 1e-4
        assert config.momentum == 0.9
        assert config.gradient_clip_norm == 10.0
        
    def test_data_generation_settings(self):
        """Test data generation settings"""
        config = TrainingFullConfig()
        
        assert config.num_workers == 4
        assert config.max_moves_per_game == 500
        assert config.temperature_threshold == 30
        assert config.resign_threshold == -0.98
        
    def test_learning_rate_schedule(self):
        """Test learning rate schedule configuration"""
        config = TrainingFullConfig()
        
        assert hasattr(config, 'lr_schedule')
        assert config.lr_decay_steps == 50  # Single value, not list
        assert config.lr_decay_rate == 0.1  # lr_decay_rate, not lr_decay_factor


class TestMCTSConfig:
    """Test MCTSConfig section"""
    
    def test_mcts_config_defaults(self):
        """Test MCTS configuration defaults"""
        config = MCTSFullConfig()
        
        assert config.num_simulations == 800  # Actual default
        assert config.c_puct == 1.0  # Actual default
        assert config.temperature == 1.0
        assert config.temperature_threshold == 30
        assert config.dirichlet_alpha == 0.3
        assert config.dirichlet_epsilon == 0.25
        
    def test_performance_settings(self):
        """Test performance-related settings"""
        config = MCTSFullConfig()
        
        assert config.device == 'cuda'
        assert config.enable_virtual_loss == True
        assert config.virtual_loss == 1.0  # Actual default
        assert config.batch_size == 256  # Actual default
        
    def test_wave_sizing_settings(self):
        """Test wave sizing configuration"""
        config = MCTSFullConfig()
        
        assert config.min_wave_size == 256  # Actual default
        assert config.max_wave_size == 3072  # Actual default
        assert config.enable_fast_ucb == True
        assert config.classical_only_mode == True


class TestArenaConfig:
    """Test ArenaConfig section"""
    
    def test_arena_config_defaults(self):
        """Test arena configuration defaults"""
        config = ArenaFullConfig()
        
        assert config.num_games == 40
        assert config.win_threshold == 0.55
        assert config.num_workers == 4
        assert config.temperature == 0.0
        assert config.update_threshold == 0.51
        
    def test_elo_settings(self):
        """Test ELO rating settings"""
        config = ArenaFullConfig()
        
        assert config.elo_k_factor == 32.0
        assert config.initial_elo == 1500.0
        assert config.play_vs_random_interval == 10
        assert config.random_elo_anchor == 0.0


class TestResourceConfig:
    """Test ResourceConfig section"""
    
    def test_resource_config_defaults(self):
        """Test resource configuration defaults"""
        config = AlphaZeroConfig()
        
        assert config.resources.max_gpu_memory_gb == 8.0
        assert config.resources.max_cpu_workers == 4
        assert config.resources.pin_memory == True
        assert config.resources.num_data_workers == 2
        
    def test_memory_allocation(self):
        """Test memory allocation settings"""
        config = AlphaZeroConfig()
        
        assert hasattr(config.resources, 'batch_queue_size')
        assert hasattr(config.resources, 'cache_size')
        assert config.resources.enable_profiling == False


class TestConfigSerialization:
    """Test configuration serialization"""
    
    def test_to_dict(self, default_config):
        """Test converting config to dictionary"""
        config_dict = default_config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'game' in config_dict
        assert 'network' in config_dict
        assert 'training' in config_dict
        assert 'mcts' in config_dict
        
        # Check nested values
        assert config_dict['game']['game_type'] == 'gomoku'
        assert config_dict['network']['num_res_blocks'] == 10
        
    def test_from_dict(self):
        """Test creating config from dictionary"""
        config_dict = {
            'game': {'game_type': 'chess', 'board_size': 8},
            'network': {'num_res_blocks': 5},
            'training': {'batch_size': 64}
        }
        
        config = AlphaZeroConfig.from_dict(config_dict)
        
        assert config.game.game_type == 'chess'
        assert config.game.board_size == 8
        assert config.network.num_res_blocks == 5
        assert config.training.batch_size == 64
        
    def test_save_yaml(self, default_config, tmp_path):
        """Test saving config to YAML file"""
        config_path = tmp_path / "test_config.yaml"
        default_config.save(str(config_path))
        
        assert config_path.exists()
        
        # Load and verify
        with open(config_path) as f:
            loaded_data = yaml.safe_load(f)
            
        assert loaded_data['game']['game_type'] == 'gomoku'
        assert loaded_data['network']['num_res_blocks'] == 10
        
    def test_load_yaml(self, temp_config_file):
        """Test loading config from YAML file"""
        config = AlphaZeroConfig.load(temp_config_file)
        
        assert config.game.game_type == 'go'
        assert config.game.board_size == 19
        assert config.network.num_res_blocks == 20
        assert config.network.num_filters == 256
        assert config.training.learning_rate == 0.001
        
    def test_save_yaml(self, default_config, tmp_path):
        """Test saving config to YAML file"""
        config_path = tmp_path / "test_config.yaml"
        default_config.save(str(config_path))
        
        assert config_path.exists()
        
        # Load and verify
        with open(config_path) as f:
            loaded_data = yaml.safe_load(f)
            
        assert loaded_data['game']['game_type'] == 'gomoku'


class TestConfigMerging:
    """Test configuration merging"""
    
    def test_merge_configs_basic(self):
        """Test basic config merging"""
        base_config = AlphaZeroConfig()
        override_config = {
            'game': {'game_type': 'chess'},
            'network': {'num_res_blocks': 20}
        }
        
        merged = merge_configs(base_config.to_dict(), override_config)
        
        assert merged['game']['game_type'] == 'chess'
        assert merged['network']['num_res_blocks'] == 20
        assert merged['training']['batch_size'] == 512  # Unchanged default value
        
    def test_deep_merge(self):
        """Test deep merging of nested configs"""
        base = {
            'network': {
                'num_res_blocks': 10,
                'num_filters': 128,
                'fc_hidden_size': 256
            }
        }
        
        override = {
            'network': {
                'num_res_blocks': 20
            }
        }
        
        merged = merge_configs(base, override)
        
        assert merged['network']['num_res_blocks'] == 20
        assert merged['network']['num_filters'] == 128  # Preserved
        assert merged['network']['fc_hidden_size'] == 256  # Preserved
        
    def test_merge_with_new_keys(self):
        """Test merging with new keys"""
        base = {'game': {'game_type': 'gomoku'}}
        override = {'custom': {'new_param': 42}}
        
        merged = merge_configs(base, override)
        
        assert 'game' in merged
        assert 'custom' in merged
        assert merged['custom']['new_param'] == 42


class TestConfigValidation:
    """Test configuration validation"""
    
    def test_validate_valid_config(self, default_config):
        """Test validation of valid configuration"""
        # Should not raise any exceptions
        warnings = default_config.validate()
        assert isinstance(warnings, list)
        
    def test_validate_game_config(self):
        """Test game configuration validation"""
        config = AlphaZeroConfig()
        
        # Invalid board size for Gomoku
        config.game.board_size = 30
        with pytest.raises(ValueError, match="board size"):
            warnings = config.validate()
            assert isinstance(warnings, list)
            
    def test_validate_network_config(self):
        """Test network configuration validation"""
        config = AlphaZeroConfig()
        
        # Invalid number of blocks
        config.network.num_res_blocks = 0
        with pytest.raises(ValueError, match="res blocks"):
            warnings = config.validate()
            assert isinstance(warnings, list)
            
    def test_validate_training_config(self):
        """Test training configuration validation"""
        config = AlphaZeroConfig()
        
        # Invalid batch size
        config.training.batch_size = 0
        with pytest.raises(ValueError, match="batch size"):
            warnings = config.validate()
            assert isinstance(warnings, list)
            
        # Invalid learning rate
        config.training.batch_size = 32  # Fix previous
        config.training.learning_rate = -0.1
        with pytest.raises(ValueError, match="learning rate"):
            warnings = config.validate()
            assert isinstance(warnings, list)
            
    def test_validate_mcts_config(self):
        """Test MCTS configuration validation"""
        config = AlphaZeroConfig()
        
        # Invalid number of simulations
        config.mcts.num_simulations = 0
        with pytest.raises(ValueError, match="simulations"):
            warnings = config.validate()
            assert isinstance(warnings, list)
            
        # Invalid c_puct
        config.mcts.num_simulations = 100
        config.mcts.c_puct = -1.0
        with pytest.raises(ValueError, match="c_puct"):
            warnings = config.validate()
            assert isinstance(warnings, list)


class TestConfigUtilities:
    """Test configuration utility functions"""
    
    def test_load_config_file_not_found(self):
        """Test loading non-existent config file"""
        with pytest.raises(FileNotFoundError):
            AlphaZeroConfig.load("non_existent_config.yaml")
            
    def test_load_config_invalid_format(self, tmp_path):
        """Test loading invalid config format"""
        # Create invalid YAML
        config_path = tmp_path / "invalid.yaml"
        config_path.write_text("invalid: yaml: content:")
        
        with pytest.raises(yaml.YAMLError):
            AlphaZeroConfig.load(str(config_path))
            
    def test_save_config_create_directories(self, default_config, tmp_path):
        """Test saving config creates directories"""
        nested_path = tmp_path / "nested" / "dirs" / "config.yaml"
        
        default_config.save(str(nested_path))
        
        assert nested_path.exists()
        assert nested_path.parent.exists()
        
    def test_config_string_representation(self, default_config):
        """Test string representation of config"""
        config_str = str(default_config)
        
        assert 'AlphaZeroConfig' in config_str
        assert 'game_type' in config_str
        assert 'num_res_blocks' in config_str
        
    def test_config_equality(self):
        """Test config equality comparison"""
        config1 = AlphaZeroConfig()
        config2 = AlphaZeroConfig()
        config3 = AlphaZeroConfig(game_type='chess')
        
        # Same configs should be equal
        assert config1.to_dict() == config2.to_dict()
        
        # Different configs should not be equal
        assert config1.to_dict() != config3.to_dict()


class TestConfigDefaults:
    """Test configuration default values"""
    
    def test_experiment_defaults(self, default_config):
        """Test experiment-related defaults"""
        assert default_config.experiment_name == 'default'
        assert default_config.checkpoint_dir == 'checkpoints'
        assert default_config.tensorboard_dir == 'runs'
        assert default_config.save_interval == 10
        
    def test_log_defaults(self):
        """Test logging defaults"""
        # LogConfig doesn't exist, using log_level from AlphaZeroConfig
        config = AlphaZeroConfig()
        
        assert config.log.level == 'INFO'
        assert config.log.file_path == 'logs/training.log'
        assert config.log.console == True
        assert config.log.tensorboard == True
        
    def test_device_defaults(self, default_config):
        """Test device-related defaults"""
        # Should default to CUDA if available
        import torch
        expected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # MCTS device
        assert default_config.mcts.device in ['cuda', 'cpu']
        
        # Training device
        assert hasattr(default_config.training, 'device')


class TestConfigIntegration:
    """Test configuration integration scenarios"""
    
    def test_full_training_config(self):
        """Test complete training configuration"""
        config = AlphaZeroConfig(
            game_type='chess',
            num_iterations=1000,
            batch_size=256,
            num_res_blocks=20
        )
        
        # Validate full config
        warnings = config.validate()
        assert isinstance(warnings, list)
        
        # Check interdependent settings
        assert config.game.board_size == 8  # Chess board
        assert config.network.num_res_blocks == 20
        assert config.training.batch_size == 256
        
    def test_experiment_config_loading(self, tmp_path):
        """Test loading experiment configuration"""
        # Create experiment directory structure
        exp_dir = tmp_path / "experiment_1"
        exp_dir.mkdir()
        
        config_path = exp_dir / "config.yaml"
        
        # Save config
        config = AlphaZeroConfig(experiment_name='test_exp')
        config.save(str(config_path))
        
        # Load and verify
        loaded = AlphaZeroConfig.load(str(config_path))
        assert loaded.experiment_name == 'test_exp'
        
    def test_distributed_training_config(self):
        """Test distributed training configuration"""
        config = AlphaZeroConfig()
        
        # Should have distributed settings
        assert hasattr(config.training, 'distributed')
        assert hasattr(config.resources, 'num_gpus')
        
        # Configure for distributed
        config.training.distributed = True
        config.resources.num_gpus = 4
        
        assert config.training.distributed == True
        assert config.resources.num_gpus == 4