"""Tests for configuration system"""

import pytest
import tempfile
import json
import os
from dataclasses import asdict

from mcts.utils.config_system import (
    AlphaZeroConfig,
    MCTSFullConfig,
    GameConfig,
    TrainingFullConfig,
    ArenaFullConfig,
    NeuralNetworkConfig,
    QuantumLevel,
    create_default_config,
    merge_configs
)


class TestQuantumLevel:
    """Test QuantumLevel enum"""
    
    def test_quantum_level_values(self):
        """Test QuantumLevel enum values"""
        assert QuantumLevel.CLASSICAL == "classical"
        assert QuantumLevel.TREE_LEVEL == "tree_level"
        assert QuantumLevel.ONE_LOOP == "one_loop"
    
    def test_quantum_level_ordering(self):
        """Test QuantumLevel ordering functionality"""
        levels = [QuantumLevel.CLASSICAL, QuantumLevel.TREE_LEVEL, 
                 QuantumLevel.ONE_LOOP]
        assert len(levels) == 3
        assert all(isinstance(level, QuantumLevel) for level in levels)


class TestAlphaZeroConfig:
    """Test AlphaZeroConfig dataclass"""
    
    def test_alphazero_config_creation_defaults(self):
        """Test creating AlphaZeroConfig with defaults"""
        config = AlphaZeroConfig()
        
        # Master config has sub-configs
        assert hasattr(config, 'game')
        assert hasattr(config, 'mcts')
        assert hasattr(config, 'network')
        assert hasattr(config, 'training')
        assert hasattr(config, 'arena')
        
        # Game config
        assert config.game.game_type == 'gomoku'
        assert config.game.board_size == 15
        
        # MCTS config
        assert config.mcts.num_simulations == 800
        assert config.mcts.c_puct == 1.0
        assert config.mcts.temperature == 1.0
        assert config.mcts.dirichlet_alpha == 0.3
        assert config.mcts.dirichlet_epsilon == 0.25
        
        # Training config
        assert config.training.batch_size == 512
        assert config.training.learning_rate == 0.01
        assert config.training.num_workers == 4
        
        # Global settings
        assert config.experiment_name == 'alphazero_experiment'
        assert config.seed == 42
        assert config.log_level == 'INFO'
        assert config.num_iterations == 1000
    
    def test_alphazero_config_creation_custom(self):
        """Test creating AlphaZeroConfig with custom values"""
        # Create custom sub-configs
        game_config = GameConfig(game_type='go', board_size=19)
        mcts_config = MCTSFullConfig(num_simulations=1600)
        training_config = TrainingFullConfig(batch_size=64, learning_rate=0.002)
        
        config = AlphaZeroConfig(
            game=game_config,
            mcts=mcts_config,
            training=training_config
        )
        
        assert config.game.game_type == 'go'
        assert config.game.board_size == 19
        assert config.mcts.num_simulations == 1600
        assert config.training.batch_size == 64
        assert config.training.learning_rate == 0.002
    
    def test_alphazero_config_to_dict(self):
        """Test converting AlphaZeroConfig to dictionary"""
        game_config = GameConfig(game_type='chess', board_size=8)
        mcts_config = MCTSFullConfig(num_simulations=400)
        
        config = AlphaZeroConfig(
            game=game_config,
            mcts=mcts_config
        )
        
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['game']['game_type'] == 'chess'
        assert config_dict['game']['board_size'] == 8
        assert config_dict['mcts']['num_simulations'] == 400
        assert 'experiment_name' in config_dict
    
    def test_alphazero_config_serialization(self):
        """Test AlphaZeroConfig JSON serialization"""
        mcts_config = MCTSFullConfig(quantum_level=QuantumLevel.TREE_LEVEL)
        
        config = AlphaZeroConfig(
            mcts=mcts_config
        )
        
        # Convert to dict
        config_dict = config.to_dict()
        
        # Should be JSON serializable
        json_str = json.dumps(config_dict)
        loaded_dict = json.loads(json_str)
        
        assert loaded_dict['game']['game_type'] == 'gomoku'
        assert loaded_dict['game']['board_size'] == 15
        assert loaded_dict['mcts']['quantum_level'] == 'tree_level'


class TestMCTSFullConfig:
    """Test MCTSFullConfig dataclass"""
    
    def test_mcts_full_config_creation_defaults(self):
        """Test creating MCTSFullConfig with defaults"""
        config = MCTSFullConfig()
        
        # Game settings
        assert config.game_type == 'gomoku'
        assert config.board_size == 15
        
        # Core MCTS settings
        assert config.num_simulations == 1000
        assert config.c_puct == 1.414
        assert config.temperature == 1.0
        assert config.dirichlet_alpha == 0.3
        assert config.dirichlet_epsilon == 0.25
        
        # Performance settings
        assert config.batch_size == 64
        assert config.device == 'cuda'
        assert config.num_threads == 1
        
        # Tree settings
        assert config.max_tree_depth == 100
        assert config.virtual_loss == 3
        
        # Evaluation settings
        assert config.evaluation_batch_size == 32
        assert config.evaluation_timeout == 30.0
        
        # Quantum settings
        assert config.quantum_level == QuantumLevel.CLASSICAL
        assert config.enable_quantum == False
        assert config.quantum_c_puct_boost == 0.1
        assert config.quantum_temperature_boost == 0.05
    
    def test_mcts_full_config_creation_custom(self):
        """Test creating MCTSFullConfig with custom values"""
        config = MCTSFullConfig(
            game_type='go',
            board_size=19,
            num_simulations=2000,
            c_puct=2.0,
            batch_size=128,
            device='cpu',
            quantum_level=QuantumLevel.ONE_LOOP,
            enable_quantum=True,
            quantum_c_puct_boost=0.2
        )
        
        assert config.game_type == 'go'
        assert config.board_size == 19
        assert config.num_simulations == 2000
        assert config.c_puct == 2.0
        assert config.batch_size == 128
        assert config.device == 'cpu'
        assert config.quantum_level == QuantumLevel.ONE_LOOP
        assert config.enable_quantum == True
        assert config.quantum_c_puct_boost == 0.2
    
    def test_mcts_full_config_to_dict(self):
        """Test converting MCTSFullConfig to dictionary"""
        config = MCTSFullConfig(
            num_simulations=500,
            quantum_level=QuantumLevel.TREE_LEVEL
        )
        
        config_dict = asdict(config)
        assert isinstance(config_dict, dict)
        assert config_dict['num_simulations'] == 500
        assert config_dict['quantum_level'] == QuantumLevel.TREE_LEVEL
        assert 'c_puct' in config_dict
        assert 'enable_quantum' in config_dict


class TestConfigUtilities:
    """Test configuration utility functions"""
    
    def test_create_default_config_alphazero(self):
        """Test creating default AlphaZero config"""
        config = create_default_config('gomoku')
        
        assert isinstance(config, AlphaZeroConfig)
        assert config.game.game_type == 'gomoku'
        assert config.game.board_size == 15
        assert config.mcts.num_simulations == 800
    
    def test_create_default_config_go(self):
        """Test creating default config for Go"""
        config = create_default_config('go')
        
        assert isinstance(config, AlphaZeroConfig)
        assert config.game.game_type == 'go'
        assert config.game.board_size == 19
        assert config.game.go_komi == 7.5
    
    def test_create_default_config_chess(self):
        """Test creating default config for Chess"""
        config = create_default_config('chess')
        
        assert isinstance(config, AlphaZeroConfig)
        assert config.game.game_type == 'chess'
        assert config.game.board_size == 8
        # Other defaults should be preserved
        assert config.mcts.num_simulations == 800
        assert config.training.batch_size == 512
    
    def test_merge_configs_alphazero(self):
        """Test merging AlphaZero configs with override dict"""
        base_config = create_default_config('gomoku')
        
        override_dict = {
            'mcts': {
                'num_simulations': 1600
            },
            'training': {
                'batch_size': 64,
                'learning_rate': 0.002
            }
        }
        
        merged = merge_configs(base_config, override_dict)
        
        # Should preserve base values where not overridden
        assert merged.game.game_type == 'gomoku'
        assert merged.game.board_size == 15
        
        # Should use override values
        assert merged.mcts.num_simulations == 1600
        assert merged.training.batch_size == 64
        assert merged.training.learning_rate == 0.002
    
    def test_merge_configs_mcts(self):
        """Test merging MCTS configs"""
        base_config = MCTSFullConfig(
            game_type='go',
            board_size=19,
            num_simulations=1000,
            c_puct=1.414
        )
        
        override_config = MCTSFullConfig(
            num_simulations=2000,
            quantum_level=QuantumLevel.TREE_LEVEL,
            enable_quantum=True
        )
        
        merged = merge_configs(base_config, override_config)
        
        # Should preserve base values
        assert merged.game_type == 'go'
        assert merged.board_size == 19
        assert merged.c_puct == 1.414
        
        # Should use override values
        assert merged.num_simulations == 2000
        assert merged.quantum_level == QuantumLevel.TREE_LEVEL
        assert merged.enable_quantum == True
    
    def test_merge_configs_deep_merge(self):
        """Test deep merging of nested configs"""
        base_config = create_default_config('gomoku')
        
        override_dict = {
            'game': {'board_size': 19},
            'mcts': {'c_puct': 2.0}
        }
        
        merged = merge_configs(base_config, override_dict)
        
        # Should merge nested values
        assert merged.game.board_size == 19
        assert merged.mcts.c_puct == 2.0
        # Should preserve other values in nested configs
        assert merged.game.game_type == 'gomoku'
        assert merged.mcts.num_simulations == 800
    
    def test_merge_configs_partial_override(self):
        """Test merging configs with partial override"""
        base_mcts = MCTSFullConfig(
            num_simulations=800,
            quantum_level=QuantumLevel.CLASSICAL
        )
        
        override_mcts = MCTSFullConfig(
            num_simulations=1200,
            quantum_level=QuantumLevel.TREE_LEVEL,
            enable_quantum=True
        )
        
        merged = merge_configs(base_mcts, override_mcts)
        
        # Should keep base values for non-overridden fields
        assert merged.c_puct == 1.0  # Default value
        assert merged.temperature == 1.0  # Default value
        
        # Should use override values
        assert merged.num_simulations == 1200
        assert merged.quantum_level == QuantumLevel.TREE_LEVEL
        assert merged.enable_quantum == True


class TestConfigValidation:
    """Test configuration validation"""
    
    def test_alphazero_config_validation_basic(self):
        """Test basic AlphaZero config validation"""
        # Valid config should not raise
        config = AlphaZeroConfig(
            game_type='gomoku',
            board_size=15,
            num_simulations=800,
            batch_size=32,
            lr=0.001
        )
        assert config.game_type == 'gomoku'
        assert config.num_simulations > 0
        assert config.batch_size > 0
        assert config.lr > 0
    
    def test_mcts_config_validation_basic(self):
        """Test basic MCTS config validation"""
        # Valid config should not raise
        config = MCTSFullConfig(
            game_type='go',
            board_size=19,
            num_simulations=1000,
            c_puct=1.414,
            temperature=1.0
        )
        assert config.game_type == 'go'
        assert config.num_simulations > 0
        assert config.c_puct > 0
        assert config.temperature >= 0
    
    def test_config_quantum_consistency(self):
        """Test quantum configuration consistency"""
        # Classical quantum level with quantum disabled
        config1 = MCTSFullConfig(
            quantum_level=QuantumLevel.CLASSICAL,
            enable_quantum=False
        )
        assert config1.quantum_level == QuantumLevel.CLASSICAL
        assert config1.enable_quantum == False
        
        # Non-classical quantum level with quantum enabled
        config2 = MCTSFullConfig(
            quantum_level=QuantumLevel.TREE_LEVEL,
            enable_quantum=True
        )
        assert config2.quantum_level == QuantumLevel.TREE_LEVEL
        assert config2.enable_quantum == True


class TestConfigPersistence:
    """Test configuration persistence (save/load)"""
    
    def test_save_load_alphazero_config(self):
        """Test saving and loading AlphaZero config"""
        game_config = GameConfig(game_type='chess', board_size=8)
        mcts_config = MCTSFullConfig(num_simulations=1600, quantum_level=QuantumLevel.TREE_LEVEL, enable_quantum=True)
        
        original_config = AlphaZeroConfig(
            game=game_config,
            mcts=mcts_config,
            experiment_name='test_chess',
            num_iterations=500
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_dict = original_config.to_dict()
            json.dump(config_dict, f)
            temp_path = f.name
        
        try:
            # Load config back
            with open(temp_path, 'r') as f:
                loaded_dict = json.load(f)
            
            loaded_config = AlphaZeroConfig.from_dict(loaded_dict)
            
            # Should match original
            assert loaded_config.game.game_type == original_config.game.game_type
            assert loaded_config.game.board_size == original_config.game.board_size
            assert loaded_config.mcts.num_simulations == original_config.mcts.num_simulations
            assert loaded_config.mcts.quantum_level == original_config.mcts.quantum_level
            assert loaded_config.mcts.enable_quantum == original_config.mcts.enable_quantum
            assert loaded_config.experiment_name == original_config.experiment_name
            assert loaded_config.num_iterations == original_config.num_iterations
        
        finally:
            os.unlink(temp_path)
    
    def test_save_load_mcts_config(self):
        """Test saving and loading MCTS config"""
        original_config = MCTSFullConfig(
            game_type='go',
            board_size=19,
            num_simulations=2000,
            c_puct=2.0,
            quantum_level=QuantumLevel.ONE_LOOP,
            enable_quantum=True,
            quantum_c_puct_boost=0.3
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_dict = asdict(original_config)
            config_dict['quantum_level'] = config_dict['quantum_level'].value
            json.dump(config_dict, f)
            temp_path = f.name
        
        try:
            with open(temp_path, 'r') as f:
                loaded_dict = json.load(f)
            
            loaded_dict['quantum_level'] = QuantumLevel(loaded_dict['quantum_level'])
            loaded_config = MCTSFullConfig(**loaded_dict)
            
            assert loaded_config.game_type == original_config.game_type
            assert loaded_config.board_size == original_config.board_size
            assert loaded_config.num_simulations == original_config.num_simulations
            assert loaded_config.c_puct == original_config.c_puct
            assert loaded_config.quantum_level == original_config.quantum_level
            assert loaded_config.enable_quantum == original_config.enable_quantum
            assert loaded_config.quantum_c_puct_boost == original_config.quantum_c_puct_boost
        
        finally:
            os.unlink(temp_path)


class TestConfigIntegration:
    """Integration tests for configuration system"""
    
    def test_config_workflow_alphazero(self):
        """Test complete AlphaZero config workflow"""
        # Create base config
        base_config = create_default_config('gomoku')
        
        # Create training-specific overrides
        training_overrides = {
            'mcts': {
                'num_simulations': 1600,
                'quantum_level': 'tree_level',
                'enable_quantum': True
            },
            'training': {
                'batch_size': 64,
                'learning_rate': 0.002
            }
        }
        
        # Merge configs
        final_config = merge_configs(base_config, training_overrides)
        
        # Verify final config
        assert final_config.game.game_type == 'gomoku'
        assert final_config.game.board_size == 15
        assert final_config.mcts.num_simulations == 1600
        assert final_config.training.batch_size == 64
        assert final_config.training.learning_rate == 0.002
        assert final_config.mcts.quantum_level == QuantumLevel.TREE_LEVEL
        assert final_config.mcts.enable_quantum == True
        
        # Should preserve other defaults
        assert final_config.network.num_filters == 128  # Gomoku default
        assert final_config.training.weight_decay == 1e-4  # Default value
    
    def test_config_workflow_go(self):
        """Test complete Go config workflow"""
        # Create base config
        base_config = create_default_config('go')
        
        # Create performance-specific overrides
        performance_overrides = {
            'mcts': {
                'num_simulations': 2000,
                'batch_size': 128,
                'device': 'cuda',
                'quantum_level': 'tree_level',
                'enable_quantum': True,
                'quantum_c_puct_boost': 0.2
            }
        }
        
        # Merge configs
        final_config = merge_configs(base_config, performance_overrides)
        
        # Verify final config
        assert final_config.game.game_type == 'go'
        assert final_config.game.board_size == 19
        assert final_config.mcts.num_simulations == 2000
        assert final_config.mcts.batch_size == 128
        assert final_config.mcts.device == 'cuda'
        assert final_config.mcts.quantum_level == QuantumLevel.TREE_LEVEL
        assert final_config.mcts.enable_quantum == True
        assert final_config.mcts.quantum_c_puct_boost == 0.2
        
        # Should preserve other defaults
        assert final_config.mcts.c_puct == 1.0  # Default value
        assert final_config.mcts.temperature == 1.0  # Default value