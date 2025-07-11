"""Unified configuration system for MCTS"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

from .core.mcts_config import MCTSConfig
from .neural_networks.self_play_manager import SelfPlayConfig


class Config:
    """Unified configuration class that combines all sub-configurations"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration
        
        Args:
            config_path: Path to YAML configuration file
        """
        # Initialize sub-configurations
        self.mcts = MCTSConfig()
        self.self_play = SelfPlayConfig()
        
        # Additional configuration sections
        self.game = {
            'game_type': 'gomoku',
            'board_size': 15
        }
        
        self.network = {
            'model_type': 'resnet',
            'num_res_blocks': 8,
            'num_filters': 96,
            'value_head_hidden_size': 192,
            'policy_head_filters': 3
        }
        
        self.training = {
            'batch_size': 512,
            'learning_rate': 0.01,
            'learning_rate_schedule': 'cosine',
            'optimizer': 'adam'
        }
        
        self.arena = {
            'num_games': 20,
            'win_threshold': 0.53,
            'mcts_simulations': 200,
            'temperature': 0.0
        }
        
        self.optimization = {
            'n_trials': 100,
            'games_per_trial': 5,
            'warmup_games': 2,
            'trial_timeout': 300,
            'metric': 'simulations_per_second'
        }
        
        self.log = {
            'checkpoint_frequency': 10,
            'level': 'INFO'
        }
        
        # Load from file if provided
        if config_path:
            self.load_from_yaml(config_path)
    
    def load_from_yaml(self, config_path: str):
        """Load configuration from YAML file
        
        Args:
            config_path: Path to YAML file
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        self.update_from_dict(config_dict)
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary
        
        Args:
            config_dict: Configuration dictionary
        """
        # Update MCTS config
        if 'mcts' in config_dict:
            for key, value in config_dict['mcts'].items():
                if hasattr(self.mcts, key):
                    setattr(self.mcts, key, value)
        
        # Update self-play config
        if 'self_play' in config_dict:
            for key, value in config_dict['self_play'].items():
                if hasattr(self.self_play, key):
                    setattr(self.self_play, key, value)
        
        # Update other sections
        for section in ['game', 'network', 'training', 'arena', 'optimization', 'log']:
            if section in config_dict:
                if hasattr(self, section):
                    if isinstance(getattr(self, section), dict):
                        getattr(self, section).update(config_dict[section])
                    else:
                        setattr(self, section, config_dict[section])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary
        
        Returns:
            Configuration dictionary
        """
        result = {}
        
        # Convert MCTS config
        mcts_dict = {}
        for key in dir(self.mcts):
            if not key.startswith('_'):
                value = getattr(self.mcts, key)
                if not callable(value):
                    mcts_dict[key] = value
        result['mcts'] = mcts_dict
        
        # Convert self-play config
        self_play_dict = {}
        for key in dir(self.self_play):
            if not key.startswith('_'):
                value = getattr(self.self_play, key)
                if not callable(value):
                    self_play_dict[key] = value
        result['self_play'] = self_play_dict
        
        # Add other sections
        for section in ['game', 'network', 'training', 'arena', 'optimization', 'log']:
            if hasattr(self, section):
                result[section] = getattr(self, section)
        
        return result
    
    def save_to_yaml(self, output_path: str):
        """Save configuration to YAML file
        
        Args:
            output_path: Output file path
        """
        config_dict = self.to_dict()
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def __repr__(self) -> str:
        """String representation"""
        return f"Config({self.to_dict()})"