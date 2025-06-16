"""Neural Network Framework for AlphaZero

This module provides the base classes and utilities for neural network models
used in the AlphaZero MCTS implementation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a trained model"""
    game_type: str
    board_size: int
    num_actions: int
    input_channels: int
    num_blocks: int
    num_filters: int
    version: str = "1.0"
    training_steps: int = 0
    elo_rating: float = 1200.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'game_type': self.game_type,
            'board_size': self.board_size,
            'num_actions': self.num_actions,
            'input_channels': self.input_channels,
            'num_blocks': self.num_blocks,
            'num_filters': self.num_filters,
            'version': self.version,
            'training_steps': self.training_steps,
            'elo_rating': self.elo_rating
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary"""
        return cls(**data)
    
    def save(self, path: str):
        """Save metadata to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class BaseGameModel(nn.Module):
    """Base class for game-specific neural network models"""
    
    def __init__(self):
        super().__init__()
        self.metadata: Optional[ModelMetadata] = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning policy and value
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of:
                - policy: Tensor of shape (batch_size, num_actions)
                - value: Tensor of shape (batch_size, 1)
        """
        raise NotImplementedError
    
    def save_checkpoint(self, path: Path, optimizer: Optional[torch.optim.Optimizer] = None):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'metadata': self.metadata.to_dict() if self.metadata else {},
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Path, optimizer: Optional[torch.optim.Optimizer] = None):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        if 'metadata' in checkpoint:
            self.metadata = ModelMetadata.from_dict(checkpoint['metadata'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from {path}")


class ModelLoader:
    """Utility class for loading models"""
    
    @staticmethod
    def load_checkpoint(path: str, device: str = 'cuda') -> Tuple[BaseGameModel, ModelMetadata]:
        """Load a model checkpoint
        
        Args:
            path: Path to checkpoint file
            device: Device to load model to
            
        Returns:
            Tuple of (model, metadata)
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=device)
        metadata = ModelMetadata.from_dict(checkpoint['metadata'])
        
        # Create model based on metadata
        if metadata.game_type == 'chess':
            from .resnet_model import create_resnet_for_game
            model = create_resnet_for_game('chess', input_channels=metadata.input_channels)
        elif metadata.game_type == 'go':
            from .resnet_model import create_resnet_for_game
            model = create_resnet_for_game('go', input_channels=metadata.input_channels)
        elif metadata.game_type == 'gomoku':
            from .resnet_model import create_resnet_for_game
            model = create_resnet_for_game('gomoku', input_channels=metadata.input_channels)
        else:
            raise ValueError(f"Unknown game type: {metadata.game_type}")
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.metadata = metadata
        model.to(device)
        model.eval()
        
        return model, metadata


class ModelRegistry:
    """Registry for managing multiple models"""
    
    def __init__(self):
        self.models: Dict[str, BaseGameModel] = {}
        self.metadata: Dict[str, ModelMetadata] = {}
    
    def register(self, name: str, model: BaseGameModel, metadata: ModelMetadata):
        """Register a model"""
        self.models[name] = model
        self.metadata[name] = metadata
    
    def get(self, name: str) -> Tuple[BaseGameModel, ModelMetadata]:
        """Get a registered model"""
        if name not in self.models:
            raise KeyError(f"Model '{name}' not found in registry")
        return self.models[name], self.metadata[name]
    
    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(self.models.keys())


class ModelEnsemble(nn.Module):
    """Ensemble of multiple models for stronger play"""
    
    def __init__(self, models: List[BaseGameModel], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = torch.tensor(weights)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through ensemble"""
        policies = []
        values = []
        
        for model, weight in zip(self.models, self.weights):
            policy, value = model(x)
            policies.append(policy * weight)
            values.append(value * weight)
        
        # Weighted average
        ensemble_policy = torch.stack(policies).sum(dim=0)
        ensemble_value = torch.stack(values).sum(dim=0)
        
        return ensemble_policy, ensemble_value


class AdapterWrapper(nn.Module):
    """Wrapper to adapt models with different interfaces"""
    
    def __init__(self, model: nn.Module, adapter_fn):
        super().__init__()
        self.model = model
        self.adapter_fn = adapter_fn
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with adaptation"""
        return self.adapter_fn(self.model, x)


class MixedPrecisionWrapper(nn.Module):
    """Wrapper for mixed precision training/inference"""
    
    def __init__(self, model: BaseGameModel):
        super().__init__()
        self.model = model
        self.metadata = model.metadata
    
    @torch.amp.autocast('cuda')
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with mixed precision"""
        return self.model(x)


def create_model_from_config(config: Dict[str, Any]) -> BaseGameModel:
    """Create a model from configuration dictionary"""
    game_type = config['game_type']
    
    if game_type in ['chess', 'go', 'gomoku']:
        from .resnet_model import create_resnet_for_game
        return create_resnet_for_game(
            game_type,
            input_channels=config.get('input_channels', 20),
            num_blocks=config.get('num_blocks', 19),
            num_filters=config.get('num_filters', 256)
        )
    else:
        raise ValueError(f"Unknown game type: {game_type}")


def load_model_for_game(game_type: str, checkpoint_path: Optional[str] = None) -> BaseGameModel:
    """Load or create a model for a specific game"""
    if checkpoint_path:
        model, _ = ModelLoader.load_checkpoint(checkpoint_path)
        return model
    else:
        # Create new model with default configuration
        from .resnet_model import create_resnet_for_game
        return create_resnet_for_game(game_type)