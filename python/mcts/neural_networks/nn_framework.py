"""Flexible Neural Network Framework for MCTS

This module provides a flexible framework for importing and using various neural network
architectures with the MCTS engine. It supports:
- Dynamic model loading from checkpoints
- Multiple model architectures (ResNet, EfficientNet, Transformer, etc.)
- Automatic shape inference and adaptation
- Mixed precision training/inference
- Model ensembles
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from abc import ABC, abstractmethod
import os
import json
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import importlib.util

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a neural network model"""
    architecture: str
    input_channels: int
    board_height: int
    board_width: int
    num_actions: int
    model_params: Dict[str, Any]
    training_config: Optional[Dict[str, Any]] = None
    performance_stats: Optional[Dict[str, Any]] = None
    
    def save(self, path: str):
        """Save metadata to JSON"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ModelMetadata':
        """Load metadata from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class BaseGameModel(nn.Module, ABC):
    """Base class for all game evaluation models"""
    
    def __init__(self, metadata: ModelMetadata):
        super().__init__()
        self.metadata = metadata
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            policy_logits: Shape (batch, num_actions)
            value: Shape (batch, 1)
        """
        pass
    
    def get_input_shape(self) -> Tuple[int, int, int]:
        """Get expected input shape (channels, height, width)"""
        return (
            self.metadata.input_channels,
            self.metadata.board_height,
            self.metadata.board_width
        )
    
    def get_num_actions(self) -> int:
        """Get number of actions"""
        return self.metadata.num_actions


class ModelRegistry:
    """Registry for model architectures"""
    
    _architectures: Dict[str, type] = {}
    _factories: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, name: str, model_class: type = None, factory: Callable = None):
        """Register a model architecture"""
        if model_class is not None:
            cls._architectures[name] = model_class
        if factory is not None:
            cls._factories[name] = factory
    
    @classmethod
    def create(cls, name: str, metadata: ModelMetadata, **kwargs) -> BaseGameModel:
        """Create a model instance"""
        if name in cls._factories:
            return cls._factories[name](metadata, **kwargs)
        elif name in cls._architectures:
            return cls._architectures[name](metadata, **kwargs)
        else:
            raise ValueError(f"Unknown architecture: {name}")
    
    @classmethod
    def list_architectures(cls) -> List[str]:
        """List all registered architectures"""
        return list(set(cls._architectures.keys()) | set(cls._factories.keys()))


class ModelLoader:
    """Utility class for loading models from various sources"""
    
    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        device: str = 'cuda',
        map_location: Optional[str] = None
    ) -> Tuple[BaseGameModel, ModelMetadata]:
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on
            map_location: Device mapping for loading
            
        Returns:
            model: Loaded model
            metadata: Model metadata
        """
        checkpoint_dir = Path(checkpoint_path).parent
        
        # Load metadata
        metadata_path = checkpoint_dir / 'metadata.json'
        if metadata_path.exists():
            metadata = ModelMetadata.load(str(metadata_path))
        else:
            # Try to infer from checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=map_location or device)
            metadata = ModelLoader._infer_metadata(checkpoint)
        
        # Create model
        model = ModelRegistry.create(metadata.architecture, metadata)
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=map_location or device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        return model, metadata
    
    @staticmethod
    def load_from_module(
        module_path: str,
        class_name: str,
        metadata: ModelMetadata,
        device: str = 'cuda'
    ) -> BaseGameModel:
        """
        Load model from Python module
        
        Args:
            module_path: Path to Python file
            class_name: Name of model class
            metadata: Model metadata
            device: Device to load model on
            
        Returns:
            model: Loaded model instance
        """
        spec = importlib.util.spec_from_file_location("custom_model", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        model_class = getattr(module, class_name)
        model = model_class(metadata)
        model.to(device)
        model.eval()
        
        return model
    
    @staticmethod
    def _infer_metadata(checkpoint: Dict[str, Any]) -> ModelMetadata:
        """Infer metadata from checkpoint"""
        # This is a fallback - better to have explicit metadata
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Try to infer from layer shapes
        # This is model-specific and should be overridden
        return ModelMetadata(
            architecture='unknown',
            input_channels=20,  # Default AlphaZero
            board_height=19,
            board_width=19,
            num_actions=361,
            model_params={}
        )


class ModelEnsemble(BaseGameModel):
    """Ensemble of multiple models for improved performance"""
    
    def __init__(self, models: List[BaseGameModel], weights: Optional[List[float]] = None):
        # Use metadata from first model
        super().__init__(models[0].metadata)
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = torch.tensor(weights)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through ensemble"""
        all_policies = []
        all_values = []
        
        for i, model in enumerate(self.models):
            policy, value = model(x)
            all_policies.append(policy * self.weights[i])
            all_values.append(value * self.weights[i])
        
        # Weighted average
        ensemble_policy = torch.stack(all_policies).sum(dim=0)
        ensemble_value = torch.stack(all_values).sum(dim=0)
        
        return ensemble_policy, ensemble_value


class AdapterWrapper(BaseGameModel):
    """Wrapper to adapt models with different interfaces"""
    
    def __init__(self, base_model: nn.Module, metadata: ModelMetadata,
                 forward_fn: Optional[Callable] = None):
        super().__init__(metadata)
        self.base_model = base_model
        self.forward_fn = forward_fn
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with adaptation"""
        if self.forward_fn is not None:
            return self.forward_fn(self.base_model, x)
        
        # Default: assume model returns (policy, value)
        output = self.base_model(x)
        
        if isinstance(output, tuple) and len(output) == 2:
            return output
        elif isinstance(output, dict):
            return output['policy'], output['value']
        else:
            raise ValueError("Cannot adapt model output format")


class MixedPrecisionWrapper(BaseGameModel):
    """Wrapper for mixed precision inference"""
    
    def __init__(self, base_model: BaseGameModel, enabled: bool = True):
        super().__init__(base_model.metadata)
        self.base_model = base_model
        self.enabled = enabled and torch.cuda.is_available()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with mixed precision"""
        if self.enabled:
            with torch.cuda.amp.autocast():
                policy, value = self.base_model(x)
                # Ensure FP32 output
                return policy.float(), value.float()
        else:
            return self.base_model(x)


# Import standard architectures
from .resnet_model import ResNetModel, ResNetConfig
# TODO: Add these when implemented
# from .efficientnet_model import EfficientNetModel, EfficientNetConfig
# from .transformer_model import TransformerModel, TransformerConfig

# Register standard architectures
ModelRegistry.register('resnet', ResNetModel)
# TODO: Register these when implemented
# ModelRegistry.register('efficientnet', EfficientNetModel)
# ModelRegistry.register('transformer', TransformerModel)


def create_model_from_config(config_path: str, device: str = 'cuda') -> BaseGameModel:
    """
    Create model from configuration file
    
    Args:
        config_path: Path to model configuration
        device: Device to load model on
        
    Returns:
        model: Created model instance
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    metadata = ModelMetadata(**config['metadata'])
    model = ModelRegistry.create(
        metadata.architecture,
        metadata,
        **config.get('model_params', {})
    )
    
    model.to(device)
    return model


def load_model_for_game(
    game_type: str,
    checkpoint_path: Optional[str] = None,
    architecture: str = 'resnet',
    device: str = 'cuda'
) -> BaseGameModel:
    """
    Load or create model for specific game
    
    Args:
        game_type: Type of game ('chess', 'go', 'gomoku')
        checkpoint_path: Path to checkpoint (optional)
        architecture: Model architecture to use
        device: Device to load on
        
    Returns:
        model: Model instance
    """
    # Game-specific configurations - All games use 20 input channels
    game_configs = {
        'chess': {
            'input_channels': 20,  # Board + player + 8 history per player + attack/defense
            'board_height': 8,
            'board_width': 8,
            'num_actions': 4096
        },
        'go': {
            'input_channels': 20,  # Board + player + 8 history per player + attack/defense
            'board_height': 19,
            'board_width': 19,
            'num_actions': 362  # 19x19 + pass
        },
        'gomoku': {
            'input_channels': 20,  # Board + player + 8 history per player + attack/defense
            'board_height': 15,
            'board_width': 15,
            'num_actions': 225
        }
    }
    
    if game_type not in game_configs:
        raise ValueError(f"Unknown game type: {game_type}")
    
    game_config = game_configs[game_type]
    
    if checkpoint_path:
        model, _ = ModelLoader.load_checkpoint(checkpoint_path, device)
    else:
        # Create new model
        metadata = ModelMetadata(
            architecture=architecture,
            **game_config,
            model_params={'num_blocks': 20, 'num_filters': 256}
        )
        model = ModelRegistry.create(architecture, metadata)
        model.to(device)
    
    return model