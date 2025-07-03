"""Neural Network Framework for AlphaZero

This module provides the base classes and utilities for neural network models
used in the AlphaZero MCTS implementation.
"""

import torch
import torch.nn as nn
import contextlib
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Autocast Utilities (moved from utils/autocast_utils.py for better organization)
# ============================================================================

def safe_autocast(device: Optional[Union[str, torch.device]] = None, 
                  enabled: bool = True,
                  dtype: Optional[torch.dtype] = None):
    """Context manager for safe autocast that avoids warnings when CUDA is not available
    
    Args:
        device: Device to use for autocast. If None, uses current default device.
                Can be 'cuda', 'cpu', or torch.device object.
        enabled: Whether to enable autocast
        dtype: Data type for autocast (e.g., torch.float16)
    
    Returns:
        Context manager that handles autocast appropriately for the device
    """
    if not enabled:
        return contextlib.nullcontext()
    
    # Convert device to torch.device if needed
    if device is None:
        device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    elif isinstance(device, str):
        device = torch.device(device)
    elif hasattr(device, 'type'):
        # It's already a torch.device
        pass
    else:
        device = torch.device('cpu')
    
    # Only use autocast for CUDA devices
    if device.type == 'cuda' and torch.cuda.is_available():
        return torch.amp.autocast(device_type='cuda', enabled=enabled, dtype=dtype)
    else:
        # For CPU or when CUDA is not available, return null context
        return contextlib.nullcontext()


def get_device_type(device: Optional[Union[str, torch.device]] = None) -> str:
    """Get the device type string suitable for autocast
    
    Args:
        device: Device specification
        
    Returns:
        'cuda' or 'cpu' based on device and availability
    """
    if device is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    elif isinstance(device, str):
        if device == 'cuda' and not torch.cuda.is_available():
            return 'cpu'
        return device
    elif hasattr(device, 'type'):
        if device.type == 'cuda' and not torch.cuda.is_available():
            return 'cpu'
        return device.type
    else:
        return 'cpu'


# ============================================================================
# Neural Network Framework Classes
# ============================================================================


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
    
    def save_checkpoint(self, path: Path, optimizer: Optional[torch.optim.Optimizer] = None, save_metadata: bool = True):
        """Save model checkpoint
        
        Args:
            path: Path to save checkpoint
            optimizer: Optional optimizer to save state
            save_metadata: Whether to include metadata in checkpoint (default: True)
        """
        if save_metadata and self.metadata:
            # Save as full checkpoint with metadata
            checkpoint = {
                'model_state_dict': self.state_dict(),
                'metadata': self.metadata.to_dict(),
            }
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            torch.save(checkpoint, path)
        else:
            # Save just the state dict (for compatibility with unified_training_pipeline)
            torch.save(self.state_dict(), path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Path, optimizer: Optional[torch.optim.Optimizer] = None):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        # Handle both checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Full checkpoint format
            self.load_state_dict(checkpoint['model_state_dict'])
            if 'metadata' in checkpoint:
                self.metadata = ModelMetadata.from_dict(checkpoint['metadata'])
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            # Pure state dict format
            self.load_state_dict(checkpoint)
            # Metadata should be set by the model class itself
            
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
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Check if it's a full checkpoint with model_state_dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                has_metadata = 'metadata' in checkpoint
            else:
                # It's a state dict wrapped in a dict (shouldn't happen, but handle it)
                state_dict = checkpoint
                checkpoint = {'model_state_dict': state_dict}
                has_metadata = False
        else:
            # It's a pure state dict (from torch.save(model.state_dict(), path))
            state_dict = checkpoint
            checkpoint = {'model_state_dict': state_dict}
            has_metadata = False
        
        # Handle checkpoints with and without metadata
        if has_metadata:
            metadata = ModelMetadata.from_dict(checkpoint['metadata'])
        else:
            # Try to infer metadata from the model structure
            logger.info(f"No metadata found in checkpoint {path}, inferring from model structure")
            
            # Default values
            game_type = 'gomoku'  # Default to gomoku if not specified
            board_size = 15  # Default board size
            
            # Infer input channels from first conv layer
            input_channels = 20  # Default
            for key in state_dict:
                if ('initial_conv' in key or 'conv_input' in key) and 'weight' in key:
                    input_channels = state_dict[key].shape[1]
                    break
            
            # Infer number of blocks by counting residual blocks
            num_blocks = 10  # Default
            block_count = sum(1 for key in state_dict if 'residual_blocks' in key and '.conv1.weight' in key)
            if block_count > 0:
                num_blocks = block_count
                
            # Infer number of filters from conv layers
            num_filters = 128  # Default
            for key in state_dict:
                if ('initial_conv' in key or 'conv_input' in key) and 'weight' in key:
                    num_filters = state_dict[key].shape[0]
                    break
            
            # Infer board size and num_actions from policy head
            num_actions = board_size * board_size
            for key in state_dict:
                if 'policy_head' in key and 'fc2' in key and 'weight' in key:
                    # Modern architecture has fc2 as final layer
                    num_actions = state_dict[key].shape[0]
                    # Try to infer board size from num_actions
                    import math
                    sqrt_actions = int(math.sqrt(num_actions))
                    if sqrt_actions * sqrt_actions == num_actions:
                        board_size = sqrt_actions
                    break
                elif 'policy_head' in key and 'fc' in key and 'weight' in key and 'fc2' not in key:
                    # Fallback for models that might use just 'fc'
                    num_actions = state_dict[key].shape[0]
                    import math
                    sqrt_actions = int(math.sqrt(num_actions))
                    if sqrt_actions * sqrt_actions == num_actions:
                        board_size = sqrt_actions
                    break
            
            metadata = ModelMetadata(
                game_type=game_type,
                board_size=board_size,
                num_actions=num_actions,
                input_channels=input_channels,
                num_blocks=num_blocks,
                num_filters=num_filters,
                version="1.0",
                training_steps=0,
                elo_rating=1200.0
            )
        
        # Create model based on metadata
        from .resnet_model import create_resnet_for_game
        
        # Pass all relevant parameters from metadata
        model = create_resnet_for_game(
            metadata.game_type,
            input_channels=metadata.input_channels,
            num_blocks=metadata.num_blocks,
            num_filters=metadata.num_filters
        )
        
        # Load weights
        model.load_state_dict(state_dict)
                
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with mixed precision"""
        device = x.device if x.is_cuda else 'cpu'
        with safe_autocast(device=device, enabled=True):
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