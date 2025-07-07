"""ResNet-based Neural Network Evaluator

This module provides a ResNet-based evaluator that replaces the mock evaluator
with a real neural network for position evaluation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
from pathlib import Path
import logging
import json
import time

from mcts.core.evaluator import EvaluatorConfig
from .base_neural_evaluator import BaseNeuralEvaluator
from .nn_framework import ModelLoader, BaseGameModel, ModelMetadata
from .resnet_model import ResNetModel, create_resnet_for_game
from mcts.utils.config_system import AlphaZeroConfig, NeuralNetworkConfig

logger = logging.getLogger(__name__)


def load_config_for_model(checkpoint_path: str) -> Optional[NeuralNetworkConfig]:
    """
    Automatically detect and load neural network config for a model checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint file
        
    Returns:
        NeuralNetworkConfig if found, None otherwise
    """
    from pathlib import Path
    
    try:
        checkpoint_path = Path(checkpoint_path)
        
        # Look for config.yaml in experiment directory
        # Try different possible locations based on checkpoint path structure
        search_paths = []
        
        # If checkpoint is in experiments/experiment_name/best_models/model.pt
        if 'experiments' in checkpoint_path.parts:
            exp_idx = checkpoint_path.parts.index('experiments')
            if exp_idx + 1 < len(checkpoint_path.parts):
                exp_dir = Path(*checkpoint_path.parts[:exp_idx+2])  # experiments/experiment_name
                search_paths.append(exp_dir / "config.yaml")
        
        # If checkpoint is in experiment_name/best_models/model.pt
        search_paths.append(checkpoint_path.parent.parent / "config.yaml")
        
        # If checkpoint is in best_models/model.pt
        search_paths.append(checkpoint_path.parent / "config.yaml")
        
        # Look for gomoku_classical.yaml in configs directory (relative to project root)
        project_root = Path(__file__).parent.parent.parent.parent
        search_paths.append(project_root / "configs" / "gomoku_classical.yaml")
        
        for config_path in search_paths:
            if config_path.exists():
                try:
                    logger.info(f"Loading neural network config from {config_path}")
                    full_config = AlphaZeroConfig.load(str(config_path))
                    return full_config.network
                except Exception as config_error:
                    logger.warning(f"Failed to load config from {config_path}: {config_error}")
                    continue  # Try next config path
        
        logger.warning(f"No config file found for checkpoint {checkpoint_path}")
        return None
        
    except Exception as e:
        logger.warning(f"Failed to load config for {checkpoint_path}: {e}")
        return None


class ResNetEvaluator(BaseNeuralEvaluator):
    """ResNet-based neural network evaluator for MCTS"""
    
    def __init__(
        self,
        model: Optional[BaseGameModel] = None,
        config: Optional[EvaluatorConfig] = None,
        game_type: str = 'gomoku',
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        network_config: Optional[NeuralNetworkConfig] = None,
        input_channels: Optional[int] = None
    ):
        """
        Initialize ResNet evaluator
        
        Args:
            model: Pre-loaded model (optional)
            config: Evaluator configuration
            game_type: Type of game if creating new model
            checkpoint_path: Path to model checkpoint
            device: Device to run on (auto-detect if None)
            network_config: Neural network architecture config (optional)
            input_channels: Number of input channels (18 for basic, 20 for enhanced)
        """
        # Load or create model if not provided
        if model is None:
            if checkpoint_path is not None:
                # Try to automatically load network config for the checkpoint
                if network_config is None:
                    network_config = load_config_for_model(checkpoint_path)
                
                model, metadata = ModelLoader.load_checkpoint(checkpoint_path, device)
            else:
                # Create new model for game
                if network_config is not None:
                    # Use config values for architecture
                    model = create_resnet_for_game(
                        game_type=game_type,
                        input_channels=network_config.input_channels,
                        num_blocks=network_config.num_res_blocks,
                        num_filters=network_config.num_filters
                    )
                else:
                    # Use explicit input_channels if provided, otherwise default to 18 for basic representation
                    channels = input_channels if input_channels is not None else 18
                    model = create_resnet_for_game(game_type, input_channels=channels)
        
        # Initialize base class with model
        super().__init__(
            model=model,
            config=config,
            game_type=game_type,
            device=device
        )
        
        # ResNet-specific configuration
        self.use_amp = getattr(config, 'use_mixed_precision', True) if config else True
        self.eval_count = 0  # Legacy counter for compatibility
        
        # Cache statistics (if caching is implemented)
        self._cache_hits = 0
        self._cache_misses = 0
        self._batch_cache = {}
    
    def _forward_model(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through ResNet model"""
        from .nn_framework import safe_autocast
        
        # Update legacy counter
        self.eval_count += states.shape[0]
        
        # Forward pass with autocast for mixed precision
        with safe_autocast(device=self.device, enabled=self.use_amp):
            log_policies, values = self.model(states)
        
        # Convert log probabilities to probabilities
        policies = F.softmax(log_policies, dim=1)
        
        return policies, values.squeeze(-1)
    
    
    @torch.no_grad()
    def forward_batch(
        self,
        states_tensor: torch.Tensor,
        legal_mask_tensor: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GPU-native forward pass that keeps tensors on GPU
        
        This method is optimized for GPU service usage and avoids any CPU-GPU transfers.
        Used by GPUEvaluatorService for maximum performance.
        
        Args:
            states_tensor: Input states already on GPU (batch_size, channels, height, width)
            legal_mask_tensor: Optional legal move masks on GPU (batch_size, num_actions)
            temperature: Temperature for policy scaling
            
        Returns:
            Tuple of (policies, values) as GPU tensors
                - policies: Action probabilities (batch_size, num_actions)
                - values: Position evaluations (batch_size,)
        """
        # Track evaluations
        batch_size = states_tensor.shape[0]
        self.eval_count += batch_size
        
        # Ensure tensor is on correct device and dtype
        states_tensor = states_tensor.to(self.device)
        if self.use_amp and self.device.type == 'cuda':
            states_tensor = states_tensor.half()
        
        # Forward pass with autocast
        from .nn_framework import safe_autocast
        with safe_autocast(device=self.device, enabled=self.use_amp):
            log_policies, values = self.model(states_tensor)
        
        # Apply temperature BEFORE converting to probabilities (more efficient)
        if temperature != 1.0:
            log_policies = log_policies / temperature
        
        # Convert to probabilities using softmax (not just exp)
        policies = F.softmax(log_policies, dim=1)
        
        # Apply legal move masking if provided
        if legal_mask_tensor is not None:
            # Zero out illegal moves
            policies = policies * legal_mask_tensor
            
            # Renormalize
            policies_sum = policies.sum(dim=-1, keepdim=True)
            policies = policies / (policies_sum + 1e-8)
        
        # Squeeze value dimension
        values = values.squeeze(-1)
        
        # Return GPU tensors (no CPU transfer!)
        return policies, values
    
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ResNet evaluator statistics"""
        # Get base stats
        stats = super().get_stats()
        
        # Add ResNet-specific stats
        total_requests = self._cache_hits + self._cache_misses
        cache_hit_rate = self._cache_hits / max(1, total_requests) if total_requests > 0 else 0.0
        
        stats.update({
            'model_params': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            'use_amp': self.use_amp,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': cache_hit_rate,
            'eval_count_legacy': self.eval_count  # Legacy counter
        })
        
        return stats
    
    def reset_statistics(self):
        """Reset evaluation statistics"""
        super().reset_statistics()
        self.eval_count = 0  # Legacy counter
        self._cache_hits = 0
        self._cache_misses = 0
        self._batch_cache.clear()
    
    def save_checkpoint(self, path: str, additional_data: Optional[Dict[str, Any]] = None):
        """Save model checkpoint with evaluator config
        
        Args:
            path: Path to save checkpoint to
            additional_data: Optional dictionary of additional data to save
        """
        # Create checkpoint data
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.get_config() if hasattr(self.model, 'get_config') else {},
            'evaluator_config': {
                'device': str(self.device),
                'batch_size': self.batch_size,
                'use_amp': self.use_amp,
                'action_size': self.action_size
            }
        }
        
        # Add any additional data
        if additional_data:
            checkpoint.update(additional_data)
        
        # Save checkpoint
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Save metadata if model has it
        if hasattr(self.model, 'metadata') and self.model.metadata:
            metadata_path = checkpoint_path.parent / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(self.model.metadata.to_dict(), f, indent=2)
    
    def load_checkpoint(self, path: str, **kwargs):
        """Load model checkpoint"""
        self.model.load_checkpoint(Path(path), **kwargs)
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: Optional[str] = None, **kwargs):
        """Create evaluator from checkpoint
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run on (auto-detect if None)
            **kwargs: Additional arguments for ResNetEvaluator
            
        Returns:
            ResNetEvaluator loaded from checkpoint
        """
        return cls(checkpoint_path=checkpoint_path, device=device, **kwargs)


def create_evaluator_for_game(
    game_type: str,
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    num_blocks: Optional[int] = None,
    num_filters: Optional[int] = None,
    config_path: Optional[str] = None,
    input_channels: Optional[int] = None,
    **kwargs
) -> ResNetEvaluator:
    """
    Create a ResNet evaluator for a specific game
    
    Args:
        game_type: Type of game ('chess', 'go', 'gomoku')
        checkpoint_path: Optional path to model checkpoint
        device: Device to run on (auto-detect if None)
        num_blocks: Number of ResNet blocks (legacy, use config_path instead)
        num_filters: Number of filters in ResNet (legacy, use config_path instead)
        config_path: Path to YAML config file with network architecture
        **kwargs: Additional arguments for ResNetEvaluator
        
    Returns:
        ResNetEvaluator configured for the game
    """
    # Load network config if provided
    network_config = None
    if config_path is not None:
        try:
            full_config = AlphaZeroConfig.load(config_path)
            network_config = full_config.network
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
    
    # Legacy support: create network config from individual parameters
    if network_config is None and (num_blocks is not None or num_filters is not None):
        network_config = NeuralNetworkConfig(
            num_res_blocks=num_blocks or 10,
            num_filters=num_filters or 256,
            input_channels=18  # Default to 18 for basic representation
        )
    
    return ResNetEvaluator(
        game_type=game_type,
        checkpoint_path=checkpoint_path,
        device=device,
        network_config=network_config,
        input_channels=input_channels,
        **kwargs
    )


def create_chess_evaluator(num_blocks: int = 20, num_filters: int = 256, **kwargs) -> ResNetEvaluator:
    """Create evaluator for Chess"""
    return create_evaluator_for_game('chess', num_blocks=num_blocks, num_filters=num_filters, **kwargs)


def create_go_evaluator(num_blocks: int = 20, num_filters: int = 256, **kwargs) -> ResNetEvaluator:
    """Create evaluator for Go"""
    return create_evaluator_for_game('go', num_blocks=num_blocks, num_filters=num_filters, **kwargs)


def create_gomoku_evaluator(num_blocks: int = 20, num_filters: int = 256, **kwargs) -> ResNetEvaluator:
    """Create evaluator for Gomoku"""
    return create_evaluator_for_game('gomoku', num_blocks=num_blocks, num_filters=num_filters, **kwargs)