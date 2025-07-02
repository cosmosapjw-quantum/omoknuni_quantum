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

from mcts.core.evaluator import Evaluator, EvaluatorConfig
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


class ResNetEvaluator(Evaluator):
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
        # Determine device from config or parameter
        if device is not None:
            # Explicit device specified
            pass
        elif config is not None and hasattr(config, 'device'):
            device = config.device
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize model
        if model is not None:
            self.model = model
            action_size = model.metadata.num_actions
        elif checkpoint_path is not None:
            # Try to automatically load network config for the checkpoint
            if network_config is None:
                network_config = load_config_for_model(checkpoint_path)
            
            self.model, metadata = ModelLoader.load_checkpoint(checkpoint_path, device)
            action_size = metadata.num_actions
        else:
            # Create new model for game
            # Use network_config if provided, otherwise use function defaults
            if network_config is not None:
                # Use config values for architecture
                self.model = create_resnet_for_game(
                    game_type=game_type,
                    input_channels=network_config.input_channels,
                    num_blocks=network_config.num_res_blocks,
                    num_filters=network_config.num_filters
                )
            else:
                # Use explicit input_channels if provided, otherwise default to 18 for basic representation
                channels = input_channels if input_channels is not None else 18
                self.model = create_resnet_for_game(game_type, input_channels=channels)
            action_size = self.model.metadata.num_actions
        
        # Move model to device
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize base class
        if config is None:
            config = EvaluatorConfig(device=device)
        super().__init__(config, action_size)
        
        # Performance tracking
        self.eval_count = 0
        self.total_time = 0.0
        
        # Cache for batch evaluation
        self._batch_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Mixed precision support
        self.use_mixed_precision = (hasattr(config, 'use_mixed_precision') and config.use_mixed_precision) or \
                                   (hasattr(config, 'use_fp16') and config.use_fp16)
        
        # Only enable AMP on CUDA devices
        if self.use_mixed_precision and self.device.type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
            self.use_amp = True
        else:
            self.use_amp = False  # Disable AMP on CPU
        
        # Store additional attributes for compatibility
        self.batch_size = config.batch_size if hasattr(config, 'batch_size') else 512
        self._return_torch_tensors = False  # Default to numpy arrays for compatibility
        
        logger.debug(f"Initialized ResNetEvaluator for {game_type} on {device}")
        logger.debug(f"Model has {sum(p.numel() for p in self.model.parameters())} parameters")
    
    @torch.no_grad()
    def evaluate_batch(
        self,
        states: Union[np.ndarray, torch.Tensor],
        legal_moves: Optional[Union[np.ndarray, torch.Tensor]] = None,
        temperature: float = 1.0
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Evaluate a batch of game states
        
        Args:
            states: Batch of game states (batch_size, channels, height, width)
            legal_moves: Optional legal move masks
            temperature: Temperature for policy scaling
            
        Returns:
            Tuple of (policies, values)
                - policies: Action probabilities (batch_size, num_actions)
                - values: Position evaluations (batch_size,)
        """
        start_time = time.time()
        
        # Track evaluations
        batch_size = states.shape[0] if states.ndim == 4 else 1
        self.eval_count += batch_size
        
        # Handle empty batch
        if batch_size == 0:
            empty_policies = np.empty((0, self.action_size), dtype=np.float32)
            empty_values = np.empty(0, dtype=np.float32)
            if isinstance(states, torch.Tensor) and self._return_torch_tensors:
                return torch.from_numpy(empty_policies), torch.from_numpy(empty_values)
            return empty_policies, empty_values
        
        # Convert to tensor if needed
        if isinstance(states, np.ndarray):
            states_tensor = torch.from_numpy(states).float().to(self.device)
            return_numpy = True
        else:
            states_tensor = states.to(self.device)
            return_numpy = False
        
        # Ensure correct shape
        if states_tensor.dim() == 3:
            states_tensor = states_tensor.unsqueeze(0)
        
        # Forward pass
        from mcts.utils.autocast_utils import safe_autocast
        with safe_autocast(device=self.device, enabled=self.use_amp):
            log_policies, values = self.model(states_tensor)
        
        # Convert log probabilities to probabilities
        policies = torch.exp(log_policies)
        
        # Apply legal move masking if provided
        if legal_moves is not None:
            if isinstance(legal_moves, np.ndarray):
                legal_moves = torch.from_numpy(legal_moves).to(self.device)
            
            # Zero out illegal moves
            policies = policies * legal_moves
            
            # Renormalize
            policies_sum = policies.sum(dim=-1, keepdim=True)
            policies = policies / (policies_sum + 1e-8)
        
        # Apply temperature if requested
        if temperature != 1.0:
            # Apply temperature to log probabilities before converting to probabilities
            log_policies = log_policies / temperature
            policies = torch.exp(log_policies)
            
            # Re-normalize if legal moves were applied
            if legal_moves is not None:
                policies = policies * legal_moves
                policies_sum = policies.sum(dim=-1, keepdim=True)
                policies = policies / (policies_sum + 1e-8)
        
        # Squeeze value dimension
        values = values.squeeze(-1)
        
        # Return in requested format
        if return_numpy or not self._return_torch_tensors:
            policies = policies.cpu().numpy()
            values = values.cpu().numpy()
        
        # Update timing
        self.total_time += time.time() - start_time
        
        return policies, values
    
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
        
        # Forward pass with autocast
        from mcts.utils.autocast_utils import safe_autocast
        with safe_autocast(device=self.device, enabled=self.use_amp):
            log_policies, values = self.model(states_tensor)
        
        # Apply temperature BEFORE converting to probabilities (more efficient)
        if temperature != 1.0:
            log_policies = log_policies / temperature
        
        # Convert to probabilities
        policies = torch.exp(log_policies)
        
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
    
    def evaluate(
        self,
        state: Union[np.ndarray, torch.Tensor],
        legal_moves: Optional[Union[np.ndarray, torch.Tensor]] = None,
        temperature: float = 1.0
    ) -> Tuple[Union[np.ndarray, torch.Tensor], float]:
        """
        Evaluate a single game state
        
        Args:
            state: Game state (channels, height, width)
            legal_moves: Optional legal move mask
            temperature: Temperature for policy scaling
            
        Returns:
            Tuple of (policy, value)
        """
        # Add batch dimension
        if isinstance(state, np.ndarray):
            state = np.expand_dims(state, 0)
        else:
            state = state.unsqueeze(0)
        
        if legal_moves is not None:
            if isinstance(legal_moves, np.ndarray):
                legal_moves = np.expand_dims(legal_moves, 0)
            else:
                legal_moves = legal_moves.unsqueeze(0)
        
        # Batch evaluation
        policies, values = self.evaluate_batch(state, legal_moves, temperature)
        
        # Extract single result
        if isinstance(policies, np.ndarray):
            return policies[0], float(values[0])
        else:
            return policies[0], values[0].item()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get evaluator statistics"""
        total_requests = self._cache_hits + self._cache_misses
        cache_hit_rate = self._cache_hits / max(1, total_requests)
        avg_time_per_eval = self.total_time / max(1, self.eval_count)
        
        return {
            'eval_count': self.eval_count,
            'total_time': self.total_time,
            'avg_time_per_eval': avg_time_per_eval,
            'model_params': sum(p.numel() for p in self.model.parameters()),
            'device': str(self.device.type),
            'use_amp': self.use_amp,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': cache_hit_rate,
            'batch_size': self.batch_size
        }
    
    def reset_statistics(self):
        """Reset evaluation statistics"""
        self.eval_count = 0
        self.total_time = 0.0
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