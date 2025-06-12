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

from mcts.core.evaluator import Evaluator, EvaluatorConfig
from .nn_framework import ModelLoader, BaseGameModel, ModelMetadata
from .resnet_model import ResNetModel, create_resnet_for_game

logger = logging.getLogger(__name__)


class ResNetEvaluator(Evaluator):
    """ResNet-based neural network evaluator for MCTS"""
    
    def __init__(
        self,
        model: Optional[BaseGameModel] = None,
        config: Optional[EvaluatorConfig] = None,
        game_type: str = 'gomoku',
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize ResNet evaluator
        
        Args:
            model: Pre-loaded model (optional)
            config: Evaluator configuration
            game_type: Type of game if creating new model
            checkpoint_path: Path to model checkpoint
            device: Device to run on (auto-detect if None)
        """
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize model
        if model is not None:
            self.model = model
            action_size = model.metadata.num_actions
        elif checkpoint_path is not None:
            self.model, metadata = ModelLoader.load_checkpoint(checkpoint_path, device)
            action_size = metadata.num_actions
        else:
            # Create new model for game
            # Override input channels to use enhanced representation (20 channels)
            self.model = create_resnet_for_game(game_type, input_channels=20)
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
        
        # Mixed precision
        self.use_amp = config.use_fp16 and device == 'cuda'
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        logger.debug(f"ResNetEvaluator initialized on {device}")
        logger.info(f"Model parameters: {self.model.count_parameters():,}")
    
    def evaluate(
        self,
        state: np.ndarray,
        legal_mask: Optional[np.ndarray] = None,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """
        Evaluate a single position
        
        Args:
            state: Board state (channels, height, width)
            legal_mask: Boolean mask for legal actions
            temperature: Temperature for policy
            
        Returns:
            policy: Probability distribution over actions
            value: Position evaluation [-1, 1]
        """
        # Add batch dimension
        state_batch = np.expand_dims(state, axis=0)
        legal_mask_batch = np.expand_dims(legal_mask, axis=0) if legal_mask is not None else None
        
        # Evaluate batch
        policies, values = self.evaluate_batch(state_batch, legal_mask_batch, temperature)
        
        return policies[0], values[0]
    
    def evaluate_batch(
        self,
        states: Union[np.ndarray, torch.Tensor],
        legal_masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate a batch of positions
        
        Args:
            states: Batch of states (batch, channels, height, width) - numpy array or torch tensor
            legal_masks: Batch of legal masks (batch, num_actions) - numpy array or torch tensor
            temperature: Temperature for policy
            
        Returns:
            policies: Batch of policies (batch, num_actions)
            values: Batch of values (batch,)
        """
        batch_size = states.shape[0]
        
        # Convert to tensors
        if isinstance(states, np.ndarray):
            states_tensor = torch.from_numpy(states).float().to(self.device)
        else:
            # Already a tensor - ensure it's on the right device
            states_tensor = states.float().to(self.device)
        
        # Ensure correct shape
        if len(states_tensor.shape) == 3:
            states_tensor = states_tensor.unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    policy_logits, value_logits = self.model(states_tensor)
                    policy_logits = policy_logits.float()
                    value_logits = value_logits.float()
            else:
                policy_logits, value_logits = self.model(states_tensor)
        
        # Apply temperature
        if temperature != 1.0:
            policy_logits = policy_logits / temperature
        
        # Apply legal move masking
        if legal_masks is not None:
            if isinstance(legal_masks, np.ndarray):
                legal_masks_tensor = torch.from_numpy(legal_masks).bool().to(self.device)
            else:
                # Already a tensor
                legal_masks_tensor = legal_masks.bool().to(self.device)
            # Set illegal moves to very negative value
            policy_logits = policy_logits.masked_fill(~legal_masks_tensor, -1e9)
        
        # Softmax to get probabilities
        policies = F.softmax(policy_logits, dim=1)
        
        # Ensure value is in [-1, 1]
        values = torch.tanh(value_logits).squeeze(-1)
        
        # Convert to numpy (keep on GPU if requested)
        if hasattr(self, '_return_torch_tensors') and self._return_torch_tensors:
            # Return torch tensors for GPU-based MCTS
            self.eval_count += batch_size
            return policies, values
        else:
            # Convert to numpy for compatibility
            policies_np = policies.cpu().numpy()
            values_np = values.cpu().numpy()
            
            self.eval_count += batch_size
            
            return policies_np, values_np
    
    def save_checkpoint(self, path: str, additional_info: Optional[Dict[str, Any]] = None):
        """
        Save model checkpoint
        
        Args:
            path: Path to save checkpoint
            additional_info: Additional information to save
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.get_config(),
            'evaluator_config': {
                'device': str(self.device),
                'use_amp': self.use_amp,
                'action_size': self.action_size
            }
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save checkpoint
        torch.save(checkpoint, path)
        
        # Save metadata separately
        metadata_path = Path(path).parent / 'metadata.json'
        self.model.metadata.save(str(metadata_path))
        
        logger.info(f"Saved checkpoint to {path}")
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[EvaluatorConfig] = None,
        device: Optional[str] = None
    ) -> 'ResNetEvaluator':
        """
        Load evaluator from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
            config: Evaluator configuration
            device: Device to load on
            
        Returns:
            evaluator: Loaded evaluator
        """
        return cls(
            checkpoint_path=checkpoint_path,
            config=config,
            device=device
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        return {
            'eval_count': self.eval_count,
            'total_time': self.total_time,
            'avg_time_per_eval': self.total_time / max(1, self.eval_count),
            'model_params': self.model.count_parameters(),
            'device': str(self.device),
            'use_amp': self.use_amp
        }


def create_evaluator_for_game(
    game_type: str,
    checkpoint_path: Optional[str] = None,
    num_blocks: int = 20,
    num_filters: int = 256,
    device: Optional[str] = None,
    **kwargs
) -> ResNetEvaluator:
    """
    Create ResNet evaluator for specific game
    
    Args:
        game_type: Type of game ('chess', 'go', 'gomoku')
        checkpoint_path: Path to checkpoint (optional)
        num_blocks: Number of ResNet blocks
        num_filters: Number of filters
        device: Device to run on
        **kwargs: Additional configuration
        
    Returns:
        evaluator: ResNet evaluator instance
    """
    config = EvaluatorConfig(
        device=device or ('cuda' if torch.cuda.is_available() else 'cpu'),
        **kwargs
    )
    
    if checkpoint_path:
        return ResNetEvaluator.from_checkpoint(checkpoint_path, config, device)
    else:
        # Create new model
        model = create_resnet_for_game(
            game_type,
            num_blocks=num_blocks,
            num_filters=num_filters
        )
        return ResNetEvaluator(model=model, config=config)


# Convenience functions for common games
def create_chess_evaluator(**kwargs) -> ResNetEvaluator:
    """Create evaluator for chess"""
    return create_evaluator_for_game('chess', **kwargs)


def create_go_evaluator(**kwargs) -> ResNetEvaluator:
    """Create evaluator for Go"""
    return create_evaluator_for_game('go', **kwargs)


def create_gomoku_evaluator(**kwargs) -> ResNetEvaluator:
    """Create evaluator for Gomoku"""
    return create_evaluator_for_game('gomoku', **kwargs)