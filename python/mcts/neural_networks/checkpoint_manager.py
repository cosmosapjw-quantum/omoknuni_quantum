"""Checkpoint management for AlphaZero training

This module handles saving and loading checkpoints, including:
- Model state
- Optimizer state  
- Replay buffer
- Training metadata
- Best model tracking
- Cleanup of old checkpoints
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any
import shutil

import torch
import torch.nn as nn
import torch.optim as optim

from mcts.neural_networks.replay_buffer import ReplayBuffer
from mcts.utils.config_system import AlphaZeroConfig


logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoints and model persistence"""
    
    def __init__(self, checkpoint_dir: str, config: AlphaZeroConfig):
        """Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            config: AlphaZero configuration
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract relevant config
        self.checkpoint_every_n_iterations = config.training.checkpoint_interval
        self.keep_last_n_checkpoints = getattr(config.training, 'checkpoint_keep_last', 5)
        self.keep_last_n_replay_buffers = getattr(config.training, 'checkpoint_keep_last', 5)
    
    def save_checkpoint(
        self,
        iteration: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        replay_buffer: ReplayBuffer,
        metadata: Dict[str, Any]
    ) -> Path:
        """Save a checkpoint
        
        Args:
            iteration: Current iteration number
            model: Model to save
            optimizer: Optimizer to save
            replay_buffer: Replay buffer to save
            metadata: Additional metadata to save
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{iteration}.pt"
        
        # Prepare checkpoint data
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metadata': metadata  # Store metadata in its own field
        }
        
        # Also include scheduler and scaler if they exist in metadata
        if 'scheduler_state_dict' in metadata:
            checkpoint['scheduler_state_dict'] = metadata.pop('scheduler_state_dict')
        if 'scaler_state_dict' in metadata:
            checkpoint['scaler_state_dict'] = metadata.pop('scaler_state_dict')
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save replay buffer separately
        replay_buffer_path = self.checkpoint_dir / f"replay_buffer_{iteration}.pkl"
        replay_buffer.save(str(replay_buffer_path))
        logger.info(f"Saved replay buffer to {replay_buffer_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load a checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary containing checkpoint data
        """
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint
    
    def load_replay_buffer(self, iteration: int) -> ReplayBuffer:
        """Load replay buffer for a specific iteration
        
        Args:
            iteration: Iteration number
            
        Returns:
            Loaded replay buffer
        """
        replay_buffer_path = self.checkpoint_dir / f"replay_buffer_{iteration}.pkl"
        replay_buffer = ReplayBuffer()
        replay_buffer.load(str(replay_buffer_path))
        logger.info(f"Loaded replay buffer from {replay_buffer_path}")
        return replay_buffer
    
    def cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones"""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: int(p.stem.split('_')[1])
        )
        
        if len(checkpoints) > self.keep_last_n_checkpoints:
            to_remove = checkpoints[:-self.keep_last_n_checkpoints]
            for checkpoint in to_remove:
                checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint}")
    
    def cleanup_old_replay_buffers(self):
        """Remove old replay buffers, keeping only the most recent ones"""
        replay_buffers = sorted(
            self.checkpoint_dir.glob("replay_buffer_*.pkl"),
            key=lambda p: int(p.stem.split('_')[2])
        )
        
        if len(replay_buffers) > self.keep_last_n_replay_buffers:
            to_remove = replay_buffers[:-self.keep_last_n_replay_buffers]
            for buffer in to_remove:
                buffer.unlink()
                logger.info(f"Removed old replay buffer: {buffer}")
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the path to the latest checkpoint
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None
        
        # Sort by iteration number
        latest = max(
            checkpoints,
            key=lambda p: int(p.stem.split('_')[1])
        )
        return latest
    
    def save_best_model(
        self,
        model: nn.Module,
        iteration: int,
        elo_rating: float
    ):
        """Save the best model
        
        Args:
            model: Model to save
            iteration: Iteration number
            elo_rating: ELO rating of the model
        """
        best_model_path = self.checkpoint_dir / "best_model.pt"
        
        # Save model state
        torch.save({
            'model_state_dict': model.state_dict(),
            'iteration': iteration,
            'elo_rating': elo_rating
        }, best_model_path)
        
        # Save metadata separately for easy access
        metadata_path = self.checkpoint_dir / "best_model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'iteration': iteration,
                'elo_rating': elo_rating,
                'timestamp': str(Path(best_model_path).stat().st_mtime)
            }, f, indent=2)
        
        logger.info(f"Saved best model from iteration {iteration} with ELO {elo_rating}")