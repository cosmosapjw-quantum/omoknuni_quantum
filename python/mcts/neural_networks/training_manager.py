"""Training manager for AlphaZero neural network

This module handles:
- Neural network training loop
- Loss computation (policy + value)
- Learning rate scheduling
- Gradient clipping
- Mixed precision training
- Training metrics tracking
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from mcts.neural_networks.replay_buffer import ReplayBuffer
from mcts.utils.config_system import AlphaZeroConfig


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for neural network training"""
    batch_size: int = 256
    num_epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    lr_scheduler_type: str = "cosine"  # "cosine", "step", "none"
    lr_scheduler_step_size: int = 10
    lr_scheduler_gamma: float = 0.1
    warmup_steps: int = 100
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    use_mixed_precision: bool = True
    num_workers: int = 0  # DataLoader workers


@dataclass
class TrainingMetrics:
    """Metrics tracking for training"""
    total_loss: List[float] = field(default_factory=list)
    policy_loss: List[float] = field(default_factory=list)
    value_loss: List[float] = field(default_factory=list)
    learning_rate: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    
    def update(self, total_loss: float, policy_loss: float, value_loss: float,
               learning_rate: float = None):
        """Update metrics with new values"""
        self.total_loss.append(total_loss)
        self.policy_loss.append(policy_loss)
        self.value_loss.append(value_loss)
        if learning_rate is not None:
            self.learning_rate.append(learning_rate)
    
    def get_latest(self) -> Dict[str, Optional[float]]:
        """Get latest metric values"""
        return {
            'total_loss': self.total_loss[-1] if self.total_loss else None,
            'policy_loss': self.policy_loss[-1] if self.policy_loss else None,
            'value_loss': self.value_loss[-1] if self.value_loss else None,
            'learning_rate': self.learning_rate[-1] if self.learning_rate else None
        }
    
    def get_averages(self, last_n: int = None) -> Dict[str, float]:
        """Get average metrics over last n updates"""
        if last_n is None:
            last_n = len(self.total_loss)
        
        return {
            'total_loss': np.mean(self.total_loss[-last_n:]) if self.total_loss else 0,
            'policy_loss': np.mean(self.policy_loss[-last_n:]) if self.policy_loss else 0,
            'value_loss': np.mean(self.value_loss[-last_n:]) if self.value_loss else 0,
            'learning_rate': np.mean(self.learning_rate[-last_n:]) if self.learning_rate else 0
        }


class TrainingManager:
    """Manages neural network training"""
    
    def __init__(self, config: AlphaZeroConfig, model: nn.Module):
        """Initialize training manager
        
        Args:
            config: AlphaZero configuration
            model: Neural network model to train
        """
        self.config = config
        self.model = model
        self.device = torch.device(config.mcts.device)
        
        # Create training config from AlphaZero config
        self.training_config = TrainingConfig(
            batch_size=config.training.batch_size,
            num_epochs=config.training.num_epochs,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            gradient_clip_norm=getattr(config.training, 'gradient_clip_norm', 1.0),
            lr_scheduler_type=getattr(config.training, 'lr_scheduler_type', 'cosine'),
            lr_scheduler_step_size=getattr(config.training, 'lr_scheduler_step_size', 10),
            lr_scheduler_gamma=getattr(config.training, 'lr_scheduler_gamma', 0.1),
            warmup_steps=getattr(config.training, 'warmup_steps', 100),
            policy_loss_weight=getattr(config.training, 'policy_loss_weight', 1.0),
            value_loss_weight=getattr(config.training, 'value_loss_weight', 1.0),
            use_mixed_precision=getattr(config.training, 'mixed_precision', True)
        )
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize loss functions
        self.policy_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.value_loss_fn = nn.MSELoss()
        
        # Mixed precision setup
        self.use_mixed_precision = (
            self.training_config.use_mixed_precision and 
            self.device.type == 'cuda' and
            torch.cuda.is_available()
        )
        if self.use_mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda')
        
        # Metrics tracking
        self.metrics = TrainingMetrics()
        self.global_step = 0
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        return optim.Adam(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        if self.training_config.lr_scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_config.num_epochs * 100,  # Approximate steps
                eta_min=self.training_config.learning_rate * 0.01
            )
        elif self.training_config.lr_scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.training_config.lr_scheduler_step_size,
                gamma=self.training_config.lr_scheduler_gamma
            )
        else:
            return None
    
    def compute_loss(self, states: torch.Tensor, target_policies: torch.Tensor,
                    target_values: torch.Tensor) -> tuple:
        """Compute training loss
        
        Args:
            states: Batch of game states
            target_policies: Target policy distributions
            target_values: Target values
            
        Returns:
            Tuple of (total_loss, policy_loss, value_loss)
        """
        # Forward pass
        pred_policies, pred_values = self.model(states)
        pred_values = pred_values.squeeze()
        
        # Ensure target_values is 1D
        if target_values.dim() > 1:
            target_values = target_values.squeeze()
        
        # Calculate losses
        # Policy loss: KL divergence between predicted and target policies
        pred_log_policies = torch.log_softmax(pred_policies, dim=1)
        policy_loss = self.policy_loss_fn(pred_log_policies, target_policies)
        
        # Value loss: MSE between predicted and target values
        value_loss = self.value_loss_fn(pred_values, target_values)
        
        # Total loss with weights
        total_loss = (
            self.training_config.policy_loss_weight * policy_loss +
            self.training_config.value_loss_weight * value_loss
        )
        
        return total_loss, policy_loss, value_loss
    
    def _clip_gradients(self):
        """Apply gradient clipping"""
        if self.training_config.gradient_clip_norm > 0:
            clip_grad_norm_(
                self.model.parameters(),
                self.training_config.gradient_clip_norm
            )
    
    def train_epoch(self, replay_buffer: ReplayBuffer) -> Dict[str, float]:
        """Train for one epoch
        
        Args:
            replay_buffer: Replay buffer with training examples
            
        Returns:
            Dictionary with epoch metrics
        """
        # Create data loader
        dataloader = DataLoader(
            replay_buffer,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=self.training_config.num_workers,
            pin_memory=True
        )
        
        # Training stats for this epoch
        epoch_total_loss = []
        epoch_policy_loss = []
        epoch_value_loss = []
        
        # Set model to training mode
        self.model.train()
        
        # Train on batches
        for batch_idx, (states, target_policies, target_values) in enumerate(dataloader):
            # Move to device
            states = states.to(self.device)
            target_policies = target_policies.to(self.device)
            target_values = target_values.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.use_mixed_precision:
                with torch.amp.autocast('cuda'):
                    total_loss, policy_loss, value_loss = self.compute_loss(
                        states, target_policies, target_values
                    )
                
                # Backward pass with scaling
                self.scaler.scale(total_loss).backward()
                
                # Unscale before clipping
                self.scaler.unscale_(self.optimizer)
                self._clip_gradients()
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular training
                total_loss, policy_loss, value_loss = self.compute_loss(
                    states, target_policies, target_values
                )
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                self._clip_gradients()
                
                # Optimizer step
                self.optimizer.step()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Track losses
            epoch_total_loss.append(total_loss.item())
            epoch_policy_loss.append(policy_loss.item())
            epoch_value_loss.append(value_loss.item())
            
            self.global_step += 1
        
        # Calculate epoch averages
        avg_total_loss = np.mean(epoch_total_loss)
        avg_policy_loss = np.mean(epoch_policy_loss)
        avg_value_loss = np.mean(epoch_value_loss)
        
        # Update metrics
        current_lr = self.optimizer.param_groups[0]['lr']
        self.metrics.update(
            total_loss=avg_total_loss,
            policy_loss=avg_policy_loss,
            value_loss=avg_value_loss,
            learning_rate=current_lr
        )
        
        return {
            'avg_total_loss': avg_total_loss,
            'avg_policy_loss': avg_policy_loss,
            'avg_value_loss': avg_value_loss,
            'learning_rate': current_lr,
            'num_batches': len(dataloader)
        }
    
    def train(self, replay_buffer: ReplayBuffer) -> Dict[str, Any]:
        """Train the model for multiple epochs
        
        Args:
            replay_buffer: Replay buffer with training examples
            
        Returns:
            Dictionary with training results
        """
        if len(replay_buffer) < self.training_config.batch_size:
            logger.warning(f"Not enough examples in replay buffer "
                         f"({len(replay_buffer)} < {self.training_config.batch_size})")
            return {
                'num_epochs': 0,
                'final_loss': 0,
                'training_time': 0
            }
        
        # Ensure model is on correct device
        self.model = self.model.to(self.device)
        
        logger.info(f"Starting training for {self.training_config.num_epochs} epochs")
        start_time = time.time()
        
        # Train for specified epochs
        for epoch in range(self.training_config.num_epochs):
            epoch_start = time.time()
            
            # Train one epoch
            epoch_metrics = self.train_epoch(replay_buffer)
            
            epoch_time = time.time() - epoch_start
            self.metrics.epoch_times.append(epoch_time)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{self.training_config.num_epochs}: "
                       f"loss={epoch_metrics['avg_total_loss']:.4f}, "
                       f"policy_loss={epoch_metrics['avg_policy_loss']:.4f}, "
                       f"value_loss={epoch_metrics['avg_value_loss']:.4f}, "
                       f"lr={epoch_metrics['learning_rate']:.6f}, "
                       f"time={epoch_time:.1f}s")
        
        training_time = time.time() - start_time
        
        # Get final metrics
        final_metrics = self.metrics.get_averages(last_n=len(replay_buffer) // self.training_config.batch_size)
        
        return {
            'num_epochs': self.training_config.num_epochs,
            'final_loss': final_metrics['total_loss'],
            'final_policy_loss': final_metrics['policy_loss'],
            'final_value_loss': final_metrics['value_loss'],
            'training_time': training_time,
            'examples_trained': len(replay_buffer),
            'global_step': self.global_step
        }