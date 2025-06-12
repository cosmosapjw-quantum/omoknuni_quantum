"""Neural network model for AlphaZero MCTS

This module implements the ResNet-based neural network architecture with
policy and value heads for game evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for neural network model
    
    Attributes:
        input_channels: Number of input channels (game-specific)
        input_height: Board height
        input_width: Board width
        num_actions: Number of possible actions
        num_res_blocks: Number of residual blocks
        num_filters: Number of filters in conv layers
        value_head_hidden_size: Hidden layer size for value head
        use_batch_norm: Whether to use batch normalization
        activation: Activation function ('relu' or 'elu')
        dropout_rate: Dropout rate (0.0 to disable)
    """
    input_channels: int = 20  # Standard AlphaZero encoding
    input_height: int = 8
    input_width: int = 8
    num_actions: int = 4096  # Default for chess
    num_res_blocks: int = 19
    num_filters: int = 256
    value_head_hidden_size: int = 256
    use_batch_norm: bool = True
    activation: str = 'relu'
    dropout_rate: float = 0.0
    

class ResidualBlock(nn.Module):
    """Residual block for ResNet architecture"""
    
    def __init__(self, num_filters: int, use_batch_norm: bool = True, 
                 activation: str = 'relu', dropout_rate: float = 0.0):
        super().__init__()
        
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(num_filters)
            self.bn2 = nn.BatchNorm2d(num_filters)
            
        self.activation = nn.ReLU() if activation == 'relu' else nn.ELU()
        
        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(dropout_rate)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block"""
        residual = x
        
        out = self.conv1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = self.activation(out)
        
        if self.dropout_rate > 0:
            out = self.dropout(out)
            
        out = self.conv2(out)
        if self.use_batch_norm:
            out = self.bn2(out)
            
        out += residual
        out = self.activation(out)
        
        return out


class PolicyHead(nn.Module):
    """Policy head for action probability output"""
    
    def __init__(self, num_filters: int, board_size: int, num_actions: int,
                 use_batch_norm: bool = True, activation: str = 'relu'):
        super().__init__()
        
        self.conv = nn.Conv2d(num_filters, 32, 1)  # 1x1 conv
        
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn = nn.BatchNorm2d(32)
            
        self.activation = nn.ReLU() if activation == 'relu' else nn.ELU()
        
        # Fully connected layer
        self.fc = nn.Linear(32 * board_size, num_actions)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through policy head
        
        Returns:
            Policy logits (before softmax)
        """
        out = self.conv(x)
        if self.use_batch_norm:
            out = self.bn(out)
        out = self.activation(out)
        
        # Flatten
        out = out.view(out.size(0), -1)
        
        # Policy logits
        policy_logits = self.fc(out)
        
        return policy_logits


class ValueHead(nn.Module):
    """Value head for position evaluation"""
    
    def __init__(self, num_filters: int, board_size: int, 
                 hidden_size: int = 256, use_batch_norm: bool = True,
                 activation: str = 'relu'):
        super().__init__()
        
        self.conv = nn.Conv2d(num_filters, 1, 1)  # 1x1 conv
        
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn = nn.BatchNorm2d(1)
            
        self.activation = nn.ReLU() if activation == 'relu' else nn.ELU()
        
        # Fully connected layers
        self.fc1 = nn.Linear(board_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through value head
        
        Returns:
            Value output (tanh activated, range [-1, 1])
        """
        out = self.conv(x)
        if self.use_batch_norm:
            out = self.bn(out)
        out = self.activation(out)
        
        # Flatten
        out = out.view(out.size(0), -1)
        
        # Hidden layer
        out = self.fc1(out)
        out = self.activation(out)
        
        # Value output
        value = torch.tanh(self.fc2(out))
        
        return value


class AlphaZeroNetwork(nn.Module):
    """Main AlphaZero neural network with ResNet backbone"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # Store for multiprocessing
        self._init_kwargs = {'config': config}
        
        # Initial convolution
        self.initial_conv = nn.Conv2d(
            config.input_channels, 
            config.num_filters, 
            3, 
            padding=1
        )
        
        if config.use_batch_norm:
            self.initial_bn = nn.BatchNorm2d(config.num_filters)
            
        self.activation = nn.ReLU() if config.activation == 'relu' else nn.ELU()
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                config.num_filters, 
                config.use_batch_norm,
                config.activation,
                config.dropout_rate
            )
            for _ in range(config.num_res_blocks)
        ])
        
        # Board size for heads
        board_size = config.input_height * config.input_width
        
        # Policy and value heads
        self.policy_head = PolicyHead(
            config.num_filters,
            board_size,
            config.num_actions,
            config.use_batch_norm,
            config.activation
        )
        
        self.value_head = ValueHead(
            config.num_filters,
            board_size,
            config.value_head_hidden_size,
            config.use_batch_norm,
            config.activation
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through network
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Tuple of (policy_logits, value)
            - policy_logits: (batch, num_actions)
            - value: (batch, 1)
        """
        # Initial convolution
        out = self.initial_conv(x)
        if self.config.use_batch_norm:
            out = self.initial_bn(out)
        out = self.activation(out)
        
        # Residual blocks
        for block in self.res_blocks:
            out = block(out)
            
        # Split to policy and value heads
        policy_logits = self.policy_head(out)
        value = self.value_head(out)
        
        return policy_logits, value
        
    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, float]:
        """Make prediction for a single position
        
        Args:
            x: Input tensor (can be numpy array)
            
        Returns:
            Tuple of (policy_probs, value)
        """
        # Convert to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
            
        # Add batch dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        # Move to same device as model
        x = x.to(next(self.parameters()).device)
        
        # Forward pass
        with torch.no_grad():
            policy_logits, value = self.forward(x)
            
        # Convert policy to probabilities
        policy_probs = F.softmax(policy_logits, dim=1)
        
        # Convert to numpy
        policy_probs = policy_probs.cpu().numpy()[0]
        value = value.cpu().numpy()[0, 0]
        
        # Ensure value is a Python float
        value = float(value)
        
        return policy_probs, value
        
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
        
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(game_type: str = 'chess', **kwargs) -> AlphaZeroNetwork:
    """Create model with game-specific configuration
    
    Args:
        game_type: Type of game ('chess', 'go', 'gomoku')
        **kwargs: Additional config parameters
        
    Returns:
        AlphaZeroNetwork instance
    """
    # Default configurations for different games
    configs = {
        'chess': ModelConfig(
            input_channels=17,  # 6 piece types * 2 colors + 5 auxiliary
            input_height=8,
            input_width=8,
            num_actions=4096,  # All possible moves
            num_res_blocks=19,
            num_filters=256
        ),
        'go': ModelConfig(
            input_channels=17,  # Stone positions + history
            input_height=19,
            input_width=19,
            num_actions=362,  # 19*19 + pass
            num_res_blocks=19,
            num_filters=256
        ),
        'gomoku': ModelConfig(
            input_channels=4,  # Current + last move
            input_height=15,
            input_width=15,
            num_actions=225,  # 15*15
            num_res_blocks=10,
            num_filters=128
        )
    }
    
    # Get base config
    if game_type in configs:
        config = configs[game_type]
    else:
        config = ModelConfig()
        
    # Override with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
            
    return AlphaZeroNetwork(config)


class EnsembleNetwork(nn.Module):
    """Ensemble of multiple AlphaZero networks for uncertainty estimation"""
    
    def __init__(self, configs: list[ModelConfig], weights: Optional[list[float]] = None):
        super().__init__()
        
        self.models = nn.ModuleList([
            AlphaZeroNetwork(config) for config in configs
        ])
        
        self.weights = weights if weights else [1.0 / len(configs)] * len(configs)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through ensemble
        
        Returns:
            Tuple of (mean_policy, mean_value, statistics)
        """
        all_policies = []
        all_values = []
        
        for model in self.models:
            policy, value = model(x)
            all_policies.append(policy)
            all_values.append(value)
            
        # Stack predictions
        policies = torch.stack(all_policies, dim=0)  # (n_models, batch, actions)
        values = torch.stack(all_values, dim=0)  # (n_models, batch, 1)
        
        # Weighted average
        weights = torch.tensor(self.weights, device=x.device).view(-1, 1, 1)
        mean_policy = (policies * weights).sum(dim=0)
        mean_value = (values * weights).sum(dim=0)
        
        # Compute statistics
        if len(self.models) > 1:
            # Standard deviation only makes sense with multiple models
            policy_std = policies.std(dim=0)
            value_std = values.std(dim=0)
        else:
            # For single model, std is zero
            policy_std = torch.zeros_like(mean_policy)
            value_std = torch.zeros_like(mean_value)
            
        stats = {
            'policy_std': policy_std,
            'value_std': value_std,
            'policy_entropy': -(F.softmax(mean_policy, dim=1) * F.log_softmax(mean_policy, dim=1)).sum(dim=1)
        }
        
        return mean_policy, mean_value, stats