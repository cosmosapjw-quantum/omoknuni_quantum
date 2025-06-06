"""ResNet Model Implementation for MCTS

This module implements a ResNet-based neural network following the AlphaZero architecture
with policy and value heads for game evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from .nn_framework import BaseGameModel, ModelMetadata


@dataclass
class ResNetConfig:
    """Configuration for ResNet model"""
    num_blocks: int = 20
    num_filters: int = 256
    value_head_hidden_size: int = 256
    use_batch_norm: bool = True
    use_squeeze_excitation: bool = False
    se_ratio: int = 16
    activation: str = 'relu'
    dropout_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'num_blocks': self.num_blocks,
            'num_filters': self.num_filters,
            'value_head_hidden_size': self.value_head_hidden_size,
            'use_batch_norm': self.use_batch_norm,
            'use_squeeze_excitation': self.use_squeeze_excitation,
            'se_ratio': self.se_ratio,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate
        }


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    
    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Squeeze
        y = self.squeeze(x).view(b, c)
        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """Residual block with optional SE"""
    
    def __init__(self, channels: int, config: ResNetConfig):
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        
        self.use_batch_norm = config.use_batch_norm
        if config.use_batch_norm:
            self.bn1 = nn.BatchNorm2d(channels)
            self.bn2 = nn.BatchNorm2d(channels)
        
        # Activation
        if config.activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif config.activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif config.activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        else:
            raise ValueError(f"Unknown activation: {config.activation}")
        
        # Optional SE block
        self.se = SEBlock(channels, config.se_ratio) if config.use_squeeze_excitation else None
        
        # Optional dropout
        self.dropout = nn.Dropout2d(config.dropout_rate) if config.dropout_rate > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = self.activation(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.conv2(out)
        if self.use_batch_norm:
            out = self.bn2(out)
        
        # Squeeze-Excitation
        if self.se is not None:
            out = self.se(out)
        
        out += identity
        out = self.activation(out)
        
        return out


class PolicyHead(nn.Module):
    """Policy head for move probabilities"""
    
    def __init__(self, in_channels: int, board_size: int, num_actions: int, config: ResNetConfig):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, 32, 1, bias=False)
        
        if config.use_batch_norm:
            self.bn = nn.BatchNorm2d(32)
        else:
            self.bn = None
        
        if config.activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif config.activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        
        # Fully connected layer
        self.fc = nn.Linear(32 * board_size, num_actions)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.bn is not None:
            out = self.bn(out)
        out = self.activation(out)
        
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out


class ValueHead(nn.Module):
    """Value head for position evaluation"""
    
    def __init__(self, in_channels: int, board_size: int, hidden_size: int, config: ResNetConfig):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, 32, 1, bias=False)
        
        if config.use_batch_norm:
            self.bn = nn.BatchNorm2d(32)
        else:
            self.bn = None
        
        if config.activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif config.activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * board_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
        # Optional dropout
        self.dropout = nn.Dropout(config.dropout_rate) if config.dropout_rate > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.bn is not None:
            out = self.bn(out)
        out = self.activation(out)
        
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.activation(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.fc2(out)
        out = torch.tanh(out)
        
        return out


class ResNetModel(BaseGameModel):
    """ResNet model for game evaluation"""
    
    def __init__(self, metadata: ModelMetadata, config: Optional[ResNetConfig] = None):
        super().__init__(metadata)
        
        if config is None:
            # Create config from metadata if available
            if 'model_params' in metadata.__dict__ and metadata.model_params:
                config = ResNetConfig(**metadata.model_params)
            else:
                config = ResNetConfig()
        
        self.config = config
        
        # Initial convolution
        self.initial_conv = nn.Conv2d(
            metadata.input_channels,
            config.num_filters,
            3,
            padding=1,
            bias=False
        )
        
        if config.use_batch_norm:
            self.initial_bn = nn.BatchNorm2d(config.num_filters)
        else:
            self.initial_bn = None
        
        if config.activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif config.activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(config.num_filters, config)
            for _ in range(config.num_blocks)
        ])
        
        # Calculate board size
        board_size = metadata.board_height * metadata.board_width
        
        # Policy and value heads
        self.policy_head = PolicyHead(
            config.num_filters,
            board_size,
            metadata.num_actions,
            config
        )
        
        self.value_head = ValueHead(
            config.num_filters,
            board_size,
            config.value_head_hidden_size,
            config
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ResNet
        
        Args:
            x: Input tensor (batch, channels, height, width)
            
        Returns:
            policy_logits: (batch, num_actions)
            value: (batch, 1)
        """
        # Initial convolution
        out = self.initial_conv(x)
        if self.initial_bn is not None:
            out = self.initial_bn(out)
        out = self.activation(out)
        
        # Residual blocks
        for block in self.res_blocks:
            out = block(out)
        
        # Heads
        policy_logits = self.policy_head(out)
        value = self.value_head(out)
        
        return policy_logits, value
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def count_parameters(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            'architecture': 'resnet',
            'metadata': self.metadata.__dict__,
            'config': self.config.to_dict()
        }


def create_resnet_for_game(
    game_type: str,
    num_blocks: int = 20,
    num_filters: int = 256,
    **kwargs
) -> ResNetModel:
    """
    Create ResNet model for specific game
    
    Args:
        game_type: Type of game ('chess', 'go', 'gomoku')
        num_blocks: Number of residual blocks
        num_filters: Number of filters
        **kwargs: Additional config parameters
        
    Returns:
        model: ResNet model instance
    """
    # Game configurations - All games use 20 input channels
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
            'num_actions': 362
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
    
    # Create metadata
    metadata = ModelMetadata(
        architecture='resnet',
        **game_configs[game_type],
        model_params={
            'num_blocks': num_blocks,
            'num_filters': num_filters,
            **kwargs
        }
    )
    
    # Create config
    config = ResNetConfig(
        num_blocks=num_blocks,
        num_filters=num_filters,
        **kwargs
    )
    
    return ResNetModel(metadata, config)