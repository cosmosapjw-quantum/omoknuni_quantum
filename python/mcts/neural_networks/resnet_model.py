"""ResNet Model for AlphaZero

This module implements a ResNet-based neural network architecture for AlphaZero,
supporting Chess, Go, and Gomoku games.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass

from .nn_framework import BaseGameModel, ModelMetadata


@dataclass
class ResNetConfig:
    """Configuration for ResNet model"""
    num_blocks: int = 19  # Number of residual blocks
    num_filters: int = 256  # Number of filters in conv layers
    input_channels: int = 20  # Enhanced feature channels
    fc_value_hidden: int = 256  # Hidden units in value head
    fc_policy_hidden: int = 256  # Hidden units in policy head


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers"""
    
    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block"""
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = F.relu(out, inplace=True)
        
        return out


class PolicyHead(nn.Module):
    """Policy head for action probability output"""
    
    def __init__(self, num_filters: int, board_size: int, num_actions: int, fc_hidden: int = 256):
        super().__init__()
        self.board_size = board_size
        self.num_actions = num_actions
        
        # Convolutional layer
        self.conv = nn.Conv2d(num_filters, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(2 * board_size * board_size, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_actions)
        
        # Custom initialization for fc2 to prevent gradient explosion
        self._initialize_policy_weights()
    
    def _initialize_policy_weights(self):
        """Initialize policy head weights to prevent gradient explosion"""
        # Standard initialization for fc1
        nn.init.kaiming_normal_(self.fc1.weight)
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)
        
        # Special small initialization for fc2 to prevent extreme log probabilities
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.005)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through policy head"""
        # Convolutional processing
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        
        # Log softmax for numerical stability during training
        return F.log_softmax(x, dim=1)


class ValueHead(nn.Module):
    """Value head for position evaluation"""
    
    def __init__(self, num_filters: int, board_size: int, fc_hidden: int = 256):
        super().__init__()
        self.board_size = board_size
        
        # Convolutional layer
        self.conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(board_size * board_size, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through value head"""
        # Convolutional processing
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        
        # Tanh activation for value in [-1, 1]
        return torch.tanh(x)


class ResNetModel(BaseGameModel):
    """ResNet model for AlphaZero"""
    
    def __init__(self, config: ResNetConfig, board_size: int, num_actions: int, game_type: str):
        super().__init__()
        
        # Store configuration
        self.config = config
        self.board_size = board_size
        self.num_actions = num_actions
        
        # Initial convolutional layer
        self.conv_input = nn.Conv2d(
            config.input_channels,
            config.num_filters,
            3,
            padding=1,
            bias=False
        )
        self.bn_input = nn.BatchNorm2d(config.num_filters)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(config.num_filters)
            for _ in range(config.num_blocks)
        ])
        
        # Policy and value heads
        self.policy_head = PolicyHead(
            config.num_filters,
            board_size,
            num_actions,
            config.fc_policy_hidden
        )
        self.value_head = ValueHead(
            config.num_filters,
            board_size,
            config.fc_value_hidden
        )
        
        # Set metadata
        self.metadata = ModelMetadata(
            game_type=game_type,
            board_size=board_size,
            num_actions=num_actions,
            input_channels=config.input_channels,
            num_blocks=config.num_blocks,
            num_filters=config.num_filters,
            version="1.0",
            training_steps=0,
            elo_rating=1200.0
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Standard initialization for most linear layers
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Policy head has its own custom initialization in PolicyHead.__init__
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through network
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of:
                - policy: Log probabilities of shape (batch_size, num_actions)
                - value: Position evaluations of shape (batch_size, 1)
        """
        # Initial convolution
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = F.relu(x, inplace=True)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Split into policy and value
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value


def create_resnet_for_game(
    game_type: str,
    input_channels: int = 20,
    num_blocks: int = 19,
    num_filters: int = 256
) -> ResNetModel:
    """Create a ResNet model for a specific game
    
    Args:
        game_type: Type of game ('chess', 'go', 'gomoku')
        input_channels: Number of input feature channels
        num_blocks: Number of residual blocks
        num_filters: Number of filters in conv layers
        
    Returns:
        ResNetModel configured for the game
    """
    # Game-specific configurations
    game_configs = {
        'chess': {
            'board_size': 8,
            'num_actions': 4096,  # 64 * 64 (from square to square)
            'default_blocks': 19,
            'default_filters': 256
        },
        'go': {
            'board_size': 19,
            'num_actions': 362,  # 19 * 19 + 1 (pass)
            'default_blocks': 39,  # Deeper for Go
            'default_filters': 256
        },
        'gomoku': {
            'board_size': 15,
            'num_actions': 225,  # 15 * 15
            'default_blocks': 10,  # Shallower for simpler game
            'default_filters': 128
        }
    }
    
    if game_type not in game_configs:
        raise ValueError(f"Unknown game type: {game_type}")
    
    game_config = game_configs[game_type]
    
    # Use provided values or defaults
    if num_blocks == 19 and game_type != 'chess':
        num_blocks = game_config['default_blocks']
    if num_filters == 256 and game_type == 'gomoku':
        num_filters = game_config['default_filters']
    
    # Create ResNet configuration
    config = ResNetConfig(
        num_blocks=num_blocks,
        num_filters=num_filters,
        input_channels=input_channels,
        fc_value_hidden=256,
        fc_policy_hidden=256
    )
    
    # Create and return model
    return ResNetModel(
        config=config,
        board_size=game_config['board_size'],
        num_actions=game_config['num_actions'],
        game_type=game_type
    )