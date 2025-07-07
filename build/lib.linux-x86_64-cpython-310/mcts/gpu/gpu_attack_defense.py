"""GPU-accelerated attack/defense computation

This module provides GPU-optimized implementations of attack/defense score
computation for improved performance.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class GPUAttackDefenseComputer:
    """GPU-accelerated attack/defense score computation"""
    
    def __init__(self, device: torch.device = None):
        """Initialize GPU computer
        
        Args:
            device: Torch device to use (defaults to cuda if available)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Pre-compute pattern detection kernels
        self._init_pattern_kernels()
        
    def _init_pattern_kernels(self):
        """Initialize convolution kernels for pattern detection"""
        # Create kernels for different line patterns (5 stones)
        # These will be used for convolution-based pattern matching
        
        # Horizontal line detection
        self.horizontal_kernel = torch.zeros(1, 1, 1, 5, device=self.device)
        self.horizontal_kernel[0, 0, 0, :] = 1.0
        
        # Vertical line detection  
        self.vertical_kernel = torch.zeros(1, 1, 5, 1, device=self.device)
        self.vertical_kernel[0, 0, :, 0] = 1.0
        
        # Diagonal line detection
        self.diagonal_kernel = torch.zeros(1, 1, 5, 5, device=self.device)
        for i in range(5):
            self.diagonal_kernel[0, 0, i, i] = 1.0
            
        # Anti-diagonal line detection
        self.antidiag_kernel = torch.zeros(1, 1, 5, 5, device=self.device)
        for i in range(5):
            self.antidiag_kernel[0, 0, i, 4-i] = 1.0
            
    def compute_gomoku_scores_gpu(
        self,
        boards: torch.Tensor,
        current_players: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attack/defense scores for Gomoku using GPU
        
        Args:
            boards: Board states tensor (batch_size, height, width)
            current_players: Current player for each board (batch_size,)
            
        Returns:
            Tuple of (attack_scores, defense_scores) tensors
        """
        batch_size, height, width = boards.shape
        
        # Create player masks
        player1_mask = (boards == 1).float()
        player2_mask = (boards == 2).float()
        empty_mask = (boards == 0).float()
        
        # Prepare output tensors
        attack_scores = torch.zeros_like(boards, dtype=torch.float32)
        defense_scores = torch.zeros_like(boards, dtype=torch.float32)
        
        # Process each board in batch
        for b in range(batch_size):
            current_player = int(current_players[b].item())
            opponent = 2 if current_player == 0 else 1
            
            # Get masks for this board
            my_stones = player1_mask[b:b+1] if current_player == 0 else player2_mask[b:b+1]
            opp_stones = player2_mask[b:b+1] if current_player == 0 else player1_mask[b:b+1]
            empty = empty_mask[b:b+1]
            
            # Compute attack scores (patterns we can complete)
            attack_scores[b] = self._compute_pattern_scores(
                my_stones.unsqueeze(0), empty.unsqueeze(0)
            ).squeeze()
            
            # Compute defense scores (opponent patterns to block)
            defense_scores[b] = self._compute_pattern_scores(
                opp_stones.unsqueeze(0), empty.unsqueeze(0)
            ).squeeze()
            
        return attack_scores, defense_scores
        
    def _compute_pattern_scores(
        self,
        stone_mask: torch.Tensor,
        empty_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute pattern scores using convolution
        
        Args:
            stone_mask: Mask of player stones (1, 1, H, W)
            empty_mask: Mask of empty positions (1, 1, H, W)
            
        Returns:
            Score tensor (1, H, W)
        """
        _, _, height, width = stone_mask.shape
        scores = torch.zeros(1, height, width, device=self.device)
        
        # Detect different patterns using convolution
        patterns = [
            (self.horizontal_kernel, 'horizontal'),
            (self.vertical_kernel, 'vertical'),
            (self.diagonal_kernel, 'diagonal'),
            (self.antidiag_kernel, 'antidiag')
        ]
        
        for kernel, name in patterns:
            # Convolve to find patterns
            conv_result = F.conv2d(stone_mask, kernel, padding='same')
            
            # Different scores for different pattern lengths
            # 4 in a row = very high score
            four_pattern = (conv_result == 4).float()
            scores += four_pattern.squeeze(0) * 1000
            
            # 3 in a row = high score
            three_pattern = (conv_result == 3).float()
            scores += three_pattern.squeeze(0) * 100
            
            # 2 in a row = medium score
            two_pattern = (conv_result == 2).float()
            scores += two_pattern.squeeze(0) * 10
            
        # Only score empty positions
        scores = scores * empty_mask.squeeze(0)
        
        # Normalize scores
        if scores.max() > 0:
            scores = scores / scores.max()
            
        return scores
        
    def compute_batch_scores(
        self,
        game_type: str,
        boards: np.ndarray,
        current_players: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute attack/defense scores for a batch of boards
        
        Args:
            game_type: Type of game ('gomoku', 'chess', 'go')
            boards: Batch of board states (batch_size, height, width)
            current_players: Current player for each board (batch_size,)
            
        Returns:
            Tuple of (attack_scores, defense_scores) arrays
        """
        # Convert to tensors
        boards_tensor = torch.from_numpy(boards).float().to(self.device)
        players_tensor = torch.from_numpy(current_players).to(self.device)
        
        if game_type == 'gomoku':
            attack_tensor, defense_tensor = self.compute_gomoku_scores_gpu(
                boards_tensor, players_tensor
            )
        else:
            # Placeholder for other games
            batch_size, height, width = boards.shape
            attack_tensor = torch.zeros(batch_size, height, width, device=self.device)
            defense_tensor = torch.zeros(batch_size, height, width, device=self.device)
            
        # Convert back to numpy
        attack_scores = attack_tensor.cpu().numpy()
        defense_scores = defense_tensor.cpu().numpy()
        
        return attack_scores, defense_scores


def create_gpu_attack_defense_computer():
    """Factory function to create GPU attack/defense computer
    
    Returns:
        GPUAttackDefenseComputer instance if GPU available, None otherwise
    """
    if torch.cuda.is_available():
        return GPUAttackDefenseComputer()
    else:
        logger.warning("GPU not available for attack/defense computation")
        return None