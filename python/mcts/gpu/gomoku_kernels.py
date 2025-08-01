"""GPU kernels for Gomoku terminal state detection

High-performance CUDA kernels for detecting terminal states in Gomoku
games directly on GPU, eliminating CPU-GPU transfer overhead.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def gpu_check_terminal_states(boards: torch.Tensor) -> torch.Tensor:
    """GPU-optimized terminal state detection for Gomoku
    
    Args:
        boards: Batch of Gomoku boards [batch_size, board_size, board_size]
                Values: 0 = empty, 1 = player 1, 2 = player 2
    
    Returns:
        terminal_states: Boolean tensor [batch_size] indicating terminal states
        
    Optimized for:
        - 15x15 Gomoku boards
        - Large batch processing
        - Memory coalescing
        - Minimal GPU memory usage
    """
    if boards.numel() == 0:
        return torch.empty(0, dtype=torch.bool, device=boards.device)
    
    batch_size, board_size, _ = boards.shape
    device = boards.device
    
    # Convert to one-hot encoding for efficient parallel processing
    # Shape: [batch_size, 3, board_size, board_size] 
    # Channel 0: empty, Channel 1: player 1, Channel 2: player 2
    boards_onehot = F.one_hot(boards.long(), num_classes=3).permute(0, 3, 1, 2).float()
    
    # Check for wins for both players simultaneously
    player1_boards = boards_onehot[:, 1:2]  # [batch_size, 1, board_size, board_size]
    player2_boards = boards_onehot[:, 2:3]  # [batch_size, 1, board_size, board_size]
    
    # Stack players for parallel processing
    player_boards = torch.cat([player1_boards, player2_boards], dim=0)  # [2*batch_size, 1, board_size, board_size]
    
    # Use convolution kernels to detect 5-in-a-row patterns
    terminal_detected = _detect_five_in_row_patterns(player_boards, board_size)
    
    # Combine results for both players
    player1_wins = terminal_detected[:batch_size]
    player2_wins = terminal_detected[batch_size:]
    
    # Terminal if either player wins
    terminal_states = player1_wins | player2_wins
    
    return terminal_states

def _detect_five_in_row_patterns(player_boards: torch.Tensor, board_size: int) -> torch.Tensor:
    """Detect 5-in-a-row patterns using convolution
    
    Args:
        player_boards: Player occupancy [batch_size, 1, board_size, board_size]
        board_size: Size of the board (15 for Gomoku)
    
    Returns:
        has_five_in_row: Boolean tensor [batch_size] indicating 5-in-a-row
    """
    device = player_boards.device
    batch_size = player_boards.shape[0]
    
    # Create convolution kernels for different directions
    # Each kernel detects 5 consecutive pieces
    
    # Horizontal kernel: [1, 1, 1, 1, 1]
    horizontal_kernel = torch.ones(1, 1, 1, 5, device=device, dtype=torch.float32)
    
    # Vertical kernel: [[1], [1], [1], [1], [1]]
    vertical_kernel = torch.ones(1, 1, 5, 1, device=device, dtype=torch.float32)
    
    # Diagonal kernel (top-left to bottom-right)
    diagonal_kernel1 = torch.zeros(1, 1, 5, 5, device=device, dtype=torch.float32)
    for i in range(5):
        diagonal_kernel1[0, 0, i, i] = 1.0
    
    # Anti-diagonal kernel (top-right to bottom-left)  
    diagonal_kernel2 = torch.zeros(1, 1, 5, 5, device=device, dtype=torch.float32)
    for i in range(5):
        diagonal_kernel2[0, 0, i, 4-i] = 1.0
    
    # Apply convolutions to detect patterns
    # Use threshold of 4.99 to detect exactly 5 consecutive pieces
    threshold = 4.99
    
    # Horizontal detection
    h_conv = F.conv2d(player_boards, horizontal_kernel, padding=0)
    h_wins = (h_conv >= threshold).any(dim=(2, 3))
    
    # Vertical detection  
    v_conv = F.conv2d(player_boards, vertical_kernel, padding=0)
    v_wins = (v_conv >= threshold).any(dim=(2, 3))
    
    # Diagonal detection
    d1_conv = F.conv2d(player_boards, diagonal_kernel1, padding=0)
    d1_wins = (d1_conv >= threshold).any(dim=(2, 3))
    
    d2_conv = F.conv2d(player_boards, diagonal_kernel2, padding=0)
    d2_wins = (d2_conv >= threshold).any(dim=(2, 3))
    
    # Combine all win conditions
    has_five_in_row = h_wins | v_wins | d1_wins | d2_wins
    
    # Ensure output is 1D tensor [batch_size]
    has_five_in_row = has_five_in_row.squeeze()
    
    return has_five_in_row

def gpu_check_terminal_states_optimized(boards: torch.Tensor) -> torch.Tensor:
    """Optimized version using custom CUDA kernel (fallback to conv version)
    
    This function provides an interface for a potential custom CUDA kernel.
    Currently falls back to the convolution-based implementation.
    """
    # For now, use the convolution-based implementation
    # In the future, this could dispatch to a custom CUDA kernel for even better performance
    return gpu_check_terminal_states(boards)

def gpu_get_valid_moves(boards: torch.Tensor) -> torch.Tensor:
    """Get valid moves for batch of Gomoku boards
    
    Args:
        boards: Batch of boards [batch_size, board_size, board_size]
    
    Returns:
        valid_moves: Boolean tensor [batch_size, board_size, board_size]
                    True where moves are valid (empty squares)
    """
    return boards == 0

def gpu_apply_moves(boards: torch.Tensor, moves: torch.Tensor, players: torch.Tensor) -> torch.Tensor:
    """Apply moves to batch of boards
    
    Args:
        boards: Batch of boards [batch_size, board_size, board_size]
        moves: Move positions [batch_size, 2] as (row, col)
        players: Player indices [batch_size] (1 or 2)
    
    Returns:
        new_boards: Updated boards with moves applied
    """
    new_boards = boards.clone()
    batch_size = boards.shape[0]
    
    # Apply moves using advanced indexing
    batch_indices = torch.arange(batch_size, device=boards.device)
    rows = moves[:, 0]
    cols = moves[:, 1]
    
    new_boards[batch_indices, rows, cols] = players
    
    return new_boards

class GomokuGPUKernels:
    """Collection of GPU kernels for Gomoku operations"""
    
    def __init__(self, device: torch.device, board_size: int = 15):
        self.device = device
        self.board_size = board_size
        
        # Pre-compile kernels for better performance
        self._compile_kernels()
    
    def _compile_kernels(self):
        """Pre-compile kernels with torch.compile for maximum performance"""
        try:
            # Compile terminal detection for speed
            self.check_terminal_states = torch.compile(
                gpu_check_terminal_states,
                mode="max-autotune",
                fullgraph=True
            )
            
            # Compile move validation
            self.get_valid_moves = torch.compile(
                gpu_get_valid_moves,
                mode="max-autotune", 
                fullgraph=True
            )
            
            logger.info("✅ Gomoku GPU kernels compiled successfully")
            
        except Exception as e:
            logger.warning(f"Failed to compile Gomoku kernels: {e}")
            # Fall back to non-compiled versions
            self.check_terminal_states = gpu_check_terminal_states
            self.get_valid_moves = gpu_get_valid_moves
    
    def batch_terminal_detection(self, boards: torch.Tensor) -> torch.Tensor:
        """Batch terminal state detection"""
        return self.check_terminal_states(boards)
    
    def batch_valid_moves(self, boards: torch.Tensor) -> torch.Tensor:
        """Batch valid move detection"""
        return self.get_valid_moves(boards)
    
    def batch_apply_moves(self, boards: torch.Tensor, moves: torch.Tensor, players: torch.Tensor) -> torch.Tensor:
        """Batch move application"""
        return gpu_apply_moves(boards, moves, players)

# Global kernel instance for easy access
_gomoku_kernels = None

def get_gomoku_gpu_kernels(device: torch.device, board_size: int = 15) -> GomokuGPUKernels:
    """Get or create global Gomoku GPU kernels instance"""
    global _gomoku_kernels
    
    if _gomoku_kernels is None or _gomoku_kernels.device != device:
        _gomoku_kernels = GomokuGPUKernels(device, board_size)
    
    return _gomoku_kernels