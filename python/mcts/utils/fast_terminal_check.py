"""Fast terminal detection for game states

This module provides efficient terminal detection for all game types.
For simple games like Gomoku, it uses optimized native implementations.
For complex games, it delegates to the C++ game interface.
"""

import numpy as np
import torch
from typing import Union, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class FastTerminalChecker:
    """Fast terminal detection for various games"""
    
    def __init__(self, game_type: str, board_size: int, variant_rules: Optional[dict] = None, 
                 game_interface=None):
        """Initialize fast terminal checker
        
        Args:
            game_type: Type of game ('gomoku', 'go', 'chess')
            board_size: Board size
            variant_rules: Optional variant-specific rules (e.g., use_renju, use_omok)
            game_interface: Optional C++ game interface for complex games
        """
        self.game_type = game_type.lower()
        self.board_size = board_size
        self.variant_rules = variant_rules or {}
        self.game_interface = game_interface
        
        # For Gomoku, precompute direction vectors
        if self.game_type == 'gomoku':
            self.directions = np.array([
                [0, 1],   # Horizontal
                [1, 0],   # Vertical
                [1, 1],   # Diagonal \
                [1, -1],  # Diagonal /
            ])
            
        # Track move history for games that need it
        self._move_histories = {}
            
    def check_terminal_cpu(self, board: np.ndarray, last_move: Optional[int] = None, 
                          state_idx: Optional[int] = None) -> Tuple[bool, int]:
        """Fast terminal check for CPU (NumPy)
        
        Args:
            board: Board state (board_size, board_size) with 0=empty, 1=player1, 2=player2
            last_move: Optional last move for optimization
            state_idx: Optional state index for tracking move history
            
        Returns:
            Tuple of (is_terminal, winner) where winner is 0=no/draw, 1=player1, 2=player2
        """
        # For games with complex rules, use C++ interface if available
        if self.game_interface and self.game_type in ['chess', 'go']:
            return self._check_terminal_via_cpp(board, state_idx)
        
        # For Gomoku, use optimized native implementation
        if self.game_type == 'gomoku':
            return self._check_gomoku_terminal_cpu(board, last_move)
        
        # Fallback for unknown games
        return False, 0
            
    def check_terminal_gpu(self, boards: torch.Tensor, last_moves: Optional[torch.Tensor] = None,
                          state_indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fast terminal check for GPU (PyTorch) - batch processing
        
        Args:
            boards: Board states (batch_size, board_size, board_size)
            last_moves: Optional last moves for each board
            state_indices: Optional state indices for tracking move history
            
        Returns:
            Tuple of (is_terminal, winner) tensors
        """
        batch_size = boards.shape[0]
        device = boards.device
        
        # For games with complex rules, delegate to game-specific kernels
        if self.game_type == 'gomoku':
            return self._check_gomoku_terminal_gpu(boards, last_moves)
        elif self.game_type == 'go':
            # Import Go-specific kernel
            try:
                from ..gpu.go_kernels import gpu_check_terminal_states as go_check_terminal
                # Go needs pass count tracking - simplified for now
                pass_counts = torch.zeros(batch_size, dtype=torch.int8, device=device)
                return go_check_terminal(boards, pass_counts)
            except ImportError:
                # Fallback if kernel not available
                pass
        elif self.game_type == 'chess':
            # Import Chess-specific kernel
            try:
                from ..gpu.chess_kernels import gpu_check_terminal_states as chess_check_terminal
                # Chess needs additional state - simplified for now
                to_move = torch.ones(batch_size, dtype=torch.int8, device=device)
                castling = torch.zeros((batch_size, 4), dtype=torch.bool, device=device)
                halfmove = torch.zeros(batch_size, dtype=torch.int16, device=device)
                return chess_check_terminal(boards, to_move, castling, halfmove)
            except ImportError:
                # Fallback if kernel not available
                pass
        
        # Default: no games are terminal
        return torch.zeros(batch_size, dtype=torch.bool, device=device), \
               torch.zeros(batch_size, dtype=torch.int8, device=device)
    
    def _check_terminal_via_cpp(self, board: np.ndarray, state_idx: Optional[int] = None) -> Tuple[bool, int]:
        """Check terminal state via C++ game interface
        
        This is used for games with complex rules (Chess, Go) where the C++ 
        implementation is authoritative.
        """
        if not self.game_interface:
            return False, 0
            
        # Get or create CPP bridge
        from .cpp_game_bridge import CPPGameBridge
        if not hasattr(self, '_cpp_bridge'):
            self._cpp_bridge = CPPGameBridge(self.game_interface)
            
        # Use C++ terminal detection
        is_terminal, winner = self._cpp_bridge.check_single_terminal(board)
        return is_terminal, winner
    
    def _check_gomoku_terminal_cpu(self, board: np.ndarray, last_move: Optional[int] = None) -> Tuple[bool, int]:
        """Check Gomoku terminal state on CPU
        
        Handles different variants:
        - Freestyle: 5+ in a row wins
        - Renju: Exactly 5 for black, 5+ for white (simplified - no forbidden move check)
        - Omok: Similar to freestyle with some restrictions
        """
        # If we have last move, only check around it (optimization)
        if last_move is not None:
            row = last_move // self.board_size
            col = last_move % self.board_size
            player = board[row, col]
            
            if player == 0:
                return False, 0
                
            # Check all 4 directions from last move
            for dx, dy in self.directions:
                count = 1  # Count the placed stone
                
                # Count in positive direction
                r, c = row + dx, col + dy
                while 0 <= r < self.board_size and 0 <= c < self.board_size and board[r, c] == player:
                    count += 1
                    r += dx
                    c += dy
                    
                # Count in negative direction
                r, c = row - dx, col - dy
                while 0 <= r < self.board_size and 0 <= c < self.board_size and board[r, c] == player:
                    count += 1
                    r -= dx
                    c -= dy
                    
                # Check win condition based on variant
                if self.variant_rules.get('use_renju') and player == 1:  # Black in Renju
                    if count == 5:  # Exactly 5 (overlines don't count)
                        return True, player
                else:
                    if count >= 5:  # 5 or more
                        return True, player
        else:
            # Full board scan (slower)
            # Check all positions
            for row in range(self.board_size):
                for col in range(self.board_size):
                    player = board[row, col]
                    if player == 0:
                        continue
                        
                    # Check horizontal (only need to check to the right)
                    if col <= self.board_size - 5:
                        if np.all(board[row, col:col+5] == player):
                            # Check for exact 5 in Renju for black
                            if self.variant_rules.get('use_renju') and player == 1:
                                if col + 5 < self.board_size and board[row, col+5] == player:
                                    continue  # Overline, not a win for black
                                if col > 0 and board[row, col-1] == player:
                                    continue  # Overline, not a win for black
                            return True, player
                            
                    # Check vertical (only need to check down)
                    if row <= self.board_size - 5:
                        if np.all(board[row:row+5, col] == player):
                            # Check for exact 5 in Renju for black
                            if self.variant_rules.get('use_renju') and player == 1:
                                if row + 5 < self.board_size and board[row+5, col] == player:
                                    continue  # Overline
                                if row > 0 and board[row-1, col] == player:
                                    continue  # Overline
                            return True, player
                            
                    # Check diagonal \
                    if row <= self.board_size - 5 and col <= self.board_size - 5:
                        if all(board[row+i, col+i] == player for i in range(5)):
                            # Check for exact 5 in Renju for black
                            if self.variant_rules.get('use_renju') and player == 1:
                                if row + 5 < self.board_size and col + 5 < self.board_size and board[row+5, col+5] == player:
                                    continue
                                if row > 0 and col > 0 and board[row-1, col-1] == player:
                                    continue
                            return True, player
                            
                    # Check diagonal /
                    if row <= self.board_size - 5 and col >= 4:
                        if all(board[row+i, col-i] == player for i in range(5)):
                            # Check for exact 5 in Renju for black
                            if self.variant_rules.get('use_renju') and player == 1:
                                if row + 5 < self.board_size and col - 5 >= 0 and board[row+5, col-5] == player:
                                    continue
                                if row > 0 and col < self.board_size - 1 and board[row-1, col+1] == player:
                                    continue
                            return True, player
        
        # Check for draw (board full)
        if np.all(board != 0):
            return True, 0
            
        return False, 0
    
    def _check_gomoku_terminal_gpu(self, boards: torch.Tensor, last_moves: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized Gomoku terminal detection on GPU
        
        Uses convolution for efficient pattern matching across entire batch
        """
        batch_size = boards.shape[0]
        device = boards.device
        
        # Initialize results
        is_terminal = torch.zeros(batch_size, dtype=torch.bool, device=device)
        winner = torch.zeros(batch_size, dtype=torch.int8, device=device)
        
        # Create convolution kernels for 5-in-a-row detection
        # We need separate kernels for each player
        kernels = []
        
        # Horizontal kernel [1, 1, 1, 1, 1]
        h_kernel = torch.ones(1, 1, 1, 5, device=device)
        kernels.append(h_kernel)
        
        # Vertical kernel
        v_kernel = torch.ones(1, 1, 5, 1, device=device)
        kernels.append(v_kernel)
        
        # Diagonal \ kernel
        d1_kernel = torch.eye(5, device=device).unsqueeze(0).unsqueeze(0)
        kernels.append(d1_kernel)
        
        # Diagonal / kernel
        d2_kernel = torch.flip(torch.eye(5, device=device), [0]).unsqueeze(0).unsqueeze(0)
        kernels.append(d2_kernel)
        
        # CRITICAL FIX: Determine winner based on position priority, not player order
        # This ensures consistency with CPU implementation
        
        # Store all winning positions for each player and kernel
        winning_positions = {}  # {(player, kernel_idx): conv_result}
        
        # Check for each player
        for player in [1, 2]:
            # Create binary mask for this player
            player_boards = (boards == player).float().unsqueeze(1)  # (batch, 1, H, W)
            
            # Apply each kernel
            for kernel_idx, kernel in enumerate(kernels):
                # Use conv2d to find 5-in-a-row patterns
                conv_result = torch.nn.functional.conv2d(player_boards, kernel, padding=0)
                
                # A value of 5.0 indicates 5 in a row
                has_five_positions = (conv_result >= 4.999)  # Keep spatial dimensions
                
                # For Renju rules (black = player 1), need to check for exactly 5
                if self.variant_rules.get('use_renju') and player == 1:
                    # TODO: Implement proper overline detection for Renju
                    # For now, accept any 5-in-a-row for black
                    pass
                
                # Store winning positions for later priority determination
                if has_five_positions.any():
                    winning_positions[(player, kernel_idx)] = has_five_positions
        
        # Determine winner based on spatial priority (like CPU implementation)
        # Process in the same order as CPU: row by row, left to right
        for batch_idx in range(batch_size):
            if is_terminal[batch_idx]:
                continue  # Already determined
                
            # Find the earliest winning position (top-left priority)
            earliest_winner = None
            earliest_position = (float('inf'), float('inf'))
            
            for (player, kernel_idx), positions in winning_positions.items():
                if batch_idx >= positions.shape[0]:
                    continue
                    
                # Find all winning positions for this player/kernel in this batch
                win_locations = torch.where(positions[batch_idx, 0])  # Remove channel dim
                
                if len(win_locations[0]) > 0:
                    # Find the topmost, then leftmost winning position
                    min_row = win_locations[0].min().item()
                    min_col_at_min_row = win_locations[1][win_locations[0] == min_row].min().item()
                    
                    position = (min_row, min_col_at_min_row)
                    
                    # Check if this is earlier than current earliest
                    if position < earliest_position:
                        earliest_position = position
                        earliest_winner = player
            
            # Set the winner based on spatial priority
            if earliest_winner is not None:
                is_terminal[batch_idx] = True
                winner[batch_idx] = earliest_winner
        
        # Check for draw (board full)
        board_full = (boards != 0).all(dim=(1, 2))
        draw_games = board_full & ~is_terminal
        is_terminal |= draw_games
        # winner remains 0 for draws
        
        return is_terminal, winner