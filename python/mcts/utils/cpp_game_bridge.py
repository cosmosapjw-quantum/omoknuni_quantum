"""Bridge between Python game states and C++ game logic for terminal detection

This module provides efficient terminal detection using the C++ game implementation
while maintaining compatibility with the Python game state management.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class CPPGameBridge:
    """Bridge to C++ game logic for terminal detection and game rules"""
    
    def __init__(self, game_interface):
        """Initialize with a C++ game interface
        
        Args:
            game_interface: C++ game interface from create_game_interface()
        """
        self.game_interface = game_interface
        self.board_size = game_interface.board_size
        
        # Cache for C++ game states to avoid repeated creation
        self._cpp_state_cache = {}
        
        # Debug counter
        self._check_count = 0
        
    def check_terminal_batch(self, boards: Union[np.ndarray, torch.Tensor], 
                           move_histories: Optional[List[List[int]]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Check terminal states for a batch of boards using C++ logic
        
        Args:
            boards: Board states of shape (batch_size, board_size, board_size)
                   Values: 0=empty, 1=player1, 2=player2 (or -1 for player2)
            move_histories: Optional list of move sequences for each board
            
        Returns:
            Tuple of (is_terminal, winner) arrays
            is_terminal: Boolean array of shape (batch_size,)
            winner: Integer array of shape (batch_size,) with 0=no winner/draw, 1=player1, 2=player2
        """
        if isinstance(boards, torch.Tensor):
            boards = boards.cpu().numpy()
            
        batch_size = boards.shape[0]
        is_terminal = np.zeros(batch_size, dtype=bool)
        winner = np.zeros(batch_size, dtype=np.int8)
        
        for i in range(batch_size):
            board = boards[i]
            
            # Create C++ state and replay moves
            cpp_state = self._reconstruct_cpp_state(board, 
                                                  move_histories[i] if move_histories else None)
            
            # Check terminal state
            if self.game_interface.is_terminal(cpp_state):
                is_terminal[i] = True
                cpp_winner = self.game_interface.get_winner(cpp_state)
                # Convert from C++ convention (-1, 0, 1) to our convention (0, 1, 2)
                if cpp_winner == 1:
                    winner[i] = 1
                elif cpp_winner == -1:
                    winner[i] = 2
                else:  # Draw
                    winner[i] = 0
                    
        return is_terminal, winner
    
    def check_single_terminal(self, board: Union[np.ndarray, torch.Tensor],
                            last_move: Optional[int] = None) -> Tuple[bool, int]:
        """Check if a single board state is terminal
        
        Args:
            board: Board state of shape (board_size, board_size)
            last_move: Optional last move played (for optimization)
            
        Returns:
            Tuple of (is_terminal, winner)
        """
        if isinstance(board, torch.Tensor):
            board = board.cpu().numpy()
            
        # For Gomoku, we can do a quick optimization
        if last_move is not None and hasattr(self.game_interface, 'game_type'):
            game_type_str = None
            if hasattr(self.game_interface.game_type, 'value'):
                game_type_str = str(self.game_interface.game_type.value).lower()
            elif hasattr(self.game_interface.game_type, 'name'):
                game_type_str = self.game_interface.game_type.name.lower()
                
            if game_type_str == 'gomoku':
                row = last_move // self.board_size
                col = last_move % self.board_size
                player = board[row, col]
                
                if player != 0 and self._quick_gomoku_check(board, row, col, player):
                    return True, int(player)
        
        # Full check using C++ logic
        cpp_state = self._reconstruct_cpp_state(board)
        
        if self.game_interface.is_terminal(cpp_state):
            cpp_winner = self.game_interface.get_winner(cpp_state)
            # Convert winner
            if cpp_winner == 1:
                winner = 1
            elif cpp_winner == -1:
                winner = 2
            else:
                winner = 0
            return True, winner
            
        return False, 0
    
    def _reconstruct_cpp_state(self, board: np.ndarray, 
                             move_history: Optional[List[int]] = None):
        """Reconstruct C++ game state from board
        
        Args:
            board: Board array
            move_history: Optional sequence of moves
            
        Returns:
            C++ game state
        """
        cpp_state = self.game_interface.create_initial_state()
        
        if move_history:
            # Use provided move history
            for move in move_history:
                if self.game_interface.is_legal_move(cpp_state, move):
                    cpp_state = self.game_interface.apply_move(cpp_state, move)
        else:
            # Reconstruct from board positions
            # Find all non-empty positions
            positions = []
            for row in range(self.board_size):
                for col in range(self.board_size):
                    if board[row, col] != 0:
                        move = row * self.board_size + col
                        player = board[row, col]
                        positions.append((move, player))
            
            # Try to reconstruct a valid game sequence
            # For Gomoku, we can use a simple heuristic: alternate players
            positions.sort(key=lambda x: (x[1], x[0]))  # Sort by player then position
            
            player1_moves = [p[0] for p in positions if p[1] == 1]
            player2_moves = [p[0] for p in positions if p[1] == 2 or p[1] == -1]
            
            # Interleave moves
            move_sequence = []
            for i in range(max(len(player1_moves), len(player2_moves))):
                if i < len(player1_moves):
                    move_sequence.append(player1_moves[i])
                if i < len(player2_moves):
                    move_sequence.append(player2_moves[i])
            
            # Apply moves
            for move in move_sequence:
                if self.game_interface.is_legal_move(cpp_state, move):
                    cpp_state = self.game_interface.apply_move(cpp_state, move)
                    
        return cpp_state
    
    def _quick_gomoku_check(self, board: np.ndarray, row: int, col: int, player: int) -> bool:
        """Quick check for 5-in-a-row in Gomoku
        
        Args:
            board: Board array
            row, col: Last move position
            player: Player who made the move
            
        Returns:
            True if player has won
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1
            
            # Check positive direction
            r, c = row + dx, col + dy
            while 0 <= r < self.board_size and 0 <= c < self.board_size and board[r, c] == player:
                count += 1
                r += dx
                c += dy
                
            # Check negative direction
            r, c = row - dx, col - dy
            while 0 <= r < self.board_size and 0 <= c < self.board_size and board[r, c] == player:
                count += 1
                r -= dx
                c -= dy
                
            if count >= 5:
                return True
                
        return False