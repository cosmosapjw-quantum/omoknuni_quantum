"""
Tactical move detection for Go to boost important moves during MCTS expansion.

This module helps MCTS find critical tactical moves (captures, escapes, atari responses)
even with an untrained neural network, preventing training bias.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional


class GoTacticalMoveDetector:
    """Detects tactical moves in Go to help MCTS exploration"""
    
    def __init__(self, board_size: int = 9, config=None):
        self.board_size = board_size
        self.config = config
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E
        
        # Load tactical parameters from config or use defaults
        if config:
            self.capture_boost = getattr(config, 'go_capture_boost', 15.0)
            self.escape_boost = getattr(config, 'go_escape_boost', 12.0)
            self.atari_boost = getattr(config, 'go_atari_boost', 10.0)
            self.save_boost = getattr(config, 'go_save_boost', 8.0)
            self.territory_boost = getattr(config, 'go_territory_boost', 5.0)
            self.connection_boost = getattr(config, 'go_connection_boost', 3.0)
            self.eye_boost = getattr(config, 'go_eye_boost', 7.0)
            self.corner_boost = getattr(config, 'go_corner_boost', 2.0)
        else:
            # Default values
            self.capture_boost = 15.0
            self.escape_boost = 12.0
            self.atari_boost = 10.0
            self.save_boost = 8.0
            self.territory_boost = 5.0
            self.connection_boost = 3.0
            self.eye_boost = 7.0
            self.corner_boost = 2.0
        
    def detect_tactical_moves(self, board: torch.Tensor, current_player: int) -> torch.Tensor:
        """
        Detect tactical moves and return a priority boost for each position.
        
        Args:
            board: Board tensor of shape (board_size, board_size) with values:
                   0 = empty, 1 = black, 2 = white
            current_player: Current player (1 = black or 2 = white)
            
        Returns:
            Priority boost tensor of shape (board_size * board_size + 1,)
            Higher values indicate more important tactical moves
        """
        boost = torch.zeros(self.board_size * self.board_size + 1)
        opponent = 3 - current_player
        
        # Convert to numpy for easier manipulation
        board_np = board.cpu().numpy() if isinstance(board, torch.Tensor) else board
        
        for y in range(self.board_size):
            for x in range(self.board_size):
                idx = y * self.board_size + x
                
                if board_np[y, x] == 0:  # Empty position
                    # Check if this move would capture
                    capture_score = self._check_captures(board_np, x, y, current_player)
                    boost[idx] += capture_score * self.capture_boost
                    
                    # Check if this saves our groups from capture
                    save_score = self._check_saves(board_np, x, y, current_player)
                    boost[idx] += save_score * self.save_boost
                    
                    # Check if this puts opponent in atari
                    atari_score = self._check_atari(board_np, x, y, current_player)
                    boost[idx] += atari_score * self.atari_boost
                    
                    # Check if this connects our groups
                    connect_score = self._check_connections(board_np, x, y, current_player)
                    boost[idx] += connect_score * self.connection_boost
                    
                    # Check if this is a good extension from existing stones
                    extension_score = self._check_extensions(board_np, x, y, current_player)
                    boost[idx] += extension_score * self.corner_boost
        
        # Small boost for pass move in endgame situations
        if self._is_endgame(board_np):
            boost[-1] = 1.0
            
        return boost
    
    def boost_prior_with_tactics(self, prior: torch.Tensor, board: torch.Tensor,
                                current_player: int, boost_strength: float = 0.3) -> torch.Tensor:
        """
        Boost prior probabilities based on tactical importance.
        
        Args:
            prior: Original prior probabilities
            board: Current board state
            current_player: Current player
            boost_strength: How much to boost tactical moves (0.0 = no boost, 1.0 = full replacement)
            
        Returns:
            Boosted prior probabilities
        """
        tactical_boost = self.detect_tactical_moves(board, current_player)
        
        # Normalize boost to probability-like values
        if tactical_boost.max() > 0:
            tactical_boost = tactical_boost / (tactical_boost.sum() + 1e-8)
            
            # Mix original prior with tactical boost
            boosted_prior = (1 - boost_strength) * prior + boost_strength * tactical_boost.to(prior.device)
            
            # Renormalize
            boosted_prior = boosted_prior / (boosted_prior.sum() + 1e-8)
            
            return boosted_prior
        
        return prior
    
    def _check_captures(self, board: np.ndarray, x: int, y: int, player: int) -> float:
        """Check if placing a stone would capture opponent groups"""
        opponent = 3 - player
        total_captured = 0
        
        # Temporarily place the stone
        board[y, x] = player
        
        # Check all adjacent opponent groups
        for dx, dy in self.directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if board[ny, nx] == opponent:
                    # Check if this opponent group would have no liberties
                    if self._count_liberties(board, nx, ny) == 0:
                        group_size = self._get_group_size(board, nx, ny)
                        total_captured += group_size
        
        # Restore board
        board[y, x] = 0
        
        # Higher score for capturing more stones
        if total_captured > 0:
            return 1.0 + np.log(total_captured + 1)
        return 0.0
    
    def _check_saves(self, board: np.ndarray, x: int, y: int, player: int) -> float:
        """Check if this move saves our groups from capture"""
        save_score = 0.0
        
        # Check all adjacent friendly groups
        for dx, dy in self.directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if board[ny, nx] == player:
                    liberties = self._count_liberties(board, nx, ny)
                    if liberties == 1:  # Group in atari
                        # Check if this move would increase liberties
                        board[y, x] = player
                        new_liberties = self._count_liberties(board, nx, ny)
                        board[y, x] = 0
                        
                        if new_liberties > 1:
                            group_size = self._get_group_size(board, nx, ny)
                            save_score += 1.0 + np.log(group_size + 1)
        
        return save_score
    
    def _check_atari(self, board: np.ndarray, x: int, y: int, player: int) -> float:
        """Check if this move puts opponent groups in atari"""
        opponent = 3 - player
        atari_score = 0.0
        
        # Temporarily place the stone
        board[y, x] = player
        
        # Check all adjacent opponent groups
        for dx, dy in self.directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if board[ny, nx] == opponent:
                    liberties = self._count_liberties(board, nx, ny)
                    if liberties == 1:  # Put in atari
                        group_size = self._get_group_size(board, nx, ny)
                        atari_score += 0.5 + 0.1 * group_size
        
        # Restore board
        board[y, x] = 0
        
        return atari_score
    
    def _check_connections(self, board: np.ndarray, x: int, y: int, player: int) -> float:
        """Check if this move connects friendly groups"""
        adjacent_groups = set()
        
        for dx, dy in self.directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if board[ny, nx] == player:
                    group_id = self._get_group_id(board, nx, ny)
                    adjacent_groups.add(group_id)
        
        # Connecting multiple groups is valuable
        if len(adjacent_groups) > 1:
            return float(len(adjacent_groups) - 1)
        return 0.0
    
    def _check_extensions(self, board: np.ndarray, x: int, y: int, player: int) -> float:
        """Check if this is a good extension from existing stones"""
        # Simple heuristic: extensions from stones with few liberties
        extension_score = 0.0
        
        for dx, dy in self.directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if board[ny, nx] == player:
                    liberties = self._count_liberties(board, nx, ny)
                    if liberties <= 3:
                        extension_score += 0.5 / liberties
        
        return min(extension_score, 1.0)
    
    def _count_liberties(self, board: np.ndarray, x: int, y: int) -> int:
        """Count liberties of the group at (x, y)"""
        if board[y, x] == 0:
            return 0
            
        color = board[y, x]
        visited = set()
        liberties = set()
        
        def flood_fill(cx, cy):
            if (cx, cy) in visited:
                return
            visited.add((cx, cy))
            
            for dx, dy in self.directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if board[ny, nx] == 0:
                        liberties.add((nx, ny))
                    elif board[ny, nx] == color:
                        flood_fill(nx, ny)
        
        flood_fill(x, y)
        return len(liberties)
    
    def _get_group_size(self, board: np.ndarray, x: int, y: int) -> int:
        """Get size of the group at (x, y)"""
        if board[y, x] == 0:
            return 0
            
        color = board[y, x]
        visited = set()
        
        def flood_fill(cx, cy):
            if (cx, cy) in visited:
                return 0
            if board[cy, cx] != color:
                return 0
            visited.add((cx, cy))
            
            size = 1
            for dx, dy in self.directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    size += flood_fill(nx, ny)
            return size
        
        return flood_fill(x, y)
    
    def _get_group_id(self, board: np.ndarray, x: int, y: int) -> tuple:
        """Get a unique identifier for the group (using min coordinates)"""
        if board[y, x] == 0:
            return (x, y)
            
        color = board[y, x]
        visited = set()
        min_pos = (x, y)
        
        def flood_fill(cx, cy):
            nonlocal min_pos
            if (cx, cy) in visited:
                return
            if board[cy, cx] != color:
                return
            visited.add((cx, cy))
            min_pos = min(min_pos, (cx, cy))
            
            for dx, dy in self.directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    flood_fill(nx, ny)
        
        flood_fill(x, y)
        return min_pos
    
    def _is_endgame(self, board: np.ndarray) -> bool:
        """Simple heuristic to detect endgame"""
        empty_count = np.sum(board == 0)
        total_positions = self.board_size * self.board_size
        # Consider endgame when board is >70% full
        return empty_count < 0.3 * total_positions
    
    def boost_prior_with_tactics(self, prior: torch.Tensor, board: torch.Tensor, 
                                 current_player: int, boost_strength: float = 0.3) -> torch.Tensor:
        """
        Boost prior probabilities based on tactical importance.
        
        Args:
            prior: Original prior probabilities
            board: Current board state
            current_player: Current player
            boost_strength: How much to boost tactical moves (0.0 = no boost, 1.0 = full replacement)
            
        Returns:
            Boosted prior probabilities
        """
        tactical_boost = self.detect_tactical_moves(board, current_player)
        
        # Normalize boost to probability-like values
        if tactical_boost.max() > 0:
            tactical_boost = tactical_boost / (tactical_boost.sum() + 1e-8)
            
            # Mix original prior with tactical boost
            boosted_prior = (1 - boost_strength) * prior + boost_strength * tactical_boost.to(prior.device)
            
            # Renormalize
            boosted_prior = boosted_prior / (boosted_prior.sum() + 1e-8)
            
            return boosted_prior
        
        return prior