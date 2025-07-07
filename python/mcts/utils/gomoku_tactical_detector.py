"""
Tactical move detection for Gomoku to boost important moves during MCTS expansion.

This module helps MCTS find critical tactical moves (winning threats, blocks, connections)
even with an untrained neural network, preventing training bias.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional


class GomokuTacticalMoveDetector:
    """Detects tactical moves in Gomoku to help MCTS exploration"""
    
    def __init__(self, board_size: int = 15, config=None):
        self.board_size = board_size
        self.config = config
        self.directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal \
            (1, -1),  # Diagonal /
        ]
        
        # Load tactical parameters from config or use defaults
        if config:
            self.win_boost = getattr(config, 'gomoku_win_boost', 100.0)
            self.block_win_boost = getattr(config, 'gomoku_block_win_boost', 90.0)
            self.open_four_boost = getattr(config, 'gomoku_open_four_boost', 50.0)
            self.block_four_boost = getattr(config, 'gomoku_block_four_boost', 45.0)
            self.threat_base_boost = getattr(config, 'gomoku_threat_base_boost', 40.0)
            self.threat_multiplier = getattr(config, 'gomoku_threat_multiplier', 5.0)
            self.three_boost = getattr(config, 'gomoku_three_boost', 20.0)
            self.block_three_boost = getattr(config, 'gomoku_block_three_boost', 18.0)
            self.center_boost = getattr(config, 'gomoku_center_boost', 3.0)
            self.connection_boost = getattr(config, 'gomoku_connection_boost', 2.0)
        else:
            # Default values
            self.win_boost = 100.0
            self.block_win_boost = 90.0
            self.open_four_boost = 50.0
            self.block_four_boost = 45.0
            self.threat_base_boost = 40.0
            self.threat_multiplier = 5.0
            self.three_boost = 20.0
            self.block_three_boost = 18.0
            self.center_boost = 3.0
            self.connection_boost = 2.0
        
    def detect_tactical_moves(self, board: torch.Tensor, current_player: int) -> torch.Tensor:
        """
        Detect tactical moves and return a priority boost for each position.
        
        Args:
            board: Board tensor of shape (board_size, board_size)
                   0 = empty, 1 = black, 2 = white
            current_player: Current player (1 or 2)
            
        Returns:
            Priority boost tensor of shape (board_size * board_size,)
            Higher values indicate more important tactical moves
        """
        boost = torch.zeros(self.board_size * self.board_size)
        opponent = 3 - current_player
        
        # Convert to numpy for easier manipulation
        board_np = board.cpu().numpy() if isinstance(board, torch.Tensor) else board
        
        for y in range(self.board_size):
            for x in range(self.board_size):
                idx = y * self.board_size + x
                
                if board_np[y, x] == 0:  # Empty position
                    move_boost = 0.0
                    
                    # 1. Win in 1 move (highest priority)
                    if self._creates_five(board_np, x, y, current_player):
                        move_boost = self.win_boost
                    
                    # 2. Block opponent's win in 1 move
                    elif self._creates_five(board_np, x, y, opponent):
                        move_boost = self.block_win_boost
                    
                    # 3. Create an open four (unstoppable threat)
                    elif self._creates_open_four(board_np, x, y, current_player):
                        move_boost = self.open_four_boost
                    
                    # 4. Block opponent's open four
                    elif self._creates_open_four(board_np, x, y, opponent):
                        move_boost = self.block_four_boost
                    
                    # 5. Create multiple threats (fork)
                    threat_count = self._count_threats(board_np, x, y, current_player)
                    if threat_count >= 2:
                        move_boost = max(move_boost, self.threat_base_boost + threat_count * self.threat_multiplier)
                    
                    # 6. Block opponent's multiple threats
                    opp_threat_count = self._count_threats(board_np, x, y, opponent)
                    if opp_threat_count >= 2:
                        move_boost = max(move_boost, (self.threat_base_boost - 5.0) + opp_threat_count * self.threat_multiplier)
                    
                    # 7. Extend to create three
                    if move_boost < 30 and self._creates_three(board_np, x, y, current_player):
                        move_boost = max(move_boost, self.three_boost)
                    
                    # 8. Block opponent's three
                    if move_boost < 25 and self._creates_three(board_np, x, y, opponent):
                        move_boost = max(move_boost, self.block_three_boost)
                    
                    # 9. Central positions (general strategy)
                    center = self.board_size // 2
                    distance_to_center = max(abs(x - center), abs(y - center))
                    if distance_to_center <= 2:
                        move_boost += self.center_boost - distance_to_center
                    
                    # 10. Connect with existing stones
                    connections = self._count_connections(board_np, x, y, current_player)
                    move_boost += connections * self.connection_boost
                    
                    boost[idx] = move_boost
                    
        return boost
    
    def _creates_five(self, board: np.ndarray, x: int, y: int, player: int) -> bool:
        """Check if placing a stone creates five in a row"""
        # Temporarily place the stone
        board[y, x] = player
        
        for dx, dy in self.directions:
            count = 1  # Count the placed stone
            
            # Count in positive direction
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and board[ny, nx] == player:
                count += 1
                nx += dx
                ny += dy
            
            # Count in negative direction
            nx, ny = x - dx, y - dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and board[ny, nx] == player:
                count += 1
                nx -= dx
                ny -= dy
            
            if count >= 5:
                board[y, x] = 0  # Restore
                return True
        
        board[y, x] = 0  # Restore
        return False
    
    def _creates_open_four(self, board: np.ndarray, x: int, y: int, player: int) -> bool:
        """Check if placing a stone creates an open four (four in a row with open ends)"""
        board[y, x] = player
        
        for dx, dy in self.directions:
            # Find the extent of the line
            stones = [(x, y)]
            
            # Extend in positive direction
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and board[ny, nx] == player:
                stones.append((nx, ny))
                nx += dx
                ny += dy
            pos_end = (nx, ny)
            
            # Extend in negative direction
            nx, ny = x - dx, y - dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and board[ny, nx] == player:
                stones.insert(0, (nx, ny))
                nx -= dx
                ny -= dy
            neg_end = (nx, ny)
            
            # Check if we have exactly 4 stones with open ends
            if len(stones) == 4:
                # Check both ends are empty and in bounds
                neg_x, neg_y = neg_end
                pos_x, pos_y = pos_end
                
                if (0 <= neg_x < self.board_size and 0 <= neg_y < self.board_size and
                    0 <= pos_x < self.board_size and 0 <= pos_y < self.board_size and
                    board[neg_y, neg_x] == 0 and board[pos_y, pos_x] == 0):
                    board[y, x] = 0  # Restore
                    return True
        
        board[y, x] = 0  # Restore
        return False
    
    def _creates_three(self, board: np.ndarray, x: int, y: int, player: int) -> bool:
        """Check if placing a stone creates an open three"""
        board[y, x] = player
        
        for dx, dy in self.directions:
            # Count consecutive stones including placed one
            count = 1
            space_before = 0
            space_after = 0
            
            # Check positive direction
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if board[ny, nx] == player:
                    count += 1
                    nx += dx
                    ny += dy
                elif board[ny, nx] == 0:
                    # Check for space
                    nnx, nny = nx + dx, ny + dy
                    if 0 <= nnx < self.board_size and 0 <= nny < self.board_size:
                        space_after += 1
                    break
                else:
                    break
            
            # Check negative direction
            nx, ny = x - dx, y - dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if board[ny, nx] == player:
                    count += 1
                    nx -= dx
                    ny -= dy
                elif board[ny, nx] == 0:
                    # Check for space
                    nnx, nny = nx - dx, ny - dy
                    if 0 <= nnx < self.board_size and 0 <= nny < self.board_size:
                        space_before += 1
                    break
                else:
                    break
            
            # Open three: exactly 3 stones with space on both sides
            if count == 3 and space_before > 0 and space_after > 0:
                board[y, x] = 0  # Restore
                return True
        
        board[y, x] = 0  # Restore
        return False
    
    def _count_threats(self, board: np.ndarray, x: int, y: int, player: int) -> int:
        """Count number of threats (potential fours) created by this move"""
        threat_count = 0
        board[y, x] = player
        
        for dx, dy in self.directions:
            # Check if this direction could form a four
            potential_four = 0
            stones = []
            
            # Scan in both directions
            for d in [-3, -2, -1, 0, 1, 2, 3]:
                nx, ny = x + d * dx, y + d * dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if board[ny, nx] == player:
                        stones.append(d)
            
            # Check for patterns that could become four
            if len(stones) >= 3:
                # Check if stones are close enough to form a four
                for i in range(len(stones) - 2):
                    if stones[i+2] - stones[i] <= 3:  # Can form four in a row
                        threat_count += 1
                        break
        
        board[y, x] = 0  # Restore
        return threat_count
    
    def _count_connections(self, board: np.ndarray, x: int, y: int, player: int) -> int:
        """Count how many friendly stones this move connects to"""
        connections = 0
        
        # Check all 8 directions
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if board[ny, nx] == player:
                        connections += 1
        
        return connections
    
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