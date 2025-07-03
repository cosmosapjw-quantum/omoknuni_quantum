"""Attack/Defense scoring module for board games

This module provides Python implementations of attack and defense scoring
functions that compute tactical bonuses for board positions.
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def compute_attack_defense_scores(
    game_type: str,
    board: np.ndarray,
    current_player: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute attack and defense score planes for a board position
    
    Args:
        game_type: Type of game ('chess', 'go', 'gomoku')
        board: Board state array of shape (height, width)
        current_player: Current player (0 or 1)
        
    Returns:
        Tuple of (attack_plane, defense_plane) arrays
    """
    if game_type == 'gomoku':
        return compute_gomoku_attack_defense(board, current_player)
    elif game_type == 'chess':
        return compute_chess_attack_defense(board, current_player)
    elif game_type == 'go':
        return compute_go_attack_defense(board, current_player)
    else:
        # Fallback: return zero planes
        return np.zeros_like(board, dtype=np.float32), np.zeros_like(board, dtype=np.float32)


def compute_gomoku_attack_defense(
    board: np.ndarray,
    current_player: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute attack/defense scores for Gomoku
    
    This function analyzes the board for tactical patterns like:
    - Lines of 2, 3, 4 stones that can be extended
    - Blocking opponent's lines
    - Creating multiple threats
    
    Args:
        board: Board state with 0=empty, 1=player1, 2=player2
        current_player: Current player (0 or 1)
        
    Returns:
        Tuple of (attack_scores, defense_scores)
    """
    height, width = board.shape
    attack_scores = np.zeros((height, width), dtype=np.float32)
    defense_scores = np.zeros((height, width), dtype=np.float32)
    
    # Convert player ID to board values
    my_stone = current_player + 1
    opp_stone = 2 if current_player == 0 else 1
    
    # Directions: horizontal, vertical, diagonal, anti-diagonal
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    # For each empty position, calculate scores
    for i in range(height):
        for j in range(width):
            if board[i, j] == 0:  # Empty position
                attack_score = 0.0
                defense_score = 0.0
                
                # Check each direction
                for di, dj in directions:
                    # Attack: Check patterns we can create
                    my_pattern_score = evaluate_line_potential(
                        board, i, j, di, dj, my_stone
                    )
                    attack_score += my_pattern_score
                    
                    # Defense: Check opponent patterns we can block
                    opp_pattern_score = evaluate_line_potential(
                        board, i, j, di, dj, opp_stone
                    )
                    defense_score += opp_pattern_score
                
                attack_scores[i, j] = attack_score
                defense_scores[i, j] = defense_score
    
    # Normalize scores to [0, 1] range
    if attack_scores.max() > 0:
        attack_scores = attack_scores / attack_scores.max()
    if defense_scores.max() > 0:
        defense_scores = defense_scores / defense_scores.max()
    
    return attack_scores, defense_scores


def evaluate_line_potential(
    board: np.ndarray,
    row: int,
    col: int,
    di: int,
    dj: int,
    stone_type: int
) -> float:
    """Evaluate the potential of a line in one direction
    
    Args:
        board: Board state
        row, col: Position to evaluate
        di, dj: Direction vector
        stone_type: Type of stone to look for
        
    Returns:
        Score for this line direction
    """
    height, width = board.shape
    score = 0.0
    
    # Check line in both directions from the position
    for direction in [-1, 1]:
        stones_in_line = 0
        empty_ends = 0
        
        # Count stones in this direction
        for dist in range(1, 5):  # Check up to 4 positions away
            r = row + direction * dist * di
            c = col + direction * dist * dj
            
            if 0 <= r < height and 0 <= c < width:
                if board[r, c] == stone_type:
                    stones_in_line += 1
                elif board[r, c] == 0:
                    empty_ends += 1
                    break  # Stop at first empty
                else:
                    break  # Stop at opponent stone
            else:
                break  # Stop at board edge
        
        # Score based on number of stones and openness
        if stones_in_line >= 4:
            score += 1000.0  # Winning move
        elif stones_in_line == 3:
            score += 100.0 if empty_ends > 0 else 20.0
        elif stones_in_line == 2:
            score += 10.0 if empty_ends > 0 else 2.0
        elif stones_in_line == 1:
            score += 1.0 if empty_ends > 0 else 0.2
    
    return score


def compute_chess_attack_defense(
    board: np.ndarray,
    current_player: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute attack/defense scores for Chess
    
    This is a simplified implementation that could be expanded with:
    - Piece attack patterns
    - King safety
    - Tactical motifs
    
    Args:
        board: Chess board state
        current_player: Current player (0 or 1)
        
    Returns:
        Tuple of (attack_scores, defense_scores)
    """
    height, width = board.shape
    
    # For now, return uniform low scores
    # Real implementation would analyze piece attacks, pins, forks, etc.
    attack_scores = np.full((height, width), 0.1, dtype=np.float32)
    defense_scores = np.full((height, width), 0.1, dtype=np.float32)
    
    return attack_scores, defense_scores


def compute_go_attack_defense(
    board: np.ndarray,
    current_player: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute attack/defense scores for Go
    
    This could analyze:
    - Capture threats
    - Atari situations
    - Life and death patterns
    - Territory invasion/defense
    
    Args:
        board: Go board state
        current_player: Current player (0 or 1)
        
    Returns:
        Tuple of (attack_scores, defense_scores)
    """
    height, width = board.shape
    attack_scores = np.zeros((height, width), dtype=np.float32)
    defense_scores = np.zeros((height, width), dtype=np.float32)
    
    # Convert player ID to board values
    my_stone = current_player + 1
    opp_stone = 2 if current_player == 0 else 1
    
    # Check each empty position
    for i in range(height):
        for j in range(width):
            if board[i, j] == 0:  # Empty position
                attack_score = 0.0
                defense_score = 0.0
                
                # Check adjacent positions for capture opportunities
                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < height and 0 <= nj < width:
                        if board[ni, nj] == opp_stone:
                            # Check if this group has few liberties
                            liberties = count_liberties(board, ni, nj)
                            if liberties <= 2:
                                attack_score += 10.0 / liberties
                        elif board[ni, nj] == my_stone:
                            # Check if our group needs protection
                            liberties = count_liberties(board, ni, nj)
                            if liberties <= 2:
                                defense_score += 5.0 / liberties
                
                attack_scores[i, j] = attack_score
                defense_scores[i, j] = defense_score
    
    # Normalize scores
    if attack_scores.max() > 0:
        attack_scores = attack_scores / attack_scores.max()
    if defense_scores.max() > 0:
        defense_scores = defense_scores / defense_scores.max()
    
    return attack_scores, defense_scores


def count_liberties(board: np.ndarray, row: int, col: int) -> int:
    """Count liberties of a Go group starting from given position
    
    Args:
        board: Go board state
        row, col: Starting position
        
    Returns:
        Number of liberties for the group
    """
    height, width = board.shape
    stone_type = board[row, col]
    if stone_type == 0:
        return 0
    
    visited = set()
    liberties = set()
    
    def dfs(r, c):
        if (r, c) in visited or r < 0 or r >= height or c < 0 or c >= width:
            return
        
        if board[r, c] == 0:
            liberties.add((r, c))
            return
        
        if board[r, c] != stone_type:
            return
        
        visited.add((r, c))
        
        # Check all 4 directions
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            dfs(r + dr, c + dc)
    
    dfs(row, col)
    return len(liberties)


def batch_compute_attack_defense_scores(
    game_type: str,
    boards: np.ndarray,
    current_players: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute attack/defense scores for a batch of boards
    
    Args:
        game_type: Type of game
        boards: Batch of board states (batch_size, height, width)
        current_players: Current player for each board (batch_size,)
        
    Returns:
        Tuple of (attack_scores, defense_scores) with shape (batch_size, height, width)
    """
    batch_size, height, width = boards.shape
    attack_scores = np.zeros((batch_size, height, width), dtype=np.float32)
    defense_scores = np.zeros((batch_size, height, width), dtype=np.float32)
    
    for i in range(batch_size):
        attack_scores[i], defense_scores[i] = compute_attack_defense_scores(
            game_type, boards[i], current_players[i]
        )
    
    return attack_scores, defense_scores


# Export main function for compatibility
__all__ = [
    'compute_attack_defense_scores',
    'compute_gomoku_attack_defense', 
    'compute_chess_attack_defense',
    'compute_go_attack_defense',
    'batch_compute_attack_defense_scores'
]