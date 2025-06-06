"""Pure Python implementation of attack/defense score computation

This module provides a fallback implementation for computing attack and defense
scores when C++ bindings are not available.
"""

import numpy as np
from typing import Tuple, List, Any
import logging

logger = logging.getLogger(__name__)


def compute_gomoku_attack_defense_scores(board: np.ndarray, current_player: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute attack and defense scores for Gomoku
    
    Args:
        board: Board state array of shape (height, width)
        current_player: Current player (0 or 1)
        
    Returns:
        Tuple of (attack_scores, defense_scores) arrays
    """
    height, width = board.shape
    attack_scores = np.zeros((height, width), dtype=np.float32)
    defense_scores = np.zeros((height, width), dtype=np.float32)
    
    # Directions for checking lines (horizontal, vertical, diagonal1, diagonal2)
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    # Pattern values for different configurations
    pattern_values = {
        'five': 100000,      # Win
        'open_four': 10000,  # Guaranteed win next move
        'four': 1000,        # Blocked four
        'open_three': 500,   # Strong threat
        'three': 100,        # Blocked three
        'open_two': 50,      # Potential
        'two': 10,           # Blocked two
    }
    
    # Check each empty position
    for i in range(height):
        for j in range(width):
            if board[i, j] != 0:  # Skip occupied positions
                continue
                
            # Evaluate placing current player's stone (attack)
            attack_score = evaluate_position(board, i, j, current_player + 1, directions, pattern_values)
            attack_scores[i, j] = attack_score
            
            # Evaluate blocking opponent's stone (defense)
            opponent = 2 if current_player == 0 else 1
            defense_score = evaluate_position(board, i, j, opponent, directions, pattern_values)
            defense_scores[i, j] = defense_score
    
    # Normalize scores to [0, 1] range
    if attack_scores.max() > 0:
        attack_scores = attack_scores / attack_scores.max()
    if defense_scores.max() > 0:
        defense_scores = defense_scores / defense_scores.max()
        
    return attack_scores, defense_scores


def evaluate_position(board: np.ndarray, row: int, col: int, player: int, 
                     directions: List[Tuple[int, int]], pattern_values: dict) -> float:
    """Evaluate the value of placing a stone at a position
    
    Args:
        board: Board state
        row, col: Position to evaluate
        player: Player to evaluate for (1 or 2)
        directions: List of direction vectors
        pattern_values: Dictionary of pattern values
        
    Returns:
        Score for the position
    """
    height, width = board.shape
    total_score = 0.0
    
    for dr, dc in directions:
        # Count stones in both directions
        count = 1  # The stone we're placing
        space_before = 0
        space_after = 0
        blocked_before = False
        blocked_after = False
        
        # Check positive direction
        r, c = row + dr, col + dc
        while 0 <= r < height and 0 <= c < width:
            if board[r, c] == player:
                count += 1
                r += dr
                c += dc
            elif board[r, c] == 0:
                space_after += 1
                if space_after > 2:  # Don't look too far
                    break
                r += dr
                c += dc
            else:  # Opponent stone
                blocked_after = True
                break
                
        # Check negative direction
        r, c = row - dr, col - dc
        while 0 <= r < height and 0 <= c < width:
            if board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            elif board[r, c] == 0:
                space_before += 1
                if space_before > 2:
                    break
                r -= dr
                c -= dc
            else:  # Opponent stone
                blocked_before = True
                break
                
        # Evaluate the pattern
        if count >= 5:
            total_score += pattern_values['five']
        elif count == 4:
            if not blocked_before and not blocked_after:
                total_score += pattern_values['open_four']
            else:
                total_score += pattern_values['four']
        elif count == 3:
            if not blocked_before and not blocked_after and space_before + space_after >= 2:
                total_score += pattern_values['open_three']
            else:
                total_score += pattern_values['three']
        elif count == 2:
            if not blocked_before and not blocked_after and space_before + space_after >= 3:
                total_score += pattern_values['open_two']
            else:
                total_score += pattern_values['two']
                
    return total_score


def compute_chess_attack_defense_scores(board: np.ndarray, current_player: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute attack and defense scores for Chess
    
    For chess, this would involve checking piece attacks, pins, forks, etc.
    This is a placeholder implementation.
    
    Args:
        board: Board state array
        current_player: Current player (0 or 1)
        
    Returns:
        Tuple of (attack_scores, defense_scores) arrays
    """
    # Placeholder - return zeros
    # Real implementation would analyze piece positions and threats
    height, width = 8, 8
    return np.zeros((height, width), dtype=np.float32), np.zeros((height, width), dtype=np.float32)


def compute_go_attack_defense_scores(board: np.ndarray, current_player: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute attack and defense scores for Go
    
    For Go, this would involve analyzing groups, liberties, territory, etc.
    This is a placeholder implementation.
    
    Args:
        board: Board state array
        current_player: Current player (0 or 1)
        
    Returns:
        Tuple of (attack_scores, defense_scores) arrays
    """
    # Placeholder - return zeros
    # Real implementation would analyze group strength, territory control, etc.
    height, width = board.shape[:2]
    return np.zeros((height, width), dtype=np.float32), np.zeros((height, width), dtype=np.float32)


def compute_attack_defense_scores(game_type: str, board: np.ndarray, current_player: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute attack and defense scores for any game type
    
    Args:
        game_type: Type of game ('chess', 'go', 'gomoku')
        board: Board state array
        current_player: Current player (0 or 1)
        
    Returns:
        Tuple of (attack_scores, defense_scores) arrays
    """
    if game_type == 'gomoku':
        return compute_gomoku_attack_defense_scores(board, current_player)
    elif game_type == 'chess':
        return compute_chess_attack_defense_scores(board, current_player)
    elif game_type == 'go':
        return compute_go_attack_defense_scores(board, current_player)
    else:
        # Unknown game type - return zeros
        height, width = board.shape[:2]
        return np.zeros((height, width), dtype=np.float32), np.zeros((height, width), dtype=np.float32)