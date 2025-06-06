"""Tests for attack/defense score computation"""

import pytest
import numpy as np
from mcts.utils.attack_defense import (
    compute_gomoku_attack_defense_scores,
    compute_attack_defense_scores,
    evaluate_position
)


class TestAttackDefense:
    """Test attack/defense computation"""
    
    def test_gomoku_empty_board(self):
        """Test scores on empty board"""
        board = np.zeros((15, 15), dtype=int)
        current_player = 0
        
        attack_scores, defense_scores = compute_gomoku_attack_defense_scores(board, current_player)
        
        # On empty board, all positions should have equal low scores
        assert attack_scores.shape == (15, 15)
        assert defense_scores.shape == (15, 15)
        assert np.all(attack_scores >= 0)
        assert np.all(defense_scores >= 0)
        assert np.all(attack_scores <= 1)
        assert np.all(defense_scores <= 1)
        
    def test_gomoku_win_threat(self):
        """Test detection of win threats"""
        board = np.zeros((15, 15), dtype=int)
        # Create a line of 4 for player 1
        board[7, 5:9] = 1  # Four in a row
        current_player = 0  # Player 1's turn
        
        attack_scores, defense_scores = compute_gomoku_attack_defense_scores(board, current_player)
        
        # Positions that complete the five should have high attack scores
        assert attack_scores[7, 4] > 0.8  # Left side
        assert attack_scores[7, 9] > 0.8  # Right side
        
    def test_gomoku_defense_priority(self):
        """Test detection of defensive moves"""
        board = np.zeros((15, 15), dtype=int)
        # Create a line of 4 for player 2 (opponent)
        board[7, 5:9] = 2  # Four in a row for opponent
        current_player = 0  # Player 1's turn
        
        attack_scores, defense_scores = compute_gomoku_attack_defense_scores(board, current_player)
        
        # Positions that block the opponent should have high defense scores
        assert defense_scores[7, 4] > 0.8  # Left side block
        assert defense_scores[7, 9] > 0.8  # Right side block
        
    def test_gomoku_open_three(self):
        """Test detection of open three patterns"""
        board = np.zeros((15, 15), dtype=int)
        # Create an open three for player 1
        board[7, [5, 6, 7]] = 1  # Three in a row
        current_player = 0
        
        attack_scores, defense_scores = compute_gomoku_attack_defense_scores(board, current_player)
        
        # Extending to open four should have good scores
        if board[7, 4] == 0 and board[7, 8] == 0:  # Both ends open
            assert attack_scores[7, 4] > 0.3
            assert attack_scores[7, 8] > 0.3
            
    def test_diagonal_patterns(self):
        """Test diagonal pattern detection"""
        board = np.zeros((15, 15), dtype=int)
        # Create diagonal line
        for i in range(3):
            board[5+i, 5+i] = 1
        current_player = 0
        
        attack_scores, defense_scores = compute_gomoku_attack_defense_scores(board, current_player)
        
        # Check diagonal extensions
        if board[4, 4] == 0:
            assert attack_scores[4, 4] > 0.1
        if board[8, 8] == 0:
            assert attack_scores[8, 8] > 0.1
            
    def test_normalize_scores(self):
        """Test that scores are normalized properly"""
        board = np.zeros((15, 15), dtype=int)
        # Create multiple patterns
        board[7, 5:8] = 1  # Three in a row
        board[10, 5:9] = 2  # Four in a row for opponent
        current_player = 0
        
        attack_scores, defense_scores = compute_gomoku_attack_defense_scores(board, current_player)
        
        # All scores should be in [0, 1]
        assert np.all(attack_scores >= 0)
        assert np.all(attack_scores <= 1)
        assert np.all(defense_scores >= 0)
        assert np.all(defense_scores <= 1)
        
        # Should have at least one maximum score
        assert np.max(defense_scores) == 1.0  # Blocking four-in-a-row
        
    def test_generic_interface(self):
        """Test generic compute function"""
        board = np.zeros((15, 15), dtype=int)
        
        # Test Gomoku
        attack, defense = compute_attack_defense_scores('gomoku', board, 0)
        assert attack.shape == (15, 15)
        assert defense.shape == (15, 15)
        
        # Test Chess (placeholder)
        chess_board = np.zeros((8, 8), dtype=int)
        attack, defense = compute_attack_defense_scores('chess', chess_board, 0)
        assert attack.shape == (8, 8)
        assert defense.shape == (8, 8)
        
        # Test Go (placeholder)
        go_board = np.zeros((19, 19), dtype=int)
        attack, defense = compute_attack_defense_scores('go', go_board, 0)
        assert attack.shape == (19, 19)
        assert defense.shape == (19, 19)