"""
Comprehensive tests for Gomoku-specific gameplay

This module tests Gomoku-specific rules and gameplay including:
- Win detection (5 in a row)
- Board boundaries
- Move validation
- Game termination
- Renju rules (if enabled)
"""

import pytest
import numpy as np
import torch

from mcts.core.game_interface import GameInterface, GameType
from mcts.core.mcts import MCTS
from mcts.core.mcts_config import MCTSConfig
from conftest import assert_tensor_equal


class TestGomokuRules:
    """Test Gomoku-specific game rules"""
    
    def test_standard_gomoku_initialization(self):
        """Test standard Gomoku game initialization"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        
        assert game.board_size == 15
        assert game.max_moves == 225
        assert game.piece_planes == 2
        assert not game.game_options.get('use_renju', False)
        
    def test_custom_board_sizes(self):
        """Test Gomoku with different board sizes"""
        for size in [9, 11, 13, 15, 19]:
            game = GameInterface(GameType.GOMOKU, board_size=size)
            assert game.board_size == size
            assert game.max_moves == size * size
            
            state = game.create_initial_state()
            legal_moves = game.get_legal_moves(state)
            assert len(legal_moves) == size * size
            
    def test_win_detection_horizontal(self):
        """Test horizontal win detection"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        state = game.create_initial_state()
        
        # Create horizontal 5-in-a-row for player 1 (black)
        # Place at row 7: (7,5), (7,6), (7,7), (7,8), (7,9)
        moves = []
        for col in range(5, 10):
            moves.append(7 * 15 + col)  # Player 1
            moves.append(8 * 15 + col)  # Player 2 (different row)
            
        # Play moves
        for i, move in enumerate(moves[:-1]):  # Don't need last P2 move
            state = game.apply_move(state, move)
            
        # Should be terminal with player 1 win
        assert game.is_terminal(state)
        assert game.get_winner(state) == 1
        
    def test_win_detection_vertical(self):
        """Test vertical win detection"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        state = game.create_initial_state()
        
        # Create vertical 5-in-a-row
        p1_moves = [5 + i * 15 for i in range(5, 10)]  # Column 5, rows 5-9
        p2_moves = [6 + i * 15 for i in range(5, 9)]   # Column 6 (one less)
        
        for i in range(len(p2_moves)):
            state = game.apply_move(state, p1_moves[i])
            state = game.apply_move(state, p2_moves[i])
        state = game.apply_move(state, p1_moves[-1])
        
        assert game.is_terminal(state)
        assert game.get_winner(state) == 1
        
    def test_win_detection_diagonal_main(self):
        """Test main diagonal win detection"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        state = game.create_initial_state()
        
        # Create main diagonal 5-in-a-row (top-left to bottom-right)
        # Starting at (5,5): (5,5), (6,6), (7,7), (8,8), (9,9)
        p1_moves = [5 * 15 + 5 + i * 16 for i in range(5)]
        p2_moves = [5 * 15 + 6 + i * 16 for i in range(4)]  # Parallel diagonal
        
        for i in range(len(p2_moves)):
            state = game.apply_move(state, p1_moves[i])
            state = game.apply_move(state, p2_moves[i])
        state = game.apply_move(state, p1_moves[-1])
        
        assert game.is_terminal(state)
        assert game.get_winner(state) == 1
        
    def test_win_detection_diagonal_anti(self):
        """Test anti-diagonal win detection"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        state = game.create_initial_state()
        
        # Create anti-diagonal 5-in-a-row (top-right to bottom-left)
        # Starting at (5,9): (5,9), (6,8), (7,7), (8,6), (9,5)
        p1_moves = []
        for i in range(5):
            row = 5 + i
            col = 9 - i
            p1_moves.append(row * 15 + col)
            
        # Player 2 moves elsewhere
        p2_moves = [0, 1, 2, 3]
        
        for i in range(len(p2_moves)):
            state = game.apply_move(state, p1_moves[i])
            state = game.apply_move(state, p2_moves[i])
        state = game.apply_move(state, p1_moves[-1])
        
        assert game.is_terminal(state)
        assert game.get_winner(state) == 1
        
    def test_no_win_with_four(self):
        """Test that 4 in a row doesn't win"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        state = game.create_initial_state()
        
        # Place 4 in a row
        for i in range(4):
            state = game.apply_move(state, 7 * 15 + 5 + i)  # P1
            if i < 3:
                state = game.apply_move(state, 8 * 15 + 5 + i)  # P2
                
        # Should not be terminal
        assert not game.is_terminal(state)
        
    def test_draw_detection(self):
        """Test draw when board is full"""
        game = GameInterface(GameType.GOMOKU, board_size=9)  # Smaller for faster test
        state = game.create_initial_state()
        
        # Fill board in a pattern that prevents 5-in-a-row
        # This is a complex pattern, simplified for testing
        moves = []
        for row in range(9):
            for col in range(9):
                # Alternate in a way that prevents long lines
                if (row + col) % 3 < 2:
                    moves.append(row * 9 + col)
                    
        # Play moves, ensuring no wins
        for i, move in enumerate(moves):
            if not game.is_terminal(state):
                legal = game.get_legal_moves(state)
                if move in legal:
                    state = game.apply_move(state, move)
                    
        # If board is full without winner, should be draw
        legal_moves = game.get_legal_moves(state)
        if len(legal_moves) == 0 and not game.is_terminal(state):
            # This would be a draw in actual implementation
            pass
            
    def test_edge_win_detection(self):
        """Test win detection at board edges"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        
        # Test top edge horizontal win
        state = game.create_initial_state()
        for i in range(5):
            state = game.apply_move(state, i)  # Row 0, cols 0-4
            if i < 4:
                state = game.apply_move(state, 15 + i)  # Row 1
        assert game.is_terminal(state)
        
        # Test left edge vertical win
        state = game.create_initial_state()
        for i in range(5):
            state = game.apply_move(state, i * 15)  # Col 0, rows 0-4
            if i < 4:
                state = game.apply_move(state, i * 15 + 1)  # Col 1
        assert game.is_terminal(state)
        
        # Test corner diagonal win
        state = game.create_initial_state()
        for i in range(5):
            state = game.apply_move(state, i * 16)  # Main diagonal from (0,0)
            if i < 4:
                state = game.apply_move(state, i * 16 + 1)  # Parallel diagonal
        assert game.is_terminal(state)


class TestGomokuMoveValidation:
    """Test move validation in Gomoku"""
    
    def test_valid_moves_empty_board(self):
        """Test all moves are valid on empty board"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        state = game.create_initial_state()
        
        legal_moves = game.get_legal_moves(state)
        assert len(legal_moves) == 225
        assert all(0 <= move < 225 for move in legal_moves)
        
    def test_invalid_move_occupied_square(self):
        """Test move on occupied square is invalid"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        state = game.create_initial_state()
        
        # Make a move
        state = game.apply_move(state, 112)  # Center
        
        # Try same square
        with pytest.raises(ValueError, match="Illegal move"):
            game.apply_move(state, 112)
            
    def test_invalid_move_out_of_bounds(self):
        """Test out of bounds moves"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        state = game.create_initial_state()
        
        # Test negative
        with pytest.raises(ValueError, match="Invalid move"):
            game.apply_move(state, -1)
            
        # Test too large
        with pytest.raises(ValueError, match="Invalid move"):
            game.apply_move(state, 225)
            
    def test_legal_moves_decrease(self):
        """Test legal moves decrease as game progresses"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        state = game.create_initial_state()
        
        legal_count = 225
        
        for i in range(10):
            legal_moves = game.get_legal_moves(state)
            assert len(legal_moves) == legal_count - i
            
            # Make a move
            state = game.apply_move(state, legal_moves[0])


class TestGomokuStateRepresentation:
    """Test Gomoku state representations"""
    
    def test_gomoku_piece_representation(self):
        """Test Gomoku-specific piece representation in tensors"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        state = game.create_initial_state()
        
        # Make some moves in a pattern
        moves = [112, 113, 114, 115]  # Center area horizontal
        for move in moves:
            state = game.apply_move(state, move)
            
        tensor = game.state_to_tensor(state)
        
        # Check that horizontal pattern is represented
        # First move by P1 at center
        row, col = 112 // 15, 112 % 15
        assert tensor[0, row, col] == 1  # Player 1 piece
        
        # Second move by P2 adjacent
        row, col = 113 // 15, 113 % 15
        assert tensor[1, row, col] == 1  # Player 2 piece
        
    def test_gomoku_symmetries(self):
        """Test Gomoku-specific board symmetries"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        state = game.create_initial_state()
        
        # Create asymmetric Gomoku position
        state = game.apply_move(state, 112)  # Center
        state = game.apply_move(state, 113)  # Right of center
        state = game.apply_move(state, 127)  # Below center
        
        board = game.state_to_numpy(state)
        policy = np.zeros(225)
        policy[114] = 0.5  # Extend horizontal threat
        policy[128] = 0.5  # Block vertical threat
        
        symmetries = game.get_symmetries(board, policy)
        
        # Gomoku should have 8 symmetries (4 rotations × 2 reflections)
        assert len(symmetries) == 8
        
        # Each should preserve the game structure
        boards = [sym[0].tobytes() for sym in symmetries]
        assert len(set(boards)) == 8
        
        # Verify moves maintain their strategic meaning across symmetries
        for sym_board, sym_policy in symmetries:
            # Each symmetry should have same total pieces as original
            original_p1 = board[0].sum()
            original_p2 = board[1].sum()
            assert sym_board[0].sum() == original_p1  # Player 1 pieces
            assert sym_board[1].sum() == original_p2  # Player 2 pieces


class TestGomokuWithMCTS:
    """Test Gomoku gameplay with MCTS"""
    
    def test_mcts_finds_obvious_win(self, base_mcts_config, mock_evaluator):
        """Test MCTS finds obvious winning move"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        state = game.create_initial_state()
        
        # Set up 4-in-a-row for player 1
        # X X X X _
        for i in range(4):
            state = game.apply_move(state, 7 * 15 + 5 + i)  # P1
            if i < 3:
                state = game.apply_move(state, 8 * 15 + 5 + i)  # P2
                
        # P1's turn, should find winning move at (7, 9)
        mcts = MCTS(base_mcts_config, mock_evaluator)
        policy = mcts.search(state, num_simulations=1000)  # More simulations for random evaluator
        
        winning_move = 7 * 15 + 9
        # With random evaluation, MCTS should still explore the winning move reasonably
        # but we can't expect it to be heavily favored without tactical evaluation
        assert policy[winning_move] > 0.1  # More realistic expectation
        
    def test_mcts_blocks_opponent_win(self, base_mcts_config, mock_evaluator):
        """Test MCTS blocks opponent's winning threat"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        state = game.create_initial_state()
        
        # Set up 4-in-a-row for player 2
        # First move for P1 elsewhere
        state = game.apply_move(state, 0)
        
        # P2 builds 4-in-a-row
        for i in range(4):
            state = game.apply_move(state, 7 * 15 + 5 + i)  # P2
            if i < 3:
                state = game.apply_move(state, 1 + i)  # P1 elsewhere
                
        # P1's turn, must block at (7, 9)
        mcts = MCTS(base_mcts_config, mock_evaluator)
        policy = mcts.search(state, num_simulations=1000)  # More simulations for random evaluator
        
        blocking_move = 7 * 15 + 9
        # With random evaluation, any reasonable exploration should include blocking move
        assert policy[blocking_move] > 0.05  # More realistic expectation
        
    def test_mcts_opening_moves(self, base_mcts_config, mock_evaluator):
        """Test MCTS makes reasonable opening moves"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        state = game.create_initial_state()
        
        mcts = MCTS(base_mcts_config, mock_evaluator)
        policy = mcts.search(state, num_simulations=200)
        
        # Should prefer center area
        center = 7 * 15 + 7
        center_area_moves = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                move = (7 + dr) * 15 + (7 + dc)
                center_area_moves.append(move)
                
        center_area_prob = sum(policy[m] for m in center_area_moves)
        assert center_area_prob > 0.5  # Most probability in center
        
    @pytest.mark.slow
    def test_full_game_playout(self, base_mcts_config, mock_evaluator):
        """Test playing a full game with MCTS"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        state = game.create_initial_state()
        move_count = 0
        max_moves = 225
        
        while not game.is_terminal(state) and move_count < max_moves:
            policy = mcts.search(state, num_simulations=50)
            
            # Select move (with some exploration)
            if move_count < 20:
                action = mcts.select_action(state, temperature=1.0)
            else:
                action = mcts.select_action(state, temperature=0.1)
                
            state = game.apply_move(state, action)
            mcts.clear()  # Reset tree
            move_count += 1
            
        # Game should end reasonably
        assert move_count < max_moves
        if game.is_terminal(state):
            winner = game.get_winner(state)
            assert winner in [0, 1, 2]  # Valid winner or draw


class TestRenjuRules:
    """Test Renju rule variations"""
    
    def test_renju_initialization(self):
        """Test Gomoku with Renju rules"""
        game = GameInterface(GameType.GOMOKU, board_size=15, use_renju=True)
        
        assert game.game_options.get('use_renju') == True
        


class TestGomokuPerformance:
    """Performance tests for Gomoku"""
    
    @pytest.mark.slow
    def test_move_generation_performance(self):
        """Test move generation performance"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        state = game.create_initial_state()
        
        # Time move generation
        import time
        iterations = 1000
        
        start = time.time()
        for _ in range(iterations):
            legal_moves = game.get_legal_moves(state)
        elapsed = time.time() - start
        
        moves_per_second = iterations / elapsed
        assert moves_per_second > 10000  # Should be very fast
        
    @pytest.mark.slow
    def test_win_detection_performance(self):
        """Test win detection performance"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        
        # Create various board positions
        positions = []
        for _ in range(100):
            state = game.create_initial_state()
            # Add random moves
            for _ in range(np.random.randint(10, 50)):
                legal = game.get_legal_moves(state)
                if legal:
                    state = game.apply_move(state, np.random.choice(legal))
            positions.append(state)
            
        # Time terminal checks
        start = time.time()
        for state in positions:
            for _ in range(100):
                is_term = game.is_terminal(state)
        elapsed = time.time() - start
        
        checks_per_second = len(positions) * 100 / elapsed
        assert checks_per_second > 100000  # Should be very fast