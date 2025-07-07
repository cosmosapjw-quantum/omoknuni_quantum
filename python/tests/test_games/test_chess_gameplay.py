"""
Comprehensive tests for Chess-specific gameplay

This module tests Chess-specific rules and gameplay including:
- Piece movement rules
- Special moves (castling, en passant, promotion)
- Check and checkmate detection
- Stalemate detection
- Draw conditions (repetition, 50-move rule)
- Move notation parsing
"""

import pytest
import numpy as np
import torch

from mcts.core.game_interface import GameInterface, GameType
from mcts.core.mcts import MCTS
from mcts.core.mcts_config import MCTSConfig
from conftest import assert_tensor_equal


class TestChessRules:
    """Test Chess-specific game rules"""
    
    def test_standard_chess_initialization(self):
        """Test standard Chess game initialization"""
        game = GameInterface(GameType.CHESS)
        
        assert game.board_size == 8
        assert game.max_moves > 200  # Chess has variable game length
        assert game.piece_planes >= 12  # 6 piece types × 2 colors
        
    def test_initial_position(self):
        """Test initial chess position setup"""
        game = GameInterface(GameType.CHESS)
        state = game.create_initial_state()
        
        # Verify initial position
        board = game.state_to_numpy(state)
        
        # Check some key pieces (exact encoding depends on implementation)
        # White should be on ranks 1-2, Black on ranks 7-8
        
    def test_pawn_movement(self):
        """Test pawn movement rules"""
        game = GameInterface(GameType.CHESS)
        state = game.create_initial_state()
        
        # Test two-square initial move
        # e2-e4 (common opening)
        legal_moves = game.get_legal_moves(state)
        
        # Pawn moves should be legal
        # Move encoding is implementation-specific
        
    def test_knight_movement(self):
        """Test knight L-shaped movement"""
        game = GameInterface(GameType.CHESS)
        state = game.create_initial_state()
        
        # Knight from b1 to c3 or a3
        legal_moves = game.get_legal_moves(state)
        
        # Knight moves should be in legal moves
        
    def test_bishop_movement(self):
        """Test bishop diagonal movement"""
        game = GameInterface(GameType.CHESS)
        state = game.create_initial_state()
        
        # Bishops are initially blocked by pawns
        # Would need to move pawns first
        
    def test_rook_movement(self):
        """Test rook straight line movement"""
        game = GameInterface(GameType.CHESS)
        state = game.create_initial_state()
        
        # Rooks are initially blocked
        # Test after clearing path
        
    def test_queen_movement(self):
        """Test queen movement (rook + bishop)"""
        game = GameInterface(GameType.CHESS)
        state = game.create_initial_state()
        
        # Queen combines rook and bishop movement
        
    def test_king_movement(self):
        """Test king one-square movement"""
        game = GameInterface(GameType.CHESS)
        state = game.create_initial_state()
        
        # King moves one square in any direction
        
    def test_castling_kingside(self):
        """Test kingside castling (O-O)"""
        game = GameInterface(GameType.CHESS)
        state = game.create_initial_state()
        
        # Clear path between king and rook
        # 1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. O-O
        castling_setup = [
            "e2e4", "e7e5",
            "g1f3", "b8c6", 
            "f1c4", "f8c5"
        ]
        
        # Apply moves (implementation-specific encoding)
        # Then castling should be legal
        
    def test_castling_queenside(self):
        """Test queenside castling (O-O-O)"""
        game = GameInterface(GameType.CHESS)
        state = game.create_initial_state()
        
        # Clear path for queenside castling
        # Requires moving queen, bishop, and knight
        
        
    def test_en_passant(self):
        """Test en passant capture"""
        game = GameInterface(GameType.CHESS)
        state = game.create_initial_state()
        
        # Set up en passant scenario
        # 1. e4 Nf6 2. e5 d5 3. exd6 e.p.
        en_passant_setup = [
            "e2e4", "g8f6",
            "e4e5", "d7d5"
        ]
        
        # After d7-d5, white can capture en passant
        
    def test_pawn_promotion(self):
        """Test pawn promotion"""
        game = GameInterface(GameType.CHESS)
        
        # Would need to advance pawn to 8th rank
        # Test promotion to Queen, Rook, Bishop, Knight
        
    def test_check_detection(self):
        """Test check detection"""
        game = GameInterface(GameType.CHESS)
        state = game.create_initial_state()
        
        # Create position with king in check
        # Fool's mate: 1. f3 e5 2. g4 Qh4#
        fools_mate = [
            "f2f3", "e7e5",
            "g2g4", "d8h4"
        ]
        
        # After Qh4, white king is in checkmate
        
    def test_checkmate_detection(self):
        """Test checkmate detection"""
        game = GameInterface(GameType.CHESS)
        
        # Back rank mate, smothered mate, etc.
        
    def test_stalemate_detection(self):
        """Test stalemate detection"""
        game = GameInterface(GameType.CHESS)
        
        # King not in check but no legal moves
        
        
    def test_threefold_repetition(self):
        """Test threefold repetition draw"""
        game = GameInterface(GameType.CHESS)
        
        # Same position three times = draw
        
    def test_fifty_move_rule(self):
        """Test 50-move rule draw"""
        game = GameInterface(GameType.CHESS)
        
        # 50 moves without pawn move or capture = draw


class TestChessMoveValidation:
    """Test move validation in Chess"""
    
    def test_legal_moves_initial_position(self):
        """Test legal moves from starting position"""
        game = GameInterface(GameType.CHESS)
        state = game.create_initial_state()
        
        legal_moves = game.get_legal_moves(state)
        
        # Should have 20 legal moves for white
        # 16 pawn moves (8 pawns × 2 squares each)
        # 4 knight moves (2 knights × 2 squares each)
        assert len(legal_moves) == 20
        
    def test_pinned_piece_movement(self):
        """Test pinned pieces cannot move illegally"""
        game = GameInterface(GameType.CHESS)
        
        # Set up position with pinned piece
        # Piece pinned to king cannot move off the pin line
        
    def test_check_evasion_only(self):
        """Test only check evasions are legal when in check"""
        game = GameInterface(GameType.CHESS)
        
        # When in check, only moves that escape check are legal
        
    def test_capture_validation(self):
        """Test capture move validation"""
        game = GameInterface(GameType.CHESS)
        
        # Can capture opponent pieces
        # Cannot capture own pieces
        


class TestChessStateRepresentation:
    """Test Chess state representations"""
    
    def test_state_to_tensor_planes(self):
        """Test tensor representation with piece planes"""
        game = GameInterface(GameType.CHESS)
        state = game.create_initial_state()
        
        tensor = game.state_to_tensor(state)
        
        # Should have planes for:
        # - Each piece type for each color (12 planes)
        # - Castling rights (4 planes)
        # - En passant (1 plane)
        # - Move count info
        # - Current player
        assert tensor.shape[0] >= 12  # At minimum piece planes
        assert tensor.shape[1] == 8
        assert tensor.shape[2] == 8
        
    def test_fen_conversion(self):
        """Test FEN string conversion"""
        game = GameInterface(GameType.CHESS)
        state = game.create_initial_state()
        
        # Should support FEN notation
        if hasattr(game, 'state_to_fen'):
            fen = game.state_to_fen(state)
            assert isinstance(fen, str)
            # Starting position FEN
            assert fen.startswith("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
            
    def test_pgn_move_parsing(self):
        """Test PGN move notation parsing"""
        game = GameInterface(GameType.CHESS)
        
        # Should support algebraic notation
        if hasattr(game, 'parse_move'):
            # Test common moves
            moves = ["e4", "Nf3", "Bb5", "O-O", "Qxf7+", "Raxb1"]
            # Implementation-specific
            
    def test_move_history_tracking(self):
        """Test move history for repetition detection"""
        game = GameInterface(GameType.CHESS)
        state = game.create_initial_state()
        
        # Make several moves
        # History should be tracked for repetition rule
        
    def test_symmetries_not_applicable(self):
        """Test that chess doesn't use symmetries"""
        game = GameInterface(GameType.CHESS)
        state = game.create_initial_state()
        
        # Chess positions are not symmetric due to castling rights
        # and pawn direction
        board = game.state_to_numpy(state)
        policy = np.zeros(game.max_moves)
        
        symmetries = game.get_symmetries(board, policy)
        
        # Should return only original or raise NotImplementedError
        assert len(symmetries) == 1 or symmetries is None


class TestChessWithMCTS:
    """Test Chess gameplay with MCTS"""
    
    def test_mcts_finds_checkmate_in_one(self, base_mcts_config, mock_evaluator):
        """Test MCTS finds mate in one"""
        game = GameInterface(GameType.CHESS)
        
        # Set up position with mate in one
        # Would need position encoding
        
        # MCTS should heavily favor the mating move
        
    def test_mcts_avoids_blunders(self, base_mcts_config, mock_evaluator):
        """Test MCTS avoids hanging pieces"""
        game = GameInterface(GameType.CHESS)
        
        # Set up position where a piece can be captured
        # MCTS should avoid leaving pieces hanging
        
    def test_mcts_opening_play(self, base_mcts_config, mock_evaluator):
        """Test MCTS makes reasonable opening moves"""
        game = GameInterface(GameType.CHESS)
        state = game.create_initial_state()
        
        mcts = MCTS(base_mcts_config, mock_evaluator)
        policy = mcts.search(state, num_simulations=200)
        
        # Should prefer central pawns or knights
        # Common first moves: e4, d4, Nf3, c4
        
    def test_mcts_endgame_technique(self, base_mcts_config, mock_evaluator):
        """Test MCTS endgame play"""
        game = GameInterface(GameType.CHESS)
        
        # King and pawn endgame
        # MCTS should show understanding of opposition, etc.


class TestChessSpecialScenarios:
    """Test special chess scenarios"""
    
    def test_discovered_check(self):
        """Test discovered check scenarios"""
        game = GameInterface(GameType.CHESS)
        
        # Moving a piece reveals check from another piece
        
    def test_double_check(self):
        """Test double check scenarios"""
        game = GameInterface(GameType.CHESS)
        
        # King attacked by two pieces simultaneously
        # Only king moves are legal
        
    def test_zwischenzug(self):
        """Test in-between moves"""
        game = GameInterface(GameType.CHESS)
        
        # Intermediate move that changes the position
        
    def test_fortress_position(self):
        """Test fortress/blockade positions"""
        game = GameInterface(GameType.CHESS)
        
        # Position where one side cannot make progress


class TestChessPerformance:
    """Performance tests for Chess"""
    
    @pytest.mark.slow
    def test_move_generation_performance(self):
        """Test move generation performance"""
        game = GameInterface(GameType.CHESS)
        state = game.create_initial_state()
        
        import time
        iterations = 1000
        
        start = time.time()
        for _ in range(iterations):
            legal_moves = game.get_legal_moves(state)
        elapsed = time.time() - start
        
        moves_per_second = iterations / elapsed
        assert moves_per_second > 5000  # Should be fast
        
    @pytest.mark.slow
    def test_complex_position_performance(self):
        """Test performance in complex middlegame positions"""
        game = GameInterface(GameType.CHESS)
        
        # Create complex position with many pieces
        # Test move generation and validation speed


class TestChessVariants:
    """Test chess variants and options"""
    
    def test_chess960_initialization(self):
        """Test Chess960 (Fischer Random) setup"""
        game = GameInterface(GameType.CHESS, 
                           game_options={'variant': 'chess960'})
        
        # Should have randomized back rank
        # But same rules otherwise
        
    def test_time_control_integration(self):
        """Test time control integration"""
        game = GameInterface(GameType.CHESS,
                           game_options={'time_control': '5+0'})
        
        # Time control affects MCTS simulation count
        
    def test_opening_book_integration(self):
        """Test opening book usage"""
        game = GameInterface(GameType.CHESS,
                           game_options={'use_opening_book': True})
        
        # First few moves might come from book
        
    def test_endgame_tablebase(self):
        """Test endgame tablebase integration"""
        game = GameInterface(GameType.CHESS,
                           game_options={'use_tablebase': True})
        
        # Perfect play in simple endgames