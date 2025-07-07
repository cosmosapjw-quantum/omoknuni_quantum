"""
Tactical move detection for Chess to boost important moves during MCTS expansion.

This module helps MCTS find critical tactical moves (captures, checks, threats)
even with an untrained neural network, preventing training bias.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional


class ChessTacticalMoveDetector:
    """Detects tactical moves in Chess to help MCTS exploration"""
    
    def __init__(self, config=None):
        self.config = config
        
        # Piece values for material evaluation
        self.piece_values = {
            'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0,  # lowercase = black
            'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0   # uppercase = white
        }
        
        # Board coordinates
        self.files = 'abcdefgh'
        self.ranks = '12345678'
        
        # Load tactical parameters from config or use defaults
        if config:
            self.capture_good_base = getattr(config, 'chess_capture_good_base', 10.0)
            self.capture_equal = getattr(config, 'chess_capture_equal', 5.0)
            self.capture_bad = getattr(config, 'chess_capture_bad', 2.0)
            self.check_boost = getattr(config, 'chess_check_boost', 8.0)
            self.check_capture_boost = getattr(config, 'chess_check_capture_boost', 4.0)
            self.promotion_boost = getattr(config, 'chess_promotion_boost', 9.0)
            self.center_boost = getattr(config, 'chess_center_boost', 1.0)
            self.center_core_boost = getattr(config, 'chess_center_core_boost', 0.5)
            self.development_boost = getattr(config, 'chess_development_boost', 2.0)
            self.castling_boost = getattr(config, 'chess_castling_boost', 6.0)
        else:
            # Default values
            self.capture_good_base = 10.0
            self.capture_equal = 5.0
            self.capture_bad = 2.0
            self.check_boost = 8.0
            self.check_capture_boost = 4.0
            self.promotion_boost = 9.0
            self.center_boost = 1.0
            self.center_core_boost = 0.5
            self.development_boost = 2.0
            self.castling_boost = 6.0
        
    def detect_tactical_moves(self, board: torch.Tensor, current_player: int, 
                             legal_moves: Optional[List[int]] = None) -> torch.Tensor:
        """
        Detect tactical moves and return a priority boost for each legal move.
        
        Args:
            board: Board tensor of shape (8, 8) with piece encodings
            current_player: Current player (1 = white, 2 = black)
            legal_moves: List of legal move indices
            
        Returns:
            Priority boost tensor of shape (4096,) for all possible moves
            Higher values indicate more important tactical moves
        """
        boost = torch.zeros(4096)  # Max possible moves in chess
        
        # Convert board to numpy for easier manipulation
        board_np = board.cpu().numpy() if isinstance(board, torch.Tensor) else board
        
        # If no legal moves provided, generate a reasonable set for testing
        if legal_moves is None:
            # Generate moves for pieces that exist on the board
            legal_moves = []
            for from_sq in range(64):
                from_row, from_col = from_sq // 8, from_sq % 8
                piece = int(board_np[from_row, from_col])
                if piece != 0:
                    # Add some reasonable moves for this piece
                    for to_sq in range(64):
                        if from_sq != to_sq:
                            legal_moves.append(from_sq * 64 + to_sq)
            # Limit to reasonable number for testing
            legal_moves = legal_moves[:1000]
        
        # For each legal move, evaluate its tactical importance
        for move_idx in legal_moves:
            # Decode move (assuming standard encoding: from_square * 64 + to_square)
            from_sq = move_idx // 64
            to_sq = move_idx % 64
            
            from_row, from_col = from_sq // 8, from_sq % 8
            to_row, to_col = to_sq // 8, to_sq % 8
            
            # Skip invalid coordinates
            if not (0 <= from_row < 8 and 0 <= from_col < 8 and 
                    0 <= to_row < 8 and 0 <= to_col < 8):
                continue
            
            # Get piece at from square
            piece = int(board_np[from_row, from_col])
            if piece == 0:
                continue
                
            # Determine if this is our piece
            is_white_piece = piece > 0 and piece <= 6  # White pieces: 1-6
            is_black_piece = piece > 6 and piece <= 12  # Black pieces: 7-12
            
            if current_player == 1 and not is_white_piece:
                continue
            if current_player == 2 and not is_black_piece:
                continue
            
            move_boost = 0.0
            
            # 1. Captures - highest priority
            target_piece = int(board_np[to_row, to_col])
            if target_piece != 0:
                # Evaluate capture value
                capture_value = self._get_piece_value(target_piece)
                attacker_value = self._get_piece_value(piece)
                
                # Good captures (winning material)
                if capture_value > attacker_value:
                    move_boost += self.capture_good_base + capture_value - attacker_value
                elif capture_value == attacker_value:
                    move_boost += self.capture_equal  # Equal trade
                else:
                    # Bad capture, but still worth considering
                    move_boost += self.capture_bad
            
            # 2. Checks - very important
            if self._is_check_move(board_np, from_row, from_col, to_row, to_col, current_player):
                move_boost += self.check_boost
                
                # Double check if it's also a capture (discovered check)
                if target_piece != 0:
                    move_boost += self.check_capture_boost
            
            # 3. Pawn promotion
            piece_type = int((piece - 1) % 6 + 1)  # Normalize to 1-6
            if piece_type == 1:  # Pawn
                if (current_player == 1 and to_row == 7) or (current_player == 2 and to_row == 0):
                    move_boost += self.promotion_boost  # Promotion is very valuable
            
            # 4. Central control (for early game)
            if 2 <= to_row <= 5 and 2 <= to_col <= 5:
                move_boost += self.center_boost
                if 3 <= to_row <= 4 and 3 <= to_col <= 4:
                    move_boost += self.center_core_boost  # Extra bonus for center squares
            
            # 5. Developing pieces (moving from back rank)
            if piece_type in [2, 3]:  # Knight or Bishop
                if (current_player == 1 and from_row == 0) or (current_player == 2 and from_row == 7):
                    move_boost += self.development_boost
            
            # 6. Castling
            if piece_type == 6:  # King
                if abs(to_col - from_col) == 2:  # Castling move
                    move_boost += self.castling_boost
            
            boost[move_idx] = move_boost
            
        return boost
    
    def boost_prior_with_tactics(self, prior: torch.Tensor, board: torch.Tensor,
                                current_player: int, boost_strength: float = 0.3,
                                legal_moves: Optional[List[int]] = None) -> torch.Tensor:
        """
        Boost prior probabilities based on tactical importance.
        
        Args:
            prior: Original prior probabilities
            board: Current board state
            current_player: Current player
            boost_strength: How much to boost tactical moves (0.0 = no boost, 1.0 = full replacement)
            legal_moves: List of legal moves (optional)
            
        Returns:
            Boosted prior probabilities
        """
        tactical_boost = self.detect_tactical_moves(board, current_player, legal_moves)
        
        # Normalize boost to probability-like values
        if tactical_boost.max() > 0:
            tactical_boost = tactical_boost / (tactical_boost.sum() + 1e-8)
            
            # Mix original prior with tactical boost
            boosted_prior = (1 - boost_strength) * prior + boost_strength * tactical_boost.to(prior.device)
            
            # Renormalize
            boosted_prior = boosted_prior / (boosted_prior.sum() + 1e-8)
            
            return boosted_prior
        
        return prior
    
    def _get_piece_value(self, piece_code: int) -> float:
        """Get material value of a piece"""
        if piece_code == 0:
            return 0
        # Map piece codes to values
        # Assuming: 1=pawn, 2=knight, 3=bishop, 4=rook, 5=queen, 6=king
        # 7-12 for black pieces
        piece_type = int((piece_code - 1) % 6 + 1)
        values = [0, 1, 3, 3, 5, 9, 0]  # 0, pawn, knight, bishop, rook, queen, king
        return values[piece_type]
    
    def _is_check_move(self, board: np.ndarray, from_row: int, from_col: int, 
                       to_row: int, to_col: int, current_player: int) -> bool:
        """Check if a move would result in check (simplified)"""
        # This is a simplified check detection
        # In practice, you'd need full move validation
        
        # Make the move temporarily
        piece = board[from_row, from_col]
        target = board[to_row, to_col]
        board[to_row, to_col] = piece
        board[from_row, from_col] = 0
        
        # Find enemy king
        enemy_king_value = 12 if current_player == 1 else 6  # Black king or white king
        king_pos = None
        for r in range(8):
            for c in range(8):
                if board[r, c] == enemy_king_value:
                    king_pos = (r, c)
                    break
            if king_pos:
                break
        
        is_check = False
        if king_pos:
            # Check if our piece at to_row, to_col attacks the king
            # Simplified: just check if it's adjacent or on same line
            kr, kc = king_pos
            
            # Check for direct attacks based on piece type
            piece_type = (piece - 1) % 6 + 1
            
            if piece_type == 1:  # Pawn
                # Pawn attacks diagonally
                if abs(kr - to_row) == 1 and abs(kc - to_col) == 1:
                    if (current_player == 1 and kr > to_row) or (current_player == 2 and kr < to_row):
                        is_check = True
            elif piece_type in [4, 5]:  # Rook or Queen
                # Same rank or file
                if kr == to_row or kc == to_col:
                    is_check = True
            elif piece_type in [3, 5]:  # Bishop or Queen  
                # Same diagonal
                if abs(kr - to_row) == abs(kc - to_col):
                    is_check = True
            elif piece_type == 2:  # Knight
                # Knight moves
                if (abs(kr - to_row), abs(kc - to_col)) in [(1, 2), (2, 1)]:
                    is_check = True
        
        # Restore board
        board[from_row, from_col] = piece
        board[to_row, to_col] = target
        
        return is_check
    
    def boost_prior_with_tactics(self, prior: torch.Tensor, board: torch.Tensor,
                                current_player: int, legal_moves: List[int],
                                boost_strength: float = 0.3) -> torch.Tensor:
        """
        Boost prior probabilities based on tactical importance.
        
        Args:
            prior: Original prior probabilities for all moves
            board: Current board state
            current_player: Current player (1 = white, 2 = black)
            legal_moves: List of legal move indices
            boost_strength: How much to boost tactical moves (0.0 = no boost, 1.0 = full replacement)
            
        Returns:
            Boosted prior probabilities
        """
        tactical_boost = self.detect_tactical_moves(board, current_player, legal_moves)
        
        # Normalize boost to probability-like values
        if tactical_boost.max() > 0:
            # Only consider boosts for legal moves
            legal_boost = torch.zeros_like(tactical_boost)
            for move in legal_moves:
                legal_boost[move] = tactical_boost[move]
            
            # Normalize
            if legal_boost.sum() > 0:
                legal_boost = legal_boost / legal_boost.sum()
                
                # Mix original prior with tactical boost
                boosted_prior = (1 - boost_strength) * prior + boost_strength * legal_boost
                
                # Renormalize
                boosted_prior = boosted_prior / (boosted_prior.sum() + 1e-8)
                
                return boosted_prior
        
        return prior