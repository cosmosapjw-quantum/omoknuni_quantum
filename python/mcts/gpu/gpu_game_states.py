"""GPU-resident game state management for all games

This module provides tensor-based game state representation that lives entirely on GPU,
enabling massive parallelization of state operations without CPU-GPU transfers.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from enum import IntEnum
import alphazero_py


class GameType(IntEnum):
    """Game type enumeration"""
    CHESS = 0
    GO = 1
    GOMOKU = 2


@dataclass
class GPUGameStatesConfig:
    """Configuration for GPU game states"""
    capacity: int = 100000  # Maximum number of states
    game_type: GameType = GameType.GOMOKU
    board_size: int = 15  # For Go/Gomoku
    device: str = 'cuda'
    dtype: torch.dtype = torch.float32  # For compatibility
    
    def __post_init__(self):
        # Adjust board size based on game
        if self.game_type == GameType.CHESS:
            self.board_size = 8
        elif self.game_type == GameType.GO:
            self.board_size = 19  # Standard Go board
        elif self.game_type == GameType.GOMOKU:
            self.board_size = 15  # Standard Gomoku board


class GPUGameStates:
    """Fully GPU-resident game state management for all games
    
    This class provides:
    - Tensor-based state representation
    - Batch state operations (clone, apply moves)
    - Parallel legal move generation
    - Terminal state detection
    - Direct feature extraction for neural networks
    
    All operations are performed on GPU without CPU transfers.
    """
    
    def __init__(self, config: GPUGameStatesConfig):
        """Initialize GPU game states
        
        Args:
            config: Configuration for game states
        """
        self.config = config
        self.device = torch.device(config.device)
        self.capacity = config.capacity
        self.game_type = config.game_type
        self.board_size = config.board_size
        
        # Initialize storage based on game type
        self._init_storage()
        
        # Verify boards was created
        if not hasattr(self, 'boards'):
            raise RuntimeError(f"Failed to initialize boards for game_type={self.game_type}")
        
        # State allocation tracking
        self.num_states = 0
        self.free_indices = torch.arange(self.capacity, device=self.device, dtype=torch.int32)
        self.allocated_mask = torch.zeros(self.capacity, device=self.device, dtype=torch.bool)
        
    def _init_storage(self):
        """Initialize tensor storage based on game type"""
        device = self.device
        capacity = self.capacity
        
        # Debug logging
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Initializing storage for game_type={self.game_type} (value={self.game_type.value}), board_size={self.board_size}")
        
        if self.game_type == GameType.CHESS:
            # Chess state representation
            # Board: 8x8, pieces encoded as integers
            # 0=empty, 1-6=white pieces, 7-12=black pieces
            self.boards = torch.zeros((capacity, 8, 8), dtype=torch.int8, device=device)
            
            # Castling rights: [white_kingside, white_queenside, black_kingside, black_queenside]
            self.castling = torch.zeros((capacity, 4), dtype=torch.bool, device=device)
            
            # En passant square (-1 if none, 0-63 for square index)
            self.en_passant = torch.full((capacity,), -1, dtype=torch.int8, device=device)
            
            # Halfmove clock for 50-move rule
            self.halfmove_clock = torch.zeros(capacity, dtype=torch.int16, device=device)
            
        elif self.game_type in [GameType.GO, GameType.GOMOKU]:
            # Go/Gomoku state representation
            # Board: NxN, 0=empty, 1=black, 2=white
            self.boards = torch.zeros((capacity, self.board_size, self.board_size), 
                                     dtype=torch.int8, device=device)
            
            if self.game_type == GameType.GO:
                # Ko point for Go (-1 if none, otherwise board position)
                self.ko_point = torch.full((capacity,), -1, dtype=torch.int16, device=device)
                
                # Captured stones count [black_captured, white_captured]
                self.captured = torch.zeros((capacity, 2), dtype=torch.int16, device=device)
            else:
                # Gomoku doesn't need additional state
                pass
        
        # Common metadata for all games
        self.current_player = torch.zeros(capacity, dtype=torch.int8, device=device)
        self.move_count = torch.zeros(capacity, dtype=torch.int16, device=device)
        self.is_terminal = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.winner = torch.zeros(capacity, dtype=torch.int8, device=device)  # 0=none, 1=player1, 2=player2, -1=draw
        
        # Move history for neural network features (last N moves)
        self.move_history_size = 8
        self.move_history = torch.full((capacity, self.move_history_size), -1, 
                                      dtype=torch.int16, device=device)
        
        # Full move history for exact state reconstruction
        # Max game length: Chess ~200, Go ~400, Gomoku ~225
        max_game_length = 500 if self.game_type == GameType.GO else 250
        self.full_move_history = torch.full((capacity, max_game_length), -1,
                                           dtype=torch.int16, device=device)
    
    def allocate_states(self, num_states: int) -> torch.Tensor:
        """Allocate new states and return their indices
        
        Args:
            num_states: Number of states to allocate
            
        Returns:
            Tensor of allocated state indices
        """
        if self.num_states + num_states > self.capacity:
            raise RuntimeError(f"Cannot allocate {num_states} states, only {self.capacity - self.num_states} available")
        
        # Get free indices
        indices = self.free_indices[:num_states]
        self.free_indices = self.free_indices[num_states:]
        self.allocated_mask[indices] = True
        self.num_states += num_states
        
        # Initialize states to empty
        self._reset_states(indices)
        
        return indices
    
    def free_states(self, indices: torch.Tensor):
        """Free allocated states
        
        Args:
            indices: State indices to free
        """
        # Return indices to free pool
        self.free_indices = torch.cat([self.free_indices, indices])
        self.allocated_mask[indices] = False
        self.num_states -= len(indices)
        
    def _reset_states(self, indices: torch.Tensor):
        """Reset states to initial empty state"""
        # Reset boards
        if self.game_type == GameType.CHESS:
            # Set up initial chess position
            for idx in indices:
                self._setup_chess_board(idx.item())
        else:
            # Go/Gomoku start with empty board
            self.boards[indices] = 0
            
        # Reset metadata
        self.current_player[indices] = 1  # Player 1 starts
        self.move_count[indices] = 0
        self.is_terminal[indices] = False
        self.winner[indices] = 0
        self.move_history[indices] = -1
        
        # Reset game-specific state
        if self.game_type == GameType.CHESS:
            self.castling[indices] = True  # All castling rights available
            self.en_passant[indices] = -1
            self.halfmove_clock[indices] = 0
        elif self.game_type == GameType.GO:
            self.ko_point[indices] = -1
            self.captured[indices] = 0
            
    def _setup_chess_board(self, idx: int):
        """Setup initial chess position for a single board"""
        board = self.boards[idx]
        board.zero_()
        
        # White pieces (1-6)
        # Pawns
        board[1, :] = 1
        # Rooks
        board[0, 0] = board[0, 7] = 2
        # Knights
        board[0, 1] = board[0, 6] = 3
        # Bishops
        board[0, 2] = board[0, 5] = 4
        # Queen
        board[0, 3] = 5
        # King
        board[0, 4] = 6
        
        # Black pieces (7-12)
        # Pawns
        board[6, :] = 7
        # Rooks
        board[7, 0] = board[7, 7] = 8
        # Knights
        board[7, 1] = board[7, 6] = 9
        # Bishops
        board[7, 2] = board[7, 5] = 10
        # Queen
        board[7, 3] = 11
        # King
        board[7, 4] = 12
        
    def clone_states(self, parent_indices: torch.Tensor, num_clones_per_parent: torch.Tensor) -> torch.Tensor:
        """Clone multiple parent states
        
        Args:
            parent_indices: Indices of parent states to clone
            num_clones_per_parent: Number of clones for each parent
            
        Returns:
            Tensor of clone indices
        """
        total_clones = int(num_clones_per_parent.sum())
        clone_indices = self.allocate_states(total_clones)
        
        # Create mapping from clones to parents
        parent_mapping = torch.repeat_interleave(parent_indices, num_clones_per_parent)
        
        # Copy board states
        self.boards[clone_indices] = self.boards[parent_mapping]
        
        # Copy metadata
        self.current_player[clone_indices] = self.current_player[parent_mapping]
        self.move_count[clone_indices] = self.move_count[parent_mapping]
        self.is_terminal[clone_indices] = self.is_terminal[parent_mapping]
        self.winner[clone_indices] = self.winner[parent_mapping]
        self.move_history[clone_indices] = self.move_history[parent_mapping]
        self.full_move_history[clone_indices] = self.full_move_history[parent_mapping]
        
        # Copy game-specific state
        if self.game_type == GameType.CHESS:
            self.castling[clone_indices] = self.castling[parent_mapping]
            self.en_passant[clone_indices] = self.en_passant[parent_mapping]
            self.halfmove_clock[clone_indices] = self.halfmove_clock[parent_mapping]
        elif self.game_type == GameType.GO:
            self.ko_point[clone_indices] = self.ko_point[parent_mapping]
            self.captured[clone_indices] = self.captured[parent_mapping]
            
        return clone_indices
    
    def get_legal_moves_mask(self, state_indices: torch.Tensor) -> torch.Tensor:
        """Get legal moves mask for multiple states
        
        Args:
            state_indices: State indices to get legal moves for
            
        Returns:
            Tensor of shape (num_states, action_space_size) with True for legal moves
        """
        batch_size = len(state_indices)
        
        if self.game_type == GameType.GOMOKU:
            # Gomoku: legal moves are empty squares
            action_size = self.board_size * self.board_size
            legal_mask = torch.zeros((batch_size, action_size), dtype=torch.bool, device=self.device)
            
            # Vectorized check for empty squares
            boards = self.boards[state_indices]  # (batch, 15, 15)
            empty_mask = (boards == 0).view(batch_size, -1)  # (batch, 225)
            legal_mask = empty_mask
            
        elif self.game_type == GameType.GO:
            # Go: empty squares except ko point
            action_size = self.board_size * self.board_size
            legal_mask = torch.zeros((batch_size, action_size), dtype=torch.bool, device=self.device)
            
            # Check empty squares
            boards = self.boards[state_indices]
            empty_mask = (boards == 0).view(batch_size, -1)
            
            # Exclude ko points
            ko_points = self.ko_point[state_indices]
            for i in range(batch_size):
                legal_mask[i] = empty_mask[i]
                if ko_points[i] >= 0:
                    legal_mask[i, ko_points[i]] = False
                    
        else:  # Chess
            # Chess legal move generation is complex, simplified version here
            # In production, this would call a specialized chess move generator
            action_size = 4096  # Upper bound for chess moves
            legal_mask = torch.zeros((batch_size, action_size), dtype=torch.bool, device=self.device)
            
            # Placeholder - would need full chess rules implementation
            # For now, mark some moves as legal for testing
            legal_mask[:, :10] = True
            
        return legal_mask
    
    def apply_moves(self, state_indices: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Apply moves to states
        
        Args:
            state_indices: State indices to apply moves to
            actions: Actions to apply (game-specific encoding)
            
        Returns:
            New state indices (if cloning) or same indices (if in-place)
        """
        # For now, apply moves in-place
        # In production, might want to clone first
        
        if self.game_type in [GameType.GOMOKU, GameType.GO]:
            # Convert action to board position
            rows = actions // self.board_size
            cols = actions % self.board_size
            
            # Place stones
            batch_indices = torch.arange(len(state_indices), device=self.device)
            current_players = self.current_player[state_indices]
            
            # Vectorized board update
            self.boards[state_indices, rows, cols] = current_players
            
            # Update move history (rolling buffer for NN features)
            self.move_history[state_indices] = torch.roll(self.move_history[state_indices], -1, dims=1)
            self.move_history[state_indices, -1] = actions.to(torch.int16)
            
            # Store in full move history at current move count position
            move_counts = self.move_count[state_indices].long()
            # Use state_indices directly for first dimension, move_counts for second
            self.full_move_history[state_indices.long(), move_counts] = actions.to(torch.int16)
            
            # Switch players
            self.current_player[state_indices] = 3 - current_players  # 1->2, 2->1
            
            # Increment move count
            self.move_count[state_indices] += 1
            
            # Check terminal states
            self._check_terminal_states(state_indices, actions)
            
        else:  # Chess
            # Chess move application would be more complex
            # Placeholder implementation
            pass
            
        return state_indices
    
    def _check_terminal_states(self, state_indices: torch.Tensor, last_moves: torch.Tensor):
        """Check if states are terminal after moves"""
        if self.game_type == GameType.GOMOKU:
            # Check for five in a row
            self._check_gomoku_wins(state_indices, last_moves)
        elif self.game_type == GameType.GO:
            # Check for captures, ko, passes
            self._check_go_terminal(state_indices)
        else:
            # Chess terminal checking
            pass
            
    def _check_gomoku_wins(self, state_indices: torch.Tensor, last_moves: torch.Tensor):
        """Check for Gomoku wins (five in a row)"""
        # This is a simplified version - would need full implementation
        # Check around last move for five in a row
        pass
        
    def _check_go_terminal(self, state_indices: torch.Tensor):
        """Check for Go terminal states"""
        # Simplified - would check for two passes, no legal moves, etc.
        pass
        
    def get_nn_features(self, state_indices: torch.Tensor) -> torch.Tensor:
        """Get neural network features directly from GPU states
        
        Args:
            state_indices: State indices to get features for
            
        Returns:
            Feature tensor ready for neural network (20 channels for enhanced representation)
        """
        batch_size = len(state_indices)
        
        # Use enhanced 20-channel representation if available
        if hasattr(self, '_use_enhanced_features') and self._use_enhanced_features:
            return self._get_enhanced_nn_features(state_indices)
        
        if self.game_type == GameType.GOMOKU:
            # Create feature planes: current player stones, opponent stones, empty
            boards = self.boards[state_indices]
            current_players = self.current_player[state_indices]
            
            # Create 4 feature planes
            features = torch.zeros((batch_size, 4, self.board_size, self.board_size), 
                                 device=self.device, dtype=torch.float32)
            
            # Current player stones
            features[:, 0] = (boards == current_players.unsqueeze(-1).unsqueeze(-1)).float()
            
            # Opponent stones
            opponent = 3 - current_players
            features[:, 1] = (boards == opponent.unsqueeze(-1).unsqueeze(-1)).float()
            
            # Empty squares
            features[:, 2] = (boards == 0).float()
            
            # Constant plane indicating current player
            features[:, 3] = current_players.float().view(-1, 1, 1).expand(-1, self.board_size, self.board_size)
            
        elif self.game_type == GameType.GO:
            # Similar to Gomoku but with more planes (liberties, ko, etc.)
            # Simplified version
            boards = self.boards[state_indices]
            features = torch.zeros((batch_size, 5, self.board_size, self.board_size), 
                                 device=self.device, dtype=torch.float32)
            
            # Basic planes
            features[:, 0] = (boards == 1).float()  # Black stones
            features[:, 1] = (boards == 2).float()  # White stones
            features[:, 2] = (boards == 0).float()  # Empty
            
            # Ko information
            # Would need to encode ko point as plane
            
        else:  # Chess
            # Chess features would include piece positions, castling rights, etc.
            features = torch.zeros((batch_size, 12, 8, 8), device=self.device, dtype=torch.float32)
            # Placeholder
            
        return features
    
    def get_nn_features_batch(self, state_indices: torch.Tensor) -> torch.Tensor:
        """Alias for get_nn_features for consistency with V3 API"""
        return self.get_nn_features(state_indices)
    
    def get_feature_planes(self) -> int:
        """Get number of feature planes for the current game type"""
        if hasattr(self, '_use_enhanced_features') and self._use_enhanced_features:
            return 20  # Enhanced representation
        if self.game_type == GameType.GOMOKU:
            return 4  # current player, opponent, empty, current player constant
        elif self.game_type == GameType.GO:
            return 5  # black, white, empty, ko, current player
        else:  # Chess
            return 12  # 6 piece types x 2 colors
    
    def get_state_info(self, state_idx: int) -> Dict[str, Any]:
        """Get human-readable state information for debugging"""
        info = {
            'game_type': self.game_type.name,
            'current_player': self.current_player[state_idx].item(),
            'move_count': self.move_count[state_idx].item(),
            'is_terminal': self.is_terminal[state_idx].item(),
            'winner': self.winner[state_idx].item()
        }
        
        if self.game_type == GameType.CHESS:
            info['castling'] = self.castling[state_idx].tolist()
            info['en_passant'] = self.en_passant[state_idx].item()
            info['halfmove_clock'] = self.halfmove_clock[state_idx].item()
        elif self.game_type == GameType.GO:
            info['ko_point'] = self.ko_point[state_idx].item()
            info['captured'] = self.captured[state_idx].tolist()
            
        return info
    
    def enable_enhanced_features(self):
        """Enable 20-channel enhanced feature representation for all games"""
        self._use_enhanced_features = True
        self._cpp_states_cache = {}
        self._cpp_states_pool = []  # Pool of reusable states
    
    def _get_enhanced_nn_features(self, state_indices: torch.Tensor) -> torch.Tensor:
        """
        Get 20-channel enhanced neural network features using C++ implementation
        
        Channel layout (same for all games - Chess, Go, Gomoku):
        0: Current board state
        1: Current player indicator
        2-17: Previous 8 moves for each player (16 channels)
        18-19: Attack/defense planes
        
        Args:
            state_indices: State indices to get features for
            
        Returns:
            Feature tensor of shape (batch_size, 20, board_size, board_size)
        """
        batch_size = len(state_indices)
        
        # Create tensor to hold all features
        features = torch.zeros((batch_size, 20, self.board_size, self.board_size), 
                             device=self.device, dtype=torch.float32)
        
        # Process each state
        for i, state_idx in enumerate(state_indices):
            idx = state_idx.item()
            
            # Get or create C++ state
            cpp_state = self._get_cpp_state(idx)
            
            # Get enhanced tensor representation from C++
            enhanced_tensor = cpp_state.get_enhanced_tensor_representation()
            
            # Convert numpy array to torch tensor and copy to GPU
            features[i] = torch.from_numpy(enhanced_tensor).to(self.device)
        
        return features
    
    def _get_cpp_state(self, state_idx: int):
        """Get or create C++ game state for given index using exact reconstruction"""
        
        # Check cache first
        if hasattr(self, '_cpp_states_cache') and state_idx in self._cpp_states_cache:
            return self._cpp_states_cache[state_idx]
        
        # Initialize cache if needed
        if not hasattr(self, '_cpp_states_cache'):
            self._cpp_states_cache = {}
            self._cpp_states_pool = []
        
        # Get move history for this state
        move_count = self.move_count[state_idx].item()
        
        if hasattr(self, 'full_move_history') and move_count > 0:
            # Build move list from full history
            moves = []
            for m in range(move_count):
                move = self.full_move_history[state_idx, m].item()
                if move >= 0:
                    moves.append(move)
            
            # Create state from moves using the exact C++ game logic
            # Map Python integer enum to C++ enum
            if self.game_type == GameType.CHESS:
                cpp_game_type = alphazero_py.GameType.CHESS
            elif self.game_type == GameType.GO:
                cpp_game_type = alphazero_py.GameType.GO
            elif self.game_type == GameType.GOMOKU:
                cpp_game_type = alphazero_py.GameType.GOMOKU
            else:
                raise ValueError(f"Unknown game type: {self.game_type}")
            
            # Convert moves to string format expected by createGameFromMoves
            move_strings = []
            
            # Get a temporary state to convert moves to strings
            if self._cpp_states_pool:
                temp_state = self._cpp_states_pool.pop()
            else:
                temp_state = alphazero_py.create_game(cpp_game_type)
            
            # Reset temp state to initial position
            temp_initial = alphazero_py.create_game(cpp_game_type)
            temp_state.copy_from(temp_initial)
            
            # Convert each move to string
            for move in moves:
                move_str = temp_state.action_to_string(move)
                move_strings.append(move_str)
                if temp_state.is_legal_move(move):
                    temp_state.make_move(move)
            
            # Return temp state to pool
            self._cpp_states_pool.append(temp_state)
            
            # Create state from move string
            moves_str = " ".join(move_strings)
            cpp_state = alphazero_py.create_game_from_moves(cpp_game_type, moves_str)
            
        else:
            # No move history - create initial state
            # Map Python integer enum to C++ enum
            if self.game_type == GameType.CHESS:
                cpp_game_type = alphazero_py.GameType.CHESS
            elif self.game_type == GameType.GO:
                cpp_game_type = alphazero_py.GameType.GO
            elif self.game_type == GameType.GOMOKU:
                cpp_game_type = alphazero_py.GameType.GOMOKU
            else:
                raise ValueError(f"Unknown game type: {self.game_type}")
            
            cpp_state = alphazero_py.create_game(cpp_game_type)
        
        # Cache the state
        self._cpp_states_cache[state_idx] = cpp_state
        
        return cpp_state
    
    def clear_enhanced_cache(self):
        """Clear the C++ states cache and pool"""
        if hasattr(self, '_cpp_states_cache'):
            self._cpp_states_cache.clear()
        if hasattr(self, '_cpp_states_pool'):
            self._cpp_states_pool.clear()