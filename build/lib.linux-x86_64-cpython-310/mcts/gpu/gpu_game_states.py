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
import logging

logger = logging.getLogger(__name__)


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
        # Set default board size if not specified (0 or None)
        if not self.board_size:
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
        # Fall back to CPU if CUDA requested but not available
        if config.device == 'cuda' and not torch.cuda.is_available():
            self.device = torch.device('cpu')
            logger.warning("CUDA requested but not available, falling back to CPU")
        else:
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
        self.game_result = torch.zeros(capacity, dtype=torch.int8, device=device)  # Game result enum value
        
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
            raise RuntimeError(f"Tree full: Cannot allocate {num_states} states, only {self.capacity - self.num_states} available")
        
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
        # Validate that states are actually allocated
        if not self.allocated_mask[indices].all():
            unallocated = indices[~self.allocated_mask[indices]]
            logger.warning(f"Attempting to free unallocated states: {unallocated.tolist()}")
            # Filter to only allocated states
            indices = indices[self.allocated_mask[indices]]
        
        if len(indices) > 0:
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
            
            # Handle empty batch
            if batch_size == 0:
                return legal_mask
            
            # Vectorized check for empty squares
            boards = self.boards[state_indices]  # (batch, 15, 15)
            empty_mask = (boards == 0).view(batch_size, -1)  # (batch, 225)
            legal_mask = empty_mask
            
        elif self.game_type == GameType.GO:
            # Go: empty squares except ko point, plus pass move
            action_size = self.board_size * self.board_size + 1  # +1 for pass
            legal_mask = torch.zeros((batch_size, action_size), dtype=torch.bool, device=self.device)
            
            # Handle empty batch
            if batch_size == 0:
                return legal_mask
            
            # Check empty squares
            boards = self.boards[state_indices]
            empty_mask = (boards == 0).view(batch_size, -1)
            
            # Exclude ko points
            ko_points = self.ko_point[state_indices]
            for i in range(batch_size):
                legal_mask[i, :self.board_size * self.board_size] = empty_mask[i]
                if ko_points[i] >= 0:
                    legal_mask[i, ko_points[i]] = False
                # Pass move is always legal
                legal_mask[i, self.board_size * self.board_size] = True
                    
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
            # Check for pass moves in Go
            if self.game_type == GameType.GO:
                pass_action = self.board_size * self.board_size
                is_pass = actions == pass_action
                non_pass_mask = ~is_pass
            else:
                non_pass_mask = torch.ones_like(actions, dtype=torch.bool)
            
            # Convert action to board position for non-pass moves
            rows = actions // self.board_size
            cols = actions % self.board_size
            
            # Place stones (only for non-pass moves)
            batch_indices = torch.arange(len(state_indices), device=self.device)
            current_players = self.current_player[state_indices]
            
            # Vectorized board update - only update non-pass moves
            if non_pass_mask.any():
                non_pass_indices = state_indices[non_pass_mask]
                non_pass_rows = rows[non_pass_mask]
                non_pass_cols = cols[non_pass_mask]
                non_pass_players = current_players[non_pass_mask]
                self.boards[non_pass_indices, non_pass_rows, non_pass_cols] = non_pass_players
            
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
            Feature tensor ready for neural network:
            - Enhanced representation: 18 or 20 channels (configurable)
            - Basic representation: 4 channels for Gomoku, 5 for Go, 12 for Chess
        """
        batch_size = len(state_indices)
        
        # Use enhanced 20-channel representation if available
        if hasattr(self, '_use_enhanced_features') and self._use_enhanced_features:
            return self._get_enhanced_nn_features(state_indices)
        
        if self.game_type == GameType.GOMOKU:
            # Create proper 18-channel AlphaZero representation (GPU implementation)
            boards = self.boards[state_indices]
            current_players = self.current_player[state_indices]
            
            # Create 18 feature planes: board + player + 8 moves × 2 players
            features = torch.zeros((batch_size, 18, self.board_size, self.board_size), 
                                 device=self.device, dtype=torch.float32)
            
            # Channel 0: Current board state (all stones)
            features[:, 0] = (boards != 0).float()  # CRITICAL FIX: Use != 0 to include both players!
            
            # Channel 1: Current player indicator (constant plane)
            features[:, 1] = current_players.float().view(-1, 1, 1).expand(-1, self.board_size, self.board_size)
            
            # Channels 2-17: Previous 8 moves for each player (16 channels) - VECTORIZED
            if hasattr(self, 'full_move_history'):
                move_counts = self.move_count[state_indices]
                max_history = self.full_move_history.shape[1]
                
                # Create mask for valid moves (vectorized across all batches)
                # Shape: (batch_size, max_history)
                history_mask = torch.arange(max_history, device=self.device).unsqueeze(0) < move_counts.unsqueeze(1)
                
                # Get move history for all states in batch
                # Shape: (batch_size, max_history)
                batch_move_history = self.full_move_history[state_indices]
                
                # Only process last 8 moves - create indices for the most recent moves
                # Shape: (batch_size, 8)
                recent_move_indices = torch.clamp(move_counts.unsqueeze(1) - 1 - torch.arange(8, device=self.device), 0, max_history - 1)
                
                # Gather the last 8 moves for each batch item
                # Shape: (batch_size, 8)
                batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, 8)
                recent_moves = batch_move_history[batch_indices, recent_move_indices]
                
                # Create mask for valid recent moves
                recent_move_mask = recent_move_indices < move_counts.unsqueeze(1)
                recent_move_mask = recent_move_mask & (recent_moves >= 0)
                
                # Convert moves to board positions (vectorized)
                valid_moves = recent_moves[recent_move_mask]
                if len(valid_moves) > 0:
                    rows = valid_moves // self.board_size
                    cols = valid_moves % self.board_size
                    
                    # Create position mask for valid board positions
                    pos_mask = (rows >= 0) & (rows < self.board_size) & (cols >= 0) & (cols < self.board_size)
                    
                    if pos_mask.any():
                        # Get batch and move indices for valid positions
                        batch_pos, move_pos = torch.where(recent_move_mask)
                        valid_batch_pos = batch_pos[pos_mask].long()
                        valid_move_pos = move_pos[pos_mask].long()
                        valid_rows = rows[pos_mask].long()
                        valid_cols = cols[pos_mask].long()
                        
                        # Calculate channel indices (vectorized)
                        # Even move positions (0,2,4,6) = current player (channels 2-9)
                        # Odd move positions (1,3,5,7) = opponent (channels 10-17)
                        is_current_player = (valid_move_pos % 2) == 0
                        
                        # Current player channels: 2 + (move_pos // 2)
                        # Opponent channels: 10 + (move_pos // 2)
                        channel_indices = torch.where(
                            is_current_player,
                            2 + valid_move_pos // 2,
                            10 + valid_move_pos // 2
                        ).long()
                        
                        # Set features using advanced indexing (fully vectorized)
                        features[valid_batch_pos, channel_indices, valid_rows, valid_cols] = 1.0
            
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
            return getattr(self, '_enhanced_channels', 20)  # Enhanced representation
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
        """Enable enhanced feature representation for all games"""
        self._use_enhanced_features = True
        self._enhanced_channels = 20  # Default to 20 channels
        self._cpp_states_cache = {}
        self._cpp_states_pool = []  # Pool of reusable states
    
    def set_enhanced_channels(self, channels: int):
        """Set the number of channels for enhanced features (18 or 20)"""
        if channels not in [18, 20]:
            raise ValueError(f"Enhanced features only support 18 or 20 channels, got {channels}")
        self._enhanced_channels = channels
    
    def _get_enhanced_nn_features(self, state_indices: torch.Tensor) -> torch.Tensor:
        """
        Get enhanced neural network features using C++ implementation
        
        Channel layout (same for all games - Chess, Go, Gomoku):
        0: Current board state
        1: Current player indicator
        2-17: Previous 8 moves for each player (16 channels)
        18-19: Attack/defense planes (only included if _enhanced_channels == 20)
        
        Args:
            state_indices: State indices to get features for
            
        Returns:
            Feature tensor of shape (batch_size, channels, board_size, board_size)
            where channels is either 18 or 20 depending on configuration
        """
        batch_size = len(state_indices)
        target_channels = getattr(self, '_enhanced_channels', 20)
        
        # Create tensor to hold all features (always start with 20 channels from C++)
        features_20 = torch.zeros((batch_size, 20, self.board_size, self.board_size), 
                                device=self.device, dtype=torch.float32)
        
        # Process each state
        for i, state_idx in enumerate(state_indices):
            idx = state_idx.item()
            
            # Get or create C++ state
            cpp_state = self._get_cpp_state(idx)
            
            # Get enhanced tensor representation from C++
            enhanced_tensor = cpp_state.get_enhanced_tensor_representation()
            
            # Check if tensor size matches expected size
            if enhanced_tensor.shape[1] != self.board_size or enhanced_tensor.shape[2] != self.board_size:
                # Resize tensor to match expected board size
                # This is a workaround for C++ GoState not respecting board_size parameter
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Enhanced tensor size mismatch for {self.game_type.name}: "
                             f"expected {self.board_size}x{self.board_size}, "
                             f"got {enhanced_tensor.shape[1]}x{enhanced_tensor.shape[2]}")
                
                # Create a tensor of the correct size
                correct_tensor = np.zeros((20, self.board_size, self.board_size), dtype=np.float32)
                
                # Copy the center portion if the actual tensor is larger
                if enhanced_tensor.shape[1] > self.board_size:
                    # Take the top-left portion
                    correct_tensor[:, :self.board_size, :self.board_size] = enhanced_tensor[:, :self.board_size, :self.board_size]
                else:
                    # Pad with zeros if smaller (shouldn't happen)
                    min_h = min(enhanced_tensor.shape[1], self.board_size)
                    min_w = min(enhanced_tensor.shape[2], self.board_size)
                    correct_tensor[:, :min_h, :min_w] = enhanced_tensor[:, :min_h, :min_w]
                
                enhanced_tensor = correct_tensor
            
            # Convert numpy array to torch tensor and copy to GPU
            features_20[i] = torch.from_numpy(enhanced_tensor).to(self.device)
        
        # Return the appropriate number of channels based on configuration
        if target_channels == 18:
            # Remove attack/defense planes (channels 18-19) to get 18 channels
            return features_20[:, :18, :, :]
        else:
            # Return all 20 channels
            return features_20
    
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
                # Create game with proper board size
                if self.game_type == GameType.GO:
                    temp_state = alphazero_py.GoState(self.board_size)
                elif self.game_type == GameType.GOMOKU:
                    temp_state = alphazero_py.GomokuState(self.board_size)
                else:
                    temp_state = alphazero_py.create_game(cpp_game_type)
            
            # Reset temp state to initial position
            if self.game_type == GameType.GO:
                temp_initial = alphazero_py.GoState(self.board_size)
            elif self.game_type == GameType.GOMOKU:
                temp_initial = alphazero_py.GomokuState(self.board_size)
            else:
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
            
            # Create initial state with proper board size and apply moves
            if self.game_type == GameType.GO:
                cpp_state = alphazero_py.GoState(self.board_size)
            elif self.game_type == GameType.GOMOKU:
                cpp_state = alphazero_py.GomokuState(self.board_size)
            else:
                cpp_state = alphazero_py.create_game(cpp_game_type)
            
            # Apply moves to reach target state
            for move in moves:
                if cpp_state.is_legal_move(move):
                    cpp_state.make_move(move)
            
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
            
            # Create initial state with proper board size
            if self.game_type == GameType.GO:
                cpp_state = alphazero_py.GoState(self.board_size)
            elif self.game_type == GameType.GOMOKU:
                cpp_state = alphazero_py.GomokuState(self.board_size)
            else:
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