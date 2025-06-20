"""GameInterface wrapper for C++ game states

This module provides a Python interface to the C++ game implementations,
handling conversions between C++ game states and numpy arrays for neural networks.
"""

from enum import Enum
from typing import List, Tuple, Optional, Any, Dict
import numpy as np
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

# Import GPU attack/defense if available
try:
    from .gpu_attack_defense import create_gpu_attack_defense_computer
    GPU_AD_COMPUTER = create_gpu_attack_defense_computer()
except ImportError:
    GPU_AD_COMPUTER = None

try:
    # Import the compiled C++ game module
    import sys
    import os
    # Add build directory to path to find the module
    build_paths = [
        os.path.join(os.path.dirname(__file__), '../../../build/lib/Release'),
        os.path.join(os.path.dirname(__file__), '../../../build/lib/Debug'),
        os.path.join(os.path.dirname(__file__), '../../../build/lib'),
        os.path.join(os.path.dirname(__file__), '../../../build'),
    ]
    
    # Also need to set LD_LIBRARY_PATH for the shared library dependencies
    lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../build/lib/Release'))
    if os.path.exists(lib_path):
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if lib_path not in current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = lib_path + ':' + current_ld_path
    
    for path in build_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and abs_path not in sys.path:
            sys.path.insert(0, abs_path)
    
    import alphazero_py
    HAS_CPP_GAMES = True
    # Successfully loaded C++ game implementations
except ImportError as e:
    # Fallback for testing without C++ module
    alphazero_py = None
    HAS_CPP_GAMES = False
    # C++ game implementations not available
    

class GameType(Enum):
    """Supported game types"""
    CHESS = "chess"
    GO = "go"
    GOMOKU = "gomoku"


class GameInterface:
    """Unified interface for all game types
    
    Provides a consistent API for interacting with different game implementations,
    handling state conversions and move representations.
    """
    
    def __init__(self, game_type: GameType, board_size: Optional[int] = None, **kwargs):
        """Initialize game interface
        
        Args:
            game_type: Type of game to create
            board_size: Board size (for Go and Gomoku)
            **kwargs: Game-specific options:
                For Chess:
                    chess960 (bool): Enable Chess960/Fischer Random Chess
                    fen (str): Starting position in FEN notation
                For Go:
                    rule_set (str): 'chinese', 'japanese', or 'korean'
                    komi (float): Komi value (default depends on rule set)
                    chinese_rules (bool): Legacy option for Chinese rules
                    enforce_superko (bool): Whether to enforce positional superko
                For Gomoku:
                    use_renju (bool): Use Renju rules (forbidden moves for Black)
                    use_omok (bool): Use Omok rules
                    use_pro_long_opening (bool): Use pro-long opening restrictions
        """
        self.game_type = game_type
        self.board_size = board_size
        self.game_options = kwargs
        
        # Map our GameType enum to C++ GameType enum
        if HAS_CPP_GAMES:
            cpp_game_type_map = {
                GameType.CHESS: alphazero_py.GameType.CHESS,
                GameType.GO: alphazero_py.GameType.GO,
                GameType.GOMOKU: alphazero_py.GameType.GOMOKU
            }
            self._cpp_game_type = cpp_game_type_map.get(game_type)
            if self._cpp_game_type is None:
                raise ValueError(f"Unknown game type: {game_type}")
        
        if game_type == GameType.CHESS:
            self.board_shape = (8, 8)
            self.max_moves = 4096  # Upper bound for chess moves
            self.piece_planes = 12  # 6 piece types * 2 colors
            if HAS_CPP_GAMES:
                self._game_class = alphazero_py.ChessState
            else:
                self._game_class = MockChessState
                
        elif game_type == GameType.GO:
            if board_size is None:
                board_size = 19
            self.board_shape = (board_size, board_size)
            self.max_moves = board_size * board_size + 1  # All points + pass
            self.piece_planes = 2  # Black and white stones
            if HAS_CPP_GAMES:
                self._game_class = alphazero_py.GoState
            else:
                self._game_class = MockGoState
                
        elif game_type == GameType.GOMOKU:
            if board_size is None:
                board_size = 15
            self.board_shape = (board_size, board_size)
            self.max_moves = board_size * board_size
            self.piece_planes = 2  # Black and white stones
            if HAS_CPP_GAMES:
                self._game_class = alphazero_py.GomokuState
            else:
                self._game_class = MockGomokuState
                
        else:
            raise ValueError(f"Unknown game type: {game_type}")
            
        self.board_size = self.board_shape[0]
        
    def create_initial_state(self) -> Any:
        """Create initial game state with game-specific options
        
        Returns:
            Initial game state object
        """
        if HAS_CPP_GAMES:
            try:
                # Create with game-specific options if bindings support them
                if self.game_type == GameType.CHESS:
                    chess960 = self.game_options.get('chess960', False)
                    fen = self.game_options.get('fen', '')
                    # Try to create with options if supported
                    try:
                        if fen:
                            return alphazero_py.ChessState(chess960, fen)
                        elif chess960:
                            return alphazero_py.ChessState(chess960)
                        else:
                            return alphazero_py.ChessState()
                    except TypeError:
                        # Fallback if bindings don't support options yet
                        return alphazero_py.ChessState()
                        
                elif self.game_type == GameType.GO:
                    # Try to use rule_set if supported
                    rule_set = self.game_options.get('rule_set', 'chinese')
                    komi = self.game_options.get('komi', -1.0)
                    chinese_rules = self.game_options.get('chinese_rules', True)
                    enforce_superko = self.game_options.get('enforce_superko', True)
                    
                    try:
                        # Try new bindings with rule set
                        if hasattr(alphazero_py, 'GoRuleSet'):
                            rule_set_map = {
                                'chinese': alphazero_py.GoRuleSet.CHINESE,
                                'japanese': alphazero_py.GoRuleSet.JAPANESE,
                                'korean': alphazero_py.GoRuleSet.KOREAN
                            }
                            if rule_set in rule_set_map:
                                return alphazero_py.GoState(self.board_size, rule_set_map[rule_set], komi)
                        
                        # Try legacy constructor
                        if komi != -1.0:
                            state = alphazero_py.GoState(self.board_size, komi, chinese_rules, enforce_superko)
                        else:
                            state = alphazero_py.GoState(self.board_size)
                            
                        # Debug: verify board size was set correctly
                        if hasattr(state, 'get_board_size'):
                            actual_size = state.get_board_size()
                            if actual_size != self.board_size:
                                logger.warning(f"GoState created with size {self.board_size} but reports size {actual_size}")
                                # Try to create with default constructor and hope it respects the size
                                state = alphazero_py.GoState(self.board_size)
                        
                        return state
                    except TypeError:
                        # Fallback if bindings don't support options yet
                        return alphazero_py.GoState(self.board_size)
                        
                elif self.game_type == GameType.GOMOKU:
                    use_renju = self.game_options.get('use_renju', False)
                    use_omok = self.game_options.get('use_omok', False)
                    use_pro_long = self.game_options.get('use_pro_long_opening', False)
                    seed = self.game_options.get('seed', 0)
                    
                    try:
                        # Try to create with options if supported
                        return alphazero_py.GomokuState(self.board_size, use_renju, use_omok, seed, use_pro_long)
                    except TypeError:
                        # Fallback if bindings don't support options yet
                        return alphazero_py.GomokuState(self.board_size)
                else:
                    raise ValueError(f"Unknown game type: {self.game_type}")
            except Exception as e:
                print(f"Warning: C++ game creation failed ({e}), falling back to mock")
                # Fall through to mock implementation
                
        # Use mock implementation
        if self.game_type == GameType.CHESS:
            return self._game_class()
        else:
            return self._game_class(self.board_size)
            
    def state_to_numpy(self, state: Any, use_enhanced: bool = True) -> np.ndarray:
        """Convert game state to numpy array
        
        Args:
            state: Game state object
            use_enhanced: If True, use 20-channel enhanced representation
            
        Returns:
            Numpy array of shape (channels, height, width) for NN input
        """
        if HAS_CPP_GAMES:
            # Get enhanced representation with all 20 channels
            if use_enhanced:
                tensor = state.get_enhanced_tensor_representation()
            else:
                tensor = state.get_tensor_representation()
            # Keep in CHW format for neural network
            return tensor
        else:
            # Mock implementation
            return state.get_board()
            
    def state_to_tensor(self, state: Any, use_enhanced: bool = True) -> 'torch.Tensor':
        """Convert game state to PyTorch tensor
        
        Args:
            state: Game state object
            use_enhanced: If True, use 20-channel enhanced representation
            
        Returns:
            PyTorch tensor representation
        """
        import torch
        
        if HAS_CPP_GAMES:
            # Get tensor representation directly
            if use_enhanced:
                tensor_data = state.get_enhanced_tensor_representation()
            else:
                tensor_data = state.get_tensor_representation()
            return torch.from_numpy(tensor_data).float()
        else:
            # Mock implementation - convert board to tensor
            board_data = state.get_board()
            return torch.from_numpy(board_data).float()
            
    def tensor_to_state(self, tensor: 'torch.Tensor') -> Any:
        """Convert PyTorch tensor back to game state
        
        Args:
            tensor: PyTorch tensor representation
            
        Returns:
            Game state object
        """
        # Note: This is a challenging conversion as we need to reconstruct
        # the full game state from just the tensor representation
        # For now, we'll raise an error to indicate this needs implementation
        raise NotImplementedError(
            "tensor_to_state conversion is not yet implemented. "
            "This requires reconstructing game metadata from tensor data."
        )
        
    def get_state_shape(self) -> tuple:
        """Get the expected shape of state tensors
        
        Returns:
            Tuple representing tensor shape
        """
        if self.game_type == GameType.CHESS:
            # Chess: 8x8 board with multiple piece channels
            return (12, 8, 8)  # 6 piece types * 2 colors
        elif self.game_type == GameType.GO:
            # Go: variable board size with player stones + captures
            return (3, self.board_size, self.board_size)  # Black, White, Empty
        elif self.game_type == GameType.GOMOKU:
            # Gomoku: variable board size with player stones
            return (3, self.board_size, self.board_size)  # Black, White, Empty
        else:
            return (2, 15, 15)  # Default
            
    def batch_state_to_numpy(self, states: List[Any]) -> np.ndarray:
        """Convert batch of states to numpy array
        
        Args:
            states: List of game states
            
        Returns:
            Numpy array of shape (batch, height, width, channels)
        """
        boards = [self.state_to_numpy(state) for state in states]
        return np.stack(boards)
        
    def get_legal_moves(self, state: Any, shuffle: bool = True) -> List[int]:
        """Get list of legal moves
        
        Args:
            state: Game state
            shuffle: Whether to shuffle the moves to eliminate ordering bias
            
        Returns:
            List of legal move indices
        """
        moves = state.get_legal_moves()
        
        if shuffle and len(moves) > 1:
            # Convert to list if needed and shuffle
            import random
            moves_list = list(moves)
            random.shuffle(moves_list)
            return moves_list
            
        return moves
        
    def batch_get_legal_moves(self, states: List[Any], shuffle: bool = True) -> List[List[int]]:
        """Get legal moves for batch of states
        
        Args:
            states: List of game states
            shuffle: Whether to shuffle moves to eliminate ordering bias
            
        Returns:
            List of legal move lists
        """
        return [self.get_legal_moves(state, shuffle=shuffle) for state in states]
        
    def apply_move(self, state: Any, move: int) -> Any:
        """Apply move to state
        
        Args:
            state: Current game state
            move: Move to apply
            
        Returns:
            New game state after move
        """
        # Special case: Go uses -1 for pass move
        if self.game_type == GameType.GO and move == -1:
            # Pass move is valid
            pass
        elif move < 0 or move >= self.max_moves:
            raise ValueError(f"Invalid move: {move}")
            
        if HAS_CPP_GAMES:
            # Check if move is legal
            if not state.is_legal_move(move):
                raise ValueError(f"Illegal move: {move}")
            new_state = state.clone()
            new_state.make_move(move)
            return new_state
        else:
            new_state = state.clone()
            new_state.apply_move(move)
            return new_state
        
    def is_terminal(self, state: Any) -> bool:
        """Check if state is terminal
        
        Args:
            state: Game state
            
        Returns:
            True if game is over
        """
        return state.is_terminal()
        
    def get_winner(self, state: Any) -> int:
        """Get winner from terminal state
        
        Args:
            state: Terminal game state
            
        Returns:
            1 for player 1 win, -1 for player 2 win, 0 for draw
        """
        if HAS_CPP_GAMES:
            result = state.get_game_result()
            if result == alphazero_py.GameResult.WIN_PLAYER1:
                return 1
            elif result == alphazero_py.GameResult.WIN_PLAYER2:
                return -1
            elif result == alphazero_py.GameResult.DRAW:
                return 0
            else:  # ONGOING
                return 0
        else:
            return state.get_winner()
        
    def get_current_player(self, state: Any) -> int:
        """Get current player to move
        
        Args:
            state: Game state
            
        Returns:
            0 or 1 for current player
        """
        return state.get_current_player()
        
    def move_to_action_index(self, move: int) -> int:
        """Convert move to action index for neural network
        
        Args:
            move: Game-specific move representation
            
        Returns:
            Action index for neural network
        """
        # Special case: Go pass move (-1) maps to last action
        if self.game_type == GameType.GO and move == -1:
            return self.max_moves - 1  # Last position is pass
        # For now, moves are already indices
        return move
        
    def action_index_to_move(self, action_index: int) -> int:
        """Convert action index to move
        
        Args:
            action_index: Neural network action index
            
        Returns:
            Game-specific move representation
        """
        return action_index
        
    def get_action_probabilities_mask(self, state: Any) -> np.ndarray:
        """Get mask for legal actions
        
        Args:
            state: Game state
            
        Returns:
            Boolean array of shape (max_moves,) with True for legal moves
        """
        mask = np.zeros(self.max_moves, dtype=bool)
        legal_moves = self.get_legal_moves(state)
        for move in legal_moves:
            mask[self.move_to_action_index(move)] = True
        return mask
        
    def get_symmetries(self, board: np.ndarray, policy: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get symmetrical board positions for data augmentation
        
        Args:
            board: Board state array
            policy: Policy probabilities
            
        Returns:
            List of (board, policy) tuples for all symmetries
        """
        if self.game_type == GameType.CHESS:
            # Chess has no usable symmetries due to castling rights
            return [(board, policy)]
            
        elif self.game_type in [GameType.GO, GameType.GOMOKU]:
            # Go and Gomoku have 8 symmetries (4 rotations * 2 reflections)
            symmetries = []
            
            for i in range(4):  # 4 rotations
                # Rotate board - handle multi-channel case
                if len(board.shape) == 3:
                    # Rotate each channel separately
                    rot_board = np.rot90(board, i, axes=(1, 2))
                else:
                    rot_board = np.rot90(board, i)
                
                # Rotate policy (reshape to board shape, rotate, flatten)
                policy_board = policy[:-1].reshape(self.board_shape) if self.game_type == GameType.GO else policy.reshape(self.board_shape)
                rot_policy_board = np.rot90(policy_board, i)
                rot_policy = rot_policy_board.flatten()
                
                if self.game_type == GameType.GO:
                    # Add pass move back
                    rot_policy = np.append(rot_policy, policy[-1])
                    
                symmetries.append((rot_board, rot_policy))
                
                # Add reflection - handle multi-channel case
                if len(rot_board.shape) == 3:
                    # Flip along the last axis (width)
                    refl_board = np.flip(rot_board, axis=2)
                else:
                    refl_board = np.fliplr(rot_board)
                refl_policy_board = np.fliplr(rot_policy_board)
                refl_policy = refl_policy_board.flatten()
                
                if self.game_type == GameType.GO:
                    refl_policy = np.append(refl_policy, policy[-1])
                    
                symmetries.append((refl_board, refl_policy))
                
            return symmetries
            
    def encode_for_nn(self, state: Any, history: List[Any]) -> np.ndarray:
        """Encode state and history for neural network input
        
        The encoding includes 20 channels:
        - Channel 0: Current board state (player pieces)
        - Channel 1: Current player indicator plane
        - Channels 2-9: Last 8 moves by player 1 (binary planes)
        - Channels 10-17: Last 8 moves by player 2 (binary planes)
        - Channel 18: Attack score plane
        - Channel 19: Defense score plane
        
        Args:
            state: Current game state
            history: List of previous states (most recent first)
            
        Returns:
            Encoded state array of shape (20, height, width)
        """
        features = []
        
        # Channel 0: Current board state
        # Use basic representation (not enhanced) to get a single channel
        current_board = self.state_to_numpy(state, use_enhanced=False)
        if len(current_board.shape) == 3:
            # For chess, we might get multiple piece channels - sum them for a basic board
            current_board = current_board.sum(axis=0)
        features.append(current_board)
        
        # Channel 1: Current player indicator plane
        current_player = self.get_current_player(state)
        player_plane = np.full(self.board_shape[:2], current_player, dtype=np.float32)
        features.append(player_plane)
        
        # Channels 2-9: Last 8 moves by player 1
        # Channels 10-17: Last 8 moves by player 2
        for player in [1, 2]:
            for i in range(8):
                move_plane = np.zeros(self.board_shape[:2], dtype=np.float32)
                # In a real implementation, you would track move history
                # For now, create dummy planes
                features.append(move_plane)
        
        # Channel 18: Attack score plane
        # Channel 19: Defense score plane
        
        # Try GPU acceleration first
        if GPU_AD_COMPUTER is not None and hasattr(self, '_batch_mode') and self._batch_mode:
            try:
                # Use GPU batch computation (assumes we're in batch context)
                # This would be set by a batch processor
                attack_plane, defense_plane = self._gpu_ad_planes
                features.append(attack_plane)
                features.append(defense_plane)
            except AttributeError:
                # Not in batch mode, fall through to other methods
                pass
        
        if len(features) < 20:  # Attack/defense not added yet
            if HAS_CPP_GAMES and hasattr(alphazero_py, 'compute_attack_defense_planes'):
                try:
                    # Compute attack/defense planes using C++ implementation
                    attack_plane, defense_plane = alphazero_py.compute_attack_defense_planes(
                        state, self.game_type.value
                    )
                    features.append(attack_plane)
                    features.append(defense_plane)
                except Exception as e:
                    # Fallback to Python implementation
                    logger.debug(f"C++ attack/defense computation failed: {e}, using Python implementation")
                    from .attack_defense import compute_attack_defense_scores
                    board = self.state_to_numpy(state, use_enhanced=False)
                    if len(board.shape) == 3:
                        board = board.sum(axis=0)  # Sum channels for basic board
                    attack_plane, defense_plane = compute_attack_defense_scores(
                        self.game_type.value, board, self.get_current_player(state)
                    )
                    features.append(attack_plane)
                    features.append(defense_plane)
            else:
                # Use Python implementation if C++ not available
                from .attack_defense import compute_attack_defense_scores
                board = self.state_to_numpy(state, use_enhanced=False)
                if len(board.shape) == 3:
                    board = board.sum(axis=0)  # Sum channels for basic board
                attack_plane, defense_plane = compute_attack_defense_scores(
                    self.game_type.value, board, self.get_current_player(state)
                )
                features.append(attack_plane)
                features.append(defense_plane)
        
        # Stack all features (20 channels)
        combined = np.stack(features, axis=0)
        return combined
            
    def get_hash(self, state: Any) -> int:
        """Get Zobrist hash of state for transposition table
        
        Args:
            state: Game state
            
        Returns:
            64-bit hash value
        """
        if HAS_CPP_GAMES or hasattr(state, 'get_hash'):
            return state.get_hash()
        else:
            # Fallback: hash the board array and game state
            board = self.state_to_numpy(state)
            # Include move count to ensure different states have different hashes
            move_count = getattr(state, '_move_count', 0)
            return hash((board.tobytes(), move_count, state._current_player))
    
    # Training interface methods
    def reset(self) -> None:
        """Reset game to initial state"""
        self._current_state = self.create_initial_state()
        self._move_history = []
        
    def get_nn_input(self) -> np.ndarray:
        """Get neural network input representation (20 channels)"""
        if HAS_CPP_GAMES:
            # Use the C++ enhanced representation which includes all 20 channels
            return self._current_state.get_enhanced_tensor_representation()
        else:
            # Fallback to manual encoding
            return self.encode_for_nn(self._current_state, self._move_history)
        
    def get_state(self) -> Any:
        """Get current game state for MCTS"""
        return self._current_state
        
    def make_move(self, action: int) -> None:
        """Apply move and update state"""
        if hasattr(self, '_current_state'):
            # Add current state to history
            self._move_history.insert(0, self._current_state)
            if len(self._move_history) > 16:  # Keep last 16 moves
                self._move_history = self._move_history[:16]
            
            # Apply move
            self._current_state = self.apply_move(self._current_state, action)
        else:
            raise RuntimeError("Game not initialized. Call reset() first.")
            
    def get_reward(self) -> float:
        """Get final game reward"""
        if not self.is_terminal(self._current_state):
            return 0.0
            
        winner = self.get_winner(self._current_state)
        current_player = self.get_current_player(self._current_state)
        
        if winner == 0:  # Draw
            return 0.0
        elif winner == 1:  # Player 1 wins
            return 1.0 if current_player == 0 else -1.0
        else:  # Player 2 wins
            return -1.0 if current_player == 0 else 1.0
    
    # Additional methods exposing full C++ functionality
    def is_legal_move(self, state: Any, move: int) -> bool:
        """Check if a move is legal in the given state"""
        if HAS_CPP_GAMES:
            return state.is_legal_move(move)
        else:
            return move in self.get_legal_moves(state)
    
    def undo_move(self, state: Any) -> None:
        """Undo the last move (modifies state in-place)"""
        if HAS_CPP_GAMES:
            state.undo_move()
        else:
            raise NotImplementedError("Undo not implemented for mock states")
    
    def get_game_result(self, state: Any) -> str:
        """Get game result (ONGOING, WIN_PLAYER1, WIN_PLAYER2, DRAW)"""
        if HAS_CPP_GAMES:
            result = state.get_game_result()
            return str(result).split('.')[-1]  # Extract enum name
        else:
            if not self.is_terminal(state):
                return "ONGOING"
            winner = self.get_winner(state)
            if winner == 0:
                return "DRAW"
            elif winner == 1:
                return "WIN_PLAYER1"
            else:
                return "WIN_PLAYER2"
    
    def get_board_size(self, state: Any) -> int:
        """Get board size from state"""
        if HAS_CPP_GAMES:
            return state.get_board_size()
        else:
            return self.board_size
    
    def get_action_space_size(self, state: Any) -> int:
        """Get total action space size"""
        if HAS_CPP_GAMES:
            return state.get_action_space_size()
        else:
            return self.max_moves
    
    def get_tensor_representation(self, state: Any) -> np.ndarray:
        """Get basic tensor representation from C++"""
        if HAS_CPP_GAMES:
            return state.get_tensor_representation()
        else:
            return self.state_to_numpy(state)
    
    def get_enhanced_tensor_representation(self, state: Any) -> np.ndarray:
        """Get enhanced tensor representation with additional features"""
        if HAS_CPP_GAMES:
            return state.get_enhanced_tensor_representation()
        else:
            # For mock, just return the same as basic representation
            return self.state_to_numpy(state)
    
    def action_to_string(self, state: Any, action: int) -> str:
        """Convert action to human-readable string"""
        if HAS_CPP_GAMES:
            return state.action_to_string(action)
        else:
            if self.game_type == GameType.CHESS:
                # Simple algebraic notation for mock
                return f"move_{action}"
            else:
                # Board coordinates
                row = action // self.board_size
                col = action % self.board_size
                return f"{chr(ord('A') + col)}{row + 1}"
    
    def string_to_action(self, state: Any, move_str: str) -> int:
        """Convert string to action index"""
        if HAS_CPP_GAMES:
            return state.string_to_action(move_str)
        else:
            # Simple parsing for mock
            if self.game_type == GameType.CHESS:
                if move_str.startswith("move_"):
                    return int(move_str[5:])
            else:
                # Parse board coordinates
                if len(move_str) >= 2:
                    col = ord(move_str[0].upper()) - ord('A')
                    row = int(move_str[1:]) - 1
                    return row * self.board_size + col
            raise ValueError(f"Invalid move string: {move_str}")
    
    def to_string(self, state: Any) -> str:
        """Get string representation of the state"""
        if HAS_CPP_GAMES:
            return state.to_string()
        else:
            # Simple board representation for mock
            board = self.state_to_numpy(state)
            return f"GameState(shape={board.shape}, player={self.get_current_player(state)})"
    
    def get_move_history(self, state: Any) -> List[int]:
        """Get history of moves played"""
        if HAS_CPP_GAMES:
            return state.get_move_history()
        else:
            # Mock states don't track history by default
            return getattr(state, '_move_history', [])
    
    def clone_state(self, state: Any) -> Any:
        """Create a deep copy of the state"""
        if HAS_CPP_GAMES:
            return state.clone()
        else:
            # For mock states, create a simple copy
            import copy
            return copy.deepcopy(state)
    
    def get_canonical_form(self, state: Any) -> np.ndarray:
        """Get canonical form of the state from current player's perspective
        
        For most games, the state is already from the current player's perspective.
        This method exists for compatibility with training pipelines.
        
        Args:
            state: Game state
            
        Returns:
            State tensor from current player's perspective
        """
        # The state is already from the current player's perspective
        # The neural network input will handle any necessary transformations
        return self.state_to_numpy(state, use_enhanced=True)
    
    def get_next_state(self, state: Any, action: int) -> Any:
        """Apply move to state (alias for apply_move)
        
        Args:
            state: Current game state
            action: Move to apply
            
        Returns:
            New game state after move
        """
        return self.apply_move(state, action)
    
    def get_value(self, state: Any) -> float:
        """Get value of terminal state from perspective of last player who moved
        
        Args:
            state: Terminal game state
            
        Returns:
            1.0 if last player won, -1.0 if they lost, 0.0 for draw
        """
        if not self.is_terminal(state):
            return 0.0
            
        winner = self.get_winner(state)
        # Note: After a terminal move, current_player has switched
        # So we need to check from the previous player's perspective
        last_player = 1 - self.get_current_player(state)
        
        if winner == 0:  # Draw
            return 0.0
        elif winner == 1:  # Player 1 wins
            return 1.0 if last_player == 0 else -1.0
        else:  # Player 2 wins
            return -1.0 if last_player == 0 else 1.0


# Mock implementations for testing without C++ module
class MockGameState:
    """Base mock game state for testing"""
    
    def __init__(self):
        self._is_terminal = False
        self._winner = 0
        self._current_player = 0
        
    def clone(self):
        """Create a copy of the state"""
        import copy
        return copy.deepcopy(self)
        
    def is_terminal(self) -> bool:
        return self._is_terminal
        
    def get_winner(self) -> int:
        return self._winner
        
    def get_current_player(self) -> int:
        return self._current_player
        
    def apply_move(self, move: int) -> None:
        self._current_player = 1 - self._current_player
        self._move_count = getattr(self, '_move_count', 0) + 1
        

class MockChessState(MockGameState):
    """Mock chess state for testing"""
    
    def __init__(self):
        super().__init__()
        self.board = self._initial_chess_board()
        
    def _initial_chess_board(self) -> np.ndarray:
        """Create initial chess position"""
        board = np.zeros((8, 8, 12), dtype=np.float32)
        
        # White pieces (planes 0-5)
        # Pawns
        board[1, :, 0] = 1
        # Rooks
        board[0, [0, 7], 1] = 1
        # Knights
        board[0, [1, 6], 2] = 1
        # Bishops
        board[0, [2, 5], 3] = 1
        # Queen
        board[0, 3, 4] = 1
        # King
        board[0, 4, 5] = 1
        
        # Black pieces (planes 6-11)
        # Pawns
        board[6, :, 6] = 1
        # Rooks
        board[7, [0, 7], 7] = 1
        # Knights
        board[7, [1, 6], 8] = 1
        # Bishops
        board[7, [2, 5], 9] = 1
        # Queen
        board[7, 3, 10] = 1
        # King
        board[7, 4, 11] = 1
        
        return board
        
    def get_board(self) -> np.ndarray:
        return self.board
        
    def get_board_size(self) -> int:
        return 8
        
    def get_tensor_representation(self) -> np.ndarray:
        """Get basic tensor representation"""
        # Return the 12-channel board representation
        return self.board.transpose(2, 0, 1)  # Convert HWC to CHW
        
    def get_enhanced_tensor_representation(self) -> np.ndarray:
        """Get enhanced tensor representation with 20 channels"""
        # Create 20-channel representation
        tensor = np.zeros((20, 8, 8), dtype=np.float32)
        # First 12 channels: piece positions
        tensor[:12] = self.board.transpose(2, 0, 1)
        # Channel 12: Current player (0 for white, 1 for black)
        tensor[12] = self._current_player
        # Remaining channels: zeros for now (would be castling rights, etc.)
        return tensor
        
    def get_legal_moves(self) -> List[int]:
        # In initial position, white has 20 moves (16 pawn + 4 knight)
        return list(range(20))
        

class MockGoState(MockGameState):
    """Mock Go state for testing"""
    
    def __init__(self, size: int = 19):
        super().__init__()
        self.size = size
        self.board = np.zeros((size, size, 2), dtype=np.float32)
        
    def get_board(self) -> np.ndarray:
        return self.board
        
    def get_board_size(self) -> int:
        return self.size
        
    def get_tensor_representation(self) -> np.ndarray:
        """Get basic tensor representation"""
        # Return a 3-channel representation: black stones, white stones, empty
        tensor = np.zeros((3, self.size, self.size), dtype=np.float32)
        # Mock some basic board state
        tensor[0] = self.board[:, :, 0]  # Black stones
        tensor[1] = self.board[:, :, 1]  # White stones
        tensor[2] = 1 - (tensor[0] + tensor[1])  # Empty spaces
        return tensor
        
    def get_enhanced_tensor_representation(self) -> np.ndarray:
        """Get enhanced tensor representation with 20 channels"""
        # Create 20-channel representation
        tensor = np.zeros((20, self.size, self.size), dtype=np.float32)
        # First 3 channels: basic board state
        tensor[0] = self.board[:, :, 0]  # Black stones
        tensor[1] = self.board[:, :, 1]  # White stones
        tensor[2] = 1 - (tensor[0] + tensor[1])  # Empty spaces
        # Channel 3: Current player (0 for black, 1 for white)
        tensor[3] = self._current_player
        # Remaining channels: zeros for now (would be move history, etc.)
        return tensor
        
    def get_legal_moves(self) -> List[int]:
        # All empty intersections + pass
        return list(range(self.size * self.size + 1))
        

class MockGomokuState(MockGameState):
    """Mock Gomoku state for testing"""
    
    def __init__(self, size: int = 15):
        super().__init__()
        self.size = size
        self.board = np.zeros((size, size, 2), dtype=np.float32)
        
    def get_board(self) -> np.ndarray:
        return self.board
        
    def get_board_size(self) -> int:
        return self.size
        
    def get_tensor_representation(self) -> np.ndarray:
        """Get basic tensor representation"""
        # Return a 3-channel representation: black stones, white stones, empty
        tensor = np.zeros((3, self.size, self.size), dtype=np.float32)
        # Mock some basic board state
        tensor[0] = self.board[:, :, 0]  # Black stones
        tensor[1] = self.board[:, :, 1]  # White stones
        tensor[2] = 1 - (tensor[0] + tensor[1])  # Empty spaces
        return tensor
        
    def get_enhanced_tensor_representation(self) -> np.ndarray:
        """Get enhanced tensor representation with 20 channels"""
        # Create 20-channel representation
        tensor = np.zeros((20, self.size, self.size), dtype=np.float32)
        # First 3 channels: basic board state
        tensor[0] = self.board[:, :, 0]  # Black stones
        tensor[1] = self.board[:, :, 1]  # White stones
        tensor[2] = 1 - (tensor[0] + tensor[1])  # Empty spaces
        # Channel 3: Current player (0 for black, 1 for white)
        tensor[3] = self._current_player
        # Remaining channels: zeros for now (would be move history, etc.)
        return tensor
        
    def get_legal_moves(self) -> List[int]:
        # All empty intersections
        return list(range(self.size * self.size))