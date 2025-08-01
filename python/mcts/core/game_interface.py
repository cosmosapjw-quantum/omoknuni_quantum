"""GameInterface wrapper for C++ game states

This module provides a Python interface to the C++ game implementations,
handling conversions between C++ game states and numpy arrays for neural networks.
"""

from enum import Enum
from typing import Tuple, Optional, Any, List
import numpy as np
import logging

logger = logging.getLogger(__name__)

# GPU attack/defense functionality removed in streamlined build
GPU_AD_COMPUTER = None

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

try:
    import alphazero_py
    HAS_CPP_GAMES = True
except ImportError as e:
    HAS_CPP_GAMES = False
    # Create mock for testing purposes
    class MockGameType:
        CHESS = 1
        GO = 2
        GOMOKU = 3
    
    class MockGameState:
        def __init__(self, *args, **kwargs):
            self.board_size = 15
            pass
        def clone(self): return self
        def get_legal_actions(self): return []
        def make_move(self, action): return self
        def get_winner(self): return 0
        def is_terminal(self): return False
        def get_state(self): return np.zeros((3, 15, 15))
        def get_board_size(self): return 15
        def get_hash(self): return 0
        
        def get_basic_tensor_representation(self):
            """Return 18-channel basic representation for empty board"""
            return np.zeros((18, 15, 15), dtype=np.float32)
            
        def get_enhanced_tensor_representation(self):
            """Return 20-channel enhanced representation for empty board"""  
            return np.zeros((20, 15, 15), dtype=np.float32)
            
        def get_tensor_representation(self):
            """Return 3-channel standard representation for empty board"""
            return np.zeros((3, 15, 15), dtype=np.float32)
    
    class MockAlphaZeroPy:
        GameType = MockGameType
        ChessState = MockGameState
        GoState = MockGameState
        GomokuState = MockGameState
    
    alphazero_py = MockAlphaZeroPy()
    

class GameType(Enum):
    """Supported game types"""
    CHESS = "chess"
    GO = "go"
    GOMOKU = "gomoku"


def create_game_interface(game_type: str = 'gomoku', board_size: Optional[int] = None, **kwargs) -> 'GameInterface':
    """Create a game interface for the specified game type
    
    Args:
        game_type: String name of the game type ('chess', 'go', 'gomoku')
        board_size: Board size (for Go and Gomoku)
        **kwargs: Additional game-specific options
        
    Returns:
        GameInterface instance
    """
    # Convert string to GameType enum
    game_type_map = {
        'chess': GameType.CHESS,
        'go': GameType.GO,
        'gomoku': GameType.GOMOKU
    }
    
    game_type_enum = game_type_map.get(game_type.lower())
    if game_type_enum is None:
        raise ValueError(f"Unknown game type: {game_type}. Choose from: {list(game_type_map.keys())}")
    
    return GameInterface(game_type_enum, board_size=board_size, **kwargs)


class GameInterface:
    """Unified interface for all game types
    
    Provides a consistent API for interacting with different game implementations,
    handling state conversions and move representations.
    """
    
    def __init__(self, game_type: GameType, board_size: Optional[int] = None, 
                 input_representation: str = 'basic', **kwargs):
        """Initialize game interface
        
        Args:
            game_type: Type of game to create
            board_size: Board size (for Go and Gomoku)
            input_representation: Type of tensor representation to use
                - 'enhanced': 20-channel enhanced representation with attack/defense
                - 'basic': 18-channel basic representation without attack/defense
                - 'standard': 3-channel standard representation
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
        # Use the specified input representation, defaulting to 'basic' (18-channel) 
        # which provides consistent input format with move history
        self.input_representation = input_representation
        self.game_options = kwargs
        
        # Map our GameType enum to C++ GameType enum
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
            if not HAS_CPP_GAMES:
                logger.warning("C++ game modules not available. Using mock for testing.")
            self._game_class = alphazero_py.ChessState
                
        elif game_type == GameType.GO:
            if board_size is None:
                board_size = 19
            self.board_shape = (board_size, board_size)
            self.max_moves = board_size * board_size + 1  # All points + pass
            self.piece_planes = 2  # Black and white stones
            if not HAS_CPP_GAMES:
                logger.warning("C++ game modules not available. Using mock for testing.")
            self._game_class = alphazero_py.GoState
                
        elif game_type == GameType.GOMOKU:
            if board_size is None:
                board_size = 15
            self.board_shape = (board_size, board_size)
            self.max_moves = board_size * board_size
            self.piece_planes = 2  # Black and white stones
            if not HAS_CPP_GAMES:
                logger.warning("C++ game modules not available. Using mock for testing.")
            self._game_class = alphazero_py.GomokuState
                
        else:
            raise ValueError(f"Unknown game type: {game_type}")
            
        self.board_size = self.board_shape[0]
        
    def create_initial_state(self) -> Any:
        """Create initial game state with game-specific options
        
        Returns:
            Initial game state object
        """
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
            raise RuntimeError(f"C++ game creation failed: {e}. Please check C++ module installation and build.") from e
            
    def state_to_numpy(self, state: Any, representation_type: str = None) -> np.ndarray:
        """Convert game state to numpy array
        
        Args:
            state: Game state object
            representation_type: Type of representation to use:
                - 'enhanced': 20-channel enhanced representation with attack/defense
                - 'basic': 18-channel basic representation without attack/defense  
                - 'minimal': 3-channel minimal representation
            
        Returns:
            Numpy array of shape (channels, height, width) for NN input
        """
        # Use configured representation if not specified
        if representation_type is None:
            representation_type = self.input_representation
            
        if representation_type == 'enhanced':
            tensor = state.get_enhanced_tensor_representation()
        elif representation_type == 'basic':
            tensor = state.get_basic_tensor_representation()
        elif representation_type == 'minimal':
            tensor = state.get_tensor_representation()
        else:
            # Legacy support: use_enhanced parameter
            if representation_type is True:
                tensor = state.get_enhanced_tensor_representation()
            else:
                tensor = state.get_tensor_representation()
        # Keep in CHW format for neural network
        return tensor
            
    def state_to_tensor(self, state: Any, representation_type: str = None) -> 'torch.Tensor':
        """Convert game state to PyTorch tensor
        
        Args:
            state: Game state object
            representation_type: Type of representation to use:
                - 'enhanced': 20-channel enhanced representation with attack/defense
                - 'basic': 18-channel basic representation without attack/defense
                - 'standard': 3-channel standard representation
            
        Returns:
            PyTorch tensor representation
        """
        import torch
        
        # Use configured representation if not specified
        if representation_type is None:
            representation_type = self.input_representation
            
        # Get tensor representation directly
        if representation_type == 'enhanced':
            tensor_data = state.get_enhanced_tensor_representation()
        elif representation_type == 'basic':
            tensor_data = state.get_basic_tensor_representation()
        elif representation_type == 'standard':
            tensor_data = state.get_tensor_representation()
        else:
            # Legacy support: use_enhanced parameter
            if representation_type is True:
                tensor_data = state.get_enhanced_tensor_representation()
            else:
                tensor_data = state.get_tensor_representation()
        return torch.from_numpy(tensor_data).float()
            
    def tensor_to_state(self, tensor: 'torch.Tensor') -> Any:
        """Convert PyTorch tensor back to game state
        
        This method reconstructs a game state from its tensor representation.
        Note: The reconstruction is approximate and may not preserve exact move order
        or game metadata (e.g., castling rights, ko state).
        
        Args:
            tensor: PyTorch tensor representation of shape (channels, height, width)
            
        Returns:
            Game state object
            
        Raises:
            ValueError: If tensor has invalid shape or corrupted data
            
        Example:
            >>> interface = GameInterface(GameType.GOMOKU, board_size=15)
            >>> state = interface.create_initial_state()
            >>> tensor = interface.state_to_tensor(state)
            >>> reconstructed = interface.tensor_to_state(tensor)
        """
        import torch
        import numpy as np
        
        # Input validation
        if tensor is None:
            raise ValueError("Tensor cannot be None")
            
        # Convert tensor to numpy array
        if isinstance(tensor, torch.Tensor):
            tensor_np = tensor.cpu().numpy()
        else:
            tensor_np = np.asarray(tensor)
            
        # Validate tensor shape
        if tensor_np.ndim != 3:
            raise ValueError(f"Expected 3D tensor (channels, height, width), got {tensor_np.ndim}D tensor with shape {tensor_np.shape}")
            
        # Validate board dimensions
        if tensor_np.shape[1:] != self.board_shape:
            raise ValueError(
                f"Tensor spatial dimensions {tensor_np.shape[1:]} don't match expected board shape {self.board_shape}"
            )
            
        # Get number of channels to determine representation type
        num_channels = tensor_np.shape[0]
        
        # Validate channel count
        valid_channels = [3, 18, 20]  # standard, basic, enhanced
        if num_channels not in valid_channels:
            raise ValueError(
                f"Invalid number of channels: {num_channels}. Expected one of {valid_channels}"
            )
        
        # Check for corrupted data (NaN or Inf values)
        if np.any(np.isnan(tensor_np)) or np.any(np.isinf(tensor_np)):
            raise ValueError("Tensor contains NaN or Inf values")
            
        # Size limit check to prevent DoS
        max_board_size = 50  # Reasonable limit for board games
        if any(dim > max_board_size for dim in self.board_shape):
            raise ValueError(f"Board size {self.board_shape} exceeds maximum allowed size {max_board_size}x{max_board_size}")
        
        try:
            # Create empty initial state
            state = self.create_initial_state()
        except Exception as e:
            raise ValueError(f"Failed to create initial state: {e}")
        
        if num_channels == 3:
            # Standard 3-channel representation
            # Channel 0: Player 1 pieces
            # Channel 1: Player 2 pieces
            # Channel 2: Current player indicator
            
            player1_pieces = tensor_np[0]
            player2_pieces = tensor_np[1]
            current_player = int(tensor_np[2, 0, 0]) + 1  # Convert back from 0/1 to 1/2
            
            # Reconstruct moves to reach this board position
            # We need to replay moves in order to properly set game state
            moves_to_make = []
            
            if self.game_type == GameType.GOMOKU:
                # Vectorized approach to find all occupied positions
                p1_positions = np.argwhere(player1_pieces > 0.5)
                p2_positions = np.argwhere(player2_pieces > 0.5)
                
                # Convert to move indices with player info
                moves_to_make = []
                for pos in p1_positions:
                    moves_to_make.append((pos[0] * self.board_size + pos[1], 1))
                for pos in p2_positions:
                    moves_to_make.append((pos[0] * self.board_size + pos[1], 2))
                    
                # Vectorized distance calculation from center
                center = self.board_size // 2
                if moves_to_make:
                    # Calculate distances in a vectorized manner
                    move_indices = np.array([m[0] for m in moves_to_make])
                    rows = move_indices // self.board_size
                    cols = move_indices % self.board_size
                    distances = np.abs(rows - center) + np.abs(cols - center)
                    
                    # Sort by distance
                    sorted_indices = np.argsort(distances)
                    moves_to_make = [moves_to_make[i] for i in sorted_indices]
                
                # Ensure moves alternate between players starting with player 1
                ordered_moves = []
                p1_moves = [m for m in moves_to_make if m[1] == 1]
                p2_moves = [m for m in moves_to_make if m[1] == 2]
                
                for i in range(max(len(p1_moves), len(p2_moves))):
                    if i < len(p1_moves):
                        ordered_moves.append(p1_moves[i][0])
                    if i < len(p2_moves):
                        ordered_moves.append(p2_moves[i][0])
                        
                # Apply moves to reconstruct state
                # Cache legal moves check for better performance
                for move in ordered_moves:
                    try:
                        new_state = state.clone()
                        new_state.make_move(move)
                        state = new_state
                    except Exception:
                        # Skip invalid moves silently during reconstruction
                        continue
                        
                # Handle case where current player doesn't match
                if state.get_current_player() != current_player:
                    # Add a dummy pass or adjust the last move
                    # This is a limitation of reconstruction without full history
                    pass
                    
            elif self.game_type == GameType.GO:
                # Similar reconstruction for Go
                # GO has additional complexity with captures, ko, etc.
                # For basic reconstruction, place stones
                for row in range(self.board_size):
                    for col in range(self.board_size):
                        move = row * self.board_size + col
                        if player1_pieces[row, col] > 0.5:
                            # Try to place black stone
                            if move in self.get_legal_moves(state) and state.get_current_player() == 1:
                                new_state = state.clone()
                                new_state.make_move(move)
                                state = new_state
                        elif player2_pieces[row, col] > 0.5:
                            # Try to place white stone
                            if move in self.get_legal_moves(state) and state.get_current_player() == 2:
                                new_state = state.clone()
                                new_state.make_move(move)
                                state = new_state
                                
            elif self.game_type == GameType.CHESS:
                # Chess reconstruction is complex due to piece types, castling rights, en passant
                # This is a simplified version that may not capture all game state
                logger.warning("Chess tensor_to_state reconstruction is approximate and may not preserve all game rules")
                
        else:
            # Enhanced representations (18 or 20 channels)
            # Use the first channel as board state and second as current player
            board_channel = tensor_np[0]
            current_player = int(tensor_np[1, 0, 0])
            
            # Simplified reconstruction for enhanced representations
            if self.game_type == GameType.GOMOKU:
                for row in range(self.board_size):
                    for col in range(self.board_size):
                        if abs(board_channel[row, col]) > 0.5:
                            move = row * self.board_size + col
                            if move in self.get_legal_moves(state):
                                new_state = state.clone()
                                new_state.make_move(move)
                                state = new_state
                                
        return state
        
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
        # Special case: Go pass move
        if self.game_type == GameType.GO:
            # Pass move can be represented in different ways
            pass_move = self.board_size * self.board_size
            if move == pass_move or move == self.max_moves - 1 or move == self.max_moves or move == -1:
                # Convert to the expected pass move representation (-1)
                move = -1  # Use -1 as pass move for C++ backend
            elif move < -1 or move > pass_move:
                raise ValueError(f"Invalid move: {move}")
        elif move < 0 or move >= self.max_moves:
            raise ValueError(f"Invalid move: {move}")
            
        # Check if move is legal
        if not state.is_legal_move(move):
            move_row = move // self.board_size
            move_col = move % self.board_size
            raise ValueError(f"Illegal move: {move} (row {move_row}, col {move_col})")
        new_state = state.clone()
        new_state.make_move(move)
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
        result = state.get_game_result()
        if result == alphazero_py.GameResult.WIN_PLAYER1:
            return 1
        elif result == alphazero_py.GameResult.WIN_PLAYER2:
            return -1
        elif result == alphazero_py.GameResult.DRAW:
            return 0
        else:  # ONGOING
            return 0
        
    def get_current_player(self, state: Any) -> int:
        """Get current player to move
        
        Args:
            state: Game state
            
        Returns:
            1 or 2 for current player (1=BLACK, 2=WHITE)
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
        # Use standard representation to get basic board state
        current_board = self.state_to_numpy(state, representation_type='standard')
        if len(current_board.shape) == 3:
            # For games with multiple channels, extract the board state
            # For Gomoku/Go: channel 0 is player 1, channel 1 is player 2
            # Combine them into a single board: 1 for player 1, -1 for player 2
            if self.game_type in [GameType.GO, GameType.GOMOKU]:
                board = current_board[0] - current_board[1]  # Player 1 minus Player 2
            else:
                # For chess or other games, sum channels
                board = current_board.sum(axis=0)
            current_board = board
        features.append(current_board)
        
        # Channel 1: Current player indicator plane
        current_player = self.get_current_player(state)
        player_plane = np.full(self.board_shape[:2], current_player, dtype=np.float32)
        features.append(player_plane)
        
        # Channels 2-9: Last 8 moves by player 1
        # Channels 10-17: Last 8 moves by player 2
        
        # Extract move history from provided history states
        move_history_p1 = []  # List of (row, col) tuples for player 1 moves
        move_history_p2 = []  # List of (row, col) tuples for player 2 moves
        
        if history:
            # Input validation
            if not isinstance(history, list):
                raise ValueError("History must be a list of game states")
                
            # Validate all states in history
            for i, hist_state in enumerate(history):
                if hist_state is None:
                    raise ValueError(f"History state at index {i} is None")
                if not hasattr(hist_state, 'get_tensor_representation'):
                    raise ValueError(f"Invalid state at history index {i}: missing get_tensor_representation method")
            
            # Size limit check - prevent processing extremely long histories
            max_history_length = 100
            if len(history) > max_history_length:
                logger.warning(f"History length {len(history)} exceeds limit {max_history_length}, truncating")
                history = history[-max_history_length:]
            
            # Add current state to the end of history for complete sequence
            full_history = history + [state]
            
            # Vectorized processing of history states
            # Pre-allocate arrays for better performance
            history_length = len(full_history)
            if history_length > 1:
                # Convert all states to tensors in batch for efficiency
                state_tensors = []
                for hist_state in full_history:
                    try:
                        tensor = self.state_to_numpy(hist_state, representation_type='standard')
                        state_tensors.append(tensor)
                    except Exception as e:
                        logger.warning(f"Failed to convert history state to tensor: {e}")
                        continue
                
                # Process in vectorized manner
                state_tensors = np.array(state_tensors)
                
                # Extract board states for Gomoku/Go
                if self.game_type in [GameType.GO, GameType.GOMOKU] and len(state_tensors) > 1:
                    # Vectorized difference calculation
                    # Shape: (history_length, channels, height, width)
                    player1_boards = state_tensors[:, 0, :, :]  # Player 1 pieces
                    player2_boards = state_tensors[:, 1, :, :]  # Player 2 pieces
                    
                    # Calculate differences between consecutive states
                    p1_diffs = player1_boards[1:] - player1_boards[:-1]
                    p2_diffs = player2_boards[1:] - player2_boards[:-1]
                    
                    # Find new moves (where difference > 0)
                    # Process from most recent to oldest
                    for i in range(len(p1_diffs) - 1, -1, -1):
                        # Player 1 moves
                        new_p1_mask = p1_diffs[i] > 0.5
                        if np.any(new_p1_mask):
                            positions = np.argwhere(new_p1_mask)
                            for pos in positions:
                                move_history_p1.append((pos[0], pos[1]))
                                if len(move_history_p1) >= 8:
                                    break
                        
                        # Player 2 moves  
                        new_p2_mask = p2_diffs[i] > 0.5
                        if np.any(new_p2_mask):
                            positions = np.argwhere(new_p2_mask)
                            for pos in positions:
                                move_history_p2.append((pos[0], pos[1]))
                                if len(move_history_p2) >= 8:
                                    break
                        
                        # Early termination if we have enough moves
                        if len(move_history_p1) >= 8 and len(move_history_p2) >= 8:
                            break
                else:
                    # Fallback for other games or single state
                    # Process history from most recent to oldest to get last N moves
                    for i in range(len(full_history) - 1, 0, -1):
                        if i >= len(state_tensors) or i-1 >= len(state_tensors):
                            continue
                            
                        # Get previous and current board states
                        prev_state_tensor = state_tensors[i-1]
                        curr_state_tensor = state_tensors[i]
                        
                        # Extract board positions
                        if len(prev_state_tensor.shape) == 3:
                            if self.game_type in [GameType.GO, GameType.GOMOKU]:
                                prev_board = prev_state_tensor[0] + prev_state_tensor[1]
                                curr_p1 = curr_state_tensor[0]
                                curr_p2 = curr_state_tensor[1]
                            else:
                                prev_board = prev_state_tensor.sum(axis=0)
                                curr_board = curr_state_tensor.sum(axis=0)
                        else:
                            prev_board = prev_state_tensor
                            curr_board = curr_state_tensor
                        
                        # For Gomoku/Go, find new stones by comparing boards
                        if self.game_type in [GameType.GO, GameType.GOMOKU]:
                            # Check for new player 1 stones
                            new_p1 = (curr_p1 > 0.5) & (prev_state_tensor[0] < 0.5)
                            p1_moves = np.argwhere(new_p1)
                            for pos in p1_moves:
                                if pos[0] < self.board_shape[0] and pos[1] < self.board_shape[1]:  # Bounds check
                                    move_history_p1.append((pos[0], pos[1]))
                            
                            # Check for new player 2 stones
                            new_p2 = (curr_p2 > 0.5) & (prev_state_tensor[1] < 0.5)
                            p2_moves = np.argwhere(new_p2)
                            for pos in p2_moves:
                                if pos[0] < self.board_shape[0] and pos[1] < self.board_shape[1]:  # Bounds check
                                    move_history_p2.append((pos[0], pos[1]))
                        else:
                            # For other games, use difference method
                            diff = curr_board - prev_board
                            move_positions = np.argwhere(np.abs(diff) > 0.5)
                            
                            if len(move_positions) > 0:
                                # The player who made the move is the current player of the previous state
                                try:
                                    move_player = self.get_current_player(full_history[i-1])
                                except Exception:
                                    continue  # Skip if we can't get player info
                                
                                for pos in move_positions:
                                    row, col = pos[0], pos[1]
                                    if row < self.board_shape[0] and col < self.board_shape[1]:  # Bounds check
                                        if move_player == 1:
                                            move_history_p1.append((row, col))
                                        else:
                                            move_history_p2.append((row, col))
                        
                        # Stop if we have enough moves for both players
                        if len(move_history_p1) >= 8 and len(move_history_p2) >= 8:
                            break
        
        # Create move history planes for player 1
        for i in range(8):
            move_plane = np.zeros(self.board_shape[:2], dtype=np.float32)
            if i < len(move_history_p1):
                row, col = move_history_p1[i]
                move_plane[row, col] = 1.0
            features.append(move_plane)
            
        # Create move history planes for player 2
        for i in range(8):
            move_plane = np.zeros(self.board_shape[:2], dtype=np.float32)
            if i < len(move_history_p2):
                row, col = move_history_p2[i]
                move_plane[row, col] = 1.0
            features.append(move_plane)
        
        # Channel 18: Attack score plane
        # Channel 19: Defense score plane
        
        # GPU attack/defense acceleration removed in streamlined build
        
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
                    board = self.state_to_numpy(state, representation_type='standard')
                    if len(board.shape) == 3:
                        if self.game_type in [GameType.GO, GameType.GOMOKU]:
                            board = board[0] - board[1]  # Player 1 minus Player 2
                        else:
                            board = board.sum(axis=0)  # Sum channels for basic board
                    attack_plane, defense_plane = compute_attack_defense_scores(
                        self.game_type.value, board, self.get_current_player(state)
                    )
                    features.append(attack_plane)
                    features.append(defense_plane)
            else:
                # Use Python implementation if C++ not available
                from .attack_defense import compute_attack_defense_scores
                board = self.state_to_numpy(state, representation_type='standard')
                if len(board.shape) == 3:
                    if self.game_type in [GameType.GO, GameType.GOMOKU]:
                        board = board[0] - board[1]  # Player 1 minus Player 2
                    else:
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
        """Get hash of state for transposition table
        
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
        # Use the C++ enhanced representation which includes all 20 channels
        return self._current_state.get_enhanced_tensor_representation()
        
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
        return state.is_legal_move(move)
    
    def undo_move(self, state: Any) -> None:
        """Undo the last move (modifies state in-place)"""
        state.undo_move()
    
    def get_game_result(self, state: Any) -> str:
        """Get game result (ONGOING, WIN_PLAYER1, WIN_PLAYER2, DRAW)"""
        result = state.get_game_result()
        return str(result).split('.')[-1]  # Extract enum name
    
    def get_board_size(self, state: Any) -> int:
        """Get board size from state"""
        return state.get_board_size()
    
    def get_action_space_size(self, state: Any) -> int:
        """Get total action space size"""
        action_space = state.get_action_space_size()
        logger.debug(f"[GAME_INTERFACE DEBUG] Action space size: {action_space} (board: {self.board_size}x{self.board_size})")
        return action_space
    
    def get_tensor_representation(self, state: Any) -> np.ndarray:
        """Get basic tensor representation from C++"""
        return state.get_tensor_representation()
    
    def get_enhanced_tensor_representation(self, state: Any) -> np.ndarray:
        """Get enhanced tensor representation with additional features"""
        return state.get_enhanced_tensor_representation()
    
    def action_to_string(self, state: Any, action: int) -> str:
        """Convert action to human-readable string"""
        return state.action_to_string(action)
    
    def string_to_action(self, state: Any, move_str: str) -> int:
        """Convert string to action index"""
        return state.string_to_action(move_str)
    
    def to_string(self, state: Any) -> str:
        """Get string representation of the state"""
        return state.to_string()
    
    def get_move_history(self, state: Any) -> List[int]:
        """Get history of moves played"""
        return state.get_move_history()
    
    def clone_state(self, state: Any) -> Any:
        """Create a deep copy of the state"""
        return state.clone()
    
    def get_canonical_form(self, state: Any, player: int = None) -> np.ndarray:
        """Get canonical form of the state from current player's perspective
        
        For most games, the state is already from the current player's perspective.
        This method exists for compatibility with training pipelines.
        
        Args:
            state: Game state
            player: Player perspective (optional, for compatibility)
            
        Returns:
            State tensor from current player's perspective
        """
        # The state is already from the current player's perspective
        # The neural network input will handle any necessary transformations
        return self.state_to_numpy(state, representation_type=self.input_representation)
    
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
        
        Note: This method has been retained for backward compatibility but
        get_winner() is preferred for clearer perspective handling.
        
        Args:
            state: Terminal game state
            
        Returns:
            1.0 if last player won, -1.0 if they lost, 0.0 for draw
        """
        if not self.is_terminal(state):
            return 0.0
            
        winner = self.get_winner(state)
        # After a terminal move, current_player has switched
        # So we need to check from the previous player's perspective
        last_player = 1 - self.get_current_player(state)
        
        if winner == 0:  # Draw
            return 0.0
        elif winner == 1:  # Player 1 wins
            return 1.0 if last_player == 0 else -1.0
        else:  # Player 2 wins
            return -1.0 if last_player == 0 else 1.0

