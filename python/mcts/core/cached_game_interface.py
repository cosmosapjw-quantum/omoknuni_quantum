"""Cached game interface for optimized MCTS

This implementation caches expensive game operations:
- Legal move generation (88ms in profiling)
- State to numpy conversions (70ms in profiling)
- Board feature extraction
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import logging

from .game_interface import GameInterface, GameType

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for game interface caching"""
    max_legal_moves_cache: int = 50000  # Increased from 10000
    max_features_cache: int = 25000     # Increased from 5000
    max_state_hash_cache: int = 100000  # Increased from 20000
    use_zobrist_hashing: bool = True
    cache_ttl_seconds: float = 3600.0  # 1 hour
    
class StateHasher:
    """Fast state hashing for cache keys"""
    
    def __init__(self, game_type: GameType, use_zobrist: bool = True):
        self.game_type = game_type
        self.use_zobrist = use_zobrist
        
        if use_zobrist:
            self._init_zobrist_tables()
            
    def _init_zobrist_tables(self):
        """Initialize Zobrist hashing tables"""
        # Random numbers for each piece at each position
        np.random.seed(42)  # Reproducible
        
        if self.game_type == GameType.GOMOKU:
            # 15x15 board, 3 states (empty=0, black=1, white=2)
            self.zobrist_table = np.random.randint(
                0, 2**63, size=(15, 15, 3), dtype=np.int64
            )
            # Additional hash for current player turn
            self.zobrist_turn = np.random.randint(0, 2**63, dtype=np.int64)
            
        elif self.game_type == GameType.GO:
            # 19x19 board, 3 states (empty=0, black=1, white=2)
            self.zobrist_table = np.random.randint(
                0, 2**63, size=(19, 19, 3), dtype=np.int64
            )
            # Additional hashes for ko position
            self.zobrist_ko = np.random.randint(0, 2**63, size=(19, 19), dtype=np.int64)
            self.zobrist_turn = np.random.randint(0, 2**63, dtype=np.int64)
            
        elif self.game_type == GameType.CHESS:
            # 8x8 board, 6 piece types * 2 colors + empty = 13
            self.zobrist_table = np.random.randint(
                0, 2**63, size=(8, 8, 13), dtype=np.int64
            )
            # Additional hashes for castling rights (4 bits), en passant (8 files)
            self.zobrist_castling = np.random.randint(0, 2**63, size=(16,), dtype=np.int64)
            self.zobrist_en_passant = np.random.randint(0, 2**63, size=(8,), dtype=np.int64)
            self.zobrist_turn = np.random.randint(0, 2**63, dtype=np.int64)
            
    def hash_state(self, state: Any) -> int:
        """Compute fast hash of game state"""
        if self.use_zobrist and hasattr(state, 'board'):
            return self._zobrist_hash(state)
        else:
            # Fallback to standard hashing
            if hasattr(state, 'tobytes'):
                return hash(state.tobytes())
            else:
                return hash(str(state))
                
    def _zobrist_hash(self, state: Any) -> int:
        """Compute Zobrist hash of board state"""
        h = 0
        
        if self.game_type == GameType.GOMOKU:
            board = state.board
            # Hash board positions
            for i in range(15):
                for j in range(15):
                    piece = board[i, j]
                    # piece: 0=empty, 1=player1, 2=player2
                    h ^= int(self.zobrist_table[i, j, piece])
            
            # Hash current player
            if hasattr(state, 'current_player'):
                if state.current_player == 2:
                    h ^= int(self.zobrist_turn)
                    
        elif self.game_type == GameType.GO:
            board = state.board if hasattr(state, 'board') else state.get_board()
            board_size = board.shape[0]
            
            # Hash board positions
            for i in range(board_size):
                for j in range(board_size):
                    piece = board[i, j]
                    # piece: 0=empty, 1=black, 2=white
                    h ^= int(self.zobrist_table[i, j, piece])
            
            # Hash ko position if exists
            if hasattr(state, 'ko_point') and state.ko_point is not None:
                ko_i, ko_j = state.ko_point
                h ^= int(self.zobrist_ko[ko_i, ko_j])
            
            # Hash current player
            if hasattr(state, 'current_player'):
                if state.current_player == 2:
                    h ^= int(self.zobrist_turn)
                    
        elif self.game_type == GameType.CHESS:
            board = state.board if hasattr(state, 'board') else state.get_board_array()
            
            # Hash board positions
            for i in range(8):
                for j in range(8):
                    piece = board[i, j]
                    if piece != 0:
                        # Map piece to Zobrist index
                        # piece values: 1-6 (white), -1 to -6 (black)
                        if piece > 0:
                            piece_idx = piece  # White pieces: 1-6
                        else:
                            piece_idx = 6 + abs(piece)  # Black pieces: 7-12
                        h ^= int(self.zobrist_table[i, j, piece_idx])
            
            # Hash castling rights (4 bits: K, Q, k, q)
            if hasattr(state, 'castling_rights'):
                castling_bits = 0
                if 'K' in state.castling_rights: castling_bits |= 1
                if 'Q' in state.castling_rights: castling_bits |= 2
                if 'k' in state.castling_rights: castling_bits |= 4
                if 'q' in state.castling_rights: castling_bits |= 8
                h ^= int(self.zobrist_castling[castling_bits])
            
            # Hash en passant square
            if hasattr(state, 'en_passant_target') and state.en_passant_target is not None:
                file_idx = ord(state.en_passant_target[0]) - ord('a')
                h ^= int(self.zobrist_en_passant[file_idx])
            
            # Hash current player
            if hasattr(state, 'current_player'):
                if state.current_player == -1:  # Black to move
                    h ^= int(self.zobrist_turn)
            
        return h

class LRUCache:
    """Enhanced LRU cache with collision detection"""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.collisions = 0
        # Store actual state data for collision detection
        self.state_verification = {}
        
    def get(self, key: Any, state_data: Optional[bytes] = None) -> Optional[Any]:
        """Get item from cache with collision detection"""
        if key in self.cache:
            # Verify no collision if state data provided
            if state_data is not None and key in self.state_verification:
                if self.state_verification[key] != state_data:
                    self.collisions += 1
                    logger.warning(f"Hash collision detected for key {key}")
                    return None
            
            self.hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        else:
            self.misses += 1
            return None
            
    def put(self, key: Any, value: Any, state_data: Optional[bytes] = None):
        """Put item in cache with collision detection"""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = value
            if state_data is not None:
                self.state_verification[key] = state_data
            
            if len(self.cache) > self.max_size:
                # Remove least recently used
                removed_key, _ = self.cache.popitem(last=False)
                if removed_key in self.state_verification:
                    del self.state_verification[removed_key]
                
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.state_verification.clear()
        self.hits = 0
        self.misses = 0
        self.collisions = 0
        
    @property
    def hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class CachedGameInterface:
    """Game interface with caching for expensive operations"""
    
    def __init__(self, base_interface: GameInterface, config: CacheConfig):
        self.base = base_interface
        self.config = config
        
        # Initialize caches
        self.legal_moves_cache = LRUCache(config.max_legal_moves_cache)
        self.features_cache = LRUCache(config.max_features_cache)
        self.state_hash_cache = LRUCache(config.max_state_hash_cache)
        
        # Initialize hasher
        self.hasher = StateHasher(base_interface.game_type, config.use_zobrist_hashing)
        
        # Statistics
        self.stats = {
            'legal_moves_calls': 0,
            'features_calls': 0,
            'hash_calls': 0
        }
        
    def get_legal_moves(self, state: Any, shuffle: bool = True) -> List[int]:
        """Get legal moves with caching"""
        self.stats['legal_moves_calls'] += 1
        
        # Get state hash and verification data
        state_hash = self._get_state_hash(state)
        state_bytes = self._get_state_bytes(state)
        
        # Check cache with collision detection
        cached_moves = self.legal_moves_cache.get(state_hash, state_bytes)
        if cached_moves is not None:
            if shuffle and len(cached_moves) > 1:
                # Return a shuffled copy of cached moves
                import random
                shuffled = list(cached_moves)
                random.shuffle(shuffled)
                return shuffled
            return list(cached_moves)  # Return copy to avoid mutation
            
        # Compute (without shuffling for cache)
        moves = self.base.get_legal_moves(state, shuffle=False)
        self.legal_moves_cache.put(state_hash, moves, state_bytes)
        
        # Now shuffle if requested
        if shuffle and len(moves) > 1:
            import random
            shuffled = list(moves)
            random.shuffle(shuffled)
            return shuffled
            
        return moves
        
    def state_to_numpy(self, state: Any) -> np.ndarray:
        """Convert state to numpy features with caching"""
        self.stats['features_calls'] += 1
        
        # Get state hash and verification data
        state_hash = self._get_state_hash(state)
        state_bytes = self._get_state_bytes(state)
        
        # Check cache with collision detection
        cached_features = self.features_cache.get(state_hash, state_bytes)
        if cached_features is not None:
            return cached_features
            
        # Compute and cache
        features = self.base.state_to_numpy(state)
        self.features_cache.put(state_hash, features, state_bytes)
        
        return features
    
    def clone_state(self, state: Any) -> Any:
        """Clone a game state - delegates to base interface"""
        return self.base.clone_state(state)
    
    def apply_move(self, state: Any, move: int) -> Any:
        """Apply a move to a state - delegates to base interface"""
        return self.base.apply_move(state, move)
    
    def get_action_size(self) -> int:
        """Get action size from base interface"""
        return self.base.max_moves
    
    @property
    def max_moves(self) -> int:
        """Get max moves from base interface"""
        return self.base.max_moves
        
    def _get_state_hash(self, state: Any) -> int:
        """Get cached state hash"""
        self.stats['hash_calls'] += 1
        
        # For simple states, use id
        state_id = id(state)
        
        # Check hash cache
        cached_hash = self.state_hash_cache.get(state_id)
        if cached_hash is not None:
            return cached_hash
            
        # Compute and cache
        state_hash = self.hasher.hash_state(state)
        self.state_hash_cache.put(state_id, state_hash)
        
        return state_hash
        
    def _get_state_bytes(self, state: Any) -> bytes:
        """Get compact byte representation of state for collision detection"""
        if hasattr(state, 'board'):
            # For board games, use board array
            board_bytes = state.board.tobytes()
            player_byte = bytes([state.current_player]) if hasattr(state, 'current_player') else b''
            return board_bytes + player_byte
        elif hasattr(state, 'tobytes'):
            return state.tobytes()
        else:
            # Fallback to string representation
            return str(state).encode('utf-8')
        
    def batch_get_legal_moves(self, states: List[Any], shuffle: bool = True) -> List[List[int]]:
        """Get legal moves for multiple states efficiently"""
        results = []
        uncached_indices = []
        uncached_states = []
        
        # Check cache for each state
        for i, state in enumerate(states):
            state_hash = self._get_state_hash(state)
            cached_moves = self.legal_moves_cache.get(state_hash)
            
            if cached_moves is not None:
                # Shuffle cached moves if requested
                if shuffle and len(cached_moves) > 1:
                    import random
                    shuffled = list(cached_moves)
                    random.shuffle(shuffled)
                    results.append(shuffled)
                else:
                    results.append(list(cached_moves))
            else:
                results.append(None)
                uncached_indices.append(i)
                uncached_states.append(state)
                
        # Batch compute uncached states
        if uncached_states:
            # Could parallelize this with ThreadPoolExecutor
            for i, state in zip(uncached_indices, uncached_states):
                moves = self.base.get_legal_moves(state, shuffle=False)
                
                # Cache the unshuffled version
                state_hash = self._get_state_hash(state)
                self.legal_moves_cache.put(state_hash, moves)
                
                # Shuffle if requested
                if shuffle and len(moves) > 1:
                    import random
                    shuffled = list(moves)
                    random.shuffle(shuffled)
                    results[i] = shuffled
                else:
                    results[i] = moves
                
        return results
        
    def batch_get_legal_moves_tensor(self, board_tensors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get legal moves for multiple boards as tensors (for GPU processing)
        
        Args:
            board_tensors: (batch_size, 15, 15) tensor of board states
            
        Returns:
            legal_masks: (batch_size, 225) boolean tensor of legal moves
            num_legal: (batch_size,) count of legal moves per board
        """
        batch_size = board_tensors.shape[0]
        flat_boards = board_tensors.view(batch_size, -1)
        
        # For Gomoku, legal moves are empty squares (value == 0)
        legal_masks = (flat_boards == 0)
        num_legal = legal_masks.sum(dim=1)
        
        return legal_masks, num_legal
    
    def get_board_size(self) -> int:
        """Get the board size for the game"""
        if self.base.game_type == GameType.GOMOKU:
            return 15 * 15  # 225
        elif self.base.game_type == GameType.GO:
            return 19 * 19  # 361
        elif self.base.game_type == GameType.CHESS:
            return 8 * 8  # 64
        else:
            return self.base.max_moves
    
    def batch_state_to_numpy(self, states: List[Any]) -> torch.Tensor:
        """Convert multiple states to features efficiently"""
        # Try to use custom CUDA kernel if available
        if (hasattr(torch.ops, 'custom_cuda_ops') and 
            hasattr(torch.ops.custom_cuda_ops, 'evaluate_gomoku_positions') and
            self.base.game_type == GameType.GOMOKU):
            
            # Extract boards and players
            boards = []
            players = []
            
            for state in states:
                boards.append(state.board)
                players.append(state.current_player)
                
            boards_tensor = torch.tensor(np.array(boards), dtype=torch.int8, device='cuda')
            players_tensor = torch.tensor(players, dtype=torch.int8, device='cuda')
            
            # Use CUDA kernel
            features = torch.ops.custom_cuda_ops.evaluate_gomoku_positions(
                boards_tensor, players_tensor
            )
            
            return features
            
        # Fallback to cached conversion
        features = []
        for state in states:
            features.append(self.state_to_numpy(state))
            
        return torch.from_numpy(np.array(features))
        
    def make_move(self, state: Any, action: int) -> Any:
        """Make move (no caching needed)"""
        return self.base.make_move(state, action)
        
    def is_terminal(self, state: Any) -> bool:
        """Check if state is terminal (could cache but usually fast)"""
        return self.base.is_terminal(state)
        
    def get_value(self, state: Any, player: int) -> float:
        """Get terminal value (could cache but usually fast)"""
        return self.base.get_value(state, player)
        
    def get_current_player(self, state: Any) -> int:
        """Get current player (no caching needed)"""
        return self.base.get_current_player(state)
        
    def render(self, state: Any) -> str:
        """Render state (no caching)"""
        return self.base.render(state)
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'legal_moves': {
                'calls': self.stats['legal_moves_calls'],
                'hit_rate': self.legal_moves_cache.hit_rate,
                'size': len(self.legal_moves_cache.cache),
                'collisions': self.legal_moves_cache.collisions
            },
            'features': {
                'calls': self.stats['features_calls'],
                'hit_rate': self.features_cache.hit_rate,
                'size': len(self.features_cache.cache),
                'collisions': self.features_cache.collisions
            },
            'state_hash': {
                'calls': self.stats['hash_calls'],
                'hit_rate': self.state_hash_cache.hit_rate,
                'size': len(self.state_hash_cache.cache),
                'collisions': self.state_hash_cache.collisions
            }
        }
        
    def clear_caches(self):
        """Clear all caches"""
        self.legal_moves_cache.clear()
        self.features_cache.clear()
        self.state_hash_cache.clear()
        logger.debug("Cleared all game interface caches")