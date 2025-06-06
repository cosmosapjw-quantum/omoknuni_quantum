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
            # 15x15 board, 2 players
            self.zobrist_table = np.random.randint(
                0, 2**63, size=(15, 15, 3), dtype=np.int64
            )
        elif self.game_type == GameType.GO:
            # 19x19 board, 2 players + ko
            self.zobrist_table = np.random.randint(
                0, 2**63, size=(19, 19, 3), dtype=np.int64
            )
        elif self.game_type == GameType.CHESS:
            # 8x8 board, 6 piece types * 2 colors
            self.zobrist_table = np.random.randint(
                0, 2**63, size=(8, 8, 13), dtype=np.int64  # 12 pieces + empty
            )
            
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
        board = state.board
        h = 0
        
        if self.game_type == GameType.GOMOKU:
            for i in range(15):
                for j in range(15):
                    piece = board[i, j]
                    if piece != 0:
                        h ^= int(self.zobrist_table[i, j, piece])
                        
        # Add current player to hash
        if hasattr(state, 'current_player'):
            h ^= int(state.current_player * 0x1234567890ABCDEF)
            
        return h

class LRUCache:
    """Simple LRU cache implementation"""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        
    def get(self, key: Any) -> Optional[Any]:
        """Get item from cache"""
        if key in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        else:
            self.misses += 1
            return None
            
    def put(self, key: Any, value: Any):
        """Put item in cache"""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = value
            if len(self.cache) > self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
                
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        
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
        
    def get_legal_moves(self, state: Any) -> List[int]:
        """Get legal moves with caching"""
        self.stats['legal_moves_calls'] += 1
        
        # Get state hash
        state_hash = self._get_state_hash(state)
        
        # Check cache
        cached_moves = self.legal_moves_cache.get(state_hash)
        if cached_moves is not None:
            return cached_moves
            
        # Compute and cache
        moves = self.base.get_legal_moves(state)
        self.legal_moves_cache.put(state_hash, moves)
        
        return moves
        
    def state_to_numpy(self, state: Any) -> np.ndarray:
        """Convert state to numpy features with caching"""
        self.stats['features_calls'] += 1
        
        # Get state hash
        state_hash = self._get_state_hash(state)
        
        # Check cache
        cached_features = self.features_cache.get(state_hash)
        if cached_features is not None:
            return cached_features
            
        # Compute and cache
        features = self.base.state_to_numpy(state)
        self.features_cache.put(state_hash, features)
        
        return features
        
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
        
    def batch_get_legal_moves(self, states: List[Any]) -> List[List[int]]:
        """Get legal moves for multiple states efficiently"""
        results = []
        uncached_indices = []
        uncached_states = []
        
        # Check cache for each state
        for i, state in enumerate(states):
            state_hash = self._get_state_hash(state)
            cached_moves = self.legal_moves_cache.get(state_hash)
            
            if cached_moves is not None:
                results.append(cached_moves)
            else:
                results.append(None)
                uncached_indices.append(i)
                uncached_states.append(state)
                
        # Batch compute uncached states
        if uncached_states:
            # Could parallelize this with ThreadPoolExecutor
            for i, state in zip(uncached_indices, uncached_states):
                moves = self.base.get_legal_moves(state)
                results[i] = moves
                
                # Cache the result
                state_hash = self._get_state_hash(state)
                self.legal_moves_cache.put(state_hash, moves)
                
        return results
        
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
                'size': len(self.legal_moves_cache.cache)
            },
            'features': {
                'calls': self.stats['features_calls'],
                'hit_rate': self.features_cache.hit_rate,
                'size': len(self.features_cache.cache)
            },
            'state_hash': {
                'calls': self.stats['hash_calls'],
                'hit_rate': self.state_hash_cache.hit_rate,
                'size': len(self.state_hash_cache.cache)
            }
        }
        
    def clear_caches(self):
        """Clear all caches"""
        self.legal_moves_cache.clear()
        self.features_cache.clear()
        self.state_hash_cache.clear()
        logger.info("Cleared all game interface caches")