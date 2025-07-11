"""MCTS Configuration classes and utilities

This module contains configuration classes for the high-performance MCTS implementation.
Separated from main MCTS code for better maintainability and modularity.
"""

import torch
from typing import Dict, Optional, Union
from dataclasses import dataclass

from ..gpu.gpu_game_states import GameType
from .game_interface import GameType as LegacyGameType


@dataclass
class MCTSConfig:
    """Configuration for optimized MCTS"""
    
    # Pre-computed mappings for optimization
    _LEGACY_GAME_TYPE_MAP = {
        LegacyGameType.CHESS: GameType.CHESS,
        LegacyGameType.GO: GameType.GO,
        LegacyGameType.GOMOKU: GameType.GOMOKU
    }
    
    _DEFAULT_BOARD_SIZES = {
        GameType.CHESS: 8,
        GameType.GO: 19,
        GameType.GOMOKU: 15
    }
    
    # Core parameters
    num_simulations: int = 10000
    c_puct: float = 1.414
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    
    # Performance optimization modes
    classical_only_mode: bool = False  # Aggressive fast-path for classical MCTS
    enable_fast_ucb: bool = True       # Use optimized UCB kernel when available
    
    # Wave parallelization - CRITICAL for performance
    wave_size: Optional[int] = None  # Auto-determine if None
    min_wave_size: int = 3072
    max_wave_size: int = 3072  # Fixed size for best performance
    wave_min_size: int = 256  # Minimum wave size for adaptive sizing
    wave_max_size: int = 2048  # Maximum wave size for adaptive sizing
    wave_adaptive_sizing: bool = True
    wave_target_sims_per_second: int = 100000
    wave_target_gpu_utilization: float = 0.95
    wave_num_pipelines: int = 3
    wave_async_expansion: bool = True
    wave_prefetch_evaluations: bool = True
    wave_memory_pool_mb: int = 1024
    
    # Device configuration
    device: str = 'cuda'
    
    # Game configuration
    game_type: Union[GameType, LegacyGameType] = GameType.GOMOKU
    board_size: int = 15
    
    # Quantum features disabled
    enable_quantum: bool = False
    
    # Virtual loss for leaf parallelization
    enable_virtual_loss: bool = True
    virtual_loss: float = 1.0  # Positive value (will be negated when applied)
    # Note: virtual_loss_value parameter removed - use virtual_loss instead
    
    # Memory configuration
    memory_pool_size_mb: int = 2048
    max_tree_nodes: int = 500000
    tree_memory_fraction: float = 0.4
    buffer_memory_fraction: float = 0.3
    max_tree_nodes_per_worker: int = 800000
    use_mixed_precision: bool = True
    use_cuda_graphs: bool = True
    use_tensor_cores: bool = True
    enable_kernel_fusion: bool = True
    gpu_memory_fraction: float = 0.9
    initial_capacity_factor: float = 0.5  # Pre-allocate 50% to avoid reallocations
    enable_memory_pooling: bool = True    # Use memory pools for efficiency
    
    # Progressive expansion
    initial_children_per_expansion: int = 8
    max_children_per_node: int = 50
    progressive_expansion_threshold: int = 5
    
    # CSR and sparse operations
    csr_max_actions: int = 10
    csr_use_sparse_operations: bool = True
    
    # CPU thread configuration
    cpu_threads_per_worker: int = 1
    
    # Batch and timing configuration
    batch_size: int = 256  # MCTS batch size
    inference_batch_size: int = 256  # NN inference batch size
    tree_batch_size: int = 8  # Batch size for tree operations
    gpu_batch_timeout: float = 0.020  # GPU batch timeout in seconds (20ms)
    worker_batch_timeout: float = 0.050  # Worker batch timeout in seconds (50ms)
    max_coordination_batch_size: int = 128
    training_batch_queue_size: int = 10
    
    # Legacy parameters (for compatibility)
    target_sims_per_second: int = 100000
    cache_legal_moves: bool = True
    cache_features: bool = True
    use_zobrist_hashing: bool = True
    
    # Subtree reuse configuration
    enable_subtree_reuse: bool = True  # Reuse search tree between moves
    subtree_reuse_min_visits: int = 10  # Min visits to preserve a subtree node
    
    # Debug options
    enable_debug_logging: bool = False
    enable_state_pool_debug: bool = False  # Specific logging for state pool management
    profile_gpu_kernels: bool = False
    
    # Tactical move detection (helps find important moves with untrained network)
    enable_tactical_boost: bool = True
    tactical_boost_strength: float = 0.3  # 0.0 = no boost, 1.0 = full replacement
    tactical_boost_decay: float = 0.99  # Decay factor per iteration to reduce over time
    
    # Gomoku tactical parameters
    gomoku_win_boost: float = 100.0      # Immediate win move
    gomoku_block_win_boost: float = 90.0 # Block opponent win
    gomoku_open_four_boost: float = 50.0 # Create open four
    gomoku_block_four_boost: float = 45.0 # Block opponent open four
    gomoku_threat_base_boost: float = 40.0 # Base for multiple threats
    gomoku_threat_multiplier: float = 5.0  # Per additional threat
    gomoku_three_boost: float = 20.0     # Create three in a row
    gomoku_block_three_boost: float = 18.0 # Block opponent three
    gomoku_center_boost: float = 3.0     # Central position bonus
    gomoku_connection_boost: float = 2.0 # Per connection bonus
    
    # Chess tactical parameters  
    chess_capture_good_base: float = 10.0 # Base good capture bonus
    chess_capture_equal: float = 5.0     # Equal trade bonus
    chess_capture_bad: float = 2.0       # Bad capture bonus
    chess_check_boost: float = 8.0       # Check move bonus
    chess_check_capture_boost: float = 4.0 # Check + capture bonus
    chess_promotion_boost: float = 9.0   # Pawn promotion bonus
    chess_center_boost: float = 1.0      # Central control bonus
    chess_center_core_boost: float = 0.5 # Center core bonus
    chess_development_boost: float = 2.0 # Piece development bonus
    chess_castling_boost: float = 6.0    # Castling bonus
    
    # Go tactical parameters
    go_capture_boost: float = 15.0       # Capture opponent stones
    go_escape_boost: float = 12.0        # Escape from atari
    go_atari_boost: float = 10.0         # Put opponent in atari
    go_save_boost: float = 8.0           # Save own stones
    go_territory_boost: float = 5.0      # Territory control
    go_connection_boost: float = 3.0     # Connect stones
    go_eye_boost: float = 7.0            # Eye formation
    go_corner_boost: float = 2.0         # Corner moves bonus
    
    def get_or_create_quantum_config(self):
        """Placeholder for quantum config (disabled)"""
        return None
    
    def _estimate_branching_factor(self) -> int:
        """Estimate branching factor based on game type"""
        if self.game_type in [GameType.GOMOKU, LegacyGameType.GOMOKU]:
            return self.board_size * self.board_size
        elif self.game_type in [GameType.GO, LegacyGameType.GO]:
            return self.board_size * self.board_size + 1  # +1 for pass
        elif self.game_type in [GameType.CHESS, LegacyGameType.CHESS]:
            return 35  # Average chess branching factor
        else:
            return 50  # Default estimate
    
    def _estimate_game_length(self) -> int:
        """Estimate average game length based on game type"""
        if self.game_type in [GameType.GOMOKU, LegacyGameType.GOMOKU]:
            return self.board_size * self.board_size // 2
        elif self.game_type in [GameType.GO, LegacyGameType.GO]:
            return self.board_size * self.board_size * 2
        elif self.game_type in [GameType.CHESS, LegacyGameType.CHESS]:
            return 80  # Average chess game length
        else:
            return 100  # Default estimate
    
    def __post_init__(self):
        # Convert legacy GameType if needed
        if isinstance(self.game_type, LegacyGameType):
            self.game_type = self._LEGACY_GAME_TYPE_MAP[self.game_type]
            
        # Set board size defaults based on game
        if self.board_size is None:
            self.board_size = self._DEFAULT_BOARD_SIZES.get(self.game_type, 15)


def create_optimized_config(
    game_type: Union[GameType, LegacyGameType] = GameType.GOMOKU,
    num_simulations: int = 10000,
    device: str = 'cuda',
    enable_quantum: bool = False,
    **kwargs
) -> MCTSConfig:
    """Create an optimized MCTS configuration
    
    Args:
        game_type: Type of game
        num_simulations: Number of MCTS simulations per search
        device: Device to run on ('cuda' or 'cpu')
        enable_quantum: Enable quantum-inspired enhancements
        **kwargs: Additional configuration parameters
        
    Returns:
        Optimized MCTSConfig instance
    """
    return MCTSConfig(
        game_type=game_type,
        num_simulations=num_simulations,
        device=device,
        enable_quantum=enable_quantum,
        **kwargs
    )


def create_performance_config(
    wave_size: int = 3072,
    memory_pool_mb: int = 2048,
    **kwargs
) -> MCTSConfig:
    """Create configuration optimized for maximum performance
    
    Args:
        wave_size: Size of MCTS wave for parallelization
        memory_pool_mb: Memory pool size in MB
        **kwargs: Additional configuration parameters
        
    Returns:
        Performance-optimized MCTSConfig instance
    """
    return MCTSConfig(
        wave_size=wave_size,
        memory_pool_size_mb=memory_pool_mb,
        classical_only_mode=True,
        enable_fast_ucb=True,
        use_mixed_precision=True,
        use_cuda_graphs=True,
        use_tensor_cores=True,
        **kwargs
    )