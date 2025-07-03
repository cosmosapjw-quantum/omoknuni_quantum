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
    
    # Device configuration
    device: str = 'cuda'
    
    # Game configuration
    game_type: Union[GameType, LegacyGameType] = GameType.GOMOKU
    board_size: int = 15
    
    # Quantum features - Full integration (v1 and v2)
    enable_quantum: bool = False
    quantum_config: Optional['QuantumConfig'] = None
    quantum_version: str = 'v2'  # 'v1' or 'v2'
    
    # v2.0 specific quantum parameters
    quantum_branching_factor: Optional[int] = None  # Auto-detect if None
    quantum_avg_game_length: Optional[int] = None   # Auto-detect if None
    enable_phase_adaptation: bool = True
    envariance_threshold: float = 1e-3
    envariance_check_interval: int = 1000
    
    # Virtual loss for leaf parallelization
    enable_virtual_loss: bool = True
    virtual_loss: float = 1.0  # Positive value (will be negated when applied)
    # Note: virtual_loss_value parameter removed - use virtual_loss instead
    
    # Memory configuration
    memory_pool_size_mb: int = 2048
    max_tree_nodes: int = 500000
    use_mixed_precision: bool = True
    use_cuda_graphs: bool = True
    use_tensor_cores: bool = True
    
    # Progressive expansion
    initial_children_per_expansion: int = 8
    max_children_per_node: int = 50
    progressive_expansion_threshold: int = 5
    
    # Legacy parameters (for compatibility)
    target_sims_per_second: int = 100000
    cache_legal_moves: bool = True
    cache_features: bool = True
    use_zobrist_hashing: bool = True
    tree_batch_size: int = 1024
    
    # Subtree reuse configuration
    enable_subtree_reuse: bool = True  # Reuse search tree between moves
    subtree_reuse_min_visits: int = 10  # Min visits to preserve a subtree node
    
    # Debug options
    enable_debug_logging: bool = False
    enable_state_pool_debug: bool = False  # Specific logging for state pool management
    profile_gpu_kernels: bool = False
    
    def get_or_create_quantum_config(self) -> 'QuantumConfig':
        """Get quantum config, creating default if needed"""
        if self.quantum_config is None:
            # Create unified quantum config compatible with new implementation
            from ..quantum import QuantumMode, QuantumConfig
            
            # Determine quantum mode based on legacy settings
            if not self.enable_quantum:
                quantum_mode = QuantumMode.CLASSICAL
            elif self.quantum_version == 'v2' or self.enable_phase_adaptation:
                quantum_mode = QuantumMode.PRAGMATIC
            else:
                quantum_mode = QuantumMode.MINIMAL
            
            self.quantum_config = QuantumConfig(
                quantum_mode=quantum_mode,
                base_c_puct=self.c_puct,
                device=self.device,
                enable_phase_adaptation=self.enable_phase_adaptation,
                enable_power_law_annealing=True if quantum_mode != QuantumMode.CLASSICAL else False
            )
        return self.quantum_config
    
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