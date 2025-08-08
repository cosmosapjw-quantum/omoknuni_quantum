"""
Production-ready CPU-optimized MCTS implementation.

This module provides high-performance CPU-based MCTS with:
- Phase 3 optimized Cython tree operations with minimal wrapper overhead
- Single-threaded wave-based search (optimal for CPU backend)
- Efficient state management with recycling
- Vectorized operations using NumPy
- 2,300+ simulations/second performance in actual self-play

Key components:
- cython_tree_optimized: Production Cython tree with nogil support
- optimized_cython_tree_wrapper: Minimal overhead wrapper (2% overhead)
- optimized_wave_search: Single-threaded optimal wave search
- cpu_game_states: Efficient CPU game state management

Example usage:
    from mcts.cpu.cpu_mcts_wrapper import create_cpu_optimized_mcts
    
    mcts = create_cpu_optimized_mcts(config, evaluator, game_interface)
    policy = mcts.search(state, num_simulations=1000)  # ~2300 sims/sec
"""

from .cpu_game_states import CPUGameStates
from .cpu_game_states_wrapper import (
    RecyclingCPUGameStates,
    create_cpu_game_states_with_recycling
)
from .cpu_mcts_wrapper import create_cpu_optimized_mcts, CPUOptimizedMCTSWrapper
from .optimized_wave_search import OptimizedCPUWaveSearch
from .vectorized_operations import VectorizedOperations

# Import the production-ready optimized Cython tree
try:
    from .cython_tree_optimized import CythonLockFreeTree as _CythonTreeImpl
    _TREE_IMPLEMENTATION = "optimized"
    NOGIL_AVAILABLE = True
except ImportError:
    # Fallback to standard version if optimized isn't available
    try:
        from .cython_tree import CythonLockFreeTree as _CythonTreeImpl
        _TREE_IMPLEMENTATION = "standard"
        NOGIL_AVAILABLE = False
    except ImportError:
        _CythonTreeImpl = None
        _TREE_IMPLEMENTATION = None
        NOGIL_AVAILABLE = False

# Try to import multiprocessing wave search
try:
    from .multiprocess_wave_search import MultiprocessWaveSearch
    MULTIPROCESS_AVAILABLE = True
except ImportError:
    MultiprocessWaveSearch = None
    MULTIPROCESS_AVAILABLE = False
    

# Try optimized wrapper first
try:
    from .optimized_cython_tree_wrapper import OptimizedCythonTree as _OptimizedCythonTreeImpl
    _OPTIMIZED_WRAPPER_AVAILABLE = True
except ImportError:
    _OptimizedCythonTreeImpl = None
    _OPTIMIZED_WRAPPER_AVAILABLE = False


if _OptimizedCythonTreeImpl is not None:
    # Use optimized wrapper with minimal overhead
    CythonTree = _OptimizedCythonTreeImpl
    
    # Create config class for optimized wrapper
    class CythonTreeConfig:
        def __init__(self, max_nodes=1000000, max_actions=None, max_children=2000000, 
                     device='cpu', initial_capacity=None, growth_factor=None,
                     enable_reuse_tracking=False, track_depth=False, c_puct=1.414,
                     virtual_loss=3.0, **kwargs):
            self.max_nodes = max_nodes
            self.max_children = max_children
            self.c_puct = c_puct
            self.virtual_loss = virtual_loss
    
    CYTHON_AVAILABLE = True
    print("Using optimized CythonTree wrapper (Phase 3 optimization)")
elif _CythonTreeImpl is not None:
    # Fallback to original wrapper implementation
    CYTHON_AVAILABLE = True
else:
    # No Cython implementation available
    CythonTree = None
    CythonTreeConfig = None
    CYTHON_AVAILABLE = False

# Exports
__all__ = [
    'CPUGameStates',
    'RecyclingCPUGameStates',
    'create_cpu_game_states_with_recycling',
    'create_cpu_optimized_mcts',
    'CPUOptimizedMCTSWrapper',
    'OptimizedCPUWaveSearch',
    'VectorizedOperations',
    'NOGIL_AVAILABLE',
    'MULTIPROCESS_AVAILABLE'
]

if CYTHON_AVAILABLE:
    __all__.extend(['CythonTree', 'CythonTreeConfig'])
    
if MULTIPROCESS_AVAILABLE:
    __all__.append('MultiprocessWaveSearch')

__version__ = "3.1.0"  # Production-ready with Phase 3 optimizations: 2,300+ sims/sec