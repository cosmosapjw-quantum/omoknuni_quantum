"""
CPU-optimized MCTS implementation.

This module provides high-performance CPU-based MCTS with:
- Optimized wave-based search with evaluation caching
- Efficient state management with recycling
- Vectorized operations using NumPy
- Progressive widening for tree efficiency
- 850+ simulations/second performance

Example usage:
    from mcts.cpu.cpu_mcts_wrapper import create_cpu_optimized_mcts
    
    mcts = create_cpu_optimized_mcts(config, evaluator, game_interface)
    policy = mcts.search(state, num_simulations=1000)
"""

from .cpu_game_states import CPUGameStates
from .cpu_game_states_wrapper import (
    RecyclingCPUGameStates,
    create_cpu_game_states_with_recycling
)
from .cpu_mcts_wrapper import create_cpu_optimized_mcts, CPUOptimizedMCTSWrapper
from .optimized_wave_search import OptimizedCPUWaveSearch
from .vectorized_operations import VectorizedOperations

# Try to import thread-safe Cython tree first
try:
    # Import thread-safe version if available
    from .build.cython_tree_safe import CythonThreadSafeTree as _CythonTreeImpl
    _TREE_IMPLEMENTATION = "thread_safe"
except ImportError:
    # Fall back to lock-free version
    try:
        from .build.cython_tree import CythonLockFreeTree as _CythonTreeImpl
        _TREE_IMPLEMENTATION = "lock_free"
    except ImportError:
        _CythonTreeImpl = None
        _TREE_IMPLEMENTATION = None
    
if _CythonTreeImpl is not None:
    # Create a wrapper that provides compatibility layer
    class CythonTree:
        def __init__(self, config):
            # Store config values for later use
            self._max_nodes = config.max_nodes
            self._max_children = getattr(config, 'max_children', 2000000)
            
            # Log which implementation is being used
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Using {_TREE_IMPLEMENTATION} Cython tree implementation")
            
            # Extract parameters from config and create tree
            self._tree = _CythonTreeImpl(
                max_nodes=self._max_nodes,
                max_children=self._max_children,
                c_puct=getattr(config, 'c_puct', 1.414),
                virtual_loss_value=getattr(config, 'virtual_loss', 3.0)
            )
        
        # Property to expose num_nodes
        @property
        def num_nodes(self):
            return self._tree.get_num_nodes()
        
        # Property for max_nodes (for compatibility)
        @property
        def max_nodes(self):
            return self._max_nodes
        
        # Property for node_data (for compatibility)
        @property
        def node_data(self):
            # Return the tree itself as it has node_data property
            return self._tree.node_data
            
        # Provide get_stats method
        def get_stats(self):
            """Return tree statistics"""
            root_visits = 0
            if self._tree.get_num_nodes() > 0:
                root_visits = self._tree.get_visit_count(0)  # Root is node 0
            
            return {
                'num_nodes': self._tree.get_num_nodes(),
                'max_nodes': self._max_nodes,
                'max_children': self._max_children,
                'root_visits': root_visits,
            }
        
        # Properties for GPU compatibility
        @property
        def children(self):
            # Mock property for GPU compatibility
            class MockChildren:
                def __setitem__(self, key, value):
                    pass  # No-op
            return MockChildren()
        
        @property
        def csr_storage(self):
            # Mock CSR storage for GPU compatibility
            class MockCSR:
                def __init__(self, tree):
                    self.tree = tree
                @property
                def row_ptr(self):
                    return [0, 0]
                def get_memory_usage_mb(self):
                    return 0.0
            return MockCSR(self._tree)
        
        # Add get_child_by_action method explicitly
        def get_child_by_action(self, node_idx, action):
            """Get child node index by action"""
            return self._tree.get_child_by_action(node_idx, action)
        
        # Add shift_root method explicitly  
        def shift_root(self, new_root_idx):
            """Shift root to new node"""
            return self._tree.shift_root(new_root_idx)
        
        
        # Forward all other attribute access to the underlying tree
        def __getattr__(self, name):
            return getattr(self._tree, name)
    
    # Create a config class that accepts various parameters but only uses what's needed
    class CythonTreeConfig:
        def __init__(self, max_nodes=1000000, max_actions=None, max_children=2000000, 
                     device='cpu', initial_capacity=None, growth_factor=None,
                     enable_reuse_tracking=False, track_depth=False, c_puct=1.414,
                     virtual_loss=3.0, **kwargs):
            self.max_nodes = max_nodes
            self.max_children = max_children
            self.c_puct = c_puct
            self.virtual_loss = virtual_loss
            # max_actions is not used by CythonLockFreeTree, but we accept it for compatibility
    
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
    'VectorizedOperations'
]

if CYTHON_AVAILABLE:
    __all__.extend(['CythonTree', 'CythonTreeConfig'])

__version__ = "2.1.0"  # Cleaned and optimized version