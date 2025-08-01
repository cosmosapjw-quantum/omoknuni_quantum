"""Hybrid MCTS - High-Performance CPU/GPU Implementation

This module provides optimized components for hybrid CPU/GPU MCTS:
- Fast Cython tree operations (36M+ selections/sec)
- Lock-free SPSC queues
- Thread-local buffers
- Memory pools
- SIMD UCB calculations
"""

from .optimized_tree import OptimizedTree
from .spsc_queue import SPSCQueue
from .thread_local_buffers import ThreadLocalBuffer, ThreadLocalBufferManager
from .memory_pool import ObjectPool, ThreadLocalMemoryPool
from .simd_ucb import SIMDUCBCalculator
from .cpu_wave_search import CPUWaveSearch

# Try to import Cython module
try:
    from .cython_tree_ops_fast import FastCythonTree
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False

from .hybrid_mcts_factory import create_hybrid_mcts, create_fast_hybrid_mcts
from .fast_hybrid_mcts import FastHybridMCTS

__all__ = [
    'OptimizedTree',
    'SPSCQueue',
    'ThreadLocalBuffer',
    'ThreadLocalBufferManager',
    'ObjectPool',
    'ThreadLocalMemoryPool',
    'SIMDUCBCalculator',
    'CPUWaveSearch',
    'FastCythonTree',
    'CYTHON_AVAILABLE',
    'create_hybrid_mcts',
    'create_fast_hybrid_mcts',
    'FastHybridMCTS'
]