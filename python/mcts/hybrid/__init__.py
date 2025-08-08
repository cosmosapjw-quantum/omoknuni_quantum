"""Hybrid MCTS - High-Performance CPU/GPU Implementation

This module provides the hybrid CPU/GPU MCTS implementation with
Cython-optimized tree operations for maximum performance.
"""

# Try to import Cython hybrid backend
try:
    from .cython_hybrid_backend import CythonHybridBackend
    CYTHON_HYBRID_AVAILABLE = True
except ImportError:
    CYTHON_HYBRID_AVAILABLE = False

from .hybrid_mcts_factory import create_hybrid_mcts
from .hybrid_wave_search import HybridWaveSearch

__all__ = [
    'CythonHybridBackend',
    'CYTHON_HYBRID_AVAILABLE',
    'create_hybrid_mcts',
    'HybridWaveSearch'
]