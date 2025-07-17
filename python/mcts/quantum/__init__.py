"""
Quantum-inspired enhancements for MCTS.

This module implements quantum and statistical physics inspired
improvements to Monte Carlo Tree Search.
"""

# Only import new physics analysis modules to avoid circular dependencies
# The old quantum_mcts module has dependencies on removed modules

__all__ = []

# Try importing modules individually to avoid failures
try:
    from .finite_size_scaling import FiniteSizeScaling
    __all__.append('FiniteSizeScaling')
except ImportError:
    pass

try:
    from .subtree_extractor import SubtreeExtractor
    __all__.append('SubtreeExtractor')
except ImportError:
    pass

try:
    from .rg_flow_tracker import RGFlowTracker
    __all__.append('RGFlowTracker')
except ImportError:
    pass

try:
    from .quantum_darwinism_measurer import QuantumDarwinismMeasurer
    __all__.append('QuantumDarwinismMeasurer')
except ImportError:
    pass

try:
    from .pointer_state_analyzer import PointerStateAnalyzer
    __all__.append('PointerStateAnalyzer')
except ImportError:
    pass

try:
    from .non_equilibrium_analyzer import NonEquilibriumAnalyzer
    __all__.append('NonEquilibriumAnalyzer')
except ImportError:
    pass