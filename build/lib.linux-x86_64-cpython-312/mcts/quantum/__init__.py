"""Quantum-inspired MCTS enhancements (v1.0 and v2.0)"""

# v1.0 imports (maintained for backward compatibility)
from .quantum_features import (
    QuantumConfig,
    QuantumMCTS,
)

# v2.0 imports
from .quantum_features_v2 import (
    QuantumConfigV2,
    QuantumMCTSV2,
    MCTSPhase,
    DiscreteTimeEvolution,
    PhaseDetector,
    OptimalParameters
)

# Unified wrapper for migration support
from .quantum_mcts_wrapper import (
    QuantumMCTSWrapper,
    UnifiedQuantumConfig,
    create_quantum_mcts,  # This replaces the v1 version
    compare_versions
)

# Core components (support both v1 and v2)
from ..utils.config_system import QuantumLevel
from .qft_engine import (
    QFTEngine,
    create_qft_engine,
    create_qft_engine_v1,
    create_qft_engine_v2
)
from .path_integral import (
    PathIntegral, 
    PathIntegralConfig,
    create_path_integral,
    create_path_integral_v1,
    create_path_integral_v2
)
from .rg_flow import RGFlowOptimizer, RGConfig
from .quantum_darwinism import QuantumDarwinismEngine, DarwinismConfig
from .decoherence import (
    DecoherenceEngine, 
    DecoherenceConfig,
    create_decoherence_engine,
    create_decoherence_engine_v1,
    create_decoherence_engine_v2,
    MCTSPhase as DecoherenceMCTSPhase  # Alias to avoid confusion
)
from .interference_gpu import MinHashInterference, MinHashConfig

__all__ = [
    # v1.0 exports (maintained for compatibility)
    "QuantumConfig",
    "QuantumMCTS",
    
    # v2.0 exports
    "QuantumConfigV2",
    "QuantumMCTSV2",
    "MCTSPhase",
    "DiscreteTimeEvolution",
    "PhaseDetector",
    "OptimalParameters",
    
    # Unified interface
    "QuantumMCTSWrapper",
    "UnifiedQuantumConfig",
    "create_quantum_mcts",  # Now returns wrapper by default
    "compare_versions",
    
    # Core components
    "QuantumLevel",
    "QFTEngine",
    "create_qft_engine",
    "create_qft_engine_v1",
    "create_qft_engine_v2",
    "PathIntegral",
    "PathIntegralConfig",
    "create_path_integral",
    "create_path_integral_v1",
    "create_path_integral_v2",
    "RGFlowOptimizer",
    "RGConfig",
    "QuantumDarwinismEngine",
    "DarwinismConfig",
    "DecoherenceEngine",
    "DecoherenceConfig",
    "create_decoherence_engine",
    "create_decoherence_engine_v1",
    "create_decoherence_engine_v2",
    "MinHashInterference",
    "MinHashConfig",
]