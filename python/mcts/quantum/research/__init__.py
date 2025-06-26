"""Quantum Path Integral MCTS Research Implementation

This package provides a complete research implementation of quantum-enhanced MCTS
based on rigorous path integral formulation from quantum field theory. 

Key modules:
- path_integral_engine: Core path integral computation with Lagrangian formulation
- one_loop_corrections: Complete effective action implementation
- lindblad_dynamics: Quantum decoherence and ℏ_eff computation
- hamiltonian_dynamics: Hamiltonian structure for quantum MCTS
- rg_flow: Renormalization group flow equations
- unified_quantum_mcts: Complete integrated quantum MCTS system

Quick start:
    >>> from mcts.quantum.research import create_unified_quantum_mcts
    >>> quantum_mcts = create_unified_quantum_mcts()
    >>> results = quantum_mcts.quantum_enhanced_selection(...)

Performance target: < 2x overhead compared to classical MCTS
Theoretical foundation: Complete closed-form effective action Γ_eff = S_cl + ΔΓ^(1) + ΔΓ_RG
"""

from .path_integral_engine import (
    PathIntegralEngine,
    PathIntegralConfig,
    InformationTimeMode,
    create_path_integral_engine
)

from .one_loop_corrections import (
    OneLoopCorrections,
    OneLoopConfig
)

from .quantum_corrections import (
    QuantumCorrections, 
    QuantumCorrectionConfig,
    create_quantum_corrections
)

from .path_integral_mc import (
    PathIntegralMC,
    PathIntegralMCConfig,
    DiscretizationMethod,
    create_path_integral_mc
)

from .wave_quantum_mcts import (
    WaveQuantumMCTS,
    WaveQuantumConfig,
    create_wave_quantum_mcts
)

from .lindblad_dynamics import (
    LindbladDynamics,
    LindbladConfig,
    EffectivePlanckConstant,
    TimeEvolutionMethod,
    create_lindblad_dynamics
)

from .lindblad_integration import (
    IntegratedQuantumEngine,
    IntegratedQuantumConfig,
    QuantumStateCache,
    create_integrated_quantum_engine
)

from .hamiltonian_dynamics import (
    HamiltonianStructure,
    HamiltonianConfig,
    create_hamiltonian_structure
)

from .rg_flow import (
    RGFlowEquations,
    RGFlowConfig,
    create_rg_flow_equations
)

from .uv_cutoff import (
    UVCutoffMechanism,
    UVCutoffConfig,
    CutoffMethod,
    create_uv_cutoff_mechanism
)

from .unified_quantum_mcts import (
    UnifiedQuantumMCTS,
    UnifiedQuantumConfig,
    QuantumMCTSMode,
    create_unified_quantum_mcts
)

# Visualization and analysis tools
from .generate_all_plots import main as generate_all_plots

import logging

logger = logging.getLogger(__name__)

# Package version
__version__ = "1.0.0"

# Package-level exports
__all__ = [
    # Main unified system
    'UnifiedQuantumMCTS',
    'UnifiedQuantumConfig', 
    'QuantumMCTSMode',
    'create_unified_quantum_mcts',
    
    # Core quantum components
    'PathIntegralEngine',
    'OneLoopCorrections',
    'LindbladDynamics',
    'IntegratedQuantumEngine',
    'HamiltonianStructure',
    'RGFlowEquations',
    'UVCutoffMechanism',
    
    # Legacy components
    'QuantumCorrections', 
    'PathIntegralMC',
    'WaveQuantumMCTS',
    
    # Configuration classes
    'PathIntegralConfig',
    'OneLoopConfig',
    'LindbladConfig',
    'IntegratedQuantumConfig',
    'HamiltonianConfig', 
    'RGFlowConfig',
    'UVCutoffConfig',
    'QuantumCorrectionConfig',
    'PathIntegralMCConfig',
    'WaveQuantumConfig',
    
    # Enums
    'InformationTimeMode',
    'DiscretizationMethod',
    'TimeEvolutionMethod',
    'CutoffMethod',
    'QuantumMCTSMode',
    
    # Factory functions
    'create_path_integral_engine',
    'create_integrated_quantum_engine',
    'create_hamiltonian_structure',
    'create_rg_flow_equations', 
    'create_uv_cutoff_mechanism',
    'create_quantum_corrections',
    'create_path_integral_mc',
    'create_wave_quantum_mcts',
    'create_lindblad_dynamics',
    
    # High-level interfaces
    'create_unified_quantum_mcts',
    
    # Visualization tools
    'generate_all_plots',
    
    # Constants
    '__version__'
]

# Initialize logging for the package
logging.getLogger(__name__).addHandler(logging.NullHandler())