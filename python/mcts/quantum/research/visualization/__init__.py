"""
Quantum MCTS Research Visualization Package

This package provides comprehensive visualization tools for quantum Monte Carlo Tree Search research.
All visualization tools extract authentic physics quantities from real MCTS tree dynamics.

Modules:
    plot_rg_flow_phase_diagrams: RG flow and phase diagram analysis
    plot_statistical_physics: Statistical physics and correlation analysis  
    plot_critical_phenomena_scaling: Critical phenomena and scaling laws
    plot_decoherence_darwinism: Decoherence and quantum Darwinism
    plot_entropy_analysis: Entropy and information theory analysis
    plot_thermodynamics: Thermodynamics and phase transitions
    plot_jarzynski_equality: Jarzynski equality and fluctuation theorems
    plot_exact_hbar_analysis: Exact ℏ_eff derivation and analysis
    run_all_visualizations: Driver script for running all tools

Usage:
    # Run individual analysis
    from plot_statistical_physics import StatisticalPhysicsVisualizer
    visualizer = StatisticalPhysicsVisualizer(mcts_data)
    report = visualizer.generate_comprehensive_report()
    
    # Run all analyses
    python run_all_visualizations.py --output-dir results/
"""

__version__ = "4.0.0"
__author__ = "Quantum MCTS Research Team"

# Import main visualization classes
from .plot_rg_flow_phase_diagrams import RGFlowPhaseVisualizer
from .plot_statistical_physics import StatisticalPhysicsVisualizer
from .plot_critical_phenomena_scaling import CriticalPhenomenaVisualizer
from .plot_decoherence_darwinism import DecoherenceDarwinismVisualizer
from .plot_entropy_analysis import EntropyAnalysisVisualizer
from .plot_thermodynamics import ThermodynamicsVisualizer
from .plot_jarzynski_equality import JarzynskiEqualityVisualizer
from .plot_exact_hbar_analysis import ExactHbarAnalysisVisualizer

# Import authentic data extractor
from .authentic_mcts_physics_extractor import AuthenticMCTSPhysicsExtractor, create_authentic_physics_data

__all__ = [
    'RGFlowPhaseVisualizer',
    'StatisticalPhysicsVisualizer', 
    'CriticalPhenomenaVisualizer',
    'DecoherenceDarwinismVisualizer',
    'EntropyAnalysisVisualizer',
    'ThermodynamicsVisualizer',
    'JarzynskiEqualityVisualizer',
    'ExactHbarAnalysisVisualizer',
    'AuthenticMCTSPhysicsExtractor',
    'create_authentic_physics_data'
]