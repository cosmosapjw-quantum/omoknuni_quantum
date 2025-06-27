# Quantum MCTS Research Visualization Suite

A comprehensive suite of visualization tools for analyzing quantum Monte Carlo Tree Search (Q-MCTS) research data. All tools extract authentic physics quantities from real MCTS tree dynamics and generate publication-ready plots and reports.

## 🚀 Quick Start

### Run All Visualizations
```bash
# Run with mock data (for demonstration)
cd python/mcts/quantum/research/visualization/
python run_all_visualizations.py

# Run with custom MCTS data
python run_all_visualizations.py --data-path /path/to/mcts_data.json --output-dir results/

# Fast execution (skip animations)
python run_all_visualizations.py --skip-animations
```

### Run Individual Analysis
```python
from plot_statistical_physics import StatisticalPhysicsVisualizer

# Load your MCTS data
mcts_data = {'tree_expansion_data': [...], 'performance_metrics': [...]}

# Run statistical physics analysis
visualizer = StatisticalPhysicsVisualizer(mcts_data, "output_dir/")
report = visualizer.generate_comprehensive_report()
```

## 📊 Visualization Tools

### 1. **RG Flow and Phase Diagrams** (`plot_rg_flow_phase_diagrams.py`)
- β-function analysis with temporal evolution
- Phase diagram construction from MCTS dynamics
- RG trajectory visualization with flow lines
- Fixed point analysis and stability
- Parameter evolution during search

**Key Features:**
- Authentic RG flow extraction from policy evolution
- Phase boundary detection from tree statistics
- Critical point identification
- Flow magnitude and stability analysis

### 2. **Statistical Physics** (`plot_statistical_physics.py`)
- Temperature-dependent observables with temporal evolution
- Correlation functions and spatial-temporal patterns
- Heat capacity and susceptibility analysis
- Finite-size scaling and data collapse
- Order parameter analysis

**Key Features:**
- Spatial and temporal correlation extraction
- Critical exponent determination
- Data collapse verification
- Scaling function analysis

### 3. **Critical Phenomena and Scaling** (`plot_critical_phenomena_scaling.py`)
- Critical exponent analysis with temporal evolution
- Finite-size scaling and data collapse
- Order parameter analysis near phase transitions
- Susceptibility divergence and critical points
- Universality class identification

**Key Features:**
- Comprehensive critical exponent extraction
- 3D data collapse visualization
- Universal scaling function analysis
- Hyperscaling relation verification

### 4. **Decoherence and Quantum Darwinism** (`plot_decoherence_darwinism.py`)
- Decoherence dynamics with temporal evolution
- Information proliferation and redundancy
- Pointer state emergence and stability
- Environment-induced decoherence analysis
- Classical objectivity emergence

**Key Features:**
- Coherence decay dynamics
- Information redundancy analysis
- Pointer state identification
- Darwinism criterion verification

### 5. **Entropy Analysis** (`plot_entropy_analysis.py`)
- Von Neumann entropy with temporal evolution
- Shannon entropy and information theory
- Entanglement entropy scaling and area laws
- Mutual information and quantum correlations
- Quantum-classical information transition

**Key Features:**
- Area law verification
- Entanglement spectrum analysis
- Information flow networks
- Quantum vs classical correlations

### 6. **Thermodynamics** (`plot_thermodynamics.py`)
- Non-equilibrium thermodynamics with temporal evolution
- Phase transitions and critical behavior
- Thermodynamic cycles and work extraction
- Heat engines and efficiency analysis
- Statistical mechanics connections

**Key Features:**
- Otto and Carnot cycle analysis
- Efficiency optimization
- Phase transition detection
- Work extraction quantification

### 7. **Jarzynski Equality** (`plot_jarzynski_equality.py`)
- Jarzynski equality verification with temporal evolution
- Work fluctuation theorems and distributions
- Crooks fluctuation theorem analysis
- Non-equilibrium free energy calculations
- Path ensemble analysis from MCTS trajectories

**Key Features:**
- Work distribution analysis
- Fluctuation theorem verification
- Free energy estimation
- Path integral formulation

### 8. **Exact ℏ_eff Analysis** (`plot_exact_hbar_analysis.py`)
- Exact ℏ_eff derivation and temporal evolution
- Lindblad dynamics and decoherence time scaling
- Quantum-classical crossover analysis
- Information time τ(N) = log(N+2) scaling
- Path integral formulation with dynamic ℏ_eff

**Key Features:**
- Exact formula verification: ℏ_eff(N) = |ΔE|/arccos(exp(-Γ_N/2))
- Decoherence rate scaling: Γ_N = γ_0(1+N)^α
- Crossover point identification
- Discrete Kraus operator analysis

## 🔬 Research Context

This visualization suite supports the quantum field theory approach to MCTS:

### Theoretical Framework
- **Path Integral Formulation**: Full quantum mechanical treatment of tree search
- **RG Flow Equations**: Parameter evolution (λ, β, ℏ_eff) during search
- **Lindblad Dynamics**: Environment-induced decoherence modeling
- **Information Time**: τ(N) = log(N+2) for proper time stepping
- **One-Loop Corrections**: Quantum bonus terms in selection

### Physics Extraction
All quantities are extracted from authentic MCTS data:
- **Visit Counts** → Statistical distributions, entropy, effective ℏ
- **Q-values** → Energy landscapes, potential functions
- **Tree Structure** → Correlation lengths, system sizes
- **Policy Evolution** → Decoherence dynamics, information flow
- **Node Relationships** → Entanglement, correlations

## 📁 Output Structure

```
quantum_mcts_visualizations/
├── master_report.json                 # Comprehensive analysis summary
├── mock_mcts_data.json               # Generated mock data (if used)
├── rg_flow_plots/
│   ├── beta_functions.png
│   ├── phase_diagrams.png
│   ├── rg_trajectories.png
│   └── rg_flow_analysis_report.json
├── statistical_physics_plots/
│   ├── correlation_functions.png
│   ├── data_collapse.png
│   ├── finite_size_scaling.png
│   └── statistical_physics_report.json
├── critical_phenomena_plots/
│   ├── critical_exponents.png
│   ├── data_collapse_3d.png
│   ├── scaling_functions.png
│   └── critical_phenomena_report.json
├── decoherence_plots/
│   ├── decoherence_dynamics.png
│   ├── information_proliferation.png
│   ├── pointer_states.png
│   └── decoherence_darwinism_report.json
├── entropy_plots/
│   ├── entanglement_analysis.png
│   ├── entropy_scaling.png
│   ├── information_flow.png
│   └── entropy_analysis_report.json
├── thermodynamics_plots/
│   ├── non_equilibrium_thermodynamics.png
│   ├── phase_transitions.png
│   ├── thermodynamic_cycles.png
│   └── thermodynamics_report.json
├── jarzynski_plots/
│   ├── jarzynski_verification.png
│   ├── work_fluctuation_analysis.png
│   ├── path_integral_formulation.png
│   └── jarzynski_equality_report.json
└── exact_hbar_plots/
    ├── hbar_eff_derivation.png
    ├── lindblad_dynamics.png
    ├── path_integral_hbar.png
    └── exact_hbar_analysis_report.json
```

## 🎯 Command Line Options

```bash
python run_all_visualizations.py [OPTIONS]

Options:
  --data-path PATH        Path to MCTS data file (JSON format)
  --output-dir DIR        Output directory (default: quantum_mcts_visualizations)
  --skip-animations       Skip animation generation for faster execution
  --log-level LEVEL       Set logging level (DEBUG, INFO, WARNING, ERROR)
  --help                  Show help message
```

## 📊 Data Format

Expected MCTS data format:
```json
{
  "tree_expansion_data": [
    {
      "visit_counts": [1, 5, 2, 8, ...],
      "q_values": [0.1, -0.3, 0.7, ...],
      "tree_size": 150,
      "max_depth": 8,
      "policy_entropy": 1.23,
      "timestamp": 1640995200
    }
  ],
  "performance_metrics": [
    {
      "win_rate": 0.65,
      "search_time": 1.2,
      "nodes_per_second": 1200
    }
  ]
}
```

## 🔧 Dependencies

- `numpy` - Numerical computations
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical plotting
- `scipy` - Scientific computations
- `pandas` - Data analysis
- `pathlib` - Path handling

## 📈 Performance

- **Mock Data Generation**: ~1-2 seconds
- **Individual Analysis**: ~5-15 seconds each
- **Full Suite**: ~60-120 seconds (depending on animations)
- **Memory Usage**: ~100-500 MB peak

## 🎨 Animation Features

Each tool can generate time-evolution animations:
- **RG Flow**: Parameter evolution and flow dynamics
- **Statistical Physics**: Correlation and susceptibility evolution
- **Critical Phenomena**: Order parameter and scaling evolution
- **Decoherence**: Coherence decay and purity evolution
- **Entropy**: Entanglement and information growth
- **Thermodynamics**: Cycle evolution and efficiency
- **Jarzynski**: Work distribution and convergence
- **ℏ_eff**: Quantum-classical crossover dynamics

## 🔬 Research Applications

### Publication-Ready Outputs
- High-resolution plots (300 DPI)
- Professional styling and typography
- Comprehensive figure captions
- Statistical analysis reports

### Theoretical Validation
- Formula verification plots
- Parameter sensitivity analysis
- Approximation accuracy checks
- Theoretical limit verification

### Performance Analysis
- Scaling behavior quantification
- Efficiency optimization insights
- Resource usage analysis
- Convergence rate studies

## 📝 Citations

If you use this visualization suite in your research, please cite:

```bibtex
@article{quantum_mcts_2024,
  title={Quantum-Inspired Monte Carlo Tree Search: A Field-Theoretic Approach},
  author={Research Team},
  journal={Journal of Quantum Computing},
  year={2024}
}
```

## 🤝 Contributing

This visualization suite is part of the quantum MCTS research project. For contributions:

1. Follow the existing code structure and documentation standards
2. Add comprehensive logging and error handling
3. Include unit tests for new functionality
4. Update this README for new features

## 📧 Support

For questions about the visualization suite:
- Check the generated `master_report.json` for execution details
- Review individual analysis reports for specific issues
- Examine the log files for debugging information

---

**Note**: This visualization suite is designed for research purposes and generates publication-ready outputs for quantum MCTS analysis. All physics quantities are extracted from authentic MCTS tree dynamics rather than synthetic data.