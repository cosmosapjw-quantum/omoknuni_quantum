# Statistical Mechanics Analysis Guide for MCTS

## Overview
This guide documents the statistical mechanics analysis framework for Monte Carlo Tree Search, providing theoretical foundations and practical implementation details for observing emergent thermodynamic behaviors in tree search dynamics.

## Table of Contents
1. [Theoretical Framework](#theoretical-framework)
2. [Implementation Components](#implementation-components)
3. [Analysis Modules](#analysis-modules)
4. [Visualization Tools](#visualization-tools)
5. [Usage Examples](#usage-examples)

---

## Theoretical Framework

### MCTS as a Statistical System
MCTS exhibits properties analogous to statistical mechanical systems:
- **Energy**: Negative expected value (-Q)
- **Temperature**: Exploration parameter (1/√N)
- **Entropy**: Policy distribution uncertainty
- **Phase Transitions**: Critical decision points

### Key Correspondences
```
MCTS Parameter          →  Statistical Mechanics
─────────────────────────────────────────────
Visit count (N)         →  Inverse temperature (β)
Q-value                 →  Energy levels
Policy π(a)             →  Boltzmann distribution
UCB exploration         →  Thermal fluctuations
Tree depth              →  System size
```

---

## Implementation Components

### 1. Thermodynamics Analyzer (`thermodynamics.py`)
```python
class ThermodynamicsAnalyzer:
    """Analyzes thermodynamic properties of MCTS dynamics"""
    
    def compute_temperature(self, visits: torch.Tensor) -> float:
        """Temperature from visit distribution: T = 1/√N"""
        return 1.0 / np.sqrt(visits.sum().item())
    
    def compute_energy(self, q_values: torch.Tensor, 
                      visits: torch.Tensor) -> float:
        """Energy as expectation of Q-values"""
        probs = visits / visits.sum()
        return -torch.sum(probs * q_values).item()
    
    def compute_entropy(self, policy: torch.Tensor) -> float:
        """Shannon entropy of policy distribution"""
        return -torch.sum(policy * torch.log(policy + 1e-10)).item()
```

### 2. Critical Phenomena Detector (`critical.py`)
```python
class CriticalPhenomenaDetector:
    """Detects phase transitions and critical points"""
    
    def find_critical_points(self, trajectory: List[TreeState]) -> List[int]:
        """Identify positions with maximum susceptibility"""
        
    def compute_order_parameter(self, state: TreeState) -> float:
        """Order parameter from action concentration"""
        
    def detect_phase_transition(self, trajectory: List[TreeState]) -> Dict:
        """Analyze phase transition characteristics"""
```

### 3. Fluctuation-Dissipation Theorem (`fdt_plotter.py`)
Validates the fundamental relation between fluctuations and response:
- **Fluctuations**: Variance in Q-values
- **Dissipation**: Policy entropy
- **FDT Relation**: χ = β × σ²

---

## Analysis Modules

### ThermodynamicsPlotter (`thermodynamics_plotter.py`)

#### Features
1. **Energy Evolution**
   - Tracks system energy over game progression
   - Shows convergence to equilibrium
   - Identifies energy barriers

2. **Temperature Dynamics**
   - Plots T = 1/√N relationship
   - Shows cooling during search
   - Compares with theoretical predictions

3. **Entropy Analysis**
   - Policy entropy evolution
   - Maximum entropy principle validation
   - Information gain tracking

#### Example Usage
```python
from mcts.quantum.analysis.thermodynamics_plotter import ThermodynamicsPlotter

plotter = ThermodynamicsPlotter()
fig = plotter.plot_energy_evolution(dynamics_data)
fig = plotter.plot_temperature_evolution(dynamics_data, show_theory=True)
fig = plotter.create_summary_plot(dynamics_data)
```

### CriticalPhenomenaPlotter (`critical_plotter.py`)

#### Features
1. **Order Parameter Evolution**
   - Tracks symmetry breaking
   - Shows phase transitions
   - Identifies critical regions

2. **Susceptibility Analysis**
   - Peaks at critical points
   - Divergence behavior
   - Finite-size scaling

3. **Correlation Functions**
   - Spatial correlations in tree
   - Temporal correlations in trajectory
   - Critical exponents

#### Visualization Methods
```python
plotter = CriticalPhenomenaPlotter()
fig = plotter.plot_order_parameter_evolution(data)
fig = plotter.plot_susceptibility(data, mark_critical=True)
fig = plotter.plot_correlation_function(data)
fig = plotter.create_critical_summary(data)
```

### FDTPlotter (`fdt_plotter.py`)

#### Features
1. **FDT Validation**
   - Scatter plot of fluctuation vs response
   - Linear fit with theoretical slope
   - Deviation analysis

2. **Time-Resolved FDT**
   - FDT ratio evolution
   - Non-equilibrium detection
   - Aging effects

3. **Comprehensive Summary**
   - Multiple FDT aspects
   - Statistical validation
   - Theoretical comparison

---

## Visualization Tools

### Plot Types and Interpretations

#### 1. Thermodynamic Summary
```
┌─────────────────────────────────┐
│  Energy Evolution               │
│  ┌─┐                           │
│  │ └─┐   Equilibration         │
│  │   └─────────                │
│  Time →                        │
├─────────────────────────────────┤
│  Temperature vs Theory          │
│  Theory: T = 1/√N              │
│  • • • • (data points)         │
│  ───── (theory line)           │
└─────────────────────────────────┘
```

#### 2. Critical Phenomena
```
┌─────────────────────────────────┐
│  Susceptibility                 │
│      ▲                         │
│      │ Critical point          │
│  ────┼────                     │
│      │                         │
│  Position →                    │
└─────────────────────────────────┘
```

#### 3. FDT Validation
```
┌─────────────────────────────────┐
│  Response vs Fluctuation        │
│    │ •••                       │
│  R │••  Slope = 1/T            │
│    │•                          │
│    └────── σ²                  │
└─────────────────────────────────┘
```

---

## Usage Examples

### Basic Statistical Analysis
```python
from mcts.quantum.analysis.auto_generator import GeneratorConfig
from mcts.quantum.analysis.thermodynamics_plotter import ThermodynamicsPlotter

# Configure for statistical mechanics only
config = GeneratorConfig(
    target_games=100,
    sims_per_game=5000,
    analysis_types=['thermodynamics', 'critical', 'fdt'],  # Stat mech only
    output_dir='./statistical_mechanics_analysis'
)

# Run analysis
generator = AutoDataGenerator(config)
result = generator.run()
```

### Advanced Analysis
```python
# Custom critical phenomena detection
from mcts.quantum.phenomena.critical import CriticalPhenomenaDetector

detector = CriticalPhenomenaDetector(
    order_parameter_threshold=0.8,
    susceptibility_window=10
)

# Analyze specific game
critical_points = detector.find_critical_points(game_trajectory)
phase_info = detector.detect_phase_transition(game_trajectory)
```

### Comparative Studies
```python
# Compare different games/parameters
configs = [
    GeneratorConfig(sims_per_game=1000, game_type='gomoku'),
    GeneratorConfig(sims_per_game=5000, game_type='gomoku'),
    GeneratorConfig(sims_per_game=20000, game_type='gomoku'),
]

results = []
for config in configs:
    generator = AutoDataGenerator(config)
    results.append(generator.run())

# Analyze scaling behavior
plot_temperature_scaling(results)
plot_critical_exponents(results)
```

---

## Output Structure

### Statistical Mechanics Analysis Output
```
statistical_mechanics_analysis/
├── data/
│   └── dynamics/
│       ├── game_0000_dynamics.npz
│       └── ...
├── plots/
│   ├── thermodynamics/
│   │   └── game_0000/
│   │       ├── energy_evolution.png
│   │       ├── temperature_evolution.png
│   │       ├── entropy_evolution.png
│   │       ├── heat_capacity.png
│   │       └── thermodynamics_summary.png
│   ├── critical/
│   │   └── game_0000/
│   │       ├── order_parameter.png
│   │       ├── susceptibility.png
│   │       ├── correlation_function.png
│   │       └── critical_summary.png
│   └── fdt/
│       └── game_0000/
│           ├── fdt_validation.png
│           ├── time_resolved_fdt.png
│           └── fdt_summary.png
└── analysis_report.json
```

---

## Theoretical Background

### 1. Equilibrium Statistical Mechanics
- **Canonical Ensemble**: Fixed temperature (exploration parameter)
- **Microcanonical Ensemble**: Fixed energy (value constraint)
- **Partition Function**: Z = Σ exp(-βE)

### 2. Non-Equilibrium Phenomena
- **Relaxation Dynamics**: Approach to equilibrium
- **Aging Effects**: Time-dependent behavior
- **Driven Systems**: External forcing (game dynamics)

### 3. Critical Phenomena
- **Universality Classes**: Similar behavior across systems
- **Scaling Laws**: Power-law relationships
- **Renormalization**: Coarse-graining effects

---

## Best Practices

1. **Data Collection**
   - Ensure sufficient statistics (>1000 simulations/position)
   - Sample uniformly across game progression
   - Include both critical and non-critical positions

2. **Analysis Parameters**
   - Use appropriate time windows for averaging
   - Consider finite-size effects
   - Validate against theoretical predictions

3. **Interpretation**
   - Remember MCTS is a driven system
   - Account for non-equilibrium effects
   - Compare with equilibrium theory carefully

---

## Troubleshooting

### Common Issues

1. **Noisy Temperature Measurements**
   - Increase simulation count
   - Use smoothing/averaging
   - Check for outliers

2. **Missing Critical Points**
   - Adjust detection thresholds
   - Increase sampling density
   - Check susceptibility calculation

3. **FDT Violations**
   - Expected in non-equilibrium
   - Check time scales
   - Verify calculation methods

---

## References

1. **Statistical Mechanics of MCTS**
   - Thermodynamic interpretation of tree search
   - Temperature-visit count relationship
   - Equilibrium and non-equilibrium aspects

2. **Critical Phenomena in Games**
   - Phase transitions at decision points
   - Order parameters and symmetry breaking
   - Finite-size scaling analysis

3. **Fluctuation-Dissipation Relations**
   - FDT in search algorithms
   - Non-equilibrium corrections
   - Practical validation methods

---

*For quantum phenomena analysis, see QUANTUM_PHENOMENA_ANALYSIS_GUIDE.md*
*For implementation details, see source code documentation*

*Last Updated: July 2025*