# MCTS Physics Analysis User Guide

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Running Physics Analysis](#running-physics-analysis)
5. [Understanding Results](#understanding-results)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

## Overview

The MCTS Physics Analysis framework reveals quantum-like phenomena in Monte Carlo Tree Search by analyzing the algorithm through the lens of quantum mechanics and statistical physics. This guide will help you understand and use these analysis tools.

### What Does It Analyze?

- **Quantum Phenomena**: Superposition, entanglement, decoherence
- **Thermodynamics**: Temperature, energy, entropy production
- **Information Theory**: Mutual information, quantum Darwinism
- **Critical Phenomena**: Phase transitions, scaling laws
- **Holographic Properties**: Boundary/bulk correspondence

## Quick Start

### Installation

```bash
# Ensure you have the requirements
pip install torch numpy scipy matplotlib seaborn

# Activate environment
source ~/venv/bin/activate
```

### Basic Analysis

```python
# Run the simple physics analysis
python run_physics_analysis_simple.py

# This will:
# 1. Create test MCTS trajectories
# 2. Run all physics analyses
# 3. Save results to physics_analysis_results.json
```

### Visualize Results

```python
from mcts.quantum.visualization import QuantumVisualizer

# Load results
with open('physics_analysis_results.json', 'r') as f:
    results = json.load(f)

# Create visualizations
viz = QuantumVisualizer()
viz.plot_decoherence_evolution(
    results['analyses']['decoherence']['coherence_evolution'],
    results['analyses']['decoherence']['purity_evolution']
)
```

## Core Concepts

### 1. Wave-Based MCTS Formulation

MCTS naturally creates quantum-like superpositions:
- Each simulation creates a "wave" W(s,a)
- Multiple simulations form an ensemble
- Visit counts encode interference patterns

### 2. Quantum States from MCTS

The framework constructs quantum states using:
- **Simulation batches**: Natural ensemble diversity
- **Path overlap**: Determines quantum phases
- **Convergence**: Creates decoherence

### 3. Key Metrics

- **Von Neumann Entropy**: S = -Tr(ρ log ρ)
  - Measures quantum uncertainty
  - High early, low late (decoherence)

- **Purity**: P = Tr(ρ²)
  - 1 = pure state (classical)
  - <1 = mixed state (quantum)

- **Coherence**: Sum of off-diagonal elements
  - Measures quantum superposition
  - Decays as MCTS converges

## Running Physics Analysis

### 1. During MCTS Self-Play

```python
from mcts.quantum.physics_metrics import create_physics_callback

# Create callback
physics_callback = create_physics_callback(compute_expensive=False)

# Add to MCTS
mcts = YourMCTS()
mcts.add_callback(physics_callback)

# Run self-play - physics metrics computed automatically
mcts.self_play(n_games=100)

# Access results
physics_data = mcts.physics_metrics
```

### 2. Post-Hoc Analysis

```python
from mcts.quantum.wave_based_quantum_state import WaveBasedQuantumState
from mcts.quantum.phenomena.decoherence import DecoherenceAnalyzer

# Load MCTS snapshots
snapshots = load_mcts_snapshots('mcts_trajectory.pkl')

# Analyze decoherence
analyzer = DecoherenceAnalyzer()
results = analyzer.analyze_decoherence(snapshots)

print(f"Decoherence rate: {results['decoherence_rate']:.4f}")
print(f"Relaxation time: {results['relaxation_time']:.4f}")
```

### 3. Batch Analysis

```python
# Analyze multiple games
all_results = []

for game_file in game_files:
    snapshots = load_snapshots(game_file)
    
    # Run all analyses
    results = {
        'decoherence': decoherence_analyzer.analyze(snapshots),
        'thermodynamics': thermo_analyzer.measure(snapshots),
        'entanglement': entanglement_analyzer.compute(snapshots)
    }
    all_results.append(results)
```

## Understanding Results

### Decoherence Analysis

```json
{
  "coherence_evolution": [0.95, 0.87, 0.72, 0.45, 0.21],
  "purity_evolution": [0.65, 0.71, 0.78, 0.89, 0.96],
  "decoherence_rate": 0.134,
  "relaxation_time": 7.46
}
```

- **Coherence decreases**: Quantum → Classical transition
- **Purity increases**: Mixed → Pure state
- **Decoherence rate**: How fast quantum features disappear

### Thermodynamic Analysis

```json
{
  "temperature": 0.316,  // T = 1/√N
  "beta": 3.165,
  "energy": -0.743,
  "free_energy": -0.821
}
```

- **Temperature**: Exploration parameter (decreases with visits)
- **Energy**: Expected value/score
- **Free energy**: F = E - TS (decision quality)

### Information Thermodynamics

```json
{
  "entropy_production": 0.234,
  "mutual_information": 0.567,
  "generalized_second_law": true
}
```

- **Entropy production**: Information gained
- **Mutual information**: I(past:future)
- **2nd law**: Validates thermodynamic consistency

## Advanced Usage

### Custom Quantum State Construction

```python
# Use specific ensemble construction
wave_constructor = WaveBasedQuantumState()

# From simulation batch
sim_batch = [
    {'action_probs': [0.4, 0.3, 0.2, 0.1], 'paths': [...], 'values': [...]},
    {'action_probs': [0.35, 0.35, 0.2, 0.1], 'paths': [...], 'values': [...]},
]
quantum_state = wave_constructor.construct_from_simulation_batch(sim_batch)

# Analyze the quantum state
entropy = compute_von_neumann_entropy(quantum_state.density_matrix)
purity = compute_purity(quantum_state.density_matrix)
```

### Quantum-Enhanced MCTS

```python
from mcts.quantum.quantum_enhanced_mcts import QuantumMCTSConfig, create_quantum_mcts_wrapper

# Configure quantum enhancements
config = QuantumMCTSConfig(
    enable_adaptive_temperature=True,
    enable_entanglement_bonus=True,
    enable_decoherence_schedule=True
)

# Wrap existing MCTS
QuantumMCTS = create_quantum_mcts_wrapper(StandardMCTS)
quantum_mcts = QuantumMCTS(quantum_config=config)

# Use as normal - quantum features applied automatically
action = quantum_mcts.select_action(state)
```

### Custom Analysis Pipeline

```python
# Create custom analysis pipeline
class MyPhysicsAnalysis:
    def __init__(self):
        self.analyzers = {
            'quantum': WaveBasedQuantumState(),
            'decoherence': DecoherenceAnalyzer(),
            'thermo': ThermodynamicsAnalyzer(),
            'holographic': HolographicBoundsAnalyzer()
        }
    
    def analyze_game(self, game_data):
        results = {}
        
        # Extract simulation batches
        for t, snapshot in enumerate(game_data['snapshots']):
            if 'simulation_batch' in snapshot:
                # Construct quantum state
                q_state = self.analyzers['quantum'].construct_from_simulation_batch(
                    snapshot['simulation_batch']
                )
                
                # Compute metrics
                results[f't_{t}'] = {
                    'entropy': compute_von_neumann_entropy(q_state.density_matrix),
                    'purity': compute_purity(q_state.density_matrix),
                    'coherence': compute_coherence(q_state.density_matrix)
                }
        
        return results
```

## Troubleshooting

### Common Issues

1. **No simulation_batch data**
   - Solution: Falls back to single visit distribution
   - Better: Ensure MCTS saves simulation batch data

2. **Memory issues with large trees**
   - Use `compute_expensive_metrics=False`
   - Process in chunks
   - Use GPU acceleration

3. **Unexpected pure states**
   - Check ensemble diversity
   - Verify multiple simulations per batch
   - Ensure exploration noise is enabled

### Performance Tips

1. **GPU Acceleration**
   ```python
   analyzer = DecoherenceAnalyzer(device='cuda')
   ```

2. **Batch Processing**
   ```python
   # Process multiple snapshots at once
   quantum_calc = OptimizedQuantumCalculator()
   results = quantum_calc.batch_construct_quantum_states(visit_batch)
   ```

3. **Selective Analysis**
   ```python
   # Only compute essential metrics
   metrics_computer = PhysicsMetricsComputer(compute_expensive_metrics=False)
   ```

### Validation

Always validate results:

```python
# Check quantum state consistency
validation = quantum_defs.validate_quantum_consistency(quantum_state)
if not validation['valid']:
    print(f"Errors: {validation['errors']}")
    print(f"Warnings: {validation['warnings']}")
```

## Examples

### Example 1: Analyzing a Go Game

```python
# Load Go game MCTS data
game_data = load_go_game('professional_game.sgf')

# Extract MCTS snapshots every 10 moves
snapshots = []
for move in range(0, len(game_data), 10):
    snapshot = extract_mcts_snapshot(game_data, move)
    snapshots.append(snapshot)

# Run physics analysis
analyzer = SimplePhysicsAnalyzer()
results = analyzer.analyze_trajectory(snapshots)

# Plot evolution
viz = QuantumVisualizer()
fig = viz.plot_phase_space_trajectory(
    results['entropy_values'],
    results['purity_values']
)
```

### Example 2: Comparing Algorithms

```python
# Compare standard vs quantum-enhanced MCTS
standard_results = run_benchmark(StandardMCTS(), n_games=100)
quantum_results = run_benchmark(QuantumEnhancedMCTS(), n_games=100)

# Analyze physics properties
for results, name in [(standard_results, 'Standard'), (quantum_results, 'Quantum')]:
    metrics = extract_physics_metrics(results)
    print(f"\n{name} MCTS:")
    print(f"  Avg decoherence rate: {np.mean(metrics['decoherence_rates']):.4f}")
    print(f"  Avg final purity: {np.mean(metrics['final_purities']):.4f}")
    print(f"  Exploration efficiency: {metrics['exploration_score']:.4f}")
```

## Further Reading

- `quantum_mcts_foundation.md` - Theoretical foundations
- `WAVE_BASED_QUANTUM_STATE_SUMMARY.md` - Wave-based implementation
- `UNIFIED_QUANTUM_FRAMEWORK_DOCUMENTATION.md` - Complete framework docs

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review test files for usage examples
3. Examine the source code documentation

Remember: The physics analysis reveals the quantum nature inherent in MCTS - it's not adding quantum mechanics artificially, but rather exposing the quantum-like phenomena that naturally emerge from the algorithm's structure.