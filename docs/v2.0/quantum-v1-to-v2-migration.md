# Quantum MCTS v1.0 to v2.0 Migration Guide

## Overview

Quantum MCTS v2.0 represents a major theoretical and implementation upgrade based on discrete information time and improved physics foundations. This guide helps you migrate from v1.0 to v2.0.

## Key Differences

### v1.0 (Deprecated)
- Continuous time formulation
- Exponential decoherence
- Manual parameter tuning
- Fixed quantum strength
- 2-3x overhead

### v2.0 (Recommended)
- Discrete information time: τ(N) = log(N+2)
- Power-law decoherence: ρᵢⱼ(N) ~ N^(-Γ₀)
- Auto-computed parameters from theory
- Phase-aware adaptation
- 1.3-1.8x overhead with neural networks

## Migration Options

### Option 1: Automatic Migration (Recommended)

The wrapper automatically handles version selection:

```python
from mcts.quantum import create_quantum_mcts

# Your existing v1 code continues to work
quantum_mcts = create_quantum_mcts(
    enable_quantum=True,
    # v1 parameters still accepted
)
# Shows deprecation warning and uses v1

# Add v2 parameters to upgrade
quantum_mcts = create_quantum_mcts(
    enable_quantum=True,
    branching_factor=20,      # New: triggers v2
    avg_game_length=100,      # New: for parameter computation
)
# Automatically uses v2
```

### Option 2: Explicit Version Control

Force specific version:

```python
# Explicitly use v1 (not recommended)
quantum_mcts = create_quantum_mcts(
    version='v1',
    enable_quantum=True
)

# Explicitly use v2 (recommended)
quantum_mcts = create_quantum_mcts(
    version='v2',
    branching_factor=20,
    avg_game_length=100,
    enable_quantum=True
)
```

### Option 3: Direct v2 Usage

Skip wrapper and use v2 directly:

```python
from mcts.quantum.quantum_features_v2 import create_quantum_mcts_v2

quantum_mcts = create_quantum_mcts_v2(
    branching_factor=20,
    avg_game_length=100,
    use_neural_network=True
)
```

## Parameter Mapping

| v1.0 Parameter | v2.0 Equivalent | Notes |
|----------------|-----------------|-------|
| `exploration_constant` | `c_puct` | Auto-computed as √(2 log b) |
| `hbar_eff` | Auto-computed | c_puct(N+2)/(√(N+1)log(N+2)) |
| `temperature` | `initial_temperature` | Now supports annealing |
| `decoherence_rate` | `power_law_exponent` | Different decay model |
| N/A | `branching_factor` | Required for auto-computation |
| N/A | `avg_game_length` | Used for optimization |
| N/A | `enable_phase_adaptation` | New feature |

## API Changes

### Selection Method

v1.0:
```python
ucb_scores = quantum_mcts.apply_quantum_to_selection(
    q_values, visit_counts, priors,
    parent_visits=parent_visit_count
)
```

v2.0:
```python
ucb_scores = quantum_mcts.apply_quantum_to_selection(
    q_values, visit_counts, priors,
    c_puct=1.414,                    # Optional
    total_simulations=1000,          # New: enables phase detection
    parent_visit=parent_visit_count, # Renamed
    is_root=False                    # New: root node flag
)
```

### New Features in v2.0

1. **Phase Information**:
```python
phase_info = quantum_mcts.get_phase_info()
print(f"Current phase: {phase_info['current_phase']}")
print(f"Phase transitions: {phase_info['phase_transitions']}")
```

2. **Convergence Detection**:
```python
if quantum_mcts.check_convergence(tree):
    print("Envariance reached - search converged")
    # Can stop early
```

3. **Version Comparison**:
```python
from mcts.quantum import compare_versions

results = compare_versions(q_values, visit_counts, priors)
print(f"v1 vs v2 correlation: {results['correlation']:.4f}")
print(f"v2 speedup: {results['speedup']:.2f}x")
```

## Integration with MCTS

Update your MCTSConfig:

```python
from mcts.core.mcts import MCTSConfig

config = MCTSConfig(
    # Standard MCTS parameters
    num_simulations=10000,
    c_puct=1.414,
    
    # Enable quantum v2
    enable_quantum=True,
    quantum_version='v2',  # Force v2
    
    # v2-specific parameters
    quantum_branching_factor=None,    # Auto-detect from game
    quantum_avg_game_length=None,     # Auto-detect from game
    enable_phase_adaptation=True,     # Recommended
    envariance_threshold=1e-3,        # Convergence threshold
    envariance_check_interval=1000,   # Check frequency
)
```

## Performance Considerations

1. **Overhead**: v2.0 has lower overhead (1.3-1.8x vs 2-3x)
2. **Memory**: Similar memory usage to v1.0
3. **Convergence**: Can terminate early with envariance
4. **Batch Size**: Larger batches benefit more from v2.0

## Troubleshooting

### Deprecation Warnings
To suppress v1 deprecation warnings:
```python
config = UnifiedQuantumConfig(
    suppress_deprecation_warnings=True,
    # ... other parameters
)
```

### Performance Issues
If v2 is slower than expected:
1. Ensure `branching_factor` is set correctly
2. Use larger batch sizes (≥64)
3. Enable mixed precision
4. Check phase adaptation is working

### Validation
To validate your migration:
```python
# Compare outputs
comparison = compare_versions(
    your_q_values, 
    your_visit_counts, 
    your_priors
)

# Should see high correlation (>0.8)
assert comparison['correlation'] > 0.8
# Should see speedup
assert comparison['speedup'] > 0.5
```

## Recommended Migration Path

1. **Test Compatibility**: Run `compare_versions()` on your data
2. **Add v2 Parameters**: Set `branching_factor` and `avg_game_length`
3. **Enable Phase Adaptation**: Better performance across search
4. **Monitor Convergence**: Use envariance for early stopping
5. **Benchmark Performance**: Ensure overhead is acceptable

## Support

- Report issues: https://github.com/your-repo/issues
- Documentation: `docs/v2.0/`
- Examples: `python/examples/quantum_v2_example.py`