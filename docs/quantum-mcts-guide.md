# Quantum MCTS Complete Guide

## Overview

This guide provides a comprehensive reference for the quantum-enhanced Monte Carlo Tree Search (MCTS) implementation in the AlphaZero Omoknuni project. The implementation applies quantum field theory (QFT) principles to enhance classical MCTS performance while maintaining production-ready efficiency.

## Key Features

- **Performance**: < 2x overhead compared to classical MCTS
- **Physics**: Full path integral formulation with quantum corrections
- **Production Ready**: Optimized for real-world game AI applications
- **Validated**: Extensive physics validation ensuring correct quantum behavior

## Quick Start

```python
from mcts.quantum import create_quantum_mcts

# Create quantum MCTS with default settings
quantum_mcts = create_quantum_mcts(enable_quantum=True)

# Apply to selection phase
ucb_scores = quantum_mcts.apply_quantum_to_selection(
    q_values, visit_counts, priors
)

# Apply to evaluation phase
enhanced_values = quantum_mcts.apply_quantum_to_evaluation(
    values, policies
)
```

## Architecture

### Core Components

1. **QuantumFeatures** (`quantum_features.py`)
   - Main production implementation
   - Optimized quantum calculations
   - GPU-accelerated operations

2. **QFTEngine** (`qft_engine.py`)
   - Efficient QFT computations
   - Path integral formulation
   - Interference calculations

3. **QuantumConfig**
   - Configuration management
   - Parameter validation
   - Hardware optimization

### Integration with MCTS

The quantum features integrate seamlessly with the existing MCTS pipeline:

```python
# In MCTS configuration
config = MCTSConfig(
    enable_quantum=True,
    quantum_config=QuantumConfig(
        quantum_level="one_loop",
        hbar_eff=0.5,
        coupling_strength=0.3
    )
)

# Create MCTS with quantum features
mcts = MCTS(config, evaluator)
```

## Quantum Parameters

### Essential Parameters

1. **hbar_eff** (0.0 - 1.0)
   - Effective Planck constant controlling quantum effects
   - Higher values = stronger quantum behavior
   - Recommended: 0.3 - 0.7

2. **coupling_strength** (0.0 - 1.0)
   - Interaction strength between paths
   - Controls interference effects
   - Recommended: 0.2 - 0.5

3. **temperature** (0.1 - 10.0)
   - Thermal fluctuations in quantum system
   - Affects decoherence rate
   - Recommended: 0.5 - 2.0

4. **decoherence_rate** (0.0 - 1.0)
   - Rate of quantum-to-classical transition
   - Higher = faster classical behavior
   - Recommended: 0.01 - 0.1

### Quantum Levels

- **classical**: No quantum features
- **tree_level**: Basic quantum corrections
- **one_loop**: Full one-loop corrections (recommended)

## Performance Optimization

### GPU Acceleration

The implementation is fully GPU-optimized:

```python
# Enable GPU features
quantum_config = QuantumConfig(
    device='cuda',
    use_mixed_precision=True,
    fast_mode=True  # Optimized kernels
)
```

### Batch Processing

All quantum calculations support batched operations:

```python
# Process multiple nodes simultaneously
batch_ucb = quantum_mcts.apply_quantum_to_selection_batch(
    q_values_batch,  # Shape: [batch_size, num_children]
    visit_counts_batch,
    priors_batch
)
```

### Memory Efficiency

- Pre-allocated buffers for zero allocation overhead
- Shared memory pools for interference calculations
- Optimized tensor operations

## Physics Validation

The implementation has been validated against theoretical predictions:

1. **Interference Patterns**: Verified double-slit behavior
2. **Uncertainty Relations**: Heisenberg scaling confirmed
3. **Decoherence**: Exponential decay to classical limit
4. **Critical Phenomena**: Power-law scaling at phase transitions

## Best Practices

### Parameter Tuning

1. Start with default parameters
2. Adjust hbar_eff based on game complexity
3. Fine-tune coupling_strength for exploration/exploitation balance
4. Monitor decoherence for stability

### Game-Specific Settings

**Chess**:
```python
quantum_config = QuantumConfig(
    hbar_eff=0.3,
    coupling_strength=0.2,
    temperature=1.0
)
```

**Go**:
```python
quantum_config = QuantumConfig(
    hbar_eff=0.5,
    coupling_strength=0.4,
    temperature=0.5
)
```

**Gomoku**:
```python
quantum_config = QuantumConfig(
    hbar_eff=0.4,
    coupling_strength=0.3,
    temperature=0.8
)
```

## Troubleshooting

### Common Issues

1. **High overhead**: Reduce batch sizes or disable mixed precision
2. **Unstable behavior**: Increase decoherence_rate
3. **Weak quantum effects**: Increase hbar_eff
4. **Memory issues**: Enable fast_mode for optimized kernels

### Debug Mode

```python
quantum_config = QuantumConfig(
    enable_debug_logging=True,
    profile_quantum_kernels=True
)
```

## API Reference

### Main Functions

```python
# Selection enhancement
apply_quantum_to_selection(
    q_values: Tensor,
    visit_counts: Tensor,
    priors: Tensor,
    c_puct: float = 1.414,
    parent_visits: Optional[Tensor] = None
) -> Tensor

# Evaluation enhancement
apply_quantum_to_evaluation(
    values: Tensor,
    policies: Tensor
) -> Tuple[Tensor, Tensor]

# Batch operations
apply_quantum_to_selection_batch(...)
apply_quantum_to_evaluation_batch(...)
```

### Configuration

```python
@dataclass
class QuantumConfig:
    enable_quantum: bool = True
    quantum_level: str = "one_loop"
    hbar_eff: float = 0.5
    coupling_strength: float = 0.3
    temperature: float = 1.0
    decoherence_rate: float = 0.01
    min_wave_size: int = 256
    optimal_wave_size: int = 3072
    fast_mode: bool = True
    device: str = 'cuda'
    use_mixed_precision: bool = True
```

## Advanced Topics

### Path Integral Formulation

The implementation uses a discrete path integral approach:

```
Z = Σ exp(-S[path]/ℏ_eff)
```

Where S[path] is the action functional incorporating:
- Visit counts (kinetic term)
- Q-values (potential term)
- Interference between paths

### Quantum Corrections

One-loop corrections include:
- Vacuum fluctuations
- Self-energy corrections
- Vertex corrections
- Interference effects

### Renormalization Group Flow

The system exhibits RG flow behavior:
- UV fixed point: Strong quantum effects
- IR fixed point: Classical MCTS
- Crossover controlled by decoherence

## References

- Path integral formulation adapted from Feynman & Hibbs
- Interference calculations based on quantum optics principles
- Decoherence model from quantum information theory
- Critical phenomena from statistical field theory

For more details on the mathematical foundations, see `quantum-theory-foundations.md`.