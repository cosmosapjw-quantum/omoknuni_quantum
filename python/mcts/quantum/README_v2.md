# Quantum MCTS v2.0

## Overview

Version 2.0 of the Quantum MCTS implementation introduces rigorous physics-based parameter selection, discrete information time, and full integration with neural network priors. This implementation is based on the complete mathematical framework detailed in `/docs/v2.0/quantum-mcts-new.md`.

## Key Features

### 1. Discrete Information Time
- **τ(N) = log(N+2)**: Natural time parameter for MCTS
- **Temperature annealing**: T(N) = T₀/log(N+2)
- **Dynamic ℏ_eff**: Automatically computed from game properties

### 2. Full PUCT Integration
- **Path integral action**: S[γ] = -Σ[log N(s,a) + λ log P(a|s)]
- **Neural network priors as external field**: No quantum corrections
- **Prior coupling strength**: λ = c_puct (optimal from theory)

### 3. Phase-Aware Strategies
- **Three phases**: Quantum (exploration), Critical (balanced), Classical (exploitation)
- **Automatic detection**: Based on simulation count and game properties
- **Adaptive parameters**: Different strategies per phase

### 4. Physics-Derived Parameters
- **Optimal c_puct**: √(2 log b) with RG corrections
- **Hash functions**: K = √(b·L) adjusted for neural networks
- **Phase kicks**: γ(N) = 1/√(N+1)

### 5. Enhanced Convergence
- **Power-law decoherence**: ρᵢⱼ(N) ~ N^(-Γ₀) instead of exponential
- **Envariance criterion**: Rigorous convergence test
- **Early stopping**: When policy becomes envariant

## Quick Start

```python
from mcts.quantum.quantum_features_v2 import create_quantum_mcts_v2

# Create with auto-computed parameters
quantum_mcts = create_quantum_mcts_v2(
    enable_quantum=True,
    branching_factor=20,      # Your game's branching factor
    avg_game_length=50,       # Average game length
    use_neural_network=True   # Using AlphaZero-style NN
)

# In your MCTS loop
for N in range(num_simulations):
    # Update simulation count (enables phase detection)
    quantum_mcts.update_simulation_count(N)
    
    # Apply quantum enhancement to selection
    ucb_scores = quantum_mcts.apply_quantum_to_selection(
        q_values, visit_counts, priors,
        simulation_count=N
    )
    
    # Check convergence periodically
    if N % 100 == 0:
        if quantum_mcts.check_envariance(tree):
            break  # Converged!
```

## Configuration Options

### Minimal Configuration (Recommended)
```python
config = {
    'branching_factor': 20,   # Required for auto-computation
    'avg_game_length': 50,    # Required for auto-computation
    'use_neural_network': True,
    'device': 'cuda'
}
quantum_mcts = create_quantum_mcts_v2(**config)
```

### Advanced Configuration
```python
from mcts.quantum.quantum_features_v2 import QuantumConfigV2

config = QuantumConfigV2(
    # Core settings
    enable_quantum=True,
    quantum_level='one_loop',     # 'tree_level' or 'one_loop'
    
    # Game parameters
    branching_factor=200,         # e.g., Go
    avg_game_length=150,
    
    # Neural network
    use_neural_prior=True,
    prior_coupling='auto',        # λ = c_puct
    
    # Phase detection
    enable_phase_adaptation=True,
    
    # Performance
    optimal_wave_size=3072,       # For RTX 3060 Ti
    use_mixed_precision=True,
    
    # Advanced
    temperature_mode='annealing',
    decoherence_base_rate=0.01,
    envariance_threshold=1e-3
)
```

## API Reference

### Main Functions

#### `create_quantum_mcts_v2`
```python
quantum_mcts = create_quantum_mcts_v2(
    enable_quantum: bool = True,
    branching_factor: Optional[int] = None,
    avg_game_length: Optional[int] = None,
    use_neural_network: bool = True,
    **kwargs
) -> QuantumMCTSV2
```

#### `apply_quantum_to_selection`
```python
ucb_scores = quantum_mcts.apply_quantum_to_selection(
    q_values: torch.Tensor,
    visit_counts: torch.Tensor,
    priors: torch.Tensor,
    c_puct: Optional[float] = None,      # Auto-computed if None
    parent_visits: Optional[torch.Tensor] = None,
    simulation_count: Optional[int] = None
) -> torch.Tensor
```

#### `check_envariance`
```python
converged = quantum_mcts.check_envariance(
    tree: Any,
    evaluators: Optional[List[Callable]] = None,
    threshold: Optional[float] = None
) -> bool
```

#### `update_simulation_count`
```python
quantum_mcts.update_simulation_count(N: int)
```

### Utility Functions

#### `OptimalParameters.compute_c_puct`
```python
c_puct = OptimalParameters.compute_c_puct(
    branching_factor: int,
    N: Optional[int] = None  # For RG correction
) -> float
```

#### `OptimalParameters.compute_num_hashes`
```python
num_hashes = OptimalParameters.compute_num_hashes(
    branching_factor: int,
    avg_game_length: int,
    has_neural_network: bool = True,
    prior_strength: float = 1.0
) -> int
```

## Performance Guidelines

### Hardware Recommendations

**RTX 3060 Ti / 3070 (8GB VRAM)**
```python
config = {
    'optimal_wave_size': 3072,
    'use_mixed_precision': True,
    'device': 'cuda'
}
```

**RTX 4090 (24GB VRAM)**
```python
config = {
    'optimal_wave_size': 4096,
    'use_mixed_precision': True,
    'device': 'cuda'
}
```

**CPU Only**
```python
config = {
    'optimal_wave_size': 256,
    'use_mixed_precision': False,
    'device': 'cpu'
}
```

### Expected Overhead

- **Without Neural Network**: 1.5-2.0x vs classical MCTS
- **With Neural Network**: 1.3-1.8x vs classical MCTS (lower due to synergy)
- **Phase-specific**: Quantum phase has highest overhead, classical phase lowest

## Examples

### Chess Implementation
```python
quantum_mcts = create_quantum_mcts_v2(
    branching_factor=35,
    avg_game_length=80,
    use_neural_network=True,
    optimal_wave_size=2048
)
```

### Go Implementation
```python
quantum_mcts = create_quantum_mcts_v2(
    branching_factor=250,
    avg_game_length=200,
    use_neural_network=True,
    optimal_wave_size=3072
)
```

### Gomoku Implementation
```python
quantum_mcts = create_quantum_mcts_v2(
    branching_factor=225,
    avg_game_length=50,
    use_neural_network=True,
    optimal_wave_size=1024
)
```

## Migration from v1

See `/docs/migration-guide-v2.md` for detailed migration instructions.

Key changes:
- Replace `quantum_features.py` imports with `quantum_features_v2.py`
- Add `branching_factor` and `avg_game_length` to configuration
- Pass `simulation_count` to selection function
- Use `update_simulation_count()` in main loop

## Testing

Run the comprehensive test suite:
```bash
python -m pytest tests/test_quantum_features_v2.py -v
```

Run the demonstration:
```bash
python examples/quantum_mcts_v2_demo.py
```

## Debugging

Enable debug logging:
```python
config = QuantumConfigV2(
    log_level='DEBUG',
    enable_profiling=True
)
```

Monitor statistics:
```python
stats = quantum_mcts.get_statistics()
print(f"Current phase: {stats['current_phase']}")
print(f"Temperature: {stats['current_temperature']:.3f}")
print(f"ℏ_eff: {stats['current_hbar_eff']:.3f}")
print(f"Phase transitions: {stats['phase_transitions']}")
```

## Theory and Documentation

- **Complete theory**: `/docs/v2.0/quantum-mcts-new.md`
- **Mathematical foundations**: `/docs/v2.0/quantum-theory-foundations-v2.md`
- **Parameter explanations**: `/docs/v2.0/quantum-parameters-explained-v2.md`
- **Implementation guide**: `/docs/v2.0/quantum-mcts-guide-v2.md`

## Citation

If using this implementation in research:

```bibtex
@software{quantum_mcts_v2,
  title = {Quantum-Enhanced MCTS v2.0: Path Integral Formulation with Discrete Information Time},
  author = {AlphaZero Omoknuni Project},
  year = {2024},
  note = {Implementation of quantum-inspired MCTS using discrete information time and full PUCT integration}
}
```