# Quantum MCTS v2.0 Migration Guide

## Overview

This guide helps you migrate from the current quantum MCTS implementation to v2.0, which introduces significant improvements based on rigorous physics foundations and better neural network integration.

## Key Changes

### 1. Import Changes

**Current:**
```python
from mcts.quantum import create_quantum_mcts, QuantumConfig
```

**v2.0:**
```python
from mcts.quantum.quantum_features_v2 import create_quantum_mcts_v2, QuantumConfigV2
```

### 2. Configuration Changes

**Current:**
```python
config = QuantumConfig(
    hbar_eff=0.5,              # Fixed value
    temperature=1.0,           # Fixed value
    coupling_strength=0.3,
    min_wave_size=32,
    optimal_wave_size=512
)
```

**v2.0:**
```python
config = QuantumConfigV2(
    # Auto-computed parameters
    hbar_eff=None,             # Auto: c_puct(N+2)/(√(N+1)log(N+2))
    temperature_mode='annealing',  # T(N) = T₀/log(N+2)
    
    # Game parameters for auto-computation
    branching_factor=20,       # Your game's branching factor
    avg_game_length=50,        # Average game length
    
    # Neural network integration
    use_neural_prior=True,     # Enable NN prior integration
    prior_coupling='auto',     # λ = c_puct
    
    # Phase adaptation
    enable_phase_adaptation=True,
    
    # Performance (3072 optimal for RTX 3060 Ti)
    optimal_wave_size=3072
)
```

### 3. API Changes

#### Creating Quantum MCTS

**Current:**
```python
quantum_mcts = create_quantum_mcts(
    enable_quantum=True,
    quantum_level='tree_level',
    hbar_eff=0.5
)
```

**v2.0:**
```python
quantum_mcts = create_quantum_mcts_v2(
    enable_quantum=True,
    branching_factor=20,      # Enables auto-computation
    avg_game_length=50,
    use_neural_network=True   # Adjusts parameters for NN
)
```

#### Selection Enhancement

**Current:**
```python
ucb_scores = quantum_mcts.apply_quantum_to_selection(
    q_values, visit_counts, priors, c_puct
)
```

**v2.0:**
```python
ucb_scores = quantum_mcts.apply_quantum_to_selection(
    q_values, visit_counts, priors, 
    c_puct=None,              # Uses auto-computed value
    simulation_count=N        # Current total simulations
)
```

### 4. New Features

#### Phase Detection

```python
# v2.0 automatically detects and adapts to phases
quantum_mcts.update_simulation_count(N)

# Get current phase
stats = quantum_mcts.get_statistics()
current_phase = stats['current_phase']  # 'quantum', 'critical', or 'classical'
```

#### Envariance Convergence

```python
# Check convergence using new criterion
converged = quantum_mcts.check_envariance(
    tree,
    threshold=1e-3  # Optional, uses config default
)
```

#### Optimal Parameter Computation

```python
from mcts.quantum.quantum_features_v2 import OptimalParameters

# Compute optimal c_puct
c_puct = OptimalParameters.compute_c_puct(branching_factor)

# Compute optimal hash functions
num_hashes = OptimalParameters.compute_num_hashes(
    branching_factor, avg_game_length, has_neural_network=True
)
```

## Migration Examples

### Example 1: Basic Migration

**Current Code:**
```python
from mcts.quantum import create_quantum_mcts

# Fixed parameters
quantum_mcts = create_quantum_mcts(
    enable_quantum=True,
    quantum_level='tree_level',
    hbar_eff=0.1,
    temperature=1.0,
    min_wave_size=32
)

# In search loop
ucb = quantum_mcts.apply_quantum_to_selection(
    q_values, visits, priors, c_puct=1.414
)
```

**v2.0 Code:**
```python
from mcts.quantum.quantum_features_v2 import create_quantum_mcts_v2

# Auto-computed parameters
quantum_mcts = create_quantum_mcts_v2(
    enable_quantum=True,
    branching_factor=20,      # Your game's branching
    avg_game_length=50,       # Your game's avg length
    use_neural_network=True
)

# In search loop
ucb = quantum_mcts.apply_quantum_to_selection(
    q_values, visits, priors,
    simulation_count=current_N  # Pass current simulation count
)
```

### Example 2: Full AlphaZero Integration

**Current Code:**
```python
class AlphaZeroMCTS:
    def __init__(self, game, model, config):
        self.quantum_mcts = create_quantum_mcts(
            enable_quantum=True,
            hbar_eff=0.5,
            temperature=1.0
        )
    
    def search(self, root_state, num_sims):
        for i in range(num_sims):
            # ... selection logic ...
            ucb = self.quantum_mcts.apply_quantum_to_selection(
                q_values, visits, priors, self.c_puct
            )
```

**v2.0 Code:**
```python
class AlphaZeroMCTSV2:
    def __init__(self, game, model, config):
        self.quantum_mcts = create_quantum_mcts_v2(
            enable_quantum=True,
            branching_factor=game.branching_factor,
            avg_game_length=game.avg_length,
            use_neural_network=True,  # Using AlphaZero NN
            optimal_wave_size=3072    # For GPU efficiency
        )
        # c_puct is auto-computed optimally
    
    def search(self, root_state, num_sims):
        for i in range(num_sims):
            # Update simulation count for phase detection
            self.quantum_mcts.update_simulation_count(i)
            
            # ... selection logic ...
            ucb = self.quantum_mcts.apply_quantum_to_selection(
                q_values, visits, priors,
                simulation_count=i
            )
            
            # Check convergence periodically
            if i % 100 == 0:
                if self.quantum_mcts.check_envariance(self.tree):
                    break  # Converged early
```

### Example 3: Custom Configuration

**Current Code:**
```python
config = QuantumConfig(
    quantum_level='one_loop',
    hbar_eff=0.3,
    coupling_strength=0.2,
    decoherence_rate=0.01,
    interference_method='phase_kick',
    phase_kick_strength=0.1
)
quantum_mcts = QuantumMCTS(config)
```

**v2.0 Code:**
```python
config = QuantumConfigV2(
    quantum_level='one_loop',
    # hbar_eff computed automatically
    coupling_strength=0.3,         # Optimal from RG
    temperature_mode='annealing',  # Better than fixed
    
    # Specify game properties
    branching_factor=50,
    avg_game_length=100,
    
    # Phase-aware strategy
    enable_phase_adaptation=True,
    
    # Interference auto-configured
    interference_method='minhash',
    num_hash_functions=None,       # Auto: √(b·L)
    
    # Neural network
    use_neural_prior=True,
    prior_coupling='auto'          # λ = c_puct
)
quantum_mcts = QuantumMCTSV2(config)
```

## Performance Optimization

### Recommended Settings for Different Hardware

**RTX 3060 Ti / 3070:**
```python
config = QuantumConfigV2(
    optimal_wave_size=3072,
    use_mixed_precision=True,
    device='cuda'
)
```

**RTX 4090:**
```python
config = QuantumConfigV2(
    optimal_wave_size=4096,
    use_mixed_precision=True,
    device='cuda'
)
```

**CPU Only:**
```python
config = QuantumConfigV2(
    optimal_wave_size=256,
    use_mixed_precision=False,
    device='cpu'
)
```

## Debugging Tips

### 1. Check Auto-Computed Parameters

```python
quantum_mcts = create_quantum_mcts_v2(branching_factor=20)
print(f"Auto-computed c_puct: {quantum_mcts.config.c_puct}")
print(f"Auto-computed num_hashes: {quantum_mcts.config.num_hash_functions}")
```

### 2. Monitor Phase Transitions

```python
stats = quantum_mcts.get_statistics()
print(f"Current phase: {stats['current_phase']}")
print(f"Phase transitions: {stats['phase_transitions']}")
print(f"Current temperature: {stats['current_temperature']:.3f}")
print(f"Current hbar_eff: {stats['current_hbar_eff']:.3f}")
```

### 3. Verify Prior Integration

```python
# Check that neural network priors are being used
config = quantum_mcts.config
assert config.use_neural_prior == True
assert config.prior_coupling == config.c_puct  # Should match
```

## Common Issues and Solutions

### Issue 1: Parameters Not Auto-Computing

**Problem:** `c_puct` or `num_hash_functions` remain None

**Solution:** Ensure you provide `branching_factor` and `avg_game_length`:
```python
quantum_mcts = create_quantum_mcts_v2(
    branching_factor=20,  # Required for auto-computation
    avg_game_length=50    # Required for hash functions
)
```

### Issue 2: No Phase Transitions

**Problem:** Always stays in quantum phase

**Solution:** Call `update_simulation_count()` regularly:
```python
for N in range(num_simulations):
    quantum_mcts.update_simulation_count(N)
    # ... rest of MCTS logic ...
```

### Issue 3: Different Results Than v1

**Expected:** v2.0 may produce different results due to:
- Better parameter selection
- Phase-aware strategies  
- Prior field interpretation
- Power-law vs exponential decoherence

These differences should generally improve performance.

## Benefits of Upgrading

1. **No Manual Tuning**: Parameters computed from first principles
2. **Better NN Integration**: Priors treated as external field
3. **Adaptive Strategy**: Automatic phase detection and adaptation
4. **Improved Convergence**: Envariance criterion and power-law decoherence
5. **Lower Overhead**: 1.3-1.8x with neural networks (vs 1.5-2x)

## Rollback Plan

If you need to rollback to v1:

1. Keep imports separate:
   ```python
   # v1
   from mcts.quantum.quantum_features import create_quantum_mcts
   
   # v2
   from mcts.quantum.quantum_features_v2 import create_quantum_mcts_v2
   ```

2. Use feature flags:
   ```python
   USE_V2 = True  # Toggle this
   
   if USE_V2:
       quantum_mcts = create_quantum_mcts_v2(...)
   else:
       quantum_mcts = create_quantum_mcts(...)
   ```

## Support

For questions or issues with migration:
1. Check the comprehensive test suite in `test_quantum_features_v2.py`
2. Refer to the full documentation in `quantum-mcts-guide-v2.md`
3. Use the debug logging: `config.log_level = 'DEBUG'`