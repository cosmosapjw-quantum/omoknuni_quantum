# Quantum Parameters Explained: Temperature, Beta, and ℏ_eff

This document explains the three key quantum parameters in the AlphaZero Quantum MCTS system and their distinct roles in controlling quantum behavior.

## Table of Contents

1. [Overview](#overview)
2. [The Three Parameters](#the-three-parameters)
3. [Physical Interpretations](#physical-interpretations)
4. [Mathematical Roles](#mathematical-roles)
5. [Effects on MCTS Behavior](#effects-on-mcts-behavior)
6. [Practical Guidelines](#practical-guidelines)
7. [Examples and Configurations](#examples-and-configurations)

## Overview

The quantum MCTS system uses three temperature-related parameters that control different aspects of quantum behavior:

1. **quantum_temperature (T)**: Physical temperature controlling thermal fluctuations
2. **path_integral_beta (β)**: Inverse temperature for path integral formulation
3. **hbar_eff (ℏ)**: Effective Planck constant controlling quantum scale

While all three influence exploration vs exploitation, they operate through fundamentally different mechanisms.

## The Three Parameters

### 1. Quantum Temperature (T)
```yaml
quantum_temperature: float = 1.0  # Direct temperature
```

**What it is**: The physical temperature of the quantum system, measured in natural units.

**What it controls**:
- Magnitude of thermal fluctuations
- Random noise in quantum corrections
- Spread of probability distributions
- Energy scale of thermal excitations

**Key formula**: Thermal energy ~ k·T (where k is Boltzmann constant, set to 1)

### 2. Path Integral Beta (β)
```yaml
path_integral_beta: float = 1.0  # Inverse temperature β = 1/T_path
```

**What it is**: The inverse temperature parameter used in path integral formulation and Wick rotation.

**What it controls**:
- Which paths contribute to the path integral
- Suppression of high-action paths
- Convergence of imaginary time evolution
- Sharpness of path selection

**Key formula**: Path weight ~ exp(-β · Action)

### 3. Effective Planck Constant (ℏ_eff)
```yaml
hbar_eff: float = 0.1  # Quantum scale parameter
```

**What it is**: The fundamental quantum scale parameter that sets the strength of all quantum effects.

**What it controls**:
- Strength of quantum uncertainty
- Size of quantum corrections
- Wave-particle duality balance
- Quantum interference strength

**Key formula**: Uncertainty ~ ℏ / √(N+1) where N is visit count

## Physical Interpretations

### Temperature vs Beta vs ℏ: A Physics Analogy

Think of exploring a landscape of possibilities:

1. **Temperature (T)**: How violently particles are jiggling
   - High T → particles jump around more
   - Low T → particles settle down
   - Controls randomness and thermal noise

2. **Beta (β)**: How selective we are about energy barriers
   - High β → only lowest valleys matter
   - Low β → can explore over hills
   - Controls path filtering in superposition

3. **ℏ (hbar_eff)**: How "quantum" vs "classical" particles behave
   - High ℏ → particles are waves, can tunnel
   - Low ℏ → particles are points, must climb
   - Controls quantum mechanical effects

### Visual Metaphor

```
Classical Particle (ℏ → 0):          Quantum Particle (ℏ large):
    ●                                 ~~~●~~~
    │                                ╱       ╲
────┴────                         ────────────
Must climb hills                  Can tunnel through

Low Temperature (T small):         High Temperature (T large):
    ●                                ●   ●
    │                              ●│ ● │●
────┴────                         ────┴────
Stuck in valley                   Jumping everywhere

High Beta (β large):              Low Beta (β small):
[Path 1] ────── weight: 0.9       [Path 1] ────── weight: 0.4
[Path 2] ╌╌╌╌╌╌ weight: 0.1       [Path 2] ╌╌╌╌╌╌ weight: 0.3
[Path 3] ...... weight: 0.0       [Path 3] ...... weight: 0.3
Only best path matters            All paths contribute
```

## Mathematical Roles

### 1. Temperature in Quantum Corrections

```python
# Thermal fluctuations
thermal_noise = sqrt(temperature) * random_gaussian()

# Boltzmann-like distributions
probability ~ exp(-energy / temperature)

# Phase fluctuations
phase_variation = temperature * random_phase()
```

### 2. Beta in Path Integrals

```python
# Path integral weight
path_weight = exp(-beta * action)

# Partition function
Z = sum([exp(-beta * energy_i) for energy_i in spectrum])

# Wick rotation to imaginary time
time_evolution = exp(-beta * H) where H is Hamiltonian
```

### 3. ℏ in Quantum Mechanics

```python
# Heisenberg uncertainty
uncertainty = hbar_eff / sqrt(1 + visit_count)

# Quantum corrections (tree-level)
quantum_boost = hbar_eff * f(visits)

# One-loop corrections
loop_correction = hbar_eff * coupling^2 * log(1 + mass_eff)

# Interference strength
interference = hbar_eff * overlap_amplitude
```

## Effects on MCTS Behavior

### Temperature Effects

| Low T (0.1) | Medium T (1.0) | High T (10.0) |
|-------------|----------------|---------------|
| Exploitation-focused | Balanced | Exploration-focused |
| Low noise | Moderate noise | High noise |
| Deterministic | Mixed | Stochastic |
| Convergent | Adaptive | Divergent |

### Beta Effects

| Low β (0.1) | Medium β (1.0) | High β (10.0) |
|-------------|----------------|---------------|
| All paths matter | Balanced weighting | Only best paths |
| Democratic | Selective | Winner-take-all |
| Broad superposition | Focused | Sharp selection |
| High tunneling | Moderate | Low tunneling |

### ℏ Effects

| Low ℏ (0.01) | Medium ℏ (0.1) | High ℏ (1.0) |
|--------------|----------------|--------------|
| Nearly classical | Quantum-inspired | Strongly quantum |
| Weak uncertainty | Moderate | Large uncertainty |
| No interference | Some interference | Strong interference |
| Particle-like | Mixed | Wave-like |

## Practical Guidelines

### Choosing Temperature

```yaml
# Conservative exploration
quantum_temperature: 0.5

# Standard setting
quantum_temperature: 1.0

# Aggressive exploration
quantum_temperature: 2.0
```

**When to increase T**:
- Early in training
- Stuck in local optima
- Need more move diversity
- Exploring new positions

**When to decrease T**:
- Late in training
- Need convergence
- Tactical positions
- Endgame scenarios

### Choosing Beta

```yaml
# Include many paths (quantum superposition)
path_integral_beta: 0.5

# Balanced path selection
path_integral_beta: 1.0

# Focus on optimal paths
path_integral_beta: 5.0
```

**When to increase β**:
- Need deterministic behavior
- Computing resources limited
- Clear best moves exist
- Convergence phase

**When to decrease β**:
- Need path diversity
- Exploring opening theory
- Strategic positions
- Avoid tunnel vision

### Choosing ℏ_eff

```yaml
# Minimal quantum effects
hbar_eff: 0.01

# Moderate quantum enhancement
hbar_eff: 0.1

# Strong quantum behavior
hbar_eff: 0.5
```

**When to increase ℏ**:
- Low-visit nodes need boost
- Increase exploration bonus
- Enable quantum tunneling
- Break symmetries

**When to decrease ℏ**:
- Reduce overhead
- Classical-like behavior
- High-visit nodes
- Precision needed

## Examples and Configurations

### Example 1: Classical-Like Behavior
```yaml
# Nearly classical with minimal quantum
quantum_temperature: 0.5    # Low thermal noise
path_integral_beta: 5.0     # Sharp path selection
hbar_eff: 0.01             # Minimal quantum effects
```
**Use case**: Endgame positions, tactical puzzles, convergence phase

### Example 2: Balanced Quantum Enhancement
```yaml
# Standard quantum enhancement
quantum_temperature: 1.0    # Moderate thermal effects
path_integral_beta: 1.0     # Balanced path weights
hbar_eff: 0.1              # Noticeable quantum boost
```
**Use case**: General training, middle game, balanced exploration

### Example 3: Strong Quantum Exploration
```yaml
# Maximum quantum exploration
quantum_temperature: 2.0    # High thermal activity
path_integral_beta: 0.5     # Include many paths
hbar_eff: 0.3              # Strong quantum effects
```
**Use case**: Opening exploration, breaking out of local optima, diversity

### Example 4: Quantum Tunneling Focus
```yaml
# Quantum tunneling without thermal noise
quantum_temperature: 0.3    # Low thermal noise
path_integral_beta: 0.3     # Enable tunneling paths
hbar_eff: 0.5              # Strong quantum mechanics
```
**Use case**: Positions with barriers, need creative solutions

### Game-Specific Recommendations

#### Gomoku
```yaml
quantum_temperature: 1.0    # Moderate exploration
path_integral_beta: 1.0     # Balanced paths
hbar_eff: 0.1              # Standard quantum
```
Pattern recognition benefits from quantum interference

#### Go 19x19
```yaml
quantum_temperature: 1.5    # Higher for strategic diversity
path_integral_beta: 0.8     # Include multiple strategies
hbar_eff: 0.15             # Stronger quantum for complexity
```
Long-term planning benefits from path integral formulation

#### Chess
```yaml
quantum_temperature: 0.8    # Lower for tactical precision
path_integral_beta: 2.0     # Focus on best lines
hbar_eff: 0.05             # Minimal quantum
```
Tactical nature requires more classical approach

## Summary

The three parameters form a complete quantum control system:

1. **Temperature (T)**: Controls thermal randomness and exploration noise
2. **Beta (β)**: Controls path selection sharpness in quantum superposition
3. **ℏ_eff**: Controls the fundamental strength of quantum effects

They work together but independently:
- **T** adds noise, **β** filters paths, **ℏ** enables quantum mechanics
- High T + Low β + High ℏ = Maximum exploration
- Low T + High β + Low ℏ = Maximum exploitation

Understanding their distinct roles allows fine-tuned control over the quantum-classical tradeoff in MCTS exploration.

## Quick Reference Table

| Parameter | Symbol | Range | Controls | Increase for | Decrease for |
|-----------|--------|-------|----------|--------------|--------------|
| quantum_temperature | T | 0.1-10 | Thermal noise | More randomness | Convergence |
| path_integral_beta | β | 0.1-10 | Path selection | Focus on best | Path diversity |
| hbar_eff | ℏ | 0.01-1 | Quantum scale | Quantum effects | Classical behavior |

## Formulas at a Glance

- **Thermal**: `noise ~ sqrt(T) × random()`
- **Path weight**: `weight ~ exp(-β × action)`
- **Uncertainty**: `Δ ~ ℏ / sqrt(visits + 1)`
- **Combined**: `quantum_correction = ℏ × f(visits) × g(T) × paths(β)`