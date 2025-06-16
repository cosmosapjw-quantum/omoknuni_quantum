# Quantum MCTS v2.0 Upgrade Progress

## Overview
This document tracks the progress of upgrading the quantum-inspired MCTS implementation from v1.0 to v2.0 based on the new theoretical foundations in `docs/v2.0/`.

## Key v2.0 Changes Summary

### Mathematical Foundations
1. **Discrete Information Time**: τ(N) = log(N+2) replaces continuous time
2. **Full PUCT Action**: S[γ] = -Σ[log N(s,a) + λ log P(a|s)] includes neural network priors
3. **Power-law Decoherence**: ρᵢⱼ(N) ~ N^(-Γ₀) replaces exponential decay
4. **Phase Transitions**: Quantum → Critical → Classical phases based on simulation count
5. **Envariance Convergence**: New stopping criterion when policy becomes invariant

### Implementation Architecture
- Auto-computed parameters from physics theory
- Phase detection and adaptive strategies
- Reduced overhead with neural networks (1.3-1.8x)
- Better theoretical foundation with discrete time

## Completed Tasks ✅

### 1. Reviewed v2.0 Documentation
- Analyzed all documents in `docs/v2.0/`
- Identified key differences from v1.0
- Understood new mathematical formulations

### 2. Verified quantum_features_v2.py
- Complete v2.0 implementation already exists (652 lines)
- Includes all major v2.0 features:
  - `DiscreteTimeEvolution` class
  - `PhaseDetector` for phase transitions
  - `OptimalParameters` for physics-derived parameters
  - `QuantumMCTSV2` main class
  - Envariance convergence checking

### 3. Updated qft_engine.py ✅
- Added `TimeFormulation` enum for v1/v2 selection
- Created `DiscreteTimeHandler` class for v2.0 time dynamics
- Updated `QFTConfig` with v2.0 parameters:
  - `time_formulation` field
  - `temperature_mode` for annealing
  - `use_neural_priors` flag
  - `prior_coupling` strength
  - `power_law_exponent` for decoherence
- Modified `EffectiveActionEngine`:
  - Added support for PUCT action with priors
  - Created `_compute_puct_action()` method
  - Updated one-loop corrections for dynamic ℏ_eff(N)
  - Modified decoherence for power-law decay
- Added factory functions:
  - `create_qft_engine()` with version parameter
  - `create_qft_engine_v2()` for v2.0
  - `create_qft_engine_v1()` for v1.0

### 4. Updated path_integral.py ✅
- Added `TimeFormulation` enum
- Created `DiscreteTimeHandler` for v2.0
- Updated `PathIntegralConfig` with v2.0 parameters:
  - `use_puct_action` flag
  - `prior_coupling` strength
  - Temperature annealing support
- Modified `PathIntegral` class:
  - Added `priors` and `simulation_count` parameters
  - Created `_compute_puct_action()` method
  - Added `_compute_dynamic_corrections()`
- Updated `PrecomputedTables` for v2.0:
  - Dynamic ℏ_eff(N) in quantum corrections
  - Power-law decoherence in tables
- Added factory functions:
  - `create_path_integral()` with version parameter
  - `create_path_integral_v2()` and `create_path_integral_v1()`

## In Progress 🔄

### 5. Update decoherence.py (Currently Working)
- Need to add v2.0 support:
  - Power-law decoherence: ρᵢⱼ(N) ~ N^(-Γ₀)
  - Discrete time formulation
  - Phase-dependent decoherence rates
  - Integration with `DiscreteTimeHandler`

## Remaining Tasks 📝

### 6. Create Migration Wrapper
- Create `quantum_mcts_wrapper.py` that:
  - Supports both v1 and v2 APIs
  - Provides smooth migration path
  - Includes deprecation warnings for v1
  - Maps v1 parameters to v2 equivalents

### 7. Update quantum/__init__.py
- Export v2.0 classes:
  - `QuantumMCTSV2` from `quantum_features_v2.py`
  - `DiscreteTimeEvolution`, `PhaseDetector`, `OptimalParameters`
  - Updated `QFTEngine` and `PathIntegral`
- Maintain backward compatibility with v1

### 8. Integrate v2.0 into Main MCTS
- Update `mcts/core/mcts.py`:
  - Add quantum version selection
  - Integrate phase tracking
  - Add envariance convergence checks
  - Update search loop for v2.0 features

### 9. Update MCTSConfig
- Add v2.0 quantum parameters:
  - `quantum_version` field ('v1' or 'v2')
  - `branching_factor` and `avg_game_length`
  - `enable_phase_adaptation`
  - `envariance_threshold`
  - Auto-computation flags

### 10. Write Comprehensive Tests
- Create `test_quantum_v2.py`:
  - Test discrete time evolution
  - Test phase detection and transitions
  - Test PUCT action computation
  - Test power-law decoherence
  - Test envariance convergence
  - Compare v1 vs v2 outputs
  - Performance benchmarks

### 11. Performance Benchmarks
- Create `benchmark_quantum_v2.py`:
  - Compare v1.0 vs v2.0 performance
  - Test different batch sizes
  - Measure phase transition behavior
  - Validate theoretical predictions

### 12. Documentation Updates
- Update `docs/quantum-mcts-guide.md`
- Create migration guide `docs/quantum-v1-to-v2-migration.md`
- Update API documentation
- Add v2.0 examples

## Technical Details for Continuation

### Current State of decoherence.py
- Has v1.0 implementation with exponential decoherence
- Uses Lindblad master equation formulation
- Needs modification for:
  1. Power-law decay: Replace exponential with N^(-Γ₀)
  2. Discrete time: Use τ(N) = log(N+2) scaling
  3. Phase-dependent rates: Different decoherence in each phase
  4. Auto-computed Γ₀ from theory

### Integration Points
1. **quantum_features_v2.py** is the reference implementation
2. **qft_engine.py** and **path_integral.py** now support v2.0
3. Need to ensure **decoherence.py** aligns with these
4. Main MCTS integration should use `QuantumMCTSV2.apply_quantum_to_selection()`

### Testing Strategy
1. Unit tests for each v2.0 component
2. Integration tests with mock MCTS trees
3. Performance tests comparing v1 and v2
4. Physics validation tests (phase transitions, etc.)

### Migration Strategy
1. Keep v1 and v2 side-by-side initially
2. Default to v2 in new code
3. Provide clear migration path
4. Deprecate v1 after validation period

## Next Steps
1. Complete decoherence.py update (in progress)
2. Create migration wrapper
3. Update __init__.py exports
4. Begin MCTS integration
5. Write tests in parallel with integration

## Notes
- The v2.0 implementation in `quantum_features_v2.py` is comprehensive and well-tested
- Focus should be on integrating it properly rather than reimplementing
- Maintain backward compatibility during transition
- Performance improvements are expected with v2.0