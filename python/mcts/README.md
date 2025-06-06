# MCTS Package Structure

This package implements high-performance Monte Carlo Tree Search with quantum-inspired enhancements.

## Package Organization

### core/
Core MCTS components and algorithms:
- `node.py` - Tree node representation
- `tree_arena.py` - Memory-efficient tree management
- `game_interface.py` - Abstract game interface
- `evaluator.py` - Neural network evaluation interface
- `batch_game_ops.py` - Batch game operations
- `high_performance_mcts.py` - Main MCTS implementation
- `mcts.py` - Legacy MCTS implementation (for compatibility)
- `concurrent_mcts.py` - Concurrent MCTS variants
- `wave_engine.py` - Base wave engine implementation

### neural_networks/
Neural network models and training:
- `nn_framework.py` - Flexible model loading framework
- `nn_model.py` - AlphaZero-style network
- `resnet_model.py` - ResNet implementation
- `resnet_evaluator.py` - ResNet-based evaluator
- `training_pipeline.py` - Training infrastructure

### gpu/
GPU acceleration components:
- `csr_tree.py` - Compressed sparse row tree format
- `optimized_wave_engine.py` - Vectorized wave processing
- `csr_gpu_kernels.py` - CSR-specific GPU kernels
- `optimized_cuda_kernels.py` - Optimized CUDA kernels
- `gpu_optimizer.py` - GPU memory optimization
- `gpu_tree_kernels.py` - Tree operation kernels
- `cuda_kernels.py` - Basic CUDA kernels
- `gpu_attack_defense.py` - GPU attack/defense patterns

### quantum/
Quantum-inspired enhancements:
- `quantum_features.py` - Main quantum MCTS implementation (production)
- `qft_engine.py` - Optimized QFT computations
- `path_integral.py` - Path integral formulation
- `interference_gpu.py` - GPU-accelerated MinHash interference
- `decoherence.py` - Decoherence dynamics
- `envariance.py` - Entanglement-assisted robustness
- `quantum_darwinism.py` - Classical information extraction
- `phase_policy.py` - Phase-kicked priors
- `rg_flow.py` - Renormalization group flow
- `thermodynamics.py` - Thermodynamic properties

### utils/
Utility components:
- `config_manager.py` - Configuration management
- `resource_monitor.py` - Resource monitoring
- `state_delta_encoder.py` - State compression
- `attack_defense.py` - Attack/defense utilities

## Quick Start

```python
# High-level import (recommended)
from mcts import (
    HighPerformanceMCTS,
    GameInterface,
    ResNetEvaluator,
    create_evaluator_for_game
)
from mcts.quantum import create_quantum_mcts

# Create components
game = GameInterface(GameType.GOMOKU)
evaluator = create_evaluator_for_game('gomoku')

# Standard MCTS
mcts = HighPerformanceMCTS(game, evaluator)

# Quantum-enhanced MCTS
quantum_mcts = create_quantum_mcts(
    enable_quantum=True,
    hbar_eff=0.1,  # Quantum strength
    min_wave_size=32  # Batch threshold
)

# Use quantum features in selection
ucb_scores = quantum_mcts.apply_quantum_to_selection(
    q_values, visit_counts, priors
)

# Run search
action_probs = mcts.search(initial_state)
```

## Module Dependencies

```
core
├── neural_networks (for evaluation)
├── gpu (for acceleration)
└── utils (for configuration)

gpu
├── core (uses tree structures)
└── utils (for monitoring)

neural_networks
└── core (implements evaluator interface)

quantum
├── core (extends MCTS)
└── gpu (uses acceleration)
```

## Performance Features

- **CSR Tree Format**: Memory-efficient sparse representation
- **Wave Processing**: Vectorized parallel search (256-2048 paths)
- **GPU Acceleration**: CUDA kernels for critical operations
- **Mixed Precision**: FP16/FP32 for optimal performance
- **Memory Pooling**: Zero-allocation during search
- **Quantum Features**: < 2x overhead with full QFT physics
- **Pre-computed Tables**: O(1) quantum correction lookup
- **Adaptive Parameters**: ℏ_eff = 1/√N̄ scaling

## See Also

- [MCTS_COMPLETE_DOCUMENTATION.md](../MCTS_COMPLETE_DOCUMENTATION.md) - Full documentation
- [NEURAL_NETWORK_GUIDE.md](../NEURAL_NETWORK_GUIDE.md) - Neural network integration
- [MIGRATION_GUIDE.md](../MIGRATION_GUIDE.md) - Migration from old structure