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
- `interference.py` - MinHash-based interference
- `phase_policy.py` - Phase-kicked priors
- `path_integral.py` - Path integral formulation

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

# Create components
game = GameInterface(GameType.GOMOKU)
evaluator = create_evaluator_for_game('gomoku')
mcts = HighPerformanceMCTS(game, evaluator)

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
- **Wave Processing**: Vectorized parallel search
- **GPU Acceleration**: CUDA kernels for critical operations
- **Mixed Precision**: FP16/FP32 for optimal performance
- **Memory Pooling**: Zero-allocation during search

## See Also

- [MCTS_COMPLETE_DOCUMENTATION.md](../MCTS_COMPLETE_DOCUMENTATION.md) - Full documentation
- [NEURAL_NETWORK_GUIDE.md](../NEURAL_NETWORK_GUIDE.md) - Neural network integration
- [MIGRATION_GUIDE.md](../MIGRATION_GUIDE.md) - Migration from old structure