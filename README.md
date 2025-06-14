# AlphaZero Omoknuni

High-performance AlphaZero implementation achieving **168k+ simulations/second** through GPU-accelerated MCTS with optional quantum enhancements.

## Features

- **Multi-Game Support**: Chess, Go (9x9, 13x13, 19x19), and Gomoku
- **High Performance**: 168k+ sims/sec on RTX 3060 Ti via wave parallelization
- **Quantum Enhancement**: Optional QFT-based improvements with < 2x overhead
- **Production Ready**: Full training pipeline with self-play, arena, and ELO tracking
- **GPU Optimized**: CUDA kernels, mixed precision, and tensor cores support

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/omoknuni_quantum.git
cd omoknuni_quantum

# Build C++ components
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON
make -j$(nproc)
cd ..

# Install Python package
cd python
pip install -e .
```

### Training a Model

```bash
cd python
python -m mcts.neural_networks.unified_training_pipeline \
    --config ../configs/gomoku_classical.yaml
```

### Running Self-Play

```python
from mcts.core.mcts import MCTS, MCTSConfig
from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
import alphazero_py

# Create evaluator and MCTS
evaluator = ResNetEvaluator(game_type='gomoku', device='cuda')
config = MCTSConfig(
    num_simulations=1600,
    min_wave_size=3072,
    max_wave_size=3072,
    adaptive_wave_sizing=False,  # Critical for performance
    device='cuda'
)
mcts = MCTS(config, evaluator)

# Run search
game_state = alphazero_py.GomokuState()
policy = mcts.search(game_state)
best_action = mcts.get_best_action(game_state)
```

## Performance Configuration

### RTX 3060 Ti (8GB VRAM)

```python
config = MCTSConfig(
    min_wave_size=3072,
    max_wave_size=3072,
    adaptive_wave_sizing=False,
    memory_pool_size_mb=2048,
    max_tree_nodes=500000,
    use_mixed_precision=True,
    use_cuda_graphs=True,
    use_tensor_cores=True
)
```

### RTX 3090 (24GB VRAM)

```python
config = MCTSConfig(
    min_wave_size=4096,
    max_wave_size=4096,
    adaptive_wave_sizing=False,
    memory_pool_size_mb=8192,
    max_tree_nodes=2000000
)
```

## Quantum Features (Optional)

Enable quantum enhancements for improved exploration:

```python
from mcts.quantum import QuantumConfig

config = MCTSConfig(
    enable_quantum=True,
    quantum_config=QuantumConfig(
        quantum_level="one_loop",
        hbar_eff=0.5,
        coupling_strength=0.3,
        temperature=1.0
    )
)
```

## Documentation

- [Training Guide](docs/alphazero-training-guide.md) - Complete training pipeline guide
- [Implementation Guide](docs/implementation-guide.md) - Architecture and implementation details
- [Performance Optimization](docs/performance-optimization.md) - GPU optimization techniques
- [Quantum MCTS Guide](docs/quantum-mcts-guide.md) - Quantum enhancement documentation
- [Project Structure](docs/PROJECT_STRUCTURE.md) - Codebase organization

## Benchmarks

| Hardware | Simulations/sec | Configuration |
|----------|----------------|---------------|
| RTX 3060 Ti | 168,000+ | Wave size: 3072, Mixed precision |
| RTX 3090 | 250,000+ | Wave size: 4096, Mixed precision |
| RTX 4090 | 400,000+ | Wave size: 8192, Mixed precision |

## Project Structure

```
omoknuni_quantum/
├── src/           # C++ game implementations
├── python/
│   ├── mcts/      # MCTS implementation
│   │   ├── core/  # Core MCTS algorithms
│   │   ├── gpu/   # GPU kernels and optimization
│   │   ├── neural_networks/  # Neural network models
│   │   └── quantum/  # Quantum enhancements
│   └── tests/     # Unit tests
├── configs/       # Training configurations
└── docs/          # Documentation
```

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to the main branch.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the AlphaZero paper by DeepMind
- Quantum enhancements inspired by path integral formulation
- Wave parallelization adapted from efficient MCTS research