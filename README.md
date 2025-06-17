# AlphaZero Omoknuni

High-performance AlphaZero implementation achieving **168k+ simulations/second** through GPU-accelerated MCTS with optional quantum enhancements.

## Features

- **Multi-Game Support**: Chess, Go (9x9, 13x13, 19x19), and Gomoku
- **High Performance**: 168k+ sims/sec on RTX 3060 Ti via wave parallelization
- **Quantum Enhancement**: Optional QFT-based improvements with < 2x overhead
- **Production Ready**: Full training pipeline with self-play, arena, and ELO tracking
- **GPU Optimized**: CUDA kernels, mixed precision, and tensor cores support

## System Requirements

### Software Requirements

- **Operating System**: Ubuntu 22.04/24.04 on WSL2 (Windows 11)
- **CUDA**: 12.2 (via NVIDIA Studio Driver 537.42 for WSL2 compatibility)
- **PyTorch**: 2.0+ with CUDA 12.1 support
- **GCC**: 12.x (required for CUDA compatibility, GCC 13+ not supported by CUDA 12.2)
- **Python**: 3.8+
- **CMake**: 3.12+
- **Ninja**: Recommended for faster builds

### WSL2 CUDA Setup

This project is optimized for WSL2 with specific driver requirements:

```bash
# On Windows host:
# Install NVIDIA Studio Driver 537.42 (critical for WSL2 CUDA stability)
# This specific version avoids common WSL2 CUDA issues
# Download from: https://www.nvidia.com/download/driverResults.aspx/208074/

# In WSL2:
# Verify CUDA is available
nvidia-smi  # Should show your GPU

# Install required dependencies
sudo apt update
sudo apt install g++-12 gcc-12 build-essential cmake ninja-build

# Install CUDA Toolkit 12.2 (if not already installed)
# Follow: https://developer.nvidia.com/cuda-12-2-2-download-archive

# Set up CUDA environment (automatic with our setup)
source cuda_config.sh
```

**Note**: NVIDIA Studio Driver 537.42 is specifically recommended for WSL2 as newer drivers may cause CUDA initialization issues.

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/omoknuni_quantum.git
cd omoknuni_quantum

# Set up CUDA environment for GCC 12
source cuda_config.sh

# Build C++ components
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON
make -j$(nproc)
cd ..

# Install Python package
cd python
pip install -e .

# Compile CUDA kernels (automatic fallback if this fails)
python compile_kernels.py
```

### Troubleshooting CUDA Compilation

If you encounter GCC version errors during CUDA kernel compilation:

```bash
# The project automatically uses GCC 12, but if issues persist:
export CUDAHOSTCXX=g++-12
export NVCC_APPEND_FLAGS="-ccbin g++-12"

# Verify correct GCC version is being used
g++-12 --version  # Should show GCC 12.x
nvcc --version    # Should show CUDA 12.2
```

### Training a Model

```bash
# Start new training
python train.py --game gomoku --iterations 100 --workers 8

# Resume from checkpoint
python train.py --resume experiments/gomoku_unified_training/checkpoints/latest_checkpoint.pt --iterations 50

# With custom config
python train.py --config configs/gomoku_classical.yaml
```

See [docs/resume-training-guide.md](docs/resume-training-guide.md) for detailed training and resume options.

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

### Test Environment
- **OS**: Windows 11 with WSL2 (Ubuntu 24.04)
- **CPU**: AMD Ryzen 9 5900X (12 cores / 24 threads)
- **RAM**: 64GB DDR4
- **GPU**: RTX 3060 Ti (8GB VRAM)
- **CUDA**: 12.2 with NVIDIA Studio Driver 537.42
- **PyTorch**: 2.0+ with CUDA 12.1 support

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