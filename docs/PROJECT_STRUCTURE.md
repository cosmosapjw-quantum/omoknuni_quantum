# Project Structure Guide

This document provides a comprehensive overview of the Omoknuni Quantum (AlphaZero) project organization.

## Directory Structure Overview

```
omoknuni_quantum/
├── build/                      # CMake build directory (generated)
├── configs/                    # Configuration files for training and optimization
├── docs/                       # Project documentation
├── include/                    # C++ header files
├── python/                     # Python implementation and MCTS code
├── src/                        # C++ source files
├── tests/                      # C++ unit tests
├── CMakeLists.txt             # Main CMake configuration
├── CLAUDE.md                  # AI assistant guidance
├── README.md                  # Project overview
└── setup.py                   # Python package setup
```

## Key Directories and Their Contents

### `/configs/` - Configuration Files
Training and optimization configurations for different games and hardware setups:
- `gomoku_classical.yaml` - Standard Gomoku training config
- `chess_classical.yaml` - Standard Chess training config
- `ryzen9_3060ti_optimized.yaml` - Hardware-optimized config for RTX 3060 Ti
- `*_ryzen9_3060ti.yaml` - Game-specific configs optimized for RTX 3060 Ti
- `minimal_test.yaml` - Quick testing configuration
- `OPTIMIZATION_GUIDE.md` - Performance tuning documentation
- `TRAINING_TIME_ESTIMATES.md` - Training duration estimates

### `/docs/` - Documentation
Comprehensive project documentation:
- `PROJECT_STRUCTURE.md` - This file
- `quantum-mcts-complete-guide.md` - Complete quantum MCTS documentation
- `alphazero-training-guide.md` - Training pipeline guide
- `IMPLEMENTATION_OVERVIEW.md` - High-level architecture overview
- `qft-mcts-*.md` - Quantum Field Theory MCTS documentation
- `quantum-parameters-explained.md` - Quantum parameter tuning guide
- `one-loop-design.md` - One-loop quantum corrections design
- `hardcoded-parameters-to-configure.md` - Configuration parameter reference

### `/include/` - C++ Headers
Header files for the C++ game engine:

#### `/include/core/` - Core Interfaces
- `igamestate.h` - Abstract game state interface (all games implement this)
- `export_macros.h` - DLL export macros
- `game_export.h` - Game-specific export definitions
- `illegal_move_exception.h` - Exception handling

#### `/include/games/` - Game Implementations
- `/chess/` - Chess game headers
  - `chess_state.h` - Chess game state
  - `chess_rules.h` - Chess rule validation
  - `chess_types.h` - Chess-specific types
  - `chess960.h` - Chess960 variant support
- `/go/` - Go game headers
  - `go_state.h` - Go game state
  - `go_rules.h` - Go rule validation
- `/gomoku/` - Gomoku game headers
  - `gomoku_state.h` - Gomoku game state
  - `gomoku_rules.h` - Gomoku rule validation

#### `/include/utils/` - Utility Headers
- `attack_defense_module.h` - Pattern recognition for tactical analysis
- `zobrist_hash.h` - Position hashing
- `logger.h` - Logging utilities
- `hash_specializations.h` - Custom hash functions

### `/src/` - C++ Source Files
Implementation files matching the header structure:
- `/core/` - Core interface implementations
- `/games/` - Game-specific implementations
- `/utils/` - Utility implementations
- `/python/` - Python bindings
  - `bindings.cpp` - Pybind11 bindings exposing C++ to Python

### `/python/` - Python Implementation
Main Python codebase for MCTS and neural networks:

#### `/python/mcts/` - MCTS Implementation
- `/core/` - Core MCTS algorithms
  - `mcts.py` - Standard MCTS implementation
  - `optimized_mcts.py` - High-performance MCTS (168k sims/sec)
  - `unified_mcts.py` - Unified MCTS interface
  - `wave_mcts.py` - Wave-based vectorized MCTS
  - `evaluator.py` - Neural network evaluation interface
  - `game_interface.py` - Game abstraction layer
  - `mcts_config.py` - Configuration classes
- `/gpu/` - GPU acceleration
  - `unified_kernels.py` - CUDA kernel wrappers
  - `csr_tree.py` - Compressed sparse row tree format
  - `*.cu` - CUDA kernel implementations
  - `*.so` - Compiled CUDA modules
- `/quantum/` - Quantum enhancements
  - `quantum_features.py` - Main quantum MCTS implementation
  - `qft_engine.py` - Quantum Fourier Transform engine
  - `path_integral.py` - Path integral formulation
  - `interference.py` - Quantum interference patterns
  - `decoherence.py` - Decoherence modeling
- `/neural_networks/` - Neural network components
  - `resnet_evaluator.py` - ResNet-based position evaluator
  - `unified_training_pipeline.py` - Training orchestration
  - `self_play_module.py` - Self-play data generation
  - `arena_module.py` - Model comparison arena
- `/utils/` - Utilities
  - `config_system.py` - Configuration management
  - `safe_multiprocessing.py` - CUDA-safe multiprocessing
  - `resource_monitor.py` - Resource usage tracking
  - `tensor_pool.py` - Tensor memory management

#### Python Scripts
- `train.py` - Main training script
- `compile_kernels.py` - CUDA kernel compilation
- `benchmark_mcts_profiler.py` - Performance benchmarking
- `benchmark_quantum_performance.py` - Quantum overhead analysis
- `example_self_play.py` - Self-play demonstration

### `/tests/` - Test Suites
Comprehensive test coverage:
- C++ tests in root `/tests/` directory
- Python tests in `/python/tests/`
- Integration tests for full pipeline validation

## Build Artifacts and Generated Files

### `/build/` - CMake Build Directory
Generated by CMake, contains:
- Compiled binaries in `bin/`
- Libraries in `lib/`
- CMake cache and configuration

### Python Build Artifacts
- `alphazero_py.so` - Compiled Python extension module
- `__pycache__/` directories - Python bytecode
- `*.cpython-*.so` - Compiled CUDA extensions

### Training Artifacts
- `/gomoku_classical/`, `/gomoku_ryzen9_3060ti_optimized/` - Training outputs
  - `best_models/` - Saved model checkpoints
  - `self_play_data/` - Generated training data
  - `arena_logs/` - Model comparison results
  - `checkpoints/` - Training checkpoints
  - `config.yaml` - Training configuration snapshot

### Profiling Results
- `/mcts_profiling_results/` - Standard MCTS performance analysis
- `/mcts_quantum_profiling_results/` - Quantum MCTS overhead analysis

## Configuration Files

### Root Configuration
- `CMakeLists.txt` - Main CMake configuration
- `setup.py` - Python package configuration
- `requirements.txt` - Python dependencies

### Training Configurations
Located in `/configs/`, YAML format:
```yaml
# Example structure
game_type: "gomoku"
mcts:
  min_wave_size: 3072
  max_wave_size: 3072
  adaptive_wave_sizing: false  # Critical for performance
neural_network:
  architecture: "resnet"
  blocks: 10
training:
  batch_size: 256
  learning_rate: 0.001
```

## Key Entry Points

### C++ Entry Points
- Game implementations: `/src/games/*/`
- Python bindings: `/src/python/bindings.cpp`
- Test executables: `/build/bin/`

### Python Entry Points
- Training: `python/train.py`
- MCTS usage: `from mcts.core.optimized_mcts import MCTS`
- Game interface: `import alphazero_py`
- Quantum MCTS: `from mcts.quantum import create_quantum_mcts`

## Documentation Map

### Getting Started
1. `README.md` - Project overview
2. `docs/IMPLEMENTATION_OVERVIEW.md` - Architecture overview
3. `CLAUDE.md` - Development guidelines

### Training and Optimization
1. `docs/alphazero-training-guide.md` - Complete training guide
2. `configs/OPTIMIZATION_GUIDE.md` - Performance tuning
3. `configs/TRAINING_TIME_ESTIMATES.md` - Resource planning

### Quantum Features
1. `docs/quantum-mcts-complete-guide.md` - Comprehensive quantum guide
2. `docs/quantum-parameters-explained.md` - Parameter tuning
3. `docs/qft-mcts-math-foundations.md` - Mathematical foundations

### Development
1. `docs/hardcoded-parameters-to-configure.md` - Configuration reference
2. `python/mcts/OPTIMIZED_MCTS_GUIDE.md` - Performance optimization
3. Test files for API usage examples

## Performance-Critical Paths

### Hot Paths
1. **MCTS Selection**: `mcts/core/optimized_mcts.py:select_batch()`
2. **Neural Evaluation**: `mcts/neural_networks/resnet_evaluator.py:evaluate_batch()`
3. **Tree Operations**: `mcts/gpu/csr_tree.py` CSR format operations
4. **CUDA Kernels**: `mcts/gpu/unified_kernels.py` GPU acceleration

### Optimization Points
- Wave size configuration (3072 optimal for RTX 3060 Ti)
- Memory pool allocation
- CUDA graph optimization
- Mixed precision computation

## Integration Points

### C++ ↔ Python
- `alphazero_py.so` module provides game states
- Numpy array conversion for neural networks
- Efficient move generation and validation

### CPU ↔ GPU
- Automatic GPU kernel dispatch when available
- CPU fallbacks for all operations
- Unified memory management

### Classical ↔ Quantum
- Modular quantum features via configuration
- < 2x overhead for quantum enhancements
- Feature flags for fine-grained control