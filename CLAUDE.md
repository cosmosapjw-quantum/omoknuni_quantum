# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the AlphaZero (Omoknuni) project - a high-performance game AI engine implementing vectorized MCTS with quantum-inspired enhancements. It combines a C++17 game engine with Python AI components for Chess, Go, and Gomoku.

## Build Commands

```bash
# Standard build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Build with all features (recommended for development)
cmake .. -DBUILD_TESTS=ON -DWITH_TORCH=ON -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# Build without GPU support
cmake .. -DWITH_TORCH=OFF -DWITH_CUDNN=OFF
make -j$(nproc)

# Build Python bindings (required for MCTS)
cmake .. -DBUILD_PYTHON_BINDINGS=ON
make -j$(nproc)
# The Python module will be built as alphazero_py.so
```

## Test Commands

```bash
# Run all tests
cd build
ctest --output-on-failure

# Run specific test suites
./bin/Debug/chess_tests
./bin/Debug/go_tests
./bin/Debug/gomoku_tests
./bin/Debug/attack_defense_tests

# Run tests with detailed output
./bin/Debug/all_tests --gtest_filter=ChessTest.*
```

## Key Architecture

### C++ Game Engine Structure
- **Core Interface**: `IGameState` in `include/core/igamestate.h` - All games implement this interface
- **Game Implementations**: Each game (Chess, Go, Gomoku) has:
  - `*_state.h/cpp`: Game state representation and move execution
  - `*_rules.h/cpp`: Move validation and game-specific rules
  - `*_types.h`: Game-specific type definitions (Chess only)

### GPU Acceleration
- Attack/defense pattern recognition has GPU versions when `WITH_TORCH=ON`
- GPU kernels in `src/utils/gpu_attack_defense_*.cpp`
- CPU fallbacks always available in `attack_defense_module.cpp`

### Python Integration
- Bindings in `src/python/bindings.cpp` expose C++ games to Python
- Provides numpy array conversions for neural network integration
- Build with `BUILD_PYTHON_BINDINGS=ON` (default)

### Performance Goals
- Target: 80k-200k simulations/second ✅ **ACHIEVED: 168k sims/sec**
- Vectorized processing: 256-2048 paths simultaneously (optimal: 3072)
- Mixed precision (FP16/FP32) computation support

### Performance Optimization Tips
- **Critical**: Set `adaptive_wave_sizing=False` and `max_wave_size=3072` for best performance
- Use `MCTS` class from `mcts.core.optimized_mcts`
- Enable CUDA and Triton kernels by compiling with `python compile_kernels.py`
- Recommended config for RTX 3060 Ti (8GB VRAM):
  ```python
  config = MCTSConfig(
      min_wave_size=3072,
      max_wave_size=3072,
      adaptive_wave_sizing=False,  # Critical for performance
      memory_pool_size_mb=2048,
      max_tree_nodes=500000,
      use_mixed_precision=True,
      use_cuda_graphs=True,
      use_tensor_cores=True
  )
  ```

### Key Innovations (from PRD)
- Wave-based vectorized MCTS processing
- MinHash-based interference for path diversity
- Phase-kicked priors for enhanced exploration
- No virtual loss required due to interference mechanism

## Quantum MCTS Implementation

### Overview
The project includes a production-ready quantum-enhanced MCTS that applies QFT principles:
- **Performance**: < 2x overhead compared to classical MCTS
- **Physics**: Full path integral formulation with quantum corrections
- **Implementation**: `python/mcts/quantum/quantum_features.py`

### Quick Usage
```python
from mcts.quantum import create_quantum_mcts

# Create quantum MCTS
quantum_mcts = create_quantum_mcts(enable_quantum=True)

# Apply to selection
ucb_scores = quantum_mcts.apply_quantum_to_selection(
    q_values, visit_counts, priors
)
```

### Documentation
See `docs/quantum-mcts-complete-guide.md` for comprehensive documentation covering:
- Mathematical foundations (QFT formulation)
- Implementation architecture
- Performance optimization
- Integration guide
- API reference

### Key Files
- `mcts/quantum/quantum_features.py` - Main production implementation
- `mcts/quantum/qft_engine.py` - Optimized QFT computations
- `mcts/quantum/path_integral.py` - Path integral formulation
- `docs/quantum-mcts-complete-guide.md` - Complete documentation

## High-Performance MCTS Usage

```python
from mcts.core.mcts import MCTS, MCTSConfig
from mcts.core.game_interface import GameType
from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
import alphazero_py

# Create evaluator
evaluator = ResNetEvaluator(game_type='gomoku', device='cuda')

# Configure for maximum performance
config = MCTSConfig(
    min_wave_size=3072,
    max_wave_size=3072,
    adaptive_wave_sizing=False,  # Critical!
    memory_pool_size_mb=2048,
    max_tree_nodes=500000,
    game_type=GameType.GOMOKU,
    device='cuda'
)

# Create MCTS instance
mcts = MCTS(config, evaluator)
mcts.optimize_for_hardware()

# Run search
game_state = alphazero_py.GomokuState()
policy = mcts.search(game_state, num_simulations=100000)

# Get best move
best_action = mcts.get_best_action(game_state)
```

## Debugging Guidelines

When debugging complex issues in this codebase, follow these comprehensive guidelines:

### General Debugging Approach
- **Engage in thorough and deep thinking** to carry out complex tasks
- **Create a comprehensive to-do list** that outlines incremental fixes and changes
- **Modify code according to test-driven development** for each step
- **Critically review and assess** the current code before entering a detailed debugging phase
- **Reflect deeply** to effectively address complicated issues
- **Prioritize precision above all**, as both correctness and detailed accuracy are vital
- **Take necessary time** to thoroughly contemplate to meet all requirements
- **Actively use chain-of-thought process** to enhance and improve results
- **Consider including detailed debug logging** throughout the codebase
- **Always maintain a critical mindset**
- **After writing each code segment**, perform a 'red-team' review to ensure thorough evaluation
- **Instead of creating new files**, strive to merge and integrate new code snippets into the existing code
- **Actively use pytest** for testing fixes and modifications
- **Ensure final output** does not merely repeat comments made during thought process
- **Always use ~/venv** for the Python virtual environment

### CUDA Multiprocessing Debugging
When debugging CUDA-related multiprocessing issues:

1. **Environment Setup**:
   - Set `CUDA_VISIBLE_DEVICES=''` BEFORE importing torch in worker processes
   - Use multiprocessing `spawn` method: `multiprocessing.set_start_method('spawn', force=True)`

2. **Tensor Serialization**:
   - Convert CUDA tensors to numpy arrays before passing to workers
   - Use `mcts.utils.safe_multiprocessing` utilities for safe serialization
   - Always verify tensors are on CPU before pickling

3. **Worker Process Debugging**:
   - Add early debug prints before any imports
   - Log PID, CUDA state, and environment variables
   - Use `sys.stderr` for immediate output visibility

4. **Common Pitfalls**:
   - CUDA tensors cannot be unpickled in different processes
   - Setting CUDA environment after torch import is too late
   - Dataclasses may contain hidden CUDA references

### Example Debug Pattern
```python
# In worker function
def worker_function(args):
    import os
    import sys
    
    # Debug output BEFORE imports
    print(f"[WORKER] PID: {os.getpid()}", file=sys.stderr)
    
    # Disable CUDA BEFORE torch import
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Now safe to import
    import torch
    print(f"[WORKER] CUDA available: {torch.cuda.is_available()}", file=sys.stderr)
    
    # ... rest of worker code
```