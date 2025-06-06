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
- Target: 80k-200k simulations/second
- Vectorized processing: 256-2048 paths simultaneously
- Mixed precision (FP16/FP32) computation support

### Key Innovations (from PRD)
- Wave-based vectorized MCTS processing
- MinHash-based interference for path diversity
- Phase-kicked priors for enhanced exploration
- No virtual loss required due to interference mechanism