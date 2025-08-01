# CPU-Optimized MCTS Implementation

This folder contains the CPU-optimized MCTS implementation achieving 850+ simulations/second.

## Core Components

- **cpu_game_states.py**: Core game state management with dynamic allocation
- **cpu_game_states_wrapper.py**: Wrapper with state recycling for efficiency  
- **cpu_mcts_wrapper.py**: Factory function to create CPU-optimized MCTS instances
- **optimized_wave_search.py**: Wave-based parallel search with evaluation caching
- **vectorized_operations.py**: Vectorized UCB calculations and tree operations

## Build Directory

The `build/` directory contains Cython source files (`.pyx`) and setup script for compiling optimized extensions.

## Usage

```python
from mcts.cpu.cpu_mcts_wrapper import create_cpu_optimized_mcts

# Create CPU-optimized MCTS
mcts = create_cpu_optimized_mcts(config, evaluator, game_interface)

# Run search
policy = mcts.search(state, num_simulations=1000)
```

## Performance

- 850+ simulations/second on modern CPUs
- Evaluation caching reduces redundant neural network calls
- Progressive widening limits tree expansion overhead
- Optimized batch sizes for CPU inference