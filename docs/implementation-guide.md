# AlphaZero Omoknuni Implementation Guide

## Overview

This guide covers the implementation details of the AlphaZero Omoknuni project, a high-performance game AI engine achieving 168k+ simulations/second through advanced parallelization and GPU optimization.

## Architecture

### Core Components

1. **C++ Game Engine** (`src/`)
   - High-performance game implementations (Chess, Go, Gomoku)
   - Vectorized operations for batch processing
   - Python bindings via pybind11

2. **Python MCTS** (`python/mcts/`)
   - GPU-accelerated Monte Carlo Tree Search
   - Wave-based parallelization (256-4096 paths)
   - Quantum enhancements (optional)

3. **Neural Networks** (`python/mcts/neural_networks/`)
   - ResNet-based evaluation
   - Mixed precision training
   - Distributed self-play

### Key Classes

#### MCTS Implementation

```python
from mcts.core.mcts import MCTS, MCTSConfig

config = MCTSConfig(
    num_simulations=10000,
    min_wave_size=3072,
    max_wave_size=3072,
    adaptive_wave_sizing=False,  # Critical for performance
    device='cuda'
)

mcts = MCTS(config, evaluator)
```

#### Game Interface

```python
from mcts.core.game_interface import GameInterface, GameType

game = GameInterface(GameType.GOMOKU, board_size=15)
state = game.create_initial_state()
legal_moves = game.get_legal_moves(state)
```

## Performance Optimization

### Wave Parallelization

The key to high performance is wave-based processing:

```python
# Process multiple MCTS paths simultaneously
wave_size = 3072  # Optimal for RTX 3060 Ti

# All operations are vectorized:
# - Selection: Batch UCB calculation
# - Expansion: Parallel node creation
# - Evaluation: Batch neural network
# - Backup: Vectorized value propagation
```

### GPU Optimization

1. **CSR Tree Structure**
   - Compressed sparse row format
   - Contiguous memory layout
   - Batch operations support

2. **GPU Game States**
   - All states resident on GPU
   - Zero-copy tensor operations
   - Efficient state cloning

3. **Memory Pooling**
   - Pre-allocated buffers
   - Zero allocation during search
   - Shared memory pools

### Configuration for Hardware

**RTX 3060 Ti (8GB VRAM)**:
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

**RTX 3090 (24GB VRAM)**:
```python
config = MCTSConfig(
    min_wave_size=4096,
    max_wave_size=4096,
    memory_pool_size_mb=8192,
    max_tree_nodes=2000000
)
```

## Training Pipeline

### Self-Play

```python
from mcts.neural_networks.self_play_module import SelfPlayModule

self_play = SelfPlayModule(
    game_type='gomoku',
    mcts_config=mcts_config,
    num_workers=12,
    device='cuda'
)

# Generate training data
games_data = self_play.run_games(
    model=current_model,
    num_games=1000
)
```

### Neural Network Training

```python
from mcts.neural_networks.nn_model import create_model

model = create_model(
    game_type='gomoku',
    num_res_blocks=10,
    num_filters=128
)

# Training loop with mixed precision
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scaler = torch.cuda.amp.GradScaler()

for batch in dataloader:
    with torch.cuda.amp.autocast():
        policy, value = model(batch['state'])
        loss = compute_loss(policy, value, batch)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Arena Evaluation

```python
from mcts.neural_networks.arena_module import ArenaManager

arena = ArenaManager(config)

# 3-way evaluation
results = arena.compare_models(
    random_evaluator,  # ELO anchor at 0
    best_model,
    current_model
)
```

## Build System

### C++ Compilation

```bash
# Standard build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# With Python bindings
cmake .. -DBUILD_PYTHON_BINDINGS=ON
make -j$(nproc)
```

### Python Installation

```bash
cd python
pip install -e .
```

## Game-Specific Implementation

### Gomoku

- 15x15 board
- 5-in-a-row to win
- No special rules by default

### Chess

- Full FIDE rules
- En passant, castling, promotion
- 50-move rule, threefold repetition

### Go

- 19x19 board (configurable)
- Chinese rules by default
- Ko rule, superko detection

## Debugging and Profiling

### Performance Profiling

```python
# Enable kernel profiling
config.profile_gpu_kernels = True

# Run benchmark
results = mcts.run_benchmark(state, duration=10.0)
print(f"Average: {results['avg_simulations_per_second']:,.0f} sims/sec")
```

### Debug Logging

```python
config.enable_debug_logging = True
logging.basicConfig(level=logging.DEBUG)
```

### Memory Monitoring

```python
stats = mcts.get_statistics()
print(f"GPU Memory: {stats['gpu_memory_mb']:.1f} MB")
print(f"Tree Nodes: {stats['tree_nodes']}")
```

## Best Practices

1. **Always use fixed wave size** for maximum performance
2. **Pre-compile CUDA kernels** before training
3. **Use mixed precision** for 2x speedup
4. **Monitor GPU memory** to avoid OOM
5. **Save checkpoints frequently** during training

## Common Issues

### Low Performance

- Check wave size settings (should be fixed, not adaptive)
- Verify CUDA graphs are enabled
- Ensure mixed precision is active
- Monitor GPU utilization

### Memory Issues

- Reduce max_tree_nodes
- Lower memory_pool_size_mb
- Decrease batch sizes
- Enable gradient accumulation

### Training Instability

- Lower learning rate
- Increase batch size
- Add gradient clipping
- Check for NaN values

## Integration Examples

### Custom Evaluator

```python
from mcts.core.evaluator import Evaluator

class CustomEvaluator(Evaluator):
    def evaluate_batch(self, states):
        # Your neural network here
        return policies, values
```

### Custom Game

```python
from mcts.core.game_interface import IGameImplementation

class CustomGame(IGameImplementation):
    def get_initial_state(self):
        # Return initial game state
        pass
    
    def get_legal_moves(self, state):
        # Return list of legal actions
        pass
    
    def apply_move(self, state, action):
        # Return new state after action
        pass
```

## References

- MCTS implementation based on AlphaZero paper
- Wave parallelization inspired by efficient MCTS research
- Quantum enhancements from theoretical physics principles