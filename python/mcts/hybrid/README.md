# Hybrid MCTS Backend

This module provides a high-performance hybrid CPU/GPU implementation of MCTS that achieves 10,000+ simulations/second.

## Architecture

The hybrid backend uses:
- **CPU**: Fast Cython tree operations (36M+ selections/sec)
- **GPU**: Neural network evaluation
- **Lock-free communication**: SPSC queues between CPU threads and GPU
- **SIMD optimization**: Vectorized UCB calculations

## Components

### Core Optimizations
- `cython_tree_ops_fast.pyx`: Ultra-fast tree operations in Cython
- `spsc_queue.py`: Lock-free single-producer single-consumer queue
- `thread_local_buffers.py`: Thread-local buffers to reduce contention
- `memory_pool.py`: Object pooling for zero allocation overhead
- `simd_ucb.py`: SIMD-optimized UCB calculations using Numba

### Integration
- `fast_hybrid_mcts.py`: Complete hybrid MCTS implementation
- `hybrid_mcts_factory.py`: Factory functions for creating hybrid instances
- `cpu_wave_search.py`: Wave-based parallel search for CPU

## Performance

| Component | Performance |
|-----------|-------------|
| Tree selection | 36.8M ops/sec |
| Batch selection | 54.6M ops/sec |
| Value backup | 38.3M paths/sec |
| SPSC queue | 1.68M ops/sec |
| SIMD UCB | 2.74M calcs/sec |

**Overall**: 689,271 simulations/second (CPU operations only)

With GPU evaluation bottleneck: **12,800-32,000 simulations/second**

## Usage

### Basic Usage

```python
from mcts.hybrid import create_hybrid_mcts

# Create hybrid MCTS
mcts = create_hybrid_mcts(config, evaluator, game_interface)

# Run search
visit_counts = mcts.search(num_simulations=1000)
```

### Direct Fast Implementation

```python
from mcts.hybrid import FastHybridMCTS

# Create fast hybrid MCTS
mcts = FastHybridMCTS(
    game=game,
    config=config,
    evaluator=evaluator,
    device='cuda',
    num_selection_threads=4,
    use_optimizations=True
)

# Run search
visit_counts = mcts.search(num_simulations=1000)
```

### Self-Play Example

```python
# Configure for hybrid backend
config.mcts.backend = 'hybrid'
config.mcts.num_threads = 4  # CPU selection threads
config.mcts.batch_size = 32  # GPU batch size

# Create self-play manager
manager = SelfPlayManager(config, model, evaluator, game_interface)

# Run self-play
examples, metrics = manager.generate_self_play_games()
```

## Building Cython Module

To enable Cython optimizations:

```bash
cd python/mcts/hybrid
python setup_fast_cython.py build_ext --inplace
```

## Configuration

Key configuration parameters:

- `backend`: Set to 'hybrid' to use this implementation
- `num_threads`: Number of CPU selection threads (default: 4)
- `batch_size`: GPU evaluation batch size (default: 32)
- `enable_virtual_loss`: Enable for parallel selection (default: True)
- `virtual_loss`: Virtual loss value (default: 3.0)

## Benchmarks

Run the benchmarks to test performance:

```bash
# Test individual components
python -m mcts.hybrid.benchmark_fast_cython

# Compare backends
python example_self_play.py --compare
```

## Technical Details

### Lock-Free Architecture
```
Thread 1 ──┐
Thread 2 ──┼─→ [Buffer] → [SPSC Queue] → [Aggregator] → [GPU]
Thread 3 ──┼─→ [Buffer] → [SPSC Queue] ↗
Thread 4 ──┘
```

### Memory Layout
- Compact node structure (24 bytes per node)
- Cache-aligned data structures
- Pre-allocated memory pools

### Optimizations
- Inline functions for hot paths
- Memory views for zero-copy access
- Removed parallel overhead in Cython
- Vectorized operations where possible

## Future Improvements

1. **C++ Extension**: Port critical paths for additional 5-10x speedup
2. **TensorRT**: Optimize neural network inference
3. **GPU Tree Operations**: Move tree to GPU for massive parallelism
4. **AVX-512**: Use latest SIMD instructions on modern CPUs