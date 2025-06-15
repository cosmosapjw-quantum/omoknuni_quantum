# AlphaZero Omoknuni Documentation

## Core Documentation

### Getting Started
- **[AlphaZero Training Guide](alphazero-training-guide.md)** - Complete guide for training AlphaZero models
- **[Project Structure](PROJECT_STRUCTURE.md)** - Overview of codebase organization

### Implementation
- **[Implementation Guide](implementation-guide.md)** - Detailed implementation reference
- **[Performance Optimization](performance-optimization.md)** - GPU optimization and performance tuning

### Quantum Features
- **[Quantum MCTS Guide](quantum-mcts-guide.md)** - Production-ready quantum enhancements
- **[Quantum Theory Foundations](quantum-theory-foundations.md)** - Mathematical foundations

## Quick Links

### Training a Model
```bash
cd python
python -m mcts.neural_networks.unified_training_pipeline \
    --config ../configs/gomoku_classical.yaml
```

### Running Benchmarks
```python
from mcts.core.mcts import MCTS, MCTSConfig

config = MCTSConfig(
    min_wave_size=3072,
    max_wave_size=3072,
    adaptive_wave_sizing=False
)

mcts = MCTS(config, evaluator)
results = mcts.run_benchmark(state)
```

### Enabling Quantum Features
```python
config = MCTSConfig(
    enable_quantum=True,
    quantum_config=QuantumConfig(
        quantum_level="one_loop",
        hbar_eff=0.5
    )
)
```

## Archive

Historical documentation and research materials are preserved in the `archive/` directory:
- `old_guides/` - Previous versions of guides
- `research/` - Research proposals and theoretical work
- `temp_summaries/` - Implementation summaries and fixes