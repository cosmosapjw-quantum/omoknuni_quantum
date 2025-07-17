# MCTS Physics Analysis

## Running the Analysis

The main entry point for physics analysis is:

```bash
./run_physics_analysis.sh [OPTIONS]
```

### Quick Examples

```bash
# Quick test (10 games, 1000 simulations)
./run_physics_analysis.sh --preset quick

# Standard analysis (50 games, 5000 simulations)
./run_physics_analysis.sh --preset standard

# Overnight analysis (1000 games, 5000 simulations)
./run_physics_analysis.sh --preset overnight

# Custom parameters
./run_physics_analysis.sh --preset quick --games 20 --sims 2000
```

### Presets

- **quick**: Fast test run (10 games, 1000 sims) - ~3 minutes
- **standard**: Standard analysis (50 games, 5000 sims) - ~30 minutes  
- **comprehensive**: Detailed analysis (100 games, 5000 sims) - ~1 hour
- **deep**: Deep analysis with parameter sweeps (200 games, 10000 sims) - ~2 hours
- **overnight**: Full overnight analysis (1000 games, 5000 sims) - ~8 hours

### Output

Results are saved to timestamped directories:
```
physics_analysis_<preset>_<timestamp>/
├── analysis_log.txt           # Full analysis log
├── complete_ensemble_analysis_results.json  # Main results
├── game_data/                 # Raw game data
├── plots/                     # Generated visualizations
└── reports/                   # Analysis reports
```

## What It Analyzes

The physics analysis extracts quantum and statistical mechanics properties from MCTS:

### Statistical Mechanics
- **Thermodynamics**: Temperature, energy, free energy
- **Critical Phenomena**: Phase transitions, scaling laws
- **Fluctuation-Dissipation**: Jarzynski equality, Sagawa-Ueda theorem

### Quantum Phenomena  
- **Decoherence**: Quantum→classical transition
- **Entanglement**: Multi-partite correlations
- **Tunneling**: Barrier penetration events

### Information Theory
- **Entropy Production**: Information gain during search
- **Mutual Information**: Past-future correlations
- **Quantum Darwinism**: Emergence of classical objectivity

## Architecture

```
run_physics_analysis.sh
└── run_mcts_physics_analysis.py
    └── python/mcts/quantum/analysis/run_complete_physics_analysis.py
        ├── auto_generator.py          # MCTS data generation
        ├── ensemble_analyzer_complete.py  # Ensemble analysis
        └── authentic_physics_extractor.py # Physics extraction
```

## Requirements

- Python 3.8+
- PyTorch with CUDA support (recommended)
- 8GB+ RAM
- GPU with 6GB+ VRAM (optional but recommended)

## Advanced Usage

### Parameter Sweeps

```bash
# c_puct parameter sweep
./run_physics_analysis.sh --preset standard --c-puct-sweep --c-puct-range 0.5 3.0 --c-puct-steps 10
```

### Different Game Types

```bash
# Analyze Go instead of Gomoku
./run_physics_analysis.sh --preset standard --game-type go
```

### Different Evaluators

```bash
# Use random evaluator (faster, for testing)
./run_physics_analysis.sh --preset quick --evaluator-type random
```

## Troubleshooting

### Out of Memory
- Reduce `--sims` parameter
- Use `--evaluator-type random` for testing
- Ensure CUDA is available for GPU acceleration

### Import Errors
- Always run via `./run_physics_analysis.sh` (not the Python script directly)
- The shell script sets up the proper environment

### Long Runtime
- Use smaller presets for testing
- Run overnight analysis in screen/tmux session
- Check estimated runtime before starting

## Documentation

Detailed documentation is in `docs/quantum_physics/`:
- `PHYSICS_ANALYSIS_USER_GUIDE.md` - Comprehensive user guide
- `QUANTUM_MCTS_PROJECT_OVERVIEW.md` - Project overview
- `OVERNIGHT_ANALYSIS_README.md` - Overnight analysis details