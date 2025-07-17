# MCTS Physics Analysis Framework

Comprehensive physics analysis framework that extracts and validates both statistical mechanics and quantum phenomena from MCTS self-play data using authentic measurements.

## Overview

The complete analysis system combines:

1. **Statistical Mechanics Modules**
   - **Thermodynamics**: Jarzynski equality, Crooks theorem, entropy production
   - **Critical Phenomena**: Phase transitions, finite-size scaling, critical exponents
   - **Fluctuation-Dissipation**: FDT validation, Onsager reciprocity, transport coefficients

2. **Quantum Phenomena Modules**
   - **Decoherence**: Coherence evolution, entropy growth, decoherence times
   - **Tunneling**: Barrier detection, tunneling events, WKB analysis
   - **Entanglement**: Von Neumann entropy, mutual information, concurrence

3. **Authentic Physics Extraction**
   - Temperature measured from visit distributions (NOT calculated as 1/√N)
   - Scaling laws discovered from data (NOT assumed)
   - All physics extracted from real MCTS statistics

## Core Components

### ensemble_analyzer_complete.py
Integrates all physics modules for comprehensive analysis:
- Statistical mechanics validation
- Quantum phenomena detection
- Cross-module validation
- Authentic measurements (no predetermined formulas)

### authentic_physics_extractor.py
Extracts physics directly from MCTS data:
- Temperature from visit distributions (NOT 1/√N)
- Scaling relations discovered from data
- All measurements include uncertainties

### auto_generator.py
Manages the complete analysis pipeline:
- MCTS self-play data generation
- Parameter sweeps for phase diagrams
- Resource management and progress tracking

## Usage

### Quick Start (Recommended)
The easiest way to run the analysis is using the provided shell script that handles all environment setup:

```bash
cd ~/omoknuni_quantum
./run_physics_analysis.sh --preset quick
```

This script automatically:
- Sets up the library paths for C++ modules
- Activates the virtual environment
- Runs the physics analysis

### Manual Setup
If you prefer to set up the environment manually:

```bash
cd ~/omoknuni_quantum
source setup_environment.sh  # Sets up library paths and activates venv
python run_mcts_physics_analysis.py --preset quick
```

### Alternative: Export Library Path Manually
```bash
cd ~/omoknuni_quantum
export LD_LIBRARY_PATH="${PWD}/python:${PWD}/build_cpp/lib/Release:${LD_LIBRARY_PATH}"
source ~/venv/bin/activate
python run_mcts_physics_analysis.py --preset standard
```

### Configuration Options

#### Quick Analysis (10 games)
```bash
python run_mcts_physics_analysis.py --preset quick
```

#### Standard Analysis (50 games)
```bash
python run_mcts_physics_analysis.py --preset standard
```

#### Comprehensive Analysis (100 games, all modules)
```bash
python run_mcts_physics_analysis.py --preset comprehensive
```

#### Custom Configuration
```bash
python run_mcts_physics_analysis.py \
    --games 200 \
    --sims 10000 \
    --output ./my_analysis \
    --disable-modules tunneling entanglement
```

## Output Structure

```
complete_mcts_physics/
├── dynamics_data/           # Raw MCTS dynamics
│   ├── dynamics_0000.npz
│   └── ...
├── complete_physics/        # Analysis results
│   ├── complete_physics_results.json
│   ├── complete_physics_summary.png
│   └── raw_measurements.json
└── checkpoint.json          # Progress checkpoint
```

## Key Features

### Cross-Module Validation
- Compares authentic temperature with FDT temperature
- Validates Jarzynski equality with measured work values
- Checks Onsager reciprocity in transport matrices
- Ensures consistency across different physics modules

### Comprehensive Visualizations
The system generates a complete summary plot showing:
- Row 1: Authentic measurements (temperature, scaling, quality)
- Row 2: Statistical mechanics (thermodynamics, critical, FDT)
- Row 3: Quantum phenomena (decoherence, tunneling, entanglement)
- Row 4: Cross-validations and overall summary

### No Predetermined Formulas
- Temperature is MEASURED by fitting π(a) ∝ exp(β·Q(a))
- Scaling laws are DISCOVERED by testing multiple models
- Critical exponents are EXTRACTED from finite-size scaling
- All physics comes from REAL MCTS data

## Module Integration

The complete analyzer seamlessly integrates all existing validated modules:

```python
# Statistical mechanics
from ..phenomena import (
    ThermodynamicsAnalyzer,
    CriticalPhenomenaAnalyzer,
    FluctuationDissipationAnalyzer
)

# Quantum phenomena
from ..phenomena import (
    DecoherenceAnalyzer,
    TunnelingDetector,
    EntanglementAnalyzer
)

# Authentic measurements
from .authentic_physics_extractor import AuthenticPhysicsExtractor
```

## Benefits

1. **Complete Coverage**: Uses ALL validated physics modules
2. **Cross-Validation**: Ensures consistency between different approaches
3. **Authentic Results**: No predetermined formulas or assumptions
4. **Publication Ready**: Comprehensive analysis suitable for research
5. **Flexible Configuration**: Enable/disable specific modules as needed

## Example Results

A typical analysis might discover:
- Temperature scaling: β ∝ N^0.48 (close to √N but measured, not assumed)
- Critical exponents: β/ν = 0.125, γ/ν = 1.75 (2D Ising universality class)
- Jarzynski equality: Satisfied with <exp(-βW)> = exp(-βΔF)
- Decoherence time: τ_D = 15.3 ± 2.1 moves
- Tunneling events: 23 detected with average barrier height 0.35

All results include proper error estimates and statistical validation.