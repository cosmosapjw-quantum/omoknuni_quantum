# AlphaZero Training Guide

This guide provides comprehensive instructions for training AlphaZero models with the enhanced training pipeline, including arena evaluation, ELO tracking, and quantum MCTS features.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration System](#configuration-system)
4. [Training Pipeline](#training-pipeline)
5. [Arena and ELO System](#arena-and-elo-system)
6. [Quantum MCTS Levels](#quantum-mcts-levels)
7. [Game-Specific Configurations](#game-specific-configurations)
8. [Advanced Topics](#advanced-topics)
9. [Troubleshooting](#troubleshooting)

## Overview

The enhanced AlphaZero training pipeline includes:

- **Self-play data generation** with parallel workers
- **Neural network training** with mixed precision and gradient accumulation
- **Arena evaluation system** to compare models
- **ELO rating tracking** with random policy as anchor (ELO 0)
- **Quantum MCTS** with three levels: classical, tree-level, and one-loop
- **YAML configuration** for all parameters

### Key Features

1. **Automatic Model Evaluation**: New models are evaluated against the current best model
2. **55% Win Rate Threshold**: New models must win >55% to be accepted as the new best
3. **ELO Tracking**: All models are rated using ELO system with random policy at ELO 0
4. **Quantum Enhancement**: Optional quantum features for enhanced exploration
5. **Full Configurability**: Every parameter can be adjusted via YAML

## Quick Start

### 1. Basic Training Command

```bash
# Train Gomoku with default settings
python -m mcts.neural_networks.enhanced_training_pipeline --game gomoku

# Train with custom config
python -m mcts.neural_networks.enhanced_training_pipeline --config configs/gomoku_quantum.yaml

# Resume training from checkpoint
python -m mcts.neural_networks.enhanced_training_pipeline --config configs/gomoku_quantum.yaml --resume checkpoints/checkpoint_iter_100.pt
```

### 2. Minimal Configuration Example

```yaml
# configs/gomoku_minimal.yaml
game:
  game_type: gomoku
  board_size: 15

mcts:
  num_simulations: 800
  quantum_level: classical  # Start with classical

training:
  num_games_per_iteration: 100
  num_workers: 4
  batch_size: 512

arena:
  num_games: 40
  win_threshold: 0.55

num_iterations: 1000
```

## Configuration System

The configuration system uses a hierarchical structure with five main sections:

### Configuration Sections

1. **game**: Game-specific settings
2. **mcts**: MCTS algorithm parameters
3. **network**: Neural network architecture
4. **training**: Training loop settings
5. **arena**: Model evaluation settings

### Complete Configuration Reference

```yaml
# Complete configuration with all parameters
experiment_name: alphazero_experiment
seed: 42
log_level: INFO
num_iterations: 1000

game:
  game_type: gomoku  # chess, go, gomoku
  board_size: 15     # For Go and Gomoku
  
  # Go-specific
  go_komi: 7.5
  go_rules: chinese  # chinese, japanese, tromp_taylor
  go_superko: true
  
  # Gomoku-specific
  gomoku_use_renju: false
  gomoku_use_omok: false
  gomoku_use_pro_long_opening: false
  
  # Chess-specific
  chess_960: false
  chess_starting_fen: null

mcts:
  # Core parameters
  num_simulations: 800
  c_puct: 1.0
  dirichlet_alpha: 0.3
  dirichlet_epsilon: 0.25
  temperature: 1.0
  temperature_threshold: 30
  temperature_final: 0.1
  
  # Performance
  min_wave_size: 256
  max_wave_size: 3072
  adaptive_wave_sizing: false  # Set false for best performance
  batch_size: 256
  virtual_loss: 3.0
  
  # Memory
  memory_pool_size_mb: 2048
  max_tree_nodes: 500000
  tree_reuse: true
  tree_reuse_fraction: 0.5
  
  # GPU optimization
  use_cuda_graphs: true
  use_mixed_precision: true
  use_tensor_cores: true
  compile_mode: reduce-overhead
  
  # Quantum features
  quantum_level: classical  # classical, tree_level, one_loop
  enable_quantum: false
  
  # Quantum physics parameters
  quantum_coupling: 0.1
  quantum_temperature: 1.0
  decoherence_rate: 0.01
  measurement_noise: 0.0
  
  # Path integral
  path_integral_steps: 10
  path_integral_beta: 1.0
  use_wick_rotation: true
  
  # Interference
  interference_alpha: 0.05
  interference_method: minhash  # minhash, phase_kick, cosine
  minhash_size: 64
  phase_kick_strength: 0.1
  
  # Device
  device: cuda
  num_threads: 4

network:
  # Architecture
  model_type: resnet  # resnet, simple, lightweight
  input_channels: 20
  num_res_blocks: 10
  num_filters: 256
  value_head_hidden_size: 256
  policy_head_filters: 2
  
  # Regularization
  dropout_rate: 0.1
  batch_norm: true
  batch_norm_momentum: 0.997
  l2_regularization: 0.0001
  
  # Activation
  activation: relu  # relu, leaky_relu, elu, gelu
  leaky_relu_alpha: 0.01
  
  # Initialization
  weight_init: he_normal  # he_normal, xavier_normal, orthogonal
  bias_init: 0.0

training:
  # Basic
  batch_size: 512
  learning_rate: 0.01
  learning_rate_schedule: step  # step, cosine, exponential, none
  lr_decay_steps: 50
  lr_decay_rate: 0.1
  min_learning_rate: 0.00001
  
  # Optimization
  optimizer: adam  # adam, sgd, adamw, lamb
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 0.00000001
  sgd_momentum: 0.9
  sgd_nesterov: true
  weight_decay: 0.0001
  
  # Gradient
  gradient_accumulation_steps: 1
  max_grad_norm: 5.0
  gradient_clip_value: null
  
  # Training loop
  num_epochs: 10
  checkpoint_interval: 100
  validation_interval: 10
  early_stopping_patience: 50
  early_stopping_min_delta: 0.0001
  
  # Self-play
  num_games_per_iteration: 100
  num_workers: 4
  games_per_worker: 25
  max_moves_per_game: 500
  resign_threshold: -0.95
  resign_check_moves: 10
  
  # Data
  window_size: 500000
  sample_weight_by_game_length: true
  augment_data: true
  shuffle_buffer_size: 10000
  
  # Mixed precision
  mixed_precision: true
  amp_opt_level: O1
  loss_scale: dynamic
  static_loss_scale: 1.0
  
  # Evaluation
  eval_temperature: 0.1
  eval_num_games: 20
  
  # Paths
  save_dir: checkpoints
  tensorboard_dir: runs
  data_dir: self_play_data

arena:
  # Battle settings
  num_games: 40
  num_workers: 4
  games_per_worker: 10
  win_threshold: 0.55
  statistical_significance: true
  confidence_level: 0.95
  
  # Game settings
  temperature: 0.1
  mcts_simulations: 400
  c_puct: 1.0
  max_moves: 500
  time_limit_seconds: null
  randomize_start_player: true  # Random vs alternating
  
  # ELO settings
  elo_k_factor: 32.0
  elo_initial_rating: 1500.0
  elo_anchor_rating: 0.0  # Random policy anchor
  update_elo: true
  
  # Random policy evaluation
  eval_vs_random_interval: 10
  eval_vs_random_games: 20
  min_win_rate_vs_random: 0.95
  
  # Tournament
  tournament_rounds: 1
  tournament_games_per_pair: 10
  
  # Data saving
  save_game_records: false
  save_arena_logs: true
  arena_log_dir: arena_logs
  elo_save_path: elo_ratings.json
```

## Training Pipeline

### Pipeline Stages

1. **Self-Play Generation**
   - Uses current best model (or latest if no best yet)
   - Parallel workers generate games
   - Temperature-based exploration (high early, low late in game)
   - Dirichlet noise at root for exploration

2. **Neural Network Training**
   - Trains on replay buffer
   - Mixed precision training for speed
   - Learning rate scheduling
   - Gradient clipping and accumulation

3. **Model Evaluation**
   - New model plays against current best
   - Must win >55% to be accepted
   - Also evaluated against random policy
   - ELO ratings updated

4. **Best Model Update**
   - If new model wins, it becomes the new best
   - Used for future self-play generation

### Training Flow

```
Initialize → Self-Play → Train → Evaluate → Update Best → Repeat
     ↑                                              ↓
     └──────────────────────────────────────────────┘
```

## Arena and ELO System

### Arena Evaluation

The arena system evaluates models through head-to-head matches:

1. **Model vs Best**: New model must beat current best by win_threshold (default 55%)
2. **Model vs Random**: Sanity check - should win >95% against random policy
3. **Tournament Mode**: Compare multiple models in round-robin format
4. **Fair Starting Position**: Games randomly assign who plays first (configurable)

### ELO Rating System

- **Anchor Point**: Random policy fixed at ELO 0
- **Initial Rating**: New models start at 1500 (configurable)
- **K-Factor**: Controls rating volatility (default 32)
- **Updates**: After every game, both players' ratings are updated

### Example Arena Configuration

```yaml
arena:
  num_games: 100  # More games for statistical significance
  win_threshold: 0.55
  
  # Stronger evaluation settings
  temperature: 0.05  # Very low for deterministic play
  mcts_simulations: 1600  # More simulations for stronger play
  randomize_start_player: true  # Random assignment for fairness
  
  # ELO tracking
  elo_k_factor: 16  # Lower for more stable ratings
  eval_vs_random_interval: 5  # Check vs random every 5 iterations
```

## Quantum MCTS Levels

The quantum MCTS system offers three levels of quantum corrections:

### 1. Classical (No Quantum)

```yaml
mcts:
  quantum_level: classical
  enable_quantum: false
```

Standard MCTS with no quantum features. Best for:
- Initial training
- Baseline comparisons
- Maximum speed

### 2. Tree-Level Quantum

```yaml
mcts:
  quantum_level: tree_level
  enable_quantum: true
  quantum_coupling: 0.1
  quantum_temperature: 1.0
  interference_method: minhash
```

Adds quantum uncertainty and interference at tree level:
- Quantum uncertainty bonus for exploration
- MinHash interference between paths
- Phase diversity for better exploration

### 3. One-Loop Quantum

```yaml
mcts:
  quantum_level: one_loop
  enable_quantum: true
  quantum_coupling: 0.15
  decoherence_rate: 0.01
  path_integral_steps: 20
  use_wick_rotation: true
```

Full quantum field theory corrections:
- Tree-level features plus:
- One-loop vacuum fluctuations
- Path integral formulation
- Decoherence effects
- Self-energy and vertex corrections

### Quantum Parameter Guidelines

| Parameter | Classical | Tree-Level | One-Loop |
|-----------|-----------|------------|----------|
| quantum_coupling | N/A | 0.05-0.15 | 0.1-0.2 |
| quantum_temperature | N/A | 0.5-2.0 | 1.0-3.0 |
| decoherence_rate | N/A | 0.0 | 0.01-0.05 |
| interference_alpha | N/A | 0.02-0.1 | 0.05-0.15 |

## Game-Specific Configurations

### Gomoku Configuration

```yaml
# configs/gomoku_optimized.yaml
experiment_name: gomoku_alphazero_optimized
num_iterations: 500

game:
  game_type: gomoku
  board_size: 15
  gomoku_use_renju: false  # Standard rules

mcts:
  num_simulations: 800
  c_puct: 1.0
  dirichlet_alpha: 0.15  # Lower for Gomoku
  
  # Optimized for RTX 3060 Ti
  min_wave_size: 3072
  max_wave_size: 3072
  adaptive_wave_sizing: false
  
  # Tree-level quantum for exploration
  quantum_level: tree_level
  enable_quantum: true
  quantum_coupling: 0.1

network:
  num_res_blocks: 10  # Smaller network for Gomoku
  num_filters: 128

training:
  batch_size: 512
  learning_rate: 0.01
  num_games_per_iteration: 100
  num_workers: 4

arena:
  num_games: 40
  win_threshold: 0.55
```

### Go Configuration

```yaml
# configs/go_19x19.yaml
experiment_name: go_alphazero_19x19
num_iterations: 1000

game:
  game_type: go
  board_size: 19
  go_komi: 7.5
  go_rules: chinese
  go_superko: true

mcts:
  num_simulations: 1600  # More for Go complexity
  c_puct: 1.0
  dirichlet_alpha: 0.03  # Much lower for 19x19
  
  # Large batches for Go
  min_wave_size: 2048
  max_wave_size: 4096
  
  # One-loop quantum for complex game
  quantum_level: one_loop
  enable_quantum: true
  quantum_coupling: 0.15
  decoherence_rate: 0.02

network:
  num_res_blocks: 20  # Deeper for Go
  num_filters: 256

training:
  batch_size: 256  # Smaller due to larger board
  learning_rate: 0.01
  num_games_per_iteration: 200
  num_workers: 8  # More workers for Go

arena:
  num_games: 100  # More games for statistical significance
  mcts_simulations: 800  # Balance speed vs strength
```

### Chess Configuration

```yaml
# configs/chess_classical.yaml
experiment_name: chess_alphazero_classical
num_iterations: 800

game:
  game_type: chess
  chess_960: false

mcts:
  num_simulations: 800
  c_puct: 1.0
  dirichlet_alpha: 0.3
  
  # Classical MCTS for chess (established theory)
  quantum_level: classical
  enable_quantum: false
  
  # Standard performance settings
  min_wave_size: 1024
  max_wave_size: 2048

network:
  num_res_blocks: 20
  num_filters: 256

training:
  batch_size: 512
  learning_rate: 0.01
  num_games_per_iteration: 150
  num_workers: 6

arena:
  num_games: 60
  win_threshold: 0.55
  mcts_simulations: 400
```

## Advanced Topics

### Performance Optimization

1. **Wave Size Tuning**
   ```yaml
   mcts:
     min_wave_size: 3072
     max_wave_size: 3072
     adaptive_wave_sizing: false  # Critical for performance
   ```

2. **GPU Optimization**
   ```yaml
   mcts:
     use_cuda_graphs: true
     use_mixed_precision: true
     use_tensor_cores: true
     compile_mode: reduce-overhead
   ```

3. **Memory Management**
   ```yaml
   mcts:
     memory_pool_size_mb: 2048
     max_tree_nodes: 500000
     tree_reuse: true
     tree_reuse_fraction: 0.5
   ```

### Learning Rate Schedules

1. **Step Decay** (Default)
   ```yaml
   training:
     learning_rate_schedule: step
     lr_decay_steps: 50
     lr_decay_rate: 0.1
   ```

2. **Cosine Annealing**
   ```yaml
   training:
     learning_rate_schedule: cosine
     lr_decay_steps: 100  # Period
     min_learning_rate: 0.00001
   ```

3. **Exponential Decay**
   ```yaml
   training:
     learning_rate_schedule: exponential
     lr_decay_rate: 0.95
   ```

### Quantum Feature Tuning

1. **Start Conservative**
   - Begin with classical or tree-level
   - Low coupling strength (0.05-0.1)
   - Minimal decoherence

2. **Gradual Increase**
   - Increase coupling as training progresses
   - Add one-loop corrections for final refinement
   - Monitor overhead vs exploration benefit

3. **Game-Specific Tuning**
   - Tactical games (Chess): Lower quantum effects
   - Strategic games (Go): Higher quantum effects
   - Pattern games (Gomoku): Medium quantum effects

### Multi-Stage Training

```yaml
# Stage 1: Bootstrap with classical
mcts:
  quantum_level: classical
  num_simulations: 400

# Stage 2: Add tree-level quantum
mcts:
  quantum_level: tree_level
  quantum_coupling: 0.1
  num_simulations: 800

# Stage 3: Refine with one-loop
mcts:
  quantum_level: one_loop
  quantum_coupling: 0.15
  num_simulations: 1600
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `max_wave_size`
   - Lower `memory_pool_size_mb`
   - Decrease `batch_size`

2. **Slow Training**
   - Disable `adaptive_wave_sizing`
   - Set fixed `wave_size` (3072 optimal)
   - Enable `use_cuda_graphs`

3. **Model Not Improving**
   - Check learning rate (too high/low)
   - Increase `num_games_per_iteration`
   - Verify arena evaluation is working

4. **Quantum Overhead Too High**
   - Increase `min_wave_size`
   - Use `tree_level` instead of `one_loop`
   - Enable `fast_mode`

### Debugging Configuration

```yaml
# Debug configuration
log_level: DEBUG
experiment_name: debug_run

training:
  num_games_per_iteration: 10
  num_workers: 1
  checkpoint_interval: 1

arena:
  num_games: 4
  num_workers: 1
```

### Monitoring Training

1. **Tensorboard**
   ```bash
   tensorboard --logdir runs/
   ```

2. **ELO Tracking**
   - Check `elo_ratings.json`
   - Monitor rating progression
   - Ensure steady improvement

3. **Arena Logs**
   - Review game outcomes
   - Check win rates
   - Verify fair matches

## Best Practices

1. **Start Simple**
   - Use classical MCTS initially
   - Small network (10 blocks, 128 filters)
   - Moderate simulations (400-800)

2. **Scale Gradually**
   - Increase network size
   - Add quantum features
   - Increase simulations

3. **Monitor Everything**
   - ELO ratings
   - Win rates vs random
   - Training loss curves
   - Arena acceptance rate

4. **Save Configurations**
   - Version control configs
   - Document changes
   - Keep experiment logs

5. **Parallel Experiments**
   - Test different quantum levels
   - Compare architectures
   - Ablation studies

## Conclusion

The enhanced AlphaZero training pipeline provides a complete system for training strong game-playing agents with:

- Automatic model evaluation and selection
- ELO-based performance tracking
- Quantum-enhanced exploration
- Full configurability

Start with the provided templates and gradually tune parameters for your specific use case. Monitor training carefully and adjust based on results.

For additional help, see the example configurations in the `configs/` directory or refer to the source code documentation.