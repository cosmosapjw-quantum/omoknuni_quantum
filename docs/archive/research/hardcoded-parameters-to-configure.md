# Hardcoded Parameters to Make Configurable

This document lists all hardcoded parameters found in the enhanced training pipeline and related files that should be made configurable through the config system.

## 1. Enhanced Training Pipeline (`enhanced_training_pipeline.py`)

### Currently Hardcoded:
- **Line 356**: `num_workers=4` - DataLoader workers for training
- **Line 357**: `pin_memory=True` - DataLoader pin memory setting
- **Line 399**: `num_res_blocks=10, num_filters=256` - ResNetEvaluator hardcoded architecture in arena worker
- **Line 410**: `num_res_blocks=10, num_filters=256` - ResNetEvaluator hardcoded architecture in arena worker (repeated)
- **Line 644**: `step = max(1, len(checkpoint_paths) // 10)` - Tournament model selection step size

### Should Add to Config:
```yaml
training:
  dataloader_num_workers: 4
  dataloader_pin_memory: true
  
arena:
  evaluator_num_res_blocks: 10  # Should match network config
  evaluator_num_filters: 256     # Should match network config
  tournament_model_step: 10      # Take every N-th model for tournament
```

## 2. Arena System (`arena.py`)

### Currently Hardcoded:
- **Line 48**: `temperature: float = 0.1` - Low temperature for evaluation
- **Line 50**: `c_puct: float = 1.0` - PUCT exploration constant
- **Line 100**: `rating: float = 1500.0` - Starting ELO
- **Line 132**: `k_factor: float = 32.0` - K-factor for ELO updates
- **Line 133**: `initial_rating: float = 1500.0` - Initial rating for new models
- **Line 134**: `anchor_rating: float = 0.0` - Fixed rating for random policy anchor
- **Line 160**: `-1e9` - Illegal move masking value in policy logits
- **Line 93**: `conv filters = 32` - PolicyHead conv filters
- **Line 128**: `conv filters = 1` - ValueHead conv filters

### Should Add to Config:
```yaml
arena:
  illegal_move_penalty: -1e9     # Value for masking illegal moves
  
network:
  policy_head_filters: 32        # Conv filters for policy head
  value_head_filters: 1          # Conv filters for value head
```

## 3. Neural Network Model (`nn_model.py`)

### Currently Hardcoded:
- **Line 31**: `input_channels: int = 20` - Standard AlphaZero encoding
- **Line 35**: `num_res_blocks: int = 19` - Number of residual blocks
- **Line 36**: `num_filters: int = 256` - Number of filters in conv layers
- **Line 37**: `value_head_hidden_size: int = 256` - Hidden layer size for value head
- **Line 40**: `dropout_rate: float = 0.0` - Dropout rate
- **Line 93**: `32` filters in PolicyHead conv layer
- **Line 132**: `1` filter in ValueHead conv layer

### Already in Config but Worth Noting:
These are configurable but have hardcoded defaults that may differ from config defaults.

## 4. Training Pipeline (`training_pipeline.py`)

### Currently Hardcoded:
- **Line 61**: `batch_size: int = 512` - Default batch size
- **Line 62**: `learning_rate: float = 0.01` - Default learning rate
- **Line 63**: `weight_decay: float = 1e-4` - L2 regularization
- **Line 66**: `num_epochs: int = 10` - Epochs per training iteration
- **Line 71**: `dirichlet_alpha: float = 0.3` - Dirichlet noise parameter
- **Line 72**: `dirichlet_epsilon: float = 0.25` - Weight of noise at root
- **Line 74**: `c_puct: float = 1.0` - PUCT exploration constant
- **Line 79**: `max_grad_norm: float = 5.0` - Maximum gradient norm for clipping
- **Line 103**: `max_size: int = 500000` - ReplayBuffer default max size

### Should Add to Config:
Most of these are already configurable, but the ReplayBuffer max_size is initialized separately.

## 5. Wave MCTS (`wave_mcts.py`)

### Currently Hardcoded:
- **Line 122**: `c_puct=1.414` - UCB selection c_puct value (different from default 1.0!)
- **Line 34**: `target_sims_per_second: int = 100000` - Performance target
- **Line 35**: `target_gpu_utilization: float = 0.95` - GPU utilization target
- **Line 44**: `pool_size_mb: int = 1024` - Memory pool size (1GB)
- **Line 183**: `actions = list(range(min(10, 225)))` - Placeholder expansion limiting to 10 moves

### Should Add to Config:
```yaml
mcts:
  wave_c_puct: 1.414              # Different from regular c_puct!
  target_sims_per_second: 100000
  target_gpu_utilization: 0.95
  memory_pool_size_mb: 1024       # Already exists but different default
  max_expansion_actions: 10       # Limit actions during expansion
```

## 6. Quantum CUDA Kernels (`quantum_cuda_kernels.py`)

### Currently Hardcoded:
- **Line 98**: `if similarity > 0.1:` - Threshold for interference
- **Line 102**: `interference_sum += similarity * 0.1` - Constructive interference factor
- **Line 104**: `interference_sum -= similarity * 0.05` - Destructive interference factor
- **Line 146**: `epistemic_var = 1.0 / (visits + 1.0)` - Epistemic uncertainty formula
- **Line 147**: `aleatoric_var = temperature * tl.sqrt(epistemic_var)` - Aleatoric uncertainty formula

### Should Add to Config:
```yaml
mcts:
  interference_threshold: 0.1
  constructive_interference_factor: 0.1
  destructive_interference_factor: 0.05
  epistemic_variance_base: 1.0    # Base for epistemic uncertainty
  aleatoric_variance_scale: 1.0   # Scale factor for aleatoric uncertainty
```

## 7. Config System (`config_system.py`)

### Currently Hardcoded Defaults:
These are already configurable but have specific defaults that users might want to adjust:

- **Line 29**: `c_puct: float = 1.0` - PUCT exploration constant
- **Line 30**: `dirichlet_alpha: float = 0.3` - Dirichlet noise
- **Line 31**: `dirichlet_epsilon: float = 0.25` - Dirichlet noise weight
- **Line 32**: `temperature: float = 1.0` - Temperature for move selection
- **Line 33**: `temperature_threshold: int = 30` - When to switch to low temperature
- **Line 34**: `temperature_final: float = 0.1` - Final temperature
- **Line 41**: `virtual_loss: float = 3.0` - Virtual loss for parallel MCTS
- **Line 47**: `tree_reuse_fraction: float = 0.5` - Fraction of tree to reuse
- **Line 73**: `minhash_size: int = 64` - MinHash signature size
- **Line 74**: `phase_kick_strength: float = 0.1` - Phase kick strength
- **Line 204**: `win_threshold: float = 0.55` - Win rate to accept new model
- **Line 384**: `config.mcts.dirichlet_alpha = 0.3` - Chess-specific
- **Line 390**: `config.mcts.dirichlet_alpha = 0.03` - Go-specific (lower for larger board)
- **Line 395**: `config.mcts.dirichlet_alpha = 0.15` - Gomoku-specific

## Summary of Key Parameters to Add

### High Priority (Performance Critical):
1. **Wave MCTS c_puct**: Currently hardcoded to 1.414, different from standard 1.0
2. **Interference thresholds and factors**: Critical for quantum performance
3. **Memory pool sizes**: Different defaults in different places
4. **Neural network architecture in arena**: Should match main network config

### Medium Priority (User Customization):
1. **DataLoader settings**: Workers, pin memory
2. **Tournament settings**: Model selection step
3. **Illegal move penalty**: Currently -1e9
4. **Policy/Value head filters**: Currently hardcoded in model

### Low Priority (Advanced Tuning):
1. **Uncertainty formulas**: Epistemic and aleatoric variance calculations
2. **GPU utilization targets**: For performance tuning
3. **Game-specific Dirichlet alphas**: Already have defaults but not exposed

## Recommended Config Structure Addition

```yaml
# Add to AlphaZeroConfig
advanced:
  # Wave MCTS specific
  wave_c_puct: 1.414
  
  # Quantum interference
  interference_threshold: 0.1
  constructive_interference_factor: 0.1
  destructive_interference_factor: 0.05
  
  # Neural network details
  illegal_move_penalty: -1e9
  policy_head_filters: 32
  value_head_filters: 1
  
  # Training details
  dataloader_num_workers: 4
  dataloader_pin_memory: true
  
  # Tournament
  tournament_model_step: 10
  
  # Uncertainty computation
  epistemic_variance_base: 1.0
  aleatoric_variance_scale: 1.0
```

This would allow users to override these advanced parameters when needed while keeping the config clean for basic usage.