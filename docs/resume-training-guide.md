# Resume Training Guide

## Overview
The AlphaZero training pipeline fully supports resuming from checkpoints. This allows you to:
- Continue training after interruptions
- Train in multiple sessions
- Recover from crashes

## How to Resume Training

### Basic Resume Command
```bash
python train.py --resume experiments/your_experiment/checkpoints/latest_checkpoint.pt
```

### Resume Options

1. **Resume from latest checkpoint** (symlink):
   ```bash
   python train.py --resume experiments/gomoku_unified_training/checkpoints/latest_checkpoint.pt
   ```

2. **Resume from specific checkpoint**:
   ```bash
   python train.py --resume experiments/gomoku_unified_training/checkpoints/checkpoint_iter_10.pt
   ```

3. **Resume from checkpoint directory** (automatically finds latest):
   ```bash
   python train.py --resume experiments/gomoku_unified_training/checkpoints/
   ```

### Command Line Options
- `--resume PATH`: Path to checkpoint file or directory
- `--iterations N`: Number of additional iterations to run (not total)
- `--config`: Override config file (uses checkpoint's config by default)
- Other flags work as normal

### Examples

**Resume and run 50 more iterations**:
```bash
python train.py --resume experiments/gomoku_unified_training/checkpoints/latest_checkpoint.pt --iterations 50
```

**Resume with different number of workers**:
```bash
python train.py --resume experiments/gomoku_unified_training/checkpoints/latest_checkpoint.pt --workers 8
```

## What Gets Saved/Restored

### Automatically Saved Every Checkpoint:
- Model weights
- Optimizer state
- Learning rate scheduler state
- Current iteration number
- Best model iteration
- ELO ratings
- Replay buffer (saved separately)
- Mixed precision scaler state (if using)

### Checkpoint Structure:
```
experiments/your_experiment/
├── checkpoints/
│   ├── checkpoint_iter_10.pt
│   ├── checkpoint_iter_20.pt
│   ├── latest_checkpoint.pt -> checkpoint_iter_20.pt  # Symlink
│   └── resume_info.json  # Metadata for easy resuming
├── data/
│   ├── replay_buffer_iter_10.pkl
│   └── replay_buffer_iter_20.pkl
├── best_models/
│   └── model_iter_15.pt
└── config.yaml
```

## Resume Info File
Each checkpoint includes a `resume_info.json` with:
```json
{
  "iteration": 20,
  "checkpoint_path": "experiments/.../checkpoint_iter_20.pt",
  "buffer_path": "experiments/.../replay_buffer_iter_20.pkl",
  "timestamp": "2024-01-15T10:30:00"
}
```

## Important Notes

1. **Iterations are additive**: When you specify `--iterations 100` with resume, it runs 100 MORE iterations, not 100 total.

2. **Replay buffer**: The replay buffer is loaded from the most recent available file if the exact iteration file isn't found.

3. **Config consistency**: The original config is preserved in the checkpoint. Command-line overrides still work.

4. **Automatic checkpoint saving**: By default, checkpoints are saved every 10 iterations (configurable via `config.training.checkpoint_interval`).

5. **Interruption handling**: Use Ctrl+C to gracefully interrupt training - it will save a checkpoint before exiting.

## Troubleshooting

**"No checkpoints found" error**:
- Check the checkpoint directory path
- Ensure at least one checkpoint has been saved

**"Replay buffer not found" warning**:
- This is okay if resuming very early in training
- The pipeline will start with an empty buffer

**Memory issues when resuming**:
- The replay buffer can be large
- Consider reducing `config.training.window_size` if needed

**Different GPU when resuming**:
- The checkpoint automatically maps to the correct device
- No manual intervention needed