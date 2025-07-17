# MCTS Performance Optimization Summary

## Issue
The comprehensive preset with 10K simulations was experiencing extreme slowness:
- Expected: ~100 seconds per game (based on 5K sims/sec benchmark)
- Actual: 5+ minutes with no progress, eventual tree overflow errors

## Root Cause Analysis
1. **Non-linear scaling**: MCTS performance doesn't scale linearly with simulation count
   - 1K sims: 2,652 sims/sec
   - 5K sims: 7,825 sims/sec (peak efficiency)
   - 10K sims: 2,343 sims/sec (70% performance drop)

2. **Memory constraints**: Even with aggressive pruning, 10K simulations cause tree overflow

3. **GPU underutilization**: Only 40% GPU usage and 6/8GB VRAM usage with poor configuration

## Solution
1. **Reduced comprehensive preset to 5K simulations**
   - Maintains high performance (~7,800 sims/sec)
   - Avoids tree overflow issues
   - Still provides comprehensive physics analysis

2. **Configuration optimizations**:
   ```python
   # 5K simulations (optimal performance)
   max_wave_size = 7936
   wave_num_pipelines = 5
   batch_size = 768
   inference_batch_size = 896
   memory_pool_size_mb = 2048
   max_tree_nodes = 1,500,000
   initial_children_per_expansion = 15
   ```

3. **Aggressive pruning for high simulation counts**:
   - 10K sims: Prune at 60% full, every 100 simulations
   - 5K sims: Prune at 75% full, every 150 simulations
   - <5K sims: Prune at 85% full, every 200 simulations

## Performance Results
- **5K simulations**: ~30-50 seconds per game
- **Overnight run (100 games)**: ~1-2 hours (previously estimated 12,500 hours!)
- **GPU utilization**: Improved to ~70% with proper configuration

## Recommendations
1. Use 5K simulations as the practical maximum for production runs
2. For deeper analysis requiring 10K+ simulations, use distributed computing
3. Monitor tree fill ratio and adjust pruning parameters as needed
4. Consider using the Optuna optimization script for finding optimal parameters for specific hardware