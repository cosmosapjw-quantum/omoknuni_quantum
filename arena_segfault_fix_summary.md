# Arena Segfault Fix Summary

## Issue Description
The training pipeline was experiencing a segmentation fault during arena evaluation at iteration 4, specifically after playing 109/120 games. The crash was accompanied by a warning about leaked semaphore objects.

## Root Cause Analysis

1. **Memory Fragmentation from Tree Reuse**: The MCTS `update_root()` method was calling `reset_tree()` which completely recreated the CSRTree structure. This caused repeated CUDA memory allocations/deallocations leading to memory fragmentation.

2. **Accumulating Memory Usage**: Over 120 arena games with ~50-100 moves each, the repeated tree resets were causing thousands of CUDA memory operations, eventually leading to memory corruption.

3. **Resource Leaks**: MCTS instances weren't being properly cleaned up between games, causing resource accumulation.

## Implemented Fixes

### 1. Disabled Tree Reuse in Arena Games
- Removed calls to `update_root()` in arena evaluation
- Tree reuse is not critical for arena games since they're one-off evaluations

### 2. Added Explicit Resource Cleanup
- Added explicit deletion of MCTS instances after each game
- Added periodic garbage collection (every 10 games by default)
- Added CUDA cache clearing when garbage collecting

### 3. Reduced Memory Footprint for Arena MCTS
- Limited wave sizes to max 64 (vs 3072 for training)
- Reduced memory pool to 128MB (vs 2048MB for training)
- Limited tree nodes to 10,000 (vs 500,000 for training)
- Disabled CUDA graphs for arena to reduce memory overhead

### 4. Added Comprehensive Debug Logging
- Memory usage tracking (RSS, VMS, GPU memory)
- MCTS instance creation/deletion logging
- Exception tracking with full tracebacks

### 5. Added Configuration Options
- `enable_tree_reuse`: Defaults to False for arena games
- `gc_frequency`: Controls how often to run garbage collection

## Files Modified
- `/home/cosmo/omoknuni_quantum/python/mcts/neural_networks/arena_module.py`

## Testing Recommendations
1. Run the training pipeline again to verify the segfault is resolved
2. Monitor memory usage during arena evaluation
3. Check that arena results are still valid without tree reuse

## Future Improvements
- Implement proper tree reuse in MCTS that doesn't recreate the entire tree structure
- Consider using a memory pool for CSRTree allocations to avoid fragmentation
- Add memory usage limits and early termination if memory exceeds threshold