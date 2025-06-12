# MCTS Refactoring Plan: CPU-GPU Hybrid Optimization

## Executive Summary

This document provides a comprehensive plan to refactor the AlphaZero MCTS implementation, focusing on:
1. Simplifying the overly complex architecture
2. Optimizing CPU-GPU workload distribution
3. Maximizing vectorization before parallelization
4. Maintaining quantum features while reducing complexity

## Current Architecture Analysis

### Problematic Design Patterns

1. **Over-layered MCTS Implementation**
   ```
   MCTS.py → WaveMCTS.py → OptimizedWaveMCTS.py
   ```
   - Too many abstraction layers
   - Duplicated logic across files
   - Hard to track execution flow

2. **Artificial Limitations**
   - `max_tree_nodes`: Causes "Tree full" errors unnecessarily
   - Fixed memory pools: Prevents dynamic growth
   - Rigid wave sizing: Should be flexible based on tree state

3. **Inefficient CPU-GPU Split**
   - Some operations that should be on CPU are forced to GPU
   - Some vectorizable CPU operations are not vectorized
   - Unnecessary data transfers between CPU-GPU

### Comprehensive CPU-GPU Workload Analysis

#### Current CUDA Kernels Assessment

Located in `/python/mcts/gpu/unified_cuda_kernels.cu`:

##### ✅ KEEP on GPU
1. **`batched_ucb_selection`** - Parallel UCB score computation
   - Highly parallel, arithmetic-heavy
   - Benefits from GPU's parallel cores
   - **Verdict**: Keep on GPU

2. **`parallel_backup`** - Tree value propagation
   - Updates many nodes simultaneously
   - Memory bandwidth intensive
   - **Verdict**: Keep on GPU

3. **`fused_minhash_interference`** - Quantum interference calculation
   - Complex mathematical operations
   - Parallel across many paths
   - **Verdict**: Keep on GPU

##### ❌ MOVE to GPU (Currently Wrong Place)
1. **`batch_process_legal_moves`** - Currently branchy
   - **Problem**: Current implementation is branchy
   - **Solution**: Rewrite as parallel mask generation
   - **Verdict**: Keep on GPU but rewrite for parallelism

2. **`evaluate_gomoku_positions`** - Game evaluation
   - **Problem**: Too game-specific
   - **Solution**: Generalize to tensor operations
   - **Verdict**: Keep simple patterns on GPU

##### 🔄 OPTIMIZE for Hybrid
1. **`find_expansion_nodes`** - Currently inefficient
   - **Problem**: Random memory access
   - **Solution**: Batch accumulate on CPU, process on GPU
   - **Verdict**: CPU selection, GPU processing

#### Current CPU Operations Assessment

##### ❌ SHOULD MOVE to GPU
1. **State Creation and Cloning** (ThreadPoolExecutor based)
   - **Current**: CPU threads clone Python objects
   - **Proposed**: GPU batch state tensor operations
   - **Impact**: 10-20x speedup potential

2. **Feature Extraction** (state_to_numpy conversions)
   - **Current**: Serial CPU numpy operations
   - **Proposed**: Direct GPU tensor transforms
   - **Impact**: 3-5x speedup

3. **Legal Move Generation** (per-state iteration)
   - **Current**: CPU loops through states
   - **Proposed**: Parallel GPU mask generation
   - **Impact**: 5-10x speedup

4. **Move Application** (state transitions)
   - **Current**: CPU object manipulation
   - **Proposed**: GPU tensor updates
   - **Impact**: 10x speedup

##### ✅ KEEP on CPU
1. **Tree Structure Management**
   - Dictionary operations
   - Pointer updates
   - Memory allocation coordination

2. **High-level Control Flow**
   - Wave orchestration
   - Timing and synchronization
   - Progress tracking

3. **Cache Management**
   - LRU cache operations
   - Hash table lookups
   - Memory pressure monitoring

4. **I/O and Logging**
   - File operations
   - Debug output
   - Statistics collection

#### New GPU-Native Components to Add

1. **Unified Game State Representation**
```python
class GPUGameStates:
    """Fully GPU-resident game state management for all games"""
    def __init__(self, capacity, game_type, max_size=19):
        if game_type == 'chess':
            # Chess: 8x8 with 6 piece types per player + empty
            self.boards = torch.zeros((capacity, 8, 8), dtype=torch.int8, device='cuda')
            self.pieces = torch.zeros((capacity, 64), dtype=torch.int8, device='cuda')
            self.castling = torch.zeros((capacity, 4), dtype=torch.bool, device='cuda')
            self.en_passant = torch.zeros((capacity,), dtype=torch.int8, device='cuda')
        elif game_type in ['go', 'gomoku']:
            # Go/Gomoku: variable size, simple stone placement
            self.boards = torch.zeros((capacity, max_size, max_size), 
                                     dtype=torch.int8, device='cuda')
            self.board_size = max_size if game_type == 'go' else 15
            self.ko_point = torch.zeros((capacity,), dtype=torch.int16, device='cuda')  # Go only
        
        # Common metadata for all games
        self.current_player = torch.zeros(capacity, dtype=torch.int8, device='cuda')
        self.move_count = torch.zeros(capacity, dtype=torch.int16, device='cuda')
        self.game_type = game_type
```

2. **Generic Legal Move Generation**
```cuda
// Unified kernel that handles all three games
__global__ void generate_legal_moves_unified(
    const int8_t* boards,
    const int8_t* metadata,  // game-specific metadata
    bool* legal_mask,        // Output: batch_size x max_moves
    int* move_counts,        // Output: legal moves per state
    const int batch_size,
    const int game_type,     // 0=chess, 1=go, 2=gomoku
    const int board_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    if (game_type == 2) {  // Gomoku - simplest
        // Check empty squares only
        for (int pos = 0; pos < board_size * board_size; pos++) {
            legal_mask[idx * board_size * board_size + pos] = 
                (boards[idx * board_size * board_size + pos] == 0);
        }
    } else if (game_type == 1) {  // Go
        // Check empty squares + ko rule
        int ko = metadata[idx];  // ko position
        for (int pos = 0; pos < board_size * board_size; pos++) {
            legal_mask[idx * board_size * board_size + pos] = 
                (boards[idx * board_size * board_size + pos] == 0) && (pos != ko);
        }
    } else {  // Chess - most complex
        // Delegate to chess-specific function
        generate_chess_moves(boards, metadata, legal_mask, idx);
    }
}
```

3. **Unified Terminal Detection**
```cuda
__global__ void check_terminal_states(
    const int8_t* boards,
    const int* last_moves,
    bool* is_terminal,
    int8_t* winners,
    const int batch_size,
    const int game_type,
    const int board_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    if (game_type == 2) {  // Gomoku
        check_five_in_row(boards, last_moves, is_terminal, winners, idx, board_size);
    } else if (game_type == 1) {  // Go
        check_go_terminal(boards, is_terminal, winners, idx, board_size);
    } else {  // Chess
        check_chess_terminal(boards, is_terminal, winners, idx);
    }
}
```

2. **Batch State Operations Kernel**
```cuda
__global__ void batch_apply_moves_kernel(
    int8_t* boards,           // State tensor
    int32_t* metadata,        // Game metadata
    const int* parent_ids,    // Parent state indices
    const int* actions,       // Moves to apply
    const int batch_size
)
```

3. **Vectorized Legal Move Kernel**
```cuda
__global__ void generate_legal_moves_mask(
    const int8_t* boards,
    bool* legal_mask,        // Output: batch_size x action_size
    int* move_counts,        // Output: legal moves per state
    const int batch_size,
    const int board_size
)
```

#### CPU-GPU Pipeline Optimization

##### Before (Current):
1. CPU: Select nodes (sequential)
2. CPU: Clone states (ThreadPool)
3. CPU: Extract features (ThreadPool)
4. GPU: Neural network
5. CPU: Get legal moves (loops)
6. CPU: Expand tree
7. GPU: Backup values

##### After (Optimized):
1. CPU: Batch select nodes (vectorized)
2. GPU: Batch clone states (kernel)
3. GPU: Extract features (tensor ops)
4. GPU: Neural network
5. GPU: Generate legal moves (kernel)
6. CPU: Update tree pointers
7. GPU: Backup values

#### Memory Layout Optimization

##### Current Problems:
- Random access to Python dict (node_states)
- Fragmented memory allocation
- Frequent CPU-GPU transfers

##### Proposed Solution:
- Contiguous GPU state pool
- Pre-allocated tensor storage
- Zero-copy state updates

#### Expected Performance Gains

| Operation | Current | Proposed | Speedup |
|-----------|---------|----------|---------|
| State Cloning | 5ms/1000 | 0.5ms/1000 | 10x |
| Feature Extraction | 3ms/1000 | 0.6ms/1000 | 5x |
| Legal Moves | 4ms/1000 | 0.4ms/1000 | 10x |
| Overall Search | 50k sims/s | 300k+ sims/s | 6x+ |

## Refined Refactoring Plan

### Phase 1: Immediate Fixes (1-2 days)

1. **Remove Tree Node Limits**
   ```python
   # In csr_tree.py, change:
   max_nodes = float('inf')  # No artificial limit
   # Use dynamic reallocation when needed
   ```

2. **Fix Float/Double Issues**
   ```python
   # Ensure all tensor operations use float32
   # Add dtype checks at kernel boundaries
   ```

3. **Simplify Memory Management**
   - Remove complex memory pools
   - Use PyTorch's allocator directly
   - Let OS/CUDA handle memory

### Phase 2: Core MCTS Consolidation (3-4 days)

1. **Create `unified_mcts.py`**
   ```python
   class UnifiedMCTS:
       """Single MCTS implementation with wave-based parallelism"""
       
       def __init__(self, game_interface, evaluator, config):
           self.tree = BatchedCSRTree(device='cuda')
           self.game = game_interface  # Runs on CPU
           self.evaluator = evaluator  # NN on GPU
           self.config = config
           
       def search(self, root_state, num_simulations):
           # Core wave-based search
           # CPU: game logic, tree traversal
           # GPU: UCB scores, NN evaluation, value backup
   ```

2. **Simplified Configuration**
   ```python
   @dataclass
   class MCTSConfig:
       # Essential only
       num_simulations: int = 800
       c_puct: float = 1.0
       wave_size: Optional[int] = None  # Auto-determine
       
       # Advanced (with defaults)
       virtual_loss: float = 1.0
       enable_quantum: bool = False
       device_placement: Dict[str, str] = field(default_factory=lambda: {
           'tree_ops': 'cuda',
           'game_logic': 'cpu',
           'nn_eval': 'cuda'
       })
   ```

### Phase 3: CPU-GPU Workload Optimization (3-4 days)

1. **CPU Operations (Vectorized)**
   ```python
   class CPUGameOps:
       """Vectorized game operations on CPU"""
       
       @torch.jit.script
       def batch_get_legal_moves(self, states: torch.Tensor) -> torch.Tensor:
           # Vectorized legal move generation
           # Use torch operations, not loops
           
       @torch.jit.script  
       def batch_apply_moves(self, states: torch.Tensor, actions: torch.Tensor):
           # Vectorized state transitions
           # Leverage CPU SIMD instructions
   ```

2. **GPU Operations (Focused)**
   ```python
   class GPUTreeOps:
       """GPU operations for tree algorithms"""
       
       def batch_ucb_scores(self, q_values, visits, priors, c_puct):
           # Pure arithmetic, perfect for GPU
           
       def parallel_backup(self, leaf_values, paths):
           # Memory bandwidth bound, good for GPU
   ```

3. **Hybrid Pipeline**
   ```python
   def search_iteration(self):
       # 1. CPU: Select paths (tree traversal)
       paths = self.cpu_select_paths()  # Vectorized
       
       # 2. CPU: Get game states for leaves
       leaf_states = self.cpu_get_states(paths)  # Batch operation
       
       # 3. GPU: Evaluate neural network
       values, policies = self.gpu_evaluate(leaf_states)
       
       # 4. CPU: Process legal moves, game logic
       legal_masks = self.cpu_get_legal_moves(leaf_states)  # Vectorized
       
       # 5. GPU: Apply quantum features (if enabled)
       if self.config.enable_quantum:
           policies = self.gpu_apply_quantum(policies, paths)
       
       # 6. CPU: Expand tree structure
       new_nodes = self.cpu_expand_tree(paths, policies, legal_masks)
       
       # 7. GPU: Backup values through tree
       self.gpu_backup_values(paths, values)
   ```

### Phase 4: Vectorization Strategy (2-3 days)

1. **Maximize Tensor Operations**
   ```python
   # Before: Loop over states
   for state in states:
       legal_moves.append(get_legal_moves(state))
   
   # After: Vectorized operation
   legal_moves = batch_get_legal_moves(states)  # Single tensor op
   ```

2. **Batch Everything**
   ```python
   # Process waves of 1024-4096 operations at once
   # No explicit multiprocessing needed initially
   ```

3. **JIT Compilation**
   ```python
   # Use torch.jit.script for CPU operations
   # Use torch.compile for main search loop
   ```

### Phase 5: Clean Multiprocessing (2 days)

1. **Only After Vectorization**
   ```python
   # Simple data parallel for self-play
   # Each process runs independent games
   # No complex GPU service needed
   ```

2. **Shared Memory Model**
   ```python
   # Model stays on GPU
   # Processes share via torch.multiprocessing
   # No serialization of tensors
   ```

## Implementation Priority Based on CPU-GPU Analysis

### Critical Path - Highest Impact First

#### Stage 1: GPU State Management (3-4 days)
**Why First**: Current bottleneck is CPU-based state management
- [ ] Implement GPUGameStates class with tensor representation
- [ ] Create batch_apply_moves CUDA kernel
- [ ] Replace Python state objects with GPU tensors
- [ ] Expected impact: 10-20x speedup on state operations

#### Stage 2: Vectorized Legal Moves (2-3 days)
**Why Second**: Legal move generation is second biggest bottleneck
- [ ] Implement generate_legal_moves_mask CUDA kernel
- [ ] Remove CPU loops for legal move checking
- [ ] Batch process legal moves on GPU
- [ ] Expected impact: 5-10x speedup

#### Stage 3: Direct GPU Feature Extraction (2 days)
**Why Third**: Eliminate CPU-GPU transfer overhead
- [ ] Implement GPU-native feature extraction
- [ ] Remove state_to_numpy conversions
- [ ] Direct tensor transformations on GPU
- [ ] Expected impact: 3-5x speedup

#### Stage 4: Tree Operation Optimization (2-3 days)
**Why Fourth**: Optimize already good GPU kernels
- [ ] Optimize memory access patterns in UCB selection
- [ ] Improve parallel_backup coalescing
- [ ] Fine-tune kernel launch parameters
- [ ] Expected impact: 1.5-2x speedup

#### Stage 5: Game Generalization (3 days)
**Why Important**: Current kernels are too game-specific
- [ ] Create unified game state tensor format
- [ ] Implement generic legal move generation
- [ ] Abstract win/terminal detection patterns
- [ ] Support Chess, Go, and Gomoku with same kernels
- [ ] Expected impact: Code reuse and maintainability

#### Stage 6: CPU Optimization (2 days)
**Why Last**: CPU becomes less critical after GPU optimizations
- [ ] Vectorize remaining CPU operations
- [ ] Optimize tree pointer updates
- [ ] Implement efficient batch buffering
- [ ] Expected impact: 1.2x speedup

## Implementation Checklist

### Week 1 - Foundation
- [ ] Fix float/double dtype issues (Day 1)
- [ ] Remove tree node limits (Day 1)
- [ ] Implement GPUGameStates class (Days 2-3)
- [ ] Create batch_apply_moves kernel (Days 4-5)

### Week 2 - Core Optimization  
- [ ] Implement legal moves GPU kernel (Days 1-2)
- [ ] Create unified_mcts.py with GPU state management (Days 3-4)
- [ ] Implement GPU feature extraction (Day 5)

### Week 3 - Integration and Polish
- [ ] Optimize existing GPU kernels (Days 1-2)
- [ ] Clean up CPU operations (Day 3)
- [ ] Performance testing and tuning (Days 4-5)
- [ ] Documentation and cleanup

## Critical Success Factors

1. **Maintain Core Algorithm**: Wave-based MCTS must work identically
2. **Preserve Quantum Features**: All quantum enhancements must remain
3. **Improve Performance**: Target 150k+ sims/second
4. **Reduce Complexity**: Fewer files, clearer logic
5. **CPU-GPU Balance**: Use each processor for its strengths

## Testing Strategy

1. **Unit Tests**: Each component in isolation
2. **Integration Tests**: Full MCTS search validation
3. **Performance Benchmarks**: Track sim/second improvements
4. **Game Tests**: Ensure game logic remains correct

## Migration Path

1. Create new implementation alongside old
2. A/B test to ensure correctness
3. Gradually switch over
4. Remove old code once stable

## File Structure After Refactoring

```
mcts/
├── core/
│   ├── unified_mcts.py       # Single MCTS implementation
│   ├── cpu_ops.py           # Vectorized CPU operations  
│   ├── gpu_ops.py           # Focused GPU operations
│   └── tree.py              # Simplified tree structure
├── quantum/
│   └── features.py          # Consolidated quantum features
└── games/
    └── game_interface.py    # Clean game abstraction
```

## Performance Targets

- **Current**: ~50k simulations/second (complex)
- **Target**: 150k+ simulations/second (optimized)
- **CPU Usage**: 80% (vectorized game logic)
- **GPU Usage**: 90% (parallel tree operations)
- **Memory**: Dynamic allocation, no artificial limits

## Next Steps

1. Start with Phase 1 immediate fixes
2. Implement GPU state management (highest impact)
3. Create unified_mcts.py with GPU-first design
4. Gradually migrate all bottleneck operations to GPU
5. Remove old implementations once stable

## Summary of Key Changes

### Architecture Simplification
- **Before**: MCTS → WaveMCTS → OptimizedWaveMCTS (3 layers)
- **After**: UnifiedMCTS (1 layer with clear CPU/GPU split)

### State Management Revolution
- **Before**: Python objects on CPU with ThreadPool parallelism
- **After**: GPU tensors with CUDA kernel parallelism

### Workload Distribution
- **CPU**: Tree pointers, control flow, I/O, caching
- **GPU**: States, moves, features, math, neural networks

### Expected Outcome
- **Performance**: 50k → 300k+ simulations/second
- **Complexity**: 3 files → 1 file
- **Maintainability**: Clear separation of concerns
- **Scalability**: No artificial limits, dynamic growth

This comprehensive plan addresses both the immediate issues and provides a roadmap for achieving state-of-the-art MCTS performance through proper CPU-GPU workload distribution and maximum vectorization before parallelization.