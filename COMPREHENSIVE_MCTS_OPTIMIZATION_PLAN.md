# Comprehensive MCTS Optimization Plan: Achieving 3000+ Sims/Sec for Gomoku Self-Play

**Target**: 3000+ simulations/second with 1000 simulations/move  
**Hardware**: RTX 3060 Ti, 12GB VRAM, Tensor Cores  
**Current Status**: Direct tests show 3038 sims/sec, but self-play shows 1398 sims/sec  
**Goal**: Eliminate the 46% performance gap and achieve consistent 3000+ sims/sec in actual self-play

## Executive Summary

Through comprehensive research using multiple specialized agents, we've identified the root causes of the performance gap between direct MCTS tests (3038 sims/sec) and actual self-play (1398 sims/sec). The research reveals five critical optimization areas with realistic performance improvement potential.

## Research Summary

### 1. Self-Play Pipeline Bottlenecks (Research Agent 1)
**Primary Issue**: 46% performance loss between direct MCTS and self-play

**Root Causes Identified**:
- **Tree Reuse Disabled** (15-20% loss): `self._disable_tree_reuse = True` forces full reconstruction after each move
- **Game State Copying** (20-25% loss): Extensive state synchronization between moves  
- **NN Evaluation Patterns** (8-12% loss): Batch size mismatches, memory fragmentation
- **Validation Overhead** (5-8% loss): Additional checks in hybrid backend

**Key Finding**: The biggest bottleneck is disabled tree reuse forcing full tree reconstruction instead of efficient subtree preservation.

### 2. GPU Game States Optimization (Research Agent 2)
**Potential Impact**: 75-150% improvement (after critical analysis: 20-30% realistic)

**High-Impact Optimizations**:
- **Asynchronous Pipeline** (40-50% claimed → 15-20% realistic): Overlap computation with data transfer
- **Fused CUDA Kernels** (35-40% claimed → 10-15% realistic): Eliminate kernel launch overhead
- **Zero-Copy Memory** (25-30% claimed → 8-12% realistic): Reduce CPU-GPU transfer overhead
- **Tensor Core Utilization** (20-25% claimed → 5-8% realistic): Limited applicability to game states

### 3. CSR Tree Optimization (Research Agent 3)
**Potential Impact**: 2-3x speedup (after critical analysis: 1.3-1.5x realistic)

**Critical Optimizations**:
- **Memory Coalescing** (2-3x claimed → 1.2-1.3x realistic): Already partially implemented
- **Fused UCB Kernels** (1.5-2x claimed → 1.1-1.2x realistic): Combine operations
- **Tree Reuse Optimization** (15-20% improvement): Enable efficient root shifting

### 4. Memory Pool and UCB Optimization (Research Agent 4)
**Potential Impact**: +130% total (after critical analysis: +25-35% realistic)

**Key Optimizations**:
- **Persistent Memory Allocation** (+40% claimed → +15% realistic): Reduce allocation overhead
- **Tensor Core UCB** (+60% claimed → +10% realistic): Limited benefit for UCB math
- **Batch Processing** (+30% claimed → +8% realistic): Better GPU utilization

### 5. Custom CUDA Kernels (Research Agent 5)
**Potential Impact**: 70-105% improvement (after critical analysis: 15-25% realistic)

**Proven Techniques**:
- **Kernel Fusion** (20-30% improvement → 8-12% realistic): Reduce launch overhead
- **Cache Optimization** (10-15% improvement → 5-8% realistic): RTX 3060 Ti specific
- **Lock-free MCTS** (25-35% improvement → 5-10% realistic): High complexity, uncertain benefit

## Master Optimization Strategy

### Performance Projection (Conservative)
- **Current**: 1398 sims/sec (self-play) / 3038 sims/sec (direct)
- **Phase 1**: 1800-2000 sims/sec (25-40% improvement)
- **Phase 2**: 2300-2700 sims/sec (additional 25-35% improvement)
- **Phase 3**: 3000-3500 sims/sec (additional 20-30% improvement)

### Phase 1: Foundation Optimizations (2-3 weeks)
**Target**: 1800-2000 sims/sec | **Risk**: Low | **Complexity**: Medium

#### 1.1 Tree Reuse Restoration (HIGHEST PRIORITY)
```python
# File: python/mcts/core/mcts.py
# Line 63: Enable tree reuse with safeguards
self._disable_tree_reuse = False  # Current bottleneck
config.enable_efficient_tree_reuse = True
config.tree_reuse_validation_level = "minimal"  # Reduce overhead
```

#### 1.2 Game State Management Optimization
```python
# File: python/mcts/gpu/gpu_game_states.py
# Implement zero-copy state updates
class ZeroCopyGameStates:
    def apply_move_incremental(self, state_idx, action):
        # Avoid full board copying
        # Use move history instead of board reconstruction
```

#### 1.3 Memory Pool Implementation
```python
# File: python/mcts/core/wave_search.py
# Add persistent memory pools
class PersistentMemoryPool:
    def __init__(self, device, capacity=2000000):
        # Pre-allocate memory in chunks
        # Avoid allocation/deallocation during play
```

#### 1.4 Batch Evaluation Optimization
```python
# File: python/example_self_play.py
# Maintain evaluation context across moves
evaluator.enable_persistent_batching(True)
evaluator.allocate_game_length_buffers(max_moves=100)
```

**Expected Impact**: +25-40% improvement (1398 → 1800-2000 sims/sec)

### Phase 2: Advanced GPU Optimizations (3-4 weeks)
**Target**: 2300-2700 sims/sec | **Risk**: Medium | **Complexity**: High

#### 2.1 Custom CUDA Kernel Development
```cuda
// File: python/mcts/gpu/mcts_kernels.cu
__global__ void fused_select_expand_kernel(
    const int* root_nodes,
    float* ucb_scores,
    int* selected_paths,
    const int batch_size
) {
    // Combine selection + expansion in single kernel
    // Reduce kernel launch overhead by 4x
}
```

#### 2.2 Asynchronous Pipeline Implementation
```python
# File: python/mcts/core/async_wave_search.py
class OptimizedAsyncPipeline:
    def run_async_evaluation(self):
        # Overlap NN evaluation with tree operations
        # Use multiple CUDA streams
        # Double-buffer game states
```

#### 2.3 Memory Coalescing Optimization
```python
# File: python/mcts/gpu/csr_tree.py
# Enable blocked memory layout (already partially implemented)
config.use_blocked_csr_layout = True
config.block_size = 128  # Optimize for RTX 3060 Ti
```

**Expected Impact**: +25-35% additional improvement (2000 → 2300-2700 sims/sec)

### Phase 3: Expert-Level Optimizations (2-3 weeks)
**Target**: 3000-3500 sims/sec | **Risk**: High | **Complexity**: Very High

#### 3.1 Multi-Stream CUDA Implementation
```python
# File: python/mcts/core/wave_search.py
class MultiStreamMCTS:
    def __init__(self):
        self.streams = [torch.cuda.Stream() for _ in range(4)]
        # Implement stream-based work scheduling
```

#### 3.2 Advanced Memory Management
```python
# File: python/mcts/gpu/memory_manager.py
class CacheOptimizedMemory:
    def __init__(self):
        # Utilize RTX 3060 Ti's 4MB L2 cache
        # Pre-allocate tensors to fit in cache
```

#### 3.3 Tensor Core Integration
```python
# File: python/mcts/neural_networks/resnet_model.py
@torch.compile(mode="max-autotune", backend="torch_tensorrt")
def forward_tensor_core_optimized(self, x):
    # Mixed precision with tensor cores
    # FP16 computation, FP32 accumulation
```

**Expected Impact**: +20-30% additional improvement (2700 → 3000-3500 sims/sec)

## Implementation Roadmap

### Week 1: Tree Reuse and State Management
- [ ] Enable tree reuse with proper safeguards (`mcts.py` line 63)
- [ ] Implement incremental game state updates (`gpu_game_states.py`)
- [ ] Add move history-based state management
- [ ] **Success Criteria**: 1600-1700 sims/sec (15% improvement)

### Week 2: Memory Pool and Batch Optimization
- [ ] Implement persistent memory pools (`wave_search.py`)
- [ ] Add game-length buffer pre-allocation (`example_self_play.py`)
- [ ] Optimize evaluation context persistence
- [ ] **Success Criteria**: 1800-2000 sims/sec (cumulative 25-40% improvement)

### Week 3: Testing and Validation
- [ ] Comprehensive performance testing across different scenarios
- [ ] Stability testing for long self-play sessions
- [ ] Memory usage optimization and leak detection
- [ ] **Success Criteria**: Stable 1800+ sims/sec performance

### Week 4-5: Custom CUDA Kernels
- [ ] Develop fused selection-expansion kernels (`mcts_kernels.cu`)
- [ ] Implement batch UCB computation kernels
- [ ] Add memory coalescing optimization
- [ ] **Success Criteria**: 2100-2300 sims/sec (additional 15-20% improvement)

### Week 6-7: Asynchronous Pipeline
- [ ] Implement multi-stream evaluation pipeline (`async_wave_search.py`)
- [ ] Add double-buffering for game states
- [ ] Optimize CPU-GPU synchronization points
- [ ] **Success Criteria**: 2300-2700 sims/sec (additional 10-15% improvement)

### Week 8-10: Expert Optimizations
- [ ] Advanced memory management with cache optimization
- [ ] Tensor core integration for inference
- [ ] Multi-GPU scaling preparation (stretch goal)
- [ ] **Success Criteria**: 3000+ sims/sec target achievement

## Risk Mitigation

### High-Risk Optimizations
1. **Tree Reuse Re-enablement**
   - **Risk**: Correctness issues, state inconsistency
   - **Mitigation**: Comprehensive unit testing, gradual rollout
   - **Fallback**: Keep current disabled state as backup

2. **Custom CUDA Kernels**
   - **Risk**: Development complexity, debugging difficulty
   - **Mitigation**: Start with simple kernels, extensive validation
   - **Fallback**: Use existing optimized PyTorch operations

3. **Multi-Stream Implementation**
   - **Risk**: Synchronization bugs, deadlocks
   - **Mitigation**: Incremental implementation, thorough testing
   - **Fallback**: Single-stream with improved batching

### Testing Strategy
```python
# Continuous validation framework
class OptimizationValidator:
    def __init__(self):
        self.baseline_performance = 1398  # Current self-play speed
        self.target_performance = 3000   # Target speed
        
    def validate_optimization(self, optimization_name):
        # A/B test against baseline
        # Performance regression detection
        # Memory usage monitoring
        # Stability testing over extended periods
```

## Technical Implementation Details

### File Structure and Changes
```
python/mcts/core/
├── mcts.py (line 63: enable tree reuse)
├── wave_search.py (add memory pools)
├── async_wave_search.py (async pipeline)
└── mcts_config.py (optimization flags)

python/mcts/gpu/
├── gpu_game_states.py (zero-copy updates)
├── csr_tree.py (memory coalescing)
├── mcts_kernels.cu (custom kernels)
└── memory_manager.py (cache optimization)

python/
├── example_self_play.py (persistent evaluation)
└── tests/ (comprehensive testing)
```

### Configuration Parameters
```python
# Optimization configuration
OPTIMIZATION_CONFIG = {
    "enable_tree_reuse": True,
    "tree_reuse_validation": "minimal",
    "enable_persistent_memory": True,
    "memory_pool_size_mb": 4096,
    "enable_async_pipeline": True,
    "num_cuda_streams": 8,
    "enable_custom_kernels": True,
    "enable_tensor_cores": True,
    "mixed_precision_level": "fp16",
}
```

## Success Metrics

### Performance Targets
- **Phase 1**: 1800-2000 sims/sec (baseline recovery)
- **Phase 2**: 2300-2700 sims/sec (advanced optimizations)
- **Phase 3**: 3000-3500 sims/sec (target achievement)

### Quality Metrics
- **Stability**: <1% performance variance over 1000+ move sessions
- **Memory**: <10GB VRAM usage (within RTX 3060 Ti limits)
- **Accuracy**: No degradation in game playing strength
- **Maintainability**: <20% increase in code complexity

## Conclusion

This comprehensive plan addresses the identified performance gap through systematic optimization of each bottleneck. The phased approach ensures manageable risk while providing clear milestones toward the 3000+ sims/sec target. The conservative projections account for implementation challenges and diminishing returns, while the detailed implementation roadmap provides actionable steps for development.

**Key Success Factors**:
1. Fix the primary bottleneck (disabled tree reuse) first
2. Implement optimizations incrementally with continuous testing
3. Focus on proven techniques over experimental approaches
4. Maintain fallback strategies for high-risk optimizations
5. Validate performance gains at each phase before proceeding

The plan is designed to be realistic, implementable, and maintainable while achieving significant performance improvements that exceed the stated target.