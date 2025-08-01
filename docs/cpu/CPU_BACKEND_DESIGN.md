# CPU Backend Design Document

## Overview
This document outlines the design for a new CPU-optimized backend for MCTS, based on analysis of the hybrid implementation and lessons learned from debugging the previous CPU backend.

## Key Design Principles

### 1. Memory Management
- **On-demand allocation**: Unlike GPU, CPU should allocate states as needed
- **Efficient recycling**: Reuse freed states but ensure complete reset
- **No pre-allocation**: Avoid pre-allocating large tensors that cause memory explosion
- **Dynamic growth**: Grow capacity incrementally as needed

### 2. State Representation
- Use numpy arrays for CPU efficiency (avoid PyTorch overhead)
- Keep state data contiguous for cache efficiency
- Minimize memory footprint per state
- Clear separation between game logic and tensor representation

### 3. Batch Operations
- Vectorize operations using numpy for CPU SIMD
- Support batch operations for efficiency
- Avoid Python loops where possible
- Use multi-threading for parallelizable operations

### 4. Interface Compatibility
- Match the GPU game states interface exactly
- Support both basic and enhanced tensor representations
- Ensure seamless integration with wave-based MCTS
- Compatible with self-play manager and training pipeline

## Core Components

### 1. CPUGameStates Class
```python
class CPUGameStates:
    """CPU-optimized game state management"""
    
    def __init__(self, capacity, game_type='gomoku', board_size=15):
        # Dynamic state pool
        # Numpy arrays for efficiency
        # No pre-allocation
        
    def allocate_states(self, num_states):
        # On-demand allocation
        # Return indices of allocated states
        
    def free_states(self, indices):
        # Mark states as free
        # Complete reset of state data
        
    def clone_states(self, parent_indices, num_clones):
        # Efficient batch cloning
        # Ensure proper state copying
        
    def apply_moves(self, state_indices, actions):
        # Vectorized move application
        # Update game state efficiently
        
    def get_legal_moves_mask(self, state_indices):
        # Batch legal move computation
        # Return boolean masks
        
    def get_nn_features(self, state_indices):
        # Extract features for neural network
        # Support basic/enhanced representations
```

### 2. State Pool Management
- Start with small initial capacity (e.g., 1000 states)
- Grow by fixed increments (e.g., 1000 states) not exponentially
- Track free indices efficiently
- Ensure O(1) allocation and deallocation

### 3. Game Logic Integration
- Pure Python implementation for game rules
- No dependency on C++ backend
- Clear and maintainable code
- Proper validation of moves

## Implementation Strategy

### Phase 1: Basic Implementation (Python)
1. Create minimal CPUGameStates class
2. Implement state allocation/deallocation
3. Add move application logic
4. Implement legal moves computation
5. Add state cloning
6. Implement batch operations
7. Add neural network feature extraction

### Phase 2: Integration Testing
1. Test with MCTS wave search
2. Ensure compatibility with self-play
3. Validate training pipeline integration
4. Performance benchmarking

### Phase 3: Optimization
1. Profile and identify bottlenecks
2. Add multi-threading for batch operations
3. Convert critical paths to Cython
4. Memory usage optimization

## Key Differences from Previous Implementation

### Problems to Avoid
1. **No pre-allocation**: Don't allocate full capacity upfront
2. **No exponential growth**: Use linear growth for memory efficiency
3. **Complete state reset**: Ensure freed states are fully reset
4. **No C++ dependency**: Pure Python/numpy for maintainability
5. **Proper synchronization**: State data must be consistent

### Improvements
1. **On-demand allocation**: Allocate only what's needed
2. **Efficient numpy operations**: Use vectorized operations
3. **Clear interfaces**: Well-defined public API
4. **Comprehensive testing**: TDD approach from the start
5. **Memory efficiency**: Minimize memory footprint

## Testing Strategy

### Unit Tests
- Test each method in isolation
- Verify state allocation/deallocation
- Test move application correctness
- Validate legal moves computation
- Test state cloning accuracy

### Integration Tests
- Test with MCTS wave search
- Verify self-play compatibility
- Test training pipeline integration
- Performance benchmarks

### Edge Cases
- Handle capacity limits gracefully
- Test with terminal states
- Verify batch operation consistency
- Test concurrent access patterns

## Performance Targets
- Memory usage: < 1GB for 100k states
- Allocation speed: > 100k states/second
- Move application: > 1M moves/second
- Legal moves computation: > 500k states/second
- Feature extraction: > 100k states/second

## Future Enhancements
- Multi-threaded batch operations
- SIMD optimizations
- Cache-friendly data layout
- Lazy evaluation where beneficial