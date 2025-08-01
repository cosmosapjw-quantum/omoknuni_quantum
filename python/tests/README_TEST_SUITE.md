# MCTS Comprehensive Test Suite

This directory contains a comprehensive test suite for the Monte Carlo Tree Search (MCTS) implementation, designed to validate all three backends (CPU, GPU, hybrid) and all four MCTS phases.

## Test Structure

### Core Test Files

1. **test_mcts_comprehensive.py**
   - Tests all backends for basic functionality
   - Validates search consistency across backends
   - Performance benchmarks for each backend
   - Memory usage and cleanup tests
   - Edge case handling

2. **test_cpu_backend_detailed.py**
   - CPU-specific game state management
   - Vectorized CPU operations
   - Thread safety tests
   - Cache efficiency validation
   - Memory pooling tests

3. **test_gpu_backend_detailed.py**
   - GPU tensor operations
   - CUDA kernel functionality
   - GPU memory management
   - Batch processing efficiency
   - Mixed precision support

4. **test_hybrid_backend_detailed.py**
   - CPU-GPU communication pipeline
   - Thread-safe parallel operations
   - Memory management across devices
   - Performance optimizations
   - Scalability tests

5. **test_mcts_phases_detailed.py**
   - Selection phase: UCB calculation, path diversity
   - Expansion phase: Node creation, legal move handling
   - Evaluation phase: Neural network batching
   - Backpropagation phase: Value updates, consistency

## Running Tests

### Run All Tests
```bash
cd python/tests
python run_comprehensive_tests.py
```

### Run Specific Backend Tests
```bash
# CPU backend only
pytest test_cpu_backend_detailed.py -v

# GPU backend only (requires CUDA)
pytest test_gpu_backend_detailed.py -v

# Hybrid backend
pytest test_hybrid_backend_detailed.py -v
```

### Run Phase-Specific Tests
```bash
# Test specific phase
pytest test_mcts_phases_detailed.py::TestSelectionPhase -v
pytest test_mcts_phases_detailed.py::TestExpansionPhase -v
pytest test_mcts_phases_detailed.py::TestEvaluationPhase -v
pytest test_mcts_phases_detailed.py::TestBackpropagationPhase -v
```

### Run with Coverage
```bash
pytest test_mcts_comprehensive.py --cov=mcts --cov-report=html
```

## Test Configuration

Tests use a mock evaluator by default for reproducibility. Key configuration options:

- **Board Size**: Tests use 9x9 boards by default (faster)
- **Tree Size**: Limited to 10,000-100,000 nodes for tests
- **Simulations**: 100-1000 simulations per test
- **Backends**: Automatically skips GPU tests if CUDA unavailable

## Expected Performance

Minimum performance thresholds:
- **CPU**: 1,000+ simulations/second
- **GPU**: 5,000+ simulations/second  
- **Hybrid**: 3,000+ simulations/second

## Debugging Failed Tests

1. **Run with verbose output**:
   ```bash
   pytest test_file.py::TestClass::test_method -vv -s
   ```

2. **Check CUDA availability**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Memory issues**:
   - Reduce `max_tree_nodes` in tests
   - Close other GPU applications
   - Use `CUDA_VISIBLE_DEVICES=0` to limit GPU

4. **Performance issues**:
   - Check CPU frequency scaling
   - Verify GPU not throttling
   - Disable other background processes

## Test Coverage Goals

The test suite aims to cover:
- ✅ All three backends (CPU, GPU, hybrid)
- ✅ All four MCTS phases
- ✅ Edge cases and error conditions
- ✅ Performance characteristics
- ✅ Memory management
- ✅ Thread safety (hybrid mode)
- ✅ Backend consistency

## Adding New Tests

When adding new tests:
1. Follow existing naming conventions
2. Use appropriate fixtures for setup
3. Include both correctness and performance tests
4. Document expected behavior
5. Add to appropriate test class

## Known Limitations

- GPU tests require CUDA-capable GPU
- Some timing tests may be flaky on heavily loaded systems
- Large tree tests may require significant memory
- Parallel tests are simulated (not truly concurrent) for determinism

## Continuous Integration

For CI/CD pipelines:
```bash
# Quick smoke test
pytest test_mcts_comprehensive.py -k "test_initialization or test_search_basic" --tb=short

# Full test suite without GPU
pytest -k "not gpu" --tb=short

# With timeout
pytest --timeout=300 --timeout-method=thread
```