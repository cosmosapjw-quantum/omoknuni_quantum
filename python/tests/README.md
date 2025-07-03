# MCTS Test Suite

This directory contains the comprehensive test suite for the MCTS project, designed to validate all functionality after streamlining optimizations.

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and shared fixtures
├── pytest.ini              # Pytest settings and markers
├── .coveragerc              # Coverage configuration
├── test_runner.py           # Custom test runner script
├── README.md                # This file
├── core/                    # Core component tests
│   ├── test_evaluator.py    # Evaluator implementations
│   ├── test_game_interface.py # Game interface functionality  
│   └── test_mcts.py         # MCTS algorithm tests
├── utils/                   # Utility component tests
│   ├── test_batch_evaluation_coordinator.py # Batch coordination
│   └── test_config_system.py # Configuration management
└── integration/             # Integration and end-to-end tests
    ├── test_mcts_integration.py # MCTS system integration
    └── test_end_to_end.py    # Complete workflows
```

## Running Tests

### Quick Start

```bash
# Run all tests
python tests/test_runner.py all

# Run only fast tests (exclude slow benchmarks)
python tests/test_runner.py fast

# Run smoke tests (basic functionality check)
python tests/test_runner.py smoke

# Run specific test suites
python tests/test_runner.py unit       # Core + Utils
python tests/test_runner.py integration
python tests/test_runner.py core
python tests/test_runner.py utils
```

### Direct Pytest Usage

```bash
# Run all tests with coverage
pytest tests/ --cov=mcts --cov-report=term-missing --cov-report=html

# Run specific test file
pytest tests/core/test_mcts.py -v

# Run tests matching pattern
pytest tests/ -k "test_mcts" -v

# Run tests excluding slow ones
pytest tests/ -m "not slow" -v

# Run only integration tests
pytest tests/integration/ -v
```

### Test Runner Options

```bash
# Verbose output
python tests/test_runner.py all -v

# Stop on first failure
python tests/test_runner.py all -x

# Run tests matching keyword
python tests/test_runner.py all -k "evaluator"

# Disable coverage reporting
python tests/test_runner.py all --no-cov

# Run tests in parallel (requires pytest-xdist)
python tests/test_runner.py all --parallel
```

## Test Categories

### Unit Tests (Core)
- **test_evaluator.py**: Tests for evaluator base class, MockEvaluator, and AlphaZeroEvaluator
- **test_game_interface.py**: Tests for game interface, game types, and board operations
- **test_mcts.py**: Tests for MCTS algorithm, configuration, and search functionality

### Unit Tests (Utils)
- **test_batch_evaluation_coordinator.py**: Tests for batch coordination and optimization
- **test_config_system.py**: Tests for configuration management and merging

### Integration Tests
- **test_mcts_integration.py**: Integration between MCTS, evaluators, and game interfaces
- **test_end_to_end.py**: Complete game simulations, self-play, and performance benchmarks

## Test Markers

The test suite uses pytest markers to categorize tests:

- `@pytest.mark.slow`: Performance tests and benchmarks (excluded from fast runs)
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.performance`: Performance benchmarks
- `@pytest.mark.gpu`: Tests requiring GPU (currently none, CPU-focused)
- `@pytest.mark.quantum`: Tests for quantum components (if enabled)

## Fixtures

### Global Fixtures (conftest.py)
- `device`: Test device (CPU for CI/CD compatibility)
- `gomoku_config`: Standard Gomoku MCTS configuration
- `small_gomoku_config`: Fast testing configuration (9x9 board, few simulations)
- `game_interface`: Standard game interface
- `mock_evaluator`: MockEvaluator instance
- `sample_game_state`: Example game state with some moves
- `batch_game_states`: Batch of game states for testing
- `winning_state`: Game state where Player 1 has won
- `near_winning_state`: Game state where Player 1 can win in one move

### Configuration Fixtures
- `alphazero_config`: AlphaZero configuration
- `config_manager`: Configuration manager instance

### Performance Fixtures
- `performance_config`: Configuration for performance testing

## Coverage

The test suite aims for high coverage of the streamlined codebase:

- **Target Coverage**: 80%+ overall
- **Excludes**: Research code, quantum research, legacy components
- **Reports**: Terminal output + HTML report in `htmlcov/`

### Viewing Coverage

```bash
# Generate and view HTML coverage report
pytest tests/ --cov=mcts --cov-report=html
firefox htmlcov/index.html  # or your preferred browser
```

## Continuous Integration

The test suite is designed to work well in CI/CD environments:

- All tests use CPU-only configurations
- No external dependencies on CUDA/GPU
- Reasonable execution times (fast tests < 2 minutes)
- Clear pass/fail reporting

## Performance Testing

### Fast Tests
- Unit tests: < 1 second each
- Integration tests: < 10 seconds each
- Total fast test time: < 2 minutes

### Slow Tests (Benchmarks)
- Performance benchmarks: 10-60 seconds each
- Memory usage tests: 30-120 seconds each
- Scalability tests: 60-300 seconds each

### Running Performance Tests
```bash
# Run only performance benchmarks
python tests/test_runner.py slow

# Run specific performance test
pytest tests/integration/test_end_to_end.py::TestPerformanceBenchmarks -v
```

## Debugging Tests

### Common Issues

1. **Import Errors**: Ensure PYTHONPATH includes the project root
2. **CUDA Warnings**: Tests use CPU-only configurations to avoid GPU dependencies
3. **Timeout Errors**: Increase timeout in pytest.ini if needed
4. **Memory Issues**: Run with smaller configurations or fewer parallel workers

### Debug Mode
```bash
# Run with debug output
pytest tests/ -v -s --tb=long

# Run single test with maximum output
pytest tests/core/test_mcts.py::TestMCTS::test_mcts_creation -v -s --tb=long

# Run with pdb on failure
pytest tests/ --pdb
```

## Test Development Guidelines

### Writing New Tests

1. **Use Fixtures**: Leverage existing fixtures from conftest.py
2. **Fast by Default**: Mark slow tests with `@pytest.mark.slow`
3. **Clear Names**: Use descriptive test method names
4. **Good Coverage**: Test both success and error cases
5. **Reproducible**: Use seeds for random components

### Test Structure
```python
class TestComponentName:
    """Test the ComponentName class"""
    
    def test_component_creation(self, fixture_name):
        """Test creating component with valid parameters"""
        # Arrange
        # Act  
        # Assert
        
    def test_component_error_handling(self):
        """Test component error handling"""
        with pytest.raises(ExpectedError):
            # Test error case
```

### Performance Test Guidelines
```python
@pytest.mark.slow
def test_performance_benchmark(self):
    """Test performance characteristics"""
    start_time = time.time()
    # ... perform operation
    elapsed_time = time.time() - start_time
    
    # Assert reasonable performance
    assert elapsed_time < MAX_ACCEPTABLE_TIME
```

## Validation After Streamlining

This test suite specifically validates that the streamlined codebase maintains all essential functionality:

### Core Validation
- ✅ MCTS search algorithm works correctly
- ✅ Game interfaces handle all game types
- ✅ Evaluators provide consistent results
- ✅ Configuration system manages settings properly

### Performance Validation  
- ✅ Batch coordination achieves optimization goals
- ✅ System maintains target simulation speeds
- ✅ Memory usage remains reasonable
- ✅ No performance regressions from streamlining

### Integration Validation
- ✅ Components work together seamlessly
- ✅ End-to-end workflows complete successfully
- ✅ Self-play and tournaments function properly
- ✅ Error handling is robust across the system

## Contributing

When adding new functionality to the MCTS codebase:

1. **Add Tests**: Write comprehensive tests for new features
2. **Update Fixtures**: Add new fixtures to conftest.py if needed
3. **Mark Appropriately**: Use correct pytest markers
4. **Maintain Coverage**: Ensure new code is covered by tests
5. **Run Full Suite**: Verify all tests pass before committing

## Support

For test-related issues:

1. Check this README for common solutions
2. Run tests with `-v` flag for detailed output
3. Use `--tb=long` for full error tracebacks
4. Check coverage report to identify untested code paths