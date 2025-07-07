# MCTS Test Suite Documentation

This directory contains comprehensive tests for the Monte Carlo Tree Search (MCTS) implementation with AlphaZero-style neural network integration.

## Test Structure

```
python/tests/
├── conftest.py                         # Shared fixtures and test configuration
├── test_core/                          # Core MCTS functionality tests
│   ├── test_mcts.py                   # Core MCTS algorithm tests
│   ├── test_tree_operations.py        # Tree operations and management
│   ├── test_wave_search.py            # Wave-based parallelization
│   ├── test_game_interface.py         # Game interface abstraction
│   └── test_evaluator.py              # Evaluator interface tests
├── test_gpu/                           # GPU acceleration tests
│   ├── test_csr_tree.py              # CSR tree structure
│   ├── test_mcts_gpu_accelerator.py  # GPU kernel acceleration
│   ├── test_node_data_manager.py     # Node data management
│   ├── test_csr_storage.py           # CSR storage format
│   ├── test_ucb_selector.py          # UCB selection on GPU
│   ├── test_memory_pool.py           # Memory pool management
│   └── test_gpu_game_states.py       # GPU game state handling
├── test_neural_networks/               # Neural network tests
│   ├── test_resnet_model.py          # ResNet architecture
│   ├── test_resnet_evaluator.py      # Neural network evaluation
│   ├── test_self_play_module.py      # Self-play generation
│   ├── test_arena_module.py          # Model comparison arena
│   └── test_training_pipeline.py     # Complete training pipeline
├── test_utils/                         # Utility tests
│   ├── test_batch_coordinator.py     # Batch coordination
│   ├── test_config_system.py         # Configuration management
│   ├── test_optimized_remote_evaluator.py  # Remote evaluation
│   ├── test_gpu_evaluator_service.py # GPU service
│   └── test_validation.py            # Input validation
├── test_integration/                   # Integration tests
│   ├── test_integration_mcts.py      # MCTS integration scenarios
│   ├── test_integration_training.py  # Training pipeline integration
│   ├── test_multiprocessing.py       # Multiprocessing scenarios
│   └── test_performance.py           # Performance benchmarks
└── test_games/                         # Game-specific tests
    ├── test_gomoku_gameplay.py        # Gomoku rules and gameplay
    ├── test_go_gameplay.py            # Go rules and gameplay
    └── test_chess_gameplay.py         # Chess rules and gameplay
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test module
pytest test_core/test_mcts.py

# Run specific test class
pytest test_core/test_mcts.py::TestMCTSCore

# Run specific test method
pytest test_core/test_mcts.py::TestMCTSCore::test_mcts_initialization
```

### Test Categories

```bash
# Run only unit tests (fast)
pytest -m "not slow and not integration"

# Run integration tests
pytest -m integration

# Run performance benchmarks
pytest -m benchmark

# Run GPU tests (requires CUDA)
pytest -m gpu

# Skip slow tests
pytest -m "not slow"
```

### Coverage Reports

```bash
# Generate coverage report
pytest --cov=mcts --cov-report=html

# View coverage in terminal
pytest --cov=mcts --cov-report=term-missing

# Generate XML coverage (for CI)
pytest --cov=mcts --cov-report=xml
```

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pip install pytest-xdist
pytest -n auto  # Uses all CPU cores
pytest -n 4     # Uses 4 processes
```

### Test Configuration

The test suite uses several configuration options via `pytest.ini`:

```ini
[pytest]
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    gpu: marks tests that require GPU
    benchmark: marks performance benchmark tests
```

### Environment Variables

```bash
# Skip GPU tests if no GPU available
export SKIP_GPU_TESTS=1

# Set test data directory
export TEST_DATA_DIR=/path/to/test/data

# Enable debug output
export MCTS_TEST_DEBUG=1
```

## Test Fixtures

Key fixtures available in `conftest.py`:

- `base_mcts_config` - Basic MCTS configuration
- `alphazero_config` - Full AlphaZero training configuration  
- `mock_evaluator` - Mock neural network evaluator
- `sample_game_state` - Sample game states for testing
- `temp_checkpoint_dir` - Temporary directory for checkpoints

## Writing New Tests

### Test Structure

```python
import pytest
from mcts.core.mcts import MCTS

class TestNewFeature:
    """Test new MCTS feature"""
    
    def test_feature_initialization(self):
        """Test feature initialization"""
        # Arrange
        config = MCTSConfig()
        
        # Act
        mcts = MCTS(config)
        
        # Assert
        assert mcts is not None
        
    @pytest.mark.slow
    def test_feature_performance(self):
        """Test feature performance"""
        # Performance test implementation
        pass
```

### Best Practices

1. **Use descriptive test names** that explain what is being tested
2. **Follow AAA pattern**: Arrange, Act, Assert
3. **Use fixtures** for common setup
4. **Mock external dependencies** to isolate tests
5. **Mark slow tests** with `@pytest.mark.slow`
6. **Use parametrization** for testing multiple cases
7. **Clean up resources** in teardown or use context managers

### Common Patterns

```python
# Parametrized tests
@pytest.mark.parametrize("board_size,expected", [
    (9, 81),
    (13, 169),
    (15, 225),
    (19, 361),
])
def test_board_sizes(board_size, expected):
    game = GameInterface(GameType.GOMOKU, board_size=board_size)
    assert game.max_moves == expected

# Testing exceptions
def test_invalid_move():
    with pytest.raises(ValueError, match="Illegal move"):
        game.apply_move(state, invalid_move)

# Mocking
def test_with_mock(mocker):
    mock_model = mocker.Mock()
    mock_model.forward.return_value = (policies, values)
```

## Continuous Integration

The test suite is designed to run in CI environments:

```yaml
# Example GitHub Actions workflow
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: |
          pytest --cov=mcts --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

## Debugging Tests

### Running with debugger

```bash
# Run with pdb on failure
pytest --pdb

# Run with pdb on first failure
pytest -x --pdb

# Set breakpoint in code
import pdb; pdb.set_trace()
```

### Verbose output

```bash
# Show print statements
pytest -s

# Show test setup/teardown
pytest --setup-show

# Show local variables on failure
pytest -l
```

### Test isolation

```bash
# Run tests in random order (requires pytest-randomly)
pip install pytest-randomly
pytest --randomly-seed=42

# Disable test isolation
pytest --forked
```

## Performance Testing

Performance benchmarks are marked with `@pytest.mark.benchmark`:

```bash
# Run only benchmarks
pytest -m benchmark

# Run with profiling
pytest --profile

# Generate performance report
pytest -m benchmark --benchmark-only
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure PYTHONPATH includes the project root
   ```bash
   export PYTHONPATH=/path/to/omoknuni_quantum/python:$PYTHONPATH
   ```

2. **CUDA errors**: Tests requiring GPU will skip if CUDA is not available
   ```bash
   # Force CPU-only tests
   export CUDA_VISIBLE_DEVICES=""
   ```

3. **Memory issues**: Some tests require significant memory
   ```bash
   # Run memory-intensive tests separately
   pytest -m "memory_intensive" -n 1
   ```

4. **Timeout issues**: Increase timeout for slow tests
   ```bash
   pytest --timeout=300  # 5 minute timeout
   ```

## Test Data

Test data and fixtures are organized as follows:

- Small test data is included directly in test files
- Larger test data is generated programmatically
- Mock models and evaluators are provided in `conftest.py`
- Temporary files are created in system temp directory

## Contributing

When adding new tests:

1. Place tests in the appropriate subdirectory
2. Follow existing naming conventions
3. Add docstrings explaining what is tested
4. Mark slow/GPU/integration tests appropriately
5. Ensure tests are deterministic and isolated
6. Update this README if adding new test categories