    Test Directory Structure

    python/tests/
    ├── conftest.py                         # Shared fixtures and test configuration
    ├── test_core/
    │   ├── test_mcts.py                   # Core MCTS functionality tests
    │   ├── test_tree_operations.py        # Tree operations and subtree reuse
    │   ├── test_wave_search.py            # Wave-based parallelization tests
    │   ├── test_game_interface.py         # Game interface and state conversions
    │   └── test_evaluator.py              # Evaluator interface tests
    ├── test_gpu/
    │   ├── test_csr_tree.py              # CSR tree structure tests
    │   ├── test_mcts_gpu_accelerator.py  # GPU acceleration tests
    │   ├── test_node_data_manager.py     # Node data management tests
    │   ├── test_csr_storage.py           # CSR storage tests
    │   ├── test_ucb_selector.py          # UCB selection tests
    │   ├── test_memory_pool.py           # Memory pool management tests
    │   └── test_gpu_game_states.py       # GPU game state management
    ├── test_neural_networks/
    │   ├── test_resnet_model.py          # ResNet model architecture tests
    │   ├── test_resnet_evaluator.py      # Neural network evaluator tests
    │   ├── test_self_play_module.py      # Self-play data generation tests
    │   ├── test_arena_module.py          # Arena evaluation tests
    │   └── test_training_pipeline.py     # Complete training pipeline tests
    ├── test_utils/
    │   ├── test_batch_coordinator.py     # Batch evaluation coordination tests
    │   ├── test_config_system.py         # Configuration management tests
    │   ├── test_remote_evaluator.py      # Remote evaluator optimization tests
    │   ├── test_gpu_evaluator_service.py # GPU service tests
    │   └── test_validation.py            # Tree validation tests
    ├── test_integration/
    │   ├── test_mcts_integration.py      # End-to-end MCTS workflow tests
    │   ├── test_training_integration.py  # Complete training cycle tests
    │   ├── test_multiprocessing.py       # Multiprocessing scenarios
    │   └── test_performance.py           # Performance benchmark tests
    └── test_games/
        ├── test_gomoku_gameplay.py        # Gomoku-specific gameplay tests
        ├── test_go_gameplay.py            # Go-specific gameplay tests
        └── test_chess_gameplay.py         # Chess-specific gameplay tests
   

    Test Coverage Details

    1. Core MCTS Tests (test_core/test_mcts.py)

    - Tree initialization and root node creation
    - Selection phase with UCB formula verification
    - Expansion with progressive widening
    - Evaluation with mock evaluator
    - Backpropagation with value negation
    - Virtual loss application and removal
    - Dirichlet noise application
    - Policy extraction and normalization
    - Temperature-based action selection
    - Tree reuse functionality
    - State management and cleanup
    - Statistics tracking

    2. Wave Search Tests (test_core/test_wave_search.py)

    - Wave-based parallel selection
    - Batch expansion handling
    - Vectorized evaluation
    - Scatter-based backpropagation
    - Per-simulation Dirichlet noise
    - Buffer allocation and management
    - Progressive widening in expansion
    - Cross-simulation coordination

    3. CSR Tree Tests (test_gpu/test_csr_tree.py)

    - Tree structure initialization
    - Node addition (single and batch)
    - Children retrieval and lookup
    - Edge storage in CSR format
    - Memory growth and reallocation
    - Shift root operation
    - Tree reset and cleanup
    - Batch operations
    - Statistics and memory usage

    4. GPU Acceleration Tests (test_gpu/test_mcts_gpu_accelerator.py)

    - CUDA kernel detection and loading
    - Batched UCB selection
    - Vectorized backup operations
    - Find expansion nodes
    - Fallback to PyTorch implementations
    - Multi-device support
    - Performance comparisons

    5. Batch Coordination Tests (test_utils/test_batch_coordinator.py)

    - Request batching across workers
    - Timeout handling
    - Cross-worker coordination
    - Immediate vs coordinated processing
    - Multiprocessing context detection
    - Statistics tracking
    - Error handling and recovery

    6. Training Pipeline Tests (test_neural_networks/test_training_pipeline.py)

    - Self-play data generation
    - Neural network training loop
    - Arena evaluation integration
    - Checkpoint saving/loading
    - Resume from checkpoint
    - Data augmentation
    - Replay buffer management
    - Mixed precision training

    7. Integration Tests (test_integration/)


    - Complete MCTS search workflow
    - Multi-game self-play sessions
    - Training iteration cycles
    - Multiprocessing scenarios
    - Performance benchmarks
    - Memory leak detection
    - Thread safety verification

    Test Implementation Strategy

    1. Fixtures: Create reusable fixtures in conftest.py for:
      - Mock evaluators with various behaviors
      - Game interfaces for different games
      - Pre-configured MCTS instances
      - Test game states and positions
      - GPU device management
    2. Parametrization: Use pytest parametrization for:
      - Different game types (Gomoku, Go, Chess)
      - Various board sizes
      - CPU vs GPU execution
      - Different batch sizes
      - Configuration variations
    3. Mocking: Mock external dependencies:
      - CUDA kernels (when not available)
      - Neural network models
      - Multiprocessing components
      - File I/O operations
    4. Performance Tests: Include benchmarks for:
      - Simulations per second
      - Batch processing throughput
      - Memory usage patterns
      - GPU utilization
    5. Edge Cases: Test handling of:
      - Empty trees
      - Full trees (max nodes)
      - Invalid moves
      - Terminal positions
      - Concurrent access
      - Resource exhaustion

    Each test module will comprehensively test all public methods and critical internal functions, with emphasis on correctness, performance, and robustness. The tests will use the existing mock_evaluator.py as a foundation and extend it with additional mocking utilities as needed.
