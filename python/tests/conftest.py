"""
Shared fixtures and utilities for MCTS test suite

This module provides common test fixtures, utilities, and configuration
that are shared across all test modules.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

# Add project root to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import project modules
from mcts.core.game_interface import GameInterface, GameType
from mcts.core.mcts_config import MCTSConfig
from mcts.core.evaluator import Evaluator
from mcts.gpu.gpu_game_states import GameType as GPUGameType
from mcts.utils.config_system import AlphaZeroConfig, create_default_config
from tests.mock_evaluator import MockEvaluator, DeterministicMockEvaluator

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ============= Device Management =============

@pytest.fixture(scope="session")
def device(request):
    """Provide device for tests (CPU by default, GPU if available and requested)"""
    if torch.cuda.is_available() and not request.config.getoption("--cpu-only", default=False):
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture(scope="session") 
def cuda_available():
    """Check if CUDA is available for tests"""
    return torch.cuda.is_available()


# ============= Game Interfaces =============

@pytest.fixture
def gomoku_game():
    """Create Gomoku game interface"""
    return GameInterface(GameType.GOMOKU, board_size=15, input_representation='basic')


@pytest.fixture
def go_game():
    """Create Go game interface"""
    return GameInterface(GameType.GO, board_size=19, input_representation='basic')


@pytest.fixture
def chess_game():
    """Create Chess game interface"""
    return GameInterface(GameType.CHESS, input_representation='basic')


@pytest.fixture(params=[GameType.GOMOKU, GameType.GO, GameType.CHESS])
def any_game(request):
    """Parametrized fixture for testing with all game types"""
    game_type = request.param
    if game_type == GameType.GOMOKU:
        board_size = 15
    elif game_type == GameType.GO:
        board_size = 19
    else:
        board_size = None
    return GameInterface(game_type, board_size=board_size, input_representation='basic')


# ============= Game States =============

@pytest.fixture
def empty_gomoku_state(gomoku_game):
    """Create empty Gomoku state"""
    return gomoku_game.create_initial_state()


@pytest.fixture
def sample_gomoku_position(gomoku_game, empty_gomoku_state):
    """Create a sample Gomoku position with some moves played"""
    state = empty_gomoku_state
    # Play some moves in the center area
    moves = [112, 113, 127, 128, 97, 98]  # Center area moves
    for i, move in enumerate(moves):
        if move in gomoku_game.get_legal_moves(state):
            state = gomoku_game.apply_move(state, move)
    return state


# ============= Mock Evaluators =============

@pytest.fixture
def mock_evaluator(device):
    """Create basic mock evaluator"""
    return MockEvaluator(game_type='gomoku', device=str(device))


@pytest.fixture
def deterministic_evaluator(device):
    """Create deterministic mock evaluator for reproducible tests"""
    return DeterministicMockEvaluator(game_type='gomoku', device=str(device))


@pytest.fixture
def mock_evaluator_factory(device):
    """Factory for creating mock evaluators with custom settings"""
    def _create_evaluator(game_type='gomoku', deterministic=False, fixed_value=0.0):
        if deterministic:
            evaluator = DeterministicMockEvaluator(game_type=game_type, device=str(device))
            evaluator.fixed_value = fixed_value
            return evaluator
        return MockEvaluator(game_type=game_type, device=str(device))
    return _create_evaluator


# ============= MCTS Configuration =============

@pytest.fixture
def base_mcts_config(device):
    """Create base MCTS configuration for tests"""
    config = MCTSConfig()
    config.device = str(device)
    config.num_simulations = 100
    config.max_tree_nodes = 10000
    config.max_children_per_node = 225
    config.c_puct = 1.4
    config.dirichlet_alpha = 0.3
    config.dirichlet_epsilon = 0.25
    config.temperature = 1.0
    config.enable_subtree_reuse = True
    config.enable_virtual_loss = True
    config.virtual_loss = 1.0
    config.board_size = 15
    config.game_type = GPUGameType.GOMOKU
    config.max_wave_size = 32
    config.enable_fast_ucb = True
    config.classical_only_mode = True  # Disable quantum features
    return config


@pytest.fixture
def small_mcts_config(base_mcts_config):
    """Create small MCTS configuration for fast tests"""
    config = base_mcts_config
    config.num_simulations = 10
    config.max_tree_nodes = 1000
    config.max_wave_size = 8
    return config


@pytest.fixture
def alphazero_config():
    """Create full AlphaZero configuration for integration tests"""
    config = create_default_config()
    config.experiment_name = "test_experiment"
    config.game.game_type = "gomoku"
    config.game.board_size = 15
    config.mcts.num_simulations = 50
    config.mcts.max_tree_nodes = 5000
    config.training.num_games_per_iteration = 2
    config.training.num_workers = 1
    config.training.batch_size = 16
    config.training.num_epochs = 1
    config.arena.num_games = 2
    config.arena.num_workers = 1
    return config


# ============= Test Data Generation =============

@pytest.fixture
def sample_board_tensor(device):
    """Create sample board tensor for testing"""
    # Create 18-channel tensor for Gomoku (basic representation)
    tensor = torch.zeros(18, 15, 15, device=device)
    # Add some stones
    tensor[0, 7, 7] = 1.0  # Player 1 stone
    tensor[1, 7, 8] = 1.0  # Player 2 stone
    tensor[0, 8, 7] = 1.0  # Player 1 stone
    # Current player indicator
    tensor[2:10, :, :] = 0.0  # Move history planes
    tensor[10:18, :, :] = 0.0  # Move history for player 2
    return tensor


@pytest.fixture
def batch_board_tensors(device):
    """Create batch of board tensors for testing"""
    batch_size = 4
    tensors = torch.zeros(batch_size, 18, 15, 15, device=device)
    # Add different patterns to each board
    for i in range(batch_size):
        # Add some random stones
        num_stones = np.random.randint(1, 5)
        for _ in range(num_stones):
            row, col = np.random.randint(0, 15, size=2)
            player = np.random.randint(0, 2)
            tensors[i, player, row, col] = 1.0
    return tensors


# ============= Temporary Directories =============

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    # Cleanup after test
    shutil.rmtree(temp_path)


@pytest.fixture
def temp_checkpoint_dir(temp_dir):
    """Create temporary checkpoint directory"""
    checkpoint_dir = temp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir


# ============= Test Utilities =============

class MCTSTestHelper:
    """Helper class for MCTS testing utilities"""
    
    @staticmethod
    def create_test_tree_nodes(tree, num_nodes: int, branching_factor: int = 3):
        """Create a test tree with specified number of nodes"""
        if num_nodes <= 1:
            return
            
        nodes_to_expand = [0]  # Start with root
        created_nodes = 1
        
        while created_nodes < num_nodes and nodes_to_expand:
            parent_idx = nodes_to_expand.pop(0)
            
            # Add children
            num_children = min(branching_factor, num_nodes - created_nodes)
            actions = list(range(num_children))
            priors = [1.0 / num_children] * num_children
            
            child_indices = tree.add_children_batch(parent_idx, actions, priors)
            
            nodes_to_expand.extend(child_indices)
            created_nodes += num_children
    
    @staticmethod
    def simulate_mcts_iterations(mcts, state, num_iterations: int):
        """Run MCTS iterations and return statistics"""
        initial_nodes = mcts.tree.num_nodes
        
        for _ in range(num_iterations):
            mcts.search(state, num_simulations=1)
            
        return {
            'nodes_created': mcts.tree.num_nodes - initial_nodes,
            'root_visits': mcts.tree.node_data.visit_counts[0].item(),
            'root_value': mcts.get_root_value()
        }
    
    @staticmethod
    def verify_tree_consistency(tree):
        """Verify tree structure consistency"""
        issues = []
        
        # Check node count
        if tree.num_nodes > tree.config.max_nodes and tree.config.max_nodes > 0:
            issues.append(f"Node count {tree.num_nodes} exceeds max {tree.config.max_nodes}")
            
        # Check parent-child relationships
        for node_idx in range(tree.num_nodes):
            if node_idx == 0:  # Root
                if tree.node_data.parent_indices[0] != -1:
                    issues.append("Root node has non-null parent")
            else:
                parent_idx = tree.node_data.parent_indices[node_idx].item()
                if parent_idx < 0 or parent_idx >= node_idx:
                    issues.append(f"Node {node_idx} has invalid parent {parent_idx}")
                    
        return issues
    
    @staticmethod
    def check_memory_usage(obj, max_mb: float = 1000.0):
        """Check if object memory usage is within limits"""
        if hasattr(obj, 'get_memory_usage'):
            memory_stats = obj.get_memory_usage()
            total_mb = memory_stats.get('total_mb', 0)
            return total_mb <= max_mb, total_mb
        return True, 0.0


@pytest.fixture
def test_helper():
    """Provide test helper utilities"""
    return MCTSTestHelper()


# ============= Performance Tracking =============

class PerformanceTracker:
    """Track performance metrics during tests"""
    
    def __init__(self):
        self.metrics = {}
        
    def record(self, name: str, value: float):
        """Record a performance metric"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        if name not in self.metrics:
            return {}
            
        values = self.metrics[name]
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }


@pytest.fixture
def perf_tracker():
    """Provide performance tracking"""
    return PerformanceTracker()


# ============= Pytest Configuration =============

def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--cpu-only", 
        action="store_true", 
        default=False,
        help="Run tests on CPU only, even if GPU is available"
    )
    parser.addoption(
        "--slow", 
        action="store_true", 
        default=False,
        help="Run slow tests (performance benchmarks, large simulations)"
    )
    parser.addoption(
        "--integration", 
        action="store_true", 
        default=False,
        help="Run integration tests"
    )


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "multiprocessing: marks tests that use multiprocessing")
    config.addinivalue_line("markers", "benchmark: marks tests as performance benchmarks")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options"""
    skip_slow = pytest.mark.skip(reason="need --slow option to run")
    skip_integration = pytest.mark.skip(reason="need --integration option to run")
    skip_gpu = pytest.mark.skip(reason="GPU not available or --cpu-only specified")
    
    for item in items:
        # Skip slow tests unless requested
        if "slow" in item.keywords and not config.getoption("--slow"):
            item.add_marker(skip_slow)
            
        # Skip integration tests unless requested
        if "integration" in item.keywords and not config.getoption("--integration"):
            item.add_marker(skip_integration)
            
        # Skip GPU tests if requested or GPU not available
        if "gpu" in item.keywords:
            if config.getoption("--cpu-only") or not torch.cuda.is_available():
                item.add_marker(skip_gpu)


# ============= Assertion Helpers =============

def assert_tensor_equal(actual: torch.Tensor, expected: torch.Tensor, tolerance: float = 1e-6):
    """Assert two tensors are equal within tolerance"""
    assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} vs {expected.shape}"
    assert torch.allclose(actual, expected, atol=tolerance), \
        f"Tensor values differ by more than {tolerance}"


def assert_valid_policy(policy: np.ndarray, legal_mask: Optional[np.ndarray] = None):
    """Assert policy is valid probability distribution"""
    assert len(policy.shape) == 1, f"Policy should be 1D, got shape {policy.shape}"
    assert np.all(policy >= 0), "Policy contains negative values"
    assert np.abs(policy.sum() - 1.0) < 1e-6, f"Policy doesn't sum to 1: {policy.sum()}"
    
    if legal_mask is not None:
        illegal_probs = policy[~legal_mask]
        if not np.all(illegal_probs < 1e-6):
            # Debug output
            illegal_indices = np.where(~legal_mask)[0]
            bad_indices = illegal_indices[illegal_probs >= 1e-6]
            print(f"Policy shape: {policy.shape}")
            print(f"Legal mask shape: {legal_mask.shape}")
            print(f"Number of legal moves: {legal_mask.sum()}")
            print(f"Number of illegal moves: {(~legal_mask).sum()}")
            print(f"Illegal moves with non-zero probability: {bad_indices[:10]}")
            print(f"Their probabilities: {policy[bad_indices[:10]]}")
        assert np.all(illegal_probs < 1e-6), "Policy has non-zero probability for illegal moves"


def assert_valid_value(value: float):
    """Assert value is in valid range [-1, 1]"""
    assert -1.0 <= value <= 1.0, f"Value {value} outside valid range [-1, 1]"