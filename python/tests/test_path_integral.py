"""Tests for path integral MCTS formulation"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import List

from mcts.quantum.path_integral import (
    PathIntegralConfig, Path, PathIntegralMCTS
)
from mcts.core.node import Node


class TestPathIntegralConfig:
    """Test PathIntegralConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = PathIntegralConfig()
        assert config.temperature == 1.0
        assert config.phase_coupling == 0.1
        assert config.regularization == 0.01
        assert config.use_complex_action is True
        assert config.max_path_length == 50
        
    def test_custom_config(self):
        """Test custom configuration"""
        config = PathIntegralConfig(
            temperature=2.0,
            phase_coupling=0.2,
            regularization=0.05,
            use_complex_action=False,
            max_path_length=30
        )
        assert config.temperature == 2.0
        assert config.phase_coupling == 0.2
        assert config.regularization == 0.05
        assert config.use_complex_action is False
        assert config.max_path_length == 30


class TestPath:
    """Test Path class"""
    
    def test_path_initialization(self):
        """Test path initialization"""
        nodes = [Mock(spec=Node) for _ in range(3)]
        actions = [0, 1]
        
        path = Path(nodes, actions)
        assert path.nodes == nodes
        assert path.actions == actions
        assert path.length == 3
        assert path._action_value is None
        assert path._probability_amplitude is None
        
    def test_get_leaf(self):
        """Test getting leaf node"""
        nodes = [Mock(spec=Node) for _ in range(3)]
        path = Path(nodes, [0, 1])
        
        assert path.get_leaf() == nodes[-1]
        
        # Empty path
        empty_path = Path([], [])
        assert empty_path.get_leaf() is None
        
    def test_get_visits(self):
        """Test getting total visits"""
        nodes = []
        for i in range(3):
            node = Mock(spec=Node)
            node.visit_count = i + 1  # 1, 2, 3
            nodes.append(node)
            
        path = Path(nodes, [0, 1])
        assert path.get_visits() == 6  # 1 + 2 + 3
        
    def test_get_value(self):
        """Test getting path value"""
        nodes = []
        for i in range(3):
            node = Mock(spec=Node)
            node.visit_count = 1
            node.value.return_value = float(i) * 0.1  # 0.0, 0.1, 0.2
            nodes.append(node)
            
        path = Path(nodes, [0, 1])
        assert np.isclose(path.get_value(), 0.1)  # mean of [0.0, 0.1, 0.2]
        
        # Empty path
        empty_path = Path([], [])
        assert empty_path.get_value() == 0.0
        
        # Path with no visits
        unvisited_nodes = []
        for i in range(2):
            node = Mock(spec=Node)
            node.visit_count = 0
            unvisited_nodes.append(node)
        
        unvisited_path = Path(unvisited_nodes, [0])
        assert unvisited_path.get_value() == 0.0


class TestPathIntegralMCTS:
    """Test PathIntegralMCTS class"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return PathIntegralConfig(
            temperature=1.0,
            phase_coupling=0.1,
            regularization=0.01,
            use_complex_action=True
        )
        
    @pytest.fixture
    def pi_mcts(self, config):
        """Create PathIntegralMCTS instance"""
        return PathIntegralMCTS(config)
        
    @pytest.fixture
    def mock_nodes(self):
        """Create mock nodes for testing"""
        nodes = []
        for i in range(4):
            node = Mock(spec=Node)
            node.visit_count = 10 * (i + 1)  # 10, 20, 30, 40
            node.value.return_value = 0.5 + i * 0.1  # 0.5, 0.6, 0.7, 0.8
            node.is_leaf.return_value = (i == 3)  # Last node is leaf
            nodes.append(node)
            
        # Set up parent-child relationships
        for i in range(3):
            nodes[i].children = {i: nodes[i+1]}
            nodes[i+1].parent = nodes[i]
            nodes[i+1].action = i
            
        nodes[0].parent = None
        return nodes
        
    def test_compute_path_action_real(self, pi_mcts):
        """Test path action computation with real values"""
        config = PathIntegralConfig(use_complex_action=False)
        pi_mcts = PathIntegralMCTS(config)
        
        nodes = []
        for i in range(3):
            node = Mock(spec=Node)
            node.visit_count = 10
            nodes.append(node)
            
        path = Path(nodes, [0, 1])
        
        action = pi_mcts.compute_path_action(path)
        
        # Classical action: -log(30 + 1) + regularization * length
        expected = -np.log(31) + 0.01 * 3
        
        assert isinstance(action, (float, np.floating))
        assert np.isclose(action, expected, rtol=1e-5)
        
        # Test caching
        assert path._action_value is not None
        action2 = pi_mcts.compute_path_action(path)
        assert action == action2
        
    def test_compute_path_action_complex(self, pi_mcts, mock_nodes):
        """Test path action computation with complex values"""
        path = Path(mock_nodes[:3], [0, 1])
        
        action = pi_mcts.compute_path_action(path)
        
        assert isinstance(action, complex)
        assert np.real(action) < 0  # Classical action is negative log
        assert np.isfinite(np.real(action))
        assert np.isfinite(np.imag(action))
        
    def test_compute_path_phase(self, pi_mcts, mock_nodes):
        """Test quantum phase computation"""
        path = Path(mock_nodes[:3], [0, 1])
        
        phase = pi_mcts._compute_path_phase(path)
        
        assert isinstance(phase, float)
        assert np.isfinite(phase)
        
        # Phase should be non-zero due to value changes
        assert phase != 0.0
        
        # Test short path
        short_path = Path([mock_nodes[0]], [])
        assert pi_mcts._compute_path_phase(short_path) == 0.0
        
    def test_compute_path_probability(self, pi_mcts, mock_nodes):
        """Test path probability computation"""
        path = Path(mock_nodes[:3], [0, 1])
        
        weight = pi_mcts.compute_path_probability(path)
        
        assert isinstance(weight, float)
        assert weight > 0  # Weight should be positive
        assert np.isfinite(weight)
        
        # Test with different temperature
        weight_high_temp = pi_mcts.compute_path_probability(path, temperature=10.0)
        weight_low_temp = pi_mcts.compute_path_probability(path, temperature=0.1)
        
        # Temperature affects the weight distribution
        assert weight_low_temp != weight_high_temp
        
    def test_compute_partition_function(self, pi_mcts, mock_nodes):
        """Test partition function computation"""
        paths = []
        for i in range(3):
            path = Path(mock_nodes[:i+2], list(range(i+1)))
            paths.append(path)
            
        z = pi_mcts.compute_partition_function(paths)
        
        assert isinstance(z, complex)
        assert np.abs(z) > 0
        assert np.isfinite(np.real(z))
        assert np.isfinite(np.imag(z))
        
        # Test empty paths
        z_empty = pi_mcts.compute_partition_function([])
        assert z_empty == 0.0 + 0j
        
    def test_generate_paths(self, pi_mcts, mock_nodes):
        """Test path generation"""
        root = mock_nodes[0]
        
        # Mock is_leaf to return False for non-leaf nodes
        root.is_leaf.return_value = False
        mock_nodes[1].is_leaf.return_value = False
        mock_nodes[2].is_leaf.return_value = False
        
        # Set up proper children structure
        root.children = {0: mock_nodes[1]}
        mock_nodes[1].children = {1: mock_nodes[2]}
        mock_nodes[2].children = {2: mock_nodes[3]}
        mock_nodes[3].children = {}
        
        # Mock UCB scores
        for node in mock_nodes:
            node.ucb_score.return_value = np.random.rand()
            
        paths = pi_mcts._generate_paths(root, num_paths=10)
        
        assert isinstance(paths, list)
        assert len(paths) <= 10
        
        for path in paths:
            assert isinstance(path, Path)
            assert len(path.nodes) > 1
            assert path.nodes[0] == root
            assert len(path.actions) == len(path.nodes) - 1
            
    def test_select_path_variational(self, pi_mcts, mock_nodes):
        """Test variational path selection"""
        root = mock_nodes[0]
        
        # Setup for path generation
        root.is_leaf.return_value = False
        root.children = {0: mock_nodes[1]}
        
        # Mock _generate_paths to return predictable paths
        test_paths = [
            Path([root, mock_nodes[1]], [0]),
            Path([root, mock_nodes[1], mock_nodes[2]], [0, 1])
        ]
        
        with patch.object(pi_mcts, '_generate_paths', return_value=test_paths):
            selected_path = pi_mcts.select_path_variational(root, num_paths=2)
            
        assert isinstance(selected_path, Path)
        assert selected_path in test_paths
        
        # Test empty paths case
        with patch.object(pi_mcts, '_generate_paths', return_value=[]):
            empty_path = pi_mcts.select_path_variational(root)
            assert empty_path.nodes == [root]
            assert empty_path.actions == []
            
    def test_compute_effective_action(self, pi_mcts, mock_nodes):
        """Test effective action computation"""
        paths = [
            Path(mock_nodes[:2], [0]),
            Path(mock_nodes[:3], [0, 1])
        ]
        
        eff_action = pi_mcts.compute_effective_action(paths)
        
        assert isinstance(eff_action, float)
        assert np.isfinite(eff_action)
        
        # Test with empty paths
        eff_action_empty = pi_mcts.compute_effective_action([])
        # Should return large finite value instead of inf for numerical stability
        assert eff_action_empty == 700.0  # temperature * 700
        
    def test_compute_path_correlations(self, pi_mcts, mock_nodes):
        """Test path correlation computation"""
        paths = []
        for i in range(3):
            path = Path(mock_nodes[:i+2], list(range(i+1)))
            paths.append(path)
            
        corr_matrix = pi_mcts.compute_path_correlations(paths)
        
        assert isinstance(corr_matrix, np.ndarray)
        assert corr_matrix.shape == (3, 3)
        assert np.allclose(corr_matrix, corr_matrix.T)  # Symmetric
        assert np.allclose(np.diag(corr_matrix), 1.0)  # Unit diagonal
        
        # Test empty paths
        corr_empty = pi_mcts.compute_path_correlations([])
        assert corr_empty.size == 0
        
    def test_get_dominant_paths(self, pi_mcts, mock_nodes):
        """Test getting dominant paths"""
        root = mock_nodes[0]
        
        # Mock path generation
        test_paths = []
        for i in range(5):
            path = Path(mock_nodes[:i%3+1], list(range(i%3)))
            test_paths.append(path)
            
        with patch.object(pi_mcts, '_generate_paths', return_value=test_paths):
            dominant = pi_mcts.get_dominant_paths(root, num_paths=5, top_k=3)
            
        assert isinstance(dominant, list)
        assert len(dominant) == 3
        
        for path, weight in dominant:
            assert isinstance(path, Path)
            assert isinstance(weight, float)
            assert weight >= 0.0
            
        # Check ordering (highest weight first)
        weights = [w for _, w in dominant]
        assert weights == sorted(weights, reverse=True)
        
    def test_compute_quantum_corrections(self, pi_mcts, mock_nodes):
        """Test quantum correction computation"""
        # Setup sibling structure
        parent = mock_nodes[1]
        child1 = mock_nodes[2]
        child2 = Mock(spec=Node)
        child2.action = 10
        child2.visit_count = 15
        child2.value.return_value = 0.6
        
        parent.children = {1: child1, 10: child2}
        
        path = Path(mock_nodes[:3], [0, 1])
        
        corrections = pi_mcts.compute_quantum_corrections(path)
        
        assert isinstance(corrections, dict)
        assert 'phase' in corrections
        assert 'amplitude' in corrections
        assert 'interference' in corrections
        
        assert isinstance(corrections['phase'], float)
        assert isinstance(corrections['amplitude'], float)
        assert isinstance(corrections['interference'], float)
        
        assert corrections['amplitude'] > 0
        assert np.isfinite(corrections['interference'])
        
    def test_edge_cases(self, pi_mcts):
        """Test edge cases and error handling"""
        # Single node path
        single_node = Mock(spec=Node)
        single_node.visit_count = 5
        single_node.value.return_value = 0.5
        
        single_path = Path([single_node], [])
        
        action = pi_mcts.compute_path_action(single_path)
        assert np.isfinite(action)
        
        weight = pi_mcts.compute_path_probability(single_path)
        assert weight > 0
        assert np.isfinite(weight)
        
        # Path with zero visits
        zero_visit_node = Mock(spec=Node)
        zero_visit_node.visit_count = 0
        zero_path = Path([zero_visit_node], [])
        
        action_zero = pi_mcts.compute_path_action(zero_path)
        assert np.isfinite(action_zero)
        
    def test_temperature_effects(self, pi_mcts, mock_nodes):
        """Test temperature parameter effects"""
        path = Path(mock_nodes[:3], [0, 1])
        
        # Test probability weight at different temperatures
        temps = [0.1, 1.0, 10.0]
        weights = []
        
        for temp in temps:
            weight = pi_mcts.compute_path_probability(path, temperature=temp)
            weights.append(weight)
            
        # Temperature should affect the weights
        assert len(set(weights)) == 3  # All weights should be different
        
        # Test partition function temperature dependence
        paths = [path]
        z_values = []
        
        for temp in temps:
            z = pi_mcts.compute_partition_function(paths, temperature=temp)
            z_values.append(np.abs(z))
            
        assert all(z > 0 for z in z_values)