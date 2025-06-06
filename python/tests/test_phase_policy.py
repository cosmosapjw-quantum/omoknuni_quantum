"""Tests for phase-kicked prior policy"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict

from mcts.quantum.phase_policy import PhaseKickedPolicy, PhaseConfig
from mcts.core.node import Node
from mcts.core.evaluator import MockEvaluator, EvaluatorConfig


class TestPhaseConfig:
    """Test PhaseConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = PhaseConfig()
        assert config.base_phase_strength == 0.1
        assert config.uncertainty_scaling == 1.0
        assert config.value_phase_coupling == 0.5
        assert config.enable_interference_patterns is True
        assert config.phase_decay_rate == 0.95
        
    def test_custom_config(self):
        """Test custom configuration"""
        custom = PhaseConfig(
            base_phase_strength=0.2,
            uncertainty_scaling=2.0,
            value_phase_coupling=0.8,
            enable_interference_patterns=False,
            phase_decay_rate=0.9
        )
        assert custom.base_phase_strength == 0.2
        assert custom.uncertainty_scaling == 2.0
        assert custom.value_phase_coupling == 0.8
        assert custom.enable_interference_patterns is False
        assert custom.phase_decay_rate == 0.9


class TestPhaseKickedPolicy:
    """Test suite for phase-kicked policy"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return PhaseConfig(
            base_phase_strength=0.5,
            uncertainty_scaling=1.0,
            value_phase_coupling=0.5,
            enable_interference_patterns=True,
            phase_decay_rate=0.95
        )
        
    @pytest.fixture
    def policy(self, config):
        """Create test policy"""
        return PhaseKickedPolicy(config)
        
    @pytest.fixture
    def evaluator(self):
        """Create mock evaluator"""
        return MockEvaluator(EvaluatorConfig(), action_size=100)
        
    @pytest.fixture
    def test_node(self):
        """Create test node with children"""
        root = Node(state="root", parent=None, action=None, prior=1.0)
        root.visit_count = 10
        
        # Add children with varying values
        for i in range(5):
            child = Node(state=f"child_{i}", parent=root, action=i, prior=0.2)
            child.visit_count = i + 1
            child.value_sum = (i - 2) * 0.1  # Varying values
            root.children[i] = child
            
        return root
        
    def test_initialization(self, config):
        """Test phase policy initialization"""
        policy = PhaseKickedPolicy(config)
        
        assert policy.config == config
        assert len(policy.phase_history) == 0
        assert isinstance(policy.phase_history, dict)
        
    def test_apply_phase_kicks_empty(self, policy, evaluator):
        """Test phase kicks with empty action probabilities"""
        node = Node(state="test", parent=None, action=None, prior=1.0)
        empty_probs = {}
        
        result = policy.apply_phase_kicks(node, empty_probs, evaluator)
        assert result == {}
        
    def test_apply_phase_kicks_basic(self, policy, test_node, evaluator):
        """Test basic phase kick application"""
        # Original probabilities
        action_probs = {i: 0.2 for i in range(5)}
        
        # Apply phase kicks
        modified_probs = policy.apply_phase_kicks(
            test_node, action_probs, evaluator
        )
        
        # Check properties
        assert len(modified_probs) == 5
        assert all(0 <= p <= 1 for p in modified_probs.values())
        assert abs(sum(modified_probs.values()) - 1.0) < 1e-6
        
        # Should be different from original (due to phases)
        assert any(
            abs(modified_probs[i] - 0.2) > 1e-6 
            for i in range(5)
        )
        
        # Phase history should be stored
        assert id(test_node) in policy.phase_history
        assert len(policy.phase_history[id(test_node)]) == 5
        
    def test_apply_phase_kicks_with_state_features(self, policy, test_node, evaluator):
        """Test phase kicks with state features"""
        action_probs = {i: 0.2 for i in range(5)}
        state_features = np.random.randn(10)
        
        modified_probs = policy.apply_phase_kicks(
            test_node, action_probs, evaluator, state_features
        )
        
        assert len(modified_probs) == 5
        assert abs(sum(modified_probs.values()) - 1.0) < 1e-6
        
    def test_uncertainty_estimation(self, policy, evaluator):
        """Test uncertainty estimation"""
        # High uncertainty: new node without children
        node1 = Node(state="new", parent=None, action=None, prior=1.0)
        node1.visit_count = 0
        uncertainty1 = policy._estimate_uncertainty(node1, evaluator)
        assert 0.8 < uncertainty1 <= 1.0
        
        # Low uncertainty: well-visited node with consistent children
        node2 = Node(state="visited", parent=None, action=None, prior=1.0)
        node2.visit_count = 1000
        
        for i in range(5):
            child = Node(state=f"child_{i}", parent=node2, action=i, prior=0.2)
            child.value_sum = 5.0  # Same value
            child.visit_count = 100
            node2.children[i] = child
            
        uncertainty2 = policy._estimate_uncertainty(node2, evaluator)
        assert uncertainty2 < 0.3
        
        # Medium uncertainty: moderate visits with variance
        node3 = Node(state="mixed", parent=None, action=None, prior=1.0)
        node3.visit_count = 50
        
        for i in range(5):
            child = Node(state=f"child_{i}", parent=node3, action=i, prior=0.2)
            child.value_sum = i * 2.0  # Varying values
            child.visit_count = 10
            node3.children[i] = child
            
        uncertainty3 = policy._estimate_uncertainty(node3, evaluator)
        # Uncertainty should be moderate (not too low, not too high)
        assert 0.05 < uncertainty3 < 0.7
        
        # Edge case: single child
        node4 = Node(state="single", parent=None, action=None, prior=1.0)
        node4.visit_count = 100
        child = Node(state="only_child", parent=node4, action=0, prior=1.0)
        child.value_sum = 1.0
        child.visit_count = 50
        node4.children[0] = child
        
        uncertainty4 = policy._estimate_uncertainty(node4, evaluator)
        assert 0 < uncertainty4 < 1
        
    def test_compute_action_phases(self, policy):
        """Test phase computation for actions"""
        node = Node(state="test", parent=None, action=None, prior=1.0)
        node.visit_count = 10
        
        # Add some children for value-dependent phase
        for i in [1, 3]:
            child = Node(state=f"child_{i}", parent=node, action=i, prior=0.3)
            child.value_sum = i * 0.1
            child.visit_count = 5
            node.children[i] = child
        
        actions = [0, 1, 2, 3, 4]
        uncertainty = 0.5
        
        phases = policy._compute_action_phases(
            node, actions, uncertainty, None
        )
        
        assert len(phases) == 5
        assert all(isinstance(p, (float, np.floating)) for p in phases)
        assert all(np.isfinite(phases))
        
        # Phases should be different for different actions
        assert len(set(phases)) > 1
        
        # Test with high visit count (decay effect)
        node.visit_count = 1000
        phases_high_visit = policy._compute_action_phases(
            node, actions, uncertainty, None
        )
        
        # Should have smaller magnitude due to decay
        assert np.abs(phases_high_visit).mean() < np.abs(phases).mean()
        
        # Test with zero uncertainty
        phases_no_uncertainty = policy._compute_action_phases(
            node, actions, 0.0, None
        )
        assert all(np.isfinite(phases_no_uncertainty))
        
    def test_add_interference_patterns(self, policy):
        """Test interference pattern generation"""
        # Test with interference enabled
        phases = np.linspace(0, 2*np.pi, 10)
        uncertainty = 0.8
        
        phases_with_interference = policy._add_interference_patterns(
            phases.copy(), uncertainty
        )
        
        # Should be modified
        assert not np.array_equal(phases, phases_with_interference)
        assert all(np.isfinite(phases_with_interference))
        
        # Test with single phase (no interference)
        single_phase = np.array([1.0])
        single_with_interference = policy._add_interference_patterns(
            single_phase.copy(), uncertainty
        )
        assert np.array_equal(single_phase, single_with_interference)
        
        # Test with zero uncertainty
        phases_zero_uncertainty = policy._add_interference_patterns(
            phases.copy(), 0.0
        )
        assert np.array_equal(phases, phases_zero_uncertainty)
        
    def test_apply_phases_to_probs(self, policy):
        """Test applying complex phases to probabilities"""
        # Test uniform probabilities
        action_probs = {i: 0.2 for i in range(5)}
        
        # Test with zero phases (should preserve probabilities)
        phases_zero = np.zeros(5)
        probs_zero = policy._apply_phases_to_probs(
            action_probs, phases_zero, 0.0
        )
        assert all(abs(probs_zero[i] - 0.2) < 1e-6 for i in range(5))
        
        # Test with random phases and interference
        phases_random = np.random.uniform(0, 2*np.pi, 5)
        probs_random = policy._apply_phases_to_probs(
            action_probs, phases_random, 0.5
        )
        
        # Check normalization
        assert abs(sum(probs_random.values()) - 1.0) < 1e-6
        assert all(0 <= p <= 1 for p in probs_random.values())
        
        # Test single action (no interference possible)
        single_action = {0: 1.0}
        single_phase = np.array([np.pi])
        probs_single = policy._apply_phases_to_probs(
            single_action, single_phase, 0.5
        )
        assert abs(probs_single[0] - 1.0) < 1e-6
        
        # Test with very skewed probabilities
        skewed_probs = {0: 0.9, 1: 0.05, 2: 0.05}
        phases_skewed = np.array([0, np.pi, np.pi/2])
        probs_skewed = policy._apply_phases_to_probs(
            skewed_probs, phases_skewed, 0.8
        )
        assert abs(sum(probs_skewed.values()) - 1.0) < 1e-6
        
    def test_compute_phase_entropy(self, policy):
        """Test phase entropy computation"""
        node = Node(state="test", parent=None, action=None, prior=1.0)
        node_id = id(node)
        
        # No phase history
        entropy0 = policy.compute_phase_entropy(node)
        assert entropy0 == 0.0
        
        # Empty phase array
        policy.phase_history[node_id] = np.array([])
        entropy_empty = policy.compute_phase_entropy(node)
        assert entropy_empty == 0.0
        
        # Uniform phases (high entropy)
        policy.phase_history[node_id] = np.linspace(0, 2*np.pi, 100)
        entropy_uniform = policy.compute_phase_entropy(node)
        assert entropy_uniform > 2.0
        
        # Concentrated phases (low entropy)
        policy.phase_history[node_id] = np.ones(100) * np.pi
        entropy_concentrated = policy.compute_phase_entropy(node)
        # Allow for small numerical errors
        assert -1e-8 <= entropy_concentrated < 1.0
        
        # Random phases
        policy.phase_history[node_id] = np.random.uniform(0, 2*np.pi, 100)
        entropy_random = policy.compute_phase_entropy(node)
        assert 0 < entropy_random < np.log(16)  # Max entropy for 16 bins
        
    def test_get_interference_strength(self, policy):
        """Test interference strength between nodes"""
        node1 = Node(state="node1", parent=None, action=None, prior=1.0)
        node2 = Node(state="node2", parent=None, action=None, prior=1.0)
        node3 = Node(state="node3", parent=None, action=None, prior=1.0)
        
        # No phase history
        strength_none = policy.get_interference_strength(node1, node2)
        assert strength_none == 0.0
        
        # Same phases = high interference
        phases_same = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        policy.phase_history[id(node1)] = phases_same
        policy.phase_history[id(node2)] = phases_same
        
        strength_same = policy.get_interference_strength(node1, node2)
        assert strength_same > 0.95
        
        # Opposite phases = low interference
        policy.phase_history[id(node2)] = phases_same + np.pi
        strength_opposite = policy.get_interference_strength(node1, node2)
        assert strength_opposite < 0.1
        
        # Different lengths
        policy.phase_history[id(node3)] = np.array([0, np.pi])
        strength_diff_len = policy.get_interference_strength(node1, node3)
        assert 0 <= strength_diff_len <= 1
        
        # Empty phases
        policy.phase_history[id(node1)] = np.array([])
        strength_empty = policy.get_interference_strength(node1, node2)
        assert strength_empty == 0.0
        
        # Phase wrap-around test
        phases_wrap1 = np.array([0.1, 2*np.pi - 0.1])
        phases_wrap2 = np.array([2*np.pi - 0.1, 0.1])
        policy.phase_history[id(node1)] = phases_wrap1
        policy.phase_history[id(node2)] = phases_wrap2
        strength_wrap = policy.get_interference_strength(node1, node2)
        assert strength_wrap > 0.9  # Should be similar despite wrap
        
    def test_visualize_phase_field(self, policy, test_node, evaluator):
        """Test phase field visualization data generation"""
        action_probs = {i: 0.1 * (i + 1) for i in range(5)}
        
        # Without phase history
        viz_data_no_history = policy.visualize_phase_field(test_node, action_probs)
        assert 'actions' in viz_data_no_history
        assert 'probabilities' in viz_data_no_history
        assert 'phases' in viz_data_no_history
        assert 'complex_field' in viz_data_no_history
        assert all(viz_data_no_history['phases'] == 0)
        
        # Apply phase kicks first to generate history
        policy.apply_phase_kicks(test_node, action_probs, evaluator)
        
        # Get visualization data
        viz_data = policy.visualize_phase_field(test_node, action_probs)
        
        assert len(viz_data['actions']) == 5
        assert len(viz_data['probabilities']) == 5
        assert len(viz_data['phases']) == 5
        assert len(viz_data['complex_field']) == 5
        
        # Check data types
        assert all(isinstance(a, (int, np.integer)) for a in viz_data['actions'])
        assert all(isinstance(p, (float, np.floating)) for p in viz_data['probabilities'])
        assert all(isinstance(c, (complex, np.complexfloating)) for c in viz_data['complex_field'])
        
        # Check phase gradient (for multiple actions)
        assert 'phase_gradient' in viz_data
        assert len(viz_data['phase_gradient']) == 5
        
        # Test empty action probs
        viz_empty = policy.visualize_phase_field(test_node, {})
        assert viz_empty == {}
        
        # Test single action (no gradient)
        single_action_probs = {0: 1.0}
        policy.phase_history[id(test_node)] = np.array([1.5])
        viz_single = policy.visualize_phase_field(test_node, single_action_probs)
        assert 'phase_gradient' not in viz_single
        
    def test_phase_decay_effect(self, policy, evaluator):
        """Test phase decay with visit count"""
        action_probs = {0: 0.5, 1: 0.5}
        
        # Low visit count
        node_low = Node(state="low", parent=None, action=None, prior=1.0)
        node_low.visit_count = 10
        for i in range(2):
            child = Node(state=f"child_{i}", parent=node_low, action=i, prior=0.5)
            child.value_sum = i * 0.2
            child.visit_count = 5
            node_low.children[i] = child
        
        probs_low = policy.apply_phase_kicks(node_low, action_probs, evaluator)
        diff_low = abs(probs_low[0] - probs_low[1])
        
        # High visit count (same structure)
        node_high = Node(state="high", parent=None, action=None, prior=1.0)
        node_high.visit_count = 1000
        for i in range(2):
            child = Node(state=f"child_{i}", parent=node_high, action=i, prior=0.5)
            child.value_sum = i * 0.2
            child.visit_count = 500
            node_high.children[i] = child
        
        probs_high = policy.apply_phase_kicks(node_high, action_probs, evaluator)
        diff_high = abs(probs_high[0] - probs_high[1])
        
        # High visit count should have less phase effect
        assert diff_high < diff_low
        
    def test_edge_cases(self, policy, evaluator):
        """Test edge cases and boundary conditions"""
        # Node with no parent, no children
        isolated_node = Node(state="isolated", parent=None, action=None, prior=1.0)
        isolated_node.visit_count = 0
        
        action_probs = {0: 0.3, 1: 0.7}
        result = policy.apply_phase_kicks(isolated_node, action_probs, evaluator)
        
        assert len(result) == 2
        assert abs(sum(result.values()) - 1.0) < 1e-6
        
        # Very large action space
        large_action_probs = {i: 1.0/1000 for i in range(1000)}
        node_large = Node(state="large", parent=None, action=None, prior=1.0)
        result_large = policy.apply_phase_kicks(
            node_large, large_action_probs, evaluator
        )
        
        assert len(result_large) == 1000
        assert abs(sum(result_large.values()) - 1.0) < 1e-5
        
        # Extreme probability distribution
        extreme_probs = {0: 0.99999, 1: 0.00001}
        result_extreme = policy.apply_phase_kicks(
            isolated_node, extreme_probs, evaluator
        )
        assert 0 <= result_extreme[0] <= 1
        assert 0 <= result_extreme[1] <= 1
        assert abs(sum(result_extreme.values()) - 1.0) < 1e-6
        
    def test_config_variations(self, evaluator):
        """Test different configuration settings"""
        # No interference patterns
        config_no_interference = PhaseConfig(
            base_phase_strength=1.0,
            enable_interference_patterns=False
        )
        policy_no_interference = PhaseKickedPolicy(config_no_interference)
        
        node = Node(state="test", parent=None, action=None, prior=1.0)
        action_probs = {i: 0.25 for i in range(4)}
        
        result = policy_no_interference.apply_phase_kicks(
            node, action_probs, evaluator
        )
        assert len(result) == 4
        
        # Zero phase strength (should preserve original)
        config_zero = PhaseConfig(base_phase_strength=0.0)
        policy_zero = PhaseKickedPolicy(config_zero)
        
        result_zero = policy_zero.apply_phase_kicks(
            node, action_probs, evaluator
        )
        assert all(abs(result_zero[i] - 0.25) < 1e-6 for i in range(4))
        
        # Very strong phases
        config_strong = PhaseConfig(
            base_phase_strength=10.0,
            uncertainty_scaling=5.0,
            value_phase_coupling=2.0
        )
        policy_strong = PhaseKickedPolicy(config_strong)
        
        result_strong = policy_strong.apply_phase_kicks(
            node, action_probs, evaluator
        )
        assert abs(sum(result_strong.values()) - 1.0) < 1e-6