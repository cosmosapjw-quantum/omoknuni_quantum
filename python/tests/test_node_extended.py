"""Extended tests for Node class to improve coverage"""

import pytest
import numpy as np
from unittest.mock import Mock

from mcts.node import Node


class TestNodeExtended:
    """Extended test suite for comprehensive Node coverage"""
    
    def test_node_repr(self):
        """Test string representation"""
        node = Node(state="test_state", parent=None, action=5, prior=0.75)
        node.visit_count = 10
        node.value_sum = 3.5
        
        repr_str = repr(node)
        assert "Node(" in repr_str
        assert "action=5" in repr_str
        assert "visits=10" in repr_str
        assert "value=0.350" in repr_str
        assert "prior=0.750" in repr_str
        assert "children=0" in repr_str
        
    def test_get_improved_policy_empty(self):
        """Test get_improved_policy with no children"""
        node = Node(state=None, parent=None, action=None, prior=1.0)
        
        policy = node.get_improved_policy()
        assert policy == {}
        
        policy_temp0 = node.get_improved_policy(temperature=0)
        assert policy_temp0 == {}
        
    def test_get_improved_policy_temperature_0(self):
        """Test deterministic policy with temperature=0"""
        root = Node(state=None, parent=None, action=None, prior=1.0)
        
        # Create children with different visit counts
        for i in range(3):
            child = Node(state=f"s{i}", parent=root, action=i, prior=0.33)
            child.visit_count = (i + 1) * 10  # 10, 20, 30
            root.children[i] = child
            
        policy = root.get_improved_policy(temperature=0)
        
        # Should select action 2 (most visited)
        assert policy[0] == 0.0
        assert policy[1] == 0.0
        assert policy[2] == 1.0
        
    def test_get_improved_policy_temperature_1(self):
        """Test proportional policy with temperature=1"""
        root = Node(state=None, parent=None, action=None, prior=1.0)
        
        # Create children with different visit counts
        visits = [10, 20, 30]
        for i in range(3):
            child = Node(state=f"s{i}", parent=root, action=i, prior=0.33)
            child.visit_count = visits[i]
            root.children[i] = child
            
        policy = root.get_improved_policy(temperature=1.0)
        
        # Should be proportional to visits
        total_visits = sum(visits)
        for i in range(3):
            expected = visits[i] / total_visits
            assert policy[i] == pytest.approx(expected)
            
    def test_get_improved_policy_high_temperature(self):
        """Test exploration with high temperature"""
        root = Node(state=None, parent=None, action=None, prior=1.0)
        
        # Create children with very different visit counts
        visits = [1, 100]
        for i in range(2):
            child = Node(state=f"s{i}", parent=root, action=i, prior=0.5)
            child.visit_count = visits[i]
            root.children[i] = child
            
        # With high temperature, should be more uniform
        policy = root.get_improved_policy(temperature=10.0)
        
        # The less visited action should have higher probability than with temp=1
        policy_temp1 = root.get_improved_policy(temperature=1.0)
        assert policy[0] > policy_temp1[0]  # Action 0 should have more probability
        
    def test_get_improved_policy_zero_visits(self):
        """Test policy when all children have zero visits"""
        root = Node(state=None, parent=None, action=None, prior=1.0)
        
        # Create children with zero visits
        for i in range(3):
            child = Node(state=f"s{i}", parent=root, action=i, prior=0.33)
            child.visit_count = 0
            root.children[i] = child
            
        policy = root.get_improved_policy(temperature=1.0)
        
        # Should be uniform when no visits
        for action, prob in policy.items():
            assert prob == pytest.approx(1/3)
            
    def test_get_action_values(self):
        """Test getting Q-values for all actions"""
        root = Node(state=None, parent=None, action=None, prior=1.0)
        
        # Create children with different values
        expected_values = {0: 0.5, 1: -0.3, 2: 0.8}
        for action, expected_value in expected_values.items():
            child = Node(state=f"s{action}", parent=root, action=action, prior=0.33)
            child.visit_count = 10
            child.value_sum = expected_value * 10
            root.children[action] = child
            
        action_values = root.get_action_values()
        
        assert len(action_values) == 3
        for action, value in action_values.items():
            assert value == pytest.approx(expected_values[action])
            
    def test_get_action_values_empty(self):
        """Test get_action_values with no children"""
        node = Node(state=None, parent=None, action=None, prior=1.0)
        
        action_values = node.get_action_values()
        assert action_values == {}
        
    def test_quantum_features(self):
        """Test quantum-inspired features initialization"""
        node = Node(state=None, parent=None, action=None, prior=1.0)
        
        # Check quantum features are initialized
        assert hasattr(node, 'phase')
        assert node.phase == 0.0
        assert hasattr(node, 'minhash_signature')
        assert node.minhash_signature is None
        
    def test_expand_already_expanded(self):
        """Test expanding an already expanded node"""
        node = Node(state=None, parent=None, action=None, prior=1.0)
        
        # First expansion
        node.expand({0: 0.5, 1: 0.5}, {0: "s0", 1: "s1"})
        assert node.is_expanded is True
        
        # Second expansion should raise error
        with pytest.raises(ValueError, match="already expanded"):
            node.expand({2: 1.0}, {2: "s2"})
            
    def test_expand_terminal_node(self):
        """Test expanding a terminal node"""
        node = Node(state=None, parent=None, action=None, prior=1.0)
        node.is_terminal = True
        
        with pytest.raises(ValueError, match="terminal"):
            node.expand({0: 1.0}, {0: "s0"})
            
    def test_select_child_no_children(self):
        """Test selecting from node with no children"""
        node = Node(state=None, parent=None, action=None, prior=1.0)
        
        with pytest.raises(ValueError, match="no children"):
            node.select_child(c_puct=1.0)
            
    def test_ucb_score_root_node(self):
        """Test UCB score for root node"""
        root = Node(state=None, parent=None, action=None, prior=1.0)
        
        # Root node should return 0 UCB score
        ucb = root.ucb_score(c_puct=1.0)
        assert ucb == 0.0
        
    def test_terminal_node_is_leaf(self):
        """Test that terminal nodes are always leaves even with children"""
        node = Node(state=None, parent=None, action=None, prior=1.0)
        
        # Add a child
        child = Node(state="child", parent=node, action=0, prior=0.5)
        node.children[0] = child
        
        # Not terminal, so not a leaf
        assert node.is_leaf() is False
        
        # Make it terminal
        node.is_terminal = True
        
        # Now it's a leaf despite having children
        assert node.is_leaf() is True
        
    def test_complex_tree_operations(self):
        """Test operations on a more complex tree"""
        # Build a tree with depth 3
        root = Node(state="root", parent=None, action=None, prior=1.0)
        root.visit_count = 1000
        
        # Level 1
        level1_nodes = []
        for i in range(3):
            child = Node(state=f"L1_{i}", parent=root, action=i, prior=0.33)
            root.children[i] = child
            level1_nodes.append(child)
            
        # Level 2
        level2_nodes = []
        for i, parent in enumerate(level1_nodes):
            for j in range(2):
                action = i * 2 + j
                child = Node(state=f"L2_{action}", parent=parent, action=action, prior=0.5)
                parent.children[action] = child
                level2_nodes.append(child)
                
        # Backup from different leaves
        level2_nodes[0].backup(1.0)
        level2_nodes[1].backup(-0.5)
        level2_nodes[3].backup(0.8)
        
        # Check propagation
        assert root.visit_count == 1003
        assert level1_nodes[0].visit_count == 2  # Two children backed up
        assert level1_nodes[1].visit_count == 1  # One child backed up
        assert level1_nodes[2].visit_count == 0  # No backups
        
    def test_backup_extreme_values(self):
        """Test backup with extreme values"""
        root = Node(state=None, parent=None, action=None, prior=1.0)
        child = Node(state="child", parent=root, action=0, prior=0.5)
        root.children[0] = child
        
        # Test with very large value
        child.backup(1e6)
        assert child.value_sum == 1e6
        assert root.value_sum == -1e6
        
        # Test with very small value
        grandchild = Node(state="grandchild", parent=child, action=1, prior=0.5)
        child.children[1] = grandchild
        
        grandchild.backup(1e-10)
        assert grandchild.value_sum == pytest.approx(1e-10)
        
    def test_node_state_handling(self):
        """Test that node properly stores and maintains state"""
        # Test with various state types
        states = [
            None,
            "string_state",
            42,
            [1, 2, 3],
            {"board": [[0, 1], [1, 0]]},
            np.array([1, 2, 3])
        ]
        
        for state in states:
            node = Node(state=state, parent=None, action=None, prior=1.0)
            
            # For numpy arrays, use array_equal
            if isinstance(state, np.ndarray):
                assert np.array_equal(node.state, state)
            else:
                assert node.state == state or node.state is state