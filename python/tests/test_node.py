"""Tests for the MCTS Node class"""

import numpy as np
import pytest

from mcts.core.node import Node


class TestNode:
    """Test suite for the Node class"""
    
    def test_node_initialization(self):
        """Test basic node creation and initialization"""
        # Create a root node with no parent
        root = Node(
            state=None,  # Root has no game state initially
            parent=None,
            action=None,
            prior=1.0
        )
        
        assert root.parent is None
        assert root.action is None
        assert root.prior == 1.0
        assert root.visit_count == 0
        assert root.value_sum == 0.0
        assert root.children == {}
        assert root.is_expanded is False
        
    def test_node_with_parent(self):
        """Test node creation with parent relationship"""
        root = Node(state=None, parent=None, action=None, prior=1.0)
        
        # Create child node
        child_state = "dummy_state"  # Placeholder for actual game state
        child = Node(
            state=child_state,
            parent=root,
            action=5,  # Action 5 led to this child
            prior=0.8
        )
        
        assert child.parent is root
        assert child.action == 5
        assert child.prior == 0.8
        assert child.state == child_state
        
    def test_node_value_methods(self):
        """Test value-related methods"""
        node = Node(state=None, parent=None, action=None, prior=1.0)
        
        # Initially no visits
        assert node.visit_count == 0
        assert node.value() == 0.0
        
        # Add some visits and value
        node.visit_count = 10
        node.value_sum = 5.0
        assert node.value() == 0.5  # 5.0 / 10
        
        # Test with negative values
        node.value_sum = -3.0
        assert node.value() == -0.3  # -3.0 / 10
        
    def test_node_ucb_score(self):
        """Test UCB (Upper Confidence Bound) score calculation"""
        root = Node(state=None, parent=None, action=None, prior=1.0)
        root.visit_count = 100
        
        child = Node(state="state", parent=root, action=0, prior=0.5)
        
        # Initial UCB should be based on prior only (no visits)
        c_puct = 1.0
        ucb = child.ucb_score(c_puct)
        # UCB = Q + c_puct * P * sqrt(parent_visits) / (1 + visits)
        # UCB = 0 + 1.0 * 0.5 * sqrt(100) / 1 = 5.0
        assert ucb == pytest.approx(5.0)
        
        # Add some visits and value
        child.visit_count = 10
        child.value_sum = 3.0  # Q = 0.3
        ucb = child.ucb_score(c_puct)
        # UCB = 0.3 + 1.0 * 0.5 * sqrt(100) / 11 ≈ 0.3 + 0.4545
        assert ucb == pytest.approx(0.3 + 0.5 * 10 / 11, rel=1e-4)
        
    def test_node_is_leaf(self):
        """Test leaf node detection"""
        node = Node(state=None, parent=None, action=None, prior=1.0)
        
        # Node with no children is a leaf
        assert node.is_leaf() is True
        
        # Add a child
        child = Node(state="state", parent=node, action=0, prior=0.5)
        node.children[0] = child
        node.is_expanded = True
        
        # Now it's not a leaf
        assert node.is_leaf() is False
        
    def test_node_expand(self):
        """Test node expansion with multiple children"""
        node = Node(state="parent_state", parent=None, action=None, prior=1.0)
        
        # Simulate expansion with action probabilities
        action_probs = {
            0: 0.3,
            1: 0.5,
            2: 0.2
        }
        
        node.expand(action_probs, child_states={
            0: "state_0",
            1: "state_1", 
            2: "state_2"
        })
        
        assert node.is_expanded is True
        assert len(node.children) == 3
        
        # Check each child
        for action, prob in action_probs.items():
            assert action in node.children
            child = node.children[action]
            assert child.parent is node
            assert child.action == action
            assert child.prior == prob
            assert child.state == f"state_{action}"
            
    def test_node_backup(self):
        """Test value backup through the tree"""
        root = Node(state=None, parent=None, action=None, prior=1.0)
        child1 = Node(state="s1", parent=root, action=0, prior=0.5)
        child2 = Node(state="s2", parent=child1, action=1, prior=0.3)
        
        root.children[0] = child1
        child1.children[1] = child2
        
        # Backup a value from leaf to root
        value = 0.8
        child2.backup(value)
        
        # Check visit counts
        assert child2.visit_count == 1
        assert child1.visit_count == 1
        assert root.visit_count == 1
        
        # Check values (note: values alternate signs for adversarial games)
        assert child2.value_sum == 0.8
        assert child1.value_sum == -0.8  # Negated
        assert root.value_sum == 0.8    # Negated again
        
        # Backup another value
        child2.backup(0.6)
        assert child2.visit_count == 2
        assert child2.value_sum == 1.4
        assert child1.value_sum == -1.4
        assert root.value_sum == 1.4
        
    def test_node_select_child(self):
        """Test child selection based on UCB scores"""
        root = Node(state=None, parent=None, action=None, prior=1.0)
        root.visit_count = 100
        
        # Create children with different priors and values
        children_data = [
            (0, 0.5, 10, 3.0),   # action, prior, visits, value_sum
            (1, 0.3, 20, 8.0),   
            (2, 0.2, 5, 2.5),    
        ]
        
        for action, prior, visits, value_sum in children_data:
            child = Node(state=f"s{action}", parent=root, action=action, prior=prior)
            child.visit_count = visits
            child.value_sum = value_sum
            root.children[action] = child
            
        # Select best child based on UCB
        c_puct = 1.0
        best_child = root.select_child(c_puct)
        
        # Verify it selected the child with highest UCB score
        ucb_scores = {
            action: child.ucb_score(c_puct) 
            for action, child in root.children.items()
        }
        best_action = max(ucb_scores, key=ucb_scores.get)
        assert best_child.action == best_action
        
    def test_node_select_child_tie_breaking(self):
        """Test random tie-breaking in child selection"""
        root = Node(state=None, parent=None, action=None, prior=1.0)
        root.visit_count = 100
        
        # Create children with identical UCB scores
        for i in range(5):
            child = Node(state=f"s{i}", parent=root, action=i, prior=0.2)
            child.visit_count = 10
            child.value_sum = 5.0  # All have same Q-value
            root.children[i] = child
            
        # Select multiple times and check distribution
        selections = []
        for _ in range(100):
            selected = root.select_child(c_puct=1.0)
            selections.append(selected.action)
            
        # Should have selected different actions (not always the same)
        unique_selections = set(selections)
        assert len(unique_selections) > 1  # Should select different children
        
        # Check roughly uniform distribution (with some tolerance)
        for action in range(5):
            count = selections.count(action)
            assert 5 < count < 35  # Roughly 20% each with tolerance
        
    def test_node_terminal_state(self):
        """Test handling of terminal game states"""
        # Terminal node should not be expandable
        node = Node(state="terminal", parent=None, action=None, prior=1.0)
        node.is_terminal = True
        
        assert node.is_terminal is True
        assert node.is_leaf() is True  # Terminal nodes are always leaves
        
        # Attempting to expand should raise an error or do nothing
        with pytest.raises(ValueError):
            node.expand({0: 0.5, 1: 0.5}, {0: "s0", 1: "s1"})
            
    def test_node_vectorized_backup(self):
        """Test vectorized backup for multiple paths"""
        root = Node(state=None, parent=None, action=None, prior=1.0)
        
        # Create a small tree
        children = []
        for i in range(3):
            child = Node(state=f"s{i}", parent=root, action=i, prior=0.33)
            root.children[i] = child
            children.append(child)
            
        # Simulate vectorized backup of multiple values
        values = np.array([0.5, -0.3, 0.8])
        paths = [children[0], children[1], children[2]]
        
        for path, value in zip(paths, values):
            path.backup(value)
            
        # Check that all paths were updated
        assert children[0].visit_count == 1
        assert children[0].value_sum == 0.5
        assert children[1].visit_count == 1  
        assert children[1].value_sum == -0.3
        assert children[2].visit_count == 1
        assert children[2].value_sum == 0.8
        
        # Root should have all visits
        assert root.visit_count == 3
        # Root value sum should be sum of negated child values
        assert root.value_sum == pytest.approx(-0.5 + 0.3 - 0.8)