"""
Comprehensive tests for tree operations

This module tests the TreeOperations class which handles:
- Tree management and manipulation
- Subtree reuse functionality  
- Node operations
- Tree statistics
- Dirichlet noise application
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock

from mcts.core.tree_operations import TreeOperations
from mcts.gpu.csr_tree import CSRTree, CSRTreeConfig
from mcts.core.mcts_config import MCTSConfig
from conftest import assert_tensor_equal


class TestTreeOperationsBasics:
    """Test basic tree operations functionality"""
    
    def test_initialization(self, device):
        """Test TreeOperations initialization"""
        # Create CSR tree
        tree_config = CSRTreeConfig(
            max_nodes=1000,
            max_edges=10000,
            max_actions=225,
            device=str(device)
        )
        tree = CSRTree(tree_config)
        
        # Create MCTS config
        mcts_config = MCTSConfig()
        mcts_config.device = str(device)
        
        # Initialize TreeOperations
        tree_ops = TreeOperations(tree, mcts_config, device)
        
        assert tree_ops.tree == tree
        assert tree_ops.config == mcts_config
        assert tree_ops.device == device
        assert tree_ops.last_selected_action is None
        
    def test_clear_tree(self, device):
        """Test clearing tree operations"""
        tree_config = CSRTreeConfig(
            max_nodes=1000,
            device=str(device)
        )
        tree = CSRTree(tree_config)
        mcts_config = MCTSConfig()
        
        tree_ops = TreeOperations(tree, mcts_config, device)
        
        # Add some nodes
        tree.add_children_batch(0, [0, 1, 2], [0.3, 0.3, 0.4])
        tree_ops.last_selected_action = 1
        
        # Clear
        tree_ops.clear()
        
        # Should reset tree and state
        assert tree.num_nodes == 1  # Only root
        assert tree_ops.last_selected_action is None
        
    def test_reset_tree(self, device):
        """Test reset is equivalent to clear"""
        tree_config = CSRTreeConfig(max_nodes=1000, device=str(device))
        tree = CSRTree(tree_config)
        mcts_config = MCTSConfig()
        
        tree_ops = TreeOperations(tree, mcts_config, device)
        
        # Add nodes
        tree.add_children_batch(0, [0, 1, 2], [0.3, 0.3, 0.4])
        
        # Reset
        tree_ops.reset_tree()
        
        # Should be cleared
        assert tree.num_nodes == 1


class TestSubtreeReuse:
    """Test subtree reuse functionality"""
    
    def test_subtree_reuse_basic(self, device):
        """Test basic subtree reuse when moving root"""
        tree_config = CSRTreeConfig(
            max_nodes=1000,
            device=str(device)
        )
        tree = CSRTree(tree_config)
        
        mcts_config = MCTSConfig()
        mcts_config.enable_subtree_reuse = True
        mcts_config.subtree_reuse_min_visits = 5
        
        tree_ops = TreeOperations(tree, mcts_config, device)
        
        # Build tree structure
        # Root (0) -> Child1 (1), Child2 (2), Child3 (3)
        # Child1 (1) -> GrandChild1 (4), GrandChild2 (5)
        child_indices = tree.add_children_batch(0, [0, 1, 2], [0.3, 0.4, 0.3])
        grandchild_indices = tree.add_children_batch(child_indices[0], [10, 11], [0.5, 0.5])
        
        # Set visit counts
        tree.node_data.visit_counts[0] = 20
        tree.node_data.visit_counts[child_indices[0]] = 10
        tree.node_data.visit_counts[child_indices[1]] = 5
        tree.node_data.visit_counts[child_indices[2]] = 5
        tree.node_data.visit_counts[grandchild_indices[0]] = 6
        tree.node_data.visit_counts[grandchild_indices[1]] = 4
        
        # Apply subtree reuse - move to child1
        mapping = tree_ops.apply_subtree_reuse(0)  # action 0 -> child1
        
        assert mapping is not None
        assert len(mapping) > 0
        
        # Child1 should now be root
        assert 0 in mapping.values()  # New root index
        assert tree.num_nodes == 3  # Root + 2 grandchildren
        
        # Check visit counts preserved
        assert tree.node_data.visit_counts[0] == 10  # Child1's visits
        
    def test_subtree_reuse_insufficient_visits(self, device):
        """Test subtree reuse skipped when insufficient visits"""
        tree_config = CSRTreeConfig(max_nodes=1000, device=str(device))
        tree = CSRTree(tree_config)
        
        mcts_config = MCTSConfig()
        mcts_config.enable_subtree_reuse = True
        mcts_config.subtree_reuse_min_visits = 10
        
        tree_ops = TreeOperations(tree, mcts_config, device)
        
        # Build small tree
        child_indices = tree.add_children_batch(0, [0, 1], [0.5, 0.5])
        tree.node_data.visit_counts[child_indices[0]] = 5  # Below threshold
        
        # Apply subtree reuse
        mapping = tree_ops.apply_subtree_reuse(0)
        
        # Should reset tree instead
        assert mapping == {}
        assert tree.num_nodes == 1
        
    def test_subtree_reuse_nonexistent_action(self, device):
        """Test subtree reuse with non-existent action"""
        tree_config = CSRTreeConfig(max_nodes=1000, device=str(device))
        tree = CSRTree(tree_config)
        
        mcts_config = MCTSConfig()
        mcts_config.enable_subtree_reuse = True
        
        tree_ops = TreeOperations(tree, mcts_config, device)
        
        # Build tree
        tree.add_children_batch(0, [0, 1], [0.5, 0.5])
        
        # Try to reuse with non-existent action
        mapping = tree_ops.apply_subtree_reuse(99)  # Action 99 doesn't exist
        
        assert mapping is None
        
    def test_subtree_reuse_disabled(self, device):
        """Test behavior when subtree reuse is disabled"""
        tree_config = CSRTreeConfig(max_nodes=1000, device=str(device))
        tree = CSRTree(tree_config)
        
        mcts_config = MCTSConfig()
        mcts_config.enable_subtree_reuse = False
        
        tree_ops = TreeOperations(tree, mcts_config, device)
        
        # Build tree
        tree.add_children_batch(0, [0, 1], [0.5, 0.5])
        
        # Apply subtree reuse
        mapping = tree_ops.apply_subtree_reuse(0)
        
        assert mapping is None
        
    def test_subtree_extraction(self, device):
        """Test internal subtree extraction method"""
        tree_config = CSRTreeConfig(max_nodes=1000, device=str(device))
        tree = CSRTree(tree_config)
        
        mcts_config = MCTSConfig()
        tree_ops = TreeOperations(tree, mcts_config, device)
        
        # Build complex tree
        # Root -> A, B
        # A -> C, D
        # B -> E
        # C -> F
        child_a, child_b = tree.add_children_batch(0, [0, 1], [0.5, 0.5])
        child_c, child_d = tree.add_children_batch(child_a, [2, 3], [0.5, 0.5])
        child_e = tree.add_child(child_b, 4, 1.0)
        child_f = tree.add_child(child_c, 5, 1.0)
        
        # Extract subtree rooted at A
        mapping = tree_ops._extract_subtree(child_a)
        
        # Should include A, C, D, F
        assert child_a in mapping
        assert child_c in mapping
        assert child_d in mapping
        assert child_f in mapping
        
        # Should not include B, E
        assert child_b not in mapping
        assert child_e not in mapping


class TestNodeInformation:
    """Test retrieving node information"""
    
    def test_get_root_children_info(self, device):
        """Test getting root children information"""
        tree_config = CSRTreeConfig(max_nodes=1000, device=str(device))
        tree = CSRTree(tree_config)
        mcts_config = MCTSConfig()
        
        tree_ops = TreeOperations(tree, mcts_config, device)
        
        # Add children to root
        actions = [0, 1, 2]
        priors = [0.3, 0.4, 0.3]
        child_indices = tree.add_children_batch(0, actions, priors)
        
        # Set some visits and values
        tree.node_data.visit_counts[child_indices[0]] = 10
        tree.node_data.value_sums[child_indices[0]] = 5.0
        tree.node_data.visit_counts[child_indices[1]] = 15
        tree.node_data.value_sums[child_indices[1]] = -3.0
        
        # Get info
        ret_actions, ret_visits, ret_values = tree_ops.get_root_children_info()
        
        assert len(ret_actions) == 3
        assert torch.allclose(ret_actions, torch.tensor(actions, device=device))
        assert ret_visits[0] == 10
        assert ret_visits[1] == 15
        assert ret_visits[2] == 0
        
        # Values should be Q-values (average)
        assert abs(ret_values[0].item() - 0.5) < 1e-5  # 5.0/10
        assert abs(ret_values[1].item() - (-0.2)) < 1e-5  # -3.0/15
        
    def test_get_root_children_info_empty(self, device):
        """Test getting info when root has no children"""
        tree_config = CSRTreeConfig(max_nodes=1000, device=str(device))
        tree = CSRTree(tree_config)
        mcts_config = MCTSConfig()
        
        tree_ops = TreeOperations(tree, mcts_config, device)
        
        # Get info from empty root
        actions, visits, values = tree_ops.get_root_children_info()
        
        assert len(actions) == 0
        assert len(visits) == 0
        assert len(values) == 0
        
    def test_get_best_child(self, device):
        """Test getting best child using UCB formula"""
        tree_config = CSRTreeConfig(max_nodes=1000, device=str(device))
        tree = CSRTree(tree_config)
        mcts_config = MCTSConfig()
        
        tree_ops = TreeOperations(tree, mcts_config, device)
        
        # Add children with different stats
        child_indices = tree.add_children_batch(0, [0, 1, 2], [0.5, 0.3, 0.2])
        
        # Set parent visits
        tree.node_data.visit_counts[0] = 20
        
        # Child 0: High value, high visits
        tree.node_data.visit_counts[child_indices[0]] = 10
        tree.node_data.value_sums[child_indices[0]] = 8.0  # Q = 0.8
        
        # Child 1: Low value, low visits
        tree.node_data.visit_counts[child_indices[1]] = 2
        tree.node_data.value_sums[child_indices[1]] = -1.0  # Q = -0.5
        
        # Child 2: Unvisited (high exploration bonus)
        tree.node_data.visit_counts[child_indices[2]] = 0
        
        # Get best child with different c_puct values
        # Low c_puct should favor exploitation
        best_low_c = tree_ops.get_best_child(0, c_puct=0.1)
        assert best_low_c == child_indices[0]  # High value child
        
        # High c_puct should favor exploration
        best_high_c = tree_ops.get_best_child(0, c_puct=5.0)
        assert best_high_c == child_indices[2]  # Unvisited child
        
    def test_get_best_child_no_children(self, device):
        """Test getting best child when node has no children"""
        tree_config = CSRTreeConfig(max_nodes=1000, device=str(device))
        tree = CSRTree(tree_config)
        mcts_config = MCTSConfig()
        
        tree_ops = TreeOperations(tree, mcts_config, device)
        
        # Get best child of root with no children
        best = tree_ops.get_best_child(0, c_puct=1.0)
        
        assert best is None


class TestDirichletNoise:
    """Test Dirichlet noise application"""
    
    def test_add_dirichlet_noise_to_root(self, device):
        """Test adding Dirichlet noise to root priors"""
        tree_config = CSRTreeConfig(max_nodes=1000, device=str(device))
        tree = CSRTree(tree_config)
        mcts_config = MCTSConfig()
        
        tree_ops = TreeOperations(tree, mcts_config, device)
        
        # Add children with uniform priors
        num_children = 5
        actions = list(range(num_children))
        priors = [1.0/num_children] * num_children
        child_indices = tree.add_children_batch(0, actions, priors)
        
        
        # Store original priors
        original_priors = tree.node_data.node_priors[child_indices].clone()
        
        # Debug: Check if priors were set correctly
        assert original_priors.shape[0] == num_children
        assert torch.allclose(original_priors, torch.full((num_children,), 1.0/num_children, device=device))
        
        # Add Dirichlet noise
        alpha = 0.3
        epsilon = 0.25
        tree_ops.add_dirichlet_noise_to_root(alpha, epsilon)
        
        # Get new priors - make sure to get a fresh copy
        new_priors = tree.node_data.node_priors[child_indices].clone()
        
        # Priors should have changed
        assert not torch.allclose(original_priors, new_priors)
        
        # Should still sum to 1
        assert abs(new_priors.sum().item() - 1.0) < 1e-5
        
        # Should be valid probabilities
        assert torch.all(new_priors >= 0)
        assert torch.all(new_priors <= 1)
        
        # Should be a mix of original and noise
        # With epsilon=0.25, should be 75% original + 25% noise
        # So no prior should deviate too much from original
        max_deviation = torch.abs(new_priors - original_priors).max()
        assert max_deviation < 0.5
        
    def test_dirichlet_noise_empty_root(self, device):
        """Test Dirichlet noise on root with no children"""
        tree_config = CSRTreeConfig(max_nodes=1000, device=str(device))
        tree = CSRTree(tree_config)
        mcts_config = MCTSConfig()
        
        tree_ops = TreeOperations(tree, mcts_config, device)
        
        # Should not crash
        tree_ops.add_dirichlet_noise_to_root(0.3, 0.25)
        
        # Tree should be unchanged
        assert tree.num_nodes == 1
        
    def test_dirichlet_noise_determinism(self, device):
        """Test Dirichlet noise produces different results each time"""
        tree_config = CSRTreeConfig(max_nodes=1000, device=str(device))
        tree = CSRTree(tree_config)
        mcts_config = MCTSConfig()
        
        tree_ops = TreeOperations(tree, mcts_config, device)
        
        # Add children
        child_indices = tree.add_children_batch(0, [0, 1, 2], [0.3, 0.4, 0.3])
        
        # Apply noise multiple times
        priors_1 = tree.node_data.node_priors[child_indices].clone()
        tree_ops.add_dirichlet_noise_to_root(0.3, 0.25)
        priors_2 = tree.node_data.node_priors[child_indices].clone()
        tree_ops.add_dirichlet_noise_to_root(0.3, 0.25)
        priors_3 = tree.node_data.node_priors[child_indices].clone()
        
        # Each application should produce different results
        assert not torch.allclose(priors_1, priors_2)
        assert not torch.allclose(priors_2, priors_3)


class TestTreeStatistics:
    """Test tree statistics computation"""
    
    def test_get_tree_statistics_basic(self, device):
        """Test basic tree statistics"""
        tree_config = CSRTreeConfig(max_nodes=1000, device=str(device))
        tree = CSRTree(tree_config)
        mcts_config = MCTSConfig()
        
        tree_ops = TreeOperations(tree, mcts_config, device)
        
        # Build a simple tree
        child_indices = tree.add_children_batch(0, [0, 1], [0.5, 0.5])
        grandchild_indices = tree.add_children_batch(child_indices[0], [2, 3], [0.5, 0.5])
        
        # Set some visits
        tree.node_data.visit_counts[0] = 20
        tree.node_data.visit_counts[child_indices[0]] = 15
        tree.node_data.visit_counts[child_indices[1]] = 5
        
        # Get statistics
        stats = tree_ops.get_tree_statistics()
        
        assert stats['num_nodes'] == 5
        assert stats['root_visits'] == 20
        assert stats['root_children'] == 2
        assert stats['max_depth'] == 2
        assert stats['memory_usage_mb'] > 0
        
    def test_calculate_max_depth(self, device):
        """Test maximum depth calculation"""
        tree_config = CSRTreeConfig(max_nodes=1000, device=str(device))
        tree = CSRTree(tree_config)
        mcts_config = MCTSConfig()
        
        tree_ops = TreeOperations(tree, mcts_config, device)
        
        # Build deep tree
        current_node = 0
        for depth in range(5):
            child = tree.add_child(current_node, depth, 1.0)
            current_node = child
            
        # Calculate depth
        max_depth = tree_ops._calculate_max_depth()
        
        assert max_depth == 5
        
    def test_calculate_max_depth_single_node(self, device):
        """Test depth calculation with only root"""
        tree_config = CSRTreeConfig(max_nodes=1000, device=str(device))
        tree = CSRTree(tree_config)
        mcts_config = MCTSConfig()
        
        tree_ops = TreeOperations(tree, mcts_config, device)
        
        # Only root
        max_depth = tree_ops._calculate_max_depth()
        
        assert max_depth == 0
        
    def test_count_root_children(self, device):
        """Test counting root children"""
        tree_config = CSRTreeConfig(max_nodes=1000, device=str(device))
        tree = CSRTree(tree_config)
        mcts_config = MCTSConfig()
        
        tree_ops = TreeOperations(tree, mcts_config, device)
        
        # Add varying number of children
        for num_children in [0, 1, 5, 10]:
            tree.reset()  # Start fresh
            
            if num_children > 0:
                actions = list(range(num_children))
                priors = [1.0/num_children] * num_children
                tree.add_children_batch(0, actions, priors)
                
            count = tree_ops._count_root_children()
            assert count == num_children
            
    def test_estimate_memory_usage(self, device):
        """Test memory usage estimation"""
        tree_config = CSRTreeConfig(max_nodes=10000, device=str(device))
        tree = CSRTree(tree_config)
        mcts_config = MCTSConfig()
        
        tree_ops = TreeOperations(tree, mcts_config, device)
        
        # Build larger tree
        # Add 100 children to root
        child_indices = tree.add_children_batch(0, list(range(100)), [0.01] * 100)
        
        # Add 10 children to each of first 10 children
        for i in range(10):
            tree.add_children_batch(child_indices[i], list(range(10)), [0.1] * 10)
            
        # Estimate memory
        memory_bytes = tree_ops._estimate_memory_usage()
        memory_mb = memory_bytes / (1024 * 1024)
        
        # Should be reasonable
        assert memory_mb > 0
        assert memory_mb < 100  # Less than 100MB for this test tree


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_operations_on_empty_tree(self, device):
        """Test operations on tree with only root"""
        tree_config = CSRTreeConfig(max_nodes=100, device=str(device))
        tree = CSRTree(tree_config)
        mcts_config = MCTSConfig()
        
        tree_ops = TreeOperations(tree, mcts_config, device)
        
        # These should all handle empty tree gracefully
        stats = tree_ops.get_tree_statistics()
        assert stats['num_nodes'] == 1
        assert stats['root_children'] == 0
        
        actions, visits, values = tree_ops.get_root_children_info()
        assert len(actions) == 0
        
        best = tree_ops.get_best_child(0, c_puct=1.0)
        assert best is None
        
    def test_operations_on_invalid_nodes(self, device):
        """Test operations on invalid node indices"""
        tree_config = CSRTreeConfig(max_nodes=100, device=str(device))
        tree = CSRTree(tree_config)
        mcts_config = MCTSConfig()
        
        tree_ops = TreeOperations(tree, mcts_config, device)
        
        # Add some children
        tree.add_children_batch(0, [0, 1], [0.5, 0.5])
        
        # Try operations on invalid nodes
        best = tree_ops.get_best_child(999, c_puct=1.0)  # Invalid node
        assert best is None
        
        # Subtree reuse with None action
        mapping = tree_ops.apply_subtree_reuse(None)
        assert mapping is None
        
    def test_large_tree_operations(self, device):
        """Test operations on larger trees"""
        tree_config = CSRTreeConfig(
            max_nodes=10000,
            max_edges=100000,
            device=str(device)
        )
        tree = CSRTree(tree_config)
        mcts_config = MCTSConfig()
        mcts_config.enable_subtree_reuse = True
        
        tree_ops = TreeOperations(tree, mcts_config, device)
        
        # Build a wide tree
        # Root has 100 children
        actions = list(range(100))
        priors = [0.01] * 100
        child_indices = tree.add_children_batch(0, actions, priors)
        
        # Each child has 10 children
        for i in range(10):  # Only first 10 to keep it manageable
            gc_actions = list(range(10))
            gc_priors = [0.1] * 10
            tree.add_children_batch(child_indices[i], gc_actions, gc_priors)
            
        # Set visits
        tree.node_data.visit_counts[0] = 1000
        for i in range(10):
            tree.node_data.visit_counts[child_indices[i]] = 50
            
        # Test operations still work
        stats = tree_ops.get_tree_statistics()
        assert stats['num_nodes'] == 1 + 100 + 10*10
        
        # Test subtree reuse
        mapping = tree_ops.apply_subtree_reuse(0)  # Move to first child
        assert mapping is not None
        assert tree.num_nodes == 1 + 10  # New root + its 10 children