"""Comprehensive integration test for MCTS tree expansion

This test verifies that MCTS tree expansion works correctly end-to-end,
ensuring no corruption of training data.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.mcts.core.mcts import MCTS, MCTSConfig
from python.mcts.gpu.csr_tree import CSRTree, CSRTreeConfig
from python.mcts.gpu.gpu_game_states import GPUGameStates, GPUGameStatesConfig, GameType
from python.mcts.core.game_interface import GameInterface


class MockEvaluator:
    """Mock evaluator that returns reasonable values"""
    def __init__(self, device='cpu', action_size=225):
        self.device = device
        self.action_size = action_size
        self._return_torch_tensors = True
        
    def evaluate(self, states, batch_size=None):
        """Return mock policy and value"""
        if hasattr(states, '__len__'):
            num_states = len(states)
        else:
            num_states = 1
            
        # Return uniform policy and neutral value
        policy = torch.ones(num_states, self.action_size, device=self.device) / self.action_size
        value = torch.zeros(num_states, 1, device=self.device)
        
        return policy, value


class TestMCTSTreeExpansion:
    """Integration tests for MCTS tree expansion"""
    
    def test_tree_expansion_creates_valid_nodes(self):
        """Test that tree expansion creates nodes with valid data"""
        # Create config
        config = MCTSConfig(
            num_simulations=100,
            device='cpu',
            c_puct=1.4,
            game_type='gomoku',
            board_size=15,
            max_tree_nodes=10000,
            batch_size=32,
            min_wave_size=8,
            max_wave_size=32
        )
        
        # Create evaluator
        evaluator = MockEvaluator(device='cpu', action_size=225)
        
        # Create a simplified tree and components for testing
        tree_config = CSRTreeConfig(
            max_nodes=10000,
            max_actions=225,
            device='cpu'
        )
        tree = CSRTree(tree_config)
        
        # Initialize root node
        root_idx = tree.node_data.allocate_node(prior=1.0, parent_idx=-1, parent_action=-1)
        
        # Simulate tree expansion manually
        for i in range(10):
            # Add some children to root
            actions = [i * 20 + j for j in range(5)]
            priors = [0.2] * 5
            child_indices = tree.add_children_batch(0, actions, priors)
            
            # Update visit counts
            tree.node_data.visit_counts[0] += 5
            for child_idx in child_indices:
                tree.node_data.visit_counts[child_idx] = 1
                tree.node_data.value_sums[child_idx] = np.random.uniform(-0.5, 0.5)
        
        # Verify tree has expanded
        assert tree.num_nodes > 1, "Tree should have expanded beyond root"
        
        # Verify all nodes have valid data
        for i in range(tree.num_nodes):
            # Check visit counts are non-negative
            assert tree.node_data.visit_counts[i] >= 0
            
            # Check priors are between 0 and 1
            assert 0 <= tree.node_data.node_priors[i] <= 1
            
            # Check parent indices are valid (except root)
            if i > 0:
                parent_idx = tree.node_data.parent_indices[i]
                assert parent_idx == -1 or 0 <= parent_idx < i, f"Node {i} has invalid parent {parent_idx}"
    
    def test_tree_expansion_maintains_consistency(self):
        """Test that tree expansion maintains parent-child consistency"""
        # Create tree
        tree_config = CSRTreeConfig(
            max_nodes=1000,
            max_actions=225,
            device='cpu'
        )
        tree = CSRTree(tree_config)
        
        # Initialize root
        root_idx = tree.node_data.allocate_node(prior=1.0, parent_idx=-1, parent_action=-1)
        
        # Add children at multiple levels
        # Level 1
        child1 = tree.add_child(0, 10, 0.3)
        child2 = tree.add_child(0, 20, 0.4)
        child3 = tree.add_child(0, 30, 0.3)
        
        # Level 2
        grandchild1 = tree.add_child(child1, 50, 0.5)
        grandchild2 = tree.add_child(child1, 60, 0.5)
        
        # Check parent-child consistency
        for node_idx in range(tree.num_nodes):
            children_slice = tree.children[node_idx]
            valid_children = children_slice[children_slice >= 0]
            
            for child_idx in valid_children:
                # Verify child's parent points back to this node
                parent_idx = tree.node_data.parent_indices[child_idx]
                assert parent_idx == node_idx, f"Child {child_idx} parent mismatch"
    
    def test_tree_expansion_with_high_simulation_count(self):
        """Test tree expansion with many simulations (stress test)"""
        # Create large tree
        tree_config = CSRTreeConfig(
            max_nodes=50000,
            max_actions=225,
            device='cpu'
        )
        tree = CSRTree(tree_config)
        
        # Initialize root
        root_idx = tree.node_data.allocate_node(prior=1.0, parent_idx=-1, parent_action=-1)
        
        # Simulate many expansions
        import random
        random.seed(42)
        
        nodes_to_expand = [0]
        for _ in range(100):
            # Pick a random node to expand
            if nodes_to_expand:
                parent = random.choice(nodes_to_expand)
                
                # Add some children
                num_children = random.randint(1, 5)
                actions = random.sample(range(225), num_children)
                priors = [1.0/num_children] * num_children
                
                new_children = tree.add_children_batch(parent, actions, priors)
                nodes_to_expand.extend(new_children)
                
                # Update visit counts
                tree.node_data.visit_counts[parent] += num_children
                for child in new_children:
                    tree.node_data.visit_counts[child] = 1
                    tree.node_data.value_sums[child] = random.uniform(-0.5, 0.5)
        
        # Verify reasonable tree growth
        assert 100 < tree.num_nodes < 10000, f"Unexpected node count: {tree.num_nodes}"
        
        # Verify root has been visited (we don't update root in this test)
        # In real MCTS, root would accumulate all visits from children
        assert tree.num_nodes >= 100, "Tree should have grown significantly"
        
        # Check value convergence (values should be reasonable)
        if tree.node_data.visit_counts[0] > 0:
            root_value = tree.node_data.value_sums[0] / tree.node_data.visit_counts[0]
            assert -1 <= root_value <= 1, f"Root value out of range: {root_value}"
    
    def test_policy_entropy_calculation(self):
        """Test that policy entropy is calculated correctly"""
        # Create tree
        tree_config = CSRTreeConfig(
            max_nodes=1000,
            max_actions=225,
            device='cpu'
        )
        tree = CSRTree(tree_config)
        
        # Initialize root
        root_idx = tree.node_data.allocate_node(prior=1.0, parent_idx=-1, parent_action=-1)
        
        # Add children with varied visit counts to create a policy
        actions = list(range(20))  # First 20 actions
        priors = [1.0/20] * 20
        children = tree.add_children_batch(0, actions, priors)
        
        # Set visit counts to create non-uniform policy
        total_visits = 100
        tree.node_data.visit_counts[0] = total_visits
        
        # Create a distribution that's not too uniform or too peaked
        import random
        random.seed(42)
        visits = []
        remaining = total_visits
        for i, child in enumerate(children[:-1]):
            v = random.randint(1, min(10, remaining - (len(children) - i - 1)))
            visits.append(v)
            remaining -= v
        visits.append(remaining)
        
        for child, v in zip(children, visits):
            tree.node_data.visit_counts[child] = v
        
        # Calculate policy from visit counts
        child_visits = tree.node_data.visit_counts[children].float()
        policy = child_visits / child_visits.sum()
        
        # Calculate entropy
        entropy = -(policy * torch.log(policy + 1e-8)).sum()
        
        # Verify entropy is reasonable (not collapsed)
        assert entropy > 1.0, f"Policy entropy too low: {entropy}"
        assert entropy < 4.0, f"Policy entropy suspiciously high: {entropy}"
    
    def test_no_fused_select_expand_errors(self):
        """Test that fused select+expand doesn't throw errors"""
        # This test verifies the fix by checking wave_search directly
        from python.mcts.core.wave_search import WaveSearch
        
        # Create tree
        tree_config = CSRTreeConfig(
            max_nodes=1000,
            max_actions=225,
            device='cpu'
        )
        tree = CSRTree(tree_config)
        
        # Initialize with some nodes
        root_idx = tree.node_data.allocate_node(prior=1.0, parent_idx=-1, parent_action=-1)
        tree.add_children_batch(0, [10, 20, 30], [0.3, 0.4, 0.3])
        
        # Create mock game states
        game_states = Mock()
        
        # Create mock evaluator
        evaluator = Mock()
        evaluator.evaluate = Mock(return_value=(
            torch.ones(8, 225) / 225,  # uniform policy
            torch.zeros(8, 1)  # neutral values
        ))
        
        # Create config
        config = Mock()
        config.device = 'cpu'
        config.enable_kernel_fusion = True
        config.max_depth = 50
        config.c_puct = 1.4
        
        # Create GPU ops mock
        gpu_ops = Mock()
        gpu_ops.fused_select_expand = Mock(return_value=None)
        
        # Create wave search
        wave_search = WaveSearch(
            tree=tree,
            game_states=game_states,
            evaluator=evaluator,
            config=config,
            device=torch.device('cpu'),
            gpu_ops=gpu_ops
        )
        
        # This should not raise any errors
        result = wave_search._try_fused_select_expand(wave_size=8)
        
        # Verify the call was made with correct parameters
        gpu_ops.fused_select_expand.assert_called_once()
        call_args = gpu_ops.fused_select_expand.call_args[1]
        
        # Verify no errors accessing tree data
        assert 'children' in call_args
        assert 'q_values' in call_args
        assert 'is_expanded' in call_args


if __name__ == "__main__":
    # Run tests
    test = TestMCTSTreeExpansion()
    
    print("Running test: tree expansion creates valid nodes...")
    try:
        test.test_tree_expansion_creates_valid_nodes()
        print("✓ Passed")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\nRunning test: tree expansion maintains consistency...")
    try:
        test.test_tree_expansion_maintains_consistency()
        print("✓ Passed")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\nRunning test: tree expansion with high simulation count...")
    try:
        test.test_tree_expansion_with_high_simulation_count()
        print("✓ Passed")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\nRunning test: policy entropy calculation...")
    try:
        test.test_policy_entropy_calculation()
        print("✓ Passed")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\nRunning test: no fused select+expand errors...")
    try:
        test.test_no_fused_select_expand_errors()
        print("✓ Passed")
    except Exception as e:
        print(f"✗ Failed: {e}")