"""Test for wave_search bug fix - TDD approach

This test verifies that wave_search correctly accesses tree attributes
when using fused_select_expand operation.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.mcts.core.wave_search import WaveSearch
from python.mcts.gpu.csr_tree import CSRTree, CSRTreeConfig
from python.mcts.gpu.gpu_game_states import GPUGameStates, GPUGameStatesConfig, GameType


class TestWaveSearchFusedSelectExpand:
    """Test cases for wave_search fused_select_expand bug"""
    
    def test_fused_select_expand_accesses_correct_tree_attributes(self):
        """Test that _try_fused_select_expand accesses tree.children instead of tree.node_data.children
        
        This test should FAIL with current implementation and PASS after fix
        """
        # Arrange - create mocked components
        device = torch.device('cpu')
        
        # Create tree config
        tree_config = CSRTreeConfig(
            max_nodes=1000,
            max_actions=225,  # Correct size for 15x15 Gomoku
            device='cpu'
        )
        
        # Create actual tree with proper structure
        tree = CSRTree(tree_config)
        
        # Add a root node using correct API
        root_idx = tree.node_data.allocate_node(prior=1.0, parent_idx=-1, parent_action=-1)
        
        # Mock the node_data to NOT have children attribute (as in reality)
        # This will cause AttributeError in current buggy implementation
        original_node_data = tree.node_data
        tree.node_data = Mock(spec=['visit_counts', 'value_sums', 'node_priors'])
        tree.node_data.visit_counts = torch.zeros(10, device=device, dtype=torch.int32)
        tree.node_data.value_sums = torch.zeros(10, device=device) 
        tree.node_data.node_priors = torch.ones(10, device=device) * 0.1
        # Explicitly NOT adding children attribute to node_data
        
        # Create game states
        game_states_config = GPUGameStatesConfig(
            capacity=100,
            device='cpu',
            game_type=GameType.GOMOKU
        )
        game_states = GPUGameStates(game_states_config)
        
        # Create evaluator mock
        evaluator = Mock()
        
        # Create config mock
        config = Mock()
        config.device = 'cpu'
        config.enable_kernel_fusion = True
        config.max_depth = 50
        config.c_puct = 1.4
        
        # Create GPU ops mock with fused_select_expand
        gpu_ops = Mock()
        gpu_ops.fused_select_expand = Mock(return_value=None)
        
        # Create wave search
        wave_search = WaveSearch(
            tree=tree,
            game_states=game_states,
            evaluator=evaluator,
            config=config,
            device=device,
            gpu_ops=gpu_ops
        )
        
        # Act - should work now with the fix
        result = wave_search._try_fused_select_expand(wave_size=8)
        
        # Verify it calls gpu_ops with correct data from tree.children
        gpu_ops.fused_select_expand.assert_called_once()
        call_args = gpu_ops.fused_select_expand.call_args[1]
        
        # Verify children come from tree.children, not node_data
        assert torch.equal(call_args['children'], tree.children)
        
        # Verify q_values are calculated correctly
        expected_q_values = tree.node_data.value_sums / (tree.node_data.visit_counts + 1e-8)
        assert torch.allclose(call_args['q_values'], expected_q_values)
        
        # Verify is_expanded is calculated from children
        expected_is_expanded = (tree.children >= 0).any(dim=1)
        assert torch.equal(call_args['is_expanded'], expected_is_expanded)
    
    def test_fused_select_expand_computes_q_values_correctly(self):
        """Test that q_values are computed from value_sums/visit_counts"""
        # This test verifies the q_value calculation is correct
        device = torch.device('cpu')
        
        # Create tree
        tree_config = CSRTreeConfig(max_nodes=100, max_actions=225, device='cpu')
        tree = CSRTree(tree_config)
        
        # Add nodes with known values
        root_idx = tree.node_data.allocate_node(prior=1.0, parent_idx=-1, parent_action=-1)
        tree.node_data.visit_counts[0] = 10
        tree.node_data.value_sums[0] = 5.0  # Q-value should be 0.5
        
        # Create mocks
        game_states = Mock()
        evaluator = Mock()
        config = Mock()
        config.device = 'cpu'
        config.enable_kernel_fusion = True
        config.max_depth = 50
        config.c_puct = 1.4
        
        gpu_ops = Mock()
        gpu_ops.fused_select_expand = Mock(return_value=None)
        
        # Create wave search
        wave_search = WaveSearch(tree, game_states, evaluator, config, device, gpu_ops)
        
        # Act
        wave_search._try_fused_select_expand(wave_size=1)
        
        # Assert
        call_args = gpu_ops.fused_select_expand.call_args[1]
        q_values = call_args['q_values']
        
        # Q-value for node 0 should be 5.0 / 10 = 0.5
        assert abs(q_values[0].item() - 0.5) < 1e-6
    
    def test_fused_select_expand_computes_is_expanded_correctly(self):
        """Test that is_expanded is computed from children array"""
        device = torch.device('cpu')
        
        # Create tree
        tree_config = CSRTreeConfig(max_nodes=100, max_actions=225, device='cpu')
        tree = CSRTree(tree_config)
        
        # Add root and one child
        root_idx = tree.node_data.allocate_node(prior=1.0, parent_idx=-1, parent_action=-1)
        child_idx = tree.add_child(parent_idx=0, action=112, child_prior=0.5)  # Center of 15x15 board
        
        # Create mocks
        game_states = Mock()
        evaluator = Mock()
        config = Mock()
        config.device = 'cpu'
        config.enable_kernel_fusion = True
        config.max_depth = 50
        config.c_puct = 1.4
        
        gpu_ops = Mock()
        gpu_ops.fused_select_expand = Mock(return_value=None)
        
        # Create wave search
        wave_search = WaveSearch(tree, game_states, evaluator, config, device, gpu_ops)
        
        # Act
        wave_search._try_fused_select_expand(wave_size=2)
        
        # Assert
        call_args = gpu_ops.fused_select_expand.call_args[1]
        is_expanded = call_args['is_expanded']
        
        # Node 0 should be expanded (has children), node 1 should not be
        assert is_expanded[0].item() == True
        assert is_expanded[1].item() == False


if __name__ == "__main__":
    # Run the tests
    test = TestWaveSearchFusedSelectExpand()
    
    print("Running test: fused_select_expand accesses correct tree attributes...")
    try:
        test.test_fused_select_expand_accesses_correct_tree_attributes()
        print("FAIL - Test should have failed but passed (bug might be already fixed)")
    except AttributeError as e:
        print(f"PASS - Test failed as expected with AttributeError: {e}")
    except Exception as e:
        print(f"UNEXPECTED ERROR: {type(e).__name__}: {e}")
    
    print("\nRunning test: q_values computation...")
    try:
        test.test_fused_select_expand_computes_q_values_correctly()
        print("Test passed")
    except Exception as e:
        print(f"Test failed: {type(e).__name__}: {e}")
    
    print("\nRunning test: is_expanded computation...")
    try:
        test.test_fused_select_expand_computes_is_expanded_correctly()
        print("Test passed")
    except Exception as e:
        print(f"Test failed: {type(e).__name__}: {e}")