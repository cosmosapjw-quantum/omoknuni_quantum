"""Test bound checking in tree operations

This module tests that tree operations properly handle invalid node indices
without crashing, following TDD principles.
"""

import pytest
import torch
from mcts.core.mcts import MCTS
from mcts.core.tree_operations import TreeOperations


class TestBoundChecking:
    """Test bound checking for tree operations"""
    
    def test_get_best_child_invalid_node_index(self):
        """Test that get_best_child handles invalid node indices gracefully
        
        This test reproduces the IndexError crash when accessing invalid node indices.
        """
        from mcts.gpu.csr_tree import CSRTree, CSRTreeConfig
        from mcts.core.tree_operations import TreeOperations
        from mcts.core.mcts_config import MCTSConfig
        
        # Create a small tree like in the failing test 
        tree_config = CSRTreeConfig(max_nodes=100, device='cpu')
        tree = CSRTree(tree_config)
        mcts_config = MCTSConfig()
        
        tree_ops = TreeOperations(tree, mcts_config, torch.device('cpu'))
        
        # Add some children to make it more realistic
        tree.add_children_batch(0, [0, 1], [0.5, 0.5])
        
        # Now the CSR storage size should be limited
        row_ptr_size = len(tree.csr_storage.row_ptr)
        print(f"DEBUG: row_ptr size: {row_ptr_size}")
        print(f"DEBUG: num_nodes: {tree.num_nodes}")
        
        # Try to access index 999 which should be out of bounds for max_nodes=100
        invalid_node_idx = 999
        
        # This should now return None gracefully instead of crashing
        best_child = tree_ops.get_best_child(invalid_node_idx, c_puct=1.0)
        assert best_child is None, f"Expected None for invalid node {invalid_node_idx}, got {best_child}"
        
    def test_get_best_child_edge_case_node_indices(self, base_mcts_config, mock_evaluator, empty_gomoku_state):
        """Test edge cases for node indices"""
        # Initialize MCTS
        mcts = MCTS(base_mcts_config, mock_evaluator)
        mcts._ensure_root_initialized(empty_gomoku_state)
        
        tree_ops = mcts.tree_ops
        num_nodes = mcts.tree.num_nodes
        
        # Test boundary cases
        test_cases = [
            -1,           # Negative index
            num_nodes,    # Exactly at boundary
            num_nodes + 1, # Just beyond boundary
            num_nodes + 100, # Far beyond boundary
        ]
        
        for invalid_idx in test_cases:
            # Should handle gracefully without crashing
            result = tree_ops.get_best_child(invalid_idx, c_puct=1.0)
            assert result is None, f"Expected None for invalid index {invalid_idx}, got {result}"
            
    def test_other_tree_operations_bound_checking(self, base_mcts_config, mock_evaluator, empty_gomoku_state):
        """Test that other tree operations also handle invalid indices"""
        # Initialize MCTS
        mcts = MCTS(base_mcts_config, mock_evaluator)
        mcts._ensure_root_initialized(empty_gomoku_state)
        
        tree_ops = mcts.tree_ops
        num_nodes = mcts.tree.num_nodes
        invalid_idx = num_nodes + 50
        
        # Test other operations that might have similar issues
        # Note: This test may need to be updated based on what methods exist
        
        # Test methods that should handle invalid indices gracefully
        if hasattr(tree_ops, 'get_children'):
            children = tree_ops.get_children(invalid_idx)
            assert children is None or len(children) == 0
            
        if hasattr(tree_ops, 'get_node_info'):
            try:
                info = tree_ops.get_node_info(invalid_idx)
                # Should either return None or handle gracefully
                assert info is None or isinstance(info, dict)
            except (IndexError, RuntimeError):
                pytest.fail(f"get_node_info should handle invalid index {invalid_idx} gracefully")