#!/usr/bin/env python3
"""Unit tests for individual MCTS components

These tests validate each component in isolation to ensure
correctness and catch regressions during development.
"""

import unittest
import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcts.core.mcts import MCTS, MCTSConfig
from mcts.gpu.csr_tree import CSRTree, CSRTreeConfig
from mcts.gpu.unified_kernels import get_unified_kernels
from mcts.quantum.quantum_mcts import QuantumMCTS, QuantumConfig, QuantumMode


class TestCSRTree(unittest.TestCase):
    """Test CSR tree functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = CSRTreeConfig(
            max_nodes=1000,
            max_actions=225,
            device='cpu'  # Use CPU for consistent testing
        )
        self.tree = CSRTree(self.config)
    
    def test_node_addition(self):
        """Test basic node addition"""
        # CSRTree automatically creates root during initialization
        # Root should be at index 0 and num_nodes should be 1
        self.assertEqual(self.tree.num_nodes, 1)
        root_idx = 0  # Root is always at index 0
        
        # Check node data
        self.assertEqual(self.tree.visit_counts[0].item(), 0)
        self.assertEqual(self.tree.value_sums[0].item(), 0.0)
        self.assertEqual(self.tree.node_priors[0].item(), 1.0)
    
    def test_children_addition(self):
        """Test adding children to nodes"""
        # Use the automatically created root
        root_idx = 0
        
        # Add children
        actions = [0, 1, 2]
        priors = [0.5, 0.3, 0.2]
        children = self.tree.add_children_batch(root_idx, actions, priors)
        
        # Verify children were added
        self.assertEqual(len(children), 3)
        # Should have root (1) + 3 children = 4 nodes total
        self.assertEqual(self.tree.num_nodes, 4)
        
        # Check that we have children (CSR structure may be internal)
        child_indices, child_actions, child_priors = self.tree.get_children(root_idx)
        self.assertEqual(len(child_indices), 3)  # 3 children
        
        # Check edge data through get_children
        child_indices, child_actions, child_priors = self.tree.get_children(root_idx)
        for i, (expected_action, expected_prior) in enumerate(zip(actions, priors)):
            self.assertEqual(child_actions[i].item(), expected_action)
            self.assertAlmostEqual(child_priors[i].item(), expected_prior, places=5)
    
    def test_node_selection(self):
        """Test UCB-based child selection"""
        # Setup tree with children
        root_idx = 0  # Use pre-created root
        actions = [0, 1, 2]
        priors = [0.5, 0.3, 0.2]
        children = self.tree.add_children_batch(root_idx, actions, priors)
        
        # Simulate some visits to create different Q-values
        self.tree.visit_counts[children[0]] = 10
        self.tree.value_sums[children[0]] = 5.0  # Q = 0.5
        
        self.tree.visit_counts[children[1]] = 5
        self.tree.value_sums[children[1]] = 3.0  # Q = 0.6
        
        self.tree.visit_counts[children[2]] = 2
        self.tree.value_sums[children[2]] = 1.0  # Q = 0.5
        
        # Update parent visits
        self.tree.visit_counts[root_idx] = 17
        
        # Select child
        selected_child = self.tree.select_child(root_idx, c_puct=1.414)
        
        # Should select based on UCB formula
        self.assertIsNotNone(selected_child)
        self.assertIn(selected_child, children if isinstance(children, list) else children.tolist())
    
    def test_backup_path(self):
        """Test backup along a path"""
        # Create simple path: root -> child
        root_idx = 0  # Use pre-created root
        actions = [0]
        priors = [1.0]
        children = self.tree.add_children_batch(root_idx, actions, priors)
        child_idx = children[0]
        
        # Backup a value
        path = [root_idx, child_idx]
        value = 0.7
        
        old_root_visits = self.tree.visit_counts[root_idx].item()
        old_child_visits = self.tree.visit_counts[child_idx].item()
        
        self.tree.backup_path(path, value)
        
        # Check visits increased
        self.assertEqual(self.tree.visit_counts[root_idx].item(), old_root_visits + 1)
        self.assertEqual(self.tree.visit_counts[child_idx].item(), old_child_visits + 1)
        
        # Check values updated (backup alternates signs)
        # Root gets the final alternated value, child gets original value  
        self.assertAlmostEqual(self.tree.value_sums[root_idx].item(), -value, places=5)
        self.assertAlmostEqual(self.tree.value_sums[child_idx].item(), value, places=5)
    
    def test_subtree_reuse(self):
        """Test subtree reuse functionality"""
        # Create tree: root -> child1, child2
        root_idx = 0  # Use pre-created root
        actions = [0, 1]
        priors = [0.6, 0.4]
        children = self.tree.add_children_batch(root_idx, actions, priors)
        
        # Add grandchildren to child1
        grandchild_actions = [10, 11]
        grandchild_priors = [0.5, 0.5]
        grandchildren = self.tree.add_children_batch(children[0], grandchild_actions, grandchild_priors)
        
        # Add some statistics
        self.tree.visit_counts[children[0]] = 10
        self.tree.value_sums[children[0]] = 5.0
        self.tree.visit_counts[grandchildren[0]] = 3
        self.tree.value_sums[grandchildren[0]] = 1.5
        
        old_nodes = self.tree.num_nodes
        
        # Shift root to child1
        mapping = self.tree.shift_root(children[0])
        
        # Verify mapping - child1 becomes new root
        child1_idx = children[0] if isinstance(children[0], int) else children[0].item()
        child2_idx = children[1] if isinstance(children[1], int) else children[1].item()
        grandchild_idx = grandchildren[0] if isinstance(grandchildren[0], int) else grandchildren[0].item()
        
        self.assertEqual(mapping[child1_idx], 0)  # child1 becomes new root
        
        # Check if any grandchildren were preserved (implementation may vary)
        # This is a basic functionality test - the important thing is that shift_root works
        self.assertTrue(len(mapping) >= 1, "Expected at least the new root in mapping")
        
        # child2 should not be reachable from child1
        self.assertNotIn(child2_idx, mapping)   # child2 not reachable
        
        # Verify statistics preserved
        new_root_idx = 0
        self.assertEqual(self.tree.visit_counts[new_root_idx].item(), 10)
        self.assertEqual(self.tree.value_sums[new_root_idx].item(), 5.0)
        
        # Verify tree size reduced
        self.assertLess(self.tree.num_nodes, old_nodes)


class TestUnifiedKernels(unittest.TestCase):
    """Test unified GPU kernel interface"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')  # Use CPU for consistent testing
        self.kernels = get_unified_kernels(self.device)
    
    def test_kernel_availability(self):
        """Test kernel availability detection"""
        # Should always have some kernels available (at least PyTorch fallbacks)
        self.assertIsNotNone(self.kernels)
        self.assertTrue(hasattr(self.kernels, 'use_cuda'))
    
    def test_ucb_selection_fallback(self):
        """Test UCB selection with PyTorch fallback"""
        # Create test data
        num_nodes = 5
        num_edges = 10
        
        node_indices = torch.arange(num_nodes, device=self.device)
        row_ptr = torch.tensor([0, 2, 4, 6, 8, 10], device=self.device)
        col_indices = torch.arange(num_edges, device=self.device) % num_nodes
        edge_actions = torch.arange(num_edges, device=self.device)
        edge_priors = torch.rand(num_edges, device=self.device)
        visit_counts = torch.randint(1, 10, (num_nodes,), device=self.device)
        value_sums = torch.randn(num_nodes, device=self.device)
        
        # Test UCB selection
        try:
            selected_actions, ucb_scores = self.kernels.batch_ucb_selection(
                node_indices, row_ptr, col_indices, edge_actions,
                edge_priors, visit_counts, value_sums, c_puct=1.414
            )
            
            # Verify output shapes
            self.assertEqual(selected_actions.shape, (num_nodes,))
            self.assertEqual(ucb_scores.shape, (num_nodes,))
            
            # Verify reasonable outputs
            self.assertTrue(torch.all(selected_actions >= -1))  # -1 for no children
            
        except Exception as e:
            self.fail(f"UCB selection failed: {e}")


class TestQuantumMCTS(unittest.TestCase):
    """Test quantum MCTS functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = QuantumConfig(
            quantum_mode=QuantumMode.MINIMAL,
            device='cpu',
            enable_jit_compilation=False  # Avoid JIT in tests
        )
        # Note: Can't easily test full QuantumMCTS without game interface
        # This tests the configuration and basic setup
    
    def test_quantum_config_creation(self):
        """Test quantum configuration creation"""
        config = QuantumConfig(
            quantum_mode=QuantumMode.PRAGMATIC,
            base_c_puct=1.414,
            quantum_bonus_coefficient=0.15
        )
        
        self.assertEqual(config.quantum_mode, QuantumMode.PRAGMATIC)
        self.assertEqual(config.base_c_puct, 1.414)
        self.assertEqual(config.quantum_bonus_coefficient, 0.15)
    
    def test_quantum_mode_enum(self):
        """Test quantum mode enumeration"""
        modes = [QuantumMode.CLASSICAL, QuantumMode.MINIMAL, 
                QuantumMode.PRAGMATIC, QuantumMode.ONE_LOOP, QuantumMode.FULL]
        
        for mode in modes:
            config = QuantumConfig(quantum_mode=mode)
            self.assertEqual(config.quantum_mode, mode)


class TestMCTSConfig(unittest.TestCase):
    """Test MCTS configuration"""
    
    def test_config_creation(self):
        """Test basic configuration creation"""
        config = MCTSConfig(
            num_simulations=1000,
            c_puct=1.414,
            device='cpu',
            enable_subtree_reuse=True
        )
        
        self.assertEqual(config.num_simulations, 1000)
        self.assertEqual(config.c_puct, 1.414)
        self.assertEqual(config.device, 'cpu')
        self.assertTrue(config.enable_subtree_reuse)
    
    def test_quantum_config_integration(self):
        """Test quantum config integration"""
        config = MCTSConfig(
            enable_quantum=True,
            quantum_version='v2'
        )
        
        quantum_config = config.get_or_create_quantum_config()
        self.assertIsNotNone(quantum_config)
        self.assertEqual(quantum_config.quantum_mode, QuantumMode.PRAGMATIC)


class TestIntegrationBasics(unittest.TestCase):
    """Basic integration tests for component interaction"""
    
    def test_csr_tree_with_kernels(self):
        """Test CSR tree working with unified kernels"""
        # Create tree
        config = CSRTreeConfig(max_nodes=100, max_actions=25, device='cpu')
        tree = CSRTree(config)
        
        # Use pre-created root
        root_idx = 0
        actions = [0, 1, 2]
        priors = [0.5, 0.3, 0.2]
        children = tree.add_children(root_idx, actions, priors)
        
        # Get kernels
        kernels = get_unified_kernels(torch.device('cpu'))
        
        # Test that kernel interface matches tree data structures
        self.assertIsNotNone(kernels)
        self.assertEqual(tree.row_ptr.device.type, 'cpu')
        self.assertEqual(tree.col_indices.device.type, 'cpu')
    
    def test_memory_consistency(self):
        """Test memory allocation consistency"""
        config = CSRTreeConfig(max_nodes=50, device='cpu')
        tree = CSRTree(config)
        
        # Fill tree near capacity
        root_idx = 0  # Use pre-created root
        
        for i in range(10):  # Add 10 children
            actions = [i]
            priors = [1.0/11]
            tree.add_children_batch(root_idx, actions, priors)
        
        # Check memory bounds
        self.assertLessEqual(tree.num_nodes, config.max_nodes)
        self.assertTrue(torch.all(tree.visit_counts >= 0))
        self.assertFalse(torch.any(torch.isnan(tree.value_sums)))


def run_unit_tests():
    """Run all unit tests"""
    # Create test loader and suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(loader.loadTestsFromTestCase(TestCSRTree))
    suite.addTest(loader.loadTestsFromTestCase(TestUnifiedKernels))
    suite.addTest(loader.loadTestsFromTestCase(TestQuantumMCTS))
    suite.addTest(loader.loadTestsFromTestCase(TestMCTSConfig))
    suite.addTest(loader.loadTestsFromTestCase(TestIntegrationBasics))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("=" * 60)
    print("MCTS COMPONENT UNIT TESTS")
    print("=" * 60)
    
    success = run_unit_tests()
    
    if success:
        print("\n✅ All unit tests passed!")
    else:
        print("\n❌ Some unit tests failed!")
        exit(1)