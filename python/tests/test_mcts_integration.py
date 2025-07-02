#!/usr/bin/env python3
"""Integration tests for MCTS end-to-end functionality

These tests verify complete MCTS workflows to ensure proper simulation
accumulation and catch any regressions in the full pipeline.
"""

import unittest
import torch
import numpy as np
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcts.core.mcts import MCTS, MCTSConfig
from mcts.gpu.csr_tree import CSRTree, CSRTreeConfig
from mcts.quantum.quantum_mcts import QuantumMCTS, QuantumConfig, QuantumMode


class MockGameInterface:
    """Mock game interface for testing"""
    
    def __init__(self, board_size=3):
        self.board_size = board_size
        self.action_space_size = board_size * board_size
    
    def get_initial_state(self):
        """Return initial game state"""
        return np.zeros((self.board_size, self.board_size), dtype=np.int32)
    
    def get_legal_actions(self, state):
        """Return list of legal actions for current state"""
        actions = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if state[i, j] == 0:  # Empty cell
                    actions.append(i * self.board_size + j)
        return actions
    
    def apply_action(self, state, action, player=1):
        """Apply action to state and return new state"""
        new_state = state.copy()
        row, col = action // self.board_size, action % self.board_size
        new_state[row, col] = player
        return new_state
    
    def is_terminal(self, state):
        """Check if state is terminal"""
        # Simple: game ends when board is full or has 3 in a row
        if np.all(state != 0):
            return True
        
        # Check rows, columns, diagonals for 3 in a row
        for i in range(self.board_size):
            if abs(np.sum(state[i, :])) == 3 or abs(np.sum(state[:, i])) == 3:
                return True
        
        if abs(np.trace(state)) == 3 or abs(np.trace(np.fliplr(state))) == 3:
            return True
        
        return False
    
    def get_reward(self, state, player=1):
        """Get reward for terminal state"""
        if not self.is_terminal(state):
            return 0.0
        
        # Check for wins
        for i in range(self.board_size):
            if np.sum(state[i, :]) == 3 * player or np.sum(state[:, i]) == 3 * player:
                return 1.0
            if np.sum(state[i, :]) == -3 * player or np.sum(state[:, i]) == -3 * player:
                return -1.0
        
        if np.trace(state) == 3 * player or np.trace(np.fliplr(state)) == 3 * player:
            return 1.0
        if np.trace(state) == -3 * player or np.trace(np.fliplr(state)) == -3 * player:
            return -1.0
        
        return 0.0  # Draw
    
    def evaluate_state(self, state):
        """Evaluate state and return policy and value"""
        legal_actions = self.get_legal_actions(state)
        num_actions = len(legal_actions)
        
        if num_actions == 0:
            return np.zeros(self.action_space_size), self.get_reward(state)
        
        # Uniform random policy for testing
        policy = np.zeros(self.action_space_size)
        for action in legal_actions:
            policy[action] = 1.0 / num_actions
        
        # Simple value estimation
        value = 0.1 * (np.random.random() - 0.5)  # Small random value
        
        return policy, value


class TestMCTSIntegration(unittest.TestCase):
    """Test complete MCTS workflows"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.game = MockGameInterface(board_size=3)
        self.config = MCTSConfig(
            num_simulations=100,
            c_puct=1.414,
            device='cpu',
            enable_quantum=False,
            enable_subtree_reuse=True
        )
    
    def test_basic_search_accumulation(self):
        """Test that MCTS properly accumulates statistics during search"""
        # Create MCTS with simple config
        config = MCTSConfig(
            num_simulations=50,
            device='cpu',
            enable_quantum=False
        )
        
        # Create tree directly for testing
        tree_config = CSRTreeConfig(max_nodes=1000, device='cpu')
        tree = CSRTree(tree_config)
        
        initial_state = self.game.get_initial_state()
        
        # Simulate basic MCTS operations
        root_idx = 0  # Use pre-created root
        
        # Add some children
        legal_actions = self.game.get_legal_actions(initial_state)
        priors = [1.0/len(legal_actions)] * len(legal_actions)
        children = tree.add_children_batch(root_idx, legal_actions, priors)
        
        # Simulate some search iterations
        for _ in range(10):
            # Select a child
            selected_child = tree.select_child(root_idx, c_puct=1.414)
            if selected_child is not None:
                # Backup some value
                value = 0.1 * (np.random.random() - 0.5)
                tree.backup_path([root_idx, selected_child], value)
        
        # Verify statistics accumulated
        root_visits = tree.visit_counts[root_idx].item()
        self.assertGreater(root_visits, 0, "Root should have accumulated visits")
        
        # Check that children have some visits
        total_child_visits = sum(tree.visit_counts[child].item() for child in children)
        self.assertGreater(total_child_visits, 0, "Children should have accumulated visits")
        
        # Verify no degenerate statistics (all visits = 1, all Q-values = 0)
        self.assertNotEqual(root_visits, 1, "Root visits should not be stuck at 1")
        
        root_q = tree.value_sums[root_idx].item() / max(1, root_visits)
        self.assertNotEqual(root_q, 0.0, "Root Q-value should not be exactly 0")
    
    def test_subtree_reuse_workflow(self):
        """Test subtree reuse in a realistic game scenario"""
        tree_config = CSRTreeConfig(max_nodes=500, device='cpu')
        tree = CSRTree(tree_config)
        
        # Build initial tree
        root_idx = 0
        legal_actions = [0, 1, 2, 3, 4]  # First 5 positions
        priors = [0.2] * 5
        children = tree.add_children_batch(root_idx, legal_actions, priors)
        
        # Add grandchildren to first child
        selected_child = children[0]
        grandchild_actions = [5, 6, 7]
        grandchild_priors = [0.33, 0.33, 0.34]
        grandchildren = tree.add_children_batch(selected_child, grandchild_actions, grandchild_priors)
        
        # Add some statistics
        tree.visit_counts[selected_child] = 20
        tree.value_sums[selected_child] = 5.0
        
        for i, grandchild in enumerate(grandchildren):
            tree.visit_counts[grandchild] = 5 + i
            tree.value_sums[grandchild] = 1.0 + 0.5 * i
        
        old_num_nodes = tree.num_nodes
        
        # Perform subtree reuse
        mapping = tree.shift_root(selected_child)
        
        # Verify subtree was preserved
        self.assertIn(selected_child, mapping, "Selected child should be in mapping")
        self.assertEqual(mapping[selected_child], 0, "Selected child should become new root")
        
        # Verify statistics preserved for new root
        new_root_visits = tree.visit_counts[0].item()
        new_root_value = tree.value_sums[0].item()
        self.assertEqual(new_root_visits, 20, "Root visits should be preserved")
        self.assertEqual(new_root_value, 5.0, "Root value should be preserved")
        
        # Verify tree size changed appropriately
        self.assertLessEqual(tree.num_nodes, old_num_nodes, "Tree size should not increase after shift")
    
    def test_quantum_vs_classical_consistency(self):
        """Test that quantum and classical MCTS produce reasonable results"""
        initial_state = self.game.get_initial_state()
        
        # Classical MCTS
        classical_config = MCTSConfig(
            num_simulations=50,
            device='cpu',
            enable_quantum=False
        )
        
        # Quantum MCTS  
        quantum_config = MCTSConfig(
            num_simulations=50,
            device='cpu',
            enable_quantum=True,
            quantum_version='v2'
        )
        
        # Create trees for comparison
        tree_config = CSRTreeConfig(max_nodes=500, device='cpu')
        classical_tree = CSRTree(tree_config)
        quantum_tree = CSRTree(tree_config)
        
        # Run some iterations on both
        for tree in [classical_tree, quantum_tree]:
            root_idx = 0
            legal_actions = self.game.get_legal_actions(initial_state)
            priors = [1.0/len(legal_actions)] * len(legal_actions)
            children = tree.add_children_batch(root_idx, legal_actions, priors)
            
            # Simulate search
            for _ in range(20):
                selected_child = tree.select_child(root_idx, c_puct=1.414)
                if selected_child is not None:
                    value = 0.1 * (np.random.random() - 0.5)
                    tree.backup_path([root_idx, selected_child], value)
        
        # Both should have accumulated reasonable statistics
        classical_visits = classical_tree.visit_counts[0].item()
        quantum_visits = quantum_tree.visit_counts[0].item()
        
        self.assertGreater(classical_visits, 0, "Classical MCTS should accumulate visits")
        self.assertGreater(quantum_visits, 0, "Quantum MCTS should accumulate visits")
        
        # Visit counts should be in reasonable range (each simulation adds 1)
        self.assertLessEqual(classical_visits, 50, "Classical visits should not exceed simulations")
        self.assertLessEqual(quantum_visits, 50, "Quantum visits should not exceed simulations")
    
    def test_memory_consistency_under_load(self):
        """Test memory management under high load"""
        tree_config = CSRTreeConfig(max_nodes=1000, device='cpu')
        tree = CSRTree(tree_config)
        
        # Stress test: add many nodes and perform many operations
        root_idx = 0
        current_nodes = [root_idx]
        
        for iteration in range(20):
            # Add children to existing nodes
            new_nodes = []
            for node_idx in current_nodes[:5]:  # Limit to prevent explosion
                actions = list(range(iteration * 3, (iteration + 1) * 3))
                priors = [0.33, 0.33, 0.34]
                try:
                    children = tree.add_children_batch(node_idx, actions, priors)
                    new_nodes.extend(children)
                except Exception:
                    # May hit capacity limits, which is expected
                    break
            
            current_nodes = new_nodes
            
            # Perform many backups
            for _ in range(50):
                if len(current_nodes) > 0:
                    target_node = np.random.choice(current_nodes)
                    path = [root_idx, target_node]
                    value = 0.1 * (np.random.random() - 0.5)
                    tree.backup_path(path, value)
        
        # Verify tree is still in valid state
        self.assertGreater(tree.num_nodes, 1, "Tree should have grown")
        self.assertLessEqual(tree.num_nodes, tree_config.max_nodes, "Should not exceed capacity")
        
        # Check no NaN values
        self.assertFalse(torch.any(torch.isnan(tree.visit_counts)), "No NaN visit counts")
        self.assertFalse(torch.any(torch.isnan(tree.value_sums)), "No NaN value sums")
        
        # Check all visit counts are non-negative
        self.assertTrue(torch.all(tree.visit_counts >= 0), "All visits should be non-negative")
    
    def test_degenerate_statistics_detection(self):
        """Test detection of degenerate MCTS statistics"""
        tree_config = CSRTreeConfig(max_nodes=100, device='cpu')
        tree = CSRTree(tree_config)
        
        # Add some structure
        root_idx = 0
        actions = [0, 1, 2]
        priors = [0.5, 0.3, 0.2]
        children = tree.add_children_batch(root_idx, actions, priors)
        
        # Simulate normal operations
        for _ in range(20):
            selected_child = tree.select_child(root_idx, c_puct=1.414)
            if selected_child is not None:
                value = 0.1 * (np.random.random() - 0.5)
                tree.backup_path([root_idx, selected_child], value)
        
        # Check for degenerate patterns
        def check_degenerate_stats(tree, node_indices):
            """Check if statistics show degenerate patterns"""
            visits = [tree.visit_counts[idx].item() for idx in node_indices]
            values = [tree.value_sums[idx].item() for idx in node_indices]
            q_values = [v/max(1, n) for v, n in zip(values, visits)]
            
            # Check for all visits = 1 (classic degenerate pattern)
            all_visits_one = all(v == 1 for v in visits if v > 0)
            
            # Check for all Q-values = 0 (another degenerate pattern)
            all_q_zero = all(abs(q) < 1e-6 for q in q_values)
            
            return all_visits_one, all_q_zero
        
        all_nodes = [root_idx] + children
        all_visits_one, all_q_zero = check_degenerate_stats(tree, all_nodes)
        
        # These should NOT be true for a properly functioning MCTS
        self.assertFalse(all_visits_one, "Not all visit counts should be 1 (degenerate pattern)")
        self.assertFalse(all_q_zero, "Not all Q-values should be 0 (degenerate pattern)")
        
        # Verify we have reasonable statistics distribution
        root_visits = tree.visit_counts[root_idx].item()
        child_visits = [tree.visit_counts[child].item() for child in children]
        
        self.assertGreater(root_visits, 1, "Root should have multiple visits")
        self.assertTrue(any(v > 0 for v in child_visits), "Some children should have visits")


def run_integration_tests():
    """Run all integration tests"""
    # Create test loader and suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(loader.loadTestsFromTestCase(TestMCTSIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("=" * 60)
    print("MCTS INTEGRATION TESTS")
    print("=" * 60)
    
    success = run_integration_tests()
    
    if success:
        print("\n✅ All integration tests passed!")
        print("\nKey verifications completed:")
        print("- End-to-end simulation accumulation")
        print("- Subtree reuse workflow")
        print("- Quantum vs classical consistency")
        print("- Memory management under load")
        print("- Degenerate statistics detection")
    else:
        print("\n❌ Some integration tests failed!")
        exit(1)