"""Test for virtual loss optimization - TDD approach

This test verifies virtual loss behavior and its parallelization.
Tests the sequential behavior first, then the optimized parallel version.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.mcts.core.wave_search import WaveSearch
from python.mcts.gpu.csr_tree import CSRTree, CSRTreeConfig
from python.mcts.gpu.gpu_game_states import GPUGameStates, GPUGameStatesConfig, GameType
from python.mcts.core.mcts_config import MCTSConfig


class TestVirtualLossOptimization:
    """Test cases for virtual loss optimization"""
    
    def setup_method(self):
        """Setup test environment"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create MCTS config
        self.config = MCTSConfig(
            num_simulations=100,
            c_puct=1.4,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            device=str(self.device),
            enable_virtual_loss=True,
            virtual_loss=1.0,
            max_wave_size=16,
            board_size=15,
            game_type=GameType.GOMOKU,
            max_children_per_node=225
        )
        
        # Create tree
        tree_config = CSRTreeConfig(
            max_nodes=10000,
            max_edges=100000,
            max_actions=225,
            device=str(self.device),
            enable_virtual_loss=True,
            virtual_loss_value=-1.0,
            batch_size=16
        )
        self.tree = CSRTree(tree_config)
        
        # Create game states
        game_config = GPUGameStatesConfig(
            capacity=10000,
            game_type=GameType.GOMOKU,
            board_size=15,
            device=str(self.device)
        )
        self.game_states = GPUGameStates(game_config)
        
        # Create mock evaluator
        self.evaluator = Mock()
        self.evaluator.evaluate_batch = Mock(return_value=(
            np.random.rand(16, 225),  # policies
            np.random.rand(16)        # values
        ))
        
        # Create wave search
        self.wave_search = WaveSearch(
            tree=self.tree,
            game_states=self.game_states,
            evaluator=self.evaluator,
            config=self.config,
            device=self.device,
            gpu_ops=None
        )
        
    def test_sequential_virtual_loss_behavior(self):
        """Test that virtual losses are applied sequentially and affect UCB calculations
        
        This test verifies the CURRENT behavior where virtual losses are applied
        one by one, affecting subsequent selections.
        """
        # Setup: Create a tree with root and children
        root_idx = 0
        self.tree.reset()
        
        # Add some children to root with specific priors and visits
        num_children = 5
        priors = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], device=self.device)
        actions = list(range(num_children))
        
        # Add children manually
        for i, (action, prior) in enumerate(zip(actions, priors)):
            child_idx = self.tree.add_child(root_idx, action, prior.item())
            # Give children some initial visits and values
            # Make them slightly different to ensure deterministic selection order
            self.tree.node_data.visit_counts[child_idx] = 10
            self.tree.node_data.value_sums[child_idx] = 5.0 - i * 0.01  # Q = 0.5, 0.49, 0.48, 0.47, 0.46
        
        # Setup wave search state
        self.wave_search.node_to_state = torch.zeros(10000, dtype=torch.int32, device=self.device)
        self.wave_search.node_to_state[0] = 0  # Root has state 0
        
        # Create game state for root
        self.game_states.allocated_mask[0] = True
        legal_mask = torch.ones(225, dtype=torch.bool, device=self.device)
        self.game_states.boards[0] = 0  # Empty board
        
        # Mock get_legal_moves_mask to return our legal mask
        self.game_states.get_legal_moves_mask = Mock(return_value=legal_mask.unsqueeze(0))
        
        # Test: Multiple simulations at the same parent node
        num_sims = 4
        active_indices = torch.arange(num_sims, device=self.device)
        active_nodes = torch.zeros(num_sims, dtype=torch.int32, device=self.device)  # All at root
        
        # Get children info using batch_get_children
        active_nodes_for_batch = torch.zeros(num_sims, dtype=torch.int32, device=self.device)
        batch_children, batch_actions, batch_priors = self.tree.batch_get_children(active_nodes_for_batch)
        valid_children_mask = batch_children >= 0
        
        # Record initial virtual loss counts
        initial_virtual_loss_counts = self.tree.node_data.virtual_loss_counts.clone()
        
        # Call the method we're testing
        selected_children = self.wave_search._parallel_select_with_virtual_loss(
            active_indices, active_nodes, batch_children, batch_priors, 
            valid_children_mask, depth=1
        )
        
        # Verify virtual losses were applied
        assert selected_children is not None
        assert len(selected_children) == num_sims
        
        # Check that virtual losses were applied to selected children
        for child in selected_children:
            if child >= 0:
                child_idx = child.item()
                # Virtual loss count should have increased
                assert self.tree.node_data.virtual_loss_counts[child_idx] > initial_virtual_loss_counts[child_idx]
        
        # Verify that multiple simulations selected different children due to virtual loss
        # (This is the sequential behavior we want to parallelize)
        unique_selections = torch.unique(selected_children[selected_children >= 0])
        assert len(unique_selections) > 1, "Sequential virtual loss should cause diverse selections"
        
    def test_virtual_loss_affects_ucb_calculation(self):
        """Test that virtual losses properly affect UCB score calculations"""
        # Setup similar tree
        root_idx = 0
        self.tree.reset()
        
        # Add one child with known values
        child_idx = self.tree.add_child(root_idx, action=0, child_prior=0.5)
        self.tree.node_data.visit_counts[child_idx] = 10
        self.tree.node_data.value_sums[child_idx] = 5.0  # Q = 0.5
        
        # Calculate UCB before virtual loss
        parent_visits = self.tree.node_data.visit_counts[root_idx]
        q_before = self.tree.node_data.value_sums[child_idx] / self.tree.node_data.visit_counts[child_idx]
        
        # Apply virtual loss
        self.tree.node_data.apply_virtual_loss(torch.tensor([child_idx], device=self.device))
        
        # Get effective values after virtual loss
        effective_visits = self.tree.node_data.get_effective_visits(torch.tensor([child_idx], device=self.device))
        effective_values = self.tree.node_data.get_effective_values(torch.tensor([child_idx], device=self.device))
        
        # Calculate Q after virtual loss
        q_after = effective_values[0] / effective_visits[0]
        
        # Virtual loss should decrease Q value
        assert q_after < q_before, f"Q should decrease with virtual loss: {q_after} >= {q_before}"
        
        # Remove virtual loss
        self.tree.node_data.remove_virtual_loss(torch.tensor([child_idx], device=self.device))
        
        # Q should be restored
        q_restored = self.tree.node_data.value_sums[child_idx] / self.tree.node_data.visit_counts[child_idx]
        assert torch.isclose(q_restored, q_before), "Q should be restored after removing virtual loss"
        
    def test_virtual_loss_removal_after_wave(self):
        """Test that virtual losses are properly removed after wave completion"""
        # This test verifies that _remove_virtual_losses_from_paths works correctly
        
        # Setup paths through tree
        wave_size = 4
        max_depth = 5
        paths = torch.zeros((wave_size, max_depth), dtype=torch.int32, device=self.device)
        path_lengths = torch.tensor([3, 2, 4, 3], device=self.device)
        
        # Create some nodes and apply virtual losses
        for i in range(10):
            self.tree.node_data.allocate_node(prior=0.1, parent_idx=-1, parent_action=-1)
        
        # Setup paths
        paths[0, :3] = torch.tensor([0, 1, 2])
        paths[1, :2] = torch.tensor([0, 1])
        paths[2, :4] = torch.tensor([0, 1, 3, 4])
        paths[3, :3] = torch.tensor([0, 1, 5])
        
        # Apply virtual losses to nodes in paths
        nodes_with_vl = [1, 2, 3, 4, 5]  # Skip root (0)
        for node in nodes_with_vl:
            self.tree.node_data.apply_virtual_loss(torch.tensor([node], device=self.device))
        
        # Record virtual losses before removal
        vl_before = self.tree.node_data.virtual_loss_counts.clone()
        
        # Remove virtual losses from paths
        print(f"Config enable_virtual_loss: {self.config.enable_virtual_loss}")
        print(f"VL before removal: {vl_before[:10]}")
        print(f"Paths: {paths}")
        print(f"Path lengths: {path_lengths}")
        self.wave_search._remove_virtual_losses_from_paths(paths, path_lengths)
        print(f"VL after removal: {self.tree.node_data.virtual_loss_counts[:10]}")
        
        # Check that virtual losses were removed correctly
        # We applied 1 virtual loss to each node initially
        # The removal should remove all virtual losses since each node
        # only had 1 virtual loss applied
        
        # All nodes should have their virtual losses removed completely
        expected_final_vl = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        for node, expected_final in expected_final_vl.items():
            actual_final = self.tree.node_data.virtual_loss_counts[node].item()
            assert actual_final == expected_final, f"Node {node}: expected final VL={expected_final}, got {actual_final}"


class TestBatchedVirtualLoss:
    """Test cases for the new batched virtual loss implementation"""
    
    def setup_method(self):
        """Setup test environment"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Similar setup as TestVirtualLossOptimization
        self.config = MCTSConfig(
            num_simulations=100,
            c_puct=1.4,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            device=str(self.device),
            enable_virtual_loss=True,
            virtual_loss=1.0,
            max_wave_size=16,
            board_size=15,
            game_type=GameType.GOMOKU,
            max_children_per_node=225
        )
        
        # Create tree
        tree_config = CSRTreeConfig(
            max_nodes=10000,
            max_edges=100000,
            max_actions=225,
            device=str(self.device),
            enable_virtual_loss=True,
            virtual_loss_value=-1.0,
            batch_size=16
        )
        self.tree = CSRTree(tree_config)
        
        # Create game states
        game_config = GPUGameStatesConfig(
            capacity=10000,
            game_type=GameType.GOMOKU,
            board_size=15,
            device=str(self.device)
        )
        self.game_states = GPUGameStates(game_config)
        
        # Create mock evaluator
        self.evaluator = Mock()
        self.evaluator.evaluate_batch = Mock(return_value=(
            np.random.rand(16, 225),  # policies
            np.random.rand(16)        # values
        ))
        
        # Create wave search
        self.wave_search = WaveSearch(
            tree=self.tree,
            game_states=self.game_states,
            evaluator=self.evaluator,
            config=self.config,
            device=self.device,
            gpu_ops=None
        )
    
    def test_batched_virtual_loss_produces_diverse_selections(self):
        """Test that batched virtual loss implementation produces diverse selections
        
        The optimized version should still ensure different simulations select
        different children when at the same parent node.
        """
        # Setup: Create a tree with root and children
        root_idx = 0
        self.tree.reset()
        
        # Add some children to root with specific priors and visits
        num_children = 5
        priors = [0.2, 0.2, 0.2, 0.2, 0.2]
        actions = list(range(num_children))
        
        # Add children manually with slightly different statistics
        for i, (action, prior) in enumerate(zip(actions, priors)):
            child_idx = self.tree.add_child(root_idx, action, prior)
            # Give children slightly different visits and values to ensure deterministic order
            self.tree.node_data.visit_counts[child_idx] = 10
            self.tree.node_data.value_sums[child_idx] = 5.0 - i * 0.01  # Q = 0.5, 0.49, 0.48, 0.47, 0.46
        
        # Setup wave search state
        self.wave_search.node_to_state = torch.zeros(10000, dtype=torch.int32, device=self.device)
        self.wave_search.node_to_state[0] = 0  # Root has state 0
        
        # Create game state for root
        self.game_states.allocated_mask[0] = True
        legal_mask = torch.ones(225, dtype=torch.bool, device=self.device)
        self.game_states.boards[0] = 0  # Empty board
        
        # Mock get_legal_moves_mask to return our legal mask
        self.game_states.get_legal_moves_mask = Mock(return_value=legal_mask.unsqueeze(0))
        
        # Test: Multiple simulations at the same parent node
        num_sims = 4
        active_indices = torch.arange(num_sims, device=self.device)
        active_nodes = torch.zeros(num_sims, dtype=torch.int32, device=self.device)  # All at root
        
        # Get children info using batch_get_children
        batch_children, batch_actions, batch_priors = self.tree.batch_get_children(active_nodes)
        valid_children_mask = batch_children >= 0
        
        # Call the optimized method
        selected_children = self.wave_search._parallel_select_with_virtual_loss(
            active_indices, active_nodes, batch_children, batch_priors, 
            valid_children_mask, depth=1
        )
        
        # Verify results
        assert selected_children is not None
        assert len(selected_children) == num_sims
        
        # Check that different children were selected (diversity)
        unique_selections = torch.unique(selected_children[selected_children >= 0])
        assert len(unique_selections) > 1, "Optimized version should produce diverse selections"
        
        # Verify virtual losses were applied
        for child in selected_children:
            if child >= 0:
                assert self.tree.node_data.virtual_loss_counts[child] > 0
    
    def test_cuda_kernel_availability(self):
        """Test if CUDA kernels are properly loaded"""
        if self.device.type == 'cuda':
            try:
                from ..python.mcts.gpu.cuda_manager import detect_cuda_kernels
                kernels = detect_cuda_kernels()
                
                if kernels is not None:
                    # Check if our new functions are available
                    assert hasattr(kernels, 'batch_apply_virtual_loss'), "batch_apply_virtual_loss kernel missing"
                    assert hasattr(kernels, 'batch_remove_virtual_loss'), "batch_remove_virtual_loss kernel missing"
                    assert hasattr(kernels, 'parallel_select_with_virtual_loss'), "parallel_select_with_virtual_loss kernel missing"
                else:
                    pytest.skip("CUDA kernels not available")
            except ImportError:
                pytest.skip("CUDA manager not available")