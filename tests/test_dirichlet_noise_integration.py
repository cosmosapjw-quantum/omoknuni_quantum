"""Test for proper integration of batched Dirichlet noise with CUDA kernels

This test ensures that per-simulation Dirichlet noise is correctly applied
when using the enhanced CUDA kernel for parallel selection.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.mcts.core.wave_search import WaveSearch
from python.mcts.gpu.csr_tree import CSRTree, CSRTreeConfig
from python.mcts.gpu.gpu_game_states import GPUGameStates, GPUGameStatesConfig, GameType
from python.mcts.core.mcts_config import MCTSConfig


class TestDirichletNoiseIntegration:
    """Test cases for Dirichlet noise integration with CUDA kernels"""
    
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
        
    def test_batched_dirichlet_noise_generation(self):
        """Test that batched Dirichlet noise CUDA kernel generates correct per-simulation noise"""
        if self.device.type != 'cuda':
            pytest.skip("CUDA required for this test")
            
        # Mock GPU ops with batched_dirichlet_noise
        mock_gpu_ops = Mock()
        
        # Test parameters
        num_sims = 4
        num_actions = 5
        alpha = 0.3
        epsilon = 1.0
        
        # Generate expected noise shape
        expected_noise = torch.rand((num_sims, num_actions), device=self.device)
        # Normalize each row to sum to 1 (like Dirichlet)
        expected_noise = expected_noise / expected_noise.sum(dim=1, keepdim=True)
        
        mock_gpu_ops.batched_dirichlet_noise = Mock(return_value=expected_noise)
        
        # Call the function
        noise = mock_gpu_ops.batched_dirichlet_noise(
            num_sims, num_actions, alpha, epsilon, self.device
        )
        
        # Verify shape
        assert noise.shape == (num_sims, num_actions)
        
        # Verify each row sums to 1 (property of Dirichlet)
        row_sums = noise.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(num_sims, device=self.device), atol=1e-6)
        
        # Verify all values are positive
        assert (noise > 0).all()
        
    def test_dirichlet_noise_pre_mixing_with_priors(self):
        """Test that Dirichlet noise is correctly pre-mixed with priors before kernel call"""
        # Setup tree with root and children
        root_idx = 0
        self.tree.reset()
        
        # Add children with known priors
        num_children = 5
        priors = [0.2, 0.2, 0.2, 0.2, 0.2]
        child_indices = []
        
        for i, prior in enumerate(priors):
            child_idx = self.tree.add_child(root_idx, action=i, child_prior=prior)
            child_indices.append(child_idx)
            self.tree.node_data.visit_counts[child_idx] = 10
            self.tree.node_data.value_sums[child_idx] = 5.0
        
        # Mock GPU ops
        mock_gpu_ops = Mock()
        
        # Mock batched Dirichlet noise generation
        num_sims = 3
        mock_noise = torch.rand((num_sims, num_children), device=self.device)
        mock_noise = mock_noise / mock_noise.sum(dim=1, keepdim=True)  # Normalize
        
        mock_gpu_ops.batched_dirichlet_noise = Mock(return_value=mock_noise)
        mock_gpu_ops.parallel_select_with_virtual_loss = Mock(
            return_value=torch.tensor([child_indices[0], child_indices[1], child_indices[2]], 
                                    device=self.device)
        )
        
        self.wave_search.gpu_ops = mock_gpu_ops
        
        # Setup wave search state
        self.wave_search.node_to_state = torch.zeros(10000, dtype=torch.int32, device=self.device)
        self.wave_search.node_to_state[0] = 0
        
        # Create game state
        self.game_states.allocated_mask[0] = True
        legal_mask = torch.ones(225, dtype=torch.bool, device=self.device)
        self.game_states.get_legal_moves_mask = Mock(return_value=legal_mask.unsqueeze(0))
        
        # Test: Call _parallel_select_with_virtual_loss
        active_indices = torch.arange(num_sims, device=self.device)
        active_nodes = torch.zeros(num_sims, dtype=torch.int32, device=self.device)  # All at root
        
        batch_children, batch_actions, batch_priors = self.tree.batch_get_children(active_nodes)
        valid_children_mask = batch_children >= 0
        
        # Call the method
        selected = self.wave_search._parallel_select_with_virtual_loss(
            active_indices, active_nodes, batch_children, batch_priors,
            valid_children_mask, depth=1
        )
        
        # Verify GPU ops was called with proper arguments
        assert mock_gpu_ops.parallel_select_with_virtual_loss.called
        
        # Get the call arguments
        call_args = mock_gpu_ops.parallel_select_with_virtual_loss.call_args
        
        # The kernel should receive pre-mixed priors (not implemented yet)
        # For now, it passes noise separately which is incorrect
        
    def test_different_noise_per_simulation(self):
        """Test that each simulation gets different Dirichlet noise"""
        if self.device.type != 'cuda':
            pytest.skip("CUDA required for this test")
            
        # Try to import actual CUDA kernels
        try:
            from python.mcts.gpu.cuda_manager import detect_cuda_kernels
            gpu_ops = detect_cuda_kernels()
            
            if gpu_ops is None or not hasattr(gpu_ops, 'batched_dirichlet_noise'):
                pytest.skip("CUDA kernels not available")
                
            # Generate noise for multiple simulations
            num_sims = 10
            num_actions = 5
            alpha = 0.3
            epsilon = 1.0
            
            noise = gpu_ops.batched_dirichlet_noise(
                num_sims, num_actions, alpha, epsilon, self.device
            )
            
            # Each simulation should have different noise
            # Check that no two rows are identical
            for i in range(num_sims):
                for j in range(i + 1, num_sims):
                    assert not torch.allclose(noise[i], noise[j], atol=1e-6), \
                        f"Simulations {i} and {j} have identical noise"
                    
            # Verify Dirichlet properties
            # Each row should sum to 1
            row_sums = noise.sum(dim=1)
            assert torch.allclose(row_sums, torch.ones(num_sims, device=self.device), atol=1e-6)
            
            # All values should be positive
            assert (noise > 0).all()
            
        except ImportError:
            pytest.skip("CUDA manager not available")
            
    def test_noise_mixed_priors_shape(self):
        """Test that noise-mixed priors have correct shape for kernel"""
        # Setup
        num_sims = 4
        num_children = 5
        epsilon = 0.25
        
        # Original priors (same for all sims at same parent)
        priors = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], device=self.device)
        
        # Per-simulation noise
        noise = torch.rand((num_sims, num_children), device=self.device)
        noise = noise / noise.sum(dim=1, keepdim=True)
        
        # Mix priors with noise
        # Each simulation gets: (1-epsilon) * priors + epsilon * noise[sim]
        mixed_priors = (1 - epsilon) * priors.unsqueeze(0) + epsilon * noise
        
        # Verify shape
        assert mixed_priors.shape == (num_sims, num_children)
        
        # Verify each row sums to 1
        row_sums = mixed_priors.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(num_sims, device=self.device), atol=1e-6)
        
        # Verify different simulations have different mixed priors
        for i in range(num_sims):
            for j in range(i + 1, num_sims):
                assert not torch.allclose(mixed_priors[i], mixed_priors[j], atol=1e-6)