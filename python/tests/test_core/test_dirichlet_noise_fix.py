"""Test Dirichlet noise non-determinism fixes

This module tests that Dirichlet noise generation produces different results
for each simulation, following TDD principles.
"""

import pytest
import torch
import numpy as np
import logging
from unittest.mock import Mock

from mcts.core.wave_search import WaveSearch

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
from mcts.gpu.csr_tree import CSRTree, CSRTreeConfig
from mcts.gpu.gpu_game_states import GPUGameStates, GPUGameStatesConfig, GameType
from mcts.gpu.mcts_gpu_accelerator import get_mcts_gpu_accelerator
from mcts.core.mcts_config import MCTSConfig


@pytest.fixture
def wave_search_setup(device):
    """Set up WaveSearch with necessary components"""
    # Create tree
    tree_config = CSRTreeConfig(
        max_nodes=10000,
        max_edges=100000,
        max_actions=225,
        device=str(device),
        batch_size=32,
        enable_batched_ops=True
    )
    tree = CSRTree(tree_config)
    
    # Create game states
    game_config = GPUGameStatesConfig(
        capacity=10000,
        game_type=GameType.GOMOKU,
        board_size=15,
        device=str(device)
    )
    game_states = GPUGameStates(game_config)
    
    # Create evaluator with dynamic batch sizing
    evaluator = Mock()
    def mock_evaluate_batch(features):
        batch_size = features.shape[0] if hasattr(features, 'shape') else len(features)
        return (
            np.random.rand(batch_size, 225).astype(np.float32),  # Policies
            np.random.uniform(-0.5, 0.5, batch_size).astype(np.float32)  # Values
        )
    evaluator.evaluate_batch = Mock(side_effect=mock_evaluate_batch)
    
    # Create config
    config = MCTSConfig()
    config.device = str(device)
    config.board_size = 15
    config.max_wave_size = 32
    config.dirichlet_epsilon = 0.25
    config.dirichlet_alpha = 0.3
    config.c_puct = 1.4
    
    # Create GPU ops (may be None if CUDA not available)
    gpu_ops = None
    if device.type == 'cuda':
        try:
            gpu_ops = get_mcts_gpu_accelerator(device)
        except:
            pass
    
    # Create WaveSearch
    wave_search = WaveSearch(tree, game_states, evaluator, config, device, gpu_ops)
    
    return {
        'wave_search': wave_search,
        'tree': tree,
        'game_states': game_states,
        'evaluator': evaluator,
        'config': config,
        'device': device
    }


class TestDirichletNoiseFix:
    """Test Dirichlet noise non-determinism fixes"""
    
    def test_dirichlet_noise_produces_different_results(self, wave_search_setup):
        """Test that Dirichlet noise generates different results each time
        
        This test reproduces the issue where cached Dirichlet distributions
        produce identical results when they should be random.
        """
        ws = wave_search_setup['wave_search']
        tree = wave_search_setup['tree']
        
        # Add children to root for testing
        children = tree.add_children_batch(0, [0, 1, 2], [0.3, 0.4, 0.3])
        priors = tree.node_data.node_priors[children]
        
        # Apply noise multiple times - should get different results
        sim_indices = torch.tensor([0, 1, 2], device=ws.device)
        
        # Debug: Check configuration
        print(f"DEBUG: epsilon = {ws.config.dirichlet_epsilon}")
        print(f"DEBUG: alpha = {ws.config.dirichlet_alpha}")
        print(f"DEBUG: original priors = {priors}")
        
        # First application
        noised_priors_1 = ws.apply_per_simulation_dirichlet_noise(
            0, sim_indices, children, priors
        )
        
        # Second application - should be different
        noised_priors_2 = ws.apply_per_simulation_dirichlet_noise(
            0, sim_indices, children, priors
        )
        
        # Should have different results for different simulations
        print(f"DEBUG: First noise shape: {noised_priors_1.shape}")
        print(f"DEBUG: Second noise shape: {noised_priors_2.shape}")
        print(f"DEBUG: First noise[0]: {noised_priors_1[0]}")
        print(f"DEBUG: First noise[1]: {noised_priors_1[1]}")
        print(f"DEBUG: Second noise[0]: {noised_priors_2[0]}")
        print(f"DEBUG: Second noise[1]: {noised_priors_2[1]}")
        
        # Different simulations in same call should have different noise
        assert not torch.allclose(noised_priors_1[0], noised_priors_1[1], atol=1e-6), \
            "Same call should give different noise for different simulations"
        assert not torch.allclose(noised_priors_1[1], noised_priors_1[2], atol=1e-6), \
            "Same call should give different noise for different simulations"
            
        # Different calls should also give different results (not identical due to caching)
        assert not torch.allclose(noised_priors_1[0], noised_priors_2[0], atol=1e-6), \
            "Different calls should give different noise (no caching issues)"
            
    def test_dirichlet_distribution_sampling_variation(self, device):
        """Test that torch.distributions.Dirichlet produces different samples"""
        alpha = 0.3
        num_children = 3
        num_sims = 3
        
        # Create distribution
        dist = torch.distributions.Dirichlet(
            torch.full((num_children,), alpha, device=device)
        )
        
        # Sample multiple times
        sample_1 = dist.sample((num_sims,))
        sample_2 = dist.sample((num_sims,))
        
        print(f"DEBUG: Sample 1: {sample_1}")
        print(f"DEBUG: Sample 2: {sample_2}")
        
        # Should be different
        assert not torch.allclose(sample_1, sample_2, atol=1e-6), \
            "Dirichlet samples should be different each time"
            
        # Within same sample, should also be different
        assert not torch.allclose(sample_1[0], sample_1[1], atol=1e-6), \
            "Different simulations should have different noise"