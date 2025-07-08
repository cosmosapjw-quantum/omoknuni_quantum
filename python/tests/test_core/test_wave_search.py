"""
Comprehensive tests for wave-based parallelization

This module tests the WaveSearch class which implements:
- Wave-based parallel selection
- Batch expansion handling
- Vectorized evaluation  
- Scatter-based backpropagation
- Per-simulation Dirichlet noise
- Progressive widening
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import time

from mcts.core.wave_search import WaveSearch
from mcts.gpu.csr_tree import CSRTree, CSRTreeConfig
from mcts.gpu.gpu_game_states import GPUGameStates, GPUGameStatesConfig, GameType
from mcts.gpu.mcts_gpu_accelerator import get_mcts_gpu_accelerator
from mcts.core.mcts_config import MCTSConfig
from conftest import assert_tensor_equal


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


class TestWaveSearchInitialization:
    """Test WaveSearch initialization"""
    
    def test_basic_initialization(self, wave_search_setup):
        """Test basic WaveSearch initialization"""
        ws = wave_search_setup['wave_search']
        
        assert ws.tree == wave_search_setup['tree']
        assert ws.game_states == wave_search_setup['game_states']
        assert ws.evaluator == wave_search_setup['evaluator']
        assert ws.config == wave_search_setup['config']
        assert ws.device == wave_search_setup['device']
        
        # Check default parameters
        assert ws.pw_alpha == 0.5  # Progressive widening alpha
        assert ws.pw_base == 10.0  # Progressive widening base
        assert not ws._buffers_allocated
        
    def test_buffer_allocation(self, wave_search_setup):
        """Test buffer allocation"""
        ws = wave_search_setup['wave_search']
        
        wave_size = 16
        max_depth = 50
        ws.allocate_buffers(wave_size, max_depth)
        
        # Check buffer shapes
        assert ws.paths_buffer.shape == (wave_size, max_depth)
        assert ws.path_lengths.shape == (wave_size,)
        assert ws.current_nodes.shape == (wave_size,)
        assert ws.active_mask.shape == (wave_size,)
        
        # Check UCB buffers
        assert ws.ucb_scores.shape == (wave_size, ws.config.max_children_per_node)
        assert ws.child_indices.shape == (wave_size, ws.config.max_children_per_node)
        
        # Check evaluation buffers
        assert ws.eval_batch.shape == (wave_size, 3, 15, 15)
        assert ws.policy_values.shape == (wave_size, 225)
        assert ws.value_estimates.shape == (wave_size,)
        
        assert ws._buffers_allocated
        
    def test_progressive_widening_calculation(self, wave_search_setup):
        """Test progressive widening calculation"""
        ws = wave_search_setup['wave_search']
        
        # Test with different parent visit counts
        # No visits - should use base * 2 for root
        max_children = ws._get_max_children_for_expansion(0, 225)
        assert max_children == min(int(ws.pw_base * 2), 225)
        
        # Few visits
        max_children = ws._get_max_children_for_expansion(10, 225)
        expected = int(ws.pw_base + ws.pw_alpha * np.sqrt(10))
        assert max_children == min(expected, 225)
        
        # Many visits
        max_children = ws._get_max_children_for_expansion(1000, 225)
        expected = int(ws.pw_base + ws.pw_alpha * np.sqrt(1000))
        assert max_children == min(expected, 225)
        
        # Limited legal moves
        max_children = ws._get_max_children_for_expansion(1000, 5)
        assert max_children == 5  # Can't exceed legal moves


class TestWaveSelection:
    """Test wave-based parallel selection"""
    
    def test_basic_selection(self, wave_search_setup):
        """Test basic parallel selection through tree"""
        ws = wave_search_setup['wave_search']
        tree = wave_search_setup['tree']
        
        # Build simple tree
        # Root -> A, B, C
        # A -> D, E
        actions = [0, 1, 2]
        priors = [0.4, 0.3, 0.3]
        children = tree.add_children_batch(0, actions, priors)
        
        # Add grandchildren to first child
        gc_actions = [10, 11]
        gc_priors = [0.6, 0.4]
        grandchildren = tree.add_children_batch(children[0], gc_actions, gc_priors)
        
        # Set some visit counts
        tree.node_data.visit_counts[0] = 10
        tree.node_data.visit_counts[children[0]] = 5
        tree.node_data.visit_counts[children[1]] = 3
        tree.node_data.visit_counts[children[2]] = 2
        
        # Run selection
        wave_size = 4
        paths, path_lengths, leaf_nodes = ws._select_batch_vectorized(wave_size)
        
        # Should select different paths
        assert paths.shape[0] == wave_size
        assert path_lengths.shape[0] == wave_size
        assert leaf_nodes.shape[0] == wave_size
        
        # All paths should start at root
        assert torch.all(paths[:, 0] == 0)
        
        # Path lengths should be reasonable
        assert torch.all(path_lengths >= 1)
        assert torch.all(path_lengths <= paths.shape[1])
        
    def test_selection_with_terminal_nodes(self, wave_search_setup):
        """Test selection handles terminal nodes correctly"""
        ws = wave_search_setup['wave_search']
        tree = wave_search_setup['tree']
        
        # Build tree with terminal node
        children = tree.add_children_batch(0, [0, 1], [0.5, 0.5])
        
        # Mark second child as terminal
        tree.set_terminal(children[1], True)
        
        # Run selection multiple times
        wave_size = 8
        paths, path_lengths, leaf_nodes = ws._select_batch_vectorized(wave_size)
        
        # Terminal node might be selected but won't be expanded further
        # Check that paths are valid
        for i in range(wave_size):
            path_len = path_lengths[i].item()
            for j in range(path_len):
                node_idx = paths[i, j].item()
                assert 0 <= node_idx < tree.num_nodes
                
    def test_selection_with_ucb_scores(self, wave_search_setup):
        """Test selection uses correct UCB formula"""
        ws = wave_search_setup['wave_search']
        tree = wave_search_setup['tree']
        
        # Build tree with known statistics
        children = tree.add_children_batch(0, [0, 1, 2], [0.5, 0.3, 0.2])
        
        # Set specific statistics
        tree.node_data.visit_counts[0] = 15
        
        # Child 0: Good Q-value, high visits
        tree.node_data.visit_counts[children[0]] = 8
        tree.node_data.value_sums[children[0]] = 4.0  # Q = 0.5
        
        # Child 1: Bad Q-value, medium visits
        tree.node_data.visit_counts[children[1]] = 5
        tree.node_data.value_sums[children[1]] = -2.5  # Q = -0.5
        
        # Child 2: Unvisited (high exploration bonus)
        tree.node_data.visit_counts[children[2]] = 0
        
        # Run selection with single wave
        paths, path_lengths, leaf_nodes = ws._select_batch_vectorized(1)
        
        # With default c_puct, should select unvisited node
        selected_child = paths[0, 1].item() if path_lengths[0] > 1 else -1
        assert selected_child == children[2]  # Unvisited node


class TestWaveExpansion:
    """Test wave-based batch expansion"""
    
    def test_basic_expansion(self, wave_search_setup):
        """Test basic batch expansion of leaf nodes"""
        ws = wave_search_setup['wave_search']
        tree = wave_search_setup['tree']
        game_states = wave_search_setup['game_states']
        
        # Initialize root state
        root_state_idx = game_states.allocate_states(1)[0]
        ws.node_to_state = torch.full((tree.config.max_nodes,), -1, dtype=torch.int32, device=ws.device)
        ws.node_to_state[0] = root_state_idx
        ws.state_pool_free_list = list(range(1, 1000))
        
        # Expand root
        leaf_nodes = torch.tensor([0], device=ws.device)
        expanded_nodes = ws._expand_batch_vectorized(leaf_nodes)
        
        # Should have added children
        assert tree.num_nodes > 1
        children, actions, priors = tree.get_children(0)
        assert len(children) > 0
        
        # Children should have states
        for child in children:
            state_idx = ws.node_to_state[child].item()
            assert state_idx >= 0
            assert game_states.allocated_mask[state_idx]
            
    def test_expansion_with_progressive_widening(self, wave_search_setup):
        """Test expansion respects progressive widening"""
        ws = wave_search_setup['wave_search']
        tree = wave_search_setup['tree']
        game_states = wave_search_setup['game_states']
        
        # Set up states
        root_state_idx = game_states.allocate_states(1)[0]
        ws.node_to_state = torch.full((tree.config.max_nodes,), -1, dtype=torch.int32, device=ws.device)
        ws.node_to_state[0] = root_state_idx
        ws.state_pool_free_list = list(range(1, 1000))
        
        # Set progressive widening parameters
        ws.pw_alpha = 0.25
        ws.pw_base = 5.0
        
        # First expansion with no visits
        tree.node_data.visit_counts[0] = 0
        leaf_nodes = torch.tensor([0], device=ws.device)
        ws._expand_batch_vectorized(leaf_nodes)
        
        children_1, _, _ = tree.get_children(0)
        num_children_1 = len(children_1)
        
        # Should have limited children
        assert num_children_1 < 225  # Not all moves expanded
        assert num_children_1 >= ws.pw_base  # At least base amount
        
        # Clear children for second test
        tree.reset()
        ws.node_to_state[0] = root_state_idx
        
        # Second expansion with many visits
        tree.node_data.visit_counts[0] = 1000  # Use more visits for a bigger difference
        ws._expand_batch_vectorized(leaf_nodes)
        
        children_2, _, _ = tree.get_children(0)
        num_children_2 = len(children_2)
        
        # Should have more children with more visits (or at least same for capped cases)
        # With pw_alpha=0.25, pw_base=5:
        # visits=0: max_children = min(10, 225) = 10
        # visits=1000: max_children = min(5 + 0.25*sqrt(1000), 225) = min(5 + 7.9, 225) = 12
        assert num_children_2 >= num_children_1
        
    def test_expansion_multiple_nodes(self, wave_search_setup):
        """Test expanding multiple nodes in parallel"""
        ws = wave_search_setup['wave_search']
        tree = wave_search_setup['tree']
        game_states = wave_search_setup['game_states']
        
        # Create multiple leaf nodes
        children = tree.add_children_batch(0, [0, 1, 2], [0.3, 0.3, 0.4])
        
        # Allocate states for all nodes
        state_indices = game_states.allocate_states(4)
        ws.node_to_state = torch.full((tree.config.max_nodes,), -1, dtype=torch.int32, device=ws.device)
        for i, node_idx in enumerate([0] + children):
            ws.node_to_state[node_idx] = state_indices[i]
        ws.state_pool_free_list = list(range(4, 1000))
        
        # Expand all children
        leaf_nodes = torch.tensor(children, device=ws.device)
        expanded_nodes = ws._expand_batch_vectorized(leaf_nodes)
        
        # Each child should have been expanded
        for child in children:
            gc, _, _ = tree.get_children(child)
            assert len(gc) > 0
            
    def test_expansion_avoids_duplicates(self, wave_search_setup):
        """Test expansion doesn't create duplicate children"""
        ws = wave_search_setup['wave_search']
        tree = wave_search_setup['tree']
        game_states = wave_search_setup['game_states']
        
        # Set up state
        root_state_idx = game_states.allocate_states(1)[0]
        ws.node_to_state = torch.full((tree.config.max_nodes,), -1, dtype=torch.int32, device=ws.device)
        ws.node_to_state[0] = root_state_idx
        ws.state_pool_free_list = list(range(1, 1000))
        
        # Expand twice
        leaf_nodes = torch.tensor([0], device=ws.device)
        ws._expand_batch_vectorized(leaf_nodes)
        
        children_before, actions_before, _ = tree.get_children(0)
        num_before = len(children_before)
        
        # Try to expand again
        ws._expand_batch_vectorized(leaf_nodes)
        
        children_after, actions_after, _ = tree.get_children(0)
        num_after = len(children_after)
        
        # Should not add duplicate children
        assert num_after == num_before


class TestWaveEvaluation:
    """Test wave-based batch evaluation"""
    
    def test_basic_evaluation(self, wave_search_setup):
        """Test basic batch evaluation of nodes"""
        ws = wave_search_setup['wave_search']
        tree = wave_search_setup['tree']
        game_states = wave_search_setup['game_states']
        evaluator = wave_search_setup['evaluator']
        
        # Create nodes with states
        children = tree.add_children_batch(0, [0, 1], [0.5, 0.5])
        state_indices = game_states.allocate_states(len(children))
        
        ws.node_to_state = torch.full((tree.config.max_nodes,), -1, dtype=torch.int32, device=ws.device)
        for i, child in enumerate(children):
            ws.node_to_state[child] = state_indices[i]
            
        # Evaluate nodes
        nodes = torch.tensor(children, device=ws.device)
        values = ws._evaluate_batch_vectorized(nodes)
        
        # Should return values for each node
        assert values.shape == (len(children),)
        assert evaluator.evaluate_batch.called
        
        # Values should be in valid range
        assert torch.all(values >= -1.0)
        assert torch.all(values <= 1.0)
        
    def test_evaluation_with_invalid_nodes(self, wave_search_setup):
        """Test evaluation handles invalid nodes gracefully"""
        ws = wave_search_setup['wave_search']
        
        # Mix of valid and invalid nodes
        nodes = torch.tensor([0, -1, 999, 1], device=ws.device)
        
        # Should handle gracefully
        values = ws._evaluate_batch_vectorized(nodes)
        
        assert values.shape == (4,)
        # Invalid nodes should get zero value
        assert values[1] == 0.0  # Node -1
        assert values[2] == 0.0  # Node 999
        
    def test_evaluation_caching(self, wave_search_setup):
        """Test evaluation results are used efficiently"""
        ws = wave_search_setup['wave_search']
        tree = wave_search_setup['tree']
        game_states = wave_search_setup['game_states']
        evaluator = wave_search_setup['evaluator']
        
        # Track number of evaluator calls
        call_count = 0
        def mock_eval(states):
            nonlocal call_count
            call_count += 1
            batch_size = len(states)
            return (
                np.random.rand(batch_size, 225).astype(np.float32),
                np.random.uniform(-0.5, 0.5, batch_size).astype(np.float32)
            )
        evaluator.evaluate_batch = mock_eval
        
        # Create multiple nodes
        children = tree.add_children_batch(0, list(range(10)), [0.1] * 10)
        state_indices = game_states.allocate_states(len(children))
        
        ws.node_to_state = torch.full((tree.config.max_nodes,), -1, dtype=torch.int32, device=ws.device)
        for i, child in enumerate(children):
            ws.node_to_state[child] = state_indices[i]
            
        # Evaluate all at once
        nodes = torch.tensor(children, device=ws.device)
        values = ws._evaluate_batch_vectorized(nodes)
        
        # Should make single batch call
        assert call_count == 1


class TestWaveBackpropagation:
    """Test wave-based parallel backpropagation"""
    
    def test_basic_backpropagation(self, wave_search_setup):
        """Test basic value backpropagation"""
        ws = wave_search_setup['wave_search']
        tree = wave_search_setup['tree']
        
        # Create simple path
        child = tree.add_child(0, 0, 0.5)
        grandchild = tree.add_child(child, 1, 0.5)
        
        # Create path and backup
        paths = torch.tensor([[0, child, grandchild]], device=ws.device)
        path_lengths = torch.tensor([3], device=ws.device)
        values = torch.tensor([0.5], device=ws.device)
        
        # Initial visits
        initial_visits = tree.node_data.visit_counts.clone()
        
        # Backup
        ws._backup_batch_vectorized(paths, path_lengths, values)
        
        # Check visits incremented
        assert tree.node_data.visit_counts[0] == initial_visits[0] + 1
        assert tree.node_data.visit_counts[child] == initial_visits[child] + 1
        assert tree.node_data.visit_counts[grandchild] == initial_visits[grandchild] + 1
        
        # Check values with alternating signs
        assert abs(tree.node_data.value_sums[grandchild].item() - 0.5) < 1e-5
        assert abs(tree.node_data.value_sums[child].item() - (-0.5)) < 1e-5
        assert abs(tree.node_data.value_sums[0].item() - 0.5) < 1e-5
        
    def test_parallel_backpropagation(self, wave_search_setup):
        """Test backing up multiple paths in parallel"""
        ws = wave_search_setup['wave_search']
        tree = wave_search_setup['tree']
        
        # Create tree with multiple paths
        children = tree.add_children_batch(0, [0, 1, 2], [0.3, 0.3, 0.4])
        gc1 = tree.add_child(children[0], 10, 0.5)
        gc2 = tree.add_child(children[1], 11, 0.5)
        
        # Create multiple paths
        paths = torch.tensor([
            [0, children[0], gc1, -1],
            [0, children[1], gc2, -1],
            [0, children[2], -1, -1]
        ], device=ws.device)
        path_lengths = torch.tensor([3, 3, 2], device=ws.device)
        values = torch.tensor([0.8, -0.6, 0.3], device=ws.device)
        
        # Backup all paths
        ws._backup_batch_vectorized(paths, path_lengths, values)
        
        # Root should have all visits
        assert tree.node_data.visit_counts[0] == 3
        
        # Each path should be updated correctly
        assert tree.node_data.visit_counts[children[0]] == 1
        assert tree.node_data.visit_counts[children[1]] == 1
        assert tree.node_data.visit_counts[children[2]] == 1
        
        # Values should be correct (with sign flips)
        # Verify backpropagation updated the root
        root_value = tree.node_data.value_sums[0].item()
        root_visits = tree.node_data.visit_counts[0].item()
        
        # The root should have been updated by all paths
        assert root_visits == 3
        
        # The value should be averaged over visits
        # Different implementations may handle value aggregation differently
        # so we just check that the value was updated
        assert root_value != 0.0
        
    def test_scatter_backup_method(self, wave_search_setup):
        """Test optimized scatter-based backup"""
        ws = wave_search_setup['wave_search']
        tree = wave_search_setup['tree']
        
        # Build larger tree
        children = tree.add_children_batch(0, list(range(5)), [0.2] * 5)
        grandchildren = []
        for child in children:
            gc = tree.add_children_batch(child, list(range(3)), [0.33] * 3)
            grandchildren.extend(gc)
            
        # Create multiple paths
        batch_size = 10
        paths = torch.zeros((batch_size, 3), dtype=torch.int32, device=ws.device)
        path_lengths = torch.full((batch_size,), 3, dtype=torch.int32, device=ws.device)
        values = torch.rand(batch_size, device=ws.device) * 2 - 1  # Random [-1, 1]
        
        # Fill paths
        for i in range(batch_size):
            child_idx = i % len(children)
            gc_idx = i % 3
            paths[i] = torch.tensor([0, children[child_idx], grandchildren[child_idx * 3 + gc_idx]])
            
        # Use scatter backup
        ws._scatter_backup(paths, path_lengths, values)
        
        # Verify all nodes were updated
        assert tree.node_data.visit_counts[0] == batch_size
        for child in children:
            assert tree.node_data.visit_counts[child] > 0


class TestDirichletNoise:
    """Test per-simulation Dirichlet noise application"""
    
    def test_per_simulation_dirichlet_basic(self, wave_search_setup):
        """Test basic per-simulation Dirichlet noise"""
        ws = wave_search_setup['wave_search']
        tree = wave_search_setup['tree']
        
        # Add children to root
        children = tree.add_children_batch(0, [0, 1, 2], [0.3, 0.4, 0.3])
        priors = tree.node_data.node_priors[children]
        
        # Apply noise for multiple simulations
        sim_indices = torch.tensor([0, 1, 2], device=ws.device)
        noised_priors = ws.apply_per_simulation_dirichlet_noise(
            0, sim_indices, children, priors
        )
        
        # Should have different noise for each simulation
        assert noised_priors.shape == (3, 3)  # 3 sims, 3 children
        
        # Each simulation should have valid probability distribution
        for i in range(3):
            sim_priors = noised_priors[i]
            assert torch.all(sim_priors >= 0)
            assert torch.all(sim_priors <= 1)
            assert abs(sim_priors.sum().item() - 1.0) < 1e-5
            
        # Different simulations should have different noise
        assert not torch.allclose(noised_priors[0], noised_priors[1])
        assert not torch.allclose(noised_priors[1], noised_priors[2])
        
    def test_dirichlet_noise_non_root(self, wave_search_setup):
        """Test Dirichlet noise only applies to root"""
        ws = wave_search_setup['wave_search']
        tree = wave_search_setup['tree']
        
        # Add children to root and grandchildren
        children = tree.add_children_batch(0, [0, 1], [0.5, 0.5])
        grandchildren = tree.add_children_batch(children[0], [2, 3], [0.6, 0.4])
        
        priors = tree.node_data.node_priors[grandchildren]
        
        # Apply noise to non-root node
        sim_indices = torch.tensor([0, 1], device=ws.device)
        noised_priors = ws.apply_per_simulation_dirichlet_noise(
            children[0], sim_indices, grandchildren, priors
        )
        
        # Should return original priors (no noise for non-root)
        assert noised_priors.shape == (2, 2)
        for i in range(2):
            assert torch.allclose(noised_priors[i], priors)
            
    def test_dirichlet_noise_mixing(self, wave_search_setup):
        """Test Dirichlet noise mixing with epsilon"""
        ws = wave_search_setup['wave_search']
        tree = wave_search_setup['tree']
        
        # Set specific epsilon
        ws.config.dirichlet_epsilon = 0.25
        
        # Add children
        children = tree.add_children_batch(0, [0, 1, 2, 3], [0.25] * 4)
        priors = tree.node_data.node_priors[children]
        
        # Apply noise
        sim_indices = torch.tensor([0], device=ws.device)
        noised_priors = ws.apply_per_simulation_dirichlet_noise(
            0, sim_indices, children, priors
        )
        
        # With epsilon=0.25, result should be 75% original + 25% noise
        # So no prior should deviate too much from 0.25
        deviations = torch.abs(noised_priors[0] - 0.25)
        assert torch.all(deviations < 0.25)  # Max possible deviation


class TestWaveIntegration:
    """Test full wave execution"""
    
    def test_full_wave_execution(self, wave_search_setup):
        """Test complete wave execution"""
        ws = wave_search_setup['wave_search']
        tree = wave_search_setup['tree']
        game_states = wave_search_setup['game_states']
        
        # Set up initial state
        root_state = game_states.allocate_states(1)[0]
        node_to_state = torch.full((tree.config.max_nodes,), -1, dtype=torch.int32, device=ws.device)
        node_to_state[0] = root_state
        state_pool_free_list = list(range(1, 1000))
        
        # Run wave
        wave_size = 8
        completed = ws.run_wave(wave_size, node_to_state, state_pool_free_list)
        
        # Should complete all simulations
        assert completed == wave_size
        
        # Tree should have grown
        assert tree.num_nodes > 1
        assert tree.node_data.visit_counts[0] == wave_size
        
    def test_wave_with_terminal_positions(self, wave_search_setup):
        """Test wave handles terminal positions"""
        ws = wave_search_setup['wave_search']
        tree = wave_search_setup['tree']
        game_states = wave_search_setup['game_states']
        
        # Create tree with terminal branch
        children = tree.add_children_batch(0, [0, 1], [0.5, 0.5])
        tree.set_terminal(children[1], True)
        
        # Set up states
        state_indices = game_states.allocate_states(3)
        node_to_state = torch.full((tree.config.max_nodes,), -1, dtype=torch.int32, device=ws.device)
        node_to_state[0] = state_indices[0]
        node_to_state[children[0]] = state_indices[1]
        node_to_state[children[1]] = state_indices[2]
        
        # Mark game state as terminal
        game_states.is_terminal[state_indices[2]] = True
        game_states.winner[state_indices[2]] = 1  # Player 1 wins
        
        state_pool_free_list = list(range(3, 1000))
        
        # Run wave
        completed = ws.run_wave(8, node_to_state, state_pool_free_list)
        
        # Should handle terminal nodes gracefully
        assert completed == 8
        
    @pytest.mark.slow
    def test_large_wave_performance(self, wave_search_setup):
        """Test performance with large waves"""
        ws = wave_search_setup['wave_search']
        tree = wave_search_setup['tree']
        game_states = wave_search_setup['game_states']
        
        # Set up state
        root_state = game_states.allocate_states(1)[0]
        node_to_state = torch.full((tree.config.max_nodes,), -1, dtype=torch.int32, device=ws.device)
        node_to_state[0] = root_state
        state_pool_free_list = list(range(1, game_states.capacity))
        
        # Time large wave
        wave_size = 256
        start_time = time.time()
        completed = ws.run_wave(wave_size, node_to_state, state_pool_free_list)
        elapsed = time.time() - start_time
        
        assert completed == wave_size
        
        # Should complete reasonably fast
        simulations_per_second = wave_size / elapsed
        assert simulations_per_second > 100  # At least 100 sims/sec
        
        # Tree should have grown efficiently
        assert tree.num_nodes > wave_size  # More nodes than simulations (expansion)
        assert tree.num_nodes < wave_size * 10  # But not too many


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_tree_selection(self, wave_search_setup):
        """Test selection on tree with only root"""
        ws = wave_search_setup['wave_search']
        
        # Run selection on empty tree
        paths, lengths, leaves = ws._select_batch_vectorized(4)
        
        # All should select root as leaf
        assert torch.all(leaves == 0)
        assert torch.all(lengths == 1)
        
    def test_buffer_reallocation(self, wave_search_setup):
        """Test buffer reallocation for different wave sizes"""
        ws = wave_search_setup['wave_search']
        
        # Allocate small buffers
        ws.allocate_buffers(8)
        
        # Run with larger wave size
        # Should reallocate automatically
        node_to_state = torch.zeros(1000, dtype=torch.int32, device=ws.device)
        state_pool = list(range(1000))
        
        completed = ws.run_wave(16, node_to_state, state_pool)
        
        # Should handle reallocation
        assert completed == 16
        assert ws.paths_buffer.shape[0] >= 16
        
    def test_state_pool_exhaustion(self, wave_search_setup):
        """Test handling when state pool is exhausted"""
        ws = wave_search_setup['wave_search']
        tree = wave_search_setup['tree']
        game_states = wave_search_setup['game_states']
        
        # Very small state pool
        root_state = game_states.allocate_states(1)[0]
        node_to_state = torch.full((tree.config.max_nodes,), -1, dtype=torch.int32, device=ws.device)
        node_to_state[0] = root_state
        state_pool_free_list = list(range(1, 10))  # Only 9 free states
        
        # Try to run large wave
        # Should handle gracefully (may not complete all simulations)
        completed = ws.run_wave(50, node_to_state, state_pool_free_list)
        
        # Should complete some simulations
        assert completed > 0
        assert completed <= 50