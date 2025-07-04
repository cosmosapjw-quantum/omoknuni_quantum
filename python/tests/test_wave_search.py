"""Tests for wave-based search module"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock

import sys
sys.path.append('/home/cosmosapjw/omoknuni_quantum/python')

from mcts.core.wave_search import WaveSearch
from mcts.gpu.csr_tree import CSRTree, CSRTreeConfig
from mcts.gpu.gpu_game_states import GPUGameStates, GPUGameStatesConfig, GameType
from mcts.core.mcts_config import MCTSConfig


class TestWaveSearch:
    """Test wave search functionality"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return MCTSConfig(
            device='cpu',
            game_type=GameType.GOMOKU,
            board_size=15,
            max_wave_size=32,
            max_children_per_node=225
        )
        
    @pytest.fixture
    def tree(self, config):
        """Create test CSR tree"""
        tree_config = CSRTreeConfig(
            max_nodes=1000,
            max_edges=10000,
            max_actions=225,
            device=config.device
        )
        tree = CSRTree(tree_config)
        tree.reset()
        return tree
        
    @pytest.fixture
    def game_states(self, config):
        """Create test GPU game states"""
        game_config = GPUGameStatesConfig(
            capacity=1000,
            game_type=config.game_type,
            board_size=config.board_size,
            device=config.device
        )
        return GPUGameStates(game_config)
        
    @pytest.fixture
    def evaluator(self):
        """Create mock evaluator"""
        evaluator = Mock()
        evaluator.evaluate = MagicMock(return_value=(
            torch.ones((32, 225)) / 225,  # Uniform policy
            torch.zeros(32)  # Zero values
        ))
        return evaluator
        
    @pytest.fixture
    def wave_search(self, tree, game_states, evaluator, config):
        """Create wave search instance"""
        return WaveSearch(
            tree=tree,
            game_states=game_states,
            evaluator=evaluator,
            config=config,
            device=torch.device(config.device)
        )
        
    def test_initialization(self, wave_search):
        """Test wave search initialization"""
        assert wave_search is not None
        assert wave_search.tree is not None
        assert wave_search.game_states is not None
        assert wave_search.evaluator is not None
        assert not wave_search._buffers_allocated
        
    def test_buffer_allocation(self, wave_search):
        """Test buffer allocation"""
        wave_size = 32
        wave_search.allocate_buffers(wave_size)
        
        assert wave_search._buffers_allocated
        assert wave_search.paths_buffer.shape == (wave_size, 100)
        assert wave_search.path_lengths.shape == (wave_size,)
        assert wave_search.current_nodes.shape == (wave_size,)
        assert wave_search.ucb_scores.shape == (wave_size, 225)
        
    def test_select_batch_vectorized(self, wave_search):
        """Test batch selection"""
        wave_size = 16
        wave_search.allocate_buffers(wave_size)
        
        # Create node_to_state mapping
        node_to_state = torch.full((1000,), -1, dtype=torch.int32)
        node_to_state[0] = 0  # Root node has state 0
        
        wave_search.node_to_state = node_to_state
        
        paths, path_lengths, leaf_nodes = wave_search._select_batch_vectorized(wave_size)
        
        assert paths.shape == (wave_size, 100)
        assert path_lengths.shape == (wave_size,)
        assert leaf_nodes.shape == (wave_size,)
        
        # All paths should start from root (0)
        assert (paths[:, 0] == 0).all()
        
    def test_evaluate_batch_vectorized(self, wave_search):
        """Test batch evaluation"""
        wave_size = 8
        wave_search.allocate_buffers(wave_size)
        
        # Create test nodes
        nodes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32)
        
        # Create node to state mapping
        node_to_state = torch.arange(1000, dtype=torch.int32)
        wave_search.node_to_state = node_to_state
        
        # Mock game states get_features_batch
        wave_search.game_states.get_features_batch = MagicMock(
            return_value=torch.randn(8, 3, 15, 15)
        )
        
        values = wave_search._evaluate_batch_vectorized(nodes)
        
        assert values.shape == (wave_size,)
        assert wave_search.evaluator.evaluate.called
        
    def test_backup_batch_vectorized(self, wave_search):
        """Test batch backup"""
        wave_size = 4
        wave_search.allocate_buffers(wave_size)
        
        # Create test paths
        paths = torch.tensor([
            [0, 1, 2, 0, 0],  # Path of length 3
            [0, 3, 0, 0, 0],  # Path of length 2
            [0, 0, 0, 0, 0],  # Path of length 1
            [0, 1, 4, 5, 0],  # Path of length 4
        ], dtype=torch.int32)
        
        path_lengths = torch.tensor([3, 2, 1, 4], dtype=torch.int32)
        values = torch.tensor([0.5, -0.3, 0.1, 0.8])
        
        # Record initial visit counts
        initial_visits = wave_search.tree.visit_counts.clone()
        
        wave_search._backup_batch_vectorized(paths, path_lengths, values)
        
        # Check that visit counts increased
        assert wave_search.tree.visit_counts[0] > initial_visits[0]
        
    def test_run_wave(self, wave_search):
        """Test complete wave execution"""
        wave_size = 16
        
        # Create node to state mapping and free list
        node_to_state = torch.full((1000,), -1, dtype=torch.int32)
        node_to_state[0] = 0  # Root has state
        state_pool_free_list = list(range(1, 100))
        
        # Mock game states methods
        wave_search.game_states.get_legal_actions_batch = MagicMock(
            return_value=[[i for i in range(225)]]  # All actions legal
        )
        wave_search.game_states.get_features_batch = MagicMock(
            return_value=torch.randn(wave_size, 3, 15, 15)
        )
        
        completed = wave_search.run_wave(wave_size, node_to_state, state_pool_free_list)
        
        assert completed == wave_size
        assert wave_search._buffers_allocated
        
        
class TestWaveSearchIntegration:
    """Integration tests for wave search with MCTS"""
    
    def test_wave_search_with_mcts_config(self):
        """Test wave search works with full MCTS config"""
        config = MCTSConfig(
            num_simulations=100,
            device='cpu',
            wave_size=32,
            max_wave_size=32,
            classical_only_mode=True
        )
        
        # Create tree
        tree_config = CSRTreeConfig(
            max_nodes=1000,
            max_edges=10000,
            max_actions=225,
            device=config.device
        )
        tree = CSRTree(tree_config)
        tree.reset()
        
        # Create game states
        game_config = GPUGameStatesConfig(
            capacity=1000,
            game_type=config.game_type,
            board_size=config.board_size,
            device=config.device
        )
        game_states = GPUGameStates(game_config)
        
        # Create evaluator
        evaluator = Mock()
        evaluator.evaluate = MagicMock(return_value=(
            torch.ones((32, 225)) / 225,
            torch.zeros(32)
        ))
        
        # Create wave search
        wave_search = WaveSearch(
            tree=tree,
            game_states=game_states,
            evaluator=evaluator,
            config=config,
            device=torch.device(config.device)
        )
        
        # Should initialize successfully
        assert wave_search is not None
        assert wave_search.config.max_wave_size == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])