"""Simple tests for MCTS to improve coverage"""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from mcts.mcts import MCTS, MCTSConfig
from mcts.node import Node
from mcts.tree_arena import MemoryConfig


class TestMCTSCoverage:
    """Tests for MCTS coverage improvement"""
    
    def test_mcts_init_without_wave_engine(self):
        """Test MCTS initialization without wave engine"""
        game = Mock()
        evaluator = Mock()
        config = MCTSConfig(use_wave_engine=False)
        
        mcts = MCTS(game, evaluator, config)
        
        assert mcts.wave_engine is None
        assert mcts.game == game
        assert mcts.evaluator == evaluator
        
    def test_get_or_create_root_reuse(self):
        """Test root reuse logic"""
        game = Mock()
        evaluator = Mock()
        config = MCTSConfig(reuse_tree=True)
        
        mcts = MCTS(game, evaluator, config)
        
        # Mock current root
        old_root = Mock(spec=Node)
        old_root.state = "old_state"
        old_root.children = {0: Mock(state="new_state")}
        mcts.current_root = old_root
        
        # Test reuse
        root = mcts._get_or_create_root("new_state", parent_action=0)
        assert root == old_root.children[0]
        
    def test_get_or_create_root_new(self):
        """Test creating new root"""
        game = Mock()
        evaluator = Mock()
        config = MCTSConfig()
        
        mcts = MCTS(game, evaluator, config)
        
        # Test new root
        root = mcts._get_or_create_root("new_state", None)
        assert isinstance(root, Node)
        assert root.state == "new_state"
        
    def test_add_dirichlet_noise(self):
        """Test Dirichlet noise addition"""
        game = Mock()
        evaluator = Mock()
        config = MCTSConfig(noise_epsilon=0.25, noise_alpha=0.3)
        
        mcts = MCTS(game, evaluator, config)
        
        # Create root with children
        root = Node("state", prior=0.5)
        root.children = {
            0: Node("child0", prior=0.3),
            1: Node("child1", prior=0.7)
        }
        
        # Add noise
        mcts._add_dirichlet_noise(root)
        
        # Check that priors changed
        assert root.children[0].prior != 0.3
        assert root.children[1].prior != 0.7
        
    def test_run_simulations_no_wave_engine(self):
        """Test running simulations without wave engine"""
        game = Mock()
        evaluator = Mock()
        config = MCTSConfig(use_wave_engine=False, num_simulations=5)
        
        mcts = MCTS(game, evaluator, config)
        
        # Mock methods
        root = Mock()
        root_id = "root_id"
        mcts._run_single_simulation = Mock()
        
        # Run simulations
        mcts._run_simulations(root, root_id)
        
        # Check calls
        assert mcts._run_single_simulation.call_count == 5
        
    def test_select_leaf(self):
        """Test leaf selection"""
        game = Mock()
        evaluator = Mock()
        config = MCTSConfig(c_puct=1.0)
        
        mcts = MCTS(game, evaluator, config)
        
        # Create simple tree
        root = Node("root", prior=1.0)
        root.visit_count = 10
        child1 = Node("child1", prior=0.6)
        child1.visit_count = 5
        child1.total_value = 2.5
        child2 = Node("child2", prior=0.4)
        child2.visit_count = 3
        child2.total_value = 1.8
        
        root.children = {0: child1, 1: child2}
        root.is_expanded = True
        
        # Test selection
        path, actions = mcts._select_leaf(root)
        
        assert len(path) >= 1
        assert len(actions) == len(path) - 1
        
    def test_expand_node(self):
        """Test node expansion"""
        game = Mock()
        evaluator = Mock()
        config = MCTSConfig()
        
        # Setup mocks
        game.get_legal_moves = Mock(return_value=[0, 1, 2])
        game.apply_move = Mock(side_effect=lambda s, a: f"{s}_move{a}")
        evaluator.evaluate = Mock(return_value=({"value": 0.5, "policy": {0: 0.3, 1: 0.5, 2: 0.2}}, None))
        
        mcts = MCTS(game, evaluator, config)
        
        # Test expansion
        node = Node("state")
        value = mcts._expand_node(node)
        
        assert node.is_expanded
        assert len(node.children) == 3
        assert value == 0.5
        
    def test_backup(self):
        """Test backup propagation"""
        game = Mock()
        evaluator = Mock()
        config = MCTSConfig()
        
        mcts = MCTS(game, evaluator, config)
        
        # Create path
        root = Node("root")
        child = Node("child")
        grandchild = Node("grandchild")
        
        path = [root, child, grandchild]
        
        # Backup
        mcts._backup(path, 0.7)
        
        # Check updates
        assert root.visit_count == 1
        assert root.total_value == -0.7  # Negated
        assert child.visit_count == 1
        assert child.total_value == 0.7
        assert grandchild.visit_count == 1
        assert grandchild.total_value == -0.7
        
    def test_get_action_probabilities(self):
        """Test action probability calculation"""
        game = Mock()
        evaluator = Mock()
        config = MCTSConfig(temperature=1.0)
        
        mcts = MCTS(game, evaluator, config)
        
        # Create root with children
        root = Node("root")
        root.children = {
            0: Mock(visit_count=10),
            1: Mock(visit_count=20),
            2: Mock(visit_count=5)
        }
        
        # Get probabilities
        probs = mcts.get_action_probabilities(root)
        
        assert len(probs) == 3
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        
    def test_get_best_action(self):
        """Test best action selection"""
        game = Mock()
        evaluator = Mock()
        config = MCTSConfig()
        
        mcts = MCTS(game, evaluator, config)
        
        # Create root with children
        root = Node("root")
        root.children = {
            0: Mock(visit_count=10),
            1: Mock(visit_count=20),
            2: Mock(visit_count=5)
        }
        
        # Get best action
        action = mcts.get_best_action(root)
        assert action == 1  # Highest visit count
        
    def test_get_pv_line(self):
        """Test principal variation extraction"""
        game = Mock()
        evaluator = Mock()
        config = MCTSConfig()
        
        mcts = MCTS(game, evaluator, config)
        
        # Create simple tree
        root = Node("root")
        child1 = Mock(visit_count=20, children={})
        child2 = Mock(visit_count=10, children={})
        root.children = {0: child1, 1: child2}
        
        # Get PV
        pv = mcts.get_pv_line(root, max_depth=5)
        assert pv == [0]  # Best child
        
    def test_get_search_statistics(self):
        """Test statistics collection"""
        game = Mock()
        evaluator = Mock()
        config = MCTSConfig()
        
        mcts = MCTS(game, evaluator, config)
        mcts.stats = {
            'search_time': 1.5,
            'searches_performed': 100
        }
        
        # Mock arena stats
        mcts.arena = Mock()
        mcts.arena.get_statistics = Mock(return_value={'nodes': 1000})
        
        # Get stats
        stats = mcts.get_search_statistics(Mock(visit_count=500))
        
        assert stats['simulations'] == 500
        assert stats['time'] == 1.5
        assert 'arena' in stats