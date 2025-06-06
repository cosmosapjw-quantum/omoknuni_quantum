"""Extended tests for WaveEngine to achieve 90%+ coverage"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time
from collections import defaultdict

from mcts.wave_engine import WaveEngine, WaveConfig, Wave
from mcts.node import Node
from mcts.game_interface import GameInterface, GameType
from mcts.evaluator import MockEvaluator, EvaluatorConfig
from mcts.tree_arena import TreeArena, MemoryConfig


class TestWaveConfig:
    """Test WaveConfig dataclass thoroughly"""
    
    def test_wave_config_all_params(self):
        """Test all WaveConfig parameters"""
        config = WaveConfig(
            min_wave_size=128,
            max_wave_size=4096,
            initial_wave_size=1024,
            c_puct=2.0,
            enable_interference=False,
            interference_threshold=0.3,
            enable_adaptive_sizing=False,
            max_concurrent_waves=8,
            enable_phase_kicks=True
        )
        
        assert config.min_wave_size == 128
        assert config.max_wave_size == 4096
        assert config.initial_wave_size == 1024
        assert config.c_puct == 2.0
        assert not config.enable_interference
        assert config.interference_threshold == 0.3
        assert not config.enable_adaptive_sizing
        assert config.max_concurrent_waves == 8
        assert config.enable_phase_kicks
        
    def test_wave_config_validation(self):
        """Test config with invalid values"""
        # Should handle negative values gracefully
        config = WaveConfig(min_wave_size=-1, max_wave_size=0)
        # The dataclass doesn't validate, but the engine should handle it
        assert config.min_wave_size == -1  # Invalid but allowed by dataclass


class TestWaveObject:
    """Test Wave dataclass in detail"""
    
    def test_wave_post_init(self):
        """Test Wave __post_init__ method"""
        wave = Wave(size=100)
        
        assert len(wave.paths) == 100
        assert len(wave.leaf_nodes) == 100
        assert all(p is None for p in wave.paths)
        assert all(n is None for n in wave.leaf_nodes)
        assert wave.active_paths == 100
        assert wave.phase == 0
        assert not wave.completed
        
    def test_wave_with_initial_data(self):
        """Test Wave with pre-initialized data"""
        paths = ["node_1", "node_2", "node_3"]
        nodes = [Mock(spec=Node) for _ in range(3)]
        
        wave = Wave(
            size=3,
            paths=paths,
            leaf_nodes=nodes,
            phase=2,
            root_id="root",
            completed=False
        )
        
        assert wave.paths == paths
        assert wave.leaf_nodes == nodes
        assert wave.phase == 2
        assert wave.active_paths == 3


class TestWaveEngineInitialization:
    """Test WaveEngine initialization scenarios"""
    
    def test_engine_without_interference(self):
        """Test engine when interference is disabled"""
        game = GameInterface(GameType.GOMOKU, board_size=9)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=81)
        arena = TreeArena(MemoryConfig.laptop_preset())
        config = WaveConfig(enable_interference=False)
        
        engine = WaveEngine(game, evaluator, arena, config)
        
        assert engine.interference_engine is None
        assert engine.current_wave_size == config.initial_wave_size
        assert len(engine.wave_history) == 0
        assert engine.stats['total_waves'] == 0
        
    @patch('mcts.wave_engine.HAS_INTERFERENCE', False)
    def test_engine_without_interference_module(self):
        """Test engine when interference module is not available"""
        game = GameInterface(GameType.GO, board_size=9)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=82)
        arena = TreeArena(MemoryConfig.laptop_preset())
        config = WaveConfig(enable_interference=True)  # Enabled but module missing
        
        engine = WaveEngine(game, evaluator, arena, config)
        
        assert engine.interference_engine is None
        
    def test_engine_with_mock_interference(self):
        """Test engine with mocked interference engine"""
        with patch('mcts.wave_engine.HAS_INTERFERENCE', True):
            mock_interference = MagicMock()
            with patch('mcts.wave_engine.InterferenceEngine', return_value=mock_interference):
                game = GameInterface(GameType.CHESS)
                evaluator = MockEvaluator(EvaluatorConfig(), action_size=4096)
                arena = TreeArena(MemoryConfig.desktop_preset())
                config = WaveConfig(
                    enable_interference=True,
                    interference_threshold=0.7
                )
                
                engine = WaveEngine(game, evaluator, arena, config)
                
                assert engine.interference_engine == mock_interference


class TestWaveCreation:
    """Test wave creation scenarios"""
    
    def test_create_wave_with_custom_size(self):
        """Test creating wave with specific size"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=225)
        arena = TreeArena(MemoryConfig.laptop_preset())
        engine = WaveEngine(game, evaluator, arena, WaveConfig())
        
        # Create root
        root = Node(state=game.create_initial_state(), parent=None, action=None, prior=1.0)
        root_id = arena.add_node(root)
        
        # Create waves of different sizes
        small_wave = engine.create_wave(root_id, size=64)
        large_wave = engine.create_wave(root_id, size=2048)
        
        assert small_wave.size == 64
        assert large_wave.size == 2048
        assert small_wave.root_id == root_id
        assert large_wave.root_id == root_id
        
    def test_create_wave_adaptive_size(self):
        """Test wave creation with adaptive sizing"""
        game = GameInterface(GameType.GO, board_size=19)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=362)
        arena = TreeArena(MemoryConfig.cloud_preset())
        config = WaveConfig(
            initial_wave_size=1024,
            enable_adaptive_sizing=True
        )
        engine = WaveEngine(game, evaluator, arena, config)
        
        # Create root
        root = Node(state=game.create_initial_state(), parent=None, action=None, prior=1.0)
        root_id = arena.add_node(root)
        
        # Modify current wave size
        engine.current_wave_size = 768
        
        # Create wave without specifying size
        wave = engine.create_wave(root_id)
        
        assert wave.size == 768  # Uses current_wave_size
        
    def test_create_wave_with_invalid_root(self):
        """Test wave creation with non-existent root"""
        game = GameInterface(GameType.CHESS)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=4096)
        arena = TreeArena(MemoryConfig.laptop_preset())
        engine = WaveEngine(game, evaluator, arena, WaveConfig())
        
        # Try to create wave with invalid root
        # get_node returns None for non-existent nodes
        wave = engine.create_wave("nonexistent_root")
        
        # All leaf nodes should be None
        assert all(node is None for node in wave.leaf_nodes)


class TestSelectionPhase:
    """Test selection phase in detail"""
    
    def test_selection_deep_tree(self):
        """Test selection through a deep tree"""
        game = GameInterface(GameType.GOMOKU, board_size=9)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=81)
        arena = TreeArena(MemoryConfig.laptop_preset())
        engine = WaveEngine(game, evaluator, arena, WaveConfig())
        
        # Build a deep tree
        root = Node(state=game.create_initial_state(), parent=None, action=None, prior=1.0)
        root_id = arena.add_node(root)
        
        # Create a path of depth 5
        current = root
        for depth in range(5):
            # Expand current node
            moves = [depth * 10 + i for i in range(3)]  # 3 moves per node
            action_probs = {m: 1.0/3 for m in moves}
            child_states = {m: f"state_{depth}_{m}" for m in moves}
            current.expand(action_probs, child_states)
            
            # Add children to arena and select one
            for action, child in current.children.items():
                arena.add_node(child)
                child.visit_count = 10 - action % 10  # Vary visit counts
                
            # Move to first child for next iteration
            current = list(current.children.values())[0]
        
        # Create wave and run selection
        wave = engine.create_wave(root_id, size=50)
        engine._run_selection_phase(wave)
        
        # Check that paths diverged
        unique_leaves = set(id(n) for n in wave.leaf_nodes if n is not None)
        assert len(unique_leaves) > 1  # Paths should have diverged
        
        # All should reach leaf nodes
        for node in wave.leaf_nodes:
            if node is not None:
                assert node.is_leaf() or not node.children
                
    def test_selection_with_terminal_nodes(self):
        """Test selection when some paths reach terminal nodes"""
        game = GameInterface(GameType.GO, board_size=9)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=82)
        arena = TreeArena(MemoryConfig.laptop_preset())
        engine = WaveEngine(game, evaluator, arena, WaveConfig())
        
        # Create root with some terminal children
        root = Node(state=game.create_initial_state(), parent=None, action=None, prior=1.0)
        root_id = arena.add_node(root)
        
        # Expand root
        action_probs = {i: 0.25 for i in range(4)}
        child_states = {i: f"child_{i}" for i in range(4)}
        root.expand(action_probs, child_states)
        
        # Mark some children as terminal
        for i, (action, child) in enumerate(root.children.items()):
            arena.add_node(child)
            if i < 2:
                child.is_terminal = True
                child.terminal_value = 1.0 if i == 0 else -1.0
                
        # Run selection
        wave = engine.create_wave(root_id, size=20)
        engine._run_selection_phase(wave)
        
        # Some paths should end at terminal nodes
        terminal_count = sum(1 for n in wave.leaf_nodes if n and n.is_terminal)
        assert terminal_count > 0
        
    def test_selection_with_interference(self):
        """Test selection phase with interference enabled"""
        game = GameInterface(GameType.CHESS)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=4096)
        arena = TreeArena(MemoryConfig.laptop_preset())
        config = WaveConfig(enable_interference=True)
        
        # Mock interference engine
        mock_interference = MagicMock()
        mock_interference.compute_interference.return_value = np.random.rand(100)
        
        with patch('mcts.wave_engine.HAS_INTERFERENCE', True):
            with patch('mcts.wave_engine.InterferenceEngine', return_value=mock_interference):
                engine = WaveEngine(game, evaluator, arena, config)
                
                # Create simple tree
                root = Node(state=game.create_initial_state(), parent=None, action=None, prior=1.0)
                root_id = arena.add_node(root)
                root.expand({0: 0.5, 1: 0.5}, {0: "state_0", 1: "state_1"})
                
                for child in root.children.values():
                    arena.add_node(child)
                    arena.node_registry[id(child)] = (0, 'cpu')  # Mock registry entry
                
                # Run selection
                wave = engine.create_wave(root_id, size=100)
                wave.active_paths = 100  # Ensure > 1 for interference
                engine._run_selection_phase(wave)
                
                # Interference should have been called
                mock_interference.compute_interference.assert_called()


class TestExpansionPhase:
    """Test expansion phase in detail"""
    
    def test_expansion_with_no_legal_moves(self):
        """Test expansion when nodes have no legal moves"""
        game = Mock(spec=GameInterface)
        game.get_legal_moves.return_value = []  # No legal moves
        
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=100)
        arena = TreeArena(MemoryConfig.laptop_preset())
        engine = WaveEngine(game, evaluator, arena, WaveConfig())
        
        # Create unexpanded node
        node = Node(state="terminal_state", parent=None, action=None, prior=1.0)
        node_id = arena.add_node(node)
        
        # Create wave pointing to this node
        wave = Wave(size=10)
        for i in range(10):
            wave.leaf_nodes[i] = node
            
        # Run expansion
        engine._run_expansion_phase(wave)
        
        # Node should remain unexpanded
        assert not node.is_expanded
        assert len(node.children) == 0
        
    def test_expansion_with_phase_kicks(self):
        """Test expansion with phase kicks enabled"""
        game = GameInterface(GameType.GOMOKU, board_size=9)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=81)
        arena = TreeArena(MemoryConfig.laptop_preset())
        config = WaveConfig(enable_phase_kicks=True)
        engine = WaveEngine(game, evaluator, arena, config)
        
        # Create node with some visits (affects uncertainty)
        node = Node(state=game.create_initial_state(), parent=None, action=None, prior=1.0)
        node.visit_count = 10
        node_id = arena.add_node(node)
        
        # Create wave
        wave = Wave(size=5)
        for i in range(5):
            wave.leaf_nodes[i] = node
            
        # Run expansion
        engine._run_expansion_phase(wave)
        
        # Node should be expanded with phase-modified priors
        assert node.is_expanded
        assert len(node.children) > 0
        
        # Check that priors sum to approximately 1 (allowing for numerical errors)
        prior_sum = sum(child.prior for child in node.children.values())
        assert 0.99 < prior_sum < 1.01
        
    def test_expansion_batch_efficiency(self):
        """Test that expansion batches identical states efficiently"""
        game = GameInterface(GameType.GO, board_size=9)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=82)
        arena = TreeArena(MemoryConfig.laptop_preset())
        engine = WaveEngine(game, evaluator, arena, WaveConfig())
        
        # Create multiple nodes with same state
        state = game.create_initial_state()
        nodes = []
        for i in range(5):
            node = Node(state=state, parent=None, action=None, prior=1.0)
            arena.add_node(node)
            nodes.append(node)
            
        # Create wave with repeated nodes
        wave = Wave(size=20)
        for i in range(20):
            wave.leaf_nodes[i] = nodes[i % 5]
            
        # Count evaluator calls
        eval_count = 0
        original_evaluate = evaluator.evaluate
        def counting_evaluate(*args, **kwargs):
            nonlocal eval_count
            eval_count += 1
            return original_evaluate(*args, **kwargs)
        evaluator.evaluate = counting_evaluate
        
        # Run expansion
        engine._run_expansion_phase(wave)
        
        # Should only evaluate each unique state once
        assert eval_count == 5  # One per unique node
        
        # All nodes should be expanded
        for node in nodes:
            assert node.is_expanded


class TestEvaluationPhase:
    """Test evaluation phase in detail"""
    
    def test_evaluation_empty_wave(self):
        """Test evaluation with no valid nodes"""
        game = GameInterface(GameType.CHESS)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=4096)
        arena = TreeArena(MemoryConfig.laptop_preset())
        engine = WaveEngine(game, evaluator, arena, WaveConfig())
        
        # Create wave with all None nodes
        wave = Wave(size=50)
        
        # Run evaluation
        values = engine._run_evaluation_phase(wave)
        
        # Should return zeros
        assert values.shape == (50,)
        assert np.all(values == 0)
        
    def test_evaluation_mixed_nodes(self):
        """Test evaluation with mix of valid and None nodes"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=225)
        arena = TreeArena(MemoryConfig.laptop_preset())
        engine = WaveEngine(game, evaluator, arena, WaveConfig())
        
        # Create wave with some valid nodes
        wave = Wave(size=10)
        valid_indices = [0, 3, 5, 7]
        
        for i in valid_indices:
            state = game.create_initial_state()
            node = Node(state=state, parent=None, action=None, prior=1.0)
            wave.leaf_nodes[i] = node
            
        # Run evaluation
        values = engine._run_evaluation_phase(wave)
        
        # Check values
        assert values.shape == (10,)
        for i in range(10):
            if i in valid_indices:
                assert -1 <= values[i] <= 1  # Valid range for values
            else:
                assert values[i] == 0  # None nodes get 0
                
    def test_evaluation_state_caching(self):
        """Test that identical states are evaluated only once"""
        game = GameInterface(GameType.GO, board_size=9)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=82)
        arena = TreeArena(MemoryConfig.laptop_preset())
        engine = WaveEngine(game, evaluator, arena, WaveConfig())
        
        # Create two different states
        state1 = game.create_initial_state()
        state2 = game.apply_move(state1, 40)  # Different state
        
        # Create wave with repeated states
        wave = Wave(size=20)
        for i in range(20):
            if i % 3 == 0:
                node = Node(state=state1, parent=None, action=None, prior=1.0)
            elif i % 3 == 1:
                node = Node(state=state2, parent=None, action=None, prior=1.0)
            else:
                node = None
            if node:
                wave.leaf_nodes[i] = node
                
        # Count batch evaluations
        eval_count = 0
        original_evaluate_batch = evaluator.evaluate_batch
        def counting_evaluate_batch(*args, **kwargs):
            nonlocal eval_count
            eval_count += 1
            return original_evaluate_batch(*args, **kwargs)
        evaluator.evaluate_batch = counting_evaluate_batch
        
        # Run evaluation
        values = engine._run_evaluation_phase(wave)
        
        # Should batch evaluate once with 2 unique states
        assert eval_count == 1
        
        # Check that values are properly mapped
        value1 = values[0]  # state1 value
        value2 = values[1]  # state2 value
        
        for i in range(20):
            if i % 3 == 0:
                assert values[i] == value1
            elif i % 3 == 1:
                assert values[i] == value2
            else:
                assert values[i] == 0


class TestBackupPhase:
    """Test backup phase in detail"""
    
    def test_backup_single_path(self):
        """Test backup along a single path"""
        game = GameInterface(GameType.CHESS)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=4096)
        arena = TreeArena(MemoryConfig.laptop_preset())
        engine = WaveEngine(game, evaluator, arena, WaveConfig())
        
        # Create a path: root -> child -> grandchild
        root = Node(state="root", parent=None, action=None, prior=1.0)
        child = Node(state="child", parent=root, action=0, prior=0.5)
        grandchild = Node(state="grandchild", parent=child, action=1, prior=0.3)
        
        root.children[0] = child
        child.children[1] = grandchild
        
        # Set initial visit counts
        root.visit_count = 10
        child.visit_count = 5
        grandchild.visit_count = 2
        
        # Create wave
        wave = Wave(size=1)
        wave.leaf_nodes[0] = grandchild
        
        # Run backup with value 0.8
        values = np.array([0.8])
        engine._run_backup_phase(wave, values)
        
        # Check that all nodes in path were updated
        assert grandchild.visit_count == 3  # 2 + 1
        assert child.visit_count == 6  # 5 + 1
        assert root.visit_count == 11  # 10 + 1
        
        # Values should be updated (flipped for opponent)
        assert grandchild.value_sum > 0  # Positive value
        assert child.value_sum < 0  # Negative (opponent)
        assert root.value_sum > 0  # Positive again
        
    def test_backup_multiple_paths(self):
        """Test backup with multiple independent paths"""
        game = GameInterface(GameType.GOMOKU, board_size=9)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=81)
        arena = TreeArena(MemoryConfig.laptop_preset())
        engine = WaveEngine(game, evaluator, arena, WaveConfig())
        
        # Create tree with multiple branches
        root = Node(state="root", parent=None, action=None, prior=1.0)
        root.visit_count = 100
        
        # Create branches
        children = []
        leaves = []
        for i in range(3):
            child = Node(state=f"child_{i}", parent=root, action=i, prior=0.33)
            child.visit_count = 30
            root.children[i] = child
            children.append(child)
            
            # Add grandchildren
            for j in range(2):
                gc = Node(state=f"gc_{i}_{j}", parent=child, action=j, prior=0.5)
                gc.visit_count = 10
                child.children[j] = gc
                leaves.append(gc)
                
        # Create wave with all leaves
        wave = Wave(size=6)
        for i, leaf in enumerate(leaves):
            wave.leaf_nodes[i] = leaf
            
        # Different values for each path
        values = np.array([0.5, -0.5, 0.8, -0.8, 0.3, -0.3])
        
        # Run backup
        engine._run_backup_phase(wave, values)
        
        # Check root received all backups
        assert root.visit_count == 106  # 100 + 6
        
        # Each child should have 2 backups
        for child in children:
            assert child.visit_count == 32  # 30 + 2
            
        # Each leaf should have 1 backup
        for leaf in leaves:
            assert leaf.visit_count == 11  # 10 + 1
            
    def test_backup_with_none_nodes(self):
        """Test backup handles None nodes gracefully"""
        game = GameInterface(GameType.GO, board_size=19)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=362)
        arena = TreeArena(MemoryConfig.laptop_preset())
        engine = WaveEngine(game, evaluator, arena, WaveConfig())
        
        # Create valid node
        node = Node(state="test", parent=None, action=None, prior=1.0)
        initial_visits = node.visit_count
        
        # Create wave with mix of valid and None
        wave = Wave(size=5)
        wave.leaf_nodes[0] = node
        wave.leaf_nodes[1] = None
        wave.leaf_nodes[2] = node
        wave.leaf_nodes[3] = None
        wave.leaf_nodes[4] = None
        
        # Values for all positions
        values = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        
        # Run backup
        engine._run_backup_phase(wave, values)
        
        # Only valid nodes should be updated
        assert node.visit_count == initial_visits + 2  # Positions 0 and 2


class TestAdaptiveWaveSizing:
    """Test adaptive wave sizing functionality"""
    
    def test_wave_size_increase(self):
        """Test wave size increases when processing is fast"""
        game = GameInterface(GameType.CHESS)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=4096)
        arena = TreeArena(MemoryConfig.desktop_preset())
        config = WaveConfig(
            min_wave_size=256,
            max_wave_size=2048,
            initial_wave_size=512,
            enable_adaptive_sizing=True
        )
        engine = WaveEngine(game, evaluator, arena, config)
        
        # Simulate fast processing
        engine.stats['total_waves'] = 10
        engine.stats['total_time'] = 0.5  # 50ms per wave average
        
        initial_size = engine.current_wave_size
        engine._update_wave_size()
        
        # Size should increase
        assert engine.current_wave_size > initial_size
        assert engine.current_wave_size <= config.max_wave_size
        
    def test_wave_size_decrease(self):
        """Test wave size decreases when processing is slow"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=225)
        arena = TreeArena(MemoryConfig.laptop_preset())
        config = WaveConfig(
            min_wave_size=256,
            max_wave_size=2048,
            initial_wave_size=1024,
            enable_adaptive_sizing=True
        )
        engine = WaveEngine(game, evaluator, arena, config)
        
        # Simulate slow processing
        engine.stats['total_waves'] = 5
        engine.stats['total_time'] = 4.0  # 800ms per wave average
        
        initial_size = engine.current_wave_size
        engine._update_wave_size()
        
        # Size should decrease
        assert engine.current_wave_size < initial_size
        assert engine.current_wave_size >= config.min_wave_size
        
    def test_wave_size_limits(self):
        """Test wave size respects min/max limits"""
        game = GameInterface(GameType.GO, board_size=9)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=82)
        arena = TreeArena(MemoryConfig.laptop_preset())
        config = WaveConfig(
            min_wave_size=100,
            max_wave_size=200,
            initial_wave_size=150,
            enable_adaptive_sizing=True
        )
        engine = WaveEngine(game, evaluator, arena, config)
        
        # Test max limit
        engine.current_wave_size = 190
        engine.stats['total_waves'] = 1
        engine.stats['total_time'] = 0.01  # Very fast
        engine._update_wave_size()
        assert engine.current_wave_size == 200  # Capped at max
        
        # Test min limit
        engine.current_wave_size = 110
        engine.stats['total_waves'] = 1
        engine.stats['total_time'] = 1.0  # Very slow
        engine._update_wave_size()
        assert engine.current_wave_size == 100  # Capped at min


class TestStatistics:
    """Test statistics collection and reporting"""
    
    def test_statistics_empty(self):
        """Test statistics when no waves processed"""
        game = GameInterface(GameType.CHESS)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=4096)
        arena = TreeArena(MemoryConfig.laptop_preset())
        engine = WaveEngine(game, evaluator, arena, WaveConfig())
        
        stats = engine.get_statistics()
        
        assert stats['total_waves'] == 0
        assert stats['total_simulations'] == 0
        assert stats['average_wave_size'] == 0
        assert stats['average_wave_time'] == 0
        assert stats['current_wave_size'] == engine.config.initial_wave_size
        
    def test_statistics_after_waves(self):
        """Test statistics after processing waves"""
        game = GameInterface(GameType.GOMOKU, board_size=9)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=81)
        arena = TreeArena(MemoryConfig.laptop_preset())
        engine = WaveEngine(game, evaluator, arena, WaveConfig())
        
        # Manually set statistics
        engine.stats['total_waves'] = 5
        engine.stats['total_simulations'] = 2560  # 5 waves * 512 average
        engine.stats['total_time'] = 2.5
        engine.stats['selection_time'] = 0.5
        engine.stats['expansion_time'] = 0.8
        engine.stats['evaluation_time'] = 0.9
        engine.stats['backup_time'] = 0.3
        
        stats = engine.get_statistics()
        
        assert stats['total_waves'] == 5
        assert stats['total_simulations'] == 2560
        assert stats['average_wave_size'] == 512
        assert stats['average_wave_time'] == 0.5
        assert stats['selection_time'] == 0.5
        assert stats['expansion_time'] == 0.8
        assert stats['evaluation_time'] == 0.9
        assert stats['backup_time'] == 0.3
        
    def test_reset_statistics(self):
        """Test resetting statistics"""
        game = GameInterface(GameType.GO, board_size=19)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=362)
        arena = TreeArena(MemoryConfig.cloud_preset())
        engine = WaveEngine(game, evaluator, arena, WaveConfig())
        
        # Set some statistics
        engine.stats['total_waves'] = 100
        engine.stats['total_simulations'] = 50000
        engine.stats['some_custom_stat'] = 42
        
        # Reset
        engine.reset_statistics()
        
        # Check reset
        assert engine.stats['total_waves'] == 0
        assert engine.stats['total_simulations'] == 0
        assert 'some_custom_stat' not in engine.stats


class TestPhaseKicks:
    """Test phase kick implementation"""
    
    def test_phase_kicks_high_uncertainty(self):
        """Test phase kicks with high uncertainty (low visits)"""
        game = GameInterface(GameType.CHESS)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=4096)
        arena = TreeArena(MemoryConfig.laptop_preset())
        engine = WaveEngine(game, evaluator, arena, WaveConfig())
        
        # Node with low visits (high uncertainty)
        node = Node(state="test", parent=None, action=None, prior=1.0)
        node.visit_count = 1
        
        # Original probabilities
        action_probs = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        
        # Apply phase kicks
        modified = engine._apply_phase_kicks(action_probs, node)
        
        # Should be normalized
        assert abs(sum(modified.values()) - 1.0) < 1e-6
        
        # Should be different from original (due to phase)
        assert any(abs(modified[a] - action_probs[a]) > 1e-6 for a in action_probs)
        
    def test_phase_kicks_low_uncertainty(self):
        """Test phase kicks with low uncertainty (high visits)"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=225)
        arena = TreeArena(MemoryConfig.laptop_preset())
        engine = WaveEngine(game, evaluator, arena, WaveConfig())
        
        # Node with high visits (low uncertainty)
        node = Node(state="test", parent=None, action=None, prior=1.0)
        node.visit_count = 1000
        
        # Original probabilities
        action_probs = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
        
        # Apply phase kicks
        modified = engine._apply_phase_kicks(action_probs, node)
        
        # Should be normalized
        assert abs(sum(modified.values()) - 1.0) < 1e-6
        
        # Changes should be smaller due to low uncertainty
        max_change = max(abs(modified[a] - action_probs[a]) for a in action_probs)
        assert max_change < 0.05  # Small changes
        
    def test_phase_kicks_empty_probs(self):
        """Test phase kicks with empty probabilities"""
        game = GameInterface(GameType.GO, board_size=9)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=82)
        arena = TreeArena(MemoryConfig.laptop_preset())
        engine = WaveEngine(game, evaluator, arena, WaveConfig())
        
        node = Node(state="test", parent=None, action=None, prior=1.0)
        
        # Empty probabilities
        action_probs = {}
        modified = engine._apply_phase_kicks(action_probs, node)
        
        # Should return empty
        assert len(modified) == 0
        
    def test_phase_kicks_single_action(self):
        """Test phase kicks with single action"""
        game = GameInterface(GameType.CHESS)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=4096)
        arena = TreeArena(MemoryConfig.laptop_preset())
        engine = WaveEngine(game, evaluator, arena, WaveConfig())
        
        node = Node(state="test", parent=None, action=None, prior=1.0)
        
        # Single action with probability 1
        action_probs = {42: 1.0}
        modified = engine._apply_phase_kicks(action_probs, node)
        
        # Should still sum to 1
        assert abs(modified[42] - 1.0) < 1e-6


class TestInterferenceComputation:
    """Test interference computation (when available)"""
    
    def test_compute_interference_empty_paths(self):
        """Test interference with empty paths"""
        game = GameInterface(GameType.GOMOKU, board_size=9)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=81)
        arena = TreeArena(MemoryConfig.laptop_preset())
        config = WaveConfig(enable_interference=True)
        
        with patch('mcts.wave_engine.HAS_INTERFERENCE', True):
            mock_interference = MagicMock()
            with patch('mcts.wave_engine.InterferenceEngine', return_value=mock_interference):
                engine = WaveEngine(game, evaluator, arena, config)
                
                # Create wave with empty paths (all leaf nodes are None)
                wave = Wave(size=10)
                wave.leaf_nodes = [None for _ in range(10)]
                wave.paths = [[] for _ in range(10)]
                
                # Apply interference - should handle empty paths gracefully
                engine._apply_interference(wave)
                
                # For empty paths, compute_interference should NOT be called
                # because the method returns early when there are no valid paths
                mock_interference.compute_interference.assert_not_called()
                
    def test_compute_interference_single_path(self):
        """Test interference with single path"""
        game = GameInterface(GameType.GO, board_size=19)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=362)
        arena = TreeArena(MemoryConfig.laptop_preset())
        config = WaveConfig(enable_interference=True)
        
        engine = WaveEngine(game, evaluator, arena, config)
        engine.interference_engine = None  # Disabled
        
        # Single path
        paths = [[1, 2, 3, 4]]
        
        # Should handle gracefully without interference engine
        # This should not raise an exception
        result = engine._apply_interference(Wave(size=1))
        assert result is None  # No interference engine


class TestCompleteWaveProcessing:
    """Test complete wave processing scenarios"""
    
    def test_wave_lifecycle(self):
        """Test complete wave lifecycle"""
        game = GameInterface(GameType.CHESS)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=4096)
        arena = TreeArena(MemoryConfig.desktop_preset())
        config = WaveConfig(initial_wave_size=128)
        engine = WaveEngine(game, evaluator, arena, config)
        
        # Create initial state
        root = Node(state=game.create_initial_state(), parent=None, action=None, prior=1.0)
        root_id = arena.add_node(root)
        
        # Process wave
        wave = engine.process_wave(root_id)
        
        # Check wave completed all phases
        assert wave.completed
        assert wave.phase == 4  # After backup
        assert wave.size == 128
        assert wave.root_id == root_id
        
        # Check statistics updated
        assert engine.stats['total_waves'] == 1
        assert engine.stats['total_simulations'] >= wave.size
        
        # Check wave added to history
        assert len(engine.wave_history) == 1
        assert engine.wave_history[0] == wave
        
        # Current wave should be None
        assert engine.current_wave is None
        
    def test_multiple_waves_sequential(self):
        """Test processing multiple waves sequentially"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=225)
        arena = TreeArena(MemoryConfig.laptop_preset())
        engine = WaveEngine(game, evaluator, arena, WaveConfig())
        
        # Create root
        root = Node(state=game.create_initial_state(), parent=None, action=None, prior=1.0)
        root_id = arena.add_node(root)
        
        # Process multiple waves
        num_waves = 5
        for i in range(num_waves):
            wave = engine.process_wave(root_id, size=256)
            assert wave.completed
            
        # Check statistics
        assert engine.stats['total_waves'] == num_waves
        assert engine.stats['total_simulations'] >= num_waves * 256
        assert len(engine.wave_history) == num_waves
        
        # Tree should have grown
        assert root.visit_count >= num_waves * 256
        assert root.is_expanded
        
    def test_wave_with_errors(self):
        """Test wave processing handles errors gracefully"""
        game = GameInterface(GameType.GO, board_size=9)
        evaluator = Mock(spec=MockEvaluator)
        # Make evaluator.evaluate return a tuple for single evaluation
        evaluator.evaluate.return_value = (np.ones(82) / 82, 0.0)
        # Make batch evaluation raise exception
        evaluator.evaluate_batch.side_effect = RuntimeError("Evaluation failed")
        
        arena = TreeArena(MemoryConfig.laptop_preset())
        engine = WaveEngine(game, evaluator, arena, WaveConfig())
        
        # Create root with a child to trigger expansion/evaluation
        root = Node(state=game.create_initial_state(), parent=None, action=None, prior=1.0)
        root_id = arena.add_node(root)
        
        # Add a child to make it expandable
        child = Node(state=game.create_initial_state(), parent=root, action=40, prior=0.1)
        child_id = arena.add_node(child)
        root.children[40] = child
        
        # Process wave should raise the error during evaluation phase
        with pytest.raises(RuntimeError):
            engine.process_wave(root_id)
            
    def test_wave_performance_tracking(self):
        """Test wave performance is tracked correctly"""
        game = GameInterface(GameType.CHESS)
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=4096)
        arena = TreeArena(MemoryConfig.cloud_preset())
        engine = WaveEngine(game, evaluator, arena, WaveConfig())
        
        # Create root
        root = Node(state=game.create_initial_state(), parent=None, action=None, prior=1.0)
        root_id = arena.add_node(root)
        
        # Process wave
        wave = engine.process_wave(root_id, size=1024)
        
        # Check timing statistics exist and are reasonable
        stats = engine.get_statistics()
        assert stats['selection_time'] > 0
        assert stats['expansion_time'] > 0
        assert stats['evaluation_time'] > 0
        assert stats['backup_time'] > 0
        assert stats['total_time'] > 0
        
        # Total time should be approximately sum of phases
        phase_sum = (stats['selection_time'] + stats['expansion_time'] + 
                    stats['evaluation_time'] + stats['backup_time'])
        assert abs(stats['total_time'] - phase_sum) < 0.1  # Allow small overhead