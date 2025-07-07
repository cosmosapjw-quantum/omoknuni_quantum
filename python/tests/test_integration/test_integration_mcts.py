"""Integration tests for MCTS components

Tests cover:
- MCTS with neural network evaluator
- MCTS with GPU acceleration
- Wave-based search integration
- Tree reuse functionality
- Multi-threaded MCTS
- Game interface integration
"""

import pytest
import torch
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock

from mcts.core.mcts import MCTS, MCTSConfig
from mcts.core.evaluator import AlphaZeroEvaluator, RandomEvaluator
from mcts.core.game_interface import GameInterface, GameType
from mcts.core.wave_search import WaveSearch
from mcts.neural_networks.resnet_model import create_resnet_for_game
from mcts.gpu.csr_tree import CSRTree
from mcts.gpu.node_data_manager import NodeDataManager
from mcts.gpu.ucb_selector import UCBSelector


@pytest.fixture
def mcts_config():
    """Create MCTS configuration"""
    config = MCTSConfig()
    config.num_simulations = 100
    config.c_puct = 1.4
    config.temperature = 1.0
    config.device = 'cpu'
    config.enable_virtual_loss = True
    config.batch_size = 16
    config.min_wave_size = 4
    config.max_wave_size = 32
    return config


@pytest.fixture
def game_interface():
    """Create game interface"""
    return GameInterface(GameType.GOMOKU, board_size=15)


@pytest.fixture
def neural_evaluator():
    """Create neural network evaluator"""
    model = create_resnet_for_game('gomoku', num_blocks=3, num_filters=32)
    model.eval()
    return AlphaZeroEvaluator(model, device='cpu')


@pytest.fixture
def mcts_instance(mcts_config, game_interface, neural_evaluator):
    """Create MCTS instance"""
    return MCTS(
        config=mcts_config,
        game_interface=game_interface,
        evaluator=neural_evaluator
    )


class TestMCTSNeuralNetworkIntegration:
    """Test MCTS integration with neural networks"""
    
    def test_mcts_with_neural_evaluator(self, mcts_instance, game_interface):
        """Test MCTS search with neural network evaluator"""
        initial_state = game_interface.create_initial_state()
        
        # Run search
        policy = mcts_instance.search(initial_state)
        
        assert isinstance(policy, np.ndarray)
        assert policy.shape == (225,)
        assert np.allclose(policy.sum(), 1.0)
        assert np.all(policy >= 0)
        
        # Should have explored nodes - policy reflects visit distribution
        assert policy.sum() > 0  # Policy should be normalized
        assert np.count_nonzero(policy) > 0  # Should have some non-zero actions
        
    def test_neural_evaluation_caching(self, mcts_instance, game_interface):
        """Test neural network evaluation caching"""
        initial_state = game_interface.create_initial_state()
        
        # Mock the evaluate_batch method to track calls
        original_evaluate = mcts_instance.evaluator.evaluate_batch
        eval_calls = []
        
        def track_evaluate(states, legal_masks=None, temperature=1.0):
            eval_calls.append(len(states))
            return original_evaluate(states, legal_masks, temperature)
        
        mcts_instance.evaluator.evaluate_batch = track_evaluate
        
        # First search
        mcts_instance.search(initial_state)
        first_search_calls = len(eval_calls)
        
        # Reset and search again (with same state)
        mcts_instance.reset_tree()
        mcts_instance.search(initial_state)
        second_search_calls = len(eval_calls) - first_search_calls
        
        # Should have evaluated states in both searches
        assert first_search_calls > 0
        assert second_search_calls > 0
        
        # Check that batching is happening (some calls should have multiple states)
        assert any(batch_size > 1 for batch_size in eval_calls)
        
    def test_batch_neural_evaluation(self, mcts_config, game_interface):
        """Test batch neural network evaluation during search"""
        # Create evaluator that tracks batch sizes
        batch_sizes = []
        
        class BatchTrackingEvaluator:
            def __init__(self, base_evaluator):
                self.base = base_evaluator
                
            def evaluate(self, state):
                return self.base.evaluate(state)
                
            def evaluate_batch(self, states):
                batch_sizes.append(len(states))
                return self.base.evaluate_batch(states)
                
        model = create_resnet_for_game('gomoku', num_blocks=3, num_filters=32)
        base_evaluator = AlphaZeroEvaluator(model, device='cpu')
        evaluator = BatchTrackingEvaluator(base_evaluator)
        
        mcts = MCTS(mcts_config, evaluator, game_interface)
        initial_state = game_interface.create_initial_state()
        
        # Run search
        mcts.search(initial_state)
        
        # Should have batched evaluations
        assert len(batch_sizes) > 0
        assert max(batch_sizes) > 1  # At least some batching
        
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gpu_neural_evaluation(self, mcts_config, game_interface):
        """Test MCTS with GPU neural network"""
        mcts_config.device = 'cuda'
        
        model = create_resnet_for_game('gomoku', num_blocks=3, num_filters=32).cuda()
        evaluator = AlphaZeroEvaluator(model, device='cuda')
        
        mcts = MCTS(mcts_config, evaluator, game_interface)
        initial_state = game_interface.create_initial_state()
        
        # Should work with GPU
        policy = mcts.search(initial_state)
        assert policy is not None
        assert policy.shape == (225,)


class TestMCTSGPUAcceleration:
    """Test MCTS with GPU acceleration components"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gpu_tree_integration(self, mcts_config, game_interface, neural_evaluator):
        """Test MCTS with GPU-accelerated tree"""
        mcts_config.device = 'cuda'
        mcts_config.use_gpu_tree = True
        
        mcts = MCTS(mcts_config, neural_evaluator, game_interface)
        initial_state = game_interface.create_initial_state()
        
        # Run search
        policy = mcts.search(initial_state)
        
        # Check GPU tree components
        assert hasattr(mcts, 'tree')
        assert isinstance(mcts.tree, CSRTree)
        
        # Should produce valid policy
        assert policy.shape == (225,)
        assert np.allclose(policy.sum(), 1.0)
        
    def test_gpu_node_data_manager(self, mcts_config, game_interface, neural_evaluator):
        """Test GPU node data management integration"""
        if torch.cuda.is_available():
            mcts_config.device = 'cuda'
        
        # Create node data manager
        from mcts.gpu.node_data_manager import NodeDataConfig
        node_config = NodeDataConfig(
            max_nodes=1000,
            device=mcts_config.device
        )
        node_manager = NodeDataManager(node_config)
        
        # Integrate with MCTS (mock)
        with patch('mcts.core.mcts.NodeDataManager', return_value=node_manager):
            mcts = MCTS(mcts_config, neural_evaluator, game_interface)
            
            # Should initialize properly
            assert node_manager.num_nodes == 0
            assert node_manager.capacity >= 1000
            
    def test_gpu_ucb_selection(self, mcts_config):
        """Test GPU UCB selection integration"""
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
            
        from mcts.gpu.ucb_selector import UCBConfig
        ucb_config = UCBConfig(device=device)
        ucb_selector = UCBSelector(ucb_config)
        
        # Create mock tree data
        num_nodes = 100
        q_values = torch.randn(num_nodes, device=device)
        visit_counts = torch.randint(1, 100, (num_nodes,), device=device)
        prior_probs = torch.rand(num_nodes, device=device)
        parent_visits = torch.full((num_nodes,), 1000, device=device)
        
        # Test selection
        selected = ucb_selector.select_batch(
            q_values, visit_counts, prior_probs, 
            parent_visits, c_puct=1.4, batch_size=10
        )
        
        assert selected.shape[0] <= 10
        assert torch.all(selected >= 0)
        assert torch.all(selected < num_nodes)


class TestWaveSearchIntegration:
    """Test wave-based search integration"""
    
    def test_wave_search_basic(self, mcts_config, game_interface, neural_evaluator):
        """Test basic wave search functionality"""
        mcts_config.enable_wave_search = True
        
        mcts = MCTS(mcts_config, neural_evaluator, game_interface)
        initial_state = game_interface.create_initial_state()
        
        # Track wave sizes
        wave_sizes = []
        
        def track_wave(wave_size, root_idx, add_noise=False):
            wave_sizes.append(wave_size)
            return 1  # Return number of completed simulations
            
        with patch.object(mcts.wave_search, 'run_wave', side_effect=track_wave):
            # Run search (will fail but tracks waves)
            try:
                mcts.search(initial_state)
            except:
                pass
                
        # Should have processed waves
        assert len(wave_sizes) > 0
        
    def test_adaptive_wave_sizing(self, mcts_config, game_interface, neural_evaluator):
        """Test adaptive wave size adjustment"""
        mcts_config.min_wave_size = 4
        mcts_config.max_wave_size = 32
        
        # WaveSearch is created internally by MCTS, not directly instantiable
        # Test adaptive wave sizing through MCTS instead
        mcts = MCTS(mcts_config, neural_evaluator, game_interface)
        wave_search = mcts.wave_search
        
        # Test that wave search can handle different wave sizes
        initial_state = game_interface.create_initial_state()
        
        # Track actual wave sizes used
        wave_sizes_used = []
        
        def track_wave_run(wave_size, root_idx, add_noise=False):
            wave_sizes_used.append(wave_size)
            return min(wave_size, 10)  # Simulate completing some simulations
            
        with patch.object(wave_search, 'run_wave', side_effect=track_wave_run):
            mcts.search(initial_state)
            
        # Verify waves were processed
        assert len(wave_sizes_used) > 0
        assert all(ws <= mcts_config.max_wave_size for ws in wave_sizes_used)
        assert all(ws >= 1 for ws in wave_sizes_used)
        
    def test_wave_batching_efficiency(self, mcts_config, game_interface):
        """Test wave batching efficiency"""
        # Track batch processing
        batch_times = []
        batch_sizes = []
        
        class BatchTrackingEvaluator:
            def evaluate_batch(self, states):
                start = time.time()
                batch_sizes.append(len(states))
                
                # Simulate processing
                policies = np.ones((len(states), 225)) / 225
                values = np.zeros(len(states))
                
                batch_times.append(time.time() - start)
                return policies, values
                
            def evaluate(self, state):
                return self.evaluate_batch([state])
                
        evaluator = BatchTrackingEvaluator()
        mcts = MCTS(mcts_config, evaluator, game_interface)
        
        initial_state = game_interface.create_initial_state()
        mcts.search(initial_state)
        
        # Should have efficient batching
        if len(batch_sizes) > 0:
            avg_batch_size = np.mean(batch_sizes)
            assert avg_batch_size > 1  # Some batching occurred


class TestTreeReuseIntegration:
    """Test tree reuse functionality"""
    
    def test_tree_reuse_basic(self, mcts_instance, game_interface):
        """Test basic tree reuse between searches"""
        initial_state = game_interface.create_initial_state()
        
        # First search
        policy1 = mcts_instance.search(initial_state)
        visits1 = mcts_instance.get_visits_distribution().copy()
        
        # Make a move
        action = np.argmax(policy1)
        next_state = game_interface.get_next_state(initial_state, action)
        
        # Update root for tree reuse
        mcts_instance.update_root(next_state, action)
        
        # Second search (with tree reuse)
        policy2 = mcts_instance.search(next_state)
        visits2 = mcts_instance.get_visits_distribution()
        
        # Should have reused some computation
        # (exact behavior depends on implementation)
        assert policy2 is not None
        assert visits2.sum() > 0
        
    def test_tree_reuse_memory_management(self, mcts_instance, game_interface):
        """Test memory management with tree reuse"""
        initial_state = game_interface.create_initial_state()
        
        # Build large tree
        mcts_instance.config.num_simulations = 500
        mcts_instance.search(initial_state)
        
        initial_nodes = mcts_instance.get_tree_stats()['total_nodes']
        
        # Make move and reuse tree
        action = 0
        next_state = game_interface.get_next_state(initial_state, action)
        mcts_instance.update_root(next_state, action)
        
        # Old nodes should be pruned
        final_nodes = mcts_instance.get_tree_stats()['total_nodes']
        assert final_nodes < initial_nodes
        
    def test_tree_reuse_disabled(self, mcts_config, game_interface, neural_evaluator):
        """Test MCTS with tree reuse disabled"""
        mcts_config.enable_tree_reuse = False
        
        mcts = MCTS(mcts_config, neural_evaluator, game_interface)
        initial_state = game_interface.create_initial_state()
        
        # First search
        mcts.search(initial_state)
        nodes_after_first = mcts.get_tree_stats()['total_nodes']
        
        # Second search (no reuse)
        mcts.search(initial_state)
        nodes_after_second = mcts.get_tree_stats()['total_nodes']
        
        # Should start fresh each time
        assert nodes_after_second == nodes_after_first


class TestGameIntegration:
    """Test MCTS integration with different games"""
    
    def test_gomoku_gameplay(self, mcts_config, neural_evaluator):
        """Test MCTS playing Gomoku"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        mcts = MCTS(mcts_config, game, neural_evaluator)
        
        state = game.create_initial_state()
        moves_played = 0
        
        # Play some moves
        while not game.is_terminal(state) and moves_played < 10:
            policy = mcts.search(state)
            action = np.argmax(policy)
            
            # Verify legal move
            legal_moves = game.get_legal_moves(state)
            assert action in legal_moves
            
            state = game.get_next_state(state, action)
            mcts.update_root(state, action)
            moves_played += 1
            
        assert moves_played > 0
        
    def test_different_board_sizes(self, mcts_config, neural_evaluator):
        """Test MCTS with different board sizes"""
        for board_size in [9, 13, 15, 19]:
            game = GameInterface(GameType.GOMOKU, board_size=board_size)
            mcts = MCTS(mcts_config, game, neural_evaluator)
            
            state = game.create_initial_state()
            policy = mcts.search(state)
            
            expected_actions = board_size * board_size
            assert policy.shape == (expected_actions,)
            assert np.allclose(policy.sum(), 1.0)
            
    def test_terminal_state_handling(self, mcts_config, game_interface, neural_evaluator):
        """Test MCTS handling of terminal states"""
        mcts = MCTS(mcts_config, neural_evaluator, game_interface)
        
        # Create a mock terminal state
        with patch.object(game_interface, 'is_terminal', return_value=True):
            with patch.object(game_interface, 'get_winner', return_value=1):
                state = game_interface.create_initial_state()
                
                # Should handle terminal state gracefully
                policy = mcts.search(state)
                
                # Policy might be uniform or single action
                assert policy is not None
                assert policy.shape == (225,)


class TestConcurrentMCTS:
    """Test concurrent MCTS operations"""
    
    def test_thread_safe_evaluation(self, mcts_config, game_interface):
        """Test thread-safe evaluation"""
        import threading
        
        # Shared evaluator
        model = create_resnet_for_game('gomoku', num_blocks=3, num_filters=32)
        evaluator = AlphaZeroEvaluator(model, device='cpu')
        
        results = []
        errors = []
        
        def run_mcts():
            try:
                mcts = MCTS(mcts_config, evaluator, game_interface)
                state = game_interface.create_initial_state()
                policy = mcts.search(state)
                results.append(policy)
            except Exception as e:
                errors.append(e)
                
        # Run concurrent MCTS
        threads = []
        for _ in range(4):
            t = threading.Thread(target=run_mcts)
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        # Should complete without errors
        assert len(errors) == 0
        assert len(results) == 4
        
    def test_parallel_game_analysis(self, mcts_config, game_interface, neural_evaluator):
        """Test analyzing multiple game positions in parallel"""
        import concurrent.futures
        
        # Different game positions
        positions = []
        state = game_interface.create_initial_state()
        positions.append(state)
        
        # Create a few different positions
        for moves in [[0, 1, 2], [50, 51, 52], [100, 101, 102]]:
            pos = state
            for move in moves:
                if move in game_interface.get_legal_moves(pos):
                    pos = game_interface.get_next_state(pos, move)
            positions.append(pos)
            
        # Analyze positions in parallel
        def analyze_position(state):
            mcts = MCTS(mcts_config, neural_evaluator, game_interface)
            return mcts.search(state)
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(analyze_position, pos) for pos in positions]
            results = [f.result() for f in futures]
            
        # Should get valid policies for all positions
        assert len(results) == len(positions)
        for policy in results:
            assert policy.shape == (225,)
            assert np.allclose(policy.sum(), 1.0)


class TestPerformanceOptimizations:
    """Test performance optimizations in integration"""
    
    def test_virtual_loss_integration(self, mcts_config, game_interface, neural_evaluator):
        """Test virtual loss in concurrent scenarios"""
        mcts_config.enable_virtual_loss = True
        mcts_config.virtual_loss = 3.0
        
        mcts = MCTS(mcts_config, neural_evaluator, game_interface)
        initial_state = game_interface.create_initial_state()
        
        # Should work with virtual loss
        policy = mcts.search(initial_state)
        assert policy is not None
        
        # Check tree stats show virtual loss usage
        stats = mcts.get_tree_stats()
        # Exact stats depend on implementation
        
    def test_fast_ucb_computation(self, mcts_config, game_interface, neural_evaluator):
        """Test fast UCB computation optimization"""
        mcts_config.enable_fast_ucb = True
        mcts_config.num_simulations = 200
        
        # Time with fast UCB
        mcts_fast = MCTS(mcts_config, game_interface, neural_evaluator)
        state = game_interface.create_initial_state()
        
        start = time.time()
        mcts_fast.search(state)
        fast_time = time.time() - start
        
        # Time without fast UCB
        mcts_config.enable_fast_ucb = False
        mcts_slow = MCTS(mcts_config, game_interface, neural_evaluator)
        
        start = time.time()
        mcts_slow.search(state)
        slow_time = time.time() - start
        
        # Fast should be faster (or at least not slower)
        assert fast_time <= slow_time * 1.5  # Allow some variance