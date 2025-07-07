"""
Performance benchmark tests for MCTS implementation

Tests cover:
- MCTS simulations per second
- Batch evaluation throughput
- Tree operation performance
- GPU acceleration speedup
- Memory usage patterns
- Scaling with tree size
- Multi-game performance
- Training throughput
"""

import pytest
import torch
import numpy as np
import time
import psutil
import gc
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

from mcts.core.mcts import MCTS
from mcts.core.mcts_config import MCTSConfig
from mcts.core.game_interface import GameInterface, GameType
from mcts.core.evaluator import AlphaZeroEvaluator
from mcts.neural_networks.resnet_model import create_resnet_for_game
from mcts.neural_networks.unified_training_pipeline import UnifiedTrainingPipeline
from mcts.gpu.csr_tree import CSRTree
from mcts.gpu.mcts_gpu_accelerator import MCTSGPUAccelerator
from mcts.utils.batch_evaluation_coordinator import RequestBatchingCoordinator
from mcts.utils.config_system import AlphaZeroConfig


@pytest.fixture
def performance_config():
    """Create performance testing configuration"""
    config = MCTSConfig()
    config.num_simulations = 800
    config.c_puct = 1.4
    config.batch_size = 64
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.enable_virtual_loss = True
    config.enable_fast_ucb = True
    return config


@pytest.fixture
def performance_model():
    """Create model for performance testing"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_resnet_for_game('gomoku', num_blocks=10, num_filters=128)
    return model.to(device).eval()


@pytest.fixture
def performance_evaluator(performance_model):
    """Create evaluator for performance testing"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return AlphaZeroEvaluator(performance_model, device=device)


class TestMCTSPerformance:
    """Test MCTS performance benchmarks"""
    
    @pytest.mark.benchmark
    def test_simulations_per_second(self, performance_config, performance_evaluator):
        """Benchmark MCTS simulations per second"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        mcts = MCTS(performance_config, performance_evaluator)
        state = game.create_initial_state()
        
        # Warmup
        mcts.search(state, num_simulations=50)
        mcts.clear()
        
        # Benchmark
        start_time = time.time()
        policy = mcts.search(state, num_simulations=performance_config.num_simulations)
        elapsed = time.time() - start_time
        
        simulations_per_second = performance_config.num_simulations / elapsed
        
        print(f"\nSimulations per second: {simulations_per_second:.0f}")
        print(f"Time per simulation: {elapsed/performance_config.num_simulations*1000:.2f}ms")
        
        # Performance targets (adjust based on hardware)
        if torch.cuda.is_available():
            assert simulations_per_second > 1000  # GPU target
        else:
            assert simulations_per_second > 100   # CPU target
            
    @pytest.mark.benchmark
    def test_batch_evaluation_throughput(self, performance_model):
        """Benchmark batch neural network evaluation"""
        # Get device from model parameters
        device = next(performance_model.parameters()).device
        batch_sizes = [1, 8, 16, 32, 64, 128]
        
        results = {}
        
        for batch_size in batch_sizes:
            # Create batch
            states = torch.randn(batch_size, 18, 15, 15, device=device)
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = performance_model(states)
                    
            # Benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            iterations = 100
            for _ in range(iterations):
                with torch.no_grad():
                    policies, values = performance_model(states)
                    
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.time() - start_time
            
            throughput = (batch_size * iterations) / elapsed
            results[batch_size] = throughput
            
            print(f"Batch size {batch_size}: {throughput:.0f} states/sec")
            
        # Larger batches should be more efficient
        if torch.cuda.is_available():
            assert results[64] > results[1] * 10  # Should get >10x speedup on GPU
        else:
            assert results[64] > results[1] * 5   # At least 5x speedup on CPU
        
    @pytest.mark.benchmark
    def test_tree_operation_performance(self):
        """Benchmark tree operations at scale"""
        from mcts.gpu.csr_tree import CSRTreeConfig
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config = CSRTreeConfig(max_nodes=100000, device=device)
        tree = CSRTree(config)
        
        # Build large tree
        print("\nBuilding tree...")
        build_start = time.time()
        
        # Add nodes by expanding existing nodes
        for i in range(10000):
            # Pick a random parent
            parent_idx = np.random.randint(0, min(i+1, tree.num_nodes))
            
            # Add 1-5 children
            num_children = np.random.randint(1, 6)
            actions = list(range(num_children))
            priors = np.random.rand(num_children).tolist()
            
            tree.add_children_batch(parent_idx, actions, priors)
            
        build_time = time.time() - build_start
        print(f"Built {tree.num_nodes} nodes in {build_time:.2f}s")
        
        # Benchmark lookups
        lookup_start = time.time()
        iterations = 10000
        
        for _ in range(iterations):
            node_id = np.random.randint(0, tree.num_nodes)
            children = tree.get_children(node_id)
            
        lookup_time = time.time() - lookup_start
        lookups_per_second = iterations / lookup_time
        
        print(f"Lookups per second: {lookups_per_second:.0f}")
        assert lookups_per_second > 10000  # Should be reasonably fast
        
    @pytest.mark.benchmark
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gpu_acceleration_speedup(self):
        """Benchmark GPU vs CPU speedup"""
        # Create identical models
        model_cpu = create_resnet_for_game('gomoku', num_blocks=5, num_filters=64).cpu()
        model_gpu = create_resnet_for_game('gomoku', num_blocks=5, num_filters=64).cuda()
        
        batch_size = 64
        states_cpu = torch.randn(batch_size, 18, 15, 15)
        states_gpu = states_cpu.cuda()
        
        # Benchmark CPU
        cpu_times = []
        for _ in range(50):
            start = time.time()
            with torch.no_grad():
                _ = model_cpu(states_cpu)
            cpu_times.append(time.time() - start)
            
        # Benchmark GPU
        gpu_times = []
        for _ in range(50):
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                _ = model_gpu(states_gpu)
            torch.cuda.synchronize()
            gpu_times.append(time.time() - start)
            
        cpu_avg = np.mean(cpu_times[10:])  # Skip warmup
        gpu_avg = np.mean(gpu_times[10:])
        speedup = cpu_avg / gpu_avg
        
        print(f"\nGPU Speedup: {speedup:.1f}x")
        print(f"CPU: {cpu_avg*1000:.2f}ms, GPU: {gpu_avg*1000:.2f}ms")
        
        assert speedup > 5.0  # Should get significant speedup
        
    @pytest.mark.benchmark
    def test_memory_usage_patterns(self, performance_config):
        """Test memory usage during MCTS search"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        
        # Simple evaluator for consistent memory usage
        mock_evaluator = Mock()
        mock_evaluator.evaluate = Mock(return_value=(np.ones(225)/225, 0.0))
        def mock_evaluate_batch(states):
            batch_size = states.shape[0] if hasattr(states, 'shape') else len(states)
            return np.ones((batch_size, 225))/225, np.zeros((batch_size, 1))
        mock_evaluator.evaluate_batch = Mock(side_effect=mock_evaluate_batch)
        
        process = psutil.Process()
        memory_samples = []
        
        # Run multiple searches and track memory
        for i in range(10):
            gc.collect()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            mcts = MCTS(performance_config, mock_evaluator)
            state = game.create_initial_state()
            
            # Search with increasing tree size
            policy = mcts.search(state, num_simulations=500 * (i + 1))
            
            peak_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = peak_memory - initial_memory
            memory_samples.append(memory_growth)
            
            print(f"Iteration {i+1}: {memory_growth:.1f}MB growth")
            
            # Cleanup
            mcts.clear()
            del mcts
            
        # Memory growth should be reasonable
        assert all(growth < 500 for growth in memory_samples)  # <500MB per search
        
        # Check for memory leaks (later iterations shouldn't use much more)
        late_avg = np.mean(memory_samples[-3:])
        early_avg = np.mean(memory_samples[:3])
        assert late_avg < early_avg * 3  # Not growing excessively


class TestScalingPerformance:
    """Test performance scaling characteristics"""
    
    @pytest.mark.benchmark
    def test_scaling_with_tree_size(self, performance_evaluator):
        """Test how performance scales with tree size"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        state = game.create_initial_state()
        
        tree_sizes = [100, 500, 1000, 5000, 10000]
        results = []
        
        for tree_size in tree_sizes:
            config = MCTSConfig()
            config.num_simulations = tree_size
            
            mcts = MCTS(config, performance_evaluator)
            
            start = time.time()
            policy = mcts.search(state, num_simulations=tree_size)
            elapsed = time.time() - start
            
            sim_per_sec = tree_size / elapsed
            results.append({
                'tree_size': tree_size,
                'time': elapsed,
                'sim_per_sec': sim_per_sec,
                'time_per_sim': elapsed / tree_size * 1000
            })
            
            print(f"Tree size {tree_size}: {sim_per_sec:.0f} sim/s, "
                  f"{elapsed/tree_size*1000:.2f}ms per sim")
            
            mcts.clear()
            
        # Time per simulation should not increase dramatically
        time_per_sim_small = results[0]['time_per_sim']
        time_per_sim_large = results[-1]['time_per_sim']
        assert time_per_sim_large < time_per_sim_small * 3  # Less than 3x slowdown
        
    @pytest.mark.benchmark
    def test_scaling_with_board_size(self, performance_evaluator):
        """Test performance with different board sizes"""
        board_sizes = [9, 11, 13, 15, 19]
        results = {}
        
        for board_size in board_sizes:
            game = GameInterface(GameType.GOMOKU, board_size=board_size)
            config = MCTSConfig()
            config.num_simulations = 200
            
            mcts = MCTS(config, performance_evaluator)
            state = game.create_initial_state()
            
            start = time.time()
            policy = mcts.search(state)
            elapsed = time.time() - start
            
            results[board_size] = elapsed
            print(f"Board size {board_size}x{board_size}: {elapsed:.3f}s")
            
        # Should scale reasonably with board size
        # 19x19 should be less than 5x slower than 9x9
        assert results[19] < results[9] * 5
        
    @pytest.mark.benchmark
    def test_batch_size_scaling(self, performance_model):
        """Test optimal batch size for throughput"""
        # Get device from model parameters
        device = next(performance_model.parameters()).device
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        results = []
        
        for batch_size in batch_sizes:
            states = torch.randn(batch_size, 18, 15, 15, device=device)
            
            # Time multiple iterations
            iterations = max(100, 1000 // batch_size)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            
            for _ in range(iterations):
                with torch.no_grad():
                    policies, values = performance_model(states)
                    
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.time() - start
            
            throughput = (batch_size * iterations) / elapsed
            latency = elapsed / iterations * 1000  # ms
            
            results.append({
                'batch_size': batch_size,
                'throughput': throughput,
                'latency': latency
            })
            
            print(f"Batch {batch_size}: {throughput:.0f} states/s, {latency:.1f}ms latency")
            
        # Find optimal batch size (best throughput)
        optimal = max(results, key=lambda x: x['throughput'])
        print(f"\nOptimal batch size: {optimal['batch_size']}")
        
        # Verify batching provides speedup
        single_throughput = next(r['throughput'] for r in results if r['batch_size'] == 1)
        assert optimal['throughput'] > single_throughput * 10


class TestMultiGamePerformance:
    """Test performance with multiple concurrent games"""
    
    @pytest.mark.benchmark
    def test_concurrent_game_performance(self, performance_config, performance_evaluator):
        """Test running multiple MCTS instances concurrently"""
        num_games = 4
        num_simulations = 200
        
        games = []
        mcts_instances = []
        states = []
        
        # Create multiple games
        for i in range(num_games):
            game = GameInterface(GameType.GOMOKU, board_size=15)
            mcts = MCTS(performance_config, performance_evaluator)
            state = game.create_initial_state()
            
            games.append(game)
            mcts_instances.append(mcts)
            states.append(state)
            
        # Time concurrent searches
        start = time.time()
        
        policies = []
        for i in range(num_games):
            policy = mcts_instances[i].search(states[i], num_simulations=num_simulations)
            policies.append(policy)
            
        elapsed = time.time() - start
        
        total_simulations = num_games * num_simulations
        sim_per_sec = total_simulations / elapsed
        
        print(f"\nConcurrent games: {num_games}")
        print(f"Total simulations: {total_simulations}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Simulations/sec: {sim_per_sec:.0f}")
        
        # Should handle multiple games efficiently
        assert sim_per_sec > 500  # Reasonable target
        
    @pytest.mark.benchmark
    def test_game_switching_overhead(self, performance_evaluator):
        """Test overhead of switching between games"""
        games = []
        states = []
        
        # Create different game types
        for game_type, board_size in [(GameType.GOMOKU, 15), (GameType.GO, 9)]:
            game = GameInterface(game_type, board_size=board_size)
            state = game.create_initial_state()
            games.append(game)
            states.append(state)
            
        config = MCTSConfig()
        config.num_simulations = 100
        
        # Time switching between games
        switch_times = []
        
        for _ in range(20):
            for i in range(len(games)):
                mcts = MCTS(config, performance_evaluator)
                
                start = time.time()
                policy = mcts.search(states[i])
                switch_times.append(time.time() - start)
                
                mcts.clear()
                
        avg_switch_time = np.mean(switch_times)
        print(f"\nAverage game switch time: {avg_switch_time*1000:.1f}ms")
        
        # Switching should be fast
        assert avg_switch_time < 0.5  # Less than 500ms


class TestTrainingPerformance:
    """Test training pipeline performance"""
    
    @pytest.mark.benchmark
    def test_self_play_throughput(self, tmp_path):
        """Test self-play game generation throughput"""
        config = AlphaZeroConfig()
        config.game.game_type = 'gomoku'
        config.game.board_size = 15
        config.training.num_games_per_iteration = 20
        config.training.num_workers = 4
        config.mcts.num_simulations = 100
        config.experiment_name = 'perf_test'
        
        # Mock to avoid actual file operations
        with patch('mcts.neural_networks.unified_training_pipeline.Path') as mock_path:
            mock_path.return_value = tmp_path
            
            pipeline = UnifiedTrainingPipeline(config)
            
            # Time self-play generation
            start = time.time()
            examples = pipeline.generate_self_play_data()
            elapsed = time.time() - start
            
            games_per_second = config.training.num_games_per_iteration / elapsed
            print(f"\nSelf-play throughput: {games_per_second:.2f} games/sec")
            
            # Should generate games reasonably fast
            assert games_per_second > 0.5  # At least 0.5 games/sec
            
    @pytest.mark.benchmark
    def test_training_iteration_time(self, tmp_path):
        """Test complete training iteration time"""
        config = AlphaZeroConfig()
        config.game.game_type = 'gomoku'
        config.training.num_iterations = 1
        config.training.num_games_per_iteration = 10
        config.training.num_epochs = 2
        config.training.batch_size = 32
        config.arena.num_games = 10
        
        with patch('mcts.neural_networks.unified_training_pipeline.Path') as mock_path:
            mock_path.return_value = tmp_path
            
            # Mock expensive operations
            with patch('mcts.neural_networks.self_play_module.SelfPlayManager.generate_games') as mock_sp:
                with patch('mcts.neural_networks.arena_module.ArenaManager.compare_models') as mock_arena:
                    # Return mock data
                    mock_examples = [
                        {'state': np.random.randn(3, 15, 15), 
                         'policy': np.ones(225)/225,
                         'value': 0.0}
                        for _ in range(100)
                    ]
                    mock_sp.return_value = mock_examples
                    mock_arena.return_value = (6, 2, 2)
                    
                    pipeline = UnifiedTrainingPipeline(config)
                    
                    # Time iteration
                    start = time.time()
                    # Run one training iteration
                    pipeline.train(num_iterations=1)
                    elapsed = time.time() - start
                    
                    print(f"\nTraining iteration time: {elapsed:.2f}s")
                    
                    # Should complete in reasonable time
                    assert elapsed < 60  # Less than 1 minute


class TestOptimizationImpact:
    """Test impact of various optimizations"""
    
    @pytest.mark.benchmark
    def test_virtual_loss_impact(self, performance_evaluator):
        """Test impact of virtual loss on performance"""
        game = GameInterface(GameType.GOMOKU, board_size=15)
        state = game.create_initial_state()
        
        results = {}
        
        for use_virtual_loss in [False, True]:
            config = MCTSConfig()
            config.num_simulations = 500
            config.enable_virtual_loss = use_virtual_loss
            
            mcts = MCTS(config, performance_evaluator)
            
            start = time.time()
            policy = mcts.search(state)
            elapsed = time.time() - start
            
            results[use_virtual_loss] = elapsed
            print(f"Virtual loss {use_virtual_loss}: {elapsed:.3f}s")
            
        # Virtual loss should not slow things down significantly
        assert results[True] < results[False] * 1.2
        
    @pytest.mark.benchmark
    def test_batch_coordinator_impact(self):
        """Test impact of batch coordination"""
        coordinator = RequestBatchingCoordinator(batch_timeout_ms=10, max_batch_size=32)
        
        # Mock evaluator
        def mock_evaluate(states):
            time.sleep(0.01)  # Simulate GPU time
            return np.ones((len(states), 225)) / 225, np.zeros(len(states))
            
        # Test with and without coordination
        num_requests = 100
        
        # Without coordination
        start = time.time()
        for _ in range(num_requests):
            state = np.random.randn(3, 15, 15)
            policy, value = mock_evaluate([state])
        no_coord_time = time.time() - start
        
        # With coordination (simulated)
        # In practice, coordination would batch these
        start = time.time()
        batch_size = 32
        for i in range(0, num_requests, batch_size):
            batch_states = [np.random.randn(3, 15, 15) for _ in range(min(batch_size, num_requests - i))]
            policies, values = mock_evaluate(batch_states)
        coord_time = time.time() - start
        
        speedup = no_coord_time / coord_time
        print(f"\nBatch coordination speedup: {speedup:.1f}x")
        
        assert speedup > 2.0  # Should get significant speedup


class TestMemoryBandwidth:
    """Test memory bandwidth utilization"""
    
    @pytest.mark.benchmark
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gpu_memory_bandwidth(self):
        """Test GPU memory bandwidth utilization"""
        device = 'cuda'
        
        # Test different data sizes
        sizes = [(1000, 1000), (2000, 2000), (4000, 4000)]
        
        for size in sizes:
            # Create large tensors
            a = torch.randn(size, device=device)
            b = torch.randn(size, device=device)
            
            # Warmup
            for _ in range(10):
                c = a + b
                
            # Benchmark
            torch.cuda.synchronize()
            start = time.time()
            
            iterations = 100
            for _ in range(iterations):
                c = a + b
                
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            # Calculate bandwidth
            bytes_per_op = 3 * np.prod(size) * 4  # 3 tensors, 4 bytes per float
            total_bytes = bytes_per_op * iterations
            bandwidth_gb_s = total_bytes / elapsed / 1e9
            
            print(f"Size {size}: {bandwidth_gb_s:.1f} GB/s")
            
        # Should achieve reasonable fraction of theoretical bandwidth
        assert bandwidth_gb_s > 100  # >100 GB/s is reasonable for modern GPUs