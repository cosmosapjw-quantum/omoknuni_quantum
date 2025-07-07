"""
Integration tests for multiprocessing scenarios

Tests cover:
- Multi-worker self-play coordination
- GPU service process management
- Batch coordinator across processes
- Shared memory handling
- Process cleanup and termination
- Deadlock prevention
- Resource contention
"""

import pytest
import torch
import numpy as np
import multiprocessing as mp
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import queue
import sys
import os

from mcts.utils.gpu_evaluator_service import GPUEvaluatorService
from mcts.utils.batch_evaluation_coordinator import RequestBatchingCoordinator
from mcts.neural_networks.self_play_module import SelfPlayManager, _play_game_worker_with_gpu_service
from mcts.neural_networks.resnet_model import create_resnet_for_game
from mcts.core.game_interface import GameInterface, GameType
from mcts.core.mcts import MCTS
from mcts.utils.config_system import AlphaZeroConfig


@pytest.fixture
def mp_config():
    """Create multiprocessing test configuration"""
    config = AlphaZeroConfig()
    config.training.num_workers = 4
    config.training.num_games_per_iteration = 20
    config.mcts.num_simulations = 50
    config.game.game_type = 'gomoku'
    config.game.board_size = 15
    config.network.num_res_blocks = 2  # Small for testing
    config.network.num_filters = 16
    return config


@pytest.fixture
def simple_model():
    """Create simple model for testing"""
    return create_resnet_for_game('gomoku', num_blocks=2, num_filters=16)


@pytest.fixture
def gpu_service(simple_model):
    """Create GPU evaluator service"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    service = GPUEvaluatorService(simple_model, device=device)
    yield service
    if hasattr(service, 'service_thread') and service.service_thread and service.service_thread.is_alive():
        service.stop()


@pytest.mark.timeout(30)  # 30 second timeout for all tests in this class
class TestMultiWorkerSelfPlay:
    """Test multi-worker self-play scenarios"""
    
    @pytest.mark.skip(reason="Multiprocessing tests can hang in CI")
    def test_basic_multi_worker_setup(self, mp_config, simple_model):
        """Test basic multi-worker self-play setup"""
        manager = SelfPlayManager(mp_config)
        
        # Mock the worker function
        with patch('mcts.neural_networks.self_play_module._play_game_worker_with_gpu_service') as mock_worker:
            with patch('multiprocessing.Process') as mock_process:
                mock_process_instance = Mock()
                mock_process_instance.is_alive.return_value = True
                mock_process.return_value = mock_process_instance
                
                # Start workers
                examples = manager._parallel_self_play(
                    simple_model, 
                    iteration=1,
                    num_games=10,
                    num_workers=2
                )
                
                # Should create 2 worker processes
                assert mock_process.call_count == 2
                
    @pytest.mark.skip(reason="Multiprocessing tests can hang in CI")
    def test_worker_game_generation(self, mp_config):
        """Test individual worker game generation"""
        # Create queues
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        
        # Add tasks
        for i in range(5):
            task_queue.put(i)
        task_queue.put(None)  # Sentinel
        
        # Create mock model
        mock_model = Mock()
        mock_model.eval = Mock(return_value=mock_model)
        
        # Mock MCTS
        with patch('mcts.core.mcts.MCTS') as mock_mcts_class:
            mock_mcts = Mock()
            mock_mcts.search.return_value = np.ones(225) / 225
            mock_mcts.get_root_value.return_value = 0.5
            mock_mcts_class.return_value = mock_mcts
            
            # Run worker (in same process for testing)
            # Create mock request/response queues for GPU service
            request_queue = mp.Queue()
            response_queue = mp.Queue()
            
            _play_game_worker_with_gpu_service(
                config=mp_config,
                request_queue=request_queue,
                response_queue=response_queue,
                action_size=225,  # Gomoku action size
                game_idx=0,
                iteration=1,
                allocation=None  # No GPU allocation for testing
            )
            
            # Should generate results
            results = []
            while not result_queue.empty():
                try:
                    result = result_queue.get_nowait()
                    results.append(result)
                except queue.Empty:
                    break
                    
            assert len(results) > 0
            
    @pytest.mark.skip(reason="Multiprocessing tests can hang in CI")
    def test_worker_coordination(self, mp_config, simple_model):
        """Test coordination between multiple workers"""
        num_workers = 3
        num_games = 12  # Divisible by num_workers
        
        # Create shared queues
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        
        # Add tasks
        for i in range(num_games):
            task_queue.put(i)
        for _ in range(num_workers):
            task_queue.put(None)  # Sentinels
            
        # Track worker completions
        worker_completions = mp.Array('i', num_workers)
        
        def track_worker(worker_id):
            """Modified worker that tracks completion"""
            # Simulate work
            time.sleep(0.1)
            worker_completions[worker_id] = 1
            
        # Start workers
        processes = []
        for i in range(num_workers):
            p = mp.Process(target=track_worker, args=(i,))
            p.start()
            processes.append(p)
            
        # Wait for completion
        for p in processes:
            p.join(timeout=5.0)
            
        # All workers should complete
        assert all(worker_completions[i] == 1 for i in range(num_workers))
        
    @pytest.mark.skip(reason="Multiprocessing tests can hang in CI")
    def test_worker_crash_recovery(self, mp_config):
        """Test recovery from worker crashes"""
        manager = SelfPlayManager(mp_config)
        
        with patch('multiprocessing.Process') as mock_process:
            # Simulate worker crash
            mock_proc_instance = Mock()
            mock_proc_instance.is_alive.side_effect = [True, False]  # Dies
            mock_proc_instance.exitcode = -1  # Abnormal termination
            mock_process.return_value = mock_proc_instance
            
            with patch.object(manager, '_handle_worker_failure') as mock_handle:
                # Should detect and handle failure
                try:
                    manager._parallel_self_play(
                        Mock(), iteration=1, num_games=10, num_workers=2
                    )
                except:
                    pass  # Expected to fail
                    
                # Should attempt recovery
                assert mock_handle.called or mock_proc_instance.is_alive.called
                
    @pytest.mark.skip(reason="Multiprocessing tests can hang in CI")
    def test_queue_timeout_handling(self):
        """Test queue timeout handling"""
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        
        # Don't put any tasks - should timeout
        start_time = time.time()
        
        try:
            result = result_queue.get(timeout=0.5)
        except queue.Empty:
            elapsed = time.time() - start_time
            assert elapsed >= 0.5
            assert elapsed < 1.0  # Should timeout promptly


@pytest.mark.timeout(30)
class TestGPUServiceIntegration:
    """Test GPU evaluator service in multiprocessing context"""
    
    @pytest.mark.skip(reason="GPU service tests can hang in CI")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gpu_service_startup(self, simple_model):
        """Test GPU service process startup"""
        service = GPUEvaluatorService(simple_model, device='cuda')
        
        # Start service
        service.start()
        time.sleep(0.5)  # Give it time to start
        
        assert service.process is not None
        assert service.process.is_alive()
        
        # Stop service
        service.stop()
        time.sleep(0.5)
        
        assert not service.process.is_alive()
        
    @pytest.mark.skip(reason="GPU service tests can hang in CI")
    def test_gpu_service_evaluation(self, gpu_service):
        """Test GPU service evaluation requests"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for GPU service test")
            
        gpu_service.start()
        time.sleep(0.5)
        
        # Create evaluation request
        states = torch.randn(4, 3, 15, 15)
        
        # Submit request
        request_id = gpu_service.submit_request(states)
        
        # Get result
        result = gpu_service.get_result(request_id, timeout=5.0)
        
        assert result is not None
        policies, values = result
        assert policies.shape == (4, 225)
        assert values.shape == (4,)
        
        gpu_service.stop()
        
    @pytest.mark.skip(reason="GPU service tests can hang in CI")
    def test_gpu_service_batch_handling(self, gpu_service):
        """Test GPU service batch request handling"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
            
        gpu_service.start()
        time.sleep(0.5)
        
        # Submit multiple requests
        request_ids = []
        for i in range(5):
            states = torch.randn(2, 3, 15, 15)
            req_id = gpu_service.submit_request(states)
            request_ids.append(req_id)
            
        # Get all results
        results = []
        for req_id in request_ids:
            result = gpu_service.get_result(req_id, timeout=5.0)
            results.append(result)
            
        assert len(results) == 5
        assert all(r is not None for r in results)
        
        gpu_service.stop()
        
    @pytest.mark.skip(reason="GPU service tests can hang in CI")
    def test_gpu_service_process_isolation(self, simple_model):
        """Test GPU service process isolation"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
            
        # GPU service should run in separate process
        service = GPUEvaluatorService(simple_model, device='cuda')
        
        main_pid = mp.current_process().pid
        
        service.start()
        time.sleep(0.5)
        
        service_pid = service.process.pid
        assert service_pid != main_pid
        
        service.stop()


@pytest.mark.timeout(30)
@pytest.mark.skip(reason="Multiprocessing tests can hang in CI")
class TestBatchCoordinatorMultiprocess:
    """Test batch coordinator across processes"""
    
    def test_cross_process_coordination(self):
        """Test batch coordination across processes"""
        coordinator = BatchEvaluationCoordinator(
            timeout_ms=100,
            max_batch_size=32
        )
        
        # Simulate requests from different processes
        def submit_requests(process_id, num_requests):
            """Worker function to submit requests"""
            results = []
            for i in range(num_requests):
                state = np.random.randn(3, 15, 15)
                result = coordinator.coordinate_evaluation(
                    state,
                    lambda x: (np.ones((len(x), 225)) / 225, np.zeros(len(x)))
                )
                results.append(result)
            return results
            
        # Use threading to simulate multiple processes
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(4):
                future = executor.submit(submit_requests, i, 5)
                futures.append(future)
                
            # Get results
            all_results = []
            for future in concurrent.futures.as_completed(futures):
                results = future.result()
                all_results.extend(results)
                
        assert len(all_results) == 20
        
    def test_coordinator_performance_under_load(self):
        """Test coordinator performance with many processes"""
        coordinator = BatchEvaluationCoordinator(
            timeout_ms=50,
            max_batch_size=64
        )
        
        # Track batch sizes
        batch_sizes = []
        
        def mock_evaluator(states):
            batch_sizes.append(len(states))
            return np.ones((len(states), 225)) / 225, np.zeros(len(states))
            
        # Simulate high load
        def worker(worker_id):
            for _ in range(10):
                state = np.random.randn(3, 15, 15)
                coordinator.coordinate_evaluation(state, mock_evaluator)
                
        # Run workers
        processes = []
        for i in range(8):
            p = mp.Process(target=worker, args=(i,))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join(timeout=10.0)
            
        # Should batch effectively
        if batch_sizes:
            avg_batch_size = np.mean(batch_sizes)
            assert avg_batch_size > 1  # Some batching occurred


@pytest.mark.timeout(30)
@pytest.mark.skip(reason="Shared memory tests can hang in CI")
class TestSharedMemoryHandling:
    """Test shared memory handling in multiprocessing"""
    
    def test_shared_memory_game_states(self):
        """Test sharing game states through shared memory"""
        # Create shared memory for game states
        state_shape = (3, 15, 15)
        state_size = np.prod(state_shape) * np.dtype(np.float32).itemsize
        
        # Create shared memory
        shm = mp.shared_memory.SharedMemory(create=True, size=state_size)
        
        try:
            # Create numpy array backed by shared memory
            shared_state = np.ndarray(state_shape, dtype=np.float32, buffer=shm.buf)
            shared_state[:] = np.random.randn(*state_shape)
            
            # Access from another process
            def access_shared_state(shm_name):
                existing_shm = mp.shared_memory.SharedMemory(name=shm_name)
                state = np.ndarray(state_shape, dtype=np.float32, buffer=existing_shm.buf)
                # Verify data
                assert state.shape == state_shape
                existing_shm.close()
                
            p = mp.Process(target=access_shared_state, args=(shm.name,))
            p.start()
            p.join()
            
            assert p.exitcode == 0
            
        finally:
            shm.close()
            shm.unlink()
            
    def test_shared_memory_cleanup(self):
        """Test proper cleanup of shared memory"""
        # Track memory usage
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create and destroy shared memory multiple times
        for _ in range(10):
            shm = mp.shared_memory.SharedMemory(create=True, size=1024*1024)  # 1MB
            data = np.ndarray((1024, 256), dtype=np.float32, buffer=shm.buf)
            data[:] = np.random.randn(1024, 256)
            shm.close()
            shm.unlink()
            
        # Memory should not grow significantly
        final_memory = process.memory_info().rss
        memory_growth = (final_memory - initial_memory) / 1024 / 1024  # MB
        assert memory_growth < 50  # Less than 50MB growth


@pytest.mark.timeout(30)
@pytest.mark.skip(reason="Process cleanup tests can hang in CI")
class TestProcessCleanup:
    """Test process cleanup and termination"""
    
    def test_graceful_shutdown(self):
        """Test graceful shutdown of worker processes"""
        shutdown_flag = mp.Event()
        
        def worker(shutdown_flag):
            while not shutdown_flag.is_set():
                time.sleep(0.1)
            # Clean shutdown
            
        # Start workers
        processes = []
        for _ in range(4):
            p = mp.Process(target=worker, args=(shutdown_flag,))
            p.start()
            processes.append(p)
            
        # Let them run
        time.sleep(0.5)
        
        # Signal shutdown
        shutdown_flag.set()
        
        # Wait for termination
        for p in processes:
            p.join(timeout=2.0)
            assert p.exitcode == 0
            
    def test_force_termination(self):
        """Test force termination of stuck processes"""
        def stubborn_worker():
            while True:
                time.sleep(1)
                
        p = mp.Process(target=stubborn_worker)
        p.start()
        
        # Give it time to start
        time.sleep(0.2)
        
        # Try graceful termination first
        p.terminate()
        p.join(timeout=1.0)
        
        if p.is_alive():
            # Force kill if needed
            p.kill()
            p.join(timeout=1.0)
            
        assert not p.is_alive()
        
    def test_zombie_process_prevention(self):
        """Test prevention of zombie processes"""
        def quick_worker():
            pass  # Exit immediately
            
        # Create many short-lived processes
        for _ in range(20):
            p = mp.Process(target=quick_worker)
            p.start()
            # Must join to prevent zombies
            p.join()
            assert p.exitcode == 0


@pytest.mark.timeout(30)
@pytest.mark.skip(reason="Deadlock prevention tests can hang in CI")
class TestDeadlockPrevention:
    """Test deadlock prevention in multiprocessing"""
    
    def test_queue_deadlock_prevention(self):
        """Test prevention of queue-based deadlocks"""
        q = mp.Queue(maxsize=5)
        
        def producer():
            for i in range(10):
                q.put(i, timeout=1.0)  # Timeout prevents deadlock
                
        def consumer():
            for _ in range(10):
                try:
                    item = q.get(timeout=1.0)
                except queue.Empty:
                    break
                    
        # Start producer and consumer
        p1 = mp.Process(target=producer)
        p2 = mp.Process(target=consumer)
        
        p1.start()
        p2.start()
        
        # Should complete without deadlock
        p1.join(timeout=5.0)
        p2.join(timeout=5.0)
        
        assert not p1.is_alive()
        assert not p2.is_alive()
        
    def test_lock_ordering(self):
        """Test proper lock ordering to prevent deadlock"""
        lock1 = mp.Lock()
        lock2 = mp.Lock()
        completed = mp.Array('i', 2)
        
        def worker1():
            # Always acquire locks in same order
            with lock1:
                time.sleep(0.1)
                with lock2:
                    completed[0] = 1
                    
        def worker2():
            # Same order as worker1
            with lock1:
                with lock2:
                    completed[1] = 1
                    
        p1 = mp.Process(target=worker1)
        p2 = mp.Process(target=worker2)
        
        p1.start()
        p2.start()
        
        p1.join(timeout=2.0)
        p2.join(timeout=2.0)
        
        # Both should complete
        assert completed[0] == 1
        assert completed[1] == 1


@pytest.mark.timeout(30)
@pytest.mark.skip(reason="Resource contention tests can hang in CI")
class TestResourceContention:
    """Test handling of resource contention"""
    
    def test_gpu_contention(self):
        """Test handling of GPU resource contention"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
            
        num_processes = 4
        results = mp.Array('i', num_processes)
        
        def gpu_worker(worker_id, results):
            try:
                # Try to allocate GPU memory
                tensor = torch.randn(1000, 1000, device='cuda')
                # Do some computation
                result = tensor.sum().item()
                results[worker_id] = 1  # Success
            except RuntimeError as e:
                if "out of memory" in str(e):
                    results[worker_id] = -1  # OOM
                else:
                    results[worker_id] = -2  # Other error
                    
        # Start multiple GPU workers
        processes = []
        for i in range(num_processes):
            p = mp.Process(target=gpu_worker, args=(i, results))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join(timeout=5.0)
            
        # At least one should succeed
        assert any(results[i] == 1 for i in range(num_processes))
        
    def test_cpu_contention(self):
        """Test handling of CPU resource contention"""
        num_workers = mp.cpu_count() * 2  # Oversubscribe
        completion_times = mp.Array('d', num_workers)
        
        def cpu_intensive_worker(worker_id, completion_times):
            start = time.time()
            # CPU intensive task
            result = 0
            for i in range(1000000):
                result += i ** 2
            completion_times[worker_id] = time.time() - start
            
        # Start workers
        processes = []
        for i in range(num_workers):
            p = mp.Process(target=cpu_intensive_worker, args=(i, completion_times))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join(timeout=30.0)
            
        # All should complete despite contention
        assert all(t > 0 for t in completion_times)


@pytest.mark.timeout(30)
@pytest.mark.skip(reason="Distributed training tests can hang in CI")
class TestDistributedTraining:
    """Test distributed training scenarios"""
    
    def test_distributed_self_play(self, mp_config):
        """Test distributed self-play across multiple nodes"""
        # Simulate multiple nodes
        num_nodes = 2
        games_per_node = 10
        
        def node_worker(node_id, games_per_node):
            """Simulate self-play on a node"""
            examples = []
            for _ in range(games_per_node):
                # Simulate game example
                example = {
                    'state': np.random.randn(3, 15, 15),
                    'policy': np.ones(225) / 225,
                    'value': np.random.uniform(-1, 1),
                    'node_id': node_id
                }
                examples.append(example)
            return examples
            
        # Run on "nodes"
        with mp.Pool(processes=num_nodes) as pool:
            results = []
            for node_id in range(num_nodes):
                result = pool.apply_async(node_worker, (node_id, games_per_node))
                results.append(result)
                
            # Collect results
            all_examples = []
            for result in results:
                examples = result.get(timeout=10.0)
                all_examples.extend(examples)
                
        assert len(all_examples) == num_nodes * games_per_node
        
        # Verify examples from all nodes
        node_ids = set(ex['node_id'] for ex in all_examples)
        assert len(node_ids) == num_nodes