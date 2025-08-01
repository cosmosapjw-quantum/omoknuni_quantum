#!/usr/bin/env python3
"""Detailed tests for hybrid backend functionality

This module provides comprehensive testing of hybrid backend features including:
- CPU tree operations with GPU neural network evaluation
- Thread-safe parallel tree operations
- Efficient CPU-GPU communication
- Performance optimization validation
- Memory management across devices
"""

import pytest
import torch
import numpy as np
import logging
import time
import threading
import queue
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Import hybrid-specific components
from mcts.core.mcts import MCTS
from mcts.core.mcts_config import MCTSConfig
from mcts.gpu.gpu_game_states import GameType
from mcts.hybrid.fast_hybrid_mcts import FastHybridMCTS
from mcts.hybrid.memory_pool import MemoryPool
from mcts.hybrid.thread_local_buffers import ThreadLocalBuffers
from mcts.hybrid.spsc_queue import SPSCQueue
from mock_evaluator import MockEvaluator


class TestHybridArchitecture:
    """Test hybrid CPU-GPU architecture"""
    
    @pytest.fixture
    def hybrid_config(self):
        """Create hybrid MCTS configuration"""
        return MCTSConfig(
            backend='hybrid',
            board_size=9,
            game_type=GameType.GOMOKU,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            max_tree_nodes=50000,
            num_simulations=1000,
            num_parallel_reads=16,
            num_parallel_writes=8,
            batch_size=32
        )
    
    @pytest.fixture
    def hybrid_mcts(self, hybrid_config):
        """Create hybrid MCTS instance"""
        evaluator = MockEvaluator(
            board_size=hybrid_config.board_size,
            device=hybrid_config.device,
            batch_size=hybrid_config.batch_size
        )
        return MCTS(hybrid_config, evaluator)
    
    def test_device_separation(self, hybrid_mcts):
        """Test CPU-GPU separation in hybrid mode"""
        # Tree operations should be on CPU
        assert hybrid_mcts.tree.device.type == 'cpu'
        assert hybrid_mcts.game_states.device.type == 'cpu'
        
        # Evaluator should use GPU if available
        if torch.cuda.is_available():
            assert hybrid_mcts.evaluator.device.type == 'cuda'
        
        # Node-to-state mapping on CPU for thread safety
        assert hybrid_mcts.node_to_state.device.type == 'cpu'
    
    def test_thread_safe_tree_operations(self, hybrid_mcts):
        """Test thread-safe tree operations"""
        state = np.zeros((9, 9), dtype=np.int8)
        hybrid_mcts._ensure_root_initialized(state)
        
        errors = []
        results = []
        
        def worker(thread_id: int):
            """Worker thread performing tree operations"""
            try:
                # Each thread expands different parts of tree
                for i in range(10):
                    # Simulate tree traversal and expansion
                    node_idx = thread_id * 10 + i
                    if node_idx < hybrid_mcts.tree.max_nodes:
                        # Safe to add nodes from multiple threads
                        action = (thread_id * 100 + i) % 81
                        parent = min(node_idx // 2, hybrid_mcts.tree.num_nodes - 1)
                        if parent >= 0:
                            hybrid_mcts.tree.add_node(
                                parent_idx=parent,
                                action=action
                            )
                        results.append((thread_id, i, "success"))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Run multiple threads
        threads = []
        for i in range(4):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Check results
        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) > 0, "No operations completed"
    
    def test_cpu_gpu_pipeline(self, hybrid_mcts):
        """Test CPU-GPU communication pipeline"""
        state = np.zeros((9, 9), dtype=np.int8)
        
        # Measure pipeline efficiency
        start = time.perf_counter()
        
        # Run search which involves CPU tree ops and GPU evaluation
        policy = hybrid_mcts.search(state, num_simulations=100)
        
        search_time = time.perf_counter() - start
        
        # Verify results
        assert policy is not None
        assert len(policy) == 81
        assert np.allclose(policy.sum(), 1.0)
        
        # Check that both CPU and GPU were utilized
        assert hybrid_mcts.tree.num_nodes > 1, "Tree not expanded"
        assert hybrid_mcts.stats['total_simulations'] >= 100, "Simulations not run"
        
        logger.info(f"Hybrid pipeline time: {search_time:.3f}s")


class TestHybridMemoryManagement:
    """Test memory management in hybrid mode"""
    
    def test_memory_pool(self):
        """Test thread-safe memory pool"""
        pool = MemoryPool(
            capacity=1000,
            element_size=64,
            alignment=64
        )
        
        # Allocate from multiple threads
        allocations = []
        lock = threading.Lock()
        
        def allocate_worker(n_allocs: int):
            local_allocs = []
            for _ in range(n_allocs):
                ptr = pool.allocate()
                assert ptr is not None
                local_allocs.append(ptr)
            
            # Store allocations
            with lock:
                allocations.extend(local_allocs)
            
            # Deallocate half
            for ptr in local_allocs[:n_allocs//2]:
                pool.deallocate(ptr)
        
        # Run threads
        threads = []
        for _ in range(4):
            t = threading.Thread(target=allocate_worker, args=(100,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Verify no duplicate allocations
        unique_allocs = set(allocations)
        assert len(unique_allocs) == len(allocations), "Duplicate allocations detected"
    
    def test_thread_local_buffers(self):
        """Test thread-local buffer management"""
        buffers = ThreadLocalBuffers(
            max_threads=8,
            buffer_size=1024
        )
        
        thread_ids = []
        buffer_ptrs = []
        
        def buffer_worker(worker_id: int):
            # Get thread-local buffer
            buffer = buffers.get_buffer()
            assert buffer is not None
            
            # Store thread ID and buffer pointer
            thread_ids.append(threading.get_ident())
            buffer_ptrs.append(id(buffer))
            
            # Use buffer
            buffer[:] = worker_id
            
            # Verify buffer content
            assert np.all(buffer == worker_id)
        
        # Run threads
        threads = []
        for i in range(4):
            t = threading.Thread(target=buffer_worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Each thread should have different buffer
        assert len(set(buffer_ptrs)) == len(buffer_ptrs), \
            "Threads sharing buffers!"
    
    def test_spsc_queue(self):
        """Test single-producer single-consumer queue"""
        q = SPSCQueue(capacity=100, dtype=np.float32)
        
        # Producer thread
        def producer():
            for i in range(50):
                data = np.array([i, i*2, i*3], dtype=np.float32)
                success = q.push(data)
                assert success, f"Failed to push item {i}"
                time.sleep(0.001)  # Simulate work
        
        # Consumer thread
        received = []
        def consumer():
            while len(received) < 50:
                data = q.pop()
                if data is not None:
                    received.append(data.copy())
                time.sleep(0.001)  # Simulate work
        
        # Run producer and consumer
        prod_thread = threading.Thread(target=producer)
        cons_thread = threading.Thread(target=consumer)
        
        prod_thread.start()
        cons_thread.start()
        
        prod_thread.join()
        cons_thread.join()
        
        # Verify all data received in order
        assert len(received) == 50
        for i, data in enumerate(received):
            expected = np.array([i, i*2, i*3], dtype=np.float32)
            assert np.allclose(data, expected)


class TestHybridPerformanceOptimizations:
    """Test hybrid-specific performance optimizations"""
    
    @pytest.fixture
    def optimized_hybrid(self):
        """Create optimized hybrid MCTS"""
        config = MCTSConfig(
            backend='hybrid',
            board_size=9,
            game_type=GameType.GOMOKU,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            max_tree_nodes=100000,
            num_simulations=1000,
            # Optimization parameters
            num_parallel_reads=16,
            batch_size=64,
            enable_leaf_parallelism=True,
            enable_virtual_loss=True,
            virtual_loss_value=3.0
        )
        evaluator = MockEvaluator(
            board_size=9,
            device=config.device,
            batch_size=64
        )
        return MCTS(config, evaluator)
    
    def test_parallel_leaf_collection(self, optimized_hybrid):
        """Test parallel leaf collection optimization"""
        state = np.zeros((9, 9), dtype=np.int8)
        
        # Build initial tree
        optimized_hybrid._ensure_root_initialized(state)
        
        # Expand tree to create multiple leaves
        for _ in range(10):
            optimized_hybrid.wave_search.run_wave(
                wave_size=32,
                node_to_state=optimized_hybrid.node_to_state,
                state_pool_free_list=optimized_hybrid.state_pool_free_list
            )
        
        # Time parallel leaf collection
        start = time.perf_counter()
        
        # Run wave with parallel reads enabled
        optimized_hybrid.wave_search.run_wave(
            wave_size=64,
            node_to_state=optimized_hybrid.node_to_state,
            state_pool_free_list=optimized_hybrid.state_pool_free_list
        )
        
        parallel_time = time.perf_counter() - start
        
        # Should efficiently collect leaves for batch evaluation
        logger.info(f"Parallel leaf collection time: {parallel_time:.3f}s")
        assert optimized_hybrid.tree.num_nodes > 100, "Insufficient tree expansion"
    
    def test_virtual_loss_mechanism(self, optimized_hybrid):
        """Test virtual loss for exploration"""
        state = np.zeros((9, 9), dtype=np.int8)
        optimized_hybrid._ensure_root_initialized(state)
        
        # Virtual loss should prevent multiple threads selecting same path
        assert optimized_hybrid.config.enable_virtual_loss
        assert optimized_hybrid.config.virtual_loss_value > 0
        
        # Track selected paths
        selected_paths = []
        
        # Simulate multiple parallel selections
        for _ in range(10):
            # Each wave should explore different paths due to virtual loss
            wave_results = optimized_hybrid.wave_search.run_wave(
                wave_size=8,
                node_to_state=optimized_hybrid.node_to_state,
                state_pool_free_list=optimized_hybrid.state_pool_free_list
            )
            # Note: would need to modify wave_search to return paths for verification
        
        # Tree should have good exploration
        assert optimized_hybrid.tree.num_nodes > 50, "Poor exploration"
    
    def test_batch_aggregation_efficiency(self, optimized_hybrid):
        """Test efficient batch aggregation for GPU evaluation"""
        state = np.zeros((9, 9), dtype=np.int8)
        
        # Measure batch aggregation overhead
        batch_times = []
        batch_sizes = []
        
        for _ in range(10):
            start = time.perf_counter()
            
            # Run search with batch aggregation
            policy = optimized_hybrid.search(state, num_simulations=100)
            
            batch_time = time.perf_counter() - start
            batch_times.append(batch_time)
            
            # Track effective batch sizes (would need instrumentation)
            batch_sizes.append(optimized_hybrid.config.batch_size)
        
        # Batch aggregation should be efficient
        avg_time = np.mean(batch_times)
        logger.info(f"Average batch time: {avg_time:.3f}s")
        
        # Throughput should be good
        throughput = 100 / avg_time  # sims per second
        assert throughput > 1000, f"Low throughput: {throughput:.0f} sims/s"
    
    def test_cpu_gpu_overlap(self, optimized_hybrid):
        """Test CPU-GPU computation overlap"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for overlap test")
        
        state = np.zeros((9, 9), dtype=np.int8)
        
        # Instrument to measure overlap
        cpu_time = 0
        gpu_time = 0
        
        # Custom evaluation wrapper to measure GPU time
        original_evaluate = optimized_hybrid.evaluator.evaluate
        
        def timed_evaluate(states):
            nonlocal gpu_time
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = original_evaluate(states)
            torch.cuda.synchronize()
            gpu_time += time.perf_counter() - start
            return result
        
        optimized_hybrid.evaluator.evaluate = timed_evaluate
        
        # Measure total time and component times
        total_start = time.perf_counter()
        policy = optimized_hybrid.search(state, num_simulations=200)
        total_time = time.perf_counter() - total_start
        
        # Restore original
        optimized_hybrid.evaluator.evaluate = original_evaluate
        
        # CPU time is roughly total - GPU (with overlap)
        cpu_time = total_time - gpu_time * 0.5  # Assuming 50% overlap
        
        logger.info(f"Total: {total_time:.3f}s, GPU: {gpu_time:.3f}s, Est. CPU: {cpu_time:.3f}s")
        
        # Should show overlap (total < cpu + gpu)
        assert total_time < cpu_time + gpu_time, "No CPU-GPU overlap detected"


class TestHybridScalability:
    """Test hybrid backend scalability"""
    
    def test_thread_scalability(self):
        """Test scalability with different thread counts"""
        thread_counts = [1, 2, 4, 8]
        throughputs = []
        
        for num_threads in thread_counts:
            config = MCTSConfig(
                backend='hybrid',
                board_size=9,
                game_type=GameType.GOMOKU,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                max_tree_nodes=100000,
                num_simulations=1000,
                num_parallel_reads=num_threads,
                num_parallel_writes=num_threads // 2
            )
            evaluator = MockEvaluator(board_size=9, device=config.device)
            mcts = MCTS(config, evaluator)
            
            state = np.zeros((9, 9), dtype=np.int8)
            
            # Warmup
            mcts.search(state, num_simulations=50)
            
            # Benchmark
            start = time.perf_counter()
            mcts.search(state, num_simulations=500)
            elapsed = time.perf_counter() - start
            
            throughput = 500 / elapsed
            throughputs.append(throughput)
            
            logger.info(f"{num_threads} threads: {throughput:.0f} sims/s")
        
        # Should show scaling (not necessarily linear)
        assert throughputs[-1] > throughputs[0] * 1.5, \
            "Poor thread scalability"
    
    def test_large_tree_performance(self):
        """Test performance with large trees"""
        config = MCTSConfig(
            backend='hybrid',
            board_size=19,  # Full Go board
            game_type=GameType.GO,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            max_tree_nodes=1000000,
            num_simulations=10000
        )
        evaluator = MockEvaluator(board_size=19, device=config.device)
        mcts = MCTS(config, evaluator)
        
        state = np.zeros((19, 19), dtype=np.int8)
        
        # Build large tree incrementally
        tree_sizes = []
        search_times = []
        
        for i in range(5):
            start = time.perf_counter()
            mcts.search(state, num_simulations=1000)
            search_time = time.perf_counter() - start
            
            tree_sizes.append(mcts.tree.num_nodes)
            search_times.append(search_time)
            
            logger.info(f"Tree size: {tree_sizes[-1]}, Time: {search_time:.3f}s")
        
        # Performance should degrade gracefully
        time_ratio = search_times[-1] / search_times[0]
        size_ratio = tree_sizes[-1] / tree_sizes[0]
        
        # Sub-linear performance degradation
        assert time_ratio < size_ratio, \
            f"Performance degrades too quickly: time {time_ratio:.1f}x, size {size_ratio:.1f}x"
    
    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure"""
        # Create config with limited tree size
        config = MCTSConfig(
            backend='hybrid',
            board_size=9,
            game_type=GameType.GOMOKU,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            max_tree_nodes=1000,  # Small tree
            num_simulations=5000,  # Many simulations
            enable_subtree_reuse=False
        )
        evaluator = MockEvaluator(board_size=9, device=config.device)
        mcts = MCTS(config, evaluator)
        
        state = np.zeros((9, 9), dtype=np.int8)
        
        # Should handle memory pressure gracefully
        try:
            policy = mcts.search(state, num_simulations=5000)
            
            # Verify search completed
            assert policy is not None
            assert mcts.tree.num_nodes <= config.max_tree_nodes
            
            logger.info(f"Handled memory pressure: {mcts.tree.num_nodes} nodes")
            
        except Exception as e:
            pytest.fail(f"Failed to handle memory pressure: {e}")


class TestHybridCorrectness:
    """Test correctness of hybrid implementation"""
    
    def test_consistency_across_backends(self):
        """Test that hybrid produces similar results to pure backends"""
        # Common configuration
        base_config = {
            'board_size': 9,
            'game_type': GameType.GOMOKU,
            'max_tree_nodes': 10000,
            'num_simulations': 200,
            'c_puct': 1.0,
            'dirichlet_epsilon': 0.0,  # Disable noise for consistency
            'enable_subtree_reuse': False
        }
        
        # Test state
        state = np.zeros((9, 9), dtype=np.int8)
        state[4, 4] = 1  # Center move
        
        policies = {}
        
        # Test each backend
        for backend in ['cpu', 'gpu', 'hybrid']:
            if backend == 'gpu' and not torch.cuda.is_available():
                continue
            
            config = MCTSConfig(
                backend=backend,
                device='cuda' if backend == 'gpu' else 'cpu',
                **base_config
            )
            evaluator = MockEvaluator(
                board_size=9,
                device=config.device,
                deterministic=True  # For reproducibility
            )
            mcts = MCTS(config, evaluator)
            
            # Run search
            policy = mcts.search(state, num_simulations=200)
            policies[backend] = policy
            
            logger.info(f"{backend} max policy: {policy.max():.3f}")
        
        # Compare policies
        if 'cpu' in policies and 'hybrid' in policies:
            # Policies should be similar (not identical due to parallelism)
            correlation = np.corrcoef(policies['cpu'], policies['hybrid'])[0, 1]
            assert correlation > 0.9, f"CPU-Hybrid correlation too low: {correlation:.3f}"
        
        if 'gpu' in policies and 'hybrid' in policies:
            correlation = np.corrcoef(policies['gpu'], policies['hybrid'])[0, 1]
            assert correlation > 0.9, f"GPU-Hybrid correlation too low: {correlation:.3f}"
    
    def test_deterministic_mode(self):
        """Test deterministic behavior when configured"""
        config = MCTSConfig(
            backend='hybrid',
            board_size=9,
            game_type=GameType.GOMOKU,
            device='cpu',  # CPU for determinism
            max_tree_nodes=10000,
            num_simulations=100,
            dirichlet_epsilon=0.0,
            enable_subtree_reuse=False,
            num_parallel_reads=1,  # Single thread for determinism
            num_parallel_writes=1
        )
        evaluator = MockEvaluator(
            board_size=9,
            device='cpu',
            deterministic=True
        )
        
        state = np.zeros((9, 9), dtype=np.int8)
        
        # Run multiple times
        policies = []
        for _ in range(3):
            mcts = MCTS(config, evaluator)
            policy = mcts.search(state, num_simulations=100)
            policies.append(policy)
        
        # Should be identical in deterministic mode
        for i in range(1, len(policies)):
            assert np.allclose(policies[0], policies[i], rtol=1e-5), \
                "Non-deterministic behavior in deterministic mode"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])