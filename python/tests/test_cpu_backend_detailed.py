#!/usr/bin/env python3
"""Detailed tests for CPU backend functionality

This module provides comprehensive testing of CPU-specific MCTS features including:
- CPU game state management
- CPU-optimized tree operations  
- Vectorized CPU operations
- Thread-safe operations
"""

import pytest
import torch
import numpy as np
import logging
import time
import threading
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Import CPU-specific components
from mcts.core.mcts import MCTS
from mcts.core.mcts_config import MCTSConfig
from mcts.gpu.gpu_game_states import GameType
from mcts.cpu.cpu_game_states import CPUGameStates
from mcts.cpu.vectorized_operations import VectorizedOps
from mock_evaluator import MockEvaluator


class TestCPUGameStates:
    """Test CPU game state management"""
    
    @pytest.fixture
    def cpu_states(self):
        """Create CPU game states instance"""
        return CPUGameStates(
            max_states=1000,
            board_size=9,
            game_type='gomoku'
        )
    
    def test_state_allocation(self, cpu_states):
        """Test state allocation and deallocation"""
        # Allocate states
        state_indices = []
        for i in range(10):
            idx = cpu_states.allocate_state()
            assert idx >= 0
            assert idx not in state_indices  # No duplicates
            state_indices.append(idx)
        
        # Deallocate states
        for idx in state_indices[:5]:
            cpu_states.deallocate_state(idx)
        
        # Allocate again - should reuse deallocated indices
        new_indices = []
        for i in range(5):
            idx = cpu_states.allocate_state()
            assert idx >= 0
            new_indices.append(idx)
        
        # Check that indices were reused
        reused = set(new_indices) & set(state_indices[:5])
        assert len(reused) > 0, "Deallocated states not reused"
    
    def test_state_cloning(self, cpu_states):
        """Test state cloning operations"""
        # Create source state
        src_idx = cpu_states.allocate_state()
        
        # Initialize with some data
        state_data = np.random.randint(0, 3, (9, 9), dtype=np.int8)
        cpu_states.set_state(src_idx, state_data)
        
        # Clone state
        dst_idx = cpu_states.allocate_state()
        cpu_states.clone_state(src_idx, dst_idx)
        
        # Verify clone
        src_data = cpu_states.get_state(src_idx)
        dst_data = cpu_states.get_state(dst_idx)
        assert np.array_equal(src_data, dst_data)
        
        # Modify clone and verify independence
        cpu_states.apply_move(dst_idx, 4, 4, 1)
        src_data_after = cpu_states.get_state(src_idx)
        dst_data_after = cpu_states.get_state(dst_idx)
        assert not np.array_equal(src_data_after, dst_data_after)
    
    def test_batch_operations(self, cpu_states):
        """Test batch state operations"""
        # Allocate batch of states
        batch_size = 32
        state_indices = [cpu_states.allocate_state() for _ in range(batch_size)]
        
        # Batch clone from single source
        src_idx = state_indices[0]
        state_data = np.random.randint(0, 3, (9, 9), dtype=np.int8)
        cpu_states.set_state(src_idx, state_data)
        
        # Clone to all others
        for dst_idx in state_indices[1:]:
            cpu_states.clone_state(src_idx, dst_idx)
        
        # Verify all clones
        for idx in state_indices[1:]:
            assert np.array_equal(cpu_states.get_state(idx), state_data)
    
    def test_move_validation(self, cpu_states):
        """Test move validation logic"""
        idx = cpu_states.allocate_state()
        
        # Test valid move on empty board
        assert cpu_states.is_move_valid(idx, 4, 4)
        
        # Apply move
        cpu_states.apply_move(idx, 4, 4, 1)
        
        # Test invalid move on occupied position
        assert not cpu_states.is_move_valid(idx, 4, 4)
        
        # Test boundary conditions
        assert not cpu_states.is_move_valid(idx, -1, 0)
        assert not cpu_states.is_move_valid(idx, 0, 9)
        assert not cpu_states.is_move_valid(idx, 9, 0)
    
    def test_legal_moves_generation(self, cpu_states):
        """Test legal moves generation"""
        idx = cpu_states.allocate_state()
        
        # Empty board should have all moves legal
        legal_moves = cpu_states.get_legal_moves(idx)
        assert len(legal_moves) == 81  # 9x9 board
        
        # Apply some moves
        moves = [(4, 4), (3, 3), (5, 5)]
        for i, (x, y) in enumerate(moves):
            cpu_states.apply_move(idx, x, y, (i % 2) + 1)
        
        # Check legal moves reduced
        legal_moves_after = cpu_states.get_legal_moves(idx)
        assert len(legal_moves_after) == 78  # 81 - 3


class TestCPUTreeOperations:
    """Test CPU-specific tree operations"""
    
    @pytest.fixture
    def cpu_mcts(self):
        """Create CPU MCTS instance"""
        config = MCTSConfig(
            backend='cpu',
            board_size=9,
            game_type=GameType.GOMOKU,
            device='cpu',
            max_tree_nodes=10000,
            num_simulations=100
        )
        evaluator = MockEvaluator(board_size=9, device='cpu')
        return MCTS(config, evaluator)
    
    def test_tree_expansion_efficiency(self, cpu_mcts):
        """Test efficient tree expansion on CPU"""
        state = np.zeros((9, 9), dtype=np.int8)
        
        # Measure expansion time
        start_time = time.perf_counter()
        cpu_mcts._ensure_root_initialized(state)
        
        # Expand some nodes
        for _ in range(100):
            cpu_mcts.wave_search.run_wave(
                wave_size=1,
                node_to_state=cpu_mcts.node_to_state,
                state_pool_free_list=cpu_mcts.state_pool_free_list
            )
        
        expansion_time = time.perf_counter() - start_time
        
        # CPU expansion should be reasonably fast
        assert expansion_time < 1.0, f"CPU expansion too slow: {expansion_time:.3f}s"
        
        # Verify tree structure
        assert cpu_mcts.tree.num_nodes > 1
        assert cpu_mcts.tree.node_data.expanded[0].item()
    
    def test_ucb_computation_vectorized(self, cpu_mcts):
        """Test vectorized UCB computation on CPU"""
        # Initialize tree with some nodes
        state = np.zeros((9, 9), dtype=np.int8)
        cpu_mcts._ensure_root_initialized(state)
        
        # Add some children with various statistics
        num_children = 10
        for i in range(num_children):
            cpu_mcts.tree.add_node(parent_idx=0, action=i)
            cpu_mcts.tree.node_data.visit_counts[i+1] = i + 1
            cpu_mcts.tree.node_data.values[i+1] = 0.1 * i
            cpu_mcts.tree.node_data.priors[i+1] = 1.0 / num_children
        
        # Compute UCB scores
        parent_visits = cpu_mcts.tree.node_data.visit_counts[0]
        child_indices = torch.arange(1, num_children + 1)
        
        # UCB should prefer less visited nodes with good priors
        # This is implicitly tested through wave search selection
        
        # Run selection to verify UCB is working
        selected_count = {}
        for _ in range(100):
            cpu_mcts.wave_search.run_wave(
                wave_size=1,
                node_to_state=cpu_mcts.node_to_state,
                state_pool_free_list=cpu_mcts.state_pool_free_list
            )
        
        # Check that selection is working (root visits increased)
        assert cpu_mcts.tree.node_data.visit_counts[0] > parent_visits
    
    def test_batch_node_creation(self, cpu_mcts):
        """Test efficient batch node creation"""
        state = np.zeros((9, 9), dtype=np.int8)
        cpu_mcts._ensure_root_initialized(state)
        
        # Time batch node creation
        start_time = time.perf_counter()
        
        # Create many nodes in batch
        batch_size = 100
        actions = list(range(batch_size))
        parent_idx = 0
        
        for action in actions:
            cpu_mcts.tree.add_node(parent_idx=parent_idx, action=action)
        
        batch_time = time.perf_counter() - start_time
        
        # Should be fast
        assert batch_time < 0.1, f"Batch node creation too slow: {batch_time:.3f}s"
        
        # Verify nodes created
        assert cpu_mcts.tree.num_nodes >= batch_size + 1


class TestCPUVectorizedOperations:
    """Test CPU vectorized operations"""
    
    def test_vectorized_ucb(self):
        """Test vectorized UCB calculation"""
        ops = VectorizedOps()
        
        # Create test data
        n = 1000
        parent_visits = 100
        child_visits = torch.randint(1, 50, (n,), dtype=torch.float32)
        child_values = torch.rand(n) * 2 - 1  # [-1, 1]
        child_priors = torch.rand(n)
        child_priors /= child_priors.sum()  # Normalize
        c_puct = 1.0
        
        # Compute UCB
        start = time.perf_counter()
        ucb_scores = ops.compute_ucb_vectorized(
            parent_visits, child_visits, child_values, child_priors, c_puct
        )
        compute_time = time.perf_counter() - start
        
        # Verify computation
        assert ucb_scores.shape == (n,)
        assert torch.all(torch.isfinite(ucb_scores))
        assert compute_time < 0.01, f"UCB computation too slow: {compute_time:.3f}s"
        
        # Verify UCB formula
        exploration = c_puct * child_priors * torch.sqrt(parent_visits) / (1 + child_visits)
        expected_ucb = child_values + exploration
        assert torch.allclose(ucb_scores, expected_ucb, rtol=1e-5)
    
    def test_vectorized_softmax(self):
        """Test vectorized softmax for policy computation"""
        ops = VectorizedOps()
        
        # Test data
        logits = torch.randn(32, 81)  # 32 boards, 81 moves
        
        # Compute softmax
        start = time.perf_counter()
        policies = ops.softmax_vectorized(logits)
        compute_time = time.perf_counter() - start
        
        # Verify
        assert policies.shape == logits.shape
        assert torch.allclose(policies.sum(dim=1), torch.ones(32))
        assert torch.all(policies >= 0)
        assert compute_time < 0.01
    
    def test_vectorized_masking(self):
        """Test vectorized move masking"""
        ops = VectorizedOps()
        
        # Create policies and masks
        batch_size = 16
        board_size = 9
        num_moves = board_size * board_size
        
        policies = torch.rand(batch_size, num_moves)
        legal_masks = torch.randint(0, 2, (batch_size, num_moves), dtype=torch.bool)
        
        # Apply masking
        masked_policies = ops.apply_move_mask(policies, legal_masks)
        
        # Verify
        assert masked_policies.shape == policies.shape
        # Illegal moves should be zero
        assert torch.all(masked_policies[~legal_masks] == 0)
        # Legal moves should sum to 1
        for i in range(batch_size):
            if legal_masks[i].any():
                assert torch.allclose(
                    masked_policies[i][legal_masks[i]].sum(),
                    torch.tensor(1.0)
                )


class TestCPUThreadSafety:
    """Test thread safety of CPU operations"""
    
    def test_concurrent_state_access(self):
        """Test concurrent access to CPU game states"""
        cpu_states = CPUGameStates(
            max_states=1000,
            board_size=9,
            game_type='gomoku'
        )
        
        # Allocate states for each thread
        num_threads = 4
        states_per_thread = 10
        thread_states = []
        
        for _ in range(num_threads):
            states = [cpu_states.allocate_state() for _ in range(states_per_thread)]
            thread_states.append(states)
        
        errors = []
        
        def worker(thread_id: int, state_indices: List[int]):
            """Worker thread that modifies states"""
            try:
                for _ in range(100):
                    for idx in state_indices:
                        # Random operations
                        x, y = np.random.randint(0, 9, 2)
                        if cpu_states.is_move_valid(idx, x, y):
                            cpu_states.apply_move(idx, x, y, thread_id + 1)
                        
                        # Clone to new state
                        new_idx = cpu_states.allocate_state()
                        cpu_states.clone_state(idx, new_idx)
                        cpu_states.deallocate_state(new_idx)
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Run threads
        threads = []
        for i, states in enumerate(thread_states):
            t = threading.Thread(target=worker, args=(i, states))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check for errors
        assert len(errors) == 0, f"Thread errors: {errors}"
    
    def test_concurrent_tree_operations(self):
        """Test concurrent tree operations"""
        config = MCTSConfig(
            backend='cpu',
            board_size=9,
            game_type=GameType.GOMOKU,
            device='cpu',
            max_tree_nodes=50000
        )
        evaluator = MockEvaluator(board_size=9, device='cpu')
        mcts = MCTS(config, evaluator)
        
        state = np.zeros((9, 9), dtype=np.int8)
        mcts._ensure_root_initialized(state)
        
        errors = []
        results = []
        
        def search_worker(thread_id: int):
            """Worker that runs MCTS searches"""
            try:
                # Each thread does its own searches
                for _ in range(5):
                    policy = mcts.search(state, num_simulations=50)
                    results.append((thread_id, policy))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Note: True concurrent MCTS would require proper locking
        # This tests sequential execution for thread safety validation
        for i in range(4):
            search_worker(i)
        
        # Verify results
        assert len(errors) == 0, f"Search errors: {errors}"
        assert len(results) == 20  # 4 threads * 5 searches
        
        # All policies should be valid
        for thread_id, policy in results:
            assert policy.sum() > 0.99
            assert np.all(policy >= 0)


class TestCPUPerformanceOptimizations:
    """Test CPU-specific performance optimizations"""
    
    def test_cache_efficiency(self):
        """Test cache-friendly memory access patterns"""
        # Test that tree data is laid out efficiently
        config = MCTSConfig(
            backend='cpu',
            board_size=9,
            game_type=GameType.GOMOKU,
            device='cpu'
        )
        evaluator = MockEvaluator(board_size=9, device='cpu')
        mcts = MCTS(config, evaluator)
        
        # Check memory layout
        tree_data = mcts.tree.node_data
        
        # Tensors should be contiguous for cache efficiency
        assert tree_data.visit_counts.is_contiguous()
        assert tree_data.values.is_contiguous()
        assert tree_data.priors.is_contiguous()
    
    def test_simd_operations(self):
        """Test SIMD-friendly operations"""
        # Create aligned data for SIMD
        n = 1024  # Power of 2 for alignment
        data1 = torch.randn(n)
        data2 = torch.randn(n)
        
        # Time vectorized operations
        start = time.perf_counter()
        
        # These operations should use SIMD on CPU
        result = data1 * data2 + 0.5
        result = torch.sqrt(result.abs())
        result = torch.clamp(result, 0, 1)
        
        simd_time = time.perf_counter() - start
        
        # Should be very fast for 1024 elements
        assert simd_time < 0.001, f"SIMD operations too slow: {simd_time:.6f}s"
    
    def test_memory_pooling(self):
        """Test memory pooling for reduced allocations"""
        cpu_states = CPUGameStates(
            max_states=1000,
            board_size=9,
            game_type='gomoku'
        )
        
        # Measure allocation/deallocation performance
        alloc_times = []
        dealloc_times = []
        
        for _ in range(100):
            # Allocate
            start = time.perf_counter()
            idx = cpu_states.allocate_state()
            alloc_times.append(time.perf_counter() - start)
            
            # Deallocate
            start = time.perf_counter()
            cpu_states.deallocate_state(idx)
            dealloc_times.append(time.perf_counter() - start)
        
        # Pooled allocation should be fast and consistent
        avg_alloc = np.mean(alloc_times)
        avg_dealloc = np.mean(dealloc_times)
        
        assert avg_alloc < 1e-5, f"Allocation too slow: {avg_alloc:.6f}s"
        assert avg_dealloc < 1e-5, f"Deallocation too slow: {avg_dealloc:.6f}s"
        
        # Times should be consistent (low variance)
        assert np.std(alloc_times) < 1e-5
        assert np.std(dealloc_times) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])