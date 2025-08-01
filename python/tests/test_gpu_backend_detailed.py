#!/usr/bin/env python3
"""Detailed tests for GPU backend functionality

This module provides comprehensive testing of GPU-specific MCTS features including:
- GPU tensor operations
- CUDA kernel functionality
- GPU memory management
- Batch processing efficiency
- Mixed precision support
"""

import pytest
import torch
import numpy as np
import logging
import time
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Import GPU-specific components
from mcts.core.mcts import MCTS
from mcts.core.mcts_config import MCTSConfig
from mcts.gpu.gpu_game_states import GPUGameStates, GPUGameStatesConfig, GameType
from mcts.gpu.csr_tree import CSRTree, CSRTreeConfig
from mcts.gpu.mcts_gpu_accelerator import get_mcts_gpu_accelerator
from mcts.gpu.cuda_manager import detect_cuda_kernels
from mock_evaluator import MockEvaluator

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestGPUGameStates:
    """Test GPU game state management"""
    
    @pytest.fixture
    def gpu_states(self):
        """Create GPU game states instance"""
        config = GPUGameStatesConfig(
            capacity=10000,
            game_type=GameType.GOMOKU,
            board_size=9,
            device='cuda',
            dtype=torch.float32
        )
        return GPUGameStates(config)
    
    def test_gpu_memory_allocation(self, gpu_states):
        """Test GPU memory allocation and layout"""
        # Check tensor allocation on GPU
        assert gpu_states.board_tensors.device.type == 'cuda'
        assert gpu_states.current_players.device.type == 'cuda'
        assert gpu_states.move_counts.device.type == 'cuda'
        
        # Check memory layout
        assert gpu_states.board_tensors.is_contiguous()
        assert gpu_states.board_tensors.shape == (10000, 9, 9)
        
        # Estimate memory usage
        tensor_bytes = gpu_states.board_tensors.numel() * gpu_states.board_tensors.element_size()
        # Should be reasonable (< 100MB for this size)
        assert tensor_bytes < 100 * 1024 * 1024
    
    def test_batch_state_operations(self, gpu_states):
        """Test batched GPU state operations"""
        batch_size = 128
        state_indices = torch.arange(batch_size, device='cuda')
        
        # Initialize states
        gpu_states.reset_states(state_indices)
        
        # Test batch cloning
        src_indices = state_indices[:64]
        dst_indices = state_indices[64:]
        
        # Set some data in source states
        gpu_states.board_tensors[src_indices] = torch.randint(
            0, 3, (64, 9, 9), device='cuda', dtype=torch.int8
        )
        
        # Batch clone
        start = time.perf_counter()
        gpu_states.clone_states(src_indices, dst_indices)
        clone_time = time.perf_counter() - start
        
        # Verify cloning
        assert torch.equal(
            gpu_states.board_tensors[src_indices],
            gpu_states.board_tensors[dst_indices]
        )
        
        # Should be very fast on GPU
        assert clone_time < 0.01, f"Batch cloning too slow: {clone_time:.3f}s"
    
    def test_vectorized_move_application(self, gpu_states):
        """Test vectorized move application on GPU"""
        n_states = 256
        state_indices = torch.arange(n_states, device='cuda')
        
        # Reset states
        gpu_states.reset_states(state_indices)
        
        # Generate random valid moves
        moves = torch.randint(0, 81, (n_states,), device='cuda')
        players = torch.ones(n_states, device='cuda', dtype=torch.int8)
        
        # Apply moves in batch
        start = time.perf_counter()
        gpu_states.apply_moves_batch(state_indices, moves, players)
        apply_time = time.perf_counter() - start
        
        # Verify moves applied
        for i in range(min(10, n_states)):  # Check first 10
            move = moves[i].item()
            row, col = move // 9, move % 9
            board_value = gpu_states.board_tensors[i, row, col].item()
            assert board_value == players[i].item()
        
        # Should be very fast
        assert apply_time < 0.01, f"Batch move application too slow: {apply_time:.3f}s"
    
    def test_legal_moves_mask_generation(self, gpu_states):
        """Test GPU legal moves mask generation"""
        n_states = 512
        state_indices = torch.arange(n_states, device='cuda')
        
        # Set up some occupied positions
        gpu_states.reset_states(state_indices)
        
        # Apply random moves to create different board states
        for _ in range(10):
            moves = torch.randint(0, 81, (n_states,), device='cuda')
            players = torch.randint(1, 3, (n_states,), device='cuda', dtype=torch.int8)
            gpu_states.apply_moves_batch(state_indices, moves, players)
        
        # Get legal moves masks
        start = time.perf_counter()
        legal_masks = gpu_states.get_legal_moves_mask(state_indices)
        mask_time = time.perf_counter() - start
        
        # Verify masks
        assert legal_masks.shape == (n_states, 81)
        assert legal_masks.dtype == torch.bool
        
        # Empty positions should be legal
        for i in range(min(10, n_states)):
            board = gpu_states.board_tensors[i]
            flat_board = board.reshape(-1)
            empty_mask = (flat_board == 0)
            assert torch.equal(legal_masks[i], empty_mask)
        
        # Should be fast
        assert mask_time < 0.01, f"Legal mask generation too slow: {mask_time:.3f}s"
    
    def test_feature_extraction(self, gpu_states):
        """Test neural network feature extraction"""
        n_states = 64
        state_indices = torch.arange(n_states, device='cuda')
        
        # Set up states with some moves
        gpu_states.reset_states(state_indices)
        for i in range(n_states):
            # Apply i moves to create variety
            for j in range(i % 10):
                move = (j * 7 + i * 3) % 81  # Pseudo-random pattern
                gpu_states.apply_moves_batch(
                    torch.tensor([i], device='cuda'),
                    torch.tensor([move], device='cuda'),
                    torch.tensor([1 + (j % 2)], device='cuda', dtype=torch.int8)
                )
        
        # Extract features
        start = time.perf_counter()
        features = gpu_states.get_nn_features(state_indices)
        feature_time = time.perf_counter() - start
        
        # Verify features
        assert features.shape[0] == n_states
        assert features.device.type == 'cuda'
        assert features.dtype == torch.float32
        
        # Should be fast
        assert feature_time < 0.01, f"Feature extraction too slow: {feature_time:.3f}s"


class TestGPUTreeOperations:
    """Test GPU-optimized tree operations"""
    
    @pytest.fixture
    def gpu_tree(self):
        """Create GPU CSR tree"""
        config = CSRTreeConfig(
            max_nodes=100000,
            max_children=362,  # Max for Go
            device='cuda'
        )
        return CSRTree(config)
    
    def test_gpu_csr_structure(self, gpu_tree):
        """Test GPU CSR tree structure"""
        # Verify GPU allocation
        assert gpu_tree.device.type == 'cuda'
        assert gpu_tree.csr_storage.row_ptr.device.type == 'cuda'
        assert gpu_tree.csr_storage.col_idx.device.type == 'cuda'
        
        # Add nodes and verify structure
        gpu_tree.add_node(parent_idx=0, action=5)
        gpu_tree.add_node(parent_idx=0, action=10)
        gpu_tree.add_node(parent_idx=1, action=3)
        
        # Check CSR structure
        assert gpu_tree.num_nodes == 4
        
        # Root should have 2 children
        root_start = gpu_tree.csr_storage.row_ptr[0].item()
        root_end = gpu_tree.csr_storage.row_ptr[1].item()
        assert root_end - root_start == 2
    
    def test_batch_node_expansion(self, gpu_tree):
        """Test batch node expansion on GPU"""
        # Add root
        gpu_tree.add_node(parent_idx=-1, action=-1)
        
        # Batch expand from root
        n_children = 81  # Full board
        actions = torch.arange(n_children, device='cuda')
        
        start = time.perf_counter()
        for action in actions:
            gpu_tree.add_node(parent_idx=0, action=action.item())
        expand_time = time.perf_counter() - start
        
        # Verify expansion
        assert gpu_tree.num_nodes == n_children + 1
        children = gpu_tree.get_children(0)
        assert len(children) == n_children
        
        # Should be fast
        assert expand_time < 0.1, f"Batch expansion too slow: {expand_time:.3f}s"
    
    def test_parallel_tree_traversal(self, gpu_tree):
        """Test parallel tree traversal on GPU"""
        # Build a tree with multiple levels
        gpu_tree.add_node(parent_idx=-1, action=-1)  # Root
        
        # Level 1
        for i in range(10):
            gpu_tree.add_node(parent_idx=0, action=i)
        
        # Level 2
        for parent in range(1, 11):
            for i in range(5):
                gpu_tree.add_node(parent_idx=parent, action=i)
        
        # Parallel traversal - select best child for multiple nodes
        parent_indices = torch.arange(11, device='cuda')  # Root + level 1 nodes
        
        # Set some scores for selection
        gpu_tree.node_data.visit_counts[:61] = torch.randint(1, 100, (61,), device='cuda')
        gpu_tree.node_data.values[:61] = torch.rand(61, device='cuda') * 2 - 1
        
        # Time parallel selection (would use custom kernel in practice)
        start = time.perf_counter()
        # Simulate parallel operation
        for idx in parent_indices:
            children = gpu_tree.get_children(idx.item())
        select_time = time.perf_counter() - start
        
        # Should handle parallel access efficiently
        assert select_time < 0.1
    
    def test_gpu_memory_coalescing(self, gpu_tree):
        """Test memory access patterns for coalescing"""
        # Add many nodes
        n_nodes = 10000
        
        # Build tree level by level for better memory locality
        current_level = [0]  # Start with root
        node_count = 1
        
        while node_count < n_nodes:
            next_level = []
            for parent in current_level:
                if node_count >= n_nodes:
                    break
                # Add 4 children per node
                for i in range(4):
                    if node_count < n_nodes:
                        gpu_tree.add_node(parent_idx=parent, action=i)
                        next_level.append(node_count)
                        node_count += 1
            current_level = next_level
        
        # Access pattern test - coalesced access
        indices = torch.arange(min(1024, node_count), device='cuda')
        
        start = time.perf_counter()
        # Coalesced reads
        values = gpu_tree.node_data.values[indices]
        visits = gpu_tree.node_data.visit_counts[indices]
        coalesced_time = time.perf_counter() - start
        
        # Random access pattern (non-coalesced)
        random_indices = torch.randperm(min(1024, node_count), device='cuda')
        
        start = time.perf_counter()
        values_random = gpu_tree.node_data.values[random_indices]
        visits_random = gpu_tree.node_data.visit_counts[random_indices]
        random_time = time.perf_counter() - start
        
        # Coalesced should be faster (though difference may be small for small sizes)
        logger.info(f"Coalesced access: {coalesced_time:.6f}s")
        logger.info(f"Random access: {random_time:.6f}s")


class TestGPUAcceleration:
    """Test GPU acceleration and CUDA kernels"""
    
    @pytest.fixture
    def gpu_accelerator(self):
        """Get GPU accelerator if available"""
        try:
            return get_mcts_gpu_accelerator(torch.device('cuda'))
        except:
            return None
    
    @pytest.fixture
    def cuda_kernels(self):
        """Get CUDA kernels if available"""
        try:
            return detect_cuda_kernels()
        except:
            return None
    
    def test_cuda_kernel_availability(self, cuda_kernels):
        """Test CUDA kernel detection"""
        if cuda_kernels is None:
            pytest.skip("CUDA kernels not available")
        
        # Check for expected kernels
        assert hasattr(cuda_kernels, 'compute_ucb_scores')
        assert hasattr(cuda_kernels, 'apply_dirichlet_noise')
        assert hasattr(cuda_kernels, 'backup_values')
    
    def test_ucb_kernel_performance(self, cuda_kernels):
        """Test UCB computation kernel performance"""
        if cuda_kernels is None:
            pytest.skip("CUDA kernels not available")
        
        # Test data
        n = 10000
        parent_visits = 1000
        child_visits = torch.randint(1, 100, (n,), device='cuda', dtype=torch.float32)
        child_values = torch.rand(n, device='cuda') * 2 - 1
        child_priors = torch.rand(n, device='cuda')
        child_priors /= child_priors.sum()
        c_puct = 1.0
        
        # Time kernel execution
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        ucb_scores = cuda_kernels.compute_ucb_scores(
            parent_visits, child_visits, child_values, child_priors, c_puct
        )
        
        torch.cuda.synchronize()
        kernel_time = time.perf_counter() - start
        
        # Verify results
        assert ucb_scores.shape == (n,)
        assert ucb_scores.device.type == 'cuda'
        
        # Should be very fast
        assert kernel_time < 0.001, f"UCB kernel too slow: {kernel_time:.3f}s"
    
    def test_dirichlet_noise_kernel(self, cuda_kernels):
        """Test Dirichlet noise application kernel"""
        if cuda_kernels is None:
            pytest.skip("CUDA kernels not available")
        
        # Test data
        n = 1000
        priors = torch.rand(n, device='cuda')
        priors /= priors.sum()
        epsilon = 0.25
        alpha = 0.3
        
        # Apply noise
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        noisy_priors = cuda_kernels.apply_dirichlet_noise(
            priors, epsilon, alpha
        )
        
        torch.cuda.synchronize()
        kernel_time = time.perf_counter() - start
        
        # Verify
        assert noisy_priors.shape == priors.shape
        assert torch.allclose(noisy_priors.sum(), torch.tensor(1.0))
        assert not torch.equal(noisy_priors, priors)  # Should be different
        
        # Should be fast
        assert kernel_time < 0.001
    
    def test_value_backup_kernel(self, cuda_kernels):
        """Test value backup kernel for tree updates"""
        if cuda_kernels is None:
            pytest.skip("CUDA kernels not available")
        
        # Create path data
        path_length = 20
        path_indices = torch.randint(0, 10000, (path_length,), device='cuda')
        path_players = torch.ones(path_length, device='cuda', dtype=torch.int8)
        path_players[1::2] = -1  # Alternating players
        
        # Current node data
        visit_counts = torch.zeros(10000, device='cuda')
        values = torch.zeros(10000, device='cuda')
        
        # Leaf value to backup
        leaf_value = 0.8
        
        # Apply backup
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        cuda_kernels.backup_values(
            path_indices, path_players, visit_counts, values, leaf_value
        )
        
        torch.cuda.synchronize()
        kernel_time = time.perf_counter() - start
        
        # Verify updates
        for i, idx in enumerate(path_indices):
            assert visit_counts[idx] == 1
            expected_value = leaf_value * path_players[i].item()
            assert abs(values[idx].item() - expected_value) < 1e-5
        
        # Should be fast
        assert kernel_time < 0.001


class TestGPUMemoryManagement:
    """Test GPU memory management and optimization"""
    
    @pytest.fixture
    def gpu_mcts(self):
        """Create GPU MCTS instance"""
        config = MCTSConfig(
            backend='gpu',
            board_size=19,  # Full Go board
            game_type=GameType.GO,
            device='cuda',
            max_tree_nodes=1000000,  # Large tree
            num_simulations=1000,
            use_mixed_precision=True
        )
        evaluator = MockEvaluator(
            board_size=19,
            device='cuda',
            use_amp=True
        )
        return MCTS(config, evaluator)
    
    def test_memory_allocation_strategy(self, gpu_mcts):
        """Test GPU memory allocation strategy"""
        # Check pre-allocated memory
        initial_memory = torch.cuda.memory_allocated()
        
        # Run search to allocate memory
        state = np.zeros((19, 19), dtype=np.int8)
        gpu_mcts.search(state, num_simulations=100)
        
        after_search_memory = torch.cuda.memory_allocated()
        memory_used = after_search_memory - initial_memory
        
        # Memory should be reasonable (< 1GB for this config)
        assert memory_used < 1024 * 1024 * 1024, \
            f"Excessive memory usage: {memory_used / 1024 / 1024:.1f}MB"
        
        # Run more searches - memory should not grow much
        for _ in range(5):
            gpu_mcts.search(state, num_simulations=100)
        
        final_memory = torch.cuda.memory_allocated()
        additional_memory = final_memory - after_search_memory
        
        # Should reuse allocated memory
        assert additional_memory < 100 * 1024 * 1024, \
            f"Memory leak detected: {additional_memory / 1024 / 1024:.1f}MB"
    
    def test_mixed_precision_operations(self, gpu_mcts):
        """Test mixed precision (FP16) operations"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for mixed precision")
        
        # Verify mixed precision is enabled
        assert gpu_mcts.config.use_mixed_precision
        
        # Run search and check tensor dtypes
        state = np.zeros((19, 19), dtype=np.int8)
        
        # Hook to capture tensor dtypes during forward pass
        captured_dtypes = []
        
        def capture_dtype_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                captured_dtypes.append(output.dtype)
        
        # Register hook if using neural network model
        if hasattr(gpu_mcts.evaluator, 'model'):
            handle = gpu_mcts.evaluator.model.register_forward_hook(capture_dtype_hook)
            
            # Run search
            gpu_mcts.search(state, num_simulations=10)
            
            # Remove hook
            handle.remove()
            
            # Check that some operations used FP16
            fp16_ops = sum(1 for dtype in captured_dtypes if dtype == torch.float16)
            assert fp16_ops > 0, "No FP16 operations detected"
    
    def test_memory_pooling_efficiency(self, gpu_mcts):
        """Test memory pool efficiency"""
        state = np.zeros((19, 19), dtype=np.int8)
        
        # Measure memory allocation patterns
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # First search - establishes memory pools
        gpu_mcts.search(state, num_simulations=100)
        first_peak = torch.cuda.max_memory_allocated()
        
        # Reset stats for second measurement
        torch.cuda.reset_peak_memory_stats()
        
        # Subsequent searches should reuse memory
        for _ in range(10):
            gpu_mcts._reset_for_new_search()
            gpu_mcts.search(state, num_simulations=100)
        
        subsequent_peak = torch.cuda.max_memory_allocated()
        
        # Subsequent searches should use less peak memory
        assert subsequent_peak <= first_peak * 1.1, \
            "Memory pooling not working efficiently"
    
    def test_gpu_oom_handling(self):
        """Test out-of-memory handling"""
        # Create config that might cause OOM
        config = MCTSConfig(
            backend='gpu',
            board_size=19,
            game_type=GameType.GO,
            device='cuda',
            max_tree_nodes=10000000,  # Very large
            num_simulations=1000
        )
        
        try:
            evaluator = MockEvaluator(board_size=19, device='cuda')
            mcts = MCTS(config, evaluator)
            
            # This might fail on GPUs with limited memory
            state = np.zeros((19, 19), dtype=np.int8)
            mcts.search(state, num_simulations=100)
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            # Should handle OOM gracefully
            assert "out of memory" in str(e).lower() or "oom" in str(e).lower()
            logger.info("OOM handled gracefully")


class TestGPUBatchProcessing:
    """Test GPU batch processing efficiency"""
    
    def test_batch_size_optimization(self):
        """Test optimal batch size selection"""
        config = MCTSConfig(
            backend='gpu',
            board_size=9,
            game_type=GameType.GOMOKU,
            device='cuda',
            max_tree_nodes=100000
        )
        
        # Test different batch sizes
        batch_sizes = [1, 8, 16, 32, 64, 128]
        throughputs = []
        
        for batch_size in batch_sizes:
            evaluator = MockEvaluator(
                board_size=9,
                device='cuda',
                batch_size=batch_size
            )
            mcts = MCTS(config, evaluator)
            
            state = np.zeros((9, 9), dtype=np.int8)
            
            # Warmup
            mcts.search(state, num_simulations=50)
            
            # Measure throughput
            start = time.perf_counter()
            mcts.search(state, num_simulations=500)
            elapsed = time.perf_counter() - start
            
            throughput = 500 / elapsed
            throughputs.append(throughput)
            
            logger.info(f"Batch size {batch_size}: {throughput:.0f} sims/sec")
        
        # Larger batch sizes should generally be faster (up to a point)
        assert max(throughputs) > min(throughputs) * 1.5, \
            "Batch size optimization not effective"
    
    def test_warp_efficiency(self):
        """Test GPU warp efficiency"""
        # Create operations aligned to warp size (32)
        warp_size = 32
        n = warp_size * 100  # Multiple of warp size
        
        # Aligned computation
        data_aligned = torch.randn(n, device='cuda')
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # Warp-friendly operations
        result = data_aligned * 2.0 + 1.0
        result = torch.sqrt(result.abs())
        
        torch.cuda.synchronize()
        aligned_time = time.perf_counter() - start
        
        # Misaligned computation
        data_misaligned = torch.randn(n + 1, device='cuda')  # Not multiple of 32
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        result = data_misaligned * 2.0 + 1.0
        result = torch.sqrt(result.abs())
        
        torch.cuda.synchronize()
        misaligned_time = time.perf_counter() - start
        
        # Log for analysis
        logger.info(f"Aligned time: {aligned_time:.6f}s")
        logger.info(f"Misaligned time: {misaligned_time:.6f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])