"""Tests for mixed precision backup kernel

This module tests the adaptive FP16/FP32 precision switching for memory
efficiency and performance optimization in MCTS value backup operations.
"""

import pytest
import torch
import numpy as np
from typing import List, Tuple
import time

from mcts.cuda_kernels import create_cuda_kernels, CUDA_AVAILABLE


class TestMixedPrecisionBackup:
    """Test mixed precision value backup operations"""
    
    def test_fp16_fp32_conversion(self):
        """Test conversion between FP16 and FP32"""
        kernels = create_cuda_kernels()
        
        # Test values that should work well in FP16
        fp32_values = torch.tensor([1.0, -0.5, 0.25, 0.0, 2.5], dtype=torch.float32)
        
        # Convert to FP16 and back
        fp16_values = kernels.convert_to_fp16(fp32_values)
        restored_values = kernels.convert_from_fp16(fp16_values)
        
        # Should be close but not exact due to precision loss
        assert torch.allclose(fp32_values.to(restored_values.device), restored_values, atol=1e-3)
        assert fp16_values.dtype == torch.float16
        assert restored_values.dtype == torch.float32
        
    def test_precision_decision_heuristic(self):
        """Test the heuristic for deciding when to use FP16 vs FP32"""
        kernels = create_cuda_kernels()
        
        # Values that should trigger FP32 (extreme values)
        extreme_values = torch.tensor([1e6, -1e-6, float('inf')])
        use_fp16 = kernels.should_use_fp16(extreme_values)
        assert not use_fp16  # Should use FP32 for extreme values
        
        # Values that should work with FP16 (normal range)
        normal_values = torch.tensor([0.5, -0.3, 1.2, 0.0])
        use_fp16 = kernels.should_use_fp16(normal_values)
        assert use_fp16  # Should use FP16 for normal values
        
        # Mixed values (some extreme, some normal)
        mixed_values = torch.tensor([0.5, 1e5, -0.3])
        use_fp16 = kernels.should_use_fp16(mixed_values)
        assert not use_fp16  # Should use FP32 if any value is extreme
        
    def test_adaptive_backup_kernel(self):
        """Test the adaptive mixed precision backup kernel"""
        kernels = create_cuda_kernels()
        
        # Create test data
        batch_size = 1000
        paths = torch.randint(0, 500, (batch_size, 10))  # Random paths
        values = torch.rand(batch_size) - 0.5  # Values in [-0.5, 0.5]
        visit_counts = torch.zeros(500)
        value_sums = torch.zeros(500)
        
        # Test adaptive backup
        new_visits, new_values, precision_used = kernels.mixed_precision_backup(
            paths, values, visit_counts, value_sums
        )
        
        # Verify results
        assert new_visits.shape == visit_counts.shape
        assert new_values.shape == value_sums.shape
        assert precision_used in ['fp16', 'fp32']
        
        # Values should be finite and reasonable
        assert torch.all(torch.isfinite(new_visits))
        assert torch.all(torch.isfinite(new_values))
        
    def test_precision_switching_threshold(self):
        """Test that precision switches appropriately based on value range"""
        kernels = create_cuda_kernels()
        
        batch_size = 100
        paths = torch.randint(0, 50, (batch_size, 5))
        
        # Test with small values (should use FP16)
        small_values = torch.rand(batch_size) * 0.1  # [0, 0.1]
        visit_counts = torch.zeros(50)
        value_sums = torch.zeros(50)
        
        _, _, precision = kernels.mixed_precision_backup(
            paths, small_values, visit_counts, value_sums
        )
        assert precision == 'fp16'
        
        # Test with large values (should use FP32)
        large_values = torch.rand(batch_size) * 1000 + 500  # [500, 1500]
        _, _, precision = kernels.mixed_precision_backup(
            paths, large_values, visit_counts, value_sums
        )
        assert precision == 'fp32'
        
    def test_memory_efficiency(self):
        """Test that FP16 actually saves memory"""
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available")
            
        # Create large tensors
        n_elements = 1000000
        
        # FP32 tensor
        fp32_tensor = torch.rand(n_elements, dtype=torch.float32, device='cuda')
        fp32_memory = fp32_tensor.element_size() * fp32_tensor.numel()
        
        # FP16 tensor
        fp16_tensor = torch.rand(n_elements, dtype=torch.float16, device='cuda')
        fp16_memory = fp16_tensor.element_size() * fp16_tensor.numel()
        
        # FP16 should use exactly half the memory
        assert fp16_memory == fp32_memory // 2
        
        print(f"\nMemory usage:")
        print(f"FP32: {fp32_memory / 1024**2:.2f} MB")
        print(f"FP16: {fp16_memory / 1024**2:.2f} MB")
        print(f"Memory savings: {(1 - fp16_memory/fp32_memory)*100:.1f}%")
        
    def test_numerical_stability(self):
        """Test numerical stability in mixed precision operations"""
        kernels = create_cuda_kernels()
        
        # Test with accumulation (common source of numerical issues)
        batch_size = 10000
        n_nodes = 100
        
        # Create many small updates that could cause precision issues
        paths = torch.randint(0, n_nodes, (batch_size, 3))
        values = torch.rand(batch_size) * 1e-3  # Very small values
        
        visit_counts = torch.zeros(n_nodes)
        value_sums = torch.zeros(n_nodes)
        
        # Perform backup
        new_visits, new_values, _ = kernels.mixed_precision_backup(
            paths, values, visit_counts, value_sums
        )
        
        # Check for numerical issues
        assert torch.all(torch.isfinite(new_visits))
        assert torch.all(torch.isfinite(new_values))
        assert torch.all(new_visits >= 0)  # Visit counts should be non-negative
        
    def test_precision_performance_comparison(self):
        """Compare performance of FP16 vs FP32 operations"""
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available")
            
        kernels = create_cuda_kernels()
        
        # Large batch for performance testing
        batch_size = 10000
        n_nodes = 1000
        max_depth = 20
        
        paths = torch.randint(0, n_nodes, (batch_size, max_depth), device='cuda')
        values_fp32 = torch.rand(batch_size, device='cuda', dtype=torch.float32)
        values_fp16 = values_fp32.to(torch.float16)
        
        visit_counts = torch.zeros(n_nodes, device='cuda')
        value_sums = torch.zeros(n_nodes, device='cuda')
        
        # Warmup
        _ = kernels._backup_fp32(paths, values_fp32, visit_counts.clone(), value_sums.clone())
        _ = kernels._backup_fp16(paths, values_fp16, visit_counts.clone().half(), value_sums.clone().half())
        torch.cuda.synchronize()
        
        # Benchmark FP32
        start = time.time()
        for _ in range(10):
            _ = kernels._backup_fp32(paths, values_fp32, visit_counts.clone(), value_sums.clone())
        torch.cuda.synchronize()
        fp32_time = time.time() - start
        
        # Benchmark FP16
        start = time.time()
        for _ in range(10):
            _ = kernels._backup_fp16(paths, values_fp16, visit_counts.clone().half(), value_sums.clone().half())
        torch.cuda.synchronize()
        fp16_time = time.time() - start
        
        speedup = fp32_time / fp16_time
        print(f"\nPerformance comparison:")
        print(f"FP32 time: {fp32_time:.4f}s")
        print(f"FP16 time: {fp16_time:.4f}s") 
        print(f"FP16 speedup: {speedup:.2f}x")
        
        # FP16 should be faster (though not always guaranteed)
        # At minimum, it shouldn't be much slower
        assert speedup > 0.8  # Allow some variance
        
    def test_gradient_precision_handling(self):
        """Test handling of gradients in mixed precision"""
        kernels = create_cuda_kernels()
        
        # Create tensors that require gradients
        values = torch.rand(100, requires_grad=True)
        visit_counts = torch.zeros(50)
        value_sums = torch.zeros(50)
        paths = torch.randint(0, 50, (100, 5))
        
        # Perform backup operation
        new_visits, new_values, precision = kernels.mixed_precision_backup(
            paths, values, visit_counts, value_sums
        )
        
        # Create a loss and backpropagate
        loss = new_values.sum()
        loss.backward()
        
        # Gradients should be computed correctly
        assert values.grad is not None
        assert torch.all(torch.isfinite(values.grad))
        
    def test_edge_cases(self):
        """Test edge cases in mixed precision operations"""
        kernels = create_cuda_kernels()
        
        # Empty batch
        empty_paths = torch.empty(0, 5, dtype=torch.long)
        empty_values = torch.empty(0)
        visit_counts = torch.zeros(10)
        value_sums = torch.zeros(10)
        
        new_visits, new_values, precision = kernels.mixed_precision_backup(
            empty_paths, empty_values, visit_counts, value_sums
        )
        
        # Should handle empty batch gracefully
        assert torch.equal(new_visits, visit_counts.to(new_visits.device))
        assert torch.equal(new_values, value_sums.to(new_values.device))
        
        # Single element batch
        single_path = torch.tensor([[1, 2, -1, -1, -1]])
        single_value = torch.tensor([0.5])
        
        new_visits, new_values, precision = kernels.mixed_precision_backup(
            single_path, single_value, visit_counts, value_sums
        )
        
        assert new_visits[1] == 1  # Node 1 should have one visit
        assert new_visits[2] == 1  # Node 2 should have one visit
        
    def test_overflow_underflow_handling(self):
        """Test handling of numerical overflow and underflow"""
        kernels = create_cuda_kernels()
        
        # Test with values that might cause overflow in FP16
        large_values = torch.tensor([65000.0, 70000.0])  # Near FP16 max
        should_use_fp16 = kernels.should_use_fp16(large_values)
        assert not should_use_fp16  # Should switch to FP32
        
        # Test with very small values (underflow risk)
        tiny_values = torch.tensor([1e-8, 1e-10])
        should_use_fp16 = kernels.should_use_fp16(tiny_values)
        assert not should_use_fp16  # Should use FP32 for tiny values
        
        # Test actual backup with overflow-prone values
        paths = torch.tensor([[0, 1, -1]])
        overflow_values = torch.tensor([60000.0])
        visit_counts = torch.zeros(5)
        value_sums = torch.zeros(5)
        
        new_visits, new_values, precision = kernels.mixed_precision_backup(
            paths, overflow_values, visit_counts, value_sums
        )
        
        # Should use FP32 and handle correctly
        assert precision == 'fp32'
        assert torch.all(torch.isfinite(new_values))


class TestMixedPrecisionIntegration:
    """Integration tests for mixed precision with other MCTS components"""
    
    def test_integration_with_ucb_kernel(self):
        """Test mixed precision with UCB computations"""
        kernels = create_cuda_kernels()
        
        # Create test data with mixed precision requirements
        batch_size = 500
        q_values = torch.rand(batch_size) - 0.5
        visit_counts = torch.randint(0, 1000, (batch_size,)).float()
        parent_visits = torch.randint(1000, 10000, (batch_size,)).float()
        priors = torch.rand(batch_size)
        priors = priors / priors.sum()
        
        # Test with forced FP16
        if kernels.should_use_fp16(q_values):
            q_fp16 = q_values.half()
            visit_fp16 = visit_counts.half()
            parent_fp16 = parent_visits.half()
            priors_fp16 = priors.half()
            
            ucb_fp16 = kernels.compute_batched_ucb(
                q_fp16, visit_fp16, parent_fp16, priors_fp16, c_puct=1.0
            )
            
            # Compare with FP32 version
            ucb_fp32 = kernels.compute_batched_ucb(
                q_values, visit_counts, parent_visits, priors, c_puct=1.0
            )
            
            # Should be reasonably close
            assert torch.allclose(ucb_fp16.float(), ucb_fp32.to(ucb_fp16.device), atol=1e-2)
            
    def test_mixed_precision_with_tree_operations(self):
        """Test mixed precision with GPU tree operations"""
        from mcts.gpu_tree_kernels import GPUTreeKernels
        
        kernels = create_cuda_kernels()
        tree_kernels = GPUTreeKernels()
        
        # Create tree structure
        n_nodes = 100
        max_children = 5
        children_tensor = torch.randint(-1, n_nodes, (n_nodes, max_children))
        
        # Make tree structure valid
        for i in range(n_nodes):
            for j in range(max_children):
                if children_tensor[i, j] >= n_nodes:
                    children_tensor[i, j] = -1
                    
        # Mixed precision node values
        q_values = torch.rand(n_nodes) * 100  # Larger range
        visit_counts = torch.rand(n_nodes) * 1000
        
        # Test precision decision for tree operations
        precision_for_q = kernels.should_use_fp16(q_values)
        precision_for_visits = kernels.should_use_fp16(visit_counts)
        
        print(f"\nPrecision decisions:")
        print(f"Q-values use FP16: {precision_for_q}")
        print(f"Visit counts use FP16: {precision_for_visits}")
        
        # Tree operations should work with chosen precision
        if precision_for_q and precision_for_visits:
            q_selected = q_values.half()
            visits_selected = visit_counts.half()
        else:
            q_selected = q_values.float()
            visits_selected = visit_counts.float()
            
        # Test tree traversal with selected precision
        priors = torch.rand(n_nodes, max_children)
        priors = priors / priors.sum(dim=1, keepdim=True)
        
        if precision_for_q and precision_for_visits:
            priors = priors.half()
            
        node_indices = torch.arange(5)
        selected = tree_kernels.parallel_select_children(
            node_indices, children_tensor, q_selected, 
            visits_selected, priors, c_puct=1.0
        )
        
        assert selected.shape == (5,)
        assert torch.all(selected >= -1)  # Valid indices or -1 for no children