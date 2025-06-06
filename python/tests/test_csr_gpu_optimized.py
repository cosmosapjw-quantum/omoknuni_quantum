"""Tests for optimized CSR GPU kernels"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock

from mcts.gpu.csr_tree import CSRTree, CSRTreeConfig
from mcts.gpu.csr_gpu_kernels_optimized import CSRBatchOperations, OptimizedCSRKernels


class TestOptimizedCSRKernels:
    """Test optimized CSR GPU kernels"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def csr_tree(self, device):
        """Create a CSR tree for testing"""
        config = CSRTreeConfig(
            max_nodes=10000,
            max_edges=50000,
            device=device.type,
            dtype_indices=torch.int32,
            dtype_actions=torch.int16,
            dtype_values=torch.float32
        )
        tree = CSRTree(config)
        
        # Add root
        root_idx = tree.add_root(prior=1.0)
        
        # Add some children
        for i in range(10):
            child_idx = tree.add_child(root_idx, action=i, child_prior=0.1)
            tree.visit_counts[child_idx] = i + 1
            tree.value_sums[child_idx] = (i + 1) * 0.5
            
            # Add grandchildren
            for j in range(5):
                gc_idx = tree.add_child(child_idx, action=j, child_prior=0.02)
                tree.visit_counts[gc_idx] = j
                tree.value_sums[gc_idx] = j * 0.1
        
        # Update row pointers if needed
        if hasattr(tree, '_rebuild_row_pointers'):
            tree._rebuild_row_pointers()
        
        return tree
    
    @pytest.fixture
    def batch_ops(self, device):
        """Create batch operations handler"""
        return CSRBatchOperations(device)
    
    def test_kernel_configuration(self, device):
        """Test optimal kernel configuration selection"""
        if device.type != 'cuda':
            pytest.skip("GPU test requires CUDA")
            
        kernels = OptimizedCSRKernels(device)
        
        # Test different problem sizes
        config_small = kernels.get_optimal_config(100)
        assert config_small['block_size'] == 128
        
        config_medium = kernels.get_optimal_config(5000)
        assert config_medium['block_size'] == 256
        
        config_large = kernels.get_optimal_config(20000)
        assert config_large['block_size'] == 512
        
        # Test caching
        config_cached = kernels.get_optimal_config(100)
        assert kernels.stats['cache_hits'] == 1
    
    def test_batch_ucb_selection(self, batch_ops, csr_tree, device):
        """Test batch UCB selection with optimized kernels"""
        # Select actions for multiple nodes
        node_indices = torch.tensor([0, 1, 2, 3], device=device)
        
        actions, scores = batch_ops.batch_select_ucb(
            csr_tree,
            node_indices,
            c_puct=1.4,
            temperature=1.0
        )
        
        assert actions.shape == (4,)
        assert scores.shape == (4,)
        
        # Root should select best child (highest visit count due to UCB)
        assert actions[0] >= 0
    
    def test_coalesced_memory_access(self, batch_ops, csr_tree, device):
        """Test coalesced memory access patterns"""
        batch_size = 32
        max_depth = 5
        
        # Create paths with varying lengths
        paths = torch.full((batch_size, max_depth), -1, device=device, dtype=torch.int32)
        
        # Fill with some valid paths
        for i in range(batch_size):
            depth = min(i % max_depth + 1, max_depth)
            for j in range(depth):
                paths[i, j] = j  # Simple ascending pattern
        
        # Random values to backup
        values = torch.randn(batch_size, device=device)
        
        # Test coalesced backup
        initial_visits = csr_tree.visit_counts.clone()
        initial_values = csr_tree.value_sums.clone()
        
        batch_ops.coalesced_backup(csr_tree, paths, values)
        
        # Verify updates
        assert not torch.equal(csr_tree.visit_counts, initial_visits)
        assert not torch.equal(csr_tree.value_sums, initial_values)
    
    def test_parallel_expansion(self, batch_ops, csr_tree, device):
        """Test parallel node expansion"""
        # Create candidate nodes
        candidate_nodes = torch.tensor([1, 2, 3, 4, 5], device=device, dtype=torch.int32)
        candidate_scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5], device=device)
        
        # Set visit counts to meet expansion threshold
        csr_tree.visit_counts[candidate_nodes] = 5
        
        # Expand nodes
        expanded = batch_ops.parallel_expand_nodes(
            csr_tree,
            candidate_nodes,
            candidate_scores,
            expansion_budget=3,
            min_visits=1
        )
        
        # Should expand up to budget
        assert len(expanded) <= 3
        
        # Expanded nodes should be marked
        if len(expanded) > 0:
            flags = csr_tree.flags[expanded]
            assert (flags & 1).all()
    
    def test_memory_efficiency(self, batch_ops, csr_tree, device):
        """Test memory efficiency of operations"""
        if device.type != 'cuda':
            pytest.skip("GPU test requires CUDA")
            
        # Large batch operation
        batch_size = 1024
        node_indices = torch.arange(min(batch_size, csr_tree.num_nodes), device=device)
        
        # Measure memory before
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()
        
        # Run batch operation
        actions, scores = batch_ops.batch_select_ucb(
            csr_tree,
            node_indices,
            c_puct=1.4
        )
        
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated()
        
        # Memory increase should be minimal
        mem_increase = mem_after - mem_before
        expected_increase = batch_size * 8  # 4 bytes for action + 4 for score
        
        # Allow some overhead but should be close
        assert mem_increase < expected_increase * 2
    
    def test_performance_scaling(self, device):
        """Test performance scaling with batch size"""
        if device.type != 'cuda':
            pytest.skip("GPU test requires CUDA")
            
        import time
        
        # Create large tree
        config = CSRTreeConfig(
            max_nodes=100000,
            device=device.type
        )
        tree = CSRTree(config)
        
        # Add many nodes
        root = tree.add_root()
        for i in range(1000):
            tree.add_child(root, i, 0.001)
        
        batch_ops = CSRBatchOperations(device)
        
        # Test different batch sizes
        batch_sizes = [32, 128, 512, 1024]
        times = []
        
        for batch_size in batch_sizes:
            node_indices = torch.arange(min(batch_size, 1000), device=device)
            
            # Warm up
            batch_ops.batch_select_ucb(tree, node_indices)
            torch.cuda.synchronize()
            
            # Time execution
            start = time.time()
            for _ in range(100):
                batch_ops.batch_select_ucb(tree, node_indices)
            torch.cuda.synchronize()
            end = time.time()
            
            times.append(end - start)
        
        # Performance should scale sub-linearly with batch size
        # Larger batches should be more efficient per element
        efficiency = [times[i] / batch_sizes[i] for i in range(len(times))]
        
        # Efficiency should improve with larger batches
        for i in range(1, len(efficiency)):
            assert efficiency[i] <= efficiency[i-1] * 1.5  # Allow some variance


@pytest.mark.benchmark
class TestCSRPerformanceBenchmarks:
    """Performance benchmarks for CSR operations"""
    
    def test_ucb_throughput(self, benchmark, device):
        """Benchmark UCB selection throughput"""
        if device.type != 'cuda':
            pytest.skip("GPU benchmark requires CUDA")
            
        # Setup
        config = CSRTreeConfig(max_nodes=50000, device=device.type)
        tree = CSRTree(config)
        
        # Create tree with many nodes
        root = tree.add_root()
        for i in range(5000):
            child = tree.add_child(root, i % 361, 0.01)
            tree.visit_counts[child] = i % 100
            tree.value_sums[child] = (i % 100) * 0.5
        
        batch_ops = CSRBatchOperations(device)
        node_indices = torch.arange(1000, device=device)
        
        # Benchmark
        def run_ucb():
            actions, scores = batch_ops.batch_select_ucb(tree, node_indices)
            torch.cuda.synchronize()
            return actions
        
        result = benchmark(run_ucb)
        
        # Should achieve high throughput
        # Target: >100k selections/second
        selections_per_second = 1000 / benchmark.stats['mean']
        assert selections_per_second > 100000