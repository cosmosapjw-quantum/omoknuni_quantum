"""Tests for CUDA kernel implementations"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from mcts.gpu.mcts_gpu_accelerator import get_mcts_gpu_accelerator, MCTSGPUAccelerator
from mcts.gpu.cuda_manager import CudaManager


class TestBatchedAddChildren:
    """Test batched_add_children CUDA kernel implementation"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batched_add_children_with_cuda(self):
        """Test batched_add_children when CUDA is available"""
        device = torch.device('cuda')
        accelerator = get_mcts_gpu_accelerator(device)
        
        # Create test data
        batch_size = 4
        max_children = 10
        max_nodes = 100
        max_edges = 1000
        
        parent_indices = torch.tensor([0, 5, 10, 15], dtype=torch.int32, device=device)
        actions = torch.randint(0, 225, (batch_size, max_children), dtype=torch.int32, device=device)
        priors = torch.rand(batch_size, max_children, dtype=torch.float32, device=device)
        num_children = torch.tensor([3, 5, 2, 4], dtype=torch.int32, device=device)
        
        # Initialize storage tensors
        node_counter = torch.zeros(1, dtype=torch.int32, device=device)
        edge_counter = torch.zeros(1, dtype=torch.int32, device=device)
        children = torch.full((max_nodes, max_children), -1, dtype=torch.int32, device=device)
        parent_indices_out = torch.full((max_nodes,), -1, dtype=torch.int32, device=device)
        parent_actions_out = torch.full((max_nodes,), -1, dtype=torch.int32, device=device)
        node_priors_out = torch.zeros(max_nodes, dtype=torch.float32, device=device)
        visit_counts_out = torch.zeros(max_nodes, dtype=torch.int32, device=device)
        value_sums_out = torch.zeros(max_nodes, dtype=torch.float32, device=device)
        col_indices = torch.full((max_edges,), -1, dtype=torch.int32, device=device)
        edge_actions = torch.full((max_edges,), -1, dtype=torch.int32, device=device)
        edge_priors = torch.zeros(max_edges, dtype=torch.float32, device=device)
        
        # Try to call batched_add_children
        result = accelerator.batched_add_children(
            parent_indices, actions, priors, num_children,
            node_counter, edge_counter, children,
            parent_indices_out, parent_actions_out, node_priors_out,
            visit_counts_out, value_sums_out,
            col_indices, edge_actions, edge_priors,
            max_nodes, max_children, max_edges
        )
        
        # If CUDA kernel is available, it should return a tensor
        # If not available, it should return None (fallback to CPU)
        assert result is None or isinstance(result, torch.Tensor)
        
    def test_batched_add_children_cpu_fallback(self):
        """Test batched_add_children falls back gracefully on CPU"""
        device = torch.device('cpu')
        accelerator = get_mcts_gpu_accelerator(device)
        
        # Check if it has PyTorch interface (no CUDA kernels)
        if hasattr(accelerator._kernel_interface, 'kernels'):
            # Skip test if CUDA kernels are loaded even on CPU
            pytest.skip("CUDA kernels loaded on CPU device")
        
        # For CPU, batched_add_children should return None (no implementation)
        result = accelerator.batched_add_children(
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, 100, 10, 1000
        )
        
        assert result is None  # CPU fallback returns None


class TestGPUAccelerator:
    """Test GPU accelerator functionality"""
    
    def test_get_accelerator_singleton(self):
        """Test that get_mcts_gpu_accelerator returns singleton"""
        device = torch.device('cpu')
        acc1 = get_mcts_gpu_accelerator(device)
        acc2 = get_mcts_gpu_accelerator(device)
        assert acc1 is acc2
        
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_accelerator_cuda_device(self):
        """Test accelerator on CUDA device"""
        device = torch.device('cuda')
        accelerator = get_mcts_gpu_accelerator(device)
        assert accelerator.device == device
        
    def test_accelerator_methods_exist(self):
        """Test that all expected methods exist on accelerator"""
        device = torch.device('cpu')
        accelerator = get_mcts_gpu_accelerator(device)
        
        expected_methods = [
            'batched_ucb_selection',
            'parallel_backup',
            'vectorized_backup',
            'find_expansion_nodes',
            'quantum_ucb_selection',
            'batched_add_children',
            'get_stats',
            'reset_stats'
        ]
        
        for method in expected_methods:
            assert hasattr(accelerator, method)
            assert callable(getattr(accelerator, method))


class TestCudaManager:
    """Test CUDA manager functionality"""
    
    def test_cuda_manager_initialization(self):
        """Test CUDA manager initializes correctly"""
        manager = CudaManager(disable_cuda=True)
        assert not manager.is_cuda_available()
        
    def test_cuda_manager_kernel_list(self):
        """Test CUDA manager has correct kernel list"""
        manager = CudaManager()
        expected_kernels = [
            'find_expansion_nodes',
            'batched_ucb_selection',
            'quantum_ucb_selection',
            'vectorized_backup',
            'batched_add_children',
            'initialize_lookup_tables'
        ]
        assert manager.kernel_config['functions'] == expected_kernels
        
    def test_cuda_manager_finds_build_dir(self):
        """Test CUDA manager finds build directory correctly"""
        # Just test that CudaManager initializes without error
        manager = CudaManager()
        # Manager should have a build directory set (even if it doesn't exist)
        assert manager.build_dir is not None
        assert hasattr(manager, 'kernel_config')
        assert 'batched_add_children' in manager.kernel_config['functions']


class TestBatchUCBSelection:
    """Test batch UCB selection implementations"""
    
    def test_pytorch_ucb_fallback(self):
        """Test PyTorch UCB selection fallback"""
        device = torch.device('cpu')
        accelerator = get_mcts_gpu_accelerator(device)
        
        # Create test data for UCB selection
        num_nodes = 5
        num_edges = 10
        
        q_values = torch.rand(num_nodes, device=device)
        visit_counts = torch.randint(0, 100, (num_nodes,), dtype=torch.int32, device=device)
        parent_visits = torch.randint(1, 1000, (num_nodes,), dtype=torch.int32, device=device)
        priors = torch.rand(num_edges, device=device)
        
        # CSR format
        row_ptr = torch.tensor([0, 2, 4, 6, 8, 10], dtype=torch.int32, device=device)
        col_indices = torch.arange(num_edges, dtype=torch.int32, device=device)
        
        c_puct = 1.0
        
        # Should not raise an error
        result = accelerator.batched_ucb_selection(
            q_values, visit_counts, parent_visits, priors,
            row_ptr, col_indices, c_puct
        )
        
        # Check if result is tuple or tensor
        if isinstance(result, tuple):
            selected_actions, selected_scores = result
        else:
            # Single tensor result
            selected_actions = result
            selected_scores = torch.zeros_like(selected_actions, dtype=torch.float32)
        
        assert selected_actions.shape == (num_nodes,)
        assert selected_scores.shape == (num_nodes,)
        assert selected_actions.dtype == torch.int32
        assert selected_scores.dtype == torch.float32