"""
Comprehensive tests for MCTS GPU acceleration

This module tests the MCTSGPUAccelerator class which provides:
- Hardware-accelerated MCTS operations
- Automatic fallback to PyTorch implementations
- Process-safe global instance management
- UCB selection, value backup, and tree expansion
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import os
import multiprocessing

from mcts.gpu.mcts_gpu_accelerator import (
    MCTSGPUAccelerator, 
    ConsolidatedKernelInterface,
    PyTorchKernelInterface,
    get_mcts_gpu_accelerator,
    get_global_accelerator,
    validate_mcts_accelerator
)
from conftest import assert_tensor_equal


class TestMCTSGPUAcceleratorInitialization:
    """Test MCTSGPUAccelerator initialization"""
    
    def test_basic_initialization_cpu(self):
        """Test basic initialization on CPU"""
        device = torch.device('cpu')
        accelerator = MCTSGPUAccelerator(device)
        
        assert accelerator.device == device
        assert accelerator._kernel_interface is not None
        assert isinstance(accelerator._kernel_interface, PyTorchKernelInterface)
        
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_initialization_cuda(self):
        """Test initialization on CUDA device"""
        device = torch.device('cuda')
        accelerator = MCTSGPUAccelerator(device)
        
        assert accelerator.device == device
        assert accelerator._kernel_interface is not None
        # Could be either ConsolidatedKernelInterface or PyTorchKernelInterface
        
    def test_default_device_selection(self):
        """Test default device selection"""
        accelerator = MCTSGPUAccelerator()
        
        expected_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert accelerator.device == expected_device
        
    def test_cuda_disabled_by_environment(self):
        """Test CUDA kernels disabled by environment variable"""
        # Save original value
        original_value = os.environ.get('DISABLE_CUDA_KERNELS')
        
        try:
            # Disable CUDA kernels
            os.environ['DISABLE_CUDA_KERNELS'] = '1'
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            accelerator = MCTSGPUAccelerator(device)
            
            # Should use PyTorch fallback
            assert isinstance(accelerator._kernel_interface, PyTorchKernelInterface)
        finally:
            # Restore original value
            if original_value is None:
                os.environ.pop('DISABLE_CUDA_KERNELS', None)
            else:
                os.environ['DISABLE_CUDA_KERNELS'] = original_value


class TestKernelLoading:
    """Test kernel loading logic"""
    
    @patch('mcts.gpu.mcts_gpu_accelerator.CUDA_MANAGER_AVAILABLE', True)
    @patch('mcts.gpu.mcts_gpu_accelerator.detect_cuda_kernels')
    def test_cuda_kernel_loading_success(self, mock_detect):
        """Test successful CUDA kernel loading"""
        # Mock successful kernel detection
        mock_kernels = Mock()
        mock_detect.return_value = mock_kernels
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        accelerator = MCTSGPUAccelerator(device)
        
        if device.type == 'cuda':
            assert isinstance(accelerator._kernel_interface, ConsolidatedKernelInterface)
            assert accelerator._kernel_interface.kernels == mock_kernels
        else:
            # On CPU, should use PyTorch fallback
            assert isinstance(accelerator._kernel_interface, PyTorchKernelInterface)
            
    @patch('mcts.gpu.mcts_gpu_accelerator.CUDA_MANAGER_AVAILABLE', True)
    @patch('mcts.gpu.mcts_gpu_accelerator.detect_cuda_kernels')
    def test_cuda_kernel_loading_failure(self, mock_detect):
        """Test fallback when CUDA kernel loading fails"""
        # Mock failed kernel detection
        mock_detect.side_effect = Exception("Kernel loading error")
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        accelerator = MCTSGPUAccelerator(device)
        
        # Should fallback to PyTorch
        assert isinstance(accelerator._kernel_interface, PyTorchKernelInterface)
        
    @patch('mcts.gpu.mcts_gpu_accelerator.CUDA_MANAGER_AVAILABLE', False)
    def test_no_cuda_manager_available(self):
        """Test when CUDA manager is not available"""
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        accelerator = MCTSGPUAccelerator(device)
        
        # Should use PyTorch fallback
        assert isinstance(accelerator._kernel_interface, PyTorchKernelInterface)


class TestBatchedUCBSelection:
    """Test batched UCB selection"""
    
    def test_pytorch_ucb_selection(self, device):
        """Test PyTorch UCB selection implementation"""
        accelerator = MCTSGPUAccelerator(device)
        # Force PyTorch implementation
        accelerator._kernel_interface = PyTorchKernelInterface(device)
        
        # Setup test data
        num_nodes = 3
        num_children = 5
        
        # CSR structure
        row_ptr = torch.tensor([0, 3, 5, 8], device=device, dtype=torch.int32)
        col_indices = torch.tensor([1, 2, 3, 4, 5, 6, 7], device=device, dtype=torch.int32)
        
        # Node data
        q_values = torch.zeros(8, device=device)
        q_values[1:4] = torch.tensor([0.5, -0.2, 0.1])  # Children of node 0
        q_values[4:6] = torch.tensor([0.3, 0.3])  # Children of node 1
        q_values[6:8] = torch.tensor([0.0, 0.4])  # Children of node 2
        
        visit_counts = torch.zeros(8, device=device, dtype=torch.int32)
        visit_counts[1:4] = torch.tensor([10, 5, 0])  # Varied visits
        
        parent_visits = torch.tensor([15, 8, 12], device=device, dtype=torch.int32)
        
        priors = torch.tensor([0.4, 0.3, 0.3, 0.5, 0.5, 0.7, 0.3], device=device)
        
        c_puct = 1.4
        
        # Run selection
        selected_actions, selected_scores = accelerator.batched_ucb_selection(
            q_values, visit_counts, parent_visits, priors, row_ptr, col_indices, c_puct
        )
        
        assert selected_actions.shape == (num_nodes,)
        assert selected_scores.shape == (num_nodes,)
        
        # Should select valid actions
        assert all(0 <= action < 3 for action in selected_actions if action >= 0)
        
    def test_ucb_selection_no_children(self, device):
        """Test UCB selection when nodes have no children"""
        accelerator = MCTSGPUAccelerator(device)
        accelerator._kernel_interface = PyTorchKernelInterface(device)
        
        # Empty CSR structure
        row_ptr = torch.tensor([0, 0, 0], device=device, dtype=torch.int32)
        col_indices = torch.empty(0, device=device, dtype=torch.int32)
        
        q_values = torch.empty(0, device=device)
        visit_counts = torch.empty(0, device=device, dtype=torch.int32)
        parent_visits = torch.tensor([10, 20], device=device, dtype=torch.int32)
        priors = torch.empty(0, device=device)
        
        selected_actions, selected_scores = accelerator.batched_ucb_selection(
            q_values, visit_counts, parent_visits, priors, row_ptr, col_indices, 1.4
        )
        
        # Should return -1 for nodes with no children
        assert torch.all(selected_actions == -1)
        assert torch.all(selected_scores == 0.0)


class TestVectorizedBackup:
    """Test vectorized backup operations"""
    
    def test_pytorch_backup_basic(self, device):
        """Test basic PyTorch backup implementation"""
        accelerator = MCTSGPUAccelerator(device)
        accelerator._kernel_interface = PyTorchKernelInterface(device)
        
        # Setup test data
        batch_size = 2
        max_depth = 4
        
        paths = torch.tensor([
            [0, 1, 3, -1],  # Path: 0 -> 1 -> 3
            [0, 2, 4, 5]    # Path: 0 -> 2 -> 4 -> 5
        ], device=device, dtype=torch.int32)
        
        path_lengths = torch.tensor([3, 4], device=device, dtype=torch.int32)
        values = torch.tensor([0.8, -0.6], device=device)
        
        # Initialize visit counts and value sums
        num_nodes = 6
        visit_counts = torch.zeros(num_nodes, device=device, dtype=torch.int32)
        value_sums = torch.zeros(num_nodes, device=device)
        
        # Run backup
        accelerator.vectorized_backup(paths, path_lengths, values, visit_counts, value_sums)
        
        # Check updates for path 1 (value = 0.8)
        # Depth 0: +0.8, Depth 1: -0.8, Depth 2: +0.8
        assert visit_counts[0] == 2  # Both paths
        assert visit_counts[1] == 1
        assert visit_counts[3] == 1
        
        assert abs(value_sums[0].item() - (0.8 + (-0.6))) < 1e-6  # Both paths
        assert abs(value_sums[1].item() - (-0.8)) < 1e-6
        assert abs(value_sums[3].item() - 0.8) < 1e-6
        
        # Check updates for path 2 (value = -0.6)
        assert visit_counts[2] == 1
        assert visit_counts[4] == 1
        assert visit_counts[5] == 1
        
        assert abs(value_sums[2].item() - 0.6) < 1e-6  # Sign flipped
        assert abs(value_sums[4].item() - (-0.6)) < 1e-6
        assert abs(value_sums[5].item() - 0.6) < 1e-6
        
    def test_backup_with_invalid_nodes(self, device):
        """Test backup handles invalid nodes gracefully"""
        accelerator = MCTSGPUAccelerator(device)
        accelerator._kernel_interface = PyTorchKernelInterface(device)
        
        paths = torch.tensor([[0, 999, -1]], device=device, dtype=torch.int32)
        path_lengths = torch.tensor([2], device=device, dtype=torch.int32)
        values = torch.tensor([0.5], device=device)
        
        visit_counts = torch.zeros(10, device=device, dtype=torch.int32)
        value_sums = torch.zeros(10, device=device)
        
        # Should not crash
        accelerator.vectorized_backup(paths, path_lengths, values, visit_counts, value_sums)
        
        # Only valid node should be updated
        assert visit_counts[0] == 1
        assert abs(value_sums[0].item() - 0.5) < 1e-6


class TestFindExpansionNodes:
    """Test finding nodes that need expansion"""
    
    def test_find_expansion_nodes_basic(self, device):
        """Test basic expansion node finding"""
        accelerator = MCTSGPUAccelerator(device)
        
        # Setup test data
        wave_size = 5
        max_children = 3
        
        current_nodes = torch.tensor([0, 1, 2, 3, 4], device=device, dtype=torch.int32)
        
        # Children array (-1 means no child)
        children = torch.tensor([
            1, 2, -1,   # Node 0 has children 1, 2
            -1, -1, -1, # Node 1 has no children
            3, -1, -1,  # Node 2 has child 3
            -1, -1, -1, # Node 3 has no children
            -1, -1, -1  # Node 4 has no children
        ], device=device, dtype=torch.int32)
        
        visit_counts = torch.tensor([10, 0, 5, 0, 3], device=device, dtype=torch.int32)
        valid_path_mask = torch.tensor([True, True, True, True, False], device=device)
        
        # Find expansion nodes
        expansion_nodes = accelerator.find_expansion_nodes(
            current_nodes, children, visit_counts, valid_path_mask, wave_size, max_children
        )
        
        # Nodes 1 and 3 should be expansion candidates (no children, no visits)
        expected = torch.tensor([1, 3], device=device, dtype=torch.int32)
        assert torch.all(expansion_nodes == expected)
        
    def test_find_expansion_nodes_empty(self, device):
        """Test when no nodes need expansion"""
        accelerator = MCTSGPUAccelerator(device)
        
        current_nodes = torch.tensor([0], device=device, dtype=torch.int32)
        children = torch.tensor([1, 2, 3], device=device, dtype=torch.int32)
        visit_counts = torch.tensor([10], device=device, dtype=torch.int32)
        valid_path_mask = torch.tensor([True], device=device)
        
        expansion_nodes = accelerator.find_expansion_nodes(
            current_nodes, children, visit_counts, valid_path_mask, 1, 3
        )
        
        assert len(expansion_nodes) == 0


class TestBatchedAddChildren:
    """Test batched children addition"""
    
    def test_batched_add_children_validation(self, device):
        """Test input validation for batched_add_children"""
        accelerator = MCTSGPUAccelerator(device)
        
        # Test with too few arguments
        with pytest.raises(ValueError, match="requires 18 arguments"):
            accelerator.batched_add_children(1, 2, 3)
            
        # Test with non-tensor arguments
        args = [None] * 18
        args[0] = "not a tensor"  # parent_indices should be tensor
        args[15:] = [1000, 10, 5000]  # Last 3 are integers
        
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            accelerator.batched_add_children(*args)
            
    def test_batched_add_children_no_cuda(self, device):
        """Test batched_add_children when CUDA kernels not available"""
        accelerator = MCTSGPUAccelerator(device)
        # Force PyTorch interface (no batched_add_children method)
        accelerator._kernel_interface = PyTorchKernelInterface(device)
        
        # Create dummy arguments
        args = [torch.zeros(1, device=device)] * 15
        args.extend([1000, 10, 5000])
        
        with pytest.raises(NotImplementedError, match="requires CUDA kernels"):
            accelerator.batched_add_children(*args)
            
    @patch('mcts.gpu.mcts_gpu_accelerator.ConsolidatedKernelInterface')
    def test_batched_add_children_cuda_success(self, mock_interface_class, device):
        """Test successful batched_add_children with mocked CUDA kernels"""
        # Create mock kernel interface
        mock_interface = Mock()
        mock_interface.batched_add_children = Mock(return_value=torch.tensor([1, 2, 3], device=device))
        
        accelerator = MCTSGPUAccelerator(device)
        accelerator._kernel_interface = mock_interface
        
        # Create dummy arguments
        batch_size = 3
        max_children = 5
        
        parent_indices = torch.tensor([0, 0, 1], device=device, dtype=torch.int32)
        actions = torch.zeros((batch_size, max_children), device=device, dtype=torch.int32)
        priors = torch.ones((batch_size, max_children), device=device, dtype=torch.float32) * 0.2
        num_children = torch.tensor([3, 2, 4], device=device, dtype=torch.int32)
        
        # Counter tensors
        node_counter = torch.zeros(1, device=device, dtype=torch.int32)
        edge_counter = torch.zeros(1, device=device, dtype=torch.int32)
        
        # Storage arrays
        children = torch.full((1000, max_children), -1, device=device, dtype=torch.int32)
        parent_indices_out = torch.full((1000,), -1, device=device, dtype=torch.int32)
        parent_actions_out = torch.full((1000,), -1, device=device, dtype=torch.int32)
        node_priors_out = torch.zeros(1000, device=device, dtype=torch.float32)
        visit_counts_out = torch.zeros(1000, device=device, dtype=torch.int32)
        value_sums_out = torch.zeros(1000, device=device, dtype=torch.float32)
        
        # CSR arrays
        col_indices = torch.zeros(5000, device=device, dtype=torch.int32)
        edge_actions = torch.zeros(5000, device=device, dtype=torch.int32)
        edge_priors = torch.zeros(5000, device=device, dtype=torch.float32)
        
        # Call the method
        result = accelerator.batched_add_children(
            parent_indices, actions, priors, num_children,
            node_counter, edge_counter, children,
            parent_indices_out, parent_actions_out, node_priors_out,
            visit_counts_out, value_sums_out,
            col_indices, edge_actions, edge_priors,
            1000, max_children, 5000
        )
        
        # Check result
        assert torch.all(result == torch.tensor([1, 2, 3], device=device))
        assert mock_interface.batched_add_children.called


class TestStatistics:
    """Test statistics tracking"""
    
    def test_get_stats_pytorch(self, device):
        """Test getting stats from PyTorch interface"""
        accelerator = MCTSGPUAccelerator(device)
        accelerator._kernel_interface = PyTorchKernelInterface(device)
        
        stats = accelerator.get_stats()
        
        assert 'device' in stats
        assert 'interface_type' in stats
        assert stats['interface_type'] == 'PyTorchKernelInterface'
        assert 'kernel_type' in stats
        assert stats['kernel_type'] == 'pytorch_fallback'
        
    @patch('mcts.gpu.mcts_gpu_accelerator.CUDA_MANAGER_AVAILABLE', True)
    @patch('mcts.gpu.mcts_gpu_accelerator.get_cuda_manager')
    def test_get_stats_cuda(self, mock_get_manager, device):
        """Test getting stats from CUDA interface"""
        # Mock CUDA manager
        mock_manager = Mock()
        mock_manager.get_kernel_info.return_value = {
            'kernel_type': 'consolidated_cuda',
            'compile_time': 123.45
        }
        mock_get_manager.return_value = mock_manager
        
        # Create accelerator with mocked CUDA interface
        mock_kernels = Mock()
        mock_interface = ConsolidatedKernelInterface(mock_kernels)
        
        accelerator = MCTSGPUAccelerator(device)
        accelerator._kernel_interface = mock_interface
        
        stats = accelerator.get_stats()
        
        assert 'kernel_type' in stats
        assert stats['kernel_type'] == 'consolidated_cuda'
        assert 'compile_time' in stats
        
    def test_reset_stats(self, device):
        """Test resetting statistics"""
        accelerator = MCTSGPUAccelerator(device)
        
        # Add some stats
        accelerator.stats = {
            'ucb_calls': 10,
            'backup_calls': 5,
            'expansion_calls': 3,
            'total_time': 1.5
        }
        
        # Reset
        accelerator.reset_stats()
        
        # Stats should be reset
        if hasattr(accelerator, 'stats'):
            assert accelerator.stats['ucb_calls'] == 0
            assert accelerator.stats['backup_calls'] == 0


class TestGlobalInstance:
    """Test global instance management"""
    
    def test_get_mcts_gpu_accelerator(self, device):
        """Test getting global accelerator instance"""
        acc1 = get_mcts_gpu_accelerator(device)
        acc2 = get_mcts_gpu_accelerator(device)
        
        # Should return same instance in same process
        assert acc1 is acc2
        
    def test_get_global_accelerator(self, device):
        """Test get_global_accelerator alias"""
        acc1 = get_global_accelerator()
        acc2 = get_mcts_gpu_accelerator()
        
        assert acc1 is acc2
        
    def test_process_isolation(self, device):
        """Test that different processes get different instances"""
        # This test uses multiprocessing to verify process isolation
        def get_accelerator_id(queue):
            """Get accelerator instance ID in subprocess"""
            acc = get_mcts_gpu_accelerator(device)
            queue.put(id(acc))
            
        # Get ID in main process
        main_acc = get_mcts_gpu_accelerator(device)
        main_id = id(main_acc)
        
        # Get ID in subprocess
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=get_accelerator_id, args=(queue,))
        process.start()
        process.join()
        
        subprocess_id = queue.get()
        
        # Should be different instances
        assert main_id != subprocess_id
        
    def test_validate_mcts_accelerator(self, device):
        """Test accelerator validation"""
        # Should succeed
        assert validate_mcts_accelerator() == True
        
        # Test with exception
        with patch('mcts.gpu.mcts_gpu_accelerator.get_mcts_gpu_accelerator', 
                   side_effect=Exception("Test error")):
            assert validate_mcts_accelerator() == False


class TestQuantumUCBSelection:
    """Test quantum UCB selection (currently disabled)"""
    
    def test_quantum_ucb_fallback(self, device):
        """Test that quantum UCB falls back to classical UCB"""
        accelerator = MCTSGPUAccelerator(device)
        accelerator._kernel_interface = PyTorchKernelInterface(device)
        
        # Setup test data (same as classical UCB test)
        row_ptr = torch.tensor([0, 2], device=device, dtype=torch.int32)
        col_indices = torch.tensor([1, 2], device=device, dtype=torch.int32)
        q_values = torch.tensor([0.0, 0.5, -0.2], device=device)
        visit_counts = torch.tensor([0, 10, 5], device=device, dtype=torch.int32)
        parent_visits = torch.tensor([15], device=device, dtype=torch.int32)
        priors = torch.tensor([0.6, 0.4], device=device)
        
        # Call quantum UCB
        quantum_actions, quantum_scores = accelerator.quantum_ucb_selection(
            q_values, visit_counts, parent_visits, priors, row_ptr, col_indices, 1.4
        )
        
        # Call classical UCB
        classical_actions, classical_scores = accelerator.batched_ucb_selection(
            q_values, visit_counts, parent_visits, priors, row_ptr, col_indices, 1.4
        )
        
        # Should be identical (quantum is disabled)
        assert torch.all(quantum_actions == classical_actions)
        assert torch.all(quantum_scores == classical_scores)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_batch_ucb_selection(self, device):
        """Test UCB selection with empty batch"""
        accelerator = MCTSGPUAccelerator(device)
        accelerator._kernel_interface = PyTorchKernelInterface(device)
        
        # Empty data
        row_ptr = torch.tensor([0], device=device, dtype=torch.int32)
        col_indices = torch.empty(0, device=device, dtype=torch.int32)
        q_values = torch.empty(0, device=device)
        visit_counts = torch.empty(0, device=device, dtype=torch.int32)
        parent_visits = torch.empty(0, device=device, dtype=torch.int32)
        priors = torch.empty(0, device=device)
        
        # Should handle gracefully
        actions, scores = accelerator.batched_ucb_selection(
            q_values, visit_counts, parent_visits, priors, row_ptr, col_indices, 1.4
        )
        
        assert actions.shape == (0,)
        assert scores.shape == (0,)
        
    def test_device_mismatch_handling(self):
        """Test handling of device mismatches"""
        if torch.cuda.is_available():
            cpu_acc = MCTSGPUAccelerator(torch.device('cpu'))
            cuda_acc = MCTSGPUAccelerator(torch.device('cuda'))
            
            # Both should work independently
            assert cpu_acc.device.type == 'cpu'
            assert cuda_acc.device.type == 'cuda'
            
    def test_dtype_error_handling(self, device):
        """Test helpful error messages for dtype mismatches"""
        accelerator = MCTSGPUAccelerator(device)
        
        # Mock CUDA interface that raises dtype error
        mock_interface = Mock()
        mock_interface.batched_add_children = Mock(
            side_effect=RuntimeError("expected scalar type Int but found Float")
        )
        accelerator._kernel_interface = mock_interface
        
        # Create arguments with wrong dtypes
        args = [torch.zeros(1, device=device, dtype=torch.float32)] * 15
        args.extend([1000, 10, 5000])
        
        with pytest.raises(RuntimeError, match="Type mismatch"):
            accelerator.batched_add_children(*args)


@pytest.mark.slow
class TestPerformance:
    """Performance tests"""
    
    def test_large_batch_ucb_selection(self, device):
        """Test UCB selection with large batch"""
        accelerator = MCTSGPUAccelerator(device)
        accelerator._kernel_interface = PyTorchKernelInterface(device)
        
        # Large batch
        num_nodes = 1000
        avg_children = 50
        total_edges = num_nodes * avg_children
        
        # Build CSR structure
        row_ptr = torch.arange(0, total_edges + 1, avg_children, 
                              device=device, dtype=torch.int32)[:num_nodes + 1]
        col_indices = torch.arange(total_edges, device=device, dtype=torch.int32)
        
        # Random data
        q_values = torch.rand(total_edges, device=device) * 2 - 1
        visit_counts = torch.randint(0, 100, (total_edges,), 
                                    device=device, dtype=torch.int32)
        parent_visits = torch.randint(10, 1000, (num_nodes,), 
                                     device=device, dtype=torch.int32)
        priors = torch.rand(total_edges, device=device)
        
        # Time the operation
        import time
        start = time.time()
        actions, scores = accelerator.batched_ucb_selection(
            q_values, visit_counts, parent_visits, priors, row_ptr, col_indices, 1.4
        )
        elapsed = time.time() - start
        
        # Should complete reasonably fast
        assert elapsed < 0.5  # 500ms for 1000 nodes
        assert actions.shape == (num_nodes,)
        assert scores.shape == (num_nodes,)
        
    def test_large_batch_backup(self, device):
        """Test backup with large batch"""
        accelerator = MCTSGPUAccelerator(device)
        accelerator._kernel_interface = PyTorchKernelInterface(device)
        
        # Large batch
        batch_size = 256
        max_depth = 50
        num_nodes = 10000
        
        # Random paths
        paths = torch.randint(-1, num_nodes, (batch_size, max_depth), 
                             device=device, dtype=torch.int32)
        # Ensure valid paths
        for i in range(batch_size):
            valid_length = torch.randint(10, max_depth, (1,)).item()
            paths[i, valid_length:] = -1
            
        path_lengths = torch.randint(10, max_depth, (batch_size,), 
                                    device=device, dtype=torch.int32)
        values = torch.rand(batch_size, device=device) * 2 - 1
        
        visit_counts = torch.zeros(num_nodes, device=device, dtype=torch.int32)
        value_sums = torch.zeros(num_nodes, device=device)
        
        # Time the operation
        import time
        start = time.time()
        accelerator.vectorized_backup(paths, path_lengths, values, 
                                     visit_counts, value_sums)
        elapsed = time.time() - start
        
        # Should complete reasonably fast
        assert elapsed < 1.0  # 1 second for large batch
        
        # Some nodes should be updated
        assert visit_counts.sum() > 0
        assert value_sums.abs().sum() > 0