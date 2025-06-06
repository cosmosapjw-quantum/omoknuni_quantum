"""Tests for GPU-accelerated tree operations

This module tests the GPU kernels for tree operations which are critical
for MCTS performance optimization.
"""

import pytest
import torch
import numpy as np
from typing import List, Tuple
import time

from mcts.cuda_kernels import CUDAKernels, create_cuda_kernels, CUDA_AVAILABLE


class TestGPUTreeOperations:
    """Test GPU-accelerated tree operations"""
    
    @pytest.fixture
    def tree_structure(self):
        """Create a sample tree structure for testing"""
        # Tree with known structure:
        # Root (0)
        # ├── Node 1
        # │   ├── Node 3
        # │   └── Node 4
        # └── Node 2
        #     ├── Node 5
        #     └── Node 6
        
        parent_indices = torch.tensor([-1, 0, 0, 1, 1, 2, 2])  # -1 for root
        child_counts = torch.tensor([2, 2, 2, 0, 0, 0, 0])
        
        # Flattened children array (padded with -1)
        max_children = 2
        children = torch.tensor([
            [1, 2],      # Root's children
            [3, 4],      # Node 1's children
            [5, 6],      # Node 2's children
            [-1, -1],    # Node 3 (leaf)
            [-1, -1],    # Node 4 (leaf)
            [-1, -1],    # Node 5 (leaf)
            [-1, -1],    # Node 6 (leaf)
        ])
        
        return {
            'parent_indices': parent_indices,
            'child_counts': child_counts,
            'children': children,
            'n_nodes': 7,
            'max_children': max_children
        }
        
    def test_batched_child_gathering(self, tree_structure):
        """Test gathering children for multiple nodes in parallel"""
        kernels = create_cuda_kernels()
        
        # Test gathering children for nodes [0, 1, 2]
        node_indices = torch.tensor([0, 1, 2])
        
        # Expected children
        expected = torch.tensor([
            [1, 2],
            [3, 4], 
            [5, 6]
        ])
        
        # Implement child gathering
        gathered = self._gather_children(
            node_indices, 
            tree_structure['children'],
            kernels.device
        )
        
        assert torch.allclose(gathered, expected.to(gathered.device))
        
    def test_parallel_leaf_finding(self, tree_structure):
        """Test finding leaf nodes from multiple starting points"""
        kernels = create_cuda_kernels()
        
        # Start from different nodes
        start_nodes = torch.tensor([0, 1, 2, 3])
        
        # Expected leaf nodes reached
        # From 0: could reach any leaf (3, 4, 5, 6)
        # From 1: could reach 3 or 4
        # From 2: could reach 5 or 6
        # From 3: already a leaf
        
        leaves = self._find_leaves_parallel(
            start_nodes,
            tree_structure['children'],
            tree_structure['child_counts'],
            kernels.device
        )
        
        # Check that node 3 returns itself (it's a leaf)
        assert leaves[3] == 3
        
        # Check others reach valid leaves
        assert leaves[0] in [3, 4, 5, 6]
        assert leaves[1] in [3, 4]
        assert leaves[2] in [5, 6]
        
    def test_vectorized_path_backup(self, tree_structure):
        """Test backing up values along multiple paths simultaneously"""
        kernels = create_cuda_kernels()
        
        # Define paths from leaves to root
        paths = torch.tensor([
            [3, 1, 0, -1],  # Leaf 3 -> Node 1 -> Root
            [4, 1, 0, -1],  # Leaf 4 -> Node 1 -> Root
            [5, 2, 0, -1],  # Leaf 5 -> Node 2 -> Root
            [6, 2, 0, -1],  # Leaf 6 -> Node 2 -> Root
        ])
        
        # Values to backup
        values = torch.tensor([1.0, -0.5, 0.8, -0.3])
        
        # Current statistics
        n_nodes = tree_structure['n_nodes']
        visit_counts = torch.zeros(n_nodes)
        value_sums = torch.zeros(n_nodes)
        
        # Perform backup
        new_visits, new_values = self._vectorized_backup(
            paths, values, visit_counts, value_sums, kernels.device
        )
        
        # Verify updates
        # Root should be updated 4 times
        assert new_visits[0] == 4
        # Nodes 1 and 2 should each be updated 2 times
        assert new_visits[1] == 2
        assert new_visits[2] == 2
        # Leaves should each be updated once
        assert torch.all(new_visits[3:7] == 1)
        
    def test_batch_ucb_with_masking(self, tree_structure):
        """Test UCB computation with action masking"""
        kernels = create_cuda_kernels()
        
        # UCB components for children of root (nodes 1 and 2)
        q_values = torch.tensor([0.5, -0.2])
        visit_counts = torch.tensor([10.0, 5.0])
        parent_visits = torch.tensor([15.0, 15.0])  # Root has 15 visits
        priors = torch.tensor([0.6, 0.4])
        
        # Compute UCB scores
        ucb_scores = kernels.compute_batched_ucb(
            q_values, visit_counts, parent_visits, priors, c_puct=1.0
        )
        
        # Manual calculation for verification
        sqrt_15 = np.sqrt(15.0)
        expected_ucb1 = 0.5 + 1.0 * 0.6 * sqrt_15 / (1 + 10)
        expected_ucb2 = -0.2 + 1.0 * 0.4 * sqrt_15 / (1 + 5)
        
        assert abs(ucb_scores[0].item() - expected_ucb1) < 1e-5
        assert abs(ucb_scores[1].item() - expected_ucb2) < 1e-5
        
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_gpu_tree_traversal_performance(self, tree_structure):
        """Benchmark GPU vs CPU tree traversal"""
        cpu_kernels = CUDAKernels(device=torch.device('cpu'))
        gpu_kernels = CUDAKernels(device=torch.device('cuda'))
        
        # Create larger tree for performance testing
        n_nodes = 10000
        n_paths = 1000
        max_depth = 20
        
        # Random paths through tree
        paths = torch.randint(0, n_nodes, (n_paths, max_depth))
        values = torch.rand(n_paths)
        visit_counts = torch.rand(n_nodes) * 100
        value_sums = torch.rand(n_nodes)
        
        # CPU timing
        cpu_paths = paths.to('cpu')
        cpu_values = values.to('cpu')
        cpu_visits = visit_counts.to('cpu')
        cpu_sums = value_sums.to('cpu')
        
        start = time.time()
        for _ in range(10):
            _ = self._vectorized_backup(
                cpu_paths, cpu_values, cpu_visits, cpu_sums, 'cpu'
            )
        cpu_time = time.time() - start
        
        # GPU timing
        gpu_paths = paths.to('cuda')
        gpu_values = values.to('cuda')
        gpu_visits = visit_counts.to('cuda')
        gpu_sums = value_sums.to('cuda')
        
        # Warmup
        _ = self._vectorized_backup(
            gpu_paths, gpu_values, gpu_visits, gpu_sums, 'cuda'
        )
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(10):
            _ = self._vectorized_backup(
                gpu_paths, gpu_values, gpu_visits, gpu_sums, 'cuda'
            )
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        speedup = cpu_time / gpu_time
        print(f"\nGPU Speedup for tree traversal: {speedup:.2f}x")
        print(f"CPU time: {cpu_time:.4f}s, GPU time: {gpu_time:.4f}s")
        
        # GPU should be faster for large batches
        assert speedup > 1.0
        
    def _gather_children(
        self, 
        node_indices: torch.Tensor,
        children: torch.Tensor,
        device: str
    ) -> torch.Tensor:
        """Gather children for multiple nodes"""
        node_indices = node_indices.to(device)
        children = children.to(device)
        
        # Index into children array
        return children[node_indices]
        
    def _find_leaves_parallel(
        self,
        start_nodes: torch.Tensor,
        children: torch.Tensor,
        child_counts: torch.Tensor,
        device: str
    ) -> torch.Tensor:
        """Find leaf nodes from multiple starting points"""
        start_nodes = start_nodes.to(device)
        children = children.to(device)
        child_counts = child_counts.to(device)
        
        current_nodes = start_nodes.clone()
        
        # Simple traversal - in practice would be more sophisticated
        for _ in range(10):  # Max depth
            # Check which nodes have children
            has_children = child_counts[current_nodes] > 0
            
            # For nodes with children, pick first child
            # In real MCTS, this would use UCB selection
            new_nodes = current_nodes.clone()
            if has_children.any():
                nodes_to_expand = current_nodes[has_children]
                first_children = children[nodes_to_expand, 0]
                new_nodes[has_children] = first_children
                
            # Check if we've converged
            if torch.equal(current_nodes, new_nodes):
                break
                
            current_nodes = new_nodes
            
        return current_nodes
        
    def _vectorized_backup(
        self,
        paths: torch.Tensor,
        values: torch.Tensor,
        visit_counts: torch.Tensor,
        value_sums: torch.Tensor,
        device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backup values along multiple paths"""
        paths = paths.to(device)
        values = values.to(device) 
        visit_counts = visit_counts.to(device).clone()
        value_sums = value_sums.to(device).clone()
        
        # Process each path
        for path_idx in range(paths.shape[0]):
            value = values[path_idx]
            path = paths[path_idx]
            
            # Backup along path
            for node_idx in path:
                if node_idx < 0:
                    break
                    
                visit_counts[node_idx] += 1
                value_sums[node_idx] += value
                
                # Negate for opponent
                value = -value
                
        return visit_counts, value_sums


class TestBatchedUCBKernel:
    """Specific tests for the batched UCB kernel"""
    
    def test_ucb_computation_correctness(self):
        """Test UCB formula implementation"""
        kernels = create_cuda_kernels()
        
        # Test cases with known results
        test_cases = [
            {
                'q': torch.tensor([0.0]),
                'visits': torch.tensor([0.0]),
                'parent_visits': torch.tensor([1.0]),
                'prior': torch.tensor([1.0]),
                'c_puct': 1.0,
                'expected': 1.0  # 0 + 1 * 1 * 1 / 1 = 1
            },
            {
                'q': torch.tensor([0.5]),
                'visits': torch.tensor([10.0]),
                'parent_visits': torch.tensor([100.0]),
                'prior': torch.tensor([0.2]),
                'c_puct': 2.0,
                'expected': 0.5 + 2.0 * 0.2 * 10.0 / 11.0  # ≈ 0.8636
            }
        ]
        
        for case in test_cases:
            result = kernels.compute_batched_ucb(
                case['q'], case['visits'], case['parent_visits'],
                case['prior'], case['c_puct']
            )
            assert abs(result.item() - case['expected']) < 1e-4
            
    def test_ucb_batch_processing(self):
        """Test UCB computation on large batches"""
        kernels = create_cuda_kernels()
        
        batch_sizes = [100, 1000, 10000]
        
        for batch_size in batch_sizes:
            q_values = torch.rand(batch_size) - 0.5
            visit_counts = torch.randint(0, 1000, (batch_size,)).float()
            parent_visits = torch.randint(1000, 10000, (batch_size,)).float()
            priors = torch.rand(batch_size)
            priors = priors / priors.sum()
            
            ucb_scores = kernels.compute_batched_ucb(
                q_values, visit_counts, parent_visits, priors, c_puct=1.414
            )
            
            # Verify properties
            assert ucb_scores.shape == (batch_size,)
            assert torch.all(torch.isfinite(ucb_scores))
            
            # UCB should be at least as large as Q (due to exploration bonus)
            assert torch.all(ucb_scores >= q_values.to(ucb_scores.device) - 1e-6)
            
    def test_ucb_edge_cases(self):
        """Test UCB computation edge cases"""
        kernels = create_cuda_kernels()
        
        # Zero visits - should have high exploration
        q = torch.tensor([0.0, 0.0])
        visits = torch.tensor([0.0, 100.0])
        parent = torch.tensor([100.0, 100.0])
        priors = torch.tensor([0.5, 0.5])
        
        ucb = kernels.compute_batched_ucb(q, visits, parent, priors, c_puct=1.0)
        
        # Node with zero visits should have higher UCB
        assert ucb[0] > ucb[1]
        
        # Very high visit counts - exploration should be minimal
        high_visits = torch.tensor([1e6, 1e6])
        ucb_high = kernels.compute_batched_ucb(
            q, high_visits, parent, priors, c_puct=1.0
        )
        
        # Should be close to Q-values
        assert torch.allclose(ucb_high, q.to(ucb_high.device), atol=1e-3)