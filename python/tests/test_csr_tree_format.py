"""Test suite for CSR (Compressed Sparse Row) tree format conversion

This module tests the conversion from sparse child storage to CSR format
for GPU-friendly memory access patterns and improved cache performance.
"""

import pytest
import torch
import numpy as np
from typing import Tuple, List, Dict
import time

# Import MCTS modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts.gpu.csr_tree import CSRTree as OptimizedTensorTree, CSRTreeConfig as OptimizedTreeConfig


class CSRTreeFormat:
    """CSR (Compressed Sparse Row) format for GPU-optimized tree storage
    
    CSR format stores tree children using three arrays:
    - row_ptr: Starting index for each node's children  
    - col_indices: Child node indices (flattened)
    - values: Associated data (actions, priors, etc.)
    
    Benefits:
    - Contiguous memory access for GPU kernels
    - Coalesced memory reads in parallel processing
    - Better cache utilization vs sparse storage
    """
    
    def __init__(self, max_nodes: int, device: str = 'cuda'):
        self.max_nodes = max_nodes
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # CSR structure arrays
        self.row_ptr = torch.zeros(max_nodes + 1, dtype=torch.int32, device=self.device)
        self.col_indices = torch.zeros(0, dtype=torch.int32, device=self.device)  # Dynamic size
        self.actions = torch.zeros(0, dtype=torch.int16, device=self.device)     # Action for each edge
        self.priors = torch.zeros(0, dtype=torch.float16, device=self.device)    # Prior for each edge
        
        # Node data (unchanged from original)
        self.visit_counts = torch.zeros(max_nodes, dtype=torch.int32, device=self.device)
        self.value_sums = torch.zeros(max_nodes, dtype=torch.float16, device=self.device)
        self.node_priors = torch.zeros(max_nodes, dtype=torch.float16, device=self.device)
        self.parent_indices = torch.full((max_nodes,), -1, dtype=torch.int16, device=self.device)
        self.flags = torch.zeros(max_nodes, dtype=torch.uint8, device=self.device)
        
        self.num_nodes = 0
        self.num_edges = 0
        
    def get_children_indices(self, node_idx: int) -> torch.Tensor:
        """Get child indices for a node using CSR indexing"""
        start = self.row_ptr[node_idx].item()
        end = self.row_ptr[node_idx + 1].item()
        return self.col_indices[start:end]
        
    def get_children_actions(self, node_idx: int) -> torch.Tensor:
        """Get child actions for a node"""
        start = self.row_ptr[node_idx].item()
        end = self.row_ptr[node_idx + 1].item()
        return self.actions[start:end]
        
    def get_children_priors(self, node_idx: int) -> torch.Tensor:
        """Get child priors for a node"""
        start = self.row_ptr[node_idx].item()
        end = self.row_ptr[node_idx + 1].item()
        return self.priors[start:end]
        
    def batch_get_children(self, node_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get children for multiple nodes efficiently (GPU-optimized)"""
        batch_size = len(node_indices)
        max_children = 0
        
        # Find maximum children count for this batch
        for i in range(batch_size):
            node_idx = node_indices[i].item()
            start = self.row_ptr[node_idx].item()
            end = self.row_ptr[node_idx + 1].item()
            max_children = max(max_children, end - start)
            
        if max_children == 0:
            return (torch.empty((batch_size, 0), dtype=torch.int32, device=self.device),
                   torch.empty((batch_size, 0), dtype=torch.int16, device=self.device))
            
        # Allocate output tensors
        batch_children = torch.full((batch_size, max_children), -1, 
                                   dtype=torch.int32, device=self.device)
        batch_actions = torch.full((batch_size, max_children), -1,
                                  dtype=torch.int16, device=self.device)
        
        # Fill batch tensors
        for i in range(batch_size):
            node_idx = node_indices[i].item()
            start = self.row_ptr[node_idx].item()
            end = self.row_ptr[node_idx + 1].item()
            num_children = end - start
            
            if num_children > 0:
                batch_children[i, :num_children] = self.col_indices[start:end]
                batch_actions[i, :num_children] = self.actions[start:end]
                
        return batch_children, batch_actions


def convert_tree_to_csr(tree: OptimizedTensorTree) -> CSRTreeFormat:
    """Convert OptimizedTensorTree to CSR format
    
    Args:
        tree: Source tree in sparse format
        
    Returns:
        CSRTreeFormat: Tree converted to CSR format
    """
    csr = CSRTreeFormat(tree.config.max_nodes, tree.config.device)
    
    # Copy node data
    csr.num_nodes = tree.num_nodes
    csr.visit_counts[:tree.num_nodes] = tree.visit_counts[:tree.num_nodes]
    csr.value_sums[:tree.num_nodes] = tree.value_sums[:tree.num_nodes]
    csr.node_priors[:tree.num_nodes] = tree.priors[:tree.num_nodes]
    csr.parent_indices[:tree.num_nodes] = tree.parent_indices[:tree.num_nodes]
    csr.flags[:tree.num_nodes] = tree.flags[:tree.num_nodes]
    
    # Build CSR structure
    total_edges = 0
    
    # Pass 1: Count total edges
    for node_idx in range(tree.num_nodes):
        num_children = int(tree.num_children[node_idx])
        total_edges += num_children
        
    # Allocate CSR arrays
    csr.col_indices = torch.zeros(total_edges, dtype=torch.int32, device=tree.device)
    csr.actions = torch.zeros(total_edges, dtype=torch.int16, device=tree.device)
    csr.priors = torch.zeros(total_edges, dtype=torch.float16, device=tree.device)
    csr.num_edges = total_edges
    
    # Pass 2: Fill CSR arrays
    edge_idx = 0
    for node_idx in range(tree.num_nodes):
        csr.row_ptr[node_idx] = edge_idx
        
        num_children = int(tree.num_children[node_idx])
        if num_children > 0:
            # Get children from sparse storage
            children_indices, children_actions = tree.get_children(node_idx)
            
            # Copy to CSR arrays
            csr.col_indices[edge_idx:edge_idx + num_children] = children_indices
            csr.actions[edge_idx:edge_idx + num_children] = children_actions
            
            # Set priors (lookup from child nodes)
            for i, child_idx in enumerate(children_indices):
                csr.priors[edge_idx + i] = tree.priors[child_idx]
                
            edge_idx += num_children
            
    # Set final row pointer
    csr.row_ptr[tree.num_nodes] = edge_idx
    
    return csr


@pytest.fixture
def device():
    """Test device - use CUDA if available"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def small_tree(device):
    """Create a small test tree for basic functionality tests"""
    config = OptimizedTreeConfig(max_nodes=100, device=device)
    tree = OptimizedTensorTree(config)
    
    # Build small tree: root with 3 children, first child has 2 children
    root = tree.add_root()
    
    # Add children to root
    child1 = tree.add_child(root, action=0, prior=0.5)
    child2 = tree.add_child(root, action=1, prior=0.3)
    child3 = tree.add_child(root, action=2, prior=0.2)
    
    # Add children to first child
    grandchild1 = tree.add_child(child1, action=0, prior=0.6)
    grandchild2 = tree.add_child(child1, action=1, prior=0.4)
    
    # Update some visit counts and values for testing
    tree.visit_counts[root] = 10
    tree.value_sums[root] = 5.5
    tree.visit_counts[child1] = 6
    tree.value_sums[child1] = 3.2
    tree.visit_counts[child2] = 3
    tree.value_sums[child2] = 1.8
    tree.visit_counts[child3] = 1
    tree.value_sums[child3] = 0.5
    
    return tree


@pytest.fixture
def large_tree(device):
    """Create a larger tree for performance testing"""
    config = OptimizedTreeConfig(max_nodes=10000, device=device)
    tree = OptimizedTensorTree(config)
    
    # Build larger tree structure
    root = tree.add_root()
    
    # Add multiple levels
    nodes_by_level = [[root]]
    for level in range(3):  # 3 levels deep
        next_level = []
        for parent in nodes_by_level[level]:
            # Add 2-5 children per node
            num_children = min(5, max(2, hash(parent) % 6))
            for action in range(num_children):
                prior = 1.0 / (num_children + 1)
                child = tree.add_child(parent, action=action, prior=prior)
                next_level.append(child)
                
                # Set some random visit counts
                tree.visit_counts[child] = abs(hash(child)) % 100 + 1
                tree.value_sums[child] = float(tree.visit_counts[child]) * 0.5
                
        nodes_by_level.append(next_level)
        
        # Stop if we get too many nodes
        if len(next_level) > 1000:
            break
            
    return tree


class TestCSRConversion:
    """Test CSR format conversion functionality"""
    
    def test_empty_tree_conversion(self, device):
        """Test converting an empty tree"""
        config = OptimizedTreeConfig(max_nodes=10, device=device)
        tree = OptimizedTensorTree(config)
        tree.add_root()  # Just root node
        
        csr = convert_tree_to_csr(tree)
        
        assert csr.num_nodes == 1
        assert csr.num_edges == 0
        assert csr.row_ptr[0].item() == 0
        assert csr.row_ptr[1].item() == 0
        assert len(csr.col_indices) == 0
        assert len(csr.actions) == 0
        
    def test_small_tree_conversion(self, small_tree):
        """Test converting a small tree with known structure"""
        csr = convert_tree_to_csr(small_tree)
        
        # Check basic properties
        assert csr.num_nodes == small_tree.num_nodes
        assert csr.num_edges == 5  # 3 children of root + 2 children of child1
        
        # Check node data preservation
        torch.testing.assert_close(csr.visit_counts[:csr.num_nodes], 
                                   small_tree.visit_counts[:small_tree.num_nodes])
        torch.testing.assert_close(csr.value_sums[:csr.num_nodes],
                                   small_tree.value_sums[:small_tree.num_nodes])
        
        # Check CSR structure for root (should have 3 children)
        root_children = csr.get_children_indices(0)
        assert len(root_children) == 3
        assert set(root_children.tolist()) == {1, 2, 3}  # Child indices
        
        # Check CSR structure for first child (should have 2 children)
        child1_children = csr.get_children_indices(1)
        assert len(child1_children) == 2
        assert set(child1_children.tolist()) == {4, 5}  # Grandchild indices
        
        # Check leaf nodes have no children
        assert len(csr.get_children_indices(2)) == 0
        assert len(csr.get_children_indices(3)) == 0
        assert len(csr.get_children_indices(4)) == 0
        assert len(csr.get_children_indices(5)) == 0
        
    def test_actions_preservation(self, small_tree):
        """Test that actions are correctly preserved in CSR format"""
        csr = convert_tree_to_csr(small_tree)
        
        # Check root children actions
        root_actions = csr.get_children_actions(0)
        assert len(root_actions) == 3
        assert set(root_actions.tolist()) == {0, 1, 2}
        
        # Check first child actions
        child1_actions = csr.get_children_actions(1)
        assert len(child1_actions) == 2
        assert set(child1_actions.tolist()) == {0, 1}
        
    def test_priors_preservation(self, small_tree):
        """Test that priors are correctly preserved in CSR format"""
        csr = convert_tree_to_csr(small_tree)
        
        # Get root children priors from CSR
        root_priors = csr.get_children_priors(0)
        assert len(root_priors) == 3
        
        # Should match the priors we set: 0.5, 0.3, 0.2
        expected_priors = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float16, device=csr.device)
        torch.testing.assert_close(torch.sort(root_priors)[0], 
                                   torch.sort(expected_priors)[0], 
                                   rtol=1e-3, atol=1e-3)


class TestCSRBatchOperations:
    """Test CSR batch operations for GPU efficiency"""
    
    def test_batch_children_access(self, small_tree):
        """Test batch access to multiple nodes' children"""
        csr = convert_tree_to_csr(small_tree)
        
        # Test batch access to root and first child
        node_indices = torch.tensor([0, 1], device=csr.device)
        batch_children, batch_actions = csr.batch_get_children(node_indices)
        
        # Should return (2, max_children) tensors
        assert batch_children.shape[0] == 2  # 2 nodes
        assert batch_actions.shape[0] == 2
        
        # Check root children (first row)
        root_children = batch_children[0][batch_children[0] >= 0]  # Filter out -1 padding
        assert len(root_children) == 3
        assert set(root_children.tolist()) == {1, 2, 3}
        
        # Check first child children (second row)
        child1_children = batch_children[1][batch_children[1] >= 0]
        assert len(child1_children) == 2
        assert set(child1_children.tolist()) == {4, 5}
        
    def test_empty_nodes_batch_access(self, small_tree):
        """Test batch access including nodes with no children"""
        csr = convert_tree_to_csr(small_tree)
        
        # Access leaf nodes (should have no children)
        node_indices = torch.tensor([2, 3, 4, 5], device=csr.device)
        batch_children, batch_actions = csr.batch_get_children(node_indices)
        
        # All should be empty (filled with -1)
        assert (batch_children == -1).all()
        assert (batch_actions == -1).all()
        
    def test_mixed_batch_access(self, small_tree):
        """Test batch access with mix of nodes with/without children"""
        csr = convert_tree_to_csr(small_tree)
        
        # Mix of nodes: root (3 children), leaf (0 children), child1 (2 children)
        node_indices = torch.tensor([0, 2, 1], device=csr.device)
        batch_children, batch_actions = csr.batch_get_children(node_indices)
        
        assert batch_children.shape[0] == 3
        
        # Root should have 3 children
        root_children = batch_children[0][batch_children[0] >= 0]
        assert len(root_children) == 3
        
        # Leaf should have no children
        leaf_children = batch_children[1][batch_children[1] >= 0]
        assert len(leaf_children) == 0
        
        # Child1 should have 2 children
        child1_children = batch_children[2][batch_children[2] >= 0]
        assert len(child1_children) == 2


class TestCSRPerformance:
    """Test CSR performance characteristics"""
    
    def test_memory_layout_efficiency(self, large_tree):
        """Test that CSR format provides better memory layout"""
        csr = convert_tree_to_csr(large_tree)
        
        # CSR should have contiguous memory for all children
        assert csr.col_indices.is_contiguous()
        assert csr.actions.is_contiguous()
        assert csr.priors.is_contiguous()
        assert csr.row_ptr.is_contiguous()
        
        # All data should be on the same device (handle cuda:0 vs cuda)
        expected_device = csr.device
        assert csr.col_indices.device.type == expected_device.type
        assert csr.actions.device.type == expected_device.type
        assert csr.priors.device.type == expected_device.type
        assert csr.row_ptr.device.type == expected_device.type
        
    @pytest.mark.parametrize("batch_size", [16, 64, 256, 1024])
    def test_batch_access_scalability(self, large_tree, batch_size):
        """Test batch access with different batch sizes"""
        csr = convert_tree_to_csr(large_tree)
        
        # Create batch of random node indices
        max_node = min(batch_size, csr.num_nodes - 1)
        node_indices = torch.randint(0, max_node, (batch_size,), device=csr.device)
        
        # Time batch access
        start_time = time.perf_counter()
        batch_children, batch_actions = csr.batch_get_children(node_indices)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # Should complete in reasonable time (not fully optimized yet)
        assert end_time - start_time < 5.0  # Less than 5 seconds
        
        # Results should have correct shape
        assert batch_children.shape[0] == batch_size
        assert batch_actions.shape[0] == batch_size
        
    def test_memory_coalescing(self, large_tree):
        """Test that memory access patterns are coalesced"""
        csr = convert_tree_to_csr(large_tree)
        
        # Sequential access should be efficient
        node_indices = torch.arange(min(256, csr.num_nodes), device=csr.device)
        
        start_time = time.perf_counter()
        for _ in range(10):  # Multiple iterations
            batch_children, batch_actions = csr.batch_get_children(node_indices)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # Should be reasonably fast (not fully optimized yet)
        avg_time = (end_time - start_time) / 10
        assert avg_time < 1.0  # Less than 1 second per iteration


class TestCSRCorrectness:
    """Test correctness of CSR conversion"""
    
    def test_conversion_preserves_tree_structure(self, large_tree):
        """Test that CSR conversion preserves exact tree structure"""
        csr = convert_tree_to_csr(large_tree)
        
        # Check every node's children match
        for node_idx in range(large_tree.num_nodes):
            # Get children from original tree
            orig_children, orig_actions = large_tree.get_children(node_idx)
            
            # Get children from CSR tree
            csr_children = csr.get_children_indices(node_idx)
            csr_actions = csr.get_children_actions(node_idx)
            
            # Should be identical
            assert len(orig_children) == len(csr_children)
            if len(orig_children) > 0:
                torch.testing.assert_close(orig_children, csr_children)
                torch.testing.assert_close(orig_actions, csr_actions)
                
    def test_visit_counts_preserved(self, large_tree):
        """Test that visit counts are exactly preserved"""
        csr = convert_tree_to_csr(large_tree)
        
        torch.testing.assert_close(
            csr.visit_counts[:csr.num_nodes],
            large_tree.visit_counts[:large_tree.num_nodes]
        )
        
    def test_value_sums_preserved(self, large_tree):
        """Test that value sums are preserved within FP16 precision"""
        csr = convert_tree_to_csr(large_tree)
        
        torch.testing.assert_close(
            csr.value_sums[:csr.num_nodes],
            large_tree.value_sums[:large_tree.num_nodes],
            rtol=1e-3, atol=1e-3  # FP16 precision tolerance
        )
        
    def test_parent_indices_preserved(self, large_tree):
        """Test that parent relationships are preserved"""
        csr = convert_tree_to_csr(large_tree)
        
        torch.testing.assert_close(
            csr.parent_indices[:csr.num_nodes],
            large_tree.parent_indices[:large_tree.num_nodes]
        )
        
    def test_flags_preserved(self, large_tree):
        """Test that node flags are preserved"""
        csr = convert_tree_to_csr(large_tree)
        
        torch.testing.assert_close(
            csr.flags[:csr.num_nodes],
            large_tree.flags[:large_tree.num_nodes]
        )


class TestCSREdgeCases:
    """Test edge cases and error conditions"""
    
    def test_single_node_tree(self, device):
        """Test CSR conversion of tree with only root node"""
        config = OptimizedTreeConfig(max_nodes=10, device=device)
        tree = OptimizedTensorTree(config)
        tree.add_root()
        
        csr = convert_tree_to_csr(tree)
        
        assert csr.num_nodes == 1
        assert csr.num_edges == 0
        assert len(csr.get_children_indices(0)) == 0
        
    def test_linear_tree(self, device):
        """Test CSR conversion of linear tree (each node has one child)"""
        config = OptimizedTreeConfig(max_nodes=10, device=device)
        tree = OptimizedTensorTree(config)
        
        # Build linear tree: 0 -> 1 -> 2 -> 3
        root = tree.add_root()
        current = root
        for i in range(3):
            current = tree.add_child(current, action=0, prior=1.0)
            
        csr = convert_tree_to_csr(tree)
        
        assert csr.num_nodes == 4
        assert csr.num_edges == 3
        
        # Each non-leaf node should have exactly one child
        assert len(csr.get_children_indices(0)) == 1
        assert len(csr.get_children_indices(1)) == 1  
        assert len(csr.get_children_indices(2)) == 1
        assert len(csr.get_children_indices(3)) == 0  # Leaf
        
    def test_wide_tree(self, device):
        """Test CSR conversion of wide tree (root has many children)"""
        config = OptimizedTreeConfig(max_nodes=100, device=device)
        tree = OptimizedTensorTree(config)
        
        # Build wide tree: root with 50 children
        root = tree.add_root()
        for i in range(50):
            tree.add_child(root, action=i, prior=0.02)
            
        csr = convert_tree_to_csr(tree)
        
        assert csr.num_nodes == 51  # root + 50 children
        assert csr.num_edges == 50
        
        # Root should have 50 children
        root_children = csr.get_children_indices(0)
        assert len(root_children) == 50
        assert set(root_children.tolist()) == set(range(1, 51))
        
        # All children should be leaves
        for i in range(1, 51):
            assert len(csr.get_children_indices(i)) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])