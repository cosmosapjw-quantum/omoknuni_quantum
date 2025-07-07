"""Tests for CSR storage management

Tests cover:
- CSR format storage initialization
- Edge addition (single and batch)
- Row pointer management
- Memory growth and reallocation
- Children retrieval
- Reset functionality
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch

from mcts.gpu.csr_storage import CSRStorage, CSRStorageConfig


@pytest.fixture
def csr_config():
    """Create CSR storage configuration"""
    return CSRStorageConfig(
        max_edges=1000,
        device='cpu',  # Use CPU for tests
        dtype_indices=torch.int32,
        dtype_actions=torch.int32,
        dtype_values=torch.float32,
        initial_capacity_factor=0.1,
        growth_factor=1.5
    )


@pytest.fixture
def small_config():
    """Create small configuration for edge case testing"""
    return CSRStorageConfig(
        max_edges=20,
        device='cpu',
        initial_capacity_factor=0.5,
        growth_factor=2.0
    )


@pytest.fixture
def storage(csr_config):
    """Create CSRStorage instance"""
    return CSRStorage(csr_config, initial_nodes=100)


class TestCSRStorageConfig:
    """Test CSRStorageConfig configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = CSRStorageConfig()
        assert config.max_edges == 500000
        assert config.device == 'cuda'
        assert config.dtype_indices == torch.int32
        assert config.dtype_actions == torch.int32
        assert config.dtype_values == torch.float32
        assert config.initial_capacity_factor == 0.1
        assert config.growth_factor == 1.5
        
    def test_custom_config(self):
        """Test custom configuration"""
        config = CSRStorageConfig(
            max_edges=10000,
            device='cpu',
            dtype_indices=torch.int64,
            growth_factor=2.0
        )
        assert config.max_edges == 10000
        assert config.device == 'cpu'
        assert config.dtype_indices == torch.int64
        assert config.growth_factor == 2.0


class TestCSRStorageInitialization:
    """Test CSRStorage initialization"""
    
    def test_initialization(self, storage, csr_config):
        """Test basic initialization"""
        assert storage.config == csr_config
        assert storage.device == torch.device('cpu')
        assert storage.num_edges == 0
        assert storage._needs_row_ptr_update == False
        
        # Check initial storage
        expected_edges = int(csr_config.max_edges * csr_config.initial_capacity_factor)
        assert storage.edge_capacity == expected_edges
        assert len(storage.col_indices) == expected_edges
        assert len(storage.edge_actions) == expected_edges
        assert len(storage.edge_priors) == expected_edges
        
        # Row pointers should have initial_nodes + 1 entries
        assert len(storage.row_ptr) == 101
        
    def test_unlimited_edges(self):
        """Test initialization with unlimited edges"""
        config = CSRStorageConfig(max_edges=0, device='cpu')
        storage = CSRStorage(config, initial_nodes=50)
        
        # Should use default initial edge capacity
        assert storage.edge_capacity == 50000
        
    def test_cuda_device_fallback(self):
        """Test CUDA device fallback to CPU"""
        with patch('torch.cuda.is_available', return_value=False):
            config = CSRStorageConfig(device='cuda')
            storage = CSRStorage(config)
            assert storage.device == torch.device('cpu')
            
    def test_initial_tensor_values(self, storage):
        """Test initial tensor values"""
        # Check all tensors are zero-initialized
        assert torch.all(storage.row_ptr == 0)
        assert torch.all(storage.col_indices == 0)
        assert torch.all(storage.edge_actions == 0)
        assert torch.all(storage.edge_priors == 0.0)


class TestEdgeAddition:
    """Test edge addition operations"""
    
    def test_single_edge_addition(self, storage):
        """Test adding a single edge"""
        parent_idx = 0
        child_idx = 5
        action = 42
        prior = 0.8
        
        edge_idx = storage.add_edge(parent_idx, child_idx, action, prior)
        
        assert edge_idx == 0
        assert storage.num_edges == 1
        assert storage.col_indices[edge_idx] == child_idx
        assert storage.edge_actions[edge_idx] == action
        assert abs(storage.edge_priors[edge_idx].item() - prior) < 1e-6
        assert storage._needs_row_ptr_update == True
        
    def test_multiple_edge_addition(self, storage):
        """Test adding multiple edges sequentially"""
        edges = []
        for i in range(5):
            edge_idx = storage.add_edge(
                parent_idx=i,
                child_idx=i+10,
                action=i*2,
                prior=float(i)/10
            )
            edges.append(edge_idx)
            
        assert storage.num_edges == 5
        assert edges == [0, 1, 2, 3, 4]
        
        # Verify edge data
        for i, edge_idx in enumerate(edges):
            assert storage.col_indices[edge_idx] == i+10
            assert storage.edge_actions[edge_idx] == i*2
            assert abs(storage.edge_priors[edge_idx].item() - float(i)/10) < 1e-6
            
    def test_batch_edge_addition(self, storage):
        """Test batch edge addition"""
        parent_idx = 0
        num_children = 10
        child_indices = torch.arange(10, 20, dtype=torch.int32)
        actions = torch.arange(0, 10, dtype=torch.int32)
        priors = torch.linspace(0.1, 1.0, num_children)
        
        edge_indices = storage.add_edges_batch(parent_idx, child_indices, actions, priors)
        
        assert len(edge_indices) == num_children
        assert storage.num_edges == num_children
        assert storage._needs_row_ptr_update == True
        
        # Verify edge data
        assert torch.all(storage.col_indices[edge_indices] == child_indices)
        assert torch.all(storage.edge_actions[edge_indices] == actions)
        assert torch.allclose(storage.edge_priors[edge_indices], priors)
        
    def test_edge_addition_triggers_growth(self, small_config):
        """Test edge addition triggers storage growth"""
        storage = CSRStorage(small_config, initial_nodes=10)
        initial_capacity = storage.edge_capacity
        
        # Add more edges than initial capacity
        for i in range(initial_capacity + 5):
            storage.add_edge(0, i, i, 0.5)
            
        # Should have grown
        assert storage.edge_capacity > initial_capacity
        assert storage.num_edges == initial_capacity + 5
        
    def test_batch_addition_triggers_growth(self, small_config):
        """Test batch addition triggers growth"""
        storage = CSRStorage(small_config, initial_nodes=10)
        initial_capacity = storage.edge_capacity
        
        # Add batch larger than capacity
        num_edges = initial_capacity + 10
        child_indices = torch.arange(num_edges, dtype=torch.int32)
        actions = torch.zeros(num_edges, dtype=torch.int32)
        priors = torch.ones(num_edges)
        
        edge_indices = storage.add_edges_batch(0, child_indices, actions, priors)
        
        assert len(edge_indices) == num_edges
        assert storage.edge_capacity > initial_capacity


class TestMemoryGrowth:
    """Test memory growth and management"""
    
    def test_edge_storage_growth(self, small_config):
        """Test edge storage growth mechanism"""
        storage = CSRStorage(small_config, initial_nodes=10)
        initial_capacity = storage.edge_capacity
        
        # Force growth
        storage._grow_edge_storage(initial_capacity + 1)
        
        new_capacity = storage.edge_capacity
        assert new_capacity >= int(initial_capacity * small_config.growth_factor)
        
        # Check all tensors grew
        assert len(storage.col_indices) == new_capacity
        assert len(storage.edge_actions) == new_capacity
        assert len(storage.edge_priors) == new_capacity
        
    def test_growth_preserves_data(self, small_config):
        """Test that growth preserves existing edge data"""
        storage = CSRStorage(small_config, initial_nodes=10)
        
        # Add some edges
        for i in range(5):
            storage.add_edge(0, i+10, i*2, float(i)/10)
            
        # Save current data
        old_col_indices = storage.col_indices[:5].clone()
        old_actions = storage.edge_actions[:5].clone()
        old_priors = storage.edge_priors[:5].clone()
        
        # Force growth
        storage._grow_edge_storage(storage.edge_capacity + 1)
        
        # Check data preserved
        assert torch.all(storage.col_indices[:5] == old_col_indices)
        assert torch.all(storage.edge_actions[:5] == old_actions)
        assert torch.allclose(storage.edge_priors[:5], old_priors)
        
    def test_max_edges_limit(self, small_config):
        """Test growth respects max_edges limit"""
        storage = CSRStorage(small_config, initial_nodes=10)
        
        # Fill up to max_edges
        while storage.num_edges < small_config.max_edges:
            storage.add_edge(0, storage.num_edges, 0, 0.5)
            
        # Try to add one more
        with pytest.raises(RuntimeError, match="Cannot grow edge storage beyond"):
            storage.add_edge(0, 100, 0, 0.5)
            
    def test_reallocation_tracking(self):
        """Test memory reallocation tracking with appropriate limits"""
        # Use config that allows growth
        config = CSRStorageConfig(
            max_edges=100,  # Allow growth
            device='cpu',
            initial_capacity_factor=0.1,  # Start small with 10 edges
            growth_factor=2.0
        )
        storage = CSRStorage(config, initial_nodes=10)
        
        # Initially no reallocations
        assert getattr(storage, 'memory_reallocations', 0) == 0
        
        # Force growth (should work since 11 < 100)
        storage._grow_edge_storage(storage.edge_capacity + 1)
        assert storage.memory_reallocations == 1
        
        # Another growth (should work since new capacity is still < 100)
        storage._grow_edge_storage(storage.edge_capacity + 1)
        assert storage.memory_reallocations == 2
        
    def test_row_ptr_growth(self, storage):
        """Test row pointer array growth"""
        initial_size = len(storage.row_ptr)
        
        # Grow row pointers
        storage.grow_row_ptr_if_needed(initial_size + 50)
        
        # Should have grown
        assert len(storage.row_ptr) > initial_size
        
        # Original data should be preserved
        assert torch.all(storage.row_ptr[:initial_size] == 0)


class TestRowPointerManagement:
    """Test row pointer operations"""
    
    def test_row_pointer_update_needed(self, storage):
        """Test row pointer update tracking"""
        # Initially no update needed
        assert not storage.needs_row_ptr_update()
        
        # Add edge marks for update
        storage.add_edge(0, 5, 10, 0.5)
        assert storage.needs_row_ptr_update()
        
    def test_children_query_requires_update(self, storage):
        """Test that children query requires row pointer update"""
        storage.add_edge(0, 5, 10, 0.5)
        
        # Should raise error before update
        with pytest.raises(RuntimeError, match="Row pointers need update"):
            storage.get_children_edges(0)
            
    def test_rebuild_row_pointers(self, storage):
        """Test row pointer rebuilding"""
        # Create children table
        num_nodes = 5
        max_children = 3
        children_table = torch.full((num_nodes, max_children), -1, dtype=torch.int32)
        
        # Node 0 has 2 children
        children_table[0, :2] = torch.tensor([1, 2])
        # Node 1 has 3 children
        children_table[1, :3] = torch.tensor([3, 4, 5])
        # Node 2 has 0 children
        # Node 3 has 1 child
        children_table[3, 0] = 6
        # Node 4 has 0 children
        
        storage._needs_row_ptr_update = True
        storage.rebuild_row_pointers(children_table)
        
        # Check row pointers
        expected_row_ptr = torch.tensor([0, 2, 5, 5, 6, 6], dtype=torch.int32)
        assert torch.all(storage.row_ptr[:6] == expected_row_ptr)
        assert not storage._needs_row_ptr_update
        
    def test_rebuild_with_large_table(self, storage):
        """Test rebuilding with table larger than row_ptr"""
        # Ensure row_ptr is smaller
        storage.row_ptr = torch.zeros(10, dtype=torch.int32)
        
        # Create large children table
        children_table = torch.full((20, 5), -1, dtype=torch.int32)
        
        # Should not crash, just process what fits
        storage._needs_row_ptr_update = True
        storage.rebuild_row_pointers(children_table)
        
        assert not storage._needs_row_ptr_update


class TestChildrenRetrieval:
    """Test children retrieval operations"""
    
    def test_get_node_children(self, storage):
        """Test retrieving node children"""
        # Add edges for node 0
        storage.add_edge(0, 10, 1, 0.1)
        storage.add_edge(0, 11, 2, 0.2)
        storage.add_edge(0, 12, 3, 0.3)
        
        # Add edges for node 1
        storage.add_edge(1, 20, 4, 0.4)
        storage.add_edge(1, 21, 5, 0.5)
        
        # Rebuild row pointers
        children_table = torch.tensor([
            [10, 11, 12, -1],
            [20, 21, -1, -1]
        ], dtype=torch.int32)
        storage.rebuild_row_pointers(children_table)
        
        # Get children for node 0
        child_indices, actions, priors = storage.get_node_children(0)
        
        assert len(child_indices) == 3
        assert torch.all(child_indices == torch.tensor([10, 11, 12]))
        assert torch.all(actions == torch.tensor([1, 2, 3]))
        assert torch.allclose(priors, torch.tensor([0.1, 0.2, 0.3]))
        
    def test_get_empty_children(self, storage):
        """Test retrieving children for node with no children"""
        # Rebuild with empty node
        children_table = torch.full((3, 2), -1, dtype=torch.int32)
        storage.rebuild_row_pointers(children_table)
        
        # Get children for node with no children
        child_indices, actions, priors = storage.get_node_children(1)
        
        assert len(child_indices) == 0
        assert len(actions) == 0
        assert len(priors) == 0
        
        # Check tensor types are correct
        assert child_indices.dtype == storage.config.dtype_indices
        assert actions.dtype == storage.config.dtype_actions
        assert priors.dtype == storage.config.dtype_values
        
    def test_get_children_edges(self, storage):
        """Test getting edge indices for node children"""
        # Setup edges and row pointers
        storage.add_edge(0, 10, 1, 0.1)
        storage.add_edge(0, 11, 2, 0.2)
        
        children_table = torch.tensor([[10, 11, -1]], dtype=torch.int32)
        storage.rebuild_row_pointers(children_table)
        
        start, end = storage.get_children_edges(0)
        assert start == 0
        assert end == 2


class TestResetFunctionality:
    """Test reset operations"""
    
    def test_reset_empty(self, storage):
        """Test reset with no edges"""
        storage.reset()
        assert storage.num_edges == 0
        assert not storage._needs_row_ptr_update
        
    def test_reset_with_data(self, storage):
        """Test reset with existing edges"""
        # Add some edges
        for i in range(10):
            storage.add_edge(i, i+10, i, float(i)/10)
            
        # Rebuild row pointers
        children_table = torch.full((10, 2), -1, dtype=torch.int32)
        storage.rebuild_row_pointers(children_table)
        
        # Reset
        storage.reset()
        
        assert storage.num_edges == 0
        assert not storage._needs_row_ptr_update
        
        # Check data cleared
        assert torch.all(storage.col_indices[:10] == 0)
        assert torch.all(storage.edge_actions[:10] == 0)
        assert torch.all(storage.edge_priors[:10] == 0.0)
        assert torch.all(storage.row_ptr == 0)
        
    def test_reset_preserves_capacity(self, storage):
        """Test reset preserves storage capacity"""
        # Add edges and trigger growth
        for i in range(200):
            storage.add_edge(0, i, i, 0.5)
            
        capacity_before = storage.edge_capacity
        
        # Reset
        storage.reset()
        
        # Capacity should be preserved
        assert storage.edge_capacity == capacity_before


class TestMemoryUsageAndStats:
    """Test memory usage and statistics"""
    
    def test_memory_usage_calculation(self, storage):
        """Test memory usage calculation"""
        memory_mb = storage.get_memory_usage_mb()
        
        # Should be positive
        assert memory_mb > 0
        
        # Rough calculation
        total_elements = (len(storage.row_ptr) + 
                         len(storage.col_indices) + 
                         len(storage.edge_actions) + 
                         len(storage.edge_priors))
        bytes_per_element = 4  # int32 and float32
        expected_mb = (total_elements * bytes_per_element) / (1024 * 1024)
        
        # Should be close to expected
        assert abs(memory_mb - expected_mb) < 0.1
        
    def test_edge_utilization(self, storage):
        """Test edge utilization calculation"""
        # Initially 0%
        assert storage.get_edge_utilization() == 0.0
        
        # Add some edges
        num_edges = 50
        for i in range(num_edges):
            storage.add_edge(0, i, i, 0.5)
            
        utilization = storage.get_edge_utilization()
        expected = num_edges / storage.edge_capacity
        assert abs(utilization - expected) < 1e-6
        
    def test_edge_utilization_full(self, small_config):
        """Test edge utilization when full"""
        storage = CSRStorage(small_config, initial_nodes=10)
        
        # Fill to capacity
        for i in range(storage.edge_capacity):
            storage.add_edge(0, i, i, 0.5)
            
        assert storage.get_edge_utilization() == 1.0


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_zero_max_edges(self):
        """Test with unlimited edges (max_edges=0)"""
        config = CSRStorageConfig(max_edges=0, device='cpu')
        storage = CSRStorage(config, initial_nodes=10)
        
        # Should be able to add many edges
        for i in range(1000):
            storage.add_edge(0, i, i, 0.5)
            
        assert storage.num_edges == 1000
        
    def test_large_batch_addition(self, storage):
        """Test adding very large batch"""
        num_edges = 500
        child_indices = torch.arange(num_edges, dtype=torch.int32)
        actions = torch.zeros(num_edges, dtype=torch.int32)
        priors = torch.ones(num_edges) * 0.1
        
        edge_indices = storage.add_edges_batch(0, child_indices, actions, priors)
        
        assert len(edge_indices) == num_edges
        assert storage.num_edges == num_edges
        
    def test_mixed_single_and_batch_addition(self, storage):
        """Test mixing single and batch edge additions"""
        # Add single edge
        storage.add_edge(0, 10, 1, 0.1)
        
        # Add batch
        child_indices = torch.tensor([20, 21, 22], dtype=torch.int32)
        actions = torch.tensor([2, 3, 4], dtype=torch.int32)
        priors = torch.tensor([0.2, 0.3, 0.4])
        storage.add_edges_batch(1, child_indices, actions, priors)
        
        # Add another single edge
        storage.add_edge(2, 30, 5, 0.5)
        
        assert storage.num_edges == 5
        
        # Verify all edges
        assert storage.col_indices[0] == 10
        assert torch.all(storage.col_indices[1:4] == child_indices)
        assert storage.col_indices[4] == 30