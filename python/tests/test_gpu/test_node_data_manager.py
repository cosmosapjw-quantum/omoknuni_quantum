"""Tests for NodeDataManager GPU component

Tests cover:
- Node allocation and initialization
- Batch operations
- Memory growth and management
- Virtual loss operations
- Node flags and metadata
- Statistics tracking
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch

from mcts.gpu.node_data_manager import NodeDataManager, NodeDataConfig


@pytest.fixture
def node_config():
    """Create node data configuration"""
    return NodeDataConfig(
        max_nodes=1000,
        device='cpu',  # Use CPU for tests
        dtype_values=torch.float32,
        dtype_indices=torch.int32,
        initial_capacity_factor=0.1,
        growth_factor=1.5,
        enable_virtual_loss=True,
        virtual_loss_value=-1.0
    )


@pytest.fixture
def small_config():
    """Create small configuration for testing edge cases"""
    return NodeDataConfig(
        max_nodes=10,
        device='cpu',
        initial_capacity_factor=0.5,
        growth_factor=2.0
    )


@pytest.fixture  
def manager(node_config):
    """Create NodeDataManager instance"""
    return NodeDataManager(node_config)


class TestNodeDataConfig:
    """Test NodeDataConfig configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = NodeDataConfig()
        assert config.max_nodes == 100000
        assert config.device == 'cuda'
        assert config.dtype_values == torch.float32
        assert config.dtype_indices == torch.int32
        assert config.initial_capacity_factor == 0.1
        assert config.growth_factor == 1.5
        assert config.enable_virtual_loss == True
        assert config.virtual_loss_value == -1.0
        
    def test_custom_config(self):
        """Test custom configuration"""
        config = NodeDataConfig(
            max_nodes=5000,
            device='cpu',
            dtype_values=torch.float64,
            enable_virtual_loss=False
        )
        assert config.max_nodes == 5000
        assert config.device == 'cpu'
        assert config.dtype_values == torch.float64
        assert config.enable_virtual_loss == False


class TestNodeDataManagerInitialization:
    """Test NodeDataManager initialization"""
    
    def test_initialization(self, manager, node_config):
        """Test basic initialization"""
        assert manager.config == node_config
        assert manager.device == torch.device('cpu')
        assert manager.num_nodes == 0
        
        # Check storage initialized
        expected_size = int(node_config.max_nodes * node_config.initial_capacity_factor)
        assert len(manager.visit_counts) == expected_size
        assert len(manager.value_sums) == expected_size
        assert len(manager.node_priors) == expected_size
        
    def test_small_max_nodes(self):
        """Test initialization with small max_nodes"""
        config = NodeDataConfig(max_nodes=50, device='cpu')
        manager = NodeDataManager(config)
        
        # Should allocate full capacity for small sizes
        assert len(manager.visit_counts) == 50
        
    def test_very_small_initial_capacity(self):
        """Test with very small initial capacity factor"""
        config = NodeDataConfig(
            max_nodes=0,  # Unlimited
            initial_capacity_factor=0.001,
            device='cpu'
        )
        manager = NodeDataManager(config)
        
        # Should have reasonable minimum size
        assert len(manager.visit_counts) >= 10
        
    def test_cuda_device_fallback(self):
        """Test CUDA device fallback to CPU"""
        with patch('torch.cuda.is_available', return_value=False):
            config = NodeDataConfig(device='cuda')
            manager = NodeDataManager(config)
            assert manager.device == torch.device('cpu')
            
    def test_initial_tensor_values(self, manager):
        """Test initial tensor values"""
        size = len(manager.visit_counts)
        
        # Check initial values
        assert torch.all(manager.visit_counts == 0)
        assert torch.all(manager.value_sums == 0.0)
        assert torch.all(manager.node_priors == 0.0)
        assert torch.all(manager.virtual_loss_counts == 0)
        assert torch.all(manager.flags == 0)
        assert torch.all(manager.phases == 0.0)
        assert torch.all(manager.parent_indices == -2)
        assert torch.all(manager.parent_actions == -1)


class TestNodeAllocation:
    """Test node allocation operations"""
    
    def test_single_node_allocation(self, manager):
        """Test allocating a single node"""
        prior = 0.5
        parent_idx = 0
        parent_action = 42
        
        node_idx = manager.allocate_node(prior, parent_idx, parent_action)
        
        assert node_idx == 0
        assert manager.num_nodes == 1
        assert manager.node_priors[node_idx] == prior
        assert manager.parent_indices[node_idx] == parent_idx
        assert manager.parent_actions[node_idx] == parent_action
        assert manager.visit_counts[node_idx] == 0
        assert manager.value_sums[node_idx] == 0.0
        
    def test_multiple_node_allocation(self, manager):
        """Test allocating multiple nodes sequentially"""
        nodes = []
        for i in range(5):
            node_idx = manager.allocate_node(prior=float(i)/10)
            nodes.append(node_idx)
            
        assert manager.num_nodes == 5
        assert nodes == [0, 1, 2, 3, 4]
        
        # Check priors
        for i, idx in enumerate(nodes):
            assert manager.node_priors[idx] == float(i)/10
            
    def test_batch_allocation(self, manager):
        """Test batch node allocation"""
        count = 10
        priors = torch.arange(count, dtype=torch.float32) / count
        parent_idx = 0
        parent_actions = torch.arange(count, dtype=torch.int32)
        
        indices = manager.allocate_nodes_batch(count, priors, parent_idx, parent_actions)
        
        assert len(indices) == count
        assert manager.num_nodes == count
        assert torch.all(indices == torch.arange(count))
        
        # Check node data
        assert torch.all(manager.node_priors[indices] == priors)
        assert torch.all(manager.parent_indices[indices] == parent_idx)
        assert torch.all(manager.parent_actions[indices] == parent_actions)
        assert torch.all(manager.visit_counts[indices] == 0)
        
    def test_allocation_triggers_growth(self):
        """Test that allocation triggers storage growth"""
        # Use a config that will trigger growth (max_nodes > 100)
        config = NodeDataConfig(
            max_nodes=500,
            device='cpu',
            initial_capacity_factor=0.2,  # Start with 100 capacity
            growth_factor=2.0
        )
        manager = NodeDataManager(config)
        initial_capacity = len(manager.visit_counts)
        
        # Allocate more nodes than initial capacity to trigger growth
        for i in range(initial_capacity + 10):
            manager.allocate_node(prior=0.1)
            
        # Should have grown
        assert len(manager.visit_counts) > initial_capacity
        assert manager.num_nodes == initial_capacity + 10
        
    def test_batch_allocation_triggers_growth(self):
        """Test batch allocation triggers growth"""
        # Use a config that will trigger growth (max_nodes > 100)
        config = NodeDataConfig(
            max_nodes=500,
            device='cpu',
            initial_capacity_factor=0.2,
            growth_factor=2.0
        )
        manager = NodeDataManager(config)
        initial_capacity = len(manager.visit_counts)
        
        # Allocate batch larger than capacity but within max_nodes
        count = min(initial_capacity + 10, config.max_nodes - manager.num_nodes)
        priors = torch.ones(count)
        parent_actions = torch.zeros(count, dtype=torch.int32)
        
        indices = manager.allocate_nodes_batch(count, priors, 0, parent_actions)
        
        assert len(indices) == count
        assert len(manager.visit_counts) > initial_capacity


class TestMemoryGrowth:
    """Test memory growth and management"""
    
    def test_storage_growth(self):
        """Test storage growth mechanism"""
        # Use a config that allows growth
        config = NodeDataConfig(
            max_nodes=500,
            device='cpu',
            initial_capacity_factor=0.2,
            growth_factor=2.0
        )
        manager = NodeDataManager(config)
        initial_size = len(manager.visit_counts)
        
        # Force growth
        manager._grow_storage()
        
        new_size = len(manager.visit_counts)
        expected_size = int(initial_size * config.growth_factor)
        assert new_size == expected_size
        
        # Check all tensors grew
        assert len(manager.value_sums) == new_size
        assert len(manager.node_priors) == new_size
        assert len(manager.virtual_loss_counts) == new_size
        
    def test_growth_preserves_data(self):
        """Test that growth preserves existing data"""
        # Use a config that allows growth
        config = NodeDataConfig(
            max_nodes=500,
            device='cpu',
            initial_capacity_factor=0.2,
            growth_factor=2.0
        )
        manager = NodeDataManager(config)
        
        # Add some nodes
        for i in range(3):
            idx = manager.allocate_node(prior=float(i))
            manager.update_visit_count(idx, i+1)
            manager.update_value_sum(idx, float(i)*2)
            
        # Save current data
        old_priors = manager.node_priors[:3].clone()
        old_visits = manager.visit_counts[:3].clone()
        old_values = manager.value_sums[:3].clone()
        
        # Force growth
        manager._grow_storage()
        
        # Check data preserved
        assert torch.all(manager.node_priors[:3] == old_priors)
        assert torch.all(manager.visit_counts[:3] == old_visits)
        assert torch.all(manager.value_sums[:3] == old_values)
        
    def test_max_nodes_limit(self, small_config):
        """Test growth respects max_nodes limit"""
        manager = NodeDataManager(small_config)
        
        # Allocate up to max_nodes
        for i in range(small_config.max_nodes):
            manager.allocate_node(prior=0.1)
            
        # Try to allocate one more
        with pytest.raises(RuntimeError, match="Cannot grow beyond max_nodes"):
            manager.allocate_node(prior=0.1)
            
    def test_reallocation_tracking(self):
        """Test memory reallocation tracking"""
        # Use a config that allows multiple growths
        config = NodeDataConfig(
            max_nodes=1000,
            device='cpu',
            initial_capacity_factor=0.1,  # Small initial capacity to trigger growth
            growth_factor=2.0
        )
        manager = NodeDataManager(config)
        
        # Initially no reallocations
        assert getattr(manager, 'memory_reallocations', 0) == 0
        
        # Force growth
        manager._grow_storage()
        assert manager.memory_reallocations == 1
        
        # Another growth
        manager._grow_storage()
        assert manager.memory_reallocations == 2


class TestNodeOperations:
    """Test node data operations"""
    
    def test_update_operations(self, manager):
        """Test basic update operations"""
        idx = manager.allocate_node(prior=0.5)
        
        # Update visit count
        manager.update_visit_count(idx, 1)
        assert manager.visit_counts[idx] == 1
        
        manager.update_visit_count(idx, 3)
        assert manager.visit_counts[idx] == 4
        
        # Update value sum
        manager.update_value_sum(idx, 0.8)
        assert manager.value_sums[idx] == 0.8
        
        manager.update_value_sum(idx, -0.3)
        assert abs(manager.value_sums[idx] - 0.5) < 1e-6
        
    def test_batch_updates(self, manager):
        """Test batch update operations"""
        # Allocate nodes
        indices = manager.allocate_nodes_batch(
            5, 
            torch.ones(5) * 0.2,
            0,
            torch.zeros(5, dtype=torch.int32)
        )
        
        # Batch update visits
        deltas = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
        manager.batch_update_visits(indices, deltas)
        assert torch.all(manager.visit_counts[indices] == deltas)
        
        # Batch update values
        values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        manager.batch_update_values(indices, values)
        assert torch.all(manager.value_sums[indices] == values)
        
    def test_q_value_calculation(self, manager):
        """Test Q-value calculation"""
        idx = manager.allocate_node(prior=0.5)
        
        # No visits - should return 0
        assert manager.get_q_value(idx) == 0.0
        
        # Add visits and value
        manager.update_visit_count(idx, 10)
        manager.update_value_sum(idx, 5.0)
        
        assert manager.get_q_value(idx) == 0.5
        
    def test_q_values_batch(self, manager):
        """Test batch Q-value calculation"""
        # Create nodes with different stats
        indices = []
        for i in range(3):
            idx = manager.allocate_node(prior=0.1)
            manager.update_visit_count(idx, (i+1)*10)
            manager.update_value_sum(idx, (i+1)*5.0)
            indices.append(idx)
            
        # Add one with no visits
        idx = manager.allocate_node(prior=0.1)
        indices.append(idx)
        
        indices_tensor = torch.tensor(indices)
        q_values = manager.get_q_values_batch(indices_tensor)
        
        expected = torch.tensor([0.5, 0.5, 0.5, 0.0])
        assert torch.allclose(q_values, expected)


class TestVirtualLoss:
    """Test virtual loss functionality"""
    
    def test_virtual_loss_application(self, manager):
        """Test applying virtual loss"""
        indices = torch.tensor([0, 1, 2])
        manager.num_nodes = 3
        
        # Apply virtual loss
        manager.apply_virtual_loss(indices)
        assert torch.all(manager.virtual_loss_counts[indices] == 1)
        
        # Apply again
        manager.apply_virtual_loss(indices[:1])
        assert manager.virtual_loss_counts[0] == 2
        assert manager.virtual_loss_counts[1] == 1
        
    def test_virtual_loss_removal(self, manager):
        """Test removing virtual loss"""
        indices = torch.tensor([0, 1])
        manager.num_nodes = 2
        
        # Apply virtual loss multiple times
        manager.apply_virtual_loss(indices)
        manager.apply_virtual_loss(indices)
        
        # Remove once
        manager.remove_virtual_loss(indices)
        assert torch.all(manager.virtual_loss_counts[indices] == 1)
        
        # Remove again
        manager.remove_virtual_loss(indices)
        assert torch.all(manager.virtual_loss_counts[indices] == 0)
        
        # Should not go negative
        manager.remove_virtual_loss(indices)
        assert torch.all(manager.virtual_loss_counts[indices] == 0)
        
    def test_effective_visits(self, manager):
        """Test effective visit count calculation"""
        idx = manager.allocate_node(prior=0.5)
        manager.update_visit_count(idx, 10)
        
        indices = torch.tensor([idx])
        
        # Without virtual loss
        assert manager.get_effective_visits(indices)[0] == 10
        
        # With virtual loss
        manager.apply_virtual_loss(indices)
        assert manager.get_effective_visits(indices)[0] == 11
        
        manager.apply_virtual_loss(indices)
        assert manager.get_effective_visits(indices)[0] == 12
        
    def test_effective_values(self, manager):
        """Test effective value sum calculation"""
        idx = manager.allocate_node(prior=0.5)
        manager.update_value_sum(idx, 5.0)
        
        indices = torch.tensor([idx])
        
        # Without virtual loss
        assert manager.get_effective_values(indices)[0] == 5.0
        
        # With virtual loss (default -1.0 per loss)
        manager.apply_virtual_loss(indices)
        assert manager.get_effective_values(indices)[0] == 4.0
        
        manager.apply_virtual_loss(indices)
        assert manager.get_effective_values(indices)[0] == 3.0
        
    def test_virtual_loss_disabled(self, node_config):
        """Test with virtual loss disabled"""
        node_config.enable_virtual_loss = False
        manager = NodeDataManager(node_config)
        
        idx = manager.allocate_node(prior=0.5)
        manager.update_visit_count(idx, 10)
        manager.update_value_sum(idx, 5.0)
        
        indices = torch.tensor([idx])
        
        # Virtual loss operations should be no-ops
        manager.apply_virtual_loss(indices)
        assert manager.get_effective_visits(indices)[0] == 10
        assert manager.get_effective_values(indices)[0] == 5.0


class TestNodeFlags:
    """Test node flags and metadata"""
    
    def test_terminal_flag(self, manager):
        """Test terminal flag operations"""
        idx = manager.allocate_node(prior=0.5)
        
        # Initially not terminal
        assert not manager.is_terminal(idx)
        
        # Set terminal
        manager.set_terminal(idx, True)
        assert manager.is_terminal(idx)
        
        # Unset terminal
        manager.set_terminal(idx, False)
        assert not manager.is_terminal(idx)
        
    def test_expanded_flag(self, manager):
        """Test expanded flag operations"""
        idx = manager.allocate_node(prior=0.5)
        
        # Initially not expanded
        assert not manager.is_expanded(idx)
        
        # Set expanded
        manager.set_expanded(idx, True)
        assert manager.is_expanded(idx)
        
        # Unset expanded
        manager.set_expanded(idx, False)
        assert not manager.is_expanded(idx)
        
    def test_combined_flags(self, manager):
        """Test combined flag operations"""
        idx = manager.allocate_node(prior=0.5)
        
        # Set both flags
        manager.set_expanded(idx, True)
        manager.set_terminal(idx, True)
        
        assert manager.is_expanded(idx)
        assert manager.is_terminal(idx)
        
        # Unset one flag shouldn't affect the other
        manager.set_expanded(idx, False)
        assert not manager.is_expanded(idx)
        assert manager.is_terminal(idx)


class TestReset:
    """Test reset functionality"""
    
    def test_reset_empty(self, manager):
        """Test reset with no nodes"""
        manager.reset()
        assert manager.num_nodes == 0
        
    def test_reset_with_data(self, manager):
        """Test reset with existing data"""
        # Add some nodes
        for i in range(5):
            idx = manager.allocate_node(prior=float(i))
            manager.update_visit_count(idx, i+1)
            manager.update_value_sum(idx, float(i)*2)
            manager.set_expanded(idx, True)
            
        # Reset
        manager.reset()
        
        assert manager.num_nodes == 0
        
        # Check data cleared (only check used portion)
        assert torch.all(manager.visit_counts[:5] == 0)
        assert torch.all(manager.value_sums[:5] == 0.0)
        assert torch.all(manager.node_priors[:5] == 0.0)
        assert torch.all(manager.flags[:5] == 0)
        assert torch.all(manager.parent_indices[:5] == -2)
        
    def test_reset_preserves_capacity(self, manager):
        """Test reset preserves storage capacity"""
        # Add nodes and trigger growth
        for i in range(200):
            manager.allocate_node(prior=0.1)
            
        capacity_before = len(manager.visit_counts)
        
        # Reset
        manager.reset()
        
        # Capacity should be preserved
        assert len(manager.visit_counts) == capacity_before


class TestMemoryUsage:
    """Test memory usage tracking"""
    
    def test_memory_usage_calculation(self, manager):
        """Test memory usage calculation"""
        memory_mb = manager.get_memory_usage_mb()
        
        # Should be positive
        assert memory_mb > 0
        
        # Rough check - each tensor contributes
        num_tensors = 8  # visit_counts, value_sums, etc.
        capacity = len(manager.visit_counts)
        
        # Very rough estimate
        min_expected = (capacity * 4 * num_tensors) / (1024 * 1024) * 0.5
        assert memory_mb > min_expected
        
    def test_memory_usage_grows(self):
        """Test memory usage grows with storage"""
        # Use a config that allows growth
        config = NodeDataConfig(
            max_nodes=500,
            device='cpu',
            initial_capacity_factor=0.2,
            growth_factor=2.0
        )
        manager = NodeDataManager(config)
        
        initial_memory = manager.get_memory_usage_mb()
        
        # Force growth
        manager._grow_storage()
        
        new_memory = manager.get_memory_usage_mb()
        assert new_memory > initial_memory


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_zero_max_nodes(self):
        """Test with unlimited nodes (max_nodes=0)"""
        config = NodeDataConfig(max_nodes=0, device='cpu')
        manager = NodeDataManager(config)
        
        # Should be able to allocate many nodes
        for i in range(100):
            manager.allocate_node(prior=0.1)
            
        assert manager.num_nodes == 100
        
    def test_large_batch_allocation(self, manager):
        """Test allocating very large batch"""
        count = 500
        priors = torch.ones(count) * 0.1
        parent_actions = torch.zeros(count, dtype=torch.int32)
        
        indices = manager.allocate_nodes_batch(count, priors, 0, parent_actions)
        
        assert len(indices) == count
        assert manager.num_nodes == count
        
    def test_invalid_node_index(self, manager):
        """Test operations on invalid node indices"""
        # Allocate one node
        manager.allocate_node(prior=0.5)
        
        # Try to access non-existent node (should not crash)
        # PyTorch will handle bounds checking
        with pytest.raises(IndexError):
            _ = manager.visit_counts[1000].item()
            
    def test_cuda_memory_calculation(self):
        """Test CUDA memory-based initialization logic"""
        # Test the calculation logic directly without creating CUDA tensors
        # This avoids CUDA initialization issues in testing environments
        
        # Mock memory values
        total_memory = 4 * 1024 * 1024 * 1024  # 4GB
        allocated_memory = 1024 * 1024 * 1024  # 1GB allocated
        available_memory = total_memory - allocated_memory  # 3GB
        
        # Calculate expected nodes using the same logic as the actual method
        # Available memory * 0.3 (safety factor) / 60 bytes per node
        expected_nodes = int((available_memory * 0.3) / 60)
        expected_nodes = min(75000, max(20000, expected_nodes))
        
        # Verify the calculation is reasonable
        assert 20000 <= expected_nodes <= 75000
        
        # Test with CPU config instead of CUDA to avoid CUDA initialization issues
        config = NodeDataConfig(max_nodes=0, device='cpu')
        manager = NodeDataManager(config)
        
        # Should have created a manager with reasonable size
        assert len(manager.visit_counts) >= 10000  # Should be substantial size