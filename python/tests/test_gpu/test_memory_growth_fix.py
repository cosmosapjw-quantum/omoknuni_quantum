"""Test memory growth configuration fixes

This module tests that memory growth mechanisms work correctly with proper
configuration, following TDD principles.
"""

import pytest
import torch
from mcts.gpu.csr_storage import CSRStorage, CSRStorageConfig


class TestMemoryGrowthFix:
    """Test memory growth configuration issues"""
    
    def test_growth_with_sufficient_max_capacity(self):
        """Test that growth works when max capacity allows it
        
        This test ensures growth works when max_edges is properly configured.
        """
        # Create config with sufficient max capacity for growth
        config = CSRStorageConfig(
            max_edges=100,  # Sufficient for growth
            device='cpu',
            initial_capacity_factor=0.1,  # Start small
            growth_factor=2.0
        )
        
        storage = CSRStorage(config, initial_nodes=5)
        
        # Check initial capacity
        initial_capacity = storage.edge_capacity
        print(f"DEBUG: Initial capacity: {initial_capacity}")
        print(f"DEBUG: Max edges: {config.max_edges}")
        
        # This should work - requesting growth within max limits
        new_required = initial_capacity + 5
        print(f"DEBUG: Requesting growth to: {new_required}")
        
        # Should not raise error since new_required < max_edges
        storage._grow_edge_storage(new_required)
        
        # Verify growth occurred
        assert storage.edge_capacity > initial_capacity
        print(f"DEBUG: New capacity: {storage.edge_capacity}")
        
    def test_growth_failure_at_max_limit(self):
        """Test that growth properly fails when hitting max limit"""
        # Create config with tight max capacity
        config = CSRStorageConfig(
            max_edges=20,
            device='cpu',
            initial_capacity_factor=0.5,  # Start with 10 edges (50% of 20)
            growth_factor=2.0
        )
        
        storage = CSRStorage(config, initial_nodes=5)
        initial_capacity = storage.edge_capacity
        print(f"DEBUG: Initial capacity: {initial_capacity}")
        print(f"DEBUG: Max edges: {config.max_edges}")
        
        # Try to grow beyond max_edges - this should fail
        beyond_max = config.max_edges + 1
        print(f"DEBUG: Trying to grow beyond max to: {beyond_max}")
        
        with pytest.raises(RuntimeError, match="Cannot grow edge storage beyond"):
            storage._grow_edge_storage(beyond_max)
            
    def test_growth_tracking_with_valid_growth(self):
        """Test that reallocation tracking works when growth is valid"""
        # Create config that allows at least 2 growth operations
        config = CSRStorageConfig(
            max_edges=100,  # Generous limit
            device='cpu',
            initial_capacity_factor=0.1,  # Start with 10 edges
            growth_factor=2.0
        )
        
        storage = CSRStorage(config, initial_nodes=5)
        
        # Check initial reallocation count
        initial_reallocations = getattr(storage, 'memory_reallocations', 0)
        assert initial_reallocations == 0
        
        # Force first growth (should work)
        first_growth = storage.edge_capacity + 1
        print(f"DEBUG: First growth to: {first_growth}")
        storage._grow_edge_storage(first_growth)
        assert storage.memory_reallocations == 1
        
        # Force second growth (should work)
        second_growth = storage.edge_capacity + 1
        print(f"DEBUG: Second growth to: {second_growth}")
        storage._grow_edge_storage(second_growth)
        assert storage.memory_reallocations == 2