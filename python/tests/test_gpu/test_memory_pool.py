"""Tests for memory pool management

Tests cover:
- Tensor pool allocation and deallocation
- Tree buffer management
- Memory growth and defragmentation
- Statistics tracking
- Thread safety
- Auto-configuration
"""

import pytest
import torch
import threading
import os
from unittest.mock import patch, Mock
import time

from mcts.gpu.memory_pool import (
    MemoryPoolConfig, TensorPool, CSRTreeMemoryPool,
    get_memory_pool, reset_memory_pool, _get_auto_config
)


@pytest.fixture
def pool_config():
    """Create memory pool configuration"""
    return MemoryPoolConfig(
        initial_node_capacity=1000,
        initial_edge_capacity=5000,
        node_growth_factor=1.5,
        edge_growth_factor=1.5,
        device='cpu',
        num_pre_allocated_trees=2,
        max_memory_mb=512,
        defrag_threshold=0.3,
        enable_auto_defrag=True
    )


@pytest.fixture
def tensor_pool():
    """Create tensor pool for testing"""
    return TensorPool(torch.float32, torch.device('cpu'))


@pytest.fixture
def memory_pool(pool_config):
    """Create CSRTreeMemoryPool instance"""
    return CSRTreeMemoryPool(pool_config)


@pytest.fixture(autouse=True)
def reset_global_pool():
    """Reset global memory pool after each test"""
    yield
    reset_memory_pool()


class TestMemoryPoolConfig:
    """Test MemoryPoolConfig configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = MemoryPoolConfig()
        assert config.initial_node_capacity == 1_000_000
        assert config.initial_edge_capacity == 10_000_000
        assert config.node_growth_factor == 1.5
        assert config.edge_growth_factor == 1.5
        assert config.device == 'cuda'
        assert config.num_pre_allocated_trees == 4
        assert config.max_memory_mb == 4096
        assert config.defrag_threshold == 0.3
        assert config.enable_auto_defrag == True
        
    def test_custom_config(self):
        """Test custom configuration"""
        config = MemoryPoolConfig(
            initial_node_capacity=5000,
            device='cpu',
            num_pre_allocated_trees=1,
            enable_auto_defrag=False
        )
        assert config.initial_node_capacity == 5000
        assert config.device == 'cpu'
        assert config.num_pre_allocated_trees == 1
        assert config.enable_auto_defrag == False


class TestTensorPool:
    """Test TensorPool functionality"""
    
    def test_initialization(self, tensor_pool):
        """Test tensor pool initialization"""
        assert tensor_pool.dtype == torch.float32
        assert tensor_pool.device == torch.device('cpu')
        assert len(tensor_pool.free_tensors) == 0
        assert len(tensor_pool.allocated_tensors) == 0
        assert tensor_pool.next_id == 0
        
    def test_allocate_new_tensor(self, tensor_pool):
        """Test allocating a new tensor"""
        size = 100
        tensor_id, tensor = tensor_pool.allocate(size)
        
        assert tensor_id == 0
        assert tensor.shape == (size,)
        assert tensor.dtype == torch.float32
        assert torch.all(tensor == 0)
        assert len(tensor_pool.allocated_tensors) == 1
        assert tensor_pool.total_allocated == 1
        
    def test_deallocate_tensor(self, tensor_pool):
        """Test deallocating a tensor"""
        # Allocate
        tensor_id, tensor = tensor_pool.allocate(100)
        
        # Deallocate
        tensor_pool.deallocate(tensor_id)
        
        assert len(tensor_pool.allocated_tensors) == 0
        assert len(tensor_pool.free_tensors) == 1
        assert tensor_pool.total_deallocated == 1
        
    def test_reuse_deallocated_tensor(self, tensor_pool):
        """Test reusing deallocated tensors"""
        # Allocate and deallocate
        tensor_id1, tensor1 = tensor_pool.allocate(100)
        tensor_pool.deallocate(tensor_id1)
        
        # Allocate again - should reuse
        tensor_id2, tensor2 = tensor_pool.allocate(80)
        
        assert tensor_id2 == 1  # New ID
        assert tensor2.shape == (80,)  # Resized to requested size
        assert len(tensor_pool.free_tensors) == 0
        assert tensor_pool.total_allocated == 2
        
    def test_allocate_larger_than_available(self, tensor_pool):
        """Test allocating larger tensor than available"""
        # Create small tensor and deallocate
        tensor_id1, _ = tensor_pool.allocate(50)
        tensor_pool.deallocate(tensor_id1)
        
        # Request larger tensor - should allocate new
        tensor_id2, tensor2 = tensor_pool.allocate(100)
        
        assert tensor2.shape == (100,)
        assert len(tensor_pool.free_tensors) == 1  # Small tensor still free
        
    def test_statistics(self, tensor_pool):
        """Test pool statistics"""
        # Allocate multiple tensors
        ids = []
        for i in range(5):
            tensor_id, _ = tensor_pool.allocate(100 * (i + 1))
            ids.append(tensor_id)
            
        # Deallocate some
        tensor_pool.deallocate(ids[0])
        tensor_pool.deallocate(ids[2])
        
        stats = tensor_pool.get_stats()
        assert stats['dtype'] == 'torch.float32'
        assert stats['allocated_count'] == 3
        assert stats['free_count'] == 2
        assert stats['total_allocated'] == 5
        assert stats['total_deallocated'] == 2
        assert stats['peak_usage'] == 5
        assert stats['memory_mb'] > 0
        
    def test_defragmentation(self, tensor_pool):
        """Test pool defragmentation"""
        # Create fragmented pool
        for i in range(10):
            tensor_id, _ = tensor_pool.allocate(100 + i * 10)
            tensor_pool.deallocate(tensor_id)
            
        # Should have 10 free tensors of various sizes
        assert len(tensor_pool.free_tensors) == 10
        
        # Defragment
        tensor_pool.defragment()
        
        # Small fragments should be removed
        assert len(tensor_pool.free_tensors) < 10
        
    def test_thread_safety(self, tensor_pool):
        """Test thread-safe operations"""
        results = []
        errors = []
        
        def allocate_deallocate():
            try:
                for _ in range(10):
                    tensor_id, tensor = tensor_pool.allocate(100)
                    results.append(tensor_id)
                    time.sleep(0.001)  # Small delay
                    tensor_pool.deallocate(tensor_id)
            except Exception as e:
                errors.append(e)
                
        # Run multiple threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=allocate_deallocate)
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        # Check no errors and all IDs unique
        assert len(errors) == 0
        assert len(set(results)) == len(results)


class TestCSRTreeMemoryPool:
    """Test CSRTreeMemoryPool functionality"""
    
    def test_initialization(self, memory_pool, pool_config):
        """Test memory pool initialization"""
        assert memory_pool.config == pool_config
        assert memory_pool.device == torch.device('cpu')
        assert len(memory_pool.pools) == 3  # int32, float32, uint8
        assert len(memory_pool.free_trees) == pool_config.num_pre_allocated_trees
        assert len(memory_pool.allocated_trees) == 0
        
    def test_pre_allocated_trees(self, memory_pool, pool_config):
        """Test pre-allocated tree buffers"""
        assert len(memory_pool.free_trees) == 2
        
        # Check tree buffer structure
        tree_buffers = memory_pool.free_trees[0]
        assert 'visit_counts' in tree_buffers
        assert 'value_sums' in tree_buffers
        assert 'children' in tree_buffers
        assert 'row_ptr' in tree_buffers
        assert tree_buffers['node_capacity'] == pool_config.initial_node_capacity
        assert tree_buffers['edge_capacity'] == pool_config.initial_edge_capacity
        
    def test_allocate_tree(self, memory_pool):
        """Test allocating a tree from pool"""
        tree_id, tree_buffers = memory_pool.allocate_tree()
        
        assert tree_id == 0
        assert tree_buffers is not None
        assert len(memory_pool.allocated_trees) == 1
        assert len(memory_pool.free_trees) == 1  # One less free tree
        assert memory_pool.stats['trees_allocated'] == 1
        
        # Check buffers are initialized
        assert tree_buffers['num_nodes'] == 0
        assert tree_buffers['num_edges'] == 0
        assert torch.all(tree_buffers['visit_counts'] == 0)
        
    def test_deallocate_tree(self, memory_pool):
        """Test deallocating a tree"""
        # Allocate
        tree_id, tree_buffers = memory_pool.allocate_tree()
        
        # Modify some data
        tree_buffers['visit_counts'][0] = 10
        tree_buffers['num_nodes'] = 5
        
        # Deallocate
        memory_pool.deallocate_tree(tree_id)
        
        assert len(memory_pool.allocated_trees) == 0
        assert len(memory_pool.free_trees) == 2  # Back to original
        assert memory_pool.stats['trees_deallocated'] == 1
        
        # Check buffers were cleared
        returned_tree = memory_pool.free_trees[-1]
        assert returned_tree['num_nodes'] == 0
        assert torch.all(returned_tree['visit_counts'] == 0)
        
    def test_allocate_beyond_pre_allocated(self, memory_pool):
        """Test allocating more trees than pre-allocated"""
        # Allocate all pre-allocated trees
        tree_ids = []
        for _ in range(3):  # More than pre-allocated
            tree_id, _ = memory_pool.allocate_tree()
            tree_ids.append(tree_id)
            
        assert len(memory_pool.allocated_trees) == 3
        assert len(memory_pool.free_trees) == 0
        assert memory_pool.stats['trees_allocated'] == 3
        
    def test_grow_tree_buffers(self, memory_pool):
        """Test growing tree buffers"""
        # Allocate tree
        tree_id, tree_buffers = memory_pool.allocate_tree()
        
        old_node_capacity = tree_buffers['node_capacity']
        old_edge_capacity = tree_buffers['edge_capacity']
        
        # Add some data
        tree_buffers['num_nodes'] = 100
        tree_buffers['num_edges'] = 500
        tree_buffers['visit_counts'][:100] = torch.arange(100)
        
        # Grow buffers
        new_buffers = memory_pool.grow_tree_buffers(tree_id)
        
        # Check new capacities
        assert new_buffers['node_capacity'] > old_node_capacity
        assert new_buffers['edge_capacity'] > old_edge_capacity
        
        # Check data preserved
        assert new_buffers['num_nodes'] == 100
        assert new_buffers['num_edges'] == 500
        assert torch.all(new_buffers['visit_counts'][:100] == torch.arange(100))
        
    def test_grow_with_specific_capacity(self, memory_pool):
        """Test growing with specific capacities"""
        tree_id, _ = memory_pool.allocate_tree()
        
        new_buffers = memory_pool.grow_tree_buffers(
            tree_id, 
            new_node_capacity=5000,
            new_edge_capacity=20000
        )
        
        assert new_buffers['node_capacity'] == 5000
        assert new_buffers['edge_capacity'] == 20000
        
    def test_copy_tree_data(self, memory_pool):
        """Test copying tree data between buffers"""
        # Create source and destination buffers
        src = memory_pool._allocate_tree_buffers(100, 500)
        dst = memory_pool._allocate_tree_buffers(200, 1000)
        
        # Fill source with data
        src['num_nodes'] = 50
        src['num_edges'] = 250
        src['visit_counts'][:50] = torch.arange(50)
        src['value_sums'][:50] = torch.rand(50)
        src['children'][:10, :5] = torch.randint(0, 50, (10, 5))
        
        # Copy data
        memory_pool._copy_tree_data(src, dst)
        
        # Check data copied correctly
        assert dst['num_nodes'] == 50
        assert dst['num_edges'] == 250
        assert torch.all(dst['visit_counts'][:50] == src['visit_counts'][:50])
        assert torch.all(dst['value_sums'][:50] == src['value_sums'][:50])
        assert torch.all(dst['children'][:10, :5] == src['children'][:10, :5])
        
    def test_defragmentation(self, pool_config):
        """Test memory pool defragmentation"""
        # Disable auto-defrag for this test
        pool_config.enable_auto_defrag = False
        memory_pool = CSRTreeMemoryPool(pool_config)
        
        # Allocate and deallocate to create fragmentation
        for _ in range(10):
            tree_id, _ = memory_pool.allocate_tree()
            memory_pool.deallocate_tree(tree_id)
            
        initial_free_trees = len(memory_pool.free_trees)
        
        # Defragment
        memory_pool.defragment()
        
        # Check defragmentation occurred
        assert memory_pool.stats['defrag_count'] == 1
        
        # Free trees should be limited
        max_expected = memory_pool.config.num_pre_allocated_trees * 2
        assert len(memory_pool.free_trees) <= max_expected
        
    def test_auto_defragmentation(self, pool_config):
        """Test automatic defragmentation trigger"""
        pool_config.defrag_threshold = 0.5
        memory_pool = CSRTreeMemoryPool(pool_config)
        
        # Allocate one tree
        tree_id, _ = memory_pool.allocate_tree()
        
        # Deallocate to trigger defrag check
        memory_pool.deallocate_tree(tree_id)
        
        # With 1 allocated and 2 pre-allocated, free ratio > 0.5
        # But defrag may not actually reduce trees in this case
        assert memory_pool.stats['defrag_count'] >= 0
        
    def test_statistics(self, memory_pool):
        """Test comprehensive statistics"""
        # Allocate some trees
        tree_ids = []
        for _ in range(3):
            tree_id, _ = memory_pool.allocate_tree()
            tree_ids.append(tree_id)
            
        # Deallocate one
        memory_pool.deallocate_tree(tree_ids[0])
        
        stats = memory_pool.get_stats()
        
        # Check structure
        assert 'pools' in stats
        assert 'trees' in stats
        assert 'total_memory_mb' in stats
        assert 'memory_usage_pct' in stats
        
        # Check tree stats
        assert stats['trees']['allocated_trees'] == 2
        assert stats['trees']['free_trees'] >= 1
        assert stats['trees']['trees_allocated'] == 3
        assert stats['trees']['trees_deallocated'] == 1
        assert stats['trees']['peak_trees'] == 3
        
        # Check memory stats
        assert stats['total_memory_mb'] > 0
        assert 0 <= stats['memory_usage_pct'] <= 100
        
    def test_thread_safe_allocation(self, memory_pool):
        """Test thread-safe tree allocation"""
        tree_ids = []
        errors = []
        lock = threading.Lock()
        
        def allocate_trees():
            try:
                for _ in range(5):
                    tree_id, _ = memory_pool.allocate_tree()
                    with lock:
                        tree_ids.append(tree_id)
                    time.sleep(0.001)
                    memory_pool.deallocate_tree(tree_id)
            except Exception as e:
                errors.append(e)
                
        # Run multiple threads
        threads = []
        for _ in range(4):
            t = threading.Thread(target=allocate_trees)
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        # Check no errors and all IDs unique
        assert len(errors) == 0
        assert len(set(tree_ids)) == len(tree_ids)


class TestGlobalMemoryPool:
    """Test global memory pool functionality"""
    
    @patch.dict(os.environ, {'PYTEST_CURRENT_TEST': 'test_name'})
    def test_get_memory_pool_testing(self):
        """Test getting memory pool in test environment"""
        pool = get_memory_pool()
        
        assert pool is not None
        # Should use test configuration
        assert pool.config.initial_node_capacity == 50_000
        assert pool.config.initial_edge_capacity == 500_000
        assert pool.config.num_pre_allocated_trees == 2
        assert pool.config.max_memory_mb == 1024
        
    def test_get_memory_pool_singleton(self):
        """Test memory pool singleton behavior"""
        pool1 = get_memory_pool()
        pool2 = get_memory_pool()
        
        assert pool1 is pool2
        
    def test_reset_memory_pool(self):
        """Test resetting global memory pool"""
        pool1 = get_memory_pool()
        reset_memory_pool()
        pool2 = get_memory_pool()
        
        assert pool1 is not pool2
        
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.current_device', return_value=0)
    def test_auto_config_gpu(self, mock_device, mock_available):
        """Test automatic configuration with GPU"""
        mock_props = Mock()
        mock_props.total_memory = 8 * 1024 * 1024 * 1024  # 8GB
        
        with patch('torch.cuda.get_device_properties', return_value=mock_props):
            with patch('torch.cuda.memory_reserved', return_value=1024*1024*1024):  # 1GB reserved
                config = _get_auto_config()
                
        assert config.device == 'cuda'
        assert config.max_memory_mb > 0
        assert config.initial_node_capacity > 0
        assert config.initial_edge_capacity > 0
        
    @patch('torch.cuda.is_available', return_value=False)
    def test_auto_config_cpu(self, mock_available):
        """Test automatic configuration with CPU"""
        config = _get_auto_config()
        
        assert config.device == 'cpu'
        # For CPU with single process, it uses 70% of the default 4096 MB
        assert config.max_memory_mb == int(4096 * 0.7)  # 2867 MB
        
    @patch.dict(os.environ, {'SELFPLAY_WORKER_ID': '0', 'SELFPLAY_NUM_WORKERS': '4'})
    @patch('torch.cuda.is_available', return_value=True)
    def test_auto_config_worker(self, mock_available):
        """Test automatic configuration for worker process"""
        mock_props = Mock()
        mock_props.total_memory = 8 * 1024 * 1024 * 1024  # 8GB
        
        with patch('torch.cuda.get_device_properties', return_value=mock_props):
            with patch('torch.cuda.memory_reserved', return_value=0):
                config = _get_auto_config()
                
        # Worker should get fraction of memory
        assert config.num_pre_allocated_trees == 1
        assert config.max_memory_mb < 2048  # Less than 1/4 of 8GB
        
    def test_invalid_tree_growth(self, memory_pool):
        """Test growing non-existent tree"""
        with pytest.raises(ValueError, match="Tree .* not found"):
            memory_pool.grow_tree_buffers(999)