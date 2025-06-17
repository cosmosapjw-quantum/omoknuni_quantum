"""
Comprehensive tests for CSRTree memory pool functionality

Tests cover:
- Memory pool initialization and configuration
- Tree allocation and deallocation
- Buffer growth operations
- Memory reuse and fragmentation prevention
- Thread safety
- Edge cases and error handling
"""

import pytest
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import logging

from mcts.gpu.memory_pool import TensorPool, CSRTreeMemoryPool, MemoryPoolConfig
from mcts.gpu.csr_tree import CSRTree, CSRTreeConfig

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestTensorPool:
    """Test cases for the TensorPool class"""
    
    def test_initialization(self):
        """Test TensorPool initialization"""
        pool = TensorPool(
            dtype=torch.float32,
            device='cpu',
            initial_capacity=1000,
            growth_factor=1.5
        )
        
        assert pool.dtype == torch.float32
        assert pool.device == torch.device('cpu')
        assert pool.initial_capacity == 1000
        assert pool.growth_factor == 1.5
        assert len(pool.buffers) == 0
        assert len(pool.free_list) == 0
    
    def test_allocate_tensor(self):
        """Test tensor allocation from pool"""
        pool = TensorPool(dtype=torch.int32, device='cpu')
        
        # First allocation should create new buffer
        buffer_id, tensor = pool.allocate(100)
        assert buffer_id == 0
        assert tensor.shape == (100,)
        assert tensor.dtype == torch.int32
        assert len(pool.buffers) == 1
        assert 0 not in pool.free_list
        
        # Second allocation should create another buffer
        buffer_id2, tensor2 = pool.allocate(200)
        assert buffer_id2 == 1
        assert tensor2.shape == (200,)
        assert len(pool.buffers) == 2
    
    def test_deallocate_and_reuse(self):
        """Test tensor deallocation and reuse"""
        pool = TensorPool(dtype=torch.float32, device='cpu', initial_capacity=100)
        
        # Allocate and deallocate
        buffer_id, tensor = pool.allocate(100)
        pool.deallocate(buffer_id)
        
        assert buffer_id in pool.free_list
        assert pool.free_list[buffer_id] == 100
        
        # Reuse should return same buffer
        buffer_id2, tensor2 = pool.allocate(80)
        assert buffer_id2 == buffer_id
        assert tensor2.data_ptr() == tensor.data_ptr()
        assert tensor2.shape == (80,)
    
    def test_growth_on_insufficient_capacity(self):
        """Test buffer growth when capacity is insufficient"""
        pool = TensorPool(
            dtype=torch.float32,
            device='cpu',
            initial_capacity=100,
            growth_factor=2.0
        )
        
        # Allocate buffer
        buffer_id, tensor = pool.allocate(100)
        assert tensor.shape == (100,)
        
        # Request growth
        new_tensor = pool.grow(buffer_id, 250)
        assert new_tensor.shape == (250,)
        assert pool.buffers[buffer_id].shape == (250,)
        
        # Old data should be preserved (zeros in this case)
        assert torch.all(new_tensor[:100] == 0)
    
    def test_find_suitable_buffer(self):
        """Test finding suitable buffer from free list"""
        pool = TensorPool(dtype=torch.float32, device='cpu', initial_capacity=100)
        
        # Create buffers of different sizes
        id1, _ = pool.allocate(50)
        id2, _ = pool.allocate(100)
        id3, _ = pool.allocate(150)
        
        # Deallocate all
        pool.deallocate(id1)
        pool.deallocate(id2)
        pool.deallocate(id3)
        
        # Request should find smallest suitable buffer
        buffer_id, tensor = pool.allocate(80)
        assert buffer_id == id2  # 100 capacity buffer
        assert tensor.shape == (80,)
        
        # Larger request should find next suitable
        buffer_id, tensor = pool.allocate(120)
        assert buffer_id == id3  # 150 capacity buffer
        assert tensor.shape == (120,)


class TestCSRTreeMemoryPool:
    """Test cases for the CSRTreeMemoryPool class"""
    
    @pytest.fixture
    def memory_pool(self):
        """Create a test memory pool"""
        config = MemoryPoolConfig(
            initial_node_capacity=1000,
            initial_edge_capacity=5000,
            num_pre_allocated_trees=2,
            device='cpu'
        )
        return CSRTreeMemoryPool(config)
    
    def test_initialization(self, memory_pool):
        """Test memory pool initialization"""
        assert memory_pool.config.initial_node_capacity == 1000
        assert memory_pool.config.initial_edge_capacity == 5000
        assert memory_pool.config.num_pre_allocated_trees == 2
        assert memory_pool.device == torch.device('cpu')
        assert len(memory_pool.allocated_trees) == 0
        
        # Check tensor pools are created
        assert 'int32' in memory_pool.pools
        assert 'float32' in memory_pool.pools
        assert 'uint8' in memory_pool.pools
    
    def test_allocate_tree(self, memory_pool):
        """Test tree allocation from pool"""
        tree_id, buffers = memory_pool.allocate_tree()
        
        assert tree_id == 0
        assert tree_id in memory_pool.allocated_trees
        assert memory_pool.allocated_trees[tree_id] is not None
        
        # Check all required buffers are allocated
        required_buffers = [
            'node_values', 'node_visits', 'node_virtual_losses',
            'edge_child_indices', 'edge_parent_indices', 'edge_actions',
            'edge_visits', 'edge_priors', 'edge_values'
        ]
        
        for buffer_name in required_buffers:
            assert buffer_name in buffers
            assert isinstance(buffers[buffer_name], torch.Tensor)
        
        # Check buffer sizes
        assert buffers['node_values'].shape == (1000,)
        assert buffers['edge_visits'].shape == (5000,)
    
    def test_deallocate_tree(self, memory_pool):
        """Test tree deallocation"""
        # Allocate and deallocate
        tree_id, _ = memory_pool.allocate_tree()
        memory_pool.deallocate_tree(tree_id)
        
        assert tree_id not in memory_pool.allocated_trees
        
        # Should be able to reuse
        tree_id2, _ = memory_pool.allocate_tree()
        assert tree_id2 == tree_id
    
    def test_grow_tree_buffers(self, memory_pool):
        """Test growing tree buffers"""
        tree_id, buffers = memory_pool.allocate_tree()
        
        # Grow buffers
        new_buffers = memory_pool.grow_tree_buffers(
            tree_id,
            new_node_capacity=2000,
            new_edge_capacity=10000
        )
        
        # Check new sizes
        assert new_buffers['node_values'].shape == (2000,)
        assert new_buffers['edge_visits'].shape == (10000,)
        
        # Verify growth was tracked
        alloc_info = memory_pool.allocated_trees[tree_id]
        assert alloc_info['node_values'][1].shape == (2000,)
        assert alloc_info['edge_visits'][1].shape == (10000,)
    
    def test_get_memory_stats(self, memory_pool):
        """Test memory statistics reporting"""
        # Allocate some trees
        tree_id1, _ = memory_pool.allocate_tree()
        tree_id2, _ = memory_pool.allocate_tree()
        memory_pool.deallocate_tree(tree_id1)
        
        stats = memory_pool.get_stats()
        
        assert len(memory_pool.allocated_trees) == 1
        assert stats['trees_allocated'] == 2
        assert stats['trees_deallocated'] == 1
        assert stats['peak_trees'] >= 2
        assert 'memory_mb' in stats
        assert stats['memory_mb'] > 0
    
    def test_clear_pool(self, memory_pool):
        """Test clearing the memory pool"""
        # Allocate trees
        tree_id1, _ = memory_pool.allocate_tree()
        tree_id2, _ = memory_pool.allocate_tree()
        
        # Clear pool
        memory_pool.clear()
        
        assert len(memory_pool.allocated_trees) == 0
        assert memory_pool.next_tree_id == 0
        
        # Should be able to allocate again
        tree_id, _ = memory_pool.allocate_tree()
        assert tree_id == 0
    
    def test_thread_safety(self, memory_pool):
        """Test thread-safe allocation and deallocation"""
        num_threads = 4
        allocations_per_thread = 10
        all_tree_ids = []
        
        def allocate_and_deallocate():
            tree_ids = []
            for _ in range(allocations_per_thread):
                tree_id, _ = memory_pool.allocate_tree()
                tree_ids.append(tree_id)
                # Simulate some work
                time.sleep(0.001)
            
            # Deallocate half
            for tree_id in tree_ids[:len(tree_ids)//2]:
                memory_pool.deallocate_tree(tree_id)
            
            return tree_ids[len(tree_ids)//2:]
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(allocate_and_deallocate) 
                      for _ in range(num_threads)]
            
            for future in futures:
                all_tree_ids.extend(future.result())
        
        # Check no duplicate allocations
        assert len(all_tree_ids) == len(set(all_tree_ids))
        
        # Check correct number of allocated trees
        assert len(memory_pool.allocated_trees) == len(all_tree_ids)


class TestCSRTreeMemoryPoolIntegration:
    """Integration tests for CSRTree with memory pool"""
    
    @pytest.fixture
    def config(self):
        """Create CSRTree configuration"""
        return CSRTreeConfig(
            max_nodes=1000,
            max_children=225,  # For 15x15 Gomoku
            device='cpu',
            dtype=torch.float32
        )
    
    @pytest.fixture
    def memory_pool(self):
        """Create memory pool for integration tests"""
        config = MemoryPoolConfig(
            initial_node_capacity=1000,
            initial_edge_capacity=5000,
            device='cpu'
        )
        return CSRTreeMemoryPool(config)
    
    def test_csr_tree_with_memory_pool(self, config, memory_pool):
        """Test CSRTree creation with memory pool"""
        # Allocate tree from pool
        tree_id, buffers = memory_pool.allocate_tree()
        
        # Create CSRTree with pool
        tree = CSRTree(config, memory_pool_tree_id=tree_id)
        tree.set_memory_pool(memory_pool)
        
        # Initialize with pool buffers
        tree._initialize_with_pool_buffers(buffers)
        
        # Verify tree is using pool buffers
        assert tree._using_pool_buffers
        assert tree.memory_pool_tree_id == tree_id
        
        # Test basic operations
        root_id = tree.add_node(is_root=True)
        assert root_id == 0
        
        # Add children
        child_ids = []
        for action in range(5):
            child_id = tree.add_node()
            edge_id = tree.add_edge(root_id, child_id, action, prior=0.2)
            child_ids.append(child_id)
        
        assert tree.get_children_count(root_id) == 5
    
    def test_tree_reset_preserves_memory(self, config, memory_pool):
        """Test that tree reset preserves memory allocations"""
        # Create tree with memory pool
        tree_id, buffers = memory_pool.allocate_tree()
        tree = CSRTree(config, memory_pool_tree_id=tree_id)
        tree.set_memory_pool(memory_pool)
        tree._initialize_with_pool_buffers(buffers)
        
        # Add some nodes
        root_id = tree.add_node(is_root=True)
        for i in range(10):
            child_id = tree.add_node()
            tree.add_edge(root_id, child_id, i, prior=0.1)
        
        # Record buffer pointers
        node_values_ptr = tree.node_values.data_ptr()
        edge_visits_ptr = tree.edge_visits.data_ptr()
        
        # Reset tree
        tree.reset()
        
        # Verify buffers are preserved
        assert tree.node_values.data_ptr() == node_values_ptr
        assert tree.edge_visits.data_ptr() == edge_visits_ptr
        assert tree.num_nodes == 0
        assert tree.num_edges == 0
        
        # Verify we can still use the tree
        new_root = tree.add_node(is_root=True)
        assert new_root == 0
    
    def test_automatic_buffer_growth(self, config, memory_pool):
        """Test automatic buffer growth through memory pool"""
        # Create small initial capacity
        config = MemoryPoolConfig(
            initial_node_capacity=10,
            initial_edge_capacity=20,
            device='cpu'
        )
        small_pool = CSRTreeMemoryPool(config)
        
        tree_id, buffers = small_pool.allocate_tree()
        tree = CSRTree(config, memory_pool_tree_id=tree_id)
        tree.set_memory_pool(small_pool)
        tree._initialize_with_pool_buffers(buffers)
        
        # Add nodes beyond initial capacity
        root_id = tree.add_node(is_root=True)
        
        # This should trigger growth
        for i in range(15):
            child_id = tree.add_node()
            tree.add_edge(root_id, child_id, i, prior=0.1)
        
        # Verify growth happened
        assert tree.node_capacity > 10
        assert tree.edge_capacity > 20
        assert tree.num_nodes == 16
        assert tree.num_edges == 15
    
    def test_memory_pool_statistics_after_operations(self, config, memory_pool):
        """Test memory pool statistics after various operations"""
        # Allocate multiple trees
        trees = []
        for i in range(3):
            tree_id, buffers = memory_pool.allocate_tree()
            tree = CSRTree(config, memory_pool_tree_id=tree_id)
            tree.set_memory_pool(memory_pool)
            tree._initialize_with_pool_buffers(buffers)
            trees.append((tree_id, tree))
        
        # Perform operations on trees
        for tree_id, tree in trees:
            root_id = tree.add_node(is_root=True)
            for j in range(50):
                child_id = tree.add_node()
                tree.add_edge(root_id, child_id, j, prior=0.02)
        
        # Deallocate one tree
        memory_pool.deallocate_tree(trees[0][0])
        
        # Check statistics
        stats = memory_pool.get_stats()
        assert len(memory_pool.allocated_trees) == 2
        assert stats['trees_allocated'] == 3
        assert stats['trees_deallocated'] == 1
        
        # Memory should be reasonable
        assert 0 < stats['memory_mb'] < 100  # Reasonable bounds
    
    def test_error_handling(self, config, memory_pool):
        """Test error handling in memory pool operations"""
        # Test deallocating non-existent tree
        with pytest.raises(ValueError, match="not allocated"):
            memory_pool.deallocate_tree(999)
        
        # Test growing non-existent tree
        with pytest.raises(ValueError, match="not allocated"):
            memory_pool.grow_tree_buffers(999, 2000)
        
        # Test double deallocation
        tree_id, _ = memory_pool.allocate_tree()
        memory_pool.deallocate_tree(tree_id)
        with pytest.raises(ValueError, match="not allocated"):
            memory_pool.deallocate_tree(tree_id)


class TestMemoryPoolPerformance:
    """Performance tests for memory pool"""
    
    @pytest.mark.slow
    def test_allocation_performance(self):
        """Benchmark allocation performance"""
        config = MemoryPoolConfig(
            initial_node_capacity=100000,
            initial_edge_capacity=1000000,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        memory_pool = CSRTreeMemoryPool(config)
        
        num_allocations = 100
        start_time = time.time()
        
        tree_ids = []
        for _ in range(num_allocations):
            tree_id, _ = memory_pool.allocate_tree()
            tree_ids.append(tree_id)
        
        allocation_time = time.time() - start_time
        
        # Deallocate all
        start_time = time.time()
        for tree_id in tree_ids:
            memory_pool.deallocate_tree(tree_id)
        deallocation_time = time.time() - start_time
        
        # Reallocate (should be faster due to reuse)
        start_time = time.time()
        for _ in range(num_allocations):
            tree_id, _ = memory_pool.allocate_tree()
        reallocation_time = time.time() - start_time
        
        logger.info(f"Allocation time: {allocation_time:.3f}s")
        logger.info(f"Deallocation time: {deallocation_time:.3f}s")
        logger.info(f"Reallocation time: {reallocation_time:.3f}s")
        
        # Reallocation should be faster
        assert reallocation_time < allocation_time * 0.5
    
    @pytest.mark.slow
    def test_memory_fragmentation_prevention(self):
        """Test that memory pool prevents fragmentation"""
        config = MemoryPoolConfig(
            initial_node_capacity=10000,
            initial_edge_capacity=100000,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        memory_pool = CSRTreeMemoryPool(config)
        
        # Simulate many allocation/deallocation cycles
        num_cycles = 50
        trees_per_cycle = 10
        
        cycle_times = []
        for cycle in range(num_cycles):
            start_time = time.time()
            
            # Allocate trees
            tree_ids = []
            for _ in range(trees_per_cycle):
                tree_id, _ = memory_pool.allocate_tree()
                tree_ids.append(tree_id)
            
            # Deallocate half randomly
            import random
            random.shuffle(tree_ids)
            for tree_id in tree_ids[:len(tree_ids)//2]:
                memory_pool.deallocate_tree(tree_id)
            
            cycle_time = time.time() - start_time
            cycle_times.append(cycle_time)
        
        # Performance should not degrade significantly
        first_quarter_avg = np.mean(cycle_times[:num_cycles//4])
        last_quarter_avg = np.mean(cycle_times[-num_cycles//4:])
        
        logger.info(f"First quarter avg: {first_quarter_avg:.3f}s")
        logger.info(f"Last quarter avg: {last_quarter_avg:.3f}s")
        
        # Last quarter should not be much slower than first
        assert last_quarter_avg < first_quarter_avg * 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])