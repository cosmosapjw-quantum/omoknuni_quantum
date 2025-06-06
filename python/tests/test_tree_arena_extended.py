"""Extended tests for TreeArena to achieve 90%+ coverage"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time
import tempfile
import os
from collections import OrderedDict

# Defer torch import to avoid double import issues
torch = None
HAS_TORCH = False

def _check_torch():
    global torch, HAS_TORCH
    if torch is None:
        try:
            import torch as _torch
            torch = _torch
            HAS_TORCH = True
        except ImportError:
            HAS_TORCH = False
    return HAS_TORCH

from mcts.tree_arena import TreeArena, MemoryConfig, NodePage
from mcts.node import Node


class TestNodePage:
    """Test NodePage class directly"""
    
    def test_node_page_initialization(self):
        """Test NodePage basic initialization"""
        page = NodePage(page_id=1, capacity=10)
        
        assert page.page_id == 1
        assert page.capacity == 10
        assert len(page.nodes) == 0
        assert page.is_on_gpu is False
        assert page.gpu_tensors is None
        assert page.last_access_time > 0
        
    def test_node_page_operations(self):
        """Test NodePage add/get operations"""
        page = NodePage(page_id=1, capacity=3)
        
        # Add nodes
        node1 = Mock(spec=Node)
        node2 = Mock(spec=Node)
        
        initial_time = page.last_access_time
        time.sleep(0.01)  # Small delay to ensure time difference
        
        page.add_node("node_1", node1)
        assert len(page.nodes) == 1
        assert page.last_access_time > initial_time
        
        # Get node updates access time
        time.sleep(0.01)
        access_time_before_get = page.last_access_time
        retrieved = page.get_node("node_1")
        assert retrieved == node1
        assert page.last_access_time > access_time_before_get
        
        # Test missing node
        assert page.get_node("nonexistent") is None
        
    def test_node_page_capacity(self):
        """Test NodePage capacity checking"""
        page = NodePage(page_id=1, capacity=2)
        
        assert not page.is_full()
        
        page.add_node("node_1", Mock(spec=Node))
        assert not page.is_full()
        
        page.add_node("node_2", Mock(spec=Node))
        assert page.is_full()
        
    def test_node_page_memory_usage(self):
        """Test memory usage calculation"""
        page = NodePage(page_id=1, capacity=10)
        node_size = 100
        
        assert page.memory_usage(node_size) == 0
        
        page.add_node("node_1", Mock(spec=Node))
        assert page.memory_usage(node_size) == 100
        
        page.add_node("node_2", Mock(spec=Node))
        assert page.memory_usage(node_size) == 200


class TestMemoryConfigPresets:
    """Test all memory configuration presets"""
    
    def test_all_presets(self):
        """Test all preset configurations"""
        desktop = MemoryConfig.desktop_preset()
        laptop = MemoryConfig.laptop_preset()
        cloud = MemoryConfig.cloud_preset()
        
        # Check relative sizes
        assert cloud.gpu_memory_limit > desktop.gpu_memory_limit > laptop.gpu_memory_limit
        assert cloud.cpu_memory_limit >= desktop.cpu_memory_limit > laptop.cpu_memory_limit
        assert cloud.page_size > desktop.page_size > laptop.page_size
        
        # Check all have mixed precision enabled
        assert desktop.enable_mixed_precision
        assert laptop.enable_mixed_precision
        assert cloud.enable_mixed_precision
        
    def test_custom_config(self):
        """Test custom configuration"""
        config = MemoryConfig(
            gpu_memory_limit=1024,
            cpu_memory_limit=2048,
            page_size=100,
            node_size_bytes=50,
            enable_mixed_precision=False,
            fp16_visit_threshold=500,
            gc_threshold=0.7
        )
        
        assert config.gpu_memory_limit == 1024
        assert config.cpu_memory_limit == 2048
        assert config.page_size == 100
        assert config.node_size_bytes == 50
        assert not config.enable_mixed_precision
        assert config.fp16_visit_threshold == 500
        assert config.gc_threshold == 0.7


class TestTreeArenaAdvanced:
    """Advanced TreeArena tests for complete coverage"""
    
    def test_arena_without_gpu(self):
        """Test arena when GPU is not available"""
        config = MemoryConfig(gpu_memory_limit=1024, cpu_memory_limit=2048)
        
        # Test with use_gpu=False
        arena = TreeArena(config, use_gpu=False)
        assert not arena.use_gpu
        
        # Test when torch is not available
        with patch('mcts.tree_arena.HAS_TORCH', False):
            arena = TreeArena(config, use_gpu=True)
            assert not arena.use_gpu
            
    def test_find_available_page(self):
        """Test _find_available_page method"""
        config = MemoryConfig(
            gpu_memory_limit=1000,
            cpu_memory_limit=2000,
            page_size=5,
            node_size_bytes=100
        )
        arena = TreeArena(config)
        
        # No pages initially
        assert arena._find_available_page() is None
        
        # Create GPU page with space
        gpu_page = NodePage(1, capacity=5)
        gpu_page.is_on_gpu = True
        gpu_page.add_node("test1", Mock(spec=Node))
        arena.gpu_pages[1] = gpu_page
        arena.node_pages[1] = gpu_page
        
        # Should find GPU page first
        found = arena._find_available_page()
        assert found == gpu_page
        
        # Fill GPU page
        for i in range(2, 6):
            gpu_page.add_node(f"test{i}", Mock(spec=Node))
        assert gpu_page.is_full()
        
        # Create CPU page with space
        cpu_page = NodePage(2, capacity=5)
        cpu_page.is_on_gpu = False
        cpu_page.add_node("cpu1", Mock(spec=Node))
        arena.cpu_pages[2] = cpu_page
        arena.node_pages[2] = cpu_page
        
        # Should find CPU page when GPU is full
        found = arena._find_available_page()
        assert found == cpu_page
        
    def test_evict_lru_gpu_page_empty(self):
        """Test LRU eviction when no GPU pages exist"""
        config = MemoryConfig(gpu_memory_limit=1024, cpu_memory_limit=2048)
        arena = TreeArena(config)
        
        # Should return False when no GPU pages
        assert not arena._evict_lru_gpu_page()
        
    def test_complex_memory_paging_scenario(self):
        """Test complex scenario with multiple page movements"""
        config = MemoryConfig(
            gpu_memory_limit=300,  # 3 nodes max
            cpu_memory_limit=1000,
            page_size=2,  # 2 nodes per page
            node_size_bytes=100
        )
        arena = TreeArena(config)
        
        # Add nodes to fill multiple pages
        node_ids = []
        for i in range(10):
            node = Node(state=f"state_{i}", parent=None, action=None, prior=0.1)
            node_id = arena.add_node(node)
            node_ids.append(node_id)
            
        # GPU should have only 1 page (2 nodes, 200 bytes)
        assert arena.gpu_nodes <= 2
        assert len(arena.gpu_pages) <= 1
        
        # Access old nodes to trigger page movement
        for i in range(3):
            node = arena.get_node(node_ids[i])
            assert node is not None
            
        # Check memory state
        assert arena.total_nodes == 10
        assert arena.gpu_nodes + arena.cpu_nodes == 10
        
    def test_can_fit_on_gpu(self):
        """Test _can_fit_on_gpu method"""
        config = MemoryConfig(
            gpu_memory_limit=500,
            cpu_memory_limit=1000,
            node_size_bytes=100
        )
        
        # Test without GPU
        arena = TreeArena(config, use_gpu=False)
        page = NodePage(1, capacity=5)
        assert not arena._can_fit_on_gpu(page)
        
        # Test with GPU
        arena = TreeArena(config, use_gpu=True)
        
        # Empty page should fit
        assert arena._can_fit_on_gpu(page)
        
        # Fill GPU memory
        arena.gpu_memory_used = 400
        
        # Page with 2 nodes (200 bytes) should not fit
        page.add_node("n1", Mock(spec=Node))
        page.add_node("n2", Mock(spec=Node))
        assert not arena._can_fit_on_gpu(page)
        
    def test_move_page_no_op(self):
        """Test _move_page when page is already at destination"""
        config = MemoryConfig(gpu_memory_limit=1024, cpu_memory_limit=2048)
        arena = TreeArena(config)
        
        # Create page on GPU
        page = NodePage(1, capacity=5)
        page.is_on_gpu = True
        page.add_node("n1", Mock(spec=Node))
        arena.gpu_pages[1] = page
        arena.node_pages[1] = page
        arena.node_registry["n1"] = (1, 'gpu')
        arena.gpu_nodes = 1
        arena.gpu_memory_used = 100
        
        # Try to move to GPU (no-op)
        initial_gpu_nodes = arena.gpu_nodes
        arena._move_page(page, to_gpu=True)
        assert arena.gpu_nodes == initial_gpu_nodes
        assert page.is_on_gpu
        
    def test_move_page_cpu_to_gpu(self):
        """Test moving page from CPU to GPU"""
        config = MemoryConfig(
            gpu_memory_limit=1024,
            cpu_memory_limit=2048,
            node_size_bytes=100
        )
        arena = TreeArena(config)
        
        # Create page on CPU
        page = NodePage(1, capacity=5)
        page.is_on_gpu = False
        page.add_node("n1", Mock(spec=Node))
        page.add_node("n2", Mock(spec=Node))
        arena.cpu_pages[1] = page
        arena.node_pages[1] = page
        arena.node_registry["n1"] = (1, 'cpu')
        arena.node_registry["n2"] = (1, 'cpu')
        arena.cpu_nodes = 2
        arena.cpu_memory_used = 200
        
        # Move to GPU
        arena._move_page(page, to_gpu=True)
        
        # Verify state
        assert page.is_on_gpu
        assert 1 in arena.gpu_pages
        assert 1 not in arena.cpu_pages
        assert arena.gpu_nodes == 2
        assert arena.cpu_nodes == 0
        assert arena.gpu_memory_used == 200
        assert arena.cpu_memory_used == 0
        assert arena.node_registry["n1"] == (1, 'gpu')
        assert arena.node_registry["n2"] == (1, 'gpu')
        
    def test_gc_with_empty_tree(self):
        """Test garbage collection with no nodes"""
        config = MemoryConfig(
            gpu_memory_limit=1024,
            cpu_memory_limit=1024,
            gc_threshold=0.8
        )
        arena = TreeArena(config)
        
        # Run GC on empty tree
        arena._run_garbage_collection()
        assert arena.total_nodes == 0
        assert len(arena.gc_history) == 1
        assert arena.gc_history[0]['removed'] == 0
        
    def test_gc_removes_low_importance_nodes(self):
        """Test GC removes nodes with low importance"""
        config = MemoryConfig(
            gpu_memory_limit=500,
            cpu_memory_limit=500,
            gc_threshold=0.8,
            node_size_bytes=100
        )
        arena = TreeArena(config)
        
        # Add nodes with varying visit counts
        high_importance_ids = []
        low_importance_ids = []
        
        # High importance nodes
        for i in range(3):
            node = Node(state=f"important_{i}", parent=None, action=None, prior=0.9)
            node.visit_count = 100 + i * 10
            node_id = arena.add_node(node)
            high_importance_ids.append(node_id)
            
        # Low importance nodes
        for i in range(3):
            node = Node(state=f"unimportant_{i}", parent=None, action=None, prior=0.1)
            node.visit_count = 1
            node_id = arena.add_node(node)
            low_importance_ids.append(node_id)
            
        initial_nodes = arena.total_nodes
        
        # Force GC
        arena._run_garbage_collection()
        
        # Check that some nodes were removed
        assert arena.total_nodes < initial_nodes
        assert len(arena.gc_history) == 1
        
        # High importance nodes should still exist
        for node_id in high_importance_ids:
            assert node_id in arena.node_registry
            
        # At least some low importance nodes should be removed
        removed_count = sum(1 for node_id in low_importance_ids 
                          if node_id not in arena.node_registry)
        assert removed_count > 0
        
    def test_gc_removes_empty_pages(self):
        """Test GC removes empty pages after node removal"""
        config = MemoryConfig(
            gpu_memory_limit=1000,
            cpu_memory_limit=1000,
            page_size=2,
            node_size_bytes=100
        )
        arena = TreeArena(config)
        
        # Create a page with one node
        node = Node(state="lonely", parent=None, action=None, prior=0.1)
        node.visit_count = 1
        node_id = arena.add_node(node)
        
        page_id = arena.node_registry[node_id][0]
        assert page_id in arena.node_pages
        
        # Force GC - should remove the low importance node
        arena._run_garbage_collection()
        
        # Page should be removed since it's empty
        assert page_id not in arena.node_pages
        assert page_id not in arena.gpu_pages
        assert page_id not in arena.cpu_pages
        
    def test_recalculate_memory_usage(self):
        """Test memory usage recalculation"""
        config = MemoryConfig(
            gpu_memory_limit=1000,
            cpu_memory_limit=1000,
            node_size_bytes=50
        )
        arena = TreeArena(config)
        
        # Add nodes
        for i in range(5):
            node = Node(state=f"state_{i}", parent=None, action=None, prior=0.1)
            arena.add_node(node)
            
        # Manually corrupt memory values
        arena.gpu_memory_used = 999999
        arena.cpu_memory_used = 888888
        
        # Recalculate
        arena._recalculate_memory_usage()
        
        # Should be corrected based on actual pages
        total_memory = arena.gpu_memory_used + arena.cpu_memory_used
        expected_memory = arena.total_nodes * config.node_size_bytes
        assert total_memory == expected_memory
        
    def test_maybe_move_page_to_gpu_scenarios(self):
        """Test various scenarios for _maybe_move_page_to_gpu"""
        config = MemoryConfig(
            gpu_memory_limit=200,  # 2 nodes max
            cpu_memory_limit=1000,
            node_size_bytes=100
        )
        arena = TreeArena(config)
        
        # Scenario 1: Page already on GPU
        gpu_page = NodePage(1, capacity=5)
        gpu_page.is_on_gpu = True
        arena._maybe_move_page_to_gpu(gpu_page)
        assert gpu_page.is_on_gpu  # Should remain on GPU
        
        # Scenario 2: use_gpu is False
        arena.use_gpu = False
        cpu_page = NodePage(2, capacity=5)
        cpu_page.is_on_gpu = False
        arena._maybe_move_page_to_gpu(cpu_page)
        assert not cpu_page.is_on_gpu  # Should remain on CPU
        
        # Scenario 3: Not enough space even after eviction
        arena.use_gpu = True
        
        # Fill GPU with important nodes
        important_page = NodePage(3, capacity=2)
        important_page.is_on_gpu = True
        for i in range(2):
            node = Mock(spec=Node)
            node.visit_count = 1000  # High importance
            important_page.add_node(f"important_{i}", node)
        arena.gpu_pages[3] = important_page
        arena.node_pages[3] = important_page
        arena.gpu_memory_used = 200  # Full
        
        # Try to move large page
        large_page = NodePage(4, capacity=5)
        large_page.is_on_gpu = False
        for i in range(3):  # 300 bytes - won't fit
            large_page.add_node(f"large_{i}", Mock(spec=Node))
        arena.cpu_pages[4] = large_page
        arena.node_pages[4] = large_page
        
        # Mock eviction to fail
        with patch.object(arena, '_evict_lru_gpu_page', return_value=False):
            arena._maybe_move_page_to_gpu(large_page)
            assert not large_page.is_on_gpu  # Should remain on CPU
            
    def test_get_gpu_tensors_empty(self):
        """Test get_gpu_tensors with no GPU pages"""
        config = MemoryConfig(gpu_memory_limit=1024, cpu_memory_limit=2048)
        
        # Test without torch
        with patch('mcts.tree_arena.HAS_TORCH', False):
            arena = TreeArena(config)
            assert arena.get_gpu_tensors() == {}
            
        # Test without GPU
        arena = TreeArena(config, use_gpu=False)
        assert arena.get_gpu_tensors() == {}
        
        # Test with GPU but no pages
        arena = TreeArena(config, use_gpu=True)
        assert arena.get_gpu_tensors() == {}
        
    @pytest.mark.skipif(not _check_torch(), reason="PyTorch not installed")
    def test_get_gpu_tensors_with_data(self):
        """Test get_gpu_tensors with actual data"""
        config = MemoryConfig(gpu_memory_limit=10240, cpu_memory_limit=20480)
        arena = TreeArena(config, use_gpu=True)
        
        # Create GPU page with nodes
        page = NodePage(1, capacity=10)
        page.is_on_gpu = True
        
        # Add nodes with different state types
        # Numpy array state
        node1 = Node(state=np.array([[1, 2], [3, 4]]), parent=None, action=None, prior=0.5)
        node1.visit_count = 10
        node1.value_sum = 5.0
        page.add_node("n1", node1)
        
        # Scalar state
        node2 = Node(state=42, parent=None, action=None, prior=0.3)
        node2.visit_count = 20
        node2.value_sum = 15.0
        page.add_node("n2", node2)
        
        # None state (should be skipped)
        node3 = Node(state=None, parent=None, action=None, prior=0.2)
        node3.visit_count = 5
        node3.value_sum = 2.0
        page.add_node("n3", node3)
        
        arena.gpu_pages[1] = page
        arena.node_pages[1] = page
        
        # Mock CUDA availability
        with patch('torch.cuda.is_available', return_value=True):
            # Mock tensor operations
            mock_tensor = MagicMock()
            mock_tensor.cuda.return_value = mock_tensor
            
            with patch('torch.from_numpy', return_value=mock_tensor):
                with patch('torch.tensor', return_value=mock_tensor):
                    with patch('torch.stack', return_value=mock_tensor):
                        tensors = arena.get_gpu_tensors()
                        
                        assert 'states' in tensors
                        assert 'priors' in tensors
                        assert 'values' in tensors
                        assert 'visit_counts' in tensors
                        
    def test_statistics_empty_tree(self):
        """Test statistics on empty tree"""
        config = MemoryConfig(gpu_memory_limit=1024, cpu_memory_limit=2048)
        arena = TreeArena(config)
        
        stats = arena.get_statistics()
        assert stats['total_nodes'] == 0
        assert stats['gpu_nodes'] == 0
        assert stats['cpu_nodes'] == 0
        assert stats['depth'] == 0
        assert stats['branching_factor'] == 0
        assert stats['gpu_pages'] == 0
        assert stats['cpu_pages'] == 0
        
    def test_save_load_edge_cases(self):
        """Test save/load with edge cases"""
        config = MemoryConfig(gpu_memory_limit=1024, cpu_memory_limit=2048)
        arena = TreeArena(config)
        
        # Add nodes with various states
        node1 = Node(state="string_state", parent=None, action=None, prior=0.5)
        node1.visit_count = 10
        node1.value_sum = 5.0
        
        node2 = Node(state=np.array([1, 2, 3]), parent=None, action=None, prior=0.3)
        node2.visit_count = 20
        node2.value_sum = 15.0
        
        node3 = Node(state=None, parent=None, action=None, prior=0.2)
        node3.visit_count = 0
        node3.value_sum = 0.0
        
        id1 = arena.add_node(node1)
        id2 = arena.add_node(node2)
        id3 = arena.add_node(node3)
        
        # Save
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            arena.save(f.name)
            temp_path = f.name
            
        # Create new arena with different config
        new_config = MemoryConfig(gpu_memory_limit=512, cpu_memory_limit=1024)
        new_arena = TreeArena(new_config)
        
        # Load should override config
        new_arena.load(temp_path)
        
        # Verify nodes
        loaded1 = new_arena.get_node(id1)
        assert loaded1.state == "string_state"
        assert loaded1.visit_count == 10
        
        loaded2 = new_arena.get_node(id2)
        assert np.array_equal(loaded2.state, np.array([1, 2, 3]))
        assert loaded2.visit_count == 20
        
        loaded3 = new_arena.get_node(id3)
        assert loaded3.state is None
        assert loaded3.visit_count == 0
        
        # Clean up
        os.unlink(temp_path)
        
    def test_get_nodes_batch_with_missing(self):
        """Test batch retrieval with missing nodes"""
        config = MemoryConfig(gpu_memory_limit=1024, cpu_memory_limit=2048)
        arena = TreeArena(config)
        
        # Add some nodes
        node_ids = []
        for i in range(5):
            node = Node(state=f"state_{i}", parent=None, action=None, prior=0.1)
            node_id = arena.add_node(node)
            node_ids.append(node_id)
            
        # Request batch with some missing nodes
        requested_ids = node_ids[:3] + ["missing_1", "missing_2"]
        nodes = arena.get_nodes_batch(requested_ids)
        
        # Should return only existing nodes
        assert len(nodes) == 3
        for i, node in enumerate(nodes):
            assert node.state == f"state_{i}"
            
    def test_check_memory_pressure_no_gc(self):
        """Test memory pressure check that doesn't trigger GC"""
        config = MemoryConfig(
            gpu_memory_limit=1000,
            cpu_memory_limit=1000,
            gc_threshold=0.9,  # 90% threshold
            node_size_bytes=100
        )
        arena = TreeArena(config)
        
        # Add nodes up to 80% capacity (below threshold)
        for i in range(16):  # 1600 bytes out of 2000
            node = Node(state=f"state_{i}", parent=None, action=None, prior=0.1)
            arena.add_node(node)
            
        initial_nodes = arena.total_nodes
        
        # Manually check memory pressure
        arena._check_memory_pressure()
        
        # Should not trigger GC
        assert arena.total_nodes == initial_nodes
        assert len(arena.gc_history) == 0
        
    def test_create_new_page_cpu_only(self):
        """Test page creation when GPU is full"""
        config = MemoryConfig(
            gpu_memory_limit=0,  # No GPU memory
            cpu_memory_limit=1000,
            page_size=5,
            node_size_bytes=100
        )
        arena = TreeArena(config, use_gpu=True)
        
        # Create page - should go to CPU
        page = arena._create_new_page()
        
        assert not page.is_on_gpu
        assert page.page_id in arena.cpu_pages
        assert page.page_id not in arena.gpu_pages
        assert arena.cpu_memory_used == 500  # 5 * 100
        
    def test_lru_ordering(self):
        """Test LRU ordering of GPU pages"""
        config = MemoryConfig(
            gpu_memory_limit=1000,
            cpu_memory_limit=2000,
            page_size=2,
            node_size_bytes=100
        )
        arena = TreeArena(config)
        
        # Add multiple pages
        page_ids = []
        for i in range(3):
            node1 = Node(state=f"state_{i}_1", parent=None, action=None, prior=0.1)
            node2 = Node(state=f"state_{i}_2", parent=None, action=None, prior=0.1)
            id1 = arena.add_node(node1)
            id2 = arena.add_node(node2)
            
            # Get page id from first node
            page_id = arena.node_registry[id1][0]
            page_ids.append(page_id)
            
        # Access middle page
        middle_page_nodes = [n for n, (pid, _) in arena.node_registry.items() 
                           if pid == page_ids[1]]
        arena.get_node(middle_page_nodes[0])
        
        # Check LRU order - middle page should be at end (most recent)
        gpu_page_list = list(arena.gpu_pages.keys())
        assert page_ids[1] == gpu_page_list[-1]
        
    def test_storage_info_without_mixed_precision(self):
        """Test storage info when mixed precision is disabled"""
        config = MemoryConfig(
            gpu_memory_limit=1024,
            cpu_memory_limit=2048,
            enable_mixed_precision=False
        )
        arena = TreeArena(config)
        
        node = Node(state="test", parent=None, action=None, prior=0.5)
        node.visit_count = 2000  # High visit count
        node_id = arena.add_node(node)
        
        info = arena.get_storage_info(node_id)
        assert info['precision'] == 'fp32'  # Should be fp32 even with high visits
        
    def test_storage_info_nonexistent_node(self):
        """Test storage info for nonexistent node"""
        config = MemoryConfig(gpu_memory_limit=1024, cpu_memory_limit=2048)
        arena = TreeArena(config)
        
        info = arena.get_storage_info("nonexistent")
        assert info == {}
        
    def test_concurrent_access_simulation(self):
        """Simulate concurrent access patterns"""
        config = MemoryConfig(
            gpu_memory_limit=5000,  # Increased to hold all nodes
            cpu_memory_limit=5000,  # Increased to hold all nodes
            page_size=2,
            node_size_bytes=100,
            gc_threshold=0.99  # Very high to prevent GC during test
        )
        arena = TreeArena(config)
        
        # Add many nodes
        node_ids = []
        for i in range(20):
            node = Node(state=f"state_{i}", parent=None, action=None, prior=0.1)
            node_id = arena.add_node(node)
            node_ids.append(node_id)
            
        # Simulate random access pattern
        import random
        random.seed(42)
        
        for _ in range(50):
            # Random single access
            if random.random() < 0.7:
                node_id = random.choice(node_ids)
                node = arena.get_node(node_id)
                assert node is not None
            else:
                # Random batch access
                batch_size = random.randint(1, 5)
                batch_ids = random.sample(node_ids, min(batch_size, len(node_ids)))
                nodes = arena.get_nodes_batch(batch_ids)
                assert len(nodes) == len(batch_ids)
                
        # Verify consistency
        assert arena.total_nodes == 20
        assert arena.gpu_nodes + arena.cpu_nodes == 20
        assert len(arena.node_registry) == 20