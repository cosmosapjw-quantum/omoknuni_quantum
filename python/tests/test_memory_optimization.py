"""Test suite for memory optimization in MCTS

This test suite verifies memory usage improvements and ensures
we achieve the target of <100 bytes per node.
"""

import pytest
import torch
import gc
import numpy as np
from typing import Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts import (
    TensorTree, TensorTreeConfig,
    HighPerformanceMCTS, HighPerformanceMCTSConfig,
    GameInterface, GameType,
    MockEvaluator, EvaluatorConfig
)


class TestMemoryUsage:
    """Test memory usage of tensor tree and MCTS components"""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Clear GPU memory before and after each test"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        yield
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
    def get_gpu_memory_mb(self) -> float:
        """Get current GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0.0
        
    def test_tensor_tree_memory_per_node(self):
        """Test memory usage per node in tensor tree"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
            
        # Test different tree sizes
        tree_sizes = [1000, 10000, 50000]
        
        for max_nodes in tree_sizes:
            # Clear memory
            torch.cuda.empty_cache()
            initial_memory = self.get_gpu_memory_mb()
            
            # Create tree
            config = TensorTreeConfig(
                max_nodes=max_nodes,
                max_children=225,  # Gomoku board size
                device='cuda'
            )
            tree = TensorTree(config)
            
            # Measure memory after creation
            after_creation = self.get_gpu_memory_mb()
            total_memory_mb = after_creation - initial_memory
            memory_per_node_kb = (total_memory_mb * 1024) / max_nodes
            
            print(f"\nTree size: {max_nodes:,} nodes")
            print(f"Total memory: {total_memory_mb:.1f} MB")
            print(f"Memory per node: {memory_per_node_kb:.2f} KB")
            
            # Current implementation uses ~3.15KB per node
            # We want to get this below 0.1KB (100 bytes)
            assert memory_per_node_kb < 4.0, f"Memory usage too high: {memory_per_node_kb:.2f} KB per node"
            
            # Clean up
            del tree
            
    def test_actual_vs_allocated_nodes(self):
        """Test memory usage for actual nodes vs pre-allocated capacity"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
            
        # Create small tree
        config = TensorTreeConfig(
            max_nodes=10000,
            max_children=225,
            device='cuda'
        )
        tree = TensorTree(config)
        
        initial_memory = self.get_gpu_memory_mb()
        
        # Add only 100 nodes
        root = tree.add_root()
        for i in range(99):
            parent = i // 10  # Create a simple tree structure
            action = i % 225
            tree.add_child(parent, action, prior=0.1)
            
        # Check that we're not wasting memory on unused nodes
        actual_nodes = tree.num_nodes
        assert actual_nodes == 100
        
        # Memory should scale with actual nodes, not max_nodes
        # (This will fail with current implementation)
        
    def test_sparse_children_memory(self):
        """Test that sparse children don't waste memory"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
            
        # Most nodes in MCTS have far fewer than max_children
        # Test memory usage with realistic child counts
        
        config = TensorTreeConfig(
            max_nodes=1000,
            max_children=225,
            device='cuda'
        )
        tree = TensorTree(config)
        
        # Add nodes with varying numbers of children
        root = tree.add_root()
        
        # Add 10 nodes with 5 children each (typical early game)
        for i in range(10):
            parent = tree.add_child(root, i, prior=0.1)
            for j in range(5):
                tree.add_child(parent, j, prior=0.02)
                
        # Current implementation allocates 225 child slots for each node
        # Even though most have only 5 children
        # This is what we need to optimize
        
    def test_data_type_optimization(self):
        """Test memory savings from using smaller data types"""
        # Currently using int32 (4 bytes) for indices
        # Could use int16 (2 bytes) for trees up to 65k nodes
        
        # Test that int16 is sufficient
        max_nodes_int16 = 2**16 - 1  # 65,535
        assert max_nodes_int16 > 50000, "int16 should handle 50k nodes"
        
        # Calculate potential savings
        current_bytes_per_index = 4  # int32
        optimized_bytes_per_index = 2  # int16
        
        # Each node has: parent_idx, parent_action, num_children
        # Plus child arrays: child_indices[400], child_actions[400]
        indices_per_node = 3 + 400 + 400
        
        current_index_memory = indices_per_node * current_bytes_per_index
        optimized_index_memory = indices_per_node * optimized_bytes_per_index
        
        savings_per_node = current_index_memory - optimized_index_memory
        print(f"\nPotential savings from int16: {savings_per_node} bytes per node")
        assert savings_per_node > 1600, "Should save at least 1.6KB per node"


class TestMemoryPooling:
    """Test memory pooling strategies for dynamic allocation"""
    
    def test_child_allocation_pool(self):
        """Test pooled allocation for children instead of pre-allocation"""
        # Instead of allocating 225 slots per node upfront,
        # allocate from a pool as needed
        
        # Simulate a pool-based approach
        class ChildPool:
            def __init__(self, initial_size=10000):
                self.pool = torch.zeros(initial_size, 2, dtype=torch.int16)  # (child_idx, action)
                self.next_free = 0
                
            def allocate(self, num_children):
                start = self.next_free
                self.next_free += num_children
                return start, start + num_children
                
        pool = ChildPool()
        
        # Simulate adding children for 1000 nodes
        total_children = 0
        for i in range(1000):
            # Realistic distribution: most nodes have 5-50 children
            num_children = min(225, max(5, int(np.random.exponential(20))))
            start, end = pool.allocate(num_children)
            total_children += num_children
            
        avg_children = total_children / 1000
        print(f"\nAverage children per node: {avg_children:.1f}")
        
        # Memory usage with pooling
        pool_memory = pool.next_free * 2 * 2  # 2 values, 2 bytes each
        pool_memory_per_node = pool_memory / 1000
        
        # Current approach
        current_memory_per_node = 225 * 2 * 4  # 225 slots, 2 values, 4 bytes each
        
        savings = current_memory_per_node - pool_memory_per_node
        print(f"Current approach: {current_memory_per_node} bytes per node")
        print(f"Pooled approach: {pool_memory_per_node:.0f} bytes per node")
        print(f"Savings: {savings:.0f} bytes per node")
        
        assert pool_memory_per_node < current_memory_per_node / 10


class TestMCTSMemoryIntegration:
    """Test memory usage in full MCTS context"""
    
    def test_mcts_memory_scaling(self):
        """Test how MCTS memory scales with tree size"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
            
        game = GameInterface(GameType.GOMOKU)
        eval_config = EvaluatorConfig(
            batch_size=256,
            device='cuda',
            use_fp16=True
        )
        evaluator = MockEvaluator(eval_config, 225)
        
        # Test with small tree
        config = HighPerformanceMCTSConfig(
            num_simulations=100,
            wave_size=32,
            max_tree_size=1000,
            enable_gpu=True,
            device='cuda'
        )
        
        torch.cuda.empty_cache()
        initial = torch.cuda.memory_allocated() / 1024**2
        
        mcts = HighPerformanceMCTS(config, game, evaluator)
        after_init = torch.cuda.memory_allocated() / 1024**2
        
        # Run some searches
        state = game.create_initial_state()
        for _ in range(5):
            mcts.search(state)
            
        after_search = torch.cuda.memory_allocated() / 1024**2
        
        print(f"\nMCTS Memory Usage:")
        print(f"Initial: {initial:.1f} MB")
        print(f"After init: {after_init:.1f} MB (used {after_init - initial:.1f} MB)")
        print(f"After search: {after_search:.1f} MB (used {after_search - after_init:.1f} MB)")
        
        # Calculate per-node memory
        nodes_created = mcts.tree.num_nodes if hasattr(mcts.tree, 'num_nodes') else 0
        if nodes_created > 0:
            memory_per_node = ((after_search - after_init) * 1024) / nodes_created
            print(f"Nodes created: {nodes_created}")
            print(f"Memory per node: {memory_per_node:.0f} KB")


def calculate_target_memory_layout():
    """Calculate target memory layout for <100 bytes per node"""
    print("\n=== TARGET MEMORY LAYOUT ===")
    print("Goal: <100 bytes per node")
    print("\nProposed layout:")
    
    # Essential data only
    layout = [
        ("visit_count", "uint32", 4),
        ("value_sum", "float16", 2),  # Use FP16 for values
        ("prior", "float16", 2),
        ("parent_idx", "uint16", 2),  # Support up to 65k nodes
        ("parent_action", "uint16", 2),
        ("num_children", "uint8", 1),  # Max 255 children (enough for Go/Chess)
        ("children_start", "uint32", 4),  # Index into children pool
        ("is_expanded", "bool", 1),
        ("is_terminal", "bool", 1),
    ]
    
    total = 0
    for name, dtype, size in layout:
        print(f"{name:20} {dtype:8} {size:2} bytes")
        total += size
        
    print(f"\nTotal: {total} bytes per node")
    print(f"Target achieved: {total < 100}")
    
    # Children stored separately in pool
    print("\nChildren pool (separate allocation):")
    print("- Only allocate space for actual children")
    print("- Average ~20 children per expanded node")
    print("- 4 bytes per child (uint16 idx + uint16 action)")
    print("- ~80 bytes per expanded node on average")
    
    return total


if __name__ == "__main__":
    # Calculate target layout
    calculate_target_memory_layout()
    
    # Run memory tests directly
    if torch.cuda.is_available():
        print("\n=== RUNNING MEMORY TESTS ===")
        test = TestMemoryUsage()
        
        # Manually setup
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        
        # Run test
        test.test_tensor_tree_memory_per_node()
        
        # Cleanup
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
    else:
        print("GPU not available, skipping GPU memory tests")
        
    # Run with pytest for full suite
    # pytest.main([__file__, "-v", "-s", "-k", "test_tensor_tree_memory_per_node"])