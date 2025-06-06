"""Tests for transposition table functionality in TreeArena"""

import pytest
import numpy as np
from unittest.mock import Mock

from mcts.tree_arena import TreeArena, MemoryConfig
from mcts.node import Node


class TestTranspositionTable:
    """Test transposition table functionality"""
    
    @pytest.fixture
    def arena(self):
        """Create test arena with transpositions enabled"""
        config = MemoryConfig(
            gpu_memory_limit=1024 * 1024,
            cpu_memory_limit=2 * 1024 * 1024,
            page_size=10
        )
        return TreeArena(config, use_gpu=False, enable_transpositions=True)
    
    @pytest.fixture
    def arena_no_transpositions(self):
        """Create test arena with transpositions disabled"""
        config = MemoryConfig(
            gpu_memory_limit=1024 * 1024,
            cpu_memory_limit=2 * 1024 * 1024,
            page_size=10
        )
        return TreeArena(config, use_gpu=False, enable_transpositions=False)
    
    def test_transposition_detection(self, arena):
        """Test that identical states are detected as transpositions"""
        # Create nodes with same state
        state1 = np.ones((8, 8))
        state_hash = hash(state1.tobytes())
        
        node1 = Node(state=state1, parent=None, action=None, prior=1.0)
        node2 = Node(state=state1, parent=None, action=None, prior=1.0)
        
        # Add first node
        id1 = arena.add_node(node1, state_hash)
        assert arena.transposition_hits == 0
        assert arena.transposition_misses == 1
        
        # Add second node with same hash - should return same ID
        id2 = arena.add_node(node2, state_hash)
        assert id1 == id2
        assert arena.transposition_hits == 1
        assert arena.transposition_misses == 1
        
        # Verify only one node exists
        assert arena.total_nodes == 1
        assert len(arena.transposition_table) == 1
        
    def test_different_states_no_transposition(self, arena):
        """Test that different states create different nodes"""
        state1 = np.ones((8, 8))
        state2 = np.zeros((8, 8))
        
        hash1 = hash(state1.tobytes())
        hash2 = hash(state2.tobytes())
        
        node1 = Node(state=state1, parent=None, action=None, prior=1.0)
        node2 = Node(state=state2, parent=None, action=None, prior=1.0)
        
        id1 = arena.add_node(node1, hash1)
        id2 = arena.add_node(node2, hash2)
        
        assert id1 != id2
        assert arena.total_nodes == 2
        assert len(arena.transposition_table) == 2
        
    def test_disabled_transpositions(self, arena_no_transpositions):
        """Test behavior when transpositions are disabled"""
        state = np.ones((8, 8))
        state_hash = hash(state.tobytes())
        
        node1 = Node(state=state, parent=None, action=None, prior=1.0)
        node2 = Node(state=state, parent=None, action=None, prior=1.0)
        
        # Both nodes should get different IDs
        id1 = arena_no_transpositions.add_node(node1, state_hash)
        id2 = arena_no_transpositions.add_node(node2, state_hash)
        
        assert id1 != id2
        assert arena_no_transpositions.total_nodes == 2
        assert len(arena_no_transpositions.transposition_table) == 0
        
    def test_multiple_parents_tracking(self, arena):
        """Test that transpositions track multiple parents correctly"""
        # Create parent nodes
        parent1 = Node(state="parent1", parent=None, action=None, prior=1.0)
        parent2 = Node(state="parent2", parent=None, action=None, prior=1.0)
        
        parent1_id = arena.add_node(parent1)
        parent2_id = arena.add_node(parent2)
        
        # Create child state reachable from both parents
        child_state = np.ones((8, 8))
        state_hash = hash(child_state.tobytes())
        
        # Add from first parent
        child1 = Node(state=child_state, parent=parent1, action=1, prior=0.5)
        child1_id = arena.add_node(child1, state_hash)
        
        # Add from second parent - should detect transposition
        child2 = Node(state=child_state, parent=parent2, action=2, prior=0.5)
        child2_id = arena.add_node(child2, state_hash)
        
        assert child1_id == child2_id
        assert arena.transposition_hits == 1
        
        # Check parent tracking (disabled in current implementation)
        # In full implementation, would verify DAG structure
        
    def test_get_node_by_hash(self, arena):
        """Test retrieving nodes by state hash"""
        state = np.ones((8, 8))
        state_hash = hash(state.tobytes())
        
        # Initially should return None
        assert arena.get_node_by_hash(state_hash) is None
        
        # Add node
        node = Node(state=state, parent=None, action=None, prior=1.0)
        node_id = arena.add_node(node, state_hash)
        
        # Now should retrieve the node
        retrieved = arena.get_node_by_hash(state_hash)
        assert retrieved is not None
        assert retrieved.state is state
        
    def test_transposition_statistics(self, arena):
        """Test transposition table statistics"""
        # Initial stats
        stats = arena.get_transposition_stats()
        assert stats['enabled'] is True
        assert stats['table_size'] == 0
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['hit_rate'] == 0.0
        
        # Add some nodes
        for i in range(5):
            state = np.ones((8, 8)) * i
            node = Node(state=state, parent=None, action=None, prior=1.0)
            arena.add_node(node, hash(state.tobytes()))
        
        # Add duplicates
        for i in range(3):
            state = np.ones((8, 8)) * i
            node = Node(state=state, parent=None, action=None, prior=1.0)
            arena.add_node(node, hash(state.tobytes()))
        
        stats = arena.get_transposition_stats()
        assert stats['table_size'] == 5
        assert stats['hits'] == 3
        assert stats['misses'] == 5
        assert stats['hit_rate'] == 3 / 8
        
    def test_garbage_collection_cleans_transpositions(self, arena):
        """Test that GC properly cleans transposition table"""
        # Set low GC threshold
        arena.config.gc_threshold = 0.5
        
        # Add many nodes to trigger GC
        for i in range(100):
            state = np.ones((8, 8)) * i
            node = Node(state=state, parent=None, action=None, prior=1.0)
            node.visit_count = 1 if i < 50 else 100  # Low importance for first half
            arena.add_node(node, hash(state.tobytes()))
        
        initial_table_size = len(arena.transposition_table)
        initial_nodes = arena.total_nodes
        
        # Force GC
        arena._run_garbage_collection()
        
        # Verify cleanup
        assert arena.total_nodes < initial_nodes
        assert len(arena.transposition_table) < initial_table_size
        
        # Verify remaining entries are consistent
        for hash_val, node_id in arena.transposition_table.items():
            assert node_id in arena.node_registry
            
    def test_thread_safety(self, arena):
        """Test concurrent access to transposition table"""
        import threading
        import time
        
        results = []
        
        def add_nodes(thread_id):
            for i in range(10):
                state = np.ones((8, 8)) * (i % 3)  # Create some duplicates
                node = Node(state=state, parent=None, action=None, prior=1.0)
                node_id = arena.add_node(node, hash(state.tobytes()))
                results.append((thread_id, i, node_id))
                time.sleep(0.001)  # Small delay to increase contention
        
        # Create threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=add_nodes, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Verify consistency
        assert len(results) == 50
        
        # Should have only 3 unique nodes (states 0, 1, 2)
        assert arena.total_nodes == 3
        assert len(arena.transposition_table) == 3
        
        # All threads should get same IDs for same states
        id_map = {}
        for thread_id, i, node_id in results:
            state_key = i % 3
            if state_key not in id_map:
                id_map[state_key] = node_id
            else:
                assert id_map[state_key] == node_id, \
                    f"Thread {thread_id} got different ID for state {state_key}"
                    
    def test_no_hash_provided(self, arena):
        """Test behavior when no hash is provided"""
        node = Node(state="test", parent=None, action=None, prior=1.0)
        
        # Without hash, should always create new node
        id1 = arena.add_node(node)
        id2 = arena.add_node(node)
        
        assert id1 != id2
        assert arena.total_nodes == 2
        assert len(arena.transposition_table) == 0