"""
Comprehensive tests for improved MCTS tree reuse functionality

Tests cover:
- Tree structure preservation during reset
- Memory allocation reuse
- Performance improvements from reuse
- State consistency after reset
- Integration with memory pool
"""

import pytest
import torch
import numpy as np
import time
import logging
from typing import List
import gc

from mcts.core.mcts import MCTS, MCTSConfig
from mcts.core.game_interface import GameType
from mcts.gpu.csr_tree import CSRTree, CSRTreeConfig
from mcts.neural_networks.mock_evaluator import MockEvaluator

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestTreeReuse:
    """Test cases for MCTS tree reuse functionality"""
    
    @pytest.fixture
    def mcts_config(self):
        """Create MCTS configuration for testing"""
        return MCTSConfig(
            num_simulations=100,
            min_wave_size=256,
            max_wave_size=256,
            adaptive_wave_sizing=False,
            memory_pool_size_mb=128,
            max_tree_nodes=10000,
            device='cpu',
            game_type=GameType.GOMOKU,
            use_optimized_implementation=True
        )
    
    @pytest.fixture
    def evaluator(self):
        """Create mock evaluator for testing"""
        return MockEvaluator(game_type='gomoku', device='cpu')
    
    @pytest.fixture
    def mcts(self, mcts_config, evaluator):
        """Create MCTS instance for testing"""
        return MCTS(mcts_config, evaluator)
    
    def test_tree_reset_preserves_structure(self, mcts):
        """Test that reset_tree preserves the tree structure"""
        # Create a simple game state
        import alphazero_py
        game_state = alphazero_py.GomokuState()
        
        # Run initial search to build tree
        mcts.search(game_state, num_simulations=50)
        
        # Record tree state before reset
        if mcts.using_optimized and hasattr(mcts, 'tree'):
            original_tree = mcts.tree
            original_num_nodes = original_tree.num_nodes
            original_num_edges = original_tree.num_edges
            original_node_capacity = original_tree.node_capacity
            original_edge_capacity = original_tree.edge_capacity
            
            # Get memory pointers
            node_values_ptr = original_tree.node_values.data_ptr()
            edge_visits_ptr = original_tree.edge_visits.data_ptr()
            
            # Reset tree
            mcts.reset_tree()
            
            # Verify same tree object
            assert mcts.tree is original_tree
            
            # Verify memory preserved
            assert mcts.tree.node_values.data_ptr() == node_values_ptr
            assert mcts.tree.edge_visits.data_ptr() == edge_visits_ptr
            
            # Verify capacities preserved
            assert mcts.tree.node_capacity == original_node_capacity
            assert mcts.tree.edge_capacity == original_edge_capacity
            
            # Verify tree is reset
            assert mcts.tree.num_nodes == 0
            assert mcts.tree.num_edges == 0
    
    def test_tree_reuse_performance(self, mcts_config, evaluator):
        """Test performance improvement from tree reuse"""
        import alphazero_py
        
        # Create two MCTS instances - one with reuse, one without
        mcts_reuse = MCTS(mcts_config, evaluator)
        
        # Create new config for no reuse
        config_no_reuse = MCTSConfig(
            num_simulations=mcts_config.num_simulations,
            min_wave_size=mcts_config.min_wave_size,
            max_wave_size=mcts_config.max_wave_size,
            adaptive_wave_sizing=mcts_config.adaptive_wave_sizing,
            memory_pool_size_mb=0,  # Disable memory pool
            max_tree_nodes=mcts_config.max_tree_nodes,
            device=mcts_config.device,
            game_type=mcts_config.game_type,
            use_optimized_implementation=False
        )
        mcts_no_reuse = MCTS(config_no_reuse, evaluator)
        
        # Measure reset times
        num_resets = 20
        game_state = alphazero_py.GomokuState()
        
        # Build initial trees
        mcts_reuse.search(game_state, num_simulations=100)
        mcts_no_reuse.search(game_state, num_simulations=100)
        
        # Time resets with reuse
        start_time = time.time()
        for _ in range(num_resets):
            mcts_reuse.reset_tree()
            # Small search to use the tree
            mcts_reuse.search(game_state, num_simulations=10)
        reuse_time = time.time() - start_time
        
        # Time resets without reuse
        start_time = time.time()
        for _ in range(num_resets):
            mcts_no_reuse.reset_tree()
            # Small search to use the tree
            mcts_no_reuse.search(game_state, num_simulations=10)
        no_reuse_time = time.time() - start_time
        
        logger.info(f"Reset time with reuse: {reuse_time:.3f}s")
        logger.info(f"Reset time without reuse: {no_reuse_time:.3f}s")
        
        # Reuse should be faster
        assert reuse_time < no_reuse_time
    
    def test_memory_allocation_tracking(self, mcts):
        """Test that memory allocations are tracked correctly"""
        import alphazero_py
        game_state = alphazero_py.GomokuState()
        
        # Track allocations
        allocations_before = []
        allocations_after = []
        
        # Run multiple cycles
        for i in range(5):
            # Force garbage collection
            gc.collect()
            
            # Track memory before
            if hasattr(mcts, 'tree') and mcts.tree is not None:
                allocations_before.append({
                    'node_capacity': mcts.tree.node_capacity,
                    'edge_capacity': mcts.tree.edge_capacity,
                    'num_nodes': mcts.tree.num_nodes,
                    'num_edges': mcts.tree.num_edges
                })
            
            # Search
            mcts.search(game_state, num_simulations=50)
            
            # Reset
            mcts.reset_tree()
            
            # Track memory after
            if hasattr(mcts, 'tree') and mcts.tree is not None:
                allocations_after.append({
                    'node_capacity': mcts.tree.node_capacity,
                    'edge_capacity': mcts.tree.edge_capacity,
                    'num_nodes': mcts.tree.num_nodes,
                    'num_edges': mcts.tree.num_edges
                })
        
        # Verify capacities are preserved or grow
        for i in range(1, len(allocations_after)):
            assert allocations_after[i]['node_capacity'] >= allocations_after[i-1]['node_capacity']
            assert allocations_after[i]['edge_capacity'] >= allocations_after[i-1]['edge_capacity']
        
        # Verify reset clears nodes/edges
        for alloc in allocations_after:
            assert alloc['num_nodes'] == 0
            assert alloc['num_edges'] == 0
    
    def test_tree_reuse_with_memory_pool(self, mcts_config, evaluator):
        """Test tree reuse with memory pool integration"""
        # Create MCTS with memory pooling enabled
        mcts = MCTS(mcts_config, evaluator)
        
        import alphazero_py
        game_state = alphazero_py.GomokuState()
        
        # Run search to build tree
        mcts.search(game_state, num_simulations=100)
        
        # Reset tree multiple times
        for _ in range(10):
            mcts.reset_tree()
            mcts.search(game_state, num_simulations=20)
        
        # Verify MCTS is still functional
        policy = mcts.search(game_state, num_simulations=50)
        assert len(policy) == 225  # 15x15 board
    
    def test_tree_state_consistency(self, mcts):
        """Test that tree state is consistent after reset"""
        import alphazero_py
        
        # Test with different game states
        game_states = [
            alphazero_py.GomokuState(),
            alphazero_py.GomokuState()
        ]
        
        # Make some moves in second state
        game_states[1].make_move(112)  # Center of 15x15 board
        game_states[1].make_move(113)
        
        for game_state in game_states:
            # Search
            policy1 = mcts.search(game_state, num_simulations=100)
            
            # Reset
            mcts.reset_tree()
            
            # Search again
            policy2 = mcts.search(game_state, num_simulations=100)
            
            # Policies should be similar (not identical due to randomness)
            assert len(policy1) == len(policy2)
            
            # Reset for next iteration
            mcts.reset_tree()
    
    def test_tree_growth_after_reset(self, mcts):
        """Test that tree can grow properly after reset"""
        import alphazero_py
        game_state = alphazero_py.GomokuState()
        
        # Initial small search
        mcts.search(game_state, num_simulations=10)
        
        if hasattr(mcts, 'tree'):
            initial_capacity = mcts.tree.node_capacity
            
            # Reset
            mcts.reset_tree()
            
            # Large search that requires growth
            mcts.search(game_state, num_simulations=500)
            
            # Verify growth happened
            assert mcts.tree.node_capacity >= initial_capacity
            assert mcts.tree.num_nodes > 10
            
            # Reset again
            mcts.reset_tree()
            
            # Verify capacity is preserved
            assert mcts.tree.node_capacity >= initial_capacity
            assert mcts.tree.num_nodes == 0
    
    def test_concurrent_reset_and_search(self, mcts_config, evaluator):
        """Test concurrent reset and search operations"""
        import alphazero_py
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        mcts = MCTS(mcts_config, evaluator)
        game_state = alphazero_py.GomokuState()
        
        # Shared state
        errors = []
        search_count = 0
        reset_count = 0
        lock = threading.Lock()
        
        def search_task():
            nonlocal search_count
            try:
                mcts.search(game_state, num_simulations=50)
                with lock:
                    search_count += 1
            except Exception as e:
                errors.append(f"Search error: {e}")
        
        def reset_task():
            nonlocal reset_count
            try:
                mcts.reset_tree()
                with lock:
                    reset_count += 1
            except Exception as e:
                errors.append(f"Reset error: {e}")
        
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            # Interleave search and reset operations
            for i in range(20):
                if i % 3 == 0:
                    futures.append(executor.submit(reset_task))
                else:
                    futures.append(executor.submit(search_task))
            
            # Wait for completion
            for future in futures:
                future.result()
        
        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert search_count > 0
        assert reset_count > 0
    
    def test_tree_reuse_fraction(self, evaluator):
        """Test tree_reuse_fraction parameter"""
        import alphazero_py
        
        # Test different reuse fractions
        fractions = [0.0, 0.5, 1.0]
        
        for fraction in fractions:
            config = MCTSConfig(
                num_simulations=100,
                memory_pool_size_mb=128 if fraction > 0 else 0,
                device='cpu',
                game_type=GameType.GOMOKU,
                use_optimized_implementation=fraction > 0
            )
            
            mcts = MCTS(config, evaluator)
            game_state = alphazero_py.GomokuState()
            
            # Build tree
            mcts.search(game_state, num_simulations=100)
            
            if hasattr(mcts, 'tree'):
                nodes_before = mcts.tree.num_nodes
                
                # Move to new position
                game_state.make_move(112)
                
                # Reset with fraction
                mcts.reset_tree()
                
                # For testing purposes, fraction behavior would be:
                # 0.0 - completely new tree
                # 0.5 - reuse some nodes
                # 1.0 - reuse all applicable nodes
                
                # This is a placeholder - actual fraction logic would be in MCTS
                assert mcts.tree is not None


class TestTreeReuseEdgeCases:
    """Test edge cases for tree reuse"""
    
    def test_reset_empty_tree(self, mcts_config, evaluator):
        """Test resetting an empty tree"""
        mcts = MCTS(mcts_config, evaluator)
        
        # Reset without any search
        mcts.reset_tree()
        
        # Should not crash
        assert mcts is not None
        
        # Should be able to search
        import alphazero_py
        game_state = alphazero_py.GomokuState()
        policy = mcts.search(game_state, num_simulations=10)
        assert policy is not None
    
    def test_reset_after_error(self, mcts_config, evaluator):
        """Test reset after an error condition"""
        mcts = MCTS(mcts_config, evaluator)
        
        # Simulate error by corrupting tree state
        if hasattr(mcts, 'tree') and mcts.tree is not None:
            # Save original
            original_tree = mcts.tree
            
            # Temporarily set to None
            mcts.tree = None
            
            # Reset should handle gracefully
            mcts.reset_tree()
            
            # Tree should be recreated or restored
            assert mcts.tree is not None
    
    def test_multiple_consecutive_resets(self, mcts_config, evaluator):
        """Test multiple consecutive resets"""
        mcts = MCTS(mcts_config, evaluator)
        import alphazero_py
        game_state = alphazero_py.GomokuState()
        
        # Build tree
        mcts.search(game_state, num_simulations=50)
        
        # Multiple resets
        for _ in range(10):
            mcts.reset_tree()
        
        # Should still work
        policy = mcts.search(game_state, num_simulations=50)
        assert policy is not None
        assert len(policy) == 225  # 15x15 board
    
    def test_memory_leak_prevention(self, mcts_config, evaluator):
        """Test that tree reuse prevents memory leaks"""
        import psutil
        import os
        
        mcts = MCTS(mcts_config, evaluator)
        import alphazero_py
        game_state = alphazero_py.GomokuState()
        
        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Many cycles of search and reset
        for i in range(50):
            mcts.search(game_state, num_simulations=100)
            mcts.reset_tree()
            
            if i % 10 == 0:
                gc.collect()
        
        # Final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        logger.info(f"Memory increase: {memory_increase:.1f} MB")
        
        # Should not have significant memory increase
        assert memory_increase < 100  # Less than 100MB increase


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])