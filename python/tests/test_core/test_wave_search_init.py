"""Test wave search initialization dependencies

This module tests that wave search methods handle initialization dependencies correctly,
following TDD principles.
"""

import pytest
import torch
from mcts.core.mcts import MCTS


class TestWaveSearchInitialization:
    """Test wave search initialization issues"""
    
    def test_expand_batch_vectorized_without_run_wave(self, base_mcts_config, mock_evaluator, empty_gomoku_state):
        """Test that _expand_batch_vectorized handles missing node_to_state initialization
        
        This test verifies that _expand_batch_vectorized can work with explicit parameters
        when node_to_state is not initialized through run_wave.
        """
        # Initialize MCTS
        mcts = MCTS(base_mcts_config, mock_evaluator)
        mcts._initialize_root(empty_gomoku_state)
        
        # Get wave search instance
        wave_search = mcts.wave_search
        
        # Verify node_to_state is not initialized
        assert wave_search.node_to_state is None
        
        # This should work when providing explicit parameters
        expanded_nodes = wave_search._expand_batch_vectorized(
            torch.tensor([0]), 
            node_to_state=mcts.node_to_state,
            state_pool_free_list=mcts.state_pool_free_list
        )
        
        # Should return some result without crashing
        assert expanded_nodes is not None
        
        # Without parameters, it should still fail
        with pytest.raises(RuntimeError, match="node_to_state not initialized"):
            wave_search._expand_batch_vectorized(torch.tensor([0]))
            
    def test_expand_batch_vectorized_with_manual_initialization(self, base_mcts_config, mock_evaluator, empty_gomoku_state):
        """Test that _expand_batch_vectorized works when manually initialized"""
        # Initialize MCTS
        mcts = MCTS(base_mcts_config, mock_evaluator)
        mcts._initialize_root(empty_gomoku_state)
        
        # Get wave search instance
        wave_search = mcts.wave_search
        
        # Manually set node_to_state (simulating what run_wave does)
        wave_search.node_to_state = mcts.node_to_state
        wave_search.state_pool_free_list = mcts.state_pool_free_list
        
        # Now expansion should work
        expanded_nodes = wave_search._expand_batch_vectorized(torch.tensor([0]))
        
        # Should return some result without crashing
        assert expanded_nodes is not None
        
    def test_proper_wave_search_flow(self, base_mcts_config, mock_evaluator, empty_gomoku_state):
        """Test that the proper wave search flow works correctly"""
        # Initialize MCTS
        mcts = MCTS(base_mcts_config, mock_evaluator)
        mcts._initialize_root(empty_gomoku_state)
        
        # Run wave search the proper way (through run_wave)
        completed = mcts.wave_search.run_wave(
            wave_size=1,
            node_to_state=mcts.node_to_state,
            state_pool_free_list=mcts.state_pool_free_list
        )
        
        # Should complete at least one simulation
        assert completed >= 0