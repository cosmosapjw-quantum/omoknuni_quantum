"""Test state pool management fixes"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from mcts.core.mcts import MCTS, MCTSConfig
from mcts.core.game_interface import GameInterface


class MockGameInterface(GameInterface):
    """Mock game interface for testing"""
    
    def __init__(self, board_size=15):
        self.board_size = board_size
        
    def get_legal_moves(self, state):
        """Return a reasonable number of legal moves"""
        # Simulate Gomoku with ~50 legal moves early in game
        num_moves = min(50, self.board_size * self.board_size - 20)
        return list(range(num_moves))
    
    def apply_move(self, state, action):
        """Apply move and return new state"""
        new_state = np.copy(state) if isinstance(state, np.ndarray) else state.copy()
        return new_state
    
    def is_terminal(self, state):
        return False
    
    def get_winner(self, state):
        return None


class TestStatePoolManagement:
    """Test state pool management and exhaustion fixes"""
    
    def setup_method(self):
        """Setup test environment"""
        self.game_interface = MockGameInterface()
        
        # Create config with small state pool for testing
        self.config = MCTSConfig(
            num_simulations=100,
            max_tree_nodes=10000,  # Small pool to trigger exhaustion
            wave_size=1024,
            enable_state_pool_debug=True,
            device='cpu'  # Use CPU for testing
        )
        
    def test_chunked_allocation_prevents_exhaustion(self):
        """Test that chunked allocation prevents state pool exhaustion"""
        with patch('mcts.core.mcts.get_unified_kernels') as mock_kernels:
            mock_kernels.return_value = None
            
            mcts = MCTS(self.config, self.game_interface)
            
            # Simulate a scenario that would cause exhaustion
            # Create a large list of children that would exceed pool capacity
            large_child_list = list(range(5000))  # More than pool capacity
            parent_nodes = [0] * 5000
            parent_actions = list(range(5000))
            
            # This should not raise an exception due to chunked allocation
            try:
                mcts._assign_states_to_children(large_child_list, parent_nodes, parent_actions)
                # If we get here, chunking worked
                assert True
            except RuntimeError as e:
                if "State pool exhausted" in str(e):
                    pytest.fail("Chunked allocation failed to prevent state pool exhaustion")
                else:
                    # Some other error, re-raise
                    raise
    
    def test_emergency_cleanup_functionality(self):
        """Test that emergency state cleanup works correctly"""
        with patch('mcts.core.mcts.get_unified_kernels') as mock_kernels:
            mock_kernels.return_value = None
            
            mcts = MCTS(self.config, self.game_interface)
            
            # Manually mark some states as used but not properly referenced
            mcts.state_pool_free[:100] = False
            initial_free_count = mcts.state_pool_free.sum().item()
            
            # Run emergency cleanup
            freed_count = mcts._emergency_state_cleanup()
            
            # Check that states were freed
            final_free_count = mcts.state_pool_free.sum().item()
            assert final_free_count > initial_free_count
            assert freed_count > 0
    
    def test_improved_error_messages(self):
        """Test that improved error messages provide useful information"""
        with patch('mcts.core.mcts.get_unified_kernels') as mock_kernels:
            mock_kernels.return_value = None
            
            mcts = MCTS(self.config, self.game_interface)
            
            # Fill up the state pool completely
            mcts.state_pool_free.fill_(False)
            
            # Try to allocate more states
            with pytest.raises(RuntimeError) as exc_info:
                mcts._allocate_states(10)
            
            error_msg = str(exc_info.value)
            # Check that error message contains useful information
            assert "need 10" in error_msg
            assert "have 0 free" in error_msg
            assert "out of" in error_msg
            assert "used)" in error_msg
    
    def test_state_pool_debug_logging(self):
        """Test that state pool debug logging works correctly"""
        with patch('mcts.core.mcts.get_unified_kernels') as mock_kernels:
            mock_kernels.return_value = None
            
            # Enable debug logging
            config = MCTSConfig(
                max_tree_nodes=1000,
                enable_state_pool_debug=True,
                device='cpu'
            )
            
            with patch('mcts.core.mcts.logger') as mock_logger:
                mcts = MCTS(config, self.game_interface)
                
                # Allocate some states to trigger logging
                mcts._allocate_states(150)  # Above the 100 threshold
                
                # Check that debug logging was called
                assert mock_logger.debug.called
                debug_calls = [call for call in mock_logger.debug.call_args_list 
                              if '[STATE_POOL]' in str(call)]
                assert len(debug_calls) > 0
    
    def test_state_pool_sizing_calculation(self):
        """Test that state pool is properly sized for expected workload"""
        # Test with configuration similar to production
        config = MCTSConfig(
            wave_size=2048,
            max_tree_nodes=1000000,
            max_children_per_node=50,
            device='cpu'
        )
        
        with patch('mcts.core.mcts.get_unified_kernels') as mock_kernels:
            mock_kernels.return_value = None
            
            mcts = MCTS(config, self.game_interface)
            
            # Check that we have enough capacity for peak allocation
            # Worst case: wave_size nodes each with max_children_per_node children
            peak_allocation = config.wave_size * config.max_children_per_node
            available_states = mcts.state_pool_free.sum().item()
            
            # Should have at least 2x the peak allocation for safety
            assert available_states >= peak_allocation * 2
    
    def test_chunked_allocation_performance(self):
        """Test that chunked allocation doesn't significantly impact performance"""
        with patch('mcts.core.mcts.get_unified_kernels') as mock_kernels:
            mock_kernels.return_value = None
            
            mcts = MCTS(self.config, self.game_interface)
            
            import time
            
            # Test allocation of moderate number of states
            num_children = 1000
            child_indices = list(range(num_children))
            parent_nodes = [0] * num_children
            parent_actions = list(range(num_children))
            
            start_time = time.time()
            mcts._assign_states_to_children(child_indices, parent_nodes, parent_actions)
            end_time = time.time()
            
            # Should complete in reasonable time (less than 1 second)
            allocation_time = end_time - start_time
            assert allocation_time < 1.0, f"Chunked allocation took too long: {allocation_time:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])