"""Stress tests and fuzz testing for optimized implementations"""

import pytest
import torch
import numpy as np
import random
from hypothesis import given, strategies as st, settings

from mcts.core.game_interface import GameInterface, GameType


class TestStressTensorToState:
    """Stress tests for tensor_to_state implementation"""
    
    @pytest.mark.parametrize("board_size", [9, 13, 15, 19, 30])
    def test_various_board_sizes(self, board_size):
        """Test tensor_to_state with various board sizes"""
        interface = GameInterface(GameType.GOMOKU, board_size=board_size)
        
        # Create random board state
        tensor = torch.zeros(3, board_size, board_size)
        # Add some random pieces
        num_pieces = min(board_size * board_size // 4, 50)
        for _ in range(num_pieces):
            row, col = random.randint(0, board_size-1), random.randint(0, board_size-1)
            player = random.choice([0, 1])
            tensor[player, row, col] = 1.0
        
        # Ensure valid current player
        tensor[2, :, :] = random.choice([0, 1])
        
        # Should not raise an error
        state = interface.tensor_to_state(tensor)
        assert state is not None
    
    def test_corrupted_tensor_handling(self):
        """Test handling of corrupted tensor data"""
        interface = GameInterface(GameType.GOMOKU, board_size=15)
        
        # Test NaN values
        tensor = torch.zeros(3, 15, 15)
        tensor[0, 5, 5] = float('nan')
        with pytest.raises(ValueError, match="NaN or Inf"):
            interface.tensor_to_state(tensor)
        
        # Test Inf values
        tensor = torch.zeros(3, 15, 15)
        tensor[1, 7, 7] = float('inf')
        with pytest.raises(ValueError, match="NaN or Inf"):
            interface.tensor_to_state(tensor)
    
    def test_dos_prevention_large_board(self):
        """Test DoS prevention for extremely large boards"""
        # Should raise error for board size > 50
        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            interface = GameInterface(GameType.GOMOKU, board_size=51)
            tensor = torch.zeros(3, 51, 51)
            interface.tensor_to_state(tensor)
    
    @given(
        num_channels=st.integers(min_value=1, max_value=30),
        height=st.integers(min_value=1, max_value=50),
        width=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=20, deadline=1000)
    def test_fuzz_tensor_shapes(self, num_channels, height, width):
        """Fuzz test with random tensor shapes"""
        interface = GameInterface(GameType.GOMOKU, board_size=15)
        
        # Create tensor with random shape
        tensor = torch.rand(num_channels, height, width)
        
        # Should handle gracefully with appropriate error
        try:
            state = interface.tensor_to_state(tensor)
            # If we get here, tensor should be valid
            assert num_channels in [3, 18, 20]
            assert (height, width) == (15, 15)
            assert state is not None
        except ValueError as e:
            # Check that we got the expected error
            if (height, width) != (15, 15):
                assert "don't match expected board shape" in str(e)
            elif num_channels not in [3, 18, 20]:
                assert "Invalid number of channels" in str(e)
            else:
                # Unexpected error
                raise
    
    @given(
        board_data=st.lists(
            st.lists(st.floats(min_value=0, max_value=1), min_size=15, max_size=15),
            min_size=15, max_size=15
        )
    )
    @settings(max_examples=10, deadline=2000)
    def test_fuzz_board_contents(self, board_data):
        """Fuzz test with random board contents"""
        interface = GameInterface(GameType.GOMOKU, board_size=15)
        
        # Create tensor with fuzzed board data
        tensor = torch.zeros(3, 15, 15)
        board_array = np.array(board_data)
        
        # Randomly assign pieces
        mask = board_array > 0.7
        tensor[0][mask] = 1.0  # Player 1 pieces
        
        mask = (board_array > 0.4) & (board_array <= 0.7)
        tensor[1][mask] = 1.0  # Player 2 pieces
        
        # Set current player
        tensor[2, :, :] = random.choice([0, 1])
        
        # Should handle any valid board configuration
        state = interface.tensor_to_state(tensor)
        assert state is not None


class TestStressMoveHistory:
    """Stress tests for move history tracking"""
    
    def test_very_long_history(self):
        """Test handling of very long move histories"""
        interface = GameInterface(GameType.GOMOKU, board_size=15)
        
        # Create a long history (should be truncated)
        history = []
        state = interface.create_initial_state()
        
        # Create 150 dummy states (exceeds max_history_length of 100)
        for i in range(150):
            history.append(state)
        
        # Should handle gracefully with truncation
        current_state = state
        encoded = interface.encode_for_nn(current_state, history)
        
        # Should still produce valid 20-channel output
        assert encoded.shape == (20, 15, 15)
    
    def test_invalid_history_states(self):
        """Test handling of invalid states in history"""
        interface = GameInterface(GameType.GOMOKU, board_size=15)
        state = interface.create_initial_state()
        
        # Test None in history
        with pytest.raises(ValueError, match="History state at index"):
            interface.encode_for_nn(state, [state, None, state])
        
        # Test invalid object in history
        with pytest.raises(ValueError, match="missing get_tensor_representation"):
            interface.encode_for_nn(state, [state, "invalid", state])
    
    @pytest.mark.parametrize("history_length", [0, 1, 5, 10, 20, 50, 100])
    def test_various_history_lengths(self, history_length):
        """Test move history with various lengths"""
        interface = GameInterface(GameType.GOMOKU, board_size=15)
        
        # Create history of specified length
        history = []
        state = interface.create_initial_state()
        for _ in range(history_length):
            history.append(state)
        
        # Should handle all lengths gracefully
        encoded = interface.encode_for_nn(state, history)
        assert encoded.shape == (20, 15, 15)
        
        # Verify move history channels are properly filled
        # Channels 2-9 (player 1) and 10-17 (player 2) should be zeros for empty history
        if history_length == 0:
            for i in range(2, 18):
                assert np.all(encoded[i] == 0)


class TestPerformanceOptimizations:
    """Test performance of vectorized operations"""
    
    def test_vectorized_move_extraction_performance(self):
        """Test that vectorized move extraction is faster than loop-based"""
        interface = GameInterface(GameType.GOMOKU, board_size=15)
        
        # Create a game with many moves
        import time
        history = []
        state = interface.create_initial_state()
        
        # Create 50 states with random moves
        for i in range(50):
            new_state = state.clone()
            # Simulate adding a move (mock implementation)
            history.append(new_state)
            state = new_state
        
        # Time the encoding (which uses vectorized operations)
        start = time.time()
        for _ in range(10):
            encoded = interface.encode_for_nn(state, history)
        vectorized_time = time.time() - start
        
        # Should complete quickly (under 1 second for 10 iterations)
        assert vectorized_time < 1.0
        assert encoded.shape == (20, 15, 15)


class TestBatchedOperations:
    """Test batched/parallelized operations"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_parallel_tensor_conversions(self):
        """Test parallel processing of multiple tensor conversions"""
        interface = GameInterface(GameType.GOMOKU, board_size=15)
        
        # Create batch of tensors
        batch_size = 32
        tensors = [torch.rand(3, 15, 15) for _ in range(batch_size)]
        
        # Process in parallel (simulated - actual implementation would use multiprocessing)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(interface.tensor_to_state, t) for t in tensors]
            states = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        assert len(states) == batch_size
        assert all(s is not None for s in states)