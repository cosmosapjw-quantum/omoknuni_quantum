"""Tests for the replay buffer module"""

import pytest
import numpy as np
import torch
from pathlib import Path

from mcts.neural_networks.replay_buffer import ReplayBuffer, GameExample


class TestGameExample:
    """Test the GameExample dataclass"""
    
    def test_game_example_creation(self):
        """Test creating a game example"""
        state = np.array([[1, 0], [0, 1]])
        policy = np.array([0.25, 0.25, 0.25, 0.25])
        value = 0.5
        
        example = GameExample(
            state=state,
            policy=policy,
            value=value,
            game_id="test_game_1",
            move_number=5
        )
        
        assert np.array_equal(example.state, state)
        assert np.array_equal(example.policy, policy)
        assert example.value == value
        assert example.game_id == "test_game_1"
        assert example.move_number == 5
    
    def test_game_example_ensures_contiguous_arrays(self):
        """Test that GameExample ensures arrays are contiguous"""
        # Create non-contiguous arrays
        state = np.array([[1, 2], [3, 4]]).T  # Transpose creates non-contiguous
        policy = np.array([0.1, 0.2, 0.3, 0.4])[::-1]  # Reverse slice is non-contiguous
        
        example = GameExample(state=state, policy=policy, value=0.0)
        
        # Should be contiguous after initialization
        assert example.state.flags['C_CONTIGUOUS']
        assert example.policy.flags['C_CONTIGUOUS']


class TestReplayBuffer:
    """Test the ReplayBuffer class"""
    
    def test_replay_buffer_initialization(self):
        """Test replay buffer initialization"""
        buffer = ReplayBuffer(max_size=100)
        
        assert len(buffer) == 0
        assert buffer.max_size == 100
    
    def test_add_examples(self):
        """Test adding examples to buffer"""
        buffer = ReplayBuffer(max_size=10)
        
        examples = [
            GameExample(
                state=np.zeros((3, 3)),
                policy=np.ones(9) / 9,
                value=0.0
            ) for _ in range(5)
        ]
        
        buffer.add(examples)
        assert len(buffer) == 5
    
    def test_buffer_max_size(self):
        """Test that buffer respects max size"""
        buffer = ReplayBuffer(max_size=5)
        
        # Add 10 examples to a buffer with max size 5
        examples = [
            GameExample(
                state=np.ones((3, 3)) * i,
                policy=np.ones(9) / 9,
                value=float(i)
            ) for i in range(10)
        ]
        
        buffer.add(examples)
        assert len(buffer) == 5
        
        # Should keep the last 5 examples (indices 5-9)
        for i, (state, policy, value) in enumerate(buffer):
            expected_value = float(i + 5)
            assert value.item() == expected_value
    
    def test_getitem_returns_tensors(self):
        """Test that __getitem__ returns proper tensors"""
        buffer = ReplayBuffer()
        
        example = GameExample(
            state=np.array([[1, 0], [0, 1]]),
            policy=np.array([0.25, 0.25, 0.25, 0.25]),
            value=0.5
        )
        buffer.add([example])
        
        state, policy, value = buffer[0]
        
        assert isinstance(state, torch.Tensor)
        assert isinstance(policy, torch.Tensor)
        assert isinstance(value, torch.Tensor)
        assert state.shape == (2, 2)
        assert policy.shape == (4,)
        assert value.shape == (1,)
    
    def test_clear_buffer(self):
        """Test clearing the buffer"""
        buffer = ReplayBuffer()
        buffer.add([GameExample(np.zeros((3, 3)), np.ones(9), 0.0) for _ in range(5)])
        
        assert len(buffer) == 5
        buffer.clear()
        assert len(buffer) == 0
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading buffer"""
        buffer = ReplayBuffer()
        
        examples = [
            GameExample(
                state=np.ones((3, 3)) * i,
                policy=np.ones(9) / 9,
                value=float(i)
            ) for i in range(3)
        ]
        buffer.add(examples)
        
        # Save buffer
        save_path = tmp_path / "test_buffer.pkl"
        buffer.save(str(save_path))
        
        # Load into new buffer
        new_buffer = ReplayBuffer()
        new_buffer.load(str(save_path))
        
        assert len(new_buffer) == 3
        for i in range(3):
            _, _, value = new_buffer[i]
            assert value.item() == float(i)