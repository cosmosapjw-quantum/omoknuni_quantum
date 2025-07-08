"""Replay buffer for AlphaZero training

This module provides the replay buffer and game example data structures
used for storing and sampling self-play training data.
"""

import pickle
from dataclasses import dataclass
from typing import List
from collections import deque

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class GameExample:
    """Training example from self-play"""
    state: np.ndarray
    policy: np.ndarray
    value: float
    game_id: str = ""
    move_number: int = 0
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays with positive strides"""
        if not isinstance(self.state, np.ndarray):
            self.state = np.ascontiguousarray(self.state)
        else:
            self.state = np.ascontiguousarray(self.state)
        if not isinstance(self.policy, np.ndarray):
            self.policy = np.ascontiguousarray(self.policy)
        else:
            self.policy = np.ascontiguousarray(self.policy)


class ReplayBuffer(Dataset):
    """Experience replay buffer for training examples"""
    
    def __init__(self, max_size: int = 500000):
        """Initialize replay buffer with maximum size
        
        Args:
            max_size: Maximum number of examples to store
        """
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add(self, examples: List[GameExample]):
        """Add examples to buffer
        
        Args:
            examples: List of game examples to add
        """
        self.buffer.extend(examples)
    
    def __len__(self):
        """Return number of examples in buffer"""
        return len(self.buffer)
    
    def __getitem__(self, idx):
        """Get item at index, returning tensors for training
        
        Args:
            idx: Index of example to retrieve
            
        Returns:
            Tuple of (state, policy, value) tensors
        """
        example = self.buffer[idx]
        # Ensure arrays have positive strides by making copies if needed
        state = np.ascontiguousarray(example.state)
        policy = np.ascontiguousarray(example.policy)
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(policy),
            torch.FloatTensor([example.value])
        )
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
    
    def save(self, path: str):
        """Save buffer to disk
        
        Args:
            path: Path to save buffer
        """
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)
    
    def load(self, path: str):
        """Load buffer from disk
        
        Args:
            path: Path to load buffer from
        """
        with open(path, 'rb') as f:
            examples = pickle.load(f)
            self.buffer = deque(examples, maxlen=self.max_size)