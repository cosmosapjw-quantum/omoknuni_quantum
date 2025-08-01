"""Wrapper for CPU game states with recycling support

Provides factory function and wrapper class for CPU game states.
"""

import numpy as np
import torch
from typing import Union, List, Optional
import logging

from .cpu_game_states import CPUGameStates

logger = logging.getLogger(__name__)


class RecyclingCPUGameStates:
    """Wrapper that adds recycling functionality to CPU game states"""
    
    def __init__(self, base_game_states: CPUGameStates):
        """Initialize with base game states"""
        self.base = base_game_states
        
        # For compatibility with MCTS
        self.capacity = base_game_states.capacity
        self.game_type = base_game_states.game_type
        self.board_size = base_game_states.board_size
        
    def __getattr__(self, name):
        """Forward all other attributes to base"""
        return getattr(self.base, name)
        
    def reset(self):
        """Reset states - ensure state 0 is allocated"""
        # Clear all states
        if hasattr(self.base, '_free_indices'):
            self.base._free_indices.clear()
        self.base.num_states = 0
        self.base._allocated_states = 0
        
        # Re-allocate state 0 for root
        indices = self.base.allocate_states(1)
        assert indices[0] == 0, f"Expected state 0 for root, got {indices[0]}"
        

def create_cpu_game_states_with_recycling(capacity: int, game_type: str = 'gomoku', 
                                         board_size: int = 15, device: str = 'cpu',
                                         enable_recycling: bool = True, **kwargs) -> RecyclingCPUGameStates:
    """Create CPU game states with optional recycling
    
    Args:
        capacity: Maximum number of states
        game_type: Type of game
        board_size: Board size
        device: Device (ignored, always CPU)
        enable_recycling: Whether to enable recycling
        **kwargs: Additional arguments (ignored)
        
    Returns:
        CPU game states instance (with recycling wrapper if enabled)
    """
    # Import here to get the potentially optimized version
    from . import CPUGameStates as GameStatesClass
    
    # Create base game states
    base_states = GameStatesClass(
        capacity=capacity,
        game_type=game_type,
        board_size=board_size
    )
    
    if enable_recycling:
        return RecyclingCPUGameStates(base_states)
    else:
        return base_states