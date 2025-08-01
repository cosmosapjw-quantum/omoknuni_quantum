"""CPU-optimized game state management

Pure Python/numpy implementation for CPU efficiency.
No pre-allocation, dynamic growth, efficient recycling.
"""

import numpy as np
import torch
from typing import List, Union, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CPUGameStates:
    """CPU-optimized game state management
    
    Key features:
    - On-demand allocation (no pre-allocation)
    - Efficient state recycling
    - Numpy-based for CPU performance
    - Compatible with wave-based MCTS
    """
    
    def __init__(self, capacity: int, game_type: str = 'gomoku', board_size: Optional[int] = None):
        """Initialize CPU game states
        
        Args:
            capacity: Maximum number of states (not pre-allocated)
            game_type: Type of game ('gomoku', 'go', 'chess')
            board_size: Board size (15 for gomoku, 19 for go, 8 for chess)
        """
        self.capacity = capacity
        self.game_type = game_type.lower()
        
        # Set default board sizes based on game type
        if board_size is None:
            if self.game_type == 'chess':
                self.board_size = 8
            elif self.game_type == 'go':
                self.board_size = 19
            else:  # gomoku
                self.board_size = 15
        else:
            self.board_size = board_size
            
        # State tracking
        self.num_states = 0
        self._allocated_states = 0  # Track actual allocated states
        
        # Action size based on game type
        self.action_size = self.board_size * self.board_size
        
        # Dynamic state storage - start empty
        self._boards = None
        self._current_player = None
        self._move_count = None
        self._is_terminal = None
        self._winner = None
        
        # Free state tracking
        self._free_indices = []
        
        # Initial allocation size
        self._chunk_size = min(1000, capacity)  # Allocate in chunks
        
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        total_bytes = 0
        
        if self._boards is not None:
            total_bytes += self._boards.nbytes
        if self._current_player is not None:
            total_bytes += self._current_player.nbytes
        if self._move_count is not None:
            total_bytes += self._move_count.nbytes
        if self._is_terminal is not None:
            total_bytes += self._is_terminal.nbytes
        if self._winner is not None:
            total_bytes += self._winner.nbytes
            
        return total_bytes / (1024 * 1024)
    
    def allocate_states(self, num_states: int) -> np.ndarray:
        """Allocate states on demand
        
        Args:
            num_states: Number of states to allocate
            
        Returns:
            Array of allocated state indices
        """
        if isinstance(num_states, (torch.Tensor, np.ndarray)):
            num_states = int(num_states.item() if hasattr(num_states, 'item') else num_states)
            
        # Check capacity
        available_space = self.capacity - self.num_states + len(self._free_indices)
        if available_space < num_states:
            # Log warning only once per capacity reached
            if not hasattr(self, '_capacity_warning_shown'):
                self._capacity_warning_shown = True
                logger.warning(f"State allocation capacity reached. Cannot allocate {num_states} states. "
                             f"Current: {self.num_states}, Free: {len(self._free_indices)}, Capacity: {self.capacity}")
            # Return empty array instead of raising exception
            return np.array([], dtype=np.int32)
        
        indices = []
        
        # First, try to reuse free indices
        while len(indices) < num_states and self._free_indices:
            indices.append(self._free_indices.pop())
            
        # If we need more states, allocate new ones
        new_states_needed = num_states - len(indices)
        if new_states_needed > 0:
            # Ensure we have enough allocated space
            self._ensure_capacity(self._allocated_states + new_states_needed)
            
            # Allocate new indices
            start_idx = self._allocated_states
            end_idx = start_idx + new_states_needed
            indices.extend(range(start_idx, end_idx))
            self._allocated_states = end_idx
            
        # Update state count
        self.num_states += num_states
        
        return np.array(indices, dtype=np.int32)
    
    def _ensure_capacity(self, required_size: int):
        """Ensure we have enough allocated capacity"""
        if self._boards is None or required_size > len(self._boards):
            # Calculate new size (grow in chunks)
            new_size = ((required_size + self._chunk_size - 1) // self._chunk_size) * self._chunk_size
            new_size = min(new_size, self.capacity)
            
            # Allocate or grow arrays
            if self._boards is None:
                # First allocation
                self._boards = np.zeros((new_size, self.board_size, self.board_size), dtype=np.int8)
                self._current_player = np.ones(new_size, dtype=np.int8)
                self._move_count = np.zeros(new_size, dtype=np.int16)
                self._is_terminal = np.zeros(new_size, dtype=bool)
                self._winner = np.zeros(new_size, dtype=np.int8)
            else:
                # Grow existing arrays
                old_size = len(self._boards)
                growth = new_size - old_size
                
                # Extend arrays
                self._boards = np.concatenate([
                    self._boards,
                    np.zeros((growth, self.board_size, self.board_size), dtype=np.int8)
                ])
                self._current_player = np.concatenate([
                    self._current_player,
                    np.ones(growth, dtype=np.int8)
                ])
                self._move_count = np.concatenate([
                    self._move_count,
                    np.zeros(growth, dtype=np.int16)
                ])
                self._is_terminal = np.concatenate([
                    self._is_terminal,
                    np.zeros(growth, dtype=bool)
                ])
                self._winner = np.concatenate([
                    self._winner,
                    np.zeros(growth, dtype=np.int8)
                ])
    
    def free_states(self, indices: Union[List[int], np.ndarray]):
        """Free states for recycling
        
        Args:
            indices: State indices to free
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        elif not isinstance(indices, np.ndarray):
            indices = np.array(indices)
            
        # Reset state data
        for idx in indices:
            if 0 <= idx < self._allocated_states:
                # Reset board
                self._boards[idx] = 0
                # Reset metadata
                self._current_player[idx] = 1
                self._move_count[idx] = 0
                self._is_terminal[idx] = False
                self._winner[idx] = 0
                
                # Add to free list
                self._free_indices.append(idx)
                
        # Update count
        self.num_states -= len(indices)
    
    def apply_moves(self, state_indices: Union[List[int], np.ndarray], 
                   actions: Union[List[int], np.ndarray]):
        """Apply moves to states
        
        Args:
            state_indices: Indices of states to modify
            actions: Actions to apply (board positions)
        """
        # Convert to numpy arrays
        if isinstance(state_indices, torch.Tensor):
            state_indices = state_indices.cpu().numpy()
        elif not isinstance(state_indices, np.ndarray):
            state_indices = np.array(state_indices)
            
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        elif not isinstance(actions, np.ndarray):
            actions = np.array(actions)
            
        # Apply moves based on game type
        if self.game_type in ['gomoku', 'go']:
            # Convert action to row/col
            rows = actions // self.board_size
            cols = actions % self.board_size
            
            # Place pieces
            for i, state_idx in enumerate(state_indices):
                player = self._current_player[state_idx]
                self._boards[state_idx, rows[i], cols[i]] = player
                
                # Update metadata
                self._current_player[state_idx] = 3 - player  # Switch player
                self._move_count[state_idx] += 1
                
                # TODO: Check for terminal states (wins)
                
    def get_legal_moves_mask(self, state_indices: Union[List[int], np.ndarray]) -> np.ndarray:
        """Get legal moves masks for states
        
        Args:
            state_indices: State indices to query
            
        Returns:
            Boolean masks of shape (num_states, action_space_size)
        """
        # Convert to numpy array
        if isinstance(state_indices, torch.Tensor):
            state_indices = state_indices.cpu().numpy()
        elif not isinstance(state_indices, np.ndarray):
            state_indices = np.array(state_indices)
            
        num_states = len(state_indices)
        action_size = self.board_size * self.board_size
        
        # Initialize masks
        legal_masks = np.zeros((num_states, action_size), dtype=bool)
        
        # For each state
        for i, state_idx in enumerate(state_indices):
            if self.game_type == 'gomoku':
                # Legal moves are empty squares
                # Convert state_idx to int if it's numpy type
                if hasattr(state_idx, 'item'):
                    state_idx = int(state_idx.item())
                else:
                    state_idx = int(state_idx)
                    
                if state_idx >= self._allocated_states:
                    continue
                board = self._boards[state_idx]
                empty_mask = (board == 0).flatten()
                legal_masks[i] = empty_mask
                
        return legal_masks
    
    def clone_states(self, parent_indices: Union[List[int], np.ndarray],
                    num_clones_per_parent: Union[List[int], np.ndarray]) -> np.ndarray:
        """Clone states
        
        Args:
            parent_indices: Indices of states to clone
            num_clones_per_parent: Number of clones for each parent
            
        Returns:
            Array of clone indices
        """
        # Convert to numpy
        if isinstance(parent_indices, torch.Tensor):
            parent_indices = parent_indices.cpu().numpy()
        elif not isinstance(parent_indices, np.ndarray):
            parent_indices = np.array(parent_indices)
            
        if isinstance(num_clones_per_parent, torch.Tensor):
            num_clones_per_parent = num_clones_per_parent.cpu().numpy()
        elif not isinstance(num_clones_per_parent, np.ndarray):
            num_clones_per_parent = np.array(num_clones_per_parent)
            
        # Calculate total clones
        total_clones = int(num_clones_per_parent.sum())
        
        # Allocate clone states
        clone_indices = self.allocate_states(total_clones)
        
        # Check if allocation failed
        if len(clone_indices) == 0:
            return np.array([], dtype=np.int32)
        
        # Copy state data
        clone_idx = 0
        for parent_idx, num_clones in zip(parent_indices, num_clones_per_parent):
            for _ in range(num_clones):
                # Copy board
                self._boards[clone_indices[clone_idx]] = self._boards[parent_idx].copy()
                # Copy metadata
                self._current_player[clone_indices[clone_idx]] = self._current_player[parent_idx]
                self._move_count[clone_indices[clone_idx]] = self._move_count[parent_idx]
                self._is_terminal[clone_indices[clone_idx]] = self._is_terminal[parent_idx]
                self._winner[clone_indices[clone_idx]] = self._winner[parent_idx]
                
                clone_idx += 1
                
        return clone_indices
    
    def apply_actions(self, state_indices: Union[List[int], np.ndarray], 
                     actions: Union[List[int], np.ndarray]):
        """Apply actions to states
        
        Args:
            state_indices: State indices to apply actions to
            actions: Actions to apply
        """
        # Convert to numpy
        if isinstance(state_indices, torch.Tensor):
            state_indices = state_indices.cpu().numpy()
        elif not isinstance(state_indices, np.ndarray):
            state_indices = np.array(state_indices)
            
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        elif not isinstance(actions, np.ndarray):
            actions = np.array(actions)
            
        # Apply actions
        for state_idx, action in zip(state_indices, actions):
            if state_idx >= 0 and state_idx < self._allocated_states:
                # Convert action to row, col
                row = action // self.board_size
                col = action % self.board_size
                
                # Apply move
                current_player = self._current_player[state_idx]
                self._boards[state_idx, row, col] = current_player
                
                # Update metadata
                self._current_player[state_idx] = 3 - current_player  # Switch player
                self._move_count[state_idx] += 1
                
                # Check for terminal state (simplified - just check if board is full)
                if self._move_count[state_idx] >= self.board_size * self.board_size:
                    self._is_terminal[state_idx] = True
    
    def get_board(self, state_idx: int) -> np.ndarray:
        """Get board for a single state"""
        return self._boards[state_idx].copy()
    
    def get_current_player(self, state_idx: int) -> int:
        """Get current player for a state"""
        return int(self._current_player[state_idx])
    
    def get_move_count(self, state_idx: int) -> int:
        """Get move count for a state"""
        return int(self._move_count[state_idx])
    
    def is_terminal(self, state_idx: int) -> bool:
        """Check if state is terminal"""
        return bool(self._is_terminal[state_idx])
    
    def get_nn_features(self, state_indices: Union[List[int], np.ndarray],
                       representation_type: str = 'basic') -> np.ndarray:
        """Get neural network features
        
        Args:
            state_indices: State indices to get features for
            representation_type: 'basic' (19 channels) or 'enhanced' (21 channels)
            
        Returns:
            Feature tensor of shape (batch_size, channels, board_size, board_size)
        """
        # Convert to numpy
        if isinstance(state_indices, torch.Tensor):
            state_indices = state_indices.cpu().numpy()
        elif not isinstance(state_indices, np.ndarray):
            state_indices = np.array(state_indices)
            
        batch_size = len(state_indices)
        
        # Basic representation (AlphaZero standard): 19 channels
        # - 8 channels for current player pieces (history)
        # - 8 channels for opponent pieces (history)  
        # - 1 channel for color to play
        # - 1 channel for total move count
        # - 1 channel for all ones
        
        if representation_type == 'basic':
            num_channels = 19
            # Create feature tensor
            features = np.zeros((batch_size, num_channels, self.board_size, self.board_size), 
                              dtype=np.float32)
            
            for i, state_idx in enumerate(state_indices):
                board = self._boards[state_idx]
                current_player = self._current_player[state_idx]
                move_count = self._move_count[state_idx]
                
                # Simple feature extraction for Gomoku
                # Channel 0: Current player pieces
                features[i, 0] = (board == current_player).astype(np.float32)
                
                # Channel 1: Opponent pieces
                opponent = 3 - current_player  # 1->2, 2->1
                features[i, 1] = (board == opponent).astype(np.float32)
                
                # For now, skip history channels (2-15) - set to zero
                
                # Channel 16: Color to play (all ones if player 1, all zeros if player 2)
                features[i, 16] = float(current_player == 1)
                
                # Channel 17: Total move count (normalized)
                features[i, 17] = move_count / 100.0  # Normalize to reasonable range
                
                # Channel 18: All ones
                features[i, 18] = 1.0
                
        else:  # enhanced - 21 channels
            num_channels = 21
            features = np.zeros((batch_size, num_channels, self.board_size, self.board_size), 
                              dtype=np.float32)
            
            # Similar to basic but with 2 additional channels
            # TODO: Implement enhanced representation if needed
            
        return features
    
    def set_board_from_tensor(self, state_idx: int, board_tensor: Union[torch.Tensor, np.ndarray]):
        """Set board state from tensor
        
        Args:
            state_idx: State index to update
            board_tensor: Board tensor with piece positions
        """
        if isinstance(board_tensor, torch.Tensor):
            board_tensor = board_tensor.cpu().numpy()
            
        # Ensure state is allocated
        if self._boards is None or state_idx >= self._allocated_states:
            raise ValueError(f"State {state_idx} not allocated")
            
        # Copy board data
        self._boards[state_idx] = board_tensor.astype(np.int8)
    
    @property
    def allocated_mask(self):
        """Property for compatibility with GPU game states"""
        if self._boards is None:
            return np.zeros(0, dtype=bool)
        mask = np.zeros(self._allocated_states, dtype=bool)
        mask[:self.num_states] = True
        # Mark free indices as not allocated
        for idx in self._free_indices:
            if idx < len(mask):
                mask[idx] = False
        return mask
    
    @property
    def free_indices(self):
        """Property for compatibility with GPU game states"""
        return self._free_indices
    
    @property
    def current_player(self):
        """Property for accessing current player array"""
        return self._current_player
    
    @property
    def move_count(self):
        """Property for accessing move count array"""
        return self._move_count
    
    @property 
    def is_terminal(self):
        """Property for accessing terminal state array"""
        return self._is_terminal
    
    @property
    def game_result(self):
        """Property for accessing game result array"""
        return self._winner  # Use winner array for game result
    
    @property
    def winner(self):
        """Property for accessing winner array (alias for game_result)"""
        return self._winner
    
    @property
    def boards(self):
        """Property for accessing boards array"""
        return self._boards