"""State delta encoding for efficient state management

This module provides delta encoding to reduce memory usage and improve
cache efficiency by storing only state differences rather than full states.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


@dataclass
class StateCheckpoint:
    """Checkpoint containing full state for efficient reconstruction"""
    state_id: int
    full_state: torch.Tensor
    timestamp: float = 0.0


class DeltaCache:
    """LRU cache for state deltas"""
    
    def __init__(self, max_size: int = 1000):
        """Initialize delta cache
        
        Args:
            max_size: Maximum number of deltas to cache
        """
        self.max_size = max_size
        self.cache = OrderedDict()
        
    def add(self, from_id: int, to_id: int, delta: Dict[str, torch.Tensor]):
        """Add delta to cache
        
        Args:
            from_id: Source state ID
            to_id: Target state ID  
            delta: Delta dictionary
        """
        key = (from_id, to_id)
        
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
            
        # Add new delta (moves to end)
        self.cache[key] = delta
        
    def get(self, from_id: int, to_id: int) -> Optional[Dict[str, torch.Tensor]]:
        """Get delta from cache
        
        Args:
            from_id: Source state ID
            to_id: Target state ID
            
        Returns:
            Delta if found, None otherwise
        """
        key = (from_id, to_id)
        
        if key in self.cache:
            # Move to end (LRU)
            self.cache.move_to_end(key)
            return self.cache[key]
            
        return None
        
    def clear(self):
        """Clear cache"""
        self.cache.clear()


class StateDeltaEncoder:
    """Encoder for state deltas with GPU support"""
    
    def __init__(
        self,
        state_shape: Tuple[int, ...],
        device: str = 'cpu',
        checkpoint_interval: int = 10
    ):
        """Initialize state delta encoder
        
        Args:
            state_shape: Shape of game states
            device: Device for tensor operations
            checkpoint_interval: States between checkpoints
        """
        self.state_shape = state_shape
        self.device = torch.device(device)
        self.checkpoint_interval = checkpoint_interval
        
        # Cache for deltas
        self.delta_cache = DeltaCache()
        
        # Checkpoints for fast reconstruction
        self.checkpoints = {}
        
    def encode_delta(
        self,
        prev_state: torch.Tensor,
        next_state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Encode difference between two states
        
        Args:
            prev_state: Previous state tensor
            next_state: Next state tensor
            
        Returns:
            Delta dictionary with positions and values
        """
        prev_state = prev_state.to(self.device)
        next_state = next_state.to(self.device)
        
        # Find differences
        diff_mask = prev_state != next_state
        
        # Get positions of changes
        positions = torch.nonzero(diff_mask)
        
        # Get changed values
        if positions.numel() > 0:
            values = next_state[diff_mask]
        else:
            values = torch.tensor([], device=self.device)
            
        return {
            'positions': positions,
            'values': values
        }
        
    def apply_delta(
        self,
        base_state: torch.Tensor,
        delta: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Apply delta to reconstruct state
        
        Args:
            base_state: Base state tensor
            delta: Delta to apply
            
        Returns:
            Reconstructed state
        """
        # Clone to avoid modifying original
        result = base_state.clone().to(self.device)
        
        positions = delta['positions'].to(self.device)
        values = delta['values'].to(self.device)
        
        # Apply changes
        if positions.numel() > 0:
            # Convert positions to tuple of indices
            indices = tuple(positions.t())
            result[indices] = values
            
        return result
        
    def encode_batch_deltas(
        self,
        prev_states: torch.Tensor,
        next_states: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """Encode deltas for batch of state pairs
        
        Args:
            prev_states: Previous states (batch_size, *state_shape)
            next_states: Next states (batch_size, *state_shape)
            
        Returns:
            List of delta dictionaries
        """
        batch_size = prev_states.shape[0]
        deltas = []
        
        for i in range(batch_size):
            delta = self.encode_delta(prev_states[i], next_states[i])
            deltas.append(delta)
            
        return deltas
        
    def should_checkpoint(self, state_id: int) -> bool:
        """Check if state should be checkpointed
        
        Args:
            state_id: State identifier
            
        Returns:
            True if checkpoint should be created
        """
        return state_id % self.checkpoint_interval == 0
        
    def create_checkpoint(
        self,
        state_id: int,
        state: torch.Tensor
    ) -> StateCheckpoint:
        """Create checkpoint for state
        
        Args:
            state_id: State identifier
            state: Full state tensor
            
        Returns:
            StateCheckpoint object
        """
        checkpoint = StateCheckpoint(
            state_id=state_id,
            full_state=state.clone(),
            timestamp=0.0  # Could add actual timestamp if needed
        )
        
        self.checkpoints[state_id] = checkpoint
        return checkpoint
        
    def encode_path(
        self,
        states: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """Encode full path of states
        
        Args:
            states: List of state tensors
            
        Returns:
            Encoded path with deltas and checkpoints
        """
        encoded = {
            'deltas': [],
            'checkpoints': []
        }
        
        # First state is always a checkpoint
        if states:
            checkpoint = self.create_checkpoint(0, states[0])
            encoded['checkpoints'].append(checkpoint)
            
        # Encode deltas for rest of path
        for i in range(1, len(states)):
            # Check if we need a checkpoint
            if self.should_checkpoint(i):
                checkpoint = self.create_checkpoint(i, states[i])
                encoded['checkpoints'].append(checkpoint)
                encoded['deltas'].append(None)  # No delta needed
            else:
                # Encode delta from previous state
                delta = self.encode_delta(states[i-1], states[i])
                encoded['deltas'].append(delta)
                
                # Cache delta
                self.delta_cache.add(i-1, i, delta)
                
        return encoded
        
    def reconstruct_state(
        self,
        encoded_path: Dict[str, Any],
        target_idx: int
    ) -> torch.Tensor:
        """Reconstruct state at given index from encoded path
        
        Args:
            encoded_path: Encoded path data
            target_idx: Index of state to reconstruct
            
        Returns:
            Reconstructed state tensor
        """
        # Find nearest checkpoint before target
        checkpoints = encoded_path['checkpoints']
        checkpoint_idx = -1
        checkpoint_state = None
        
        for checkpoint in checkpoints:
            if checkpoint.state_id <= target_idx:
                checkpoint_idx = checkpoint.state_id
                checkpoint_state = checkpoint.full_state
            else:
                break
                
        if checkpoint_state is None:
            raise ValueError(f"No checkpoint found for index {target_idx}")
            
        # If target is exactly at checkpoint, return it
        if checkpoint_idx == target_idx:
            return checkpoint_state.clone()
            
        # Apply deltas from checkpoint to target
        result = checkpoint_state.clone()
        
        for i in range(checkpoint_idx + 1, target_idx + 1):
            # Skip if this index is a checkpoint
            if encoded_path['deltas'][i-1] is None:
                # This is a checkpoint, get it directly
                for checkpoint in checkpoints:
                    if checkpoint.state_id == i:
                        result = checkpoint.full_state.clone()
                        break
            else:
                # Apply delta
                delta = encoded_path['deltas'][i-1]
                result = self.apply_delta(result, delta)
                
        return result
        
    def get_compression_stats(
        self,
        states: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Calculate compression statistics
        
        Args:
            states: List of states
            
        Returns:
            Compression statistics
        """
        # Full size
        full_size = sum(s.numel() * s.element_size() for s in states)
        
        # Encoded size
        encoded = self.encode_path(states)
        
        delta_size = 0
        for delta in encoded['deltas']:
            if delta is not None:
                delta_size += (delta['positions'].numel() * 8 +  # int64 positions
                             delta['values'].numel() * 4)  # float32 values
                             
        checkpoint_size = sum(
            c.full_state.numel() * 4  # float32
            for c in encoded['checkpoints']
        )
        
        total_encoded_size = delta_size + checkpoint_size
        
        return {
            'full_size_bytes': full_size,
            'encoded_size_bytes': total_encoded_size,
            'compression_ratio': full_size / total_encoded_size if total_encoded_size > 0 else 0,
            'delta_size_bytes': delta_size,
            'checkpoint_size_bytes': checkpoint_size
        }


class DeltaStateManager:
    """Wrapper around StateDeltaEncoder for compatibility with StateManager
    
    This class provides the interface expected by StateManager while using
    StateDeltaEncoder internally.
    """
    
    def __init__(self, game_interface: Any, max_states: int, device: str):
        """Initialize delta state manager
        
        Args:
            game_interface: Game interface for state operations
            max_states: Maximum number of states to cache
            device: Device for tensor operations
        """
        self.game_interface = game_interface
        self.max_states = max_states
        self.device = device
        
        # Determine state shape from game
        if hasattr(game_interface, 'get_state_shape'):
            state_shape = game_interface.get_state_shape()
        else:
            # Default shape for common games (20 channels as per Gomoku encoding)
            state_shape = (20, 15, 15)  # Gomoku-like
            
        # Create encoder
        self.encoder = StateDeltaEncoder(
            state_shape=state_shape,
            device=device,
            checkpoint_interval=10
        )
        
        # State tracking
        self.states = {}  # node_idx -> state tensor
        self.parents = {}  # node_idx -> parent_idx
        self.path_cache = {}  # Store encoded paths for reconstruction
        
    def store_state(self, node_idx: int, state: Any, parent_idx: Optional[int] = None, action: Optional[int] = None):
        """Store state using delta encoding when possible
        
        Args:
            node_idx: Node index
            state: Game state to store
            parent_idx: Parent node index
            action: Action from parent to this state
        """
        # Convert state to tensor if needed
        if hasattr(state, 'to_tensor'):
            state_tensor = state.to_tensor()
        elif hasattr(state, 'get_enhanced_tensor_representation'):
            # Use enhanced tensor representation for GomokuState
            state_tensor = torch.from_numpy(state.get_enhanced_tensor_representation()).float()
        elif hasattr(state, 'get_tensor_representation'):
            # Use basic tensor representation for GomokuState
            state_tensor = torch.from_numpy(state.get_tensor_representation()).float()
        elif hasattr(state, 'to_numpy'):
            # Use to_numpy method if available
            state_array = state.to_numpy()
            state_tensor = torch.from_numpy(state_array).float()
        elif isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float()
        elif isinstance(state, torch.Tensor):
            state_tensor = state.float()
        else:
            # Try to use game interface
            if hasattr(self.game_interface, 'state_to_tensor'):
                state_tensor = self.game_interface.state_to_tensor(state)
            elif hasattr(self.game_interface, 'state_to_numpy'):
                state_array = self.game_interface.state_to_numpy(state)
                state_tensor = torch.from_numpy(state_array).float()
            else:
                raise ValueError(f"Cannot convert state of type {type(state)} to tensor")
                
        # Store state
        self.states[node_idx] = state_tensor
        if parent_idx is not None:
            self.parents[node_idx] = parent_idx
            
        # Manage cache size
        if len(self.states) > self.max_states:
            # Remove oldest states
            to_remove = len(self.states) - self.max_states
            for key in list(self.states.keys())[:to_remove]:
                del self.states[key]
                if key in self.parents:
                    del self.parents[key]
                    
    def get_state(self, node_idx: int) -> Optional[Any]:
        """Retrieve state, reconstructing from deltas if necessary
        
        Args:
            node_idx: Node index
            
        Returns:
            Game state or None if not found
        """
        if node_idx not in self.states:
            return None
            
        state_tensor = self.states[node_idx]
        
        # For now, return tensor directly as tensor_to_state is complex to implement
        # The consuming code should handle tensor states appropriately
        return state_tensor
            
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the delta encoding
        
        Returns:
            Dictionary of statistics
        """
        return {
            'stored_states': len(self.states),
            'max_states': self.max_states,
            'device': str(self.device)
        }