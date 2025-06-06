"""Batch game operations for vectorized MCTS

This module provides efficient batch operations for game state management,
reducing the overhead of sequential state updates.
"""

import torch
from typing import List, Dict, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
from dataclasses import dataclass
import numpy as np


@dataclass
class BatchGameOpsConfig:
    """Configuration for batch game operations"""
    device: str = 'cuda'
    max_batch_size: int = 2048
    cache_states: bool = True
    

class BatchGameOps:
    """Batch operations for game state management
    
    This class provides vectorized game operations to minimize
    sequential processing overhead.
    """
    
    def __init__(self, config: BatchGameOpsConfig):
        """Initialize batch game operations
        
        Args:
            config: Configuration
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # State cache for faster access
        if config.cache_states:
            self.state_cache = {}
        else:
            self.state_cache = None
            
    def batch_clone_states(self, states: List[Any]) -> List[Any]:
        """Clone multiple states efficiently
        
        Args:
            states: List of game states
            
        Returns:
            List of cloned states
        """
        # For simple states, we can use list comprehension
        # For tensor-based states, we could use batch operations
        return [self._fast_clone(state) for state in states]
        
    def batch_make_moves(self, 
                        states: List[Any], 
                        actions: List[int],
                        game_interface: Any) -> List[Any]:
        """Apply moves to multiple states in batch
        
        Args:
            states: List of game states
            actions: List of actions to apply
            game_interface: Game interface for move application
            
        Returns:
            List of resulting states
        """
        batch_size = len(states)
        result_states = []
        
        # Group by state for efficiency
        state_groups = {}
        for i, (state, action) in enumerate(zip(states, actions)):
            state_key = self._get_state_key(state)
            if state_key not in state_groups:
                state_groups[state_key] = []
            state_groups[state_key].append((i, action))
            
        # Process each group
        for state_key, action_list in state_groups.items():
            base_state = states[action_list[0][0]]
            
            # Apply all actions from this state
            for idx, action in action_list:
                if self.state_cache is not None:
                    cache_key = f"{state_key}_{action}"
                    if cache_key in self.state_cache:
                        result_states.append((idx, self.state_cache[cache_key]))
                        continue
                        
                # Apply move
                new_state = game_interface.make_move(
                    game_interface.clone_state(base_state), 
                    action
                )
                
                # Cache result
                if self.state_cache is not None:
                    self.state_cache[cache_key] = new_state
                    
                result_states.append((idx, new_state))
                
        # Sort by original index
        result_states.sort(key=lambda x: x[0])
        return [state for _, state in result_states]
        
    def batch_get_legal_moves(self, 
                             states: List[Any],
                             game_interface: Any) -> List[List[int]]:
        """Get legal moves for multiple states
        
        Args:
            states: List of game states
            game_interface: Game interface
            
        Returns:
            List of legal move lists
        """
        # Cache legal moves by state
        legal_moves_cache = {}
        results = []
        
        for state in states:
            state_key = self._get_state_key(state)
            if state_key in legal_moves_cache:
                results.append(legal_moves_cache[state_key])
            else:
                legal_moves = game_interface.get_legal_moves(state)
                legal_moves_cache[state_key] = legal_moves
                results.append(legal_moves)
                
        return results
        
    def batch_evaluate_terminal(self,
                               states: List[Any],
                               game_interface: Any) -> Tuple['torch.Tensor', 'torch.Tensor']:
        """Check terminal status and get values for multiple states
        
        Args:
            states: List of game states
            game_interface: Game interface
            
        Returns:
            Tuple of:
                - is_terminal: Boolean tensor of terminal status
                - terminal_values: Tensor of terminal values
        """
        batch_size = len(states)
        is_terminal = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        terminal_values = torch.zeros(batch_size, device=self.device)
        
        for i, state in enumerate(states):
            # This could be optimized with game-specific batch operations
            if hasattr(game_interface, 'is_terminal'):
                is_terminal[i] = game_interface.is_terminal(state)
                if is_terminal[i]:
                    terminal_values[i] = game_interface.get_terminal_value(state)
                    
        return is_terminal, terminal_values
        
    def _get_state_key(self, state: Any) -> str:
        """Get hash key for state
        
        Args:
            state: Game state
            
        Returns:
            Hash key
        """
        # Simple string representation for now
        # Could be optimized with proper hashing
        return str(state)
        
    def _fast_clone(self, state: Any) -> Any:
        """Fast state cloning
        
        Args:
            state: State to clone
            
        Returns:
            Cloned state
        """
        # For tensor states, use torch.clone()
        if isinstance(state, torch.Tensor):
            return state.clone()
        elif isinstance(state, np.ndarray):
            return state.copy()
        else:
            # Fallback to deepcopy
            import copy
            return copy.deepcopy(state)
            
    def clear_cache(self):
        """Clear state cache"""
        if self.state_cache is not None:
            self.state_cache.clear()
            
            
class TensorGameState:
    """Base class for tensor-based game states
    
    This enables fully vectorized game operations on GPU.
    """
    
    def __init__(self, board_tensor: 'torch.Tensor', metadata: Dict[str, Any]):
        """Initialize tensor game state
        
        Args:
            board_tensor: Board representation as tensor
            metadata: Game-specific metadata
        """
        self.board = board_tensor
        self.metadata = metadata
        
    def clone(self) -> 'TensorGameState':
        """Clone the state"""
        return TensorGameState(
            self.board.clone(),
            self.metadata.copy()
        )
        
    def to_device(self, device: 'torch.device') -> 'TensorGameState':
        """Move state to device"""
        return TensorGameState(
            self.board.to(device),
            self.metadata
        )