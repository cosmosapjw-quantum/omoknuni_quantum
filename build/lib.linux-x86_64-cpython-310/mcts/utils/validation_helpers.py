"""Input validation helper functions

This module provides validation functions for various MCTS inputs.
Note: These are placeholder implementations - add actual validation logic as needed.
"""

import numpy as np
import torch
from typing import Any, List, Dict, Optional, Union


def validate_state(state: np.ndarray) -> None:
    """Validate game state array"""
    if not isinstance(state, np.ndarray):
        raise TypeError("State must be a numpy array")
    if state.ndim != 3:
        raise ValueError("State must be 3-dimensional (channels, height, width)")


def validate_action(action: int, action_space_size: int) -> None:
    """Validate action is within valid range"""
    if not isinstance(action, (int, np.integer)):
        raise TypeError("Action must be an integer")
    if action < 0 or action >= action_space_size:
        raise ValueError(f"Action {action} outside valid range [0, {action_space_size})")


def validate_policy(policy: np.ndarray, action_space_size: int, tolerance: float = 1e-5, 
                   legal_mask: Optional[np.ndarray] = None) -> None:
    """Validate policy probability distribution"""
    if not isinstance(policy, np.ndarray):
        raise TypeError("Policy must be a numpy array")
    if policy.shape != (action_space_size,):
        raise ValueError(f"Expected size {action_space_size}, got {policy.shape[0]}")
    if np.any(policy < 0):
        raise ValueError("Policy contains negative values")
    if np.any(np.isnan(policy)):
        raise ValueError("Policy contains NaN values")
    
    # Check legal moves if mask provided (do this before sum check)
    if legal_mask is not None:
        illegal_probs = policy[~legal_mask]
        if np.any(illegal_probs > 0):
            raise ValueError("Policy has non-zero probability on illegal moves")
            
    if not np.allclose(policy.sum(), 1.0, rtol=tolerance):
        raise ValueError("Policy doesn't sum to 1.0")


def validate_value(value: float, min_val: float = -1.0, max_val: float = 1.0) -> None:
    """Validate value is in valid range"""
    if not isinstance(value, (float, np.floating)):
        raise TypeError("Value must be float")
    if np.isnan(value):
        raise ValueError("Value is NaN")
    if not min_val <= value <= max_val:
        raise ValueError(f"Value {value} outside valid range [{min_val}, {max_val}]")


def validate_batch_states(states: np.ndarray, channels: Optional[int] = None, 
                         board_size: Optional[int] = None) -> None:
    """Validate batch of states"""
    if not isinstance(states, np.ndarray):
        raise TypeError("States must be a numpy array")
    if states.ndim != 4:
        raise ValueError("Batch states must be 4-dimensional (batch, channels, height, width)")
    
    # Check specific dimensions if provided
    if channels is not None and states.shape[1] != channels:
        raise ValueError(f"Expected {channels} channels, got {states.shape[1]}")
    if board_size is not None:
        if states.shape[2] != board_size or states.shape[3] != board_size:
            raise ValueError(f"Expected board size {board_size}x{board_size}, got {states.shape[2]}x{states.shape[3]}")


def validate_batch_policies(policies: np.ndarray, action_space_size: int) -> None:
    """Validate batch of policies"""
    if not isinstance(policies, np.ndarray):
        raise TypeError("Policies must be a numpy array")
    if policies.ndim != 2:
        raise ValueError("Batch policies must be 2-dimensional")
    if policies.shape[1] != action_space_size:
        raise ValueError(f"Policy dimension {policies.shape[1]} doesn't match action space {action_space_size}")
    
    # Check normalization for each policy
    policy_sums = policies.sum(axis=1)
    if not np.allclose(policy_sums, 1.0, rtol=1e-5):
        raise ValueError("Batch policies not normalized to sum to 1.0")


def validate_batch_values(values: np.ndarray, min_val: float = -1.0, max_val: float = 1.0) -> None:
    """Validate batch of values"""
    if not isinstance(values, np.ndarray):
        raise TypeError("Values must be a numpy array")
    if values.ndim != 1:
        raise ValueError("Batch values must be 1-dimensional")
    
    # Check value range
    if np.any(values < min_val) or np.any(values > max_val):
        raise ValueError(f"Batch values out of range [{min_val}, {max_val}]")


def validate_config_parameters(config: Dict[str, Any], constraints: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
    """Validate configuration parameters
    
    Args:
        config: Configuration dictionary
        constraints: Optional constraints dict with keys as parameter names and values as dicts with:
            - 'min': minimum value
            - 'max': maximum value  
            - 'type': expected type
            - 'required': whether parameter is required (default: False)
    """
    if not isinstance(config, dict):
        raise TypeError("Config must be a dictionary")
        
    if constraints is None:
        return
        
    for param_name, constraint in constraints.items():
        # Check required parameters
        if constraint.get('required', False) and param_name not in config:
            raise ValueError(f"Missing required parameter: {param_name}")
            
        if param_name in config:
            value = config[param_name]
            
            # Check type
            if 'type' in constraint:
                expected_type = constraint['type']
                if not isinstance(value, expected_type):
                    raise TypeError(f"Expected type {expected_type.__name__} for {param_name}, got {type(value).__name__}")
                    
            # Check range
            if 'min' in constraint and value < constraint['min']:
                raise ValueError(f"{param_name} out of range: {value} < {constraint['min']}")
            if 'max' in constraint and value > constraint['max']:
                raise ValueError(f"{param_name} out of range: {value} > {constraint['max']}")


def validate_model_output(policy: Any, value: Any, batch_size: int, action_space_size: int) -> None:
    """Validate neural network model output
    
    Args:
        policy: Policy output (logits or probabilities)
        value: Value output
        batch_size: Expected batch size
        action_space_size: Expected action space size
    """
    # Handle both numpy and torch tensors
    if hasattr(policy, 'shape'):
        policy_shape = policy.shape
    else:
        policy_shape = (len(policy), len(policy[0]) if hasattr(policy[0], '__len__') else 1)
        
    if hasattr(value, 'shape'):
        value_shape = value.shape
    else:
        value_shape = (len(value),)
    
    if policy_shape != (batch_size, action_space_size):
        raise ValueError(f"Policy shape {policy_shape} doesn't match expected ({batch_size}, {action_space_size})")
    if value_shape != (batch_size, 1) and value_shape != (batch_size,):
        raise ValueError(f"Value shape {value_shape} doesn't match expected batch size {batch_size}")


def validate_game_example(example: Any, channels: Optional[int] = None, 
                        board_size: Optional[int] = None) -> None:
    """Validate training example
    
    Args:
        example: Game example dict with fields:
            - state: Game state array
            - policy: Policy distribution
            - value: Value estimate
            - game_id: Game identifier (optional)
            - move_number: Move number (optional)
        channels: Expected number of channels (optional)
        board_size: Expected board size (optional)
    """
    required_fields = ['state', 'policy', 'value']
    
    # Check it's a dict-like object
    if not hasattr(example, 'get'):
        raise TypeError("Game example must be a dictionary")
        
    # Check required fields
    for field in required_fields:
        if field not in example:
            raise ValueError(f"Missing required field: {field}")
            
    # Validate state
    state = example['state']
    if channels is not None or board_size is not None:
        if not isinstance(state, np.ndarray):
            raise ValueError("Invalid state: must be numpy array")
        if state.ndim != 3:
            raise ValueError("Invalid state: must be 3-dimensional")
        if channels is not None and state.shape[0] != channels:
            raise ValueError(f"Invalid state: expected {channels} channels, got {state.shape[0]}")
        if board_size is not None and (state.shape[1] != board_size or state.shape[2] != board_size):
            raise ValueError(f"Invalid state: expected {board_size}x{board_size} board")
            
    # Validate policy
    policy = example['policy']
    if hasattr(state, 'shape'):
        expected_size = state.shape[1] * state.shape[2]
        if hasattr(policy, 'shape'):
            if policy.shape[0] != expected_size:
                raise ValueError(f"Policy size {policy.shape[0]} doesn't match board size {expected_size}")
        elif len(policy) != expected_size:
            raise ValueError(f"Policy size {len(policy)} doesn't match board size {expected_size}")
            
    # Validate value
    value = example['value']
    if not isinstance(value, (int, float, np.floating)):
        if hasattr(value, 'item'):  # Handle single-element tensors
            value = value.item()
        else:
            raise TypeError("Value must be numeric")
            
    # Check consistency
    if 'move_number' in example:
        move_num = example['move_number']
        if not isinstance(move_num, (int, np.integer)) or move_num < 0:
            raise ValueError("Invalid move_number: must be non-negative integer")


def check_tensor_device(tensor: torch.Tensor, expected_device: str) -> bool:
    """Check tensor is on expected device"""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Expected torch.Tensor")
    
    # Handle 'cuda' matching any cuda device (cuda:0, cuda:1, etc.)
    if expected_device == 'cuda':
        return tensor.is_cuda
    else:
        return str(tensor.device) == expected_device


def check_tensor_dtype(tensor: torch.Tensor, expected_dtype: torch.dtype) -> bool:
    """Check tensor has expected dtype"""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Expected torch.Tensor")
    return tensor.dtype == expected_dtype


def ensure_numpy_array(data: Union[np.ndarray, torch.Tensor, list], 
                      dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Convert data to numpy array"""
    if data is None:
        raise TypeError("Cannot convert None to numpy array")
    elif isinstance(data, np.ndarray):
        result = data
    elif isinstance(data, torch.Tensor):
        result = data.cpu().numpy()
    elif isinstance(data, list):
        # Check for complex numbers
        def has_complex(lst):
            for item in lst:
                if isinstance(item, complex):
                    return True
                if isinstance(item, list):
                    if has_complex(item):
                        return True
            return False
        
        if has_complex(data):
            raise TypeError("Cannot convert complex numbers to numpy array")
        result = np.array(data)
    else:
        raise TypeError(f"Cannot convert {type(data)} to numpy array")
    
    # Convert dtype if specified
    if dtype is not None and result.dtype != dtype:
        result = result.astype(dtype)
    
    return result


def ensure_torch_tensor(data: Union[np.ndarray, torch.Tensor, list], 
                       device: Optional[Union[str, torch.device]] = None,
                       dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Convert data to torch tensor"""
    if isinstance(data, torch.Tensor):
        result = data
    elif isinstance(data, np.ndarray):
        result = torch.from_numpy(data)
    elif isinstance(data, list):
        result = torch.tensor(data)
    else:
        raise TypeError(f"Cannot convert {type(data)} to torch tensor")
    
    # Convert dtype if specified
    if dtype is not None and result.dtype != dtype:
        result = result.to(dtype)
    
    # Move to device if specified
    if device is not None:
        result = result.to(device)
    
    return result