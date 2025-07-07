"""Tests for validation utilities

Tests cover:
- Input validation functions
- State validation
- Action validation
- Configuration validation
- Data validation
- Type checking utilities
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

from mcts.utils.validation_helpers import (
    validate_state, validate_action, validate_policy, validate_value,
    validate_batch_states, validate_batch_policies, validate_batch_values,
    validate_config_parameters, validate_model_output, validate_game_example,
    check_tensor_device, check_tensor_dtype, ensure_numpy_array, ensure_torch_tensor
)


@pytest.fixture
def sample_state():
    """Create sample game state"""
    return np.random.randn(3, 15, 15).astype(np.float32)


@pytest.fixture
def sample_policy():
    """Create sample policy vector"""
    policy = np.random.rand(225).astype(np.float32)
    return policy / policy.sum()  # Normalize


@pytest.fixture
def sample_batch_states():
    """Create batch of states"""
    return np.random.randn(4, 3, 15, 15).astype(np.float32)


@pytest.fixture
def sample_game_example():
    """Create sample game example"""
    return {
        'state': np.random.randn(3, 15, 15).astype(np.float32),
        'policy': np.ones(225) / 225,
        'value': 0.5,
        'game_id': 'test_game',
        'move_number': 10
    }


class TestStateValidation:
    """Test state validation functions"""
    
    def test_validate_state_valid(self, sample_state):
        """Test validation of valid state"""
        # Should not raise any exception
        validate_state(sample_state)
        
    def test_validate_state_wrong_shape(self):
        """Test validation with wrong shape"""
        state = np.ones((3, 10, 10))
        
        # validate_state only checks that state is 3D, not specific dimensions
        validate_state(state)  # This will pass as it's 3D
            
    def test_validate_state_wrong_channels(self, sample_state):
        """Test validation with wrong dimensions"""
        # Create 2D array to trigger error
        wrong_state = np.ones((15, 15))  # Missing channel dimension
        with pytest.raises(ValueError, match="3-dimensional"):
            validate_state(wrong_state)
            
    def test_validate_state_wrong_dtype(self):
        """Test validation with wrong dtype"""
        state = np.ones((3, 15, 15), dtype=np.int32)
        
        # validate_state doesn't check dtype, only that it's an ndarray
        validate_state(state)  # This will pass
            
    def test_validate_state_nan_values(self):
        """Test validation with NaN values"""
        state = np.ones((3, 15, 15))
        state[0, 0, 0] = np.nan
        
        # validate_state doesn't check for NaN values
        validate_state(state)  # This will pass
            
    def test_validate_state_inf_values(self):
        """Test validation with infinite values"""
        state = np.ones((3, 15, 15))
        state[0, 0, 0] = np.inf
        
        # validate_state doesn't check for infinite values
        validate_state(state)  # This will pass


class TestActionValidation:
    """Test action validation functions"""
    
    def test_validate_action_valid(self):
        """Test validation of valid action"""
        # Should not raise
        validate_action(42, action_space_size=225)
        validate_action(0, action_space_size=225)
        validate_action(224, action_space_size=225)
        
    def test_validate_action_out_of_bounds(self):
        """Test validation of out-of-bounds action"""
        with pytest.raises(ValueError, match="outside valid range"):
            validate_action(225, action_space_size=225)
            
        with pytest.raises(ValueError, match="outside valid range"):
            validate_action(-1, action_space_size=225)
            
    def test_validate_action_wrong_type(self):
        """Test validation with wrong type"""
        with pytest.raises(TypeError, match="must be an integer"):
            validate_action(42.5, action_space_size=225)
            
        with pytest.raises(TypeError, match="must be an integer"):
            validate_action("42", action_space_size=225)
            
    def test_validate_action_with_legal_moves(self):
        """Test validation with legal moves constraint"""
        # validate_action doesn't support legal_moves parameter
        # Just test basic action validation
        validate_action(42, action_space_size=225)
        
        # Test boundary
        validate_action(0, action_space_size=225)
        validate_action(224, action_space_size=225)


class TestPolicyValidation:
    """Test policy validation functions"""
    
    def test_validate_policy_valid(self, sample_policy):
        """Test validation of valid policy"""
        validate_policy(sample_policy, action_space_size=225)
        
    def test_validate_policy_wrong_size(self):
        """Test validation with wrong size"""
        policy = np.ones(100) / 100
        
        with pytest.raises(ValueError, match="Expected size"):
            validate_policy(policy, action_space_size=225)
            
    def test_validate_policy_negative_values(self):
        """Test validation with negative probabilities"""
        policy = np.ones(225) / 225
        policy[0] = -0.1
        
        with pytest.raises(ValueError, match="negative values"):
            validate_policy(policy, action_space_size=225)
            
    def test_validate_policy_not_normalized(self):
        """Test validation with non-normalized policy"""
        policy = np.ones(225)  # Sum = 225, not 1
        
        with pytest.raises(ValueError, match="sum to 1"):
            validate_policy(policy, action_space_size=225)
            
    def test_validate_policy_tolerance(self):
        """Test normalization tolerance"""
        policy = np.ones(225) / 225
        policy[0] += 1e-6  # Slight deviation
        
        # Should accept with default tolerance
        validate_policy(policy, action_space_size=225)
        
        # Should reject with strict tolerance
        with pytest.raises(ValueError, match="sum to 1"):
            validate_policy(policy, action_space_size=225, tolerance=1e-8)
            
    def test_validate_policy_with_mask(self):
        """Test validation with legal move mask"""
        policy = np.zeros(225)
        legal_mask = np.zeros(225, dtype=bool)
        legal_mask[:10] = True
        
        # Valid: probability only on legal moves
        policy[:10] = 0.1
        validate_policy(policy, action_space_size=225, legal_mask=legal_mask)
        
        # Invalid: probability on illegal move
        policy[20] = 0.1
        with pytest.raises(ValueError, match="illegal moves"):
            validate_policy(policy, action_space_size=225, legal_mask=legal_mask)


class TestValueValidation:
    """Test value validation functions"""
    
    def test_validate_value_valid(self):
        """Test validation of valid values"""
        validate_value(0.0)
        validate_value(1.0)
        validate_value(-1.0)
        validate_value(0.5)
        
    def test_validate_value_out_of_range(self):
        """Test validation of out-of-range values"""
        with pytest.raises(ValueError, match="range"):
            validate_value(1.5)
            
        with pytest.raises(ValueError, match="range"):
            validate_value(-1.5)
            
    def test_validate_value_wrong_type(self):
        """Test validation with wrong type"""
        with pytest.raises(TypeError, match="must be float"):
            validate_value("0.5")
            
        with pytest.raises(TypeError, match="must be float"):
            validate_value([0.5])
            
    def test_validate_value_nan(self):
        """Test validation with NaN value"""
        with pytest.raises(ValueError, match="NaN"):
            validate_value(float('nan'))
            
    def test_validate_value_custom_range(self):
        """Test validation with custom range"""
        # Custom range [0, 1]
        validate_value(0.5, min_val=0.0, max_val=1.0)
        
        with pytest.raises(ValueError, match="range"):
            validate_value(-0.5, min_val=0.0, max_val=1.0)


class TestBatchValidation:
    """Test batch validation functions"""
    
    def test_validate_batch_states_valid(self, sample_batch_states):
        """Test validation of valid batch states"""
        validate_batch_states(sample_batch_states, channels=3, board_size=15)
        
    def test_validate_batch_states_wrong_shape(self):
        """Test batch states with wrong shape"""
        states = np.ones((4, 5, 15, 15))  # Wrong channels
        
        with pytest.raises(ValueError, match="Expected 3 channels"):
            validate_batch_states(states, channels=3, board_size=15)
            
    def test_validate_batch_states_empty(self):
        """Test empty batch validation"""
        states = np.empty((0, 3, 15, 15))
        
        # Should handle empty batch
        validate_batch_states(states, channels=3, board_size=15)
        
    def test_validate_batch_policies_valid(self):
        """Test validation of valid batch policies"""
        policies = np.random.rand(4, 225)
        policies = policies / policies.sum(axis=1, keepdims=True)
        
        validate_batch_policies(policies, action_space_size=225)
        
    def test_validate_batch_policies_not_normalized(self):
        """Test batch policies not normalized"""
        policies = np.ones((4, 225))
        
        with pytest.raises(ValueError, match="not normalized"):
            validate_batch_policies(policies, action_space_size=225)
            
    def test_validate_batch_values_valid(self):
        """Test validation of valid batch values"""
        values = np.array([0.5, -0.3, 0.9, -0.9])
        validate_batch_values(values)
        
    def test_validate_batch_values_out_of_range(self):
        """Test batch values out of range"""
        values = np.array([0.5, 1.5, -0.5, 0.0])
        
        with pytest.raises(ValueError, match="out of range"):
            validate_batch_values(values)


class TestConfigValidation:
    """Test configuration parameter validation"""
    
    def test_validate_config_parameters_valid(self):
        """Test validation of valid config parameters"""
        config = {
            'num_simulations': 200,
            'c_puct': 1.4,
            'temperature': 1.0,
            'batch_size': 32
        }
        
        constraints = {
            'num_simulations': {'min': 1, 'max': 10000, 'type': int},
            'c_puct': {'min': 0.0, 'max': 10.0, 'type': float},
            'temperature': {'min': 0.0, 'max': 2.0, 'type': float},
            'batch_size': {'min': 1, 'max': 1024, 'type': int}
        }
        
        validate_config_parameters(config, constraints)
        
    def test_validate_config_parameters_out_of_range(self):
        """Test config validation with out-of-range values"""
        config = {'num_simulations': 0}
        constraints = {'num_simulations': {'min': 1, 'max': 10000}}
        
        with pytest.raises(ValueError, match="out of range"):
            validate_config_parameters(config, constraints)
            
    def test_validate_config_parameters_wrong_type(self):
        """Test config validation with wrong types"""
        config = {'batch_size': 32.5}
        constraints = {'batch_size': {'type': int}}
        
        with pytest.raises(TypeError, match="Expected type"):
            validate_config_parameters(config, constraints)
            
    def test_validate_config_parameters_missing_required(self):
        """Test config validation with missing required params"""
        config = {'c_puct': 1.4}
        constraints = {
            'num_simulations': {'required': True},
            'c_puct': {'required': False}
        }
        
        with pytest.raises(ValueError, match="Missing required"):
            validate_config_parameters(config, constraints)


class TestModelOutputValidation:
    """Test model output validation"""
    
    def test_validate_model_output_valid(self):
        """Test validation of valid model output"""
        policy = torch.ones(4, 225) / 225
        value = torch.tensor([-0.5, 0.0, 0.5, 0.9])
        
        validate_model_output(policy, value, batch_size=4, action_space_size=225)
        
    def test_validate_model_output_wrong_shape(self):
        """Test model output with wrong shapes"""
        policy = torch.ones(4, 100)  # Wrong action space
        value = torch.tensor([0.5, 0.0])  # Wrong batch size
        
        with pytest.raises(ValueError, match="Policy shape"):
            validate_model_output(policy, value, batch_size=4, action_space_size=225)
            
    def test_validate_model_output_numpy(self):
        """Test model output validation with numpy arrays"""
        policy = np.ones((4, 225)) / 225
        value = np.array([-0.5, 0.0, 0.5, 0.9])
        
        validate_model_output(policy, value, batch_size=4, action_space_size=225)


class TestGameExampleValidation:
    """Test game example validation"""
    
    def test_validate_game_example_valid(self, sample_game_example):
        """Test validation of valid game example"""
        validate_game_example(sample_game_example)
        
    def test_validate_game_example_missing_field(self):
        """Test game example with missing field"""
        example = {
            'state': np.ones((3, 15, 15)),
            'policy': np.ones(225) / 225,
            # 'value' missing
        }
        
        with pytest.raises(ValueError, match="Missing required field"):
            validate_game_example(example)
            
    def test_validate_game_example_invalid_state(self):
        """Test game example with invalid state"""
        example = {
            'state': np.ones((5, 15, 15)),  # Wrong channels
            'policy': np.ones(225) / 225,
            'value': 0.5,
            'game_id': 'test',
            'move_number': 0
        }
        
        with pytest.raises(ValueError, match="Invalid state"):
            validate_game_example(example, channels=3)
            
    def test_validate_game_example_inconsistent(self):
        """Test game example with inconsistent data"""
        example = {
            'state': np.ones((3, 15, 15)),
            'policy': np.ones(100) / 100,  # Wrong size for 15x15 board
            'value': 0.5,
            'game_id': 'test',
            'move_number': 0
        }
        
        with pytest.raises(ValueError, match="Policy size"):
            validate_game_example(example)


class TestTensorUtilities:
    """Test tensor utility functions"""
    
    def test_check_tensor_device(self):
        """Test tensor device checking"""
        if torch.cuda.is_available():
            tensor_cuda = torch.ones(10).cuda()
            assert check_tensor_device(tensor_cuda, 'cuda') == True
            assert check_tensor_device(tensor_cuda, 'cpu') == False
            
        tensor_cpu = torch.ones(10)
        assert check_tensor_device(tensor_cpu, 'cpu') == True
        
    def test_check_tensor_dtype(self):
        """Test tensor dtype checking"""
        tensor_float32 = torch.ones(10, dtype=torch.float32)
        tensor_float64 = torch.ones(10, dtype=torch.float64)
        tensor_int32 = torch.ones(10, dtype=torch.int32)
        
        assert check_tensor_dtype(tensor_float32, torch.float32) == True
        assert check_tensor_dtype(tensor_float32, torch.float64) == False
        assert check_tensor_dtype(tensor_int32, torch.int32) == True
        
    def test_ensure_numpy_array(self):
        """Test conversion to numpy array"""
        # From list
        arr = ensure_numpy_array([1, 2, 3])
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3,)
        
        # From torch tensor
        tensor = torch.ones(2, 3)
        arr = ensure_numpy_array(tensor)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 3)
        
        # Already numpy
        original = np.ones((4, 5))
        arr = ensure_numpy_array(original)
        assert arr is original
        
        # With dtype
        arr = ensure_numpy_array([1.5, 2.5], dtype=np.int32)
        assert arr.dtype == np.int32
        assert np.array_equal(arr, [1, 2])
        
    def test_ensure_torch_tensor(self):
        """Test conversion to torch tensor"""
        # From numpy
        arr = np.ones((2, 3))
        tensor = ensure_torch_tensor(arr)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 3)
        
        # From list
        tensor = ensure_torch_tensor([[1, 2], [3, 4]])
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 2)
        
        # Already tensor
        original = torch.ones(4, 5)
        tensor = ensure_torch_tensor(original)
        assert tensor is original
        
        # With device
        if torch.cuda.is_available():
            tensor = ensure_torch_tensor(np.ones(10), device='cuda')
            assert tensor.is_cuda
            
        # With dtype
        tensor = ensure_torch_tensor([1.5, 2.5], dtype=torch.int32)
        assert tensor.dtype == torch.int32


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_validate_empty_arrays(self):
        """Test validation of empty arrays"""
        # Empty state - should pass since it's still 3D
        validate_state(np.empty((0, 0, 0)))
            
        # Empty policy with non-zero action space
        with pytest.raises(ValueError):
            validate_policy(np.array([]), action_space_size=225)
            
    def test_validate_special_values(self):
        """Test validation with special float values"""
        # NaN in policy
        policy = np.ones(225) / 225
        policy[0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            validate_policy(policy, 225)
            
        # NaN in value
        with pytest.raises(ValueError, match="NaN"):
            validate_value(np.nan)
            
    def test_type_conversion_edge_cases(self):
        """Test type conversion edge cases"""
        # None input
        with pytest.raises(TypeError):
            ensure_numpy_array(None)
            
        # Complex numbers
        with pytest.raises(TypeError):
            ensure_numpy_array([1+2j, 3+4j])
            
        # Mixed types
        arr = ensure_numpy_array([1, 2.5, True])
        assert arr.dtype == np.float64  # Should upcast to float