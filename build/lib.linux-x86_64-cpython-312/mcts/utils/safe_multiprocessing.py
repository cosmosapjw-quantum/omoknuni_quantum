"""Safe multiprocessing utilities for CUDA environments"""

import torch
import numpy as np
import copy
from typing import Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array, handling CUDA tensors safely"""
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.detach().numpy()


def numpy_to_tensor(array: np.ndarray, dtype=None) -> torch.Tensor:
    """Convert numpy array back to tensor"""
    if dtype is None:
        return torch.from_numpy(array)
    return torch.from_numpy(array).to(dtype)


def serialize_state_dict_for_multiprocessing(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert state dict to a format safe for multiprocessing.
    Converts all tensors to numpy arrays to avoid CUDA references.
    """
    
    serialized = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            # Convert tensor to numpy array
            serialized[key] = {
                'type': 'tensor',
                'data': tensor_to_numpy(value),
                'dtype': str(value.dtype),
                'shape': tuple(value.shape)
            }
        else:
            # Non-tensor values
            serialized[key] = {
                'type': 'other',
                'data': copy.deepcopy(value)
            }
    
    return serialized


def deserialize_state_dict_from_multiprocessing(serialized: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deserialize state dict from multiprocessing-safe format.
    Converts numpy arrays back to tensors.
    """
    
    state_dict = {}
    for key, value_info in serialized.items():
        if value_info['type'] == 'tensor':
            # Convert numpy array back to tensor
            numpy_data = value_info['data']
            dtype_str = value_info['dtype']
            
            # Map string dtype back to torch dtype
            dtype_map = {
                'torch.float32': torch.float32,
                'torch.float64': torch.float64,
                'torch.float16': torch.float16,
                'torch.int32': torch.int32,
                'torch.int64': torch.int64,
                'torch.int8': torch.int8,
                'torch.uint8': torch.uint8,
                'torch.bool': torch.bool,
            }
            dtype = dtype_map.get(dtype_str, torch.float32)
            
            tensor = torch.from_numpy(numpy_data).to(dtype)
            state_dict[key] = tensor
        else:
            # Non-tensor values
            state_dict[key] = value_info['data']
    
    return state_dict


def make_config_multiprocessing_safe(config: Any) -> Any:
    """
    Make config object safe for multiprocessing by removing CUDA references.
    """
    import dataclasses
    
    if not dataclasses.is_dataclass(config):
        return config
    
    # Create a copy
    config_dict = dataclasses.asdict(config)
    
    def clean_cuda_references(obj):
        """Recursively clean CUDA references"""
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                if k == 'device' and v == 'cuda':
                    cleaned[k] = 'cpu'
                else:
                    cleaned[k] = clean_cuda_references(v)
            return cleaned
        elif isinstance(obj, list):
            return [clean_cuda_references(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(clean_cuda_references(item) for item in obj)
        else:
            return obj
    
    cleaned_dict = clean_cuda_references(config_dict)
    
    # Reconstruct the dataclass
    # Need to handle nested dataclasses
    def reconstruct_dataclass(cls, data):
        """Recursively reconstruct dataclasses from dict"""
        if not dataclasses.is_dataclass(cls):
            return data
            
        field_values = {}
        for field in dataclasses.fields(cls):
            field_name = field.name
            if field_name in data:
                field_value = data[field_name]
                # Check if this field is also a dataclass
                if dataclasses.is_dataclass(field.type):
                    field_value = reconstruct_dataclass(field.type, field_value)
                elif hasattr(field.type, '__origin__') and field.type.__origin__ is type:
                    # Handle Optional[dataclass] or similar
                    if isinstance(field_value, dict):
                        # Try to find the actual dataclass type
                        for arg in field.type.__args__:
                            if dataclasses.is_dataclass(arg):
                                field_value = reconstruct_dataclass(arg, field_value)
                                break
                field_values[field_name] = field_value
        
        return cls(**field_values)
    
    return reconstruct_dataclass(type(config), cleaned_dict)


def test_serialization():
    """Test serialization/deserialization"""
    print("=== Testing Safe Serialization ===")
    
    # Create test state dict
    state_dict = {
        'weight': torch.randn(10, 10),
        'bias': torch.randn(10),
        'running_mean': torch.zeros(10),
        'num_batches_tracked': torch.tensor(0)
    }
    
    # Test with CUDA tensors if available
    if torch.cuda.is_available():
        state_dict['cuda_weight'] = torch.randn(5, 5).cuda()
    
    print(f"Original state dict: {len(state_dict)} items")
    
    # Serialize
    serialized = serialize_state_dict_for_multiprocessing(state_dict)
    print(f"Serialized: {len(serialized)} items")
    
    # Check no tensors remain
    tensor_count = 0
    for k, v in serialized.items():
        if torch.is_tensor(v):
            tensor_count += 1
    print(f"Tensors in serialized dict: {tensor_count} (should be 0)")
    
    # Deserialize
    restored = deserialize_state_dict_from_multiprocessing(serialized)
    print(f"Restored: {len(restored)} items")
    
    # Verify
    for key in state_dict:
        original = state_dict[key]
        restored_tensor = restored[key]
        
        if torch.is_tensor(original):
            original_cpu = original.cpu()
            if torch.allclose(original_cpu, restored_tensor):
                print(f"✓ {key} matches")
            else:
                print(f"✗ {key} differs!")


if __name__ == "__main__":
    test_serialization()