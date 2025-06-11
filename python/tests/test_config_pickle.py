"""Test if config object can be pickled safely"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import torch
from mcts.utils.config_system import create_default_config


def test_config_pickle():
    """Test if config contains any CUDA tensors"""
    print("=== Testing Config Pickling ===")
    
    # Create config
    config = create_default_config(game_type="gomoku")
    config.mcts.device = 'cuda'
    
    print(f"Config type: {type(config)}")
    print(f"Config device setting: {config.mcts.device}")
    
    # Try to pickle config
    try:
        pickled = pickle.dumps(config)
        print(f"SUCCESS: Config can be pickled (size: {len(pickled)} bytes)")
        
        # Unpickle and check
        unpickled = pickle.loads(pickled)
        print(f"Unpickled device: {unpickled.mcts.device}")
    except Exception as e:
        print(f"FAILED to pickle config: {e}")
        
        # Check what's in the config
        print("\nInspecting config attributes:")
        for attr in dir(config):
            if not attr.startswith('_'):
                try:
                    value = getattr(config, attr)
                    print(f"  {attr}: {type(value)}")
                    
                    # Check if it's a tensor
                    if isinstance(value, torch.Tensor):
                        print(f"    WARNING: Found tensor! Device: {value.device}")
                    
                    # Check nested attributes
                    if hasattr(value, '__dict__'):
                        for nested_attr in dir(value):
                            if not nested_attr.startswith('_'):
                                try:
                                    nested_value = getattr(value, nested_attr)
                                    if isinstance(nested_value, torch.Tensor):
                                        print(f"    WARNING: Found nested tensor in {attr}.{nested_attr}! Device: {nested_value.device}")
                                except:
                                    pass
                except:
                    pass


def check_config_references():
    """Check if config contains any object references that might have CUDA"""
    from mcts.utils.config_system import AlphaZeroConfig
    import dataclasses
    
    print("\n=== Checking Config Structure ===")
    
    # Create config
    config = create_default_config(game_type="gomoku")
    
    # Get all fields
    fields = dataclasses.fields(config)
    print(f"Config has {len(fields)} fields")
    
    for field in fields:
        value = getattr(config, field.name)
        print(f"\n{field.name}: {type(value)}")
        
        # If it's another dataclass, check its fields
        if dataclasses.is_dataclass(value):
            sub_fields = dataclasses.fields(value)
            for sub_field in sub_fields:
                sub_value = getattr(value, sub_field.name)
                print(f"  {sub_field.name}: {type(sub_value)} = {sub_value}")


if __name__ == "__main__":
    test_config_pickle()
    check_config_references()