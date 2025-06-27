"""Utilities for safe autocast usage that avoid CUDA warnings in CPU-only environments"""

import torch
import contextlib
from typing import Optional, Union


def safe_autocast(device: Optional[Union[str, torch.device]] = None, 
                  enabled: bool = True,
                  dtype: Optional[torch.dtype] = None):
    """Context manager for safe autocast that avoids warnings when CUDA is not available
    
    Args:
        device: Device to use for autocast. If None, uses current default device.
                Can be 'cuda', 'cpu', or torch.device object.
        enabled: Whether to enable autocast
        dtype: Data type for autocast (e.g., torch.float16)
    
    Returns:
        Context manager that handles autocast appropriately for the device
    """
    if not enabled:
        return contextlib.nullcontext()
    
    # Convert device to torch.device if needed
    if device is None:
        device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    elif isinstance(device, str):
        device = torch.device(device)
    elif hasattr(device, 'type'):
        # It's already a torch.device
        pass
    else:
        device = torch.device('cpu')
    
    # Only use autocast for CUDA devices
    if device.type == 'cuda' and torch.cuda.is_available():
        return torch.amp.autocast(device_type='cuda', enabled=enabled, dtype=dtype)
    else:
        # For CPU or when CUDA is not available, return null context
        return contextlib.nullcontext()


def get_device_type(device: Optional[Union[str, torch.device]] = None) -> str:
    """Get the device type string suitable for autocast
    
    Args:
        device: Device specification
        
    Returns:
        'cuda' or 'cpu' based on device and availability
    """
    if device is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    elif isinstance(device, str):
        if device == 'cuda' and not torch.cuda.is_available():
            return 'cpu'
        return device
    elif hasattr(device, 'type'):
        if device.type == 'cuda' and not torch.cuda.is_available():
            return 'cpu'
        return device.type
    else:
        return 'cpu'