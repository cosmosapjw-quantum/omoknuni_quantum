#!/usr/bin/env python3
"""Memory optimization: Pre-allocate tensors to reduce allocation overhead"""

import torch
from typing import Dict, Tuple, Optional

class TensorPool:
    """Pre-allocated tensor pool to reduce memory allocation overhead"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.pools: Dict[Tuple[torch.dtype, Tuple[int, ...]], torch.Tensor] = {}
        self.allocated_count = 0
        self.reuse_count = 0
        
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, 
                   fill_value: Optional[float] = None) -> torch.Tensor:
        """Get a tensor from the pool or allocate a new one"""
        key = (dtype, shape)
        
        if key in self.pools:
            tensor = self.pools[key]
            self.reuse_count += 1
        else:
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
            self.pools[key] = tensor
            self.allocated_count += 1
        
        if fill_value is not None:
            tensor.fill_(fill_value)
        
        return tensor
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics"""
        return {
            'allocated': self.allocated_count,
            'reused': self.reuse_count,
            'pool_size': len(self.pools),
            'reuse_rate': self.reuse_count / max(1, self.allocated_count + self.reuse_count)
        }

# Global tensor pool
_tensor_pools: Dict[torch.device, TensorPool] = {}

def get_tensor_pool(device: torch.device) -> TensorPool:
    """Get or create tensor pool for device"""
    if device not in _tensor_pools:
        _tensor_pools[device] = TensorPool(device)
    return _tensor_pools[device]
