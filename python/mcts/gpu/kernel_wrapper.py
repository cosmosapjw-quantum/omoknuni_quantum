"""Wrapper for CUDA kernels to provide a consistent interface

This module handles the differences between kernels loaded as modules
vs kernels loaded via torch.ops.
"""

import torch
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CUDAKernelWrapper:
    """Wrapper that provides a consistent interface for CUDA kernels
    
    Handles both module-style kernels and torch.ops style kernels.
    """
    
    def __init__(self, kernel_module: Any):
        """Initialize wrapper with loaded kernel module
        
        Args:
            kernel_module: Either a Python module with kernel functions
                          or a torch.ops namespace
        """
        self._module = kernel_module
        self._is_torch_ops = self._detect_torch_ops()
        self._available_kernels = self._discover_kernels()
        
        if self._available_kernels:
            logger.debug(f"Kernel wrapper initialized with functions: {list(self._available_kernels.keys())}")
        else:
            logger.warning("No kernel functions found in module")
    
    def _detect_torch_ops(self) -> bool:
        """Detect if this is a torch.ops style module"""
        module_type = str(type(self._module))
        return (
            module_type.startswith("<class 'torch._ops") or
            hasattr(self._module, 'default') or
            hasattr(self._module, '_qualified_op_name')
        )
    
    def _discover_kernels(self) -> dict:
        """Discover available kernel functions"""
        kernels = {}
        
        if self._is_torch_ops:
            # For torch.ops modules, the kernels are registered but may not show up in dir()
            # We'll check for known kernel names
            known_kernels = [
                'batched_ucb_selection', 'batched_ucb_selection_quantum',
                'parallel_backup', 'quantum_interference', 'batched_add_children',
                'evaluate_gomoku_positions', 'find_expansion_nodes',
                'batch_process_legal_moves', 'fused_minhash_interference',
                'phase_kicked_policy', 'batch_apply_moves', 'generate_legal_moves_mask'
            ]
            
            for kernel_name in known_kernels:
                try:
                    if hasattr(self._module, kernel_name):
                        attr = getattr(self._module, kernel_name)
                        kernels[kernel_name] = attr
                except:
                    pass
                    
            # Also check dir() for any additional ops
            for attr_name in dir(self._module):
                if not attr_name.startswith('_') and attr_name not in kernels:
                    try:
                        attr = getattr(self._module, attr_name)
                        if callable(attr) or hasattr(attr, 'default'):
                            kernels[attr_name] = attr
                    except:
                        pass
        else:
            # Regular module - check for callable attributes
            for attr_name in dir(self._module):
                if not attr_name.startswith('_'):
                    attr = getattr(self._module, attr_name)
                    if callable(attr):
                        kernels[attr_name] = attr
        
        return kernels
    
    def __getattr__(self, name: str) -> Callable:
        """Get kernel function by name"""
        if name in self._available_kernels:
            kernel = self._available_kernels[name]
            
            # For torch.ops, we might need to use .default
            if self._is_torch_ops and hasattr(kernel, 'default'):
                return kernel.default
            else:
                return kernel
        
        # Check if it's a torch.ops style access (e.g., module.kernel_name.default)
        if self._is_torch_ops and hasattr(self._module, name):
            return getattr(self._module, name)
        
        raise AttributeError(f"Kernel function '{name}' not found. Available: {list(self._available_kernels.keys())}")
    
    def has_kernel(self, name: str) -> bool:
        """Check if a kernel function exists"""
        return name in self._available_kernels or (self._is_torch_ops and hasattr(self._module, name))
    
    @property
    def available_kernels(self) -> list:
        """Get list of available kernel names"""
        return list(self._available_kernels.keys())


def wrap_kernel_module(module: Any) -> Optional[CUDAKernelWrapper]:
    """Wrap a kernel module to provide consistent interface
    
    Args:
        module: Kernel module to wrap
        
    Returns:
        CUDAKernelWrapper or None if module is invalid
    """
    if module is None:
        return None
    
    try:
        return CUDAKernelWrapper(module)
    except Exception as e:
        logger.error(f"Failed to wrap kernel module: {e}")
        return None