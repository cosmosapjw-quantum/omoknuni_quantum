"""Worker process initialization for CUDA-safe multiprocessing

This module provides initialization functions for worker processes to ensure
they don't interfere with CUDA operations in the main process.
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)


def init_worker_process():
    """Initialize worker process to disable CUDA before torch import
    
    This must be called at the very beginning of worker processes,
    before importing torch or any module that imports torch.
    
    Note: CUDA_VISIBLE_DEVICES should already be set to '' before calling this.
    """
    # Aggressively disable CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['CUDA_CACHE_DISABLE'] = '1'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_CACHE_PATH'] = '/tmp/cuda_cache_disabled'
    
    # Log initialization only in debug mode
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Worker process {os.getpid()} initialized with CUDA disabled")


def verify_cuda_disabled():
    """Verify that CUDA is properly disabled in this process"""
    try:
        import torch
        if torch.cuda.is_available():
            # Force disable CUDA functions comprehensively
            torch.cuda.is_available = lambda: False
            torch.cuda.device_count = lambda: 0
            torch.cuda.get_device_name = lambda x=None: "CUDA_DISABLED"
            torch.cuda.current_device = lambda: None
            torch.cuda.set_device = lambda x: None
            torch.cuda.synchronize = lambda device=None: None
            torch.cuda.empty_cache = lambda: None
            torch.cuda.memory_allocated = lambda device=None: 0
            torch.cuda.memory_reserved = lambda device=None: 0
            torch.cuda.max_memory_allocated = lambda device=None: 0
            torch.cuda.max_memory_reserved = lambda device=None: 0
            logger.debug("Comprehensive PyTorch CUDA disable applied in worker")
            return False
        return True
    except ImportError:
        return True


def get_cpu_device():
    """Get CPU device for worker processes"""
    import torch
    return torch.device('cpu')


def ensure_cpu_tensor(tensor):
    """Ensure a tensor is on CPU"""
    import torch
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu()
    return tensor


def create_cpu_only_config(config):
    """Create a CPU-only version of MCTS config for workers"""
    if hasattr(config, 'device'):
        config.device = 'cpu'
    if hasattr(config, 'use_mixed_precision'):
        config.use_mixed_precision = False
    if hasattr(config, 'use_cuda_graphs'):
        config.use_cuda_graphs = False
    if hasattr(config, 'use_tensor_cores'):
        config.use_tensor_cores = False
    return config