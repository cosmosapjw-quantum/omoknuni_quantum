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
    # Ensure CUDA is disabled (should already be set by worker function)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        logger.warning("CUDA_VISIBLE_DEVICES not set to '' before init_worker_process!")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Also set other CUDA-related environment variables
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    # Log initialization only in debug mode
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Worker process {os.getpid()} initialized with CUDA disabled")


def verify_cuda_disabled():
    """Verify that CUDA is properly disabled in this process"""
    try:
        import torch
        if torch.cuda.is_available():
            # Don't print warning - this is expected before init_worker_process is called
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