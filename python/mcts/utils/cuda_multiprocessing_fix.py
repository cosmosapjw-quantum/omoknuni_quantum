"""CUDA Multiprocessing Fix for Phase 2.1 Optimization

This module provides comprehensive fixes for CUDA multiprocessing visibility warnings
by ensuring proper CUDA context isolation between main process and worker processes.
"""

import os
import sys
import multiprocessing as mp
import logging
import functools
import threading
from typing import Callable, Any, Dict, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class CUDAMultiprocessingManager:
    """Manager for CUDA multiprocessing context isolation
    
    This class ensures that CUDA is properly isolated between the main process
    (which runs the GPU evaluator service) and worker processes (which should
    only use CPU operations).
    """
    
    def __init__(self):
        self.original_environ = {}
        self.cuda_disabled_in_workers = False
        self._lock = threading.Lock()
    
    def setup_main_process_cuda(self):
        """Setup CUDA for main process (GPU evaluator service)"""
        with self._lock:
            # Ensure CUDA is available for main process
            if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '':
                # Restore CUDA visibility for main process
                if 'ORIGINAL_CUDA_VISIBLE_DEVICES' in os.environ:
                    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['ORIGINAL_CUDA_VISIBLE_DEVICES']
                else:
                    # Remove the restriction entirely
                    del os.environ['CUDA_VISIBLE_DEVICES']
            
            logger.debug("CUDA enabled for main process (GPU evaluator service)")
    
    def create_worker_safe_environment(self) -> Dict[str, str]:
        """Create environment variables for worker processes with CUDA disabled"""
        worker_env = os.environ.copy()
        
        # Store original CUDA_VISIBLE_DEVICES if not already stored
        if 'CUDA_VISIBLE_DEVICES' in worker_env and 'ORIGINAL_CUDA_VISIBLE_DEVICES' not in worker_env:
            worker_env['ORIGINAL_CUDA_VISIBLE_DEVICES'] = worker_env['CUDA_VISIBLE_DEVICES']
        
        # Aggressively disable CUDA for workers
        worker_env.update({
            'CUDA_VISIBLE_DEVICES': '',
            'CUDA_LAUNCH_BLOCKING': '0',
            'CUDA_CACHE_DISABLE': '1',
            'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
            'CUDA_CACHE_PATH': '/tmp/cuda_cache_disabled',
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:0',  # Disable CUDA memory allocation
            'TORCH_CUDA_ARCH_LIST': '',  # Disable CUDA architecture detection
        })
        
        return worker_env
    
    def wrap_worker_function(self, func: Callable) -> Callable:
        """Wrap worker function to ensure CUDA is disabled before any imports"""
        
        @functools.wraps(func)
        def worker_wrapper(*args, **kwargs):
            # This runs in the worker process
            
            # First, aggressively set environment variables
            worker_env = self.create_worker_safe_environment()
            os.environ.update(worker_env)
            
            # Import our worker initialization
            from .worker_init import init_worker_process, verify_cuda_disabled
            
            # Initialize worker process
            init_worker_process()
            
            # Verify CUDA is disabled
            cuda_properly_disabled = verify_cuda_disabled()
            
            # Import torch AFTER environment setup
            import torch
            
            # Double-check CUDA status
            if torch.cuda.is_available():
                logger.warning(f"[WORKER {os.getpid()}] CUDA still available despite environment setup")
                # Force disable CUDA functions
                torch.cuda.is_available = lambda: False
                torch.cuda.device_count = lambda: 0
                torch.cuda.get_device_name = lambda x=None: "CUDA_DISABLED"
                torch.cuda.current_device = lambda: None
                torch.cuda.set_device = lambda x: None
                torch.cuda.synchronize = lambda device=None: None
                torch.cuda.empty_cache = lambda: None
            else:
                logger.debug(f"[WORKER {os.getpid()}] CUDA properly disabled")
            
            # Execute original function
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"[WORKER {os.getpid()}] Error in worker function: {e}")
                raise
        
        return worker_wrapper


# Global instance
_cuda_mp_manager: Optional[CUDAMultiprocessingManager] = None
_manager_lock = threading.Lock()


def get_cuda_multiprocessing_manager() -> CUDAMultiprocessingManager:
    """Get or create the global CUDA multiprocessing manager"""
    global _cuda_mp_manager
    
    with _manager_lock:
        if _cuda_mp_manager is None:
            _cuda_mp_manager = CUDAMultiprocessingManager()
        return _cuda_mp_manager


@contextmanager
def cuda_multiprocessing_context():
    """Context manager for safe CUDA multiprocessing setup"""
    manager = get_cuda_multiprocessing_manager()
    
    # Setup main process for CUDA
    manager.setup_main_process_cuda()
    
    try:
        yield manager
    finally:
        # Cleanup if needed
        pass


def create_safe_worker_pool(worker_function: Callable, 
                           num_workers: int,
                           worker_args: tuple = (),
                           worker_kwargs: dict = None,
                           start_method: str = 'spawn') -> mp.Pool:
    """Create a multiprocessing pool with CUDA-safe worker processes
    
    Args:
        worker_function: Function to run in each worker
        num_workers: Number of worker processes
        worker_args: Arguments to pass to worker function
        worker_kwargs: Keyword arguments to pass to worker function
        start_method: Multiprocessing start method
    
    Returns:
        mp.Pool with CUDA-safe worker processes
    """
    if worker_kwargs is None:
        worker_kwargs = {}
    
    # Set multiprocessing start method
    try:
        mp.set_start_method(start_method, force=True)
        logger.debug(f"Set multiprocessing start method to '{start_method}'")
    except RuntimeError as e:
        logger.debug(f"Multiprocessing start method already set: {e}")
    
    # Get manager and wrap worker function
    manager = get_cuda_multiprocessing_manager()
    safe_worker_function = manager.wrap_worker_function(worker_function)
    
    # Create worker environment
    worker_env = manager.create_worker_safe_environment()
    
    # Create process pool with custom initialization
    def worker_init():
        """Initialize worker process with CUDA disabled"""
        os.environ.update(worker_env)
        from .worker_init import init_worker_process
        init_worker_process()
    
    try:
        pool = mp.Pool(
            processes=num_workers,
            initializer=worker_init
        )
        logger.info(f"Created CUDA-safe worker pool with {num_workers} processes")
        return pool
    except Exception as e:
        logger.error(f"Failed to create worker pool: {e}")
        raise


def fix_existing_worker_process():
    """Fix CUDA visibility in already running worker process
    
    This function should be called at the beginning of worker functions
    that are already running to ensure CUDA is properly disabled.
    """
    worker_pid = os.getpid()
    
    # Check if we're in a worker process (not main process)
    try:
        import psutil
        parent_pid = os.getppid()
        current_process = psutil.Process(worker_pid)
        parent_process = psutil.Process(parent_pid)
        
        # If parent has different command line, we're likely in a worker
        if current_process.cmdline() != parent_process.cmdline():
            is_worker = True
        else:
            is_worker = False
    except:
        # Fallback: assume we're in worker if CUDA should be disabled
        is_worker = os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    
    if is_worker:
        # Apply worker-specific CUDA fixes
        manager = get_cuda_multiprocessing_manager()
        worker_env = manager.create_worker_safe_environment()
        os.environ.update(worker_env)
        
        # Import and apply worker initialization
        from .worker_init import init_worker_process, verify_cuda_disabled
        init_worker_process()
        
        # Force disable CUDA in PyTorch if already imported
        try:
            import torch
            if torch.cuda.is_available():
                logger.debug(f"[WORKER {worker_pid}] Fixing CUDA availability in running worker")
                
                # Override CUDA functions
                torch.cuda.is_available = lambda: False
                torch.cuda.device_count = lambda: 0
                torch.cuda.current_device = lambda: None
                torch.cuda.set_device = lambda x: None
                torch.cuda.synchronize = lambda device=None: None
                torch.cuda.empty_cache = lambda: None
                
                logger.debug(f"[WORKER {worker_pid}] CUDA disabled successfully")
            else:
                logger.debug(f"[WORKER {worker_pid}] CUDA already properly disabled")
        except ImportError:
            # PyTorch not imported yet, which is good
            logger.debug(f"[WORKER {worker_pid}] PyTorch not yet imported - CUDA will be disabled on import")


class CUDAWorkerProcess(mp.Process):
    """Custom Process class with automatic CUDA disabling for workers"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original_target = self._target
        self._target = self._wrap_target_with_cuda_fix
    
    def _wrap_target_with_cuda_fix(self):
        """Wrap target function with CUDA fix"""
        # Apply CUDA fix before running target
        fix_existing_worker_process()
        
        # Run original target
        if self._original_target:
            return self._original_target(*self._args, **self._kwargs)


def patch_multiprocessing_for_cuda_safety():
    """Patch multiprocessing module to use CUDA-safe processes by default"""
    
    # Store original Process class
    if not hasattr(mp, '_original_Process'):
        mp._original_Process = mp.Process
    
    # Replace with CUDA-safe version
    mp.Process = CUDAWorkerProcess
    
    logger.info("Patched multiprocessing.Process with CUDA safety")


def restore_original_multiprocessing():
    """Restore original multiprocessing behavior"""
    
    if hasattr(mp, '_original_Process'):
        mp.Process = mp._original_Process
        delattr(mp, '_original_Process')
        logger.info("Restored original multiprocessing.Process")