"""
CUDA Module Singleton

This module ensures that CUDA kernels are compiled only once and shared across 
all modules that need them, preventing compilation conflicts and hangs.
"""

import torch
import threading
import logging
import time
import os
import fcntl
import tempfile
import multiprocessing
import pickle
import json
from typing import Optional, Any

logger = logging.getLogger(__name__)

class CUDAModuleSingleton:
    """Singleton class to manage CUDA module compilation and loading
    
    This implementation is process-safe and ensures CUDA kernels are compiled
    only once across all processes.
    """
    
    _instance = None
    _lock = threading.Lock()
    _cuda_module = None
    _compilation_in_progress = False
    _compilation_complete = False
    _process_id = None  # Track which process compiled the module
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(CUDAModuleSingleton, cls).__new__(cls)
        return cls._instance
    
    def get_cuda_module(self) -> Optional[Any]:
        """Get the compiled CUDA module, compiling if necessary
        
        This method is process-safe and ensures compilation happens only once.
        """
        
        current_pid = os.getpid()
        
        # CRITICAL: In worker processes, don't attempt CUDA compilation at all
        # This prevents memory corruption from multiple compilation attempts
        if hasattr(multiprocessing.current_process(), '_identity') and multiprocessing.current_process()._identity:
            # This is a worker process - return None to force fallback
            logger.debug(f"Worker process {current_pid} - using fallback, no CUDA")
            return None
        
        # Check if we need to reload the module (different process)
        if self._compilation_complete and self._cuda_module is not None:
            if self._process_id == current_pid:
                return self._cuda_module
            else:
                # Different process - but this should only be main process now
                logger.debug(f"Process {current_pid} reloading pre-compiled CUDA module")
                self._cuda_module = self._load_compiled_module()
                self._process_id = current_pid
                return self._cuda_module
        
        # If compilation is in progress, wait for it to complete
        if self._compilation_in_progress:
            logger.debug("⏳ CUDA compilation in progress, waiting...")
            while self._compilation_in_progress:
                time.sleep(0.1)
            return self._cuda_module
        
        # Use a lock file to coordinate compilation across processes
        lock_file = os.path.join(tempfile.gettempdir(), 'cuda_singleton.lock')
        compiled_marker = os.path.join(tempfile.gettempdir(), 'cuda_singleton_compiled.marker')
        
        # First check if another process already compiled
        if os.path.exists(compiled_marker):
            try:
                with open(compiled_marker, 'r') as f:
                    marker_data = json.load(f)
                    if marker_data.get('success', False):
                        logger.debug(f"Process {current_pid} loading pre-compiled module from marker")
                        self._cuda_module = self._load_compiled_module()
                        self._compilation_complete = True
                        self._process_id = current_pid
                        return self._cuda_module
            except Exception as e:
                logger.debug(f"Failed to read marker file: {e}")
        
        with open(lock_file, 'w') as f:
            try:
                # Acquire exclusive file lock to prevent multi-process compilation
                # Use blocking lock with timeout instead of non-blocking
                # This prevents processes from repeatedly retrying and corrupting memory
                import signal
                
                class TimeoutException(Exception):
                    pass
                
                def timeout_handler(signum, frame):
                    raise TimeoutException()
                
                # Set alarm for 60 second timeout
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)
                
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Blocking lock
                    signal.alarm(0)  # Cancel alarm
                except TimeoutException:
                    signal.signal(signal.SIGALRM, old_handler)
                    logger.warning("⏰ Timeout waiting for CUDA compilation lock")
                    time.sleep(2)  # Wait before retry
                    return self.get_cuda_module()
                except Exception as e:
                    logger.error(f"Failed to acquire file lock: {e}")
                    return None
                finally:
                    signal.signal(signal.SIGALRM, old_handler)
                
                # Now we have the file lock, proceed with compilation
                with self._lock:
                    # Double-check after acquiring lock
                    if self._compilation_complete and self._cuda_module is not None:
                        return self._cuda_module
                    
                    if self._compilation_in_progress:
                        # Another thread started compilation while we were waiting
                        while self._compilation_in_progress:
                            time.sleep(0.1)
                        return self._cuda_module
                    
                    # Start compilation
                    self._compilation_in_progress = True
                    
                    try:
                        # Double-check marker file in case it was created while waiting for lock
                        if os.path.exists(compiled_marker):
                            try:
                                with open(compiled_marker, 'r') as mf:
                                    marker_data = json.load(mf)
                                    if marker_data.get('success', False):
                                        logger.debug(f"Process {current_pid} found marker after acquiring lock")
                                        self._cuda_module = self._load_compiled_module()
                                        self._compilation_complete = True
                                        self._process_id = current_pid
                                        return self._cuda_module
                            except:
                                pass
                        
                        logger.debug(f"Process {current_pid} starting CUDA compilation")
                        self._cuda_module = self._compile_cuda_module()
                        self._compilation_complete = True
                        self._process_id = current_pid
                        
                        # Write marker file for other processes
                        try:
                            import json
                            with open(compiled_marker, 'w') as mf:
                                json.dump({
                                    'success': True,
                                    'pid': current_pid,
                                    'timestamp': time.time(),
                                    'module_name': self._get_module_name()
                                }, mf)
                        except Exception as e:
                            logger.debug(f"Failed to write marker file: {e}")
                        
                        logger.debug("✅ CUDA module compilation completed")
                    except Exception as e:
                        logger.error(f"❌ CUDA module compilation failed: {e}")
                        self._cuda_module = None
                        self._compilation_complete = False
                        # Write failure marker
                        try:
                            import json
                            with open(compiled_marker, 'w') as mf:
                                json.dump({
                                    'success': False,
                                    'pid': current_pid,
                                    'error': str(e)
                                }, mf)
                        except:
                            pass
                    finally:
                        self._compilation_in_progress = False
                    
                    return self._cuda_module
            except Exception as e:
                logger.error(f"Error during CUDA compilation process: {e}")
                return None
    
    def _get_module_name(self) -> str:
        """Get deterministic module name based on source hash"""
        from pathlib import Path
        import hashlib
        
        cuda_source = Path(__file__).parent / 'unified_cuda_kernels.cu'
        if cuda_source.exists():
            with open(cuda_source, 'rb') as f:
                source_hash = hashlib.md5(f.read()).hexdigest()[:8]
            return f'mcts_cuda_singleton_{source_hash}'
        return 'mcts_cuda_singleton_default'
    
    def _load_compiled_module(self) -> Optional[Any]:
        """Load an already compiled CUDA module"""
        try:
            from torch.utils.cpp_extension import load
            from pathlib import Path
            
            module_name = self._get_module_name()
            cuda_source = Path(__file__).parent / 'unified_cuda_kernels.cu'
            
            # Load without recompiling - torch will use cached version
            module = load(
                name=module_name,
                sources=[str(cuda_source)],
                extra_cuda_cflags=['-O1', '--use_fast_math'],
                verbose=False,  # Quiet for loading
                with_cuda=True
            )
            
            return module
        except Exception as e:
            logger.error(f"Failed to load pre-compiled module: {e}")
            return None
    
    def _compile_cuda_module(self) -> Optional[Any]:
        """Compile the CUDA module"""
        
        logger.debug("🔨 Starting CUDA module compilation...")
        
        # Set optimal compilation environment
        os.environ.setdefault('MAX_JOBS', '1')
        os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.6')
        
        try:
            from torch.utils.cpp_extension import load
            from pathlib import Path
            import hashlib
            
            # Use the unified CUDA kernels file
            cuda_source = Path(__file__).parent / 'unified_cuda_kernels.cu'
            if not cuda_source.exists():
                raise FileNotFoundError(f"CUDA source not found: {cuda_source}")
            
            # Use consistent module name
            module_name = self._get_module_name()
            
            start_time = time.time()
            
            module = load(
                name=module_name,
                sources=[str(cuda_source)],
                extra_cuda_cflags=['-O1', '--use_fast_math'],
                verbose=True,
                with_cuda=True
            )
            
            compile_time = time.time() - start_time
            logger.debug(f"✅ CUDA module compiled in {compile_time:.2f}s")
            
            # Verify expected functions are available
            expected_functions = [
                'batched_ucb_selection', 'parallel_backup', 'vectorized_backup',
                'fused_selection_traversal', 'batched_add_children', 'quantum_interference'
            ]
            
            missing_functions = []
            for func_name in expected_functions:
                if not hasattr(module, func_name):
                    missing_functions.append(func_name)
            
            if missing_functions:
                logger.warning(f"⚠️ Missing functions: {missing_functions}")
            else:
                logger.debug(f"✅ All {len(expected_functions)} expected functions available")
            
            return module
            
        except Exception as e:
            logger.error(f"❌ CUDA compilation failed: {e}")
            raise
    
    def has_function(self, func_name: str) -> bool:
        """Check if a function is available in the CUDA module"""
        module = self.get_cuda_module()
        return module is not None and hasattr(module, func_name)
    
    def get_function(self, func_name: str) -> Optional[Any]:
        """Get a specific function from the CUDA module"""
        module = self.get_cuda_module()
        if module is not None and hasattr(module, func_name):
            return getattr(module, func_name)
        return None

# Global singleton instance
cuda_singleton = CUDAModuleSingleton()

# Convenience functions
def get_cuda_module():
    """Get the compiled CUDA module"""
    return cuda_singleton.get_cuda_module()

def get_cuda_function(func_name: str):
    """Get a specific CUDA function"""
    return cuda_singleton.get_function(func_name)

def has_cuda_function(func_name: str) -> bool:
    """Check if a CUDA function is available"""
    return cuda_singleton.has_function(func_name)