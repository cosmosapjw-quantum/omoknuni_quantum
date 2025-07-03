"""
Simplified CUDA Kernel Detection and Loading System

This module provides detection and loading of pre-compiled CUDA kernels.
CUDA compilation is now handled by setup.py using manual nvcc approach.

Provides a clean interface for CUDA kernel detection and management.
"""

import os
import torch
import logging
import importlib.util
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)



class CudaManager:
    """
    Simplified CUDA kernel detection and loading system
    
    Features:
    - Detection of pre-compiled CUDA kernels
    - Module loading and validation
    - Fallback to CPU implementations when kernels unavailable
    - Clean separation from compilation (handled by setup.py)
    """
    
    def __init__(self, force_rebuild: bool = False, disable_cuda: bool = False):
        """
        Initialize CUDA manager for detection and loading only
        
        Args:
            force_rebuild: Ignored (compilation now handled by setup.py)
            disable_cuda: Disable CUDA detection entirely
        """
        self.current_dir = Path(__file__).parent
        # Use a shared build directory that all processes can access
        project_root = self.current_dir.parent.parent.parent
        self.build_dir = project_root / "build_cuda_shared"
        
        # Environment settings
        self.disable_cuda = disable_cuda or os.environ.get('DISABLE_CUDA_KERNELS', '0') == '1'
        
        # Kernel cache and availability
        self._kernel_module: Optional[Any] = None
        self._load_attempted = False
        
        # Kernel source configuration
        self.kernel_config = {
            'module_name': 'mcts_cuda_kernels',
            'functions': [
                'find_expansion_nodes',
                'batched_ucb_selection', 
                'quantum_ucb_selection',
                'vectorized_backup',
                'initialize_lookup_tables'
            ]
        }
        
        logger.debug(f"CudaManager initialized: disable_cuda={self.disable_cuda}")
    
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available and compilation is enabled"""
        if self.disable_cuda:
            return False
        return torch.cuda.is_available()
    
    def _check_compiled_module_exists(self) -> bool:
        """Check if compiled module exists"""
        module_path = self.build_dir / f"{self.kernel_config['module_name']}.so"
        return module_path.exists()
    
    def _load_compiled_kernels(self) -> bool:
        """
        Load pre-compiled CUDA kernels (compilation handled by setup.py)
        
        Returns:
            bool: True if loading successful, False otherwise
        """
        if not self.is_cuda_available():
            logger.debug("CUDA not available - skipping kernel loading")
            return False
        
        if self._load_attempted:
            return self._kernel_module is not None
        
        self._load_attempted = True
        
        # Try to load existing compiled module
        return self._try_load_existing_module()
    
    def get_kernels(self, compile_if_missing: bool = True) -> Optional[Any]:
        """
        Get compiled CUDA kernels module (detection only, no compilation)
        
        Args:
            compile_if_missing: Ignored (compilation handled by setup.py)
        
        Returns:
            Module with CUDA functions if available, None otherwise
        """
        if not self.is_cuda_available():
            logger.debug("CUDA not available, returning None")
            return None
        
        if self._kernel_module is None:
            # Try to load existing compiled module
            if not self._load_compiled_kernels():
                logger.debug("No pre-compiled kernels found")
                return None
        
        logger.debug(f"Returning kernel module: {self._kernel_module}")
        return self._kernel_module
    
    def _try_load_existing_module(self) -> bool:
        """Try to load an existing compiled module without recompilation"""
        try:
            module_path = self.build_dir / f"{self.kernel_config['module_name']}.so"
            logger.debug(f"Checking for module at: {module_path}")
            logger.debug(f"Module exists: {module_path.exists()}")
            logger.debug(f"Working directory: {os.getcwd()}")
            logger.debug(f"Build directory: {self.build_dir}")
            
            if not module_path.exists():
                logger.debug(f"Module not found at {module_path}")
                return False
            
            # Try to load the existing module using importlib
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                self.kernel_config['module_name'], 
                str(module_path)
            )
            if spec is None or spec.loader is None:
                logger.debug("Failed to create module spec")
                return False
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Add parallel_backup alias if needed
            if hasattr(module, 'vectorized_backup') and not hasattr(module, 'parallel_backup'):
                setattr(module, 'parallel_backup', module.vectorized_backup)
                logger.debug("Added parallel_backup alias to loaded module")
            
            self._kernel_module = module
            logger.debug(f"Successfully loaded existing module from {module_path}")
            return True
            
        except Exception as e:
            logger.debug(f"Failed to load existing module: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return False
    
    def validate_kernels(self) -> bool:
        """
        Validate that all expected kernel functions are available
        
        Returns:
            bool: True if all functions available, False otherwise
        """
        kernels = self.get_kernels(compile_if_missing=False)
        if kernels is None:
            return False
        
        try:
            for func_name in self.kernel_config['functions']:
                if not hasattr(kernels, func_name):
                    logger.error(f"Missing kernel function: {func_name}")
                    return False
            
            logger.debug("All kernel functions validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Kernel validation failed: {e}")
            return False
    
    def get_kernel_info(self) -> Dict[str, Any]:
        """Get information about compiled kernels"""
        return {
            'cuda_available': self.is_cuda_available(),
            'load_attempted': self._load_attempted,
            'kernels_available': self._kernel_module is not None,
            'compiled_module_exists': self._check_compiled_module_exists(),
            'functions': self.kernel_config['functions']
        }
    
    def clear_cache(self):
        """Clear kernel module cache (compilation now handled by setup.py)"""
        self._kernel_module = None
        self._load_attempted = False
        logger.info("CUDA kernel cache cleared - run 'python setup.py install' to recompile")


# Global manager instance
_cuda_manager: Optional[CudaManager] = None

def get_cuda_manager(force_rebuild: bool = False, disable_cuda: bool = False) -> CudaManager:
    """
    Get CUDA manager instance (per-process in multiprocessing)
    
    Args:
        force_rebuild: Force recompilation of kernels
        disable_cuda: Disable CUDA compilation
        
    Returns:
        CudaManager instance
    """
    global _cuda_manager
    
    # In multiprocessing, create a new manager for each process
    current_pid = os.getpid()
    if (_cuda_manager is None or 
        force_rebuild or 
        not hasattr(_cuda_manager, '_process_id') or 
        getattr(_cuda_manager, '_process_id', None) != current_pid):
        
        _cuda_manager = CudaManager(force_rebuild=force_rebuild, disable_cuda=disable_cuda)
        _cuda_manager._process_id = current_pid
        logger.debug(f"Created new CUDA manager for process {current_pid}")
    
    return _cuda_manager

def get_cuda_kernels(compile_if_missing: bool = True) -> Optional[Any]:
    """
    Convenience function to get compiled CUDA kernels
    
    Args:
        compile_if_missing: If True, attempt compilation if kernels not found.
                           If False, only try to load existing kernels.
    
    Returns:
        Compiled CUDA kernels module or None if not available
    """
    manager = get_cuda_manager()
    return manager.get_kernels(compile_if_missing=compile_if_missing)

def detect_cuda_kernels() -> Optional[Any]:
    """
    Detect if CUDA kernels are available without triggering compilation
    
    Returns:
        Compiled CUDA kernels module or None if not available
    """
    return get_cuda_kernels(compile_if_missing=False)

def validate_cuda_setup() -> bool:
    """
    Validate complete CUDA setup
    
    Returns:
        bool: True if CUDA setup is working, False otherwise
    """
    manager = get_cuda_manager()
    return manager.validate_kernels()

def get_cuda_info() -> Dict[str, Any]:
    """Get comprehensive CUDA setup information"""
    manager = get_cuda_manager()
    return manager.get_kernel_info()

def clear_cuda_cache():
    """Clear CUDA compilation cache"""
    manager = get_cuda_manager()
    manager.clear_cache()

def force_cuda_reload():
    """Force reload of CUDA kernels (recompilation handled by setup.py)"""
    manager = get_cuda_manager()
    manager.clear_cache()