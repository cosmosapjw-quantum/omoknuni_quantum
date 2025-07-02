"""
Modular CUDA Kernel Manager for MCTS

This module manages the compilation and loading of modular CUDA kernels
for improved compilation speed and maintainability.

The previous unified_cuda_kernels.cu file (1593 lines) has been split into:
- mcts_core_kernels.cu: Core MCTS operations
- mcts_selection_kernels.cu: UCB selection logic
- mcts_quantum_kernels.cu: Quantum enhancements
- Additional specialized modules as needed

Benefits:
- Faster compilation (smaller files)
- Better maintainability
- Incremental compilation support
- Cleaner separation of concerns
"""

import os
import sys
import time
import signal
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
import torch.utils.cpp_extension as cpp_extension

# Suppress verbose torch compilation logging
torch_ext_logger = logging.getLogger('torch.utils.cpp_extension')
torch_ext_logger.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

class ModularKernelManager:
    """Manages modular CUDA kernel compilation and loading"""
    
    def __init__(self, device: str = 'cuda', enable_caching: bool = True):
        self.device = torch.device(device)
        self.enable_caching = enable_caching
        self.loaded_modules: Dict[str, Any] = {}
        self.compilation_times: Dict[str, float] = {}
        
        # Define kernel modules
        self.kernel_modules = {
            'core': {
                'file': 'mcts_core_kernels.cu',
                'functions': ['find_expansion_nodes', 'batch_process_legal_moves'],
                'description': 'Core MCTS tree operations'
            },
            'selection': {
                'file': 'mcts_selection_kernels.cu', 
                'functions': ['batched_ucb_selection', 'optimized_ucb_selection'],
                'description': 'UCB selection algorithms'
            },
            'quantum': {
                'file': 'mcts_quantum_kernels.cu',
                'functions': ['batched_ucb_selection_quantum', 'apply_quantum_interference', 'phase_kicked_policy'],
                'description': 'Quantum-enhanced MCTS features'
            }
        }
        
        self.gpu_dir = Path(__file__).parent
        
    def get_kernel_path(self, module_name: str) -> Path:
        """Get the full path to a kernel module file"""
        return self.gpu_dir / self.kernel_modules[module_name]['file']
        
    def is_module_available(self, module_name: str) -> bool:
        """Check if a kernel module file exists"""
        return self.get_kernel_path(module_name).exists()
        
    def compile_module(self, module_name: str, force_recompile: bool = False, timeout: int = 60) -> Optional[Any]:
        """Compile a specific kernel module with timeout protection"""
        if module_name not in self.kernel_modules:
            logger.error(f"Unknown kernel module: {module_name}")
            return None
            
        module_info = self.kernel_modules[module_name]
        kernel_path = self.get_kernel_path(module_name)
        
        if not kernel_path.exists():
            logger.warning(f"Kernel file not found: {kernel_path}")
            return None
            
        # Check if already loaded and caching is enabled
        if not force_recompile and self.enable_caching and module_name in self.loaded_modules:
            logger.debug(f"Using cached module: {module_name}")
            return self.loaded_modules[module_name]
        
        # Try to use pre-compiled modules first
        try:
            precompiled_name = f"mcts_{module_name}_kernels_precompiled"
            if self._try_load_precompiled(precompiled_name):
                logger.debug(f"Using pre-compiled {module_name} module")
                return self.loaded_modules.get(module_name)
        except Exception as e:
            logger.debug(f"Pre-compiled loading failed for {module_name}: {e}")
            
        logger.info(f"Compiling {module_info['description']}: {kernel_path.name}")
        
        try:
            start_time = time.time()
            
            # Add timeout protection
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Compilation of {module_name} timed out after {timeout}s")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            try:
                # Compile the CUDA module with reduced optimization to avoid hanging
                # Import suppress_output from cuda_compile
                from .cuda_compile import suppress_output
                with suppress_output():
                    module = cpp_extension.load(
                        name=f"mcts_{module_name}_kernels",
                        sources=[str(kernel_path)],
                        extra_cflags=['-O1'],  # Reduced optimization
                        extra_cuda_cflags=['-O1', '--use_fast_math', '-lineinfo'],
                        verbose=False
                    )
                
                compilation_time = time.time() - start_time
                self.compilation_times[module_name] = compilation_time
                
                # Cache the compiled module
                self.loaded_modules[module_name] = module
                
                logger.info(f"Successfully compiled {module_name} in {compilation_time:.2f}s")
                return module
                
            finally:
                signal.alarm(0)  # Disable alarm
                signal.signal(signal.SIGALRM, old_handler)
            
        except TimeoutError as e:
            logger.error(f"Compilation timeout: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to compile {module_name}: {e}")
            return None
    
    def _try_load_precompiled(self, module_name: str) -> bool:
        """Try to load a pre-compiled module"""
        try:
            # First try to use torch.utils.cpp_extension.load to load cached module
            from torch.utils.cpp_extension import load
            
            # Look for the source file
            module_info = self.kernel_modules.get(module_name.replace('mcts_', '').replace('_kernels_precompiled', ''))
            if not module_info:
                return False
                
            kernel_path = self.get_kernel_path(module_name.replace('mcts_', '').replace('_kernels_precompiled', ''))
            if not kernel_path.exists():
                return False
            
            # Try to load using torch's caching system  
            from .cuda_compile import suppress_output
            with suppress_output():
                module = load(
                    name=module_name,
                    sources=[str(kernel_path)],
                    extra_cflags=['-O1'],
                    extra_cuda_cflags=['-O1', '--use_fast_math'],
                    verbose=False
                )
            
            if module:
                # Cache it in our loaded modules
                clean_name = module_name.replace('mcts_', '').replace('_kernels_precompiled', '')
                self.loaded_modules[clean_name] = module
                logger.debug(f"Successfully loaded pre-compiled module: {module_name}")
                return True
                
        except Exception as e:
            logger.debug(f"Failed to load pre-compiled {module_name}: {e}")
            
        return False
            
    def load_all_modules(self, required_modules: Optional[List[str]] = None, timeout_per_module: int = 30) -> Dict[str, Any]:
        """Load all available kernel modules with fallback for problematic ones"""
        if required_modules is None:
            required_modules = list(self.kernel_modules.keys())
            
        loaded = {}
        total_start_time = time.time()
        
        # Priority order: load essential modules first, optional ones last
        essential_modules = ['core', 'selection']
        optional_modules = [m for m in required_modules if m not in essential_modules]
        
        for module_name in essential_modules + optional_modules:
            if module_name not in required_modules:
                continue
                
            if self.is_module_available(module_name):
                try:
                    logger.debug(f"Loading {module_name} module...")
                    module = self.compile_module(module_name, timeout=timeout_per_module)
                    if module is not None:
                        loaded[module_name] = module
                        logger.debug(f"✅ {module_name} module loaded successfully")
                    else:
                        logger.warning(f"⚠️ Failed to load {module_name} module, continuing without it")
                        if module_name in essential_modules:
                            logger.error(f"Essential module {module_name} failed to load!")
                except Exception as e:
                    logger.error(f"❌ Error loading {module_name}: {e}")
                    if module_name in optional_modules:
                        logger.info(f"Skipping optional module {module_name}, training can continue")
                    else:
                        logger.error(f"Essential module {module_name} failed!")
            else:
                logger.warning(f"Module file not available: {module_name}")
                
        total_time = time.time() - total_start_time
        logger.info(f"Module loading completed: {len(loaded)}/{len(required_modules)} modules in {total_time:.2f}s")
        
        # Warn if no modules loaded
        if len(loaded) == 0:
            logger.warning("No CUDA modules loaded - falling back to PyTorch implementations")
        
        return loaded
        
    def get_module(self, module_name: str) -> Optional[Any]:
        """Get a loaded module by name"""
        if module_name in self.loaded_modules:
            return self.loaded_modules[module_name]
        else:
            return self.compile_module(module_name)
            
    def get_function(self, module_name: str, function_name: str):
        """Get a specific function from a module"""
        module = self.get_module(module_name)
        if module is None:
            raise RuntimeError(f"Module {module_name} not available")
            
        if not hasattr(module, function_name):
            raise RuntimeError(f"Function {function_name} not found in module {module_name}")
            
        return getattr(module, function_name)
        
    def get_compilation_stats(self) -> Dict[str, Any]:
        """Get compilation statistics"""
        stats = {
            'modules_loaded': len(self.loaded_modules),
            'total_modules': len(self.kernel_modules),
            'compilation_times': self.compilation_times.copy(),
            'total_compilation_time': sum(self.compilation_times.values())
        }
        
        if stats['total_compilation_time'] > 0:
            stats['average_compilation_time'] = stats['total_compilation_time'] / len(self.compilation_times)
        else:
            stats['average_compilation_time'] = 0.0
            
        return stats
        
    def cleanup(self):
        """Clean up loaded modules"""
        self.loaded_modules.clear()
        self.compilation_times.clear()
        logger.info("Cleaned up kernel manager")


class UnifiedKernelInterface:
    """Unified interface that provides backward compatibility with the old unified_kernels"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.kernel_manager = ModularKernelManager(device)
        self._ensure_modules_loaded()
        
    def _ensure_modules_loaded(self):
        """Ensure required modules are loaded with fallback handling"""
        # Load core modules that are always needed
        required_modules = ['core', 'selection']
        
        # Add quantum module if available (but it's optional)
        if self.kernel_manager.is_module_available('quantum'):
            required_modules.append('quantum')
            
        try:
            # Use shorter timeout to avoid hanging during training
            self.modules = self.kernel_manager.load_all_modules(required_modules, timeout_per_module=20)
            
            # Log what we successfully loaded
            if self.modules:
                loaded_names = list(self.modules.keys())
                logger.info(f"Modular kernels ready: {loaded_names}")
            else:
                logger.warning("No modular kernels loaded, using fallback implementations")
                
        except Exception as e:
            logger.error(f"Error loading modular kernels: {e}")
            logger.info("Continuing with fallback implementations")
            self.modules = {}
        
    def find_expansion_nodes(self, *args, **kwargs):
        """Core: Find nodes needing expansion"""
        try:
            if 'core' in self.modules:
                func = self.kernel_manager.get_function('core', 'find_expansion_nodes')
                return func(*args, **kwargs)
            else:
                raise RuntimeError("Core module not available")
        except Exception as e:
            logger.warning(f"Core kernel unavailable: {e}")
            raise NotImplementedError("Fallback for find_expansion_nodes not implemented")
        
    def batch_process_legal_moves(self, *args, **kwargs):
        """Core: Process legal moves in batch"""
        try:
            if 'core' in self.modules:
                func = self.kernel_manager.get_function('core', 'batch_process_legal_moves')
                return func(*args, **kwargs)
            else:
                raise RuntimeError("Core module not available")
        except Exception as e:
            logger.warning(f"Core kernel unavailable: {e}")
            raise NotImplementedError("Fallback for batch_process_legal_moves not implemented")
        
    def batched_ucb_selection(self, *args, **kwargs):
        """Selection: Standard UCB selection"""
        try:
            if 'selection' in self.modules:
                func = self.kernel_manager.get_function('selection', 'batched_ucb_selection')
                return func(*args, **kwargs)
            else:
                raise RuntimeError("Selection module not available")
        except Exception as e:
            logger.warning(f"Selection kernel unavailable: {e}")
            raise NotImplementedError("Fallback for batched_ucb_selection not implemented")
        
    def optimized_ucb_selection(self, *args, **kwargs):
        """Selection: Optimized UCB selection with shared memory"""
        try:
            if 'selection' in self.modules:
                func = self.kernel_manager.get_function('selection', 'optimized_ucb_selection')
                return func(*args, **kwargs)
            else:
                # Fall back to standard UCB selection
                return self.batched_ucb_selection(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Optimized UCB kernel unavailable: {e}")
            return self.batched_ucb_selection(*args, **kwargs)
        
    def batched_ucb_selection_quantum(self, *args, **kwargs):
        """Quantum: Quantum-enhanced UCB selection"""
        try:
            if 'quantum' in self.modules:
                func = self.kernel_manager.get_function('quantum', 'batched_ucb_selection_quantum')
                return func(*args, **kwargs)
            else:
                logger.debug("Quantum module not available, falling back to classical UCB")
                return self.batched_ucb_selection(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Quantum kernels not available: {e}")
            return self.batched_ucb_selection(*args, **kwargs)
            
    def apply_quantum_interference(self, *args, **kwargs):
        """Quantum: Apply quantum interference to probabilities"""
        try:
            if 'quantum' in self.modules:
                func = self.kernel_manager.get_function('quantum', 'apply_quantum_interference')
                return func(*args, **kwargs)
            else:
                logger.debug("Quantum module not available, skipping interference")
                return args[0]  # Return input unchanged
        except Exception as e:
            logger.warning(f"Quantum interference unavailable: {e}")
            return args[0]  # Return input unchanged
        
    def phase_kicked_policy(self, *args, **kwargs):
        """Quantum: Apply phase kicks to policy"""
        try:
            if 'quantum' in self.modules:
                func = self.kernel_manager.get_function('quantum', 'phase_kicked_policy')
                return func(*args, **kwargs)
            else:
                logger.debug("Quantum module not available, skipping phase kicks")
                return args[0]  # Return input unchanged
        except Exception as e:
            logger.warning(f"Phase kicked policy unavailable: {e}")
            return args[0]  # Return input unchanged
        
    def get_stats(self) -> Dict[str, Any]:
        """Get kernel compilation and usage statistics"""
        return self.kernel_manager.get_compilation_stats()


# Factory function for backward compatibility
def get_unified_kernels(device: str = 'cuda') -> UnifiedKernelInterface:
    """
    Get unified kernel interface (replaces old unified_kernels)
    
    This function provides backward compatibility while using the new
    modular kernel system for improved compilation speed.
    """
    return UnifiedKernelInterface(device)


# Module-level instance for global access
_global_kernel_interface: Optional[UnifiedKernelInterface] = None

def get_global_kernels(device: str = 'cuda') -> UnifiedKernelInterface:
    """Get global kernel interface instance"""
    global _global_kernel_interface
    
    if _global_kernel_interface is None:
        _global_kernel_interface = UnifiedKernelInterface(device)
        
    return _global_kernel_interface

def cleanup_global_kernels():
    """Clean up global kernel interface"""
    global _global_kernel_interface
    
    if _global_kernel_interface is not None:
        _global_kernel_interface.kernel_manager.cleanup()
        _global_kernel_interface = None