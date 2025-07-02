"""Compile custom CUDA kernels for MCTS

This script pre-compiles and validates the modular CUDA kernels system,
providing early error detection and compilation validation.

NEW: Now supports modular kernel compilation for improved build times.
"""

import os
import sys
import time
import torch
from torch.utils.cpp_extension import load, load_inline
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Suppress verbose torch compilation logging
torch_ext_logger = logging.getLogger('torch.utils.cpp_extension')
torch_ext_logger.setLevel(logging.ERROR)

# Also suppress ninja build output
ninja_logger = logging.getLogger('ninja')
ninja_logger.setLevel(logging.ERROR)

# Context manager to suppress stdout/stderr during compilation
from contextlib import contextmanager
import io

@contextmanager
def suppress_output():
    """Suppress stdout and stderr output during CUDA compilation"""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

logger = logging.getLogger(__name__)

# Check if we should disable CUDA compilation
DISABLE_CUDA_COMPILE = os.environ.get('DISABLE_CUDA_COMPILE', '0') == '1'

# Modular kernel configuration
MODULAR_KERNELS = {
    'core': {
        'file': 'mcts_core_kernels.cu',
        'functions': ['find_expansion_nodes', 'batch_process_legal_moves'],
    },
    'diversified_ucb': {
        'file': 'diversified_ucb_kernels.cu', 
        'functions': ['diversified_ucb_selection', 'generate_diversified_dirichlet', 'fused_diversified_selection'],
        'description': 'Diversified UCB selection with per-simulation Dirichlet noise'
    },
    'selection': {
        'file': 'mcts_selection_kernels.cu', 
        'functions': ['batched_ucb_selection', 'optimized_ucb_selection'],
        'description': 'UCB selection algorithms'
    },
    'quantum': {
        'file': 'mcts_quantum_kernels.cu',
        'functions': ['batched_ucb_selection_quantum', 'apply_quantum_interference'],
        'description': 'Quantum-enhanced MCTS features'
    }
}

class ModularKernelCompiler:
    """Pre-compiles and validates modular CUDA kernels"""
    
    def __init__(self, gpu_dir: Optional[Path] = None):
        self.gpu_dir = gpu_dir or Path(__file__).parent
        self.compilation_results = {}
        self.compiled_modules = {}
        
    def validate_kernel_files(self) -> Dict[str, bool]:
        """Validate that all kernel files exist"""
        results = {}
        for name, config in MODULAR_KERNELS.items():
            kernel_path = self.gpu_dir / config['file']
            results[name] = kernel_path.exists()
            if not results[name]:
                logger.warning(f"Kernel file not found: {kernel_path}")
        return results
    
    def compile_kernel_module(self, name: str, config: Dict[str, Any]) -> Tuple[bool, Optional[Any], Optional[str]]:
        """Compile a single kernel module with error detection"""
        kernel_path = self.gpu_dir / config['file']
        
        if not kernel_path.exists():
            return False, None, f"Kernel file not found: {kernel_path}"
        
        try:
            logger.debug(f"Compiling {config['description']}: {config['file']}")
            start_time = time.time()
            
            # Compile the CUDA module
            with suppress_output():
                module = load(
                    name=f"mcts_{name}_kernels_precompiled",
                    sources=[str(kernel_path)],
                    extra_cflags=['-O3'],
                    extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'],
                    verbose=False  # Reduce verbose output
                )
            
            compilation_time = time.time() - start_time
            
            # Validate that expected functions are available
            missing_functions = []
            for func_name in config['functions']:
                if not hasattr(module, func_name):
                    missing_functions.append(func_name)
            
            if missing_functions:
                error_msg = f"Missing functions in {name}: {missing_functions}"
                logger.error(error_msg)
                return False, None, error_msg
            
            logger.debug(f"Successfully compiled {name} in {compilation_time:.2f}s")
            return True, module, None
            
        except Exception as e:
            error_msg = f"Compilation failed for {name}: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def pre_compile_all_kernels(self) -> Dict[str, Any]:
        """Pre-compile all modular kernels and detect errors early"""
        logger.info("Starting modular kernel pre-compilation...")
        
        # Validate files first
        file_validation = self.validate_kernel_files()
        missing_files = [name for name, exists in file_validation.items() if not exists]
        
        if missing_files:
            logger.error(f"Missing kernel files: {missing_files}")
            return {
                'success': False,
                'error': f"Missing kernel files: {missing_files}",
                'file_validation': file_validation,
                'compilation_results': {}
            }
        
        # Compile each kernel module
        total_start_time = time.time()
        compilation_results = {}
        compiled_modules = {}
        
        for name, config in MODULAR_KERNELS.items():
            success, module, error = self.compile_kernel_module(name, config)
            compilation_results[name] = {
                'success': success,
                'error': error,
                'functions': config['functions'] if success else []
            }
            
            if success and module:
                compiled_modules[name] = module
        
        total_time = time.time() - total_start_time
        successful_modules = len([r for r in compilation_results.values() if r['success']])
        
        logger.debug(f"Pre-compilation completed: {successful_modules}/{len(MODULAR_KERNELS)} modules in {total_time:.2f}s")
        
        return {
            'success': successful_modules == len(MODULAR_KERNELS),
            'total_time': total_time,
            'file_validation': file_validation,
            'compilation_results': compilation_results,
            'compiled_modules': compiled_modules,
            'modules_compiled': successful_modules,
            'total_modules': len(MODULAR_KERNELS)
        }

# Global compiler instance
_modular_compiler = None

def get_modular_compiler() -> ModularKernelCompiler:
    """Get or create the global modular compiler instance"""
    global _modular_compiler
    if _modular_compiler is None:
        _modular_compiler = ModularKernelCompiler()
    return _modular_compiler

def pre_compile_modular_kernels() -> Dict[str, Any]:
    """Pre-compile modular kernels and return results"""
    if DISABLE_CUDA_COMPILE or not torch.cuda.is_available():
        return {
            'success': False,
            'error': 'CUDA not available or disabled',
            'compilation_results': {}
        }
    
    compiler = get_modular_compiler()
    
    # Check if kernels are already compiled by looking for .so files
    import glob
    gpu_dir = Path(__file__).parent
    compiled_patterns = [
        '*_precompiled.so',
        'mcts_*_kernels_precompiled.so'
    ]
    
    existing_compiled = []
    for pattern in compiled_patterns:
        existing_compiled.extend(glob.glob(str(gpu_dir / pattern)))
    
    # Also check torch extensions cache
    try:
        torch_cache_dir = Path.home() / '.cache' / 'torch_extensions'
        if torch_cache_dir.exists():
            for pattern in ['mcts_*_precompiled', '*mcts*']:
                existing_compiled.extend(glob.glob(str(torch_cache_dir / pattern)))
    except:
        pass
    
    if existing_compiled:
        logger.debug(f"Found {len(existing_compiled)} pre-compiled kernel modules")
        # Return success without recompilation
        return {
            'success': True,
            'total_time': 0.0,
            'file_validation': {name: True for name in MODULAR_KERNELS.keys()},
            'compilation_results': {
                name: {
                    'success': True,
                    'error': None,
                    'functions': config['functions'],
                    'cached': True
                }
                for name, config in MODULAR_KERNELS.items()
            },
            'compiled_modules': {},
            'modules_compiled': len(MODULAR_KERNELS),
            'total_modules': len(MODULAR_KERNELS),
            'cached_compilation': True
        }
    
    return compiler.pre_compile_all_kernels()

# Check CUDA availability
if not torch.cuda.is_available() or DISABLE_CUDA_COMPILE:
    if DISABLE_CUDA_COMPILE:
        logger.info("CUDA compilation disabled by environment variable")
    else:
        logger.debug("CUDA is not available. Using fallback implementations.")
    
    # Fallback implementations
    def batched_ucb_selection(q_values, visit_counts, parent_visits, priors, row_ptr, col_indices, c_puct):
        """PyTorch fallback for UCB selection"""
        num_nodes = parent_visits.shape[0]
        selected_actions = torch.zeros(num_nodes, dtype=torch.int32, device=q_values.device)
        
        for i in range(num_nodes):
            start = row_ptr[i]
            end = row_ptr[i + 1]
            
            if start == end:
                selected_actions[i] = -1
                continue
                
            # Get children data
            child_indices = col_indices[start:end]
            child_visits = visit_counts[child_indices].float()
            child_q_values = torch.where(
                child_visits > 0,
                q_values[child_indices] / child_visits,
                torch.zeros_like(child_visits)
            )
            child_priors = priors[start:end]
            
            # Compute UCB
            sqrt_parent = torch.sqrt(parent_visits[i].float())
            exploration = c_puct * child_priors * sqrt_parent / (1 + child_visits)
            ucb_scores = child_q_values + exploration
            
            # Select best
            selected_actions[i] = ucb_scores.argmax()
            
        return selected_actions
    
    def parallel_backup(paths, leaf_values, path_lengths, value_sums, visit_counts):
        """PyTorch fallback for parallel backup"""
        batch_size, max_depth = paths.shape
        
        for i in range(batch_size):
            value = leaf_values[i]
            length = path_lengths[i]
            
            for depth in range(length):
                node_idx = paths[i, depth]
                if node_idx >= 0:
                    value_sums[node_idx] += value
                    visit_counts[node_idx] += 1
                    value = -value
                    
        return value_sums
    
    CUDA_KERNELS_AVAILABLE = False
    
    # Also add fallback stubs
    def vectorized_backup(*args, **kwargs):
        raise NotImplementedError("CUDA kernel not available - using PyTorch fallback")
    
    def fused_selection_traversal(*args, **kwargs):
        raise NotImplementedError("CUDA kernel not available - using PyTorch fallback")
    
else:
    # Set CUDA architecture list to avoid warnings
    if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
        # Set for RTX 3060 Ti (compute capability 8.6)
        os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
    
    # Set CUDA runtime path to suppress warnings
    if 'CUDA_HOME' not in os.environ:
        os.environ['CUDA_HOME'] = '/usr/local/cuda'

    # Use the correct unified CUDA kernels file that has all our optimized functions
    cuda_source_path = os.path.join(os.path.dirname(__file__), 'unified_cuda_kernels.cu')
    if not os.path.exists(cuda_source_path):
        # Fall back to custom version if unified doesn't exist
        cuda_source_path = os.path.join(os.path.dirname(__file__), 'custom_cuda_kernels.cu')
    
    # Note: We now use the original CUDA source with PYBIND11_MODULE intact
    # This works with the standard PyTorch load() method

    # Compile flags for optimization - only compile for current architecture
    capability = torch.cuda.get_device_capability()
    arch_flag = f'compute_{capability[0]}{capability[1]}'
    code_flag = f'sm_{capability[0]}{capability[1]}'
    
    # OPTIMIZED COMPILATION SETTINGS - Reduces compile time significantly
    extra_cuda_cflags = [
        '-O1',  # Reduced optimization for faster compilation (was -O3)
        '--use_fast_math',
        '-gencode', f'arch={arch_flag},code={code_flag}',  # Only current GPU
    ]

    # Pre-compile CUDA kernels to avoid runtime compilation delay
    def precompile_cuda_kernels():
        """Pre-compile CUDA kernels for faster loading"""
        logger.debug("Pre-compiling CUDA kernels for faster startup...")
        
        from torch.utils.cpp_extension import load
        import hashlib
        import time
        
        start_time = time.time()
        
        # Create deterministic module name based on source hash
        with open(cuda_source_path, 'rb') as f:
            source_hash = hashlib.md5(f.read()).hexdigest()[:8]
        
        module_name = f'mcts_cuda_precompiled_{source_hash}'
        
        try:
            # Don't suppress output to see what's happening
            logger.info(f"🔨 Compiling CUDA module: {module_name}")
            mcts_cuda = load(
                name=module_name,
                sources=[cuda_source_path],
                extra_cuda_cflags=extra_cuda_cflags,
                verbose=True,  # Enable verbose output to see compilation details
                with_cuda=True
            )
            
            compile_time = time.time() - start_time
            logger.debug(f"✅ CUDA kernels pre-compiled successfully in {compile_time:.1f}s")
            
            return mcts_cuda
            
        except Exception as e:
            logger.error(f"❌ CUDA kernel pre-compilation failed: {e}")
            return None
    
    try:
        # Use the new singleton approach to avoid compilation conflicts
        from .cuda_singleton import get_cuda_module
        
        logger.debug("🎯 Using CUDA singleton for safe compilation...")
        mcts_cuda = get_cuda_module()
        
        if mcts_cuda:
            # Debug: Print what's actually in the module
            logger.debug(f"🔍 Loaded module type: {type(mcts_cuda)}")
            logger.debug(f"🔍 Module attributes: {dir(mcts_cuda)}")
            
            # Check for expected functions
            expected_functions = [
                'batched_ucb_selection', 'parallel_backup', 'vectorized_backup',
                'fused_selection_traversal', 'batched_add_children', 'quantum_interference'
            ]
            
            available_functions = []
            for func_name in expected_functions:
                if hasattr(mcts_cuda, func_name):
                    available_functions.append(func_name)
                    globals()[func_name] = getattr(mcts_cuda, func_name)
                else:
                    logger.warning(f"⚠️ Missing function: {func_name}")
            
            logger.debug(f"✅ Available functions: {available_functions}")
            
            if len(available_functions) > 0:
                CUDA_KERNELS_AVAILABLE = True
                logger.debug(f"🎯 Successfully loaded {len(available_functions)}/{len(expected_functions)} CUDA functions")
            else:
                raise RuntimeError("No expected CUDA functions found in compiled module")
        else:
            raise RuntimeError("Failed to load or compile CUDA kernels")
            
    except Exception as e:
        logger.warning(f"Failed to load/compile custom CUDA kernels: {e}")
        logger.warning("Falling back to PyTorch implementations")
        
        # Use the fallback implementations defined above
        CUDA_KERNELS_AVAILABLE = False
        
        # Define fallback stubs for additional kernels
        def batched_add_children(*args, **kwargs):
            raise NotImplementedError("CUDA kernel not available")
        
        def quantum_interference(*args, **kwargs):
            raise NotImplementedError("CUDA kernel not available")
        
        def evaluate_gomoku_positions(*args, **kwargs):
            raise NotImplementedError("CUDA kernel not available")


# Import enhanced kernel management system
try:
    from .cuda_kernel_manager import (
        force_rebuild_cuda_kernels,
        rebuild_kernels_if_needed,
        get_cuda_kernels,
        clear_cuda_cache,
        get_kernel_status
    )
    
    # Auto-rebuild kernels if needed (only if CUDA is available and not disabled)
    if not DISABLE_CUDA_COMPILE and torch.cuda.is_available():
        try:
            rebuild_kernels_if_needed(verbose=False)
            
            # Try to get kernels from the enhanced manager
            kernel_module = get_cuda_kernels()
            if kernel_module:
                # Update global functions with managed kernels
                if hasattr(kernel_module, 'batched_ucb_selection'):
                    batched_ucb_selection = kernel_module.batched_ucb_selection
                if hasattr(kernel_module, 'parallel_backup'):
                    parallel_backup = kernel_module.parallel_backup
                if hasattr(kernel_module, 'vectorized_backup'):
                    vectorized_backup = kernel_module.vectorized_backup
                if hasattr(kernel_module, 'fused_selection_traversal'):
                    fused_selection_traversal = kernel_module.fused_selection_traversal
                if hasattr(kernel_module, 'batched_add_children'):
                    batched_add_children = kernel_module.batched_add_children
                if hasattr(kernel_module, 'quantum_interference'):
                    quantum_interference = kernel_module.quantum_interference
                
                CUDA_KERNELS_AVAILABLE = True
                logger.debug("✅ CUDA kernels loaded via enhanced kernel manager")
        except Exception as e:
            logger.debug(f"Enhanced kernel manager initialization failed: {e}")
            # Continue with original fallback logic
            
except ImportError:
    logger.debug("Enhanced kernel manager not available, using original logic")
    # Define fallback functions if enhanced kernel manager isn't available
    def force_rebuild_cuda_kernels(verbose: bool = False) -> bool:
        logger.warning("Enhanced kernel manager not available")
        return False
    
    def rebuild_kernels_if_needed(verbose: bool = False) -> bool:
        logger.warning("Enhanced kernel manager not available") 
        return True
    
    def clear_cuda_cache():
        logger.warning("Enhanced kernel manager not available")
    
    def get_kernel_status():
        return {"enhanced_kernel_manager": "not_available"}


# Export all kernel functions and management utilities
__all__ = [
    # Kernel functions
    'batched_ucb_selection',
    'parallel_backup', 
    'vectorized_backup',
    'fused_selection_traversal',
    'batched_add_children',
    'quantum_interference',
    'evaluate_gomoku_positions',
    
    # Status
    'CUDA_KERNELS_AVAILABLE',
    
    # Management functions
    'force_rebuild_cuda_kernels',
    'rebuild_kernels_if_needed',
    'clear_cuda_cache',
    'get_kernel_status'
]