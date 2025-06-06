"""Compile custom CUDA kernels for MCTS

This script compiles the custom CUDA kernels and makes them available as a Python module.
"""

import os
import torch
from torch.utils.cpp_extension import load_inline
import logging

logger = logging.getLogger(__name__)

# Check if we should disable CUDA compilation
DISABLE_CUDA_COMPILE = os.environ.get('DISABLE_CUDA_COMPILE', '0') == '1'

# Check CUDA availability
if not torch.cuda.is_available() or DISABLE_CUDA_COMPILE:
    if DISABLE_CUDA_COMPILE:
        logger.info("CUDA compilation disabled by environment variable")
    else:
        logger.info("CUDA is not available. Using fallback implementations.")
    
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
    
else:
    # Set CUDA architecture list to avoid warnings
    if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
        # Set for RTX 3060 Ti (compute capability 8.6)
        os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'

    # Read CUDA source - use minimal version for now to debug compilation
    cuda_source_path = os.path.join(os.path.dirname(__file__), 'custom_cuda_kernels_minimal.cu')
    if not os.path.exists(cuda_source_path):
        # Fall back to full version if minimal doesn't exist
        cuda_source_path = os.path.join(os.path.dirname(__file__), 'custom_cuda_kernels.cu')
    
    with open(cuda_source_path, 'r') as f:
        cuda_source = f.read()

    # C++ wrapper source
    cpp_source = """
#include <torch/extension.h>

// Forward declarations of CUDA functions
torch::Tensor batched_ucb_selection_cuda(
    torch::Tensor q_values,
    torch::Tensor visit_counts,
    torch::Tensor parent_visits,
    torch::Tensor priors,
    torch::Tensor row_ptr,
    torch::Tensor col_indices,
    float c_puct
);

torch::Tensor parallel_backup_cuda(
    torch::Tensor paths,
    torch::Tensor leaf_values,
    torch::Tensor path_lengths,
    torch::Tensor value_sums,
    torch::Tensor visit_counts
);

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_ucb_selection", &batched_ucb_selection_cuda, "Batched UCB selection (CUDA)");
    m.def("parallel_backup", &parallel_backup_cuda, "Parallel backup (CUDA)");
}
"""

    # Compile flags for optimization - only compile for current architecture
    capability = torch.cuda.get_device_capability()
    arch_flag = f'compute_{capability[0]}{capability[1]}'
    code_flag = f'sm_{capability[0]}{capability[1]}'
    
    extra_cuda_cflags = [
        '-O3',
        '-use_fast_math',
        '-gencode', f'arch={arch_flag},code={code_flag}',  # Only current GPU
    ]

    # Try to load pre-compiled module first
    try:
        # Check if pre-compiled module exists
        module_path = os.path.join(os.path.dirname(__file__), 'mcts_cuda_kernels.cpython-312-x86_64-linux-gnu.so')
        if os.path.exists(module_path):
            # Load pre-compiled module
            import importlib.util
            spec = importlib.util.spec_from_file_location("mcts_cuda_kernels", module_path)
            mcts_cuda = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mcts_cuda)
            
            logger.info("Successfully loaded pre-compiled CUDA kernels")
            
            # Make kernels available
            batched_ucb_selection = mcts_cuda.batched_ucb_selection
            parallel_backup = mcts_cuda.parallel_backup
            
            # Additional kernels from full version (if available)
            if hasattr(mcts_cuda, 'batched_add_children'):
                batched_add_children = mcts_cuda.batched_add_children
            if hasattr(mcts_cuda, 'quantum_interference'):
                quantum_interference = mcts_cuda.quantum_interference
            if hasattr(mcts_cuda, 'evaluate_gomoku_positions'):
                evaluate_gomoku_positions = mcts_cuda.evaluate_gomoku_positions
            
            CUDA_KERNELS_AVAILABLE = True
        else:
            # Try JIT compilation as fallback
            logger.info("Pre-compiled module not found, attempting JIT compilation...")
            
            # Set compilation timeout to prevent hanging
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("CUDA compilation timed out")
            
            # Set a 30 second timeout
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)
            
            # Create build directory
            import tempfile
            build_dir = tempfile.mkdtemp(prefix='mcts_cuda_')
            
            # Load and compile the CUDA extension
            mcts_cuda = load_inline(
                name='mcts_cuda',
                cpp_sources=[cpp_source],
                cuda_sources=[cuda_source],
                extra_cuda_cflags=extra_cuda_cflags,
                verbose=False,  # Disable verbose to reduce output
                with_cuda=True,
                build_directory=build_dir
            )
            
            # Cancel timeout if compilation succeeded
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            
            logger.info("Successfully compiled custom CUDA kernels")
            
            # Make kernels available
            batched_ucb_selection = mcts_cuda.batched_ucb_selection
            parallel_backup = mcts_cuda.parallel_backup
            
            # Additional kernels from full version (if available)
            if hasattr(mcts_cuda, 'batched_add_children'):
                batched_add_children = mcts_cuda.batched_add_children
            if hasattr(mcts_cuda, 'quantum_interference'):
                quantum_interference = mcts_cuda.quantum_interference
            if hasattr(mcts_cuda, 'evaluate_gomoku_positions'):
                evaluate_gomoku_positions = mcts_cuda.evaluate_gomoku_positions
            
            CUDA_KERNELS_AVAILABLE = True
        
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