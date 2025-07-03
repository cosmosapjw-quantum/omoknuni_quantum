"""GPU acceleration module for MCTS

This module provides GPU-accelerated operations for the vectorized MCTS implementation.
The main component is the MCTSGPUAccelerator which provides hardware-accelerated 
implementations of computationally intensive MCTS operations including UCB selection, 
tree traversal, and value backup.

Key Components:
- MCTSGPUAccelerator: Main acceleration interface with CUDA/PyTorch fallbacks
- CSRTree: Compressed sparse row tree structure for efficient memory usage
- CSRGPUKernels: Specialized kernels for CSR tree operations
"""

# Import MCTS GPU accelerator
from .mcts_gpu_accelerator import (
    MCTSGPUAccelerator, 
    get_mcts_gpu_accelerator
)

# Import CSR tree components
from .csr_tree import CSRTree, CSRTreeConfig

# Import CSR operations (now using unified kernels)
from .csr_gpu_kernels import (
    CSRBatchOperations,
    get_csr_batch_operations,
    check_cuda_available,
    csr_batch_ucb_torch,
    csr_coalesced_children_gather
)

# Note: GPU optimizers removed as they were not used in the codebase

# Note: Legacy aliases removed in streamlined build

# GPU tree kernels (if available)
try:
    from .gpu_tree_kernels import GPUTreeKernels
except ImportError:
    GPUTreeKernels = None

HAS_QUANTUM_CUDA = True  # Now integrated into unified kernels

# Optional GPU attack/defense module
try:
    from .gpu_attack_defense import gpu_compute_attack_defense_scores
    HAS_GPU_ATTACK_DEFENSE = True
except ImportError:
    gpu_compute_attack_defense_scores = None
    HAS_GPU_ATTACK_DEFENSE = False

# Check CUDA availability
import torch
CUDA_AVAILABLE = torch.cuda.is_available()

# Note: Triton support removed in streamlined build

__all__ = [
    # MCTS GPU accelerator
    "MCTSGPUAccelerator",
    "get_mcts_gpu_accelerator",
    
    # CSR Tree
    "CSRTree",
    "CSRTreeConfig",
    
    # CSR GPU operations
    "CSRBatchOperations",
    "get_csr_batch_operations",
    "check_cuda_available",
    "csr_batch_ucb_torch",
    "csr_coalesced_children_gather",
    
    
    # Status flags
    "CUDA_AVAILABLE",
]

if GPUTreeKernels is not None:
    __all__.append("GPUTreeKernels")

# Note: Legacy quantum aliases removed in streamlined build

if HAS_GPU_ATTACK_DEFENSE:
    __all__.append("gpu_compute_attack_defense_scores")