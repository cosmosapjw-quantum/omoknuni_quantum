"""GPU acceleration module for MCTS

This module provides GPU-accelerated operations for the vectorized MCTS implementation.
It uses a unified kernel architecture for consistency and performance.
"""

# Import unified kernels first
from .unified_kernels import UnifiedGPUKernels, get_unified_kernels

# Import CSR tree components
from .csr_tree import CSRTree, CSRTreeConfig

# Import CSR operations (now using unified kernels)
from .csr_gpu_kernels import (
    CSRBatchOperations,
    get_csr_batch_operations,
    get_csr_kernels,
    check_cuda_available,
    csr_batch_ucb_torch,
    csr_coalesced_children_gather,
    # Legacy aliases
    CSRGPUKernels,
    OptimizedCSRKernels
)

# Note: GPU optimizers removed as they were not used in the codebase

# Legacy aliases for backward compatibility
OptimizedCUDAKernels = UnifiedGPUKernels
CUDAKernels = UnifiedGPUKernels
OptimizedQuantumKernels = UnifiedGPUKernels
create_optimized_quantum_kernels = lambda: UnifiedGPUKernels()

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

# Check Triton availability
try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

__all__ = [
    # Unified kernel interface
    "UnifiedGPUKernels",
    "get_unified_kernels",
    
    # CSR Tree
    "CSRTree",
    "CSRTreeConfig",
    
    # CSR GPU operations
    "CSRBatchOperations",
    "get_csr_batch_operations",
    "get_csr_kernels",
    "check_cuda_available",
    "csr_batch_ucb_torch",
    "csr_coalesced_children_gather",
    
    # Legacy compatibility
    "CSRGPUKernels",
    "OptimizedCSRKernels",
    "OptimizedCUDAKernels",
    "CUDAKernels",
    
    
    # Status flags
    "CUDA_AVAILABLE",
    "HAS_TRITON",
]

if GPUTreeKernels is not None:
    __all__.append("GPUTreeKernels")

if HAS_QUANTUM_CUDA:
    __all__.extend(["create_optimized_quantum_kernels", "OptimizedQuantumKernels"])

if HAS_GPU_ATTACK_DEFENSE:
    __all__.append("gpu_compute_attack_defense_scores")