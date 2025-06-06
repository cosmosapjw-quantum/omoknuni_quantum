"""GPU acceleration components"""

from .csr_tree import CSRTree, CSRTreeConfig
from .optimized_wave_engine import OptimizedWaveEngine, OptimizedWaveConfig
from .gpu_optimizer import GPUOptimizer, AsyncEvaluator, StateMemoryPool
from .csr_gpu_kernels import get_csr_kernels, CSRGPUKernels
from .optimized_cuda_kernels import OptimizedCUDAKernels
from .gpu_tree_kernels import GPUTreeKernels

# Optional GPU attack/defense module
try:
    from .gpu_attack_defense import gpu_compute_attack_defense_scores
    HAS_GPU_ATTACK_DEFENSE = True
except ImportError:
    gpu_compute_attack_defense_scores = None
    HAS_GPU_ATTACK_DEFENSE = False

__all__ = [
    "CSRTree",
    "CSRTreeConfig",
    "OptimizedWaveEngine",
    "OptimizedWaveConfig",
    "GPUOptimizer",
    "AsyncEvaluator",
    "StateMemoryPool",
    "get_csr_kernels",
    "CSRGPUKernels",
    "OptimizedCUDAKernels",
    "GPUTreeKernels",
]

if HAS_GPU_ATTACK_DEFENSE:
    __all__.append("gpu_compute_attack_defense_scores")