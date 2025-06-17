"""CUDA setup module to ensure GCC 12 is used for compilation"""

import os

def setup_cuda_env():
    """Set up CUDA environment to use GCC 12"""
    # Set GCC 12 as host compiler
    os.environ['CUDAHOSTCXX'] = 'g++-12'
    os.environ['CUDACXX'] = 'g++-12'
    
    # Set CUDA architecture for RTX 3060 Ti
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
    
    # Memory allocation config
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # nvcc flags
    os.environ['NVCC_APPEND_FLAGS'] = '-ccbin g++-12'
    
    # Suppress GCC version warnings
    os.environ['TORCH_CUDA_CC'] = 'g++-12'

# Auto-setup when imported
setup_cuda_env()