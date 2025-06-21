"""CUDA setup module for CUDA environment configuration"""

import os

def setup_cuda_env():
    """Set up CUDA environment"""
    # Let CUDA use system default compiler
    # CUDAHOSTCXX and CUDACXX will use defaults if not set
    
    # Set CUDA architecture for RTX 3060 Ti
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
    
    # Memory allocation config
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # nvcc will use default compiler settings
    # No need to override compiler choice

# Auto-setup when imported
setup_cuda_env()