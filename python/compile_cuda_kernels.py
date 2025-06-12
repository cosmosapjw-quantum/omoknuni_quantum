#!/usr/bin/env python3
"""Compile CUDA kernels for the unified MCTS implementation"""

import os
import sys
import torch
from torch.utils.cpp_extension import load

def compile_cuda_kernels():
    """Compile the unified CUDA kernels"""
    print("Compiling CUDA kernels...")
    
    # Get paths
    gpu_dir = os.path.join(os.path.dirname(__file__), 'mcts', 'gpu')
    cuda_file = os.path.join(gpu_dir, 'unified_cuda_kernels.cu')
    
    if not os.path.exists(cuda_file):
        print(f"Error: CUDA file not found at {cuda_file}")
        return False
        
    try:
        # JIT compile the kernels
        mcts_cuda_kernels = load(
            name='mcts_cuda_kernels',
            sources=[cuda_file],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=True
        )
        
        print("\nSuccessfully compiled CUDA kernels!")
        print("Available functions:")
        for attr in dir(mcts_cuda_kernels):
            if not attr.startswith('_'):
                print(f"  - {attr}")
                
        return True
        
    except Exception as e:
        print(f"Error compiling CUDA kernels: {e}")
        return False


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping kernel compilation.")
        sys.exit(0)
        
    success = compile_cuda_kernels()
    sys.exit(0 if success else 1)