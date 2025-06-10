#!/usr/bin/env python3
"""Build script for CUDA kernels using setuptools

This script compiles the CUDA kernels into a Python extension module.
"""

from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os
from pathlib import Path
import warnings

# Suppress version bounds warning for CUDA 12.8
warnings.filterwarnings('ignore', message='There are no .* version bounds defined for CUDA version')

# Get paths
current_dir = Path(__file__).parent
gpu_dir = current_dir / "mcts" / "gpu"
build_dir = current_dir / "build_cuda"
build_dir.mkdir(exist_ok=True)

# Find CUDA source files
cuda_sources = []

# Check for unified kernel first
if (gpu_dir / "unified_cuda_kernels.cu").exists():
    cuda_sources.append(str(gpu_dir / "unified_cuda_kernels.cu"))
elif (gpu_dir / "custom_cuda_kernels_optimized.cu").exists():
    cuda_sources.append(str(gpu_dir / "custom_cuda_kernels_optimized.cu"))
elif (gpu_dir / "custom_cuda_kernels.cu").exists():
    cuda_sources.append(str(gpu_dir / "custom_cuda_kernels.cu"))

# Add wrapper if exists
wrapper_path = gpu_dir / "cuda_ops_wrapper.cpp"
if wrapper_path.exists():
    cuda_sources.append(str(wrapper_path))

# Filter to only existing files
cuda_sources = [src for src in cuda_sources if Path(src).exists()]

if not cuda_sources:
    print("Error: No CUDA source files found!")
    exit(1)

print(f"Found CUDA sources: {cuda_sources}")

# Get CUDA compute capability
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability()
    arch_list = [f'{capability[0]}.{capability[1]}']
    print(f"Building for GPU compute capability: {capability[0]}.{capability[1]}")
else:
    # Default to common architectures
    arch_list = ['7.5', '8.0', '8.6']  # Turing, Ampere, RTX 3060 Ti
    print("No GPU detected, building for architectures: " + ', '.join(arch_list))

# Set environment variables for faster compilation
os.environ['MAX_JOBS'] = '4'
os.environ['TORCH_CUDA_ARCH_LIST'] = ';'.join(arch_list)

# Define the extension
ext_modules = [
    CUDAExtension(
        name='mcts.gpu.mcts_cuda_kernels',
        sources=cuda_sources,
        include_dirs=[str(gpu_dir)],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '--expt-relaxed-constexpr',
                '-gencode', f'arch=compute_{int(float(arch_list[0]) * 10)},code=sm_{int(float(arch_list[0]) * 10)}',
            ]
        },
        libraries=[],
        library_dirs=[],
    )
]

# Run setup
setup(
    name='mcts_cuda_kernels',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)

print("CUDA kernels compiled successfully!")