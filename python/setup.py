import os
from setuptools import setup, find_packages

# Handle torch import for build process
try:
    import torch
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Dummy classes for when torch is not available
    class CUDAExtension:
        def __init__(self, *args, **kwargs):
            pass
    class BuildExtension:
        def __init__(self, *args, **kwargs):
            pass

with open("requirements.txt") as f:
    requirements = f.read().splitlines()
    # Remove comments and empty lines
    requirements = [r for r in requirements if r and not r.startswith("#")]

# Determine if CUDA is available
USE_CUDA = TORCH_AVAILABLE and torch.cuda.is_available()

# Define CUDA extensions
ext_modules = []
if USE_CUDA:
    # Get CUDA compute capabilities
    compute_caps = []
    if TORCH_AVAILABLE:
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            compute_caps.append(f"compute_{major}{minor}")
            compute_caps.append(f"sm_{major}{minor}")
    
    # Remove duplicates and sort
    compute_caps = sorted(set(compute_caps))
    
    # Build gencode flags
    gencode_flags = []
    for cap in compute_caps:
        if cap.startswith("compute_"):
            gencode_flags.append(f"-gencode=arch={cap},code={cap}")
    
    # Add common architectures if none detected
    if not gencode_flags:
        gencode_flags = [
            '-gencode=arch=compute_70,code=sm_70',
            '-gencode=arch=compute_75,code=sm_75',
            '-gencode=arch=compute_80,code=sm_80',
            '-gencode=arch=compute_86,code=sm_86',
        ]
    
    # Quantum CUDA extension
    ext_modules.append(
        CUDAExtension(
            name='mcts.gpu.quantum_cuda_kernels',
            sources=['mcts/gpu/quantum_cuda_kernels.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math'] + gencode_flags
            }
        )
    )
    
    # Unified CUDA kernels extension
    ext_modules.append(
        CUDAExtension(
            name='mcts.gpu.unified_cuda_kernels',
            sources=['mcts/gpu/unified_cuda_kernels.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math'] + gencode_flags
            }
        )
    )
    
    # Quantum v5.0 CUDA kernels extension
    ext_modules.append(
        CUDAExtension(
            name='mcts.gpu.quantum_v5_cuda_kernels',
            sources=['mcts/gpu/quantum_v5_cuda_kernels.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math'] + gencode_flags
            }
        )
    )

setup(
    name="omoknuni-mcts",
    version="0.1.0",
    author="Omoknuni Team",
    description="Quantum-inspired Monte Carlo Tree Search for game AI",
    long_description=open("../docs/PRD.md").read() if os.path.exists("../docs/PRD.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "flake8>=6.1.0",
        ],
        "test": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-benchmark>=4.0.0",
        ],
        "cuda": [
            "ninja",  # For faster compilation
        ],
    },
    ext_modules=ext_modules if USE_CUDA else [],
    cmdclass={'build_ext': BuildExtension} if USE_CUDA and TORCH_AVAILABLE else {},
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Games/Entertainment :: Board Games",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: CUDA" if USE_CUDA else "",
    ],
)