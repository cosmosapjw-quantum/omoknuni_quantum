#!/bin/bash
# CUDA Configuration for AlphaZero Omoknuni
# This file sets up the environment to use GCC 12 for CUDA compilation

# Set GCC 12 as the host compiler for CUDA
export CUDAHOSTCXX=g++-12
export CUDACXX=g++-12

# Set CUDA architecture for RTX 3060 Ti
export TORCH_CUDA_ARCH_LIST="8.6"

# Memory allocation config to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Set nvcc flags to use GCC 12
export NVCC_APPEND_FLAGS="-ccbin g++-12"

echo "CUDA environment configured for GCC 12:"
echo "  CUDAHOSTCXX=$CUDAHOSTCXX"
echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "  NVCC_APPEND_FLAGS=$NVCC_APPEND_FLAGS"

# Verify GCC 12 is available
if command -v g++-12 &> /dev/null; then
    echo "✓ GCC 12 found at: $(which g++-12)"
    g++-12 --version | head -1
else
    echo "❌ ERROR: GCC 12 not found! Please install with: sudo apt install g++-12"
    exit 1
fi

# Verify CUDA is available
if command -v nvcc &> /dev/null; then
    echo "✓ CUDA found at: $(which nvcc)"
    nvcc --version | grep release
else
    echo "⚠ WARNING: nvcc not found in PATH"
fi