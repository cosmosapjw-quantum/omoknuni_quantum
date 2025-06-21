#!/bin/bash
# CUDA Configuration for AlphaZero Omoknuni
# This file sets up the CUDA environment

# Set CUDA architecture for RTX 3060 Ti
export TORCH_CUDA_ARCH_LIST="8.6"

# Memory allocation config to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "CUDA environment configured:"
echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

# Verify system compiler is available
if command -v g++ &> /dev/null; then
    echo "✓ System C++ compiler found at: $(which g++)"
    g++ --version | head -1
else
    echo "❌ ERROR: System C++ compiler not found! Please install with: sudo apt install build-essential"
    exit 1
fi

# Verify CUDA is available
if command -v nvcc &> /dev/null; then
    echo "✓ CUDA found at: $(which nvcc)"
    nvcc --version | grep release
else
    echo "⚠ WARNING: nvcc not found in PATH"
fi