#!/bin/bash
# Set CUDA architecture for RTX 3060 Ti (sm_86)
export TORCH_CUDA_ARCH_LIST="8.6"

# Also set memory allocation config to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "CUDA environment variables set:"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"