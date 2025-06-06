#!/usr/bin/env python3
"""Compile CUDA kernels offline to avoid hanging during runtime"""

import os
import torch
from torch.utils.cpp_extension import load

print("Compiling custom CUDA kernels offline...")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("CUDA not available, skipping compilation")
    exit(1)

# Set CUDA architecture
if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
    os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0;7.5;8.0;8.6'

# Paths
cuda_source = os.path.join('mcts', 'gpu', 'custom_cuda_kernels.cu')
build_dir = os.path.join(os.getcwd(), 'build_cuda')

print(f"CUDA source: {cuda_source}")
print(f"Build directory: {build_dir}")

# Create build directory
os.makedirs(build_dir, exist_ok=True)

# C++ wrapper
cpp_source = """
#include <torch/extension.h>

// Forward declarations
torch::Tensor batched_ucb_selection_cuda(
    torch::Tensor q_values,
    torch::Tensor visit_counts,
    torch::Tensor parent_visits,
    torch::Tensor priors,
    torch::Tensor row_ptr,
    torch::Tensor col_indices,
    float c_puct
);

torch::Tensor parallel_backup_cuda(
    torch::Tensor paths,
    torch::Tensor leaf_values,
    torch::Tensor path_lengths,
    torch::Tensor value_sums,
    torch::Tensor visit_counts
);

PYBIND11_MODULE(custom_cuda_ops, m) {
    m.def("batched_ucb_selection", &batched_ucb_selection_cuda);
    m.def("parallel_backup", &parallel_backup_cuda);
}
"""

# Write wrapper
cpp_file = os.path.join(build_dir, 'wrapper.cpp')
with open(cpp_file, 'w') as f:
    f.write(cpp_source)

print("\nCompiling...")
try:
    # Compile with explicit settings
    custom_cuda_ops = load(
        name='custom_cuda_ops',
        sources=[cpp_file, cuda_source],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        build_directory=build_dir,
        verbose=True
    )
    
    print("\n✓ Compilation successful!")
    print(f"Module saved to: {build_dir}")
    
    # Copy to the mcts/gpu directory
    import shutil
    import glob
    
    # Find the compiled module
    so_files = glob.glob(os.path.join(build_dir, '*.so'))
    if so_files:
        target = os.path.join('mcts', 'gpu', 'custom_cuda_ops.so')
        shutil.copy(so_files[0], target)
        print(f"✓ Copied module to: {target}")
        print("\nTo use custom kernels, set: export USE_CUSTOM_CUDA_KERNELS=1")
    
except Exception as e:
    print(f"\n✗ Compilation failed: {e}")
    import traceback
    traceback.print_exc()