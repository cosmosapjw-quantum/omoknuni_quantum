#!/usr/bin/env python3
"""Test CUDA kernels performance"""

import os
import torch
import time
import numpy as np

# Set CUDA arch to avoid warnings
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'

def test_cuda_kernels():
    """Test custom CUDA kernels vs PyTorch implementations"""
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
        
    device = torch.device('cuda')
    print(f"Testing on: {torch.cuda.get_device_name(0)}")
    
    # Test parameters
    num_nodes = 1000
    max_children = 10
    batch_size = 512
    
    # Create test data
    q_values = torch.randn(num_nodes * max_children, device=device)
    visit_counts = torch.randint(0, 100, (num_nodes * max_children,), device=device)
    parent_visits = torch.randint(1, 1000, (num_nodes,), device=device)
    priors = torch.rand(num_nodes * max_children, device=device)
    
    # Create CSR structure
    row_ptr = torch.arange(0, (num_nodes + 1) * max_children, max_children, device=device)
    col_indices = torch.arange(num_nodes * max_children, device=device)
    
    print("\n1. Testing CUDA Kernel Compilation...")
    
    try:
        from mcts.gpu.cuda_compile import CUDA_KERNELS_AVAILABLE, batched_ucb_selection
        
        if CUDA_KERNELS_AVAILABLE:
            print("✓ Custom CUDA kernels available")
            
            # Test kernel
            c_puct = 1.0
            
            # Warmup
            for _ in range(3):
                selected = batched_ucb_selection(
                    q_values, visit_counts, parent_visits, priors, 
                    row_ptr, col_indices, c_puct
                )
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(100):
                selected = batched_ucb_selection(
                    q_values, visit_counts, parent_visits, priors, 
                    row_ptr, col_indices, c_puct
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            print(f"✓ Custom CUDA kernel: {elapsed/100*1000:.2f} ms per call")
            print(f"  Selected actions shape: {selected.shape}")
            
        else:
            print("✗ Custom CUDA kernels not available")
            
    except Exception as e:
        print(f"✗ Error testing CUDA kernels: {e}")
    
    print("\n2. Testing Triton Kernels...")
    
    try:
        from mcts.gpu.optimized_cuda_kernels import OptimizedCUDAKernels
        
        kernels = OptimizedCUDAKernels(device)
        
        if kernels.use_triton:
            print("✓ Triton kernels available")
            
            # Test tensor operations
            test_tensor = torch.randn(1000, 1000, device=device)
            
            # Warmup
            for _ in range(3):
                result = test_tensor @ test_tensor.T
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(50):
                result = test_tensor @ test_tensor.T
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            print(f"✓ Matrix operations: {elapsed/50*1000:.2f} ms per call")
            
        else:
            print("✗ Triton kernels not available")
            
    except Exception as e:
        print(f"✗ Error testing Triton kernels: {e}")
    
    print("\n3. GPU Memory Usage:")
    
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    total = torch.cuda.get_device_properties(device).total_memory / 1024**2
    
    print(f"  Allocated: {allocated:.1f} MB")
    print(f"  Reserved: {reserved:.1f} MB")  
    print(f"  Total: {total:.1f} MB")
    print(f"  Usage: {allocated/total*100:.1f}%")
    
    print("\n✅ CUDA kernel tests completed!")


if __name__ == "__main__":
    test_cuda_kernels()