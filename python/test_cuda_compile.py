#!/usr/bin/env python3
"""Test CUDA kernel compilation"""

import os
import sys
import torch

# Ensure we don't hang
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# Try importing with CUDA compilation
print("\nTesting CUDA compilation...")
try:
    import mcts.gpu.cuda_compile
    print(f"CUDA kernels available: {mcts.gpu.cuda_compile.CUDA_KERNELS_AVAILABLE}")
    
    if mcts.gpu.cuda_compile.CUDA_KERNELS_AVAILABLE:
        print("✓ Custom CUDA kernels compiled successfully!")
        
        # Test the kernels
        print("\nTesting kernels...")
        device = torch.device('cuda')
        
        # Test data
        n = 10
        q_values = torch.randn(n, device=device)
        visit_counts = torch.randint(0, 100, (n,), device=device, dtype=torch.int32)
        parent_visits = torch.randint(1, 100, (n,), device=device, dtype=torch.int32)
        priors = torch.rand(n, device=device)
        row_ptr = torch.arange(n+1, device=device, dtype=torch.int32)
        col_indices = torch.arange(n, device=device, dtype=torch.int32)
        
        # Test UCB selection
        result = mcts.gpu.cuda_compile.batched_ucb_selection(
            q_values, visit_counts, parent_visits, priors, 
            row_ptr, col_indices, 1.414
        )
        print(f"UCB selection result shape: {result.shape}")
        print("✓ UCB selection kernel works!")
        
    else:
        print("✗ CUDA kernels not available, using fallback")
        
except Exception as e:
    print(f"✗ Error during compilation: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")