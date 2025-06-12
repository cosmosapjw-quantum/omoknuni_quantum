#!/usr/bin/env python3
"""Check if CUDA kernels are properly loaded"""

import torch
from mcts.gpu.unified_kernels import get_unified_kernels

def check_cuda_kernels():
    print("Checking CUDA Kernels")
    print("=" * 60)
    
    # Get kernels
    kernels = get_unified_kernels(torch.device('cuda'))
    
    print(f"CUDA kernels loaded: {kernels.use_cuda}")
    print(f"Device: {kernels.device}")
    
    # Check internal state
    from mcts.gpu.unified_kernels import _UNIFIED_KERNELS, _KERNELS_AVAILABLE
    print(f"\n_KERNELS_AVAILABLE: {_KERNELS_AVAILABLE}")
    print(f"_UNIFIED_KERNELS: {_UNIFIED_KERNELS}")
    
    if _UNIFIED_KERNELS:
        print(f"\nAvailable functions:")
        for attr in dir(_UNIFIED_KERNELS):
            if not attr.startswith('_') and callable(getattr(_UNIFIED_KERNELS, attr)):
                print(f"  - {attr}")
    
    # Test UCB selection with parent_visit=0
    print("\nTesting UCB selection with parent_visit=0:")
    
    # Create test data
    batch_size = 4
    num_children = 8
    
    # Simple test case
    node_indices = torch.tensor([0, 0, 0, 0], device='cuda', dtype=torch.int32)
    row_ptr = torch.tensor([0, 8, 16], device='cuda', dtype=torch.int32)
    col_indices = torch.arange(16, device='cuda', dtype=torch.int32)
    edge_actions = torch.arange(16, device='cuda', dtype=torch.int32)
    edge_priors = torch.ones(16, device='cuda', dtype=torch.float32) / 8
    visit_counts = torch.zeros(16, device='cuda', dtype=torch.int32)
    value_sums = torch.zeros(16, device='cuda', dtype=torch.float32)
    
    # Call UCB selection
    selected_actions, selected_scores = kernels.batch_ucb_selection(
        node_indices,
        row_ptr,
        col_indices,
        edge_actions,
        edge_priors,
        visit_counts,
        value_sums,
        c_puct=1.414,
        temperature=1.0
    )
    
    print(f"\nResults:")
    print(f"  Selected actions: {selected_actions.cpu().numpy()}")
    print(f"  Selected scores: {selected_scores.cpu().numpy()}")
    
    # Check if it's using CUDA kernel or fallback
    if kernels.stats['ucb_calls'] > 0:
        print(f"\nKernel stats:")
        print(f"  UCB calls: {kernels.stats['ucb_calls']}")
        print(f"  Total nodes processed: {kernels.stats['total_nodes_processed']}")

if __name__ == "__main__":
    check_cuda_kernels()