#!/usr/bin/env python3
"""Test quantum kernel loading"""

# Force reload of the module
import sys
if 'mcts.gpu.unified_kernels' in sys.modules:
    del sys.modules['mcts.gpu.unified_kernels']

import torch
from mcts.gpu.unified_kernels import get_unified_kernels, _UNIFIED_KERNELS, _KERNELS_AVAILABLE

print("Testing quantum kernel loading")
print("=" * 60)

# Get kernels
kernels = get_unified_kernels(torch.device('cuda'))

print(f"\n_KERNELS_AVAILABLE: {_KERNELS_AVAILABLE}")
print(f"kernels.use_cuda: {kernels.use_cuda}")

if _UNIFIED_KERNELS:
    print(f"\nLoaded module: {_UNIFIED_KERNELS}")
    funcs = [attr for attr in dir(_UNIFIED_KERNELS) if not attr.startswith('_') and callable(getattr(_UNIFIED_KERNELS, attr))]
    print(f"\nAvailable functions ({len(funcs)}):")
    for f in sorted(funcs):
        print(f"  - {f}")
    
    # Check for quantum function
    has_quantum = hasattr(_UNIFIED_KERNELS, 'batched_ucb_selection_quantum')
    print(f"\nHas batched_ucb_selection_quantum: {has_quantum}")

# Test quantum kernel call
if hasattr(_UNIFIED_KERNELS, 'batched_ucb_selection_quantum'):
    print("\nTesting quantum kernel call...")
    
    # Create test data
    batch_size = 4
    num_nodes = 10
    num_edges = 8
    
    q_values = torch.rand(num_nodes, device='cuda', dtype=torch.float32)
    visit_counts = torch.randint(0, 10, (num_nodes,), device='cuda', dtype=torch.int32)
    parent_visits = torch.ones(batch_size, device='cuda', dtype=torch.int32)
    priors = torch.ones(num_edges, device='cuda', dtype=torch.float32) / num_edges
    row_ptr = torch.tensor([0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], device='cuda', dtype=torch.int32)
    col_indices = torch.arange(1, 9, device='cuda', dtype=torch.int32)
    
    # Quantum parameters
    quantum_phases = torch.zeros(num_edges, device='cuda', dtype=torch.float32)
    uncertainty_table = torch.ones(10000, device='cuda', dtype=torch.float32) * 0.1
    
    # Call kernel
    actions, scores = kernels.batch_ucb_selection(
        torch.tensor([0, 0, 0, 0], device='cuda', dtype=torch.int32),  # node_indices
        row_ptr,
        col_indices,
        torch.arange(num_edges, device='cuda', dtype=torch.int32),  # edge_actions
        priors,
        visit_counts,
        q_values * visit_counts,  # value_sums
        1.414,  # c_puct
        1.0,    # temperature
        quantum_phases=quantum_phases,
        uncertainty_table=uncertainty_table,
        enable_quantum=True
    )
    
    print(f"\nQuantum kernel call successful!")
    print(f"  Actions: {actions.cpu().numpy()}")
    print(f"  Scores: {scores.cpu().numpy()}")
    print(f"  Quantum calls in stats: {kernels.stats.get('quantum_calls', 0)}")