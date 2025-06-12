#!/usr/bin/env python3
"""Simple test of CSR tree structure"""

import torch
from mcts.gpu.csr_tree import CSRTree, CSRTreeConfig

def test_csr_tree():
    print("Testing CSR Tree")
    print("=" * 60)
    
    # Create config
    config = CSRTreeConfig(
        max_nodes=1000,
        max_edges=10000,
        device='cuda',
        enable_batched_ops=False
    )
    
    # Create tree
    tree = CSRTree(config)
    print(f"Initial tree state:")
    print(f"  num_nodes: {tree.num_nodes}")
    print(f"  num_edges: {tree.num_edges}")
    print(f"  row_ptr[:5]: {tree.row_ptr[:5].cpu().numpy()}")
    
    # Add children to root
    print("\nAdding 8 children to root...")
    actions = [0, 1, 2, 3, 4, 5, 6, 7]
    priors = [0.125] * 8
    child_indices = tree.add_children_batch(0, actions, priors)
    print(f"Child indices: {child_indices}")
    
    print(f"\nAfter adding children:")
    print(f"  num_nodes: {tree.num_nodes}")
    print(f"  num_edges: {tree.num_edges}")
    print(f"  row_ptr[:10]: {tree.row_ptr[:10].cpu().numpy()}")
    print(f"  col_indices[:10]: {tree.col_indices[:10].cpu().numpy()}")
    print(f"  _needs_row_ptr_update: {tree._needs_row_ptr_update}")
    
    # Ensure consistent
    print("\nCalling ensure_consistent()...")
    tree.ensure_consistent()
    
    print(f"\nAfter ensure_consistent:")
    print(f"  row_ptr[:10]: {tree.row_ptr[:10].cpu().numpy()}")
    print(f"  _needs_row_ptr_update: {tree._needs_row_ptr_update}")
    
    # Test UCB selection
    print("\nTesting UCB selection on root...")
    node_indices = torch.tensor([0, 0, 0, 0], device='cuda', dtype=torch.int32)
    
    # Direct call to batch_select_ucb_optimized
    actions, scores = tree.batch_select_ucb_optimized(node_indices, c_puct=1.414)
    print(f"Selected actions: {actions.cpu().numpy()}")
    print(f"Selected scores: {scores.cpu().numpy()}")
    
    # Check root's CSR data
    start = tree.row_ptr[0].item()
    end = tree.row_ptr[1].item()
    print(f"\nRoot CSR data:")
    print(f"  Start: {start}, End: {end}")
    if end > start:
        print(f"  Children: {tree.col_indices[start:end].cpu().numpy()}")

if __name__ == "__main__":
    test_csr_tree()