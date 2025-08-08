"""GPU-accelerated BFS for tree operations

This module provides vectorized BFS implementations to eliminate
CPU-GPU synchronization from .item() calls.
"""

import torch
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import CUDA kernels
_cuda_kernels_available = False
try:
    from . import mcts_cuda_kernels
    _cuda_kernels_available = True
    logger.info("✅ CUDA BFS kernels available for acceleration")
except ImportError:
    logger.debug("CUDA BFS kernels not available, using PyTorch fallback")


def parallel_bfs_subtree(
    tree,
    new_root_idx: int,
    max_nodes: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Parallel BFS to extract subtree without .item() calls
    
    Args:
        tree: CSRTree instance
        new_root_idx: Index of node that will become new root
        max_nodes: Maximum nodes to process (defaults to tree.num_nodes)
        
    Returns:
        Tuple of:
        - in_subtree: Boolean mask of nodes in subtree
        - node_remap: Mapping from old to new indices (-1 if not in subtree)
        - num_subtree_nodes: Number of nodes in subtree
    """
    if max_nodes is None:
        max_nodes = tree.num_nodes
        
    device = tree.device
    
    # Use CUDA kernel if available for better performance
    if _cuda_kernels_available and hasattr(tree, 'csr_storage') and hasattr(tree.csr_storage, 'row_ptr'):
        try:
            in_subtree, node_remap, subtree_count = mcts_cuda_kernels.parallel_bfs_subtree(
                tree.csr_storage.row_ptr,
                tree.csr_storage.col_indices,
                root_idx=new_root_idx,
                max_nodes=max_nodes
            )
            logger.debug("Using CUDA BFS kernel for tree reuse")
            return in_subtree, node_remap, subtree_count.item()
        except Exception as e:
            logger.debug(f"CUDA BFS failed, falling back to PyTorch: {e}")
    
    # Initialize masks and mappings
    in_subtree = torch.zeros(max_nodes, dtype=torch.bool, device=device)
    node_remap = torch.full((max_nodes,), -1, dtype=torch.int32, device=device)
    
    # BFS frontier - use two buffers for current and next level
    current_level = torch.zeros(max_nodes, dtype=torch.int32, device=device)
    next_level = torch.zeros(max_nodes, dtype=torch.int32, device=device)
    
    # Initialize with root
    current_level[0] = new_root_idx
    current_size = 1
    in_subtree[new_root_idx] = True
    node_remap[new_root_idx] = 0
    new_node_count = 1
    
    # Process levels until frontier is empty
    while current_size > 0:
        # Reset next level
        next_size = 0
        
        # Process all nodes in current level in parallel
        if hasattr(tree, 'csr_storage') and hasattr(tree.csr_storage, 'row_ptr'):
            # Get CSR row pointers for current level nodes
            current_nodes = current_level[:current_size]
            
            # Vectorized child extraction
            row_starts = tree.csr_storage.row_ptr[current_nodes]
            row_ends = tree.csr_storage.row_ptr[current_nodes + 1]
            
            # Process each node's children
            for i in range(current_size):
                start = row_starts[i]
                end = row_ends[i]
                
                if start < end:
                    # Get children indices
                    children = tree.csr_storage.col_indices[start:end]
                    
                    # Mask for unvisited children
                    unvisited_mask = ~in_subtree[children]
                    unvisited_children = children[unvisited_mask]
                    
                    if unvisited_children.numel() > 0:
                        # Mark as visited
                        in_subtree[unvisited_children] = True
                        
                        # Assign new indices
                        num_new = unvisited_children.numel()
                        new_indices = torch.arange(
                            new_node_count, 
                            new_node_count + num_new,
                            device=device,
                            dtype=torch.int32
                        )
                        node_remap[unvisited_children] = new_indices
                        
                        # Add to next level
                        next_level[next_size:next_size + num_new] = unvisited_children
                        next_size += num_new
                        new_node_count += num_new
        
        # Swap buffers
        current_level, next_level = next_level, current_level
        current_size = next_size
    
    return in_subtree, node_remap, new_node_count


def extract_edges_vectorized(
    tree,
    in_subtree: torch.Tensor,
    node_remap: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Extract edges for subtree nodes using vectorized operations
    
    Args:
        tree: CSRTree instance
        in_subtree: Boolean mask of nodes in subtree
        node_remap: Mapping from old to new indices
        
    Returns:
        Tuple of (parent_indices, child_indices, actions, num_edges)
    """
    device = tree.device
    max_edges = tree.num_edges
    
    # Use CUDA kernel if available for better performance
    if _cuda_kernels_available and hasattr(tree, 'csr_storage') and hasattr(tree.csr_storage, 'row_ptr'):
        try:
            edge_parents, edge_children, edge_actions = mcts_cuda_kernels.extract_subtree_edges(
                tree.csr_storage.row_ptr,
                tree.csr_storage.col_indices,
                tree.csr_storage.edge_actions if hasattr(tree.csr_storage, 'edge_actions') else None,
                in_subtree,
                node_remap,
                max_edges=max_edges
            )
            return edge_parents, edge_children, edge_actions, edge_parents.shape[0]
        except Exception as e:
            logger.debug(f"CUDA edge extraction failed, falling back to PyTorch: {e}")
    
    # Pre-allocate edge arrays
    edge_parents = torch.zeros(max_edges, dtype=torch.int32, device=device)
    edge_children = torch.zeros(max_edges, dtype=torch.int32, device=device)
    edge_actions = torch.zeros(max_edges, dtype=torch.int16, device=device)
    
    # Get all nodes in subtree
    subtree_nodes = torch.nonzero(in_subtree, as_tuple=True)[0]
    
    # Process edges in batches
    edge_count = 0
    
    if hasattr(tree, 'csr_storage') and hasattr(tree.csr_storage, 'row_ptr'):
        # Get row pointers for all subtree nodes
        row_starts = tree.csr_storage.row_ptr[subtree_nodes]
        row_ends = tree.csr_storage.row_ptr[subtree_nodes + 1]
        
        # Process each node's edges
        for i, node_idx in enumerate(subtree_nodes):
            start = row_starts[i]
            end = row_ends[i]
            
            if start < end:
                # Get all children for this node
                children = tree.csr_storage.col_indices[start:end]
                
                # Filter children that are in subtree
                child_mask = in_subtree[children]
                valid_children = children[child_mask]
                
                if valid_children.numel() > 0:
                    # Get remapped indices
                    num_valid = valid_children.numel()
                    new_parent = node_remap[node_idx]
                    new_children = node_remap[valid_children]
                    
                    # Extract actions for valid edges
                    valid_indices = torch.nonzero(child_mask, as_tuple=True)[0] + start
                    valid_actions = tree.csr_storage.edge_actions[valid_indices]
                    
                    # Store edges
                    edge_slice = slice(edge_count, edge_count + num_valid)
                    edge_parents[edge_slice] = new_parent
                    edge_children[edge_slice] = new_children
                    edge_actions[edge_slice] = valid_actions
                    
                    edge_count += num_valid
    
    return edge_parents[:edge_count], edge_children[:edge_count], edge_actions[:edge_count], edge_count


def parallel_bfs_mark_subtree(
    row_ptr: torch.Tensor,
    col_indices: torch.Tensor,
    root_idx: int,
    max_nodes: int
) -> torch.Tensor:
    """Mark all nodes in subtree using parallel BFS (kernel-friendly version)
    
    This version is designed to be easily converted to a CUDA kernel.
    
    Args:
        row_ptr: CSR row pointers
        col_indices: CSR column indices
        root_idx: Root node of subtree
        max_nodes: Maximum number of nodes
        
    Returns:
        Boolean mask of nodes in subtree
    """
    device = row_ptr.device
    
    # Masks and queues
    in_subtree = torch.zeros(max_nodes, dtype=torch.bool, device=device)
    visited = torch.zeros(max_nodes, dtype=torch.bool, device=device)
    
    # Wave-based BFS
    current_wave = torch.zeros(max_nodes, dtype=torch.bool, device=device)
    next_wave = torch.zeros(max_nodes, dtype=torch.bool, device=device)
    
    # Initialize with root
    current_wave[root_idx] = True
    in_subtree[root_idx] = True
    visited[root_idx] = True
    
    # Process waves
    max_iterations = 100  # Prevent infinite loops
    
    for _ in range(max_iterations):
        # Check if current wave is empty
        if not current_wave.any():
            break
            
        # Find all children of current wave nodes
        wave_nodes = torch.nonzero(current_wave, as_tuple=True)[0]
        
        # Clear next wave
        next_wave.zero_()
        
        # Process each node in current wave
        for node in wave_nodes:
            start = row_ptr[node]
            end = row_ptr[node + 1]
            
            if start < end:
                children = col_indices[start:end]
                
                # Mark unvisited children
                unvisited = ~visited[children]
                new_children = children[unvisited]
                
                if new_children.numel() > 0:
                    next_wave[new_children] = True
                    in_subtree[new_children] = True
                    visited[new_children] = True
        
        # Swap waves
        current_wave, next_wave = next_wave, current_wave
    
    return in_subtree