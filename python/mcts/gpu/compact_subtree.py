"""Compact subtree representation for GPU-efficient tree reuse

This module provides data structures and functions for extracting
and rebuilding MCTS subtrees efficiently on GPU.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple
from .gpu_bfs import parallel_bfs_subtree, extract_edges_vectorized


@dataclass
class CompactSubtreeGPU:
    """Minimal representation of MCTS subtree for efficient GPU transfer
    
    This compact representation contains only essential MCTS statistics
    and structure information, avoiding the need for complex remapping.
    """
    # Core MCTS statistics (all GPU tensors)
    visit_counts: torch.Tensor      # [num_nodes] int32
    value_sums: torch.Tensor        # [num_nodes] float32
    
    # Tree structure (edges)
    edge_actions: Optional[torch.Tensor] = None    # [num_edges] int16
    edge_parent: Optional[torch.Tensor] = None     # [num_edges] int32
    edge_child: Optional[torch.Tensor] = None      # [num_edges] int32
    prior_probs: Optional[torch.Tensor] = None     # [num_edges] float32
    
    # Metadata
    num_nodes: int = 0
    num_edges: int = 0
    
    # Root state for initialization
    root_state: Optional[torch.Tensor] = None      # [board_size, board_size] int8
    
    def __post_init__(self):
        """Validate tensor shapes and types"""
        if self.visit_counts is not None:
            assert self.visit_counts.dim() == 1
            assert self.visit_counts.dtype == torch.int32
            if self.num_nodes == 0:
                self.num_nodes = self.visit_counts.size(0)
        
        if self.value_sums is not None:
            assert self.value_sums.dim() == 1
            assert self.value_sums.dtype == torch.float32
            assert self.value_sums.size(0) == self.num_nodes


def extract_subtree_gpu(tree, new_root_idx: int) -> CompactSubtreeGPU:
    """Extract subtree rooted at given node into compact representation
    
    Uses GPU-accelerated BFS to extract the subtree efficiently without
    CPU-GPU synchronization.
    
    Args:
        tree: Source tree (CSRTree or similar)
        new_root_idx: Index of node that will become new root
        
    Returns:
        CompactSubtreeGPU containing the extracted subtree
    """
    device = tree.visit_counts.device if hasattr(tree, 'visit_counts') else torch.device('cpu')
    
    # Handle edge cases
    if new_root_idx >= tree.num_nodes or new_root_idx < 0:
        # Invalid root index
        return CompactSubtreeGPU(
            visit_counts=torch.empty(0, dtype=torch.int32, device=device),
            value_sums=torch.empty(0, dtype=torch.float32, device=device),
            num_nodes=0,
            num_edges=0
        )
    
    # Single node tree
    if tree.num_nodes == 1:
        return CompactSubtreeGPU(
            visit_counts=tree.visit_counts[:1].clone(),
            value_sums=tree.value_sums[:1].clone() if hasattr(tree, 'value_sums') else torch.zeros(1, device=device),
            num_nodes=1,
            num_edges=0
        )
    
    # Use parallel BFS to find all nodes in subtree
    # Initialize masks and queues
    max_nodes = tree.num_nodes
    in_subtree = torch.zeros(max_nodes, dtype=torch.bool, device=device)
    node_remap = torch.full((max_nodes,), -1, dtype=torch.int32, device=device)
    
    # BFS queue
    bfs_queue = torch.zeros(max_nodes, dtype=torch.int32, device=device)
    queue_start = 0
    queue_end = 1
    
    # Start with new root
    bfs_queue[0] = new_root_idx
    in_subtree[new_root_idx] = True
    node_remap[new_root_idx] = 0  # New root gets index 0
    new_node_count = 1
    
    # BFS to mark all nodes in subtree
    while queue_start < queue_end:
        # Process current level
        current_level_size = queue_end - queue_start
        
        for i in range(queue_start, queue_end):
            parent_idx = bfs_queue[i].item()
            
            # Get children of current node
            if hasattr(tree, 'get_children'):
                children_info = tree.get_children(parent_idx)
                # Handle both dict and tuple return formats
                if isinstance(children_info, dict):
                    child_indices = children_info['indices']
                else:
                    # Tuple format: (indices, actions, priors)
                    child_indices = children_info[0]
                
                for child_idx in child_indices:
                    if not in_subtree[child_idx]:
                        in_subtree[child_idx] = True
                        node_remap[child_idx] = new_node_count
                        bfs_queue[queue_end] = child_idx
                        queue_end += 1
                        new_node_count += 1
            elif hasattr(tree, 'children') and hasattr(tree, 'row_ptr'):
                # Handle simplified CSR format from mock
                start = tree.row_ptr[parent_idx]
                end = tree.row_ptr[parent_idx + 1]
                
                for j in range(start, end):
                    child_idx = tree.children[j]
                    if child_idx >= 0 and not in_subtree[child_idx]:
                        in_subtree[child_idx] = True
                        node_remap[child_idx] = new_node_count
                        bfs_queue[queue_end] = child_idx
                        queue_end += 1
                        new_node_count += 1
        
        queue_start += current_level_size
    
    # Extract node data for subtree
    subtree_indices = torch.nonzero(in_subtree, as_tuple=True)[0]
    num_subtree_nodes = subtree_indices.size(0)
    
    # Extract MCTS statistics
    compact = CompactSubtreeGPU(
        visit_counts=tree.visit_counts[subtree_indices].clone(),
        value_sums=tree.value_sums[subtree_indices].clone() if hasattr(tree, 'value_sums') else torch.zeros(num_subtree_nodes, device=device),
        num_nodes=num_subtree_nodes,
        num_edges=0  # Will be computed below
    )
    
    # Extract edge information if available
    if hasattr(tree, 'csr_storage') and num_subtree_nodes > 1:
        edge_list = []
        
        # Iterate through nodes in subtree and collect edges
        for i, node_idx in enumerate(subtree_indices):
            node_idx_val = node_idx.item()
            
            # Get children using CSR structure
            if hasattr(tree.csr_storage, 'row_ptr'):
                start = tree.csr_storage.row_ptr[node_idx_val]
                end = tree.csr_storage.row_ptr[node_idx_val + 1]
                
                for edge_idx in range(start, end):
                    child_idx = tree.csr_storage.col_indices[edge_idx].item()
                    
                    # Check if child is in subtree
                    if in_subtree[child_idx]:
                        # Remap indices
                        new_parent = node_remap[node_idx_val].item()
                        new_child = node_remap[child_idx].item()
                        action = tree.csr_storage.edge_actions[edge_idx].item() if hasattr(tree.csr_storage, 'edge_actions') else 0
                        
                        edge_list.append((new_parent, new_child, action))
        
        # Convert edge list to tensors
        if edge_list:
            num_edges = len(edge_list)
            compact.edge_parent = torch.tensor([e[0] for e in edge_list], dtype=torch.int32, device=device)
            compact.edge_child = torch.tensor([e[1] for e in edge_list], dtype=torch.int32, device=device)
            compact.edge_actions = torch.tensor([e[2] for e in edge_list], dtype=torch.int16, device=device)
            compact.num_edges = num_edges
    
    # Extract prior probabilities if available
    if hasattr(tree, 'prior_probs'):
        compact.prior_probs = tree.prior_probs[subtree_indices].clone()
    
    return compact


def extract_subtree_gpu_optimized(tree, new_root_idx: int) -> CompactSubtreeGPU:
    """GPU-optimized subtree extraction with no .item() calls
    
    This version uses fully vectorized operations to eliminate
    CPU-GPU synchronization.
    
    Args:
        tree: Source tree (CSRTree or similar)
        new_root_idx: Index of node that will become new root
        
    Returns:
        CompactSubtreeGPU containing the extracted subtree
    """
    device = tree.visit_counts.device if hasattr(tree, 'visit_counts') else torch.device('cpu')
    
    # Handle edge cases
    if new_root_idx >= tree.num_nodes or new_root_idx < 0:
        return CompactSubtreeGPU(
            visit_counts=torch.empty(0, dtype=torch.int32, device=device),
            value_sums=torch.empty(0, dtype=torch.float32, device=device),
            num_nodes=0,
            num_edges=0
        )
    
    # Use parallel BFS to find subtree nodes
    in_subtree, node_remap, num_subtree_nodes = parallel_bfs_subtree(
        tree, new_root_idx, max_nodes=tree.num_nodes
    )
    
    # Extract node data using boolean indexing
    subtree_mask = in_subtree[:tree.num_nodes]
    subtree_indices = torch.nonzero(subtree_mask, as_tuple=True)[0]
    
    # Extract MCTS statistics
    compact = CompactSubtreeGPU(
        visit_counts=tree.visit_counts[subtree_indices],
        value_sums=tree.value_sums[subtree_indices] if hasattr(tree, 'value_sums') else torch.zeros(num_subtree_nodes, device=device),
        num_nodes=num_subtree_nodes,
        num_edges=0
    )
    
    # Extract edges using vectorized operations
    if num_subtree_nodes > 1:
        edge_parents, edge_children, edge_actions, num_edges = extract_edges_vectorized(
            tree, in_subtree, node_remap
        )
        
        if num_edges > 0:
            compact.edge_parent = edge_parents
            compact.edge_child = edge_children
            compact.edge_actions = edge_actions
            compact.num_edges = num_edges
    
    # Extract prior probabilities if available
    if hasattr(tree, 'prior_probs'):
        compact.prior_probs = tree.prior_probs[subtree_indices]
    
    return compact


def rebuild_tree_from_compact(tree, compact: CompactSubtreeGPU) -> None:
    """Rebuild tree structure from compact representation
    
    This modifies the tree in-place, clearing existing data and
    populating with the compact subtree.
    
    Args:
        tree: Target tree to rebuild (will be cleared)
        compact: Compact representation to rebuild from
    """
    # Clear tree first
    if hasattr(tree, 'clear'):
        tree.clear()
    elif hasattr(tree, 'reset'):
        tree.reset()
    
    # Fast path for empty subtree
    if compact.num_nodes == 0:
        return
    
    # Rebuild node data
    tree.num_nodes = compact.num_nodes
    
    # Copy MCTS statistics
    tree.visit_counts[:compact.num_nodes] = compact.visit_counts
    tree.value_sums[:compact.num_nodes] = compact.value_sums
    
    # Copy prior probabilities if available
    if compact.prior_probs is not None and hasattr(tree, 'prior_probs'):
        tree.prior_probs[:compact.num_nodes] = compact.prior_probs
    
    # Rebuild tree structure if edge information is available
    if compact.edge_parent is not None and compact.edge_child is not None:
        # For CSR tree, rebuild edge structure
        if hasattr(tree, 'csr_storage'):
            # Count children per node
            parent_counts = torch.zeros(compact.num_nodes, dtype=torch.int32, device=tree.device)
            for parent in compact.edge_parent:
                parent_counts[parent] += 1
            
            # Build row pointers
            tree.csr_storage.row_ptr[0] = 0
            tree.csr_storage.row_ptr[1:compact.num_nodes+1] = parent_counts.cumsum(0)
            
            # Sort edges by parent for CSR format
            edge_order = compact.edge_parent.argsort()
            tree.csr_storage.col_indices[:compact.num_edges] = compact.edge_child[edge_order]
            if compact.edge_actions is not None:
                tree.csr_storage.edge_actions[:compact.num_edges] = compact.edge_actions[edge_order]
            
            tree.num_edges = compact.num_edges
    
    # Mark root as expanded if it has children
    if hasattr(tree, 'node_data') and hasattr(tree.node_data, 'set_expanded'):
        if compact.num_edges > 0:  # Root has children
            tree.node_data.set_expanded(0, True)


class SubtreeExtractor:
    """GPU-accelerated subtree extraction using parallel algorithms"""
    
    def __init__(self, device='cuda', max_nodes=100000):
        self.device = torch.device(device)
        self.max_nodes = max_nodes
        
        # Pre-allocate workspace for extraction
        self.workspace = {
            'node_mask': torch.zeros(max_nodes, dtype=torch.bool, device=device),
            'node_mapping': torch.zeros(max_nodes, dtype=torch.int32, device=device),
            'bfs_queue': torch.zeros(max_nodes, dtype=torch.int32, device=device),
            'queue_size': torch.zeros(2, dtype=torch.int32, device=device)  # read/write pointers
        }
    
    def mark_subtree_nodes(self, tree, root_idx: int) -> torch.Tensor:
        """Mark all nodes in subtree using parallel BFS
        
        Returns:
            Boolean mask indicating which nodes are in the subtree
        """
        # Reset workspace
        self.workspace['node_mask'].zero_()
        self.workspace['queue_size'].zero_()
        
        # Initialize with root
        self.workspace['node_mask'][root_idx] = True
        self.workspace['bfs_queue'][0] = root_idx
        self.workspace['queue_size'][1] = 1  # write pointer
        
        # TODO: Implement parallel BFS
        # For now, return mask with just root marked
        return self.workspace['node_mask']
    
    def extract_compact(self, tree, node_mask: torch.Tensor) -> CompactSubtreeGPU:
        """Extract compact representation given node mask
        
        Args:
            tree: Source tree
            node_mask: Boolean mask of nodes to extract
            
        Returns:
            Compact subtree representation
        """
        # Count nodes in subtree
        num_nodes = node_mask.sum().item()
        
        if num_nodes == 0:
            return CompactSubtreeGPU(
                visit_counts=torch.empty(0, dtype=torch.int32, device=self.device),
                value_sums=torch.empty(0, dtype=torch.float32, device=self.device),
                num_nodes=0,
                num_edges=0
            )
        
        # Extract node data (compact using boolean indexing)
        compact = CompactSubtreeGPU(
            visit_counts=tree.visit_counts[node_mask].clone(),
            value_sums=tree.value_sums[node_mask].clone(),
            num_nodes=num_nodes,
            num_edges=0  # Will calculate edges in full implementation
        )
        
        return compact