"""Custom PyTorch operators for optimized CSR operations

This module implements custom operators for CSR tree operations that can be
fused by PyTorch's JIT compiler and optimized by torch.compile.
"""

import torch
import torch.library
from typing import Tuple, Optional

# Register custom operator library
LIBRARY_NAME = "csr_ops"

# Define operator schemas first
torch.library.define(
    "csr_ops::gather_children",
    "(Tensor node_indices, Tensor row_ptr, Tensor col_indices, Tensor edge_actions, int max_children) -> (Tensor, Tensor, Tensor)"
)

torch.library.define(
    "csr_ops::ucb_selection", 
    "(Tensor parent_indices, Tensor row_ptr, Tensor col_indices, Tensor edge_priors, Tensor node_visits, Tensor node_values, float c_puct) -> (Tensor, Tensor, Tensor)"
)

def _csr_gather_children_cpu(
    node_indices: torch.Tensor,
    row_ptr: torch.Tensor, 
    col_indices: torch.Tensor,
    edge_actions: torch.Tensor,
    max_children: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU implementation of CSR gather children"""
    batch_size = node_indices.size(0)
    device = node_indices.device
    
    # Initialize output tensors
    children = torch.full((batch_size, max_children), -1, device=device, dtype=torch.int32)
    actions = torch.full((batch_size, max_children), -1, device=device, dtype=torch.int32)
    valid_mask = torch.zeros((batch_size, max_children), device=device, dtype=torch.bool)
    
    # Process each node
    for i in range(batch_size):
        node_idx = node_indices[i].item()
        start_idx = row_ptr[node_idx].item()
        end_idx = row_ptr[node_idx + 1].item()
        num_children = min(end_idx - start_idx, max_children)
        
        if num_children > 0:
            children[i, :num_children] = col_indices[start_idx:start_idx + num_children]
            actions[i, :num_children] = edge_actions[start_idx:start_idx + num_children]
            valid_mask[i, :num_children] = True
    
    return children, actions, valid_mask

def _csr_gather_children_cuda(
    node_indices: torch.Tensor,
    row_ptr: torch.Tensor,
    col_indices: torch.Tensor, 
    edge_actions: torch.Tensor,
    max_children: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CUDA implementation of CSR gather children (optimized for coalescing)"""
    batch_size = node_indices.size(0)
    device = node_indices.device
    
    # Initialize output tensors with proper alignment
    children = torch.full((batch_size, max_children), -1, device=device, dtype=torch.int32)
    actions = torch.full((batch_size, max_children), -1, device=device, dtype=torch.int32)
    valid_mask = torch.zeros((batch_size, max_children), device=device, dtype=torch.bool)
    
    # Optimize for memory coalescing by processing in chunks
    # Use vectorized operations where possible
    start_indices = row_ptr[node_indices]
    end_indices = row_ptr[node_indices + 1]
    num_children_per_node = torch.minimum(end_indices - start_indices, 
                                         torch.tensor(max_children, device=device))
    
    # Find nodes with children to minimize work
    has_children_mask = num_children_per_node > 0
    nodes_with_children = torch.where(has_children_mask)[0]
    
    if len(nodes_with_children) == 0:
        return children, actions, valid_mask
    
    # Process nodes with children in coalesced manner
    # Group by similar start indices to improve cache locality
    active_start_indices = start_indices[nodes_with_children]
    active_num_children = num_children_per_node[nodes_with_children]
    
    # Sort by start indices for better memory access pattern
    sorted_indices = torch.argsort(active_start_indices)
    sorted_node_positions = nodes_with_children[sorted_indices]
    sorted_start_indices = active_start_indices[sorted_indices]
    sorted_num_children = active_num_children[sorted_indices]
    
    # Process nodes with children in coalesced manner
    for i, node_pos in enumerate(sorted_node_positions):
        start_idx = sorted_start_indices[i].item()
        num_children = min(sorted_num_children[i].item(), max_children)
        
        if num_children > 0:
            end_idx = start_idx + num_children
            
            # Coalesced memory access
            children[node_pos, :num_children] = col_indices[start_idx:end_idx]
            actions[node_pos, :num_children] = edge_actions[start_idx:end_idx]
            valid_mask[node_pos, :num_children] = True
    
    return children, actions, valid_mask

def _csr_ucb_selection_impl(
    parent_indices: torch.Tensor,
    row_ptr: torch.Tensor,
    col_indices: torch.Tensor,
    edge_priors: torch.Tensor,
    node_visits: torch.Tensor,
    node_values: torch.Tensor,
    c_puct: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Implementation of CSR UCB selection"""
    batch_size = parent_indices.size(0)
    device = parent_indices.device
    
    selected_children = torch.full((batch_size,), -1, device=device, dtype=torch.int32)
    selected_actions = torch.full((batch_size,), -1, device=device, dtype=torch.int32)
    max_ucb_scores = torch.full((batch_size,), -float('inf'), device=device, dtype=torch.float32)
    
    for i in range(batch_size):
        parent_idx = parent_indices[i].item()
        start_idx = row_ptr[parent_idx].item()
        end_idx = row_ptr[parent_idx + 1].item()
        num_children = end_idx - start_idx
        
        if num_children > 0:
            # Get children data
            child_indices = col_indices[start_idx:end_idx]
            priors = edge_priors[start_idx:end_idx]
            
            # Get child statistics
            child_visits = node_visits[child_indices].float()
            child_values = node_values[child_indices]
            parent_visits = node_visits[parent_idx].float()
            
            # Compute Q-values
            q_values = torch.where(child_visits > 0, child_values / child_visits, torch.zeros_like(child_values))
            
            # Compute UCB scores
            # FIXED: Remove epsilon to match CPU implementation exactly
            parent_sqrt = torch.sqrt(parent_visits)
            exploration = c_puct * priors * parent_sqrt / (1.0 + child_visits)
            ucb_scores = q_values + exploration
            
            # Select best child
            best_idx = torch.argmax(ucb_scores)
            selected_children[i] = child_indices[best_idx]
            selected_actions[i] = start_idx + best_idx  # Edge index
            max_ucb_scores[i] = ucb_scores[best_idx]
    
    return selected_children, selected_actions, max_ucb_scores

# Register the gather children operator
@torch.library.register_fake("csr_ops::gather_children")
def _gather_children_meta(
    node_indices: torch.Tensor,
    row_ptr: torch.Tensor,
    col_indices: torch.Tensor,
    edge_actions: torch.Tensor,
    max_children: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Meta function for shape inference"""
    batch_size = node_indices.size(0)
    device = node_indices.device
    
    children = torch.empty((batch_size, max_children), device=device, dtype=torch.int32)
    actions = torch.empty((batch_size, max_children), device=device, dtype=torch.int32)
    valid_mask = torch.empty((batch_size, max_children), device=device, dtype=torch.bool)
    
    return children, actions, valid_mask

@torch.library.register_kernel("csr_ops::gather_children", "cpu")
def _gather_children_cpu_kernel(
    node_indices: torch.Tensor,
    row_ptr: torch.Tensor,
    col_indices: torch.Tensor,
    edge_actions: torch.Tensor,
    max_children: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _csr_gather_children_cpu(node_indices, row_ptr, col_indices, edge_actions, max_children)

@torch.library.register_kernel("csr_ops::gather_children", "cuda")
def _gather_children_cuda_kernel(
    node_indices: torch.Tensor,
    row_ptr: torch.Tensor,
    col_indices: torch.Tensor,
    edge_actions: torch.Tensor,
    max_children: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _csr_gather_children_cuda(node_indices, row_ptr, col_indices, edge_actions, max_children)

# Register the UCB selection operator
@torch.library.register_fake("csr_ops::ucb_selection")
def _ucb_selection_meta(
    parent_indices: torch.Tensor,
    row_ptr: torch.Tensor,
    col_indices: torch.Tensor,
    edge_priors: torch.Tensor,
    node_visits: torch.Tensor,
    node_values: torch.Tensor,
    c_puct: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Meta function for UCB selection shape inference"""
    batch_size = parent_indices.size(0)
    device = parent_indices.device
    
    selected_children = torch.empty((batch_size,), device=device, dtype=torch.int32)
    selected_actions = torch.empty((batch_size,), device=device, dtype=torch.int32)
    ucb_scores = torch.empty((batch_size,), device=device, dtype=torch.float32)
    
    return selected_children, selected_actions, ucb_scores

@torch.library.register_kernel("csr_ops::ucb_selection", "cpu")
@torch.library.register_kernel("csr_ops::ucb_selection", "cuda")
def _ucb_selection_kernel(
    parent_indices: torch.Tensor,
    row_ptr: torch.Tensor,
    col_indices: torch.Tensor,
    edge_priors: torch.Tensor,
    node_visits: torch.Tensor,
    node_values: torch.Tensor,
    c_puct: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _csr_ucb_selection_impl(parent_indices, row_ptr, col_indices, edge_priors, 
                                  node_visits, node_values, c_puct)


# Public API functions
def csr_gather_children(
    node_indices: torch.Tensor,
    row_ptr: torch.Tensor,
    col_indices: torch.Tensor,
    edge_actions: torch.Tensor,
    max_children: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather children for nodes in CSR format with memory coalescing optimization
    
    Args:
        node_indices: Indices of nodes to gather children for [batch_size]
        row_ptr: CSR row pointer array [num_nodes + 1]
        col_indices: CSR column indices (child node IDs) [num_edges]
        edge_actions: Actions for each edge [num_edges]
        max_children: Maximum children per node
        
    Returns:
        children: Child node indices [batch_size, max_children]
        actions: Actions for each child [batch_size, max_children]
        valid_mask: Mask indicating valid children [batch_size, max_children]
    """
    # Prefetch hint for better memory access (torch.compile compatible)
    if node_indices.device.type == 'cuda' and node_indices.numel() > 32:
        # Touch memory ranges that will be accessed to trigger prefetching
        start_indices = row_ptr[node_indices]
        end_indices = row_ptr[node_indices + 1]
        
        if start_indices.numel() > 0:
            # Use torch operations instead of .item() for compile compatibility
            min_idx = start_indices.min()
            max_idx = end_indices.max()
            
            # Create index range for prefetching (compile-friendly)
            range_size = max_idx - min_idx
            if range_size > 0:
                # Touch the memory to hint at future access
                prefetch_indices = torch.arange(min_idx, max_idx, device=node_indices.device, dtype=torch.int32)
                if prefetch_indices.numel() > 0 and prefetch_indices[-1] < col_indices.numel():
                    _ = col_indices[prefetch_indices].sum()
                    _ = edge_actions[prefetch_indices].sum()
    
    return torch.ops.csr_ops.gather_children(
        node_indices, row_ptr, col_indices, edge_actions, max_children
    )

def csr_ucb_selection(
    parent_indices: torch.Tensor,
    row_ptr: torch.Tensor,
    col_indices: torch.Tensor,
    edge_priors: torch.Tensor,
    node_visits: torch.Tensor,
    node_values: torch.Tensor,
    c_puct: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform UCB selection for parents in CSR format
    
    Args:
        parent_indices: Parent node indices [batch_size]
        row_ptr: CSR row pointer array [num_nodes + 1]
        col_indices: CSR column indices (child node IDs) [num_edges]
        edge_priors: Prior probabilities for each edge [num_edges]
        node_visits: Visit counts for all nodes [num_nodes]
        node_values: Value sums for all nodes [num_nodes]
        c_puct: UCB exploration constant
        
    Returns:
        selected_children: Selected child node indices [batch_size]
        selected_actions: Selected edge indices [batch_size]
        ucb_scores: UCB scores for selected children [batch_size]
    """
    return torch.ops.csr_ops.ucb_selection(
        parent_indices, row_ptr, col_indices, edge_priors, 
        node_visits, node_values, c_puct
    )

# Note: torch.compile can be enabled later after resolving .item() compatibility
# For now, focus on core memory coalescing optimizations
# csr_gather_children = torch.compile(csr_gather_children, fullgraph=True)
# csr_ucb_selection = torch.compile(csr_ucb_selection, fullgraph=True)