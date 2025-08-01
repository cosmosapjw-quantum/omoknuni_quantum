"""Register MCTS CUDA kernels as PyTorch custom operators

This allows torch.compile to properly optimize the entire MCTS pipeline
by making our custom CUDA kernels visible to the compiler.
"""

import torch
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import CUDA kernels
try:
    from mcts.gpu.cuda_manager import CudaManager
    cuda_manager = CudaManager()
    mcts_cuda = cuda_manager.get_kernels()
    HAS_CUDA = mcts_cuda is not None
except Exception as e:
    logger.warning(f"CUDA kernels not available: {e}")
    mcts_cuda = None
    HAS_CUDA = False


# Custom operator for vectorized backup
@torch.library.custom_op("mcts::vectorized_backup", mutates_args=["visit_counts", "value_sums"])
def vectorized_backup(
    paths: torch.Tensor,
    path_lengths: torch.Tensor,
    values: torch.Tensor,
    visit_counts: torch.Tensor,
    value_sums: torch.Tensor
) -> None:
    """Vectorized backup operation for MCTS
    
    Args:
        paths: Node indices along paths [batch_size, max_depth]
        path_lengths: Length of each path [batch_size]
        values: Values to backup [batch_size]
        visit_counts: Visit counts to update (mutated) [num_nodes]
        value_sums: Value sums to update (mutated) [num_nodes]
    """
    if HAS_CUDA and paths.is_cuda:
        mcts_cuda.vectorized_backup(paths, path_lengths, values, visit_counts, value_sums)
    else:
        # Fallback to Python implementation
        batch_size = paths.shape[0]
        for i in range(batch_size):
            path_len = path_lengths[i].item()
            value = values[i].item()
            for j in range(path_len):
                node_idx = paths[i, j].item()
                visit_counts[node_idx] += 1
                value_sums[node_idx] += value


@torch.library.register_fake("mcts::vectorized_backup")
def vectorized_backup_fake(
    paths: torch.Tensor,
    path_lengths: torch.Tensor,
    values: torch.Tensor,
    visit_counts: torch.Tensor,
    value_sums: torch.Tensor
) -> None:
    """Fake implementation for torch.compile shape inference"""
    # Just pass - the tensors are mutated in-place
    pass


# Custom operator for warp-optimized backup
@torch.library.custom_op("mcts::warp_vectorized_backup", mutates_args=["visit_counts", "value_sums"])
def warp_vectorized_backup(
    paths: torch.Tensor,
    path_lengths: torch.Tensor,
    values: torch.Tensor,
    visit_counts: torch.Tensor,
    value_sums: torch.Tensor
) -> None:
    """Warp-optimized vectorized backup with warp primitives
    
    Args:
        paths: Node indices along paths [batch_size, max_depth]
        path_lengths: Length of each path [batch_size]
        values: Values to backup [batch_size]
        visit_counts: Visit counts to update (mutated) [num_nodes]
        value_sums: Value sums to update (mutated) [num_nodes]
    """
    if HAS_CUDA and paths.is_cuda:
        mcts_cuda.warp_vectorized_backup(paths, path_lengths, values, visit_counts, value_sums)
    else:
        # Fallback to regular vectorized backup
        vectorized_backup(paths, path_lengths, values, visit_counts, value_sums)


@torch.library.register_fake("mcts::warp_vectorized_backup")
def warp_vectorized_backup_fake(
    paths: torch.Tensor,
    path_lengths: torch.Tensor,
    values: torch.Tensor,
    visit_counts: torch.Tensor,
    value_sums: torch.Tensor
) -> None:
    """Fake implementation for torch.compile shape inference"""
    pass


# Custom operator for fused UCB selection
@torch.library.custom_op("mcts::fused_ucb_selection", mutates_args=[])
def fused_ucb_selection(
    node_indices: torch.Tensor,
    children_start: torch.Tensor,
    children_end: torch.Tensor,
    children: torch.Tensor,
    q_values: torch.Tensor,
    visit_counts: torch.Tensor,
    priors: torch.Tensor,
    c_puct: float
) -> torch.Tensor:
    """Fused UCB selection for maximum performance
    
    Args:
        node_indices: Parent node indices [batch_size]
        children_start: Start indices for children [num_nodes]
        children_end: End indices for children [num_nodes]
        children: Child node indices [num_edges]
        q_values: Q-values [num_nodes]
        visit_counts: Visit counts [num_nodes]
        priors: Prior probabilities [num_edges]
        c_puct: Exploration constant
        
    Returns:
        selected_actions: Selected action indices [batch_size]
    """
    if HAS_CUDA and node_indices.is_cuda:
        # Use fused UCB selection directly - it now accepts the Python interface
        return mcts_cuda.fused_ucb_selection(
            node_indices, children_start, children_end, children,
            q_values, visit_counts, priors, c_puct
        )
    else:
        # CPU fallback implementation
        batch_size = node_indices.shape[0]
        selected = torch.zeros(batch_size, dtype=torch.int32, device=node_indices.device)
        
        for i in range(batch_size):
            node_idx = node_indices[i].item()
            start = children_start[node_idx].item()
            end = children_end[node_idx].item()
            
            if start >= end:
                selected[i] = -1
                continue
            
            best_ucb = -1e10
            best_action = -1
            
            parent_visit = visit_counts[node_idx].item()
            sqrt_parent = (parent_visit ** 0.5) if parent_visit > 0 else 1.0
            
            for j in range(start, end):
                child_idx = children[j].item()
                child_visit = visit_counts[child_idx].item()
                
                if child_visit > 0:
                    q_value = q_values[child_idx].item()
                else:
                    q_value = 0.0
                
                exploration = c_puct * priors[j].item() * sqrt_parent / (1.0 + child_visit)
                ucb = q_value + exploration
                
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_action = j - start
            
            selected[i] = best_action
        
        return selected


@torch.library.register_fake("mcts::fused_ucb_selection")
def fused_ucb_selection_fake(
    node_indices: torch.Tensor,
    children_start: torch.Tensor,
    children_end: torch.Tensor,
    children: torch.Tensor,
    q_values: torch.Tensor,
    visit_counts: torch.Tensor,
    priors: torch.Tensor,
    c_puct: float
) -> torch.Tensor:
    """Fake implementation for torch.compile shape inference"""
    return torch.empty_like(node_indices, dtype=torch.int32)


def register_all_ops():
    """Register all custom MCTS operators with PyTorch
    
    This should be called once at module initialization to ensure
    torch.compile can see and optimize our custom operators.
    """
    # The decorators already register the ops, but we can do additional setup here
    if HAS_CUDA:
        logger.info("✅ MCTS custom operators registered for torch.compile")
        
        # Try to set optimization flags if available
        try:
            if hasattr(torch._inductor.config, 'cpp'):
                if hasattr(torch._inductor.config.cpp, 'enable_kernel_fusion'):
                    torch._inductor.config.cpp.enable_kernel_fusion = True
            if hasattr(torch._inductor.config, 'triton'):
                if hasattr(torch._inductor.config.triton, 'unique_kernel_names'):
                    torch._inductor.config.triton.unique_kernel_names = True
            if hasattr(torch._inductor.config, 'coordinate_descent_tuning'):
                torch._inductor.config.coordinate_descent_tuning = True
        except Exception:
            # Ignore errors - these are optional optimizations
            pass
    else:
        logger.warning("CUDA not available - custom operators will use fallback implementations")


# Register ops on module import
register_all_ops()


# Convenience functions that use the registered operators
def optimized_backup(paths, path_lengths, values, visit_counts, value_sums, use_warp=True):
    """Use optimized backup with custom operators
    
    Args:
        paths: Node paths [batch_size, max_depth]
        path_lengths: Path lengths [batch_size]
        values: Values to backup [batch_size]
        visit_counts: Visit counts (will be updated) [num_nodes]
        value_sums: Value sums (will be updated) [num_nodes]
        use_warp: Whether to use warp-optimized version
    """
    if use_warp and HAS_CUDA:
        torch.ops.mcts.warp_vectorized_backup(
            paths, path_lengths, values, visit_counts, value_sums
        )
    else:
        torch.ops.mcts.vectorized_backup(
            paths, path_lengths, values, visit_counts, value_sums
        )


def optimized_ucb_selection(
    node_indices, children_start, children_end, children,
    q_values, visit_counts, priors, c_puct
):
    """Use optimized UCB selection with custom operators
    
    Returns:
        selected_actions: Selected actions [batch_size]
    """
    return torch.ops.mcts.fused_ucb_selection(
        node_indices, children_start, children_end, children,
        q_values, visit_counts, priors, c_puct
    )