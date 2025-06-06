"""Wrapper for custom CUDA kernels with proper fallback"""

import torch
import logging
import os

logger = logging.getLogger(__name__)

# Check if custom kernels should be used
USE_CUSTOM_KERNELS = os.environ.get('USE_CUSTOM_CUDA_KERNELS', '0') == '1'
CUDA_AVAILABLE = torch.cuda.is_available()

if USE_CUSTOM_KERNELS and CUDA_AVAILABLE:
    try:
        # Try to import pre-compiled kernels
        from . import custom_cuda_ops
        CUSTOM_KERNELS_AVAILABLE = True
        logger.info("Custom CUDA kernels loaded successfully")
    except ImportError:
        CUSTOM_KERNELS_AVAILABLE = False
        logger.info("Custom CUDA kernels not available, using PyTorch fallback")
else:
    CUSTOM_KERNELS_AVAILABLE = False
    if not CUDA_AVAILABLE:
        logger.info("CUDA not available, using CPU fallback")
    else:
        logger.info("Custom CUDA kernels disabled")


def batched_ucb_selection(q_values, visit_counts, parent_visits, priors, 
                         row_ptr, col_indices, c_puct):
    """Batched UCB selection with custom kernel or fallback"""
    
    if CUSTOM_KERNELS_AVAILABLE:
        return custom_cuda_ops.batched_ucb_selection(
            q_values, visit_counts, parent_visits, priors,
            row_ptr, col_indices, c_puct
        )
    else:
        # PyTorch fallback implementation
        num_nodes = parent_visits.shape[0]
        device = q_values.device
        selected_actions = torch.zeros(num_nodes, dtype=torch.int32, device=device)
        
        for i in range(num_nodes):
            start = row_ptr[i].item()
            end = row_ptr[i + 1].item()
            
            if start == end:
                selected_actions[i] = -1
                continue
                
            # Get children data
            child_indices = col_indices[start:end]
            child_visits = visit_counts[child_indices].float()
            child_q_values = torch.where(
                child_visits > 0,
                q_values[child_indices] / child_visits,
                torch.zeros_like(child_visits)
            )
            child_priors = priors[start:end]
            
            # Compute UCB
            sqrt_parent = torch.sqrt(parent_visits[i].float())
            exploration = c_puct * child_priors * sqrt_parent / (1 + child_visits)
            ucb_scores = child_q_values + exploration
            
            # Select best
            selected_actions[i] = ucb_scores.argmax()
            
        return selected_actions


def parallel_backup(paths, leaf_values, path_lengths, value_sums, visit_counts):
    """Parallel backup with custom kernel or fallback"""
    
    if CUSTOM_KERNELS_AVAILABLE:
        return custom_cuda_ops.parallel_backup(
            paths, leaf_values, path_lengths, value_sums, visit_counts
        )
    else:
        # PyTorch fallback implementation
        batch_size, max_depth = paths.shape
        device = paths.device
        
        # Vectorized backup using scatter operations
        for depth in range(max_depth):
            # Get all nodes at this depth
            nodes_at_depth = paths[:, depth]
            
            # Create mask for valid nodes
            valid_mask = (nodes_at_depth >= 0) & (nodes_at_depth < value_sums.shape[0])
            
            if not valid_mask.any():
                continue
                
            # Get valid nodes and values
            valid_nodes = nodes_at_depth[valid_mask]
            depth_values = leaf_values[valid_mask] * ((-1) ** depth)
            
            # Use index_add for atomic updates
            value_sums.index_add_(0, valid_nodes, depth_values)
            visit_counts.index_add_(0, valid_nodes, 
                                   torch.ones_like(valid_nodes, dtype=visit_counts.dtype))
                    
        return value_sums