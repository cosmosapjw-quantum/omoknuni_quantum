"""GPU Kernels optimized for CSR Tree Format

This module provides a clean interface to GPU-accelerated operations for CSR trees.
It uses the unified kernel implementation for consistency and performance.
"""

import torch
from typing import Tuple, Optional, Dict, Any
import logging

from .unified_kernels import get_unified_kernels

logger = logging.getLogger(__name__)

# For backward compatibility
CUDA_KERNELS_AVAILABLE = False
CUSTOM_KERNELS_AVAILABLE = False

def check_cuda_available():
    """Check if CUDA is available"""
    return torch.cuda.is_available()


def csr_batch_ucb_torch(
    node_indices: torch.Tensor,
    row_ptr: torch.Tensor,
    col_indices: torch.Tensor,
    edge_actions: torch.Tensor,
    edge_priors: torch.Tensor,
    visit_counts: torch.Tensor,
    value_sums: torch.Tensor,
    parent_visits: torch.Tensor,
    c_puct: float,
    temperature: float = 1.0
) -> torch.Tensor:
    """Batch UCB calculation using unified kernels
    
    This function provides backward compatibility while using the new unified implementation.
    """
    kernels = get_unified_kernels(node_indices.device)
    return kernels.batch_ucb_selection(
        node_indices, row_ptr, col_indices, edge_actions, edge_priors,
        visit_counts, value_sums, c_puct, temperature
    )


class CSRBatchOperations:
    """High-performance batch operations for CSR tree using unified kernels"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.kernels = get_unified_kernels(device)
        
        # Performance tracking (delegates to unified kernels)
        self.kernel_stats = self.kernels.stats
        
        logger.debug("Using unified GPU kernels for CSR operations")
    
    def batch_select_ucb(
        self,
        node_indices: torch.Tensor,
        row_ptr: torch.Tensor,
        col_indices: torch.Tensor,
        edge_actions: torch.Tensor,
        edge_priors: torch.Tensor,
        visit_counts: torch.Tensor,
        value_sums: torch.Tensor,
        c_puct: float = 1.0,
        temperature: float = 1.0,
        **kwargs  # Ignore extra args for compatibility
    ) -> torch.Tensor:
        """Batch UCB selection using unified kernels"""
        return self.kernels.batch_ucb_selection(
            node_indices, row_ptr, col_indices, edge_actions, edge_priors,
            visit_counts, value_sums, c_puct, temperature
        )
    
    def coalesced_backup(
        self,
        paths: torch.Tensor,
        values: torch.Tensor,
        path_lengths: torch.Tensor,
        visit_counts: torch.Tensor,
        value_sums: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Coalesced backup using unified kernels"""
        return self.kernels.parallel_backup(
            paths, values, path_lengths, visit_counts, value_sums
        )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.kernels.get_stats()
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.kernels.reset_stats()


# For backward compatibility
OptimizedCSRKernels = CSRBatchOperations
CSRGPUKernels = CSRBatchOperations


# Global instance management
_csr_batch_ops = None

def get_csr_batch_operations(device: torch.device) -> CSRBatchOperations:
    """Get or create the global CSR batch operations instance"""
    global _csr_batch_ops
    
    if _csr_batch_ops is None or _csr_batch_ops.device != device:
        _csr_batch_ops = CSRBatchOperations(device)
    
    return _csr_batch_ops


def get_csr_kernels() -> CSRBatchOperations:
    """Legacy function for backward compatibility"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return get_csr_batch_operations(device)


def csr_coalesced_children_gather(
    node_indices: torch.Tensor,
    row_ptr: torch.Tensor,
    col_indices: torch.Tensor,
    max_children: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimized children gathering with coalesced memory access"""
    batch_size = node_indices.shape[0]
    device = node_indices.device
    
    # Pre-allocate output
    batch_children = torch.full([batch_size, max_children], -1,
                               dtype=torch.int32, device=device)
    valid_mask = torch.zeros([batch_size, max_children], 
                            dtype=torch.bool, device=device)
    
    # Vectorized gathering
    for i in range(batch_size):
        node_idx = node_indices[i]
        start = row_ptr[node_idx]
        end = row_ptr[node_idx + 1]
        num_children = end - start
        
        if num_children > 0:
            actual_children = min(num_children.item(), max_children)
            batch_children[i, :actual_children] = col_indices[start:start + actual_children]
            valid_mask[i, :actual_children] = True
            
    return batch_children, valid_mask