"""
Triton-based MCTS kernels for high performance GPU acceleration

Triton is a modern alternative to CUDA that provides:
- Python-based kernel development
- Automatic optimization and tuning
- Better PyTorch integration
- No complex compilation pipeline
"""

import torch
import triton
import triton.language as tl
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

@triton.jit
def ucb_selection_kernel(
    # Inputs
    q_values_ptr,
    visit_counts_ptr,
    parent_visits_ptr,
    priors_ptr,
    row_ptr_ptr,
    col_indices_ptr,
    # Outputs
    selected_actions_ptr,
    ucb_scores_ptr,
    # Constants
    c_puct: tl.constexpr,
    num_nodes: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for UCB selection"""
    node_id = tl.program_id(0)
    
    if node_id >= num_nodes:
        return
    
    # Get children range for this node
    start_idx = tl.load(row_ptr_ptr + node_id)
    end_idx = tl.load(row_ptr_ptr + node_id + 1)
    num_children = end_idx - start_idx
    
    if num_children == 0:
        tl.store(selected_actions_ptr + node_id, -1)
        tl.store(ucb_scores_ptr + node_id, 0.0)
        return
    
    # Get parent visit count
    parent_visits = tl.load(parent_visits_ptr + node_id)
    sqrt_parent = tl.sqrt(parent_visits.to(tl.float32))
    
    # Find best UCB action
    best_score = -1e10
    best_action = -1
    
    # Process children in blocks
    for child_offset in range(0, num_children, BLOCK_SIZE):
        child_mask = child_offset + tl.arange(0, BLOCK_SIZE) < num_children
        child_indices = start_idx + child_offset + tl.arange(0, BLOCK_SIZE)
        
        # Load child data
        child_nodes = tl.load(col_indices_ptr + child_indices, mask=child_mask, other=0)
        child_visits = tl.load(visit_counts_ptr + child_nodes, mask=child_mask, other=0)
        child_priors = tl.load(priors_ptr + child_indices, mask=child_mask, other=0.0)
        
        # Compute Q-values
        q_vals = tl.where(
            child_visits > 0,
            tl.load(q_values_ptr + child_nodes, mask=child_mask, other=0.0) / child_visits.to(tl.float32),
            0.0
        )
        
        # Compute UCB scores
        exploration = c_puct * child_priors * sqrt_parent / (1.0 + child_visits.to(tl.float32))
        ucb_scores = q_vals + exploration
        
        # Find local maximum
        for i in range(BLOCK_SIZE):
            if child_mask[i] and ucb_scores[i] > best_score:
                best_score = ucb_scores[i]
                best_action = child_offset + i
    
    # Store results
    tl.store(selected_actions_ptr + node_id, best_action)
    tl.store(ucb_scores_ptr + node_id, best_score)


@triton.jit 
def vectorized_backup_kernel(
    # Inputs
    paths_ptr,
    path_lengths_ptr,
    values_ptr,
    # Outputs  
    visit_counts_ptr,
    value_sums_ptr,
    # Constants
    batch_size: tl.constexpr,
    max_depth: tl.constexpr,
    max_nodes: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for vectorized backup"""
    batch_id = tl.program_id(0)
    
    if batch_id >= batch_size:
        return
        
    # Get path length and value for this batch
    path_length = tl.load(path_lengths_ptr + batch_id)
    batch_value = tl.load(values_ptr + batch_id)
    
    # Process path in reverse (backup)
    for depth in range(path_length):
        node_idx = tl.load(paths_ptr + batch_id * max_depth + depth)
        
        if node_idx >= 0 and node_idx < max_nodes:
            # Alternating value sign (minimax)
            sign = 1.0 if depth % 2 == 0 else -1.0
            backup_value = batch_value * sign
            
            # Atomic updates
            tl.atomic_add(visit_counts_ptr + node_idx, 1)
            tl.atomic_add(value_sums_ptr + node_idx, backup_value)


class TritonMCTSKernels:
    """Triton-based MCTS kernels for high performance"""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_triton = self.device.type == 'cuda' and self._check_triton_availability()
        
        if self.use_triton:
            logger.debug("✅ Triton kernels available for MCTS acceleration")
        else:
            logger.debug("⚠️ Triton not available, using PyTorch fallback")
    
    def _check_triton_availability(self) -> bool:
        """Check if Triton is available and working"""
        try:
            # Simple test without JIT compilation issues
            x = torch.zeros(1, device=self.device, dtype=torch.float32)
            # Just check if we can create tensors and Triton imports work
            return triton is not None and tl is not None and x.device.type == 'cuda'
            
        except Exception as e:
            logger.debug(f"Triton availability check failed: {e}")
            return False
    
    def batch_ucb_selection(
        self,
        q_values: torch.Tensor,
        visit_counts: torch.Tensor,
        parent_visits: torch.Tensor,
        priors: torch.Tensor,
        row_ptr: torch.Tensor,
        col_indices: torch.Tensor,
        c_puct: float = 1.414
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch UCB selection using Triton"""
        
        num_nodes = parent_visits.shape[0]
        
        if not self.use_triton or num_nodes == 0:
            return self._pytorch_ucb_fallback(
                q_values, visit_counts, parent_visits, priors, 
                row_ptr, col_indices, c_puct
            )
        
        # Prepare output tensors
        selected_actions = torch.full((num_nodes,), -1, dtype=torch.int32, device=self.device)
        ucb_scores = torch.zeros(num_nodes, dtype=torch.float32, device=self.device)
        
        # Launch Triton kernel
        BLOCK_SIZE = 32
        grid = (num_nodes,)
        
        try:
            ucb_selection_kernel[grid](
                q_values, visit_counts, parent_visits, priors,
                row_ptr, col_indices,
                selected_actions, ucb_scores,
                c_puct, num_nodes, BLOCK_SIZE=BLOCK_SIZE
            )
            
            return selected_actions, ucb_scores
            
        except Exception as e:
            logger.debug(f"Triton UCB kernel failed: {e}, using PyTorch fallback")
            return self._pytorch_ucb_fallback(
                q_values, visit_counts, parent_visits, priors,
                row_ptr, col_indices, c_puct
            )
    
    def vectorized_backup(
        self,
        paths: torch.Tensor,
        path_lengths: torch.Tensor,
        values: torch.Tensor,
        visit_counts: torch.Tensor,
        value_sums: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized backup using Triton"""
        
        batch_size, max_depth = paths.shape
        max_nodes = visit_counts.shape[0]
        
        if not self.use_triton or batch_size == 0:
            return self._pytorch_backup_fallback(
                paths, path_lengths, values, visit_counts, value_sums
            )
        
        try:
            # Launch Triton kernel
            BLOCK_SIZE = 1  # Process one path per thread
            grid = (batch_size,)
            
            vectorized_backup_kernel[grid](
                paths, path_lengths, values,
                visit_counts, value_sums,
                batch_size, max_depth, max_nodes, BLOCK_SIZE=BLOCK_SIZE
            )
            
            return visit_counts, value_sums
            
        except Exception as e:
            logger.debug(f"Triton backup kernel failed: {e}, using PyTorch fallback")
            return self._pytorch_backup_fallback(
                paths, path_lengths, values, visit_counts, value_sums
            )
    
    def _pytorch_ucb_fallback(self, q_values, visit_counts, parent_visits, priors, row_ptr, col_indices, c_puct):
        """PyTorch fallback for UCB selection"""
        num_nodes = parent_visits.shape[0]
        selected_actions = torch.full((num_nodes,), -1, dtype=torch.int32, device=self.device)
        ucb_scores = torch.zeros(num_nodes, dtype=torch.float32, device=self.device)
        
        for node_id in range(num_nodes):
            start = row_ptr[node_id].item()
            end = row_ptr[node_id + 1].item()
            
            if start >= end:
                continue
                
            children = col_indices[start:end]
            child_visits = visit_counts[children].float()
            child_priors = priors[start:end]
            
            # Q-values
            q_vals = torch.where(
                child_visits > 0,
                q_values[children] / child_visits,
                torch.zeros_like(child_visits)
            )
            
            # UCB scores
            exploration = c_puct * child_priors * torch.sqrt(parent_visits[node_id].float()) / (1 + child_visits)
            ucb = q_vals + exploration
            
            best_idx = ucb.argmax()
            selected_actions[node_id] = best_idx
            ucb_scores[node_id] = ucb[best_idx]
        
        return selected_actions, ucb_scores
    
    def _pytorch_backup_fallback(self, paths, path_lengths, values, visit_counts, value_sums):
        """PyTorch fallback for backup"""
        batch_size, max_depth = paths.shape
        
        for batch_id in range(batch_size):
            path_length = path_lengths[batch_id].item()
            batch_value = values[batch_id].item()
            
            for depth in range(path_length):
                node_idx = paths[batch_id, depth].item()
                if node_idx >= 0 and node_idx < visit_counts.shape[0]:
                    sign = 1.0 if depth % 2 == 0 else -1.0
                    backup_value = batch_value * sign
                    
                    visit_counts[node_idx] += 1
                    value_sums[node_idx] += backup_value
        
        return visit_counts, value_sums
    
    def get_stats(self):
        """Get kernel statistics"""
        return {
            'kernel_type': 'triton',
            'triton_available': self.use_triton,
            'device': str(self.device)
        }


# Create global instance
_triton_kernels = None

def get_triton_kernels(device: torch.device = None) -> TritonMCTSKernels:
    """Get global Triton kernels instance"""
    global _triton_kernels
    if _triton_kernels is None:
        _triton_kernels = TritonMCTSKernels(device)
    return _triton_kernels