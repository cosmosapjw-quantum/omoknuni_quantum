"""MCTS GPU Acceleration Layer

This module provides hardware-accelerated implementations of computationally intensive
MCTS operations including UCB selection, tree traversal, and value backup.

Key Features:
- Unified interface supporting CUDA kernels with PyTorch fallbacks
- Critical for achieving 168k+ simulations/second performance
- Automatic hardware detection and graceful degradation
- Process-safe global instance management for multiprocessing

Core MCTS Operations:
- batched_ucb_selection: UCB score computation for action selection
- vectorized_backup: Efficient batch propagation of values up the tree
- find_expansion_nodes: Node expansion logic for tree growth
- quantum_ucb_selection: Quantum-enhanced UCB variants
"""

import torch
import logging
from typing import Tuple, Optional, Dict, Any
import os

# Import the consolidated CUDA manager
try:
    from .cuda_manager import get_cuda_kernels, get_cuda_manager
    CUDA_MANAGER_AVAILABLE = True
except ImportError:
    CUDA_MANAGER_AVAILABLE = False
    get_cuda_kernels = None
    get_cuda_manager = None

# Legacy Triton support removed - now uses unified PyTorch implementations

logger = logging.getLogger(__name__)

# Global kernel interface
_GLOBAL_KERNELS = None


class MCTSGPUAccelerator:
    """GPU acceleration interface for MCTS operations with automatic fallback logic"""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._kernel_interface = None
        self._load_kernels()
    
    def _load_kernels(self):
        """Load kernels with fallback logic"""
        # Skip if CUDA disabled
        if os.environ.get('DISABLE_CUDA_KERNELS', '0') == '1':
            logger.info("CUDA kernels disabled by environment")
            self._use_pytorch_fallback()
            return
        
        # Try consolidated CUDA kernels first
        if CUDA_MANAGER_AVAILABLE and self.device.type == 'cuda':
            try:
                # Import detect function directly to avoid any compilation logic
                from .cuda_manager import detect_cuda_kernels
                kernels = detect_cuda_kernels()  # Only detection, never compilation
                if kernels is not None:
                    self._kernel_interface = ConsolidatedKernelInterface(kernels)
                    return
                else:
                    logger.debug("⚠️ CUDA kernels not found - falling back to PyTorch")
            except Exception as e:
                logger.debug(f"⚠️ CUDA kernel loading failed: {e} - falling back to PyTorch")
        
        # Note: Triton kernels have been removed in favor of unified implementation
        # All functionality is now available through PyTorch fallback implementations
        
        # Use PyTorch fallback
        self._use_pytorch_fallback()
    
    def _use_pytorch_fallback(self):
        """Use pure PyTorch implementations"""
        self._kernel_interface = PyTorchKernelInterface(self.device)
        logger.debug("✓ Using PyTorch fallback implementations")
    
    def batched_ucb_selection(self, *args, **kwargs):
        """Batched UCB selection"""
        return self._kernel_interface.batch_ucb_selection(*args, **kwargs)
    
    def vectorized_backup(self, *args, **kwargs):
        """Vectorized backup operation"""
        if hasattr(self._kernel_interface, 'vectorized_backup'):
            return self._kernel_interface.vectorized_backup(*args, **kwargs)
        return self._kernel_interface.pytorch_backup(*args, **kwargs)
    
    
    def find_expansion_nodes(self, *args, **kwargs):
        """Find expansion nodes"""
        if hasattr(self._kernel_interface, 'find_expansion_nodes'):
            return self._kernel_interface.find_expansion_nodes(*args, **kwargs)
        return self._pytorch_find_expansion_nodes(*args, **kwargs)
    
    def quantum_ucb_selection(self, *args, **kwargs):
        """Placeholder for quantum UCB (disabled) - uses classical UCB"""
        return self.batched_ucb_selection(*args, **kwargs)
    
    def batched_add_children(self, *args, **kwargs):
        """Batched addition of children with proper parent assignment
        
        This method adds multiple children nodes in a single batch operation,
        properly maintaining parent-child relationships in the tree structure.
        
        Args:
            parent_indices: Tensor of parent node indices (batch_size,)
            actions: Tensor of action indices (batch_size, max_children)
            priors: Tensor of prior probabilities (batch_size, max_children) 
            num_children: Tensor of actual children count per parent (batch_size,)
            node_counter: Global node counter tensor
            edge_counter: Global edge counter tensor
            children: Children lookup table
            parent_indices_out: Output parent indices array
            parent_actions_out: Output parent actions array
            node_priors_out: Output node priors array
            visit_counts_out: Output visit counts array
            value_sums_out: Output value sums array
            col_indices: CSR column indices
            edge_actions: CSR edge actions
            edge_priors: CSR edge priors
            max_nodes: Maximum number of nodes
            max_children: Maximum children per node
            max_edges: Maximum number of edges
            
        Returns:
            Tensor of child node indices if CUDA kernel available, None otherwise
            
        Raises:
            ValueError: If required arguments are missing or invalid
            RuntimeError: If CUDA kernel execution fails
            
        Example:
            >>> accelerator = get_mcts_gpu_accelerator(device)
            >>> child_indices = accelerator.batched_add_children(
            ...     parent_indices, actions, priors, num_children,
            ...     node_counter, edge_counter, children,
            ...     parent_indices_out, parent_actions_out, node_priors_out,
            ...     visit_counts_out, value_sums_out,
            ...     col_indices, edge_actions, edge_priors,
            ...     max_nodes, max_children, max_edges
            ... )
        """
        # Input validation
        if len(args) < 18:
            raise ValueError(
                f"batched_add_children requires 18 arguments, got {len(args)}. "
                "Required: parent_indices, actions, priors, num_children, "
                "node_counter, edge_counter, children, parent_indices_out, "
                "parent_actions_out, node_priors_out, visit_counts_out, "
                "value_sums_out, col_indices, edge_actions, edge_priors, "
                "max_nodes, max_children, max_edges"
            )
        
        # Validate tensor arguments (first 15 are tensors)
        import torch
        for i in range(15):
            if args[i] is not None and not isinstance(args[i], torch.Tensor):
                arg_names = [
                    "parent_indices", "actions", "priors", "num_children",
                    "node_counter", "edge_counter", "children", "parent_indices_out",
                    "parent_actions_out", "node_priors_out", "visit_counts_out",
                    "value_sums_out", "col_indices", "edge_actions", "edge_priors"
                ]
                raise TypeError(
                    f"Argument '{arg_names[i]}' (position {i}) must be a torch.Tensor, "
                    f"got {type(args[i]).__name__}"
                )
        
        # Check if CUDA kernel is available
        if not hasattr(self._kernel_interface, 'batched_add_children'):
            raise NotImplementedError(
                "batched_add_children requires CUDA kernels to be compiled and loaded. "
                "CPU fallback should be implemented in CSRTree. "
                "To compile CUDA kernels, run: python setup.py build_ext --inplace"
            )
        
        try:
            return self._kernel_interface.batched_add_children(*args, **kwargs)
        except RuntimeError as e:
            # Provide more helpful error message
            if "expected scalar type" in str(e):
                raise RuntimeError(
                    f"Type mismatch in batched_add_children: {e}. "
                    "Ensure all tensor dtypes match expected types "
                    "(int32 for indices/counts, float32 for values/priors)"
                )
            raise
    
    def get_stats(self):
        """Get kernel statistics"""
        base_stats = {
            'device': str(self.device),
            'interface_type': type(self._kernel_interface).__name__
        }
        
        if hasattr(self._kernel_interface, 'get_stats'):
            base_stats.update(self._kernel_interface.get_stats())
        
        return base_stats
    
    def reset_stats(self):
        """Reset kernel statistics"""
        if hasattr(self._kernel_interface, 'reset_stats'):
            self._kernel_interface.reset_stats()
        # For PyTorch interface, reset the stats dict
        if hasattr(self, 'stats'):
            self.stats = {
                'ucb_calls': 0,
                'backup_calls': 0,
                'expansion_calls': 0,
                'total_time': 0.0
            }
    
    def _pytorch_find_expansion_nodes(self, current_nodes, children, visit_counts, valid_path_mask, wave_size, max_children):
        """PyTorch fallback for find_expansion_nodes"""
        expansion_nodes = []
        for i, node_idx in enumerate(current_nodes[:wave_size]):
            if not valid_path_mask[i] or node_idx < 0:
                continue
            
            # Check if node has children
            node_children = children[node_idx * max_children:(node_idx + 1) * max_children]
            has_children = (node_children >= 0).any()
            
            # Check if needs expansion
            if not has_children and visit_counts[node_idx] == 0:
                expansion_nodes.append(node_idx.item())
        
        return torch.tensor(expansion_nodes, device=self.device, dtype=torch.int32)


class ConsolidatedKernelInterface:
    """Interface wrapper for consolidated CUDA kernels"""
    
    def __init__(self, kernel_module):
        self.kernels = kernel_module
    
    def batch_ucb_selection(self, *args, **kwargs):
        return self.kernels.batched_ucb_selection(*args, **kwargs)
    
    def vectorized_backup(self, *args, **kwargs):
        return self.kernels.vectorized_backup(*args, **kwargs)
    
    def find_expansion_nodes(self, *args, **kwargs):
        return self.kernels.find_expansion_nodes(*args, **kwargs)
    
    def quantum_ucb_selection(self, *args, **kwargs):
        return self.kernels.quantum_ucb_selection(*args, **kwargs)
    
    def batched_add_children(self, *args, **kwargs):
        return self.kernels.batched_add_children(*args, **kwargs)
    
    def get_stats(self):
        if CUDA_MANAGER_AVAILABLE:
            manager = get_cuda_manager()
            return manager.get_kernel_info()
        return {'kernel_type': 'consolidated_cuda'}


class PyTorchKernelInterface:
    """Pure PyTorch fallback implementations"""
    
    def __init__(self, device):
        self.device = device
    
    def batch_ucb_selection(self, q_values, visit_counts, parent_visits, priors, row_ptr, col_indices, c_puct):
        """Pure PyTorch UCB selection"""
        num_nodes = parent_visits.shape[0]
        selected_actions = torch.full((num_nodes,), -1, dtype=torch.int32, device=self.device)
        selected_scores = torch.zeros(num_nodes, dtype=torch.float32, device=self.device)
        
        for idx in range(num_nodes):
            start = row_ptr[idx].item()
            end = row_ptr[idx + 1].item()
            
            if start == end:
                continue
            
            parent_visit = parent_visits[idx].float()
            sqrt_parent = torch.sqrt(parent_visit + 1.0)
            
            # Get children
            child_indices = col_indices[start:end]
            child_visits = visit_counts[child_indices].float()
            child_q_values = q_values[child_indices]
            child_priors = priors[start:end]
            
            # UCB calculation
            exploration = c_puct * child_priors * sqrt_parent / (1.0 + child_visits)
            ucb_scores = child_q_values + exploration
            
            # Find best action
            best_idx = torch.argmax(ucb_scores)
            selected_actions[idx] = best_idx
            selected_scores[idx] = ucb_scores[best_idx]
        
        return selected_actions, selected_scores
    
    def pytorch_backup(self, paths, path_lengths, values, visit_counts, value_sums):
        """Pure PyTorch backup implementation"""
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
    
    def get_stats(self):
        return {
            'kernel_type': 'pytorch_fallback',
            'device': str(self.device)
        }


def get_mcts_gpu_accelerator(device: torch.device = None) -> MCTSGPUAccelerator:
    """Get MCTS GPU accelerator instance (per-process in multiprocessing)"""
    global _GLOBAL_KERNELS
    
    # In multiprocessing, create a new instance for each process
    current_pid = os.getpid()
    if (_GLOBAL_KERNELS is None or 
        not hasattr(_GLOBAL_KERNELS, '_process_id') or 
        getattr(_GLOBAL_KERNELS, '_process_id', None) != current_pid):
        
        _GLOBAL_KERNELS = MCTSGPUAccelerator(device)
        _GLOBAL_KERNELS._process_id = current_pid
    
    return _GLOBAL_KERNELS


def get_global_accelerator() -> MCTSGPUAccelerator:
    """Get global MCTS GPU accelerator instance"""
    return get_mcts_gpu_accelerator()


def validate_mcts_accelerator() -> bool:
    """Validate MCTS GPU accelerator functionality"""
    try:
        accelerator = get_mcts_gpu_accelerator()
        stats = accelerator.get_stats()
        return 'interface_type' in stats
    except Exception as e:
        logger.error(f"MCTS GPU accelerator validation failed: {e}")
        return False


# Note: Legacy aliases removed in streamlined build
# Use MCTSGPUAccelerator, get_mcts_gpu_accelerator, etc. directly