"""Clean Unified GPU Kernel Interface for MCTS

This module provides a simplified, consolidated interface to all GPU kernels,
using the new consolidated CUDA manager for cleaner code and better maintainability.
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

# Import Triton kernels as fallback
try:
    from .triton_kernels import get_triton_kernels
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    get_triton_kernels = None

logger = logging.getLogger(__name__)

# Global kernel interface
_GLOBAL_KERNELS = None


class UnifiedGPUKernels:
    """Unified interface for all GPU kernels with clean fallback logic"""
    
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
                    logger.debug("✓ Using consolidated CUDA kernels")
                    return
                else:
                    logger.debug("⚠️ CUDA kernels not found - falling back to PyTorch")
            except Exception as e:
                logger.debug(f"⚠️ CUDA kernel loading failed: {e} - falling back to PyTorch")
        
        # Try Triton kernels as fallback
        if TRITON_AVAILABLE and self.device.type == 'cuda':
            triton_kernels = get_triton_kernels(self.device)
            if triton_kernels is not None:
                self._kernel_interface = triton_kernels
                logger.info("✓ Using Triton kernels as fallback")
                return
        
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
    
    def parallel_backup(self, *args, **kwargs):
        """Parallel backup operation (alias for vectorized_backup)"""
        return self.vectorized_backup(*args, **kwargs)
    
    def find_expansion_nodes(self, *args, **kwargs):
        """Find expansion nodes"""
        if hasattr(self._kernel_interface, 'find_expansion_nodes'):
            return self._kernel_interface.find_expansion_nodes(*args, **kwargs)
        return self._pytorch_find_expansion_nodes(*args, **kwargs)
    
    def quantum_ucb_selection(self, *args, **kwargs):
        """Quantum-enhanced UCB selection"""
        if hasattr(self._kernel_interface, 'quantum_ucb_selection'):
            return self._kernel_interface.quantum_ucb_selection(*args, **kwargs)
        return self.batched_ucb_selection(*args, **kwargs)  # Fallback to classical
    
    def get_stats(self):
        """Get kernel statistics"""
        base_stats = {
            'device': str(self.device),
            'interface_type': type(self._kernel_interface).__name__
        }
        
        if hasattr(self._kernel_interface, 'get_stats'):
            base_stats.update(self._kernel_interface.get_stats())
        
        return base_stats
    
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
        
        # Add parallel_backup alias dynamically if not present
        if hasattr(self.kernels, 'vectorized_backup') and not hasattr(self.kernels, 'parallel_backup'):
            setattr(self.kernels, 'parallel_backup', self.kernels.vectorized_backup)
    
    def batch_ucb_selection(self, *args, **kwargs):
        return self.kernels.batched_ucb_selection(*args, **kwargs)
    
    def vectorized_backup(self, *args, **kwargs):
        return self.kernels.vectorized_backup(*args, **kwargs)
    
    def parallel_backup(self, *args, **kwargs):
        """Parallel backup operation (alias for vectorized_backup)"""
        return self.kernels.vectorized_backup(*args, **kwargs)
    
    def find_expansion_nodes(self, *args, **kwargs):
        return self.kernels.find_expansion_nodes(*args, **kwargs)
    
    def quantum_ucb_selection(self, *args, **kwargs):
        return self.kernels.quantum_ucb_selection(*args, **kwargs)
    
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


def get_unified_kernels(device: torch.device = None) -> UnifiedGPUKernels:
    """Get unified kernel interface (per-process in multiprocessing)"""
    global _GLOBAL_KERNELS
    
    # In multiprocessing, create a new instance for each process
    current_pid = os.getpid()
    if (_GLOBAL_KERNELS is None or 
        not hasattr(_GLOBAL_KERNELS, '_process_id') or 
        getattr(_GLOBAL_KERNELS, '_process_id', None) != current_pid):
        
        _GLOBAL_KERNELS = UnifiedGPUKernels(device)
        _GLOBAL_KERNELS._process_id = current_pid
        logger.debug(f"Created new unified kernels for process {current_pid}")
    
    return _GLOBAL_KERNELS


def get_global_kernels() -> UnifiedGPUKernels:
    """Get global kernel instance"""
    return get_unified_kernels()


def validate_kernels() -> bool:
    """Validate kernel functionality"""
    try:
        kernels = get_unified_kernels()
        stats = kernels.get_stats()
        return 'interface_type' in stats
    except Exception as e:
        logger.error(f"Kernel validation failed: {e}")
        return False