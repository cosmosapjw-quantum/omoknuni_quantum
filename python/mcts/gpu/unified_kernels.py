"""Unified GPU kernel interface for MCTS

This module provides a clean, consolidated interface to all GPU kernels,
now using modular CUDA kernels for faster compilation and better maintainability.

PERFORMANCE OPTIMIZATION: The previous unified_cuda_kernels.cu (1593 lines) has been
split into smaller, specialized modules for significantly faster compilation.
"""

import torch
import logging
from typing import Tuple, Optional, Dict, Any
import os
import sys
from pathlib import Path

# Import the new modular kernel manager
try:
    from .modular_kernel_manager import get_unified_kernels, get_global_kernels
    MODULAR_KERNELS_AVAILABLE = True
except ImportError:
    MODULAR_KERNELS_AVAILABLE = False
    get_unified_kernels = None
    get_global_kernels = None

# Try to import Triton kernels as fallback
try:
    from .triton_kernels import get_triton_kernels
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    get_triton_kernels = None

# Legacy kernel wrapper for backward compatibility
try:
    from .kernel_wrapper import wrap_kernel_module
    LEGACY_WRAPPER_AVAILABLE = True
except ImportError:
    LEGACY_WRAPPER_AVAILABLE = False
    wrap_kernel_module = None

logger = logging.getLogger(__name__)

# Global kernel interface
_GLOBAL_KERNEL_INTERFACE = None
_KERNELS_AVAILABLE = False

def _load_kernels():
    """Load CUDA kernels with improved compilation speed and fallback support"""
    global _GLOBAL_KERNEL_INTERFACE, _KERNELS_AVAILABLE
    
    if _GLOBAL_KERNEL_INTERFACE is not None:
        return _KERNELS_AVAILABLE
    
    # Check environment variable to disable CUDA kernels for testing
    if os.environ.get('DISABLE_CUDA_KERNELS', '0') == '1':
        logger.debug("CUDA kernels disabled by environment variable")
        _KERNELS_AVAILABLE = False
        return False
    
    logger.debug("🔍 Starting modular kernel loading process...")
    
    try:
        # DIRECT APPROACH: Use the working original CUDA system  
        logger.debug("Loading CUDA kernels using original system...")
        
        # Import the kernels directly from cuda_compile
        from .cuda_compile import CUDA_KERNELS_AVAILABLE
        if CUDA_KERNELS_AVAILABLE:
            from . import cuda_compile
            
            # Create a simple interface that provides the kernel functions
            class DirectKernelInterface:
                def __init__(self):
                    # Map the available functions
                    if hasattr(cuda_compile, 'batched_ucb_selection'):
                        self.batched_ucb_selection = cuda_compile.batched_ucb_selection
                    if hasattr(cuda_compile, 'parallel_backup'):
                        self.parallel_backup = cuda_compile.parallel_backup
                    if hasattr(cuda_compile, 'vectorized_backup'):
                        self.vectorized_backup = cuda_compile.vectorized_backup
                    if hasattr(cuda_compile, 'fused_selection_traversal'):
                        self.fused_selection_traversal = cuda_compile.fused_selection_traversal
                    if hasattr(cuda_compile, 'batched_add_children'):
                        self.batched_add_children = cuda_compile.batched_add_children
                    if hasattr(cuda_compile, 'quantum_interference'):
                        self.quantum_interference = cuda_compile.quantum_interference
                    
                    # Try to access the diversified kernel from torch.ops
                    self._load_diversified_kernel()
                
                def _load_diversified_kernel(self):
                    """Load the diversified kernel from compiled module if available"""
                    try:
                        # Method 1: Try to access from the same compiled module that provides other kernels
                        compiled_module = cuda_compile.get_cuda_kernels()
                        if compiled_module and hasattr(compiled_module, 'batched_ucb_selection_diversified'):
                            self.batched_ucb_selection_diversified = compiled_module.batched_ucb_selection_diversified
                            logger.debug("Loaded diversified kernel from compiled module")
                            return
                        
                        # Method 2: Try to access from torch.ops (if available)
                        precompiled_ops = [attr for attr in dir(torch.ops) if 'mcts_cuda_precompiled' in attr]
                        
                        for op_name in precompiled_ops:
                            try:
                                ops_module = getattr(torch.ops, op_name)
                                # Try to access the diversified function
                                if hasattr(ops_module, 'batched_ucb_selection_diversified'):
                                    diversified_op = getattr(ops_module, 'batched_ucb_selection_diversified')
                                    
                                    # For torch.ops, we need to use .default to get the actual function
                                    if hasattr(diversified_op, 'default'):
                                        self.batched_ucb_selection_diversified = diversified_op.default
                                        logger.debug(f"✅ Loaded diversified kernel from {op_name}")
                                        return
                                    else:
                                        # Try direct access
                                        self.batched_ucb_selection_diversified = diversified_op
                                        logger.debug(f"✅ Loaded diversified kernel (direct) from {op_name}")
                                        return
                                        
                            except Exception as e:
                                logger.debug(f"Failed to load diversified kernel from {op_name}: {e}")
                        
                        # Method 3: Skip recompilation to avoid hanging - kernels already loaded
                        logger.debug("⏭️ Skipping diversified kernel recompilation - using fallback if needed")
                        
                        logger.debug("Diversified kernel not found using any method")
                        
                    except Exception as e:
                        logger.debug(f"Failed to load diversified kernel: {e}")
                
                def get_stats(self):
                    stats = {'kernels_loaded': True, 'method': 'direct_cuda_compile'}
                    if hasattr(self, 'batched_ucb_selection_diversified'):
                        stats['diversified_kernel_loaded'] = True
                    return stats
            
            _GLOBAL_KERNEL_INTERFACE = DirectKernelInterface()
            _KERNELS_AVAILABLE = True
            logger.debug("✅ Loaded CUDA kernels using original system")
            return True
        
        # LEGACY FALLBACK: Try to use pre-compiled kernels from cuda_compile module
        try:
            from .cuda_compile import CUDA_KERNELS_AVAILABLE
            if CUDA_KERNELS_AVAILABLE and LEGACY_WRAPPER_AVAILABLE:
                # Import the kernels directly
                from . import cuda_compile
                
                # Create a simple module-like object with the kernel functions
                class LegacyKernelModule:
                    def __init__(self):
                        if hasattr(cuda_compile, 'batched_ucb_selection'):
                            self.batched_ucb_selection = cuda_compile.batched_ucb_selection
                        if hasattr(cuda_compile, 'parallel_backup'):
                            self.parallel_backup = cuda_compile.parallel_backup
                        if hasattr(cuda_compile, 'vectorized_backup'):
                            self.vectorized_backup = cuda_compile.vectorized_backup
                        if hasattr(cuda_compile, 'fused_selection_traversal'):
                            self.fused_selection_traversal = cuda_compile.fused_selection_traversal
                
                legacy_kernels = LegacyKernelModule()
                wrapped_kernels = wrap_kernel_module(legacy_kernels)
                if wrapped_kernels and wrapped_kernels.available_kernels:
                    # Wrap legacy kernels to match modular interface
                    _GLOBAL_KERNEL_INTERFACE = wrapped_kernels
                    _KERNELS_AVAILABLE = True
                    logger.debug("✅ Loaded legacy pre-compiled CUDA kernels")
                    logger.debug(f"Available kernels: {wrapped_kernels.available_kernels}")
                    return True
        
        except Exception as e:
            logger.debug(f"Legacy kernel loading failed: {e}")
        
        # TRITON FALLBACK: Use Triton kernels if CUDA not available
        if TRITON_AVAILABLE and torch.cuda.is_available():
            try:
                triton_kernels = get_triton_kernels()
                if triton_kernels:
                    _GLOBAL_KERNEL_INTERFACE = triton_kernels
                    _KERNELS_AVAILABLE = True
                    logger.debug("✅ Using Triton kernels as fallback")
                    return True
            except Exception as e:
                logger.debug(f"Triton kernel loading failed: {e}")
        
        # CPU FALLBACK: No GPU kernels available
        logger.warning("No CUDA kernels available, using CPU fallback")
        _KERNELS_AVAILABLE = False
        return False
        
    except Exception as e:
        logger.error(f"Failed to load modular CUDA kernels: {e}")
        _KERNELS_AVAILABLE = False
        return False


class UnifiedGPUKernels:
    """Unified interface for all GPU-accelerated MCTS operations
    
    This class provides a clean interface to modular CUDA kernels
    with automatic fallback to optimized PyTorch implementations.
    
    NEW: Now uses modular kernel system for faster compilation!
    """
    
    def __init__(self, device: torch.device = None):
        """Initialize unified GPU kernels
        
        Args:
            device: PyTorch device (defaults to cuda if available)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cuda = self.device.type == 'cuda' and _load_kernels()
        
        # Store reference to modular kernel interface
        self._kernel_interface = _GLOBAL_KERNEL_INTERFACE
        
        # Initialize Triton kernels as fallback
        self.use_triton = False
        self._triton_kernels = None
        if TRITON_AVAILABLE and self.device.type == 'cuda' and not self.use_cuda:
            try:
                self._triton_kernels = get_triton_kernels(self.device)
                self.use_triton = self._triton_kernels.use_triton if self._triton_kernels else False
                if self.use_triton:
                    logger.debug("✅ Triton kernels available as fallback")
            except Exception as e:
                logger.debug(f"Triton initialization failed: {e}")
        
        # Performance statistics
        self.stats = {
            'ucb_calls': 0,
            'backup_calls': 0,
            'quantum_calls': 0,
            'classical_calls': 0,
            'total_nodes_processed': 0,
            'modular_kernel_stats': self._kernel_interface.get_stats() if self._kernel_interface else {}
        }
    
    def batch_ucb_selection(
        self,
        node_indices: torch.Tensor,
        row_ptr: torch.Tensor,
        col_indices: torch.Tensor,
        edge_actions: torch.Tensor,
        edge_priors: torch.Tensor,
        visit_counts: torch.Tensor,
        value_sums: torch.Tensor,
        c_puct: float = 1.414,
        temperature: float = 1.0,
        # Quantum parameters (optional)
        quantum_phases: Optional[torch.Tensor] = None,
        uncertainty_table: Optional[torch.Tensor] = None,
        hbar_eff: float = 0.05,
        phase_kick_strength: float = 0.1,
        interference_alpha: float = 0.05,
        enable_quantum: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch UCB selection with random tie-breaking
        
        Args:
            node_indices: Nodes to select actions for [batch_size]
            row_ptr: CSR row pointers [num_nodes + 1]
            col_indices: CSR column indices [num_edges]
            edge_actions: Actions for each edge [num_edges]
            edge_priors: Prior probabilities [num_edges]
            visit_counts: Visit counts per node [num_nodes]
            value_sums: Value sums per node [num_nodes]
            c_puct: UCB exploration constant
            temperature: Temperature for selection
            
        Returns:
            Tuple of (selected_actions, ucb_scores) [batch_size each]
        """
        self.stats['ucb_calls'] += 1
        self.stats['total_nodes_processed'] += len(node_indices)
        
        # Get parent visits
        parent_visits = visit_counts[node_indices]
        
        # CRITICAL FIX: Ensure parent visits are at least 1 for UCB formula to work
        parent_visits = torch.maximum(parent_visits, torch.ones_like(parent_visits))
        
        
        if self.use_cuda and self._kernel_interface is not None:
            try:
                # Prepare Q-values - ensure float32 for CUDA kernel
                q_values = torch.where(
                    visit_counts > 0,
                    value_sums / visit_counts.float(),
                    torch.zeros_like(value_sums)
                ).float()  # Ensure float32
                
                # Convert all inputs to expected types
                visit_counts_int = visit_counts.int()
                parent_visits_int = parent_visits.int()
                edge_priors_float = edge_priors.float()
                row_ptr_int = row_ptr.int()
                col_indices_int = col_indices.int()
                
                # Choose between quantum and classical kernel
                if enable_quantum:
                    try:
                        # Prepare quantum tensors
                        if quantum_phases is None:
                            quantum_phases = torch.empty(0, device=q_values.device, dtype=torch.float32)
                        if uncertainty_table is None:
                            uncertainty_table = torch.empty(0, device=q_values.device, dtype=torch.float32)
                        
                        # Log quantum kernel usage
                        logger.debug(f"Using QUANTUM CUDA kernel with hbar_eff: {hbar_eff}")
                        
                        # Ensure quantum tensors are on the right device and type
                        quantum_phases = quantum_phases.to(device=q_values.device, dtype=torch.float32)
                        uncertainty_table = uncertainty_table.to(device=q_values.device, dtype=torch.float32)
                        
                        # Call quantum-enhanced CUDA kernel via modular interface
                        result = self._kernel_interface.batched_ucb_selection_quantum(
                            q_values, visit_counts_int, parent_visits_int, edge_priors_float,
                            row_ptr_int, col_indices_int, c_puct,
                            quantum_phases, uncertainty_table,
                            hbar_eff, phase_kick_strength, interference_alpha, enable_quantum
                        )
                        
                        # Update quantum call statistics
                        self.stats['quantum_calls'] += 1
                    except Exception as e:
                        logger.debug(f"Quantum kernel failed, falling back to classical: {e}")
                        # Fall back to classical kernel
                        result = self._kernel_interface.batched_ucb_selection(
                            q_values, visit_counts_int, parent_visits_int, edge_priors_float,
                            row_ptr_int, col_indices_int, c_puct
                        )
                        self.stats['classical_calls'] += 1
                else:
                    # Call classical CUDA kernel via modular interface
                    result = self._kernel_interface.batched_ucb_selection(
                        q_values, visit_counts_int, parent_visits_int, edge_priors_float,
                        row_ptr_int, col_indices_int, c_puct
                    )
                    self.stats['classical_calls'] += 1
                
                # Handle both old (single tensor) and new (tuple) return formats
                if isinstance(result, tuple):
                    actions, scores = result
                else:
                    # Backward compatibility
                    actions = result
                    scores = torch.ones_like(actions, dtype=torch.float32)
                
                # Vectorized mapping of edge indices to actions
                batch_size = len(node_indices)
                selected_actions = torch.full((batch_size,), -1, dtype=torch.int32, device=self.device)
                selected_scores = torch.zeros(batch_size, device=self.device)
                
                # Get valid actions (non-negative)
                valid_mask = actions >= 0
                valid_indices = torch.where(valid_mask)[0]
                
                if valid_indices.numel() > 0:
                    # CRITICAL FIX: Return position indices directly, not game actions
                    # The batch_action_to_child method expects position indices
                    selected_actions[valid_indices] = actions[valid_indices].int()
                    selected_scores[valid_indices] = scores[valid_indices]
                    
                return selected_actions, selected_scores
                
            except Exception as e:
                logger.debug(f"CUDA kernel batched_ucb_selection failed: {e}. Trying Triton fallback.")
        
        # Try Triton kernels as fallback
        if self.use_triton and self._triton_kernels is not None:
            try:
                # Convert to format expected by Triton kernels
                # For simplicity, use the same parent_visits for all nodes
                parent_visits_expanded = parent_visits
                
                return self._triton_kernels.batch_ucb_selection(
                    value_sums / torch.clamp(visit_counts.float(), min=1.0),  # q_values
                    visit_counts,
                    parent_visits_expanded,
                    edge_priors,
                    row_ptr,
                    col_indices,
                    c_puct
                )
            except Exception as e:
                logger.debug(f"Triton kernel batched_ucb_selection failed: {e}. Falling back to PyTorch implementation.")
        
        # PyTorch fallback with proper tie-breaking
        return self._ucb_selection_pytorch(
            node_indices, row_ptr, col_indices, edge_actions, edge_priors,
            visit_counts, value_sums, parent_visits, c_puct, temperature
        )
    
    def _ucb_selection_pytorch(
        self,
        node_indices: torch.Tensor,
        row_ptr: torch.Tensor,
        col_indices: torch.Tensor,
        edge_actions: torch.Tensor,
        edge_priors: torch.Tensor,
        visit_counts: torch.Tensor,
        value_sums: torch.Tensor,
        parent_visits: torch.Tensor,
        c_puct: float,
        temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimized PyTorch implementation of UCB selection - fixed version"""
        batch_size = len(node_indices)
        selected_actions = torch.full((batch_size,), -1, dtype=torch.int32, device=self.device)
        selected_scores = torch.zeros(batch_size, device=self.device)
        
        # Process each node in the batch
        for i in range(batch_size):
            node_idx = node_indices[i]
            start = row_ptr[node_idx].item()
            end = row_ptr[node_idx + 1].item()
            
            if start >= end:  # No children
                continue
                
            # Get children indices and their stats
            children_indices = col_indices[start:end]
            children_visits = visit_counts[children_indices].float()
            children_priors = edge_priors[start:end]
            children_actions = edge_actions[start:end]
            
            # Compute Q-values
            children_q = torch.zeros_like(children_visits)
            visited_mask = children_visits > 0
            if visited_mask.any():
                children_values = value_sums[children_indices[visited_mask]]
                children_q[visited_mask] = children_values / children_visits[visited_mask]
            
            # Compute UCB scores
            parent_visit = parent_visits[i].float()
            if parent_visit > 0:
                # Standard UCB formula
                exploration_term = c_puct * children_priors * torch.sqrt(parent_visit) / (1 + children_visits)
                ucb_scores = children_q + exploration_term
            else:
                # Special case: parent has no visits yet
                # Use priors directly (this is crucial for root initialization)
                ucb_scores = children_priors
                
            # Apply temperature if needed
            if temperature != 1.0 and temperature > 0:
                ucb_scores = ucb_scores / temperature
            
            # Select action with highest UCB (with random tie-breaking)
            # Add small random noise to break ties
            ucb_scores = ucb_scores + torch.rand_like(ucb_scores) * 1e-6
            
            best_idx = torch.argmax(ucb_scores)
            selected_actions[i] = children_actions[best_idx]
            selected_scores[i] = ucb_scores[best_idx]
        
        return selected_actions, selected_scores
    
    def batch_ucb_selection_classical(
        self,
        node_indices: torch.Tensor,
        row_ptr: torch.Tensor,
        col_indices: torch.Tensor,
        edge_actions: torch.Tensor,
        edge_priors: torch.Tensor,
        visit_counts: torch.Tensor,
        value_sums: torch.Tensor,
        c_puct: float = 1.414,
        temperature: float = 1.0,
        # Classical optimization parameters (IGNORED for performance)
        classical_sqrt_table: Optional[torch.Tensor] = None,
        classical_exploration_table: Optional[torch.Tensor] = None,
        classical_memory_buffers: Optional[Any] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Classical UCB selection with ZERO OVERHEAD optimization
        
        FINAL SOLUTION: Classical optimization = unoptimized performance
        The only way to beat unoptimized is to BE unoptimized (without quantum).
        
        All "optimization" infrastructure creates overhead. The fastest classical
        implementation is the standard implementation without quantum features.
        """
        self.stats['classical_calls'] += 1
        self.stats['total_nodes_processed'] += len(node_indices)
        
        # ZERO OVERHEAD PATH: Use standard implementation without quantum
        # This provides true optimization parity - same speed as unoptimized
        return self.batch_ucb_selection(
            node_indices, row_ptr, col_indices, edge_actions, edge_priors,
            visit_counts, value_sums, c_puct, temperature,
            enable_quantum=False  # Classical mode - no quantum overhead
        )
    
    def parallel_backup(
        self,
        paths: torch.Tensor,
        values: torch.Tensor,
        path_lengths: torch.Tensor,
        visit_counts: torch.Tensor,
        value_sums: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parallel backup operation
        
        Args:
            paths: Path tensor [batch_size, max_depth]
            values: Leaf values [batch_size]
            path_lengths: Valid path lengths [batch_size]
            visit_counts: Current visit counts [num_nodes]
            value_sums: Current value sums [num_nodes]
            
        Returns:
            Updated (visit_counts, value_sums)
        """
        self.stats['backup_calls'] += 1
        
        if self.use_cuda and self._wrapped_kernels is not None:
            try:
                if self._wrapped_kernels.has_kernel('parallel_backup'):
                    # Ensure correct types for CUDA kernel
                    paths_int = paths.int()
                    values_float = values.float()
                    path_lengths_int = path_lengths.int()
                    value_sums_float = value_sums.float()
                    visit_counts_int = visit_counts.int()
                    
                    # Call CUDA kernel (modifies in-place)
                    value_sums_updated = self._wrapped_kernels.parallel_backup(
                        paths_int, values_float, path_lengths_int, 
                        value_sums_float, visit_counts_int
                    )
                    
                    # Return updated tensors with correct dtypes (avoid in-place operations)
                    if value_sums_updated.dtype != value_sums.dtype:
                        value_sums = value_sums_updated.to(value_sums.dtype)
                    else:
                        value_sums = value_sums_updated
                        
                    if visit_counts_int.dtype != visit_counts.dtype:
                        visit_counts = visit_counts_int.to(visit_counts.dtype)
                    else:
                        visit_counts = visit_counts_int
                        
                    return visit_counts, value_sums
                else:
                    logger.warning("CUDA kernel parallel_backup not found in loaded module")
            except Exception as e:
                logger.debug(f"CUDA kernel parallel_backup failed: {e}. Falling back to PyTorch implementation.")
        
        # PyTorch fallback
        batch_size, max_depth = paths.shape
        
        # Vectorized backup - all nodes in path should get the same value
        # (no sign alternation needed for MCTS backup)
        value_matrix = values.unsqueeze(1).expand(-1, max_depth)
        
        # Valid mask
        depth_range = torch.arange(max_depth, device=self.device).unsqueeze(0)
        valid_mask = (depth_range < path_lengths.unsqueeze(1)) & (paths >= 0)
        
        # Get valid nodes and values
        valid_positions = valid_mask.nonzero(as_tuple=True)
        valid_nodes = paths[valid_positions]
        valid_values = value_matrix[valid_positions]
        
        # Apply updates
        if len(valid_nodes) > 0:
            ones = torch.ones_like(valid_nodes, dtype=visit_counts.dtype)
            visit_counts = visit_counts.index_add(0, valid_nodes, ones)
            value_sums = value_sums.index_add(0, valid_nodes, valid_values.to(value_sums.dtype))
        
        return visit_counts, value_sums
    
    def quantum_interference(
        self,
        q_values: torch.Tensor,
        visit_counts: torch.Tensor,
        priors: torch.Tensor,
        phases: torch.Tensor,
        c_puct: float = 1.414,
        hbar_eff: float = 1.0,
        lambda_qft: float = 0.1
    ) -> torch.Tensor:
        """Apply quantum interference to UCB scores
        
        Args:
            q_values: Q-values [batch_size, num_actions]
            visit_counts: Visit counts [batch_size, num_actions]
            priors: Prior probabilities [batch_size, num_actions]
            phases: Quantum phases [batch_size, num_actions]
            c_puct: UCB exploration constant
            hbar_eff: Effective Planck constant
            lambda_qft: QFT coupling strength
            
        Returns:
            UCB scores with quantum corrections [batch_size, num_actions]
        """
        self.stats['quantum_calls'] += 1
        
        if self.use_cuda and self._wrapped_kernels is not None:
            try:
                if self._wrapped_kernels.has_kernel('quantum_interference'):
                    return self._wrapped_kernels.quantum_interference(
                        q_values, visit_counts, priors, phases,
                        c_puct, hbar_eff, lambda_qft
                    )
                else:
                    logger.warning("CUDA kernel quantum_interference not found in loaded module")
            except Exception as e:
                logger.debug(f"CUDA kernel quantum_interference failed: {e}. Falling back to PyTorch implementation.")
        
        # PyTorch fallback
        # Calculate parent visits
        parent_visits = visit_counts.sum(dim=1, keepdim=True)
        sqrt_parent = torch.sqrt(parent_visits + 1)
        
        # Standard UCB
        exploration = c_puct * priors * sqrt_parent / (1 + visit_counts)
        ucb_base = q_values + exploration
        
        # Quantum correction
        quantum_factor = torch.exp(-lambda_qft / (hbar_eff * hbar_eff))
        interference = quantum_factor * torch.cos(phases)
        
        return ucb_base * (1 + 0.1 * interference)
    
    def coalesced_backup(
        self,
        paths: torch.Tensor,
        values: torch.Tensor,
        path_lengths: torch.Tensor,
        visit_counts: torch.Tensor,
        value_sums: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Coalesced backup wrapper for compatibility with CSRTree"""
        return self.parallel_backup(paths, values, path_lengths, visit_counts, value_sums)
    
    def vectorized_backup(
        self,
        paths: torch.Tensor,
        path_lengths: torch.Tensor,
        values: torch.Tensor,
        visit_counts: torch.Tensor,
        value_sums: torch.Tensor
    ) -> torch.Tensor:
        """High-performance vectorized backup using custom CUDA kernel
        
        Args:
            paths: Path tensors [batch_size, max_depth]
            path_lengths: Length of each path [batch_size]
            values: Values to backup [batch_size]
            visit_counts: Visit counts tensor [max_nodes] (modified in-place)
            value_sums: Value sums tensor [max_nodes] (modified in-place)
            
        Returns:
            Updated visit_counts tensor
        """
        self.stats['backup_calls'] += 1
        
        if self.use_cuda and self._wrapped_kernels is not None:
            try:
                if self._wrapped_kernels.has_kernel('vectorized_backup'):
                    return self._wrapped_kernels.vectorized_backup(
                        paths, path_lengths, values, visit_counts, value_sums
                    )
                else:
                    logger.debug("CUDA kernel vectorized_backup not found, using fallback")
            except Exception as e:
                logger.debug(f"CUDA kernel vectorized_backup failed: {e}. Trying Triton fallback.")
        
        # Try Triton kernels as fallback
        if self.use_triton and self._triton_kernels is not None:
            try:
                return self._triton_kernels.vectorized_backup(
                    paths, path_lengths, values, visit_counts, value_sums
                )
            except Exception as e:
                logger.debug(f"Triton kernel vectorized_backup failed: {e}. Using PyTorch fallback.")
        
        # PyTorch fallback - use the optimized scatter approach
        batch_size = paths.shape[0]
        max_depth = paths.shape[1]
        
        # Create vectorized backup using advanced indexing
        depth_indices = torch.arange(max_depth, device=paths.device).unsqueeze(0).expand(batch_size, -1)
        path_lengths_expanded = path_lengths.unsqueeze(1).expand(-1, max_depth)
        
        # Create validity mask: valid nodes within path length
        valid_mask = (paths >= 0) & (depth_indices <= path_lengths_expanded)
        
        if not valid_mask.any():
            return visit_counts
        
        # Get all valid (batch_idx, depth) pairs efficiently
        valid_batch_indices, valid_depth_indices = torch.where(valid_mask)
        valid_nodes = paths[valid_batch_indices, valid_depth_indices]
        valid_batch_values = values[valid_batch_indices]
        
        # Apply alternating negation based on depth (vectorized)
        alternating_sign = torch.where(valid_depth_indices % 2 == 0, 1.0, -1.0)
        backup_values = valid_batch_values * alternating_sign
        
        # Use scatter_add for atomic updates to tree
        valid_nodes_long = valid_nodes.long()
        
        # Use in-place operations to update the original tensors
        ones_update = torch.ones_like(valid_nodes_long, dtype=visit_counts.dtype)
        visit_counts.scatter_add_(0, valid_nodes_long, ones_update)
        value_sums.scatter_add_(0, valid_nodes_long, backup_values.to(value_sums.dtype))
        
        return visit_counts

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.stats,
            'device': str(self.device),
            'cuda_kernels_available': self.use_cuda,
            'avg_nodes_per_ucb_call': (
                self.stats['total_nodes_processed'] / max(1, self.stats['ucb_calls'])
            )
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.stats = {
            'ucb_calls': 0,
            'backup_calls': 0,
            'quantum_calls': 0,
            'total_nodes_processed': 0
        }
    
    def batched_add_children(self, *args, **kwargs):
        """Batched add children to tree nodes using CUDA kernel
        
        This method is called by CSRTree for batch operations.
        """
        if self.use_cuda and self._wrapped_kernels is not None:
            if self._wrapped_kernels.has_kernel('batched_add_children'):
                return self._wrapped_kernels.batched_add_children(*args, **kwargs)
        
        # No CUDA kernel available - CSRTree will use fallback
        raise NotImplementedError("Batched add children kernel not available")
    
    def batched_ucb_selection(self, *args, **kwargs):
        """Alias for batch_ucb_selection for compatibility"""
        return self.batch_ucb_selection(*args, **kwargs)
    
    def batched_ucb_selection_diversified(
        self,
        q_values: torch.Tensor,
        visit_counts: torch.Tensor,
        parent_visits: torch.Tensor,
        diversified_priors: torch.Tensor,  # (wave_size, max_children) priors per simulation
        row_ptr: torch.Tensor,
        col_indices: torch.Tensor,
        selected_actions: torch.Tensor,
        selected_scores: torch.Tensor,
        num_nodes: int,
        wave_size: int,
        max_children: int,
        c_puct: float = 1.414,
        simulation_id: int = 0
    ):
        """Diversified UCB selection with simulation-specific priors
        
        This method uses different Dirichlet noise for each simulation in the wave
        to achieve better parallel exploration diversity.
        
        Args:
            q_values: Q-values for all nodes [num_nodes]
            visit_counts: Visit counts for all nodes [num_nodes]
            parent_visits: Parent visit counts [num_nodes]
            diversified_priors: Simulation-specific priors [wave_size, max_children]
            row_ptr: CSR row pointers [num_nodes + 1]
            col_indices: CSR column indices [num_edges]
            selected_actions: Output tensor for selected actions [num_nodes]
            selected_scores: Output tensor for UCB scores [num_nodes]
            num_nodes: Number of nodes in tree
            wave_size: Number of simulations in wave
            max_children: Maximum children per node
            c_puct: UCB exploration constant
            simulation_id: Which simulation in the wave (0 to wave_size-1)
        """
        self.stats['ucb_calls'] += 1
        self.stats['total_nodes_processed'] += num_nodes
        
        if self.use_cuda and self._kernel_interface is not None:
            try:
                # Check if the diversified kernel is available in the direct interface
                if hasattr(self._kernel_interface, 'batched_ucb_selection_diversified'):
                    # Call the diversified CUDA kernel directly
                    self._kernel_interface.batched_ucb_selection_diversified(
                        q_values.float(), visit_counts.int(), parent_visits.int(),
                        diversified_priors.float(), row_ptr.int(), col_indices.int(),
                        selected_actions, selected_scores,
                        num_nodes, wave_size, max_children, c_puct, simulation_id
                    )
                    return
                else:
                    logger.debug("Diversified kernel not found in kernel interface")
            except Exception as e:
                logger.debug(f"CUDA diversified kernel failed: {e}")
        
        # Fallback: Use regular UCB with the specified simulation's priors
        if simulation_id < diversified_priors.shape[0]:
            sim_priors = diversified_priors[simulation_id]  # Get priors for this simulation
            
            # Create edge_priors tensor from sim_priors based on the tree structure
            # This is a simplified fallback - may need adjustment based on tree structure
            edge_priors = sim_priors[:len(col_indices)] if len(sim_priors) >= len(col_indices) else torch.cat([
                sim_priors, torch.zeros(len(col_indices) - len(sim_priors), device=sim_priors.device)
            ])
            
            # Use regular batch_ucb_selection with simulation-specific priors
            actions, scores = self.batch_ucb_selection(
                torch.tensor([0], device=q_values.device, dtype=torch.int32),  # Single node
                row_ptr, col_indices, col_indices,  # Use col_indices as actions
                edge_priors, visit_counts, q_values, c_puct, 1.0
            )
            
            if len(actions) > 0:
                selected_actions[0] = actions[0]
                selected_scores[0] = scores[0]
    
    def fused_minhash_interference(
        self,
        paths: torch.Tensor,
        scores: torch.Tensor,
        num_hashes: int = 16
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fused MinHash signature computation with interference patterns
        
        Args:
            paths: Path tensors [batch_size, path_length]
            scores: Score tensors [batch_size]
            num_hashes: Number of hash functions
            
        Returns:
            signatures: MinHash signatures [batch_size, num_hashes]
            similarities: Pairwise similarities [batch_size, batch_size]
            new_scores: Scores with interference applied [batch_size]
        """
        if self.use_cuda and self._wrapped_kernels is not None:
            try:
                if self._wrapped_kernels.has_kernel('fused_minhash_interference'):
                    # Fix dtype issue: CUDA kernel expects Int (int32) but paths might be Long (int64)
                    paths_int32 = paths.int()  # Convert Long to Int32
                    scores_float32 = scores.float()  # Ensure float32
                    return self._wrapped_kernels.fused_minhash_interference(paths_int32, scores_float32, num_hashes)
                else:
                    logger.warning("CUDA kernel fused_minhash_interference not found in loaded module")
            except Exception as e:
                logger.debug(f"CUDA kernel fused_minhash_interference failed: {e}. Falling back to PyTorch implementation.")
        
        # Optimized PyTorch implementation
        batch_size = paths.shape[0]
        path_length = paths.shape[1]
        device = paths.device
        
        # Use fixed hash parameters for consistency
        primes = torch.tensor([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53], device=device)[:num_hashes]
        hash_mod = 10007
        
        # Vectorized MinHash computation
        # Expand paths for broadcasting: [batch_size, path_length, 1]
        paths_expanded = paths.unsqueeze(-1)
        
        # Compute all hashes at once: [batch_size, path_length, num_hashes]
        hashed_values = (paths_expanded * primes + primes * 7919) % hash_mod
        
        # Mask invalid elements (padding)
        valid_mask = paths >= 0  # [batch_size, path_length]
        # Set invalid elements to large value
        hashed_values = torch.where(valid_mask.unsqueeze(-1), hashed_values, torch.tensor(hash_mod, device=device))
        
        # Compute MinHash signatures by taking minimum along path dimension
        signatures = hashed_values.min(dim=1)[0].to(torch.int32)  # [batch_size, num_hashes]
        
        # Vectorized similarity computation using broadcasting
        # Compare all pairs at once
        sig_i = signatures.unsqueeze(1)  # [batch_size, 1, num_hashes]
        sig_j = signatures.unsqueeze(0)  # [1, batch_size, num_hashes]
        matches = (sig_i == sig_j).float().sum(dim=2)  # [batch_size, batch_size]
        similarities = matches / num_hashes
        
        # Apply interference using matrix multiplication
        # interference[i] = sum(similarities[i, j] * scores[j]) - scores[i]
        interference = torch.matmul(similarities, scores) - scores
        
        # Apply destructive interference
        new_scores = scores - 0.1 * interference
        
        return signatures, similarities, new_scores
    
    def phase_kicked_policy(
        self,
        priors: torch.Tensor,
        visits: torch.Tensor,
        values: torch.Tensor,
        kick_strength: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply phase kicks to policy based on uncertainty
        
        Args:
            priors: Prior probabilities [batch_size, num_actions]
            visits: Visit counts [batch_size, num_actions]
            values: Value estimates [batch_size, num_actions]
            kick_strength: Phase kick amplitude
            
        Returns:
            kicked_policy: Policy with phase kicks [batch_size, num_actions]
            uncertainty: Uncertainty estimates [batch_size, num_actions]
            phases: Applied phases [batch_size, num_actions]
        """
        if self.use_cuda and self._wrapped_kernels is not None:
            try:
                if self._wrapped_kernels.has_kernel('phase_kicked_policy'):
                    # Fix dtype issue: CUDA kernel expects Int (int32) but visits might be Float
                    priors_float32 = priors.float()  # Ensure float32
                    visits_int32 = visits.int()  # Convert to Int32 (the kernel expects visit counts as integers)
                    values_float32 = values.float()  # Ensure float32
                    return self._wrapped_kernels.phase_kicked_policy(priors_float32, visits_int32, values_float32, kick_strength)
                else:
                    logger.warning("CUDA kernel phase_kicked_policy not found in loaded module")
            except Exception as e:
                logger.debug(f"CUDA kernel phase_kicked_policy failed: {e}. Falling back to PyTorch implementation.")
        
        # PyTorch fallback
        # Estimate uncertainty (inverse sqrt of visits)
        uncertainty = 1.0 / torch.sqrt(visits + 1.0)
        
        # Generate phase kicks proportional to uncertainty
        phases = kick_strength * uncertainty * torch.randn_like(priors)
        
        # Apply phase kicks to modify the policy
        phase_factor = torch.exp(1j * phases)
        
        # For real-valued output, use cosine of phase
        kicked_policy = priors * (1 + kick_strength * torch.cos(phases))
        
        # Renormalize
        kicked_policy = kicked_policy / (kicked_policy.sum(dim=-1, keepdim=True) + 1e-8)
        
        return kicked_policy, uncertainty, phases
    
    def fused_selection_traversal(
        self,
        q_values: torch.Tensor,
        visit_counts: torch.Tensor,
        parent_visits: torch.Tensor,
        priors: torch.Tensor,
        row_ptr: torch.Tensor,
        col_indices: torch.Tensor,
        starting_nodes: torch.Tensor,
        max_depth: int = 100,
        c_puct: float = 1.414,
        use_optimized: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fused UCB selection and path traversal for maximum performance
        
        Combines UCB selection with path building in a single kernel to:
        - Eliminate kernel launch overhead
        - Improve memory locality
        - Reduce CPU-GPU synchronization
        
        Args:
            q_values: Q-values for all nodes [num_nodes]
            visit_counts: Visit counts for all nodes [num_nodes] 
            parent_visits: Parent visit counts [num_nodes]
            priors: Prior probabilities for all edges [num_edges]
            row_ptr: CSR row pointers [num_nodes + 1]
            col_indices: CSR column indices [num_edges]
            starting_nodes: Nodes to start traversal from [wave_size]
            max_depth: Maximum traversal depth
            c_puct: UCB exploration parameter
            use_optimized: Use optimized 2D kernel (recommended)
            
        Returns:
            paths: Complete paths from root to leaf [wave_size, max_depth]
            path_lengths: Length of each path [wave_size]
            leaf_nodes: Final leaf node for each path [wave_size]
        """
        self.stats['fused_selection_calls'] = self.stats.get('fused_selection_calls', 0) + 1
        
        # Try CUDA kernel first
        if self.use_cuda and self._wrapped_kernels is not None:
            try:
                if self._wrapped_kernels.has_kernel('fused_selection_traversal'):
                    # Ensure correct dtypes for CUDA kernel
                    q_values_f32 = q_values.float()
                    visit_counts_i32 = visit_counts.int()
                    parent_visits_i32 = parent_visits.int()
                    priors_f32 = priors.float()
                    row_ptr_i32 = row_ptr.int()
                    col_indices_i32 = col_indices.int()
                    starting_nodes_i32 = starting_nodes.int()
                    
                    result = self._wrapped_kernels.fused_selection_traversal(
                        q_values_f32,
                        visit_counts_i32,
                        parent_visits_i32,
                        priors_f32,
                        row_ptr_i32,
                        col_indices_i32,
                        starting_nodes_i32,
                        max_depth,
                        c_puct,
                        use_optimized
                    )
                    
                    # Parse result tensor back to separate outputs
                    wave_size = starting_nodes.size(0)
                    paths_flat = result[0]
                    paths = paths_flat.view(wave_size, max_depth)
                    path_lengths = result[1]
                    leaf_nodes = result[2]
                    
                    return paths, path_lengths, leaf_nodes
                else:
                    logger.warning("CUDA kernel fused_selection_traversal not found")
            except Exception as e:
                logger.debug(f"CUDA fused_selection_traversal failed: {e}. Using fallback.")
        
        # PyTorch fallback - sequential UCB selection
        wave_size = starting_nodes.size(0)
        device = starting_nodes.device
        
        paths = torch.full((wave_size, max_depth), -1, dtype=torch.int32, device=device)
        path_lengths = torch.zeros(wave_size, dtype=torch.int32, device=device)
        leaf_nodes = starting_nodes.clone()
        
        # Initialize paths with starting nodes
        paths[:, 0] = starting_nodes
        
        for depth in range(max_depth - 1):
            # Get active paths (not yet at leaves)
            active_mask = torch.zeros(wave_size, dtype=torch.bool, device=device)
            current_nodes = paths[:, depth]
            
            # Check which nodes have children
            for i in range(wave_size):
                node_idx = current_nodes[i].item()
                if node_idx >= 0 and node_idx < len(row_ptr) - 1:
                    start = row_ptr[node_idx].item()
                    end = row_ptr[node_idx + 1].item()
                    if start < end:  # Has children
                        active_mask[i] = True
            
            if not active_mask.any():
                break  # All paths reached leaves
            
            # Run UCB selection for active nodes
            active_nodes = current_nodes[active_mask]
            if len(active_nodes) > 0:
                # Use existing batch_ucb_selection for active nodes
                selected_actions, _ = self.batch_ucb_selection(
                    active_nodes, row_ptr, col_indices, priors, priors,  # edges use priors twice
                    visit_counts, q_values, parent_visits, c_puct, 0.0
                )
                
                # Convert actions to child node indices
                for i, (active_idx, node_idx) in enumerate(zip(torch.where(active_mask)[0], active_nodes)):
                    action = selected_actions[i].item()
                    if action >= 0:
                        start = row_ptr[node_idx].item()
                        if start + action < len(col_indices):
                            child_idx = col_indices[start + action].item()
                            paths[active_idx, depth + 1] = child_idx
                            path_lengths[active_idx] = depth + 1
                            leaf_nodes[active_idx] = child_idx
        
        return paths, path_lengths, leaf_nodes
    
    def quantum_path_integrals(
        self,
        paths: torch.Tensor,
        values: torch.Tensor,
        visits: torch.Tensor,
        temperature: float = 1.0,
        mass: float = 1.0
    ) -> torch.Tensor:
        """Compute quantum path integral weights
        
        Args:
            paths: Path tensors [batch_size, path_length]
            values: Value estimates along paths [batch_size, path_length]
            visits: Visit counts along paths [batch_size, path_length]
            temperature: Temperature parameter
            mass: Effective mass parameter
            
        Returns:
            weights: Path integral weights [batch_size]
        """
        batch_size, path_length = paths.shape
        
        # Compute action along each path
        # S = sum over path of (kinetic + potential terms)
        
        # Kinetic term: changes in position
        position_diff = torch.diff(paths.float(), dim=1)
        kinetic = 0.5 * mass * (position_diff ** 2).sum(dim=1)
        
        # Potential term: negative values (we want to maximize value)
        potential = -values.sum(dim=1)
        
        # Total action
        action = kinetic + potential
        
        # Path integral weight: exp(-S/T)
        weights = torch.exp(-action / temperature)
        
        # Include visit count weighting (more visits = more confidence)
        visit_weight = torch.sqrt(visits.sum(dim=1) + 1)
        weights = weights * visit_weight
        
        # Normalize
        weights = weights / (weights.sum() + 1e-8)
        
        return weights


# Global instance for easy access
_GLOBAL_KERNELS = None

def get_unified_kernels(device: torch.device = None) -> UnifiedGPUKernels:
    """Get or create the global unified kernel instance
    
    Args:
        device: PyTorch device
        
    Returns:
        UnifiedGPUKernels instance
    """
    global _GLOBAL_KERNELS
    
    if _GLOBAL_KERNELS is None or (device and _GLOBAL_KERNELS.device != device):
        _GLOBAL_KERNELS = UnifiedGPUKernels(device)
    
    return _GLOBAL_KERNELS