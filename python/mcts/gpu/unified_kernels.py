"""Unified GPU kernel interface for MCTS

This module provides a clean, consolidated interface to all GPU kernels,
replacing the multiple legacy implementations with a single, optimized version.
"""

import torch
import logging
from typing import Tuple, Optional, Dict, Any
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Check for compiled kernels
_UNIFIED_KERNELS = None
_KERNELS_AVAILABLE = False

def _load_kernels():
    """Load compiled CUDA kernels"""
    global _UNIFIED_KERNELS, _KERNELS_AVAILABLE
    
    if _UNIFIED_KERNELS is not None:
        return _KERNELS_AVAILABLE
    
    try:
        # Try to load pre-compiled kernels
        kernel_paths = [
            Path(__file__).parent / "mcts_cuda_kernels.cpython-312-x86_64-linux-gnu.so",
            Path(__file__).parent / "unified_cuda_kernels.so",
            Path(__file__).parent.parent.parent / "build_cuda" / "unified_cuda_kernels.so",
        ]
        
        for path in kernel_paths:
            if path.exists():
                # Load the module directly instead of using torch.ops
                import importlib.util
                spec = importlib.util.spec_from_file_location('mcts_cuda_kernels', str(path))
                _UNIFIED_KERNELS = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_UNIFIED_KERNELS)
                _KERNELS_AVAILABLE = True
                logger.debug(f"Loaded CUDA kernels from {path}")
                break
                
        if not _KERNELS_AVAILABLE and torch.cuda.is_available():
            # Try JIT compilation as fallback
            try:
                from torch.utils.cpp_extension import load
                _UNIFIED_KERNELS = load(
                    name='unified_cuda_kernels',
                    sources=[str(Path(__file__).parent / "unified_cuda_kernels.cu")],
                    verbose=False
                )
                _KERNELS_AVAILABLE = True
                logger.debug("JIT compiled CUDA kernels")
            except Exception as e:
                logger.debug(f"JIT compilation failed: {e}")
                
    except Exception as e:
        logger.debug(f"Failed to load CUDA kernels: {e}")
    
    return _KERNELS_AVAILABLE


class UnifiedGPUKernels:
    """Unified interface for all GPU-accelerated MCTS operations
    
    This class consolidates all GPU kernels into a single, clean interface
    with automatic fallback to optimized PyTorch implementations.
    """
    
    def __init__(self, device: torch.device = None):
        """Initialize unified GPU kernels
        
        Args:
            device: PyTorch device (defaults to cuda if available)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cuda = self.device.type == 'cuda' and _load_kernels()
        
        # Logging removed for performance
            
        # Performance statistics
        self.stats = {
            'ucb_calls': 0,
            'backup_calls': 0,
            'quantum_calls': 0,
            'total_nodes_processed': 0
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
        temperature: float = 1.0
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
        
        if self.use_cuda and _UNIFIED_KERNELS is not None:
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
                
                # Call CUDA kernel directly from loaded module
                # The kernel now returns (actions, scores) tuple
                result = _UNIFIED_KERNELS.batched_ucb_selection(
                    q_values, visit_counts_int, parent_visits_int, edge_priors_float,
                    row_ptr_int, col_indices_int, c_puct
                )
                
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
                    # Vectorized computation of start indices
                    starts = row_ptr[node_indices[valid_indices]]
                    edge_positions = starts + actions[valid_indices]
                    selected_actions[valid_indices] = edge_actions[edge_positions]
                    selected_scores[valid_indices] = scores[valid_indices]
                    
                return selected_actions, selected_scores
                
            except Exception as e:
                pass  # Silently fall back
        
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
        
        if self.use_cuda:
            try:
                # Try different kernel namespaces
                kernel_func = None
                if hasattr(torch.ops, 'custom_cuda_ops') and hasattr(torch.ops.custom_cuda_ops, 'parallel_backup'):
                    kernel_func = torch.ops.custom_cuda_ops.parallel_backup
                elif hasattr(torch.ops, 'unified_cuda_kernels') and hasattr(torch.ops.unified_cuda_kernels, 'parallel_backup'):
                    kernel_func = torch.ops.unified_cuda_kernels.parallel_backup
                elif hasattr(torch.ops, 'mcts_cuda_kernels') and hasattr(torch.ops.mcts_cuda_kernels, 'parallel_backup'):
                    kernel_func = torch.ops.mcts_cuda_kernels.parallel_backup
                
                if kernel_func:
                    # Ensure correct types for CUDA kernel
                    paths_int = paths.int()
                    values_float = values.float()
                    path_lengths_int = path_lengths.int()
                    value_sums_float = value_sums.float()
                    visit_counts_int = visit_counts.int()
                    
                    # Call CUDA kernel (modifies in-place)
                    value_sums_updated = kernel_func(
                        paths_int, values_float, path_lengths_int, 
                        value_sums_float, visit_counts_int
                    )
                    
                    # Copy back to original tensors if needed
                    if value_sums_updated.dtype != value_sums.dtype:
                        value_sums.copy_(value_sums_updated.to(value_sums.dtype))
                    else:
                        value_sums = value_sums_updated
                        
                    if visit_counts_int.dtype != visit_counts.dtype:
                        visit_counts.copy_(visit_counts_int.to(visit_counts.dtype))
                        
                    return visit_counts, value_sums
            except Exception as e:
                logger.debug(f"CUDA backup failed, using fallback: {e}")
        
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
        
        if self.use_cuda:
            try:
                # Try different kernel namespaces
                kernel_func = None
                if hasattr(torch.ops, 'custom_cuda_ops') and hasattr(torch.ops.custom_cuda_ops, 'quantum_interference'):
                    kernel_func = torch.ops.custom_cuda_ops.quantum_interference
                elif hasattr(torch.ops, 'unified_cuda_kernels') and hasattr(torch.ops.unified_cuda_kernels, 'quantum_interference'):
                    kernel_func = torch.ops.unified_cuda_kernels.quantum_interference
                elif hasattr(torch.ops, 'mcts_cuda_kernels') and hasattr(torch.ops.mcts_cuda_kernels, 'quantum_interference'):
                    kernel_func = torch.ops.mcts_cuda_kernels.quantum_interference
                
                if kernel_func:
                    return kernel_func(
                        q_values, visit_counts, priors, phases,
                        c_puct, hbar_eff, lambda_qft
                    )
            except Exception as e:
                logger.debug(f"CUDA quantum kernel failed, using fallback: {e}")
        
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
        if self.use_cuda and _UNIFIED_KERNELS is not None:
            if hasattr(_UNIFIED_KERNELS, 'batched_add_children'):
                return _UNIFIED_KERNELS.batched_add_children(*args, **kwargs)
        
        # No CUDA kernel available - CSRTree will use fallback
        raise NotImplementedError("Batched add children kernel not available")
    
    def batched_ucb_selection(self, *args, **kwargs):
        """Alias for batch_ucb_selection for compatibility"""
        return self.batch_ucb_selection(*args, **kwargs)
    
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
        if self.use_cuda:
            try:
                # Try different kernel namespaces
                kernel_func = None
                if hasattr(torch.ops, 'mcts_cuda_kernels') and hasattr(torch.ops.mcts_cuda_kernels, 'fused_minhash_interference'):
                    kernel_func = torch.ops.mcts_cuda_kernels.fused_minhash_interference
                
                if kernel_func:
                    return kernel_func(paths, scores, num_hashes)
            except Exception as e:
                logger.debug(f"CUDA minhash kernel failed, using fallback: {e}")
        
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
        if self.use_cuda:
            try:
                # Try different kernel namespaces
                kernel_func = None
                if hasattr(torch.ops, 'mcts_cuda_kernels') and hasattr(torch.ops.mcts_cuda_kernels, 'phase_kicked_policy'):
                    kernel_func = torch.ops.mcts_cuda_kernels.phase_kicked_policy
                
                if kernel_func:
                    return kernel_func(priors, visits, values, kick_strength)
            except Exception as e:
                logger.debug(f"CUDA phase kick kernel failed, using fallback: {e}")
        
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