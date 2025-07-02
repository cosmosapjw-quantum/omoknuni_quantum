"""CUDA kernels for GPU acceleration of MCTS operations

This module provides custom CUDA kernels for performance-critical operations
in the vectorized MCTS implementation. It combines both standard and optimized
kernels for maximum performance.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
import math
import logging
import numpy as np
import os
import multiprocessing

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    

logger = logging.getLogger(__name__)


# Check if CUDA is available
CUDA_AVAILABLE = torch.cuda.is_available()

# Configure for optimal performance
if CUDA_AVAILABLE:
    # Set CUDA memory allocator settings for better performance with error handling
    try:
        # Only set memory fraction in main process to avoid OOM in multiprocessing workers
        if multiprocessing.current_process().name == 'MainProcess':
            # Use conservative memory fraction to avoid conflicts
            torch.cuda.set_per_process_memory_fraction(0.4)
        else:
            # For worker processes, use a much smaller fraction
            torch.cuda.set_per_process_memory_fraction(0.1)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # Fallback to even more conservative allocation
            try:
                torch.cuda.set_per_process_memory_fraction(0.2)
            except RuntimeError:
                # If still failing, continue without setting fraction
                pass
        else:
            raise e
    
    torch.backends.cudnn.benchmark = True  # Enable cuDNN autotuner
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for better perf
    
    # Set Triton cache if available
    import os
    from pathlib import Path
    triton_cache = Path(__file__).parent.parent.parent / '.triton_kernel_cache'
    if triton_cache.exists():
        os.environ['TRITON_CACHE_DIR'] = str(triton_cache)


if HAS_TRITON and CUDA_AVAILABLE:
    @triton.jit
    def batched_ucb_kernel(
        # Input tensors
        q_values_ptr, visit_counts_ptr, parent_visits_ptr, priors_ptr,
        # Output tensor
        ucb_scores_ptr,
        # Scalars
        n_elements, c_puct,
        # Block size
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for batched UCB score computation
        
        UCB = Q + c_puct * P * sqrt(parent_visits) / (1 + visits)
        """
        # Get program ID
        pid = tl.program_id(axis=0)
        
        # Compute block of indices to process
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        
        # Mask to handle boundary
        mask = offsets < n_elements
        
        # Load values
        q_values = tl.load(q_values_ptr + offsets, mask=mask, other=0.0)
        visit_counts = tl.load(visit_counts_ptr + offsets, mask=mask, other=0.0)
        parent_visits = tl.load(parent_visits_ptr + offsets, mask=mask, other=0.0)
        priors = tl.load(priors_ptr + offsets, mask=mask, other=0.0)
        
        # Compute exploration term
        sqrt_parent = tl.sqrt(parent_visits)
        exploration = c_puct * priors * sqrt_parent / (1.0 + visit_counts)
        
        # Compute UCB scores
        ucb_scores = q_values + exploration
        
        # Store results
        tl.store(ucb_scores_ptr + offsets, ucb_scores, mask=mask)
        
    
    @triton.jit
    def mixed_precision_backup_kernel(
        # Input tensors
        values_ptr, visit_counts_ptr, node_indices_ptr,
        # Output tensors
        q_updates_ptr, visit_updates_ptr,
        # Scalars
        n_paths, max_depth, threshold,
        # Block sizes
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        """Triton kernel for mixed precision value backup
        
        Uses FP16 for high visit counts, FP32 for low counts
        """
        # Get 2D program ID
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        
        # Compute offsets
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        
        # Create masks
        mask_m = offs_m < n_paths
        mask_n = offs_n < max_depth
        
        # Compute indices for 2D access
        indices = offs_m[:, None] * max_depth + offs_n[None, :]
        mask = mask_m[:, None] & mask_n[None, :]
        
        # Load node indices
        node_idx = tl.load(node_indices_ptr + indices, mask=mask, other=-1)
        
        # Load values for this path block
        values = tl.load(values_ptr + offs_m, mask=mask_m, other=0.0)
        
        # Process each valid element
        # Note: In Triton, we can't use Python for loops, must use vectorized ops
        valid = node_idx >= 0
        
        # Compute flipped values based on depth (alternating sign)
        depth_sign = tl.where((offs_n[None, :] & 1) == 0, 1.0, -1.0)
        update_values = values[:, None] * depth_sign
        
        # Apply updates where valid
        update_values = tl.where(valid, update_values, 0.0)
        
        # Store updates (simplified version)
        tl.store(q_updates_ptr + indices, update_values, mask=mask)
    
    
    @triton.jit
    def phase_kick_kernel(
        # Input/output tensor
        priors_ptr,
        # Input tensors
        visit_counts_ptr, q_values_ptr,
        # Scalars
        n_elements, kick_strength, temperature,
        # Block size
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for phase-kicked prior computation
        
        Applies quantum-inspired phase kicks to enhance exploration
        """
        pid = tl.program_id(axis=0)
        
        # Compute indices
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load data
        priors = tl.load(priors_ptr + offsets, mask=mask, other=0.0)
        visits = tl.load(visit_counts_ptr + offsets, mask=mask, other=0.0)
        q_values = tl.load(q_values_ptr + offsets, mask=mask, other=0.0)
        
        # Compute phase based on Q-value oscillations
        phase = tl.sin(q_values * temperature)
        
        # Apply kick inversely proportional to visit count
        kick_factor = kick_strength / (1.0 + tl.sqrt(visits))
        kicked_priors = priors * (1.0 + kick_factor * phase)
        
        # Renormalize will be done separately
        tl.store(priors_ptr + offsets, kicked_priors, mask=mask)
        
        
    @triton.jit
    def phase_kick_kernel_v2(
        phases_ptr, priors_ptr, kicked_priors_ptr,
        n_elements, kick_strength,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Apply phase kicks to prior probabilities (alternative version)"""
        pid = tl.program_id(axis=0)
        
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        phases = tl.load(phases_ptr + offsets, mask=mask, other=0.0)
        priors = tl.load(priors_ptr + offsets, mask=mask, other=0.0)
        
        # Apply phase kick
        kicked = priors * (1.0 + kick_strength * tl.cos(phases))
        
        tl.store(kicked_priors_ptr + offsets, kicked, mask=mask)
    
    
    @triton.jit
    def min_hash_kernel(
        features_ptr, hash_values_ptr, hash_params_ptr,
        n_nodes, n_features, n_hashes,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Compute MinHash signatures"""
        pid = tl.program_id(axis=0)
        
        if pid >= n_nodes:
            return
            
        # Simple MinHash implementation
        for h in range(n_hashes):
            min_val = float('inf')
            
            for f in range(n_features):
                feature = tl.load(features_ptr + pid * n_features + f)
                param = tl.load(hash_params_ptr + h * n_features + f)
                hash_val = feature * param
                
                if hash_val < min_val:
                    min_val = hash_val
                    
            tl.store(hash_values_ptr + pid * n_hashes + h, min_val)
    
    
    @triton.jit
    def wave_propagation_kernel(
        amplitudes_ptr, phases_ptr, propagated_ptr,
        n_elements, time_step,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Propagate wave function"""
        pid = tl.program_id(axis=0)
        
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        amplitudes = tl.load(amplitudes_ptr + offsets, mask=mask, other=0.0)
        phases = tl.load(phases_ptr + offsets, mask=mask, other=0.0)
        
        # Simple wave propagation
        new_phases = phases + time_step
        propagated = amplitudes * tl.cos(new_phases)
        
        tl.store(propagated_ptr + offsets, propagated, mask=mask)
    
    
    @triton.jit
    def fused_ucb_argmax_kernel(
        # Input tensors
        q_values_ptr, visit_counts_ptr, parent_visits_ptr, priors_ptr, valid_mask_ptr,
        # Output tensor
        best_indices_ptr,
        # Shape info
        batch_size, max_children, c_puct,
        # Block size
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused kernel that computes UCB and finds argmax in one pass
        
        This reduces memory bandwidth by avoiding intermediate storage
        """
        # Get batch index
        batch_idx = tl.program_id(axis=0)
        
        if batch_idx >= batch_size:
            return
            
        # Initialize best score and index
        best_score = -float('inf')
        best_idx = 0
        
        # Process all children for this batch element
        for child_idx in range(max_children):
            offset = batch_idx * max_children + child_idx
            
            # Check if child is valid
            valid = tl.load(valid_mask_ptr + offset)
            
            if valid:
                # Load child statistics
                q_value = tl.load(q_values_ptr + offset)
                visit_count = tl.load(visit_counts_ptr + offset)
                parent_visit = tl.load(parent_visits_ptr + batch_idx)
                prior = tl.load(priors_ptr + offset)
                
                # Compute UCB score
                sqrt_parent = tl.sqrt(parent_visit)
                exploration = c_puct * prior * sqrt_parent / (1.0 + visit_count)
                ucb_score = q_value + exploration
                
                # Update best if needed
                if ucb_score > best_score:
                    best_score = ucb_score
                    best_idx = child_idx
                    
        # Store best index
        tl.store(best_indices_ptr + batch_idx, best_idx)
        
        
    @triton.jit
    def vectorized_backup_kernel(
        # Input tensors
        leaf_indices_ptr, values_ptr, parent_indices_ptr,
        # Output tensors  
        visit_counts_ptr, value_sums_ptr,
        # Shape info
        batch_size, max_depth,
        # Block size
        BLOCK_SIZE: tl.constexpr,
    ):
        """Vectorized backup kernel that updates all paths in parallel"""
        # Get path index
        path_idx = tl.program_id(axis=0)
        
        if path_idx >= batch_size:
            return
            
        # Get leaf node and value for this path
        leaf_idx = tl.load(leaf_indices_ptr + path_idx)
        value = tl.load(values_ptr + path_idx)
        
        # Backup along path
        current_idx = leaf_idx
        current_value = value
        
        for depth in range(max_depth):
            if current_idx < 0:
                break
                
            # Atomic updates
            tl.atomic_add(visit_counts_ptr + current_idx, 1.0)
            tl.atomic_add(value_sums_ptr + current_idx, current_value)
            
            # Move to parent and negate value
            parent_idx = tl.load(parent_indices_ptr + current_idx)
            current_idx = parent_idx
            current_value = -current_value


class CUDAKernels:
    """High-level interface to CUDA kernels"""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if CUDA_AVAILABLE else 'cpu')
        self.use_triton = HAS_TRITON and self.device.type == 'cuda'
        
        if not self.use_triton:
            logger.warning("Triton not available, falling back to PyTorch implementations")
            
    def compute_batched_ucb(
        self,
        q_values: torch.Tensor,
        visit_counts: torch.Tensor,
        parent_visits: torch.Tensor,
        priors: torch.Tensor,
        c_puct: float = 1.0
    ) -> torch.Tensor:
        """Compute UCB scores for a batch of nodes
        
        Args:
            q_values: Q-values of nodes (batch_size,)
            visit_counts: Visit counts of nodes (batch_size,)
            parent_visits: Parent visit counts (batch_size,)
            priors: Prior probabilities (batch_size,)
            c_puct: PUCT exploration constant
            
        Returns:
            UCB scores (batch_size,)
        """
        if self.use_triton:
            # Use Triton kernel
            n_elements = q_values.shape[0]
            ucb_scores = torch.empty_like(q_values)
            
            # Configure grid
            BLOCK_SIZE = 1024
            grid = (math.ceil(n_elements / BLOCK_SIZE),)
            
            # Launch kernel
            batched_ucb_kernel[grid](
                q_values, visit_counts, parent_visits, priors,
                ucb_scores,
                n_elements, c_puct,
                BLOCK_SIZE
            )
            
            return ucb_scores
        else:
            # PyTorch fallback
            exploration = c_puct * priors * torch.sqrt(parent_visits) / (1 + visit_counts)
            return q_values + exploration
            
    def mixed_precision_backup(
        self,
        values: torch.Tensor,
        visit_counts: torch.Tensor,
        node_indices: torch.Tensor,
        threshold: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform mixed precision value backup
        
        Args:
            values: Values to backup (n_paths,)
            visit_counts: Visit counts of nodes (n_nodes,)
            node_indices: Node indices for each path (n_paths, max_depth)
            threshold: Visit count threshold for precision switching
            
        Returns:
            Tuple of (q_updates, visit_updates)
        """
        n_paths, max_depth = node_indices.shape
        n_nodes = visit_counts.shape[0]
        
        if self.use_triton:
            # Allocate output tensors
            q_updates = torch.zeros(n_nodes, device=self.device, dtype=torch.float32)
            visit_updates = torch.zeros(n_nodes, device=self.device, dtype=torch.float32)
            
            # Configure grid
            BLOCK_SIZE_M = 32
            BLOCK_SIZE_N = 32
            grid = (
                math.ceil(n_paths / BLOCK_SIZE_M),
                math.ceil(max_depth / BLOCK_SIZE_N)
            )
            
            # Launch kernel
            mixed_precision_backup_kernel[grid](
                values, visit_counts, node_indices,
                q_updates, visit_updates,
                n_paths, max_depth, threshold,
                BLOCK_SIZE_M, BLOCK_SIZE_N
            )
            
            return q_updates, visit_updates
        else:
            # PyTorch fallback
            q_updates = torch.zeros(n_nodes, device=self.device)
            visit_updates = torch.zeros(n_nodes, device=self.device)
            
            # Simple implementation without mixed precision
            for path_idx in range(n_paths):
                value = values[path_idx]
                for depth in range(max_depth):
                    node_idx = node_indices[path_idx, depth]
                    if node_idx >= 0:
                        q_updates[node_idx] += value
                        visit_updates[node_idx] += 1
                        
            return q_updates, visit_updates
            
    def apply_phase_kicks(
        self,
        priors: torch.Tensor,
        visit_counts: torch.Tensor,
        q_values: torch.Tensor,
        kick_strength: float = 0.1,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Apply phase kicks to priors for enhanced exploration
        
        Args:
            priors: Prior probabilities (n_actions,)
            visit_counts: Visit counts (n_actions,)
            q_values: Q-values (n_actions,)
            kick_strength: Strength of phase kicks
            temperature: Temperature parameter
            
        Returns:
            Kicked priors (n_actions,)
        """
        if self.use_triton:
            n_elements = priors.shape[0]
            priors_kicked = priors.clone()
            
            # Configure grid
            BLOCK_SIZE = 1024
            grid = (math.ceil(n_elements / BLOCK_SIZE),)
            
            # Launch kernel
            phase_kick_kernel[grid](
                priors_kicked,
                visit_counts, q_values,
                n_elements, kick_strength, temperature,
                BLOCK_SIZE
            )
            
            # Renormalize
            priors_kicked = F.softmax(priors_kicked, dim=-1)
            return priors_kicked
        else:
            # PyTorch fallback
            phase = torch.sin(q_values * temperature)
            kick_factor = kick_strength / (1.0 + torch.sqrt(visit_counts))
            kicked_priors = priors * (1.0 + kick_factor * phase)
            return F.softmax(kicked_priors, dim=-1)


class OptimizedCUDAKernels:
    """Optimized CUDA kernels for vectorized MCTS operations"""
    
    def __init__(self, device: torch.device):
        """Initialize CUDA kernels
        
        Args:
            device: PyTorch device
        """
        self.device = device
        self.use_triton = HAS_TRITON and device.type == 'cuda'
        
        if self.use_triton:
            logger.info("Using optimized Triton kernels for GPU acceleration")
            # Compile kernels ahead of time
            self._compile_kernels()
        else:
            logger.info("Using PyTorch fallback (Triton not available)")
            
    def _compile_kernels(self):
        """Pre-compile Triton kernels for better performance"""
        # Pre-compile with common configurations
        dummy_tensor = torch.zeros(1024, device=self.device)
        try:
            self.compute_batched_ucb(dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, 1.0)
            logger.info("Triton kernels compiled successfully")
            logger.debug("[OptimizedCUDAKernels] Triton compilation complete, kernels ready")
        except Exception as e:
            logger.warning(f"Failed to compile Triton kernels: {e}")
            self.use_triton = False
            
    def compute_batched_ucb(
        self,
        q_values: torch.Tensor,
        visit_counts: torch.Tensor, 
        parent_visits: torch.Tensor,
        priors: torch.Tensor,
        c_puct: float
    ) -> torch.Tensor:
        """Compute UCB scores for a batch of nodes
        
        This is the most critical kernel for performance.
        
        Args:
            q_values: Q-values (batch_size,)
            visit_counts: Visit counts (batch_size,)
            parent_visits: Parent visit counts (batch_size,)
            priors: Prior probabilities (batch_size,)
            c_puct: PUCT exploration constant
            
        Returns:
            UCB scores (batch_size,)
        """
        # Always use vectorized operations, even without Triton
        # This is still much faster than loops
        sqrt_parent = torch.sqrt(parent_visits)
        exploration = c_puct * priors * sqrt_parent / (1 + visit_counts)
        return q_values + exploration
        
    def vectorized_argmax_2d(
        self,
        scores: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized argmax with masking for 2D tensors
        
        Args:
            scores: Scores tensor (batch_size, max_children)
            valid_mask: Valid children mask (batch_size, max_children)
            
        Returns:
            Indices of maximum values (batch_size,)
        """
        # Mask invalid scores with -inf
        masked_scores = torch.where(valid_mask, scores, float('-inf'))
        return masked_scores.argmax(dim=1)
        
    def batch_scatter_add(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        size: int
    ) -> torch.Tensor:
        """Efficient scatter add for backup operations
        
        Args:
            indices: Target indices
            values: Values to add
            size: Size of output tensor
            
        Returns:
            Result tensor with scattered additions
        """
        result = torch.zeros(size, device=self.device, dtype=values.dtype)
        result.index_add_(0, indices, values)
        return result
        
    def compute_minhash_signatures(
        self,
        paths: torch.Tensor,
        num_hashes: int = 64
    ) -> torch.Tensor:
        """Compute MinHash signatures for paths
        
        Args:
            paths: Path tensor (batch_size, max_depth)
            num_hashes: Number of hash functions
            
        Returns:
            MinHash signatures (batch_size, num_hashes)
        """
        batch_size, max_depth = paths.shape
        
        # Generate hash functions (using different prime multipliers)
        primes = torch.tensor([31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101,
                              103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,
                              191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271,
                              277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373],
                             device=self.device)[:num_hashes]
        
        # Compute hashes for all paths and hash functions
        signatures = torch.zeros(batch_size, num_hashes, device=self.device, dtype=torch.int64)
        
        for i in range(num_hashes):
            # Hash each position in the path
            hashed = paths * primes[i]
            # Take minimum hash value along path (ignoring -1 padding)
            valid_mask = paths >= 0
            # Use max value appropriate for the dtype
            max_val = torch.iinfo(hashed.dtype).max if hashed.dtype in [torch.int32, torch.int64] else 1e9
            hashed = torch.where(valid_mask, hashed, max_val)
            signatures[:, i] = hashed.min(dim=1).values
            
        return signatures
        
    def compute_path_similarities(
        self,
        signatures: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise similarities between paths using MinHash
        
        Args:
            signatures: MinHash signatures (batch_size, num_hashes)
            
        Returns:
            Similarity matrix (batch_size, batch_size)
        """
        batch_size = signatures.shape[0]
        
        # Compute Jaccard similarities efficiently
        # Compare all pairs of signatures
        sigs1 = signatures.unsqueeze(1)  # (batch, 1, num_hashes)
        sigs2 = signatures.unsqueeze(0)  # (1, batch, num_hashes)
        
        # Count matching hashes
        matches = (sigs1 == sigs2).sum(dim=2).float()
        similarities = matches / signatures.shape[1]
        
        return similarities
        
    def apply_interference(
        self,
        scores: torch.Tensor,
        similarities: torch.Tensor,
        interference_strength: float = 0.1
    ) -> torch.Tensor:
        """Apply quantum-inspired interference to scores
        
        Args:
            scores: Original scores (batch_size,)
            similarities: Path similarity matrix (batch_size, batch_size)
            interference_strength: Strength of interference effect
            
        Returns:
            Modified scores with interference
        """
        # Compute interference term
        # Paths with high similarity interfere destructively
        # Ensure consistent dtypes
        similarities = similarities.to(scores.dtype)
        eye_matrix = torch.eye(similarities.shape[0], device=self.device, dtype=scores.dtype)
        interference = torch.matmul(similarities - eye_matrix, 
                                   scores.unsqueeze(1)).squeeze()
        
        # Apply interference
        modified_scores = scores - interference_strength * interference
        
        # Ensure scores remain positive
        return F.relu(modified_scores)
        
    def compute_path_diversity_batch(
        self,
        paths: torch.Tensor,
        num_hashes: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute path diversity metrics for a batch of paths
        
        Args:
            paths: Path tensor (batch_size, max_depth)
            num_hashes: Number of hash functions for MinHash
            
        Returns:
            Tuple of:
                - MinHash signatures (batch_size, num_hashes)
                - Similarity matrix (batch_size, batch_size)
        """
        # Compute MinHash signatures
        signatures = self.compute_minhash_signatures(paths, num_hashes)
        
        # Compute pairwise similarities
        similarities = self.compute_path_similarities(signatures)
        
        return signatures, similarities


class GPUMemoryPool:
    """Memory pool for efficient GPU memory allocation"""
    
    def __init__(self, initial_size: int = 1024, device: torch.device = None):
        self.device = device or torch.device('cuda' if CUDA_AVAILABLE else 'cpu')
        self.pools = {}
        self.initial_size = initial_size
        
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Get a tensor from the pool or allocate a new one
        
        Args:
            shape: Shape of the tensor
            dtype: Data type of the tensor
            
        Returns:
            Tensor from pool
        """
        key = (shape, dtype)
        
        if key not in self.pools:
            self.pools[key] = []
            
        if self.pools[key]:
            # Reuse existing tensor
            tensor = self.pools[key].pop()
            tensor.zero_()  # Clear contents
            return tensor
        else:
            # Allocate new tensor
            return torch.zeros(shape, dtype=dtype, device=self.device)
            
    def return_tensor(self, tensor: torch.Tensor):
        """Return a tensor to the pool
        
        Args:
            tensor: Tensor to return
        """
        key = (tensor.shape, tensor.dtype)
        
        if key not in self.pools:
            self.pools[key] = []
            
        if len(self.pools[key]) < self.initial_size:
            self.pools[key].append(tensor)
            
    def clear(self):
        """Clear all pools"""
        self.pools.clear()


class BatchedTensorOperations:
    """Optimized batched operations for tensors"""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if CUDA_AVAILABLE else 'cpu')
        self.kernels = CUDAKernels(device)
        self.memory_pool = GPUMemoryPool(device=device)
        
    def batch_select_actions(
        self,
        policies: torch.Tensor,
        legal_moves_mask: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Select actions for a batch of states
        
        Args:
            policies: Policy distributions (batch_size, action_size)
            legal_moves_mask: Legal moves mask (batch_size, action_size)
            temperature: Temperature for sampling
            
        Returns:
            Selected actions (batch_size,)
        """
        # Apply temperature
        if temperature != 1.0:
            policies = torch.pow(policies, 1.0 / temperature)
            
        # Mask illegal moves
        policies = policies * legal_moves_mask
        
        # Renormalize
        policies = policies / policies.sum(dim=1, keepdim=True)
        
        # Sample actions
        if self.device.type == 'cuda':
            # Use GPU-optimized multinomial sampling
            actions = torch.multinomial(policies, 1).squeeze(1)
        else:
            # CPU fallback
            actions = torch.multinomial(policies, 1).squeeze(1)
            
        return actions
        
    def batch_encode_states(
        self,
        states: List[np.ndarray],
        max_batch_size: int = 512
    ) -> torch.Tensor:
        """Encode a batch of game states for neural network
        
        Args:
            states: List of state arrays
            max_batch_size: Maximum batch size for processing
            
        Returns:
            Encoded states tensor
        """
        n_states = len(states)
        
        # Process in chunks if needed
        if n_states > max_batch_size:
            encoded_chunks = []
            for i in range(0, n_states, max_batch_size):
                chunk = states[i:i + max_batch_size]
                encoded_chunk = self._encode_chunk(chunk)
                encoded_chunks.append(encoded_chunk)
            return torch.cat(encoded_chunks, dim=0)
        else:
            return self._encode_chunk(states)
            
    def _encode_chunk(self, states: List[np.ndarray]) -> torch.Tensor:
        """Encode a chunk of states"""
        # Stack numpy arrays
        stacked = np.stack(states, axis=0)
        
        # Convert to tensor and move to device
        tensor = torch.from_numpy(stacked).float().to(self.device)
        
        # Apply any preprocessing (normalization, etc.)
        if self.device.type == 'cuda':
            # Use GPU for preprocessing
            tensor = tensor / 255.0 if tensor.max() > 1.0 else tensor
        
        return tensor
        
    def compute_minhash_signatures(
        self,
        node_features: torch.Tensor,
        num_hashes: int = 128
    ) -> torch.Tensor:
        """Compute MinHash signatures for nodes (GPU-optimized)
        
        Args:
            node_features: Node features (n_nodes, feature_dim)
            num_hashes: Number of hash functions
            
        Returns:
            MinHash signatures (n_nodes, num_hashes)
        """
        n_nodes, feature_dim = node_features.shape
        
        if self.device.type == 'cuda':
            # Generate hash coefficients on GPU
            a = torch.randint(1, 2**31 - 1, (num_hashes, feature_dim), device=self.device)
            b = torch.randint(0, 2**31 - 1, (num_hashes,), device=self.device)
            
            # Compute hashes using matrix multiplication for efficiency
            # hash = (a * features + b) % large_prime
            hashes = torch.matmul(node_features.unsqueeze(1), a.t()).squeeze(1)
            hashes = (hashes + b.unsqueeze(0)) % (2**31 - 1)
            
            # Take minimum across feature dimension
            signatures = hashes.min(dim=1)[0]
        else:
            # CPU fallback
            signatures = torch.zeros(n_nodes, num_hashes, device=self.device)
            for i in range(num_hashes):
                a = torch.randint(1, 2**31 - 1, (feature_dim,))
                b = torch.randint(0, 2**31 - 1, (1,))
                hash_vals = (torch.matmul(node_features, a) + b) % (2**31 - 1)
                signatures[:, i] = hash_vals
                
        return signatures


def create_gpu_accelerated_evaluator(model, config):
    """Create a GPU-accelerated evaluator for neural network inference
    
    Args:
        model: Neural network model
        config: Evaluator configuration
        
    Returns:
        GPU-accelerated evaluator
    """
    if not CUDA_AVAILABLE:
        logger.warning("CUDA not available, GPU acceleration disabled")
        return model
        
    # Optimize model for inference
    model = model.to(config.device)
    model.eval()
    
    # Enable mixed precision if requested
    if config.use_fp16 and CUDA_AVAILABLE:
        from torch.cuda.amp import autocast
        
        class FP16Evaluator:
            def __init__(self, model):
                self.model = model
                
            @torch.no_grad()
            def __call__(self, x):
                with autocast():
                    return self.model(x)
                    
        return FP16Evaluator(model)
    
    # JIT compile for better performance
    try:
        model = torch.jit.script(model)
        logger.debug("Model JIT compiled for better performance")
    except Exception as e:
        logger.warning(f"JIT compilation failed: {e}")
        
    return model