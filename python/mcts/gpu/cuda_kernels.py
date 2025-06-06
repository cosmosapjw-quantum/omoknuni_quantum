"""CUDA kernels for GPU acceleration of MCTS operations

This module provides custom CUDA kernels for performance-critical operations
in the vectorized MCTS implementation.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
import math
import logging
import numpy as np

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
    # Set CUDA memory allocator settings for better performance
    torch.cuda.set_per_process_memory_fraction(0.9)  # Use up to 90% of GPU memory
    torch.backends.cudnn.benchmark = True  # Enable cuDNN autotuner
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for better perf


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
        
        # Compute indices
        indices = offs_m[:, None] * max_depth + offs_n[None, :]
        mask = mask_m[:, None] & mask_n[None, :]
        
        # Load data
        node_idx = tl.load(node_indices_ptr + indices, mask=mask, other=-1)
        values = tl.load(values_ptr + offs_m, mask=mask_m, other=0.0)
        
        # Process updates based on visit count
        for i in range(BLOCK_SIZE_M):
            if offs_m[i] < n_paths:
                path_value = values[i]
                
                for j in range(BLOCK_SIZE_N):
                    if offs_n[j] < max_depth:
                        idx = node_idx[i, j]
                        if idx >= 0:
                            visits = tl.load(visit_counts_ptr + idx)
                            
                            # Choose precision based on visit count
                            if visits > threshold:
                                # Use FP16 for high visits
                                update = path_value.to(tl.float16)
                            else:
                                # Use FP32 for low visits
                                update = path_value
                                
                            # Atomic add to handle concurrent updates
                            tl.atomic_add(q_updates_ptr + idx, update)
                            tl.atomic_add(visit_updates_ptr + idx, 1.0)
    
    
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
        logger.info("Model JIT compiled for better performance")
    except Exception as e:
        logger.warning(f"JIT compilation failed: {e}")
        
    return model