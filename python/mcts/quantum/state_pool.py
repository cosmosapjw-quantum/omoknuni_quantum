"""
Quantum State Pool and Memory Optimization
=========================================

This module implements efficient quantum state recycling and memory management
for the quantum MCTS framework. It provides:
- State object pooling to reduce allocations
- Compressed state caching
- Memory-mapped state storage for large trees
- GPU memory optimization

Key Features:
- O(1) state allocation/deallocation
- Automatic compression of sparse states
- Thread-safe state management
- GPU/CPU memory transfer optimization
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from collections import deque
import threading
import gc
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Configuration for quantum state pool"""
    max_states: int = 50000  # Increased from 1000 for 64GB RAM
    initial_pool_size: int = 5000  # Increased from 100
    compression_threshold: float = 0.95  # Only compress very sparse states
    rank_threshold: float = 0.95  # Keep more ranks for accuracy
    enable_gpu_optimization: bool = True
    gc_interval: int = 1000  # Less frequent GC with more RAM
    enable_memory_mapping: bool = False  # Not needed with 64GB RAM
    mmap_threshold_mb: float = 1000.0  # Higher threshold


class CompressedState:
    """Compressed representation of quantum states"""
    
    def __init__(self, state_type: str, data: Dict[str, torch.Tensor]):
        self.state_type = state_type
        self.data = data
        self.original_shape = data.get('shape', None)
        
    def decompress(self, device: torch.device) -> torch.Tensor:
        """Decompress state back to full tensor"""
        if self.state_type == 'sparse':
            # Reconstruct from sparse representation
            indices = self.data['indices'].to(device)
            values = self.data['values'].to(device)
            shape = self.data['shape']
            
            state = torch.zeros(shape, device=device, dtype=values.dtype)
            if len(indices) > 0:
                state[indices[:, 0], indices[:, 1]] = values
            return state
            
        elif self.state_type == 'low_rank':
            # Reconstruct from low-rank factors
            U = self.data['U'].to(device)
            S = self.data['S'].to(device)
            V = self.data['V'].to(device)
            
            # Ensure correct dtype from stored info
            original_dtype = self.data.get('dtype', U.dtype)
            if original_dtype.is_complex:
                # Ensure all components have correct complex dtype
                U = U.to(original_dtype)
                V = V.to(original_dtype)
                S_diag = torch.diag(S.to(torch.float32))  # S is always real
                S_diag = S_diag.to(original_dtype)
            else:
                S_diag = torch.diag(S)
            
            # Reconstruct: U @ diag(S) @ V^T
            state = torch.matmul(U, torch.matmul(S_diag, V.T))
            return state
            
        elif self.state_type == 'diagonal':
            # Reconstruct diagonal matrix
            diag_values = self.data['diagonal'].to(device)
            state = torch.diag(diag_values)
            return state
            
        else:
            raise ValueError(f"Unknown compression type: {self.state_type}")
    
    def memory_size(self) -> int:
        """Estimate memory usage in bytes"""
        total = 0
        for tensor in self.data.values():
            if isinstance(tensor, torch.Tensor):
                total += tensor.element_size() * tensor.nelement()
        return total


class StateCompressor:
    """Handles quantum state compression"""
    
    def __init__(self, config: PoolConfig):
        self.config = config
        
    def compress(self, state: torch.Tensor) -> Optional[CompressedState]:
        """Compress quantum state if beneficial"""
        # Check if state is diagonal first (special case of sparse)
        if self._is_diagonal(state):
            return self._diagonal_compress(state)
        
        # Check if state is sparse
        sparsity = self._compute_sparsity(state)
        
        if sparsity > self.config.compression_threshold:
            return self._sparse_compress(state)
            
        # Check if state is low-rank
        rank_ratio = self._estimate_rank_ratio(state)
        
        if rank_ratio < self.config.rank_threshold:
            return self._low_rank_compress(state)
            
        # No beneficial compression
        return None
    
    def _compute_sparsity(self, state: torch.Tensor) -> float:
        """Compute sparsity (fraction of zeros)"""
        num_zeros = (state.abs() < 1e-10).sum().item()
        total_elements = state.numel()
        return num_zeros / total_elements if total_elements > 0 else 0.0
    
    def _estimate_rank_ratio(self, state: torch.Tensor) -> float:
        """Estimate effective rank as ratio of singular values"""
        if state.dim() != 2:
            return 1.0
            
        # Use randomized SVD for efficiency
        try:
            _, S, _ = torch.svd_lowrank(state, q=min(10, min(state.shape)))
            
            # Compute effective rank
            S_normalized = S / (S[0] + 1e-10)
            effective_rank = torch.sum(S_normalized > 0.01).item()
            
            return effective_rank / min(state.shape)
        except:
            return 1.0
    
    def _is_diagonal(self, state: torch.Tensor) -> bool:
        """Check if matrix is diagonal"""
        if state.dim() != 2 or state.shape[0] != state.shape[1]:
            return False
            
        # Check off-diagonal elements
        mask = torch.ones_like(state, dtype=torch.bool)
        mask.fill_diagonal_(False)
        
        return torch.all(torch.abs(state[mask]) < 1e-10)
    
    def _sparse_compress(self, state: torch.Tensor) -> CompressedState:
        """Compress using sparse representation"""
        # Find non-zero elements
        mask = torch.abs(state) > 1e-10
        indices = torch.nonzero(mask)
        values = state[mask]
        
        data = {
            'indices': indices.cpu(),
            'values': values.cpu(),
            'shape': state.shape
        }
        
        return CompressedState('sparse', data)
    
    def _low_rank_compress(self, state: torch.Tensor) -> CompressedState:
        """Compress using low-rank approximation"""
        # Use truncated SVD
        rank = max(1, int(min(state.shape) * self.config.rank_threshold))
        
        try:
            if state.is_complex():
                # For complex matrices, use full SVD then truncate
                # svd_lowrank doesn't support complex dtypes
                U_full, S_full, Vh_full = torch.linalg.svd(state, full_matrices=False)
                
                # Keep only top 'rank' components
                U = U_full[:, :rank]
                S = S_full[:rank]
                V = Vh_full[:rank, :].conj().T  # V not V^H for consistency
                
            else:
                # For real matrices, use efficient low-rank SVD
                U, S, V = torch.svd_lowrank(state, q=rank)
            
            data = {
                'U': U.cpu(),
                'S': S.cpu(),
                'V': V.cpu(),
                'shape': state.shape,
                'dtype': state.dtype  # Store original dtype
            }
            
            return CompressedState('low_rank', data)
        except Exception as e:
            # Fallback to sparse if SVD fails
            logger.debug(f"Low-rank compression failed: {e}")
            return self._sparse_compress(state)
    
    def _diagonal_compress(self, state: torch.Tensor) -> CompressedState:
        """Compress diagonal matrix"""
        diagonal = torch.diag(state)
        
        data = {
            'diagonal': diagonal.cpu(),
            'shape': state.shape
        }
        
        return CompressedState('diagonal', data)


class QuantumStatePool:
    """
    Efficient pool for quantum state management
    
    Provides recycling of quantum states to reduce memory allocations
    and improve performance in quantum MCTS.
    """
    
    def __init__(self, config: PoolConfig, device: torch.device):
        self.config = config
        self.device = device
        self.compressor = StateCompressor(config)
        
        # State pools by dimension
        self.pools: Dict[Tuple[int, ...], deque] = {}
        
        # Compressed state cache
        self.compressed_cache: Dict[int, CompressedState] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'allocations': 0,
            'reuses': 0,
            'compressions': 0,
            'decompressions': 0,
            'gc_runs': 0,
            'current_pool_size': 0,
            'compressed_states': 0
        }
        
        # Initialize pools
        self._initialize_pools()
        
        logger.debug(f"QuantumStatePool initialized with max_states={config.max_states}")
    
    def _initialize_pools(self):
        """Pre-allocate common state sizes - optimized for 64GB RAM"""
        common_sizes = [
            (4, 4),      # Small systems
            (8, 8),      # Medium systems
            (16, 16),    # Large systems
            (32, 32),    # Very large systems
            (64, 64),    # Extra large systems
            (128, 128),  # Huge systems (feasible with 64GB)
            (256, 256),  # Maximum practical size
        ]
        
        # Allocate more states per size with 64GB RAM
        allocations_per_size = {
            (4, 4): 1000,
            (8, 8): 500,
            (16, 16): 200,
            (32, 32): 100,
            (64, 64): 50,
            (128, 128): 20,
            (256, 256): 10
        }
        
        for shape in common_sizes:
            if shape not in self.pools:
                self.pools[shape] = deque()
                
            # Pre-allocate states based on size
            num_to_allocate = min(
                allocations_per_size.get(shape, 10),
                self.config.initial_pool_size // len(common_sizes)
            )
            
            for _ in range(num_to_allocate):
                state = torch.zeros(shape, device=self.device, dtype=torch.complex64)
                self.pools[shape].append(state)
                
        self.stats['current_pool_size'] = sum(len(pool) for pool in self.pools.values())
    
    def get_state(self, shape: Union[int, Tuple[int, ...]], dtype: torch.dtype = torch.complex64) -> torch.Tensor:
        """
        Get a quantum state from pool or allocate new
        
        Args:
            shape: Shape of state (int for square matrix, tuple for general)
            dtype: Data type of state
            
        Returns:
            Zeroed quantum state tensor
        """
        # Normalize shape
        if isinstance(shape, int):
            shape = (shape, shape)
        shape = tuple(shape)
        
        with self.lock:
            # Check if we have a pooled state
            if shape in self.pools and len(self.pools[shape]) > 0:
                state = self.pools[shape].popleft()
                self.stats['reuses'] += 1
                
                # Ensure correct dtype and zero out
                if state.dtype != dtype:
                    state = state.to(dtype)
                state.zero_()
                
                return state
            
            # Allocate new state
            self.stats['allocations'] += 1
            state = torch.zeros(shape, device=self.device, dtype=dtype)
            
            # Run GC if needed
            if self.stats['allocations'] % self.config.gc_interval == 0:
                self._garbage_collect()
                
            return state
    
    def return_state(self, state: torch.Tensor, compress: bool = True):
        """
        Return state to pool for reuse
        
        Args:
            state: State to return
            compress: Whether to try compressing the state
        """
        if state is None:
            return
            
        shape = tuple(state.shape)
        
        with self.lock:
            # Try to compress if enabled
            if compress and self.config.compression_threshold < 1.0:
                compressed = self.compressor.compress(state)
                if compressed is not None:
                    # Store compressed version
                    state_id = id(state)
                    self.compressed_cache[state_id] = compressed
                    self.stats['compressions'] += 1
                    self.stats['compressed_states'] = len(self.compressed_cache)
                    return
            
            # Return to pool if not full
            if shape not in self.pools:
                self.pools[shape] = deque()
                
            if len(self.pools[shape]) < self.config.max_states // 10:  # Limit per shape
                self.pools[shape].append(state)
                self.stats['current_pool_size'] = sum(len(pool) for pool in self.pools.values())
            else:
                # Pool full, let it be garbage collected
                pass
    
    def get_compressed_state(self, state_id: int) -> Optional[torch.Tensor]:
        """Retrieve and decompress a compressed state"""
        with self.lock:
            if state_id in self.compressed_cache:
                compressed = self.compressed_cache[state_id]
                self.stats['decompressions'] += 1
                return compressed.decompress(self.device)
        return None
    
    def compress_state(self, state: torch.Tensor) -> Optional[CompressedState]:
        """Explicitly compress a state"""
        return self.compressor.compress(state)
    
    def decompress_state(self, compressed: CompressedState) -> torch.Tensor:
        """Decompress a compressed state"""
        self.stats['decompressions'] += 1
        return compressed.decompress(self.device)
    
    def _garbage_collect(self):
        """Run garbage collection to free memory"""
        with self.lock:
            self.stats['gc_runs'] += 1
            
            # Remove old compressed states
            if len(self.compressed_cache) > self.config.max_states:
                # Keep only most recent
                to_remove = len(self.compressed_cache) - self.config.max_states // 2
                for _ in range(to_remove):
                    if self.compressed_cache:
                        self.compressed_cache.pop(next(iter(self.compressed_cache)))
                        
            # Trim pools
            for shape, pool in self.pools.items():
                max_pool_size = self.config.max_states // (len(self.pools) + 1)
                while len(pool) > max_pool_size:
                    pool.pop()
                    
            self.stats['current_pool_size'] = sum(len(pool) for pool in self.pools.values())
            self.stats['compressed_states'] = len(self.compressed_cache)
            
            # Force Python GC
            gc.collect()
            
            # CUDA memory cleanup if available
            if self.device.type == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def clear(self):
        """Clear all pooled states"""
        with self.lock:
            self.pools.clear()
            self.compressed_cache.clear()
            self.stats['current_pool_size'] = 0
            self.stats['compressed_states'] = 0
            
            if self.device.type == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        with self.lock:
            pooled_memory = 0
            for shape, pool in self.pools.items():
                if pool:
                    state_size = pool[0].element_size() * pool[0].nelement()
                    pooled_memory += state_size * len(pool)
                    
            compressed_memory = sum(
                comp.memory_size() for comp in self.compressed_cache.values()
            )
            
            return {
                'pooled_memory_mb': pooled_memory / (1024 * 1024),
                'compressed_memory_mb': compressed_memory / (1024 * 1024),
                'total_memory_mb': (pooled_memory + compressed_memory) / (1024 * 1024),
                'num_pooled_states': self.stats['current_pool_size'],
                'num_compressed_states': self.stats['compressed_states']
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics"""
        stats = dict(self.stats)
        memory_usage = self.get_memory_usage()
        stats.update(memory_usage)
        
        # Compute efficiency metrics
        total_gets = stats['allocations'] + stats['reuses']
        if total_gets > 0:
            stats['reuse_rate'] = stats['reuses'] / total_gets
        else:
            stats['reuse_rate'] = 0.0
            
        if stats['compressions'] > 0:
            stats['compression_ratio'] = stats['compressed_states'] / stats['compressions']
        else:
            stats['compression_ratio'] = 0.0
            
        # Calculate memory saved
        original_memory = stats.get('total_memory_mb', 0) * 2  # Estimate uncompressed size
        compressed_memory = stats.get('compressed_memory_mb', 0)
        stats['memory_saved_mb'] = max(0, original_memory - compressed_memory)
            
        return stats


# Factory function
def create_quantum_state_pool(
    device: Union[str, torch.device] = 'cuda',
    max_states: int = 1000,
    compression_threshold: float = 0.8,
    **kwargs
) -> QuantumStatePool:
    """
    Create quantum state pool with specified configuration
    
    Args:
        device: Device for state allocation
        max_states: Maximum states to pool
        compression_threshold: Threshold for compression
        **kwargs: Additional config parameters
        
    Returns:
        Configured QuantumStatePool
    """
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    config_dict = {
        'max_states': max_states,
        'compression_threshold': compression_threshold,
        'enable_gpu_optimization': device.type == 'cuda'
    }
    config_dict.update(kwargs)
    
    config = PoolConfig(**config_dict)
    
    return QuantumStatePool(config, device)


# Alias for compatibility
create_state_pool = create_quantum_state_pool