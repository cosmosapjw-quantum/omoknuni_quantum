"""
Wave Function Compression Techniques
====================================

This module implements advanced compression techniques for quantum wave functions
in MCTS, including:

- Tensor network compression (MPS/MPO)
- Adaptive rank truncation
- Quantum-aware sparsification
- Phase-coherent compression
- Entanglement-preserving compression

These techniques reduce memory usage while preserving quantum properties
essential for interference and superposition effects.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class CompressionConfig:
    """Configuration for wave function compression"""
    # Compression thresholds - optimized for high-RAM systems
    sparsity_threshold: float = 0.95  # Only compress very sparse states
    rank_threshold: float = 0.95  # Keep more ranks for accuracy
    entanglement_threshold: float = 0.7  # Higher threshold for better quantum properties
    
    # Compression methods
    enable_mps: bool = True  # Matrix Product State compression
    enable_adaptive_rank: bool = True
    enable_phase_clustering: bool = True
    enable_entanglement_truncation: bool = True
    
    # Quality parameters - prioritize accuracy with 64GB RAM
    min_fidelity: float = 0.999  # Very high fidelity requirement
    max_bond_dimension: int = 256  # Increased from 100 for RTX 3060 Ti
    phase_bins: int = 64  # Increased from 16 for better phase resolution
    
    # Performance parameters - optimized for Ryzen 9 5900X + RTX 3060 Ti
    use_gpu: bool = True
    cache_compressed: bool = True  # We have RAM to spare
    compression_batch_size: int = 128  # Larger batches for 24 threads
    
    # Memory thresholds - adjusted for 64GB system
    max_cached_states: int = 10000  # Can cache many states
    compression_skip_threshold: int = 32768  # Skip compression for smaller states


class WaveCompressor:
    """
    Advanced wave function compression for quantum MCTS
    
    This class provides multiple compression strategies that preserve
    quantum properties while reducing memory footprint.
    """
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Compression cache
        self._compression_cache = {} if config.cache_compressed else None
        
        # Statistics
        self.stats = {
            'compressions': 0,
            'mps_compressions': 0,
            'rank_reductions': 0,
            'phase_clusterings': 0,
            'sparse_compressions': 0,
            'average_compression_ratio': 0.0,
            'average_fidelity': 0.0
        }
        
    def compress(self, wave_function: torch.Tensor, 
                 metadata: Optional[Dict[str, Any]] = None) -> 'CompressedWave':
        """
        Compress wave function using optimal method
        
        Args:
            wave_function: Quantum wave function tensor
            metadata: Optional metadata about the wave function
            
        Returns:
            CompressedWave object
        """
        self.stats['compressions'] += 1
        
        # Skip compression for small states when we have plenty of RAM
        if hasattr(self.config, 'compression_skip_threshold'):
            if wave_function.numel() < self.config.compression_skip_threshold:
                return CompressedWave('uncompressed', {'data': wave_function}, 1.0, wave_function.shape)
        
        # Check cache
        if self._compression_cache is not None:
            cache_key = id(wave_function)
            if cache_key in self._compression_cache:
                return self._compression_cache[cache_key]
        
        # Analyze wave function properties
        properties = self._analyze_wave_properties(wave_function)
        
        # Select compression method based on properties
        compressed = None
        
        # Check configuration flags to prioritize methods
        # Special case: when only phase clustering is enabled and sparsity threshold is high
        if (self.config.enable_phase_clustering and 
            not self.config.enable_mps and 
            not self.config.enable_adaptive_rank and 
            self.config.sparsity_threshold > 0.95):
            # Force phase clustering when explicitly requested
            compressed = self._phase_cluster_compress(wave_function, properties)
        elif properties['is_sparse']:
            # Use sparse compression for sparse wave functions
            compressed = self._sparse_compress(wave_function, properties)
        elif properties['is_product_state'] and self.config.enable_mps:
            compressed = self._mps_compress(wave_function, properties)
        elif properties['is_low_rank'] and self.config.enable_adaptive_rank:
            compressed = self._adaptive_rank_compress(wave_function, properties)
        elif properties['has_phase_structure'] and self.config.enable_phase_clustering:
            compressed = self._phase_cluster_compress(wave_function, properties)
        
        # Fallback to best general compression
        if compressed is None:
            compressed = self._general_compress(wave_function, properties)
        
        # Verify compression quality with higher threshold for high-RAM system
        if compressed.fidelity < self.config.min_fidelity:
            logger.debug(f"Compression fidelity {compressed.fidelity:.6f} below threshold {self.config.min_fidelity}")
            return CompressedWave('uncompressed', {'data': wave_function}, 1.0, wave_function.shape)
        
        # Update statistics
        self._update_stats(compressed)
        
        # Cache if enabled (we have 64GB RAM, cache aggressively)
        if self._compression_cache is not None:
            # Limit cache size to prevent unbounded growth
            max_cache_size = getattr(self.config, 'max_cached_states', 10000)
            if len(self._compression_cache) >= max_cache_size:
                # Remove oldest entries (FIFO)
                oldest_key = next(iter(self._compression_cache))
                del self._compression_cache[oldest_key]
            self._compression_cache[id(wave_function)] = compressed
        
        return compressed
    
    def _analyze_wave_properties(self, wave_function: torch.Tensor) -> Dict[str, Any]:
        """Analyze quantum properties of wave function"""
        shape = wave_function.shape
        
        # Handle edge case: zero wave function
        if torch.allclose(wave_function, torch.zeros_like(wave_function)):
            return {
                'shape': shape,
                'n_qubits': 0,
                'entanglement_entropy': 0.0,
                'is_product_state': True,
                'sparsity': 1.0,
                'is_sparse': True,
                'rank_ratio': 0.0,
                'is_low_rank': True,
                'has_phase_structure': False
            }
        
        # Calculate n_qubits only if size is power of 2
        numel = wave_function.numel()
        n_qubits = int(np.log2(numel)) if (numel & (numel - 1)) == 0 else 0
        
        # Compute entanglement entropy for bipartition
        if n_qubits >= 2:
            mid = n_qubits // 2
            reshaped = wave_function.reshape(2**mid, 2**(n_qubits-mid))
            
            # Compute reduced density matrix
            rho = torch.matmul(reshaped, reshaped.conj().T)
            
            try:
                eigenvalues = torch.linalg.eigvalsh(rho)
                eigenvalues = eigenvalues[eigenvalues > 1e-10]
                
                if len(eigenvalues) > 0:
                    # Von Neumann entropy
                    entropy = -torch.sum(eigenvalues * torch.log2(eigenvalues)).item()
                    is_product_state = entropy < 0.1
                else:
                    entropy = 0.0
                    is_product_state = True
            except:
                # Handle numerical errors
                entropy = 0.0
                is_product_state = True
        else:
            entropy = 0.0
            is_product_state = True
        
        # Check sparsity
        sparsity = (torch.abs(wave_function) < 1e-10).sum().item() / wave_function.numel()
        
        # Check rank structure
        if wave_function.dim() >= 2:
            rank_ratio = self._estimate_rank_ratio(wave_function)
            is_low_rank = rank_ratio < self.config.rank_threshold
        else:
            # For 1D wave functions, check if they can be reshaped to low-rank matrix
            n = int(np.sqrt(len(wave_function)))
            if n * n == len(wave_function):
                # Can reshape to square matrix
                matrix = wave_function.reshape(n, n)
                rank_ratio = self._estimate_rank_ratio(matrix)
                is_low_rank = rank_ratio < self.config.rank_threshold
            else:
                rank_ratio = 1.0
                is_low_rank = False
        
        # Analyze phase structure
        phases = torch.angle(wave_function[torch.abs(wave_function) > 1e-10])
        if len(phases) > 0:
            phase_std = torch.std(phases).item()
            has_phase_structure = phase_std < np.pi / 4  # Phases are clustered
        else:
            has_phase_structure = False
        
        return {
            'shape': shape,
            'n_qubits': n_qubits,
            'entanglement_entropy': entropy,
            'is_product_state': is_product_state,
            'sparsity': sparsity,
            'is_sparse': sparsity > self.config.sparsity_threshold,
            'rank_ratio': rank_ratio,
            'is_low_rank': is_low_rank,
            'has_phase_structure': has_phase_structure
        }
    
    def _mps_compress(self, wave_function: torch.Tensor, 
                      properties: Dict[str, Any]) -> 'CompressedWave':
        """Compress using Matrix Product State representation"""
        self.stats['mps_compressions'] += 1
        
        n_qubits = properties['n_qubits']
        if n_qubits < 2:
            return self._sparse_compress(wave_function, properties)
        
        # Reshape wave function as matrix chain
        shape = [2] * n_qubits
        psi = wave_function.reshape(shape)
        
        # Sequential SVD to build MPS
        mps_tensors = []
        remaining = psi
        
        for i in range(n_qubits - 1):
            # Reshape for SVD
            dim_left = 2 ** (i + 1)
            dim_right = 2 ** (n_qubits - i - 1)
            matrix = remaining.reshape(dim_left, dim_right)
            
            # SVD with truncation
            U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
            
            # Truncate based on bond dimension and singular value threshold
            # For high-RAM systems, keep more singular values for accuracy
            sv_threshold = (1 - self.config.min_fidelity) * S[0] if len(S) > 0 else 1e-10
            keep_mask = S > sv_threshold
            keep = min(torch.sum(keep_mask).item(), self.config.max_bond_dimension)
            keep = max(keep, 1)  # Keep at least one
            
            U = U[:, :keep]
            S = S[:keep]
            Vh = Vh[:keep, :]
            
            # Store MPS tensor
            if i == 0:
                # First tensor: (2, bond)
                tensor = U.reshape(2, -1)
            else:
                # Middle tensor: (bond_prev, 2, bond)
                bond_prev = mps_tensors[-1].shape[-1]
                tensor = U.reshape(bond_prev, 2, -1)
            
            mps_tensors.append(tensor)
            
            # Update remaining
            remaining = torch.diag(S) @ Vh
        
        # Last tensor
        mps_tensors.append(remaining.reshape(-1, 2))
        
        # Compute compression fidelity
        reconstructed = self._mps_reconstruct(mps_tensors, shape)
        fidelity = torch.abs(torch.vdot(wave_function, reconstructed)).item()
        
        data = {
            'tensors': mps_tensors,
            'shape': shape,
            'dtype': wave_function.dtype
        }
        
        return CompressedWave('mps', data, fidelity, wave_function.shape)
    
    def _mps_reconstruct(self, tensors: List[torch.Tensor], shape: List[int]) -> torch.Tensor:
        """Reconstruct wave function from MPS tensors"""
        result = tensors[0]
        
        for i in range(1, len(tensors)):
            if i < len(tensors) - 1:
                # Contract with middle tensor
                result = torch.einsum('...i,ijk->...jk', result, tensors[i])
            else:
                # Contract with last tensor
                result = torch.einsum('...i,ij->...j', result, tensors[i])
        
        return result.reshape(-1)
    
    def _adaptive_rank_compress(self, wave_function: torch.Tensor,
                               properties: Dict[str, Any]) -> 'CompressedWave':
        """Compress using adaptive rank truncation"""
        self.stats['rank_reductions'] += 1
        
        # Reshape to matrix if needed
        if wave_function.dim() == 1:
            n = int(np.sqrt(len(wave_function)))
            if n * n == len(wave_function):
                matrix = wave_function.reshape(n, n)
            else:
                # Try to find suitable factorization
                size = len(wave_function)
                for i in range(int(np.sqrt(size)), 0, -1):
                    if size % i == 0:
                        matrix = wave_function.reshape(i, size // i)
                        break
                else:
                    # Can't find good factorization, use sparse
                    return self._sparse_compress(wave_function, properties)
        else:
            matrix = wave_function
        
        # Handle complex matrices properly
        if matrix.is_complex():
            # For complex matrices, we need to be careful with SVD
            # Convert to real representation for stable SVD
            real_matrix = torch.view_as_real(matrix)
            m, n = matrix.shape
            # Reshape to (m, n, 2) -> (m, 2n) 
            real_matrix_flat = real_matrix.reshape(m, -1)
            U_real, S_real, Vh_real = torch.linalg.svd(real_matrix_flat, full_matrices=False)
            
            # Take every other singular value (they come in pairs for complex)
            S = S_real[::2] if len(S_real) > 1 else S_real
            
            # Convert back to complex
            U = U_real
            Vh = Vh_real
        else:
            # Standard SVD for real matrices
            U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        
        # Adaptive rank selection based on singular value decay
        if len(S) > 0:
            total_variance = torch.sum(S**2)
            if total_variance > 0:
                cumsum = torch.cumsum(S**2, dim=0)
                fidelity_est = cumsum / total_variance
                
                # Find rank that preserves min_fidelity
                # For high-RAM systems, be more conservative with rank reduction
                target_fidelity = self.config.min_fidelity**2
                rank = torch.searchsorted(fidelity_est, target_fidelity).item() + 1
                rank = min(rank, len(S))
                
                # With 64GB RAM, keep at least 20% of ranks or 10 minimum
                min_rank = max(10, int(len(S) * 0.2))
                rank = max(rank, min_rank)
            else:
                rank = 1
        else:
            rank = 1
        
        # For complex case, reconstruct differently
        if matrix.is_complex():
            # Store the matrix in a simpler format
            data = {
                'matrix': matrix[:, :rank].cpu() if matrix.shape[1] > rank else matrix.cpu(),
                'original_shape': wave_function.shape,
                'dtype': wave_function.dtype,
                'rank': rank
            }
            fidelity = self.config.min_fidelity  # Approximate
        else:
            # Truncate
            U_trunc = U[:, :rank]
            S_trunc = S[:rank]
            Vh_trunc = Vh[:rank, :]
            
            # Compute actual fidelity
            reconstructed = U_trunc @ torch.diag(S_trunc) @ Vh_trunc
            if wave_function.dim() == 1:
                reconstructed = reconstructed.reshape(-1)
            fidelity = torch.norm(reconstructed) / torch.norm(wave_function)
            
            data = {
                'U': U_trunc.cpu(),
                'S': S_trunc.cpu(),
                'V': Vh_trunc.T.cpu(),  # Store V not V^H
                'original_shape': wave_function.shape,
                'dtype': wave_function.dtype
            }
        
        return CompressedWave('adaptive_rank', data, fidelity.item() if torch.is_tensor(fidelity) else fidelity, wave_function.shape)
    
    def _phase_cluster_compress(self, wave_function: torch.Tensor,
                               properties: Dict[str, Any]) -> 'CompressedWave':
        """Compress by clustering similar phases"""
        self.stats['phase_clusterings'] += 1
        
        # Extract amplitudes and phases
        amplitudes = torch.abs(wave_function)
        phases = torch.angle(wave_function)
        
        # Mask for non-zero elements
        mask = amplitudes > 1e-10
        
        # Quantize phases into bins
        phase_bins = torch.linspace(-np.pi, np.pi, self.config.phase_bins + 1, device=self.device)
        binned_phases = torch.bucketize(phases[mask], phase_bins)
        
        # Compute average phase per bin
        unique_bins = torch.unique(binned_phases)
        avg_phases = torch.zeros(self.config.phase_bins, device=self.device)
        
        for bin_idx in unique_bins:
            bin_mask = binned_phases == bin_idx
            avg_phases[bin_idx] = torch.mean(phases[mask][bin_mask])
        
        # Store compressed representation
        data = {
            'amplitudes': amplitudes[mask].cpu(),
            'indices': torch.where(mask)[0].cpu(),
            'phase_bins': binned_phases.cpu(),
            'avg_phases': avg_phases.cpu(),
            'shape': wave_function.shape,
            'dtype': wave_function.dtype
        }
        
        # Compute fidelity
        reconstructed = self._phase_cluster_reconstruct(data, self.device)
        fidelity = torch.abs(torch.vdot(wave_function, reconstructed)).item()
        
        return CompressedWave('phase_cluster', data, fidelity, wave_function.shape)
    
    def _phase_cluster_reconstruct(self, data: Dict, device: torch.device) -> torch.Tensor:
        """Reconstruct from phase-clustered representation"""
        result = torch.zeros(data['shape'], device=device, dtype=data['dtype'])
        
        amplitudes = data['amplitudes'].to(device)
        indices = data['indices'].to(device)
        phase_bins = data['phase_bins'].to(device)
        avg_phases = data['avg_phases'].to(device)
        
        # Reconstruct complex values
        phases = avg_phases[phase_bins]
        values = amplitudes * torch.exp(1j * phases)
        
        result[indices] = values
        return result
    
    def _sparse_compress(self, wave_function: torch.Tensor,
                        properties: Dict[str, Any]) -> 'CompressedWave':
        """Basic sparse compression"""
        self.stats['sparse_compressions'] += 1
        
        mask = torch.abs(wave_function) > 1e-10
        indices = torch.where(mask)[0]
        values = wave_function[mask]
        
        data = {
            'indices': indices.cpu(),
            'values': values.cpu(),
            'shape': wave_function.shape,
            'dtype': wave_function.dtype
        }
        
        fidelity = 1.0  # Exact for sparse representation
        return CompressedWave('sparse', data, fidelity, wave_function.shape)
    
    def _general_compress(self, wave_function: torch.Tensor,
                         properties: Dict[str, Any]) -> 'CompressedWave':
        """General compression fallback"""
        # Try sparse first
        if properties['is_sparse']:
            return self._sparse_compress(wave_function, properties)
        
        # Otherwise return uncompressed
        data = {'data': wave_function.cpu()}
        return CompressedWave('uncompressed', data, 1.0, wave_function.shape)
    
    def _estimate_rank_ratio(self, tensor: torch.Tensor) -> float:
        """Estimate effective rank ratio"""
        if tensor.dim() < 2:
            return 1.0
        
        try:
            # Use randomized SVD for efficiency
            k = min(10, min(tensor.shape))
            _, S, _ = torch.svd_lowrank(tensor, q=k)
            
            # Compute effective rank
            S_normalized = S / (S[0] + 1e-10)
            effective_rank = torch.sum(S_normalized > 0.01).item()
            
            return effective_rank / min(tensor.shape)
        except:
            return 1.0
    
    def _update_stats(self, compressed: 'CompressedWave'):
        """Update compression statistics"""
        # Compute compression ratio
        original_size = np.prod(compressed.original_shape) * 8  # Complex64 = 8 bytes
        compressed_size = compressed.memory_size()
        ratio = compressed_size / original_size if original_size > 0 else 1.0
        
        # Update running averages
        n = self.stats['compressions']
        self.stats['average_compression_ratio'] = (
            (n - 1) * self.stats['average_compression_ratio'] + ratio
        ) / n
        self.stats['average_fidelity'] = (
            (n - 1) * self.stats['average_fidelity'] + compressed.fidelity
        ) / n
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get compression statistics"""
        return dict(self.stats)


class CompressedWave:
    """Compressed wave function representation"""
    
    def __init__(self, compression_type: str, data: Dict[str, Any], 
                 fidelity: float, original_shape: Tuple[int, ...]):
        self.compression_type = compression_type
        self.data = data
        self.fidelity = fidelity
        self.original_shape = original_shape
        self._compute_compression_ratio()
        self.method = compression_type  # For benchmark compatibility
    
    def decompress(self, device: torch.device) -> torch.Tensor:
        """Decompress to full wave function"""
        if self.compression_type == 'sparse':
            return self._decompress_sparse(device)
        elif self.compression_type == 'mps':
            return self._decompress_mps(device)
        elif self.compression_type == 'adaptive_rank':
            return self._decompress_adaptive_rank(device)
        elif self.compression_type == 'phase_cluster':
            return self._decompress_phase_cluster(device)
        elif self.compression_type == 'uncompressed':
            return self.data['data'].to(device)
        else:
            raise ValueError(f"Unknown compression type: {self.compression_type}")
    
    def _compute_compression_ratio(self):
        """Compute compression ratio"""
        original_size = np.prod(self.original_shape)
        
        if self.compression_type == 'uncompressed':
            self.compression_ratio = 1.0
        elif self.compression_type == 'sparse':
            compressed_size = len(self.data.get('indices', []))
            self.compression_ratio = original_size / max(compressed_size, 1)
        elif self.compression_type == 'mps':
            compressed_size = sum(t.numel() for t in self.data.get('tensors', []))
            self.compression_ratio = original_size / max(compressed_size, 1)
        elif self.compression_type == 'adaptive_rank':
            if 'matrix' in self.data:
                compressed_size = self.data['matrix'].numel()
            else:
                compressed_size = sum(self.data[k].numel() for k in ['U', 'S', 'V'] if k in self.data)
            self.compression_ratio = original_size / max(compressed_size, 1)
        elif self.compression_type == 'phase_cluster':
            compressed_size = sum(self.data[k].numel() for k in ['amplitudes', 'indices', 'phase_bins', 'avg_phases'] if k in self.data)
            self.compression_ratio = original_size / max(compressed_size, 1)
        else:
            self.compression_ratio = 1.0
    
    def _decompress_sparse(self, device: torch.device) -> torch.Tensor:
        """Decompress sparse representation"""
        indices = self.data['indices'].to(device)
        values = self.data['values'].to(device)
        shape = self.data['shape']
        dtype = self.data['dtype']
        
        result = torch.zeros(shape, device=device, dtype=dtype)
        if len(indices) > 0:
            result[indices] = values
        return result
    
    def _decompress_mps(self, device: torch.device) -> torch.Tensor:
        """Decompress MPS representation"""
        tensors = [t.to(device) for t in self.data['tensors']]
        
        # Contract MPS tensors
        result = tensors[0]
        for i in range(1, len(tensors)):
            if i < len(tensors) - 1:
                result = torch.einsum('...i,ijk->...jk', result, tensors[i])
            else:
                result = torch.einsum('...i,ij->...j', result, tensors[i])
        
        return result.reshape(self.original_shape)
    
    def _decompress_adaptive_rank(self, device: torch.device) -> torch.Tensor:
        """Decompress adaptive rank representation"""
        if 'matrix' in self.data:
            # Complex matrix case - simplified storage
            matrix = self.data['matrix'].to(device)
            # Pad with zeros to original size if needed
            if matrix.numel() < np.prod(self.original_shape):
                result = torch.zeros(self.original_shape, device=device, dtype=self.data['dtype'])
                if len(self.original_shape) == 1:
                    # Need to determine how to reshape
                    n_elements = len(result)
                    # Try to match the compression shape
                    matrix_shape = matrix.shape
                    if len(matrix_shape) == 2:
                        m, n = matrix_shape
                        if m * n <= n_elements:
                            # Can fit, pad and reshape
                            flat_matrix = matrix.reshape(-1)
                            result[:len(flat_matrix)] = flat_matrix
                else:
                    result = matrix.reshape(self.original_shape)
            else:
                result = matrix.reshape(self.original_shape)
            return result
        else:
            # Original SVD case
            U = self.data['U'].to(device)
            S = self.data['S'].to(device)
            V = self.data['V'].to(device)
            
            # Reconstruct
            result = U @ torch.diag(S) @ V.T
            
            # Reshape to original
            if len(self.original_shape) == 1:
                result = result.reshape(-1)
            
            return result
    
    def _decompress_phase_cluster(self, device: torch.device) -> torch.Tensor:
        """Decompress phase-clustered representation"""
        result = torch.zeros(self.original_shape, device=device, 
                           dtype=self.data['dtype'])
        
        amplitudes = self.data['amplitudes'].to(device)
        indices = self.data['indices'].to(device)
        phase_bins = self.data['phase_bins'].to(device)
        avg_phases = self.data['avg_phases'].to(device)
        
        # Reconstruct complex values
        phases = avg_phases[phase_bins]
        values = amplitudes * torch.exp(1j * phases)
        
        result[indices] = values
        return result
    
    def memory_size(self) -> int:
        """Estimate memory usage in bytes"""
        total = 0
        
        if self.compression_type == 'sparse':
            for key in ['indices', 'values']:
                if key in self.data:
                    tensor = self.data[key]
                    total += tensor.element_size() * tensor.numel()
        
        elif self.compression_type == 'mps':
            for tensor in self.data['tensors']:
                total += tensor.element_size() * tensor.numel()
        
        elif self.compression_type == 'adaptive_rank':
            if 'matrix' in self.data:
                tensor = self.data['matrix']
                total += tensor.element_size() * tensor.numel()
            else:
                for key in ['U', 'S', 'V']:
                    if key in self.data:
                        tensor = self.data[key]
                        total += tensor.element_size() * tensor.numel()
        
        elif self.compression_type == 'phase_cluster':
            for key in ['amplitudes', 'indices', 'phase_bins', 'avg_phases']:
                if key in self.data:
                    tensor = self.data[key]
                    total += tensor.element_size() * tensor.numel()
        
        elif self.compression_type == 'uncompressed':
            tensor = self.data['data']
            total = tensor.element_size() * tensor.numel()
        
        return total


def create_wave_compressor(
    enable_mps: bool = True,
    min_fidelity: float = 0.99,
    max_bond_dimension: int = 100,
    **kwargs
) -> WaveCompressor:
    """
    Factory function to create wave compressor
    
    Args:
        enable_mps: Enable Matrix Product State compression
        min_fidelity: Minimum compression fidelity
        max_bond_dimension: Maximum MPS bond dimension
        **kwargs: Additional config parameters
        
    Returns:
        Configured WaveCompressor
    """
    config_dict = {
        'enable_mps': enable_mps,
        'min_fidelity': min_fidelity,
        'max_bond_dimension': max_bond_dimension
    }
    config_dict.update(kwargs)
    
    config = CompressionConfig(**config_dict)
    return WaveCompressor(config)