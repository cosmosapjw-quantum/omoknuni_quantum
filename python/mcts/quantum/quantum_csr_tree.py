"""
Quantum-Enhanced CSR Tree Format
================================

This module extends the CSR (Compressed Sparse Row) tree format to properly
handle quantum superposition states in MCTS. It provides:

- Quantum state representation in CSR format
- Efficient sparse storage for quantum amplitudes
- GPU-optimized operations for quantum state evolution
- Integration with decoherence and interference engines

Key Features:
- Stores complex amplitudes for superposition states
- Maintains quantum coherence information
- Efficient density matrix operations in CSR format
- Supports entanglement between tree paths

Based on: QFT-MCTS framework with quantum superposition support
"""

import torch
import torch.sparse as sparse
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from functools import lru_cache

from ..gpu.csr_tree import CSRTree, CSRTreeConfig
from .state_pool import QuantumStatePool, create_quantum_state_pool

logger = logging.getLogger(__name__)


@dataclass
class QuantumCSRConfig(CSRTreeConfig):
    """Configuration for quantum-enhanced CSR tree"""
    # Quantum-specific parameters
    enable_superposition: bool = True
    amplitude_threshold: float = 1e-6  # Threshold for pruning small amplitudes
    max_entangled_paths: int = 100     # Maximum paths in superposition
    coherence_time: float = 10.0       # Decoherence timescale
    
    # Density matrix parameters
    use_density_matrix: bool = True
    density_matrix_rank: int = 50      # For low-rank approximation
    
    # Memory optimization
    compress_quantum_states: bool = True
    quantum_pool_size: int = 1000      # Size of quantum state pool
    
    # GPU optimization
    use_sparse_kernels: bool = True
    sparse_block_size: int = 32        # Block size for sparse operations
    
    def __post_init__(self):
        """Initialize parent and quantum-specific dtypes"""
        super().__post_init__()
        # Complex dtypes for quantum amplitudes
        # Always use complex64 to avoid ComplexHalf issues
        self.dtype_amplitude = torch.complex64


class QuantumCSRTree(CSRTree):
    """
    Quantum-enhanced CSR tree with superposition support
    
    This class extends the CSR tree format to handle quantum superposition
    states efficiently. It maintains both classical tree structure and
    quantum amplitudes in a GPU-friendly format.
    
    Key additions:
    - Complex amplitudes for each node (superposition coefficients)
    - Density matrix representation for mixed states
    - Entanglement tracking between paths
    - Efficient sparse operations for quantum evolution
    """
    
    def __init__(self, config: QuantumCSRConfig):
        # Initialize parent CSR tree
        super().__init__(config)
        self.quantum_config = config
        
        # Initialize quantum-specific storage
        self._init_quantum_storage()
        
        # Create quantum state pool for memory efficiency
        if config.compress_quantum_states:
            self.quantum_pool = create_quantum_state_pool(
                device=self.device,
                max_states=config.quantum_pool_size,
                compression_threshold=0.8
            )
        else:
            self.quantum_pool = None
        
        # Quantum statistics
        self.quantum_stats = {
            'superposition_count': 0,
            'max_entanglement': 0,
            'decoherence_events': 0,
            'quantum_compressions': 0
        }
        
        logger.info(f"Initialized QuantumCSRTree with superposition support")
    
    def _init_quantum_storage(self):
        """Initialize quantum-specific storage arrays"""
        n = self.max_nodes
        device = self.device
        config = self.quantum_config
        
        # Quantum amplitudes (complex) - initially all nodes have zero amplitude
        self.quantum_amplitudes = torch.zeros(n, device=device, dtype=config.dtype_amplitude)
        
        # Coherence factors (real) - track decoherence
        self.coherence_factors = torch.ones(n, device=device, dtype=config.dtype_values)
        
        # Entanglement matrix (sparse) - tracks quantum correlations
        # Start with empty sparse tensor
        indices = torch.zeros((2, 0), dtype=torch.int64, device=device)
        values = torch.zeros(0, dtype=config.dtype_amplitude, device=device)
        self.entanglement_matrix = torch.sparse_coo_tensor(
            indices, values, (n, n), device=device
        )
        
        # Quantum phase tracking (in addition to parent's phases)
        self.quantum_phases = torch.zeros(n, device=device, dtype=config.dtype_values)
        
        # Density matrix blocks for mixed states (if enabled)
        if config.use_density_matrix:
            # Store density matrix in block-sparse format
            self.density_blocks = {}  # Dict mapping block indices to tensors
            self.block_indices = torch.zeros((n, 2), dtype=torch.int32, device=device)
            self.block_indices.fill_(-1)  # -1 indicates no block assigned
    
    def create_quantum_superposition(
        self,
        node_indices: torch.Tensor,
        amplitudes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Create quantum superposition of multiple nodes
        
        Args:
            node_indices: Tensor of node indices to superpose
            amplitudes: Optional complex amplitudes (normalized automatically)
            
        Returns:
            Normalized quantum state vector
        """
        if not self.quantum_config.enable_superposition:
            # Fall back to classical - return one-hot
            state = torch.zeros(self.num_nodes, device=self.device, dtype=self.quantum_config.dtype_amplitude)
            if len(node_indices) > 0:
                state[node_indices[0]] = 1.0
            return state
        
        # Create superposition state
        num_paths = len(node_indices)
        
        if amplitudes is None:
            # Equal superposition
            amplitude = 1.0 / np.sqrt(num_paths)
            amplitudes = torch.full((num_paths,), amplitude, 
                                  device=self.device, 
                                  dtype=self.quantum_config.dtype_amplitude)
        else:
            # Normalize provided amplitudes
            norm = torch.sqrt(torch.sum(torch.abs(amplitudes)**2))
            amplitudes = amplitudes / norm
        
        # Update quantum amplitudes
        self.quantum_amplitudes.zero_()
        self.quantum_amplitudes[node_indices] = amplitudes
        
        # Update statistics
        self.quantum_stats['superposition_count'] += 1
        self.quantum_stats['max_entanglement'] = max(
            self.quantum_stats['max_entanglement'], num_paths
        )
        
        return self.quantum_amplitudes[node_indices]
    
    def apply_quantum_interference(
        self,
        source_nodes: torch.Tensor,
        target_nodes: torch.Tensor,
        interference_matrix: Optional[torch.Tensor] = None
    ):
        """
        Apply quantum interference between paths
        
        This modifies amplitudes based on phase relationships and
        creates entanglement between interfering paths.
        """
        if interference_matrix is None:
            # Default: Use phase differences
            source_phases = self.quantum_phases[source_nodes]
            target_phases = self.quantum_phases[target_nodes]
            
            # Phase difference matrix (ensure float32 to avoid ComplexHalf)
            phase_diff = source_phases.unsqueeze(1) - target_phases.unsqueeze(0)
            phase_diff = phase_diff.to(torch.float32)
            interference_matrix = torch.exp(1j * phase_diff)
        
        # Apply interference to amplitudes
        source_amps = self.quantum_amplitudes[source_nodes]
        target_amps = self.quantum_amplitudes[target_nodes]
        
        # Interference term
        interference = torch.matmul(interference_matrix, target_amps)
        
        # Update amplitudes with interference
        self.quantum_amplitudes[source_nodes] += 0.1 * interference  # Coupling strength
        
        # Renormalize
        self._normalize_quantum_state()
        
        # Update entanglement matrix
        self._update_entanglement(source_nodes, target_nodes, interference_matrix)
    
    def compute_density_matrix_csr(
        self,
        node_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute density matrix in CSR format for efficiency
        
        Returns:
            Tuple of (row_ptr, col_indices, values) for CSR representation
        """
        if node_indices is None:
            # Use all nodes with non-zero amplitude
            mask = torch.abs(self.quantum_amplitudes) > self.quantum_config.amplitude_threshold
            node_indices = torch.where(mask)[0]
        
        n = len(node_indices)
        if n == 0:
            # Empty density matrix
            row_ptr = torch.zeros(1, dtype=torch.int32, device=self.device)
            col_indices = torch.zeros(0, dtype=torch.int32, device=self.device)
            values = torch.zeros(0, dtype=self.quantum_config.dtype_amplitude, device=self.device)
            return row_ptr, col_indices, values
        
        # Get amplitudes
        amplitudes = self.quantum_amplitudes[node_indices]
        
        # Compute density matrix elements: ρ_ij = ψ_i ψ_j*
        # For CSR format, we only store non-zero elements
        
        # Threshold for considering elements non-zero
        # Ensure threshold is a tensor on the same device for comparison
        threshold = torch.tensor(self.quantum_config.amplitude_threshold**2, 
                                device=self.device, dtype=torch.float32)
        
        # Build CSR arrays
        row_offsets = []
        col_list = []
        value_list = []
        
        for i in range(n):
            row_start = len(col_list)
            amp_i = amplitudes[i]
            
            for j in range(n):
                amp_j_conj = amplitudes[j].conj()
                value = amp_i * amp_j_conj
                
                # Ensure we handle scalar tensors properly for CUDA
                abs_value = torch.abs(value)
                if abs_value.item() > threshold.item():
                    col_list.append(j)
                    value_list.append(value)
            
            row_offsets.append(row_start)
        
        row_offsets.append(len(col_list))  # Final offset
        
        # Convert to tensors
        row_ptr = torch.tensor(row_offsets, dtype=torch.int32, device=self.device)
        col_indices = torch.tensor(col_list, dtype=torch.int32, device=self.device)
        values = torch.stack(value_list) if value_list else torch.zeros(0, dtype=self.quantum_config.dtype_amplitude, device=self.device)
        
        return row_ptr, col_indices, values
    
    def evolve_quantum_state(
        self,
        hamiltonian: torch.Tensor,
        time_step: float = 0.01
    ):
        """
        Evolve quantum state under Hamiltonian
        
        Uses sparse matrix operations for efficiency.
        """
        # Get active nodes
        mask = torch.abs(self.quantum_amplitudes) > self.quantum_config.amplitude_threshold
        active_nodes = torch.where(mask)[0]
        
        if len(active_nodes) == 0:
            return
        
        # Extract sub-Hamiltonian for active nodes
        H_sub = hamiltonian[active_nodes][:, active_nodes]
        
        # Current amplitudes
        psi = self.quantum_amplitudes[active_nodes]
        
        # Time evolution: psi(t+dt) = exp(-iHdt/ℏ) psi(t)
        # Use first-order approximation for small dt
        # Ensure H_sub is complex for matrix multiplication with complex psi
        H_sub_complex = H_sub.to(dtype=self.quantum_config.dtype_amplitude)
        dpsi = -1j * torch.matmul(H_sub_complex, psi) * time_step
        
        # Update amplitudes
        self.quantum_amplitudes[active_nodes] += dpsi
        
        # Apply decoherence
        self._apply_decoherence(active_nodes, time_step)
        
        # Renormalize
        self._normalize_quantum_state()
    
    def _apply_decoherence(self, node_indices: torch.Tensor, time_step: float):
        """Apply decoherence to quantum amplitudes"""
        # Exponential decay of coherence
        decay_rate = 1.0 / self.quantum_config.coherence_time
        decay_factor = torch.exp(torch.tensor(-decay_rate * time_step, device=self.device))
        
        # Update coherence factors
        self.coherence_factors[node_indices] *= decay_factor
        
        # Apply to amplitudes
        self.quantum_amplitudes[node_indices] *= torch.sqrt(decay_factor)
        
        # Track decoherence events
        decoherent_nodes = self.coherence_factors[node_indices] < 0.1
        if torch.any(decoherent_nodes):
            self.quantum_stats['decoherence_events'] += torch.sum(decoherent_nodes).item()
    
    def _normalize_quantum_state(self):
        """Normalize quantum amplitudes to maintain unitarity"""
        norm_squared = torch.sum(torch.abs(self.quantum_amplitudes)**2)
        if norm_squared > 0:
            self.quantum_amplitudes /= torch.sqrt(norm_squared)
    
    def _update_entanglement(
        self,
        nodes1: torch.Tensor,
        nodes2: torch.Tensor,
        correlation: torch.Tensor
    ):
        """Update entanglement matrix with quantum correlations"""
        # Build sparse update
        n1, n2 = len(nodes1), len(nodes2)
        
        # Create index pairs
        idx1 = nodes1.repeat_interleave(n2)
        idx2 = nodes2.repeat(n1)
        indices = torch.stack([idx1, idx2])
        
        # Flatten correlation values
        values = correlation.flatten()
        
        # Add to existing entanglement matrix
        update = torch.sparse_coo_tensor(
            indices, values, 
            self.entanglement_matrix.shape,
            device=self.device
        )
        
        self.entanglement_matrix = self.entanglement_matrix + update
    
    def extract_classical_probabilities(self) -> torch.Tensor:
        """
        Extract classical probabilities from quantum state
        
        Returns:
            Probability distribution over nodes
        """
        # Born rule: P(i) = |ψ_i|²
        probabilities = torch.abs(self.quantum_amplitudes)**2
        
        # Weight by coherence factors
        probabilities *= self.coherence_factors
        
        # Renormalize
        total = torch.sum(probabilities)
        if total > 0:
            probabilities /= total
        
        return probabilities
    
    def get_quantum_statistics(self) -> Dict[str, Any]:
        """Get quantum-specific statistics"""
        stats = dict(self.quantum_stats)
        
        # Add current state info
        mask = torch.abs(self.quantum_amplitudes) > self.quantum_config.amplitude_threshold
        stats['active_superposition_size'] = torch.sum(mask).item()
        stats['average_coherence'] = torch.mean(self.coherence_factors).item()
        stats['entanglement_sparsity'] = self.entanglement_matrix._nnz() / (self.num_nodes**2)
        
        if self.quantum_pool:
            stats['quantum_pool_stats'] = self.quantum_pool.get_statistics()
        
        return stats
    
    def compress_quantum_states(self):
        """Compress quantum states using state pool"""
        if not self.quantum_pool:
            return
        
        # Compress sparse amplitudes
        if torch.sum(torch.abs(self.quantum_amplitudes) > 0) < 0.1 * self.num_nodes:
            compressed = self.quantum_pool.compress_state(
                self.quantum_amplitudes.unsqueeze(0)
            )
            if compressed:
                self.quantum_stats['quantum_compressions'] += 1
                # Store compressed representation
                self._compressed_amplitudes = compressed
                # Clear original to save memory
                self.quantum_amplitudes.zero_()


def create_quantum_csr_tree(
    max_nodes: int = 500_000,
    device: Union[str, torch.device] = 'cuda',
    enable_superposition: bool = True,
    **kwargs
) -> QuantumCSRTree:
    """
    Factory function to create quantum CSR tree
    
    Args:
        max_nodes: Maximum number of tree nodes
        device: Device for computation
        enable_superposition: Whether to enable quantum features
        **kwargs: Additional config parameters
        
    Returns:
        Initialized QuantumCSRTree
    """
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    config_dict = {
        'max_nodes': max_nodes,
        'device': device.type,
        'enable_superposition': enable_superposition,
        'max_edges': max_nodes * 5  # Estimate
    }
    config_dict.update(kwargs)
    
    config = QuantumCSRConfig(**config_dict)
    
    return QuantumCSRTree(config)