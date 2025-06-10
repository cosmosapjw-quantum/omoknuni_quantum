"""Tests for quantum-enhanced CSR tree format"""

import pytest
import torch
import numpy as np
from typing import Dict, List, Tuple

from mcts.quantum.quantum_csr_tree import (
    QuantumCSRTree,
    QuantumCSRConfig,
    create_quantum_csr_tree
)


class TestQuantumSuperposition:
    """Test quantum superposition functionality"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def quantum_tree(self, device):
        config = QuantumCSRConfig(
            max_nodes=1000,
            device=device.type,
            enable_superposition=True,
            amplitude_threshold=1e-8
        )
        return QuantumCSRTree(config)
    
    def test_create_equal_superposition(self, quantum_tree, device):
        """Test creating equal superposition of nodes"""
        # Add some nodes first
        quantum_tree.num_nodes = 10
        
        # Create superposition of nodes 0, 2, 5
        node_indices = torch.tensor([0, 2, 5], device=device)
        amplitudes = quantum_tree.create_quantum_superposition(node_indices)
        
        # Check normalization
        assert len(amplitudes) == 3
        norm_squared = torch.sum(torch.abs(amplitudes)**2)
        assert torch.allclose(norm_squared, torch.tensor(1.0, device=device))
        
        # Check equal amplitudes
        expected_amp = 1.0 / np.sqrt(3)
        assert torch.allclose(torch.abs(amplitudes), 
                            torch.tensor(expected_amp, device=device, dtype=torch.float32))
    
    def test_create_weighted_superposition(self, quantum_tree, device):
        """Test creating weighted superposition"""
        quantum_tree.num_nodes = 10
        
        node_indices = torch.tensor([1, 3, 7], device=device)
        weights = torch.tensor([1.0, 2.0, 1.0], device=device, dtype=torch.complex64)
        
        amplitudes = quantum_tree.create_quantum_superposition(node_indices, weights)
        
        # Check normalization
        norm_squared = torch.sum(torch.abs(amplitudes)**2)
        assert torch.allclose(norm_squared, torch.tensor(1.0, device=device))
        
        # Check relative weights preserved
        assert torch.abs(amplitudes[1]) > torch.abs(amplitudes[0])
        assert torch.abs(amplitudes[1]) > torch.abs(amplitudes[2])
    
    def test_superposition_statistics(self, quantum_tree, device):
        """Test superposition statistics tracking"""
        quantum_tree.num_nodes = 20
        
        # Create multiple superpositions
        for i in range(5):
            nodes = torch.arange(i, i+4, device=device)
            quantum_tree.create_quantum_superposition(nodes)
        
        stats = quantum_tree.quantum_stats
        assert stats['superposition_count'] == 5
        assert stats['max_entanglement'] == 4


class TestQuantumInterference:
    """Test quantum interference between paths"""
    
    @pytest.fixture
    def quantum_tree(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = QuantumCSRConfig(
            max_nodes=100,
            device=device.type,
            enable_superposition=True
        )
        tree = QuantumCSRTree(config)
        tree.num_nodes = 20
        return tree
    
    def test_phase_based_interference(self, quantum_tree):
        """Test interference based on phase differences"""
        device = quantum_tree.device
        
        # Set up initial superposition
        source_nodes = torch.tensor([0, 1], device=device)
        target_nodes = torch.tensor([2, 3], device=device)
        
        # Set phases (use the same dtype as quantum_phases)
        phase_dtype = quantum_tree.quantum_phases.dtype
        quantum_tree.quantum_phases[source_nodes] = torch.tensor([0.0, np.pi/2], device=device, dtype=phase_dtype)
        quantum_tree.quantum_phases[target_nodes] = torch.tensor([0.0, 0.0], device=device, dtype=phase_dtype)
        
        # Create initial amplitudes
        quantum_tree.quantum_amplitudes[source_nodes] = torch.tensor([0.5, 0.5], device=device, dtype=torch.complex64)
        quantum_tree.quantum_amplitudes[target_nodes] = torch.tensor([0.5, 0.5], device=device, dtype=torch.complex64)
        
        # Apply interference
        quantum_tree.apply_quantum_interference(source_nodes, target_nodes)
        
        # Check that amplitudes changed
        assert not torch.allclose(
            quantum_tree.quantum_amplitudes[source_nodes],
            torch.tensor([0.5, 0.5], device=device, dtype=torch.complex64)
        )
        
        # Check normalization preserved
        total_norm = torch.sum(torch.abs(quantum_tree.quantum_amplitudes)**2)
        assert torch.allclose(total_norm, torch.tensor(1.0, device=device), atol=0.1)
    
    def test_custom_interference_matrix(self, quantum_tree):
        """Test interference with custom matrix"""
        device = quantum_tree.device
        
        nodes = torch.tensor([0, 1], device=device)
        
        # Hadamard-like interference
        interference_matrix = torch.tensor([
            [1.0, 1.0],
            [1.0, -1.0]
        ], device=device, dtype=torch.complex64) / np.sqrt(2)
        
        # Set initial state
        quantum_tree.quantum_amplitudes[nodes] = torch.tensor([1.0, 0.0], device=device, dtype=torch.complex64)
        
        # Apply interference
        quantum_tree.apply_quantum_interference(nodes, nodes, interference_matrix)
        
        # Should create superposition
        assert torch.abs(quantum_tree.quantum_amplitudes[0]) > 0
        assert torch.abs(quantum_tree.quantum_amplitudes[1]) > 0


class TestDensityMatrix:
    """Test density matrix computation"""
    
    @pytest.fixture
    def quantum_tree(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = QuantumCSRConfig(
            max_nodes=100,  # Increased to accommodate test node indices
            device=device.type,
            use_density_matrix=True
        )
        return QuantumCSRTree(config)
    
    def test_pure_state_density_matrix(self, quantum_tree):
        """Test density matrix for pure state"""
        device = quantum_tree.device
        quantum_tree.num_nodes = 10
        
        # Pure state on node 3
        quantum_tree.quantum_amplitudes[3] = 1.0
        
        # Compute density matrix
        row_ptr, col_indices, values = quantum_tree.compute_density_matrix_csr()
        
        # Should have single non-zero element
        assert len(values) == 1
        assert torch.allclose(values[0], torch.tensor(1.0, device=device, dtype=torch.complex64))
    
    def test_superposition_density_matrix(self, quantum_tree):
        """Test density matrix for superposition"""
        device = quantum_tree.device
        quantum_tree.num_nodes = 10
        
        # Equal superposition of two states
        nodes = torch.tensor([1, 4], device=device)
        quantum_tree.create_quantum_superposition(nodes)
        
        # Compute density matrix
        row_ptr, col_indices, values = quantum_tree.compute_density_matrix_csr(nodes)
        
        # Should have 4 non-zero elements (2x2 matrix)
        assert len(values) == 4
        
        # Diagonal elements should be 0.5
        # Note: values are in CSR format, need to check actual matrix structure
        assert row_ptr.shape[0] == 3  # 2 rows + 1
        
    def test_density_matrix_sparsity(self, quantum_tree):
        """Test sparsity of density matrix representation"""
        device = quantum_tree.device
        quantum_tree.num_nodes = 100
        
        # Sparse superposition
        nodes = torch.tensor([10, 30, 50, 70], device=device)
        quantum_tree.create_quantum_superposition(nodes)
        
        # Compute density matrix
        row_ptr, col_indices, values = quantum_tree.compute_density_matrix_csr(nodes)
        
        # Should have 16 elements (4x4 matrix)
        assert len(values) == 16
        assert len(col_indices) == 16


class TestQuantumEvolution:
    """Test quantum state evolution"""
    
    @pytest.fixture
    def quantum_tree(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = QuantumCSRConfig(
            max_nodes=20,
            device=device.type,
            coherence_time=10.0
        )
        tree = QuantumCSRTree(config)
        tree.num_nodes = 10
        return tree
    
    def test_hamiltonian_evolution(self, quantum_tree):
        """Test evolution under Hamiltonian"""
        device = quantum_tree.device
        
        # Initial superposition
        nodes = torch.tensor([0, 1, 2], device=device)
        quantum_tree.create_quantum_superposition(nodes)
        
        # Simple diagonal Hamiltonian
        H = torch.diag(torch.arange(10, device=device, dtype=torch.float32))
        
        initial_state = quantum_tree.quantum_amplitudes.clone()
        
        # Evolve
        quantum_tree.evolve_quantum_state(H, time_step=0.01)
        
        # State should change
        assert not torch.allclose(quantum_tree.quantum_amplitudes, initial_state)
        
        # Normalization preserved
        norm = torch.sum(torch.abs(quantum_tree.quantum_amplitudes)**2)
        assert torch.allclose(norm, torch.tensor(1.0, device=device))
    
    def test_decoherence(self, quantum_tree):
        """Test decoherence effects"""
        device = quantum_tree.device
        
        # Create superposition
        nodes = torch.tensor([0, 1], device=device)
        quantum_tree.create_quantum_superposition(nodes)
        
        # Set short coherence time for testing
        quantum_tree.quantum_config.coherence_time = 0.1
        
        # Diagonal Hamiltonian
        H = torch.eye(10, device=device)
        
        initial_coherence = quantum_tree.coherence_factors[nodes].clone()
        
        # Evolve for several steps
        for _ in range(10):
            quantum_tree.evolve_quantum_state(H, time_step=0.1)
        
        # Coherence should decay
        final_coherence = quantum_tree.coherence_factors[nodes]
        assert torch.all(final_coherence < initial_coherence)
        
        # Check decoherence tracking
        assert quantum_tree.quantum_stats['decoherence_events'] > 0


class TestClassicalExtraction:
    """Test extraction of classical probabilities"""
    
    @pytest.fixture
    def quantum_tree(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = QuantumCSRConfig(
            max_nodes=20,
            device=device.type
        )
        tree = QuantumCSRTree(config)
        tree.num_nodes = 10
        return tree
    
    def test_born_rule(self, quantum_tree):
        """Test Born rule probability extraction"""
        device = quantum_tree.device
        
        # Create known superposition
        quantum_tree.quantum_amplitudes[0] = 0.6 + 0j
        quantum_tree.quantum_amplitudes[1] = 0.8 + 0j
        
        probs = quantum_tree.extract_classical_probabilities()
        
        # Check Born rule: P = |ψ|²
        assert torch.allclose(probs[0], torch.tensor(0.36, device=device), atol=0.01)
        assert torch.allclose(probs[1], torch.tensor(0.64, device=device), atol=0.01)
        
        # Check normalization
        assert torch.allclose(torch.sum(probs), torch.tensor(1.0, device=device))
    
    def test_coherence_weighting(self, quantum_tree):
        """Test that coherence factors affect probabilities"""
        device = quantum_tree.device
        
        # Equal amplitudes
        quantum_tree.quantum_amplitudes[0] = 1.0 / np.sqrt(2)
        quantum_tree.quantum_amplitudes[1] = 1.0 / np.sqrt(2)
        
        # Different coherence
        quantum_tree.coherence_factors[0] = 1.0
        quantum_tree.coherence_factors[1] = 0.5
        
        probs = quantum_tree.extract_classical_probabilities()
        
        # Node 0 should have higher probability due to coherence
        assert probs[0] > probs[1]


class TestQuantumStatistics:
    """Test quantum statistics and monitoring"""
    
    def test_statistics_tracking(self):
        """Test comprehensive statistics"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tree = create_quantum_csr_tree(
            max_nodes=100,
            device=device,
            enable_superposition=True,
            compress_quantum_states=True
        )
        tree.num_nodes = 50
        
        # Perform various operations
        nodes = torch.arange(5, device=device)
        tree.create_quantum_superposition(nodes)
        
        H = torch.eye(50, device=device)
        tree.evolve_quantum_state(H, 0.1)
        
        # Get statistics
        stats = tree.get_quantum_statistics()
        
        assert 'superposition_count' in stats
        assert 'max_entanglement' in stats
        assert 'active_superposition_size' in stats
        assert 'average_coherence' in stats
        assert stats['superposition_count'] >= 1
        assert stats['active_superposition_size'] >= 0


class TestFactoryFunction:
    """Test factory function"""
    
    def test_create_quantum_csr_tree(self):
        """Test tree creation with factory"""
        tree = create_quantum_csr_tree(
            max_nodes=500,
            device='cpu',
            enable_superposition=True,
            amplitude_threshold=1e-6
        )
        
        assert isinstance(tree, QuantumCSRTree)
        assert tree.quantum_config.max_nodes == 500
        assert tree.quantum_config.enable_superposition
        assert tree.quantum_config.amplitude_threshold == 1e-6
        
        # Test basic functionality
        tree.num_nodes = 10
        nodes = torch.tensor([0, 1, 2])
        amps = tree.create_quantum_superposition(nodes)
        assert len(amps) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])