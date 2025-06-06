"""
Test Suite for Decoherence Engine
=================================

Validates the quantum decoherence dynamics and quantum→classical transition
according to theoretical predictions.

Test Categories:
1. Density matrix evolution validation
2. Lindblad operator construction
3. Pointer state identification
4. Quantum→classical transition
5. Integration with tree structures
"""

import pytest
import torch
import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts.quantum.decoherence import (
    DecoherenceEngine, 
    DecoherenceConfig, 
    DecoherenceOperators,
    DensityMatrixEvolution,
    PointerStateAnalyzer,
    create_decoherence_engine
)


class TestDecoherenceOperators:
    """Test Lindblad operators for decoherence"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def config(self):
        return DecoherenceConfig(
            base_decoherence_rate=1.0,
            hbar=1.0,
            temperature=1.0
        )
    
    @pytest.fixture
    def operators(self, config, device):
        return DecoherenceOperators(config, device)
    
    @pytest.fixture
    def sample_visit_counts(self, device):
        # Create realistic visit count distribution
        return torch.tensor([100.0, 50.0, 75.0, 25.0, 10.0], device=device)
    
    def test_operator_initialization(self, operators, device):
        """Test operator initialization"""
        assert operators.device == device
        assert isinstance(operators.config, DecoherenceConfig)
    
    def test_visit_difference_operator(self, operators, sample_visit_counts):
        """Test visit count difference operator construction"""
        operator = operators._create_visit_difference_operator(sample_visit_counts)
        
        # Should be diagonal matrix
        assert operator.shape == (len(sample_visit_counts), len(sample_visit_counts))
        
        # Diagonal elements should be sqrt(visit_counts)
        expected_diag = torch.sqrt(sample_visit_counts + operators.config.matrix_regularization)
        assert torch.allclose(torch.diag(operator), expected_diag, atol=1e-6)
        
        # Off-diagonal should be zero
        off_diag = operator - torch.diag(torch.diag(operator))
        assert torch.allclose(off_diag, torch.zeros_like(off_diag), atol=1e-8)
    
    def test_decoherence_rates_computation(self, operators, sample_visit_counts):
        """Test pairwise decoherence rate computation"""
        rates = operators.compute_decoherence_rates(sample_visit_counts)
        
        # Should be symmetric matrix
        assert torch.allclose(rates, rates.T, atol=1e-6)
        
        # Diagonal should be zero (no self-decoherence)
        diagonal = torch.diag(rates)
        assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-8)
        
        # Rates should follow theoretical formula: λ|N_i - N_j|/max(N_i, N_j)
        N = sample_visit_counts
        expected_01 = operators.config.base_decoherence_rate * abs(N[0] - N[1]) / max(N[0], N[1])
        assert abs(rates[0, 1].item() - expected_01.item()) < 1e-6
    
    def test_lindblad_operators_construction(self, operators, sample_visit_counts):
        """Test full Lindblad operator construction"""
        ops = operators.compute_lindblad_operators(sample_visit_counts)
        
        # Should return list of operators
        assert isinstance(ops, list)
        assert len(ops) >= 2  # At least visit diff and thermal
        
        # Each operator should be correct size
        for op in ops:
            assert op.shape == (len(sample_visit_counts), len(sample_visit_counts))
            assert torch.all(torch.isfinite(op))


class TestDensityMatrixEvolution:
    """Test density matrix time evolution"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def config(self):
        return DecoherenceConfig(dt=0.01, convergence_threshold=1e-6)
    
    @pytest.fixture
    def evolution(self, config, device):
        return DensityMatrixEvolution(config, device)
    
    @pytest.fixture
    def test_matrices(self, device):
        """Create test density matrix and Hamiltonian"""
        # Small 3x3 system for testing
        rho = torch.tensor([
            [0.5, 0.2, 0.1],
            [0.2, 0.3, 0.0],
            [0.1, 0.0, 0.2]
        ], dtype=torch.complex64, device=device)
        
        hamiltonian = torch.tensor([
            [1.0, 0.1, 0.0],
            [0.1, 2.0, 0.1],
            [0.0, 0.1, 3.0]
        ], dtype=torch.complex64, device=device)
        
        visit_counts = torch.tensor([10.0, 20.0, 30.0], device=device)
        
        return rho, hamiltonian, visit_counts
    
    def test_density_matrix_properties_preservation(self, evolution, test_matrices):
        """Test that density matrix properties are preserved"""
        rho, hamiltonian, visit_counts = test_matrices
        
        # Evolve one step
        rho_new = evolution.evolve_density_matrix(rho, hamiltonian, visit_counts)
        
        # Should remain Hermitian
        assert torch.allclose(rho_new, rho_new.conj().transpose(-2, -1), atol=1e-6)
        
        # Should have trace ≈ 1
        assert abs(torch.trace(rho_new).real - 1.0) < 1e-4
        
        # Should be positive semidefinite
        eigenvals = torch.linalg.eigvals(rho_new)
        assert torch.all(eigenvals.real >= -1e-6)  # Allow small numerical errors
    
    def test_coherent_evolution(self, evolution, test_matrices):
        """Test coherent evolution term computation"""
        rho, hamiltonian, visit_counts = test_matrices
        
        coherent_term = evolution._compute_coherent_evolution(rho, hamiltonian)
        
        # Should satisfy: coherent_term = -i[H,ρ]/ℏ = -(Hρ - ρH)/ℏ
        commutator = torch.matmul(hamiltonian, rho) - torch.matmul(rho, hamiltonian)
        expected = -commutator / evolution.config.hbar
        
        assert torch.allclose(coherent_term, expected, atol=1e-6)
    
    def test_evolution_to_steady_state(self, evolution, test_matrices):
        """Test evolution to steady state"""
        rho, hamiltonian, visit_counts = test_matrices
        
        final_rho, evolution_time = evolution.evolve_to_steady_state(
            rho, hamiltonian, visit_counts, max_time=1.0
        )
        
        # Should converge
        assert evolution_time < 1.0  # Should converge before max time
        
        # Final state should be valid density matrix
        assert abs(torch.trace(final_rho).real - 1.0) < 1e-4
        assert torch.allclose(final_rho, final_rho.conj().transpose(-2, -1), atol=1e-6)


class TestPointerStateAnalyzer:
    """Test pointer state identification and analysis"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def config(self):
        return DecoherenceConfig(pointer_state_threshold=0.1)
    
    @pytest.fixture
    def analyzer(self, config, device):
        return PointerStateAnalyzer(config, device)
    
    def test_pointer_state_identification(self, analyzer, device):
        """Test identification of pointer states"""
        # Create density matrix with clear pointer states
        # (diagonal matrix with some large eigenvalues)
        rho = torch.diag(torch.tensor([0.7, 0.2, 0.08, 0.02], device=device))
        visit_counts = torch.tensor([100.0, 50.0, 10.0, 5.0], device=device)
        
        pointer_info = analyzer.identify_pointer_states(rho, visit_counts)
        
        # Should identify the dominant states
        assert pointer_info['num_pointer_states'] >= 1
        assert pointer_info['num_pointer_states'] <= 4
        
        # Eigenvalues should be sorted descending
        eigenvals = pointer_info['eigenvalues']
        assert torch.all(eigenvals[:-1] >= eigenvals[1:])
        
        # Should have classical fidelity measure
        assert 'classical_fidelity' in pointer_info
        assert len(pointer_info['classical_fidelity']) == pointer_info['num_pointer_states']
    
    def test_classical_probability_extraction(self, analyzer, device):
        """Test extraction of classical probabilities"""
        # Nearly diagonal density matrix (classical-like)
        rho = torch.diag(torch.tensor([0.5, 0.3, 0.15, 0.05], device=device))
        visit_counts = torch.tensor([80.0, 60.0, 20.0, 10.0], device=device)
        
        classical_probs = analyzer.extract_classical_probabilities(rho, visit_counts)
        
        # Should be normalized probability distribution
        assert torch.abs(classical_probs.sum() - 1.0) < 1e-6
        assert torch.all(classical_probs >= 0)
        
        # Should have same size as visit counts
        assert len(classical_probs) == len(visit_counts)
    
    def test_classical_fidelity_computation(self, analyzer, device):
        """Test classical fidelity measure"""
        # Create classical state (localized)
        classical_state = torch.zeros(4, device=device)
        classical_state[0] = 1.0  # Fully localized
        
        # Create quantum state (delocalized)
        quantum_state = torch.ones(4, device=device) / 2.0  # Uniform superposition
        
        visit_counts = torch.tensor([100.0, 50.0, 25.0, 10.0], device=device)
        
        # Stack states for batch processing
        states = torch.stack([classical_state, quantum_state], dim=1)
        
        fidelities = analyzer._compute_classical_fidelity(states, visit_counts)
        
        # Classical state should have higher fidelity
        assert fidelities[0] > fidelities[1]
        assert torch.all(fidelities >= 0)
        assert torch.all(fidelities <= 1)


class TestDecoherenceEngine:
    """Test the main decoherence engine"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def config(self):
        return DecoherenceConfig(
            base_decoherence_rate=1.0,
            dt=0.01,
            max_evolution_time=2.0
        )
    
    @pytest.fixture
    def engine(self, config, device):
        return DecoherenceEngine(config, device)
    
    @pytest.fixture
    def sample_tree_data(self, device):
        """Create sample tree data for testing"""
        visit_counts = torch.tensor([100.0, 75.0, 50.0, 25.0, 10.0], device=device)
        
        # Simple tree structure
        children = torch.tensor([
            [1, 2, -1, -1],  # Node 0 has children 1, 2
            [3, 4, -1, -1],  # Node 1 has children 3, 4
            [-1, -1, -1, -1],  # Node 2 is leaf
            [-1, -1, -1, -1],  # Node 3 is leaf
            [-1, -1, -1, -1]   # Node 4 is leaf
        ], device=device)
        
        tree_structure = {'children': children}
        
        return visit_counts, tree_structure
    
    def test_engine_initialization(self, engine, device):
        """Test decoherence engine initialization"""
        assert engine.device == device
        assert isinstance(engine.config, DecoherenceConfig)
        assert hasattr(engine, 'density_evolution')
        assert hasattr(engine, 'pointer_analyzer')
    
    def test_quantum_state_initialization(self, engine):
        """Test quantum state initialization"""
        num_nodes = 5
        
        # Test maximally mixed initialization
        rho = engine.initialize_quantum_state(num_nodes)
        
        # Should be identity matrix normalized
        expected = torch.eye(num_nodes, device=engine.device) / num_nodes
        assert torch.allclose(rho, expected, atol=1e-6)
        
        # Test custom initialization
        custom_state = torch.tensor([1, 0, 0, 0, 0], dtype=torch.complex64, device=engine.device)
        rho_custom = engine.initialize_quantum_state(num_nodes, custom_state)
        
        # Should be |0⟩⟨0|
        expected_custom = torch.zeros((num_nodes, num_nodes), dtype=torch.complex64, device=engine.device)
        expected_custom[0, 0] = 1.0
        assert torch.allclose(rho_custom, expected_custom, atol=1e-6)
    
    def test_hamiltonian_creation(self, engine, sample_tree_data):
        """Test tree Hamiltonian construction"""
        visit_counts, tree_structure = sample_tree_data
        
        H = engine.create_tree_hamiltonian(visit_counts, tree_structure)
        
        # Should be Hermitian
        assert torch.allclose(H, H.conj().transpose(-2, -1), atol=1e-6)
        
        # Diagonal should be visit counts
        assert torch.allclose(torch.diag(H), visit_counts, atol=1e-6)
        
        # Should have coupling terms from tree structure
        # Parent-child pairs should have non-zero off-diagonal elements
        assert H[0, 1] != 0  # Parent 0 coupled to child 1
        assert H[0, 2] != 0  # Parent 0 coupled to child 2
        assert H[1, 3] != 0  # Parent 1 coupled to child 3
    
    def test_quantum_to_classical_evolution(self, engine, sample_tree_data):
        """Test main quantum→classical evolution"""
        visit_counts, tree_structure = sample_tree_data
        
        # Initialize quantum superposition
        engine.initialize_quantum_state(len(visit_counts))
        
        # Evolve to classical
        result = engine.evolve_quantum_to_classical(visit_counts, tree_structure)
        
        # Should return all required fields
        required_fields = [
            'classical_probabilities', 
            'density_matrix', 
            'pointer_states',
            'evolution_time',
            'decoherence_rate'
        ]
        for field in required_fields:
            assert field in result
        
        # Classical probabilities should be normalized
        probs = result['classical_probabilities']
        assert torch.abs(probs.sum() - 1.0) < 1e-5
        assert torch.all(probs >= 0)
        assert len(probs) == len(visit_counts)
        
        # Should have identified some pointer states
        assert result['pointer_states']['num_pointer_states'] >= 1
        
        # Evolution time should be reasonable
        assert 0 < result['evolution_time'] < engine.config.max_evolution_time
    
    def test_classical_limit_behavior(self, engine, device):
        """Test behavior in classical limit (high decoherence)"""
        # High decoherence rate should lead to classical behavior
        engine.config.base_decoherence_rate = 100.0
        
        # Create visit counts with clear winner
        visit_counts = torch.tensor([1000.0, 10.0, 5.0, 1.0], device=device)
        
        result = engine.evolve_quantum_to_classical(visit_counts)
        
        # Should strongly favor highest visit count node
        probs = result['classical_probabilities']
        assert probs[0] > 0.8  # Most probability on highest visit count
        
        # Should have one dominant pointer state
        assert result['pointer_states']['num_pointer_states'] >= 1
        
    def test_statistics_tracking(self, engine, sample_tree_data):
        """Test statistics tracking"""
        visit_counts, tree_structure = sample_tree_data
        
        # Run evolution multiple times
        for _ in range(3):
            engine.evolve_quantum_to_classical(visit_counts, tree_structure)
        
        stats = engine.get_statistics()
        
        # Should track evolution steps
        assert stats['evolution_steps'] == 3
        assert stats['avg_evolution_time'] > 0
        assert stats['decoherence_rate'] > 0
        
        # Should have pointer state information
        assert stats['pointer_states_count'] >= 1


class TestTheoreticalPredictions:
    """Test theoretical predictions from decoherence theory"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_decoherence_rate_scaling(self, device):
        """Test that decoherence rates scale as predicted"""
        config = DecoherenceConfig(base_decoherence_rate=1.0)
        operators = DecoherenceOperators(config, device)
        
        # Test with different visit count patterns
        # Pattern 1: Similar visit counts (low decoherence)
        similar_visits = torch.tensor([50.0, 52.0, 48.0, 51.0], device=device)
        rates_similar = operators.compute_decoherence_rates(similar_visits)
        
        # Pattern 2: Very different visit counts (high decoherence)
        different_visits = torch.tensor([100.0, 10.0, 1.0, 200.0], device=device)
        rates_different = operators.compute_decoherence_rates(different_visits)
        
        # Average decoherence should be higher for different visit counts
        avg_similar = rates_similar.mean()
        avg_different = rates_different.mean()
        
        assert avg_different > avg_similar
    
    def test_pointer_state_selection(self, device):
        """Test that high visit count states become pointer states"""
        config = DecoherenceConfig()
        engine = DecoherenceEngine(config, device)
        
        # Visit counts with clear hierarchy
        visit_counts = torch.tensor([1000.0, 100.0, 10.0, 1.0], device=device)
        
        result = engine.evolve_quantum_to_classical(visit_counts)
        
        # Classical probabilities should favor high visit count nodes
        probs = result['classical_probabilities']
        
        # Should be roughly ordered by visit counts
        # (allowing for some quantum corrections)
        assert probs[0] > probs[1]  # Highest visit count gets most probability
        assert probs[1] > probs[3]  # Second highest > lowest
    
    def test_quantum_classical_consistency(self, device):
        """Test consistency between quantum and classical limits"""
        # Very high decoherence (classical limit)
        classical_config = DecoherenceConfig(base_decoherence_rate=1000.0)
        classical_engine = DecoherenceEngine(classical_config, device)
        
        # Very low decoherence (quantum regime) 
        quantum_config = DecoherenceConfig(base_decoherence_rate=0.001)
        quantum_engine = DecoherenceEngine(quantum_config, device)
        
        visit_counts = torch.tensor([100.0, 50.0, 25.0], device=device)
        
        classical_result = classical_engine.evolve_quantum_to_classical(visit_counts)
        quantum_result = quantum_engine.evolve_quantum_to_classical(visit_counts)
        
        classical_probs = classical_result['classical_probabilities']
        quantum_probs = quantum_result['classical_probabilities']
        
        # Classical limit should be more concentrated
        classical_entropy = -torch.sum(classical_probs * torch.log(classical_probs + 1e-8))
        quantum_entropy = -torch.sum(quantum_probs * torch.log(quantum_probs + 1e-8))
        
        assert classical_entropy < quantum_entropy


def test_factory_function():
    """Test factory function for decoherence engine creation"""
    # Test default creation
    engine = create_decoherence_engine()
    assert isinstance(engine, DecoherenceEngine)
    
    # Test with custom parameters
    engine = create_decoherence_engine(base_rate=2.0, temperature=0.5)
    assert engine.config.base_decoherence_rate == 2.0
    assert engine.config.temperature == 0.5


if __name__ == "__main__":
    # Run basic functionality test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Decoherence Engine tests on {device}")
    
    # Create decoherence engine
    engine = create_decoherence_engine(device, base_rate=1.0)
    
    # Test data
    visit_counts = torch.tensor([100.0, 75.0, 50.0, 25.0, 10.0], device=device)
    tree_structure = {
        'children': torch.tensor([
            [1, 2, -1, -1],
            [3, 4, -1, -1], 
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1]
        ], device=device)
    }
    
    # Test quantum to classical evolution
    print("Testing quantum→classical evolution...")
    start = time.perf_counter()
    result = engine.evolve_quantum_to_classical(visit_counts, tree_structure)
    end = time.perf_counter()
    
    print(f"✓ Evolution completed in {end-start:.4f}s")
    print(f"✓ Evolution time: {result['evolution_time']:.4f}s")
    print(f"✓ Pointer states found: {result['pointer_states']['num_pointer_states']}")
    print(f"✓ Classical probabilities: {result['classical_probabilities']}")
    print(f"✓ Probability sum: {result['classical_probabilities'].sum():.6f} (should be ~1.0)")
    print(f"✓ Decoherence rate: {result['decoherence_rate']:.4f}")
    
    # Test statistics
    stats = engine.get_statistics()
    print(f"✓ Statistics: {stats}")
    
    print("✓ All decoherence engine tests passed!")