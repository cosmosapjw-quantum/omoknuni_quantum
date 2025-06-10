"""Tests for enhanced Lindblad master equation implementation"""

import pytest
import torch
import numpy as np
from typing import Dict, List

from mcts.quantum.enhanced_lindblad import (
    EnhancedLindbladEvolution,
    SUNOperatorBasis,
    AdaptiveDecoherenceRates,
    QuantumClassicalTransition,
    LindbladeConfig,
    create_enhanced_lindblad_evolution
)


class TestSUNOperatorBasis:
    """Test SU(N) operator basis generation"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def sun_basis(self, device):
        config = LindbladeConfig(use_gpu_kernels=device.type == 'cuda')
        return SUNOperatorBasis(dimension=4, config=config)
    
    def test_basis_completeness(self, sun_basis):
        """Test that SU(N) basis has correct number of operators"""
        n = sun_basis.dimension
        expected_operators = n**2 - 1  # SU(N) has N²-1 generators
        
        assert len(sun_basis.basis_operators) == expected_operators
        
    def test_basis_orthogonality(self, sun_basis):
        """Test orthogonality: Tr(λ_i λ_j) = 2δ_ij"""
        basis = sun_basis.basis_operators
        n_ops = len(basis)
        
        # The normalization should give Tr(λ_i λ_j) = 2δ_ij
        # but numerical errors can accumulate. Check relative orthogonality
        for i in range(n_ops):
            for j in range(n_ops):
                trace = torch.trace(torch.matmul(basis[i].conj().T, basis[j]))
                
                if i == j:
                    # Diagonal should be close to 2
                    assert torch.abs(trace.real - 2.0) < 0.01  # Within 0.5% 
                else:
                    # Off-diagonal should be close to 0
                    assert torch.abs(trace.real) < 1e-4
                    
                assert torch.abs(trace.imag) < 1e-5
    
    def test_basis_traceless(self, sun_basis):
        """Test that all basis operators are traceless"""
        for op in sun_basis.basis_operators:
            trace = torch.trace(op)
            assert torch.abs(trace) < 1e-6
    
    def test_basis_hermitian(self, sun_basis):
        """Test that basis operators are Hermitian"""
        for op in sun_basis.basis_operators:
            diff = op - op.conj().T
            assert torch.norm(diff) < 1e-6
    
    def test_structure_constants(self, sun_basis):
        """Test SU(N) structure constants satisfy Jacobi identity"""
        f = sun_basis.get_structure_constants()
        n = f.shape[0]
        
        # Check antisymmetry: f_ijk = -f_jik
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    assert torch.allclose(f[i,j,k], -f[j,i,k], atol=1e-5)
        
        # Jacobi identity: f_ijl f_lkm + f_jkl f_lim + f_kil f_ljm = 0
        # (Check a few random combinations due to computational cost)
        for _ in range(10):
            i, j, k, m = torch.randint(0, n, (4,))
            jacobi = torch.zeros(1, device=f.device)
            
            for l in range(n):
                jacobi += f[i,j,l] * f[l,k,m] + f[j,k,l] * f[l,i,m] + f[k,i,l] * f[l,j,m]
            
            assert torch.abs(jacobi) < 1e-4
    
    def test_operator_expansion(self, sun_basis):
        """Test expanding and reconstructing operators"""
        # Create a random Hermitian operator
        n = sun_basis.dimension
        H_real = torch.randn(n, n, device=sun_basis.device)
        H = (H_real + H_real.T).to(torch.complex64)  # Make real symmetric (Hermitian)
        
        # Remove trace (project to SU(N))
        trace_H = torch.trace(H)
        H = H - (trace_H / n) * torch.eye(n, device=sun_basis.device, dtype=torch.complex64)
        
        # Verify traceless
        assert torch.abs(torch.trace(H)) < 1e-6
        
        # Expand in basis
        coeffs = sun_basis.expand_operator(H)
        
        # Reconstruct
        H_reconstructed = sun_basis.reconstruct_operator(coeffs)
        
        # Check reconstruction accuracy
        reconstruction_error = torch.norm(H - H_reconstructed) / torch.norm(H)
        assert reconstruction_error < 0.01, f"Reconstruction error: {reconstruction_error}"


class TestAdaptiveDecoherenceRates:
    """Test adaptive decoherence rate computation"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def adaptive_rates(self, device):
        config = LindbladeConfig(
            enable_adaptive_rates=True,
            min_decoherence_rate=0.01,
            max_decoherence_rate=10.0
        )
        return AdaptiveDecoherenceRates(config, device)
    
    def test_rate_bounds(self, adaptive_rates, device):
        """Test that rates stay within configured bounds"""
        # Create test states
        n = 4
        rho = torch.eye(n, device=device) / n
        H_int = torch.randn(n, n, device=device)
        H_int = H_int + H_int.T
        
        bath_state = {'temperature': 1.0}
        
        rates = adaptive_rates.compute_adaptive_rates(rho, bath_state, H_int)
        
        # Check bounds
        assert torch.all(rates >= adaptive_rates.config.min_decoherence_rate)
        assert torch.all(rates <= adaptive_rates.config.max_decoherence_rate)
    
    def test_temperature_dependence(self, adaptive_rates, device):
        """Test that rates depend on temperature correctly"""
        n = 4
        rho = torch.eye(n, device=device) / n
        H_int = torch.eye(n, device=device)
        
        # Low temperature
        bath_low_T = {'temperature': 0.1}
        rates_low = adaptive_rates.compute_adaptive_rates(rho, bath_low_T, H_int)
        
        # High temperature
        bath_high_T = {'temperature': 10.0}
        rates_high = adaptive_rates.compute_adaptive_rates(rho, bath_high_T, H_int)
        
        # High temperature should generally have higher rates
        assert torch.mean(rates_high) > torch.mean(rates_low)
    
    def test_entanglement_dependence(self, adaptive_rates, device):
        """Test rates depend on system-bath entanglement"""
        n = 4
        H_int = torch.eye(n, device=device)
        bath_state = {'temperature': 1.0}
        
        # Pure state (low entanglement)
        rho_pure = torch.zeros(n, n, device=device)
        rho_pure[0, 0] = 1.0
        rates_pure = adaptive_rates.compute_adaptive_rates(rho_pure, bath_state, H_int)
        
        # Mixed state (high entanglement)
        rho_mixed = torch.eye(n, device=device) / n
        rates_mixed = adaptive_rates.compute_adaptive_rates(rho_mixed, bath_state, H_int)
        
        # Mixed state should have higher decoherence
        assert torch.mean(rates_mixed) > torch.mean(rates_pure)
    
    def test_rate_smoothing(self, adaptive_rates, device):
        """Test that rates are smoothly adapted"""
        n = 4
        rho = torch.eye(n, device=device) / n
        H_int = torch.randn(n, n, device=device)
        H_int = H_int + H_int.T
        bath_state = {'temperature': 1.0}
        
        # First call
        rates1 = adaptive_rates.compute_adaptive_rates(rho, bath_state, H_int)
        
        # Second call with slightly different state
        rho2 = rho + 0.01 * torch.randn(n, n, device=device)
        rho2 = 0.5 * (rho2 + rho2.conj().T)  # Hermitian
        rho2 = rho2 / torch.trace(rho2)  # Normalize
        
        rates2 = adaptive_rates.compute_adaptive_rates(rho2, bath_state, H_int)
        
        # Rates should not change drastically
        relative_change = torch.norm(rates2 - rates1) / torch.norm(rates1)
        assert relative_change < 0.5  # Less than 50% change


class TestQuantumClassicalTransition:
    """Test quantum-to-classical transition detection"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def transition_detector(self, device):
        config = LindbladeConfig(
            classicality_threshold=0.95,
            purity_threshold=0.99,
            entropy_threshold=0.01
        )
        return QuantumClassicalTransition(config, device)
    
    def test_pure_state_classical(self, transition_detector, device):
        """Test that pure states are detected as classical"""
        n = 4
        
        # Pure state
        rho = torch.zeros(n, n, device=device, dtype=torch.complex64)
        rho[0, 0] = 1.0
        
        result = transition_detector.check_classicality(rho)
        
        assert result['is_classical']
        assert result['purity'] > 0.99
        assert result['entropy'] < 0.01
        assert result['diagonality'] > 0.99
    
    def test_mixed_state_quantum(self, transition_detector, device):
        """Test that maximally mixed states are quantum"""
        n = 4
        
        # Maximally mixed
        rho = torch.eye(n, device=device, dtype=torch.complex64) / n
        
        result = transition_detector.check_classicality(rho)
        
        assert not result['is_classical']
        assert result['purity'] < 0.5
        assert result['entropy'] > 1.0
    
    def test_superposition_quantum(self, transition_detector, device):
        """Test that coherent superpositions are quantum"""
        n = 4
        
        # Superposition state |ψ⟩ = (|0⟩ + |1⟩)/√2
        psi = torch.zeros(n, device=device, dtype=torch.complex64)
        psi[0] = 1.0 / np.sqrt(2)
        psi[1] = 1.0 / np.sqrt(2)
        
        rho = torch.outer(psi, psi.conj())
        
        result = transition_detector.check_classicality(rho)
        
        # Pure but not diagonal
        assert not result['is_classical']
        assert result['purity'] > 0.99
        assert result['diagonality'] < 0.9
    
    def test_pointer_state_projection(self, transition_detector, device):
        """Test pointer state projection calculation"""
        n = 4
        
        # Diagonal state (classical)
        rho = torch.diag(torch.tensor([0.5, 0.3, 0.2, 0.0], device=device, dtype=torch.complex64))
        
        # Pointer states = computational basis
        pointer_states = torch.eye(n, device=device, dtype=torch.complex64)
        
        result = transition_detector.check_classicality(rho, pointer_states)
        
        assert result['is_classical']
        assert result['pointer_projection'] > 0.99


class TestEnhancedLindbladEvolution:
    """Test full enhanced Lindblad evolution"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def lindblad_evolution(self, device):
        return create_enhanced_lindblad_evolution(
            device=device,
            use_full_basis=True,
            enable_adaptive=True,
            dt=0.01
        )
    
    def test_evolution_preserves_trace(self, lindblad_evolution, device):
        """Test that evolution preserves trace"""
        n = 4
        
        # Initial state
        rho = torch.eye(n, device=device, dtype=torch.complex64) / n
        
        # Hamiltonian
        H = torch.randn(n, n, device=device)
        H = H + H.T
        H = H.to(torch.complex64)
        
        # Environment
        env_state = {'temperature': 1.0}
        
        # Evolve
        rho_evolved, info = lindblad_evolution.evolve(rho, H, env_state)
        
        # Check trace preservation
        assert torch.allclose(torch.trace(rho_evolved), torch.tensor(1.0, device=device), atol=1e-5)
    
    def test_evolution_preserves_hermiticity(self, lindblad_evolution, device):
        """Test that evolution preserves Hermiticity"""
        n = 4
        
        rho = torch.eye(n, device=device, dtype=torch.complex64) / n
        H = torch.randn(n, n, device=device)
        H = (H + H.T).to(torch.complex64)
        env_state = {'temperature': 1.0}
        
        rho_evolved, _ = lindblad_evolution.evolve(rho, H, env_state)
        
        # Check Hermiticity
        diff = rho_evolved - rho_evolved.conj().T
        assert torch.norm(diff) < 1e-5
    
    def test_evolution_preserves_positivity(self, lindblad_evolution, device):
        """Test that evolution preserves positive semidefiniteness"""
        n = 4
        
        rho = torch.eye(n, device=device, dtype=torch.complex64) / n
        H = torch.randn(n, n, device=device)
        H = (H + H.T).to(torch.complex64)
        env_state = {'temperature': 1.0}
        
        # Evolve for multiple steps
        for _ in range(10):
            rho, _ = lindblad_evolution.evolve(rho, H, env_state)
        
        # Check positive semidefinite
        eigenvals = torch.linalg.eigvalsh(rho)
        assert torch.all(eigenvals > -1e-6)
    
    def test_decoherence_to_classical(self, lindblad_evolution, device):
        """Test that decoherence leads to classical states"""
        n = 4
        
        # Start with superposition
        psi = torch.ones(n, device=device, dtype=torch.complex64) / np.sqrt(n)
        rho = torch.outer(psi, psi.conj())
        
        # Energy basis Hamiltonian
        H = torch.diag(torch.arange(n, device=device, dtype=torch.float32))
        H = H.to(torch.complex64)
        
        env_state = {'temperature': 0.1}  # Low temperature for pointer states
        
        # Evolve to equilibrium
        rho_final, info = lindblad_evolution.evolve_to_equilibrium(
            rho, H, env_state, max_time=10.0
        )
        
        # Should become more classical
        assert info['evolution_history'][-1]['classicality'] > info['evolution_history'][0]['classicality']
        assert info['evolution_history'][-1]['entropy'] < info['evolution_history'][0]['entropy']
    
    def test_thermal_equilibrium(self, lindblad_evolution, device):
        """Test approach to thermal equilibrium"""
        n = 4
        
        # Start far from equilibrium
        rho = torch.zeros(n, n, device=device, dtype=torch.complex64)
        rho[0, 0] = 1.0
        
        # Simple Hamiltonian
        H = torch.diag(torch.arange(n, device=device, dtype=torch.float32))
        H = H.to(torch.complex64)
        
        temperature = 1.0
        env_state = {'temperature': temperature}
        
        # Evolve to equilibrium
        rho_final, info = lindblad_evolution.evolve_to_equilibrium(
            rho, H, env_state, max_time=20.0
        )
        
        # Check if close to thermal state
        # ρ_thermal ∝ exp(-H/T)
        eigenvals_H = torch.diag(H).real
        thermal_pops = torch.exp(-eigenvals_H / temperature)
        thermal_pops = thermal_pops / torch.sum(thermal_pops)
        
        final_pops = torch.diag(rho_final).real
        
        # Should be close to thermal distribution
        assert torch.allclose(final_pops, thermal_pops, atol=0.1)
    
    def test_pointer_basis_computation(self, lindblad_evolution, device):
        """Test pointer basis calculation"""
        n = 4
        
        lindblad_evolution.initialize_sun_basis(n)
        
        # Hamiltonian
        H = torch.diag(torch.arange(n, device=device, dtype=torch.float32))
        H = H.to(torch.complex64)
        
        # Get some Lindblad operators
        lindblad_ops = lindblad_evolution.sun_basis.basis_operators[:5]
        
        # Compute pointer basis
        pointer_basis = lindblad_evolution.compute_pointer_basis(H, lindblad_ops)
        
        # Should be unitary (orthonormal columns)
        should_be_identity = torch.matmul(pointer_basis.conj().T, pointer_basis)
        assert torch.allclose(should_be_identity, torch.eye(n, device=device, dtype=torch.complex64), atol=1e-5)


class TestIntegration:
    """Integration tests for complete workflow"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_full_quantum_to_classical_transition(self, device):
        """Test complete quantum to classical transition"""
        n = 8
        
        # Create evolution engine
        evolution = create_enhanced_lindblad_evolution(
            device=device,
            use_full_basis=True,
            enable_adaptive=True,
            dt=0.01,
            basis_truncation=20  # Use subset of basis for speed
        )
        
        # Initial superposition state
        psi = torch.ones(n, device=device, dtype=torch.complex64) / np.sqrt(n)
        rho_init = torch.outer(psi, psi.conj())
        
        # Hamiltonian with non-degenerate spectrum
        H = torch.diag(torch.arange(n, device=device, dtype=torch.float32) ** 2)
        H = H.to(torch.complex64)
        
        # Environment
        env_state = {
            'temperature': 0.5,
            'correlation_time': 0.01
        }
        
        # Evolve
        rho_final, info = evolution.evolve_to_equilibrium(
            rho_init, H, env_state, max_time=5.0
        )
        
        # Verify evolution
        assert info['final_time'] > 0
        assert len(info['evolution_history']) > 0
        
        # Check physical properties
        assert torch.allclose(torch.trace(rho_final), torch.tensor(1.0, device=device), atol=1e-5)
        eigenvals = torch.linalg.eigvalsh(rho_final)
        assert torch.all(eigenvals > -1e-6)
        
        # Should have increased classicality
        initial_classicality = info['evolution_history'][0]['classicality']
        final_classicality = info['evolution_history'][-1]['classicality']
        assert final_classicality > initial_classicality
    
    def test_performance_scaling(self, device):
        """Test performance with different system sizes"""
        import time
        
        sizes = [4, 8, 16]
        times = []
        
        for n in sizes:
            evolution = create_enhanced_lindblad_evolution(
                device=device,
                use_full_basis=False,  # Use truncated basis for larger systems
                basis_truncation=min(n**2 - 1, 20)
            )
            
            rho = torch.eye(n, device=device, dtype=torch.complex64) / n
            H = torch.randn(n, n, device=device)
            H = (H + H.T).to(torch.complex64)
            env_state = {'temperature': 1.0}
            
            # Time single evolution step
            start = time.perf_counter()
            evolution.evolve(rho, H, env_state)
            end = time.perf_counter()
            
            times.append(end - start)
        
        # Check reasonable scaling (should be roughly O(n²) to O(n³))
        # Just verify it doesn't explode
        assert times[-1] / times[0] < 100  # Less than 100x slowdown for 4x size increase


if __name__ == "__main__":
    pytest.main([__file__, "-v"])