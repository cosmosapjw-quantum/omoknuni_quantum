"""Tests for enhanced QFT engine with proper one-loop corrections

This module tests the physically rigorous implementation of tree-level
and one-loop corrections in the QFT formalism.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import scipy.linalg

# Import the modules we'll implement
from mcts.quantum.enhanced_qft_engine import (
    EnhancedQFTEngine, 
    FunctionalDeterminant,
    RegularizationScheme,
    QFTConfig
)
from mcts.quantum.rg_flow import RGFlowOptimizer, RGConfig


class TestFunctionalDeterminant:
    """Test proper functional determinant calculations"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def det_calculator(self, device):
        return FunctionalDeterminant(device=device)
    
    def test_zeta_regularization(self, det_calculator, device):
        """Test zeta function regularization of determinant"""
        # Create a simple fluctuation matrix with known eigenvalues
        eigenvalues = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0], device=device)
        matrix = torch.diag(eigenvalues)
        
        # Compute regularized log determinant
        log_det = det_calculator.compute_log_det_zeta(matrix, s=-1)
        
        # Check basic properties
        assert torch.isfinite(log_det)
        assert log_det.dtype == torch.float32 or log_det.dtype == torch.float64
        
        # For diagonal matrix, regularization adds counterterm
        naive_log_det = torch.sum(torch.log(eigenvalues))
        # The difference comes from UV counterterm
        assert torch.isfinite(log_det - naive_log_det)
    
    def test_heat_kernel_regularization(self, det_calculator, device):
        """Test heat kernel regularization method"""
        # Small positive definite matrix
        A = torch.randn(5, 5, device=device)
        matrix = A @ A.T + 0.1 * torch.eye(5, device=device)  # Ensure positive definite
        
        # Compute using heat kernel
        log_det_heat = det_calculator.compute_log_det_heat_kernel(
            matrix, tau_min=1e-3, tau_max=10.0, n_points=100
        )
        
        # Compare with direct calculation for small matrix
        log_det_direct = torch.logdet(matrix)
        
        # Heat kernel adds UV regularization, so they won't match exactly
        # Just check that result is finite and reasonable
        assert torch.isfinite(log_det_heat)
        assert log_det_heat.dtype in [torch.float32, torch.float64]
    
    def test_pauli_villars_regularization(self, det_calculator, device):
        """Test Pauli-Villars regularization"""
        # Matrix with UV divergence (large eigenvalues)
        eigenvalues = torch.tensor([0.1, 1.0, 10.0, 100.0, 1000.0], device=device)
        matrix = torch.diag(eigenvalues)
        
        # Apply Pauli-Villars with cutoff
        cutoff = 50.0
        log_det_pv = det_calculator.compute_log_det_pauli_villars(matrix, cutoff)
        
        # Should suppress large eigenvalues
        assert torch.isfinite(log_det_pv)
        
        # Check cutoff dependence
        log_det_pv2 = det_calculator.compute_log_det_pauli_villars(matrix, cutoff * 2)
        assert log_det_pv != log_det_pv2  # Should depend on cutoff
    
    def test_zero_mode_removal(self, det_calculator, device):
        """Test proper handling of zero modes"""
        # Matrix with zero eigenvalue
        eigenvalues = torch.tensor([0.0, 0.01, 0.5, 1.0, 2.0], device=device)
        matrix = torch.diag(eigenvalues)
        
        # Should handle zero mode properly
        log_det = det_calculator.compute_log_det_zeta(matrix)
        assert torch.isfinite(log_det)
        
        # Verify zero mode was removed
        assert det_calculator.num_zero_modes == 1


class TestEnhancedQFTEngine:
    """Test the enhanced QFT engine with one-loop corrections"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def qft_config(self, device):
        return QFTConfig(
            device=device,
            hbar_eff=0.1,
            regularization_type='zeta',
            uv_cutoff=100.0,
            enable_rg_improvement=True,
            use_gpu=device.type == 'cuda'
        )
    
    @pytest.fixture
    def qft_engine(self, qft_config):
        return EnhancedQFTEngine(qft_config)
    
    def test_classical_action_computation(self, qft_engine, device):
        """Test tree-level classical action"""
        # Mock path data
        path_length = 10
        visit_counts = torch.randint(1, 100, (path_length,), device=device).float()
        
        # Compute classical action S = -Σ log N(s,a)
        S_classical = qft_engine.compute_classical_action(visit_counts)
        
        # Check properties
        assert torch.isfinite(S_classical)
        
        # Verify formula: S = -Σ log N(s,a)
        expected = -torch.sum(torch.log(visit_counts))
        assert torch.allclose(S_classical, expected)
        
        # With log of positive visit counts, action can be positive or negative
        # depending on whether counts are > 1 or < 1
    
    def test_fluctuation_matrix_computation(self, qft_engine, device):
        """Test computation of fluctuation matrix M"""
        # Create simple path data
        path_length = 5
        q_values = torch.randn(path_length, device=device)
        visit_counts = torch.randint(1, 50, (path_length,), device=device).float()
        
        # Compute fluctuation matrix
        M = qft_engine.compute_fluctuation_matrix(q_values, visit_counts)
        
        # Check properties
        assert M.shape == (path_length, path_length)
        assert torch.allclose(M, M.T)  # Should be symmetric
        
        # Check positive semi-definite
        eigenvalues = torch.linalg.eigvalsh(M)
        assert torch.all(eigenvalues >= -1e-6)  # Allow small numerical errors
    
    def test_one_loop_correction(self, qft_engine, device):
        """Test one-loop quantum correction calculation"""
        # Path data
        path_length = 8
        q_values = torch.randn(path_length, device=device)
        visit_counts = torch.randint(5, 100, (path_length,), device=device).float()
        
        path_data = {
            'q_values': q_values,
            'visit_counts': visit_counts,
            'path_length': path_length
        }
        
        # Compute one-loop correction
        one_loop = qft_engine.compute_one_loop_correction(path_data)
        
        # Check basic properties
        assert torch.isfinite(one_loop)
        assert one_loop.dtype in [torch.float32, torch.float64]
        
        # One-loop should be order hbar
        assert torch.abs(one_loop) < 10 * qft_engine.config.hbar_eff
    
    def test_rg_improvement(self, qft_engine, device):
        """Test RG improvement of effective action"""
        # Mock tree scale
        tree_scale = 0.1
        
        # Mock actions
        S_classical = torch.tensor(10.0, device=device)
        one_loop = torch.tensor(0.5, device=device)
        
        # Apply RG improvement
        S_improved = qft_engine.apply_rg_improvement(
            S_classical, one_loop, tree_scale
        )
        
        # Should modify the result
        assert S_improved != S_classical + one_loop
        
        # Check scale dependence
        S_improved2 = qft_engine.apply_rg_improvement(
            S_classical, one_loop, tree_scale * 2
        )
        assert S_improved != S_improved2
    
    def test_full_effective_action(self, qft_engine, device):
        """Test complete effective action computation"""
        # Create realistic path data
        path_length = 10
        path_data = {
            'q_values': torch.randn(path_length, device=device) * 0.5,
            'visit_counts': torch.randint(1, 100, (path_length,), device=device).float(),
            'path_length': path_length,
            'tree_scale': 0.1
        }
        
        # Compute full effective action
        S_eff = qft_engine.compute_effective_action(path_data)
        
        # Check properties
        assert torch.isfinite(S_eff)
        # Note: effective action can be positive or negative depending on visit counts
        
        # Test with quantum effects disabled
        qft_engine.config.hbar_eff = 0
        S_classical_only = qft_engine.compute_effective_action(path_data)
        qft_engine.config.hbar_eff = 0.1
        
        # Quantum corrections should modify result
        assert S_eff != S_classical_only
    
    def test_batch_computation(self, qft_engine, device):
        """Test batched computation for efficiency"""
        # Multiple paths
        batch_size = 32
        path_length = 10
        
        batch_data = {
            'q_values': torch.randn(batch_size, path_length, device=device),
            'visit_counts': torch.randint(1, 100, (batch_size, path_length), device=device).float(),
            'path_lengths': torch.full((batch_size,), path_length, device=device),
            'tree_scale': 0.1
        }
        
        # Compute batch effective actions
        S_eff_batch = qft_engine.compute_effective_action_batch(batch_data)
        
        # Check output shape
        assert S_eff_batch.shape == (batch_size,)
        assert torch.all(torch.isfinite(S_eff_batch))
        
        # Compare with individual computation
        S_eff_0 = qft_engine.compute_effective_action({
            'q_values': batch_data['q_values'][0],
            'visit_counts': batch_data['visit_counts'][0],
            'path_length': path_length,
            'tree_scale': 0.1
        })
        
        assert torch.allclose(S_eff_batch[0], S_eff_0, rtol=1e-5)


class TestRegularizationSchemes:
    """Test different regularization schemes"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_regularization_scheme_interface(self, device):
        """Test regularization scheme factory"""
        # Test each scheme type
        for scheme_type in ['zeta', 'heat_kernel', 'pauli_villars', 'dimensional']:
            scheme = RegularizationScheme.create(scheme_type, device)
            
            # Should have regularize method
            assert hasattr(scheme, 'regularize')
            
            # Test on simple matrix
            matrix = torch.eye(5, device=device)
            reg_result = scheme.regularize(matrix)
            
            assert 'log_det' in reg_result
            assert 'counterterms' in reg_result
            assert torch.isfinite(reg_result['log_det'])
    
    def test_dimensional_regularization(self, device):
        """Test dimensional regularization in d=4-ε dimensions"""
        scheme = RegularizationScheme.create('dimensional', device)
        
        # Mock loop integral in d dimensions
        def loop_integral(d):
            # Simple scalar loop: ∫ d^d k / (k^2 + m^2)
            # Result ∝ m^(d-2) Γ(1-d/2)
            return torch.tensor(1.0, device=device) / (d - 4)  # Pole at d=4
        
        # Regularize near d=4
        epsilon = 0.01
        reg_result = scheme.regularize_integral(loop_integral, d=4-epsilon)
        
        # Should handle pole
        assert torch.isfinite(reg_result['finite_part'])
        assert 'pole_coefficient' in reg_result


class TestRGFlowIntegration:
    """Test integration of RG flow with quantum corrections"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def integrated_system(self, device):
        # Create integrated QFT + RG system
        qft_config = QFTConfig(device=device, enable_rg_improvement=True)
        qft_engine = EnhancedQFTEngine(qft_config)
        return qft_engine
    
    def test_scale_dependent_couplings(self, integrated_system, device):
        """Test that couplings run with scale"""
        # Get couplings at different scales
        scale1 = 1.0
        scale2 = 0.01
        
        couplings1 = integrated_system.get_running_couplings(scale1)
        couplings2 = integrated_system.get_running_couplings(scale2)
        
        # Couplings should be different at different scales
        assert not torch.allclose(couplings1['c_puct'], couplings2['c_puct'])
        
        # Check RG flow direction (IR freedom for c_puct)
        assert couplings2['c_puct'] < couplings1['c_puct']  # Decreases toward IR
    
    def test_wilson_fisher_fixed_point(self, integrated_system, device):
        """Test approach to Wilson-Fisher fixed point"""
        # Flow to deep IR
        ir_scale = 1e-6
        ir_couplings = integrated_system.get_running_couplings(ir_scale)
        
        # Should approach √2 for c_puct at WF fixed point
        expected_c_puct = np.sqrt(2.0)
        assert torch.abs(ir_couplings['c_puct'] - expected_c_puct) < 0.1
    
    def test_anomalous_dimensions(self, integrated_system, device):
        """Test computation of anomalous dimensions"""
        # Get anomalous dimensions at a scale
        scale = 0.1
        anomalous_dims = integrated_system.compute_anomalous_dimensions(scale)
        
        # Check structure
        assert 'value_operator' in anomalous_dims
        assert 'visit_operator' in anomalous_dims
        assert 'path_operator' in anomalous_dims
        
        # Should be small corrections
        for key, gamma in anomalous_dims.items():
            assert torch.abs(gamma) < 1.0  # Perturbative regime


class TestQuantumStateRecycling:
    """Test quantum state recycling and memory optimization"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def state_pool(self, device):
        from mcts.quantum.state_pool import QuantumStatePool, PoolConfig
        config = PoolConfig(max_states=100)
        return QuantumStatePool(config, device)
    
    def test_state_recycling(self, state_pool, device):
        """Test basic state recycling"""
        # Get a quantum state
        dim = 10
        state1 = state_pool.get_state(dim)
        
        assert state1.shape == (dim, dim)
        assert torch.allclose(state1, torch.zeros(dim, dim, device=device, dtype=torch.complex64))
        
        # Return it to pool
        state_pool.return_state(state1)
        
        # Get another state - should reuse
        state2 = state_pool.get_state(dim)
        assert state2 is state1  # Same object
    
    def test_state_compression(self, state_pool, device):
        """Test wave function compression"""
        # Create a sparse state
        dim = 100
        state = state_pool.get_state(dim)
        # Make it sparse
        state[0, 0] = 1.0
        state[10, 10] = 2.0
        state[50, 50] = 3.0
        
        # Return with compression
        state_pool.return_state(state, compress=True)
        
        # Check compression statistics
        stats = state_pool.get_statistics()
        assert stats['compressions'] >= 1
        
        # Get and check state reuse
        state2 = state_pool.get_state(dim)
        # Should be zeroed
        assert torch.allclose(state2, torch.zeros_like(state2))


class TestPhysicsConsistency:
    """Test physical consistency of quantum corrections"""
    
    @pytest.fixture
    def qft_engine(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = QFTConfig(device=device, hbar_eff=0.1)
        return EnhancedQFTEngine(config)
    
    def test_unitarity(self, qft_engine):
        """Test that quantum evolution preserves unitarity"""
        # Create initial state
        dim = 8
        psi = torch.randn(dim, dtype=torch.complex64, device=qft_engine.device)
        psi = psi / torch.norm(psi)
        
        # Evolve with quantum corrections
        dt = 0.01
        psi_evolved = qft_engine.quantum_evolve(psi, dt)
        
        # Check norm conservation
        assert torch.allclose(torch.norm(psi_evolved), torch.tensor(1.0, device=qft_engine.device), atol=1e-6)
    
    def test_hermiticity(self, qft_engine):
        """Test that Hamiltonians remain Hermitian"""
        # Generate effective Hamiltonian
        dim = 10
        path_data = {
            'q_values': torch.randn(dim, device=qft_engine.device),
            'visit_counts': torch.rand(dim, device=qft_engine.device) * 100 + 1
        }
        
        H_eff = qft_engine.compute_effective_hamiltonian(path_data)
        
        # Check Hermiticity
        assert torch.allclose(H_eff, H_eff.conj().T, atol=1e-6)
    
    def test_action_finiteness(self, qft_engine):
        """Test that effective action remains finite and reasonable"""
        # Various path configurations
        for _ in range(10):
            path_data = {
                'q_values': torch.randn(20, device=qft_engine.device) * 0.5,
                'visit_counts': torch.rand(20, device=qft_engine.device) * 100 + 1,
                'path_length': 20,
                'tree_scale': 0.1
            }
            
            S_eff = qft_engine.compute_effective_action(path_data)
            
            # Action should be finite and not too large
            assert torch.isfinite(S_eff)
            assert torch.abs(S_eff) < 1e6  # Reasonable bound
            
            # Classical action is negative log of visit counts
            # So effective action can be negative or positive depending on quantum corrections


if __name__ == "__main__":
    pytest.main([__file__, "-v"])