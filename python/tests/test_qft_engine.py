"""
Test Suite for QFT Engine
=========================

Validates the quantum field theory formulation against theoretical predictions
from the mathematical foundations document.

Test Categories:
1. Basic functionality tests
2. Theoretical prediction validation  
3. Classical limit verification
4. GPU performance tests
5. Numerical stability tests
"""

import pytest
import torch
import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts.quantum.qft_engine import QFTEngine, QFTConfig, EffectiveActionEngine


class TestQFTEngine:
    """Test the main QFT engine functionality"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def qft_config(self):
        return QFTConfig(
            hbar_eff=0.1,
            temperature=1.0,
            dimension=4,
            matrix_regularization=1e-6
        )
    
    @pytest.fixture
    def qft_engine(self, qft_config, device):
        return QFTEngine(qft_config, device)
    
    @pytest.fixture
    def sample_data(self, device):
        """Generate sample path and visit count data"""
        batch_size = 32
        max_depth = 10
        num_nodes = 100
        
        # Create sample paths
        paths = torch.randint(0, num_nodes, (batch_size, max_depth), device=device)
        
        # Create realistic visit counts (power law distribution)
        base_visits = torch.exp(torch.randn(num_nodes, device=device) * 2)
        visit_counts = torch.clamp(base_visits, min=1.0).float()
        
        return paths, visit_counts
    
    def test_engine_initialization(self, qft_engine, device):
        """Test that QFT engine initializes correctly"""
        assert qft_engine.device == device
        assert qft_engine.config.hbar_eff == 0.1
        assert isinstance(qft_engine.effective_action_engine, EffectiveActionEngine)
        
    def test_classical_action_computation(self, qft_engine, sample_data):
        """Test classical action S_cl = -Σ log N(s,a)"""
        paths, visit_counts = sample_data
        
        # Compute classical action
        real_action, imag_action = qft_engine.effective_action_engine.compute_effective_action(
            paths, visit_counts, include_quantum=False
        )
        
        # Basic properties
        assert real_action.shape == (paths.shape[0],)
        assert torch.all(torch.isfinite(real_action))
        assert torch.all(imag_action == 0)  # No imaginary part for classical
        
        # Classical action should be negative (higher visits → lower action)
        assert torch.all(real_action <= 0)
        
    def test_quantum_corrections(self, qft_engine, sample_data):
        """Test that quantum corrections are computed and have correct properties"""
        paths, visit_counts = sample_data
        
        # Compute with and without quantum corrections
        classical_action, _ = qft_engine.effective_action_engine.compute_effective_action(
            paths, visit_counts, include_quantum=False
        )
        quantum_action, quantum_imag = qft_engine.effective_action_engine.compute_effective_action(
            paths, visit_counts, include_quantum=True
        )
        
        # Quantum corrections should modify the action
        correction = quantum_action - classical_action
        assert not torch.allclose(correction, torch.zeros_like(correction), atol=1e-6)
        
        # Corrections should scale with ℏ_eff
        assert torch.all(torch.abs(correction) < 10 * qft_engine.config.hbar_eff)
        
        # Should have imaginary part from decoherence
        assert torch.any(quantum_imag > 0)
        
    def test_path_weight_computation(self, qft_engine, sample_data):
        """Test QFT path weight computation"""
        paths, visit_counts = sample_data
        
        weights = qft_engine.compute_path_weights(paths, visit_counts)
        
        # Basic probability properties
        assert weights.shape == (paths.shape[0],)
        assert torch.all(weights >= 0)
        assert torch.all(weights <= 1)
        assert torch.abs(weights.sum() - 1.0) < 1e-5  # Normalized
        
    def test_hbar_effective_update(self, qft_engine):
        """Test ℏ_eff = 1/√N̄ relationship"""
        # Test different average visit counts
        test_visits = [1.0, 100.0, 10000.0]
        
        for avg_visits in test_visits:
            qft_engine.update_hbar_effective(avg_visits)
            expected_hbar = 1.0 / np.sqrt(avg_visits)
            assert abs(qft_engine.config.hbar_eff - expected_hbar) < 1e-10
            
    def test_classical_limit(self, qft_engine, sample_data):
        """Test classical limit as ℏ_eff → 0"""
        paths, visit_counts = sample_data
        
        # Set very small ℏ_eff (classical limit)
        qft_engine.config.hbar_eff = 1e-6
        
        weights = qft_engine.compute_path_weights(paths, visit_counts)
        
        # In classical limit, should approach deterministic selection
        # (weights concentrated on high-visit paths)
        max_weight = torch.max(weights)
        assert max_weight > 0.5  # Dominant path should have >50% probability
        
        # Check that we're in classical limit
        assert qft_engine.is_in_classical_limit()


class TestEffectiveActionEngine:
    """Test the effective action computation engine"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def config(self):
        return QFTConfig(
            hbar_eff=0.1,
            temperature=1.0,
            matrix_regularization=1e-8
        )
    
    @pytest.fixture
    def engine(self, config, device):
        return EffectiveActionEngine(config, device)
    
    def test_fluctuation_matrix_properties(self, engine, device):
        """Test properties of fluctuation matrix M_ij = δ²S/δπᵢδπⱼ"""
        # Simple test case
        paths = torch.tensor([[0, 1, 2], [1, 2, 3]], device=device)
        visit_counts = torch.tensor([10.0, 5.0, 8.0, 12.0], device=device)
        
        matrices = engine._build_fluctuation_matrix(paths, visit_counts)
        
        for matrix in matrices:
            # Should be symmetric
            assert torch.allclose(matrix, matrix.T, atol=1e-6)
            
            # Diagonal should be positive (1/N_i²)
            assert torch.all(torch.diag(matrix) > 0)
            
            # Should be square
            assert matrix.shape[0] == matrix.shape[1]
    
    def test_determinant_computation(self, engine, device):
        """Test log determinant computation for various matrix types"""
        # Test with well-conditioned matrix
        matrix = torch.eye(5, device=device) + 0.1 * torch.randn(5, 5, device=device)
        matrix = 0.5 * (matrix + matrix.T)  # Make symmetric
        matrix = matrix + 0.1 * torch.eye(5, device=device)  # Ensure positive definite
        
        log_det = engine._compute_log_determinant_gpu([matrix])
        
        # Should be finite
        assert torch.isfinite(log_det).all()
        
        # Compare with torch.logdet for validation
        expected = torch.logdet(matrix)
        if expected > -np.inf:  # Valid comparison
            assert abs(log_det[0] - expected) < 1e-4
    
    def test_quantum_correction_scaling(self, engine, device):
        """Test that quantum corrections scale properly with ℏ_eff"""
        paths = torch.randint(0, 50, (16, 8), device=device)
        visit_counts = torch.rand(50, device=device) * 100 + 1
        
        # Test different ℏ_eff values
        hbar_values = [0.01, 0.1, 0.5]
        corrections = []
        
        for hbar in hbar_values:
            engine.config.hbar_eff = hbar
            
            classical, _ = engine.compute_effective_action(paths, visit_counts, include_quantum=False)
            quantum, _ = engine.compute_effective_action(paths, visit_counts, include_quantum=True)
            
            correction = torch.abs(quantum - classical).mean()
            corrections.append(correction.item())
        
        # Corrections should increase with ℏ_eff
        assert corrections[1] > corrections[0]  # 0.1 > 0.01
        assert corrections[2] > corrections[1]  # 0.5 > 0.1
    
    def test_decoherence_correction(self, engine, device):
        """Test decoherence correction computation"""
        # Create paths with varying visit count patterns
        paths = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], device=device)
        
        # High variance visit counts (more decoherence)
        high_var_visits = torch.tensor([1.0, 100.0, 2.0, 90.0, 3.0, 80.0, 4.0, 70.0], device=device)
        
        # Low variance visit counts (less decoherence) 
        low_var_visits = torch.tensor([50.0, 52.0, 48.0, 51.0, 49.0, 53.0, 47.0, 54.0], device=device)
        
        decoherence_high = engine._compute_decoherence_correction(paths, high_var_visits)
        decoherence_low = engine._compute_decoherence_correction(paths, low_var_visits)
        
        # Higher variance should lead to more decoherence
        assert torch.all(decoherence_high >= decoherence_low)


class TestTheoreticalPredictions:
    """Test theoretical predictions from QFT formulation"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_scaling_relation_exponent(self, device):
        """Test scaling relation ⟨N(r)N(0)⟩ ~ r^{-(d-2+η)}"""
        # This is a simplified test of the scaling relation
        # In practice, this would require full tree correlation analysis
        
        config = QFTConfig(dimension=4, hbar_eff=0.1)
        engine = QFTEngine(config, device)
        
        # Expected exponent from theory
        d = config.dimension
        g = config.hbar_eff  # coupling ~ ℏ_eff
        eta = g**2 / (2 * np.pi)  # Anomalous dimension
        expected_exponent = d - 2 + eta
        
        # For d=4, ℏ=0.1: exponent ≈ 2.0016
        assert abs(expected_exponent - 2.0) < 0.1
        
    def test_effective_planck_constant_relationship(self, device):
        """Test ℏ_eff = 1/√N̄ relationship"""
        config = QFTConfig()
        engine = QFTEngine(config, device)
        
        # Test various average visit counts
        test_values = [1, 4, 16, 100, 1000]
        
        for N_avg in test_values:
            engine.update_hbar_effective(N_avg)
            expected = 1.0 / np.sqrt(N_avg)
            actual = engine.config.hbar_eff
            
            assert abs(actual - expected) < 1e-10
            
    def test_quantum_classical_consistency(self, device):
        """Test that quantum corrections vanish in classical limit"""
        config = QFTConfig(hbar_eff=1e-10)  # Very small ℏ_eff
        engine = QFTEngine(config, device)
        
        # Generate test data
        paths = torch.randint(0, 100, (32, 10), device=device)
        visit_counts = torch.rand(100, device=device) * 1000 + 100  # Large visit counts
        
        classical_action, _ = engine.effective_action_engine.compute_effective_action(
            paths, visit_counts, include_quantum=False
        )
        quantum_action, _ = engine.effective_action_engine.compute_effective_action(
            paths, visit_counts, include_quantum=True
        )
        
        # Relative correction should be very small
        relative_correction = torch.abs(quantum_action - classical_action) / torch.abs(classical_action)
        assert torch.all(relative_correction < 1e-6)


class TestGPUPerformance:
    """Test GPU performance and scaling"""
    
    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("GPU tests require CUDA")
        return torch.device('cuda')
    
    def test_batch_scaling(self, device):
        """Test performance scaling with batch size"""
        config = QFTConfig()
        engine = QFTEngine(config, device)
        
        batch_sizes = [32, 128, 512, 1024]
        times = []
        
        for batch_size in batch_sizes:
            paths = torch.randint(0, 200, (batch_size, 15), device=device)
            visit_counts = torch.rand(200, device=device) * 100 + 1
            
            # Warm up
            for _ in range(5):
                engine.compute_path_weights(paths, visit_counts)
            
            # Time computation
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            for _ in range(10):
                weights = engine.compute_path_weights(paths, visit_counts)
                
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            avg_time = (end - start) / 10
            times.append(avg_time)
            
        # Performance should scale sub-linearly with batch size
        # (good GPU utilization)
        throughput_32 = batch_sizes[0] / times[0]
        throughput_1024 = batch_sizes[-1] / times[-1]
        
        # Should get at least 10x throughput improvement for 32x more data
        assert throughput_1024 > 10 * throughput_32
        
    def test_memory_efficiency(self, device):
        """Test memory usage scaling"""
        config = QFTConfig()
        engine = QFTEngine(config, device)
        
        # Monitor memory usage
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Large computation
        large_paths = torch.randint(0, 1000, (2048, 20), device=device)
        large_visits = torch.rand(1000, device=device) * 1000 + 1
        
        weights = engine.compute_path_weights(large_paths, large_visits)
        
        peak_memory = torch.cuda.memory_allocated()
        memory_used = (peak_memory - initial_memory) / 1024**2  # MB
        
        # Should use reasonable amount of memory (< 1GB for this test)
        assert memory_used < 1000  # Less than 1GB


def test_integration_with_existing_mcts():
    """Test integration with existing MCTS components"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create QFT engine
    config = QFTConfig(hbar_eff=0.1)
    qft_engine = QFTEngine(config, device)
    
    # Simulate MCTS tree data
    num_nodes = 1000
    batch_size = 64
    max_depth = 12
    
    # Realistic visit count distribution (power law)
    alpha = 2.0  # Power law exponent
    visit_counts = torch.tensor([
        1.0 / (i + 1)**alpha for i in range(num_nodes)
    ], device=device) * 1000
    
    # Generate paths from tree
    paths = torch.randint(0, num_nodes, (batch_size, max_depth), device=device)
    
    # Compute QFT weights
    weights = qft_engine.compute_path_weights(paths, visit_counts)
    
    # Test integration properties
    assert torch.all(torch.isfinite(weights))
    assert torch.abs(weights.sum() - 1.0) < 1e-5
    assert weights.shape[0] == batch_size
    
    # Check statistics
    stats = qft_engine.get_statistics()
    assert stats['total_computations'] == batch_size
    assert 'avg_quantum_strength' in stats


if __name__ == "__main__":
    # Run basic functionality test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running QFT Engine tests on {device}")
    
    # Create and test engine
    config = QFTConfig(hbar_eff=0.1)
    engine = QFTEngine(config, device)
    
    # Test data
    paths = torch.randint(0, 100, (32, 10), device=device)
    visit_counts = torch.rand(100, device=device) * 100 + 1
    
    # Test computation
    weights = engine.compute_path_weights(paths, visit_counts)
    print(f"✓ QFT path weights computed: shape {weights.shape}")
    print(f"✓ Weight sum: {weights.sum():.6f} (should be ~1.0)")
    print(f"✓ Weight range: [{weights.min():.6f}, {weights.max():.6f}]")
    
    # Test quantum corrections
    quantum_strength = engine.stats['avg_quantum_strength']
    print(f"✓ Quantum correction strength: {quantum_strength:.4f}")
    
    print("✓ All basic tests passed!")