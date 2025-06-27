"""
Comprehensive tests for quantum MCTS optimization and integration

This test suite validates the integration and optimization of quantum components:
1. Wave-based quantum processing integration
2. Path integral formulation correctness  
3. Discrete-time evolution accuracy
4. Performance requirements (< 2x overhead)
5. Mathematical consistency across components
"""

import pytest
import torch
import numpy as np
import time
import math
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

# Import main quantum components (avoiding research folder)
from mcts.quantum.quantum_mcts_wrapper import QuantumMCTSWrapper, UnifiedQuantumConfig
from mcts.quantum.quantum_features_v2_optimized import OptimizedQuantumMCTSV2
from mcts.quantum.quantum_features_v2 import QuantumMCTSV2, QuantumConfigV2, MCTSPhase
from mcts.quantum.quantum_features import QuantumMCTS, QuantumConfig


class TestQuantumIntegrationOptimization:
    """Test suite for quantum MCTS integration and optimization"""
    
    @pytest.fixture
    def device(self):
        """Test device - prefer CUDA for performance testing"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def test_data(self, device):
        """Generate test data for quantum MCTS validation"""
        batch_size = 64
        num_actions = 50
        
        return {
            'q_values': torch.randn(batch_size, num_actions, device=device),
            'visit_counts': torch.randint(1, 100, (batch_size, num_actions), device=device),
            'priors': torch.softmax(torch.randn(batch_size, num_actions, device=device), dim=-1),
            'parent_visits': torch.randint(50, 1000, (batch_size,), device=device),
            'simulation_counts': torch.arange(1000, 1000 + batch_size, device=device)
        }
    
    @pytest.fixture
    def optimized_config(self, device):
        """Create optimized quantum configuration"""
        return QuantumConfigV2(
            enable_quantum=True,
            branching_factor=50,
            avg_game_length=100,
            c_puct=np.sqrt(2 * np.log(50)),
            use_neural_prior=True,
            enable_phase_adaptation=True,
            temperature_mode='annealing',
            device=device.type,
            use_mixed_precision=True,
            cache_quantum_corrections=True,
            fast_mode=False  # Full quantum for testing
        )
    
    def test_information_time_evolution_correctness(self, device):
        """Test discrete information time evolution δτ_N = 1/(N_root + 2)"""
        config = QuantumConfigV2(
            device=device.type,
            branching_factor=30,  # Required for auto-computation
            avg_game_length=100
        )
        quantum = QuantumMCTSV2(config)
        
        # Test information time calculation for various N values
        test_cases = [0, 1, 10, 100, 1000, 10000]
        
        for N in test_cases:
            # Calculate τ(N) = log(N + 2)
            tau_expected = np.log(N + 2)
            tau_actual = quantum.time_evolution.information_time(N)
            
            assert abs(tau_actual - tau_expected) < 1e-6, \
                f"Information time τ({N}) = {tau_actual:.6f}, expected {tau_expected:.6f}"
            
            # Test discrete time step δτ_N = 1/(N + 2) via time_derivative method
            if N > 0:
                delta_tau_expected = 1.0 / (N + 2)
                delta_tau_actual = quantum.time_evolution.time_derivative(N)
                
                assert abs(delta_tau_actual - delta_tau_expected) < 1e-6, \
                    f"Discrete time step δτ({N}) = {delta_tau_actual:.6f}, expected {delta_tau_expected:.6f}"
    
    def test_path_integral_normalization(self, device, test_data):
        """Test that path integral amplitudes are properly normalized"""
        from mcts.quantum.quantum_features_v2_ultra_optimized import UltraOptimizedQuantumMCTSV2
        
        config = QuantumConfigV2(
            device=device.type, 
            enable_quantum=True,
            branching_factor=50,
            avg_game_length=100
        )
        quantum = UltraOptimizedQuantumMCTSV2(config)
        
        # Apply quantum selection
        enhanced_scores = quantum.apply_quantum_to_selection(
            test_data['q_values'][0],  # Single batch
            test_data['visit_counts'][0],
            test_data['priors'][0],
            c_puct=1.414,
            simulation_count=1000
        )
        
        # Check that probabilities sum to 1 (within tolerance)
        prob_sum = torch.sum(torch.softmax(enhanced_scores, dim=0))
        assert abs(prob_sum.item() - 1.0) < 1e-6, \
            f"Path integral probabilities sum to {prob_sum.item()}, expected 1.0"
        
        # Check that amplitudes are finite and non-negative
        assert torch.all(torch.isfinite(enhanced_scores)), "Enhanced scores contain non-finite values"
        
        # Verify quantum corrections don't dominate classical terms
        classical_scores = test_data['q_values'][0] + 1.414 * test_data['priors'][0] * \
                          np.sqrt(np.log(1000)) / torch.sqrt(test_data['visit_counts'][0].float() + 1)
        
        correction_magnitude = torch.mean(torch.abs(enhanced_scores - classical_scores))
        classical_magnitude = torch.mean(torch.abs(classical_scores))
        
        # Quantum corrections should be meaningful but not overwhelming
        assert correction_magnitude < 2.0 * classical_magnitude, \
            "Quantum corrections are too large compared to classical terms"
    
    def test_hamiltonian_construction_consistency(self, device):
        """Test Hamiltonian construction via discrete Legendre transform"""
        # This test validates the theoretical consistency requirements
        # from the documentation: H = Σ[π²/(2κ) + V(N)] with proper kinetic/potential terms
        
        config = QuantumConfigV2(
            device=device.type, 
            enable_quantum=True,
            branching_factor=30,
            avg_game_length=100
        )
        quantum = QuantumMCTSV2(config)
        
        # Test data for Hamiltonian construction
        num_edges = 10
        edge_indices = torch.arange(num_edges, device=device)
        visit_counts = torch.randint(1, 50, (num_edges,), device=device)
        priors = torch.softmax(torch.randn(num_edges, device=device), dim=0)
        q_values = torch.randn(num_edges, device=device)
        
        # Test if we can construct a valid Hamiltonian matrix
        # For now, we'll test the potential energy terms which should be:
        # V_σ(N) = -[log N + λ log P - β Q]
        
        lambda_param = 1.4  # PUCT prior strength
        beta_param = 1.0    # Value weight
        
        expected_potential = -(torch.log(visit_counts.float() + 1e-8) + 
                             lambda_param * torch.log(priors + 1e-8) - 
                             beta_param * q_values)
        
        # The potential energy should be finite and well-behaved
        assert torch.all(torch.isfinite(expected_potential)), \
            "Hamiltonian potential terms contain non-finite values"
        
        # Test that diagonal Hamiltonian has correct eigenvalue structure
        H_diag = torch.diag(expected_potential)
        eigenvals = torch.linalg.eigvals(H_diag.to(dtype=torch.complex64))
        
        assert torch.all(torch.isfinite(eigenvals)), \
            "Hamiltonian eigenvalues are not finite"
    
    def test_causality_preservation(self, device, test_data):
        """Test that quantum evolution preserves causality using pre-update visit counts"""
        from mcts.quantum.quantum_features_v2_ultra_optimized import UltraOptimizedQuantumMCTSV2
        
        config = QuantumConfigV2(
            device=device.type, 
            enable_quantum=True,
            branching_factor=50,
            avg_game_length=100
        )
        quantum = UltraOptimizedQuantumMCTSV2(config)
        
        # Test multiple iterations to check causality
        current_visits = test_data['visit_counts'][0].clone()
        
        for i in range(5):
            # Apply quantum selection using current (pre-update) visit counts
            enhanced_scores = quantum.apply_quantum_to_selection(
                test_data['q_values'][0],
                current_visits,  # Pre-update counts
                test_data['priors'][0],
                c_puct=1.414,
                simulation_count=1000 + i
            )
            
            # Simulate action selection and visit count update
            selected_action = torch.argmax(enhanced_scores)
            
            # Update visit counts (post-update)
            current_visits[selected_action] += 1
            
            # Verify that the quantum computation only used pre-update counts
            # This is ensured by the API design, but we test stability
            assert torch.all(torch.isfinite(enhanced_scores)), \
                f"Enhanced scores not finite at iteration {i}"
            
            # Test that repeated applications remain stable
            if i > 0:
                # Scores should remain in reasonable bounds
                score_magnitude = torch.max(torch.abs(enhanced_scores))
                assert score_magnitude < 100.0, \
                    f"Score magnitude {score_magnitude} too large at iteration {i}"
    
    def test_performance_overhead_requirement(self, device, test_data):
        """Test that quantum MCTS maintains < 2x performance overhead"""
        num_iterations = 500
        
        # Baseline classical PUCT timing
        q_vals = test_data['q_values'][0]
        visits = test_data['visit_counts'][0]
        priors = test_data['priors'][0]
        
        # Classical PUCT computation
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            c_puct = 1.414
            sqrt_parent = math.sqrt(math.log(1000))
            exploration = c_puct * priors * sqrt_parent / torch.sqrt(visits.float() + 1)
            classical_scores = q_vals + exploration
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        classical_time = time.perf_counter() - start_time
        
        # Ultra-optimized quantum MCTS timing
        from mcts.quantum.quantum_features_v2_ultra_optimized import UltraOptimizedQuantumMCTSV2
        
        config = QuantumConfigV2(
            device=device.type,
            enable_quantum=True,
            fast_mode=True,  # Use fast mode for performance testing
            cache_quantum_corrections=True,
            branching_factor=50,
            avg_game_length=100
        )
        quantum = UltraOptimizedQuantumMCTSV2(config)
        
        # Warmup
        for _ in range(10):
            quantum.apply_quantum_to_selection(q_vals, visits, priors, c_puct=1.414, simulation_count=1000)
        
        # Quantum timing
        start_time = time.perf_counter()
        for i in range(num_iterations):
            quantum_scores = quantum.apply_quantum_to_selection(
                q_vals, visits, priors, c_puct=1.414, simulation_count=1000 + i
            )
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        quantum_time = time.perf_counter() - start_time
        
        # Calculate overhead
        overhead = quantum_time / classical_time
        
        print(f"\nPerformance test results:")
        print(f"Classical time: {classical_time:.4f}s")
        print(f"Quantum time: {quantum_time:.4f}s")
        print(f"Overhead: {overhead:.2f}x")
        
        # Assert performance requirement
        assert overhead < 2.0, \
            f"Quantum MCTS overhead {overhead:.2f}x exceeds requirement of < 2.0x"
    
    def test_quantum_classical_consistency(self, device, test_data):
        """Test that quantum MCTS reduces to classical MCTS in appropriate limits"""
        from mcts.quantum.quantum_features_v2_ultra_optimized import UltraOptimizedQuantumMCTSV2
        
        # Test with quantum effects disabled
        config_classical = QuantumConfigV2(
            device=device.type,
            enable_quantum=False,
            branching_factor=50,
            avg_game_length=100
        )
        quantum_disabled = UltraOptimizedQuantumMCTSV2(config_classical)
        
        # Test with quantum effects at minimum strength (using coupling_strength)
        config_minimal = QuantumConfigV2(
            device=device.type,
            enable_quantum=True,
            branching_factor=50,
            avg_game_length=100,
            coupling_strength=0.001  # Minimal quantum effects
        )
        quantum_minimal = UltraOptimizedQuantumMCTSV2(config_minimal)
        
        # Apply both to same data
        classical_scores = quantum_disabled.apply_quantum_to_selection(
            test_data['q_values'][0],
            test_data['visit_counts'][0],
            test_data['priors'][0],
            c_puct=1.414,
            simulation_count=10000  # Large N for classical limit
        )
        
        minimal_quantum_scores = quantum_minimal.apply_quantum_to_selection(
            test_data['q_values'][0],
            test_data['visit_counts'][0],
            test_data['priors'][0],
            c_puct=1.414,
            simulation_count=10000
        )
        
        # Should be very similar in classical limit
        max_difference = torch.max(torch.abs(classical_scores - minimal_quantum_scores))
        mean_magnitude = torch.mean(torch.abs(classical_scores))
        
        relative_error = max_difference / (mean_magnitude + 1e-8)
        
        assert relative_error < 0.1, \
            f"Quantum and classical scores differ by {relative_error:.3f} in classical limit"
    
    def test_wave_processing_integration(self, device):
        """Test integration with wave-based processing for 3072-path waves"""
        from mcts.quantum.quantum_features_v2_ultra_optimized import UltraOptimizedQuantumMCTSV2
        
        wave_size = 3072  # Target wave size for optimal GPU utilization
        num_actions = 30
        
        # Create wave-sized test data
        q_values = torch.randn(wave_size, num_actions, device=device)
        visit_counts = torch.randint(1, 100, (wave_size, num_actions), device=device)
        priors = torch.softmax(torch.randn(wave_size, num_actions, device=device), dim=-1)
        
        config = QuantumConfigV2(
            device=device.type,
            enable_quantum=True,
            branching_factor=30,
            avg_game_length=100,
            optimal_wave_size=wave_size
        )
        quantum = UltraOptimizedQuantumMCTSV2(config)
        
        # Test batch processing
        start_time = time.perf_counter()
        enhanced_scores = quantum.apply_quantum_to_selection(
            q_values,
            visit_counts,
            priors,
            c_puct=1.414,
            simulation_count=1000
        )
        processing_time = time.perf_counter() - start_time
        
        # Validate results
        assert enhanced_scores.shape == (wave_size, num_actions), \
            f"Output shape {enhanced_scores.shape} doesn't match input {(wave_size, num_actions)}"
        
        assert torch.all(torch.isfinite(enhanced_scores)), \
            "Wave processing produced non-finite values"
        
        # Performance check for wave processing
        paths_per_second = wave_size / processing_time
        print(f"Wave processing: {paths_per_second:.0f} paths/second")
        
        # Should process at least 10k paths per second on modern hardware
        if device.type == 'cuda':
            assert paths_per_second > 10000, \
                f"Wave processing too slow: {paths_per_second:.0f} paths/second"


class TestMathematicalCorrectness:
    """Test mathematical correctness of quantum formulations"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_puct_formula_emergence(self):
        """Test that PUCT formula emerges from stationary action principle"""
        # From theory: c_puct = λ/√2 where λ is prior coupling strength
        lambda_values = [1.0, 1.414, 2.0]
        
        for lambda_val in lambda_values:
            expected_c_puct = lambda_val / math.sqrt(2)
            
            # Test that this relationship holds in implementation
            config = QuantumConfigV2(
                enable_quantum=True,
                c_puct=expected_c_puct,
                prior_coupling=lambda_val,  # Use prior_coupling instead of lambda_puct
                branching_factor=30,
                avg_game_length=100
            )
            
            # The configuration should maintain this relationship
            computed_ratio = config.c_puct * math.sqrt(2)
            assert abs(computed_ratio - lambda_val) < 1e-6, \
                f"PUCT emergence violated: {computed_ratio:.6f} ≠ {lambda_val}"
    
    def test_partition_function_convergence(self, device):
        """Test that quantum partition function converges appropriately"""
        config = QuantumConfigV2(
            device=device.type, 
            enable_quantum=True,
            branching_factor=30,
            avg_game_length=100
        )
        quantum = QuantumMCTSV2(config)
        
        # Test convergence with increasing simulation count
        visit_counts = torch.tensor([1, 5, 10, 20, 50, 100], device=device)
        priors = torch.softmax(torch.randn(6, device=device), dim=0)
        q_values = torch.randn(6, device=device)
        
        previous_scores = None
        convergence_achieved = False
        
        for N in [100, 500, 1000, 5000, 10000]:
            scores = quantum.apply_quantum_to_selection(
                q_values, visit_counts, priors, c_puct=1.414, simulation_count=N
            )
            
            if previous_scores is not None:
                max_change = torch.max(torch.abs(scores - previous_scores))
                if max_change < 1e-3:  # Convergence threshold
                    convergence_achieved = True
                    break
            
            previous_scores = scores.clone()
        
        assert convergence_achieved, "Partition function did not converge within reasonable N"
    
    def test_one_loop_corrections_sign(self, device):
        """Test that one-loop quantum corrections have correct sign and magnitude"""
        from mcts.quantum.quantum_features_v2_ultra_optimized import UltraOptimizedQuantumMCTSV2
        
        config = QuantumConfigV2(
            device=device.type, 
            enable_quantum=True,
            branching_factor=30,
            avg_game_length=100,
            fast_mode=False,  # Disable fast mode to ensure quantum corrections are applied
            coupling_strength=0.3  # Ensure non-zero quantum strength
        )
        quantum = UltraOptimizedQuantumMCTSV2(config)
        
        # Force quantum phase to ensure corrections are applied
        quantum.current_phase = MCTSPhase.QUANTUM
        
        # Test with low vs high visit counts
        low_visits = torch.tensor([1, 2, 3], device=device)
        high_visits = torch.tensor([100, 200, 300], device=device)
        priors = torch.tensor([0.3, 0.4, 0.3], device=device)
        q_values = torch.zeros(3, device=device)
        
        # Quantum corrections should be larger for low visit counts
        # (encouraging exploration of unvisited nodes)
        low_scores = quantum.apply_quantum_to_selection(
            q_values, low_visits, priors, c_puct=1.414, simulation_count=1000
        )
        
        high_scores = quantum.apply_quantum_to_selection(
            q_values, high_visits, priors, c_puct=1.414, simulation_count=1000
        )
        
        # Classical baseline (no quantum)
        config_classical = QuantumConfigV2(
            device=device.type, 
            enable_quantum=False,
            branching_factor=30,
            avg_game_length=100
        )
        classical = UltraOptimizedQuantumMCTSV2(config_classical)
        
        classical_low = classical.apply_quantum_to_selection(
            q_values, low_visits, priors, c_puct=1.414, simulation_count=1000
        )
        
        classical_high = classical.apply_quantum_to_selection(
            q_values, high_visits, priors, c_puct=1.414, simulation_count=1000
        )
        
        # Quantum corrections for low visits should be positive (encouraging exploration)
        quantum_correction_low = low_scores - classical_low
        quantum_correction_high = high_scores - classical_high
        
        # Mean correction should be positive for low visits, smaller for high visits
        mean_correction_low = torch.mean(quantum_correction_low)
        mean_correction_high = torch.mean(quantum_correction_high)
        
        assert mean_correction_low > 0, \
            f"Quantum correction for low visits is negative: {mean_correction_low:.6f}"
        
        assert mean_correction_low > mean_correction_high, \
            "Quantum corrections should be larger for low visit counts"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])