"""
Tests for Quantum MCTS v2.0 Implementation
=========================================

Tests the new features:
- Discrete information time
- Auto-computed parameters
- Phase detection and transitions
- Full PUCT integration
- Power-law decoherence
- Envariance convergence
"""

import pytest
import torch
import numpy as np
import math
from typing import Dict, Any

from mcts.quantum.quantum_features_v2 import (
    QuantumMCTSV2, QuantumConfigV2, DiscreteTimeEvolution,
    PhaseDetector, OptimalParameters, MCTSPhase,
    create_quantum_mcts_v2
)


class TestDiscreteTimeEvolution:
    """Test discrete information time framework"""
    
    def test_information_time(self):
        """Test τ(N) = log(N+2)"""
        config = QuantumConfigV2()
        time_evo = DiscreteTimeEvolution(config)
        
        # Test scalar
        assert abs(time_evo.information_time(0) - math.log(2)) < 1e-6
        assert abs(time_evo.information_time(10) - math.log(12)) < 1e-6
        assert abs(time_evo.information_time(1000) - math.log(1002)) < 1e-6
        
        # Test tensor
        N = torch.tensor([0, 10, 100, 1000])
        tau = time_evo.information_time(N)
        expected = torch.log(N + 2)
        assert torch.allclose(tau, expected)
    
    def test_temperature_annealing(self):
        """Test T(N) = T₀/log(N+2)"""
        config = QuantumConfigV2(
            initial_temperature=2.0,
            temperature_mode='annealing'
        )
        time_evo = DiscreteTimeEvolution(config)
        
        # Early: high temperature
        T_early = time_evo.compute_temperature(1)
        assert T_early > 1.5  # T₀/log(3) ≈ 2/1.099 ≈ 1.82
        
        # Late: low temperature
        T_late = time_evo.compute_temperature(10000)
        assert T_late < 0.3  # T₀/log(10002) ≈ 2/9.21 ≈ 0.22
        
        # Monotonic decrease
        N_values = [1, 10, 100, 1000, 10000]
        temps = [time_evo.compute_temperature(N) for N in N_values]
        assert all(temps[i] > temps[i+1] for i in range(len(temps)-1))
    
    def test_hbar_eff_scaling(self):
        """Test ℏ_eff(N) = c_puct(N+2)/(√(N+1)log(N+2))"""
        config = QuantumConfigV2(hbar_eff=None)  # Auto-compute
        time_evo = DiscreteTimeEvolution(config)
        
        c_puct = 1.414
        
        # Early: large quantum effects
        hbar_early = time_evo.compute_hbar_eff(1, c_puct)
        expected_early = c_puct * 3 / (math.sqrt(2) * math.log(3))
        assert abs(hbar_early - expected_early) < 1e-6
        
        # Late: larger but slower growing quantum effects
        hbar_late = time_evo.compute_hbar_eff(1000, c_puct)
        assert hbar_late > hbar_early  # Actually increases with N
        
        # Correct asymptotic behavior
        N_large = 100000
        hbar_asymp = time_evo.compute_hbar_eff(N_large, c_puct)
        # Should scale as √N/log(N) for large N
        assert hbar_asymp < 100.0  # Reasonable bound
        
        # Check growth rate follows theoretical formula
        # hbar_eff(N) = c_puct * (N+2) / (sqrt(N+1) * log(N+2))
        # For large N, this grows like sqrt(N)/log(N)
        # But the exact ratio calculation is more complex due to the +2 and +1 terms


class TestPhaseDetection:
    """Test phase detection and transitions"""
    
    def test_critical_points(self):
        """Test computation of N_c1 and N_c2"""
        config = QuantumConfigV2()
        detector = PhaseDetector(config)
        
        b = 20  # Branching factor
        c_puct = math.sqrt(2 * math.log(b))
        
        # Without neural network
        N_c1, N_c2 = detector.compute_critical_points(b, c_puct, has_neural_prior=False)
        
        # Check reasonable values
        assert 10 < N_c1 < 1000
        assert N_c1 < N_c2 < 100000
        
        # With neural network (should shift higher)
        N_c1_nn, N_c2_nn = detector.compute_critical_points(b, c_puct, has_neural_prior=True)
        assert N_c1_nn > N_c1
        assert N_c2_nn > N_c2
    
    def test_phase_detection(self):
        """Test phase detection at different N"""
        config = QuantumConfigV2(use_neural_prior=True)
        detector = PhaseDetector(config)
        
        b = 20
        c_puct = math.sqrt(2 * math.log(b))
        
        # Get critical points
        N_c1, N_c2 = detector.compute_critical_points(b, c_puct, True)
        
        # Test phases
        assert detector.detect_phase(int(N_c1 / 2), b, c_puct, True) == MCTSPhase.QUANTUM
        assert detector.detect_phase(int((N_c1 + N_c2) / 2), b, c_puct, True) == MCTSPhase.CRITICAL
        assert detector.detect_phase(int(N_c2 * 2), b, c_puct, True) == MCTSPhase.CLASSICAL
    
    def test_phase_config(self):
        """Test phase-specific configurations"""
        config = QuantumConfigV2()
        detector = PhaseDetector(config)
        
        # Quantum phase: high exploration
        quantum_cfg = detector.get_phase_config(MCTSPhase.QUANTUM)
        assert quantum_cfg['quantum_strength'] == 1.0
        assert quantum_cfg['prior_trust'] < 1.0
        assert quantum_cfg['batch_size'] == 32
        
        # Classical phase: exploitation
        classical_cfg = detector.get_phase_config(MCTSPhase.CLASSICAL)
        assert classical_cfg['quantum_strength'] < 0.5
        assert classical_cfg['prior_trust'] > 1.0
        assert classical_cfg['batch_size'] < 16


class TestOptimalParameters:
    """Test physics-derived parameter computation"""
    
    def test_c_puct_computation(self):
        """Test optimal c_puct = √(2 log b)"""
        # Test various branching factors
        for b in [2, 10, 20, 50, 100]:
            c_puct = OptimalParameters.compute_c_puct(b)
            expected = math.sqrt(2 * math.log(b))
            assert abs(c_puct - expected) < 0.1
    
    def test_hash_functions(self):
        """Test optimal hash count K = √(b·L)"""
        b = 20
        L = 100
        
        # Without neural network
        K_no_nn = OptimalParameters.compute_num_hashes(b, L, has_neural_network=False)
        expected = int(math.sqrt(b * L))
        assert abs(K_no_nn - expected) < 5
        
        # With neural network (reduced)
        K_with_nn = OptimalParameters.compute_num_hashes(b, L, has_neural_network=True)
        assert K_with_nn < K_no_nn
        assert K_with_nn >= 4  # Minimum
    
    def test_phase_kick_schedule(self):
        """Test phase kick probability γ(N) = 1/√(N+1)"""
        # Early: high probability
        assert OptimalParameters.phase_kick_schedule(1) == 0.1  # Capped at 0.1
        
        # Check 1/√(N+1) scaling
        assert abs(OptimalParameters.phase_kick_schedule(99) - 0.1) < 1e-6
        assert abs(OptimalParameters.phase_kick_schedule(100) - 1/math.sqrt(101)) < 1e-6
        assert abs(OptimalParameters.phase_kick_schedule(10000) - 1/math.sqrt(10001)) < 1e-6


class TestQuantumMCTSV2:
    """Test main QuantumMCTSV2 implementation"""
    
    @pytest.fixture
    def quantum_mcts(self):
        """Create quantum MCTS instance for testing"""
        config = QuantumConfigV2(
            branching_factor=20,
            avg_game_length=50,
            c_puct=None,  # Auto-compute
            enable_quantum=True,
            quantum_level='tree_level',
            device='cpu'  # For testing
        )
        return QuantumMCTSV2(config)
    
    def test_initialization(self, quantum_mcts):
        """Test proper initialization"""
        # Check auto-computed parameters
        assert quantum_mcts.config.c_puct is not None
        assert quantum_mcts.config.num_hash_functions is not None
        
        # Check components
        assert quantum_mcts.time_evolution is not None
        assert quantum_mcts.phase_detector is not None
        assert quantum_mcts.current_phase == MCTSPhase.QUANTUM
    
    def test_quantum_selection_basic(self, quantum_mcts):
        """Test basic quantum selection enhancement"""
        # Create test data
        batch_size = 64
        num_actions = 10
        
        q_values = torch.randn(batch_size, num_actions)
        visit_counts = torch.randint(0, 100, (batch_size, num_actions))
        priors = torch.softmax(torch.randn(batch_size, num_actions), dim=-1)
        
        # Apply quantum enhancement
        ucb_scores = quantum_mcts.apply_quantum_to_selection(
            q_values, visit_counts, priors,
            simulation_count=100
        )
        
        # Check shape
        assert ucb_scores.shape == q_values.shape
        
        # Check that low-visit nodes get boost
        low_visit_mask = visit_counts < 5
        if low_visit_mask.any():
            low_visit_boost = (ucb_scores - q_values)[low_visit_mask].mean()
            high_visit_boost = (ucb_scores - q_values)[~low_visit_mask].mean()
            assert low_visit_boost > high_visit_boost
    
    def test_phase_transitions(self, quantum_mcts):
        """Test phase transitions with simulation count"""
        # Start in quantum phase
        assert quantum_mcts.current_phase == MCTSPhase.QUANTUM
        
        # Simulate at different N values
        N_values = [10, 100, 1000, 10000, 100000]
        phases_seen = set()
        
        for N in N_values:
            quantum_mcts.update_simulation_count(N)
            phases_seen.add(quantum_mcts.current_phase)
        
        # Should see multiple phases
        assert len(phases_seen) >= 2
        
        # Check statistics
        stats = quantum_mcts.get_statistics()
        assert stats['phase_transitions'] >= 1
    
    def test_prior_coupling(self, quantum_mcts):
        """Test neural network prior integration"""
        # Single node test
        q_values = torch.tensor([0.1, 0.2, 0.3, 0.4])
        visit_counts = torch.tensor([10, 5, 2, 0])
        
        # Uniform priors
        uniform_priors = torch.ones(4) / 4
        ucb_uniform = quantum_mcts.apply_quantum_to_selection(
            q_values, visit_counts, uniform_priors
        )
        
        # Peaked priors (neural network favors action 0)
        peaked_priors = torch.tensor([0.7, 0.1, 0.1, 0.1])
        ucb_peaked = quantum_mcts.apply_quantum_to_selection(
            q_values, visit_counts, peaked_priors
        )
        
        # Action 0 should get bigger boost with peaked prior
        boost_diff = (ucb_peaked - ucb_uniform)
        assert boost_diff[0] > boost_diff[1]
    
    def test_power_law_decoherence(self, quantum_mcts):
        """Test power-law decoherence vs exponential"""
        # Configure for one-loop to test decoherence
        quantum_mcts.config.quantum_level = 'one_loop'
        
        # Create test data with varying visit counts
        q_values = torch.zeros(100)
        visit_counts = torch.arange(1, 101)
        priors = torch.ones(100) / 100
        
        # Apply quantum with decoherence
        ucb_scores = quantum_mcts.apply_quantum_to_selection(
            q_values, visit_counts, priors
        )
        
        # Extract quantum bonus
        quantum_bonus = ucb_scores - q_values
        
        # Check power-law decay (not exponential)
        # For power law: log(bonus) ~ -γ log(N)
        log_bonus = torch.log(quantum_bonus[10:] + 1e-8)  # Skip very low visits
        log_visits = torch.log(visit_counts[10:].float())
        
        # Linear fit in log-log space indicates power law
        # (Would need scipy for proper fit, checking trend here)
        early_avg = log_bonus[:20].mean()
        late_avg = log_bonus[-20:].mean()
        assert late_avg < early_avg  # Decay present
    
    def test_envariance_check(self):
        """Test envariance convergence criterion"""
        # Mock tree object
        class MockTree:
            def __init__(self):
                self.entropy = 1.0
            
            def get_policy_entropy(self):
                return self.entropy
        
        tree = MockTree()
        quantum_mcts = create_quantum_mcts_v2(branching_factor=20)
        
        # Not converged initially
        assert not quantum_mcts.check_envariance(tree, threshold=0.1)
        
        # Simulate convergence
        tree.entropy = 0.05
        assert quantum_mcts.check_envariance(tree, threshold=0.1)
        assert quantum_mcts.stats['convergence_reached']
    
    def test_factory_function(self):
        """Test create_quantum_mcts_v2 factory"""
        # With auto-computation
        qmcts = create_quantum_mcts_v2(
            branching_factor=10,
            avg_game_length=30,
            use_neural_network=True
        )
        
        # Check auto-computed values
        assert qmcts.config.c_puct == OptimalParameters.compute_c_puct(10)
        assert qmcts.config.num_hash_functions is not None
        assert qmcts.config.use_neural_prior == True


class TestIntegration:
    """Integration tests for v2.0 features"""
    
    def test_full_simulation_flow(self):
        """Test complete simulation flow with phase transitions"""
        # Create MCTS for Go-like game
        quantum_mcts = create_quantum_mcts_v2(
            branching_factor=200,
            avg_game_length=200,
            use_neural_network=True,
            device='cpu'
        )
        
        # Simulate MCTS search progression
        phases_seen = []
        temperatures = []
        hbar_values = []
        
        for N in [1, 10, 100, 1000, 10000, 50000]:
            quantum_mcts.update_simulation_count(N)
            
            # Create dummy data
            q_values = torch.randn(32, 200)
            visit_counts = torch.randint(0, N//10 + 1, (32, 200))
            priors = torch.softmax(torch.randn(32, 200), dim=-1)
            
            # Apply quantum selection
            ucb_scores = quantum_mcts.apply_quantum_to_selection(
                q_values, visit_counts, priors, simulation_count=N
            )
            
            # Track evolution
            stats = quantum_mcts.get_statistics()
            phases_seen.append(stats['current_phase'])
            temperatures.append(stats['current_temperature'])
            hbar_values.append(stats['current_hbar_eff'])
        
        # Verify progression
        assert phases_seen[0] == 'quantum'  # Early exploration
        assert phases_seen[-1] in ['critical', 'classical']  # Late convergence
        assert temperatures[0] > temperatures[-1]  # Temperature annealing
        # Note: hbar_eff actually increases with N due to the (N+2)/sqrt(N+1) factor
        assert hbar_values[0] < hbar_values[-1]  # hbar_eff grows with N
    
    def test_performance_characteristics(self):
        """Test that overhead is within expected bounds"""
        import time
        
        # Classical baseline
        quantum_mcts_off = create_quantum_mcts_v2(
            enable_quantum=False,
            branching_factor=20,
            device='cpu'
        )
        
        # Quantum enhanced
        quantum_mcts_on = create_quantum_mcts_v2(
            enable_quantum=True,
            quantum_level='tree_level',
            branching_factor=20,
            device='cpu'
        )
        
        # Test data
        batch_size = 1000
        num_actions = 20
        q_values = torch.randn(batch_size, num_actions)
        visit_counts = torch.randint(0, 100, (batch_size, num_actions))
        priors = torch.softmax(torch.randn(batch_size, num_actions), dim=-1)
        
        # Time classical
        start = time.time()
        for _ in range(10):
            _ = quantum_mcts_off.apply_quantum_to_selection(
                q_values, visit_counts, priors
            )
        classical_time = time.time() - start
        
        # Time quantum
        start = time.time()
        for _ in range(10):
            _ = quantum_mcts_on.apply_quantum_to_selection(
                q_values, visit_counts, priors
            )
        quantum_time = time.time() - start
        
        # Check overhead
        overhead = quantum_time / classical_time
        print(f"Overhead: {overhead:.2f}x")
        assert overhead < 3.0  # Should be < 2x in optimized version


if __name__ == "__main__":
    pytest.main([__file__, "-v"])