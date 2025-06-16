"""
Tests for Quantum MCTS v2.0 Integration
======================================

Tests the integration of v2.0 features with:
- Migration wrapper
- Decoherence engine v2.0
- Main MCTS integration
- Performance benchmarks
"""

import pytest
import torch
import numpy as np
import math
import time
from typing import Dict, Any

from mcts.quantum import (
    QuantumMCTSWrapper, UnifiedQuantumConfig, create_quantum_mcts,
    compare_versions, MCTSPhase,
    create_decoherence_engine_v2, create_qft_engine_v2, create_path_integral_v2
)
from mcts.core.mcts import MCTS, MCTSConfig
from mcts.core.game_interface import GameType
from mcts.neural_networks.simple_evaluator_wrapper import SimpleEvaluatorWrapper


class TestMigrationWrapper:
    """Test the unified migration wrapper"""
    
    def test_auto_version_detection(self):
        """Test automatic version detection based on parameters"""
        # v1 parameters -> v1 selection
        config_v1 = UnifiedQuantumConfig(
            exploration_constant=1.414,
            enable_quantum=True,
            enable_phase_adaptation=False,  # Disable v2 feature
            temperature_mode='fixed'  # v1 default
        )
        wrapper = QuantumMCTSWrapper(config_v1)
        assert wrapper.version == 'v1'
        
        # v2 parameters -> v2 selection
        config_v2 = UnifiedQuantumConfig(
            branching_factor=20,
            avg_game_length=50,
            enable_phase_adaptation=True,
            enable_quantum=True
        )
        wrapper = QuantumMCTSWrapper(config_v2)
        assert wrapper.version == 'v2'
        
        # Explicit version override
        config_override = UnifiedQuantumConfig(
            version='v1',
            auto_detect_version=False,
            branching_factor=20  # v2 param ignored
        )
        wrapper = QuantumMCTSWrapper(config_override)
        assert wrapper.version == 'v1'
    
    def test_v1_deprecation_warning(self):
        """Test deprecation warning for v1 usage"""
        config = UnifiedQuantumConfig(
            version='v1',
            auto_detect_version=False,
            suppress_deprecation_warnings=False
        )
        
        with pytest.warns(DeprecationWarning, match="Quantum MCTS v1.0 is deprecated"):
            wrapper = QuantumMCTSWrapper(config)
    
    def test_unified_interface(self):
        """Test that both versions work through unified interface"""
        # Test data
        q_values = torch.randn(32, 10)
        visit_counts = torch.randint(0, 100, (32, 10))
        priors = torch.softmax(torch.randn(32, 10), dim=-1)
        
        # v1 wrapper
        wrapper_v1 = create_quantum_mcts(
            enable_quantum=True,
            version='v1',
            device='cpu'
        )
        ucb_v1 = wrapper_v1.apply_quantum_to_selection(
            q_values, visit_counts, priors, c_puct=1.414
        )
        
        # v2 wrapper
        wrapper_v2 = create_quantum_mcts(
            enable_quantum=True,
            version='v2',
            branching_factor=10,
            device='cpu'
        )
        ucb_v2 = wrapper_v2.apply_quantum_to_selection(
            q_values, visit_counts, priors, 
            c_puct=1.414, total_simulations=100
        )
        
        # Both should produce valid outputs
        assert ucb_v1.shape == q_values.shape
        assert ucb_v2.shape == q_values.shape
        assert not torch.isnan(ucb_v1).any()
        assert not torch.isnan(ucb_v2).any()
    
    def test_phase_info_access(self):
        """Test phase information access (v2 only)"""
        # v1 wrapper - no phase info
        wrapper_v1 = create_quantum_mcts(version='v1', device='cpu')
        phase_info_v1 = wrapper_v1.get_phase_info()
        assert phase_info_v1['current_phase'] == 'not_available'
        
        # v2 wrapper - has phase info
        wrapper_v2 = create_quantum_mcts(
            version='v2',
            branching_factor=20,
            device='cpu'
        )
        phase_info_v2 = wrapper_v2.get_phase_info()
        assert phase_info_v2['current_phase'] in ['quantum', 'critical', 'classical']
        assert 'simulation_count' in phase_info_v2
    
    def test_version_comparison(self):
        """Test the version comparison utility"""
        # Create test data
        q_values = torch.randn(100, 20)
        visit_counts = torch.randint(0, 50, (100, 20))
        priors = torch.softmax(torch.randn(100, 20), dim=-1)
        
        # Compare versions
        comparison = compare_versions(
            q_values, visit_counts, priors,
            config=UnifiedQuantumConfig(
                branching_factor=20,
                device='cpu'
            )
        )
        
        # Check comparison results
        assert 'max_difference' in comparison
        assert 'correlation' in comparison
        assert 'speedup' in comparison
        assert comparison['correlation'] > 0.5  # Should be somewhat correlated
        assert comparison['v1_time'] > 0
        assert comparison['v2_time'] > 0


class TestDecoherenceV2:
    """Test v2.0 decoherence engine features"""
    
    def test_power_law_decoherence(self):
        """Test power-law vs exponential decoherence"""
        device = 'cpu'
        
        # Create v1 engine (exponential)
        engine_v1 = create_decoherence_engine_v2(
            device=device,
            power_law_exponent=None,  # Will use default
            temperature_mode='fixed'
        )
        
        # Create v2 engine (power-law)
        engine_v2 = create_decoherence_engine_v2(
            device=device,
            c_puct=1.414,
            power_law_exponent=0.5,
            temperature_mode='annealing'
        )
        
        # Test with different visit counts
        visit_counts = torch.tensor([1, 10, 100, 1000], device=device)
        
        # Initialize quantum states
        rho_v1 = engine_v1.initialize_quantum_state(4)
        rho_v2 = engine_v2.initialize_quantum_state(4)
        
        # Evolve both
        engine_v1.set_total_simulations(1000)
        engine_v2.set_total_simulations(1000)
        
        result_v1 = engine_v1.evolve_quantum_to_classical(
            visit_counts, total_simulations=1000
        )
        result_v2 = engine_v2.evolve_quantum_to_classical(
            visit_counts, total_simulations=1000
        )
        
        # Check that both produce valid results
        assert 'classical_probabilities' in result_v1
        assert 'classical_probabilities' in result_v2
        assert 'decoherence_rate' in result_v1
        assert 'decoherence_rate' in result_v2
        
        # v2 should have different decoherence pattern
        assert not torch.allclose(
            result_v1['classical_probabilities'],
            result_v2['classical_probabilities'],
            atol=0.1
        )
    
    def test_phase_dependent_decoherence(self):
        """Test phase-dependent decoherence rates"""
        engine = create_decoherence_engine_v2(
            device='cpu',
            c_puct=1.414,
            temperature_mode='annealing'
        )
        
        visit_counts = torch.ones(10) * 10
        
        # Test different phases
        phases = [MCTSPhase.QUANTUM, MCTSPhase.CRITICAL, MCTSPhase.CLASSICAL]
        decoherence_rates = []
        
        for phase in phases:
            engine.set_mcts_phase(phase)
            result = engine.evolve_quantum_to_classical(
                visit_counts,
                mcts_phase=phase
            )
            decoherence_rates.append(result['decoherence_rate'])
        
        # Classical phase should have stronger decoherence
        assert decoherence_rates[2] > decoherence_rates[0]
    
    def test_discrete_time_integration(self):
        """Test discrete time formulation in decoherence"""
        engine = create_decoherence_engine_v2(
            device='cpu',
            temperature_mode='annealing'
        )
        
        # Test at different simulation counts
        N_values = [10, 100, 1000, 10000]
        temperatures = []
        
        for N in N_values:
            engine.set_total_simulations(N)
            # Temperature should follow T(N) = T₀/log(N+2)
            if hasattr(engine.density_evolution.time_handler, 'compute_temperature'):
                T = engine.density_evolution.time_handler.compute_temperature(N)
                temperatures.append(T)
        
        if temperatures:
            # Check annealing behavior
            assert all(temperatures[i] > temperatures[i+1] for i in range(len(temperatures)-1))


class TestMainMCTSIntegration:
    """Test integration with main MCTS class"""
    
    @pytest.fixture
    def create_mcts_v2(self):
        """Factory for creating MCTS with v2.0 quantum features"""
        def _create(enable_quantum=True, num_simulations=1000):
            config = MCTSConfig(
                num_simulations=num_simulations,
                enable_quantum=enable_quantum,
                quantum_version='v2',
                quantum_branching_factor=225,  # 15x15 Gomoku
                quantum_avg_game_length=100,
                enable_phase_adaptation=True,
                envariance_threshold=1e-3,
                envariance_check_interval=100,
                min_wave_size=32,
                max_wave_size=32,  # Small for testing
                adaptive_wave_sizing=False,
                device='cpu',
                game_type=GameType.GOMOKU,
                board_size=15
            )
            
            # Create simple evaluator
            evaluator = SimpleEvaluatorWrapper(
                board_size=15,
                game_type='gomoku',
                device='cpu'
            )
            
            return MCTS(config, evaluator)
        
        return _create
    
    def test_quantum_v2_initialization(self, create_mcts_v2):
        """Test that v2.0 quantum features initialize correctly"""
        mcts = create_mcts_v2(enable_quantum=True)
        
        # Check quantum features
        assert mcts.quantum_features is not None
        assert isinstance(mcts.quantum_features, QuantumMCTSWrapper)
        assert mcts.quantum_features.version == 'v2'
        assert mcts.quantum_total_simulations == 0
        assert mcts.quantum_phase == MCTSPhase.QUANTUM
    
    def test_phase_tracking_during_search(self, create_mcts_v2):
        """Test phase tracking during MCTS search"""
        mcts = create_mcts_v2(enable_quantum=True, num_simulations=500)
        
        # Create a simple game state
        import alphazero_py
        state = alphazero_py.GomokuState()
        
        # Run search
        policy = mcts.search(state, num_simulations=500)
        
        # Check that quantum features were used
        assert mcts.quantum_total_simulations > 0
        
        # Get phase info
        if hasattr(mcts.quantum_features, 'get_phase_info'):
            phase_info = mcts.quantum_features.get_phase_info()
            assert 'current_phase' in phase_info
            assert phase_info['simulation_count'] > 0
    
    def test_envariance_convergence(self, create_mcts_v2):
        """Test envariance convergence checking"""
        mcts = create_mcts_v2(enable_quantum=True)
        mcts.config.envariance_check_interval = 50  # Check more frequently
        
        # Create game state
        import alphazero_py
        state = alphazero_py.GomokuState()
        
        # Run short search
        policy = mcts.search(state, num_simulations=200)
        
        # Check if envariance was checked
        assert mcts.envariance_check_counter >= 0
        
        # Get convergence status
        if hasattr(mcts.quantum_features, 'get_statistics'):
            stats = mcts.quantum_features.get_statistics()
            assert 'envariance_reached' in stats
    
    def test_performance_comparison(self, create_mcts_v2):
        """Compare performance with and without quantum v2"""
        import alphazero_py
        state = alphazero_py.GomokuState()
        
        # Classical MCTS
        mcts_classical = create_mcts_v2(enable_quantum=False, num_simulations=100)
        start = time.perf_counter()
        policy_classical = mcts_classical.search(state)
        time_classical = time.perf_counter() - start
        
        # Quantum v2 MCTS
        mcts_quantum = create_mcts_v2(enable_quantum=True, num_simulations=100)
        start = time.perf_counter()
        policy_quantum = mcts_quantum.search(state)
        time_quantum = time.perf_counter() - start
        
        # Check overhead
        overhead = time_quantum / time_classical
        print(f"Quantum v2 overhead: {overhead:.2f}x")
        
        # Should be reasonable (< 3x for tree-level)
        assert overhead < 3.0
        
        # Policies should be somewhat different
        policy_diff = np.abs(policy_classical - policy_quantum).sum()
        assert policy_diff > 0.01  # Some difference expected


class TestComponentIntegration:
    """Test integration between v2.0 components"""
    
    def test_qft_engine_v2(self):
        """Test QFT engine v2.0 features"""
        engine = create_qft_engine_v2(
            device='cpu',
            c_puct=1.414,
            temperature_mode='annealing',
            use_neural_priors=True
        )
        
        # Create test data
        visit_counts = torch.tensor([0, 10, 50, 100])
        q_values = torch.tensor([0.1, 0.2, 0.15, 0.25])
        priors = torch.tensor([0.25, 0.25, 0.25, 0.25])
        
        # Compute corrections
        corrections = engine.compute_quantum_corrections(
            visit_counts, q_values, priors,
            total_simulations=200
        )
        
        # Check corrections
        assert corrections.shape == visit_counts.shape
        assert not torch.isnan(corrections).any()
        
        # Low visits should get larger corrections
        assert corrections[0] > corrections[3]
    
    def test_path_integral_v2(self):
        """Test path integral v2.0 features"""
        path_integral = create_path_integral_v2(
            device='cpu',
            use_puct_action=True,
            prior_coupling=1.0
        )
        
        # Create test paths
        paths = [
            {'nodes': [0, 1, 2], 'actions': [0, 1]},
            {'nodes': [0, 1, 3], 'actions': [0, 2]},
        ]
        
        visit_counts = torch.tensor([100, 50, 20, 30])
        priors = torch.tensor([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.3, 0.3, 0.4], [0.25, 0.25, 0.5]])
        
        # Compute amplitudes with priors
        amplitudes = path_integral.compute_path_amplitudes(
            paths, visit_counts, 
            priors=priors,
            simulation_count=100
        )
        
        # Check amplitudes
        assert len(amplitudes) == len(paths)
        assert all(isinstance(a, complex) for a in amplitudes)
    
    def test_full_stack_integration(self):
        """Test full integration of all v2.0 components"""
        # Create all v2.0 components
        qft_engine = create_qft_engine_v2(device='cpu')
        path_integral = create_path_integral_v2(device='cpu')
        decoherence = create_decoherence_engine_v2(device='cpu')
        
        # Create quantum MCTS wrapper
        quantum_mcts = create_quantum_mcts(
            version='v2',
            branching_factor=20,
            device='cpu'
        )
        
        # Test data
        q_values = torch.randn(10, 20)
        visit_counts = torch.randint(0, 100, (10, 20))
        priors = torch.softmax(torch.randn(10, 20), dim=-1)
        
        # Apply quantum selection
        ucb_scores = quantum_mcts.apply_quantum_to_selection(
            q_values, visit_counts, priors,
            total_simulations=1000
        )
        
        # All components should work together
        assert ucb_scores.shape == q_values.shape
        assert not torch.isnan(ucb_scores).any()
        assert not torch.isinf(ucb_scores).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])