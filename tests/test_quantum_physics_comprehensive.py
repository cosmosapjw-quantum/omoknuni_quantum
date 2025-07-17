"""
Comprehensive quantum physics tests for MCTS implementation.

This consolidates all quantum physics tests into a single, well-organized test suite.
"""

import pytest
import torch
import numpy as np
from typing import List, Dict, Any

# Import quantum modules
from python.mcts.quantum.quantum_definitions import (
    UnifiedQuantumDefinitions,
    MCTSQuantumState,
    construct_quantum_state_from_ensemble
)
from python.mcts.quantum.wave_based_quantum_state import WaveBasedQuantumState
from python.mcts.quantum.phenomena import (
    DecoherenceAnalyzer,
    EntanglementAnalyzer,
    TunnelingAnalyzer,
    ThermodynamicsExtractor,
    FluctuationDissipationAnalyzer
)


class TestQuantumDefinitions:
    """Test unified quantum definitions and basic quantum state operations"""
    
    def test_quantum_state_construction(self):
        """Test construction of quantum states from visit distributions"""
        definitions = UnifiedQuantumDefinitions()
        
        # Test single distribution
        visits = torch.tensor([10, 5, 3, 2], dtype=torch.float32)
        state = definitions.construct_quantum_state_from_single_visits(visits)
        
        assert state.n_actions == 4
        assert state.n_outcomes == 1
        assert torch.allclose(torch.trace(state.density_matrix), torch.tensor(1.0))
        
    def test_ensemble_state_construction(self):
        """Test construction from ensemble of distributions"""
        definitions = UnifiedQuantumDefinitions()
        
        # Multiple visit distributions
        distributions = [
            torch.tensor([10, 5, 3], dtype=torch.float32),
            torch.tensor([8, 7, 4], dtype=torch.float32),
            torch.tensor([6, 9, 5], dtype=torch.float32)
        ]
        weights = torch.tensor([0.5, 0.3, 0.2])
        
        state = definitions.construct_quantum_state_from_visit_ensemble(distributions, weights)
        
        assert state.n_actions == 3
        assert state.n_outcomes == 3
        assert torch.allclose(torch.sum(state.ensemble_weights), torch.tensor(1.0))
        
    def test_von_neumann_entropy(self):
        """Test von Neumann entropy calculation"""
        definitions = UnifiedQuantumDefinitions()
        
        # Pure state (zero entropy)
        pure_state = torch.eye(4) * 0
        pure_state[0, 0] = 1.0
        entropy_pure = definitions.compute_von_neumann_entropy(pure_state)
        assert entropy_pure < 1e-6
        
        # Maximally mixed state (maximum entropy)
        mixed_state = torch.eye(4) / 4
        entropy_mixed = definitions.compute_von_neumann_entropy(mixed_state)
        expected_max = np.log(4)
        assert abs(entropy_mixed - expected_max) < 0.1
        
    def test_quantum_consistency(self):
        """Test quantum state consistency checks"""
        definitions = UnifiedQuantumDefinitions()
        
        visits = torch.tensor([10, 5, 3, 2], dtype=torch.float32)
        state = definitions.construct_quantum_state_from_single_visits(visits)
        
        results = definitions.validate_quantum_consistency(state)
        assert results['valid']
        assert abs(results['purity'] - 1.0) < 0.1  # Nearly pure state


class TestWaveBasedQuantumState:
    """Test wave-based quantum state construction from MCTS dynamics"""
    
    def test_wave_construction(self):
        """Test wave-based state construction"""
        constructor = WaveBasedQuantumState()
        
        # Simulate MCTS trajectory
        trajectory = []
        for t in range(10):
            visits = torch.randint(1, 100, (5,)).float()
            trajectory.append({
                'visits': visits,
                'node_id': t,
                'depth': t
            })
        
        state = constructor.construct_from_trajectory(trajectory)
        
        assert state is not None
        assert state.n_actions == 5
        assert torch.allclose(torch.trace(state.density_matrix), torch.tensor(1.0))
        
    def test_decoherence_dynamics(self):
        """Test natural decoherence from wave construction"""
        constructor = WaveBasedQuantumState()
        
        # Create trajectory with increasing certainty
        trajectory = []
        for t in range(20):
            # Visits become more concentrated over time
            visits = torch.zeros(5)
            visits[0] = 10 + t * 5  # Dominant action
            visits[1:] = torch.randint(1, 5, (4,)).float()
            
            trajectory.append({
                'visits': visits,
                'node_id': t,
                'depth': t
            })
        
        states = constructor.extract_decoherence_dynamics(trajectory)
        
        # Entropy should decrease (decoherence)
        entropies = [s['entropy'] for s in states]
        assert entropies[0] > entropies[-1]
        
        # Purity should increase
        purities = [s['purity'] for s in states]
        assert purities[0] < purities[-1]


class TestQuantumPhenomena:
    """Test quantum phenomena extractors"""
    
    @pytest.fixture
    def sample_trajectory(self):
        """Create sample MCTS trajectory for testing"""
        trajectory = []
        for t in range(50):
            visits = torch.randint(1, 100, (10,)).float()
            q_values = torch.randn(10) * 0.5
            
            trajectory.append({
                'visits': visits,
                'q_values': q_values,
                'node_id': t,
                'depth': t % 10,
                'parent_id': max(0, t - 1)
            })
        return trajectory
    
    def test_decoherence_analysis(self, sample_trajectory):
        """Test decoherence analyzer"""
        analyzer = DecoherenceAnalyzer()
        results = analyzer.analyze_trajectory(sample_trajectory)
        
        assert 'decoherence_rate' in results
        assert 'coherence_time' in results
        assert 'entropy_evolution' in results
        assert len(results['entropy_evolution']) > 0
        
    def test_entanglement_analysis(self, sample_trajectory):
        """Test entanglement analyzer"""
        analyzer = EntanglementAnalyzer()
        results = analyzer.analyze_trajectory(sample_trajectory)
        
        assert 'entanglement_entropy' in results
        assert 'mutual_information' in results
        assert results['entanglement_entropy'] >= 0
        
    def test_tunneling_analysis(self, sample_trajectory):
        """Test tunneling analyzer"""
        analyzer = TunnelingAnalyzer()
        results = analyzer.analyze_trajectory(sample_trajectory)
        
        assert 'tunneling_events' in results
        assert 'tunneling_rate' in results
        assert isinstance(results['tunneling_events'], list)
        
    def test_thermodynamics_extraction(self, sample_trajectory):
        """Test thermodynamics extractor"""
        extractor = ThermodynamicsExtractor()
        results = extractor.extract_from_trajectory(sample_trajectory)
        
        assert 'temperature' in results
        assert 'free_energy' in results
        assert 'entropy' in results
        assert results['temperature'] > 0
        
    def test_fluctuation_dissipation(self, sample_trajectory):
        """Test fluctuation-dissipation analyzer"""
        analyzer = FluctuationDissipationAnalyzer()
        results = analyzer.analyze_trajectory(sample_trajectory)
        
        assert 'fdt_violation' in results
        assert 'response_function' in results
        assert 'correlation_function' in results


class TestQuantumMCTSIntegration:
    """Test integration with MCTS system"""
    
    def test_mcts_quantum_state_extraction(self):
        """Test extracting quantum states from MCTS nodes"""
        # Create mock MCTS node data
        visits = torch.tensor([100, 50, 30, 20, 10], dtype=torch.float32)
        
        # Use quantum definitions
        definitions = UnifiedQuantumDefinitions()
        state = definitions.construct_quantum_state_from_single_visits(visits)
        
        # Verify quantum properties
        entropy = definitions.compute_von_neumann_entropy(state.density_matrix)
        purity = definitions.compute_purity(state.density_matrix)
        coherence = definitions.compute_coherence(state.density_matrix)
        
        assert entropy >= 0
        assert 0 <= purity <= 1
        assert coherence >= 0
        
    def test_batch_quantum_operations(self):
        """Test batch processing of quantum operations"""
        definitions = UnifiedQuantumDefinitions(device='cpu')
        
        # Create batch of visit distributions
        batch_size = 10
        action_size = 5
        batch_visits = torch.rand(batch_size, action_size) * 100
        
        # Process batch
        states = []
        for i in range(batch_size):
            state = definitions.construct_quantum_state_from_single_visits(batch_visits[i])
            states.append(state)
        
        # Verify all states are valid
        for state in states:
            assert torch.allclose(torch.trace(state.density_matrix), torch.tensor(1.0))


class TestQuantumPerformance:
    """Test performance aspects of quantum computations"""
    
    def test_gpu_acceleration(self):
        """Test GPU acceleration if available"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        definitions = UnifiedQuantumDefinitions(device=device)
        
        # Large state
        visits = torch.rand(100, device=device) * 1000
        state = definitions.construct_quantum_state_from_single_visits(visits)
        
        assert state.density_matrix.device.type == device
        assert state.n_actions == 100
        
    def test_memory_efficiency(self):
        """Test memory-efficient operations"""
        definitions = UnifiedQuantumDefinitions()
        
        # Test with large action space
        large_visits = torch.rand(1000) * 100
        state = definitions.construct_quantum_state_from_single_visits(large_visits)
        
        # Should handle large states without error
        entropy = definitions.compute_von_neumann_entropy(state.density_matrix)
        assert entropy >= 0
        
    def test_numerical_stability(self):
        """Test numerical stability with edge cases"""
        definitions = UnifiedQuantumDefinitions()
        
        # Test with very small probabilities
        visits = torch.tensor([1e-6, 1e-6, 1.0, 1e-6])
        state = definitions.construct_quantum_state_from_single_visits(visits)
        
        # Should handle without numerical errors
        entropy = definitions.compute_von_neumann_entropy(state.density_matrix)
        assert not torch.isnan(entropy)
        assert not torch.isinf(entropy)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])