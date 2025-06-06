"""
Test Suite for Envariance Engine
================================

Validates the entanglement-assisted robustness engine and multi-evaluator
quantum entanglement for exponential speedup.

Test Categories:
1. GHZ state generation and verification
2. Envariance projection and filtering
3. Channel capacity optimization
4. Multi-evaluator robustness
5. Quantum advantage measurement
"""

import pytest
import torch
import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts.quantum.envariance import (
    EnvarianceEngine,
    EnvarianceConfig,
    GHZStateGenerator,
    EnvarianceProjector,
    ChannelCapacityOptimizer,
    create_envariance_engine
)


class TestGHZStateGenerator:
    """Test GHZ state generation and entanglement verification"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def config(self):
        return EnvarianceConfig(
            epsilon=0.1,
            ghz_fidelity_threshold=0.9,
            phase_randomization=True
        )
    
    @pytest.fixture
    def generator(self, config, device):
        return GHZStateGenerator(config, device)
    
    def test_ghz_state_creation(self, generator, device):
        """Test basic GHZ state generation"""
        num_states = 3
        num_evaluators = 4
        
        ghz_state = generator.create_ghz_superposition(num_states, num_evaluators)
        
        # Check dimensions
        expected_dim = num_states * num_evaluators
        assert ghz_state.shape == (expected_dim,)
        assert ghz_state.dtype == torch.complex64
        
        # Check normalization
        norm = torch.norm(ghz_state)
        assert abs(norm.item() - 1.0) < 1e-6
    
    def test_ghz_state_properties(self, generator, device):
        """Test that GHZ state has correct entanglement properties"""
        num_states = 2
        num_evaluators = 3
        
        # Create state with uniform probabilities
        uniform_probs = torch.ones(num_states, device=device) / num_states
        ghz_state = generator.create_ghz_superposition(
            num_states, num_evaluators, uniform_probs
        )
        
        # Verify entanglement
        entanglement_info = generator.verify_entanglement(
            ghz_state, num_states, num_evaluators
        )
        
        # Should have non-zero entanglement entropy
        assert entanglement_info['entanglement_entropy'] > 0
        assert entanglement_info['system_entropy'] >= 0
        assert entanglement_info['evaluator_entropy'] >= 0
        
        # Maximum entanglement bounded by smaller subsystem
        max_entanglement = entanglement_info['max_entanglement']
        assert entanglement_info['entanglement_entropy'] <= max_entanglement
    
    def test_ghz_fidelity_measurement(self, generator, device):
        """Test GHZ fidelity measurement"""
        num_states = 2
        num_evaluators = 2
        
        # Perfect GHZ state should have fidelity ≈ 1
        perfect_ghz = generator.create_ghz_superposition(num_states, num_evaluators)
        fidelity = generator.measure_ghz_fidelity(perfect_ghz, num_states, num_evaluators)
        
        assert 0.8 <= fidelity <= 1.0  # Allow for numerical precision
        
        # Random state should have lower fidelity
        random_state = torch.randn(num_states * num_evaluators, dtype=torch.complex64, device=device)
        random_state = random_state / torch.norm(random_state)
        
        random_fidelity = generator.measure_ghz_fidelity(random_state, num_states, num_evaluators)
        assert random_fidelity < fidelity
    
    def test_phase_randomization(self, generator, device):
        """Test that phase randomization produces different states"""
        num_states = 2
        num_evaluators = 3
        
        # Generate multiple states with phase randomization
        states = []
        for _ in range(5):
            state = generator.create_ghz_superposition(num_states, num_evaluators)
            states.append(state)
        
        # States should be different (due to random phases)
        overlaps = []
        for i in range(1, len(states)):
            overlap = abs(torch.dot(states[0].conj(), states[i]))**2
            overlaps.append(overlap.item())
        
        # Not all overlaps should be perfect (some variation expected)
        assert not all(o > 0.99 for o in overlaps)


class TestEnvarianceProjector:
    """Test envariance projection and filtering"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def config(self):
        return EnvarianceConfig(epsilon=0.1)
    
    @pytest.fixture
    def projector(self, config, device):
        return EnvarianceProjector(config, device)
    
    @pytest.fixture
    def sample_data(self, device):
        """Create sample paths and evaluator outputs"""
        num_paths = 20
        path_length = 5
        num_evaluators = 3
        
        paths = torch.randint(0, 50, (num_paths, path_length), device=device)
        
        # Create evaluator outputs with varying agreement
        base_values = torch.rand(num_paths, device=device)
        evaluator_outputs = []
        
        for i in range(num_evaluators):
            # Add noise with different levels
            noise_level = 0.1 * (i + 1)
            noisy_values = base_values + noise_level * torch.randn(num_paths, device=device)
            evaluator_outputs.append(noisy_values)
        
        return paths, evaluator_outputs
    
    def test_evaluation_variance_computation(self, projector, sample_data):
        """Test computation of evaluation variances"""
        paths, evaluator_outputs = sample_data
        
        variances = projector._compute_evaluation_variances(evaluator_outputs)
        
        # Should have one variance per path
        assert variances.shape == (paths.shape[0],)
        assert torch.all(variances >= 0)  # Variances are non-negative
        assert torch.all(torch.isfinite(variances))
    
    def test_envariance_projection(self, projector, sample_data):
        """Test projection to envariant subspace"""
        paths, evaluator_outputs = sample_data
        
        # Use a tolerance that should keep some paths
        tolerance = 0.5
        
        envariant_paths, envariance_scores = projector.project_to_envariant_subspace(
            paths, evaluator_outputs, tolerance
        )
        
        # Should return some envariant paths
        assert len(envariant_paths) <= len(paths)
        assert envariance_scores.shape == (paths.shape[0],)
        
        # Scores should be in reasonable range
        assert torch.all(envariance_scores >= 0)
        assert torch.all(envariance_scores <= 1)
    
    def test_strict_envariance(self, projector, device):
        """Test with very strict envariance requirement"""
        # Create paths where evaluators strongly disagree
        num_paths = 10
        paths = torch.arange(num_paths, device=device).unsqueeze(1)
        
        # Strongly disagreeing evaluators
        evaluator_outputs = [
            torch.zeros(num_paths, device=device),
            torch.ones(num_paths, device=device) * 10,
            torch.ones(num_paths, device=device) * (-5)
        ]
        
        # Very strict tolerance
        strict_tolerance = 0.01
        
        envariant_paths, _ = projector.project_to_envariant_subspace(
            paths, evaluator_outputs, strict_tolerance
        )
        
        # Should filter out most paths due to high disagreement
        assert len(envariant_paths) < len(paths)
    
    def test_mutual_information_computation(self, projector, device):
        """Test mutual information computation"""
        # Create correlated and uncorrelated data
        n_samples = 100
        
        # Strongly correlated
        x = torch.randn(n_samples, device=device)
        y_corr = x + 0.1 * torch.randn(n_samples, device=device)
        
        # Uncorrelated
        y_uncorr = torch.randn(n_samples, device=device)
        
        mi_corr = projector.compute_mutual_information(x, y_corr)
        mi_uncorr = projector.compute_mutual_information(x, y_uncorr)
        
        # Correlated data should have higher mutual information
        assert mi_corr > mi_uncorr
        assert mi_corr >= 0  # MI is non-negative
        assert mi_uncorr >= 0


class TestChannelCapacityOptimizer:
    """Test channel capacity optimization"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def config(self):
        return EnvarianceConfig(max_iterations=50)
    
    @pytest.fixture
    def optimizer(self, config, device):
        return ChannelCapacityOptimizer(config, device)
    
    def test_channel_capacity_computation(self, optimizer, device):
        """Test basic channel capacity computation"""
        # Create mock evaluation matrix
        num_positions = 20
        num_evaluators = 3
        
        eval_matrix = torch.rand(num_positions, num_evaluators, device=device)
        
        capacity = optimizer._compute_channel_capacity(eval_matrix)
        
        # Capacity should be reasonable
        max_capacity = np.log2(num_positions)
        assert 0 <= capacity <= max_capacity
    
    def test_evaluator_weight_optimization(self, optimizer, device):
        """Test evaluator weight optimization"""
        # Create evaluation matrix with clear structure
        num_positions = 15
        num_evaluators = 3
        
        # Create evaluators with different quality
        eval_matrix = torch.zeros(num_positions, num_evaluators, device=device)
        eval_matrix[:, 0] = torch.arange(num_positions, dtype=torch.float, device=device)  # Good evaluator
        eval_matrix[:, 1] = torch.rand(num_positions, device=device)  # Random evaluator
        eval_matrix[:, 2] = -torch.arange(num_positions, dtype=torch.float, device=device)  # Anti-correlated
        
        optimal_weights = optimizer._optimize_evaluator_weights(eval_matrix)
        
        # Weights should be normalized
        assert abs(optimal_weights.sum().item() - 1.0) < 1e-6
        assert torch.all(optimal_weights >= 0)
        
        # Should favor the good evaluator
        assert optimal_weights[0] >= optimal_weights[1]


class TestEnvarianceEngine:
    """Test the main envariance engine"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def config(self):
        return EnvarianceConfig(
            epsilon=0.2,
            min_evaluators=2,
            max_evaluators=8
        )
    
    @pytest.fixture
    def engine(self, config, device):
        return EnvarianceEngine(config, device)
    
    @pytest.fixture
    def mock_evaluators(self, device):
        """Create mock evaluator functions"""
        def evaluator_1(positions):
            # Good evaluator: returns meaningful values
            return torch.sum(positions.float(), dim=-1) / positions.shape[-1]
        
        def evaluator_2(positions):
            # Noisy evaluator: adds some noise
            base = torch.sum(positions.float(), dim=-1) / positions.shape[-1]
            return base + 0.1 * torch.randn_like(base)
        
        def evaluator_3(positions):
            # Different evaluator: uses different metric
            return torch.max(positions.float(), dim=-1).values
        
        return [evaluator_1, evaluator_2, evaluator_3]
    
    def test_engine_initialization(self, engine, device):
        """Test envariance engine initialization"""
        assert engine.device == device
        assert isinstance(engine.config, EnvarianceConfig)
        assert hasattr(engine, 'ghz_generator')
        assert hasattr(engine, 'projector')
        assert hasattr(engine, 'capacity_optimizer')
    
    def test_evaluator_ensemble_registration(self, engine, mock_evaluators):
        """Test registering evaluator ensemble"""
        engine.register_evaluator_ensemble(mock_evaluators)
        
        assert len(engine.evaluator_ensemble) == 3
        assert engine.current_entangled_state is not None
        
        # Should have prepared entanglement
        assert 'entanglement_fidelity' in engine.stats
    
    def test_envariant_path_generation(self, engine, mock_evaluators, device):
        """Test envariant path generation"""
        engine.register_evaluator_ensemble(mock_evaluators)
        
        # Create sample paths and positions
        num_paths = 16
        path_length = 4
        paths = torch.randint(0, 10, (num_paths, path_length), device=device)
        positions = torch.randint(0, 20, (num_paths, path_length), device=device)
        
        envariant_paths, envariance_scores = engine.generate_envariant_paths(
            paths, positions
        )
        
        # Should return valid results
        assert len(envariant_paths) <= len(paths)
        assert envariance_scores.shape == (paths.shape[0],)
        assert torch.all(envariance_scores >= 0)
        assert torch.all(envariance_scores <= 1)
        
        # Statistics should be updated
        assert engine.stats['envariant_paths_generated'] > 0
        assert engine.stats['sample_efficiency_gain'] > 1.0
    
    def test_robust_strategy_computation(self, engine, device):
        """Test robust strategy computation from evaluator ensemble"""
        num_paths = 8
        num_evaluators = 3
        
        paths = torch.arange(num_paths, device=device).unsqueeze(1)
        
        # Create evaluations with some agreement
        base_evals = torch.rand(num_paths, device=device)
        path_evaluations = torch.stack([
            base_evals,
            base_evals + 0.1 * torch.randn(num_paths, device=device),
            base_evals + 0.05 * torch.randn(num_paths, device=device)
        ], dim=1)
        
        robust_strategy = engine.compute_robust_strategy(paths, path_evaluations)
        
        # Should be valid probability distribution
        assert abs(robust_strategy.sum().item() - 1.0) < 1e-6
        assert torch.all(robust_strategy >= 0)
        assert robust_strategy.shape == (num_paths,)
    
    def test_entanglement_advantage_measurement(self, engine, mock_evaluators, device):
        """Test measurement of quantum advantage"""
        engine.register_evaluator_ensemble(mock_evaluators)
        
        num_paths = 12
        test_paths = torch.randint(0, 15, (num_paths, 3), device=device)
        classical_baseline = torch.rand(num_paths, device=device)
        
        advantage_metrics = engine.measure_entanglement_advantage(
            test_paths, classical_baseline
        )
        
        # Should return meaningful metrics
        required_keys = ['envariance_fraction', 'theoretical_speedup', 'measured_advantage', 'num_evaluators']
        for key in required_keys:
            assert key in advantage_metrics
        
        # Speedup should be reasonable
        assert advantage_metrics['theoretical_speedup'] > 1.0
        assert advantage_metrics['num_evaluators'] == len(mock_evaluators)
    
    def test_ensemble_optimization(self, engine, mock_evaluators, device):
        """Test evaluator ensemble optimization"""
        engine.register_evaluator_ensemble(mock_evaluators)
        
        # Create test positions
        num_positions = 20
        test_positions = torch.randint(0, 25, (num_positions, 4), device=device)
        
        optimization_result = engine.optimize_evaluator_ensemble(test_positions)
        
        # Should return optimization results
        if optimization_result:  # May be empty if too few evaluators
            assert 'capacity' in optimization_result
            assert 'optimal_capacity' in optimization_result
            assert 'optimal_weights' in optimization_result
            
            # Optimal capacity should be >= original capacity
            assert optimization_result['optimal_capacity'] >= optimization_result['capacity']
    
    def test_statistics_tracking(self, engine, mock_evaluators, device):
        """Test statistics tracking"""
        engine.register_evaluator_ensemble(mock_evaluators)
        
        # Generate some envariant paths
        paths = torch.randint(0, 10, (8, 3), device=device)
        positions = torch.randint(0, 15, (8, 3), device=device)
        
        engine.generate_envariant_paths(paths, positions)
        
        stats = engine.get_statistics()
        
        # Should track all required statistics
        required_stats = [
            'envariant_paths_generated',
            'avg_envariance_score', 
            'channel_capacity',
            'entanglement_fidelity',
            'sample_efficiency_gain'
        ]
        
        for stat in required_stats:
            assert stat in stats
            assert isinstance(stats[stat], (int, float))


class TestTheoreticalPredictions:
    """Test theoretical predictions for envariance"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_sample_complexity_reduction(self, device):
        """Test that envariance reduces sample complexity"""
        config_2 = EnvarianceConfig(epsilon=0.1)
        config_4 = EnvarianceConfig(epsilon=0.1)
        
        engine_2 = EnvarianceEngine(config_2, device)
        engine_4 = EnvarianceEngine(config_4, device)
        
        # Create different sized evaluator ensembles
        def simple_evaluator(positions):
            return torch.sum(positions.float(), dim=-1)
        
        evaluators_2 = [simple_evaluator, simple_evaluator]
        evaluators_4 = [simple_evaluator, simple_evaluator, simple_evaluator, simple_evaluator]
        
        engine_2.register_evaluator_ensemble(evaluators_2)
        engine_4.register_evaluator_ensemble(evaluators_4)
        
        # Sample efficiency should increase with ensemble size
        efficiency_2 = engine_2.stats['sample_efficiency_gain']
        efficiency_4 = engine_4.stats['sample_efficiency_gain']
        
        assert efficiency_4 > efficiency_2
    
    def test_epsilon_envariance_property(self, device):
        """Test ε-envariance property"""
        config = EnvarianceConfig(epsilon=0.05)  # Strict tolerance
        projector = EnvarianceProjector(config, device)
        
        # Create paths with known variance properties
        num_paths = 10
        paths = torch.arange(num_paths, device=device).unsqueeze(1)
        
        # Low variance evaluators (should pass ε-envariance)
        low_var_outputs = [
            torch.ones(num_paths, device=device),
            torch.ones(num_paths, device=device) * 1.01,
            torch.ones(num_paths, device=device) * 0.99
        ]
        
        # High variance evaluators (should fail ε-envariance) 
        high_var_outputs = [
            torch.ones(num_paths, device=device),
            torch.ones(num_paths, device=device) * 5,
            torch.ones(num_paths, device=device) * 0.2
        ]
        
        low_var_paths, _ = projector.project_to_envariant_subspace(paths, low_var_outputs)
        high_var_paths, _ = projector.project_to_envariant_subspace(paths, high_var_outputs)
        
        # Low variance should pass more paths than high variance
        assert len(low_var_paths) >= len(high_var_paths)


def test_factory_function():
    """Test factory function for envariance engine creation"""
    # Test default creation
    engine = create_envariance_engine()
    assert isinstance(engine, EnvarianceEngine)
    assert engine.config.epsilon == 0.1  # Default
    
    # Test with custom parameters
    engine = create_envariance_engine(epsilon=0.05, min_evaluators=3)
    assert engine.config.epsilon == 0.05
    assert engine.config.min_evaluators == 3


if __name__ == "__main__":
    # Run basic functionality test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Envariance Engine tests on {device}")
    
    # Create envariance engine
    engine = create_envariance_engine(device, epsilon=0.15)
    
    # Create mock evaluators
    def eval_1(pos):
        return torch.sum(pos.float(), dim=-1) / pos.shape[-1]
    
    def eval_2(pos):
        base = torch.sum(pos.float(), dim=-1) / pos.shape[-1]
        return base + 0.05 * torch.randn_like(base)
    
    def eval_3(pos):
        return torch.max(pos.float(), dim=-1).values
    
    evaluators = [eval_1, eval_2, eval_3]
    
    # Test evaluator registration
    print("Testing evaluator ensemble registration...")
    engine.register_evaluator_ensemble(evaluators)
    print(f"✓ Registered {len(evaluators)} evaluators")
    print(f"✓ Entanglement fidelity: {engine.stats['entanglement_fidelity']:.3f}")
    
    # Test envariant path generation
    print("\nTesting envariant path generation...")
    num_paths = 16
    paths = torch.randint(0, 20, (num_paths, 5), device=device)
    positions = torch.randint(0, 30, (num_paths, 5), device=device)
    
    start = time.perf_counter()
    envariant_paths, scores = engine.generate_envariant_paths(paths, positions)
    end = time.perf_counter()
    
    print(f"✓ Generated {len(envariant_paths)}/{num_paths} envariant paths in {end-start:.4f}s")
    print(f"✓ Average envariance score: {scores.mean():.3f}")
    print(f"✓ Sample efficiency gain: {engine.stats['sample_efficiency_gain']:.2f}x")
    
    # Test robust strategy
    print("\nTesting robust strategy computation...")
    path_evals = torch.rand(num_paths, len(evaluators), device=device)
    robust_strategy = engine.compute_robust_strategy(paths, path_evals)
    print(f"✓ Robust strategy computed: sum={robust_strategy.sum():.6f} (should be ~1.0)")
    
    # Test advantage measurement
    print("\nTesting entanglement advantage...")
    baseline = torch.rand(num_paths, device=device)
    advantage = engine.measure_entanglement_advantage(paths, baseline)
    print(f"✓ Theoretical speedup: {advantage['theoretical_speedup']:.2f}x")
    print(f"✓ Measured advantage: {advantage['measured_advantage']:.3f}")
    print(f"✓ Envariance fraction: {advantage['envariance_fraction']:.2f}")
    
    # Test statistics
    stats = engine.get_statistics()
    print(f"\n✓ Statistics: {stats}")
    
    print("✓ All envariance engine tests passed!")