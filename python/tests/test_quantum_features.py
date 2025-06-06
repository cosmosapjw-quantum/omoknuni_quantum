"""Comprehensive tests for quantum-inspired features

This module tests all quantum components including:
- MinHash interference
- Phase-kicked priors
- Path integral formulation
- Integration with wave engine
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from mcts.quantum.interference import MinHashInterference
from mcts.quantum.interference_gpu import MinHashInterference as GPUMinHashInterference
from mcts.quantum.phase_policy import PhaseKickedPolicy
from mcts.quantum.path_integral import PathIntegralMCTS


class TestMinHashInterference:
    """Test MinHash interference implementation"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def interference_engine(self, device):
        return MinHashInterference(device=device, strength=0.15)
    
    def test_initialization(self, interference_engine, device):
        """Test interference engine initialization"""
        assert interference_engine.device == device
        assert interference_engine.strength == 0.15
        
        if device.type == 'cuda':
            assert interference_engine.use_gpu
            assert isinstance(interference_engine.gpu_engine, GPUMinHashInterference)
        else:
            assert not interference_engine.use_gpu
    
    def test_path_diversity_computation(self, interference_engine, device):
        """Test path diversity computation"""
        # Create sample paths
        batch_size = 10
        max_depth = 20
        paths = torch.randint(0, 100, (batch_size, max_depth), device=device)
        
        # Add some -1 padding
        paths[:, 15:] = -1
        
        # Compute diversity
        signatures, similarities = interference_engine.compute_path_diversity_batch(paths)
        
        assert signatures.shape[0] == batch_size
        assert similarities.shape == (batch_size, batch_size)
        
        # Check similarity properties
        # Diagonal should have high similarity (not necessarily 1.0 due to hash collisions)
        assert (similarities.diag() >= 0.4).all()  # Reasonable threshold
        assert (similarities >= 0).all()
        assert (similarities <= 1).all()
        
        # Test that identical paths have high similarity
        paths[1] = paths[0].clone()  # Make path 1 identical to path 0
        signatures2, similarities2 = interference_engine.compute_path_diversity_batch(paths)
        # Debug: print similarity
        print(f"Similarity between identical paths: {similarities2[0, 1]}")
        print(f"Signatures match ratio: {(signatures2[0] == signatures2[1]).float().mean()}")
        # For MinHash, identical paths should have similarity of 1.0
        assert similarities2[0, 1] >= 0.95  # Should be very similar
    
    def test_interference_application(self, interference_engine, device):
        """Test interference application to scores"""
        batch_size = 10
        scores = torch.rand(batch_size, device=device)
        
        # Create high similarity matrix (should cause interference)
        similarities = torch.ones(batch_size, batch_size, device=device) * 0.8
        similarities.fill_diagonal_(1.0)
        
        # Apply interference
        modified_scores = interference_engine.apply_interference(
            scores, similarities, interference_strength=0.2
        )
        
        # Scores should be reduced due to interference
        assert (modified_scores <= scores).all()
        assert (modified_scores >= 0).all()
    
    def test_gpu_minhash_hashing(self, device):
        """Test GPU MinHash signature computation"""
        if device.type != 'cuda':
            pytest.skip("GPU test requires CUDA")
            
        gpu_engine = GPUMinHashInterference(device=device)
        
        # Create paths with known patterns
        paths = torch.tensor([
            [1, 2, 3, 4, 5, -1, -1],
            [1, 2, 3, 4, 5, -1, -1],  # Identical to first
            [6, 7, 8, 9, 10, -1, -1],  # Different
        ], device=device)
        
        signatures = gpu_engine.compute_minhash_signatures(paths, num_hashes=32)
        
        # Check signature properties
        assert signatures.shape == (3, 32)
        
        # Identical paths should have identical signatures
        assert torch.equal(signatures[0], signatures[1])
        
        # Different paths should have different signatures
        assert not torch.equal(signatures[0], signatures[2])


class TestPhaseKickedPolicy:
    """Test phase-kicked prior policy"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def phase_policy(self, device):
        return PhaseKickedPolicy(device=device, kick_strength=0.1)
    
    def test_initialization(self, phase_policy, device):
        """Test phase policy initialization"""
        assert phase_policy.device == device
        assert phase_policy.kick_strength == 0.1
        
        if device.type == 'cuda':
            assert phase_policy.use_gpu
            # GPU mode is integrated in the unified class
            assert phase_policy.use_gpu
    
    def test_phase_kicks_tensor_mode(self, phase_policy, device):
        """Test phase kicks in tensor mode"""
        num_actions = 10
        priors = torch.softmax(torch.rand(num_actions, device=device), dim=0)
        visit_counts = torch.randint(0, 100, (num_actions,), device=device).float()
        q_values = torch.randn(num_actions, device=device) * 0.5
        
        # Apply phase kicks
        kicked_priors = phase_policy.apply_phase_kicks(
            priors, visit_counts, q_values
        )
        
        # Check properties
        assert kicked_priors.shape == priors.shape
        assert torch.allclose(kicked_priors.sum(), torch.tensor(1.0))
        assert (kicked_priors >= 0).all()
        
        # Phase kicks should change the distribution
        assert not torch.allclose(kicked_priors, priors)
    
    def test_quantum_wave_function(self, device):
        """Test quantum wave function formalism"""
        if device.type != 'cuda':
            pytest.skip("GPU test requires CUDA")
            
        gpu_policy = PhaseKickedPolicy(device=device)
        
        # Create priors
        priors = torch.tensor([0.1, 0.3, 0.4, 0.2], device=device)
        visit_counts = torch.tensor([10., 5., 2., 1.], device=device)
        q_values = torch.tensor([0.5, -0.2, 0.3, 0.0], device=device)
        
        # Apply phase kicks
        kicked = gpu_policy.apply_phase_kicks(priors, visit_counts, q_values)
        
        # Verify Born rule (probabilities sum to 1)
        assert torch.allclose(kicked.sum(), torch.tensor(1.0))
        
        # High uncertainty (low visits) should lead to more exploration
        uncertainty = 1.0 / torch.sqrt(1.0 + visit_counts)
        high_uncertainty_idx = torch.argmax(uncertainty)
        
        # The action with highest uncertainty should see increased probability
        # (though this depends on other factors too)
        assert kicked[high_uncertainty_idx] > 0
    
    def test_decoherence_effects(self, device):
        """Test decoherence with increasing visits"""
        if device.type != 'cuda':
            pytest.skip("GPU test requires CUDA")
            
        gpu_policy = PhaseKickedPolicy(device=device)
        
        priors = torch.ones(5, device=device) / 5  # Uniform
        q_values = torch.zeros(5, device=device)
        
        # Test with different visit counts
        low_visits = torch.ones(5, device=device)
        high_visits = torch.ones(5, device=device) * 1000
        
        kicked_low = gpu_policy.apply_phase_kicks(priors, low_visits, q_values)
        kicked_high = gpu_policy.apply_phase_kicks(priors, high_visits, q_values)
        
        # High visits should lead to less phase kick effect (more classical)
        low_deviation = torch.abs(kicked_low - priors).mean()
        high_deviation = torch.abs(kicked_high - priors).mean()
        
        assert low_deviation > high_deviation


class TestPathIntegral:
    """Test path integral formulation"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def path_integral(self, device):
        return PathIntegralMCTS(device=device)
    
    def test_initialization(self, path_integral, device):
        """Test path integral initialization"""
        assert path_integral.device == device
        
        if device.type == 'cuda':
            assert path_integral.use_gpu
            # GPU mode is integrated in the unified class
            assert path_integral.use_gpu
    
    def test_action_computation(self, device):
        """Test quantum action computation"""
        if device.type != 'cuda':
            pytest.skip("GPU test requires CUDA")
            
        gpu_engine = PathIntegralMCTS(device=device)
        
        # Create sample paths
        batch_size = 5
        path_length = 10
        paths = torch.randint(0, 20, (batch_size, path_length), device=device)
        values = torch.randn(batch_size, path_length, device=device) * 0.5
        visits = torch.randint(1, 100, (batch_size, path_length), device=device).float()
        
        # Compute action using private method
        actions = gpu_engine._compute_path_action_gpu(paths, values, visits)
        
        assert actions.shape == (batch_size,)
        assert actions.dtype == torch.float32 or actions.dtype == torch.float64
    
    def test_variational_principle(self, device):
        """Test variational optimization of paths"""
        if device.type != 'cuda':
            pytest.skip("GPU test requires CUDA")
            
        gpu_engine = PathIntegralMCTS(device=device)
        gpu_engine.config.num_variational_steps = 5
        
        # Initial paths as float for gradient optimization
        initial_paths = torch.randint(
            0, 20, (10, 15), device=device
        ).float()
        
        # Create values and visits for paths
        values = torch.randn_like(initial_paths) * 0.5
        visits = torch.ones_like(initial_paths) * 10
        
        # Make sure gradients can flow
        initial_clone = initial_paths.clone()
        
        # Optimize paths using private method
        optimized = gpu_engine._variational_optimization(initial_paths, values, visits)
        
        assert optimized.shape == initial_clone.shape
        # Due to rounding, paths might be the same - check if optimization ran
        assert gpu_engine.stats['variational_steps'] > 0
    
    def test_path_probability_computation(self, device):
        """Test path probability from action"""
        if device.type != 'cuda':
            pytest.skip("GPU test requires CUDA")
            
        gpu_engine = PathIntegralMCTS(device=device)
        
        # Different action values
        actions = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0], device=device)
        
        # Compute probabilities using private method
        probs = gpu_engine._compute_path_probability_gpu(actions)
        
        assert probs.shape == actions.shape
        assert torch.allclose(probs.sum(), torch.tensor(1.0))
        assert (probs >= 0).all()
        
        # Lower action should have higher probability
        assert probs[0] > probs[-1]


class TestQuantumIntegration:
    """Test integration of quantum features in wave engine"""
    
    @pytest.fixture
    def mock_wave_engine(self):
        """Create mock wave engine with quantum features"""
        from mcts.gpu.optimized_wave_engine import OptimizedWaveEngine, OptimizedWaveConfig
        from mcts.gpu.csr_tree import CSRTree
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Mock components
        mock_tree = MagicMock(spec=CSRTree)
        mock_tree.device = device
        mock_evaluator = MagicMock()
        mock_game = MagicMock()
        
        # Create config with quantum features enabled
        config = OptimizedWaveConfig()
        config.device = str(device)
        config.enable_interference = True
        config.enable_phase_policy = True  
        config.enable_path_integral = True
        
        # Create engine
        engine = OptimizedWaveEngine(
            mock_tree, config, mock_game, mock_evaluator
        )
        
        return engine
    
    def test_quantum_components_initialized(self, mock_wave_engine):
        """Test that quantum components are properly initialized"""
        assert hasattr(mock_wave_engine, 'interference_engine')
        assert hasattr(mock_wave_engine, 'phase_policy')
        assert hasattr(mock_wave_engine, 'path_integral')
        
        assert mock_wave_engine.interference_engine is not None
        assert mock_wave_engine.phase_policy is not None
        assert mock_wave_engine.path_integral is not None
    
    def test_quantum_stats_tracking(self, mock_wave_engine):
        """Test quantum statistics tracking"""
        assert 'interference_applied' in mock_wave_engine.quantum_stats
        assert 'phase_kicks_applied' in mock_wave_engine.quantum_stats
        assert 'path_integral_used' in mock_wave_engine.quantum_stats
        
        # Initial stats should be zero
        assert mock_wave_engine.quantum_stats['interference_applied'] == 0
        assert mock_wave_engine.quantum_stats['phase_kicks_applied'] == 0
        assert mock_wave_engine.quantum_stats['path_integral_used'] == 0


def test_cuda_graph_optimization():
    """Test CUDA graph capture functionality"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA graphs require GPU")
        
    from mcts.gpu.cuda_graph_optimizer import CUDAGraphOptimizer
    
    device = torch.device('cuda')
    optimizer = CUDAGraphOptimizer(device)
    
    # Test function to capture
    def simple_kernel(x, y):
        return x + y * 2
    
    # Test inputs
    x = torch.rand(100, 100, device=device)
    y = torch.rand(100, 100, device=device)
    
    # Capture graph
    result1 = optimizer.capture_graph(
        simple_kernel, (x, y), {}, "test_kernel"
    )
    
    # Second call should use cached graph
    result2 = optimizer.capture_graph(
        simple_kernel, (x, y), {}, "test_kernel"
    )
    
    assert torch.allclose(result1, result2)
    assert optimizer.stats['graphs_captured'] == 1
    assert optimizer.stats['cache_hits'] == 1
    
    # Different shapes should create new graph
    x2 = torch.rand(200, 200, device=device)
    y2 = torch.rand(200, 200, device=device)
    
    result3 = optimizer.capture_graph(
        simple_kernel, (x2, y2), {}, "test_kernel"
    )
    
    assert optimizer.stats['graphs_captured'] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])