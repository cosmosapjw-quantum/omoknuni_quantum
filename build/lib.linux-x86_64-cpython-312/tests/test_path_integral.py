"""Comprehensive tests for path integral MCTS formulation

This module tests the quantum path integral implementation including:
- Pre-computed table generation and lookups
- Path integral computation with caching
- Interference calculations
- Performance optimizations
"""

import pytest

# Skip entire module - quantum features are under development
pytestmark = pytest.mark.skip(reason="Quantum features are under development")
import torch
import numpy as np
from unittest.mock import Mock, patch
import logging

from mcts.quantum.path_integral import (
    PathIntegralConfig,
    PrecomputedTables,
    PathIntegral
)


class TestPathIntegralConfig:
    """Test configuration dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = PathIntegralConfig()
        
        assert config.hbar_eff == 0.1
        assert config.temperature == 1.0
        assert config.lambda_qft == 1.0
        assert config.use_lookup_tables == True
        assert config.table_size == 10000
        assert config.max_path_length == 50
        assert config.cache_path_actions == True
        assert config.use_mixed_precision == True
        assert config.batch_size == 256
        assert config.device == 'cuda'
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = PathIntegralConfig(
            hbar_eff=0.5,
            temperature=2.0,
            device='cpu',
            table_size=5000
        )
        
        assert config.hbar_eff == 0.5
        assert config.temperature == 2.0
        assert config.device == 'cpu'
        assert config.table_size == 5000


class TestPrecomputedTables:
    """Test pre-computed lookup tables"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return PathIntegralConfig(device='cpu', table_size=1000)
    
    @pytest.fixture
    def tables(self, config):
        """Create pre-computed tables"""
        return PrecomputedTables(config)
    
    def test_table_initialization(self, tables, config):
        """Test that all tables are properly initialized"""
        # Check length factors
        assert tables.length_factors is not None
        assert tables.length_factors.shape[0] == config.max_path_length
        assert tables.length_factors.device.type == 'cpu'
        
        # Check Boltzmann table
        assert tables.boltzmann_table is not None
        assert tables.boltzmann_table.shape[0] == config.table_size
        
        # Check quantum corrections
        assert tables.quantum_corrections is not None
        assert tables.quantum_corrections.shape[0] == 1000  # Fixed size in implementation
        
        # Check phase table
        assert tables.phase_table is not None
        assert tables.phase_table.shape[0] == 1000  # Fixed size in implementation
        
        # Check action cache
        assert tables.action_cache == {}  # Should be empty dict when caching enabled
    
    def test_length_factors_computation(self, config):
        """Test length factor pre-computation"""
        tables = PrecomputedTables(config)
        
        # Length factors should decay exponentially
        factors = tables.length_factors.cpu().numpy()
        assert factors[0] == pytest.approx(1.0)  # exp(0) = 1
        assert factors[1] < factors[0]  # Monotonically decreasing
        
        # Check specific values
        expected_factor_5 = np.exp(-config.lambda_qft * 5 / config.hbar_eff)
        assert factors[5] == pytest.approx(expected_factor_5)
    
    def test_boltzmann_factors(self, config):
        """Test Boltzmann factor computation"""
        tables = PrecomputedTables(config)
        
        # Boltzmann factors should be positive
        factors = tables.boltzmann_table.cpu().numpy()
        assert np.all(factors > 0)
        
        # Check monotonicity (higher action = lower probability)
        mid_idx = config.table_size // 2
        assert factors[mid_idx - 100] > factors[mid_idx + 100]
    
    def test_quantum_corrections(self, tables):
        """Test quantum correction factors"""
        corrections = tables.quantum_corrections.cpu().numpy()
        
        # All corrections should be >= 1 (quantum adds to classical)
        assert np.all(corrections >= 1.0)
        
        # Higher visit counts should have smaller corrections
        assert corrections[100] < corrections[10]
    
    def test_phase_table(self, tables):
        """Test phase factor computation"""
        phases = tables.phase_table.cpu().numpy()
        
        # Phase factors should be in [-1, 1]
        assert np.all(phases >= -1.0)
        assert np.all(phases <= 1.0)
        
        # Check specific values
        assert phases[0] == pytest.approx(1.0)  # cos(0) = 1
        # With linspace(0, 1, 1000), index 500 corresponds to similarity 500/999
        similarity_at_500 = 500.0 / 999.0
        expected_phase = np.cos(np.pi * similarity_at_500)
        assert phases[500] == pytest.approx(expected_phase, abs=1e-6)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_tables(self):
        """Test table generation on GPU"""
        config = PathIntegralConfig(device='cuda')
        tables = PrecomputedTables(config)
        
        assert tables.length_factors.device.type == 'cuda'
        assert tables.boltzmann_table.device.type == 'cuda'
        assert tables.quantum_corrections.device.type == 'cuda'
        assert tables.phase_table.device.type == 'cuda'


class TestPathIntegral:
    """Test path integral computation"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return PathIntegralConfig(
            device='cpu',
            table_size=1000,
            cache_path_actions=True,
            max_path_length=20
        )
    
    @pytest.fixture
    def path_integral(self, config):
        """Create path integral instance"""
        return PathIntegral(config)
    
    def test_initialization(self, path_integral, config):
        """Test path integral initialization"""
        assert path_integral.config == config
        assert path_integral.device.type == 'cpu'
        assert path_integral.tables is not None
        assert path_integral.stats['cache_hits'] == 0
        assert path_integral.stats['cache_misses'] == 0
        assert path_integral.stats['paths_evaluated'] == 0
    
    def test_single_path_computation(self, path_integral):
        """Test computation for a single path"""
        # Create a simple path
        paths = torch.tensor([[0, 1, 2, -1, -1]], dtype=torch.long)
        values = torch.tensor([0.5])
        visits = torch.tensor([10])
        
        result = path_integral.compute_path_integral_batch(paths, values, visits)
        
        assert result.shape == (1,)
        assert result[0] > 0  # Should be positive
        assert path_integral.stats['paths_evaluated'] == 1
    
    def test_batch_computation(self, path_integral):
        """Test batch path integral computation"""
        batch_size = 10
        max_depth = 5
        
        # Create random paths
        paths = torch.randint(0, 100, (batch_size, max_depth))
        paths[paths > 50] = -1  # Add padding
        
        values = torch.rand(batch_size)
        visits = torch.randint(1, 100, (batch_size,))
        
        result = path_integral.compute_path_integral_batch(paths, values, visits)
        
        assert result.shape == (batch_size,)
        assert torch.all(result >= 0)  # All should be non-negative
        assert path_integral.stats['paths_evaluated'] == batch_size
    
    def test_length_factor_lookup(self, path_integral):
        """Test length factor lookup functionality"""
        # Test various path lengths
        lengths = torch.tensor([0, 5, 10, 25, 100])
        
        factors = path_integral._lookup_length_factors(lengths)
        
        assert factors.shape == lengths.shape
        assert factors[0] == path_integral.tables.length_factors[0]
        assert factors[1] == path_integral.tables.length_factors[5]
        # Check clamping for out-of-range values
        assert factors[4] == path_integral.tables.length_factors[19]  # max_path_length - 1
    
    def test_action_discretization(self, path_integral):
        """Test action discretization for table lookup"""
        # Test various action values
        actions = torch.tensor([-15.0, -10.0, 0.0, 10.0, 15.0])
        
        indices = path_integral._discretize_actions(actions)
        
        assert indices.shape == actions.shape
        assert torch.all(indices >= 0)
        assert torch.all(indices < path_integral.config.table_size)
        
        # Check specific mappings
        assert indices[1] == 0  # -10 maps to 0
        # For action=0: normalized = 0.5, index = 0.5 * 999 = 499.5 → 499 (after .long())
        assert indices[2] == 499  # 0 maps to 499, not 500
        assert indices[3] == path_integral.config.table_size - 1  # 10 maps to end
    
    def test_action_caching(self, path_integral):
        """Test action caching mechanism"""
        # Create paths
        paths = torch.tensor([[0, 1, 2, -1], [3, 4, 5, -1]])
        values = torch.tensor([0.5, 0.7])
        visits = torch.tensor([10, 20])
        
        # First computation - should miss cache
        actions1 = path_integral._get_cached_actions(paths, values, visits)
        assert path_integral.stats['cache_misses'] == 2
        assert path_integral.stats['cache_hits'] == 0
        
        # Second computation with same paths - should hit cache
        actions2 = path_integral._get_cached_actions(paths, values, visits)
        assert path_integral.stats['cache_hits'] == 2
        assert torch.allclose(actions1, actions2)
    
    def test_action_computation(self, path_integral):
        """Test vectorized action computation"""
        batch_size = 5
        max_depth = 10
        
        paths = torch.randint(0, 100, (batch_size, max_depth))
        paths[paths > 50] = -1  # Add padding
        values = torch.rand(batch_size)
        visits = torch.randint(1, 100, (batch_size,))
        
        actions = path_integral._compute_actions_vectorized(paths, values, visits)
        
        assert actions.shape == (batch_size,)
        # Actions should be negative (as per formula)
        assert torch.all(actions <= 0)
    
    def test_interference_computation(self, path_integral):
        """Test interference computation between paths"""
        # Create similar paths for strong interference
        paths = torch.tensor([
            [0, 1, 2, 3, -1],
            [0, 1, 2, 4, -1],  # Differs only in last node
            [5, 6, 7, 8, -1]   # Completely different
        ])
        
        amplitudes = torch.ones(3)
        
        result = path_integral._apply_interference_fast(paths, amplitudes)
        
        assert result.shape == (3,)
        assert torch.all(result > 0)  # Should remain positive after interference
        
        # Paths 0 and 1 should interfere more than paths 0 and 2
        # This is reflected in the modulated amplitudes
    
    def test_interference_single_path(self, path_integral):
        """Test that single path has no interference"""
        paths = torch.tensor([[0, 1, 2, -1]])
        amplitudes = torch.tensor([1.0])
        
        result = path_integral._apply_interference_fast(paths, amplitudes)
        
        assert torch.allclose(result, amplitudes)  # Should be unchanged
    
    def test_similarity_computation(self, path_integral):
        """Test path similarity computation for interference"""
        # Create test paths
        paths = torch.tensor([
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],  # Identical
            [0, 1, 2, 3, 5],  # 80% similar
            [5, 6, 7, 8, 9]   # Completely different
        ])
        
        amplitudes = torch.ones(4)
        
        # Apply interference to test similarity computation
        result = path_integral._apply_interference_fast(paths, amplitudes)
        
        # Identical paths (0,1) should have maximum interference
        # Different paths (0,3) should have minimal interference
        assert result.shape == (4,)
    
    def test_statistics_tracking(self, path_integral):
        """Test performance statistics tracking"""
        # Run some computations
        paths = torch.tensor([[0, 1, 2, -1]])
        values = torch.tensor([0.5])
        visits = torch.tensor([10])
        
        path_integral.compute_path_integral_batch(paths, values, visits)
        path_integral.compute_path_integral_batch(paths, values, visits)
        
        stats = path_integral.get_stats()
        
        assert stats['paths_evaluated'] == 2
        assert stats['cache_hit_rate'] > 0  # Should have cache hits on second call
        assert stats['cache_size'] > 0
    
    def test_cache_clearing(self, path_integral):
        """Test cache clearing functionality"""
        # Populate cache
        paths = torch.tensor([[0, 1, 2, -1]])
        values = torch.tensor([0.5])
        visits = torch.tensor([10])
        
        path_integral.compute_path_integral_batch(paths, values, visits)
        assert len(path_integral.tables.action_cache) > 0
        
        # Clear cache
        path_integral.clear_cache()
        
        assert len(path_integral.tables.action_cache) == 0
        assert path_integral.stats['cache_hits'] == 0
        assert path_integral.stats['cache_misses'] == 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_computation(self):
        """Test path integral computation on GPU"""
        config = PathIntegralConfig(device='cuda')
        path_integral = PathIntegral(config)
        
        # Create GPU tensors
        paths = torch.randint(0, 100, (10, 5), device='cuda')
        paths[paths > 50] = -1
        values = torch.rand(10, device='cuda')
        visits = torch.randint(1, 100, (10,), device='cuda')
        
        result = path_integral.compute_path_integral_batch(paths, values, visits)
        
        assert result.device.type == 'cuda'
        assert result.shape == (10,)
    
    def test_mixed_precision(self):
        """Test mixed precision computation"""
        config = PathIntegralConfig(device='cpu', use_mixed_precision=True)
        path_integral = PathIntegral(config)
        
        # The implementation should handle mixed precision internally
        paths = torch.randint(0, 100, (5, 5))
        values = torch.rand(5)
        visits = torch.randint(1, 100, (5,))
        
        result = path_integral.compute_path_integral_batch(paths, values, visits)
        
        assert result.dtype == torch.float32  # Should return float32 even with mixed precision
    
    def test_empty_paths(self, path_integral):
        """Test handling of empty paths (all padding)"""
        # Create paths with all -1 (empty)
        paths = torch.full((3, 5), -1, dtype=torch.long)
        values = torch.rand(3)
        visits = torch.ones(3)
        
        result = path_integral.compute_path_integral_batch(paths, values, visits)
        
        assert result.shape == (3,)
        assert torch.all(torch.isfinite(result))  # Should not contain inf/nan
    
    def test_very_long_paths(self, path_integral):
        """Test handling of paths longer than max_path_length"""
        # Create very long paths
        paths = torch.randint(0, 100, (2, 100))
        values = torch.rand(2)
        visits = torch.ones(2)
        
        result = path_integral.compute_path_integral_batch(paths, values, visits)
        
        assert result.shape == (2,)
        assert torch.all(torch.isfinite(result))  # Should handle gracefully


class TestIntegration:
    """Integration tests for path integral with MCTS"""
    
    def test_with_tree_structure(self):
        """Test path integral with tree structure information"""
        config = PathIntegralConfig(device='cpu')
        path_integral = PathIntegral(config)
        
        # Create mock tree structure
        tree = {
            'children': torch.tensor([[1, 2, -1], [3, 4, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]])
        }
        
        paths = torch.tensor([[0, 1, 3, -1], [0, 2, -1, -1]])
        values = torch.rand(2)
        visits = torch.randint(1, 50, (2,))
        
        result = path_integral.compute_path_integral_batch(paths, values, visits, tree)
        
        assert result.shape == (2,)
        assert torch.all(result > 0)
    
    def test_performance_characteristics(self):
        """Test that path integral maintains performance characteristics"""
        config = PathIntegralConfig(device='cpu', cache_path_actions=True)
        path_integral = PathIntegral(config)
        
        # Generate many paths
        batch_size = 100
        paths = torch.randint(0, 1000, (batch_size, 10))
        paths[paths > 500] = -1
        values = torch.rand(batch_size)
        visits = torch.randint(1, 100, (batch_size,))
        
        import time
        
        # First run (cold cache)
        start = time.time()
        result1 = path_integral.compute_path_integral_batch(paths, values, visits)
        cold_time = time.time() - start
        
        # Second run (warm cache)
        start = time.time()
        result2 = path_integral.compute_path_integral_batch(paths, values, visits)
        warm_time = time.time() - start
        
        # Warm cache should be faster
        assert warm_time < cold_time
        assert torch.allclose(result1, result2)
        
        # Check that we're using pre-computed tables efficiently
        stats = path_integral.get_stats()
        assert stats['cache_hit_rate'] >= 0.5  # Should have >= 50% cache hits (first run all misses, second run all hits)