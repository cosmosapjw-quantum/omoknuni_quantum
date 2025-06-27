"""Tests for neural network evaluator interface"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock

from mcts.core.evaluator import Evaluator, MockEvaluator, EvaluatorConfig


class TestEvaluator:
    """Test suite for neural network evaluator interface"""
    
    def test_evaluator_config(self):
        """Test evaluator configuration"""
        config = EvaluatorConfig()
        assert config.batch_size == 512
        assert config.device == 'cuda' if torch.cuda.is_available() else 'cpu'
        assert config.num_channels == 256
        assert config.num_blocks == 20
        
        # Custom config
        custom_config = EvaluatorConfig(
            batch_size=1024,
            device='cpu',
            num_channels=128,
            num_blocks=10
        )
        assert custom_config.batch_size == 1024
        assert custom_config.device == 'cpu'
        
    def test_mock_evaluator_initialization(self):
        """Test mock evaluator creation"""
        config = EvaluatorConfig()
        evaluator = MockEvaluator(config, action_size=362)  # Go 19x19
        
        assert evaluator.config == config
        assert evaluator.action_size == 362
        
    def test_mock_evaluator_single_evaluation(self):
        """Test evaluating a single position"""
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=225)  # Gomoku 15x15
        
        # Create dummy input
        state = np.random.rand(20, 15, 15).astype(np.float32)  # 20 channels
        
        policy, value = evaluator.evaluate(state)
        
        assert isinstance(policy, np.ndarray)
        assert isinstance(value, float)
        assert policy.shape == (225,)
        assert np.abs(policy.sum() - 1.0) < 1e-6  # Should sum to 1
        assert -1 <= value <= 1  # Value should be in [-1, 1]
        
    def test_mock_evaluator_batch_evaluation(self):
        """Test batch evaluation"""
        batch_size = 32
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=4096)  # Chess
        
        # Create batch of states
        states = np.random.rand(batch_size, 20, 8, 8).astype(np.float32)
        
        policies, values = evaluator.evaluate_batch(states)
        
        assert policies.shape == (batch_size, 4096)
        assert values.shape == (batch_size,)
        assert np.all(np.abs(policies.sum(axis=1) - 1.0) < 1e-6)
        assert np.all((-1 <= values) & (values <= 1))
        
    def test_mock_evaluator_with_noise(self):
        """Test that mock evaluator adds noise for diversity"""
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=100)
        state = np.random.rand(20, 10, 10).astype(np.float32)
        
        # Get multiple evaluations
        results = []
        for _ in range(10):
            policy, value = evaluator.evaluate(state)
            results.append((policy.copy(), value))
            
        # Check that results are not identical (due to noise)
        policies = [r[0] for r in results]
        values = [r[1] for r in results]
        
        # Policies should be similar but not identical
        for i in range(1, len(policies)):
            assert not np.array_equal(policies[0], policies[i])
            
        # Values should vary slightly
        assert np.std(values) > 0.01
        
    def test_evaluator_caching(self):
        """Test position caching in evaluator"""
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=361)
        evaluator.enable_cache(max_size=1000)
        
        state = np.random.rand(20, 19, 19).astype(np.float32)
        
        # First evaluation
        policy1, value1 = evaluator.evaluate(state)
        assert evaluator.cache_hits == 0
        assert evaluator.cache_misses == 1
        
        # Second evaluation should hit cache
        policy2, value2 = evaluator.evaluate(state)
        assert evaluator.cache_hits == 1
        assert evaluator.cache_misses == 1
        
        # Results should be identical (from cache)
        assert np.array_equal(policy1, policy2)
        assert value1 == value2
        
    def test_evaluator_legal_moves_masking(self):
        """Test that evaluator respects legal move masks"""
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=225)
        state = np.random.rand(20, 15, 15).astype(np.float32)
        
        # Create legal moves mask (only some moves legal)
        legal_mask = np.zeros(225, dtype=bool)
        legal_indices = [0, 5, 10, 50, 100, 224]
        legal_mask[legal_indices] = True
        
        policy, value = evaluator.evaluate(state, legal_mask=legal_mask)
        
        # Policy should only have probabilities for legal moves
        assert np.all(policy[~legal_mask] == 0)
        assert np.abs(policy[legal_mask].sum() - 1.0) < 1e-6
        
    def test_evaluator_temperature_scaling(self):
        """Test temperature scaling for policy"""
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=100)
        state = np.random.rand(20, 10, 10).astype(np.float32)
        
        # Get policies at different temperatures
        policy_t1 = evaluator.evaluate(state, temperature=1.0)[0]
        policy_t0_5 = evaluator.evaluate(state, temperature=0.5)[0]
        policy_t2 = evaluator.evaluate(state, temperature=2.0)[0]
        
        # Lower temperature should be more peaked
        entropy_t0_5 = -np.sum(policy_t0_5 * np.log(policy_t0_5 + 1e-8))
        entropy_t1 = -np.sum(policy_t1 * np.log(policy_t1 + 1e-8))
        entropy_t2 = -np.sum(policy_t2 * np.log(policy_t2 + 1e-8))
        
        assert entropy_t0_5 < entropy_t1 < entropy_t2
        
    def test_evaluator_device_placement(self):
        """Test that evaluator handles device placement correctly"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        # Test GPU evaluator
        gpu_config = EvaluatorConfig(device='cuda')
        gpu_evaluator = MockEvaluator(gpu_config, action_size=361)
        
        state = np.random.rand(20, 19, 19).astype(np.float32)
        policy, value = gpu_evaluator.evaluate(state)
        
        # Should work without errors
        assert policy.shape == (361,)
        assert isinstance(value, float)
        
        # Test CPU evaluator
        cpu_config = EvaluatorConfig(device='cpu')
        cpu_evaluator = MockEvaluator(cpu_config, action_size=361)
        
        policy2, value2 = cpu_evaluator.evaluate(state)
        assert policy2.shape == (361,)
        
    def test_evaluator_abstract_methods(self):
        """Test that base Evaluator class enforces abstract methods"""
        with pytest.raises(TypeError):
            # Should not be able to instantiate abstract base class
            Evaluator(EvaluatorConfig(), action_size=100)
            
    def test_evaluator_warmup(self):
        """Test evaluator warmup functionality"""
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=361)
        
        # Warmup should run without errors
        evaluator.warmup(num_iterations=10)
        
        # After warmup, evaluation should be fast
        import time
        state = np.random.rand(20, 19, 19).astype(np.float32)
        
        start = time.time()
        for _ in range(100):
            evaluator.evaluate(state)
        elapsed = time.time() - start
        
        # Should be fast (less than 10ms per evaluation on average)
        assert elapsed < 1.0  # 100 evaluations in less than 1 second