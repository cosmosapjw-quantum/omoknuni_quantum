"""Tests for evaluator interface and implementations

Tests cover:
- Abstract evaluator interface
- AlphaZero evaluator with neural networks
- Random evaluator baseline
- Caching functionality
- Statistics tracking
- Error handling
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import time

from mcts.core.evaluator import (
    Evaluator, EvaluatorConfig, AlphaZeroEvaluator, RandomEvaluator
)


class MockAlphaZeroModel(nn.Module):
    """Mock neural network model for testing"""
    
    def __init__(self, action_size=225):
        super().__init__()
        self.action_size = action_size
        self.fc1 = nn.Linear(225, 128)
        self.fc_policy = nn.Linear(128, action_size)
        self.fc_value = nn.Linear(128, 1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(batch_size, -1)
        
        # Ensure correct input size
        if x.shape[1] != 225:
            x = nn.functional.adaptive_avg_pool1d(x.unsqueeze(1), 225).squeeze(1)
            
        x = torch.relu(self.fc1(x))
        policy_logits = self.fc_policy(x)
        value = torch.tanh(self.fc_value(x))
        return policy_logits, value


class ConcreteEvaluator(Evaluator):
    """Concrete implementation for testing abstract methods"""
    
    def evaluate(self, state, legal_mask=None, temperature=1.0):
        import time
        start_time = time.time()
        
        policy = np.ones(self.action_size) / self.action_size
        if legal_mask is not None:
            policy[~legal_mask] = 0
            if policy.sum() > 0:
                policy = policy / policy.sum()
        
        eval_time = time.time() - start_time
        self._update_stats(1, eval_time, is_batch=False)
        return policy, 0.0
    
    def evaluate_batch(self, states, legal_masks=None, temperature=1.0):
        import time
        start_time = time.time()
        
        batch_size = states.shape[0]
        policies = np.ones((batch_size, self.action_size)) / self.action_size
        if legal_masks is not None:
            for i in range(batch_size):
                policies[i][~legal_masks[i]] = 0
                if policies[i].sum() > 0:
                    policies[i] = policies[i] / policies[i].sum()
        values = np.zeros(batch_size)
        
        eval_time = time.time() - start_time
        self._update_stats(batch_size, eval_time, is_batch=True)
        return policies, values


@pytest.fixture
def evaluator_config():
    """Create evaluator configuration"""
    return EvaluatorConfig(
        batch_size=32,
        device='cpu',
        timeout=1.0,
        enable_caching=True,
        cache_size=100
    )


@pytest.fixture
def mock_model():
    """Create mock AlphaZero model"""
    return MockAlphaZeroModel(action_size=225)


@pytest.fixture
def game_state():
    """Create sample game state"""
    return np.zeros((3, 15, 15), dtype=np.float32)


@pytest.fixture
def legal_mask():
    """Create sample legal mask"""
    mask = np.zeros(225, dtype=bool)
    mask[:100] = True  # First 100 moves are legal
    return mask


class TestEvaluatorConfig:
    """Test evaluator configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = EvaluatorConfig()
        assert config.batch_size == 64
        assert config.timeout == 1.0
        assert config.enable_caching == False
        assert config.cache_size == 10000
        
    def test_custom_config(self):
        """Test custom configuration"""
        config = EvaluatorConfig(
            batch_size=128,
            device='cuda',
            timeout=2.0,
            enable_caching=True,
            cache_size=5000
        )
        assert config.batch_size == 128
        assert config.device == 'cuda'
        assert config.timeout == 2.0
        assert config.enable_caching == True
        assert config.cache_size == 5000


class TestAbstractEvaluator:
    """Test abstract evaluator interface"""
    
    def test_initialization(self, evaluator_config):
        """Test evaluator initialization"""
        evaluator = ConcreteEvaluator(evaluator_config, action_size=225)
        
        assert evaluator.config == evaluator_config
        assert evaluator.action_size == 225
        assert evaluator.stats['evaluations'] == 0
        assert evaluator.stats['batch_evaluations'] == 0
        assert evaluator.cache is not None  # Caching enabled
        
    def test_stats_tracking(self, evaluator_config):
        """Test statistics tracking"""
        evaluator = ConcreteEvaluator(evaluator_config, action_size=225)
        
        # Update stats
        evaluator._update_stats(batch_size=10, eval_time=0.1)
        
        stats = evaluator.get_stats()
        assert stats['evaluations'] == 10
        assert stats['batch_evaluations'] == 1
        assert stats['total_time'] == 0.1
        assert stats['avg_time'] == 0.01
        
    def test_stats_reset(self, evaluator_config):
        """Test statistics reset"""
        evaluator = ConcreteEvaluator(evaluator_config, action_size=225)
        
        # Add some stats
        evaluator._update_stats(batch_size=10, eval_time=0.1)
        evaluator.reset_stats()
        
        stats = evaluator.get_stats()
        assert stats['evaluations'] == 0
        assert stats['batch_evaluations'] == 0
        assert stats['total_time'] == 0.0
        assert stats['avg_time'] == 0.0
        
    def test_cache_operations(self, evaluator_config):
        """Test cache operations"""
        evaluator = ConcreteEvaluator(evaluator_config, action_size=225)
        
        # Test cache put/get
        policy = np.ones(225) / 225
        value = 0.5
        state_key = "test_state"
        
        evaluator._cache_put(state_key, policy, value)
        cached = evaluator._cache_get(state_key)
        
        assert cached is not None
        np.testing.assert_array_equal(cached[0], policy)
        assert cached[1] == value
        
    def test_cache_size_limit(self, evaluator_config):
        """Test cache size limiting"""
        evaluator_config.cache_size = 5
        evaluator = ConcreteEvaluator(evaluator_config, action_size=225)
        
        # Fill cache beyond limit
        for i in range(10):
            policy = np.ones(225) / 225
            evaluator._cache_put(f"state_{i}", policy, float(i))
            
        # Check cache size
        assert len(evaluator.cache) == 5
        
        # Verify oldest entries were removed
        assert evaluator._cache_get("state_0") is None
        assert evaluator._cache_get("state_4") is None
        assert evaluator._cache_get("state_5") is not None
        
    def test_cache_disabled(self):
        """Test with caching disabled"""
        config = EvaluatorConfig(enable_caching=False)
        evaluator = ConcreteEvaluator(config, action_size=225)
        
        assert evaluator.cache is None
        
        # Cache operations should be no-ops
        policy = np.ones(225) / 225
        evaluator._cache_put("test", policy, 0.5)
        assert evaluator._cache_get("test") is None
        
    def test_warmup(self, evaluator_config, game_state):
        """Test evaluator warmup"""
        evaluator = ConcreteEvaluator(evaluator_config, action_size=225)
        
        # Run warmup
        evaluator.warmup(game_state, num_iterations=5)
        
        # Check that evaluations were performed
        stats = evaluator.get_stats()
        assert stats['evaluations'] == 10  # 5 single + 5 batch
        assert stats['batch_evaluations'] == 5


class TestAlphaZeroEvaluator:
    """Test AlphaZero neural network evaluator"""
    
    def test_initialization(self, mock_model, evaluator_config):
        """Test AlphaZero evaluator initialization"""
        evaluator = AlphaZeroEvaluator(
            mock_model, 
            config=evaluator_config,
            action_size=225
        )
        
        assert evaluator.model == mock_model
        assert evaluator.action_size == 225
        assert evaluator.device == torch.device('cpu')
        
    def test_action_size_inference(self, evaluator_config):
        """Test automatic action size inference"""
        model = MockAlphaZeroModel(action_size=361)  # Go board size
        evaluator = AlphaZeroEvaluator(model, config=evaluator_config)
        
        assert evaluator.action_size == 361
        
    def test_single_evaluation(self, mock_model, game_state, legal_mask):
        """Test single state evaluation"""
        evaluator = AlphaZeroEvaluator(mock_model, action_size=225)
        
        policy, value = evaluator.evaluate(game_state, legal_mask)
        
        # Check outputs
        assert policy.shape == (225,)
        assert isinstance(value, float)
        assert -1 <= value <= 1
        
        # Check legal mask applied
        assert np.all(policy[~legal_mask] == 0)
        assert np.isclose(policy.sum(), 1.0)
        
        # Check stats updated
        stats = evaluator.get_stats()
        assert stats['evaluations'] == 1
        
    def test_batch_evaluation(self, mock_model, game_state, legal_mask):
        """Test batch evaluation"""
        evaluator = AlphaZeroEvaluator(mock_model, action_size=225)
        
        # Create batch
        batch_size = 8
        states = np.repeat(game_state[np.newaxis, ...], batch_size, axis=0)
        legal_masks = np.repeat(legal_mask[np.newaxis, ...], batch_size, axis=0)
        
        policies, values = evaluator.evaluate_batch(states, legal_masks)
        
        # Check outputs
        assert policies.shape == (batch_size, 225)
        assert values.shape == (batch_size,)
        assert np.all(values >= -1) and np.all(values <= 1)
        
        # Check legal masks applied
        for i in range(batch_size):
            assert np.all(policies[i][~legal_masks[i]] == 0)
            assert np.isclose(policies[i].sum(), 1.0)
            
        # Check stats updated
        stats = evaluator.get_stats()
        assert stats['evaluations'] == batch_size
        
    def test_temperature_scaling(self, mock_model, game_state):
        """Test temperature scaling in evaluation"""
        evaluator = AlphaZeroEvaluator(mock_model, action_size=225)
        
        # Evaluate with different temperatures
        policy_t1, _ = evaluator.evaluate(game_state, temperature=1.0)
        policy_t2, _ = evaluator.evaluate(game_state, temperature=2.0)
        policy_t05, _ = evaluator.evaluate(game_state, temperature=0.5)
        
        # Higher temperature should make distribution more uniform
        entropy_t1 = -np.sum(policy_t1 * np.log(policy_t1 + 1e-8))
        entropy_t2 = -np.sum(policy_t2 * np.log(policy_t2 + 1e-8))
        entropy_t05 = -np.sum(policy_t05 * np.log(policy_t05 + 1e-8))
        
        assert entropy_t2 > entropy_t1  # Higher temp = higher entropy
        assert entropy_t05 < entropy_t1  # Lower temp = lower entropy
        
    def test_no_legal_moves(self, mock_model, game_state):
        """Test handling when no legal moves"""
        evaluator = AlphaZeroEvaluator(mock_model, action_size=225)
        
        # No legal moves
        legal_mask = np.zeros(225, dtype=bool)
        
        policy, value = evaluator.evaluate(game_state, legal_mask)
        
        # Should return zeros (no valid distribution possible)
        assert np.all(policy == 0)
        
    def test_device_handling(self, mock_model):
        """Test device handling"""
        # Test CPU device
        config = EvaluatorConfig(device='cpu')
        evaluator = AlphaZeroEvaluator(mock_model, config=config)
        assert evaluator.device == torch.device('cpu')
        assert next(evaluator.model.parameters()).device.type == 'cpu'
        
    @patch('torch.cuda.is_available')
    def test_cuda_device(self, mock_cuda_available, mock_model):
        """Test CUDA device handling"""
        mock_cuda_available.return_value = True
        
        # Mock model.to() to avoid actual CUDA operations in test
        mock_model.to = Mock(return_value=mock_model)
        
        config = EvaluatorConfig(device='cuda')
        evaluator = AlphaZeroEvaluator(mock_model, config=config)
        
        # Verify to() was called with cuda device
        mock_model.to.assert_called_once()
        device_arg = mock_model.to.call_args[0][0]
        assert device_arg.type == 'cuda'
        
    def test_tensor_input_handling(self, mock_model):
        """Test handling of tensor inputs"""
        evaluator = AlphaZeroEvaluator(mock_model, action_size=225)
        
        # Create tensor input instead of numpy
        state_tensor = torch.zeros(3, 15, 15)
        
        policy, value = evaluator.evaluate(state_tensor)
        
        assert policy.shape == (225,)
        assert isinstance(value, float)
        
    def test_single_batch_value_handling(self, mock_model, game_state):
        """Test handling of single-element batch values"""
        evaluator = AlphaZeroEvaluator(mock_model, action_size=225)
        
        # Single element batch
        states = game_state[np.newaxis, ...]
        
        policies, values = evaluator.evaluate_batch(states)
        
        # Values should be 1D array even for single batch
        assert values.shape == (1,)
        assert values.ndim == 1


class TestRandomEvaluator:
    """Test random evaluator baseline"""
    
    def test_initialization(self, evaluator_config):
        """Test random evaluator initialization"""
        evaluator = RandomEvaluator(action_size=361, config=evaluator_config)
        
        assert evaluator.action_size == 361
        assert evaluator.rng is not None
        
    def test_single_evaluation(self, game_state, legal_mask):
        """Test single random evaluation"""
        evaluator = RandomEvaluator(action_size=225)
        
        policy, value = evaluator.evaluate(game_state, legal_mask)
        
        # Check outputs
        assert policy.shape == (225,)
        assert isinstance(value, float)
        assert -0.1 <= value <= 0.1  # Random evaluator uses small values
        
        # Check legal mask applied
        assert np.all(policy[~legal_mask] == 0)
        assert np.isclose(policy.sum(), 1.0)
        
    def test_batch_evaluation(self, game_state, legal_mask):
        """Test batch random evaluation"""
        evaluator = RandomEvaluator(action_size=225)
        
        # Create batch
        batch_size = 16
        states = np.repeat(game_state[np.newaxis, ...], batch_size, axis=0)
        legal_masks = np.repeat(legal_mask[np.newaxis, ...], batch_size, axis=0)
        
        policies, values = evaluator.evaluate_batch(states, legal_masks)
        
        # Check outputs
        assert policies.shape == (batch_size, 225)
        assert values.shape == (batch_size,)
        assert np.all(values >= -0.1) and np.all(values <= 0.1)
        
        # Check legal masks applied
        for i in range(batch_size):
            assert np.all(policies[i][~legal_masks[i]] == 0)
            assert np.isclose(policies[i].sum(), 1.0)
            
    def test_reproducibility(self, game_state):
        """Test random evaluator reproducibility"""
        evaluator1 = RandomEvaluator(action_size=225)
        evaluator2 = RandomEvaluator(action_size=225)
        
        # Should produce same results with same seed
        policy1, value1 = evaluator1.evaluate(game_state)
        policy2, value2 = evaluator2.evaluate(game_state)
        
        np.testing.assert_array_equal(policy1, policy2)
        assert value1 == value2
        
    def test_no_legal_moves(self, game_state):
        """Test handling when no legal moves"""
        evaluator = RandomEvaluator(action_size=225)
        
        # No legal moves
        legal_mask = np.zeros(225, dtype=bool)
        
        policy, value = evaluator.evaluate(game_state, legal_mask)
        
        # Should return zeros
        assert np.all(policy == 0)
        
    def test_all_legal_moves(self, game_state):
        """Test with all moves legal"""
        evaluator = RandomEvaluator(action_size=225)
        
        # All moves legal (no mask)
        policy, value = evaluator.evaluate(game_state)
        
        # Should be normalized random distribution
        assert policy.shape == (225,)
        assert np.isclose(policy.sum(), 1.0)
        assert np.all(policy >= 0)


class TestEvaluatorIntegration:
    """Integration tests for evaluators"""
    
    def test_evaluator_comparison(self, mock_model, game_state, legal_mask):
        """Compare different evaluator implementations"""
        # Create evaluators
        alphazero_eval = AlphaZeroEvaluator(mock_model, action_size=225)
        random_eval = RandomEvaluator(action_size=225)
        
        # Evaluate same position
        az_policy, az_value = alphazero_eval.evaluate(game_state, legal_mask)
        rand_policy, rand_value = random_eval.evaluate(game_state, legal_mask)
        
        # Both should respect legal moves
        assert np.all(az_policy[~legal_mask] == 0)
        assert np.all(rand_policy[~legal_mask] == 0)
        
        # Both should be normalized
        assert np.isclose(az_policy.sum(), 1.0)
        assert np.isclose(rand_policy.sum(), 1.0)
        
        # Values should be in valid range
        assert -1 <= az_value <= 1
        assert -0.1 <= rand_value <= 0.1
        
    def test_large_batch_evaluation(self, mock_model):
        """Test evaluation with large batches"""
        evaluator = AlphaZeroEvaluator(mock_model, action_size=225)
        
        # Large batch
        batch_size = 128
        states = np.zeros((batch_size, 3, 15, 15), dtype=np.float32)
        
        policies, values = evaluator.evaluate_batch(states)
        
        assert policies.shape == (batch_size, 225)
        assert values.shape == (batch_size,)
        
    def test_mixed_legal_masks(self, mock_model, game_state):
        """Test batch with different legal masks per position"""
        evaluator = AlphaZeroEvaluator(mock_model, action_size=225)
        
        batch_size = 4
        states = np.repeat(game_state[np.newaxis, ...], batch_size, axis=0)
        
        # Different legal masks
        legal_masks = np.zeros((batch_size, 225), dtype=bool)
        legal_masks[0, :50] = True   # First 50 moves legal
        legal_masks[1, 50:100] = True # Next 50 moves legal
        legal_masks[2, :] = True      # All moves legal
        legal_masks[3, :10] = True    # Only 10 moves legal
        
        policies, values = evaluator.evaluate_batch(states, legal_masks)
        
        # Check each policy respects its mask
        for i in range(batch_size):
            assert np.all(policies[i][~legal_masks[i]] == 0)
            assert np.isclose(policies[i].sum(), 1.0)
            
    def test_performance_tracking(self, mock_model, game_state):
        """Test performance statistics tracking"""
        evaluator = AlphaZeroEvaluator(mock_model, action_size=225)
        
        # Perform multiple evaluations
        for _ in range(10):
            evaluator.evaluate(game_state)
            
        batch_states = np.repeat(game_state[np.newaxis, ...], 8, axis=0)
        for _ in range(5):
            evaluator.evaluate_batch(batch_states)
            
        stats = evaluator.get_stats()
        assert stats['evaluations'] == 50  # 10 single + 5*8 batch
        assert stats['batch_evaluations'] == 5
        assert stats['total_time'] > 0
        assert stats['avg_time'] > 0


class TestErrorHandling:
    """Test error handling in evaluators"""
    
    def test_import_error_without_torch(self):
        """Test error when torch not available"""
        with patch('mcts.core.evaluator.HAS_TORCH', False):
            with pytest.raises(ImportError, match="PyTorch is required"):
                AlphaZeroEvaluator(Mock(), action_size=225)
                
    def test_invalid_action_size(self):
        """Test handling of invalid action sizes"""
        evaluator = RandomEvaluator(action_size=0)
        
        # Should handle gracefully
        state = np.zeros((3, 15, 15))
        policy, value = evaluator.evaluate(state)
        
        assert policy.shape == (0,)
        
    def test_model_forward_error(self, evaluator_config, game_state):
        """Test handling of model forward pass errors"""
        # Create model that raises error
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock()
        mock_model.action_size = 225
        mock_model.side_effect = RuntimeError("Model error")  # Make model callable raise error
        
        evaluator = AlphaZeroEvaluator(mock_model, config=evaluator_config)
        
        with pytest.raises(RuntimeError, match="Model error"):
            evaluator.evaluate(game_state)