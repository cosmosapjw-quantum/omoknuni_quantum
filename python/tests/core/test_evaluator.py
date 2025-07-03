"""Tests for evaluator module"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from mcts.core.evaluator import Evaluator, MockEvaluator, AlphaZeroEvaluator, EvaluatorConfig


class TestEvaluator:
    """Test the abstract Evaluator base class"""
    
    def test_evaluator_is_abstract(self):
        """Test that Evaluator cannot be instantiated directly"""
        with pytest.raises(TypeError):
            Evaluator(EvaluatorConfig(), action_size=225)
    
    def test_evaluator_interface(self):
        """Test that evaluator has required abstract methods"""
        # Create a concrete implementation for testing
        class ConcreteEvaluator(Evaluator):
            def __init__(self):
                super().__init__(EvaluatorConfig(), action_size=225)
                
            def evaluate(self, state, legal_mask=None, temperature=1.0):
                # Single state evaluation
                policy = np.random.rand(225).astype(np.float32)
                policy = policy / policy.sum()  # Normalize
                value = np.random.rand().astype(np.float32) * 2 - 1  # [-1, 1]
                return policy, value
                
            def evaluate_batch(self, states, legal_masks=None, temperature=1.0):
                batch_size = states.shape[0]
                action_size = 225  # Gomoku action space
                policies = np.random.rand(batch_size, action_size).astype(np.float32)
                # Normalize each policy
                policies = policies / policies.sum(axis=1, keepdims=True)
                values = np.random.rand(batch_size).astype(np.float32) * 2 - 1
                return policies, values
        
        evaluator = ConcreteEvaluator()
        assert hasattr(evaluator, 'evaluate')
        assert callable(evaluator.evaluate)
        assert hasattr(evaluator, 'evaluate_batch')
        assert callable(evaluator.evaluate_batch)


class TestMockEvaluator:
    """Test the MockEvaluator implementation"""
    
    def test_mock_evaluator_creation(self, mock_evaluator):
        """Test MockEvaluator can be created"""
        assert isinstance(mock_evaluator, MockEvaluator)
        assert isinstance(mock_evaluator, Evaluator)
    
    def test_mock_evaluator_single_state(self, mock_evaluator, sample_game_state):
        """Test evaluation of single state"""
        # evaluate() expects a single state, not a batch
        policy, value = mock_evaluator.evaluate(sample_game_state)
        
        assert policy.shape == (225,)  # Gomoku action space
        assert isinstance(value, (float, np.floating))
        assert policy.dtype in [np.float32, np.float64]  # Accept both float types
        
        # Check policy sums to approximately 1 (normalized)
        assert abs(policy.sum() - 1.0) < 0.01
        
        # Check value is in reasonable range
        assert -1.0 <= value <= 1.0
    
    def test_mock_evaluator_batch(self, mock_evaluator, batch_game_states):
        """Test evaluation of batch of states"""
        # evaluate_batch() is for batches
        policies, values = mock_evaluator.evaluate_batch(batch_game_states)
        
        batch_size = batch_game_states.shape[0]
        assert policies.shape == (batch_size, 225)
        assert values.shape == (batch_size,)
        
        # Check all policies are normalized
        for i in range(batch_size):
            assert abs(policies[i].sum() - 1.0) < 0.01
            assert -1.0 <= values[i] <= 1.0
    
    def test_mock_evaluator_with_legal_masks(self, mock_evaluator, sample_game_state, sample_legal_moves):
        """Test evaluation with legal move masks"""
        # Test single state evaluation with legal mask
        policy, value = mock_evaluator.evaluate(sample_game_state, legal_mask=sample_legal_moves)
        
        # Check that illegal moves have zero probability
        illegal_positions = ~sample_legal_moves
        assert np.all(policy[illegal_positions] == 0.0)
    
    def test_mock_evaluator_with_temperature(self, mock_evaluator, sample_game_state):
        """Test evaluation with different temperatures"""
        # Test with different temperatures on single state
        for temp in [0.1, 1.0, 2.0]:
            policy, value = mock_evaluator.evaluate(sample_game_state, temperature=temp)
            assert policy.shape == (225,)
            assert isinstance(value, (float, np.floating))
    
    def test_mock_evaluator_reproducibility(self, sample_game_state):
        """Test that MockEvaluator gives consistent results with deterministic mode"""
        # Create two evaluators with deterministic mode
        evaluator1 = MockEvaluator(deterministic=True, fixed_value=0.5)
        evaluator2 = MockEvaluator(deterministic=True, fixed_value=0.5)
        
        policy1, value1 = evaluator1.evaluate(sample_game_state)
        policy2, value2 = evaluator2.evaluate(sample_game_state)
        
        np.testing.assert_array_almost_equal(policy1, policy2)
        assert value1 == value2


class TestEvaluatorConfig:
    """Test the EvaluatorConfig dataclass"""
    
    def test_evaluator_config_creation(self):
        """Test creating EvaluatorConfig with defaults"""
        config = EvaluatorConfig()
        
        assert config.device in ['cuda', 'cpu']  # Depends on torch availability
        assert config.batch_size == 64
        assert config.timeout == 1.0
        assert config.enable_caching == False
    
    def test_evaluator_config_custom(self):
        """Test creating EvaluatorConfig with custom values"""
        config = EvaluatorConfig(
            device='cpu',
            batch_size=32,
            timeout=2.0,
            enable_caching=True,
            cache_size=5000
        )
        
        assert config.device == 'cpu'
        assert config.batch_size == 32
        assert config.timeout == 2.0
        assert config.enable_caching == True
        assert config.cache_size == 5000


class TestAlphaZeroEvaluator:
    """Test the AlphaZeroEvaluator implementation"""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock neural network model"""
        model = Mock()
        model.eval = Mock()
        model.forward = Mock()
        
        # Mock the forward pass
        def mock_forward(x):
            batch_size = x.shape[0]
            policy_logits = torch.randn(batch_size, 225)  # Gomoku action space
            value = torch.randn(batch_size, 1)
            return policy_logits, value
        
        # Make the model callable (model(x) calls this)
        model.side_effect = mock_forward
        model.to = Mock(return_value=model)
        
        return model
    
    def test_alphazero_evaluator_creation(self, mock_model):
        """Test AlphaZeroEvaluator creation"""
        config = EvaluatorConfig(device='cpu')
        evaluator = AlphaZeroEvaluator(mock_model, config)
        
        assert evaluator.model == mock_model
        assert evaluator.config == config
        # Check that model.to was called with some device (string or torch.device)
        mock_model.to.assert_called_once()
        mock_model.eval.assert_called_once()
    
    def test_alphazero_evaluator_single_state(self, mock_model, sample_game_state):
        """Test evaluation of single state"""
        config = EvaluatorConfig(device='cpu')
        evaluator = AlphaZeroEvaluator(mock_model, config)
        
        # evaluate() expects single state
        with patch('torch.no_grad'):
            policy, value = evaluator.evaluate(sample_game_state)
        
        assert policy.shape == (225,)
        assert isinstance(value, (float, np.floating))
        assert policy.dtype in [np.float32, np.float64]  # Accept both float types
        
        # Check that model was called (model(state_tensor))
        mock_model.assert_called_once()
    
    def test_alphazero_evaluator_batch(self, mock_model, batch_game_states):
        """Test evaluation of batch of states"""
        config = EvaluatorConfig(device='cpu')
        evaluator = AlphaZeroEvaluator(mock_model, config)
        
        with patch('torch.no_grad'):
            policies, values = evaluator.evaluate_batch(batch_game_states)
        
        batch_size = batch_game_states.shape[0]
        assert policies.shape == (batch_size, 225)
        assert values.shape == (batch_size,)
    
    def test_alphazero_evaluator_with_legal_masks(self, mock_model, sample_game_state, sample_legal_moves):
        """Test evaluation with legal move masks"""
        config = EvaluatorConfig(device='cpu')
        evaluator = AlphaZeroEvaluator(mock_model, config)
        
        with patch('torch.no_grad'):
            policy, value = evaluator.evaluate(sample_game_state, legal_mask=sample_legal_moves)
        
        # Check that illegal moves have zero probability
        illegal_positions = ~sample_legal_moves
        assert np.all(policy[illegal_positions] == 0.0)
    
    def test_alphazero_evaluator_temperature_scaling(self, mock_model, sample_game_state):
        """Test evaluation with temperature scaling"""
        config = EvaluatorConfig(device='cpu')
        evaluator = AlphaZeroEvaluator(mock_model, config)
        
        with patch('torch.no_grad'):
            # Test different temperatures
            policy_low, _ = evaluator.evaluate(sample_game_state, temperature=0.1)
            policy_high, _ = evaluator.evaluate(sample_game_state, temperature=2.0)
        
        # Low temperature should be more concentrated (higher max probability)
        # High temperature should be more uniform (lower max probability)
        assert policy_low.max() > policy_high.max()
    
    def test_alphazero_evaluator_caching(self, mock_model, sample_game_state):
        """Test AlphaZeroEvaluator with caching enabled"""
        config = EvaluatorConfig(device='cpu', enable_caching=True, cache_size=100)
        evaluator = AlphaZeroEvaluator(mock_model, config)
        
        with patch('torch.no_grad'):
            # First evaluation
            policy1, value1 = evaluator.evaluate(sample_game_state)
            
            # Second evaluation with same state should use cache
            policy2, value2 = evaluator.evaluate(sample_game_state)
            
            # Results should be consistent
            assert policy1.shape == policy2.shape
            assert isinstance(value1, (float, np.floating))
            assert isinstance(value2, (float, np.floating))
    
    def test_alphazero_evaluator_error_handling(self, mock_model):
        """Test error handling in AlphaZeroEvaluator"""
        config = EvaluatorConfig(device='cpu')
        evaluator = AlphaZeroEvaluator(mock_model, config)
        
        # Configure mock to raise an error for testing
        mock_model.side_effect = RuntimeError("Model forward failed")
        
        # Test with invalid input
        invalid_state = np.random.rand(2, 15, 15).astype(np.float32)
        
        with pytest.raises(RuntimeError):
            evaluator.evaluate(invalid_state)


class TestEvaluatorIntegration:
    """Integration tests for evaluator components"""
    
    def test_evaluator_interface_consistency(self, mock_evaluator, sample_game_state):
        """Test that all evaluators follow the same interface"""
        states = sample_game_state.reshape(1, 3, 15, 15)
        
        # Test MockEvaluator with batch evaluation
        mock_policies, mock_values = mock_evaluator.evaluate_batch(states)
        
        # Check output shapes and types are consistent
        assert mock_policies.shape == (1, 225)
        assert mock_values.shape == (1,)
        assert isinstance(mock_policies, np.ndarray)
        assert isinstance(mock_values, np.ndarray)
    
    def test_evaluator_performance_comparison(self, sample_game_state):
        """Test relative performance of different evaluators"""
        # Create batch by repeating the sample state
        states = np.tile(sample_game_state, (10, 1, 1, 1))  # Batch of 10
        
        import time
        
        # Test MockEvaluator performance
        mock_evaluator = MockEvaluator()
        start_time = time.time()
        for _ in range(10):
            mock_evaluator.evaluate_batch(states)
        mock_time = time.time() - start_time
        
        # MockEvaluator should be very fast
        assert mock_time < 1.0, f"MockEvaluator too slow: {mock_time:.3f}s"