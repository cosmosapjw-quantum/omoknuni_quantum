"""Tests for optimized remote evaluator

Tests cover:
- OptimizedRemoteEvaluator initialization
- Batch coordination integration
- Synchronous evaluation
- Error handling
- Performance statistics
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock, call
import multiprocessing as mp
import queue

from mcts.utils.optimized_remote_evaluator import OptimizedRemoteEvaluator
from mcts.utils.batch_evaluation_coordinator import (
    RequestBatchingCoordinator, BatchEvaluationRequest, BatchEvaluationResponse
)


@pytest.fixture
def evaluator_config():
    """Create evaluator configuration"""
    # Configuration for OptimizedRemoteEvaluator
    return {
        'batch_timeout': 0.1,
        'enable_coordination': True,
        'max_coordination_batch_size': 64
    }


@pytest.fixture
def mock_coordinator():
    """Create mock batch coordinator"""
    coordinator = Mock(spec=RequestBatchingCoordinator)
    coordinator.coordinate_evaluation_batch.return_value = (
        np.ones((1, 225)) / 225,  # Policies (batch format)
        np.array([0.5])  # Values (batch format)
    )
    return coordinator


@pytest.fixture
def remote_evaluator(evaluator_config, mock_coordinator):
    """Create OptimizedRemoteEvaluator instance"""
    request_queue = Mock()
    response_queue = Mock()
    
    with patch('mcts.utils.optimized_remote_evaluator.get_global_batching_coordinator', 
               return_value=mock_coordinator):
        evaluator = OptimizedRemoteEvaluator(
            request_queue=request_queue,
            response_queue=response_queue,
            action_size=225,  # 15x15 Gomoku
            worker_id=0,
            **evaluator_config
        )
        return evaluator


@pytest.fixture
def sample_states():
    """Create sample game states"""
    return [
        np.random.randn(3, 15, 15).astype(np.float32)
        for _ in range(4)
    ]


class TestOptimizedRemoteEvaluatorInitialization:
    """Test OptimizedRemoteEvaluator initialization"""
    
    def test_initialization(self, evaluator_config):
        """Test basic initialization"""
        request_queue = Mock()
        response_queue = Mock()
        
        with patch('mcts.utils.optimized_remote_evaluator.get_global_batching_coordinator') as mock_get_coordinator:
            evaluator = OptimizedRemoteEvaluator(
                request_queue=request_queue,
                response_queue=response_queue,
                action_size=225,
                worker_id=1,
                **evaluator_config
            )
            
            assert evaluator.worker_id == 1
            assert evaluator.batch_timeout == 0.1
            assert evaluator.enable_coordination == True
            assert evaluator.request_queue == request_queue
            assert evaluator.response_queue == response_queue
            assert evaluator.action_size == 225
            
    def test_initialization_without_coordination(self, evaluator_config):
        """Test initialization without coordination"""
        request_queue = Mock()
        response_queue = Mock()
        
        evaluator = OptimizedRemoteEvaluator(
            request_queue=request_queue,
            response_queue=response_queue,
            action_size=225,
            worker_id=2,
            enable_coordination=False
        )
        
        assert evaluator.coordinator is None
        assert evaluator.enable_coordination == False
        
    def test_coordinator_setup(self, remote_evaluator):
        """Test coordinator setup"""
        assert hasattr(remote_evaluator, 'coordinator')
        assert remote_evaluator.coordinator is not None
        assert remote_evaluator.enable_coordination == True
        
    def test_statistics_initialization(self, remote_evaluator):
        """Test statistics tracking initialization"""
        stats = remote_evaluator.get_performance_stats()
        
        assert stats['total_evaluations'] == 0
        assert stats['batch_evaluations'] == 0
        assert stats['coordinated_evaluations'] == 0
        assert stats['avg_batch_size'] == 0.0
        assert stats['total_time'] == 0.0
        assert stats['coordination_time'] == 0.0
        assert stats['evaluations_per_second'] == 0.0
        assert stats['coordination_overhead_percent'] == 0.0


class TestSynchronousEvaluation:
    """Test synchronous evaluation methods"""
    
    def test_evaluate_single_state(self, remote_evaluator, sample_states):
        """Test single state evaluation"""
        state = sample_states[0]
        
        # Mock coordinator to return proper batch format
        remote_evaluator.coordinator.coordinate_evaluation_batch.return_value = (
            np.ones((1, 225)) / 225,  # Policies
            np.array([0.5])  # Values
        )
        
        policy, value = remote_evaluator.evaluate(state)
        
        assert isinstance(policy, np.ndarray)
        assert policy.shape == (225,)
        assert isinstance(value, float)
        assert -1 <= value <= 1
        
        # Check coordinator was called
        remote_evaluator.coordinator.coordinate_evaluation_batch.assert_called_once()
        
    def test_evaluate_multiple_states(self, remote_evaluator, sample_states):
        """Test multiple single state evaluations"""
        # Mock coordinator to return proper batch format
        remote_evaluator.coordinator.coordinate_evaluation_batch.return_value = (
            np.ones((1, 225)) / 225,  # Policies
            np.array([0.5])  # Values
        )
        
        # Evaluate multiple states
        for state in sample_states[:2]:
            policy, value = remote_evaluator.evaluate(state)
            assert policy.shape == (225,)
            assert isinstance(value, float)
        
        # Check coordinator was called for each evaluation
        assert remote_evaluator.coordinator.coordinate_evaluation_batch.call_count == 2
        
    def test_evaluate_batch(self, remote_evaluator, sample_states):
        """Test batch evaluation"""
        # For batch of 4 (small batch), coordinator will be used
        remote_evaluator.coordinator.coordinate_evaluation_batch.return_value = (
            np.ones((4, 225)) / 225,  # Policies
            np.array([0.5, 0.5, 0.5, 0.5])  # Values
        )
        
        # Convert states to numpy array
        states_array = np.array(sample_states)
        policies, values = remote_evaluator.evaluate_batch(states_array)
        
        assert isinstance(policies, np.ndarray)
        assert policies.shape == (4, 225)
        assert isinstance(values, np.ndarray)
        assert values.shape == (4,)
        
        # For small batches, coordinator should be called
        remote_evaluator.coordinator.coordinate_evaluation_batch.assert_called()
            
    def test_evaluate_batch_with_temperatures(self, remote_evaluator, sample_states):
        """Test batch evaluation with different temperatures"""
        # Mock coordinator return value
        remote_evaluator.coordinator.coordinate_evaluation_batch.return_value = (
            np.ones((4, 225)) / 225,  # Policies
            np.array([0.5, 0.5, 0.5, 0.5])  # Values
        )
        
        states_array = np.array(sample_states)
        temperatures = np.array([1.0, 0.5, 2.0, 1.5])
        
        policies, values = remote_evaluator.evaluate_batch(states_array, temperatures=temperatures)
        
        assert policies.shape == (4, 225)
        assert values.shape == (4,)
        
        # Check that temperatures were passed to coordinator
        call_args = remote_evaluator.coordinator.coordinate_evaluation_batch.call_args
        assert 'temperatures' in call_args[1]
        np.testing.assert_array_equal(call_args[1]['temperatures'], temperatures)


class TestBatchCoordinatorIntegration:
    """Test integration with batch coordinator"""
    
    def test_worker_id_propagation(self, remote_evaluator, sample_states):
        """Test worker ID is correctly propagated"""
        state = sample_states[0]
        remote_evaluator.evaluate(state)
        
        call_args = remote_evaluator.coordinator.coordinate_evaluation_batch.call_args
        assert call_args[1]['worker_id'] == 0
        
    def test_coordinator_timeout_handling(self, remote_evaluator, sample_states):
        """Test coordinator timeout fallback"""
        # Make coordinator raise exception
        remote_evaluator.coordinator.coordinate_evaluation_batch.side_effect = TimeoutError("Timeout")
        
        # Mock response queue for fallback
        batch_response = Mock(spec=BatchEvaluationResponse)
        batch_response.policies = np.ones((1, 225)) / 225
        batch_response.values = np.array([0.5])
        remote_evaluator.response_queue.get.return_value = batch_response
        
        # Should fallback to direct evaluation
        policy, value = remote_evaluator.evaluate(sample_states[0])
        
        assert policy.shape == (225,)
        assert isinstance(value, float)
        
        # Request should have been sent to queue
        remote_evaluator.request_queue.put.assert_called_once()
        
    def test_coordinator_batch_optimization(self, remote_evaluator, sample_states):
        """Test batch size optimization"""
        # Small batch should use coordinator
        small_batch = np.array(sample_states[:2])
        remote_evaluator.evaluate_batch(small_batch)
        assert remote_evaluator.coordinator.coordinate_evaluation_batch.called
        
        # Reset mock
        remote_evaluator.coordinator.coordinate_evaluation_batch.reset_mock()
        
        # Large batch should bypass coordinator
        large_batch = np.array([np.random.randn(3, 15, 15) for _ in range(40)])
        batch_response = Mock(spec=BatchEvaluationResponse)
        batch_response.policies = np.ones((40, 225)) / 225
        batch_response.values = np.ones(40) * 0.5
        remote_evaluator.response_queue.get.return_value = batch_response
        
        remote_evaluator.evaluate_batch(large_batch)
        assert not remote_evaluator.coordinator.coordinate_evaluation_batch.called


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_queue_full_error(self, remote_evaluator, sample_states):
        """Test handling of full request queue"""
        remote_evaluator.request_queue.put.side_effect = queue.Full("Queue full")
        
        # Mock coordinator to also fail to force fallback
        remote_evaluator.coordinator.coordinate_evaluation_batch.side_effect = Exception("Coordinator failed")
        
        # Should return fallback response
        policies, values = remote_evaluator.evaluate_batch(np.array(sample_states))
        
        assert policies.shape == (4, 225)
        assert values.shape == (4,)
        assert np.all(values == 0.0)  # Fallback values are zeros
        
    def test_response_timeout(self, remote_evaluator, sample_states):
        """Test response timeout handling"""
        remote_evaluator.response_queue.get.side_effect = queue.Empty("Timeout")
        
        # Bypass coordinator for direct test
        remote_evaluator.coordinator = None
        
        # Should return fallback response
        policies, values = remote_evaluator.evaluate_batch(np.array(sample_states))
        
        assert policies.shape == (4, 225)
        assert values.shape == (4,)
        
    def test_invalid_response_type(self, remote_evaluator, sample_states):
        """Test handling of invalid response type"""
        remote_evaluator.response_queue.get.return_value = "Invalid response"
        
        # Bypass coordinator
        remote_evaluator.coordinator = None
        
        # Should return fallback response
        policies, values = remote_evaluator.evaluate_batch(np.array(sample_states))
        
        assert policies.shape == (4, 225)
        assert values.shape == (4,)


class TestPerformanceOptimizations:
    """Test performance optimization features"""
    
    def test_batch_size_optimization(self, remote_evaluator):
        """Test batch size affects evaluation strategy"""
        # Small batch uses coordinator
        small_states = np.random.randn(10, 3, 15, 15)
        remote_evaluator.coordinator.coordinate_evaluation_batch.return_value = (
            np.ones((10, 225)) / 225,
            np.ones(10) * 0.5
        )
        
        remote_evaluator.evaluate_batch(small_states)
        assert remote_evaluator.stats['coordinated_evaluations'] == 10
        
        # Large batch uses direct evaluation
        large_states = np.random.randn(50, 3, 15, 15)
        batch_response = Mock(spec=BatchEvaluationResponse)
        batch_response.policies = np.ones((50, 225)) / 225
        batch_response.values = np.ones(50) * 0.5
        remote_evaluator.response_queue.get.return_value = batch_response
        
        remote_evaluator.evaluate_batch(large_states)
        assert remote_evaluator.stats['batch_evaluations'] == 50
        
    def test_statistics_tracking(self, remote_evaluator, sample_states):
        """Test performance statistics tracking"""
        # Perform evaluations
        remote_evaluator.evaluate(sample_states[0])
        remote_evaluator.evaluate_batch(np.array(sample_states))
        
        stats = remote_evaluator.get_performance_stats()
        
        assert stats['total_evaluations'] == 5  # 1 + 4
        assert stats['evaluations_per_second'] > 0
        assert stats['total_time'] > 0
        
    def test_reset_statistics(self, remote_evaluator, sample_states):
        """Test statistics reset"""
        # Perform some evaluations
        remote_evaluator.evaluate(sample_states[0])
        
        # Reset stats
        remote_evaluator.reset_stats()
        
        stats = remote_evaluator.get_performance_stats()
        assert stats['total_evaluations'] == 0
        assert stats['batch_evaluations'] == 0
        assert stats['coordinated_evaluations'] == 0


class TestIntegration:
    """Test integration scenarios"""
    
    def test_temperature_handling(self, remote_evaluator):
        """Test temperature parameter handling"""
        state = np.random.randn(3, 15, 15)
        
        # Test scalar temperature
        policy1, value1 = remote_evaluator.evaluate(state, temperature=0.5)
        assert policy1.shape == (225,)
        
        # Test array temperature for batch
        states = np.random.randn(3, 3, 15, 15)
        temps = [1.0, 0.5, 2.0]
        
        # Mock coordinator to return correct batch size
        remote_evaluator.coordinator.coordinate_evaluation_batch.return_value = (
            np.ones((3, 225)) / 225,  # Policies
            np.array([0.5, 0.5, 0.5])  # Values
        )
        
        policies, values = remote_evaluator.evaluate_batch(states, temperatures=temps)
        assert policies.shape == (3, 225)
        
    def test_legal_mask_handling(self, remote_evaluator):
        """Test legal mask parameter handling"""
        state = np.random.randn(3, 15, 15)
        legal_mask = np.ones(225)
        legal_mask[0] = 0  # First move illegal
        
        remote_evaluator.coordinator.coordinate_evaluation_batch.return_value = (
            np.ones((1, 225)) / 224,  # Adjusted for legal moves
            np.array([0.5])
        )
        
        policy, value = remote_evaluator.evaluate(state, legal_mask=legal_mask)
        
        # Check legal mask was passed
        call_args = remote_evaluator.coordinator.coordinate_evaluation_batch.call_args
        assert 'legal_masks' in call_args[1]
        assert call_args[1]['legal_masks'] is not None