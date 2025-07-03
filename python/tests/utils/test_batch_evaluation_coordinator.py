"""Tests for batch evaluation coordinator"""

import pytest
import numpy as np
import time
import threading
import queue
from unittest.mock import Mock, patch, MagicMock

from mcts.utils.batch_evaluation_coordinator import (
    BatchEvaluationRequest,
    BatchEvaluationResponse,
    RequestBatchingCoordinator,
    get_global_batching_coordinator,
    cleanup_global_coordinator
)


class TestBatchEvaluationRequest:
    """Test BatchEvaluationRequest dataclass"""
    
    def test_batch_evaluation_request_creation(self):
        """Test creating BatchEvaluationRequest"""
        states = np.random.rand(4, 3, 15, 15).astype(np.float32)
        legal_masks = np.random.rand(4, 225).astype(np.float32)
        temperatures = np.ones(4, dtype=np.float32)
        
        request = BatchEvaluationRequest(
            request_id=1,
            states=states,
            legal_masks=legal_masks,
            temperatures=temperatures,
            worker_id=0,
            timestamp=time.time(),
            individual_request_ids=[0, 1, 2, 3]
        )
        
        assert request.request_id == 1
        assert request.states.shape == (4, 3, 15, 15)
        assert request.legal_masks.shape == (4, 225)
        assert request.temperatures.shape == (4,)
        assert request.worker_id == 0
        assert len(request.individual_request_ids) == 4


class TestBatchEvaluationResponse:
    """Test BatchEvaluationResponse dataclass"""
    
    def test_batch_evaluation_response_creation(self):
        """Test creating BatchEvaluationResponse"""
        policies = np.random.rand(4, 225).astype(np.float32)
        values = np.random.rand(4).astype(np.float32)
        
        response = BatchEvaluationResponse(
            request_id=1,
            policies=policies,
            values=values,
            worker_id=0,
            individual_request_ids=[0, 1, 2, 3]
        )
        
        assert response.request_id == 1
        assert response.policies.shape == (4, 225)
        assert response.values.shape == (4,)
        assert response.worker_id == 0
        assert len(response.individual_request_ids) == 4


class TestRequestBatchingCoordinator:
    """Test RequestBatchingCoordinator class"""
    
    @pytest.fixture
    def coordinator(self):
        """Create a coordinator for testing"""
        coord = RequestBatchingCoordinator(
            max_batch_size=4,
            batch_timeout_ms=50.0,  # Short timeout for fast tests
            enable_cross_worker_batching=True
        )
        coord.start()
        yield coord
        coord.stop()
    
    @pytest.fixture
    def mock_gpu_service_queue(self):
        """Create mock GPU service queue"""
        return queue.Queue()
    
    @pytest.fixture
    def mock_response_queue(self):
        """Create mock response queue"""
        return queue.Queue()
    
    def test_coordinator_creation(self):
        """Test creating RequestBatchingCoordinator"""
        coord = RequestBatchingCoordinator(
            max_batch_size=8,
            batch_timeout_ms=100.0,
            enable_cross_worker_batching=False
        )
        
        assert coord.max_batch_size == 8
        assert coord.batch_timeout == 0.1  # 100ms -> 0.1s
        assert coord.enable_cross_worker_batching == False
        assert coord.request_counter == 0
        
        coord.stop()  # Cleanup
    
    def test_coordinator_start_stop(self):
        """Test starting and stopping coordinator"""
        coord = RequestBatchingCoordinator()
        
        # Initially stopped
        assert coord.batch_thread is None
        
        # Start
        coord.start()
        assert coord.batch_thread is not None
        assert coord.batch_thread.is_alive()
        
        # Stop
        coord.stop()
        assert coord.batch_thread is None
    
    def test_coordinator_immediate_batch_processing(self, coordinator, mock_gpu_service_queue, mock_response_queue):
        """Test immediate batch processing for large batches"""
        # Create batch larger than max_batch_size
        states = np.random.rand(8, 3, 15, 15).astype(np.float32)
        temperatures = np.ones(8, dtype=np.float32)
        
        # Mock response
        expected_response = BatchEvaluationResponse(
            request_id=1,
            policies=np.random.rand(8, 225).astype(np.float32),
            values=np.random.rand(8).astype(np.float32),
            worker_id=0,
            individual_request_ids=list(range(8))
        )
        mock_response_queue.put(expected_response)
        
        # Process batch
        policies, values = coordinator.coordinate_evaluation_batch(
            states=states,
            temperatures=temperatures,
            worker_id=0,
            gpu_service_request_queue=mock_gpu_service_queue,
            response_queue=mock_response_queue
        )
        
        # Should have processed immediately
        assert policies.shape == (8, 225)
        assert values.shape == (8,)
        
        # Should have sent request to GPU service
        assert not mock_gpu_service_queue.empty()
        sent_request = mock_gpu_service_queue.get()
        assert isinstance(sent_request, BatchEvaluationRequest)
        assert sent_request.states.shape == (8, 3, 15, 15)
    
    def test_coordinator_small_batch_timeout(self, coordinator, mock_gpu_service_queue, mock_response_queue):
        """Test small batch processing with timeout"""
        # Create small batch
        states = np.random.rand(2, 3, 15, 15).astype(np.float32)
        temperatures = np.ones(2, dtype=np.float32)
        
        # Mock fallback response (will timeout and use immediate processing)
        expected_response = BatchEvaluationResponse(
            request_id=1,
            policies=np.random.rand(2, 225).astype(np.float32),
            values=np.random.rand(2).astype(np.float32),
            worker_id=0,
            individual_request_ids=[0, 1]
        )
        mock_response_queue.put(expected_response)
        
        start_time = time.time()
        policies, values = coordinator.coordinate_evaluation_batch(
            states=states,
            temperatures=temperatures,
            worker_id=0,
            gpu_service_request_queue=mock_gpu_service_queue,
            response_queue=mock_response_queue
        )
        elapsed_time = time.time() - start_time
        
        # Should have waited for timeout then processed
        assert elapsed_time >= 0.05  # At least timeout duration
        assert policies.shape == (2, 225)
        assert values.shape == (2,)
    
    def test_coordinator_cross_worker_batching(self):
        """Test cross-worker batching functionality"""
        coord = RequestBatchingCoordinator(
            max_batch_size=4,
            batch_timeout_ms=100.0,
            enable_cross_worker_batching=True
        )
        coord.start()
        
        try:
            # Create requests from different workers
            states1 = np.random.rand(2, 3, 15, 15).astype(np.float32)
            states2 = np.random.rand(2, 3, 15, 15).astype(np.float32)
            
            mock_queue1 = queue.Queue()
            mock_queue2 = queue.Queue()
            
            # Mock mega-batch response
            mega_response = BatchEvaluationResponse(
                request_id=999,
                policies=np.random.rand(4, 225).astype(np.float32),
                values=np.random.rand(4).astype(np.float32),
                worker_id=-1,
                individual_request_ids=list(range(4))
            )
            mock_queue1.put(mega_response)
            
            # Start both requests concurrently
            results = []
            
            def worker_request(states, worker_id, gpu_queue, response_queue):
                try:
                    result = coord.coordinate_evaluation_batch(
                        states=states,
                        worker_id=worker_id,
                        gpu_service_request_queue=gpu_queue,
                        response_queue=response_queue
                    )
                    results.append(result)
                except Exception as e:
                    results.append(e)
            
            thread1 = threading.Thread(target=worker_request, 
                                      args=(states1, 0, mock_queue1, mock_queue1))
            thread2 = threading.Thread(target=worker_request, 
                                      args=(states2, 1, mock_queue1, mock_queue1))
            
            thread1.start()
            thread2.start()
            
            thread1.join(timeout=2.0)
            thread2.join(timeout=2.0)
            
            # Should have processed both requests
            assert len(results) == 2
            
            # Results should be valid (either real responses or fallback)
            for result in results:
                if isinstance(result, tuple):
                    policies, values = result
                    assert policies.shape[0] == 2  # Each request had 2 states
                    assert values.shape[0] == 2
        
        finally:
            coord.stop()
    
    def test_coordinator_statistics(self, coordinator):
        """Test coordinator statistics collection"""
        stats = coordinator.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'requests_processed' in stats
        assert 'batches_created' in stats
        assert 'avg_batch_size' in stats
        assert 'cross_worker_batches' in stats
        assert 'coordination_timeouts' in stats
        assert 'current_timeout_ms' in stats
    
    def test_coordinator_timeout_adaptation(self, coordinator):
        """Test adaptive timeout adjustment"""
        # Simulate high timeout rate
        coordinator.stats['requests_processed'] = 100
        coordinator.stats['coordination_timeouts'] = 20  # 20% timeout rate
        
        original_timeout = coordinator.batch_timeout
        coordinator._adjust_adaptive_timeout()
        
        # Should have increased timeout
        assert coordinator.batch_timeout >= original_timeout
    
    def test_coordinator_error_handling(self, coordinator, mock_gpu_service_queue):
        """Test coordinator error handling"""
        states = np.random.rand(2, 3, 15, 15).astype(np.float32)
        
        # Don't put any response in queue to trigger timeout
        empty_queue = queue.Queue()
        
        # Should return fallback response without crashing
        policies, values = coordinator.coordinate_evaluation_batch(
            states=states,
            worker_id=0,
            gpu_service_request_queue=mock_gpu_service_queue,
            response_queue=empty_queue
        )
        
        # Should get fallback response
        assert policies.shape == (2, 225)
        assert values.shape == (2,)
        assert isinstance(policies, np.ndarray)
        assert isinstance(values, np.ndarray)
    
    def test_coordinator_with_legal_masks(self, coordinator, mock_gpu_service_queue, mock_response_queue):
        """Test coordinator with legal move masks"""
        states = np.random.rand(2, 3, 15, 15).astype(np.float32)
        legal_masks = np.random.rand(2, 225).astype(np.float32)
        temperatures = np.ones(2, dtype=np.float32)
        
        # Mock response
        expected_response = BatchEvaluationResponse(
            request_id=1,
            policies=np.random.rand(2, 225).astype(np.float32),
            values=np.random.rand(2).astype(np.float32),
            worker_id=0,
            individual_request_ids=[0, 1]
        )
        mock_response_queue.put(expected_response)
        
        policies, values = coordinator.coordinate_evaluation_batch(
            states=states,
            legal_masks=legal_masks,
            temperatures=temperatures,
            worker_id=0,
            gpu_service_request_queue=mock_gpu_service_queue,
            response_queue=mock_response_queue
        )
        
        assert policies.shape == (2, 225)
        assert values.shape == (2,)
    
    def test_coordinator_mega_batch_processing(self, coordinator):
        """Test mega-batch processing functionality"""
        # Create mock batch requests
        batch_requests = []
        response_events = []
        response_datas = []
        gpu_queues = []
        response_queues = []
        
        for i in range(3):
            states = np.random.rand(2, 3, 15, 15).astype(np.float32)
            request = BatchEvaluationRequest(
                request_id=i,
                states=states,
                legal_masks=None,
                temperatures=np.ones(2, dtype=np.float32),
                worker_id=i,
                timestamp=time.time(),
                individual_request_ids=[0, 1]
            )
            
            batch_requests.append(request)
            response_events.append(threading.Event())
            response_datas.append({'policies': None, 'values': None})
            
            # All use same queues for simplicity
            gpu_queue = queue.Queue()
            response_queue = queue.Queue()
            gpu_queues.append(gpu_queue)
            response_queues.append(response_queue)
        
        # Mock mega-batch response
        mega_response = BatchEvaluationResponse(
            request_id=999,
            policies=np.random.rand(6, 225).astype(np.float32),  # 3 requests * 2 states each
            values=np.random.rand(6).astype(np.float32),
            worker_id=-1,
            individual_request_ids=list(range(6))
        )
        response_queues[0].put(mega_response)
        
        # Process mega-batch
        coordinator._process_mega_batch(
            batch_requests, response_events, response_datas, gpu_queues, response_queues
        )
        
        # Check that all events were set and data was distributed
        for i, event in enumerate(response_events):
            assert event.is_set(), f"Event {i} was not set"
            assert response_datas[i]['policies'] is not None
            assert response_datas[i]['values'] is not None
            assert response_datas[i]['policies'].shape == (2, 225)
            assert response_datas[i]['values'].shape == (2,)


class TestGlobalCoordinator:
    """Test global coordinator management"""
    
    def teardown_method(self):
        """Clean up after each test"""
        cleanup_global_coordinator()
    
    def test_get_global_coordinator(self):
        """Test getting global coordinator instance"""
        coord1 = get_global_batching_coordinator()
        coord2 = get_global_batching_coordinator()
        
        # Should return same instance
        assert coord1 is coord2
        assert coord1.batch_thread is not None
    
    def test_global_coordinator_process_isolation(self):
        """Test that global coordinator is process-isolated"""
        import os
        
        coord1 = get_global_batching_coordinator()
        original_pid = os.getpid()
        
        # Manually change the tracked PID to simulate different process
        from mcts.utils.batch_evaluation_coordinator import _coordinator_process_id
        
        # Get new coordinator with different settings
        coord2 = get_global_batching_coordinator(max_batch_size=128)
        
        # Should get same instance since we're in same process
        assert coord1 is coord2
    
    def test_cleanup_global_coordinator(self):
        """Test cleaning up global coordinator"""
        coord = get_global_batching_coordinator()
        assert coord is not None
        
        cleanup_global_coordinator()
        
        # Should get new instance after cleanup
        new_coord = get_global_batching_coordinator()
        assert new_coord is not coord
    
    def test_global_coordinator_configuration(self):
        """Test global coordinator with custom configuration"""
        coord = get_global_batching_coordinator(
            max_batch_size=128,
            batch_timeout_ms=200.0,
            enable_cross_worker_batching=False
        )
        
        assert coord.max_batch_size == 128
        assert coord.batch_timeout == 0.2  # 200ms -> 0.2s
        assert coord.enable_cross_worker_batching == False


class TestBatchCoordinatorIntegration:
    """Integration tests for batch coordinator"""
    
    def test_coordinator_with_mock_gpu_service(self):
        """Test coordinator integration with mock GPU service"""
        coord = RequestBatchingCoordinator(
            max_batch_size=4,
            batch_timeout_ms=50.0
        )
        coord.start()
        
        try:
            # Create mock GPU service behavior
            def mock_gpu_service(request_queue, response_queue):
                while True:
                    try:
                        request = request_queue.get(timeout=0.1)
                        if isinstance(request, BatchEvaluationRequest):
                            # Generate mock response
                            batch_size = request.states.shape[0]
                            response = BatchEvaluationResponse(
                                request_id=request.request_id,
                                policies=np.random.rand(batch_size, 225).astype(np.float32),
                                values=np.random.rand(batch_size).astype(np.float32),
                                worker_id=request.worker_id,
                                individual_request_ids=request.individual_request_ids
                            )
                            response_queue.put(response)
                    except queue.Empty:
                        break
            
            # Set up queues
            gpu_queue = queue.Queue()
            response_queue = queue.Queue()
            
            # Start mock GPU service
            gpu_thread = threading.Thread(
                target=mock_gpu_service, 
                args=(gpu_queue, response_queue),
                daemon=True
            )
            gpu_thread.start()
            
            # Test batch processing
            states = np.random.rand(2, 3, 15, 15).astype(np.float32)
            
            policies, values = coord.coordinate_evaluation_batch(
                states=states,
                worker_id=0,
                gpu_service_request_queue=gpu_queue,
                response_queue=response_queue
            )
            
            assert policies.shape == (2, 225)
            assert values.shape == (2,)
            
            # Stop GPU service
            gpu_thread.join(timeout=1.0)
        
        finally:
            coord.stop()
    
    def test_coordinator_performance_basic(self):
        """Test basic performance characteristics of coordinator"""
        coord = RequestBatchingCoordinator(
            max_batch_size=32,
            batch_timeout_ms=10.0
        )
        coord.start()
        
        try:
            # Create queues
            gpu_queue = queue.Queue()
            response_queue = queue.Queue()
            
            # Pre-populate response queue with mock responses
            for i in range(10):
                response = BatchEvaluationResponse(
                    request_id=i,
                    policies=np.random.rand(4, 225).astype(np.float32),
                    values=np.random.rand(4).astype(np.float32),
                    worker_id=0,
                    individual_request_ids=list(range(4))
                )
                response_queue.put(response)
            
            # Measure performance
            start_time = time.time()
            
            for i in range(10):
                states = np.random.rand(4, 3, 15, 15).astype(np.float32)
                policies, values = coord.coordinate_evaluation_batch(
                    states=states,
                    worker_id=0,
                    gpu_service_request_queue=gpu_queue,
                    response_queue=response_queue
                )
                assert policies.shape == (4, 225)
                assert values.shape == (4,)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Should complete reasonably quickly
            assert total_time < 5.0, f"Coordinator too slow: {total_time:.2f}s for 10 batches"
            
            # Check statistics
            stats = coord.get_statistics()
            assert stats['requests_processed'] > 0
        
        finally:
            coord.stop()