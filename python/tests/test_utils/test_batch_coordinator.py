"""
Comprehensive tests for batch evaluation coordinator

This module tests the RequestBatchingCoordinator which provides:
- Request batching to reduce communication overhead
- Cross-worker coordination for efficiency
- Adaptive timeout handling
- Process-safe global instance management
"""

import pytest
import numpy as np
import torch
import time
import threading
import queue
import multiprocessing
from unittest.mock import Mock, patch, MagicMock
import os

from mcts.utils.batch_evaluation_coordinator import (
    RequestBatchingCoordinator,
    BatchEvaluationRequest,
    BatchEvaluationResponse,
    get_global_batching_coordinator,
    cleanup_global_coordinator
)


# Module-level function for multiprocessing pickling
def _get_subprocess_coordinator_id(queue):
    """Helper function for test_process_isolation that can be pickled"""
    coord = get_global_batching_coordinator()
    queue.put(id(coord))
    cleanup_global_coordinator()


class TestRequestBatchingCoordinator:
    """Test RequestBatchingCoordinator initialization and configuration"""
    
    def test_basic_initialization(self):
        """Test basic coordinator initialization"""
        coordinator = RequestBatchingCoordinator(
            max_batch_size=32,
            batch_timeout_ms=50.0,
            enable_cross_worker_batching=True
        )
        
        assert coordinator.max_batch_size == 32
        assert coordinator.batch_timeout == 0.05  # 50ms in seconds
        assert coordinator.batch_thread is None  # Not started yet
        assert coordinator.request_counter == 0
        
    def test_multiprocessing_detection(self):
        """Test multiprocessing context detection"""
        coordinator = RequestBatchingCoordinator()
        
        # Test various detection methods
        with patch.dict(os.environ, {'MULTIPROCESSING_FORKED': '1'}):
            assert coordinator._detect_multiprocessing_context() == True
            
        with patch.dict(os.environ, {'MP_MAIN_FILE': 'test.py'}):
            assert coordinator._detect_multiprocessing_context() == True
            
        # Clear environment
        with patch.dict(os.environ, {}, clear=True):
            # Mock process name check
            with patch('multiprocessing.current_process') as mock_process:
                mock_process.return_value.name = 'Process-1'
                assert coordinator._detect_multiprocessing_context() == True
                
                mock_process.return_value.name = 'MainProcess'
                assert coordinator._detect_multiprocessing_context() == False
                
    def test_multiprocessing_disables_cross_worker_batching(self):
        """Test that multiprocessing context disables cross-worker batching"""
        with patch.object(RequestBatchingCoordinator, '_detect_multiprocessing_context', return_value=True):
            coordinator = RequestBatchingCoordinator(enable_cross_worker_batching=True)
            assert coordinator.enable_cross_worker_batching == False
            
    def test_start_stop(self):
        """Test starting and stopping coordinator"""
        coordinator = RequestBatchingCoordinator()
        
        # Start
        coordinator.start()
        assert coordinator.batch_thread is not None
        assert coordinator.batch_thread.is_alive()
        
        # Start again should be idempotent
        coordinator.start()
        
        # Stop
        coordinator.stop()
        assert coordinator.batch_thread is None
        
        # Stop again should be idempotent
        coordinator.stop()


class TestBatchCoordination:
    """Test batch coordination functionality"""
    
    def test_immediate_batch_processing(self):
        """Test immediate processing for large batches"""
        coordinator = RequestBatchingCoordinator(max_batch_size=16)
        coordinator.start()
        
        try:
            # Create mock queues
            request_queue = queue.Queue()
            response_queue = queue.Queue()
            
            # Large batch should process immediately
            states = np.random.rand(20, 3, 15, 15).astype(np.float32)
            
            # Mock response
            expected_policies = np.random.rand(20, 225).astype(np.float32)
            expected_values = np.random.rand(20).astype(np.float32)
            
            def mock_gpu_service():
                req = request_queue.get(timeout=1.0)
                assert isinstance(req, BatchEvaluationRequest)
                assert req.states.shape[0] == 20
                
                resp = BatchEvaluationResponse(
                    request_id=req.request_id,
                    policies=expected_policies,
                    values=expected_values,
                    worker_id=req.worker_id,
                    individual_request_ids=req.individual_request_ids
                )
                response_queue.put(resp)
                
            # Start mock GPU service
            gpu_thread = threading.Thread(target=mock_gpu_service)
            gpu_thread.start()
            
            # Process batch
            policies, values = coordinator.coordinate_evaluation_batch(
                states=states,
                worker_id=0,
                gpu_service_request_queue=request_queue,
                response_queue=response_queue
            )
            
            gpu_thread.join()
            
            # Check results
            np.testing.assert_array_equal(policies, expected_policies)
            np.testing.assert_array_equal(values, expected_values)
            
            # Should have used immediate processing
            stats = coordinator.get_statistics()
            assert stats['fallback_to_direct'] == 1
            
        finally:
            coordinator.stop()
            
    def test_small_batch_coordination(self):
        """Test coordination of small batches"""
        coordinator = RequestBatchingCoordinator(
            max_batch_size=16,
            batch_timeout_ms=20.0
        )
        coordinator.start()
        
        try:
            # Create mock queues
            request_queue = queue.Queue()
            response_queue = queue.Queue()
            
            # Track requests
            received_requests = []
            
            def mock_gpu_service():
                while True:
                    try:
                        req = request_queue.get(timeout=0.5)
                        received_requests.append(req)
                        
                        # Create response
                        batch_size = req.states.shape[0]
                        resp = BatchEvaluationResponse(
                            request_id=req.request_id,
                            policies=np.random.rand(batch_size, 225).astype(np.float32),
                            values=np.random.rand(batch_size).astype(np.float32),
                            worker_id=req.worker_id,
                            individual_request_ids=req.individual_request_ids
                        )
                        response_queue.put(resp)
                    except queue.Empty:
                        break
                        
            # Start mock GPU service
            gpu_thread = threading.Thread(target=mock_gpu_service)
            gpu_thread.start()
            
            # Send multiple small batches from different threads
            results = []
            threads = []
            
            def send_batch(worker_id, batch_size):
                states = np.random.rand(batch_size, 3, 15, 15).astype(np.float32)
                policies, values = coordinator.coordinate_evaluation_batch(
                    states=states,
                    worker_id=worker_id,
                    gpu_service_request_queue=request_queue,
                    response_queue=response_queue
                )
                results.append((worker_id, policies, values))
                
            # Send small batches
            for i in range(3):
                t = threading.Thread(target=send_batch, args=(i, 4))
                threads.append(t)
                t.start()
                
            # Wait for all threads
            for t in threads:
                t.join(timeout=1.0)
                
            gpu_thread.join(timeout=1.0)
            
            # Should have coordinated batches
            assert len(received_requests) <= 3  # Should batch some together
            assert len(results) == 3
            
            # Check all results received
            for worker_id, policies, values in results:
                assert policies.shape == (4, 225)
                assert values.shape == (4,)
                
        finally:
            coordinator.stop()
            
    def test_adaptive_timeout(self):
        """Test adaptive timeout based on load"""
        coordinator = RequestBatchingCoordinator(
            max_batch_size=32,
            batch_timeout_ms=10.0
        )
        coordinator.start()
        
        try:
            # Test timeout calculation with different pending counts
            # Light load
            coordinator.pending_requests = [None] * 3
            
            request_queue = queue.Queue()
            response_queue = queue.Queue()
            
            # Create a slow GPU service
            def slow_gpu_service():
                time.sleep(0.5)  # Slow response
                
            gpu_thread = threading.Thread(target=slow_gpu_service)
            gpu_thread.start()
            
            # Should timeout and fallback
            start_time = time.time()
            states = np.random.rand(2, 3, 15, 15).astype(np.float32)
            
            # This should timeout
            with patch.object(coordinator, '_process_immediate_batch') as mock_immediate:
                mock_immediate.return_value = (
                    np.random.rand(2, 225).astype(np.float32),
                    np.random.rand(2).astype(np.float32)
                )
                
                policies, values = coordinator.coordinate_evaluation_batch(
                    states=states,
                    worker_id=0,
                    gpu_service_request_queue=request_queue,
                    response_queue=response_queue
                )
                
                elapsed = time.time() - start_time
                assert elapsed < 0.2  # Should timeout quickly
                assert mock_immediate.called
                
            gpu_thread.join(timeout=1.0)
            
        finally:
            coordinator.stop()


class TestBatchProcessing:
    """Test batch processing logic"""
    
    def test_batch_combination(self):
        """Test combining multiple requests into mega-batch"""
        coordinator = RequestBatchingCoordinator()
        
        # Create test requests
        requests = []
        for i in range(3):
            req = BatchEvaluationRequest(
                request_id=i+1,
                states=np.random.rand(4, 3, 15, 15).astype(np.float32),
                legal_masks=np.ones((4, 225), dtype=bool),
                temperatures=np.ones(4, dtype=np.float32),
                worker_id=i,
                timestamp=time.time(),
                individual_request_ids=list(range(4))
            )
            event = threading.Event()
            data = {'policies': None, 'values': None, 'error': None}
            
            # Mock queues
            gpu_queue = Mock()
            resp_queue = Mock()
            
            # Mock response
            mega_response = BatchEvaluationResponse(
                request_id=999,
                policies=np.random.rand(12, 225).astype(np.float32),
                values=np.random.rand(12).astype(np.float32),
                worker_id=-1,
                individual_request_ids=list(range(12))
            )
            resp_queue.get.return_value = mega_response
            
            requests.append((req, event, data, gpu_queue, resp_queue))
            
        # Process batch
        coordinator._process_request_batch(requests)
        
        # Check mega-batch was created - it uses the first worker's queue
        first_gpu_queue = requests[0][3]  # Get the first request's gpu_queue
        first_gpu_queue.put.assert_called_once()
        mega_request = first_gpu_queue.put.call_args[0][0]
        assert isinstance(mega_request, BatchEvaluationRequest)
        assert mega_request.states.shape[0] == 12  # 3 x 4
        
        # Check results distributed
        for _, event, data, _, _ in requests:
            assert event.is_set()
            assert data['policies'] is not None
            assert data['values'] is not None
            assert data['policies'].shape == (4, 225)
            assert data['values'].shape == (4,)
            
    def test_mixed_tensor_numpy_inputs(self):
        """Test handling mixed tensor and numpy inputs"""
        coordinator = RequestBatchingCoordinator()
        
        # Create mixed requests
        requests = []
        
        # NumPy request
        req1 = BatchEvaluationRequest(
            request_id=1,
            states=np.random.rand(2, 3, 15, 15).astype(np.float32),
            legal_masks=np.ones((2, 225), dtype=bool),
            temperatures=np.ones(2, dtype=np.float32),
            worker_id=0,
            timestamp=time.time(),
            individual_request_ids=[0, 1]
        )
        
        # Torch request
        req2 = BatchEvaluationRequest(
            request_id=2,
            states=torch.rand(3, 3, 15, 15, dtype=torch.float32),
            legal_masks=torch.ones(3, 225, dtype=torch.bool),
            temperatures=torch.ones(3, dtype=torch.float32),
            worker_id=1,
            timestamp=time.time(),
            individual_request_ids=[0, 1, 2]
        )
        
        # Mock response
        gpu_queue = Mock()
        resp_queue = Mock()
        mega_response = BatchEvaluationResponse(
            request_id=999,
            policies=np.random.rand(5, 225).astype(np.float32),
            values=np.random.rand(5).astype(np.float32),
            worker_id=-1,
            individual_request_ids=list(range(5))
        )
        resp_queue.get.return_value = mega_response
        
        for req in [req1, req2]:
            event = threading.Event()
            data = {'policies': None, 'values': None, 'error': None}
            requests.append((req, event, data, gpu_queue, resp_queue))
            
        # Process batch
        coordinator._process_request_batch(requests)
        
        # Should handle mixed inputs
        gpu_queue.put.assert_called_once()
        mega_request = gpu_queue.put.call_args[0][0]
        assert mega_request.states.shape[0] == 5
        assert isinstance(mega_request.states, np.ndarray)


class TestErrorHandling:
    """Test error handling and recovery"""
    
    def test_missing_queues_error(self):
        """Test error when GPU queues not provided"""
        coordinator = RequestBatchingCoordinator()
        coordinator.start()
        
        try:
            states = np.random.rand(4, 3, 15, 15).astype(np.float32)
            
            with pytest.raises(ValueError, match="GPU service queues not provided"):
                coordinator.coordinate_evaluation_batch(
                    states=states,
                    worker_id=0,
                    gpu_service_request_queue=None,
                    response_queue=None
                )
        finally:
            coordinator.stop()
            
    def test_gpu_service_timeout_handling(self):
        """Test handling GPU service timeout"""
        coordinator = RequestBatchingCoordinator()
        coordinator.start()
        
        try:
            request_queue = queue.Queue()
            response_queue = queue.Queue()
            
            # No GPU service running - will timeout
            states = np.random.rand(20, 3, 15, 15).astype(np.float32)
            
            # Should fallback gracefully
            policies, values = coordinator.coordinate_evaluation_batch(
                states=states,
                worker_id=0,
                gpu_service_request_queue=request_queue,
                response_queue=response_queue
            )
            
            # Should return fallback values
            assert policies.shape == (20, 225)
            assert values.shape == (20,)
            assert np.all(values == 0.0)  # Fallback zeros
            
        finally:
            coordinator.stop()
            
    def test_batch_processing_error_recovery(self):
        """Test error recovery in batch processing"""
        coordinator = RequestBatchingCoordinator()
        
        # Create request with error
        req = BatchEvaluationRequest(
            request_id=1,
            states=np.random.rand(2, 3, 15, 15).astype(np.float32),
            legal_masks=None,
            temperatures=np.ones(2),
            worker_id=0,
            timestamp=time.time(),
            individual_request_ids=[0, 1]
        )
        
        event = threading.Event()
        data = {'policies': None, 'values': None, 'error': None}
        
        # Mock queues that raise error
        gpu_queue = Mock()
        gpu_queue.put.side_effect = Exception("GPU queue error")
        resp_queue = Mock()
        
        requests = [(req, event, data, gpu_queue, resp_queue)]
        
        # Process batch - should handle error
        coordinator._process_request_batch(requests)
        
        # Error should be set
        assert event.is_set()
        assert data['error'] is not None
        assert "GPU queue error" in data['error']
        
    def test_coordination_loop_error_recovery(self):
        """Test coordination loop continues after errors"""
        coordinator = RequestBatchingCoordinator(batch_timeout_ms=10.0)
        
        # Mock process_request_batch to raise error once
        call_count = 0
        original_process = coordinator._process_request_batch
        
        def mock_process(requests):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Test error")
            return original_process(requests)
            
        coordinator._process_request_batch = mock_process
        
        # Start coordinator
        coordinator.start()
        
        try:
            # Add request to trigger error
            req = Mock()
            req.timestamp = time.time()
            coordinator.pending_requests.append((req, Mock(), {}, Mock(), Mock()))
            
            # Wait a bit
            time.sleep(0.05)
            
            # Coordinator should still be running
            assert coordinator.batch_thread.is_alive()
            
        finally:
            coordinator.stop()


class TestStatistics:
    """Test statistics tracking"""
    
    def test_statistics_tracking(self):
        """Test performance statistics tracking"""
        coordinator = RequestBatchingCoordinator()
        coordinator.start()
        
        try:
            # Process some requests
            request_queue = queue.Queue()
            response_queue = queue.Queue()
            
            # Mock GPU service with proper synchronization
            stop_event = threading.Event()
            def mock_gpu_service():
                while not stop_event.is_set():
                    try:
                        req = request_queue.get(timeout=0.1)
                        resp = BatchEvaluationResponse(
                            request_id=req.request_id,
                            policies=np.random.rand(req.states.shape[0], 225).astype(np.float32),
                            values=np.random.rand(req.states.shape[0]).astype(np.float32),
                            worker_id=req.worker_id,
                            individual_request_ids=req.individual_request_ids
                        )
                        response_queue.put(resp)
                    except queue.Empty:
                        continue
                        
            gpu_thread = threading.Thread(target=mock_gpu_service)
            gpu_thread.start()
            
            # Process a large batch that should trigger fallback_to_direct
            states = np.random.rand(65, 3, 15, 15).astype(np.float32)  # 65 > max_batch_size (64)
            coordinator.coordinate_evaluation_batch(
                states, worker_id=0,
                gpu_service_request_queue=request_queue,
                response_queue=response_queue
            )
            
            # Stop GPU service and wait
            stop_event.set()
            gpu_thread.join(timeout=1.0)
            
            # Check statistics
            stats = coordinator.get_statistics()
            assert stats['fallback_to_direct'] == 1
            assert 'coordination_success_rate' in stats
            assert 'timeout_rate' in stats
            
        finally:
            coordinator.stop()
            
    def test_cross_worker_batch_tracking(self):
        """Test tracking of cross-worker batches"""
        coordinator = RequestBatchingCoordinator()
        
        # Create requests from different workers
        requests = []
        for worker_id in [0, 1, 2]:
            req = BatchEvaluationRequest(
                request_id=worker_id,
                states=np.random.rand(2, 3, 15, 15).astype(np.float32),
                legal_masks=None,
                temperatures=np.ones(2),
                worker_id=worker_id,
                timestamp=time.time(),
                individual_request_ids=[0, 1]
            )
            
            # Mock response
            gpu_queue = Mock()
            resp_queue = Mock()
            resp_queue.get.return_value = BatchEvaluationResponse(
                request_id=999,
                policies=np.random.rand(6, 225).astype(np.float32),
                values=np.random.rand(6).astype(np.float32),
                worker_id=-1,
                individual_request_ids=list(range(6))
            )
            
            requests.append((req, threading.Event(), {}, gpu_queue, resp_queue))
            
        # Process batch
        coordinator._process_request_batch(requests)
        
        # Check statistics
        stats = coordinator.get_statistics()
        assert stats['cross_worker_batches'] == 1
        assert stats['requests_processed'] == 3
        assert stats['batches_created'] == 1


class TestGlobalCoordinator:
    """Test global coordinator management"""
    
    def test_get_global_coordinator(self):
        """Test getting global coordinator instance"""
        cleanup_global_coordinator()  # Start clean
        
        try:
            coord1 = get_global_batching_coordinator()
            coord2 = get_global_batching_coordinator()
            
            # Should be same instance
            assert coord1 is coord2
            
            # Should be started
            assert coord1.batch_thread is not None
            assert coord1.batch_thread.is_alive()
            
        finally:
            cleanup_global_coordinator()
            
    def test_process_isolation(self):
        """Test coordinator is per-process"""
        cleanup_global_coordinator()
        
        try:
            # Get coordinator in main process
            main_coord = get_global_batching_coordinator()
            main_id = id(main_coord)
            
            # Check in subprocess
            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=_get_subprocess_coordinator_id,
                args=(result_queue,)
            )
            process.start()
            process.join()
            
            subprocess_id = result_queue.get()
            
            # Should be different instances
            assert main_id != subprocess_id
            
        finally:
            cleanup_global_coordinator()
            
    def test_cleanup_global_coordinator(self):
        """Test cleaning up global coordinator"""
        # Create coordinator
        coord = get_global_batching_coordinator()
        assert coord.batch_thread is not None
        
        # Cleanup
        cleanup_global_coordinator()
        
        # Should be stopped
        assert coord.batch_thread is None
        
        # Getting again should create new instance
        coord2 = get_global_batching_coordinator()
        assert coord2 is not coord
        
        cleanup_global_coordinator()


class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.slow
    def test_concurrent_requests_stress(self):
        """Test handling many concurrent requests"""
        coordinator = RequestBatchingCoordinator(
            max_batch_size=32,
            batch_timeout_ms=20.0
        )
        coordinator.start()
        
        try:
            request_queue = queue.Queue()
            response_queue = queue.Queue()
            
            # GPU service that processes batches
            stop_gpu = threading.Event()
            processed_count = 0
            
            def gpu_service():
                nonlocal processed_count
                while not stop_gpu.is_set():
                    try:
                        req = request_queue.get(timeout=0.1)
                        processed_count += 1
                        
                        # Simulate processing time
                        time.sleep(0.01)
                        
                        resp = BatchEvaluationResponse(
                            request_id=req.request_id,
                            policies=np.random.rand(req.states.shape[0], 225).astype(np.float32),
                            values=np.random.rand(req.states.shape[0]).astype(np.float32),
                            worker_id=req.worker_id,
                            individual_request_ids=req.individual_request_ids
                        )
                        response_queue.put(resp)
                    except queue.Empty:
                        continue
                        
            gpu_thread = threading.Thread(target=gpu_service)
            gpu_thread.start()
            
            # Send many requests concurrently
            num_workers = 10
            requests_per_worker = 20
            results = []
            threads = []
            
            def worker_thread(worker_id):
                for i in range(requests_per_worker):
                    batch_size = np.random.randint(1, 8)
                    states = np.random.rand(batch_size, 3, 15, 15).astype(np.float32)
                    
                    policies, values = coordinator.coordinate_evaluation_batch(
                        states, worker_id=worker_id,
                        gpu_service_request_queue=request_queue,
                        response_queue=response_queue
                    )
                    
                    results.append((worker_id, i, policies.shape, values.shape))
                    
            # Start workers
            start_time = time.time()
            for worker_id in range(num_workers):
                t = threading.Thread(target=worker_thread, args=(worker_id,))
                threads.append(t)
                t.start()
                
            # Wait for completion
            for t in threads:
                t.join()
                
            elapsed = time.time() - start_time
            
            # Stop GPU service
            stop_gpu.set()
            gpu_thread.join()
            
            # Verify results
            assert len(results) == num_workers * requests_per_worker
            
            # Check batching efficiency
            stats = coordinator.get_statistics()
            assert stats['requests_processed'] == num_workers * requests_per_worker
            assert stats['batches_created'] < stats['requests_processed']  # Some batching
            
            # Performance check
            requests_per_second = (num_workers * requests_per_worker) / elapsed
            assert requests_per_second > 100  # Should handle >100 req/s
            
        finally:
            coordinator.stop()
            
    def test_mixed_batch_sizes(self):
        """Test handling mixed batch sizes efficiently"""
        coordinator = RequestBatchingCoordinator(
            max_batch_size=32,
            batch_timeout_ms=15.0
        )
        coordinator.start()
        
        try:
            request_queue = queue.Queue()
            response_queue = queue.Queue()
            
            # Mock GPU service
            def gpu_service():
                while True:
                    try:
                        req = request_queue.get(timeout=0.5)
                        resp = BatchEvaluationResponse(
                            request_id=req.request_id,
                            policies=np.random.rand(req.states.shape[0], 225).astype(np.float32),
                            values=np.random.rand(req.states.shape[0]).astype(np.float32),
                            worker_id=req.worker_id,
                            individual_request_ids=req.individual_request_ids
                        )
                        response_queue.put(resp)
                    except queue.Empty:
                        break
                        
            gpu_thread = threading.Thread(target=gpu_service)
            gpu_thread.start()
            
            # Send various batch sizes
            batch_sizes = [1, 2, 4, 8, 16, 32, 64]
            results = []
            
            for size in batch_sizes:
                states = np.random.rand(size, 3, 15, 15).astype(np.float32)
                policies, values = coordinator.coordinate_evaluation_batch(
                    states, worker_id=0,
                    gpu_service_request_queue=request_queue,
                    response_queue=response_queue
                )
                results.append((size, policies.shape[0], values.shape[0]))
                
            gpu_thread.join()
            
            # Verify all processed correctly
            for size, p_size, v_size in results:
                assert p_size == size
                assert v_size == size
                
            # Check statistics
            stats = coordinator.get_statistics()
            # Large batches should use direct processing
            assert stats['fallback_to_direct'] >= 2  # At least 32 and 64
            
        finally:
            coordinator.stop()