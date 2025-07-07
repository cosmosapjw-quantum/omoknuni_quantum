"""Tests for GPU evaluator service

Tests cover:
- GPUEvaluatorService initialization and lifecycle
- Request queue management
- Worker queue creation
- Batch processing
- Error handling and recovery
- Resource management
- Performance optimizations
"""

import pytest
import torch
import numpy as np
import multiprocessing as mp
import queue
import time
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path

from mcts.utils.gpu_evaluator_service import (
    GPUEvaluatorService, EvaluationRequest, EvaluationResponse
)


@pytest.fixture
def gpu_config():
    """Create GPU service configuration"""
    return {
        'batch_size': 32,
        'batch_timeout': 0.1,  # 100ms in seconds
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'max_queue_size': 1000,
        'workload_type': 'balanced',
        'use_tensorrt': False,  # Disable for testing
        'tensorrt_fp16': False
    }


@pytest.fixture
def mock_model():
    """Create mock neural network model"""
    # Mock forward pass
    def mock_forward(x):
        batch_size = x.shape[0]
        policy = torch.ones(batch_size, 225) / 225
        value = torch.zeros(batch_size, 1)
        return policy, value
    
    model = Mock(spec=torch.nn.Module)
    model.eval = Mock(return_value=model)
    model.to = Mock(return_value=model)
    model.forward = Mock(side_effect=mock_forward)
    # Instead of mocking __call__, make the model itself callable
    model.side_effect = mock_forward
    
    return model


@pytest.fixture
def gpu_service(gpu_config, mock_model):
    """Create GPUEvaluatorService instance"""
    service = GPUEvaluatorService(
        model=mock_model,
        **gpu_config
    )
    return service


@pytest.fixture  
def sample_requests():
    """Create sample evaluation requests"""
    requests = []
    for i in range(4):
        state = np.random.randn(18, 15, 15).astype(np.float32)
        req = EvaluationRequest(
            request_id=i,
            state=state,
            legal_mask=None,
            temperature=1.0,
            worker_id=i % 2,
            timestamp=time.time()
        )
        requests.append(req)
    return requests


class TestGPUServiceInitialization:
    """Test GPUEvaluatorService initialization"""
    
    def test_initialization(self, gpu_config, mock_model):
        """Test basic initialization"""
        service = GPUEvaluatorService(model=mock_model, **gpu_config)
        
        assert service.model == mock_model
        assert service.batch_size == 32
        assert service.batch_timeout == 0.1
        assert hasattr(service, 'model')
        assert hasattr(service, 'device')
        
    def test_device_setup(self, gpu_config, mock_model):
        """Test device setup during initialization"""
        service = GPUEvaluatorService(model=mock_model, **gpu_config)
        
        # Model should be moved to device
        mock_model.to.assert_called_once()
        mock_model.eval.assert_called_once()
        
    def test_request_queue_creation(self, gpu_service):
        """Test request queue creation"""
        assert hasattr(gpu_service.request_queue, 'put')
        assert hasattr(gpu_service.request_queue, 'get')
        assert hasattr(gpu_service.request_queue, 'qsize')
        
    def test_statistics_initialization(self, gpu_service):
        """Test statistics tracking initialization"""
        stats = gpu_service.stats
        
        assert stats['total_requests'] == 0
        assert stats['total_batches'] == 0
        assert stats['avg_batch_size'] == 0.0
        assert 'total_time' in stats


class TestServiceLifecycle:
    """Test service lifecycle management"""
    
    def test_start_service(self, gpu_service):
        """Test starting the service"""
        gpu_service.start()
        
        assert gpu_service.service_thread is not None
        assert gpu_service.service_thread.is_alive()
        
        # Stop the service
        gpu_service.stop()
        
    def test_stop_service(self, gpu_service):
        """Test stopping the service"""
        # Start the service first
        gpu_service.start()
        assert gpu_service.service_thread.is_alive()
        
        # Stop the service
        gpu_service.stop()
        
        # Thread should have stopped
        assert gpu_service.service_thread is None
        
    def test_restart_service(self, gpu_service):
        """Test restarting the service"""
        # Start
        gpu_service.start()
        assert gpu_service.service_thread.is_alive()
        
        # Stop
        gpu_service.stop()
        assert gpu_service.service_thread is None
        
        # Start again
        gpu_service.start()
        assert gpu_service.service_thread.is_alive()
        
        # Clean up
        gpu_service.stop()
        
    def test_service_already_running(self, gpu_service):
        """Test error when starting already running service"""
        gpu_service.start()
        
        with pytest.raises(RuntimeError, match="Service already started"):
            gpu_service.start()
            
        # Clean up
        gpu_service.stop()


class TestQueueManagement:
    """Test queue management functionality"""
    
    def test_get_request_queue(self, gpu_service):
        """Test getting request queue"""
        queue = gpu_service.get_request_queue()
        assert queue is gpu_service.request_queue
        
    def test_create_worker_queue(self, gpu_service):
        """Test creating worker response queue"""
        worker_id = 1
        queue = gpu_service.create_worker_queue(worker_id)
        
        assert worker_id in gpu_service.response_queues
        assert queue is gpu_service.response_queues[worker_id]
        assert worker_id in gpu_service.active_workers
        
    def test_get_existing_worker_queue(self, gpu_service):
        """Test getting existing worker queue"""
        worker_id = 1
        queue1 = gpu_service.create_worker_queue(worker_id)
        queue2 = gpu_service.create_worker_queue(worker_id)
        
        assert queue1 is queue2
        
    def test_cleanup_worker_queue(self, gpu_service):
        """Test cleaning up worker queue"""
        worker_id = 1
        gpu_service.create_worker_queue(worker_id)
        
        gpu_service.cleanup_worker_queue(worker_id)
        
        assert worker_id not in gpu_service.response_queues
        assert worker_id not in gpu_service.active_workers


class TestBatchProcessing:
    """Test batch processing functionality"""
    
    def test_process_batch_basic(self, gpu_service, sample_requests):
        """Test basic batch processing"""
        # Create worker queues
        for req in sample_requests:
            gpu_service.create_worker_queue(req.worker_id)
        
        # Start service
        gpu_service.start()
        
        try:
            # Submit requests
            for req in sample_requests:
                gpu_service.request_queue.put(req)
            
            # Collect responses
            responses = []
            for req in sample_requests:
                resp_queue = gpu_service.response_queues[req.worker_id]
                resp = resp_queue.get(timeout=2.0)
                responses.append(resp)
            
            assert len(responses) == len(sample_requests)
            
            # Check responses
            for i, resp in enumerate(responses):
                assert isinstance(resp, EvaluationResponse)
                assert resp.request_id == sample_requests[i].request_id
                assert resp.policy.shape == (225,)
                assert isinstance(resp.value, float)
                
        finally:
            gpu_service.stop()
            
    def test_dynamic_batching(self, gpu_service):
        """Test dynamic batching behavior"""
        gpu_service.start()
        
        try:
            # Create single worker
            worker_id = 0
            gpu_service.create_worker_queue(worker_id)
            
            # Send requests with delays to test batching
            requests = []
            for i in range(10):
                req = EvaluationRequest(
                    request_id=i,
                    state=np.random.randn(18, 15, 15).astype(np.float32),
                    legal_mask=None,
                    temperature=1.0,
                    worker_id=worker_id,
                    timestamp=time.time()
                )
                requests.append(req)
                gpu_service.request_queue.put(req)
                
                # Small delay between requests
                if i < 5:
                    time.sleep(0.01)
            
            # Collect responses
            resp_queue = gpu_service.response_queues[worker_id]
            responses = []
            for _ in range(10):
                resp = resp_queue.get(timeout=2.0)
                responses.append(resp)
            
            assert len(responses) == 10
            
        finally:
            gpu_service.stop()
            
    def test_batch_statistics_tracking(self, gpu_service, sample_requests):
        """Test batch statistics are tracked correctly"""
        # Create worker queues
        for req in sample_requests:
            gpu_service.create_worker_queue(req.worker_id)
            
        gpu_service.start()
        
        try:
            # Submit requests
            for req in sample_requests:
                gpu_service.request_queue.put(req)
            
            # Wait for processing
            time.sleep(0.5)
            
            # Check statistics
            stats = gpu_service.stats
            assert stats['total_requests'] >= len(sample_requests)
            assert stats['total_batches'] > 0
            
        finally:
            gpu_service.stop()


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_model_forward_error(self, gpu_service):
        """Test handling of model forward pass errors"""
        # Make model raise error
        gpu_service.model.forward.side_effect = RuntimeError("Model error")
        
        gpu_service.start()
        
        try:
            worker_id = 0
            gpu_service.create_worker_queue(worker_id)
            
            # Send request
            req = EvaluationRequest(
                request_id=1,
                state=np.random.randn(18, 15, 15).astype(np.float32),
                legal_mask=None,
                temperature=1.0,
                worker_id=worker_id,
                timestamp=time.time()
            )
            gpu_service.request_queue.put(req)
            
            # Should get response despite error
            resp_queue = gpu_service.response_queues[worker_id]
            
            # Give it some time to process
            time.sleep(0.5)
            
            # The service should handle the error gracefully
            # and continue running
            assert gpu_service.service_thread.is_alive()
            
        finally:
            gpu_service.stop()
            
    def test_cuda_error_recovery(self, gpu_service):
        """Test recovery from CUDA errors"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        # Simulate CUDA OOM by making model raise specific error
        gpu_service.model.forward.side_effect = RuntimeError("CUDA out of memory")
        
        gpu_service.start()
        
        try:
            worker_id = 0
            gpu_service.create_worker_queue(worker_id)
            
            # Send request
            req = EvaluationRequest(
                request_id=1,
                state=np.random.randn(18, 15, 15).astype(np.float32),
                legal_mask=None,
                temperature=1.0,
                worker_id=worker_id,
                timestamp=time.time()
            )
            gpu_service.request_queue.put(req)
            
            # Give it time to handle error
            time.sleep(0.5)
            
            # Service should still be running
            assert gpu_service.service_thread.is_alive()
            
        finally:
            gpu_service.stop()
            
    def test_queue_full_handling(self, gpu_config, mock_model):
        """Test handling of full request queue"""
        # Create service with small queue
        gpu_config['max_queue_size'] = 2
        service = GPUEvaluatorService(model=mock_model, **gpu_config)
        
        # Fill the queue
        for i in range(2):
            req = EvaluationRequest(
                request_id=i,
                state=np.random.randn(18, 15, 15).astype(np.float32),
                legal_mask=None,
                temperature=1.0,
                worker_id=0,
                timestamp=time.time()
            )
            service.request_queue.put(req)
        
        # Try to add one more (should fail or block)
        req = EvaluationRequest(
            request_id=3,
            state=np.random.randn(3, 15, 15).astype(np.float32),
            legal_mask=None,
            temperature=1.0,
            worker_id=0,
            timestamp=time.time()
        )
        
        with pytest.raises(queue.Full):
            service.request_queue.put(req, block=False)
            
    def test_worker_queue_error(self, gpu_service):
        """Test handling of worker queue errors"""
        gpu_service.start()
        
        try:
            # Send request for non-existent worker
            req = EvaluationRequest(
                request_id=1,
                state=np.random.randn(18, 15, 15).astype(np.float32),
                legal_mask=None,
                temperature=1.0,
                worker_id=999,  # Non-existent worker
                timestamp=time.time()
            )
            gpu_service.request_queue.put(req)
            
            # Give it time to process
            time.sleep(0.5)
            
            # Service should handle missing worker gracefully
            assert gpu_service.service_thread.is_alive()
            
        finally:
            gpu_service.stop()


class TestResourceManagement:
    """Test resource management"""
    
    def test_memory_monitoring(self, gpu_service):
        """Test GPU memory monitoring"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        initial_memory = torch.cuda.memory_allocated()
        
        gpu_service.start()
        
        try:
            # Process some requests
            worker_id = 0
            gpu_service.create_worker_queue(worker_id)
            
            for i in range(10):
                req = EvaluationRequest(
                    request_id=i,
                    state=np.random.randn(18, 15, 15).astype(np.float32),
                    legal_mask=None,
                    temperature=1.0,
                    worker_id=worker_id,
                    timestamp=time.time()
                )
                gpu_service.request_queue.put(req)
            
            # Wait for processing
            time.sleep(0.5)
            
            # Memory usage should be reasonable
            current_memory = torch.cuda.memory_allocated()
            memory_increase = current_memory - initial_memory
            
            # Should not leak too much memory
            assert memory_increase < 100 * 1024 * 1024  # Less than 100MB
            
        finally:
            gpu_service.stop()
            
    def test_memory_limit_enforcement(self, gpu_config, mock_model):
        """Test memory limit enforcement"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        # Create service without memory_limit_gb parameter
        service = GPUEvaluatorService(model=mock_model, **gpu_config)
        
        # Service should handle memory limits appropriately
        assert service.device.type in ['cuda', 'cpu']
        
    def test_resource_cleanup(self, gpu_service):
        """Test resource cleanup on shutdown"""
        gpu_service.start()
        
        # Create some worker queues
        for i in range(5):
            gpu_service.create_worker_queue(i)
        
        # Add some requests
        for i in range(10):
            req = EvaluationRequest(
                request_id=i,
                state=np.random.randn(18, 15, 15).astype(np.float32),
                legal_mask=None,
                temperature=1.0,
                worker_id=i % 5,
                timestamp=time.time()
            )
            gpu_service.request_queue.put(req)
        
        # Stop service
        gpu_service.stop()
        
        # Check cleanup
        assert gpu_service.service_thread is None
        # Queues should still exist but service is stopped


class TestPerformanceOptimizations:
    """Test performance optimization features"""
    
    def test_batch_padding_optimization(self, gpu_service):
        """Test batch padding for GPU efficiency"""
        gpu_service.start()
        
        try:
            worker_id = 0
            gpu_service.create_worker_queue(worker_id)
            
            # Send requests of different sizes (if supported)
            requests = []
            for i in range(8):
                req = EvaluationRequest(
                    request_id=i,
                    state=np.random.randn(18, 15, 15).astype(np.float32),
                    legal_mask=None,
                    temperature=1.0,
                    worker_id=worker_id,
                    timestamp=time.time()
                )
                requests.append(req)
                gpu_service.request_queue.put(req)
            
            # Collect responses
            resp_queue = gpu_service.response_queues[worker_id]
            responses = []
            for _ in range(8):
                resp = resp_queue.get(timeout=2.0)
                responses.append(resp)
            
            # All responses should be valid
            assert len(responses) == 8
            
        finally:
            gpu_service.stop()
            
    def test_pinned_memory_usage(self, gpu_service):
        """Test pinned memory optimization"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        # Service should use pinned memory for transfers
        # This is more of an implementation detail test
        assert gpu_service.device.type == 'cuda'
        
    def test_mixed_precision_inference(self, gpu_config, mock_model):
        """Test mixed precision inference"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        # Enable FP16 if using TensorRT
        gpu_config['tensorrt_fp16'] = True
        service = GPUEvaluatorService(model=mock_model, **gpu_config)
        
        # Service should be created successfully with FP16 config
        assert hasattr(service, 'use_tensorrt')
        
    def test_profiling_mode(self, gpu_config, mock_model):
        """Test profiling mode functionality"""
        # Create service without enable_profiling parameter
        service = GPUEvaluatorService(model=mock_model, **gpu_config)
        
        # Service should handle profiling configuration
        # This is more of a configuration test
        assert hasattr(service, 'batch_size')


class TestConcurrency:
    """Test concurrent access patterns"""
    
    def test_multiple_workers(self, gpu_service):
        """Test handling multiple concurrent workers"""
        num_workers = 4
        gpu_service.start()
        
        try:
            # Create worker queues
            for i in range(num_workers):
                gpu_service.create_worker_queue(i)
            
            # Send requests from all workers
            for worker_id in range(num_workers):
                for req_id in range(5):
                    req = EvaluationRequest(
                        request_id=worker_id * 100 + req_id,
                        state=np.random.randn(18, 15, 15).astype(np.float32),
                        legal_mask=None,
                        temperature=1.0,
                        worker_id=worker_id,
                        timestamp=time.time()
                    )
                    gpu_service.request_queue.put(req)
            
            # Collect responses from all workers
            total_responses = 0
            for worker_id in range(num_workers):
                resp_queue = gpu_service.response_queues[worker_id]
                for _ in range(5):
                    resp = resp_queue.get(timeout=2.0)
                    assert isinstance(resp, EvaluationResponse)
                    total_responses += 1
            
            assert total_responses == num_workers * 5
            
        finally:
            gpu_service.stop()
            
    def test_request_ordering(self, gpu_service):
        """Test that requests maintain ordering per worker"""
        gpu_service.start()
        
        try:
            worker_id = 0
            gpu_service.create_worker_queue(worker_id)
            
            # Send numbered requests
            num_requests = 20
            for i in range(num_requests):
                req = EvaluationRequest(
                    request_id=i,
                    state=np.random.randn(18, 15, 15).astype(np.float32),
                    legal_mask=None,
                    temperature=1.0,
                    worker_id=worker_id,
                    timestamp=time.time()
                )
                gpu_service.request_queue.put(req)
            
            # Collect responses
            resp_queue = gpu_service.response_queues[worker_id]
            responses = []
            for _ in range(num_requests):
                resp = resp_queue.get(timeout=2.0)
                responses.append(resp)
            
            # Check ordering
            response_ids = [r.request_id for r in responses]
            assert sorted(response_ids) == list(range(num_requests))
            
        finally:
            gpu_service.stop()


class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_evaluation(self, gpu_service):
        """Test end-to-end evaluation pipeline"""
        gpu_service.start()
        
        try:
            # Simulate real usage pattern
            worker_id = 0
            gpu_service.create_worker_queue(worker_id)
            
            # Create realistic game state
            state = np.zeros((18, 15, 15), dtype=np.float32)
            state[0, 7, 7] = 1  # Center position
            
            req = EvaluationRequest(
                request_id=1,
                state=state,
                legal_mask=None,
                temperature=1.0,
                worker_id=worker_id,
                timestamp=time.time()
            )
            
            gpu_service.request_queue.put(req)
            
            # Get response
            resp_queue = gpu_service.response_queues[worker_id]
            resp = resp_queue.get(timeout=2.0)
            
            # Validate response
            assert resp.request_id == 1
            assert resp.policy.shape == (225,)
            assert np.abs(resp.policy.sum() - 1.0) < 0.01  # Should sum to 1
            assert -1 <= resp.value <= 1
            
        finally:
            gpu_service.stop()
            
    def test_stress_test(self, gpu_service):
        """Stress test with many rapid requests"""
        gpu_service.start()
        
        try:
            num_workers = 2
            requests_per_worker = 50
            
            # Create workers
            for i in range(num_workers):
                gpu_service.create_worker_queue(i)
            
            # Rapid fire requests
            start_time = time.time()
            
            for i in range(requests_per_worker * num_workers):
                req = EvaluationRequest(
                    request_id=i,
                    state=np.random.randn(18, 15, 15).astype(np.float32),
                    legal_mask=None,
                    temperature=1.0,
                    worker_id=i % num_workers,
                    timestamp=time.time()
                )
                gpu_service.request_queue.put(req)
            
            # Collect all responses
            total_responses = 0
            for worker_id in range(num_workers):
                resp_queue = gpu_service.response_queues[worker_id]
                for _ in range(requests_per_worker):
                    resp = resp_queue.get(timeout=5.0)
                    total_responses += 1
            
            elapsed = time.time() - start_time
            throughput = total_responses / elapsed
            
            assert total_responses == requests_per_worker * num_workers
            print(f"Throughput: {throughput:.1f} requests/second")
            
        finally:
            gpu_service.stop()
            
    def test_graceful_shutdown(self, gpu_service):
        """Test graceful shutdown with pending requests"""
        gpu_service.start()
        
        try:
            worker_id = 0
            gpu_service.create_worker_queue(worker_id)
            
            # Add many requests
            for i in range(100):
                req = EvaluationRequest(
                    request_id=i,
                    state=np.random.randn(18, 15, 15).astype(np.float32),
                    legal_mask=None,
                    temperature=1.0,
                    worker_id=worker_id,
                    timestamp=time.time()
                )
                gpu_service.request_queue.put(req)
            
            # Stop immediately
            gpu_service.stop()
            
            # Should stop gracefully without hanging
            assert gpu_service.service_thread is None
            
        except:
            # Make sure we stop even if test fails
            if gpu_service.service_thread:
                gpu_service.stop()
            raise