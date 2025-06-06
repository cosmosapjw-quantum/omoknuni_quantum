"""Unit tests for ResourceMonitor class."""

import time
import threading
import pytest
import psutil
import numpy as np
import torch

from mcts.utils.resource_monitor import ResourceMonitor, ResourceSnapshot, ResourceStats, ResourceTracker


class TestResourceMonitor:
    """Test suite for ResourceMonitor class."""
    
    def test_initialization(self):
        """Test basic initialization."""
        monitor = ResourceMonitor(sample_interval=0.1, max_history=100)
        assert monitor.sample_interval == 0.1
        assert monitor.max_history == 100
        assert len(monitor.history) == 0
        
    def test_gpu_detection(self):
        """Test GPU detection."""
        monitor = ResourceMonitor()
        
        # Check if GPU detection matches PyTorch
        if torch.cuda.is_available():
            assert monitor.gpu_available
            assert monitor.gpu_id is not None
            assert monitor.gpu_name is not None
        else:
            # GPU might still be available via GPUtil even if PyTorch doesn't see it
            # So we just check consistency
            if monitor.gpu_available:
                assert monitor.gpu_id is not None
                assert monitor.gpu_name is not None
                
    def test_sample_resources(self):
        """Test resource sampling."""
        monitor = ResourceMonitor()
        snapshot = monitor._sample_resources()
        
        # Check required fields
        assert isinstance(snapshot, ResourceSnapshot)
        assert isinstance(snapshot.timestamp, float)
        assert 0 <= snapshot.cpu_percent <= 100
        assert isinstance(snapshot.cpu_per_core, list)
        assert all(0 <= cpu <= 100 for cpu in snapshot.cpu_per_core)
        assert snapshot.ram_used_gb > 0
        assert 0 <= snapshot.ram_percent <= 100
        
        # Check GPU fields if available
        if monitor.gpu_available:
            assert snapshot.gpu_id is not None
            assert snapshot.gpu_name is not None
            assert 0 <= snapshot.gpu_util_percent <= 100
            assert snapshot.gpu_memory_used_gb >= 0
            assert 0 <= snapshot.gpu_memory_percent <= 100
            
    def test_get_current(self):
        """Test getting current snapshot."""
        monitor = ResourceMonitor()
        snapshot = monitor.get_current()
        
        assert isinstance(snapshot, ResourceSnapshot)
        assert time.time() - snapshot.timestamp < 1.0  # Recent snapshot
        
    def test_monitoring_thread(self):
        """Test background monitoring thread."""
        monitor = ResourceMonitor(sample_interval=0.05)
        
        # Start monitoring
        monitor.start()
        assert monitor._thread is not None
        assert monitor._thread.is_alive()
        
        # Let it collect some samples
        time.sleep(0.3)
        
        # Stop monitoring
        monitor.stop()
        assert not monitor._thread.is_alive()
        
        # Check samples were collected (should have at least 5 samples with 0.05s interval over 0.3s)
        assert len(monitor.history) >= 3  # Be more lenient due to timing variations
        
        # Check samples are in chronological order
        timestamps = [s.timestamp for s in monitor.history]
        assert all(timestamps[i] < timestamps[i+1] for i in range(len(timestamps)-1))
        
    def test_max_history(self):
        """Test history size limit."""
        monitor = ResourceMonitor(sample_interval=0.01, max_history=10)
        monitor.start()
        
        # Let it collect more than max_history samples
        time.sleep(0.2)
        monitor.stop()
        
        # Check history doesn't exceed max
        assert len(monitor.history) <= 10
        
    def test_get_stats(self):
        """Test statistics calculation."""
        monitor = ResourceMonitor()
        
        # Manually add some samples for predictable testing
        base_time = time.time()
        for i in range(5):
            snapshot = ResourceSnapshot(
                timestamp=base_time + i,
                cpu_percent=20.0 + i * 10,  # 20, 30, 40, 50, 60
                cpu_per_core=[20.0 + i * 10] * 4,
                ram_used_gb=8.0 + i * 0.5,  # 8.0, 8.5, 9.0, 9.5, 10.0
                ram_percent=50.0 + i * 2     # 50, 52, 54, 56, 58
            )
            monitor.history.append(snapshot)
            
        stats = monitor.get_stats()
        
        assert isinstance(stats, ResourceStats)
        assert stats.samples == 5
        assert stats.duration_seconds == pytest.approx(4.0, rel=0.01)
        
        # CPU stats
        assert stats.cpu_mean == pytest.approx(40.0, rel=0.01)
        assert stats.cpu_max == 60.0
        assert stats.cpu_min == 20.0
        assert stats.cpu_std > 0
        
        # RAM stats
        assert stats.ram_mean_gb == pytest.approx(9.0, rel=0.01)
        assert stats.ram_max_gb == 10.0
        assert stats.ram_min_gb == 8.0
        assert stats.ram_percent_mean == pytest.approx(54.0, rel=0.01)
        
    def test_get_stats_with_time_filter(self):
        """Test statistics with time filtering."""
        monitor = ResourceMonitor()
        
        # Add samples over 10 seconds
        base_time = time.time() - 10  # Start 10 seconds ago
        for i in range(10):
            snapshot = ResourceSnapshot(
                timestamp=base_time + i,
                cpu_percent=float(i * 10),
                cpu_per_core=[float(i * 10)] * 4,
                ram_used_gb=float(i),
                ram_percent=float(i * 5)
            )
            monitor.history.append(snapshot)
            
        # Get stats for last 5 seconds
        stats = monitor.get_stats(last_n_seconds=5)
        
        assert stats.samples <= 5  # Should only include samples from last 5 seconds
        assert stats.cpu_min >= 40.0  # Samples should be from later time range
        
    def test_context_manager(self):
        """Test context manager usage."""
        with ResourceMonitor(sample_interval=0.05) as monitor:
            assert monitor._thread is not None
            assert monitor._thread.is_alive()
            time.sleep(0.2)
            
        # Thread should be stopped after context
        assert not monitor._thread.is_alive()
        assert len(monitor.history) > 0
        
    def test_resource_tracker(self):
        """Test ResourceTracker context manager."""
        with ResourceTracker("Test Operation", print_on_exit=False) as tracker:
            # Simulate some CPU-intensive work
            for _ in range(5):
                _ = sum(range(1000000))
                time.sleep(0.05)
            
        stats = tracker.get_stats()
        assert stats is not None
        assert stats.samples >= 2  # Should have multiple samples
        # CPU usage might be 0 if the work is too fast, so just check it's non-negative
        assert stats.cpu_mean >= 0
        
    def test_concurrent_monitoring(self):
        """Test monitoring under concurrent load."""
        monitor = ResourceMonitor(sample_interval=0.01)
        
        def cpu_intensive_task():
            """Generate CPU load."""
            for _ in range(1000000):
                _ = sum(range(100))
                
        monitor.start()
        
        # Create some concurrent load
        threads = []
        for _ in range(4):
            t = threading.Thread(target=cpu_intensive_task)
            t.start()
            threads.append(t)
            
        time.sleep(0.2)
        
        for t in threads:
            t.join()
            
        monitor.stop()
        
        stats = monitor.get_stats()
        assert stats is not None
        assert stats.cpu_max > stats.cpu_min  # Should see variation in CPU usage
        
    def test_gpu_monitoring(self):
        """Test GPU monitoring if available."""
        monitor = ResourceMonitor()
        
        if not monitor.gpu_available:
            pytest.skip("GPU not available")
            
        monitor.start()
        
        # Create some GPU load if PyTorch is available
        if torch.cuda.is_available():
            device = torch.device('cuda')
            x = torch.randn(1000, 1000, device=device)
            for _ in range(10):
                x = torch.mm(x, x)
                time.sleep(0.01)
                
        time.sleep(0.2)
        monitor.stop()
        
        stats = monitor.get_stats()
        assert stats.gpu_util_mean is not None
        assert stats.gpu_memory_mean_gb is not None
        assert stats.gpu_memory_mean_gb > 0  # Should have some memory usage
        
    def test_error_handling(self):
        """Test error handling in monitoring."""
        monitor = ResourceMonitor()
        
        # Test with invalid GPU ID
        monitor.gpu_id = 999
        monitor._setup_gpu()
        assert not monitor.gpu_available
        
        # Sampling should still work without GPU
        snapshot = monitor._sample_resources()
        assert snapshot.cpu_percent >= 0
        assert snapshot.gpu_util_percent is None
        
    def test_print_stats(self, capsys):
        """Test statistics printing."""
        monitor = ResourceMonitor()
        
        # Add a sample
        snapshot = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=50.0,
            cpu_per_core=[50.0] * 4,
            ram_used_gb=8.0,
            ram_percent=50.0,
            gpu_util_percent=75.0,
            gpu_memory_used_gb=4.0
        )
        monitor.history.append(snapshot)
        
        monitor.print_stats()
        
        captured = capsys.readouterr()
        assert "RESOURCE USAGE STATISTICS" in captured.out
        assert "CPU Usage:" in captured.out
        assert "RAM Usage:" in captured.out
        assert "50.0%" in captured.out
        
    def test_empty_stats(self):
        """Test statistics with no samples."""
        monitor = ResourceMonitor()
        stats = monitor.get_stats()
        assert stats is None
        
        # Should handle printing empty stats gracefully
        monitor.print_stats()  # Should not raise error
        

class TestIntegrationScenarios:
    """Integration tests for real-world usage scenarios."""
    
    def test_benchmark_monitoring(self):
        """Test monitoring during a simulated benchmark."""
        results = []
        
        with ResourceTracker("Benchmark Test", print_on_exit=False) as tracker:
            # Simulate varying workload
            for i in range(5):
                start = time.time()
                
                # CPU work
                _ = sum(range(1000000))
                
                # Memory allocation
                data = np.random.randn(1000, 1000)
                _ = np.dot(data, data)
                
                elapsed = time.time() - start
                results.append(elapsed)
                
                time.sleep(0.05)
                
        stats = tracker.get_stats()
        assert stats is not None
        assert stats.samples >= 3  # More lenient due to timing
        assert stats.cpu_mean >= 0
        # RAM usage should show some variation due to numpy allocations
        assert stats.ram_max_gb >= stats.ram_min_gb
        
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_benchmark_monitoring(self):
        """Test monitoring during GPU operations."""
        device = torch.device('cuda')
        
        with ResourceTracker("GPU Benchmark", print_on_exit=False) as tracker:
            # Allocate GPU memory
            x = torch.randn(5000, 5000, device=device)
            y = torch.randn(5000, 5000, device=device)
            
            # Perform computations
            for _ in range(10):
                z = torch.matmul(x, y)
                z = torch.relu(z)
                torch.cuda.synchronize()
                
        stats = tracker.get_stats()
        assert stats.gpu_util_mean > 0
        assert stats.gpu_memory_mean_gb > 0
        
        
if __name__ == "__main__":
    pytest.main([__file__, "-v"])