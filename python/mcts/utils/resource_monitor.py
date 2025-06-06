"""Resource monitoring for benchmarking and performance analysis."""

import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import statistics

import psutil
import GPUtil
import torch


@dataclass
class ResourceSnapshot:
    """Single snapshot of system resources."""
    timestamp: float
    cpu_percent: float
    cpu_per_core: List[float]
    ram_used_gb: float
    ram_percent: float
    gpu_id: Optional[int] = None
    gpu_name: Optional[str] = None
    gpu_util_percent: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_temperature: Optional[float] = None
    gpu_power_watts: Optional[float] = None


@dataclass
class ResourceStats:
    """Statistics for resource usage over a period."""
    duration_seconds: float
    samples: int
    
    # CPU stats
    cpu_mean: float
    cpu_max: float
    cpu_min: float
    cpu_std: float
    
    # RAM stats
    ram_mean_gb: float
    ram_max_gb: float
    ram_min_gb: float
    ram_percent_mean: float
    
    # GPU stats (optional)
    gpu_util_mean: Optional[float] = None
    gpu_util_max: Optional[float] = None
    gpu_memory_mean_gb: Optional[float] = None
    gpu_memory_max_gb: Optional[float] = None
    gpu_temperature_mean: Optional[float] = None
    gpu_power_mean_watts: Optional[float] = None


class ResourceMonitor:
    """Monitors CPU, RAM, and GPU resources during benchmarks."""
    
    def __init__(self, 
                 sample_interval: float = 0.1,
                 gpu_id: Optional[int] = None,
                 max_history: int = 10000):
        """Initialize resource monitor.
        
        Args:
            sample_interval: Time between samples in seconds
            gpu_id: GPU device ID to monitor (None for auto-detect)
            max_history: Maximum number of samples to keep in history
        """
        self.sample_interval = sample_interval
        self.max_history = max_history
        self.history: deque[ResourceSnapshot] = deque(maxlen=max_history)
        
        # Threading control
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        # GPU setup
        self.gpu_id = gpu_id
        self.gpu_available = False
        self.gpu_name = None
        self._setup_gpu()
        
    def _setup_gpu(self):
        """Setup GPU monitoring if available."""
        self.gpu_available = False
        self.gpu_name = None
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                if self.gpu_id is None:
                    # Auto-detect: use first GPU or the one PyTorch is using
                    if torch.cuda.is_available():
                        self.gpu_id = torch.cuda.current_device()
                    else:
                        self.gpu_id = 0
                
                if self.gpu_id < len(gpus):
                    self.gpu_available = True
                    self.gpu_name = gpus[self.gpu_id].name
                    print(f"GPU monitoring enabled for: {self.gpu_name} (ID: {self.gpu_id})")
                else:
                    self.gpu_available = False
                    print(f"GPU ID {self.gpu_id} not found")
        except Exception as e:
            self.gpu_available = False
            print(f"GPU monitoring not available: {e}")
            
    def _sample_resources(self) -> ResourceSnapshot:
        """Take a single snapshot of current resources."""
        timestamp = time.time()
        
        # CPU and RAM
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        mem = psutil.virtual_memory()
        ram_used_gb = mem.used / (1024**3)
        ram_percent = mem.percent
        
        snapshot = ResourceSnapshot(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            cpu_per_core=cpu_per_core,
            ram_used_gb=ram_used_gb,
            ram_percent=ram_percent
        )
        
        # GPU if available
        if self.gpu_available:
            try:
                gpus = GPUtil.getGPUs()
                if self.gpu_id < len(gpus):
                    gpu = gpus[self.gpu_id]
                    snapshot.gpu_id = self.gpu_id
                    snapshot.gpu_name = gpu.name
                    snapshot.gpu_util_percent = gpu.load * 100
                    snapshot.gpu_memory_used_gb = gpu.memoryUsed / 1024
                    snapshot.gpu_memory_percent = gpu.memoryUtil * 100
                    snapshot.gpu_temperature = gpu.temperature
                    
                    # Try to get power consumption (may not be available on all GPUs)
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W
                        snapshot.gpu_power_watts = power
                    except:
                        pass
            except Exception as e:
                # GPU monitoring can fail during heavy load
                pass
                
        return snapshot
    
    def _monitor_loop(self):
        """Main monitoring loop running in separate thread."""
        # Initial CPU percent call to initialize
        psutil.cpu_percent(interval=None)
        
        while not self._stop_event.is_set():
            try:
                snapshot = self._sample_resources()
                self.history.append(snapshot)
            except Exception as e:
                print(f"Resource sampling error: {e}")
                
            self._stop_event.wait(self.sample_interval)
    
    def start(self):
        """Start monitoring in background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
            
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        """Stop monitoring."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            
    def get_current(self) -> ResourceSnapshot:
        """Get current resource snapshot."""
        return self._sample_resources()
    
    def get_stats(self, last_n_seconds: Optional[float] = None) -> Optional[ResourceStats]:
        """Calculate statistics over monitoring period.
        
        Args:
            last_n_seconds: Only consider samples from last N seconds (None for all)
            
        Returns:
            ResourceStats or None if no samples
        """
        if not self.history:
            return None
            
        # Filter samples by time if requested
        samples = list(self.history)
        if last_n_seconds is not None:
            current_time = time.time()
            cutoff_time = current_time - last_n_seconds
            samples = [s for s in samples if s.timestamp >= cutoff_time]
            
        if not samples:
            return None
            
        # Calculate duration
        duration = samples[-1].timestamp - samples[0].timestamp
        
        # CPU stats
        cpu_values = [s.cpu_percent for s in samples]
        cpu_mean = statistics.mean(cpu_values)
        cpu_max = max(cpu_values)
        cpu_min = min(cpu_values)
        cpu_std = statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0.0
        
        # RAM stats
        ram_values = [s.ram_used_gb for s in samples]
        ram_percent_values = [s.ram_percent for s in samples]
        
        stats = ResourceStats(
            duration_seconds=duration,
            samples=len(samples),
            cpu_mean=cpu_mean,
            cpu_max=cpu_max,
            cpu_min=cpu_min,
            cpu_std=cpu_std,
            ram_mean_gb=statistics.mean(ram_values),
            ram_max_gb=max(ram_values),
            ram_min_gb=min(ram_values),
            ram_percent_mean=statistics.mean(ram_percent_values)
        )
        
        # GPU stats if available
        gpu_samples = [s for s in samples if s.gpu_util_percent is not None]
        if gpu_samples:
            gpu_util_values = [s.gpu_util_percent for s in gpu_samples]
            gpu_mem_values = [s.gpu_memory_used_gb for s in gpu_samples]
            
            stats.gpu_util_mean = statistics.mean(gpu_util_values)
            stats.gpu_util_max = max(gpu_util_values)
            stats.gpu_memory_mean_gb = statistics.mean(gpu_mem_values)
            stats.gpu_memory_max_gb = max(gpu_mem_values)
            
            temp_samples = [s for s in gpu_samples if s.gpu_temperature is not None]
            if temp_samples:
                stats.gpu_temperature_mean = statistics.mean([s.gpu_temperature for s in temp_samples])
                
            power_samples = [s for s in gpu_samples if s.gpu_power_watts is not None]
            if power_samples:
                stats.gpu_power_mean_watts = statistics.mean([s.gpu_power_watts for s in power_samples])
                
        return stats
    
    def print_stats(self, stats: Optional[ResourceStats] = None):
        """Print formatted resource statistics."""
        if stats is None:
            stats = self.get_stats()
            
        if stats is None:
            print("No resource statistics available")
            return
            
        print(f"\n{'='*60}")
        print("RESOURCE USAGE STATISTICS")
        print(f"{'='*60}")
        print(f"Duration: {stats.duration_seconds:.1f}s ({stats.samples} samples)")
        print(f"\nCPU Usage:")
        print(f"  Mean: {stats.cpu_mean:.1f}%")
        print(f"  Max:  {stats.cpu_max:.1f}%")
        print(f"  Min:  {stats.cpu_min:.1f}%")
        print(f"  Std:  {stats.cpu_std:.1f}%")
        
        print(f"\nRAM Usage:")
        print(f"  Mean: {stats.ram_mean_gb:.2f} GB ({stats.ram_percent_mean:.1f}%)")
        print(f"  Max:  {stats.ram_max_gb:.2f} GB")
        print(f"  Min:  {stats.ram_min_gb:.2f} GB")
        
        if stats.gpu_util_mean is not None:
            print(f"\nGPU Usage:")
            print(f"  Utilization: {stats.gpu_util_mean:.1f}% (max: {stats.gpu_util_max:.1f}%)")
            print(f"  Memory: {stats.gpu_memory_mean_gb:.2f} GB (max: {stats.gpu_memory_max_gb:.2f} GB)")
            if stats.gpu_temperature_mean is not None:
                print(f"  Temperature: {stats.gpu_temperature_mean:.1f}°C")
            if stats.gpu_power_mean_watts is not None:
                print(f"  Power: {stats.gpu_power_mean_watts:.1f}W")
        print(f"{'='*60}\n")
        
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        
        
class ResourceTracker:
    """Simple context manager for tracking resource usage during code execution."""
    
    def __init__(self, name: str = "Operation", print_on_exit: bool = True):
        self.name = name
        self.print_on_exit = print_on_exit
        self.monitor = ResourceMonitor(sample_interval=0.05)
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.monitor.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.stop()
        elapsed = time.time() - self.start_time
        
        if self.print_on_exit:
            print(f"\n{self.name} completed in {elapsed:.2f}s")
            self.monitor.print_stats()
            
    def get_stats(self) -> Optional[ResourceStats]:
        """Get resource statistics."""
        return self.monitor.get_stats()