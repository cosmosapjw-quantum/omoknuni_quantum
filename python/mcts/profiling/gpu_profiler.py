"""
Comprehensive GPU MCTS Profiler
Tracks CPU/GPU time, memory, and bottlenecks at microsecond precision
"""

import torch
import torch.profiler
import time
import contextlib
from collections import defaultdict
from typing import Dict, List, Optional, Any
import numpy as np
import threading
import psutil
import GPUtil


class DetailedGPUProfiler:
    """Ultra-detailed profiler for GPU MCTS operations."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timings = defaultdict(list)
        self.gpu_timings = defaultdict(list)
        self.counts = defaultdict(int)
        self.memory_snapshots = []
        self.cuda_events = {}
        self.cpu_percent_samples = []
        self.gpu_util_samples = []
        
        # Thread for continuous monitoring
        self.monitoring = False
        self.monitor_thread = None
        
        # PyTorch profiler
        self.torch_profiler = None
        
        # CUDA synchronization tracking
        self.sync_points = []
        
    def start_monitoring(self):
        """Start background monitoring of CPU/GPU utilization."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def _monitor_loop(self):
        """Background monitoring loop."""
        process = psutil.Process()
        
        while self.monitoring:
            # CPU usage
            cpu_percent = process.cpu_percent(interval=0.01)
            self.cpu_percent_samples.append((time.time(), cpu_percent))
            
            # GPU usage
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_util = gpus[0].load * 100
                    gpu_memory = gpus[0].memoryUsed
                    self.gpu_util_samples.append((time.time(), gpu_util, gpu_memory))
            except:
                pass
                
            time.sleep(0.01)  # 100Hz sampling
    
    @contextlib.contextmanager
    def profile(self, name: str, sync: bool = False):
        """Profile a code section with CPU timing."""
        if not self.enabled:
            yield
            return
            
        start_time = time.perf_counter()
        
        # Track if this causes GPU sync
        if sync:
            self.sync_points.append((name, start_time))
            
        try:
            yield
        finally:
            end_time = time.perf_counter()
            elapsed = (end_time - start_time) * 1000  # Convert to ms
            self.timings[name].append(elapsed)
            self.counts[name] += 1
            
    @contextlib.contextmanager
    def profile_gpu(self, name: str):
        """Profile GPU operations with CUDA events."""
        if not self.enabled or not torch.cuda.is_available():
            yield
            return
            
        # Create CUDA events for this operation
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        
        try:
            yield
        finally:
            end_event.record()
            
            # Store events for later synchronization
            if name not in self.cuda_events:
                self.cuda_events[name] = []
            self.cuda_events[name].append((start_event, end_event))
            
    def profile_memory(self, name: str):
        """Take a memory snapshot."""
        if not self.enabled or not torch.cuda.is_available():
            return
            
        allocated = torch.cuda.memory_allocated() / 1e9  # GB
        reserved = torch.cuda.memory_reserved() / 1e9   # GB
        
        self.memory_snapshots.append({
            'name': name,
            'time': time.time(),
            'allocated_gb': allocated,
            'reserved_gb': reserved
        })
        
    def synchronize_gpu_timings(self):
        """Synchronize and collect GPU timings."""
        if not torch.cuda.is_available():
            return
            
        torch.cuda.synchronize()
        
        for name, events in self.cuda_events.items():
            for start_event, end_event in events:
                elapsed = start_event.elapsed_time(end_event)  # ms
                self.gpu_timings[name].append(elapsed)
                
        self.cuda_events.clear()
        
    def get_report(self) -> str:
        """Generate comprehensive profiling report."""
        self.synchronize_gpu_timings()
        
        report = ["=" * 80]
        report.append("COMPREHENSIVE GPU MCTS PROFILING REPORT")
        report.append("=" * 80)
        
        # CPU Timings
        report.append("\nCPU TIMINGS (sorted by total time):")
        report.append("-" * 80)
        report.append(f"{'Operation':<40} {'Count':>8} {'Total(ms)':>12} {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}")
        report.append("-" * 80)
        
        cpu_totals = []
        for name, times in self.timings.items():
            if times:
                total = sum(times)
                avg = np.mean(times)
                min_t = min(times)
                max_t = max(times)
                cpu_totals.append((total, name, len(times), avg, min_t, max_t))
                
        for total, name, count, avg, min_t, max_t in sorted(cpu_totals, reverse=True):
            report.append(f"{name:<40} {count:>8} {total:>12.2f} {avg:>10.2f} {min_t:>10.2f} {max_t:>10.2f}")
            
        # GPU Timings
        if self.gpu_timings:
            report.append("\n\nGPU KERNEL TIMINGS:")
            report.append("-" * 80)
            report.append(f"{'Kernel':<40} {'Count':>8} {'Total(ms)':>12} {'Avg(ms)':>10}")
            report.append("-" * 80)
            
            gpu_totals = []
            for name, times in self.gpu_timings.items():
                if times:
                    total = sum(times)
                    avg = np.mean(times)
                    gpu_totals.append((total, name, len(times), avg))
                    
            for total, name, count, avg in sorted(gpu_totals, reverse=True):
                report.append(f"{name:<40} {count:>8} {total:>12.2f} {avg:>10.2f}")
                
        # Synchronization points
        if self.sync_points:
            report.append("\n\nGPU SYNCHRONIZATION POINTS:")
            report.append("-" * 80)
            for name, timestamp in self.sync_points[-10:]:  # Last 10
                report.append(f"{name} at {timestamp:.3f}")
                
        # Memory usage
        if self.memory_snapshots:
            report.append("\n\nMEMORY USAGE:")
            report.append("-" * 80)
            last_snapshot = self.memory_snapshots[-1]
            report.append(f"Allocated: {last_snapshot['allocated_gb']:.2f} GB")
            report.append(f"Reserved:  {last_snapshot['reserved_gb']:.2f} GB")
            
        # CPU/GPU utilization
        if self.cpu_percent_samples:
            cpu_avg = np.mean([x[1] for x in self.cpu_percent_samples[-100:]])
            report.append(f"\nAverage CPU usage: {cpu_avg:.1f}%")
            
        if self.gpu_util_samples:
            gpu_avg = np.mean([x[1] for x in self.gpu_util_samples[-100:]])
            gpu_mem_avg = np.mean([x[2] for x in self.gpu_util_samples[-100:]])
            report.append(f"Average GPU usage: {gpu_avg:.1f}%")
            report.append(f"Average GPU memory: {gpu_mem_avg:.0f} MB")
            
        # Bottleneck analysis
        report.append("\n\nBOTTLENECK ANALYSIS:")
        report.append("-" * 80)
        
        if cpu_totals:
            total_cpu_time = sum(x[0] for x in cpu_totals)
            report.append(f"Total CPU time: {total_cpu_time:.2f} ms")
            
            # Top 5 CPU bottlenecks
            report.append("\nTop 5 CPU bottlenecks:")
            for total, name, _, _, _, _ in cpu_totals[:5]:
                percent = (total / total_cpu_time) * 100
                report.append(f"  {name}: {percent:.1f}% of CPU time")
                
        return "\n".join(report)
    
    def save_profiling_report(self, filename: str):
        """Save profiling report to file."""
        with open(filename, 'w') as f:
            f.write(self.get_report())
        
    def start_torch_profiler(self, tensorboard_dir: Optional[str] = None):
        """Start PyTorch profiler for detailed trace."""
        if not self.enabled:
            return
            
        schedule = torch.profiler.schedule(
            skip_first=10,
            wait=5,
            warmup=5,
            active=10,
            repeat=1
        )
        
        self.torch_profiler = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(tensorboard_dir) if tensorboard_dir else None,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True
        )
        
        self.torch_profiler.start()
        
    def stop_torch_profiler(self):
        """Stop PyTorch profiler."""
        if self.torch_profiler:
            self.torch_profiler.stop()
            
    def step_torch_profiler(self):
        """Step PyTorch profiler."""
        if self.torch_profiler:
            self.torch_profiler.step()


# Global profiler instance
_global_profiler = None


def get_profiler() -> DetailedGPUProfiler:
    """Get global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = DetailedGPUProfiler()
    return _global_profiler


def profile(name: str, sync: bool = False):
    """Decorator for profiling functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            with profiler.profile(name, sync=sync):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def profile_gpu(name: str):
    """Decorator for profiling GPU operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            with profiler.profile_gpu(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class MCTSProfilingMixin:
    """Mixin to add profiling to MCTS classes."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiler = get_profiler()
        self.profiler.start_monitoring()
        
    def print_profiling_report(self):
        """Print profiling report."""
        print(self.profiler.get_report())
        
    def save_profiling_report(self, filename: str):
        """Save profiling report to file."""
        with open(filename, 'w') as f:
            f.write(self.profiler.get_report())