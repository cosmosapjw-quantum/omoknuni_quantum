"""Training Pipeline Profiler - Comprehensive timing instrumentation

This module provides detailed profiling of the entire training pipeline to identify bottlenecks.
"""

import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from contextlib import contextmanager
import torch

logger = logging.getLogger(__name__)


@dataclass
class TimingStats:
    """Statistics for a timed operation"""
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    times: List[float] = field(default_factory=list)
    
    def add(self, duration: float):
        """Add a timing measurement"""
        self.count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.times.append(duration)
    
    @property
    def avg_time(self) -> float:
        """Average time per operation"""
        return self.total_time / self.count if self.count > 0 else 0.0
    
    @property
    def std_time(self) -> float:
        """Standard deviation of times"""
        if self.count < 2:
            return 0.0
        return np.std(self.times)
    
    def percentile(self, p: float) -> float:
        """Get percentile of times"""
        if not self.times:
            return 0.0
        return np.percentile(self.times, p)


class TrainingProfiler:
    """Comprehensive profiler for AlphaZero training pipeline"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.stats: Dict[str, TimingStats] = defaultdict(TimingStats)
        self.phase_stack: List[str] = []
        self.start_times: Dict[str, float] = {}
        
        # GPU memory tracking
        self.gpu_memory_stats: Dict[str, List[float]] = defaultdict(list)
        self.track_gpu_memory = torch.cuda.is_available()
        
    @contextmanager
    def timer(self, phase_name: str, log_immediate: bool = False):
        """Context manager for timing a phase"""
        if not self.enabled:
            yield
            return
            
        # Track nested phases
        if self.phase_stack:
            full_name = f"{'/'.join(self.phase_stack)}/{phase_name}"
        else:
            full_name = phase_name
            
        self.phase_stack.append(phase_name)
        
        # Track GPU memory before
        if self.track_gpu_memory:
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated() / 1024**3  # GB
        
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.stats[full_name].add(duration)
            
            # Track GPU memory after
            if self.track_gpu_memory:
                torch.cuda.synchronize()
                mem_after = torch.cuda.memory_allocated() / 1024**3  # GB
                mem_delta = mem_after - mem_before
                self.gpu_memory_stats[full_name].append(mem_delta)
            
            self.phase_stack.pop()
            
            if log_immediate and duration > 1.0:  # Log slow operations
                logger.info(f"[PROFILER] {full_name}: {duration:.2f}s")
    
    def start_phase(self, phase_name: str):
        """Start timing a phase (for non-context manager usage)"""
        if not self.enabled:
            return
        self.start_times[phase_name] = time.perf_counter()
    
    def end_phase(self, phase_name: str):
        """End timing a phase"""
        if not self.enabled or phase_name not in self.start_times:
            return
        
        duration = time.perf_counter() - self.start_times[phase_name]
        self.stats[phase_name].add(duration)
        del self.start_times[phase_name]
    
    def log_summary(self, top_n: int = 20, min_time: float = 0.01):
        """Log profiling summary"""
        if not self.enabled or not self.stats:
            return
        
        logger.info("=" * 80)
        logger.info("TRAINING PIPELINE PROFILING SUMMARY")
        logger.info("=" * 80)
        
        # Sort by total time
        sorted_stats = sorted(
            [(name, stats) for name, stats in self.stats.items() if stats.total_time >= min_time],
            key=lambda x: x[1].total_time,
            reverse=True
        )
        
        # Calculate total time
        total_time = sum(stats.total_time for _, stats in sorted_stats)
        
        # Log top operations
        logger.info(f"{'Operation':<40} {'Count':<8} {'Total':<10} {'Avg':<10} {'Min':<10} {'Max':<10} {'%':<6}")
        logger.info("-" * 94)
        
        for i, (name, stats) in enumerate(sorted_stats[:top_n]):
            percent = 100 * stats.total_time / total_time if total_time > 0 else 0
            logger.info(
                f"{name:<40} {stats.count:<8} "
                f"{stats.total_time:<10.2f} {stats.avg_time:<10.3f} "
                f"{stats.min_time:<10.3f} {stats.max_time:<10.3f} "
                f"{percent:<6.1f}"
            )
        
        # Log percentiles for key operations
        logger.info("\nPERCENTILES FOR KEY OPERATIONS:")
        logger.info("-" * 80)
        
        key_ops = ['self_play', 'self_play/game', 'training/epoch', 'arena/game']
        for op in key_ops:
            if op in self.stats and self.stats[op].count > 10:
                stats = self.stats[op]
                logger.info(
                    f"{op}: p50={stats.percentile(50):.3f}s, "
                    f"p90={stats.percentile(90):.3f}s, "
                    f"p99={stats.percentile(99):.3f}s"
                )
        
        # Log GPU memory usage if available
        if self.track_gpu_memory and self.gpu_memory_stats:
            logger.info("\nGPU MEMORY USAGE (GB):")
            logger.info("-" * 80)
            
            for name, deltas in sorted(self.gpu_memory_stats.items(), 
                                      key=lambda x: max(x[1]) if x[1] else 0, 
                                      reverse=True)[:10]:
                if deltas:
                    max_delta = max(deltas)
                    avg_delta = np.mean(deltas)
                    if abs(max_delta) > 0.01:  # Only show significant changes
                        logger.info(f"{name}: max={max_delta:+.2f}GB, avg={avg_delta:+.3f}GB")
        
        logger.info("=" * 80)
    
    def get_stats_dict(self) -> Dict[str, Dict[str, float]]:
        """Get statistics as a dictionary"""
        return {
            name: {
                'count': stats.count,
                'total_time': stats.total_time,
                'avg_time': stats.avg_time,
                'min_time': stats.min_time,
                'max_time': stats.max_time,
                'std_time': stats.std_time,
                'p50': stats.percentile(50),
                'p90': stats.percentile(90),
                'p99': stats.percentile(99)
            }
            for name, stats in self.stats.items()
        }
    
    def reset(self):
        """Reset all statistics"""
        self.stats.clear()
        self.phase_stack.clear()
        self.start_times.clear()
        self.gpu_memory_stats.clear()


# Global profiler instance
_global_profiler = TrainingProfiler(enabled=False)


def get_profiler() -> TrainingProfiler:
    """Get the global profiler instance"""
    return _global_profiler


def enable_profiling():
    """Enable global profiling"""
    _global_profiler.enabled = True
    logger.info("Training profiling enabled")


def disable_profiling():
    """Disable global profiling"""
    _global_profiler.enabled = False


@contextmanager
def profile_phase(phase_name: str, log_immediate: bool = False):
    """Convenience function for profiling a phase"""
    with _global_profiler.timer(phase_name, log_immediate):
        yield


def log_profiling_summary(top_n: int = 20, min_time: float = 0.01):
    """Log the profiling summary"""
    _global_profiler.log_summary(top_n, min_time)


def reset_profiler():
    """Reset the profiler statistics"""
    _global_profiler.reset()