"""Adaptive Parameter Tuning for Dynamic CPU-GPU Load Balancing

This module provides intelligent parameter tuning that adapts to simulation load patterns
to maintain optimal CPU-GPU throughput balance across different simulation counts.
"""

import time
import threading
import numpy as np
import torch
import logging
from typing import Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import deque
import psutil
from queue import Queue

logger = logging.getLogger(__name__)


@dataclass
class ThroughputMetrics:
    """Real-time throughput metrics"""
    timestamp: float
    cpu_utilization: float
    gpu_utilization: float
    gpu_memory_utilization: float
    simulations_per_second: float
    avg_batch_size: float
    avg_latency_ms: float
    queue_depth: int
    active_workers: int
    simulation_count: int  # Current simulation count per move


@dataclass 
class AdaptiveParameters:
    """Dynamic optimization parameters"""
    batch_size: int
    batch_timeout_ms: float
    queue_size: int
    coordination_timeout_ms: float
    max_concurrent_workers: int
    wave_size: int
    gpu_batch_timeout_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'batch_size': self.batch_size,
            'batch_timeout_ms': self.batch_timeout_ms,
            'queue_size': self.queue_size,
            'coordination_timeout_ms': self.coordination_timeout_ms,
            'max_concurrent_workers': self.max_concurrent_workers,
            'wave_size': self.wave_size,
            'gpu_batch_timeout_ms': self.gpu_batch_timeout_ms
        }


class AdaptiveParameterTuner:
    """Intelligent parameter tuning for optimal CPU-GPU balance
    
    This system continuously monitors throughput patterns and dynamically adjusts
    batch sizes, timeouts, and coordination parameters to maintain optimal balance
    between CPU and GPU utilization across different simulation loads.
    """
    
    def __init__(self, 
                 target_cpu_utilization: float = 0.85,
                 target_gpu_utilization: float = 0.90,
                 adjustment_interval: float = 3.0,
                 stability_threshold: float = 0.15):
        """Initialize adaptive parameter tuner
        
        Args:
            target_cpu_utilization: Target CPU utilization (0.0-1.0)
            target_gpu_utilization: Target GPU utilization (0.0-1.0) 
            adjustment_interval: How often to adjust parameters (seconds)
            stability_threshold: Minimum change to trigger adjustment
        """
        self.target_cpu_util = target_cpu_utilization
        self.target_gpu_util = target_gpu_utilization
        self.adjustment_interval = adjustment_interval
        self.stability_threshold = stability_threshold
        
        # Metrics tracking
        self.metrics_history = deque(maxlen=200)  # Last 200 measurements
        self.metrics_lock = threading.Lock()
        
        # Parameter adjustment callbacks
        self.parameter_callbacks: Dict[str, Callable] = {}
        
        # Current adaptive parameters per simulation range
        self.parameter_profiles = {
            'low': AdaptiveParameters(  # 100-300 simulations
                batch_size=32,
                batch_timeout_ms=5.0,
                queue_size=500,
                coordination_timeout_ms=10.0,
                max_concurrent_workers=4,
                wave_size=1024,
                gpu_batch_timeout_ms=3.0
            ),
            'medium': AdaptiveParameters(  # 300-600 simulations
                batch_size=64,
                batch_timeout_ms=15.0,
                queue_size=1000,
                coordination_timeout_ms=25.0,
                max_concurrent_workers=6,
                wave_size=2048,
                gpu_batch_timeout_ms=8.0
            ),
            'high': AdaptiveParameters(  # 600+ simulations
                batch_size=96,
                batch_timeout_ms=35.0,
                queue_size=1500,
                coordination_timeout_ms=50.0,
                max_concurrent_workers=8,
                wave_size=3072,
                gpu_batch_timeout_ms=20.0
            )
        }
        
        # Monitoring thread
        self.monitor_thread = None
        self.stop_event = threading.Event()
        
        # GPU monitoring setup
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_device = torch.cuda.current_device()
        
        # Performance tracking
        self.last_adjustment_time = time.time()
        self.adjustment_count = 0
        self.performance_score_history = deque(maxlen=30)
        self.current_simulation_count = 100
        
        logger.debug(f"Adaptive parameter tuner initialized: "
                    f"CPU target={target_cpu_utilization:.1%}, "
                    f"GPU target={target_gpu_utilization:.1%}")
    
    def start_monitoring(self):
        """Start the adaptive monitoring thread"""
        if self.monitor_thread is not None:
            return
        
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.debug("Adaptive parameter tuning monitoring started")
    
    def stop_monitoring(self):
        """Stop the adaptive monitoring thread"""
        if self.monitor_thread is None:
            return
        
        self.stop_event.set()
        self.monitor_thread.join(timeout=2.0)
        self.monitor_thread = None
        logger.debug("Adaptive parameter tuning monitoring stopped")
    
    def register_parameter_callback(self, component: str, callback: Callable[[AdaptiveParameters], None]):
        """Register a callback to receive parameter updates
        
        Args:
            component: Name of component (e.g., 'batch_coordinator', 'gpu_service')
            callback: Function that takes AdaptiveParameters and applies them
        """
        self.parameter_callbacks[component] = callback
        logger.debug(f"Registered parameter callback for {component}")
    
    def record_metrics(self, 
                      simulations_per_second: float,
                      simulation_count: int,
                      avg_batch_size: float = 0.0,
                      avg_latency_ms: float = 0.0,
                      queue_depth: int = 0,
                      active_workers: int = 0):
        """Record current throughput metrics"""
        
        # Get system metrics
        cpu_util = psutil.cpu_percent(interval=None) / 100.0
        
        gpu_util = 0.0
        gpu_mem_util = 0.0
        if self.gpu_available:
            try:
                gpu_util = torch.cuda.utilization(self.gpu_device) / 100.0
                gpu_mem_used = torch.cuda.memory_allocated(self.gpu_device)
                gpu_mem_total = torch.cuda.max_memory_allocated(self.gpu_device)
                if gpu_mem_total > 0:
                    gpu_mem_util = gpu_mem_used / gpu_mem_total
            except:
                pass
        
        metrics = ThroughputMetrics(
            timestamp=time.time(),
            cpu_utilization=cpu_util,
            gpu_utilization=gpu_util,
            gpu_memory_utilization=gpu_mem_util,
            simulations_per_second=simulations_per_second,
            avg_batch_size=avg_batch_size,
            avg_latency_ms=avg_latency_ms,
            queue_depth=queue_depth,
            active_workers=active_workers,
            simulation_count=simulation_count
        )
        
        with self.metrics_lock:
            self.metrics_history.append(metrics)
        
        # Update current simulation count
        self.current_simulation_count = simulation_count
    
    def get_optimal_parameters(self, simulation_count: int) -> AdaptiveParameters:
        """Get optimal parameters for a given simulation count"""
        
        # Select base profile based on simulation count
        if simulation_count <= 300:
            base_params = self.parameter_profiles['low']
            profile_name = 'low'
        elif simulation_count <= 600:
            base_params = self.parameter_profiles['medium']
            profile_name = 'medium'
        else:
            base_params = self.parameter_profiles['high']
            profile_name = 'high'
        
        # Apply adaptive adjustments based on recent metrics
        adjusted_params = self._apply_adaptive_adjustments(base_params, profile_name)
        
        return adjusted_params
    
    def _apply_adaptive_adjustments(self, base_params: AdaptiveParameters, profile_name: str) -> AdaptiveParameters:
        """Apply adaptive adjustments based on recent performance metrics"""
        
        if len(self.metrics_history) < 5:
            return base_params
        
        with self.metrics_lock:
            # Get recent metrics for the same simulation range
            recent_metrics = []
            for m in reversed(list(self.metrics_history)):
                if self._get_profile_for_simulation_count(m.simulation_count) == profile_name:
                    recent_metrics.append(m)
                if len(recent_metrics) >= 10:  # Enough samples
                    break
            
            if len(recent_metrics) < 3:
                return base_params
        
        # Calculate average utilization
        avg_cpu = np.mean([m.cpu_utilization for m in recent_metrics])
        avg_gpu = np.mean([m.gpu_utilization for m in recent_metrics])
        avg_sims_per_sec = np.mean([m.simulations_per_second for m in recent_metrics])
        avg_latency = np.mean([m.avg_latency_ms for m in recent_metrics])
        
        # Create adjusted parameters
        adjusted_params = AdaptiveParameters(
            batch_size=base_params.batch_size,
            batch_timeout_ms=base_params.batch_timeout_ms,
            queue_size=base_params.queue_size,
            coordination_timeout_ms=base_params.coordination_timeout_ms,
            max_concurrent_workers=base_params.max_concurrent_workers,
            wave_size=base_params.wave_size,
            gpu_batch_timeout_ms=base_params.gpu_batch_timeout_ms
        )
        
        # CPU too high, GPU too low -> Increase batch size, reduce timeout
        if avg_cpu > self.target_cpu_util + 0.15 and avg_gpu < self.target_gpu_util - 0.15:
            adjusted_params.batch_size = min(512, int(base_params.batch_size * 1.4))
            adjusted_params.batch_timeout_ms = max(2.0, base_params.batch_timeout_ms * 0.6)
            adjusted_params.gpu_batch_timeout_ms = max(1.0, base_params.gpu_batch_timeout_ms * 0.7)
            pass  # Silent adjustment
        
        # GPU too high, CPU too low -> Decrease batch size, increase timeout  
        elif avg_gpu > self.target_gpu_util + 0.15 and avg_cpu < self.target_cpu_util - 0.15:
            adjusted_params.batch_size = max(8, int(base_params.batch_size * 0.6))
            adjusted_params.batch_timeout_ms = min(100.0, base_params.batch_timeout_ms * 1.5)
            adjusted_params.gpu_batch_timeout_ms = min(50.0, base_params.gpu_batch_timeout_ms * 1.3)
            pass  # Silent adjustment
        
        # Both utilizations too low -> Increase concurrency
        elif avg_cpu < 0.4 and avg_gpu < 0.4:
            adjusted_params.max_concurrent_workers = min(16, base_params.max_concurrent_workers + 1)
            adjusted_params.queue_size = int(base_params.queue_size * 1.3)
            adjusted_params.batch_timeout_ms = max(5.0, base_params.batch_timeout_ms * 0.8)
            pass  # Silent adjustment
        
        # Both utilizations too high -> Reduce pressure
        elif avg_cpu > 0.95 and avg_gpu > 0.95:
            adjusted_params.max_concurrent_workers = max(2, base_params.max_concurrent_workers - 1)
            adjusted_params.batch_timeout_ms = min(200.0, base_params.batch_timeout_ms * 1.2)
            pass  # Silent adjustment
        
        # High latency -> Reduce batch sizes and timeouts
        if avg_latency > 500.0:  # 500ms+ latency is too high
            adjusted_params.batch_size = max(16, int(adjusted_params.batch_size * 0.8))
            adjusted_params.batch_timeout_ms = max(2.0, adjusted_params.batch_timeout_ms * 0.7)
            adjusted_params.gpu_batch_timeout_ms = max(1.0, adjusted_params.gpu_batch_timeout_ms * 0.8)
            pass  # Silent adjustment
        
        return adjusted_params
    
    def _get_profile_for_simulation_count(self, simulation_count: int) -> str:
        """Get profile name for simulation count"""
        if simulation_count <= 300:
            return 'low'
        elif simulation_count <= 600:
            return 'medium'
        else:
            return 'high'
    
    def _monitoring_loop(self):
        """Main monitoring loop for adaptive adjustments"""
        logger.debug("Adaptive parameter tuning loop started")
        
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # Check if it's time for adjustment
                if current_time - self.last_adjustment_time >= self.adjustment_interval:
                    self._perform_adjustment_check()
                    self.last_adjustment_time = current_time
                
                # Sleep briefly
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in adaptive parameter tuning loop: {e}")
                time.sleep(2.0)  # Longer sleep on error
        
        logger.debug("Adaptive parameter tuning loop stopped")
    
    def _perform_adjustment_check(self):
        """Check if parameters need adjustment and notify callbacks"""
        
        if len(self.metrics_history) < 10:
            return  # Need more data
        
        # Get optimal parameters for current simulation count
        optimal_params = self.get_optimal_parameters(self.current_simulation_count)
        
        # Notify all registered callbacks
        for component_name, callback in self.parameter_callbacks.items():
            try:
                callback(optimal_params)
                logger.debug(f"Updated parameters for {component_name}")
            except Exception as e:
                logger.error(f"Failed to update parameters for {component_name}: {e}")
        
        self.adjustment_count += 1
        
        # Calculate and track performance score
        with self.metrics_lock:
            if self.metrics_history:
                recent_metrics = list(self.metrics_history)[-5:]
                cpu_scores = [1.0 - abs(m.cpu_utilization - self.target_cpu_util) for m in recent_metrics]
                gpu_scores = [1.0 - abs(m.gpu_utilization - self.target_gpu_util) for m in recent_metrics]
                avg_score = (np.mean(cpu_scores) + np.mean(gpu_scores)) / 2.0
                self.performance_score_history.append(avg_score)
    
    def force_parameter_update(self, simulation_count: int):
        """Force an immediate parameter update for a specific simulation count"""
        optimal_params = self.get_optimal_parameters(simulation_count)
        
        for component_name, callback in self.parameter_callbacks.items():
            try:
                callback(optimal_params)
                logger.debug(f"Force updated parameters for {component_name} (sim_count={simulation_count})")
            except Exception as e:
                logger.error(f"Failed to force update parameters for {component_name}: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report"""
        
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        with self.metrics_lock:
            recent_metrics = list(self.metrics_history)[-20:]  # Last 20 measurements
        
        # Group by simulation count ranges
        low_sim_metrics = [m for m in recent_metrics if m.simulation_count <= 300]
        med_sim_metrics = [m for m in recent_metrics if 300 < m.simulation_count <= 600]
        high_sim_metrics = [m for m in recent_metrics if m.simulation_count > 600]
        
        report = {
            "timestamp": time.time(),
            "current_simulation_count": self.current_simulation_count,
            "adjustment_count": self.adjustment_count,
            "performance_score": np.mean(list(self.performance_score_history)) if self.performance_score_history else 0.0,
            "profiles": {}
        }
        
        # Add profile-specific metrics
        for profile_name, metrics_list in [('low', low_sim_metrics), ('medium', med_sim_metrics), ('high', high_sim_metrics)]:
            if metrics_list:
                report["profiles"][profile_name] = {
                    "sample_count": len(metrics_list),
                    "avg_cpu_utilization": np.mean([m.cpu_utilization for m in metrics_list]),
                    "avg_gpu_utilization": np.mean([m.gpu_utilization for m in metrics_list]),
                    "avg_simulations_per_second": np.mean([m.simulations_per_second for m in metrics_list]),
                    "avg_latency_ms": np.mean([m.avg_latency_ms for m in metrics_list]),
                    "current_parameters": self.parameter_profiles[profile_name].to_dict()
                }
        
        return report
    
    def get_current_parameters(self) -> AdaptiveParameters:
        """Get current optimal parameters based on latest simulation count"""
        return self.get_optimal_parameters(self.current_simulation_count)


# Global adaptive parameter tuner instance (process-aware)
_global_tuner: Optional[AdaptiveParameterTuner] = None
_tuner_lock = threading.Lock()
_tuner_process_id: Optional[int] = None
_has_logged_activation = False  # Track if we've already logged activation


def get_global_parameter_tuner() -> AdaptiveParameterTuner:
    """Get or create the global adaptive parameter tuner (process-aware singleton)"""
    global _global_tuner, _tuner_process_id, _has_logged_activation
    import os
    import tempfile
    
    current_pid = os.getpid()
    
    with _tuner_lock:
        # Check if we need a new instance for this process
        if _global_tuner is None or _tuner_process_id != current_pid:
            if _global_tuner is not None:
                # Clean up old instance from different process
                try:
                    _global_tuner.stop_monitoring()
                except:
                    pass
            
            _global_tuner = AdaptiveParameterTuner()
            _global_tuner.start_monitoring()
            _tuner_process_id = current_pid
            
            # Use file-based flag to ensure only one log message across all processes
            flag_file = os.path.join(tempfile.gettempdir(), '.adaptive_tuner_logged')
            if not os.path.exists(flag_file):
                try:
                    # Create flag file atomically
                    with open(flag_file, 'x') as f:
                        f.write(str(current_pid))
                    logger.debug("Adaptive parameter tuning enabled for dynamic CPU-GPU optimization")
                except FileExistsError:
                    # Another process already logged
                    pass
        
        return _global_tuner


def cleanup_global_parameter_tuner():
    """Clean up the global adaptive parameter tuner"""
    global _global_tuner, _tuner_process_id
    
    with _tuner_lock:
        if _global_tuner is not None:
            _global_tuner.stop_monitoring()
            _global_tuner = None
            _tuner_process_id = None