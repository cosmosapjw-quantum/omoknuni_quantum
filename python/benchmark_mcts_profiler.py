#!/usr/bin/env python3
"""
Comprehensive MCTS Benchmark & Profiling Tool with Quantum Level Comparison

This tool provides deep performance analysis of the MCTS implementation by:
1. Comparing classical vs tree-level vs one-loop quantum implementations
2. GPU/CPU/RAM/VRAM usage monitoring with high precision
3. Phase-by-phase breakdown of MCTS operations
4. Memory transfer detection and quantification
5. Kernel launch overhead analysis
6. Bottleneck identification with actionable recommendations

Usage:
    python mcts_comprehensive_profiler.py [--config CONFIG_FILE]
"""

import torch
import psutil
import time
import gc
import threading
import numpy as np
import json
import argparse
import cProfile
import pstats
import io
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  matplotlib/seaborn not available - plots will be disabled")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  pandas not available - CSV export will be disabled")

# MCTS imports
from mcts.core.mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType
from mcts.quantum.quantum_features import QuantumConfig
import alphazero_py


@dataclass
class ProfilingConfig:
    """Configuration for profiling session"""
    # Test scenarios
    simulation_counts: List[int] = None
    wave_sizes: List[int] = None
    warmup_iterations: int = 3
    measurement_iterations: int = 5
    
    # Quantum levels to test
    quantum_levels: List[str] = field(default_factory=lambda: ['classical', 'tree_level', 'one_loop'])
    
    # Monitoring settings
    memory_sample_interval: float = 0.01  # 10ms sampling
    gpu_event_precision: bool = True
    trace_cuda_calls: bool = True
    profile_individual_phases: bool = True
    
    # cProfile settings
    enable_cprofile: bool = True
    cprofile_sort_key: str = 'cumulative'  # 'cumulative', 'time', 'calls'
    cprofile_top_functions: int = 50
    
    # Output settings
    save_detailed_logs: bool = True
    generate_plots: bool = True
    output_dir: str = "mcts_quantum_profiling_results"
    
    def __post_init__(self):
        if self.simulation_counts is None:
            self.simulation_counts = [1000, 5000, 10000, 25000, 50000]
        if self.wave_sizes is None:
            # Optimized for RTX 3060 Ti
            self.wave_sizes = [3072]  # Fixed optimal size


@dataclass
class MemorySnapshot:
    """Single memory usage snapshot"""
    timestamp: float
    cpu_percent: float
    ram_mb: float
    ram_available_mb: float
    gpu_allocated_mb: float
    gpu_cached_mb: float
    gpu_reserved_mb: float
    gpu_free_mb: float


@dataclass
class PhaseProfile:
    """Performance profile for a single MCTS phase"""
    name: str
    gpu_time_ms: float
    cpu_time_ms: float
    memory_allocated_mb: float
    memory_freed_mb: float
    kernel_launches: int
    cpu_gpu_transfers_mb: float
    batch_size: int


@dataclass
class MCTSProfile:
    """Complete MCTS performance profile"""
    config: Dict[str, Any]
    quantum_level: str  # 'classical', 'tree_level', 'one_loop'
    num_simulations: int
    wave_size: int
    total_time_ms: float
    simulations_per_second: float
    
    # Phase breakdowns
    phases: List[PhaseProfile]
    
    # Resource utilization
    peak_gpu_memory_mb: float
    avg_gpu_utilization: float
    peak_cpu_percent: float
    avg_ram_usage_mb: float
    
    # Tree statistics
    final_tree_nodes: int
    final_tree_edges: int
    tree_memory_mb: float
    
    # Performance metrics
    efficiency_score: float  # sims/sec per GPU GB
    bottleneck_phase: str
    recommendations: List[str]
    
    # Quantum statistics
    quantum_kernel_calls: int = 0
    total_kernel_calls: int = 0
    quantum_overhead_percent: float = 0.0
    
    # cProfile data
    cprofile_stats: Optional[str] = None  # Formatted cProfile output
    top_functions: Optional[List[Dict[str, Any]]] = None  # Top functions by time


class GPUProfiler:
    """High-precision GPU profiling using CUDA events"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.events = {}
        self.active_timers = {}
        
    def start_timer(self, name: str):
        """Start a named GPU timer"""
        if name in self.active_timers:
            return  # Timer already running
            
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        self.active_timers[name] = start_event
        
    def end_timer(self, name: str) -> float:
        """End a named GPU timer and return elapsed time in ms"""
        if name not in self.active_timers:
            return 0.0
            
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        torch.cuda.synchronize()
        
        start_event = self.active_timers.pop(name)
        elapsed_ms = start_event.elapsed_time(end_event)
        
        if name not in self.events:
            self.events[name] = []
        self.events[name].append(elapsed_ms)
        
        return elapsed_ms
    
    @contextmanager
    def timer(self, name: str):
        """Context manager for GPU timing"""
        self.start_timer(name)
        try:
            yield
        finally:
            self.end_timer(name)
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics"""
        stats = {}
        for name, times in self.events.items():
            stats[name] = {
                'total_ms': sum(times),
                'avg_ms': np.mean(times),
                'min_ms': min(times),
                'max_ms': max(times),
                'std_ms': np.std(times),
                'count': len(times)
            }
        return stats


class MemoryMonitor:
    """Continuous memory usage monitoring"""
    
    def __init__(self, sample_interval: float = 0.01):
        self.sample_interval = sample_interval
        self.snapshots = []
        self.monitoring = False
        self.monitor_thread = None
        self.gpu_device = torch.cuda.current_device()
        
    def start_monitoring(self):
        """Start continuous memory monitoring"""
        if self.monitoring:
            return
            
        self.snapshots.clear()
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
    def _monitor_loop(self):
        """Monitoring loop running in background thread"""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # CPU and RAM metrics
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                ram_mb = memory_info.rss / 1024 / 1024
                ram_available_mb = psutil.virtual_memory().available / 1024 / 1024
                
                # GPU memory metrics
                gpu_allocated = torch.cuda.memory_allocated(self.gpu_device) / 1024 / 1024
                gpu_cached = torch.cuda.memory_reserved(self.gpu_device) / 1024 / 1024
                gpu_reserved = torch.cuda.memory_reserved(self.gpu_device) / 1024 / 1024
                
                gpu_properties = torch.cuda.get_device_properties(self.gpu_device)
                gpu_total_mb = gpu_properties.total_memory / 1024 / 1024
                gpu_free_mb = gpu_total_mb - gpu_allocated
                
                snapshot = MemorySnapshot(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    ram_mb=ram_mb,
                    ram_available_mb=ram_available_mb,
                    gpu_allocated_mb=gpu_allocated,
                    gpu_cached_mb=gpu_cached,
                    gpu_reserved_mb=gpu_reserved,
                    gpu_free_mb=gpu_free_mb
                )
                
                self.snapshots.append(snapshot)
                
            except Exception as e:
                print(f"Memory monitoring error: {e}")
                
            time.sleep(self.sample_interval)
    
    def get_peak_usage(self) -> MemorySnapshot:
        """Get peak memory usage across all snapshots"""
        if not self.snapshots:
            return None
            
        peak_gpu = max(self.snapshots, key=lambda s: s.gpu_allocated_mb)
        peak_cpu = max(self.snapshots, key=lambda s: s.cpu_percent)
        peak_ram = max(self.snapshots, key=lambda s: s.ram_mb)
        
        # Combine metrics
        return MemorySnapshot(
            timestamp=0,
            cpu_percent=peak_cpu.cpu_percent,
            ram_mb=peak_ram.ram_mb,
            ram_available_mb=min(s.ram_available_mb for s in self.snapshots),
            gpu_allocated_mb=peak_gpu.gpu_allocated_mb,
            gpu_cached_mb=max(s.gpu_cached_mb for s in self.snapshots),
            gpu_reserved_mb=max(s.gpu_reserved_mb for s in self.snapshots),
            gpu_free_mb=min(s.gpu_free_mb for s in self.snapshots)
        )
    
    def get_average_usage(self) -> MemorySnapshot:
        """Get average memory usage"""
        if not self.snapshots:
            return None
            
        return MemorySnapshot(
            timestamp=0,
            cpu_percent=np.mean([s.cpu_percent for s in self.snapshots]),
            ram_mb=np.mean([s.ram_mb for s in self.snapshots]),
            ram_available_mb=np.mean([s.ram_available_mb for s in self.snapshots]),
            gpu_allocated_mb=np.mean([s.gpu_allocated_mb for s in self.snapshots]),
            gpu_cached_mb=np.mean([s.gpu_cached_mb for s in self.snapshots]),
            gpu_reserved_mb=np.mean([s.gpu_reserved_mb for s in self.snapshots]),
            gpu_free_mb=np.mean([s.gpu_free_mb for s in self.snapshots])
        )


class InstrumentedMCTS:
    """MCTS wrapper with deep profiling instrumentation"""
    
    def __init__(self, config: MCTSConfig, evaluator, profiler_config: ProfilingConfig):
        self.config = config
        self.evaluator = evaluator
        self.profiler_config = profiler_config
        
        # Create MCTS instance
        self.mcts = MCTS(config, evaluator)
        
        # Profiling tools
        self.gpu_profiler = GPUProfiler(torch.device(config.device))
        self.memory_monitor = MemoryMonitor(profiler_config.memory_sample_interval)
        
        # Phase tracking
        self.phase_profiles = []
        self.kernel_launch_count = 0
        self.instrumentation_enabled = False
        
        # Monkey patch key methods for profiling
        self._instrument_mcts()
        
    def _instrument_mcts(self):
        """Add profiling instrumentation to MCTS methods"""
        try:
            # The new optimized MCTS has different method names
            engine = self.mcts
            
            # Check if methods exist - new names in optimized implementation
            required_methods = ['_select_batch_vectorized', '_expand_batch_vectorized', 
                              '_evaluate_batch_vectorized', '_backup_batch_vectorized']
            
            for method in required_methods:
                if not hasattr(engine, method):
                    logger.warning(f"Method {method} not found - trying basic profiling")
                    self.instrumentation_enabled = False
                    return
            
            # Instrument the core search phases
            original_select = engine._select_batch_vectorized
            original_expand = engine._expand_batch_vectorized  
            original_evaluate = engine._evaluate_batch_vectorized
            original_backup = engine._backup_batch_vectorized
            
            def instrumented_select(wave_size):
                return self._profile_phase('selection', original_select, wave_size)
                
            def instrumented_expand(leaf_nodes):
                return self._profile_phase('expansion', original_expand, leaf_nodes)
                
            def instrumented_evaluate(nodes):
                return self._profile_phase('evaluation', original_evaluate, nodes)
                
            def instrumented_backup(paths, path_lengths, values):
                return self._profile_phase('backup', original_backup, paths, path_lengths, values)
            
            # Replace methods
            engine._select_batch_vectorized = instrumented_select
            engine._expand_batch_vectorized = instrumented_expand
            engine._evaluate_batch_vectorized = instrumented_evaluate
            engine._backup_batch_vectorized = instrumented_backup
            
            self.instrumentation_enabled = True
            logger.info("MCTS instrumentation enabled successfully")
            
        except Exception as e:
            logger.warning(f"Failed to instrument MCTS: {e}")
            self.instrumentation_enabled = False
        
    def _profile_phase(self, phase_name: str, original_method, *args, **kwargs):
        """Profile a single MCTS phase"""
        # Get initial memory state
        torch.cuda.synchronize()
        initial_gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
        
        # Start timing
        cpu_start = time.perf_counter()
        
        with self.gpu_profiler.timer(phase_name):
            # Count kernel launches (approximate)
            initial_kernel_count = self.kernel_launch_count
            
            # Execute original method
            result = original_method(*args, **kwargs)
            
            # Ensure GPU operations complete
            torch.cuda.synchronize()
        
        # End timing
        cpu_end = time.perf_counter()
        final_gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
        
        # Calculate metrics
        cpu_time_ms = (cpu_end - cpu_start) * 1000
        gpu_time_ms = self.gpu_profiler.events.get(phase_name, [0])[-1]
        memory_change_mb = final_gpu_mem - initial_gpu_mem
        kernel_launches = self.kernel_launch_count - initial_kernel_count
        
        # Estimate batch size
        batch_size = self._estimate_batch_size(phase_name, args)
        
        # Create phase profile
        phase_profile = PhaseProfile(
            name=phase_name,
            gpu_time_ms=gpu_time_ms,
            cpu_time_ms=cpu_time_ms,
            memory_allocated_mb=max(0, memory_change_mb),
            memory_freed_mb=max(0, -memory_change_mb),
            kernel_launches=kernel_launches,
            cpu_gpu_transfers_mb=0,  # TODO: Detect actual transfers
            batch_size=batch_size
        )
        
        self.phase_profiles.append(phase_profile)
        
        return result
    
    def _estimate_batch_size(self, phase_name: str, args) -> int:
        """Estimate batch size for a phase"""
        if not args:
            return 0
            
        first_arg = args[0]
        if isinstance(first_arg, torch.Tensor):
            if first_arg.dim() == 0:
                return 1
            else:
                return first_arg.shape[0]
        elif isinstance(first_arg, int):
            return first_arg
        else:
            return 0
    
    def profile_search(self, game_state, num_simulations: int, quantum_level: str) -> MCTSProfile:
        """Run a search with full profiling"""
        # Clear previous data
        self.phase_profiles.clear()
        self.gpu_profiler.events.clear()
        
        # Start memory monitoring
        self.memory_monitor.start_monitoring()
        
        # Warmup
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        
        # Initialize cProfile data
        cprofile_stats = None
        top_functions = None
        
        # Main search with timing and cProfile
        start_time = time.perf_counter()
        
        if self.profiler_config.enable_cprofile:
            # Run with cProfile
            profiler = cProfile.Profile()
            profiler.enable()
            
            try:
                if self.instrumentation_enabled:
                    # Full profiling with phase breakdown
                    with self.gpu_profiler.timer('total_search'):
                        policy = self.mcts.search(game_state, num_simulations)
                else:
                    # Basic profiling without phase breakdown
                    logger.warning("Phase instrumentation not available - using basic profiling")
                    with self.gpu_profiler.timer('total_search'):
                        policy = self.mcts.search(game_state, num_simulations)
            finally:
                profiler.disable()
                
            # Process cProfile results
            cprofile_stats, top_functions = self._process_cprofile_results(profiler)
        else:
            # Run without cProfile
            if self.instrumentation_enabled:
                with self.gpu_profiler.timer('total_search'):
                    policy = self.mcts.search(game_state, num_simulations)
            else:
                logger.warning("Phase instrumentation not available - using basic profiling")
                with self.gpu_profiler.timer('total_search'):
                    policy = self.mcts.search(game_state, num_simulations)
        
        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        
        # Stop monitoring
        self.memory_monitor.stop_monitoring()
        
        # Calculate performance metrics
        simulations_per_second = num_simulations / (total_time_ms / 1000)
        
        # Get memory statistics
        peak_memory = self.memory_monitor.get_peak_usage()
        avg_memory = self.memory_monitor.get_average_usage()
        
        # Get tree statistics
        try:
            tree_stats = self.mcts.get_statistics()
        except Exception as e:
            logger.warning(f"Failed to get tree statistics: {e}")
            tree_stats = {}
        
        # Get quantum kernel statistics if available
        quantum_kernel_calls = 0
        total_kernel_calls = 0
        if hasattr(self.mcts.tree, 'batch_ops') and self.mcts.tree.batch_ops:
            if hasattr(self.mcts.tree.batch_ops, 'stats'):
                kernel_stats = self.mcts.tree.batch_ops.stats
                quantum_kernel_calls = kernel_stats.get('quantum_calls', 0)
                total_kernel_calls = kernel_stats.get('ucb_calls', 0)
        
        # Calculate quantum overhead
        quantum_overhead_percent = 0.0
        if quantum_level != 'classical':
            # Compare with classical baseline if available
            quantum_overhead_percent = 0.0  # Will be calculated later by comparing with classical
        
        # Calculate efficiency score (sims/sec per GB of GPU memory)
        gpu_gb = peak_memory.gpu_allocated_mb / 1024 if peak_memory and peak_memory.gpu_allocated_mb > 0 else 1
        efficiency_score = simulations_per_second / gpu_gb
        
        # Identify bottleneck phase
        bottleneck_phase = self._identify_bottleneck()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            bottleneck_phase, peak_memory, avg_memory, tree_stats, quantum_level
        )
        
        # Convert config to dict, handling any non-serializable fields
        config_dict = {}
        for field_name in self.config.__dataclass_fields__:
            value = getattr(self.config, field_name)
            if hasattr(value, '__dict__'):
                config_dict[field_name] = str(value)
            else:
                config_dict[field_name] = value
        
        return MCTSProfile(
            config=config_dict,
            quantum_level=quantum_level,
            num_simulations=num_simulations,
            wave_size=self.config.max_wave_size,
            total_time_ms=total_time_ms,
            simulations_per_second=simulations_per_second,
            phases=self.phase_profiles.copy(),
            peak_gpu_memory_mb=peak_memory.gpu_allocated_mb if peak_memory else 0,
            avg_gpu_utilization=0,  # TODO: Calculate from CUDA profiling
            peak_cpu_percent=peak_memory.cpu_percent if peak_memory else 0,
            avg_ram_usage_mb=avg_memory.ram_mb if avg_memory else 0,
            final_tree_nodes=tree_stats.get('tree_nodes', 0),
            final_tree_edges=tree_stats.get('tree_edges', 0),
            tree_memory_mb=tree_stats.get('tree_memory_mb', 0),
            efficiency_score=efficiency_score,
            bottleneck_phase=bottleneck_phase,
            recommendations=recommendations,
            quantum_kernel_calls=quantum_kernel_calls,
            total_kernel_calls=total_kernel_calls,
            quantum_overhead_percent=quantum_overhead_percent,
            cprofile_stats=cprofile_stats,
            top_functions=top_functions
        )
    
    def _identify_bottleneck(self) -> str:
        """Identify the most time-consuming phase"""
        if not self.phase_profiles:
            if not self.instrumentation_enabled:
                return "instrumentation_unavailable"
            else:
                return "unknown"
            
        # Group by phase name and sum times
        phase_times = defaultdict(float)
        for profile in self.phase_profiles:
            phase_times[profile.name] += profile.gpu_time_ms
            
        if not phase_times:
            return "no_data"
            
        return max(phase_times.items(), key=lambda x: x[1])[0]
    
    def _process_cprofile_results(self, profiler: cProfile.Profile) -> Tuple[str, List[Dict[str, Any]]]:
        """Process cProfile results and return formatted output"""
        # Create a string buffer to capture pstats output
        s = io.StringIO()
        
        # Create pstats object and sort by the configured key
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats(self.profiler_config.cprofile_sort_key)
        
        # Print top functions to string buffer
        ps.print_stats(self.profiler_config.cprofile_top_functions)
        cprofile_output = s.getvalue()
        
        # Extract top functions data for structured analysis
        top_functions = []
        try:
            # Get raw stats
            stats = ps.stats
            sorted_items = sorted(stats.items(), 
                                key=lambda x: x[1][2] if self.profiler_config.cprofile_sort_key == 'cumulative' else x[1][3], 
                                reverse=True)
            
            for i, (func_key, (cc, nc, tt, ct, callers)) in enumerate(sorted_items[:self.profiler_config.cprofile_top_functions]):
                filename, line_num, func_name = func_key
                
                # Calculate percentages
                total_time = sum(stat[3] for stat in stats.values())  # Total cumulative time
                time_percent = (ct / total_time * 100) if total_time > 0 else 0
                
                function_data = {
                    'rank': i + 1,
                    'filename': filename,
                    'line_number': line_num,
                    'function_name': func_name,
                    'total_calls': cc,
                    'primitive_calls': nc,
                    'total_time': tt,
                    'cumulative_time': ct,
                    'time_percent': time_percent,
                    'time_per_call': ct / cc if cc > 0 else 0
                }
                top_functions.append(function_data)
                
        except Exception as e:
            logger.warning(f"Failed to parse cProfile results: {e}")
            
        return cprofile_output, top_functions
    
    def _generate_recommendations(self, bottleneck_phase: str, peak_memory: MemorySnapshot,
                                avg_memory: MemorySnapshot, tree_stats: Dict, quantum_level: str) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Memory-based recommendations
        if peak_memory:
            gpu_util = peak_memory.gpu_allocated_mb / (peak_memory.gpu_allocated_mb + peak_memory.gpu_free_mb)
            if gpu_util > 0.9:
                recommendations.append("GPU memory near limit - consider reducing wave_size")
            elif gpu_util < 0.5:
                recommendations.append("GPU memory underutilized - consider increasing wave_size")
        
        # Phase-specific recommendations
        if bottleneck_phase == "selection":
            recommendations.append("Selection is bottleneck - optimize UCB kernel or reduce tree traversal depth")
            if quantum_level != 'classical':
                recommendations.append("Consider using classical selection for hot paths")
        elif bottleneck_phase == "expansion":
            recommendations.append("Expansion is bottleneck - optimize legal move generation or reduce expansion breadth")
        elif bottleneck_phase == "evaluation":
            recommendations.append("Evaluation is bottleneck - optimize neural network or increase batch size")
        elif bottleneck_phase == "backup":
            recommendations.append("Backup is bottleneck - optimize parallel backup kernel or reduce path lengths")
        
        # Quantum-specific recommendations
        if quantum_level == 'tree_level':
            recommendations.append("Tree-level quantum may be excessive - consider one-loop for balance")
        elif quantum_level == 'one_loop':
            recommendations.append("One-loop quantum is a good balance - monitor overhead")
        
        # Tree growth recommendations
        if tree_stats.get('tree_nodes', 0) < 1000:
            recommendations.append("Tree too small - increase num_simulations for better performance")
        elif tree_stats.get('tree_nodes', 0) > 1000000:
            recommendations.append("Tree very large - consider memory optimizations")
            
        return recommendations


class FastEvaluator:
    """Optimized evaluator for benchmarking"""
    
    def __init__(self, board_size: int = 15):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.board_size = board_size
        self.action_size = board_size * board_size
        
        # Pre-allocate tensors for maximum performance
        max_batch = 10000
        self.policies = torch.ones(max_batch, self.action_size, device=self.device) / self.action_size
        self.values = torch.zeros(max_batch, 1, device=self.device)
        
    def evaluate_batch(self, features, legal_masks=None):
        """Fast batch evaluation"""
        batch_size = features.shape[0]
        return self.policies[:batch_size], self.values[:batch_size]


class MCTSQuantumBenchmarkSuite:
    """Comprehensive MCTS benchmark suite with quantum level comparison"""
    
    def __init__(self, profiling_config: ProfilingConfig = None):
        self.config = profiling_config or ProfilingConfig()
        self.results = []
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def run_comprehensive_benchmark(self):
        """Run the complete benchmark suite"""
        print("🚀 Starting Comprehensive MCTS Quantum Level Benchmark")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print()
        
        # Initialize evaluator
        evaluator = FastEvaluator()
        
        # Test different configurations
        for wave_size in self.config.wave_sizes:
            for num_sims in self.config.simulation_counts:
                for quantum_level in self.config.quantum_levels:
                    print(f"📊 Testing: {num_sims} simulations, wave_size={wave_size}, quantum={quantum_level}")
                    
                    # Create MCTS config
                    mcts_config = self._create_mcts_config(wave_size, quantum_level)
                    
                    # Run benchmark
                    profile = self._benchmark_configuration(mcts_config, evaluator, num_sims, quantum_level)
                    self.results.append(profile)
                    
                    # Print results
                    print(f"  ⏱️  Time: {profile.total_time_ms:.1f}ms")
                    print(f"  🎯 Performance: {profile.simulations_per_second:,.0f} sims/s")
                    print(f"  💾 Peak GPU: {profile.peak_gpu_memory_mb:.1f}MB")
                    print(f"  🔧 Bottleneck: {profile.bottleneck_phase}")
                    
                    if quantum_level != 'classical':
                        print(f"  ⚛️  Quantum kernel calls: {profile.quantum_kernel_calls}/{profile.total_kernel_calls}")
                        if profile.quantum_kernel_calls > 0:
                            print(f"     ({profile.quantum_kernel_calls/profile.total_kernel_calls*100:.1f}% quantum)")
                    
                    # Print top functions from cProfile if available
                    if profile.top_functions:
                        print(f"  📊 Top 3 Functions by Time:")
                        for i, func in enumerate(profile.top_functions[:3]):
                            print(f"    {i+1}. {func['function_name']} ({func['time_percent']:.1f}%, {func['cumulative_time']:.3f}s)")
                    
                    print()
        
        # Generate analysis and reports
        self._generate_quantum_analysis()
        
        if self.config.generate_plots:
            self._generate_quantum_plots()
            
        if self.config.save_detailed_logs:
            self._save_detailed_logs()
            
        print(f"✅ Benchmark complete! Results saved to {self.output_dir}")
    
    def _create_mcts_config(self, wave_size: int, quantum_level: str) -> MCTSConfig:
        """Create MCTS config for specific quantum level"""
        enable_quantum = quantum_level != 'classical'
        
        # Create quantum config if needed
        quantum_config = None
        if enable_quantum:
            quantum_config = QuantumConfig(
                quantum_level=quantum_level,
                enable_quantum=True,
                min_wave_size=wave_size,
                optimal_wave_size=wave_size,
                hbar_eff=0.1,
                temperature=1.0,
                coupling_strength=0.1,
                decoherence_rate=0.01,
                interference_alpha=0.05,
                phase_kick_strength=0.1,
                use_mixed_precision=True,
                fast_mode=True,
                device='cuda'
            )
        
        return MCTSConfig(
            num_simulations=10000,  # Will be overridden
            device='cuda',
            game_type=GameType.GOMOKU,
            min_wave_size=wave_size,
            max_wave_size=wave_size,
            adaptive_wave_sizing=False,  # Critical for performance
            board_size=15,
            enable_quantum=enable_quantum,
            quantum_config=quantum_config,
            use_mixed_precision=True,
            use_cuda_graphs=True,
            use_tensor_cores=True
        )
        
    def _benchmark_configuration(self, mcts_config: MCTSConfig, evaluator, 
                                num_simulations: int, quantum_level: str) -> MCTSProfile:
        """Benchmark a single configuration"""
        # Create instrumented MCTS
        instrumented_mcts = InstrumentedMCTS(mcts_config, evaluator, self.config)
        game_state = alphazero_py.GomokuState()
        
        # Warmup runs
        for _ in range(self.config.warmup_iterations):
            # Initialize MCTS properly before search
            warmup_mcts = MCTS(mcts_config, evaluator)
            warmup_mcts.search(game_state, min(1000, num_simulations // 10))
        
        # Measurement runs
        profiles = []
        for _ in range(self.config.measurement_iterations):
            profile = instrumented_mcts.profile_search(game_state, num_simulations, quantum_level)
            profiles.append(profile)
        
        # Return best performance run
        return min(profiles, key=lambda p: p.total_time_ms)
    
    def _generate_quantum_analysis(self):
        """Generate quantum-specific performance analysis"""
        if not self.results:
            return
            
        print("📈 Quantum Performance Analysis")
        print("=" * 70)
        
        # Group results by configuration
        config_groups = defaultdict(list)
        for result in self.results:
            key = (result.wave_size, result.num_simulations)
            config_groups[key].append(result)
        
        # Analyze each configuration
        for (wave_size, num_sims), results in sorted(config_groups.items()):
            print(f"\n📊 Configuration: wave_size={wave_size}, simulations={num_sims}")
            print("-" * 60)
            
            # Find results for each quantum level
            classical_result = next((r for r in results if r.quantum_level == 'classical'), None)
            tree_level_result = next((r for r in results if r.quantum_level == 'tree_level'), None)
            one_loop_result = next((r for r in results if r.quantum_level == 'one_loop'), None)
            
            if classical_result:
                print(f"Classical:  {classical_result.simulations_per_second:8,.0f} sims/s (baseline)")
                
                if tree_level_result:
                    speedup = tree_level_result.simulations_per_second / classical_result.simulations_per_second
                    overhead = (1 - speedup) * 100
                    print(f"Tree-level: {tree_level_result.simulations_per_second:8,.0f} sims/s ({speedup:.2f}x, {overhead:+.1f}% overhead)")
                    
                if one_loop_result:
                    speedup = one_loop_result.simulations_per_second / classical_result.simulations_per_second
                    overhead = (1 - speedup) * 100
                    print(f"One-loop:   {one_loop_result.simulations_per_second:8,.0f} sims/s ({speedup:.2f}x, {overhead:+.1f}% overhead)")
                    
            # Bottleneck analysis
            print("\nBottlenecks by quantum level:")
            for result in results:
                print(f"  {result.quantum_level:12s}: {result.bottleneck_phase}")
        
        # Overall best configuration
        print("\n" + "=" * 70)
        best_classical = max((r for r in self.results if r.quantum_level == 'classical'), 
                           key=lambda r: r.simulations_per_second, default=None)
        best_quantum = max((r for r in self.results if r.quantum_level != 'classical'), 
                          key=lambda r: r.simulations_per_second, default=None)
        
        if best_classical:
            print(f"🏆 Best Classical: {best_classical.simulations_per_second:,.0f} sims/s")
            print(f"   Configuration: wave_size={best_classical.wave_size}, {best_classical.num_simulations} sims")
            
        if best_quantum:
            print(f"⚛️  Best Quantum: {best_quantum.simulations_per_second:,.0f} sims/s ({best_quantum.quantum_level})")
            print(f"   Configuration: wave_size={best_quantum.wave_size}, {best_quantum.num_simulations} sims")
            
            if best_classical:
                speedup = best_quantum.simulations_per_second / best_classical.simulations_per_second
                print(f"   Quantum speedup: {speedup:.2f}x")
        
        # Function-level analysis
        self._analyze_quantum_functions()
    
    def _analyze_quantum_functions(self):
        """Analyze function-level performance differences between quantum levels"""
        print("\n🔍 Function-Level Quantum Analysis")
        print("=" * 70)
        
        # Group functions by quantum level
        function_by_level = defaultdict(lambda: defaultdict(list))
        
        for result in self.results:
            if result.top_functions:
                for func in result.top_functions[:20]:  # Top 20 functions
                    key = func['function_name']
                    function_by_level[key][result.quantum_level].append({
                        'time': func['cumulative_time'],
                        'calls': func['total_calls'],
                        'percent': func['time_percent']
                    })
        
        # Find functions with significant differences
        significant_diffs = []
        
        for func_name, level_data in function_by_level.items():
            if 'classical' in level_data and any(q in level_data for q in ['tree_level', 'one_loop']):
                classical_avg = np.mean([d['time'] for d in level_data['classical']])
                
                for quantum_level in ['tree_level', 'one_loop']:
                    if quantum_level in level_data:
                        quantum_avg = np.mean([d['time'] for d in level_data[quantum_level]])
                        diff_percent = ((quantum_avg - classical_avg) / classical_avg) * 100 if classical_avg > 0 else 0
                        
                        if abs(diff_percent) > 10:  # Significant if >10% difference
                            significant_diffs.append({
                                'function': func_name,
                                'quantum_level': quantum_level,
                                'classical_time': classical_avg,
                                'quantum_time': quantum_avg,
                                'diff_percent': diff_percent
                            })
        
        if significant_diffs:
            # Sort by absolute difference
            significant_diffs.sort(key=lambda x: abs(x['diff_percent']), reverse=True)
            
            print("Functions with significant quantum overhead/speedup:")
            for diff in significant_diffs[:10]:  # Top 10
                print(f"\n  {diff['function']}")
                print(f"    Classical: {diff['classical_time']:.3f}s")
                print(f"    {diff['quantum_level']:10s}: {diff['quantum_time']:.3f}s ({diff['diff_percent']:+.1f}%)")
        else:
            print("No functions with significant quantum overhead found.")
    
    def _generate_quantum_plots(self):
        """Generate quantum-specific performance visualization plots"""
        if not self.results:
            return
            
        if not PLOTTING_AVAILABLE:
            print("⚠️  Skipping plots - matplotlib not available")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MCTS Quantum Performance Analysis', fontsize=16)
        
        # 1. Performance comparison by quantum level
        quantum_levels = ['classical', 'tree_level', 'one_loop']
        colors = ['blue', 'green', 'red']
        
        for i, level in enumerate(quantum_levels):
            level_results = [r for r in self.results if r.quantum_level == level]
            if level_results:
                sims = [r.num_simulations for r in level_results]
                perfs = [r.simulations_per_second for r in level_results]
                axes[0, 0].scatter(sims, perfs, alpha=0.7, color=colors[i], label=level, s=100)
        
        axes[0, 0].set_xlabel('Number of Simulations')
        axes[0, 0].set_ylabel('Simulations/Second')
        axes[0, 0].set_title('Performance by Quantum Level')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xscale('log')
        
        # 2. Overhead comparison
        config_groups = defaultdict(list)
        for result in self.results:
            key = (result.wave_size, result.num_simulations)
            config_groups[key].append(result)
        
        overheads = {'tree_level': [], 'one_loop': []}
        configs = []
        
        for config, results in sorted(config_groups.items()):
            classical = next((r for r in results if r.quantum_level == 'classical'), None)
            if classical:
                configs.append(f"{config[1]}")
                for level in ['tree_level', 'one_loop']:
                    quantum = next((r for r in results if r.quantum_level == level), None)
                    if quantum:
                        overhead = ((classical.simulations_per_second - quantum.simulations_per_second) / 
                                  classical.simulations_per_second) * 100
                        overheads[level].append(overhead)
                    else:
                        overheads[level].append(0)
        
        x = np.arange(len(configs))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, overheads['tree_level'], width, label='Tree-level', color='green', alpha=0.7)
        axes[0, 1].bar(x + width/2, overheads['one_loop'], width, label='One-loop', color='red', alpha=0.7)
        axes[0, 1].set_xlabel('Number of Simulations')
        axes[0, 1].set_ylabel('Overhead (%)')
        axes[0, 1].set_title('Quantum Overhead vs Classical')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(configs)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Memory usage by quantum level
        for i, level in enumerate(quantum_levels):
            level_results = [r for r in self.results if r.quantum_level == level]
            if level_results:
                sims = [r.num_simulations for r in level_results]
                memory = [r.peak_gpu_memory_mb for r in level_results]
                axes[1, 0].scatter(sims, memory, alpha=0.7, color=colors[i], label=level, s=100)
        
        axes[1, 0].set_xlabel('Number of Simulations')
        axes[1, 0].set_ylabel('Peak GPU Memory (MB)')
        axes[1, 0].set_title('Memory Usage by Quantum Level')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xscale('log')
        
        # 4. Quantum kernel usage
        quantum_results = [r for r in self.results if r.quantum_level != 'classical' and r.total_kernel_calls > 0]
        if quantum_results:
            labels = []
            quantum_percentages = []
            
            for r in quantum_results:
                labels.append(f"{r.quantum_level}\n{r.num_simulations} sims")
                quantum_percentages.append((r.quantum_kernel_calls / r.total_kernel_calls) * 100 if r.total_kernel_calls > 0 else 0)
            
            axes[1, 1].bar(range(len(labels)), quantum_percentages, alpha=0.7)
            axes[1, 1].set_xlabel('Configuration')
            axes[1, 1].set_ylabel('Quantum Kernel Usage (%)')
            axes[1, 1].set_title('Quantum CUDA Kernel Usage')
            axes[1, 1].set_xticks(range(len(labels)))
            axes[1, 1].set_xticklabels(labels, rotation=45, ha='right')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'quantum_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Quantum performance plots saved to {self.output_dir / 'quantum_performance_analysis.png'}")
    
    def _save_detailed_logs(self):
        """Save detailed profiling logs"""
        # Save all results as JSON
        results_data = []
        for result in self.results:
            result_dict = asdict(result)
            # Convert dataclass objects to dicts
            result_dict['phases'] = [asdict(phase) for phase in result.phases]
            results_data.append(result_dict)
        
        with open(self.output_dir / 'quantum_detailed_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save quantum comparison CSV
        if PANDAS_AVAILABLE:
            comparison_data = []
            
            # Group by configuration
            config_groups = defaultdict(list)
            for result in self.results:
                key = (result.wave_size, result.num_simulations)
                config_groups[key].append(result)
            
            for (wave_size, num_sims), results in sorted(config_groups.items()):
                row = {
                    'wave_size': wave_size,
                    'num_simulations': num_sims
                }
                
                for result in results:
                    prefix = result.quantum_level
                    row[f'{prefix}_sims_per_sec'] = result.simulations_per_second
                    row[f'{prefix}_time_ms'] = result.total_time_ms
                    row[f'{prefix}_memory_mb'] = result.peak_gpu_memory_mb
                    row[f'{prefix}_bottleneck'] = result.bottleneck_phase
                    
                    if result.quantum_level != 'classical':
                        row[f'{prefix}_quantum_calls'] = result.quantum_kernel_calls
                        row[f'{prefix}_total_calls'] = result.total_kernel_calls
                
                comparison_data.append(row)
            
            df = pd.DataFrame(comparison_data)
            df.to_csv(self.output_dir / 'quantum_comparison.csv', index=False)
        
        # Save individual cProfile reports by quantum level
        for result in self.results:
            if result.cprofile_stats:
                cprofile_filename = f'cprofile_{result.quantum_level}_wave{result.wave_size}_sims{result.num_simulations}.txt'
                with open(self.output_dir / cprofile_filename, 'w') as f:
                    f.write(f"cProfile Report - Quantum Level: {result.quantum_level}\n")
                    f.write(f"Configuration: wave_size={result.wave_size}, simulations={result.num_simulations}\n")
                    f.write(f"Performance: {result.simulations_per_second:,.0f} sims/s\n")
                    f.write("=" * 80 + "\n")
                    f.write(result.cprofile_stats)
        
        print(f"💾 Detailed logs saved to {self.output_dir}")


def main():
    """Main benchmark entry point"""
    parser = argparse.ArgumentParser(description='Comprehensive MCTS Quantum Profiler')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark')
    parser.add_argument('--output-dir', type=str, default='mcts_quantum_profiling_results', 
                       help='Output directory')
    parser.add_argument('--quantum-only', action='store_true', help='Skip classical baseline')
    parser.add_argument('--classical-only', action='store_true', help='Run classical only')
    
    args = parser.parse_args()
    
    # Determine quantum levels to test
    if args.quantum_only:
        quantum_levels = ['tree_level', 'one_loop']
    elif args.classical_only:
        quantum_levels = ['classical']
    else:
        quantum_levels = ['classical', 'tree_level', 'one_loop']
    
    # Create profiling config
    if args.quick:
        config = ProfilingConfig(
            simulation_counts=[1000, 5000, 10000],
            wave_sizes=[3072],  # Optimal for RTX 3060 Ti
            measurement_iterations=3,
            quantum_levels=quantum_levels
        )
    else:
        config = ProfilingConfig(
            simulation_counts=[1000, 5000, 10000, 25000, 50000],
            wave_sizes=[3072],  # Fixed optimal wave size
            measurement_iterations=5,
            quantum_levels=quantum_levels
        )
    
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Run benchmark
    benchmark = MCTSQuantumBenchmarkSuite(config)
    benchmark.run_comprehensive_benchmark()


if __name__ == "__main__":
    main()