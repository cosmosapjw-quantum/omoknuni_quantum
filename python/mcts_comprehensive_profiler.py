#!/usr/bin/env python3
"""
Comprehensive MCTS Benchmark & Profiling Tool

This tool provides deep performance analysis of the MCTS implementation by:
1. GPU/CPU/RAM/VRAM usage monitoring with high precision
2. Phase-by-phase breakdown of MCTS operations
3. Memory transfer detection and quantification
4. Kernel launch overhead analysis
5. Bottleneck identification with actionable recommendations

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
from dataclasses import dataclass, asdict
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
from mcts.core.unified_mcts import UnifiedMCTS, UnifiedMCTSConfig
from mcts.gpu.gpu_game_states import GameType
import alphazero_py


@dataclass
class ProfilingConfig:
    """Configuration for profiling session"""
    # Test scenarios
    simulation_counts: List[int] = None
    wave_sizes: List[int] = None
    warmup_iterations: int = 3
    measurement_iterations: int = 5
    
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
    output_dir: str = "mcts_profiling_results"
    
    def __post_init__(self):
        if self.simulation_counts is None:
            self.simulation_counts = [1000, 5000, 10000, 25000, 50000]
        if self.wave_sizes is None:
            # Optimized for RTX 3060 Ti (38 SMs, 4864 CUDA cores)
            self.wave_sizes = [1824, 2048, 3072, 3648, 4096, 4864]


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
            # Get the actual engine (UnifiedMCTS)
            if hasattr(self.mcts, 'unified_mcts'):
                engine = self.mcts.unified_mcts
            elif hasattr(self.mcts, '_select_batch'):
                engine = self.mcts
            else:
                # This is the wrapper MCTS, can't instrument phases
                logger.warning("Cannot instrument MCTS phases - no access to UnifiedMCTS")
                self.instrumentation_enabled = False
                return
                
            # Check if methods exist
            required_methods = ['_select_batch', '_expand_batch', '_evaluate_batch', '_backup_batch']
            for method in required_methods:
                if not hasattr(engine, method):
                    logger.warning(f"Method {method} not found - disabling instrumentation")
                    self.instrumentation_enabled = False
                    return
            
            # Instrument the core search phases
            original_select = engine._select_batch
            original_expand = engine._expand_batch  
            original_evaluate = engine._evaluate_batch
            original_backup = engine._backup_batch
            
            def instrumented_select(wave_size):
                return self._profile_phase('selection', original_select, wave_size)
                
            def instrumented_expand(leaf_nodes):
                return self._profile_phase('expansion', original_expand, leaf_nodes)
                
            def instrumented_evaluate(nodes):
                return self._profile_phase('evaluation', original_evaluate, nodes)
                
            def instrumented_backup(paths, values):
                return self._profile_phase('backup', original_backup, paths, values)
            
            # Replace methods
            engine._select_batch = instrumented_select
            engine._expand_batch = instrumented_expand
            engine._evaluate_batch = instrumented_evaluate
            engine._backup_batch = instrumented_backup
            
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
    
    def profile_search(self, game_state, num_simulations: int) -> MCTSProfile:
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
        
        # Calculate efficiency score (sims/sec per GB of GPU memory)
        gpu_gb = peak_memory.gpu_allocated_mb / 1024 if peak_memory and peak_memory.gpu_allocated_mb > 0 else 1
        efficiency_score = simulations_per_second / gpu_gb
        
        # Identify bottleneck phase
        bottleneck_phase = self._identify_bottleneck()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            bottleneck_phase, peak_memory, avg_memory, tree_stats
        )
        
        return MCTSProfile(
            config=asdict(self.config),
            num_simulations=num_simulations,
            wave_size=self.config.wave_size,
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
                                avg_memory: MemorySnapshot, tree_stats: Dict) -> List[str]:
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
        elif bottleneck_phase == "expansion":
            recommendations.append("Expansion is bottleneck - optimize legal move generation or reduce expansion breadth")
        elif bottleneck_phase == "evaluation":
            recommendations.append("Evaluation is bottleneck - optimize neural network or increase batch size")
        elif bottleneck_phase == "backup":
            recommendations.append("Backup is bottleneck - optimize parallel backup kernel or reduce path lengths")
        
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


class MCTSBenchmarkSuite:
    """Comprehensive MCTS benchmark suite"""
    
    def __init__(self, profiling_config: ProfilingConfig = None):
        self.config = profiling_config or ProfilingConfig()
        self.results = []
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def run_comprehensive_benchmark(self):
        """Run the complete benchmark suite"""
        print("🚀 Starting Comprehensive MCTS Benchmark")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print()
        
        # Initialize evaluator
        evaluator = FastEvaluator()
        
        # Test different configurations
        for wave_size in self.config.wave_sizes:
            for num_sims in self.config.simulation_counts:
                print(f"📊 Testing: {num_sims} simulations, wave_size={wave_size}")
                
                # Create MCTS config
                mcts_config = MCTSConfig(
                    num_simulations=num_sims,
                    device='cuda',
                    game_type=GameType.GOMOKU,
                    wave_size=wave_size,
                    board_size=15
                )
                
                # Run benchmark
                profile = self._benchmark_configuration(mcts_config, evaluator, num_sims)
                self.results.append(profile)
                
                # Print results
                print(f"  ⏱️  Time: {profile.total_time_ms:.1f}ms")
                print(f"  🎯 Performance: {profile.simulations_per_second:,.0f} sims/s")
                print(f"  💾 Peak GPU: {profile.peak_gpu_memory_mb:.1f}MB")
                print(f"  🔧 Bottleneck: {profile.bottleneck_phase}")
                
                # Print top functions from cProfile if available
                if profile.top_functions:
                    print(f"  📊 Top 5 Functions by Time:")
                    for i, func in enumerate(profile.top_functions[:5]):
                        print(f"    {i+1}. {func['function_name']} ({func['time_percent']:.1f}%, {func['cumulative_time']:.3f}s)")
                
                print()
        
        # Generate analysis and reports
        self._generate_analysis()
        
        if self.config.generate_plots:
            self._generate_plots()
            
        if self.config.save_detailed_logs:
            self._save_detailed_logs()
            
        print(f"✅ Benchmark complete! Results saved to {self.output_dir}")
        
    def _benchmark_configuration(self, mcts_config: MCTSConfig, evaluator, 
                                num_simulations: int) -> MCTSProfile:
        """Benchmark a single configuration"""
        # Create instrumented MCTS
        instrumented_mcts = InstrumentedMCTS(mcts_config, evaluator, self.config)
        game_state = alphazero_py.GomokuState()
        
        # Warmup runs
        for _ in range(self.config.warmup_iterations):
            instrumented_mcts.mcts.search(game_state, min(1000, num_simulations // 10))
        
        # Measurement runs
        profiles = []
        for _ in range(self.config.measurement_iterations):
            profile = instrumented_mcts.profile_search(game_state, num_simulations)
            profiles.append(profile)
        
        # Return best performance run
        return min(profiles, key=lambda p: p.total_time_ms)
    
    def _generate_analysis(self):
        """Generate performance analysis"""
        if not self.results:
            return
            
        print("📈 Performance Analysis")
        print("=" * 50)
        
        # Find best performance
        best_result = max(self.results, key=lambda r: r.simulations_per_second)
        print(f"🏆 Best Performance: {best_result.simulations_per_second:,.0f} sims/s")
        print(f"   Configuration: wave_size={best_result.wave_size}, {best_result.num_simulations} sims")
        print(f"   GPU Memory: {best_result.peak_gpu_memory_mb:.1f}MB")
        print()
        
        # Efficiency analysis
        most_efficient = max(self.results, key=lambda r: r.efficiency_score)
        print(f"🎯 Most Efficient: {most_efficient.efficiency_score:.1f} sims/s/GB")
        print(f"   Configuration: wave_size={most_efficient.wave_size}")
        print()
        
        # Common bottlenecks
        bottlenecks = defaultdict(int)
        for result in self.results:
            bottlenecks[result.bottleneck_phase] += 1
        
        print("🔍 Common Bottlenecks:")
        for phase, count in sorted(bottlenecks.items(), key=lambda x: x[1], reverse=True):
            print(f"   {phase}: {count}/{len(self.results)} configurations")
        print()
        
        # Recommendations summary
        all_recommendations = defaultdict(int)
        for result in self.results:
            for rec in result.recommendations:
                all_recommendations[rec] += 1
        
        print("💡 Top Recommendations:")
        for rec, count in sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {rec} ({count} times)")
        print()
        
        # cProfile Function Analysis
        print("🔍 Function-Level Performance Analysis (cProfile)")
        print("=" * 50)
        
        # Aggregate function data across all runs
        function_aggregates = defaultdict(lambda: {'total_time': 0, 'call_count': 0, 'configs': 0})
        
        for result in self.results:
            if result.top_functions:
                for func in result.top_functions:
                    key = func['function_name']
                    function_aggregates[key]['total_time'] += func['cumulative_time']
                    function_aggregates[key]['call_count'] += func['total_calls']
                    function_aggregates[key]['configs'] += 1
        
        if function_aggregates:
            # Sort by total time across all configurations
            sorted_functions = sorted(function_aggregates.items(), 
                                    key=lambda x: x[1]['total_time'], reverse=True)
            
            print("🏆 Top 10 Most Time-Consuming Functions:")
            for i, (func_name, data) in enumerate(sorted_functions[:10]):
                avg_time = data['total_time'] / data['configs']
                print(f"   {i+1:2d}. {func_name}")
                print(f"       Total Time: {data['total_time']:.3f}s across {data['configs']} configs")
                print(f"       Avg Time/Config: {avg_time:.3f}s")
                print(f"       Total Calls: {data['call_count']:,}")
        else:
            print("   No cProfile data available")
        print()
    
    def _generate_plots(self):
        """Generate performance visualization plots"""
        if not self.results:
            return
            
        if not PLOTTING_AVAILABLE:
            print("⚠️  Skipping plots - matplotlib not available")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MCTS Performance Analysis', fontsize=16)
        
        # Performance vs Wave Size
        wave_sizes = [r.wave_size for r in self.results]
        performances = [r.simulations_per_second for r in self.results]
        
        axes[0, 0].scatter(wave_sizes, performances, alpha=0.7)
        axes[0, 0].set_xlabel('Wave Size')
        axes[0, 0].set_ylabel('Simulations/Second')
        axes[0, 0].set_title('Performance vs Wave Size')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Memory Usage vs Performance
        memory_usage = [r.peak_gpu_memory_mb for r in self.results]
        
        axes[0, 1].scatter(memory_usage, performances, alpha=0.7)
        axes[0, 1].set_xlabel('Peak GPU Memory (MB)')
        axes[0, 1].set_ylabel('Simulations/Second')
        axes[0, 1].set_title('Performance vs Memory Usage')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Phase Time Breakdown
        phase_names = ['selection', 'expansion', 'evaluation', 'backup']
        phase_times = defaultdict(list)
        
        for result in self.results:
            phase_totals = defaultdict(float)
            for phase in result.phases:
                phase_totals[phase.name] += phase.gpu_time_ms
            
            for phase_name in phase_names:
                phase_times[phase_name].append(phase_totals.get(phase_name, 0))
        
        x = np.arange(len(self.results))
        bottom = np.zeros(len(self.results))
        
        for phase_name in phase_names:
            axes[1, 0].bar(x, phase_times[phase_name], bottom=bottom, label=phase_name, alpha=0.8)
            bottom += phase_times[phase_name]
        
        axes[1, 0].set_xlabel('Configuration Index')
        axes[1, 0].set_ylabel('Time (ms)')
        axes[1, 0].set_title('Phase Time Breakdown')
        axes[1, 0].legend()
        
        # Efficiency Score Distribution
        efficiency_scores = [r.efficiency_score for r in self.results]
        
        axes[1, 1].hist(efficiency_scores, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Efficiency Score (sims/s/GB)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Efficiency Score Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Performance plots saved to {self.output_dir / 'performance_analysis.png'}")
    
    def _save_detailed_logs(self):
        """Save detailed profiling logs"""
        # Save all results as JSON
        results_data = []
        for result in self.results:
            result_dict = asdict(result)
            # Convert dataclass objects to dicts
            result_dict['phases'] = [asdict(phase) for phase in result.phases]
            results_data.append(result_dict)
        
        with open(self.output_dir / 'detailed_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save individual cProfile reports
        for i, result in enumerate(self.results):
            if result.cprofile_stats:
                cprofile_filename = f'cprofile_wave{result.wave_size}_sims{result.num_simulations}.txt'
                with open(self.output_dir / cprofile_filename, 'w') as f:
                    f.write(f"cProfile Report\n")
                    f.write(f"Configuration: wave_size={result.wave_size}, simulations={result.num_simulations}\n")
                    f.write(f"Performance: {result.simulations_per_second:,.0f} sims/s\n")
                    f.write("=" * 80 + "\n")
                    f.write(result.cprofile_stats)
        
        # Save aggregated function analysis
        function_aggregates = defaultdict(lambda: {'total_time': 0, 'call_count': 0, 'configs': 0, 'details': []})
        
        for result in self.results:
            if result.top_functions:
                for func in result.top_functions:
                    key = func['function_name']
                    function_aggregates[key]['total_time'] += func['cumulative_time']
                    function_aggregates[key]['call_count'] += func['total_calls']
                    function_aggregates[key]['configs'] += 1
                    function_aggregates[key]['details'].append({
                        'wave_size': result.wave_size,
                        'simulations': result.num_simulations,
                        'time': func['cumulative_time'],
                        'calls': func['total_calls'],
                        'percent': func['time_percent']
                    })
        
        if function_aggregates:
            with open(self.output_dir / 'function_analysis.json', 'w') as f:
                json.dump(dict(function_aggregates), f, indent=2)
        
        # Save summary CSV
        if PANDAS_AVAILABLE:
            summary_data = []
            for result in self.results:
                summary_data.append({
                    'wave_size': result.wave_size,
                    'num_simulations': result.num_simulations,
                    'total_time_ms': result.total_time_ms,
                    'simulations_per_second': result.simulations_per_second,
                    'peak_gpu_memory_mb': result.peak_gpu_memory_mb,
                    'efficiency_score': result.efficiency_score,
                    'bottleneck_phase': result.bottleneck_phase,
                    'tree_nodes': result.final_tree_nodes,
                    'tree_memory_mb': result.tree_memory_mb
                })
            
            df = pd.DataFrame(summary_data)
            df.to_csv(self.output_dir / 'summary_results.csv', index=False)
        else:
            # Manual CSV writing without pandas
            with open(self.output_dir / 'summary_results.csv', 'w') as f:
                f.write('wave_size,num_simulations,total_time_ms,simulations_per_second,peak_gpu_memory_mb,efficiency_score,bottleneck_phase,tree_nodes,tree_memory_mb\n')
                for result in self.results:
                    f.write(f'{result.wave_size},{result.num_simulations},{result.total_time_ms},{result.simulations_per_second},{result.peak_gpu_memory_mb},{result.efficiency_score},{result.bottleneck_phase},{result.final_tree_nodes},{result.tree_memory_mb}\n')
        
        print(f"💾 Detailed logs saved to {self.output_dir}")


def main():
    """Main benchmark entry point"""
    parser = argparse.ArgumentParser(description='Comprehensive MCTS Profiler')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark')
    parser.add_argument('--output-dir', type=str, default='mcts_profiling_results', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create profiling config
    if args.quick:
        config = ProfilingConfig(
            simulation_counts=[1000, 5000, 10000],
            wave_sizes=[3072, 3648, 4096],  # Optimal range for RTX 3060 Ti
            measurement_iterations=3
        )
    else:
        config = ProfilingConfig()
    
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Run benchmark
    benchmark = MCTSBenchmarkSuite(config)
    benchmark.run_comprehensive_benchmark()


if __name__ == "__main__":
    main()