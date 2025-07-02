"""Hardware Detection and Optimization System for MCTS
This module provides automatic hardware detection and adaptive resource allocation
specifically optimized for MCTS data collection and GPU evaluation services.
"""

import os
import platform
import psutil
import multiprocessing as mp
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import math
try:
    import subprocess
except ImportError:
    subprocess = None

logger = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    """Complete hardware profile for a system"""
    # CPU
    cpu_model: str
    cpu_cores_physical: int
    cpu_threads: int
    cpu_frequency_ghz: float
    cpu_cache_mb: float
    
    # Memory
    total_ram_gb: float
    available_ram_gb: float
    ram_speed_mhz: Optional[int]
    
    # GPU
    gpu_available: bool
    gpu_model: str
    gpu_memory_mb: int
    gpu_cuda_cores: int
    gpu_compute_capability: Tuple[int, int]
    gpu_memory_bandwidth_gb: float
    
    # System
    os_name: str
    python_version: str
    numa_nodes: int
    
    # Performance metrics
    estimated_cpu_performance_score: float
    estimated_gpu_performance_score: float
    estimated_memory_bandwidth_score: float


@dataclass
class OptimalResourceAllocation:
    """Optimal resource allocation for MCTS workloads"""
    # Data collection
    data_collector_chunk_size: int
    data_collector_workers: int
    data_collector_max_concurrent: int
    data_collector_batch_timeout: float
    
    # GPU evaluator service
    gpu_batch_size: int
    gpu_batch_timeout: float
    gpu_queue_size: int
    gpu_memory_fraction: float
    
    # MCTS configuration
    mcts_wave_size: int
    mcts_max_tree_nodes: int
    mcts_memory_pool_mb: int
    mcts_simulations_per_move: int
    
    # Memory allocation
    memory_per_worker_mb: int
    gpu_memory_per_worker_mb: int
    
    # Concurrency
    optimal_thread_pool_size: int
    io_bound_workers: int
    cpu_bound_workers: int


class HardwareOptimizer:
    """Automatic hardware detection and optimization for MCTS"""
    
    def __init__(self):
        self.hardware_profile = None
        self.resource_allocation = None
        
    def detect_hardware(self) -> HardwareProfile:
        """Comprehensively detect system hardware"""
        # CPU Detection
        cpu_info = self._detect_cpu_info()
        
        # Memory Detection
        mem_info = self._detect_memory_info()
        
        # GPU Detection
        gpu_info = self._detect_gpu_info()
        
        # System Info
        os_name = f"{platform.system()} {platform.release()}"
        python_version = platform.python_version()
        numa_nodes = self._detect_numa_nodes()
        
        # Calculate performance scores
        cpu_score = self._calculate_cpu_score(cpu_info)
        gpu_score = self._calculate_gpu_score(gpu_info) if gpu_info['available'] else 0.0
        mem_score = self._calculate_memory_score(mem_info)
        
        self.hardware_profile = HardwareProfile(
            cpu_model=cpu_info['model'],
            cpu_cores_physical=cpu_info['cores_physical'],
            cpu_threads=cpu_info['threads'],
            cpu_frequency_ghz=cpu_info['frequency_ghz'],
            cpu_cache_mb=cpu_info['cache_mb'],
            total_ram_gb=mem_info['total_gb'],
            available_ram_gb=mem_info['available_gb'],
            ram_speed_mhz=mem_info['speed_mhz'],
            gpu_available=gpu_info['available'],
            gpu_model=gpu_info['model'],
            gpu_memory_mb=gpu_info['memory_mb'],
            gpu_cuda_cores=gpu_info['cuda_cores'],
            gpu_compute_capability=gpu_info['compute_capability'],
            gpu_memory_bandwidth_gb=gpu_info['memory_bandwidth_gb'],
            os_name=os_name,
            python_version=python_version,
            numa_nodes=numa_nodes,
            estimated_cpu_performance_score=cpu_score,
            estimated_gpu_performance_score=gpu_score,
            estimated_memory_bandwidth_score=mem_score
        )
        
        return self.hardware_profile
    
    def _detect_cpu_info(self) -> Dict[str, Any]:
        """Detect detailed CPU information"""
        cpu_count = os.cpu_count() or 1
        cpu_cores_physical = psutil.cpu_count(logical=False) or cpu_count // 2
        cpu_threads = psutil.cpu_count(logical=True) or cpu_count
        
        # Try to get CPU frequency
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                frequency_ghz = cpu_freq.max / 1000.0 if cpu_freq.max else cpu_freq.current / 1000.0
            else:
                frequency_ghz = 3.5  # Default estimate
        except:
            frequency_ghz = 3.5
        
        # Try to get CPU model and cache info
        cpu_model = "Unknown"
        cache_mb = 32.0  # Default estimate
        
        try:
            if platform.system() == "Linux" and subprocess:
                # Try lscpu
                result = subprocess.run(['lscpu'], capture_output=True, text=True)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'Model name:' in line:
                            cpu_model = line.split(':')[1].strip()
                        elif 'L3 cache:' in line:
                            cache_str = line.split(':')[1].strip()
                            # Parse cache size (e.g., "8192K" or "8M")
                            if 'M' in cache_str:
                                cache_mb = float(cache_str.replace('M', '').strip())
                            elif 'K' in cache_str:
                                cache_mb = float(cache_str.replace('K', '').strip()) / 1024
            elif platform.system() == "Darwin" and subprocess:  # macOS
                # Use sysctl
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    cpu_model = result.stdout.strip()
        except:
            pass
        
        # For Ryzen 9 5900X specifically
        if "5900X" in cpu_model or "5900x" in cpu_model:
            cache_mb = 64.0  # 64MB L3 cache
            frequency_ghz = 4.5  # Boost frequency
        
        return {
            'model': cpu_model,
            'cores_physical': cpu_cores_physical,
            'threads': cpu_threads,
            'frequency_ghz': frequency_ghz,
            'cache_mb': cache_mb
        }
    
    def _detect_memory_info(self) -> Dict[str, Any]:
        """Detect memory information"""
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        
        # Try to detect RAM speed (platform-specific)
        speed_mhz = None
        try:
            if platform.system() == "Linux" and subprocess:
                result = subprocess.run(['dmidecode', '-t', 'memory'], 
                                      capture_output=True, text=True, shell=True)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'Speed:' in line and 'MHz' in line:
                            speed_str = line.split(':')[1].strip()
                            speed_mhz = int(speed_str.split()[0])
                            break
        except:
            pass
        
        # Default RAM speed estimates based on total RAM
        if speed_mhz is None:
            if total_gb >= 64:
                speed_mhz = 3200  # DDR4-3200 common for high-end systems
            elif total_gb >= 32:
                speed_mhz = 2666  # DDR4-2666
            else:
                speed_mhz = 2400  # DDR4-2400
        
        return {
            'total_gb': total_gb,
            'available_gb': available_gb,
            'speed_mhz': speed_mhz
        }
    
    def _detect_gpu_info(self) -> Dict[str, Any]:
        """Detect GPU information"""
        gpu_info = {
            'available': False,
            'model': 'None',
            'memory_mb': 0,
            'cuda_cores': 0,
            'compute_capability': (0, 0),
            'memory_bandwidth_gb': 0.0
        }
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info['available'] = True
                gpu_props = torch.cuda.get_device_properties(0)
                gpu_info['model'] = gpu_props.name
                gpu_info['memory_mb'] = gpu_props.total_memory // (1024*1024)
                gpu_info['compute_capability'] = (gpu_props.major, gpu_props.minor)
                
                # CUDA cores estimation based on GPU model
                if "3060 Ti" in gpu_props.name:
                    gpu_info['cuda_cores'] = 4864
                    gpu_info['memory_bandwidth_gb'] = 448.0  # GB/s
                elif "3060" in gpu_props.name:
                    gpu_info['cuda_cores'] = 3584
                    gpu_info['memory_bandwidth_gb'] = 360.0
                elif "3070" in gpu_props.name:
                    gpu_info['cuda_cores'] = 5888
                    gpu_info['memory_bandwidth_gb'] = 448.0
                elif "3080" in gpu_props.name:
                    gpu_info['cuda_cores'] = 8704
                    gpu_info['memory_bandwidth_gb'] = 760.0
                elif "3090" in gpu_props.name:
                    gpu_info['cuda_cores'] = 10496
                    gpu_info['memory_bandwidth_gb'] = 936.0
                elif "4060" in gpu_props.name:
                    gpu_info['cuda_cores'] = 3072
                    gpu_info['memory_bandwidth_gb'] = 272.0
                elif "4070" in gpu_props.name:
                    gpu_info['cuda_cores'] = 5888
                    gpu_info['memory_bandwidth_gb'] = 504.0
                elif "4080" in gpu_props.name:
                    gpu_info['cuda_cores'] = 9728
                    gpu_info['memory_bandwidth_gb'] = 736.0
                elif "4090" in gpu_props.name:
                    gpu_info['cuda_cores'] = 16384
                    gpu_info['memory_bandwidth_gb'] = 1008.0
                else:
                    # Generic estimate based on memory
                    gpu_info['cuda_cores'] = gpu_info['memory_mb'] // 2
                    gpu_info['memory_bandwidth_gb'] = gpu_info['memory_mb'] / 20.0
        except ImportError:
            pass
        
        return gpu_info
    
    def _detect_numa_nodes(self) -> int:
        """Detect number of NUMA nodes"""
        try:
            if platform.system() == "Linux" and subprocess:
                result = subprocess.run(['lscpu'], capture_output=True, text=True)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'NUMA node(s):' in line:
                            return int(line.split(':')[1].strip())
        except:
            pass
        return 1
    
    def _calculate_cpu_score(self, cpu_info: Dict[str, Any]) -> float:
        """Calculate relative CPU performance score"""
        # Base score on cores * frequency * IPC estimate
        base_score = cpu_info['cores_physical'] * cpu_info['frequency_ghz']
        
        # IPC multiplier based on CPU generation
        ipc_multiplier = 1.0
        if "Ryzen 9 5" in cpu_info['model']:  # Zen 3
            ipc_multiplier = 1.2
        elif "Ryzen 7 5" in cpu_info['model']:  # Zen 3
            ipc_multiplier = 1.2
        elif "Ryzen" in cpu_info['model'] and "3" in cpu_info['model']:  # Zen 2
            ipc_multiplier = 1.0
        elif "Intel" in cpu_info['model'] and "12" in cpu_info['model']:  # Alder Lake
            ipc_multiplier = 1.3
        elif "Intel" in cpu_info['model'] and "13" in cpu_info['model']:  # Raptor Lake
            ipc_multiplier = 1.35
        
        # Cache bonus
        cache_bonus = math.log2(cpu_info['cache_mb'] / 8.0) * 0.1
        
        return base_score * ipc_multiplier * (1.0 + cache_bonus)
    
    def _calculate_gpu_score(self, gpu_info: Dict[str, Any]) -> float:
        """Calculate relative GPU performance score"""
        if not gpu_info['available']:
            return 0.0
        
        # Base score on CUDA cores and memory bandwidth
        compute_score = gpu_info['cuda_cores'] / 1000.0
        bandwidth_score = gpu_info['memory_bandwidth_gb'] / 100.0
        memory_score = math.log2(gpu_info['memory_mb'] / 1024.0) * 10.0
        
        # Compute capability bonus
        cc_major, cc_minor = gpu_info['compute_capability']
        cc_bonus = (cc_major + cc_minor / 10.0) / 8.0  # Normalize to ~1.0 for CC 8.0
        
        return (compute_score + bandwidth_score + memory_score) * cc_bonus
    
    def _calculate_memory_score(self, mem_info: Dict[str, Any]) -> float:
        """Calculate memory bandwidth score"""
        # Estimate bandwidth based on speed and channels (assume dual channel)
        bandwidth_gb = (mem_info['speed_mhz'] * 8 * 2) / 8000.0  # MB/s to GB/s
        capacity_score = math.log2(mem_info['total_gb'] / 8.0) * 10.0
        
        return bandwidth_gb + capacity_score
    
    def calculate_optimal_allocation(self, 
                                   workload_type: str = "balanced",
                                   target_workers: Optional[int] = None,
                                   simulation_count_hint: Optional[int] = None) -> OptimalResourceAllocation:
        """Calculate optimal resource allocation for detected hardware
        
        Args:
            workload_type: Type of workload - "latency", "throughput", or "balanced"
            target_workers: Override number of workers (None for auto)
            simulation_count_hint: Expected simulation count per move (None for auto)
        """
        if self.hardware_profile is None:
            self.detect_hardware()
        
        hw = self.hardware_profile
        
        # Calculate number of workers
        if target_workers is None:
            if workload_type == "latency":
                # Fewer workers for lower latency
                num_workers = min(hw.cpu_cores_physical // 2, 4)
            elif workload_type == "throughput":
                # Maximum workers for throughput, but cap to prevent resource exhaustion
                num_workers = min(hw.cpu_cores_physical + 2, 16)  # More conservative
            else:  # balanced
                num_workers = min(hw.cpu_cores_physical, 12)
        else:
            num_workers = target_workers
        
        # Data collector optimization
        # Chunk size based on CPU cache and memory latency
        cache_per_worker = hw.cpu_cache_mb / max(num_workers, 1)
        if cache_per_worker >= 8:
            base_chunk_size = 200
        elif cache_per_worker >= 4:
            base_chunk_size = 100
        else:
            base_chunk_size = 50
        
        # Adjust for memory bandwidth
        mem_bandwidth_factor = hw.estimated_memory_bandwidth_score / 50.0
        chunk_size = int(base_chunk_size * mem_bandwidth_factor)
        chunk_size = max(25, min(500, chunk_size))  # Clamp to reasonable range
        
        # Max concurrent workers based on system resources
        ram_per_worker_gb = 0.8  # Increased requirement for stability
        max_by_ram = int((hw.total_ram_gb - 6) / ram_per_worker_gb)  # More reserved RAM
        max_by_cpu = hw.cpu_threads - 4  # More conservative CPU usage
        
        if hw.gpu_available:
            # GPU can handle NN evaluation, but be more conservative with concurrency
            max_concurrent = min(num_workers, max_by_ram, max_by_cpu, 12)  # Cap at 12
            # Phase 1 Optimization: Aggressive batch timeout reduction for optimized coordinator
            # Legacy: 20ms (latency) / 100ms (throughput)
            # Optimized: 5ms (latency) / 10ms (throughput) - coordinator handles larger batches
            base_timeout = 0.005 if workload_type == "latency" else 0.01  # Optimized for coordination
            
            # Apply simulation-count-aware timeout adjustment
            if simulation_count_hint:
                if simulation_count_hint <= 200:
                    # Low simulation count - very short timeouts for responsiveness
                    timeout_multiplier = 0.5
                elif simulation_count_hint <= 500:
                    # Medium simulation count - balanced timeouts
                    timeout_multiplier = 1.0
                else:
                    # High simulation count - longer timeouts to allow batch accumulation
                    timeout_multiplier = 2.0
            else:
                timeout_multiplier = 1.0
            
            batch_timeout = base_timeout * timeout_multiplier
        else:
            # CPU-only needs more conservative settings
            max_concurrent = min(num_workers // 2, max_by_ram, max_by_cpu // 2, 6)  # Cap at 6
            batch_timeout = 0.005  # Optimized for CPU-only coordination
        
        # GPU evaluator optimization
        if hw.gpu_available:
            # Batch size based on GPU memory and compute capability
            gpu_memory_reserved = 2048  # Reserve 2GB for PyTorch/CUDA overhead
            gpu_memory_available = hw.gpu_memory_mb - gpu_memory_reserved
            
            # Estimate memory per sample (depends on network size)
            memory_per_sample = 4  # MB, rough estimate for ResNet
            max_batch_by_memory = gpu_memory_available // memory_per_sample
            
            # Adjust for compute capability
            if hw.gpu_compute_capability[0] >= 8:  # Ampere or newer
                compute_multiplier = 1.5
            elif hw.gpu_compute_capability[0] >= 7:  # Turing/Volta
                compute_multiplier = 1.2
            else:
                compute_multiplier = 1.0
            
            # Apply simulation-count-aware batch sizing
            if simulation_count_hint:
                if simulation_count_hint <= 200:
                    # Low simulation count - smaller batches for lower latency
                    batch_size_multiplier = 0.5
                elif simulation_count_hint <= 500:
                    # Medium simulation count - balanced approach
                    batch_size_multiplier = 1.0
                else:
                    # High simulation count - larger batches for throughput
                    batch_size_multiplier = 1.5
            else:
                batch_size_multiplier = 1.0
            
            if workload_type == "latency":
                gpu_batch_size = min(64, max_batch_by_memory)
            elif workload_type == "throughput":
                # Reduce batch size to minimize GPU-to-CPU conversion latency
                gpu_batch_size = min(256, int(max_batch_by_memory * compute_multiplier))
            else:
                gpu_batch_size = min(128, int(max_batch_by_memory * compute_multiplier * 0.7))
            
            # Apply simulation count adjustment
            gpu_batch_size = int(gpu_batch_size * batch_size_multiplier)
            gpu_batch_size = max(16, min(512, gpu_batch_size))  # Clamp to reasonable range
            
            # Queue size based on workers and batch size
            gpu_queue_size = max(1000, num_workers * gpu_batch_size * 2)
            gpu_memory_fraction = min(0.9, (gpu_memory_available / hw.gpu_memory_mb) * 0.95)
            gpu_memory_per_worker = 128  # MB, for worker-side tensors
        else:
            gpu_batch_size = 32
            gpu_queue_size = 1000
            gpu_memory_fraction = 0.0
            gpu_memory_per_worker = 0
        
        # MCTS optimization
        if hw.gpu_available:
            # Wave size optimization for GPU
            if hw.gpu_cuda_cores >= 4000:
                wave_size = 4096
            elif hw.gpu_cuda_cores >= 2000:
                wave_size = 2048
            else:
                wave_size = 1024
            
            # Tree size based on available GPU memory (not RAM!)
            # Use conservative fraction of GPU memory for tree storage
            tree_memory_fraction = 0.2  # Use 20% of GPU memory for tree
            tree_memory_mb = int(gpu_memory_available * tree_memory_fraction)
            # Account for max_children=400 (Go support) in memory calculation
            bytes_per_node = 400 * 4 + 60  # children tensor + other node data
            max_tree_nodes = (tree_memory_mb * 1024 * 1024) // bytes_per_node
            
            # Set reasonable caps based on use case
            # For data collection and balanced workloads, use smaller trees to allow multiple workers
            if workload_type in ['data_collection', 'balanced']:
                max_tree_nodes = min(max_tree_nodes, 50000)  # Smaller for multiple workers
            elif workload_type == 'throughput':
                max_tree_nodes = min(max_tree_nodes, 75000)  # Medium for throughput
            else:  # latency or other
                max_tree_nodes = min(max_tree_nodes, 100000)  # Larger for analysis
            
            # Memory pool for GPU operations
            memory_pool_mb = min(4096, gpu_memory_available // 2)
        else:
            wave_size = 512  # Smaller for CPU
            max_tree_nodes = 200000
            memory_pool_mb = 512
        
        # Simulations per move based on hardware performance
        cpu_factor = hw.estimated_cpu_performance_score / 50.0
        gpu_factor = hw.estimated_gpu_performance_score / 100.0 if hw.gpu_available else 0.0
        combined_factor = cpu_factor + gpu_factor * 0.5
        
        if workload_type == "latency":
            simulations = int(400 * combined_factor)
        elif workload_type == "throughput":
            simulations = int(800 * combined_factor)
        else:
            simulations = int(600 * combined_factor)
        
        simulations = max(200, min(2000, simulations))  # Reasonable bounds
        
        # Memory allocation
        memory_per_worker_mb = int((hw.total_ram_gb * 1024 - 4096) / num_workers)
        memory_per_worker_mb = min(2048, memory_per_worker_mb)  # Cap at 2GB per worker
        
        # Thread pool sizing
        if hw.cpu_cores_physical >= 8:
            thread_pool_size = hw.cpu_cores_physical
        else:
            thread_pool_size = hw.cpu_threads
        
        # IO vs CPU bound workers
        io_bound_workers = min(num_workers // 2, hw.cpu_threads // 4)
        cpu_bound_workers = num_workers - io_bound_workers
        
        self.resource_allocation = OptimalResourceAllocation(
            data_collector_chunk_size=chunk_size,
            data_collector_workers=num_workers,
            data_collector_max_concurrent=max_concurrent,
            data_collector_batch_timeout=batch_timeout,
            gpu_batch_size=gpu_batch_size,
            # Phase 1 Optimization: Separate GPU timeout for optimized batch coordinator
            gpu_batch_timeout=batch_timeout * 0.5,  # Even more aggressive for GPU service
            gpu_queue_size=gpu_queue_size * 2,  # Larger queues for batch coordination
            gpu_memory_fraction=gpu_memory_fraction,
            mcts_wave_size=wave_size,
            mcts_max_tree_nodes=max_tree_nodes,
            mcts_memory_pool_mb=memory_pool_mb,
            mcts_simulations_per_move=simulations,
            memory_per_worker_mb=memory_per_worker_mb,
            gpu_memory_per_worker_mb=gpu_memory_per_worker,
            optimal_thread_pool_size=thread_pool_size,
            io_bound_workers=io_bound_workers,
            cpu_bound_workers=cpu_bound_workers
        )
        
        return self.resource_allocation
    
    def get_optimization_report(self) -> str:
        """Generate a human-readable optimization report"""
        if self.hardware_profile is None:
            self.detect_hardware()
        if self.resource_allocation is None:
            self.calculate_optimal_allocation()
        
        hw = self.hardware_profile
        alloc = self.resource_allocation
        
        report = []
        report.append("=" * 70)
        report.append("HARDWARE OPTIMIZATION REPORT")
        report.append("=" * 70)
        
        # Hardware Summary
        report.append("\nDETECTED HARDWARE:")
        report.append(f"  CPU: {hw.cpu_model}")
        report.append(f"  - Cores: {hw.cpu_cores_physical} physical, {hw.cpu_threads} threads")
        report.append(f"  - Frequency: {hw.cpu_frequency_ghz:.1f} GHz")
        report.append(f"  - L3 Cache: {hw.cpu_cache_mb:.0f} MB")
        report.append(f"  - Performance Score: {hw.estimated_cpu_performance_score:.1f}")
        
        report.append(f"\n  Memory: {hw.total_ram_gb:.1f} GB @ {hw.ram_speed_mhz or 'Unknown'} MHz")
        report.append(f"  - Available: {hw.available_ram_gb:.1f} GB")
        report.append(f"  - Bandwidth Score: {hw.estimated_memory_bandwidth_score:.1f}")
        
        if hw.gpu_available:
            report.append(f"\n  GPU: {hw.gpu_model}")
            report.append(f"  - Memory: {hw.gpu_memory_mb} MB")
            report.append(f"  - CUDA Cores: {hw.gpu_cuda_cores}")
            report.append(f"  - Compute Capability: {hw.gpu_compute_capability[0]}.{hw.gpu_compute_capability[1]}")
            report.append(f"  - Performance Score: {hw.estimated_gpu_performance_score:.1f}")
        else:
            report.append("\n  GPU: Not available")
        
        # Optimization Summary
        report.append("\n" + "=" * 70)
        report.append("OPTIMIZED RESOURCE ALLOCATION:")
        report.append("=" * 70)
        
        report.append("\nDATA COLLECTION:")
        report.append(f"  Workers: {alloc.data_collector_workers}")
        report.append(f"  Chunk Size: {alloc.data_collector_chunk_size} games")
        report.append(f"  Max Concurrent: {alloc.data_collector_max_concurrent}")
        report.append(f"  Memory per Worker: {alloc.memory_per_worker_mb} MB")
        
        report.append("\nGPU EVALUATOR SERVICE:")
        report.append(f"  Batch Size: {alloc.gpu_batch_size}")
        report.append(f"  Batch Timeout: {alloc.gpu_batch_timeout:.3f}s")
        report.append(f"  Queue Size: {alloc.gpu_queue_size}")
        report.append(f"  GPU Memory Fraction: {alloc.gpu_memory_fraction:.2f}")
        
        report.append("\nMCTS CONFIGURATION:")
        report.append(f"  Wave Size: {alloc.mcts_wave_size}")
        report.append(f"  Max Tree Nodes: {alloc.mcts_max_tree_nodes:,}")
        report.append(f"  Memory Pool: {alloc.mcts_memory_pool_mb} MB")
        report.append(f"  Simulations/Move: {alloc.mcts_simulations_per_move}")
        
        report.append("\nCONCURRENCY:")
        report.append(f"  Thread Pool Size: {alloc.optimal_thread_pool_size}")
        report.append(f"  I/O Bound Workers: {alloc.io_bound_workers}")
        report.append(f"  CPU Bound Workers: {alloc.cpu_bound_workers}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)


# Convenience function for quick optimization
def optimize_for_hardware(workload_type: str = "balanced", 
                         target_workers: Optional[int] = None,
                         simulation_count_hint: Optional[int] = None) -> Tuple[HardwareProfile, OptimalResourceAllocation]:
    """Quick function to detect hardware and calculate optimal allocation
    
    Args:
        workload_type: "latency", "throughput", or "balanced"
        target_workers: Override number of workers (None for auto)
        simulation_count_hint: Expected simulation count per move (None for auto)
    
    Returns:
        Tuple of (hardware_profile, resource_allocation)
    """
    optimizer = HardwareOptimizer()
    hardware = optimizer.detect_hardware()
    allocation = optimizer.calculate_optimal_allocation(workload_type, target_workers, simulation_count_hint)
    
    # Log optimization report
    logger.info("Hardware optimization completed")
    logger.debug(optimizer.get_optimization_report())
    
    return hardware, allocation


if __name__ == "__main__":
    # Test the optimizer
    import sys
    
    workload = sys.argv[1] if len(sys.argv) > 1 else "balanced"
    optimize_for_hardware(workload)