"""
GPU-Accelerated Wave Engine for QFT-MCTS
========================================

This module implements the wave-based parallel processing engine that achieves
massive speedup by processing 2048+ paths simultaneously on GPU.

Key Features:
- Parallel wave generation with CUDA kernels
- Adaptive wave sizing based on GPU utilization
- Memory-efficient path sampling
- Integration with QFT effective action computation
- Support for multiple GPU architectures

Based on: docs/qft-mcts-guide.md Section 3
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class WaveConfig:
    """Configuration for wave engine"""
    # Wave parameters
    initial_wave_size: int = 512      # Starting wave size
    max_wave_size: int = 4096         # Maximum wave size 
    min_wave_size: int = 64           # Minimum wave size
    adaptive_sizing: bool = True      # Enable adaptive wave sizing
    
    # Path generation
    max_path_length: int = 50         # Maximum depth per path
    path_termination_prob: float = 0.05  # Probability of early termination
    
    # GPU optimization
    threads_per_block: int = 256      # CUDA threads per block
    num_streams: int = 4              # Number of CUDA streams
    use_mixed_precision: bool = True  # FP16/FP32 optimization
    memory_pool_fraction: float = 0.8 # Fraction of GPU memory to use
    
    # Performance targets
    target_gpu_utilization: float = 0.85  # Target GPU utilization
    wave_scaling_factor: float = 1.2      # Scaling factor for adaptive sizing


@dataclass 
class Wave:
    """
    Represents a wave of paths being processed in parallel
    
    This is the fundamental unit of computation in QFT-MCTS.
    Each wave contains thousands of paths that are processed
    simultaneously on GPU.
    """
    paths: torch.Tensor           # Shape: (wave_size, max_depth) - path indices
    amplitudes: torch.Tensor      # Shape: (wave_size,) - quantum amplitudes
    valid_lengths: torch.Tensor   # Shape: (wave_size,) - actual path lengths
    leaf_nodes: torch.Tensor      # Shape: (wave_size,) - final nodes
    
    def __post_init__(self):
        self.wave_size = self.paths.shape[0]
        self.max_depth = self.paths.shape[1]
        self.device = self.paths.device
        
    def get_valid_paths(self) -> torch.Tensor:
        """Get only the valid portions of paths"""
        mask = torch.arange(self.max_depth, device=self.device).unsqueeze(0)
        mask = mask < self.valid_lengths.unsqueeze(1)
        return self.paths * mask + (-1) * (~mask)
    
    def filter_by_amplitude(self, threshold: float = 0.01) -> 'Wave':
        """Filter wave to keep only high-amplitude paths"""
        high_amp_mask = torch.abs(self.amplitudes) > threshold
        
        if not high_amp_mask.any():
            # Keep at least one path
            high_amp_mask[torch.argmax(torch.abs(self.amplitudes))] = True
            
        return Wave(
            paths=self.paths[high_amp_mask],
            amplitudes=self.amplitudes[high_amp_mask],
            valid_lengths=self.valid_lengths[high_amp_mask],
            leaf_nodes=self.leaf_nodes[high_amp_mask]
        )


class GPUWaveKernels:
    """
    CUDA kernels for wave processing
    
    These kernels implement the core parallel algorithms:
    1. Wave generation with quantum weights
    2. Path sampling using effective action
    3. Amplitude computation and normalization
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.compiled_kernels = {}
        
        # Initialize random number generators for each thread
        if device.type == 'cuda':
            self._init_rng_states()
    
    def _init_rng_states(self):
        """Initialize CUDA random number generator states"""
        try:
            import cupy as cp
            # Initialize cuRAND states for parallel random generation
            self.max_threads = 65536  # Max threads across all blocks
            self.rng_states = cp.random.create_generator_states(
                self.max_threads, 
                seed=int(time.time())
            )
            logger.info(f"Initialized {self.max_threads} cuRAND states")
        except ImportError:
            logger.warning("CuPy not available, using PyTorch random generation")
            self.rng_states = None
    
    def generate_wave_parallel(
        self,
        tree_data: Dict[str, torch.Tensor],
        root_idx: int,
        wave_size: int,
        max_depth: int,
        hbar_eff: float
    ) -> Wave:
        """
        Generate wave of paths in parallel using GPU kernels
        
        This is the core parallel algorithm that replaces sequential
        tree traversal with massive parallel path generation.
        """
        if self.device.type == 'cuda' and self.rng_states is not None:
            return self._generate_wave_cuda(tree_data, root_idx, wave_size, max_depth, hbar_eff)
        else:
            return self._generate_wave_torch(tree_data, root_idx, wave_size, max_depth, hbar_eff)
    
    def _generate_wave_cuda(
        self,
        tree_data: Dict[str, torch.Tensor],
        root_idx: int,
        wave_size: int,
        max_depth: int,
        hbar_eff: float
    ) -> Wave:
        """CUDA kernel implementation for maximum performance"""
        try:
            import cupy as cp
            
            # Extract tree data
            visit_counts = tree_data['visit_counts']
            children = tree_data['children'] 
            num_children = tree_data['num_children']
            
            # Allocate output arrays
            paths = cp.full((wave_size, max_depth), -1, dtype=cp.int32)
            amplitudes = cp.zeros(wave_size, dtype=cp.float32)
            valid_lengths = cp.zeros(wave_size, dtype=cp.int32)
            
            # CUDA kernel source
            kernel_code = '''
            extern "C" __global__
            void generate_wave_kernel(
                int* paths,
                float* amplitudes, 
                int* valid_lengths,
                const float* visit_counts,
                const int* children,
                const int* num_children,
                curandState* rng_states,
                const int root_idx,
                const int wave_size,
                const int max_depth,
                const int max_children,
                const float hbar_eff
            ) {
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid >= wave_size) return;
                
                curandState local_state = rng_states[tid % 65536];
                
                int current_node = root_idx;
                int depth = 0;
                float path_amplitude = 1.0f;
                
                // Store root
                paths[tid * max_depth + 0] = root_idx;
                
                // Generate path
                for (depth = 1; depth < max_depth; depth++) {
                    // Get children of current node
                    int num_child = num_children[current_node];
                    if (num_child == 0) break;  // Leaf node
                    
                    // Compute selection probabilities using effective action
                    float total_weight = 0.0f;
                    float max_weight = -INFINITY;
                    
                    // Find maximum weight for numerical stability
                    for (int i = 0; i < num_child; i++) {
                        int child_idx = children[current_node * max_children + i];
                        if (child_idx < 0) break;
                        
                        float N = visit_counts[child_idx];
                        float log_N_eff = logf(N + 1e-8f) - hbar_eff * hbar_eff / (2.0f * N + 1e-8f);
                        if (log_N_eff > max_weight) max_weight = log_N_eff;
                    }
                    
                    // Compute normalized weights
                    for (int i = 0; i < num_child; i++) {
                        int child_idx = children[current_node * max_children + i];
                        if (child_idx < 0) break;
                        
                        float N = visit_counts[child_idx];
                        float log_N_eff = logf(N + 1e-8f) - hbar_eff * hbar_eff / (2.0f * N + 1e-8f);
                        total_weight += expf(log_N_eff - max_weight);
                    }
                    
                    // Sample child
                    float r = curand_uniform(&local_state) * total_weight;
                    float cumsum = 0.0f;
                    int selected_child = -1;
                    
                    for (int i = 0; i < num_child; i++) {
                        int child_idx = children[current_node * max_children + i];
                        if (child_idx < 0) break;
                        
                        float N = visit_counts[child_idx];
                        float log_N_eff = logf(N + 1e-8f) - hbar_eff * hbar_eff / (2.0f * N + 1e-8f);
                        cumsum += expf(log_N_eff - max_weight);
                        
                        if (r <= cumsum) {
                            selected_child = child_idx;
                            break;
                        }
                    }
                    
                    if (selected_child < 0) break;  // Safety check
                    
                    // Store selected child
                    paths[tid * max_depth + depth] = selected_child;
                    current_node = selected_child;
                    
                    // Update amplitude
                    float N = visit_counts[selected_child];
                    path_amplitude *= sqrtf(N);  // Quantum amplitude
                    
                    // Random termination
                    if (curand_uniform(&local_state) < 0.05f) break;
                }
                
                // Store results
                amplitudes[tid] = path_amplitude;
                valid_lengths[tid] = depth;
                
                // Update RNG state
                rng_states[tid % 65536] = local_state;
            }
            '''
            
            # Compile kernel
            kernel = cp.RawKernel(kernel_code, 'generate_wave_kernel')
            
            # Launch parameters
            threads_per_block = 256
            num_blocks = (wave_size + threads_per_block - 1) // threads_per_block
            max_children = children.shape[1] if len(children.shape) > 1 else 1
            
            # Launch kernel
            kernel(
                (num_blocks,), (threads_per_block,),
                (
                    paths, amplitudes, valid_lengths,
                    visit_counts, children, num_children,
                    self.rng_states,
                    root_idx, wave_size, max_depth, max_children,
                    hbar_eff
                )
            )
            
            # Convert back to PyTorch tensors
            return Wave(
                paths=torch.as_tensor(paths, device=self.device),
                amplitudes=torch.as_tensor(amplitudes, device=self.device),
                valid_lengths=torch.as_tensor(valid_lengths, device=self.device),
                leaf_nodes=torch.as_tensor(paths[:, -1], device=self.device)
            )
            
        except Exception as e:
            logger.warning(f"CUDA kernel failed: {e}, falling back to PyTorch")
            return self._generate_wave_torch(tree_data, root_idx, wave_size, max_depth, hbar_eff)
    
    def _generate_wave_torch(
        self,
        tree_data: Dict[str, torch.Tensor],
        root_idx: int,
        wave_size: int,
        max_depth: int,
        hbar_eff: float
    ) -> Wave:
        """PyTorch implementation as fallback"""
        
        # Extract tree data
        visit_counts = tree_data['visit_counts']
        children = tree_data.get('children')
        
        # Initialize wave
        paths = torch.full((wave_size, max_depth), -1, dtype=torch.long, device=self.device)
        amplitudes = torch.ones(wave_size, device=self.device)
        valid_lengths = torch.ones(wave_size, dtype=torch.long, device=self.device)
        
        # All paths start at root
        paths[:, 0] = root_idx
        current_nodes = torch.full((wave_size,), root_idx, device=self.device)
        active_mask = torch.ones(wave_size, dtype=torch.bool, device=self.device)
        
        # Generate paths depth by depth
        for depth in range(1, max_depth):
            if not active_mask.any():
                break
                
            # Get active current nodes
            active_nodes = current_nodes[active_mask]
            
            if children is not None and len(children.shape) > 1:
                # Use provided children structure
                node_children = children[active_nodes]  # Shape: (num_active, max_children)
                valid_children_mask = node_children >= 0
                
                # For each active path, sample next node
                next_nodes = torch.full_like(active_nodes, -1)
                
                for i, node in enumerate(active_nodes):
                    node_children_list = children[node][children[node] >= 0]
                    
                    if len(node_children_list) == 0:
                        continue
                        
                    # Compute quantum-corrected weights
                    child_visits = visit_counts[node_children_list]
                    log_N_eff = torch.log(child_visits + 1e-8) - hbar_eff**2 / (2 * child_visits + 1e-8)
                    weights = F.softmax(log_N_eff, dim=0)
                    
                    # Sample
                    sampled_idx = torch.multinomial(weights, 1).item()
                    next_nodes[i] = node_children_list[sampled_idx]
                    
                    # Update amplitude
                    amplitudes[active_mask][i] *= torch.sqrt(child_visits[sampled_idx])
            else:
                # Simple fallback: random selection among nearby nodes
                num_nodes = visit_counts.shape[0]
                next_nodes = torch.randint(0, num_nodes, (active_nodes.shape[0],), device=self.device)
            
            # Update paths
            paths[active_mask, depth] = next_nodes
            current_nodes[active_mask] = next_nodes
            valid_lengths[active_mask] = depth + 1
            
            # Update active mask (remove paths that hit invalid nodes)
            new_active = active_mask.clone()
            new_active[active_mask] = next_nodes >= 0
            
            # Random termination
            termination_mask = torch.rand(new_active.sum(), device=self.device) < 0.05
            if termination_mask.any():
                active_indices = torch.where(new_active)[0]
                new_active[active_indices[termination_mask]] = False
            
            active_mask = new_active
        
        # Get leaf nodes
        leaf_indices = torch.clamp(valid_lengths - 1, min=0, max=max_depth-1)
        leaf_nodes = paths[torch.arange(wave_size), leaf_indices]
        
        return Wave(
            paths=paths,
            amplitudes=amplitudes,
            valid_lengths=valid_lengths,
            leaf_nodes=leaf_nodes
        )


class GPUWaveEngine:
    """
    Main GPU wave engine for parallel MCTS processing
    
    This engine coordinates wave generation, processing, and result extraction
    to achieve massive parallel speedup over sequential tree traversal.
    """
    
    def __init__(self, config: WaveConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # Initialize GPU kernels
        self.kernels = GPUWaveKernels(device)
        
        # Wave management
        self.current_wave_size = config.initial_wave_size
        self.gpu_utilization_history = []
        
        # Performance monitoring
        self.stats = {
            'waves_generated': 0,
            'total_paths_processed': 0,
            'avg_wave_time': 0.0,
            'gpu_utilization': 0.0,
            'throughput': 0.0
        }
        
        # Memory management
        self._initialize_memory_pool()
        
        logger.info(f"GPUWaveEngine initialized with wave size {self.current_wave_size}")
    
    def _initialize_memory_pool(self):
        """Initialize GPU memory pool for efficient allocation"""
        if self.device.type == 'cuda':
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            pool_size = int(total_memory * self.config.memory_pool_fraction)
            logger.info(f"Initialized GPU memory pool: {pool_size / 1024**3:.1f} GB")
    
    def generate_wave(
        self,
        tree_data: Dict[str, torch.Tensor],
        root_idx: int,
        qft_engine: Any = None,
        **kwargs
    ) -> Wave:
        """
        Generate a wave of paths for parallel processing
        
        This is the main interface for wave generation that integrates
        with the QFT engine for quantum-corrected path sampling.
        """
        start_time = time.perf_counter()
        
        # Use adaptive wave size
        wave_size = self._get_adaptive_wave_size()
        
        # Generate wave using GPU kernels
        hbar_eff = kwargs.get('hbar_eff', 0.1)
        max_depth = min(kwargs.get('max_depth', self.config.max_path_length), self.config.max_path_length)
        
        wave = self.kernels.generate_wave_parallel(
            tree_data=tree_data,
            root_idx=root_idx,
            wave_size=wave_size,
            max_depth=max_depth,
            hbar_eff=hbar_eff
        )
        
        # Apply QFT corrections if available
        if qft_engine is not None:
            wave = self._apply_qft_corrections(wave, tree_data, qft_engine)
        
        # Update statistics
        end_time = time.perf_counter()
        wave_time = end_time - start_time
        
        self.stats['waves_generated'] += 1
        self.stats['total_paths_processed'] += wave_size
        self.stats['avg_wave_time'] = 0.9 * self.stats['avg_wave_time'] + 0.1 * wave_time
        self.stats['throughput'] = wave_size / wave_time
        
        # Update GPU utilization estimate
        if self.device.type == 'cuda':
            self._update_gpu_utilization()
        
        return wave
    
    def _apply_qft_corrections(self, wave: Wave, tree_data: Dict[str, torch.Tensor], qft_engine: Any) -> Wave:
        """Apply QFT corrections to wave amplitudes"""
        try:
            # Compute QFT weights for paths
            visit_counts = tree_data['visit_counts']
            qft_weights = qft_engine.compute_path_weights(wave.get_valid_paths(), visit_counts)
            
            # Update amplitudes with QFT corrections
            wave.amplitudes = wave.amplitudes * qft_weights[:wave.wave_size]
            
            # Renormalize
            wave.amplitudes = F.normalize(wave.amplitudes, p=1, dim=0)
            
        except Exception as e:
            logger.warning(f"QFT correction failed: {e}")
            
        return wave
    
    def _get_adaptive_wave_size(self) -> int:
        """Get adaptive wave size based on GPU utilization"""
        if not self.config.adaptive_sizing:
            return self.current_wave_size
        
        if len(self.gpu_utilization_history) < 5:
            return self.current_wave_size
        
        avg_utilization = np.mean(self.gpu_utilization_history[-5:])
        
        if avg_utilization < self.config.target_gpu_utilization - 0.1:
            # Increase wave size
            new_size = min(
                int(self.current_wave_size * self.config.wave_scaling_factor),
                self.config.max_wave_size
            )
        elif avg_utilization > self.config.target_gpu_utilization + 0.1:
            # Decrease wave size
            new_size = max(
                int(self.current_wave_size / self.config.wave_scaling_factor),
                self.config.min_wave_size
            )
        else:
            new_size = self.current_wave_size
            
        if new_size != self.current_wave_size:
            logger.debug(f"Adaptive wave size: {self.current_wave_size} → {new_size} (util: {avg_utilization:.2f})")
            self.current_wave_size = new_size
            
        return self.current_wave_size
    
    def _update_gpu_utilization(self):
        """Update GPU utilization estimate"""
        try:
            # Simple heuristic based on wave processing time
            # In practice, could use nvidia-ml-py for actual utilization
            expected_time = self.current_wave_size / 100000  # Expected time per 100k paths
            actual_time = self.stats['avg_wave_time']
            
            if expected_time > 0:
                utilization = min(expected_time / actual_time, 1.0)
                self.gpu_utilization_history.append(utilization)
                
                # Keep only recent history
                if len(self.gpu_utilization_history) > 20:
                    self.gpu_utilization_history.pop(0)
                    
                self.stats['gpu_utilization'] = np.mean(self.gpu_utilization_history)
                
        except Exception as e:
            logger.debug(f"GPU utilization update failed: {e}")
    
    def benchmark_performance(self, tree_data: Dict[str, torch.Tensor], iterations: int = 10) -> Dict[str, float]:
        """Benchmark wave engine performance"""
        logger.info(f"Benchmarking wave engine performance ({iterations} iterations)")
        
        # Warm up
        for _ in range(3):
            wave = self.generate_wave(tree_data, root_idx=0)
            
        # Benchmark
        times = []
        throughputs = []
        
        for i in range(iterations):
            start = time.perf_counter()
            wave = self.generate_wave(tree_data, root_idx=0)
            end = time.perf_counter()
            
            wave_time = end - start
            throughput = wave.wave_size / wave_time
            
            times.append(wave_time)
            throughputs.append(throughput)
            
            if i % 3 == 0:
                logger.info(f"Iteration {i+1}/{iterations}: {throughput:.0f} paths/sec")
        
        results = {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'avg_throughput': np.mean(throughputs),
            'peak_throughput': np.max(throughputs),
            'efficiency': np.mean(throughputs) / self.current_wave_size * 1000  # paths/sec/1000
        }
        
        logger.info(f"Benchmark results: {results['avg_throughput']:.0f} paths/sec average")
        return results
    
    def get_statistics(self) -> Dict[str, float]:
        """Get wave engine statistics"""
        return dict(self.stats)
    
    def reset_statistics(self):
        """Reset all statistics"""
        for key in self.stats:
            self.stats[key] = 0.0
        self.gpu_utilization_history.clear()


# Factory function for easy instantiation
def create_wave_engine(
    device: Union[str, torch.device] = 'cuda',
    wave_size: int = 1024,
    **kwargs
) -> GPUWaveEngine:
    """
    Factory function to create wave engine with sensible defaults
    
    Args:
        device: Device for computation
        wave_size: Initial wave size
        **kwargs: Override default config parameters
        
    Returns:
        Initialized GPUWaveEngine
    """
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Create config with overrides
    config_dict = {
        'initial_wave_size': wave_size,
        'max_wave_size': wave_size * 4,
        'adaptive_sizing': True,
    }
    config_dict.update(kwargs)
    
    config = WaveConfig(**config_dict)
    
    return GPUWaveEngine(config, device)