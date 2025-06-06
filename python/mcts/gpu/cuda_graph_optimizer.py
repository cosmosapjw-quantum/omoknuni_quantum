"""CUDA Graph Capture for Kernel Optimization

This module implements CUDA graph capture to reduce kernel launch overhead
and improve GPU performance for the vectorized MCTS.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable, Any
import logging
from dataclasses import dataclass
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class CUDAGraphConfig:
    """Configuration for CUDA graph capture"""
    enable_graphs: bool = True
    warmup_iterations: int = 3
    capture_iterations: int = 1
    pool_size: int = 10  # Number of graphs to cache
    profile_kernels: bool = False
    verbose: bool = False


class CUDAGraphOptimizer:
    """Optimizer for capturing and replaying CUDA graphs
    
    CUDA graphs capture sequences of GPU operations and replay them
    with minimal CPU overhead, significantly improving performance
    for repetitive workloads like MCTS.
    """
    
    def __init__(
        self,
        device: torch.device,
        config: Optional[CUDAGraphConfig] = None
    ):
        """Initialize CUDA graph optimizer
        
        Args:
            device: PyTorch CUDA device
            config: CUDA graph configuration
        """
        self.device = device
        self.config = config or CUDAGraphConfig()
        
        # Check CUDA availability
        if not torch.cuda.is_available() or device.type != 'cuda':
            logger.warning("CUDA graphs require CUDA device")
            self.config.enable_graphs = False
            
        # Graph storage
        self.graphs = {}  # name -> (graph, static_inputs, static_outputs)
        self.graph_pools = {}  # name -> list of graphs for different sizes
        
        # Statistics
        self.stats = {
            'graphs_captured': 0,
            'graph_replays': 0,
            'capture_time': 0.0,
            'replay_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Profiling
        if self.config.profile_kernels:
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True
            )
        else:
            self.profiler = None
    
    def capture_graph(
        self,
        func: Callable,
        args: Tuple,
        kwargs: Dict,
        name: str,
        static_input_indices: Optional[List[int]] = None,
        static_output_indices: Optional[List[int]] = None
    ) -> Any:
        """Capture a CUDA graph for a function
        
        Args:
            func: Function to capture
            args: Function arguments
            kwargs: Function keyword arguments
            name: Name for the graph
            static_input_indices: Indices of static (non-changing) inputs
            static_output_indices: Indices of static outputs
            
        Returns:
            Function output (from warmup run)
        """
        if not self.config.enable_graphs:
            return func(*args, **kwargs)
            
        # Generate cache key based on tensor shapes
        cache_key = self._generate_cache_key(name, args)
        
        # Check if graph exists
        if cache_key in self.graphs:
            self.stats['cache_hits'] += 1
            return self._replay_graph(cache_key, args, kwargs)
        
        self.stats['cache_misses'] += 1
        
        # Warmup iterations
        if self.config.verbose:
            logger.info(f"Warming up for graph capture: {name}")
            
        for _ in range(self.config.warmup_iterations):
            output = func(*args, **kwargs)
            
        # Synchronize before capture
        torch.cuda.synchronize()
        
        # Prepare static tensors
        static_inputs, static_outputs = self._prepare_static_tensors(
            args, output, static_input_indices, static_output_indices
        )
        
        # Capture graph
        graph = torch.cuda.CUDAGraph()
        
        with torch.cuda.graph(graph):
            # Use static inputs during capture
            capture_args = self._replace_with_static(args, static_inputs, static_input_indices)
            capture_output = func(*capture_args, **kwargs)
            
        # Store graph
        self.graphs[cache_key] = (graph, static_inputs, static_outputs, capture_output)
        self.stats['graphs_captured'] += 1
        
        if self.config.verbose:
            logger.info(f"Captured CUDA graph: {name}")
            
        return output
    
    def _replay_graph(
        self,
        cache_key: str,
        args: Tuple,
        kwargs: Dict
    ) -> Any:
        """Replay a captured CUDA graph
        
        Args:
            cache_key: Cache key for the graph
            args: Current function arguments
            kwargs: Current function keyword arguments
            
        Returns:
            Function output
        """
        graph, static_inputs, static_outputs, capture_output = self.graphs[cache_key]
        
        # Copy current inputs to static buffers
        self._copy_to_static(args, static_inputs)
        
        # Replay graph
        graph.replay()
        
        # Copy outputs from static buffers
        output = self._copy_from_static(capture_output, static_outputs)
        
        self.stats['graph_replays'] += 1
        
        return output
    
    def _generate_cache_key(self, name: str, args: Tuple) -> str:
        """Generate cache key based on tensor shapes"""
        shapes = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                shapes.append(tuple(arg.shape))
            else:
                shapes.append(type(arg).__name__)
                
        return f"{name}_{'_'.join(str(s) for s in shapes)}"
    
    def _prepare_static_tensors(
        self,
        args: Tuple,
        output: Any,
        static_input_indices: Optional[List[int]],
        static_output_indices: Optional[List[int]]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Prepare static tensors for graph capture"""
        # Static inputs
        static_inputs = []
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                if static_input_indices is None or i in static_input_indices:
                    static_inputs.append(arg.clone())
                else:
                    static_inputs.append(None)
            else:
                static_inputs.append(None)
                
        # Static outputs
        static_outputs = []
        if isinstance(output, torch.Tensor):
            static_outputs.append(output.clone())
        elif isinstance(output, (list, tuple)):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    if static_output_indices is None or i in static_output_indices:
                        static_outputs.append(out.clone())
                    else:
                        static_outputs.append(None)
                else:
                    static_outputs.append(None)
                    
        return static_inputs, static_outputs
    
    def _replace_with_static(
        self,
        args: Tuple,
        static_tensors: List[torch.Tensor],
        indices: Optional[List[int]]
    ) -> Tuple:
        """Replace arguments with static tensors"""
        new_args = []
        for i, (arg, static) in enumerate(zip(args, static_tensors)):
            if static is not None and (indices is None or i in indices):
                new_args.append(static)
            else:
                new_args.append(arg)
        return tuple(new_args)
    
    def _copy_to_static(self, args: Tuple, static_tensors: List[torch.Tensor]):
        """Copy current inputs to static buffers"""
        for arg, static in zip(args, static_tensors):
            if static is not None and isinstance(arg, torch.Tensor):
                static.copy_(arg)
                
    def _copy_from_static(self, output: Any, static_outputs: List[torch.Tensor]) -> Any:
        """Copy outputs from static buffers"""
        if isinstance(output, torch.Tensor) and static_outputs:
            return static_outputs[0].clone()
        elif isinstance(output, (list, tuple)) and static_outputs:
            results = []
            static_idx = 0
            for out in output:
                if isinstance(out, torch.Tensor) and static_idx < len(static_outputs):
                    if static_outputs[static_idx] is not None:
                        results.append(static_outputs[static_idx].clone())
                        static_idx += 1
                    else:
                        results.append(out)
                else:
                    results.append(out)
            return type(output)(results)
        else:
            return output
    
    @staticmethod
    def graph_capture(
        name: str,
        static_input_indices: Optional[List[int]] = None,
        static_output_indices: Optional[List[int]] = None
    ):
        """Decorator for automatic graph capture
        
        Usage:
            @CUDAGraphOptimizer.graph_capture("my_kernel")
            def my_kernel(x, y):
                return x + y
        """
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                if hasattr(self, 'cuda_graph_optimizer') and self.cuda_graph_optimizer is not None:
                    # Create a bound method that includes self
                    bound_method = lambda *a, **kw: func(self, *a, **kw)
                    return self.cuda_graph_optimizer.capture_graph(
                        bound_method, args, kwargs, name,
                        static_input_indices, static_output_indices
                    )
                else:
                    return func(self, *args, **kwargs)
            return wrapper
        return decorator
    
    def optimize_wave_engine_kernels(self, wave_engine):
        """Optimize wave engine kernels with CUDA graphs
        
        Args:
            wave_engine: OptimizedWaveEngine instance
        """
        if not self.config.enable_graphs:
            return
            
        # Attach optimizer to wave engine
        wave_engine.cuda_graph_optimizer = self
        
        # List of methods to optimize
        methods_to_optimize = [
            ('_compute_ucb_vectorized', None, None),
            ('_parallel_ucb_selection', None, None),
            ('_batch_evaluate_states', None, None),
            ('_vectorized_backup', None, None),
        ]
        
        # Wrap methods for graph capture
        for method_name, input_indices, output_indices in methods_to_optimize:
            if hasattr(wave_engine, method_name):
                original_method = getattr(wave_engine, method_name)
                
                @wraps(original_method)
                def wrapped_method(*args, **kwargs):
                    return self.capture_graph(
                        original_method, args, kwargs,
                        method_name, input_indices, output_indices
                    )
                
                setattr(wave_engine, method_name, wrapped_method)
                
        logger.info(f"Optimized {len(methods_to_optimize)} wave engine kernels with CUDA graphs")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get CUDA graph statistics"""
        stats = dict(self.stats)
        stats['num_graphs'] = len(self.graphs)
        stats['cache_hit_rate'] = (
            self.stats['cache_hits'] / 
            (self.stats['cache_hits'] + self.stats['cache_misses'])
            if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0
            else 0.0
        )
        return stats
    
    def clear_graphs(self):
        """Clear all captured graphs"""
        self.graphs.clear()
        self.graph_pools.clear()
        self.stats['graphs_captured'] = 0
        logger.info("Cleared all CUDA graphs")


def create_optimized_wave_engine(wave_engine, device: torch.device):
    """Create wave engine with CUDA graph optimization
    
    Args:
        wave_engine: Original wave engine
        device: CUDA device
        
    Returns:
        Optimized wave engine
    """
    if device.type != 'cuda':
        logger.warning("CUDA graphs require CUDA device")
        return wave_engine
        
    # Create optimizer
    optimizer = CUDAGraphOptimizer(device)
    
    # Optimize wave engine
    optimizer.optimize_wave_engine_kernels(wave_engine)
    
    return wave_engine