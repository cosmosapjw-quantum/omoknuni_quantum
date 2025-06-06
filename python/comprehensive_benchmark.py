#!/usr/bin/env python3
"""Comprehensive MCTS Benchmark Tool

This consolidated benchmark tool combines all benchmarking functionality:
- Performance throughput measurement
- CSR tree format efficiency testing  
- Wave engine optimization benchmarking
- Hardware utilization monitoring
- Multi-configuration testing
- Statistical analysis and reporting

Usage:
    python comprehensive_benchmark.py              # Run full benchmark suite
    python comprehensive_benchmark.py --quick      # Quick test (5 iterations)
    python comprehensive_benchmark.py --csr        # Focus on CSR format testing
    python comprehensive_benchmark.py --wave       # Focus on wave engine testing
    python comprehensive_benchmark.py --profile    # Enable GPU profiling
"""

import sys
import os
import time
import json
import torch
import numpy as np
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import psutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    print("GPUtil not available - GPU monitoring disabled")

from mcts import (
    CSRTree, CSRTreeConfig,
    OptimizedWaveEngine, OptimizedWaveConfig,
    HighPerformanceMCTS, HighPerformanceMCTSConfig,
    get_csr_kernels
)


@dataclass
class BenchmarkConfig:
    """Configuration for comprehensive benchmarking"""
    # Test parameters
    test_sizes: List[str] = None  # 'small', 'medium', 'large'
    wave_sizes: List[int] = None
    num_iterations: int = 10
    warmup_iterations: int = 2
    
    # Features to test
    test_csr_format: bool = True
    test_wave_engine: bool = True
    test_full_mcts: bool = True
    test_gpu_kernels: bool = True
    
    # Hardware config
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    enable_profiling: bool = False
    
    def __post_init__(self):
        if self.test_sizes is None:
            self.test_sizes = ['small', 'medium', 'large']
        if self.wave_sizes is None:
            self.wave_sizes = [256, 512, 1024, 2048]


class ComprehensiveBenchmark:
    """Main benchmark class that runs all tests"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available(),
            'tests': {}
        }
        
        if torch.cuda.is_available():
            self.results['gpu_info'] = {
                'name': torch.cuda.get_device_name(0),
                'memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)
            }
    
    def run(self) -> Dict:
        """Run complete benchmark suite"""
        print(f"=== Comprehensive MCTS Benchmark ===")
        print(f"Device: {self.device}")
        print(f"PyTorch: {torch.__version__}")
        
        if self.config.test_csr_format:
            print("\n1. Testing CSR Tree Format...")
            self.results['tests']['csr_format'] = self._benchmark_csr_format()
        
        if self.config.test_wave_engine:
            print("\n2. Testing Wave Engine...")
            self.results['tests']['wave_engine'] = self._benchmark_wave_engine()
        
        if self.config.test_full_mcts:
            print("\n3. Testing Full MCTS...")
            self.results['tests']['full_mcts'] = self._benchmark_full_mcts()
        
        if self.config.test_gpu_kernels and torch.cuda.is_available():
            print("\n4. Testing GPU Kernels...")
            self.results['tests']['gpu_kernels'] = self._benchmark_gpu_kernels()
        
        # Summary
        self._print_summary()
        
        return self.results
    
    def _benchmark_csr_format(self) -> Dict:
        """Benchmark CSR tree format operations"""
        results = {}
        
        for size in self.config.test_sizes:
            print(f"\n  Testing {size} tree...")
            tree = self._create_test_tree(size)
            
            # Measure operations
            ops_results = {}
            
            # Batch UCB selection
            batch_sizes = [16, 64, 256, 1024]
            for batch_size in batch_sizes:
                if batch_size > tree.num_nodes:
                    continue
                    
                node_indices = torch.randint(0, tree.num_nodes, (batch_size,), 
                                           device=self.device, dtype=torch.int32)
                
                times = []
                for _ in range(self.config.num_iterations):
                    start = time.perf_counter()
                    
                    kernels = get_csr_kernels()
                    selected = kernels.batch_ucb_selection(
                        node_indices=node_indices,
                        row_ptr=tree.row_ptr,
                        col_indices=tree.col_indices,
                        edge_actions=tree.edge_actions,
                        edge_priors=tree.edge_priors,
                        visit_counts=tree.visit_counts,
                        value_sums=tree.value_sums,
                        c_puct=1.0
                    )
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    times.append(time.perf_counter() - start)
                
                avg_time = np.mean(times[self.config.warmup_iterations:])
                throughput = batch_size / avg_time
                
                ops_results[f'ucb_batch_{batch_size}'] = {
                    'avg_time_ms': avg_time * 1000,
                    'throughput_ops_per_sec': throughput
                }
            
            # Memory usage
            memory_stats = tree.get_memory_usage()
            
            results[size] = {
                'tree_stats': {
                    'nodes': tree.num_nodes,
                    'edges': tree.num_edges,
                    'memory_mb': memory_stats['total_mb'],
                    'bytes_per_node': memory_stats['bytes_per_node']
                },
                'operations': ops_results
            }
        
        return results
    
    def _benchmark_wave_engine(self) -> Dict:
        """Benchmark wave engine performance"""
        results = {}
        
        # Create medium-sized tree for wave testing
        tree = self._create_test_tree('medium')
        
        for wave_size in self.config.wave_sizes:
            print(f"\n  Testing wave size {wave_size}...")
            
            config = OptimizedWaveConfig(
                wave_size=wave_size,
                enable_memory_pooling=True,
                enable_mixed_precision=True,
                device=str(self.device)
            )
            
            engine = OptimizedWaveEngine(tree, config)
            
            # Warmup
            for _ in range(self.config.warmup_iterations):
                engine.run_wave(None)
            
            # Benchmark
            times = []
            for _ in range(self.config.num_iterations):
                start = time.perf_counter()
                result = engine.run_wave(None)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
            
            avg_time = np.mean(times)
            throughput = wave_size / avg_time
            
            # Get detailed stats
            perf_stats = engine.get_performance_stats()
            
            results[f'wave_{wave_size}'] = {
                'avg_time_ms': avg_time * 1000,
                'throughput_sims_per_sec': throughput,
                'selection_efficiency': perf_stats.get('selection_efficiency', 0),
                'breakdown': {
                    'selection_ms': result['timing']['selection'] * 1000,
                    'evaluation_ms': result['timing']['evaluation'] * 1000,
                    'backup_ms': result['timing']['backup'] * 1000
                }
            }
        
        return results
    
    def _benchmark_full_mcts(self) -> Dict:
        """Benchmark complete MCTS search"""
        results = {}
        
        # Test different configurations
        configs = [
            ('baseline', {'wave_size': 256, 'enable_interference': False}),
            ('optimized', {'wave_size': 1024, 'enable_interference': False}),
            ('large_wave', {'wave_size': 2048, 'enable_interference': False}),
        ]
        
        if HAS_GPUTIL and torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            gpu_start_util = gpu.load * 100
        
        for name, config_overrides in configs:
            print(f"\n  Testing {name} configuration...")
            
            # Create config
            config = HighPerformanceMCTSConfig(
                num_simulations=10000,
                wave_size=config_overrides.get('wave_size', 1024),
                enable_interference=config_overrides.get('enable_interference', False),
                device=str(self.device),
                enable_gpu=True,  # Force GPU usage
                mixed_precision=True  # Enable mixed precision for better GPU performance
            )
            
            # Mock components
            class MockGame:
                def clone_state(self, state):
                    return state
                def get_legal_moves(self, state):
                    return list(range(10))
                def apply_move(self, state, move):
                    return state
                def is_terminal(self, state):
                    return False
                    
            class MockEvaluator:
                def evaluate(self, states):
                    batch_size = len(states) if isinstance(states, list) else 1
                    return torch.randn(batch_size), torch.randn(batch_size, 10)
            
            # Create MCTS
            mcts = HighPerformanceMCTS(config, MockGame(), MockEvaluator())
            
            # Benchmark
            start = time.perf_counter()
            mcts.search({'board': None})
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_time = time.perf_counter() - start
            
            throughput = config.num_simulations / total_time
            
            result = {
                'total_time_sec': total_time,
                'throughput_sims_per_sec': throughput,
                'config': {
                    'wave_size': config.wave_size,
                    'num_simulations': config.num_simulations
                }
            }
            
            # GPU utilization
            if HAS_GPUTIL and torch.cuda.is_available():
                gpu = GPUtil.getGPUs()[0]
                result['gpu_utilization'] = gpu.load * 100 - gpu_start_util
            
            results[name] = result
        
        return results
    
    def _benchmark_gpu_kernels(self) -> Dict:
        """Benchmark custom GPU kernels"""
        results = {}
        
        kernels = get_csr_kernels()
        if not kernels.cuda_available:
            return {'error': 'CUDA kernels not available'}
        
        # Create test data
        tree = self._create_test_tree('large')
        batch_sizes = [256, 1024, 4096]
        
        for batch_size in batch_sizes:
            if batch_size > tree.num_nodes:
                continue
                
            print(f"\n  Testing batch size {batch_size}...")
            
            node_indices = torch.randint(0, tree.num_nodes, (batch_size,), 
                                       device=self.device, dtype=torch.int32)
            
            # Test with and without CUDA kernel
            for use_cuda in [False, True]:
                times = []
                
                for _ in range(self.config.num_iterations):
                    start = time.perf_counter()
                    
                    selected = kernels.batch_ucb_selection(
                        node_indices=node_indices,
                        row_ptr=tree.row_ptr,
                        col_indices=tree.col_indices,
                        edge_actions=tree.edge_actions,
                        edge_priors=tree.edge_priors,
                        visit_counts=tree.visit_counts,
                        value_sums=tree.value_sums,
                        c_puct=1.0,
                        use_cuda_kernel=use_cuda
                    )
                    
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - start)
                
                avg_time = np.mean(times[self.config.warmup_iterations:])
                throughput = batch_size / avg_time
                
                key = f'batch_{batch_size}_{"cuda" if use_cuda else "pytorch"}'
                results[key] = {
                    'avg_time_ms': avg_time * 1000,
                    'throughput_ops_per_sec': throughput
                }
        
        return results
    
    def _create_test_tree(self, size: str) -> CSRTree:
        """Create a test tree of specified size"""
        if size == 'small':
            max_nodes, depth = 1000, 3
        elif size == 'medium':
            max_nodes, depth = 10000, 4
        else:  # large
            max_nodes, depth = 100000, 5
        
        config = CSRTreeConfig(
            max_nodes=max_nodes,
            max_edges=max_nodes * 5,
            device=str(self.device)
        )
        
        tree = CSRTree(config)
        
        # Build tree
        queue = [tree.add_root(1.0)]
        nodes_created = 1
        
        for d in range(depth):
            if not queue or nodes_created >= max_nodes // 2:
                break
                
            next_queue = []
            for parent in queue:
                num_children = min(4, max_nodes - nodes_created)
                for action in range(num_children):
                    if nodes_created >= max_nodes // 2:
                        break
                    child = tree.add_child(parent, action, child_prior=0.25)
                    next_queue.append(child)
                    nodes_created += 1
                    
                    # Add some visits/values
                    tree.update_visit_count(child, np.random.randint(1, 50))
                    tree.update_value_sum(child, np.random.randn() * 0.3)
            
            queue = next_queue
        
        return tree
    
    def _print_summary(self):
        """Print benchmark summary"""
        print("\n=== BENCHMARK SUMMARY ===")
        
        # Find best throughput
        best_throughput = 0
        best_config = ""
        
        if 'wave_engine' in self.results['tests']:
            for config, data in self.results['tests']['wave_engine'].items():
                throughput = data['throughput_sims_per_sec']
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_config = config
        
        if 'full_mcts' in self.results['tests']:
            for config, data in self.results['tests']['full_mcts'].items():
                throughput = data['throughput_sims_per_sec']
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_config = f"mcts_{config}"
        
        print(f"Best throughput: {best_throughput:.0f} sims/sec ({best_config})")
        
        # Phase targets
        phase1_target = 10000
        phase2_target = 50000
        phase3_target = 100000
        
        if best_throughput >= phase3_target:
            print(f"✅ Phase 3 target achieved! ({best_throughput:.0f} >= {phase3_target})")
        elif best_throughput >= phase2_target:
            print(f"✅ Phase 2 target achieved! ({best_throughput:.0f} >= {phase2_target})")
        elif best_throughput >= phase1_target:
            print(f"✅ Phase 1 target achieved! ({best_throughput:.0f} >= {phase1_target})")
        else:
            gap = phase1_target / best_throughput
            print(f"❌ Phase 1 target not reached. Need {gap:.1f}x improvement")
        
        # Memory efficiency
        if 'csr_format' in self.results['tests']:
            for size, data in self.results['tests']['csr_format'].items():
                bytes_per_node = data['tree_stats']['bytes_per_node']
                print(f"{size.capitalize()} tree: {bytes_per_node:.0f} bytes/node")
    
    def save_results(self, filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        filepath = os.path.join('benchmarks', 'results', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Comprehensive MCTS Benchmark')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick test (5 iterations)')
    parser.add_argument('--csr', action='store_true',
                       help='Test only CSR format')
    parser.add_argument('--wave', action='store_true',
                       help='Test only wave engine')
    parser.add_argument('--profile', action='store_true',
                       help='Enable GPU profiling')
    parser.add_argument('--save', action='store_true',
                       help='Save results to file')
    
    args = parser.parse_args()
    
    # Configure benchmark
    config = BenchmarkConfig()
    
    if args.quick:
        config.num_iterations = 5
        config.test_sizes = ['medium']
        config.wave_sizes = [512, 1024]
    
    if args.csr:
        config.test_wave_engine = False
        config.test_full_mcts = False
    elif args.wave:
        config.test_csr_format = False
        config.test_full_mcts = False
        config.test_gpu_kernels = False
    
    config.enable_profiling = args.profile
    
    # Run benchmark
    benchmark = ComprehensiveBenchmark(config)
    results = benchmark.run()
    
    # Save results
    if args.save:
        benchmark.save_results()
    
    # Print performance grade
    print("\n" + "="*50)
    print("PERFORMANCE GRADE:", end=" ")
    
    best_throughput = 0
    for test_type, test_data in results['tests'].items():
        if isinstance(test_data, dict):
            for config_name, config_data in test_data.items():
                if 'throughput_sims_per_sec' in config_data:
                    best_throughput = max(best_throughput, config_data['throughput_sims_per_sec'])
    
    if best_throughput >= 100000:
        print("A+ (Production Elite)")
    elif best_throughput >= 50000:
        print("A (Production Ready)")
    elif best_throughput >= 10000:
        print("B (Good Performance)")
    elif best_throughput >= 1000:
        print("C (Needs Optimization)")
    else:
        print("D (Poor Performance)")


if __name__ == '__main__':
    main()