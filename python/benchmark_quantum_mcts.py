"""
Comprehensive Benchmark Suite for Quantum-Enhanced MCTS
=======================================================

This benchmark evaluates all quantum features and their impact on MCTS performance.
Designed for system with Ryzen 9 5900X (24 threads) + RTX 3060 Ti (8GB VRAM).
"""

import torch
import numpy as np
import time
import json
import psutil
import GPUtil
from datetime import datetime
from typing import Dict, List, Any, Tuple
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available, plotting disabled")
from pathlib import Path

# Import quantum components
from mcts.quantum import (
    create_quantum_mcts,
    create_quantum_parallel_evaluator,
    create_state_pool,
    create_quantum_csr_tree,
    create_wave_compressor
)

# Import GPU kernels
from mcts.gpu import create_optimized_quantum_kernels

# Import core components
from mcts.core.mcts import MCTS, MCTSConfig
from mcts.core.game_interface import GameType
from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
import alphazero_py


class QuantumMCTSBenchmark:
    """Comprehensive benchmark suite for quantum MCTS"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # System info
        self.system_info = self._get_system_info()
        
        # Results storage
        self.results = {
            'system_info': self.system_info,
            'timestamp': datetime.now().isoformat(),
            'benchmarks': {}
        }
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information"""
        info = {
            'cpu': {
                'model': 'AMD Ryzen 9 5900X',  # Hardcoded as per user spec
                'cores': psutil.cpu_count(logical=False),
                'threads': psutil.cpu_count(logical=True),
                'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None
            },
            'memory': {
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'available_gb': psutil.virtual_memory().available / (1024**3)
            }
        }
        
        if self.device.type == 'cuda':
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                info['gpu'] = {
                    'name': gpu.name,
                    'memory_total_gb': gpu.memoryTotal / 1024,
                    'memory_free_gb': gpu.memoryFree / 1024,
                    'cuda_version': torch.version.cuda
                }
        
        return info
    
    def benchmark_quantum_state_pool(self) -> Dict[str, Any]:
        """Benchmark quantum state pool performance"""
        print("\n📊 Benchmarking Quantum State Pool")
        print("-" * 50)
        
        results = {
            'name': 'Quantum State Pool',
            'configs': [],
            'metrics': {}
        }
        
        # Test different pool configurations
        pool_configs = [
            {'max_states': 10000, 'compression_threshold': 0.9},
            {'max_states': 50000, 'compression_threshold': 0.95},
            {'max_states': 100000, 'compression_threshold': 0.99},
        ]
        
        for config in pool_configs:
            print(f"\nTesting config: {config}")
            pool = create_state_pool(**config)
            
            # Benchmark allocation/deallocation
            start = time.perf_counter()
            states = []
            
            # Allocate states
            for i in range(1000):
                dim = np.random.randint(100, 1000)
                state = pool.get_state(dim)
                # Make some sparse
                if i % 3 == 0:
                    indices = torch.randperm(dim)[:20]
                    state.view(-1)[indices] = torch.randn(20, dtype=torch.complex64, device=self.device)
                states.append(state)
            
            alloc_time = time.perf_counter() - start
            
            # Return states with compression
            start = time.perf_counter()
            for state in states:
                pool.return_state(state, compress=True)
            return_time = time.perf_counter() - start
            
            stats = pool.get_statistics()
            
            config_results = {
                'config': config,
                'allocation_time': alloc_time,
                'return_time': return_time,
                'total_time': alloc_time + return_time,
                'stats': stats
            }
            
            results['configs'].append(config_results)
            
            print(f"  Allocation: {alloc_time:.3f}s")
            print(f"  Return: {return_time:.3f}s")
            print(f"  Memory saved: {stats['memory_saved_mb']:.1f} MB")
            print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
        
        return results
    
    def benchmark_wave_compression(self) -> Dict[str, Any]:
        """Benchmark wave function compression methods"""
        print("\n📊 Benchmarking Wave Function Compression")
        print("-" * 50)
        
        results = {
            'name': 'Wave Function Compression',
            'methods': {},
            'dimensions': []
        }
        
        compressor = create_wave_compressor(
            enable_mps=True,
            max_bond_dimension=256
        )
        
        # Test different state dimensions
        dimensions = [256, 512, 1024, 2048, 4096]
        methods = ['sparse', 'low_rank', 'diagonal', 'mps']
        
        for dim in dimensions:
            print(f"\nDimension: {dim}")
            dim_results = {'dimension': dim, 'methods': {}}
            
            # Create test states with different sparsity
            test_states = {
                'sparse': self._create_sparse_state(dim, sparsity=0.05),
                'medium': self._create_sparse_state(dim, sparsity=0.3),
                'dense': torch.randn(dim, dtype=torch.complex64, device=self.device)
            }
            
            for state_type, state in test_states.items():
                state = state / torch.norm(state)  # Normalize
                
                # Test compression without specifying method - let compressor choose
                try:
                    start = time.perf_counter()
                    compressed = compressor.compress(state.clone())
                    compress_time = time.perf_counter() - start
                    
                    start = time.perf_counter()
                    decompressed = compressed.decompress(self.device)
                    decompress_time = time.perf_counter() - start
                    
                    # Compute fidelity manually
                    overlap = torch.dot(state.conj(), decompressed)
                    fidelity = torch.abs(overlap).item() ** 2
                    
                    result = {
                        'compress_time': compress_time,
                        'decompress_time': decompress_time,
                        'compression_ratio': compressed.compression_ratio,
                        'fidelity': fidelity,
                        'state_type': state_type,
                        'method': compressed.method
                    }
                    
                    key = f"{compressed.method}_{state_type}"
                    dim_results['methods'][key] = result
                    
                except Exception as e:
                    print(f"    Compression ({state_type}) failed: {e}")
            
            results['dimensions'].append(dim_results)
        
        return results
    
    def benchmark_quantum_parallelism(self) -> Dict[str, Any]:
        """Benchmark quantum parallelism and Grover amplification"""
        print("\n📊 Benchmarking Quantum Parallelism")
        print("-" * 50)
        
        results = {
            'name': 'Quantum Parallelism',
            'path_counts': [],
            'grover_iterations': []
        }
        
        evaluator = create_quantum_parallel_evaluator(
            max_superposition=1024,
            use_gpu=self.device.type == 'cuda'
        )
        
        # Test different path counts
        path_counts = [100, 500, 1000, 2000, 5000]
        
        for num_paths in path_counts:
            print(f"\nTesting {num_paths} paths")
            
            # Create paths
            paths = torch.randint(0, 100, (num_paths, 30), device=self.device)
            visits = torch.randint(1, 100, (num_paths,), device=self.device).float()
            
            # Value function with 5% good paths
            def value_func(paths):
                values = torch.rand(len(paths), device=self.device) * 0.1
                good_indices = torch.randperm(len(paths))[:len(paths)//20]
                values[good_indices] = 0.8 + torch.rand(len(good_indices), device=self.device) * 0.2
                return values
            
            # Test different Grover iterations
            grover_results = []
            for grover_iter in [0, 1, 2, 3, 5]:
                evaluator.config.grover_iterations = grover_iter
                
                start = time.perf_counter()
                selected, info = evaluator.select_paths_hybrid(paths, value_func, visits)
                elapsed = time.perf_counter() - start
                
                grover_results.append({
                    'iterations': grover_iter,
                    'time': elapsed,
                    'mode': info['mode'],
                    'paths_selected': len(selected) if hasattr(selected, '__len__') else selected.shape[0]
                })
            
            results['path_counts'].append({
                'num_paths': num_paths,
                'grover_results': grover_results
            })
        
        return results
    
    def benchmark_gpu_kernels(self) -> Dict[str, Any]:
        """Benchmark optimized GPU kernels"""
        print("\n📊 Benchmarking GPU Kernels")
        print("-" * 50)
        
        if self.device.type != 'cuda':
            print("  Skipping - GPU not available")
            return {'name': 'GPU Kernels', 'skipped': True}
        
        results = {
            'name': 'GPU Kernels',
            'kernels': {}
        }
        
        kernels = create_optimized_quantum_kernels()
        
        # Test configurations
        batch_sizes = [256, 512, 1024, 2048]
        path_lengths = [20, 30, 50]
        
        for batch_size in batch_sizes:
            for path_length in path_lengths:
                config_key = f"batch_{batch_size}_len_{path_length}"
                print(f"\nTesting {config_key}")
                
                # Prepare test data
                paths = torch.randint(0, 100, (batch_size, path_length), device='cuda')
                scores = torch.rand(batch_size, device='cuda')
                priors = torch.softmax(torch.rand(batch_size, 50), dim=1).cuda()
                visits = torch.randint(0, 100, (batch_size, 50), device='cuda').float()
                values = torch.rand(batch_size, 50, device='cuda')
                # For path integrals, create values/visits along paths
                path_values = torch.rand(batch_size, path_length, device='cuda')
                path_visits = torch.randint(1, 100, (batch_size, path_length), device='cuda').float()
                
                kernel_times = {}
                
                # 1. Fused MinHash + Interference
                torch.cuda.synchronize()
                start = time.perf_counter()
                sigs, sims, new_scores = kernels.fused_minhash_interference(paths, scores)
                torch.cuda.synchronize()
                kernel_times['minhash_interference'] = time.perf_counter() - start
                
                # 2. Phase-kicked policy
                torch.cuda.synchronize()
                start = time.perf_counter()
                kicked, uncert, phases = kernels.phase_kicked_policy(priors, visits, values)
                torch.cuda.synchronize()
                kernel_times['phase_policy'] = time.perf_counter() - start
                
                # 3. Path integrals
                torch.cuda.synchronize()
                start = time.perf_counter()
                weights = kernels.quantum_path_integrals(
                    paths[:50], path_values[:50], path_visits[:50], temperature=1.0
                )
                torch.cuda.synchronize()
                kernel_times['path_integral'] = time.perf_counter() - start
                
                results['kernels'][config_key] = kernel_times
                
                for kernel, time_ms in kernel_times.items():
                    print(f"  {kernel}: {time_ms*1000:.2f}ms")
        
        return results
    
    def benchmark_full_mcts(self) -> Dict[str, Any]:
        """Benchmark full MCTS with quantum features"""
        print("\n📊 Benchmarking Full Quantum MCTS")
        print("-" * 50)
        
        results = {
            'name': 'Full MCTS Benchmark',
            'configurations': []
        }
        
        # Create evaluator
        evaluator = ResNetEvaluator(
            game_type='gomoku',
            device=self.device
        )
        
        # Test configurations
        configs = {
            'Classical': {
                'enable_quantum': False
            },
            'Quantum-Light': {
                'enable_quantum': True,
                'quantum_config': {
                    'hbar_eff': 0.05,
                    'temperature': 1.0,
                    'fast_mode': True,
                    'min_wave_size': 32
                }
            },
            'Quantum-Full': {
                'enable_quantum': True,
                'quantum_config': {
                    'hbar_eff': 0.1,
                    'temperature': 1.5,
                    'fast_mode': False,
                    'min_wave_size': 64,
                    'optimal_wave_size': 512,
                    'cache_corrections': True
                }
            }
        }
        
        # Base config - Optimized for Ryzen 9 5900X + RTX 3060 Ti (8GB VRAM)
        base_config = {
            'min_wave_size': 3072,
            'max_wave_size': 3072,
            'adaptive_wave_sizing': False,
            'memory_pool_size_mb': 2048,
            'max_tree_nodes': 500000,
            'use_mixed_precision': True,
            'use_cuda_graphs': True,
            'use_tensor_cores': True,
            'game_type': GameType.GOMOKU,
            'device': self.device
        }
        
        # Test each configuration
        for name, quantum_config in configs.items():
            print(f"\nTesting {name} configuration")
            
            # Create config
            config_dict = {**base_config, **quantum_config}
            config = MCTSConfig(**config_dict)
            
            # Create MCTS
            mcts = MCTS(config, evaluator)
            mcts.optimize_for_hardware()
            
            # Test position
            game = alphazero_py.GomokuState()
            
            # Warmup
            for _ in range(3):
                mcts.search(game, num_simulations=1000)
            
            # Benchmark different simulation counts
            sim_counts = [1000, 5000, 10000, 50000]
            config_results = {'name': name, 'simulations': []}
            
            for num_sims in sim_counts:
                # Run multiple times for stability
                times = []
                for _ in range(3):
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
                    start = time.perf_counter()
                    policy = mcts.search(game, num_simulations=num_sims)
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
                    elapsed = time.perf_counter() - start
                    times.append(elapsed)
                
                avg_time = np.mean(times)
                std_time = np.std(times)
                
                sim_result = {
                    'count': num_sims,
                    'avg_time': avg_time,
                    'std_time': std_time,
                    'speed': num_sims / avg_time,
                    'policy_entropy': -np.sum(policy * np.log(policy + 1e-8))
                }
                
                config_results['simulations'].append(sim_result)
                
                print(f"  {num_sims} sims: {avg_time:.3f}±{std_time:.3f}s "
                      f"({num_sims/avg_time:.0f} sims/sec)")
            
            results['configurations'].append(config_results)
        
        return results
    
    def _create_sparse_state(self, dim: int, sparsity: float) -> torch.Tensor:
        """Create a sparse quantum state"""
        state = torch.zeros(dim, dtype=torch.complex64, device=self.device)
        nnz = int(dim * sparsity)
        indices = torch.randperm(dim)[:nnz]
        state[indices] = torch.randn(nnz, dtype=torch.complex64, device=self.device)
        return state
    
    def plot_results(self):
        """Generate plots from benchmark results"""
        if not HAS_PLOTTING:
            print("\n⚠️  Plotting skipped (matplotlib/seaborn not available)")
            return
            
        print("\n📈 Generating plots...")
        
        # Create plots directory
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. MCTS Speed Comparison
        if 'Full MCTS Benchmark' in self.results['benchmarks']:
            self._plot_mcts_speed_comparison(plots_dir)
        
        # 2. Compression Performance
        if 'Wave Function Compression' in self.results['benchmarks']:
            self._plot_compression_performance(plots_dir)
        
        # 3. Quantum Parallelism Scaling
        if 'Quantum Parallelism' in self.results['benchmarks']:
            self._plot_parallelism_scaling(plots_dir)
        
        print(f"✓ Plots saved to {plots_dir}")
    
    def _plot_mcts_speed_comparison(self, plots_dir: Path):
        """Plot MCTS speed comparison"""
        data = self.results['benchmarks']['Full MCTS Benchmark']['configurations']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Speed comparison
        for config in data:
            sims = [s['count'] for s in config['simulations']]
            speeds = [s['speed'] for s in config['simulations']]
            ax1.plot(sims, speeds, marker='o', label=config['name'], linewidth=2)
        
        ax1.set_xlabel('Number of Simulations')
        ax1.set_ylabel('Simulations per Second')
        ax1.set_title('MCTS Speed Comparison')
        ax1.legend()
        ax1.set_xscale('log')
        
        # Overhead analysis
        classical_speeds = {}
        quantum_overheads = {}
        
        for config in data:
            for sim in config['simulations']:
                if config['name'] == 'Classical':
                    classical_speeds[sim['count']] = sim['speed']
                else:
                    if sim['count'] in classical_speeds:
                        overhead = (classical_speeds[sim['count']] / sim['speed'] - 1) * 100
                        if config['name'] not in quantum_overheads:
                            quantum_overheads[config['name']] = []
                        quantum_overheads[config['name']].append((sim['count'], overhead))
        
        for name, overheads in quantum_overheads.items():
            sims, ovh = zip(*overheads)
            ax2.plot(sims, ovh, marker='o', label=name, linewidth=2)
        
        ax2.set_xlabel('Number of Simulations')
        ax2.set_ylabel('Overhead (%)')
        ax2.set_title('Quantum Feature Overhead')
        ax2.legend()
        ax2.set_xscale('log')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'mcts_speed_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_compression_performance(self, plots_dir: Path):
        """Plot compression performance"""
        data = self.results['benchmarks']['Wave Function Compression']['dimensions']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Compression ratio vs dimension
        methods = set()
        for dim_data in data:
            for method_key in dim_data['methods']:
                methods.add(method_key.split('_')[0])
        
        for method in methods:
            dims = []
            ratios = []
            fidelities = []
            
            for dim_data in data:
                for method_key, result in dim_data['methods'].items():
                    if method_key.startswith(method):
                        dims.append(dim_data['dimension'])
                        ratios.append(result['compression_ratio'])
                        fidelities.append(result['fidelity'])
                        break
            
            if dims:
                ax1.plot(dims, ratios, marker='o', label=method, linewidth=2)
                ax2.plot(dims, fidelities, marker='s', label=method, linewidth=2)
        
        ax1.set_xlabel('State Dimension')
        ax1.set_ylabel('Compression Ratio')
        ax1.set_title('Compression Ratio vs Dimension')
        ax1.legend()
        ax1.set_xscale('log')
        
        ax2.set_xlabel('State Dimension')
        ax2.set_ylabel('Fidelity')
        ax2.set_title('Compression Fidelity vs Dimension')
        ax2.legend()
        ax2.set_xscale('log')
        ax2.set_ylim(0.9, 1.01)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'compression_performance.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_parallelism_scaling(self, plots_dir: Path):
        """Plot quantum parallelism scaling"""
        data = self.results['benchmarks']['Quantum Parallelism']['path_counts']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Extract data for different Grover iterations
        grover_iters = set()
        for path_data in data:
            for result in path_data['grover_results']:
                grover_iters.add(result['iterations'])
        
        for grover in sorted(grover_iters):
            path_counts = []
            times = []
            
            for path_data in data:
                for result in path_data['grover_results']:
                    if result['iterations'] == grover:
                        path_counts.append(path_data['num_paths'])
                        times.append(result['time'])
                        break
            
            ax.plot(path_counts, times, marker='o', 
                   label=f'Grover iterations: {grover}', linewidth=2)
        
        ax.set_xlabel('Number of Paths')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Quantum Parallelism Scaling')
        ax.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'parallelism_scaling.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_all_benchmarks(self):
        """Run all benchmarks"""
        print("\n🚀 Starting Quantum MCTS Benchmark Suite")
        print("=" * 60)
        
        # Run benchmarks
        benchmarks = [
            self.benchmark_quantum_state_pool,
            self.benchmark_wave_compression,
            self.benchmark_quantum_parallelism,
            self.benchmark_gpu_kernels,
            self.benchmark_full_mcts
        ]
        
        for benchmark_func in benchmarks:
            try:
                result = benchmark_func()
                self.results['benchmarks'][result['name']] = result
            except Exception as e:
                print(f"\n❌ {benchmark_func.__name__} failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Save results
        results_file = self.output_dir / 'benchmark_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n✅ Results saved to {results_file}")
        
        # Generate plots
        self.plot_results()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "="*60)
        print("📊 BENCHMARK SUMMARY")
        print("="*60)
        
        if 'Full MCTS Benchmark' in self.results['benchmarks']:
            mcts_data = self.results['benchmarks']['Full MCTS Benchmark']
            
            print("\n🎯 MCTS Performance (50k simulations):")
            for config in mcts_data['configurations']:
                for sim in config['simulations']:
                    if sim['count'] == 50000:
                        print(f"  {config['name']:15} {sim['speed']:>8.0f} sims/sec")
        
        if 'Wave Function Compression' in self.results['benchmarks']:
            comp_data = self.results['benchmarks']['Wave Function Compression']
            
            print("\n🗜️  Best Compression Ratios (4096 dim):")
            best_ratios = {}
            for dim_data in comp_data['dimensions']:
                if dim_data['dimension'] == 4096:
                    for method_key, result in dim_data['methods'].items():
                        method = method_key.split('_')[0]
                        if method not in best_ratios or result['compression_ratio'] > best_ratios[method]:
                            best_ratios[method] = result['compression_ratio']
            
            for method, ratio in sorted(best_ratios.items(), key=lambda x: x[1], reverse=True):
                print(f"  {method:15} {ratio:>6.1f}x")
        
        print("\n✨ Quantum features successfully integrated!")
        print(f"📁 Full results in: {self.output_dir}")


if __name__ == "__main__":
    # Run benchmarks
    benchmark = QuantumMCTSBenchmark()
    
    try:
        benchmark.run_all_benchmarks()
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()