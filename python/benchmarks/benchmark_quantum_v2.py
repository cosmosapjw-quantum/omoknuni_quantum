"""
Benchmark Quantum MCTS v2.0 vs v1.0
==================================

This script compares the performance of quantum MCTS v1.0 and v2.0
across different scenarios and configurations.
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime

from mcts.quantum import (
    create_quantum_mcts, compare_versions,
    UnifiedQuantumConfig, MCTSPhase
)
from mcts.core.mcts import MCTS, MCTSConfig
from mcts.core.game_interface import GameType
# Import a mock evaluator instead
import torch
import numpy as np
from typing import Tuple

class MockEvaluator:
    """Simple mock evaluator for benchmarking"""
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
    
    def evaluate_batch(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return random policy and value"""
        batch_size = features.shape[0]
        board_size_sq = 225  # 15x15 for Gomoku
        
        # Random policy (normalized)
        policies = torch.rand(batch_size, board_size_sq, device=self.device)
        policies = torch.softmax(policies, dim=-1)
        
        # Random values between -1 and 1
        values = torch.rand(batch_size, 1, device=self.device) * 2 - 1
        
        return policies, values


class QuantumBenchmark:
    """Benchmark suite for quantum MCTS versions"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'device': self.device,
            'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
            'benchmarks': []
        }
    
    def benchmark_selection_overhead(self, 
                                   batch_sizes: List[int] = [32, 64, 128, 256, 512, 1024],
                                   num_actions: int = 50,
                                   num_iterations: int = 100) -> Dict[str, Any]:
        """Benchmark quantum selection overhead for different batch sizes"""
        print("\n=== Benchmarking Selection Overhead ===")
        
        results = {
            'name': 'selection_overhead',
            'batch_sizes': batch_sizes,
            'v1_times': [],
            'v2_times': [],
            'overheads': []
        }
        
        for batch_size in batch_sizes:
            print(f"\nBatch size: {batch_size}")
            
            # Create test data
            q_values = torch.randn(batch_size, num_actions, device=self.device)
            visit_counts = torch.randint(0, 100, (batch_size, num_actions), device=self.device)
            priors = torch.softmax(torch.randn(batch_size, num_actions, device=self.device), dim=-1)
            
            # Create v1 and v2 instances
            quantum_v1 = create_quantum_mcts(
                version='v1',
                device=self.device,
                enable_quantum=True
            )
            
            quantum_v2 = create_quantum_mcts(
                version='v2',
                branching_factor=num_actions,
                device=self.device,
                enable_quantum=True
            )
            
            # Warmup
            for _ in range(10):
                _ = quantum_v1.apply_quantum_to_selection(q_values, visit_counts, priors)
                _ = quantum_v2.apply_quantum_to_selection(q_values, visit_counts, priors)
            
            # Benchmark v1
            torch.cuda.synchronize() if self.device == 'cuda' else None
            start = time.perf_counter()
            for _ in range(num_iterations):
                _ = quantum_v1.apply_quantum_to_selection(q_values, visit_counts, priors)
            torch.cuda.synchronize() if self.device == 'cuda' else None
            v1_time = time.perf_counter() - start
            
            # Benchmark v2
            torch.cuda.synchronize() if self.device == 'cuda' else None
            start = time.perf_counter()
            for _ in range(num_iterations):
                _ = quantum_v2.apply_quantum_to_selection(
                    q_values, visit_counts, priors,
                    c_puct=1.414,
                    total_simulations=1000
                )
            torch.cuda.synchronize() if self.device == 'cuda' else None
            v2_time = time.perf_counter() - start
            
            overhead = v2_time / v1_time
            results['v1_times'].append(v1_time)
            results['v2_times'].append(v2_time)
            results['overheads'].append(overhead)
            
            print(f"  v1.0: {v1_time:.4f}s ({num_iterations/v1_time:.0f} ops/sec)")
            print(f"  v2.0: {v2_time:.4f}s ({num_iterations/v2_time:.0f} ops/sec)")
            print(f"  Overhead: {overhead:.2f}x")
        
        self.results['benchmarks'].append(results)
        return results
    
    def benchmark_phase_transitions(self,
                                  simulation_counts: List[int] = [10, 100, 1000, 10000, 50000],
                                  branching_factor: int = 50) -> Dict[str, Any]:
        """Benchmark behavior across phase transitions"""
        print("\n=== Benchmarking Phase Transitions ===")
        
        results = {
            'name': 'phase_transitions',
            'simulation_counts': simulation_counts,
            'phases': [],
            'temperatures': [],
            'hbar_values': [],
            'overhead_by_phase': {}
        }
        
        # Create v2 instance
        quantum_v2 = create_quantum_mcts(
            version='v2',
            branching_factor=branching_factor,
            avg_game_length=100,
            device=self.device,
            enable_quantum=True
        )
        
        # Test data
        batch_size = 64
        q_values = torch.randn(batch_size, branching_factor, device=self.device)
        visit_counts = torch.randint(0, 50, (batch_size, branching_factor), device=self.device)
        priors = torch.softmax(torch.randn(batch_size, branching_factor, device=self.device), dim=-1)
        
        for N in simulation_counts:
            print(f"\nSimulation count: {N}")
            
            # Apply quantum selection
            _ = quantum_v2.apply_quantum_to_selection(
                q_values, visit_counts, priors,
                total_simulations=N
            )
            
            # Get phase info
            phase_info = quantum_v2.get_phase_info()
            stats = quantum_v2.get_statistics()
            
            results['phases'].append(phase_info['current_phase'])
            if 'current_temperature' in stats:
                results['temperatures'].append(stats['current_temperature'])
            if 'current_hbar_eff' in stats:
                results['hbar_values'].append(stats['current_hbar_eff'])
            
            print(f"  Phase: {phase_info['current_phase']}")
            if 'current_temperature' in stats:
                print(f"  Temperature: {stats['current_temperature']:.4f}")
        
        self.results['benchmarks'].append(results)
        return results
    
    def benchmark_mcts_integration(self,
                                 num_simulations: int = 1000,
                                 num_searches: int = 10) -> Dict[str, Any]:
        """Benchmark full MCTS integration"""
        print("\n=== Benchmarking MCTS Integration ===")
        
        results = {
            'name': 'mcts_integration',
            'num_simulations': num_simulations,
            'classical_times': [],
            'v1_times': [],
            'v2_times': [],
            'v1_overhead': 0,
            'v2_overhead': 0
        }
        
        # Create game state
        try:
            import alphazero_py
            state = alphazero_py.GomokuState()
            game_available = True
        except ImportError:
            print("Warning: alphazero_py not available, using mock state")
            game_available = False
            state = None
        
        if game_available:
            # Create mock evaluator for benchmarking
            evaluator = MockEvaluator(device=self.device)
            
            # Classical MCTS
            config_classical = MCTSConfig(
                num_simulations=num_simulations,
                enable_quantum=False,
                min_wave_size=32,
                max_wave_size=256,
                adaptive_wave_sizing=False,
                device=self.device,
                game_type=GameType.GOMOKU
            )
            mcts_classical = MCTS(config_classical, evaluator)
            
            # Quantum v1 MCTS
            config_v1 = MCTSConfig(
                num_simulations=num_simulations,
                enable_quantum=True,
                quantum_version='v1',
                min_wave_size=32,
                max_wave_size=256,
                adaptive_wave_sizing=False,
                device=self.device,
                game_type=GameType.GOMOKU
            )
            mcts_v1 = MCTS(config_v1, evaluator)
            
            # Quantum v2 MCTS
            config_v2 = MCTSConfig(
                num_simulations=num_simulations,
                enable_quantum=True,
                quantum_version='v2',
                quantum_branching_factor=225,
                quantum_avg_game_length=100,
                min_wave_size=32,
                max_wave_size=256,
                adaptive_wave_sizing=False,
                device=self.device,
                game_type=GameType.GOMOKU
            )
            mcts_v2 = MCTS(config_v2, evaluator)
            
            # Warmup
            print("Warming up...")
            for mcts in [mcts_classical, mcts_v1, mcts_v2]:
                mcts.search(state, num_simulations=100)
            
            # Benchmark
            print("\nRunning benchmarks...")
            for i in range(num_searches):
                # Classical
                start = time.perf_counter()
                _ = mcts_classical.search(state)
                results['classical_times'].append(time.perf_counter() - start)
                
                # v1
                start = time.perf_counter()
                _ = mcts_v1.search(state)
                results['v1_times'].append(time.perf_counter() - start)
                
                # v2
                start = time.perf_counter()
                _ = mcts_v2.search(state)
                results['v2_times'].append(time.perf_counter() - start)
            
            # Calculate overheads
            avg_classical = np.mean(results['classical_times'])
            avg_v1 = np.mean(results['v1_times'])
            avg_v2 = np.mean(results['v2_times'])
            
            results['v1_overhead'] = avg_v1 / avg_classical
            results['v2_overhead'] = avg_v2 / avg_classical
            
            print(f"\nResults:")
            print(f"  Classical: {avg_classical:.4f}s ({num_simulations/avg_classical:.0f} sims/sec)")
            print(f"  Quantum v1: {avg_v1:.4f}s ({num_simulations/avg_v1:.0f} sims/sec) - {results['v1_overhead']:.2f}x overhead")
            print(f"  Quantum v2: {avg_v2:.4f}s ({num_simulations/avg_v2:.0f} sims/sec) - {results['v2_overhead']:.2f}x overhead")
        
        self.results['benchmarks'].append(results)
        return results
    
    def plot_results(self):
        """Generate plots from benchmark results"""
        print("\n=== Generating Plots ===")
        
        # Selection overhead plot
        for benchmark in self.results['benchmarks']:
            if benchmark['name'] == 'selection_overhead':
                plt.figure(figsize=(10, 6))
                plt.plot(benchmark['batch_sizes'], benchmark['overheads'], 'o-', label='v2.0 overhead')
                plt.axhline(y=1.8, color='r', linestyle='--', label='Target: 1.8x')
                plt.xlabel('Batch Size')
                plt.ylabel('Overhead (v2/v1)')
                plt.title('Quantum MCTS v2.0 Selection Overhead')
                plt.legend()
                plt.grid(True)
                plt.savefig('quantum_v2_selection_overhead.png')
                print("Saved: quantum_v2_selection_overhead.png")
        
        # Phase transition plot
        for benchmark in self.results['benchmarks']:
            if benchmark['name'] == 'phase_transitions' and benchmark['temperatures']:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                
                # Temperature plot
                ax1.semilogx(benchmark['simulation_counts'], benchmark['temperatures'], 'o-')
                ax1.set_xlabel('Simulation Count')
                ax1.set_ylabel('Temperature')
                ax1.set_title('Temperature Annealing in v2.0')
                ax1.grid(True)
                
                # Phase plot
                phase_map = {'quantum': 0, 'critical': 1, 'classical': 2}
                phase_values = [phase_map.get(p, -1) for p in benchmark['phases']]
                ax2.semilogx(benchmark['simulation_counts'], phase_values, 'o-')
                ax2.set_xlabel('Simulation Count')
                ax2.set_ylabel('Phase')
                ax2.set_yticks([0, 1, 2])
                ax2.set_yticklabels(['Quantum', 'Critical', 'Classical'])
                ax2.set_title('Phase Transitions in v2.0')
                ax2.grid(True)
                
                plt.tight_layout()
                plt.savefig('quantum_v2_phase_transitions.png')
                print("Saved: quantum_v2_phase_transitions.png")
    
    def save_results(self, filename: str = 'quantum_v2_benchmark_results.json'):
        """Save benchmark results to JSON"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {filename}")
    
    def run_all_benchmarks(self):
        """Run complete benchmark suite"""
        print(f"Starting Quantum MCTS v2.0 Benchmarks")
        print(f"Device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
        print("=" * 50)
        
        # Run benchmarks
        self.benchmark_selection_overhead()
        self.benchmark_phase_transitions()
        self.benchmark_mcts_integration()
        
        # Generate plots
        self.plot_results()
        
        # Save results
        self.save_results()
        
        # Print summary
        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)
        
        for benchmark in self.results['benchmarks']:
            if benchmark['name'] == 'selection_overhead':
                avg_overhead = np.mean(benchmark['overheads'])
                print(f"\nSelection Overhead (avg): {avg_overhead:.2f}x")
                print(f"  Target: < 1.8x with neural networks")
                print(f"  Status: {'✓ PASSED' if avg_overhead < 1.8 else '✗ NEEDS OPTIMIZATION'}")
            
            elif benchmark['name'] == 'mcts_integration' and benchmark['v2_overhead'] > 0:
                print(f"\nMCTS Integration Overhead:")
                print(f"  v1.0: {benchmark['v1_overhead']:.2f}x")
                print(f"  v2.0: {benchmark['v2_overhead']:.2f}x")
                print(f"  Status: {'✓ PASSED' if benchmark['v2_overhead'] < 2.0 else '✗ NEEDS OPTIMIZATION'}")


if __name__ == "__main__":
    # Run benchmarks
    benchmark = QuantumBenchmark()
    benchmark.run_all_benchmarks()
    
    # Also run quick comparison
    print("\n\n=== Quick Version Comparison ===")
    q_values = torch.randn(100, 50)
    visit_counts = torch.randint(0, 100, (100, 50))
    priors = torch.softmax(torch.randn(100, 50), dim=-1)
    
    comparison = compare_versions(
        q_values, visit_counts, priors,
        UnifiedQuantumConfig(branching_factor=50, device='cpu')
    )
    
    print(f"Max difference: {comparison['max_difference']:.6f}")
    print(f"Correlation: {comparison['correlation']:.4f}")
    print(f"Speedup: {comparison['speedup']:.2f}x")