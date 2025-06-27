"""
Performance Optimization Test for Quantum MCTS
==============================================

Goal: Make quantum MCTS faster than classical MCTS
Current status: 1.67x overhead (need to get below 1.0x)

This test identifies bottlenecks and optimizes performance systematically.
"""

import torch
import numpy as np
import time
import sys
import os
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcts.quantum.pragmatic_quantum_mcts import create_pragmatic_quantum_mcts
from mcts.quantum.selective_quantum_optimized import create_selective_quantum_mcts

class PerformanceOptimizer:
    """Systematically optimize quantum MCTS performance"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.results = {}
        
    def benchmark_baseline_classical(self, batch_size=1000, num_actions=30):
        """Benchmark pure classical MCTS performance"""
        print("🔬 Benchmarking Classical MCTS Baseline...")
        
        # Use selective MCTS in classical mode
        classical_mcts = create_selective_quantum_mcts(
            device=self.device, enable_cuda_kernels=False
        )
        
        # Generate test data
        q_values = torch.randn(batch_size, num_actions, device=self.device) * 0.3
        visit_counts = torch.randint(1, 100, (batch_size, num_actions), 
                                   dtype=torch.float32, device=self.device)
        priors = torch.softmax(torch.randn(batch_size, num_actions, device=self.device), dim=-1)
        parent_visits = torch.sum(visit_counts, dim=-1)
        
        # Benchmark classical processing
        start_time = time.time()
        for i in range(batch_size):
            scores = classical_mcts._classical_v5_vectorized(
                q_values[i], visit_counts[i], priors[i], parent_visits[i].item()
            )
        classical_time = time.time() - start_time
        
        classical_throughput = batch_size / classical_time
        
        self.results['classical_baseline'] = {
            'time': classical_time,
            'throughput': classical_throughput,
            'batch_size': batch_size
        }
        
        print(f"✓ Classical throughput: {classical_throughput:.0f} operations/sec")
        return classical_throughput
    
    def benchmark_current_quantum(self, batch_size=1000, num_actions=30):
        """Benchmark current quantum implementation"""
        print("⚛️  Benchmarking Current Quantum MCTS...")
        
        quantum_mcts = create_pragmatic_quantum_mcts(device=self.device)
        
        # Generate same test data
        q_values = torch.randn(batch_size, num_actions, device=self.device) * 0.3
        visit_counts = torch.randint(1, 100, (batch_size, num_actions), 
                                   dtype=torch.float32, device=self.device)
        priors = torch.softmax(torch.randn(batch_size, num_actions, device=self.device), dim=-1)
        parent_visits = torch.sum(visit_counts, dim=-1)
        
        # Benchmark quantum processing
        start_time = time.time()
        for i in range(batch_size):
            scores = quantum_mcts.compute_enhanced_ucb_scores(
                q_values[i], visit_counts[i], priors[i], 
                parent_visits[i].item(), simulation_count=1000
            )
        quantum_time = time.time() - start_time
        
        quantum_throughput = batch_size / quantum_time
        
        self.results['quantum_current'] = {
            'time': quantum_time,
            'throughput': quantum_throughput,
            'batch_size': batch_size
        }
        
        overhead_ratio = quantum_time / self.results['classical_baseline']['time']
        
        print(f"✓ Quantum throughput: {quantum_throughput:.0f} operations/sec")
        print(f"✓ Overhead ratio: {overhead_ratio:.2f}x")
        
        return quantum_throughput, overhead_ratio
    
    def benchmark_vectorized_quantum(self, batch_size=1000, num_actions=30):
        """Benchmark vectorized quantum processing"""
        print("🌊 Benchmarking Vectorized Quantum MCTS...")
        
        quantum_mcts = create_pragmatic_quantum_mcts(device=self.device)
        
        # Generate test data
        q_values = torch.randn(batch_size, num_actions, device=self.device) * 0.3
        visit_counts = torch.randint(1, 100, (batch_size, num_actions), 
                                   dtype=torch.float32, device=self.device)
        priors = torch.softmax(torch.randn(batch_size, num_actions, device=self.device), dim=-1)
        parent_visits = torch.sum(visit_counts, dim=-1)
        
        # Benchmark vectorized processing
        start_time = time.time()
        scores = quantum_mcts.batch_compute_enhanced_ucb(
            q_values, visit_counts, priors, parent_visits, simulation_count=1000
        )
        vectorized_time = time.time() - start_time
        
        vectorized_throughput = batch_size / vectorized_time
        
        self.results['quantum_vectorized'] = {
            'time': vectorized_time,
            'throughput': vectorized_throughput,
            'batch_size': batch_size
        }
        
        overhead_ratio = vectorized_time / self.results['classical_baseline']['time']
        speedup_vs_individual = self.results['quantum_current']['time'] / vectorized_time
        
        print(f"✓ Vectorized throughput: {vectorized_throughput:.0f} operations/sec")
        print(f"✓ Overhead vs classical: {overhead_ratio:.2f}x")
        print(f"✓ Speedup vs individual: {speedup_vs_individual:.2f}x")
        
        return vectorized_throughput, overhead_ratio
    
    def optimize_quantum_features(self):
        """Create optimized quantum configurations"""
        print("🔧 Creating Optimized Quantum Configurations...")
        
        # Configuration 1: Minimal quantum (focus on speed)
        minimal_config = {
            'device': self.device,
            'quantum_bonus_coefficient': 0.05,  # Reduced
            'enable_quantum_bonus': True,
            'enable_power_law_annealing': False,  # Disable for speed
            'enable_phase_adaptation': False,     # Disable for speed
            'enable_correlation_prioritization': False,  # Disable for speed
        }
        
        # Configuration 2: Performance-focused quantum
        performance_config = {
            'device': self.device,
            'quantum_bonus_coefficient': 0.08,
            'enable_quantum_bonus': True,
            'enable_power_law_annealing': True,
            'enable_phase_adaptation': True,
            'enable_correlation_prioritization': False,  # Still disabled for speed
            'quantum_crossover_threshold': 30,  # Lower threshold for less quantum processing
        }
        
        # Configuration 3: Ultra-fast quantum
        ultrafast_config = {
            'device': self.device,
            'quantum_bonus_coefficient': 0.03,  # Very small
            'enable_quantum_bonus': True,
            'enable_power_law_annealing': False,
            'enable_phase_adaptation': False,
            'enable_correlation_prioritization': False,
            'quantum_crossover_threshold': 20,  # Very low threshold
        }
        
        configs = {
            'minimal': minimal_config,
            'performance': performance_config,
            'ultrafast': ultrafast_config
        }
        
        # Test each configuration
        batch_size = 1000
        num_actions = 30
        
        q_values = torch.randn(batch_size, num_actions, device=self.device) * 0.3
        visit_counts = torch.randint(1, 100, (batch_size, num_actions), 
                                   dtype=torch.float32, device=self.device)
        priors = torch.softmax(torch.randn(batch_size, num_actions, device=self.device), dim=-1)
        parent_visits = torch.sum(visit_counts, dim=-1)
        
        config_results = {}
        
        for config_name, config in configs.items():
            from mcts.quantum.pragmatic_quantum_mcts import PragmaticQuantumConfig, PragmaticQuantumMCTS
            
            quantum_config = PragmaticQuantumConfig(**config)
            quantum_mcts = PragmaticQuantumMCTS(quantum_config)
            
            # Benchmark this configuration
            start_time = time.time()
            scores = quantum_mcts.batch_compute_enhanced_ucb(
                q_values, visit_counts, priors, parent_visits, simulation_count=1000
            )
            config_time = time.time() - start_time
            
            config_throughput = batch_size / config_time
            overhead_ratio = config_time / self.results['classical_baseline']['time']
            
            config_results[config_name] = {
                'time': config_time,
                'throughput': config_throughput,
                'overhead_ratio': overhead_ratio,
                'config': config
            }
            
            print(f"✓ {config_name.title()} config: {config_throughput:.0f} ops/sec ({overhead_ratio:.2f}x overhead)")
        
        self.results['optimized_configs'] = config_results
        return config_results
    
    def test_cuda_acceleration(self):
        """Test CUDA acceleration if available"""
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, skipping GPU tests")
            return {}
        
        print("🚀 Testing CUDA Acceleration...")
        
        # Test with CUDA device
        quantum_mcts_cuda = create_selective_quantum_mcts(
            device='cuda', enable_cuda_kernels=True
        )
        
        batch_size = 3072  # Optimal wave size
        num_actions = 30
        
        # Generate test data on GPU
        q_values = torch.randn(batch_size, num_actions, device='cuda') * 0.3
        visit_counts = torch.randint(1, 100, (batch_size, num_actions), 
                                   dtype=torch.float32, device='cuda')
        priors = torch.softmax(torch.randn(batch_size, num_actions, device='cuda'), dim=-1)
        parent_visits = torch.sum(visit_counts, dim=-1)
        
        # Benchmark CUDA processing
        torch.cuda.synchronize()
        start_time = time.time()
        
        for i in range(batch_size):
            scores = quantum_mcts_cuda.apply_selective_quantum(
                q_values[i], visit_counts[i], priors[i],
                parent_visits=parent_visits[i].item(), simulation_count=1000
            )
        
        torch.cuda.synchronize()
        cuda_time = time.time() - start_time
        
        cuda_throughput = batch_size / cuda_time
        
        # Compare with CPU baseline
        cpu_baseline_throughput = self.results['classical_baseline']['throughput']
        speedup = cuda_throughput / cpu_baseline_throughput
        
        cuda_results = {
            'throughput': cuda_throughput,
            'speedup_vs_cpu_classical': speedup,
            'batch_size': batch_size
        }
        
        self.results['cuda_acceleration'] = cuda_results
        
        print(f"✓ CUDA throughput: {cuda_throughput:.0f} operations/sec")
        print(f"✓ Speedup vs CPU classical: {speedup:.2f}x")
        
        return cuda_results
    
    def run_comprehensive_optimization(self):
        """Run comprehensive performance optimization"""
        print("Performance Optimization for Quantum MCTS")
        print("=" * 50)
        print("Goal: Make quantum MCTS faster than classical MCTS")
        print("=" * 50)
        
        # Step 1: Establish baselines
        classical_throughput = self.benchmark_baseline_classical()
        quantum_throughput, quantum_overhead = self.benchmark_current_quantum()
        vectorized_throughput, vectorized_overhead = self.benchmark_vectorized_quantum()
        
        print(f"\n📊 BASELINE RESULTS:")
        print(f"Classical: {classical_throughput:.0f} ops/sec")
        print(f"Quantum (individual): {quantum_throughput:.0f} ops/sec ({quantum_overhead:.2f}x)")
        print(f"Quantum (vectorized): {vectorized_throughput:.0f} ops/sec ({vectorized_overhead:.2f}x)")
        
        # Step 2: Optimize quantum configurations
        optimized_configs = self.optimize_quantum_features()
        
        # Step 3: Test CUDA acceleration
        cuda_results = self.test_cuda_acceleration()
        
        # Step 4: Find best performing configuration
        print(f"\n🏆 OPTIMIZATION RESULTS:")
        print("-" * 30)
        
        best_config = None
        best_throughput = 0
        
        for config_name, result in optimized_configs.items():
            if result['throughput'] > best_throughput:
                best_throughput = result['throughput']
                best_config = config_name
        
        if best_config:
            best_result = optimized_configs[best_config]
            print(f"Best configuration: {best_config}")
            print(f"Throughput: {best_result['throughput']:.0f} ops/sec")
            print(f"Overhead: {best_result['overhead_ratio']:.2f}x")
            
            if best_result['overhead_ratio'] < 1.0:
                print("🎉 SUCCESS: Quantum MCTS is faster than classical!")
            else:
                print(f"⚠️  Still {best_result['overhead_ratio']:.2f}x slower, need more optimization")
        
        if cuda_results and 'speedup_vs_cpu_classical' in cuda_results:
            cuda_speedup = cuda_results['speedup_vs_cpu_classical']
            if cuda_speedup > 1.0:
                print(f"🚀 CUDA acceleration: {cuda_speedup:.2f}x faster than classical!")
            else:
                print(f"⚠️  CUDA needs optimization: {cuda_speedup:.2f}x vs classical")
        
        # Summary and next steps
        print(f"\n📋 NEXT OPTIMIZATION STEPS:")
        
        if vectorized_overhead > 1.0:
            print("1. 🔧 Optimize vectorized quantum processing")
            print("2. ⚡ Reduce quantum computation overhead")
            print("3. 🎯 Implement selective quantum application")
        
        if best_config and optimized_configs[best_config]['overhead_ratio'] > 1.0:
            print("4. 🚀 Improve CUDA kernel performance")
            print("5. 📦 Optimize memory access patterns")
            print("6. ⚙️  Reduce feature complexity for speed")
        
        return self.results

def main():
    """Main optimization testing"""
    device = 'cpu'  # Start with CPU for consistency
    
    optimizer = PerformanceOptimizer(device=device)
    results = optimizer.run_comprehensive_optimization()
    
    # Determine if we've achieved the goal
    best_overhead = float('inf')
    for config_name, result in results.get('optimized_configs', {}).items():
        if result['overhead_ratio'] < best_overhead:
            best_overhead = result['overhead_ratio']
    
    goal_achieved = best_overhead < 1.0
    
    print(f"\n" + "=" * 50)
    print("PERFORMANCE OPTIMIZATION SUMMARY")
    print("=" * 50)
    
    if goal_achieved:
        print("✅ GOAL ACHIEVED: Quantum MCTS is faster than classical!")
    else:
        print("⚠️  Goal not yet achieved - continuing optimization needed")
        print(f"Best overhead ratio: {best_overhead:.2f}x (target: < 1.0x)")
    
    return goal_achieved, results

if __name__ == "__main__":
    success, results = main()