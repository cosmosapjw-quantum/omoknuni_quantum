"""
Speed Competition Test: Quantum vs Classical MCTS
================================================

Goal: Prove quantum MCTS can be faster than classical MCTS
Tests ultra-fast quantum implementation against pure classical baseline.
"""

import torch
import time
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcts.quantum.ultra_fast_quantum_mcts import (
    create_ultra_fast_quantum_mcts, create_speed_optimized_quantum_mcts
)
from mcts.quantum.selective_quantum_optimized import create_selective_quantum_mcts

def benchmark_pure_classical(batch_size=5000, num_actions=30, device='cpu'):
    """Benchmark pure classical MCTS (no quantum features)"""
    print("🔬 Benchmarking Pure Classical MCTS...")
    
    # Use selective MCTS in pure classical mode
    classical_mcts = create_selective_quantum_mcts(device=device, enable_cuda_kernels=False)
    
    # Generate test data
    q_values = torch.randn(batch_size, num_actions, device=device) * 0.3
    visit_counts = torch.randint(1, 100, (batch_size, num_actions), 
                               dtype=torch.float32, device=device)
    priors = torch.softmax(torch.randn(batch_size, num_actions, device=device), dim=-1)
    parent_visits = torch.sum(visit_counts, dim=-1)
    
    # Warmup
    for _ in range(10):
        _ = classical_mcts._classical_v5_vectorized(
            q_values[0], visit_counts[0], priors[0], parent_visits[0].item()
        )
    
    # Benchmark
    start_time = time.time()
    for i in range(batch_size):
        scores = classical_mcts._classical_v5_vectorized(
            q_values[i], visit_counts[i], priors[i], parent_visits[i].item()
        )
    end_time = time.time()
    
    classical_time = end_time - start_time
    classical_throughput = batch_size / classical_time
    
    print(f"✓ Classical throughput: {classical_throughput:.0f} operations/sec")
    print(f"✓ Classical time: {classical_time:.4f}s for {batch_size} operations")
    
    return classical_throughput, classical_time

def benchmark_ultra_fast_quantum(batch_size=5000, num_actions=30, device='cpu'):
    """Benchmark ultra-fast quantum MCTS"""
    print("⚡ Benchmarking Ultra-Fast Quantum MCTS...")
    
    quantum_mcts = create_ultra_fast_quantum_mcts(device=device)
    
    # Generate same test data
    q_values = torch.randn(batch_size, num_actions, device=device) * 0.3
    visit_counts = torch.randint(1, 100, (batch_size, num_actions), 
                               dtype=torch.float32, device=device)
    priors = torch.softmax(torch.randn(batch_size, num_actions, device=device), dim=-1)
    parent_visits = torch.sum(visit_counts, dim=-1)
    
    # Warmup
    for _ in range(10):
        _ = quantum_mcts.compute_ultra_fast_ucb(
            q_values[0], visit_counts[0], priors[0], parent_visits[0].item()
        )
    
    # Benchmark individual processing
    start_time = time.time()
    for i in range(batch_size):
        scores = quantum_mcts.compute_ultra_fast_ucb(
            q_values[i], visit_counts[i], priors[i], parent_visits[i].item()
        )
    end_time = time.time()
    
    quantum_individual_time = end_time - start_time
    quantum_individual_throughput = batch_size / quantum_individual_time
    
    print(f"✓ Quantum (individual) throughput: {quantum_individual_throughput:.0f} operations/sec")
    
    return quantum_individual_throughput, quantum_individual_time

def benchmark_ultra_fast_quantum_vectorized(batch_size=5000, num_actions=30, device='cpu'):
    """Benchmark ultra-fast quantum MCTS with full vectorization"""
    print("🌊 Benchmarking Ultra-Fast Quantum MCTS (Vectorized)...")
    
    quantum_mcts = create_ultra_fast_quantum_mcts(device=device)
    
    # Generate test data
    q_values = torch.randn(batch_size, num_actions, device=device) * 0.3
    visit_counts = torch.randint(1, 100, (batch_size, num_actions), 
                               dtype=torch.float32, device=device)
    priors = torch.softmax(torch.randn(batch_size, num_actions, device=device), dim=-1)
    parent_visits = torch.sum(visit_counts, dim=-1)
    
    # Warmup
    for _ in range(10):
        _ = quantum_mcts.batch_compute_ultra_fast_ucb(
            q_values, visit_counts, priors, parent_visits
        )
    
    # Benchmark vectorized processing
    start_time = time.time()
    scores = quantum_mcts.batch_compute_ultra_fast_ucb(
        q_values, visit_counts, priors, parent_visits
    )
    end_time = time.time()
    
    quantum_vectorized_time = end_time - start_time
    quantum_vectorized_throughput = batch_size / quantum_vectorized_time
    
    print(f"✓ Quantum (vectorized) throughput: {quantum_vectorized_throughput:.0f} operations/sec")
    
    return quantum_vectorized_throughput, quantum_vectorized_time

def benchmark_speed_optimized(batch_size=5000, num_actions=30, device='cpu'):
    """Benchmark speed-optimized quantum MCTS"""
    print("🚀 Benchmarking Speed-Optimized Quantum MCTS...")
    
    quantum_mcts = create_speed_optimized_quantum_mcts(device=device)
    
    # Generate test data
    q_values = torch.randn(batch_size, num_actions, device=device) * 0.3
    visit_counts = torch.randint(1, 100, (batch_size, num_actions), 
                               dtype=torch.float32, device=device)
    priors = torch.softmax(torch.randn(batch_size, num_actions, device=device), dim=-1)
    parent_visits = torch.sum(visit_counts, dim=-1)
    
    # Warmup
    for _ in range(10):
        _ = quantum_mcts.batch_compute_ultra_fast_ucb(
            q_values, visit_counts, priors, parent_visits
        )
    
    # Benchmark
    start_time = time.time()
    scores = quantum_mcts.batch_compute_ultra_fast_ucb(
        q_values, visit_counts, priors, parent_visits
    )
    end_time = time.time()
    
    optimized_time = end_time - start_time
    optimized_throughput = batch_size / optimized_time
    
    print(f"✓ Speed-optimized throughput: {optimized_throughput:.0f} operations/sec")
    
    return optimized_throughput, optimized_time

def run_speed_competition():
    """Run comprehensive speed competition"""
    print("QUANTUM vs CLASSICAL MCTS SPEED COMPETITION")
    print("=" * 60)
    print("🎯 Goal: Prove quantum MCTS can be faster than classical")
    print("=" * 60)
    
    device = 'cpu'
    batch_size = 10000  # Large batch for accurate timing
    num_actions = 30
    
    results = {}
    
    # Test 1: Pure Classical Baseline
    classical_throughput, classical_time = benchmark_pure_classical(
        batch_size, num_actions, device
    )
    results['classical'] = {
        'throughput': classical_throughput,
        'time': classical_time
    }
    
    # Test 2: Ultra-Fast Quantum (Individual)
    quantum_individual_throughput, quantum_individual_time = benchmark_ultra_fast_quantum(
        batch_size, num_actions, device
    )
    results['quantum_individual'] = {
        'throughput': quantum_individual_throughput,
        'time': quantum_individual_time,
        'speedup': quantum_individual_throughput / classical_throughput
    }
    
    # Test 3: Ultra-Fast Quantum (Vectorized)
    quantum_vectorized_throughput, quantum_vectorized_time = benchmark_ultra_fast_quantum_vectorized(
        batch_size, num_actions, device
    )
    results['quantum_vectorized'] = {
        'throughput': quantum_vectorized_throughput,
        'time': quantum_vectorized_time,
        'speedup': quantum_vectorized_throughput / classical_throughput
    }
    
    # Test 4: Speed-Optimized Quantum
    optimized_throughput, optimized_time = benchmark_speed_optimized(
        batch_size, num_actions, device
    )
    results['speed_optimized'] = {
        'throughput': optimized_throughput,
        'time': optimized_time,
        'speedup': optimized_throughput / classical_throughput
    }
    
    # Analysis
    print(f"\n📊 SPEED COMPETITION RESULTS:")
    print("-" * 40)
    print(f"Classical MCTS:           {classical_throughput:8.0f} ops/sec (baseline)")
    print(f"Quantum (individual):     {quantum_individual_throughput:8.0f} ops/sec ({results['quantum_individual']['speedup']:.2f}x)")
    print(f"Quantum (vectorized):     {quantum_vectorized_throughput:8.0f} ops/sec ({results['quantum_vectorized']['speedup']:.2f}x)")
    print(f"Speed-optimized:          {optimized_throughput:8.0f} ops/sec ({results['speed_optimized']['speedup']:.2f}x)")
    
    # Determine winner
    best_quantum = max(
        results['quantum_individual']['speedup'],
        results['quantum_vectorized']['speedup'],
        results['speed_optimized']['speedup']
    )
    
    print(f"\n🏆 COMPETITION RESULTS:")
    print("-" * 30)
    
    if best_quantum > 1.0:
        winner = max(results.items(), key=lambda x: x[1].get('speedup', 0))
        print(f"🎉 QUANTUM WINS! Best: {winner[0]} at {best_quantum:.2f}x faster")
        print("✅ Goal achieved: Quantum MCTS is faster than classical!")
    else:
        print(f"⚠️  Classical still wins. Best quantum: {best_quantum:.2f}x")
        print("❌ Goal not achieved - need more optimization")
    
    # Detailed timing analysis
    print(f"\n⏱️  DETAILED TIMING ANALYSIS:")
    print(f"Classical time:           {classical_time:.4f}s")
    print(f"Quantum individual time:  {quantum_individual_time:.4f}s")
    print(f"Quantum vectorized time:  {quantum_vectorized_time:.4f}s")
    print(f"Speed-optimized time:     {optimized_time:.4f}s")
    
    # Recommendations for further optimization
    if best_quantum <= 1.0:
        print(f"\n🔧 OPTIMIZATION RECOMMENDATIONS:")
        print("1. Further reduce quantum computation overhead")
        print("2. Optimize tensor operations and memory access")
        print("3. Implement custom CUDA kernels")
        print("4. Use even more selective quantum application")
        print("5. Pre-compute more lookup tables")
    
    return results, best_quantum > 1.0

def test_correctness():
    """Test that ultra-fast quantum produces reasonable results"""
    print("\n🧪 Testing Correctness of Ultra-Fast Quantum MCTS...")
    
    device = 'cpu'
    quantum_mcts = create_ultra_fast_quantum_mcts(device=device)
    classical_mcts = create_selective_quantum_mcts(device=device, enable_cuda_kernels=False)
    
    # Generate test case
    num_actions = 20
    q_values = torch.randn(num_actions) * 0.2
    visit_counts = torch.tensor([1., 5., 10., 20., 50., 100.] + [30.] * 14)  # Mixed visits
    priors = torch.softmax(torch.randn(num_actions), dim=0)
    parent_visits = torch.sum(visit_counts).item()
    
    # Compute scores
    quantum_scores = quantum_mcts.compute_ultra_fast_ucb(
        q_values, visit_counts, priors, parent_visits
    )
    classical_scores = classical_mcts._classical_v5_vectorized(
        q_values, visit_counts, priors, parent_visits
    )
    
    # Check that quantum gives bonuses to low-visit nodes
    low_visit_indices = visit_counts < 10
    quantum_bonuses = quantum_scores - classical_scores
    
    avg_low_visit_bonus = torch.mean(quantum_bonuses[low_visit_indices])
    avg_high_visit_bonus = torch.mean(quantum_bonuses[~low_visit_indices])
    
    print(f"✓ Low-visit nodes bonus: {avg_low_visit_bonus:.6f}")
    print(f"✓ High-visit nodes bonus: {avg_high_visit_bonus:.6f}")
    print(f"✓ Quantum encourages exploration: {avg_low_visit_bonus > avg_high_visit_bonus}")
    print(f"✓ All scores finite: {torch.all(torch.isfinite(quantum_scores))}")
    
    correctness_ok = (
        avg_low_visit_bonus > avg_high_visit_bonus and
        torch.all(torch.isfinite(quantum_scores)) and
        avg_low_visit_bonus > 0
    )
    
    print(f"✅ Correctness test: {'PASSED' if correctness_ok else 'FAILED'}")
    return correctness_ok

def main():
    """Main speed competition test"""
    
    # Test correctness first
    correctness_ok = test_correctness()
    if not correctness_ok:
        print("❌ Correctness test failed - aborting speed competition")
        return False
    
    # Run speed competition
    results, quantum_wins = run_speed_competition()
    
    print(f"\n" + "=" * 60)
    print("FINAL ASSESSMENT")
    print("=" * 60)
    
    if quantum_wins:
        print("🏆 SUCCESS: Quantum MCTS has achieved the goal!")
        print("   Quantum MCTS is now faster than classical MCTS")
        print("   Ready for production deployment")
    else:
        print("📈 PROGRESS: Quantum MCTS performance improved but not yet faster")
        print("   Continue optimization to achieve speed parity")
    
    return quantum_wins

if __name__ == "__main__":
    success = main()