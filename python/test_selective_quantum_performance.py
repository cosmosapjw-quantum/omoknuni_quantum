"""
Performance validation for Selective Quantum MCTS
================================================

Tests the aggressive optimization with selective quantum features:
1. Performance overhead validation (< 1.5x target)
2. Exploration improvement measurement (10-15% target)
3. CUDA kernel efficiency testing
4. Batch processing throughput validation
5. Selective feature application verification
"""

import torch
import time
import math
import numpy as np
from typing import Dict, List, Tuple

from mcts.quantum.selective_quantum_optimized import (
    SelectiveQuantumMCTS, SelectiveQuantumConfig,
    create_selective_quantum_mcts, create_ultra_performance_quantum_mcts
)


def classical_v5_baseline(
    q_values: torch.Tensor,      # Q_k
    visit_counts: torch.Tensor,  # N_k
    priors: torch.Tensor,        # p_k
    kappa: float = 1.414,        # κ - exploration strength
    beta: float = 1.0,           # β - value weight
    parent_visits: float = 1000.0 # N_tot
) -> torch.Tensor:
    """v5.0 classical baseline: κ p_k (N_k/N_tot) + β Q_k (no quantum bonus)"""
    safe_visits = torch.clamp(visit_counts, min=1.0)
    
    # v5.0 Formula components
    exploration_term = kappa * priors * (safe_visits / parent_visits)
    exploitation_term = beta * q_values
    
    return exploration_term + exploitation_term


def measure_exploration_benefit(
    quantum_mcts: SelectiveQuantumMCTS,
    q_values: torch.Tensor,
    visit_counts: torch.Tensor,
    priors: torch.Tensor,
    num_trials: int = 100
) -> Dict[str, float]:
    """Measure exploration improvement from selective quantum features"""
    
    device = q_values.device
    exploration_metrics = {
        'classical_entropy': 0.0,
        'quantum_entropy': 0.0,
        'low_visit_boost': 0.0,
        'selection_diversity': 0.0
    }
    
    for _ in range(num_trials):
        # Classical v5.0 selection
        classical_scores = classical_v5_baseline(
            q_values, visit_counts, priors, 
            kappa=quantum_mcts.config.kappa, beta=quantum_mcts.config.beta
        )
        classical_probs = torch.softmax(classical_scores, dim=-1)
        classical_entropy = -torch.sum(classical_probs * torch.log(classical_probs + 1e-8))
        
        # Selective quantum selection
        quantum_scores = quantum_mcts.apply_selective_quantum(
            q_values, visit_counts, priors, simulation_count=500  # Early phase for quantum
        )
        quantum_probs = torch.softmax(quantum_scores, dim=-1)
        quantum_entropy = -torch.sum(quantum_probs * torch.log(quantum_probs + 1e-8))
        
        # Measure boost for low visit count nodes
        low_visit_mask = visit_counts < 5
        if torch.any(low_visit_mask):
            classical_low_prob = torch.sum(classical_probs[low_visit_mask])
            quantum_low_prob = torch.sum(quantum_probs[low_visit_mask])
            low_visit_boost = (quantum_low_prob - classical_low_prob) / (classical_low_prob + 1e-8)
            exploration_metrics['low_visit_boost'] += low_visit_boost.item()
        
        exploration_metrics['classical_entropy'] += classical_entropy.item()
        exploration_metrics['quantum_entropy'] += quantum_entropy.item()
    
    # Average results
    for key in exploration_metrics:
        exploration_metrics[key] /= num_trials
    
    # Calculate exploration improvement
    entropy_improvement = (exploration_metrics['quantum_entropy'] - exploration_metrics['classical_entropy']) / \
                         (exploration_metrics['classical_entropy'] + 1e-8)
    exploration_metrics['entropy_improvement_percent'] = entropy_improvement * 100
    
    return exploration_metrics


def test_performance_overhead():
    """Test performance overhead vs classical baseline"""
    print("Testing Performance Overhead...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create selective quantum MCTS
    quantum_mcts = create_ultra_performance_quantum_mcts(device=device.type)
    
    # Test data with different scenarios
    test_scenarios = [
        {'size': 50, 'iterations': 1000, 'name': 'Small batch'},
        {'size': 200, 'iterations': 500, 'name': 'Medium batch'},
        {'size': 1000, 'iterations': 200, 'name': 'Large batch'},
        {'size': 4000, 'iterations': 50, 'name': 'XL batch (CUDA kernel)'},
    ]
    
    results = {}
    
    for scenario in test_scenarios:
        size = scenario['size']
        iterations = scenario['iterations']
        name = scenario['name']
        
        print(f"\n{name} ({size} actions, {iterations} iterations):")
        
        # Generate test data
        q_values = torch.randn(size, device=device)
        visit_counts = torch.randint(1, 50, (size,), device=device)
        priors = torch.softmax(torch.randn(size, device=device), dim=0)
        
        # Classical v5.0 baseline timing
        start_time = time.perf_counter()
        for i in range(iterations):
            classical_scores = classical_v5_baseline(
                q_values, visit_counts, priors,
                kappa=quantum_mcts.config.kappa, beta=quantum_mcts.config.beta
            )
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        classical_time = time.perf_counter() - start_time
        
        # Selective quantum timing (with warmup)
        for _ in range(10):  # Warmup
            quantum_mcts.apply_selective_quantum(q_values, visit_counts, priors)
        
        start_time = time.perf_counter()
        for i in range(iterations):
            quantum_scores = quantum_mcts.apply_selective_quantum(
                q_values, visit_counts, priors, simulation_count=1000 + i
            )
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        quantum_time = time.perf_counter() - start_time
        
        # Calculate metrics
        overhead = quantum_time / classical_time
        throughput_classical = (size * iterations) / classical_time
        throughput_quantum = (size * iterations) / quantum_time
        
        results[name] = {
            'overhead': overhead,
            'classical_time': classical_time,
            'quantum_time': quantum_time,
            'throughput_classical': throughput_classical,
            'throughput_quantum': throughput_quantum
        }
        
        print(f"  Classical time: {classical_time:.4f}s")
        print(f"  Quantum time: {quantum_time:.4f}s")
        print(f"  Overhead: {overhead:.2f}x")
        print(f"  Throughput: {throughput_quantum:.0f} ops/sec")
        
        # Check performance target
        if overhead < 1.5:
            print(f"  ✓ Meets < 1.5x target")
        else:
            print(f"  ✗ Exceeds 1.5x target")
    
    return results


def test_exploration_improvement():
    """Test exploration improvement from selective quantum features"""
    print("\nTesting Exploration Improvement...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    quantum_mcts = create_selective_quantum_mcts(device=device.type)
    
    # Create test scenario with clear exploration challenge
    size = 100
    q_values = torch.randn(size, device=device) * 0.1  # Low variance values
    
    # Create visit distribution with exploration opportunity
    visit_counts = torch.ones(size, device=device)
    visit_counts[:20] = torch.randint(20, 100, (20,))  # Some well-explored actions
    visit_counts[20:] = torch.randint(1, 5, (80,))     # Many under-explored actions
    
    priors = torch.softmax(torch.randn(size, device=device), dim=0)
    
    # Measure exploration benefit
    exploration_metrics = measure_exploration_benefit(
        quantum_mcts, q_values, visit_counts, priors, num_trials=200
    )
    
    print(f"Exploration Analysis:")
    print(f"  Classical entropy: {exploration_metrics['classical_entropy']:.4f}")
    print(f"  Quantum entropy: {exploration_metrics['quantum_entropy']:.4f}")
    print(f"  Entropy improvement: {exploration_metrics['entropy_improvement_percent']:.1f}%")
    print(f"  Low visit boost: {exploration_metrics['low_visit_boost']:.3f}")
    
    # Check exploration target
    if exploration_metrics['entropy_improvement_percent'] >= 10.0:
        print(f"  ✓ Meets ≥10% exploration improvement target")
    else:
        print(f"  ⚠ Below 10% exploration improvement target")
    
    return exploration_metrics


def test_selective_feature_application():
    """Test that quantum features are applied selectively"""
    print("\nTesting Selective Feature Application...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    quantum_mcts = create_selective_quantum_mcts(device=device.type)
    
    size = 50
    q_values = torch.randn(size, device=device)
    priors = torch.softmax(torch.randn(size, device=device), dim=0)
    
    # Test different visit count scenarios
    scenarios = [
        {'visits': torch.ones(size), 'sim_count': 100, 'name': 'Early phase, low visits'},
        {'visits': torch.ones(size) * 20, 'sim_count': 100, 'name': 'Early phase, high visits'},
        {'visits': torch.ones(size), 'sim_count': 10000, 'name': 'Late phase, low visits'},
        {'visits': torch.ones(size) * 20, 'sim_count': 10000, 'name': 'Late phase, high visits'},
    ]
    
    for scenario in scenarios:
        visit_counts = scenario['visits'].to(device)
        sim_count = scenario['sim_count']
        name = scenario['name']
        
        # Reset stats
        quantum_mcts.reset_stats()
        
        # Apply quantum
        quantum_scores = quantum_mcts.apply_selective_quantum(
            q_values, visit_counts, priors, simulation_count=sim_count
        )
        
        # Get stats
        stats = quantum_mcts.get_performance_stats()
        quantum_applied = stats['quantum_ratio'] > 0
        
        # Calculate difference from classical v5.0
        classical_scores = classical_v5_baseline(
            q_values, visit_counts, priors,
            kappa=quantum_mcts.config.kappa, beta=quantum_mcts.config.beta
        )
        max_diff = torch.max(torch.abs(quantum_scores - classical_scores)).item()
        
        print(f"  {name}:")
        print(f"    Quantum applied: {quantum_applied}")
        print(f"    Max difference: {max_diff:.6f}")
        print(f"    Quantum ratio: {stats['quantum_ratio']:.3f}")


def test_batch_processing_throughput():
    """Test batch processing efficiency"""
    print("\nTesting Batch Processing Throughput...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    quantum_mcts = create_ultra_performance_quantum_mcts(device=device.type)
    
    # Large batch test
    batch_size = 1024
    num_actions = 64
    num_iterations = 100
    
    # Generate batch data
    q_values_batch = torch.randn(batch_size, num_actions, device=device)
    visit_counts_batch = torch.randint(1, 20, (batch_size, num_actions), device=device)
    priors_batch = torch.softmax(torch.randn(batch_size, num_actions, device=device), dim=-1)
    simulation_counts = torch.randint(100, 2000, (batch_size,), device=device)
    
    # Warmup
    for _ in range(10):
        quantum_mcts.batch_process(
            q_values_batch, visit_counts_batch, priors_batch, simulation_counts
        )
    
    # Timing
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        result = quantum_mcts.batch_process(
            q_values_batch, visit_counts_batch, priors_batch, simulation_counts
        )
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    total_time = time.perf_counter() - start_time
    total_operations = batch_size * num_actions * num_iterations
    throughput = total_operations / total_time
    
    print(f"  Batch size: {batch_size} x {num_actions}")
    print(f"  Total operations: {total_operations:,}")
    print(f"  Time: {total_time:.4f}s")
    print(f"  Throughput: {throughput:.0f} ops/sec")
    
    # Check result validity
    assert result.shape == (batch_size, num_actions), f"Invalid output shape: {result.shape}"
    assert torch.all(torch.isfinite(result)), "Non-finite values in result"
    print(f"  ✓ Results valid")
    
    return throughput


def main():
    """Run comprehensive performance validation"""
    print("Selective Quantum MCTS - Performance Validation")
    print("=" * 60)
    
    # Performance overhead test
    performance_results = test_performance_overhead()
    
    # Exploration improvement test
    exploration_results = test_exploration_improvement()
    
    # Selective application test
    test_selective_feature_application()
    
    # Batch processing test
    batch_throughput = test_batch_processing_throughput()
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    # Performance summary
    max_overhead = max(result['overhead'] for result in performance_results.values())
    print(f"Maximum overhead: {max_overhead:.2f}x (target: <1.5x)")
    if max_overhead < 1.5:
        print("✓ Performance target MET")
    else:
        print("✗ Performance target MISSED")
    
    # Exploration summary
    exploration_improvement = exploration_results['entropy_improvement_percent']
    print(f"Exploration improvement: {exploration_improvement:.1f}% (target: ≥10%)")
    if exploration_improvement >= 10.0:
        print("✓ Exploration target MET")
    else:
        print("⚠ Exploration target not fully met")
    
    # Throughput summary
    print(f"Batch throughput: {batch_throughput:.0f} ops/sec")
    
    print("\nSelective quantum optimization validation complete!")


if __name__ == "__main__":
    main()