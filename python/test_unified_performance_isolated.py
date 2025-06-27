"""
Isolated performance test for unified quantum MCTS
This test isolates the performance measurement to debug overhead issues
"""

import torch
import time
import math
import numpy as np

from mcts.quantum.unified_quantum_mcts_optimized import create_performance_optimized_quantum_mcts
from mcts.quantum.quantum_features_v2_ultra_optimized import UltraOptimizedQuantumMCTSV2, QuantumConfigV2

def test_direct_ultra_optimized():
    """Test the ultra-optimized version directly"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create ultra-optimized directly
    config = QuantumConfigV2(
        device=device.type,
        enable_quantum=True,
        fast_mode=True,
        branching_factor=30,
        avg_game_length=100
    )
    quantum = UltraOptimizedQuantumMCTSV2(config)
    
    # Test data
    num_iterations = 300
    q_vals = torch.randn(50, device=device)
    visits = torch.randint(1, 100, (50,), device=device)
    priors = torch.softmax(torch.randn(50, device=device), dim=0)
    
    # Classical baseline
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        c_puct = 1.414
        parent_visits = 1000
        sqrt_parent = math.sqrt(math.log(parent_visits + 1))
        exploration = c_puct * priors * sqrt_parent / torch.sqrt(visits.float() + 1)
        classical_scores = q_vals + exploration
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    classical_time = time.perf_counter() - start_time
    
    # Ultra-optimized timing
    for _ in range(10):  # Warmup
        quantum.apply_quantum_to_selection(q_vals, visits, priors, simulation_count=1000)
    
    start_time = time.perf_counter()
    for i in range(num_iterations):
        quantum_scores = quantum.apply_quantum_to_selection(
            q_vals, visits, priors, simulation_count=1000 + i
        )
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    quantum_time = time.perf_counter() - start_time
    
    overhead = quantum_time / classical_time
    print(f"Direct Ultra-Optimized:")
    print(f"  Classical time: {classical_time:.4f}s")
    print(f"  Quantum time: {quantum_time:.4f}s")
    print(f"  Overhead: {overhead:.2f}x")
    
    return overhead


def test_unified_with_fast_path():
    """Test unified version with ultra-fast path"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create unified with performance optimization
    quantum_mcts = create_performance_optimized_quantum_mcts(
        branching_factor=30, device=device.type
    )
    
    # Verify configuration
    print(f"Unified config:")
    print(f"  enable_ultra_optimization: {quantum_mcts.config.enable_ultra_optimization}")
    print(f"  enable_performance_monitoring: {quantum_mcts.config.enable_performance_monitoring}")
    print(f"  log_quantum_statistics: {quantum_mcts.config.log_quantum_statistics}")
    print(f"  enable_wave_processing: {quantum_mcts.config.enable_wave_processing}")
    
    # Test data
    num_iterations = 300
    q_vals = torch.randn(50, device=device)
    visits = torch.randint(1, 100, (50,), device=device)
    priors = torch.softmax(torch.randn(50, device=device), dim=0)
    
    # Check if wave processing will be used
    is_wave = q_vals.dim() > 1 and q_vals.shape[0] >= 32
    print(f"  Will use wave processing: {is_wave}")
    print(f"  Input shape: {q_vals.shape}")
    print(f"  Should take ultra-fast path: {quantum_mcts.config.enable_ultra_optimization and not quantum_mcts.config.enable_performance_monitoring and not quantum_mcts.config.log_quantum_statistics and not quantum_mcts.config.enable_wave_processing}")
    
    # Classical baseline
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        c_puct = 1.414
        parent_visits = 1000
        sqrt_parent = math.sqrt(math.log(parent_visits + 1))
        exploration = c_puct * priors * sqrt_parent / torch.sqrt(visits.float() + 1)
        classical_scores = q_vals + exploration
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    classical_time = time.perf_counter() - start_time
    
    # Unified timing
    for _ in range(10):  # Warmup
        quantum_mcts.apply_quantum_to_selection(q_vals, visits, priors, simulation_count=1000)
    
    start_time = time.perf_counter()
    for i in range(num_iterations):
        quantum_scores = quantum_mcts.apply_quantum_to_selection(
            q_vals, visits, priors, simulation_count=1000 + i
        )
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    quantum_time = time.perf_counter() - start_time
    
    overhead = quantum_time / classical_time
    print(f"Unified with Fast Path:")
    print(f"  Classical time: {classical_time:.4f}s")
    print(f"  Quantum time: {quantum_time:.4f}s")
    print(f"  Overhead: {overhead:.2f}x")
    
    # Get statistics
    stats = quantum_mcts.get_performance_statistics()
    print(f"  Total operations: {stats['total_operations']}")
    
    return overhead


if __name__ == "__main__":
    print("Testing quantum MCTS performance...")
    
    print("\n1. Direct Ultra-Optimized Version:")
    direct_overhead = test_direct_ultra_optimized()
    
    print("\n2. Unified Version with Fast Path:")
    unified_overhead = test_unified_with_fast_path()
    
    print(f"\nPerformance Summary:")
    print(f"  Direct ultra-optimized: {direct_overhead:.2f}x")
    print(f"  Unified with fast path: {unified_overhead:.2f}x")
    print(f"  Unified overhead vs direct: {unified_overhead/direct_overhead:.2f}x")
    
    if unified_overhead < 2.0:
        print("  ✓ Unified version meets < 2x target")
    else:
        print("  ✗ Unified version exceeds 2x target")