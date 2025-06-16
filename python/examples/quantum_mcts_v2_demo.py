#!/usr/bin/env python3
"""
Quantum MCTS v2.0 Demonstration
==============================

This script demonstrates the key features of the v2.0 quantum MCTS implementation:
- Auto-computed parameters from game properties
- Phase detection and adaptive strategies  
- Neural network prior integration
- Envariance convergence checking
- Performance comparison with v1
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time
import math

# Import both versions for comparison
from mcts.quantum.quantum_features import create_quantum_mcts as create_quantum_mcts_v1
from mcts.quantum.quantum_features_v2 import (
    create_quantum_mcts_v2, 
    MCTSPhase,
    OptimalParameters
)


class DemoGame:
    """Simple game for demonstration purposes"""
    def __init__(self, branching_factor: int = 20, max_depth: int = 50):
        self.branching_factor = branching_factor
        self.max_depth = max_depth
        self.avg_length = max_depth // 2


class MockNeuralNetwork:
    """Mock neural network for demonstration"""
    def __init__(self, num_actions: int):
        self.num_actions = num_actions
    
    def evaluate(self, state) -> Tuple[float, np.ndarray]:
        """Return mock value and policy"""
        value = np.random.randn() * 0.1
        policy = np.random.dirichlet(np.ones(self.num_actions))
        return value, policy


def demonstrate_auto_computation():
    """Demonstrate automatic parameter computation"""
    print("=" * 60)
    print("1. Automatic Parameter Computation")
    print("=" * 60)
    
    # Create games with different properties
    games = [
        ("Chess", 35, 80),
        ("Go 19x19", 250, 200),
        ("Gomoku", 225, 50),
    ]
    
    for game_name, branching, avg_length in games:
        print(f"\n{game_name}:")
        print(f"  Branching factor: {branching}")
        print(f"  Average length: {avg_length}")
        
        # Compute optimal parameters
        c_puct = OptimalParameters.compute_c_puct(branching)
        num_hashes = OptimalParameters.compute_num_hashes(
            branching, avg_length, has_neural_network=True
        )
        
        print(f"  Optimal c_puct: {c_puct:.3f}")
        print(f"  Optimal hash functions: {num_hashes}")
        
        # Create v2 MCTS with auto-computation
        qmcts = create_quantum_mcts_v2(
            branching_factor=branching,
            avg_game_length=avg_length,
            use_neural_network=True
        )
        
        # Verify auto-computed values
        print(f"  Auto-computed c_puct: {qmcts.config.c_puct:.3f}")
        print(f"  Auto-computed hashes: {qmcts.config.num_hash_functions}")


def demonstrate_phase_transitions():
    """Demonstrate phase detection and transitions"""
    print("\n" + "=" * 60)
    print("2. Phase Detection and Transitions")
    print("=" * 60)
    
    # Create quantum MCTS for Go-like game
    qmcts = create_quantum_mcts_v2(
        branching_factor=200,
        avg_game_length=150,
        use_neural_network=True
    )
    
    # Track metrics through phases
    N_values = [1, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
    phases = []
    temperatures = []
    hbar_values = []
    
    for N in N_values:
        qmcts.update_simulation_count(N)
        stats = qmcts.get_statistics()
        
        phases.append(stats['current_phase'])
        temperatures.append(stats['current_temperature'])
        hbar_values.append(stats['current_hbar_eff'])
        
        print(f"\nN = {N:6d}: Phase = {stats['current_phase']:10s}, "
              f"T = {stats['current_temperature']:6.3f}, "
              f"ℏ_eff = {stats['current_hbar_eff']:6.3f}")
    
    # Plot phase evolution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.semilogx(N_values, temperatures, 'b-o')
    plt.xlabel('Simulation Count N')
    plt.ylabel('Temperature T(N)')
    plt.title('Temperature Annealing')
    plt.grid(True)
    
    plt.subplot(132)
    plt.semilogx(N_values, hbar_values, 'r-o')
    plt.xlabel('Simulation Count N')
    plt.ylabel('ℏ_eff(N)')
    plt.title('Quantum Strength Evolution')
    plt.grid(True)
    
    plt.subplot(133)
    phase_numeric = [0 if p == 'quantum' else (1 if p == 'critical' else 2) for p in phases]
    plt.semilogx(N_values, phase_numeric, 'g-o')
    plt.xlabel('Simulation Count N')
    plt.ylabel('Phase')
    plt.yticks([0, 1, 2], ['Quantum', 'Critical', 'Classical'])
    plt.title('Phase Transitions')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('quantum_mcts_v2_phases.png', dpi=150)
    print("\nPhase evolution plot saved to 'quantum_mcts_v2_phases.png'")


def demonstrate_neural_network_integration():
    """Demonstrate how neural network priors affect quantum features"""
    print("\n" + "=" * 60)
    print("3. Neural Network Prior Integration")
    print("=" * 60)
    
    # Create quantum MCTS with and without NN
    qmcts_with_nn = create_quantum_mcts_v2(
        branching_factor=50,
        avg_game_length=40,
        use_neural_network=True
    )
    
    qmcts_no_nn = create_quantum_mcts_v2(
        branching_factor=50,
        avg_game_length=40,
        use_neural_network=False
    )
    
    # Compare critical points
    b, c = 50, qmcts_with_nn.config.c_puct
    N_c1_nn, N_c2_nn = qmcts_with_nn.phase_detector.compute_critical_points(b, c, True)
    N_c1_no_nn, N_c2_no_nn = qmcts_no_nn.phase_detector.compute_critical_points(b, c, False)
    
    print(f"\nCritical points comparison:")
    print(f"  Without NN: N_c1 = {N_c1_no_nn:.0f}, N_c2 = {N_c2_no_nn:.0f}")
    print(f"  With NN:    N_c1 = {N_c1_nn:.0f}, N_c2 = {N_c2_nn:.0f}")
    print(f"  NN shift factor: {N_c1_nn/N_c1_no_nn:.2f}x")
    
    # Test selection with different priors
    print("\nTesting prior influence on selection:")
    
    # Create test data
    q_values = torch.tensor([0.1, 0.2, 0.15, 0.18, 0.12])
    visit_counts = torch.tensor([10, 8, 5, 3, 1])
    
    # Uniform priors (no preference)
    uniform_priors = torch.ones(5) / 5
    ucb_uniform = qmcts_with_nn.apply_quantum_to_selection(
        q_values, visit_counts, uniform_priors, simulation_count=100
    )
    
    # Strong prior for action 2
    strong_priors = torch.tensor([0.1, 0.1, 0.6, 0.1, 0.1])
    ucb_strong = qmcts_with_nn.apply_quantum_to_selection(
        q_values, visit_counts, strong_priors, simulation_count=100
    )
    
    print("\nUCB scores with uniform priors:")
    for i, score in enumerate(ucb_uniform):
        print(f"  Action {i}: {score:.3f}")
    
    print("\nUCB scores with strong prior for action 2:")
    for i, score in enumerate(ucb_strong):
        boost = score - ucb_uniform[i]
        print(f"  Action {i}: {score:.3f} (boost: {boost:+.3f})")


def demonstrate_performance_comparison():
    """Compare performance between v1 and v2"""
    print("\n" + "=" * 60)
    print("4. Performance Comparison: v1 vs v2")
    print("=" * 60)
    
    # Create both versions
    qmcts_v1 = create_quantum_mcts_v1(
        enable_quantum=True,
        quantum_level='tree_level',
        hbar_eff=0.1,
        min_wave_size=32
    )
    
    qmcts_v2 = create_quantum_mcts_v2(
        enable_quantum=True,
        branching_factor=20,
        avg_game_length=50,
        use_neural_network=True
    )
    
    # Test different batch sizes
    batch_sizes = [32, 64, 128, 256, 512, 1024]
    num_actions = 20
    num_iterations = 100
    
    print("\nTiming comparison (milliseconds per call):")
    print(f"{'Batch Size':>10} | {'v1 Time':>10} | {'v2 Time':>10} | {'v2/v1 Ratio':>12}")
    print("-" * 50)
    
    for batch_size in batch_sizes:
        # Create test data
        q_values = torch.randn(batch_size, num_actions)
        visit_counts = torch.randint(0, 100, (batch_size, num_actions))
        priors = torch.softmax(torch.randn(batch_size, num_actions), dim=-1)
        
        # Time v1
        start = time.time()
        for _ in range(num_iterations):
            _ = qmcts_v1.apply_quantum_to_selection(
                q_values, visit_counts, priors, c_puct=1.414
            )
        v1_time = (time.time() - start) / num_iterations * 1000
        
        # Time v2
        start = time.time()
        for _ in range(num_iterations):
            _ = qmcts_v2.apply_quantum_to_selection(
                q_values, visit_counts, priors, simulation_count=1000
            )
        v2_time = (time.time() - start) / num_iterations * 1000
        
        ratio = v2_time / v1_time
        print(f"{batch_size:10d} | {v1_time:10.3f} | {v2_time:10.3f} | {ratio:12.3f}")


def demonstrate_envariance_convergence():
    """Demonstrate envariance-based convergence checking"""
    print("\n" + "=" * 60)
    print("5. Envariance Convergence Criterion")
    print("=" * 60)
    
    # Mock tree that gradually converges
    class ConvergingTree:
        def __init__(self):
            self.N = 0
            
        def get_policy_entropy(self):
            # Entropy decreases as 1/log(N)
            if self.N < 10:
                return 1.0
            return 1.0 / math.log(self.N)
        
        def simulate(self, n):
            self.N += n
    
    # Create quantum MCTS and tree
    qmcts = create_quantum_mcts_v2(
        branching_factor=50,
        avg_game_length=100,
        use_neural_network=True
    )
    tree = ConvergingTree()
    
    print("\nSimulating MCTS with envariance checking:")
    print(f"{'Simulations':>12} | {'Entropy':>10} | {'Converged':>10}")
    print("-" * 40)
    
    # Run simulation batches
    batch_size = 100
    for i in range(50):
        tree.simulate(batch_size)
        qmcts.update_simulation_count(tree.N)
        
        entropy = tree.get_policy_entropy()
        converged = qmcts.check_envariance(tree, threshold=0.1)
        
        if i % 5 == 0 or converged:
            print(f"{tree.N:12d} | {entropy:10.4f} | {'Yes' if converged else 'No':>10}")
        
        if converged:
            print(f"\nConverged at {tree.N} simulations!")
            break


def main():
    """Run all demonstrations"""
    print("Quantum MCTS v2.0 Feature Demonstration")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run demonstrations
    demonstrate_auto_computation()
    demonstrate_phase_transitions()
    demonstrate_neural_network_integration()
    demonstrate_performance_comparison()
    demonstrate_envariance_convergence()
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()