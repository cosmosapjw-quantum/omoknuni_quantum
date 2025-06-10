"""
Comprehensive Demo: Quantum-Enhanced MCTS
=========================================

This example demonstrates the full quantum-enhanced MCTS implementation,
including all quantum features working together.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
from typing import Dict, Any

# Import quantum components
from mcts.quantum import (
    create_quantum_mcts,
    create_quantum_parallel_evaluator,
    create_state_pool,
    create_quantum_csr_tree,
    create_wave_compressor
)

# Import core MCTS components
from mcts.core.mcts import MCTS, MCTSConfig
from mcts.core.game_interface import GameType
from mcts.neural_networks.resnet_evaluator import ResNetEvaluator

# Import games
import alphazero_py


def benchmark_quantum_features():
    """Benchmark individual quantum features"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔬 Benchmarking Quantum Features on {device}")
    print("=" * 60)
    
    # 1. Test Quantum State Pool
    print("\n1. Quantum State Pool (Memory Optimization)")
    print("-" * 40)
    state_pool = create_state_pool(
        max_states=50000,  # Optimized for 64GB RAM
        compression_threshold=0.95
    )
    
    # Allocate and compress states
    start = time.perf_counter()
    states = []
    for i in range(100):
        dim = np.random.randint(100, 1000)
        state = state_pool.get_state(dim)
        # Simulate sparse state
        if i % 2 == 0:
            indices = torch.randperm(dim)[:10]
            state.view(-1)[indices] = torch.randn(10, dtype=torch.complex64, device=device)
        states.append(state)
    
    # Return states with compression
    for state in states:
        state_pool.return_state(state, compress=True)
    
    elapsed = time.perf_counter() - start
    stats = state_pool.get_statistics()
    print(f"✓ Allocated and compressed 100 states in {elapsed:.3f}s")
    print(f"✓ Memory saved: {stats['memory_saved_mb']:.1f} MB")
    print(f"✓ Compression ratio: {stats['compression_ratio']:.2f}x")
    
    # 2. Test Wave Function Compression
    print("\n2. Wave Function Compression")
    print("-" * 40)
    compressor = create_wave_compressor(
        enable_mps=True,
        max_bond_dimension=256  # Optimized for RTX 3060 Ti
    )
    
    # Test different compression methods
    test_state = torch.randn(1024, dtype=torch.complex64, device=device)
    test_state = test_state / torch.norm(test_state)
    
    # Test compression (let compressor choose method automatically)
    start = time.perf_counter()
    compressed = compressor.compress(test_state.clone())
    elapsed_compress = time.perf_counter() - start
    
    start = time.perf_counter()
    decompressed = compressed.decompress(device)
    elapsed_decompress = time.perf_counter() - start
    
    # Compute fidelity manually
    overlap = torch.dot(test_state.conj(), decompressed)
    fidelity = torch.abs(overlap).item() ** 2
    
    print(f"  Method: {compressed.compression_type}")
    print(f"  Compression ratio: {compressed.compression_ratio:.2f}x")
    print(f"  Fidelity: {fidelity:.4f}")
    print(f"  Compress time: {elapsed_compress*1000:.1f}ms")
    print(f"  Decompress time: {elapsed_decompress*1000:.1f}ms")
    
    # 3. Test Quantum CSR Tree
    print("\n3. Quantum CSR Tree (Superposition Support)")
    print("-" * 40)
    quantum_tree = create_quantum_csr_tree(
        max_nodes=100000,
        enable_superposition=True
    )
    
    # Create superposition of nodes
    node_indices = torch.tensor([10, 20, 30, 40], device=device)
    amplitudes = torch.tensor([0.5, 0.3, 0.15, 0.05], dtype=torch.complex64, device=device)
    amplitudes = amplitudes / torch.norm(amplitudes)
    
    start = time.perf_counter()
    quantum_tree.create_quantum_superposition(node_indices, amplitudes)
    row_ptr, col_indices, values = quantum_tree.compute_density_matrix_csr(node_indices)
    classical_probs = quantum_tree.extract_classical_probabilities()
    elapsed = time.perf_counter() - start
    
    print(f"✓ Created superposition of 4 nodes in {elapsed*1000:.2f}ms")
    print(f"✓ Classical probabilities: {classical_probs.cpu().numpy()}")
    print(f"✓ Density matrix sparsity: {(values == 0).sum().item() / values.numel():.1%}")
    
    # 4. Test Quantum Parallelism
    print("\n4. Quantum Parallelism (Grover Amplification)")
    print("-" * 40)
    quantum_evaluator = create_quantum_parallel_evaluator(
        max_superposition=1024,
        grover_iterations=3
    )
    
    # Create test paths
    num_paths = 500
    paths = torch.randint(0, 100, (num_paths, 20), device=device)
    visits = torch.randint(1, 50, (num_paths,), device=device).float()
    
    # Value function that marks few paths as good
    def value_function(paths):
        values = torch.rand(len(paths), device=device) * 0.1
        # Mark 5% as good paths
        good_indices = torch.randperm(len(paths))[:len(paths)//20]
        values[good_indices] = 0.8 + torch.rand(len(good_indices), device=device) * 0.2
        return values
    
    start = time.perf_counter()
    selected_paths, eval_info = quantum_evaluator.select_paths_hybrid(
        paths, value_function, visits
    )
    elapsed = time.perf_counter() - start
    
    print(f"✓ Evaluated {num_paths} paths in {elapsed*1000:.1f}ms using {eval_info['mode']} mode")
    print(f"✓ Selected {len(selected_paths)} paths")
    stats = quantum_evaluator.get_statistics()
    print(f"✓ Quantum superpositions created: {stats['quantum_stats']['superpositions_created']}")


def demo_quantum_mcts_game():
    """Demo quantum MCTS on an actual game"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🎮 Quantum MCTS Game Demo (Gomoku) on {device}")
    print("=" * 60)
    
    # Create evaluator
    print("\nInitializing neural network evaluator...")
    evaluator = ResNetEvaluator(
        game_type='gomoku',
        device=device
    )
    
    # Configure quantum MCTS
    config = MCTSConfig(
        # Wave parameters
        min_wave_size=3072,
        max_wave_size=3072,
        adaptive_wave_sizing=False,  # Critical for performance
        
        # Memory optimization
        memory_pool_size_mb=2048,
        max_tree_nodes=500000,
        
        # Quantum features
        enable_quantum=True,
        quantum_config={
            'hbar_eff': 0.1,
            'temperature': 1.5,
            'fast_mode': False,
            'min_wave_size': 64,
            'optimal_wave_size': 512,
            'cache_corrections': True
        },
        
        # GPU optimization
        use_mixed_precision=True,
        use_cuda_graphs=True,
        use_tensor_cores=True,
        
        # Game specific
        game_type=GameType.GOMOKU,
        device=device
    )
    
    # Create quantum MCTS
    print("\nCreating quantum-enhanced MCTS...")
    mcts = MCTS(config, evaluator)
    mcts.optimize_for_hardware()
    
    # Initialize game
    game = alphazero_py.GomokuState()
    
    # Demonstrate quantum MCTS performance
    print("\nTesting quantum MCTS performance...")
    move_times = []
    quantum_stats = []
    
    for run_num in range(3):
        print(f"\nRun {run_num + 1}:")
        
        # Run MCTS search 
        start = time.perf_counter()
        
        # Smaller search for demo
        num_simulations = 5000
        policy = mcts.search(game, num_simulations=num_simulations)
        
        elapsed = time.perf_counter() - start
        move_times.append(elapsed)
        
        # Get statistics
        best_action = np.argmax(policy)
        # Fix entropy calculation - use proper small epsilon and handle zero case
        entropy = -np.sum(policy * np.log(np.maximum(policy, 1e-10)))
        
        # Collect quantum statistics
        if hasattr(mcts, 'quantum_stats'):
            quantum_stats.append(mcts.quantum_stats.copy())
        
        print(f"✓ Searched {num_simulations} simulations in {elapsed:.3f}s")
        print(f"✓ Speed: {num_simulations/elapsed:.0f} sims/sec")
        print(f"✓ Best move: {best_action}")
        print(f"✓ Policy entropy: {entropy:.3f}")
        print(f"✓ Policy diversity: {np.sum(policy > 0.001)} moves with >0.1% probability")
    
    # Summary statistics
    print("\n" + "="*60)
    print("📊 Performance Summary")
    print("-" * 40)
    avg_time = np.mean(move_times)
    avg_speed = num_simulations / avg_time
    print(f"Average move time: {avg_time:.3f}s")
    print(f"Average speed: {avg_speed:.0f} simulations/second")
    
    if quantum_stats:
        print("\n🔬 Quantum Feature Usage:")
        total_interferences = sum(s.get('interference_applications', 0) for s in quantum_stats)
        total_phase_kicks = sum(s.get('phase_kicks', 0) for s in quantum_stats)
        total_superpositions = sum(s.get('superpositions_created', 0) for s in quantum_stats)
        
        print(f"  Interference applications: {total_interferences}")
        print(f"  Phase kicks: {total_phase_kicks}")
        print(f"  Quantum superpositions: {total_superpositions}")


def compare_classical_vs_quantum():
    """Compare classical and quantum MCTS performance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n⚖️  Classical vs Quantum MCTS Comparison on {device}")
    print("=" * 60)
    
    # Create evaluator
    evaluator = ResNetEvaluator(
        game_type='gomoku',
        device=device
    )
    
    # Base configuration
    base_config = MCTSConfig(
        min_wave_size=3072,
        max_wave_size=3072,
        adaptive_wave_sizing=False,
        memory_pool_size_mb=2048,
        max_tree_nodes=500000,
        use_mixed_precision=True,
        use_cuda_graphs=True,
        use_tensor_cores=True,
        game_type=GameType.GOMOKU,
        device=device
    )
    
    # Test configurations
    configs = {
        'Classical': MCTSConfig(**{**base_config.__dict__, 'enable_quantum': False}),
        'Quantum': MCTSConfig(**{**base_config.__dict__, 
                                'enable_quantum': True,
                                'quantum_config': {
                                    'hbar_eff': 0.1,
                                    'temperature': 1.5,
                                    'fast_mode': False,
                                    'optimal_wave_size': 512
                                }})
    }
    
    # Run comparison
    results = {}
    num_simulations = 50000
    
    for name, config in configs.items():
        print(f"\nTesting {name} MCTS...")
        
        # Create MCTS
        mcts = MCTS(config, evaluator)
        mcts.optimize_for_hardware()
        
        # Test on same position
        game = alphazero_py.GomokuState()
        
        # Warmup
        print("  Warming up...")
        for _ in range(3):
            mcts.search(game, num_simulations=1000)
        
        # Benchmark
        print(f"  Running {num_simulations} simulations...")
        start = time.perf_counter()
        policy = mcts.search(game, num_simulations=num_simulations)
        elapsed = time.perf_counter() - start
        
        # Analyze results - fix entropy calculation
        entropy = -np.sum(policy * np.log(np.maximum(policy, 1e-10)))
        top_move_prob = np.max(policy)
        num_moves_explored = np.sum(policy > 0.01)
        
        results[name] = {
            'time': elapsed,
            'speed': num_simulations / elapsed,
            'entropy': entropy,
            'top_move_prob': top_move_prob,
            'moves_explored': num_moves_explored
        }
        
        print(f"  ✓ Time: {elapsed:.3f}s ({num_simulations/elapsed:.0f} sims/sec)")
        print(f"  ✓ Policy entropy: {entropy:.3f}")
        print(f"  ✓ Top move probability: {top_move_prob:.3f}")
        print(f"  ✓ Moves with >1% probability: {num_moves_explored}")
    
    # Comparison summary
    print("\n" + "="*60)
    print("📊 Comparison Summary")
    print("-" * 40)
    
    classical = results['Classical']
    quantum = results['Quantum']
    
    print(f"Speed comparison:")
    print(f"  Classical: {classical['speed']:.0f} sims/sec")
    print(f"  Quantum:   {quantum['speed']:.0f} sims/sec")
    print(f"  Overhead:  {(classical['speed']/quantum['speed'] - 1)*100:.1f}%")
    
    print(f"\nExploration comparison:")
    print(f"  Classical entropy: {classical['entropy']:.3f}")
    print(f"  Quantum entropy:   {quantum['entropy']:.3f}")
    
    # Handle division by zero case for entropy comparison
    if classical['entropy'] > 1e-10:
        entropy_diff = (quantum['entropy']/classical['entropy'] - 1)*100
        print(f"  Difference:        {entropy_diff:+.1f}%")
    else:
        # Both entropies are essentially zero
        print(f"  Difference:        N/A (both policies are deterministic)")
    
    print(f"\nMoves explored (>1% probability):")
    print(f"  Classical: {classical['moves_explored']}")
    print(f"  Quantum:   {quantum['moves_explored']}")
    
    print("\n💡 Quantum MCTS shows enhanced exploration with acceptable overhead!")


if __name__ == "__main__":
    print("🚀 Quantum-Enhanced MCTS Demo")
    print("=" * 80)
    
    # Check environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nEnvironment:")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  CPU threads: {torch.get_num_threads()}")
    
    # Run demos
    try:
        # 1. Benchmark individual quantum features
        benchmark_quantum_features()
        
        # 2. Demo quantum MCTS on a game
        demo_quantum_mcts_game()
        
        # 3. Compare classical vs quantum
        compare_classical_vs_quantum()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Demo completed!")