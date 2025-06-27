"""Final validation tests for quantum MCTS v2.0 optimizations"""

import pytest
import torch
import numpy as np
import time
from typing import Dict

from mcts.quantum.quantum_features_v2 import QuantumMCTSV2, QuantumConfigV2, MCTSPhase
from mcts.quantum.quantum_mcts_wrapper import QuantumMCTSWrapper, UnifiedQuantumConfig, compare_versions
from mcts.core.mcts import MCTS, MCTSConfig


class TestV2FinalValidation:
    """Comprehensive validation of quantum MCTS v2.0 optimizations"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_performance_targets_met(self, device):
        """Test that performance targets are achieved"""
        # Create test data
        batch_size = 64
        num_actions = 50
        num_iterations = 1000
        
        q_values = torch.randn(batch_size, num_actions, device=device)
        visit_counts = torch.randint(0, 100, (batch_size, num_actions), device=device)
        priors = torch.softmax(torch.randn(batch_size, num_actions, device=device), dim=-1)
        
        # Test classical baseline
        start = time.perf_counter()
        for _ in range(num_iterations):
            sqrt_parent = torch.sqrt(torch.log(torch.tensor(100.0, device=device)))
            visit_factor = torch.sqrt(visit_counts + 1)
            exploration = 1.414 * priors * sqrt_parent / visit_factor
            ucb_scores = q_values + exploration
        if device.type == 'cuda':
            torch.cuda.synchronize()
        classical_time = time.perf_counter() - start
        
        # Test optimized v2
        config = QuantumConfigV2(
            enable_quantum=True,
            branching_factor=50,
            device=device.type,
            cache_quantum_corrections=True,
            fast_mode=True
        )
        quantum = QuantumMCTSV2(config)
        
        # Warmup
        for _ in range(10):
            _ = quantum.apply_quantum_to_selection(
                q_values, visit_counts, priors,
                c_puct=1.414, simulation_count=1000
            )
        
        # Time v2
        start = time.perf_counter()
        for i in range(num_iterations):
            _ = quantum.apply_quantum_to_selection(
                q_values, visit_counts, priors,
                c_puct=1.414, simulation_count=1000 + i
            )
        if device.type == 'cuda':
            torch.cuda.synchronize()
        v2_time = time.perf_counter() - start
        
        overhead = v2_time / classical_time
        print(f"\nSelection overhead: {overhead:.2f}x (target: < 1.8x)")
        
        # Should meet performance target
        assert overhead < 1.8, f"Selection overhead {overhead:.2f}x exceeds target of 1.8x"
    
    def test_mathematical_correctness(self, device):
        """Test that optimizations preserve mathematical correctness"""
        config = QuantumConfigV2(
            enable_quantum=True,
            branching_factor=30,
            avg_game_length=100,
            c_puct=1.414,
            device=device.type
        )
        quantum = QuantumMCTSV2(config)
        
        # Test discrete information time
        N_values = [0, 10, 100, 1000, 10000]
        for N in N_values:
            # Check tau computation
            tau = quantum.time_evolution.information_time(N)
            expected_tau = np.log(N + 2)
            assert abs(tau - expected_tau) < 1e-6, f"τ({N}) incorrect"
            
            # Check temperature annealing
            T = quantum.time_evolution.compute_temperature(N)
            expected_T = config.initial_temperature / (expected_tau + 1e-8)
            assert abs(T - expected_T) < 1e-6, f"T({N}) incorrect"
        
        # Test phase transitions
        N_c1, N_c2 = quantum.phase_detector.compute_critical_points(
            config.branching_factor, config.c_puct, config.use_neural_prior
        )
        
        # Check phase detection
        assert quantum.phase_detector.detect_phase(10, 30, 1.414) == MCTSPhase.QUANTUM
        assert quantum.phase_detector.detect_phase(int(N_c1 + 10), 30, 1.414) == MCTSPhase.CRITICAL
        assert quantum.phase_detector.detect_phase(int(N_c2 + 10), 30, 1.414) == MCTSPhase.CLASSICAL
    
    def test_cuda_kernel_consistency(self, device):
        """Test that CUDA kernels produce consistent results"""
        if device.type != 'cuda':
            pytest.skip("CUDA not available")
        
        # Create quantum instance
        config = QuantumConfigV2(
            enable_quantum=True,
            branching_factor=50,
            device='cuda'
        )
        quantum = QuantumMCTSV2(config)
        
        # Test data
        batch_size = 32
        num_actions = 50
        
        q_values = torch.randn(batch_size, num_actions, device=device)
        visit_counts = torch.randint(0, 100, (batch_size, num_actions), device=device)
        priors = torch.softmax(torch.randn(batch_size, num_actions, device=device), dim=-1)
        
        # Run multiple times to check consistency
        results = []
        for i in range(5):
            result = quantum.apply_quantum_to_selection(
                q_values, visit_counts, priors,
                c_puct=1.414,
                parent_visits=torch.full((batch_size,), 100, device=device),
                simulation_count=1000
            )
            results.append(result)
        
        # Check consistency (allowing for small random variations)
        for i in range(1, len(results)):
            # Most values should be similar (quantum adds some randomness)
            diff = torch.abs(results[0] - results[i])
            similar_ratio = (diff < 0.1).float().mean()
            assert similar_ratio > 0.8, f"Results not consistent enough: {similar_ratio:.2f}"
    
    def test_memory_efficiency(self, device):
        """Test that memory usage is efficient"""
        if device.type != 'cuda':
            pytest.skip("CUDA memory tracking requires GPU")
        
        config = QuantumConfigV2(
            enable_quantum=True,
            branching_factor=50,
            device='cuda',
            cache_quantum_corrections=True
        )
        quantum = QuantumMCTSV2(config)
        
        # Test data
        batch_size = 64
        num_actions = 50
        
        q_values = torch.randn(batch_size, num_actions, device=device)
        visit_counts = torch.randint(0, 100, (batch_size, num_actions), device=device)
        priors = torch.softmax(torch.randn(batch_size, num_actions, device=device), dim=-1)
        
        # Warmup
        for _ in range(10):
            _ = quantum.apply_quantum_to_selection(
                q_values, visit_counts, priors,
                c_puct=1.414, simulation_count=1000
            )
        
        # Track memory
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
        
        # Run many iterations
        for i in range(1000):
            _ = quantum.apply_quantum_to_selection(
                q_values, visit_counts, priors,
                c_puct=1.414, simulation_count=1000 + i
            )
        
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        memory_increase = peak_memory - initial_memory
        
        # Should have minimal memory growth
        memory_increase_mb = memory_increase / (1024 * 1024)
        print(f"\nMemory increase over 1000 calls: {memory_increase_mb:.2f} MB")
        assert memory_increase_mb < 10, f"Excessive memory growth: {memory_increase_mb:.2f} MB"
    
    def test_version_compatibility(self, device):
        """Test that v1 and v2 produce compatible results"""
        # Create test data
        batch_size = 32
        num_actions = 50
        
        q_values = torch.randn(batch_size, num_actions, device=device)
        visit_counts = torch.randint(0, 100, (batch_size, num_actions), device=device)
        priors = torch.softmax(torch.randn(batch_size, num_actions, device=device), dim=-1)
        
        # Compare versions
        config = UnifiedQuantumConfig(
            enable_quantum=True,
            branching_factor=50,
            device=device.type
        )
        
        comparison = compare_versions(q_values, visit_counts, priors, config)
        
        print(f"\nVersion comparison:")
        print(f"  Max difference: {comparison['max_difference']:.6f}")
        print(f"  Correlation: {comparison['correlation']:.4f}")
        print(f"  v2 speedup: {comparison['speedup']:.2f}x")
        
        # Versions should produce similar results
        assert comparison['correlation'] > 0.7, f"Low correlation: {comparison['correlation']:.4f}"
        assert comparison['max_difference'] < 10.0, f"Large difference: {comparison['max_difference']:.4f}"
        
        # v2 should be faster
        assert comparison['speedup'] > 1.0, f"v2 slower than v1: {comparison['speedup']:.2f}x"
    
    def test_full_mcts_integration(self, device):
        """Test full MCTS integration performance"""
        # Skip if not on GPU (too slow on CPU)
        if device.type != 'cuda':
            pytest.skip("Full MCTS test requires GPU")
        
        # Import game state
        import alphazero_py
        
        # Test configurations
        num_simulations = 1000
        
        # Classical MCTS
        config_classical = MCTSConfig(
            num_simulations=num_simulations,
            enable_quantum=False,
            device=device.type,
            min_wave_size=3072,
            max_wave_size=3072,
            adaptive_wave_sizing=False
        )
        
        # Quantum v2 MCTS
        config_quantum = MCTSConfig(
            num_simulations=num_simulations,
            enable_quantum=True,
            quantum_version='v2',
            device=device.type,
            min_wave_size=3072,
            max_wave_size=3072,
            adaptive_wave_sizing=False,
            quantum_branching_factor=225,  # 15x15 for Gomoku
            quantum_avg_game_length=100
        )
        
        # Create game state
        game_state = alphazero_py.GomokuState()
        
        # Mock evaluator
        class MockEvaluator:
            def __init__(self):
                self.device = device
            
            def evaluate(self, states):
                batch_size = len(states) if hasattr(states, '__len__') else 1
                values = torch.zeros(batch_size, device=self.device)
                policies = torch.ones(batch_size, 225, device=self.device) / 225
                return values, policies
            
            def evaluate_batch(self, features):
                # Features is already a tensor
                batch_size = features.shape[0] if features.dim() > 2 else 1
                values = torch.zeros(batch_size, device=self.device)
                policies = torch.ones(batch_size, 225, device=self.device) / 225
                return policies, values
        
        evaluator = MockEvaluator()
        
        # Create MCTS instances
        mcts_classical = MCTS(config_classical, evaluator)
        mcts_quantum = MCTS(config_quantum, evaluator)
        
        # Warmup
        for mcts in [mcts_classical, mcts_quantum]:
            for _ in range(3):
                _ = mcts.search(game_state, num_simulations=100)
        
        # Time classical
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = mcts_classical.search(game_state, num_simulations=num_simulations)
        torch.cuda.synchronize()
        classical_time = time.perf_counter() - start
        
        # Time quantum v2
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = mcts_quantum.search(game_state, num_simulations=num_simulations)
        torch.cuda.synchronize()
        quantum_time = time.perf_counter() - start
        
        overhead = quantum_time / classical_time
        print(f"\nFull MCTS integration overhead: {overhead:.2f}x")
        print(f"Classical: {num_simulations/classical_time:.0f} sims/sec")
        print(f"Quantum v2: {num_simulations/quantum_time:.0f} sims/sec")
        
        # Should meet target
        assert overhead < 2.0, f"MCTS overhead {overhead:.2f}x exceeds target of 2.0x"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])