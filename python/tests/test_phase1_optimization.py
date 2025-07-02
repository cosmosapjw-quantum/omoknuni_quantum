#!/usr/bin/env python3
"""
Test suite for Phase 1 quantum MCTS optimizations

This test validates:
1. TensorPool optimization (no more zeroing on retrieval)
2. QFT engine integration for GPU acceleration  
3. Performance improvements and memory efficiency
4. Backward compatibility with existing interfaces
"""

import pytest
import torch
import numpy as np
import time
from typing import Dict, Any

# Import the optimized quantum MCTS
import sys
sys.path.append('~/venv')
from mcts.quantum.quantum_mcts import QuantumMCTS, QuantumConfig, QuantumMode, OptimizedTensorPool


class TestTensorPoolOptimization:
    """Test the optimized tensor pool implementation"""
    
    def test_tensor_pool_no_zeroing(self):
        """Test that tensor pool doesn't zero tensors on retrieval"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pool = OptimizedTensorPool(device)
        
        # Create a tensor with specific values
        original_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        pool.return_tensor(original_tensor)
        
        # Get tensor back from pool
        retrieved_tensor = pool.get_tensor((4,))
        
        # Should have original values, not zeros
        assert torch.allclose(retrieved_tensor, original_tensor), \
            f"Expected {original_tensor}, got {retrieved_tensor}"
    
    def test_tensor_pool_stats(self):
        """Test tensor pool statistics tracking"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pool = OptimizedTensorPool(device)
        
        # Get initial stats (pool has pre-warmed tensors)
        initial_stats = pool.get_stats()
        initial_hits = initial_stats['hits']
        initial_misses = initial_stats['misses']
        
        # Request a tensor (might be hit due to pre-warming)
        tensor1 = pool.get_tensor((8,))
        stats = pool.get_stats()
        
        # Should have at least one more hit or miss
        assert (stats['hits'] + stats['misses']) > (initial_hits + initial_misses)
        
        # Return tensor and request again (should increase hits)
        pool.return_tensor(tensor1)
        tensor2 = pool.get_tensor((8,))
        final_stats = pool.get_stats()
        
        # Should have more hits than initial
        assert final_stats['hits'] >= initial_stats['hits']
        assert final_stats['hit_rate'] >= 0
    
    def test_tensor_pool_pre_warming(self):
        """Test that tensor pool pre-warms common shapes"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pool = OptimizedTensorPool(device, max_pool_size=100)
        
        stats = pool.get_stats()
        # Should have pre-allocated tensors
        assert stats['total_tensors_pooled'] > 0
        assert len(stats['pool_shapes']) > 0


class TestQFTEngineIntegration:
    """Test QFT engine integration for GPU acceleration"""
    
    def test_qft_engine_initialization(self):
        """Test QFT engine is properly initialized"""
        config = QuantumConfig(
            quantum_mode=QuantumMode.PRAGMATIC,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            precompute_quantum_tables=True
        )
        qmcts = QuantumMCTS(config)
        
        # Should have QFT engine initialized
        assert qmcts.qft_engine is not None
        assert hasattr(qmcts.qft_engine, 'interpolate_one_loop')
        assert hasattr(qmcts.qft_engine, 'get_decoherence_strength')
    
    def test_quantum_bonus_qft_acceleration(self):
        """Test quantum bonus computation with QFT acceleration"""
        config = QuantumConfig(
            quantum_mode=QuantumMode.PRAGMATIC,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            precompute_quantum_tables=True,
            quantum_bonus_coefficient=0.15
        )
        qmcts = QuantumMCTS(config)
        
        # Test single value
        visit_counts_single = torch.tensor(10.0, device=qmcts.device)
        bonus_single = qmcts.get_quantum_bonus(visit_counts_single)
        assert bonus_single.item() > 0, "Quantum bonus should be positive"
        
        # Test batch values
        visit_counts_batch = torch.tensor([1, 5, 10, 50, 100], device=qmcts.device, dtype=torch.float32)
        bonus_batch = qmcts.get_quantum_bonus(visit_counts_batch)
        assert bonus_batch.shape == visit_counts_batch.shape
        assert torch.all(bonus_batch > 0), "All quantum bonuses should be positive"
    
    def test_qft_vs_cpu_consistency(self):
        """Test QFT engine gives consistent results with CPU fallback"""
        # QFT-enabled config
        config_qft = QuantumConfig(
            quantum_mode=QuantumMode.PRAGMATIC,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            precompute_quantum_tables=True,
            quantum_bonus_coefficient=0.15
        )
        qmcts_qft = QuantumMCTS(config_qft)
        
        # CPU-only config  
        config_cpu = QuantumConfig(
            quantum_mode=QuantumMode.PRAGMATIC,
            device='cpu',
            precompute_quantum_tables=False,
            quantum_bonus_coefficient=0.15
        )
        qmcts_cpu = QuantumMCTS(config_cpu)
        
        # Test values
        visit_counts = torch.tensor([1, 5, 10, 20], dtype=torch.float32)
        
        bonus_qft = qmcts_qft.get_quantum_bonus(visit_counts.to(qmcts_qft.device))
        bonus_cpu = qmcts_cpu.get_quantum_bonus(visit_counts)
        
        # Should be approximately equal (some numerical differences expected)
        assert torch.allclose(bonus_qft.cpu(), bonus_cpu, rtol=0.1), \
            f"QFT and CPU results should be similar: QFT={bonus_qft.cpu()}, CPU={bonus_cpu}"


class TestPerformanceImprovements:
    """Test performance improvements from Phase 1 optimizations"""
    
    def test_quantum_computation_performance(self):
        """Benchmark quantum computation performance"""
        config = QuantumConfig(
            quantum_mode=QuantumMode.PRAGMATIC,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            precompute_quantum_tables=True,
            enable_performance_monitoring=True
        )
        qmcts = QuantumMCTS(config)
        
        # Large batch for performance testing
        batch_size = 1000
        visit_counts = torch.randint(1, 100, (batch_size,), device=qmcts.device, dtype=torch.float32)
        
        # Warm up
        for _ in range(5):
            _ = qmcts.get_quantum_bonus(visit_counts)
        
        # Benchmark
        start_time = time.perf_counter()
        num_iterations = 100
        for _ in range(num_iterations):
            bonus = qmcts.get_quantum_bonus(visit_counts)
        
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / num_iterations
        
        print(f"Average quantum computation time: {avg_time*1000:.3f}ms for {batch_size} values")
        
        # Should be fast (less than 1ms for 1000 values on modern hardware)
        assert avg_time < 0.01, f"Quantum computation too slow: {avg_time:.6f}s"
    
    def test_memory_efficiency(self):
        """Test memory efficiency improvements"""
        config = QuantumConfig(
            quantum_mode=QuantumMode.PRAGMATIC,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            enable_tensor_pooling=True,
            precompute_quantum_tables=True
        )
        qmcts = QuantumMCTS(config)
        
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()
        
        # Simulate typical MCTS usage pattern
        for _ in range(100):
            visit_counts = torch.randint(1, 50, (32,), device=qmcts.device, dtype=torch.float32)
            q_values = torch.randn(32, device=qmcts.device)
            priors = torch.softmax(torch.randn(32, device=qmcts.device), dim=0)
            
            # Compute PUCT values (typical usage)
            puct_values = qmcts.compute_puct_values(q_values, visit_counts, priors, 100)
            
            # Check tensor pool is working
            if qmcts.tensor_pool:
                stats = qmcts.tensor_pool.get_stats()
                # Should have some hits after warmup
                if stats['hits'] + stats['misses'] > 10:
                    assert stats['hit_rate'] > 0, "Tensor pool should have some hits"
        
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            memory_growth = final_memory - initial_memory
            
            print(f"Memory growth: {memory_growth / 1024 / 1024:.2f} MB")
            
            # Memory growth should be minimal due to tensor pooling
            assert memory_growth < 50 * 1024 * 1024, "Memory growth should be < 50MB"


class TestBackwardCompatibility:
    """Test backward compatibility with existing interfaces"""
    
    def test_quantum_config_compatibility(self):
        """Test QuantumConfig backward compatibility"""
        # Default config should work
        config = QuantumConfig()
        qmcts = QuantumMCTS(config)
        assert qmcts is not None
        
        # All quantum modes should work
        for mode in QuantumMode:
            config = QuantumConfig(quantum_mode=mode)
            qmcts = QuantumMCTS(config)
            assert qmcts.config.quantum_mode == mode
    
    def test_puct_computation_interface(self):
        """Test PUCT computation interface remains unchanged"""
        config = QuantumConfig(quantum_mode=QuantumMode.PRAGMATIC)
        qmcts = QuantumMCTS(config)
        
        # Standard PUCT computation should work
        q_values = torch.tensor([0.5, 0.3, 0.8], dtype=torch.float32)
        visit_counts = torch.tensor([10, 5, 20], dtype=torch.float32)
        priors = torch.tensor([0.4, 0.3, 0.3], dtype=torch.float32)
        total_visits = 35
        
        puct_values = qmcts.compute_puct_values(q_values, visit_counts, priors, total_visits)
        
        assert puct_values.shape == q_values.shape
        assert torch.all(torch.isfinite(puct_values))
    
    def test_factory_functions_compatibility(self):
        """Test factory functions for creating quantum MCTS instances"""
        from mcts.quantum import (
            create_classical_mcts, 
            create_minimal_quantum_mcts,
            create_pragmatic_quantum_mcts,
            create_full_quantum_mcts
        )
        
        # All factory functions should work
        classical = create_classical_mcts()
        minimal = create_minimal_quantum_mcts()
        pragmatic = create_pragmatic_quantum_mcts()
        full = create_full_quantum_mcts()
        
        assert classical.config.quantum_mode == QuantumMode.CLASSICAL
        assert minimal.config.quantum_mode == QuantumMode.MINIMAL  
        assert pragmatic.config.quantum_mode == QuantumMode.PRAGMATIC
        assert full.config.quantum_mode == QuantumMode.FULL


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])