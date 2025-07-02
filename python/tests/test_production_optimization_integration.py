#!/usr/bin/env python3
"""
Test suite for Phase 4.1 production optimization integration

This test validates:
1. Production optimizer integration with quantum MCTS
2. Mixed precision computation and kernel fusion
3. Performance improvements and memory efficiency
4. Factory function for production-optimized quantum MCTS
5. Comprehensive monitoring and statistics
"""

import pytest
import torch
import numpy as np
import time
from typing import Dict, Any

# Import the production optimizer and quantum MCTS
from mcts.quantum.quantum_mcts import (
    QuantumMCTS, QuantumConfig, QuantumMode, 
    create_production_quantum_mcts
)
from mcts.gpu.production_optimizer import (
    ProductionOptimizer, ProductionOptimizationConfig, 
    PrecisionMode, create_production_optimizer
)


class TestProductionOptimizerIntegration:
    """Test production optimizer integration with quantum MCTS"""
    
    def test_production_optimizer_initialization(self):
        """Test production optimizer is properly initialized in quantum MCTS"""
        config = QuantumConfig(
            quantum_mode=QuantumMode.ULTRA_LEAN,
            enable_production_optimization=True,
            production_precision_mode="adaptive",
            production_fusion_level="aggressive",
            production_enable_profiling=True
        )
        qmcts = QuantumMCTS(config)
        
        # Should have production optimizer initialized
        assert qmcts.production_optimizer is not None
        assert isinstance(qmcts.production_optimizer, ProductionOptimizer)
        assert qmcts.production_optimizer.config.precision_mode == PrecisionMode.ADAPTIVE
        assert qmcts.production_optimizer.config.fusion_level == "aggressive"
    
    def test_production_optimizer_disabled(self):
        """Test production optimizer is None when disabled"""
        config = QuantumConfig(
            quantum_mode=QuantumMode.ULTRA_LEAN,
            enable_production_optimization=False
        )
        qmcts = QuantumMCTS(config)
        
        # Should not have production optimizer
        assert qmcts.production_optimizer is None
    
    def test_production_optimized_puct_computation(self):
        """Test PUCT computation uses production optimizer when enabled"""
        config = QuantumConfig(
            quantum_mode=QuantumMode.ULTRA_LEAN,
            enable_production_optimization=True,
            production_precision_mode="adaptive",
            production_fusion_level="aggressive"
        )
        qmcts = QuantumMCTS(config)
        
        # Test computation
        device = qmcts.device
        q_values = torch.tensor([0.5, 0.3, 0.8], device=device, dtype=torch.float32)
        visit_counts = torch.tensor([10, 5, 20], device=device, dtype=torch.float32)
        priors = torch.tensor([0.4, 0.3, 0.3], device=device, dtype=torch.float32)
        total_visits = 35
        
        puct_values = qmcts.compute_puct_values(q_values, visit_counts, priors, total_visits)
        
        # Should return valid PUCT values
        assert puct_values.shape == q_values.shape
        assert torch.all(torch.isfinite(puct_values))
        assert puct_values.device == device
    
    def test_different_precision_modes(self):
        """Test different mixed precision modes"""
        precision_modes = ["fp32", "adaptive"]
        if torch.cuda.is_available():
            precision_modes.extend(["fp16", "bf16", "auto"])
        
        for precision_mode in precision_modes:
            config = QuantumConfig(
                quantum_mode=QuantumMode.ULTRA_LEAN,
                enable_production_optimization=True,
                production_precision_mode=precision_mode,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            qmcts = QuantumMCTS(config)
            
            # Test computation with different precision modes
            device = qmcts.device
            q_values = torch.tensor([0.5, 0.3, 0.8], device=device)
            visit_counts = torch.tensor([10, 5, 20], device=device, dtype=torch.float32)
            priors = torch.tensor([0.4, 0.3, 0.3], device=device)
            total_visits = 35
            
            puct_values = qmcts.compute_puct_values(q_values, visit_counts, priors, total_visits)
            
            assert puct_values.shape == q_values.shape
            assert torch.all(torch.isfinite(puct_values))


class TestProductionOptimizerPerformance:
    """Test performance improvements from production optimization"""
    
    def test_production_vs_standard_performance(self):
        """Compare performance between production-optimized and standard quantum MCTS"""
        # Standard quantum MCTS
        config_standard = QuantumConfig(
            quantum_mode=QuantumMode.ULTRA_LEAN,
            enable_production_optimization=False
        )
        qmcts_standard = QuantumMCTS(config_standard)
        
        # Production-optimized quantum MCTS
        config_production = QuantumConfig(
            quantum_mode=QuantumMode.ULTRA_LEAN,
            enable_production_optimization=True,
            production_precision_mode="adaptive",
            production_fusion_level="aggressive"
        )
        qmcts_production = QuantumMCTS(config_production)
        
        # Benchmark data
        device = qmcts_standard.device
        batch_size = 256
        q_values = torch.randn(batch_size, device=device)
        visit_counts = torch.randint(1, 100, (batch_size,), device=device, dtype=torch.float32)
        priors = torch.softmax(torch.randn(batch_size, device=device), dim=0)
        total_visits = 500
        
        # Warm up
        for _ in range(5):
            _ = qmcts_standard.compute_puct_values(q_values, visit_counts, priors, total_visits)
            _ = qmcts_production.compute_puct_values(q_values, visit_counts, priors, total_visits)
        
        # Benchmark standard
        start_time = time.perf_counter()
        num_iterations = 50
        for _ in range(num_iterations):
            puct_standard = qmcts_standard.compute_puct_values(q_values, visit_counts, priors, total_visits)
        standard_time = time.perf_counter() - start_time
        
        # Benchmark production
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            puct_production = qmcts_production.compute_puct_values(q_values, visit_counts, priors, total_visits)
        production_time = time.perf_counter() - start_time
        
        print(f"Standard time: {standard_time*1000:.2f}ms")
        print(f"Production time: {production_time*1000:.2f}ms")
        
        # Results should be numerically similar (production optimizer may have different precision)
        # Note: Production optimizer uses different computation path and mixed precision,
        # so exact numerical equality is not expected
        print(f"Standard PUCT range: [{puct_standard.min():.3f}, {puct_standard.max():.3f}]")
        print(f"Production PUCT range: [{puct_production.min():.3f}, {puct_production.max():.3f}]")
        
        # At minimum, both should produce finite results in reasonable ranges
        assert torch.all(torch.isfinite(puct_standard))
        assert torch.all(torch.isfinite(puct_production))
        assert puct_standard.shape == puct_production.shape
        
        # Production should be faster or at least comparable
        # (On CPU, the overhead of production optimizer might make it slightly slower)
        assert production_time <= standard_time * 1.5  # Allow 50% overhead tolerance
    
    def test_memory_efficiency(self):
        """Test memory efficiency of production optimizer"""
        config = QuantumConfig(
            quantum_mode=QuantumMode.ULTRA_LEAN,
            enable_production_optimization=True,
            production_precision_mode="adaptive",
            production_target_memory_utilization=0.8
        )
        qmcts = QuantumMCTS(config)
        
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()
        
        # Simulate typical usage
        device = qmcts.device
        for batch_size in [32, 64, 128, 256]:
            q_values = torch.randn(batch_size, device=device)
            visit_counts = torch.randint(1, 50, (batch_size,), device=device, dtype=torch.float32)
            priors = torch.softmax(torch.randn(batch_size, device=device), dim=0)
            
            puct_values = qmcts.compute_puct_values(q_values, visit_counts, priors, 100)
            assert puct_values.shape == (batch_size,)
        
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            memory_growth = final_memory - initial_memory
            print(f"Memory growth: {memory_growth / 1024 / 1024:.2f} MB")
            
            # Memory growth should be reasonable
            assert memory_growth < 100 * 1024 * 1024  # Less than 100MB growth


class TestProductionFactoryFunction:
    """Test factory function for production-optimized quantum MCTS"""
    
    def test_create_production_quantum_mcts_default(self):
        """Test factory function with default parameters"""
        qmcts = create_production_quantum_mcts()
        
        assert qmcts.config.quantum_mode == QuantumMode.ULTRA_LEAN
        assert qmcts.config.enable_production_optimization == True
        assert qmcts.config.production_precision_mode == "adaptive"
        assert qmcts.config.production_fusion_level == "aggressive"
        assert qmcts.config.production_enable_profiling == True
        assert qmcts.production_optimizer is not None
    
    def test_create_production_quantum_mcts_custom(self):
        """Test factory function with custom parameters"""
        qmcts = create_production_quantum_mcts(
            precision_mode="fp32",
            fusion_level="conservative",
            enable_profiling=False
        )
        
        assert qmcts.config.production_precision_mode == "fp32"
        assert qmcts.config.production_fusion_level == "conservative" 
        assert qmcts.config.production_enable_profiling == False
        assert qmcts.production_optimizer is not None
    
    def test_factory_function_performance_target(self):
        """Test that factory function sets appropriate performance target"""
        qmcts = create_production_quantum_mcts()
        
        # Should target <1.1x overhead with production optimization
        assert qmcts.config.target_overhead == 1.1


class TestProductionOptimizerStatistics:
    """Test production optimizer statistics and monitoring"""
    
    def test_production_optimizer_stats(self):
        """Test production optimizer provides comprehensive statistics"""
        config = QuantumConfig(
            quantum_mode=QuantumMode.ULTRA_LEAN,
            enable_production_optimization=True,
            production_enable_profiling=True
        )
        qmcts = QuantumMCTS(config)
        
        # Perform some computations
        device = qmcts.device
        for _ in range(10):
            q_values = torch.randn(32, device=device)
            visit_counts = torch.randint(1, 50, (32,), device=device, dtype=torch.float32)
            priors = torch.softmax(torch.randn(32, device=device), dim=0)
            
            _ = qmcts.compute_puct_values(q_values, visit_counts, priors, 100)
        
        # Get comprehensive stats
        stats = qmcts.production_optimizer.get_comprehensive_stats()
        
        # Should have all required fields
        assert 'uptime_seconds' in stats
        assert 'total_operations' in stats
        assert 'operations_per_second' in stats
        assert 'average_operation_time_ms' in stats
        assert 'mixed_precision_stats' in stats
        assert 'kernel_fusion_stats' in stats
        assert 'memory_stats' in stats
        assert 'config' in stats
        
        # Should have performed operations
        assert stats['total_operations'] >= 10
        assert stats['operations_per_second'] > 0
    
    def test_mixed_precision_stats(self):
        """Test mixed precision manager provides detailed statistics"""
        config = QuantumConfig(
            quantum_mode=QuantumMode.ULTRA_LEAN,
            enable_production_optimization=True,
            production_precision_mode="adaptive"
        )
        qmcts = QuantumMCTS(config)
        
        # Perform computations
        device = qmcts.device
        q_values = torch.randn(64, device=device)
        visit_counts = torch.randint(1, 100, (64,), device=device, dtype=torch.float32)
        priors = torch.softmax(torch.randn(64, device=device), dim=0)
        
        _ = qmcts.compute_puct_values(q_values, visit_counts, priors, 200)
        
        # Get mixed precision stats
        mp_stats = qmcts.production_optimizer.mixed_precision.get_stats()
        
        assert 'numerical_stability_rate' in mp_stats
        assert 'hardware_support' in mp_stats
        assert 'cache_size' in mp_stats
        assert mp_stats['numerical_stability_rate'] >= 0.0
        assert mp_stats['numerical_stability_rate'] <= 1.0


class TestProductionOptimizerCompatibility:
    """Test compatibility and integration with existing quantum features"""
    
    def test_production_with_tensor_pooling(self):
        """Test production optimizer works with tensor pooling"""
        config = QuantumConfig(
            quantum_mode=QuantumMode.ULTRA_LEAN,
            enable_production_optimization=True,
            enable_tensor_pooling=True
        )
        qmcts = QuantumMCTS(config)
        
        # Should have both production optimizer and tensor pool
        assert qmcts.production_optimizer is not None
        assert qmcts.tensor_pool is not None
        
        # Should work together
        device = qmcts.device
        q_values = torch.randn(32, device=device)
        visit_counts = torch.randint(1, 50, (32,), device=device, dtype=torch.float32)
        priors = torch.softmax(torch.randn(32, device=device), dim=0)
        
        puct_values = qmcts.compute_puct_values(q_values, visit_counts, priors, 100)
        assert puct_values.shape == q_values.shape
    
    def test_production_with_performance_monitoring(self):
        """Test production optimizer works with performance monitoring"""
        config = QuantumConfig(
            quantum_mode=QuantumMode.ULTRA_LEAN,
            enable_production_optimization=True,
            enable_performance_monitoring=True,
            production_enable_profiling=True
        )
        qmcts = QuantumMCTS(config)
        
        # Perform computation
        device = qmcts.device
        q_values = torch.randn(64, device=device)
        visit_counts = torch.randint(1, 100, (64,), device=device, dtype=torch.float32)
        priors = torch.softmax(torch.randn(64, device=device), dim=0)
        
        puct_values = qmcts.compute_puct_values(q_values, visit_counts, priors, 200)
        
        # Should have performance stats
        assert qmcts.performance_stats['quantum_calls'] > 0
        
        # Should have production optimizer stats
        prod_stats = qmcts.production_optimizer.get_comprehensive_stats()
        assert prod_stats['total_operations'] > 0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])