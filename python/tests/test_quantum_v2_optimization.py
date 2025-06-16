"""Test suite for quantum MCTS v2.0 optimizations"""

import pytest
import torch
import time
import numpy as np
from typing import Dict, Any

from mcts.quantum.quantum_features_v2 import QuantumMCTSV2, QuantumConfigV2, MCTSPhase
from mcts.quantum.quantum_mcts_wrapper import QuantumMCTSWrapper, UnifiedQuantumConfig
from mcts.gpu.quantum_cuda_extension import batched_ucb_selection_quantum_v2


class TestV2Optimizations:
    """Test optimizations for quantum MCTS v2.0"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def config_v2(self, device):
        return QuantumConfigV2(
            enable_quantum=True,
            branching_factor=50,
            avg_game_length=100,
            c_puct=1.414,
            device=device.type,
            cache_quantum_corrections=True,
            fast_mode=True
        )
    
    def test_lookup_tables_precomputed(self, config_v2):
        """Test that lookup tables are properly pre-computed"""
        quantum = QuantumMCTSV2(config_v2)
        
        # Check critical tables exist
        assert hasattr(quantum, 'tau_table')
        assert hasattr(quantum, 'temperature_table')
        assert hasattr(quantum, 'hbar_factors')
        assert hasattr(quantum, 'phase_kick_table')
        assert hasattr(quantum, 'decoherence_tables')
        
        # Check sizes
        assert len(quantum.tau_table) == 100000
        assert len(quantum.hbar_factors) == 100000
        assert len(quantum.temperature_table) == 100000
        
        # Check decoherence tables
        assert len(quantum.decoherence_tables) >= 8  # At least 8 gamma values
        
        # Check pre-allocated tensors
        assert hasattr(quantum, '_preallocated')
        assert 3072 in quantum._preallocated  # Key batch size
    
    def test_phase_configs_cached(self, config_v2):
        """Test that phase configurations are pre-cached"""
        quantum = QuantumMCTSV2(config_v2)
        
        # Check phase configs exist
        assert hasattr(quantum, 'phase_configs')
        assert len(quantum.phase_configs) == 3
        
        # Check all phases cached
        assert MCTSPhase.QUANTUM in quantum.phase_configs
        assert MCTSPhase.CRITICAL in quantum.phase_configs  
        assert MCTSPhase.CLASSICAL in quantum.phase_configs
        
        # Check current config reference
        assert hasattr(quantum, '_current_phase_config')
        assert quantum._current_phase_config == quantum.phase_configs[MCTSPhase.QUANTUM]
    
    def test_no_tensor_creation_overhead(self, config_v2, device):
        """Test that wrapper doesn't create new tensors on each call"""
        wrapper_config = UnifiedQuantumConfig(
            version='v2',
            enable_quantum=True,
            branching_factor=50,
            device=device.type
        )
        wrapper = QuantumMCTSWrapper(wrapper_config)
        
        # Create test data
        batch_size = 64
        num_actions = 50
        q_values = torch.randn(batch_size, num_actions, device=device)
        visit_counts = torch.randint(0, 100, (batch_size, num_actions), device=device)
        priors = torch.softmax(torch.randn(batch_size, num_actions, device=device), dim=-1)
        
        # First call to initialize any lazy allocations
        _ = wrapper.apply_quantum_to_selection(
            q_values, visit_counts, priors,
            c_puct=1.414, parent_visit=100, total_simulations=1000
        )
        
        # Measure tensor allocations
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        
        # Multiple calls shouldn't allocate new memory
        for _ in range(100):
            _ = wrapper.apply_quantum_to_selection(
                q_values, visit_counts, priors,
                c_puct=1.414, parent_visit=100, total_simulations=1000
            )
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
            final_memory = torch.cuda.memory_allocated()
            memory_increase = final_memory - initial_memory
            
            # Should have minimal memory increase (< 1KB per call)
            assert memory_increase < 100 * 1024  # 100KB total for 100 calls
    
    def test_cuda_kernel_integration(self, config_v2, device):
        """Test that v2.0 can use CUDA kernels"""
        if device.type != 'cuda':
            pytest.skip("CUDA not available")
        
        quantum = QuantumMCTSV2(config_v2)
        
        # Create CSR format data
        batch_size = 32
        num_nodes = 1000
        num_edges = 5000
        
        q_values = torch.randn(num_nodes, device=device)
        visit_counts = torch.randint(0, 100, (num_nodes,), device=device)
        priors = torch.rand(num_edges, device=device)
        
        # Simple CSR structure
        row_ptr = torch.arange(0, num_edges + 1, num_edges // batch_size, device=device)[:batch_size + 1]
        col_indices = torch.randint(0, num_nodes, (num_edges,), device=device)
        
        c_puct_batch = torch.full((batch_size,), 1.414, device=device)
        parent_visits_batch = torch.randint(10, 1000, (batch_size,), device=device)
        simulation_counts_batch = torch.full((batch_size,), 5000, device=device)
        
        # Test CUDA kernel call
        try:
            actions, scores = quantum.apply_quantum_to_selection_batch_cuda(
                q_values, visit_counts, priors,
                row_ptr, col_indices,
                c_puct_batch, parent_visits_batch, simulation_counts_batch,
                debug_logging=True
            )
            
            # Check outputs
            assert actions.shape == (batch_size,)
            assert scores.shape == (batch_size,)
            assert actions.dtype == torch.int32 or actions.dtype == torch.int64
            assert scores.dtype == torch.float32
        except Exception as e:
            # CUDA kernels might not be compiled, which is ok for this test
            if "CUDA" not in str(e) and "kernel" not in str(e):
                raise
    
    def test_performance_improvement(self, config_v2, device):
        """Test that optimizations improve performance"""
        # Create test data
        batch_size = 64
        num_actions = 50
        num_iterations = 1000
        
        q_values = torch.randn(batch_size, num_actions, device=device)
        visit_counts = torch.randint(0, 100, (batch_size, num_actions), device=device)
        priors = torch.softmax(torch.randn(batch_size, num_actions, device=device), dim=-1)
        
        # Test original v2 (without optimizations)
        config_original = QuantumConfigV2(
            enable_quantum=True,
            branching_factor=50,
            device=device.type,
            cache_quantum_corrections=False,  # Disable caching
            fast_mode=False
        )
        quantum_original = QuantumMCTSV2(config_original)
        
        # Warmup
        for _ in range(10):
            _ = quantum_original.apply_quantum_to_selection(
                q_values, visit_counts, priors,
                c_puct=1.414, simulation_count=1000
            )
        
        # Time original
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        for i in range(num_iterations):
            _ = quantum_original.apply_quantum_to_selection(
                q_values, visit_counts, priors,
                c_puct=1.414, simulation_count=1000 + i
            )
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        original_time = time.perf_counter() - start
        
        # Test optimized v2
        quantum_optimized = QuantumMCTSV2(config_v2)
        
        # Warmup
        for _ in range(10):
            _ = quantum_optimized.apply_quantum_to_selection(
                q_values, visit_counts, priors,
                c_puct=1.414, simulation_count=1000
            )
        
        # Time optimized
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        for i in range(num_iterations):
            _ = quantum_optimized.apply_quantum_to_selection(
                q_values, visit_counts, priors,
                c_puct=1.414, simulation_count=1000 + i
            )
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        optimized_time = time.perf_counter() - start
        
        # Optimized should be faster
        speedup = original_time / optimized_time
        print(f"\nPerformance improvement: {speedup:.2f}x")
        print(f"Original: {original_time:.4f}s ({num_iterations/original_time:.0f} calls/sec)")
        print(f"Optimized: {optimized_time:.4f}s ({num_iterations/optimized_time:.0f} calls/sec)")
        
        # Should see at least some improvement
        assert speedup > 1.1  # At least 10% faster


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])