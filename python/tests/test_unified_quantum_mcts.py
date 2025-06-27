"""
Tests for Unified Quantum MCTS Implementation

This test suite validates the complete unified quantum MCTS system:
- Performance requirements (< 2x overhead)
- Mathematical correctness across all components
- Wave processing integration
- State management and causality preservation
- Adaptive quantum/classical switching
- Comprehensive monitoring and validation
"""

import pytest
import torch
import numpy as np
import time
import math
from typing import Dict, Any

from mcts.quantum.unified_quantum_mcts_optimized import (
    UnifiedQuantumMCTS, UnifiedQuantumConfig, QuantumStateManager, WaveQuantumProcessor,
    create_unified_quantum_mcts, create_performance_optimized_quantum_mcts
)
from mcts.quantum.quantum_features_v2 import MCTSPhase


class TestUnifiedQuantumMCTS:
    """Test suite for unified quantum MCTS"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def config(self, device):
        return UnifiedQuantumConfig(
            branching_factor=30,
            avg_game_length=100,
            target_overhead=1.8,
            enable_wave_processing=True,
            device=device.type,
            enable_mathematical_validation=True,
            enable_performance_monitoring=True
        )
    
    @pytest.fixture
    def quantum_mcts(self, config):
        return UnifiedQuantumMCTS(config)
    
    @pytest.fixture
    def test_data_single(self, device):
        """Single-path test data"""
        return {
            'q_values': torch.randn(10, device=device),
            'visit_counts': torch.randint(1, 50, (10,), device=device),
            'priors': torch.softmax(torch.randn(10, device=device), dim=0),
            'simulation_count': 1000
        }
    
    @pytest.fixture
    def test_data_wave(self, device):
        """Wave batch test data"""
        batch_size = 128
        num_actions = 30
        return {
            'q_values': torch.randn(batch_size, num_actions, device=device),
            'visit_counts': torch.randint(1, 100, (batch_size, num_actions), device=device),
            'priors': torch.softmax(torch.randn(batch_size, num_actions, device=device), dim=-1),
            'simulation_count': 5000
        }
    
    def test_unified_creation_and_initialization(self, device):
        """Test that unified quantum MCTS can be created and initialized properly"""
        # Test factory function
        quantum_mcts = create_unified_quantum_mcts(
            branching_factor=25,
            avg_game_length=80,
            device=device.type
        )
        
        assert quantum_mcts.config.branching_factor == 25
        assert quantum_mcts.config.avg_game_length == 80
        assert quantum_mcts.device == device
        
        # Test performance optimized version
        perf_mcts = create_performance_optimized_quantum_mcts(
            branching_factor=30,
            device=device.type
        )
        
        assert perf_mcts.config.target_overhead == 1.5
        assert perf_mcts.config.fast_mode == True
        assert perf_mcts.config.enable_ultra_optimization == True
    
    def test_single_path_processing(self, quantum_mcts, test_data_single):
        """Test single-path quantum processing"""
        enhanced_scores = quantum_mcts.apply_quantum_to_selection(
            test_data_single['q_values'],
            test_data_single['visit_counts'],
            test_data_single['priors'],
            simulation_count=test_data_single['simulation_count']
        )
        
        # Validate results
        assert enhanced_scores.shape == test_data_single['q_values'].shape
        assert torch.all(torch.isfinite(enhanced_scores))
        
        # Check that probabilities sum to 1
        prob_sum = torch.sum(torch.softmax(enhanced_scores, dim=0))
        assert abs(prob_sum.item() - 1.0) < 1e-6
        
        # Verify quantum enhancement occurred
        classical_diff = enhanced_scores - test_data_single['q_values']
        assert torch.any(torch.abs(classical_diff) > 1e-6)
    
    def test_wave_batch_processing(self, quantum_mcts, test_data_wave):
        """Test wave batch processing for large batches"""
        enhanced_scores = quantum_mcts.apply_quantum_to_selection(
            test_data_wave['q_values'],
            test_data_wave['visit_counts'],
            test_data_wave['priors'],
            simulation_count=test_data_wave['simulation_count']
        )
        
        # Validate batch results
        assert enhanced_scores.shape == test_data_wave['q_values'].shape
        assert torch.all(torch.isfinite(enhanced_scores))
        
        # Check that each row sums to approximately 1 after softmax
        probs = torch.softmax(enhanced_scores, dim=-1)
        prob_sums = torch.sum(probs, dim=-1)
        assert torch.all(torch.abs(prob_sums - 1.0) < 1e-6)
        
        # Verify wave processing was used
        wave_stats = quantum_mcts.get_performance_statistics()['wave_stats']
        assert wave_stats['waves_processed'] > 0
    
    def test_performance_overhead_requirement(self, device):
        """Test that unified quantum MCTS meets performance requirements"""
        # Create performance-optimized version
        quantum_mcts = create_performance_optimized_quantum_mcts(
            branching_factor=30, device=device.type
        )
        
        # Test data
        num_iterations = 300
        q_vals = torch.randn(50, device=device)
        visits = torch.randint(1, 100, (50,), device=device)
        priors = torch.softmax(torch.randn(50, device=device), dim=0)
        
        # Classical baseline timing
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
        
        # Quantum timing with warmup
        for _ in range(10):
            quantum_mcts.apply_quantum_to_selection(q_vals, visits, priors, simulation_count=1000)
        
        start_time = time.perf_counter()
        for i in range(num_iterations):
            quantum_scores = quantum_mcts.apply_quantum_to_selection(
                q_vals, visits, priors, simulation_count=1000 + i
            )
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        quantum_time = time.perf_counter() - start_time
        
        # Calculate overhead
        overhead = quantum_time / classical_time
        
        print(f"\nUnified Performance Test:")
        print(f"Classical time: {classical_time:.4f}s")
        print(f"Quantum time: {quantum_time:.4f}s")
        print(f"Overhead: {overhead:.2f}x")
        
        # Should meet performance target
        target = quantum_mcts.config.target_overhead
        assert overhead < target, f"Overhead {overhead:.2f}x exceeds target {target}x"
    
    def test_adaptive_quantum_classical_switching(self, quantum_mcts, test_data_single):
        """Test adaptive switching between quantum and classical regimes"""
        # Test quantum regime (low simulation count)
        quantum_mcts.update_simulation_count(100)  # Force quantum regime
        enhanced_scores_quantum = quantum_mcts.apply_quantum_to_selection(
            test_data_single['q_values'],
            test_data_single['visit_counts'],
            test_data_single['priors'],
            simulation_count=100
        )
        
        # Test classical regime (high simulation count)
        quantum_mcts.update_simulation_count(50000)  # Force classical regime
        enhanced_scores_classical = quantum_mcts.apply_quantum_to_selection(
            test_data_single['q_values'],
            test_data_single['visit_counts'],
            test_data_single['priors'],
            simulation_count=50000
        )
        
        # Quantum regime should have larger corrections than classical regime
        quantum_correction = enhanced_scores_quantum - test_data_single['q_values']
        classical_correction = enhanced_scores_classical - test_data_single['q_values']
        
        quantum_magnitude = torch.mean(torch.abs(quantum_correction))
        classical_magnitude = torch.mean(torch.abs(classical_correction))
        
        assert quantum_magnitude > classical_magnitude, \
            "Quantum regime should have larger corrections than classical regime"
    
    def test_state_management_and_causality(self, quantum_mcts, test_data_single):
        """Test state management and causality preservation"""
        # Apply quantum selection multiple times with incrementing simulation count
        previous_scores = None
        
        for sim_count in [1000, 1500, 2000, 2500, 3000]:
            enhanced_scores = quantum_mcts.apply_quantum_to_selection(
                test_data_single['q_values'],
                test_data_single['visit_counts'],
                test_data_single['priors'],
                simulation_count=sim_count
            )
            
            # Verify results are finite and stable
            assert torch.all(torch.isfinite(enhanced_scores))
            
            # Check that changes are reasonable between iterations
            if previous_scores is not None:
                max_change = torch.max(torch.abs(enhanced_scores - previous_scores))
                assert max_change < 1.0, "Excessive change between iterations"
            
            previous_scores = enhanced_scores.clone()
        
        # Verify state manager tracked the updates
        state_summary = quantum_mcts.state_manager.get_state_summary()
        assert state_summary['total_simulations'] == 3000
        assert state_summary['causality_preserved'] == True
    
    def test_fallback_mechanism(self, device):
        """Test fallback to classical MCTS when quantum fails"""
        # Create config that allows fallback
        config = UnifiedQuantumConfig(
            enable_quantum=True,
            fallback_to_classical=True,
            quantum_failure_tolerance=1,
            device=device.type
        )
        
        quantum_mcts = UnifiedQuantumMCTS(config)
        
        # Simulate quantum failures by forcing error conditions
        quantum_mcts.state_manager.quantum_failures = 2  # Exceed tolerance
        
        # Should still work via classical fallback
        q_values = torch.randn(10, device=device)
        visit_counts = torch.randint(1, 50, (10,), device=device)
        priors = torch.softmax(torch.randn(10, device=device), dim=0)
        
        result = quantum_mcts.apply_quantum_to_selection(
            q_values, visit_counts, priors, simulation_count=1000
        )
        
        assert torch.all(torch.isfinite(result))
        assert result.shape == q_values.shape
        
        # Verify fallback was used
        stats = quantum_mcts.get_performance_statistics()
        assert stats['performance_stats']['fallback_count'] > 0
    
    def test_mathematical_property_validation(self, quantum_mcts):
        """Test built-in mathematical property validation"""
        validation_results = quantum_mcts.validate_mathematical_properties()
        
        # All mathematical properties should be valid
        assert validation_results['path_integral_normalized'] == True, \
            "Path integral normalization failed"
        assert validation_results['causality_preserved'] == True, \
            "Causality preservation failed"  
        assert validation_results['quantum_classical_consistency'] == True, \
            "Quantum-classical consistency failed"
    
    def test_performance_monitoring_context(self, quantum_mcts, test_data_single):
        """Test performance monitoring context manager"""
        with quantum_mcts.performance_monitoring():
            for i in range(10):
                quantum_mcts.apply_quantum_to_selection(
                    test_data_single['q_values'],
                    test_data_single['visit_counts'],
                    test_data_single['priors'],
                    simulation_count=1000 + i
                )
        
        # Should have recorded performance statistics
        stats = quantum_mcts.get_performance_statistics()
        assert stats['total_operations'] >= 10
    
    def test_comprehensive_statistics(self, quantum_mcts, test_data_single, test_data_wave):
        """Test comprehensive performance and usage statistics"""
        # Perform both single and wave operations
        quantum_mcts.apply_quantum_to_selection(
            test_data_single['q_values'],
            test_data_single['visit_counts'],
            test_data_single['priors'],
            simulation_count=1000
        )
        
        quantum_mcts.apply_quantum_to_selection(
            test_data_wave['q_values'],
            test_data_wave['visit_counts'],
            test_data_wave['priors'],
            simulation_count=2000
        )
        
        # Get comprehensive statistics
        stats = quantum_mcts.get_performance_statistics()
        
        # Validate statistics structure
        required_keys = [
            'current_regime', 'total_simulations', 'performance_stats',
            'wave_stats', 'single_processor_stats', 'total_operations',
            'component_status'
        ]
        
        for key in required_keys:
            assert key in stats, f"Missing statistics key: {key}"
        
        # Validate component status
        component_status = stats['component_status']
        for component in ['state_manager', 'wave_processor', 'single_processor']:
            assert component_status[component] == 'active'
        
        # Validate operation counts
        assert stats['total_operations'] >= 2
        assert stats['wave_stats']['waves_processed'] >= 1
    
    def test_reset_functionality(self, quantum_mcts, test_data_single):
        """Test reset functionality"""
        # Perform some operations
        quantum_mcts.apply_quantum_to_selection(
            test_data_single['q_values'],
            test_data_single['visit_counts'],
            test_data_single['priors'],
            simulation_count=1000
        )
        
        # Verify state before reset
        stats_before = quantum_mcts.get_performance_statistics()
        assert stats_before['total_operations'] > 0
        
        # Reset
        quantum_mcts.reset()
        
        # Verify state after reset
        stats_after = quantum_mcts.get_performance_statistics()
        assert stats_after['total_operations'] == 0
        assert stats_after['total_simulations'] == 0


class TestQuantumStateManager:
    """Test quantum state manager independently"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture 
    def config(self, device):
        return UnifiedQuantumConfig(device=device.type)
    
    @pytest.fixture
    def state_manager(self, config):
        return QuantumStateManager(config)
    
    def test_state_updates_and_regime_detection(self, state_manager):
        """Test state updates and regime detection"""
        # Test quantum regime
        state_manager.update_state(500)
        assert state_manager.current_regime == MCTSPhase.QUANTUM
        assert state_manager.total_simulations == 500
        
        # Test critical regime  
        state_manager.update_state(5000)
        assert state_manager.current_regime == MCTSPhase.CRITICAL
        
        # Test classical regime
        state_manager.update_state(50000)
        assert state_manager.current_regime == MCTSPhase.CLASSICAL
    
    def test_causality_preservation(self, state_manager):
        """Test causality preservation tracking"""
        # Multiple updates should preserve causality
        for sim_count in [100, 200, 300, 400, 500]:
            state_manager.update_state(sim_count)
        
        # Should have stored pre-update states
        assert len(state_manager.pre_update_states) > 0
        assert state_manager.causality_preserved == True
        
        # Check state cleanup (should keep only last 100)
        for i in range(150):
            state_manager.update_state(1000 + i)
        
        assert len(state_manager.pre_update_states) <= 100
    
    def test_quantum_failure_tracking(self, state_manager):
        """Test quantum failure tracking and fallback logic"""
        # Record successful applications
        for _ in range(5):
            state_manager.record_quantum_application(True, 1.5)
        
        assert state_manager.quantum_failures == 0
        assert state_manager.should_use_quantum() == True
        
        # Record failures
        for _ in range(3):
            state_manager.record_quantum_application(False)
        
        assert state_manager.quantum_failures == 3
        assert state_manager.should_use_quantum() == False  # Should fallback
    
    def test_performance_tracking(self, state_manager):
        """Test performance statistics tracking"""
        # Record various applications with different overheads
        overheads = [1.2, 1.5, 1.8, 1.3, 1.6]
        for overhead in overheads:
            state_manager.record_quantum_application(True, overhead)
        
        stats = state_manager.get_state_summary()['performance_stats']
        
        # Should track applications and average overhead
        assert stats['quantum_applications'] == 5
        expected_avg = sum(overheads) / len(overheads)
        assert abs(stats['average_overhead'] - expected_avg) < 1e-6


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])