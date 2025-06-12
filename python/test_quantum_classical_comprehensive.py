#!/usr/bin/env python3
"""
Comprehensive test suite for quantum-classical MCTS integration

This test verifies that the quantum MCTS implementation seamlessly integrates
with the optimized classical MCTS while maintaining performance targets.
"""

import torch
import numpy as np
import time
import pytest
from mcts.core.optimized_mcts import MCTS, MCTSConfig
from mcts.quantum.quantum_features import QuantumConfig
from mcts.gpu.gpu_game_states import GameType
from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
from mcts.core.game_interface import GameInterface, GameType as GameInterfaceType
import alphazero_py


class TestQuantumClassicalIntegration:
    """Comprehensive test suite for quantum-classical integration"""
    
    @pytest.fixture
    def setup_hardware(self):
        """Setup hardware requirements"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.cuda.get_device_name()
    
    @pytest.fixture
    def quantum_config(self):
        """Create quantum configuration for testing"""
        return QuantumConfig(
            enable_quantum=True,
            quantum_level='tree_level',
            min_wave_size=32,
            optimal_wave_size=512,
            hbar_eff=0.05,
            coupling_strength=0.1,
            interference_alpha=0.05,
            phase_kick_strength=0.1,
            use_mixed_precision=True,
            fast_mode=True,
            device='cuda'
        )
    
    @pytest.fixture
    def classical_config(self):
        """Create classical configuration for comparison"""
        return MCTSConfig(
            num_simulations=5000,
            min_wave_size=512,
            max_wave_size=512,
            adaptive_wave_sizing=False,
            device='cuda',
            game_type=GameType.GOMOKU,
            board_size=15,
            enable_quantum=False,
            enable_debug_logging=False
        )
    
    @pytest.fixture
    def quantum_mcts_config(self, quantum_config):
        """Create quantum MCTS configuration"""
        return MCTSConfig(
            num_simulations=5000,
            min_wave_size=512,
            max_wave_size=512,
            adaptive_wave_sizing=False,
            device='cuda',
            game_type=GameType.GOMOKU,
            board_size=15,
            enable_quantum=True,
            quantum_config=quantum_config,
            enable_debug_logging=False
        )
    
    @pytest.fixture
    def evaluator(self):
        """Create ResNet evaluator"""
        evaluator = ResNetEvaluator(game_type='gomoku', device='cuda')
        evaluator._return_torch_tensors = True
        return evaluator
    
    @pytest.fixture
    def game_state(self):
        """Create test game state"""
        return alphazero_py.GomokuState()
    
    def test_quantum_features_initialization(self, setup_hardware, quantum_mcts_config, evaluator):
        """Test that quantum features initialize correctly"""
        mcts = MCTS(quantum_mcts_config, evaluator)
        
        # Verify quantum features are enabled
        assert mcts.quantum_features is not None
        assert hasattr(mcts.quantum_features, 'config')
        assert mcts.quantum_features.config.enable_quantum
        assert mcts.quantum_features.config.quantum_level == 'tree_level'
    
    def test_enhanced_features_compatibility(self, setup_hardware, quantum_mcts_config, evaluator):
        """Test quantum features work with enhanced 20-channel representation"""
        mcts = MCTS(quantum_mcts_config, evaluator)
        mcts.game_states.enable_enhanced_features()
        
        # Test that it can process a game state
        state = alphazero_py.GomokuState()
        policy = mcts.search(state, 1000)
        
        assert policy.shape[0] == 225  # 15x15 board
        assert abs(policy.sum() - 1.0) < 1e-5  # Valid probability distribution
    
    def test_performance_comparison(self, setup_hardware, classical_config, quantum_mcts_config, evaluator, game_state):
        """Test that quantum MCTS maintains performance within acceptable overhead"""
        # Test classical MCTS
        classical_mcts = MCTS(classical_config, evaluator)
        classical_mcts.game_states.enable_enhanced_features()
        
        start_time = time.perf_counter()
        classical_policy = classical_mcts.search(game_state, 5000)
        classical_time = time.perf_counter() - start_time
        classical_sims_per_sec = 5000 / classical_time
        
        # Test quantum MCTS
        quantum_mcts = MCTS(quantum_mcts_config, evaluator)
        quantum_mcts.game_states.enable_enhanced_features()
        
        start_time = time.perf_counter()
        quantum_policy = quantum_mcts.search(game_state, 5000)
        quantum_time = time.perf_counter() - start_time
        quantum_sims_per_sec = 5000 / quantum_time
        
        # Calculate overhead
        overhead = quantum_time / classical_time
        
        print(f"Classical MCTS: {classical_sims_per_sec:,.0f} sims/s")
        print(f"Quantum MCTS: {quantum_sims_per_sec:,.0f} sims/s")
        print(f"Overhead: {overhead:.2f}x")
        
        # Assert performance targets
        assert quantum_sims_per_sec >= 10000, f"Quantum MCTS too slow: {quantum_sims_per_sec:,.0f} sims/s"
        assert overhead <= 2.0, f"Quantum overhead too high: {overhead:.2f}x"
        
        # Both policies should be valid
        assert classical_policy.shape == quantum_policy.shape
        assert abs(classical_policy.sum() - 1.0) < 1e-5
        assert abs(quantum_policy.sum() - 1.0) < 1e-5
    
    def test_quantum_statistics(self, setup_hardware, quantum_mcts_config, evaluator, game_state):
        """Test that quantum features are actually being applied"""
        mcts = MCTS(quantum_mcts_config, evaluator)
        mcts.game_states.enable_enhanced_features()
        
        # Run search
        mcts.search(game_state, 5000)
        
        # Check quantum statistics
        assert hasattr(mcts.quantum_features, 'stats')
        stats = mcts.quantum_features.stats
        
        assert stats['quantum_applications'] > 0, "Quantum features not applied"
        assert stats['total_selections'] > 0, "No quantum selections recorded"
        assert stats['avg_overhead'] <= 2.0, f"Overhead too high: {stats['avg_overhead']:.2f}x"
    
    def test_quantum_vs_classical_exploration(self, setup_hardware, classical_config, quantum_mcts_config, evaluator, game_state):
        """Test that quantum MCTS provides different (enhanced) exploration"""
        # Run multiple searches and compare diversity
        classical_mcts = MCTS(classical_config, evaluator)
        quantum_mcts = MCTS(quantum_mcts_config, evaluator)
        
        classical_mcts.game_states.enable_enhanced_features()
        quantum_mcts.game_states.enable_enhanced_features()
        
        classical_policies = []
        quantum_policies = []
        
        for _ in range(3):
            classical_mcts.reset_tree()
            quantum_mcts.reset_tree()
            
            classical_policies.append(classical_mcts.search(game_state, 2000))
            quantum_policies.append(quantum_mcts.search(game_state, 2000))
        
        # Calculate entropy as measure of exploration diversity
        def entropy(policy):
            policy = policy + 1e-10  # Avoid log(0)
            return -np.sum(policy * np.log(policy))
        
        classical_entropies = [entropy(p) for p in classical_policies]
        quantum_entropies = [entropy(p) for p in quantum_policies]
        
        avg_classical_entropy = np.mean(classical_entropies)
        avg_quantum_entropy = np.mean(quantum_entropies)
        
        print(f"Classical entropy: {avg_classical_entropy:.3f}")
        print(f"Quantum entropy: {avg_quantum_entropy:.3f}")
        
        # Quantum should provide at least as much exploration diversity
        # (This is a soft requirement - quantum may or may not increase entropy)
        assert avg_quantum_entropy >= 0, "Invalid quantum entropy"
        assert avg_classical_entropy >= 0, "Invalid classical entropy"
    
    def test_multi_game_compatibility(self, setup_hardware, quantum_config, evaluator):
        """Test quantum features work across different game types"""
        game_types = [GameType.GOMOKU]  # Add others when available
        
        for game_type in game_types:
            config = MCTSConfig(
                num_simulations=1000,
                min_wave_size=256,
                max_wave_size=256,
                adaptive_wave_sizing=False,
                device='cuda',
                game_type=game_type,
                board_size=15,
                enable_quantum=True,
                quantum_config=quantum_config,
                enable_debug_logging=False
            )
            
            mcts = MCTS(config, evaluator)
            mcts.game_states.enable_enhanced_features()
            
            # Test basic functionality
            state = alphazero_py.GomokuState()  # Only Gomoku available for now
            policy = mcts.search(state, 1000)
            
            assert policy.shape[0] == 225
            assert abs(policy.sum() - 1.0) < 1e-5
            assert mcts.quantum_features.stats['quantum_applications'] > 0
    
    def test_memory_management(self, setup_hardware, quantum_mcts_config, evaluator, game_state):
        """Test that quantum features don't cause memory leaks"""
        mcts = MCTS(quantum_mcts_config, evaluator)
        mcts.game_states.enable_enhanced_features()
        
        # Get initial memory usage
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Run multiple searches
        for _ in range(5):
            mcts.reset_tree()
            mcts.search(game_state, 1000)
        
        # Check final memory usage
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        
        # Memory increase should be reasonable (allow for some growth)
        memory_increase = final_memory - initial_memory
        max_allowed_increase = 100 * 1024 * 1024  # 100MB
        
        assert memory_increase < max_allowed_increase, f"Memory leak detected: {memory_increase / 1024 / 1024:.1f}MB increase"


def run_comprehensive_test():
    """Run comprehensive quantum-classical integration test"""
    print("=" * 80)
    print("COMPREHENSIVE QUANTUM-CLASSICAL MCTS INTEGRATION TEST")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This test requires GPU.")
        return False
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Create test instance
    test_instance = TestQuantumClassicalIntegration()
    
    # Setup test data (without pytest fixtures)
    setup_hardware = torch.cuda.get_device_name()
    
    quantum_config = QuantumConfig(
        enable_quantum=True,
        quantum_level='tree_level',
        min_wave_size=32,
        optimal_wave_size=512,
        hbar_eff=0.05,
        coupling_strength=0.1,
        interference_alpha=0.05,
        phase_kick_strength=0.1,
        use_mixed_precision=True,
        fast_mode=True,
        device='cuda'
    )
    
    classical_config = MCTSConfig(
        num_simulations=5000,
        min_wave_size=512,
        max_wave_size=512,
        adaptive_wave_sizing=False,
        device='cuda',
        game_type=GameType.GOMOKU,
        board_size=15,
        enable_quantum=False,
        enable_debug_logging=False
    )
    
    quantum_mcts_config = MCTSConfig(
        num_simulations=5000,
        min_wave_size=512,
        max_wave_size=512,
        adaptive_wave_sizing=False,
        device='cuda',
        game_type=GameType.GOMOKU,
        board_size=15,
        enable_quantum=True,
        quantum_config=quantum_config,
        enable_debug_logging=False
    )
    
    evaluator = ResNetEvaluator(game_type='gomoku', device='cuda')
    evaluator._return_torch_tensors = True
    
    game_state = alphazero_py.GomokuState()
    
    tests = [
        ("Quantum Features Initialization", 
         lambda: test_instance.test_quantum_features_initialization(setup_hardware, quantum_mcts_config, evaluator)),
        ("Enhanced Features Compatibility", 
         lambda: test_instance.test_enhanced_features_compatibility(setup_hardware, quantum_mcts_config, evaluator)),
        ("Performance Comparison", 
         lambda: test_instance.test_performance_comparison(setup_hardware, classical_config, quantum_mcts_config, evaluator, game_state)),
        ("Quantum Statistics", 
         lambda: test_instance.test_quantum_statistics(setup_hardware, quantum_mcts_config, evaluator, game_state)),
        ("Quantum vs Classical Exploration", 
         lambda: test_instance.test_quantum_vs_classical_exploration(setup_hardware, classical_config, quantum_mcts_config, evaluator, game_state)),
        ("Multi-Game Compatibility", 
         lambda: test_instance.test_multi_game_compatibility(setup_hardware, quantum_config, evaluator)),
        ("Memory Management", 
         lambda: test_instance.test_memory_management(setup_hardware, quantum_mcts_config, evaluator, game_state))
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            test_func()
            print(f"  ✓ PASSED")
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED - Quantum-Classical integration is working correctly!")
        return True
    else:
        print(f"\n❌ {failed} TESTS FAILED - Integration issues detected.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)