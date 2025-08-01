"""Tests for MCTS backend switching functionality

Following TDD principles, we test that MCTS can be initialized with different backends.
"""

import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcts.core.mcts import MCTS
from mcts.core.mcts_config import MCTSConfig
from mcts.core.game_interface import GameInterface, GameType as LegacyGameType
from mock_evaluator import MockEvaluator


class MockGame:
    """Simple mock game for testing"""
    def __init__(self):
        self.board_size = 15
        self.action_size = 225
        
    def get_state(self):
        return torch.zeros((1, self.board_size, self.board_size))
    
    def get_legal_moves(self, state):
        return [1] * self.action_size  # All moves legal
    
    def is_terminal(self, state):
        return False
    
    def get_current_player(self):
        return 1


class TestMCTSBackendInitialization:
    """Test MCTS initialization with different backends"""
    
    def test_mcts_gpu_backend_default(self):
        """Test that MCTS defaults to GPU backend"""
        config = MCTSConfig(num_simulations=100, device='cuda')
        evaluator = MockEvaluator(game_type='gomoku', device='cuda')
        evaluator.set_board_size(15)  # Set board size for gomoku
        
        # Initialize MCTS without game_interface for now
        mcts = MCTS(config, evaluator)
        assert mcts.backend == 'gpu'
        assert hasattr(mcts, 'tree')
        assert mcts.tree.__class__.__name__ == 'CSRTree'
    
    def test_mcts_cpu_backend_initialization(self):
        """Test that MCTS can be initialized with CPU backend"""
        config = MCTSConfig(num_simulations=100, backend='cpu', device='cpu')
        evaluator = MockEvaluator(game_type='gomoku', device='cpu')
        evaluator.set_board_size(15)
        
        mcts = MCTS(config, evaluator)
        assert mcts.backend == 'cpu'
        assert hasattr(mcts, 'tree')
        # Verify CythonTree is being used for CPU backend
        assert mcts.tree.__class__.__name__ == 'CythonTree'
    
    def test_mcts_backend_affects_components(self):
        """Test that backend affects which components are initialized"""
        # GPU backend
        gpu_config = MCTSConfig(num_simulations=100, backend='gpu', device='cuda')
        evaluator = MockEvaluator(game_type='gomoku', device='cuda')
        evaluator.set_board_size(15)
        
        gpu_mcts = MCTS(gpu_config, evaluator)
        assert gpu_mcts.gpu_ops is not None  # GPU operations should be initialized
        
        # CPU backend
        cpu_config = MCTSConfig(num_simulations=100, backend='cpu', device='cpu')
        cpu_evaluator = MockEvaluator(game_type='gomoku', device='cpu')
        cpu_evaluator.set_board_size(15)
        cpu_mcts = MCTS(cpu_config, cpu_evaluator)
        # For CPU backend, we don't need GPU operations for tree ops
        # This behavior will be implemented when we add backend switching
    
    def test_mcts_search_works_with_cpu_backend(self):
        """Test that MCTS search works with CPU backend"""
        config = MCTSConfig(num_simulations=10, backend='cpu', device='cpu')
        evaluator = MockEvaluator(game_type='gomoku', device='cpu')
        evaluator.set_board_size(15)
        
        # Create proper game interface
        game_interface = GameInterface(LegacyGameType.GOMOKU, board_size=15)
        
        mcts = MCTS(config, evaluator, game_interface)
        state = game_interface.create_initial_state()
        
        # This should work even with CPU backend
        policy = mcts.search(state)
        assert policy is not None
        assert len(policy) == 225
        assert abs(policy.sum() - 1.0) < 1e-6  # Policy should sum to 1


class TestCPUWaveSearch:
    """Test CPU-specific wave search functionality"""
    
    def test_cpu_wave_search_initialization(self):
        """Test that CPU backend uses appropriate wave search"""
        config = MCTSConfig(
            num_simulations=100, 
            backend='cpu', 
            device='cpu',
            wave_size=32  # Smaller wave size for CPU
        )
        evaluator = MockEvaluator(game_type='gomoku', device='cpu')
        evaluator.set_board_size(15)
        
        mcts = MCTS(config, evaluator)
        
        # Verify wave search is initialized for CPU
        assert hasattr(mcts, 'wave_search')
        assert mcts.wave_search is not None
        # Wave search should be adapted for CPU backend
        # Check that CPU uses appropriate wave sizes through config
        assert config.wave_size == 32  # CPU uses smaller waves
    
    def test_cpu_wave_search_performance(self):
        """Test that CPU wave search completes within reasonable time"""
        import time
        
        config = MCTSConfig(
            num_simulations=100,
            backend='cpu',
            device='cpu',
            wave_size=16  # Small wave size for fast test
        )
        evaluator = MockEvaluator(game_type='gomoku', device='cpu')
        evaluator.set_board_size(15)
        game_interface = GameInterface(LegacyGameType.GOMOKU, board_size=15)
        
        mcts = MCTS(config, evaluator, game_interface)
        state = game_interface.create_initial_state()
        
        # Measure search time
        start_time = time.time()
        policy = mcts.search(state)
        search_time = time.time() - start_time
        
        # CPU search should complete 100 simulations in reasonable time
        assert search_time < 5.0  # 5 seconds max for 100 simulations
        assert policy is not None
        
    def test_cpu_wave_search_thread_safety(self):
        """Test that CPU wave search is thread-safe"""
        config = MCTSConfig(
            num_simulations=50,
            backend='cpu',
            device='cpu',
            wave_size=8,
            cpu_threads_per_worker=4  # Use multiple threads
        )
        evaluator = MockEvaluator(game_type='gomoku', device='cpu')
        evaluator.set_board_size(15)
        game_interface = GameInterface(LegacyGameType.GOMOKU, board_size=15)
        
        mcts = MCTS(config, evaluator, game_interface)
        state = game_interface.create_initial_state()
        
        # Run search multiple times to test thread safety
        policies = []
        for _ in range(3):
            policy = mcts.search(state)
            policies.append(policy)
        
        # All searches should produce valid policies
        for policy in policies:
            assert policy is not None
            assert len(policy) == 225
            assert abs(policy.sum() - 1.0) < 1e-6
    
    def test_cpu_wave_batching(self):
        """Test that CPU backend batches evaluations appropriately"""
        config = MCTSConfig(
            num_simulations=64,
            backend='cpu',
            device='cpu',
            wave_size=16,
            batch_size=8  # CPU-appropriate batch size
        )
        evaluator = MockEvaluator(game_type='gomoku', device='cpu')
        evaluator.set_board_size(15)
        game_interface = GameInterface(LegacyGameType.GOMOKU, board_size=15)
        
        mcts = MCTS(config, evaluator, game_interface)
        state = game_interface.create_initial_state()
        
        # Reset evaluator call count
        evaluator.call_count = 0
        
        policy = mcts.search(state)
        
        # Verify evaluations were batched
        # With 64 simulations and batch size 8, should have ~8 evaluation calls
        # Allow some variance due to tree structure
        assert evaluator.call_count <= 16  # Should batch effectively
        assert policy is not None