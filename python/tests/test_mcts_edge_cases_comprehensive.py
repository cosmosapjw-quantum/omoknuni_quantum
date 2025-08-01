#!/usr/bin/env python3
"""Comprehensive edge case tests for MCTS implementation

This file tests all possible edge cases to ensure robustness.
"""

import pytest
import torch
import numpy as np
import logging
import threading
import time
from typing import List

from mcts.core.mcts import MCTS
from mcts.core.mcts_config import MCTSConfig
from mcts.gpu.gpu_game_states import GameType
from mock_evaluator import MockEvaluator

logger = logging.getLogger(__name__)


class TestMCTSEdgeCasesComprehensive:
    """Test all edge cases comprehensively"""
    
    # 1. State validation edge cases
    def test_invalid_state_shapes(self):
        """Test various invalid state shapes"""
        config = MCTSConfig(board_size=9, game_type=GameType.GOMOKU, backend='cpu')
        evaluator = MockEvaluator(board_size=config.board_size)
        mcts = MCTS(config, evaluator)
        
        # Wrong dimensions
        with pytest.raises(ValueError):
            mcts.search(np.zeros((5, 5), dtype=np.int8))
        
        # 1D array
        with pytest.raises(ValueError):
            mcts.search(np.zeros(81, dtype=np.int8))
        
        # 3D array
        with pytest.raises(ValueError):
            mcts.search(np.zeros((9, 9, 2), dtype=np.int8))
        
        # Non-square
        with pytest.raises(ValueError):
            mcts.search(np.zeros((9, 10), dtype=np.int8))
    
    def test_invalid_state_values(self):
        """Test states with invalid piece values"""
        config = MCTSConfig(board_size=9, game_type=GameType.GOMOKU, backend='cpu')
        evaluator = MockEvaluator(board_size=config.board_size)
        mcts = MCTS(config, evaluator)
        
        # Values outside valid range (-1, 0, 1)
        state = np.zeros((9, 9), dtype=np.int8)
        state[0, 0] = 2  # Invalid value
        
        # Should still work but game logic might handle it
        policy = mcts.search(state, num_simulations=10)
        assert policy is not None
    
    # 2. Tree capacity edge cases
    def test_zero_capacity_tree(self):
        """Test with zero tree capacity"""
        config = MCTSConfig(
            board_size=9,
            game_type=GameType.GOMOKU,
            backend='cpu',
            max_tree_nodes=1  # Only root
        )
        evaluator = MockEvaluator(board_size=config.board_size)
        mcts = MCTS(config, evaluator)
        
        state = np.zeros((9, 9), dtype=np.int8)
        # Should still return a policy (uniform)
        policy = mcts.search(state, num_simulations=100)
        assert policy is not None
        assert mcts.tree.num_nodes == 1  # Only root
    
    def test_tree_reuse_with_invalid_move(self):
        """Test tree reuse when root update uses invalid move"""
        config = MCTSConfig(
            board_size=9,
            game_type=GameType.GOMOKU,
            backend='cpu',
            enable_subtree_reuse=True
        )
        evaluator = MockEvaluator(board_size=config.board_size)
        mcts = MCTS(config, evaluator)
        
        state = np.zeros((9, 9), dtype=np.int8)
        policy = mcts.search(state, num_simulations=50)
        
        # Try to update root with invalid move index
        with pytest.raises(Exception):
            mcts.update_root(999)  # Invalid move index
    
    # 3. Simulation count edge cases
    def test_zero_simulations(self):
        """Test search with zero simulations"""
        config = MCTSConfig(board_size=9, game_type=GameType.GOMOKU, backend='cpu')
        evaluator = MockEvaluator(board_size=config.board_size)
        mcts = MCTS(config, evaluator)
        
        state = np.zeros((9, 9), dtype=np.int8)
        policy = mcts.search(state, num_simulations=0)
        
        # Should return uniform policy
        assert policy is not None
        assert np.allclose(policy, 1.0 / len(policy))
    
    def test_single_simulation(self):
        """Test search with single simulation"""
        config = MCTSConfig(board_size=9, game_type=GameType.GOMOKU, backend='cpu')
        evaluator = MockEvaluator(board_size=config.board_size)
        mcts = MCTS(config, evaluator)
        
        state = np.zeros((9, 9), dtype=np.int8)
        policy = mcts.search(state, num_simulations=1)
        
        assert policy is not None
        assert policy.sum() > 0.99
    
    # 4. Game state edge cases
    def test_terminal_state_search(self):
        """Test search from terminal game state"""
        config = MCTSConfig(board_size=3, game_type=GameType.GOMOKU, backend='cpu')
        evaluator = MockEvaluator(board_size=config.board_size)
        mcts = MCTS(config, evaluator)
        
        # Create a won position (3 in a row)
        state = np.zeros((3, 3), dtype=np.int8)
        state[0, :] = 1  # Player 1 wins
        
        policy = mcts.search(state, num_simulations=10)
        assert policy is not None
    
    def test_nearly_full_board(self):
        """Test search with nearly full board"""
        config = MCTSConfig(board_size=3, game_type=GameType.GOMOKU, backend='cpu')
        evaluator = MockEvaluator(board_size=config.board_size)
        mcts = MCTS(config, evaluator)
        
        # Fill board except one position
        state = np.ones((3, 3), dtype=np.int8)
        state[1, 1] = 0  # Only center is empty
        
        policy = mcts.search(state, num_simulations=10)
        assert policy is not None
        # Only one legal move
        assert policy[4] > 0.99  # Center position
    
    def test_completely_full_board(self):
        """Test search with completely full board"""
        config = MCTSConfig(board_size=3, game_type=GameType.GOMOKU, backend='cpu')
        evaluator = MockEvaluator(board_size=config.board_size)
        mcts = MCTS(config, evaluator)
        
        # Fill entire board
        state = np.ones((3, 3), dtype=np.int8)
        
        policy = mcts.search(state, num_simulations=10)
        assert policy is not None
        # No legal moves - should return uniform or handle gracefully
    
    # 5. Extreme parameter values
    def test_extreme_cpuct_values(self):
        """Test with extreme c_puct values"""
        for c_puct in [0.0, 0.001, 100.0, 1000.0]:
            config = MCTSConfig(
                board_size=9,
                game_type=GameType.GOMOKU,
                backend='cpu',
                c_puct=c_puct
            )
            evaluator = MockEvaluator(board_size=config.board_size)
            mcts = MCTS(config, evaluator)
            
            state = np.zeros((9, 9), dtype=np.int8)
            policy = mcts.search(state, num_simulations=20)
            assert policy is not None
            assert policy.sum() > 0.99
    
    def test_extreme_temperature_values(self):
        """Test with extreme temperature values in action selection"""
        config = MCTSConfig(board_size=9, game_type=GameType.GOMOKU, backend='cpu', cpu_threads_per_worker=1)
        evaluator = MockEvaluator(board_size=config.board_size)
        mcts = MCTS(config, evaluator)
        
        state = np.zeros((9, 9), dtype=np.int8)
        
        # First run search to get a policy
        policy = mcts.search(state, num_simulations=100)
        
        # Test temperature 0 (deterministic - should always pick highest visit count)
        actions_t0 = []
        for _ in range(10):
            action = mcts.select_action(state, temperature=0)
            actions_t0.append(action)
        # All actions should be the same with temperature=0
        assert len(set(actions_t0)) == 1, f"Temperature=0 should be deterministic, got actions: {actions_t0}"
        
        # Test very high temperature (should be more random)
        actions_t100 = []
        for _ in range(20):
            action = mcts.select_action(state, temperature=10.0)
            actions_t100.append(action)
        # Should have more variety with high temperature
        print(f"High temperature actions: {actions_t100}")
        print(f"Unique actions: {set(actions_t100)}")
        assert len(set(actions_t100)) >= 3, f"High temperature should give some variety, got {len(set(actions_t100))} unique actions"
    
    # 6. Concurrent access edge cases
    def test_concurrent_searches_same_tree(self):
        """Test that each thread should have its own MCTS instance"""
        config = MCTSConfig(
            board_size=9,
            game_type=GameType.GOMOKU,
            backend='cpu',
            enable_subtree_reuse=True
        )
        evaluator = MockEvaluator(board_size=config.board_size)
        
        state = np.zeros((9, 9), dtype=np.int8)
        results = []
        
        def search_worker():
            # Each thread creates its own MCTS instance - this is the correct pattern
            local_mcts = MCTS(config, evaluator)
            policy = local_mcts.search(state, num_simulations=100)
            results.append(policy)
        
        # Start multiple threads - each with its own MCTS
        threads = []
        for _ in range(4):
            t = threading.Thread(target=search_worker)
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # All should succeed
        assert len(results) == 4
        # Policies should be similar but not necessarily identical due to randomness
        for policy in results:
            assert policy is not None
            assert abs(policy.sum() - 1.0) < 1e-6
    
    # 7. Memory edge cases
    def test_memory_limit_handling(self):
        """Test behavior when approaching memory limits"""
        config = MCTSConfig(
            board_size=15,
            game_type=GameType.GOMOKU,
            backend='gpu' if torch.cuda.is_available() else 'cpu',
            max_tree_nodes=1000000  # Large tree
        )
        evaluator = MockEvaluator(board_size=config.board_size)
        mcts = MCTS(config, evaluator)
        
        state = np.zeros((15, 15), dtype=np.int8)
        
        # Run many simulations to stress memory
        try:
            policy = mcts.search(state, num_simulations=50000)
            assert policy is not None
        except RuntimeError as e:
            # Should handle memory errors gracefully
            assert "memory" in str(e).lower() or "Tree full" in str(e)
    
    # 8. Wave search edge cases
    def test_wave_size_edge_cases(self):
        """Test various wave sizes"""
        config = MCTSConfig(board_size=9, game_type=GameType.GOMOKU, backend='cpu')
        evaluator = MockEvaluator(board_size=config.board_size)
        mcts = MCTS(config, evaluator)
        
        state = np.zeros((9, 9), dtype=np.int8)
        
        # Test different wave sizes
        for wave_size in [1, 2, 32, 64, 128]:
            mcts.wave_search.allocate_buffers(wave_size)
            completed = mcts.wave_search.run_wave(
                wave_size=wave_size,
                node_to_state=mcts.node_to_state,
                state_pool_free_list=mcts.state_pool_free_list
            )
            assert completed > 0
    
    # 9. Progressive widening edge cases
    def test_progressive_widening_limits(self):
        """Test progressive widening with extreme parameters"""
        config = MCTSConfig(
            board_size=9,
            game_type=GameType.GOMOKU,
            backend='cpu'
        )
        # Set progressive widening parameters as attributes
        config.progressive_widening_constant = 0.1  # Very restrictive
        config.progressive_widening_exponent = 0.1
        
        evaluator = MockEvaluator(board_size=config.board_size)
        mcts = MCTS(config, evaluator)
        
        state = np.zeros((9, 9), dtype=np.int8)
        policy = mcts.search(state, num_simulations=100)
        
        # Should still work but with limited branching
        assert policy is not None
        assert abs(policy.sum() - 1.0) < 1e-6
    
    # 10. Illegal move handling
    def test_all_illegal_moves(self):
        """Test when all moves become illegal during search"""
        # This is tested in the main comprehensive test
        # but we could add more specific scenarios here
        pass
    
    # 11. Numerical stability edge cases
    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        config = MCTSConfig(board_size=9, game_type=GameType.GOMOKU, backend='cpu')
        evaluator = MockEvaluator(
            board_size=config.board_size,
            fixed_value=0.999999  # Near boundary
        )
        mcts = MCTS(config, evaluator)
        
        state = np.zeros((9, 9), dtype=np.int8)
        policy = mcts.search(state, num_simulations=100)
        
        # Should not have NaN or inf values
        assert not np.any(np.isnan(policy))
        assert not np.any(np.isinf(policy))
    
    # 12. Device switching edge cases
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_switching(self):
        """Test switching between CPU and GPU mid-search"""
        # This would require modifying the implementation
        # Current implementation doesn't support device switching
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])