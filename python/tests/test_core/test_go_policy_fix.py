"""Test Go game policy shape mismatch fixes

This module tests that Go game policy extraction produces consistent shapes
for the policy vector, following TDD principles.
"""

import pytest
import torch
import numpy as np
from mcts.core.mcts import MCTS
from mcts.core.mcts_config import MCTSConfig
from mcts.core.game_interface import GameInterface, GameType
from mcts.gpu.gpu_game_states import GameType as GPUGameType


class TestGoPolicyFix:
    """Test Go policy shape consistency fixes"""
    
    def test_go_policy_shape_consistency(self, go_game, mock_evaluator_factory, device):
        """Test that Go policy always has consistent shape (81 + 1 pass = 82)
        
        This test reproduces the issue where Go policies sometimes return 81
        elements and sometimes 82 elements.
        """
        # Create a mock evaluator specifically for 9x9 Go
        mock_evaluator = mock_evaluator_factory(game_type='go')
        # Override action space for 9x9 Go (81 positions + 1 pass)
        mock_evaluator.action_space = 82
        
        # Create MCTS config for 9x9 Go
        config = MCTSConfig()
        config.device = str(device)
        config.board_size = 9
        config.game_type = GPUGameType.GO
        config.max_children_per_node = 82
        config.num_simulations = 10
        config.max_tree_nodes = 1000
        config.c_puct = 1.4
        config.dirichlet_alpha = 0.3
        config.dirichlet_epsilon = 0.25
        config.temperature = 1.0
        config.enable_subtree_reuse = True
        config.enable_virtual_loss = True
        config.virtual_loss = 1.0
        config.max_wave_size = 8
        config.enable_fast_ucb = True
        config.classical_only_mode = True
        
        # Create game and initial state
        game = GameInterface(GameType.GO, board_size=9)
        state = game.create_initial_state()
        
        # Test 1: Normal case with visits
        mcts = MCTS(config, mock_evaluator)
        policy = mcts.search(state, num_simulations=10)
        
        print(f"DEBUG: Normal case policy shape: {policy.shape}")
        print(f"DEBUG: Expected shape: 82 (81 board + 1 pass)")
        
        # Should always be 82 for 9x9 Go (81 board positions + 1 pass)
        assert policy.shape[0] == 82, f"Expected policy shape 82, got {policy.shape[0]}"
        assert np.abs(policy.sum() - 1.0) < 0.001, f"Policy should sum to 1, got {policy.sum()}"
        assert np.all(policy >= 0), "All policy values should be non-negative"
        
        # Test 2: Edge case with zero visits (fallback path)
        # Force zero visits by creating MCTS that doesn't actually simulate
        mock_evaluator_zero = mock_evaluator_factory(game_type='go', deterministic=True, fixed_value=0.0)
        mock_evaluator_zero.action_space = 82
        
        # Create a new MCTS instance to ensure clean state
        mcts_zero = MCTS(config, mock_evaluator_zero)
        # This should trigger the fallback path
        policy_zero = mcts_zero.search(state, num_simulations=1)
        
        print(f"DEBUG: Fallback case policy shape: {policy_zero.shape}")
        
        # Should still be 82 even in fallback case
        assert policy_zero.shape[0] == 82, f"Fallback case should also have shape 82, got {policy_zero.shape[0]}"
        assert np.abs(policy_zero.sum() - 1.0) < 0.001, f"Fallback policy should sum to 1, got {policy_zero.sum()}"
        assert np.all(policy_zero >= 0), "All fallback policy values should be non-negative"
        
    def test_go_vs_gomoku_policy_shapes(self, mock_evaluator_factory, device):
        """Test that Go and Gomoku have different but consistent policy shapes"""
        # Go: 9x9 = 81 + 1 pass = 82
        go_evaluator = mock_evaluator_factory(game_type='go')
        go_evaluator.action_space = 82
        
        go_config = MCTSConfig()
        go_config.device = str(device)
        go_config.board_size = 9
        go_config.game_type = GPUGameType.GO
        go_config.max_children_per_node = 82
        go_config.num_simulations = 5
        go_config.max_tree_nodes = 1000
        go_config.c_puct = 1.4
        go_config.classical_only_mode = True
        
        # Gomoku: 15x15 = 225 (no pass move)
        gomoku_evaluator = mock_evaluator_factory(game_type='gomoku')
        gomoku_evaluator.action_space = 225
        
        gomoku_config = MCTSConfig()
        gomoku_config.device = str(device)
        gomoku_config.board_size = 15
        gomoku_config.game_type = GPUGameType.GOMOKU
        gomoku_config.max_children_per_node = 225
        gomoku_config.num_simulations = 5
        gomoku_config.max_tree_nodes = 1000
        gomoku_config.c_puct = 1.4
        gomoku_config.classical_only_mode = True
        
        # Test Go
        go_game = GameInterface(GameType.GO, board_size=9)
        go_state = go_game.create_initial_state()
        go_mcts = MCTS(go_config, go_evaluator)
        go_policy = go_mcts.search(go_state, num_simulations=5)
        
        # Test Gomoku  
        gomoku_game = GameInterface(GameType.GOMOKU, board_size=15)
        gomoku_state = gomoku_game.create_initial_state()
        gomoku_mcts = MCTS(gomoku_config, gomoku_evaluator)
        gomoku_policy = gomoku_mcts.search(gomoku_state, num_simulations=5)
        
        print(f"DEBUG: Go policy shape: {go_policy.shape}")
        print(f"DEBUG: Gomoku policy shape: {gomoku_policy.shape}")
        
        # Verify shapes
        assert go_policy.shape[0] == 82, f"Go should have 82 actions, got {go_policy.shape[0]}"
        assert gomoku_policy.shape[0] == 225, f"Gomoku should have 225 actions, got {gomoku_policy.shape[0]}"
        
        # Both should sum to 1
        assert np.abs(go_policy.sum() - 1.0) < 0.001
        assert np.abs(gomoku_policy.sum() - 1.0) < 0.001