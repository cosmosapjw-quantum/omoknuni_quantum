"""Debug terminal position detection flow

This module analyzes the step-by-step flow to understand why terminal positions
still expand to many nodes instead of stopping early.
"""

import pytest
import torch
import numpy as np
from mcts.core.mcts import MCTS
from mcts.core.game_interface import GameInterface, GameType


class TestTerminalDetectionDebug:
    """Debug terminal position detection flow"""
    
    def test_terminal_position_step_by_step(self, base_mcts_config, mock_evaluator):
        """Step through terminal position handling to understand the issue"""
        # Create terminal state
        game = GameInterface(GameType.GOMOKU, board_size=15)
        state = game.create_initial_state()
        
        # Create winning sequence
        moves = [112, 127, 113, 128, 114, 129, 115, 130, 116]  # Horizontal line
        for move in moves:
            state = game.apply_move(state, move)
            
        print(f"DEBUG: State is terminal: {state.is_terminal()}")
        print(f"DEBUG: Game result: {state.get_game_result()}")
        
        # Initialize MCTS
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Check initial state
        print(f"DEBUG: Initial tree nodes: {mcts.tree.num_nodes}")
        
        # Initialize root with terminal state
        mcts._ensure_root_initialized(state)
        root_state_idx = mcts.node_to_state[0].item()
        
        print(f"DEBUG: Root state index: {root_state_idx}")
        print(f"DEBUG: Tree nodes after root init: {mcts.tree.num_nodes}")
        
        # Check if GPU game state reflects terminal status
        if hasattr(mcts.game_states, 'is_terminal'):
            gpu_terminal = mcts.game_states.is_terminal[root_state_idx].item()
            print(f"DEBUG: GPU game state terminal: {gpu_terminal}")
        
        # Check what happens during one simulation
        print(f"DEBUG: Running one simulation...")
        completed = mcts.wave_search.run_wave(
            wave_size=1,
            node_to_state=mcts.node_to_state,
            state_pool_free_list=mcts.state_pool_free_list
        )
        
        print(f"DEBUG: Completed simulations: {completed}")
        print(f"DEBUG: Tree nodes after one simulation: {mcts.tree.num_nodes}")
        
        # Check root children
        children, actions, priors = mcts.tree.get_children(0)
        print(f"DEBUG: Root has {len(children)} children")
        print(f"DEBUG: Actions: {actions[:10] if len(actions) > 10 else actions}")
        
        # Check if root was expanded
        if len(children) > 0:
            print(f"DEBUG: Root was expanded even though state is terminal!")
            
            # Check legal moves on terminal state
            legal_mask = mcts.game_states.get_legal_moves_mask(
                torch.tensor([root_state_idx], device=mcts.device)
            )[0]
            legal_moves = torch.nonzero(legal_mask).squeeze(-1).cpu().numpy()
            print(f"DEBUG: Legal moves on terminal state: {len(legal_moves)}")
            print(f"DEBUG: First 10 legal moves: {legal_moves[:10]}")
        
        # This reveals why terminal detection is failing
        assert False, "Debug test - examine output above"