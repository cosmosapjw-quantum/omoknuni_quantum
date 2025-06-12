#!/usr/bin/env python3
"""Debug test for MCTS legal move handling"""

import torch
import numpy as np
import logging
from mcts.core.optimized_mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType
from mcts.quantum.quantum_features import QuantumConfig
import alphazero_py

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence other loggers
logging.getLogger('mcts').setLevel(logging.WARNING)


class DebugEvaluator:
    """Simple evaluator that returns uniform policy"""
    def __init__(self, board_size=15, device='cuda'):
        self.board_size = board_size
        self.device = torch.device(device)
        self.num_actions = board_size * board_size
    
    def evaluate_batch(self, features):
        batch_size = features.shape[0]
        # Return uniform policy
        policies = torch.ones((batch_size, self.num_actions), device=self.device) / self.num_actions
        values = torch.zeros((batch_size, 1), device=self.device)
        return policies, values


def test_legal_moves():
    """Test that MCTS only returns legal moves"""
    
    logger.info("Testing MCTS legal move handling...")
    
    # Create evaluator
    evaluator = DebugEvaluator()
    
    # Create MCTS config
    config = MCTSConfig(
        num_simulations=100,
        min_wave_size=256,
        max_wave_size=256,
        adaptive_wave_sizing=False,
        device='cuda',
        game_type=GameType.GOMOKU,
        board_size=15,
        enable_quantum=False,
        c_puct=1.4,
        temperature=1.0,
        enable_debug_logging=False
    )
    
    # Create MCTS
    mcts = MCTS(config, evaluator)
    mcts.optimize_for_hardware()
    
    # Test 1: Empty board
    logger.info("\nTest 1: Empty board")
    game = alphazero_py.GomokuState()
    
    # Get legal moves from game
    legal_moves = game.get_legal_moves()
    legal_indices = np.where(legal_moves)[0]
    logger.info(f"Number of legal moves: {len(legal_indices)}")
    logger.info(f"Total positions: {len(legal_moves)}")
    
    # Run MCTS
    policy = mcts.search(game, num_simulations=50)
    
    # Check policy
    logger.info(f"Policy shape: {policy.shape}")
    logger.info(f"Policy sum: {policy.sum():.6f}")
    
    # Check that all non-zero policy values are on legal moves
    non_zero_indices = np.where(policy > 1e-8)[0]
    logger.info(f"Non-zero policy positions: {len(non_zero_indices)}")
    
    illegal_moves = []
    for idx in non_zero_indices:
        if not legal_moves[idx]:
            illegal_moves.append(idx)
    
    if illegal_moves:
        logger.error(f"ERROR: Found {len(illegal_moves)} illegal moves with non-zero probability!")
        logger.error(f"Illegal move indices: {illegal_moves[:10]}")
    else:
        logger.info("✓ All non-zero policy values are on legal moves")
    
    # Test 2: After some moves
    logger.info("\nTest 2: After making some moves")
    
    # Make a few moves
    moves = [112, 113, 127, 128]  # Center area moves
    for move in moves:
        if legal_moves[move]:
            game.make_move(move)
            logger.info(f"Made move {move} (row={move//15}, col={move%15})")
        else:
            logger.warning(f"Skipping illegal move {move}")
    
    # Get new legal moves
    legal_moves = game.get_legal_moves()
    legal_indices = np.where(legal_moves)[0]
    logger.info(f"Number of legal moves after {len(moves)} moves: {len(legal_indices)}")
    
    # Reset MCTS tree for new position
    mcts.reset_tree()
    
    # Run MCTS again
    policy = mcts.search(game, num_simulations=50)
    
    # Check again
    non_zero_indices = np.where(policy > 1e-8)[0]
    logger.info(f"Non-zero policy positions: {len(non_zero_indices)}")
    
    illegal_moves = []
    for idx in non_zero_indices:
        if not legal_moves[idx]:
            illegal_moves.append(idx)
    
    if illegal_moves:
        logger.error(f"ERROR: Found {len(illegal_moves)} illegal moves with non-zero probability!")
        logger.error(f"Illegal move indices: {illegal_moves[:10]}")
        # Debug info
        for idx in illegal_moves[:3]:
            logger.error(f"  Move {idx} (row={idx//15}, col={idx%15}): policy={policy[idx]:.6f}")
    else:
        logger.info("✓ All non-zero policy values are on legal moves")
    
    # Test 3: Quantum version
    logger.info("\nTest 3: Testing quantum MCTS")
    
    quantum_config = MCTSConfig(
        num_simulations=100,
        min_wave_size=256,
        max_wave_size=256,
        adaptive_wave_sizing=False,
        device='cuda',
        game_type=GameType.GOMOKU,
        board_size=15,
        enable_quantum=True,
        quantum_config=QuantumConfig(
            quantum_level='tree_level',
            enable_quantum=True,
            min_wave_size=32,
            optimal_wave_size=256,
            hbar_eff=0.1,
            phase_kick_strength=0.1,
            interference_alpha=0.05,
            fast_mode=True,
            device='cuda'
        ),
        c_puct=1.4,
        temperature=1.0,
        enable_debug_logging=False
    )
    
    quantum_mcts = MCTS(quantum_config, evaluator)
    quantum_mcts.optimize_for_hardware()
    
    # Reset game to initial state
    game = alphazero_py.GomokuState()
    legal_moves = game.get_legal_moves()
    
    # Run quantum MCTS
    policy = quantum_mcts.search(game, num_simulations=50)
    
    # Check again
    non_zero_indices = np.where(policy > 1e-8)[0]
    logger.info(f"Quantum - Non-zero policy positions: {len(non_zero_indices)}")
    
    illegal_moves = []
    for idx in non_zero_indices:
        if not legal_moves[idx]:
            illegal_moves.append(idx)
    
    if illegal_moves:
        logger.error(f"ERROR: Quantum MCTS found {len(illegal_moves)} illegal moves!")
    else:
        logger.info("✓ Quantum MCTS: All non-zero policy values are on legal moves")
    
    logger.info("\nTest complete!")


if __name__ == "__main__":
    test_legal_moves()