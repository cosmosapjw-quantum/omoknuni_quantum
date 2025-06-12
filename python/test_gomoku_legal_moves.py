#!/usr/bin/env python3
"""Test Gomoku legal moves to understand the game rules"""

import alphazero_py
import numpy as np

# Create a new game
game = alphazero_py.GomokuState()

# Get legal moves
legal_moves_list = game.get_legal_moves()
legal_moves = np.array(legal_moves_list)
print(f"Initial legal moves array length: {len(legal_moves)}")
print(f"Number of legal moves: {np.sum(legal_moves)}")
print(f"Board size should be: {int(np.sqrt(len(legal_moves)))}")

# Check if all positions are legal initially
all_legal = np.all(legal_moves)
print(f"Are all positions legal initially? {all_legal}")

if not all_legal:
    illegal_positions = np.where(~legal_moves)[0]
    print(f"Illegal positions: {illegal_positions}")
    for pos in illegal_positions:
        row, col = pos // 15, pos % 15
        print(f"  Position {pos}: row={row}, col={col}")

# Test making a move
print("\nTesting move mechanics:")
move = 112  # Center position
print(f"Attempting move at position {move} (row={move//15}, col={move%15})")
print(f"Is this move legal? {legal_moves[move]}")

if legal_moves[move]:
    game.make_move(move)
    print("Move successful!")
    
    # Check new legal moves
    new_legal_moves = game.get_legal_moves()
    print(f"Legal moves array length after move: {len(new_legal_moves)}")
    print(f"Number of legal moves after move: {np.sum(new_legal_moves)}")
    
    # The position we just played should now be illegal
    print(f"Is position {move} still legal? {new_legal_moves[move]}")
else:
    print("Move was not legal!")