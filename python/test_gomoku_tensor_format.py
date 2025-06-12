#!/usr/bin/env python3
"""Test the exact format of Gomoku tensor representation"""

import alphazero_py
import numpy as np

# Create game and make moves
game = alphazero_py.GomokuState()

print("=== Testing Gomoku Tensor Format ===")

# Initial state
print("\n1. Initial empty board:")
tensor = game.get_tensor_representation()
print(f"Shape: {tensor.shape}")
print(f"Channel 0 sum: {np.sum(tensor[0])}")
print(f"Channel 1 sum: {np.sum(tensor[1])}")
print(f"Channel 2 sum: {np.sum(tensor[2])}")

# Make first move (player 1)
print(f"\n2. After player 1 moves at position 112 (center):")
print(f"Current player before move: {game.get_current_player()}")
game.make_move(112)
print(f"Current player after move: {game.get_current_player()}")

tensor = game.get_tensor_representation()
print(f"Channel 0 sum: {np.sum(tensor[0])}")
print(f"Channel 1 sum: {np.sum(tensor[1])}")
print(f"Channel 0 at center: {tensor[0, 7, 7]}")
print(f"Channel 1 at center: {tensor[1, 7, 7]}")

# Make second move (player 2)
print(f"\n3. After player 2 moves at position 113:")
game.make_move(113)
print(f"Current player: {game.get_current_player()}")

tensor = game.get_tensor_representation()
print(f"Channel 0 sum: {np.sum(tensor[0])}")
print(f"Channel 1 sum: {np.sum(tensor[1])}")
print(f"Channel 0 at 112: {tensor[0, 7, 7]}")
print(f"Channel 1 at 112: {tensor[1, 7, 7]}")
print(f"Channel 0 at 113: {tensor[0, 7, 8]}")
print(f"Channel 1 at 113: {tensor[1, 7, 8]}")

# Reconstruct board
print("\n4. Reconstructed board (1=player1, -1=player2, 0=empty):")
board = tensor[0].astype(int) - tensor[1].astype(int)
print(f"Board shape: {board.shape}")
print(f"Board center 5x5:")
for i in range(5, 10):
    row = []
    for j in range(5, 10):
        val = board[i, j]
        if val == 1:
            row.append(" X")
        elif val == -1:
            row.append(" O")
        else:
            row.append(" .")
    print("".join(row))

print("\n5. Understanding the representation:")
print("- Channel 0: Current player's stones (from their perspective)")
print("- Channel 1: Opponent's stones (from current player's perspective)")
print("- Channel 2: Additional info (maybe legal moves or history)")
print("- Board reconstruction: channel0 - channel1 gives absolute board state")