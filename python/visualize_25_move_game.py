#!/usr/bin/env python3
"""
Visualize the 25-move game to understand why Black won
"""

import numpy as np

# Move sequence from the debug output
moves = [
    (7, 7),   # Move 1: Black
    (7, 6),   # Move 2: White
    (8, 7),   # Move 3: Black
    (7, 8),   # Move 4: White
    (6, 7),   # Move 5: Black
    (9, 7),   # Move 6: White
    (6, 6),   # Move 7: Black
    (6, 8),   # Move 8: White
    (7, 5),   # Move 9: Black
    (7, 9),   # Move 10: White
    (8, 8),   # Move 11: Black
    (8, 6),   # Move 12: White
    (5, 7),   # Move 13: Black
    (8, 5),   # Move 14: White
    (8, 9),   # Move 15: Black
    (9, 6),   # Move 16: White
    (9, 8),   # Move 17: Black
    (10, 7),  # Move 18: White
    (5, 6),   # Move 19: Black
    (5, 8),   # Move 20: White
    (6, 5),   # Move 21: Black
    (6, 9),   # Move 22: White
    (7, 10),  # Move 23: Black
    (7, 4),   # Move 24: White
    (4, 7),   # Move 25: Black
]

# Create board
board = np.zeros((15, 15), dtype=int)

# Place stones
for i, (row, col) in enumerate(moves):
    if i % 2 == 0:  # Black moves (odd move numbers)
        board[row, col] = 1
    else:  # White moves (even move numbers)
        board[row, col] = -1

# Print board with focus on the center area
print("\nFinal board position after 25 moves:")
print("(B = Black, W = White, . = Empty)")
print("\n   ", end="")
for col in range(2, 13):
    print(f"{col:2}", end=" ")
print()

for row in range(2, 13):
    print(f"{row:2} ", end="")
    for col in range(2, 13):
        if board[row, col] == 1:
            print(" B", end=" ")
        elif board[row, col] == -1:
            print(" W", end=" ")
        else:
            print(" .", end=" ")
    print()

# Check for 5 in a row
def check_win(board, player):
    """Check if player has 5 in a row"""
    # Check horizontal
    for row in range(15):
        for col in range(11):
            if all(board[row, col+i] == player for i in range(5)):
                return True, "horizontal", row, col
    
    # Check vertical
    for row in range(11):
        for col in range(15):
            if all(board[row+i, col] == player for i in range(5)):
                return True, "vertical", row, col
    
    # Check diagonal (top-left to bottom-right)
    for row in range(11):
        for col in range(11):
            if all(board[row+i, col+i] == player for i in range(5)):
                return True, "diagonal \\", row, col
    
    # Check diagonal (top-right to bottom-left)
    for row in range(11):
        for col in range(4, 15):
            if all(board[row+i, col-i] == player for i in range(5)):
                return True, "diagonal /", row, col
    
    return False, None, None, None

# Check Black's win
black_win, direction, start_row, start_col = check_win(board, 1)
print(f"\nBlack wins: {black_win}")
if black_win:
    print(f"Winning pattern: {direction} starting at ({start_row}, {start_col})")
    
    # Show the winning stones
    print("\nWinning stones:")
    if direction == "horizontal":
        for i in range(5):
            print(f"  ({start_row}, {start_col + i})")
    elif direction == "vertical":
        for i in range(5):
            print(f"  ({start_row + i}, {start_col})")
    elif direction == "diagonal \\":
        for i in range(5):
            print(f"  ({start_row + i}, {start_col + i})")
    elif direction == "diagonal /":
        for i in range(5):
            print(f"  ({start_row + i}, {start_col - i})")

# Analyze Black's stones to find the winning line
print("\nAll Black stones (in order):")
black_stones = [(row, col) for i, (row, col) in enumerate(moves) if i % 2 == 0]
for i, (row, col) in enumerate(black_stones):
    print(f"  Move {2*i + 1}: ({row}, {col})")

# Let's check column 7 specifically since many moves were there
print("\nColumn 7 analysis:")
col7_stones = [(i, row) for row in range(15) for i in [1, -1] if board[row, 7] == i]
for player, row in sorted(col7_stones, key=lambda x: x[1]):
    print(f"  Row {row}: {'Black' if player == 1 else 'White'}")