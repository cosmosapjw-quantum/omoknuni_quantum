#!/usr/bin/env python3
"""Test the alphazero_py.GomokuState interface"""

import alphazero_py
import numpy as np

# Create game
game = alphazero_py.GomokuState()

print("=== Testing GomokuState Interface ===")
print(f"Type: {type(game)}")
print(f"Dir: {[attr for attr in dir(game) if not attr.startswith('_')]}")

# Test methods
print("\nTesting methods:")

# Make a move
print(f"Making move at position 112...")
game.make_move(112)

# Try different ways to get the board
print("\nTrying to access board state:")

# Method 1: Direct attributes
for attr in ['board', 'get_board', 'to_tensor', 'get_state']:
    if hasattr(game, attr):
        print(f"  Has {attr}: Yes")
        try:
            value = getattr(game, attr)
            if callable(value):
                result = value()
                print(f"    Result type: {type(result)}")
                if hasattr(result, 'shape'):
                    print(f"    Shape: {result.shape}")
            else:
                print(f"    Value type: {type(value)}")
        except Exception as e:
            print(f"    Error: {e}")
    else:
        print(f"  Has {attr}: No")

# Method 2: Check for __getitem__ or array interface
if hasattr(game, '__getitem__'):
    print("\n  Has __getitem__: Yes")
    try:
        # Try accessing like an array
        value = game[0]
        print(f"    game[0] = {value}")
    except Exception as e:
        print(f"    Error accessing game[0]: {e}")

# Method 3: Check string representation
print(f"\nString representation:")
print(f"  str(game): {str(game)[:100]}...")

# Method 4: Try to get observation
if hasattr(game, 'get_observation'):
    print(f"\n  Has get_observation: Yes")
    try:
        obs = game.get_observation()
        print(f"    Type: {type(obs)}")
        if hasattr(obs, 'shape'):
            print(f"    Shape: {obs.shape}")
    except Exception as e:
        print(f"    Error: {e}")

# Method 5: Check for canonical board
if hasattr(game, 'get_canonical_board'):
    print(f"\n  Has get_canonical_board: Yes")
    try:
        board = game.get_canonical_board()
        print(f"    Type: {type(board)}")
        if hasattr(board, 'shape'):
            print(f"    Shape: {board.shape}")
    except Exception as e:
        print(f"    Error: {e}")

# Test the tensor representation methods
print("\n=== Testing tensor representation methods ===")

# Reset game
game = alphazero_py.GomokuState()

# Get initial tensor
print("\nInitial state:")
tensor_repr = game.get_tensor_representation()
print(f"  Type: {type(tensor_repr)}")
print(f"  Shape: {np.array(tensor_repr).shape if hasattr(tensor_repr, '__len__') else 'N/A'}")

# Make a move and check again
game.make_move(112)  # Center position
print("\nAfter move at position 112:")
tensor_repr = game.get_tensor_representation()
tensor_array = np.array(tensor_repr)
print(f"  Type: {type(tensor_repr)}")
print(f"  Shape: {tensor_array.shape}")
print(f"  Non-zero positions: {np.count_nonzero(tensor_array)}")

# Check enhanced version
print("\nEnhanced tensor representation:")
enhanced = game.get_enhanced_tensor_representation()
enhanced_array = np.array(enhanced)
print(f"  Type: {type(enhanced)}")
print(f"  Shape: {enhanced_array.shape}")

# Show the board around center
if tensor_array.shape == (15, 15):
    print("\nBoard center (5x5):")
    for i in range(5, 10):
        row = []
        for j in range(5, 10):
            row.append(str(int(tensor_array[i, j])))
        print("  " + " ".join(row))

print("\n=== Summary ===")
print("Use game.get_tensor_representation() to get board state as numpy array")