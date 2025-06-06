#!/usr/bin/env python3
"""Demo showing the 20-channel neural network input format"""

import sys
sys.path.append('..')

import numpy as np
from mcts import GameInterface, GameType

def demonstrate_nn_input():
    """Demonstrate the 20-channel neural network input format"""
    
    print("=== Neural Network Input Format Demo ===\n")
    
    # Create games for each type
    games = {
        'Chess': GameInterface(GameType.CHESS),
        'Go': GameInterface(GameType.GO, board_size=19),
        'Gomoku': GameInterface(GameType.GOMOKU, board_size=15)
    }
    
    for game_name, game in games.items():
        print(f"\n{game_name}:")
        print("-" * 40)
        
        # Create initial state
        state = game.create_initial_state()
        
        # Get enhanced tensor representation (20 channels)
        nn_input = game.state_to_numpy(state, use_enhanced=True)
        
        print(f"Input shape: {nn_input.shape}")
        print(f"Expected: (20, {game.board_shape[0]}, {game.board_shape[1]})")
        print(f"Data type: {nn_input.dtype}")
        
        # Describe the channels
        print("\nChannel breakdown:")
        print("  Channel 0: Current board state (current player's pieces)")
        print("  Channel 1: Current player indicator (0 or 1 filled plane)")
        print("  Channels 2-9: Last 8 moves by player 1")
        print("  Channels 10-17: Last 8 moves by player 2")
        print("  Channel 18: Attack score plane")
        print("  Channel 19: Defense score plane")
        
        # Check some channel statistics
        print("\nChannel statistics:")
        for i in range(min(5, nn_input.shape[0])):  # Show first 5 channels
            channel = nn_input[i]
            print(f"  Channel {i}: min={channel.min():.2f}, max={channel.max():.2f}, "
                  f"mean={channel.mean():.4f}, non-zero={np.count_nonzero(channel)}")
        
        # Make a few moves and see how input changes
        if game_name == 'Gomoku':
            print("\nAfter making some moves:")
            # Make center move
            center = 7 * 15 + 7  # Row 7, Col 7
            state = game.apply_move(state, center)
            
            # Make opponent move
            adjacent = 7 * 15 + 8  # Row 7, Col 8
            state = game.apply_move(state, adjacent)
            
            # Get new input
            nn_input_after = game.state_to_numpy(state, use_enhanced=True)
            
            # Show changes
            print(f"  Board channel (0) changed pixels: {np.sum(nn_input_after[0] != nn_input[0])}")
            print(f"  Attack plane (18) changed pixels: {np.sum(nn_input_after[18] != nn_input[18])}")
            print(f"  Defense plane (19) changed pixels: {np.sum(nn_input_after[19] != nn_input[19])}")
            
            # Show attack/defense scores around moves
            print("\n  Attack scores near moves:")
            for r in range(6, 9):
                for c in range(6, 9):
                    score = nn_input_after[18, r, c]
                    if score > 0:
                        print(f"    Position ({r},{c}): {score:.2f}")


def check_consistency():
    """Check that all games produce consistent 20-channel format"""
    
    print("\n\n=== Consistency Check ===\n")
    
    # Test different board sizes
    test_configs = [
        ('Chess', GameType.CHESS, None),
        ('Go 9x9', GameType.GO, 9),
        ('Go 19x19', GameType.GO, 19),
        ('Gomoku 15x15', GameType.GOMOKU, 15),
        ('Gomoku 19x19', GameType.GOMOKU, 19),
    ]
    
    for name, game_type, board_size in test_configs:
        game = GameInterface(game_type, board_size)
        state = game.create_initial_state()
        nn_input = game.state_to_numpy(state, use_enhanced=True)
        
        print(f"{name:15} -> Shape: {nn_input.shape}")
        
        # Verify it's always 20 channels
        assert nn_input.shape[0] == 20, f"{name} doesn't have 20 channels!"
    
    print("\n✓ All games produce 20-channel input format")


def benchmark_tensor_creation():
    """Benchmark enhanced tensor creation performance"""
    
    print("\n\n=== Performance Benchmark ===\n")
    
    import time
    
    game = GameInterface(GameType.GOMOKU, board_size=15)
    state = game.create_initial_state()
    
    # Warmup
    for _ in range(10):
        _ = game.state_to_numpy(state, use_enhanced=True)
    
    # Benchmark
    n_iterations = 1000
    start = time.time()
    
    for _ in range(n_iterations):
        nn_input = game.state_to_numpy(state, use_enhanced=True)
    
    elapsed = time.time() - start
    
    print(f"Enhanced tensor creation:")
    print(f"  {n_iterations} iterations in {elapsed:.3f} seconds")
    print(f"  {elapsed/n_iterations*1000:.3f} ms per tensor")
    print(f"  {n_iterations/elapsed:.0f} tensors/second")
    
    # Compare with basic representation
    start = time.time()
    for _ in range(n_iterations):
        basic = game.state_to_numpy(state, use_enhanced=False)
    elapsed_basic = time.time() - start
    
    print(f"\nBasic tensor creation:")
    print(f"  {n_iterations} iterations in {elapsed_basic:.3f} seconds")
    print(f"  {elapsed_basic/n_iterations*1000:.3f} ms per tensor")
    
    print(f"\nEnhanced is {elapsed/elapsed_basic:.1f}x slower (includes attack/defense computation)")


if __name__ == "__main__":
    try:
        demonstrate_nn_input()
        check_consistency()
        benchmark_tensor_creation()
    except Exception as e:
        print(f"\nNote: This demo requires the C++ game implementations to be built.")
        print(f"Error: {e}")
        print("\nTo build the C++ module:")
        print("  cd /home/cosmos/omoknuni_quantum")
        print("  mkdir build && cd build")
        print("  cmake .. -DBUILD_PYTHON_BINDINGS=ON")
        print("  make -j$(nproc)")
    
    print("\nDemo complete!")