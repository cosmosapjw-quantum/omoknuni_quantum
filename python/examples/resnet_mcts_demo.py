#!/usr/bin/env python3
"""Demo of MCTS with ResNet evaluator"""

import sys
sys.path.append('..')

import torch
import numpy as np
from pathlib import Path

from mcts import (
    HighPerformanceMCTS, HighPerformanceMCTSConfig,
    GameInterface, GameType,
    ResNetEvaluator, create_evaluator_for_game,
    EvaluatorConfig
)


def demo_resnet_evaluator():
    """Demonstrate ResNet evaluator usage"""
    
    print("=== ResNet Evaluator Demo ===\n")
    
    # 1. Create evaluator for Gomoku
    print("1. Creating ResNet evaluator for Gomoku...")
    evaluator = create_evaluator_for_game(
        'gomoku',
        num_blocks=20,  # Standard AlphaZero depth
        num_filters=256,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"   - Model parameters: {evaluator.model.count_parameters():,}")
    print(f"   - Device: {evaluator.device}")
    print(f"   - Architecture: {evaluator.model.metadata.architecture}")
    
    # 2. Test single position evaluation
    print("\n2. Testing single position evaluation...")
    # Create a sample board state (4 channels x 15 x 15 for Gomoku)
    state = np.zeros((4, 15, 15), dtype=np.float32)
    # Set some example positions
    state[0, 7, 7] = 1  # Current player stone
    state[1, 7, 8] = 1  # Opponent stone
    
    legal_mask = np.ones(225, dtype=bool)  # All moves legal except occupied
    legal_mask[7 * 15 + 7] = False
    legal_mask[7 * 15 + 8] = False
    
    policy, value = evaluator.evaluate(state, legal_mask)
    
    print(f"   - Value: {value:.4f}")
    print(f"   - Top 5 policy moves:")
    top_moves = np.argsort(policy)[-5:][::-1]
    for move in top_moves:
        row, col = move // 15, move % 15
        print(f"     Move ({row}, {col}): {policy[move]:.4f}")
    
    # 3. Test batch evaluation
    print("\n3. Testing batch evaluation...")
    batch_size = 32
    states = np.random.randn(batch_size, 4, 15, 15).astype(np.float32)
    legal_masks = np.ones((batch_size, 225), dtype=bool)
    
    policies, values = evaluator.evaluate_batch(states, legal_masks)
    
    print(f"   - Batch size: {batch_size}")
    print(f"   - Policy shape: {policies.shape}")
    print(f"   - Values shape: {values.shape}")
    print(f"   - Mean value: {values.mean():.4f}")
    
    # 4. Save and load checkpoint
    print("\n4. Testing checkpoint save/load...")
    checkpoint_path = "demo_checkpoint.pt"
    
    # Save
    evaluator.save_checkpoint(checkpoint_path, additional_info={
        'training_steps': 1000,
        'game_type': 'gomoku'
    })
    print(f"   - Saved checkpoint to {checkpoint_path}")
    
    # Load
    new_evaluator = ResNetEvaluator.from_checkpoint(checkpoint_path)
    print(f"   - Loaded checkpoint successfully")
    
    # Verify they produce same output
    new_policy, new_value = new_evaluator.evaluate(state, legal_mask)
    print(f"   - Value match: {np.allclose(value, new_value)}")
    print(f"   - Policy match: {np.allclose(policy, new_policy)}")
    
    # Clean up
    Path(checkpoint_path).unlink()
    Path("metadata.json").unlink()
    
    return evaluator


def play_with_resnet_mcts():
    """Play a game using ResNet-based MCTS"""
    
    print("\n\n=== Playing with ResNet MCTS ===\n")
    
    # Initialize game
    game = GameInterface(GameType.GOMOKU, board_size=15)
    
    # Create evaluator
    evaluator = create_evaluator_for_game(
        'gomoku',
        num_blocks=10,  # Smaller for faster demo
        num_filters=128
    )
    
    # MCTS configuration optimized for ResNet
    config = HighPerformanceMCTSConfig(
        num_simulations=800,
        c_puct=1.5,
        temperature=1.0,
        wave_size=512,  # Good balance for ResNet
        enable_gpu=True,  # Enable GPU
        device='cuda' if torch.cuda.is_available() else 'cpu',
        mixed_precision=True  # Enable mixed precision for better GPU performance
    )
    
    # Create MCTS
    mcts = HighPerformanceMCTS(
        game=game,
        evaluator=evaluator,
        config=config
    )
    
    # Play some moves
    state = game.create_initial_state()
    
    print("Playing first 5 moves...")
    for move_num in range(5):
        print(f"\nMove {move_num + 1}:")
        
        # Search
        import time
        start_time = time.time()
        action_probs = mcts.search(state)
        search_time = time.time() - start_time
        
        # Get best move
        if move_num < 2:
            # Use temperature for first moves
            action = np.random.choice(len(action_probs), p=action_probs)
        else:
            # Play deterministically
            action = np.argmax(action_probs)
        
        row, col = action // 15, action % 15
        
        print(f"  - Selected move: ({row}, {col})")
        print(f"  - Move probability: {action_probs[action]:.3f}")
        print(f"  - Search time: {search_time:.2f}s")
        print(f"  - Simulations/second: {config.num_simulations / search_time:.0f}")
        
        # Apply move
        state = game.apply_move(state, action)
        
        # Show top 5 alternatives
        top_actions = np.argsort(action_probs)[-5:][::-1]
        print("  - Top 5 moves:")
        for a in top_actions:
            r, c = a // 15, a % 15
            print(f"    ({r}, {c}): {action_probs[a]:.3f}")
    
    # Get final statistics
    stats = mcts.get_statistics()
    print(f"\nFinal statistics:")
    print(f"  - Total nodes: {stats['total_nodes']:,}")
    print(f"  - Cache hits: {stats.get('cache_hits', 0):,}")
    print(f"  - Average depth: {stats.get('average_depth', 0):.1f}")


def compare_evaluators():
    """Compare MockEvaluator vs ResNetEvaluator performance"""
    
    print("\n\n=== Evaluator Comparison ===\n")
    
    from mcts import MockEvaluator
    import time
    
    # Setup
    batch_sizes = [1, 16, 64, 256]
    num_iterations = 100
    
    # Create evaluators
    mock_eval = MockEvaluator(
        EvaluatorConfig(device='cuda' if torch.cuda.is_available() else 'cpu'),
        action_size=225
    )
    
    resnet_eval = create_evaluator_for_game(
        'gomoku',
        num_blocks=5,  # Small for fair comparison
        num_filters=64
    )
    
    print("Comparing evaluation speed...\n")
    print("Batch Size | Mock (ms) | ResNet (ms) | Ratio")
    print("-" * 45)
    
    for batch_size in batch_sizes:
        # Prepare data
        states = np.random.randn(batch_size, 4, 15, 15).astype(np.float32)
        legal_masks = np.ones((batch_size, 225), dtype=bool)
        
        # Warmup
        for _ in range(10):
            mock_eval.evaluate_batch(states, legal_masks)
            resnet_eval.evaluate_batch(states, legal_masks)
        
        # Benchmark Mock
        start = time.time()
        for _ in range(num_iterations):
            mock_eval.evaluate_batch(states, legal_masks)
        mock_time = (time.time() - start) / num_iterations * 1000
        
        # Benchmark ResNet
        start = time.time()
        for _ in range(num_iterations):
            resnet_eval.evaluate_batch(states, legal_masks)
        resnet_time = (time.time() - start) / num_iterations * 1000
        
        ratio = resnet_time / mock_time
        print(f"{batch_size:10d} | {mock_time:9.2f} | {resnet_time:11.2f} | {ratio:5.2f}x")
    
    print("\nNote: ResNet provides learned position evaluation vs Mock's random values")


if __name__ == "__main__":
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cpu':
        print("WARNING: Running on CPU will be slower. GPU recommended for best performance.\n")
    
    # Run demos
    evaluator = demo_resnet_evaluator()
    play_with_resnet_mcts()
    compare_evaluators()
    
    print("\n\nDemo complete!")