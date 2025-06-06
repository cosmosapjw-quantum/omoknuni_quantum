#!/usr/bin/env python3
"""Simple demo of MCTS for game playing"""

import sys
sys.path.append('..')

import torch

from mcts import (
    HighPerformanceMCTS as MCTS, HighPerformanceMCTSConfig,
    GameInterface, GameType,
    MockEvaluator, ResNetEvaluator, create_evaluator_for_game, EvaluatorConfig,
    MemoryConfig, MCTSConfig
)


def play_game(use_resnet=True):
    """Play a simple game using MCTS
    
    Args:
        use_resnet: If True, use ResNetEvaluator; if False, use MockEvaluator
    """
    
    # Initialize components
    print("Initializing MCTS components...")
    
    # Game: 9x9 Gomoku
    game = GameInterface(GameType.GOMOKU, board_size=9)
    
    # Evaluator: ResNet or Mock neural network
    if use_resnet:
        print("Using ResNet evaluator...")
        # Create ResNet evaluator for 9x9 Gomoku
        # Note: In practice, you'd load a trained model checkpoint
        evaluator = create_evaluator_for_game(
            'gomoku',
            num_blocks=10,  # Smaller network for demo
            num_filters=128,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    else:
        print("Using Mock evaluator...")
        evaluator = MockEvaluator(
            EvaluatorConfig(batch_size=256),
            action_size=81  # 9x9 board
        )
    
    # MCTS configuration
    mcts_config = HighPerformanceMCTSConfig(
        num_simulations=400,
        c_puct=1.0,
        temperature=1.0,
        wave_size=256,
        enable_gpu=True,  # Enable GPU
        device='cuda' if torch.cuda.is_available() else 'cpu',
        mixed_precision=True  # Enable mixed precision for better GPU performance
    )
    
    # Initialize MCTS
    mcts = MCTS(
        game=game,
        evaluator=evaluator,
        config=mcts_config,
        memory_config=MemoryConfig.laptop_preset()
    )
    
    # Play a few moves
    state = game.create_initial_state()
    move_history = []
    
    print("\nStarting game...")
    print("Board size: 9x9")
    print("Using vectorized MCTS with wave size 256")
    print()
    
    for move_num in range(10):
        print(f"Move {move_num + 1}:")
        
        # Run MCTS search
        root = mcts.search(state)
        
        # Get action probabilities
        probs = mcts.get_action_probabilities(root, temperature=1.0)
        
        # Select best action
        best_action = mcts.get_best_action(root)
        move_history.append(best_action)
        
        # Display info
        print(f"  Best move: {best_action} (row {best_action//9}, col {best_action%9})")
        print(f"  Visit count: {root.children[best_action].visit_count}")
        print(f"  Win rate: {root.children[best_action].value():.3f}")
        print(f"  Total simulations: {root.visit_count}")
        
        # Get top 5 moves
        top_moves = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
        print("  Top 5 moves:")
        for move, prob in top_moves:
            if move in root.children:
                child = root.children[move]
                print(f"    Move {move}: prob={prob:.3f}, visits={child.visit_count}, value={child.value():.3f}")
        
        # Apply move
        state = game.apply_move(state, best_action)
        
        # Check if game ended
        if game.is_terminal(state):
            winner = game.get_winner(state)
            print(f"\nGame ended! Winner: {winner}")
            break
            
        print()
    
    # Display statistics
    print("\nMCTS Statistics:")
    stats = mcts.get_statistics()
    print(f"  Total simulations: {stats['total_simulations']}")
    print(f"  Nodes created: {stats['total_nodes']}")
    print(f"  Search time: {stats['search_time']:.3f}s")
    print(f"  Simulations/second: {stats['simulations_per_second']:.0f}")
    
    if 'average_wave_size' in stats:
        print(f"  Average wave size: {stats['average_wave_size']:.0f}")
        print(f"  Total waves: {stats['total_waves']}")


def benchmark_performance(use_resnet=False):
    """Benchmark MCTS performance
    
    Args:
        use_resnet: If True, use ResNetEvaluator; if False, use MockEvaluator
    """
    import time
    
    print("\nBenchmarking MCTS performance...")
    
    # Test different configurations
    configs = [
        ("Wave MCTS (256)", HighPerformanceMCTSConfig(num_simulations=1000, wave_config={'wave_size': 256})),
        ("Wave MCTS (512)", HighPerformanceMCTSConfig(num_simulations=1000, wave_config={'wave_size': 512})),
        ("Wave MCTS (1024)", HighPerformanceMCTSConfig(num_simulations=1000, wave_config={'wave_size': 1024})),
    ]
    
    game = GameInterface(GameType.GO, board_size=9)
    
    if use_resnet:
        print("Using ResNet evaluator for benchmark...")
        evaluator = create_evaluator_for_game(
            'go',
            num_blocks=5,  # Small network for benchmarking
            num_filters=64,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    else:
        print("Using Mock evaluator for benchmark...")
        evaluator = MockEvaluator(EvaluatorConfig(), action_size=82)
    
    for name, config in configs:
        mcts = MCTS(game, evaluator, config)
        state = game.create_initial_state()
        
        # Warmup
        mcts.search(state)
        
        # Benchmark
        start = time.time()
        root = mcts.search(state)
        elapsed = time.time() - start
        
        sims_per_sec = config.num_simulations / elapsed
        print(f"\n{name}:")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Simulations/second: {sims_per_sec:.0f}")
        print(f"  Nodes created: {mcts.get_statistics()['total_nodes']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Omoknuni MCTS Demo")
    parser.add_argument('--use-resnet', action='store_true',
                        help='Use ResNet evaluator instead of Mock evaluator')
    parser.add_argument('--benchmark-only', action='store_true',
                        help='Only run benchmark, skip playing')
    args = parser.parse_args()
    
    print("=== Omoknuni MCTS Demo ===\n")
    
    if not args.benchmark_only:
        # Play a game
        play_game(use_resnet=args.use_resnet)
    
    # Run benchmark
    benchmark_performance(use_resnet=args.use_resnet)
    
    print("\nDemo complete!")