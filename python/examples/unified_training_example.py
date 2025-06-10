#!/usr/bin/env python3
"""Example script demonstrating the unified training pipeline

This script shows how to use the consolidated training pipeline with:
- Parallel self-play with progress tracking
- Neural network training with tqdm progress bars
- Arena evaluation and ELO tracking
- Minimal logging in production mode
"""

import argparse
import logging
from pathlib import Path

from mcts.neural_networks import UnifiedTrainingPipeline
from mcts.utils.config_system import create_default_config, QuantumLevel


def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero with unified pipeline")
    parser.add_argument("--game", type=str, default="gomoku", 
                        choices=["chess", "go", "gomoku"],
                        help="Game to train on")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--games-per-iter", type=int, default=100,
                        help="Self-play games per iteration")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--quantum", type=str, default="classical",
                        choices=["classical", "tree", "one_loop"],
                        help="Quantum MCTS level")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Experiment name (default: auto-generated)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Create experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"{args.game}_unified_{args.quantum}"
    
    # Create configuration
    config = create_default_config(
        game_type=args.game,
        experiment_name=args.experiment_name
    )
    
    # Set log level based on debug flag
    config.log_level = "DEBUG" if args.debug else "INFO"
    
    # Configure training parameters
    config.training.num_workers = args.workers
    config.training.num_games_per_iteration = args.games_per_iter
    
    # Configure MCTS for optimal performance
    config.mcts.min_wave_size = 3072
    config.mcts.max_wave_size = 3072
    config.mcts.adaptive_wave_sizing = False  # Critical for performance
    config.mcts.num_simulations = 800
    
    # Enable arena evaluation
    config.arena.enabled = True
    config.arena.evaluation_interval = 10
    config.arena.num_games = 40
    config.arena.num_workers = args.workers
    
    # Configure quantum features
    if args.quantum != "classical":
        config.quantum.enabled = True
        config.quantum.quantum_level = QuantumLevel[args.quantum.upper()]
    
    # Mixed precision for faster training
    config.training.mixed_precision = True
    
    # Create and run pipeline
    print(f"Starting {args.game} training with {args.quantum} MCTS")
    print(f"Experiment: {args.experiment_name}")
    print(f"Workers: {args.workers}, Games/iter: {args.games_per_iter}")
    print("-" * 60)
    
    pipeline = UnifiedTrainingPipeline(config, resume_from=args.resume)
    
    try:
        pipeline.train(num_iterations=args.iterations)
        
        # Run final tournament if we have enough models
        if config.arena.run_final_tournament:
            print("\nRunning final tournament...")
            pipeline.run_final_tournament()
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving checkpoint...")
        pipeline.save_checkpoint()
        print("Checkpoint saved. Use --resume to continue.")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()