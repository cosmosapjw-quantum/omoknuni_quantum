#!/usr/bin/env python3
"""Example training script for Gomoku using AlphaZero self-play

This example demonstrates how to use the training pipeline to train
a neural network for playing Gomoku through self-play reinforcement learning.
"""

import os
import sys
import logging
import argparse
import multiprocessing

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mcts.neural_networks.training_pipeline import TrainingPipeline, TrainingConfig
from mcts.core.game_interface import GameInterface, GameType


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_gomoku_game():
    """Factory function for creating Gomoku game instances"""
    return GameInterface(GameType.GOMOKU, board_size=15)


def main():
    parser = argparse.ArgumentParser(description='Train Gomoku AI using AlphaZero')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of training iterations')
    parser.add_argument('--games-per-iteration', type=int, default=100,
                        help='Self-play games per iteration')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Training batch size')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel self-play workers')
    parser.add_argument('--simulations', type=int, default=800,
                        help='MCTS simulations per move')
    parser.add_argument('--save-dir', type=str, default='checkpoints/gomoku',
                        help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Create training configuration
    config = TrainingConfig(
        game_type='gomoku',
        batch_size=args.batch_size,
        num_games_per_iteration=args.games_per_iteration,
        num_workers=args.workers,
        mcts_simulations=args.simulations,
        save_dir=args.save_dir,
        device=args.device,
        
        # Gomoku-specific settings
        max_moves_per_game=225,  # 15x15 board
        temperature_threshold=15,  # Switch to greedy after 15 moves
        
        # Training hyperparameters
        learning_rate=0.01,
        weight_decay=1e-4,
        num_epochs=10,
        window_size=500000,
        checkpoint_interval=10,
        
        # Mixed precision for faster training
        mixed_precision=True if args.device == 'cuda' else False,
        gradient_accumulation_steps=2,
        max_grad_norm=5.0
    )
    
    # Create training pipeline
    logger.info("Creating training pipeline...")
    pipeline = TrainingPipeline(config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        pipeline.load_checkpoint(args.checkpoint)
    
    # Start training
    logger.info("Starting training...")
    logger.info(f"Configuration:")
    logger.info(f"  - Iterations: {args.iterations}")
    logger.info(f"  - Games per iteration: {config.num_games_per_iteration}")
    logger.info(f"  - Batch size: {config.batch_size}")
    logger.info(f"  - Workers: {config.num_workers}")
    logger.info(f"  - MCTS simulations: {config.mcts_simulations}")
    logger.info(f"  - Device: {config.device}")
    logger.info(f"  - Save directory: {config.save_dir}")
    
    try:
        # Run training loop
        pipeline.run_training_loop(create_gomoku_game, args.iterations)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    
    logger.info("Training complete!")


if __name__ == "__main__":
    # Set spawn method for CUDA compatibility
    multiprocessing.set_start_method('spawn', force=True)
    main()