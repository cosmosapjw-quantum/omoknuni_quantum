#!/usr/bin/env python3
"""
Convenient script to run the unified training pipeline.
Run from the project root directory.
"""

import sys
import os
import multiprocessing
import warnings

# Suppress multiprocessing resource tracker warnings
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')

# Set multiprocessing start method to 'spawn' for CUDA compatibility
multiprocessing.set_start_method('spawn', force=True)

# Add python directory to path
python_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'python')
sys.path.insert(0, python_dir)

# Change to python directory for proper imports
os.chdir(python_dir)

# Import and run the training pipeline
from mcts.neural_networks.unified_training_pipeline import UnifiedTrainingPipeline
from mcts.utils.config_system import AlphaZeroConfig, create_default_config
import argparse

def _precompile_cuda_kernels(device: str):
    """Pre-compile CUDA kernels to avoid JIT overhead during training"""
    try:
        from mcts.gpu.unified_kernels import get_unified_kernels
        from mcts.core.mcts import MCTS, MCTSConfig
        from mcts.neural_networks.mock_evaluator import MockEvaluator
        import torch
        
        # Force kernel loading
        print("  📦 Loading unified kernels...")
        kernels = get_unified_kernels(torch.device(device))
        
        # Create minimal MCTS instance to trigger kernel compilation
        print("  ⚙️  Initializing MCTS system...")
        mcts_config = MCTSConfig(
            num_simulations=10,  # Minimal simulations
            wave_size=100,       # Small wave size
            device=device,
            enable_quantum=False  # Keep it simple
        )
        
        evaluator = MockEvaluator()
        mcts = MCTS(mcts_config, evaluator)
        
        # Run a minimal search to trigger any lazy compilation
        print("  🔥 Warming up kernels...")
        import alphazero_py
        game_state = alphazero_py.GomokuState()
        mcts.search(game_state, 5)  # Very small search to trigger compilation
        
        print("  ✅ CUDA kernels pre-compiled successfully!")
        
    except Exception as e:
        print(f"  ⚠️  Kernel pre-compilation failed: {e}")
        print("  📝 Continuing with PyTorch fallback...")

def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero with Unified Pipeline")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--game", type=str, default="gomoku", choices=["chess", "go", "gomoku"],
                        help="Game type (if not using config file)")
    parser.add_argument("--iterations", type=int, help="Number of training iterations (overrides config)")
    parser.add_argument("--experiment", type=str, help="Experiment name")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--workers", type=int, help="Number of parallel workers")
    parser.add_argument("--games-per-iter", type=int, help="Self-play games per iteration")
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        # Handle relative paths from project root
        if not os.path.isabs(args.config):
            args.config = os.path.join('..', args.config)
        config = AlphaZeroConfig.load(args.config)
        if args.experiment:
            config.experiment_name = args.experiment
    else:
        # Create default config
        config = create_default_config(game_type=args.game)
        config.experiment_name = args.experiment or f"{args.game}_unified_training"
        
        # Set some defaults
        config.training.num_workers = args.workers or 4
        config.training.num_games_per_iteration = args.games_per_iter or 100
        config.training.validation_interval = 10
        config.num_iterations = 100  # Default if not using config file
    
    # Override with command line args (these take priority over config file)
    if args.iterations is not None:
        config.num_iterations = args.iterations
    if args.workers:
        config.training.num_workers = args.workers
        config.arena.num_workers = args.workers
    if args.games_per_iter:
        config.training.num_games_per_iteration = args.games_per_iter
    
    print(f"Starting training for {config.game.game_type}")
    print(f"Experiment: {config.experiment_name}")
    print(f"Workers: {config.training.num_workers}")
    print(f"Games per iteration: {config.training.num_games_per_iteration}")
    print(f"Number of iterations: {config.num_iterations}")
    print(f"Device: {config.mcts.device}")
    print("-" * 60)
    
    # Pre-compile CUDA kernels to avoid JIT overhead during training
    if config.mcts.device == 'cuda':
        print("🔧 Pre-compiling CUDA kernels...")
        _precompile_cuda_kernels(config.mcts.device)
    
    # Create and run pipeline
    pipeline = UnifiedTrainingPipeline(config, resume_from=args.resume)
    
    try:
        pipeline.train(num_iterations=config.num_iterations)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        pipeline.save_checkpoint()
        print("Checkpoint saved.")
        return
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()