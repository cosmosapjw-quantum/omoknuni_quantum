#!/usr/bin/env python3
"""
Convenient script to run the unified training pipeline.
Run from the project root directory.
"""

import sys
import os
import time
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

def _detect_cuda_kernels(device: str):
    """Detect pre-compiled CUDA kernels (no compilation)"""
    # Skip if CUDA kernels are disabled
    if os.environ.get('DISABLE_CUDA_KERNELS', '0') == '1':
        print("  ⚠️  CUDA kernels disabled - using CPU fallback")
        return
        
    try:
        print("  📦 Detecting pre-compiled CUDA kernels...")
        print(f"  🔧 Process ID: {os.getpid()}")
        
        from mcts.gpu.cuda_manager import detect_cuda_kernels as get_cuda_module
        
        start_time = time.time()
        module = get_cuda_module()
        detection_time = time.time() - start_time
        
        if module:
            print(f"  ✅ CUDA kernels detected successfully in {detection_time:.3f}s")
            
            # Verify critical functions are available
            critical_functions = ['batched_ucb_selection', 'parallel_backup', 'vectorized_backup']
            available = []
            for func in critical_functions:
                if hasattr(module, func):
                    available.append(func)
            
            print(f"  🎯 Available functions: {len(available)}/{len(critical_functions)}")
            if len(available) < len(critical_functions):
                missing = [f for f in critical_functions if f not in available]
                print(f"  ⚠️  Missing functions: {missing}")
                print("  💡 Run 'python setup.py install' to compile missing functions")
        else:
            print("  ⚠️  No pre-compiled CUDA kernels found")
            print("  💡 Run 'python setup.py install' to compile CUDA kernels")
            print("  🔄 Falling back to CPU implementations")
            
    except Exception as e:
        print(f"  ⚠️  Error detecting CUDA kernels: {e}")
        print("  🔄 Falling back to CPU implementations")

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
    
    # Detect pre-compiled CUDA kernels and perform GPU warmup
    if config.mcts.device == 'cuda':
        print("🔧 Detecting CUDA kernels...")
        
        # Clean up any stale marker files from previous runs
        import tempfile
        marker_file = os.path.join(tempfile.gettempdir(), 'cuda_singleton_compiled.marker')
        if os.path.exists(marker_file):
            try:
                os.remove(marker_file)
                print("  🧹 Cleaned up stale marker file")
            except:
                pass
        
        _detect_cuda_kernels(config.mcts.device)
    
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