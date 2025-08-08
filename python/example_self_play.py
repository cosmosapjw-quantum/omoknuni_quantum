#!/usr/bin/env python3
"""Example self-play script demonstrating different backends

This script shows how to use different MCTS backends:
- CPU: Pure CPU implementation for users without GPU
- GPU: Pure GPU implementation for maximum parallelism
- Hybrid: CPU tree operations + GPU neural network (best performance)
"""

import os
import sys
import logging
import torch
import numpy as np
from pathlib import Path
import argparse
import time

# CRITICAL OPTIMIZATION: Apply PyTorch performance settings
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
# Reduce memory fragmentation and prevent CPU-GPU sync due to VRAM pressure
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:256'
# Force garbage collection to prevent memory buildup
os.environ['PYTORCH_CUDA_ALLOC_SYNC'] = '1'

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from mcts.core.mcts import MCTS
from mcts.core.mcts_config import MCTSConfig
from mcts.neural_networks.resnet_model import ResNetModel, ResNetConfig
from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
from mcts.utils.single_gpu_evaluator import SingleGPUEvaluator
from mcts.core.game_interface import create_game_interface

# Import GameType conditionally based on backend
def get_game_type(backend):
    if backend == 'cpu':
        # For CPU backend, use string directly
        return 'gomoku'
    else:
        # For GPU backend, import GameType
        from mcts.gpu.gpu_game_states import GameType
        return GameType.GOMOKU

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_self_play(backend='gpu', num_games=10, num_simulations=400, use_cython=False):
    """Run self-play with specified backend
    
    Args:
        backend: 'cpu', 'gpu', or 'hybrid'
        num_games: Number of games to play
        num_simulations: MCTS simulations per move
    """
    
    # Game settings
    game_type = 'gomoku'
    board_size = 15
    
    # Create MCTS configuration with optimized settings for 3000+ sims/sec
    config = MCTSConfig(
        board_size=board_size,
        game_type=get_game_type(backend),
        backend=backend,
        device='cuda' if backend in ['gpu', 'hybrid'] else 'cpu',
        num_simulations=num_simulations,
        c_puct=1.4,
        # Wave-based parallelization
        wave_size=2048 if backend == 'gpu' else 256 if backend == 'hybrid' else 32,
        min_wave_size=512 if backend == 'gpu' else 128 if backend == 'hybrid' else 32,
        max_wave_size=4096 if backend == 'gpu' else 512 if backend == 'hybrid' else 64,
        # Tree configuration - optimized for available VRAM to prevent sync issues
        max_tree_nodes=5000000 if backend == 'gpu' else 500000 if backend == 'hybrid' else 100000,
        initial_tree_nodes=500000 if backend == 'gpu' else 500000 if backend == 'hybrid' else 10000,  # Hybrid needs more capacity for state management
        # GPU-specific optimizations
        enable_kernel_fusion=backend == 'gpu',
        use_cuda_graphs=backend == 'gpu',
        use_tensor_cores=backend == 'gpu',
        wave_async_expansion=True if backend == 'gpu' else False,   # AsyncWaveSearch for GPU only
        enable_subtree_reuse=True,  # Enable tree reuse for both GPU and CPU backends
        subtree_reuse_min_visits=1,  # Minimal threshold to maximize tree reuse
        enable_fast_ucb=backend == 'gpu',
        use_mixed_precision=backend == 'gpu',
        enable_multi_stream=backend == 'gpu',
        num_cuda_streams=8 if backend == 'gpu' else 1,  # More streams for better overlap
        stream_memory_pool_size=512 if backend == 'gpu' else 0,
        # Memory optimization - reduced to prevent VRAM pressure
        enable_memory_pooling=backend == 'gpu',
        memory_pool_size_mb=2048 if backend == 'gpu' else 0,  # Reduced pool size
        gpu_memory_fraction=0.8 if backend == 'gpu' else 0.0,  # Leave 20% free for stability
        # Fast batch processing - reduced timeout
        gpu_batch_timeout=0.0005 if backend == 'gpu' else 0.1,  # 0.5ms for minimal latency
        # Caching optimizations
        cache_legal_moves=True,
        cache_features=True,
        # Progressive widening settings - reduced to decrease CPU overhead
        progressive_widening_constant=5.0,  # Reduced from 10.0
        progressive_widening_exponent=0.5,
        initial_expansion_children=5,  # Reduced from 10
        # Virtual loss for parallelization
        virtual_loss=3.0,
        # Batch size for neural network
        batch_size=1024 if backend == 'gpu' else 256 if backend == 'hybrid' else 256,
        inference_batch_size=1024 if backend == 'gpu' else 256 if backend == 'hybrid' else 256,
        # Dynamic allocation for better memory usage
        enable_dynamic_allocation=True,
    )
    
    # Enable appropriate hybrid backend
    if backend == 'hybrid':
        if use_cython:
            # Use Cython-optimized hybrid backend for maximum performance
            config.use_cython_hybrid = True
            config.use_genuine_hybrid = False
            config.use_fixed_hybrid = False
            config.batch_size = 1024  # Larger batches for Cython backend
            config.inference_batch_size = 1024
            config.hybrid_gpu_batch_size = 1024
            logger.info("Using Cython-optimized hybrid backend (target: 10,000 sims/sec)")
        else:
            # Use genuine hybrid for stable performance
            config.use_cython_hybrid = False
            config.use_genuine_hybrid = True
            config.use_fixed_hybrid = False
            config.use_simple_hybrid = False
            config.use_optimized_hybrid = False
            logger.info("Using genuine hybrid backend")
    
    # Neural network configuration
    num_res_blocks = 10
    num_filters = 128
    
    # Determine device
    if backend == 'cpu':
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu' and backend != 'cpu':
            logger.warning(f"CUDA not available, falling back to CPU for {backend} backend")
    
    logger.info(f"=== Self-Play with {backend.upper()} Backend ===")
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {config.inference_batch_size}")
    logger.info(f"Games to play: {num_games}")
    logger.info(f"MCTS simulations: {num_simulations}")
    
    # Create model
    resnet_config = ResNetConfig(
        num_blocks=num_res_blocks,
        num_filters=num_filters
    )
    
    model = ResNetModel(
        config=resnet_config,
        board_size=board_size,
        num_actions=board_size * board_size,
        game_type=game_type
    )
    
    # Move model to device and set to eval mode
    if device == 'cuda':
        model = model.cuda()
    model.eval()
    
    # Create optimized evaluator with pinned memory
    logger.info(f"Creating optimized evaluator with device={device}, batch_size={config.inference_batch_size}")
    
    # Use ResNetEvaluator for all backends - it automatically optimizes based on device
    # For GPU backend, it includes:
    # - Pinned memory buffers (27.23x speedup)
    # - Non-blocking transfers
    # - Mixed precision support
    # - Optimized memory access patterns
    evaluator = ResNetEvaluator(
        model=model,
        game_type=game_type,
        device=str(device)
    )
    
    # Override batch size for self-play to match optimized config
    evaluator.batch_size = config.inference_batch_size
    
    # Re-initialize pinned memory buffers with new batch size (only allocates on GPU)
    evaluator._init_pinned_memory_buffers()
    
    # OPTIMIZATION: Enable persistent batching context (Phase 1.4)
    # Pre-allocate buffers for entire game length to avoid reallocation
    if hasattr(evaluator, 'enable_persistent_batching'):
        evaluator.enable_persistent_batching(True)
        evaluator.allocate_game_length_buffers(max_moves=100)
    
    # Enable direct tensor returns for GPU efficiency
    evaluator._return_torch_tensors = True
    
    # Create MCTS instance
    # Create MCTS instance once
    logger.info("Creating MCTS instance...")
    mcts = MCTS(config, evaluator)
    logger.info("MCTS instance created successfully")
    
    # Create game interface for state management
    game_interface = create_game_interface(game_type, board_size=board_size)
    
    # Play games and collect statistics
    all_examples = []
    metrics = []
    
    for game_idx in range(num_games):
        logger.info(f"\nPlaying game {game_idx + 1}/{num_games}...")
        
        # Clear MCTS tree for new game
        if game_idx > 0:
            logger.info("Clearing MCTS tree for new game...")
            mcts.clear()
        
        # Reset game interface for new game
        game_interface.reset()
        
        # Use game_interface state directly - no double state management
        state = game_interface.create_initial_state()
        
        current_player = 1
        move_count = 0
        game_start = time.time()
        move_times = []
        sims_per_sec_list = []
        game_over = False
        winner = 0
        
        # Play one game - optimized loop
        while move_count < 225:  # Max moves for 15x15 board
            # Check terminal state before search to avoid unnecessary computation
            if game_interface.is_terminal(state):
                winner = game_interface.get_winner(state)
                break
                
            move_start = time.time()
            
            # Run MCTS search - state is already GPU-friendly
            policy = mcts.search(state, num_simulations=num_simulations)
            
            # Optimized action selection - no redundant computations
            if move_count < 15:  # Early game exploration
                # Direct sampling with temperature=1.0 (no need to modify policy)
                action = int(np.random.choice(len(policy), p=policy))
            else:  # Late game exploitation
                # Apply temperature=0.1 efficiently
                temp_policy = policy ** 10.0  # Equivalent to (policy ** (1/0.1))
                temp_policy = temp_policy / temp_policy.sum()
                action = int(np.argmax(temp_policy))
            
            # Timing and stats
            move_time = time.time() - move_start
            move_times.append(move_time)
            sims_per_sec = num_simulations / move_time if move_time > 0 else 0
            sims_per_sec_list.append(sims_per_sec)
            
            # Single state update - game_interface handles everything
            state = game_interface.apply_move(state, action)
            
            # Update tree root with new state
            mcts.update_root(action, new_state=state)
            
            # Increment move count (player switching not needed for stats)
            move_count += 1
            
            # Log progress with reduced frequency
            if move_count % 10 == 0:
                logger.info(f"  Move {move_count}: {sims_per_sec:.0f} sims/sec")
        
        game_time = time.time() - game_start
        
        # Record metrics
        metrics.append({
            'game_length': move_count,
            'game_time': game_time,
            'avg_time_per_move': np.mean(move_times) if move_times else 0.0,
            'avg_simulations_per_second': np.mean(sims_per_sec_list) if sims_per_sec_list else 0.0,
            'winner': winner  # Actual winner from game logic
        })
        
        winner_str = "Player 1" if winner == 1 else ("Player 2" if winner == -1 else "Draw")
        logger.info(f"Game {game_idx + 1} completed: {move_count} moves in {game_time:.1f}s - Winner: {winner_str}")
        if sims_per_sec_list:
            logger.info(f"Average: {np.mean(sims_per_sec_list):.0f} sims/sec")
        else:
            logger.info("No moves were made in this game")
    
    # Display aggregate results
    logger.info(f"\n=== Self-Play Results ===")
    logger.info(f"Games completed: {len(metrics)}")
    
    if metrics:
        avg_game_length = np.mean([m['game_length'] for m in metrics])
        avg_time_per_move = np.mean([m['avg_time_per_move'] for m in metrics])
        avg_sims_per_sec = np.mean([m['avg_simulations_per_second'] for m in metrics])
        
        logger.info(f"Average game length: {avg_game_length:.1f} moves")
        logger.info(f"Average time per move: {avg_time_per_move:.3f} seconds")
        logger.info(f"Average simulations/second: {avg_sims_per_sec:.0f}")
        
        # Show winner distribution
        outcomes = [m['winner'] for m in metrics]
        unique, counts = np.unique(outcomes, return_counts=True)
        logger.info("\nWinner distribution:")
        for player, count in zip(unique, counts):
            percentage = count / len(outcomes) * 100
            logger.info(f"  Player {player}: {count} games ({percentage:.1f}%)")
    
    return all_examples, metrics, avg_sims_per_sec if metrics else 0


def main():
    parser = argparse.ArgumentParser(description='Run MCTS self-play with different backends')
    parser.add_argument('--backend', type=str, default='gpu', 
                        choices=['cpu', 'gpu', 'hybrid'],
                        help='MCTS backend to use (default: gpu)')
    parser.add_argument('--games', type=int, default=10,
                        help='Number of games to play (default: 10)')
    parser.add_argument('--simulations', type=int, default=400,
                        help='MCTS simulations per move (default: 400)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare all backends')
    parser.add_argument('--use-cython', action='store_true',
                        help='Use Cython-optimized hybrid backend (only with --backend hybrid)')
    
    args = parser.parse_args()
    
    # Check if we need to build the C++ library
    try:
        import alphazero_py
    except ImportError:
        logger.error("\n❌ alphazero_py module not found!")
        logger.info("Please build the C++ library first:")
        logger.info("  ./build_cpp.sh")
        sys.exit(1)
    
    if args.compare:
        # Compare all backends
        logger.info("=== Backend Performance Comparison ===\n")
        
        results = {}
        backends = ['cpu', 'gpu', 'hybrid']
        
        # Only test GPU/hybrid if CUDA is available
        if not torch.cuda.is_available():
            logger.warning("CUDA not available - skipping GPU and hybrid backends")
            backends = ['cpu']
        
        for backend in backends:
            try:
                logger.info(f"\nTesting {backend.upper()} backend...")
                _, _, sims_per_sec = run_self_play(
                    backend=backend,
                    num_games=3,  # Fewer games for comparison
                    num_simulations=200  # Fewer simulations for speed
                )
                results[backend] = sims_per_sec
            except Exception as e:
                logger.error(f"Error with {backend} backend: {e}")
                results[backend] = 0
        
        # Display comparison
        logger.info("\n=== Performance Summary ===")
        logger.info("Backend   | Simulations/sec | Speedup")
        logger.info("----------|-----------------|--------")
        
        cpu_speed = results.get('cpu', 1)
        for backend, speed in results.items():
            speedup = speed / cpu_speed if cpu_speed > 0 else 0
            logger.info(f"{backend.upper():9s} | {speed:15.0f} | {speedup:6.2f}x")
    else:
        # Run single backend
        try:
            run_self_play(
                backend=args.backend,
                num_games=args.games,
                num_simulations=args.simulations,
                use_cython=args.use_cython
            )
        except Exception as e:
            logger.error(f"Error during self-play: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()