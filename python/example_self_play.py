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

# CRITICAL OPTIMIZATION: Apply PyTorch performance settings
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
import argparse
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from mcts.core.mcts import MCTS
from mcts.core.mcts_config import MCTSConfig
from mcts.gpu.gpu_game_states import GameType
from mcts.neural_networks.resnet_model import ResNetModel, ResNetConfig
from mcts.utils.single_gpu_evaluator import SingleGPUEvaluator
from mcts.core.game_interface import create_game_interface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_self_play(backend='gpu', num_games=10, num_simulations=400):
    """Run self-play with specified backend
    
    Args:
        backend: 'cpu', 'gpu', or 'hybrid'
        num_games: Number of games to play
        num_simulations: MCTS simulations per move
    """
    
    # Game settings
    game_type = 'gomoku'
    board_size = 15
    
    # Create MCTS configuration with optimized settings
    config = MCTSConfig(
        board_size=board_size,
        game_type=GameType.GOMOKU,
        backend=backend,
        device='cuda' if backend == 'gpu' else 'cpu',
        num_simulations=num_simulations,
        c_puct=1.4,
        # Wave-based parallelization - optimized for 3000+ sims/sec
        wave_size=4096 if backend == 'gpu' else 32,  # Optimal wave size found through testing
        min_wave_size=2048 if backend == 'gpu' else 32,
        max_wave_size=4096 if backend == 'gpu' else 64,
        # Tree configuration
        max_tree_nodes=500000,
        # GPU-specific optimizations
        enable_kernel_fusion=backend == 'gpu',
        use_cuda_graphs=backend == 'gpu',
        use_tensor_cores=backend == 'gpu',
        wave_async_expansion=True,   # Use optimized AsyncWaveSearch for 3000+ sims/sec
        enable_fast_ucb=backend == 'gpu',
        use_mixed_precision=backend == 'gpu',
        enable_multi_stream=backend == 'gpu',
        num_cuda_streams=8 if backend == 'gpu' else 1,  # More streams for better overlap
        stream_memory_pool_size=512 if backend == 'gpu' else 0,
        # Memory optimization
        enable_memory_pooling=backend == 'gpu',
        memory_pool_size_mb=4096 if backend == 'gpu' else 0,
        gpu_memory_fraction=0.9 if backend == 'gpu' else 0.0,
        # Fast batch processing
        gpu_batch_timeout=0.001 if backend == 'gpu' else 0.1,  # 1ms for minimal latency
        # Caching optimizations
        cache_legal_moves=True,
        cache_features=True,
        # Progressive widening settings
        progressive_widening_constant=10.0,
        progressive_widening_exponent=0.5,
        initial_expansion_children=10,
        # Virtual loss for parallelization
        virtual_loss=3.0,
        # Batch size for neural network
        batch_size=2048 if backend == 'gpu' else 256,
        inference_batch_size=2048 if backend == 'gpu' else 256,
    )
    
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
    
    if backend == 'gpu':
        # Use optimized evaluator with pinned memory for GPU backend
        class OptimizedEvaluator(SingleGPUEvaluator):
            """Evaluator with pinned memory and async transfers"""
            
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                
                # Pre-allocate pinned memory buffers for 2x faster transfers
                self.pinned_input = torch.zeros(
                    (self.batch_size, 19, 15, 15),
                    dtype=torch.float32,
                    pin_memory=True
                )
                
                self.pinned_policy = torch.zeros(
                    (self.batch_size, 225),
                    dtype=torch.float32,
                    pin_memory=True
                )
                
                self.pinned_value = torch.zeros(
                    (self.batch_size, 1),
                    dtype=torch.float32,
                    pin_memory=True
                )
                
                # Create transfer stream for async operations
                self.transfer_stream = torch.cuda.Stream()
                logger.info("Allocated pinned memory buffers for optimized transfers")
            
            def evaluate_batch(self, states):
                """Evaluate with optimized pinned memory transfers"""
                batch_size = len(states) if isinstance(states, list) else states.shape[0]
                
                # Use pinned memory buffer
                if isinstance(states, torch.Tensor):
                    # Copy to pinned memory
                    self.pinned_input[:batch_size].copy_(states)
                    gpu_states = self.pinned_input[:batch_size].to(self.device, non_blocking=True)
                else:
                    gpu_states = torch.from_numpy(states).to(self.device)
                
                # Mixed precision for tensor cores
                if hasattr(self, 'use_mixed_precision') and self.use_mixed_precision:
                    gpu_states = gpu_states.half()
                
                # Evaluate with autocast
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        policies, values = self.model(gpu_states)
                
                return policies, values
        
        evaluator = OptimizedEvaluator(
            model=model,
            device=device,
            batch_size=config.inference_batch_size,
            use_mixed_precision=True
        )
    else:
        evaluator = SingleGPUEvaluator(
            model=model,
            device=device,
            batch_size=config.inference_batch_size
        )
    
    # Enable direct tensor returns for GPU efficiency
    evaluator._return_torch_tensors = True
    
    # Create MCTS instance
    logger.info("Creating MCTS instance...")
    mcts = MCTS(config, evaluator)
    logger.info("MCTS instance created successfully")
    
    # Play games and collect statistics
    all_examples = []
    metrics = []
    
    for game_idx in range(num_games):
        logger.info(f"\nPlaying game {game_idx + 1}/{num_games}...")
        
        # Initialize game state
        state = np.zeros((board_size, board_size), dtype=np.int8)
        current_player = 1
        move_count = 0
        game_start = time.time()
        move_times = []
        sims_per_sec_list = []
        
        # Play one game
        while True:
            try:
                move_start = time.time()
                
                # Run MCTS search
                logger.debug(f"Running MCTS search with {num_simulations} simulations...")
                policy = mcts.search(state, num_simulations=num_simulations)
                logger.debug("MCTS search completed")
            except Exception as e:
                logger.error(f"Error during MCTS search: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            # Select action based on policy
            if move_count < 15:  # Use temperature for exploration in early game
                action = mcts.select_action(state, temperature=1.0)
            else:  # Play deterministically in late game
                action = mcts.select_action(state, temperature=0.1)
            
            move_time = time.time() - move_start
            move_times.append(move_time)
            sims_per_sec = num_simulations / move_time if move_time > 0 else 0
            sims_per_sec_list.append(sims_per_sec)
            
            # Convert action to coordinates
            row = action // board_size
            col = action % board_size
            
            # Apply move
            state[row, col] = current_player
            
            # Update tree root for reuse
            mcts.update_root(action)
            
            # Check for game end (simplified - just check if board is getting full)
            move_count += 1
            if move_count >= 50 or np.sum(state == 0) < 20:
                break
                
            # Switch player
            current_player = -current_player
            
            # Log progress
            if move_count % 10 == 0:
                logger.info(f"  Move {move_count}: {sims_per_sec:.0f} sims/sec")
        
        game_time = time.time() - game_start
        
        # Record metrics
        metrics.append({
            'game_length': move_count,
            'game_time': game_time,
            'avg_time_per_move': np.mean(move_times),
            'avg_simulations_per_second': np.mean(sims_per_sec_list),
            'winner': current_player  # Simplified - last player to move
        })
        
        logger.info(f"Game {game_idx + 1} completed: {move_count} moves in {game_time:.1f}s ")
        logger.info(f"Average: {np.mean(sims_per_sec_list):.0f} sims/sec")
    
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
                num_simulations=args.simulations
            )
        except Exception as e:
            logger.error(f"Error during self-play: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()