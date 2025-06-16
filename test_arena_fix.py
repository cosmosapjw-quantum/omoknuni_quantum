#!/usr/bin/env python3
"""Test script to verify arena segfault fix"""

import logging
import torch
import gc
import psutil
import os
from mcts.neural_networks.arena_module import ArenaManager, ArenaConfig
from mcts.neural_networks.nn_model import create_model
from mcts.utils.config_system import create_default_config
from mcts.core.evaluator import RandomEvaluator, EvaluatorConfig

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def monitor_memory():
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"Memory Usage - RSS: {mem_info.rss/1024/1024:.1f}MB, VMS: {mem_info.vms/1024/1024:.1f}MB")
    
    if torch.cuda.is_available():
        logger.info(f"GPU Memory - Allocated: {torch.cuda.memory_allocated()/1024/1024:.1f}MB, Reserved: {torch.cuda.memory_reserved()/1024/1024:.1f}MB")

def test_arena_stability():
    """Test arena stability with the fixes"""
    logger.info("Starting arena stability test...")
    
    # Create config
    config = create_default_config()
    
    # Create arena config with aggressive settings to test stability
    arena_config = ArenaConfig(
        num_games=150,  # More than the 120 that was crashing
        mcts_simulations=100,  # Reduced for faster testing
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    # Set the new attributes directly since they have defaults
    arena_config.gc_frequency = 5  # More frequent GC for testing
    arena_config.enable_tree_reuse = False  # Ensure tree reuse is disabled
    
    # Create arena manager
    arena = ArenaManager(config, arena_config)
    
    # Create a simple model
    model = create_model(
        game_type=config.game.game_type,
        input_height=config.game.board_size,
        input_width=config.game.board_size,
        num_actions=config.game.board_size * config.game.board_size,
        input_channels=config.network.input_channels,
        num_res_blocks=config.network.num_res_blocks,
        num_filters=config.network.num_filters
    )
    model.eval()
    
    # Create random evaluator
    eval_config = EvaluatorConfig(
        device=arena_config.device,
        use_fp16=False
    )
    random_evaluator = RandomEvaluator(
        eval_config,
        config.game.board_size * config.game.board_size
    )
    
    logger.info("Initial memory state:")
    monitor_memory()
    
    try:
        # Run arena evaluation
        logger.info(f"Running arena evaluation with {arena_config.num_games} games...")
        wins, draws, losses = arena.compare_models(
            model, random_evaluator,
            model1_name="test_model",
            model2_name="random",
            num_games=arena_config.num_games,
            silent=False
        )
        
        logger.info(f"Arena completed successfully! Results: {wins}W-{draws}D-{losses}L")
        
        # Final memory state
        logger.info("Final memory state:")
        monitor_memory()
        
        # Force cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Memory after cleanup:")
        monitor_memory()
        
        return True
        
    except Exception as e:
        logger.error(f"Arena test failed with error: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_arena_stability()
    if success:
        logger.info("✓ Arena stability test PASSED!")
    else:
        logger.error("✗ Arena stability test FAILED!")
    
    exit(0 if success else 1)