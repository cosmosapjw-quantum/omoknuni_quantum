#!/usr/bin/env python3
"""
Simple test to debug MCTS hanging issue
"""

import os
import sys
import time
import torch
import logging

# Disable CUDA compilation temporarily
os.environ['DISABLE_CUDA_COMPILE'] = '1'

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_basic_mcts():
    """Test basic MCTS functionality"""
    logger.info("Starting basic MCTS test...")
    
    # Import after setting environment
    import alphazero_py
    from mcts.core.high_performance_mcts import HighPerformanceMCTS, HighPerformanceMCTSConfig
    from mcts.core.game_interface import GameInterface, GameType
    from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
    
    # Create game
    logger.info("Creating game...")
    game = alphazero_py.GomokuState()
    game_interface = GameInterface(GameType.GOMOKU)
    
    # Create evaluator
    logger.info("Creating evaluator...")
    evaluator = ResNetEvaluator(
        game_type='gomoku',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create MCTS with small parameters
    logger.info("Creating MCTS...")
    config = HighPerformanceMCTSConfig(
        num_simulations=10,  # Very small
        wave_size=8,  # Small wave size
        c_puct=1.0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        enable_interference=False,  # Disable quantum features
        enable_phase_policy=False,
        enable_path_integral=False
    )
    
    mcts = HighPerformanceMCTS(config, game_interface, evaluator)
    
    # Run search
    logger.info("Running search...")
    start = time.time()
    policy = mcts.search(game)
    end = time.time()
    
    logger.info(f"Search completed in {end-start:.3f}s")
    logger.info(f"Policy: {policy}")
    
    # Get stats
    stats = mcts.get_search_statistics()
    logger.info(f"Stats: {stats}")
    
    return True

def test_wave_engine():
    """Test wave engine directly"""
    logger.info("Testing wave engine...")
    
    import alphazero_py
    from mcts.gpu.csr_tree import CSRTree, CSRTreeConfig
    from mcts.gpu.optimized_wave_engine import OptimizedWaveEngine, OptimizedWaveConfig
    from mcts.core.game_interface import GameInterface, GameType
    from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
    
    # Create components
    game = alphazero_py.GomokuState()
    game_interface = GameInterface(GameType.GOMOKU)
    evaluator = ResNetEvaluator(game_type='gomoku', device='cuda')
    
    # Create tree
    tree_config = CSRTreeConfig(
        max_nodes=1000,
        max_edges=5000,
        device='cuda'
    )
    tree = CSRTree(tree_config)
    tree.add_root(state=game)
    
    # Create wave engine
    wave_config = OptimizedWaveConfig(
        wave_size=8,
        c_puct=1.0,
        device='cuda',
        enable_interference=False,
        enable_phase_policy=False,
        enable_path_integral=False
    )
    
    wave_engine = OptimizedWaveEngine(tree, wave_config, game_interface, evaluator)
    
    # Run single wave
    logger.info("Running single wave...")
    start = time.time()
    result = wave_engine.run_wave(game, wave_size=8)
    end = time.time()
    
    logger.info(f"Wave completed in {end-start:.3f}s")
    logger.info(f"Result: {result}")
    
    return True

def main():
    """Run tests"""
    try:
        logger.info("=" * 60)
        logger.info("MCTS HANG DEBUG TEST")
        logger.info("=" * 60)
        
        # Test 1: Basic MCTS
        if test_basic_mcts():
            logger.info("✓ Basic MCTS test passed")
        else:
            logger.error("✗ Basic MCTS test failed")
            
        # Test 2: Wave engine
        if test_wave_engine():
            logger.info("✓ Wave engine test passed")
        else:
            logger.error("✗ Wave engine test failed")
            
        logger.info("All tests completed!")
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()