#!/usr/bin/env python3
"""
Simple Self-Play Test - Test core MCTS without multiprocessing complexity

This test focuses on validating the core MCTS performance improvements
without the multiprocessing overhead.
"""

import sys
import time
import torch
from pathlib import Path

# Add MCTS modules to path
mcts_root = Path(__file__).parent
sys.path.insert(0, str(mcts_root))

from mcts.utils.config_system import AlphaZeroConfig
from mcts.neural_networks.self_play_module import SelfPlayManager
from mcts.neural_networks.mock_evaluator import MockEvaluator
from mcts.utils.training_profiler import enable_profiling, log_profiling_summary, reset_profiler


def test_single_game_performance():
    """Test single game performance without multiprocessing"""
    print("=" * 60)
    print("SIMPLE SELF-PLAY TEST (No Multiprocessing)")
    print("=" * 60)
    
    # Create simple configuration
    config = AlphaZeroConfig()
    config.game.game_type = "gomoku"
    config.game.board_size = 15
    
    # MCTS settings - reduced for faster testing
    config.mcts.num_simulations = 200  # Reduced for quick test
    config.mcts.min_wave_size = 512    # Smaller for testing
    config.mcts.max_wave_size = 512
    config.mcts.adaptive_wave_sizing = False
    config.mcts.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.mcts.quantum_level = 'classical'
    config.mcts.enable_quantum = False
    config.mcts.memory_pool_size_mb = 1024
    config.mcts.max_tree_nodes = 50000
    
    # Training settings
    config.training.num_games_per_iteration = 1  # Single game
    config.training.num_workers = 1  # No multiprocessing
    config.training.max_moves_per_game = 50  # Shorter for testing
    config.training.resign_threshold = -0.8
    config.training.resign_check_moves = 5
    config.training.resign_start_iteration = 1
    
    # Network settings - minimal
    config.network.input_channels = 18
    config.network.input_representation = 'basic'
    
    # Logging
    config.log_level = 'INFO'
    
    print(f"Device: {config.mcts.device}")
    print(f"Simulations: {config.mcts.num_simulations}")
    print(f"Max moves: {config.training.max_moves_per_game}")
    print()
    
    # Create self-play manager
    self_play_manager = SelfPlayManager(config)
    
    # Enable profiling
    enable_profiling()
    reset_profiler()
    
    print("🚀 Starting single game test...")
    start_time = time.perf_counter()
    
    try:
        # Create MockEvaluator directly
        evaluator = MockEvaluator(game_type='gomoku', device=config.mcts.device)
        
        # Generate single game using direct method
        examples = self_play_manager._play_single_game(
            model=None,  # Not used with MockEvaluator
            evaluator=evaluator,
            game_idx=0,
            iteration=1
        )
        
        total_time = time.perf_counter() - start_time
        
        # Results
        print("✅ Single game completed successfully!")
        print()
        print("PERFORMANCE RESULTS:")
        print("-" * 30)
        print(f"Game time:         {total_time:.2f}s")
        print(f"Examples generated: {len(examples) if examples else 0}")
        
        if examples:
            avg_moves = len(examples)
            print(f"Moves played:      {avg_moves}")
            print(f"Time per move:     {total_time / max(1, avg_moves):.3f}s")
            
            # Estimate simulations per second
            total_sims = avg_moves * config.mcts.num_simulations
            sims_per_sec = total_sims / total_time if total_time > 0 else 0
            print(f"MCTS sims/s:       {sims_per_sec:.0f}")
        
        # Performance analysis
        print()
        print("ANALYSIS:")
        print("-" * 30)
        
        if total_time > 30:
            print("❌ Game too slow (>30s)")
        elif total_time > 10:
            print("⚠️  Game slower than ideal (<10s)")
        else:
            print("✅ Game speed good (<10s)")
        
        if not examples:
            print("❌ No examples generated")
            return False
        elif len(examples) < 10:
            print("⚠️  Very short game")
        else:
            print("✅ Reasonable game length")
        
        # Success criteria
        success = (
            total_time < 60 and  # Reasonable time
            len(examples) > 0    # Examples generated
        )
        
        print()
        if success:
            print("🎉 SIMPLE TEST PASSED!")
        else:
            print("❌ SIMPLE TEST FAILED!")
        
        # Show profiling
        print()
        log_profiling_summary(top_n=10, min_time=0.01)
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    try:
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Run the test
        success = test_single_game_performance()
        
        print("\n" + "=" * 60)
        if success:
            print("✅ SIMPLE TEST PASSED - Core MCTS is working!")
        else:
            print("❌ SIMPLE TEST FAILED - Core MCTS needs fixing")
        print("=" * 60)
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())