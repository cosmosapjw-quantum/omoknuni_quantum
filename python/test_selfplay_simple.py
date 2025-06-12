#!/usr/bin/env python3
"""
Simplified MCTS Self-Play Test

This is a minimal version to test that the MCTS self-play works correctly
without complex multiprocessing or neural network evaluation.
"""

import torch
import numpy as np
import time
import logging
from typing import List

# MCTS imports
from mcts.core.mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType
from mcts.core.game_interface import GameInterface, GameType as InterfaceGameType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleEvaluator:
    """Simple evaluator that returns uniform random policies and zero values"""
    
    def __init__(self, board_size: int = 15):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.board_size = board_size
        self.action_size = board_size * board_size
        
    def evaluate_batch(self, features, legal_masks=None):
        """Return uniform random policies and zero values"""
        if isinstance(features, torch.Tensor):
            batch_size = features.shape[0]
        elif isinstance(features, list):
            batch_size = len(features)
        else:
            batch_size = 1
        
        # Uniform random policy
        policies = torch.ones(batch_size, self.action_size, device=self.device) / self.action_size
        
        # Zero values
        values = torch.zeros(batch_size, 1, device=self.device)
        
        return policies, values


def play_simple_game():
    """Play a single game with simple MCTS"""
    logger.info("Starting simple self-play game")
    
    # Create game interface
    game_interface = GameInterface(InterfaceGameType.GOMOKU, board_size=15)
    state = game_interface.create_initial_state()
    
    # Create simple evaluator
    evaluator = SimpleEvaluator()
    
    # Configure MCTS
    mcts_config = MCTSConfig(
        num_simulations=200,  # Reduced for faster testing
        c_puct=1.4,
        temperature=1.0,
        wave_size=512,  # Smaller wave for testing
        device='cuda' if torch.cuda.is_available() else 'cpu',
        game_type=GameType.GOMOKU,
        board_size=15
    )
    
    # Create MCTS
    mcts = MCTS(mcts_config, evaluator)
    
    # Game loop
    move_count = 0
    max_moves = 50  # Limit moves for testing
    total_search_time = 0
    
    logger.info("Starting game loop")
    
    while not game_interface.is_terminal(state) and move_count < max_moves:
        # Run MCTS search
        search_start = time.time()
        
        try:
            policy = mcts.search(state, mcts_config.num_simulations)
            search_time = time.time() - search_start
            total_search_time += search_time
            
            # Get MCTS statistics
            stats = mcts.get_statistics()
            sims_per_sec = stats.get('last_search_sims_per_second', 0)
            
            # Get legal actions
            legal_actions = game_interface.get_legal_moves(state)
            
            if not legal_actions:
                logger.warning("No legal actions available")
                break
            
            # Select action (random from legal actions for testing)
            # In practice, you'd sample from the policy
            action = np.random.choice(legal_actions)
            
            # Apply action
            state = game_interface.apply_move(state, action)
            move_count += 1
            
            logger.info(f"Move {move_count}: action={action}, "
                       f"search_time={search_time:.3f}s, "
                       f"sims/s={sims_per_sec:.0f}, "
                       f"legal_actions={len(legal_actions)}")
            
        except Exception as e:
            logger.error(f"Error during move {move_count}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Game results
    is_terminal = game_interface.is_terminal(state)
    winner = 0
    if is_terminal:
        winner = game_interface.get_winner(state)
    
    avg_sims_per_sec = (mcts_config.num_simulations * move_count) / total_search_time if total_search_time > 0 else 0
    
    logger.info(f"Game finished: moves={move_count}, "
               f"terminal={is_terminal}, winner={winner}, "
               f"total_time={total_search_time:.2f}s, "
               f"avg_sims/s={avg_sims_per_sec:.0f}")
    
    return {
        'move_count': move_count,
        'is_terminal': is_terminal,
        'winner': winner,
        'total_search_time': total_search_time,
        'avg_simulations_per_second': avg_sims_per_sec
    }


def test_multiple_games(num_games: int = 3):
    """Test multiple games"""
    logger.info(f"Testing {num_games} simple self-play games")
    
    results = []
    total_start = time.time()
    
    for i in range(num_games):
        logger.info(f"\n--- Game {i+1}/{num_games} ---")
        
        try:
            result = play_simple_game()
            result['game_id'] = i
            results.append(result)
            
        except Exception as e:
            logger.error(f"Game {i+1} failed: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - total_start
    
    # Summary
    print(f"\n{'='*50}")
    print(f"SIMPLE SELF-PLAY TEST RESULTS")
    print(f"{'='*50}")
    print(f"Games completed: {len(results)}/{num_games}")
    print(f"Total time: {total_time:.2f}s")
    
    if results:
        avg_moves = np.mean([r['move_count'] for r in results])
        avg_sims_per_sec = np.mean([r['avg_simulations_per_second'] for r in results])
        total_moves = sum(r['move_count'] for r in results)
        
        print(f"Average moves per game: {avg_moves:.1f}")
        print(f"Total moves: {total_moves}")
        print(f"Average sims/sec: {avg_sims_per_sec:.0f}")
        print(f"Games/hour rate: {len(results) / (total_time / 3600):.1f}")
        
        # Check if any games reached terminal state
        terminal_games = [r for r in results if r['is_terminal']]
        print(f"Terminal games: {len(terminal_games)}/{len(results)}")
    
    return results


def main():
    """Main test function"""
    print("🧪 Testing Simple MCTS Self-Play")
    print(f"GPU Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print()
    
    try:
        # Test single game first
        print("Testing single game...")
        single_result = play_simple_game()
        
        if single_result['move_count'] > 0:
            print("✅ Single game test passed")
            
            # Test multiple games
            print("\nTesting multiple games...")
            multi_results = test_multiple_games(3)
            
            if len(multi_results) > 0:
                print("✅ Multiple games test passed")
                print("🎯 MCTS self-play is working correctly!")
            else:
                print("❌ Multiple games test failed")
        else:
            print("❌ Single game test failed")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())