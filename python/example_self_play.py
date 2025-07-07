#!/usr/bin/env python3
"""
Self-play example with correct API usage and board visualization.

This example demonstrates high-performance MCTS with:
- Correct board visualization using proper tensor channels
- API calls matching the current implementation  
- Proper handling of legal moves and action selection
- Optional GPU service for batch processing
"""

import torch
import numpy as np
import time
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import logging
import sys
import os

# MCTS imports
from mcts.core.mcts import MCTS
from mcts.core.mcts_config import MCTSConfig
from mcts.core.game_interface import GameInterface, GameType
from mcts.core.evaluator import AlphaZeroEvaluator
from mcts.neural_networks.resnet_model import create_resnet_for_game
from mcts.utils.optimized_remote_evaluator import OptimizedRemoteEvaluator
from mcts.utils.gpu_evaluator_service import GPUEvaluatorService
import multiprocessing as mp
import queue
import threading

# Set up logging
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class GameRecord:
    """Record of a single self-play game"""
    states: List[np.ndarray]
    policies: List[np.ndarray]
    actions: List[int]
    winner: int  # 1 for player 1, -1 for player 2, 0 for draw
    game_length: int
    total_simulations: int
    avg_sims_per_move: float
    total_time: float


class OptimizedSelfPlayWorker:
    """Self-play worker with correct API usage"""
    
    def __init__(
        self,
        mcts_config: MCTSConfig,
        evaluator,
        game_type: GameType = GameType.GOMOKU,
        board_size: int = 15,
        temperature_threshold: int = 30
    ):
        self.mcts_config = mcts_config
        self.evaluator = evaluator
        self.game_type = game_type
        self.board_size = board_size
        self.temperature_threshold = temperature_threshold
        
        # Create game interface with basic representation (18 channels for ResNet)
        self.game_interface = GameInterface(
            game_type, 
            board_size=board_size, 
            input_representation='basic'
        )
        
        # Create MCTS instance
        self.mcts = MCTS(mcts_config, evaluator, self.game_interface)
        
        # Statistics
        self.games_played = 0
        self.total_moves = 0
        self.total_time = 0.0
        
    def play_game(self, verbose: bool = False) -> GameRecord:
        """Play a single self-play game with correct API usage"""
        
        # Initialize game state
        state = self.game_interface.create_initial_state()
        
        # Game data
        states = []
        policies = []
        actions = []
        
        # Timing
        game_start = time.perf_counter()
        total_simulations = 0
        move_count = 0
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Starting new {self.game_type.name} game")
            print(f"Board size: {self.board_size}x{self.board_size}")
            print(f"{'='*50}")
        
        # Play until game ends
        while not self.game_interface.is_terminal(state):
            move_count += 1
            
            # Record state
            state_tensor = self.game_interface.state_to_numpy(state)
            states.append(state_tensor.copy())
            
            # Determine temperature for move selection
            if move_count <= self.temperature_threshold:
                temperature = 1.0
            else:
                temperature = 0.0  # Play deterministically after threshold
            
            # Display current position
            if verbose:
                current_player = self.game_interface.get_current_player(state)
                print(f"\nMove {move_count} - Player {current_player}")
                self._display_board(state)
            
            search_start = time.perf_counter()
            
            # Run MCTS search - reset tree first to ensure clean state
            self.mcts.reset_tree()
            policy = self.mcts.search(state, num_simulations=self.mcts_config.num_simulations)
            
            # Get legal moves for validation
            legal_moves = self.game_interface.get_legal_moves(state)
            
            # Ensure policy only has probability on legal moves
            masked_policy = np.zeros(len(policy))
            for move in legal_moves:
                masked_policy[move] = policy[move]
            
            # Renormalize
            if masked_policy.sum() > 0:
                masked_policy = masked_policy / masked_policy.sum()
            else:
                # Fallback: uniform over legal moves
                for move in legal_moves:
                    masked_policy[move] = 1.0 / len(legal_moves)
            
            # Select action from masked policy
            if temperature > 0 and len(legal_moves) > 1:
                # Sample from policy distribution
                action = np.random.choice(len(masked_policy), p=masked_policy)
            else:
                # Select best action among legal moves
                legal_values = [(move, masked_policy[move]) for move in legal_moves]
                action = max(legal_values, key=lambda x: x[1])[0]
            
            # Validate selected action
            if action not in legal_moves:
                logger.error(f"Selected illegal action {action}! Legal moves: {legal_moves[:10]}...")
                # Force legal move
                action = legal_moves[0]
            
            search_time = time.perf_counter() - search_start
            total_simulations += self.mcts_config.num_simulations
            
            # Record policy and action
            policies.append(masked_policy.copy())
            actions.append(action)
            
            # Print move info
            if verbose:
                row = action // self.board_size
                col = action % self.board_size
                sims_per_sec = self.mcts_config.num_simulations / search_time if search_time > 0 else 0
                print(f"  Selected move: ({row}, {col}) [action {action}]")
                print(f"  Policy value: {masked_policy[action]:.3f}")
                print(f"  Search time: {search_time:.3f}s")
                print(f"  Simulations/second: {sims_per_sec:,.0f}")
                
                # Show top 5 moves
                top_indices = np.argsort(masked_policy)[-5:][::-1]
                print(f"  Top 5 moves:")
                for i, idx in enumerate(top_indices):
                    if masked_policy[idx] > 0:
                        r, c = idx // self.board_size, idx % self.board_size
                        print(f"    {i+1}. ({r}, {c}): {masked_policy[idx]:.3f}")
            
            # Make move using correct API
            state = self.game_interface.apply_move(state, action)
            
            # Display board periodically
            if verbose and move_count % 5 == 0:
                self._display_board(state)
        
        # Game ended
        game_time = time.perf_counter() - game_start
        
        # Determine winner
        winner = self.game_interface.get_winner(state)
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Game ended after {move_count} moves")
            if winner == 1:
                print("Winner: Player 1 (Black/X)")
            elif winner == -1:
                print("Winner: Player 2 (White/O)")
            else:
                print("Result: Draw")
            print(f"Total time: {game_time:.1f}s")
            print(f"Average time per move: {game_time/move_count:.2f}s")
            print(f"Total simulations: {total_simulations:,}")
            print(f"Average simulations/second: {total_simulations/game_time:,.0f}")
            print(f"{'='*50}")
            
            # Display final board
            self._display_board(state)
        
        # Update statistics
        self.games_played += 1
        self.total_moves += move_count
        self.total_time += game_time
        
        return GameRecord(
            states=states,
            policies=policies,
            actions=actions,
            winner=winner,
            game_length=move_count,
            total_simulations=total_simulations,
            avg_sims_per_move=total_simulations / move_count,
            total_time=game_time
        )
    
    def _display_board(self, state):
        """Display the current board state correctly"""
        if self.game_type == GameType.GOMOKU:
            # Get the basic tensor representation
            board_tensor = self.game_interface.state_to_numpy(state, representation_type='basic')
            
            # Basic representation has 18 channels:
            # Channel 0: Current player's stones
            # Channel 1: Opponent's stones
            # Channels 2-9: Move history
            # etc.
            
            print("\n  ", end="")
            for i in range(self.board_size):
                print(f"{i:2}", end=" ")
            print()
            
            for row in range(self.board_size):
                print(f"{row:2}", end=" ")
                for col in range(self.board_size):
                    # Check current and opponent channels
                    current_stone = board_tensor[0, row, col]
                    opponent_stone = board_tensor[1, row, col]
                    
                    if current_stone > 0:
                        # Determine which player based on current player
                        current_player = self.game_interface.get_current_player(state)
                        if current_player == 1:  # Current is P1, so this is P1's stone
                            print(" X", end=" ")
                        else:  # Current is P2, so this is P2's stone
                            print(" O", end=" ")
                    elif opponent_stone > 0:
                        # Opponent's stone
                        current_player = self.game_interface.get_current_player(state)
                        if current_player == 1:  # Current is P1, so opponent is P2
                            print(" O", end=" ")
                        else:  # Current is P2, so opponent is P1
                            print(" X", end=" ")
                    else:
                        print(" .", end=" ")
                print()
            
        elif self.game_type == GameType.GO:
            # Similar logic for Go
            board_tensor = self.game_interface.state_to_numpy(state, representation_type='basic')
            
            print("\n  ", end="")
            for i in range(self.board_size):
                print(f"{i:2}", end=" ")
            print()
            
            for row in range(self.board_size):
                print(f"{row:2}", end=" ")
                for col in range(self.board_size):
                    current_stone = board_tensor[0, row, col]
                    opponent_stone = board_tensor[1, row, col]
                    
                    if current_stone > 0:
                        print(" ●", end=" ")  # Black stone
                    elif opponent_stone > 0:
                        print(" ○", end=" ")  # White stone
                    else:
                        print(" ·", end=" ")  # Empty
                print()
    
    def play_games(self, num_games: int, verbose_interval: int = 10) -> List[GameRecord]:
        """Play multiple self-play games"""
        games = []
        
        print(f"Starting {num_games} self-play games...")
        print(f"MCTS config: {self.mcts_config.num_simulations} simulations per move")
        print(f"Board size: {self.board_size}x{self.board_size}")
        print()
        
        for i in range(num_games):
            verbose = (i % verbose_interval == 0)
            
            if not verbose and i % max(1, num_games // 10) == 0:
                print(f"Progress: {i}/{num_games} games ({i/num_games*100:.1f}%)")
                self._print_statistics()
            
            game = self.play_game(verbose=verbose)
            games.append(game)
        
        print(f"\nCompleted {num_games} games!")
        self._print_statistics()
        
        return games
    
    def _print_statistics(self):
        """Print current statistics"""
        if self.games_played > 0:
            avg_moves = self.total_moves / self.games_played
            avg_time = self.total_time / self.games_played
            total_sims = self.total_moves * self.mcts_config.num_simulations
            avg_sims_per_sec = total_sims / self.total_time if self.total_time > 0 else 0
            
            print(f"  Games played: {self.games_played}")
            print(f"  Average moves per game: {avg_moves:.1f}")
            print(f"  Average time per game: {avg_time:.1f}s")
            print(f"  Average simulations/second: {avg_sims_per_sec:,.0f}")
            print()


def main():
    """Main self-play demonstration with correct API usage"""
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, running on CPU (will be slower)")
        device = 'cpu'
    else:
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()
    
    # Create MCTS configuration
    mcts_config = MCTSConfig()
    mcts_config.num_simulations = 800  # Standard setting
    mcts_config.c_puct = 1.4
    mcts_config.device = device
    mcts_config.batch_size = 64
    mcts_config.enable_virtual_loss = True
    mcts_config.enable_fast_ucb = True
    mcts_config.enable_subtree_reuse = True  # Enable for better performance
    
    # Create neural network model
    print("Creating neural network model...")
    model = create_resnet_for_game(
        game_type='gomoku',
        input_channels=18,  # Basic representation
        num_blocks=10,
        num_filters=128
    )
    model = model.to(device)
    model.eval()
    
    # Create evaluator
    print("Creating evaluator...")
    evaluator = AlphaZeroEvaluator(model, device=device)
    
    # For high-performance batch processing, you can optionally use GPU service:
    # 1. Start GPU service in a separate thread/process
    # 2. Use OptimizedRemoteEvaluator instead of AlphaZeroEvaluator
    # See the documentation for batch processing setup
    
    # Create self-play worker
    worker = OptimizedSelfPlayWorker(
        mcts_config=mcts_config,
        evaluator=evaluator,
        game_type=GameType.GOMOKU,
        board_size=15,
        temperature_threshold=30
    )
    
    # Play demonstration games
    print("\n" + "="*70)
    print("SELF-PLAY DEMONSTRATION")
    print("="*70)
    
    # Play games with detailed output  
    games = worker.play_games(num_games=3, verbose_interval=1)
    
    # Analyze results
    print("\n" + "="*70)
    print("GAME ANALYSIS")
    print("="*70)
    
    # Win statistics
    player1_wins = sum(1 for g in games if g.winner == 1)
    player2_wins = sum(1 for g in games if g.winner == -1)
    draws = sum(1 for g in games if g.winner == 0)
    
    print(f"Results from {len(games)} games:")
    print(f"  Player 1 wins: {player1_wins} ({player1_wins/len(games)*100:.1f}%)")
    print(f"  Player 2 wins: {player2_wins} ({player2_wins/len(games)*100:.1f}%)")
    print(f"  Draws: {draws} ({draws/len(games)*100:.1f}%)")
    
    # Game length statistics
    game_lengths = [g.game_length for g in games]
    print(f"\nGame length statistics:")
    print(f"  Average: {np.mean(game_lengths):.1f} moves")
    print(f"  Min: {min(game_lengths)} moves")
    print(f"  Max: {max(game_lengths)} moves")
    
    # Performance statistics
    total_sims = sum(g.total_simulations for g in games)
    total_time = sum(g.total_time for g in games)
    
    print(f"\nPerformance statistics:")
    print(f"  Total simulations: {total_sims:,}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Average simulations/second: {total_sims/total_time:,.0f}")
    
    print("\n" + "="*70)
    print("Self-play demonstration complete!")
    print("="*70)


if __name__ == "__main__":
    main()