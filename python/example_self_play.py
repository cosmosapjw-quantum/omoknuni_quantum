#!/usr/bin/env python3
"""
Self-play example demonstrating the optimized MCTS in a real game environment.

This example shows:
1. Setting up MCTS with a real neural network evaluator
2. Playing complete games using MCTS for both players
3. Collecting training data from self-play games
4. Performance monitoring and statistics
"""

import torch
import numpy as np
import time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import logging

# MCTS imports
from mcts.core.mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType
from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
import alphazero_py

# Set up logging
logging.basicConfig(
    level=logging.INFO,
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


class SelfPlayWorker:
    """Worker for generating self-play games"""
    
    def __init__(
        self,
        mcts_config: MCTSConfig,
        evaluator,
        game_type: GameType = GameType.GOMOKU,
        board_size: int = 15,
        temperature_threshold: int = 30
    ):
        """
        Initialize self-play worker
        
        Args:
            mcts_config: Configuration for MCTS
            evaluator: Neural network evaluator
            game_type: Type of game to play
            board_size: Size of the game board
            temperature_threshold: Move number after which to play deterministically
        """
        self.mcts_config = mcts_config
        self.evaluator = evaluator
        self.game_type = game_type
        self.board_size = board_size
        self.temperature_threshold = temperature_threshold
        
        # Create MCTS instance
        self.mcts = MCTS(mcts_config, evaluator)
        self.mcts.optimize_for_hardware()
        
        # Statistics
        self.games_played = 0
        self.total_moves = 0
        self.total_time = 0.0
        
    def play_game(self, verbose: bool = False) -> GameRecord:
        """
        Play a single self-play game
        
        Args:
            verbose: Whether to print game progress
            
        Returns:
            GameRecord containing game data
        """
        # Initialize game state
        if self.game_type == GameType.GOMOKU:
            state = alphazero_py.GomokuState()
        elif self.game_type == GameType.CHESS:
            state = alphazero_py.ChessState()
        elif self.game_type == GameType.GO:
            state = alphazero_py.GoState(self.board_size)
        else:
            raise ValueError(f"Unsupported game type: {self.game_type}")
        
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
            print(f"{'='*50}")
        
        # Play until game ends
        while not state.is_terminal():
            move_count += 1
            
            # Record state
            state_tensor = self._get_state_tensor(state)
            states.append(state_tensor)
            
            # Determine temperature
            if move_count <= self.temperature_threshold:
                temperature = 1.0
            else:
                temperature = 0.0  # Play deterministically
            
            # Run MCTS search
            if verbose:
                print(f"\nMove {move_count} - Player {state.current_player() + 1}")
            
            search_start = time.perf_counter()
            policy = self.mcts.search(state, self.mcts_config.num_simulations)
            search_time = time.perf_counter() - search_start
            
            total_simulations += self.mcts_config.num_simulations
            
            # Record policy
            policies.append(policy.copy())
            
            # Select action
            if temperature > 0:
                # Sample from policy distribution
                action = np.random.choice(len(policy), p=policy)
            else:
                # Select best action
                action = np.argmax(policy)
            
            actions.append(action)
            
            # Print move info
            if verbose:
                row = action // self.board_size
                col = action % self.board_size
                print(f"  Position: ({row}, {col})")
                print(f"  Search time: {search_time:.2f}s")
                print(f"  Simulations/second: {self.mcts_config.num_simulations/search_time:,.0f}")
                print(f"  Top 5 moves:")
                top_actions = np.argsort(policy)[-5:][::-1]
                for i, a in enumerate(top_actions):
                    r, c = a // self.board_size, a % self.board_size
                    print(f"    {i+1}. ({r}, {c}): {policy[a]:.3f}")
            
            # Make move
            state = state.clone()
            state.make_move(action)
            
            # Reset MCTS tree for next move
            self.mcts.reset_tree()
            
            # Display board periodically
            if verbose and move_count % 5 == 0:
                self._display_board(state)
        
        # Game ended
        game_time = time.perf_counter() - game_start
        
        # Determine winner
        if hasattr(state, 'get_winner'):
            winner = state.get_winner()
        else:
            # For games without explicit winner method
            winner = state.get_result(0)  # From player 0's perspective
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Game ended after {move_count} moves")
            if winner == 1:
                print("Winner: Player 1 (Black)")
            elif winner == -1:
                print("Winner: Player 2 (White)")
            else:
                print("Result: Draw")
            print(f"Total time: {game_time:.1f}s")
            print(f"Average time per move: {game_time/move_count:.1f}s")
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
    
    def play_games(self, num_games: int, verbose_interval: int = 10) -> List[GameRecord]:
        """
        Play multiple self-play games
        
        Args:
            num_games: Number of games to play
            verbose_interval: Print detailed info every N games
            
        Returns:
            List of game records
        """
        games = []
        
        print(f"Starting {num_games} self-play games...")
        print(f"MCTS config: {self.mcts_config.num_simulations} simulations per move")
        print(f"Wave size: {self.mcts_config.min_wave_size}")
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
    
    def _get_state_tensor(self, state) -> np.ndarray:
        """Convert game state to tensor representation"""
        # This would normally use your game-specific feature extraction
        # For now, we'll use a simple board representation
        if hasattr(state, 'to_numpy'):
            return state.to_numpy()
        else:
            # Fallback for states without to_numpy
            board = np.zeros((self.board_size, self.board_size), dtype=np.float32)
            # You would fill this based on the state
            return board
    
    def _display_board(self, state):
        """Display the current board state"""
        if self.game_type == GameType.GOMOKU:
            # Simple text representation
            print("\n  ", end="")
            for i in range(self.board_size):
                print(f"{i:2}", end=" ")
            print()
            
            for row in range(self.board_size):
                print(f"{row:2}", end=" ")
                for col in range(self.board_size):
                    idx = row * self.board_size + col
                    # This is a placeholder - actual implementation would check state
                    print(" .", end=" ")
                print()
    
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
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get worker statistics"""
        return {
            'games_played': self.games_played,
            'total_moves': self.total_moves,
            'total_time': self.total_time,
            'avg_moves_per_game': self.total_moves / max(1, self.games_played),
            'avg_time_per_game': self.total_time / max(1, self.games_played),
            'avg_time_per_move': self.total_time / max(1, self.total_moves),
            'total_simulations': self.total_moves * self.mcts_config.num_simulations,
            'avg_sims_per_second': (self.total_moves * self.mcts_config.num_simulations) / max(0.001, self.total_time)
        }


def main():
    """Main self-play demonstration"""
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, running on CPU (will be slow)")
        device = 'cpu'
    else:
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    
    # Create MCTS configuration optimized for performance
    mcts_config = MCTSConfig(
        num_simulations=5000,  # Moderate for demonstration
        min_wave_size=3072,
        max_wave_size=3072,
        adaptive_wave_sizing=False,  # Critical for performance
        device=device,
        game_type=GameType.GOMOKU,
        board_size=15,
        c_puct=1.414,
        temperature=1.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        memory_pool_size_mb=2048,
        max_tree_nodes=500000,
        use_mixed_precision=True,
        use_cuda_graphs=True,
        use_tensor_cores=True,
        enable_virtual_loss=True,
        virtual_loss=3.0,  # Positive value (will be negated when applied)
        enable_debug_logging=False
    )
    
    # Create neural network evaluator
    print("Initializing neural network evaluator...")
    evaluator = ResNetEvaluator(
        game_type='gomoku',
        board_size=15,
        device=device,
        use_mixed_precision=True
    )
    
    # Create self-play worker
    worker = SelfPlayWorker(
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
    
    # Play a few games with detailed output
    games = worker.play_games(num_games=5, verbose_interval=1)
    
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
    
    # Final statistics
    stats = worker.get_statistics()
    print(f"\nOverall self-play statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("Self-play demonstration complete!")
    print("="*70)
    
    # Save one game for analysis (optional)
    if games:
        print("\nSaving first game data...")
        game_data = {
            'states': [s.tolist() for s in games[0].states],
            'policies': [p.tolist() for p in games[0].policies],
            'actions': games[0].actions,
            'winner': games[0].winner,
            'game_length': games[0].game_length
        }
        
        import json
        with open('example_game.json', 'w') as f:
            json.dump(game_data, f)
        print("Game data saved to example_game.json")


if __name__ == "__main__":
    main()