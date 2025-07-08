"""Self-play manager for AlphaZero training

This module handles self-play data generation including:
- Parallel game execution
- MCTS integration
- Training example collection
- Game metrics tracking
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Type
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

from mcts.neural_networks.replay_buffer import GameExample
from mcts.core.mcts import MCTS
from mcts.core.mcts_config import MCTSConfig
from mcts.core.game_interface import GameInterface
from mcts.utils.config_system import AlphaZeroConfig


logger = logging.getLogger(__name__)


@dataclass
class SelfPlayConfig:
    """Configuration for self-play"""
    num_games_per_iteration: int = 100
    num_workers: int = 4
    mcts_simulations: int = 800
    temperature: float = 1.0
    temperature_threshold: int = 30
    resign_threshold: float = -0.98
    enable_resign: bool = True
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    c_puct: float = 1.5
    enable_progress_bar: bool = True
    batch_size: int = 8  # For MCTS batching


class SelfPlayGame:
    """Manages a single self-play game"""
    
    def __init__(self, game, evaluator, config: SelfPlayConfig, game_id: str):
        """Initialize self-play game
        
        Args:
            game: Game instance
            evaluator: Neural network evaluator
            config: Self-play configuration
            game_id: Unique game identifier
        """
        self.game = game
        self.evaluator = evaluator
        self.config = config
        self.game_id = game_id
        self.examples = []
        self.current_player = 1
        self.move_count = 0
        
        # Initialize MCTS with proper config
        mcts_config = MCTSConfig(
            num_simulations=config.mcts_simulations,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
            tree_batch_size=config.batch_size,
            device='cpu'  # Use CPU for self-play by default
        )
        
        # Create game interface
        self.game_interface = GameInterface(game)
        
        # Initialize MCTS
        self.mcts = MCTS(
            config=mcts_config,
            evaluator=evaluator,
            game_interface=self.game_interface
        )
    
    def play_single_move(self):
        """Play a single move in the game"""
        # Get current state
        state = self.game.get_state()
        self.current_player = self.game.get_current_player()
        
        # Run MCTS search to get policy
        action_probs = self.mcts.search(state)
        
        # Store training example
        self.examples.append({
            'state': state.copy(),
            'policy': action_probs.copy(),
            'player': self.current_player,
            'move_number': self.move_count
        })
        
        # Select action based on temperature
        temperature = self._get_temperature()
        if temperature > 0:
            # Sample from distribution
            action = np.random.choice(len(action_probs), p=action_probs)
        else:
            # Select best action
            action = np.argmax(action_probs)
        
        self.game.make_action(action)
        self.move_count += 1
        
        # Check for resignation
        if self.config.enable_resign and self.move_count > 10:
            # Get value from MCTS tree
            if hasattr(self.mcts, 'tree') and hasattr(self.mcts.tree, 'get_node_value'):
                value = self.mcts.tree.get_node_value(0)  # Root node value
                if value < self.config.resign_threshold:
                    return True  # Resign
        
        return False
    
    def play_game(self) -> List[GameExample]:
        """Play a complete game and return training examples"""
        resigned = False
        
        while not self.game.is_terminal():
            resigned = self.play_single_move()
            if resigned:
                break
        
        # Get final outcome
        if resigned:
            # Current player resigned, so they lost
            winner = -self.current_player
        else:
            winner = self.game.get_winner()
        
        # Convert examples to GameExample objects with correct values
        game_examples = []
        for example in self.examples:
            # Value is from the perspective of the player who made the move
            value = winner * example['player']
            
            game_examples.append(GameExample(
                state=example['state'],
                policy=example['policy'],
                value=float(value),
                game_id=self.game_id,
                move_number=example['move_number']
            ))
        
        return game_examples
    
    def _get_temperature(self) -> float:
        """Get temperature for current move"""
        if self.move_count < self.config.temperature_threshold:
            return self.config.temperature
        return 0.0


class SelfPlayManager:
    """Manages self-play data generation"""
    
    def __init__(self, config: AlphaZeroConfig, game_class: Type, evaluator):
        """Initialize self-play manager
        
        Args:
            config: AlphaZero configuration
            game_class: Game class to instantiate
            evaluator: Neural network evaluator
        """
        self.config = config
        self.game_class = game_class
        self.evaluator = evaluator
        
        # Create self-play config from AlphaZero config
        self.self_play_config = SelfPlayConfig(
            num_games_per_iteration=config.training.self_play_games_per_iteration,
            num_workers=config.training.num_workers,
            mcts_simulations=config.mcts.num_simulations,
            temperature=config.mcts.temperature,
            temperature_threshold=config.mcts.temperature_threshold,
            resign_threshold=getattr(config.mcts, 'resign_threshold', -0.98),
            enable_resign=getattr(config.mcts, 'enable_resign', True),
            dirichlet_alpha=config.mcts.dirichlet_alpha,
            dirichlet_epsilon=config.mcts.dirichlet_epsilon,
            c_puct=config.mcts.c_puct,
            batch_size=getattr(config.mcts, 'tree_batch_size', 8)
        )
    
    def generate_self_play_data(self) -> List[GameExample]:
        """Generate self-play training data
        
        Returns:
            List of training examples
        """
        logger.info(f"Generating {self.self_play_config.num_games_per_iteration} self-play games")
        
        if self.self_play_config.num_workers <= 1:
            # Single process execution
            return self._generate_single_process()
        else:
            # Multi-process execution
            return self._generate_multi_process()
    
    def _generate_single_process(self) -> List[GameExample]:
        """Generate self-play data in single process"""
        all_examples = []
        
        # Progress bar
        if self.self_play_config.enable_progress_bar:
            pbar = tqdm(
                total=self.self_play_config.num_games_per_iteration,
                desc="Self-play games"
            )
        
        for game_idx in range(self.self_play_config.num_games_per_iteration):
            # Create new game instance
            game = self.game_class()
            
            # Play game
            sp_game = SelfPlayGame(
                game=game,
                evaluator=self.evaluator,
                config=self.self_play_config,
                game_id=f"game_{game_idx}"
            )
            
            examples = sp_game.play_game()
            all_examples.extend(examples)
            
            if self.self_play_config.enable_progress_bar:
                pbar.update(1)
        
        if self.self_play_config.enable_progress_bar:
            pbar.close()
        
        return all_examples
    
    def _generate_multi_process(self) -> List[GameExample]:
        """Generate self-play data using multiple processes"""
        all_examples = []
        
        # Calculate games per worker
        games_per_worker = self.self_play_config.num_games_per_iteration // self.self_play_config.num_workers
        remainder = self.self_play_config.num_games_per_iteration % self.self_play_config.num_workers
        
        # Create work distribution
        work_distribution = []
        for i in range(self.self_play_config.num_workers):
            num_games = games_per_worker + (1 if i < remainder else 0)
            if num_games > 0:
                work_distribution.append((i, num_games))
        
        # Execute in parallel
        with ProcessPoolExecutor(max_workers=self.self_play_config.num_workers) as executor:
            futures = []
            
            for worker_id, num_games in work_distribution:
                future = executor.submit(
                    self._play_games_worker,
                    worker_id,
                    num_games
                )
                futures.append(future)
            
            # Collect results with progress bar
            if self.self_play_config.enable_progress_bar:
                pbar = tqdm(
                    total=self.self_play_config.num_games_per_iteration,
                    desc="Self-play games"
                )
            
            for future in as_completed(futures):
                examples = future.result()
                all_examples.extend(examples)
                
                if self.self_play_config.enable_progress_bar:
                    # Update based on number of games completed
                    num_games = len(set(ex.game_id for ex in examples))
                    pbar.update(num_games)
            
            if self.self_play_config.enable_progress_bar:
                pbar.close()
        
        return all_examples
    
    def _play_games_worker(self, worker_id: int, num_games: int) -> List[GameExample]:
        """Worker function for parallel self-play
        
        Args:
            worker_id: Worker identifier
            num_games: Number of games to play
            
        Returns:
            List of training examples
        """
        examples = []
        
        for game_idx in range(num_games):
            # Create new game instance
            game = self.game_class()
            
            # Play game
            sp_game = SelfPlayGame(
                game=game,
                evaluator=self.evaluator,
                config=self.self_play_config,
                game_id=f"worker_{worker_id}_game_{game_idx}"
            )
            
            game_examples = sp_game.play_game()
            examples.extend(game_examples)
        
        return examples
    
    def collect_game_metrics(self, examples: List[GameExample]) -> Dict[str, Any]:
        """Collect metrics from self-play games
        
        Args:
            examples: List of training examples
            
        Returns:
            Dictionary of metrics
        """
        if not examples:
            return {
                'total_games': 0,
                'total_moves': 0,
                'avg_game_length': 0,
                'win_rates': {'player1': 0, 'player2': 0, 'draw': 0}
            }
        
        # Group by game
        games = {}
        for ex in examples:
            if ex.game_id not in games:
                games[ex.game_id] = []
            games[ex.game_id].append(ex)
        
        # Calculate metrics
        total_games = len(games)
        total_moves = len(examples)
        avg_game_length = total_moves / total_games if total_games > 0 else 0
        
        # Calculate win rates
        player1_wins = 0
        player2_wins = 0
        draws = 0
        
        for game_examples in games.values():
            # Check the final value (from player 1's perspective)
            final_value = game_examples[0].value  # All examples in a game have consistent values
            if final_value > 0:
                player1_wins += 1
            elif final_value < 0:
                player2_wins += 1
            else:
                draws += 1
        
        win_rates = {
            'player1': player1_wins / total_games if total_games > 0 else 0,
            'player2': player2_wins / total_games if total_games > 0 else 0,
            'draw': draws / total_games if total_games > 0 else 0
        }
        
        return {
            'total_games': total_games,
            'total_moves': total_moves,
            'avg_game_length': avg_game_length,
            'win_rates': win_rates,
            'resignation_rate': sum(1 for g in games.values() if len(g) < 50) / total_games if total_games > 0 else 0
        }