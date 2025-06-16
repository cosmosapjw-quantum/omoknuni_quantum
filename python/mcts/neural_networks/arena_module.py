"""Arena module for model evaluation and comparison

This module handles model battles, ELO tracking, and tournament organization.
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm
import numpy as np

import torch

logger = logging.getLogger(__name__)


@dataclass
class ArenaConfig:
    """Configuration for arena battles"""
    num_games: int = 40
    win_threshold: float = 0.55
    num_workers: int = 4
    mcts_simulations: int = 400
    c_puct: float = 1.0
    temperature: float = 0.0
    temperature_threshold: int = 0
    timeout_seconds: int = 300
    device: str = "cuda"
    use_progress_bar: bool = True
    save_game_records: bool = False


class ELOTracker:
    """Tracks ELO ratings for models"""
    
    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1500.0):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings: Dict[str, float] = defaultdict(lambda: initial_rating)
        self.game_history: List[Dict] = []
    
    def update_ratings(self, player1: str, player2: str,
                      wins: int, draws: int, losses: int):
        """Update ELO ratings based on match results"""
        total_games = wins + draws + losses
        if total_games == 0:
            return
        
        # Get current ratings
        r1 = self.ratings[player1]
        r2 = self.ratings[player2]
        
        # Calculate expected scores
        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))
        
        # Actual scores
        s1 = (wins + 0.5 * draws) / total_games
        s2 = (losses + 0.5 * draws) / total_games
        
        # Update ratings (but keep "random" fixed at 0 as anchor)
        # Standard ELO formula: new_rating = old_rating + K * (actual_score - expected_score)
        # Note: We do NOT multiply by total_games - K factor already accounts for match weight
        if player1 != "random":
            self.ratings[player1] = r1 + self.k_factor * (s1 - e1)
        if player2 != "random":
            self.ratings[player2] = r2 + self.k_factor * (s2 - e2)
        
        # Record history
        self.game_history.append({
            "timestamp": datetime.now().isoformat(),
            "player1": player1,
            "player2": player2,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "old_rating1": r1,
            "old_rating2": r2,
            "new_rating1": self.ratings[player1],
            "new_rating2": self.ratings[player2]
        })
    
    def get_rating(self, player: str) -> float:
        """Get current rating for a player"""
        return self.ratings[player]
    
    def get_leaderboard(self) -> List[Tuple[str, float]]:
        """Get sorted leaderboard"""
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
    
    def save_to_file(self, filepath: str):
        """Save ratings and history to JSON file"""
        data = {
            "ratings": dict(self.ratings),
            "history": self.game_history,
            "k_factor": self.k_factor,
            "initial_rating": self.initial_rating
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """Load ratings and history from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.ratings = defaultdict(lambda: self.initial_rating, data["ratings"])
        self.game_history = data["history"]
        self.k_factor = data.get("k_factor", self.k_factor)
        self.initial_rating = data.get("initial_rating", self.initial_rating)


class ArenaManager:
    """Manages model evaluation battles"""
    
    def __init__(self, config, arena_config: Optional[ArenaConfig] = None):
        from mcts.core.game_interface import GameInterface, GameType
        
        self.config = config
        self.arena_config = arena_config or ArenaConfig()
        self.game_type = GameType[config.game.game_type.upper()]
        self.game_interface = GameInterface(
            self.game_type,
            board_size=config.game.board_size
        )
        self.elo_tracker = ELOTracker(k_factor=config.arena.elo_k_factor)
    
    def compare_models(self, model1, model2,
                      model1_name: str = "model1", model2_name: str = "model2",
                      num_games: Optional[int] = None, silent: bool = False) -> Tuple[int, int, int]:
        """Compare two models/evaluators in arena battles
        
        Args:
            model1: First model (torch.nn.Module) or evaluator
            model2: Second model (torch.nn.Module) or evaluator  
            model1_name: Name for logging
            model2_name: Name for logging
            num_games: Number of games to play
            silent: If True, suppress progress bars and logging
            
        Returns:
            Tuple of (wins, draws, losses) from model1's perspective
        """
        num_games = num_games or self.arena_config.num_games
        
        if not silent:
            logger.info(f"Arena: {model1_name} vs {model2_name} ({num_games} games)")
        
        # Check if we're dealing with evaluators or models
        from mcts.core.evaluator import Evaluator
        is_evaluator1 = isinstance(model1, Evaluator)
        is_evaluator2 = isinstance(model2, Evaluator)
        
        if self.arena_config.num_workers <= 1 or is_evaluator1 or is_evaluator2:
            # Use sequential for evaluators (can't pickle them easily)
            return self._sequential_arena(model1, model2, num_games, silent)
        else:
            return self._parallel_arena(model1, model2, num_games)
    
    def _sequential_arena(self, model1, model2,
                         num_games: int, silent: bool = False) -> Tuple[int, int, int]:
        """Run arena games sequentially"""
        from mcts.core.evaluator import AlphaZeroEvaluator, Evaluator
        
        wins, draws, losses = 0, 0, 0
        
        # Create evaluators if needed
        if isinstance(model1, Evaluator):
            evaluator1 = model1
        else:
            evaluator1 = AlphaZeroEvaluator(
                model=model1,
                device=self.arena_config.device
            )
            
        if isinstance(model2, Evaluator):
            evaluator2 = model2
        else:
            evaluator2 = AlphaZeroEvaluator(
                model=model2,
                device=self.arena_config.device
            )
        
        # Progress bar
        disable_progress = silent or logger.level > logging.INFO or not self.arena_config.use_progress_bar
        
        # Arena progress at position 4 to avoid overlap with other progress bars
        with tqdm(total=num_games, desc="Arena games", unit="game",
                 disable=disable_progress, position=4, leave=False) as pbar:
            for game_idx in range(num_games):
                # Alternate who plays first
                if game_idx % 2 == 0:
                    result = self._play_single_game(evaluator1, evaluator2, game_idx)
                else:
                    result = -self._play_single_game(evaluator2, evaluator1, game_idx)
                
                if result > 0:
                    wins += 1
                elif result < 0:
                    losses += 1
                else:
                    draws += 1
                
                pbar.update(1)
                pbar.set_postfix({
                    'W': wins,
                    'D': draws,
                    'L': losses,
                    'WR': f'{wins/(wins+draws+losses):.1%}'
                })
        
        return wins, draws, losses
    
    def _parallel_arena(self, model1: torch.nn.Module, model2: torch.nn.Module,
                       num_games: int) -> Tuple[int, int, int]:
        """Run arena games in parallel"""
        wins, draws, losses = 0, 0, 0
        
        model1_state = model1.state_dict()
        model2_state = model2.state_dict()
        
        with ProcessPoolExecutor(max_workers=self.arena_config.num_workers) as executor:
            # Submit games
            futures = []
            for game_idx in range(num_games):
                # Alternate who plays first
                if game_idx % 2 == 0:
                    future = executor.submit(
                        _play_arena_game_worker,
                        self.config,
                        self.arena_config,
                        model1_state,
                        model2_state,
                        game_idx,
                        False  # model1 plays first
                    )
                else:
                    future = executor.submit(
                        _play_arena_game_worker,
                        self.config,
                        self.arena_config,
                        model2_state,
                        model1_state,
                        game_idx,
                        True  # model2 plays first, invert result
                    )
                futures.append((future, game_idx % 2 == 1))
            
            # Collect results
            disable_progress = logger.level > logging.INFO or not self.arena_config.use_progress_bar
            
            # Arena progress at position 4 to avoid overlap with other progress bars
            with tqdm(total=len(futures), desc="Arena games", unit="game",
                     disable=disable_progress, position=4, leave=False) as pbar:
                for future, should_invert in futures:
                    try:
                        result = future.result(timeout=self.arena_config.timeout_seconds)
                        if should_invert:
                            result = -result
                        
                        if result > 0:
                            wins += 1
                        elif result < 0:
                            losses += 1
                        else:
                            draws += 1
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            'W': wins,
                            'D': draws,
                            'L': losses,
                            'WR': f'{wins/(wins+draws+losses):.1%}'
                        })
                    except Exception as e:
                        logger.error(f"Arena game failed: {e}")
                        draws += 1  # Count errors as draws
                        pbar.update(1)
        
        return wins, draws, losses
    
    def _play_single_game(self, evaluator1: Any, evaluator2: Any,
                         game_idx: int) -> int:
        """Play a single arena game
        
        Returns:
            1 if player 1 wins, -1 if player 2 wins, 0 for draw
        """
        # Create MCTS instances
        mcts1 = self._create_mcts(evaluator1)
        mcts2 = self._create_mcts(evaluator2)
        
        # Play game
        state = self.game_interface.create_initial_state()
        current_player = 1
        
        for move_num in range(self.config.training.max_moves_per_game):
            # Get current MCTS
            current_mcts = mcts1 if current_player == 1 else mcts2
            
            # Get action
            if move_num < self.arena_config.temperature_threshold:
                temp = self.arena_config.temperature
            else:
                temp = 0.0
            
            policy = current_mcts.get_action_probabilities(state, temperature=temp)
            
            if temp == 0:
                # Deterministic - choose best legal move
                valid_actions, valid_probs = current_mcts.get_valid_actions_and_probabilities(state, temperature=0.0)
                if not valid_actions:
                    raise ValueError(f"No valid actions available at move {move_num}")
                action = valid_actions[np.argmax(valid_probs)]
            else:
                # Sample from valid moves only
                valid_actions, valid_probs = current_mcts.get_valid_actions_and_probabilities(state, temperature=temp)
                if not valid_actions:
                    raise ValueError(f"No valid actions available at move {move_num}")
                action = np.random.choice(valid_actions, p=valid_probs)
            
            # Apply action
            state = self.game_interface.get_next_state(state, action)
            
            # Update MCTS trees for tree reuse
            if current_player == 1:
                mcts1.update_root(action)
                # Other player needs to know about opponent's move too
                mcts2.update_root(action)
            else:
                mcts2.update_root(action)
                mcts1.update_root(action)
            
            # Check terminal
            if self.game_interface.is_terminal(state):
                # Return from player 1's perspective
                value = self.game_interface.get_value(state)
                if current_player == 2:
                    value = -value
                return int(np.sign(value))
            
            # Switch player
            current_player = 3 - current_player
        
        # Draw if max moves reached
        return 0
    
    def _create_mcts(self, evaluator: Any):
        """Create MCTS instance for arena"""
        from mcts.core.mcts import MCTS, MCTSConfig
        
        # Use wave size settings from main config to ensure consistency
        mcts_config = MCTSConfig(
            num_simulations=self.arena_config.mcts_simulations,
            c_puct=self.arena_config.c_puct,
            temperature=0.0,
            device=self.arena_config.device,
            game_type=self.game_type,
            # Wave size settings from main config
            min_wave_size=self.config.mcts.min_wave_size,
            max_wave_size=self.config.mcts.max_wave_size,
            adaptive_wave_sizing=self.config.mcts.adaptive_wave_sizing,
            # Other performance settings
            use_mixed_precision=self.config.mcts.use_mixed_precision,
            use_cuda_graphs=self.config.mcts.use_cuda_graphs,
            use_tensor_cores=self.config.mcts.use_tensor_cores
        )
        
        return MCTS(mcts_config, evaluator)
    
    def run_tournament(self, models: Dict[str, torch.nn.Module],
                      games_per_match: Optional[int] = None) -> Dict[str, Any]:
        """Run round-robin tournament between multiple models
        
        Args:
            models: Dictionary mapping model names to models
            games_per_match: Games per match (overrides config)
            
        Returns:
            Tournament results dictionary
        """
        games_per_match = games_per_match or self.arena_config.num_games
        
        logger.info(f"Starting tournament with {len(models)} models")
        
        # Generate all pairs
        from itertools import combinations
        model_pairs = list(combinations(models.keys(), 2))
        
        # Results storage
        results = {}
        standings = defaultdict(lambda: {"wins": 0, "draws": 0, "losses": 0, "games": 0})
        
        # Run matches
        with tqdm(total=len(model_pairs), desc="Tournament matches") as pbar:
            for model1_name, model2_name in model_pairs:
                # Run match
                wins, draws, losses = self.compare_models(
                    models[model1_name], models[model2_name],
                    model1_name=model1_name,
                    model2_name=model2_name,
                    num_games=games_per_match
                )
                
                # Store results
                results[(model1_name, model2_name)] = (wins, draws, losses)
                
                # Update standings
                standings[model1_name]["wins"] += wins
                standings[model1_name]["draws"] += draws
                standings[model1_name]["losses"] += losses
                standings[model1_name]["games"] += wins + draws + losses
                
                standings[model2_name]["wins"] += losses
                standings[model2_name]["draws"] += draws
                standings[model2_name]["losses"] += wins
                standings[model2_name]["games"] += wins + draws + losses
                
                # Update ELO
                self.elo_tracker.update_ratings(
                    model1_name, model2_name,
                    wins, draws, losses
                )
                
                pbar.update(1)
        
        # Calculate final standings
        for model_name, stats in standings.items():
            total_games = stats["games"]
            if total_games > 0:
                stats["win_rate"] = stats["wins"] / total_games
                stats["points"] = stats["wins"] + 0.5 * stats["draws"]
            else:
                stats["win_rate"] = 0
                stats["points"] = 0
            
            stats["elo"] = self.elo_tracker.get_rating(model_name)
        
        # Sort by points, then by ELO
        sorted_standings = sorted(
            standings.items(),
            key=lambda x: (x[1]["points"], x[1]["elo"]),
            reverse=True
        )
        
        # Create results dictionary
        tournament_results = {
            "timestamp": datetime.now().isoformat(),
            "models": list(models.keys()),
            "matches": [
                {
                    "model1": m1,
                    "model2": m2,
                    "wins": w,
                    "draws": d,
                    "losses": l
                }
                for (m1, m2), (w, d, l) in results.items()
            ],
            "standings": [
                {
                    "rank": rank,
                    "model": model_name,
                    **stats
                }
                for rank, (model_name, stats) in enumerate(sorted_standings, 1)
            ],
            "elo_ratings": dict(self.elo_tracker.ratings)
        }
        
        # Print results
        self._print_tournament_results(tournament_results)
        
        return tournament_results
    
    def _print_tournament_results(self, results: Dict[str, Any]):
        """Print tournament results in a nice format"""
        logger.info("\n" + "=" * 70)
        logger.info("TOURNAMENT RESULTS")
        logger.info("=" * 70)
        logger.info(f"{'Rank':<6} {'Model':<20} {'W-D-L':<15} {'Points':<8} {'Win%':<8} {'ELO':<8}")
        logger.info("-" * 70)
        
        for standing in results["standings"]:
            w, d, l = standing["wins"], standing["draws"], standing["losses"]
            record = f"{w}-{d}-{l}"
            logger.info(
                f"{standing['rank']:<6} {standing['model']:<20} "
                f"{record:<15} {standing['points']:<8.1f} "
                f"{standing['win_rate']:<8.1%} {standing['elo']:<8.1f}"
            )
        
        logger.info("=" * 70)


def _play_arena_game_worker(config, arena_config: ArenaConfig,
                           model1_state: Dict, model2_state: Dict,
                           game_idx: int, invert_result: bool) -> int:
    """Worker function for parallel arena games"""
    from mcts.core.game_interface import GameInterface, GameType
    from mcts.core.mcts import MCTS, MCTSConfig
    
    # Set up logging
    logging.basicConfig(level=getattr(logging, config.log_level))
    
    # Create game interface
    game_type = GameType[config.game.game_type.upper()]
    game_interface = GameInterface(game_type, config.game.board_size)
    
    # Get action space size
    initial_state = game_interface.create_initial_state()
    action_size = game_interface.get_action_space_size(initial_state)
    
    # Create models
    from mcts.neural_networks.nn_model import create_model
    
    model1 = create_model(
        game_type=config.game.game_type,
        input_height=config.game.board_size,
        input_width=config.game.board_size,
        num_actions=action_size,
        input_channels=config.network.input_channels,
        num_res_blocks=config.network.num_res_blocks,
        num_filters=config.network.num_filters
    )
    model1.load_state_dict(model1_state)
    model1.eval()
    
    model2 = create_model(
        game_type=config.game.game_type,
        input_height=config.game.board_size,
        input_width=config.game.board_size,
        num_actions=action_size,
        input_channels=config.network.input_channels,
        num_res_blocks=config.network.num_res_blocks,
        num_filters=config.network.num_filters
    )
    model2.load_state_dict(model2_state)
    model2.eval()
    
    # Create evaluators
    from mcts.core.evaluator import AlphaZeroEvaluator
    evaluator1 = AlphaZeroEvaluator(
        model=model1,
        device=arena_config.device
    )
    evaluator2 = AlphaZeroEvaluator(
        model=model2,
        device=arena_config.device
    )
    
    # Create MCTS instances
    mcts_config = MCTSConfig(
        num_simulations=arena_config.mcts_simulations,
        c_puct=arena_config.c_puct,
        temperature=0.0,
        device=arena_config.device,
        game_type=game_type,
        board_size=config.game.board_size,
        # Add minimal required parameters
        min_wave_size=32,
        max_wave_size=32,
        adaptive_wave_sizing=False,
        memory_pool_size_mb=128,
        max_tree_nodes=10000,
        use_mixed_precision=False,
        use_cuda_graphs=False,
        use_tensor_cores=False
    )
    
    mcts1 = MCTS(mcts_config, evaluator1)
    mcts2 = MCTS(mcts_config, evaluator2)
    
    # Play game
    state = game_interface.create_initial_state()
    current_player = 1
    
    for move_num in range(config.training.max_moves_per_game):
        # Get current MCTS
        current_mcts = mcts1 if current_player == 1 else mcts2
        
        # Run MCTS search first
        policy = current_mcts.search(state, num_simulations=arena_config.mcts_simulations)
        
        # Get best action (deterministic)
        action = current_mcts.get_best_action(state)
        
        # Apply action
        state = game_interface.get_next_state(state, action)
        
        # Update MCTS trees for tree reuse
        if current_player == 1:
            mcts1.update_root(action)
            mcts2.update_root(action)
        else:
            mcts2.update_root(action)
            mcts1.update_root(action)
        
        # Check terminal
        if game_interface.is_terminal(state):
            # Return from player 1's perspective
            value = game_interface.get_value(state)
            if current_player == 2:
                value = -value
            return int(np.sign(value))
        
        # Switch player
        current_player = 3 - current_player
    
    # Draw if max moves reached
    return 0