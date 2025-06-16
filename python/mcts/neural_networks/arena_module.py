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
import gc
import psutil
import os
import traceback

import torch
import torch.multiprocessing as mp

# Set start method for CUDA compatibility
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

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
    enable_tree_reuse: bool = False  # Disable by default to avoid memory issues
    gc_frequency: int = 10  # Run garbage collection every N games


class ELOTracker:
    """Tracks ELO ratings for models"""
    
    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1500.0):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        # Use a regular dict instead of defaultdict with lambda
        self.ratings: Dict[str, float] = {}
        self.game_history: List[Dict] = []
    
    def update_ratings(self, player1: str, player2: str,
                      wins: int, draws: int, losses: int):
        """Update ELO ratings based on match results"""
        total_games = wins + draws + losses
        if total_games == 0:
            return
        
        # Get current ratings (use initial_rating if not found)
        r1 = self.ratings.get(player1, self.initial_rating)
        r2 = self.ratings.get(player2, self.initial_rating)
        
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
        return self.ratings.get(player, self.initial_rating)
    
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
        
        self.ratings = data["ratings"]
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
        from mcts.core.evaluator import Evaluator, RandomEvaluator
        is_evaluator1 = isinstance(model1, Evaluator)
        is_evaluator2 = isinstance(model2, Evaluator)
        
        # Use parallel arena only when one of the models is RandomEvaluator
        # This avoids CUDA multiprocessing issues when comparing two neural networks
        is_random1 = isinstance(model1, RandomEvaluator)
        is_random2 = isinstance(model2, RandomEvaluator)
        
        if (is_random1 or is_random2) and self.arena_config.num_workers > 1:
            # Use parallel arena for NN vs Random matches
            return self._parallel_arena(model1, model2, num_games, silent)
        else:
            # Use sequential arena for NN vs NN matches to avoid CUDA issues
            return self._sequential_arena(model1, model2, num_games, silent)
    
    def _serialize_config(self, config):
        """Serialize config for multiprocessing, removing CUDA references"""
        # Create a dictionary representation to avoid any CUDA references
        return {
            'game': {
                'game_type': config.game.game_type,
                'board_size': config.game.board_size
            },
            'network': {
                'input_channels': config.network.input_channels,
                'num_res_blocks': config.network.num_res_blocks,
                'num_filters': config.network.num_filters,
                'value_head_hidden_size': config.network.value_head_hidden_size,
                'policy_head_filters': config.network.policy_head_filters
            },
            'training': {
                'max_moves_per_game': config.training.max_moves_per_game
            },
            'mcts': {
                'min_wave_size': config.mcts.min_wave_size,
                'max_wave_size': config.mcts.max_wave_size,
                'adaptive_wave_sizing': config.mcts.adaptive_wave_sizing,
                'use_mixed_precision': config.mcts.use_mixed_precision,
                'use_cuda_graphs': config.mcts.use_cuda_graphs,
                'use_tensor_cores': config.mcts.use_tensor_cores,
                'memory_pool_size_mb': config.mcts.memory_pool_size_mb,
                'max_tree_nodes': config.mcts.max_tree_nodes
            },
            'arena': {
                'elo_k_factor': config.arena.elo_k_factor
            },
            'log_level': config.log_level
        }
    
    def _serialize_arena_config(self, arena_config: ArenaConfig):
        """Serialize arena config for multiprocessing"""
        return {
            'num_games': arena_config.num_games,
            'win_threshold': arena_config.win_threshold,
            'num_workers': arena_config.num_workers,
            'mcts_simulations': arena_config.mcts_simulations,
            'c_puct': arena_config.c_puct,
            'temperature': arena_config.temperature,
            'temperature_threshold': arena_config.temperature_threshold,
            'timeout_seconds': arena_config.timeout_seconds,
            'use_progress_bar': arena_config.use_progress_bar,
            'save_game_records': arena_config.save_game_records
        }
    
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
    
    def _parallel_arena(self, model1, model2,
                       num_games: int, silent: bool = False) -> Tuple[int, int, int]:
        """Run arena games in parallel"""
        from mcts.utils.safe_multiprocessing import serialize_state_dict_for_multiprocessing
        from mcts.core.evaluator import Evaluator, RandomEvaluator
        
        wins, draws, losses = 0, 0, 0
        
        # Handle evaluators vs models
        if isinstance(model1, RandomEvaluator):
            model1_state = None
            is_random1 = True
        elif isinstance(model1, Evaluator):
            # Extract model from evaluator
            model1_state = serialize_state_dict_for_multiprocessing(model1.model.state_dict())
            is_random1 = False
        else:
            # Direct model
            model1_state = serialize_state_dict_for_multiprocessing(model1.state_dict())
            is_random1 = False
            
        if isinstance(model2, RandomEvaluator):
            model2_state = None
            is_random2 = True
        elif isinstance(model2, Evaluator):
            # Extract model from evaluator
            model2_state = serialize_state_dict_for_multiprocessing(model2.model.state_dict())
            is_random2 = False
        else:
            # Direct model
            model2_state = serialize_state_dict_for_multiprocessing(model2.state_dict())
            is_random2 = False
        
        # Use spawn context for CUDA compatibility
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=self.arena_config.num_workers, mp_context=ctx) as executor:
            # Submit games
            futures = []
            for game_idx in range(num_games):
                # Alternate who plays first
                if game_idx % 2 == 0:
                    future = executor.submit(
                        _play_arena_game_worker,
                        self._serialize_config(self.config),
                        self._serialize_arena_config(self.arena_config),
                        model1_state,
                        model2_state,
                        game_idx,
                        False,  # model1 plays first
                        is_random1,
                        is_random2
                    )
                else:
                    future = executor.submit(
                        _play_arena_game_worker,
                        self._serialize_config(self.config),
                        self._serialize_arena_config(self.arena_config),
                        model2_state,
                        model1_state,
                        game_idx,
                        True,  # model2 plays first, invert result
                        is_random2,
                        is_random1
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
        # Debug: Log memory usage at game start
        if game_idx % 10 == 0:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            logger.debug(f"[ARENA] Game {game_idx} - Memory: RSS={mem_info.rss/1024/1024:.1f}MB, VMS={mem_info.vms/1024/1024:.1f}MB")
            
            # Check GPU memory if CUDA available
            if torch.cuda.is_available():
                logger.debug(f"[ARENA] Game {game_idx} - GPU Memory: Allocated={torch.cuda.memory_allocated()/1024/1024:.1f}MB, Reserved={torch.cuda.memory_reserved()/1024/1024:.1f}MB")
        
        try:
            # Create MCTS instances
            mcts1 = self._create_mcts(evaluator1)
            mcts2 = self._create_mcts(evaluator2)
            
            if game_idx % 10 == 0:
                logger.debug(f"[ARENA] Game {game_idx} - MCTS instances created successfully")
        
            # Play game
            state = self.game_interface.create_initial_state()
            current_player = 1
            
            for move_num in range(self.config.training.max_moves_per_game):
                # Debug logging for specific games where crash occurs
                if game_idx >= 100 and move_num % 10 == 0:
                    logger.debug(f"[ARENA] Game {game_idx}, Move {move_num}, Player {current_player}")
                    
                # Get current MCTS
                current_mcts = mcts1 if current_player == 1 else mcts2
                
                # Get action
                if move_num < self.arena_config.temperature_threshold:
                    temp = self.arena_config.temperature
                else:
                    temp = 0.0
                
                # Run MCTS search
                policy = current_mcts.search(state, num_simulations=self.arena_config.mcts_simulations)
                
                if temp == 0:
                    # Deterministic - choose best action
                    action = current_mcts.get_best_action(state)
                else:
                    # Sample from policy
                    valid_actions, valid_probs = current_mcts.get_valid_actions_and_probabilities(state, temperature=temp)
                    if not valid_actions:
                        raise ValueError(f"No valid actions available at move {move_num}")
                    action = np.random.choice(valid_actions, p=valid_probs)
                
                # Apply action
                state = self.game_interface.get_next_state(state, action)
                
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
            
        except Exception as e:
            logger.error(f"[ARENA] Game {game_idx} failed with error: {e}")
            logger.error(f"[ARENA] Full traceback:\n{traceback.format_exc()}")
            raise
        finally:
            # Explicit cleanup to prevent memory leaks
            if game_idx % 10 == 0:
                logger.debug(f"[ARENA] Game {game_idx} - Cleaning up MCTS instances")
            
            # Delete MCTS instances explicitly
            del mcts1
            del mcts2
            
            # Force garbage collection periodically
            if game_idx % self.arena_config.gc_frequency == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug(f"[ARENA] Game {game_idx} - Forced garbage collection and CUDA cache clear")
    
    def _create_mcts(self, evaluator: Any):
        """Create MCTS instance for arena"""
        from mcts.core.mcts import MCTS, MCTSConfig
        
        # Use same optimal settings as self-play for consistency
        mcts_config = MCTSConfig(
            # Core MCTS parameters
            num_simulations=self.arena_config.mcts_simulations,
            c_puct=self.arena_config.c_puct,
            temperature=0.0,  # Arena always uses deterministic play
            
            # Performance and wave parameters - use full config values for optimal performance
            min_wave_size=self.config.mcts.min_wave_size,
            max_wave_size=self.config.mcts.max_wave_size,
            adaptive_wave_sizing=self.config.mcts.adaptive_wave_sizing,
            
            # Memory and optimization - slightly reduced for arena but still substantial
            memory_pool_size_mb=max(512, self.config.mcts.memory_pool_size_mb // 4),  # Same as self-play workers
            max_tree_nodes=self.config.mcts.max_tree_nodes // 2,  # Same as self-play workers
            use_mixed_precision=self.config.mcts.use_mixed_precision,
            use_cuda_graphs=self.config.mcts.use_cuda_graphs,  # Enable for performance
            use_tensor_cores=self.config.mcts.use_tensor_cores,
            
            # Game and device configuration
            device=self.arena_config.device,
            game_type=self.game_type,
            board_size=self.config.game.board_size,
            
            # Disable virtual loss for arena (not needed for sequential play)
            enable_virtual_loss=False,
            virtual_loss=0.0,
            
            # Debug options
            enable_debug_logging=(self.config.log_level == "DEBUG"),
            profile_gpu_kernels=False
        )
        
        # Optimize for hardware
        mcts = MCTS(mcts_config, evaluator)
        mcts.optimize_for_hardware()
        
        return mcts
    
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


def _play_arena_game_worker(config_dict: Dict, arena_config_dict: Dict,
                           model1_state: Optional[Dict], model2_state: Optional[Dict],
                           game_idx: int, invert_result: bool,
                           is_random1: bool, is_random2: bool) -> int:
    """Worker function for parallel arena games"""
    import os
    import sys
    
    # Set up proper multiprocessing for CUDA
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    import torch
    from mcts.core.game_interface import GameInterface, GameType
    from mcts.core.mcts import MCTS, MCTSConfig
    from mcts.utils.safe_multiprocessing import deserialize_state_dict_from_multiprocessing
    
    # Set up logging
    logging.basicConfig(level=getattr(logging, config_dict['log_level']))
    
    # Determine device - use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.debug(f"[ARENA WORKER] PID: {os.getpid()}, Game: {game_idx}, Device: {device}")
    
    # Create game interface
    game_type = GameType[config_dict['game']['game_type'].upper()]
    game_interface = GameInterface(game_type, config_dict['game']['board_size'])
    
    # Get action space size
    initial_state = game_interface.create_initial_state()
    action_size = game_interface.get_action_space_size(initial_state)
    
    # Create evaluators
    from mcts.neural_networks.nn_model import create_model
    from mcts.core.evaluator import AlphaZeroEvaluator, RandomEvaluator, EvaluatorConfig
    
    # Create evaluator 1
    if is_random1:
        evaluator1 = RandomEvaluator(
            config=EvaluatorConfig(device=device),
            action_size=action_size
        )
    else:
        model1 = create_model(
            game_type=config_dict['game']['game_type'],
            input_height=config_dict['game']['board_size'],
            input_width=config_dict['game']['board_size'],
            num_actions=action_size,
            input_channels=config_dict['network']['input_channels'],
            num_res_blocks=config_dict['network']['num_res_blocks'],
            num_filters=config_dict['network']['num_filters'],
            value_head_hidden_size=config_dict['network']['value_head_hidden_size'],
            policy_head_filters=config_dict['network']['policy_head_filters']
        )
        # Deserialize and load state dict
        model1_state_deserialized = deserialize_state_dict_from_multiprocessing(model1_state)
        model1.load_state_dict(model1_state_deserialized)
        model1.to(device)
        model1.eval()
        
        evaluator1 = AlphaZeroEvaluator(
            model=model1,
            device=device
        )
    
    # Create evaluator 2
    if is_random2:
        evaluator2 = RandomEvaluator(
            config=EvaluatorConfig(device=device),
            action_size=action_size
        )
    else:
        model2 = create_model(
            game_type=config_dict['game']['game_type'],
            input_height=config_dict['game']['board_size'],
            input_width=config_dict['game']['board_size'],
            num_actions=action_size,
            input_channels=config_dict['network']['input_channels'],
            num_res_blocks=config_dict['network']['num_res_blocks'],
            num_filters=config_dict['network']['num_filters'],
            value_head_hidden_size=config_dict['network']['value_head_hidden_size'],
            policy_head_filters=config_dict['network']['policy_head_filters']
        )
        # Deserialize and load state dict
        model2_state_deserialized = deserialize_state_dict_from_multiprocessing(model2_state)
        model2.load_state_dict(model2_state_deserialized)
        model2.to(device)
        model2.eval()
        
        evaluator2 = AlphaZeroEvaluator(
            model=model2,
            device=device
        )
    
    # Create MCTS instances with optimal settings
    mcts_config = MCTSConfig(
        # Core MCTS parameters
        num_simulations=arena_config_dict['mcts_simulations'],
        c_puct=arena_config_dict['c_puct'],
        temperature=0.0,  # Arena always uses deterministic play
        
        # Performance and wave parameters - use full config values
        min_wave_size=config_dict['mcts']['min_wave_size'],
        max_wave_size=config_dict['mcts']['max_wave_size'],
        adaptive_wave_sizing=config_dict['mcts']['adaptive_wave_sizing'],
        
        # Memory and optimization - same as main process
        memory_pool_size_mb=max(512, config_dict['mcts']['memory_pool_size_mb'] // 4),
        max_tree_nodes=config_dict['mcts']['max_tree_nodes'] // 2,
        use_mixed_precision=config_dict['mcts']['use_mixed_precision'],
        use_cuda_graphs=config_dict['mcts']['use_cuda_graphs'],
        use_tensor_cores=config_dict['mcts']['use_tensor_cores'],
        
        # Game and device configuration
        device=device,  # Use the detected device
        game_type=game_type,
        board_size=config_dict['game']['board_size'],
        
        # Disable virtual loss for arena
        enable_virtual_loss=False,
        virtual_loss=0.0
    )
    
    mcts1 = MCTS(mcts_config, evaluator1)
    mcts1.optimize_for_hardware()
    
    mcts2 = MCTS(mcts_config, evaluator2)
    mcts2.optimize_for_hardware()
    
    # Play game
    state = game_interface.create_initial_state()
    current_player = 1
    
    for move_num in range(config_dict['training']['max_moves_per_game']):
        # Get current MCTS
        current_mcts = mcts1 if current_player == 1 else mcts2
        
        # Run MCTS search first
        policy = current_mcts.search(state, num_simulations=arena_config_dict['mcts_simulations'])
        
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