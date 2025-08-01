"""Arena manager for model evaluation and comparison

This module handles:
- Model vs model battles
- ELO rating tracking
- Win rate calculation
- Tournament organization
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Type, Any
from collections import defaultdict
from pathlib import Path
import numpy as np
from tqdm import tqdm

from mcts.core.mcts import MCTS
from mcts.core.mcts_config import MCTSConfig
from mcts.core.game_interface import GameInterface
from mcts.utils.config_system import AlphaZeroConfig
from mcts.utils.single_gpu_evaluator import SingleGPUEvaluator


logger = logging.getLogger(__name__)


@dataclass
class ArenaConfig:
    """Configuration for arena battles"""
    num_games: int = 40
    win_threshold: float = 0.55
    mcts_simulations: int = 200  # Reduced for faster arena matches
    c_puct: float = 1.0
    temperature: float = 0.0
    temperature_threshold: int = 0
    timeout_seconds: int = 300
    device: str = "cuda"
    backend: str = "gpu"  # 'gpu', 'cpu', or 'hybrid'
    use_progress_bar: bool = True
    save_game_records: bool = False
    enable_tree_reuse: bool = False  # Disable to avoid memory issues
    gc_frequency: int = 5  # Run garbage collection every N games
    initial_elo: float = 1500.0
    k_factor: float = 32.0  # ELO K-factor


class EloRatingSystem:
    """Advanced ELO rating system for tracking model strength
    
    Features:
    - Adaptive K-factor based on rating difference and games played
    - ELO deflation to prevent inflation over time
    - Uncertainty tracking for confidence intervals
    - Special handling for baseline models (random)
    - ELO inheritance for model iterations
    """
    
    def __init__(self, initial_rating: float = 1500.0, k_factor: float = 32.0,
                 anchor_rating: float = 0.0):
        """Initialize ELO rating system
        
        Args:
            initial_rating: Starting ELO rating for new models
            k_factor: Base K-factor for ELO updates
            anchor_rating: Rating for anchor model (random)
        """
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.anchor_rating = anchor_rating
        self.ratings = {}
        self.history = defaultdict(list)
        self.games_played = defaultdict(int)
        self.uncertainty = defaultdict(lambda: 100.0)  # Initial uncertainty
        
        # Enhanced features
        self.use_deflation = True
        self.deflation_factor = 0.99  # Deflate by 1% per generation
        self.min_k_factor = 8.0
        self.max_k_factor = 64.0
    
    def get_rating(self, player: str) -> float:
        """Get current rating for a player
        
        Args:
            player: Player name
            
        Returns:
            Current ELO rating
        """
        if player not in self.ratings:
            self.ratings[player] = self.initial_rating
            self.history[player].append(self.initial_rating)
        return self.ratings[player]
    
    def expected_score(self, rating1: float, rating2: float) -> float:
        """Calculate expected score for player 1
        
        Args:
            rating1: Player 1's rating
            rating2: Player 2's rating
            
        Returns:
            Expected score (0-1)
        """
        return 1.0 / (1.0 + 10 ** ((rating2 - rating1) / 400.0))
    
    def get_adaptive_k_factor(self, player1: str, player2: str) -> float:
        """Calculate adaptive K-factor based on rating difference and games played
        
        Args:
            player1: First player name
            player2: Second player name
            
        Returns:
            Adaptive K-factor
        """
        rating_diff = abs(self.get_rating(player1) - self.get_rating(player2))
        games1 = self.games_played[player1]
        games2 = self.games_played[player2]
        min_games = min(games1, games2)
        
        # Higher K for larger rating differences (more volatile)
        diff_factor = 1.0 + (rating_diff / 400.0)
        
        # Lower K for more games played (more stable)
        games_factor = max(0.5, 1.0 - (min_games / 100.0))
        
        # Calculate adaptive K
        adaptive_k = self.k_factor * diff_factor * games_factor
        
        # Clamp to reasonable range
        return max(self.min_k_factor, min(self.max_k_factor, adaptive_k))
    
    def should_play_vs_random(self, iteration: int, current_elo: float) -> bool:
        """Determine if model should play against random baseline
        
        Uses adaptive logic to reduce unnecessary random matches for strong models
        
        Args:
            iteration: Current iteration number
            current_elo: Current model's ELO rating
            
        Returns:
            Whether to play against random
        """
        # Always play random for first few iterations
        if iteration <= 3:
            return True
        
        # Skip if ELO is already very high (model is clearly strong)
        if current_elo > 600:  # 600 ELO above random is very strong
            # Only play every 10th iteration for validation
            return iteration % 10 == 0
        
        # Play more frequently for weaker models
        if current_elo < 200:
            return True  # Always play if weak
        
        # Adaptive frequency based on strength
        frequency = max(2, int(current_elo / 100))
        return iteration % frequency == 0
    
    def update_ratings(self, player1: str, player2: str, wins: int = None, 
                      draws: int = None, losses: int = None, score1: float = None):
        """Update ratings after a game or match
        
        Args:
            player1: First player name
            player2: Second player name
            wins: Number of wins for player1 (optional)
            draws: Number of draws (optional)  
            losses: Number of losses for player1 (optional)
            score1: Direct score for player 1 (optional, 1=win, 0.5=draw, 0=loss)
        """
        # Calculate score from wins/draws/losses if provided
        if wins is not None and draws is not None and losses is not None:
            total_games = wins + draws + losses
            if total_games > 0:
                score1 = (wins + 0.5 * draws) / total_games
            else:
                return  # No games played
        elif score1 is None:
            raise ValueError("Must provide either wins/draws/losses or score1")
        
        rating1 = self.get_rating(player1)
        rating2 = self.get_rating(player2)
        
        expected1 = self.expected_score(rating1, rating2)
        expected2 = 1.0 - expected1
        
        # Use adaptive K-factor
        adaptive_k = self.get_adaptive_k_factor(player1, player2)
        
        # Update ratings
        new_rating1 = rating1 + adaptive_k * (score1 - expected1)
        new_rating2 = rating2 + adaptive_k * ((1 - score1) - expected2)
        
        # Apply deflation if enabled
        if self.use_deflation and player1 != "random" and player2 != "random":
            new_rating1 *= self.deflation_factor
            new_rating2 *= self.deflation_factor
        
        # Ensure random stays at anchor rating
        if player1 == "random":
            new_rating1 = self.anchor_rating
        if player2 == "random":
            new_rating2 = self.anchor_rating
        
        self.ratings[player1] = new_rating1
        self.ratings[player2] = new_rating2
        
        # Update games played
        self.games_played[player1] += total_games if wins is not None else 1
        self.games_played[player2] += total_games if wins is not None else 1
        
        # Update uncertainty (decreases with more games)
        self.uncertainty[player1] *= 0.95
        self.uncertainty[player2] *= 0.95
        
        # Track history
        self.history[player1].append(new_rating1)
        self.history[player2].append(new_rating2)
    
    def get_rating_history(self, player: str) -> List[float]:
        """Get rating history for a player
        
        Args:
            player: Player name
            
        Returns:
            List of historical ratings
        """
        if player not in self.history:
            return [self.initial_rating]
        return self.history[player]


class ArenaMatch:
    """Manages a single arena match between two models"""
    
    def __init__(self, game_interface, evaluator1, evaluator2, 
                 config: ArenaConfig, match_id: str, log_dir: Optional[Path] = None):
        """Initialize arena match
        
        Args:
            game_interface: Game interface for creating games
            evaluator1: First model's evaluator
            evaluator2: Second model's evaluator
            config: Arena configuration
            match_id: Unique match identifier
            log_dir: Directory to save game logs (optional)
        """
        self.game_interface = game_interface
        self.evaluator1 = evaluator1
        self.evaluator2 = evaluator2
        self.config = config
        self.match_id = match_id
        self.game_results = []
        self.log_dir = log_dir
        
        # Storage for detailed game records if logging is enabled
        self.game_records = [] if config.save_game_records else None
    
    def play_single_game(self, game_idx: int) -> int:
        """Play a single game
        
        Args:
            game_idx: Game index
            
        Returns:
            Winner: 1 for model 1, -1 for model 2, 0 for draw
        """
        # Create new game state
        state = self.game_interface.create_initial_state()
        
        # Create MCTS configs for both players
        mcts_config = MCTSConfig(
            num_simulations=self.config.mcts_simulations,
            c_puct=self.config.c_puct,
            temperature=self.config.temperature,
            device=self.config.device,
            backend=self.config.backend,
            enable_subtree_reuse=self.config.enable_tree_reuse
        )
        
        # Initialize MCTS for both players - use CPU optimized version if backend is CPU
        if self.config.backend == 'cpu':
            from mcts.cpu.cpu_mcts_wrapper import create_cpu_optimized_mcts
            mcts1 = create_cpu_optimized_mcts(mcts_config, self.evaluator1, self.game_interface)
            mcts2 = create_cpu_optimized_mcts(mcts_config, self.evaluator2, self.game_interface)
        else:
            mcts1 = MCTS(config=mcts_config, evaluator=self.evaluator1, 
                         game_interface=self.game_interface)
            mcts2 = MCTS(config=mcts_config, evaluator=self.evaluator2,
                         game_interface=self.game_interface)
        
        # Map player to MCTS (use 0-based indexing to match game interface)
        player_mcts = {0: mcts1, 1: mcts2}
        
        # Play game
        move_count = 0
        current_player = 0  # Start with player 0 (game interface uses 0-based indexing)
        
        # Initialize game record if logging is enabled
        if self.config.save_game_records:
            game_record = {
                'game_idx': game_idx,
                'match_id': self.match_id,
                'moves': [],
                'action_probs': [],
                'timestamps': [],
                'evaluator1': str(type(self.evaluator1).__name__),
                'evaluator2': str(type(self.evaluator2).__name__)
            }
            start_time = time.time()
        
        while not self.game_interface.is_terminal(state):
            # Get canonical form for current player
            canonical_state = self.game_interface.get_canonical_form(state, current_player)
            
            # Get action from appropriate MCTS
            mcts = player_mcts[current_player]
            action_probs = mcts.search(canonical_state)
            
            # Select best action (temperature=0 for arena)
            # CRITICAL FIX: Ensure selected action is legal
            # Note: get_legal_moves needs the original state, not the canonical numpy array
            legal_moves = self.game_interface.get_legal_moves(state)
            if len(legal_moves) == 0:
                raise RuntimeError("No legal moves available")
            
            # Find the best legal action
            legal_probs = action_probs[legal_moves]
            best_legal_idx = np.argmax(legal_probs)
            action = legal_moves[best_legal_idx]
            
            # Record move details if logging is enabled
            if self.config.save_game_records:
                game_record['moves'].append({
                    'move_number': move_count,
                    'player': current_player + 1,  # Convert to 1-based for display
                    'action': int(action),
                    'action_string': self.game_interface.action_to_string(state, action),
                    'probability': float(action_probs[action])
                })
                game_record['action_probs'].append(action_probs.tolist())
                game_record['timestamps'].append(time.time() - start_time)
            
            # Make move
            state = self.game_interface.get_next_state(state, action)
            move_count += 1
            
            # Switch player (toggle between 0 and 1)
            current_player = 1 - current_player
            
            # Prevent infinite games
            if move_count > 500:
                logger.warning(f"Game {game_idx} exceeded 500 moves, declaring draw")
                return 0
        
        # Get winner
        winner = self.game_interface.get_winner(state)
        self.game_results.append({
            'game_idx': game_idx,
            'winner': winner,
            'moves': move_count
        })
        
        # Complete and store game record if logging is enabled
        if self.config.save_game_records:
            game_record['winner'] = int(winner)
            game_record['total_moves'] = move_count
            game_record['duration'] = time.time() - start_time
            self.game_records.append(game_record)
        
        return winner
    
    def play_match(self) -> Dict[str, Any]:
        """Play a full match
        
        Returns:
            Match results
        """
        model1_wins = 0
        model2_wins = 0
        draws = 0
        
        # Add progress bar if enabled
        if self.config.use_progress_bar:
            pbar = tqdm(
                total=self.config.num_games,
                desc=f"Arena match {self.match_id}",
                leave=False
            )
        
        for game_idx in range(self.config.num_games):
            # Alternate starting player
            if game_idx % 2 == 0:
                winner = self.play_single_game(game_idx)
            else:
                # Swap evaluators for this game
                self.evaluator1, self.evaluator2 = self.evaluator2, self.evaluator1
                winner = -self.play_single_game(game_idx)  # Negate result
                self.evaluator1, self.evaluator2 = self.evaluator2, self.evaluator1
            
            if winner == 1:
                model1_wins += 1
            elif winner == -1:
                model2_wins += 1
            else:
                draws += 1
            
            # Update progress bar
            if self.config.use_progress_bar:
                pbar.update(1)
                pbar.set_postfix({
                    'M1': model1_wins,
                    'M2': model2_wins,
                    'D': draws
                })
        
        # Close progress bar
        if self.config.use_progress_bar:
            pbar.close()
        
        total_games = model1_wins + model2_wins + draws
        win_rate = model1_wins / total_games if total_games > 0 else 0.5
        
        # Save game records if logging is enabled
        if self.config.save_game_records and self.game_records and self.log_dir:
            self.save_game_records()
        
        return {
            'total_games': total_games,
            'model1_wins': model1_wins,
            'model2_wins': model2_wins,
            'draws': draws,
            'win_rate': win_rate,
            'game_results': self.game_results
        }
    
    def save_game_records(self):
        """Save detailed game records to file"""
        if not self.log_dir or not self.game_records:
            return
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = self.log_dir / f"arena_games_{self.match_id}_{timestamp}.json"
        
        # Save records
        import json
        with open(filename, 'w') as f:
            json.dump({
                'match_id': self.match_id,
                'num_games': len(self.game_records),
                'evaluator1': str(type(self.evaluator1).__name__),
                'evaluator2': str(type(self.evaluator2).__name__),
                'games': self.game_records
            }, f, indent=2)
        
        logger.info(f"Saved {len(self.game_records)} game records to {filename}")


class ArenaManager:
    """Manages arena evaluation between models"""
    
    def __init__(self, config: AlphaZeroConfig, arena_config: ArenaConfig = None, game_interface=None, log_dir: Optional[Path] = None):
        """Initialize arena manager
        
        Args:
            config: AlphaZero configuration
            arena_config: Arena configuration (optional, for backward compatibility)
            game_interface: Game interface to use (optional)
            log_dir: Directory to save game logs (optional)
        """
        self.config = config
        self.game_interface = game_interface
        self.log_dir = log_dir
        
        # Use provided arena config or create from AlphaZero config
        if arena_config is not None:
            self.arena_config = arena_config
        else:
            self.arena_config = ArenaConfig(
                num_games=config.arena.num_games,
                win_threshold=config.arena.win_threshold,
                num_workers=config.arena.num_workers,
                mcts_simulations=config.arena.mcts_simulations,
                temperature=config.arena.temperature,
                device=config.mcts.device,
                initial_elo=getattr(config.arena, 'initial_elo', 1500.0),
                k_factor=getattr(config.arena, 'k_factor', 32.0)
            )
        
        # Initialize ELO system with enhanced features
        self.elo_system = EloRatingSystem(
            initial_rating=self.arena_config.initial_elo,
            k_factor=self.arena_config.k_factor,
            anchor_rating=getattr(config.arena, 'elo_anchor_rating', 0.0)
        )
        
        # Configure enhanced ELO features from config
        if hasattr(config.arena, 'elo_enable_deflation'):
            self.elo_system.use_deflation = config.arena.elo_enable_deflation
        if hasattr(config.arena, 'elo_deflation_factor'):
            self.elo_system.deflation_factor = config.arena.elo_deflation_factor
        
        # Set random model as anchor
        self.elo_system.ratings["random"] = self.elo_system.anchor_rating
    
    def evaluate_models(self, evaluator1, evaluator2, 
                       model1_name: str, model2_name: str) -> Dict[str, Any]:
        """Evaluate two models against each other
        
        Args:
            evaluator1: First model's evaluator
            evaluator2: Second model's evaluator
            model1_name: Name of first model
            model2_name: Name of second model
            
        Returns:
            Evaluation results including win rate and ELO updates
        """
        logger.info(f"Arena: {model1_name} vs {model2_name} - {self.arena_config.num_games} games (Single GPU)")
        
        # Convert evaluators to SingleGPUEvaluator if needed
        if hasattr(evaluator1, 'model'):
            gpu_eval1 = SingleGPUEvaluator(
                model=evaluator1.model,
                device=self.arena_config.device,
                action_size=self.config.game.board_size ** 2,
                use_mixed_precision=getattr(self.config.mcts, 'use_mixed_precision', True)
            )
        else:
            gpu_eval1 = evaluator1
            
        if hasattr(evaluator2, 'model'):
            gpu_eval2 = SingleGPUEvaluator(
                model=evaluator2.model,
                device=self.arena_config.device,
                action_size=self.config.game.board_size ** 2,
                use_mixed_precision=getattr(self.config.mcts, 'use_mixed_precision', True)
            )
        else:
            gpu_eval2 = evaluator2
        
        # Always use single-process matches for single-GPU
        results = self._run_single_gpu_matches(gpu_eval1, gpu_eval2)
        
        # Update ELO ratings using wins/draws/losses
        self.elo_system.update_ratings(
            model1_name, model2_name,
            wins=results['model1_wins'],
            draws=results['draws'],
            losses=results['model2_wins']
        )
        
        # Add ELO info to results
        results['elo_ratings'] = {
            model1_name: self.elo_system.get_rating(model1_name),
            model2_name: self.elo_system.get_rating(model2_name)
        }
        results['passed_threshold'] = results['win_rate'] >= self.arena_config.win_threshold
        
        logger.info(f"Arena results: {model1_name} win rate = {results['win_rate']:.2%}")
        logger.info(f"ELO ratings: {model1_name}={results['elo_ratings'][model1_name]:.0f}, "
                   f"{model2_name}={results['elo_ratings'][model2_name]:.0f}")
        
        return results
    
    def _run_single_gpu_matches(self, evaluator1, evaluator2) -> Dict[str, Any]:
        """Run matches in a single process"""
        # Create game interface if not provided
        if self.game_interface is None:
            from mcts.core.game_interface import GameInterface, GameType
            # Get game type from config
            game_type_str = getattr(self.config, 'game_type', 'gomoku')
            game_type_map = {
                'chess': GameType.CHESS,
                'go': GameType.GO,
                'gomoku': GameType.GOMOKU
            }
            game_type = game_type_map.get(game_type_str.lower(), GameType.GOMOKU)
            self.game_interface = GameInterface(game_type)
        
        match = ArenaMatch(
            game_interface=self.game_interface,
            evaluator1=evaluator1,
            evaluator2=evaluator2,
            config=self.arena_config,
            match_id="single_gpu",
            log_dir=self.log_dir
        )
        
        return match.play_match()
    
    def _run_parallel_matches_deprecated(self, evaluator1, evaluator2) -> Dict[str, Any]:
        """Run matches in parallel"""
        # Divide games among workers
        games_per_worker = self.arena_config.num_games // self.arena_config.num_workers
        remainder = self.arena_config.num_games % self.arena_config.num_workers
        
        work_distribution = []
        for i in range(self.arena_config.num_workers):
            num_games = games_per_worker + (1 if i < remainder else 0)
            if num_games > 0:
                work_distribution.append((i, num_games))
        
        # Run matches in parallel
        all_results = []
        with ProcessPoolExecutor(max_workers=self.arena_config.num_workers) as executor:
            futures = []
            
            for worker_id, num_games in work_distribution:
                # Create a subset arena config
                worker_config = ArenaConfig(**self.arena_config.__dict__)
                worker_config.num_games = num_games
                
                future = executor.submit(
                    self._worker_play_matches,
                    worker_id,
                    evaluator1,
                    evaluator2,
                    worker_config
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                results = future.result()
                all_results.append(results)
        
        # Aggregate results
        total_games = sum(r['total_games'] for r in all_results)
        model1_wins = sum(r['model1_wins'] for r in all_results)
        model2_wins = sum(r['model2_wins'] for r in all_results)
        draws = sum(r['draws'] for r in all_results)
        
        return {
            'total_games': total_games,
            'model1_wins': model1_wins,
            'model2_wins': model2_wins,
            'draws': draws,
            'win_rate': model1_wins / total_games if total_games > 0 else 0.5
        }
    
    def _worker_play_matches_deprecated(self, worker_id: int, evaluator1, evaluator2,
                           config: ArenaConfig) -> Dict[str, Any]:
        """Worker function for parallel match execution"""
        # Create game interface if not provided
        if self.game_interface is None:
            from mcts.core.game_interface import GameInterface, GameType
            # Get game type from config
            game_type_str = getattr(self.config, 'game_type', 'gomoku')
            game_type_map = {
                'chess': GameType.CHESS,
                'go': GameType.GO,
                'gomoku': GameType.GOMOKU
            }
            game_type = game_type_map.get(game_type_str.lower(), GameType.GOMOKU)
            game_interface = GameInterface(game_type)
        else:
            game_interface = self.game_interface
            
        match = ArenaMatch(
            game_interface=game_interface,
            evaluator1=evaluator1,
            evaluator2=evaluator2,
            config=config,
            match_id=f"worker_{worker_id}"
        )
        
        return match.play_match()
    
    def inherit_elo(self, new_model: str, parent_model: str):
        """Set up ELO inheritance for a new model
        
        Args:
            new_model: Name of the new model
            parent_model: Name of the parent model to inherit from
        """
        parent_elo = self.elo_system.get_rating(parent_model)
        self.elo_system.ratings[new_model] = parent_elo
        self.elo_system.history[new_model].append(parent_elo)
        logger.info(f"Model {new_model} inheriting ELO {parent_elo:.1f} from {parent_model}")
    
    def compare_models(self, model1, model2, model1_name: str, model2_name: str,
                      silent: bool = False) -> Tuple[int, int, int]:
        """Compare two models (backward compatibility method)
        
        Args:
            model1: First model or evaluator
            model2: Second model or evaluator
            model1_name: Name of first model
            model2_name: Name of second model
            silent: Whether to suppress output
            
        Returns:
            Tuple of (wins, draws, losses) for model1
        """
        # If models are nn.Module, wrap them in evaluators
        if hasattr(model1, 'forward'):
            # It's a neural network model, create evaluator
            from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
            from mcts.core.evaluator import EvaluatorConfig
            
            eval_config = EvaluatorConfig(device=self.arena_config.device)
            evaluator1 = ResNetEvaluator(model1, eval_config)
        else:
            # Already an evaluator
            evaluator1 = model1
            
        if hasattr(model2, 'forward'):
            # It's a neural network model, create evaluator
            from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
            from mcts.core.evaluator import EvaluatorConfig
            
            eval_config = EvaluatorConfig(device=self.arena_config.device)
            evaluator2 = ResNetEvaluator(model2, eval_config)
        else:
            # Already an evaluator
            evaluator2 = model2
        
        # Run evaluation
        results = self.evaluate_models(evaluator1, evaluator2, model1_name, model2_name)
        
        # Return in old format (wins, draws, losses)
        return results['model1_wins'], results['draws'], results['model2_wins']
    
    def get_leaderboard(self) -> List[Tuple[str, float]]:
        """Get current ELO leaderboard
        
        Returns:
            List of (model_name, elo_rating) tuples sorted by rating
        """
        leaderboard = [(name, rating) for name, rating in self.elo_system.ratings.items()]
        leaderboard.sort(key=lambda x: x[1], reverse=True)
        return leaderboard
    
    def get_dynamic_win_rate_threshold(self, iteration: int) -> float:
        """Calculate dynamic win rate threshold using logarithmic scheduling
        
        This implements a progressive training approach where early models
        need to beat random by smaller margins, but later models need to
        demonstrate stronger performance.
        
        Formula: threshold(N) = 0.4366 * log10(N / 0.7682)
        Target: 100% at iteration 150
        Clamped between 5% and 100%
        
        Args:
            iteration: Current training iteration
            
        Returns:
            Win rate threshold for accepting the model
        """
        import numpy as np
        
        # Logarithmic scheduling parameters (pre-calculated for target=150)
        A = 0.4366
        B = 0.7682
        
        # Calculate threshold
        if iteration < 1:
            return 0.05
        
        threshold = A * np.log10(iteration / B)
        
        # Clamp between 5% and 100%
        return min(max(threshold, 0.05), 1.0)
    
    def evaluate_with_previous(self, current_evaluator, current_name: str,
                             previous_model_path: Path, iteration: int) -> Dict[str, Any]:
        """Evaluate current model against previous iteration for ELO calibration
        
        This is critical for maintaining ELO consistency as models should
        inherit their ELO from the previous iteration they were trained from.
        
        Args:
            current_evaluator: Evaluator for current model
            current_name: Name of current model (e.g. "iter_5")
            previous_model_path: Path to previous model checkpoint
            iteration: Current iteration number
            
        Returns:
            Evaluation results including ELO updates
        """
        if iteration <= 1 or not previous_model_path.exists():
            return None
            
        logger.info(f"Evaluating {current_name} vs previous iteration")
        
        # Load previous model
        import torch
        from mcts.neural_networks.resnet_model import create_resnet_for_game
        
        # Create model instance
        previous_model = create_resnet_for_game(
            game_type=self.config.game.game_type,
            input_channels=self.config.network.input_channels,
            num_blocks=self.config.network.num_res_blocks,
            num_filters=self.config.network.num_filters
        )
        
        # Load checkpoint
        checkpoint = torch.load(previous_model_path, map_location=self.arena_config.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            previous_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            previous_model.load_state_dict(checkpoint)
        previous_model.eval()
        previous_model = previous_model.to(self.arena_config.device)
        
        # Create evaluator for previous model
        from mcts.core.evaluator import AlphaZeroEvaluator
        previous_evaluator = AlphaZeroEvaluator(
            model=previous_model,
            device=self.arena_config.device
        )
        
        previous_name = f"iter_{iteration - 1}"
        
        # Run evaluation
        results = self.evaluate_models(
            current_evaluator, previous_evaluator,
            current_name, previous_name
        )
        
        # Clean up
        del previous_model, previous_evaluator
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return results
    
    def adjust_elo_for_new_best(self, current_model: str, best_model: str, 
                                win_rate_vs_best: float) -> float:
        """Adjust ELO for new best model to ensure monotonic improvement
        
        When a model beats the current best and becomes the new best, its ELO
        must be higher than the previous best. This method calculates the actual
        ELO gain based on match performance and adjusts if necessary.
        
        Algorithm:
        1. Calculate actual ELO gain based on win rate using ELO formula
        2. If current ELO < best ELO despite winning, assume both started equal
        3. Add the actual performance-based gain to ensure new best > old best
        
        Args:
            current_model: Name of current model that beat the best
            best_model: Name of previous best model
            win_rate_vs_best: Win rate of current vs best (>= win_threshold)
            
        Returns:
            Adjusted ELO rating for the new best model
        """
        current_elo = self.elo_system.get_rating(current_model)
        best_elo = self.elo_system.get_rating(best_model)
        
        # Calculate the actual ELO gain based on match performance
        # This uses the actual ELO system calculations
        actual_score = win_rate_vs_best  # Win rate is the actual score
        expected_score = 0.5  # If both models had equal rating
        
        # Get the K-factor that would be used for this match
        k_factor = self.elo_system.get_adaptive_k_factor(current_model, best_model)
        
        # Calculate the ELO gain based on actual performance
        # This represents how much better the new model is than the old best
        actual_elo_gain = k_factor * (actual_score - expected_score)
        
        # The new best model should have AT LEAST the old best ELO plus the actual gain
        # This ensures monotonic improvement while respecting the ELO system's math
        required_elo = best_elo + actual_elo_gain
        
        if current_elo < required_elo:
            # Adjust ELO to meet the requirement
            logger.info(f"Adjusting ELO for new best model (using actual ELO gain calculation):")
            logger.info(f"  Current ELO from matches: {current_elo:.1f}")
            logger.info(f"  Previous best ELO: {best_elo:.1f}")
            logger.info(f"  Win rate vs best: {win_rate_vs_best:.1%}")
            logger.info(f"  Calculated ELO gain: +{actual_elo_gain:.1f} (K={k_factor:.1f}, score={actual_score:.3f})")
            logger.info(f"  Adjusting to: {required_elo:.1f}")
            
            # Update the rating in the tracker
            self.elo_system.ratings[current_model] = required_elo
            current_elo = required_elo
        
        # Log the progression
        elo_gain = current_elo - best_elo
        logger.info(f"Best model progression: {best_model} (ELO: {best_elo:.1f}) -> {current_model} (ELO: {current_elo:.1f})")
        logger.info(f"Net ELO gain: {elo_gain:+.1f}")
        
        # Validation
        if elo_gain <= 0:
            raise ValueError(f"New best model must have higher ELO than previous best (gain: {elo_gain:+.1f})")
        
        return current_elo