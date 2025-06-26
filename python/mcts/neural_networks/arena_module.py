"""Arena module for model evaluation and comparison

This module handles model battles, ELO tracking, and tournament organization.
"""

import logging
import json
import time
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
import queue

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
    mcts_simulations: int = 200  # Reduced for faster arena matches
    c_puct: float = 1.0
    temperature: float = 0.0
    temperature_threshold: int = 0
    timeout_seconds: int = 300
    device: str = "cuda"
    use_progress_bar: bool = True
    save_game_records: bool = False
    enable_tree_reuse: bool = False  # Disable by default to avoid memory issues
    gc_frequency: int = 5  # Run garbage collection every N games (reduced from 10)
    memory_monitoring: bool = True  # Enable memory monitoring
    max_memory_gb: float = 6.0  # Maximum GPU memory to use (leave 2GB buffer)


class ELOTracker:
    """Tracks ELO ratings for models with adaptive K-factor and match scheduling"""
    
    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1500.0):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        # Use a regular dict instead of defaultdict with lambda
        self.ratings: Dict[str, float] = {}
        self.game_history: List[Dict] = []
        self.iteration_elos: Dict[int, float] = {}  # Track ELO by iteration
    
    def get_adaptive_k_factor(self, player: str, opponent: str, base_k: float = None) -> float:
        """Calculate adaptive K-factor based on player strength and opponent"""
        if base_k is None:
            base_k = self.k_factor
            
        # Special handling for matches against random
        if opponent == "random":
            player_elo = self.ratings.get(player, self.initial_rating)
            
            # Sophisticated adaptive K-factor for random matches
            # Phase 1 (ELO 0-200): High K-factor for rapid initial learning
            # Phase 2 (ELO 200-500): Moderate K-factor for calibration
            # Phase 3 (ELO 500-1000): Reduced K-factor to prevent inflation
            # Phase 4 (ELO 1000+): Minimal K-factor, mostly for verification
            
            if player_elo < 200:
                # Early phase: Full K-factor for rapid ELO growth
                k_multiplier = 1.0
            elif player_elo < 500:
                # Calibration phase: Gradual reduction
                k_multiplier = 1.0 - 0.3 * (player_elo - 200) / 300
            elif player_elo < 1000:
                # Stabilization phase: Further reduction
                k_multiplier = 0.7 - 0.4 * (player_elo - 500) / 500
            else:
                # Mature phase: Minimal gains from beating random
                # Use logarithmic decay to ensure smooth transition
                k_multiplier = 0.3 * (1000 / player_elo) ** 0.5
            
            # Apply win rate adjustment: if model dominates random, reduce K further
            # Get recent performance vs random
            recent_vs_random = self.get_recent_performance(player, "random", last_n=3)
            if recent_vs_random is not None and recent_vs_random > 0.95:
                # Near-perfect performance: reduce K-factor by additional 50%
                k_multiplier *= 0.5
            elif recent_vs_random is not None and recent_vs_random > 0.98:
                # Essentially solved: reduce K-factor by 80%
                k_multiplier *= 0.2
            
            return base_k * k_multiplier
        
        # For non-random opponents, use standard K-factor with minor adjustments
        # Reduce K-factor for very high-rated players to prevent volatility
        player_elo = self.ratings.get(player, self.initial_rating)
        if player_elo > 2000:
            return base_k * 0.8
        elif player_elo > 2500:
            return base_k * 0.6
        else:
            return base_k
    
    def get_recent_performance(self, player: str, opponent: str, last_n: int = 5) -> Optional[float]:
        """Get recent win rate of player vs opponent in last N matches"""
        recent_games = []
        for record in reversed(self.game_history):
            if (record["player1"] == player and record["player2"] == opponent) or \
               (record["player1"] == opponent and record["player2"] == player):
                recent_games.append(record)
                if len(recent_games) >= last_n:
                    break
        
        if not recent_games:
            return None
        
        total_wins = 0
        total_games = 0
        for game in recent_games:
            if game["player1"] == player:
                total_wins += game["wins"] + 0.5 * game["draws"]
                total_games += game["wins"] + game["draws"] + game["losses"]
            else:
                total_wins += game["losses"] + 0.5 * game["draws"]
                total_games += game["wins"] + game["draws"] + game["losses"]
        
        return total_wins / total_games if total_games > 0 else None
    
    def should_play_vs_random(self, iteration: int, current_elo: float) -> bool:
        """Determine if model should play against random based on sophisticated criteria"""
        # Always play in first few iterations for calibration
        if iteration <= 3:
            return True
        
        # Phase-based approach with ELO consideration
        # Phase 1 (iter 1-10): Frequent matches for calibration
        # Phase 2 (iter 11-50): Reduce frequency based on ELO growth
        # Phase 3 (iter 51-100): Sparse matches for verification
        # Phase 4 (iter 100+): Rare matches only when needed
        
        # Get ELO growth rate
        elo_growth_rate = self.get_elo_growth_rate(iteration)
        
        if iteration <= 10:
            # Early phase: Play frequently but skip if ELO is growing steadily
            if elo_growth_rate > 20 and current_elo > 200:
                return iteration % 3 == 0
            return True
        
        elif iteration <= 50:
            # Mid phase: Adaptive frequency based on ELO and growth
            if current_elo < 500:
                # Still calibrating: play every 3 iterations
                return iteration % 3 == 0
            elif current_elo < 1000:
                # Stabilizing: play every 5 iterations
                return iteration % 5 == 0
            else:
                # Strong model: play every 10 iterations
                return iteration % 10 == 0
        
        elif iteration <= 100:
            # Late phase: Only for verification
            if elo_growth_rate < 5 and iteration % 20 == 0:
                # Stagnating: check against random
                return True
            elif current_elo < 1500:
                # Not yet expert level: occasional checks
                return iteration % 15 == 0
            else:
                # Expert level: rare checks
                return iteration % 25 == 0
        
        else:
            # Very late phase: Minimal random matches
            # Only when concerning patterns emerge
            if elo_growth_rate < 0 and iteration % 30 == 0:
                # Declining ELO: verify against baseline
                return True
            elif iteration % 50 == 0:
                # Periodic sanity check
                return True
            else:
                return False
    
    def get_elo_growth_rate(self, iteration: int, window: int = 5) -> float:
        """Calculate average ELO growth rate over recent iterations"""
        if iteration < window:
            return 0.0
        
        recent_elos = []
        for i in range(max(1, iteration - window), iteration + 1):
            if i in self.iteration_elos:
                recent_elos.append(self.iteration_elos[i])
        
        if len(recent_elos) < 2:
            return 0.0
        
        # Calculate average growth per iteration
        growth = (recent_elos[-1] - recent_elos[0]) / len(recent_elos)
        return growth
    
    def update_ratings(self, player1: str, player2: str,
                      wins: int, draws: int, losses: int):
        """Update ELO ratings based on match results with adaptive K-factor"""
        total_games = wins + draws + losses
        if total_games == 0:
            return
        
        # Get current ratings (use initial_rating if not found)
        r1 = self.ratings.get(player1, self.initial_rating)
        r2 = self.ratings.get(player2, self.initial_rating)
        
        # Get adaptive K-factors for both players
        k1 = self.get_adaptive_k_factor(player1, player2)
        k2 = self.get_adaptive_k_factor(player2, player1)
        
        # Calculate expected scores
        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))
        
        # Actual scores
        s1 = (wins + 0.5 * draws) / total_games
        s2 = (losses + 0.5 * draws) / total_games
        
        # Calculate rating changes with adaptive K-factors
        delta1 = k1 * (s1 - e1)
        delta2 = k2 * (s2 - e2)
        
        # Update ratings (but keep "random" fixed at 0 as anchor)
        # Standard ELO formula: new_rating = old_rating + K * (actual_score - expected_score)
        # Note: We do NOT multiply by total_games - K factor already accounts for match weight
        if player1 != "random":
            new_r1 = r1 + delta1
            self.ratings[player1] = new_r1
            
            # Track iteration ELO if player is an iteration model
            if player1.startswith("iter_"):
                try:
                    iter_num = int(player1.split("_")[1])
                    self.iteration_elos[iter_num] = new_r1
                except (ValueError, IndexError):
                    pass
            
            # Log with adaptive K-factor info
            logger.debug(f"ELO Update: {player1} {r1:.1f} -> {new_r1:.1f} (Δ{delta1:+.1f}, K={k1:.1f}) after {wins}W-{draws}D-{losses}L vs {player2}")
        else:
            new_r1 = r1
            
        if player2 != "random":
            new_r2 = r2 + delta2
            self.ratings[player2] = new_r2
            
            # Track iteration ELO if player is an iteration model
            if player2.startswith("iter_"):
                try:
                    iter_num = int(player2.split("_")[1])
                    self.iteration_elos[iter_num] = new_r2
                except (ValueError, IndexError):
                    pass
            
            logger.debug(f"ELO Update: {player2} {r2:.1f} -> {new_r2:.1f} (Δ{delta2:+.1f}, K={k2:.1f}) after {losses}W-{draws}D-{wins}L vs {player1}")
        else:
            new_r2 = r2
        
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
            "new_rating1": new_r1,
            "new_rating2": new_r2,
            "k_factor1": k1,
            "k_factor2": k2
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
        input_representation = getattr(config.network, 'input_representation', 'basic')
        self.game_interface = GameInterface(
            self.game_type,
            board_size=config.game.board_size,
            input_representation=input_representation
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
        
        # Use default num_games if not specified
        if num_games is None:
            num_games = self.arena_config.num_games
        
        # Clean GPU memory before starting arena
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            if (is_random1 or is_random2) and self.arena_config.num_workers > 1:
                # Use parallel arena for NN vs Random matches
                return self._parallel_arena(model1, model2, num_games, silent)
            else:
                # Use sequential arena for NN vs NN matches to avoid CUDA issues
                return self._sequential_arena(model1, model2, num_games, silent)
        except Exception as e:
            logger.error(f"[ARENA] Error in compare_models: {e}")
            # Handle CUDA errors gracefully
            if "CUDA error" in str(e):
                logger.warning("[ARENA] CUDA error encountered, attempting recovery")
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                except:
                    logger.warning("[ARENA] Could not clear CUDA cache during cleanup")
            raise
        finally:
            # Always clean up after arena battles
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except:
                    logger.warning("[ARENA] Could not clear CUDA cache during cleanup")
    
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
        # Track if we created evaluators to clean them up later
        created_evaluator1 = False
        created_evaluator2 = False
        
        if isinstance(model1, Evaluator):
            evaluator1 = model1
        else:
            # Ensure model is in eval mode and gradients disabled
            model1.eval()
            # Get action size from game interface
            initial_state = self.game_interface.create_initial_state()
            action_size = self.game_interface.get_action_space_size(initial_state)
            evaluator1 = AlphaZeroEvaluator(
                model=model1,
                device=self.arena_config.device,
                action_size=action_size
            )
            created_evaluator1 = True
            
        if isinstance(model2, Evaluator):
            evaluator2 = model2
        else:
            # Ensure model is in eval mode and gradients disabled
            model2.eval()
            # Get action size from game interface
            initial_state = self.game_interface.create_initial_state()
            action_size = self.game_interface.get_action_space_size(initial_state)
            evaluator2 = AlphaZeroEvaluator(
                model=model2,
                device=self.arena_config.device,
                action_size=action_size
            )
            created_evaluator2 = True
        
        # Warm up neural networks to trigger CUDA compilation before arena starts
        if not silent and torch.cuda.is_available():
            # Get the correct number of input channels from models
            input_channels1 = getattr(self.config.network, 'input_channels', 18)
            input_channels2 = getattr(self.config.network, 'input_channels', 18)
            
            # Try to get actual input channels from model metadata if available
            if hasattr(evaluator1, 'model') and hasattr(evaluator1.model, 'metadata'):
                input_channels1 = evaluator1.model.metadata.input_channels
            if hasattr(evaluator2, 'model') and hasattr(evaluator2.model, 'metadata'):
                input_channels2 = evaluator2.model.metadata.input_channels
                
            with torch.no_grad():
                # Warm up both evaluators if they have models
                if hasattr(evaluator1, 'model') and hasattr(evaluator1.model, 'forward'):
                    dummy_state1 = torch.randn(1, input_channels1, self.config.game.board_size, 
                                              self.config.game.board_size, device=self.arena_config.device)
                    _ = evaluator1.model(dummy_state1)
                if hasattr(evaluator2, 'model') and hasattr(evaluator2.model, 'forward'):
                    dummy_state2 = torch.randn(1, input_channels2, self.config.game.board_size, 
                                              self.config.game.board_size, device=self.arena_config.device)
                    _ = evaluator2.model(dummy_state2)
                torch.cuda.synchronize()
                
            # Pre-compile CUDA kernels to avoid JIT compilation during games
            try:
                from mcts.gpu.unified_kernels import get_unified_kernels
                logger.debug("Pre-loading CUDA kernels for arena...")
                kernels = get_unified_kernels('cuda')
                # Force compilation by calling a simple operation
                if kernels and hasattr(kernels, 'compile'):
                    kernels.compile()
            except Exception as e:
                logger.debug(f"Could not pre-compile CUDA kernels: {e}")
        
        # Progress bar
        disable_progress = silent or logger.level > logging.INFO or not self.arena_config.use_progress_bar
        
        # Arena progress at position 4 to avoid overlap with other progress bars
        with tqdm(total=num_games, desc="Arena games", unit="game",
                 disable=disable_progress, position=4, leave=False) as pbar:
            # Run all games with no gradient tracking
            with torch.no_grad():
                for game_idx in range(num_games):
                    # Alternate who plays first
                    if game_idx % 2 == 0:
                        result = self._play_single_game(evaluator1, evaluator2, game_idx)
                    else:
                        result = -self._play_single_game(evaluator2, evaluator1, game_idx)
                    
                    # Count result
                    if result > 0:
                        wins += 1
                    elif result < 0:
                        losses += 1
                    else:
                        draws += 1
                    
                    # Update progress
                    pbar.update(1)
                    pbar.set_postfix({
                        'W': wins,
                        'D': draws,
                        'L': losses,
                        'WR': f'{wins/(wins+draws+losses):.1%}'
                    })
        
        # Clean up evaluators if we created them
        if created_evaluator1:
            del evaluator1
        if created_evaluator2:
            del evaluator2
            
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return wins, draws, losses
    
    def _parallel_arena(self, model1, model2,
                       num_games: int, silent: bool = False) -> Tuple[int, int, int]:
        """Run arena games in parallel using GPU evaluation service"""
        from mcts.utils.gpu_evaluator_service import GPUEvaluatorService
        from mcts.core.evaluator import Evaluator, RandomEvaluator, AlphaZeroEvaluator
        import multiprocessing as mp
        
        wins, draws, losses = 0, 0, 0
        
        # Get resource allocation from config (similar to self-play)
        allocation = getattr(self.config, '_resource_allocation', None)
        if allocation is None:
            # Fallback: calculate allocation if not already done
            hardware = self.config.detect_hardware()
            allocation = self.config.calculate_resource_allocation(hardware, self.arena_config.num_workers)
            self.config._resource_allocation = allocation
        
        logger.info(f"[ARENA] Starting parallel arena with {self.arena_config.num_workers} workers")
        logger.debug(f"[ARENA] Resource allocation: {allocation}")
        
        # Set up multiprocessing start method
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            # Already set, that's fine
            pass
        
        # Check model types
        is_random1 = isinstance(model1, RandomEvaluator)
        is_random2 = isinstance(model2, RandomEvaluator)
        
        # Create GPU evaluation services for non-random models
        gpu_service1 = None
        gpu_service2 = None
        
        try:
            # Service for model1 if it's not random
            if not is_random1:
                # Extract actual model
                if isinstance(model1, Evaluator):
                    actual_model1 = model1.model
                else:
                    actual_model1 = model1
                
                gpu_service1 = GPUEvaluatorService(
                    model=actual_model1,
                    device=self.config.mcts.device,
                    batch_size=256,
                    batch_timeout=0.01
                )
                gpu_service1.start()
                logger.debug(f"[ARENA] GPU evaluation service 1 started")
            
            # Service for model2 if it's not random
            if not is_random2:
                # Extract actual model
                if isinstance(model2, Evaluator):
                    actual_model2 = model2.model
                else:
                    actual_model2 = model2
                
                gpu_service2 = GPUEvaluatorService(
                    model=actual_model2,
                    device=self.config.mcts.device,
                    batch_size=256,
                    batch_timeout=0.01
                )
                gpu_service2.start()
                logger.debug(f"[ARENA] GPU evaluation service 2 started")
            
            # Get action space size
            initial_state = self.game_interface.create_initial_state()
            action_size = self.game_interface.get_action_space_size(initial_state)
            
            # Use Process directly for better control
            processes = []
            result_queues = []
            
            # Use hardware-aware concurrent worker limit
            max_concurrent = allocation['max_concurrent_workers']
            games_per_batch = min(max_concurrent, self.arena_config.num_workers)
            
            logger.debug(f"[ARENA] Processing games in batches of {games_per_batch}")
            
            # Progress bar setup
            disable_progress = silent or logger.level > logging.INFO or not self.arena_config.use_progress_bar
            
            with tqdm(total=num_games, desc="Arena games", unit="game",
                     disable=disable_progress, position=4, leave=False) as pbar:
                
                for batch_start in range(0, num_games, games_per_batch):
                    batch_end = min(batch_start + games_per_batch, num_games)
                    batch_processes = []
                    batch_queues = []
                    
                    # Start processes for this batch
                    for game_idx in range(batch_start, batch_end):
                        # Create result queue for this game
                        result_queue = mp.Queue()
                        result_queues.append(result_queue)
                        batch_queues.append(result_queue)
                        
                        # Determine which model plays first
                        if game_idx % 2 == 0:
                            # Model1 plays first
                            request_queue1 = gpu_service1.get_request_queue() if gpu_service1 else None
                            response_queue1 = gpu_service1.create_worker_queue(game_idx) if gpu_service1 else None
                            request_queue2 = gpu_service2.get_request_queue() if gpu_service2 else None
                            response_queue2 = gpu_service2.create_worker_queue(game_idx) if gpu_service2 else None
                            
                            p = mp.Process(
                                target=_play_arena_game_worker_with_gpu_service,
                                args=(self._serialize_config(self.config),
                                      self._serialize_arena_config(self.arena_config),
                                      request_queue1, response_queue1, is_random1,
                                      request_queue2, response_queue2, is_random2,
                                      action_size, game_idx, result_queue, allocation, False)
                            )
                        else:
                            # Model2 plays first (swap queues)
                            request_queue1 = gpu_service2.get_request_queue() if gpu_service2 else None
                            response_queue1 = gpu_service2.create_worker_queue(game_idx) if gpu_service2 else None
                            request_queue2 = gpu_service1.get_request_queue() if gpu_service1 else None
                            response_queue2 = gpu_service1.create_worker_queue(game_idx) if gpu_service1 else None
                            
                            p = mp.Process(
                                target=_play_arena_game_worker_with_gpu_service,
                                args=(self._serialize_config(self.config),
                                      self._serialize_arena_config(self.arena_config),
                                      request_queue1, response_queue1, is_random2,
                                      request_queue2, response_queue2, is_random1,
                                      action_size, game_idx, result_queue, allocation, True)
                            )
                        
                        p.start()
                        processes.append(p)
                        batch_processes.append(p)
                        logger.debug(f"[ARENA] Started process for game {game_idx}")
                    
                    # Collect results from this batch
                    logger.debug(f"[ARENA] Collecting results from batch of {len(batch_processes)} games...")
                    
                    for idx, (p, q) in enumerate(zip(batch_processes, batch_queues)):
                        game_idx = batch_start + idx
                        try:
                            # Wait for result
                            result = q.get(timeout=self.arena_config.timeout_seconds)
                            
                            # Handle inverted results for odd games
                            if game_idx % 2 == 1:
                                result = -result
                            
                            if result > 0:
                                wins += 1
                            elif result < 0:
                                losses += 1
                            else:
                                draws += 1
                            
                            logger.debug(f"[ARENA] Game {game_idx} completed with result: {result}")
                            
                        except queue.Empty:
                            logger.error(f"[ARENA] Timeout waiting for result from game {game_idx}")
                            draws += 1  # Count timeout as draw
                            
                        except Exception as e:
                            logger.error(f"[ARENA] Failed to get result from game {game_idx}: {e}")
                            draws += 1
                            
                        finally:
                            # Ensure process is terminated
                            if p.is_alive():
                                p.terminate()
                            p.join(timeout=5)
                            pbar.update(1)
                            pbar.set_postfix({
                                'W': wins,
                                'D': draws,
                                'L': losses,
                                'WR': f'{wins/(wins+draws+losses):.1%}'
                            })
                            
        finally:
            # Stop GPU services
            if gpu_service1:
                gpu_service1.stop()
                logger.debug("[ARENA] GPU evaluation service 1 stopped")
            if gpu_service2:
                gpu_service2.stop()
                logger.debug("[ARENA] GPU evaluation service 2 stopped")
            
            # Clean up GPU memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return wins, draws, losses
    
    def _play_single_game(self, evaluator1: Any, evaluator2: Any,
                         game_idx: int) -> int:
        """Play a single arena game
        
        Returns:
            1 if player 1 wins, -1 if player 2 wins, 0 for draw
        """
        # Enhanced memory monitoring
        if self.arena_config.memory_monitoring and (game_idx % 5 == 0 or game_idx < 3):
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            logger.debug(f"[ARENA] Game {game_idx} START - Process Memory: RSS={mem_info.rss/1024/1024:.1f}MB, VMS={mem_info.vms/1024/1024:.1f}MB")
            
            # Check GPU memory if CUDA available
            if torch.cuda.is_available():
                gpu_allocated = torch.cuda.memory_allocated()/1024/1024/1024
                gpu_reserved = torch.cuda.memory_reserved()/1024/1024/1024
                gpu_cached = torch.cuda.memory_reserved()/1024/1024/1024
                logger.debug(f"[ARENA] Game {game_idx} START - GPU Memory: Allocated={gpu_allocated:.3f}GB, Reserved={gpu_reserved:.3f}GB, Cached={gpu_cached:.3f}GB")
                
                # Log GPU memory summary
                if game_idx < 3:
                    logger.debug(f"[ARENA] GPU Memory Summary:\n{torch.cuda.memory_summary()}")
                
                # Check if we're approaching memory limit
                if gpu_allocated > self.arena_config.max_memory_gb:
                    logger.warning(f"[ARENA] GPU memory usage ({gpu_allocated:.2f}GB) exceeds limit ({self.arena_config.max_memory_gb}GB)")
                    # Force cleanup
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Log memory after cleanup
                    gpu_allocated_after = torch.cuda.memory_allocated()/1024/1024/1024
                    logger.info(f"[ARENA] After cleanup - GPU Allocated: {gpu_allocated_after:.3f}GB")
        
        try:
            # Create MCTS instances
            start_time = time.time()
            mcts1 = self._create_mcts(evaluator1)
            mcts2 = self._create_mcts(evaluator2)
            mcts_creation_time = time.time() - start_time
            
            if game_idx % 10 == 0:
                logger.debug(f"[ARENA] Game {game_idx} - MCTS instances created in {mcts_creation_time:.3f}s")
                
                # Log MCTS memory usage
                if hasattr(mcts1, 'get_memory_usage'):
                    mcts1_mem = mcts1.get_memory_usage()
                    mcts2_mem = mcts2.get_memory_usage()
                    logger.debug(f"[ARENA] MCTS memory: Player1={mcts1_mem/1024/1024:.1f}MB, Player2={mcts2_mem/1024/1024:.1f}MB")
        
            # Play game
            state = self.game_interface.create_initial_state()
            current_player = 1
            move_times = []
            
            for move_num in range(self.config.training.max_moves_per_game):
                move_start = time.time()
                
                # Periodic memory check during game
                if self.arena_config.memory_monitoring and move_num > 0 and move_num % 20 == 0:
                    if torch.cuda.is_available():
                        gpu_allocated = torch.cuda.memory_allocated()/1024/1024/1024
                        logger.debug(f"[ARENA] Game {game_idx}, Move {move_num} - GPU Allocated: {gpu_allocated:.3f}GB")
                    
                # Get current MCTS
                current_mcts = mcts1 if current_player == 1 else mcts2
                
                # Get action
                if move_num < self.arena_config.temperature_threshold:
                    temp = self.arena_config.temperature
                else:
                    temp = 0.0
                
                # Clear MCTS tree before search to ensure fresh state
                if hasattr(current_mcts, 'reset_tree'):
                    current_mcts.reset_tree()
                elif hasattr(current_mcts, 'clear'):
                    current_mcts.clear()
                
                # Run MCTS search
                with torch.no_grad():  # Ensure no gradients are tracked
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
                
                # Track move time
                move_time = time.time() - move_start
                move_times.append(move_time)
                
                # Check terminal
                if self.game_interface.is_terminal(state):
                    # Return from player 1's perspective
                    value = self.game_interface.get_value(state)
                    if current_player == 2:
                        value = -value
                    result = int(np.sign(value))
                    
                    # Log game summary
                    if game_idx % 10 == 0:
                        avg_move_time = np.mean(move_times)
                        total_time = sum(move_times)
                        logger.debug(f"[ARENA] Game {game_idx} completed - Moves: {move_num+1}, Total time: {total_time:.2f}s, Avg move: {avg_move_time:.3f}s")
                    
                    return result
                
                # Switch player
                current_player = 3 - current_player
            
            # Draw if max moves reached
            if game_idx % 10 == 0:
                logger.debug(f"[ARENA] Game {game_idx} ended in draw after {len(move_times)} moves")
            return 0
            
        except Exception as e:
            logger.error(f"[ARENA] Game {game_idx} failed with error: {e}")
            logger.error(f"[ARENA] Full traceback:\n{traceback.format_exc()}")
            
            # Log memory state on error
            if torch.cuda.is_available():
                try:
                    gpu_allocated = torch.cuda.memory_allocated()/1024/1024/1024
                    logger.error(f"[ARENA] GPU memory at error: {gpu_allocated:.3f}GB")
                except:
                    logger.error("[ARENA] Could not query GPU memory due to CUDA error")
            
            # For CUDA errors, attempt recovery
            if "CUDA error" in str(e):
                logger.warning(f"[ARENA] Attempting CUDA error recovery for game {game_idx}")
                try:
                    # Synchronize to ensure all operations complete
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                except:
                    pass
                
                # Force cleanup of MCTS instances
                if 'mcts1' in locals():
                    try:
                        if hasattr(mcts1, 'clear_tree'):
                            mcts1.clear_tree()
                        del mcts1
                    except:
                        pass
                        
                if 'mcts2' in locals():
                    try:
                        if hasattr(mcts2, 'clear_tree'):
                            mcts2.clear_tree()
                        del mcts2
                    except:
                        pass
                
                # Clear CUDA cache
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except:
                    pass
                
                # Return draw on CUDA error to continue tournament
                logger.warning(f"[ARENA] Treating game {game_idx} as draw due to CUDA error")
                return 0
            
            raise
        finally:
            # Explicit cleanup to prevent memory leaks
            cuda_available = False
            try:
                cuda_available = torch.cuda.is_available()
            except:
                # CUDA may be in error state
                pass
                
            if self.arena_config.memory_monitoring and game_idx % 5 == 0:
                logger.debug(f"[ARENA] Game {game_idx} - Starting cleanup")
                
                # Log memory before cleanup
                gpu_before = 0
                if cuda_available:
                    try:
                        gpu_before = torch.cuda.memory_allocated()/1024/1024/1024
                    except:
                        pass
            
            # Clear MCTS trees before deletion if they exist
            if 'mcts1' in locals():
                try:
                    if hasattr(mcts1, 'clear_tree'):
                        mcts1.clear_tree()
                    elif hasattr(mcts1, 'reset_tree'):
                        mcts1.reset_tree()
                except Exception as e:
                    logger.warning(f"Error during MCTS cleanup: {e}")
                del mcts1
            if 'mcts2' in locals():
                try:
                    if hasattr(mcts2, 'clear_tree'):
                        mcts2.clear_tree()
                    elif hasattr(mcts2, 'reset_tree'):
                        mcts2.reset_tree()
                except Exception as e:
                    logger.warning(f"Error during MCTS cleanup: {e}")
                del mcts2
            
            # Force garbage collection more aggressively
            if game_idx % max(1, self.arena_config.gc_frequency // 2) == 0:
                gc.collect()
                if cuda_available:
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # Ensure all CUDA operations complete
                        
                        if self.arena_config.memory_monitoring and game_idx % 5 == 0:
                            gpu_after = torch.cuda.memory_allocated()/1024/1024/1024
                            logger.debug(f"[ARENA] Game {game_idx} - Cleanup complete. GPU memory: {gpu_before:.3f}GB -> {gpu_after:.3f}GB")
                    except Exception as e:
                        logger.warning(f"[ARENA] Could not clear CUDA cache during cleanup")
                else:
                    if self.arena_config.memory_monitoring and game_idx % 5 == 0:
                        logger.debug(f"[ARENA] Game {game_idx} - Cleanup complete (CPU only)")
    
    def _create_mcts(self, evaluator: Any):
        """Create MCTS instance for arena"""
        from mcts.core.mcts import MCTS, MCTSConfig
        
        # Use same settings as self-play for consistency
        mcts_config = MCTSConfig(
            # Core MCTS parameters
            num_simulations=self.arena_config.mcts_simulations,
            c_puct=self.arena_config.c_puct,
            temperature=0.0,  # Arena always uses deterministic play
            
            # Performance and wave parameters - same as self-play
            min_wave_size=self.config.mcts.min_wave_size,
            max_wave_size=self.config.mcts.max_wave_size,
            adaptive_wave_sizing=self.config.mcts.adaptive_wave_sizing,
            
            # Memory and optimization - optimized for arena performance
            # Arena needs balanced memory allocation for good performance
            memory_pool_size_mb=min(768, self.config.mcts.memory_pool_size_mb // 3),
            # Tree nodes - arena games are typically shorter, so we can use fewer nodes
            max_tree_nodes=min(25000, self.config.mcts.max_tree_nodes // 4),
            use_mixed_precision=self.config.mcts.use_mixed_precision,
            use_cuda_graphs=False,  # Disable CUDA graphs to save memory in arena
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




def _play_arena_game_worker_with_gpu_service(config_dict: Dict, arena_config_dict: Dict,
                                            request_queue1, response_queue1, is_random1: bool,
                                            request_queue2, response_queue2, is_random2: bool,
                                            action_size: int, game_idx: int, result_queue,
                                            allocation: Dict, invert_result: bool) -> None:
    """Worker function for parallel arena games using GPU evaluation service"""
    # CRITICAL: Disable CUDA before ANY imports that might load torch
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    # Now safe to import worker init
    from mcts.utils.worker_init import init_worker_process, verify_cuda_disabled
    init_worker_process()
    
    try:
        # Now safe to import torch and other modules
        import torch
        
        # Verify CUDA is properly disabled
        verify_cuda_disabled()
        
        from mcts.core.game_interface import GameInterface, GameType
        from mcts.core.mcts import MCTS, MCTSConfig
        from mcts.gpu.gpu_game_states import GameType as GPUGameType
        from mcts.utils.gpu_evaluator_service import RemoteEvaluator
        from mcts.core.evaluator import RandomEvaluator, EvaluatorConfig
        import numpy as np
        
        # Set up logging
        logging.basicConfig(level=getattr(logging, config_dict['log_level']))
        logger = logging.getLogger(__name__)
        
        logger.debug(f"[ARENA WORKER {game_idx}] Worker started - using GPU service for evaluations")
        logger.debug(f"[ARENA WORKER {game_idx}] CUDA available: {torch.cuda.is_available()}")
        
        # Create game interface
        game_type = GameType[config_dict['game']['game_type'].upper()]
        input_representation = config_dict.get('network', {}).get('input_representation', 'basic')
        game_interface = GameInterface(game_type, config_dict['game']['board_size'], input_representation=input_representation)
        
        # Create evaluators
        if is_random1:
            evaluator1 = RandomEvaluator(
                config=EvaluatorConfig(device='cpu'),  # Random evaluator always on CPU
                action_size=action_size
            )
        else:
            # Use remote evaluator for neural network
            evaluator1 = RemoteEvaluator(request_queue1, response_queue1, action_size, worker_id=game_idx)
        
        if is_random2:
            evaluator2 = RandomEvaluator(
                config=EvaluatorConfig(device='cpu'),  # Random evaluator always on CPU
                action_size=action_size
            )
        else:
            # Use remote evaluator for neural network
            evaluator2 = RemoteEvaluator(request_queue2, response_queue2, action_size, worker_id=game_idx)
        
        # Wrap evaluators to return torch tensors if GPU is available
        class TensorEvaluator:
            def __init__(self, evaluator, device):
                self.evaluator = evaluator
                self.device = device
                self._return_torch_tensors = True
            
            def evaluate(self, state, legal_mask=None, temperature=1.0):
                policy, value = self.evaluator.evaluate(state, legal_mask, temperature)
                # Convert numpy to torch tensors
                policy_tensor = torch.from_numpy(policy).float().to(self.device)
                value_tensor = torch.tensor(value, dtype=torch.float32, device=self.device)
                return policy_tensor, value_tensor
            
            def evaluate_batch(self, states, legal_masks=None, temperature=1.0):
                # Convert torch tensors to numpy for the remote evaluator
                if isinstance(states, torch.Tensor):
                    states = states.cpu().numpy()
                if legal_masks is not None and isinstance(legal_masks, torch.Tensor):
                    legal_masks = legal_masks.cpu().numpy()
                
                policies, values = self.evaluator.evaluate_batch(states, legal_masks, temperature)
                
                # Convert numpy to torch tensors
                policies_tensor = torch.from_numpy(policies).float().to(self.device)
                values_tensor = torch.from_numpy(values).float().to(self.device)
                return policies_tensor, values_tensor
        
        # CRITICAL: Workers must use CPU for tensors to avoid CUDA multiprocessing issues
        tensor_device = 'cpu'  # Always use CPU in workers
        
        # Wrap evaluators if they're remote evaluators (not random)
        if not is_random1 and hasattr(evaluator1, 'request_queue'):
            evaluator1 = TensorEvaluator(evaluator1, tensor_device)
        if not is_random2 and hasattr(evaluator2, 'request_queue'):
            evaluator2 = TensorEvaluator(evaluator2, tensor_device)
        
        logger.debug(f"[ARENA WORKER {game_idx}] Created evaluators on device: {tensor_device}")
        
        # Create MCTS instances with optimal settings
        mcts_config = MCTSConfig(
            # Core MCTS parameters
            num_simulations=arena_config_dict['mcts_simulations'],
            c_puct=arena_config_dict['c_puct'],
            temperature=0.0,  # Arena always uses deterministic play
            
            # Performance and wave parameters
            min_wave_size=config_dict['mcts']['min_wave_size'],
            max_wave_size=config_dict['mcts']['max_wave_size'],
            adaptive_wave_sizing=config_dict['mcts']['adaptive_wave_sizing'],
            
            # Dynamic memory pool based on hardware allocation
            memory_pool_size_mb=min(allocation.get('memory_per_worker_mb', 512) // 2, 512),
            use_mixed_precision=False,  # Disabled for CPU workers
            use_cuda_graphs=False,  # Disabled for CPU workers
            use_tensor_cores=False,  # Disabled for CPU workers
            max_tree_nodes=min(50000, config_dict['mcts']['max_tree_nodes'] // allocation.get('num_workers', 4)),
            
            # Game and device configuration
            device='cpu',  # Workers must use CPU
            game_type=GPUGameType[game_type.name],
            board_size=config_dict['game']['board_size'],
            
            # Disable virtual loss for arena
            enable_virtual_loss=False,
            virtual_loss=0.0,
            
            # Debug options
            enable_debug_logging=False,
            profile_gpu_kernels=False
        )
        
        mcts1 = MCTS(mcts_config, evaluator1)
        mcts2 = MCTS(mcts_config, evaluator2)
        
        logger.debug(f"[ARENA WORKER {game_idx}] MCTS configured - device: {mcts_config.device}, memory: {mcts_config.memory_pool_size_mb}MB")
        
        # Optimize MCTS for hardware
        if hasattr(mcts1, 'optimize_for_hardware'):
            try:
                mcts1.optimize_for_hardware()
                mcts2.optimize_for_hardware()
                logger.debug(f"[ARENA WORKER {game_idx}] MCTS optimized for hardware")
            except Exception as e:
                logger.warning(f"[ARENA WORKER {game_idx}] Failed to optimize MCTS: {e}")
        
        logger.debug(f"[ARENA WORKER {game_idx}] MCTS configured - device: {mcts_config.device}, memory: {mcts_config.memory_pool_size_mb}MB")
        
        # Play game
        state = game_interface.create_initial_state()
        current_player = 1
        
        for move_num in range(config_dict['training']['max_moves_per_game']):
            # Get current MCTS
            current_mcts = mcts1 if current_player == 1 else mcts2
            
            # Run MCTS search
            with torch.no_grad():  # Ensure no gradients are tracked
                policy = current_mcts.search(state, num_simulations=arena_config_dict['mcts_simulations'])
            
            # Get best action (deterministic)
            action = current_mcts.get_best_action(state)
            
            # Apply action
            state = game_interface.get_next_state(state, action)
            
            # No tree reuse in arena to save memory
            mcts1.reset_tree()
            mcts2.reset_tree()
            
            # Check terminal
            if game_interface.is_terminal(state):
                # Return from player 1's perspective
                value = game_interface.get_value(state)
                if current_player == 2:
                    value = -value
                result = int(np.sign(value))
                break
            
            # Switch player
            current_player = 3 - current_player
        else:
            # Draw if max moves reached
            result = 0
        
        logger.debug(f"[ARENA WORKER {game_idx}] Game completed with result: {result}")
        
        # Clean up GPU memory before returning
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Put result in queue
        result_queue.put(result)
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"[ARENA WORKER {game_idx}] Failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # Put draw result on error
        result_queue.put(0)