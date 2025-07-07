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
        
        # Uncertainty tracking
        self.rating_uncertainty: Dict[str, float] = {}  # Rating standard deviation
        self.game_counts: Dict[str, int] = {}  # Number of games played
        
        # Rating anchoring and deflation
        self.anchor_players: Dict[str, float] = {"random": 0.0}  # Fixed anchor points
        self.rating_sum_target = 0.0  # Target sum of all ratings
        self.deflation_factor = 0.99  # Slight deflation per update
        self.use_deflation = True  # Enable rating deflation
        
        # Performance validation
        self.validation_history: List[Dict] = []  # Track actual vs expected performance
    
    def get_adaptive_k_factor(self, player: str, opponent: str, base_k: float = None) -> float:
        """Calculate adaptive K-factor based on player strength, uncertainty, and game count"""
        if base_k is None:
            base_k = self.k_factor
            
        # Get player stats
        player_elo = self.ratings.get(player, self.initial_rating)
        player_games = self.game_counts.get(player, 0)
        player_uncertainty = self.rating_uncertainty.get(player, 350.0)  # Initial uncertainty
        
        # Base K-factor adjustment based on game count (confidence)
        # More games = more confidence = lower K-factor
        if player_games < 10:
            confidence_multiplier = 1.0  # Full K for new players
        elif player_games < 30:
            confidence_multiplier = 0.8  # 80% K after initial games
        elif player_games < 100:
            confidence_multiplier = 0.6  # 60% K for established players
        else:
            confidence_multiplier = 0.4  # 40% K for well-established players
            
        # Uncertainty-based adjustment
        # Higher uncertainty = higher K-factor to allow faster convergence
        # For standard case (uncertainty=350), multiplier should be 1.0
        uncertainty_multiplier = min(1.0, player_uncertainty / 350.0)
        
        # Special handling for matches against random
        if opponent == "random":
            # More aggressive K-factor reduction for random matches
            
            if player_elo < 200:
                # Early phase: Moderate K-factor (not full to prevent early inflation)
                k_multiplier = 0.8
            elif player_elo < 500:
                # Calibration phase: Quick reduction
                k_multiplier = 0.6 - 0.3 * (player_elo - 200) / 300
            elif player_elo < 1000:
                # Stabilization phase: Strong reduction
                k_multiplier = 0.3 - 0.2 * (player_elo - 500) / 500
            else:
                # Mature phase: Minimal gains from beating random
                # Exponential decay to prevent any inflation
                k_multiplier = 0.1 * np.exp(-player_elo / 1000)
            
            # Apply win rate adjustment
            recent_vs_random = self.get_recent_performance(player, "random", last_n=5)
            if recent_vs_random is not None:
                if recent_vs_random > 0.99:
                    # Nearly perfect: extreme K reduction
                    k_multiplier *= 0.1
                elif recent_vs_random > 0.95:
                    # Very strong: heavy K reduction
                    k_multiplier *= 0.3
                elif recent_vs_random > 0.90:
                    # Strong: moderate K reduction
                    k_multiplier *= 0.5
            
            # Apply all multipliers
            return base_k * k_multiplier * confidence_multiplier * min(1.0, uncertainty_multiplier)
        
        # For non-random opponents
        # Consider rating difference for K-factor adjustment
        opponent_elo = self.ratings.get(opponent, self.initial_rating)
        rating_diff = abs(player_elo - opponent_elo)
        
        # Reduce K-factor for mismatched games (prevents inflation from beating weak opponents)
        if rating_diff > 400:
            mismatch_multiplier = 0.5
        elif rating_diff > 200:
            mismatch_multiplier = 0.7
        else:
            mismatch_multiplier = 1.0
            
        # High-rating adjustment
        if player_elo > 2000:
            rating_multiplier = 0.7
        elif player_elo > 2500:
            rating_multiplier = 0.5
        else:
            rating_multiplier = 1.0
            
        # Combine all factors
        return base_k * confidence_multiplier * uncertainty_multiplier * mismatch_multiplier * rating_multiplier
    
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
                      wins: int, draws: int, losses: int,
                      protect_best_elo: bool = False, best_player: Optional[str] = None):
        """Update ELO ratings based on match results with adaptive K-factor, uncertainty tracking, and deflation
        
        Args:
            player1: First player name
            player2: Second player name
            wins: Number of wins for player1
            draws: Number of draws
            losses: Number of losses for player1
            protect_best_elo: If True, prevent best model's ELO from decreasing when it wins
            best_player: The player identifier of the current best model
        """
        total_games = wins + draws + losses
        if total_games == 0:
            return
        
        # Initialize players if new (MUST be done first)
        if player1 not in self.ratings:
            # Use anchor rating if this is an anchor player
            if player1 in self.anchor_players:
                self.ratings[player1] = self.anchor_players[player1]
            else:
                self.ratings[player1] = self.initial_rating
            self.rating_uncertainty[player1] = 350.0  # Initial uncertainty (Glicko-like)
            self.game_counts[player1] = 0
            
        if player2 not in self.ratings:
            # Use anchor rating if this is an anchor player
            if player2 in self.anchor_players:
                self.ratings[player2] = self.anchor_players[player2]
            else:
                self.ratings[player2] = self.initial_rating
            self.rating_uncertainty[player2] = 350.0
            self.game_counts[player2] = 0
        
        # Get current ratings
        r1 = self.ratings[player1]
        r2 = self.ratings[player2]
        
        # Get adaptive K-factors for both players
        k1 = self.get_adaptive_k_factor(player1, player2)
        k2 = self.get_adaptive_k_factor(player2, player1)
        
        # Calculate expected scores
        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))
        
        # Actual scores
        s1 = (wins + 0.5 * draws) / total_games
        s2 = (losses + 0.5 * draws) / total_games
        
        # Track validation: actual vs expected performance
        self.validation_history.append({
            "timestamp": datetime.now().isoformat(),
            "player1": player1,
            "player2": player2,
            "expected_score1": e1,
            "actual_score1": s1,
            "expected_score2": e2,
            "actual_score2": s2,
            "games": total_games,
            "rating_diff": r1 - r2
        })
        
        # Calculate rating changes with adaptive K-factors
        delta1 = k1 * (s1 - e1)
        delta2 = k2 * (s2 - e2)
        
        # Update game counts (only for non-anchor players)
        if player1 not in self.anchor_players:
            if player1 not in self.game_counts:
                self.game_counts[player1] = 0
            self.game_counts[player1] += total_games
        if player2 not in self.anchor_players:
            if player2 not in self.game_counts:
                self.game_counts[player2] = 0
            self.game_counts[player2] += total_games
        
        # Update uncertainty (decreases with more games, increases with surprising results)
        if player1 not in self.anchor_players:
            # Ensure player exists in rating_uncertainty
            if player1 not in self.rating_uncertainty:
                self.rating_uncertainty[player1] = 350.0
            # Reduce uncertainty based on games played
            uncertainty_decay = 0.98
            self.rating_uncertainty[player1] *= uncertainty_decay
            
            # Increase uncertainty if result was surprising
            surprise_factor = abs(s1 - e1)
            if surprise_factor > 0.3:  # Surprising result
                self.rating_uncertainty[player1] *= (1 + surprise_factor * 0.2)
                
            # Clamp uncertainty to reasonable range
            self.rating_uncertainty[player1] = np.clip(self.rating_uncertainty[player1], 50.0, 350.0)
            
        if player2 not in self.anchor_players:
            # Ensure player exists in rating_uncertainty
            if player2 not in self.rating_uncertainty:
                self.rating_uncertainty[player2] = 350.0
            uncertainty_decay = 0.98
            self.rating_uncertainty[player2] *= uncertainty_decay
            
            surprise_factor = abs(s2 - e2)
            if surprise_factor > 0.3:
                self.rating_uncertainty[player2] *= (1 + surprise_factor * 0.2)
                
            self.rating_uncertainty[player2] = np.clip(self.rating_uncertainty[player2], 50.0, 350.0)
        
        # Store old ratings for deflation calculation
        old_ratings_sum = sum(self.ratings.values())
        
        # Update ratings (but keep anchor players fixed)
        if player1 not in self.anchor_players:
            new_r1 = r1 + delta1
            
            # CRITICAL: Protect best model's ELO from decreasing when it wins
            if protect_best_elo and best_player and player1 == best_player:
                # Check if best model won (positive score means it won)
                if s1 > 0.5:  # Best model won overall
                    # Never let best model's ELO decrease when it wins
                    if new_r1 < r1:
                        logger.debug(f"Protecting best model {player1} ELO: would decrease {r1:.1f} -> {new_r1:.1f}, keeping at {r1:.1f}")
                        new_r1 = r1
            
            self.ratings[player1] = new_r1
            
            # Track iteration ELO if player is an iteration model
            if player1.startswith("iter_"):
                try:
                    iter_num = int(player1.split("_")[1])
                    self.iteration_elos[iter_num] = new_r1
                except (ValueError, IndexError):
                    pass
            
            # Log with detailed info
            logger.debug(f"ELO Update: {player1} {r1:.1f} -> {new_r1:.1f} (Δ{delta1:+.1f}, K={k1:.1f}, σ={self.rating_uncertainty[player1]:.1f}) after {wins}W-{draws}D-{losses}L vs {player2}")
        else:
            new_r1 = r1
            
        if player2 not in self.anchor_players:
            new_r2 = r2 + delta2
            
            # CRITICAL: Protect best model's ELO from decreasing when it wins
            if protect_best_elo and best_player and player2 == best_player:
                # Check if best model won (negative score for player2 means it won)
                if s2 > 0.5:  # Best model won overall
                    # Never let best model's ELO decrease when it wins
                    if new_r2 < r2:
                        logger.debug(f"Protecting best model {player2} ELO: would decrease {r2:.1f} -> {new_r2:.1f}, keeping at {r2:.1f}")
                        new_r2 = r2
            
            self.ratings[player2] = new_r2
            
            # Track iteration ELO if player is an iteration model
            if player2.startswith("iter_"):
                try:
                    iter_num = int(player2.split("_")[1])
                    self.iteration_elos[iter_num] = new_r2
                except (ValueError, IndexError):
                    pass
            
            logger.debug(f"ELO Update: {player2} {r2:.1f} -> {new_r2:.1f} (Δ{delta2:+.1f}, K={k2:.1f}, σ={self.rating_uncertainty[player2]:.1f}) after {losses}W-{draws}D-{wins}L vs {player1}")
        else:
            new_r2 = r2
        
        # Apply rating deflation to prevent drift
        if self.use_deflation and len(self.ratings) > 2:
            self._apply_rating_deflation(old_ratings_sum)
        
        # Record history with enhanced information
        self.game_history.append({
            "timestamp": datetime.now().isoformat(),
            "player1": player1,
            "player2": player2,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "old_rating1": r1,
            "old_rating2": r2,
            "new_rating1": self.ratings.get(player1, new_r1),
            "new_rating2": self.ratings.get(player2, new_r2),
            "k_factor1": k1,
            "k_factor2": k2,
            "uncertainty1": self.rating_uncertainty.get(player1, 0),
            "uncertainty2": self.rating_uncertainty.get(player2, 0),
            "game_count1": self.game_counts.get(player1, 0),
            "game_count2": self.game_counts.get(player2, 0)
        })
    
    def _apply_rating_deflation(self, old_ratings_sum: float):
        """Apply rating deflation to prevent overall rating drift"""
        # Calculate current sum of ratings
        current_sum = sum(self.ratings.values())
        
        # Calculate target sum (should stay close to initial sum)
        # Allow slight growth based on number of non-anchor players
        non_anchor_count = len([p for p in self.ratings if p not in self.anchor_players])
        target_sum = self.rating_sum_target or (non_anchor_count * self.initial_rating)
        
        # Apply deflation if ratings are inflating
        if current_sum > target_sum * 1.05:  # 5% tolerance
            # Calculate deflation factor
            deflation_ratio = target_sum / current_sum
            
            # Apply deflation to all non-anchor players
            for player in self.ratings:
                if player not in self.anchor_players:
                    old_rating = self.ratings[player]
                    # Deflate towards initial rating
                    deflated_rating = self.initial_rating + (old_rating - self.initial_rating) * deflation_ratio
                    self.ratings[player] = deflated_rating
                    
                    # Update iteration ELO if applicable
                    if player.startswith("iter_"):
                        try:
                            iter_num = int(player.split("_")[1])
                            self.iteration_elos[iter_num] = deflated_rating
                        except (ValueError, IndexError):
                            pass
                    
                    if abs(old_rating - deflated_rating) > 1.0:
                        logger.debug(f"Deflation applied to {player}: {old_rating:.1f} -> {deflated_rating:.1f}")
    
    def get_rating(self, player: str) -> float:
        """Get current rating for a player"""
        return self.ratings.get(player, self.initial_rating)
    
    def get_rating_with_uncertainty(self, player: str) -> Tuple[float, float]:
        """Get rating with uncertainty bounds (rating ± 2*std_dev)"""
        rating = self.ratings.get(player, self.initial_rating)
        uncertainty = self.rating_uncertainty.get(player, 350.0)
        return rating, uncertainty
    
    def get_confidence_interval(self, player: str, confidence: float = 0.95) -> Tuple[float, float]:
        """Get confidence interval for player rating"""
        rating, uncertainty = self.get_rating_with_uncertainty(player)
        # Using 2 standard deviations for ~95% confidence
        z_score = 2.0 if confidence >= 0.95 else 1.0
        lower = rating - z_score * uncertainty
        upper = rating + z_score * uncertainty
        return lower, upper
    
    def get_leaderboard(self) -> List[Tuple[str, float]]:
        """Get sorted leaderboard"""
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
    
    def get_detailed_leaderboard(self, max_iterations: int = 10) -> List[Dict]:
        """Get detailed leaderboard with uncertainty and game counts
        
        Args:
            max_iterations: Maximum number of recent iterations to show (0 = all)
        """
        leaderboard = []
        
        # Get current highest iteration number to determine recent iterations
        max_iter = 0
        for player in self.ratings.keys():
            if player.startswith("iter_"):
                try:
                    iter_num = int(player.split("_")[1])
                    max_iter = max(max_iter, iter_num)
                except (ValueError, IndexError):
                    pass
        
        # Determine which iterations to include
        if max_iterations > 0 and max_iter > max_iterations:
            min_iter_to_show = max_iter - max_iterations + 1
        else:
            min_iter_to_show = 0
        
        for player, rating in self.ratings.items():
            # Filter out old iterations if requested
            if max_iterations > 0 and player.startswith("iter_"):
                try:
                    iter_num = int(player.split("_")[1])
                    if iter_num < min_iter_to_show:
                        continue  # Skip old iterations
                except (ValueError, IndexError):
                    pass  # Keep non-standard iter names
            
            uncertainty = self.rating_uncertainty.get(player, 0)
            games = self.game_counts.get(player, 0)
            lower, upper = self.get_confidence_interval(player)
            
            leaderboard.append({
                "player": player,
                "rating": rating,
                "uncertainty": uncertainty,
                "games": games,
                "confidence_lower": lower,
                "confidence_upper": upper,
                "is_anchor": player in self.anchor_players
            })
        
        # Sort by rating
        leaderboard.sort(key=lambda x: x["rating"], reverse=True)
        return leaderboard
    
    def get_validation_metrics(self) -> Dict:
        """Calculate validation metrics to check if ELO predictions match actual results"""
        if not self.validation_history:
            return {"status": "no_data"}
        
        # Calculate prediction accuracy
        correct_predictions = 0
        total_predictions = 0
        squared_errors = []
        
        for record in self.validation_history:
            # Was the prediction correct? (considering draws)
            expected1 = record["expected_score1"]
            actual1 = record["actual_score1"]
            
            # Binary prediction accuracy
            if expected1 > 0.5 and actual1 > 0.5:
                correct_predictions += 1
            elif expected1 < 0.5 and actual1 < 0.5:
                correct_predictions += 1
            elif abs(expected1 - 0.5) < 0.1 and abs(actual1 - 0.5) < 0.1:
                correct_predictions += 1  # Both near draw
                
            total_predictions += 1
            
            # Calculate squared error
            error = (expected1 - actual1) ** 2
            squared_errors.append(error)
        
        # Calculate metrics
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        mse = np.mean(squared_errors) if squared_errors else 0
        rmse = np.sqrt(mse)
        
        # Check for systematic bias
        recent_records = self.validation_history[-20:]  # Last 20 matches
        recent_errors = [r["actual_score1"] - r["expected_score1"] for r in recent_records]
        bias = np.mean(recent_errors) if recent_errors else 0
        
        return {
            "prediction_accuracy": accuracy,
            "mse": mse,
            "rmse": rmse,
            "bias": bias,
            "total_validations": len(self.validation_history),
            "rating_inflation_detected": abs(bias) > 0.1,
            "recommendation": self._get_validation_recommendation(accuracy, rmse, bias)
        }
    
    def _get_validation_recommendation(self, accuracy: float, rmse: float, bias: float) -> str:
        """Get recommendation based on validation metrics"""
        if accuracy < 0.6:
            return "Poor prediction accuracy - ratings may not reflect true strength"
        elif rmse > 0.3:
            return "High prediction error - consider adjusting K-factor"
        elif abs(bias) > 0.1:
            if bias > 0:
                return "Systematic underestimation - ratings may be inflated"
            else:
                return "Systematic overestimation - ratings may be deflated"
        else:
            return "Ratings appear well-calibrated"
    
    def save_to_file(self, filepath: str):
        """Save ratings and history to JSON file"""
        data = {
            "ratings": dict(self.ratings),
            "rating_uncertainty": dict(self.rating_uncertainty),
            "game_counts": dict(self.game_counts),
            "history": self.game_history,
            "validation_history": self.validation_history,
            "k_factor": self.k_factor,
            "initial_rating": self.initial_rating,
            "anchor_players": dict(self.anchor_players),
            "use_deflation": self.use_deflation,
            "deflation_factor": self.deflation_factor,
            "iteration_elos": dict(self.iteration_elos)
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
        
        # Load new fields with defaults for backward compatibility
        self.rating_uncertainty = data.get("rating_uncertainty", {})
        self.game_counts = data.get("game_counts", {})
        self.validation_history = data.get("validation_history", [])
        self.anchor_players = data.get("anchor_players", {"random": 0.0})
        self.use_deflation = data.get("use_deflation", True)
        self.deflation_factor = data.get("deflation_factor", 0.99)
        # Convert string keys back to integers for iteration_elos
        iteration_elos_data = data.get("iteration_elos", {})
        self.iteration_elos = {int(k): v for k, v in iteration_elos_data.items()}
    
    def get_health_report(self) -> Dict:
        """Generate comprehensive health report of the rating system"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_players": len(self.ratings),
            "total_games": len(self.game_history),
            "rating_statistics": self._get_rating_statistics(),
            "validation_metrics": self.get_validation_metrics(),
            "inflation_indicators": self._get_inflation_indicators(),
            "recommendations": []
        }
        
        # Add recommendations based on analysis
        if report["inflation_indicators"].get("inflation_detected", False):
            report["recommendations"].append("Rating inflation detected - consider adjusting K-factors or deflation settings")
            
        validation = report["validation_metrics"]
        if validation.get("rating_inflation_detected", False):
            report["recommendations"].append(validation.get("recommendation", ""))
            
        if report["rating_statistics"]["avg_uncertainty"] > 200:
            report["recommendations"].append("High average uncertainty - more games needed for reliable ratings")
            
        return report
    
    def _get_rating_statistics(self) -> Dict:
        """Calculate rating distribution statistics"""
        if not self.ratings:
            return {"status": "no_data"}
            
        ratings = list(self.ratings.values())
        uncertainties = [self.rating_uncertainty.get(p, 0) for p in self.ratings]
        game_counts = [self.game_counts.get(p, 0) for p in self.ratings]
        
        return {
            "mean_rating": np.mean(ratings),
            "std_rating": np.std(ratings),
            "min_rating": np.min(ratings),
            "max_rating": np.max(ratings),
            "rating_range": np.max(ratings) - np.min(ratings),
            "avg_uncertainty": np.mean(uncertainties) if uncertainties else 0,
            "avg_games_played": np.mean(game_counts) if game_counts else 0,
            "rating_distribution": self._get_rating_distribution(ratings)
        }
    
    def _get_rating_distribution(self, ratings: List[float]) -> Dict:
        """Get rating distribution by ranges"""
        distribution = {
            "0-500": 0,
            "500-1000": 0,
            "1000-1500": 0,
            "1500-2000": 0,
            "2000-2500": 0,
            "2500+": 0
        }
        
        for rating in ratings:
            if rating < 500:
                distribution["0-500"] += 1
            elif rating < 1000:
                distribution["500-1000"] += 1
            elif rating < 1500:
                distribution["1000-1500"] += 1
            elif rating < 2000:
                distribution["1500-2000"] += 1
            elif rating < 2500:
                distribution["2000-2500"] += 1
            else:
                distribution["2500+"] += 1
                
        return distribution
    
    def _get_inflation_indicators(self) -> Dict:
        """Check for rating inflation indicators"""
        if len(self.iteration_elos) < 10:
            return {"status": "insufficient_data"}
            
        # Get ELO progression over iterations
        iterations = sorted(self.iteration_elos.keys())
        elos = [self.iteration_elos[i] for i in iterations]
        
        # Calculate growth rate
        if len(elos) >= 2:
            total_growth = elos[-1] - elos[0]
            avg_growth_per_iteration = total_growth / (iterations[-1] - iterations[0])
            
            # Check recent vs historical growth
            mid_point = len(elos) // 2
            early_growth = (elos[mid_point] - elos[0]) / (iterations[mid_point] - iterations[0])
            late_growth = (elos[-1] - elos[mid_point]) / (iterations[-1] - iterations[mid_point])
            
            # Detect acceleration in growth
            growth_acceleration = late_growth - early_growth
            
            return {
                "total_elo_growth": total_growth,
                "avg_growth_per_iteration": avg_growth_per_iteration,
                "early_growth_rate": early_growth,
                "late_growth_rate": late_growth,
                "growth_acceleration": growth_acceleration,
                "inflation_detected": growth_acceleration > 5.0 or avg_growth_per_iteration > 10.0,
                "current_iteration_elo": elos[-1] if elos else None
            }
        
        return {"status": "insufficient_data"}
    
    def cleanup_old_iterations(self, keep_recent: int = 15):
        """Remove old iteration entries to keep the ratings dictionary clean
        
        Args:
            keep_recent: Number of recent iterations to keep (default: 15)
        """
        # Find current highest iteration number
        max_iter = 0
        iteration_players = []
        
        for player in list(self.ratings.keys()):
            if player.startswith("iter_"):
                try:
                    iter_num = int(player.split("_")[1])
                    max_iter = max(max_iter, iter_num)
                    iteration_players.append((player, iter_num))
                except (ValueError, IndexError):
                    pass
        
        # Determine cutoff for old iterations
        if max_iter >= keep_recent:
            cutoff_iter = max_iter - keep_recent + 1
            
            # Remove old iterations
            removed_count = 0
            for player, iter_num in iteration_players:
                if iter_num < cutoff_iter:
                    # Remove from all tracking dictionaries
                    if player in self.ratings:
                        del self.ratings[player]
                    if player in self.rating_uncertainty:
                        del self.rating_uncertainty[player]
                    if player in self.game_counts:
                        del self.game_counts[player]
                    if iter_num in self.iteration_elos:
                        del self.iteration_elos[iter_num]
                    removed_count += 1
            
            if removed_count > 0:
                logger.debug(f"Cleaned up {removed_count} old iteration entries (kept recent {keep_recent})")
    
    def log_detailed_summary(self):
        """Log detailed summary of current ratings state"""
        logger.info("=" * 80)
        logger.info("ELO TRACKER DETAILED SUMMARY")
        logger.info("=" * 80)
        
        # Clean up old iterations before showing summary
        self.cleanup_old_iterations()
        
        # Leaderboard (show recent iterations only)
        leaderboard = self.get_detailed_leaderboard(max_iterations=10)
        logger.info(f"{'Player':<20} {'Rating':<10} {'±σ':<8} {'Games':<8} {'95% CI':<20}")
        logger.info("-" * 70)
        
        for entry in leaderboard[:10]:  # Top 10
            ci_str = f"[{entry['confidence_lower']:.0f}, {entry['confidence_upper']:.0f}]"
            logger.info(f"{entry['player']:<20} {entry['rating']:<10.1f} "
                       f"{entry['uncertainty']:<8.1f} {entry['games']:<8} {ci_str:<20}")
        
        # Health report
        health = self.get_health_report()
        logger.info("\nSYSTEM HEALTH:")
        logger.info(f"Total Players: {health['total_players']}")
        logger.info(f"Total Games: {health['total_games']}")
        
        if "mean_rating" in health["rating_statistics"]:
            stats = health["rating_statistics"]
            logger.info(f"Rating Stats: μ={stats['mean_rating']:.1f}, σ={stats['std_rating']:.1f}, "
                       f"range=[{stats['min_rating']:.1f}, {stats['max_rating']:.1f}]")
        
        # Validation metrics
        validation = health["validation_metrics"]
        if "prediction_accuracy" in validation:
            logger.info(f"Prediction Accuracy: {validation['prediction_accuracy']:.1%}")
            logger.info(f"RMSE: {validation['rmse']:.3f}")
            logger.info(f"Bias: {validation['bias']:+.3f}")
        
        # Inflation check
        inflation = health["inflation_indicators"]
        if "inflation_detected" in inflation:
            if inflation["inflation_detected"]:
                logger.warning("WARNING: Rating inflation detected!")
                logger.info(f"Growth rate: {inflation['avg_growth_per_iteration']:.1f} ELO/iteration")
        
        # Recommendations
        if health["recommendations"]:
            logger.info("\nRECOMMENDATIONS:")
            for rec in health["recommendations"]:
                logger.info(f"- {rec}")
        
        logger.info("=" * 80)


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
            
            # Try to get actual input channels from model if available
            if hasattr(evaluator1, 'model'):
                # Check for config attribute in model
                if hasattr(evaluator1.model, 'config') and hasattr(evaluator1.model.config, 'input_channels'):
                    input_channels1 = evaluator1.model.config.input_channels
                # Or check for direct input_channels attribute
                elif hasattr(evaluator1.model, 'input_channels'):
                    input_channels1 = evaluator1.model.input_channels
                    
            if hasattr(evaluator2, 'model'):
                # Check for config attribute in model
                if hasattr(evaluator2.model, 'config') and hasattr(evaluator2.model.config, 'input_channels'):
                    input_channels2 = evaluator2.model.config.input_channels
                # Or check for direct input_channels attribute
                elif hasattr(evaluator2.model, 'input_channels'):
                    input_channels2 = evaluator2.model.input_channels
                
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
                from mcts.gpu.mcts_gpu_accelerator import get_mcts_gpu_accelerator
                logger.debug("Pre-loading CUDA kernels for arena...")
                kernels = get_mcts_gpu_accelerator('cuda')
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
            # Fallback: use default allocation for arena
            import psutil
            cpu_count = psutil.cpu_count(logical=True) or 4
            allocation = {
                'num_workers': min(self.arena_config.num_workers, cpu_count),
                'max_concurrent_workers': min(self.arena_config.num_workers, cpu_count),
                'memory_per_worker_mb': 512,  # Conservative default for arena
                'batch_size': min(256, self.arena_config.num_games),
                'gpu_batch_timeout_ms': 100  # Arena prefers lower latency
            }
        
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
                    workload_type="latency",  # Arena prefers low latency
                    use_tensorrt=getattr(self.config.mcts, 'use_tensorrt', False),  # Default to False for arena stability
                    tensorrt_fp16=getattr(self.config.mcts, 'tensorrt_fp16', False)  # Default to False for arena stability
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
                    workload_type="latency",  # Arena prefers low latency
                    use_tensorrt=getattr(self.config.mcts, 'use_tensorrt', False),  # Default to False for arena stability
                    tensorrt_fp16=getattr(self.config.mcts, 'tensorrt_fp16', False)  # Default to False for arena stability
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
                logger.debug(f"[ARENA] Game {game_idx} START - GPU Memory: Allocated={gpu_allocated:.3f}GB, Reserved={gpu_reserved:.3f}GB")
                
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
                
                # Log MCTS tree size instead of memory usage
                if hasattr(mcts1, 'tree') and hasattr(mcts1.tree, 'num_nodes'):
                    mcts1_nodes = mcts1.tree.num_nodes
                    mcts2_nodes = mcts2.tree.num_nodes if hasattr(mcts2, 'tree') else 0
                    logger.debug(f"[ARENA] MCTS tree size: Player1={mcts1_nodes} nodes, Player2={mcts2_nodes} nodes")
        
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
                if hasattr(current_mcts, 'clear'):
                    current_mcts.clear()
                
                # Run MCTS search
                with torch.no_grad():  # Ensure no gradients are tracked
                    policy = current_mcts.search(state, num_simulations=self.arena_config.mcts_simulations)
                
                if temp == 0:
                    # Deterministic - choose best action
                    action = current_mcts.select_action(state, temperature=0.0)
                else:
                    # Sample from policy with temperature
                    action = current_mcts.select_action(state, temperature=temp)
                
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
                        if hasattr(mcts1, 'clear'):
                            mcts1.clear()
                        del mcts1
                    except:
                        pass
                        
                if 'mcts2' in locals():
                    try:
                        if hasattr(mcts2, 'clear'):
                            mcts2.clear()
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
                    if hasattr(mcts1, 'clear'):
                        mcts1.clear()
                except Exception as e:
                    logger.warning(f"Error during MCTS cleanup: {e}")
                del mcts1
            if 'mcts2' in locals():
                try:
                    if hasattr(mcts2, 'clear'):
                        mcts2.clear()
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
        
        # Create MCTS instance
        mcts = MCTS(mcts_config, evaluator)
        
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
            # Use remote evaluator for neural network with optimized timeout
            batch_timeout = config_dict.get('mcts', {}).get('gpu_batch_timeout', 0.1)  # Arena timeout
            evaluator1 = RemoteEvaluator(request_queue1, response_queue1, action_size, 
                                       worker_id=game_idx, batch_timeout=batch_timeout)
        
        if is_random2:
            evaluator2 = RandomEvaluator(
                config=EvaluatorConfig(device='cpu'),  # Random evaluator always on CPU
                action_size=action_size
            )
        else:
            # Use remote evaluator for neural network with optimized timeout
            batch_timeout = config_dict.get('mcts', {}).get('gpu_batch_timeout', 0.1)  # Arena timeout
            evaluator2 = RemoteEvaluator(request_queue2, response_queue2, action_size, 
                                       worker_id=game_idx, batch_timeout=batch_timeout)
        
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
        
        # MCTS instances are ready to use
        logger.debug(f"[ARENA WORKER {game_idx}] MCTS instances created successfully")
        
        logger.debug(f"[ARENA WORKER {game_idx}] MCTS configured - device: {mcts_config.device}, memory: {mcts_config.memory_pool_size_mb}MB")
        
        # Play game
        state = game_interface.create_initial_state()
        current_player = 1
        
        for move_num in range(config_dict['training']['max_moves_per_game']):
            # Get current MCTS
            current_mcts = mcts1 if current_player == 1 else mcts2
            
            # Run MCTS search and get best action (deterministic)
            with torch.no_grad():  # Ensure no gradients are tracked
                action = current_mcts.select_action(state, temperature=0.0)
            
            # Apply action
            state = game_interface.get_next_state(state, action)
            
            # No tree reuse in arena to save memory
            if hasattr(mcts1, 'clear'):
                mcts1.clear()
                
            if hasattr(mcts2, 'clear'):
                mcts2.clear()
            
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