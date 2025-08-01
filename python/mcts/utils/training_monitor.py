"""Training monitoring utilities to detect and prevent training collapse"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class TrainingHealthMetrics:
    """Metrics to monitor training health"""
    policy_entropy: float
    value_variance: float
    avg_game_length: float
    win_rate_balance: float  # abs(p1_wins - p2_wins) / total_games
    value_accuracy: float
    resign_rate: float
    
    def is_healthy(self) -> Tuple[bool, List[str]]:
        """Check if training metrics are healthy
        
        Returns:
            Tuple of (is_healthy, list_of_issues)
        """
        issues = []
        
        # Check policy entropy
        if self.policy_entropy < 0.1:
            issues.append(f"Policy entropy too low: {self.policy_entropy:.4f} < 0.1")
        
        # Check value variance
        if self.value_variance < 0.01:
            issues.append(f"Value variance too low: {self.value_variance:.4f} < 0.01")
        
        # Check game length
        if self.avg_game_length < 15:
            issues.append(f"Games too short: {self.avg_game_length:.1f} < 15 moves")
        
        # Check win rate balance
        if self.win_rate_balance > 0.3:
            issues.append(f"Win rate imbalance: {self.win_rate_balance:.2f} > 0.3")
        
        # Check value accuracy
        if self.value_accuracy < 0.3:
            issues.append(f"Value accuracy too low: {self.value_accuracy:.2f} < 0.3")
        
        # Check resign rate
        if self.resign_rate > 0.8:
            issues.append(f"Resign rate too high: {self.resign_rate:.2f} > 0.8")
        
        return len(issues) == 0, issues


class TrainingMonitor:
    """Monitor training health and detect issues"""
    
    def __init__(self, alert_threshold: int = 3, early_stopping_enabled: bool = True):
        """Initialize training monitor
        
        Args:
            alert_threshold: Number of consecutive unhealthy iterations before alerting
            early_stopping_enabled: Whether to trigger early stopping on critical issues
        """
        self.alert_threshold = alert_threshold
        self.early_stopping_enabled = early_stopping_enabled
        self.unhealthy_count = 0
        self.previous_metrics: Optional[TrainingHealthMetrics] = None
        
        # Policy stability tracking
        self.policy_entropy_history = []
        self.win_rate_history = []
        self.value_variance_history = []
        
        # Early stopping thresholds
        self.min_policy_entropy = 0.05  # Stop if entropy drops below this
        self.max_win_rate_imbalance = 0.95  # Stop if one player wins >95% of games
        self.min_value_variance = 0.001  # Stop if values collapse
        self.consecutive_bad_iterations = 5  # Stop after this many bad iterations
        
    def check_training_health(self, game_metrics: Dict[str, any]) -> Tuple[bool, List[str]]:
        """Check if training is healthy based on game metrics
        
        Args:
            game_metrics: Dictionary of metrics from self-play games
            
        Returns:
            Tuple of (should_continue_training, list_of_warnings)
        """
        # Check if we have aggregated metrics from collect_game_metrics
        if 'total_games' in game_metrics and 'avg_entropy' in game_metrics:
            # Use aggregated metrics directly
            total_games = game_metrics.get('total_games', 0)
            if total_games == 0:
                return True, ["No games to analyze"]
            
            metrics = TrainingHealthMetrics(
                policy_entropy=game_metrics.get('avg_entropy', 0.0),
                value_variance=game_metrics.get('value_variance', 0.1),  # Not available in current metrics
                avg_game_length=game_metrics.get('avg_game_length', 0.0),
                win_rate_balance=abs(game_metrics.get('player1_wins', 0) - game_metrics.get('player2_wins', 0)) / total_games,
                value_accuracy=game_metrics.get('value_accuracy', 0.0),
                resign_rate=game_metrics.get('resignation_count', 0) / total_games if total_games > 0 else 0.0
            )
        else:
            # Fall back to old logic for individual game data
            # Extract metrics
            policy_entropies = []
            value_predictions = []
            game_lengths = []
            p1_wins = 0
            p2_wins = 0
            draws = 0
            resigned_games = 0
            value_accuracies = []
            
            if 'games' in game_metrics:
                games = game_metrics['games']
            else:
                # Handle flat metrics structure
                games = [game_metrics]
            
            for game in games:
                if 'policy_entropies' in game and game['policy_entropies']:
                    policy_entropies.extend(game['policy_entropies'])
                
                if 'value_predictions' in game and game['value_predictions']:
                    value_predictions.extend([v for v, _ in game['value_predictions']])
                
                if 'game_length' in game:
                    game_lengths.append(game['game_length'])
                
                if 'winner' in game:
                    if game['winner'] == 1:
                        p1_wins += 1
                    elif game['winner'] == -1:
                        p2_wins += 1
                    else:
                        draws += 1
                
                if 'resigned' in game and game['resigned']:
                    resigned_games += 1
                
                if 'value_accuracies' in game and game['value_accuracies']:
                    value_accuracies.extend(game['value_accuracies'])
            
            # Calculate aggregate metrics
            total_games = len(games)
            if total_games == 0:
                return True, ["No games to analyze"]
            
            metrics = TrainingHealthMetrics(
                policy_entropy=np.mean(policy_entropies) if policy_entropies else 0.0,
                value_variance=np.var(value_predictions) if value_predictions else 0.0,
                avg_game_length=np.mean(game_lengths) if game_lengths else 0.0,
                win_rate_balance=abs(p1_wins - p2_wins) / total_games if total_games > 0 else 0.0,
                value_accuracy=np.mean(value_accuracies) if value_accuracies else 0.0,
                resign_rate=resigned_games / total_games if total_games > 0 else 0.0
            )
        
        # Update tracking histories
        self.update_policy_stability(metrics.policy_entropy)
        self.update_win_rate(
            game_metrics.get('player1_wins', 0),
            game_metrics.get('player2_wins', 0),
            game_metrics.get('total_games', 1)
        )
        self.update_value_variance(metrics.value_variance)
        
        # Check health
        is_healthy, issues = metrics.is_healthy()
        
        warnings = []
        if not is_healthy:
            self.unhealthy_count += 1
            warnings.extend(issues)
            
            if self.unhealthy_count >= self.alert_threshold:
                warnings.append(f"⚠️  CRITICAL: Training has been unhealthy for {self.unhealthy_count} iterations!")
                
                # Check for catastrophic collapse
                if metrics.policy_entropy < 0.01 and metrics.avg_game_length < 15:
                    warnings.append("🚨 TRAINING COLLAPSE DETECTED! Consider stopping and fixing issues.")
                    return False, warnings
        else:
            self.unhealthy_count = 0
        
        # Check for sudden changes
        if self.previous_metrics:
            if abs(metrics.policy_entropy - self.previous_metrics.policy_entropy) > 1.0:
                warnings.append(f"Large entropy change: {self.previous_metrics.policy_entropy:.2f} → {metrics.policy_entropy:.2f}")
            
            if abs(metrics.avg_game_length - self.previous_metrics.avg_game_length) > 20:
                warnings.append(f"Large game length change: {self.previous_metrics.avg_game_length:.1f} → {metrics.avg_game_length:.1f}")
        
        self.previous_metrics = metrics
        
        # Check if we should stop training
        if self.early_stopping_enabled:
            should_stop, stop_reason = self.should_stop_training()
            if should_stop:
                warnings.append(f"🛑 EARLY STOPPING TRIGGERED: {stop_reason}")
                return False, warnings
        
        # Log current metrics (only in debug mode to avoid duplication)
        logger.debug(f"Training metrics - Entropy: {metrics.policy_entropy:.3f}, "
                    f"Value Var: {metrics.value_variance:.3f}, "
                    f"Avg Length: {metrics.avg_game_length:.1f}, "
                    f"Win Balance: {metrics.win_rate_balance:.2f}")
        
        return True, warnings
    
    def update_policy_stability(self, policy_entropy: float):
        """Update policy stability tracking
        
        Args:
            policy_entropy: Current policy entropy value
        """
        self.policy_entropy_history.append(policy_entropy)
        
        # Keep only recent history (last 20 iterations)
        if len(self.policy_entropy_history) > 20:
            self.policy_entropy_history.pop(0)
    
    def update_win_rate(self, player1_wins: int, player2_wins: int, total_games: int):
        """Update win rate tracking
        
        Args:
            player1_wins: Number of wins for player 1
            player2_wins: Number of wins for player 2
            total_games: Total number of games
        """
        if total_games > 0:
            win_rate_imbalance = abs(player1_wins - player2_wins) / total_games
            self.win_rate_history.append(win_rate_imbalance)
            
            # Keep only recent history
            if len(self.win_rate_history) > 20:
                self.win_rate_history.pop(0)
    
    def update_value_variance(self, value_variance: float):
        """Update value variance tracking
        
        Args:
            value_variance: Current value variance
        """
        self.value_variance_history.append(value_variance)
        
        # Keep only recent history
        if len(self.value_variance_history) > 20:
            self.value_variance_history.pop(0)
    
    def should_stop_training(self) -> Tuple[bool, str]:
        """Check if training should be stopped due to critical issues
        
        Returns:
            Tuple of (should_stop, reason_message)
        """
        if not self.early_stopping_enabled:
            return False, ""
        
        # Check for critically low policy entropy
        if self.policy_entropy_history:
            recent_entropy = self.policy_entropy_history[-1]
            if recent_entropy < self.min_policy_entropy:
                return True, f"Policy entropy critically low: {recent_entropy:.4f} < {self.min_policy_entropy}"
        
        # Check for extreme win rate imbalance
        if self.win_rate_history:
            recent_imbalance = self.win_rate_history[-1]
            if recent_imbalance > self.max_win_rate_imbalance:
                return True, f"Extreme win rate imbalance: {recent_imbalance:.2f} > {self.max_win_rate_imbalance}"
        
        # Check for collapsed value variance
        if self.value_variance_history:
            recent_variance = self.value_variance_history[-1]
            if recent_variance < self.min_value_variance:
                return True, f"Value variance collapsed: {recent_variance:.6f} < {self.min_value_variance}"
        
        # Check for sustained bad iterations
        if self.unhealthy_count >= self.consecutive_bad_iterations:
            return True, f"Training unhealthy for {self.unhealthy_count} consecutive iterations"
        
        # Check for entropy trend (decreasing over time)
        if len(self.policy_entropy_history) >= 10:
            # Calculate trend over last 10 iterations
            recent_entropies = self.policy_entropy_history[-10:]
            entropy_trend = np.polyfit(range(10), recent_entropies, 1)[0]
            
            # If entropy is decreasing rapidly
            if entropy_trend < -0.05:  # Losing 0.05 entropy per iteration
                avg_entropy = np.mean(recent_entropies)
                if avg_entropy < 0.5:  # And already at low entropy
                    return True, f"Policy entropy rapidly decreasing: trend={entropy_trend:.4f}/iter, avg={avg_entropy:.3f}"
        
        return False, ""
    
    def get_stability_metrics(self) -> Dict[str, Any]:
        """Get current stability metrics
        
        Returns:
            Dictionary of stability metrics
        """
        metrics = {
            'current_entropy': self.policy_entropy_history[-1] if self.policy_entropy_history else None,
            'avg_entropy_last_10': np.mean(self.policy_entropy_history[-10:]) if len(self.policy_entropy_history) >= 10 else None,
            'entropy_trend': None,
            'current_win_rate_imbalance': self.win_rate_history[-1] if self.win_rate_history else None,
            'current_value_variance': self.value_variance_history[-1] if self.value_variance_history else None,
            'consecutive_unhealthy': self.unhealthy_count
        }
        
        # Calculate entropy trend if enough data
        if len(self.policy_entropy_history) >= 5:
            recent_entropies = self.policy_entropy_history[-5:]
            metrics['entropy_trend'] = np.polyfit(range(5), recent_entropies, 1)[0]
        
        return metrics
    
    def get_recommendations(self, issues: List[str]) -> List[str]:
        """Get recommendations based on detected issues
        
        Args:
            issues: List of detected issues
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        for issue in issues:
            if "entropy too low" in issue:
                recommendations.append("• Increase Dirichlet noise (dirichlet_epsilon)")
                recommendations.append("• Check if value predictions are working correctly")
                recommendations.append("• Consider reducing learning rate")
                
            elif "variance too low" in issue:
                recommendations.append("• Check value head - it may be outputting constant values")
                recommendations.append("• Verify value target calculation is correct")
                recommendations.append("• Consider adding L2 regularization")
                
            elif "Games too short" in issue:
                recommendations.append("• Check for repeated patterns in self-play")
                recommendations.append("• Verify game termination logic")
                recommendations.append("• Consider increasing exploration (temperature)")
                
            elif "Win rate imbalance" in issue:
                recommendations.append("• Check for first-player advantage")
                recommendations.append("• Verify symmetry augmentation is working")
                recommendations.append("• Consider adjusting komi (for Go) or other balance mechanisms")
                
            elif "Value accuracy too low" in issue:
                recommendations.append("• Value head may need more training")
                recommendations.append("• Check if value targets are calculated correctly")
                recommendations.append("• Consider adjusting value loss weight")
                
            elif "Resign rate too high" in issue:
                recommendations.append("• Lower resign threshold (make it more negative)")
                recommendations.append("• Check if value predictions are too pessimistic")
                recommendations.append("• Consider disabling resignation temporarily")
        
        return list(set(recommendations))  # Remove duplicates