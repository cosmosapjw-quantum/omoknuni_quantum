"""Training monitoring utilities to detect and prevent training collapse"""

import logging
from typing import Dict, List, Optional, Tuple
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
    
    def __init__(self, alert_threshold: int = 3):
        """Initialize training monitor
        
        Args:
            alert_threshold: Number of consecutive unhealthy iterations before alerting
        """
        self.alert_threshold = alert_threshold
        self.unhealthy_count = 0
        self.previous_metrics: Optional[TrainingHealthMetrics] = None
        
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
        
        # Log current metrics
        logger.info(f"Training metrics - Entropy: {metrics.policy_entropy:.3f}, "
                   f"Value Var: {metrics.value_variance:.3f}, "
                   f"Avg Length: {metrics.avg_game_length:.1f}, "
                   f"Win Balance: {metrics.win_rate_balance:.2f}")
        
        return True, warnings
    
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