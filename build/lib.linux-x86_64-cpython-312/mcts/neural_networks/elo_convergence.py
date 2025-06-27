"""ELO Convergence Detection and Automatic Training Stop

This module implements algorithms to detect when model ELO ratings have
converged and automatically stop training to prevent overfitting and
save computational resources.

Features:
- Moving average based convergence detection
- Statistical significance testing
- Configurable convergence criteria
- Early stopping with patience
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Deque
from collections import deque
from dataclasses import dataclass
import logging
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceConfig:
    """Configuration for ELO convergence detection"""
    # Moving average window
    window_size: int = 10  # Number of iterations to consider
    
    # Convergence thresholds
    elo_improvement_threshold: float = 5.0  # Minimum ELO gain to be considered improvement
    convergence_patience: int = 5  # Iterations to wait after convergence detected
    
    # Statistical testing
    use_statistical_test: bool = True
    confidence_level: float = 0.95  # Confidence level for statistical tests
    
    # Minimum requirements
    min_iterations: int = 20  # Minimum iterations before checking convergence
    min_elo_above_baseline: float = 100.0  # Minimum ELO above baseline (e.g., random)
    
    # Trend analysis
    use_trend_analysis: bool = True
    trend_window: int = 20  # Window for trend analysis
    max_negative_trend: float = -0.5  # Maximum allowed negative trend (ELO/iteration)
    
    # Plateau detection
    plateau_threshold: float = 2.0  # ELO variance threshold for plateau
    plateau_window: int = 15  # Window for plateau detection
    
    # Auto-stop settings
    enable_auto_stop: bool = True
    save_best_on_stop: bool = True


class ELOConvergenceDetector:
    """Detects convergence in ELO ratings and manages automatic training stop"""
    
    def __init__(self, config: ConvergenceConfig):
        self.config = config
        
        # ELO history tracking
        self.elo_history: List[float] = []
        self.iteration_history: List[int] = []
        self.model_checkpoints: List[str] = []
        
        # Moving windows
        self.recent_elos: Deque[float] = deque(maxlen=config.window_size)
        self.trend_window: Deque[float] = deque(maxlen=config.trend_window)
        
        # Convergence state
        self.convergence_detected = False
        self.convergence_iteration = None
        self.patience_counter = 0
        self.should_stop = False
        
        # Best model tracking
        self.best_elo = float('-inf')
        self.best_iteration = None
        self.best_checkpoint = None
        
        # Statistics
        self.stats = {
            'checks_performed': 0,
            'plateaus_detected': 0,
            'negative_trends_detected': 0,
            'convergence_triggers': []
        }
    
    def update(self, iteration: int, elo: float, checkpoint_path: Optional[str] = None) -> bool:
        """Update with new ELO rating and check for convergence
        
        Args:
            iteration: Current training iteration
            elo: Current model ELO rating
            checkpoint_path: Path to model checkpoint
            
        Returns:
            True if training should stop, False otherwise
        """
        # Update history
        self.elo_history.append(elo)
        self.iteration_history.append(iteration)
        if checkpoint_path:
            self.model_checkpoints.append(checkpoint_path)
        
        # Update moving windows
        self.recent_elos.append(elo)
        self.trend_window.append(elo)
        
        # Track best model
        if elo > self.best_elo:
            self.best_elo = elo
            self.best_iteration = iteration
            self.best_checkpoint = checkpoint_path
            logger.info(f"New best model: ELO {elo:.1f} at iteration {iteration}")
        
        # Check if we should start convergence checking
        if iteration < self.config.min_iterations:
            return False
        
        # Perform convergence check
        self.stats['checks_performed'] += 1
        
        # Check various convergence criteria
        converged = False
        reasons = []
        
        # 1. Check for improvement stagnation
        if self._check_improvement_stagnation():
            converged = True
            reasons.append("improvement_stagnation")
        
        # 2. Check for statistical plateau
        if self.config.use_statistical_test and self._check_statistical_plateau():
            converged = True
            reasons.append("statistical_plateau")
            self.stats['plateaus_detected'] += 1
        
        # 3. Check for negative trend
        if self.config.use_trend_analysis and self._check_negative_trend():
            converged = True
            reasons.append("negative_trend")
            self.stats['negative_trends_detected'] += 1
        
        # 4. Check if below minimum threshold
        if not self._check_minimum_performance():
            converged = False  # Override - not good enough yet
            reasons = []
        
        # Handle convergence detection
        if converged and not self.convergence_detected:
            self.convergence_detected = True
            self.convergence_iteration = iteration
            self.stats['convergence_triggers'] = reasons
            logger.warning(f"Convergence detected at iteration {iteration} (ELO: {elo:.1f}). "
                          f"Reasons: {', '.join(reasons)}")
        
        # Handle patience counter
        if self.convergence_detected:
            self.patience_counter += 1
            
            if self.patience_counter >= self.config.convergence_patience:
                self.should_stop = True
                logger.warning(f"Stopping training after {self.patience_counter} iterations "
                             f"of patience. Final ELO: {elo:.1f}")
        else:
            # Reset patience if we start improving again
            self.patience_counter = 0
        
        return self.should_stop and self.config.enable_auto_stop
    
    def _check_improvement_stagnation(self) -> bool:
        """Check if ELO improvement has stagnated"""
        if len(self.recent_elos) < self.config.window_size:
            return False
        
        # Calculate improvement over window
        window_start = self.recent_elos[0]
        window_end = self.recent_elos[-1]
        improvement = window_end - window_start
        
        # Check if improvement is below threshold
        return improvement < self.config.elo_improvement_threshold
    
    def _check_statistical_plateau(self) -> bool:
        """Check for statistical plateau using variance analysis"""
        if len(self.elo_history) < self.config.plateau_window:
            return False
        
        # Get recent ELOs for analysis
        recent = self.elo_history[-self.config.plateau_window:]
        
        # Calculate variance
        variance = np.var(recent)
        
        # Low variance indicates plateau
        return variance < self.config.plateau_threshold ** 2
    
    def _check_negative_trend(self) -> bool:
        """Check for negative trend in ELO ratings"""
        if len(self.trend_window) < self.config.trend_window:
            return False
        
        # Fit linear regression
        x = np.arange(len(self.trend_window))
        y = np.array(self.trend_window)
        
        # Calculate trend
        slope, _, _, _, _ = stats.linregress(x, y)
        
        # Check if trend is too negative
        return slope < self.config.max_negative_trend
    
    def _check_minimum_performance(self) -> bool:
        """Check if model meets minimum performance requirements"""
        if not self.elo_history:
            return False
        
        current_elo = self.elo_history[-1]
        
        # Assuming baseline (random) is ELO 0
        return current_elo >= self.config.min_elo_above_baseline
    
    def get_convergence_report(self) -> Dict:
        """Get detailed convergence analysis report"""
        if not self.elo_history:
            return {"status": "no_data"}
        
        report = {
            "status": "converged" if self.convergence_detected else "training",
            "current_elo": self.elo_history[-1],
            "best_elo": self.best_elo,
            "best_iteration": self.best_iteration,
            "iterations_trained": len(self.elo_history),
            "convergence_detected": self.convergence_detected,
            "convergence_iteration": self.convergence_iteration,
            "should_stop": self.should_stop,
            "statistics": self.stats
        }
        
        # Add trend analysis
        if len(self.elo_history) >= 2:
            recent_improvement = self.elo_history[-1] - self.elo_history[-min(10, len(self.elo_history))]
            total_improvement = self.elo_history[-1] - self.elo_history[0]
            
            report["recent_improvement"] = recent_improvement
            report["total_improvement"] = total_improvement
            report["average_elo_per_iteration"] = total_improvement / len(self.elo_history)
        
        # Add variance analysis
        if len(self.elo_history) >= self.config.plateau_window:
            recent = self.elo_history[-self.config.plateau_window:]
            report["recent_variance"] = np.var(recent)
            report["recent_std"] = np.std(recent)
        
        return report
    
    def plot_convergence(self, save_path: Optional[str] = None):
        """Plot ELO progression and convergence indicators"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available, skipping plot")
            return
        
        if not self.elo_history:
            logger.warning("No ELO history to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Main ELO plot
        ax1.plot(self.iteration_history, self.elo_history, 'b-', label='ELO Rating')
        
        # Mark best model
        if self.best_iteration is not None:
            ax1.axvline(x=self.best_iteration, color='g', linestyle='--', 
                       label=f'Best Model (ELO: {self.best_elo:.1f})')
        
        # Mark convergence point
        if self.convergence_iteration is not None:
            ax1.axvline(x=self.convergence_iteration, color='r', linestyle='--',
                       label='Convergence Detected')
        
        # Add moving average
        if len(self.elo_history) >= self.config.window_size:
            ma = np.convolve(self.elo_history, 
                           np.ones(self.config.window_size) / self.config.window_size,
                           mode='valid')
            ma_iterations = self.iteration_history[self.config.window_size - 1:]
            ax1.plot(ma_iterations, ma, 'r-', alpha=0.5, 
                    label=f'Moving Average ({self.config.window_size})')
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('ELO Rating')
        ax1.set_title('ELO Rating Progression')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Improvement rate plot
        if len(self.elo_history) >= 2:
            improvements = np.diff(self.elo_history)
            ax2.plot(self.iteration_history[1:], improvements, 'b-', alpha=0.5)
            
            # Add zero line
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax2.axhline(y=self.config.elo_improvement_threshold, color='g', 
                       linestyle='--', alpha=0.5, label='Improvement Threshold')
            
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('ELO Improvement per Iteration')
            ax2.set_title('ELO Improvement Rate')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Convergence plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_state(self, filepath: str):
        """Save detector state for resume"""
        import json
        
        state = {
            'elo_history': self.elo_history,
            'iteration_history': self.iteration_history,
            'model_checkpoints': self.model_checkpoints,
            'convergence_detected': self.convergence_detected,
            'convergence_iteration': self.convergence_iteration,
            'patience_counter': self.patience_counter,
            'should_stop': self.should_stop,
            'best_elo': self.best_elo,
            'best_iteration': self.best_iteration,
            'best_checkpoint': self.best_checkpoint,
            'stats': self.stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load detector state for resume"""
        import json
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.elo_history = state['elo_history']
        self.iteration_history = state['iteration_history']
        self.model_checkpoints = state['model_checkpoints']
        self.convergence_detected = state['convergence_detected']
        self.convergence_iteration = state['convergence_iteration']
        self.patience_counter = state['patience_counter']
        self.should_stop = state['should_stop']
        self.best_elo = state['best_elo']
        self.best_iteration = state['best_iteration']
        self.best_checkpoint = state['best_checkpoint']
        self.stats = state['stats']
        
        # Rebuild deques
        self.recent_elos = deque(self.elo_history[-self.config.window_size:],
                               maxlen=self.config.window_size)
        self.trend_window = deque(self.elo_history[-self.config.trend_window:],
                                maxlen=self.config.trend_window)