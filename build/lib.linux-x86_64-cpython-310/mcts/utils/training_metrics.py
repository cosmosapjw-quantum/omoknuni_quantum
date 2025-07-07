"""Training metrics recording and analysis system

This module provides a comprehensive system for tracking and analyzing
training metrics during AlphaZero training, including:
- Loss tracking (policy, value, total)
- Performance metrics (win rates, ELO ratings)
- Training dynamics (learning rate, gradient norms)
- Visualization and export capabilities
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Single snapshot of training metrics at a point in time"""
    timestamp: float
    iteration: int
    epoch: int = 0
    
    # Loss metrics
    policy_loss: Optional[float] = None
    value_loss: Optional[float] = None
    total_loss: Optional[float] = None
    
    # Performance metrics
    win_rate: Optional[float] = None
    elo_rating: Optional[float] = None
    elo_change: Optional[float] = None
    
    # Training dynamics
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None
    gradient_max: Optional[float] = None
    
    # Self-play metrics
    games_per_second: Optional[float] = None
    avg_game_length: Optional[float] = None
    policy_entropy: Optional[float] = None
    mcts_value_accuracy: Optional[float] = None
    
    # Resource metrics
    gpu_memory_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    
    # Additional custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class TrainingMetricsRecorder:
    """Records and manages training metrics throughout the training process"""
    
    def __init__(self, 
                 save_dir: Optional[Path] = None,
                 window_size: int = 100,
                 auto_save_interval: int = 10):
        """Initialize metrics recorder
        
        Args:
            save_dir: Directory to save metrics (None to disable auto-save)
            window_size: Size of rolling window for moving averages
            auto_save_interval: Save metrics every N iterations
        """
        self.save_dir = Path(save_dir) if save_dir else None
        self.window_size = window_size
        self.auto_save_interval = auto_save_interval
        
        # Storage
        self.snapshots: List[MetricSnapshot] = []
        self.rolling_windows: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size * 3)  # Allow enough data for trend analysis
        )
        
        # Current state
        self.current_iteration = 0
        self.current_epoch = 0
        self.start_time = time.time()
        self.last_save_iteration = 0
        
        # Statistics
        self.best_metrics = {
            'win_rate': 0.0,
            'elo_rating': 0.0,
            'min_loss': float('inf')
        }
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def record_training_step(self,
                           iteration: int,
                           epoch: int,
                           policy_loss: float,
                           value_loss: float,
                           total_loss: float,
                           learning_rate: float,
                           gradient_norm: Optional[float] = None,
                           gradient_max: Optional[float] = None,
                           **custom_metrics):
        """Record metrics from a training step"""
        snapshot = MetricSnapshot(
            timestamp=time.time(),
            iteration=iteration,
            epoch=epoch,
            policy_loss=policy_loss,
            value_loss=value_loss,
            total_loss=total_loss,
            learning_rate=learning_rate,
            gradient_norm=gradient_norm,
            gradient_max=gradient_max,
            custom_metrics=custom_metrics
        )
        
        self._add_snapshot(snapshot)
        
        # Update rolling windows
        self.rolling_windows['policy_loss'].append(policy_loss)
        self.rolling_windows['value_loss'].append(value_loss)
        self.rolling_windows['total_loss'].append(total_loss)
        
        # Update best metrics
        if total_loss < self.best_metrics['min_loss']:
            self.best_metrics['min_loss'] = total_loss
            self.best_metrics['min_loss_iteration'] = iteration
    
    def record_evaluation(self,
                         iteration: int,
                         win_rate: float,
                         elo_rating: float,
                         elo_change: float,
                         **custom_metrics):
        """Record evaluation metrics"""
        # Find or create snapshot for this iteration
        snapshot = self._get_or_create_snapshot(iteration)
        
        snapshot.win_rate = win_rate
        snapshot.elo_rating = elo_rating
        snapshot.elo_change = elo_change
        
        if custom_metrics:
            snapshot.custom_metrics.update(custom_metrics)
        
        # Update rolling windows
        self.rolling_windows['win_rate'].append(win_rate)
        self.rolling_windows['elo_rating'].append(elo_rating)
        
        # Update best metrics
        if win_rate > self.best_metrics['win_rate']:
            self.best_metrics['win_rate'] = win_rate
            self.best_metrics['best_win_rate_iteration'] = iteration
            
        if elo_rating > self.best_metrics['elo_rating']:
            self.best_metrics['elo_rating'] = elo_rating
            self.best_metrics['best_elo_iteration'] = iteration
        
        self._auto_save_check(iteration)
    
    def record_self_play_metrics(self,
                               iteration: int,
                               games_per_second: float,
                               avg_game_length: float,
                               policy_entropy: float,
                               mcts_value_accuracy: Optional[float] = None,
                               **custom_metrics):
        """Record self-play generation metrics"""
        snapshot = self._get_or_create_snapshot(iteration)
        
        snapshot.games_per_second = games_per_second
        snapshot.avg_game_length = avg_game_length
        snapshot.policy_entropy = policy_entropy
        snapshot.mcts_value_accuracy = mcts_value_accuracy
        
        if custom_metrics:
            snapshot.custom_metrics.update(custom_metrics)
        
        # Update rolling windows
        self.rolling_windows['games_per_second'].append(games_per_second)
        self.rolling_windows['avg_game_length'].append(avg_game_length)
        self.rolling_windows['policy_entropy'].append(policy_entropy)
    
    def record_resource_usage(self,
                            iteration: int,
                            gpu_memory_mb: float,
                            cpu_percent: float):
        """Record resource usage metrics"""
        snapshot = self._get_or_create_snapshot(iteration)
        snapshot.gpu_memory_mb = gpu_memory_mb
        snapshot.cpu_percent = cpu_percent
    
    def get_moving_average(self, metric_name: str, window: Optional[int] = None) -> float:
        """Get moving average for a metric
        
        Args:
            metric_name: Name of the metric
            window: Window size (None uses default)
            
        Returns:
            Moving average value
        """
        if metric_name not in self.rolling_windows:
            return 0.0
            
        values = list(self.rolling_windows[metric_name])
        if not values:
            return 0.0
            
        if window:
            values = values[-window:]
            
        return np.mean(values)
    
    def get_metric_trend(self, metric_name: str, window: int = 10) -> str:
        """Get trend direction for a metric
        
        Returns:
            'improving', 'declining', 'stable', or 'unknown'
        """
        if metric_name not in self.rolling_windows:
            return 'unknown'
            
        values = list(self.rolling_windows[metric_name])
        if len(values) < 2 * window:
            return 'unknown'
            
        recent = np.mean(values[-window:])
        previous = np.mean(values[-2*window:-window])
        
        # For losses, lower is better
        if 'loss' in metric_name:
            if recent < previous * 0.98:  # Less strict threshold for tests
                return 'improving'
            elif recent > previous * 1.02:
                return 'declining'
        else:
            # For other metrics, higher is better
            if recent > previous * 1.02:
                return 'improving'
            elif recent < previous * 0.98:
                return 'declining'
                
        return 'stable'
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of training metrics"""
        if not self.snapshots:
            return {}
            
        latest = self.snapshots[-1]
        elapsed_time = time.time() - self.start_time
        
        summary = {
            'current_iteration': latest.iteration,
            'current_epoch': latest.epoch,
            'elapsed_time_hours': elapsed_time / 3600,
            'iterations_per_hour': latest.iteration / (elapsed_time / 3600),
            
            # Current values
            'current_metrics': {
                'policy_loss': latest.policy_loss,
                'value_loss': latest.value_loss,
                'total_loss': latest.total_loss,
                'win_rate': latest.win_rate,
                'elo_rating': latest.elo_rating,
                'learning_rate': latest.learning_rate,
            },
            
            # Moving averages
            'moving_averages': {
                'policy_loss': self.get_moving_average('policy_loss'),
                'value_loss': self.get_moving_average('value_loss'),
                'total_loss': self.get_moving_average('total_loss'),
                'win_rate': self.get_moving_average('win_rate'),
                'games_per_second': self.get_moving_average('games_per_second'),
            },
            
            # Trends
            'trends': {
                'loss_trend': self.get_metric_trend('total_loss'),
                'win_rate_trend': self.get_metric_trend('win_rate'),
                'elo_trend': self.get_metric_trend('elo_rating'),
            },
            
            # Best metrics
            'best_metrics': self.best_metrics.copy(),
            
            # Performance
            'performance': {
                'avg_games_per_second': self.get_moving_average('games_per_second'),
                'avg_game_length': self.get_moving_average('avg_game_length'),
                'policy_entropy': self.get_moving_average('policy_entropy'),
            }
        }
        
        return summary
    
    def print_summary(self, detailed: bool = False):
        """Print formatted summary to console"""
        summary = self.get_summary()
        if not summary:
            logger.info("No metrics recorded yet")
            return
            
        logger.info("\n" + "="*80)
        logger.info("TRAINING METRICS SUMMARY")
        logger.info("="*80)
        
        logger.info(f"Iteration: {summary['current_iteration']} | "
                   f"Epoch: {summary['current_epoch']} | "
                   f"Time: {summary['elapsed_time_hours']:.1f}h")
        
        current = summary['current_metrics']
        logger.info(f"\nCurrent Metrics:")
        logger.info(f"  Loss: {current['total_loss']:.4f} "
                   f"(P: {current['policy_loss']:.4f}, V: {current['value_loss']:.4f})")
        logger.info(f"  Win Rate: {current['win_rate']:.1%} | "
                   f"ELO: {current['elo_rating']:.0f}")
        logger.info(f"  Learning Rate: {current['learning_rate']:.2e}")
        
        trends = summary['trends']
        logger.info(f"\nTrends:")
        logger.info(f"  Loss: {trends['loss_trend']} | "
                   f"Win Rate: {trends['win_rate_trend']} | "
                   f"ELO: {trends['elo_trend']}")
        
        if detailed:
            avgs = summary['moving_averages']
            logger.info(f"\nMoving Averages ({self.window_size} samples):")
            logger.info(f"  Loss: {avgs['total_loss']:.4f} | "
                       f"Win Rate: {avgs['win_rate']:.1%}")
            logger.info(f"  Games/sec: {avgs['games_per_second']:.1f}")
            
            best = summary['best_metrics']
            logger.info(f"\nBest Results:")
            logger.info(f"  Min Loss: {best['min_loss']:.4f} (iter {best.get('min_loss_iteration', 'N/A')})")
            logger.info(f"  Max Win Rate: {best['win_rate']:.1%} (iter {best.get('best_win_rate_iteration', 'N/A')})")
            logger.info(f"  Max ELO: {best['elo_rating']:.0f} (iter {best.get('best_elo_iteration', 'N/A')})")
        
        logger.info("="*80 + "\n")
    
    def save(self, path: Optional[Path] = None):
        """Save metrics to JSON file"""
        if path is None:
            if self.save_dir is None:
                return
            path = self.save_dir / f"metrics_iter{self.current_iteration}.json"
        
        data = {
            'snapshots': [asdict(s) for s in self.snapshots],
            'best_metrics': self.best_metrics,
            'summary': self.get_summary(),
            'metadata': {
                'window_size': self.window_size,
                'start_time': self.start_time,
                'save_time': time.time(),
            }
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.debug(f"Saved metrics to {path}")
    
    def load(self, path: Path):
        """Load metrics from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
            
        self.snapshots = [MetricSnapshot(**s) for s in data['snapshots']]
        self.best_metrics = data['best_metrics']
        
        # Rebuild rolling windows
        for snapshot in self.snapshots:
            for attr, value in asdict(snapshot).items():
                if value is not None and attr not in ['timestamp', 'iteration', 'epoch', 'custom_metrics']:
                    self.rolling_windows[attr].append(value)
        
        logger.info(f"Loaded {len(self.snapshots)} metric snapshots from {path}")
    
    def export_for_plotting(self) -> Dict[str, List[Tuple[int, float]]]:
        """Export metrics in format suitable for plotting
        
        Returns dict where each metric maps to list of (iteration, value) tuples
        """
        if not self.snapshots:
            return {}
            
        metrics = defaultdict(list)
        
        for snapshot in self.snapshots:
            data = asdict(snapshot)
            iteration = snapshot.iteration
            
            for key, value in data.items():
                if key not in ['custom_metrics', 'iteration', 'epoch', 'timestamp'] and value is not None:
                    metrics[key].append((iteration, value))
                    
            # Add custom metrics
            for key, value in snapshot.custom_metrics.items():
                metrics[f'custom_{key}'].append((iteration, value))
        
        return dict(metrics)
    
    def _add_snapshot(self, snapshot: MetricSnapshot):
        """Add a snapshot and update current state"""
        self.snapshots.append(snapshot)
        self.current_iteration = snapshot.iteration
        self.current_epoch = snapshot.epoch
        
        self._auto_save_check(snapshot.iteration)
    
    def _get_or_create_snapshot(self, iteration: int) -> MetricSnapshot:
        """Get existing snapshot for iteration or create new one"""
        # Check if we already have a snapshot for this iteration
        for snapshot in reversed(self.snapshots):
            if snapshot.iteration == iteration:
                return snapshot
                
        # Create new snapshot
        snapshot = MetricSnapshot(
            timestamp=time.time(),
            iteration=iteration,
            epoch=self.current_epoch
        )
        self.snapshots.append(snapshot)
        return snapshot
    
    def _auto_save_check(self, iteration: int):
        """Check if we should auto-save"""
        if self.save_dir and iteration - self.last_save_iteration >= self.auto_save_interval:
            self.save()
            self.last_save_iteration = iteration


class MetricsVisualizer:
    """Visualize training metrics using matplotlib"""
    
    def __init__(self, metrics_recorder: TrainingMetricsRecorder):
        self.recorder = metrics_recorder
    
    def plot_training_curves(self, save_path: Optional[Path] = None, show: bool = False):
        """Plot training curves"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            logger.warning("matplotlib not installed, skipping visualization")
            return
            
        data = self.recorder.export_for_plotting()
        if not data:
            logger.warning("No data to plot")
            return
            
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig)
        
        # Helper to extract iterations and values
        def extract_data(metric_data):
            if not metric_data:
                return [], []
            iterations, values = zip(*metric_data)
            return list(iterations), list(values)
        
        # Loss curves
        ax1 = fig.add_subplot(gs[0, :])
        if 'policy_loss' in data:
            iters, vals = extract_data(data['policy_loss'])
            ax1.plot(iters, vals, label='Policy Loss', alpha=0.7)
        if 'value_loss' in data:
            iters, vals = extract_data(data['value_loss'])
            ax1.plot(iters, vals, label='Value Loss', alpha=0.7)
        if 'total_loss' in data:
            iters, vals = extract_data(data['total_loss'])
            ax1.plot(iters, vals, label='Total Loss', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Losses')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Win rate
        ax2 = fig.add_subplot(gs[1, 0])
        if 'win_rate' in data:
            iters, vals = extract_data(data['win_rate'])
            if iters:
                ax2.plot(iters, vals, color='green', linewidth=2, marker='o')
            ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Win Rate')
        ax2.set_title('Win Rate vs Previous Best')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        # ELO rating
        ax3 = fig.add_subplot(gs[1, 1])
        if 'elo_rating' in data:
            iters, vals = extract_data(data['elo_rating'])
            if iters:
                ax3.plot(iters, vals, color='blue', linewidth=2, marker='o')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('ELO Rating')
        ax3.set_title('ELO Progression')
        ax3.grid(True, alpha=0.3)
        
        # Learning rate
        ax4 = fig.add_subplot(gs[2, 0])
        if 'learning_rate' in data:
            iters, vals = extract_data(data['learning_rate'])
            if iters:
                ax4.semilogy(iters, vals, color='orange')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.grid(True, alpha=0.3)
        
        # Game metrics
        ax5 = fig.add_subplot(gs[2, 1])
        if 'policy_entropy' in data:
            iters, vals = extract_data(data['policy_entropy'])
            if iters:
                ax5.plot(iters, vals, label='Policy Entropy', alpha=0.7)
        if 'avg_game_length' in data:
            iters, vals = extract_data(data['avg_game_length'])
            if iters:
                ax5_twin = ax5.twinx()
                ax5_twin.plot(iters, vals, label='Avg Game Length', color='red', alpha=0.7)
                ax5_twin.set_ylabel('Game Length')
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Entropy')
        ax5.set_title('Game Statistics')
        ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
            
        if show:
            plt.show()
        else:
            plt.close()