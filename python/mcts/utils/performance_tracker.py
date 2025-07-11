"""Performance tracking utilities for MCTS and self-play optimization"""

import time
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import psutil
import GPUtil
from datetime import datetime


@dataclass
class SimulationMetrics:
    """Metrics for a single MCTS simulation batch"""
    batch_size: int
    simulations: int
    time_seconds: float
    gpu_memory_mb: float
    gpu_utilization: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class GameMetrics:
    """Metrics for a complete game"""
    game_id: str
    total_moves: int
    total_simulations: int
    total_time: float
    move_times: List[float]
    simulation_counts: List[int]
    gpu_memory_peak_mb: float
    gpu_utilization_avg: float
    value_predictions: List[float]
    policy_entropies: List[float]
    resigned: bool = False
    final_result: Optional[float] = None


class PerformanceTracker:
    """Track and analyze MCTS performance metrics"""
    
    def __init__(self, window_size: int = 100):
        """Initialize performance tracker
        
        Args:
            window_size: Size of rolling window for metrics
        """
        self.window_size = window_size
        self.simulation_metrics = deque(maxlen=window_size)
        self.game_metrics = []
        self.current_game_start = None
        self.current_game_moves = []
        self.current_game_sims = []
        self.current_game_times = []
        self.gpu_memory_samples = []
        self.gpu_util_samples = []
        
        # Performance counters
        self.total_simulations = 0
        self.total_time = 0.0
        self.total_games = 0
        
        # GPU monitoring
        self.has_gpu = torch.cuda.is_available()
        self.gpu_device = 0 if self.has_gpu else None
        
    def start_game(self, game_id: str):
        """Start tracking a new game"""
        self.current_game_start = time.time()
        self.current_game_id = game_id
        self.current_game_moves = []
        self.current_game_sims = []
        self.current_game_times = []
        self.gpu_memory_samples = []
        self.gpu_util_samples = []
        
    def record_move(self, simulations: int, move_time: float, 
                    value_pred: float = None, policy_entropy: float = None):
        """Record metrics for a single move"""
        if self.current_game_start is None:
            return
            
        self.current_game_moves.append({
            'simulations': simulations,
            'time': move_time,
            'value': value_pred,
            'entropy': policy_entropy
        })
        self.current_game_sims.append(simulations)
        self.current_game_times.append(move_time)
        
        # Sample GPU metrics
        if self.has_gpu:
            gpu_memory = torch.cuda.memory_allocated(self.gpu_device) / 1024 / 1024
            self.gpu_memory_samples.append(gpu_memory)
            
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_util = gpus[self.gpu_device].load * 100
                    self.gpu_util_samples.append(gpu_util)
            except:
                pass
    
    def end_game(self, resigned: bool = False, result: float = None) -> GameMetrics:
        """End current game and return metrics"""
        if self.current_game_start is None:
            return None
            
        total_time = time.time() - self.current_game_start
        
        # Extract move data
        value_preds = [m['value'] for m in self.current_game_moves if m['value'] is not None]
        entropies = [m['entropy'] for m in self.current_game_moves if m['entropy'] is not None]
        
        game_metrics = GameMetrics(
            game_id=self.current_game_id,
            total_moves=len(self.current_game_moves),
            total_simulations=sum(self.current_game_sims),
            total_time=total_time,
            move_times=self.current_game_times,
            simulation_counts=self.current_game_sims,
            gpu_memory_peak_mb=max(self.gpu_memory_samples) if self.gpu_memory_samples else 0,
            gpu_utilization_avg=np.mean(self.gpu_util_samples) if self.gpu_util_samples else 0,
            value_predictions=value_preds,
            policy_entropies=entropies,
            resigned=resigned,
            final_result=result
        )
        
        self.game_metrics.append(game_metrics)
        self.total_games += 1
        self.total_simulations += game_metrics.total_simulations
        self.total_time += total_time
        
        # Reset current game tracking
        self.current_game_start = None
        
        return game_metrics
    
    def record_simulation_batch(self, batch_size: int, simulations: int, 
                              time_seconds: float):
        """Record metrics for a simulation batch"""
        gpu_memory = 0
        gpu_util = 0
        
        if self.has_gpu:
            gpu_memory = torch.cuda.memory_allocated(self.gpu_device) / 1024 / 1024
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_util = gpus[self.gpu_device].load * 100
            except:
                pass
        
        metrics = SimulationMetrics(
            batch_size=batch_size,
            simulations=simulations,
            time_seconds=time_seconds,
            gpu_memory_mb=gpu_memory,
            gpu_utilization=gpu_util
        )
        
        self.simulation_metrics.append(metrics)
        
    def get_current_simulations_per_second(self) -> float:
        """Get current simulations per second (rolling window)"""
        if not self.simulation_metrics:
            return 0.0
            
        recent_sims = sum(m.simulations for m in self.simulation_metrics)
        recent_time = sum(m.time_seconds for m in self.simulation_metrics)
        
        if recent_time > 0:
            return recent_sims / recent_time
        return 0.0
    
    def get_overall_simulations_per_second(self) -> float:
        """Get overall simulations per second"""
        if self.total_time > 0:
            return self.total_simulations / self.total_time
        return 0.0
    
    def get_games_per_second(self) -> float:
        """Get games per second"""
        if self.total_time > 0:
            return self.total_games / self.total_time
        return 0.0
    
    def get_average_game_length(self) -> float:
        """Get average game length in moves"""
        if not self.game_metrics:
            return 0.0
        return np.mean([g.total_moves for g in self.game_metrics])
    
    def get_batch_efficiency(self) -> float:
        """Get batch utilization efficiency (0-1)"""
        if not self.simulation_metrics:
            return 0.0
            
        # Calculate average batch fullness
        batch_sizes = [m.batch_size for m in self.simulation_metrics]
        if batch_sizes:
            avg_batch = np.mean(batch_sizes)
            max_batch = max(batch_sizes)
            if max_batch > 0:
                return avg_batch / max_batch
        return 0.0
    
    def get_gpu_metrics(self) -> Dict[str, float]:
        """Get GPU utilization metrics"""
        if not self.has_gpu:
            return {'memory_mb': 0, 'utilization': 0, 'memory_peak_mb': 0}
            
        current_memory = torch.cuda.memory_allocated(self.gpu_device) / 1024 / 1024
        peak_memory = torch.cuda.max_memory_allocated(self.gpu_device) / 1024 / 1024
        
        # Average GPU utilization from recent samples
        recent_utils = []
        for metrics in self.game_metrics[-10:]:  # Last 10 games
            if metrics.gpu_utilization_avg > 0:
                recent_utils.append(metrics.gpu_utilization_avg)
        
        avg_util = np.mean(recent_utils) if recent_utils else 0
        
        return {
            'memory_mb': current_memory,
            'memory_peak_mb': peak_memory,
            'utilization': avg_util
        }
    
    def get_summary(self) -> Dict[str, any]:
        """Get comprehensive performance summary"""
        gpu_metrics = self.get_gpu_metrics()
        
        return {
            'simulations_per_second': self.get_current_simulations_per_second(),
            'overall_sims_per_second': self.get_overall_simulations_per_second(),
            'games_per_second': self.get_games_per_second(),
            'average_game_length': self.get_average_game_length(),
            'batch_efficiency': self.get_batch_efficiency(),
            'total_simulations': self.total_simulations,
            'total_games': self.total_games,
            'total_time': self.total_time,
            'gpu_memory_mb': gpu_metrics['memory_mb'],
            'gpu_memory_peak_mb': gpu_metrics['memory_peak_mb'],
            'gpu_utilization': gpu_metrics['utilization']
        }
    
    def print_summary(self):
        """Print performance summary"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Simulations/second: {summary['simulations_per_second']:.1f} (current), "
              f"{summary['overall_sims_per_second']:.1f} (overall)")
        print(f"Games/second: {summary['games_per_second']:.3f}")
        print(f"Average game length: {summary['average_game_length']:.1f} moves")
        print(f"Batch efficiency: {summary['batch_efficiency']:.2%}")
        print(f"Total: {summary['total_simulations']:,} simulations, "
              f"{summary['total_games']} games in {summary['total_time']:.1f}s")
        
        if self.has_gpu:
            print(f"GPU: {summary['gpu_memory_mb']:.0f}MB used "
                  f"(peak: {summary['gpu_memory_peak_mb']:.0f}MB), "
                  f"{summary['gpu_utilization']:.1f}% utilization")
        print("="*60)


# Global performance tracker instance
_global_tracker = None


def get_global_tracker() -> PerformanceTracker:
    """Get or create global performance tracker"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = PerformanceTracker()
    return _global_tracker


def reset_global_tracker():
    """Reset global performance tracker"""
    global _global_tracker
    _global_tracker = PerformanceTracker()