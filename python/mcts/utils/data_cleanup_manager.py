"""Data cleanup manager for AlphaZero training pipeline

This module handles automatic cleanup of training data to prevent disk space issues:
- Replay buffers (self-play data)
- Model checkpoints
- Arena logs
- Training metrics
- Log files
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataCleanupManager:
    """Manages automatic cleanup of training data to prevent disk space issues"""
    
    def __init__(self, experiment_dir: Path, config):
        """Initialize the data cleanup manager
        
        Args:
            experiment_dir: Root experiment directory
            config: Training configuration with cleanup settings
        """
        self.experiment_dir = Path(experiment_dir)
        self.config = config
        
        # Key directories
        self.data_dir = self.experiment_dir / config.training.data_dir
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.arena_log_dir = self.experiment_dir / getattr(config.arena, 'arena_log_dir', 'arena_logs')
        self.metrics_dir = self.experiment_dir / "metrics"
        self.logs_dir = self.experiment_dir / "logs"
        self.best_models_dir = self.experiment_dir / "best_models"
        
        # Cleanup settings from config
        self.enabled = getattr(config.training, 'enable_data_cleanup', True)
        self.cleanup_interval = getattr(config.training, 'cleanup_interval', 5)
        self.keep_replay_buffers = getattr(config.training, 'keep_last_n_replay_buffers', 10)
        self.keep_checkpoints = getattr(config.training, 'keep_last_n_checkpoints', 5)
        self.keep_arena_logs = getattr(config.training, 'keep_last_n_arena_logs', 20)
        self.keep_metrics = getattr(config.training, 'keep_last_n_metrics', 50)
        self.keep_logs = getattr(config.training, 'keep_last_n_logs', 10)
        self.max_disk_usage_gb = getattr(config.training, 'max_disk_usage_gb', None)
        self.cleanup_arena_logs = getattr(config.training, 'cleanup_arena_logs', True)
        self.cleanup_metrics = getattr(config.training, 'cleanup_metrics', False)
        self.cleanup_logs = getattr(config.training, 'cleanup_logs', True)
        
        logger.info(f"Data cleanup manager initialized:")
        logger.info(f"  Enabled: {self.enabled}")
        logger.info(f"  Cleanup interval: every {self.cleanup_interval} iterations")
        logger.info(f"  Keep replay buffers: {self.keep_replay_buffers}")
        logger.info(f"  Keep checkpoints: {self.keep_checkpoints}")
        if self.cleanup_arena_logs:
            logger.info(f"  Keep arena logs: {self.keep_arena_logs}")
        if self.cleanup_metrics:
            logger.info(f"  Keep metrics: {self.keep_metrics}")
        if self.cleanup_logs:
            logger.info(f"  Keep log files: {self.keep_logs}")
        if self.max_disk_usage_gb:
            logger.info(f"  Max disk usage: {self.max_disk_usage_gb} GB")
    
    def should_cleanup(self, iteration: int) -> bool:
        """Check if cleanup should be performed at this iteration
        
        Args:
            iteration: Current training iteration
            
        Returns:
            True if cleanup should be performed
        """
        if not self.enabled:
            return False
        
        # Always cleanup on first iteration to establish baseline
        if iteration == 1:
            return True
            
        # Cleanup every N iterations
        return iteration % self.cleanup_interval == 0
    
    def cleanup_all(self, iteration: int) -> Dict[str, int]:
        """Perform comprehensive cleanup of all data types
        
        Args:
            iteration: Current training iteration
            
        Returns:
            Dictionary with cleanup statistics
        """
        if not self.enabled:
            logger.debug("Data cleanup is disabled")
            return {}
        
        logger.info(f"🧹 Starting data cleanup at iteration {iteration}")
        
        stats = {}
        total_space_freed = 0
        
        # Get initial disk usage
        initial_usage = self._get_directory_size(self.experiment_dir)
        logger.info(f"📊 Current experiment directory size: {initial_usage / (1024**3):.2f} GB")
        
        # Clean up replay buffers (highest priority - these are largest)
        if self.data_dir.exists():
            freed_gb, count = self._cleanup_replay_buffers()
            stats['replay_buffers_removed'] = count
            stats['replay_buffers_space_freed_gb'] = freed_gb
            total_space_freed += freed_gb
        
        # Clean up checkpoints (keep essential ones)
        if self.checkpoint_dir.exists():
            freed_gb, count = self._cleanup_checkpoints()
            stats['checkpoints_removed'] = count
            stats['checkpoint_space_freed_gb'] = freed_gb
            total_space_freed += freed_gb
        
        # Clean up arena logs if enabled
        if self.cleanup_arena_logs and self.arena_log_dir.exists():
            freed_gb, count = self._cleanup_arena_logs()
            stats['arena_logs_removed'] = count
            stats['arena_logs_space_freed_gb'] = freed_gb
            total_space_freed += freed_gb
        
        # Clean up metrics if enabled
        if self.cleanup_metrics and self.metrics_dir.exists():
            freed_gb, count = self._cleanup_metrics()
            stats['metrics_removed'] = count  
            stats['metrics_space_freed_gb'] = freed_gb
            total_space_freed += freed_gb
        
        # Clean up log files if enabled
        if self.cleanup_logs and self.logs_dir.exists():
            freed_gb, count = self._cleanup_logs()
            stats['logs_removed'] = count
            stats['logs_space_freed_gb'] = freed_gb
            total_space_freed += freed_gb
        
        # Check if we need emergency cleanup due to disk usage
        if self.max_disk_usage_gb:
            final_usage = self._get_directory_size(self.experiment_dir)
            final_usage_gb = final_usage / (1024**3)
            if final_usage_gb > self.max_disk_usage_gb:
                logger.warning(f"⚠️  Disk usage ({final_usage_gb:.2f} GB) exceeds limit ({self.max_disk_usage_gb} GB)")
                emergency_freed = self._emergency_cleanup(final_usage_gb - self.max_disk_usage_gb)
                stats['emergency_cleanup_gb'] = emergency_freed
                total_space_freed += emergency_freed
        
        # Final statistics
        final_usage = self._get_directory_size(self.experiment_dir)
        stats['total_space_freed_gb'] = total_space_freed
        stats['final_directory_size_gb'] = final_usage / (1024**3)
        
        if total_space_freed > 0:
            logger.info(f"✅ Cleanup completed: freed {total_space_freed:.2f} GB")
            logger.info(f"📊 Final directory size: {stats['final_directory_size_gb']:.2f} GB")
        else:
            logger.info("✅ Cleanup completed: no files removed")
        
        return stats
    
    def _cleanup_replay_buffers(self) -> Tuple[float, int]:
        """Clean up old replay buffers, keeping the most recent N
        
        Returns:
            Tuple of (space_freed_gb, files_removed)
        """
        pattern = "replay_buffer_*.pkl"
        files = sorted(
            self.data_dir.glob(pattern),
            key=lambda f: self._extract_iteration_from_filename(f)
        )
        
        if len(files) <= self.keep_replay_buffers:
            logger.debug(f"Only {len(files)} replay buffers found, keeping all")
            return 0.0, 0
        
        # Remove old files
        to_remove = files[:-self.keep_replay_buffers]
        space_freed = 0
        removed_count = 0
        
        for file_path in to_remove:
            try:
                size = file_path.stat().st_size
                file_path.unlink()
                space_freed += size
                removed_count += 1
                iteration = self._extract_iteration_from_filename(file_path)
                logger.debug(f"Removed replay buffer for iteration {iteration}: {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {e}")
        
        space_freed_gb = space_freed / (1024**3)
        if removed_count > 0:
            logger.info(f"🗑️  Cleaned up {removed_count} replay buffers, freed {space_freed_gb:.2f} GB")
        
        return space_freed_gb, removed_count
    
    def _cleanup_checkpoints(self) -> Tuple[float, int]:
        """Clean up old checkpoints, keeping the most recent N
        
        Returns:
            Tuple of (space_freed_gb, files_removed)
        """
        pattern = "checkpoint_*.pt"
        files = sorted(
            self.checkpoint_dir.glob(pattern),
            key=lambda f: self._extract_iteration_from_filename(f)
        )
        
        if len(files) <= self.keep_checkpoints:
            logger.debug(f"Only {len(files)} checkpoints found, keeping all")
            return 0.0, 0
        
        # Remove old files
        to_remove = files[:-self.keep_checkpoints]
        space_freed = 0
        removed_count = 0
        
        for file_path in to_remove:
            try:
                size = file_path.stat().st_size
                file_path.unlink()
                space_freed += size
                removed_count += 1
                iteration = self._extract_iteration_from_filename(file_path)
                logger.debug(f"Removed checkpoint for iteration {iteration}: {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {e}")
        
        space_freed_gb = space_freed / (1024**3)
        if removed_count > 0:
            logger.info(f"🗑️  Cleaned up {removed_count} checkpoints, freed {space_freed_gb:.2f} GB")
        
        return space_freed_gb, removed_count
    
    def _cleanup_arena_logs(self) -> Tuple[float, int]:
        """Clean up old arena log files
        
        Returns:
            Tuple of (space_freed_gb, files_removed)
        """
        # Arena logs typically have timestamp-based names
        pattern = "arena_games_*.json"
        files = sorted(
            self.arena_log_dir.glob(pattern),
            key=lambda f: f.stat().st_mtime
        )
        
        if len(files) <= self.keep_arena_logs:
            logger.debug(f"Only {len(files)} arena logs found, keeping all")
            return 0.0, 0
        
        # Remove old files
        to_remove = files[:-self.keep_arena_logs]
        space_freed = 0
        removed_count = 0
        
        for file_path in to_remove:
            try:
                size = file_path.stat().st_size
                file_path.unlink()
                space_freed += size
                removed_count += 1
                logger.debug(f"Removed arena log: {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {e}")
        
        space_freed_gb = space_freed / (1024**3)
        if removed_count > 0:
            logger.info(f"🗑️  Cleaned up {removed_count} arena logs, freed {space_freed_gb:.2f} GB")
        
        return space_freed_gb, removed_count
    
    def _cleanup_metrics(self) -> Tuple[float, int]:
        """Clean up old metrics files
        
        Returns:
            Tuple of (space_freed_gb, files_removed)
        """
        pattern = "metrics_iter*.json"
        files = sorted(
            self.metrics_dir.glob(pattern),
            key=lambda f: self._extract_iteration_from_filename(f)
        )
        
        if len(files) <= self.keep_metrics:
            logger.debug(f"Only {len(files)} metrics files found, keeping all")
            return 0.0, 0
        
        # Remove old files
        to_remove = files[:-self.keep_metrics]
        space_freed = 0
        removed_count = 0
        
        for file_path in to_remove:
            try:
                size = file_path.stat().st_size
                file_path.unlink()
                space_freed += size
                removed_count += 1
                iteration = self._extract_iteration_from_filename(file_path)
                logger.debug(f"Removed metrics for iteration {iteration}: {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {e}")
        
        space_freed_gb = space_freed / (1024**3)
        if removed_count > 0:
            logger.info(f"🗑️  Cleaned up {removed_count} metrics files, freed {space_freed_gb:.2f} GB")
        
        return space_freed_gb, removed_count
    
    def _cleanup_logs(self) -> Tuple[float, int]:
        """Clean up old log files (excluding the current one)
        
        Returns:
            Tuple of (space_freed_gb, files_removed)
        """
        # Log files typically have timestamp-based names
        pattern = "training_*.log*"  # Includes rotated logs (.log.1, .log.2, etc.)
        files = sorted(
            self.logs_dir.glob(pattern),
            key=lambda f: f.stat().st_mtime,
            reverse=True  # Most recent first
        )
        
        if len(files) <= self.keep_logs:
            logger.debug(f"Only {len(files)} log files found, keeping all")
            return 0.0, 0
        
        # Remove old files (keep the most recent N)
        to_remove = files[self.keep_logs:]
        space_freed = 0
        removed_count = 0
        
        for file_path in to_remove:
            try:
                size = file_path.stat().st_size
                file_path.unlink()
                space_freed += size
                removed_count += 1
                logger.debug(f"Removed log file: {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {e}")
        
        space_freed_gb = space_freed / (1024**3)
        if removed_count > 0:
            logger.info(f"🗑️  Cleaned up {removed_count} log files, freed {space_freed_gb:.2f} GB")
        
        return space_freed_gb, removed_count
    
    def _emergency_cleanup(self, target_gb_to_free: float) -> float:
        """Perform emergency cleanup when disk usage exceeds limits
        
        Args:
            target_gb_to_free: Target amount of space to free in GB
            
        Returns:
            Actual space freed in GB
        """
        logger.warning(f"🚨 Emergency cleanup: attempting to free {target_gb_to_free:.2f} GB")
        
        total_freed = 0.0
        
        # More aggressive cleanup of replay buffers
        if total_freed < target_gb_to_free:
            freed, _ = self._aggressive_cleanup_replay_buffers(target_gb_to_free - total_freed)
            total_freed += freed
        
        # More aggressive cleanup of checkpoints
        if total_freed < target_gb_to_free:
            freed, _ = self._aggressive_cleanup_checkpoints(target_gb_to_free - total_freed)
            total_freed += freed
        
        # Clean up all arena logs if needed
        if total_freed < target_gb_to_free and self.arena_log_dir.exists():
            freed, _ = self._cleanup_all_arena_logs()
            total_freed += freed
        
        logger.warning(f"🚨 Emergency cleanup freed {total_freed:.2f} GB")
        return total_freed
    
    def _aggressive_cleanup_replay_buffers(self, target_gb: float) -> Tuple[float, int]:
        """Aggressively clean replay buffers to meet disk usage target"""
        files = sorted(
            self.data_dir.glob("replay_buffer_*.pkl"),
            key=lambda f: self._extract_iteration_from_filename(f)
        )
        
        # Keep only the last 3 replay buffers in emergency
        min_keep = min(3, len(files))
        to_remove = files[:-min_keep] if len(files) > min_keep else []
        
        space_freed = 0
        removed_count = 0
        
        for file_path in to_remove:
            if space_freed / (1024**3) >= target_gb:
                break
            try:
                size = file_path.stat().st_size
                file_path.unlink()
                space_freed += size
                removed_count += 1
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {e}")
        
        space_freed_gb = space_freed / (1024**3)
        logger.warning(f"⚠️  Emergency: removed {removed_count} replay buffers, freed {space_freed_gb:.2f} GB")
        return space_freed_gb, removed_count
    
    def _aggressive_cleanup_checkpoints(self, target_gb: float) -> Tuple[float, int]:
        """Aggressively clean checkpoints to meet disk usage target"""
        files = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda f: self._extract_iteration_from_filename(f)
        )
        
        # Keep only the last 2 checkpoints in emergency
        min_keep = min(2, len(files))
        to_remove = files[:-min_keep] if len(files) > min_keep else []
        
        space_freed = 0
        removed_count = 0
        
        for file_path in to_remove:
            if space_freed / (1024**3) >= target_gb:
                break
            try:
                size = file_path.stat().st_size
                file_path.unlink()
                space_freed += size
                removed_count += 1
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {e}")
        
        space_freed_gb = space_freed / (1024**3)
        logger.warning(f"⚠️  Emergency: removed {removed_count} checkpoints, freed {space_freed_gb:.2f} GB")
        return space_freed_gb, removed_count
    
    def _cleanup_all_arena_logs(self) -> Tuple[float, int]:
        """Remove all arena logs except the most recent one"""
        files = sorted(
            self.arena_log_dir.glob("arena_games_*.json"),
            key=lambda f: f.stat().st_mtime
        )
        
        # Keep only the most recent arena log
        to_remove = files[:-1] if len(files) > 1 else []
        
        space_freed = 0
        removed_count = 0
        
        for file_path in to_remove:
            try:
                size = file_path.stat().st_size
                file_path.unlink()
                space_freed += size
                removed_count += 1
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {e}")
        
        space_freed_gb = space_freed / (1024**3)
        logger.warning(f"⚠️  Emergency: removed {removed_count} arena logs, freed {space_freed_gb:.2f} GB")
        return space_freed_gb, removed_count
    
    def _extract_iteration_from_filename(self, file_path: Path) -> int:
        """Extract iteration number from filename
        
        Args:
            file_path: Path to file
            
        Returns:
            Iteration number, or 0 if not found
        """
        try:
            # Handle various filename patterns
            name = file_path.stem
            if 'replay_buffer_' in name:
                return int(name.split('replay_buffer_')[1])
            elif 'checkpoint_' in name:
                return int(name.split('checkpoint_')[1])
            elif 'metrics_iter' in name:
                return int(name.split('metrics_iter')[1])
            else:
                return 0
        except (ValueError, IndexError):
            return 0
    
    def _get_directory_size(self, directory: Path) -> int:
        """Get total size of directory in bytes
        
        Args:
            directory: Directory to measure
            
        Returns:
            Size in bytes
        """
        if not directory.exists():
            return 0
        
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except (OSError, FileNotFoundError):
                        # File might have been deleted during iteration
                        pass
        except Exception as e:
            logger.warning(f"Error calculating directory size for {directory}: {e}")
        
        return total_size
    
    def get_cleanup_status(self) -> Dict[str, Any]:
        """Get current cleanup status and disk usage information
        
        Returns:
            Dictionary with cleanup status information
        """
        status = {
            'enabled': self.enabled,
            'cleanup_interval': self.cleanup_interval,
            'experiment_dir_size_gb': self._get_directory_size(self.experiment_dir) / (1024**3),
            'max_disk_usage_gb': self.max_disk_usage_gb,
        }
        
        # Count current files
        if self.data_dir.exists():
            status['replay_buffers_count'] = len(list(self.data_dir.glob("replay_buffer_*.pkl")))
        
        if self.checkpoint_dir.exists():
            status['checkpoints_count'] = len(list(self.checkpoint_dir.glob("checkpoint_*.pt")))
        
        if self.arena_log_dir.exists():
            status['arena_logs_count'] = len(list(self.arena_log_dir.glob("arena_games_*.json")))
        
        if self.metrics_dir.exists():
            status['metrics_count'] = len(list(self.metrics_dir.glob("metrics_iter*.json")))
        
        return status