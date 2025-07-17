"""
Memory Management for Overnight MCTS Physics Analysis

This module provides memory management utilities for handling large-scale overnight
MCTS physics analysis with thousands of games and millions of data points.

Key Features:
- Automatic memory cleanup and garbage collection
- Data streaming and batching for large datasets
- Progress checkpointing for long-running analyses
- Memory usage monitoring and alerts
- Compressed data storage and retrieval
"""

import gc
import psutil
import time
import logging
from typing import Dict, List, Any, Optional, Iterator, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pickle
import gzip
import json
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    """Configuration for memory management"""
    max_memory_gb: float = 8.0  # Maximum memory usage in GB
    cleanup_threshold: float = 0.8  # Cleanup when memory usage exceeds this fraction
    checkpoint_interval: int = 100  # Save checkpoint every N games
    compression_level: int = 6  # Compression level for data storage
    batch_size: int = 50  # Number of games to process in each batch
    enable_monitoring: bool = True  # Enable memory monitoring
    
    def __post_init__(self):
        """Validate configuration"""
        if self.max_memory_gb <= 0:
            raise ValueError("max_memory_gb must be positive")
        if not 0 < self.cleanup_threshold <= 1:
            raise ValueError("cleanup_threshold must be between 0 and 1")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")

class MemoryMonitor:
    """Monitor system memory usage and provide alerts"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.process = psutil.Process()
        self.start_time = time.time()
        self.memory_history = []
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        try:
            # Process memory
            process_memory = self.process.memory_info()
            process_mb = process_memory.rss / 1024 / 1024
            
            # System memory
            system_memory = psutil.virtual_memory()
            system_available_gb = system_memory.available / 1024 / 1024 / 1024
            system_used_percent = system_memory.percent
            
            # GPU memory if available
            gpu_memory_mb = 0
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            except ImportError:
                pass
            
            usage = {
                'process_memory_mb': process_mb,
                'system_available_gb': system_available_gb,
                'system_used_percent': system_used_percent,
                'gpu_memory_mb': gpu_memory_mb,
                'timestamp': time.time()
            }
            
            if self.config.enable_monitoring:
                self.memory_history.append(usage)
            
            return usage
            
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return {'process_memory_mb': 0, 'system_available_gb': 0, 'system_used_percent': 0}
    
    def should_cleanup(self) -> bool:
        """Check if memory cleanup should be performed"""
        usage = self.get_memory_usage()
        
        # Check process memory
        process_gb = usage['process_memory_mb'] / 1024
        if process_gb > self.config.max_memory_gb * self.config.cleanup_threshold:
            return True
        
        # Check system memory
        if usage['system_used_percent'] > 85:  # System memory critically high
            return True
        
        return False
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary"""
        if not self.memory_history:
            return {}
        
        recent_usage = self.memory_history[-10:]  # Last 10 measurements
        
        return {
            'current_usage': self.get_memory_usage(),
            'average_process_mb': np.mean([u['process_memory_mb'] for u in recent_usage]),
            'peak_process_mb': max([u['process_memory_mb'] for u in recent_usage]),
            'memory_trend': self._calculate_memory_trend(),
            'total_measurements': len(self.memory_history),
            'runtime_hours': (time.time() - self.start_time) / 3600
        }
    
    def _calculate_memory_trend(self) -> str:
        """Calculate memory usage trend"""
        if len(self.memory_history) < 5:
            return 'unknown'
        
        recent_usage = [u['process_memory_mb'] for u in self.memory_history[-5:]]
        early_usage = [u['process_memory_mb'] for u in self.memory_history[:5]]
        
        recent_avg = np.mean(recent_usage)
        early_avg = np.mean(early_usage)
        
        if recent_avg > early_avg * 1.2:
            return 'increasing'
        elif recent_avg < early_avg * 0.8:
            return 'decreasing'
        else:
            return 'stable'

class DataBatcher:
    """Batch data processing for memory efficiency"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.current_batch = []
        self.batch_count = 0
        
    def add_data(self, data: Any) -> Optional[List[Any]]:
        """Add data to current batch, return completed batch if ready"""
        self.current_batch.append(data)
        
        if len(self.current_batch) >= self.config.batch_size:
            completed_batch = self.current_batch.copy()
            self.current_batch = []
            self.batch_count += 1
            return completed_batch
        
        return None
    
    def get_final_batch(self) -> Optional[List[Any]]:
        """Get final batch (even if incomplete)"""
        if self.current_batch:
            final_batch = self.current_batch.copy()
            self.current_batch = []
            self.batch_count += 1
            return final_batch
        
        return None
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batching statistics"""
        return {
            'batches_processed': self.batch_count,
            'current_batch_size': len(self.current_batch),
            'configured_batch_size': self.config.batch_size
        }

class CheckpointManager:
    """Manage checkpoints for long-running analyses"""
    
    def __init__(self, config: MemoryConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.checkpoint_dir = output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_counter = 0
        self.last_checkpoint_time = time.time()
        
    def should_checkpoint(self, items_processed: int) -> bool:
        """Check if checkpoint should be created"""
        return items_processed % self.config.checkpoint_interval == 0
    
    def save_checkpoint(self, data: Dict[str, Any], checkpoint_id: str = None) -> str:
        """Save checkpoint data"""
        if checkpoint_id is None:
            checkpoint_id = f"checkpoint_{self.checkpoint_counter:06d}"
        
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pkl.gz"
        
        # Add metadata
        checkpoint_data = {
            'data': data,
            'timestamp': time.time(),
            'checkpoint_id': checkpoint_id,
            'items_processed': data.get('items_processed', 0)
        }
        
        # Save compressed
        with gzip.open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        self.checkpoint_counter += 1
        self.last_checkpoint_time = time.time()
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint data"""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pkl.gz"
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with gzip.open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
            return checkpoint_data['data']
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None
    
    def list_checkpoints(self) -> List[str]:
        """List available checkpoints"""
        return [f.stem.replace('.pkl', '') for f in self.checkpoint_dir.glob('*.pkl.gz')]
    
    def cleanup_old_checkpoints(self, keep_last: int = 10) -> None:
        """Remove old checkpoints, keeping only the most recent"""
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_*.pkl.gz'))
        
        if len(checkpoints) > keep_last:
            for old_checkpoint in checkpoints[:-keep_last]:
                old_checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {old_checkpoint}")

class StreamingDataProcessor:
    """Process large datasets in streaming fashion"""
    
    def __init__(self, config: MemoryConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.monitor = MemoryMonitor(config)
        self.batcher = DataBatcher(config)
        self.checkpoint_manager = CheckpointManager(config, output_dir)
        
        self.processed_count = 0
        self.start_time = time.time()
        
    def process_data_stream(self, data_generator: Iterator[Any], 
                           process_func: callable) -> Iterator[Any]:
        """Process data stream with memory management"""
        logger.info("Starting streaming data processing")
        
        for item in data_generator:
            # Add to batch
            batch = self.batcher.add_data(item)
            
            if batch is not None:
                # Process batch
                try:
                    processed_batch = process_func(batch)
                    yield processed_batch
                    
                    self.processed_count += len(batch)
                    
                    # Memory management
                    if self.monitor.should_cleanup():
                        self._perform_cleanup()
                    
                    # Checkpointing
                    if self.checkpoint_manager.should_checkpoint(self.processed_count):
                        checkpoint_data = {
                            'processed_count': self.processed_count,
                            'batch_stats': self.batcher.get_batch_stats(),
                            'memory_summary': self.monitor.get_memory_summary()
                        }
                        self.checkpoint_manager.save_checkpoint(checkpoint_data)
                    
                    # Progress logging
                    if self.processed_count % 100 == 0:
                        memory_usage = self.monitor.get_memory_usage()
                        logger.info(f"Processed {self.processed_count} items, "
                                  f"Memory: {memory_usage['process_memory_mb']:.1f}MB")
                
                except Exception as e:
                    logger.error(f"Failed to process batch: {e}")
                    continue
        
        # Process final batch
        final_batch = self.batcher.get_final_batch()
        if final_batch:
            try:
                processed_batch = process_func(final_batch)
                yield processed_batch
                self.processed_count += len(final_batch)
            except Exception as e:
                logger.error(f"Failed to process final batch: {e}")
        
        # Final checkpoint
        final_checkpoint = {
            'processed_count': self.processed_count,
            'total_runtime': time.time() - self.start_time,
            'memory_summary': self.monitor.get_memory_summary(),
            'final_batch_stats': self.batcher.get_batch_stats()
        }
        self.checkpoint_manager.save_checkpoint(final_checkpoint, 'final_checkpoint')
        
        logger.info(f"Completed streaming processing: {self.processed_count} items")
    
    def _perform_cleanup(self) -> None:
        """Perform memory cleanup"""
        logger.info("Performing memory cleanup")
        
        # Python garbage collection
        gc.collect()
        
        # PyTorch GPU memory cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        # Log memory usage after cleanup
        usage = self.monitor.get_memory_usage()
        logger.info(f"Memory after cleanup: {usage['process_memory_mb']:.1f}MB")

class CompressedDataStorage:
    """Handle compressed storage of large datasets"""
    
    def __init__(self, config: MemoryConfig, storage_dir: Path):
        self.config = config
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
    def save_data(self, data: Any, filename: str) -> str:
        """Save data with compression"""
        filepath = self.storage_dir / f"{filename}.pkl.gz"
        
        with gzip.open(filepath, 'wb', compresslevel=self.config.compression_level) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Log compression ratio
        original_size = len(pickle.dumps(data))
        compressed_size = filepath.stat().st_size
        ratio = compressed_size / original_size if original_size > 0 else 0
        
        logger.info(f"Saved {filename}: {compressed_size/1024/1024:.1f}MB "
                   f"(compression ratio: {ratio:.2f})")
        
        return str(filepath)
    
    def load_data(self, filename: str) -> Any:
        """Load compressed data"""
        filepath = self.storage_dir / f"{filename}.pkl.gz"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        with gzip.open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Loaded {filename}: {filepath.stat().st_size/1024/1024:.1f}MB")
        return data
    
    def list_stored_data(self) -> List[str]:
        """List all stored data files"""
        return [f.stem.replace('.pkl', '') for f in self.storage_dir.glob('*.pkl.gz')]

class OvernightAnalysisManager:
    """Manage overnight analysis runs with memory optimization"""
    
    def __init__(self, config: MemoryConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.monitor = MemoryMonitor(config)
        self.checkpoint_manager = CheckpointManager(config, output_dir)
        self.storage = CompressedDataStorage(config, output_dir / 'compressed_data')
        
        self.start_time = time.time()
        self.analysis_log = []
        
    def run_overnight_analysis(self, analysis_function: callable, 
                             data_generator: Iterator[Any]) -> Dict[str, Any]:
        """Run overnight analysis with full memory management"""
        logger.info("Starting overnight analysis with memory management")
        
        # Initialize progress tracking
        progress = {
            'items_processed': 0,
            'start_time': self.start_time,
            'checkpoints_created': 0,
            'memory_cleanups': 0
        }
        
        try:
            # Process data in streaming fashion
            processor = StreamingDataProcessor(self.config, self.output_dir)
            
            results = []
            for batch_result in processor.process_data_stream(data_generator, analysis_function):
                results.append(batch_result)
                progress['items_processed'] += 1
                
                # Memory monitoring
                if self.monitor.should_cleanup():
                    self._perform_comprehensive_cleanup()
                    progress['memory_cleanups'] += 1
                
                # Log progress
                if progress['items_processed'] % 50 == 0:
                    self._log_progress(progress)
            
            # Final results compilation
            final_results = {
                'analysis_results': results,
                'progress': progress,
                'memory_summary': self.monitor.get_memory_summary(),
                'runtime_hours': (time.time() - self.start_time) / 3600
            }
            
            # Save final results
            self.storage.save_data(final_results, 'overnight_analysis_results')
            
            return final_results
            
        except KeyboardInterrupt:
            logger.info("Analysis interrupted by user")
            return self._handle_interruption(progress)
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return self._handle_failure(progress, e)
    
    def _perform_comprehensive_cleanup(self) -> None:
        """Perform comprehensive memory cleanup"""
        logger.info("Performing comprehensive memory cleanup")
        
        # Python garbage collection
        gc.collect()
        
        # PyTorch cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
        
        # Cleanup old checkpoints
        self.checkpoint_manager.cleanup_old_checkpoints()
        
        # Log memory state
        memory_summary = self.monitor.get_memory_summary()
        logger.info(f"Memory after cleanup: {memory_summary['current_usage']['process_memory_mb']:.1f}MB")
    
    def _log_progress(self, progress: Dict[str, Any]) -> None:
        """Log analysis progress"""
        runtime_hours = (time.time() - self.start_time) / 3600
        memory_usage = self.monitor.get_memory_usage()
        
        progress_msg = (
            f"Progress: {progress['items_processed']} items processed, "
            f"Runtime: {runtime_hours:.1f}h, "
            f"Memory: {memory_usage['process_memory_mb']:.1f}MB"
        )
        
        logger.info(progress_msg)
        self.analysis_log.append({
            'timestamp': time.time(),
            'message': progress_msg,
            'progress': progress.copy(),
            'memory_usage': memory_usage
        })
    
    def _handle_interruption(self, progress: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user interruption gracefully"""
        logger.info("Handling analysis interruption")
        
        # Save interruption checkpoint
        interruption_data = {
            'interruption_time': time.time(),
            'progress': progress,
            'memory_summary': self.monitor.get_memory_summary(),
            'analysis_log': self.analysis_log
        }
        
        self.checkpoint_manager.save_checkpoint(interruption_data, 'interruption_checkpoint')
        
        return {
            'status': 'interrupted',
            'progress': progress,
            'checkpoint_available': True
        }
    
    def _handle_failure(self, progress: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        """Handle analysis failure"""
        logger.error(f"Handling analysis failure: {error}")
        
        # Save failure checkpoint
        failure_data = {
            'failure_time': time.time(),
            'error': str(error),
            'progress': progress,
            'memory_summary': self.monitor.get_memory_summary(),
            'analysis_log': self.analysis_log
        }
        
        self.checkpoint_manager.save_checkpoint(failure_data, 'failure_checkpoint')
        
        return {
            'status': 'failed',
            'error': str(error),
            'progress': progress,
            'checkpoint_available': True
        }

# Convenience functions for common use cases

def configure_for_overnight_analysis(max_memory_gb: float = 8.0,
                                   batch_size: int = 50,
                                   checkpoint_interval: int = 100) -> MemoryConfig:
    """Configure memory management for overnight analysis"""
    return MemoryConfig(
        max_memory_gb=max_memory_gb,
        cleanup_threshold=0.8,
        checkpoint_interval=checkpoint_interval,
        compression_level=6,
        batch_size=batch_size,
        enable_monitoring=True
    )

def create_overnight_manager(output_dir: Path, 
                           max_memory_gb: float = 8.0) -> OvernightAnalysisManager:
    """Create overnight analysis manager with default configuration"""
    config = configure_for_overnight_analysis(max_memory_gb)
    return OvernightAnalysisManager(config, output_dir)

# Example usage
if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example of memory-managed overnight analysis
    config = configure_for_overnight_analysis(max_memory_gb=8.0)
    output_dir = Path('./overnight_analysis_test')
    
    manager = OvernightAnalysisManager(config, output_dir)
    
    # Example data generator
    def example_data_generator():
        for i in range(1000):
            yield {'data': np.random.rand(100, 100), 'index': i}
    
    # Example analysis function
    def example_analysis(batch):
        return [{'result': np.mean(item['data']), 'index': item['index']} for item in batch]
    
    # Run overnight analysis
    results = manager.run_overnight_analysis(example_analysis, example_data_generator())
    
    print(f"Analysis completed: {len(results['analysis_results'])} batches processed")
    print(f"Runtime: {results['runtime_hours']:.1f} hours")
    print(f"Memory cleanups: {results['progress']['memory_cleanups']}")