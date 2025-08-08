"""
Compatibility wrapper for CPU-optimized MCTS.

This wrapper configures the existing MCTS to use CPU-optimized settings
while maintaining full compatibility with the existing infrastructure.
"""

import numpy as np
from typing import Optional, Dict, Any
import logging
import os

from ..core.mcts import MCTS
from ..core.mcts_config import MCTSConfig

logger = logging.getLogger(__name__)


class CPUOptimizedMCTSWrapper(MCTS):
    """
    CPU-optimized MCTS wrapper that:
    - Uses the existing optimized MCTS implementation
    - Forces CPU backend configuration
    - Enables CPU-specific optimizations
    - Ensures proper batch sizes and parallelization
    """
    
    def __init__(self, config: MCTSConfig, evaluator, game_interface):
        """Initialize with CPU-optimized configuration"""
        # Force CPU backend settings
        config.backend = 'cpu'
        config.device = 'cpu'
        
        # CPU-optimized parameters if not already set
        if not hasattr(config, 'batch_size'):
            config.batch_size = 64  # Optimal for CPU
        
        if not hasattr(config, 'max_tree_nodes'):
            config.max_tree_nodes = 800000  # Allow larger trees on CPU
        
        if not hasattr(config, 'virtual_loss'):
            config.virtual_loss = 3.0
        
        if not hasattr(config, 'num_parallel_reads'):
            config.num_parallel_reads = 16  # Parallel reads for CPU
        
        if not hasattr(config, 'cpu_threads_per_worker'):
            config.cpu_threads_per_worker = 1
            
        # Set wave size for CPU optimization
        if not hasattr(config, 'wave_size'):
            config.wave_size = 64  # Optimal wave size for CPU
            
        if not hasattr(config, 'cpu_wave_size'):
            config.cpu_wave_size = 128  # CPU-specific wave size
        
        # Disable GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        # Initialize base MCTS with CPU settings
        super().__init__(config, evaluator, game_interface)
        
        logger.info("Initialized CPU-optimized MCTS wrapper")
    
    def get_stats(self):
        """Get MCTS statistics"""
        # Check if parent class has stats
        if hasattr(self, 'stats'):
            return self.stats
        else:
            # Return basic stats if not available
            return {
                'total_simulations': getattr(self, 'total_simulations', 0),
                'last_search_sims_per_second': getattr(self, 'last_search_sims_per_second', 0),
                'tree_nodes': self.tree.get_num_nodes() if hasattr(self.tree, 'get_num_nodes') else 0
            }


def create_cpu_optimized_mcts(config: MCTSConfig, evaluator, game_interface) -> MCTS:
    """
    Factory function to create CPU-optimized MCTS instance.
    
    This can be used as a drop-in replacement for MCTS creation:
    
    Instead of:
        mcts = MCTS(config, evaluator, game_interface)
    
    Use:
        mcts = create_cpu_optimized_mcts(config, evaluator, game_interface)
    """
    # Force CPU backend settings
    config.backend = 'cpu'
    config.device = 'cpu'
    
    # CPU-specific optimizations - don't override user settings
    if not hasattr(config, 'batch_size'):
        config.batch_size = 32  # Default small batch for CPU
    
    # Don't override max_tree_nodes if already set - let user control capacity
    if not hasattr(config, 'max_tree_nodes'):
        config.max_tree_nodes = 200000  # Default capacity for CPU
    
    # Wave-based optimization for CPU
    if not hasattr(config, 'wave_size'):
        config.wave_size = 64  # Larger waves for better batching
    
    if not hasattr(config, 'cpu_wave_size'):
        config.cpu_wave_size = 128  # CPU-specific wave size
        
    if not hasattr(config, 'num_workers'):
        import multiprocessing as mp
        config.num_workers = max(1, mp.cpu_count() // 2)  # Use half the CPU cores
        
    # Enable wave features
    config.enable_wave_features = True
    
    # Use the wrapper
    return CPUOptimizedMCTSWrapper(config, evaluator, game_interface)