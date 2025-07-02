"""
Classical Memory Buffers for Optimized MCTS
==========================================

This module provides pre-allocated memory buffers for classical MCTS computation,
ensuring memory optimization parity with quantum-inspired MCTS.

The buffers eliminate memory allocation overhead during UCB computation,
providing the same optimization benefits that quantum MCTS receives.
"""

import torch
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClassicalMemoryConfig:
    """Configuration for classical memory buffers"""
    max_batch_size: int = 4096        # Maximum batch size for vectorized operations
    max_actions_per_node: int = 512   # Maximum actions per node (game-specific)
    device: str = 'cuda'              # Computation device
    dtype: torch.dtype = torch.float32  # Data type for buffers
    
    # Memory optimization settings
    enable_memory_pooling: bool = True    # Reuse buffers across calls
    preallocate_workspace: bool = True    # Pre-allocate all workspace tensors
    enable_gradient_cache: bool = False   # Cache gradients (usually not needed for MCTS)


class ClassicalMemoryBuffers:
    """
    Pre-allocated memory buffers for classical UCB computation.
    
    This provides equivalent memory optimization to quantum MCTS's parameter tensors,
    eliminating allocation overhead and ensuring fair performance comparison.
    
    Features:
    - Zero allocation during UCB computation
    - Memory pool reuse across calls
    - GPU-optimized tensor layouts
    - Automatic resizing for larger batches
    """
    
    def __init__(self, config: ClassicalMemoryConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Initialize all buffers
        self._allocate_buffers()
        
        # Performance statistics
        self.stats = {
            'buffer_hits': 0,
            'buffer_resizes': 0, 
            'memory_allocated_mb': 0.0,
            'peak_memory_mb': 0.0
        }
        
        logger.debug(f"Classical memory buffers initialized for batch_size={config.max_batch_size}")
        logger.debug(f"Memory allocated: {self._calculate_memory_usage():.2f} MB")
        
    def _allocate_buffers(self):
        """Allocate all pre-computed memory buffers"""
        batch_size = self.config.max_batch_size
        max_actions = self.config.max_actions_per_node
        device = self.device
        dtype = self.config.dtype
        
        # Core UCB computation buffers
        self.ucb_workspace = torch.zeros(
            (batch_size, max_actions), device=device, dtype=dtype
        )
        
        self.q_value_buffer = torch.zeros(
            (batch_size, max_actions), device=device, dtype=dtype
        )
        
        self.exploration_buffer = torch.zeros(
            (batch_size, max_actions), device=device, dtype=dtype
        )
        
        self.prior_buffer = torch.zeros(
            (batch_size, max_actions), device=device, dtype=dtype
        )
        
        # Visit count buffers (int32 for efficiency)
        self.child_visits_buffer = torch.zeros(
            (batch_size, max_actions), device=device, dtype=torch.int32
        )
        
        self.parent_visits_buffer = torch.zeros(
            batch_size, device=device, dtype=torch.int32
        )
        
        # Selection result buffers
        self.ucb_scores_buffer = torch.zeros(
            (batch_size, max_actions), device=device, dtype=dtype
        )
        
        self.selected_actions_buffer = torch.zeros(
            batch_size, device=device, dtype=torch.int32
        )
        
        self.selected_scores_buffer = torch.zeros(
            batch_size, device=device, dtype=dtype
        )
        
        # Mask buffers (boolean)
        self.valid_mask_buffer = torch.zeros(
            (batch_size, max_actions), device=device, dtype=torch.bool
        )
        
        self.visited_mask_buffer = torch.zeros(
            (batch_size, max_actions), device=device, dtype=torch.bool
        )
        
        # Intermediate computation buffers
        self.sqrt_parent_buffer = torch.zeros(
            batch_size, device=device, dtype=dtype
        )
        
        self.exploration_factors_buffer = torch.zeros(
            (batch_size, max_actions), device=device, dtype=dtype
        )
        
        # Advanced computation buffers for research variants
        self.value_sums_buffer = torch.zeros(
            (batch_size, max_actions), device=device, dtype=dtype
        )
        
        self.virtual_loss_buffer = torch.zeros(
            (batch_size, max_actions), device=device, dtype=dtype
        )
        
        # Temperature and noise buffers
        self.temperature_buffer = torch.zeros(
            (batch_size, max_actions), device=device, dtype=dtype
        )
        
        self.noise_buffer = torch.zeros(
            (batch_size, max_actions), device=device, dtype=dtype
        )
        
        # Reduction buffers for finding max/argmax
        self.reduction_workspace = torch.zeros(
            batch_size, device=device, dtype=dtype
        )
        
        self.indices_buffer = torch.zeros(
            batch_size, device=device, dtype=torch.int64
        )
        
        # Mark buffers as ready
        self._buffers_ready = True
        
    def get_ucb_workspace(self, batch_size: int, num_actions: int) -> torch.Tensor:
        """
        Get workspace tensor for UCB computation
        
        Args:
            batch_size: Current batch size
            num_actions: Number of actions per node
            
        Returns:
            Pre-allocated workspace tensor
        """
        self._check_and_resize(batch_size, num_actions)
        self.stats['buffer_hits'] += 1
        
        # Return slice of pre-allocated buffer
        return self.ucb_workspace[:batch_size, :num_actions]
    
    def get_computation_buffers(self, batch_size: int, num_actions: int) -> Dict[str, torch.Tensor]:
        """
        Get all computation buffers for UCB calculation
        
        Args:
            batch_size: Current batch size
            num_actions: Number of actions per node
            
        Returns:
            Dictionary of pre-allocated buffers
        """
        self._check_and_resize(batch_size, num_actions)
        self.stats['buffer_hits'] += 1
        
        return {
            'q_values': self.q_value_buffer[:batch_size, :num_actions],
            'exploration': self.exploration_buffer[:batch_size, :num_actions],
            'priors': self.prior_buffer[:batch_size, :num_actions],
            'child_visits': self.child_visits_buffer[:batch_size, :num_actions],
            'parent_visits': self.parent_visits_buffer[:batch_size],
            'ucb_scores': self.ucb_scores_buffer[:batch_size, :num_actions],
            'valid_mask': self.valid_mask_buffer[:batch_size, :num_actions],
            'visited_mask': self.visited_mask_buffer[:batch_size, :num_actions],
            'sqrt_parent': self.sqrt_parent_buffer[:batch_size],
            'exploration_factors': self.exploration_factors_buffer[:batch_size, :num_actions]
        }
    
    def get_selection_buffers(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Get buffers for selection results
        
        Args:
            batch_size: Current batch size
            
        Returns:
            Dictionary of selection result buffers
        """
        self._check_and_resize(batch_size, 1)  # Only need to check batch size
        self.stats['buffer_hits'] += 1
        
        return {
            'selected_actions': self.selected_actions_buffer[:batch_size],
            'selected_scores': self.selected_scores_buffer[:batch_size],
            'reduction_workspace': self.reduction_workspace[:batch_size],
            'indices': self.indices_buffer[:batch_size]
        }
    
    def get_advanced_buffers(self, batch_size: int, num_actions: int) -> Dict[str, torch.Tensor]:
        """
        Get advanced computation buffers for research variants
        
        Args:
            batch_size: Current batch size
            num_actions: Number of actions per node
            
        Returns:
            Dictionary of advanced computation buffers
        """
        self._check_and_resize(batch_size, num_actions)
        self.stats['buffer_hits'] += 1
        
        return {
            'value_sums': self.value_sums_buffer[:batch_size, :num_actions],
            'virtual_loss': self.virtual_loss_buffer[:batch_size, :num_actions],
            'temperature': self.temperature_buffer[:batch_size, :num_actions],
            'noise': self.noise_buffer[:batch_size, :num_actions]
        }
    
    def _check_and_resize(self, batch_size: int, num_actions: int):
        """Check if buffers need resizing and resize if necessary"""
        current_batch_size = self.ucb_workspace.shape[0]
        current_num_actions = self.ucb_workspace.shape[1]
        
        needs_resize = (batch_size > current_batch_size or 
                       num_actions > current_num_actions)
        
        if needs_resize:
            logger.info(f"Resizing classical buffers: ({current_batch_size}, {current_num_actions}) -> ({batch_size}, {num_actions})")
            self.stats['buffer_resizes'] += 1
            
            # Calculate new sizes with some headroom
            new_batch_size = max(batch_size, current_batch_size) * 2
            new_num_actions = max(num_actions, current_num_actions) * 2
            
            # Update config and reallocate
            old_config = self.config
            self.config = ClassicalMemoryConfig(
                max_batch_size=new_batch_size,
                max_actions_per_node=new_num_actions,
                device=old_config.device,
                dtype=old_config.dtype,
                enable_memory_pooling=old_config.enable_memory_pooling,
                preallocate_workspace=old_config.preallocate_workspace
            )
            
            self._allocate_buffers()
    
    def clear_buffers(self):
        """Clear all buffers (set to zero)"""
        if not self._buffers_ready:
            return
            
        # Clear main computation buffers
        self.ucb_workspace.zero_()
        self.q_value_buffer.zero_()
        self.exploration_buffer.zero_()
        self.prior_buffer.zero_()
        self.ucb_scores_buffer.zero_()
        
        # Clear visit count buffers  
        self.child_visits_buffer.zero_()
        self.parent_visits_buffer.zero_()
        
        # Clear result buffers
        self.selected_actions_buffer.fill_(-1)  # Use -1 for invalid actions
        self.selected_scores_buffer.zero_()
        
        # Clear mask buffers
        self.valid_mask_buffer.fill_(False)
        self.visited_mask_buffer.fill_(False)
        
        # Clear intermediate buffers
        self.sqrt_parent_buffer.zero_()
        self.exploration_factors_buffer.zero_()
        
    def prefetch_buffers(self, batch_size: int, num_actions: int):
        """
        Prefetch buffers to GPU memory for optimal access patterns
        
        Args:
            batch_size: Expected batch size
            num_actions: Expected number of actions
        """
        if self.device.type != 'cuda':
            return  # Only relevant for CUDA
            
        self._check_and_resize(batch_size, num_actions)
        
        # Touch all buffers to ensure they're in GPU memory
        buffers = self.get_computation_buffers(batch_size, num_actions)
        for buffer in buffers.values():
            _ = buffer.sum()  # Minimal operation to ensure GPU residency
            
        selection_buffers = self.get_selection_buffers(batch_size)
        for buffer in selection_buffers.values():
            _ = buffer.sum()
    
    def _calculate_memory_usage(self) -> float:
        """Calculate total memory usage in MB"""
        total_bytes = 0
        
        # Calculate size of each buffer
        for attr_name in dir(self):
            if attr_name.endswith('_buffer') and hasattr(self, attr_name):
                buffer = getattr(self, attr_name)
                if isinstance(buffer, torch.Tensor):
                    total_bytes += buffer.numel() * buffer.element_size()
        
        # Convert to MB
        memory_mb = total_bytes / (1024 * 1024)
        self.stats['memory_allocated_mb'] = memory_mb
        self.stats['peak_memory_mb'] = max(self.stats['peak_memory_mb'], memory_mb)
        
        return memory_mb
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.stats,
            'current_batch_size': self.ucb_workspace.shape[0],
            'current_num_actions': self.ucb_workspace.shape[1],
            'device': str(self.device),
            'buffers_ready': self._buffers_ready
        }
    
    def __del__(self):
        """Cleanup GPU memory when object is destroyed"""
        if hasattr(self, '_buffers_ready') and self._buffers_ready:
            logger.debug("Cleaning up classical memory buffers")


def create_classical_memory_buffers(config: Optional[ClassicalMemoryConfig] = None) -> ClassicalMemoryBuffers:
    """Factory function for creating classical memory buffers"""
    if config is None:
        config = ClassicalMemoryConfig()
    
    return ClassicalMemoryBuffers(config)


# Export main classes and functions
__all__ = [
    'ClassicalMemoryBuffers',
    'ClassicalMemoryConfig',
    'create_classical_memory_buffers'
]