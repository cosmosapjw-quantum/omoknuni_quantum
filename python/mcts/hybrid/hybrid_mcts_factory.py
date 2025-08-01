"""Hybrid MCTS Factory - Creates optimized hybrid CPU/GPU MCTS instances

This module provides factory functions for creating hybrid MCTS instances
that use CPU for tree operations and GPU for neural network evaluation.
"""

import logging
from typing import Optional

from ..core.mcts import MCTS
from ..core.mcts_config import MCTSConfig

logger = logging.getLogger(__name__)


def create_hybrid_mcts(config: MCTSConfig, evaluator, game_interface=None) -> MCTS:
    """
    Factory function to create hybrid CPU/GPU MCTS instance.
    
    This uses:
    - CPU for tree operations (fast Cython implementation)
    - GPU for neural network evaluation
    - Lock-free communication between CPU and GPU
    
    Args:
        config: MCTS configuration
        evaluator: Neural network evaluator (should be on GPU)
        game_interface: Optional game interface
        
    Returns:
        MCTS instance configured for hybrid operation
    """
    # Configure for hybrid backend
    config.backend = 'hybrid'
    
    # Determine device from evaluator
    if hasattr(evaluator, 'device'):
        if hasattr(evaluator.device, 'type'):
            config.device = 'cuda' if evaluator.device.type == 'cuda' else 'cpu'
        else:
            config.device = str(evaluator.device)
    else:
        # Default to CUDA if available
        import torch
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Hybrid-specific optimizations
    if not hasattr(config, 'batch_size'):
        config.batch_size = 32  # Optimal for GPU evaluation
        
    if not hasattr(config, 'num_threads'):
        config.num_threads = 4  # CPU selection threads
        
    if not hasattr(config, 'virtual_loss'):
        config.virtual_loss = 3.0  # For parallel selection
        
    if not hasattr(config, 'enable_virtual_loss'):
        config.enable_virtual_loss = True
        
    # Larger trees since CPU memory is cheaper
    if not hasattr(config, 'max_tree_nodes'):
        config.max_tree_nodes = 800000
        
    # Wave size for batching
    if not hasattr(config, 'wave_size'):
        config.wave_size = config.batch_size
    
    logger.info(f"Creating hybrid MCTS with device={config.device}, "
                f"batch_size={config.batch_size}, num_threads={config.num_threads}")
    
    # Create MCTS with hybrid backend
    return MCTS(
        config=config,
        evaluator=evaluator,
        game_interface=game_interface
    )


def create_fast_hybrid_mcts(config: MCTSConfig, evaluator, game=None) -> 'FastHybridMCTS':
    """
    Create FastHybridMCTS instance directly with all optimizations.
    
    This bypasses the standard MCTS class and uses the optimized
    implementation directly.
    
    Args:
        config: MCTS configuration
        evaluator: Neural network evaluator
        game: Game instance
        
    Returns:
        FastHybridMCTS instance
    """
    from .fast_hybrid_mcts import FastHybridMCTS
    
    # Determine device
    if hasattr(evaluator, 'device'):
        device = str(evaluator.device)
    else:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create fast hybrid instance
    return FastHybridMCTS(
        game=game,
        config=config,
        evaluator=evaluator,
        device=device,
        num_selection_threads=getattr(config, 'num_threads', 4),
        use_optimizations=True
    )