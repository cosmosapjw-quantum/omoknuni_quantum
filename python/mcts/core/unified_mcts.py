"""Unified MCTS implementation stub

This is a minimal implementation to allow the main MCTS to work.
In practice, the optimized implementation is always used.
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class UnifiedMCTSConfig:
    """Configuration for unified MCTS"""
    num_simulations: int = 1000
    c_puct: float = 1.414
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    wave_size: Optional[int] = None
    min_wave_size: int = 32
    max_wave_size: int = 64
    device: str = 'cuda'
    game_type: Any = None
    board_size: int = 15
    enable_quantum: bool = False
    quantum_config: Any = None
    enable_virtual_loss: bool = True
    virtual_loss_value: float = -3.0


class UnifiedMCTS:
    """Unified MCTS implementation (fallback/compatibility mode)"""
    
    def __init__(self, config: UnifiedMCTSConfig, evaluator: Any):
        self.config = config
        self.evaluator = evaluator
        
    def search(self, state: Any, num_simulations: Optional[int] = None) -> np.ndarray:
        """Run MCTS search - stub implementation"""
        # Return uniform policy for stub
        action_size = 225  # Default for 15x15 board
        return np.ones(action_size) / action_size
    
    def get_statistics(self) -> dict:
        """Get search statistics"""
        return {
            'total_simulations': 0,
            'tree_nodes': 0
        }