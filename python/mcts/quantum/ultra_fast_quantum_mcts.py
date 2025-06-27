"""
Ultra-Fast Quantum MCTS - Speed-First Implementation
==================================================

Goal: Make quantum MCTS faster than classical MCTS
Strategy: Minimal quantum overhead with maximum vectorization

Key optimizations:
1. Pre-computed lookup tables for quantum corrections
2. Minimal branching and conditionals
3. Pure tensor operations without loops
4. Selective quantum only where absolutely beneficial
5. JIT compilation for critical paths
6. Memory-efficient implementations

Target: < 1.0x overhead vs classical MCTS
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class UltraFastQuantumConfig:
    """Ultra-minimal configuration for maximum speed"""
    
    device: str = 'cpu'
    base_c_puct: float = 1.4
    
    # Ultra-minimal quantum settings
    quantum_bonus_coefficient: float = 0.02  # Very small for speed
    quantum_crossover_threshold: int = 10     # Very low threshold
    enable_quantum_bonus: bool = True
    
    # Disable all non-essential features
    enable_power_law_annealing: bool = False
    enable_phase_adaptation: bool = False
    enable_correlation_prioritization: bool = False
    enable_coherent_state_management: bool = False
    enable_performance_monitoring: bool = False
    
    # Performance optimizations
    precompute_quantum_tables: bool = True
    max_precomputed_visits: int = 100
    use_jit_compilation: bool = True

class UltraFastQuantumMCTS:
    """
    Ultra-optimized quantum MCTS focused purely on speed
    
    Eliminates all overhead while maintaining minimal quantum benefits.
    """
    
    def __init__(self, config: UltraFastQuantumConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Pre-compute quantum lookup tables for speed
        if config.precompute_quantum_tables:
            self._precompute_quantum_tables()
        else:
            self.quantum_table = None
        
        # Compile JIT functions
        if config.use_jit_compilation:
            self._compile_jit_functions()
        
        logger.info(f"UltraFastQuantumMCTS initialized on {config.device}")
        logger.info(f"  Quantum bonus: {config.quantum_bonus_coefficient}")
        logger.info(f"  Crossover threshold: {config.quantum_crossover_threshold}")
    
    def _precompute_quantum_tables(self):
        """Pre-compute quantum bonuses for common visit counts"""
        max_visits = self.config.max_precomputed_visits
        visit_range = torch.arange(1, max_visits + 1, dtype=torch.float32, device=self.device)
        
        # Pre-compute quantum bonuses: coefficient / visit_count
        self.quantum_table = self.config.quantum_bonus_coefficient / visit_range
        
        # Create threshold mask
        self.threshold_mask = visit_range < self.config.quantum_crossover_threshold
        
        logger.debug(f"Pre-computed quantum table for visits 1-{max_visits}")
    
    def _compile_jit_functions(self):
        """Compile JIT functions for critical paths"""
        
        @torch.jit.script
        def fast_ucb_classical(
            q_values: torch.Tensor,
            visit_counts: torch.Tensor,
            priors: torch.Tensor,
            parent_visits: float,
            c_puct: float
        ) -> torch.Tensor:
            """JIT-compiled classical UCB computation"""
            safe_visits = torch.clamp(visit_counts, min=1.0)
            sqrt_parent = math.sqrt(parent_visits)
            
            exploration = c_puct * priors * sqrt_parent / torch.sqrt(safe_visits)
            return q_values + exploration
        
        @torch.jit.script
        def fast_quantum_bonus(
            visit_counts: torch.Tensor,
            coefficient: float,
            threshold: float
        ) -> torch.Tensor:
            """JIT-compiled quantum bonus computation"""
            safe_visits = torch.clamp(visit_counts, min=1.0)
            bonus = coefficient / safe_visits
            mask = visit_counts < threshold
            return torch.where(mask, bonus, torch.zeros_like(bonus))
        
        self.jit_classical_ucb = fast_ucb_classical
        self.jit_quantum_bonus = fast_quantum_bonus
        
        logger.debug("JIT functions compiled successfully")
    
    def compute_ultra_fast_ucb(
        self,
        q_values: torch.Tensor,
        visit_counts: torch.Tensor,
        priors: torch.Tensor,
        parent_visits: float
    ) -> torch.Tensor:
        """
        Ultra-fast UCB computation with minimal quantum enhancement
        
        Optimized for maximum speed with minimal overhead.
        """
        if hasattr(self, 'jit_classical_ucb') and self.config.use_jit_compilation:
            # Use JIT-compiled classical UCB
            classical_scores = self.jit_classical_ucb(
                q_values, visit_counts, priors, parent_visits, self.config.base_c_puct
            )
            
            if self.config.enable_quantum_bonus:
                # Add minimal quantum bonus using JIT
                quantum_bonus = self.jit_quantum_bonus(
                    visit_counts, 
                    self.config.quantum_bonus_coefficient,
                    float(self.config.quantum_crossover_threshold)
                )
                return classical_scores + quantum_bonus
            else:
                return classical_scores
        
        else:
            # Fallback to pure tensor operations
            return self._tensor_optimized_ucb(q_values, visit_counts, priors, parent_visits)
    
    def _tensor_optimized_ucb(
        self,
        q_values: torch.Tensor,
        visit_counts: torch.Tensor,
        priors: torch.Tensor,
        parent_visits: float
    ) -> torch.Tensor:
        """Tensor-optimized UCB without JIT"""
        
        # Classical UCB with tensor operations
        safe_visits = torch.clamp(visit_counts, min=1.0)
        sqrt_term = torch.sqrt(parent_visits / safe_visits)
        exploration = self.config.base_c_puct * priors * sqrt_term
        classical_scores = q_values + exploration
        
        if not self.config.enable_quantum_bonus:
            return classical_scores
        
        # Ultra-fast quantum bonus using pre-computed tables
        if self.quantum_table is not None:
            # Use lookup table for speed
            visit_indices = torch.clamp(visit_counts.long() - 1, 0, len(self.quantum_table) - 1)
            quantum_bonuses = self.quantum_table[visit_indices]
            
            # Apply threshold mask
            threshold_indices = torch.clamp(visit_counts.long() - 1, 0, len(self.threshold_mask) - 1)
            mask = self.threshold_mask[threshold_indices]
            quantum_bonuses = torch.where(mask, quantum_bonuses, torch.zeros_like(quantum_bonuses))
        else:
            # Direct computation (fallback)
            quantum_bonuses = torch.where(
                visit_counts < self.config.quantum_crossover_threshold,
                self.config.quantum_bonus_coefficient / safe_visits,
                torch.zeros_like(safe_visits)
            )
        
        return classical_scores + quantum_bonuses
    
    def batch_compute_ultra_fast_ucb(
        self,
        q_values_batch: torch.Tensor,      # [batch_size, num_actions]
        visit_counts_batch: torch.Tensor,  # [batch_size, num_actions]
        priors_batch: torch.Tensor,        # [batch_size, num_actions]
        parent_visits_batch: torch.Tensor  # [batch_size]
    ) -> torch.Tensor:
        """
        Ultra-fast batch processing with pure tensor operations
        
        No loops, minimal overhead, maximum vectorization.
        """
        batch_size, num_actions = q_values_batch.shape
        
        # Vectorized classical UCB
        safe_visits = torch.clamp(visit_counts_batch, min=1.0)
        parent_expanded = parent_visits_batch.unsqueeze(-1)  # [batch_size, 1]
        
        sqrt_term = torch.sqrt(parent_expanded / safe_visits)
        exploration_terms = self.config.base_c_puct * priors_batch * sqrt_term
        classical_scores = q_values_batch + exploration_terms
        
        if not self.config.enable_quantum_bonus:
            return classical_scores
        
        # Vectorized quantum bonuses
        quantum_bonuses = torch.zeros_like(q_values_batch)
        threshold_mask = visit_counts_batch < self.config.quantum_crossover_threshold
        
        if torch.any(threshold_mask):
            coefficient = self.config.quantum_bonus_coefficient
            quantum_bonuses[threshold_mask] = coefficient / safe_visits[threshold_mask]
        
        return classical_scores + quantum_bonuses

class MinimalQuantumSelector:
    """
    Minimal quantum node selector for MCTS integration
    
    Provides the absolute minimum quantum enhancement needed for
    integration with existing MCTS while maintaining speed.
    """
    
    def __init__(self, config: UltraFastQuantumConfig):
        self.quantum_mcts = UltraFastQuantumMCTS(config)
        self.config = config
    
    def select_action(
        self,
        q_values: torch.Tensor,
        visit_counts: torch.Tensor,
        priors: torch.Tensor,
        parent_visits: int
    ) -> int:
        """Select action using ultra-fast quantum UCB"""
        
        if len(q_values.shape) == 0:
            # Single values, expand to vectors
            q_values = q_values.unsqueeze(0)
            visit_counts = visit_counts.unsqueeze(0)
            priors = priors.unsqueeze(0)
        
        ucb_scores = self.quantum_mcts.compute_ultra_fast_ucb(
            q_values, visit_counts, priors, float(parent_visits)
        )
        
        return torch.argmax(ucb_scores).item()
    
    def batch_select_actions(
        self,
        q_values_batch: torch.Tensor,
        visit_counts_batch: torch.Tensor,
        priors_batch: torch.Tensor,
        parent_visits_batch: torch.Tensor
    ) -> torch.Tensor:
        """Batch action selection for maximum throughput"""
        
        ucb_scores_batch = self.quantum_mcts.batch_compute_ultra_fast_ucb(
            q_values_batch, visit_counts_batch, priors_batch, parent_visits_batch
        )
        
        return torch.argmax(ucb_scores_batch, dim=-1)

# Factory functions for easy integration
def create_ultra_fast_quantum_mcts(
    device: str = 'cpu',
    quantum_bonus_coefficient: float = 0.02,
    quantum_crossover_threshold: int = 10,
    **kwargs
) -> UltraFastQuantumMCTS:
    """Create ultra-fast quantum MCTS with minimal overhead"""
    config = UltraFastQuantumConfig(
        device=device,
        quantum_bonus_coefficient=quantum_bonus_coefficient,
        quantum_crossover_threshold=quantum_crossover_threshold,
        **kwargs
    )
    return UltraFastQuantumMCTS(config)

def create_speed_optimized_quantum_mcts(device: str = 'cpu') -> UltraFastQuantumMCTS:
    """Create quantum MCTS optimized purely for speed"""
    config = UltraFastQuantumConfig(
        device=device,
        quantum_bonus_coefficient=0.01,  # Minimal quantum effect
        quantum_crossover_threshold=5,   # Very selective
        precompute_quantum_tables=True,
        use_jit_compilation=True,
        enable_quantum_bonus=True,
        # All other features disabled for speed
    )
    return UltraFastQuantumMCTS(config)

def create_minimal_quantum_selector(device: str = 'cpu') -> MinimalQuantumSelector:
    """Create minimal quantum selector for MCTS integration"""
    config = UltraFastQuantumConfig(
        device=device,
        quantum_bonus_coefficient=0.015,
        quantum_crossover_threshold=8,
        precompute_quantum_tables=True,
        use_jit_compilation=True,
    )
    return MinimalQuantumSelector(config)

# Export main classes
__all__ = [
    'UltraFastQuantumMCTS',
    'UltraFastQuantumConfig',
    'MinimalQuantumSelector',
    'create_ultra_fast_quantum_mcts',
    'create_speed_optimized_quantum_mcts',
    'create_minimal_quantum_selector'
]