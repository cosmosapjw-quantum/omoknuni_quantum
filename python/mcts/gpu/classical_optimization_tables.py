"""
Classical Optimization Tables for Fair MCTS Comparison
=====================================================

This module provides precomputed lookup tables and JIT-compiled kernels
for classical MCTS, ensuring optimization parity with quantum-inspired MCTS.

The tables provide equivalent optimizations to quantum's uncertainty_table
and hbar_factors, enabling fair algorithmic comparison.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClassicalOptimizationConfig:
    """Configuration for classical optimization tables"""
    max_visits: int = 5000           # Maximum visit count to precompute
    c_puct: float = 1.4             # UCB exploration constant
    device: str = 'cuda'            # Computation device
    dtype: torch.dtype = torch.float32  # Data type for tables
    
    # Performance tuning
    cache_parent_visits: bool = True     # Cache parent visit sqrt values
    enable_exploration_cache: bool = True # Cache exploration denominators
    enable_jit_compilation: bool = True   # Use JIT-compiled kernels


class ClassicalOptimizationTables:
    """
    Precomputed lookup tables for classical UCB computation.
    
    This provides equivalent optimization to quantum MCTS's uncertainty_table,
    enabling fair performance comparison between classical and quantum approaches.
    
    Features:
    - O(1) lookup for common UCB terms
    - JIT-compiled optimized computation kernels
    - Memory-efficient tensor storage
    - GPU-optimized data layout
    """
    
    def __init__(self, config: ClassicalOptimizationConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Initialize lookup tables
        self._build_optimization_tables()
        
        # Performance statistics
        self.stats = {
            'table_hits': 0,
            'table_misses': 0,
            'jit_compilations': 0,
            'memory_usage_mb': 0.0
        }
        
        logger.debug(f"Classical optimization tables initialized with {config.max_visits} entries")
        logger.debug(f"Memory usage: {self._estimate_memory_usage():.2f} MB")
        
    def _build_optimization_tables(self):
        """Build all precomputed lookup tables"""
        # Visit count range for precomputation
        visits = torch.arange(0, self.config.max_visits + 1, 
                            dtype=self.config.dtype, device=self.device)
        
        # Table 1: Square root values for parent visits
        # sqrt(parent_visits + 1) - commonly used in UCB exploration term
        self.sqrt_table = torch.sqrt(visits + 1)
        
        # Table 2: Exploration denominators 
        # 1 / (1 + child_visits) - denominator in UCB exploration term
        self.exploration_denom = 1.0 / (1.0 + visits)
        
        # Table 3: C_PUCT scaled exploration factors
        # c_puct / (1 + child_visits) - complete exploration scaling
        self.c_puct_factors = self.config.c_puct * self.exploration_denom
        
        # Table 4: Common visit count powers for advanced UCB variants
        # visits^(0.5), visits^(0.25) etc. for research variants
        self.visit_powers = {
            0.5: torch.pow(visits + 1e-8, 0.5),    # sqrt
            0.25: torch.pow(visits + 1e-8, 0.25),  # fourth root
            0.75: torch.pow(visits + 1e-8, 0.75),  # three-quarter power
        }
        
        # Table 5: Inverse visit count terms
        # 1/sqrt(visits), 1/visits etc. for normalization
        self.inverse_visit_terms = {
            'inv_sqrt': 1.0 / torch.sqrt(visits + 1),
            'inv_linear': 1.0 / (visits + 1),
            'inv_log': 1.0 / torch.log(visits + 2),  # log normalization
        }
        
        # Table 6: Temperature-adjusted factors (for temperature != 1.0)
        self.temperature_factors = {}
        for temp in [0.1, 0.5, 1.0, 1.5, 2.0]:
            if temp > 0:
                self.temperature_factors[temp] = torch.pow(visits + 1e-8, 1.0 / temp - 1.0)
        
        # Mark tables as ready
        self._tables_ready = True
        
    def get_sqrt_parent_visits(self, parent_visits: torch.Tensor) -> torch.Tensor:
        """
        Fast lookup for sqrt(parent_visits + 1) using precomputed table
        
        Args:
            parent_visits: Parent visit counts
            
        Returns:
            Square root values with table lookup optimization
        """
        self.stats['table_hits'] += len(parent_visits)
        
        # Use table lookup for values within range
        in_range = (parent_visits >= 0) & (parent_visits <= self.config.max_visits)
        result = torch.zeros_like(parent_visits, dtype=self.config.dtype)
        
        if in_range.any():
            # Table lookup for in-range values
            table_indices = parent_visits[in_range].long()
            result[in_range] = self.sqrt_table[table_indices]
        
        # Direct computation for out-of-range values
        out_of_range = ~in_range
        if out_of_range.any():
            self.stats['table_misses'] += out_of_range.sum().item()
            result[out_of_range] = torch.sqrt(parent_visits[out_of_range].float() + 1)
            
        return result
    
    def get_exploration_factors(self, child_visits: torch.Tensor) -> torch.Tensor:
        """
        Fast lookup for c_puct / (1 + child_visits) using precomputed table
        
        Args:
            child_visits: Child visit counts
            
        Returns:
            Exploration factors with table lookup optimization
        """
        self.stats['table_hits'] += len(child_visits)
        
        # Use table lookup for values within range
        in_range = (child_visits >= 0) & (child_visits <= self.config.max_visits)
        result = torch.zeros_like(child_visits, dtype=self.config.dtype)
        
        if in_range.any():
            # Table lookup for in-range values
            table_indices = child_visits[in_range].long()
            result[in_range] = self.c_puct_factors[table_indices]
        
        # Direct computation for out-of-range values
        out_of_range = ~in_range
        if out_of_range.any():
            self.stats['table_misses'] += out_of_range.sum().item()
            result[out_of_range] = self.config.c_puct / (1.0 + child_visits[out_of_range].float())
            
        return result
    
    def get_inverse_visit_factors(self, visits: torch.Tensor, factor_type: str = 'inv_linear') -> torch.Tensor:
        """
        Fast lookup for inverse visit count factors
        
        Args:
            visits: Visit counts
            factor_type: Type of inverse factor ('inv_sqrt', 'inv_linear', 'inv_log')
            
        Returns:
            Inverse factors with table lookup optimization
        """
        if factor_type not in self.inverse_visit_terms:
            raise ValueError(f"Unknown factor type: {factor_type}")
            
        self.stats['table_hits'] += len(visits)
        table = self.inverse_visit_terms[factor_type]
        
        # Use table lookup for values within range
        in_range = (visits >= 0) & (visits <= self.config.max_visits)
        result = torch.zeros_like(visits, dtype=self.config.dtype)
        
        if in_range.any():
            # Table lookup for in-range values
            table_indices = visits[in_range].long()
            result[in_range] = table[table_indices]
        
        # Direct computation for out-of-range values
        out_of_range = ~in_range
        if out_of_range.any():
            self.stats['table_misses'] += out_of_range.sum().item()
            if factor_type == 'inv_sqrt':
                result[out_of_range] = 1.0 / torch.sqrt(visits[out_of_range].float() + 1)
            elif factor_type == 'inv_linear':
                result[out_of_range] = 1.0 / (visits[out_of_range].float() + 1)
            elif factor_type == 'inv_log':
                result[out_of_range] = 1.0 / torch.log(visits[out_of_range].float() + 2)
                
        return result
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        total_elements = 0
        
        # Count elements in all tables
        total_elements += self.sqrt_table.numel()
        total_elements += self.exploration_denom.numel()
        total_elements += self.c_puct_factors.numel()
        
        for table in self.visit_powers.values():
            total_elements += table.numel()
            
        for table in self.inverse_visit_terms.values():
            total_elements += table.numel()
            
        for table in self.temperature_factors.values():
            total_elements += table.numel()
        
        # Estimate memory (4 bytes per float32)
        memory_mb = (total_elements * 4) / (1024 * 1024)
        self.stats['memory_usage_mb'] = memory_mb
        
        return memory_mb
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_queries = self.stats['table_hits'] + self.stats['table_misses']
        hit_rate = (self.stats['table_hits'] / total_queries * 100) if total_queries > 0 else 0
        
        return {
            **self.stats,
            'hit_rate_percent': hit_rate,
            'table_size': self.config.max_visits,
            'device': str(self.device)
        }


# JIT-compiled kernels for maximum performance
@torch.jit.script
def classical_ucb_optimized_jit(
    q_values: torch.Tensor,
    child_visits: torch.Tensor,
    parent_visits: torch.Tensor,
    priors: torch.Tensor,
    sqrt_table: torch.Tensor,
    exploration_factors: torch.Tensor,
    max_table_size: int
) -> torch.Tensor:
    """
    JIT-compiled classical UCB computation with lookup tables
    
    This provides equivalent optimization to quantum's JIT kernels,
    ensuring fair performance comparison.
    
    Args:
        q_values: Q-values for each action
        child_visits: Visit counts for each child
        parent_visits: Visit counts for parent nodes  
        priors: Prior probabilities
        sqrt_table: Precomputed sqrt table
        exploration_factors: Precomputed c_puct factors
        max_table_size: Maximum table size for bounds checking
        
    Returns:
        UCB scores for each action
    """
    # Use lookup tables for optimized computation
    batch_size = q_values.shape[0]
    device = q_values.device
    
    # Get sqrt of parent visits using table lookup
    parent_sqrt = torch.zeros_like(parent_visits, dtype=torch.float32)
    in_range_parent = (parent_visits >= 0) & (parent_visits < max_table_size)
    
    # Table lookup for in-range parent visits
    if in_range_parent.any():
        parent_indices = parent_visits[in_range_parent].long()
        parent_sqrt[in_range_parent] = sqrt_table[parent_indices]
    
    # Direct computation for out-of-range parent visits
    out_range_parent = ~in_range_parent
    if out_range_parent.any():
        parent_sqrt[out_range_parent] = torch.sqrt(parent_visits[out_range_parent].float() + 1)
    
    # Get exploration factors using table lookup
    exploration = torch.zeros_like(child_visits, dtype=torch.float32)
    in_range_child = (child_visits >= 0) & (child_visits < max_table_size)
    
    # Table lookup for in-range child visits
    if in_range_child.any():
        child_indices = child_visits[in_range_child].long()
        exploration[in_range_child] = exploration_factors[child_indices]
    
    # Direct computation for out-of-range child visits
    out_range_child = ~in_range_child
    if out_range_child.any():
        exploration[out_range_child] = 1.4 / (1.0 + child_visits[out_range_child].float())
    
    # Compute final UCB scores
    # UCB = Q + c_puct * prior * sqrt(parent_visits) / (1 + child_visits)
    ucb_exploration = priors * parent_sqrt.unsqueeze(-1) * exploration
    ucb_scores = q_values + ucb_exploration
    
    return ucb_scores


@torch.jit.script  
def classical_batch_ucb_jit(
    batch_q_values: torch.Tensor,
    batch_child_visits: torch.Tensor,
    batch_parent_visits: torch.Tensor,
    batch_priors: torch.Tensor,
    sqrt_table: torch.Tensor,
    exploration_factors: torch.Tensor,
    max_table_size: int
) -> torch.Tensor:
    """
    Vectorized batch UCB computation with lookup tables
    
    Args:
        batch_q_values: [batch_size, num_actions] Q-values
        batch_child_visits: [batch_size, num_actions] child visit counts
        batch_parent_visits: [batch_size] parent visit counts
        batch_priors: [batch_size, num_actions] prior probabilities
        sqrt_table: Precomputed sqrt lookup table
        exploration_factors: Precomputed exploration factors
        max_table_size: Maximum table size
        
    Returns:
        [batch_size, num_actions] UCB scores
    """
    batch_size, num_actions = batch_q_values.shape
    
    # Vectorized parent sqrt computation using table lookup
    parent_sqrt = torch.zeros_like(batch_parent_visits, dtype=torch.float32)
    in_range = (batch_parent_visits >= 0) & (batch_parent_visits < max_table_size)
    
    if in_range.any():
        indices = batch_parent_visits[in_range].long()
        parent_sqrt[in_range] = sqrt_table[indices]
    
    out_range = ~in_range
    if out_range.any():
        parent_sqrt[out_range] = torch.sqrt(batch_parent_visits[out_range].float() + 1)
    
    # Expand parent sqrt for broadcasting
    parent_sqrt_expanded = parent_sqrt.unsqueeze(1)  # [batch_size, 1]
    
    # Vectorized exploration factor computation using table lookup
    exploration = torch.zeros_like(batch_child_visits, dtype=torch.float32)
    child_in_range = (batch_child_visits >= 0) & (batch_child_visits < max_table_size)
    
    if child_in_range.any():
        child_indices = batch_child_visits[child_in_range].long()
        exploration[child_in_range] = exploration_factors[child_indices]
    
    child_out_range = ~child_in_range
    if child_out_range.any():
        exploration[child_out_range] = 1.4 / (1.0 + batch_child_visits[child_out_range].float())
    
    # Compute vectorized UCB scores
    ucb_exploration = batch_priors * parent_sqrt_expanded * exploration
    ucb_scores = batch_q_values + ucb_exploration
    
    return ucb_scores


def create_classical_optimization_tables(config: Optional[ClassicalOptimizationConfig] = None) -> ClassicalOptimizationTables:
    """Factory function for creating classical optimization tables"""
    if config is None:
        config = ClassicalOptimizationConfig()
    
    return ClassicalOptimizationTables(config)


# Export main classes and functions
__all__ = [
    'ClassicalOptimizationTables',
    'ClassicalOptimizationConfig', 
    'classical_ucb_optimized_jit',
    'classical_batch_ucb_jit',
    'create_classical_optimization_tables'
]