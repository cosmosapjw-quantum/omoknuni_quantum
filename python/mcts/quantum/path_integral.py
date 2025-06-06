"""Path integral formulation for MCTS with optimized performance

This implementation uses pre-computed tables and lookup operations
to achieve < 2x overhead compared to classical MCTS.
"""

import torch
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
import logging
import math

logger = logging.getLogger(__name__)

@dataclass 
class PathIntegralConfig:
    """Configuration for path integral formulation"""
    # Core parameters
    hbar_eff: float = 0.1
    temperature: float = 1.0
    lambda_qft: float = 1.0
    
    # Optimization parameters
    use_lookup_tables: bool = True
    table_size: int = 10000
    max_path_length: int = 50
    cache_path_actions: bool = True
    use_mixed_precision: bool = True
    
    # Batching
    batch_size: int = 256
    
    # Device
    device: str = 'cuda'

class PrecomputedTables:
    """Pre-computed lookup tables for fast path integral evaluation"""
    
    def __init__(self, config: PathIntegralConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        logger.info("Pre-computing path integral tables...")
        
        # Pre-compute exponential decay factors for different path lengths
        self.length_factors = self._compute_length_factors()
        
        # Pre-compute Boltzmann factors for different action values
        self.boltzmann_table = self._compute_boltzmann_table()
        
        # Pre-compute quantum correction factors
        self.quantum_corrections = self._compute_quantum_corrections()
        
        # Pre-compute phase factors for interference
        self.phase_table = self._compute_phase_table()
        
        # Cache for path actions
        self.action_cache = {} if config.cache_path_actions else None
        
        logger.info("Pre-computation complete")
        
    def _compute_length_factors(self) -> torch.Tensor:
        """Pre-compute exp(-lambda * length) for different lengths"""
        lengths = torch.arange(0, self.config.max_path_length, device=self.device)
        factors = torch.exp(-self.config.lambda_qft * lengths / self.config.hbar_eff)
        return factors
        
    def _compute_boltzmann_table(self) -> torch.Tensor:
        """Pre-compute Boltzmann factors for discretized actions"""
        # Discretize action values into bins
        action_bins = torch.linspace(-10, 10, self.config.table_size, device=self.device)
        boltzmann = torch.exp(-action_bins / self.config.temperature)
        return boltzmann
        
    def _compute_quantum_corrections(self) -> torch.Tensor:
        """Pre-compute quantum correction factors"""
        # One-loop corrections for different visit counts
        visits = torch.arange(0, 1000, device=self.device).float()
        
        # Effective Planck constant scales with visits
        hbar_eff = self.config.hbar_eff / torch.sqrt(visits + 1)
        
        # Quantum correction: 1 + hbar_eff^2 * correction_term
        corrections = 1.0 + hbar_eff**2 * 0.1  # Simplified correction
        
        return corrections
        
    def _compute_phase_table(self) -> torch.Tensor:
        """Pre-compute phase factors for interference"""
        # Phase factors for different path similarities
        similarities = torch.linspace(0, 1, 1000, device=self.device)
        phases = torch.cos(math.pi * similarities)  # Constructive/destructive interference
        return phases

class PathIntegral:
    """Path integral implementation with pre-computed tables for performance"""
    
    def __init__(self, config: PathIntegralConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.tables = PrecomputedTables(config)
        
        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'paths_evaluated': 0
        }
        
    def compute_path_integral_batch(
        self,
        paths: torch.Tensor,
        values: torch.Tensor,
        visits: torch.Tensor,
        tree: Optional[Any] = None
    ) -> torch.Tensor:
        """Compute path integral for a batch of paths using pre-computed tables
        
        Args:
            paths: Path tensor [batch_size, max_depth]
            values: Value estimates [batch_size]
            visits: Visit counts [batch_size]
            tree: Optional tree structure for additional info
            
        Returns:
            Path integral values [batch_size]
        """
        batch_size = paths.shape[0]
        self.stats['paths_evaluated'] += batch_size
        
        # Get path lengths (vectorized)
        path_lengths = (paths >= 0).sum(dim=1)
        
        # Look up length factors
        length_factors = self._lookup_length_factors(path_lengths)
        
        # Compute or retrieve cached actions
        if self.config.cache_path_actions and self.tables.action_cache is not None:
            actions = self._get_cached_actions(paths, values, visits)
        else:
            actions = self._compute_actions_vectorized(paths, values, visits)
            
        # Discretize actions for table lookup
        action_indices = self._discretize_actions(actions)
        
        # Look up Boltzmann factors
        boltzmann_factors = self._lookup_boltzmann(action_indices)
        
        # Look up quantum corrections based on visits
        quantum_corrections = self._lookup_quantum_corrections(visits)
        
        # Compute path integrals with all factors
        path_integrals = length_factors * boltzmann_factors * quantum_corrections
        
        # Apply interference if we have multiple paths
        if batch_size > 1:
            path_integrals = self._apply_interference_fast(paths, path_integrals)
            
        return path_integrals
        
    def _lookup_length_factors(self, lengths: torch.Tensor) -> torch.Tensor:
        """Fast lookup of pre-computed length factors"""
        # Clamp to valid range
        lengths = torch.clamp(lengths, 0, self.config.max_path_length - 1)
        return self.tables.length_factors[lengths]
        
    def _discretize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Discretize continuous actions to table indices"""
        # Map action values to [0, table_size-1]
        normalized = (actions + 10) / 20  # Assuming actions in [-10, 10]
        indices = (normalized * (self.config.table_size - 1)).long()
        indices = torch.clamp(indices, 0, self.config.table_size - 1)
        return indices
        
    def _lookup_boltzmann(self, indices: torch.Tensor) -> torch.Tensor:
        """Fast lookup of Boltzmann factors"""
        return self.tables.boltzmann_table[indices]
        
    def _lookup_quantum_corrections(self, visits: torch.Tensor) -> torch.Tensor:
        """Fast lookup of quantum corrections"""
        visit_indices = torch.clamp(visits.long(), 0, 999)
        return self.tables.quantum_corrections[visit_indices]
        
    def _get_cached_actions(
        self, 
        paths: torch.Tensor, 
        values: torch.Tensor,
        visits: torch.Tensor
    ) -> torch.Tensor:
        """Get actions from cache or compute and cache them"""
        # Create cache key from path hashes
        path_hashes = []
        for path in paths:
            # Simple hash: sum of node indices
            path_hash = path[path >= 0].sum().item()
            path_hashes.append(path_hash)
            
        actions = torch.zeros(len(paths), device=self.device)
        uncached_mask = torch.zeros(len(paths), dtype=torch.bool, device=self.device)
        
        # Check cache
        for i, h in enumerate(path_hashes):
            if h in self.tables.action_cache:
                actions[i] = self.tables.action_cache[h]
                self.stats['cache_hits'] += 1
            else:
                uncached_mask[i] = True
                self.stats['cache_misses'] += 1
                
        # Compute uncached actions
        if uncached_mask.any():
            uncached_actions = self._compute_actions_vectorized(
                paths[uncached_mask],
                values[uncached_mask],
                visits[uncached_mask]
            )
            actions[uncached_mask] = uncached_actions
            
            # Update cache
            uncached_indices = torch.where(uncached_mask)[0]
            for i, action in zip(uncached_indices, uncached_actions):
                self.tables.action_cache[path_hashes[i.item()]] = action.item()
                
        return actions
        
    def _compute_actions_vectorized(
        self,
        paths: torch.Tensor,
        values: torch.Tensor, 
        visits: torch.Tensor
    ) -> torch.Tensor:
        """Compute classical actions for paths (vectorized)"""
        # Simplified action: combination of path length and value
        path_lengths = (paths >= 0).sum(dim=1).float()
        
        # Action = -value * path_length / sqrt(visits + 1)
        actions = -values * path_lengths / torch.sqrt(visits + 1)
        
        return actions
        
    def _apply_interference_fast(
        self,
        paths: torch.Tensor,
        amplitudes: torch.Tensor
    ) -> torch.Tensor:
        """Apply quantum interference using pre-computed phase table"""
        batch_size = paths.shape[0]
        
        if batch_size <= 1:
            return amplitudes
            
        # Compute path similarities using vectorized operations
        # Simple similarity: fraction of shared nodes
        path_vectors = (paths >= 0).float()  # Binary vectors
        
        # Compute pairwise dot products
        similarities = torch.matmul(path_vectors, path_vectors.t())
        path_lengths = path_vectors.sum(dim=1, keepdim=True)
        
        # Normalize by path lengths
        normalizer = torch.sqrt(path_lengths * path_lengths.t())
        similarities = similarities / (normalizer + 1e-8)
        
        # Look up phase factors
        similarity_indices = (similarities * 999).long()
        similarity_indices = torch.clamp(similarity_indices, 0, 999)
        phase_matrix = self.tables.phase_table[similarity_indices]
        
        # Apply interference
        # Paths interfere with each other based on similarity
        interference = torch.matmul(phase_matrix - torch.eye(batch_size, device=self.device),
                                  amplitudes.unsqueeze(1)).squeeze()
        
        # Modulate amplitudes with interference
        modulated = amplitudes + 0.1 * interference
        
        # Ensure positive amplitudes
        return torch.abs(modulated)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_accesses = self.stats['cache_hits'] + self.stats['cache_misses']
        cache_hit_rate = self.stats['cache_hits'] / total_accesses if total_accesses > 0 else 0
        
        return {
            'paths_evaluated': self.stats['paths_evaluated'],
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.tables.action_cache) if self.tables.action_cache else 0
        }
        
    def clear_cache(self):
        """Clear action cache"""
        if self.tables.action_cache is not None:
            self.tables.action_cache.clear()
            self.stats['cache_hits'] = 0
            self.stats['cache_misses'] = 0