"""Hamiltonian Structure for Quantum MCTS

This module implements the complete Hamiltonian formulation from the v4.0 theoretical framework:
H = E₀ H_diag + H_hop

Key components:
1. Diagonal Hamiltonian: H_diag = Σ L(s,a; N^pre) |s,a⟩⟨s,a|
2. Hopping Hamiltonian: H_hop = Σ κ_N |s',a'⟩⟨s,a| + h.c.
3. Energy scale E₀ = k_B T_room
4. Hopping strength κ_N = E₀ κ₀/√(N+1)

This provides the quantum mechanical foundation for MCTS dynamics.
"""

import torch
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


@dataclass 
class HamiltonianConfig:
    """Configuration for Hamiltonian dynamics"""
    
    # Energy scales
    E0: float = 1.0                      # Natural energy scale (k_B T_room)
    hbar: float = 1.0                    # Fundamental Planck constant
    
    # Hopping parameters
    kappa_0: float = 0.1                 # Bare hopping strength
    enable_hopping: bool = True          # Enable off-diagonal hopping terms
    hopping_range: int = 1               # Range of hopping connections
    
    # System parameters
    lambda_prior: float = 1.4            # Prior weight parameter
    beta_value: float = 1.0              # Value weight parameter
    epsilon_N: float = 1e-8              # Visit count regularization
    
    # Numerical parameters
    max_hilbert_dim: int = 10000         # Maximum Hilbert space dimension
    sparse_threshold: float = 0.1        # Sparsity threshold for matrix operations
    use_sparse_matrices: bool = True     # Use sparse representations
    
    # Device configuration
    device: str = 'cuda'


class HamiltonianStructure:
    """Constructs and manages the total Hamiltonian H = E₀ H_diag + H_hop"""
    
    def __init__(self, config: HamiltonianConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Cache for constructed Hamiltonians
        self._hamiltonian_cache: Dict[str, torch.Tensor] = {}
        self._cached_tree_states: Dict[str, Any] = {}
        
        # Statistics
        self.stats = {
            'hamiltonians_constructed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_construction_time': 0.0
        }
    
    def construct_total_hamiltonian(self, 
                                  edge_indices: torch.Tensor,
                                  visit_counts: torch.Tensor,
                                  priors: torch.Tensor,
                                  q_values: torch.Tensor,
                                  tree_structure: Optional[Dict] = None) -> torch.Tensor:
        """Construct complete Hamiltonian H = E₀ H_diag + H_hop
        
        Args:
            edge_indices: Tensor of edge indices [num_edges]
            visit_counts: Visit counts N^pre [num_edges]
            priors: Neural network priors P(s,a) [num_edges]
            q_values: Backed-up Q-values [num_edges]
            tree_structure: Tree connectivity information
            
        Returns:
            Total Hamiltonian matrix [dim, dim]
        """
        import time
        start_time = time.time()
        
        # Create cache key
        cache_key = self._create_cache_key(edge_indices, visit_counts, priors, q_values)
        
        if cache_key in self._hamiltonian_cache:
            self.stats['cache_hits'] += 1
            return self._hamiltonian_cache[cache_key]
        
        self.stats['cache_misses'] += 1
        
        # Construct diagonal Hamiltonian
        H_diag = self._construct_diagonal_hamiltonian(
            edge_indices, visit_counts, priors, q_values
        )
        
        # Construct hopping Hamiltonian
        if self.config.enable_hopping and tree_structure is not None:
            H_hop = self._construct_hopping_hamiltonian(
                edge_indices, visit_counts, tree_structure
            )
        else:
            # No hopping - purely diagonal
            H_hop = torch.zeros_like(H_diag)
        
        # Total Hamiltonian
        H_total = self.config.E0 * H_diag + H_hop
        
        # Cache result
        self._hamiltonian_cache[cache_key] = H_total
        
        # Update statistics
        construction_time = time.time() - start_time
        self.stats['hamiltonians_constructed'] += 1
        self.stats['average_construction_time'] = (
            (self.stats['average_construction_time'] * (self.stats['hamiltonians_constructed'] - 1) + 
             construction_time) / self.stats['hamiltonians_constructed']
        )
        
        return H_total
    
    def _construct_diagonal_hamiltonian(self,
                                      edge_indices: torch.Tensor,
                                      visit_counts: torch.Tensor, 
                                      priors: torch.Tensor,
                                      q_values: torch.Tensor) -> torch.Tensor:
        """Construct Hamiltonian from discrete Legendre transform
        
        Following the technical note:
        Discrete Lagrangian: L_N = [log N^pre + λ log P - β Q] + (κ_N/2δτ_N)(Δφ)²
        Discrete Hamiltonian: H_N = (π²/2κ_N) + V_σ(N̂)
        Potential: V_σ(N̂) = -[log N̂ + λ log P - β Q]
        
        Note: This constructs the potential part V_σ. Kinetic part (π²/2κ_N) is handled
        in hopping terms where π = -iℏ_eff ∂/∂φ
        """
        num_edges = len(edge_indices)
        H_diag = torch.zeros((num_edges, num_edges), device=self.device, dtype=torch.complex64)
        
        # Use PRE-UPDATE visit counts as specified in technical note
        # Ensure we use N^pre (current visit counts before update)
        safe_visits = visit_counts + self.config.epsilon_N
        safe_priors = torch.clamp(priors, min=1e-12)
        
        # Potential V_σ(N̂) = -[log N̂ + λ log P - β Q] from discrete Legendre transform
        info_potential = -torch.log(safe_visits)  # -log N^pre
        prior_potential = -self.config.lambda_prior * torch.log(safe_priors)  # -λ log P  
        value_potential = self.config.beta_value * q_values  # +β Q (sign flipped)
        
        # Total potential energy V_σ
        potential_energy = info_potential + prior_potential + value_potential
        
        # Diagonal Hamiltonian: H_diag = diag(V_σ(s,a))
        for i in range(num_edges):
            H_diag[i, i] = potential_energy[i]
        
        return H_diag
    
    def _construct_hopping_hamiltonian(self,
                                     edge_indices: torch.Tensor,
                                     visit_counts: torch.Tensor,
                                     tree_structure: Dict) -> torch.Tensor:
        """Construct hopping Hamiltonian from kinetic terms in discrete Legendre transform
        
        Following technical note:
        - Kinetic term from Lagrangian: (κ_N/2δτ_N)(Δφ)²  
        - After Legendre transform: (π²/2κ_N) where π = -iℏ_eff ∂/∂φ
        - Hopping strength: κ_N = κ₀/√(N_root + 1)
        - Off-diagonal terms represent kinetic coupling between neighboring edges
        """
        num_edges = len(edge_indices)
        H_hop = torch.zeros((num_edges, num_edges), device=self.device, dtype=torch.complex64)
        
        # Compute hopping strength κ_N = κ₀/√(N_root + 1) from technical note
        N_root = visit_counts.sum().item()
        kappa_N = self.config.kappa_0 / math.sqrt(N_root + 1)
        
        # Get tree connections for kinetic coupling
        connections = tree_structure.get('connections', [])
        
        for connection in connections:
            i, j = connection['from_idx'], connection['to_idx']
            
            if i < num_edges and j < num_edges:
                # Off-diagonal kinetic coupling terms
                # These represent the kinetic energy (π²/2κ_N) in position representation
                H_hop[i, j] = kappa_N
                H_hop[j, i] = kappa_N  # Hermitian conjugate
        
        return H_hop
    
    def _create_cache_key(self, 
                         edge_indices: torch.Tensor,
                         visit_counts: torch.Tensor,
                         priors: torch.Tensor, 
                         q_values: torch.Tensor) -> str:
        """Create cache key for Hamiltonian"""
        # Use hash of tensor contents
        edges_hash = hash(tuple(edge_indices.cpu().numpy()))
        visits_hash = hash(tuple(visit_counts.cpu().numpy()))
        priors_hash = hash(tuple((priors * 1000).int().cpu().numpy()))  # Quantize for hashing
        values_hash = hash(tuple((q_values * 1000).int().cpu().numpy()))
        
        return f"{edges_hash}_{visits_hash}_{priors_hash}_{values_hash}"
    
    def get_eigenspectrum(self, hamiltonian: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute eigenvalues and eigenvectors of Hamiltonian
        
        Args:
            hamiltonian: Hamiltonian matrix [dim, dim]
            
        Returns:
            Tuple of (eigenvalues [dim], eigenvectors [dim, dim])
        """
        try:
            # For Hermitian matrices, use eigh for better numerical stability
            eigenvals, eigenvecs = torch.linalg.eigh(hamiltonian)
            return eigenvals.real, eigenvecs
            
        except Exception as e:
            logger.warning(f"Eigendecomposition failed: {e}, using fallback")
            # Fallback to general eigenvalue solver
            eigenvals, eigenvecs = torch.linalg.eig(hamiltonian)
            return eigenvals, eigenvecs
    
    def validate_hamiltonian_properties(self, hamiltonian: torch.Tensor) -> Dict[str, Any]:
        """Validate quantum mechanical properties of Hamiltonian
        
        Returns:
            Dictionary of validation results
        """
        validation = {}
        
        # Check Hermiticity
        hermitian_error = torch.norm(hamiltonian - hamiltonian.conj().T).item()
        validation['is_hermitian'] = hermitian_error < 1e-10
        validation['hermitian_error'] = hermitian_error
        
        # Check energy scale
        eigenvals, _ = self.get_eigenspectrum(hamiltonian)
        validation['min_eigenvalue'] = eigenvals.min().item()
        validation['max_eigenvalue'] = eigenvals.max().item()
        validation['energy_range'] = (eigenvals.max() - eigenvals.min()).item()
        
        # Check numerical conditioning
        condition_number = torch.linalg.cond(hamiltonian).item()
        validation['condition_number'] = condition_number
        validation['well_conditioned'] = condition_number < 1e12
        
        # Check sparsity
        total_elements = hamiltonian.numel()
        nonzero_elements = torch.count_nonzero(hamiltonian).item()
        sparsity = 1.0 - nonzero_elements / total_elements
        validation['sparsity'] = sparsity
        validation['sparse_eligible'] = sparsity > self.config.sparse_threshold
        
        return validation
    
    def compute_time_evolution_operator(self,
                                      hamiltonian: torch.Tensor,
                                      time: float) -> torch.Tensor:
        """Compute time evolution operator U(t) = exp(-iHt/ℏ)
        
        Args:
            hamiltonian: Hamiltonian matrix [dim, dim]
            time: Evolution time
            
        Returns:
            Unitary time evolution operator [dim, dim]
        """
        # Exponentiate: U(t) = exp(-iHt/ℏ)
        exponent = -1j * hamiltonian * time / self.config.hbar
        
        # Use matrix exponential
        try:
            U = torch.linalg.matrix_exp(exponent)
            return U
        except Exception as e:
            logger.warning(f"Matrix exponential failed: {e}, using diagonalization")
            
            # Fallback: diagonalize and exponentiate eigenvalues
            eigenvals, eigenvecs = self.get_eigenspectrum(hamiltonian)
            exp_eigenvals = torch.exp(-1j * eigenvals * time / self.config.hbar)
            U = torch.matmul(eigenvecs, torch.matmul(torch.diag(exp_eigenvals), eigenvecs.conj().T))
            return U
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Hamiltonian construction statistics"""
        stats = self.stats.copy()
        stats['cache_hit_rate'] = (
            self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
            if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0.0
        )
        stats['cache_size'] = len(self._hamiltonian_cache)
        return stats
    
    def clear_cache(self):
        """Clear Hamiltonian cache"""
        self._hamiltonian_cache.clear()
        self._cached_tree_states.clear()


def create_hamiltonian_structure(config: Optional[HamiltonianConfig] = None) -> HamiltonianStructure:
    """Factory function to create HamiltonianStructure with default configuration"""
    if config is None:
        config = HamiltonianConfig()
    
    return HamiltonianStructure(config)


# Export main classes and functions
__all__ = [
    'HamiltonianStructure',
    'HamiltonianConfig', 
    'create_hamiltonian_structure'
]