"""
Decoherence Engine for QFT-MCTS
===============================

This module implements quantum decoherence dynamics that naturally explain
the emergence of classical MCTS behavior from quantum superpositions.

Key Features:
- Density matrix evolution with master equation
- Lindblad decoherence operators based on visit count differences
- Pointer state identification (classically stable states)
- GPU-accelerated density matrix computations
- Automatic quantum→classical transition

Based on: docs/qft-mcts-math-foundations.md Section 3.2
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class DecoherenceConfig:
    """Configuration for decoherence engine"""
    # Physical parameters
    base_decoherence_rate: float = 1.0    # λ - system-environment coupling
    hbar: float = 1.0                     # Reduced Planck constant
    temperature: float = 1.0              # Environment temperature
    
    # Decoherence model
    visit_count_sensitivity: float = 1.0  # How strongly visit differences drive decoherence
    pointer_state_threshold: float = 0.9  # Threshold for pointer state identification
    
    # Numerical parameters  
    dt: float = 0.01                      # Time step for evolution
    max_evolution_time: float = 10.0      # Maximum evolution time
    convergence_threshold: float = 1e-6   # Convergence criterion
    matrix_regularization: float = 1e-8   # Numerical stability
    
    # GPU optimization
    use_sparse_matrices: bool = True      # Use sparse matrices for efficiency
    chunk_size: int = 1024               # Chunk size for large matrices
    use_mixed_precision: bool = True      # FP16/FP32 optimization


class DecoherenceOperators:
    """
    Lindblad operators that describe decoherence processes
    
    These operators encode how the environment monitors the tree search
    process, leading to classical behavior through selective decoherence.
    """
    
    def __init__(self, config: DecoherenceConfig, device: torch.device):
        self.config = config
        self.device = device
        
    def compute_lindblad_operators(
        self, 
        visit_counts: torch.Tensor,
        tree_structure: Optional[Dict[str, torch.Tensor]] = None
    ) -> List[torch.Tensor]:
        """
        Compute Lindblad operators L_k for decoherence
        
        The Lindblad operators represent how the environment selectively
        monitors different nodes based on their visit count differences.
        
        Args:
            visit_counts: Tensor of shape (num_nodes,) with visit counts
            tree_structure: Optional tree connectivity information
            
        Returns:
            List of Lindblad operators as tensors
        """
        num_nodes = visit_counts.shape[0]
        operators = []
        
        # Create visit count difference operator
        # This preferentially decoheres superpositions of nodes with different visit counts
        visit_diff_operator = self._create_visit_difference_operator(visit_counts)
        operators.append(visit_diff_operator)
        
        # Add path-based decoherence operators if tree structure is available
        if tree_structure is not None:
            path_operators = self._create_path_decoherence_operators(
                visit_counts, tree_structure
            )
            operators.extend(path_operators)
        
        # Add thermal decoherence (universal)
        thermal_operator = self._create_thermal_operator(visit_counts)
        operators.append(thermal_operator)
        
        return operators
    
    def _create_visit_difference_operator(self, visit_counts: torch.Tensor) -> torch.Tensor:
        """
        Create operator that decoheres based on visit count differences
        
        This is the primary decoherence mechanism: states with very different
        visit counts decohere rapidly, while similar states remain coherent.
        """
        num_nodes = visit_counts.shape[0]
        
        # Create diagonal operator based on visit counts
        # L_visit = diag(√N_i) - implements "which path" measurement
        visit_sqrt = torch.sqrt(visit_counts + self.config.matrix_regularization)
        operator = torch.diag(visit_sqrt)
        
        return operator.to(self.device)
    
    def _create_path_decoherence_operators(
        self, 
        visit_counts: torch.Tensor,
        tree_structure: Dict[str, torch.Tensor]
    ) -> List[torch.Tensor]:
        """Create operators based on tree path structure"""
        operators = []
        num_nodes = visit_counts.shape[0]
        
        # Extract tree connectivity if available
        if 'children' in tree_structure:
            children = tree_structure['children']
            
            # Create operators that distinguish parent-child relationships
            for parent in range(min(num_nodes, children.shape[0])):
                parent_visits = visit_counts[parent]
                
                if len(children.shape) > 1:
                    node_children = children[parent][children[parent] >= 0]
                else:
                    continue
                    
                if len(node_children) == 0:
                    continue
                
                # Create operator that measures "which child was chosen"
                child_operator = torch.zeros((num_nodes, num_nodes), device=self.device)
                
                for child in node_children:
                    if child < num_nodes:
                        child_visits = visit_counts[child]
                        
                        # Strength proportional to visit difference
                        strength = torch.abs(parent_visits - child_visits) / (
                            torch.max(parent_visits, child_visits) + 1e-8
                        )
                        
                        child_operator[child, child] = strength
                
                if torch.any(child_operator > 0):
                    operators.append(child_operator)
        
        return operators
    
    def _create_thermal_operator(self, visit_counts: torch.Tensor) -> torch.Tensor:
        """Create thermal decoherence operator"""
        num_nodes = visit_counts.shape[0]
        
        # Thermal decoherence scales with energy (visit count)
        thermal_rates = torch.sqrt(visit_counts / self.config.temperature)
        thermal_operator = torch.diag(thermal_rates)
        
        return thermal_operator.to(self.device)
    
    def compute_decoherence_rates(
        self, 
        visit_counts: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise decoherence rates Γ_ij
        
        From theory: Γ_ij = λ|N_i - N_j|/max(N_i, N_j)
        """
        num_nodes = visit_counts.shape[0]
        
        # Create pairwise visit difference matrix
        visit_i = visit_counts.unsqueeze(1)  # Shape: (num_nodes, 1)
        visit_j = visit_counts.unsqueeze(0)  # Shape: (1, num_nodes)
        
        # Compute |N_i - N_j|
        visit_diff = torch.abs(visit_i - visit_j)
        
        # Compute max(N_i, N_j)
        visit_max = torch.max(visit_i, visit_j)
        
        # Decoherence rate: Γ_ij = λ|N_i - N_j|/max(N_i, N_j)
        rates = self.config.base_decoherence_rate * visit_diff / (
            visit_max + self.config.matrix_regularization
        )
        
        return rates


class DensityMatrixEvolution:
    """
    Handles density matrix evolution under master equation
    
    Evolves ρ according to: dρ/dt = -i[H,ρ]/ℏ + D[ρ] + J[ρ]
    where D[ρ] is the decoherence superoperator.
    """
    
    def __init__(self, config: DecoherenceConfig, device: torch.device):
        self.config = config
        self.device = device
        self.operators = DecoherenceOperators(config, device)
        
    def evolve_density_matrix(
        self,
        rho: torch.Tensor,
        hamiltonian: torch.Tensor,
        visit_counts: torch.Tensor,
        dt: Optional[float] = None,
        tree_structure: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Evolve density matrix for one time step
        
        Args:
            rho: Current density matrix (num_nodes, num_nodes)
            hamiltonian: System Hamiltonian  
            visit_counts: Current visit counts
            dt: Time step (uses config default if None)
            tree_structure: Optional tree connectivity
            
        Returns:
            Evolved density matrix
        """
        if dt is None:
            dt = self.config.dt
            
        # Get Lindblad operators for current state
        lindblad_ops = self.operators.compute_lindblad_operators(
            visit_counts, tree_structure
        )
        
        # Compute coherent evolution: -i[H,ρ]/ℏ
        coherent_term = self._compute_coherent_evolution(rho, hamiltonian)
        
        # Compute decoherence term: D[ρ]
        decoherence_term = self._compute_decoherence_term(rho, lindblad_ops)
        
        # Total evolution
        drho_dt = coherent_term + decoherence_term
        
        # Update density matrix
        rho_new = rho + dt * drho_dt
        
        # Ensure density matrix properties
        rho_new = self._ensure_density_matrix_properties(rho_new)
        
        return rho_new
    
    def _compute_coherent_evolution(
        self, 
        rho: torch.Tensor, 
        hamiltonian: torch.Tensor
    ) -> torch.Tensor:
        """Compute coherent evolution term -i[H,ρ]/ℏ"""
        
        # Commutator [H,ρ] = Hρ - ρH
        commutator = torch.matmul(hamiltonian, rho) - torch.matmul(rho, hamiltonian)
        
        # -i[H,ρ]/ℏ (assuming imaginary unit is absorbed into coefficient)
        coherent_term = -commutator / self.config.hbar
        
        return coherent_term
    
    def _compute_decoherence_term(
        self, 
        rho: torch.Tensor, 
        lindblad_ops: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute Lindblad decoherence term
        
        D[ρ] = Σ_k γ_k (L_k ρ L_k† - {L_k†L_k, ρ}/2)
        """
        decoherence_term = torch.zeros_like(rho)
        
        for L_k in lindblad_ops:
            L_k_dag = L_k.conj().transpose(-2, -1)
            
            # L_k ρ L_k†
            dissipator = torch.matmul(torch.matmul(L_k, rho), L_k_dag)
            
            # L_k†L_k
            L_dag_L = torch.matmul(L_k_dag, L_k)
            
            # {L_k†L_k, ρ}/2 = (L_k†L_k ρ + ρ L_k†L_k)/2
            anticommutator = 0.5 * (
                torch.matmul(L_dag_L, rho) + torch.matmul(rho, L_dag_L)
            )
            
            # Add contribution
            decoherence_term += dissipator - anticommutator
        
        return decoherence_term
    
    def _ensure_density_matrix_properties(self, rho: torch.Tensor) -> torch.Tensor:
        """Ensure density matrix remains valid (Hermitian, positive, trace=1)"""
        
        # Make Hermitian
        rho = 0.5 * (rho + rho.conj().transpose(-2, -1))
        
        # Ensure positive semidefinite (eigenvalue decomposition)
        eigenvals, eigenvecs = torch.linalg.eigh(rho)
        eigenvals = torch.clamp(eigenvals, min=0.0)
        rho = torch.matmul(eigenvecs, torch.matmul(torch.diag(eigenvals), eigenvecs.conj().transpose(-2, -1)))
        
        # Normalize trace
        trace = torch.trace(rho)
        if trace > self.config.matrix_regularization:
            rho = rho / trace
        
        return rho
    
    def evolve_to_steady_state(
        self,
        initial_rho: torch.Tensor,
        hamiltonian: torch.Tensor,
        visit_counts: torch.Tensor,
        max_time: Optional[float] = None,
        tree_structure: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Evolve density matrix to steady state
        
        Returns:
            Tuple of (final_rho, evolution_time)
        """
        if max_time is None:
            max_time = self.config.max_evolution_time
            
        rho = initial_rho.clone()
        t = 0.0
        
        while t < max_time:
            # Store previous state
            rho_prev = rho.clone()
            
            # Evolve one step
            rho = self.evolve_density_matrix(
                rho, hamiltonian, visit_counts, tree_structure=tree_structure
            )
            
            # Check convergence
            diff = torch.norm(rho - rho_prev).item()
            if diff < self.config.convergence_threshold:
                logger.debug(f"Density matrix converged at t={t:.3f}")
                break
                
            t += self.config.dt
        
        return rho, t


class PointerStateAnalyzer:
    """
    Analyzes pointer states - the classically stable states that survive decoherence
    
    From theory: pointer states are eigenstates of the monitored observable
    (visit count operator in our case).
    """
    
    def __init__(self, config: DecoherenceConfig, device: torch.device):
        self.config = config
        self.device = device
        
    def identify_pointer_states(
        self, 
        rho: torch.Tensor,
        visit_counts: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Identify pointer states from density matrix
        
        Pointer states are the classical states that the quantum system
        naturally selects through decoherence.
        """
        # Diagonalize density matrix
        eigenvals, eigenvecs = torch.linalg.eigh(rho)
        
        # Sort by eigenvalue (population)
        sorted_indices = torch.argsort(eigenvals, descending=True)
        eigenvals = eigenvals[sorted_indices]
        eigenvecs = eigenvecs[:, sorted_indices]
        
        # Identify pointer states (high population states)
        pointer_mask = eigenvals > self.config.pointer_state_threshold * eigenvals[0]
        
        pointer_states = {
            'eigenvalues': eigenvals[pointer_mask],
            'eigenvectors': eigenvecs[:, pointer_mask],
            'populations': eigenvals[pointer_mask],
            'num_pointer_states': pointer_mask.sum().item()
        }
        
        # Analyze classical character
        classical_fidelity = self._compute_classical_fidelity(
            eigenvecs[:, pointer_mask], visit_counts
        )
        pointer_states['classical_fidelity'] = classical_fidelity
        
        return pointer_states
    
    def _compute_classical_fidelity(
        self, 
        pointer_eigenvecs: torch.Tensor,
        visit_counts: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute how "classical" the pointer states are
        
        Classical states should be localized in visit count space.
        """
        num_states = pointer_eigenvecs.shape[1]
        fidelities = []
        
        for i in range(num_states):
            state = pointer_eigenvecs[:, i]
            
            # Compute localization in visit count
            # Classical states have low variance in visit count
            mean_visits = torch.sum(state.abs()**2 * visit_counts)
            variance_visits = torch.sum(state.abs()**2 * (visit_counts - mean_visits)**2)
            
            # Fidelity = 1 / (1 + variance) (high fidelity = low variance)
            fidelity = 1.0 / (1.0 + variance_visits + self.config.matrix_regularization)
            fidelities.append(fidelity)
        
        return torch.stack(fidelities)
    
    def extract_classical_probabilities(
        self, 
        rho: torch.Tensor,
        visit_counts: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract classical probabilities from decoherent density matrix
        
        This provides the bridge from quantum to classical MCTS.
        """
        # Get pointer states
        pointer_info = self.identify_pointer_states(rho, visit_counts)
        
        if pointer_info['num_pointer_states'] == 0:
            # Fallback: uniform distribution
            return torch.ones(rho.shape[0], device=self.device) / rho.shape[0]
        
        # Weighted combination of pointer states
        pointer_populations = pointer_info['populations']
        pointer_eigenvecs = pointer_info['eigenvectors']
        
        # Extract probabilities as diagonal elements of reduced density matrix
        classical_probs = torch.zeros(rho.shape[0], device=self.device)
        
        for i, population in enumerate(pointer_populations):
            state = pointer_eigenvecs[:, i]
            # Probability distribution from this pointer state
            state_probs = torch.abs(state)**2
            classical_probs += population * state_probs
        
        # Normalize
        classical_probs = F.normalize(classical_probs, p=1, dim=0)
        
        return classical_probs


class DecoherenceEngine:
    """
    Main decoherence engine coordinating all decoherence processes
    
    This engine manages the quantum→classical transition and provides
    the interface for MCTS integration.
    """
    
    def __init__(self, config: DecoherenceConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # Initialize sub-engines
        self.density_evolution = DensityMatrixEvolution(config, device)
        self.pointer_analyzer = PointerStateAnalyzer(config, device)
        
        # State tracking
        self.current_rho = None
        self.current_hamiltonian = None
        
        # Statistics
        self.stats = {
            'evolution_steps': 0,
            'avg_evolution_time': 0.0,
            'pointer_states_count': 0,
            'classical_fidelity': 0.0,
            'decoherence_rate': 0.0
        }
        
        logger.info(f"DecoherenceEngine initialized on {device}")
    
    def initialize_quantum_state(
        self, 
        num_nodes: int,
        initial_superposition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Initialize quantum density matrix
        
        Args:
            num_nodes: Number of tree nodes
            initial_superposition: Optional initial quantum state
            
        Returns:
            Initial density matrix
        """
        if initial_superposition is not None:
            # Create density matrix from state vector
            psi = initial_superposition.to(self.device)
            rho = torch.outer(psi.conj(), psi)
        else:
            # Start with maximally mixed state (quantum superposition)
            rho = torch.eye(num_nodes, device=self.device) / num_nodes
        
        # Ensure proper normalization
        rho = rho / torch.trace(rho)
        
        self.current_rho = rho
        return rho
    
    def create_tree_hamiltonian(
        self, 
        visit_counts: torch.Tensor,
        tree_structure: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Create effective Hamiltonian for tree search dynamics
        
        The Hamiltonian encodes the tree structure and drives quantum evolution.
        """
        num_nodes = visit_counts.shape[0]
        
        # Start with visit count energy (diagonal)
        # Higher visit counts → higher energy
        H = torch.diag(visit_counts.float())
        
        # Add tree connectivity terms if available
        if tree_structure is not None and 'children' in tree_structure:
            children = tree_structure['children']
            
            # Add parent-child coupling terms
            coupling_strength = 0.1  # Tunable parameter
            
            for parent in range(min(num_nodes, children.shape[0])):
                if len(children.shape) > 1:
                    node_children = children[parent][children[parent] >= 0]
                    
                    for child in node_children:
                        if child < num_nodes:
                            # Off-diagonal coupling
                            H[parent, child] += coupling_strength
                            H[child, parent] += coupling_strength
        
        self.current_hamiltonian = H.to(self.device)
        return self.current_hamiltonian
    
    def evolve_quantum_to_classical(
        self,
        visit_counts: torch.Tensor,
        tree_structure: Optional[Dict[str, torch.Tensor]] = None,
        evolution_time: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Evolve quantum state to classical (main interface)
        
        This is the primary method that MCTS calls to get quantum-corrected
        probabilities through decoherence.
        """
        # Initialize or update Hamiltonian
        if self.current_hamiltonian is None or self.current_hamiltonian.shape[0] != len(visit_counts):
            self.create_tree_hamiltonian(visit_counts, tree_structure)
        
        # Initialize density matrix if needed
        if self.current_rho is None or self.current_rho.shape[0] != len(visit_counts):
            self.initialize_quantum_state(len(visit_counts))
        
        # Evolve to steady state
        start_time = time.perf_counter()
        
        final_rho, evolution_t = self.density_evolution.evolve_to_steady_state(
            self.current_rho,
            self.current_hamiltonian,
            visit_counts,
            max_time=evolution_time,
            tree_structure=tree_structure
        )
        
        end_time = time.perf_counter()
        
        # Update state
        self.current_rho = final_rho
        
        # Analyze pointer states
        pointer_info = self.pointer_analyzer.identify_pointer_states(
            final_rho, visit_counts
        )
        
        # Extract classical probabilities
        classical_probs = self.pointer_analyzer.extract_classical_probabilities(
            final_rho, visit_counts
        )
        
        # Update statistics
        self.stats['evolution_steps'] += 1
        self.stats['avg_evolution_time'] = 0.9 * self.stats['avg_evolution_time'] + 0.1 * (end_time - start_time)
        self.stats['pointer_states_count'] = pointer_info['num_pointer_states']
        if len(pointer_info['classical_fidelity']) > 0:
            self.stats['classical_fidelity'] = pointer_info['classical_fidelity'].mean().item()
        
        # Compute decoherence rate
        operators = DecoherenceOperators(self.config, self.device)
        rates = operators.compute_decoherence_rates(visit_counts)
        self.stats['decoherence_rate'] = rates.mean().item()
        
        return {
            'classical_probabilities': classical_probs,
            'density_matrix': final_rho,
            'pointer_states': pointer_info,
            'evolution_time': evolution_t,
            'decoherence_rate': self.stats['decoherence_rate']
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """Get decoherence engine statistics"""
        return dict(self.stats)
    
    def reset_quantum_state(self):
        """Reset to quantum superposition"""
        self.current_rho = None
        self.current_hamiltonian = None


# Factory function for easy instantiation
def create_decoherence_engine(
    device: Union[str, torch.device] = 'cuda',
    base_rate: float = 1.0,
    **kwargs
) -> DecoherenceEngine:
    """
    Factory function to create decoherence engine
    
    Args:
        device: Device for computation
        base_rate: Base decoherence rate λ
        **kwargs: Override default config parameters
        
    Returns:
        Initialized DecoherenceEngine
    """
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Create config with overrides
    config_dict = {
        'base_decoherence_rate': base_rate,
        'hbar': 1.0,
        'temperature': 1.0,
    }
    config_dict.update(kwargs)
    
    config = DecoherenceConfig(**config_dict)
    
    return DecoherenceEngine(config, device)