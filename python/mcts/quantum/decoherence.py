"""
Decoherence Engine for QFT-MCTS (v1.0 and v2.0)
===============================================

This module implements quantum decoherence dynamics that naturally explain
the emergence of classical MCTS behavior from quantum superpositions.

Key Features:
- Density matrix evolution with master equation
- Lindblad decoherence operators based on visit count differences
- Pointer state identification (classically stable states)
- GPU-accelerated density matrix computations
- Automatic quantum→classical transition

v2.0 Features:
- Power-law decoherence: ρᵢⱼ(N) ~ N^(-Γ₀)
- Discrete information time: τ(N) = log(N+2)
- Phase-dependent decoherence rates
- Temperature annealing: T(N) = T₀/log(N+2)
- Auto-computed Γ₀ from theory: 2c_puct·σ²_eval·T

Based on:
- v1.0: docs/qft-mcts-math-foundations.md Section 3.2
- v2.0: docs/v2.0/mathematical-foundations.md
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
import time
import math
from enum import Enum

logger = logging.getLogger(__name__)


class TimeFormulation(Enum):
    """Time formulation for decoherence dynamics"""
    CONTINUOUS = "continuous"  # v1.0 - continuous time
    DISCRETE = "discrete"      # v2.0 - discrete information time


@dataclass
class DecoherenceConfig:
    """Configuration for decoherence engine"""
    # Version settings
    time_formulation: TimeFormulation = TimeFormulation.CONTINUOUS  # v1 or v2
    version: str = 'v1'  # 'v1' or 'v2'
    
    # Physical parameters
    base_decoherence_rate: float = 1.0    # λ - system-environment coupling
    hbar: float = 1.0                     # Reduced Planck constant
    temperature: float = 1.0              # Environment temperature
    
    # v2.0 specific parameters
    power_law_exponent: Optional[float] = None  # Γ₀ for power-law decay (auto-computed if None)
    c_puct: Optional[float] = None        # For auto-computing parameters
    temperature_mode: str = 'fixed'       # 'fixed' or 'annealing'
    initial_temperature: float = 1.0      # T₀ for annealing
    
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


class DiscreteTimeHandler:
    """Handles discrete information time for v2.0"""
    
    def __init__(self, config: DecoherenceConfig):
        self.config = config
        self.eps = 1e-8
    
    def information_time(self, N: Union[int, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Compute information time τ(N) = log(N+2)"""
        if isinstance(N, torch.Tensor):
            return torch.log(N + 2)
        return math.log(N + 2)
    
    def time_derivative(self, N: Union[int, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Compute d/dτ = (N+2)d/dN"""
        return 1.0 / (N + 2)
    
    def compute_temperature(self, N: Union[int, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Compute temperature for annealing"""
        if self.config.temperature_mode == 'fixed':
            return self.config.initial_temperature
        elif self.config.temperature_mode == 'annealing':
            tau = self.information_time(N)
            return self.config.initial_temperature / (tau + self.eps)
        else:
            return self.config.temperature


class DecoherenceOperators:
    """
    Lindblad operators that describe decoherence processes
    
    These operators encode how the environment monitors the tree search
    process, leading to classical behavior through selective decoherence.
    """
    
    def __init__(self, config: DecoherenceConfig, device: torch.device):
        self.config = config
        self.device = device
        self.time_handler = DiscreteTimeHandler(config) if config.version == 'v2' else None
        
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
        visit_counts: torch.Tensor,
        total_simulations: Optional[int] = None,
        q_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute pairwise decoherence rates Γ_ij
        
        v1.0: Γ_ij = λ|N_i - N_j|/max(N_i, N_j) (exponential decay)
        v2.0: ρᵢⱼ(N) ~ N^(-Γ₀) (power-law decay)
        """
        num_nodes = visit_counts.shape[0]
        
        if self.config.version == 'v2' and self.config.time_formulation == TimeFormulation.DISCRETE:
            # v2.0: Power-law decoherence
            if self.config.power_law_exponent is not None:
                gamma = self.config.power_law_exponent
            else:
                # Auto-compute from theory: Γ₀ = 2c_puct·σ²_eval·T₀
                if q_values is not None and self.config.c_puct is not None:
                    sigma_Q = torch.std(q_values)
                    T = self.config.initial_temperature
                    if total_simulations is not None and self.config.temperature_mode == 'annealing':
                        T = self.time_handler.compute_temperature(total_simulations)
                    gamma = 2 * self.config.c_puct * sigma_Q**2 * T
                else:
                    gamma = 0.5  # Default fallback
            
            # Power-law decay matrix
            decoherence_matrix = (visit_counts + 1).unsqueeze(0).pow(-gamma)
            
            # Apply pairwise differences as modulation
            visit_i = visit_counts.unsqueeze(1)
            visit_j = visit_counts.unsqueeze(0)
            visit_diff = torch.abs(visit_i - visit_j)
            visit_max = torch.max(visit_i, visit_j)
            
            # Modulate power-law by visit differences
            modulation = visit_diff / (visit_max + self.config.matrix_regularization)
            rates = self.config.base_decoherence_rate * decoherence_matrix * modulation
            
        else:
            # v1.0: Original exponential decoherence
            visit_i = visit_counts.unsqueeze(1)
            visit_j = visit_counts.unsqueeze(0)
            visit_diff = torch.abs(visit_i - visit_j)
            visit_max = torch.max(visit_i, visit_j)
            rates = self.config.base_decoherence_rate * visit_diff / (
                visit_max + self.config.matrix_regularization
            )
        
        return rates


class MCTSPhase(Enum):
    """MCTS phase for v2.0 phase-dependent decoherence"""
    QUANTUM = "quantum"       # N < N_c1
    CRITICAL = "critical"     # N_c1 < N < N_c2
    CLASSICAL = "classical"   # N > N_c2


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
        self.time_handler = DiscreteTimeHandler(config) if config.version == 'v2' else None
        self.total_simulations = 0  # Track for v2.0
        self.current_phase = MCTSPhase.QUANTUM  # Track phase for v2.0
        
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
        
        # Compute decoherence rates (v2.0 needs q_values)
        if self.config.version == 'v2':
            q_values = hamiltonian.diag() if hamiltonian is not None else None
            decoherence_rates = self.operators.compute_decoherence_rates(
                visit_counts, self.total_simulations, q_values
            )
        else:
            decoherence_rates = self.operators.compute_decoherence_rates(visit_counts)
        
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
        
        # Ensure Hamiltonian has same dtype as density matrix
        if hamiltonian.dtype != rho.dtype:
            hamiltonian = hamiltonian.to(rho.dtype)
        
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
        
        v2.0: Includes phase-dependent scaling of decoherence strength
        """
        decoherence_term = torch.zeros_like(rho)
        
        # Phase-dependent scaling for v2.0
        phase_scaling = 1.0
        if self.config.version == 'v2' and hasattr(self, 'current_phase'):
            if self.current_phase == MCTSPhase.QUANTUM:
                phase_scaling = 0.5  # Weaker decoherence in quantum phase
            elif self.current_phase == MCTSPhase.CRITICAL:
                phase_scaling = 1.0  # Standard decoherence
            elif self.current_phase == MCTSPhase.CLASSICAL:
                phase_scaling = 2.0  # Stronger decoherence to enforce classicality
        
        for L_k in lindblad_ops:
            # Ensure Lindblad operator has same dtype as density matrix
            if L_k.dtype != rho.dtype:
                L_k = L_k.to(rho.dtype)
            L_k_dag = L_k.conj().transpose(-2, -1)
            
            # L_k ρ L_k†
            dissipator = torch.matmul(torch.matmul(L_k, rho), L_k_dag)
            
            # L_k†L_k
            L_dag_L = torch.matmul(L_k_dag, L_k)
            
            # {L_k†L_k, ρ}/2 = (L_k†L_k ρ + ρ L_k†L_k)/2
            anticommutator = 0.5 * (
                torch.matmul(L_dag_L, rho) + torch.matmul(rho, L_dag_L)
            )
            
            # Add contribution with phase scaling
            decoherence_term += phase_scaling * (dissipator - anticommutator)
        
        return decoherence_term
    
    def _ensure_density_matrix_properties(self, rho: torch.Tensor) -> torch.Tensor:
        """Ensure density matrix remains valid (Hermitian, positive, trace=1)"""
        
        # Make Hermitian
        rho = 0.5 * (rho + rho.conj().transpose(-2, -1))
        
        # Ensure positive semidefinite (eigenvalue decomposition)
        eigenvals, eigenvecs = torch.linalg.eigh(rho)
        eigenvals = torch.clamp(eigenvals, min=0.0)
        # Create diagonal matrix with same dtype as eigenvectors
        diag_eigenvals = torch.diag(eigenvals).to(eigenvecs.dtype)
        rho = torch.matmul(eigenvecs, torch.matmul(diag_eigenvals, eigenvecs.conj().transpose(-2, -1)))
        
        # Normalize trace
        trace = torch.trace(rho)
        # Trace should be real for a density matrix, but may have small imaginary part due to numerics
        trace_real = trace.real
        if trace_real > self.config.matrix_regularization:
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
            
            # Increment time
            t += self.config.dt
            
            # Check convergence
            diff = torch.norm(rho - rho_prev).item()
            if diff < self.config.convergence_threshold:
                logger.debug(f"Density matrix converged at t={t:.3f}")
                break
        
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
        
        logger.debug(f"DecoherenceEngine initialized on {device}")
    
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
    
    def set_total_simulations(self, N: int):
        """Set total simulation count for v2.0 features"""
        self.total_simulations = N
        if hasattr(self.density_evolution, 'total_simulations'):
            self.density_evolution.total_simulations = N
    
    def set_mcts_phase(self, phase: MCTSPhase):
        """Set current MCTS phase for v2.0 phase-dependent decoherence"""
        self.current_phase = phase
        if hasattr(self.density_evolution, 'current_phase'):
            self.density_evolution.current_phase = phase
    
    def evolve_quantum_to_classical(
        self,
        visit_counts: torch.Tensor,
        tree_structure: Optional[Dict[str, torch.Tensor]] = None,
        evolution_time: Optional[float] = None,
        total_simulations: Optional[int] = None,
        mcts_phase: Optional[MCTSPhase] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Evolve quantum state to classical (main interface)
        
        This is the primary method that MCTS calls to get quantum-corrected
        probabilities through decoherence.
        
        Args:
            visit_counts: Node visit counts
            tree_structure: Optional tree connectivity
            evolution_time: Max evolution time
            total_simulations: Total MCTS simulations (v2.0)
            mcts_phase: Current MCTS phase (v2.0)
        """
        # Update v2.0 state if provided
        if total_simulations is not None:
            self.set_total_simulations(total_simulations)
        if mcts_phase is not None:
            self.set_mcts_phase(mcts_phase)
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
        if self.config.version == 'v2':
            # Extract Q values from diagonal of Hamiltonian for v2.0
            q_values = self.current_hamiltonian.diag() if self.current_hamiltonian is not None else None
            rates = operators.compute_decoherence_rates(visit_counts, self.total_simulations, q_values)
        else:
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


# Factory functions for easy instantiation
def create_decoherence_engine(
    device: Union[str, torch.device] = 'cuda',
    base_rate: float = 1.0,
    version: str = 'v1',
    **kwargs
) -> DecoherenceEngine:
    """
    Factory function to create decoherence engine
    
    Args:
        device: Device for computation
        base_rate: Base decoherence rate λ
        version: 'v1' or 'v2' for different formulations
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
        'version': version,
    }
    
    # Set v2-specific defaults
    if version == 'v2':
        config_dict.update({
            'time_formulation': TimeFormulation.DISCRETE,
            'temperature_mode': 'annealing',
            'initial_temperature': 1.0,
        })
    
    config_dict.update(kwargs)
    
    config = DecoherenceConfig(**config_dict)
    
    return DecoherenceEngine(config, device)


def create_decoherence_engine_v2(
    device: Union[str, torch.device] = 'cuda',
    c_puct: Optional[float] = None,
    power_law_exponent: Optional[float] = None,
    temperature_mode: str = 'annealing',
    **kwargs
) -> DecoherenceEngine:
    """
    Create v2.0 decoherence engine with power-law decay
    
    Args:
        device: Device for computation
        c_puct: PUCT constant for auto-computing parameters
        power_law_exponent: Γ₀ for power-law decay (auto if None)
        temperature_mode: 'fixed' or 'annealing'
        **kwargs: Additional config overrides
    """
    v2_config = {
        'c_puct': c_puct,
        'power_law_exponent': power_law_exponent,
        'temperature_mode': temperature_mode,
    }
    v2_config.update(kwargs)
    
    return create_decoherence_engine(
        device=device,
        version='v2',
        **v2_config
    )


def create_decoherence_engine_v1(
    device: Union[str, torch.device] = 'cuda',
    base_rate: float = 1.0,
    **kwargs
) -> DecoherenceEngine:
    """
    Create v1.0 decoherence engine with exponential decay
    """
    return create_decoherence_engine(
        device=device,
        base_rate=base_rate,
        version='v1',
        **kwargs
    )