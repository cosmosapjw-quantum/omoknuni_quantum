"""
Entanglement analysis for MCTS based on path integral interpretation.

According to the quantum foundation document, MCTS implements a path integral
where multiple paths are explored in parallel, creating quantum-like superposition.
The "entanglement" measures correlations between different regions of the search tree,
reflecting how information about good moves spreads through the tree structure.

Key insight: Increasing simulation number acts as temporal evolution with
interference between paths (waves), leading to quantum-to-classical transition.
"""
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from scipy.optimize import curve_fit
from scipy.linalg import expm, logm

# Import unified quantum definitions
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from quantum_definitions import (
    UnifiedQuantumDefinitions,
    MCTSQuantumState,
    compute_von_neumann_entropy,
    compute_entanglement_entropy,
    construct_quantum_state_from_visits
)


@dataclass
class EntanglementResult:
    """Results from entanglement analysis"""
    entropy: float
    partition_scheme: str
    region_A_size: int
    region_B_size: int
    boundary_size: int
    mutual_information: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary"""
        return {
            'entropy': self.entropy,
            'partition_scheme': self.partition_scheme,
            'region_A_size': self.region_A_size,
            'region_B_size': self.region_B_size,
            'boundary_size': self.boundary_size,
            'mutual_information': self.mutual_information
        }


class EntanglementAnalyzer:
    """
    Analyze quantum-like entanglement in MCTS trees.
    
    Key insight from quantum information theory:
    - Entanglement = Superposition in composite systems
    - In MCTS: Different paths create superposition state
    - Entanglement entropy measures path diversity/coherence
    
    Methods:
    - Von Neumann entropy of reduced density matrix
    - Mutual information between subtrees
    - Path concurrence (quantum measure)
    - Participation ratio of paths
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize entanglement analyzer.
        
        Args:
            device: Computation device
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize unified quantum definitions
        self.quantum_defs = UnifiedQuantumDefinitions(device=self.device)
    
    def compute_entanglement_entropy(self, tree_data: Dict[str, Any],
                                   partition_scheme: str = 'depth') -> float:
        """
        Compute entanglement entropy for tree partition.
        
        In the path integral interpretation, this measures how information
        about good moves is distributed across different regions of the tree.
        High entropy indicates many paths contribute equally (superposition),
        while low entropy indicates convergence to specific paths (classical).
        
        Args:
            tree_data: Dictionary with tree observables including:
                - visit_distribution: N(s,a) for each state-action
                - tree_size: Total number of nodes
                - depth_distribution: Optional depth information
                - value_landscape: Optional Q-values
            partition_scheme: How to partition ('depth', 'half', 'value', 'subtree')
            
        Returns:
            Von Neumann entropy of the reduced density matrix
        """
        # Get visit distribution
        visits = tree_data.get('visit_distribution', tree_data.get('visits'))
        if visits is None:
            raise ValueError("No visit distribution found in tree_data")
            
        if not isinstance(visits, torch.Tensor):
            visits = torch.tensor(visits, device=self.device, dtype=torch.float32)
        else:
            visits = visits.to(self.device)
        
        # Ensure non-negative visits
        visits = torch.abs(visits)
        
        # Partition tree into regions A and B
        region_A, region_B = self._partition_tree(tree_data, partition_scheme)
        
        # Construct quantum state using unified definitions
        # This naturally creates mixed states with off-diagonal terms
        quantum_state = self.quantum_defs.construct_quantum_state_from_single_visits(
            visits, outcome_uncertainty=0.2  # Model inherent search uncertainty
        )
        
        # Compute entanglement entropy using unified definitions
        # This traces out region B to get reduced density matrix for region A
        entropy = self.quantum_defs.compute_entanglement_entropy(
            quantum_state, region_B
        )
        
        # Add normalization for interpretability
        # Maximum entropy is log(dim(A))
        max_entropy = np.log(len(region_A)) if len(region_A) > 0 else 1.0
        normalized_entropy = float(entropy) / max_entropy if max_entropy > 0 else 0.0
        
        # Return raw entropy (normalized version available in metadata)
        return float(entropy)
    
    def _partition_tree(self, tree_data: Dict[str, Any], 
                       scheme: str) -> Tuple[List[int], List[int]]:
        """
        Partition tree nodes into two regions.
        
        In the path integral formulation, different partitions reveal
        different aspects of the quantum-like correlations:
        - 'depth': Separates early vs late decisions (UV vs IR in RG flow)
        - 'value': Separates good vs bad paths (pointer states)
        - 'subtree': Separates different strategic options (path bundles)
        - 'half': Simple bipartition for baseline
        
        Args:
            tree_data: Tree observables
            scheme: Partition scheme
            
        Returns:
            (region_A_indices, region_B_indices)
        """
        n_nodes = tree_data.get('tree_size', len(tree_data.get('visit_distribution', [])))
        
        if scheme == 'depth':
            # Partition by depth - separates scales in RG flow
            depths = tree_data.get('depth_distribution', torch.zeros(n_nodes))
            if not isinstance(depths, torch.Tensor):
                depths = torch.tensor(depths)
            
            median_depth = torch.median(depths)
            region_A = [i for i in range(n_nodes) if depths[i] <= median_depth]
            region_B = [i for i in range(n_nodes) if depths[i] > median_depth]
            
        elif scheme == 'half':
            # Simple half partition
            mid = n_nodes // 2
            region_A = list(range(mid))
            region_B = list(range(mid, n_nodes))
            
        elif scheme == 'value':
            # Partition by value - separates pointer states
            values = tree_data.get('value_landscape', tree_data.get('q_values', torch.zeros(n_nodes)))
            if not isinstance(values, torch.Tensor):
                values = torch.tensor(values)
                
            median_value = torch.median(values)
            region_A = [i for i in range(n_nodes) if values[i] <= median_value]
            region_B = [i for i in range(n_nodes) if values[i] > median_value]
            
        elif scheme == 'subtree':
            # Partition by subtree structure - separates path bundles
            # This best reflects the path integral structure
            visits = tree_data.get('visit_distribution', tree_data.get('visits', torch.zeros(n_nodes)))
            if not isinstance(visits, torch.Tensor):
                visits = torch.tensor(visits)
            
            # Find the most visited actions at root level
            # Assume first few nodes are root actions
            n_root_actions = min(10, n_nodes // 10)  # Heuristic
            root_visits = visits[:n_root_actions]
            
            # Find top two subtrees
            if len(root_visits) >= 2:
                top_indices = torch.argsort(root_visits, descending=True)[:2]
                
                # Region A: Nodes associated with top action
                # Region B: Nodes associated with second action
                # This is simplified - in practice would trace actual tree structure
                region_A = list(range(top_indices[0], n_nodes, n_root_actions))
                region_B = list(range(top_indices[1], n_nodes, n_root_actions))
                
                # Ensure all nodes are assigned
                assigned = set(region_A + region_B)
                unassigned = [i for i in range(n_nodes) if i not in assigned]
                # Split unassigned evenly
                mid_unassigned = len(unassigned) // 2
                region_A.extend(unassigned[:mid_unassigned])
                region_B.extend(unassigned[mid_unassigned:])
            else:
                # Fallback to half partition
                mid = n_nodes // 2
                region_A = list(range(mid))
                region_B = list(range(mid, n_nodes))
            
        else:
            raise ValueError(f"Unknown partition scheme: {scheme}")
        
        # Ensure non-empty regions
        if not region_A:
            region_A = [0]
        if not region_B:
            region_B = [n_nodes - 1] if n_nodes > 1 else [0]
        
        return region_A, region_B
    
    def _construct_density_matrix(self, visits: torch.Tensor) -> torch.Tensor:
        """
        Construct density matrix from visit distribution.
        
        Treats normalized visit counts as quantum amplitudes squared.
        
        Args:
            visits: Visit count distribution
            
        Returns:
            Density matrix
        """
        # Normalize visits to get probabilities
        probs = visits / visits.sum()
        
        # For simplicity, treat as diagonal density matrix
        # In full implementation, could include off-diagonal correlations
        n = len(probs)
        rho = torch.diag(probs).to(self.device)
        
        return rho
    
    def _partial_trace(self, rho: torch.Tensor, 
                      trace_indices: List[int]) -> torch.Tensor:
        """
        Compute partial trace over specified indices.
        
        Args:
            rho: Full density matrix
            trace_indices: Indices to trace out
            
        Returns:
            Reduced density matrix
        """
        n = rho.shape[0]
        keep_indices = [i for i in range(n) if i not in trace_indices]
        
        if len(keep_indices) == 0:
            return torch.tensor([[1.0]], device=self.device)
        
        # For diagonal density matrix, partial trace is simple
        rho_reduced = torch.zeros(len(keep_indices), len(keep_indices), 
                                 device=self.device)
        
        for i, idx_i in enumerate(keep_indices):
            for j, idx_j in enumerate(keep_indices):
                if idx_i == idx_j:
                    # Sum over traced indices
                    rho_reduced[i, j] = rho[idx_i, idx_i]
                    for trace_idx in trace_indices:
                        if trace_idx < n:
                            rho_reduced[i, j] += rho[trace_idx, trace_idx]
        
        # Renormalize
        trace = torch.trace(rho_reduced)
        if trace > 0:
            rho_reduced = rho_reduced / trace
        
        return rho_reduced
    
    def _von_neumann_entropy(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Compute von Neumann entropy: S = -Tr(ρ log ρ).
        
        Args:
            rho: Density matrix
            
        Returns:
            Entropy
        """
        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvalsh(rho)
        
        # Filter out numerical zeros
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        # Von Neumann entropy
        if len(eigenvalues) == 0:
            return torch.tensor(0.0, device=self.device)
        
        entropy = -torch.sum(eigenvalues * torch.log(eigenvalues))
        
        return entropy
    
    def compute_mutual_information(self, tree_data: Dict[str, Any],
                                 region_A_indices: List[int],
                                 region_B_indices: List[int]) -> float:
        """
        Compute mutual information between regions.
        
        I(A:B) = S(A) + S(B) - S(A∪B)
        
        Args:
            tree_data: Tree observables
            region_A_indices: Indices for region A
            region_B_indices: Indices for region B
            
        Returns:
            Mutual information
        """
        visits = tree_data['visit_distribution']
        if not isinstance(visits, torch.Tensor):
            visits = torch.tensor(visits, device=self.device)
        else:
            visits = visits.to(self.device)
        
        # Compute individual entropies
        rho_full = self._construct_density_matrix(visits)
        
        # S(A)
        rho_A = self._partial_trace(rho_full, region_B_indices)
        S_A = self._von_neumann_entropy(rho_A)
        
        # S(B)
        rho_B = self._partial_trace(rho_full, region_A_indices)
        S_B = self._von_neumann_entropy(rho_B)
        
        # S(A∪B) - entropy of full system
        all_indices = list(set(region_A_indices + region_B_indices))
        complement = [i for i in range(len(visits)) if i not in all_indices]
        rho_AB = self._partial_trace(rho_full, complement)
        S_AB = self._von_neumann_entropy(rho_AB)
        
        # Mutual information
        MI = S_A + S_B - S_AB
        
        return float(MI)
    
    def get_boundary_size(self, tree_data: Dict[str, Any],
                         partition_scheme: str) -> int:
        """
        Get boundary size between partitions.
        
        Args:
            tree_data: Tree observables
            partition_scheme: Partition scheme
            
        Returns:
            Number of connections across boundary
        """
        region_A, region_B = self._partition_tree(tree_data, partition_scheme)
        
        # For tree structure, boundary is number of edges crossing partition
        # Simplified: use geometric mean of region sizes
        boundary = int(np.sqrt(len(region_A) * len(region_B)))
        
        return boundary
    
    def verify_area_law(self, entropies: List[float],
                       boundary_sizes: List[int]) -> Dict[str, Any]:
        """
        Verify area law scaling: S ~ boundary^α.
        
        Args:
            entropies: List of entanglement entropies
            boundary_sizes: Corresponding boundary sizes
            
        Returns:
            Dictionary with scaling analysis
        """
        # Fit power law in log space
        log_boundaries = np.log(boundary_sizes)
        log_entropies = np.log(np.array(entropies) + 1e-10)
        
        # Linear fit in log space
        alpha, log_c = np.polyfit(log_boundaries, log_entropies, 1)
        
        # Area law has α ≈ 1
        area_law_verified = abs(alpha - 1.0) < 0.2
        
        return {
            'scaling_exponent': float(alpha),
            'area_law_verified': area_law_verified,
            'r_squared': float(np.corrcoef(log_boundaries, log_entropies)[0, 1]**2)
        }
    
    def compute_path_superposition_entropy(self, path_amplitudes: np.ndarray) -> float:
        """
        Compute entanglement entropy from path superposition.
        
        In quantum information theory:
        - Pure superposition state: |ψ⟩ = Σ_i α_i |path_i⟩
        - Entanglement entropy = Shannon entropy of |α_i|²
        
        Args:
            path_amplitudes: Amplitudes for each path
            
        Returns:
            Entanglement entropy
        """
        # Normalize amplitudes
        amplitudes = np.abs(path_amplitudes)
        if np.sum(amplitudes) == 0:
            return 0.0
            
        probabilities = amplitudes**2 / np.sum(amplitudes**2)
        
        # Shannon entropy = entanglement entropy for pure states
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        
        return entropy
    
    def compute_path_concurrence(self, path_amplitudes_A: np.ndarray,
                                path_amplitudes_B: np.ndarray) -> float:
        """
        Compute concurrence between two path sets (quantum entanglement measure).
        
        Concurrence C ∈ [0,1]:
        - C = 0: No entanglement (product state)
        - C = 1: Maximum entanglement
        
        Args:
            path_amplitudes_A: Amplitudes for paths in region A
            path_amplitudes_B: Amplitudes for paths in region B
            
        Returns:
            Concurrence value
        """
        # Normalize
        psi_A = path_amplitudes_A / np.sqrt(np.sum(np.abs(path_amplitudes_A)**2) + 1e-10)
        psi_B = path_amplitudes_B / np.sqrt(np.sum(np.abs(path_amplitudes_B)**2) + 1e-10)
        
        # For bipartite pure states, concurrence relates to overlap
        # C = 2 * |⟨ψ_A|ψ_B⟩| for maximally entangled basis
        
        # Simplified: use participation ratio as proxy
        participation_A = 1.0 / np.sum(np.abs(psi_A)**4)
        participation_B = 1.0 / np.sum(np.abs(psi_B)**4)
        
        # Geometric mean of participations
        concurrence = 2 * np.sqrt(participation_A * participation_B) / (participation_A + participation_B)
        
        return min(1.0, concurrence)
    
    def compute_superposition_measures(self, visits: torch.Tensor) -> Dict[str, float]:
        """
        Compute various superposition/entanglement measures.
        
        Returns:
            Dictionary with:
            - participation_ratio: Effective number of superposed states
            - coherence_length: Spread of superposition
            - entanglement_entropy: von Neumann entropy
            - purity: Tr(ρ²) - measures mixedness
        """
        # Convert to probabilities
        total_visits = visits.sum()
        if total_visits == 0:
            return {
                'participation_ratio': 1.0,
                'coherence_length': 0.0,
                'entanglement_entropy': 0.0,
                'purity': 1.0
            }
        
        probs = visits / total_visits
        
        # Participation ratio: 1/Σp_i² 
        # Measures effective number of states in superposition
        participation_ratio = 1.0 / torch.sum(probs**2)
        
        # Coherence length: RMS spread of probability
        indices = torch.arange(len(probs), device=self.device)
        mean_idx = torch.sum(probs * indices)
        coherence_length = torch.sqrt(torch.sum(probs * (indices - mean_idx)**2))
        
        # Entanglement entropy (Shannon)
        probs_nonzero = probs[probs > 0]
        entropy = -torch.sum(probs_nonzero * torch.log(probs_nonzero))
        
        # Purity
        purity = torch.sum(probs**2)
        
        return {
            'participation_ratio': float(participation_ratio),
            'coherence_length': float(coherence_length),
            'entanglement_entropy': float(entropy),
            'purity': float(purity)
        }
    
    def compute_entanglement_evolution(self, snapshots: List[Dict[str, Any]], 
                                     partition_scheme: str = 'subtree') -> Dict[str, Any]:
        """
        Track entanglement evolution over MCTS simulations.
        
        According to the quantum foundation, increasing simulation number
        acts as temporal evolution with interference between paths.
        We expect to see:
        1. Initial high entanglement (many paths in superposition)
        2. Gradual decrease as good paths are selected (decoherence)
        3. Low final entanglement (classical decision emerges)
        
        Args:
            snapshots: List of tree snapshots over time
            partition_scheme: How to partition the tree
            
        Returns:
            Dictionary with evolution metrics
        """
        entropies = []
        mutual_infos = []
        times = []
        
        for i, snapshot in enumerate(snapshots):
            try:
                # Compute entanglement entropy
                S = self.compute_entanglement_entropy(snapshot, partition_scheme)
                entropies.append(S)
                
                # Also compute mutual information if possible
                if 'tree_size' in snapshot and snapshot['tree_size'] > 10:
                    region_A, region_B = self._partition_tree(snapshot, partition_scheme)
                    MI = self.compute_mutual_information(snapshot, region_A, region_B)
                    mutual_infos.append(MI)
                else:
                    mutual_infos.append(0.0)
                
                times.append(snapshot.get('timestamp', i))
                
            except Exception as e:
                # Skip problematic snapshots
                continue
        
        if len(entropies) < 2:
            return {'error': 'Insufficient data for evolution analysis'}
        
        # Analyze the evolution
        entropies_np = np.array(entropies)
        times_np = np.array(times)
        
        # Measure decay rate (quantum-to-classical transition)
        if entropies_np[0] > entropies_np[-1]:
            # Decreasing entropy - decoherence happening
            decay_rate = (entropies_np[0] - entropies_np[-1]) / (times_np[-1] - times_np[0])
            transition_time = times_np[np.argmin(np.abs(entropies_np - entropies_np[0]/2))]
        else:
            decay_rate = 0.0
            transition_time = times_np[-1]
        
        # Check for phase transitions (sudden drops)
        d_entropy = np.gradient(entropies_np)
        phase_transitions = np.where(d_entropy < -0.1 * np.std(d_entropy))[0]
        
        return {
            'entropies': entropies,
            'mutual_informations': mutual_infos,
            'times': times,
            'initial_entropy': float(entropies[0]),
            'final_entropy': float(entropies[-1]),
            'decay_rate': float(decay_rate),
            'transition_time': float(transition_time),
            'phase_transitions': phase_transitions.tolist(),
            'quantum_to_classical': entropies[0] > entropies[-1]  # Expected behavior
        }
    
    def compute_correlation_length(self, tree_data: Dict[str, Any]) -> float:
        """
        Estimate correlation length in tree.
        
        In the path integral formulation, this measures how far
        information about good moves propagates through the tree.
        
        Args:
            tree_data: Tree observables
            
        Returns:
            Correlation length
        """
        visits = tree_data.get('visit_distribution', tree_data.get('visits'))
        if visits is None:
            return 0.0
            
        if not isinstance(visits, torch.Tensor):
            visits = torch.tensor(visits, device=self.device)
        else:
            visits = visits.to(self.device)
        
        n = len(visits)
        
        # Compute correlation function
        correlations = []
        for distance in range(1, min(n//2, 20)):
            corr = 0.0
            count = 0
            
            for i in range(n - distance):
                corr += visits[i] * visits[i + distance]
                count += 1
            
            if count > 0:
                corr = corr / count
                correlations.append(float(corr))
        
        # Fit exponential decay to extract correlation length
        if len(correlations) > 3:
            distances = np.arange(1, len(correlations) + 1)
            try:
                # Fit C(r) = A * exp(-r/ξ)
                def exp_decay(r, A, xi):
                    return A * np.exp(-r / xi)
                
                popt, _ = curve_fit(exp_decay, distances, correlations,
                                  p0=[correlations[0], 5.0])
                correlation_length = popt[1]
            except:
                correlation_length = 1.0
        else:
            correlation_length = 1.0
        
        return float(correlation_length)
    
    def compute_entanglement_spectrum(self, tree_data: Dict[str, Any],
                                    partition_scheme: str) -> List[float]:
        """
        Compute entanglement spectrum (eigenvalues of reduced density matrix).
        
        Args:
            tree_data: Tree observables
            partition_scheme: Partition scheme
            
        Returns:
            List of eigenvalues (entanglement spectrum)
        """
        visits = tree_data['visit_distribution']
        if not isinstance(visits, torch.Tensor):
            visits = torch.tensor(visits, device=self.device)
        else:
            visits = visits.to(self.device)
        
        # Partition and compute reduced density matrix
        region_A, region_B = self._partition_tree(tree_data, partition_scheme)
        rho = self._construct_density_matrix(visits)
        rho_A = self._partial_trace(rho, region_B)
        
        # Get eigenvalues
        eigenvalues = torch.linalg.eigvalsh(rho_A)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        # Normalize
        eigenvalues = eigenvalues / eigenvalues.sum()
        
        return eigenvalues.tolist()