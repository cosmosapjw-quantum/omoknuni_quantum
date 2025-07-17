"""
Gauge-Invariant Policy Learning for MCTS

This module implements gauge theory concepts for robust policy updates in MCTS.
The key insight is that policy updates should be invariant under certain transformations
(gauge transformations) while preserving the underlying physics of the value function.

Mathematical Foundation:
- Lattice gauge theory on the MCTS tree
- Wilson loops as gauge-invariant observables
- Discrete gauge groups acting on policy space
- Fisher information metric for policy space geometry
- Gauge fixing for policy regularization

Key Concepts:
1. Policy Field: φ(s,a) - policy parameters at each state-action
2. Gauge Field: A_μ(s) - connections between adjacent states
3. Wilson Loops: W(C) = Tr[P exp(i∮_C A_μ dx^μ)] - gauge invariant quantities
4. Gauge Transformations: φ'(s,a) = U(s) φ(s,a) U†(s)
5. Covariant Derivatives: D_μ φ = ∂_μ φ + i[A_μ, φ]

Physical Interpretation:
- Policy updates as parallel transport on curved policy manifold
- Gauge invariance ensures robustness to local reparameterizations
- Wilson loops detect policy inconsistencies and cycles
- Curvature measures policy learning complexity
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from scipy.linalg import expm, logm
from scipy.optimize import minimize
import networkx as nx
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class GaugeField:
    """
    Represents a gauge field A_μ on the MCTS tree lattice.
    
    In lattice gauge theory, the gauge field lives on the links (edges)
    of the lattice and encodes the "connection" between adjacent sites.
    """
    # Link variables: A[edge] = group element for each edge
    link_variables: Dict[Tuple[int, int], torch.Tensor]
    # Gauge group dimension
    group_dim: int
    # Lattice structure (adjacency)
    adjacency: Dict[int, List[int]]
    
    def __post_init__(self):
        """Initialize gauge field with identity elements"""
        if not self.link_variables:
            self.link_variables = {}
            for node, neighbors in self.adjacency.items():
                for neighbor in neighbors:
                    if (node, neighbor) not in self.link_variables:
                        # Initialize with identity + small random perturbation
                        identity = torch.eye(self.group_dim, dtype=torch.float32)
                        perturbation = torch.randn(self.group_dim, self.group_dim) * 0.01
                        self.link_variables[(node, neighbor)] = identity + perturbation
                        
    def get_link(self, node_i: int, node_j: int) -> torch.Tensor:
        """Get link variable between nodes i and j"""
        if (node_i, node_j) in self.link_variables:
            return self.link_variables[(node_i, node_j)]
        elif (node_j, node_i) in self.link_variables:
            # Use hermitian conjugate for reverse direction
            return self.link_variables[(node_j, node_i)].conj().T
        else:
            # Return identity if no link exists
            return torch.eye(self.group_dim, dtype=torch.float32)


@dataclass
class WilsonLoop:
    """
    Represents a Wilson loop - a gauge-invariant observable.
    
    Wilson loops are computed as the trace of the ordered product
    of link variables around a closed path.
    """
    path: List[int]  # Sequence of nodes forming closed loop
    value: torch.Tensor  # Computed Wilson loop value
    length: int  # Number of links in the loop
    
    def is_closed(self) -> bool:
        """Check if path forms a closed loop"""
        return len(self.path) > 2 and self.path[0] == self.path[-1]


@dataclass
class GaugeInvariantPolicy:
    """
    Policy that transforms covariantly under gauge transformations.
    
    The policy field φ(s,a) lives in the gauge group representation
    and transforms as φ'(s,a) = U(s) φ(s,a) U†(s) under gauge transformations.
    """
    # Policy field values at each state-action
    policy_field: Dict[Tuple[int, int], torch.Tensor]
    # Gauge group dimension
    group_dim: int
    # Current gauge (for gauge fixing)
    gauge_condition: str = "coulomb"  # "coulomb", "landau", "axial"
    
    def gauge_transform(self, gauge_transformation: Dict[int, torch.Tensor]) -> 'GaugeInvariantPolicy':
        """
        Apply gauge transformation to policy field.
        
        Args:
            gauge_transformation: U(s) for each state s
            
        Returns:
            Gauge-transformed policy
        """
        new_policy_field = {}
        
        for (state, action), field_value in self.policy_field.items():
            if state in gauge_transformation:
                U = gauge_transformation[state]
                # Transform: φ'(s,a) = U(s) φ(s,a) U†(s)
                new_field_value = U @ field_value @ U.conj().T
                new_policy_field[(state, action)] = new_field_value
            else:
                new_policy_field[(state, action)] = field_value
                
        return GaugeInvariantPolicy(
            policy_field=new_policy_field,
            group_dim=self.group_dim,
            gauge_condition=self.gauge_condition
        )


class GaugeInvariantPolicyLearner:
    """
    Implements gauge-invariant policy learning for MCTS.
    
    The key insight is that policy updates should be robust to local
    reparameterizations of the policy space. This is achieved by:
    1. Treating policies as gauge fields on the tree lattice
    2. Using Wilson loops to detect inconsistencies
    3. Implementing gauge-invariant update rules
    4. Gauge fixing for computational efficiency
    """
    
    def __init__(self, group_dim: int = 2, gauge_coupling: float = 1.0,
                 device: Optional[str] = None):
        """
        Initialize gauge-invariant policy learner.
        
        Args:
            group_dim: Dimension of gauge group (e.g., 2 for SU(2))
            gauge_coupling: Strength of gauge interaction
            device: Computation device
        """
        self.group_dim = group_dim
        self.gauge_coupling = gauge_coupling
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Initialize gauge fields
        self.gauge_field = None
        self.policy = None
        
        # Track Wilson loops for monitoring
        self.wilson_loops = []
        
    def initialize_gauge_fields(self, tree_structure: Dict[str, Any]):
        """
        Initialize gauge fields on the MCTS tree lattice.
        
        Args:
            tree_structure: Dictionary containing tree information
        """
        # Extract adjacency from tree structure
        adjacency = self._extract_adjacency(tree_structure)
        
        # Initialize gauge field
        self.gauge_field = GaugeField(
            link_variables={},
            group_dim=self.group_dim,
            adjacency=adjacency
        )
        
        # Initialize policy field
        n_states = tree_structure.get('n_states', len(adjacency))
        n_actions = tree_structure.get('n_actions', 2)  # Default binary actions
        
        policy_field = {}
        for state in range(n_states):
            for action in range(n_actions):
                # Initialize with random hermitian matrix
                field_value = torch.randn(self.group_dim, self.group_dim, dtype=torch.float32)
                field_value = 0.5 * (field_value + field_value.conj().T)  # Make hermitian
                policy_field[(state, action)] = field_value
                
        self.policy = GaugeInvariantPolicy(
            policy_field=policy_field,
            group_dim=self.group_dim
        )
        
    def _extract_adjacency(self, tree_structure: Dict[str, Any]) -> Dict[int, List[int]]:
        """Extract adjacency list from tree structure"""
        adjacency = defaultdict(list)
        
        if 'edges' in tree_structure:
            for parent, child in tree_structure['edges']:
                adjacency[parent].append(child)
                adjacency[child].append(parent)  # Undirected for gauge theory
                
        return dict(adjacency)
    
    def compute_wilson_loop(self, path: List[int]) -> WilsonLoop:
        """
        Compute Wilson loop for a closed path.
        
        W(C) = Tr[∏_{links in C} U_link]
        
        Args:
            path: Closed path of nodes
            
        Returns:
            Wilson loop object
        """
        if len(path) < 3 or path[0] != path[-1]:
            raise ValueError("Path must be closed loop with at least 3 nodes")
            
        # Initialize with identity
        loop_product = torch.eye(self.group_dim, dtype=torch.float32)
        
        # Multiply link variables around the loop
        for i in range(len(path) - 1):
            node_i = path[i]
            node_j = path[i + 1]
            
            # Get link variable
            link_var = self.gauge_field.get_link(node_i, node_j)
            
            # Multiply (ordered product)
            loop_product = loop_product @ link_var
            
        # Wilson loop is the trace
        wilson_value = torch.trace(loop_product)
        
        return WilsonLoop(
            path=path,
            value=wilson_value,
            length=len(path) - 1
        )
    
    def find_fundamental_loops(self, max_length: int = 6) -> List[List[int]]:
        """
        Find fundamental loops in the tree structure.
        
        For trees, we need to add "virtual" edges to create loops.
        These represent policy consistency constraints.
        
        Args:
            max_length: Maximum loop length to consider
            
        Returns:
            List of fundamental loops
        """
        if self.gauge_field is None:
            return []
            
        # Create graph from adjacency
        G = nx.Graph()
        for node, neighbors in self.gauge_field.adjacency.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
                
        # Find all simple cycles (this will be empty for trees)
        cycles = []
        
        # For trees, create virtual loops using triangle completion
        virtual_loops = []
        nodes = list(G.nodes())
        
        for i, node in enumerate(nodes):
            neighbors = list(G.neighbors(node))
            # Create triangular loops with virtual edges
            for j in range(len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    neighbor1 = neighbors[j]
                    neighbor2 = neighbors[k]
                    
                    # Form triangle: node -> neighbor1 -> neighbor2 -> node
                    virtual_loop = [node, neighbor1, neighbor2, node]
                    if len(virtual_loop) <= max_length and len(virtual_loop) >= 4:
                        virtual_loops.append(virtual_loop)
                        
        # If no virtual loops found, create minimal loops from existing edges
        if not virtual_loops:
            for node in nodes:
                neighbors = list(G.neighbors(node))
                if len(neighbors) >= 2:
                    # Create a minimal loop with backtracking
                    neighbor1 = neighbors[0]
                    neighbor2 = neighbors[1]
                    minimal_loop = [node, neighbor1, node, neighbor2, node]
                    if len(minimal_loop) <= max_length:
                        virtual_loops.append(minimal_loop)
                        
        return cycles + virtual_loops
    
    def compute_policy_curvature(self, state: int) -> torch.Tensor:
        """
        Compute curvature of policy field at given state.
        
        Curvature F_μν = ∂_μ A_ν - ∂_ν A_μ + i[A_μ, A_ν]
        measures how much the gauge field deviates from being flat.
        
        Args:
            state: State to compute curvature at
            
        Returns:
            Curvature tensor (always hermitian)
        """
        if self.gauge_field is None:
            return torch.zeros(self.group_dim, self.group_dim)
            
        neighbors = self.gauge_field.adjacency.get(state, [])
        if len(neighbors) < 2:
            return torch.zeros(self.group_dim, self.group_dim)
            
        # Compute discrete curvature using plaquette
        curvature = torch.zeros(self.group_dim, self.group_dim, dtype=torch.float32)
        
        for i, neighbor1 in enumerate(neighbors):
            for j, neighbor2 in enumerate(neighbors[i+1:], i+1):
                # Get link variables
                A1 = self.gauge_field.get_link(state, neighbor1)
                A2 = self.gauge_field.get_link(state, neighbor2)
                
                # Discrete curvature: F ~ [A1, A2]
                commutator = A1 @ A2 - A2 @ A1
                curvature += commutator
                
        # Ensure curvature is hermitian (required for physical consistency)
        curvature = 0.5 * (curvature + curvature.conj().T)
        
        return curvature
    
    def gauge_invariant_update(self, state: int, action: int, 
                             value_gradient: torch.Tensor) -> torch.Tensor:
        """
        Perform gauge-invariant policy update.
        
        The update rule ensures that the policy remains gauge-invariant
        while incorporating information from the value function gradient.
        
        Args:
            state: State to update
            action: Action to update
            value_gradient: Gradient from value function
            
        Returns:
            Updated policy field value
        """
        if self.policy is None:
            raise ValueError("Policy not initialized")
            
        current_field = self.policy.policy_field.get((state, action))
        if current_field is None:
            return torch.zeros(self.group_dim, self.group_dim)
            
        # Compute covariant derivative
        covariant_grad = self._compute_covariant_derivative(state, action, value_gradient)
        
        # Gauge-invariant update rule
        # δφ = -η D_μ ∂V/∂φ where D_μ is covariant derivative
        learning_rate = 0.01
        update = -learning_rate * covariant_grad
        
        # Ensure update preserves hermiticity
        update = 0.5 * (update + update.conj().T)
        
        # Apply update
        new_field = current_field + update
        
        # Project back to group manifold if needed
        new_field = self._project_to_group_manifold(new_field)
        
        return new_field
    
    def _compute_covariant_derivative(self, state: int, action: int,
                                    gradient: torch.Tensor) -> torch.Tensor:
        """
        Compute covariant derivative D_μ φ = ∂_μ φ + i[A_μ, φ].
        
        Args:
            state: State
            action: Action
            gradient: Ordinary gradient
            
        Returns:
            Covariant derivative
        """
        if self.policy is None or self.gauge_field is None:
            return gradient
            
        # Get current field value
        current_field = self.policy.policy_field.get((state, action))
        if current_field is None:
            return gradient
            
        # Ensure gradient has correct dimensions
        if gradient.shape != current_field.shape:
            # Pad or truncate gradient to match field dimensions
            if gradient.shape[0] < self.group_dim:
                padded_gradient = torch.zeros(self.group_dim, self.group_dim, dtype=gradient.dtype)
                padded_gradient[:gradient.shape[0], :gradient.shape[1]] = gradient
                gradient = padded_gradient
            else:
                gradient = gradient[:self.group_dim, :self.group_dim]
            
        # Compute gauge connection contribution
        neighbors = self.gauge_field.adjacency.get(state, [])
        connection_term = torch.zeros_like(current_field)
        
        for neighbor in neighbors:
            # Get link variable
            link_var = self.gauge_field.get_link(state, neighbor)
            
            # Add connection term: i[A_μ, φ]
            commutator = 1j * (link_var @ current_field - current_field @ link_var)
            connection_term += commutator.real  # Take real part
            
        # Covariant derivative
        covariant_grad = gradient + self.gauge_coupling * connection_term
        
        return covariant_grad
    
    def _project_to_group_manifold(self, field: torch.Tensor) -> torch.Tensor:
        """
        Project field value back to group manifold.
        
        For SU(N), this means ensuring the matrix is unitary and traceless.
        
        Args:
            field: Field value to project
            
        Returns:
            Projected field value
        """
        if self.group_dim == 2:
            # For SU(2), use Pauli matrices basis
            # Ensure traceless and hermitian
            field_traceless = field - torch.trace(field) * torch.eye(2) / 2
            field_hermitian = 0.5 * (field_traceless + field_traceless.conj().T)
            return field_hermitian
        else:
            # General case: project to hermitian traceless matrices
            field_hermitian = 0.5 * (field + field.conj().T)
            trace_part = torch.trace(field_hermitian) / self.group_dim
            field_traceless = field_hermitian - trace_part * torch.eye(self.group_dim)
            return field_traceless
    
    def detect_gauge_anomalies(self, tolerance: float = 1e-6) -> List[Dict[str, Any]]:
        """
        Detect gauge anomalies by checking Wilson loop consistency.
        
        Gauge anomalies manifest as Wilson loops that don't satisfy
        expected symmetries or conservation laws.
        
        Args:
            tolerance: Tolerance for anomaly detection
            
        Returns:
            List of detected anomalies
        """
        if self.gauge_field is None:
            return []
            
        anomalies = []
        
        # Find fundamental loops
        loops = self.find_fundamental_loops()
        
        for loop_path in loops:
            try:
                # Compute Wilson loop
                wilson_loop = self.compute_wilson_loop(loop_path)
                
                # Check for anomalies
                # 1. Wilson loop should be approximately unitary
                loop_matrix = torch.eye(self.group_dim, dtype=torch.complex64)
                for i in range(len(loop_path) - 1):
                    node_i = loop_path[i]
                    node_j = loop_path[i + 1]
                    link_var = self.gauge_field.get_link(node_i, node_j)
                    loop_matrix = loop_matrix @ link_var.to(torch.complex64)
                
                # Check unitarity: U†U = I
                unitarity_check = loop_matrix.conj().T @ loop_matrix
                identity = torch.eye(self.group_dim, dtype=torch.complex64)
                unitarity_error = torch.norm(unitarity_check - identity)
                
                if unitarity_error > tolerance:
                    anomalies.append({
                        'type': 'unitarity_violation',
                        'loop': loop_path,
                        'error': float(unitarity_error),
                        'wilson_value': wilson_loop.value
                    })
                    
                # 2. Check gauge invariance of Wilson loop
                # Wilson loop should be real for hermitian gauge fields
                if torch.abs(wilson_loop.value.imag) > tolerance:
                    anomalies.append({
                        'type': 'complex_wilson_loop',
                        'loop': loop_path,
                        'imaginary_part': float(wilson_loop.value.imag),
                        'wilson_value': wilson_loop.value
                    })
                    
            except Exception as e:
                anomalies.append({
                    'type': 'computation_error',
                    'loop': loop_path,
                    'error': str(e)
                })
                
        return anomalies
    
    def apply_gauge_fixing(self, gauge_condition: str = "coulomb") -> Dict[str, Any]:
        """
        Apply gauge fixing to reduce gauge redundancy.
        
        Gauge fixing breaks gauge invariance in a controlled way to
        make computations more efficient while preserving physical content.
        
        Args:
            gauge_condition: Type of gauge fixing ("coulomb", "landau", "axial")
            
        Returns:
            Gauge fixing results
        """
        if self.gauge_field is None or self.policy is None:
            return {'status': 'error', 'message': 'Fields not initialized'}
            
        if gauge_condition == "coulomb":
            return self._apply_coulomb_gauge()
        elif gauge_condition == "landau":
            return self._apply_landau_gauge()
        elif gauge_condition == "axial":
            return self._apply_axial_gauge()
        else:
            raise ValueError(f"Unknown gauge condition: {gauge_condition}")
    
    def _apply_coulomb_gauge(self) -> Dict[str, Any]:
        """
        Apply Coulomb gauge fixing: ∇ · A = 0.
        
        This removes the scalar (temporal) component of the gauge field.
        """
        # For discrete lattice, Coulomb gauge means
        # Σ_μ (A_μ(x) - A_μ(x-μ)) = 0
        
        gauge_transformations = {}
        
        # Compute divergence of gauge field at each node
        for node in self.gauge_field.adjacency:
            neighbors = self.gauge_field.adjacency[node]
            
            # Compute discrete divergence
            divergence = torch.zeros(self.group_dim, self.group_dim, dtype=torch.float32)
            
            for neighbor in neighbors:
                link_forward = self.gauge_field.get_link(node, neighbor)
                link_backward = self.gauge_field.get_link(neighbor, node)
                
                # Discrete divergence: A_μ(x) - A_μ(x-μ)
                divergence += link_forward - link_backward
                
            # Generate gauge transformation to minimize divergence
            # This is a simplified approach - full implementation would solve Poisson equation
            if torch.norm(divergence) > 1e-6:
                # Generate gauge transformation U = exp(iα) to reduce divergence
                alpha = -0.1 * divergence  # Small parameter
                U = torch.matrix_exp(1j * alpha).to(torch.float32)
                gauge_transformations[node] = U
                
        # Apply gauge transformations
        if gauge_transformations:
            # Transform policy field
            self.policy = self.policy.gauge_transform(gauge_transformations)
            
            # Transform gauge field
            self._transform_gauge_field(gauge_transformations)
            
        return {
            'status': 'success',
            'gauge_condition': 'coulomb',
            'transformations_applied': len(gauge_transformations)
        }
    
    def _apply_landau_gauge(self) -> Dict[str, Any]:
        """
        Apply Landau gauge fixing: ∂_μ A_μ = 0.
        
        This is the covariant generalization of Coulomb gauge.
        """
        # Landau gauge requires solving: ∂_μ D_μ α = -∂_μ A_μ
        # where D_μ is the covariant derivative
        
        # For simplicity, we use a relaxation method
        gauge_transformations = {}
        max_iterations = 100
        tolerance = 1e-6
        
        for iteration in range(max_iterations):
            total_change = 0.0
            
            for node in self.gauge_field.adjacency:
                neighbors = self.gauge_field.adjacency[node]
                
                # Compute gauge-covariant divergence
                covariant_divergence = torch.zeros(self.group_dim, self.group_dim, dtype=torch.float32)
                
                for neighbor in neighbors:
                    link_var = self.gauge_field.get_link(node, neighbor)
                    # Simplified covariant divergence
                    covariant_divergence += link_var - torch.eye(self.group_dim)
                    
                # Generate small gauge transformation
                if torch.norm(covariant_divergence) > tolerance:
                    alpha = -0.01 * covariant_divergence
                    U = torch.matrix_exp(1j * alpha).to(torch.float32)
                    
                    if node not in gauge_transformations:
                        gauge_transformations[node] = U
                    else:
                        gauge_transformations[node] = gauge_transformations[node] @ U
                        
                    total_change += torch.norm(alpha)
                    
            if total_change < tolerance:
                break
                
        # Apply final gauge transformations
        if gauge_transformations:
            self.policy = self.policy.gauge_transform(gauge_transformations)
            self._transform_gauge_field(gauge_transformations)
            
        return {
            'status': 'success',
            'gauge_condition': 'landau',
            'iterations': iteration + 1,
            'transformations_applied': len(gauge_transformations)
        }
    
    def _apply_axial_gauge(self) -> Dict[str, Any]:
        """
        Apply axial gauge fixing: A_μ(x) · n_μ = 0.
        
        This sets the gauge field to zero in a chosen direction.
        """
        # Choose axial direction (e.g., towards root of tree)
        # For MCTS tree, we can choose direction towards root
        
        root_node = min(self.gauge_field.adjacency.keys())  # Assume smallest index is root
        gauge_transformations = {}
        
        # Set gauge field to zero in direction towards root
        for node in self.gauge_field.adjacency:
            if node != root_node:
                # Find path to root (simplified - use direct connection if exists)
                if root_node in self.gauge_field.adjacency[node]:
                    # Set link to root to identity
                    link_to_root = self.gauge_field.get_link(node, root_node)
                    
                    # Generate gauge transformation to make this link identity
                    if torch.norm(link_to_root - torch.eye(self.group_dim)) > 1e-6:
                        # U such that U A U† = I, so U = A†
                        U = link_to_root.conj().T
                        gauge_transformations[node] = U
                        
        # Apply gauge transformations
        if gauge_transformations:
            self.policy = self.policy.gauge_transform(gauge_transformations)
            self._transform_gauge_field(gauge_transformations)
            
        return {
            'status': 'success',
            'gauge_condition': 'axial',
            'transformations_applied': len(gauge_transformations)
        }
    
    def _transform_gauge_field(self, gauge_transformations: Dict[int, torch.Tensor]):
        """
        Apply gauge transformations to the gauge field.
        
        A'_μ(x) = U(x) A_μ(x) U†(x+μ) + U(x) ∂_μ U†(x)
        
        Args:
            gauge_transformations: U(x) for each node x
        """
        new_link_variables = {}
        
        for (node_i, node_j), link_var in self.gauge_field.link_variables.items():
            U_i = gauge_transformations.get(node_i, torch.eye(self.group_dim))
            U_j = gauge_transformations.get(node_j, torch.eye(self.group_dim))
            
            # Transform link variable: A'_ij = U_i A_ij U_j†
            new_link_var = U_i @ link_var @ U_j.conj().T
            new_link_variables[(node_i, node_j)] = new_link_var
            
        self.gauge_field.link_variables = new_link_variables
    
    def compute_gauge_invariant_observables(self) -> Dict[str, Any]:
        """
        Compute gauge-invariant observables for monitoring.
        
        Returns:
            Dictionary of gauge-invariant quantities
        """
        observables = {}
        
        if self.gauge_field is None:
            return observables
            
        # 1. Wilson loops
        loops = self.find_fundamental_loops()
        wilson_values = []
        
        for loop_path in loops:
            try:
                wilson_loop = self.compute_wilson_loop(loop_path)
                wilson_values.append(float(wilson_loop.value.real))
            except:
                continue
                
        observables['wilson_loops'] = {
            'values': wilson_values,
            'mean': np.mean(wilson_values) if wilson_values else 0.0,
            'std': np.std(wilson_values) if wilson_values else 0.0
        }
        
        # 2. Plaquette (average Wilson loop for elementary squares)
        plaquette_values = []
        for loop_path in loops:
            if len(loop_path) == 4:  # Elementary square
                try:
                    wilson_loop = self.compute_wilson_loop(loop_path)
                    plaquette_values.append(float(wilson_loop.value.real))
                except:
                    continue
                    
        observables['plaquette'] = {
            'values': plaquette_values,
            'mean': np.mean(plaquette_values) if plaquette_values else 0.0
        }
        
        # 3. Policy field strength
        field_strengths = []
        for node in self.gauge_field.adjacency:
            curvature = self.compute_policy_curvature(node)
            field_strength = torch.norm(curvature)
            field_strengths.append(float(field_strength))
            
        observables['field_strength'] = {
            'values': field_strengths,
            'mean': np.mean(field_strengths) if field_strengths else 0.0,
            'max': np.max(field_strengths) if field_strengths else 0.0
        }
        
        # 4. Gauge anomalies
        anomalies = self.detect_gauge_anomalies()
        observables['anomalies'] = {
            'count': len(anomalies),
            'types': [a['type'] for a in anomalies]
        }
        
        return observables
    
    def optimize_gauge_action(self, beta: float = 1.0) -> Dict[str, Any]:
        """
        Optimize gauge field configuration using Wilson action.
        
        The Wilson action is: S = β Σ_plaquettes (1 - Re[W_plaquette])
        
        Args:
            beta: Inverse coupling constant
            
        Returns:
            Optimization results
        """
        if self.gauge_field is None:
            return {'status': 'error', 'message': 'Gauge field not initialized'}
            
        # Find all plaquettes (elementary loops)
        plaquettes = [path for path in self.find_fundamental_loops() if len(path) == 4]
        
        if not plaquettes:
            return {'status': 'error', 'message': 'No plaquettes found'}
            
        def wilson_action(link_params):
            """Compute Wilson action for given link parameters"""
            # Reshape parameters back to link variables
            param_idx = 0
            for (node_i, node_j) in self.gauge_field.link_variables:
                n_params = self.group_dim * self.group_dim
                link_matrix = link_params[param_idx:param_idx + n_params].reshape(self.group_dim, self.group_dim)
                self.gauge_field.link_variables[(node_i, node_j)] = torch.tensor(link_matrix, dtype=torch.float32)
                param_idx += n_params
                
            # Compute action
            action = 0.0
            for plaquette in plaquettes:
                try:
                    wilson_loop = self.compute_wilson_loop(plaquette)
                    action += 1.0 - wilson_loop.value.real
                except:
                    continue
                    
            return beta * action
        
        # Flatten current link variables for optimization
        initial_params = []
        for link_var in self.gauge_field.link_variables.values():
            initial_params.extend(link_var.flatten().tolist())
            
        initial_params = np.array(initial_params)
        
        # Optimize
        result = minimize(wilson_action, initial_params, method='L-BFGS-B')
        
        # Extract final observables
        observables = self.compute_gauge_invariant_observables()
        
        return {
            'status': 'success',
            'optimization_result': result,
            'final_action': result.fun,
            'observables': observables
        }


def apply_gauge_invariant_policy_learning(tree_structure: Dict[str, Any],
                                        policy_updates: Dict[Tuple[int, int], torch.Tensor],
                                        config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Apply gauge-invariant policy learning to MCTS tree.
    
    Args:
        tree_structure: Tree structure information
        policy_updates: Policy gradients for each state-action
        config: Configuration parameters
        
    Returns:
        Results from gauge-invariant policy learning
    """
    if config is None:
        config = {}
        
    # Initialize gauge learner
    learner = GaugeInvariantPolicyLearner(
        group_dim=config.get('group_dim', 2),
        gauge_coupling=config.get('gauge_coupling', 1.0)
    )
    
    # Initialize gauge fields
    learner.initialize_gauge_fields(tree_structure)
    
    # Apply gauge-invariant updates
    updated_policy = {}
    for (state, action), gradient in policy_updates.items():
        updated_field = learner.gauge_invariant_update(state, action, gradient)
        updated_policy[(state, action)] = updated_field
        
    # Apply gauge fixing
    gauge_fixing_result = learner.apply_gauge_fixing(config.get('gauge_condition', 'coulomb'))
    
    # Compute observables
    observables = learner.compute_gauge_invariant_observables()
    
    # Detect anomalies
    anomalies = learner.detect_gauge_anomalies()
    
    # Optimize gauge action
    optimization_result = learner.optimize_gauge_action(config.get('beta', 1.0))
    
    return {
        'updated_policy': updated_policy,
        'gauge_fixing': gauge_fixing_result,
        'observables': observables,
        'anomalies': anomalies,
        'optimization': optimization_result,
        'wilson_loops': [
            {
                'path': loop.path,
                'value': float(loop.value.real),
                'length': loop.length
            }
            for loop in learner.wilson_loops
        ]
    }