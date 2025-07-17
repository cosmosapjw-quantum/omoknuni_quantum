"""
Topological Analysis for MCTS

This module implements topological data analysis methods for MCTS:
1. Persistent homology of value landscapes
2. Morse theory for critical point analysis
3. Topological phase detection in search behavior
4. Simplicial complex analysis of tree structure
5. Homotopy type analysis of decision boundaries

Mathematical Foundation:
- Persistent homology: H_k(K_t) for filtration parameter t
- Morse theory: Critical points and gradient flows
- Simplicial homology: Betti numbers β_k = rank(H_k)
- Homotopy equivalence: Topological invariants
- Topological phase transitions: Changes in Betti numbers

Physical Interpretation:
- Value landscapes as Morse functions on strategy space
- Critical points as phase transitions in search behavior
- Persistent features as robust strategic patterns
- Topological invariants as universal game properties
- Homotopy classes as equivalence classes of strategies
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize_scalar
from collections import defaultdict
import itertools
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import logging

logger = logging.getLogger(__name__)


@dataclass
class PersistentFeature:
    """
    Represents a persistent homological feature.
    
    Persistent features correspond to topological holes that persist
    across multiple filtration levels.
    """
    dimension: int  # Homological dimension (0, 1, 2, ...)
    birth_time: float  # Filtration parameter when feature appears
    death_time: float  # Filtration parameter when feature disappears
    persistence: float  # death_time - birth_time
    representative_cycle: Optional[List[int]]  # Representative cycle
    
    def __post_init__(self):
        """Calculate persistence"""
        if self.death_time == float('inf'):
            self.persistence = float('inf')
        else:
            self.persistence = self.death_time - self.birth_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary"""
        return {
            'dimension': self.dimension,
            'birth_time': self.birth_time,
            'death_time': self.death_time,
            'persistence': self.persistence,
            'representative_cycle': self.representative_cycle
        }


@dataclass
class MorseCriticalPoint:
    """
    Represents a critical point in Morse theory.
    
    Critical points are where the gradient of the value function vanishes.
    """
    position: List[float]  # Position in strategy space
    value: float  # Function value at critical point
    morse_index: int  # Number of negative eigenvalues of Hessian
    stability: str  # 'stable', 'unstable', 'saddle'
    basin_size: float  # Size of basin of attraction/repulsion
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary"""
        return {
            'position': self.position,
            'value': self.value,
            'morse_index': self.morse_index,
            'stability': self.stability,
            'basin_size': self.basin_size
        }


@dataclass
class TopologicalPhase:
    """
    Represents a topological phase of the search process.
    
    Topological phases are characterized by topological invariants
    that remain constant under continuous deformations.
    """
    betti_numbers: List[int]  # Betti numbers β_k
    euler_characteristic: int  # χ = Σ(-1)^k β_k
    phase_label: str  # Phase classification
    transition_points: List[float]  # Phase transition locations
    order_parameter: float  # Topological order parameter
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary"""
        return {
            'betti_numbers': self.betti_numbers,
            'euler_characteristic': self.euler_characteristic,
            'phase_label': self.phase_label,
            'transition_points': self.transition_points,
            'order_parameter': self.order_parameter
        }


class SimplicalComplex:
    """
    Represents a simplicial complex for topological analysis.
    
    Simplicial complexes are generalizations of graphs where
    higher-dimensional simplicies (triangles, tetrahedra) are allowed.
    """
    
    def __init__(self, vertices: List[int], simplicies: Dict[int, List[List[int]]]):
        """
        Initialize simplicial complex.
        
        Args:
            vertices: List of vertex indices
            simplicies: Dictionary mapping dimension to list of simplicies
        """
        self.vertices = vertices
        self.simplicies = simplicies
        self.n_vertices = len(vertices)
        
        # Compute boundary matrices
        self.boundary_matrices = self._compute_boundary_matrices()
        
        # Compute Betti numbers
        self.betti_numbers = self._compute_betti_numbers()
    
    def _compute_boundary_matrices(self) -> Dict[int, csr_matrix]:
        """Compute boundary matrices for each dimension"""
        boundary_matrices = {}
        
        for dim in range(1, max(self.simplicies.keys()) + 1):
            if dim in self.simplicies and (dim - 1) in self.simplicies:
                # Create boundary matrix from dim-simplicies to (dim-1)-simplicies
                n_k = len(self.simplicies[dim])
                n_k_minus_1 = len(self.simplicies[dim - 1])
                
                if n_k > 0 and n_k_minus_1 > 0:
                    boundary_matrix = self._create_boundary_matrix(dim)
                    boundary_matrices[dim] = csr_matrix(boundary_matrix)
        
        return boundary_matrices
    
    def _create_boundary_matrix(self, dim: int) -> np.ndarray:
        """Create boundary matrix for given dimension"""
        k_simplices = self.simplicies[dim]
        k_minus_1_simplices = self.simplicies[dim - 1]
        
        # Create index mapping for (k-1)-simplices
        simplex_to_index = {}
        for i, simplex in enumerate(k_minus_1_simplices):
            simplex_to_index[tuple(sorted(simplex))] = i
        
        # Initialize boundary matrix
        boundary = np.zeros((len(k_minus_1_simplices), len(k_simplices)))
        
        # Fill boundary matrix
        for j, k_simplex in enumerate(k_simplices):
            for i in range(len(k_simplex)):
                # Create (k-1)-face by removing i-th vertex
                face = k_simplex[:i] + k_simplex[i+1:]
                face_key = tuple(sorted(face))
                
                if face_key in simplex_to_index:
                    face_index = simplex_to_index[face_key]
                    boundary[face_index, j] = (-1) ** i
        
        return boundary
    
    def _compute_betti_numbers(self) -> List[int]:
        """Compute Betti numbers using boundary matrices"""
        betti_numbers = []
        
        # β_0 = number of connected components
        if 0 in self.simplicies:
            # Create graph from 1-simplicies (edges)
            if 1 in self.simplicies:
                edges = self.simplicies[1]
                if edges:
                    G = nx.Graph()
                    G.add_edges_from(edges)
                    betti_0 = nx.number_connected_components(G)
                else:
                    betti_0 = len(self.vertices)
            else:
                betti_0 = len(self.vertices)
            betti_numbers.append(betti_0)
        
        # β_k = dim(ker(∂_k)) - dim(im(∂_{k+1})) for k ≥ 1
        for k in range(1, len(self.boundary_matrices) + 1):
            if k in self.boundary_matrices:
                # Compute rank of boundary matrix
                boundary_k = self.boundary_matrices[k]
                rank_boundary_k = np.linalg.matrix_rank(boundary_k.toarray())
                
                # Compute dimension of kernel
                n_k_simplicies = len(self.simplicies.get(k, []))
                dim_kernel_k = n_k_simplicies - rank_boundary_k
                
                # Compute dimension of image of next boundary matrix
                if (k + 1) in self.boundary_matrices:
                    boundary_k_plus_1 = self.boundary_matrices[k + 1]
                    dim_image_k_plus_1 = np.linalg.matrix_rank(boundary_k_plus_1.toarray())
                else:
                    dim_image_k_plus_1 = 0
                
                # Betti number
                betti_k = dim_kernel_k - dim_image_k_plus_1
                betti_numbers.append(max(0, betti_k))
        
        return betti_numbers
    
    def euler_characteristic(self) -> int:
        """Compute Euler characteristic χ = Σ(-1)^k β_k"""
        chi = 0
        for k, beta_k in enumerate(self.betti_numbers):
            chi += (-1) ** k * beta_k
        return chi


class TopologicalAnalyzer:
    """
    Comprehensive topological analysis for MCTS.
    
    Implements persistent homology, Morse theory, and topological
    phase detection for understanding search behavior.
    """
    
    def __init__(self, max_dimension: int = 2, filtration_steps: int = 50):
        """
        Initialize topological analyzer.
        
        Args:
            max_dimension: Maximum homological dimension to compute
            filtration_steps: Number of filtration steps
        """
        self.max_dimension = max_dimension
        self.filtration_steps = filtration_steps
        
        # Storage for computed features
        self.persistent_features: List[PersistentFeature] = []
        self.critical_points: List[MorseCriticalPoint] = []
        self.topological_phases: List[TopologicalPhase] = []
    
    def compute_persistent_homology(self, tree_data: Dict[str, Any]) -> List[PersistentFeature]:
        """
        Compute persistent homology of value landscape.
        
        Args:
            tree_data: Dictionary containing tree structure and values
            
        Returns:
            List of persistent features
        """
        # Extract values and positions
        values = tree_data.get('values', [])
        positions = tree_data.get('positions', [])
        
        if not values or not positions:
            logger.warning("No values or positions found in tree data")
            return []
        
        # Convert to numpy arrays
        values = np.array(values)
        positions = np.array(positions)
        
        # Create filtration
        filtration_values = np.linspace(np.min(values), np.max(values), self.filtration_steps)
        
        # Compute persistent homology for each filtration level
        persistent_features = []
        
        # Track births and deaths of connected components (H_0)
        component_births = {}
        component_deaths = {}
        
        for i, threshold in enumerate(filtration_values):
            # Create sublevel set
            sublevel_indices = np.where(values <= threshold)[0]
            
            if len(sublevel_indices) == 0:
                continue
            
            # Build simplicial complex for sublevel set
            simplicial_complex = self._build_vietoris_rips_complex(
                positions[sublevel_indices], threshold
            )
            
            # Compute Betti numbers
            betti_numbers = simplicial_complex.betti_numbers
            
            # Track changes in topology
            if i == 0:
                # Initial components
                for j in range(betti_numbers[0]):
                    component_births[j] = threshold
            else:
                # Track births and deaths
                prev_betti_0 = len([f for f in persistent_features if f.dimension == 0 and f.death_time == float('inf')])
                current_betti_0 = betti_numbers[0]
                
                # Handle component mergers (deaths)
                if current_betti_0 < prev_betti_0:
                    # Some components died
                    deaths_needed = prev_betti_0 - current_betti_0
                    for _ in range(deaths_needed):
                        # Find youngest component to kill
                        youngest_birth = max(component_births.values())
                        for comp_id, birth_time in component_births.items():
                            if birth_time == youngest_birth and comp_id not in component_deaths:
                                component_deaths[comp_id] = threshold
                                break
                
                # Handle component births
                if current_betti_0 > prev_betti_0:
                    # New components born
                    births_needed = current_betti_0 - prev_betti_0
                    for _ in range(births_needed):
                        new_id = len(component_births)
                        component_births[new_id] = threshold
        
        # Create persistent features from birth/death tracking
        for comp_id, birth_time in component_births.items():
            death_time = component_deaths.get(comp_id, float('inf'))
            
            feature = PersistentFeature(
                dimension=0,
                birth_time=birth_time,
                death_time=death_time,
                persistence=death_time - birth_time if death_time != float('inf') else float('inf'),
                representative_cycle=None
            )
            persistent_features.append(feature)
        
        self.persistent_features = persistent_features
        return persistent_features
    
    def _build_vietoris_rips_complex(self, points: np.ndarray, threshold: float) -> SimplicalComplex:
        """Build Vietoris-Rips complex for given points and threshold"""
        n_points = len(points)
        
        # Compute distance matrix
        distances = squareform(pdist(points))
        
        # Build simplicies
        simplicies = {0: [[i] for i in range(n_points)]}  # 0-simplices (vertices)
        
        # 1-simplices (edges)
        edges = []
        for i in range(n_points):
            for j in range(i + 1, n_points):
                if distances[i, j] <= threshold:
                    edges.append([i, j])
        
        if edges:
            simplicies[1] = edges
        
        # 2-simplices (triangles) - for max_dimension >= 2
        if self.max_dimension >= 2 and edges:
            triangles = []
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    for k in range(j + 1, n_points):
                        if (distances[i, j] <= threshold and 
                            distances[j, k] <= threshold and 
                            distances[i, k] <= threshold):
                            triangles.append([i, j, k])
            
            if triangles:
                simplicies[2] = triangles
        
        vertices = list(range(n_points))
        return SimplicalComplex(vertices, simplicies)
    
    def compute_morse_critical_points(self, tree_data: Dict[str, Any]) -> List[MorseCriticalPoint]:
        """
        Compute critical points using Morse theory.
        
        Args:
            tree_data: Dictionary containing tree structure and values
            
        Returns:
            List of critical points
        """
        values = tree_data.get('values', [])
        positions = tree_data.get('positions', [])
        
        if not values or not positions:
            logger.warning("No values or positions found in tree data")
            return []
        
        values = np.array(values)
        positions = np.array(positions)
        
        critical_points = []
        
        # Find local extrema as critical points
        for i in range(len(values)):
            # Build local neighborhood
            distances = np.linalg.norm(positions - positions[i], axis=1)
            neighbor_indices = np.where(distances < 0.1)[0]  # Adjust threshold as needed
            
            if len(neighbor_indices) > 1:
                neighbor_values = values[neighbor_indices]
                current_value = values[i]
                
                # Check if it's a local extremum
                is_maximum = np.all(neighbor_values <= current_value)
                is_minimum = np.all(neighbor_values >= current_value)
                
                if is_maximum or is_minimum:
                    # Estimate Morse index (simplified)
                    morse_index = 0 if is_minimum else len(positions[0]) if is_maximum else 1
                    
                    stability = 'stable' if is_minimum else 'unstable' if is_maximum else 'saddle'
                    
                    # Estimate basin size
                    basin_size = len(neighbor_indices) / len(values)
                    
                    critical_point = MorseCriticalPoint(
                        position=positions[i].tolist(),
                        value=current_value,
                        morse_index=morse_index,
                        stability=stability,
                        basin_size=basin_size
                    )
                    critical_points.append(critical_point)
        
        self.critical_points = critical_points
        return critical_points
    
    def detect_topological_phases(self, tree_data: Dict[str, Any]) -> List[TopologicalPhase]:
        """
        Detect topological phases in search behavior.
        
        Args:
            tree_data: Dictionary containing tree structure and values
            
        Returns:
            List of topological phases
        """
        # Use persistent homology to detect phases
        persistent_features = self.compute_persistent_homology(tree_data)
        
        # Group features by birth time to identify phases
        phase_transitions = []
        betti_evolution = []
        
        # Track evolution of Betti numbers
        values = np.array(tree_data.get('values', []))
        
        if len(values) == 0:
            return []
        
        filtration_values = np.linspace(np.min(values), np.max(values), self.filtration_steps)
        
        for threshold in filtration_values:
            # Count features alive at this threshold
            betti_numbers = [0] * (self.max_dimension + 1)
            
            for feature in persistent_features:
                if (feature.birth_time <= threshold and 
                    (feature.death_time > threshold or feature.death_time == float('inf'))):
                    betti_numbers[feature.dimension] += 1
            
            betti_evolution.append(betti_numbers)
        
        # Detect phase transitions (changes in Betti numbers)
        phases = []
        current_betti = betti_evolution[0]
        phase_start = filtration_values[0]
        
        for i, betti_numbers in enumerate(betti_evolution[1:], 1):
            if betti_numbers != current_betti:
                # Phase transition detected
                phase_transitions.append(filtration_values[i])
                
                # Create phase for previous segment
                euler_char = sum((-1) ** k * beta_k for k, beta_k in enumerate(current_betti))
                order_parameter = float(np.sum(current_betti))  # Simple order parameter
                
                phase = TopologicalPhase(
                    betti_numbers=current_betti,
                    euler_characteristic=euler_char,
                    phase_label=f"Phase_{len(phases)}",
                    transition_points=phase_transitions[-1:],
                    order_parameter=order_parameter
                )
                phases.append(phase)
                
                current_betti = betti_numbers
                phase_start = filtration_values[i]
        
        # Add final phase
        if current_betti:
            euler_char = sum((-1) ** k * beta_k for k, beta_k in enumerate(current_betti))
            order_parameter = float(np.sum(current_betti))
            
            phase = TopologicalPhase(
                betti_numbers=current_betti,
                euler_characteristic=euler_char,
                phase_label=f"Phase_{len(phases)}",
                transition_points=[],
                order_parameter=order_parameter
            )
            phases.append(phase)
        
        self.topological_phases = phases
        return phases
    
    def analyze_decision_boundaries(self, tree_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze decision boundaries using homotopy theory.
        
        Args:
            tree_data: Dictionary containing tree structure and values
            
        Returns:
            Dictionary with homotopy analysis results
        """
        values = tree_data.get('values', [])
        positions = tree_data.get('positions', [])
        
        if not values or not positions:
            return {'error': 'No values or positions found'}
        
        values = np.array(values)
        positions = np.array(positions)
        
        # Find decision boundaries (level sets)
        decision_thresholds = np.percentile(values, [25, 50, 75])
        
        boundary_analysis = {}
        
        for i, threshold in enumerate(decision_thresholds):
            # Create level set
            level_set_indices = np.where(np.abs(values - threshold) < 0.1)[0]
            
            if len(level_set_indices) > 2:
                level_set_positions = positions[level_set_indices]
                
                # Build simplicial complex for level set
                simplicial_complex = self._build_vietoris_rips_complex(
                    level_set_positions, threshold=0.2
                )
                
                # Compute topological invariants
                betti_numbers = simplicial_complex.betti_numbers
                euler_char = simplicial_complex.euler_characteristic()
                
                boundary_analysis[f'threshold_{i}'] = {
                    'threshold': threshold,
                    'betti_numbers': betti_numbers,
                    'euler_characteristic': euler_char,
                    'n_components': len(level_set_indices)
                }
        
        return boundary_analysis
    
    def compute_topological_complexity(self, tree_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute various measures of topological complexity.
        
        Args:
            tree_data: Dictionary containing tree structure and values
            
        Returns:
            Dictionary with complexity measures
        """
        # Compute persistent homology
        persistent_features = self.compute_persistent_homology(tree_data)
        
        # Compute complexity measures
        complexity_measures = {}
        
        # Persistence entropy
        if persistent_features:
            persistences = [f.persistence for f in persistent_features if f.persistence != float('inf')]
            if persistences:
                total_persistence = sum(persistences)
                if total_persistence > 0:
                    normalized_persistences = [p / total_persistence for p in persistences]
                    persistence_entropy = -sum(p * np.log(p) for p in normalized_persistences if p > 0)
                    complexity_measures['persistence_entropy'] = persistence_entropy
        
        # Betti number complexity
        max_betti = max(len(phase.betti_numbers) for phase in self.topological_phases) if self.topological_phases else 0
        complexity_measures['betti_complexity'] = max_betti
        
        # Phase transition complexity
        total_transitions = sum(len(phase.transition_points) for phase in self.topological_phases)
        complexity_measures['phase_transition_complexity'] = total_transitions
        
        # Critical point complexity
        complexity_measures['critical_point_complexity'] = len(self.critical_points)
        
        # Overall topological complexity
        complexity_measures['overall_complexity'] = (
            complexity_measures.get('persistence_entropy', 0) +
            complexity_measures.get('betti_complexity', 0) +
            complexity_measures.get('phase_transition_complexity', 0) +
            complexity_measures.get('critical_point_complexity', 0)
        )
        
        return complexity_measures
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of topological analysis.
        
        Returns:
            Dictionary with analysis summary
        """
        return {
            'persistent_features': [f.to_dict() for f in self.persistent_features],
            'critical_points': [cp.to_dict() for cp in self.critical_points],
            'topological_phases': [tp.to_dict() for tp in self.topological_phases],
            'n_persistent_features': len(self.persistent_features),
            'n_critical_points': len(self.critical_points),
            'n_topological_phases': len(self.topological_phases),
            'max_persistence': max([f.persistence for f in self.persistent_features if f.persistence != float('inf')], default=0),
            'total_phase_transitions': sum(len(phase.transition_points) for phase in self.topological_phases)
        }


def apply_topological_analysis(tree_data: Dict[str, Any], 
                             max_dimension: int = 2,
                             filtration_steps: int = 50) -> Dict[str, Any]:
    """
    Apply comprehensive topological analysis to MCTS tree data.
    
    Args:
        tree_data: Dictionary containing tree structure and values
        max_dimension: Maximum homological dimension to compute
        filtration_steps: Number of filtration steps
        
    Returns:
        Dictionary with topological analysis results
    """
    analyzer = TopologicalAnalyzer(max_dimension, filtration_steps)
    
    # Compute all topological analyses
    persistent_features = analyzer.compute_persistent_homology(tree_data)
    critical_points = analyzer.compute_morse_critical_points(tree_data)
    topological_phases = analyzer.detect_topological_phases(tree_data)
    decision_boundaries = analyzer.analyze_decision_boundaries(tree_data)
    complexity_measures = analyzer.compute_topological_complexity(tree_data)
    
    # Get summary
    summary = analyzer.get_analysis_summary()
    
    return {
        'persistent_features': [f.to_dict() for f in persistent_features],
        'critical_points': [cp.to_dict() for cp in critical_points],
        'topological_phases': [tp.to_dict() for tp in topological_phases],
        'decision_boundaries': decision_boundaries,
        'complexity_measures': complexity_measures,
        'summary': summary
    }