"""
Holographic Entropy Bounds for MCTS

EXPERIMENTAL MODULE: This module implements holographic entropy bounds inspired by the 
Ryu-Takayanagi formula from AdS/CFT correspondence. This is an experimental extension
beyond the core theoretical framework in quantum_mcts_foundation.md.

The key insight is that MCTS tree structure naturally exhibits emergent hyperbolic 
geometry, where:

- Tree depth corresponds to the emergent bulk dimension
- Leaves form the boundary where decisions are made
- Information flow follows holographic principles
- Computational complexity is bounded by minimal surfaces

Key concepts:
1. Ryu-Takayanagi surfaces: Minimal surfaces that bound entanglement entropy
2. Emergent hyperbolic geometry: Tree metric induces AdS-like structure
3. Holographic screens: Fixed-depth surfaces that encode information
4. Complexity bounds: Surface area bounds computational requirements

INTERPRETATION: This should be treated as a phenomenological tool for analyzing
tree structure rather than a fundamental physical principle.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import logging
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class HolographicSurface:
    """Represents a minimal surface in the tree geometry"""
    nodes: List[int]  # Nodes defining the surface
    area: float  # "Area" of surface (sum of edge weights)
    depth: int  # Average depth of surface
    entropy_bound: float  # Entropy bounded by this surface
    

@dataclass 
class BulkBoundaryMap:
    """Maps between bulk (tree interior) and boundary (leaves)"""
    bulk_nodes: Set[int]
    boundary_nodes: Set[int]
    radial_coordinates: Dict[int, float]  # Depth-based radial coord
    angular_coordinates: Dict[int, float]  # Branch-based angular coord
    metric_tensor: np.ndarray  # Discrete metric on tree


class HolographicBoundsAnalyzer:
    """
    Implements holographic entropy bounds for MCTS trees.
    
    The holographic principle states that information in a volume
    is bounded by the area of its boundary. In MCTS:
    - Volume = subtree computation
    - Area = cut through tree (RT surface)
    - Information = entanglement entropy
    
    This provides fundamental bounds on:
    1. Minimum simulations needed (area law)
    2. Information compression limits
    3. Optimal tree pruning strategies
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize holographic analyzer.
        
        Args:
            device: Computation device
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # AdS radius parameter (controls curvature)
        self.ads_radius = 1.0
        
        # Initialize quantum definitions
        try:
            from .quantum_definitions import UnifiedQuantumDefinitions, compute_von_neumann_entropy, compute_purity, compute_coherence
        except ImportError:
            from quantum_definitions import UnifiedQuantumDefinitions, compute_von_neumann_entropy, compute_purity, compute_coherence
        self.quantum_defs = UnifiedQuantumDefinitions(device=self.device)
        
        # Store compute functions as attributes for use in methods
        self.compute_von_neumann_entropy = compute_von_neumann_entropy
        self.compute_purity = compute_purity
        self.compute_coherence = compute_coherence
        
    def compute_tree_metric(self, tree_structure: Dict[str, Any]) -> np.ndarray:
        """
        Compute discrete metric on tree with emergent hyperbolic geometry.
        
        The metric assigns distances based on:
        - Graph distance weighted by visit counts (geodesics)
        - Hyperbolic scaling with depth (AdS radial direction)
        - Information-theoretic distance from value differences
        
        Args:
            tree_structure: Dictionary with tree information
            
        Returns:
            Metric tensor as distance matrix
        """
        n_nodes = tree_structure.get('n_nodes', 0)
        if n_nodes == 0:
            return np.array([[0.0]])
            
        # Build adjacency matrix from parent-child relations
        adjacency = np.zeros((n_nodes, n_nodes))
        
        parent_child_pairs = tree_structure.get('edges', [])
        visits = tree_structure.get('visits', np.ones(n_nodes))
        depths = tree_structure.get('depths', np.zeros(n_nodes))
        
        # Weight edges by inverse visit count (frequently visited = shorter)
        for parent, child in parent_child_pairs:
            if parent < n_nodes and child < n_nodes:
                # Information-theoretic weight
                weight = 1.0 / (1.0 + np.sqrt(visits[parent] * visits[child]))
                adjacency[parent, child] = weight
                adjacency[child, parent] = weight
        
        # Add hyperbolic scaling based on depth
        # In AdS, metric ~ 1/z² where z is radial coordinate
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if adjacency[i, j] > 0:
                    # Average depth
                    avg_depth = (depths[i] + depths[j]) / 2.0
                    z = self.ads_radius + avg_depth
                    
                    # AdS metric factor
                    ads_factor = self.ads_radius / z
                    adjacency[i, j] *= ads_factor
                    adjacency[j, i] *= ads_factor
        
        # Compute all-pairs shortest paths (geodesics)
        # This gives the proper distance in tree geometry
        metric = dijkstra(csr_matrix(adjacency), directed=False)
        
        # Handle disconnected components
        metric[np.isinf(metric)] = n_nodes  # Large but finite
        
        return metric
    
    def find_minimal_surface(self, tree_structure: Dict[str, Any],
                           region_A: List[int], region_B: List[int]) -> HolographicSurface:
        """
        Find minimal surface (RT surface) separating regions A and B.
        
        This implements the discrete Ryu-Takayanagi formula:
        S(A) = min(Area(γ_A)) / 4G_N
        
        where γ_A is the minimal surface homologous to A.
        
        Args:
            tree_structure: Tree information
            region_A: Nodes in region A
            region_B: Nodes in region B
            
        Returns:
            Minimal surface with area and entropy bound
        """
        n_nodes = tree_structure.get('n_nodes', 0)
        if n_nodes == 0 or not region_A or not region_B:
            return HolographicSurface([], 0.0, 0, 0.0)
            
        # Convert to sets for efficiency
        set_A = set(region_A)
        set_B = set(region_B)
        
        # Build graph representation
        G = nx.Graph()
        edges = tree_structure.get('edges', [])
        visits = tree_structure.get('visits', np.ones(n_nodes))
        depths = tree_structure.get('depths', np.zeros(n_nodes))
        
        # Add edges with weights
        for parent, child in edges:
            if parent < n_nodes and child < n_nodes:
                # Weight encodes "area" of edge
                weight = self._compute_edge_area(
                    parent, child, visits, depths
                )
                G.add_edge(parent, child, weight=weight)
        
        # Find minimal cut between A and B
        # This is the discrete analogue of finding minimal surface
        
        # First check if regions are disjoint
        if not set_A.isdisjoint(set_B):
            # Overlapping regions - use symmetric difference
            set_A = set_A - set_B
            if not set_A:
                return HolographicSurface([], 0.0, 0, 0.0)
                
        # Build a proper flow network
        # Add super source and super sink
        super_source = 's'
        super_sink = 't'
        
        # Add edges from super source to all nodes in A
        for node in set_A:
            G.add_edge(super_source, node, weight=float('inf'))
            
        # Add edges from all nodes in B to super sink
        for node in set_B:
            G.add_edge(node, super_sink, weight=float('inf'))
            
        try:
            # Find min cut using max flow
            cut_value, partition = nx.minimum_cut(G, super_source, super_sink)
            
            # Remove super nodes from partition
            partition_A = partition[0] - {super_source}
            partition_B = partition[1] - {super_sink}
            
            # Find actual cut edges (excluding super node edges)
            surface_edges = []
            surface_nodes = set()
            
            for u, v, data in G.edges(data=True):
                # Skip super node edges
                if u in [super_source, super_sink] or v in [super_source, super_sink]:
                    continue
                    
                # Check if edge crosses partition
                if (u in partition_A and v in partition_B) or \
                   (u in partition_B and v in partition_A):
                    surface_edges.append((u, v))
                    surface_nodes.add(u)
                    surface_nodes.add(v)
                    
            surface_nodes = list(surface_nodes)
            
            # Compute actual surface area (sum of cut edge weights)
            area = sum(G[u][v]['weight'] for u, v in surface_edges)
            
            # If no cut found, compute direct boundary
            if area == 0:
                # Use boundary nodes between regions
                boundary_found = False
                for node in set_A:
                    if node in G:
                        for neighbor in G.neighbors(node):
                            if neighbor in set_B and neighbor not in [super_source, super_sink]:
                                surface_nodes.extend([node, neighbor])
                                area += G[node][neighbor].get('weight', 1.0)
                                boundary_found = True
                            
                surface_nodes = list(set(surface_nodes))
                
                # If still no boundary, the cut value itself represents the area
                if not boundary_found and cut_value > 0:
                    area = cut_value
            
            # Average depth of surface
            if surface_nodes and all(n < len(depths) for n in surface_nodes):
                avg_depth = np.mean([depths[n] for n in surface_nodes])
            else:
                avg_depth = 0
                
            # Entropy bound from RT formula
            # S = Area / 4G_N, where G_N is "Newton's constant"
            G_newton = 0.25  # Effective gravitational constant
            entropy_bound = area / (4 * G_newton) if area > 0 else 0.0
            
            return HolographicSurface(
                nodes=surface_nodes,
                area=area,
                depth=int(avg_depth),
                entropy_bound=entropy_bound
            )
            
        except (nx.NetworkXError, nx.NetworkXUnbounded):
            # No valid cut exists - regions might be disconnected
            # Try to find boundary manually
            boundary_nodes = []
            boundary_area = 0.0
            
            # Find nodes at interface between regions
            for node in set_A:
                if node in G:
                    for neighbor in G.neighbors(node):
                        if neighbor in set_B:
                            boundary_nodes.extend([node, neighbor])
                            boundary_area += G[node][neighbor].get('weight', 1.0)
                            
            boundary_nodes = list(set(boundary_nodes))
            
            if boundary_nodes and all(n < len(depths) for n in boundary_nodes):
                avg_depth = np.mean([depths[n] for n in boundary_nodes])
            else:
                avg_depth = 0
                
            entropy_bound = boundary_area / 4.0 if boundary_area > 0 else 0.0
            
            return HolographicSurface(
                nodes=boundary_nodes,
                area=boundary_area,
                depth=int(avg_depth),
                entropy_bound=entropy_bound
            )
    
    def _compute_edge_area(self, node1: int, node2: int,
                          visits: np.ndarray, depths: np.ndarray) -> float:
        """
        Compute "area" of edge in emergent geometry.
        
        In AdS/CFT, area elements scale with the metric.
        Here we use visits and depth to define effective area.
        
        Args:
            node1, node2: Edge endpoints  
            visits: Visit counts
            depths: Node depths
            
        Returns:
            Edge area
        """
        # Information content using quantum definitions for consistency
        # Convert visits to quantum state amplitudes
        local_visits = torch.tensor([visits[node1], visits[node2]], 
                                   dtype=torch.float32, device=self.device)
        local_quantum_state = self.quantum_defs.construct_quantum_state_from_single_visits(
            local_visits, outcome_uncertainty=0.01  # Low uncertainty for edge computation
        )
        
        # Use von Neumann entropy as information measure
        info_content = float(self.compute_von_neumann_entropy(local_quantum_state.density_matrix))
        
        # Geometric factor from depth (AdS warping)
        avg_depth = (depths[node1] + depths[node2]) / 2.0
        z = self.ads_radius + avg_depth
        geometric_factor = (self.ads_radius / z) ** 2
        
        # Combined area with quantum correction
        area = (info_content + np.log(2 + visits[node1] + visits[node2])) * geometric_factor
        
        return area
    
    def compute_holographic_screen(self, tree_structure: Dict[str, Any],
                                 screen_depth: int) -> Tuple[List[int], float]:
        """
        Compute holographic screen at fixed depth.
        
        The screen encodes all information about the bulk region
        it bounds, providing a complexity measure.
        
        Args:
            tree_structure: Tree information
            screen_depth: Depth at which to place screen
            
        Returns:
            (screen_nodes, screen_area)
        """
        depths = tree_structure.get('depths', [])
        visits = tree_structure.get('visits', [])
        
        # Find nodes at screen depth
        screen_nodes = [i for i, d in enumerate(depths) if d == screen_depth]
        
        if not screen_nodes:
            # Find closest depth
            unique_depths = sorted(set(depths))
            closest_depth = min(unique_depths, key=lambda d: abs(d - screen_depth))
            screen_nodes = [i for i, d in enumerate(depths) if d == closest_depth]
        
        # Compute screen "area" (information capacity)
        # Area ~ sum of visit-weighted node contributions
        screen_area = 0.0
        for node in screen_nodes:
            # Each node contributes based on its subtree size
            node_contribution = np.log(2 + visits[node])
            
            # Apply geometric factor
            z = self.ads_radius + depths[node]
            geometric_factor = (self.ads_radius / z) ** 2
            
            screen_area += node_contribution * geometric_factor
            
        return screen_nodes, screen_area
    
    def compute_complexity_bound(self, tree_structure: Dict[str, Any],
                               target_nodes: List[int]) -> Dict[str, float]:
        """
        Compute holographic complexity bound for reaching target nodes.
        
        Complexity is bounded by the minimal surface containing targets.
        This provides a lower bound on computational requirements.
        
        Args:
            tree_structure: Tree information
            target_nodes: Nodes we want to reach/evaluate
            
        Returns:
            Dictionary with complexity measures
        """
        n_nodes = tree_structure.get('n_nodes', 0)
        if n_nodes == 0 or not target_nodes:
            return {'complexity': 0.0, 'bound': 0.0}
            
        # Find root (assume node 0)
        root = 0
        
        # Find minimal surface separating root from targets
        # This gives the information that must cross the surface
        surface = self.find_minimal_surface(
            tree_structure, [root], target_nodes
        )
        
        # Complexity bounded by surface area
        complexity_bound = surface.area
        
        # Also compute volume complexity (CV duality)
        # Volume = number of nodes in minimal subtree containing targets
        subtree_nodes = self._find_minimal_subtree(tree_structure, target_nodes)
        volume_complexity = len(subtree_nodes)
        
        return {
            'surface_complexity': complexity_bound,
            'volume_complexity': volume_complexity,
            'entropy_bound': surface.entropy_bound,
            'minimal_surface_size': len(surface.nodes),
            'effective_dimension': np.log(volume_complexity + 1) / np.log(complexity_bound + 2) if complexity_bound > 0 else 0.0
        }
    
    def _find_minimal_subtree(self, tree_structure: Dict[str, Any],
                            target_nodes: List[int]) -> Set[int]:
        """Find minimal subtree containing target nodes"""
        # Build parent pointers
        edges = tree_structure.get('edges', [])
        parents = {}
        
        for parent, child in edges:
            parents[child] = parent
            
        # Find all ancestors of target nodes
        subtree = set(target_nodes)
        
        for node in target_nodes:
            current = node
            while current in parents:
                current = parents[current]
                subtree.add(current)
                
        return subtree
    
    def analyze_information_spreading(self, tree_structure: Dict[str, Any],
                                    source_nodes: List[int]) -> Dict[str, Any]:
        """
        Analyze how information spreads holographically from sources.
        
        Information spreading follows minimal surfaces, giving
        insights into optimal exploration strategies.
        
        Args:
            tree_structure: Tree information
            source_nodes: Initial information sources
            
        Returns:
            Spreading analysis including wavefronts and bounds
        """
        n_nodes = tree_structure.get('n_nodes', 0)
        if n_nodes == 0:
            return {}
            
        depths = tree_structure.get('depths', np.zeros(n_nodes))
        visits = tree_structure.get('visits', np.ones(n_nodes))
        
        # Compute metric
        metric = self.compute_tree_metric(tree_structure)
        
        # Find information wavefronts at different "times"
        # Time = geodesic distance in tree metric
        wavefronts = []
        current_front = set(source_nodes)
        visited = set(source_nodes)
        
        # Propagate information
        max_time = 10  # Maximum propagation steps
        for t in range(max_time):
            next_front = set()
            
            # Find all nodes at distance t+1 from sources
            for node in range(n_nodes):
                if node not in visited:
                    # Check if within distance t+1 from any source
                    min_dist = min(metric[s, node] for s in source_nodes)
                    if min_dist <= (t + 1):
                        next_front.add(node)
                        visited.add(node)
                        
            if not next_front:
                break
                
            # Compute holographic screen for this wavefront
            front_list = list(next_front)
            screen_area = sum(np.log(2 + visits[n]) * 
                            (self.ads_radius / (self.ads_radius + depths[n]))**2
                            for n in front_list)
            
            wavefronts.append({
                'time': t + 1,
                'nodes': front_list,
                'size': len(front_list),
                'area': screen_area,
                'entropy': screen_area / 4.0  # RT formula
            })
            
            current_front = next_front
            
        # Analyze spreading rate
        if len(wavefronts) >= 2:
            # Fit exponential growth
            times = np.array([w['time'] for w in wavefronts])
            sizes = np.array([w['size'] for w in wavefronts])
            areas = np.array([w['area'] for w in wavefronts])
            
            # Lyapunov exponent (information spreading rate)
            if len(times) > 1 and np.any(sizes > 0):
                log_sizes = np.log(sizes + 1)
                lyapunov = np.polyfit(times, log_sizes, 1)[0]
            else:
                lyapunov = 0.0
                
            # Butterfly velocity (speed of information spreading)
            if len(times) > 1:
                butterfly_velocity = np.mean(np.diff(sizes) / np.diff(times))
            else:
                butterfly_velocity = 0.0
        else:
            lyapunov = 0.0
            butterfly_velocity = 0.0
            
        return {
            'wavefronts': wavefronts,
            'lyapunov_exponent': lyapunov,
            'butterfly_velocity': butterfly_velocity,
            'total_nodes_reached': len(visited),
            'saturation_time': len(wavefronts)
        }
    
    def compute_holographic_bounds_summary(self, tree_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute comprehensive holographic bounds for the tree.
        
        Returns:
            Dictionary with all holographic measures
        """
        n_nodes = tree_structure.get('n_nodes', 0)
        if n_nodes == 0:
            return {
                'tree_size': 0,
                'max_depth': 0,
                'screens': [],
                'complexity_bounds': [],
                'information_spreading': {
                    'wavefronts': [],
                    'lyapunov_exponent': 0.0,
                    'butterfly_velocity': 0.0,
                    'total_nodes_reached': 0,
                    'saturation_time': 0
                }
            }
            
        depths = tree_structure.get('depths', np.zeros(n_nodes))
        max_depth = int(max(depths)) if len(depths) > 0 else 0
        
        # Sample different partitions
        results = {
            'tree_size': n_nodes,
            'max_depth': max_depth,
            'screens': [],
            'complexity_bounds': []
        }
        
        # Compute screens at different depths
        for d in range(1, min(max_depth + 1, 10)):
            nodes, area = self.compute_holographic_screen(tree_structure, d)
            results['screens'].append({
                'depth': d,
                'n_nodes': len(nodes),
                'area': area,
                'entropy_capacity': area / 4.0
            })
            
        # Compute complexity for reaching different depth levels
        for target_depth in range(1, min(max_depth + 1, 5)):
            target_nodes = [i for i, d in enumerate(depths) if d >= target_depth]
            if target_nodes:
                complexity = self.compute_complexity_bound(tree_structure, target_nodes[:10])
                complexity['target_depth'] = target_depth
                results['complexity_bounds'].append(complexity)
                
        # Information spreading from root
        spreading = self.analyze_information_spreading(tree_structure, [0])
        results['information_spreading'] = spreading
        
        # Verify area law
        if results['screens']:
            areas = [s['area'] for s in results['screens']]
            entropies = [s['entropy_capacity'] for s in results['screens']]
            sizes = [s['n_nodes'] for s in results['screens']]
            
            # Check if entropy ~ area (not volume)
            if len(sizes) > 2:
                # Linear fit to log(S) vs log(N)
                log_sizes = np.log(np.array(sizes) + 1)
                log_entropies = np.log(np.array(entropies) + 1)
                scaling_exponent = np.polyfit(log_sizes, log_entropies, 1)[0]
                
                # Area law has exponent < 1 (vs volume law = 1)
                results['area_law_exponent'] = scaling_exponent
                results['satisfies_area_law'] = scaling_exponent < 0.9
            else:
                results['area_law_exponent'] = 0.0
                results['satisfies_area_law'] = False
        
        # Add experimental disclaimer
        results['disclaimer'] = 'EXPERIMENTAL: Holographic analysis is phenomenological, not fundamental physics'
        results['interpretation'] = 'Tree structure analysis using AdS/CFT-inspired methods'
                
        return results
    
    def compute_rt_surface(self, tree_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute Ryu-Takayanagi surface for entanglement entropy calculation.
        
        This method automatically partitions the tree and finds the minimal surface.
        
        Args:
            tree_structure: Tree information
            
        Returns:
            Dictionary with surface area and surface nodes
        """
        n_nodes = tree_structure.get('n_nodes', 0)
        if n_nodes == 0:
            return {'area': 0.0, 'surface_nodes': []}
        
        # Auto-partition: split at middle depth
        depths = tree_structure.get('depths', [])
        if len(depths) == 0:
            return {'area': 0.0, 'surface_nodes': []}
        
        max_depth = max(depths)
        mid_depth = max_depth // 2
        
        # Region A: shallow nodes, Region B: deep nodes
        region_A = [i for i, d in enumerate(depths) if d <= mid_depth]
        region_B = [i for i, d in enumerate(depths) if d > mid_depth]
        
        if not region_A or not region_B:
            # Fallback: use first half vs second half
            region_A = list(range(n_nodes // 2))
            region_B = list(range(n_nodes // 2, n_nodes))
        
        # Find minimal surface
        surface = self.find_minimal_surface(tree_structure, region_A, region_B)
        
        return {
            'area': surface.area,
            'surface_nodes': surface.nodes,
            'depth': surface.depth
        }
    
    def compute_entanglement_entropy(self, tree_structure: Dict[str, Any], 
                                   partition_depth: int = 2) -> float:
        """
        Compute entanglement entropy using holographic bounds.
        
        Uses the Ryu-Takayanagi formula: S = Area/4G_N
        
        Args:
            tree_structure: Tree information
            partition_depth: Depth at which to partition the tree
            
        Returns:
            Entanglement entropy
        """
        n_nodes = tree_structure.get('n_nodes', 0)
        if n_nodes == 0:
            return 0.0
        
        # Get RT surface
        rt_surface = self.compute_rt_surface(tree_structure)
        
        # S = Area/4G_N (set G_N = 1 for simplicity)
        entanglement_entropy = rt_surface['area'] / 4.0
        
        return entanglement_entropy
    
    def compute_complexity_bound_simple(self, tree_structure: Dict[str, Any]) -> float:
        """
        Compute simple complexity bound using holographic principles.
        
        The complexity is bounded by the volume of the minimal surface.
        
        Args:
            tree_structure: Tree information
            
        Returns:
            Complexity bound
        """
        n_nodes = tree_structure.get('n_nodes', 0)
        if n_nodes == 0:
            return 0.0
        
        # Simple bound: complexity ~ sqrt(n_nodes) for holographic systems
        return np.sqrt(n_nodes)
    
    def compute_quantum_consistency_check(self, tree_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check consistency between holographic and quantum information approaches.
        
        This validates that our holographic bounds are consistent with
        the unified quantum definitions from quantum_mcts_foundation.md.
        
        Args:
            tree_structure: Tree information
            
        Returns:
            Dictionary with consistency metrics
        """
        n_nodes = tree_structure.get('n_nodes', 0)
        if n_nodes == 0:
            return {'consistent': True, 'message': 'Empty tree'}
        
        # Extract visits
        visits = tree_structure.get('visits', torch.ones(n_nodes))
        if not isinstance(visits, torch.Tensor):
            visits = torch.tensor(visits, dtype=torch.float32, device=self.device)
        else:
            visits = visits.to(self.device)
        
        # Construct quantum state using unified definitions
        quantum_state = self.quantum_defs.construct_quantum_state_from_single_visits(
            visits, outcome_uncertainty=0.2
        )
        
        # Compare holographic and quantum measures
        results = {
            'n_nodes': n_nodes,
            'quantum_entropy': float(self.compute_von_neumann_entropy(quantum_state.density_matrix)),
            'quantum_purity': float(self.compute_purity(quantum_state.density_matrix)),
            'quantum_coherence': float(self.compute_coherence(quantum_state.density_matrix)),
            'has_off_diagonal': float(quantum_state.density_matrix.abs().sum() - 
                                     quantum_state.density_matrix.diag().abs().sum()) > 1e-6
        }
        
        # Get holographic entropy
        holographic_entropy = self.compute_entanglement_entropy(tree_structure)
        results['holographic_entropy'] = holographic_entropy
        
        # Check consistency
        entropy_ratio = results['quantum_entropy'] / (holographic_entropy + 1e-10)
        results['entropy_ratio'] = entropy_ratio
        results['consistent'] = 0.1 < entropy_ratio < 10.0  # Order of magnitude agreement
        
        # Validate quantum state
        validation = self.quantum_defs.validate_quantum_consistency(quantum_state)
        results['quantum_state_valid'] = validation['valid']
        results['validation_details'] = validation
        
        return results


def apply_holographic_bounds(tree_structure: Dict[str, Any],
                           config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Apply holographic bounds analysis to tree.
    
    Args:
        tree_structure: Tree information including edges, visits, depths
        config: Optional configuration
        
    Returns:
        Holographic analysis results
    """
    analyzer = HolographicBoundsAnalyzer()
    
    # Ensure tree structure has required fields
    if 'edges' not in tree_structure:
        # Try to reconstruct from parent pointers if available
        tree_structure['edges'] = []
        if 'parents' in tree_structure:
            parents = tree_structure['parents']
            for child, parent in enumerate(parents):
                if parent >= 0:  # Valid parent
                    tree_structure['edges'].append((parent, child))
    
    # Ensure we have node count
    if 'n_nodes' not in tree_structure:
        if 'visits' in tree_structure:
            tree_structure['n_nodes'] = len(tree_structure['visits'])
        elif 'edges' in tree_structure:
            nodes = set()
            for p, c in tree_structure['edges']:
                nodes.add(p)
                nodes.add(c)
            tree_structure['n_nodes'] = len(nodes)
        else:
            tree_structure['n_nodes'] = 0
            
    return analyzer.compute_holographic_bounds_summary(tree_structure)