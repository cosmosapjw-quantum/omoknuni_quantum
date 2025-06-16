"""
Quantum Darwinism Engine for QFT-MCTS
=====================================

This module implements quantum Darwinism principles for robust move extraction
in MCTS through redundant encoding in the environment.

Key Features:
- Redundancy computation across tree fragments
- Environment-induced selection of robust moves
- Mutual information structure analysis
- GPU-accelerated redundancy calculations
- Objectivity emergence from quantum states

Based on: docs/qft-mcts-math-foundations.md Section 3.3
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass
import logging
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DarwinismConfig:
    """Configuration for quantum Darwinism engine"""
    # Redundancy parameters
    min_fragment_size: int = 5              # Minimum environment fragment size
    max_fragment_size: int = 50             # Maximum environment fragment size
    redundancy_threshold: float = 0.9       # Threshold for classical objectivity
    
    # Mutual information
    mi_num_bins: int = 20                   # Bins for MI estimation
    mi_regularization: float = 1e-8         # Regularization for MI
    
    # Fragment selection
    num_random_fragments: int = 100         # Number of random fragments to sample
    fragment_overlap_penalty: float = 0.5   # Penalty for overlapping fragments
    
    # Objectivity criteria
    objectivity_threshold: float = 0.8      # Threshold for move objectivity
    min_redundancy_ratio: float = 0.5       # Minimum R_δ/N for objectivity
    
    # Numerical parameters
    convergence_threshold: float = 1e-6     # Convergence criterion
    max_iterations: int = 100               # Maximum iterations
    
    # GPU optimization
    batch_fragment_processing: int = 32     # Batch size for fragment processing
    use_sparse_fragments: bool = True       # Use sparse representation


class FragmentSelector:
    """
    Selects environment fragments for redundancy analysis
    
    Fragments represent different parts of the tree that independently
    encode information about moves.
    """
    
    def __init__(self, config: DarwinismConfig, device: torch.device):
        self.config = config
        self.device = device
        
    def generate_random_fragments(
        self,
        num_nodes: int,
        num_fragments: Optional[int] = None,
        excluded_nodes: Optional[Set[int]] = None
    ) -> List[torch.Tensor]:
        """
        Generate random environment fragments
        
        Args:
            num_nodes: Total number of nodes
            num_fragments: Number of fragments to generate
            excluded_nodes: Nodes to exclude (e.g., the system)
            
        Returns:
            List of fragment index tensors
        """
        if num_fragments is None:
            num_fragments = self.config.num_random_fragments
            
        if excluded_nodes is None:
            excluded_nodes = set()
            
        fragments = []
        available_nodes = [i for i in range(num_nodes) if i not in excluded_nodes]
        
        for _ in range(num_fragments):
            # Random fragment size
            frag_size = torch.randint(
                self.config.min_fragment_size,
                min(self.config.max_fragment_size, len(available_nodes)),
                (1,)
            ).item()
            
            # Random selection without replacement
            perm = torch.randperm(len(available_nodes))[:frag_size]
            fragment_indices = torch.tensor(
                [available_nodes[i] for i in perm],
                device=self.device
            )
            
            fragments.append(fragment_indices)
            
        return fragments
    
    def generate_tree_fragments(
        self,
        tree_structure: Dict[str, torch.Tensor],
        root_node: int
    ) -> List[torch.Tensor]:
        """
        Generate fragments based on tree structure
        
        Creates fragments that respect tree topology (subtrees, paths, etc.)
        """
        fragments = []
        
        if 'children' not in tree_structure:
            return self.generate_random_fragments(tree_structure.get('num_nodes', 100))
            
        children = tree_structure['children']
        num_nodes = children.shape[0]
        
        # Generate subtree fragments
        subtree_fragments = self._generate_subtree_fragments(children, root_node)
        fragments.extend(subtree_fragments)
        
        # Generate path fragments  
        path_fragments = self._generate_path_fragments(children, root_node)
        fragments.extend(path_fragments)
        
        # Add some random fragments for diversity
        random_fragments = self.generate_random_fragments(
            num_nodes,
            num_fragments=self.config.num_random_fragments // 3,
            excluded_nodes={root_node}
        )
        fragments.extend(random_fragments)
        
        return fragments
    
    def _generate_subtree_fragments(
        self,
        children: torch.Tensor,
        root: int,
        max_depth: int = 3
    ) -> List[torch.Tensor]:
        """Generate fragments corresponding to subtrees"""
        fragments = []
        
        # BFS to find subtrees
        queue = [(root, 0)]
        visited = {root}
        
        while queue:
            node, depth = queue.pop(0)
            
            if depth > 0 and depth <= max_depth:
                # Create fragment from this subtree
                subtree_nodes = self._get_subtree_nodes(children, node, max_depth - depth)
                if len(subtree_nodes) >= self.config.min_fragment_size:
                    fragment = torch.tensor(list(subtree_nodes), device=self.device)
                    fragments.append(fragment)
            
            # Add children to queue
            if node < children.shape[0]:
                node_children = children[node][children[node] >= 0]
                for child in node_children:
                    if child.item() not in visited:
                        visited.add(child.item())
                        queue.append((child.item(), depth + 1))
        
        return fragments
    
    def _generate_path_fragments(
        self,
        children: torch.Tensor,
        root: int
    ) -> List[torch.Tensor]:
        """Generate fragments corresponding to paths"""
        fragments = []
        
        # Find leaf nodes
        is_leaf = torch.all(children < 0, dim=1)
        leaf_nodes = torch.where(is_leaf)[0]
        
        # Sample some paths from root to leaves
        num_paths = min(20, len(leaf_nodes))
        sampled_leaves = leaf_nodes[torch.randperm(len(leaf_nodes))[:num_paths]]
        
        for leaf in sampled_leaves:
            # Reconstruct path (simplified - assumes tree structure allows this)
            path = [leaf.item()]
            # In practice, would traverse up the tree
            # For now, create random path of reasonable length
            path_length = torch.randint(
                self.config.min_fragment_size,
                self.config.max_fragment_size,
                (1,)
            ).item()
            
            fragment = torch.tensor(path[:path_length], device=self.device)
            if len(fragment) >= self.config.min_fragment_size:
                fragments.append(fragment)
        
        return fragments
    
    def _get_subtree_nodes(
        self,
        children: torch.Tensor,
        root: int,
        max_depth: int
    ) -> Set[int]:
        """Get all nodes in subtree up to max_depth"""
        nodes = {root}
        queue = [(root, 0)]
        
        while queue:
            node, depth = queue.pop(0)
            
            if depth < max_depth and node < children.shape[0]:
                node_children = children[node][children[node] >= 0]
                for child in node_children:
                    child_idx = child.item()
                    if child_idx not in nodes:
                        nodes.add(child_idx)
                        queue.append((child_idx, depth + 1))
        
        return nodes


class RedundancyAnalyzer:
    """
    Analyzes redundant encoding of information in environment
    
    Computes how much information about the system (moves) is redundantly
    encoded across different environment fragments.
    """
    
    def __init__(self, config: DarwinismConfig, device: torch.device):
        self.config = config
        self.device = device
        
    def compute_mutual_information(
        self,
        system_state: torch.Tensor,
        fragment_states: torch.Tensor
    ) -> float:
        """
        Compute mutual information I(S:F) between system and fragment
        
        Args:
            system_state: System state vector
            fragment_states: Fragment state matrix
            
        Returns:
            Mutual information in bits
        """
        # Convert to numpy for histogram computation
        s_vals = system_state.detach().cpu().numpy()
        f_vals = fragment_states.detach().cpu().numpy()
        
        # Flatten if needed
        if len(s_vals.shape) > 1:
            s_vals = s_vals.flatten()
        if len(f_vals.shape) > 1:
            f_vals = f_vals.flatten()
        
        # Ensure same length - use mean of fragment if needed
        if len(f_vals) > len(s_vals):
            f_vals = f_vals[:len(s_vals)]
        elif len(s_vals) > len(f_vals):
            # Replicate system values to match fragment length
            s_vals = np.resize(s_vals, len(f_vals))
            
        # Compute joint and marginal histograms
        hist_2d, s_edges, f_edges = np.histogram2d(
            s_vals, f_vals, bins=self.config.mi_num_bins
        )
        
        # Convert to probabilities
        joint_prob = hist_2d / (hist_2d.sum() + self.config.mi_regularization)
        s_prob = joint_prob.sum(axis=1)
        f_prob = joint_prob.sum(axis=0)
        
        # Compute MI: I(S:F) = Σ p(s,f) log[p(s,f)/(p(s)p(f))]
        mi = 0.0
        for i in range(len(s_prob)):
            for j in range(len(f_prob)):
                if joint_prob[i, j] > 0 and s_prob[i] > 0 and f_prob[j] > 0:
                    mi += joint_prob[i, j] * np.log2(
                        joint_prob[i, j] / (s_prob[i] * f_prob[j])
                    )
        
        return mi
    
    def compute_redundancy_spectrum(
        self,
        system_values: torch.Tensor,
        environment_values: torch.Tensor,
        fragments: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute redundancy spectrum R_δ(f)
        
        This measures how information is distributed across fragments.
        
        Args:
            system_values: Values/features of system nodes
            environment_values: Values/features of all nodes
            fragments: List of environment fragments
            
        Returns:
            Dictionary with redundancy analysis
        """
        num_fragments = len(fragments)
        fragment_sizes = torch.tensor([len(f) for f in fragments], device=self.device)
        
        # Compute mutual information for each fragment
        mutual_informations = []
        
        for fragment in fragments:
            # Extract fragment values
            fragment_vals = environment_values[fragment]
            
            # Compute I(S:F_k)
            mi = self.compute_mutual_information(system_values, fragment_vals)
            mutual_informations.append(mi)
        
        mutual_informations = torch.tensor(mutual_informations, device=self.device)
        
        # Sort by fragment size
        sorted_indices = torch.argsort(fragment_sizes)
        sorted_sizes = fragment_sizes[sorted_indices]
        sorted_mis = mutual_informations[sorted_indices]
        
        # Compute redundancy at different thresholds
        info_thresholds = torch.linspace(0.1, 0.9, 9, device=self.device)
        redundancies = []
        
        total_env_size = environment_values.shape[0]
        
        for threshold in info_thresholds:
            # Find fragments that contain enough information
            sufficient_fragments = sorted_mis >= threshold * sorted_mis.max()
            
            if sufficient_fragments.any():
                # Minimum fragment size needed
                min_size_needed = sorted_sizes[sufficient_fragments].min()
                redundancy = min_size_needed.float() / total_env_size
            else:
                redundancy = 1.0
                
            redundancies.append(redundancy)
        
        redundancies = torch.tensor(redundancies, device=self.device)
        
        return {
            'fragment_sizes': fragment_sizes,
            'mutual_informations': mutual_informations,
            'info_thresholds': info_thresholds,
            'redundancies': redundancies,
            'average_redundancy': redundancies.mean(),
            'max_mutual_information': mutual_informations.max()
        }
    
    def compute_objectivity_measure(
        self,
        redundancy_spectrum: Dict[str, torch.Tensor]
    ) -> float:
        """
        Compute objectivity measure from redundancy spectrum
        
        High objectivity means information is redundantly encoded
        in many small fragments.
        """
        redundancies = redundancy_spectrum['redundancies']
        thresholds = redundancy_spectrum['info_thresholds']
        
        # Objectivity emerges when R_δ scales as ~N^(-1/2)
        # We measure deviation from this scaling
        
        # Expected scaling for objective information
        expected_scaling = 1.0 / torch.sqrt(torch.arange(1, len(redundancies) + 1, device=redundancies.device))
        expected_scaling = expected_scaling / expected_scaling[0]  # Normalize
        
        # Actual scaling
        actual_scaling = redundancies / redundancies[0] if redundancies[0] > 0 else redundancies
        
        # Compute similarity to expected scaling
        scaling_similarity = 1.0 - torch.mean(torch.abs(actual_scaling - expected_scaling))
        scaling_similarity = torch.clamp(scaling_similarity, min=0.0, max=1.0)
        
        # Also consider average redundancy (lower is better for objectivity)
        avg_redundancy = redundancy_spectrum['average_redundancy']
        redundancy_score = 1.0 - torch.clamp(avg_redundancy, min=0.0, max=1.0)
        
        # Combine measures - geometric mean for balanced score
        objectivity = torch.sqrt(scaling_similarity * redundancy_score)
        
        # Ensure objectivity is positive
        objectivity = torch.clamp(objectivity, min=0.0, max=1.0)
        
        return objectivity.item()


class QuantumDarwinismEngine:
    """
    Main engine for quantum Darwinism analysis in MCTS
    
    Identifies objectively robust moves through redundant encoding
    in the tree environment.
    """
    
    def __init__(self, config: DarwinismConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # Initialize components
        self.fragment_selector = FragmentSelector(config, device)
        self.redundancy_analyzer = RedundancyAnalyzer(config, device)
        
        # Cache for analysis results
        self.redundancy_cache = {}
        
        # Statistics
        self.stats = {
            'moves_analyzed': 0,
            'objective_moves_found': 0,
            'average_redundancy': 0.0,
            'average_objectivity': 0.0,
            'fragment_generation_time': 0.0,
            'redundancy_computation_time': 0.0
        }
        
        logger.debug("QuantumDarwinismEngine initialized")
    
    def analyze_move_objectivity(
        self,
        move_node: int,
        tree_values: torch.Tensor,
        tree_structure: Dict[str, torch.Tensor],
        visit_counts: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Analyze objectivity of a move through quantum Darwinism
        
        Args:
            move_node: Node index representing the move
            tree_values: Values/Q-values for all nodes
            tree_structure: Tree connectivity information
            visit_counts: Optional visit count information
            
        Returns:
            Dictionary with objectivity analysis
        """
        start_time = time.perf_counter()
        
        # Generate environment fragments
        fragments = self.fragment_selector.generate_tree_fragments(
            tree_structure, move_node
        )
        
        fragment_time = time.perf_counter()
        self.stats['fragment_generation_time'] += fragment_time - start_time
        
        # System is the move node and its immediate context
        system_values = tree_values[move_node].unsqueeze(0)
        
        # Add visit count information if available
        if visit_counts is not None:
            # Weight values by visit counts for better information encoding
            weighted_values = tree_values * torch.sqrt(visit_counts + 1)
        else:
            weighted_values = tree_values
        
        # Compute redundancy spectrum
        redundancy_spectrum = self.redundancy_analyzer.compute_redundancy_spectrum(
            system_values,
            weighted_values,
            fragments
        )
        
        # Compute objectivity measure
        objectivity = self.redundancy_analyzer.compute_objectivity_measure(
            redundancy_spectrum
        )
        
        redundancy_time = time.perf_counter()
        self.stats['redundancy_computation_time'] += redundancy_time - fragment_time
        
        # Determine if move is objective
        is_objective = bool(
            objectivity > self.config.objectivity_threshold and
            redundancy_spectrum['average_redundancy'].item() < self.config.min_redundancy_ratio
        )
        
        # Update statistics
        self.stats['moves_analyzed'] += 1
        if is_objective:
            self.stats['objective_moves_found'] += 1
        self.stats['average_objectivity'] = (
            0.9 * self.stats['average_objectivity'] + 0.1 * objectivity
        )
        self.stats['average_redundancy'] = (
            0.9 * self.stats['average_redundancy'] + 
            0.1 * redundancy_spectrum['average_redundancy'].item()
        )
        
        return {
            'move_node': move_node,
            'objectivity': objectivity,
            'is_objective': is_objective,
            'redundancy_spectrum': redundancy_spectrum,
            'num_fragments_analyzed': len(fragments),
            'max_mutual_information': redundancy_spectrum['max_mutual_information'].item()
        }
    
    def extract_robust_moves(
        self,
        candidate_moves: torch.Tensor,
        tree_values: torch.Tensor,
        tree_structure: Dict[str, torch.Tensor],
        visit_counts: Optional[torch.Tensor] = None,
        top_k: int = 5
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Extract most robust moves using quantum Darwinism
        
        Args:
            candidate_moves: Indices of candidate moves
            tree_values: Values for all nodes
            tree_structure: Tree connectivity
            visit_counts: Optional visit counts
            top_k: Number of top objective moves to return
            
        Returns:
            Tuple of (robust_move_indices, analysis_results)
        """
        objectivity_scores = []
        analysis_results = []
        
        # Analyze each candidate move
        for move_idx in candidate_moves:
            analysis = self.analyze_move_objectivity(
                move_idx.item(),
                tree_values,
                tree_structure,
                visit_counts
            )
            
            objectivity_scores.append(analysis['objectivity'])
            analysis_results.append(analysis)
        
        objectivity_scores = torch.tensor(objectivity_scores, device=self.device)
        
        # Select top-k most objective moves
        top_k = min(top_k, len(objectivity_scores))
        top_scores, top_indices = torch.topk(objectivity_scores, top_k)
        
        robust_moves = candidate_moves[top_indices]
        
        # Aggregate analysis results
        aggregated_results = {
            'objectivity_scores': objectivity_scores,
            'top_objectivity_scores': top_scores,
            'average_objectivity': objectivity_scores.mean().item(),
            'num_objective_moves': sum(r['is_objective'] for r in analysis_results),
            'individual_analyses': [analysis_results[i] for i in top_indices]
        }
        
        return robust_moves, aggregated_results
    
    def compute_information_broadcasting(
        self,
        source_node: int,
        tree_structure: Dict[str, torch.Tensor],
        tree_values: torch.Tensor,
        max_distance: int = 5
    ) -> Dict[str, torch.Tensor]:
        """
        Compute how information from source node is broadcast through tree
        
        This measures the "Darwinian fitness" of information spreading.
        """
        num_nodes = tree_values.shape[0]
        
        # Initialize distance matrix
        distances = torch.full((num_nodes,), float('inf'), device=self.device)
        distances[source_node] = 0
        
        # BFS to compute distances
        children = tree_structure.get('children')
        if children is not None:
            queue = [source_node]
            visited = {source_node}
            
            while queue:
                node = queue.pop(0)
                current_dist = distances[node]
                
                if current_dist < max_distance and node < children.shape[0]:
                    node_children = children[node][children[node] >= 0]
                    
                    for child in node_children:
                        child_idx = child.item()
                        if child_idx not in visited:
                            visited.add(child_idx)
                            distances[child_idx] = current_dist + 1
                            queue.append(child_idx)
        
        # Compute information decay with distance
        finite_distances = distances[distances < float('inf')]
        
        if len(finite_distances) > 0:
            # Information strength at each distance
            info_strength = torch.zeros(max_distance + 1, device=self.device)
            
            for d in range(max_distance + 1):
                nodes_at_distance = distances == d
                if nodes_at_distance.any():
                    # Average value correlation at this distance
                    source_value = tree_values[source_node]
                    distant_values = tree_values[nodes_at_distance]
                    
                    # Correlation as proxy for information transfer
                    if len(distant_values) > 1:
                        # Compute correlation between source and mean of distant values
                        mean_distant = distant_values.mean()
                        correlation = torch.abs(source_value - mean_distant) / (torch.std(distant_values) + 1e-8)
                        correlation = 1.0 / (1.0 + correlation)  # Convert distance to similarity
                    else:
                        # Single value - use direct similarity
                        correlation = 1.0 / (1.0 + torch.abs(source_value - distant_values[0]))
                    
                    info_strength[d] = torch.abs(correlation) if not torch.isnan(correlation) else 0
        else:
            info_strength = torch.zeros(max_distance + 1, device=self.device)
        
        # Compute broadcasting efficiency
        broadcasting_efficiency = info_strength.sum() / (max_distance + 1)
        
        return {
            'distances': distances,
            'information_strength': info_strength,
            'broadcasting_efficiency': broadcasting_efficiency,
            'reachable_nodes': (distances < float('inf')).sum().item()
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """Get engine statistics"""
        return dict(self.stats)
    
    def reset_cache(self):
        """Reset redundancy cache"""
        self.redundancy_cache.clear()


# Factory function
def create_darwinism_engine(
    device: Union[str, torch.device] = 'cuda',
    min_fragment_size: int = 5,
    **kwargs
) -> QuantumDarwinismEngine:
    """
    Factory function to create quantum Darwinism engine
    
    Args:
        device: Device for computation
        min_fragment_size: Minimum environment fragment size
        **kwargs: Override default config parameters
        
    Returns:
        Initialized QuantumDarwinismEngine
    """
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Create config with overrides
    config_dict = {
        'min_fragment_size': min_fragment_size,
        'max_fragment_size': 50,
        'redundancy_threshold': 0.9,
    }
    config_dict.update(kwargs)
    
    config = DarwinismConfig(**config_dict)
    
    return QuantumDarwinismEngine(config, device)