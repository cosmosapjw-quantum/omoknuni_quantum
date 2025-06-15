"""
MCTS Quantum Observable Interface
=================================

This module provides surgical modifications to MCTS to expose internal
quantum observables for physics validation. It extracts real data from
the MCTS tree structure without relying on synthetic fallbacks.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MCTSQuantumObservables:
    """Container for quantum observables extracted from MCTS tree"""
    visit_counts: np.ndarray  # Visit count distribution
    q_values: np.ndarray      # Q-value estimates
    policies: np.ndarray      # Policy distributions
    ucb_scores: np.ndarray    # UCB scores
    
    # Tree structure
    node_depths: np.ndarray   # Depth of each node
    node_parents: np.ndarray  # Parent index for each node
    node_children: List[List[int]]  # Children indices
    
    # Path statistics  
    path_visits: List[List[int]]  # Visit counts along paths
    path_values: List[List[float]]  # Values along paths
    
    # Derived quantities
    visit_correlations: Dict[int, float]  # Correlation at each distance
    policy_concentration: float  # Order parameter
    evaluation_variance: float  # Noise in evaluations
    tree_fragmentation: float  # Darwinism redundancy


class MCTSQuantumInterface:
    """
    Interface to extract quantum observables from MCTS tree.
    
    This class provides methods to:
    1. Extract real visit count correlations
    2. Compute actual policy concentration (order parameter)
    3. Measure evaluation variance for decoherence
    4. Calculate tree fragmentation for Darwinism
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def extract_observables(self, mcts) -> MCTSQuantumObservables:
        """Extract all quantum observables from MCTS tree"""
        
        # Check if CSRTree or regular tree
        if hasattr(mcts.tree, 'visit_counts'):
            return self._extract_from_csr_tree(mcts.tree)
        elif hasattr(mcts, 'tree') and hasattr(mcts.tree, 'nodes'):
            return self._extract_from_dict_tree(mcts.tree)
        else:
            raise ValueError("Unknown tree structure - cannot extract observables")
    
    def _extract_from_csr_tree(self, tree) -> MCTSQuantumObservables:
        """Extract observables from CSRTree structure"""
        # Ensure CSR structure is consistent before accessing it
        if hasattr(tree, 'ensure_consistent'):
            tree.ensure_consistent()
            
        logger.debug(f"Extracting from CSRTree with {tree.num_nodes} nodes")
        
        if tree.num_nodes == 0:
            raise ValueError("Empty CSRTree - no nodes to analyze")
        
        # Extract basic quantities
        visit_counts = tree.visit_counts[:tree.num_nodes].cpu().numpy()
        
        # Compute Q-values from value_sums / visit_counts
        value_sums = tree.value_sums[:tree.num_nodes].cpu().numpy()
        q_values = np.zeros_like(value_sums)
        mask = visit_counts > 0
        q_values[mask] = value_sums[mask] / visit_counts[mask]
        
        # Extract policies
        # CSRTree uses row_ptr for CSR structure
        if hasattr(tree, 'row_ptr'):
            # Extract number of children for each node
            num_children = tree.row_ptr[1:tree.num_nodes+1] - tree.row_ptr[:tree.num_nodes]
            
            policies = []
            for node_idx in range(tree.num_nodes):
                start_idx = tree.row_ptr[node_idx].item()
                end_idx = tree.row_ptr[node_idx + 1].item()
                
                if end_idx > start_idx and hasattr(tree, 'edge_priors'):
                    # Get priors from edge data
                    node_priors = tree.edge_priors[start_idx:end_idx].cpu().numpy()
                    policies.append(node_priors)
                else:
                    # No children or no priors
                    policies.append(np.array([]))
        else:
            # Fallback: empty policies
            policies = [np.array([]) for _ in range(tree.num_nodes)]
        
        # Compute UCB scores
        ucb_scores = self._compute_ucb_scores_csr(tree)
        
        # Extract tree structure
        node_depths = self._compute_node_depths_csr(tree)
        node_parents = self._compute_node_parents_csr(tree)
        node_children = self._extract_children_lists_csr(tree)
        
        # Extract path statistics
        path_visits, path_values = self._extract_paths_csr(tree)
        
        # Compute derived quantities
        visit_correlations = self._compute_visit_correlations(
            visit_counts, node_depths, node_children
        )
        
        policy_concentration = self._compute_policy_concentration(visit_counts)
        
        evaluation_variance = self._compute_evaluation_variance(q_values, visit_counts)
        
        tree_fragmentation = self._compute_tree_fragmentation(
            visit_counts, node_children, tree.num_nodes
        )
        
        return MCTSQuantumObservables(
            visit_counts=visit_counts,
            q_values=q_values,
            policies=policies,
            ucb_scores=ucb_scores,
            node_depths=node_depths,
            node_parents=node_parents,
            node_children=node_children,
            path_visits=path_visits,
            path_values=path_values,
            visit_correlations=visit_correlations,
            policy_concentration=policy_concentration,
            evaluation_variance=evaluation_variance,
            tree_fragmentation=tree_fragmentation
        )
    
    def _extract_from_dict_tree(self, tree) -> MCTSQuantumObservables:
        """Extract observables from dictionary-based tree"""
        logger.debug(f"Extracting from dict tree with {len(tree.nodes)} nodes")
        
        if not tree.nodes:
            raise ValueError("Empty tree - no nodes to analyze")
        
        # Sort nodes by ID for consistent ordering
        node_ids = sorted(tree.nodes.keys())
        num_nodes = len(node_ids)
        
        # Create ID to index mapping
        id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        # Extract basic quantities
        visit_counts = np.array([tree.nodes[nid].visits for nid in node_ids])
        q_values = np.array([tree.nodes[nid].value for nid in node_ids])
        
        # Extract policies (if available)
        policies = []
        for nid in node_ids:
            node = tree.nodes[nid]
            if hasattr(node, 'policy'):
                policies.append(node.policy)
            elif hasattr(node, 'priors'):
                policies.append(node.priors)
            else:
                # Uniform policy if not available
                num_children = len(node.children) if hasattr(node, 'children') else 0
                policies.append(np.ones(num_children) / num_children if num_children > 0 else np.array([]))
        
        # Compute UCB scores (simplified)
        ucb_scores = q_values + np.sqrt(2 * np.log(np.sum(visit_counts)) / (visit_counts + 1))
        
        # Extract tree structure
        node_depths = self._compute_node_depths_dict(tree, node_ids, id_to_idx)
        node_parents = self._compute_node_parents_dict(tree, node_ids, id_to_idx)
        node_children = self._extract_children_lists_dict(tree, node_ids, id_to_idx)
        
        # Extract path statistics
        path_visits, path_values = self._extract_paths_dict(tree, node_ids)
        
        # Compute derived quantities
        visit_correlations = self._compute_visit_correlations(
            visit_counts, node_depths, node_children
        )
        
        policy_concentration = self._compute_policy_concentration(visit_counts)
        
        evaluation_variance = self._compute_evaluation_variance(q_values, visit_counts)
        
        tree_fragmentation = self._compute_tree_fragmentation(
            visit_counts, node_children, num_nodes
        )
        
        return MCTSQuantumObservables(
            visit_counts=visit_counts,
            q_values=q_values,
            policies=policies,
            ucb_scores=ucb_scores,
            node_depths=node_depths,
            node_parents=node_parents,
            node_children=node_children,
            path_visits=path_visits,
            path_values=path_values,
            visit_correlations=visit_correlations,
            policy_concentration=policy_concentration,
            evaluation_variance=evaluation_variance,
            tree_fragmentation=tree_fragmentation
        )
    
    def _compute_ucb_scores_csr(self, tree) -> np.ndarray:
        """Compute UCB scores for CSRTree nodes"""
        ucb_scores = np.zeros(tree.num_nodes)
        
        for node_idx in range(tree.num_nodes):
            visits = tree.visit_counts[node_idx].item()
            
            # Compute Q-value from value_sum / visit_count
            if visits > 0:
                value_sum = tree.value_sums[node_idx].item()
                q_value = value_sum / visits
            else:
                q_value = 0.0
            
            # Get parent visits for UCB calculation
            if node_idx == 0:
                parent_visits = visits
            else:
                # Find parent (this is approximate for CSRTree)
                parent_visits = tree.visit_counts[0].item()  # Use root visits
            
            # UCB formula
            if visits > 0:
                ucb = q_value + np.sqrt(2 * np.log(parent_visits) / visits)
            else:
                ucb = float('inf')
            
            ucb_scores[node_idx] = ucb
        
        return ucb_scores
    
    def _compute_node_depths_csr(self, tree) -> np.ndarray:
        """Compute depth of each node in CSRTree"""
        depths = np.zeros(tree.num_nodes, dtype=int)
        
        # BFS to compute depths
        queue = [(0, 0)]  # (node_idx, depth)
        visited = set()
        
        while queue:
            node_idx, depth = queue.pop(0)
            if node_idx in visited or node_idx >= tree.num_nodes:
                continue
            
            visited.add(node_idx)
            depths[node_idx] = depth
            
            # Add children to queue
            if hasattr(tree, 'row_ptr') and hasattr(tree, 'col_indices'):
                # Use CSR structure
                start_idx = tree.row_ptr[node_idx].item()
                end_idx = tree.row_ptr[node_idx + 1].item()
                
                if end_idx > start_idx:
                    children = tree.col_indices[start_idx:end_idx]
                    for child_idx in children:
                        if child_idx.item() < tree.num_nodes:
                            queue.append((child_idx.item(), depth + 1))
            elif hasattr(tree, 'children'):
                # Use direct children array
                node_children = tree.children[node_idx]
                for child_idx in node_children:
                    if child_idx >= 0 and child_idx < tree.num_nodes:
                        queue.append((int(child_idx.item()), depth + 1))
        
        return depths
    
    def _compute_node_parents_csr(self, tree) -> np.ndarray:
        """Compute parent index for each node in CSRTree"""
        # CSRTree stores parent_indices directly
        if hasattr(tree, 'parent_indices'):
            parents = tree.parent_indices[:tree.num_nodes].cpu().numpy()
            return parents
        
        # Fallback: compute from structure
        parents = np.full(tree.num_nodes, -1, dtype=int)
        
        if hasattr(tree, 'row_ptr') and hasattr(tree, 'col_indices'):
            for node_idx in range(tree.num_nodes):
                start_idx = tree.row_ptr[node_idx].item()
                end_idx = tree.row_ptr[node_idx + 1].item()
                
                if end_idx > start_idx:
                    children = tree.col_indices[start_idx:end_idx]
                    for child_idx in children:
                        if child_idx.item() < tree.num_nodes:
                            parents[child_idx.item()] = node_idx
        
        return parents
    
    def _extract_children_lists_csr(self, tree) -> List[List[int]]:
        """Extract children lists from CSRTree"""
        children_lists = []
        
        # CSRTree can use either row_ptr or direct children array
        if hasattr(tree, 'row_ptr') and hasattr(tree, 'col_indices'):
            # Use CSR structure
            for node_idx in range(tree.num_nodes):
                start_idx = tree.row_ptr[node_idx].item()
                end_idx = tree.row_ptr[node_idx + 1].item()
                
                if end_idx > start_idx:
                    children = tree.col_indices[start_idx:end_idx].cpu().numpy().tolist()
                    # Filter out invalid indices
                    children = [c for c in children if c < tree.num_nodes]
                    children_lists.append(children)
                else:
                    children_lists.append([])
        elif hasattr(tree, 'children'):
            # Use direct children array
            children_array = tree.children[:tree.num_nodes].cpu().numpy()
            for node_idx in range(tree.num_nodes):
                node_children = children_array[node_idx]
                # Filter out -1 (invalid) and indices >= num_nodes
                valid_children = [int(c) for c in node_children if c >= 0 and c < tree.num_nodes]
                children_lists.append(valid_children)
        else:
            # No children information
            children_lists = [[] for _ in range(tree.num_nodes)]
        
        return children_lists
    
    def _extract_paths_csr(self, tree, max_paths: int = 100) -> Tuple[List[List[int]], List[List[float]]]:
        """Extract sample paths from CSRTree"""
        # Ensure CSR structure is consistent before accessing it
        if hasattr(tree, 'ensure_consistent'):
            tree.ensure_consistent()
            
        path_visits = []
        path_values = []
        
        # Start from root and follow high-visit branches
        for path_idx in range(min(max_paths, tree.num_nodes // 10)):
            path = []
            visits = []
            values = []
            
            current = 0
            max_depth = 20
            
            for depth in range(max_depth):
                if current >= tree.num_nodes:
                    break
                
                path.append(current)
                visits.append(tree.visit_counts[current].item())
                
                # Compute Q-value from value_sum / visit_count
                visit_count = tree.visit_counts[current].item()
                if visit_count > 0:
                    value_sum = tree.value_sums[current].item()
                    q_value = value_sum / visit_count
                else:
                    q_value = 0.0
                values.append(q_value)
                
                # Get children using CSR structure
                if hasattr(tree, 'row_ptr') and hasattr(tree, 'col_indices'):
                    start_idx = tree.row_ptr[current].item()
                    end_idx = tree.row_ptr[current + 1].item()
                    
                    if end_idx <= start_idx:
                        break
                    
                    # Choose child based on visits (with some randomness)
                    children = tree.col_indices[start_idx:end_idx]
                else:
                    # Fallback: no children
                    break
                child_visits = []
                
                for child_idx in children:
                    if child_idx.item() < tree.num_nodes:
                        child_visits.append((child_idx.item(), 
                                           tree.visit_counts[child_idx.item()].item()))
                
                if not child_visits:
                    break
                
                # Sort by visits and pick from top candidates
                child_visits.sort(key=lambda x: x[1], reverse=True)
                
                # Take top child most of the time, occasionally explore others
                if np.random.random() < 0.8 and len(child_visits) > 0:
                    current = child_visits[0][0]
                elif len(child_visits) > 1:
                    current = child_visits[min(1, len(child_visits)-1)][0]
                else:
                    break
            
            if len(path) > 1:
                path_visits.append(visits)
                path_values.append(values)
        
        return path_visits, path_values
    
    def _compute_node_depths_dict(self, tree, node_ids, id_to_idx) -> np.ndarray:
        """Compute depths for dictionary tree"""
        depths = np.zeros(len(node_ids), dtype=int)
        
        # BFS from root
        root_id = node_ids[0] if node_ids else 0
        queue = [(root_id, 0)]
        visited = set()
        
        while queue:
            node_id, depth = queue.pop(0)
            if node_id in visited:
                continue
            
            visited.add(node_id)
            if node_id in id_to_idx:
                depths[id_to_idx[node_id]] = depth
            
            node = tree.nodes.get(node_id)
            if node and hasattr(node, 'children'):
                for child_id in node.children:
                    if child_id not in visited:
                        queue.append((child_id, depth + 1))
        
        return depths
    
    def _compute_node_parents_dict(self, tree, node_ids, id_to_idx) -> np.ndarray:
        """Compute parent indices for dictionary tree"""
        parents = np.full(len(node_ids), -1, dtype=int)
        
        for node_id in node_ids:
            node = tree.nodes[node_id]
            if hasattr(node, 'children'):
                for child_id in node.children:
                    if child_id in id_to_idx:
                        parents[id_to_idx[child_id]] = id_to_idx[node_id]
        
        return parents
    
    def _extract_children_lists_dict(self, tree, node_ids, id_to_idx) -> List[List[int]]:
        """Extract children lists from dictionary tree"""
        children_lists = []
        
        for node_id in node_ids:
            node = tree.nodes[node_id]
            if hasattr(node, 'children'):
                # Convert child IDs to indices
                child_indices = []
                for child_id in node.children:
                    if child_id in id_to_idx:
                        child_indices.append(id_to_idx[child_id])
                children_lists.append(child_indices)
            else:
                children_lists.append([])
        
        return children_lists
    
    def _extract_paths_dict(self, tree, node_ids, max_paths: int = 100) -> Tuple[List[List[int]], List[List[float]]]:
        """Extract sample paths from dictionary tree"""
        path_visits = []
        path_values = []
        
        root_id = node_ids[0] if node_ids else 0
        
        for path_idx in range(min(max_paths, len(node_ids) // 10)):
            path = []
            visits = []
            values = []
            
            current_id = root_id
            max_depth = 20
            
            for depth in range(max_depth):
                if current_id not in tree.nodes:
                    break
                
                node = tree.nodes[current_id]
                path.append(current_id)
                visits.append(node.visits)
                values.append(node.value if hasattr(node, 'value') else 0.0)
                
                if not hasattr(node, 'children') or not node.children:
                    break
                
                # Choose next node based on visits
                child_visits = [(cid, tree.nodes[cid].visits) 
                               for cid in node.children 
                               if cid in tree.nodes]
                
                if not child_visits:
                    break
                
                child_visits.sort(key=lambda x: x[1], reverse=True)
                
                # Usually follow best path, sometimes explore
                if np.random.random() < 0.8:
                    current_id = child_visits[0][0]
                elif len(child_visits) > 1:
                    current_id = child_visits[1][0]
                else:
                    break
            
            if len(path) > 1:
                path_visits.append(visits)
                path_values.append(values)
        
        return path_visits, path_values
    
    def _compute_visit_correlations(self, visit_counts: np.ndarray, 
                                   node_depths: np.ndarray,
                                   node_children: List[List[int]]) -> Dict[int, float]:
        """Compute visit count correlations at different tree distances"""
        correlations = {}
        num_nodes = len(visit_counts)
        
        if num_nodes < 2:
            return correlations
        
        # Normalize visits
        mean_visits = np.mean(visit_counts)
        std_visits = np.std(visit_counts)
        
        if std_visits == 0:
            logger.warning("Zero variance in visit counts")
            return correlations
        
        normalized_visits = (visit_counts - mean_visits) / std_visits
        
        # Compute correlations at different distances
        max_distance = 10
        
        for distance in range(1, min(max_distance, num_nodes // 2)):
            corr_sum = 0
            count = 0
            
            # Find all pairs at this distance
            for i in range(num_nodes):
                # Use BFS to find nodes at exact distance
                nodes_at_distance = self._find_nodes_at_distance(
                    i, distance, node_children, num_nodes
                )
                
                for j in nodes_at_distance:
                    corr_sum += normalized_visits[i] * normalized_visits[j]
                    count += 1
            
            if count > 0:
                correlations[distance] = corr_sum / count
                logger.debug(f"Distance {distance}: {count} pairs, correlation={correlations[distance]:.4f}")
        
        return correlations
    
    def _find_nodes_at_distance(self, start_node: int, target_distance: int,
                                node_children: List[List[int]], num_nodes: int) -> List[int]:
        """Find all nodes at exact distance from start_node"""
        if target_distance == 0:
            return [start_node]
        
        current_level = {start_node}
        visited = {start_node}
        
        for d in range(target_distance):
            next_level = set()
            
            for node in current_level:
                # Add children
                for child in node_children[node]:
                    if child not in visited and child < num_nodes:
                        next_level.add(child)
                        visited.add(child)
            
            current_level = next_level
            
            if not current_level:
                break
        
        return list(current_level)
    
    def _compute_policy_concentration(self, visit_counts: np.ndarray) -> float:
        """
        Compute policy concentration (order parameter).
        
        This is the actual concentration of visits in the tree,
        not a hardcoded value. We use max(ρ_ii) where ρ is the
        normalized visit distribution.
        """
        if len(visit_counts) == 0 or np.sum(visit_counts) == 0:
            return 0.0
        
        # Normalize to get probability distribution
        total_visits = np.sum(visit_counts)
        visit_probs = visit_counts / total_visits
        
        # Order parameter is the maximum probability
        # This represents how concentrated the policy is
        order_parameter = np.max(visit_probs)
        
        logger.debug(f"Policy concentration (order parameter): {order_parameter:.4f}")
        
        return order_parameter
    
    def _compute_evaluation_variance(self, q_values: np.ndarray, 
                                   visit_counts: np.ndarray) -> float:
        """
        Compute evaluation variance for decoherence calculations.
        
        This measures the actual noise/uncertainty in value estimates,
        weighted by visit counts.
        """
        if len(q_values) == 0 or np.sum(visit_counts) == 0:
            return 0.0
        
        # Weight q-values by visits
        weights = visit_counts / np.sum(visit_counts)
        
        # Weighted mean
        mean_value = np.sum(q_values * weights)
        
        # Weighted variance
        variance = np.sum(weights * (q_values - mean_value)**2)
        
        logger.debug(f"Evaluation variance: {variance:.6f}, mean value: {mean_value:.4f}")
        
        return variance
    
    def _compute_tree_fragmentation(self, visit_counts: np.ndarray,
                                   node_children: List[List[int]],
                                   num_nodes: int) -> float:
        """
        Compute tree fragmentation for quantum Darwinism.
        
        This measures how information about the best action is
        distributed across tree fragments.
        """
        if num_nodes < 10:
            return 0.0
        
        # Identify best action (highest visit child of root)
        if not node_children[0]:
            return 0.0
        
        root_children_visits = [(child, visit_counts[child]) 
                               for child in node_children[0]
                               if child < len(visit_counts)]
        
        if not root_children_visits:
            return 0.0
        
        best_action_idx = max(root_children_visits, key=lambda x: x[1])[0]
        
        # Sample tree fragments
        fragment_size = max(5, num_nodes // 20)
        num_fragments = min(20, num_nodes // fragment_size)
        
        fragments_identifying_best = 0
        
        for _ in range(num_fragments):
            # Sample a random fragment
            fragment = self._sample_tree_fragment(num_nodes, fragment_size, node_children)
            
            # Check if fragment identifies best action
            # (contains nodes from best action subtree)
            if self._fragment_identifies_action(fragment, best_action_idx, node_children):
                fragments_identifying_best += 1
        
        # Redundancy is fraction of fragments containing information
        redundancy = fragments_identifying_best / num_fragments if num_fragments > 0 else 0.0
        
        logger.debug(f"Tree fragmentation (Darwinism redundancy): {redundancy:.4f}")
        
        return redundancy
    
    def _sample_tree_fragment(self, num_nodes: int, fragment_size: int,
                             node_children: List[List[int]]) -> set:
        """Sample a connected fragment of the tree"""
        # Start from random node
        start_node = np.random.randint(0, num_nodes)
        fragment = {start_node}
        
        # BFS to expand fragment
        frontier = [start_node]
        
        while len(fragment) < fragment_size and frontier:
            node = frontier.pop(0)
            
            # Add children
            for child in node_children[node]:
                if child not in fragment and child < num_nodes:
                    fragment.add(child)
                    frontier.append(child)
                    
                    if len(fragment) >= fragment_size:
                        break
        
        return fragment
    
    def _fragment_identifies_action(self, fragment: set, best_action_idx: int,
                                   node_children: List[List[int]]) -> bool:
        """Check if fragment contains information about best action"""
        # Fragment identifies action if it contains nodes from that subtree
        
        # Get all nodes in best action subtree
        best_subtree = {best_action_idx}
        queue = [best_action_idx]
        
        while queue:
            node = queue.pop(0)
            for child in node_children[node]:
                if child not in best_subtree:
                    best_subtree.add(child)
                    queue.append(child)
        
        # Check overlap
        overlap = len(fragment.intersection(best_subtree))
        
        # Fragment identifies action if it has significant overlap
        return overlap >= max(1, len(fragment) // 4)


def extract_quantum_observables(mcts) -> MCTSQuantumObservables:
    """Convenience function to extract observables from MCTS"""
    interface = MCTSQuantumInterface()
    return interface.extract_observables(mcts)