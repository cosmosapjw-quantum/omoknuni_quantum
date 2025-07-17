"""
Extract subtree information from MCTS for finite-size scaling.

This module provides utilities to extract depth-wise information from
MCTS trees, enabling finite-size scaling analysis using subtrees.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class SubtreeExtractor:
    """
    Extracts subtree information at different depths from MCTS data.
    
    The key insight is that subtrees at different depths act like
    systems of different effective sizes for scaling analysis.
    """
    
    def __init__(self):
        self.depth_data = defaultdict(lambda: {
            'visits': [],
            'q_values': [],
            'policies': [],
            'node_count': 0
        })
        
    def extract_depth_wise_data(self, mcts_snapshot: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """
        Extract data organized by tree depth from an MCTS snapshot.
        
        Args:
            mcts_snapshot: Single MCTS snapshot containing tree data
            
        Returns:
            Dictionary mapping depth to node data at that depth
        """
        depth_data = defaultdict(lambda: {
            'visits': [],
            'q_values': [],
            'policies': [],
            'node_count': 0
        })
        
        # Check if we have the necessary tree structure data
        if 'tree_structure' in mcts_snapshot:
            # Ideal case: tree structure is provided
            tree = mcts_snapshot['tree_structure']
            self._traverse_tree(tree, depth=0, depth_data=depth_data)
            
        elif 'children_data' in mcts_snapshot:
            # Alternative: children data with depth info
            for child_info in mcts_snapshot['children_data']:
                depth = child_info.get('depth', 0)
                if 'visits' in child_info:
                    depth_data[depth]['visits'].append(child_info['visits'])
                if 'q_value' in child_info:
                    depth_data[depth]['q_values'].append(child_info['q_value'])
                if 'policy' in child_info:
                    depth_data[depth]['policies'].append(child_info['policy'])
                depth_data[depth]['node_count'] += 1
                
        else:
            # Fallback: estimate from flat data
            # Assume root node has most visits
            if 'visits' in mcts_snapshot and 'q_values' in mcts_snapshot:
                visits = np.array(mcts_snapshot['visits'])
                q_values = np.array(mcts_snapshot['q_values'])
                
                # Sort by visits to approximate tree structure
                sorted_idx = np.argsort(visits)[::-1]
                
                # Heuristic: assign depths based on visit counts
                # Top nodes are shallower
                n_nodes = len(visits)
                for i, idx in enumerate(sorted_idx):
                    # Estimate depth based on position in sorted order
                    estimated_depth = int(np.log2(i + 1))
                    depth_data[estimated_depth]['visits'].append(visits[idx])
                    depth_data[estimated_depth]['q_values'].append(q_values[idx])
                    depth_data[estimated_depth]['node_count'] += 1
        
        # Convert lists to arrays
        for depth in depth_data:
            if depth_data[depth]['visits']:
                depth_data[depth]['visits'] = np.array(depth_data[depth]['visits'])
                depth_data[depth]['q_values'] = np.array(depth_data[depth]['q_values'])
                if depth_data[depth]['policies']:
                    depth_data[depth]['policies'] = np.array(depth_data[depth]['policies'])
        
        return dict(depth_data)
    
    def _traverse_tree(self, node: Dict[str, Any], depth: int, 
                      depth_data: Dict[int, Dict[str, Any]]) -> None:
        """Recursively traverse tree structure to extract depth-wise data"""
        # Extract node data
        if 'visits' in node:
            depth_data[depth]['visits'].append(node['visits'])
        if 'q_value' in node:
            depth_data[depth]['q_values'].append(node['q_value'])
        if 'policy' in node:
            depth_data[depth]['policies'].append(node['policy'])
        depth_data[depth]['node_count'] += 1
        
        # Traverse children
        if 'children' in node:
            for child in node['children']:
                self._traverse_tree(child, depth + 1, depth_data)
    
    def compute_subtree_properties(self, depth_data: Dict[int, Dict[str, Any]], 
                                 max_depth: int = 10) -> List[Dict[str, Any]]:
        """
        Compute thermodynamic properties for subtrees at different depths.
        
        Args:
            depth_data: Depth-wise node data
            max_depth: Maximum depth to analyze
            
        Returns:
            List of subtree properties for each depth
        """
        subtree_properties = []
        
        for depth in range(1, min(max_depth + 1, max(depth_data.keys()) + 1)):
            # Aggregate data up to this depth
            subtree_visits = []
            subtree_q_values = []
            total_nodes = 0
            
            for d in range(depth + 1):
                if d in depth_data:
                    if len(depth_data[d]['visits']) > 0:
                        subtree_visits.extend(depth_data[d]['visits'])
                        subtree_q_values.extend(depth_data[d]['q_values'])
                        total_nodes += depth_data[d]['node_count']
            
            if len(subtree_visits) < 10:  # Need minimum data
                continue
            
            visits = np.array(subtree_visits)
            q_values = np.array(subtree_q_values)
            
            # Compute subtree properties
            props = {
                'depth': depth,
                'size': total_nodes,
                'effective_size': 2**depth,  # Tree grows exponentially
                'total_visits': int(visits.sum()),
                'mean_visits': float(visits.mean()),
                'visit_entropy': self._compute_entropy(visits),
                'mean_q_value': float(np.average(q_values, weights=visits)),
                'q_value_variance': float(self._weighted_variance(q_values, visits)),
                'participation_ratio': self._compute_participation_ratio(visits),
                'effective_branching': self._estimate_branching_factor(depth_data, depth)
            }
            
            subtree_properties.append(props)
        
        return subtree_properties
    
    def _compute_entropy(self, visits: np.ndarray) -> float:
        """Compute Shannon entropy of visit distribution"""
        if visits.sum() == 0:
            return 0.0
        probs = visits / visits.sum()
        probs = probs[probs > 1e-10]
        return -np.sum(probs * np.log(probs))
    
    def _weighted_variance(self, values: np.ndarray, weights: np.ndarray) -> float:
        """Compute weighted variance"""
        if weights.sum() == 0:
            return 0.0
        mean = np.average(values, weights=weights)
        variance = np.average((values - mean)**2, weights=weights)
        return variance
    
    def _compute_participation_ratio(self, visits: np.ndarray) -> float:
        """Compute inverse participation ratio (effective number of states)"""
        if visits.sum() == 0:
            return 1.0
        probs = visits / visits.sum()
        return 1.0 / np.sum(probs**2)
    
    def _estimate_branching_factor(self, depth_data: Dict[int, Dict[str, Any]], 
                                  max_depth: int) -> float:
        """Estimate effective branching factor up to given depth"""
        if max_depth == 0 or 0 not in depth_data:
            return 1.0
            
        # Count nodes at each depth
        counts = []
        for d in range(max_depth + 1):
            if d in depth_data:
                counts.append(depth_data[d]['node_count'])
        
        if len(counts) < 2:
            return 1.0
        
        # Estimate branching factor from growth rate
        branching_factors = []
        for i in range(1, len(counts)):
            if counts[i-1] > 0:
                bf = counts[i] / counts[i-1]
                branching_factors.append(bf)
        
        return np.mean(branching_factors) if branching_factors else 1.0
    
    def extract_scaling_observables(self, subtree_props: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Extract observables for finite-size scaling from subtree properties.
        
        Args:
            subtree_props: List of subtree property dictionaries
            
        Returns:
            Dictionary of scaling observables indexed by name
        """
        if not subtree_props:
            return {}
        
        # Extract arrays
        sizes = np.array([p['effective_size'] for p in subtree_props])
        entropies = np.array([p['visit_entropy'] for p in subtree_props])
        mean_q = np.array([p['mean_q_value'] for p in subtree_props])
        q_variance = np.array([p['q_value_variance'] for p in subtree_props])
        participation = np.array([p['participation_ratio'] for p in subtree_props])
        
        # Define scaling observables
        observables = {
            'sizes': sizes,
            'order_parameter': 1.0 - entropies / np.log(sizes),  # Deviation from maximum entropy
            'susceptibility': q_variance * sizes,  # Scaled variance
            'correlation_length': participation / sizes,  # Normalized participation
            'specific_heat': entropies * sizes,  # Extensive entropy
            'magnetization': np.abs(mean_q),  # Absolute Q-value as magnetization analog
        }
        
        return observables