"""
Quantum Darwinism measurement for MCTS.

This module measures actual information redundancy in MCTS trees,
tracking how information about good moves proliferates through the
tree structure via multiple paths and correlations.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import logging
from collections import defaultdict
from scipy.stats import entropy

logger = logging.getLogger(__name__)


@dataclass
class RedundancyMeasurement:
    """Measurement of information redundancy at a specific fragment size"""
    fragment_size: int
    mutual_information: float
    redundancy_ratio: float
    n_fragments: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'fragment_size': self.fragment_size,
            'mutual_information': self.mutual_information,
            'redundancy_ratio': self.redundancy_ratio,
            'n_fragments': self.n_fragments
        }


class QuantumDarwinismMeasurer:
    """
    Measures Quantum Darwinism phenomena in MCTS.
    
    Key insight: Information about good moves becomes redundantly
    encoded across many tree branches, similar to how quantum
    information proliferates into the environment.
    """
    
    def __init__(self):
        self.measurements = []
        
    def measure_information_redundancy(self, mcts_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Measure how information is redundantly encoded in tree structure.
        
        Args:
            mcts_snapshot: MCTS tree snapshot with node data
            
        Returns:
            Dictionary with redundancy measurements
        """
        results = {
            'redundancy_curve': [],
            'plateau_fragment_size': None,
            'total_mutual_information': 0.0,
            'effective_redundancy': 0.0
        }
        
        # Extract tree structure
        if 'tree_structure' in mcts_snapshot:
            tree = mcts_snapshot['tree_structure']
        else:
            # Try to reconstruct from flat data
            tree = self._reconstruct_tree_structure(mcts_snapshot)
        
        if not tree:
            logger.warning("No tree structure available for Darwinism analysis")
            return results
        
        # Identify the system of interest (top actions at root)
        system_actions = self._identify_system_actions(tree)
        
        if not system_actions:
            return results
        
        # Measure mutual information for different fragment sizes
        max_fragment_size = min(50, self._count_total_nodes(tree) // 2)
        
        for fragment_size in range(1, max_fragment_size + 1):
            measurement = self._measure_fragment_information(
                tree, system_actions, fragment_size
            )
            results['redundancy_curve'].append(measurement.to_dict())
        
        # Find plateau (where redundancy saturates)
        results['plateau_fragment_size'] = self._find_redundancy_plateau(
            results['redundancy_curve']
        )
        
        # Calculate total and effective redundancy
        if results['redundancy_curve']:
            mi_values = [r['mutual_information'] for r in results['redundancy_curve']]
            results['total_mutual_information'] = float(np.max(mi_values))
            
            # Effective redundancy: how many independent copies of information
            if results['plateau_fragment_size']:
                plateau_mi = results['redundancy_curve'][results['plateau_fragment_size']-1]['mutual_information']
                total_nodes = self._count_total_nodes(tree)
                results['effective_redundancy'] = total_nodes / results['plateau_fragment_size']
        
        return results
    
    def _reconstruct_tree_structure(self, snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Reconstruct tree structure from flat snapshot data"""
        if 'visits' not in snapshot or 'q_values' not in snapshot:
            return None
        
        # Simple reconstruction: create root with children
        visits = np.array(snapshot['visits'])
        q_values = np.array(snapshot['q_values'])
        
        root = {
            'visits': visits.sum(),
            'q_value': np.average(q_values, weights=visits),
            'children': []
        }
        
        # Add children (actions)
        for i, (v, q) in enumerate(zip(visits, q_values)):
            if v > 0:
                child = {
                    'action': i,
                    'visits': int(v),
                    'q_value': float(q),
                    'children': []  # Leaf nodes
                }
                root['children'].append(child)
        
        return root
    
    def _identify_system_actions(self, tree: Dict[str, Any]) -> List[int]:
        """Identify the 'system' - top actions we want to track"""
        if 'children' not in tree or not tree['children']:
            return []
        
        # Sort children by visits
        children = sorted(tree['children'], 
                         key=lambda c: c.get('visits', 0), 
                         reverse=True)
        
        # Take top 2-3 actions as the "system"
        n_system_actions = min(3, len(children))
        return [c['action'] for c in children[:n_system_actions]]
    
    def _measure_fragment_information(self, tree: Dict[str, Any], 
                                    system_actions: List[int],
                                    fragment_size: int) -> RedundancyMeasurement:
        """
        Measure mutual information between system and environment fragment.
        
        Args:
            tree: Tree structure
            system_actions: Actions that define the system
            fragment_size: Size of environment fragment
            
        Returns:
            RedundancyMeasurement
        """
        # Collect all environment fragments of given size
        all_nodes = self._collect_all_nodes(tree)
        
        if len(all_nodes) < fragment_size:
            return RedundancyMeasurement(
                fragment_size=fragment_size,
                mutual_information=0.0,
                redundancy_ratio=0.0,
                n_fragments=0
            )
        
        # Sample random fragments and measure MI
        n_samples = min(100, len(all_nodes) // fragment_size)
        mi_values = []
        
        for _ in range(n_samples):
            # Random fragment
            fragment_idx = np.random.choice(len(all_nodes), fragment_size, replace=False)
            fragment = [all_nodes[i] for i in fragment_idx]
            
            # Measure mutual information
            mi = self._mutual_information_with_system(fragment, system_actions)
            mi_values.append(mi)
        
        # Average mutual information
        avg_mi = np.mean(mi_values) if mi_values else 0.0
        
        # Redundancy ratio: how much of system information is in fragment
        system_entropy = self._system_entropy(system_actions)
        redundancy_ratio = avg_mi / system_entropy if system_entropy > 0 else 0.0
        
        return RedundancyMeasurement(
            fragment_size=fragment_size,
            mutual_information=float(avg_mi),
            redundancy_ratio=float(redundancy_ratio),
            n_fragments=n_samples
        )
    
    def _collect_all_nodes(self, tree: Dict[str, Any], nodes: List[Dict] = None) -> List[Dict]:
        """Recursively collect all nodes in tree"""
        if nodes is None:
            nodes = []
        
        nodes.append(tree)
        
        if 'children' in tree:
            for child in tree['children']:
                self._collect_all_nodes(child, nodes)
        
        return nodes
    
    def _count_total_nodes(self, tree: Dict[str, Any]) -> int:
        """Count total number of nodes in tree"""
        count = 1  # Count self
        
        if 'children' in tree:
            for child in tree['children']:
                count += self._count_total_nodes(child)
        
        return count
    
    def _mutual_information_with_system(self, fragment: List[Dict], 
                                      system_actions: List[int]) -> float:
        """
        Calculate mutual information between fragment and system.
        
        MI(S:E) = H(S) + H(E) - H(S,E)
        """
        # Extract features from fragment that correlate with system
        fragment_features = self._extract_fragment_features(fragment, system_actions)
        
        if fragment_features.size == 0:
            return 0.0
        
        # Convert to probability distributions
        # System distribution (uniform over identified actions)
        p_system = np.ones(len(system_actions)) / len(system_actions)
        
        # Fragment distribution (based on extracted features)
        p_fragment = fragment_features / fragment_features.sum()
        
        # Joint distribution (simplified: correlation strength)
        correlation = self._measure_correlation(fragment, system_actions)
        
        # Mutual information approximation
        h_system = entropy(p_system)
        h_fragment = entropy(p_fragment)
        
        # Joint entropy depends on correlation
        h_joint = h_system + h_fragment * (1 - correlation)
        
        mi = h_system + h_fragment - h_joint
        
        return max(0.0, mi)  # Ensure non-negative
    
    def _extract_fragment_features(self, fragment: List[Dict], 
                                  system_actions: List[int]) -> np.ndarray:
        """Extract features from fragment relevant to system actions"""
        features = []
        
        for node in fragment:
            # Check if node contains information about system actions
            if 'action' in node and node['action'] in system_actions:
                # Direct information
                features.append(node.get('visits', 0))
            elif 'children' in node:
                # Indirect information through children
                for child in node['children']:
                    if child.get('action') in system_actions:
                        features.append(child.get('visits', 0))
        
        return np.array(features) if features else np.array([0])
    
    def _measure_correlation(self, fragment: List[Dict], 
                           system_actions: List[int]) -> float:
        """Measure correlation between fragment and system"""
        # Count how many fragment nodes reference system actions
        references = 0
        total = len(fragment)
        
        for node in fragment:
            # Direct reference
            if 'action' in node and node['action'] in system_actions:
                references += 1
            # Indirect reference through Q-values
            elif 'q_value' in node and 'children' in node:
                # Check if Q-values correlate with system actions
                for child in node['children']:
                    if child.get('action') in system_actions and child.get('visits', 0) > 0:
                        references += 0.5  # Partial correlation
                        break
        
        return references / total if total > 0 else 0.0
    
    def _system_entropy(self, system_actions: List[int]) -> float:
        """Calculate entropy of system (action distribution)"""
        # Uniform distribution over system actions
        return np.log(len(system_actions)) if system_actions else 0.0
    
    def _find_redundancy_plateau(self, redundancy_curve: List[Dict[str, float]]) -> Optional[int]:
        """Find where redundancy curve plateaus"""
        if len(redundancy_curve) < 3:
            return None
        
        mi_values = [r['mutual_information'] for r in redundancy_curve]
        
        # Find where MI stops increasing significantly
        threshold = 0.9 * max(mi_values) if mi_values else 0
        
        for i, mi in enumerate(mi_values):
            if mi >= threshold:
                return i + 1  # Fragment size (1-indexed)
        
        return None
    
    def analyze_information_proliferation(self, trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze how information proliferates over time.
        
        Args:
            trajectory: List of MCTS snapshots over time
            
        Returns:
            Analysis results
        """
        results = {
            'proliferation_timeline': [],
            'saturation_time': None,
            'proliferation_rate': 0.0,
            'final_redundancy': 0.0
        }
        
        for t, snapshot in enumerate(trajectory):
            redundancy = self.measure_information_redundancy(snapshot)
            
            if redundancy['effective_redundancy'] > 0:
                results['proliferation_timeline'].append({
                    'time': t,
                    'redundancy': redundancy['effective_redundancy'],
                    'mutual_information': redundancy['total_mutual_information']
                })
        
        if results['proliferation_timeline']:
            # Find saturation time
            redundancies = [p['redundancy'] for p in results['proliferation_timeline']]
            max_redundancy = max(redundancies)
            
            for i, r in enumerate(redundancies):
                if r >= 0.9 * max_redundancy:
                    results['saturation_time'] = i
                    break
            
            # Calculate proliferation rate
            if len(redundancies) >= 2:
                # Linear fit to early proliferation
                early_points = min(5, len(redundancies) // 2)
                times = np.arange(early_points)
                early_redundancies = redundancies[:early_points]
                
                if np.std(early_redundancies) > 0:
                    rate = np.polyfit(times, early_redundancies, 1)[0]
                    results['proliferation_rate'] = float(rate)
            
            results['final_redundancy'] = float(redundancies[-1])
        
        return results