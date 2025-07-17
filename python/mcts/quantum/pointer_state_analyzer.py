"""
Pointer state analysis for MCTS.

This module measures how certain states (action sequences) become
"preferred" or robust under the dynamics, analogous to quantum
einselection and pointer states.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy.stats import entropy
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PointerState:
    """Represents a pointer state (robust action sequence)"""
    action_sequence: Tuple[int, ...]
    robustness: float  # How stable under perturbations
    concentration: float  # How dominant in visit distribution
    persistence_time: int  # How long it remains preferred
    envariance: float  # Combined robustness-concentration score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action_sequence': list(self.action_sequence),
            'robustness': self.robustness,
            'concentration': self.concentration,
            'persistence_time': self.persistence_time,
            'envariance': self.envariance
        }


class PointerStateAnalyzer:
    """
    Analyzes pointer states in MCTS trees.
    
    Key insights:
    - Heavily visited action sequences are like pointer states
    - They remain robust under tree growth (environmental monitoring)
    - Envariance = product of robustness and concentration
    """
    
    def __init__(self, sequence_length: int = 3):
        self.sequence_length = sequence_length
        self.pointer_states: List[PointerState] = []
        
    def analyze_pointer_states(self, mcts_trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze pointer states across MCTS trajectory.
        
        Args:
            mcts_trajectory: List of MCTS snapshots over time
            
        Returns:
            Dictionary with pointer state analysis
        """
        results = {
            'pointer_states': [],
            'envariance_evolution': [],
            'einselection_rate': 0.0,
            'dominant_sequences': [],
            'stability_analysis': {}
        }
        
        if len(mcts_trajectory) < 2:
            logger.warning("Need at least 2 snapshots for pointer state analysis")
            return results
        
        # Track action sequences across time
        sequence_history = []
        
        for t, snapshot in enumerate(mcts_trajectory):
            sequences = self._extract_action_sequences(snapshot)
            sequence_history.append(sequences)
            
            if t > 0:
                # Measure envariance at this timestep
                envariance = self._compute_envariance(sequences, sequence_history)
                results['envariance_evolution'].append({
                    'time': t,
                    'envariance': envariance['total'],
                    'robustness': envariance['avg_robustness'],
                    'concentration': envariance['avg_concentration'],
                    'n_pointer_states': envariance['n_pointer_states']
                })
        
        # Identify persistent pointer states
        persistent_states = self._identify_persistent_states(sequence_history)
        
        for seq, data in persistent_states.items():
            pointer_state = PointerState(
                action_sequence=seq,
                robustness=data['robustness'],
                concentration=data['concentration'],
                persistence_time=data['persistence'],
                envariance=data['robustness'] * data['concentration']
            )
            results['pointer_states'].append(pointer_state.to_dict())
        
        # Calculate einselection rate
        if results['envariance_evolution']:
            envariances = [e['envariance'] for e in results['envariance_evolution']]
            if len(envariances) >= 2:
                # Fit exponential growth to measure selection rate
                times = np.arange(len(envariances))
                log_env = np.log(np.array(envariances) + 1e-10)
                
                if not np.any(np.isnan(log_env)):
                    rate = np.polyfit(times, log_env, 1)[0]
                    results['einselection_rate'] = float(rate)
        
        # Find dominant sequences
        if results['pointer_states']:
            sorted_states = sorted(results['pointer_states'], 
                                 key=lambda x: x['envariance'], 
                                 reverse=True)
            results['dominant_sequences'] = sorted_states[:5]  # Top 5
        
        # Stability analysis
        results['stability_analysis'] = self._analyze_stability(sequence_history)
        
        return results
    
    def _extract_action_sequences(self, snapshot: Dict[str, Any]) -> Dict[Tuple[int, ...], float]:
        """Extract action sequences and their visit frequencies"""
        sequences = defaultdict(float)
        
        if 'tree_structure' in snapshot:
            self._extract_sequences_recursive(
                snapshot['tree_structure'], 
                tuple(), 
                sequences
            )
        else:
            # Try to reconstruct from action history if available
            if 'action_sequences' in snapshot:
                for seq_data in snapshot['action_sequences']:
                    seq = tuple(seq_data['actions'][:self.sequence_length])
                    sequences[seq] += seq_data.get('visits', 1)
        
        # Normalize to probabilities
        total = sum(sequences.values())
        if total > 0:
            for seq in sequences:
                sequences[seq] /= total
        
        return dict(sequences)
    
    def _extract_sequences_recursive(self, node: Dict[str, Any], 
                                   current_seq: Tuple[int, ...],
                                   sequences: Dict[Tuple[int, ...], float],
                                   depth: int = 0):
        """Recursively extract action sequences from tree"""
        if depth >= self.sequence_length:
            sequences[current_seq[:self.sequence_length]] += node.get('visits', 0)
            return
        
        if 'children' not in node:
            if len(current_seq) > 0:
                sequences[current_seq] += node.get('visits', 0)
            return
        
        for child in node['children']:
            action = child.get('action', 0)
            new_seq = current_seq + (action,)
            self._extract_sequences_recursive(child, new_seq, sequences, depth + 1)
    
    def _compute_envariance(self, current_sequences: Dict[Tuple[int, ...], float],
                           history: List[Dict[Tuple[int, ...], float]]) -> Dict[str, float]:
        """Compute envariance scores for current timestep"""
        results = {
            'total': 0.0,
            'avg_robustness': 0.0,
            'avg_concentration': 0.0,
            'n_pointer_states': 0
        }
        
        if not current_sequences or not history:
            return results
        
        pointer_threshold = 0.1  # Minimum concentration to be considered
        robustness_scores = []
        concentration_scores = []
        
        for seq, concentration in current_sequences.items():
            if concentration < pointer_threshold:
                continue
            
            # Robustness: how stable across history
            robustness = self._compute_robustness(seq, history)
            
            # Envariance for this sequence
            envariance = robustness * concentration
            
            results['total'] += envariance
            robustness_scores.append(robustness)
            concentration_scores.append(concentration)
            results['n_pointer_states'] += 1
        
        if robustness_scores:
            results['avg_robustness'] = np.mean(robustness_scores)
            results['avg_concentration'] = np.mean(concentration_scores)
        
        return results
    
    def _compute_robustness(self, sequence: Tuple[int, ...], 
                           history: List[Dict[Tuple[int, ...], float]]) -> float:
        """Compute robustness of a sequence across history"""
        if not history:
            return 0.0
        
        # Track presence and stability
        presences = []
        concentrations = []
        
        for past_sequences in history:
            if sequence in past_sequences:
                presences.append(1.0)
                concentrations.append(past_sequences[sequence])
            else:
                presences.append(0.0)
                concentrations.append(0.0)
        
        if not presences:
            return 0.0
        
        # Robustness combines persistence and stability
        persistence = np.mean(presences)
        
        # Stability: low variance in concentration when present
        present_concentrations = [c for c, p in zip(concentrations, presences) if p > 0]
        if len(present_concentrations) >= 2:
            stability = 1.0 / (1.0 + np.std(present_concentrations))
        else:
            stability = 1.0 if present_concentrations else 0.0
        
        robustness = persistence * stability
        
        return float(robustness)
    
    def _identify_persistent_states(self, 
                                   history: List[Dict[Tuple[int, ...], float]]) -> Dict[Tuple[int, ...], Dict]:
        """Identify sequences that persist as pointer states"""
        persistent = {}
        
        # Count persistence and average properties
        for sequences in history:
            for seq, conc in sequences.items():
                if seq not in persistent:
                    persistent[seq] = {
                        'count': 0,
                        'concentrations': [],
                        'first_seen': len(persistent)
                    }
                
                persistent[seq]['count'] += 1
                persistent[seq]['concentrations'].append(conc)
        
        # Filter and compute final properties
        min_persistence = max(2, len(history) // 4)  # At least 25% of time
        
        pointer_states = {}
        for seq, data in persistent.items():
            if data['count'] >= min_persistence:
                avg_concentration = np.mean(data['concentrations'])
                
                # Only keep if significant concentration
                if avg_concentration >= 0.05:
                    robustness = data['count'] / len(history)
                    
                    pointer_states[seq] = {
                        'robustness': robustness,
                        'concentration': avg_concentration,
                        'persistence': data['count']
                    }
        
        return pointer_states
    
    def _analyze_stability(self, history: List[Dict[Tuple[int, ...], float]]) -> Dict[str, Any]:
        """Analyze stability of the pointer state landscape"""
        if len(history) < 2:
            return {}
        
        # Measure how distributions change over time
        js_divergences = []
        
        for i in range(1, len(history)):
            js_div = self._js_divergence(history[i-1], history[i])
            js_divergences.append(js_div)
        
        # Analyze convergence
        convergence_rate = 0.0
        if len(js_divergences) >= 3:
            # Fit exponential decay to divergences
            times = np.arange(len(js_divergences))
            log_divs = np.log(np.array(js_divergences) + 1e-10)
            
            if not np.any(np.isnan(log_divs)):
                convergence_rate = -np.polyfit(times, log_divs, 1)[0]
        
        return {
            'average_divergence': float(np.mean(js_divergences)) if js_divergences else 0.0,
            'final_divergence': float(js_divergences[-1]) if js_divergences else 0.0,
            'convergence_rate': float(convergence_rate),
            'is_converged': js_divergences[-1] < 0.01 if js_divergences else False
        }
    
    def _js_divergence(self, dist1: Dict[Tuple[int, ...], float], 
                      dist2: Dict[Tuple[int, ...], float]) -> float:
        """Compute Jensen-Shannon divergence between distributions"""
        # Get all sequences
        all_seqs = set(dist1.keys()) | set(dist2.keys())
        
        if not all_seqs:
            return 0.0
        
        # Create probability arrays
        p1 = np.array([dist1.get(seq, 0.0) for seq in all_seqs])
        p2 = np.array([dist2.get(seq, 0.0) for seq in all_seqs])
        
        # Normalize (in case not already)
        p1 = p1 / (p1.sum() + 1e-10)
        p2 = p2 / (p2.sum() + 1e-10)
        
        # JS divergence
        m = 0.5 * (p1 + p2)
        js_div = 0.5 * entropy(p1, m) + 0.5 * entropy(p2, m)
        
        return float(js_div)