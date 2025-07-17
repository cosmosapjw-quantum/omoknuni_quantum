"""
Non-equilibrium statistical mechanics analyzer for MCTS.

This module measures actual work, heat, and entropy production
in MCTS dynamics, verifying fluctuation theorems.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy.stats import gaussian_kde

logger = logging.getLogger(__name__)


@dataclass
class WorkMeasurement:
    """Single work measurement along MCTS trajectory"""
    initial_state: int  # Initial action/node
    final_state: int    # Final action/node
    work: float        # Work done (change in Q-values)
    heat: float        # Heat dissipated
    time_step: int     # When measurement was taken
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'initial_state': self.initial_state,
            'final_state': self.final_state,
            'work': self.work,
            'heat': self.heat,
            'time_step': self.time_step
        }


class NonEquilibriumAnalyzer:
    """
    Analyzes non-equilibrium properties of MCTS dynamics.
    
    Key measurements:
    - Work: Change in Q-values along trajectories
    - Heat: Energy dissipated through exploration
    - Entropy production: Irreversibility measure
    - Fluctuation theorem verification
    """
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.work_measurements = []
        
    def analyze_trajectory(self, mcts_trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze non-equilibrium properties of MCTS trajectory.
        
        Args:
            mcts_trajectory: List of MCTS snapshots over time
            
        Returns:
            Dictionary with non-equilibrium analysis results
        """
        results = {
            'work_measurements': [],
            'heat_measurements': [],
            'entropy_production': [],
            'jarzynski_verification': {},
            'crooks_verification': {},
            'landauer_bound': {},
            'time_reversal_asymmetry': 0.0
        }
        
        if len(mcts_trajectory) < 2:
            logger.warning("Need at least 2 snapshots for non-equilibrium analysis")
            return results
        
        # Extract work and heat measurements
        work_values = []
        heat_values = []
        
        for t in range(1, len(mcts_trajectory)):
            work, heat = self._measure_work_and_heat(
                mcts_trajectory[t-1], 
                mcts_trajectory[t],
                t
            )
            
            if work is not None:
                work_values.append(work)
                heat_values.append(heat)
                
                results['work_measurements'].append({
                    'time': t,
                    'work': work,
                    'heat': heat
                })
        
        # Calculate entropy production
        entropy_production = self._calculate_entropy_production(
            mcts_trajectory, work_values, heat_values
        )
        results['entropy_production'] = entropy_production
        
        # Verify Jarzynski equality
        if work_values:
            results['jarzynski_verification'] = self._verify_jarzynski(
                work_values, self.temperature
            )
        
        # Verify Crooks fluctuation theorem
        if len(work_values) >= 20:
            results['crooks_verification'] = self._verify_crooks(
                work_values, self.temperature
            )
        
        # Check Landauer bound
        results['landauer_bound'] = self._check_landauer_bound(
            mcts_trajectory, heat_values
        )
        
        # Measure time-reversal asymmetry
        results['time_reversal_asymmetry'] = self._measure_time_asymmetry(
            mcts_trajectory
        )
        
        return results
    
    def _measure_work_and_heat(self, snapshot1: Dict[str, Any], 
                              snapshot2: Dict[str, Any], 
                              time_step: int) -> Tuple[Optional[float], Optional[float]]:
        """
        Measure work and heat between two snapshots.
        
        Work = Change in "energy" (Q-values)
        Heat = Energy dissipated through exploration
        """
        if 'q_values' not in snapshot1 or 'q_values' not in snapshot2:
            return None, None
        
        q1 = np.array(snapshot1['q_values'])
        q2 = np.array(snapshot2['q_values'])
        
        if 'visits' in snapshot1 and 'visits' in snapshot2:
            v1 = np.array(snapshot1['visits'])
            v2 = np.array(snapshot2['visits'])
            
            # Ensure arrays have same length
            min_len = min(len(q1), len(q2))
            q1, q2 = q1[:min_len], q2[:min_len]
            v1, v2 = v1[:min_len], v2[:min_len]
            
            # Work: Average Q-value change weighted by visits
            if v1.sum() > 0 and v2.sum() > 0:
                avg_q1 = np.average(q1, weights=v1)
                avg_q2 = np.average(q2, weights=v2)
                work = avg_q2 - avg_q1
                
                # Heat: Energy "lost" to exploration (increase in entropy)
                # Approximated by change in visit distribution entropy
                p1 = v1 / v1.sum()
                p2 = v2 / v2.sum()
                
                entropy1 = -np.sum(p1 * np.log(p1 + 1e-10))
                entropy2 = -np.sum(p2 * np.log(p2 + 1e-10))
                
                heat = self.temperature * (entropy2 - entropy1)
                
                return float(work), float(heat)
        
        # Fallback: just use Q-value change
        work = float(np.mean(q2) - np.mean(q1))
        heat = 0.0  # No visit information
        
        return work, heat
    
    def _calculate_entropy_production(self, trajectory: List[Dict[str, Any]],
                                    work_values: List[float],
                                    heat_values: List[float]) -> List[Dict[str, float]]:
        """
        Calculate entropy production rate.
        
        dS/dt = (dQ/dt)/T + dS_internal/dt
        """
        entropy_production = []
        
        for t, (work, heat) in enumerate(zip(work_values, heat_values)):
            # Entropy production = heat/T + internal entropy change
            # For MCTS, internal entropy change relates to exploration
            
            if t < len(trajectory) - 1:
                snapshot = trajectory[t+1]
                
                # Internal entropy from visit distribution
                if 'visits' in snapshot:
                    visits = np.array(snapshot['visits'])
                    if visits.sum() > 0:
                        probs = visits / visits.sum()
                        internal_entropy = -np.sum(probs * np.log(probs + 1e-10))
                    else:
                        internal_entropy = 0.0
                else:
                    internal_entropy = 0.0
                
                # Total entropy production
                sigma = heat / self.temperature + internal_entropy
                
                entropy_production.append({
                    'time': t,
                    'entropy_production_rate': sigma,
                    'heat_contribution': heat / self.temperature,
                    'internal_contribution': internal_entropy
                })
        
        return entropy_production
    
    def _verify_jarzynski(self, work_values: List[float], 
                         temperature: float) -> Dict[str, Any]:
        """
        Verify Jarzynski equality: <exp(-βW)> = exp(-βΔF)
        """
        beta = 1.0 / temperature
        work_array = np.array(work_values)
        
        # Left side: average of exp(-βW)
        exp_work = np.exp(-beta * work_array)
        jarzynski_avg = np.mean(exp_work)
        
        # Estimate free energy difference from average work
        delta_f_estimate = np.mean(work_array)
        
        # Right side: exp(-βΔF)
        exp_delta_f = np.exp(-beta * delta_f_estimate)
        
        # Check equality (should be close to 1 if satisfied)
        ratio = jarzynski_avg / exp_delta_f
        
        return {
            'jarzynski_average': float(jarzynski_avg),
            'exp_delta_f': float(exp_delta_f),
            'ratio': float(ratio),
            'satisfied': abs(ratio - 1.0) < 0.5,  # Loose criterion for finite samples
            'delta_f_estimate': float(delta_f_estimate),
            'n_samples': len(work_values)
        }
    
    def _verify_crooks(self, work_values: List[float], 
                      temperature: float) -> Dict[str, Any]:
        """
        Verify Crooks fluctuation theorem.
        
        P(W) / P(-W) = exp(β(W - ΔF))
        """
        beta = 1.0 / temperature
        work_array = np.array(work_values)
        
        # Estimate distributions
        # For MCTS, "reverse" work is approximated by negative perturbations
        forward_work = work_array[work_array > 0]
        reverse_work = -work_array[work_array < 0]  # Approximate reverse process
        
        if len(forward_work) < 5 or len(reverse_work) < 5:
            return {
                'verified': False,
                'reason': 'Insufficient forward/reverse samples'
            }
        
        # Estimate PDFs using KDE
        try:
            kde_forward = gaussian_kde(forward_work)
            kde_reverse = gaussian_kde(reverse_work)
            
            # Test points
            test_work = np.linspace(0.1, min(forward_work.max(), reverse_work.max()), 20)
            
            # Calculate ratio
            p_forward = kde_forward(test_work)
            p_reverse = kde_reverse(test_work)
            
            # Avoid division by zero
            valid_idx = p_reverse > 1e-10
            test_work = test_work[valid_idx]
            p_forward = p_forward[valid_idx]
            p_reverse = p_reverse[valid_idx]
            
            if len(test_work) == 0:
                return {'verified': False, 'reason': 'No valid test points'}
            
            ratio_measured = p_forward / p_reverse
            
            # Expected ratio from Crooks
            delta_f = np.mean(work_array)
            ratio_expected = np.exp(beta * (test_work - delta_f))
            
            # Check agreement
            log_ratio_diff = np.abs(np.log(ratio_measured) - np.log(ratio_expected))
            mean_deviation = np.mean(log_ratio_diff)
            
            return {
                'verified': mean_deviation < 1.0,  # Loose criterion
                'mean_log_deviation': float(mean_deviation),
                'test_points': len(test_work),
                'delta_f': float(delta_f)
            }
            
        except Exception as e:
            return {
                'verified': False,
                'reason': f'KDE estimation failed: {str(e)}'
            }
    
    def _check_landauer_bound(self, trajectory: List[Dict[str, Any]],
                            heat_values: List[float]) -> Dict[str, Any]:
        """
        Check Landauer's principle: Heat ≥ kT ln(2) × (information erased)
        """
        # For MCTS, information erasure happens when pruning branches
        # or when visit counts reset
        
        info_erased_bits = 0
        
        for t in range(1, len(trajectory)):
            if 'visits' in trajectory[t-1] and 'visits' in trajectory[t]:
                v1 = np.array(trajectory[t-1]['visits'])
                v2 = np.array(trajectory[t]['visits'])
                
                # Detect information loss (decrease in total visits or branches)
                if v1.sum() > v2.sum():
                    # Approximate bits erased from visit reduction
                    reduction_ratio = 1 - v2.sum() / v1.sum()
                    info_erased_bits += reduction_ratio * np.log2(len(v1))
        
        # Landauer bound
        landauer_heat = self.temperature * np.log(2) * info_erased_bits
        actual_heat = sum(h for h in heat_values if h > 0)  # Only dissipated heat
        
        return {
            'information_erased_bits': float(info_erased_bits),
            'landauer_bound_heat': float(landauer_heat),
            'actual_heat_dissipated': float(actual_heat),
            'bound_satisfied': actual_heat >= landauer_heat * 0.9,  # 90% tolerance
            'efficiency': float(landauer_heat / actual_heat) if actual_heat > 0 else 0.0
        }
    
    def _measure_time_asymmetry(self, trajectory: List[Dict[str, Any]]) -> float:
        """
        Measure time-reversal asymmetry in MCTS dynamics.
        
        Compares forward and backward transition probabilities.
        """
        if len(trajectory) < 3:
            return 0.0
        
        asymmetry_scores = []
        
        for t in range(1, len(trajectory) - 1):
            if 'visits' not in trajectory[t]:
                continue
            
            visits = np.array(trajectory[t]['visits'])
            if visits.sum() == 0:
                continue
            
            # Forward transition probability (t -> t+1)
            if 'visits' in trajectory[t+1]:
                visits_next = np.array(trajectory[t+1]['visits'])
                if len(visits) == len(visits_next):
                    # Approximate transition matrix from visit changes
                    forward_flow = np.abs(visits_next - visits)
                    forward_prob = forward_flow / (forward_flow.sum() + 1e-10)
            
            # Backward transition probability (t -> t-1)
            if 'visits' in trajectory[t-1]:
                visits_prev = np.array(trajectory[t-1]['visits'])
                if len(visits) == len(visits_prev):
                    backward_flow = np.abs(visits - visits_prev)
                    backward_prob = backward_flow / (backward_flow.sum() + 1e-10)
                    
                    # KL divergence as asymmetry measure
                    kl_div = np.sum(forward_prob * np.log(
                        (forward_prob + 1e-10) / (backward_prob + 1e-10)
                    ))
                    asymmetry_scores.append(kl_div)
        
        return float(np.mean(asymmetry_scores)) if asymmetry_scores else 0.0
    
    def generate_work_distribution(self, all_trajectories: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Generate work distribution from multiple trajectories.
        
        Args:
            all_trajectories: List of MCTS trajectories from multiple games
            
        Returns:
            Work distribution statistics
        """
        all_work_values = []
        
        for trajectory in all_trajectories:
            results = self.analyze_trajectory(trajectory)
            work_measurements = results.get('work_measurements', [])
            all_work_values.extend([w['work'] for w in work_measurements])
        
        if not all_work_values:
            return {}
        
        work_array = np.array(all_work_values)
        
        return {
            'mean_work': float(np.mean(work_array)),
            'std_work': float(np.std(work_array)),
            'min_work': float(np.min(work_array)),
            'max_work': float(np.max(work_array)),
            'n_measurements': len(work_array),
            'positive_work_fraction': float(np.mean(work_array > 0)),
            'histogram': {
                'bins': np.histogram(work_array, bins=50)[1].tolist(),
                'counts': np.histogram(work_array, bins=50)[0].tolist()
            }
        }