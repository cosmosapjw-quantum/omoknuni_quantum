"""
RG Flow measurement tracking Q-value evolution in MCTS.

This module implements measurement of renormalization group flow
through the MCTS tree, tracking how Q-values transform under
coarse-graining (backpropagation).
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class RGFlowPoint:
    """A point in RG flow trajectory"""
    scale: int  # Tree depth (UV to IR: leaves to root)
    q_mean: float  # Mean Q-value at this scale
    q_variance: float  # Q-value variance
    visit_entropy: float  # Shannon entropy of visit distribution
    n_nodes: int  # Number of nodes at this scale
    participation_ratio: float  # Effective number of active nodes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'scale': self.scale,
            'q_mean': self.q_mean,
            'q_variance': self.q_variance,
            'visit_entropy': self.visit_entropy,
            'n_nodes': self.n_nodes,
            'participation_ratio': self.participation_ratio
        }


@dataclass
class RGTrajectory:
    """Complete RG flow trajectory for one game"""
    flow_points: List[RGFlowPoint]
    beta_function: Optional[List[float]] = None  # dg/d(scale)
    fixed_points: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'flow_points': [p.to_dict() for p in self.flow_points],
            'beta_function': self.beta_function,
            'fixed_points': self.fixed_points
        }


class RGFlowTracker:
    """
    Tracks renormalization group flow through MCTS tree.
    
    Key insight: Backpropagation implements RG transformation,
    coarse-graining information from leaves (UV) to root (IR).
    """
    
    def __init__(self):
        self.trajectories: List[RGTrajectory] = []
        self.flow_data_by_depth = defaultdict(list)
        
    def extract_rg_flow(self, mcts_snapshot: Dict[str, Any]) -> RGTrajectory:
        """
        Extract RG flow from a single MCTS snapshot.
        
        Args:
            mcts_snapshot: MCTS tree snapshot with depth-wise data
            
        Returns:
            RG flow trajectory
        """
        flow_points = []
        
        # Extract depth-wise statistics
        if 'depth_wise_data' in mcts_snapshot:
            depth_data = mcts_snapshot['depth_wise_data']
        else:
            # Try to reconstruct from flat data
            depth_data = self._reconstruct_depth_data(mcts_snapshot)
        
        # Process each scale (depth)
        for depth in sorted(depth_data.keys()):
            data = depth_data[depth]
            
            if 'visits' not in data or len(data['visits']) == 0:
                continue
                
            visits = np.array(data['visits'])
            q_values = np.array(data['q_values']) if 'q_values' in data else np.zeros_like(visits)
            
            # Calculate RG observables
            point = self._calculate_rg_point(depth, visits, q_values)
            flow_points.append(point)
        
        # Create trajectory
        trajectory = RGTrajectory(flow_points=flow_points)
        
        # Calculate beta function if enough points
        if len(flow_points) >= 3:
            trajectory.beta_function = self._calculate_beta_function(flow_points)
            trajectory.fixed_points = self._find_fixed_points(flow_points, trajectory.beta_function)
        
        return trajectory
    
    def _reconstruct_depth_data(self, snapshot: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """Reconstruct depth-wise data from flat snapshot"""
        depth_data = defaultdict(lambda: {'visits': [], 'q_values': []})
        
        # Simple heuristic: use visit counts to estimate depth
        if 'visits' in snapshot and 'q_values' in snapshot:
            visits = np.array(snapshot['visits'])
            q_values = np.array(snapshot['q_values'])
            
            # Sort by visits (higher visits = shallower in tree)
            sorted_idx = np.argsort(visits)[::-1]
            
            # Assign depths based on visit ranking
            for i, idx in enumerate(sorted_idx):
                estimated_depth = int(np.log2(i + 1))
                depth_data[estimated_depth]['visits'].append(visits[idx])
                depth_data[estimated_depth]['q_values'].append(q_values[idx])
        
        # Convert lists to arrays
        for depth in depth_data:
            depth_data[depth]['visits'] = np.array(depth_data[depth]['visits'])
            depth_data[depth]['q_values'] = np.array(depth_data[depth]['q_values'])
        
        return dict(depth_data)
    
    def _calculate_rg_point(self, depth: int, visits: np.ndarray, 
                           q_values: np.ndarray) -> RGFlowPoint:
        """Calculate RG observables at given scale"""
        # Basic statistics
        total_visits = visits.sum()
        if total_visits == 0:
            return RGFlowPoint(
                scale=depth,
                q_mean=0.0,
                q_variance=0.0,
                visit_entropy=0.0,
                n_nodes=len(visits),
                participation_ratio=1.0
            )
        
        # Weighted Q-value statistics
        weights = visits / total_visits
        q_mean = np.average(q_values, weights=weights)
        q_variance = np.average((q_values - q_mean)**2, weights=weights)
        
        # Visit entropy
        probs = visits / total_visits
        probs = probs[probs > 0]
        visit_entropy = -np.sum(probs * np.log(probs))
        
        # Participation ratio
        participation_ratio = 1.0 / np.sum(weights**2)
        
        return RGFlowPoint(
            scale=depth,
            q_mean=float(q_mean),
            q_variance=float(q_variance),
            visit_entropy=float(visit_entropy),
            n_nodes=len(visits),
            participation_ratio=float(participation_ratio)
        )
    
    def _calculate_beta_function(self, flow_points: List[RGFlowPoint]) -> List[float]:
        """
        Calculate RG beta function: β(g) = dg/d(scale)
        
        Here g represents the running coupling (e.g., Q-value variance).
        """
        if len(flow_points) < 2:
            return []
        
        beta_function = []
        
        for i in range(1, len(flow_points)):
            # Change in coupling (using variance as proxy)
            dg = flow_points[i].q_variance - flow_points[i-1].q_variance
            
            # Change in scale
            d_scale = flow_points[i].scale - flow_points[i-1].scale
            
            if d_scale != 0:
                beta = dg / d_scale
            else:
                beta = 0.0
                
            beta_function.append(beta)
        
        return beta_function
    
    def _find_fixed_points(self, flow_points: List[RGFlowPoint], 
                          beta_function: List[float]) -> List[Dict[str, Any]]:
        """Find RG fixed points where beta function crosses zero"""
        fixed_points = []
        
        if len(beta_function) < 2:
            return fixed_points
        
        # Look for sign changes in beta function
        for i in range(1, len(beta_function)):
            if beta_function[i-1] * beta_function[i] < 0:  # Sign change
                # Interpolate to find crossing point
                alpha = abs(beta_function[i-1]) / (abs(beta_function[i-1]) + abs(beta_function[i]))
                
                # Interpolate scale and coupling
                scale = flow_points[i].scale * alpha + flow_points[i+1].scale * (1 - alpha)
                q_variance = flow_points[i].q_variance * alpha + flow_points[i+1].q_variance * (1 - alpha)
                
                # Classify fixed point
                if i < len(beta_function) - 1:
                    # Check derivative of beta function
                    d_beta = beta_function[i] - beta_function[i-1]
                    stability = 'stable' if d_beta < 0 else 'unstable'
                else:
                    stability = 'unknown'
                
                fixed_points.append({
                    'scale': float(scale),
                    'coupling': float(q_variance),
                    'type': stability,
                    'beta_derivative': float(d_beta) if 'd_beta' in locals() else 0.0
                })
        
        return fixed_points
    
    def analyze_rg_flow_ensemble(self, trajectories: List[RGTrajectory]) -> Dict[str, Any]:
        """
        Analyze ensemble of RG flow trajectories.
        
        Args:
            trajectories: List of RG trajectories from multiple games
            
        Returns:
            Ensemble analysis results
        """
        if not trajectories:
            return {}
        
        # Collect flow points by scale
        scale_data = defaultdict(lambda: {
            'q_means': [],
            'q_variances': [],
            'entropies': [],
            'participation_ratios': []
        })
        
        for traj in trajectories:
            for point in traj.flow_points:
                scale_data[point.scale]['q_means'].append(point.q_mean)
                scale_data[point.scale]['q_variances'].append(point.q_variance)
                scale_data[point.scale]['entropies'].append(point.visit_entropy)
                scale_data[point.scale]['participation_ratios'].append(point.participation_ratio)
        
        # Calculate ensemble averages
        ensemble_flow = []
        for scale in sorted(scale_data.keys()):
            data = scale_data[scale]
            ensemble_flow.append({
                'scale': scale,
                'q_mean': float(np.mean(data['q_means'])),
                'q_mean_std': float(np.std(data['q_means'])),
                'q_variance': float(np.mean(data['q_variances'])),
                'q_variance_std': float(np.std(data['q_variances'])),
                'entropy': float(np.mean(data['entropies'])),
                'entropy_std': float(np.std(data['entropies'])),
                'participation_ratio': float(np.mean(data['participation_ratios'])),
                'n_samples': len(data['q_means'])
            })
        
        # Analyze fixed points
        all_fixed_points = []
        for traj in trajectories:
            if traj.fixed_points:
                all_fixed_points.extend(traj.fixed_points)
        
        # Cluster fixed points
        fixed_point_clusters = self._cluster_fixed_points(all_fixed_points)
        
        # Calculate flow universality (how similar are trajectories?)
        flow_similarity = self._calculate_flow_similarity(trajectories)
        
        return {
            'ensemble_flow': ensemble_flow,
            'fixed_point_clusters': fixed_point_clusters,
            'flow_similarity': flow_similarity,
            'n_trajectories': len(trajectories),
            'universality_measure': self._calculate_universality(trajectories)
        }
    
    def _cluster_fixed_points(self, fixed_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster fixed points by location in parameter space"""
        if not fixed_points:
            return []
        
        # Simple clustering by scale and coupling
        clusters = []
        tolerance = 0.1  # Relative tolerance for clustering
        
        for fp in fixed_points:
            # Find matching cluster
            matched = False
            for cluster in clusters:
                if (abs(cluster['scale'] - fp['scale']) / (cluster['scale'] + 1e-10) < tolerance and
                    abs(cluster['coupling'] - fp['coupling']) / (cluster['coupling'] + 1e-10) < tolerance):
                    # Add to cluster
                    cluster['count'] += 1
                    cluster['scales'].append(fp['scale'])
                    cluster['couplings'].append(fp['coupling'])
                    if fp['type'] == 'stable':
                        cluster['stable_count'] += 1
                    matched = True
                    break
            
            if not matched:
                # Create new cluster
                clusters.append({
                    'scale': fp['scale'],
                    'coupling': fp['coupling'],
                    'count': 1,
                    'scales': [fp['scale']],
                    'couplings': [fp['coupling']],
                    'stable_count': 1 if fp['type'] == 'stable' else 0
                })
        
        # Finalize clusters
        for cluster in clusters:
            cluster['mean_scale'] = float(np.mean(cluster['scales']))
            cluster['mean_coupling'] = float(np.mean(cluster['couplings']))
            cluster['stability'] = cluster['stable_count'] / cluster['count']
            # Remove temporary lists
            del cluster['scales']
            del cluster['couplings']
        
        return sorted(clusters, key=lambda c: c['count'], reverse=True)
    
    def _calculate_flow_similarity(self, trajectories: List[RGTrajectory]) -> float:
        """Calculate similarity between flow trajectories"""
        if len(trajectories) < 2:
            return 1.0
        
        # Compare trajectories pairwise using correlation
        correlations = []
        
        for i in range(len(trajectories)):
            for j in range(i+1, len(trajectories)):
                traj1 = trajectories[i]
                traj2 = trajectories[j]
                
                # Match points by scale
                scales1 = [p.scale for p in traj1.flow_points]
                scales2 = [p.scale for p in traj2.flow_points]
                common_scales = sorted(set(scales1) & set(scales2))
                
                if len(common_scales) >= 2:
                    # Extract q_variances at common scales
                    vars1 = [p.q_variance for p in traj1.flow_points if p.scale in common_scales]
                    vars2 = [p.q_variance for p in traj2.flow_points if p.scale in common_scales]
                    
                    # Calculate correlation
                    if len(vars1) == len(vars2) and len(vars1) >= 2:
                        corr = np.corrcoef(vars1, vars2)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
        
        return float(np.mean(correlations)) if correlations else 0.0
    
    def _calculate_universality(self, trajectories: List[RGTrajectory]) -> Dict[str, float]:
        """Calculate universality measures for RG flow"""
        if not trajectories:
            return {}
        
        # Extract scaling exponents from each trajectory
        scaling_exponents = []
        
        for traj in trajectories:
            if len(traj.flow_points) >= 3:
                # Fit power law to q_variance vs scale
                scales = [p.scale for p in traj.flow_points]
                variances = [p.q_variance for p in traj.flow_points]
                
                if min(variances) > 0:  # Need positive values for log
                    try:
                        # Log-log regression
                        log_scales = np.log(scales)
                        log_vars = np.log(variances)
                        exponent = np.polyfit(log_scales, log_vars, 1)[0]
                        scaling_exponents.append(exponent)
                    except:
                        pass
        
        if scaling_exponents:
            return {
                'mean_scaling_exponent': float(np.mean(scaling_exponents)),
                'std_scaling_exponent': float(np.std(scaling_exponents)),
                'n_valid_trajectories': len(scaling_exponents)
            }
        else:
            return {
                'mean_scaling_exponent': np.nan,
                'std_scaling_exponent': np.nan,
                'n_valid_trajectories': 0
            }