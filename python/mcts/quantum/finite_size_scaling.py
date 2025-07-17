"""
Finite-size scaling analysis using MCTS subtrees.

This module implements the brilliant insight from the user that we can
use subtrees at different depths as proxies for different system sizes,
enabling finite-size scaling analysis without multiple board sizes.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from scipy.optimize import curve_fit
import logging

logger = logging.getLogger(__name__)


@dataclass
class SubtreeData:
    """Data extracted from a subtree at a specific depth"""
    depth: int
    size: int  # Number of nodes in subtree
    visits: np.ndarray
    q_values: np.ndarray
    temperature: float
    order_parameter: float
    susceptibility: float
    correlation_length: float
    energy: float
    entropy: float


class FiniteSizeScaling:
    """
    Performs finite-size scaling analysis using MCTS subtrees.
    
    Key insight: Subtrees at different depths act like systems of different
    sizes, allowing us to extract critical exponents without changing board size.
    """
    
    def __init__(self):
        self.subtree_data: List[SubtreeData] = []
        self.critical_temp: Optional[float] = None
        self.critical_exponents: Dict[str, float] = {}
        
    def extract_subtree_data(self, mcts_trajectory: List[Dict[str, Any]], 
                           max_depth: int = 10) -> List[SubtreeData]:
        """
        Extract subtree data at different depths from MCTS trajectory.
        
        Args:
            mcts_trajectory: List of MCTS snapshots
            max_depth: Maximum subtree depth to analyze
            
        Returns:
            List of SubtreeData for different depths
        """
        subtree_data = []
        
        for depth in range(2, max_depth + 1):
            # Extract nodes up to this depth
            subtree_visits = []
            subtree_q_values = []
            
            for snapshot in mcts_trajectory:
                # Check if snapshot has depth information
                if 'tree_depth' in snapshot and snapshot['tree_depth'] >= depth:
                    # Extract nodes at this depth level
                    if 'depth_wise_data' in snapshot:
                        depth_data = snapshot['depth_wise_data'].get(depth, {})
                        if 'visits' in depth_data and 'q_values' in depth_data:
                            subtree_visits.extend(depth_data['visits'])
                            subtree_q_values.extend(depth_data['q_values'])
            
            if len(subtree_visits) > 10:  # Need minimum data
                visits = np.array(subtree_visits)
                q_values = np.array(subtree_q_values)
                
                # Calculate thermodynamic quantities for this subtree
                temp = self._extract_temperature(visits, q_values)
                order_param = self._calculate_order_parameter(visits, q_values)
                susceptibility = self._calculate_susceptibility(visits, q_values, temp)
                corr_length = self._estimate_correlation_length(visits, depth)
                energy = self._calculate_energy(q_values, visits)
                entropy = self._calculate_entropy(visits)
                
                subtree = SubtreeData(
                    depth=depth,
                    size=len(visits),
                    visits=visits,
                    q_values=q_values,
                    temperature=temp,
                    order_parameter=order_param,
                    susceptibility=susceptibility,
                    correlation_length=corr_length,
                    energy=energy,
                    entropy=entropy
                )
                
                subtree_data.append(subtree)
                logger.info(f"Extracted subtree at depth {depth}: size={len(visits)}, T={temp:.3f}")
        
        self.subtree_data = subtree_data
        return subtree_data
    
    def _extract_temperature(self, visits: np.ndarray, q_values: np.ndarray) -> float:
        """Extract effective temperature from visit distribution"""
        if len(visits) < 2:
            return np.nan
            
        # Normalize visits
        probs = visits / visits.sum()
        
        # Remove zeros for log
        mask = probs > 1e-10
        if mask.sum() < 2:
            return np.nan
            
        probs = probs[mask]
        q_vals = q_values[mask]
        
        # Fit to Boltzmann distribution: p ~ exp(β * Q)
        try:
            log_probs = np.log(probs)
            # Linear regression in log space
            coeffs = np.polyfit(q_vals, log_probs, 1)
            beta = coeffs[0]
            temp = 1.0 / abs(beta) if abs(beta) > 0.01 else np.nan
            return temp
        except:
            return np.nan
    
    def _calculate_order_parameter(self, visits: np.ndarray, q_values: np.ndarray) -> float:
        """Calculate order parameter (e.g., magnetization analog)"""
        # Use Q-value asymmetry as order parameter
        probs = visits / visits.sum()
        mean_q = np.average(q_values, weights=probs)
        
        # Order parameter: weighted deviation from mean
        order = np.average(np.abs(q_values - mean_q), weights=probs)
        return order
    
    def _calculate_susceptibility(self, visits: np.ndarray, q_values: np.ndarray, 
                                temp: float) -> float:
        """Calculate susceptibility (response to perturbation)"""
        if np.isnan(temp) or temp <= 0:
            return np.nan
            
        probs = visits / visits.sum()
        mean_q = np.average(q_values, weights=probs)
        variance = np.average((q_values - mean_q)**2, weights=probs)
        
        # Susceptibility = variance / temperature
        susceptibility = variance / temp
        return susceptibility
    
    def _estimate_correlation_length(self, visits: np.ndarray, depth: int) -> float:
        """Estimate correlation length from visit distribution"""
        # Use participation ratio as proxy for correlation length
        probs = visits / visits.sum()
        participation_ratio = 1.0 / np.sum(probs**2)
        
        # Scale by subtree depth
        correlation_length = participation_ratio / depth
        return correlation_length
    
    def _calculate_energy(self, q_values: np.ndarray, visits: np.ndarray) -> float:
        """Calculate average energy (negative Q-value)"""
        probs = visits / visits.sum()
        energy = -np.average(q_values, weights=probs)
        return energy
    
    def _calculate_entropy(self, visits: np.ndarray) -> float:
        """Calculate entropy from visit distribution"""
        probs = visits / visits.sum()
        # Remove zeros
        probs = probs[probs > 1e-10]
        entropy = -np.sum(probs * np.log(probs))
        return entropy
    
    def perform_finite_size_scaling(self) -> Dict[str, Any]:
        """
        Perform finite-size scaling analysis on subtree data.
        
        Returns:
            Dictionary with critical exponents and scaling results
        """
        if len(self.subtree_data) < 3:
            logger.warning("Insufficient subtree data for scaling analysis")
            return {}
        
        # Extract arrays for analysis
        sizes = np.array([st.size for st in self.subtree_data])
        temps = np.array([st.temperature for st in self.subtree_data])
        order_params = np.array([st.order_parameter for st in self.subtree_data])
        susceptibilities = np.array([st.susceptibility for st in self.subtree_data])
        corr_lengths = np.array([st.correlation_length for st in self.subtree_data])
        
        # Remove invalid data
        valid_mask = ~(np.isnan(temps) | np.isnan(order_params) | np.isnan(susceptibilities))
        if valid_mask.sum() < 3:
            logger.warning("Too much invalid data in finite-size scaling")
            return {}
        
        sizes = sizes[valid_mask]
        temps = temps[valid_mask]
        order_params = order_params[valid_mask]
        susceptibilities = susceptibilities[valid_mask]
        corr_lengths = corr_lengths[valid_mask]
        
        results = {
            'subtree_sizes': sizes.tolist(),
            'temperatures': temps.tolist(),
            'order_parameters': order_params.tolist(),
            'susceptibilities': susceptibilities.tolist(),
            'correlation_lengths': corr_lengths.tolist()
        }
        
        # Fit scaling relations
        try:
            # Order parameter scaling: m ~ L^(-β/ν)
            log_sizes = np.log(sizes)
            
            # Filter out zero or negative order parameters
            valid_mask = order_params > 1e-10
            if np.sum(valid_mask) >= 2:  # Need at least 2 points for fitting
                log_order = np.log(order_params[valid_mask])
                beta_over_nu_fit = np.polyfit(log_sizes[valid_mask], log_order, 1)[0]
                self.critical_exponents['beta_over_nu'] = -beta_over_nu_fit
            else:
                self.critical_exponents['beta_over_nu'] = np.nan
            
            # Susceptibility scaling: χ ~ L^(γ/ν)
            valid_mask_chi = susceptibilities > 1e-10
            if np.sum(valid_mask_chi) >= 2:
                log_chi = np.log(susceptibilities[valid_mask_chi])
                gamma_over_nu_fit = np.polyfit(log_sizes[valid_mask_chi], log_chi, 1)[0]
                self.critical_exponents['gamma_over_nu'] = gamma_over_nu_fit
            else:
                self.critical_exponents['gamma_over_nu'] = np.nan
            
            # Correlation length scaling: ξ ~ L
            valid_mask_xi = corr_lengths > 1e-10
            if np.sum(valid_mask_xi) >= 2:
                log_xi = np.log(corr_lengths[valid_mask_xi])
                xi_scaling = np.polyfit(log_sizes[valid_mask_xi], log_xi, 1)[0]
            else:
                xi_scaling = 0.0
            
            # Estimate nu from correlation length scaling
            self.critical_exponents['nu'] = 1.0 / xi_scaling if xi_scaling != 0 else 1.0
            
            # Hyperscaling relations
            d_eff = 2  # Effective dimension (tree is approximately 2D)
            self.critical_exponents['alpha'] = 2 - d_eff * self.critical_exponents['nu']
            self.critical_exponents['beta'] = self.critical_exponents['beta_over_nu'] * self.critical_exponents['nu']
            self.critical_exponents['gamma'] = self.critical_exponents['gamma_over_nu'] * self.critical_exponents['nu']
            
            results['critical_exponents'] = self.critical_exponents
            
            # Estimate critical temperature from peak susceptibility
            if len(susceptibilities) > 0:
                max_idx = np.argmax(susceptibilities)
                self.critical_temp = temps[max_idx]
                results['critical_temperature'] = float(self.critical_temp)
            
            logger.info(f"Finite-size scaling complete: β/ν={self.critical_exponents['beta_over_nu']:.3f}, "
                       f"γ/ν={self.critical_exponents['gamma_over_nu']:.3f}")
            
        except Exception as e:
            logger.error(f"Error in finite-size scaling: {e}")
            results['error'] = str(e)
        
        return results
    
    def perform_scaling_collapse(self, observable: str = 'order_parameter') -> Dict[str, Any]:
        """
        Perform scaling collapse to verify critical exponents.
        
        Args:
            observable: Which observable to collapse ('order_parameter' or 'susceptibility')
            
        Returns:
            Dictionary with collapsed data and quality metrics
        """
        if not self.critical_temp or not self.critical_exponents:
            logger.warning("Need to run finite_size_scaling first")
            return {}
        
        collapse_data = []
        
        for subtree in self.subtree_data:
            if np.isnan(subtree.temperature):
                continue
                
            # Reduced temperature
            t = (subtree.temperature - self.critical_temp) / self.critical_temp
            
            # Scaling variables
            L = subtree.size
            nu = self.critical_exponents.get('nu', 1.0)
            
            if observable == 'order_parameter':
                beta_over_nu = self.critical_exponents.get('beta_over_nu', 0.125)
                x_scaled = t * L**(1/nu)
                y_scaled = subtree.order_parameter * L**(beta_over_nu)
            else:  # susceptibility
                gamma_over_nu = self.critical_exponents.get('gamma_over_nu', 1.75)
                x_scaled = t * L**(1/nu)
                y_scaled = subtree.susceptibility * L**(-gamma_over_nu)
            
            collapse_data.append({
                'size': L,
                'scaled_variable': float(x_scaled),
                'scaled_observable': float(y_scaled),
                'temperature': subtree.temperature
            })
        
        # Calculate collapse quality
        if len(collapse_data) >= 2:
            x_vals = [d['scaled_variable'] for d in collapse_data]
            y_vals = [d['scaled_observable'] for d in collapse_data]
            
            # Measure spread in y for similar x values
            x_bins = np.linspace(min(x_vals), max(x_vals), 10)
            bin_variance = []
            
            for i in range(len(x_bins)-1):
                mask = (np.array(x_vals) >= x_bins[i]) & (np.array(x_vals) < x_bins[i+1])
                if mask.sum() > 1:
                    bin_y = np.array(y_vals)[mask]
                    bin_variance.append(np.var(bin_y))
            
            quality = 1.0 / (1.0 + np.mean(bin_variance)) if bin_variance else 0.0
        else:
            quality = 0.0
        
        return {
            'collapse_data': collapse_data,
            'observable': observable,
            'quality': float(quality),
            'critical_temperature': float(self.critical_temp),
            'critical_exponents': self.critical_exponents
        }