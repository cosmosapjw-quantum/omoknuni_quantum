#!/usr/bin/env python3
"""
Markovian approximation validation and analytical justification for MCTS.

This module provides tools to:
1. Test the validity of the Markovian approximation in MCTS
2. Verify analytical predictions about correlation times and scaling
3. Validate the mean-field approach used in the theoretical framework
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import curve_fit
import logging

logger = logging.getLogger(__name__)


class MarkovianValidator:
    """Main class for validating Markovian approximation in MCTS"""
    
    def __init__(self):
        self.autocorr_analyzer = AutocorrelationAnalyzer()
        self.markov_tester = MarkovPropertyTester()
        self.predictions = AnalyticalPredictions()
    
    def validate(self, game_data: List[Dict]) -> Dict:
        """
        Run complete validation pipeline on MCTS game data.
        
        Args:
            game_data: List of game dictionaries with 'values', 'visit_counts', 'q_values'
            
        Returns:
            Dictionary with validation results
        """
        # Run validation components
        autocorr_results = self._validate_autocorrelation(game_data)
        markov_results = self._validate_markov_property(game_data)
        theory_results = self._compare_with_theory(game_data)
        
        # Merge results, preserving raw data for visualization
        results = {
            'autocorrelation': autocorr_results,
            'markov_test': markov_results,
            'analytical_comparison': theory_results
        }
        
        # Add raw data at top level for easier access in visualizations
        if 'raw_data' in autocorr_results:
            results['raw_data'] = autocorr_results['raw_data']
        
        # Add timescales at top level
        if 'timescales' in theory_results:
            results['timescales'] = theory_results['timescales']
        
        return results
    
    def _validate_autocorrelation(self, game_data: List[Dict]) -> Dict:
        """Validate autocorrelation predictions"""
        # Extract value sequences
        all_values = [game['values'] for game in game_data]
        
        # Compute autocorrelation
        c1_mean, c1_lower, c1_upper = self.autocorr_analyzer.bootstrap_correlation(
            all_values, lag=1
        )
        
        # Fit exponential decay
        correlations = []
        all_tau_c = []
        
        for values in all_values:
            corr_func = self.autocorr_analyzer.compute_autocorrelation_function(
                values, max_lag=10
            )
            correlations.append(corr_func)
            
            # Fit each game individually for distribution
            tau_c_single, _ = self.autocorr_analyzer.fit_exponential_decay(corr_func)
            all_tau_c.append(tau_c_single)
        
        mean_corr = np.mean(correlations, axis=0)
        tau_c, r_squared = self.autocorr_analyzer.fit_exponential_decay(mean_corr)
        
        return {
            'c1': c1_mean,
            'c1_ci': (c1_lower, c1_upper),
            'tau_c': tau_c,
            'fit_quality': r_squared,
            'correlation_function': mean_corr,
            'lags': np.arange(len(mean_corr)),
            'raw_data': {
                'all_correlations': correlations,
                'all_tau_c': np.array(all_tau_c),
                'game_lengths': np.array([len(game['values']) for game in game_data])
            }
        }
    
    def _validate_markov_property(self, game_data: List[Dict]) -> Dict:
        """Test Markov property directly"""
        # Extract state trajectories
        trajectories = []
        for game in game_data:
            n = len(game['values'])
            states = np.column_stack([
                game['visit_counts'],
                game['q_values'],
                self._compute_variance(game['values'], game['q_values'])
            ])
            trajectories.append(states)
        
        # Test Markov property
        results = self.markov_tester.test_markov_property(trajectories, max_order=3)
        
        # Get transition matrices for visualization
        discrete_trajectories = [self.markov_tester.discretize_states(traj, n_bins=10) 
                               for traj in trajectories]
        trans_matrix_1 = self.markov_tester.compute_transition_matrix(discrete_trajectories, order=1)
        
        return {
            'js_divergences': results['js_divergences'],
            'js_divergence_order2': results['js_divergences'][2],
            'js_divergence_order3': results['js_divergences'][3],
            'markovian': results['js_divergences'][2] < 0.01,
            'transition_matrices': {
                1: trans_matrix_1
            }
        }
    
    def _compare_with_theory(self, game_data: List[Dict]) -> Dict:
        """Compare measured values with analytical predictions"""
        # Average visit counts
        avg_n = np.mean([np.mean(game['visit_counts']) for game in game_data])
        
        # Measure C(1)
        all_values = [game['values'] for game in game_data]
        c1_measured, _, _ = self.autocorr_analyzer.bootstrap_correlation(all_values, lag=1)
        
        # Get typical variance of values
        sigma_values = np.mean([np.std(game['values']) for game in game_data])
        
        # Predict C(1) - using absolute value to handle both positive and negative correlations
        c1_predicted = self.predictions.predict_c1(
            n_visits=avg_n,
            beta=1.0,  # Typical value
            sigma_q=sigma_values
        )
        
        # Compute ratio using absolute values
        c1_ratio = abs(c1_measured) / c1_predicted if c1_predicted > 0 else np.inf
        
        # Generate scaling predictions for visualization
        n_visits_range = np.logspace(1, 3, 50)
        c1_scaling = self.predictions.predict_c1(
            n_visits=n_visits_range,
            beta=1.0,
            sigma_q=sigma_values
        )
        
        # Compute timescales
        tau_env, tau_sys = self.predictions.compute_timescales(avg_n)
        
        return {
            'c1_measured': c1_measured,
            'c1_predicted': c1_predicted,
            'c1_ratio': c1_ratio,
            'avg_n': avg_n,
            'n_visits_range': n_visits_range,
            'c1_scaling': c1_scaling,
            'timescales': {
                'tau_env': tau_env,
                'tau_sys': tau_sys,
                'separation_ratio': tau_sys / tau_env
            }
        }
    
    def _compute_variance(self, values: np.ndarray, q_values: np.ndarray) -> np.ndarray:
        """Compute running variance of values"""
        n = len(values)
        variance = np.zeros(n)
        
        for i in range(n):
            if i == 0:
                variance[i] = 0.25  # Initial uncertainty
            else:
                # Running variance
                variance[i] = np.var(values[:i+1])
        
        return variance


class AutocorrelationAnalyzer:
    """Analyzer for temporal correlations in MCTS value sequences"""
    
    def compute_autocorrelation(self, values: np.ndarray, lag: int) -> float:
        """
        Compute autocorrelation at specific lag.
        
        Args:
            values: Sequence of values
            lag: Time lag
            
        Returns:
            Correlation coefficient
        """
        if lag >= len(values):
            return 0.0
        
        # Remove mean
        values_centered = values - np.mean(values)
        
        # Compute correlation
        if lag == 0:
            return 1.0
        
        n = len(values) - lag
        numerator = np.sum(values_centered[:-lag] * values_centered[lag:]) / n
        denominator = np.sum(values_centered ** 2) / len(values)
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def compute_autocorrelation_function(self, values: np.ndarray, max_lag: int) -> np.ndarray:
        """
        Compute autocorrelation function for multiple lags.
        
        Args:
            values: Sequence of values
            max_lag: Maximum lag to compute
            
        Returns:
            Array of correlation coefficients
        """
        correlations = []
        for lag in range(max_lag + 1):
            correlations.append(self.compute_autocorrelation(values, lag))
        
        return np.array(correlations)
    
    def fit_exponential_decay(self, correlations: np.ndarray) -> Tuple[float, float]:
        """
        Fit exponential decay to correlation function.
        
        Args:
            correlations: Autocorrelation function values
            
        Returns:
            (tau_c, r_squared) - decay time and fit quality
        """
        # Exclude C(0) = 1 from fit
        lags = np.arange(1, len(correlations))
        corr_positive = correlations[1:]
        
        # Only fit positive correlations
        mask = corr_positive > 0
        if np.sum(mask) < 2:
            return 0.0, 0.0
        
        lags_fit = lags[mask]
        corr_fit = corr_positive[mask]
        
        try:
            # Fit exponential: C(tau) = exp(-tau/tau_c)
            def exp_func(tau, tau_c):
                return np.exp(-tau / tau_c)
            
            # Better initial guess and bounds
            popt, _ = curve_fit(exp_func, lags_fit, corr_fit, 
                              p0=[2.0], bounds=(0.1, 100.0))
            tau_c = popt[0]
            
            # Compute R²
            residuals = corr_fit - exp_func(lags_fit, tau_c)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((corr_fit - np.mean(corr_fit))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            return tau_c, r_squared
            
        except Exception as e:
            # If exponential fit fails, estimate from first few points
            # Find where correlation drops to 1/e
            for i, c in enumerate(corr_positive):
                if c < 1/np.e:
                    return float(i), 0.0
            return float(len(corr_positive)), 0.0
    
    def bootstrap_correlation(self, all_values: List[np.ndarray], lag: int = 1,
                            n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence intervals for correlation.
        
        Args:
            all_values: List of value sequences from multiple runs
            lag: Time lag
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            
        Returns:
            (mean, lower_bound, upper_bound)
        """
        # Compute correlation for each run
        correlations = []
        for values in all_values:
            c = self.compute_autocorrelation(values, lag)
            correlations.append(c)
        
        correlations = np.array(correlations)
        
        # Bootstrap
        bootstrap_means = []
        n_runs = len(correlations)
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n_runs, n_runs, replace=True)
            bootstrap_sample = correlations[indices]
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Compute confidence intervals
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        mean = np.mean(correlations)
        
        return mean, lower, upper


class MarkovPropertyTester:
    """Direct test of Markov property in MCTS state sequences"""
    
    def discretize_states(self, states: np.ndarray, n_bins: int = 10) -> np.ndarray:
        """
        Discretize continuous states into bins.
        
        Args:
            states: Array of continuous states (n_samples, n_features)
            n_bins: Number of bins per feature
            
        Returns:
            Array of discrete state indices
        """
        n_samples, n_features = states.shape
        
        # Compute bin edges for each feature using percentiles
        bin_edges_list = []
        for i in range(n_features):
            # Use percentile-based binning for better distribution
            percentiles = np.linspace(0, 100, n_bins + 1)
            bin_edges = np.percentile(states[:, i], percentiles)
            # Ensure unique edges
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2:
                bin_edges = np.array([states[:, i].min(), states[:, i].max()])
            bin_edges_list.append(bin_edges)
        
        # Discretize each sample
        bin_indices = np.zeros(n_samples, dtype=np.int32)
        
        for i in range(n_samples):
            index = 0
            multiplier = 1
            
            for j in range(n_features):
                # Find bin for this feature
                bin_idx = np.searchsorted(bin_edges_list[j][1:-1], states[i, j])
                index += bin_idx * multiplier
                multiplier *= len(bin_edges_list[j]) - 1
        
        return bin_indices
    
    def compute_transition_matrix(self, sequences: List[np.ndarray], order: int = 1) -> np.ndarray:
        """
        Compute empirical transition probabilities.
        
        Args:
            sequences: List of state sequences
            order: Markov order (1 = first-order, 2 = second-order, etc.)
            
        Returns:
            Transition probability matrix
        """
        # Collect all transitions
        transitions = []
        
        for seq in sequences:
            for i in range(order, len(seq)):
                if order == 1:
                    history = seq[i-1]
                else:
                    history = tuple(seq[i-order:i])
                next_state = seq[i]
                transitions.append((history, next_state))
        
        # Count transitions
        from collections import defaultdict
        counts = defaultdict(lambda: defaultdict(int))
        
        for history, next_state in transitions:
            counts[history][next_state] += 1
        
        # Get all unique states
        all_states = sorted(set(s for seq in sequences for s in seq))
        n_states = len(all_states)
        
        # Create all possible histories for higher orders
        if order == 1:
            all_histories = all_states
            history_to_idx = {s: i for i, s in enumerate(all_states)}
        else:
            # Generate all possible tuples of states
            import itertools
            all_histories = list(itertools.product(all_states, repeat=order))
            history_to_idx = {h: i for i, h in enumerate(all_histories)}
        
        state_to_idx = {s: i for i, s in enumerate(all_states)}
        
        # Build matrix with shape based on all possible histories
        matrix = np.zeros((len(all_histories), n_states))
        
        # Fill in observed transitions
        for history, next_counts in counts.items():
            if history in history_to_idx:
                row_idx = history_to_idx[history]
                total = sum(next_counts.values())
                
                for next_state, count in next_counts.items():
                    if next_state in state_to_idx:
                        col_idx = state_to_idx[next_state]
                        matrix[row_idx, col_idx] = count / total
        
        # For unobserved histories, use uniform distribution
        for i in range(matrix.shape[0]):
            if matrix[i].sum() == 0:
                matrix[i] = 1.0 / n_states
        
        return matrix
    
    def jensen_shannon_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute Jensen-Shannon divergence between two distributions.
        
        Args:
            p, q: Probability distributions
            
        Returns:
            JS divergence (between 0 and 1)
        """
        # Ensure proper normalization
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Average distribution
        m = 0.5 * (p + q)
        
        # KL divergences
        def kl_div(a, b):
            mask = a > 0
            if not np.any(mask):
                return 0.0
            return np.sum(a[mask] * np.log(a[mask] / b[mask]))
        
        kl_pm = kl_div(p, m)
        kl_qm = kl_div(q, m)
        
        # JS divergence (using log base 2 for [0,1] range)
        js = 0.5 * (kl_pm + kl_qm) / np.log(2)
        
        return js
    
    def test_markov_property(self, trajectories: List[np.ndarray], max_order: int = 3) -> Dict:
        """
        Test Markov property up to specified order.
        
        Args:
            trajectories: List of state trajectories
            max_order: Maximum Markov order to test
            
        Returns:
            Dictionary with test results
        """
        # Discretize states
        discretized = []
        for traj in trajectories:
            discrete = self.discretize_states(traj, n_bins=10)
            discretized.append(discrete)
        
        # Compute transition matrices
        results = {'js_divergences': {}}
        
        # First-order reference
        p1 = self.compute_transition_matrix(discretized, order=1)
        
        # Higher orders
        for order in range(2, max_order + 1):
            p_higher = self.compute_transition_matrix(discretized, order=order)
            
            # Compare marginal distributions
            # For fair comparison, we need to marginalize the higher-order matrix
            # This is approximate - we compare average JS divergence
            
            js_sum = 0.0
            n_comparisons = 0
            
            # Sample some rows to compare
            n_samples = min(100, p_higher.shape[0])
            sample_indices = np.random.choice(p_higher.shape[0], n_samples, replace=False)
            
            for idx in sample_indices:
                if np.sum(p_higher[idx]) > 0 and np.sum(p1[idx % p1.shape[0]]) > 0:
                    js = self.jensen_shannon_divergence(
                        p_higher[idx],
                        p1[idx % p1.shape[0]]
                    )
                    js_sum += js
                    n_comparisons += 1
            
            avg_js = js_sum / n_comparisons if n_comparisons > 0 else 0.0
            results['js_divergences'][order] = avg_js
        
        return results


class AnalyticalPredictions:
    """Analytical predictions for MCTS correlations and timescales"""
    
    def predict_c1(self, n_visits: float, beta: float = 1.0, sigma_q: float = 0.2) -> float:
        """
        Predict one-step autocorrelation C(1).
        
        Based on: C(1) ~ beta * sigma_Q / N_typ
        
        Args:
            n_visits: Typical number of visits
            beta: Inverse temperature parameter
            sigma_q: Q-value spread
            
        Returns:
            Predicted C(1)
        """
        return beta * sigma_q / n_visits
    
    def predict_cm(self, m: int, n_visits: float) -> float:
        """
        Predict m-step autocorrelation C(m).
        
        Based on: C(m) ~ (1/N)^m
        
        Args:
            m: Time lag
            n_visits: Typical number of visits
            
        Returns:
            Predicted C(m)
        """
        if m == 0:
            return 1.0
        
        return (1.0 / n_visits) ** m
    
    def compute_timescales(self, n_visits: float) -> Tuple[float, float]:
        """
        Compute environment and system timescales.
        
        Args:
            n_visits: Number of visits
            
        Returns:
            (tau_env, tau_sys) - environment and system timescales
        """
        # Environment timescale - O(1)
        tau_env = 1.0
        
        # System timescale - O(N) for Q-value convergence
        tau_sys = n_visits
        
        return tau_env, tau_sys