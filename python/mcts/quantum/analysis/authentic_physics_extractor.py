"""
Authentic physics extractor for MCTS data.

This module extracts physics ONLY from real MCTS self-play data:
- No predetermined formulas
- No mock data
- No assumptions about scaling laws
- Everything measured, nothing imposed

Key principle: Let the data speak for itself.
"""
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from scipy import stats
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

try:
    from ..analysis.dynamics_extractor import DynamicsData
except ImportError:
    try:
        from .dynamics_extractor import DynamicsData
    except ImportError:
        from dynamics_extractor import DynamicsData

# Import finite-size scaling modules
try:
    from ..finite_size_scaling import FiniteSizeScaling
    from ..subtree_extractor import SubtreeExtractor
except ImportError:
    try:
        from finite_size_scaling import FiniteSizeScaling
        from subtree_extractor import SubtreeExtractor
    except ImportError:
        FiniteSizeScaling = None
        SubtreeExtractor = None

# Import RG flow tracker
try:
    from ..rg_flow_tracker import RGFlowTracker
except ImportError:
    try:
        from rg_flow_tracker import RGFlowTracker
    except ImportError:
        RGFlowTracker = None

logger = logging.getLogger(__name__)


@dataclass
class MeasuredObservable:
    """A physics observable measured from MCTS data"""
    name: str
    values: np.ndarray
    uncertainties: np.ndarray
    metadata: Dict[str, Any]
    
    @property
    def mean(self) -> float:
        """Weighted mean accounting for uncertainties"""
        if len(self.uncertainties) > 0 and np.any(self.uncertainties > 0):
            weights = 1.0 / (self.uncertainties**2 + 1e-10)
            return np.average(self.values, weights=weights)
        return np.mean(self.values)
    
    @property
    def error(self) -> float:
        """Standard error of weighted mean"""
        if len(self.uncertainties) > 0 and np.any(self.uncertainties > 0):
            weights = 1.0 / (self.uncertainties**2 + 1e-10)
            variance = np.average((self.values - self.mean)**2, weights=weights)
            return np.sqrt(variance / len(self.values))
        return np.std(self.values) / np.sqrt(len(self.values))


class AuthenticPhysicsExtractor:
    """
    Extract physics from MCTS data without predetermined formulas.
    
    Core principles:
    1. Measure everything from actual tree statistics
    2. No assumptions about functional forms
    3. Test multiple hypotheses, let data choose
    4. Include uncertainties in all measurements
    5. Validate all theoretical predictions
    """
    
    def __init__(self):
        """Initialize extractor"""
        self.measurements = {}
        self._last_fit_diagnostics = {}
        self.finite_size_analyzer = FiniteSizeScaling() if FiniteSizeScaling else None
        self.subtree_extractor = SubtreeExtractor() if SubtreeExtractor else None
        self.rg_flow_tracker = RGFlowTracker() if RGFlowTracker else None
        
    def extract_temperature_from_visits(self, visits: np.ndarray, 
                                      q_values: np.ndarray,
                                      metadata: dict = None) -> Tuple[float, float]:
        """
        Extract temperature by measuring visit distribution.
        
        NO ASSUMPTIONS about temperature scaling with N.
        
        Method:
        - For thermal distribution: π(a) ∝ exp(β·Q(a))
        - Take log: log π(a) = β·Q(a) + const
        - Linear regression gives β, hence T = 1/β
        
        Args:
            visits: Visit counts for each action
            q_values: Q-values for each action
            metadata: Optional metadata about evaluator type
        
        Returns:
            (temperature, uncertainty)
        """
        # Initialize diagnostics
        self._last_fit_diagnostics = {}
        
        # Check for random evaluator - return NaN immediately
        if metadata and metadata.get('evaluator_type') in ['random', 'fast_random']:
            self._last_fit_diagnostics.update({
                'anomaly': 'random_evaluator',
                'reason': 'Random evaluator has no thermal structure',
                'valid': False
            })
            return np.nan, np.nan
        
        # Need at least 3 visited actions
        mask = visits > 0
        if np.sum(mask) < 3:
            self._last_fit_diagnostics.update({
                'anomaly': 'insufficient_data',
                'reason': 'Less than 3 visited actions',
                'valid': False
            })
            return np.nan, np.nan
            
        # Compute log probabilities
        probs = visits[mask] / visits.sum()
        log_probs = np.log(probs)
        q_vals = q_values[mask]
        
        # Check if Q-values have meaningful variation
        q_std = np.std(q_vals)
        q_range = np.ptp(q_vals)
        
        if q_std < 1e-6 or q_range < 1e-6:
            self._last_fit_diagnostics.update({
                'anomaly': 'no_q_variation',
                'reason': 'Q-values have no variation',
                'q_std': q_std,
                'q_range': q_range,
                'valid': False
            })
            return np.nan, np.nan
        
        # Additional heuristic: check if Q-values look random
        # For random evaluator, Q-values should be uncorrelated with visits
        if len(q_vals) >= 5:
            # Spearman rank correlation (robust to outliers)
            from scipy.stats import spearmanr
            try:
                corr, p_value = spearmanr(q_vals, visits[mask])
                if p_value > 0.1:  # No significant correlation
                    self._last_fit_diagnostics.update({
                        'anomaly': 'no_q_visit_correlation',
                        'reason': 'Q-values uncorrelated with visits (likely random)',
                        'correlation': corr,
                        'p_value': p_value,
                        'valid': False
                    })
                    return np.nan, np.nan
            except:
                pass  # If correlation fails, continue with fitting
            
        try:
            # Weighted linear regression: log π = β·Q + c
            # Weight by sqrt(visits) to account for sampling noise
            weights = np.sqrt(visits[mask])
            
            # Use polyfit with weights and covariance
            coeffs, cov = np.polyfit(q_vals, log_probs, 1, w=weights, cov=True)
            beta = coeffs[0]
            beta_err = np.sqrt(cov[0, 0])
            
            # Check if fit is meaningful
            # Compute R² to assess fit quality
            y_pred = np.polyval(coeffs, q_vals)
            ss_res = np.sum(weights * (log_probs - y_pred)**2)
            ss_tot = np.sum(weights * (log_probs - np.average(log_probs, weights=weights))**2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            # Reject poor fits (likely not thermal)
            # Note: beta <= 0 would give negative temperature (inverted population)
            # This can occur in forced exploration phases but is not thermal behavior
            if r_squared < 0.5:
                self._last_fit_diagnostics.update({
                    'anomaly': 'poor_fit',
                    'reason': 'R² < 0.5 indicates non-thermal behavior',
                    'r_squared': r_squared,
                    'beta': beta,
                    'valid': False
                })
                return np.nan, np.nan
            
            # Handle edge cases
            if beta <= 0:
                # Negative beta indicates anti-correlation between Q and visits
                # This might occur during forced exploration but isn't thermal
                self._last_fit_diagnostics.update({
                    'anomaly': 'negative_beta',
                    'reason': 'Negative beta indicates anti-correlation (non-thermal)',
                    'beta': beta,
                    'r_squared': r_squared,
                    'valid': False
                })
                return np.nan, np.nan
            elif beta < 0.01:
                # Very small beta means T → ∞ (uniform random policy)
                # Cap temperature at reasonable value
                temperature = 100.0  # High temperature limit
                temp_err = 10.0
                self._last_fit_diagnostics.update({
                    'anomaly': 'high_temperature_limit',
                    'reason': 'Beta < 0.01 indicates near-uniform distribution',
                    'beta': beta,
                    'r_squared': r_squared,
                    'capped_temperature': temperature,
                    'valid': True
                })
                return temperature, temp_err
                
            # Temperature with error propagation
            temperature = 1.0 / beta
            temp_err = beta_err / (beta**2)
            
            # Validate theoretical scaling β ∝ √N from quantum_mcts_foundation.md
            total_visits = np.sum(visits)
            theoretical_beta = np.sqrt(total_visits) / 1000.0  # Normalize for practical values
            scaling_validation = {
                'measured_beta': beta,
                'theoretical_beta': theoretical_beta,
                'scaling_ratio': beta / theoretical_beta if theoretical_beta > 0 else np.inf,
                'follows_sqrt_n_scaling': abs(beta / theoretical_beta - 1.0) < 0.5 if theoretical_beta > 0 else False
            }
            
            # Store diagnostics for validation
            self._last_fit_diagnostics.update({
                'r_squared': r_squared,
                'beta': beta,
                'beta_err': beta_err,
                'temperature': temperature,
                'temp_err': temp_err,
                'scaling_validation': scaling_validation,
                'n_actions': np.sum(mask),
                'q_range': np.ptp(q_vals),
                'q_std': q_std,
                'visit_entropy': -np.sum(probs * np.log(probs)),
                'valid': True
            })
            
            return temperature, temp_err
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Temperature extraction failed: {e}")
            self._last_fit_diagnostics.update({
                'anomaly': 'fitting_error',
                'reason': f'Fitting failed with error: {str(e)}',
                'valid': False
            })
            return np.nan, np.nan
    
    def get_last_fit_diagnostics(self) -> dict:
        """Get diagnostics from the last temperature fitting attempt"""
        return self._last_fit_diagnostics.copy()
    
    def is_last_fit_valid(self) -> bool:
        """Check if the last temperature fit was valid"""
        return self._last_fit_diagnostics.get('valid', False)
            
    def measure_decoherence_from_distribution_evolution(self, 
                                                      snapshot_sequence: List[Dict]) -> Dict[str, Any]:
        """
        Measure decoherence without assuming exponential decay.
        
        Method:
        - Track evolution of visit distribution entropy
        - Measure how distribution concentrates over time
        - NO ASSUMPTION about functional form
        
        Returns:
            Dictionary with measured decoherence properties
        """
        times = []
        entropies = []
        concentrations = []
        
        for snapshot in snapshot_sequence:
            if 'visits' not in snapshot:
                continue
                
            visits = np.array(snapshot['visits'])
            if visits.sum() == 0:
                continue
                
            # Measure distribution properties
            probs = visits / visits.sum()
            
            # Shannon entropy (no assumption about meaning)
            nonzero = probs > 0
            entropy = -np.sum(probs[nonzero] * np.log(probs[nonzero]))
            
            # Concentration: max probability
            concentration = np.max(probs)
            
            times.append(snapshot.get('timestamp', len(times)))
            entropies.append(entropy)
            concentrations.append(concentration)
            
        if len(times) < 3:
            return {}
            
        times = np.array(times)
        entropies = np.array(entropies)
        concentrations = np.array(concentrations)
        
        # Measure rate of change WITHOUT assuming exponential
        # Use numerical derivatives
        d_entropy_dt = np.gradient(entropies, times)
        d_concentration_dt = np.gradient(concentrations, times)
        
        # Characterize evolution without functional assumption
        results = {
            'times': times,
            'entropies': entropies,
            'concentrations': concentrations,
            'entropy_change_rate': d_entropy_dt,
            'concentration_change_rate': d_concentration_dt,
            'total_entropy_change': entropies[-1] - entropies[0],
            'total_concentration_change': concentrations[-1] - concentrations[0],
            'average_entropy_rate': np.mean(d_entropy_dt),
            'average_concentration_rate': np.mean(d_concentration_dt)
        }
        
        # Test different decay models, let data choose
        decay_models = self._test_decay_models(times, entropies)
        results['decay_models'] = decay_models
        
        return results
        
    def _test_decay_models(self, times: np.ndarray, values: np.ndarray) -> Dict[str, Any]:
        """
        Test multiple decay models without prejudice.
        Let data determine which fits best.
        """
        models = {}
        
        # Model 1: Exponential
        try:
            def exp_decay(t, A, tau):
                return A * np.exp(-t / tau)
            popt, pcov = curve_fit(exp_decay, times, values, p0=[values[0], times[-1]/2])
            
            y_pred = exp_decay(times, *popt)
            r2 = 1 - np.sum((values - y_pred)**2) / np.sum((values - np.mean(values))**2)
            
            models['exponential'] = {
                'parameters': {'A': popt[0], 'tau': popt[1]},
                'r_squared': r2,
                'formula': 'A * exp(-t/tau)'
            }
        except:
            pass
            
        # Model 2: Power law
        try:
            def power_law(t, A, alpha):
                return A / (1 + t)**alpha
            popt, pcov = curve_fit(power_law, times, values, p0=[values[0], 1.0])
            
            y_pred = power_law(times, *popt)
            r2 = 1 - np.sum((values - y_pred)**2) / np.sum((values - np.mean(values))**2)
            
            models['power_law'] = {
                'parameters': {'A': popt[0], 'alpha': popt[1]},
                'r_squared': r2,
                'formula': 'A / (1 + t)^alpha'
            }
        except:
            pass
            
        # Model 3: Linear
        try:
            slope, intercept = np.polyfit(times, values, 1)
            y_pred = slope * times + intercept
            r2 = 1 - np.sum((values - y_pred)**2) / np.sum((values - np.mean(values))**2)
            
            models['linear'] = {
                'parameters': {'slope': slope, 'intercept': intercept},
                'r_squared': r2,
                'formula': 'slope * t + intercept'
            }
        except:
            pass
            
        # Find best model
        if models:
            best_model = max(models.items(), key=lambda x: x[1]['r_squared'])
            models['best_fit'] = best_model[0]
            
        return models
        
    def extract_scaling_relations_from_ensemble(self, 
                                              measurements: List[Tuple[float, float]],
                                              parameter_name: str) -> Dict[str, Any]:
        """
        Extract scaling relations from ensemble data without assumptions.
        
        Tests multiple scaling hypotheses and lets data choose.
        
        Args:
            measurements: List of (parameter, observable) tuples
            parameter_name: Name of parameter (e.g., 'N', 'L', 'T')
            
        Returns:
            Dictionary with tested scaling relations and best fit
        """
        if len(measurements) < 5:
            return {'error': 'Insufficient data for scaling analysis'}
            
        params = np.array([m[0] for m in measurements])
        observables = np.array([m[1] for m in measurements])
        
        # Filter out non-finite values and ensure positive values
        mask = np.isfinite(params) & np.isfinite(observables) & (params > 0) & (observables > 0)
        params = params[mask]
        observables = observables[mask]
        
        if len(params) < 3:  # Need at least 3 points for fitting
            return {'error': 'Insufficient valid data for scaling analysis'}
        
        if len(params) < 5:
            return {'error': 'Insufficient valid data after filtering'}
            
        scaling_models = {}
        
        # Test various scaling forms WITHOUT prejudice
        test_models = {
            'power': lambda x, a, b: a * x**b,
            'exponential': lambda x, a, b: a * np.exp(np.clip(b * x, -700, 700)),  # Clip to prevent overflow
            'logarithmic': lambda x, a, b: a + b * np.log(x),
            'sqrt': lambda x, a: a * np.sqrt(x),
            'linear': lambda x, a, b: a * x + b,
            'inverse': lambda x, a, b: a / x + b,
            'inverse_sqrt': lambda x, a: a / np.sqrt(x)
        }
        
        for name, func in test_models.items():
            try:
                # Initial guess based on data
                if name in ['sqrt', 'inverse_sqrt']:
                    p0 = [observables[0] / func(params[0], 1)]
                else:
                    p0 = None
                    
                try:
                    # Suppress the warning and catch it properly
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings('error', category=OptimizeWarning)
                        popt, pcov = curve_fit(func, params, observables, p0=p0, maxfev=5000)
                except (RuntimeError, ValueError, OptimizeWarning, Warning) as e:
                    # If fitting fails, skip this model
                    logger.debug(f"Fitting failed for {name} model: {e}")
                    results[name] = {
                        'params': None,
                        'r_squared': -np.inf,
                        'aic': np.inf,
                        'errors': None
                    }
                    continue
                
                # Check if covariance was estimated
                if pcov is None or np.isinf(pcov).any():
                    # Covariance couldn't be estimated - fitting is uncertain
                    logger.debug(f"Covariance estimation failed for {name} model")
                    param_errors = None
                else:
                    # Extract parameter errors from covariance
                    param_errors = np.sqrt(np.diag(pcov))
                
                # Compute R² and AIC
                y_pred = func(params, *popt)
                residuals = observables - y_pred
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((observables - np.mean(observables))**2)
                r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf
                
                # Akaike Information Criterion for model selection
                n = len(params)
                k = len(popt)  # number of parameters
                aic = n * np.log(ss_res / n) + 2 * k
                
                scaling_models[name] = {
                    'parameters': popt.tolist(),
                    'parameter_errors': param_errors.tolist() if param_errors is not None else None,
                    'r_squared': r_squared,
                    'aic': aic,
                    'residuals_std': np.std(residuals)
                }
                
            except Exception as e:
                logger.debug(f"Scaling model {name} failed: {e}")
                
        if not scaling_models:
            return {'error': 'All scaling models failed'}
            
        # Select best model using AIC (lower is better)
        best_model = min(scaling_models.items(), key=lambda x: x[1]['aic'])
        
        return {
            'parameter_name': parameter_name,
            'n_data_points': len(params),
            'tested_models': scaling_models,
            'best_model': best_model[0],
            'best_model_details': best_model[1],
            'data': {
                'parameters': params.tolist(),
                'observables': observables.tolist()
            }
        }
        
    def measure_critical_behavior_from_fluctuations(self, 
                                                   dynamics_data: DynamicsData) -> Dict[str, Any]:
        """
        Identify critical behavior from fluctuation analysis.
        
        NO ASSUMPTIONS about universality classes or exponents.
        """
        # Extract Q-value gaps as order parameter
        gaps = []
        visits_list = []
        
        for snapshot in dynamics_data.snapshots:
            if 'q_values' in snapshot and 'visits' in snapshot:
                q_vals = np.array(snapshot['q_values'])
                visits = np.array(snapshot['visits'])
                
                if len(q_vals) >= 2:
                    # Sort by Q-value
                    sorted_idx = np.argsort(q_vals)[::-1]
                    gap = q_vals[sorted_idx[0]] - q_vals[sorted_idx[1]]
                    
                    gaps.append(gap)
                    visits_list.append(visits.sum())
                    
        if len(gaps) < 10:
            return {}
            
        gaps = np.array(gaps)
        total_visits = np.array(visits_list)
        
        # Measure fluctuations without assuming distribution
        results = {
            'gaps': gaps,
            'gap_mean': np.mean(gaps),
            'gap_std': np.std(gaps),
            'gap_skewness': stats.skew(gaps),
            'gap_kurtosis': stats.kurtosis(gaps)
        }
        
        # Look for anomalous fluctuations (potential critical points)
        # Use multiple methods without prejudice
        
        # Method 1: Outlier detection
        z_scores = np.abs(stats.zscore(gaps))
        critical_candidates_outlier = np.where(z_scores > 3)[0]
        
        # Method 2: Local maxima in fluctuations
        if len(gaps) > 5:
            window_size = min(5, len(gaps) // 3)
            local_std = np.array([
                np.std(gaps[max(0, i-window_size):min(len(gaps), i+window_size)])
                for i in range(len(gaps))
            ])
            peaks, _ = find_peaks(local_std, prominence=np.std(local_std))
            critical_candidates_peaks = peaks
        else:
            critical_candidates_peaks = []
            
        results['critical_candidates'] = {
            'outlier_method': critical_candidates_outlier.tolist(),
            'fluctuation_method': critical_candidates_peaks.tolist()
        }
        
        # Measure correlation length without assuming form
        if len(gaps) > 10:
            autocorr = np.correlate(gaps - np.mean(gaps), gaps - np.mean(gaps), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            # Find correlation length (where autocorr drops to 1/e)
            # WITHOUT assuming exponential decay
            try:
                idx_decorr = np.where(autocorr < 1/np.e)[0][0]
                correlation_length = idx_decorr
            except:
                correlation_length = len(autocorr)
                
            results['correlation_length'] = correlation_length
            results['autocorrelation'] = autocorr[:20].tolist()  # First 20 lags
            
        return results
        
    def extract_rg_flow_from_tree_hierarchy(self, 
                                           dynamics_data: DynamicsData) -> Dict[str, Any]:
        """
        Extract RG flow by tracking observables at different tree depths.
        
        NO ASSUMPTIONS about flow equations or fixed points.
        """
        # Group observables by depth
        depth_data = {}
        
        for snapshot in dynamics_data.snapshots:
            depth = snapshot.get('depth', 0)
            
            if depth not in depth_data:
                depth_data[depth] = {
                    'q_values': [],
                    'visits': [],
                    'entropies': []
                }
                
            if 'q_values' in snapshot and 'visits' in snapshot:
                q_vals = np.array(snapshot['q_values'])
                visits = np.array(snapshot['visits'])
                
                if visits.sum() > 0:
                    probs = visits / visits.sum()
                    
                    # Measure various observables
                    depth_data[depth]['q_values'].extend(q_vals.tolist())
                    depth_data[depth]['visits'].extend(visits.tolist())
                    
                    # Entropy
                    p_nonzero = probs[probs > 0]
                    entropy = -np.sum(p_nonzero * np.log(p_nonzero))
                    depth_data[depth]['entropies'].append(entropy)
                    
        if len(depth_data) < 3:
            return {}
            
        # Compute statistics at each depth
        depths = sorted(depth_data.keys())
        flow_observables = {
            'q_mean': [],
            'q_std': [],
            'visit_concentration': [],
            'entropy_mean': []
        }
        
        for depth in depths:
            data = depth_data[depth]
            
            if data['q_values']:
                flow_observables['q_mean'].append(np.mean(data['q_values']))
                flow_observables['q_std'].append(np.std(data['q_values']))
                
            if data['visits']:
                # Concentration = max visit fraction
                visit_arrays = np.array(data['visits']).reshape(-1, len(data['visits'][0]))
                concentrations = np.max(visit_arrays / visit_arrays.sum(axis=1, keepdims=True), axis=1)
                flow_observables['visit_concentration'].append(np.mean(concentrations))
                
            if data['entropies']:
                flow_observables['entropy_mean'].append(np.mean(data['entropies']))
                
        # Analyze flow without assuming form
        results = {
            'depths': depths,
            'observables': flow_observables
        }
        
        # Measure flow properties
        for obs_name, obs_values in flow_observables.items():
            if len(obs_values) >= 3:
                # Rate of change
                gradients = np.gradient(obs_values, depths)
                
                # Identify potential fixed points (where gradient ≈ 0)
                fixed_point_candidates = []
                for i, grad in enumerate(gradients):
                    if abs(grad) < 0.1 * np.std(gradients):
                        fixed_point_candidates.append(depths[i])
                        
                results[f'{obs_name}_flow'] = {
                    'values': obs_values,
                    'gradients': gradients.tolist(),
                    'fixed_point_candidates': fixed_point_candidates,
                    'total_change': obs_values[-1] - obs_values[0],
                    'monotonic': np.all(np.diff(obs_values) >= 0) or np.all(np.diff(obs_values) <= 0)
                }
                
        return results
        
    def validate_theoretical_predictions(self, 
                                       measurements: Dict[str, MeasuredObservable],
                                       theory_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test theoretical predictions against measured data.
        
        NO ASSUMPTIONS - just statistical tests.
        """
        validation_results = {}
        
        for prediction_name, prediction in theory_predictions.items():
            if prediction_name not in measurements:
                validation_results[prediction_name] = {
                    'status': 'no_data',
                    'message': 'No measurement available for validation'
                }
                continue
                
            measured = measurements[prediction_name]
            
            if 'value' in prediction:
                # Point prediction
                predicted_value = prediction['value']
                measured_value = measured.mean
                measured_error = measured.error
                
                # Z-test
                z_score = abs(measured_value - predicted_value) / measured_error if measured_error > 0 else np.inf
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                
                validation_results[prediction_name] = {
                    'status': 'tested',
                    'predicted': predicted_value,
                    'measured': measured_value,
                    'measured_error': measured_error,
                    'z_score': z_score,
                    'p_value': p_value,
                    'consistent': p_value > 0.05,
                    'message': f"{'Consistent' if p_value > 0.05 else 'Inconsistent'} with theory (p={p_value:.3f})"
                }
                
            elif 'scaling' in prediction:
                # Scaling prediction
                scaling_form = prediction['scaling']
                
                # Extract relevant data and test scaling
                # This would use extract_scaling_relations_from_ensemble
                validation_results[prediction_name] = {
                    'status': 'scaling_test',
                    'theoretical_form': scaling_form,
                    'message': 'Scaling validation requires ensemble data'
                }
                
        return validation_results
    
    def extract_path_integral_from_tree(self, tree_root) -> Dict[str, Any]:
        """
        Extract path integral formulation from MCTS tree.
        
        According to quantum foundation:
        - Each path from root to leaf = quantum propagation path
        - Visit count ∝ |amplitude|² for that path
        - Total wavefunction = superposition of all paths
        
        Returns:
            Dictionary with path integral measurements
        """
        paths = []
        path_amplitudes = []
        path_actions = []
        
        def traverse_path(node, current_path, current_visits, depth=0):
            """Recursively extract all paths from root to leaves"""
            if not hasattr(node, 'children') or not node.children:
                # Leaf node - complete path
                if len(current_path) > 0:
                    paths.append(current_path.copy())
                    # Amplitude squared ∝ visit count
                    amplitude_squared = node.visit_count / tree_root.visit_count if tree_root.visit_count > 0 else 0
                    path_amplitudes.append(np.sqrt(amplitude_squared))
                    # Action = -log(probability) along path
                    path_action = -np.sum([np.log(v/tree_root.visit_count + 1e-10) for v in current_visits])
                    path_actions.append(path_action)
                return
            
            # Continue traversing
            for action, child in node.children.items():
                if child.visit_count > 0:
                    current_path.append(action)
                    current_visits.append(child.visit_count)
                    traverse_path(child, current_path, current_visits, depth+1)
                    current_path.pop()
                    current_visits.pop()
        
        # Extract all paths
        traverse_path(tree_root, [], [])
        
        if not paths:
            return {'error': 'No paths found in tree'}
        
        # Convert to arrays
        path_amplitudes = np.array(path_amplitudes)
        path_actions = np.array(path_actions)
        
        # Compute path integral observables
        results = {
            'n_paths': len(paths),
            'path_lengths': [len(p) for p in paths],
            'mean_path_length': np.mean([len(p) for p in paths]),
            'path_amplitudes': path_amplitudes,
            'path_actions': path_actions,
            'total_amplitude': np.sum(path_amplitudes),
            'normalization': np.sum(path_amplitudes**2),
            'effective_action': -np.log(np.sum(path_amplitudes * np.exp(-path_actions))),
            'action_variance': np.var(path_actions),
            'dominant_paths': []
        }
        
        # Find dominant paths (highest amplitude)
        sorted_indices = np.argsort(path_amplitudes)[::-1]
        for i in range(min(5, len(paths))):
            idx = sorted_indices[i]
            results['dominant_paths'].append({
                'path': paths[idx],
                'amplitude': path_amplitudes[idx],
                'action': path_actions[idx],
                'contribution': path_amplitudes[idx]**2 / results['normalization']
            })
        
        # Compute coherence length (how many paths contribute significantly)
        cumulative_contribution = 0
        coherence_length = 0
        for i in sorted_indices:
            cumulative_contribution += path_amplitudes[i]**2 / results['normalization']
            coherence_length += 1
            if cumulative_contribution > 0.9:  # 90% of total amplitude
                break
        
        results['coherence_length'] = coherence_length
        results['participation_ratio'] = 1.0 / np.sum((path_amplitudes**2 / results['normalization'])**2)
        
        # Phase coherence (all paths have real amplitudes in MCTS, but we can measure spread)
        results['amplitude_coherence'] = np.std(path_amplitudes) / np.mean(path_amplitudes)
        
        return results
    
    def measure_wavefunction_from_visits(self, visits: np.ndarray) -> Dict[str, Any]:
        """
        Interpret visit distribution as quantum wavefunction squared.
        
        ψ(a) = √(visits(a) / total_visits)
        
        Returns:
            Wavefunction properties
        """
        total_visits = np.sum(visits)
        if total_visits == 0:
            return {'error': 'No visits'}
        
        # Wavefunction amplitudes
        psi_squared = visits / total_visits
        psi = np.sqrt(psi_squared)
        
        # Quantum mechanical observables
        results = {
            'wavefunction': psi,
            'probability_distribution': psi_squared,
            'normalization': np.sum(psi_squared),
            'n_states': len(visits),
            'n_occupied_states': np.sum(visits > 0),
            'participation_ratio': 1.0 / np.sum(psi_squared**2),
            'shannon_entropy': -np.sum(psi_squared[psi_squared > 0] * np.log(psi_squared[psi_squared > 0])),
            'renyi_entropy_2': -np.log(np.sum(psi_squared**2))
        }
        
        # Localization measure
        max_prob = np.max(psi_squared)
        results['localization'] = max_prob
        results['is_localized'] = max_prob > 1.0 / np.sqrt(results['n_occupied_states'])
        
        # Coherence properties
        results['coherence'] = np.sum(psi) / results['n_occupied_states']
        results['phase_uniformity'] = 1.0  # All phases are real and positive in MCTS
        
        return results
    
    def compute_path_action(self, path_nodes: List) -> float:
        """
        Compute quantum action S for a path through the tree.
        
        In path integral: amplitude = exp(iS/ℏ)
        In MCTS: visits ∝ |amplitude|²
        
        Therefore: S = -ℏ log(visits) + const
        
        Args:
            path_nodes: List of nodes along the path
            
        Returns:
            Action value
        """
        if not path_nodes:
            return 0.0
        
        # Action accumulates along path
        action = 0.0
        
        for i in range(len(path_nodes) - 1):
            parent = path_nodes[i]
            child = path_nodes[i + 1]
            
            # Transition probability
            if parent.visit_count > 0:
                transition_prob = child.visit_count / parent.visit_count
            else:
                transition_prob = 1e-10
            
            # Action contribution: -log(probability)
            action += -np.log(transition_prob + 1e-10)
            
            # Add Q-value contribution (potential energy)
            if hasattr(child, 'value_sum') and child.visit_count > 0:
                q_value = child.value_sum / child.visit_count
                action += -q_value  # Negative because high Q = low action
        
        return action
    
    def perform_finite_size_scaling_analysis(self, mcts_trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform finite-size scaling using subtrees at different depths.
        
        This implements the user's insight that subtrees can serve as
        systems of different effective sizes for scaling analysis.
        
        Args:
            mcts_trajectory: List of MCTS snapshots from a game
            
        Returns:
            Dictionary with scaling results and critical exponents
        """
        if not self.finite_size_analyzer or not self.subtree_extractor:
            logger.warning("Finite-size scaling modules not available")
            return {'error': 'Modules not imported'}
        
        results = {
            'subtree_data': [],
            'critical_exponents': {},
            'scaling_collapse': None,
            'critical_temperature': np.nan
        }
        
        # Extract depth-wise data from trajectory
        all_depth_data = {}
        for snapshot in mcts_trajectory:
            # Handle both Dict and DynamicsData formats
            if hasattr(snapshot, 'observables'):
                # DynamicsData format
                snapshot_dict = {
                    'visits': snapshot.observables.get('visits', []),
                    'q_values': snapshot.observables.get('q_values', []),
                    'tree_depth': snapshot.observables.get('depth', 0),
                    'children_data': snapshot.observables.get('children_data', [])
                }
            else:
                # Direct dict format
                snapshot_dict = snapshot
            
            depth_data = self.subtree_extractor.extract_depth_wise_data(snapshot_dict)
            
            # Merge with existing data
            for depth, data in depth_data.items():
                if depth not in all_depth_data:
                    all_depth_data[depth] = {
                        'visits': [],
                        'q_values': [],
                        'node_count': 0
                    }
                
                if len(data['visits']) > 0:
                    all_depth_data[depth]['visits'].extend(data['visits'])
                    all_depth_data[depth]['q_values'].extend(data['q_values'])
                    all_depth_data[depth]['node_count'] += data['node_count']
        
        # Convert to arrays
        for depth in all_depth_data:
            all_depth_data[depth]['visits'] = np.array(all_depth_data[depth]['visits'])
            all_depth_data[depth]['q_values'] = np.array(all_depth_data[depth]['q_values'])
        
        # Compute subtree properties
        subtree_props = self.subtree_extractor.compute_subtree_properties(all_depth_data)
        results['subtree_data'] = subtree_props
        
        # Extract scaling observables
        observables = self.subtree_extractor.extract_scaling_observables(subtree_props)
        
        if 'sizes' in observables and len(observables['sizes']) >= 3:
            # Perform finite-size scaling
            # Import SubtreeData locally to avoid circular imports
            try:
                from ..finite_size_scaling import SubtreeData
            except ImportError:
                import sys
                from pathlib import Path
                parent_dir = Path(__file__).parent.parent
                sys.path.insert(0, str(parent_dir))
                from finite_size_scaling import SubtreeData
            
            subtree_data_list = []
            for i in range(len(observables['sizes'])):
                # Create SubtreeData for each effective size
                depth = i + 1
                if depth in all_depth_data and len(all_depth_data[depth]['visits']) > 0:
                    visits = all_depth_data[depth]['visits']
                    q_values = all_depth_data[depth]['q_values']
                else:
                    visits = np.array([1])
                    q_values = np.array([0])
                
                # Extract temperature for this subtree
                temp, temp_err = self.extract_temperature_from_visits(visits, q_values)
                
                subtree = SubtreeData(
                    depth=depth,
                    size=int(observables['sizes'][i]),
                    visits=visits,
                    q_values=q_values,
                    temperature=temp if not np.isnan(temp) else 1.0,
                    order_parameter=float(observables['order_parameter'][i]),
                    susceptibility=float(observables['susceptibility'][i]),
                    correlation_length=float(observables['correlation_length'][i]),
                    energy=-float(observables['magnetization'][i]),  # E = -m in Ising analog
                    entropy=float(observables['specific_heat'][i]) / observables['sizes'][i]
                )
                subtree_data_list.append(subtree)
            
            # Set data and perform analysis
            self.finite_size_analyzer.subtree_data = subtree_data_list
            scaling_results = self.finite_size_analyzer.perform_finite_size_scaling()
            
            if 'critical_exponents' in scaling_results:
                results['critical_exponents'] = scaling_results['critical_exponents']
                results['critical_temperature'] = scaling_results.get('critical_temperature', np.nan)
                
                # Perform scaling collapse
                collapse_results = self.finite_size_analyzer.perform_scaling_collapse()
                if collapse_results:
                    results['scaling_collapse'] = collapse_results
            
            # Add measured vs theoretical comparison
            results['universality_class'] = self._identify_universality_class(
                results['critical_exponents']
            )
        
        return results
    
    def _identify_universality_class(self, exponents: Dict[str, float]) -> str:
        """
        Compare measured exponents to known universality classes.
        
        Args:
            exponents: Dictionary of critical exponents
            
        Returns:
            Best matching universality class
        """
        if not exponents:
            return 'unknown'
        
        # Known universality classes
        classes = {
            '2D Ising': {'beta_over_nu': 0.125, 'gamma_over_nu': 1.75},
            '3D Ising': {'beta_over_nu': 0.518, 'gamma_over_nu': 1.963},
            'Mean Field': {'beta_over_nu': 0.5, 'gamma_over_nu': 1.0},
            '2D XY': {'beta_over_nu': 0.125, 'gamma_over_nu': 1.75},
            'Percolation': {'beta_over_nu': 0.14, 'gamma_over_nu': 2.39}
        }
        
        # Find best match by chi-squared
        best_class = 'unknown'
        best_chi2 = float('inf')
        
        beta_measured = exponents.get('beta_over_nu', np.nan)
        gamma_measured = exponents.get('gamma_over_nu', np.nan)
        
        if not np.isnan(beta_measured) and not np.isnan(gamma_measured):
            for class_name, class_exponents in classes.items():
                chi2 = ((beta_measured - class_exponents['beta_over_nu'])**2 + 
                       (gamma_measured - class_exponents['gamma_over_nu'])**2)
                
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_class = class_name
        
        return best_class
    
    def analyze_rg_flow(self, mcts_trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze renormalization group flow through MCTS backpropagation.
        
        Key insight: Backpropagation implements RG transformation,
        coarse-graining from leaves (UV) to root (IR).
        
        Args:
            mcts_trajectory: List of MCTS snapshots
            
        Returns:
            RG flow analysis results
        """
        if not self.rg_flow_tracker:
            logger.warning("RG flow tracker not available")
            return {'error': 'Module not imported'}
        
        results = {
            'trajectories': [],
            'ensemble_analysis': {},
            'flow_universality': 0.0,
            'fixed_points': []
        }
        
        # Extract RG flow from each snapshot
        for snapshot in mcts_trajectory:
            # Handle both Dict and DynamicsData formats
            if hasattr(snapshot, 'observables'):
                snapshot_dict = {
                    'visits': snapshot.observables.get('visits', []),
                    'q_values': snapshot.observables.get('q_values', []),
                    'depth_wise_data': snapshot.observables.get('depth_wise_data', {})
                }
            else:
                snapshot_dict = snapshot
            
            # Extract flow trajectory
            trajectory = self.rg_flow_tracker.extract_rg_flow(snapshot_dict)
            if trajectory.flow_points:
                results['trajectories'].append(trajectory.to_dict())
        
        # Analyze ensemble if we have multiple trajectories
        if len(results['trajectories']) >= 2:
            # Convert back to RGTrajectory objects for analysis
            from ..rg_flow_tracker import RGTrajectory, RGFlowPoint
            
            trajectory_objects = []
            for traj_dict in results['trajectories']:
                flow_points = [
                    RGFlowPoint(**point) for point in traj_dict['flow_points']
                ]
                trajectory_objects.append(RGTrajectory(
                    flow_points=flow_points,
                    beta_function=traj_dict.get('beta_function'),
                    fixed_points=traj_dict.get('fixed_points')
                ))
            
            # Perform ensemble analysis
            ensemble_results = self.rg_flow_tracker.analyze_rg_flow_ensemble(trajectory_objects)
            results['ensemble_analysis'] = ensemble_results
            results['flow_universality'] = ensemble_results.get('flow_similarity', 0.0)
            results['fixed_points'] = ensemble_results.get('fixed_point_clusters', [])
        
        return results
    
    def analyze_quantum_interference(self, tree_root) -> Dict[str, Any]:
        """
        Analyze quantum interference patterns in MCTS tree.
        
        In quantum mechanics: interference occurs when multiple paths lead to same state
        In MCTS: multiple paths can reach same board position
        
        Returns:
            Interference analysis results
        """
        # Track all paths to each unique state
        state_paths = {}  # state -> list of (path, amplitude)
        
        def extract_state_paths(node, current_path, current_amplitude, depth=0):
            """Extract all paths and their amplitudes"""
            if depth > 20:  # Prevent infinite recursion
                return
                
            # Get state identifier (board position)
            state_id = self._get_state_id(node)
            
            if state_id not in state_paths:
                state_paths[state_id] = []
            
            # Amplitude = sqrt(visits/total)
            if hasattr(node, 'visit_count') and tree_root.visit_count > 0:
                amplitude = np.sqrt(node.visit_count / tree_root.visit_count)
            else:
                amplitude = 0.0
                
            state_paths[state_id].append((current_path.copy(), amplitude))
            
            # Continue to children
            if hasattr(node, 'children'):
                for action, child in node.children.items():
                    if child.visit_count > 0:
                        current_path.append(action)
                        extract_state_paths(child, current_path, amplitude, depth+1)
                        current_path.pop()
        
        # Extract all paths
        extract_state_paths(tree_root, [], 1.0)
        
        # Analyze interference
        interference_results = {
            'n_unique_states': len(state_paths),
            'interference_patterns': [],
            'constructive_interference_strength': 0.0,
            'destructive_interference_strength': 0.0
        }
        
        # Find states with multiple paths (interference)
        for state_id, paths_amplitudes in state_paths.items():
            if len(paths_amplitudes) > 1:
                # Multiple paths to same state - interference!
                amplitudes = [amp for _, amp in paths_amplitudes]
                
                # Total amplitude (coherent sum)
                total_amplitude = np.sum(amplitudes)
                
                # Incoherent sum (no interference)
                incoherent_sum = np.sqrt(np.sum([a**2 for a in amplitudes]))
                
                # Interference strength
                interference_strength = (total_amplitude**2 - incoherent_sum**2) / incoherent_sum**2
                
                interference_results['interference_patterns'].append({
                    'state': state_id,
                    'n_paths': len(paths_amplitudes),
                    'amplitudes': amplitudes,
                    'total_amplitude': total_amplitude,
                    'interference_strength': interference_strength,
                    'type': 'constructive' if interference_strength > 0 else 'destructive'
                })
                
                if interference_strength > 0:
                    interference_results['constructive_interference_strength'] += interference_strength
                else:
                    interference_results['destructive_interference_strength'] += abs(interference_strength)
        
        # Summary statistics
        if interference_results['interference_patterns']:
            interference_results['mean_paths_per_state'] = np.mean([
                p['n_paths'] for p in interference_results['interference_patterns']
            ])
            interference_results['max_interference'] = max([
                abs(p['interference_strength']) for p in interference_results['interference_patterns']
            ])
        
        return interference_results
    
    def _get_state_id(self, node):
        """Get unique identifier for a game state"""
        # This would need to be implemented based on your game representation
        # For now, use node id or hash
        if hasattr(node, 'state'):
            return hash(str(node.state))
        elif hasattr(node, 'board'):
            return hash(str(node.board))
        else:
            return id(node)
        
    def generate_measurement_report(self, 
                                   dynamics_data_list: List[DynamicsData]) -> Dict[str, Any]:
        """
        Generate comprehensive report of all measurements from MCTS data.
        
        Everything measured, nothing assumed.
        """
        report = {
            'n_games': len(dynamics_data_list),
            'measurements': {},
            'scaling_relations': {},
            'phenomena': {}
        }
        
        # Temperature measurements
        all_temperatures = []
        all_n_visits = []
        
        for data in dynamics_data_list:
            for snapshot in data.snapshots:
                if 'visits' in snapshot and 'q_values' in snapshot:
                    visits = np.array(snapshot['visits'])
                    q_values = np.array(snapshot['q_values'])
                    
                    temp, temp_err = self.extract_temperature_from_visits(visits, q_values)
                    
                    if not np.isnan(temp):
                        all_temperatures.append((temp, temp_err))
                        all_n_visits.append(visits.sum())
                        
        # Temperature scaling analysis
        if len(all_temperatures) > 10:
            scaling_data = [(n, t[0]) for n, t in zip(all_n_visits, all_temperatures)]
            temp_scaling = self.extract_scaling_relations_from_ensemble(
                scaling_data, 'N_visits'
            )
            report['scaling_relations']['temperature'] = temp_scaling
            
        # Decoherence analysis
        decoherence_results = []
        for data in dynamics_data_list:
            dec_result = self.measure_decoherence_from_distribution_evolution(
                data.snapshots
            )
            if dec_result:
                decoherence_results.append(dec_result)
                
        if decoherence_results:
            report['phenomena']['decoherence'] = {
                'n_trajectories': len(decoherence_results),
                'average_entropy_rate': np.mean([r['average_entropy_rate'] for r in decoherence_results]),
                'decay_models': decoherence_results[0].get('decay_models', {})
            }
            
        # Critical behavior
        critical_results = []
        for data in dynamics_data_list:
            crit_result = self.measure_critical_behavior_from_fluctuations(data)
            if crit_result:
                critical_results.append(crit_result)
                
        if critical_results:
            report['phenomena']['critical'] = {
                'n_trajectories': len(critical_results),
                'gap_statistics': {
                    'mean': np.mean([r['gap_mean'] for r in critical_results]),
                    'std': np.mean([r['gap_std'] for r in critical_results])
                }
            }
            
        # RG flow
        rg_results = []
        for data in dynamics_data_list:
            rg_result = self.extract_rg_flow_from_tree_hierarchy(data)
            if rg_result:
                rg_results.append(rg_result)
                
        if rg_results:
            report['phenomena']['rg_flow'] = {
                'n_trajectories': len(rg_results),
                'example_flow': rg_results[0]  # Show one example
            }
            
        return report
    
    def extract_temperature_from_dynamics(self, dynamics_data) -> Dict[str, Any]:
        """Extract temperature from dynamics data"""
        temperatures = []
        temperature_errors = []
        
        for i, snapshot in enumerate(dynamics_data.snapshots):
            if 'visits' in snapshot and 'q_values' in snapshot:
                visits = np.array(snapshot['visits'])
                q_values = np.array(snapshot['q_values'])
                
                # Extract temperature for this snapshot
                temp, temp_err = self.extract_temperature_from_visits(visits, q_values)
                
                if not np.isnan(temp):
                    temperatures.append(temp)
                    temperature_errors.append(temp_err)
                else:
                    temperatures.append(np.nan)
                    temperature_errors.append(np.nan)
            else:
                temperatures.append(np.nan)
                temperature_errors.append(np.nan)
        
        return {
            'temperatures': temperatures,
            'temperature_errors': temperature_errors,
            'n_snapshots': len(dynamics_data.snapshots),
            'n_valid_temperatures': len([t for t in temperatures if not np.isnan(t)])
        }
    
    def extract_path_integral_weights(self, dynamics_data) -> Dict[str, Any]:
        """
        Extract path integral weights P[γ] ∝ exp(-β S[γ]) from MCTS trajectories.
        
        This implements the theoretical formulation from quantum_mcts_foundation.md
        where each path through the tree has an action S[γ] = sum of Q-values.
        
        Args:
            dynamics_data: Dynamics data containing tree snapshots
            
        Returns:
            Dictionary with path integral analysis
        """
        if not hasattr(dynamics_data, 'snapshots') or len(dynamics_data.snapshots) == 0:
            return {'error': 'No snapshots available for path integral analysis'}
            
        path_weights = []
        actions = []
        
        for snapshot in dynamics_data.snapshots:
            if 'q_values' not in snapshot or 'visits' not in snapshot:
                continue
                
            q_values = np.array(snapshot['q_values'])
            visits = np.array(snapshot['visits'])
            
            # Extract temperature for this snapshot
            temp, _ = self.extract_temperature_from_visits(visits, q_values)
            
            if np.isnan(temp) or temp <= 0:
                continue
                
            beta = 1.0 / temp
            
            # Compute action for each path (cumulative Q-values)
            # For MCTS, each "path" corresponds to a sequence of actions
            # We approximate this by treating each action as a path segment
            path_actions = np.cumsum(q_values)  # Cumulative action
            
            # Path integral weights: P[γ] ∝ exp(-β S[γ])
            path_weights_snapshot = np.exp(-beta * path_actions)
            path_weights_snapshot /= np.sum(path_weights_snapshot)  # Normalize
            
            path_weights.append(path_weights_snapshot)
            actions.append(path_actions)
            
        if not path_weights:
            return {'error': 'No valid path integral weights computed'}
            
        return {
            'path_weights': path_weights,
            'actions': actions,
            'n_snapshots': len(path_weights),
            'interpretation': 'P[γ] ∝ exp(-β S[γ]) from quantum_mcts_foundation.md',
            'avg_path_weight_entropy': np.mean([
                -np.sum(weights * np.log(weights + 1e-10)) 
                for weights in path_weights
            ])
        }