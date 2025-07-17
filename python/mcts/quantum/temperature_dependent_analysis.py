"""
Temperature-dependent phenomena analysis for MCTS.

This module analyzes how quantum and statistical mechanical phenomena 
depend on the effective temperature extracted from MCTS visit distributions.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import optimize
from scipy.stats import entropy
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class TemperatureDependentResult:
    """Result of temperature-dependent analysis"""
    temperature: float
    entropy: float
    free_energy: float
    specific_heat: float
    susceptibility: float
    correlation_length: float
    order_parameter: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'temperature': self.temperature,
            'entropy': self.entropy,
            'free_energy': self.free_energy,
            'specific_heat': self.specific_heat,
            'susceptibility': self.susceptibility,
            'correlation_length': self.correlation_length,
            'order_parameter': self.order_parameter
        }


class TemperatureDependentAnalyzer:
    """
    Analyzes temperature-dependent phenomena in MCTS.
    
    This analyzer studies how quantum and statistical mechanical 
    properties change with the effective temperature extracted 
    from visit distributions.
    
    IMPORTANT DISCLAIMER: MCTS is fundamentally a non-equilibrium process.
    The thermodynamic quantities computed here should be interpreted as
    phenomenological measures rather than true equilibrium thermodynamics.
    The "temperature" is an effective parameter extracted from visit
    distributions, not a thermal equilibrium temperature.
    """
    
    def __init__(self):
        """Initialize temperature-dependent analyzer"""
        self.results_cache = {}
    
    def analyze_temperature_dependence(self, 
                                     dynamics_data_list: List[Any],
                                     temperature_extractor: Any) -> Dict[str, Any]:
        """
        Analyze temperature-dependent phenomena across games.
        
        Args:
            dynamics_data_list: List of dynamics data from different games
            temperature_extractor: Physics extractor for temperature calculation
            
        Returns:
            Dictionary with temperature-dependent analysis results
        """
        logger.info("Analyzing temperature-dependent phenomena...")
        
        # Extract temperatures and corresponding observables
        temperature_data = []
        
        for i, dynamics_data in enumerate(dynamics_data_list):
            try:
                # Extract temperature from each game
                T_results = temperature_extractor.extract_temperature_from_dynamics(dynamics_data)
                
                if T_results and 'temperatures' in T_results:
                    temperatures = T_results['temperatures']
                    
                    # Extract observables for each temperature
                    for j, temp in enumerate(temperatures):
                        if temp > 0 and not np.isnan(temp):
                            observables = self._extract_observables_at_temperature(
                                dynamics_data, j, temp
                            )
                            
                            if observables:
                                temperature_data.append({
                                    'game_index': i,
                                    'snapshot_index': j,
                                    'temperature': temp,
                                    **observables
                                })
                            
            except Exception as e:
                logger.warning(f"Failed to extract temperature data for game {i}: {e}")
                continue
        
        if not temperature_data:
            return {'error': 'No temperature data extracted'}
        
        # Analyze temperature dependence
        results = self._analyze_temperature_scaling(temperature_data)
        
        # Identify phase transitions
        phase_transitions = self._identify_phase_transitions(temperature_data)
        
        # Compute temperature-dependent quantities
        thermodynamic_quantities = self._compute_thermodynamic_quantities(temperature_data)
        
        return {
            'individual_data': temperature_data,
            'temperature_scaling': results,
            'phase_transitions': phase_transitions,
            'thermodynamic_quantities': thermodynamic_quantities,
            'n_temperature_points': len(temperature_data),
            'temperature_range': {
                'min': min(d['temperature'] for d in temperature_data),
                'max': max(d['temperature'] for d in temperature_data)
            },
            'disclaimer': 'MCTS is non-equilibrium: thermodynamic quantities are phenomenological measures only',
            'interpretation': 'Temperature is effective parameter from visit distributions, not thermal equilibrium'
        }
    
    def _extract_observables_at_temperature(self, 
                                          dynamics_data: Any, 
                                          snapshot_index: int,
                                          temperature: float) -> Optional[Dict[str, Any]]:
        """Extract observables at a specific temperature"""
        try:
            if snapshot_index >= len(dynamics_data.snapshots):
                return None
                
            snapshot = dynamics_data.snapshots[snapshot_index]
            
            # Extract basic observables
            visits = snapshot.get('visits', [])
            q_values = snapshot.get('q_values', [])
            
            if not visits or not q_values:
                return None
            
            visits = np.array(visits)
            q_values = np.array(q_values)
            
            # Total visits (system size)
            total_visits = np.sum(visits)
            if total_visits == 0:
                return None
            
            # Probability distribution
            probs = visits / total_visits
            
            # Entropy
            entropy_val = entropy(probs + 1e-10)
            
            # Average Q-value (energy)
            avg_q = np.sum(probs * q_values)
            
            # Q-value variance (energy fluctuations)
            q_variance = np.sum(probs * (q_values - avg_q)**2)
            
            # Order parameter (concentration on best move)
            order_param = np.max(probs)
            
            # Correlation length (from Q-value correlations)
            correlation_length = self._compute_correlation_length(q_values, probs)
            
            # Susceptibility (response to perturbations)
            susceptibility = self._compute_susceptibility(q_values, probs, temperature)
            
            # Free energy F = -T ln(Z) where Z = sum(exp(beta * Q))
            if temperature > 0:
                beta = 1.0 / temperature
                log_Z = np.log(np.sum(np.exp(beta * q_values)))
                free_energy = -temperature * log_Z
            else:
                free_energy = 0.0
            
            return {
                'entropy': entropy_val,
                'avg_q_value': avg_q,
                'q_variance': q_variance,
                'order_parameter': order_param,
                'correlation_length': correlation_length,
                'susceptibility': susceptibility,
                'free_energy': free_energy,
                'total_visits': total_visits,
                'n_actions': len(q_values)
            }
            
        except Exception as e:
            logger.debug(f"Failed to extract observables at temperature {temperature}: {e}")
            return None
    
    def _compute_correlation_length(self, q_values: np.ndarray, probs: np.ndarray) -> float:
        """Compute correlation length from Q-value correlations"""
        n = len(q_values)
        if n < 2:
            return 0.0
        
        # Simple correlation length estimate
        # Based on how quickly correlations decay with action distance
        correlations = []
        for lag in range(1, min(n, 5)):
            if n - lag > 0:
                corr = np.corrcoef(q_values[:-lag], q_values[lag:])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        if correlations:
            # Find exponential decay length
            xi = 1.0 / max(np.log(max(correlations[0], 1e-10)), 1e-10)
            return min(xi, n)
        
        return 1.0
    
    def _compute_susceptibility(self, q_values: np.ndarray, probs: np.ndarray, temperature: float) -> float:
        """Compute susceptibility (response to perturbations)"""
        if temperature <= 0:
            return 0.0
        
        # Susceptibility = beta * variance of Q-values
        # This measures how much the system responds to small changes
        beta = 1.0 / temperature
        avg_q = np.sum(probs * q_values)
        variance = np.sum(probs * (q_values - avg_q)**2)
        
        return beta * variance
    
    def _analyze_temperature_scaling(self, temperature_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how observables scale with temperature"""
        if len(temperature_data) < 3:
            return {'error': 'Not enough data points for scaling analysis'}
        
        # Sort by temperature
        sorted_data = sorted(temperature_data, key=lambda x: x['temperature'])
        
        temperatures = np.array([d['temperature'] for d in sorted_data])
        entropies = np.array([d['entropy'] for d in sorted_data])
        order_params = np.array([d['order_parameter'] for d in sorted_data])
        susceptibilities = np.array([d['susceptibility'] for d in sorted_data])
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(temperatures) | np.isnan(entropies) | 
                      np.isnan(order_params) | np.isnan(susceptibilities))
        
        if np.sum(valid_mask) < 3:
            return {'error': 'Not enough valid data points'}
        
        temperatures = temperatures[valid_mask]
        entropies = entropies[valid_mask]
        order_params = order_params[valid_mask]
        susceptibilities = susceptibilities[valid_mask]
        
        results = {}
        
        # Entropy scaling S(T)
        try:
            # Fit S ~ T^alpha
            log_T = np.log(temperatures)
            log_S = np.log(entropies + 1e-10)
            entropy_fit = np.polyfit(log_T, log_S, 1)
            results['entropy_scaling'] = {
                'exponent': entropy_fit[0],
                'prefactor': np.exp(entropy_fit[1]),
                'fit_quality': np.corrcoef(log_T, log_S)[0, 1]**2
            }
        except:
            results['entropy_scaling'] = {'error': 'Failed to fit entropy scaling'}
        
        # Order parameter scaling O(T)
        try:
            # Often O ~ (T_c - T)^beta near critical point
            # For now, fit power law
            log_T = np.log(temperatures)
            log_O = np.log(order_params + 1e-10)
            order_fit = np.polyfit(log_T, log_O, 1)
            results['order_parameter_scaling'] = {
                'exponent': order_fit[0],
                'prefactor': np.exp(order_fit[1]),
                'fit_quality': np.corrcoef(log_T, log_O)[0, 1]**2
            }
        except:
            results['order_parameter_scaling'] = {'error': 'Failed to fit order parameter scaling'}
        
        # Susceptibility scaling χ(T)
        try:
            # χ ~ T^(-gamma) or diverges at T_c
            log_T = np.log(temperatures)
            log_chi = np.log(susceptibilities + 1e-10)
            chi_fit = np.polyfit(log_T, log_chi, 1)
            results['susceptibility_scaling'] = {
                'exponent': chi_fit[0],
                'prefactor': np.exp(chi_fit[1]),
                'fit_quality': np.corrcoef(log_T, log_chi)[0, 1]**2
            }
        except:
            results['susceptibility_scaling'] = {'error': 'Failed to fit susceptibility scaling'}
        
        return results
    
    def _identify_phase_transitions(self, temperature_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify potential phase transitions in temperature data"""
        if len(temperature_data) < 5:
            return []
        
        # Sort by temperature
        sorted_data = sorted(temperature_data, key=lambda x: x['temperature'])
        
        temperatures = np.array([d['temperature'] for d in sorted_data])
        order_params = np.array([d['order_parameter'] for d in sorted_data])
        susceptibilities = np.array([d['susceptibility'] for d in sorted_data])
        
        phase_transitions = []
        
        # Look for peaks in susceptibility (critical points)
        if len(susceptibilities) > 3:
            # Find local maxima
            for i in range(1, len(susceptibilities) - 1):
                if (susceptibilities[i] > susceptibilities[i-1] and 
                    susceptibilities[i] > susceptibilities[i+1] and
                    susceptibilities[i] > np.mean(susceptibilities) + np.std(susceptibilities)):
                    
                    phase_transitions.append({
                        'type': 'critical_point',
                        'temperature': temperatures[i],
                        'susceptibility': susceptibilities[i],
                        'order_parameter': order_params[i],
                        'confidence': 'medium'
                    })
        
        # Look for rapid changes in order parameter
        if len(order_params) > 3:
            # Check if temperatures are properly spaced
            temp_diffs = np.diff(temperatures)
            if np.all(temp_diffs > 0) and np.min(temp_diffs) > 1e-10:
                order_gradients = np.gradient(order_params, temperatures)
                steep_indices = np.where(np.abs(order_gradients) > 2 * np.std(order_gradients))[0]
            else:
                # Fallback: use simple differences if temperatures are not properly spaced
                order_gradients = np.diff(order_params) / np.maximum(temp_diffs, 1e-10)
                steep_indices = np.where(np.abs(order_gradients) > 2 * np.std(order_gradients))[0]
            
            for idx in steep_indices:
                if 0 < idx < len(temperatures) - 1:
                    phase_transitions.append({
                        'type': 'order_parameter_transition',
                        'temperature': temperatures[idx],
                        'gradient': order_gradients[idx],
                        'order_parameter_change': order_params[idx+1] - order_params[idx-1],
                        'confidence': 'low'
                    })
        
        return phase_transitions
    
    def _compute_thermodynamic_quantities(self, temperature_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute thermodynamic quantities from temperature data"""
        if len(temperature_data) < 3:
            return {'error': 'Not enough data points'}
        
        # Sort by temperature
        sorted_data = sorted(temperature_data, key=lambda x: x['temperature'])
        
        temperatures = np.array([d['temperature'] for d in sorted_data])
        entropies = np.array([d['entropy'] for d in sorted_data])
        free_energies = np.array([d['free_energy'] for d in sorted_data])
        
        # Remove NaN values
        valid_mask = ~(np.isnan(temperatures) | np.isnan(entropies) | np.isnan(free_energies))
        
        if np.sum(valid_mask) < 3:
            return {'error': 'Not enough valid data points'}
        
        temperatures = temperatures[valid_mask]
        entropies = entropies[valid_mask]
        free_energies = free_energies[valid_mask]
        
        results = {}
        
        # Specific heat C = dE/dT = -T^2 d(F/T)/dT
        if len(temperatures) > 2:
            try:
                # Check if temperatures are properly spaced for gradient calculation
                temp_diffs = np.diff(temperatures)
                if np.all(temp_diffs > 0) and np.min(temp_diffs) > 1e-10:
                    # C = -T^2 d²F/dT²
                    d2F_dT2 = np.gradient(np.gradient(free_energies, temperatures), temperatures)
                    specific_heat = -temperatures**2 * d2F_dT2
                else:
                    # Fallback: use finite differences
                    dF_dT = np.diff(free_energies) / np.maximum(temp_diffs, 1e-10)
                    # Pad to maintain array size
                    dF_dT = np.append(dF_dT, dF_dT[-1])
                    d2F_dT2 = np.diff(dF_dT) / np.maximum(temp_diffs, 1e-10)
                    d2F_dT2 = np.append(d2F_dT2, d2F_dT2[-1])
                    specific_heat = -temperatures**2 * d2F_dT2
                
                results['specific_heat'] = {
                    'temperatures': temperatures.tolist(),
                    'values': specific_heat.tolist(),
                    'average': np.mean(specific_heat),
                    'peak_temperature': temperatures[np.argmax(specific_heat)]
                }
            except:
                results['specific_heat'] = {'error': 'Failed to compute specific heat'}
        
        # Entropy vs temperature relationship
        results['entropy_temperature'] = {
            'temperatures': temperatures.tolist(),
            'entropies': entropies.tolist(),
            'max_entropy': np.max(entropies),
            'max_entropy_temperature': temperatures[np.argmax(entropies)]
        }
        
        # Free energy vs temperature
        results['free_energy_temperature'] = {
            'temperatures': temperatures.tolist(),
            'free_energies': free_energies.tolist(),
            'min_free_energy': np.min(free_energies),
            'min_free_energy_temperature': temperatures[np.argmin(free_energies)]
        }
        
        return results
    
    def create_temperature_plots(self, 
                                analysis_results: Dict[str, Any],
                                output_dir: str) -> List[str]:
        """Create plots of temperature-dependent phenomena"""
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        output_path = Path(output_dir)
        plot_paths = []
        
        if 'individual_data' not in analysis_results:
            return plot_paths
        
        data = analysis_results['individual_data']
        if not data:
            return plot_paths
        
        # Sort by temperature
        sorted_data = sorted(data, key=lambda x: x['temperature'])
        temperatures = [d['temperature'] for d in sorted_data]
        
        # Plot 1: Temperature vs observables
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Temperature-Dependent Phenomena in MCTS', fontsize=16)
        
        # Entropy vs temperature
        entropies = [d['entropy'] for d in sorted_data]
        axes[0, 0].plot(temperatures, entropies, 'bo-', label='Measured')
        axes[0, 0].set_xlabel('Temperature')
        axes[0, 0].set_ylabel('Entropy')
        axes[0, 0].set_title('Entropy vs Temperature')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Order parameter vs temperature
        order_params = [d['order_parameter'] for d in sorted_data]
        axes[0, 1].plot(temperatures, order_params, 'ro-', label='Measured')
        axes[0, 1].set_xlabel('Temperature')
        axes[0, 1].set_ylabel('Order Parameter')
        axes[0, 1].set_title('Order Parameter vs Temperature')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Susceptibility vs temperature
        susceptibilities = [d['susceptibility'] for d in sorted_data]
        axes[1, 0].plot(temperatures, susceptibilities, 'go-', label='Measured')
        axes[1, 0].set_xlabel('Temperature')
        axes[1, 0].set_ylabel('Susceptibility')
        axes[1, 0].set_title('Susceptibility vs Temperature')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Free energy vs temperature
        free_energies = [d['free_energy'] for d in sorted_data]
        axes[1, 1].plot(temperatures, free_energies, 'mo-', label='Measured')
        axes[1, 1].set_xlabel('Temperature')
        axes[1, 1].set_ylabel('Free Energy')
        axes[1, 1].set_title('Free Energy vs Temperature')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Add measurement label
        fig.text(0.02, 0.02, 'MEASURED FROM MCTS', fontsize=10, alpha=0.7, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
        
        plot_path = output_path / 'temperature_dependent_phenomena.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(plot_path))
        
        return plot_paths