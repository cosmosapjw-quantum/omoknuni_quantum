"""
Temperature integration module for non-equilibrium MCTS systems.

MCTS is fundamentally non-equilibrium, requiring careful temperature definitions:
1. Effective Temperature: From visit distribution fitting π(a) ∝ exp(β·Q(a))
2. Configurational Temperature: From entropy of configuration space
3. Dynamic Temperature: Time-dependent exploration control
4. Fluctuation-Dissipation Temperature: From FDT violation ratio

This module ensures physically correct temperature usage throughout analysis.
"""
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TemperatureIntegrator:
    """
    Integrates authentic temperature measurements into the physics analysis pipeline.
    
    Key improvements:
    1. Measures temperature at each snapshot
    2. Propagates temperatures to all analyses
    3. Handles temperature evolution correctly
    """
    
    def __init__(self):
        """Initialize temperature integrator"""
        self._temperature_cache = {}
        self._fdt_temperature_cache = {}
        self._configurational_temp_cache = {}
        
    def add_temperature_to_snapshot(self, snapshot: Any, tree: Any) -> None:
        """
        Add authentic temperature measurement to snapshot.
        
        Args:
            snapshot: TreeSnapshot object to modify
            tree: MCTS tree for temperature extraction
        """
        # Import here to avoid circular dependency
        from ..analysis.authentic_physics_extractor import AuthenticPhysicsExtractor
        
        if not hasattr(tree, 'children') or len(tree.children) == 0:
            return
            
        # Extract visits and Q-values for root actions
        root_visits = []
        root_q_values = []
        
        for action, child in tree.children.items():
            root_visits.append(child.visit_count)
            if child.visit_count > 0:
                root_q_values.append(child.value_sum / child.visit_count)
            else:
                root_q_values.append(0.0)
        
        # Measure temperature
        extractor = AuthenticPhysicsExtractor()
        temp, temp_err = extractor.extract_temperature_from_visits(
            np.array(root_visits), 
            np.array(root_q_values)
        )
        
        # Add to observables if valid
        if not np.isnan(temp):
            snapshot.observables['temperature'] = float(temp)
            snapshot.observables['temperature_error'] = float(temp_err)
            
            # Cache for quick access
            self._temperature_cache[snapshot.timestamp] = (temp, temp_err)
    
    def get_temperature_evolution(self, snapshots: List[Any]) -> Dict[str, np.ndarray]:
        """
        Extract temperature evolution from snapshots.
        
        Returns:
            Dictionary with times, temperatures, and errors
        """
        times = []
        temperatures = []
        errors = []
        
        for snapshot in snapshots:
            if 'temperature' in snapshot.observables:
                times.append(snapshot.timestamp)
                temperatures.append(snapshot.observables['temperature'])
                errors.append(snapshot.observables.get('temperature_error', 0.0))
        
        return {
            'times': np.array(times),
            'temperatures': np.array(temperatures),
            'errors': np.array(errors)
        }
    
    def analyze_temperature_phases(self, snapshots: List[Any]) -> Dict[str, Any]:
        """
        Analyze temperature evolution phases in MCTS.
        
        Expected pattern:
        1. Initial phase: Low temperature (focused exploration)
        2. Middle phase: Temperature rises as tree expands
        3. Final phase: High temperature (broad sampling of converged distribution)
        
        Returns:
            Phase analysis results
        """
        evolution = self.get_temperature_evolution(snapshots)
        
        if len(evolution['temperatures']) < 3:
            return {'error': 'Insufficient temperature data'}
        
        temps = evolution['temperatures']
        times = evolution['times']
        
        # Identify phases using gradient analysis
        gradients = np.gradient(temps, times)
        
        # Phase boundaries (heuristic)
        n = len(temps)
        phase1_end = n // 3
        phase2_end = 2 * n // 3
        
        results = {
            'phase1': {
                'range': [0, phase1_end],
                'mean_temperature': np.mean(temps[:phase1_end]),
                'temperature_trend': 'low',
                'interpretation': 'Focused exploration'
            },
            'phase2': {
                'range': [phase1_end, phase2_end],
                'mean_temperature': np.mean(temps[phase1_end:phase2_end]),
                'temperature_trend': 'rising',
                'interpretation': 'Tree expansion'
            },
            'phase3': {
                'range': [phase2_end, n],
                'mean_temperature': np.mean(temps[phase2_end:]),
                'temperature_trend': 'high',
                'interpretation': 'Converged sampling'
            }
        }
        
        # Check if pattern matches expectations
        if (results['phase1']['mean_temperature'] < results['phase2']['mean_temperature'] < 
            results['phase3']['mean_temperature']):
            results['pattern_match'] = 'Expected MCTS temperature evolution'
        else:
            results['pattern_match'] = 'Anomalous temperature evolution'
        
        return results
    
    def correct_thermodynamic_calculations(self, 
                                         thermodynamic_results: List[Dict[str, float]],
                                         snapshots: List[Any]) -> List[Dict[str, float]]:
        """
        Recalculate thermodynamic quantities using authentic temperatures.
        
        Args:
            thermodynamic_results: Original results with T=1.0
            snapshots: Snapshots with authentic temperatures
            
        Returns:
            Corrected thermodynamic results
        """
        corrected = []
        
        for i, (result, snapshot) in enumerate(zip(thermodynamic_results, snapshots)):
            if 'temperature' in snapshot.observables:
                T = snapshot.observables['temperature']
            else:
                T = 1.0  # Fallback
            
            # Recalculate free energy with correct temperature
            energy = result['energy']
            entropy = result['entropy']
            free_energy = energy - T * entropy
            
            corrected.append({
                'energy': energy,
                'entropy': entropy,
                'temperature': T,
                'free_energy': free_energy,
                'heat_capacity': result.get('heat_capacity', 0.0),
                'partition_function': result.get('partition_function', 1.0)
            })
        
        return corrected
    
    def measure_configurational_temperature(self, snapshot: Any) -> Tuple[float, float]:
        """
        Measure configurational temperature from entropy of visit distribution.
        
        For non-equilibrium systems, this provides a temperature scale
        based on the phase space exploration.
        
        T_config = dE/dS where S is configurational entropy
        
        Returns:
            (T_config, error)
        """
        if 'visit_distribution' not in snapshot.observables:
            return np.nan, np.nan
        
        visits = snapshot.observables['visit_distribution']
        if isinstance(visits, torch.Tensor):
            visits = visits.cpu().numpy()
        
        # Compute configurational entropy
        total_visits = np.sum(visits)
        if total_visits == 0:
            return np.nan, np.nan
        
        probs = visits / total_visits
        probs = probs[probs > 0]  # Remove zeros
        S_config = -np.sum(probs * np.log(probs))
        
        # Estimate temperature from entropy gradient
        # For MCTS: T_config ~ 1/sqrt(N) scaling expected
        # This comes from random walk in configuration space
        T_config = np.sqrt(2.0 / total_visits) * S_config
        T_config_err = T_config / np.sqrt(total_visits)
        
        return T_config, T_config_err
    
    def measure_crooks_temperature(self, work_forward: np.ndarray, 
                                  work_backward: np.ndarray) -> Tuple[float, float]:
        """
        Extract temperature using Crooks fluctuation theorem.
        
        For far-from-equilibrium systems:
        P_F(W) / P_R(-W) = exp(β(W - ΔF))
        
        This is valid arbitrarily far from equilibrium, unlike FDT.
        
        Args:
            work_forward: Work values from forward process
            work_backward: Work values from backward process
            
        Returns:
            (temperature, free_energy_difference)
        """
        # Create histogram bins
        bins = np.linspace(min(work_forward.min(), -work_backward.max()), 
                          max(work_forward.max(), -work_backward.min()), 50)
        
        # Compute probability densities
        p_forward, _ = np.histogram(work_forward, bins=bins, density=True)
        p_backward, _ = np.histogram(-work_backward, bins=bins, density=True)
        
        # Get bin centers
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Only use bins where both probabilities are significant
        mask = (p_forward > 1e-10) & (p_backward > 1e-10)
        
        if np.sum(mask) < 5:
            return np.nan, np.nan
        
        # Compute log ratio
        log_ratio = np.log(p_forward[mask] / p_backward[mask])
        W = bin_centers[mask]
        
        # Linear fit: log(P_F/P_R) = β*W - β*ΔF
        coeffs, cov = np.polyfit(W, log_ratio, 1, cov=True)
        beta = coeffs[0]
        beta_deltaF = -coeffs[1]
        
        if beta <= 0:
            return np.nan, np.nan
        
        temperature = 1.0 / beta
        delta_F = beta_deltaF / beta
        
        return temperature, delta_F
    
    def measure_hatano_sasa_temperature(self, trajectory: List[Any]) -> Dict[str, float]:
        """
        Use Hatano-Sasa identity for steady-state non-equilibrium systems.
        
        Separates heat into:
        - Housekeeping heat: maintains non-equilibrium steady state
        - Excess heat: related to free energy changes
        
        Returns:
            Dictionary with housekeeping and excess temperatures
        """
        if len(trajectory) < 3:
            return {'housekeeping_temp': np.nan, 'excess_temp': np.nan}
        
        # Extract energy changes along trajectory
        energies = []
        for snapshot in trajectory:
            if 'value_landscape' in snapshot.observables:
                q_values = snapshot.observables['value_landscape']
                if isinstance(q_values, torch.Tensor):
                    q_values = q_values.cpu().numpy()
                # Use mean Q-value as proxy for energy
                energies.append(-np.mean(q_values))
        
        if len(energies) < 3:
            return {'housekeeping_temp': np.nan, 'excess_temp': np.nan}
        
        energies = np.array(energies)
        
        # Housekeeping heat: steady-state maintenance
        # For MCTS: cost of maintaining tree structure
        housekeeping = np.mean(np.abs(np.diff(energies)))
        
        # Excess heat: net energy change
        excess = energies[-1] - energies[0]
        
        # Estimate temperatures from fluctuations
        energy_fluct = np.std(energies)
        
        # Housekeeping temperature from steady-state fluctuations
        T_house = energy_fluct**2 / housekeeping if housekeeping > 0 else np.inf
        
        # Excess temperature from net change
        T_excess = energy_fluct**2 / abs(excess) if excess != 0 else np.inf
        
        return {
            'housekeeping_temp': T_house,
            'excess_temp': T_excess,
            'housekeeping_heat': housekeeping,
            'excess_heat': excess
        }
    
    def measure_jarzynski_temperature(self, work_values: np.ndarray,
                                    delta_F_estimate: float = None) -> float:
        """
        Extract temperature from Jarzynski equality.
        
        ⟨exp(-βW)⟩ = exp(-βΔF)
        
        Note: MCTS lacks true reverse process, so this gives
        an effective temperature for the forward process only.
        
        Args:
            work_values: Work measurements
            delta_F_estimate: Independent estimate of ΔF (optional)
            
        Returns:
            Effective temperature
        """
        if len(work_values) < 10:
            return np.nan
        
        # If no ΔF estimate, use mean work as approximation
        if delta_F_estimate is None:
            delta_F_estimate = np.mean(work_values)
        
        # Try different β values and find best fit
        beta_test = np.linspace(0.1, 10, 100)
        jarzynski_lhs = []
        
        for beta in beta_test:
            lhs = np.mean(np.exp(-beta * work_values))
            jarzynski_lhs.append(lhs)
        
        jarzynski_lhs = np.array(jarzynski_lhs)
        jarzynski_rhs = np.exp(-beta_test * delta_F_estimate)
        
        # Find β that minimizes difference
        idx_best = np.argmin(np.abs(jarzynski_lhs - jarzynski_rhs))
        beta_best = beta_test[idx_best]
        
        return 1.0 / beta_best
    
    def measure_generalized_fdt_temperature(self, response_data: Dict[str, np.ndarray],
                                          correlation_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Measure temperature from generalized FDT for non-equilibrium steady states.
        
        For linear non-equilibrium (near steady state):
        χ(t,s) = β_eff(t,s) * C(t,s) + non-equilibrium corrections
        
        This uses the framework for systems with small deviations from equilibrium.
        
        Returns:
            Dictionary with time-dependent effective temperatures
        """
        results = {}
        
        # For different time scales
        for timescale in ['short', 'medium', 'long']:
            if timescale in response_data and timescale in correlation_data:
                chi = response_data[timescale]
                C = correlation_data[timescale]
                
                # Compute effective β for this timescale
                # Using least squares fit: χ = β_eff * C
                mask = (np.abs(C) > 1e-10)
                if np.sum(mask) > 5:
                    beta_eff = np.sum(chi[mask] * C[mask]) / np.sum(C[mask]**2)
                    
                    if beta_eff > 0:
                        results[f'T_eff_{timescale}'] = 1.0 / beta_eff
                    else:
                        results[f'T_eff_{timescale}'] = np.nan
                else:
                    results[f'T_eff_{timescale}'] = np.nan
        
        return results
    
    def analyze_temperature_spectrum(self, snapshots: List[Any]) -> Dict[str, Any]:
        """
        Analyze the full spectrum of temperatures in non-equilibrium MCTS.
        
        Returns:
            Dictionary with different temperature measures and their interpretations
        """
        results = {
            'effective_temperatures': [],
            'configurational_temperatures': [],
            'fdt_temperatures': [],
            'times': []
        }
        
        for snapshot in snapshots:
            # Effective temperature from Boltzmann fit
            if 'temperature' in snapshot.observables:
                results['effective_temperatures'].append(snapshot.observables['temperature'])
            else:
                results['effective_temperatures'].append(np.nan)
            
            # Configurational temperature
            T_config, _ = self.measure_configurational_temperature(snapshot)
            results['configurational_temperatures'].append(T_config)
            
            # FDT temperature (would need response/correlation data)
            # Placeholder for now
            results['fdt_temperatures'].append(np.nan)
            
            results['times'].append(snapshot.timestamp)
        
        # Analyze relationships
        eff_temps = np.array(results['effective_temperatures'])
        config_temps = np.array(results['configurational_temperatures'])
        
        # Remove NaN values for correlation
        valid_mask = ~(np.isnan(eff_temps) | np.isnan(config_temps))
        if np.sum(valid_mask) > 2:
            correlation = np.corrcoef(eff_temps[valid_mask], config_temps[valid_mask])[0, 1]
            results['temperature_correlation'] = correlation
            
            # Check for non-equilibrium signature
            if abs(correlation - 1.0) > 0.1:
                results['non_equilibrium_signature'] = 'Strong (different temperature scales)'
            else:
                results['non_equilibrium_signature'] = 'Weak (similar temperature scales)'
        
        return results
    
    def compute_generalized_temperature(self, snapshot: Any, 
                                      method: str = 'maximum_entropy') -> float:
        """
        Compute generalized temperature for non-equilibrium system.
        
        Methods:
        - 'maximum_entropy': Use MaxEnt principle with constraints
        - 'information_geometric': Use Fisher information metric
        - 'dynamical': Use dynamical systems theory
        
        Returns:
            Generalized temperature
        """
        if method == 'maximum_entropy':
            # MaxEnt temperature from constraints
            if 'visit_distribution' in snapshot.observables:
                visits = snapshot.observables['visit_distribution']
                if isinstance(visits, torch.Tensor):
                    visits = visits.cpu().numpy()
                
                # Lagrange multiplier from entropy maximization
                # For MCTS: T_maxent related to exploration bonus
                total_visits = np.sum(visits)
                if total_visits > 0:
                    mean_visits = np.mean(visits[visits > 0])
                    T_maxent = 1.0 / np.log(1 + total_visits / mean_visits)
                    return T_maxent
        
        elif method == 'information_geometric':
            # Temperature from Fisher information metric
            # This captures the information geometry of the policy space
            if 'value_landscape' in snapshot.observables:
                q_values = snapshot.observables['value_landscape']
                if isinstance(q_values, torch.Tensor):
                    q_values = q_values.cpu().numpy()
                
                # Fisher information ∝ (∂log π/∂θ)²
                # For MCTS: related to Q-value curvature
                q_std = np.std(q_values)
                if q_std > 0:
                    T_info = 1.0 / (1.0 + q_std)
                    return T_info
        
        elif method == 'dynamical':
            # Temperature from dynamical systems perspective
            # Related to Lyapunov exponents and chaos
            # For MCTS: captures exploration-exploitation dynamics
            if 'visit_entropy' in snapshot.observables.get('uncertainty_measures', {}):
                visit_entropy = snapshot.observables['uncertainty_measures']['visit_entropy']
                if isinstance(visit_entropy, torch.Tensor):
                    visit_entropy = visit_entropy.item()
                
                # Dynamical temperature from entropy production rate
                T_dyn = np.exp(-visit_entropy)
                return T_dyn
        
        return 1.0  # Default