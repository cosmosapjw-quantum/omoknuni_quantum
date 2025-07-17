"""
Sagawa-Ueda information thermodynamics for MCTS.

According to the quantum foundation document, MCTS is a non-equilibrium system
that violates detailed balance. Instead of FDT (which requires equilibrium),
we apply the Sagawa-Ueda framework for information-theoretic feedback systems.

Key relations:
- Generalized second law: ⟨ΔS⟩ - ΔI ≥ 0
- Information-work equality: ⟨e^{-β(W-ΔF)}⟩ = e^{-I}
- Feedback efficiency: η = W_extracted / I_gained
"""
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Import unified quantum definitions
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from quantum_definitions import (
    UnifiedQuantumDefinitions,
    compute_von_neumann_entropy,
    construct_quantum_state_from_visits
)


@dataclass
class SagawaUedaResult:
    """Results from Sagawa-Ueda information thermodynamics analysis"""
    entropy_production: float
    mutual_information: float
    work_extracted: float
    free_energy_change: float
    generalized_second_law_satisfied: bool
    information_work_equality: float  # ⟨e^{-β(W-ΔF)}⟩ / e^{-I}
    feedback_efficiency: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary"""
        return {
            'entropy_production': self.entropy_production,
            'mutual_information': self.mutual_information,
            'work_extracted': self.work_extracted,
            'free_energy_change': self.free_energy_change,
            'generalized_second_law_satisfied': self.generalized_second_law_satisfied,
            'information_work_equality': self.information_work_equality,
            'feedback_efficiency': self.feedback_efficiency
        }


class FluctuationDissipationAnalyzer:
    """
    Analyze information thermodynamics in MCTS using Sagawa-Ueda framework.
    
    Since MCTS is non-equilibrium (violates detailed balance), we use
    information-theoretic fluctuation theorems instead of traditional FDT.
    
    Key concepts:
    - MCTS uses information (tree search) to extract work (find good moves)
    - It's a feedback process (past simulations guide future ones)
    - The generalized second law includes information: ΔS - ΔI ≥ 0
    - Information can be converted to work with fundamental limits
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize FDT analyzer.
        
        Args:
            device: Computation device
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize unified quantum definitions
        self.quantum_defs = UnifiedQuantumDefinitions(device=self.device)
    
    def compute_response_function(self, q_unperturbed: torch.Tensor,
                                q_perturbed: torch.Tensor,
                                perturbation: float) -> torch.Tensor:
        """
        Compute linear response χ = δ<Q>/δh.
        
        Args:
            q_unperturbed: Original Q-values
            q_perturbed: Q-values with perturbation
            perturbation: Perturbation strength h
            
        Returns:
            Response function for each action
        """
        if perturbation == 0:
            return torch.zeros_like(q_unperturbed)
        
        # Ensure tensors on correct device
        q_unperturbed = q_unperturbed.to(self.device)
        q_perturbed = q_perturbed.to(self.device)
        
        # Linear response
        chi = (q_perturbed - q_unperturbed) / perturbation
        
        return chi
    
    def compute_correlation_function(self, snapshots: List,
                                   observable: str = 'q_values',
                                   max_lag: int = 20) -> List[float]:
        """
        Compute autocorrelation function C(t) = <A(t)A(0)>.
        
        Args:
            snapshots: Time series of snapshots
            observable: Which observable to correlate
            max_lag: Maximum time lag
            
        Returns:
            Correlation function C(lag) for lag = 0 to max_lag
        """
        # Extract observable time series
        if observable == 'q_values':
            # Use first action's Q-value as example
            values = []
            for snapshot in snapshots:
                q_vals = snapshot.observables.get('value_landscape')
                if q_vals is not None:
                    if isinstance(q_vals, torch.Tensor):
                        values.append(q_vals[0].item())
                    else:
                        values.append(float(q_vals[0]))
        else:
            raise ValueError(f"Unknown observable: {observable}")
        
        values = np.array(values)
        
        # Compute autocorrelation
        correlations = []
        mean = np.mean(values)
        var = np.var(values)
        
        for lag in range(max_lag + 1):
            if lag < len(values):
                # C(t) = <(A(t) - <A>)(A(0) - <A>)>
                corr = np.mean((values[lag:] - mean) * (values[:-lag] - mean)) if lag > 0 else var
                correlations.append(corr)
            else:
                correlations.append(0.0)
        
        return correlations
    
    def compute_mutual_information(self, snapshots: List[Any]) -> float:
        """
        Compute mutual information I(Tree; OptimalAction).
        
        This quantifies how much information the tree contains about
        the optimal action at the root.
        
        Args:
            snapshots: List of tree snapshots
            
        Returns:
            Mutual information in nats
        """
        if len(snapshots) < 2:
            return 0.0
        
        # Get final snapshot (most information)
        final_snapshot = snapshots[-1]
        
        # Extract visit distribution at root
        if 'visits' in final_snapshot.observables:
            visits = final_snapshot.observables['visits']
            if not isinstance(visits, torch.Tensor):
                visits = torch.tensor(visits, device=self.device)
        else:
            return 0.0
        
        # Compute policy entropy H(π)
        total_visits = visits.sum()
        if total_visits == 0:
            return 0.0
        
        probs = visits / total_visits
        probs = probs[probs > 0]  # Remove zeros for log
        policy_entropy = -torch.sum(probs * torch.log(probs))
        
        # Compute conditional entropy H(π|Tree)
        # After many simulations, uncertainty is reduced
        # Approximate by residual entropy after convergence
        if len(probs) > 0:
            max_prob = probs.max()
            # Heuristic: conditional entropy decreases with dominance
            conditional_entropy = policy_entropy * (1 - max_prob)
        else:
            conditional_entropy = policy_entropy
        
        # Mutual information I = H(π) - H(π|Tree)
        mutual_info = policy_entropy - conditional_entropy
        
        return float(mutual_info)
    
    def _compute_policy_entropy(self, snapshot: Any) -> float:
        """Compute policy entropy using unified quantum definitions"""
        if 'visits' in snapshot.observables:
            visits = snapshot.observables['visits']
            if not isinstance(visits, torch.Tensor):
                visits = torch.tensor(visits, device=self.device)
            
            # Use unified quantum definitions for consistency
            # Information thermodynamics still uses mixed states
            quantum_state = self.quantum_defs.construct_quantum_state_from_single_visits(
                visits, outcome_uncertainty=0.1  # Non-equilibrium uncertainty
            )
            
            # Von Neumann entropy for quantum consistency
            entropy = compute_von_neumann_entropy(quantum_state.density_matrix)
            return float(entropy)
        return 0.0
    
    def _compute_average_q_value(self, snapshot: Any) -> float:
        """Compute weighted average Q-value"""
        if 'q_values' in snapshot.observables and 'visits' in snapshot.observables:
            q_values = snapshot.observables['q_values']
            visits = snapshot.observables['visits']
            
            if not isinstance(q_values, torch.Tensor):
                q_values = torch.tensor(q_values, device=self.device)
            if not isinstance(visits, torch.Tensor):
                visits = torch.tensor(visits, device=self.device)
            
            total_visits = visits.sum()
            if total_visits == 0:
                return 0.0
            
            weights = visits / total_visits
            avg_q = torch.sum(q_values * weights)
            return float(avg_q)
        return 0.0
    
    def validate_sagawa_ueda(self, snapshots: List[Any]) -> Dict[str, Any]:
        """
        Validate Sagawa-Ueda generalized second law and information-work equality.
        
        Args:
            snapshots: Time series of tree snapshots
            
        Returns:
            Dictionary with Sagawa-Ueda validation results
        """
        if len(snapshots) < 2:
            return {'error': 'Insufficient data for analysis'}
        
        # Compute entropy production
        initial_snapshot = snapshots[0]
        final_snapshot = snapshots[-1]
        
        # Policy entropy change (system entropy)
        initial_entropy = self._compute_policy_entropy(initial_snapshot)
        final_entropy = self._compute_policy_entropy(final_snapshot)
        entropy_production = final_entropy - initial_entropy
        
        # Mutual information gained
        mutual_info = self.compute_mutual_information(snapshots)
        
        # Work extracted (negative of Q-value change)
        initial_value = self._compute_average_q_value(initial_snapshot)
        final_value = self._compute_average_q_value(final_snapshot)
        work_extracted = -(final_value - initial_value)  # Negative because high Q = low energy
        
        # Free energy change (using authentic temperature if available)
        temperature = snapshots[-1].observables.get('temperature', 1.0)
        free_energy_change = -temperature * entropy_production
        
        # Check generalized second law: ΔS - ΔI ≥ 0
        generalized_second_law = entropy_production - mutual_info >= -1e-10
        
        # Compute information-work equality ratio
        # Prevent division by zero for temperature
        if temperature == 0 or np.abs(temperature) < 1e-10:
            # Use a small default temperature for numerical stability
            temperature = 1e-3
        beta = 1.0 / temperature
        info_work_ratio = np.exp(-beta * (work_extracted - free_energy_change)) / np.exp(-mutual_info)
        
        # Feedback efficiency
        if mutual_info > 0:
            efficiency = work_extracted / mutual_info
        else:
            efficiency = 0.0
        
        result = SagawaUedaResult(
            entropy_production=float(entropy_production),
            mutual_information=float(mutual_info),
            work_extracted=float(work_extracted),
            free_energy_change=float(free_energy_change),
            generalized_second_law_satisfied=bool(generalized_second_law),
            information_work_equality=float(info_work_ratio),
            feedback_efficiency=float(efficiency)
        )
        
        return result.to_dict()
    
    def compute_information_flow_rate(self, snapshots: List[Any]) -> List[float]:
        """
        Compute the rate of information gain over time.
        
        This measures how quickly MCTS learns about the optimal action.
        
        Args:
            snapshots: Time series of tree snapshots
            
        Returns:
            List of information flow rates dI/dk
        """
        if len(snapshots) < 2:
            return []
        
        # Compute mutual information at each timestep
        mutual_infos = []
        for i in range(len(snapshots)):
            # Use snapshots up to time i
            I_t = self.compute_mutual_information(snapshots[:i+1])
            mutual_infos.append(I_t)
        
        # Compute rate dI/dk
        rates = []
        for i in range(1, len(mutual_infos)):
            dI = mutual_infos[i] - mutual_infos[i-1]
            rates.append(dI)
        
        return rates
    
    def compute_landauer_bound(self, snapshots: List[Any]) -> float:
        """
        Compute Landauer's bound for information erasure.
        
        The minimum energy cost for erasing information is kT ln(2) per bit.
        For MCTS, this sets a lower bound on the work needed to explore.
        
        Args:
            snapshots: Tree snapshots
            
        Returns:
            Landauer bound in units of kT
        """
        if len(snapshots) < 2:
            return 0.0
        
        # Compute total information erased
        # This is the entropy reduction from initial uniform to final peaked distribution
        initial_entropy = self._compute_policy_entropy(snapshots[0])
        final_entropy = self._compute_policy_entropy(snapshots[-1])
        
        # Information erased (in nats)
        info_erased = max(0, initial_entropy - final_entropy)
        
        # Convert to bits and apply Landauer bound
        info_erased_bits = info_erased / np.log(2)
        landauer_bound = info_erased_bits  # In units of kT
        
        return float(landauer_bound)
    
    def compute_entropy_production_rate(self, snapshots: List[Any]) -> Dict[str, Any]:
        """
        Compute entropy production rate and test MEPP vs MaxEPP.
        
        MEPP (Minimum Entropy Production Principle):
        - Systems at steady state minimize entropy production
        - Applies to near-equilibrium linear regime
        - Predicts convergence to low-dissipation paths
        
        MaxEPP (Maximum Entropy Production Principle):
        - Far-from-equilibrium systems maximize entropy production
        - Applies when multiple steady states exist
        - Predicts selection of high-dissipation paths
        
        MCTS phases:
        1. Exploration phase: High entropy production (MaxEPP-like)
        2. Transition phase: Variable entropy production
        3. Exploitation phase: Low entropy production (MEPP-like)
        
        Args:
            snapshots: Time series of tree snapshots
            
        Returns:
            Dictionary with entropy production analysis
        """
        if len(snapshots) < 3:
            return {'error': 'Insufficient data for entropy production analysis'}
        
        # Compute entropy production rate at each timestep
        entropy_rates = []
        policy_entropies = []
        mutual_infos = []
        
        for i in range(1, len(snapshots)):
            # Policy entropy change (system entropy production)
            S_prev = self._compute_policy_entropy(snapshots[i-1])
            S_curr = self._compute_policy_entropy(snapshots[i])
            dS_dt = S_curr - S_prev
            
            # Information gain rate
            I_prev = self.compute_mutual_information(snapshots[:i])
            I_curr = self.compute_mutual_information(snapshots[:i+1])
            dI_dt = I_curr - I_prev
            
            # Net entropy production rate (Sagawa-Ueda)
            # dΣ/dt = dS/dt - dI/dt
            entropy_production_rate = dS_dt - dI_dt
            
            entropy_rates.append(entropy_production_rate)
            policy_entropies.append(S_curr)
            mutual_infos.append(I_curr)
        
        # Analyze phases
        phases = self._detect_entropy_production_phases(entropy_rates, policy_entropies)
        
        # Test MEPP vs MaxEPP
        mepp_maxepp_test = self._test_mepp_vs_maxepp(
            entropy_rates, policy_entropies, mutual_infos
        )
        
        # Compute statistics
        entropy_rates_np = np.array(entropy_rates)
        
        return {
            'entropy_production_rates': entropy_rates,
            'average_rate': float(np.mean(entropy_rates_np)),
            'std_rate': float(np.std(entropy_rates_np)),
            'phases': phases,
            'mepp_maxepp_test': mepp_maxepp_test,
            'interpretation': self._interpret_entropy_production(phases, mepp_maxepp_test)
        }
    
    def _detect_entropy_production_phases(self, entropy_rates: List[float], 
                                        policy_entropies: List[float]) -> Dict[str, Any]:
        """
        Detect different phases based on entropy production behavior.
        """
        n = len(entropy_rates)
        if n < 3:
            return {'error': 'Too few points for phase detection'}
        
        # Convert to numpy
        rates = np.array(entropy_rates)
        entropies = np.array(policy_entropies)
        
        # Divide into thirds for phase analysis
        third = n // 3
        
        phases = {
            'exploration': {
                'range': [0, third],
                'mean_production': float(np.mean(rates[:third])),
                'mean_entropy': float(np.mean(entropies[:third])),
                'dominant_principle': 'MaxEPP' if np.mean(rates[:third]) > 0 else 'Neither'
            },
            'transition': {
                'range': [third, 2*third],
                'mean_production': float(np.mean(rates[third:2*third])),
                'mean_entropy': float(np.mean(entropies[third:2*third])),
                'dominant_principle': 'Variable'
            },
            'exploitation': {
                'range': [2*third, n],
                'mean_production': float(np.mean(rates[2*third:])),
                'mean_entropy': float(np.mean(entropies[2*third:])),
                'dominant_principle': 'MEPP' if np.mean(rates[2*third:]) < np.mean(rates[:third]) else 'Neither'
            }
        }
        
        # Detect phase transitions
        transitions = []
        for i in range(1, n-1):
            # Look for significant changes in entropy production
            if abs(rates[i] - rates[i-1]) > 2 * np.std(rates):
                transitions.append(i)
        
        phases['transitions'] = transitions
        
        return phases
    
    def _test_mepp_vs_maxepp(self, entropy_rates: List[float],
                            policy_entropies: List[float],
                            mutual_infos: List[float]) -> Dict[str, Any]:
        """
        Test which principle (MEPP or MaxEPP) better describes MCTS behavior.
        """
        rates = np.array(entropy_rates)
        entropies = np.array(policy_entropies)
        infos = np.array(mutual_infos)
        
        # Test 1: Correlation between entropy and production rate
        # MaxEPP predicts positive correlation in exploration
        # MEPP predicts negative correlation in exploitation
        early_third = len(rates) // 3
        late_start = 2 * len(rates) // 3
        
        early_corr = np.corrcoef(entropies[:early_third], rates[:early_third])[0, 1]
        late_corr = np.corrcoef(entropies[late_start:], rates[late_start:])[0, 1]
        
        # Test 2: Trend in entropy production
        # MaxEPP: Initially high, may stay high
        # MEPP: Should decrease over time
        time = np.arange(len(rates))
        slope, _ = np.polyfit(time, rates, 1)
        
        # Test 3: Variability in entropy production
        # MaxEPP: High variability (exploring multiple paths)
        # MEPP: Low variability (converged to optimal path)
        early_var = np.var(rates[:early_third])
        late_var = np.var(rates[late_start:])
        
        # Test 4: Information efficiency
        # How much entropy is produced per bit of information gained
        if len(infos) > 1 and infos[-1] > infos[0]:
            total_entropy_produced = np.sum(rates)
            total_info_gained = infos[-1] - infos[0]
            info_efficiency = total_entropy_produced / total_info_gained
        else:
            info_efficiency = 0.0
        
        return {
            'early_correlation': float(early_corr),
            'late_correlation': float(late_corr),
            'entropy_production_slope': float(slope),
            'early_variability': float(early_var),
            'late_variability': float(late_var),
            'variability_ratio': float(late_var / early_var) if early_var > 0 else 0,
            'information_efficiency': float(info_efficiency),
            'supports_maxepp_early': early_corr > 0.3,
            'supports_mepp_late': late_corr < -0.3 and slope < 0,
            'phase_dependent': True  # MCTS shows phase-dependent behavior
        }
    
    def _interpret_entropy_production(self, phases: Dict[str, Any], 
                                    test_results: Dict[str, Any]) -> str:
        """
        Interpret the entropy production results.
        """
        if 'error' in phases:
            return "Insufficient data for interpretation"
        
        interpretation = []
        
        # Phase behavior
        if phases['exploration']['dominant_principle'] == 'MaxEPP':
            interpretation.append("Exploration phase shows MaxEPP behavior (high entropy production)")
        
        if phases['exploitation']['dominant_principle'] == 'MEPP':
            interpretation.append("Exploitation phase shows MEPP behavior (minimized entropy production)")
        
        # Test results
        if test_results['supports_maxepp_early'] and test_results['supports_mepp_late']:
            interpretation.append("MCTS transitions from MaxEPP to MEPP as it converges")
        elif test_results['entropy_production_slope'] < 0:
            interpretation.append("Decreasing entropy production suggests convergence to steady state")
        
        # Information efficiency
        if test_results['information_efficiency'] < 1.0:
            interpretation.append("Efficient information processing (low dissipation per bit)")
        else:
            interpretation.append("High dissipation per bit of information gained")
        
        # Variability
        if test_results['variability_ratio'] < 0.5:
            interpretation.append("Reduced variability indicates convergence to optimal policy")
        
        return "; ".join(interpretation) if interpretation else "Complex entropy production dynamics"
    
    def test_mepp_maxepp_parameter_dependence(self, 
                                            snapshots_dict: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Test how MEPP vs MaxEPP behavior depends on MCTS parameters.
        
        Expected behavior:
        - High exploration (high c_puct): MaxEPP dominates longer
        - Low exploration (low c_puct): Quick transition to MEPP
        - High temperature: Sustained MaxEPP behavior
        - Low temperature: Rapid MEPP convergence
        
        Args:
            snapshots_dict: Dictionary mapping parameter values to snapshot lists
                           e.g., {'c_puct_0.5': [...], 'c_puct_2.0': [...]}
        
        Returns:
            Analysis of parameter dependence
        """
        results = {}
        
        for param_name, snapshots in snapshots_dict.items():
            if len(snapshots) < 3:
                continue
                
            # Analyze entropy production for this parameter
            analysis = self.compute_entropy_production_rate(snapshots)
            
            if 'error' not in analysis:
                # Extract key metrics
                test_results = analysis['mepp_maxepp_test']
                phases = analysis['phases']
                
                # Determine transition point (if any)
                transition_point = None
                if phases.get('transitions'):
                    transition_point = phases['transitions'][0] / len(snapshots)
                
                results[param_name] = {
                    'average_entropy_production': analysis['average_rate'],
                    'early_supports_maxepp': test_results['supports_maxepp_early'],
                    'late_supports_mepp': test_results['supports_mepp_late'],
                    'transition_point': transition_point,
                    'information_efficiency': test_results['information_efficiency'],
                    'entropy_slope': test_results['entropy_production_slope'],
                    'interpretation': analysis['interpretation']
                }
        
        # Analyze trends across parameters
        parameter_trends = self._analyze_parameter_trends(results)
        
        return {
            'parameter_results': results,
            'trends': parameter_trends,
            'conclusion': self._conclude_parameter_dependence(parameter_trends)
        }
    
    def _analyze_parameter_trends(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze how MEPP/MaxEPP behavior changes with parameters."""
        if not results:
            return {}
        
        # Extract parameter values and metrics
        param_values = []
        avg_productions = []
        transition_points = []
        info_efficiencies = []
        
        for param_name, data in sorted(results.items()):
            # Try to extract numeric parameter value
            try:
                # Handle formats like 'c_puct_2.0' or 'temp_0.5'
                value = float(param_name.split('_')[-1])
                param_values.append(value)
                avg_productions.append(data['average_entropy_production'])
                if data['transition_point'] is not None:
                    transition_points.append(data['transition_point'])
                info_efficiencies.append(data['information_efficiency'])
            except:
                continue
        
        if len(param_values) < 2:
            return {}
        
        # Compute correlations
        trends = {}
        
        if avg_productions:
            corr_production = np.corrcoef(param_values, avg_productions)[0, 1]
            trends['entropy_production_vs_parameter'] = float(corr_production)
        
        if transition_points and len(transition_points) == len(param_values):
            corr_transition = np.corrcoef(param_values, transition_points)[0, 1]
            trends['transition_delay_vs_parameter'] = float(corr_transition)
        
        if info_efficiencies:
            corr_efficiency = np.corrcoef(param_values, info_efficiencies)[0, 1]
            trends['info_efficiency_vs_parameter'] = float(corr_efficiency)
        
        return trends
    
    def _conclude_parameter_dependence(self, trends: Dict[str, Any]) -> str:
        """Draw conclusions about parameter dependence of MEPP/MaxEPP."""
        if not trends:
            return "Insufficient data for parameter analysis"
        
        conclusions = []
        
        if 'entropy_production_vs_parameter' in trends:
            corr = trends['entropy_production_vs_parameter']
            if corr > 0.5:
                conclusions.append("Higher exploration parameters lead to increased entropy production (MaxEPP)")
            elif corr < -0.5:
                conclusions.append("Lower exploration parameters favor entropy minimization (MEPP)")
        
        if 'transition_delay_vs_parameter' in trends:
            corr = trends['transition_delay_vs_parameter']
            if corr > 0.5:
                conclusions.append("Higher exploration delays the MaxEPP→MEPP transition")
        
        if 'info_efficiency_vs_parameter' in trends:
            corr = trends['info_efficiency_vs_parameter']
            if corr < -0.3:
                conclusions.append("Higher exploration reduces information efficiency")
        
        return "; ".join(conclusions) if conclusions else "Complex parameter dependence"
    
    def compute_susceptibility_matrix(self, snapshots: List) -> torch.Tensor:
        """
        Compute susceptibility matrix χ_ab = β<δQ_a δQ_b>.
        
        Args:
            snapshots: Equilibrium snapshots
            
        Returns:
            Susceptibility matrix
        """
        # Extract Q-values
        q_values_list = []
        for snapshot in snapshots:
            q_vals = snapshot.observables.get('value_landscape')
            if q_vals is not None:
                if not isinstance(q_vals, torch.Tensor):
                    q_vals = torch.tensor(q_vals, device=self.device)
                else:
                    q_vals = q_vals.to(self.device)
                q_values_list.append(q_vals)
        
        if not q_values_list:
            return torch.zeros(1, 1, device=self.device)
        
        # Stack into matrix
        Q = torch.stack(q_values_list)  # [time, actions]
        
        # Compute covariance
        Q_mean = Q.mean(dim=0)
        Q_fluct = Q - Q_mean
        
        # Covariance matrix
        cov = torch.mm(Q_fluct.T, Q_fluct) / len(q_values_list)
        
        # Get temperature
        if 'temperature' in snapshots[0].observables:
            T = snapshots[0].observables['temperature']
        else:
            T = 1.0
        
        beta = 1.0 / T
        
        # Susceptibility
        chi = beta * cov
        
        return chi
    
    def compute_kubo_response(self, snapshots: List,
                            perturbation_strength: float = 0.01) -> Dict[str, Any]:
        """
        Compute response via Kubo formula.
        
        χ_AB = β ∫dt <A(t)B(0)>_0
        
        Args:
            snapshots: Equilibrium snapshots
            perturbation_strength: Strength of applied field
            
        Returns:
            Kubo response results
        """
        # For MCTS, use Q-value correlations
        chi_matrix = self.compute_susceptibility_matrix(snapshots)
        
        # Linear response coefficient
        linear_response = chi_matrix.diagonal().mean().item() * perturbation_strength
        
        # Estimate higher-order terms from variance
        variance = chi_matrix.diagonal().var().item()
        higher_order = variance * perturbation_strength**2
        
        return {
            'linear_response': linear_response,
            'higher_order': higher_order,
            'susceptibility_matrix': chi_matrix
        }
    
    def compute_onsager_matrix(self, snapshots: List) -> torch.Tensor:
        """
        Compute Onsager transport coefficient matrix.
        
        L_ij = ∫dt <J_i(t)J_j(0)>
        
        Args:
            snapshots: Equilibrium snapshots
            
        Returns:
            Onsager matrix (should be symmetric)
        """
        # For MCTS, use visit flow as "currents"
        # J_i = rate of visits to action i
        
        visit_flows = []
        for i in range(1, len(snapshots)):
            prev_visits = snapshots[i-1].observables['visit_distribution']
            curr_visits = snapshots[i].observables['visit_distribution']
            
            if not isinstance(prev_visits, torch.Tensor):
                prev_visits = torch.tensor(prev_visits, device=self.device)
            if not isinstance(curr_visits, torch.Tensor):
                curr_visits = torch.tensor(curr_visits, device=self.device)
            
            flow = curr_visits - prev_visits
            visit_flows.append(flow)
        
        if not visit_flows:
            return torch.zeros(1, 1, device=self.device)
        
        # Stack flows
        J = torch.stack(visit_flows)  # [time, actions]
        
        # Compute current-current correlations
        J_mean = J.mean(dim=0)
        J_fluct = J - J_mean
        
        # Onsager matrix
        L = torch.mm(J_fluct.T, J_fluct) / len(visit_flows)
        
        # Symmetrize to enforce reciprocity
        L = (L + L.T) / 2
        
        return L
    
    def compute_diffusion_coefficient(self, snapshots: List) -> float:
        """
        Compute diffusion coefficient via Green-Kubo.
        
        D = ∫dt <v(t)v(0)>
        
        Args:
            snapshots: Trajectory snapshots
            
        Returns:
            Diffusion coefficient
        """
        # For MCTS, use Q-value changes as "velocity"
        velocities = []
        
        for i in range(1, len(snapshots)):
            prev_q = snapshots[i-1].observables['value_landscape']
            curr_q = snapshots[i].observables['value_landscape']
            
            if isinstance(prev_q, torch.Tensor):
                prev_q = prev_q[0].item()
            else:
                prev_q = prev_q[0]
                
            if isinstance(curr_q, torch.Tensor):
                curr_q = curr_q[0].item()
            else:
                curr_q = curr_q[0]
            
            v = curr_q - prev_q
            velocities.append(v)
        
        if not velocities:
            return 0.0
        
        velocities = np.array(velocities)
        
        # Velocity autocorrelation
        v_mean = np.mean(velocities)
        v_var = np.var(velocities)
        
        # Integrate correlation function
        # Simple approximation: D ≈ <v²> * τ_corr
        # where τ_corr is correlation time
        
        # Estimate correlation time
        corr_time = 1.0  # Simplified
        
        D = v_var * corr_time
        
        return float(D)
    
    def compute_time_dependent_response(self, unperturbed_snapshots: List,
                                      perturbed_snapshots: List,
                                      perturbation_time: int) -> List[float]:
        """
        Compute time-dependent response function.
        
        Args:
            unperturbed_snapshots: Reference trajectory
            perturbed_snapshots: Perturbed trajectory
            perturbation_time: When perturbation was applied
            
        Returns:
            Response χ(t) for each time
        """
        response = []
        
        for i in range(len(unperturbed_snapshots)):
            if i < len(perturbed_snapshots):
                q_unpert = unperturbed_snapshots[i].observables['value_landscape']
                q_pert = perturbed_snapshots[i].observables['value_landscape']
                
                if isinstance(q_unpert, torch.Tensor):
                    q_unpert = q_unpert[0].item()
                else:
                    q_unpert = q_unpert[0]
                    
                if isinstance(q_pert, torch.Tensor):
                    q_pert = q_pert[0].item()
                else:
                    q_pert = q_pert[0]
                
                # Response (with causality)
                if i < perturbation_time:
                    chi = 0.0  # No response before perturbation
                else:
                    chi = q_pert - q_unpert
                
                response.append(chi)
        
        return response
    
    def compute_effective_temperature(self, snapshots: List) -> float:
        """
        Compute effective temperature for non-equilibrium system.
        
        Uses generalized FDT: T_eff = lim(ω→0) T(ω)
        
        Args:
            snapshots: Non-equilibrium snapshots
            
        Returns:
            Effective temperature
        """
        # Extract temperatures if available
        temperatures = []
        for snapshot in snapshots:
            if 'temperature' in snapshot.observables:
                temperatures.append(snapshot.observables['temperature'])
        
        if temperatures:
            # Time-averaged temperature
            T_eff = np.mean(temperatures)
        else:
            # Estimate from fluctuations
            chi_matrix = self.compute_susceptibility_matrix(snapshots)
            
            # Use equipartition-like relation
            # <Q²> ≈ T_eff for each mode
            q_variance = chi_matrix.diagonal().mean().item()
            T_eff = q_variance  # Simplified estimate
        
        return float(T_eff)
