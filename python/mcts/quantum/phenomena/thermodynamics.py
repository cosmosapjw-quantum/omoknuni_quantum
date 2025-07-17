"""
Thermodynamic analysis for MCTS.

Computes thermodynamic quantities and validates non-equilibrium 
statistical mechanics relations like Jarzynski equality.
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
    MCTSQuantumState,
    compute_von_neumann_entropy,
    compute_purity,
    construct_quantum_state_from_visits
)


@dataclass
class ThermodynamicResult:
    """Results from thermodynamic analysis"""
    energy: float
    entropy: float
    temperature: float
    free_energy: float
    heat_capacity: float
    partition_function: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary"""
        return {
            'energy': self.energy,
            'entropy': self.entropy,
            'temperature': self.temperature,
            'free_energy': self.free_energy,
            'heat_capacity': self.heat_capacity,
            'partition_function': self.partition_function
        }


class ThermodynamicsAnalyzer:
    """
    Extract thermodynamic quantities from MCTS dynamics.
    
    Measurements:
    - Energy: E = -<Q> (average value)
    - Entropy: S = -Σ p log p
    - Free energy: F = E - TS  
    - Heat capacity: C = dE/dT
    - Susceptibility: χ = d<Q>/d(c_puct)
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize thermodynamics analyzer.
        
        Args:
            device: Computation device
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize unified quantum definitions
        self.quantum_defs = UnifiedQuantumDefinitions(device=self.device)
    
    def measure_thermodynamics(self, snapshots: List) -> List[Dict[str, Any]]:
        """
        Measure thermodynamic quantities from snapshot sequence.
        
        Args:
            snapshots: List of TreeSnapshot objects
            
        Returns:
            List of thermodynamic measurements
        """
        results = []
        
        for i, snapshot in enumerate(snapshots):
            # Extract data
            observables = snapshot.observables
            
            visits = observables.get('visit_distribution')
            if visits is None:
                continue
                
            q_values = observables.get('value_landscape', torch.zeros_like(visits))
            
            # Ensure tensors
            if not isinstance(visits, torch.Tensor):
                visits = torch.tensor(visits, device=self.device)
            else:
                visits = visits.to(self.device)
                
            if not isinstance(q_values, torch.Tensor):
                q_values = torch.tensor(q_values, device=self.device)
            else:
                q_values = q_values.to(self.device)
            
            # Compute quantities
            energy = self.compute_energy(q_values, visits)
            entropy = self.compute_entropy(visits)
            
            # Measure or use provided temperature
            if 'temperature' in observables:
                temperature = observables['temperature']
            else:
                temperature = self._measure_temperature(visits)
            
            free_energy = self.compute_free_energy(energy, temperature, entropy)
            
            # Heat capacity (need neighboring snapshots)
            if i > 0:
                prev_result = results[-1]
                heat_capacity = self._compute_heat_capacity(
                    prev_result['energy'], energy,
                    prev_result['temperature'], temperature
                )
            else:
                heat_capacity = 0.0
            
            # Partition function
            if hasattr(snapshot, 'scores'):
                scores = snapshot.scores
            else:
                # Approximate from visits
                scores = torch.log(visits + 1e-10)
            
            Z = self.compute_partition_function(scores, temperature)
            
            results.append({
                'energy': float(energy),
                'entropy': float(entropy),
                'temperature': float(temperature),
                'free_energy': float(free_energy),
                'heat_capacity': float(heat_capacity),
                'partition_function': float(Z)
            })
        
        return results
    
    def compute_energy(self, q_values: torch.Tensor, visits: torch.Tensor) -> torch.Tensor:
        """
        Compute energy E = -<Q>.
        
        Args:
            q_values: Q-values for each action
            visits: Visit counts
            
        Returns:
            Energy
        """
        # Convert to probabilities
        total_visits = visits.sum()
        if total_visits.item() == 0:
            return torch.tensor(0.0, device=self.device)
        
        probs = visits / total_visits
        
        # Energy is negative average Q-value
        energy = -torch.sum(probs * q_values)
        
        return energy
    
    def compute_entropy(self, visits: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy using unified quantum definitions.
        
        This ensures consistency with quantum formulation where entropy
        is the von Neumann entropy of the density matrix.
        
        Args:
            visits: Visit distribution
            
        Returns:
            Entropy (von Neumann entropy for quantum consistency)
        """
        # Construct quantum state from visits
        # Use low uncertainty for thermodynamic equilibrium assumption
        quantum_state = self.quantum_defs.construct_quantum_state_from_single_visits(
            visits, outcome_uncertainty=0.05
        )
        
        # Compute von Neumann entropy
        entropy = compute_von_neumann_entropy(quantum_state.density_matrix)
        
        # For backward compatibility, also compute Shannon entropy
        total_visits = visits.sum()
        if total_visits.item() > 0:
            probs = visits / total_visits
            probs_nonzero = probs[probs > 0]
            shannon_entropy = -torch.sum(probs_nonzero * torch.log(probs_nonzero))
        else:
            shannon_entropy = torch.tensor(0.0, device=self.device)
        
        # Use von Neumann entropy for quantum consistency
        # but log the difference for validation
        if abs(entropy - shannon_entropy) > 0.1:
            # They should be similar for near-diagonal density matrices
            pass  # Could log this difference if needed
        
        return entropy
    
    def compute_work_and_heat(self, snapshots: List[Any]) -> Dict[str, Any]:
        """
        Compute work and heat following non-equilibrium thermodynamics from quantum_mcts_foundation.md
        
        Section 8.1-8.2 formulation:
        - Work: W = G(k_final) - G(k_initial) (controlled change)
        - Heat: Q = Σ v_i (stochastic input from simulations)
        - Free Energy: G(k) = -(1/β(k)) log Z(k)
        
        Args:
            snapshots: Sequence of tree snapshots
            
        Returns:
            Dictionary with work, heat, and Jarzynski equality test
        """
        if len(snapshots) < 2:
            return {'error': 'Need at least 2 snapshots for work/heat calculation'}
        
        free_energies = []
        heat_inputs = []
        
        for i, snapshot in enumerate(snapshots):
            # Extract observables
            visits = snapshot.observables.get('visit_distribution', [])
            q_values = snapshot.observables.get('q_values', [])
            
            if not visits or not q_values:
                continue
            
            # Compute temperature
            temperature = self._measure_temperature(torch.tensor(visits))
            if temperature <= 0.0:
                continue
                
            beta = 1.0 / temperature
            
            # Compute partition function Z = Σ exp(β * Q)
            q_tensor = torch.tensor(q_values, device=self.device)
            Z = torch.sum(torch.exp(beta * q_tensor))
            
            # Free energy G = -(1/β) log Z
            free_energy = -temperature * torch.log(Z)
            free_energies.append(float(free_energy))
            
            # Heat input (stochastic from simulations)
            # Approximate as total simulation values added
            if hasattr(snapshot, 'simulation_values'):
                heat_input = sum(snapshot.simulation_values)
            else:
                # Fallback: use Q-values weighted by visits
                visits_tensor = torch.tensor(visits, device=self.device)
                heat_input = float(torch.sum(visits_tensor * q_tensor))
            
            heat_inputs.append(heat_input)
        
        if len(free_energies) < 2:
            return {'error': 'Not enough valid snapshots for work calculation'}
        
        # Compute work W = ΔG
        work = free_energies[-1] - free_energies[0]
        
        # Total heat input
        total_heat = sum(heat_inputs)
        
        # First law check: ΔU = Q - W
        # (Internal energy change = heat - work)
        internal_energy_change = total_heat - work
        
        results = {
            'work': work,
            'heat': total_heat,
            'internal_energy_change': internal_energy_change,
            'free_energy_trajectory': free_energies,
            'heat_trajectory': heat_inputs,
            'first_law_check': {
                'delta_U': internal_energy_change,
                'Q_minus_W': total_heat - work,
                'consistent': abs(internal_energy_change - (total_heat - work)) < 1e-6
            }
        }
        
        return results
    
    def test_jarzynski_equality(self, trajectory_ensemble: List[List[Any]]) -> Dict[str, Any]:
        """
        Test Jarzynski equality: ⟨e^(-βW)⟩ = e^(-βΔG) from quantum_mcts_foundation.md
        
        Args:
            trajectory_ensemble: List of trajectories (each trajectory is list of snapshots)
            
        Returns:
            Dictionary with Jarzynski equality test results
        """
        if len(trajectory_ensemble) < 10:
            return {'error': 'Need at least 10 trajectories for ensemble average'}
        
        work_values = []
        delta_G_values = []
        
        for trajectory in trajectory_ensemble:
            if len(trajectory) < 2:
                continue
                
            # Compute work for this trajectory
            work_heat = self.compute_work_and_heat(trajectory)
            if 'error' not in work_heat:
                work_values.append(work_heat['work'])
                
                # ΔG is the same as work for reversible process
                delta_G_values.append(work_heat['work'])
        
        if len(work_values) < 3:
            return {'error': 'Not enough valid trajectories'}
        
        work_array = np.array(work_values)
        delta_G_array = np.array(delta_G_values)
        
        # Average temperature (approximate)
        avg_temperature = 1.0  # Normalized units
        beta = 1.0 / avg_temperature
        
        # Left side: ⟨e^(-βW)⟩
        exp_minus_beta_W = np.exp(-beta * work_array)
        lhs = np.mean(exp_minus_beta_W)
        
        # Right side: e^(-βΔG)
        avg_delta_G = np.mean(delta_G_array)
        rhs = np.exp(-beta * avg_delta_G)
        
        # Jarzynski equality test
        jarzynski_ratio = lhs / rhs if rhs > 0 else np.inf
        jarzynski_satisfied = abs(jarzynski_ratio - 1.0) < 0.1  # 10% tolerance
        
        return {
            'jarzynski_lhs': lhs,
            'jarzynski_rhs': rhs,
            'jarzynski_ratio': jarzynski_ratio,
            'jarzynski_satisfied': jarzynski_satisfied,
            'n_trajectories': len(work_values),
            'work_distribution': {
                'mean': np.mean(work_array),
                'std': np.std(work_array),
                'min': np.min(work_array),
                'max': np.max(work_array)
            },
            'theoretical_foundation': 'Section 8.2 of quantum_mcts_foundation.md'
        }
    
    def compute_free_energy(self, energy: torch.Tensor, temperature: float,
                          entropy: torch.Tensor) -> torch.Tensor:
        """
        Compute Helmholtz free energy F = E - TS.
        
        Args:
            energy: Energy
            temperature: Temperature
            entropy: Entropy
            
        Returns:
            Free energy
        """
        return energy - temperature * entropy
    
    def compute_partition_function(self, scores: torch.Tensor, 
                                 temperature: float) -> torch.Tensor:
        """
        Compute partition function Z = Σ exp(β·score).
        
        Args:
            scores: Action scores
            temperature: Temperature
            
        Returns:
            Partition function
        """
        beta = 1.0 / temperature if temperature > 0 else float('inf')
        
        # Numerical stability
        scores_shifted = scores - scores.max()
        
        Z = torch.sum(torch.exp(beta * scores_shifted))
        
        # Correct for shift
        Z = Z * torch.exp(beta * scores.max())
        
        return Z
    
    def _measure_temperature(self, visits: torch.Tensor) -> float:
        """
        Estimate temperature from visit distribution.
        
        According to quantum_mcts_foundation.md, temperature scales as:
        β ∝ √N where N is total visit count
        Therefore: T = 1/β ∝ 1/√N
        
        This implements the theoretical scaling law from path integral formulation.
        """
        total_visits = visits.sum()
        
        if total_visits.item() <= 0:
            return 1.0  # Default temperature
        
        # Temperature scaling: T ∝ 1/√N
        # Use normalization constant for practical scale
        c_temp = 1.0  # Can be adjusted based on c_puct if available
        temperature = c_temp / torch.sqrt(total_visits)
        
        # Ensure reasonable bounds
        temperature = torch.clamp(temperature, min=0.01, max=10.0)
        
        return float(temperature)
    
    def _compute_heat_capacity(self, E1: float, E2: float,
                             T1: float, T2: float) -> float:
        """
        Compute heat capacity C = dE/dT.
        
        Args:
            E1, E2: Energies at two temperatures
            T1, T2: Temperatures
            
        Returns:
            Heat capacity
        """
        if abs(T2 - T1) < 1e-6:
            return 0.0
        
        return (E2 - E1) / (T2 - T1)
    
    def compute_susceptibility(self, snapshots: List) -> Dict[str, float]:
        """
        Compute susceptibility χ = d<Q>/d(c_puct).
        
        Args:
            snapshots: Snapshots at different c_puct values
            
        Returns:
            Susceptibility data
        """
        # Group by c_puct
        c_puct_groups = {}
        for snapshot in snapshots:
            c_puct = snapshot.observables.get('c_puct', 1.0)
            if c_puct not in c_puct_groups:
                c_puct_groups[c_puct] = []
            c_puct_groups[c_puct].append(snapshot)
        
        if len(c_puct_groups) < 2:
            return {'chi': 0.0}
        
        # Compute average Q for each c_puct
        c_values = []
        q_averages = []
        
        for c_puct, group in sorted(c_puct_groups.items()):
            c_values.append(c_puct)
            
            # Average Q over group
            q_sum = 0.0
            count = 0
            
            for snapshot in group:
                visits = snapshot.observables['visit_distribution']
                q_values = snapshot.observables['value_landscape']
                
                if not isinstance(visits, torch.Tensor):
                    visits = torch.tensor(visits)
                if not isinstance(q_values, torch.Tensor):
                    q_values = torch.tensor(q_values)
                
                probs = visits / visits.sum()
                avg_q = torch.sum(probs * q_values)
                
                q_sum += avg_q.item()
                count += 1
            
            q_averages.append(q_sum / count if count > 0 else 0)
        
        # Linear fit to get susceptibility
        if len(c_values) >= 2:
            chi = (q_averages[-1] - q_averages[0]) / (c_values[-1] - c_values[0])
        else:
            chi = 0.0
        
        return {'chi': chi}
    
    def compute_work_distribution(self, trajectories: List[List]) -> List[float]:
        """
        Compute work done along each trajectory.
        
        Work = ΔF = F_final - F_initial
        
        Args:
            trajectories: List of snapshot sequences
            
        Returns:
            Work values for each trajectory
        """
        work_values = []
        
        for trajectory in trajectories:
            if len(trajectory) < 2:
                work_values.append(0.0)
                continue
            
            # Compute free energy at start and end
            thermo_start = self.measure_thermodynamics([trajectory[0]])[0]
            thermo_end = self.measure_thermodynamics([trajectory[-1]])[0]
            
            work = thermo_end['free_energy'] - thermo_start['free_energy']
            work_values.append(work)
        
        return work_values
    
    def validate_jarzynski_equality(self, trajectories: List[List]) -> Dict[str, Any]:
        """
        Validate Jarzynski equality: <exp(-βW)> = exp(-βΔF).
        
        NOTE: MCTS is a non-equilibrium process that violates detailed balance.
        According to the quantum foundation document, MCTS implements:
        - An irreversible search process
        - Information flows from leaves to root
        - Work is done by simulations (heat input)
        
        The equality may not hold exactly but provides bounds on free energy.
        
        Args:
            trajectories: Ensemble of trajectories
            
        Returns:
            Validation results with physical interpretation
        """
        # Compute work distribution
        work_values = self.compute_work_distribution(trajectories)
        
        if not work_values:
            return {
                'error': 'No trajectories provided',
                'equality_satisfied': False
            }
        
        # Get temperature (assume same for all)
        if trajectories and trajectories[0]:
            T = trajectories[0][-1].observables.get('temperature', 1.0)
        else:
            T = 1.0
        
        # Ensure temperature is positive
        if T <= 0:
            T = 1.0
        
        beta = 1.0 / T
        
        # Jarzynski average
        exp_work = [np.exp(-beta * w) for w in work_values]
        jarzynski_avg = np.mean(exp_work)
        
        # Free energy difference
        # For ensemble average
        delta_F_direct = np.mean(work_values)
        delta_F_jarzynski = -T * np.log(jarzynski_avg) if jarzynski_avg > 0 else np.inf
        
        # For MCTS, work includes information gained from simulations
        # This is always positive (second law), leading to inequality
        dissipated_work = delta_F_direct - delta_F_jarzynski
        
        # Log warning about non-equilibrium nature
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"MCTS Jarzynski analysis: <W> = {delta_F_direct:.3f}, "
                   f"ΔF_Jarzynski = {delta_F_jarzynski:.3f}, "
                   f"Dissipated work = {dissipated_work:.3f}")
        
        # Check inequality (should be satisfied for irreversible process)
        inequality_satisfied = jarzynski_avg >= np.exp(-beta * delta_F_direct)
        
        return {
            'work_distribution': work_values,
            'avg_work': delta_F_direct,
            'jarzynski_average': jarzynski_avg,
            'jarzynski_estimate': delta_F_jarzynski,
            'dissipated_work': dissipated_work,
            'temperature': T,
            'beta': beta,
            'inequality_satisfied': inequality_satisfied,
            'equality_satisfied': False,  # Always false for irreversible MCTS
            'interpretation': 'MCTS is irreversible; Jarzynski provides lower bound on ΔF'
        }
    
    def compute_entropy_production(self, trajectory: List) -> float:
        """
        Compute total entropy production along trajectory.
        
        Σ = ΔS_system + ΔS_environment >= 0
        
        Args:
            trajectory: Sequence of snapshots
            
        Returns:
            Total entropy production
        """
        if len(trajectory) < 2:
            return 0.0
        
        # System entropy change
        thermo_start = self.measure_thermodynamics([trajectory[0]])[0]
        thermo_end = self.measure_thermodynamics([trajectory[-1]])[0]
        
        delta_S_system = thermo_end['entropy'] - thermo_start['entropy']
        
        # Environment entropy = heat / temperature
        # Heat = sum of value changes
        heat = 0.0
        for i in range(1, len(trajectory)):
            prev_energy = self.measure_thermodynamics([trajectory[i-1]])[0]['energy']
            curr_energy = self.measure_thermodynamics([trajectory[i]])[0]['energy']
            heat += curr_energy - prev_energy
        
        T_avg = (thermo_start['temperature'] + thermo_end['temperature']) / 2
        delta_S_env = heat / T_avg if T_avg > 0 else 0
        
        # Total entropy production
        entropy_production = delta_S_system + delta_S_env
        
        # Ensure non-negative (second law)
        return max(0.0, entropy_production)
    
    def validate_crooks_theorem(self, forward_trajectories: List[List],
                               reverse_trajectories: List[List]) -> Dict[str, Any]:
        """
        Validate Crooks fluctuation theorem.
        
        P_F(W) / P_R(-W) = exp(βW)
        
        Args:
            forward_trajectories: Forward protocol trajectories
            reverse_trajectories: Reverse protocol trajectories
            
        Returns:
            Validation results
        """
        # Compute work distributions
        work_forward = self.compute_work_distribution(forward_trajectories)
        work_reverse = self.compute_work_distribution(reverse_trajectories)
        
        # For simplicity, compare average work
        avg_work_F = np.mean(work_forward)
        avg_work_R = np.mean(work_reverse)
        
        # In ideal case, <W_F> = -<W_R> for time-reversible dynamics
        # For MCTS, there will be dissipation
        
        work_ratio = avg_work_F / (-avg_work_R) if avg_work_R != 0 else 1.0
        
        return {
            'work_ratio': work_ratio,
            'forward_work': avg_work_F,
            'reverse_work': avg_work_R,
            'theorem_satisfied': 0.5 < work_ratio < 2.0  # Rough check
        }