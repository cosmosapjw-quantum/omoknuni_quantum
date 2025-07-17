"""
Decoherence dynamics analyzer for MCTS.

Measures how classical decisions emerge from quantum-like superposition
in the tree search process.
"""
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from scipy.optimize import curve_fit

# Import unified quantum definitions
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from quantum_definitions import (
    UnifiedQuantumDefinitions, 
    MCTSQuantumState,
    compute_von_neumann_entropy,
    compute_purity,
    compute_coherence,
    construct_quantum_state_from_visits
)


@dataclass
class DecoherenceResult:
    """Results from decoherence analysis"""
    coherence_evolution: List[float]
    purity_evolution: List[float]
    decoherence_rate: float
    relaxation_time: float
    pointer_states: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Export results to dictionary"""
        return {
            'coherence_evolution': self.coherence_evolution,
            'purity_evolution': self.purity_evolution,
            'decoherence_rate': self.decoherence_rate,
            'relaxation_time': self.relaxation_time,
            'pointer_states': self.pointer_states
        }


class DecoherenceAnalyzer:
    """
    Analyze decoherence dynamics in MCTS.
    
    In path integral formulation:
    - Multiple paths to same state create quantum interference
    - Path selection process causes decoherence
    - Density matrix evolves from pure superposition to mixed state
    
    Key metrics:
    - Coherence measure: Off-diagonal elements of density matrix
    - Purity: P(t) = Tr(ρ²)
    - Decoherence rate: Γ from coherence decay
    - Von Neumann entropy: S = -Tr(ρ log ρ)
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize decoherence analyzer.
        
        Args:
            device: Computation device ('cuda' or 'cpu')
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize unified quantum definitions
        self.quantum_defs = UnifiedQuantumDefinitions(device=self.device)
        self._last_decoherence_fit = {}
    
    def analyze_decoherence(self, snapshots: List) -> Dict[str, Any]:
        """
        Analyze decoherence dynamics from snapshot sequence.
        
        In MCTS path integral:
        - Initial state: superposition of all possible paths (pure state)
        - Evolution: path selection causes decoherence
        - Final state: classical decision (pointer state)
        
        Args:
            snapshots: List of TreeSnapshot objects
            
        Returns:
            Dictionary with decoherence metrics
        """
        coherences = []
        purities = []
        von_neumann_entropies = []
        times = []
        quantum_states = []
        
        # Import wave-based quantum state constructor
        try:
            from ..wave_based_quantum_state import WaveBasedQuantumState
            wave_constructor = WaveBasedQuantumState(device=self.device)
        except ImportError:
            import sys
            from pathlib import Path
            parent_dir = Path(__file__).parent.parent
            sys.path.insert(0, str(parent_dir))
            from wave_based_quantum_state import WaveBasedQuantumState
            wave_constructor = WaveBasedQuantumState(device=self.device)
        
        # Process snapshots using wave-based approach
        for i, snapshot in enumerate(snapshots):
            # Extract simulation batch data if available
            sim_batch = snapshot.observables.get('simulation_batch')
            
            if sim_batch is not None:
                # Use wave-based construction from simulation batch
                try:
                    quantum_state = wave_constructor.construct_from_simulation_batch(sim_batch)
                    quantum_states.append(quantum_state)
                except Exception as e:
                    # Fallback to single visits if batch construction fails
                    visits = snapshot.observables.get('visit_distribution')
                    if visits is not None:
                        if isinstance(visits, torch.Tensor):
                            visits = visits.to(self.device)
                        else:
                            visits = torch.tensor(visits, device=self.device)
                        # Use legacy method as fallback
                        quantum_state = self.quantum_defs.construct_quantum_state_from_single_visits(visits)
                        quantum_states.append(quantum_state)
            else:
                # Fallback: use single visit distribution
                visits = snapshot.observables.get('visit_distribution')
                if visits is not None:
                    if isinstance(visits, torch.Tensor):
                        visits = visits.to(self.device)
                    else:
                        visits = torch.tensor(visits, device=self.device)
                    # Use legacy method as fallback
                    quantum_state = self.quantum_defs.construct_quantum_state_from_single_visits(visits)
                    quantum_states.append(quantum_state)
            
            # Get density matrix from quantum state
            density_matrix = quantum_state.density_matrix
            
            # Compute decoherence metrics using unified definitions
            coherence = compute_coherence(density_matrix)
            purity = compute_purity(density_matrix)
            von_neumann = compute_von_neumann_entropy(density_matrix)
            
            coherences.append(coherence.item() if isinstance(coherence, torch.Tensor) else coherence)
            purities.append(purity.item() if isinstance(purity, torch.Tensor) else purity)
            von_neumann_entropies.append(von_neumann.item() if isinstance(von_neumann, torch.Tensor) else von_neumann)
            times.append(snapshot.timestamp)
        
        # Fit exponential decay
        if len(coherences) >= 3:
            decoherence_rate = self.fit_exponential_decay(coherences, times)
        else:
            decoherence_rate = 0.0
        
        # Compute relaxation time
        relaxation_time = 1.0 / decoherence_rate if decoherence_rate > 0 else float('inf')
        
        # Identify pointer states from final snapshot
        if snapshots:
            pointer_states = self.identify_pointer_states(snapshots[-1])
        else:
            pointer_states = []
        
        return {
            'coherence_evolution': coherences,
            'purity_evolution': purities,
            'decoherence_rate': decoherence_rate,
            'relaxation_time': relaxation_time,
            'pointer_states': pointer_states
        }
    
    def compute_coherence(self, visits: torch.Tensor) -> torch.Tensor:
        """
        Compute coherence measure (Shannon entropy).
        
        Args:
            visits: Visit count distribution
            
        Returns:
            Coherence value
        """
        # Convert to probabilities
        total_visits = visits.sum()
        if total_visits == 0:
            return torch.tensor(0.0, device=self.device)
        
        probs = visits / total_visits
        
        # Remove zeros for log computation
        probs_nonzero = probs[probs > 0]
        
        # Shannon entropy
        coherence = -torch.sum(probs_nonzero * torch.log(probs_nonzero))
        
        return coherence
    
    def compute_purity(self, visits: torch.Tensor) -> torch.Tensor:
        """
        Compute purity (inverse participation ratio).
        
        Args:
            visits: Visit count distribution
            
        Returns:
            Purity value
        """
        # Convert to probabilities
        total_visits = visits.sum()
        if total_visits == 0:
            return torch.tensor(1.0, device=self.device)
        
        probs = visits / total_visits
        
        # Purity is sum of squared probabilities
        purity = torch.sum(probs ** 2)
        
        return purity
    
    def fit_exponential_decay(self, coherences: List[float], times: List[float]) -> float:
        """
        Fit decoherence law S_vN(k) = S_0 exp(-k/k_dec) + S_∞ from quantum_mcts_foundation.md
        
        This implements the theoretical formulation from Section 7.3:
        - S_0: Initial entropy (superposition)
        - k_dec: Decoherence time 
        - S_∞: Residual entropy (final mixed state)
        
        Args:
            coherences: von Neumann entropy values S_vN(k)
            times: Simulation counts k
            
        Returns:
            Decoherence rate 1/k_dec
        """
        if len(coherences) < 4:  # Need at least 4 points for 3-parameter fit
            return 0.0
        
        # Convert to numpy
        coherences_np = np.array(coherences)
        times_np = np.array(times)
        
        # Fit S_vN(k) = S_0 exp(-k/k_dec) + S_∞
        def decoherence_model(k, S_0, k_dec, S_inf):
            return S_0 * np.exp(-k / k_dec) + S_inf
        
        try:
            # Initial parameter estimates
            S_inf_guess = np.min(coherences_np)  # Final entropy
            S_0_guess = coherences_np[0] - S_inf_guess  # Initial excess entropy
            k_dec_guess = times_np[-1] / 3  # Characteristic decay time
            
            # Bounds: S_0 > 0, k_dec > 0, S_inf >= 0
            bounds = ([0, 1e-6, 0], [np.inf, np.inf, np.inf])
            
            popt, _ = curve_fit(
                decoherence_model, 
                times_np, 
                coherences_np,
                p0=[S_0_guess, k_dec_guess, S_inf_guess],
                bounds=bounds,
                maxfev=1000
            )
            
            S_0_fit, k_dec_fit, S_inf_fit = popt
            
            # Store fit parameters for analysis
            self._last_decoherence_fit = {
                'S_0': S_0_fit,
                'k_dec': k_dec_fit,
                'S_inf': S_inf_fit,
                'fit_quality': self._compute_fit_quality(times_np, coherences_np, popt)
            }
            
            # Return decoherence rate
            return 1.0 / k_dec_fit if k_dec_fit > 0 else 0.0
            
        except Exception as e:
            # Fallback to simple exponential if 3-parameter fit fails
            try:
                # Simple exponential S(k) = S_0 exp(-k/k_dec)
                def simple_exp(k, S_0, k_dec):
                    return S_0 * np.exp(-k / k_dec)
                
                popt, _ = curve_fit(simple_exp, times_np, coherences_np, maxfev=500)
                S_0_fit, k_dec_fit = popt
                
                self._last_decoherence_fit = {
                    'S_0': S_0_fit,
                    'k_dec': k_dec_fit,
                    'S_inf': 0.0,
                    'fit_quality': 'fallback_simple_exponential'
                }
                
                return 1.0 / k_dec_fit if k_dec_fit > 0 else 0.0
                
            except:
                # Final fallback: characteristic time from data
                return self._compute_characteristic_decay_rate(coherences_np, times_np)
    
    def _compute_fit_quality(self, times: np.ndarray, coherences: np.ndarray, popt: np.ndarray) -> float:
        """Compute R² for decoherence fit quality"""
        def decoherence_model(k, S_0, k_dec, S_inf):
            return S_0 * np.exp(-k / k_dec) + S_inf
        
        predicted = decoherence_model(times, *popt)
        ss_res = np.sum((coherences - predicted) ** 2)
        ss_tot = np.sum((coherences - np.mean(coherences)) ** 2)
        
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    def _compute_characteristic_decay_rate(self, coherences: np.ndarray, times: np.ndarray) -> float:
        """Fallback: compute characteristic decay rate from data"""
        initial_coherence = coherences[0]
        if initial_coherence <= 0:
            return 0.0
            
        half_coherence = initial_coherence / 2
        
        # Find crossing point
        for i, c in enumerate(coherences):
            if c <= half_coherence:
                # Linear interpolation for better estimate
                if i > 0:
                    t1, c1 = times[i-1], coherences[i-1]
                    t2, c2 = times[i], coherences[i]
                    # Solve for t where c(t) = half_coherence
                    if c1 != c2:
                        t_half = t1 + (half_coherence - c1) * (t2 - t1) / (c2 - c1)
                    else:
                        t_half = t1
                    
                    # Characteristic rate = 1 / half-life
                    if t_half > 0:
                        return 1.0 / t_half
                    else:
                        return 1.0  # Instantaneous convergence
                        
        # If no crossing found, estimate from overall change
        total_change = abs(coherences_np[-1] - coherences_np[0])
        total_time = times_np[-1] - times_np[0]
        
        if total_time > 0 and total_change > 0:
            # Rough estimate of rate
            return total_change / (initial_coherence * total_time)
        else:
            return 0.0
    
    def identify_pointer_states(self, snapshot) -> List[Dict[str, Any]]:
        """
        Identify pointer states (robust classical outcomes).
        
        Args:
            snapshot: TreeSnapshot object
            
        Returns:
            List of pointer state information
        """
        visits = snapshot.observables.get('visit_distribution')
        if visits is None:
            return []
        
        # Ensure tensor
        if not isinstance(visits, torch.Tensor):
            visits = torch.tensor(visits, device=self.device)
        else:
            visits = visits.to(self.device)
        
        # Convert to probabilities
        total_visits = visits.sum()
        if total_visits == 0:
            return []
        
        probs = visits / total_visits
        
        # Pointer states are high-probability outcomes
        # Use threshold of mean + std
        threshold = probs.mean() + probs.std()
        
        pointer_states = []
        for idx, prob in enumerate(probs):
            if prob > threshold:
                pointer_states.append({
                    'node_index': int(idx),
                    'visit_fraction': float(prob),
                    'visits': int(visits[idx])
                })
        
        # Sort by visit fraction
        pointer_states.sort(key=lambda x: x['visit_fraction'], reverse=True)
        
        return pointer_states
    
    def compute_density_matrix(self, visits: torch.Tensor) -> torch.Tensor:
        """
        Compute density matrix from visit distribution.
        
        In path integral formulation:
        ρ = |ψ⟩⟨ψ| where |ψ⟩ = Σ_i √(p_i) |i⟩
        
        Args:
            visits: Visit counts for each action
            
        Returns:
            Density matrix
        """
        # Normalize to get probabilities
        total_visits = visits.sum()
        if total_visits == 0:
            n_actions = len(visits)
            return torch.eye(n_actions, device=self.device) / n_actions
        
        # Wavefunction amplitudes
        psi = torch.sqrt(visits / total_visits)
        
        # Density matrix ρ = |ψ⟩⟨ψ|
        rho = torch.outer(psi, psi)
        
        return rho
    
    def compute_coherence_from_density_matrix(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Compute coherence from density matrix.
        
        Coherence = sum of absolute values of off-diagonal elements
        
        Args:
            rho: Density matrix
            
        Returns:
            Coherence measure
        """
        n = rho.shape[0]
        # Get off-diagonal elements
        mask = ~torch.eye(n, dtype=torch.bool, device=self.device)
        coherence = torch.abs(rho[mask]).sum()
        return coherence
    
    def compute_purity_from_density_matrix(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Compute purity from density matrix.
        
        Purity = Tr(ρ²)
        - Pure state: P = 1
        - Maximally mixed: P = 1/d
        
        Args:
            rho: Density matrix
            
        Returns:
            Purity
        """
        rho_squared = torch.matmul(rho, rho)
        purity = torch.trace(rho_squared)
        return purity
    
    def compute_von_neumann_entropy(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Compute von Neumann entropy.
        
        S = -Tr(ρ log ρ)
        
        Args:
            rho: Density matrix
            
        Returns:
            von Neumann entropy
        """
        # Eigendecomposition
        eigenvalues, _ = torch.linalg.eigh(rho)
        
        # Filter out zero eigenvalues to avoid log(0)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        # von Neumann entropy
        entropy = -torch.sum(eigenvalues * torch.log(eigenvalues))
        
        return entropy
    
    def compute_coherence_batch(self, visits_batch: torch.Tensor) -> List[float]:
        """
        Compute coherence for batch of distributions.
        
        Args:
            visits_batch: [batch_size, n_actions] tensor
            
        Returns:
            List of coherence values
        """
        batch_size = visits_batch.shape[0]
        coherences = []
        
        for i in range(batch_size):
            coherence = self.compute_coherence(visits_batch[i])
            coherences.append(coherence.item())
        
        return coherences
    
    def measure_information_redundancy(self, snapshot) -> float:
        """
        Measure information redundancy (Quantum Darwinism signature).
        
        High redundancy means the "best move" information is 
        robustly encoded throughout the tree.
        
        Args:
            snapshot: TreeSnapshot
            
        Returns:
            Redundancy measure (0 to 1)
        """
        visits = snapshot.observables.get('visit_distribution')
        if visits is None:
            return 0.0
        
        # Ensure tensor
        if not isinstance(visits, torch.Tensor):
            visits = torch.tensor(visits, device=self.device)
        else:
            visits = visits.to(self.device)
        
        # Find dominant action
        best_action = torch.argmax(visits)
        best_visits = visits[best_action]
        total_visits = visits.sum()
        
        if total_visits == 0:
            return 0.0
        
        # Redundancy is how dominant the best action is
        # High redundancy = clear winner
        dominance = best_visits / total_visits
        
        # Also consider gap to second best
        other_visits = visits.clone()
        other_visits[best_action] = 0
        second_best = other_visits.max()
        
        if second_best > 0:
            gap = (best_visits - second_best) / total_visits
        else:
            gap = dominance
        
        # Combine dominance and gap
        redundancy = (dominance + gap) / 2.0
        
        return float(redundancy)