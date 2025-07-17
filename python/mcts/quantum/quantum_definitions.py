"""
Unified quantum definitions for MCTS quantum physics framework.

This module provides consistent definitions for all quantum mechanical concepts
used across the physics modules, ensuring internal consistency based on the
quantum_mcts_foundation.md theoretical framework.

Key insight: Off-diagonal terms arise naturally from wave vector superposition
ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ| where |ψᵢ⟩ = Σₐ √(πᵢ(a|s)) |a⟩
This naturally produces ρₐᵦ = Σᵢ pᵢ √(πᵢ(a|s)) √(πᵢ(b|s))
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MCTSQuantumState:
    """
    Quantum state representation for MCTS following quantum_mcts_foundation.md
    
    Based on the insight that at each node s, we have:
    - Pure state components: |ψₛ,ᵢ⟩ = Σₐ √(πᵢ(a|s)) |a⟩
    - Mixed state ensemble: ρₛ = Σᵢ pᵢ |ψₛ,ᵢ⟩⟨ψₛ,ᵢ|
    - Off-diagonal terms emerge naturally from superposition
    
    Attributes:
        ensemble_weights: Classical probabilities pᵢ for each outcome
        action_amplitudes: Complex amplitudes √(πᵢ(a|s)) for each outcome i and action a
        density_matrix: Full density matrix ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|
        n_actions: Number of actions at this node
        n_outcomes: Number of possible simulation outcomes
    """
    ensemble_weights: torch.Tensor      # Shape: [n_outcomes]
    action_amplitudes: torch.Tensor     # Shape: [n_outcomes, n_actions]
    density_matrix: torch.Tensor        # Shape: [n_actions, n_actions]
    n_actions: int
    n_outcomes: int
    device: str = 'cpu'
    
    def __post_init__(self):
        """Validate quantum state consistency"""
        # Check ensemble weight normalization
        weight_sum = torch.sum(self.ensemble_weights)
        if abs(weight_sum - 1.0) > 1e-6:
            logger.warning(f"Ensemble weights not normalized: sum = {weight_sum}")
        
        # Vectorized amplitude normalization check
        amp_sums = torch.sum(torch.abs(self.action_amplitudes) ** 2, dim=1)
        invalid_amplitudes = torch.abs(amp_sums - 1.0) > 1e-6
        if torch.any(invalid_amplitudes):
            invalid_indices = torch.nonzero(invalid_amplitudes).squeeze()
            logger.warning(f"Action amplitudes not normalized for outcomes: {invalid_indices.tolist()}")
        
        # Check density matrix properties
        trace = torch.trace(self.density_matrix)
        if abs(trace - 1.0) > 1e-6:
            logger.warning(f"Density matrix trace ≠ 1: trace = {trace}")


class UnifiedQuantumDefinitions:
    """
    Unified quantum definitions following quantum_mcts_foundation.md
    
    Implements the corrected framework where:
    1. Each node has superposition |ψᵢ⟩ = Σₐ √(πᵢ(a|s)) |a⟩
    2. Mixed states arise from outcome uncertainty: ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|
    3. Off-diagonal terms emerge naturally from superposition
    4. Decoherence occurs as outcome uncertainty decreases
    """
    
    def __init__(self, device: Optional[str] = None):
        """Initialize unified quantum definitions"""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.regularization = 1e-10  # Numerical stability
        
    def construct_quantum_state_from_visit_ensemble(self, 
                                                  visit_distributions: List[torch.Tensor],
                                                  ensemble_weights: Optional[torch.Tensor] = None) -> MCTSQuantumState:
        """
        Construct quantum state from ensemble of visit distributions.
        
        This is the corrected implementation following quantum_mcts_foundation.md:
        - Each visit distribution creates a wave vector |ψᵢ⟩ = Σₐ √(πᵢ(a)) |a⟩
        - Mixed state: ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|
        - Off-diagonal terms emerge naturally from cross-terms
        
        Args:
            visit_distributions: List of visit count distributions for different outcomes
            ensemble_weights: Classical probabilities pᵢ for each outcome
            
        Returns:
            MCTSQuantumState with natural off-diagonal terms
        """
        n_outcomes = len(visit_distributions)
        if n_outcomes == 0:
            raise ValueError("Empty visit distributions")
        
        # Vectorized conversion and normalization
        if isinstance(visit_distributions[0], torch.Tensor):
            visit_tensors = torch.stack([v.to(self.device) for v in visit_distributions])
        else:
            visit_tensors = torch.tensor(visit_distributions, dtype=torch.float32, device=self.device)
        
        visit_tensors = torch.clamp(visit_tensors, min=0.0)
        n_actions = visit_tensors.shape[1]
        
        # Default uniform ensemble weights if not provided
        if ensemble_weights is None:
            ensemble_weights = torch.ones(n_outcomes, device=self.device) / n_outcomes
        elif not isinstance(ensemble_weights, torch.Tensor):
            ensemble_weights = torch.tensor(ensemble_weights, dtype=torch.float32, device=self.device)
        else:
            ensemble_weights = ensemble_weights.to(self.device)
        
        # Normalize ensemble weights
        ensemble_weights = ensemble_weights / torch.sum(ensemble_weights)
        
        # Vectorized computation of action amplitudes
        total_visits = torch.sum(visit_tensors, dim=1, keepdim=True)
        total_visits = torch.clamp(total_visits, min=1.0)  # Avoid division by zero
        
        # Convert visits to probabilities
        probs = visit_tensors / total_visits
        
        # Amplitudes are sqrt of probabilities
        action_amplitudes = torch.sqrt(probs)
        
        # Vectorized density matrix construction: ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|
        # Using einsum for efficient batch outer product
        density_matrix = torch.einsum('i,ij,ik->jk', 
                                     ensemble_weights, 
                                     action_amplitudes, 
                                     action_amplitudes)
        
        # Add regularization for numerical stability
        density_matrix += torch.eye(n_actions, device=self.device) * self.regularization
        
        # Normalize density matrix to ensure trace = 1
        trace = torch.trace(density_matrix)
        if trace <= 0:
            raise ValueError(f"Invalid density matrix: trace = {trace}. This indicates all visit distributions are zero or numerical error in quantum state construction.")
        
        density_matrix = density_matrix / trace
        
        return MCTSQuantumState(
            ensemble_weights=ensemble_weights,
            action_amplitudes=action_amplitudes,
            density_matrix=density_matrix,
            n_actions=n_actions,
            n_outcomes=n_outcomes,
            device=self.device
        )
    
    def construct_quantum_state_from_single_visits(self, 
                                                 visits: Union[torch.Tensor, np.ndarray, List[float]],
                                                 outcome_uncertainty: float = 0.1) -> MCTSQuantumState:
        """
        Construct quantum state from single visit distribution.
        
        DEPRECATED: Use wave_based_quantum_state.WaveBasedQuantumState instead.
        This method is kept for backward compatibility only.
        
        Args:
            visits: Single visit count distribution
            outcome_uncertainty: Legacy parameter (ignored)
            
        Returns:
            MCTSQuantumState (nearly pure state)
        """
        if not isinstance(visits, torch.Tensor):
            visits = torch.tensor(visits, dtype=torch.float32, device=self.device)
        else:
            visits = visits.to(self.device)
        
        visits = torch.clamp(visits, min=0.0)
        
        # Single snapshot creates pure state
        # Use WaveBasedQuantumState for proper ensemble construction
        import warnings
        warnings.warn(
            "construct_quantum_state_from_single_visits is deprecated. "
            "Use WaveBasedQuantumState for proper ensemble construction from MCTS data.",
            DeprecationWarning
        )
        
        visit_distributions = [visits]
        return self.construct_quantum_state_from_visit_ensemble(visit_distributions)
    
    def compute_von_neumann_entropy(self, density_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute von Neumann entropy S = -Tr(ρ log ρ).
        
        Unified implementation consistent with quantum_mcts_foundation.md
        """
        try:
            eigenvalues = torch.linalg.eigvalsh(density_matrix)
        except Exception as e:
            logger.warning(f"Eigendecomposition failed: {e}")
            eigenvalues = torch.diag(density_matrix)
        
        eigenvalues = torch.real(eigenvalues)
        eigenvalues = torch.clamp(eigenvalues, min=0.0)
        eigenvalues = eigenvalues[eigenvalues > self.regularization]
        
        if len(eigenvalues) == 0:
            return torch.tensor(0.0, device=self.device)
        
        eigenvalues = eigenvalues / torch.sum(eigenvalues)
        entropy = -torch.sum(eigenvalues * torch.log(eigenvalues))
        
        return entropy
    
    def compute_purity(self, density_matrix: torch.Tensor) -> torch.Tensor:
        """Compute purity P = Tr(ρ²)"""
        purity = torch.trace(density_matrix @ density_matrix)
        return torch.real(purity)
    
    def compute_coherence(self, density_matrix: torch.Tensor) -> torch.Tensor:
        """Compute coherence as sum of absolute off-diagonal elements"""
        n = density_matrix.shape[0]
        # Ensure mask is on same device as density_matrix
        mask = ~torch.eye(n, dtype=torch.bool, device=density_matrix.device)
        coherence = torch.abs(density_matrix[mask]).sum()
        return coherence
    
    def compute_path_integral_weights(self, 
                                    q_values: Union[torch.Tensor, np.ndarray, List[float]],
                                    temperature: float) -> torch.Tensor:
        """
        Compute path integral weights P[γ] ∝ exp(-β S[γ]) from quantum_mcts_foundation.md
        """
        if not isinstance(q_values, torch.Tensor):
            q_values = torch.tensor(q_values, dtype=torch.float32, device=self.device)
        else:
            q_values = q_values.to(self.device)
        
        if temperature <= 0:
            logger.warning(f"Invalid temperature: {temperature}")
            temperature = 1.0
        
        beta = 1.0 / temperature
        
        # Action S[γ] = -Σ Score(s,a) where Score = Q + U
        # For simplicity, use Q-values as scores
        path_actions = torch.cumsum(q_values, dim=0)
        
        # Path integral weights: P[γ] ∝ exp(-β S[γ])
        weights = torch.exp(-beta * path_actions)
        weights = weights / torch.sum(weights)
        
        return weights
    
    def partial_trace_optimized(self, 
                              quantum_state: MCTSQuantumState,
                              subsystem_indices: List[int]) -> torch.Tensor:
        """
        Optimized partial trace computation using the ensemble structure.
        
        Instead of tracing the full density matrix, we can work directly
        with the ensemble representation for efficiency.
        """
        n_actions = quantum_state.n_actions
        keep_indices = [i for i in range(n_actions) if i not in subsystem_indices]
        
        if not keep_indices:
            return torch.tensor([[1.0]], device=self.device)
        
        n_keep = len(keep_indices)
        reduced_density = torch.zeros(n_keep, n_keep, device=self.device)
        
        # Vectorized computation using ensemble structure
        # Extract relevant amplitudes for all outcomes at once
        reduced_amplitudes = quantum_state.action_amplitudes[:, keep_indices]
        
        # Renormalize
        norms = torch.sum(torch.abs(reduced_amplitudes) ** 2, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-10)
        reduced_amplitudes = reduced_amplitudes / torch.sqrt(norms)
        
        # Compute reduced density matrix using einsum
        reduced_density = torch.einsum('i,ij,ik->jk',
                                      quantum_state.ensemble_weights,
                                      reduced_amplitudes,
                                      reduced_amplitudes)
        
        return reduced_density
    
    def compute_entanglement_entropy(self, 
                                   quantum_state: MCTSQuantumState,
                                   subsystem_indices: List[int]) -> torch.Tensor:
        """
        Compute entanglement entropy using optimized partial trace.
        """
        reduced_rho = self.partial_trace_optimized(quantum_state, subsystem_indices)
        return self.compute_von_neumann_entropy(reduced_rho)
    
    def evolve_quantum_state(self, 
                           quantum_state: MCTSQuantumState,
                           new_visits: torch.Tensor,
                           decoherence_rate: float = 0.1) -> MCTSQuantumState:
        """
        Evolve quantum state with new visit information following decoherence dynamics.
        
        Implements the decoherence law: S_vN(k) = S_0 exp(-k/k_dec) + S_∞
        by reducing outcome uncertainty as more information is gained.
        """
        # Reduce outcome uncertainty (decoherence)
        current_uncertainty = self.compute_outcome_uncertainty(quantum_state)
        new_uncertainty = current_uncertainty * np.exp(-decoherence_rate)
        
        # Create new quantum state with reduced uncertainty
        return self.construct_quantum_state_from_single_visits(new_visits, new_uncertainty)
    
    def compute_outcome_uncertainty(self, quantum_state: MCTSQuantumState) -> float:
        """
        Compute outcome uncertainty from ensemble structure.
        
        High uncertainty = many outcomes with similar weights
        Low uncertainty = few dominant outcomes
        """
        weights = quantum_state.ensemble_weights
        # Use entropy of ensemble weights as uncertainty measure
        entropy = -torch.sum(weights * torch.log(weights + self.regularization))
        max_entropy = np.log(quantum_state.n_outcomes)
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0
    
    def validate_quantum_consistency(self, quantum_state: MCTSQuantumState) -> Dict[str, Any]:
        """
        Validate quantum state consistency according to quantum_mcts_foundation.md
        """
        results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check ensemble weight normalization
        weight_sum = torch.sum(quantum_state.ensemble_weights)
        if abs(weight_sum - 1.0) > 1e-6:
            results['warnings'].append(f"Ensemble weights not normalized: sum = {weight_sum}")
        
        # Check density matrix properties
        trace = torch.trace(quantum_state.density_matrix)
        if abs(trace - 1.0) > 1e-6:
            results['warnings'].append(f"Density matrix trace ≠ 1: trace = {trace}")
        
        # Check Hermiticity
        if not torch.allclose(quantum_state.density_matrix, 
                            quantum_state.density_matrix.conj().T, atol=1e-8):
            results['errors'].append("Density matrix not Hermitian")
            results['valid'] = False
        
        # Check positive semi-definiteness
        try:
            eigenvalues = torch.linalg.eigvalsh(quantum_state.density_matrix)
            if torch.any(eigenvalues < -1e-8):
                results['errors'].append("Density matrix not positive semi-definite")
                results['valid'] = False
        except Exception as e:
            results['errors'].append(f"Failed to check positive semi-definiteness: {e}")
        
        # Check natural off-diagonal structure
        purity = self.compute_purity(quantum_state.density_matrix)
        coherence = self.compute_coherence(quantum_state.density_matrix)
        
        results['purity'] = float(purity)
        results['coherence'] = float(coherence)
        results['is_pure'] = purity > 0.99
        results['has_coherence'] = coherence > 1e-6
        
        return results


# Create global instance for consistency
quantum_definitions = UnifiedQuantumDefinitions()

# Convenience functions for module consistency
def construct_quantum_state_from_visits(visits: Union[torch.Tensor, np.ndarray, List[float]],
                                       outcome_uncertainty: float = 0.1) -> MCTSQuantumState:
    """Construct quantum state from visits with natural off-diagonal terms"""
    return quantum_definitions.construct_quantum_state_from_single_visits(visits, outcome_uncertainty)

def construct_quantum_state_from_ensemble(visit_distributions: List[torch.Tensor],
                                        ensemble_weights: Optional[torch.Tensor] = None) -> MCTSQuantumState:
    """Construct quantum state from ensemble of visit distributions"""
    return quantum_definitions.construct_quantum_state_from_visit_ensemble(visit_distributions, ensemble_weights)

def compute_von_neumann_entropy(density_matrix: torch.Tensor) -> torch.Tensor:
    """Compute von Neumann entropy (unified implementation)"""
    return quantum_definitions.compute_von_neumann_entropy(density_matrix)

def compute_purity(density_matrix: torch.Tensor) -> torch.Tensor:
    """Compute purity (unified implementation)"""
    return quantum_definitions.compute_purity(density_matrix)

def compute_coherence(density_matrix: torch.Tensor) -> torch.Tensor:
    """Compute coherence (unified implementation)"""
    return quantum_definitions.compute_coherence(density_matrix)

def compute_path_integral_weights(q_values: Union[torch.Tensor, np.ndarray, List[float]],
                                temperature: float) -> torch.Tensor:
    """Compute path integral weights (unified implementation)"""
    return quantum_definitions.compute_path_integral_weights(q_values, temperature)

def compute_entanglement_entropy(quantum_state: MCTSQuantumState,
                               subsystem_indices: List[int]) -> torch.Tensor:
    """Compute entanglement entropy (unified implementation)"""
    return quantum_definitions.compute_entanglement_entropy(quantum_state, subsystem_indices)