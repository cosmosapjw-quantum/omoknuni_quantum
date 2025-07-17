"""
Wave-based quantum state construction from MCTS dynamics.

This module implements quantum state construction using the natural wave
structure of MCTS, where each simulation batch provides ensemble diversity.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from .quantum_definitions import MCTSQuantumState, UnifiedQuantumDefinitions
except ImportError:
    from quantum_definitions import MCTSQuantumState, UnifiedQuantumDefinitions


@dataclass
class WaveData:
    """Data from a single MCTS wave/simulation."""
    path_probabilities: torch.Tensor  # P[γ] for each path
    action_visits: torch.Tensor       # W(s,a) for this wave
    paths: List[List[int]]           # Actual paths explored
    values: torch.Tensor             # Value estimates
    simulation_id: int               # Unique ID for this simulation


@dataclass
class MCTSWaveEnsemble:
    """Ensemble of waves from MCTS simulations."""
    waves: List[WaveData]
    total_visits: torch.Tensor      # Cumulative N(s,a)
    n_actions: int
    device: str


class WaveBasedQuantumState:
    """
    Extract quantum states from MCTS wave dynamics.
    
    Key insights:
    1. Each simulation in a batch explores different paths
    2. Wave amplitudes W(s,a) naturally form quantum superposition
    3. Path overlap determines phase relationships
    4. Convergence creates natural decoherence
    """
    
    def __init__(self, device: Optional[str] = None):
        """Initialize wave-based quantum state constructor."""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.quantum_defs = UnifiedQuantumDefinitions(device=self.device)
    
    def construct_from_simulation_batch(self,
                                      simulation_batch: List[Dict[str, Any]]) -> MCTSQuantumState:
        """
        Construct quantum state from a batch of MCTS simulations.
        
        Each simulation in the batch provides a wave function |ψᵢ⟩.
        The density matrix is constructed preserving interference patterns.
        
        Args:
            simulation_batch: List of simulation data, each containing:
                - 'action_probs': Action probabilities from this simulation
                - 'paths': List of paths explored
                - 'path_probs': Probability of each path
                - 'values': Value estimates
                
        Returns:
            MCTSQuantumState with natural wave interference
        """
        if not simulation_batch:
            raise ValueError("Empty simulation batch")
        
        # Extract wave amplitudes from each simulation
        wave_amplitudes = []
        path_data = []
        
        for sim in simulation_batch:
            # Get action probabilities as wave amplitudes
            action_probs = sim.get('action_probs')
            if action_probs is None:
                # Fallback: use visit counts
                visits = sim.get('visits', sim.get('action_visits'))
                if visits is None:
                    continue
                total = sum(visits)
                action_probs = [v/total if total > 0 else 1/len(visits) for v in visits]
            
            # Convert to tensor and take sqrt for amplitudes
            if not isinstance(action_probs, torch.Tensor):
                action_probs = torch.tensor(action_probs, dtype=torch.float32, device=self.device)
            
            amplitudes = torch.sqrt(action_probs)
            wave_amplitudes.append(amplitudes)
            
            # Store path information for phase calculation
            paths = sim.get('paths', [])
            path_data.append(paths)
        
        if not wave_amplitudes:
            raise ValueError("No valid wave data in batch")
        
        # Ensure all amplitudes have same dimension
        n_actions = len(wave_amplitudes[0])
        wave_amplitudes = [a for a in wave_amplitudes if len(a) == n_actions]
        
        # Construct density matrix with phase information
        density_matrix = self._construct_density_matrix_with_phases(
            wave_amplitudes, path_data
        )
        
        # Create ensemble weights (uniform for simulation batch)
        n_waves = len(wave_amplitudes)
        ensemble_weights = torch.ones(n_waves, device=self.device) / n_waves
        
        # Stack wave amplitudes
        action_amplitudes = torch.stack(wave_amplitudes)
        
        return MCTSQuantumState(
            ensemble_weights=ensemble_weights,
            action_amplitudes=action_amplitudes,
            density_matrix=density_matrix,
            n_actions=n_actions,
            n_outcomes=n_waves,
            device=self.device
        )
    
    def construct_from_wave_ensemble(self,
                                   wave_ensemble: MCTSWaveEnsemble) -> MCTSQuantumState:
        """
        Construct quantum state from ensemble of MCTS waves.
        
        This method handles the full wave structure including:
        - Individual wave amplitudes W_k(s,a)
        - Path interference patterns
        - Natural decoherence from convergence
        
        Args:
            wave_ensemble: Collection of waves from MCTS
            
        Returns:
            MCTSQuantumState capturing wave dynamics
        """
        waves = wave_ensemble.waves
        if not waves:
            raise ValueError("Empty wave ensemble")
        
        # Extract amplitudes from each wave
        wave_amplitudes = []
        path_data = []
        values = []
        
        for wave in waves:
            # Normalize action visits to get probabilities
            total_visits = wave.action_visits.sum()
            if total_visits > 0:
                probs = wave.action_visits / total_visits
            else:
                probs = torch.ones_like(wave.action_visits) / len(wave.action_visits)
            
            amplitudes = torch.sqrt(probs)
            wave_amplitudes.append(amplitudes)
            path_data.append(wave.paths)
            values.append(wave.values)
        
        # Construct density matrix preserving wave interference
        density_matrix = self._construct_density_matrix_with_phases(
            wave_amplitudes, path_data
        )
        
        # Ensemble weights based on wave importance
        # Later waves (more simulations) get higher weight
        simulation_ids = torch.tensor([w.simulation_id for w in waves], device=self.device)
        weights = torch.softmax(simulation_ids.float() * 0.1, dim=0)
        
        # Stack amplitudes
        action_amplitudes = torch.stack(wave_amplitudes)
        
        return MCTSQuantumState(
            ensemble_weights=weights,
            action_amplitudes=action_amplitudes,
            density_matrix=density_matrix,
            n_actions=wave_ensemble.n_actions,
            n_outcomes=len(waves),
            device=self.device
        )
    
    def _construct_density_matrix_with_phases(self,
                                            wave_amplitudes: List[torch.Tensor],
                                            path_data: List[List[List[int]]]) -> torch.Tensor:
        """
        Construct density matrix preserving interference patterns.
        
        The phase between waves is determined by path overlap:
        - Same paths: constructive interference (phase = 0)
        - Different paths: phase determined by path structure
        
        Args:
            wave_amplitudes: List of amplitude vectors
            path_data: Path information for each wave
            
        Returns:
            Density matrix with natural interference patterns
        """
        n_actions = len(wave_amplitudes[0])
        n_waves = len(wave_amplitudes)
        
        # Initialize complex density matrix
        rho = torch.zeros(n_actions, n_actions, dtype=torch.complex64, device=self.device)
        
        # Compute density matrix with phases
        for i in range(n_waves):
            for j in range(n_waves):
                # Compute phase from path overlap
                phase = self._compute_phase_from_paths(
                    path_data[i] if i < len(path_data) else [],
                    path_data[j] if j < len(path_data) else []
                )
                
                # Add contribution with phase
                contribution = torch.outer(wave_amplitudes[i], wave_amplitudes[j])
                phase_factor = torch.tensor(np.exp(1j * phase), dtype=torch.complex64)
                rho += contribution * phase_factor
        
        # Normalize
        rho = rho / n_waves
        
        # Ensure Hermiticity and convert to real
        rho = (rho + rho.conj().T) / 2
        
        # For real-valued MCTS, we can work with real part
        # (imaginary part should be negligible due to symmetry)
        rho = rho.real
        
        # Renormalize to ensure trace = 1
        trace = torch.trace(rho)
        if trace > 0:
            rho = rho / trace
            
        return rho
    
    def _compute_phase_from_paths(self, 
                                 paths1: List[List[int]], 
                                 paths2: List[List[int]]) -> float:
        """
        Compute relative phase between two sets of paths.
        
        Phase encodes path similarity:
        - Identical paths: phase = 0 (constructive)
        - Completely different: phase = π/2
        - Partial overlap: intermediate phase
        
        Args:
            paths1: First set of paths
            paths2: Second set of paths
            
        Returns:
            Phase in radians
        """
        if not paths1 or not paths2:
            # No path information - assume random phase
            return 0.0
        
        # Compute path overlap score
        # Compare corresponding paths when possible
        min_paths = min(len(paths1), len(paths2))
        
        if min_paths > 0:
            overlap_scores = []
            
            # First, compare corresponding paths
            for i in range(min(min_paths, 5)):  # Limit for efficiency
                p1 = paths1[i]
                p2 = paths2[i]
                min_len = min(len(p1), len(p2))
                if min_len > 0:
                    matches = sum(1 for j in range(min_len) if p1[j] == p2[j])
                    overlap_scores.append(matches / min_len)
            
            # If sets are different sizes, add penalty
            size_diff = abs(len(paths1) - len(paths2))
            if size_diff > 0:
                overlap_scores.extend([0.0] * min(size_diff, 3))
            
            overlap_score = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0
        else:
            overlap_score = 0.0
        
        # Convert overlap to phase
        # High overlap (1.0) → phase 0
        # No overlap (0.0) → phase π/2
        phase = (1 - overlap_score) * np.pi / 2
        
        return phase
    
    def extract_decoherence_from_convergence(self,
                                           mcts_trajectory: List[MCTSWaveEnsemble]) -> Tuple[List[float], List[float]]:
        """
        Extract natural decoherence from MCTS convergence.
        
        As MCTS converges:
        - Early: diverse waves → low purity (mixed state)
        - Late: similar waves → high purity (pure state)
        
        Args:
            mcts_trajectory: Sequence of wave ensembles over time
            
        Returns:
            Lists of purities and coherences over time
        """
        purities = []
        coherences = []
        
        for ensemble in mcts_trajectory:
            # Construct quantum state
            quantum_state = self.construct_from_wave_ensemble(ensemble)
            
            # Compute metrics
            purity = self.quantum_defs.compute_purity(quantum_state.density_matrix)
            coherence = self.quantum_defs.compute_coherence(quantum_state.density_matrix)
            
            purities.append(float(purity))
            coherences.append(float(coherence))
        
        return purities, coherences
    
    def compute_wave_diversity(self, wave_ensemble: MCTSWaveEnsemble) -> float:
        """
        Compute diversity of waves in ensemble.
        
        High diversity indicates:
        - High uncertainty
        - Good exploration
        - Mixed quantum state
        
        Args:
            wave_ensemble: Collection of waves
            
        Returns:
            Diversity score [0, 1]
        """
        if len(wave_ensemble.waves) < 2:
            return 0.0
        
        # Compute pairwise distances between wave amplitudes
        amplitudes = []
        for wave in wave_ensemble.waves:
            total = wave.action_visits.sum()
            if total > 0:
                probs = wave.action_visits / total
                amplitudes.append(torch.sqrt(probs))
        
        if len(amplitudes) < 2:
            return 0.0
        
        # Compute average pairwise distance
        total_distance = 0.0
        n_pairs = 0
        
        for i in range(len(amplitudes)):
            for j in range(i + 1, len(amplitudes)):
                distance = torch.norm(amplitudes[i] - amplitudes[j])
                total_distance += distance
                n_pairs += 1
        
        if n_pairs > 0:
            avg_distance = total_distance / n_pairs
            # Normalize by maximum possible distance (√2)
            diversity = float(avg_distance / np.sqrt(2))
            return min(1.0, diversity)
        
        return 0.0