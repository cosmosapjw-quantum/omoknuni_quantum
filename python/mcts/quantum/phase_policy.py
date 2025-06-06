"""Unified phase-kicked prior policy implementation

This module implements the quantum-inspired phase-kicked prior policy that
adds phase based on value uncertainty to create interference patterns.
Includes both CPU and GPU implementations with automatic selection.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class PhaseConfig:
    """Configuration for phase-kicked priors (CPU mode)"""
    base_phase_strength: float = 0.1
    uncertainty_scaling: float = 1.0
    value_phase_coupling: float = 0.5
    enable_interference_patterns: bool = True
    phase_decay_rate: float = 0.95
    # Add GPU compatibility attributes
    potential_scale: float = 1.0
    hbar: float = 1.0
    mass: float = 1.0
    kick_strength: float = 0.1
    uncertainty_coupling: float = 2.0
    coherence_length: float = 5.0
    decoherence_rate: float = 0.01
    temperature: float = 0.1
    enable_path_interference: bool = True
    interference_range: int = 3
    phase_correlation_decay: float = 0.8


@dataclass
class QuantumPhaseConfig:
    """Configuration for quantum phase-kicked priors (GPU mode)"""
    # Quantum parameters
    hbar: float = 1.0  # Reduced Planck constant (normalized)
    mass: float = 1.0  # Effective "mass" of the quantum particle
    potential_scale: float = 1.0  # Scale for value-based potential
    
    # Phase kick parameters
    kick_strength: float = 0.1  # Base phase kick strength
    uncertainty_coupling: float = 2.0  # How uncertainty affects phase spread
    coherence_length: float = 5.0  # Coherence length for interference
    
    # Decay parameters
    decoherence_rate: float = 0.01  # Rate of decoherence with visits
    temperature: float = 0.1  # Effective temperature for thermal effects
    
    # Interference parameters
    enable_path_interference: bool = True
    interference_range: int = 3  # Range for action interference
    phase_correlation_decay: float = 0.8  # Decay of phase correlations


class PhaseKickedPolicy:
    """Unified phase-kicked prior policy with CPU/GPU support
    
    This adds complex-valued phases to action probabilities based on
    value uncertainty, creating quantum-like interference patterns that
    enhance exploration without requiring hyperparameter tuning.
    """
    
    def __init__(
        self, 
        device: Union[str, torch.device] = 'cuda',
        kick_strength: float = 0.1,
        config: Optional[Union[PhaseConfig, QuantumPhaseConfig]] = None
    ):
        """Initialize phase-kicked policy
        
        Args:
            device: Device for computation
            kick_strength: Kick strength parameter
            config: Phase configuration
        """
        if isinstance(device, str):
            device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.kick_strength = kick_strength
        self.use_gpu = device.type == 'cuda'
        
        if self.use_gpu:
            logger.info("Using GPU-accelerated quantum phase kicks")
            self.config = config or QuantumPhaseConfig(kick_strength=kick_strength)
            self._init_gpu_state()
        else:
            logger.info("Using CPU phase kick implementation")
            self.config = config or PhaseConfig(base_phase_strength=kick_strength)
            self.phase_history: Dict[str, np.ndarray] = {}
        
        # Statistics
        self.stats = {
            'phase_kicks_applied': 0,
            'avg_coherence': 0.0,
            'avg_interference': 0.0
        }
    
    def _init_gpu_state(self):
        """Initialize GPU-specific state"""
        # Wave function cache for maintaining coherence
        self.wave_functions = {}
        # Phase field for spatial correlations
        self.phase_field = None
    
    def apply_phase_kicks(
        self,
        priors: Union[torch.Tensor, Dict[int, float]],
        visit_counts: Optional[Union[torch.Tensor, Dict[int, float]]] = None,
        q_values: Optional[Union[torch.Tensor, Dict[int, float]]] = None,
        node: Optional['Node'] = None,
        evaluator: Optional['Evaluator'] = None,
        state_features: Optional[np.ndarray] = None
    ) -> Union[torch.Tensor, Dict[int, float]]:
        """Apply phase kicks to action probabilities
        
        Supports both tensor inputs (GPU mode) and dictionary inputs (CPU mode).
        
        Args:
            priors: Prior probabilities as tensor or dict
            visit_counts: Visit counts as tensor or dict
            q_values: Q-values as tensor or dict
            node: Current node (for CPU mode)
            evaluator: Neural network evaluator (for CPU mode)
            state_features: Optional state features
            
        Returns:
            Modified probabilities with phase kicks
        """
        # GPU tensor mode - if we have tensors with visit_counts and q_values, use GPU mode
        if isinstance(priors, torch.Tensor) and visit_counts is not None and q_values is not None:
            # Use GPU implementation even if use_gpu is False when we have tensor inputs
            return self._apply_phase_kicks_gpu(priors, visit_counts, q_values)
        
        # CPU mode or tensor conversion needed
        if isinstance(priors, torch.Tensor):
            # Convert to CPU dict format
            priors_np = priors.cpu().numpy()
            # Handle both 1D and multi-dimensional tensors
            if priors_np.ndim > 1:
                # For batched priors, process only first batch
                priors_flat = priors_np[0] if len(priors_np) > 0 else priors_np.flatten()
            else:
                priors_flat = priors_np
            action_probs = {i: float(p) for i, p in enumerate(priors_flat) if p > 0}
        else:
            action_probs = priors
            
        # CPU dictionary mode
        if not action_probs:
            return action_probs
            
        if node is None:
            raise ValueError("CPU mode requires node parameter")
            
        return self._apply_phase_kicks_cpu(node, action_probs, evaluator, state_features)
    
    def _apply_phase_kicks_gpu(
        self,
        priors: torch.Tensor,
        visit_counts: torch.Tensor,
        q_values: torch.Tensor
    ) -> torch.Tensor:
        """GPU implementation using quantum mechanics"""
        # Ensure 2D tensor
        if priors.dim() == 1:
            priors = priors.unsqueeze(0)
            visit_counts = visit_counts.unsqueeze(0)
            q_values = q_values.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, num_actions = priors.shape
        
        # Convert priors to wave function amplitudes (ensure full precision for complex operations)
        priors_full = priors.float()  # Ensure full precision
        psi = torch.sqrt(priors_full + 1e-10) * torch.exp(
            1j * torch.zeros_like(priors_full)
        )
        
        # Compute quantum phases (ensure consistent dtype)
        q_values_full = q_values.float() if q_values.dtype != torch.float32 else q_values
        visit_counts_full = visit_counts.float() if visit_counts.dtype != torch.float32 else visit_counts
        
        phases = self._compute_quantum_phases(
            q_values_full, visit_counts_full, batch_size, num_actions
        )
        
        # Apply phase kicks
        psi_kicked = psi * torch.exp(1j * phases)
        
        # Apply interference if enabled
        if self.config.enable_path_interference:
            psi_kicked = self._apply_interference(psi_kicked)
        
        # Apply decoherence based on visit counts
        psi_kicked = self._apply_decoherence(psi_kicked, visit_counts)
        
        # Extract probabilities using Born rule
        kicked_priors = torch.abs(psi_kicked) ** 2
        
        # Normalize
        kicked_priors = F.normalize(kicked_priors, p=1, dim=-1)
        
        # Convert back to original dtype if needed
        if priors.dtype != torch.float32:
            kicked_priors = kicked_priors.to(priors.dtype)
        
        # Update statistics
        self.stats['phase_kicks_applied'] += batch_size
        self.stats['avg_coherence'] = torch.mean(torch.abs(psi_kicked)).item()
        
        if squeeze_output:
            kicked_priors = kicked_priors.squeeze(0)
        
        return kicked_priors
    
    def _apply_phase_kicks_cpu(
        self,
        node: 'Node',
        action_probs: Dict[int, float],
        evaluator: Optional['Evaluator'] = None,
        state_features: Optional[np.ndarray] = None
    ) -> Dict[int, float]:
        """CPU implementation for backward compatibility"""
        # Estimate uncertainty
        uncertainty = self._estimate_uncertainty(node, evaluator)
        
        # Compute phase for each action
        actions = list(action_probs.keys())
        phases = self._compute_action_phases(
            node, actions, uncertainty, state_features
        )
        
        # Apply phases to probabilities
        modified_probs = self._apply_phases_to_probs(
            action_probs, phases, uncertainty
        )
        
        # Store phase history
        node_id = id(node)
        self.phase_history[node_id] = phases
        
        self.stats['phase_kicks_applied'] += 1
        
        return modified_probs
    
    def _compute_quantum_phases(
        self,
        q_values: torch.Tensor,
        visit_counts: torch.Tensor,
        batch_size: int,
        num_actions: int
    ) -> torch.Tensor:
        """Compute quantum mechanical phases (GPU)"""
        # Uncertainty from visit counts
        uncertainty = 1.0 / torch.sqrt(1.0 + visit_counts)
        
        # Effective potential from Q-values
        potential = self.config.potential_scale * q_values
        
        # Kinetic term
        kinetic = self.config.uncertainty_coupling * uncertainty
        
        # Classical action
        action = kinetic - potential
        
        # Quantum phase
        phase = action / self.config.hbar
        
        # Add quantum fluctuations
        phase_noise = torch.randn_like(phase) * self.config.temperature
        phase = phase + phase_noise * uncertainty
        
        # Position-dependent phase
        position_phase = self._compute_position_phase(batch_size, num_actions)
        phase = phase + self.kick_strength * position_phase
        
        return phase
    
    def _compute_position_phase(
        self,
        batch_size: int,
        num_actions: int
    ) -> torch.Tensor:
        """Compute position-dependent phase (GPU)"""
        coords = torch.arange(num_actions, device=self.device, dtype=torch.float32)
        coords = coords.unsqueeze(0).expand(batch_size, -1)
        
        # Normalize to [-π, π]
        coords_norm = (coords / num_actions - 0.5) * 2 * math.pi
        
        # Standing wave pattern
        phase = torch.sin(2.0 * coords_norm) + 0.5 * torch.sin(3.0 * coords_norm)
        
        return phase
    
    def _apply_interference(self, psi: torch.Tensor) -> torch.Tensor:
        """Apply quantum interference between nearby actions (GPU)"""
        batch_size, num_actions = psi.shape
        range_val = self.config.interference_range
        
        # Create interference kernel
        x = torch.arange(-range_val, range_val + 1, device=self.device, dtype=torch.float32)
        kernel = torch.exp(-x**2 / (2 * self.config.coherence_length**2))
        kernel = kernel / kernel.sum()
        
        # Apply as complex convolution
        psi_real = psi.real.unsqueeze(1)
        psi_imag = psi.imag.unsqueeze(1)
        
        # Pad for circular convolution
        pad_size = range_val
        psi_real_pad = F.pad(psi_real, (pad_size, pad_size), mode='circular')
        psi_imag_pad = F.pad(psi_imag, (pad_size, pad_size), mode='circular')
        
        # Convolve
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        psi_real_conv = F.conv1d(psi_real_pad, kernel, padding=0)
        psi_imag_conv = F.conv1d(psi_imag_pad, kernel, padding=0)
        
        # Recombine
        psi_interfered = torch.complex(
            psi_real_conv.squeeze(1),
            psi_imag_conv.squeeze(1)
        )
        
        # Mix with original
        coherence = self.config.phase_correlation_decay
        psi_final = coherence * psi_interfered + (1 - coherence) * psi
        
        # Update statistics
        self.stats['avg_interference'] = torch.mean(
            torch.abs(psi_final - psi)
        ).item()
        
        return psi_final
    
    def _apply_decoherence(
        self,
        psi: torch.Tensor,
        visit_counts: torch.Tensor
    ) -> torch.Tensor:
        """Apply decoherence effects (GPU)"""
        # Decoherence factor increases with visits
        decoherence = 1.0 - torch.exp(
            -self.config.decoherence_rate * visit_counts
        )
        
        # Classical state: pure amplitudes, no phase
        psi_classical = torch.abs(psi) * torch.exp(
            1j * torch.zeros_like(psi.real)
        )
        
        # Mix based on decoherence
        psi_decohered = (1 - decoherence) * psi + decoherence * psi_classical
        
        return psi_decohered
    
    def _estimate_uncertainty(self, node: 'Node', evaluator: Optional['Evaluator']) -> float:
        """Estimate uncertainty for CPU mode"""
        # Base uncertainty from visit count
        visit_uncertainty = 1.0 / (1.0 + np.sqrt(node.visit_count))
        
        # Value variance if available
        if hasattr(node, 'value_variance') and node.value_variance > 0:
            value_uncertainty = np.tanh(node.value_variance)
        else:
            value_uncertainty = 1.0 / (1.0 + abs(node.value()))
            
        # Combine uncertainties
        uncertainty = 0.7 * visit_uncertainty + 0.3 * value_uncertainty
        
        return uncertainty
    
    def _compute_action_phases(
        self,
        node: 'Node',
        actions: List[int],
        uncertainty: float,
        state_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute phases for CPU mode"""
        num_actions = len(actions)
        
        # Base phase proportional to uncertainty
        base_phase = self.config.base_phase_strength * uncertainty
        
        # Action-specific phase
        action_phases = np.zeros(num_actions)
        
        for i, action in enumerate(actions):
            # Deterministic phase based on action index
            action_phase = (action % 7) * np.pi / 7
            
            # Add value-based phase if child exists
            if hasattr(node, 'children') and action in node.children:
                child = node.children[action]
                child_value = child.value()
                value_phase = self.config.value_phase_coupling * np.tanh(child_value)
                action_phase += value_phase
                
            action_phases[i] = action_phase
            
        # Scale by base phase and uncertainty
        phases = base_phase * action_phases * self.config.uncertainty_scaling
        
        # Apply decay based on node visits
        decay_factor = self.config.phase_decay_rate ** np.sqrt(node.visit_count)
        phases *= decay_factor
        
        return phases
    
    def _apply_phases_to_probs(
        self,
        action_probs: Dict[int, float],
        phases: np.ndarray,
        uncertainty: float
    ) -> Dict[int, float]:
        """Apply phases to probabilities (CPU)"""
        actions = list(action_probs.keys())
        probs = np.array([action_probs[a] for a in actions])
        
        # Convert to complex amplitudes
        amplitudes = np.sqrt(probs)
        
        # Apply phase rotation
        complex_amplitudes = amplitudes * np.exp(1j * phases)
        
        if self.config.enable_interference_patterns:
            # Create interference between nearby actions
            interfered = complex_amplitudes.copy()
            
            for i in range(len(complex_amplitudes)):
                # Interference with neighbors
                if i > 0:
                    interfered[i] += 0.1 * uncertainty * complex_amplitudes[i-1]
                if i < len(complex_amplitudes) - 1:
                    interfered[i] += 0.1 * uncertainty * complex_amplitudes[i+1]
                    
            complex_amplitudes = interfered
            
        # Extract probabilities
        magnitudes = np.abs(complex_amplitudes)
        new_probs = magnitudes ** 2
        
        # Normalize
        if new_probs.sum() > 0:
            new_probs /= new_probs.sum()
        else:
            new_probs = probs
            
        # Convert back to dictionary
        modified_probs = {
            action: float(new_probs[i])
            for i, action in enumerate(actions)
        }
        
        return modified_probs
    
    def get_statistics(self) -> Dict[str, float]:
        """Get phase kick statistics"""
        return dict(self.stats)