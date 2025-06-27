"""
Discrete-Time Evolution with Causality Preservation
==================================================

Implements discrete-time evolution for quantum MCTS with strict causality preservation
using pre-update visit counts as specified in docs/v5.0/new_quantum_mcts.md.

Key Features:
- Discrete information time: τ(N) = log(N+2)
- Time derivative: δτ_N = 1/(N+2) 
- Causality preservation using pre-update visit counts
- Quantum-classical crossover dynamics
- Reversible discrete evolution operators
- Memory-efficient implementation with caching

Mathematical Foundation:
From docs/v5.0: "Absence of plaquettes (closed loops) implies the action Hessian 
is diagonal; there are no gauge constraints; and one-loop path integrals factorise child-wise."
"""

import torch
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class EvolutionRegime(Enum):
    """Evolution regime classification"""
    QUANTUM = "quantum"          # Full quantum evolution
    SEMICLASSICAL = "semiclassical"  # Mixed quantum-classical
    CLASSICAL = "classical"      # Pure classical evolution

@dataclass
class DiscreteTimeParams:
    """Parameters for discrete-time evolution"""
    
    # Information time parameters
    base_time: float = 0.0              # Base time offset
    time_scale: float = 1.0             # Overall time scaling
    
    # Causality preservation
    causality_buffer_size: int = 100    # Number of pre-update states to keep
    enable_causality_validation: bool = True
    causality_violation_threshold: float = 1e-6
    
    # Evolution parameters
    enable_reversible_evolution: bool = True
    max_evolution_steps: int = 1000
    evolution_tolerance: float = 1e-8
    
    # Performance settings
    use_cached_derivatives: bool = True
    cache_size: int = 10000
    enable_vectorized_ops: bool = True

class CausalityPreservingEvolution:
    """
    Implements discrete-time evolution with strict causality preservation
    
    The evolution preserves causality by using pre-update visit counts for all
    quantum computations, ensuring that no information from future states
    affects past decisions.
    """
    
    def __init__(self, params: DiscreteTimeParams):
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Causality preservation state
        self.pre_update_states: Dict[int, torch.Tensor] = {}
        self.causality_violations: List[Tuple[int, float]] = []
        
        # Caching for performance
        if params.use_cached_derivatives:
            self._derivative_cache: Dict[int, float] = {}
            self._tau_cache: Dict[int, float] = {}
        
        # Evolution tracking
        self.total_evolution_steps = 0
        self.successful_evolutions = 0
        self.causality_checks = 0
        
        logger.info(f"CausalityPreservingEvolution initialized")
        logger.info(f"  Causality validation: {params.enable_causality_validation}")
        logger.info(f"  Caching: {params.use_cached_derivatives}")
    
    def information_time(self, N: int) -> float:
        """
        Calculate discrete information time: τ(N) = log(N+2)
        
        From docs/v5.0: "τ = log(N_tot+2)" represents Euclidean time
        in the path integral formulation.
        """
        if self.params.use_cached_derivatives and N in self._tau_cache:
            return self._tau_cache[N]
        
        tau = self.params.base_time + self.params.time_scale * math.log(N + 2)
        
        if self.params.use_cached_derivatives and len(self._tau_cache) < self.params.cache_size:
            self._tau_cache[N] = tau
        
        return tau
    
    def time_derivative(self, N: int) -> float:
        """
        Calculate discrete time derivative: δτ_N = 1/(N+2)
        
        This represents the discrete time step for information time evolution.
        """
        if self.params.use_cached_derivatives and N in self._derivative_cache:
            return self._derivative_cache[N]
        
        derivative = self.params.time_scale / (N + 2)
        
        if self.params.use_cached_derivatives and len(self._derivative_cache) < self.params.cache_size:
            self._derivative_cache[N] = derivative
        
        return derivative
    
    def store_pre_update_state(self, simulation_count: int, visit_counts: torch.Tensor):
        """
        Store pre-update visit counts for causality preservation
        
        These counts represent the state BEFORE the current simulation step,
        ensuring that quantum corrections don't use future information.
        """
        # Store a copy to prevent inadvertent modification
        self.pre_update_states[simulation_count] = visit_counts.clone().detach()
        
        # Cleanup old states to manage memory
        if len(self.pre_update_states) > self.params.causality_buffer_size:
            oldest_key = min(self.pre_update_states.keys())
            del self.pre_update_states[oldest_key]
    
    def get_causality_safe_counts(self, simulation_count: int) -> Optional[torch.Tensor]:
        """
        Get visit counts that preserve causality for the given simulation
        
        Returns the pre-update counts that should be used for quantum calculations
        to ensure no future information leaks into past decisions.
        """
        return self.pre_update_states.get(simulation_count)
    
    def validate_causality(
        self, 
        current_counts: torch.Tensor,
        previous_counts: torch.Tensor,
        simulation_count: int
    ) -> bool:
        """
        Validate that causality is preserved between time steps
        
        Checks that visit counts are non-decreasing (causality constraint)
        and that no information from future states affects past computations.
        """
        if not self.params.enable_causality_validation:
            return True
        
        self.causality_checks += 1
        
        # Check for causality violations (visit counts decreasing)
        violations = current_counts < previous_counts
        if torch.any(violations):
            violation_magnitude = torch.max(previous_counts - current_counts).item()
            
            if violation_magnitude > self.params.causality_violation_threshold:
                self.causality_violations.append((simulation_count, violation_magnitude))
                logger.warning(f"Causality violation at N={simulation_count}: magnitude={violation_magnitude:.6f}")
                return False
        
        return True
    
    def evolve_discrete_step(
        self,
        visit_counts: torch.Tensor,
        simulation_count: int,
        quantum_corrections: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perform one discrete time evolution step with causality preservation
        
        Args:
            visit_counts: Current visit counts
            simulation_count: Current simulation count N
            quantum_corrections: Optional quantum corrections to apply
            
        Returns:
            Evolved visit counts with causality preserved
        """
        self.total_evolution_steps += 1
        
        # Store pre-update state for causality
        self.store_pre_update_state(simulation_count, visit_counts)
        
        # Get time parameters
        tau = self.information_time(simulation_count)
        dt = self.time_derivative(simulation_count)
        
        # Start with current state
        evolved_counts = visit_counts.clone()
        
        # Apply quantum corrections if provided
        if quantum_corrections is not None:
            # Scale corrections by discrete time step for proper evolution
            scaled_corrections = quantum_corrections * dt
            evolved_counts = evolved_counts + scaled_corrections
        
        # Ensure non-negativity (physical constraint)
        evolved_counts = torch.clamp(evolved_counts, min=0.0)
        
        # Validate causality if we have a previous state
        if simulation_count > 0:
            prev_counts = self.get_causality_safe_counts(simulation_count - 1)
            if prev_counts is not None:
                causality_ok = self.validate_causality(evolved_counts, prev_counts, simulation_count)
                if causality_ok:
                    self.successful_evolutions += 1
            else:
                self.successful_evolutions += 1
        else:
            self.successful_evolutions += 1
        
        return evolved_counts
    
    def compute_discrete_action(
        self,
        visit_counts: torch.Tensor,
        priors: torch.Tensor,
        q_values: torch.Tensor,
        simulation_count: int,
        kappa: float = 1.0,
        beta: float = 1.0
    ) -> torch.Tensor:
        """
        Compute discrete action using causality-preserving visit counts
        
        From docs/v5.0: S_cl = κ N_tot Σ(q_k-p_k)² - β Σ N_k Q_k
        where q_k = N_k/N_tot are the visit fractions.
        """
        # Use causality-safe counts
        safe_counts = self.get_causality_safe_counts(simulation_count)
        if safe_counts is None:
            safe_counts = visit_counts
        
        N_tot = torch.sum(safe_counts)
        if N_tot > 0:
            q_fractions = safe_counts / N_tot
        else:
            q_fractions = torch.zeros_like(safe_counts)
        
        # Classical action: κ N_tot Σ(q_k-p_k)² - β Σ N_k Q_k
        hellinger_term = kappa * N_tot * torch.sum((q_fractions - priors) ** 2)
        value_term = beta * torch.sum(safe_counts * q_values)
        
        return hellinger_term - value_term
    
    def detect_evolution_regime(self, simulation_count: int) -> EvolutionRegime:
        """
        Detect evolution regime based on simulation count and time scales
        
        The regime determines how quantum corrections are applied in the evolution.
        """
        tau = self.information_time(simulation_count)
        dt = self.time_derivative(simulation_count)
        
        # Quantum regime: fast evolution, small time steps
        if dt > 1e-3 and simulation_count < 1000:
            return EvolutionRegime.QUANTUM
        
        # Classical regime: slow evolution, large time steps  
        elif dt < 1e-4 or simulation_count > 10000:
            return EvolutionRegime.CLASSICAL
        
        # Semiclassical: intermediate regime
        else:
            return EvolutionRegime.SEMICLASSICAL
    
    def compute_crossover_dynamics(
        self,
        visit_counts: torch.Tensor,
        simulation_count: int
    ) -> Dict[str, float]:
        """
        Compute quantum-classical crossover dynamics
        
        Returns crossover parameters that determine the quantum/classical mixing.
        """
        regime = self.detect_evolution_regime(simulation_count)
        tau = self.information_time(simulation_count)
        dt = self.time_derivative(simulation_count)
        
        if regime == EvolutionRegime.QUANTUM:
            quantum_weight = 1.0
            classical_weight = 0.0
        elif regime == EvolutionRegime.CLASSICAL:
            quantum_weight = 0.1
            classical_weight = 0.9
        else:  # Semiclassical
            # Smooth crossover based on time derivative
            quantum_weight = min(1.0, dt * 1000)
            classical_weight = 1.0 - quantum_weight
        
        return {
            'regime': regime.value,
            'quantum_weight': quantum_weight,
            'classical_weight': classical_weight,
            'information_time': tau,
            'time_derivative': dt,
            'simulation_count': simulation_count
        }
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get evolution statistics for monitoring and debugging"""
        total_checks = self.causality_checks
        success_rate = (self.successful_evolutions / max(1, self.total_evolution_steps)) * 100
        
        return {
            'total_evolution_steps': self.total_evolution_steps,
            'successful_evolutions': self.successful_evolutions,
            'causality_checks': self.causality_checks,
            'causality_violations': len(self.causality_violations),
            'success_rate_percent': success_rate,
            'cache_sizes': {
                'tau_cache': len(getattr(self, '_tau_cache', {})),
                'derivative_cache': len(getattr(self, '_derivative_cache', {})),
                'pre_update_states': len(self.pre_update_states)
            },
            'latest_violations': self.causality_violations[-5:] if self.causality_violations else []
        }
    
    def reset(self):
        """Reset evolution state to initial conditions"""
        self.pre_update_states.clear()
        self.causality_violations.clear()
        
        if hasattr(self, '_derivative_cache'):
            self._derivative_cache.clear()
        if hasattr(self, '_tau_cache'):
            self._tau_cache.clear()
        
        self.total_evolution_steps = 0
        self.successful_evolutions = 0
        self.causality_checks = 0
        
        logger.info("Discrete-time evolution reset")

# Vectorized operations for performance
class VectorizedTimeEvolution:
    """Vectorized discrete-time evolution for batch processing"""
    
    @staticmethod
    def batch_information_time(N_batch: torch.Tensor, base_time: float = 0.0, time_scale: float = 1.0) -> torch.Tensor:
        """Vectorized information time calculation"""
        return base_time + time_scale * torch.log(N_batch.float() + 2)
    
    @staticmethod
    def batch_time_derivative(N_batch: torch.Tensor, time_scale: float = 1.0) -> torch.Tensor:
        """Vectorized time derivative calculation"""
        return time_scale / (N_batch.float() + 2)
    
    @staticmethod
    def batch_evolve_step(
        visit_counts_batch: torch.Tensor,  # [batch_size, num_actions]
        simulation_counts: torch.Tensor,   # [batch_size]
        quantum_corrections: Optional[torch.Tensor] = None  # [batch_size, num_actions]
    ) -> torch.Tensor:
        """Vectorized evolution step for batch processing"""
        
        # Compute time derivatives for each batch element
        dt_batch = VectorizedTimeEvolution.batch_time_derivative(simulation_counts).unsqueeze(-1)
        
        # Start with current counts
        evolved_batch = visit_counts_batch.clone()
        
        # Apply quantum corrections if provided
        if quantum_corrections is not None:
            scaled_corrections = quantum_corrections * dt_batch
            evolved_batch = evolved_batch + scaled_corrections
        
        # Ensure non-negativity
        evolved_batch = torch.clamp(evolved_batch, min=0.0)
        
        return evolved_batch

# Factory functions
def create_discrete_time_evolution(
    enable_causality_validation: bool = True,
    causality_buffer_size: int = 100,
    use_cached_derivatives: bool = True,
    **kwargs
) -> CausalityPreservingEvolution:
    """Create discrete-time evolution with standard parameters"""
    params = DiscreteTimeParams(
        enable_causality_validation=enable_causality_validation,
        causality_buffer_size=causality_buffer_size,
        use_cached_derivatives=use_cached_derivatives,
        **kwargs
    )
    return CausalityPreservingEvolution(params)

# Export main classes
__all__ = [
    'CausalityPreservingEvolution',
    'VectorizedTimeEvolution', 
    'DiscreteTimeParams',
    'EvolutionRegime',
    'create_discrete_time_evolution'
]