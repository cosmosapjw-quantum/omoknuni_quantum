"""
Quantum Features for MCTS - Version 2.0 Implementation
=====================================================

This module implements the v2.0 quantum-enhanced MCTS based on:
- Discrete information time: τ(N) = log(N+2)
- Full PUCT action: S[γ] = -Σ[log N(s,a) + λ log P(a|s)]
- Power-law decoherence: ρᵢⱼ(N) ~ N^(-Γ₀)
- Physics-derived parameters from RG analysis
- Phase-aware adaptive strategies
- Neural network priors as external field

Key improvements:
- Auto-computed parameters from first principles
- Phase detection and adaptive strategies
- Envariance convergence criterion
- Reduced overhead with neural networks (1.3-1.8x)
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union, Callable, List
from dataclasses import dataclass, field
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class MCTSPhase(Enum):
    """MCTS phase based on simulation count"""
    QUANTUM = "quantum"       # N < N_c1
    CRITICAL = "critical"     # N_c1 < N < N_c2
    CLASSICAL = "classical"   # N > N_c2


@dataclass
class QuantumConfigV2:
    """Configuration for v2.0 quantum features"""
    # Core settings
    enable_quantum: bool = True
    quantum_level: str = 'tree_level'  # 'classical', 'tree_level', 'one_loop'
    
    # Physics parameters (None = auto-compute)
    hbar_eff: Optional[float] = None  # Auto: c_puct(N+2)/(√(N+1)log(N+2))
    coupling_strength: float = 0.3     # From RG fixed point
    temperature_mode: str = 'annealing'  # 'fixed', 'annealing'
    initial_temperature: float = 1.0
    
    # Neural network integration
    use_neural_prior: bool = True
    prior_coupling: Union[str, float] = 'auto'  # 'auto' uses c_puct
    
    # Interference settings
    interference_method: str = 'minhash'
    num_hash_functions: Optional[int] = None  # None = auto: √(b·L)
    
    # Phase detection
    enable_phase_adaptation: bool = True
    phase_transition_smoothing: float = 0.1
    
    # Performance
    min_wave_size: int = 32
    optimal_wave_size: int = 3072
    use_mixed_precision: bool = True
    device: str = 'cuda'
    
    # Game parameters (for auto-computation)
    branching_factor: Optional[int] = None
    avg_game_length: Optional[int] = None
    c_puct: Optional[float] = None  # Auto: √(2·log(b))
    
    # Decoherence
    decoherence_base_rate: float = 0.01
    power_law_exponent: Optional[float] = None  # Auto: 2c_puct·σ²_eval·T₀
    
    # Envariance
    envariance_threshold: float = 1e-3
    envariance_check_interval: Optional[int] = None  # Auto: √(N+1)
    
    # Cache and optimization
    cache_quantum_corrections: bool = True
    fast_mode: bool = True
    enable_profiling: bool = False
    log_level: str = 'INFO'


class DiscreteTimeEvolution:
    """Handles discrete information time dynamics"""
    
    def __init__(self, config: QuantumConfigV2):
        self.config = config
        self.T0 = config.initial_temperature
        self.eps = 1e-8  # Numerical stability
    
    def information_time(self, N: Union[int, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Compute information time τ(N) = log(N+2)"""
        if isinstance(N, torch.Tensor):
            return torch.log(N + 2)
        return math.log(N + 2)
    
    def time_derivative(self, N: Union[int, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Compute d/dτ = (N+2)d/dN"""
        return 1.0 / (N + 2)
    
    def compute_temperature(self, N: Union[int, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Compute temperature T(N) = T₀/log(N+2)"""
        if self.config.temperature_mode == 'fixed':
            return self.T0
        elif self.config.temperature_mode == 'annealing':
            tau = self.information_time(N)
            return self.T0 / (tau + self.eps)
        else:
            raise ValueError(f"Unknown temperature mode: {self.config.temperature_mode}")
    
    def compute_hbar_eff(self, N: Union[int, torch.Tensor], 
                         c_puct: Optional[float] = None) -> Union[float, torch.Tensor]:
        """Compute effective Planck constant"""
        if self.config.hbar_eff is not None:
            return self.config.hbar_eff
        
        if c_puct is None:
            c_puct = self.config.c_puct
            if c_puct is None:
                raise ValueError("c_puct must be provided or set in config")
        
        tau = self.information_time(N)
        if isinstance(N, torch.Tensor):
            return c_puct * (N + 2) / (torch.sqrt(N + 1) * tau)
        else:
            return c_puct * (N + 2) / (math.sqrt(N + 1) * tau)


class PhaseDetector:
    """Detects current MCTS phase based on simulation count"""
    
    def __init__(self, config: QuantumConfigV2):
        self.config = config
        self._critical_points_cache = {}
    
    def compute_critical_points(self, branching_factor: int, c_puct: float, 
                               has_neural_prior: bool = True) -> Tuple[float, float]:
        """Compute phase transition points N_c1 and N_c2"""
        # Check cache
        cache_key = (branching_factor, c_puct, has_neural_prior)
        if cache_key in self._critical_points_cache:
            return self._critical_points_cache[cache_key]
        
        # Base critical points
        N_c1 = branching_factor * math.exp(math.sqrt(2 * math.pi) / c_puct) - 2
        N_c2 = branching_factor**2 * math.exp(4 * math.pi / c_puct**2) - 2
        
        # Adjust for neural network priors
        if has_neural_prior and self.config.use_neural_prior:
            # Get effective prior coupling
            if self.config.prior_coupling == 'auto':
                lambda_eff = c_puct * 0.8
            else:
                lambda_eff = float(self.config.prior_coupling)
            
            # Shift critical points
            N_c1 *= (1 + lambda_eff / (2 * math.pi))
            N_c2 *= (1 + lambda_eff / math.pi)
        
        # Cache result
        self._critical_points_cache[cache_key] = (N_c1, N_c2)
        return N_c1, N_c2
    
    def detect_phase(self, N: int, branching_factor: int, c_puct: float,
                    has_neural_prior: bool = True) -> MCTSPhase:
        """Determine current phase of MCTS"""
        N_c1, N_c2 = self.compute_critical_points(branching_factor, c_puct, has_neural_prior)
        
        if N < N_c1:
            return MCTSPhase.QUANTUM
        elif N < N_c2:
            return MCTSPhase.CRITICAL
        else:
            return MCTSPhase.CLASSICAL
    
    def get_phase_config(self, phase: MCTSPhase) -> Dict[str, Any]:
        """Get phase-specific configuration"""
        if phase == MCTSPhase.QUANTUM:
            return {
                'quantum_strength': 1.0,
                'temperature_boost': 2.0,
                'prior_trust': 0.5,  # Less trust in priors
                'batch_size': 32,
                'interference_strength': 0.1
            }
        elif phase == MCTSPhase.CRITICAL:
            return {
                'quantum_strength': 0.5,
                'temperature_boost': 1.0,
                'prior_trust': 1.0,  # Standard PUCT weight
                'batch_size': 16,
                'interference_strength': 0.05
            }
        else:  # CLASSICAL
            return {
                'quantum_strength': 0.1,
                'temperature_boost': 0.5,
                'prior_trust': 1.5,  # High prior trust
                'batch_size': 8,
                'interference_strength': 0.01
            }


class OptimalParameters:
    """Computes optimal parameters from physics theory"""
    
    @staticmethod
    def compute_c_puct(branching_factor: int, N: Optional[int] = None) -> float:
        """Compute optimal c_puct from branching factor"""
        c_puct = math.sqrt(2 * math.log(branching_factor))
        
        # RG flow correction if N provided
        if N is not None:
            N_c = branching_factor * math.exp(math.sqrt(2 * math.pi) / c_puct) - 2
            c_puct *= (1 + 1 / (4 * math.log(N_c)))
        
        return c_puct
    
    @staticmethod
    def compute_num_hashes(branching_factor: int, avg_game_length: int,
                          has_neural_network: bool = True, prior_strength: float = 1.0) -> int:
        """Compute optimal number of hash functions"""
        K_base = int(math.sqrt(branching_factor * avg_game_length))
        
        if has_neural_network:
            # Reduce hash count based on prior strength
            reduction = 1 - prior_strength / (2 * math.pi * math.sqrt(2 * math.log(branching_factor)))
            K_opt = int(K_base * max(0.5, reduction))
        else:
            K_opt = K_base
        
        return max(4, K_opt)  # Minimum 4 hashes
    
    @staticmethod
    def phase_kick_schedule(N: int) -> float:
        """Compute phase kick probability"""
        return min(0.1, 1.0 / math.sqrt(N + 1))
    
    @staticmethod
    def update_interval(N: int) -> int:
        """Compute update interval for quantum parameters"""
        return int(math.sqrt(N + 1))


class QuantumMCTSV2:
    """Version 2.0 Quantum MCTS implementation with discrete time and full PUCT"""
    
    def __init__(self, config: Optional[QuantumConfigV2] = None):
        """Initialize v2.0 quantum MCTS"""
        if isinstance(config, dict):
            self.config = QuantumConfigV2(**config)
        else:
            self.config = config or QuantumConfigV2()
        
        # Set device
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else 'cpu'
        )
        
        # Initialize components
        self.time_evolution = DiscreteTimeEvolution(self.config)
        self.phase_detector = PhaseDetector(self.config)
        
        # Auto-compute parameters if needed
        self._auto_compute_parameters()
        
        # Initialize state
        self.total_simulations = 0
        self.current_phase = MCTSPhase.QUANTUM
        self.last_update_N = 0
        
        # Statistics
        self.stats = {
            'phase_transitions': 0,
            'quantum_applications': 0,
            'envariance_checks': 0,
            'convergence_reached': False
        }
        
        # Pre-compute tables for efficiency
        self._init_quantum_tables()
        
        # Pre-cache phase configurations
        self._init_phase_configs()
        
        logger.debug(f"Initialized QuantumMCTSV2 on {self.device}")
        logger.debug(f"Auto-computed parameters: c_puct={self.config.c_puct:.3f}, "
                    f"num_hashes={self.config.num_hash_functions}")
    
    def _auto_compute_parameters(self):
        """Auto-compute parameters from game properties"""
        # c_puct
        if self.config.c_puct is None and self.config.branching_factor is not None:
            self.config.c_puct = OptimalParameters.compute_c_puct(
                self.config.branching_factor
            )
        
        # Number of hash functions
        if (self.config.num_hash_functions is None and 
            self.config.branching_factor is not None and
            self.config.avg_game_length is not None):
            self.config.num_hash_functions = OptimalParameters.compute_num_hashes(
                self.config.branching_factor,
                self.config.avg_game_length,
                self.config.use_neural_prior
            )
        
        # Prior coupling
        if self.config.prior_coupling == 'auto' and self.config.c_puct is not None:
            self.config.prior_coupling = self.config.c_puct
        
        # Update interval
        if self.config.envariance_check_interval is None:
            self.config.envariance_check_interval = lambda N: int(math.sqrt(N + 1))
    
    def _init_quantum_tables(self):
        """Pre-compute quantum corrections for efficiency"""
        logger.debug("Pre-computing quantum lookup tables for v2.0...")
        
        # Always pre-compute critical tables regardless of cache setting
        max_N = 100000
        max_visits = 10000
        N_range = torch.arange(0, max_N, device=self.device, dtype=torch.float32)
        
        # Pre-compute information time τ(N) = log(N+2)
        self.tau_table = torch.log(N_range + 2)
        self.log_N_table = torch.log(N_range[1:] + 2)  # 1-based for compatibility
        self.sqrt_N_table = torch.sqrt(N_range[1:] + 1)  # 1-based for compatibility
        
        # Pre-compute temperature values for annealing mode
        if self.config.temperature_mode == 'annealing':
            self.temperature_table = self.config.initial_temperature / (self.tau_table + 1e-8)
        else:
            self.temperature_table = torch.full((max_N,), self.config.initial_temperature, device=self.device)
        
        # Pre-compute hbar_eff factors (without c_puct which may vary)
        # hbar_eff = c_puct * (N+2) / (sqrt(N+1) * log(N+2))
        self.hbar_factors = torch.where(
            N_range > 0,
            (N_range + 2) / (torch.sqrt(N_range + 1) * self.tau_table),
            torch.ones_like(N_range)
        )
        
        # Phase kick probabilities with 1-based indexing for compatibility
        N_range_1based = torch.arange(1, max_N + 1, device=self.device, dtype=torch.float32)
        self.phase_kick_table = torch.minimum(
            torch.tensor(0.1, device=self.device),
            1.0 / torch.sqrt(N_range_1based + 1)
        )
        
        if not self.config.cache_quantum_corrections:
            logger.debug("Basic tables pre-computed (cache_quantum_corrections=False)")
            return
        
        # Pre-compute power-law decoherence factors for common gamma values
        visit_range = torch.arange(0, max_visits, device=self.device, dtype=torch.float32)
        self.decoherence_tables = {}
        
        # Common gamma values based on c_puct ranges
        gamma_values = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
        for gamma in gamma_values:
            # Compute N^(-gamma) with numerical stability
            self.decoherence_tables[gamma] = torch.where(
                visit_range > 0,
                visit_range.pow(-gamma),
                torch.ones_like(visit_range)
            )
        
        # Get closest decoherence table for current gamma
        if self.config.power_law_exponent is not None:
            gamma = self.config.power_law_exponent
            closest_gamma = min(gamma_values, key=lambda x: abs(x - gamma))
            self.decoherence_table = self.decoherence_tables[closest_gamma]
        else:
            # Default to gamma=0.5
            self.decoherence_table = self.decoherence_tables[0.5]
        
        # Pre-compute critical points for phase transitions
        if self.config.branching_factor and self.config.c_puct:
            self._critical_points = self.phase_detector.compute_critical_points(
                self.config.branching_factor,
                self.config.c_puct,
                self.config.use_neural_prior
            )
        else:
            self._critical_points = None
        
        # Pre-allocate commonly used tensors for different batch sizes
        self._preallocated = {}
        for size in [32, 64, 128, 256, 512, 1024, 2048, 3072]:
            self._preallocated[size] = {
                'low_visit_mask': torch.zeros(size, device=self.device, dtype=torch.bool),
                'random_phases': torch.zeros(size, device=self.device),
                'phase_kicks': torch.zeros(size, device=self.device),
                'ones': torch.ones(size, device=self.device),
                'zeros': torch.zeros(size, device=self.device),
            }
        
        logger.debug(f"Pre-computed tables: tau_table={len(self.tau_table)}, "
                    f"hbar_factors={len(self.hbar_factors)}, "
                    f"decoherence_tables={len(self.decoherence_tables)} gamma values")
    
    def _init_phase_configs(self):
        """Pre-cache all phase configurations to avoid repeated computation"""
        self.phase_configs = {
            MCTSPhase.QUANTUM: self.phase_detector.get_phase_config(MCTSPhase.QUANTUM),
            MCTSPhase.CRITICAL: self.phase_detector.get_phase_config(MCTSPhase.CRITICAL),
            MCTSPhase.CLASSICAL: self.phase_detector.get_phase_config(MCTSPhase.CLASSICAL)
        }
        
        # Current phase config reference (updated when phase changes)
        self._current_phase_config = self.phase_configs[self.current_phase]
        
        logger.debug(f"Pre-cached phase configurations: {list(self.phase_configs.keys())}")
    
    def apply_quantum_to_selection_batch_cuda(
        self,
        q_values: torch.Tensor,
        visit_counts: torch.Tensor,
        priors: torch.Tensor,
        row_ptr: torch.Tensor,
        col_indices: torch.Tensor,
        c_puct_batch: torch.Tensor,
        parent_visits_batch: torch.Tensor,
        simulation_counts_batch: torch.Tensor,
        debug_logging: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched quantum selection using CUDA kernels for maximum performance
        
        This method directly interfaces with the CUDA kernels without any
        tensor creation overhead. All inputs must be properly batched tensors.
        
        Args:
            q_values: Q-values for all nodes
            visit_counts: Visit counts for all nodes
            priors: Prior probabilities for edges
            row_ptr: CSR format row pointers
            col_indices: CSR format column indices
            c_puct_batch: Batched c_puct values
            parent_visits_batch: Batched parent visit counts
            simulation_counts_batch: Batched simulation counts
            debug_logging: Enable debug logging
            
        Returns:
            Tuple of (selected_actions, selected_scores)
        """
        # Import here to avoid circular dependency
        from ..gpu.quantum_cuda_extension import batched_ucb_selection_quantum_v2
        
        # Use cached phase configuration
        phase_config = self._current_phase_config
        
        # Use CUDA kernel with pre-computed tables
        return batched_ucb_selection_quantum_v2(
            q_values=q_values,
            visit_counts=visit_counts,
            parent_visits=parent_visits_batch,
            priors=priors,
            row_ptr=row_ptr,
            col_indices=col_indices,
            c_puct_batch=c_puct_batch,
            simulation_counts=simulation_counts_batch,
            phase_config=phase_config,
            tau_table=getattr(self, 'tau_table', None),
            hbar_factors=getattr(self, 'hbar_factors', None),
            decoherence_table=getattr(self, 'decoherence_table', None),
            enable_quantum=self.config.enable_quantum,
            debug_logging=debug_logging
        )
    
    def apply_quantum_to_selection(
        self,
        q_values: torch.Tensor,
        visit_counts: torch.Tensor,
        priors: torch.Tensor,
        c_puct: Optional[float] = None,
        parent_visits: Optional[torch.Tensor] = None,
        simulation_count: Optional[int] = None
    ) -> torch.Tensor:
        """Apply v2.0 quantum features during MCTS selection
        
        Implements: UCB_quantum = Q + c_puct·P·√(log N_parent/N) + ℏ_eff/√(N+1)
        
        Args:
            q_values: Q-values for actions
            visit_counts: Visit counts for actions
            priors: Prior probabilities from neural network
            c_puct: Exploration constant (uses config if None)
            parent_visits: Total visits at parent node(s)
            simulation_count: Current total simulation count
            
        Returns:
            Quantum-enhanced UCB scores
        """
        # Use current simulation count if not provided
        if simulation_count is None:
            simulation_count = self.total_simulations
        
        # Use config c_puct if not provided
        if c_puct is None:
            c_puct = self.config.c_puct
            if c_puct is None:
                c_puct = math.sqrt(2)  # Default fallback
        
        # Update phase if needed
        self._update_phase(simulation_count)
        
        # Handle batched vs single cases
        is_batched = q_values.dim() > 1
        batch_size = q_values.shape[0] if is_batched else 1
        
        # Standard PUCT computation
        if parent_visits is None:
            parent_visits = visit_counts.sum(dim=-1, keepdim=True) if is_batched else visit_counts.sum()
        
        sqrt_parent = torch.sqrt(torch.log(parent_visits + 1))
        visit_factor = torch.sqrt(visit_counts + 1)
        
        # Classical PUCT exploration
        exploration = c_puct * priors * sqrt_parent / visit_factor
        
        # Check if quantum is enabled
        if not self.config.enable_quantum or self.config.quantum_level == 'classical':
            return q_values + exploration
        
        # Check batch size threshold
        if batch_size < self.config.min_wave_size:
            return q_values + exploration
        
        # Use cached phase configuration
        phase_config = self._current_phase_config
        
        # Compute quantum corrections
        with torch.amp.autocast(
            self.device.type,
            enabled=self.config.use_mixed_precision and self.device.type == 'cuda'
        ):
            # Use pre-computed hbar factors for efficiency
            if simulation_count < len(self.hbar_factors):
                hbar_eff = c_puct * self.hbar_factors[simulation_count]
            else:
                # Fallback for large N
                hbar_eff = self.time_evolution.compute_hbar_eff(simulation_count, c_puct)
            
            # Scale by phase-specific quantum strength
            hbar_eff = hbar_eff * phase_config['quantum_strength']
            
            # Quantum uncertainty bonus
            quantum_bonus = hbar_eff / visit_factor
            
            # Apply prior trust modification based on phase
            prior_weight = phase_config['prior_trust']
            exploration = exploration * prior_weight
            
            # Interference effects for low-visit nodes
            if self.config.quantum_level in ['tree_level', 'one_loop']:
                low_visit_mask = visit_counts < 10
                
                # Phase kick probability
                if simulation_count < len(self.phase_kick_table):
                    kick_prob = self.phase_kick_table[simulation_count]
                else:
                    kick_prob = OptimalParameters.phase_kick_schedule(simulation_count)
                
                # Apply phase kicks with pre-allocated tensors if possible
                if self.config.interference_method == 'phase_kick':
                    # Use pre-allocated tensors for common sizes
                    size = q_values.numel()
                    if size in self._preallocated and is_batched:
                        # Use pre-allocated random tensors
                        random_phases = self._preallocated[size]['random_phases']
                        random_phases.uniform_(0, 2 * 3.14159265)
                        phase_noise = torch.sin(random_phases).view_as(q_values) * phase_config['interference_strength']
                    else:
                        # Fallback to new allocation
                        phase_noise = torch.randn_like(q_values) * phase_config['interference_strength']
                    
                    phase_kick = torch.where(
                        low_visit_mask & (torch.rand_like(q_values) < kick_prob),
                        phase_noise,
                        torch.zeros_like(q_values)
                    )
                    quantum_bonus = quantum_bonus + phase_kick
            
            # One-loop corrections
            if self.config.quantum_level == 'one_loop':
                loop_correction = self._compute_one_loop_v2(
                    q_values, visit_counts, priors, hbar_eff
                )
                quantum_bonus = quantum_bonus + loop_correction
            
            # Combine all terms
            ucb_scores = q_values + exploration + quantum_bonus
        
        # Update statistics
        self.stats['quantum_applications'] += 1
        
        return ucb_scores
    
    def _compute_one_loop_v2(
        self,
        q_values: torch.Tensor,
        visit_counts: torch.Tensor,
        priors: torch.Tensor,
        hbar_eff: float
    ) -> torch.Tensor:
        """Compute one-loop corrections with full PUCT action"""
        # Effective action includes both visits and priors
        g = self.config.coupling_strength
        
        # Visit field fluctuations
        visit_fluct = -0.5 * hbar_eff * torch.log(visit_counts + 1) / (visit_counts + 1)
        
        # Prior field is external - no quantum corrections
        # But it affects the visit field through coupling
        prior_influence = g * torch.log(priors + 1e-8) / (4 * math.pi)
        
        # Power-law decoherence
        if self.config.power_law_exponent is not None:
            gamma = self.config.power_law_exponent
        else:
            # Auto-compute from theory
            sigma_Q = torch.std(q_values)
            T = self.time_evolution.compute_temperature(self.total_simulations)
            gamma = 2 * self.config.c_puct * sigma_Q**2 * T
        
        decoherence = (visit_counts + 1) ** (-gamma)
        
        return (visit_fluct + prior_influence) * decoherence
    
    def _update_phase(self, N: int):
        """Update current phase based on simulation count"""
        if not self.config.enable_phase_adaptation:
            return
        
        # Get game parameters
        b = self.config.branching_factor
        c_puct = self.config.c_puct
        
        if b is None or c_puct is None:
            return
        
        # Detect phase
        new_phase = self.phase_detector.detect_phase(
            N, b, c_puct, self.config.use_neural_prior
        )
        
        # Track phase transitions
        if new_phase != self.current_phase:
            self.current_phase = new_phase
            self._current_phase_config = self.phase_configs[self.current_phase]  # Update cached config
            self.stats['phase_transitions'] += 1
            logger.debug(f"Phase transition at N={N}: {self.current_phase.value}")
    
    def check_envariance(
        self,
        tree: Any,
        evaluators: Optional[List[Callable]] = None,
        threshold: Optional[float] = None
    ) -> bool:
        """Check if MCTS has reached envariant state (convergence)
        
        Args:
            tree: MCTS tree structure
            evaluators: List of evaluation functions (optional)
            threshold: Convergence threshold (uses config if None)
            
        Returns:
            True if converged
        """
        if threshold is None:
            threshold = self.config.envariance_threshold
        
        self.stats['envariance_checks'] += 1
        
        # Simple implementation - check policy stability
        # In full implementation, would check invariance under evaluator transformations
        if hasattr(tree, 'get_policy_entropy'):
            entropy = tree.get_policy_entropy()
            converged = entropy < threshold
            
            if converged and not self.stats['convergence_reached']:
                self.stats['convergence_reached'] = True
                logger.debug(f"Envariance reached at N={self.total_simulations}")
            
            return converged
        
        return False
    
    def update_simulation_count(self, N: int):
        """Update total simulation count and related parameters"""
        # Skip if no change
        if N == self.total_simulations:
            return
            
        self.total_simulations = N
        
        # Only update phase periodically (every 100 simulations)
        if self.config.enable_phase_adaptation and N - self.last_update_N >= 100:
            self._update_phase(N)
            self.last_update_N = N
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = dict(self.stats)
        stats.update({
            'current_phase': self.current_phase.value,
            'total_simulations': self.total_simulations,
            'current_temperature': self.time_evolution.compute_temperature(self.total_simulations),
            'current_hbar_eff': self.time_evolution.compute_hbar_eff(
                self.total_simulations, self.config.c_puct
            )
        })
        return stats
    
    def apply_quantum_to_evaluation(
        self,
        values: torch.Tensor,
        policies: torch.Tensor,
        state_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply quantum corrections to neural network outputs
        
        In v2.0, this respects the prior field interpretation - minimal modifications
        to preserve neural network guidance while adding exploration.
        """
        if not self.config.enable_quantum:
            return values, policies
        
        # Get phase configuration
        phase_config = self.phase_detector.get_phase_config(self.current_phase)
        
        # Temperature from information time
        T = self.time_evolution.compute_temperature(self.total_simulations)
        
        # Value fluctuations scaled by phase
        value_noise_scale = T * phase_config['quantum_strength'] * 0.02
        value_noise = torch.randn_like(values) * value_noise_scale
        values_enhanced = values + value_noise
        
        # Policy remains mostly unchanged (external field)
        # Only apply minimal temperature scaling
        if policies.dim() == 2 and phase_config['quantum_strength'] > 0.1:
            temp_scale = 1.0 + 0.1 * phase_config['quantum_strength']
            policies_enhanced = F.softmax(
                torch.log(policies + 1e-10) / temp_scale, dim=-1
            )
        else:
            policies_enhanced = policies
        
        return values_enhanced, policies_enhanced


def create_quantum_mcts_v2(
    enable_quantum: bool = True,
    branching_factor: Optional[int] = None,
    avg_game_length: Optional[int] = None,
    use_neural_network: bool = True,
    **kwargs
) -> QuantumMCTSV2:
    """Factory function to create v2.0 quantum MCTS
    
    Args:
        enable_quantum: Whether to enable quantum features
        branching_factor: Game branching factor (for auto-computation)
        avg_game_length: Average game length (for auto-computation)
        use_neural_network: Whether neural network is being used
        **kwargs: Additional config parameters
        
    Returns:
        QuantumMCTSV2 instance
    """
    # Auto-compute optimal parameters if game properties provided
    if branching_factor is not None:
        kwargs['branching_factor'] = branching_factor
        if 'c_puct' not in kwargs:
            kwargs['c_puct'] = OptimalParameters.compute_c_puct(branching_factor)
    
    if avg_game_length is not None:
        kwargs['avg_game_length'] = avg_game_length
    
    kwargs['use_neural_prior'] = use_neural_network
    kwargs['enable_quantum'] = enable_quantum
    
    config = QuantumConfigV2(**kwargs)
    
    return QuantumMCTSV2(config)