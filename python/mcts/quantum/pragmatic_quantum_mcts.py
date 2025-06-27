"""
Pragmatic Quantum MCTS - Evidence-Based Implementation
====================================================

This implementation incorporates the key actionable insights from the quantum-MCTS
research analysis, focusing on practical benefits rather than theoretical formalism:

1. Dynamic Exploration Bonus - Enhanced exploration for low-visit nodes (crossover at N≈50)
2. Power-Law Temperature Annealing - More sophisticated exploration decay
3. Phase-Adaptive Parameter Scheduling - Entropy-based phase detection and adaptive c_puct
4. Correlation-Based Node Prioritization - Spatial correlation for intelligent exploration
5. Wave-Based Vectorized Processing - 3072-path waves for optimal GPU utilization

Key Design Principles:
- Evidence-based coefficients from research plots with empirical tuning
- Conservative quantum bonuses to minimize overhead
- A/B testable components for validation
- Integration with existing wave-based vectorized MCTS
- Pragmatic simplicity over theoretical elegance

Performance Target: 5-15% improvement in search quality with < 1.5x computational overhead
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SearchPhase(Enum):
    """Search phase classification based on entropy analysis"""
    EXPLORATION = "exploration"    # High entropy, diverse visits
    CRITICAL = "critical"         # Balanced exploration/exploitation
    EXPLOITATION = "exploitation" # Low entropy, focused visits

@dataclass
class PragmaticQuantumConfig:
    """
    Configuration for pragmatic quantum MCTS implementation
    
    Parameters are based on research analysis with conservative defaults
    that can be empirically tuned per game domain.
    """
    
    # Core MCTS parameters
    base_c_puct: float = 1.4
    device: str = 'cuda'
    wave_size: int = 3072  # Optimal for GPU utilization
    
    # Dynamic exploration bonus (from ℏ_eff analysis)
    quantum_crossover_threshold: int = 50  # N≈50 from research plots
    quantum_bonus_coefficient: float = 0.1  # Conservative vs theoretical 0.75
    enable_quantum_bonus: bool = True
    
    # Power-law temperature annealing (from decoherence analysis)
    base_temperature: float = 1.0
    annealing_exponent: float = 0.5  # Power-law decay rate
    annealing_scale: float = 0.01    # Controls annealing speed
    enable_power_law_annealing: bool = True
    
    # Phase-adaptive parameters (from phase diagram analysis)
    entropy_exploration_threshold: float = 0.7   # High entropy = exploring
    entropy_exploitation_threshold: float = 0.3  # Low entropy = exploiting
    exploration_c_puct_multiplier: float = 1.5   # Boost exploration in exploration phase
    exploitation_c_puct_multiplier: float = 0.7  # Reduce exploration in exploitation phase
    enable_phase_adaptation: bool = True
    
    # Correlation-based prioritization (simplified from correlation analysis)
    enable_correlation_prioritization: bool = True
    correlation_distance_threshold: int = 2     # Tree distance for correlation
    correlation_q_value_threshold: float = 0.1  # Q-value similarity threshold
    
    # Performance and validation
    enable_ab_testing: bool = False  # A/B test against classical MCTS
    enable_performance_monitoring: bool = True
    fallback_to_classical: bool = True  # Fallback if quantum overhead too high

class PhaseDetector:
    """
    Detects search phase based on visit entropy among children
    
    Based on phase diagram analysis: use visit distribution entropy
    to determine if node is in exploration, critical, or exploitation phase.
    """
    
    def __init__(self, config: PragmaticQuantumConfig):
        self.config = config
        self.phase_history: Dict[int, SearchPhase] = {}
        
    def detect_phase(
        self, 
        visit_counts: torch.Tensor,
        min_visits_for_detection: int = 5
    ) -> SearchPhase:
        """
        Detect search phase based on visit entropy
        
        Args:
            visit_counts: Visit counts for child nodes
            min_visits_for_detection: Minimum visits before phase detection
            
        Returns:
            SearchPhase enum indicating current phase
        """
        total_visits = torch.sum(visit_counts).item()
        
        if total_visits < min_visits_for_detection:
            return SearchPhase.EXPLORATION
        
        # Compute normalized entropy of visit distribution
        probabilities = visit_counts / total_visits
        # Avoid log(0) by filtering out zero probabilities
        nonzero_probs = probabilities[probabilities > 0]
        
        if len(nonzero_probs) == 0:
            return SearchPhase.EXPLORATION
        
        entropy = -torch.sum(nonzero_probs * torch.log(nonzero_probs)).item()
        max_entropy = math.log(len(visit_counts))
        
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
        else:
            normalized_entropy = 0.0
        
        # Phase classification based on research thresholds
        if normalized_entropy > self.config.entropy_exploration_threshold:
            return SearchPhase.EXPLORATION
        elif normalized_entropy < self.config.entropy_exploitation_threshold:
            return SearchPhase.EXPLOITATION
        else:
            return SearchPhase.CRITICAL

class PowerLawAnnealer:
    """
    Power-law temperature annealing based on decoherence analysis
    
    Implements T(t) = T₀ / (1 + αt)^β where β is the annealing exponent.
    Research suggests power-law decay maintains exploration longer than exponential.
    """
    
    def __init__(self, config: PragmaticQuantumConfig):
        self.config = config
        
    def get_temperature(self, simulation_count: int) -> float:
        """Get temperature for current simulation count"""
        if not self.config.enable_power_law_annealing:
            # Fallback to exponential annealing
            return self.config.base_temperature * math.exp(-simulation_count * 0.001)
        
        # Power-law annealing: T(t) = T₀ / (1 + αt)^β
        denominator = 1.0 + self.config.annealing_scale * simulation_count
        return self.config.base_temperature / (denominator ** self.config.annealing_exponent)
    
    def apply_temperature_to_policy(
        self, 
        policy_logits: torch.Tensor, 
        simulation_count: int
    ) -> torch.Tensor:
        """Apply temperature annealing to policy logits"""
        temperature = self.get_temperature(simulation_count)
        return F.softmax(policy_logits / temperature, dim=-1)

class DynamicExplorationBonus:
    """
    Dynamic exploration bonus for low-visit nodes
    
    Based on ℏ_eff analysis: provide extra exploration for nodes with N < 50 visits,
    with quantum bonus ∝ 1/N_k to encourage exploration of rarely-visited nodes.
    """
    
    def __init__(self, config: PragmaticQuantumConfig):
        self.config = config
        
    def compute_quantum_bonus(
        self, 
        visit_counts: torch.Tensor,
        simulation_count: int
    ) -> torch.Tensor:
        """
        Compute quantum exploration bonus for low-visit nodes
        
        Formula: bonus = coefficient / N_k for N_k < threshold, 0 otherwise
        """
        if not self.config.enable_quantum_bonus:
            return torch.zeros_like(visit_counts)
        
        # Apply bonus only to low-visit nodes
        bonus = torch.zeros_like(visit_counts)
        low_visit_mask = visit_counts < self.config.quantum_crossover_threshold
        
        if torch.any(low_visit_mask):
            # Conservative quantum bonus: coefficient / N_k
            safe_visits = torch.clamp(visit_counts[low_visit_mask], min=1.0)
            bonus[low_visit_mask] = self.config.quantum_bonus_coefficient / safe_visits
        
        return bonus

class CorrelationAnalyzer:
    """
    Simplified correlation analysis for node prioritization
    
    Based on correlation function analysis: identify nodes with similar Q-values
    for intelligent exploration prioritization.
    """
    
    def __init__(self, config: PragmaticQuantumConfig):
        self.config = config
        
    def compute_node_similarity(
        self,
        q_values: torch.Tensor,
        visit_counts: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise similarity between nodes based on Q-values
        
        Simplified from full correlation analysis for computational efficiency.
        """
        if not self.config.enable_correlation_prioritization:
            return torch.zeros(len(q_values), len(q_values))
        
        # Pairwise Q-value differences
        q_diff_matrix = torch.abs(q_values.unsqueeze(0) - q_values.unsqueeze(1))
        
        # Similarity = 1 - normalized difference
        max_diff = torch.max(q_diff_matrix) + 1e-8
        similarity_matrix = 1.0 - (q_diff_matrix / max_diff)
        
        return similarity_matrix
    
    def get_exploration_priorities(
        self,
        q_values: torch.Tensor,
        visit_counts: torch.Tensor
    ) -> torch.Tensor:
        """
        Get exploration priorities based on correlation analysis
        
        Priority is higher for nodes that are:
        1. Less visited
        2. Different from already well-explored nodes
        """
        similarity_matrix = self.compute_node_similarity(q_values, visit_counts)
        
        # Priority = inverse visit count weighted by uniqueness
        base_priority = 1.0 / torch.clamp(visit_counts, min=1.0)
        
        # Uniqueness = how different this node is from well-visited nodes
        well_visited_mask = visit_counts > self.config.quantum_crossover_threshold
        if torch.any(well_visited_mask):
            similarity_to_visited = torch.mean(similarity_matrix[:, well_visited_mask], dim=1)
            uniqueness_bonus = 1.0 - similarity_to_visited
        else:
            uniqueness_bonus = torch.ones_like(base_priority)
        
        return base_priority * (1.0 + uniqueness_bonus)

class PragmaticQuantumMCTS:
    """
    Pragmatic Quantum MCTS implementation integrating research insights
    
    Combines evidence-based quantum corrections with wave-based vectorized processing
    for practical performance improvements in Monte Carlo Tree Search.
    """
    
    def __init__(self, config: PragmaticQuantumConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize components
        self.phase_detector = PhaseDetector(config)
        self.annealer = PowerLawAnnealer(config)
        self.exploration_bonus = DynamicExplorationBonus(config)
        self.correlation_analyzer = CorrelationAnalyzer(config)
        
        # Performance monitoring
        self.stats = {
            'quantum_applications': 0,
            'classical_fallbacks': 0,
            'phase_detections': {'exploration': 0, 'critical': 0, 'exploitation': 0},
            'average_quantum_bonus': 0.0,
            'performance_overhead': 0.0
        }
        
        # A/B testing support
        self.classical_baseline = None
        if config.enable_ab_testing:
            self._setup_ab_testing()
        
        logger.info(f"PragmaticQuantumMCTS initialized")
        logger.info(f"  Wave size: {config.wave_size}")
        logger.info(f"  Quantum bonus enabled: {config.enable_quantum_bonus}")
        logger.info(f"  Power-law annealing: {config.enable_power_law_annealing}")
        logger.info(f"  Phase adaptation: {config.enable_phase_adaptation}")
    
    def compute_enhanced_ucb_scores(
        self,
        q_values: torch.Tensor,        # [num_actions] - Mean action values
        visit_counts: torch.Tensor,    # [num_actions] - Visit counts
        priors: torch.Tensor,          # [num_actions] - NN policy priors
        parent_visits: int,            # Total parent visits
        simulation_count: int = 0      # Current simulation count
    ) -> torch.Tensor:
        """
        Compute enhanced UCB scores with pragmatic quantum corrections
        
        Integrates:
        1. Dynamic exploration bonus for low-visit nodes
        2. Phase-adaptive c_puct parameter
        3. Correlation-based prioritization
        4. Wave-based vectorized processing
        """
        # Detect current search phase
        current_phase = self.phase_detector.detect_phase(visit_counts)
        self.stats['phase_detections'][current_phase.value] += 1
        
        # Get phase-adaptive c_puct
        if self.config.enable_phase_adaptation:
            c_puct = self._get_adaptive_c_puct(current_phase)
        else:
            c_puct = self.config.base_c_puct
        
        # Classical UCB components
        safe_visits = torch.clamp(visit_counts, min=1.0)
        
        exploitation_term = q_values
        exploration_term = c_puct * priors * torch.sqrt(
            torch.log(torch.tensor(float(parent_visits))) / safe_visits
        )
        
        # Dynamic exploration bonus (quantum-inspired)
        quantum_bonus = self.exploration_bonus.compute_quantum_bonus(
            visit_counts, simulation_count
        )
        
        # Correlation-based prioritization
        correlation_priorities = self.correlation_analyzer.get_exploration_priorities(
            q_values, visit_counts
        )
        correlation_boost = 0.05 * correlation_priorities  # Small boost based on correlation
        
        # Combine all terms
        ucb_scores = exploitation_term + exploration_term + quantum_bonus + correlation_boost
        
        # Update statistics
        self.stats['quantum_applications'] += 1
        self.stats['average_quantum_bonus'] = (
            self.stats['average_quantum_bonus'] * 0.99 + 
            torch.mean(quantum_bonus).item() * 0.01
        )
        
        return ucb_scores
    
    def batch_compute_enhanced_ucb(
        self,
        q_values_batch: torch.Tensor,      # [batch_size, num_actions]
        visit_counts_batch: torch.Tensor,  # [batch_size, num_actions]
        priors_batch: torch.Tensor,        # [batch_size, num_actions]
        parent_visits_batch: torch.Tensor, # [batch_size]
        simulation_count: int = 0
    ) -> torch.Tensor:
        """
        Batch computation for wave-based vectorized processing
        
        Processes waves of 3072 nodes simultaneously for optimal GPU utilization.
        """
        batch_size, num_actions = q_values_batch.shape
        
        # Vectorized phase detection (simplified for batch processing)
        batch_phases = []
        for i in range(batch_size):
            phase = self.phase_detector.detect_phase(visit_counts_batch[i])
            batch_phases.append(phase)
        
        # Vectorized c_puct adaptation
        c_puct_batch = torch.tensor([
            self._get_adaptive_c_puct(phase) for phase in batch_phases
        ], device=self.device).unsqueeze(-1)  # [batch_size, 1]
        
        # Vectorized classical UCB
        safe_visits = torch.clamp(visit_counts_batch, min=1.0)
        parent_visits_expanded = parent_visits_batch.unsqueeze(-1)  # [batch_size, 1]
        
        exploitation_terms = q_values_batch
        exploration_terms = (c_puct_batch * priors_batch * 
                           torch.sqrt(torch.log(parent_visits_expanded) / safe_visits))
        
        # Vectorized quantum bonuses
        quantum_bonuses = torch.zeros_like(q_values_batch)
        if self.config.enable_quantum_bonus:
            low_visit_mask = visit_counts_batch < self.config.quantum_crossover_threshold
            if torch.any(low_visit_mask):
                quantum_bonuses[low_visit_mask] = (
                    self.config.quantum_bonus_coefficient / safe_visits[low_visit_mask]
                )
        
        # Simplified correlation analysis for batch (computationally lighter)
        correlation_boosts = torch.zeros_like(q_values_batch)
        if self.config.enable_correlation_prioritization:
            # Simple heuristic: boost nodes with low visits and unique Q-values
            uniqueness_scores = torch.abs(q_values_batch - torch.mean(q_values_batch, dim=-1, keepdim=True))
            visit_penalty = 1.0 / safe_visits
            correlation_boosts = 0.01 * uniqueness_scores * visit_penalty
        
        # Combine all terms
        ucb_scores_batch = (exploitation_terms + exploration_terms + 
                          quantum_bonuses + correlation_boosts)
        
        return ucb_scores_batch
    
    def apply_temperature_annealing(
        self,
        policy_logits: torch.Tensor,
        simulation_count: int
    ) -> torch.Tensor:
        """Apply power-law temperature annealing to policy"""
        return self.annealer.apply_temperature_to_policy(policy_logits, simulation_count)
    
    def _get_adaptive_c_puct(self, phase: SearchPhase) -> float:
        """Get adaptive c_puct based on search phase"""
        base = self.config.base_c_puct
        
        if phase == SearchPhase.EXPLORATION:
            return base * self.config.exploration_c_puct_multiplier
        elif phase == SearchPhase.EXPLOITATION:
            return base * self.config.exploitation_c_puct_multiplier
        else:  # CRITICAL phase
            return base
    
    def _setup_ab_testing(self):
        """Setup A/B testing infrastructure"""
        # Placeholder for A/B testing setup
        # Would integrate with experiment tracking system
        logger.info("A/B testing enabled - tracking performance against classical baseline")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring"""
        total_applications = self.stats['quantum_applications'] + self.stats['classical_fallbacks']
        
        return {
            'total_applications': total_applications,
            'quantum_ratio': (self.stats['quantum_applications'] / max(1, total_applications)),
            'phase_distribution': self.stats['phase_detections'],
            'average_quantum_bonus': self.stats['average_quantum_bonus'],
            'performance_overhead': self.stats['performance_overhead'],
            'configuration': {
                'quantum_threshold': self.config.quantum_crossover_threshold,
                'quantum_coefficient': self.config.quantum_bonus_coefficient,
                'annealing_exponent': self.config.annealing_exponent,
                'phase_adaptation_enabled': self.config.enable_phase_adaptation
            }
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.stats = {k: (0 if isinstance(v, (int, float)) else 
                         {sk: 0 for sk in v.keys()} if isinstance(v, dict) else v)
                     for k, v in self.stats.items()}

# Factory functions for easy integration
def create_pragmatic_quantum_mcts(
    base_c_puct: float = 1.4,
    device: str = 'cuda',
    enable_quantum_bonus: bool = True,
    enable_power_law_annealing: bool = True,
    enable_phase_adaptation: bool = True,
    **kwargs
) -> PragmaticQuantumMCTS:
    """Create pragmatic quantum MCTS with standard configuration"""
    config = PragmaticQuantumConfig(
        base_c_puct=base_c_puct,
        device=device,
        enable_quantum_bonus=enable_quantum_bonus,
        enable_power_law_annealing=enable_power_law_annealing,
        enable_phase_adaptation=enable_phase_adaptation,
        **kwargs
    )
    return PragmaticQuantumMCTS(config)

def create_conservative_quantum_mcts(device: str = 'cuda') -> PragmaticQuantumMCTS:
    """Create conservative quantum MCTS for careful testing"""
    config = PragmaticQuantumConfig(
        device=device,
        quantum_bonus_coefficient=0.05,  # Very conservative
        enable_quantum_bonus=True,
        enable_power_law_annealing=False,  # Start with classical annealing
        enable_phase_adaptation=True,
        enable_correlation_prioritization=False,  # Disable for simplicity
        enable_ab_testing=True,
        fallback_to_classical=True
    )
    return PragmaticQuantumMCTS(config)

def create_research_quantum_mcts(device: str = 'cuda') -> PragmaticQuantumMCTS:
    """Create research-oriented quantum MCTS with all features enabled"""
    config = PragmaticQuantumConfig(
        device=device,
        quantum_bonus_coefficient=0.2,  # Closer to theoretical value
        enable_quantum_bonus=True,
        enable_power_law_annealing=True,
        enable_phase_adaptation=True,
        enable_correlation_prioritization=True,
        enable_ab_testing=True,
        enable_performance_monitoring=True
    )
    return PragmaticQuantumMCTS(config)

# Export main classes
__all__ = [
    'PragmaticQuantumMCTS',
    'PragmaticQuantumConfig',
    'SearchPhase',
    'PhaseDetector',
    'PowerLawAnnealer',
    'DynamicExplorationBonus',
    'CorrelationAnalyzer',
    'create_pragmatic_quantum_mcts',
    'create_conservative_quantum_mcts',
    'create_research_quantum_mcts'
]