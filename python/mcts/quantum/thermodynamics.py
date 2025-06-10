"""
Thermodynamic Monitoring for QFT-MCTS
=====================================

This module implements thermodynamic monitoring and efficiency bounds for
the quantum field theoretic MCTS, tracking entropy production, work extraction,
and information-theoretic limits.

Key Features:
- Entropy production tracking
- Landauer bound verification
- Work extraction efficiency
- Information heat engine analysis
- GPU-accelerated thermodynamic computations

Based on: docs/qft-mcts-math-foundations.md Section 5.3
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
import time
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ThermodynamicsConfig:
    """Configuration for thermodynamic monitoring"""
    # Physical constants
    k_B: float = 1.0                       # Boltzmann constant (natural units)
    temperature: float = 1.0               # System temperature
    beta: float = 1.0                      # Inverse temperature (1/k_B T)
    
    # Landauer bound
    landauer_coefficient: float = 0.693    # k_B T ln(2) factor
    erasure_efficiency: float = 0.9        # Practical efficiency < 1
    
    # Work extraction
    max_work_per_bit: float = 0.693        # k_B T ln(2) theoretical max
    extraction_efficiency: float = 0.8      # Practical extraction efficiency
    
    # Monitoring parameters
    history_window: int = 1000             # Window for moving averages
    measurement_interval: int = 10         # Steps between measurements
    
    # Bounds and thresholds
    min_entropy: float = 1e-8              # Minimum entropy (avoid log(0))
    max_entropy_production_rate: float = 10.0  # Maximum allowed rate
    efficiency_warning_threshold: float = 0.5   # Warn if efficiency drops
    
    # GPU optimization
    batch_size: int = 256                  # Batch size for computations
    use_mixed_precision: bool = True       # FP16/FP32 optimization


class EntropyCalculator:
    """
    Calculates various entropy measures for MCTS states
    
    Tracks Shannon entropy, relative entropy, and mutual information
    to monitor information flow and dissipation.
    """
    
    def __init__(self, config: ThermodynamicsConfig, device: torch.device):
        self.config = config
        self.device = device
        
    def compute_shannon_entropy(
        self,
        probabilities: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute Shannon entropy H = -Σ p log p
        
        Args:
            probabilities: Probability distribution
            normalize: Whether to normalize probabilities
            
        Returns:
            Shannon entropy in nats
        """
        if normalize:
            probabilities = F.normalize(probabilities, p=1, dim=-1)
        
        # Avoid log(0) with small epsilon
        probs_safe = torch.clamp(probabilities, min=self.config.min_entropy)
        
        # H = -Σ p log p
        entropy = -torch.sum(probs_safe * torch.log(probs_safe), dim=-1)
        
        return entropy
    
    def compute_relative_entropy(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute KL divergence D(p||q) = Σ p log(p/q)
        
        This measures the entropy cost of using q instead of p.
        """
        if normalize:
            p = F.normalize(p, p=1, dim=-1)
            q = F.normalize(q, p=1, dim=-1)
        
        # Safe computation
        p_safe = torch.clamp(p, min=self.config.min_entropy)
        q_safe = torch.clamp(q, min=self.config.min_entropy)
        
        # D(p||q) = Σ p log(p/q)
        kl_div = torch.sum(p_safe * (torch.log(p_safe) - torch.log(q_safe)), dim=-1)
        
        return kl_div
    
    def compute_entropy_production(
        self,
        initial_dist: torch.Tensor,
        final_dist: torch.Tensor,
        reverse_dist: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute entropy production in state transition
        
        From fluctuation theorem: ΔS = D(P_forward||P_reverse)
        """
        # Forward entropy change
        delta_s_forward = self.compute_shannon_entropy(final_dist) - self.compute_shannon_entropy(initial_dist)
        
        # If reverse distribution provided, compute full entropy production
        if reverse_dist is not None:
            entropy_production = self.compute_relative_entropy(final_dist, reverse_dist)
        else:
            # Estimate from forward process only
            entropy_production = torch.abs(delta_s_forward)
        
        return {
            'entropy_production': entropy_production,
            'forward_entropy_change': delta_s_forward,
            'initial_entropy': self.compute_shannon_entropy(initial_dist),
            'final_entropy': self.compute_shannon_entropy(final_dist)
        }
    
    def compute_mutual_information(
        self,
        joint_dist: torch.Tensor,
        marginal_x: torch.Tensor,
        marginal_y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute mutual information I(X;Y) = H(X) + H(Y) - H(X,Y)
        
        Measures information shared between subsystems.
        """
        h_x = self.compute_shannon_entropy(marginal_x)
        h_y = self.compute_shannon_entropy(marginal_y)
        h_xy = self.compute_shannon_entropy(joint_dist.flatten())
        
        mutual_info = h_x + h_y - h_xy
        
        return torch.clamp(mutual_info, min=0)  # MI is non-negative


class LandauerBoundChecker:
    """
    Verifies Landauer's principle for information erasure
    
    Landauer bound: ΔS ≥ k_B ln(2) per bit erased
    This sets fundamental limits on computation efficiency.
    """
    
    def __init__(self, config: ThermodynamicsConfig, device: torch.device):
        self.config = config
        self.device = device
        self.erasure_events = deque(maxlen=config.history_window)
        
    def check_erasure_bound(
        self,
        bits_erased: torch.Tensor,
        entropy_produced: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Check if erasure satisfies Landauer bound
        
        Args:
            bits_erased: Number of bits erased
            entropy_produced: Entropy produced in process
            
        Returns:
            Dictionary with bound check results
        """
        # Theoretical minimum entropy: k_B T ln(2) per bit
        min_entropy_required = bits_erased * self.config.landauer_coefficient
        
        # Check if bound is satisfied
        bound_satisfied = entropy_produced >= min_entropy_required * self.config.erasure_efficiency
        
        # Compute efficiency
        if bits_erased > 0:
            efficiency = min_entropy_required / (entropy_produced + self.config.min_entropy)
            efficiency = torch.clamp(efficiency, max=1.0)
        else:
            efficiency = torch.ones_like(bits_erased)
        
        # Track violation events
        violations = ~bound_satisfied
        
        result = {
            'bound_satisfied': bound_satisfied,
            'min_entropy_required': min_entropy_required,
            'actual_entropy': entropy_produced,
            'efficiency': efficiency,
            'violation_rate': violations.float().mean()
        }
        
        # Update history
        self.erasure_events.append({
            'bits': bits_erased.mean().item(),
            'entropy': entropy_produced.mean().item(),
            'efficiency': efficiency.mean().item()
        })
        
        return result
    
    def compute_information_erasure(
        self,
        old_visit_dist: torch.Tensor,
        new_visit_dist: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute information erased in tree update
        
        Returns:
            Tuple of (bits_erased, entropy_produced)
        """
        # Normalize distributions
        old_dist = F.normalize(old_visit_dist, p=1, dim=-1)
        new_dist = F.normalize(new_visit_dist, p=1, dim=-1)
        
        # Information content change
        old_info = -torch.log2(old_dist + self.config.min_entropy)
        new_info = -torch.log2(new_dist + self.config.min_entropy)
        
        # Bits erased (reduction in information)
        info_reduction = torch.clamp(old_info - new_info, min=0)
        
        if node_mask is not None:
            info_reduction = info_reduction * node_mask
        
        bits_erased = info_reduction.sum(dim=-1)
        
        # Entropy production (always positive)
        entropy_calc = EntropyCalculator(self.config, self.device)
        entropy_prod = entropy_calc.compute_relative_entropy(new_dist, old_dist)
        
        return bits_erased, entropy_prod


class WorkExtractor:
    """
    Analyzes work extraction from information processing
    
    Based on information heat engines and Maxwell's demon,
    computes extractable work from information gain.
    """
    
    def __init__(self, config: ThermodynamicsConfig, device: torch.device):
        self.config = config
        self.device = device
        self.work_history = deque(maxlen=config.history_window)
        
    def compute_extractable_work(
        self,
        information_gain: torch.Tensor,
        entropy_cost: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute maximum extractable work from information
        
        From Szilard engine: W_max = k_B T I - T ΔS
        where I is information gain and ΔS is entropy cost.
        """
        # Maximum theoretical work
        max_work = self.config.k_B * self.config.temperature * information_gain
        
        # Subtract entropy cost
        entropy_penalty = self.config.temperature * entropy_cost
        
        # Net extractable work
        net_work = max_work - entropy_penalty
        
        # Apply practical efficiency
        extractable_work = net_work * self.config.extraction_efficiency
        
        # Compute efficiency metrics
        if information_gain > 0:
            info_to_work_efficiency = extractable_work / (self.config.k_B * self.config.temperature * information_gain)
            info_to_work_efficiency = torch.clamp(info_to_work_efficiency, min=0, max=1)
        else:
            info_to_work_efficiency = torch.zeros_like(information_gain)
        
        return {
            'max_theoretical_work': max_work,
            'entropy_cost': entropy_penalty,
            'net_work': net_work,
            'extractable_work': extractable_work,
            'efficiency': info_to_work_efficiency,
            'work_per_bit': extractable_work / (information_gain + self.config.min_entropy)
        }
    
    def analyze_mcts_work_cycle(
        self,
        selection_probs: torch.Tensor,
        expansion_info: torch.Tensor,
        backup_entropy: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Analyze work in MCTS cycle (selection → expansion → backup)
        
        Treats MCTS as an information heat engine.
        """
        entropy_calc = EntropyCalculator(self.config, self.device)
        
        # Selection: information gain from choosing path
        selection_entropy = entropy_calc.compute_shannon_entropy(selection_probs)
        selection_info = torch.log2(torch.tensor(len(selection_probs))) - selection_entropy
        
        # Expansion: information from new nodes
        expansion_gain = expansion_info
        
        # Backup: entropy production from value updates
        backup_cost = backup_entropy
        
        # Total information gain
        total_info_gain = selection_info + expansion_gain
        
        # Compute work
        work_result = self.compute_extractable_work(total_info_gain, backup_cost)
        
        # Compute cycle efficiency
        cycle_efficiency = work_result['extractable_work'] / (total_info_gain * self.config.k_B * self.config.temperature + self.config.min_entropy)
        
        # Update history
        self.work_history.append({
            'info_gain': total_info_gain.item(),
            'work': work_result['extractable_work'].item(),
            'efficiency': cycle_efficiency.item()
        })
        
        return {
            'selection_info': selection_info,
            'expansion_info': expansion_gain,
            'backup_entropy': backup_cost,
            'total_info_gain': total_info_gain,
            'work_extracted': work_result['extractable_work'],
            'cycle_efficiency': cycle_efficiency,
            'work_details': work_result
        }


class ThermodynamicEfficiencyAnalyzer:
    """
    Analyzes overall thermodynamic efficiency of MCTS
    
    Computes Carnot-like bounds and tracks efficiency trends.
    """
    
    def __init__(self, config: ThermodynamicsConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # Initialize calculators
        self.entropy_calc = EntropyCalculator(config, device)
        self.landauer_checker = LandauerBoundChecker(config, device)
        self.work_extractor = WorkExtractor(config, device)
        
        # Efficiency tracking
        self.efficiency_history = deque(maxlen=config.history_window)
        self.total_entropy_produced = 0.0
        self.total_work_extracted = 0.0
        self.total_information_processed = 0.0
        
    def compute_carnot_efficiency(
        self,
        hot_temperature: float,
        cold_temperature: float
    ) -> float:
        """
        Compute Carnot efficiency bound
        
        η_Carnot = 1 - T_cold/T_hot
        """
        if hot_temperature <= cold_temperature:
            return 0.0
        
        return 1.0 - cold_temperature / hot_temperature
    
    def compute_information_efficiency(
        self,
        info_processed: torch.Tensor,
        entropy_produced: torch.Tensor,
        work_extracted: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute information processing efficiency
        
        Relates information gain to thermodynamic cost.
        """
        # Information efficiency: bits processed per unit entropy
        info_efficiency = info_processed / (entropy_produced + self.config.min_entropy)
        
        # Thermodynamic efficiency: work out / entropy cost
        thermo_efficiency = work_extracted / (self.config.temperature * entropy_produced + self.config.min_entropy)
        thermo_efficiency = torch.clamp(thermo_efficiency, min=0, max=1)
        
        # Overall efficiency (geometric mean)
        overall_efficiency = torch.sqrt(info_efficiency * thermo_efficiency)
        
        # Compare to theoretical limits
        landauer_limit_efficiency = self.config.landauer_coefficient / (entropy_produced / info_processed + self.config.min_entropy)
        efficiency_ratio = overall_efficiency / landauer_limit_efficiency
        
        return {
            'information_efficiency': info_efficiency,
            'thermodynamic_efficiency': thermo_efficiency,
            'overall_efficiency': overall_efficiency,
            'landauer_limit_efficiency': landauer_limit_efficiency,
            'efficiency_ratio': efficiency_ratio
        }
    
    def analyze_mcts_thermodynamics(
        self,
        tree_state: Dict[str, torch.Tensor],
        action_probs: torch.Tensor,
        value_updates: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Complete thermodynamic analysis of MCTS state
        
        Args:
            tree_state: Current tree state (visits, values, etc.)
            action_probs: Action selection probabilities  
            value_updates: Value function updates
            
        Returns:
            Comprehensive thermodynamic analysis
        """
        # Extract relevant quantities
        visit_dist = F.normalize(tree_state['visit_counts'], p=1, dim=-1)
        
        # Compute entropy of current state
        state_entropy = self.entropy_calc.compute_shannon_entropy(visit_dist)
        
        # Information content of action selection
        action_info = -torch.log2(action_probs + self.config.min_entropy).mean()
        
        # Entropy production from value updates
        value_entropy = torch.abs(value_updates).mean() * self.config.beta
        
        # Check Landauer bounds
        bits_processed = action_info * len(action_probs)
        landauer_result = self.landauer_checker.check_erasure_bound(
            bits_processed.unsqueeze(0),
            value_entropy.unsqueeze(0)
        )
        
        # Compute extractable work
        work_result = self.work_extractor.compute_extractable_work(
            action_info.unsqueeze(0),
            value_entropy.unsqueeze(0)
        )
        
        # Overall efficiency
        efficiency_result = self.compute_information_efficiency(
            bits_processed.unsqueeze(0),
            value_entropy.unsqueeze(0),
            work_result['extractable_work']
        )
        
        # Update totals
        self.total_entropy_produced += value_entropy.item()
        self.total_work_extracted += work_result['extractable_work'].item()
        self.total_information_processed += bits_processed.item()
        
        # Compile results
        return {
            'state_entropy': state_entropy,
            'action_information': action_info,
            'entropy_production': value_entropy,
            'landauer_analysis': landauer_result,
            'work_analysis': work_result,
            'efficiency_analysis': efficiency_result,
            'cumulative_stats': {
                'total_entropy': self.total_entropy_produced,
                'total_work': self.total_work_extracted,
                'total_information': self.total_information_processed,
                'average_efficiency': self.total_work_extracted / (
                    self.config.temperature * self.total_entropy_produced + self.config.min_entropy
                )
            }
        }


class ThermodynamicMonitor:
    """
    Main thermodynamic monitoring system for QFT-MCTS
    
    Provides real-time monitoring of thermodynamic quantities
    and efficiency bounds.
    """
    
    def __init__(self, config: ThermodynamicsConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # Initialize analyzer
        self.analyzer = ThermodynamicEfficiencyAnalyzer(config, device)
        
        # Monitoring state
        self.step_count = 0
        self.measurements = deque(maxlen=config.history_window)
        
        # Alerts and warnings
        self.efficiency_warnings = []
        self.bound_violations = []
        
        logger.debug("ThermodynamicMonitor initialized")
    
    def monitor_step(
        self,
        tree_state: Dict[str, torch.Tensor],
        action_probs: torch.Tensor,
        value_updates: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Monitor thermodynamics for one MCTS step
        
        Returns:
            Thermodynamic measurements and any warnings
        """
        self.step_count += 1
        
        # Only measure at intervals
        if self.step_count % self.config.measurement_interval != 0:
            return {}
        
        # Perform analysis
        analysis = self.analyzer.analyze_mcts_thermodynamics(
            tree_state, action_probs, value_updates
        )
        
        # Check for warnings
        warnings = []
        
        # Efficiency warning
        overall_eff = analysis['efficiency_analysis']['overall_efficiency'].item()
        if overall_eff < self.config.efficiency_warning_threshold:
            warning = f"Low efficiency: {overall_eff:.3f} < {self.config.efficiency_warning_threshold}"
            warnings.append(warning)
            self.efficiency_warnings.append((self.step_count, warning))
        
        # Landauer bound violation
        if not analysis['landauer_analysis']['bound_satisfied'].item():
            warning = f"Landauer bound violated at step {self.step_count}"
            warnings.append(warning)
            self.bound_violations.append(self.step_count)
        
        # Excessive entropy production
        entropy_rate = analysis['entropy_production'].item()
        if entropy_rate > self.config.max_entropy_production_rate:
            warning = f"High entropy production rate: {entropy_rate:.3f}"
            warnings.append(warning)
        
        # Store measurement
        measurement = {
            'step': self.step_count,
            'state_entropy': analysis['state_entropy'].item(),
            'entropy_production': analysis['entropy_production'].item(),
            'work_extracted': analysis['work_analysis']['extractable_work'].item(),
            'overall_efficiency': overall_eff,
            'warnings': warnings
        }
        self.measurements.append(measurement)
        
        # Add warnings to result
        analysis['warnings'] = warnings
        analysis['step'] = self.step_count
        
        return analysis
    
    def get_efficiency_trends(self) -> Dict[str, List[float]]:
        """Get efficiency trends over time"""
        if not self.measurements:
            return {}
        
        steps = [m['step'] for m in self.measurements]
        entropy = [m['state_entropy'] for m in self.measurements]
        production = [m['entropy_production'] for m in self.measurements]
        work = [m['work_extracted'] for m in self.measurements]
        efficiency = [m['overall_efficiency'] for m in self.measurements]
        
        return {
            'steps': steps,
            'state_entropy': entropy,
            'entropy_production': production,
            'work_extracted': work,
            'efficiency': efficiency
        }
    
    def get_summary_statistics(self) -> Dict[str, float]:
        """Get summary statistics"""
        return {
            'total_steps': self.step_count,
            'measurements_taken': len(self.measurements),
            'average_efficiency': self.analyzer.total_work_extracted / (
                self.config.temperature * self.analyzer.total_entropy_produced + 1e-8
            ),
            'total_entropy_produced': self.analyzer.total_entropy_produced,
            'total_work_extracted': self.analyzer.total_work_extracted,
            'total_information_processed': self.analyzer.total_information_processed,
            'efficiency_warnings': len(self.efficiency_warnings),
            'landauer_violations': len(self.bound_violations),
            'violation_rate': len(self.bound_violations) / max(self.step_count, 1)
        }


# Factory function
def create_thermodynamic_monitor(
    device: Union[str, torch.device] = 'cuda',
    temperature: float = 1.0,
    **kwargs
) -> ThermodynamicMonitor:
    """
    Factory function to create thermodynamic monitor
    
    Args:
        device: Device for computation
        temperature: System temperature
        **kwargs: Override default config parameters
        
    Returns:
        Initialized ThermodynamicMonitor
    """
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Create config with overrides
    config_dict = {
        'temperature': temperature,
        'beta': 1.0 / temperature,
        'k_B': 1.0,
    }
    config_dict.update(kwargs)
    
    config = ThermodynamicsConfig(**config_dict)
    
    return ThermodynamicMonitor(config, device)