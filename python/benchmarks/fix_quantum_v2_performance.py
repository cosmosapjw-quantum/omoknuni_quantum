"""
Fix quantum v2 performance by removing bottlenecks
"""

import torch
import time
import numpy as np
from typing import Optional

from mcts.quantum.quantum_features_v2 import (
    QuantumMCTSV2, QuantumConfigV2, DiscreteTimeEvolution,
    PhaseDetector, OptimalParameters, MCTSPhase
)


class OptimizedQuantumMCTSV2(QuantumMCTSV2):
    """Optimized version of QuantumMCTSV2 with performance fixes"""
    
    def __init__(self, config: QuantumConfigV2):
        super().__init__(config)
        # Pre-compute phase configurations
        self.phase_configs = {
            MCTSPhase.QUANTUM: self.phase_detector.get_phase_config(MCTSPhase.QUANTUM),
            MCTSPhase.CRITICAL: self.phase_detector.get_phase_config(MCTSPhase.CRITICAL),
            MCTSPhase.CLASSICAL: self.phase_detector.get_phase_config(MCTSPhase.CLASSICAL)
        }
        # Cache current phase config
        self.current_phase_config = self.phase_configs[self.current_phase]
    
    def update_simulation_count(self, N: int):
        """Optimized update - only when phase changes"""
        if N == self.total_simulations:
            return
        
        old_phase = self.current_phase
        super().update_simulation_count(N)
        
        # Update cached config if phase changed
        if self.current_phase != old_phase:
            self.current_phase_config = self.phase_configs[self.current_phase]
    
    def apply_quantum_to_selection(
        self,
        q_values: torch.Tensor,
        visit_counts: torch.Tensor,
        priors: torch.Tensor,
        c_puct: Optional[float] = None,
        parent_visits: Optional[torch.Tensor] = None,
        simulation_count: Optional[int] = None
    ) -> torch.Tensor:
        """Optimized quantum selection without autocast overhead"""
        
        # Quick classical path for disabled quantum
        if not self.config.enable_quantum or self.config.quantum_level == 'classical':
            sqrt_parent = torch.sqrt(torch.log(parent_visits + 1))
            visit_factor = torch.sqrt(visit_counts + 1)
            exploration = c_puct * priors * sqrt_parent / visit_factor
            return q_values + exploration
        
        # Check batch size threshold
        batch_size = q_values.shape[0] if q_values.dim() > 1 else 1
        if batch_size < self.config.min_wave_size:
            sqrt_parent = torch.sqrt(torch.log(parent_visits + 1))
            visit_factor = torch.sqrt(visit_counts + 1)
            exploration = c_puct * priors * sqrt_parent / visit_factor
            return q_values + exploration
        
        # Use default values
        if c_puct is None:
            c_puct = self.config.c_puct or np.sqrt(2 * np.log(self.config.branching_factor))
        
        if simulation_count is None:
            simulation_count = self.total_simulations
        
        # Basic computations
        is_batched = q_values.dim() > 1
        
        if parent_visits is None:
            parent_visits = visit_counts.sum(dim=-1, keepdim=True) if is_batched else visit_counts.sum()
        
        sqrt_parent = torch.sqrt(torch.log(parent_visits + 1))
        visit_factor = torch.sqrt(visit_counts + 1)
        
        # Classical PUCT exploration
        exploration = c_puct * priors * sqrt_parent / visit_factor
        
        # Get cached phase config
        phase_config = self.current_phase_config
        
        # Compute quantum corrections WITHOUT autocast
        # Compute effective Planck constant
        hbar_eff = self.time_evolution.compute_hbar_eff(simulation_count, c_puct)
        
        # Scale by phase-specific quantum strength
        hbar_eff = hbar_eff * phase_config['quantum_strength']
        
        # Quantum uncertainty bonus
        quantum_bonus = hbar_eff / visit_factor
        
        # Apply prior trust modification based on phase
        prior_weight = phase_config['prior_trust']
        exploration = exploration * prior_weight
        
        # Skip interference for classical phase
        if self.current_phase == MCTSPhase.CLASSICAL:
            return q_values + exploration + quantum_bonus
        
        # Interference effects for low-visit nodes (only in quantum/critical phases)
        if self.config.quantum_level in ['tree_level', 'one_loop']:
            low_visit_mask = visit_counts < 10
            
            if low_visit_mask.any() and self.config.interference_method == 'phase_kick':
                # Simplified phase kick
                kick_prob = phase_config['interference_strength']
                random_phases = torch.rand_like(q_values) * 2 * np.pi
                phase_kicks = kick_prob * torch.sin(random_phases)
                quantum_bonus = quantum_bonus + torch.where(low_visit_mask, phase_kicks, 0.0)
        
        # One-loop corrections
        if self.config.quantum_level == 'one_loop':
            one_loop = self._compute_one_loop_v2(q_values, visit_counts, priors, hbar_eff)
            quantum_bonus = quantum_bonus + one_loop
        
        ucb_scores = q_values + exploration + quantum_bonus
        
        return ucb_scores


def benchmark_optimized_v2():
    """Benchmark the optimized version"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 64
    num_actions = 50
    num_calls = 1000
    
    # Create test data
    q_values = torch.randn(batch_size, num_actions, device=device)
    visit_counts = torch.randint(0, 100, (batch_size, num_actions), device=device)
    priors = torch.softmax(torch.randn(batch_size, num_actions, device=device), dim=-1)
    
    # Create original v2
    config = QuantumConfigV2(
        branching_factor=num_actions,
        device=device,
        enable_quantum=True
    )
    original_v2 = QuantumMCTSV2(config)
    
    # Create optimized v2
    optimized_v2 = OptimizedQuantumMCTSV2(config)
    
    # Warmup both
    for _ in range(10):
        _ = original_v2.apply_quantum_to_selection(
            q_values, visit_counts, priors,
            c_puct=1.414, simulation_count=1000
        )
        _ = optimized_v2.apply_quantum_to_selection(
            q_values, visit_counts, priors,
            c_puct=1.414, simulation_count=1000
        )
    
    # Benchmark original
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.perf_counter()
    for i in range(num_calls):
        _ = original_v2.apply_quantum_to_selection(
            q_values, visit_counts, priors,
            c_puct=1.414, simulation_count=1000
        )
    torch.cuda.synchronize() if device == 'cuda' else None
    original_time = time.perf_counter() - start
    
    # Benchmark optimized
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.perf_counter()
    for i in range(num_calls):
        _ = optimized_v2.apply_quantum_to_selection(
            q_values, visit_counts, priors,
            c_puct=1.414, simulation_count=1000
        )
    torch.cuda.synchronize() if device == 'cuda' else None
    optimized_time = time.perf_counter() - start
    
    print(f"Original v2: {original_time:.4f}s ({num_calls/original_time:.0f} calls/sec)")
    print(f"Optimized v2: {optimized_time:.4f}s ({num_calls/optimized_time:.0f} calls/sec)")
    print(f"Speedup: {original_time/optimized_time:.2f}x")
    
    # Verify outputs are similar
    out1 = original_v2.apply_quantum_to_selection(
        q_values, visit_counts, priors,
        c_puct=1.414, simulation_count=1000
    )
    out2 = optimized_v2.apply_quantum_to_selection(
        q_values, visit_counts, priors,
        c_puct=1.414, simulation_count=1000
    )
    
    max_diff = torch.max(torch.abs(out1 - out2)).item()
    print(f"\nMax difference: {max_diff:.6f}")


if __name__ == "__main__":
    benchmark_optimized_v2()