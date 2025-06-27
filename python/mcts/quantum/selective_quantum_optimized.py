"""
Selective Quantum MCTS - Aggressively Optimized Implementation
===========================================================

This module implements quantum features ONLY where they provide clear exploration/exploitation benefits:
- Quantum exploration boost for low visit counts (uncertainty principle)
- Critical point detection for quantum/classical switching  
- Minimal quantum corrections without overhead
- Extended CUDA kernels for maximum performance
- Full vectorization/tensorization for parallel processing

Features OMITTED (overhead without benefit):
- Complex Hamiltonian dynamics
- Full path integral computation
- Lindblad evolution
- Extensive state tracking
- Complex wave processing

Target: < 1.5x overhead while providing 10-15% exploration improvement
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Extended CUDA kernels for maximum performance - v5.0 Formula
CUDA_QUANTUM_KERNEL = """
__global__ void selective_quantum_kernel_v5(
    float* q_values,           // [batch_size] - Q_k (mean action-values)
    float* visit_counts,       // [batch_size] - N_k (visit counts)
    float* priors,             // [batch_size] - p_k (NN priors)
    float* output,             // [batch_size]
    float kappa,               // exploration strength (stiffness)
    float beta,                // value weight (inverse temperature)
    float hbar_0,              // base Planck scale
    float alpha,               // annealing exponent
    float parent_visits,       // N_tot
    int batch_size,
    int simulation_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    float N_k = visit_counts[idx];
    float p_k = priors[idx];
    float Q_k = q_values[idx];
    float N_tot = parent_visits;
    
    // Ensure N_k is positive for stability
    float safe_N_k = fmaxf(N_k, 1.0f);
    
    // v5.0 Quantum-Augmented Score Formula:
    // Score(k) = κ p_k (N_k/N_tot) + β Q_k + (4 ℏ_eff(N_tot))/(3 N_k)
    
    // Calculate ℏ_eff(N_tot) = ℏ_0 (1 + N_tot)^(-α/2)
    float hbar_eff = hbar_0 * powf(1.0f + N_tot, -alpha * 0.5f);
    
    // Apply selective quantum: only in early search phases and low visit counts
    float quantum_bonus = 0.0f;
    if (N_k < 10.0f && simulation_count < 5000) {  // Selective application
        quantum_bonus = (4.0f * hbar_eff) / (3.0f * safe_N_k);
    }
    
    // Full v5.0 formula components
    float exploration_term = kappa * p_k * (safe_N_k / N_tot);  // κ p_k (N_k/N_tot)
    float exploitation_term = beta * Q_k;                        // β Q_k
    
    output[idx] = exploration_term + exploitation_term + quantum_bonus;
}

__global__ void batch_critical_detection(
    int* simulation_counts,    // [batch_size]
    float* quantum_factors,    // [batch_size] output
    int batch_size,
    int critical_point_1,
    int critical_point_2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    int sim_count = simulation_counts[idx];
    
    // Quantum factor based on regime
    if (sim_count < critical_point_1) {
        quantum_factors[idx] = 1.0f;      // Full quantum
    } else if (sim_count < critical_point_2) {
        quantum_factors[idx] = 0.5f;      // Critical transition
    } else {
        quantum_factors[idx] = 0.1f;      // Minimal quantum (avoid pure classical)
    }
}
"""

@dataclass
class SelectiveQuantumConfig:
    """Minimal configuration for selective quantum features - v5.0 Formula"""
    
    # Core v5.0 parameters (from docs/v5.0/new_quantum_mcts.md)
    branching_factor: int = 30
    enable_quantum: bool = True
    
    # v5.0 Physical parameters
    kappa: Optional[float] = None          # κ - exploration strength (stiffness)
    beta: float = 1.0                      # β - value weight (inverse temperature)
    hbar_0: float = 0.1                    # ℏ_0 - base Planck scale  
    alpha: float = 0.5                     # α - annealing exponent for ℏ_eff
    
    # Performance settings
    device: str = 'cuda'
    use_mixed_precision: bool = True
    enable_cuda_kernels: bool = True
    max_batch_size: int = 4096  # Large batches for GPU efficiency
    
    # Selective quantum thresholds (optimization)
    exploration_visit_threshold: int = 10      # Apply quantum only below this
    quantum_phase_threshold: int = 5000        # Switch to classical above this
    critical_transition_threshold: int = 1000  # Critical region start
    
    # State management
    enable_coherent_state_management: bool = True
    enable_causality_preservation: bool = True
    
    def __post_init__(self):
        if self.kappa is None:
            # κ should scale with branching factor for exploration strength
            self.kappa = math.sqrt(2 * math.log(self.branching_factor))
    
    def hbar_eff(self, N_tot: float) -> float:
        """Calculate ℏ_eff(N_tot) = ℏ_0 (1 + N_tot)^(-α/2)"""
        return self.hbar_0 * ((1.0 + N_tot) ** (-self.alpha * 0.5))


class CudaKernelManager:
    """Manages custom CUDA kernels for quantum operations"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.kernels_compiled = False
        self.cuda_module = None
        self._compile_kernels()
    
    def _compile_kernels(self):
        """Compile custom CUDA kernels"""
        if not self.device.type == 'cuda':
            logger.debug("CUDA kernels not available on CPU device")
            return
            
        try:
            # Try to import existing CUDA module first
            try:
                from mcts.gpu import quantum_v5_cuda_kernels
                self.cuda_module = quantum_v5_cuda_kernels
                self.kernels_compiled = True
                logger.info("Loaded pre-compiled v5.0 CUDA kernels")
                return
            except ImportError:
                pass
            
            # Try to compile v5.0 kernels inline
            from torch.utils.cpp_extension import load_inline
            import tempfile
            import os
            
            build_dir = tempfile.mkdtemp(prefix='quantum_v5_cuda_')
            
            # Read v5.0 CUDA source
            cuda_source_path = os.path.join(os.path.dirname(__file__), '..', 'gpu', 'quantum_v5_cuda_kernels.cu')
            if os.path.exists(cuda_source_path):
                with open(cuda_source_path, 'r') as f:
                    cuda_source = f.read()
                
                # Compile v5.0 kernels
                self.cuda_module = load_inline(
                    name='quantum_v5_kernels_inline',
                    cuda_sources=[cuda_source],
                    extra_cuda_cflags=['-O3', '--use_fast_math'],
                    verbose=False,
                    build_directory=build_dir,
                    functions=['selective_quantum_v5', 'batch_quantum_v5', 'quantum_regime_detection']
                )
                
                self.kernels_compiled = True
                logger.info("Compiled v5.0 CUDA kernels successfully")
            else:
                logger.warning("v5.0 CUDA source not found, using PyTorch fallback")
                self.kernels_compiled = False
                
        except Exception as e:
            logger.warning(f"Failed to compile v5.0 CUDA kernels: {e}")
            logger.warning("Using PyTorch fallback implementations")
            self.kernels_compiled = False
    
    def selective_quantum_kernel_call_v5(
        self, 
        q_values: torch.Tensor,      # Q_k - mean action-values
        visit_counts: torch.Tensor,  # N_k - visit counts
        priors: torch.Tensor,        # p_k - NN priors
        kappa: float,                # κ - exploration strength
        beta: float,                 # β - value weight
        hbar_0: float,               # ℏ_0 - base Planck scale
        alpha: float,                # α - annealing exponent
        parent_visits: float,        # N_tot
        simulation_count: int
    ) -> torch.Tensor:
        """Call selective quantum CUDA kernel v5.0"""
        
        if self.kernels_compiled and self.cuda_module is not None:
            try:
                # Use compiled CUDA kernel
                return self.cuda_module.selective_quantum_v5(
                    q_values, visit_counts, priors,
                    kappa, beta, hbar_0, alpha, parent_visits, simulation_count
                )
            except Exception as e:
                logger.warning(f"CUDA kernel call failed: {e}, falling back to PyTorch")
        
        # PyTorch fallback implementation
        # v5.0 Formula: Score(k) = κ p_k (N_k/N_tot) + β Q_k + (4 ℏ_eff(N_tot))/(3 N_k)
        
        N_k = torch.clamp(visit_counts, min=1.0)  # Ensure positive for stability
        N_tot = parent_visits
        
        # Calculate ℏ_eff(N_tot) = ℏ_0 (1 + N_tot)^(-α/2)
        hbar_eff = hbar_0 * ((1.0 + N_tot) ** (-alpha * 0.5))
        
        # v5.0 Formula components
        exploration_term = kappa * priors * (N_k / N_tot)  # κ p_k (N_k/N_tot)
        exploitation_term = beta * q_values                 # β Q_k
        
        # Selective quantum bonus: only where beneficial for exploration
        quantum_bonus = torch.zeros_like(q_values)
        quantum_mask = (visit_counts < 10.0) & (simulation_count < 5000)
        
        if torch.any(quantum_mask):
            # Apply v5.0 quantum bonus: (4 ℏ_eff(N_tot))/(3 N_k)
            masked_visits = N_k[quantum_mask]
            quantum_bonus[quantum_mask] = (4.0 * hbar_eff) / (3.0 * masked_visits)
        
        return exploration_term + exploitation_term + quantum_bonus


class SelectiveQuantumMCTS:
    """
    Aggressively optimized quantum MCTS with selective features
    
    Key optimizations:
    - Custom CUDA kernels for quantum corrections
    - Selective quantum application (only where beneficial)
    - Full vectorization for batch processing
    - Minimal state tracking overhead
    - Critical point detection for adaptive behavior
    """
    
    def __init__(self, config: SelectiveQuantumConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize CUDA kernel manager
        self.cuda_manager = CudaKernelManager(self.device)
        
        # Initialize coherent state management
        if config.enable_coherent_state_management:
            try:
                from .quantum_state_manager import create_quantum_state_manager
                self.state_manager = create_quantum_state_manager(
                    critical_point_1=config.critical_transition_threshold,
                    critical_point_2=config.quantum_phase_threshold,
                    enable_causality_preservation=config.enable_causality_preservation,
                    thread_safe=True
                )
                # Register this component with the state manager
                self.state_manager.register_component("selective_quantum_mcts", self)
                self.coherent_state_enabled = True
                logger.info("Coherent quantum state management enabled")
            except ImportError:
                logger.warning("Quantum state manager not available, using basic state tracking")
                self.state_manager = None
                self.coherent_state_enabled = False
        else:
            self.state_manager = None
            self.coherent_state_enabled = False
        
        # Pre-compute critical points for adaptive behavior
        self._compute_critical_points()
        
        # Pre-allocate tensors for maximum batch efficiency
        self._preallocate_tensors()
        
        # Statistics (minimal overhead)
        self.stats = {
            'total_calls': 0,
            'quantum_applications': 0,
            'classical_applications': 0,
            'cuda_kernel_calls': 0,
            'batch_sizes': []
        }
        
        logger.info(f"SelectiveQuantumMCTS v5.0 initialized on {self.device}")
        logger.info(f"  κ={self.config.kappa:.3f}, β={self.config.beta:.3f}, ℏ₀={self.config.hbar_0:.3f}, α={self.config.alpha:.3f}")
        logger.info(f"  Coherent state management: {self.coherent_state_enabled}")
    
    def _compute_critical_points(self):
        """Pre-compute critical transition points"""
        self.critical_point_1 = self.config.critical_transition_threshold
        self.critical_point_2 = self.config.quantum_phase_threshold
        
        # Pre-compute quantum factors for different regimes
        self.quantum_factors = {
            'quantum': 1.0,
            'critical': 0.5, 
            'classical': 0.1
        }
    
    def _preallocate_tensors(self):
        """Pre-allocate tensors for common batch sizes"""
        self.tensor_cache = {}
        common_sizes = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        
        for size in common_sizes:
            self.tensor_cache[size] = {
                'temp1': torch.zeros(size, device=self.device),
                'temp2': torch.zeros(size, device=self.device),
                'mask': torch.zeros(size, dtype=torch.bool, device=self.device)
            }
    
    def apply_selective_quantum(
        self,
        q_values: torch.Tensor,
        visit_counts: torch.Tensor,
        priors: torch.Tensor,
        parent_visits: Optional[Union[int, float]] = None,
        simulation_count: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply selective quantum corrections with v5.0 formula
        
        v5.0 Formula: Score(k) = κ p_k (N_k/N_tot) + β Q_k + (4 ℏ_eff(N_tot))/(3 N_k)
        
        Only applies quantum where it improves exploration/exploitation:
        - Low visit count nodes (exploration boost)
        - Early simulation phases (quantum regime)
        """
        self.stats['total_calls'] += 1
        
        # Flatten for vectorized processing if needed
        original_shape = q_values.shape
        if q_values.dim() > 1:
            q_values = q_values.view(-1)
            visit_counts = visit_counts.view(-1)
            priors = priors.view(-1)
        
        batch_size = q_values.shape[0]
        self.stats['batch_sizes'].append(batch_size)
        
        # v5.0 parameters
        parent_vis = parent_visits or visit_counts.sum().item()
        sim_count = simulation_count or 1000
        
        # Update coherent state management
        if self.coherent_state_enabled and self.state_manager:
            self.state_manager.update_simulation_count(sim_count)
            # Get adaptive parameters from state manager
            adaptive_params = self.state_manager.get_adaptive_parameters()
            quantum_factor = adaptive_params['quantum_factor']
        else:
            # Fallback to local computation
            quantum_factor = self._get_quantum_factor(sim_count)
        
        if not self.config.enable_quantum or quantum_factor < 0.05:
            # Pure classical path using v5.0 formula (without quantum bonus)
            self.stats['classical_applications'] += 1
            result = self._classical_v5_vectorized(q_values, visit_counts, priors, parent_vis)
        else:
            # Selective quantum path
            self.stats['quantum_applications'] += 1
            
            if self.cuda_manager.kernels_compiled and batch_size >= 32:
                # Use v5.0 CUDA kernel for large batches
                self.stats['cuda_kernel_calls'] += 1
                result = self.cuda_manager.selective_quantum_kernel_call_v5(
                    q_values, visit_counts, priors,
                    self.config.kappa, self.config.beta,
                    self.config.hbar_0, self.config.alpha,
                    parent_vis, sim_count
                )
            else:
                # Fallback to optimized PyTorch v5.0
                result = self._selective_quantum_v5_pytorch(
                    q_values, visit_counts, priors, parent_vis, sim_count
                )
        
        # Restore original shape
        if len(original_shape) > 1:
            result = result.view(original_shape)
        
        return result
    
    def _get_quantum_factor(self, simulation_count: int) -> float:
        """Get quantum factor based on simulation regime"""
        if simulation_count < self.critical_point_1:
            return self.quantum_factors['quantum']
        elif simulation_count < self.critical_point_2:
            return self.quantum_factors['critical']
        else:
            return self.quantum_factors['classical']
    
    def _classical_v5_vectorized(
        self,
        q_values: torch.Tensor,      # Q_k
        visit_counts: torch.Tensor,  # N_k
        priors: torch.Tensor,        # p_k
        parent_visits: float         # N_tot
    ) -> torch.Tensor:
        """v5.0 classical formula: κ p_k (N_k/N_tot) + β Q_k (no quantum bonus)"""
        safe_visits = torch.clamp(visit_counts, min=1.0)
        
        # v5.0 Formula components
        exploration_term = self.config.kappa * priors * (safe_visits / parent_visits)
        exploitation_term = self.config.beta * q_values
        
        return exploration_term + exploitation_term
    
    def _selective_quantum_v5_pytorch(
        self,
        q_values: torch.Tensor,      # Q_k
        visit_counts: torch.Tensor,  # N_k 
        priors: torch.Tensor,        # p_k
        parent_visits: float,        # N_tot
        simulation_count: int
    ) -> torch.Tensor:
        """v5.0 selective quantum implementation using optimized PyTorch"""
        
        safe_visits = torch.clamp(visit_counts, min=1.0)
        
        # v5.0 Formula: Score(k) = κ p_k (N_k/N_tot) + β Q_k + (4 ℏ_eff(N_tot))/(3 N_k)
        
        # Component 1: Exploration term - κ p_k (N_k/N_tot)
        exploration_term = self.config.kappa * priors * (safe_visits / parent_visits)
        
        # Component 2: Exploitation term - β Q_k
        exploitation_term = self.config.beta * q_values
        
        # Component 3: Selective quantum bonus - (4 ℏ_eff(N_tot))/(3 N_k)
        quantum_bonus = torch.zeros_like(q_values)
        
        # Apply quantum corrections selectively (only where beneficial)
        exploration_mask = (visit_counts < self.config.exploration_visit_threshold) & \
                          (simulation_count < self.config.quantum_phase_threshold)
        
        if torch.any(exploration_mask):
            # Calculate ℏ_eff(N_tot) = ℏ_0 (1 + N_tot)^(-α/2)
            hbar_eff = self.config.hbar_eff(parent_visits)
            
            # Apply v5.0 quantum bonus: (4 ℏ_eff(N_tot))/(3 N_k)
            masked_visits = safe_visits[exploration_mask]
            quantum_bonus[exploration_mask] = (4.0 * hbar_eff) / (3.0 * masked_visits)
        
        return exploration_term + exploitation_term + quantum_bonus
    
    def batch_process(
        self,
        q_values_batch: torch.Tensor,     # [batch_size, num_actions]
        visit_counts_batch: torch.Tensor, # [batch_size, num_actions]  
        priors_batch: torch.Tensor,       # [batch_size, num_actions]
        simulation_counts: torch.Tensor,  # [batch_size]
        parent_visits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        v5.0 Highly optimized batch processing for maximum throughput
        
        Uses v5.0 formula: Score(k) = κ p_k (N_k/N_tot) + β Q_k + (4 ℏ_eff(N_tot))/(3 N_k)
        """
        batch_size, num_actions = q_values_batch.shape
        
        if parent_visits is None:
            parent_visits = visit_counts_batch.sum(dim=-1)
        
        # v5.0 vectorized processing
        safe_visits = torch.clamp(visit_counts_batch, min=1.0)
        
        # Component 1: Exploration term - κ p_k (N_k/N_tot)
        parent_visits_expanded = parent_visits.unsqueeze(-1)  # [batch_size, 1]
        exploration_term = self.config.kappa * priors_batch * (safe_visits / parent_visits_expanded)
        
        # Component 2: Exploitation term - β Q_k
        exploitation_term = self.config.beta * q_values_batch
        
        # Component 3: Selective quantum bonus - (4 ℏ_eff(N_tot))/(3 N_k)
        quantum_bonus = torch.zeros_like(q_values_batch)
        
        # Determine which elements need quantum corrections
        quantum_active_mask = simulation_counts < self.config.quantum_phase_threshold
        
        if torch.any(quantum_active_mask):
            # Element-wise exploration mask
            exploration_element_mask = (visit_counts_batch < self.config.exploration_visit_threshold) & \
                                     (simulation_counts.unsqueeze(-1) < self.config.quantum_phase_threshold)
            
            if torch.any(exploration_element_mask):
                # Calculate ℏ_eff for each batch element: ℏ_0 (1 + N_tot)^(-α/2)
                hbar_eff_batch = self.config.hbar_0 * ((1.0 + parent_visits) ** (-self.config.alpha * 0.5))
                hbar_eff_expanded = hbar_eff_batch.unsqueeze(-1)  # [batch_size, 1]
                
                # Apply v5.0 quantum bonus: (4 ℏ_eff(N_tot))/(3 N_k)
                quantum_bonus_full = (4.0 * hbar_eff_expanded) / (3.0 * safe_visits)
                
                # Apply only where beneficial (selective mask)
                quantum_bonus[exploration_element_mask] = quantum_bonus_full[exploration_element_mask]
        
        return exploration_term + exploitation_term + quantum_bonus
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_calls = self.stats['total_calls']
        if total_calls > 0:
            quantum_ratio = self.stats['quantum_applications'] / total_calls
            avg_batch_size = sum(self.stats['batch_sizes']) / len(self.stats['batch_sizes'])
        else:
            quantum_ratio = 0.0
            avg_batch_size = 0.0
        
        return {
            'total_calls': total_calls,
            'quantum_ratio': quantum_ratio,
            'cuda_kernel_calls': self.stats['cuda_kernel_calls'],
            'average_batch_size': avg_batch_size,
            'cuda_kernels_available': self.cuda_manager.kernels_compiled
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.stats = {k: 0 if isinstance(v, int) else [] if isinstance(v, list) else v 
                     for k, v in self.stats.items()}


# Factory functions
def create_selective_quantum_mcts(
    branching_factor: int = 30,
    device: str = 'cuda',
    hbar_0: float = 0.1,
    **kwargs
) -> SelectiveQuantumMCTS:
    """Create selective quantum MCTS with performance optimization"""
    config = SelectiveQuantumConfig(
        branching_factor=branching_factor,
        device=device,
        hbar_0=hbar_0,
        **kwargs
    )
    return SelectiveQuantumMCTS(config)


def create_ultra_performance_quantum_mcts(
    branching_factor: int = 30,
    device: str = 'cuda'
) -> SelectiveQuantumMCTS:
    """Create quantum MCTS optimized for maximum performance"""
    config = SelectiveQuantumConfig(
        branching_factor=branching_factor,
        device=device,
        hbar_0=0.08,  # Reduced for performance
        enable_cuda_kernels=True,
        max_batch_size=8192,    # Large batches
        exploration_visit_threshold=8,  # More selective
        quantum_phase_threshold=3000,   # Faster transition to classical
    )
    return SelectiveQuantumMCTS(config)


# Export main classes
__all__ = [
    'SelectiveQuantumMCTS',
    'SelectiveQuantumConfig', 
    'create_selective_quantum_mcts',
    'create_ultra_performance_quantum_mcts'
]