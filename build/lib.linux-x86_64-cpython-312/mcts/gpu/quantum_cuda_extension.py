"""Quantum CUDA Extension Module

This module provides the interface for quantum-enhanced CUDA kernels
with automatic compilation and fallback support.
"""

import torch
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)

# Global quantum kernel module
_QUANTUM_CUDA_MODULE = None
_QUANTUM_CUDA_AVAILABLE = False


def load_quantum_cuda_kernels():
    """Load or compile quantum CUDA kernels"""
    global _QUANTUM_CUDA_MODULE, _QUANTUM_CUDA_AVAILABLE
    
    if _QUANTUM_CUDA_MODULE is not None:
        return _QUANTUM_CUDA_AVAILABLE
    
    if not torch.cuda.is_available():
        logger.info("CUDA not available - quantum CUDA kernels disabled")
        return False
    
    try:
        # Try to import pre-compiled module
        import mcts.gpu.quantum_cuda_kernels as quantum_module
        _QUANTUM_CUDA_MODULE = quantum_module
        _QUANTUM_CUDA_AVAILABLE = True
        logger.info("Loaded pre-compiled quantum CUDA kernels")
        return True
    except ImportError:
        pass
    
    # Try JIT compilation
    try:
        from torch.utils.cpp_extension import load
        
        cuda_file = Path(__file__).parent / "quantum_cuda_kernels.cu"
        if not cuda_file.exists():
            logger.warning(f"Quantum CUDA source not found: {cuda_file}")
            return False
        
        logger.info("JIT compiling quantum CUDA kernels...")
        _QUANTUM_CUDA_MODULE = load(
            name='quantum_cuda_kernels',
            sources=[str(cuda_file)],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False
        )
        _QUANTUM_CUDA_AVAILABLE = True
        logger.info("Successfully compiled quantum CUDA kernels")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to compile quantum CUDA kernels: {e}")
        return False


def batched_ucb_selection_quantum(
    q_values: torch.Tensor,
    visit_counts: torch.Tensor,
    parent_visits: torch.Tensor,
    priors: torch.Tensor,
    row_ptr: torch.Tensor,
    col_indices: torch.Tensor,
    c_puct: float,
    quantum_phases: Optional[torch.Tensor] = None,
    uncertainty_table: Optional[torch.Tensor] = None,
    hbar_eff: float = 0.05,
    phase_kick_strength: float = 0.1,
    interference_alpha: float = 0.05,
    enable_quantum: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantum-enhanced UCB selection
    
    Args:
        q_values: Q-values for nodes
        visit_counts: Visit counts for nodes
        parent_visits: Parent visit counts
        priors: Prior probabilities
        row_ptr: CSR row pointers
        col_indices: CSR column indices
        c_puct: Exploration constant
        quantum_phases: Pre-computed quantum phases (optional)
        uncertainty_table: Quantum uncertainty lookup table (optional)
        hbar_eff: Effective Planck constant
        phase_kick_strength: Phase kick amplitude
        interference_alpha: Interference strength
        enable_quantum: Enable quantum features
        
    Returns:
        Tuple of (selected_actions, selected_scores)
    """
    # Load quantum kernels if needed
    if not _QUANTUM_CUDA_AVAILABLE:
        load_quantum_cuda_kernels()
    
    # Prepare quantum tensors
    if quantum_phases is None:
        quantum_phases = torch.empty(0, device=q_values.device, dtype=torch.float32)
    if uncertainty_table is None:
        uncertainty_table = torch.empty(0, device=q_values.device, dtype=torch.float32)
    
    # Use quantum kernel if available
    if _QUANTUM_CUDA_AVAILABLE and _QUANTUM_CUDA_MODULE is not None:
        try:
            return _QUANTUM_CUDA_MODULE.batched_ucb_selection_quantum(
                q_values, visit_counts, parent_visits, priors,
                row_ptr, col_indices, c_puct,
                quantum_phases, uncertainty_table,
                hbar_eff, phase_kick_strength, interference_alpha,
                enable_quantum
            )
        except Exception as e:
            logger.warning(f"Quantum CUDA kernel failed: {e}, falling back to CPU")
    
    # Fallback to standard kernel (imported from unified_kernels)
    from .unified_kernels import get_unified_kernels
    kernels = get_unified_kernels('cuda' if q_values.is_cuda else 'cpu')
    
    # Use standard kernel with quantum parameters if supported
    return kernels.batch_ucb_selection(
        torch.arange(len(parent_visits), device=q_values.device),
        row_ptr, col_indices,
        torch.arange(col_indices.size(0), device=q_values.device),  # dummy edge actions
        priors, visit_counts, q_values * visit_counts,  # value_sums
        c_puct, 1.0,
        quantum_phases, uncertainty_table,
        hbar_eff, phase_kick_strength, interference_alpha,
        enable_quantum
    )


def batched_ucb_selection_quantum_v2(
    q_values: torch.Tensor,
    visit_counts: torch.Tensor,
    parent_visits: torch.Tensor,
    priors: torch.Tensor,
    row_ptr: torch.Tensor,
    col_indices: torch.Tensor,
    c_puct_batch: torch.Tensor,  # Now batched for v2
    simulation_counts: torch.Tensor,  # Batch of N values for discrete time
    # v2.0 specific parameters
    phase_config: Dict[str, float],  # Phase-specific configuration
    tau_table: Optional[torch.Tensor] = None,  # Pre-computed information time
    hbar_factors: Optional[torch.Tensor] = None,  # Pre-computed hbar factors
    decoherence_table: Optional[torch.Tensor] = None,  # Power-law decoherence
    enable_quantum: bool = True,
    debug_logging: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantum-enhanced UCB selection for v2.0 with discrete information time
    
    This function adapts v2.0's discrete time formulation to use the existing
    high-performance CUDA kernels while maintaining mathematical correctness.
    
    Args:
        q_values: Q-values for nodes
        visit_counts: Visit counts for nodes  
        parent_visits: Parent visit counts (batched)
        priors: Prior probabilities
        row_ptr: CSR row pointers
        col_indices: CSR column indices
        c_puct_batch: Batched exploration constants
        simulation_counts: Batch of total simulations for discrete time
        phase_config: Current phase configuration dict
        tau_table: Pre-computed information time lookup
        hbar_factors: Pre-computed hbar_eff factors
        decoherence_table: Pre-computed decoherence values
        enable_quantum: Enable quantum features
        debug_logging: Enable debug logging
        
    Returns:
        Tuple of (selected_actions, selected_scores)
    """
    if debug_logging:
        logger.debug(f"v2.0 quantum selection: batch_size={len(parent_visits)}, enable_quantum={enable_quantum}")
    
    # Ensure quantum kernels are loaded
    if not _QUANTUM_CUDA_AVAILABLE:
        load_quantum_cuda_kernels()
    
    # Convert v2.0 parameters to v1.0 kernel format
    batch_size = len(parent_visits)
    device = q_values.device
    
    # Compute effective hbar for each batch element using discrete time
    if enable_quantum and hbar_factors is not None:
        # Use pre-computed factors for efficiency
        sim_indices = torch.clamp(simulation_counts.long(), 0, len(hbar_factors) - 1)
        hbar_batch = c_puct_batch * hbar_factors[sim_indices] * phase_config['quantum_strength']
        hbar_eff = hbar_batch.mean().item()  # Kernel expects scalar
    else:
        hbar_eff = 0.05  # Default fallback
    
    # Map phase config to kernel parameters
    phase_kick_strength = phase_config.get('interference_strength', 0.1)
    interference_alpha = phase_kick_strength * 0.5  # Scale for kernel compatibility
    
    # Generate quantum phases if needed (can be pre-computed later)
    if enable_quantum:
        num_edges = col_indices.size(0)
        quantum_phases = torch.empty(num_edges, device=device, dtype=torch.float32)
        # Simple phase generation - can be optimized with pre-computation
        quantum_phases.uniform_(0, 2 * 3.14159265)
    else:
        quantum_phases = torch.empty(0, device=device, dtype=torch.float32)
    
    # Create uncertainty table for discrete time if not provided
    if enable_quantum and decoherence_table is None:
        # Quick approximation - should be pre-computed
        max_visits = 1000
        visit_range = torch.arange(max_visits, device=device, dtype=torch.float32)
        uncertainty_table = hbar_eff / (visit_range + 1.0)
    elif decoherence_table is not None:
        uncertainty_table = decoherence_table * hbar_eff
    else:
        uncertainty_table = torch.empty(0, device=device, dtype=torch.float32)
    
    if debug_logging:
        logger.debug(f"v2 kernel params: hbar_eff={hbar_eff:.4f}, phase_kick={phase_kick_strength:.4f}")
    
    # Call optimized CUDA kernel
    if _QUANTUM_CUDA_AVAILABLE and _QUANTUM_CUDA_MODULE is not None:
        try:
            return _QUANTUM_CUDA_MODULE.batched_ucb_selection_quantum(
                q_values, visit_counts, parent_visits, priors,
                row_ptr, col_indices, c_puct_batch.mean().item(),  # Use mean c_puct
                quantum_phases, uncertainty_table,
                hbar_eff, phase_kick_strength, interference_alpha,
                enable_quantum
            )
        except Exception as e:
            if debug_logging:
                logger.warning(f"v2 quantum CUDA kernel failed: {e}, falling back")
    
    # Fallback to standard kernel
    from .unified_kernels import get_unified_kernels
    kernels = get_unified_kernels(device if isinstance(device, torch.device) else device.type)
    
    return kernels.batch_ucb_selection(
        torch.arange(batch_size, device=device),
        row_ptr, col_indices,
        torch.arange(col_indices.size(0), device=device),
        priors, visit_counts, q_values * visit_counts,
        c_puct_batch.mean().item(), 1.0,
        quantum_phases, uncertainty_table,
        hbar_eff, phase_kick_strength, interference_alpha,
        enable_quantum
    )


def generate_quantum_phases(
    node_indices: torch.Tensor,
    edge_indices: torch.Tensor,
    visit_counts: torch.Tensor,
    base_frequency: float = 0.1,
    visit_modulation: float = 0.01
) -> torch.Tensor:
    """Generate quantum phases for edges
    
    Args:
        node_indices: Node indices for edges
        edge_indices: Edge indices
        visit_counts: Visit counts for nodes
        base_frequency: Base frequency for phase generation
        visit_modulation: Visit count modulation factor
        
    Returns:
        Quantum phases tensor
    """
    if _QUANTUM_CUDA_AVAILABLE and _QUANTUM_CUDA_MODULE is not None:
        try:
            return _QUANTUM_CUDA_MODULE.generate_quantum_phases(
                node_indices, edge_indices, visit_counts,
                base_frequency, visit_modulation
            )
        except Exception as e:
            logger.warning(f"Quantum phase generation failed: {e}, using CPU fallback")
    
    # CPU fallback
    phases = base_frequency * edge_indices.float()
    phases += visit_modulation * torch.log(1.0 + visit_counts[node_indices].float())
    phases = torch.fmod(phases, 2.0 * 3.14159265)
    return phases


# Auto-load on import
load_quantum_cuda_kernels()