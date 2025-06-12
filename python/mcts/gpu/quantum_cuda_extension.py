"""Quantum CUDA Extension Module

This module provides the interface for quantum-enhanced CUDA kernels
with automatic compilation and fallback support.
"""

import torch
import logging
from pathlib import Path
from typing import Tuple, Optional

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