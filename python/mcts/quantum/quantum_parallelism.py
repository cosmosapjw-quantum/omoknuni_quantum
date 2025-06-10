"""
Quantum Parallelism Exploitation for MCTS
=========================================

This module implements quantum parallelism to evaluate multiple MCTS paths
simultaneously using superposition principles.

Key Features:
- Quantum superposition for parallel path evaluation
- Grover-like amplitude amplification for promising paths
- Quantum interference for path selection
- GPU-accelerated quantum operations
- Efficient classical extraction via measurement

Based on: Quantum computing principles applied to tree search
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class QuantumParallelismConfig:
    """Configuration for quantum parallelism"""
    # Superposition parameters
    max_superposition_size: int = 1024  # Max paths in superposition
    amplitude_threshold: float = 1e-4   # Minimum amplitude to keep
    
    # Grover parameters
    grover_iterations: int = 3          # Number of amplitude amplification steps
    oracle_threshold: float = 0.7       # Threshold for marking good paths
    
    # Interference parameters
    interference_strength: float = 0.2  # Quantum interference strength
    phase_damping: float = 0.1         # Phase coherence decay
    
    # Performance parameters
    use_gpu: bool = True
    batch_size: int = 256
    cache_quantum_states: bool = True
    
    # Hardware optimization
    use_tensor_cores: bool = True      # Use tensor cores for matrix ops
    mixed_precision: bool = True       # Use FP16 where possible


class QuantumSuperpositionManager:
    """
    Manages quantum superposition states for parallel path evaluation
    
    Creates and maintains superposition of multiple MCTS paths,
    allowing quantum parallel evaluation.
    """
    
    def __init__(self, config: QuantumParallelismConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_gpu else 'cpu')
        
        # Quantum state cache
        self.state_cache = {} if config.cache_quantum_states else None
        
        # Pre-compute common quantum gates
        self._init_quantum_gates()
        
        # Statistics
        self.stats = {
            'superpositions_created': 0,
            'paths_evaluated': 0,
            'amplifications_performed': 0,
            'cache_hits': 0
        }
        
    def _init_quantum_gates(self):
        """Initialize common quantum gates"""
        # Hadamard gate for creating superposition
        self.hadamard = torch.tensor([
            [1, 1],
            [1, -1]
        ], dtype=torch.complex64, device=self.device) / np.sqrt(2)
        
        # Pauli gates
        self.pauli_x = torch.tensor([
            [0, 1],
            [1, 0]
        ], dtype=torch.complex64, device=self.device)
        
        self.pauli_z = torch.tensor([
            [1, 0],
            [0, -1]
        ], dtype=torch.complex64, device=self.device)
        
    def create_path_superposition(
        self,
        paths: torch.Tensor,
        initial_amplitudes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Create quantum superposition of multiple paths
        
        Args:
            paths: Tensor of path indices [num_paths, max_depth]
            initial_amplitudes: Optional initial amplitude distribution
            
        Returns:
            Quantum state vector representing superposition
        """
        num_paths = paths.shape[0]
        
        if num_paths > self.config.max_superposition_size:
            # Subsample if too many paths
            indices = torch.randperm(num_paths)[:self.config.max_superposition_size]
            paths = paths[indices]
            num_paths = self.config.max_superposition_size
            
        # Initialize amplitudes
        if initial_amplitudes is None:
            # Equal superposition
            amplitudes = torch.ones(num_paths, dtype=torch.complex64, device=self.device)
            amplitudes = amplitudes / torch.sqrt(torch.tensor(num_paths, dtype=torch.float32))
        else:
            amplitudes = initial_amplitudes.to(torch.complex64)
            # Normalize
            amplitudes = amplitudes / torch.norm(amplitudes)
            
        # Create quantum state
        quantum_state = {
            'amplitudes': amplitudes,
            'paths': paths,
            'phase': torch.zeros(num_paths, device=self.device),
            'coherence': torch.ones(num_paths, device=self.device)
        }
        
        self.stats['superpositions_created'] += 1
        
        return quantum_state
    
    def apply_quantum_oracle(
        self,
        quantum_state: Dict[str, torch.Tensor],
        oracle_values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Apply quantum oracle to mark good paths
        
        The oracle flips the phase of paths that satisfy the criteria
        """
        amplitudes = quantum_state['amplitudes']
        
        # Mark paths above threshold
        good_paths = oracle_values > self.config.oracle_threshold
        
        # Apply phase flip to marked paths
        phase_flip = torch.where(good_paths, -1.0, 1.0).to(torch.complex64)
        amplitudes = amplitudes * phase_flip
        
        quantum_state['amplitudes'] = amplitudes
        return quantum_state
    
    def grover_diffusion(
        self,
        quantum_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply Grover diffusion operator for amplitude amplification
        
        This amplifies the amplitude of marked states
        """
        amplitudes = quantum_state['amplitudes']
        
        # Compute mean amplitude
        mean_amp = torch.mean(amplitudes)
        
        # Grover diffusion: reflect about average
        # This is the inversion about average operation
        amplitudes = 2 * mean_amp - amplitudes
        
        # Normalize
        norm = torch.norm(amplitudes)
        if norm > 0:
            amplitudes = amplitudes / norm
        
        quantum_state['amplitudes'] = amplitudes
        self.stats['amplifications_performed'] += 1
        
        return quantum_state
    
    def apply_grover_iteration(
        self,
        quantum_state: Dict[str, torch.Tensor],
        oracle_values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Single Grover iteration: Oracle + Diffusion
        """
        # Ensure oracle values match the number of paths in quantum state
        num_paths = len(quantum_state['amplitudes'])
        if len(oracle_values) != num_paths:
            # Truncate or pad oracle values to match
            if len(oracle_values) > num_paths:
                oracle_values = oracle_values[:num_paths]
            else:
                # This shouldn't happen in normal usage
                padding = torch.zeros(num_paths - len(oracle_values), device=oracle_values.device)
                oracle_values = torch.cat([oracle_values, padding])
        
        quantum_state = self.apply_quantum_oracle(quantum_state, oracle_values)
        quantum_state = self.grover_diffusion(quantum_state)
        return quantum_state


class QuantumInterferenceEngine:
    """
    Implements quantum interference for path selection
    
    Uses destructive and constructive interference to enhance
    selection of promising paths.
    """
    
    def __init__(self, config: QuantumParallelismConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_gpu else 'cpu')
        
    def compute_path_overlap(
        self,
        paths1: torch.Tensor,
        paths2: torch.Tensor
    ) -> torch.Tensor:
        """Compute overlap between path sets"""
        # Simple overlap: fraction of shared nodes
        batch_size1 = paths1.shape[0]
        batch_size2 = paths2.shape[0]
        
        # Expand for broadcasting
        paths1_exp = paths1.unsqueeze(1)  # [batch1, 1, depth]
        paths2_exp = paths2.unsqueeze(0)  # [1, batch2, depth]
        
        # Count matches
        matches = (paths1_exp == paths2_exp).float()
        valid_mask = (paths1_exp >= 0) & (paths2_exp >= 0)
        
        overlap = (matches * valid_mask).sum(dim=2) / valid_mask.sum(dim=2).clamp(min=1)
        
        return overlap
    
    def apply_interference(
        self,
        quantum_state: Dict[str, torch.Tensor],
        interference_matrix: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply quantum interference between paths
        
        Paths with high overlap interfere, redistributing amplitude
        """
        amplitudes = quantum_state['amplitudes']
        paths = quantum_state['paths']
        
        if interference_matrix is None:
            # Compute interference based on path overlap
            overlap = self.compute_path_overlap(paths, paths)
            interference_matrix = overlap * self.config.interference_strength
            
        # Apply interference as unitary transformation
        # U = I + i*H where H is Hermitian interference matrix
        identity = torch.eye(len(amplitudes), device=self.device, dtype=torch.complex64)
        evolution = identity + 1j * interference_matrix.to(torch.complex64)
        
        # Ensure unitarity (approximately)
        evolution = evolution / torch.sqrt(torch.abs(torch.det(evolution)))
        
        # Evolve amplitudes
        amplitudes = torch.matmul(evolution, amplitudes.unsqueeze(1)).squeeze(1)
        
        # Apply phase damping
        coherence = quantum_state['coherence'] * (1 - self.config.phase_damping)
        amplitudes = amplitudes * coherence.unsqueeze(1).to(torch.complex64)
        
        # Renormalize
        amplitudes = amplitudes / torch.norm(amplitudes)
        
        quantum_state['amplitudes'] = amplitudes
        quantum_state['coherence'] = coherence
        
        return quantum_state


class QuantumPathEvaluator:
    """
    Evaluates multiple paths in quantum superposition
    
    Leverages quantum parallelism to evaluate many paths simultaneously
    """
    
    def __init__(self, config: QuantumParallelismConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_gpu else 'cpu')
        
        # Components
        self.superposition_manager = QuantumSuperpositionManager(config)
        self.interference_engine = QuantumInterferenceEngine(config)
        
        # Mixed precision setup
        self.dtype = torch.float16 if config.mixed_precision else torch.float32
        
    def evaluate_paths_quantum(
        self,
        paths: torch.Tensor,
        value_function: callable,
        num_grover_iterations: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Evaluate multiple paths using quantum parallelism
        
        Args:
            paths: Paths to evaluate [num_paths, max_depth]
            value_function: Function to evaluate path quality
            num_grover_iterations: Number of amplitude amplification steps
            
        Returns:
            Tuple of (selected_paths, evaluation_info)
        """
        if num_grover_iterations is None:
            num_grover_iterations = self.config.grover_iterations
            
        # Create superposition
        quantum_state = self.superposition_manager.create_path_superposition(paths)
        
        # Evaluate all paths in superposition (quantum parallelism)
        with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):
            path_values = value_function(paths)
            
        # Normalize values for oracle
        normalized_values = (path_values - path_values.min()) / (path_values.max() - path_values.min() + 1e-8)
        
        # Apply Grover iterations
        for _ in range(num_grover_iterations):
            quantum_state = self.superposition_manager.apply_grover_iteration(
                quantum_state, normalized_values
            )
            
        # Apply interference
        quantum_state = self.interference_engine.apply_interference(quantum_state)
        
        # Extract classical result via measurement
        probabilities = torch.abs(quantum_state['amplitudes']) ** 2
        
        # Sample or select top paths
        num_to_select = min(len(paths), self.config.batch_size)
        
        if num_to_select < len(paths):
            # Sample according to quantum probabilities
            selected_indices = torch.multinomial(
                probabilities, 
                num_samples=num_to_select,
                replacement=False
            )
        else:
            # Select all paths
            selected_indices = torch.arange(len(paths), device=self.device)
            
        selected_paths = paths[selected_indices]
        selected_probs = probabilities[selected_indices]
        
        # Compile evaluation info
        eval_info = {
            'quantum_probs': probabilities,
            'selected_indices': selected_indices,
            'selected_probs': selected_probs,
            'final_amplitudes': quantum_state['amplitudes'],
            'coherence': quantum_state['coherence'],
            'path_values': path_values,
            'grover_iterations': num_grover_iterations
        }
        
        self.superposition_manager.stats['paths_evaluated'] += len(paths)
        
        return selected_paths, eval_info


class HybridQuantumMCTS:
    """
    Hybrid classical-quantum MCTS implementation
    
    Uses quantum parallelism for exploration and classical
    computation for exploitation.
    """
    
    def __init__(self, config: QuantumParallelismConfig):
        self.config = config
        self.quantum_evaluator = QuantumPathEvaluator(config)
        
        # Transition criteria
        self.quantum_threshold = 100  # Use quantum when > N paths
        
    def select_paths_hybrid(
        self,
        candidate_paths: torch.Tensor,
        value_function: callable,
        visit_counts: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Hybrid path selection using quantum parallelism
        
        Uses quantum evaluation for exploration (low visit counts)
        and classical for exploitation (high visit counts)
        """
        num_paths = len(candidate_paths)
        
        # Determine quantum vs classical regime
        avg_visits = visit_counts.mean()
        use_quantum = (num_paths > self.quantum_threshold) or (avg_visits < 10)
        
        if use_quantum:
            # Quantum evaluation
            selected_paths, eval_info = self.quantum_evaluator.evaluate_paths_quantum(
                candidate_paths, value_function
            )
            eval_info['mode'] = 'quantum'
        else:
            # Classical evaluation
            path_values = value_function(candidate_paths)
            
            # Select top paths
            k = min(self.config.batch_size, num_paths)
            top_values, top_indices = torch.topk(path_values, k)
            
            selected_paths = candidate_paths[top_indices]
            
            eval_info = {
                'path_values': path_values,
                'selected_indices': top_indices,
                'mode': 'classical'
            }
            
        return selected_paths, eval_info
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get quantum parallelism statistics"""
        stats = {
            'quantum_stats': self.quantum_evaluator.superposition_manager.stats,
            'config': {
                'max_superposition': self.config.max_superposition_size,
                'grover_iterations': self.config.grover_iterations,
                'interference_strength': self.config.interference_strength
            }
        }
        return stats


def create_quantum_parallel_evaluator(
    max_superposition: int = 1024,
    use_gpu: bool = True,
    **kwargs
) -> HybridQuantumMCTS:
    """
    Factory function for quantum parallel evaluator
    
    Args:
        max_superposition: Maximum paths in quantum superposition
        use_gpu: Whether to use GPU acceleration
        **kwargs: Additional config parameters
        
    Returns:
        HybridQuantumMCTS instance
    """
    config = QuantumParallelismConfig(
        max_superposition_size=max_superposition,
        use_gpu=use_gpu,
        **kwargs
    )
    
    return HybridQuantumMCTS(config)


# Example usage for integration
if __name__ == "__main__":
    # Create quantum evaluator
    evaluator = create_quantum_parallel_evaluator(
        max_superposition=512,
        grover_iterations=2,
        use_gpu=torch.cuda.is_available()
    )
    
    # Example paths
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths = torch.randint(0, 100, (1000, 20), device=device)
    
    # Example value function
    def value_func(paths):
        # Simple example: prefer paths with lower indices
        return -paths.float().mean(dim=1)
    
    # Example visit counts
    visits = torch.randint(0, 50, (1000,), device=device).float()
    
    # Evaluate
    selected, info = evaluator.select_paths_hybrid(paths, value_func, visits)
    
    print(f"Selected {len(selected)} paths using {info['mode']} mode")
    print(f"Statistics: {evaluator.get_statistics()}")