"""
Vectorized operations for CPU MCTS using NumPy and Numba.

This module provides:
- SIMD-optimized UCB calculations via NumPy/Numba
- Batch processing for multiple nodes
- Parallel operations using Numba's prange
- Efficient memory access patterns
"""

import numpy as np
from numba import jit, njit, prange, vectorize, float32, int32
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


def ensure_numba_compiled():
    """Precompile Numba functions to avoid JIT overhead during search
    
    This function is called at startup to compile all Numba functions
    with common array sizes used in MCTS.
    """
    # Common sizes in MCTS
    common_sizes = [8, 16, 32, 64, 128, 256]
    
    for size in common_sizes:
        # Create dummy data
        visit_counts = np.ones(size, dtype=np.float32)
        value_sums = np.ones(size, dtype=np.float32)
        priors = np.ones(size, dtype=np.float32) / size
        child_indices = np.arange(size, dtype=np.int32)
        
        # Force compilation by calling the function
        try:
            _ = batch_ucb_scores(visit_counts, value_sums, priors, 
                               10, child_indices, 1.4)
        except Exception:
            pass  # Ignore any errors during precompilation
    
    logger.info("Numba functions precompiled for common array sizes")

# Numba compilation flags for maximum performance
NUMBA_FLAGS = {
    'parallel': True,
    'fastmath': True,
    'cache': True
}




@njit(**NUMBA_FLAGS)
def batch_ucb_scores(
    visit_counts: np.ndarray,
    value_sums: np.ndarray,
    priors: np.ndarray,
    parent_visits: int,
    child_indices: np.ndarray,
    c_puct: float = 1.4
) -> np.ndarray:
    """
    Compute UCB scores for a batch of children using vectorized operations.
    
    This function leverages NumPy's internal SIMD optimizations and
    Numba's parallel execution for maximum performance.
    
    Args:
        visit_counts: Global visit count array
        value_sums: Global value sum array
        priors: Global prior array
        parent_visits: Parent node visit count
        child_indices: Indices of children to evaluate
        c_puct: UCB exploration constant
        
    Returns:
        UCB scores for each child
    """
    n = len(child_indices)
    ucb_scores = np.empty(n, dtype=np.float32)
    
    # Precompute sqrt for efficiency
    sqrt_parent = np.sqrt(parent_visits)
    
    # Parallel loop using Numba's prange
    for i in prange(n):
        idx = child_indices[i]
        visits = visit_counts[idx]
        
        # Q value with numerical stability
        if visits > 0:
            q_value = value_sums[idx] / visits
        else:
            q_value = 0.0
        
        # Exploration term
        exploration = c_puct * priors[idx] * sqrt_parent / (visits + 1)
        
        ucb_scores[i] = q_value + exploration
    
    return ucb_scores


@njit(**NUMBA_FLAGS)
def vectorized_ucb_matrix(
    visit_counts: np.ndarray,
    value_sums: np.ndarray,
    priors: np.ndarray,
    parent_visits: np.ndarray,
    child_indices_matrix: np.ndarray,
    c_puct: float = 1.4
) -> np.ndarray:
    """
    Compute UCB scores for multiple parents and their children in parallel.
    
    This is ideal for wave-based MCTS where we process multiple nodes simultaneously.
    
    Args:
        visit_counts: Global visit count array
        value_sums: Global value sum array
        priors: Global prior array
        parent_visits: Array of parent visit counts
        child_indices_matrix: 2D array where each row contains child indices for a parent
        c_puct: UCB exploration constant
        
    Returns:
        2D array of UCB scores
    """
    n_parents, n_children = child_indices_matrix.shape
    ucb_matrix = np.empty((n_parents, n_children), dtype=np.float32)
    
    # Process all parent-child pairs in parallel
    for i in prange(n_parents):
        sqrt_parent = np.sqrt(parent_visits[i])
        
        for j in range(n_children):
            idx = child_indices_matrix[i, j]
            
            # Invalid child marker
            if idx < 0:
                ucb_matrix[i, j] = -1e9
                continue
            
            visits = visit_counts[idx]
            
            # Q value
            if visits > 0:
                q_value = value_sums[idx] / visits
            else:
                q_value = 0.0
            
            # Exploration
            exploration = c_puct * priors[idx] * sqrt_parent / (visits + 1)
            
            ucb_matrix[i, j] = q_value + exploration
    
    return ucb_matrix


@njit(parallel=True)
def parallel_argmax_rows(matrix: np.ndarray) -> np.ndarray:
    """
    Find argmax for each row in parallel.
    
    Args:
        matrix: 2D array
        
    Returns:
        Array of argmax indices for each row
    """
    n_rows, n_cols = matrix.shape
    best_indices = np.empty(n_rows, dtype=np.int32)
    
    for i in prange(n_rows):
        best_idx = 0
        best_val = matrix[i, 0]
        
        for j in range(1, n_cols):
            if matrix[i, j] > best_val:
                best_val = matrix[i, j]
                best_idx = j
        
        best_indices[i] = best_idx
    
    return best_indices


@njit(**NUMBA_FLAGS)
def vectorized_backup(
    visit_counts: np.ndarray,
    value_sums: np.ndarray,
    nodes: np.ndarray,
    values: np.ndarray
):
    """
    Backup values for multiple nodes in parallel.
    
    Args:
        visit_counts: Global visit count array
        value_sums: Global value sum array
        nodes: Array of node indices to update
        values: Array of values to add
    """
    n = len(nodes)
    
    # Parallel update
    for i in prange(n):
        node = nodes[i]
        value_sums[node] += values[i]
        visit_counts[node] += 1


@njit(parallel=True)
def wave_selection(
    visit_counts: np.ndarray,
    value_sums: np.ndarray,
    priors: np.ndarray,
    parents: np.ndarray,
    children_start: np.ndarray,
    children_count: np.ndarray,
    wave_size: int,
    c_puct: float = 1.4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select multiple paths in parallel for wave-based MCTS.
    
    Returns:
        Tuple of (selected_nodes, depths)
    """
    selected = np.empty(wave_size, dtype=np.int32)
    depths = np.empty(wave_size, dtype=np.int32)
    
    # Process waves in parallel
    for w in prange(wave_size):
        current = 0  # Start from root
        depth = 0
        
        # Traverse until leaf
        while children_count[current] > 0 and depth < 100:  # Max depth safety
            # Get children range
            start = children_start[current]
            count = children_count[current]
            
            # Compute UCB scores for children
            parent_visits = visit_counts[current]
            sqrt_parent = np.sqrt(parent_visits)
            
            best_child = start
            best_ucb = -1e9
            
            # Find best child
            for i in range(count):
                child = start + i
                visits = visit_counts[child]
                
                # Q value
                if visits > 0:
                    q_value = value_sums[child] / visits
                else:
                    q_value = 0.0
                
                # Exploration
                exploration = c_puct * priors[child] * sqrt_parent / (visits + 1)
                ucb = q_value + exploration
                
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child
            
            current = best_child
            depth += 1
        
        selected[w] = current
        depths[w] = depth
    
    return selected, depths


class VectorizedOperations:
    """
    High-level interface for vectorized MCTS operations.
    
    This class provides optimized operations that leverage:
    - NumPy's SIMD capabilities
    - Numba's JIT compilation and parallelization
    - Efficient memory access patterns
    - Batch processing
    """
    
    def __init__(self, c_puct: float = 1.4, use_numba: bool = True):
        self.c_puct = c_puct
        self.use_numba = use_numba
        
        # Warm up JIT compilation
        if use_numba:
            self._warmup_jit()
    
    def _warmup_jit(self):
        """Warm up Numba JIT compilation"""
        dummy_data = np.zeros(10, dtype=np.int32)
        dummy_float = np.zeros(10, dtype=np.float32)
        dummy_indices = np.arange(10, dtype=np.int32)
        
        # Trigger compilation
        _ = batch_ucb_scores(dummy_data, dummy_float, dummy_float, 100, dummy_indices)
        logger.debug("Numba JIT compilation completed")
    
    def batch_ucb_scores(
        self,
        visit_counts: np.ndarray,
        value_sums: np.ndarray,
        priors: np.ndarray,
        parent_visits: int,
        child_indices: np.ndarray
    ) -> np.ndarray:
        """Compute UCB scores for a batch of children"""
        if self.use_numba:
            return batch_ucb_scores(
                visit_counts, value_sums, priors,
                parent_visits, child_indices, self.c_puct
            )
        else:
            # Fallback to pure NumPy
            visits = visit_counts[child_indices]
            values = value_sums[child_indices]
            prior_vals = priors[child_indices]
            
            q_values = np.where(visits > 0, values / visits, 0.0)
            exploration = self.c_puct * prior_vals * np.sqrt(parent_visits) / (visits + 1)
            
            return q_values + exploration
    
    def select_best_children_batch(
        self,
        visit_counts: np.ndarray,
        value_sums: np.ndarray,
        priors: np.ndarray,
        parent_indices: np.ndarray,
        children_matrix: np.ndarray
    ) -> np.ndarray:
        """Select best children for multiple parents"""
        # Get parent visits
        parent_visits = visit_counts[parent_indices]
        
        # Compute UCB matrix
        ucb_matrix = vectorized_ucb_matrix(
            visit_counts, value_sums, priors,
            parent_visits, children_matrix, self.c_puct
        )
        
        # Find best children
        best_indices = parallel_argmax_rows(ucb_matrix)
        
        # Convert to actual child nodes
        n_parents = len(parent_indices)
        best_children = np.empty(n_parents, dtype=np.int32)
        
        for i in range(n_parents):
            if best_indices[i] >= 0 and children_matrix[i, best_indices[i]] >= 0:
                best_children[i] = children_matrix[i, best_indices[i]]
            else:
                best_children[i] = -1
        
        return best_children
    
    def backup_wave(
        self,
        visit_counts: np.ndarray,
        value_sums: np.ndarray,
        paths: List[List[int]],
        values: np.ndarray
    ):
        """Backup values for multiple paths efficiently"""
        # Flatten paths and values
        all_nodes = []
        all_values = []
        
        for path, value in zip(paths, values):
            current_value = value
            for node in reversed(path):
                all_nodes.append(node)
                all_values.append(current_value)
                current_value = -current_value  # Flip for opponent
        
        if all_nodes:
            nodes_array = np.array(all_nodes, dtype=np.int32)
            values_array = np.array(all_values, dtype=np.float32)
            vectorized_backup(visit_counts, value_sums, nodes_array, values_array)
    
    @staticmethod
    def create_batch_selector(wave_size: int = 64):
        """Create a batch selector for wave-based selection"""
        return lambda *args: wave_selection(*args, wave_size)