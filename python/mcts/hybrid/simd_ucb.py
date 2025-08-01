"""SIMD-Optimized UCB Calculations

Uses NumPy's vectorized operations which compile to SIMD instructions.
For maximum performance, a C++ extension with explicit AVX2 would be needed.
"""

import numpy as np
import torch
from typing import Tuple, Optional
import numba
from numba import njit, prange


class SIMDUCBCalculator:
    """SIMD-optimized UCB calculations using vectorized operations
    
    Features:
    - Batch UCB calculation for multiple nodes
    - Vectorized arithmetic operations
    - Cache-friendly data layout
    - Optional Numba JIT compilation
    """
    
    def __init__(self, c_puct: float = 1.414, use_numba: bool = True):
        """Initialize SIMD UCB calculator
        
        Args:
            c_puct: Exploration constant
            use_numba: Whether to use Numba JIT compilation
        """
        self.c_puct = c_puct
        self.use_numba = use_numba
        
        # Pre-compile Numba functions if enabled
        if use_numba:
            self._calculate_ucb_numba = self._create_numba_ucb()
            
    def calculate_ucb_batch(
        self,
        visit_counts: np.ndarray,
        value_sums: np.ndarray,
        priors: np.ndarray,
        parent_visits: int,
        virtual_visits: Optional[np.ndarray] = None,
        virtual_loss_value: float = -3.0
    ) -> np.ndarray:
        """Calculate UCB values for a batch of children
        
        Args:
            visit_counts: Visit counts for children (N,)
            value_sums: Value sums for children (N,)
            priors: Prior probabilities (N,)
            parent_visits: Parent node visit count
            virtual_visits: Virtual visit counts for parallelization (N,)
            virtual_loss_value: Value of virtual loss
            
        Returns:
            UCB values for all children (N,)
        """
        
        if self.use_numba and hasattr(self, '_calculate_ucb_numba'):
            return self._calculate_ucb_numba(
                visit_counts, value_sums, priors, 
                parent_visits, virtual_visits, 
                virtual_loss_value, self.c_puct
            )
        else:
            return self._calculate_ucb_numpy(
                visit_counts, value_sums, priors,
                parent_visits, virtual_visits,
                virtual_loss_value
            )
            
    def _calculate_ucb_numpy(
        self,
        visit_counts: np.ndarray,
        value_sums: np.ndarray,
        priors: np.ndarray,
        parent_visits: int,
        virtual_visits: Optional[np.ndarray],
        virtual_loss_value: float
    ) -> np.ndarray:
        """NumPy vectorized UCB calculation"""
        
        # Apply virtual loss if provided
        if virtual_visits is not None:
            effective_visits = visit_counts + virtual_visits
            effective_values = value_sums + (virtual_visits * virtual_loss_value)
        else:
            effective_visits = visit_counts
            effective_values = value_sums
            
        # Avoid division by zero
        effective_visits_safe = np.maximum(effective_visits, 1)
        
        # Q-values: average value per visit
        q_values = effective_values / effective_visits_safe
        
        # Exploration term: c_puct * P * sqrt(parent) / (1 + N)
        sqrt_parent = np.sqrt(parent_visits)
        exploration = self.c_puct * priors * sqrt_parent / (1 + effective_visits)
        
        # UCB = Q + U
        ucb_values = q_values + exploration
        
        return ucb_values
        
    @staticmethod
    def _create_numba_ucb():
        """Create Numba-compiled UCB function"""
        
        @njit(fastmath=True, cache=True)
        def calculate_ucb_numba(
            visit_counts: np.ndarray,
            value_sums: np.ndarray,
            priors: np.ndarray,
            parent_visits: int,
            virtual_visits: Optional[np.ndarray],
            virtual_loss_value: float,
            c_puct: float
        ) -> np.ndarray:
            """Numba JIT-compiled UCB calculation"""
            
            n = len(visit_counts)
            ucb_values = np.empty(n, dtype=np.float32)
            
            sqrt_parent = np.sqrt(parent_visits)
            
            for i in range(n):
                # Apply virtual loss
                if virtual_visits is not None:
                    visits = visit_counts[i] + virtual_visits[i]
                    values = value_sums[i] + (virtual_visits[i] * virtual_loss_value)
                else:
                    visits = visit_counts[i]
                    values = value_sums[i]
                    
                # Avoid division by zero
                if visits == 0:
                    q_value = 0.0
                else:
                    q_value = values / visits
                    
                # Exploration term
                exploration = c_puct * priors[i] * sqrt_parent / (1 + visits)
                
                # UCB
                ucb_values[i] = q_value + exploration
                
            return ucb_values
            
        return calculate_ucb_numba
        
    def select_best_child_batch(
        self,
        children_per_parent: np.ndarray,
        visit_counts: np.ndarray,
        value_sums: np.ndarray,
        priors: np.ndarray,
        parent_visits: np.ndarray,
        virtual_visits: Optional[np.ndarray] = None,
        virtual_loss_value: float = -3.0
    ) -> np.ndarray:
        """Select best child for multiple parent nodes
        
        Args:
            children_per_parent: Number of children for each parent (P,)
            visit_counts: Flattened visit counts for all children
            value_sums: Flattened value sums for all children
            priors: Flattened priors for all children
            parent_visits: Visit counts for each parent (P,)
            virtual_visits: Flattened virtual visits
            virtual_loss_value: Virtual loss value
            
        Returns:
            Best child index for each parent (P,)
        """
        
        num_parents = len(children_per_parent)
        best_children = np.zeros(num_parents, dtype=np.int32)
        
        offset = 0
        for i in range(num_parents):
            n_children = children_per_parent[i]
            if n_children == 0:
                continue
                
            # Extract data for this parent's children
            child_visits = visit_counts[offset:offset + n_children]
            child_values = value_sums[offset:offset + n_children]
            child_priors = priors[offset:offset + n_children]
            
            child_virtual = None
            if virtual_visits is not None:
                child_virtual = virtual_visits[offset:offset + n_children]
                
            # Calculate UCB for all children
            ucb_values = self.calculate_ucb_batch(
                child_visits, child_values, child_priors,
                parent_visits[i], child_virtual, virtual_loss_value
            )
            
            # Select best
            best_idx = np.argmax(ucb_values)
            best_children[i] = offset + best_idx
            
            offset += n_children
            
        return best_children


class BatchedUCBSelector:
    """Batched UCB selection for multiple nodes simultaneously
    
    Processes multiple nodes in parallel using SIMD operations.
    """
    
    def __init__(self, batch_size: int = 32, c_puct: float = 1.414):
        """Initialize batched selector
        
        Args:
            batch_size: Number of nodes to process simultaneously
            c_puct: Exploration constant
        """
        self.batch_size = batch_size
        self.ucb_calculator = SIMDUCBCalculator(c_puct=c_puct)
        
    def select_batch(
        self,
        tree,
        node_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select best children for a batch of nodes
        
        Args:
            tree: CSR tree structure
            node_indices: Indices of parent nodes (B,)
            
        Returns:
            best_children: Best child index for each parent (B,)
            best_actions: Best action for each parent (B,)
        """
        
        batch_size = len(node_indices)
        best_children = np.zeros(batch_size, dtype=np.int32)
        best_actions = np.zeros(batch_size, dtype=np.int32)
        
        # Process each node (could be further vectorized)
        for i, node_idx in enumerate(node_indices):
            # Get children data from tree
            children, actions, priors = tree.get_children(node_idx)
            
            if len(children) == 0:
                continue
                
            # Get statistics
            child_visits = tree.node_data.visit_counts[children].cpu().numpy()
            child_values = tree.node_data.value_sums[children].cpu().numpy()
            parent_visits = tree.node_data.visit_counts[node_idx].item()
            
            # Virtual loss if enabled
            child_virtual = None
            if tree.config.enable_virtual_loss:
                child_virtual = tree.node_data.virtual_loss_counts[children].cpu().numpy()
                
            # Calculate UCB values
            ucb_values = self.ucb_calculator.calculate_ucb_batch(
                child_visits,
                child_values,
                priors.cpu().numpy(),
                parent_visits,
                child_virtual,
                tree.config.virtual_loss_value
            )
            
            # Select best
            best_idx = np.argmax(ucb_values)
            best_children[i] = children[best_idx].item()
            best_actions[i] = actions[best_idx].item()
            
        return best_children, best_actions


def benchmark_simd_ucb():
    """Benchmark SIMD UCB performance"""
    
    import time
    
    # Test data
    n_children = 225  # Typical for 15x15 Gomoku
    n_iterations = 10000
    
    # Random data
    np.random.seed(42)
    visit_counts = np.random.randint(0, 100, n_children).astype(np.float32)
    value_sums = np.random.randn(n_children).astype(np.float32) * visit_counts
    priors = np.random.rand(n_children).astype(np.float32)
    priors = priors / priors.sum()  # Normalize
    parent_visits = 1000
    
    # Test standard NumPy
    calculator_numpy = SIMDUCBCalculator(use_numba=False)
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        ucb_values = calculator_numpy.calculate_ucb_batch(
            visit_counts, value_sums, priors, parent_visits
        )
    numpy_time = time.perf_counter() - start
    
    # Test Numba
    calculator_numba = SIMDUCBCalculator(use_numba=True)
    
    # Warmup
    for _ in range(10):
        ucb_values_numba = calculator_numba.calculate_ucb_batch(
            visit_counts, value_sums, priors, parent_visits
        )
        
    start = time.perf_counter()
    for _ in range(n_iterations):
        ucb_values_numba = calculator_numba.calculate_ucb_batch(
            visit_counts, value_sums, priors, parent_visits
        )
    numba_time = time.perf_counter() - start
    
    # Verify same results
    np.testing.assert_allclose(ucb_values, ucb_values_numba, rtol=1e-5)
    
    print(f"UCB Calculation Benchmark ({n_children} children, {n_iterations} iterations):")
    print(f"  NumPy: {numpy_time:.3f}s ({n_iterations/numpy_time:.0f} ops/sec)")
    print(f"  Numba: {numba_time:.3f}s ({n_iterations/numba_time:.0f} ops/sec)")
    print(f"  Speedup: {numpy_time/numba_time:.2f}x")
    
    # Time per operation
    numpy_us = (numpy_time / n_iterations) * 1e6
    numba_us = (numba_time / n_iterations) * 1e6
    print(f"\nPer-operation time:")
    print(f"  NumPy: {numpy_us:.1f} μs")
    print(f"  Numba: {numba_us:.1f} μs")


if __name__ == "__main__":
    benchmark_simd_ucb()