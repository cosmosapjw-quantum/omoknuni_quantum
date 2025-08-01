"""Optimized Tree Implementation with Cython Support

Provides a unified interface that can use either:
- Cython-optimized tree operations (if available)
- Pure Python fallback
"""

import numpy as np
import logging
from typing import Tuple, List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Try to import Cython version
try:
    # Try relative import first
    try:
        from .cython_tree_ops import CythonTree
    except ImportError:
        # Try direct import for testing
        import cython_tree_ops
        CythonTree = cython_tree_ops.CythonTree
    CYTHON_AVAILABLE = True
    logger.info("Cython tree operations available")
except ImportError as e:
    CYTHON_AVAILABLE = False
    logger.warning(f"Cython tree operations not available: {e}, using Python fallback")


class OptimizedTree:
    """High-performance tree with automatic Cython optimization
    
    Automatically uses Cython implementation if available,
    otherwise falls back to pure Python.
    """
    
    def __init__(
        self,
        capacity: int,
        c_puct: float = 1.414,
        use_cython: bool = True
    ):
        """Initialize optimized tree
        
        Args:
            capacity: Maximum number of nodes
            c_puct: Exploration constant
            use_cython: Whether to use Cython if available
        """
        self.capacity = capacity
        self.c_puct = c_puct
        self.use_cython = use_cython and CYTHON_AVAILABLE
        
        if self.use_cython:
            self._tree = CythonTree(capacity, c_puct)
            logger.info("Using Cython-optimized tree")
        else:
            self._tree = PythonTree(capacity, c_puct)
            logger.info("Using pure Python tree")
            
    def add_node(self, parent_idx: int, prior: float) -> int:
        """Add a new node"""
        return self._tree.add_node(parent_idx, prior)
        
    def add_children_batch(
        self,
        parent_idx: int,
        actions: np.ndarray,
        priors: np.ndarray
    ):
        """Add multiple children to a node"""
        self._tree.add_children_batch(parent_idx, actions, priors)
        
    def select_best_child(self, node_idx: int) -> int:
        """Select best child using UCB"""
        return self._tree.select_best_child(node_idx)
        
    def select_batch_parallel(self, node_indices: np.ndarray) -> np.ndarray:
        """Select best children for multiple nodes"""
        if self.use_cython:
            return self._tree.select_batch_parallel(node_indices)
        else:
            # Python fallback
            results = np.zeros(len(node_indices), dtype=np.int32)
            for i, node_idx in enumerate(node_indices):
                results[i] = self._tree.select_best_child(node_idx)
            return results
            
    def update_value(self, node_idx: int, value: float):
        """Update node value"""
        self._tree.update_value(node_idx, value)
        
    def backup_values_batch(self, paths: np.ndarray, values: np.ndarray):
        """Backup values for multiple paths"""
        self._tree.backup_values_batch(paths, values)
        
    def get_children_info(self, node_idx: int) -> Tuple:
        """Get children information"""
        return self._tree.get_children_info(node_idx)
        
    def calculate_ucb_vectorized(
        self,
        children: np.ndarray,
        parent_visits: float
    ) -> np.ndarray:
        """Vectorized UCB calculation"""
        if self.use_cython:
            return self._tree.calculate_ucb_vectorized(children, parent_visits)
        else:
            # Python fallback using NumPy
            return self._calculate_ucb_numpy(children, parent_visits)
            
    def _calculate_ucb_numpy(
        self,
        children: np.ndarray,
        parent_visits: float
    ) -> np.ndarray:
        """NumPy-based UCB calculation"""
        # Get child data
        _, priors, visit_counts, value_sums = self.get_children_info(children[0])
        
        # Q-values
        q_values = np.where(
            visit_counts > 0,
            value_sums / visit_counts,
            np.zeros_like(value_sums)
        )
        
        # Exploration
        sqrt_parent = np.sqrt(parent_visits)
        exploration = self.c_puct * priors * sqrt_parent / (1 + visit_counts)
        
        return q_values + exploration
        
    def get_stats(self) -> Dict[str, Any]:
        """Get tree statistics"""
        return self._tree.get_stats()
        
    @property
    def num_nodes(self) -> int:
        """Get number of nodes in tree"""
        if hasattr(self._tree, 'num_nodes'):
            return self._tree.num_nodes
        else:
            return self._tree.get_stats()['num_nodes']


class PythonTree:
    """Pure Python tree implementation (fallback)"""
    
    def __init__(self, capacity: int, c_puct: float = 1.414):
        self.capacity = capacity
        self.c_puct = c_puct
        self.num_nodes = 0
        
        # Node storage
        self.visit_counts = np.zeros(capacity, dtype=np.int32)
        self.value_sums = np.zeros(capacity, dtype=np.float32)
        self.priors = np.zeros(capacity, dtype=np.float32)
        self.parent_indices = np.full(capacity, -1, dtype=np.int32)
        self.first_child_indices = np.full(capacity, -1, dtype=np.int32)
        self.num_children = np.zeros(capacity, dtype=np.int32)
        self.is_expanded = np.zeros(capacity, dtype=bool)
        self.is_terminal = np.zeros(capacity, dtype=bool)
        
    def add_node(self, parent_idx: int, prior: float) -> int:
        """Add a new node"""
        if self.num_nodes >= self.capacity:
            return -1
            
        node_idx = self.num_nodes
        self.priors[node_idx] = prior
        self.parent_indices[node_idx] = parent_idx
        self.num_nodes += 1
        
        return node_idx
        
    def add_children_batch(
        self,
        parent_idx: int,
        actions: np.ndarray,
        priors: np.ndarray
    ):
        """Add multiple children"""
        num_children = len(actions)
        if self.num_nodes + num_children > self.capacity:
            return
            
        first_child = self.num_nodes
        self.first_child_indices[parent_idx] = first_child
        self.num_children[parent_idx] = num_children
        self.is_expanded[parent_idx] = True
        
        for i, prior in enumerate(priors):
            self.add_node(parent_idx, prior)
            
    def select_best_child(self, node_idx: int) -> int:
        """Select best child using UCB"""
        first_child = self.first_child_indices[node_idx]
        num_children = self.num_children[node_idx]
        
        if num_children == 0 or first_child < 0:
            return -1
            
        parent_visits = self.visit_counts[node_idx]
        if parent_visits == 0:
            parent_visits = 1
            
        best_idx = -1
        best_ucb = -np.inf
        
        sqrt_parent = np.sqrt(parent_visits)
        
        for i in range(num_children):
            child_idx = first_child + i
            
            # Q-value
            visits = self.visit_counts[child_idx]
            if visits > 0:
                q_value = self.value_sums[child_idx] / visits
            else:
                q_value = 0.0
                
            # Exploration
            prior = self.priors[child_idx]
            u_value = self.c_puct * prior * sqrt_parent / (1 + visits)
            
            # UCB
            ucb = q_value + u_value
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_idx = child_idx
                
        return best_idx
        
    def update_value(self, node_idx: int, value: float):
        """Update node value"""
        self.visit_counts[node_idx] += 1
        self.value_sums[node_idx] += value
        
    def backup_values_batch(self, paths: np.ndarray, values: np.ndarray):
        """Backup values for multiple paths"""
        for i in range(len(paths)):
            value = values[i]
            for node_idx in paths[i]:
                if node_idx < 0:
                    break
                self.update_value(node_idx, value)
                value = -value
                
    def get_children_info(self, node_idx: int) -> Tuple:
        """Get children information"""
        first_child = self.first_child_indices[node_idx]
        num_children = self.num_children[node_idx]
        
        if num_children == 0 or first_child < 0:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float32)
            )
            
        children_indices = np.arange(first_child, first_child + num_children, dtype=np.int32)
        priors = self.priors[children_indices]
        visit_counts = self.visit_counts[children_indices]
        value_sums = self.value_sums[children_indices]
        
        return (children_indices, priors, visit_counts, value_sums)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get tree statistics"""
        return {
            'num_nodes': self.num_nodes,
            'capacity': self.capacity,
            'total_visits': np.sum(self.visit_counts[:self.num_nodes]),
            'expanded_nodes': np.sum(self.is_expanded[:self.num_nodes]),
            'terminal_nodes': np.sum(self.is_terminal[:self.num_nodes]),
            'memory_usage_mb': (self.num_nodes * 40) / (1024.0 * 1024.0)
        }


def benchmark_tree_operations():
    """Benchmark Cython vs Python tree operations"""
    
    import time
    
    capacity = 100000
    num_operations = 10000
    
    # Test both implementations
    for use_cython in [False, True]:
        if use_cython and not CYTHON_AVAILABLE:
            continue
            
        tree = OptimizedTree(capacity, use_cython=use_cython)
        
        # Add root
        tree.add_node(-1, 1.0)
        
        # Benchmark node addition
        start = time.perf_counter()
        for i in range(num_operations):
            parent = i % 100  # Use various parents
            tree.add_node(parent, np.random.rand())
        add_time = time.perf_counter() - start
        
        # Benchmark selection
        node_indices = np.random.randint(0, min(tree.num_nodes, 1000), 1000)
        
        start = time.perf_counter()
        for _ in range(100):
            results = tree.select_batch_parallel(node_indices)
        select_time = time.perf_counter() - start
        
        # Benchmark backup
        paths = np.random.randint(0, min(tree.num_nodes, 100), (100, 20))
        values = np.random.randn(100)
        
        start = time.perf_counter()
        tree.backup_values_batch(paths, values)
        backup_time = time.perf_counter() - start
        
        impl_name = "Cython" if use_cython else "Python"
        print(f"\n{impl_name} Implementation:")
        print(f"  Add nodes: {num_operations/add_time:.0f} ops/sec")
        print(f"  Selection: {100*1000/select_time:.0f} selections/sec")
        print(f"  Backup: {100/backup_time:.0f} paths/sec")
        
        stats = tree.get_stats()
        print(f"  Memory: {stats['memory_usage_mb']:.1f} MB")


if __name__ == "__main__":
    benchmark_tree_operations()