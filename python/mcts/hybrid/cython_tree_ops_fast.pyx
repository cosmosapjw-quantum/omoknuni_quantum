# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

"""Fast Cython Tree Operations - Optimized for single-threaded performance

Key optimizations:
- No parallel overhead
- Inline functions
- Memory views for zero-copy access
- Compact data structures
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
from libc.string cimport memset

# Type definitions
ctypedef np.int32_t INT32
ctypedef np.float32_t FLOAT32
ctypedef np.uint8_t UINT8

# Constants
DEF MAX_CHILDREN = 512
DEF CACHE_LINE_SIZE = 64


cdef struct NodeData:
    # Compact node data structure for cache efficiency
    INT32 visit_count
    FLOAT32 value_sum
    FLOAT32 prior
    INT32 parent_idx
    INT32 first_child_idx
    INT32 num_children
    UINT8 is_expanded
    UINT8 is_terminal


@cython.final
cdef class FastCythonTree:
    """Ultra-fast tree implementation optimized for single-threaded performance"""
    
    cdef:
        NodeData* nodes
        INT32 capacity
        INT32 num_nodes
        FLOAT32 c_puct
        # Pre-computed constants
        FLOAT32 sqrt_2
        
    def __init__(self, int capacity, float c_puct=1.414):
        self.capacity = capacity
        self.num_nodes = 0
        self.c_puct = c_puct
        self.sqrt_2 = sqrt(2.0)
        
        # Allocate aligned memory
        self.nodes = <NodeData*>malloc(capacity * sizeof(NodeData))
        if self.nodes == NULL:
            raise MemoryError("Failed to allocate tree memory")
            
        # Zero-initialize all memory
        memset(self.nodes, 0, capacity * sizeof(NodeData))
        
    def __dealloc__(self):
        if self.nodes != NULL:
            free(self.nodes)
            
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_node(self, INT32 parent_idx, FLOAT32 prior):
        """Add a new node (fast version)"""
        cdef INT32 node_idx = self.num_nodes
        
        if node_idx >= self.capacity:
            return -1
            
        # Direct struct access
        self.nodes[node_idx].visit_count = 0
        self.nodes[node_idx].value_sum = 0.0
        self.nodes[node_idx].prior = prior
        self.nodes[node_idx].parent_idx = parent_idx
        self.nodes[node_idx].first_child_idx = -1
        self.nodes[node_idx].num_children = 0
        self.nodes[node_idx].is_expanded = 0
        self.nodes[node_idx].is_terminal = 0
        
        self.num_nodes += 1
        return node_idx
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_children_batch_fast(
        self, 
        INT32 parent_idx,
        FLOAT32[:] priors  # Use memory view
    ):
        """Add children using memory views for speed"""
        cdef:
            int num_children = priors.shape[0]
            int i
            INT32 first_child = self.num_nodes
            
        if self.num_nodes + num_children > self.capacity:
            return
            
        # Mark parent as expanded
        self.nodes[parent_idx].is_expanded = 1
        self.nodes[parent_idx].first_child_idx = first_child
        self.nodes[parent_idx].num_children = num_children
        
        # Add children in tight loop
        for i in range(num_children):
            self.nodes[self.num_nodes].prior = priors[i]
            self.nodes[self.num_nodes].parent_idx = parent_idx
            self.nodes[self.num_nodes].first_child_idx = -1
            self.num_nodes += 1
            
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline INT32 _select_best_child_inline(self, INT32 node_idx) nogil:
        """Inline UCB selection for maximum speed"""
        cdef:
            NodeData* node = &self.nodes[node_idx]
            INT32 first_child = node.first_child_idx
            INT32 num_children = node.num_children
            INT32 i, child_idx, best_idx = -1
            FLOAT32 parent_visits = <FLOAT32>node.visit_count
            FLOAT32 sqrt_parent = sqrt(parent_visits) if parent_visits > 0 else 1.0
            FLOAT32 q_value, u_value, ucb_value, best_ucb = -1e10
            NodeData* child
            
        if num_children == 0 or first_child < 0:
            return -1
            
        # Unrolled loop for common case (small number of children)
        if num_children <= 8:
            for i in range(num_children):
                child = &self.nodes[first_child + i]
                
                # Fast Q-value calculation
                q_value = child.value_sum / (<FLOAT32>child.visit_count + 1e-8)
                
                # Fast exploration term
                u_value = self.c_puct * child.prior * sqrt_parent / (1.0 + <FLOAT32>child.visit_count)
                
                ucb_value = q_value + u_value
                
                if ucb_value > best_ucb:
                    best_ucb = ucb_value
                    best_idx = first_child + i
        else:
            # Regular loop for many children
            for i in range(num_children):
                child_idx = first_child + i
                child = &self.nodes[child_idx]
                
                if child.visit_count > 0:
                    q_value = child.value_sum / <FLOAT32>child.visit_count
                else:
                    q_value = 0.0
                    
                u_value = self.c_puct * child.prior * sqrt_parent / (1.0 + <FLOAT32>child.visit_count)
                ucb_value = q_value + u_value
                
                if ucb_value > best_ucb:
                    best_ucb = ucb_value
                    best_idx = child_idx
                    
        return best_idx
        
    def select_best_child(self, INT32 node_idx):
        """Python-accessible selection"""
        return self._select_best_child_inline(node_idx)
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def select_batch_fast(self, INT32[:] node_indices, INT32[:] results):
        """Fast batch selection without parallel overhead"""
        cdef int i, n = node_indices.shape[0]
        
        # Simple loop is faster than parallel for most cases
        for i in range(n):
            results[i] = self._select_best_child_inline(node_indices[i])
            
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void _update_value_inline(self, INT32 node_idx, FLOAT32 value) nogil:
        """Inline value update"""
        self.nodes[node_idx].visit_count += 1
        self.nodes[node_idx].value_sum += value
        
    def update_value(self, INT32 node_idx, FLOAT32 value):
        """Python-accessible value update"""
        self._update_value_inline(node_idx, value)
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def backup_path_fast(self, INT32[:] path, FLOAT32 value):
        """Fast single path backup"""
        cdef:
            int i
            INT32 node_idx
            FLOAT32 current_value = value
            
        for i in range(path.shape[0]):
            node_idx = path[i]
            if node_idx < 0:
                break
            self._update_value_inline(node_idx, current_value)
            current_value = -current_value  # Flip for opponent
            
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def backup_paths_batch_fast(self, INT32[:, :] paths, FLOAT32[:] values):
        """Fast batch backup without parallel overhead"""
        cdef:
            int i, j
            int num_paths = paths.shape[0]
            int max_depth = paths.shape[1]
            INT32 node_idx
            FLOAT32 value
            
        for i in range(num_paths):
            value = values[i]
            for j in range(max_depth):
                node_idx = paths[i, j]
                if node_idx < 0:
                    break
                self._update_value_inline(node_idx, value)
                value = -value
                
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_children_arrays(
        self,
        INT32 node_idx,
        INT32[:] children_out,
        FLOAT32[:] priors_out,
        INT32[:] visits_out,
        FLOAT32[:] values_out
    ):
        """Get children info directly into pre-allocated arrays"""
        cdef:
            NodeData* node = &self.nodes[node_idx]
            INT32 first_child = node.first_child_idx
            INT32 num_children = node.num_children
            int i
            NodeData* child
            
        if num_children == 0 or first_child < 0:
            return 0
            
        for i in range(num_children):
            child = &self.nodes[first_child + i]
            children_out[i] = first_child + i
            priors_out[i] = child.prior
            visits_out[i] = child.visit_count
            values_out[i] = child.value_sum
            
        return num_children
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def calculate_ucb_batch_fast(
        self,
        INT32[:] children,
        FLOAT32 parent_visits,
        FLOAT32[:] ucb_out
    ):
        """Fast UCB calculation directly into output array"""
        cdef:
            int i, n = children.shape[0]
            FLOAT32 sqrt_parent = sqrt(parent_visits) if parent_visits > 0 else 1.0
            NodeData* child
            FLOAT32 q_value, u_value
            
        for i in range(n):
            child = &self.nodes[children[i]]
            
            # Q-value with small epsilon to avoid division by zero
            q_value = child.value_sum / (<FLOAT32>child.visit_count + 1e-8)
            
            # Exploration term
            u_value = self.c_puct * child.prior * sqrt_parent / (1.0 + <FLOAT32>child.visit_count)
            
            ucb_out[i] = q_value + u_value
            
    def get_stats(self):
        """Get tree statistics"""
        cdef:
            int i
            int total_visits = 0
            int expanded_nodes = 0
            int terminal_nodes = 0
            
        for i in range(self.num_nodes):
            total_visits += self.nodes[i].visit_count
            if self.nodes[i].is_expanded:
                expanded_nodes += 1
            if self.nodes[i].is_terminal:
                terminal_nodes += 1
                
        return {
            'num_nodes': self.num_nodes,
            'capacity': self.capacity,
            'total_visits': total_visits,
            'expanded_nodes': expanded_nodes,
            'terminal_nodes': terminal_nodes,
            'memory_usage_mb': (self.num_nodes * sizeof(NodeData)) / (1024.0 * 1024.0)
        }
        
    # Properties for direct access
    @property
    def node_count(self):
        return self.num_nodes
        
    @property
    def exploration_constant(self):
        return self.c_puct