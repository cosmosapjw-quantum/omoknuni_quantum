# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: profile=False

"""
Lock-free Cython tree implementation for high-performance CPU MCTS.

This implementation uses:
- Array-based tree structure for cache efficiency
- Lock-free atomic operations for thread safety
- Memory pool allocation to minimize overhead
- SIMD-friendly data layout
- Optimized UCB calculations
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, log
from libc.string cimport memset
from libc.stdlib cimport malloc, free
from libcpp.atomic cimport atomic
from cython.parallel import prange

# NumPy types
ctypedef np.float32_t FLOAT32
ctypedef np.int32_t INT32
ctypedef np.uint32_t UINT32
ctypedef np.int8_t INT8

# Constants
DEF MAX_ACTIONS = 512
DEF CACHE_LINE_SIZE = 64

# Node data structure - aligned to cache line
cdef struct Node:
    atomic[int] visit_count      # Atomic visit counter
    atomic[float] total_value    # Atomic value accumulator
    float prior                  # Prior probability
    int parent                   # Parent node index
    int first_child             # First child index in children array
    int num_children            # Number of children
    int action                  # Action that led to this node
    char expanded               # Expansion flag
    char terminal               # Terminal flag
    char _padding[CACHE_LINE_SIZE - 38]  # Pad to cache line

# Edge structure for children
cdef struct Edge:
    int child_idx
    int action

cdef class CythonLockFreeTree:
    """Lock-free tree implementation using atomic operations"""
    
    # Tree data
    cdef Node* nodes                    # Node array
    cdef Edge* edges                    # Edge array
    cdef atomic[int] num_nodes          # Current number of nodes
    cdef atomic[int] num_edges          # Current number of edges
    cdef int max_nodes                  # Maximum nodes
    cdef int max_edges                  # Maximum edges
    
    # UCB parameters
    cdef float c_puct
    cdef float virtual_loss_value
    
    # Memory pools
    cdef int* node_pool                 # Pool of available node indices
    cdef atomic[int] pool_head          # Head of free list
    
    # Statistics
    cdef atomic[long] total_visits      # Total visits across tree
    
    def __cinit__(self, int max_nodes, int max_children, 
                  float c_puct=1.414, float virtual_loss_value=3.0):
        """Initialize lock-free tree"""
        self.max_nodes = max_nodes
        self.max_edges = max_children
        self.c_puct = c_puct
        self.virtual_loss_value = virtual_loss_value
        
        # Allocate aligned memory for nodes
        self.nodes = <Node*>malloc(max_nodes * sizeof(Node))
        if self.nodes == NULL:
            raise MemoryError("Failed to allocate node memory")
        memset(self.nodes, 0, max_nodes * sizeof(Node))
        
        # Allocate edge memory
        self.edges = <Edge*>malloc(max_children * sizeof(Edge))
        if self.edges == NULL:
            free(self.nodes)
            raise MemoryError("Failed to allocate edge memory")
        memset(self.edges, 0, max_children * sizeof(Edge))
        
        # Initialize node pool
        self.node_pool = <int*>malloc(max_nodes * sizeof(int))
        if self.node_pool == NULL:
            free(self.nodes)
            free(self.edges)
            raise MemoryError("Failed to allocate node pool")
        
        # Fill pool with available indices (skip root)
        cdef int i
        for i in range(1, max_nodes):
            self.node_pool[i-1] = i
        self.pool_head.store(max_nodes - 1)
        
        # Initialize root node
        self.num_nodes.store(1)
        self.num_edges.store(0)
        self.total_visits.store(0)
        
        # Set root node properties
        self.nodes[0].visit_count.store(0)
        self.nodes[0].total_value.store(0.0)
        self.nodes[0].prior = 1.0
        self.nodes[0].parent = -1
        self.nodes[0].first_child = -1
        self.nodes[0].num_children = 0
        self.nodes[0].action = -1
        self.nodes[0].expanded = 0
        self.nodes[0].terminal = 0
    
    def __dealloc__(self):
        """Clean up allocated memory"""
        if self.nodes != NULL:
            free(self.nodes)
        if self.edges != NULL:
            free(self.edges)
        if self.node_pool != NULL:
            free(self.node_pool)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int allocate_node(self) nogil:
        """Allocate a new node from the pool (lock-free)"""
        cdef int old_head, new_head, node_idx
        
        while True:
            old_head = self.pool_head.load()
            if old_head < 0:
                return -1  # Pool exhausted
            
            node_idx = self.node_pool[old_head]
            new_head = old_head - 1
            
            # Try to update head atomically
            if self.pool_head.compare_exchange_strong(old_head, new_head):
                self.num_nodes.fetch_add(1)
                return node_idx
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void release_node(self, int node_idx) nogil:
        """Return node to pool (lock-free)"""
        cdef int old_head, new_head
        
        # Clear node data
        self.nodes[node_idx].visit_count.store(0)
        self.nodes[node_idx].total_value.store(0.0)
        self.nodes[node_idx].expanded = 0
        self.nodes[node_idx].terminal = 0
        
        # Return to pool
        while True:
            old_head = self.pool_head.load()
            new_head = old_head + 1
            if new_head >= self.max_nodes:
                break  # Pool full
            
            self.node_pool[new_head] = node_idx
            if self.pool_head.compare_exchange_strong(old_head, new_head):
                self.num_nodes.fetch_sub(1)
                break
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef float compute_ucb(self, int node_idx, int parent_visits) nogil:
        """Compute UCB value for a node"""
        cdef int visits = self.nodes[node_idx].visit_count.load()
        cdef float value = self.nodes[node_idx].total_value.load()
        cdef float prior = self.nodes[node_idx].prior
        
        if visits == 0:
            return prior * self.c_puct * sqrt(<float>parent_visits)
        
        cdef float avg_value = value / visits
        cdef float exploration = self.c_puct * prior * sqrt(<float>parent_visits) / (1 + visits)
        
        return avg_value + exploration
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def select_best_child(self, int node_idx):
        """Select child with highest UCB value"""
        cdef int parent_visits = self.nodes[node_idx].visit_count.load()
        cdef int first_child = self.nodes[node_idx].first_child
        cdef int num_children = self.nodes[node_idx].num_children
        
        if num_children == 0:
            return -1
        
        cdef int best_child = -1
        cdef float best_ucb = -1e9
        cdef float ucb
        cdef int i, child_idx
        
        for i in range(num_children):
            child_idx = self.edges[first_child + i].child_idx
            ucb = self.compute_ucb(child_idx, parent_visits)
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child_idx
        
        return best_child
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def expand_node(self, int parent_idx, np.ndarray[FLOAT32, ndim=1] priors, 
                    np.ndarray[INT32, ndim=1] actions):
        """Expand a node with children"""
        cdef int num_actions = actions.shape[0]
        if num_actions == 0:
            return
        
        # Allocate edge space
        cdef int edge_start = self.num_edges.fetch_add(num_actions)
        if edge_start + num_actions > self.max_edges:
            self.num_edges.fetch_sub(num_actions)
            raise MemoryError("Edge pool exhausted")
        
        # Update parent
        self.nodes[parent_idx].first_child = edge_start
        self.nodes[parent_idx].num_children = num_actions
        self.nodes[parent_idx].expanded = 1
        
        # Create children
        cdef int i, child_idx
        for i in range(num_actions):
            child_idx = self.allocate_node()
            if child_idx < 0:
                # Rollback on failure
                self.nodes[parent_idx].num_children = i
                break
            
            # Initialize child
            self.nodes[child_idx].parent = parent_idx
            self.nodes[child_idx].action = actions[i]
            self.nodes[child_idx].prior = priors[i]
            self.nodes[child_idx].first_child = -1
            self.nodes[child_idx].num_children = 0
            
            # Add edge
            self.edges[edge_start + i].child_idx = child_idx
            self.edges[edge_start + i].action = actions[i]
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_virtual_loss(self, int node_idx):
        """Add virtual loss for parallel MCTS"""
        self.nodes[node_idx].visit_count.fetch_add(1)
        self.nodes[node_idx].total_value.fetch_sub(self.virtual_loss_value)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def remove_virtual_loss(self, int node_idx):
        """Remove virtual loss after evaluation"""
        self.nodes[node_idx].visit_count.fetch_sub(1)
        self.nodes[node_idx].total_value.fetch_add(self.virtual_loss_value)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def backup_value(self, int node_idx, float value):
        """Backup value from leaf to root"""
        cdef int current = node_idx
        cdef float backup_value = value
        
        while current >= 0:
            # Update statistics atomically
            self.nodes[current].visit_count.fetch_add(1)
            self.nodes[current].total_value.fetch_add(backup_value)
            self.total_visits.fetch_add(1)
            
            # Flip value for opponent
            backup_value = -backup_value
            
            # Move to parent
            current = self.nodes[current].parent
    
    def get_child_by_action(self, int node_idx, int action):
        """Get child node by action"""
        cdef int first_child = self.nodes[node_idx].first_child
        cdef int num_children = self.nodes[node_idx].num_children
        cdef int i
        
        for i in range(num_children):
            if self.edges[first_child + i].action == action:
                return self.edges[first_child + i].child_idx
        
        return -1
    
    def get_node_info(self, int node_idx):
        """Get node information as dictionary"""
        if node_idx < 0 or node_idx >= self.num_nodes.load():
            return None
        
        return {
            'visit_count': self.nodes[node_idx].visit_count.load(),
            'total_value': self.nodes[node_idx].total_value.load(),
            'prior': self.nodes[node_idx].prior,
            'parent': self.nodes[node_idx].parent,
            'action': self.nodes[node_idx].action,
            'num_children': self.nodes[node_idx].num_children,
            'expanded': bool(self.nodes[node_idx].expanded),
            'terminal': bool(self.nodes[node_idx].terminal)
        }
    
    def get_children_info(self, int node_idx):
        """Get information about all children"""
        cdef int first_child = self.nodes[node_idx].first_child
        cdef int num_children = self.nodes[node_idx].num_children
        
        children = []
        cdef int i, child_idx
        for i in range(num_children):
            child_idx = self.edges[first_child + i].child_idx
            action = self.edges[first_child + i].action
            visits = self.nodes[child_idx].visit_count.load()
            value = self.nodes[child_idx].total_value.load()
            
            children.append({
                'child_idx': child_idx,
                'action': action,
                'visits': visits,
                'value': value,
                'prior': self.nodes[child_idx].prior
            })
        
        return children
    
    def reset(self):
        """Reset tree to initial state"""
        # Clear all nodes except root
        cdef int i
        for i in range(1, self.num_nodes.load()):
            self.release_node(i)
        
        # Reset root
        self.nodes[0].visit_count.store(0)
        self.nodes[0].total_value.store(0.0)
        self.nodes[0].first_child = -1
        self.nodes[0].num_children = 0
        self.nodes[0].expanded = 0
        self.nodes[0].terminal = 0
        
        # Reset counters
        self.num_nodes.store(1)
        self.num_edges.store(0)
        self.total_visits.store(0)
    
    # Properties for compatibility
    @property
    def node_data(self):
        """Compatibility property"""
        return self
    
    def get_num_nodes(self):
        """Get current number of nodes"""
        return self.num_nodes.load()
    
    def get_visit_count(self, int node_idx):
        """Get visit count for a node"""
        return self.nodes[node_idx].visit_count.load()
    
    def shift_root(self, int new_root_idx):
        """Shift root to a new node (for tree reuse)"""
        # This is a simplified version - full implementation would
        # need to properly handle memory management and subtree extraction
        if new_root_idx <= 0 or new_root_idx >= self.num_nodes.load():
            return False
        
        # For now, just reset the tree
        self.reset()
        return True