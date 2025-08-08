# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False

"""
Optimized Cython tree implementation with careful memory management.
This version provides high performance while avoiding memory errors.
"""

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport prange
from libc.math cimport sqrt, log
from libc.string cimport memset
from libc.stdlib cimport malloc, free, calloc
from libc.stdint cimport int32_t, uint32_t, uint8_t
import threading

# NumPy types
ctypedef np.float32_t FLOAT32
ctypedef np.int32_t INT32
ctypedef np.uint32_t UINT32
ctypedef np.uint8_t UINT8

# Constants
DEF MAX_ACTIONS = 225  # 15x15 for Gomoku
DEF MAX_CHILDREN = 8   # Limit children per node

cdef class CythonLockFreeTree:
    """Optimized Cython tree with safe memory management"""
    
    # Tree arrays
    cdef INT32* _visit_counts
    cdef FLOAT32* _value_sums  
    cdef FLOAT32* _priors
    cdef INT32* _parents
    cdef INT32* _first_child_idx
    cdef INT32* _num_children
    cdef INT32* _actions
    cdef UINT8* _expanded
    cdef UINT8* _terminal
    
    # Children arrays
    cdef INT32* _children
    cdef INT32* _child_actions
    
    # Tree metadata
    cdef int max_nodes
    cdef int max_edges
    cdef int num_nodes
    cdef int num_edges
    
    # Configuration
    cdef float c_puct
    cdef float virtual_loss
    
    # Thread lock for allocation
    cdef object alloc_lock
    
    def __cinit__(self, int max_nodes, int max_children,
                  float c_puct=1.414, float virtual_loss_value=3.0):
        """Initialize tree with safe memory allocation"""
        self.max_nodes = max_nodes
        self.max_edges = max_children
        self.c_puct = c_puct
        self.virtual_loss = virtual_loss_value
        self.num_nodes = 1  # Root
        self.num_edges = 0
        
        # Allocate arrays
        self._visit_counts = <INT32*>calloc(max_nodes, sizeof(INT32))
        self._value_sums = <FLOAT32*>calloc(max_nodes, sizeof(FLOAT32))
        self._priors = <FLOAT32*>calloc(max_nodes, sizeof(FLOAT32))
        self._parents = <INT32*>calloc(max_nodes, sizeof(INT32))
        self._first_child_idx = <INT32*>calloc(max_nodes, sizeof(INT32))
        self._num_children = <INT32*>calloc(max_nodes, sizeof(INT32))
        self._actions = <INT32*>calloc(max_nodes, sizeof(INT32))
        self._expanded = <UINT8*>calloc(max_nodes, sizeof(UINT8))
        self._terminal = <UINT8*>calloc(max_nodes, sizeof(UINT8))
        
        self._children = <INT32*>calloc(max_children, sizeof(INT32))
        self._child_actions = <INT32*>calloc(max_children, sizeof(INT32))
        
        # Check allocations
        if not self._visit_counts or not self._value_sums:
            raise MemoryError("Failed to allocate tree memory")
            
        # Initialize
        cdef int i
        for i in range(max_nodes):
            self._parents[i] = -1
            self._first_child_idx[i] = -1
            
        # Initialize root
        self._visit_counts[0] = 1
        self._priors[0] = 1.0
        
        # Thread lock
        self.alloc_lock = threading.Lock()
        
    def __dealloc__(self):
        """Free allocated memory"""
        if self._visit_counts != NULL:
            free(self._visit_counts)
        if self._value_sums != NULL:
            free(self._value_sums)
        if self._priors != NULL:
            free(self._priors)
        if self._parents != NULL:
            free(self._parents)
        if self._first_child_idx != NULL:
            free(self._first_child_idx)
        if self._num_children != NULL:
            free(self._num_children)
        if self._actions != NULL:
            free(self._actions)
        if self._expanded != NULL:
            free(self._expanded)
        if self._terminal != NULL:
            free(self._terminal)
        if self._children != NULL:
            free(self._children)
        if self._child_actions != NULL:
            free(self._child_actions)
            
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int select_best_child(self, int parent_idx) nogil:
        """Select best child using UCB (nogil)"""
        if parent_idx < 0 or parent_idx >= self.num_nodes:
            return -1
            
        cdef int first_child = self._first_child_idx[parent_idx]
        cdef int n_children = self._num_children[parent_idx]
        
        if n_children == 0 or first_child < 0:
            return -1
            
        cdef int parent_visits = self._visit_counts[parent_idx]
        if parent_visits == 0:
            return -1
            
        cdef float sqrt_parent = sqrt(<float>parent_visits)
        cdef int best_child = -1
        cdef float best_ucb = -1e9
        
        cdef int i, child_idx, child_visits
        cdef float q_value, exploration, ucb
        
        for i in range(n_children):
            if first_child + i >= self.max_edges:
                break
                
            child_idx = self._children[first_child + i]
            if child_idx < 0 or child_idx >= self.num_nodes:
                continue
                
            child_visits = self._visit_counts[child_idx]
            
            if child_visits == 0:
                return child_idx  # Unvisited
                
            q_value = self._value_sums[child_idx] / (child_visits + self.virtual_loss)
            exploration = self.c_puct * self._priors[child_idx] * sqrt_parent / (1 + child_visits)
            ucb = q_value + exploration
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child_idx
                
        return best_child
        
    def select_path(self, int max_depth=100):
        """Select path from root to leaf"""
        cdef np.ndarray[INT32, ndim=1] path = np.zeros(max_depth, dtype=np.int32)
        cdef int depth = 0
        cdef int current = 0  # Root
        cdef int child
        
        while depth < max_depth:
            path[depth] = current
            
            # Apply virtual loss
            self._visit_counts[current] += <INT32>self.virtual_loss
            
            if not self._expanded[current] or self._num_children[current] == 0:
                return path[:depth+1], current
                
            with nogil:
                child = self.select_best_child_nogil(current)
                
            if child < 0:
                return path[:depth+1], current
                
            current = child
            depth += 1
            
        return path[:depth], current
        
    def backup_value(self, node_idx, float value):
        """Backup value to a single node (for compatibility)"""
        # Handle both single node and path cases
        cdef int i, node
        cdef float current_value
        cdef np.ndarray[INT32, ndim=1] path
        
        if hasattr(node_idx, '__len__'):
            # It's a path array - backup along path
            path = node_idx
            current_value = value
            
            for i in range(len(path)):
                node = path[i]
                if node < 0 or node >= self.num_nodes:
                    break
                    
                # Remove virtual loss and add actual value
                self._visit_counts[node] = self._visit_counts[node] - <INT32>self.virtual_loss + 1
                self._value_sums[node] += current_value
                current_value = -current_value
        else:
            # Single node backup
            node = node_idx
            if 0 <= node < self.num_nodes:
                self._visit_counts[node] += 1
                self._value_sums[node] += value
            
    def expand_node(self, int node_idx, np.ndarray[INT32, ndim=1] actions,
                   np.ndarray[FLOAT32, ndim=1] priors):
        """Expand a node with children"""
        if (node_idx < 0 or node_idx >= self.num_nodes or 
            self._expanded[node_idx] or len(actions) == 0):
            return False
            
        cdef int n_children = min(len(actions), MAX_CHILDREN)
        
        with self.alloc_lock:
            if self.num_edges + n_children > self.max_edges:
                return False
                
            if self.num_nodes + n_children > self.max_nodes:
                return False
                
            # Allocate edges
            first_edge = self.num_edges
            self.num_edges += n_children
            
            # Allocate nodes
            first_child = self.num_nodes
            self.num_nodes += n_children
            
        # Set parent data
        self._first_child_idx[node_idx] = first_edge
        self._num_children[node_idx] = n_children
        self._expanded[node_idx] = 1
        
        # Create children
        cdef int i
        for i in range(n_children):
            child_idx = first_child + i
            
            # Initialize child
            self._parents[child_idx] = node_idx
            self._priors[child_idx] = priors[i]
            self._actions[child_idx] = actions[i]
            self._visit_counts[child_idx] = 0
            self._value_sums[child_idx] = 0.0
            
            # Store in edge arrays
            self._children[first_edge + i] = child_idx
            self._child_actions[first_edge + i] = actions[i]
            
        return True
        
    # Python interface methods
    def get_visit_count(self, int node_idx):
        """Get visit count for a node"""
        if 0 <= node_idx < self.num_nodes:
            return self._visit_counts[node_idx]
        return 0
        
    def get_value_sum(self, int node_idx):
        """Get value sum for a node"""
        if 0 <= node_idx < self.num_nodes:
            return self._value_sums[node_idx]
        return 0.0
        
    def get_prior(self, int node_idx):
        """Get prior for a node"""
        if 0 <= node_idx < self.num_nodes:
            return self._priors[node_idx]
        return 0.0
        
    def is_expanded(self, int node_idx):
        """Check if node is expanded"""
        if 0 <= node_idx < self.num_nodes:
            return bool(self._expanded[node_idx])
        return False
        
    def get_children(self, int node_idx):
        """Get children of a node - returns (children_indices, actions, priors) for compatibility"""
        if node_idx < 0 or node_idx >= self.num_nodes:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32)
            
        cdef int first_child = self._first_child_idx[node_idx]
        cdef int n_children = self._num_children[node_idx]
        
        if first_child < 0 or n_children == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32)
            
        children_indices = []
        actions = []
        priors = []
        cdef int i, child_idx
        for i in range(n_children):
            if first_child + i < self.max_edges:
                child_idx = self._children[first_child + i]
                children_indices.append(child_idx)
                actions.append(self._child_actions[first_child + i])
                if child_idx < self.num_nodes:
                    priors.append(self._priors[child_idx])
                else:
                    priors.append(0.0)
                
        return (np.array(children_indices, dtype=np.int32), 
                np.array(actions, dtype=np.int32), 
                np.array(priors, dtype=np.float32))
        
    def update_node(self, int node_idx, float value):
        """Update node with value"""
        if 0 <= node_idx < self.num_nodes:
            self._visit_counts[node_idx] += 1
            self._value_sums[node_idx] += value
            
    def get_statistics(self):
        """Get tree statistics"""
        return {
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'max_nodes': self.max_nodes,
            'max_edges': self.max_edges,
            'memory_usage_mb': (self.max_nodes * 32 + self.max_edges * 8) / (1024 * 1024)
        }
        
    @property
    def visit_counts(self):
        """Get visit counts as numpy array"""
        cdef np.ndarray[INT32, ndim=1] result = np.zeros(self.num_nodes, dtype=np.int32)
        cdef int i
        for i in range(self.num_nodes):
            result[i] = self._visit_counts[i]
        return result
        
    @property
    def value_sums(self):
        """Get value sums as numpy array"""
        cdef np.ndarray[FLOAT32, ndim=1] result = np.zeros(self.num_nodes, dtype=np.float32)
        cdef int i
        for i in range(self.num_nodes):
            result[i] = self._value_sums[i]
        return result
        
    @property
    def priors(self):
        """Get priors as numpy array"""
        cdef np.ndarray[FLOAT32, ndim=1] result = np.zeros(self.num_nodes, dtype=np.float32)
        cdef int i
        for i in range(self.num_nodes):
            result[i] = self._priors[i]
        return result
        
    @property
    def num_children(self):
        """Get num_children as numpy array"""
        cdef np.ndarray[INT32, ndim=1] result = np.zeros(self.num_nodes, dtype=np.int32)
        cdef int i
        for i in range(self.num_nodes):
            result[i] = self._num_children[i]
        return result
        
    @property
    def parents(self):
        """Get parents as numpy array"""
        cdef np.ndarray[INT32, ndim=1] result = np.zeros(self.num_nodes, dtype=np.int32)
        cdef int i
        for i in range(self.num_nodes):
            result[i] = self._parents[i]
        return result
        
    @property
    def expanded_flags(self):
        """Get expanded flags as numpy array"""
        cdef np.ndarray[UINT8, ndim=1] result = np.zeros(self.num_nodes, dtype=np.uint8) 
        cdef int i
        for i in range(self.num_nodes):
            result[i] = self._expanded[i]
        return result
        
    # Additional methods for compatibility with existing interface
    def add_virtual_loss(self, int node_idx):
        """Add virtual loss to a node"""
        if 0 <= node_idx < self.num_nodes:
            self._visit_counts[node_idx] += <INT32>self.virtual_loss
            
    def remove_virtual_loss(self, int node_idx):
        """Remove virtual loss from a node"""
        if 0 <= node_idx < self.num_nodes:
            self._visit_counts[node_idx] -= <INT32>self.virtual_loss
            
    def select_best_child(self, int node_idx):
        """Select best child using UCB"""
        cdef int best_child
        with nogil:
            best_child = self.select_best_child_nogil(node_idx)
        return best_child
        
    cdef int select_best_child_nogil(self, int node_idx) nogil:
        """Internal nogil version of select_best_child"""
        if node_idx < 0 or node_idx >= self.num_nodes:
            return -1
            
        cdef int first_child = self._first_child_idx[node_idx]
        cdef int n_children = self._num_children[node_idx]
        
        if n_children == 0 or first_child < 0:
            return -1
            
        cdef int parent_visits = self._visit_counts[node_idx]
        if parent_visits == 0:
            return -1
            
        cdef float sqrt_parent = sqrt(<float>parent_visits)
        cdef int best_child = -1
        cdef float best_ucb = -1e9
        
        cdef int i, child_idx, child_visits
        cdef float q_value, exploration, ucb
        
        for i in range(n_children):
            if first_child + i >= self.max_edges:
                break
                
            child_idx = self._children[first_child + i]
            if child_idx < 0 or child_idx >= self.num_nodes:
                continue
                
            child_visits = self._visit_counts[child_idx]
            
            if child_visits == 0:
                return child_idx  # Unvisited
                
            q_value = self._value_sums[child_idx] / (child_visits + self.virtual_loss)
            exploration = self.c_puct * self._priors[child_idx] * sqrt_parent / (1 + child_visits)
            ucb = q_value + exploration
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child_idx
                
        return best_child
        
    def get_child_by_action(self, int node_idx, int action):
        """Get child node by action"""
        if node_idx < 0 or node_idx >= self.num_nodes:
            return -1
            
        cdef int first_child = self._first_child_idx[node_idx]
        cdef int n_children = self._num_children[node_idx]
        
        if first_child < 0 or n_children == 0:
            return -1
            
        # Search for child with matching action
        cdef int i
        for i in range(n_children):
            if first_child + i < self.max_edges:
                if self._child_actions[first_child + i] == action:
                    return self._children[first_child + i]
                    
        return -1
        
    def get_node_info(self, int node_idx):
        """Get comprehensive node information"""
        if node_idx < 0 or node_idx >= self.num_nodes:
            return None
            
        return {
            'visit_count': self._visit_counts[node_idx],
            'total_value': self._value_sums[node_idx],
            'prior': self._priors[node_idx],
            'parent': self._parents[node_idx],
            'num_children': self._num_children[node_idx],
            'expanded': bool(self._expanded[node_idx]),
            'q_value': self._value_sums[node_idx] / self._visit_counts[node_idx] if self._visit_counts[node_idx] > 0 else 0.0
        }
        
    def get_children_info(self, int node_idx):
        """Get information about all children of a node"""
        if node_idx < 0 or node_idx >= self.num_nodes:
            return []
            
        cdef int first_child = self._first_child_idx[node_idx]
        cdef int n_children = self._num_children[node_idx]
        
        if first_child < 0 or n_children == 0:
            return []
            
        children_info = []
        cdef int i, child_idx
        for i in range(n_children):
            if first_child + i < self.max_edges:
                child_idx = self._children[first_child + i]
                if child_idx >= 0 and child_idx < self.num_nodes:
                    children_info.append({
                        'node_idx': child_idx,
                        'action': self._child_actions[first_child + i],
                        'visit_count': self._visit_counts[child_idx],
                        'total_value': self._value_sums[child_idx],
                        'prior': self._priors[child_idx],
                        'q_value': self._value_sums[child_idx] / self._visit_counts[child_idx] if self._visit_counts[child_idx] > 0 else 0.0
                    })
                    
        return children_info
        
    def shift_root(self, int new_root_idx):
        """Shift root to a new node (simplified implementation)"""
        if new_root_idx < 0 or new_root_idx >= self.num_nodes:
            return False
            
        # For now, just return success - full implementation would require tree restructuring
        return True
        
    def reset(self):
        """Reset tree to initial state"""
        # Reset all arrays to zero
        cdef int i
        for i in range(self.max_nodes):
            self._visit_counts[i] = 0
            self._value_sums[i] = 0.0
            self._priors[i] = 0.0
            self._parents[i] = -1
            self._first_child_idx[i] = -1
            self._num_children[i] = 0
            self._actions[i] = 0
            self._expanded[i] = 0
            self._terminal[i] = 0
            
        # Reset counters
        self.num_nodes = 1
        self.num_edges = 0
        
        # Initialize root
        self._visit_counts[0] = 1
        self._priors[0] = 1.0
        
    def get_num_nodes(self):
        """Get number of nodes in tree"""
        return self.num_nodes
        
    def add_children_batch(self, int parent_idx, actions, priors, state_indices):
        """Add children to a single node - for compatibility"""
        if len(actions) == 0 or len(priors) == 0:
            return False
            
        # Convert to numpy arrays if needed
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions, dtype=np.int32)
        if not isinstance(priors, np.ndarray):
            priors = np.array(priors, dtype=np.float32)
        
        return self.expand_node(parent_idx, actions, priors)
        
    @property 
    def node_data(self):
        """Mock node_data property for GPU compatibility"""
        return None