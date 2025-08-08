
class OptimizedCythonTree:
    """Optimized wrapper with minimal overhead"""
    
    def __init__(self, config):
        from mcts.cpu.cython_tree_optimized import CythonLockFreeTree
        
        # Store config values
        self._max_nodes = config.max_nodes
        self._max_children = getattr(config, 'max_children', 2000000)
        
        # Create tree directly - no logging overhead
        self._tree = CythonLockFreeTree(
            max_nodes=self._max_nodes,
            max_children=self._max_children,
            c_puct=getattr(config, 'c_puct', 1.414),
            virtual_loss_value=getattr(config, 'virtual_loss', 3.0)
        )
    
    # Direct property forwarding (fastest possible)
    @property 
    def num_nodes(self):
        return self._tree.get_num_nodes()
    
    @property
    def max_nodes(self):
        return self._max_nodes
        
    @property
    def node_data(self):
        return None
    
    # Direct method forwarding without any wrapper logic
    def select_best_child(self, node_idx):
        return self._tree.select_best_child(node_idx)
    
    def get_children(self, node_idx):
        return self._tree.get_children(node_idx)
        
    def get_visit_count(self, node_idx):
        return self._tree.get_visit_count(node_idx)
        
    def add_virtual_loss(self, node_idx):
        self._tree.add_virtual_loss(node_idx)
        
    def remove_virtual_loss(self, node_idx):
        self._tree.remove_virtual_loss(node_idx)
        
    def expand_node(self, node_idx, actions, priors):
        return self._tree.expand_node(node_idx, actions, priors)
        
    def backup_value(self, node_idx, value):
        self._tree.backup_value(node_idx, value)
        
    def add_children_batch(self, parent_idx, actions, priors, state_indices):
        return self._tree.add_children_batch(parent_idx, actions, priors, state_indices)
        
    def get_child_by_action(self, node_idx, action):
        return self._tree.get_child_by_action(node_idx, action)
        
    def shift_root(self, new_root_idx):
        """Shift root to a new node, preserving subtree
        
        Returns:
            Dict mapping old node indices to new indices
        """
        success = self._tree.shift_root(new_root_idx)
        if success:
            # For CPU backend, we don't track the exact mapping
            # Return a simple mapping indicating success
            # The calling code just needs to know if reuse worked
            return {new_root_idx: 0}  # Old root index -> new root (0)
        return None
    
    def get_value_sum(self, node_idx):
        return self._tree.get_value_sum(node_idx)
    
    def reset(self):
        return self._tree.reset()
        
    def get_stats(self):
        stats = self._tree.get_statistics()
        stats.update({
            'num_nodes': self._tree.get_num_nodes(),
            'max_nodes': self._max_nodes,
            'max_children': self._max_children,
            'root_visits': self._tree.get_visit_count(0) if self._tree.get_num_nodes() > 0 else 0,
        })
        return stats
    
    # Properties for compatibility with GPU backend
    @property
    def visit_counts(self):
        """Return visit counts for all nodes (for tree reuse compatibility)"""
        # Always return an object, even if tree is empty
        class VisitCounts:
            def __init__(self, tree):
                self.tree = tree
            def __getitem__(self, idx):
                if self.tree.get_num_nodes() <= idx:
                    return 0
                return self.tree.get_visit_count(idx)
            def item(self):
                if self.tree.get_num_nodes() == 0:
                    return 0
                return self.tree.get_visit_count(0)  # For root
            def fill(self, value):
                # For reset operation
                pass
        return VisitCounts(self._tree)
    
    @property
    def value_sums(self):
        """Return value sums for all nodes (for tree reuse compatibility)"""
        # Always return an object, even if tree is empty
        class ValueSums:
            def __init__(self, tree):
                self.tree = tree
            def __getitem__(self, idx):
                if self.tree.get_num_nodes() <= idx:
                    return 0.0
                return self.tree.get_value_sum(idx)
            def fill(self, value):
                # For reset operation
                pass
        return ValueSums(self._tree)
    
    # Mock properties for GPU compatibility  
    @property
    def children(self):
        class MockChildren:
            def __setitem__(self, key, value): pass
        return MockChildren()
    
    @property
    def csr_storage(self):
        class MockCSR:
            def __init__(self, tree):
                self.tree = tree
            @property 
            def row_ptr(self):
                return [0, 0]
            def get_memory_usage_mb(self):
                return 0.0
        return MockCSR(self._tree)
