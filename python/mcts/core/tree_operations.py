"""Tree operations for MCTS

This module contains tree-related operations extracted from the main MCTS class.
Includes operations for tree management, node manipulation, and subtree reuse.
"""

import torch
from typing import Dict, Optional, Tuple, Any

from ..gpu.csr_tree import CSRTree


class TreeOperations:
    """Handles tree-specific operations for MCTS"""
    
    def __init__(self, tree: CSRTree, config: Any, device: torch.device):
        """Initialize tree operations
        
        Args:
            tree: CSR tree structure
            config: MCTS configuration
            device: Torch device
        """
        self.tree = tree
        self.config = config
        self.device = device
        
        # Track last action for subtree reuse
        self.last_selected_action = None
        
    def clear(self):
        """Clear and reset the tree"""
        self.tree.reset()
        self.last_selected_action = None
            
    def reset_tree(self):
        """Reset tree to initial state"""
        self.clear()
        
    def apply_subtree_reuse(self, new_root_action: int) -> Optional[Dict[int, int]]:
        """Apply subtree reuse when updating root
        
        Args:
            new_root_action: Action taken from previous root
            
        Returns:
            Mapping of old node indices to new indices if reuse applied
        """
        if not self.config.enable_subtree_reuse:
            return None
            
        if new_root_action is None:
            return None
            
        # Find the child node corresponding to the action
        child_idx = self.tree.get_child_by_action(0, new_root_action)
        if child_idx is None:
            return None
            
        # Check if the subtree is worth preserving
        subtree_visits = self.tree.node_data.visit_counts[child_idx].item()
        if subtree_visits < self.config.subtree_reuse_min_visits:
            # Not worth preserving - just reset the tree
            self.tree.reset()
            return {}
            
        # Use shift_root to efficiently reuse the subtree
        mapping = self.tree.shift_root(child_idx)
        
        # NOTE: DO NOT clear the root's children after shift_root!
        # The whole point of tree reuse is to preserve the subtree structure.
        # The children are still valid - they represent the same game positions
        # relative to the new root state.
        
        # The shift_root operation has already:
        # 1. Made the target child the new root (index 0) 
        # 2. Preserved all its descendants with remapped indices
        # 3. Updated all parent-child relationships correctly
        # 4. Discarded nodes outside the subtree (siblings, old root)
        
        # The preserved children are still valid moves from the new root position
        
        # Important: If only the root was preserved (mapping size 1), 
        # ensure it's properly initialized for the next search
        if len(mapping) == 1:
            # The root needs to be marked as not expanded so it will be
            # expanded in the next search
            self.tree.node_data.set_expanded(0, False)
            # Also ensure the root has no children in the CSR structure
            # This is important for proper expansion detection
            if hasattr(self.tree, 'csr_storage') and hasattr(self.tree.csr_storage, 'row_ptr'):
                # Set row_ptr[0] = row_ptr[1] = 0 to indicate no edges from root
                self.tree.csr_storage.row_ptr[0] = 0
                self.tree.csr_storage.row_ptr[1] = 0
        
        return mapping
        
    def _extract_subtree(self, new_root_idx: int) -> Dict[int, int]:
        """Extract subtree rooted at given node using efficient shift_root
        
        Args:
            new_root_idx: Index of new root node
            
        Returns:
            Mapping from old node indices to new indices
        """
        # Use the efficient shift_root method from CSRTree
        # This preserves the entire subtree structure and remaps indices
        mapping = self.tree.shift_root(new_root_idx)
        
        return mapping
        

    def get_root_children_info(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get information about root node's children
        
        Returns:
            Tuple of (actions, visit_counts, value_averages)
        """
        # Use the get_children method which handles the complexity
        children_indices, actions, _ = self.tree.get_children(0)
        
        if len(children_indices) == 0:
            # No children
            return (torch.tensor([], dtype=torch.int32),
                    torch.tensor([], dtype=torch.int32), 
                    torch.tensor([], dtype=torch.float32))
                    
        visits = self.tree.node_data.visit_counts[children_indices]
        values = self.tree.node_data.value_sums[children_indices] / (visits + 1e-8)
        
        return actions, visits, values
        
    def get_best_child(self, node_idx: int, c_puct: float) -> Optional[int]:
        """Get best child of a node using UCB formula
        
        Args:
            node_idx: Parent node index
            c_puct: Exploration constant
            
        Returns:
            Index of best child or None if no children
        """
        # Check bounds to prevent IndexError
        if node_idx < 0 or node_idx >= len(self.tree.csr_storage.row_ptr):
            return None
            
        start = self.tree.csr_storage.row_ptr[node_idx].item()
        end = self.tree.csr_storage.row_ptr[node_idx + 1].item() if node_idx + 1 < self.tree.num_nodes else start
        
        if start == end:
            return None
            
        children = self.tree.csr_storage.col_indices[start:end]
        
        # Calculate UCB scores
        parent_visits = self.tree.node_data.visit_counts[node_idx]
        child_visits = self.tree.node_data.visit_counts[children]
        child_values = self.tree.node_data.value_sums[children] / (child_visits + 1e-8)
        child_priors = self.tree.node_data.node_priors[children]
        
        ucb_scores = child_values + c_puct * child_priors * torch.sqrt(parent_visits) / (1 + child_visits)
        
        best_idx = ucb_scores.argmax()
        return children[best_idx].item()
        
    def add_dirichlet_noise_to_root(self, alpha: float, epsilon: float):
        """Add Dirichlet noise to root node priors
        
        Args:
            alpha: Dirichlet alpha parameter
            epsilon: Mixing parameter
        """
        # Get root children using new API
        root_start = self.tree.csr_storage.row_ptr[0].item()
        root_end = self.tree.csr_storage.row_ptr[1].item() if self.tree.num_nodes > 1 else root_start
        
        if root_start == root_end:
            return
            
        num_children = root_end - root_start
        
        # Generate Dirichlet noise
        # Create a new distribution each time to ensure different samples
        dirichlet_dist = torch.distributions.Dirichlet(
            torch.full((num_children,), alpha, device=self.device)
        )
        noise = dirichlet_dist.sample()
        
        # Get child indices and their priors
        child_indices = self.tree.csr_storage.col_indices[root_start:root_end]
        
        # Mix with original priors from node data
        original_priors = self.tree.node_data.node_priors[child_indices].clone()
        noisy_priors = (1 - epsilon) * original_priors + epsilon * noise
        
        # Update priors in node data
        self.tree.node_data.node_priors[child_indices] = noisy_priors
        
    def get_tree_statistics(self) -> Dict[str, Any]:
        """Get tree statistics
        
        Returns:
            Dictionary of tree statistics
        """
        stats = {
            'num_nodes': self.tree.num_nodes,
            'max_depth': self._calculate_max_depth(),
            'root_visits': self.tree.node_data.visit_counts[0].item(),
            'root_children': self._count_root_children(),
            'memory_usage_mb': self._estimate_memory_usage() / (1024 * 1024)
        }
        
        return stats
        
    def _calculate_max_depth(self) -> int:
        """Calculate maximum depth of tree"""
        if self.tree.num_nodes <= 1:
            return 0
            
        depths = torch.zeros(self.tree.num_nodes, dtype=torch.int32, device=self.device)
        
        # BFS to calculate depths
        for node in range(1, self.tree.num_nodes):
            parent = self.tree.node_data.parent_indices[node].item()
            if parent >= 0:
                depths[node] = depths[parent] + 1
                
        return depths.max().item()
        
    def _count_root_children(self) -> int:
        """Count number of root children"""
        # Simply use get_children which handles this correctly
        children, _, _ = self.tree.get_children(0)
        count = len(children)
        
        # Debug
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"_count_root_children: children={children}, count={count}")
        
        return count
        
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of tree in bytes"""
        # Get memory usage from components
        node_data_mb = self.tree.node_data.get_memory_usage_mb()
        csr_storage_mb = self.tree.csr_storage.get_memory_usage_mb()
        
        # Convert MB to bytes
        total_bytes = (node_data_mb + csr_storage_mb) * 1024 * 1024
        return float(total_bytes)