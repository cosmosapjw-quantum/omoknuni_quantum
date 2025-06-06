"""GPU-optimized tree operation kernels

High-performance GPU kernels specifically designed for MCTS tree operations,
including parallel node selection, expansion, and backup.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List
import logging

from .cuda_kernels import CUDAKernels, CUDA_AVAILABLE

logger = logging.getLogger(__name__)


class GPUTreeKernels:
    """GPU kernels optimized for tree operations"""
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize GPU tree kernels
        
        Args:
            device: Device to use (defaults to CUDA if available)
        """
        self.device = device or torch.device('cuda' if CUDA_AVAILABLE else 'cpu')
        self.cuda_kernels = CUDAKernels(self.device)
        
    def parallel_select_children(
        self,
        node_indices: torch.Tensor,
        children_tensor: torch.Tensor,
        q_values: torch.Tensor,
        visit_counts: torch.Tensor,
        priors: torch.Tensor,
        c_puct: float = 1.0,
        valid_actions_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Select best children for multiple nodes in parallel using UCB
        
        Args:
            node_indices: Indices of parent nodes (batch_size,)
            children_tensor: Children for all nodes (n_nodes, max_children)
            q_values: Q-values for all nodes (n_nodes,)
            visit_counts: Visit counts for all nodes (n_nodes,)
            priors: Prior probabilities (n_nodes, max_children)
            c_puct: PUCT constant
            valid_actions_mask: Mask for valid actions (batch_size, max_children)
            
        Returns:
            Selected child indices (batch_size,)
        """
        batch_size = node_indices.shape[0]
        max_children = children_tensor.shape[1]
        
        # Move to device
        node_indices = node_indices.to(self.device)
        children_tensor = children_tensor.to(self.device)
        q_values = q_values.to(self.device)
        visit_counts = visit_counts.to(self.device)
        priors = priors.to(self.device)
        
        # Gather children for each node
        node_children = children_tensor[node_indices]  # (batch_size, max_children)
        
        # Gather statistics for children
        # Use -1 as invalid index placeholder
        valid_mask = node_children >= 0
        
        # Clamp indices to valid range
        n_nodes = q_values.shape[0]
        safe_indices = torch.clamp(node_children, min=0, max=n_nodes-1)
        
        child_q = q_values[safe_indices]
        child_visits = visit_counts[safe_indices]
        
        # For priors, we need to ensure we're gathering from the right dimensions
        # priors shape: (n_nodes, max_children)
        # We want to gather priors for the children of each parent node
        child_indices = torch.arange(max_children, device=self.device).unsqueeze(0).expand(batch_size, -1)
        child_priors = torch.gather(priors[node_indices], 1, child_indices)
        
        # Parent visits for UCB calculation
        parent_visits = visit_counts[node_indices].unsqueeze(1).expand(-1, max_children)
        
        # Compute UCB scores for all children
        ucb_scores = self._compute_ucb_2d(
            child_q, child_visits, parent_visits, child_priors, c_puct
        )
        
        # Apply masks
        if valid_actions_mask is not None:
            valid_mask = valid_mask & valid_actions_mask.to(self.device)
            
        # Set invalid actions to -inf
        ucb_scores = torch.where(valid_mask, ucb_scores, float('-inf'))
        
        # Select best children
        best_child_indices = ucb_scores.argmax(dim=1)
        
        # Get actual child node indices
        selected_children = torch.gather(node_children, 1, 
                                       best_child_indices.unsqueeze(1)).squeeze(1)
        
        return selected_children
        
    def vectorized_expand_nodes(
        self,
        leaf_nodes: torch.Tensor,
        game_states: torch.Tensor,
        policy_logits: torch.Tensor,
        values: torch.Tensor,
        tree_capacity: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand multiple leaf nodes in parallel
        
        Args:
            leaf_nodes: Indices of leaf nodes to expand (batch_size,)
            game_states: Game states for leaves (batch_size, state_shape...)
            policy_logits: Policy network outputs (batch_size, n_actions)
            values: Value network outputs (batch_size,)
            tree_capacity: Maximum number of nodes in tree
            
        Returns:
            Tuple of:
                - New node indices (batch_size, n_actions)
                - Prior probabilities (batch_size, n_actions)
                - Initial Q-values (batch_size, n_actions)
        """
        batch_size, n_actions = policy_logits.shape
        
        # Move to device
        leaf_nodes = leaf_nodes.to(self.device)
        policy_logits = policy_logits.to(self.device)
        values = values.to(self.device)
        
        # Convert logits to probabilities
        priors = F.softmax(policy_logits, dim=-1)
        
        # Allocate new nodes
        # In practice, this would interact with tree memory management
        # For now, simulate with sequential allocation
        first_new_node = tree_capacity  # Simplified - would track actual capacity
        
        new_node_indices = torch.arange(
            first_new_node, 
            first_new_node + batch_size * n_actions,
            device=self.device
        ).reshape(batch_size, n_actions)
        
        # Initialize Q-values with parent value (negated for opponent)
        initial_q = -values.unsqueeze(1).expand(-1, n_actions)
        
        return new_node_indices, priors, initial_q
        
    def parallel_backup_paths(
        self,
        paths: torch.Tensor,
        values: torch.Tensor,
        visit_counts: torch.Tensor,
        value_sums: torch.Tensor,
        max_depth: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backup values along multiple paths in parallel
        
        Args:
            paths: Path tensors (batch_size, max_depth)
            values: Leaf values (batch_size,)
            visit_counts: Current visit counts (n_nodes,)
            value_sums: Current value sums (n_nodes,)
            max_depth: Maximum path depth
            
        Returns:
            Updated (visit_counts, value_sums)
        """
        batch_size = paths.shape[0]
        n_nodes = visit_counts.shape[0]
        
        # Move to device
        paths = paths.to(self.device)
        values = values.to(self.device)
        visit_counts = visit_counts.to(self.device).clone()
        value_sums = value_sums.to(self.device).clone()
        
        # Create update tensors
        visit_updates = torch.zeros(n_nodes, device=self.device)
        value_updates = torch.zeros(n_nodes, device=self.device)
        
        # Process each depth level in parallel
        for depth in range(max_depth):
            # Get nodes at this depth
            nodes_at_depth = paths[:, depth]
            
            # Mask for valid nodes
            valid_mask = nodes_at_depth >= 0
            if not valid_mask.any():
                break
                
            valid_nodes = nodes_at_depth[valid_mask]
            valid_values = values[valid_mask]
            
            # Accumulate updates using scatter_add
            visit_updates.scatter_add_(0, valid_nodes, 
                                      torch.ones_like(valid_nodes, dtype=torch.float))
            value_updates.scatter_add_(0, valid_nodes, valid_values)
            
            # Negate values for next level (minimax)
            values = -values
            
        # Apply updates
        visit_counts += visit_updates
        value_sums += value_updates
        
        return visit_counts, value_sums
        
    def compute_subtree_statistics(
        self,
        root_nodes: torch.Tensor,
        children_tensor: torch.Tensor,
        visit_counts: torch.Tensor,
        value_sums: torch.Tensor,
        max_depth: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute statistics for subtrees rooted at given nodes
        
        Args:
            root_nodes: Root nodes of subtrees (batch_size,)
            children_tensor: Children structure (n_nodes, max_children)
            visit_counts: Visit counts (n_nodes,)
            value_sums: Value sums (n_nodes,)
            max_depth: Maximum depth to explore
            
        Returns:
            Tuple of:
                - Total visits in subtree (batch_size,)
                - Average values in subtree (batch_size,) 
                - Number of nodes in subtree (batch_size,)
        """
        batch_size = root_nodes.shape[0]
        n_nodes = children_tensor.shape[0]
        
        # Move to device
        root_nodes = root_nodes.to(self.device)
        children_tensor = children_tensor.to(self.device)
        visit_counts = visit_counts.to(self.device)
        value_sums = value_sums.to(self.device)
        
        # Initialize statistics
        total_visits = torch.zeros(batch_size, device=self.device)
        total_values = torch.zeros(batch_size, device=self.device)
        node_counts = torch.zeros(batch_size, device=self.device)
        
        # BFS to explore subtrees
        current_level = root_nodes.unsqueeze(1)  # (batch_size, 1)
        
        for depth in range(max_depth):
            # Mask for valid nodes
            valid_mask = current_level >= 0
            
            # Accumulate statistics for current level
            for i in range(batch_size):
                valid_nodes = current_level[i][valid_mask[i]]
                if valid_nodes.numel() == 0:
                    continue
                    
                total_visits[i] += visit_counts[valid_nodes].sum()
                total_values[i] += value_sums[valid_nodes].sum()
                node_counts[i] += valid_nodes.numel()
                
            # Get children for next level
            # This is simplified - in practice would handle variable branching
            if depth < max_depth - 1:
                next_level = []
                for i in range(batch_size):
                    level_children = []
                    for node in current_level[i]:
                        if node >= 0:
                            node_children = children_tensor[node]
                            level_children.extend(node_children[node_children >= 0].tolist())
                    
                    # Pad to fixed size
                    max_level_size = 100  # Reasonable limit
                    if len(level_children) > max_level_size:
                        level_children = level_children[:max_level_size]
                    while len(level_children) < max_level_size:
                        level_children.append(-1)
                        
                    next_level.append(torch.tensor(level_children, device=self.device))
                    
                current_level = torch.stack(next_level)
                
        # Compute averages
        avg_values = torch.where(
            total_visits > 0,
            total_values / total_visits,
            torch.zeros_like(total_values)
        )
        
        return total_visits, avg_values, node_counts
        
    def _compute_ucb_2d(
        self,
        q_values: torch.Tensor,
        visit_counts: torch.Tensor,
        parent_visits: torch.Tensor,
        priors: torch.Tensor,
        c_puct: float
    ) -> torch.Tensor:
        """Compute UCB for 2D tensors (batch_size, n_children)"""
        sqrt_parent = torch.sqrt(parent_visits)
        exploration = c_puct * priors * sqrt_parent / (1.0 + visit_counts)
        return q_values + exploration
        
    def find_principal_variations(
        self,
        root_nodes: torch.Tensor,
        children_tensor: torch.Tensor,
        visit_counts: torch.Tensor,
        max_depth: int = 10
    ) -> torch.Tensor:
        """Find principal variations (most visited paths) from root nodes
        
        Args:
            root_nodes: Starting nodes (batch_size,)
            children_tensor: Children structure (n_nodes, max_children)
            visit_counts: Visit counts (n_nodes,)
            max_depth: Maximum PV depth
            
        Returns:
            Principal variations (batch_size, max_depth)
        """
        batch_size = root_nodes.shape[0]
        
        # Move to device
        root_nodes = root_nodes.to(self.device)
        children_tensor = children_tensor.to(self.device)
        visit_counts = visit_counts.to(self.device)
        
        # Initialize PV storage
        pv = torch.full((batch_size, max_depth), -1, 
                       dtype=torch.long, device=self.device)
        
        # Start with root nodes
        current_nodes = root_nodes.clone()
        
        for depth in range(max_depth):
            # Store current nodes in PV
            pv[:, depth] = current_nodes
            
            # Get children for current nodes
            node_children = children_tensor[current_nodes]
            
            # Get visit counts for children
            valid_mask = node_children >= 0
            safe_indices = torch.where(valid_mask, node_children, 0)
            child_visits = visit_counts[safe_indices]
            
            # Mask out invalid children
            child_visits = torch.where(valid_mask, child_visits, -1.0)
            
            # Select most visited children
            best_children_idx = child_visits.argmax(dim=1)
            
            # Get actual child nodes
            next_nodes = torch.gather(node_children, 1, 
                                    best_children_idx.unsqueeze(1)).squeeze(1)
            
            # Check for terminal nodes
            terminal_mask = next_nodes < 0
            if terminal_mask.all():
                break
                
            current_nodes = next_nodes
            
        return pv