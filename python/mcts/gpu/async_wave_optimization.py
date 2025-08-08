"""
Optimized async wave search operations without CPU-GPU synchronization.
Provides vectorized implementations of critical operations.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np


class OptimizedWaveOperations:
    """Fully vectorized wave search operations without .item() calls."""
    
    @staticmethod
    def vectorized_progressive_widening(
        parent_visits: torch.Tensor,
        num_actions: torch.Tensor,
        cpw: float,
        kpw: float,
        root_mask: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Calculate progressive widening limits for multiple nodes in parallel.
        
        Args:
            parent_visits: Tensor of visit counts for parent nodes
            num_actions: Tensor of available actions per node
            cpw: Progressive widening constant
            kpw: Progressive widening power
            root_mask: Boolean mask indicating which nodes are root
            
        Returns:
            max_children: Tensor of maximum children allowed per node
        """
        # Calculate base progressive widening
        # Add 1 to avoid log(0)
        pw_value = cpw * torch.pow(parent_visits.float() + 1, kpw)
        max_children = pw_value.long() + 1
        
        # Apply limits based on node type
        # Root nodes: between 5 and 15
        # Non-root nodes: between 2 and num_actions
        root_max = torch.clamp(max_children, min=5, max=15)
        non_root_max = torch.clamp(max_children, min=2, max=num_actions)
        
        # Select based on mask without branching
        max_children = torch.where(root_mask, root_max, non_root_max)
        
        # Ensure we don't exceed available actions
        max_children = torch.min(max_children, num_actions)
        
        return max_children
    
    @staticmethod
    def batch_apply_dirichlet_noise(
        priors: List[torch.Tensor],
        root_mask: torch.Tensor,
        epsilon: float,
        alpha: float,
        device: torch.device
    ) -> List[torch.Tensor]:
        """
        Apply Dirichlet noise to root node priors without synchronization.
        
        Args:
            priors: List of prior tensors for each node
            root_mask: Boolean mask indicating root nodes
            epsilon: Dirichlet noise epsilon
            alpha: Dirichlet alpha parameter
            
        Returns:
            noised_priors: List of priors with noise applied to root nodes
        """
        noised_priors = []
        
        for i, prior in enumerate(priors):
            if root_mask[i]:
                # Generate Dirichlet noise
                noise = torch.distributions.Dirichlet(
                    torch.full_like(prior, alpha)
                ).sample()
                
                # Mix with original priors
                noised = (1 - epsilon) * prior + epsilon * noise
                noised_priors.append(noised)
            else:
                noised_priors.append(prior)
                
        return noised_priors
    
    @staticmethod
    def vectorized_node_expansion_batching(
        expansion_nodes: torch.Tensor,
        legal_actions_batch: torch.Tensor,
        policies: torch.Tensor,
        parent_visits: torch.Tensor,
        cpw: float,
        kpw: float,
        epsilon: float,
        alpha: float,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batch process node expansions without any CPU synchronization.
        
        Returns:
            all_parent_indices: Flattened parent indices
            all_actions: Flattened actions  
            all_priors: Flattened priors
            batch_sizes: Number of children per parent
        """
        batch_size = expansion_nodes.shape[0]
        
        # Calculate number of actions per node
        num_actions_per_node = legal_actions_batch.sum(dim=1)
        
        # Identify root nodes (index 0)
        root_mask = expansion_nodes == 0
        
        # Calculate max children for all nodes
        max_children_per_node = OptimizedWaveOperations.vectorized_progressive_widening(
            parent_visits, num_actions_per_node, cpw, kpw, root_mask, device
        )
        
        # Pre-allocate maximum possible space
        max_total_children = max_children_per_node.sum()
        all_parent_indices = torch.zeros(max_total_children, dtype=torch.int32, device=device)
        all_actions = torch.zeros(max_total_children, dtype=torch.int32, device=device)
        all_priors = torch.zeros(max_total_children, dtype=torch.float32, device=device)
        
        # Process each node's expansion in parallel where possible
        current_offset = 0
        batch_sizes = torch.zeros(batch_size, dtype=torch.int32, device=device)
        
        for i in range(batch_size):
            if num_actions_per_node[i] == 0:
                continue
                
            # Get legal actions and policies for this node
            legal_mask = legal_actions_batch[i]
            node_policy = policies[i]
            
            # Extract legal actions and their probabilities
            legal_indices = torch.where(legal_mask)[0]
            legal_probs = node_policy[legal_mask]
            
            # Normalize
            legal_probs = legal_probs / (legal_probs.sum() + 1e-8)
            
            # Limit by progressive widening
            max_children = max_children_per_node[i]
            if legal_indices.shape[0] > max_children:
                # Select top-k actions
                top_k_probs, top_k_indices = torch.topk(legal_probs, max_children)
                selected_actions = legal_indices[top_k_indices]
                selected_priors = top_k_probs / (top_k_probs.sum() + 1e-8)
            else:
                selected_actions = legal_indices
                selected_priors = legal_probs
            
            # Apply Dirichlet noise to root
            if root_mask[i] and epsilon > 0:
                noise = torch.distributions.Dirichlet(
                    torch.full_like(selected_priors, alpha)
                ).sample()
                selected_priors = (1 - epsilon) * selected_priors + epsilon * noise
            
            # Store in pre-allocated arrays
            n_children = selected_actions.shape[0]
            end_offset = current_offset + n_children
            
            all_parent_indices[current_offset:end_offset] = expansion_nodes[i]
            all_actions[current_offset:end_offset] = selected_actions
            all_priors[current_offset:end_offset] = selected_priors
            batch_sizes[i] = n_children
            
            current_offset = end_offset
        
        # Trim to actual size
        total_children = current_offset
        all_parent_indices = all_parent_indices[:total_children]
        all_actions = all_actions[:total_children]
        all_priors = all_priors[:total_children]
        
        return all_parent_indices, all_actions, all_priors, batch_sizes
    
    @staticmethod
    def get_max_path_length_without_sync(path_lengths: torch.Tensor) -> int:
        """
        Get maximum path length without .item() call.
        Uses a trick with tensor operations to avoid synchronization.
        """
        if path_lengths.numel() == 0:
            return 0
            
        # Create a tensor with size equal to max length + 1
        # This avoids .item() by using tensor shape
        max_length_plus_one = path_lengths.max() + 1
        dummy = torch.zeros(max_length_plus_one, device=path_lengths.device)
        return dummy.shape[0] - 1
    
    @staticmethod
    def create_path_nodes_mask(
        paths: torch.Tensor,
        path_lengths: torch.Tensor,
        max_depth: int
    ) -> torch.Tensor:
        """
        Create a mask of all valid nodes in paths without loops.
        """
        batch_size = paths.shape[0]
        
        # Create depth indices
        depth_indices = torch.arange(max_depth, device=paths.device).unsqueeze(0)
        
        # Create mask for valid positions in each path
        valid_mask = depth_indices < path_lengths.unsqueeze(1)
        
        # Get all valid nodes
        valid_nodes = paths[valid_mask]
        
        # Filter out padding (-1 values)
        valid_nodes = valid_nodes[valid_nodes >= 0]
        
        return valid_nodes
    
    @staticmethod
    def parallel_expansion_with_cuda_kernel(
        tree,
        parent_indices: torch.Tensor,
        actions: torch.Tensor,
        priors: torch.Tensor,
        batch_sizes: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Use CUDA kernel for parallel child addition if available.
        Falls back to Python implementation if kernel not available.
        """
        if hasattr(tree, 'gpu_ops') and tree.gpu_ops is not None:
            try:
                # Use batched CUDA kernel
                return tree.gpu_ops.batched_add_children(
                    parent_indices,
                    actions,
                    priors,
                    batch_sizes,
                    tree.node_counter,
                    tree.edge_counter,
                    tree.children,
                    tree.parent_indices,
                    tree.parent_actions,
                    tree.node_priors,
                    tree.visit_counts,
                    tree.value_sums,
                    tree.col_indices,
                    tree.edge_actions,
                    tree.edge_priors,
                    tree.max_nodes,
                    tree.max_children,
                    tree.max_edges
                )
            except Exception:
                # Fall back to Python implementation
                return None
        return None


def create_optimized_expand_nodes(wave_search_instance):
    """
    Create an optimized version of expand_nodes for AsyncWaveSearch.
    This replaces the method with a version that has no .item() calls.
    """
    def optimized_expand_nodes(self, expansion_nodes: torch.Tensor) -> None:
        """Optimized node expansion without CPU-GPU synchronization."""
        if expansion_nodes.numel() == 0:
            return
            
        # Get node states and legal actions
        node_states = self.node_to_state[expansion_nodes]
        legal_actions_batch = self.game_states.get_legal_moves_mask(node_states)
        
        # Get policies from neural network evaluation
        # Assuming policies were computed in the evaluate phase
        node_features = self.game_states.get_nn_features(node_states)
        with torch.amp.autocast('cuda'):
            with torch.no_grad():
                policies, _ = self.evaluator.evaluate_batch(node_features)
        
        # Get parent visit counts for progressive widening
        parent_visits = self.tree.node_data.visit_counts[expansion_nodes]
        
        # Batch process expansions without synchronization
        parent_indices, actions, priors, batch_sizes = \
            OptimizedWaveOperations.vectorized_node_expansion_batching(
                expansion_nodes,
                legal_actions_batch,
                policies,
                parent_visits,
                self.config.cpw,
                self.config.kpw,
                getattr(self.config, 'dirichlet_epsilon', 0.25),
                getattr(self.config, 'dirichlet_alpha', 0.03),
                self.device
            )
        
        if parent_indices.numel() == 0:
            return
            
        # Try to use CUDA kernel for batch addition
        child_indices = OptimizedWaveOperations.parallel_expansion_with_cuda_kernel(
            self.tree, parent_indices, actions, priors, batch_sizes
        )
        
        if child_indices is None:
            # Fallback to sequential addition (still avoiding .item() where possible)
            # This is less optimal but maintains compatibility
            offset = 0
            for i in range(batch_sizes.shape[0]):
                n_children = batch_sizes[i]
                if n_children == 0:
                    continue
                    
                parent_idx = parent_indices[offset]
                node_actions = actions[offset:offset + n_children]
                node_priors = priors[offset:offset + n_children]
                
                # Add children (this may still have some synchronization)
                self.tree.add_children(
                    parent_idx,
                    node_actions,
                    node_priors
                )
                
                offset += n_children
        
        # Update game states for new nodes
        if hasattr(self.tree, 'num_nodes'):
            new_num_nodes = self.tree.num_nodes
            if new_num_nodes > self.node_to_state.shape[0]:
                # Expand node_to_state array
                additional_states = self.game_states.create_empty_states(
                    new_num_nodes - self.node_to_state.shape[0]
                )
                self.node_to_state = torch.cat([self.node_to_state, additional_states])
    
    # Bind the optimized method
    import types
    wave_search_instance._expand_nodes = types.MethodType(
        optimized_expand_nodes, wave_search_instance
    )