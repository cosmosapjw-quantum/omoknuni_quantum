"""Wave-based parallelization for MCTS

This module contains the wave-based parallelization logic extracted from the main MCTS class.
Wave parallelization allows processing multiple MCTS paths in parallel for improved performance.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

from ..gpu.csr_tree import CSRTree
from ..gpu.gpu_game_states import GPUGameStates
from ..gpu.mcts_gpu_accelerator import get_mcts_gpu_accelerator
from ..gpu.cuda_manager import detect_cuda_kernels

logger = logging.getLogger(__name__)


class WaveSearch:
    """Wave-based parallel search implementation for MCTS
    
    This class handles the vectorized/batched operations for processing
    multiple MCTS simulations in parallel waves.
    """
    
    def __init__(
        self,
        tree: CSRTree,
        game_states: GPUGameStates,
        evaluator: Any,
        config: Any,
        device: torch.device,
        gpu_ops: Optional[Any] = None
    ):
        """Initialize wave search
        
        Args:
            tree: CSR tree structure
            game_states: GPU game states manager
            evaluator: Neural network evaluator
            config: MCTS configuration
            device: Torch device
            gpu_ops: Optional GPU operations accelerator
        """
        self.tree = tree
        self.game_states = game_states
        self.evaluator = evaluator
        self.config = config
        self.device = device
        self.gpu_ops = gpu_ops
        
        # Try to get GPU accelerator for optimized kernels
        if gpu_ops is None:
            try:
                self.gpu_ops = get_mcts_gpu_accelerator(device)
                logger.debug("Loaded GPU accelerator for wave search")
            except Exception as e:
                logger.debug(f"GPU accelerator not available: {e}")
                self.gpu_ops = None
        
        # Also try to get direct kernel access for wave search optimizations
        self.cuda_kernels = None
        try:
            self.cuda_kernels = detect_cuda_kernels()
            if self.cuda_kernels:
                pass  # CUDA kernels available
        except Exception as e:
            logger.debug(f"Direct CUDA kernels not available: {e}")
        
        # Buffers will be allocated on first use
        self._buffers_allocated = False
        
        # Store original root priors for Dirichlet noise mixing
        self.original_root_priors = None
        
        # Cache for batched Dirichlet noise generation
        self._dirichlet_cache = {}
        
    def allocate_buffers(self, wave_size: int, max_depth: int = 100):
        """Allocate work buffers for wave operations
        
        Args:
            wave_size: Size of parallel wave
            max_depth: Maximum search depth
        """
        # Selection buffers
        self.paths_buffer = torch.zeros((wave_size, max_depth), dtype=torch.int32, device=self.device)
        self.path_lengths = torch.zeros(wave_size, dtype=torch.int32, device=self.device)
        self.current_nodes = torch.zeros(wave_size, dtype=torch.int32, device=self.device)
        self.next_nodes = torch.zeros(wave_size, dtype=torch.int32, device=self.device)
        self.active_mask = torch.ones(wave_size, dtype=torch.bool, device=self.device)
        
        # UCB computation buffers
        max_children = self.config.max_children_per_node
        self.ucb_scores = torch.zeros((wave_size, max_children), device=self.device)
        self.child_indices = torch.zeros((wave_size, max_children), dtype=torch.int32, device=self.device)
        self.child_mask = torch.zeros((wave_size, max_children), dtype=torch.bool, device=self.device)
        
        # Expansion buffers
        self.expansion_nodes = torch.zeros(wave_size, dtype=torch.int32, device=self.device)
        self.expansion_count = torch.zeros(wave_size, dtype=torch.int32, device=self.device)
        self.node_features = torch.zeros((wave_size, 3, self.config.board_size, self.config.board_size), device=self.device)
        
        # Evaluation buffers
        self.eval_batch = torch.zeros((wave_size, 3, self.config.board_size, self.config.board_size), device=self.device)
        self.policy_values = torch.zeros((wave_size, self.config.board_size * self.config.board_size), device=self.device)
        self.value_estimates = torch.zeros(wave_size, device=self.device)
        
        # Backup buffers
        self.backup_values = torch.zeros(wave_size, device=self.device)
        self.visit_increments = torch.zeros(wave_size, dtype=torch.int32, device=self.device)
        
        # State management
        self.state_indices = torch.zeros(wave_size, dtype=torch.int32, device=self.device)
        self.node_to_state = getattr(self, 'node_to_state', None)  # Reference from parent
        self.state_pool_free_list = getattr(self, 'state_pool_free_list', None)  # Reference from parent
        
        self._buffers_allocated = True
        
    def apply_per_simulation_dirichlet_noise(self, node_idx: int, sim_indices: torch.Tensor, 
                                           children: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        """Apply different Dirichlet noise to each simulation's view of the priors
        
        Args:
            node_idx: Node index (typically root=0)
            sim_indices: Indices of simulations at this node
            children: Child node indices
            priors: Original prior probabilities
            
        Returns:
            Tensor of shape (num_sims, num_children) with noised priors
        """
        if node_idx != 0 or self.config.dirichlet_epsilon <= 0:
            # Only apply to root and if epsilon > 0
            return priors.unsqueeze(0).expand(len(sim_indices), -1)
            
        num_sims = len(sim_indices)
        num_children = len(children)
        
        # Try to use CUDA kernel for batched Dirichlet noise
        if self.cuda_kernels and hasattr(self.cuda_kernels, 'batched_dirichlet_noise'):
            try:
                noise = self.cuda_kernels.batched_dirichlet_noise(
                    num_sims, num_children, 
                    self.config.dirichlet_alpha,
                    self.config.dirichlet_epsilon,
                    self.device
                )
            except Exception as e:
                # Fallback to PyTorch implementation
                alpha = self.config.dirichlet_alpha
                noise_dist = torch.distributions.Dirichlet(
                    torch.full((num_children,), alpha, device=self.device)
                )
                noise = torch.stack([noise_dist.sample() for _ in range(num_sims)])
        else:
            # Generate different Dirichlet noise for each simulation
            alpha = self.config.dirichlet_alpha
            # Use cached distribution if available
            cache_key = (num_children, alpha)
            if cache_key not in self._dirichlet_cache:
                self._dirichlet_cache[cache_key] = torch.distributions.Dirichlet(
                    torch.full((num_children,), alpha, device=self.device)
                )
            noise_dist = self._dirichlet_cache[cache_key]
            
            # Batch sample for efficiency
            noise = noise_dist.sample((num_sims,))
        
        # Mix with original priors
        epsilon = self.config.dirichlet_epsilon
        noised_priors = (1 - epsilon) * priors.unsqueeze(0) + epsilon * noise
        
        return noised_priors
        
    def run_wave(self, wave_size: int, node_to_state: torch.Tensor, state_pool_free_list: List[int]) -> int:
        """Run one wave of parallel MCTS simulations
        
        Args:
            wave_size: Number of parallel simulations
            node_to_state: Mapping from nodes to states
            state_pool_free_list: Free list for state allocation
            
        Returns:
            Number of simulations completed
        """
        # Store references for state management
        self.node_to_state = node_to_state
        self.state_pool_free_list = state_pool_free_list
        
        # Ensure buffers are allocated
        if not self._buffers_allocated or self.paths_buffer.shape[0] < wave_size:
            self.allocate_buffers(wave_size)
            
        # Phase 1: Selection - traverse tree in parallel
        paths, path_lengths, leaf_nodes = self._select_batch_vectorized(wave_size)
        
        # Phase 2: Expansion - expand leaf nodes
        expanded_nodes = self._expand_batch_vectorized(leaf_nodes)
        
        # Phase 2.5: If we expanded nodes, we need to select one of their children
        # This is critical for the root node case where it has no children initially
        final_nodes, updated_paths, updated_lengths = self._select_from_expanded_nodes(
            expanded_nodes, leaf_nodes, paths, path_lengths
        )
        
        # Phase 3: Evaluation - evaluate expanded nodes or their selected children
        values = self._evaluate_batch_vectorized(final_nodes)
        
        # Phase 4: Backup - propagate values up the tree
        self._backup_batch_vectorized(updated_paths, updated_lengths, values)
        
        return wave_size
        
    def _select_batch_vectorized(self, wave_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select paths through tree in parallel with full vectorization
        
        Args:
            wave_size: Number of parallel paths
            
        Returns:
            Tuple of (paths, path_lengths, leaf_nodes)
        """
        # Initialize paths starting from root
        self.paths_buffer[:wave_size, 0] = 0  # Root node
        self.path_lengths[:wave_size] = 1
        self.current_nodes[:wave_size] = 0
        self.active_mask[:wave_size] = True
        
        max_depth = self.paths_buffer.shape[1]
        
        for depth in range(1, max_depth):
            # Get active indices
            active_indices = self.active_mask[:wave_size].nonzero(as_tuple=True)[0]
            if len(active_indices) == 0:
                break
                
            active_nodes = self.current_nodes[active_indices]
            
            # Use batch_get_children for vectorized child retrieval
            batch_children, batch_actions, batch_priors = self.tree.batch_get_children(active_nodes)
            
            # Create masks for valid children
            valid_children_mask = batch_children >= 0
            
            # Vectorized computation of UCB scores
            with torch.no_grad():
                # Get visits and values for all children in batch
                flat_children = batch_children[valid_children_mask]
                if len(flat_children) > 0:
                    child_visits_flat = self.tree.node_data.visit_counts[flat_children].float()
                    child_values_flat = self.tree.node_data.value_sums[flat_children]
                    
                    # Reshape back to batch format
                    child_visits = torch.zeros_like(batch_children, dtype=torch.float32)
                    child_values = torch.zeros_like(batch_children, dtype=torch.float32)
                    child_visits[valid_children_mask] = child_visits_flat
                    child_values[valid_children_mask] = child_values_flat
                    
                    # Calculate Q-values
                    q_values = torch.where(
                        child_visits > 0,
                        child_values / child_visits,
                        torch.zeros_like(child_values)
                    )
                    
                    # Get parent visits
                    parent_visits = self.tree.node_data.visit_counts[active_nodes].float()
                    parent_visits_sqrt = torch.sqrt(torch.maximum(parent_visits, torch.ones(1, device=self.device)))
                    
                    # Apply per-simulation Dirichlet noise for root
                    noised_priors = torch.zeros_like(batch_priors)
                    for i, node in enumerate(active_nodes):
                        if node == 0 and self.config.dirichlet_epsilon > 0:  # Root node
                            node_children = batch_children[i][valid_children_mask[i]]
                            node_priors = batch_priors[i][valid_children_mask[i]]
                            sim_idx = active_indices[i:i+1]
                            
                            noise_priors = self.apply_per_simulation_dirichlet_noise(
                                0, sim_idx, node_children, node_priors
                            )[0]
                            noised_priors[i][valid_children_mask[i]] = noise_priors
                        else:
                            noised_priors[i] = batch_priors[i]
                    
                    # Try fused UCB computation with CUDA kernel
                    if (node == 0 and self.cuda_kernels and 
                        hasattr(self.cuda_kernels, 'fused_ucb_with_noise')):
                        try:
                            # Use fused kernel for root node with Dirichlet noise
                            noise = self.apply_dirichlet_noise_batched(
                                len(active_indices), len(node_children),
                                self.config.dirichlet_alpha, 
                                self.config.dirichlet_epsilon
                            )
                            selected_indices = self.cuda_kernels.fused_ucb_with_noise(
                                child_values[valid_children_mask],
                                child_visits[valid_children_mask],
                                batch_priors[valid_children_mask],
                                noise,
                                parent_visits_sqrt[0].item(),
                                self.config.c_puct,
                                self.config.dirichlet_epsilon
                            )
                            # Map back to batch indices
                            best_indices = selected_indices
                            selected_children = batch_children[torch.arange(len(batch_children)), best_indices]
                        except Exception as e:
                            # Standard computation
                            exploration = (self.config.c_puct * noised_priors * 
                                         parent_visits_sqrt.unsqueeze(1) / (1 + child_visits))
                            ucb_scores = q_values + exploration
                    else:
                        # Standard vectorized UCB calculation
                        exploration = (self.config.c_puct * noised_priors * 
                                     parent_visits_sqrt.unsqueeze(1) / (1 + child_visits))
                        ucb_scores = q_values + exploration
                    
                    # Mask out invalid children
                    ucb_scores = torch.where(
                        valid_children_mask,
                        ucb_scores,
                        torch.full_like(ucb_scores, -float('inf'))
                    )
                    
                    # Select best children
                    best_indices = ucb_scores.argmax(dim=1)
                    selected_children = batch_children.gather(1, best_indices.unsqueeze(1)).squeeze(1)
                    
                    # Update only nodes that have children
                    has_children = valid_children_mask.any(dim=1)
                    nodes_with_children = active_indices[has_children]
                    children_to_select = selected_children[has_children]
                    
                    # Update paths and next nodes
                    self.paths_buffer[nodes_with_children, depth] = children_to_select
                    self.path_lengths[nodes_with_children] = depth + 1
                    self.next_nodes[nodes_with_children] = children_to_select
                    
                    # Mark nodes without children as inactive
                    nodes_without_children = active_indices[~has_children]
                    self.active_mask[nodes_without_children] = False
                else:
                    # No valid children for any active node
                    self.active_mask[active_indices] = False
            
            # Move to next nodes
            self.current_nodes[:wave_size] = self.next_nodes[:wave_size]
        
        # Vectorized leaf node extraction
        path_indices = torch.arange(wave_size, device=self.device)
        path_lengths_clamped = (self.path_lengths[:wave_size] - 1).clamp(min=0)
        leaf_nodes = self.paths_buffer[path_indices, path_lengths_clamped]
        
        return self.paths_buffer[:wave_size].clone(), self.path_lengths[:wave_size].clone(), leaf_nodes
        
    def _expand_batch_vectorized(self, leaf_nodes: torch.Tensor) -> torch.Tensor:
        """Expand leaf nodes in parallel with vectorized operations
        
        Args:
            leaf_nodes: Tensor of leaf node indices
            
        Returns:
            Tensor of expanded node indices
        """
        expanded_nodes = leaf_nodes.clone()
        
        # Get unique nodes to expand
        unique_nodes, inverse_indices = torch.unique(leaf_nodes, return_inverse=True)
        
        # Filter valid nodes
        valid_mask = (unique_nodes >= 0) & (unique_nodes < self.tree.num_nodes)
        valid_unique_nodes = unique_nodes[valid_mask]
        
        if len(valid_unique_nodes) == 0:
            return expanded_nodes
        
        # Check which nodes need expansion using batch operation
        needs_expansion = torch.zeros_like(valid_unique_nodes, dtype=torch.bool)
        state_indices = torch.zeros_like(valid_unique_nodes, dtype=torch.int32)
        
        for i, node_idx in enumerate(valid_unique_nodes):
            children, _, _ = self.tree.get_children(node_idx.item())
            if len(children) == 0:
                state_idx = self.node_to_state[node_idx].item()
                if state_idx >= 0:
                    needs_expansion[i] = True
                    state_indices[i] = state_idx
        
        nodes_to_expand = valid_unique_nodes[needs_expansion]
        states_to_expand = state_indices[needs_expansion]
        
        if len(nodes_to_expand) == 0:
            return expanded_nodes
        
        # Batch process all nodes that need expansion
        with torch.no_grad():
            # Get legal moves for all states at once
            legal_masks = self.game_states.get_legal_moves_mask(states_to_expand)
            
            # Get features for all states
            state_features = self.game_states.get_nn_features_batch(states_to_expand)
            
            # Evaluate all states at once
            if hasattr(self.evaluator, 'evaluate_batch'):
                features_np = state_features.cpu().numpy()
                policies, _ = self.evaluator.evaluate_batch(features_np)
                if isinstance(policies, np.ndarray):
                    policies = torch.from_numpy(policies).to(self.device)
                elif not isinstance(policies, torch.Tensor):
                    policies = torch.tensor(policies, device=self.device)
            else:
                # Fallback to sequential evaluation
                policies_list = []
                for feat in state_features:
                    feat_np = feat.cpu().numpy()
                    policy, _ = self.evaluator.evaluate(feat_np)
                    policies_list.append(torch.from_numpy(policy).to(self.device))
                policies = torch.stack(policies_list)
            
            # Process each node's expansion
            for i, (node_idx, state_idx, legal_mask, policy) in enumerate(
                zip(nodes_to_expand, states_to_expand, legal_masks, policies)
            ):
                legal_actions = torch.nonzero(legal_mask).squeeze(-1)
                
                if len(legal_actions) > 0:
                    # Extract and normalize priors
                    priors = policy[legal_actions]
                    priors = priors / (priors.sum() + 1e-8)
                    
                    # Prepare for batch operations
                    num_actions = len(legal_actions)
                    
                    # Clone states for children
                    parent_indices = state_idx.unsqueeze(0)
                    num_clones = torch.tensor([num_actions], dtype=torch.int32, device=self.device)
                    child_state_indices = self.game_states.clone_states(parent_indices, num_clones)
                    
                    # Apply moves
                    self.game_states.apply_moves(child_state_indices, legal_actions)
                    
                    # Add children to tree
                    actions_list = legal_actions.cpu().tolist()
                    priors_list = priors.cpu().tolist()
                    child_states_list = child_state_indices.cpu().tolist()
                    
                    child_indices = self.tree.add_children_batch(
                        node_idx.item(),
                        actions_list,
                        priors_list,
                        child_states_list
                    )
                    
                    # Update node-to-state mapping
                    for child_idx, child_state_idx in zip(child_indices, child_states_list):
                        if child_idx < len(self.node_to_state):
                            self.node_to_state[child_idx] = child_state_idx
        
        return expanded_nodes
        
    def _select_from_expanded_nodes(self, expanded_nodes: torch.Tensor, original_leaf_nodes: torch.Tensor,
                                   paths: torch.Tensor, path_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select children from newly expanded nodes
        
        Args:
            expanded_nodes: Nodes that were expanded
            original_leaf_nodes: Original leaf nodes before expansion
            paths: Current paths through the tree
            path_lengths: Lengths of each path
            
        Returns:
            Tuple of (final_nodes, updated_paths, updated_lengths)
        """
        final_nodes = expanded_nodes.clone()
        updated_paths = paths.clone()
        updated_lengths = path_lengths.clone()
        
        # Group simulations by which node they're at
        unique_nodes = torch.unique(expanded_nodes)
        
        for node_idx in unique_nodes:
            node_idx_val = node_idx.item()
            if node_idx_val < 0 or node_idx_val >= self.tree.num_nodes:
                continue
                
            # Find all simulations at this node
            sim_mask = expanded_nodes == node_idx
            sim_indices = torch.where(sim_mask)[0]
            
            if len(sim_indices) == 0:
                continue
                
            # Get children
            children, _, priors = self.tree.get_children(node_idx_val)
            
            if len(children) > 0:
                # This node has children now
                visits = self.tree.node_data.visit_counts[children]
                values = self.tree.node_data.value_sums[children] / (visits + 1e-8)
                
                # Apply per-simulation Dirichlet noise if at root
                noised_priors = self.apply_per_simulation_dirichlet_noise(
                    node_idx_val, sim_indices, children, priors
                )
                
                # Compute UCB for each simulation with its own noised priors
                parent_visits = self.tree.node_data.visit_counts[node_idx_val]
                parent_visits_sqrt = torch.sqrt(torch.maximum(parent_visits.float(), torch.ones(1, device=self.device)))
                
                # Expand visits and values to match simulation count
                visits_expanded = visits.unsqueeze(0).expand(len(sim_indices), -1)
                values_expanded = values.unsqueeze(0).expand(len(sim_indices), -1)
                
                # UCB formula with per-simulation priors
                exploration = self.config.c_puct * noised_priors * parent_visits_sqrt / (1 + visits_expanded)
                ucb = values_expanded + exploration
                
                # Each simulation selects its best child based on its own UCB scores
                best_indices = ucb.argmax(dim=1)
                selected_children = children[best_indices]
                
                # Update final nodes and paths for these simulations
                for i, (sim_idx, child) in enumerate(zip(sim_indices, selected_children)):
                    final_nodes[sim_idx] = child
                    
                    # Update path to include the selected child
                    current_length = updated_lengths[sim_idx].item()
                    if current_length < updated_paths.shape[1]:
                        updated_paths[sim_idx, current_length] = child
                        updated_lengths[sim_idx] = current_length + 1
                
        return final_nodes, updated_paths, updated_lengths
        
    def _evaluate_batch_vectorized(self, nodes: torch.Tensor) -> torch.Tensor:
        """Evaluate nodes using neural network
        
        Args:
            nodes: Tensor of node indices to evaluate
            
        Returns:
            Tensor of value estimates
        """
        batch_size = nodes.shape[0]
        values = torch.zeros(batch_size, device=self.device)
        
        # Get states for nodes
        valid_mask = (nodes >= 0) & (nodes < self.tree.num_nodes)
        valid_nodes = nodes[valid_mask]
        
        if len(valid_nodes) > 0:
            state_indices = self.node_to_state[valid_nodes]
            valid_states_mask = state_indices >= 0
            
            if valid_states_mask.any():
                # Get features for valid states
                valid_state_indices = state_indices[valid_states_mask]
                features = self.game_states.get_nn_features_batch(valid_state_indices)
                
                # Evaluate with neural network
                with torch.no_grad():
                    # Convert to numpy for evaluator
                    features_np = features.cpu().numpy()
                    # Use evaluate_batch since we may have multiple states
                    if hasattr(self.evaluator, 'evaluate_batch'):
                        policies, value_preds = self.evaluator.evaluate_batch(features_np)
                    else:
                        # Fallback for single state evaluation
                        policies = []
                        value_preds = []
                        for feat in features_np:
                            p, v = self.evaluator.evaluate(feat)
                            policies.append(p)
                            value_preds.append(v)
                        policies = np.array(policies)
                        value_preds = np.array(value_preds)
                    
                # Store values back
                if isinstance(value_preds, np.ndarray):
                    value_preds = torch.from_numpy(value_preds).to(self.device).float()
                elif isinstance(value_preds, torch.Tensor):
                    value_preds = value_preds.float()  # Ensure float dtype
                valid_indices = valid_mask.nonzero().squeeze(-1)
                valid_indices = valid_indices[valid_states_mask]
                values[valid_indices] = value_preds.squeeze() if value_preds.ndim > 1 else value_preds
                
        return values
        
    def _backup_batch_vectorized(self, paths: torch.Tensor, path_lengths: torch.Tensor, values: torch.Tensor):
        """Backup values through paths in parallel
        
        Args:
            paths: Tensor of paths through tree
            path_lengths: Length of each path
            values: Value estimates to backup
        """
        batch_size = paths.shape[0]
        
        # Try to use optimized CUDA kernel
        if self.gpu_ops and hasattr(self.gpu_ops, 'vectorized_backup'):
            try:
                self.gpu_ops.vectorized_backup(
                    paths, path_lengths, values,
                    self.tree.node_data.visit_counts,
                    self.tree.node_data.value_sums
                )
                return
            except Exception as e:
                pass
        
        # Try scatter-based vectorized backup
        try:
            self._scatter_backup(paths, path_lengths, values)
            return
        except Exception as e:
            pass
        
        # Fallback to sequential processing
        for i in range(batch_size):
            path_len = path_lengths[i].item()
            if path_len == 0:
                continue
                
            value = values[i].item()
            
            # Backup through path (reverse order)
            for j in range(path_len - 1, -1, -1):
                node = paths[i, j].item()
                if node < 0 or node >= self.tree.num_nodes:
                    continue
                    
                # Update node statistics
                self.tree.node_data.visit_counts[node] += 1
                self.tree.node_data.value_sums[node] += value
                
                # Flip value for opponent
                value = -value
    
    def _scatter_backup(self, paths: torch.Tensor, lengths: torch.Tensor, values: torch.Tensor):
        """Optimized backup using scatter operations"""
        batch_size = paths.shape[0]
        max_length = lengths.max().item()
        
        if max_length == 0:
            return
            
        # Create mask for valid path positions
        length_range = torch.arange(max_length, device=self.device).unsqueeze(0)
        valid_mask = length_range < lengths.unsqueeze(1)
        
        # Flatten paths for valid positions
        valid_nodes = paths[:, :max_length][valid_mask]
        
        # Create alternating signs for player perspective
        signs = torch.ones((batch_size, max_length), device=self.device)
        signs[:, 1::2] = -1  # Flip sign for odd depths
        
        # Expand values and apply signs
        expanded_values = values.unsqueeze(1).expand(-1, max_length)
        signed_values = expanded_values * signs
        valid_values = signed_values[valid_mask]
        
        # Perform atomic scatter updates
        self.tree.node_data.visit_counts.scatter_add_(
            0, valid_nodes.long(), 
            torch.ones_like(valid_nodes, dtype=torch.int32)
        )
        self.tree.node_data.value_sums.scatter_add_(
            0, valid_nodes.long(), 
            valid_values
        )