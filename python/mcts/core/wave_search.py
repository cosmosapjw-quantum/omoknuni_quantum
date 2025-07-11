"""Wave-based parallelization for MCTS

This module contains the wave-based parallelization logic extracted from the main MCTS class.
Wave parallelization allows processing multiple MCTS paths in parallel for improved performance.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

from ..gpu.csr_tree import CSRTree
from ..gpu.gpu_game_states import GPUGameStates, GameType
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
            except Exception as e:
                self.gpu_ops = None
        
        # Also try to get direct kernel access for wave search optimizations
        self.cuda_kernels = None
        try:
            self.cuda_kernels = detect_cuda_kernels()
        except Exception as e:
            self.cuda_kernels = None
        
        # Buffers will be allocated on first use
        self._buffers_allocated = False
        self.paths_buffer = torch.empty(0, device=self.device)  # Initialize empty tensor
        
        # State management - will be set during run_wave
        self.node_to_state = None
        self.state_pool_free_list = None
        
        # Store original root priors for Dirichlet noise mixing
        self.original_root_priors = None
        
        # Cache for batched Dirichlet noise generation
        self._dirichlet_cache = {}
        
        # Initialize tactical detector based on game type if enabled
        self._tactical_detector = None
        if config.enable_tactical_boost:
            try:
                if config.game_type == GameType.GO:
                    from ..utils.go_tactical_detector import GoTacticalMoveDetector
                    self._tactical_detector = GoTacticalMoveDetector(config.board_size, config)
                elif config.game_type == GameType.CHESS:
                    from ..utils.chess_tactical_detector import ChessTacticalMoveDetector
                    self._tactical_detector = ChessTacticalMoveDetector(config)
                elif config.game_type == GameType.GOMOKU:
                    from ..utils.gomoku_tactical_detector import GomokuTacticalMoveDetector
                    self._tactical_detector = GomokuTacticalMoveDetector(config.board_size, config)
            except ImportError as e:
                logger.warning(f"Tactical move detector not available: {e}")
                self._tactical_detector = None
        
    def _get_max_children_for_expansion(self, parent_visits: int, num_legal_moves: int) -> int:
        """Calculate maximum children to expand using progressive widening
        
        Args:
            parent_visits: Number of visits to parent node
            num_legal_moves: Total number of legal moves
            
        Returns:
            Maximum number of children to expand
        """
        # Always expand all legal moves (no progressive widening)
        return num_legal_moves
        
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
        
        # Defensive check: ensure priors match children count
        if len(priors) != num_children:
            logger.error(f"Priors length {len(priors)} != children length {num_children}")
            # Return expanded priors without noise as fallback
            return priors.unsqueeze(0).expand(len(sim_indices), -1)
        
        # Try to use CUDA kernel for batched Dirichlet noise (only on CUDA devices)
        if self.cuda_kernels and hasattr(self.cuda_kernels, 'batched_dirichlet_noise') and self.device.type == 'cuda':
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
            
            # Initialize cache if needed
            if not hasattr(self, '_dirichlet_cache'):
                self._dirichlet_cache = {}
            
            # Use cached distribution if available
            cache_key = (num_children, alpha)
            if cache_key not in self._dirichlet_cache:
                try:
                    self._dirichlet_cache[cache_key] = torch.distributions.Dirichlet(
                        torch.full((num_children,), alpha, device=self.device)
                    )
                except Exception as e:
                    logger.error(f"Failed to create Dirichlet distribution: {e}")
                    # Return expanded priors without noise
                    return priors.unsqueeze(0).expand(len(sim_indices), -1)
            
            noise_dist = self._dirichlet_cache[cache_key]
            
            # Batch sample for efficiency
            try:
                noise = noise_dist.sample((num_sims,))
            except Exception as e:
                logger.error(f"Failed to sample Dirichlet noise: {e}")
                # Return expanded priors without noise
                return priors.unsqueeze(0).expand(len(sim_indices), -1)
        
        # Mix with original priors
        epsilon = self.config.dirichlet_epsilon
        try:
            priors_expanded = priors.unsqueeze(0).expand(len(sim_indices), -1)
            if priors_expanded.shape != noise.shape:
                # This can happen if noise was generated for wrong batch size
                # Just expand priors without noise as fallback
                return priors_expanded
                
            noised_priors = (1 - epsilon) * priors_expanded + epsilon * noise
        except Exception as e:
            logger.error(f"Failed to mix priors with noise: {e}")
            return priors.unsqueeze(0).expand(len(sim_indices), -1)
        
        return noised_priors
    
    def reset_search_state(self):
        """Reset search state for a new search"""
        self._global_noise_cache = None
        
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
            
        # Try fused select+expand if available
        if self.gpu_ops and hasattr(self.gpu_ops, 'fused_select_expand') and self.config.enable_kernel_fusion:
            fused_result = self._try_fused_select_expand(wave_size)
            if fused_result is not None:
                paths, path_lengths, leaf_nodes, expanded_nodes = fused_result
            else:
                # Fallback to separate phases
                paths, path_lengths, leaf_nodes = self._select_batch_vectorized(wave_size)
                expanded_nodes = self._expand_batch_vectorized(leaf_nodes)
        else:
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
        
        # Phase 5: Remove virtual losses from all paths
        self._remove_virtual_losses_from_paths(updated_paths, updated_lengths)
        
        return wave_size
        
    def _try_fused_select_expand(self, wave_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Try to use fused select+expand kernel if available
        
        Returns:
            Tuple of (paths, path_lengths, leaf_nodes, expanded_nodes) or None if not available
        """
        try:
            # Prepare inputs for fused kernel
            roots = torch.arange(wave_size, device=self.device, dtype=torch.int32)
            
            # Get tree data directly from node_data
            children = self.tree.node_data.children  # Shape: [num_nodes, max_children]
            visit_counts = self.tree.node_data.visit_counts
            q_values = self.tree.node_data.q_values
            prior_probs = self.tree.node_data.prior_probs
            is_expanded = self.tree.node_data.is_expanded
            
            # Call fused kernel
            result = self.gpu_ops.fused_select_expand(
                roots=roots,
                children=children,
                visit_counts=visit_counts,
                q_values=q_values,
                prior_probs=prior_probs,
                is_expanded=is_expanded,
                max_depth=self.config.max_depth,
                c_puct=self.config.c_puct
            )
            
            if result is None:
                return None
                
            paths, path_lengths, expand_nodes, needs_expansion = result
            
            # Process results to match expected format
            leaf_nodes = expand_nodes[needs_expansion]
            expanded_nodes = self._expand_nodes_batch(leaf_nodes)
            
            return paths, path_lengths, leaf_nodes, expanded_nodes
            
        except Exception as e:
            logger.debug(f"Fused select+expand failed: {e}")
            return None
    
    def _select_batch_vectorized(self, wave_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select paths through tree in parallel with full vectorization
        
        Args:
            wave_size: Number of parallel paths
            
        Returns:
            Tuple of (paths, path_lengths, leaf_nodes)
        """
        # Ensure buffers are allocated
        if not self._buffers_allocated or self.paths_buffer.shape[0] < wave_size:
            self.allocate_buffers(wave_size)
            
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
            
            # Optimized parallel selection with proper virtual loss synchronization
            selected_children_all = self._parallel_select_with_virtual_loss(
                active_indices, active_nodes, batch_children, batch_priors, valid_children_mask, depth
            )
            
            # Update paths and next nodes for successful selections
            if selected_children_all is not None:
                has_children = selected_children_all >= 0
                nodes_with_children = active_indices[has_children]
                children_to_select = selected_children_all[has_children]
                
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
        
    def _expand_batch_vectorized(self, leaf_nodes: torch.Tensor, 
                                node_to_state: Optional[torch.Tensor] = None,
                                state_pool_free_list: Optional[List[int]] = None) -> torch.Tensor:
        """Expand leaf nodes in parallel with vectorized operations
        
        Args:
            leaf_nodes: Tensor of leaf node indices
            node_to_state: Optional node-to-state mapping (if not provided, uses stored value)
            state_pool_free_list: Optional state pool free list (if not provided, uses stored value)
            
        Returns:
            Tensor of expanded node indices
        """
        # Use provided parameters or fall back to stored values
        if node_to_state is not None:
            current_node_to_state = node_to_state
        else:
            current_node_to_state = self.node_to_state
            
        if state_pool_free_list is not None:
            current_state_pool_free_list = state_pool_free_list
        else:
            current_state_pool_free_list = self.state_pool_free_list
        
        # Ensure node_to_state is available
        if current_node_to_state is None:
            raise RuntimeError("node_to_state not initialized. Call run_wave first or provide node_to_state parameter.")
            
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
            node_idx_val = node_idx.item()
            children, _, _ = self.tree.get_children(node_idx_val)
            
            if len(children) == 0:
                state_idx = current_node_to_state[node_idx].item()
                if state_idx >= 0:
                    # CRITICAL: Check if state is terminal before expanding
                    is_terminal = self.game_states.is_terminal[state_idx].item()
                    if not is_terminal:
                        needs_expansion[i] = True
                        state_indices[i] = state_idx
                    else:
                        continue  # Skip terminal node expansion
        
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
                    # Handle batch dimension - evaluator returns (batch_size, action_size)
                    # but we need just (action_size) for each state
                    if isinstance(policy, np.ndarray) and policy.ndim == 2 and policy.shape[0] == 1:
                        policy = policy[0]  # Remove batch dimension if batch_size=1
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
                    
                    # Apply tactical boost if enabled for any game
                    if (self.config.enable_tactical_boost and
                        hasattr(self, '_tactical_detector') and
                        self._tactical_detector is not None):
                        # Get board state
                        board = self.game_states.boards[state_idx]
                        # GPUGameStates already uses 1-based players (1, 2)
                        current_player = self.game_states.current_player[state_idx].item()
                        
                        # Debug: log board state
                        # Board state ready for tactical detection
                        
                        # Create full prior vector for boosting
                        if self.config.game_type == GameType.CHESS:
                            # Chess has different move encoding
                            full_priors = torch.zeros(4096, device=self.device)  # Max chess moves
                            full_priors[legal_actions] = priors
                            
                            # Apply tactical boost with legal moves for chess
                            boosted_priors = self._tactical_detector.boost_prior_with_tactics(
                                full_priors, board, current_player, 
                                legal_actions.cpu().tolist(),
                                self.config.tactical_boost_strength
                            )
                        else:
                            # Go and Gomoku use board position encoding
                            action_space_size = self.config.board_size ** 2
                            if self.config.game_type == GameType.GO:
                                action_space_size += 1  # Add pass move
                            
                            full_priors = torch.zeros(action_space_size, device=self.device, dtype=priors.dtype)
                            full_priors[legal_actions] = priors
                            
                            # Apply tactical boost
                            boosted_priors = self._tactical_detector.boost_prior_with_tactics(
                                full_priors, board, current_player, 
                                self.config.tactical_boost_strength
                            )
                        
                        # Extract boosted priors for legal actions
                        old_priors = priors.clone()
                        priors = boosted_priors[legal_actions]
                        priors = priors / (priors.sum() + 1e-8)
                        
                        # Store the boosted priors in the tree
                        # This ensures UCB calculations use the boosted values
                        
                        # Debug logging
                        if node_idx.item() == 0:  # Root node
                            pass  # Board shape checked
                            
                            # Check boost values
                            boost_values = self._tactical_detector.detect_tactical_moves(board, current_player)
                            pass  # Boost values computed
                            
                            # Log the actual boost calculation
                            capture_idx = 29  # Capture move at (3,2)
                            if capture_idx in legal_actions:
                                idx_in_legal = (legal_actions == capture_idx).nonzero(as_tuple=True)[0]
                                if len(idx_in_legal) > 0:
                                    idx = idx_in_legal[0]
                                    pass  # Capture move boosted
                    
                    # Expand all legal moves (no progressive widening)
                    
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
                        if child_idx < len(current_node_to_state):
                            current_node_to_state[child_idx] = child_state_idx
        
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
            children, actions, priors = self.tree.get_children(node_idx_val)
            
            if len(children) > 0:
                # CRITICAL FIX: Filter out children whose actions are no longer legal
                # This prevents illegal moves from being selected by UCB
                state_idx = self.node_to_state[node_idx_val].item()
                if state_idx >= 0:
                    # Get current legal moves for this state
                    legal_mask = self.game_states.get_legal_moves_mask(
                        torch.tensor([state_idx], device=self.device)
                    )[0]
                    
                    # Vectorized legal action filtering
                    child_actions = self.tree.node_data.parent_actions[children]
                    valid_actions = (child_actions >= 0) & (child_actions < legal_mask.shape[0])
                    legal_children_mask = torch.zeros(len(children), dtype=torch.bool, device=self.device)
                    if valid_actions.any():
                        # Use advanced indexing only for valid action indices
                        legal_children_mask[valid_actions] = legal_mask[child_actions[valid_actions]]
                    
                    # Only keep legal children
                    if legal_children_mask.any():
                        children = children[legal_children_mask]
                        actions = actions[legal_children_mask] if actions is not None else None
                        priors = priors[legal_children_mask]
                        
                        # Debug logging for illegal child filtering
                        if not legal_children_mask.all():
                            num_illegal = (~legal_children_mask).sum().item()
                            pass  # Illegal children filtered
                    else:
                        # All children are illegal - treat as no children
                        logger.warning(f"All children at node {node_idx_val} have illegal actions - treating as leaf")
                        children = torch.tensor([], dtype=torch.int32, device=self.device)
                        actions = torch.tensor([], dtype=torch.int32, device=self.device)
                        priors = torch.tensor([], dtype=torch.float32, device=self.device)
                
                # This node has legal children now
                # Use effective visits/values that include virtual losses
                effective_visits = self.tree.node_data.get_effective_visits(children).float()
                effective_values = self.tree.node_data.get_effective_values(children)
                
                # Calculate Q-values using effective visits
                q_values = torch.where(
                    effective_visits > 0,
                    effective_values / effective_visits,
                    torch.zeros_like(effective_values)
                )
                
                # Apply per-simulation Dirichlet noise if at root
                noised_priors = self.apply_per_simulation_dirichlet_noise(
                    node_idx_val, sim_indices, children, priors
                )
                
                # Compute UCB for each simulation with its own noised priors
                parent_visits = self.tree.node_data.visit_counts[node_idx_val]
                parent_visits_sqrt = torch.sqrt(torch.maximum(parent_visits.float(), torch.ones(1, device=self.device)))
                
                # Expand visits and q-values to match simulation count
                visits_expanded = effective_visits.unsqueeze(0).expand(len(sim_indices), -1)
                q_values_expanded = q_values.unsqueeze(0).expand(len(sim_indices), -1)
                
                # UCB formula with per-simulation priors
                exploration = self.config.c_puct * noised_priors * parent_visits_sqrt / (1 + visits_expanded)
                ucb = q_values_expanded + exploration
                
                # Each simulation selects its best child based on its own UCB scores
                best_indices = ucb.argmax(dim=1)
                selected_children = children[best_indices]
                
                # Apply virtual losses to selected children
                if len(selected_children) > 0:
                    self.tree.node_data.apply_virtual_loss(selected_children)
                
                # Update final nodes and paths for these simulations
                for i, (sim_idx, child) in enumerate(zip(sim_indices, selected_children)):
                    final_nodes[sim_idx] = child
                    
                    # Update path to include the selected child
                    current_length = updated_lengths[sim_idx].item()
                    if current_length < updated_paths.shape[1]:
                        updated_paths[sim_idx, current_length] = child
                        updated_lengths[sim_idx] = current_length + 1
                
        return final_nodes, updated_paths, updated_lengths
    
    def _parallel_select_with_virtual_loss(self, active_indices: torch.Tensor, active_nodes: torch.Tensor, 
                                         batch_children: torch.Tensor, batch_priors: torch.Tensor,
                                         valid_children_mask: torch.Tensor, depth: int) -> torch.Tensor:
        """Perform parallel selection with proper virtual loss synchronization
        
        This method maintains GPU parallelization while ensuring virtual losses are applied
        correctly when multiple simulations are at the same parent node.
        
        Args:
            active_indices: Indices of active simulations
            active_nodes: Current node for each active simulation
            batch_children: Children for each active node
            batch_priors: Priors for each child
            valid_children_mask: Mask for valid children
            depth: Current depth in the search
            
        Returns:
            Selected children for each active simulation (-1 if no valid children)
        """
        if len(active_indices) == 0:
            return None
            
        # Initialize result tensor
        selected_children = torch.full_like(active_indices, -1, dtype=torch.int32)
        
        # Group simulations by parent node to handle virtual losses correctly
        unique_nodes, node_indices = torch.unique(active_nodes, return_inverse=True)
        
        # Process each unique parent node
        for node_idx, parent_node in enumerate(unique_nodes):
            # Find simulations at this parent node
            sims_at_node = active_indices[node_indices == node_idx]
            
            if len(sims_at_node) == 0:
                continue
                
            # Get children and priors for this node
            node_children = batch_children[node_indices == node_idx][0]  # All same for same parent
            node_priors = batch_priors[node_indices == node_idx][0]
            node_valid_mask = valid_children_mask[node_indices == node_idx][0]
            
            if not node_valid_mask.any():
                continue
                
            # Get valid children for this node
            valid_children = node_children[node_valid_mask]
            valid_priors = node_priors[node_valid_mask]
            
            # CRITICAL FIX: Filter out children whose actions are no longer legal
            # This prevents illegal moves from being selected by UCB
            state_idx = self.node_to_state[parent_node].item()
            if state_idx >= 0:
                # Get current legal moves for this state
                legal_mask = self.game_states.get_legal_moves_mask(
                    torch.tensor([state_idx], device=self.device)
                )[0]
                
                # Vectorized legal action filtering
                child_actions = self.tree.node_data.parent_actions[valid_children]
                valid_actions = (child_actions >= 0) & (child_actions < legal_mask.shape[0])
                legal_children_mask = torch.zeros(len(valid_children), dtype=torch.bool, device=self.device)
                if valid_actions.any():
                    # Use advanced indexing only for valid action indices
                    legal_children_mask[valid_actions] = legal_mask[child_actions[valid_actions]]
                
                # Only keep legal children
                if legal_children_mask.any():
                    valid_children = valid_children[legal_children_mask]
                    valid_priors = valid_priors[legal_children_mask]
                    
                    # Debug logging for illegal child filtering
                    if not legal_children_mask.all():
                        num_illegal = (~legal_children_mask).sum().item()
                        pass  # Illegal children filtered
                else:
                    # All children are illegal - skip this node
                    logger.warning(f"All children at node {parent_node} have illegal actions - skipping parallel selection")
                    continue
            
            # CRITICAL FIX: Apply virtual losses BETWEEN each simulation's selection
            # This ensures proper exploration by preventing collision on same nodes
            
            # Get parent visits once for this node
            parent_visits = self.tree.node_data.visit_counts[parent_node].float()
            parent_visits_sqrt = torch.sqrt(torch.maximum(parent_visits, torch.ones(1, device=self.device)))
            
            # Process each simulation sequentially within this parent node
            for sim_idx in sims_at_node:
                # Get current effective visits/values (including virtual losses from previous selections)
                effective_visits = self.tree.node_data.get_effective_visits(valid_children).float()
                effective_values = self.tree.node_data.get_effective_values(valid_children)
                
                # Calculate Q-values with current virtual losses
                q_values = torch.where(
                    effective_visits > 0,
                    effective_values / effective_visits,
                    torch.zeros_like(effective_values)
                )
                
                # Apply GLOBAL Dirichlet noise for root (standard AlphaZero approach)
                if parent_node == 0 and self.config.dirichlet_epsilon > 0:
                    # Use global noise shared by all simulations at this node
                    if not hasattr(self, '_global_noise_cache') or self._global_noise_cache is None:
                        # Generate global noise once for this root
                        noise = torch.distributions.Dirichlet(
                            torch.full((len(valid_children),), self.config.dirichlet_alpha, device=self.device)
                        ).sample()
                        self._global_noise_cache = (1 - self.config.dirichlet_epsilon) * valid_priors + self.config.dirichlet_epsilon * noise
                    noised_priors = self._global_noise_cache
                else:
                    # No noise for non-root nodes
                    noised_priors = valid_priors
                
                # Calculate UCB scores for this simulation
                exploration = (self.config.c_puct * noised_priors * 
                             parent_visits_sqrt / (1 + effective_visits))
                ucb_scores = q_values + exploration
                
                # This simulation selects its best child
                best_idx = ucb_scores.argmax().item()
                selected_child = valid_children[best_idx]
                
                # CRITICAL: Apply virtual loss immediately after selection
                # This affects the UCB calculation for the next simulation
                self.tree.node_data.apply_virtual_loss(selected_child.unsqueeze(0))
                
                # Store selection
                selected_children[active_indices == sim_idx] = selected_child
        
        return selected_children
        
    def _evaluate_batch_vectorized(self, nodes: torch.Tensor) -> torch.Tensor:
        """Evaluate nodes using neural network
        
        Args:
            nodes: Tensor of node indices to evaluate
            
        Returns:
            Tensor of value estimates
        """
        # Ensure node_to_state is available
        if self.node_to_state is None:
            # For testing, create a dummy mapping if not initialized
            logger.warning("node_to_state not initialized, creating dummy mapping for testing")
            self.node_to_state = torch.arange(self.tree.num_nodes + 1000, device=self.device)
            
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
                    # Check if evaluator supports direct tensor evaluation
                    if hasattr(self.evaluator, '_return_torch_tensors') and self.evaluator._return_torch_tensors:
                        # Direct GPU evaluation - no CPU transfers
                        if hasattr(self.evaluator, 'evaluate_batch'):
                            policies, value_preds = self.evaluator.evaluate_batch(features)
                        else:
                            # This path should rarely be used with optimized evaluator
                            logger.warning("Evaluator doesn't support batch evaluation with tensors")
                            # Convert to numpy as fallback
                            features_np = features.cpu().numpy()
                            policies = []
                            value_preds = []
                            for feat in features_np:
                                p, v = self.evaluator.evaluate(feat)
                                policies.append(p)
                                value_preds.append(v)
                            policies = np.array(policies)
                            value_preds = torch.from_numpy(np.array(value_preds)).to(self.device).float()
                    else:
                        # Legacy path: Convert to numpy for evaluator
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
                        
                        # Convert back to tensor
                        value_preds = torch.from_numpy(value_preds).to(self.device).float()
                    
                # Ensure tensor format
                if isinstance(value_preds, torch.Tensor):
                    value_preds = value_preds.float()  # Ensure float dtype
                else:
                    # Should not happen with properly configured evaluator
                    logger.error("Value predictions are not tensors after evaluation")
                    value_preds = torch.tensor(value_preds, device=self.device, dtype=torch.float32)
                
                # Squeeze to remove extra dimension if present (batch_size, 1) -> (batch_size,)
                if value_preds.ndim == 2 and value_preds.shape[1] == 1:
                    value_preds = value_preds.squeeze(1)
                
                # Debug logging
                # Value predictions processed
                
                # Get the final indices where we should store values
                # valid_mask tells us which nodes were valid
                # valid_states_mask tells us which of those valid nodes had valid states
                valid_node_indices = valid_mask.nonzero(as_tuple=True)[0]
                final_indices = valid_node_indices[valid_states_mask]
                
                # Ensure value_preds has the right shape
                if value_preds.ndim > 1:
                    value_preds = value_preds.squeeze(-1)
                
                # Check shape compatibility
                if len(final_indices) != len(value_preds):
                    logger.error(f"Shape mismatch: final_indices={len(final_indices)}, value_preds={len(value_preds)}")
                    # Try to handle common case where evaluator returns fixed batch size
                    if len(value_preds) > len(final_indices):
                        value_preds = value_preds[:len(final_indices)]
                    else:
                        raise RuntimeError(f"Value predictions ({len(value_preds)}) don't match valid states ({len(final_indices)})")
                
                # Debug shapes before assignment
                # Assign value predictions
                
                # Ensure value_preds is 1D for assignment
                if value_preds.ndim == 2 and value_preds.shape[1] == 1:
                    value_preds = value_preds.squeeze(1)
                    
                values[final_indices] = value_preds
                
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
                
        # RAVE updates if enabled
        if self.config.enable_rave:
            self._update_rave_statistics(paths, path_lengths, values)
    
    def _remove_virtual_losses_from_paths(self, paths: torch.Tensor, path_lengths: torch.Tensor):
        """Remove virtual losses from nodes that were selected in this wave
        
        CRITICAL FIX: Only remove virtual losses from nodes that had selections applied,
        and only remove the exact number of virtual losses that were applied.
        
        Args:
            paths: Tensor of paths through the tree
            path_lengths: Length of each path
        """
        if not self.config.enable_virtual_loss:
            return
            
        # Track how many virtual losses to remove from each node
        # Each path represents one simulation that applied virtual losses during selection
        virtual_loss_counts = {}
        
        for i in range(paths.shape[0]):
            path_len = path_lengths[i].item()
            # Skip the root node (index 0) and only count selection nodes
            for j in range(1, path_len):  # Start from 1 to skip root
                node = paths[i, j].item()
                if node >= 0 and node < self.tree.num_nodes:
                    virtual_loss_counts[node] = virtual_loss_counts.get(node, 0) + 1
        
        # Remove the appropriate number of virtual losses from each node
        for node, count in virtual_loss_counts.items():
            node_tensor = torch.tensor([node], device=self.device)
            for _ in range(count):
                self.tree.node_data.remove_virtual_loss(node_tensor)
    
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
        # Signs should alternate based on distance from leaf (reverse order)
        signs = torch.ones((batch_size, max_length), device=self.device)
        for i in range(batch_size):
            path_len = lengths[i].item()
            # Create alternating pattern from leaf backwards
            for j in range(path_len):
                if (path_len - 1 - j) % 2 == 1:
                    signs[i, j] = -1
        
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
        
