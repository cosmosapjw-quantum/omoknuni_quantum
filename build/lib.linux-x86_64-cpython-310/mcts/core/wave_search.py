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
        self.paths_buffer = torch.empty(0, device=self.device)  # Initialize empty tensor
        
        # State management - will be set during run_wave
        self.node_to_state = None
        self.state_pool_free_list = None
        
        # Store original root priors for Dirichlet noise mixing
        self.original_root_priors = None
        
        # Cache for batched Dirichlet noise generation
        self._dirichlet_cache = {}
        
        # Progressive widening parameters
        self.pw_alpha = getattr(config, 'progressive_widening_alpha', 0.5)  # k = alpha * sqrt(n)
        self.pw_base = getattr(config, 'progressive_widening_base', 10.0)   # Minimum children
        
        # Initialize tactical detector based on game type if enabled
        self._tactical_detector = None
        if config.enable_tactical_boost:
            try:
                if config.game_type == GameType.GO:
                    from ..utils.go_tactical_detector import GoTacticalMoveDetector
                    self._tactical_detector = GoTacticalMoveDetector(config.board_size, config)
                    logger.debug("Initialized tactical move detector for Go")
                elif config.game_type == GameType.CHESS:
                    from ..utils.chess_tactical_detector import ChessTacticalMoveDetector
                    self._tactical_detector = ChessTacticalMoveDetector(config)
                    logger.debug("Initialized tactical move detector for Chess")
                elif config.game_type == GameType.GOMOKU:
                    from ..utils.gomoku_tactical_detector import GomokuTacticalMoveDetector
                    self._tactical_detector = GomokuTacticalMoveDetector(config.board_size, config)
                    logger.debug("Initialized tactical move detector for Gomoku")
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
        # Progressive widening: expand more children as parent gets more visits
        # Formula: min(num_legal_moves, base + alpha * sqrt(parent_visits))
        import math
        max_children = int(self.pw_base + self.pw_alpha * math.sqrt(max(1, parent_visits)))
        
        # For root node on first expansion, be more aggressive
        if parent_visits == 0:
            max_children = min(int(self.pw_base * 2), num_legal_moves)
        
        return min(max_children, num_legal_moves)
        
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
            
            # DEBUG: Log expansion check
            if node_idx_val == 0:  # Root node
                logger.debug(f"Checking root expansion: has {len(children)} children")
            
            if len(children) == 0:
                state_idx = current_node_to_state[node_idx].item()
                if state_idx >= 0:
                    # CRITICAL: Check if state is terminal before expanding
                    is_terminal = self.game_states.is_terminal[state_idx].item()
                    if not is_terminal:
                        needs_expansion[i] = True
                        state_indices[i] = state_idx
                    else:
                        logger.debug(f"Skipping expansion of terminal node {node_idx_val} (state {state_idx})")
        
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
                    
                    # Apply tactical boost if enabled for any game
                    if (self.config.enable_tactical_boost and
                        hasattr(self, '_tactical_detector') and
                        self._tactical_detector is not None):
                        # Get board state
                        board = self.game_states.boards[state_idx]
                        current_player = self.game_states.current_player[state_idx].item()
                        
                        # Debug: log board state
                        if node_idx.item() == 0:
                            logger.debug(f"Board state for tactical detection:")
                            logger.debug(f"Current player: {current_player}")
                            for y in range(min(5, board.shape[0])):  # Show first 5 rows
                                row = []
                                for x in range(min(5, board.shape[1])):  # First 5 cols
                                    val = board[y, x].item()
                                    if val == 0:
                                        row.append('.')
                                    elif val == 1:
                                        row.append('B')
                                    else:
                                        row.append('W')
                                logger.debug(f"  {' '.join(row)}")
                        
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
                            logger.debug(f"Board shape: {board.shape}, current player: {current_player}")
                            
                            # Check boost values
                            boost_values = self._tactical_detector.detect_tactical_moves(board, current_player)
                            logger.debug(f"Max boost value: {boost_values.max():.4f}")
                            
                            # Log the actual boost calculation
                            capture_idx = 29  # Capture move at (3,2)
                            if capture_idx in legal_actions:
                                idx_in_legal = (legal_actions == capture_idx).nonzero(as_tuple=True)[0]
                                if len(idx_in_legal) > 0:
                                    idx = idx_in_legal[0]
                                    logger.debug(f"Capture move boost: {boost_values[capture_idx]:.4f}")
                                    logger.debug(f"Capture move prior: {old_priors[idx]:.6f} -> {priors[idx]:.6f}")
                                    logger.debug(f"Full priors before boost: min={full_priors.min():.6f}, max={full_priors.max():.6f}, mean={full_priors.mean():.6f}")
                                    logger.debug(f"Boosted priors: min={boosted_priors.min():.6f}, max={boosted_priors.max():.6f}, mean={boosted_priors.mean():.6f}")
                    
                    # CRITICAL: Implement progressive widening to avoid eager expansion
                    # Only expand a subset of moves based on visit count
                    parent_visits = self.tree.node_data.visit_counts[node_idx].item()
                    
                    # Progressive widening formula: k * sqrt(n) where n is parent visits
                    # Start with a small number and grow
                    max_children = self._get_max_children_for_expansion(parent_visits, len(legal_actions))
                    
                    if max_children < len(legal_actions):
                        # Select moves for progressive widening
                        logger.debug(f"Progressive widening: Node {node_idx} with {parent_visits} visits, "
                                   f"expanding {max_children}/{len(legal_actions)} children")
                        
                        # For games with tactical boost, ensure tactical moves are included
                        if (self.config.enable_tactical_boost and
                            hasattr(self, '_tactical_detector') and
                            self._tactical_detector is not None):
                            # Get tactical boost values for all legal actions
                            board = self.game_states.boards[state_idx]
                            current_player = self.game_states.current_player[state_idx].item()
                            
                            if self.config.game_type == GameType.CHESS:
                                # Chess needs legal moves list
                                boost_values = self._tactical_detector.detect_tactical_moves(
                                    board, current_player, legal_actions.cpu().tolist()
                                )
                            else:
                                # Go and Gomoku work with board positions
                                boost_values = self._tactical_detector.detect_tactical_moves(board, current_player)
                            
                            # Get boost values for legal actions
                            # Ensure both tensors are on same device (CPU) for indexing
                            legal_actions_cpu = legal_actions.cpu() if legal_actions.is_cuda else legal_actions
                            legal_boost = boost_values[legal_actions_cpu]
                            
                            # Convert to tensor on same device as other tensors if needed
                            if not isinstance(legal_boost, torch.Tensor):
                                legal_boost = torch.tensor(legal_boost, device=legal_actions.device)
                            elif legal_boost.device != legal_actions.device:
                                legal_boost = legal_boost.to(legal_actions.device)
                            
                            # Find high-value tactical moves (boost > threshold)
                            tactical_threshold = 5.0  # Captures typically have boost > 10
                            tactical_mask = legal_boost > tactical_threshold
                            num_tactical = tactical_mask.sum().item()
                            
                            if num_tactical > 0:
                                # Ensure all tactical moves are included
                                tactical_indices = torch.where(tactical_mask)[0]
                                non_tactical_indices = torch.where(~tactical_mask)[0]
                                
                                # How many non-tactical moves can we include?
                                remaining_slots = max(0, max_children - num_tactical)
                                
                                if remaining_slots > 0 and len(non_tactical_indices) > 0:
                                    # Select from non-tactical moves
                                    if remaining_slots < len(non_tactical_indices):
                                        # Random selection for non-tactical moves
                                        perm = torch.randperm(len(non_tactical_indices), device=priors.device)
                                        selected_non_tactical = non_tactical_indices[perm[:remaining_slots]]
                                    else:
                                        selected_non_tactical = non_tactical_indices
                                    
                                    # Combine tactical and non-tactical
                                    selected_indices = torch.cat([tactical_indices, selected_non_tactical])
                                else:
                                    # Only tactical moves (or all if we have space)
                                    selected_indices = tactical_indices[:max_children]
                                
                                logger.debug(f"Including {num_tactical} tactical moves in progressive widening")
                            else:
                                # No tactical moves, use standard selection
                                prior_std = priors.std()
                                prior_mean = priors.mean()
                                
                                if prior_std < prior_mean * 0.01:  # 1% tolerance
                                    # Use random selection for uniform priors
                                    perm = torch.randperm(len(priors), device=priors.device)
                                    selected_indices = perm[:max_children]
                                else:
                                    # Use topk for non-uniform priors
                                    selected_indices = torch.topk(priors, min(max_children, len(priors))).indices
                        else:
                            # Standard progressive widening for non-Go games
                            prior_std = priors.std()
                            prior_mean = priors.mean()
                            
                            if prior_std < prior_mean * 0.01:  # 1% tolerance
                                # Use random selection for uniform priors
                                perm = torch.randperm(len(priors), device=priors.device)
                                selected_indices = perm[:max_children]
                            else:
                                # Use topk for non-uniform priors
                                selected_indices = torch.topk(priors, min(max_children, len(priors))).indices
                        
                        legal_actions = legal_actions[selected_indices]
                        priors = priors[selected_indices]
                        priors = priors / priors.sum()  # Re-normalize
                    
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
                
                # Add RAVE if enabled
                if self.config.enable_rave and hasattr(self.tree.node_data, 'rave_visits'):
                    # Get RAVE statistics for this node
                    if node_idx_val in self.tree.node_data.rave_visits:
                        # Get actions for children
                        _, actions, _ = self.tree.get_children(node_idx_val)
                        
                        # Compute RAVE values
                        rave_visits = torch.zeros(len(children), device=self.device)
                        rave_values = torch.zeros(len(children), device=self.device)
                        
                        for i, action in enumerate(actions):
                            action_idx = action.item()
                            if action_idx < len(self.tree.node_data.rave_visits[node_idx_val]):
                                rave_visits[i] = self.tree.node_data.rave_visits[node_idx_val][action_idx]
                                rave_values[i] = self.tree.node_data.rave_values[node_idx_val][action_idx]
                        
                        # Compute RAVE Q-values
                        rave_q = rave_values / (rave_visits + 1e-8)
                        
                        # Beta scheduling: β = sqrt(rave_c / (rave_c + n))
                        beta = torch.sqrt(self.config.rave_c / (self.config.rave_c + parent_visits.float()))
                        
                        # Expand for simulations
                        rave_q_expanded = rave_q.unsqueeze(0).expand(len(sim_indices), -1)
                        
                        # Combine UCB and RAVE
                        ucb = (1 - beta) * ucb + beta * rave_q_expanded
                
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
        
        # RAVE updates if enabled
        if self.config.enable_rave:
            self._update_rave_statistics(paths, lengths, values)
    
    def _update_rave_statistics(self, paths: torch.Tensor, path_lengths: torch.Tensor, values: torch.Tensor):
        """Update RAVE statistics using All-Moves-As-First (AMAF) logic
        
        For each node in the path, update RAVE statistics for all actions
        that were played later in the simulation.
        
        Args:
            paths: Tensor of paths through tree [batch_size, max_path_length]
            path_lengths: Length of each path [batch_size]
            values: Value estimates to backup [batch_size]
        """
        batch_size = paths.shape[0]
        
        # Process each simulation path
        for i in range(batch_size):
            path_len = path_lengths[i].item()
            if path_len <= 1:  # Need at least 2 nodes for RAVE
                continue
                
            value = values[i].item()
            
            # Get the path for this simulation
            path = paths[i, :path_len]
            
            # For each node in the path (except the last one)
            for j in range(path_len - 1):
                node_idx = path[j].item()
                if node_idx < 0 or node_idx >= self.tree.num_nodes:
                    continue
                
                # Get the action taken from this node (the next node in path)
                next_node_idx = path[j + 1].item()
                if next_node_idx < 0 or next_node_idx >= self.tree.num_nodes:
                    continue
                
                # Find the action that leads to next_node_idx
                action = self._get_action_for_child(node_idx, next_node_idx)
                if action is None:
                    continue
                
                # Initialize RAVE statistics for this node if needed
                if hasattr(self.tree.node_data, 'initialize_rave_for_node'):
                    # Get action space size from the game state
                    try:
                        if self.node_to_state is not None and node_idx < len(self.node_to_state):
                            state_idx = self.node_to_state[node_idx]
                            if state_idx >= 0:
                                action_space_size = self.game_states.get_action_space_size(state_idx)
                                self.tree.node_data.initialize_rave_for_node(node_idx, action_space_size)
                    except:
                        # Fallback to default action space size
                        action_space_size = 64 * 64 + 1  # Go-style board + pass
                        self.tree.node_data.initialize_rave_for_node(node_idx, action_space_size)
                
                # Update RAVE statistics for this action
                # Value should be from the perspective of the player to move at this node
                # Calculate the perspective adjustment based on depth
                perspective_value = value if (path_len - 1 - j) % 2 == 0 else -value
                
                if hasattr(self.tree.node_data, 'update_rave'):
                    self.tree.node_data.update_rave(node_idx, action, perspective_value)
    
    def _get_action_for_child(self, parent_idx: int, child_idx: int) -> Optional[int]:
        """Get the action that leads from parent to child node
        
        Args:
            parent_idx: Index of parent node
            child_idx: Index of child node
            
        Returns:
            Action index if found, None otherwise
        """
        # This requires access to the tree structure to map child nodes back to actions
        # For now, we'll use the parent_actions field in node_data if available
        if hasattr(self.tree.node_data, 'parent_actions'):
            try:
                return self.tree.node_data.parent_actions[child_idx].item()
            except:
                return None
        return None