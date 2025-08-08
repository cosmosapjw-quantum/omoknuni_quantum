"""
Hybrid Wave Search - Optimized wave-based search for CPU-GPU coordination

This implementation properly handles device placement and provides efficient
batch processing across CPU and GPU devices.

Key Features:
1. Proper device management - no tensor device mismatches
2. Efficient batch evaluation with minimal CPU-GPU transfers
3. Asynchronous execution where possible
4. Smart work distribution based on operation characteristics
"""

import torch
import numpy as np
import logging
import time
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class HybridWaveSearch:
    """Wave search optimized for hybrid CPU-GPU execution"""
    
    def __init__(self, tree, game_states, evaluator, config, device='cuda'):
        self.tree = tree
        self.game_states = game_states
        self.evaluator = evaluator
        self.config = config
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Determine actual devices for components
        self.tree_device = self._get_tree_device()
        self.game_states_device = self._get_game_states_device()
        self.eval_device = self.device  # Evaluator always on main device
        
        logger.info(f"HybridWaveSearch initialized - tree: {self.tree_device}, "
                   f"states: {self.game_states_device}, eval: {self.eval_device}")
        
        # Wave configuration
        self.max_wave_size = getattr(config, 'max_wave_size', 256)
        self.min_batch_size = getattr(config, 'min_batch_size', 32)
        
        # Virtual loss for parallel exploration
        self.virtual_loss = getattr(config, 'virtual_loss', 3.0)
        self.use_virtual_loss = getattr(config, 'enable_virtual_loss', True)
        
        # Dirichlet noise for exploration
        self.dirichlet_alpha = getattr(config, 'dirichlet_alpha', 0.03)
        self.dirichlet_epsilon = getattr(config, 'dirichlet_epsilon', 0.25)
        self.add_dirichlet_noise = getattr(config, 'add_dirichlet_noise', True)
        
        # Performance optimization
        self.use_cuda_streams = (self.device.type == 'cuda' and 
                                getattr(config, 'use_cuda_streams', True))
        if self.use_cuda_streams:
            self.eval_stream = torch.cuda.Stream()
            self.transfer_stream = torch.cuda.Stream()
        else:
            self.eval_stream = None
            self.transfer_stream = None
            
        # Batch accumulation for efficiency
        self.batch_accumulator = []
        self.batch_paths = []
        
        # Statistics
        self.total_simulations = 0
        self.total_expansions = 0
        
        # Thread pool for CPU operations
        self.cpu_workers = ThreadPoolExecutor(max_workers=4)
        
    def _get_tree_device(self) -> str:
        """Determine the device where tree is stored"""
        if hasattr(self.tree, 'device'):
            return str(self.tree.device)
        elif hasattr(self.tree, 'visit_counts'):
            if isinstance(self.tree.visit_counts, torch.Tensor):
                return str(self.tree.visit_counts.device)
        return 'cpu'  # Default assumption
        
    def _get_game_states_device(self) -> str:
        """Determine the device where game states are stored"""
        if hasattr(self.game_states, 'device'):
            return str(self.game_states.device)
        elif hasattr(self.game_states, '_boards'):
            if isinstance(self.game_states._boards, torch.Tensor):
                return str(self.game_states._boards.device)
        return 'cpu'  # Default assumption
        
    def reset_search_state(self):
        """Reset search state for new search"""
        self.total_simulations = 0
        self.total_expansions = 0
        self.batch_accumulator.clear()
        self.batch_paths.clear()
        
    def run_wave(self, wave_size: int, node_to_state: Any) -> int:
        """Run a single wave of simulations
        
        Returns:
            Number of simulations completed
        """
        # Collect paths for this wave
        paths = []
        leaf_nodes = []
        
        # Phase 1: Selection (CPU-friendly operation)
        for _ in range(wave_size):
            path, leaf = self._select_path()
            if leaf is not None:
                paths.append(path)
                leaf_nodes.append(leaf)
                
        if not paths:
            return 0
            
        # Phase 2: Batch expansion and evaluation
        expanded_count = self._batch_expand_and_evaluate(
            paths, leaf_nodes, node_to_state
        )
        
        self.total_simulations += len(paths)
        self.total_expansions += expanded_count
        
        return len(paths)
        
    def _select_path(self) -> Tuple[List[int], Optional[int]]:
        """Select a path from root to leaf using UCB
        
        Returns:
            (path, leaf_node) where path is list of node indices
        """
        path = []
        current = 0  # Start at root
        
        # Apply virtual loss if enabled
        if self.use_virtual_loss:
            virtual_path = []
            
        while True:
            path.append(current)
            
            # Apply virtual loss
            if self.use_virtual_loss:
                virtual_path.append(current)
                self._apply_virtual_loss(current)
                
            # Get children
            children = self._get_children(current)
            
            if len(children) == 0:
                # Leaf node - needs expansion
                break
                
            # Check if any child is unexplored
            unexplored = self._get_unexplored_children(current, children)
            if len(unexplored) > 0:
                # Select random unexplored child
                current = unexplored[torch.randint(len(unexplored), (1,)).item()]
                path.append(current)
                break
                
            # All children explored - select by UCB
            current = self._select_child_ucb(current, children)
            
        # Store virtual path for later removal
        if self.use_virtual_loss:
            self.virtual_paths = getattr(self, 'virtual_paths', [])
            self.virtual_paths.append(virtual_path)
            
        return path, current
        
    def _get_children(self, node_idx: int) -> torch.Tensor:
        """Get children of a node, handling device placement"""
        if hasattr(self.tree, 'get_children'):
            children, _, _ = self.tree.get_children(node_idx)
            # Ensure tensor is accessible from CPU for selection logic
            if isinstance(children, torch.Tensor) and children.is_cuda:
                children = children.cpu()
            return children
        else:
            # Fallback for custom tree implementations
            return torch.tensor([], dtype=torch.int32)
            
    def _get_unexplored_children(self, parent: int, children: torch.Tensor) -> torch.Tensor:
        """Get unexplored children (visit count = 0)"""
        if len(children) == 0:
            return torch.tensor([], dtype=torch.int32)
            
        # Get visit counts for children
        if hasattr(self.tree, 'get_visit_counts'):
            visits = self.tree.get_visit_counts(children)
        else:
            # Direct access to visit counts
            visits = self.tree.visit_counts[children]
            
        # Move to CPU for logic
        if isinstance(visits, torch.Tensor) and visits.is_cuda:
            visits = visits.cpu()
            
        return children[visits == 0]
        
    def _select_child_ucb(self, parent: int, children: torch.Tensor) -> int:
        """Select child using UCB formula"""
        if len(children) == 0:
            return parent
            
        # Get node statistics
        parent_visits = self._get_visit_count(parent)
        if parent_visits == 0:
            # Shouldn't happen, but handle gracefully
            return children[0].item()
            
        # Calculate UCB scores for all children
        ucb_scores = self._calculate_ucb_scores(parent, children, parent_visits)
        
        # Add Dirichlet noise at root
        if parent == 0 and self.add_dirichlet_noise:
            noise = torch.from_numpy(
                np.random.dirichlet([self.dirichlet_alpha] * len(children))
            ).float()
            ucb_scores = (1 - self.dirichlet_epsilon) * ucb_scores + \
                        self.dirichlet_epsilon * noise
                        
        # Select best child
        best_idx = torch.argmax(ucb_scores).item()
        return children[best_idx].item()
        
    def _calculate_ucb_scores(self, parent: int, children: torch.Tensor, 
                             parent_visits: int) -> torch.Tensor:
        """Calculate UCB scores for children"""
        c_puct = getattr(self.config, 'c_puct', 1.4)
        
        # Get statistics for children
        if hasattr(self.tree, 'get_children_stats'):
            visits, values, priors = self.tree.get_children_stats(parent, children)
        else:
            # Direct access
            visits = self.tree.visit_counts[children].float()
            values = self.tree.value_sums[children]
            priors = self._get_priors(parent, children)
            
        # Move to CPU for calculation
        if visits.is_cuda:
            visits = visits.cpu()
            values = values.cpu()
            priors = priors.cpu()
            
        # Calculate Q values
        q_values = torch.where(visits > 0, values / visits, torch.zeros_like(values))
        
        # Calculate exploration term
        sqrt_parent = np.sqrt(parent_visits)
        exploration = c_puct * priors * sqrt_parent / (visits + 1)
        
        return q_values + exploration
        
    def _get_visit_count(self, node_idx: int) -> int:
        """Get visit count for a node"""
        if hasattr(self.tree, 'get_visit_count'):
            return self.tree.get_visit_count(node_idx)
        else:
            count = self.tree.visit_counts[node_idx]
            if isinstance(count, torch.Tensor):
                return count.item()
            return int(count)
            
    def _get_priors(self, parent: int, children: torch.Tensor) -> torch.Tensor:
        """Get prior probabilities for children"""
        # This would access the stored priors from tree
        # For now, return uniform priors
        return torch.ones(len(children)) / len(children)
        
    def _apply_virtual_loss(self, node_idx: int):
        """Apply virtual loss to a node"""
        if hasattr(self.tree, 'apply_virtual_loss'):
            self.tree.apply_virtual_loss(node_idx, self.virtual_loss)
        else:
            # Direct modification
            self.tree.visit_counts[node_idx] += 1
            self.tree.value_sums[node_idx] -= self.virtual_loss
            
    def _remove_virtual_loss(self, node_idx: int):
        """Remove virtual loss from a node"""
        if hasattr(self.tree, 'remove_virtual_loss'):
            self.tree.remove_virtual_loss(node_idx, self.virtual_loss)
        else:
            # Direct modification
            self.tree.visit_counts[node_idx] -= 1
            self.tree.value_sums[node_idx] += self.virtual_loss
            
    def _batch_expand_and_evaluate(self, paths: List[List[int]], 
                                   leaf_nodes: List[int],
                                   node_to_state: Any) -> int:
        """Batch expand nodes and evaluate with neural network
        
        Returns:
            Number of nodes expanded
        """
        if not paths:
            return 0
            
        # Collect states that need evaluation
        states_to_eval = []
        eval_indices = []
        
        for i, (path, leaf) in enumerate(zip(paths, leaf_nodes)):
            # Get or create state for leaf
            state_idx = self._get_state_for_node(leaf, path, node_to_state)
            if state_idx is not None:
                states_to_eval.append(state_idx)
                eval_indices.append(i)
                
        if not states_to_eval:
            return 0
            
        # Get features for neural network
        features = self._get_features_batch(states_to_eval)
        
        # Transfer to evaluation device if needed
        if features.device != self.eval_device:
            if self.use_cuda_streams and self.eval_device.type == 'cuda':
                with torch.cuda.stream(self.transfer_stream):
                    features = features.to(self.eval_device, non_blocking=True)
            else:
                features = features.to(self.eval_device)
                
        # Evaluate with neural network
        if self.use_cuda_streams and self.eval_device.type == 'cuda':
            with torch.cuda.stream(self.eval_stream):
                policies, values = self.evaluator.evaluate_batch(features)
                # Ensure computation completes
                self.eval_stream.synchronize()
        else:
            policies, values = self.evaluator.evaluate_batch(features)
            
        # Expand nodes and backup values
        expanded = 0
        for idx, path_idx in enumerate(eval_indices):
            path = paths[path_idx]
            leaf = leaf_nodes[path_idx]
            policy = policies[idx]
            value = values[idx].item()
            
            # Expand node if needed
            if self._should_expand(leaf):
                self._expand_node(leaf, policy, states_to_eval[idx])
                expanded += 1
                
            # Backup value
            self._backup_path(path, value)
            
            # Remove virtual loss
            if self.use_virtual_loss and hasattr(self, 'virtual_paths'):
                if path_idx < len(self.virtual_paths):
                    for node in self.virtual_paths[path_idx]:
                        self._remove_virtual_loss(node)
                        
        # Clear virtual paths
        if hasattr(self, 'virtual_paths'):
            self.virtual_paths.clear()
            
        return expanded
        
    def _get_state_for_node(self, node_idx: int, path: List[int], 
                            node_to_state: Any) -> Optional[int]:
        """Get or create game state for a node"""
        # Check if node already has a state
        if node_to_state[node_idx] >= 0:
            return node_to_state[node_idx]
            
        # Need to create state by replaying moves
        if len(path) < 2:
            return None  # Can't create state for root
            
        # Get parent state
        parent_node = path[-2]
        parent_state = node_to_state[parent_node]
        if parent_state < 0:
            return None
            
        # Clone parent state and apply action
        # This would use actual game logic
        # For now, return parent state as placeholder
        return parent_state
        
    def _get_features_batch(self, state_indices: List[int]) -> torch.Tensor:
        """Get neural network features for batch of states"""
        if hasattr(self.game_states, 'get_nn_features'):
            features = self.game_states.get_nn_features(state_indices)
        else:
            # Fallback - create dummy features
            batch_size = len(state_indices)
            board_size = getattr(self.config, 'board_size', 15)
            features = torch.zeros((batch_size, 18, board_size, board_size))
            
        return features
        
    def _should_expand(self, node_idx: int) -> bool:
        """Check if node should be expanded"""
        # Expand if node has no children
        children = self._get_children(node_idx)
        return len(children) == 0
        
    def _expand_node(self, node_idx: int, policy: torch.Tensor, state_idx: int):
        """Expand a node with children based on policy"""
        # Get legal actions for state
        if hasattr(self.game_states, 'get_legal_actions'):
            legal_actions = self.game_states.get_legal_actions(state_idx)
        else:
            # Assume all actions legal for now
            legal_actions = torch.arange(len(policy))
            
        # Filter policy to legal actions
        legal_policy = policy[legal_actions]
        
        # Normalize
        if legal_policy.sum() > 0:
            legal_policy = legal_policy / legal_policy.sum()
        else:
            legal_policy = torch.ones_like(legal_policy) / len(legal_policy)
            
        # Add children to tree
        if hasattr(self.tree, 'expand_node'):
            self.tree.expand_node(node_idx, legal_policy, legal_actions)
            
    def _backup_path(self, path: List[int], value: float):
        """Backup value along path"""
        # Flip value for each level (alternating players)
        current_value = value
        
        for node_idx in reversed(path):
            if hasattr(self.tree, 'update_node'):
                self.tree.update_node(node_idx, current_value)
            else:
                # Direct update
                self.tree.visit_counts[node_idx] += 1
                self.tree.value_sums[node_idx] += current_value
                
            current_value = -current_value  # Flip for opponent
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get search statistics"""
        return {
            'total_simulations': self.total_simulations,
            'total_expansions': self.total_expansions,
            'expansion_rate': (self.total_expansions / max(1, self.total_simulations))
        }
    
    def _expand_batch_vectorized(self, leaf_nodes, node_to_state=None, 
                                device=None, root_noise_cache=None) -> torch.Tensor:
        """Vectorized batch expansion for compatibility with main MCTS
        
        This method is required for integration with the main MCTS class.
        It expands multiple nodes in parallel.
        
        Args:
            leaf_nodes: Tensor of node indices to expand
            node_to_state: Mapping from nodes to game states
            device: Device for tensor operations
            root_noise_cache: Cache for Dirichlet noise (optional)
            
        Returns:
            Tensor of expanded node indices
        """
        if isinstance(leaf_nodes, torch.Tensor):
            leaf_nodes = leaf_nodes.cpu().numpy().tolist()
        elif not isinstance(leaf_nodes, list):
            leaf_nodes = [leaf_nodes]
            
        expanded = []
        
        for node_idx in leaf_nodes:
            # Check if node needs expansion
            if self._should_expand(node_idx):
                # Get state for node
                if node_to_state is not None:
                    state_idx = self._get_state_for_node(
                        node_idx, [node_idx], node_to_state
                    )
                else:
                    state_idx = 0  # Default state
                    
                if state_idx is not None:
                    # Get features and evaluate
                    features = self._get_features_batch([state_idx])
                    
                    # Move to evaluation device if needed
                    if features.device != self.eval_device:
                        features = features.to(self.eval_device)
                        
                    # Evaluate
                    policies, values = self.evaluator.evaluate_batch(features)
                    
                    # Expand node
                    self._expand_node(node_idx, policies[0], state_idx)
                    expanded.append(node_idx)
                    
        # Ensure device is a proper torch.device object
        if device is not None:
            if isinstance(device, list):
                device = device[0] if device else 'cpu'
            if isinstance(device, str):
                device = torch.device(device)
        else:
            device = torch.device('cpu')
            
        if expanded:
            return torch.tensor(expanded, dtype=torch.int32, device=device)
        else:
            return torch.tensor([], dtype=torch.int32, device=device)