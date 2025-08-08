"""Optimized CPU wave search with reduced thread coordination overhead

Phase 1.5 optimization: Replace fine-grained thread pool tasks with
coarse-grained batch processing to eliminate thread coordination bottleneck.

Key optimizations:
1. Batch multiple waves together to reduce task switching
2. Use larger work chunks per thread to amortize coordination costs  
3. Direct synchronous batching instead of many small async tasks
4. Minimize thread pool submissions from 4,140 to ~100 per search
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Any, Dict
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

from .vectorized_operations import batch_ucb_scores


logger = logging.getLogger(__name__)


class OptimizedCPUWaveSearch:
    """CPU wave search optimized for minimal thread coordination overhead"""
    
    def __init__(self, tree: Any, game_states: Any, evaluator: Any, 
                 config: Any, device: torch.device):
        """Initialize optimized CPU wave search
        
        Args:
            tree: CPU tree
            game_states: CPU game states  
            evaluator: Neural network evaluator
            config: MCTS configuration
            device: Device (should be CPU)
        """
        self.tree = tree
        self.game_states = game_states
        self.evaluator = evaluator
        self.config = config
        self.device = device
        
        # Thread configuration - optimized for Cython tree performance
        cpu_threads = getattr(config, 'cpu_threads_per_worker', None)
        if cpu_threads is None:
            # OPTIMIZATION: Single-threaded is faster due to optimized Cython tree
            # Multi-threading creates GIL contention and context switching overhead
            # that outweighs benefits when tree operations are already very fast
            self.num_threads = 1
            logger.info(f"Using optimized single-threaded CPU wave search (eliminates GIL contention)")
        else:
            self.num_threads = cpu_threads
            logger.info(f"Using {self.num_threads} threads for CPU wave search (user override)")
            
        # Batch configuration for reduced task switching
        self.waves_per_batch = 32  # Process many waves per thread task for efficiency
        self.max_eval_batch_size = getattr(config, 'inference_batch_size', 256)
        
        # Pre-allocate reusable arrays
        self.temp_paths = []
        self.temp_values = []
        
        # For compatibility with WaveSearch interface
        self._global_noise_cache = None
        
        # Simple evaluation cache to reduce NN calls
        self._eval_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"OptimizedCPUWaveSearch: {self.num_threads} threads, "
                   f"{self.waves_per_batch} waves/batch")
    
    def reset_search_state(self):
        """Reset search state for a new search"""
        self._global_noise_cache = None
        # Clear evaluation cache for new search
        self._eval_cache.clear()
        if self._cache_hits > 0 or self._cache_misses > 0:
            cache_rate = self._cache_hits / (self._cache_hits + self._cache_misses) * 100 if (self._cache_hits + self._cache_misses) > 0 else 0
            logger.debug(f"Eval cache stats: {self._cache_hits} hits, {self._cache_misses} misses ({cache_rate:.1f}% hit rate)")
        self._cache_hits = 0
        self._cache_misses = 0
        
    def allocate_buffers(self, max_tree_size: int, buffer_size: int, 
                        max_states: int, max_workers: int = 1):
        """Allocate buffers for search (compatibility method)"""
        # OptimizedCPUWaveSearch doesn't need pre-allocated buffers
        # It uses dynamic allocation for better memory efficiency
        pass
        
    def _expand_batch_vectorized(self, leaf_nodes: List[int], 
                               node_to_state=None,
                               state_pool_free_list=None) -> int:
        """Expand batch of nodes (compatibility method)
        
        Args:
            leaf_nodes: List of node indices to expand
            node_to_state: Node to state mapping (if not provided, uses self.node_to_state)
            state_pool_free_list: Unused for CPU backend
            
        Returns:
            Number of nodes expanded
        """
        # Use provided node_to_state or stored value
        if node_to_state is None:
            node_to_state = getattr(self, 'node_to_state', None)
            if node_to_state is None:
                raise RuntimeError("node_to_state not available")
        
        # Convert torch tensor to numpy if needed
        if hasattr(node_to_state, 'cpu'):
            node_to_state = node_to_state.cpu().numpy()
        elif not isinstance(node_to_state, np.ndarray):
            node_to_state = np.array(node_to_state)
            
        # This is handled internally by _batch_expand_and_backup
        paths = [[node] for node in leaf_nodes]  # Simple paths for compatibility
        return self._batch_expand_and_backup(paths, leaf_nodes, node_to_state)
    
    def run_wave(self, wave_size: int, node_to_state, 
                 state_pool_free_list: Optional[List[int]] = None) -> int:
        """Run optimized wave with minimal thread coordination
        
        Args:
            wave_size: Number of simulations in this wave
            node_to_state: Mapping from tree nodes to game states (numpy array or torch tensor)
            state_pool_free_list: Optional free list (for compatibility)
            
        Returns:
            Number of nodes expanded
        """
        if wave_size <= 0:
            return 0
            
        # Convert torch tensor to numpy if needed
        if hasattr(node_to_state, 'cpu'):
            node_to_state = node_to_state.cpu().numpy()
        elif not isinstance(node_to_state, np.ndarray):
            node_to_state = np.array(node_to_state)
            
        # Use parallel processing for better CPU utilization
        if self.num_threads == 1:
            # Use direct execution to avoid thread pool overhead
            return self._run_wave_direct(wave_size, node_to_state)
        else:
            return self._run_wave_batched_parallel(wave_size, node_to_state)
    
    def _run_wave_direct(self, wave_size: int, node_to_state: np.ndarray) -> int:
        """Run wave directly without thread coordination for small waves"""
        expanded_count = 0
        simulation_paths = []
        simulation_values = []
        
        # Collect all simulations
        for _ in range(wave_size):
            path, leaf_node = self._select_path_to_leaf()
            if leaf_node is None:
                continue
                
            children, _, _ = self.tree.get_children(leaf_node)
            if children.size == 0:  # Needs expansion
                # Accumulate for batch evaluation
                simulation_paths.append(path)
                simulation_values.append(leaf_node)
        
        # Batch evaluate all expansions
        if simulation_values:
            expanded_count = self._batch_expand_and_backup(
                simulation_paths, simulation_values, node_to_state
            )
            
        return expanded_count
    
    def _run_wave_batched_parallel(self, wave_size: int, node_to_state: np.ndarray) -> int:
        """Run wave with optimized parallel batching
        
        Key optimization: Each thread processes multiple simulations
        to reduce task switching overhead.
        """
        # Calculate larger chunk sizes to reduce thread coordination
        sims_per_thread = max(8, wave_size // self.num_threads)
        num_actual_threads = min(self.num_threads, (wave_size + sims_per_thread - 1) // sims_per_thread)
        
        # Create thread pool only when needed
        with ThreadPoolExecutor(max_workers=num_actual_threads) as executor:
            futures = []
            
            for thread_id in range(num_actual_threads):
                start_sim = thread_id * sims_per_thread
                end_sim = min(start_sim + sims_per_thread, wave_size)
                
                if start_sim < end_sim:
                    future = executor.submit(
                        self._run_simulation_batch,
                        start_sim, end_sim, node_to_state
                    )
                    futures.append(future)
            
            # Collect results
            total_expanded = 0
            all_paths = []
            all_leaf_nodes = []
            
            for future in futures:
                expanded, paths, leaf_nodes = future.result()
                total_expanded += expanded
                all_paths.extend(paths)
                all_leaf_nodes.extend(leaf_nodes)
        
        return total_expanded
    
    def _run_simulation_batch(self, start_sim: int, end_sim: int, 
                             node_to_state: np.ndarray) -> Tuple[int, List, List]:
        """Run a batch of simulations in a single thread
        
        This does much more work per thread task to reduce coordination overhead.
        """
        batch_paths = []
        batch_leaf_nodes = []
        
        # Phase 1: Collect all simulations for this batch
        for sim_idx in range(start_sim, end_sim):
            path, leaf_node = self._select_path_to_leaf()
            if leaf_node is None:
                continue
                
            children, _, _ = self.tree.get_children(leaf_node)
            if children.size == 0:  # Needs expansion
                batch_paths.append(path)
                batch_leaf_nodes.append(leaf_node)
        
        # Phase 2: Batch evaluate all leaf nodes
        expanded_count = 0
        if batch_leaf_nodes:
            expanded_count = self._batch_expand_and_backup(
                batch_paths, batch_leaf_nodes, node_to_state
            )
        
        return expanded_count, batch_paths, batch_leaf_nodes
    
    def _select_path_to_leaf(self) -> Tuple[List[int], Optional[int]]:
        """Select path from root to leaf using optimized UCB"""
        path = []
        current_node = 0  # Start at root
        
        # Tree traversal with efficient UCB calculation
        max_depth = 100  # Prevent infinite loops
        
        for _ in range(max_depth):
            path.append(current_node)
            
            # CRITICAL FIX: Check if current node is terminal
            if hasattr(self, 'node_to_state') and self.node_to_state is not None:
                state_idx = self.node_to_state[current_node]
                if self.game_states.is_terminal[state_idx]:
                    # Found terminal node - stop here
                    return path, current_node
            
            # Get children efficiently
            children, actions, priors = self.tree.get_children(current_node)
            
            if children.size == 0:
                # Found leaf node
                return path, current_node
            
            # Calculate UCB scores with optimized vectorized operation
            if len(children) == 1:
                # Single child - no need for UCB calculation
                current_node = children[0]
            else:
                # Multiple children - use fully vectorized UCB
                # Get all data in one vectorized call if tree supports it
                if hasattr(self.tree, 'get_children_stats'):
                    visit_counts, value_sums = self.tree.get_children_stats(children)
                else:
                    # Fallback to individual calls but vectorize as much as possible
                    visit_counts = np.array([self.tree.get_visit_count(child) for child in children], dtype=np.float32)
                    value_sums = np.array([self.tree.get_value_sum(child) for child in children], dtype=np.float32)
                
                parent_visits = self.tree.get_visit_count(current_node)
                
                ucb_scores = batch_ucb_scores(
                    visit_counts, value_sums, priors.astype(np.float32),
                    parent_visits, children.astype(np.int32), self.config.c_puct
                )
                
                # Select best child
                best_child_idx = np.argmax(ucb_scores)
                current_node = children[best_child_idx]
        
        # Shouldn't reach here, but return what we have
        return path, current_node if path else None
    
    def _batch_expand_and_backup(self, paths: List[List[int]], 
                                leaf_nodes: List[int], 
                                node_to_state: np.ndarray) -> int:
        """Batch expand and backup with minimal overhead
        
        Key optimization: Batch neural network evaluation and tree updates
        """
        if not leaf_nodes:
            return 0
        
        # Phase 1: Batch neural network evaluation with caching
        state_indices = [node_to_state[node] for node in leaf_nodes]
        
        # Separate cached and uncached states
        uncached_indices = []
        uncached_positions = []
        policies = np.zeros((len(leaf_nodes), self.game_states.action_size), dtype=np.float32)
        values = np.zeros(len(leaf_nodes), dtype=np.float32)
        
        # Check cache for each state
        for i, state_idx in enumerate(state_indices):
            cache_key = state_idx
            if cache_key in self._eval_cache:
                # Use cached evaluation
                cached_policy, cached_value = self._eval_cache[cache_key]
                policies[i] = cached_policy
                values[i] = cached_value
                self._cache_hits += 1
            else:
                # Need to evaluate
                uncached_indices.append(state_idx)
                uncached_positions.append(i)
                self._cache_misses += 1
        
        # Evaluate uncached states if any
        if uncached_indices:
            try:
                # Get features with basic representation (19 channels for AlphaZero)
                features = self.game_states.get_nn_features(uncached_indices, representation_type='basic')
                if isinstance(features, tuple):
                    features = features[0]  # Take first element if tuple
                
                # Convert numpy to torch tensor if needed
                if isinstance(features, np.ndarray):
                    features = torch.from_numpy(features).float()
                    
                # Batch evaluation
                new_policies, new_values = self.evaluator.evaluate_batch(features)
                
                # Convert to numpy if needed
                if hasattr(new_policies, 'cpu'):
                    new_policies = new_policies.cpu().numpy()
                if hasattr(new_values, 'cpu'):
                    new_values = new_values.cpu().numpy()
                
                # Fill results and update cache
                for j, pos in enumerate(uncached_positions):
                    policies[pos] = new_policies[j]
                    values[pos] = new_values[j]
                    # Cache the evaluation (limit cache size)
                    if len(self._eval_cache) < 10000:  # Simple size limit
                        cache_key = uncached_indices[j]
                        self._eval_cache[cache_key] = (new_policies[j].copy(), float(new_values[j]))
                    
            except Exception as e:
                logger.warning(f"Batch evaluation failed: {e}")
                return 0
        
        # Phase 2: Batch expansion
        expanded_count = 0
        for i, (leaf_node, policy, value) in enumerate(zip(leaf_nodes, policies, values)):
            try:
                # Get legal actions efficiently
                state_idx = state_indices[i]
                
                # CRITICAL FIX: Don't expand terminal nodes
                if self.game_states.is_terminal[state_idx]:
                    continue
                legal_mask = self.game_states.get_legal_moves_mask([state_idx])[0]
                legal_actions = np.where(legal_mask)[0]
                
                if len(legal_actions) == 0:
                    continue
                
                # Create children with legal actions only
                legal_priors = policy[legal_actions]
                legal_priors = legal_priors / np.sum(legal_priors)  # Renormalize
                
                # Apply Dirichlet noise to root node for exploration
                if leaf_node == 0 and hasattr(self.config, 'dirichlet_epsilon') and self.config.dirichlet_epsilon > 0:
                    alpha = getattr(self.config, 'dirichlet_alpha', 0.03)
                    epsilon = self.config.dirichlet_epsilon
                    noise = np.random.dirichlet([alpha] * len(legal_priors))
                    legal_priors = (1 - epsilon) * legal_priors + epsilon * noise
                
                # Limit expansion to reduce tree overhead
                # Keep expansions minimal for throughput
                if leaf_node == 0:  # Root node
                    # For root, use progressive widening if enabled
                    if hasattr(self.config, 'progressive_widening_constant'):
                        # Progressive widening: expand k * n^alpha children
                        visits = self.tree.get_visit_count(leaf_node)
                        pw_constant = getattr(self.config, 'progressive_widening_constant', 10.0)
                        pw_exponent = getattr(self.config, 'progressive_widening_exponent', 0.5)
                        max_children = min(len(legal_actions), int(pw_constant * (visits ** pw_exponent)) + 1)
                        max_children = max(5, min(max_children, 15))  # Between 5 and 15
                    else:
                        max_children = min(len(legal_actions), 15)
                else:
                    # For other nodes, very selective
                    max_children = min(len(legal_actions), 3)
                if len(legal_actions) > max_children:
                    # Select top actions by prior probability
                    top_indices = np.argpartition(legal_priors, -max_children)[-max_children:]
                    legal_actions = legal_actions[top_indices]
                    legal_priors = legal_priors[top_indices]
                    legal_priors = legal_priors / np.sum(legal_priors)  # Renormalize
                
                # Expand node
                child_states = self.game_states.clone_states(
                    [state_idx] * len(legal_actions),
                    [1] * len(legal_actions)  # One clone per action
                )
                
                # Check if allocation succeeded
                if len(child_states) == 0:
                    # Skip expansion if we can't allocate states
                    continue
                
                # Apply actions to the cloned states
                self.game_states.apply_actions(child_states, legal_actions)
                
                self.tree.add_children_batch(
                    leaf_node, 
                    legal_actions.tolist(),
                    legal_priors.tolist(),
                    child_states.tolist()
                )
                
                expanded_count += 1
                
            except Exception as e:
                # Only log the first few expansion failures to avoid spam
                if not hasattr(self, '_expansion_failure_count'):
                    self._expansion_failure_count = 0
                self._expansion_failure_count += 1
                if self._expansion_failure_count <= 3:
                    logger.warning(f"Node expansion failed: {e}")
                    if self._expansion_failure_count == 3:
                        logger.warning("Further expansion failures will be suppressed")
                continue
        
        # Phase 3: Batch backup
        if paths:
            self._batch_backup_paths(paths, values)
            
        return expanded_count
    
    def _batch_backup_paths(self, paths: List[List[int]], values: np.ndarray):
        """Batch backup values through multiple paths efficiently"""
        if not paths:
            return
            
        # Check if tree has optimized batch backup
        if hasattr(self.tree, 'batch_backup_optimized'):
            # Convert paths to numpy array for batch processing
            max_path_len = max(len(path) for path in paths)
            path_array = np.full((len(paths), max_path_len), -1, dtype=np.int32)
            for i, path in enumerate(paths):
                path_array[i, :len(path)] = path
            self.tree.batch_backup_optimized(path_array, values.astype(np.float32))
        else:
            # Fallback: backup each path individually
            for i, path in enumerate(paths):
                value = float(values[i]) if i < len(values) else 0.0
                # Backup value along the path
                for node_idx in reversed(path):
                    if hasattr(self.tree, 'backup_value'):
                        self.tree.backup_value(node_idx, value)
                    else:
                        # Manual backup if no method available
                        self.tree.visit_counts[node_idx] += 1
                        self.tree.value_sums[node_idx] += value
                    # Flip value for opponent
                    value = -value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics"""
        return {
            'num_threads': self.num_threads,
            'waves_per_batch': self.waves_per_batch,
            'max_eval_batch_size': self.max_eval_batch_size,
        }