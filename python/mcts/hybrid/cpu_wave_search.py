"""CPU Wave Search Implementation

Wave-based parallel MCTS that processes multiple nodes simultaneously
in waves, maximizing CPU utilization and minimizing synchronization.
"""

import numpy as np
import torch
import time
import logging
import threading
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

from .simd_ucb import BatchedUCBSelector
from .memory_pool import ThreadLocalMemoryPool, ObjectPool
from .thread_local_buffers import ThreadLocalBufferManager

logger = logging.getLogger(__name__)


@dataclass
class WaveNode:
    """Node in a wave of parallel selections"""
    node_idx: int
    path: List[int]
    state_idx: int
    depth: int
    thread_id: int
    
    def reset(self):
        self.node_idx = -1
        self.path = []
        self.state_idx = -1
        self.depth = 0
        self.thread_id = -1


class CPUWaveSearch:
    """CPU-optimized wave-based parallel MCTS
    
    Key features:
    - Processes nodes in waves for better parallelism
    - Minimal synchronization between waves
    - SIMD operations within each wave
    - Thread-local memory management
    """
    
    def __init__(
        self,
        tree: Any,
        game_states: Any,
        evaluator: Any,
        config: Any,
        wave_size: int = 256,
        num_workers: int = 8,
        enable_simd: bool = True
    ):
        self.tree = tree
        self.game_states = game_states
        self.evaluator = evaluator
        self.config = config
        
        # Wave configuration
        self.wave_size = wave_size
        self.num_workers = num_workers
        self.enable_simd = enable_simd
        
        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        # SIMD UCB selector
        if enable_simd:
            self.ucb_selector = BatchedUCBSelector(
                batch_size=32,
                c_puct=config.c_puct
            )
            
        # Memory pools
        self._init_memory_pools()
        
        # Statistics
        self.stats = {
            'total_waves': 0,
            'total_nodes_processed': 0,
            'avg_wave_size': 0.0,
            'selection_time': 0.0,
            'expansion_time': 0.0,
            'backup_time': 0.0
        }
        
        logger.info(
            f"CPUWaveSearch initialized: "
            f"wave_size={wave_size}, "
            f"workers={num_workers}, "
            f"SIMD={'enabled' if enable_simd else 'disabled'}"
        )
        
    def _init_memory_pools(self):
        """Initialize memory pools for wave nodes"""
        self.wave_node_pool = ObjectPool(
            WaveNode,
            pool_size=self.wave_size * 2
        )
        
        # Thread-local pools for paths
        self.path_pool = ThreadLocalMemoryPool(
            block_size=256,  # Enough for deep paths
            blocks_per_thread=self.wave_size // self.num_workers
        )
        
    def run_simulations(self, num_simulations: int, root_state: Any) -> int:
        """Run wave-based simulations
        
        Args:
            num_simulations: Target number of simulations
            root_state: Root game state
            
        Returns:
            Actual number of simulations completed
        """
        completed_simulations = 0
        
        # Initialize root if needed
        if self.tree.num_nodes == 0:
            self.tree.add_node(0, -1, -1, 1.0)  # Root node
            
        while completed_simulations < num_simulations:
            # Determine wave size
            remaining = num_simulations - completed_simulations
            current_wave_size = min(self.wave_size, remaining)
            
            # Execute one wave
            wave_completed = self._execute_wave(current_wave_size)
            completed_simulations += wave_completed
            
            self.stats['total_waves'] += 1
            
        # Update statistics
        if self.stats['total_waves'] > 0:
            self.stats['avg_wave_size'] = (
                completed_simulations / self.stats['total_waves']
            )
            
        logger.info(
            f"Completed {completed_simulations} simulations "
            f"in {self.stats['total_waves']} waves"
        )
        
        return completed_simulations
        
    def _execute_wave(self, wave_size: int) -> int:
        """Execute one wave of parallel simulations
        
        Returns:
            Number of simulations completed
        """
        wave_start = time.perf_counter()
        
        # Phase 1: Parallel selection
        selection_start = time.perf_counter()
        leaf_nodes = self._parallel_selection(wave_size)
        self.stats['selection_time'] += time.perf_counter() - selection_start
        
        if not leaf_nodes:
            return 0
            
        # Phase 2: Batch neural network evaluation
        features_batch = self._extract_features_batch(leaf_nodes)
        policies, values = self._evaluate_batch(features_batch)
        
        # Phase 3: Parallel expansion
        expansion_start = time.perf_counter()
        expanded_nodes = self._parallel_expansion(leaf_nodes, policies)
        self.stats['expansion_time'] += time.perf_counter() - expansion_start
        
        # Phase 4: Parallel backup
        backup_start = time.perf_counter()
        self._parallel_backup(leaf_nodes, values)
        self.stats['backup_time'] += time.perf_counter() - backup_start
        
        # Update stats
        self.stats['total_nodes_processed'] += len(leaf_nodes)
        
        # Return wave nodes to pool
        for node in leaf_nodes:
            self.wave_node_pool.release(node)
            
        return len(leaf_nodes)
        
    def _parallel_selection(self, wave_size: int) -> List[WaveNode]:
        """Select leaf nodes in parallel
        
        Returns:
            List of selected leaf nodes
        """
        # Divide work among threads
        nodes_per_worker = wave_size // self.num_workers
        remainder = wave_size % self.num_workers
        
        futures = []
        
        for worker_id in range(self.num_workers):
            # Calculate this worker's share
            worker_nodes = nodes_per_worker
            if worker_id < remainder:
                worker_nodes += 1
                
            if worker_nodes > 0:
                future = self.executor.submit(
                    self._worker_select_nodes,
                    worker_id,
                    worker_nodes
                )
                futures.append(future)
                
        # Collect results
        all_leaf_nodes = []
        for future in as_completed(futures):
            leaf_nodes = future.result()
            all_leaf_nodes.extend(leaf_nodes)
            
        return all_leaf_nodes
        
    def _worker_select_nodes(self, worker_id: int, num_nodes: int) -> List[WaveNode]:
        """Worker function to select multiple leaf nodes
        
        Returns:
            List of selected leaf nodes
        """
        leaf_nodes = []
        
        for _ in range(num_nodes):
            # Get wave node from pool
            wave_node = self.wave_node_pool.acquire()
            if wave_node is None:
                wave_node = WaveNode(-1, [], -1, 0, worker_id)
            else:
                wave_node.reset()
                wave_node.thread_id = worker_id
                
            # Perform selection
            if self._select_leaf_node(wave_node):
                leaf_nodes.append(wave_node)
                
        return leaf_nodes
        
    def _select_leaf_node(self, wave_node: WaveNode) -> bool:
        """Select a single leaf node using UCB
        
        Returns:
            True if valid leaf found
        """
        node_idx = 0  # Start at root
        path = []
        
        # Apply virtual loss along the path
        virtual_loss_nodes = []
        
        while True:
            # Apply virtual loss
            if self.config.enable_virtual_loss:
                self.tree.apply_virtual_loss(torch.tensor([node_idx], dtype=torch.int64))
                virtual_loss_nodes.append(node_idx)
                
            # Get children
            children, actions, priors = self.tree.get_children(node_idx)
            
            if len(children) == 0:
                # Leaf found
                break
                
            # Select best child
            if self.enable_simd and len(children) >= 8:
                # Use SIMD for larger branching factors
                best_children, _ = self.ucb_selector.select_batch(
                    self.tree,
                    np.array([node_idx])
                )
                best_child = best_children[0]
            else:
                # Standard selection for small branching
                best_child = self._select_best_child(node_idx, children, priors)
                
            if best_child < 0:
                break
                
            path.append(node_idx)
            node_idx = best_child
            
        # Remove virtual loss
        if virtual_loss_nodes:
            self.tree.remove_virtual_loss(
                torch.tensor(virtual_loss_nodes, dtype=torch.int64)
            )
            
        # Update wave node
        wave_node.node_idx = node_idx
        wave_node.path = path
        wave_node.depth = len(path)
        wave_node.state_idx = self._get_or_create_state(node_idx)
        
        return wave_node.node_idx >= 0
        
    def _select_best_child(self, parent_idx: int, children: torch.Tensor, 
                          priors: torch.Tensor) -> int:
        """Select best child using standard UCB"""
        
        if len(children) == 0:
            return -1
            
        # Get statistics
        parent_visits = self.tree.node_data.visit_counts[parent_idx].item()
        if parent_visits == 0:
            parent_visits = 1
            
        child_visits = self.tree.node_data.visit_counts[children]
        child_values = self.tree.node_data.value_sums[children]
        
        # Calculate Q-values
        q_values = torch.where(
            child_visits > 0,
            child_values / child_visits.float(),
            torch.zeros_like(child_values)
        )
        
        # Calculate exploration
        sqrt_parent = np.sqrt(parent_visits)
        exploration = self.config.c_puct * priors * sqrt_parent / (1 + child_visits.float())
        
        # UCB values
        ucb_values = q_values + exploration
        
        # Select best
        best_idx = torch.argmax(ucb_values).item()
        return children[best_idx].item()
        
    def _get_or_create_state(self, node_idx: int) -> int:
        """Get or create game state for node"""
        # Simplified version - in practice would track node-to-state mapping
        return 0  # Use root state for now
        
    def _extract_features_batch(self, leaf_nodes: List[WaveNode]) -> torch.Tensor:
        """Extract features for all leaf nodes"""
        
        features_list = []
        
        for node in leaf_nodes:
            # Get features from game state
            if hasattr(self.game_states, 'get_nn_features'):
                features = self.game_states.get_nn_features([node.state_idx])
                if isinstance(features, np.ndarray):
                    features = torch.from_numpy(features[0])
                features_list.append(features)
            else:
                # Dummy features
                features_list.append(torch.zeros(19, 15, 15))
                
        return torch.stack(features_list)
        
    def _evaluate_batch(self, features: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate batch of positions"""
        
        with torch.no_grad():
            features = features.to(self.config.device)
            policies, values = self.evaluator.evaluate_batch(features)
            
        return policies.cpu().numpy(), values.cpu().numpy()
        
    def _parallel_expansion(self, leaf_nodes: List[WaveNode], 
                          policies: np.ndarray) -> List[int]:
        """Expand leaf nodes in parallel"""
        
        # Divide work among threads
        nodes_per_worker = len(leaf_nodes) // self.num_workers
        
        futures = []
        offset = 0
        
        for worker_id in range(self.num_workers):
            # Calculate range for this worker
            start = offset
            if worker_id == self.num_workers - 1:
                end = len(leaf_nodes)
            else:
                end = offset + nodes_per_worker
                
            if end > start:
                future = self.executor.submit(
                    self._worker_expand_nodes,
                    leaf_nodes[start:end],
                    policies[start:end]
                )
                futures.append(future)
                
            offset = end
            
        # Collect expanded nodes
        expanded = []
        for future in as_completed(futures):
            expanded.extend(future.result())
            
        return expanded
        
    def _worker_expand_nodes(self, nodes: List[WaveNode], 
                           policies: np.ndarray) -> List[int]:
        """Worker function to expand nodes"""
        
        expanded = []
        
        for i, node in enumerate(nodes):
            policy = policies[i]
            
            # Get top actions
            top_k = min(20, len(policy))
            top_actions = np.argpartition(policy, -top_k)[-top_k:]
            top_actions = top_actions[np.argsort(policy[top_actions])[::-1]]
            
            valid_actions = []
            valid_priors = []
            
            for action in top_actions:
                if policy[action] > 1e-6:
                    valid_actions.append(int(action))
                    valid_priors.append(float(policy[action]))
                    
            if valid_actions:
                # Normalize priors
                total = sum(valid_priors)
                valid_priors = [p / total for p in valid_priors]
                
                # Expand node
                self.tree.add_children(node.node_idx, valid_actions, valid_priors)
                expanded.append(node.node_idx)
                
        return expanded
        
    def _parallel_backup(self, leaf_nodes: List[WaveNode], values: np.ndarray):
        """Backup values in parallel"""
        
        # Group nodes by depth for efficient backup
        nodes_by_depth = {}
        for i, node in enumerate(leaf_nodes):
            depth = node.depth
            if depth not in nodes_by_depth:
                nodes_by_depth[depth] = []
            nodes_by_depth[depth].append((node, values[i]))
            
        # Backup from deepest to shallowest
        for depth in sorted(nodes_by_depth.keys(), reverse=True):
            nodes_values = nodes_by_depth[depth]
            
            # Parallel backup at this depth
            futures = []
            for node, value in nodes_values:
                future = self.executor.submit(
                    self._backup_single_path,
                    node,
                    value
                )
                futures.append(future)
                
            # Wait for completion
            for future in as_completed(futures):
                future.result()
                
    def _backup_single_path(self, node: WaveNode, value: float):
        """Backup value along a single path"""
        
        current_value = value
        
        # Update leaf
        self.tree.update_visit_count(node.node_idx, 1)
        self.tree.update_value_sum(node.node_idx, current_value)
        
        # Update path
        for node_idx in reversed(node.path):
            current_value = -current_value  # Flip for opponent
            self.tree.update_visit_count(node_idx, 1)
            self.tree.update_value_sum(node_idx, current_value)
            
    def get_stats(self) -> Dict[str, Any]:
        """Get wave search statistics"""
        return self.stats.copy()
        
    def shutdown(self):
        """Shutdown thread pool"""
        self.executor.shutdown(wait=True)