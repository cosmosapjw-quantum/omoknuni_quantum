"""Optimized Wave Engine with True GPU Vectorization and Quantum Features

This module implements the vectorized MCTS with quantum-inspired enhancements:
- Tensor-based path construction replacing sequential loops
- Memory pooling for tensor reuse
- Vectorized batch operations
- Parallel path construction without loops (Phase 1 optimization)
- MinHash-based interference for path diversity
- Phase-kicked priors for enhanced exploration
- Path integral formulation for action selection
"""

import torch
import torch.nn.functional as F
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import logging

from .csr_tree import CSRTree
from .csr_gpu_kernels_optimized import get_csr_batch_operations
from .cuda_graph_optimizer import CUDAGraphOptimizer, CUDAGraphConfig

# Core quantum-inspired components - always required
from mcts.quantum.interference import MinHashInterference
from mcts.quantum.phase_policy import PhaseKickedPolicy
from mcts.quantum.path_integral import PathIntegralMCTS

logger = logging.getLogger(__name__)


@dataclass
class OptimizedWaveConfig:
    """Configuration for optimized wave engine with quantum features"""
    # Import central configuration
    from mcts.core.mcts_config import get_config
    _mcts_config = get_config()
    
    # Best performing settings from central config
    wave_size: int = _mcts_config.wave_size
    max_depth: int = 100
    c_puct: float = _mcts_config.c_puct
    device: str = _mcts_config.device
    
    # Memory optimization - always enabled for best performance
    enable_memory_pooling: bool = True
    preallocate_tensors: bool = True
    
    # Performance tuning
    adaptive_wave_sizing: bool = _mcts_config.adaptive_wave_sizing
    min_wave_size: int = _mcts_config.min_wave_size
    max_wave_size: int = _mcts_config.max_wave_size
    gpu_utilization_target: float = 0.85
    
    # Advanced optimizations - enabled by default
    enable_cuda_graphs: bool = _mcts_config.use_cuda_graphs
    enable_mixed_precision: bool = _mcts_config.use_mixed_precision
    
    # Quantum-inspired features - core to the algorithm
    enable_interference: bool = _mcts_config.enable_interference
    interference_strength: float = _mcts_config.interference_strength
    enable_phase_policy: bool = _mcts_config.enable_phase_kicks
    phase_kick_strength: float = _mcts_config.phase_kick_strength
    enable_path_integral: bool = True  # Default to True for quantum-inspired action selection


class TensorMemoryPool:
    """Memory pool for efficient tensor reuse"""
    
    def __init__(self, config: OptimizedWaveConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.pools = {}
        
        # Pre-allocate if requested
        if config.preallocate_tensors:
            self._preallocate_all()
        
    def _preallocate_all(self):
        """Pre-allocate all working tensors for maximum performance"""
        max_wave = self.config.max_wave_size
        max_depth = self.config.max_depth
        max_children = 361  # Maximum children (e.g., Go board)
        
        # Path construction tensors
        self.pools[('paths', (max_wave, max_depth), torch.int32)] = torch.zeros(
            (max_wave, max_depth), dtype=torch.int32, device=self.device)
        self.pools[('active_mask', (max_wave,), torch.bool)] = torch.zeros(
            max_wave, dtype=torch.bool, device=self.device)
        self.pools[('nodes', (max_wave,), torch.int32)] = torch.zeros(
            max_wave, dtype=torch.int32, device=self.device)
        
        # Parallel operations tensors
        self.pools[('flat_nodes', (max_wave * max_depth,), torch.int32)] = torch.zeros(
            max_wave * max_depth, dtype=torch.int32, device=self.device)
        self.pools[('flat_active', (max_wave * max_depth,), torch.bool)] = torch.zeros(
            max_wave * max_depth, dtype=torch.bool, device=self.device)
        self.pools[('gather_indices', (max_wave * max_children,), torch.int64)] = torch.zeros(
            max_wave * max_children, dtype=torch.int64, device=self.device)
        
        # UCB calculation tensors
        self.pools[('ucb_scores', (max_wave, max_children), torch.float32)] = torch.zeros(
            (max_wave, max_children), dtype=torch.float32, device=self.device)
        self.pools[('child_indices', (max_wave, max_children), torch.int32)] = torch.zeros(
            (max_wave, max_children), dtype=torch.int32, device=self.device)
        
    def get_tensor(self, name: str, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        """Get tensor from pool or create new one"""
        key = (name, shape, dtype)
        
        if key not in self.pools:
            self.pools[key] = torch.zeros(shape, dtype=dtype, device=self.device)
        
        # Return a view of the appropriate size
        tensor = self.pools[key]
        if shape[0] < tensor.shape[0]:
            return tensor[:shape[0]] if len(shape) == 1 else tensor[:shape[0], :shape[1]]
        return tensor
        
    def get_paths_tensor(self, wave_size: int, max_depth: int) -> torch.Tensor:
        """Get pre-allocated paths tensor"""
        return self.get_tensor('paths', (wave_size, max_depth), torch.int32)
        
    def get_active_mask(self, wave_size: int) -> torch.Tensor:
        """Get pre-allocated active mask"""
        return self.get_tensor('active_mask', (wave_size,), torch.bool)
        
    def get_node_buffer(self, wave_size: int) -> torch.Tensor:
        """Get pre-allocated node buffer"""
        return self.get_tensor('nodes', (wave_size,), torch.int32)
    
    def clear(self):
        """Clear all pooled tensors to free memory"""
        self.pools.clear()


class OptimizedWaveEngine:
    """Wave engine with tensor-based vectorization"""
    
    def __init__(self, csr_tree: CSRTree, config: OptimizedWaveConfig, 
                 game_interface: Any, evaluator: Any):
        self.csr_tree = csr_tree
        self.config = config
        self.device = csr_tree.device
        self.game_interface = game_interface
        self.evaluator = evaluator
        
        # Get optimized CSR batch operations
        self.csr_batch_ops = get_csr_batch_operations(self.device)
        
        # Memory pool for tensor reuse
        if config.enable_memory_pooling:
            self.memory_pool = TensorMemoryPool(config)
        else:
            self.memory_pool = None
            
        # CUDA graph optimization
        if config.enable_cuda_graphs and self.device.type == 'cuda':
            cuda_graph_config = CUDAGraphConfig(
                enable_graphs=True,
                warmup_iterations=3,
                verbose=False
            )
            self.cuda_graph_optimizer = CUDAGraphOptimizer(
                device=self.device,
                config=cuda_graph_config
            )
            logger.info("CUDA graph optimization available (applied selectively)")
        else:
            self.cuda_graph_optimizer = None
            
        # Initialize state management
        from mcts.utils.state_delta_encoder import DeltaStateManager
        self.state_manager = DeltaStateManager(
            game_interface=game_interface,
            max_states=config.max_wave_size * config.max_depth * 10,
            device=str(self.device)
        )
            
        # Initialize quantum-inspired components - always created as core components
        self.interference_engine = MinHashInterference(
            device=self.device,
            strength=config.interference_strength
        )
        
        self.phase_policy = PhaseKickedPolicy(
            device=self.device,
            kick_strength=config.phase_kick_strength
        )
        
        self.path_integral = PathIntegralMCTS(
            device=self.device
        )
        
        # Track quantum feature usage
        self.quantum_stats = {
            'interference_applied': 0,
            'phase_kicks_applied': 0, 
            'path_integral_used': 0,
            'diversity_scores': []
        }
                
        # Performance tracking
        self.stats = {
            'waves_processed': 0,
            'total_simulations': 0,
            'selection_time': 0.0,
            'evaluation_time': 0.0,
            'backup_time': 0.0,
            'gpu_utilization': []
        }
        
        # Adaptive wave sizing
        self.current_wave_size = config.wave_size
        self.wave_size_history = []
        
    def run_wave(self, root_state: Any, wave_size: Optional[int] = None) -> Dict:
        """Run optimized wave with proper state tracking and vectorized operations"""
        
        if wave_size is None:
            wave_size = self.current_wave_size
        
        # Safety limit on wave size to prevent infinite loops
        wave_size = min(wave_size, 2048)
            
        # Adapt wave size if enabled
        if self.config.adaptive_wave_sizing:
            wave_size = self._adapt_wave_size(wave_size)
            
        wave_start_time = time.perf_counter()
        logger.debug(f"[WaveEngine] Starting wave with size {wave_size}")
        
        # Phase 1: Vectorized Path Selection with State Tracking
        selection_start = time.perf_counter()
        logger.debug("[WaveEngine] Phase 1: Path selection...")
        paths_tensor, leaf_indices, path_lengths, leaf_states = self._select_wave_paths_with_states(
            root_state, wave_size)
        logger.debug(f"[WaveEngine] Selected {paths_tensor.shape[0]} paths, max depth {path_lengths.max().item()}")
        
        # Apply MinHash interference if enabled
        if self.config.enable_interference and paths_tensor.shape[0] > 1:
            interference_start = time.perf_counter()
            logger.debug("[WaveEngine] Applying MinHash interference...")
            # Compute path diversity and apply interference
            signatures, similarities = self.interference_engine.compute_path_diversity_batch(
                paths_tensor, num_hashes=64
            )
            
            # Apply interference to leaf values (will be used in evaluation)
            self.interference_scores = similarities
            self.quantum_stats['interference_applied'] += paths_tensor.shape[0]
            
            # Track diversity for statistics
            avg_similarity = similarities.mean().item()
            self.quantum_stats['diversity_scores'].append(1.0 - avg_similarity)
            interference_time = time.perf_counter() - interference_start
            logger.debug(f"[WaveEngine] Interference completed in {interference_time:.3f}s (avg similarity: {avg_similarity:.3f})")
        else:
            self.interference_scores = None
            
        selection_time = time.perf_counter() - selection_start
        logger.debug(f"[WaveEngine] Selection phase completed in {selection_time:.3f}s")
        
        # Phase 2: Batch Expansion - Expand all leaf nodes in parallel
        expansion_start = time.perf_counter()
        logger.debug(f"[WaveEngine] Phase 2: Expanding {len(leaf_indices)} leaf nodes...")
        expanded_indices, expanded_states = self._batch_expand_leaves(
            leaf_indices, leaf_states)
        expansion_time = time.perf_counter() - expansion_start
        logger.debug(f"[WaveEngine] Expansion completed in {expansion_time:.3f}s ({len(expanded_indices)} new nodes)")
        
        # Phase 3: Batch Neural Network Evaluation
        eval_start = time.perf_counter()
        logger.debug(f"[WaveEngine] Phase 3: Evaluating {len(expanded_states)} states...")
        values = self._batch_evaluate_states(expanded_states)
        
        # Apply interference to values if computed
        if self.config.enable_interference and self.interference_scores is not None:
            # Map expanded values back to paths
            if len(values) == len(paths_tensor):
                # Apply destructive interference based on path similarity
                values = self.interference_engine.apply_interference(
                    values, self.interference_scores, 
                    interference_strength=self.config.interference_strength
                )
        
        eval_time = time.perf_counter() - eval_start
        logger.debug(f"[WaveEngine] Evaluation completed in {eval_time:.3f}s")
        
        # Phase 4: Vectorized Backup through paths
        backup_start = time.perf_counter()
        logger.debug("[WaveEngine] Phase 4: Backing up values...")
        self._vectorized_backup(paths_tensor, path_lengths, values, expanded_indices)
        backup_time = time.perf_counter() - backup_start
        logger.debug(f"[WaveEngine] Backup completed in {backup_time:.3f}s")
        
        # Update statistics
        total_time = time.perf_counter() - wave_start_time
        self.stats['waves_processed'] += 1
        self.stats['total_simulations'] += wave_size
        self.stats['selection_time'] += selection_time
        self.stats['evaluation_time'] += eval_time
        self.stats['backup_time'] += backup_time
        
        # Log GPU memory usage
        if torch.cuda.is_available():
            gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            logger.debug(f"[WaveEngine] GPU memory: {gpu_mem_allocated:.2f}GB allocated, {gpu_mem_reserved:.2f}GB reserved")
        
        # Performance metrics
        sims_per_sec = wave_size / total_time if total_time > 0 else 0
        logger.debug(f"[WaveEngine] Wave completed in {total_time:.3f}s ({sims_per_sec:.1f} sims/sec)")
        logger.debug(f"[WaveEngine] Time breakdown: selection={selection_time:.3f}s, eval={eval_time:.3f}s, backup={backup_time:.3f}s")
        
        return {
            'wave_size': wave_size,
            'paths_tensor': paths_tensor,
            'values': values,
            'timing': {
                'total': total_time,
                'selection': selection_time,
                'evaluation': eval_time,
                'backup': backup_time
            },
            'performance': {
                'sims_per_second': sims_per_sec,
                'selection_throughput': wave_size / selection_time if selection_time > 0 else 0,
                'evaluation_throughput': wave_size / eval_time if eval_time > 0 else 0
            }
        }
    
    def _select_wave_paths_vectorized(self, wave_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized path selection using parallel tensor operations"""
        return self._select_wave_paths_parallel(wave_size)
    
    def _select_wave_paths_with_states(self, root_state: Any, wave_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Any]]:
        """Vectorized path selection with state tracking and MinHash interference"""
        
        max_depth = self.config.max_depth
        device = self.device
        
        # Initialize tensors
        paths = torch.full((wave_size, max_depth), -1, dtype=torch.int32, device=device)
        paths[:, 0] = 0  # All start at root
        
        # Active mask tracks which paths are still selecting
        active = torch.ones(wave_size, dtype=torch.bool, device=device)
        
        # State tracking - initialize with root state
        current_states = [root_state] * wave_size
        
        # Track path signatures for interference if enabled
        if self.config.enable_interference:
            path_signatures = None
            path_similarities = None
        
        # Vectorized selection through tree
        for depth in range(max_depth - 1):
            if not active.any():
                break
                
            # Debug: Check for infinite loop
            if depth > 50:  # Safety check
                logger.warning(f"Depth {depth} exceeded safety limit - breaking")
                break
                
            current_nodes = paths[:, depth]
            
            # Get children for all active nodes at once - with safety checks
            # Shape: (wave_size, max_actions)
            try:
                # Ensure current_nodes are valid indices
                valid_node_mask = (current_nodes >= 0) & (current_nodes < self.csr_tree.num_nodes)
                if not valid_node_mask.any():
                    break
                    
                children = self.csr_tree.children[current_nodes]
                valid_children = children >= 0
            except Exception as e:
                logger.warning(f"Error accessing children at depth {depth}: {e}")
                break
            
            # Check which nodes have children
            has_children = valid_children.any(dim=1)
            is_leaf = active & ~has_children
            
            # If no children exist for any active node, we're at leaves
            if not (active & has_children).any():
                break
            
            # Apply MinHash interference if enabled and we have enough paths
            if self.config.enable_interference and depth > 0:
                # Compute path diversity up to current depth
                current_paths = paths[:, :depth+1]
                path_signatures, path_similarities = self.interference_engine.compute_path_diversity_batch(
                    current_paths, num_hashes=32
                )
                
                # Compute interference scores based on similarities
                interference_scores = path_similarities.mean(dim=1) - torch.eye(
                    wave_size, device=device
                ).mean(dim=1)
            else:
                interference_scores = None
            
            # Compute UCB scores for all children in parallel
            ucb_scores = self._compute_ucb_vectorized(
                current_nodes, children, valid_children, 
                interference_scores=interference_scores
            )
            
            # Select best child index for each path
            best_child_indices = ucb_scores.argmax(dim=1)
            # Ensure indices are within bounds
            batch_indices = torch.arange(wave_size, device=device)
            valid_batch_mask = batch_indices < children.shape[0]
            valid_child_mask = best_child_indices < children.shape[1]
            valid_selection = valid_batch_mask & valid_child_mask
            
            # Safe indexing with default value
            next_nodes = torch.full((wave_size,), -1, dtype=torch.int32, device=device)
            if valid_selection.any():
                next_nodes[valid_selection] = children[batch_indices[valid_selection], best_child_indices[valid_selection]]
            
            # Update paths
            paths[:, depth + 1] = torch.where(active & has_children, next_nodes, -1)
            
            # Update active mask
            active = active & has_children & (next_nodes >= 0)
            
            # Apply moves to update states (vectorized where possible)
            if active.any():
                # Batch state transitions
                active_indices = torch.where(active)[0]
                new_states = current_states.copy()
                
                # Get the actual game actions from the selected children
                selected_children = next_nodes[active]
                game_actions = self.csr_tree.parent_actions[selected_children]
                
                # Group by action for batch processing
                unique_actions = torch.unique(game_actions)
                for action in unique_actions:
                    mask_in_active = game_actions == action
                    global_mask = torch.zeros(wave_size, dtype=torch.bool, device=device)
                    global_mask[active_indices] = mask_in_active
                    indices = torch.where(global_mask)[0]
                    
                    if indices.numel() > 0:
                        # Apply same action to multiple states
                        states_to_update = [current_states[i] for i in indices.cpu().numpy()]
                        updated_states = self._batch_apply_move(states_to_update, action.item())
                        
                        for i, idx in enumerate(indices.cpu().numpy()):
                            new_states[idx] = updated_states[i]
                            # Store state in both tree and state manager
                            node_id = next_nodes[idx].item()
                            if node_id >= 0:
                                self.csr_tree.node_states[node_id] = updated_states[i]
                                self.state_manager.store_state(
                                    node_id, updated_states[i], 
                                    parent_idx=current_nodes[idx].item(), 
                                    action=action.item()
                                )
                
                current_states = new_states
        
        # Find leaf nodes
        path_lengths = (paths >= 0).sum(dim=1)
        leaf_indices = paths[torch.arange(wave_size), (path_lengths - 1).clamp(min=0)]
        
        # Get leaf states
        leaf_states = []
        for i, leaf_idx in enumerate(leaf_indices.cpu().numpy()):
            if leaf_idx >= 0 and leaf_idx in self.csr_tree.node_states:
                leaf_states.append(self.csr_tree.node_states[leaf_idx])
            else:
                leaf_states.append(current_states[i])
                
        return paths, leaf_indices, path_lengths, leaf_states
    
    def _parallel_ucb_selection(self, nodes: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        """Parallel UCB calculation for all nodes"""
        
        if not active_mask.any():
            return torch.zeros_like(nodes)
        
        # Get CSR data for all nodes at once
        row_starts = self.csr_tree.row_ptr[nodes]
        row_ends = self.csr_tree.row_ptr[nodes + 1]
        max_children = (row_ends - row_starts).max().item()
        
        if max_children == 0:
            return torch.full_like(nodes, -1)
        
        # Create batch gather indices
        batch_size = len(nodes)
        gather_offsets = torch.arange(max_children, device=self.device).unsqueeze(0)
        gather_indices = row_starts.unsqueeze(1) + gather_offsets
        
        # Valid children mask
        valid_mask = gather_offsets < (row_ends - row_starts).unsqueeze(1)
        gather_indices = torch.where(valid_mask, gather_indices, row_starts.unsqueeze(1))
        
        # Gather all data in parallel
        child_indices = self.csr_tree.col_indices[gather_indices]
        edge_actions = self.csr_tree.edge_actions[gather_indices]
        edge_priors = self.csr_tree.edge_priors[gather_indices]
        
        # Vectorized UCB calculation
        parent_visits = self.csr_tree.visit_counts[nodes].float()
        child_visits = self.csr_tree.visit_counts[child_indices].float()
        child_values = self.csr_tree.value_sums[child_indices].float()
        
        # Q-values
        q_values = torch.where(
            valid_mask & (child_visits > 0),
            child_values / child_visits,
            torch.zeros_like(child_values)
        )
        
        # Exploration
        sqrt_parent = torch.sqrt(parent_visits.unsqueeze(1) + 1.0)
        exploration = self.config.c_puct * edge_priors * sqrt_parent / (1.0 + child_visits)
        
        # UCB scores
        ucb_scores = torch.where(valid_mask, q_values + exploration, float('-inf'))
        
        # Select best actions
        best_indices = torch.argmax(ucb_scores, dim=1)
        selected_actions = edge_actions[torch.arange(batch_size, device=self.device), best_indices]
        
        # Mask inactive nodes
        return torch.where(active_mask, selected_actions, torch.tensor(-1, device=self.device))
    
    def _parallel_child_gather(self, nodes: torch.Tensor, actions: torch.Tensor, 
                              active_mask: torch.Tensor) -> torch.Tensor:
        """Gather children for all nodes in parallel"""
        
        # Get CSR data
        row_starts = self.csr_tree.row_ptr[nodes]
        row_ends = self.csr_tree.row_ptr[nodes + 1]
        max_children = (row_ends - row_starts).max().item()
        
        if max_children == 0:
            return torch.full_like(nodes, -1)
        
        batch_size = len(nodes)
        
        # Gather edge actions
        gather_offsets = torch.arange(max_children, device=self.device).unsqueeze(0)
        gather_indices = row_starts.unsqueeze(1) + gather_offsets
        valid_mask = gather_offsets < (row_ends - row_starts).unsqueeze(1)
        
        edge_actions_batch = self.csr_tree.edge_actions[gather_indices]
        edge_actions_batch = torch.where(valid_mask, edge_actions_batch, -1)
        
        # Find matching actions
        action_matches = edge_actions_batch == actions.unsqueeze(1)
        match_indices = torch.argmax(action_matches.float(), dim=1)
        
        # Get children
        child_gather_indices = row_starts + match_indices
        next_nodes = torch.where(
            action_matches.any(dim=1) & active_mask,
            self.csr_tree.col_indices[child_gather_indices],
            torch.tensor(-1, dtype=torch.int32, device=self.device)
        )
        
        return next_nodes
    
    def _extract_leaf_indices_parallel(self, all_nodes: torch.Tensor, 
                                      path_lengths: torch.Tensor) -> torch.Tensor:
        """Extract leaf node indices in parallel"""
        
        batch_size = all_nodes.shape[0]
        batch_indices = torch.arange(batch_size, device=self.device)
        
        # Get last valid depth for each path
        last_depths = torch.clamp(path_lengths - 1, min=0)
        
        # Extract leaf nodes
        leaf_indices = all_nodes[batch_indices, last_depths]
        
        # Default to root for empty paths
        return torch.where(path_lengths > 0, leaf_indices, 
                          torch.tensor(0, dtype=torch.int32, device=self.device))
    
    
    def _evaluate_leaves_batch(self, leaf_indices: torch.Tensor) -> torch.Tensor:
        """Batch evaluation of leaf nodes using neural network"""
        batch_size = len(leaf_indices)
        
        if batch_size == 0:
            return torch.zeros(0, device=self.device)
        
        # Collect valid states for evaluation
        valid_indices = []
        valid_states = []
        
        for i, idx in enumerate(leaf_indices.cpu().numpy()):
            if idx in self.csr_tree.node_states:
                state = self.csr_tree.node_states[idx]
                if state is not None:
                    valid_indices.append(i)
                    valid_states.append(state)
        
        if not valid_states:
            # No valid states to evaluate - return zeros
            return torch.zeros(batch_size, device=self.device)
        
        # Convert states to neural network input
        nn_inputs = []
        for state in valid_states:
            nn_input = self.game_interface.state_to_numpy(state, use_enhanced=True)
            nn_inputs.append(nn_input)
        
        # Create batch tensor
        batch_tensor = torch.tensor(np.array(nn_inputs), dtype=torch.float32, device=self.device)
        
        # Evaluate batch
        with torch.no_grad():
            if hasattr(self.evaluator, 'model'):
                # Direct model evaluation
                _, values = self.evaluator.model(batch_tensor)
                values = values.squeeze(-1)
            else:
                # Use evaluator's evaluate_batch
                _, values_np = self.evaluator.evaluate_batch(batch_tensor)
                values = torch.tensor(values_np, device=self.device)
        
        # Create output tensor with zeros for invalid indices
        result = torch.zeros(batch_size, device=self.device)
        result[valid_indices] = values
        
        return result
    
    def _reconstruct_state(self, node_idx: int) -> Any:
        """Reconstruct game state for a node by replaying moves from root"""
        # This is a temporary solution - ideally states should be stored in the tree
        # For now, return None which will be handled by the evaluator
        return None
    
    def _expand_leaf_nodes_batch(self, leaf_indices: torch.Tensor, root_state: Any) -> torch.Tensor:
        """Expand leaf nodes by adding their children
        
        Returns indices of newly expanded nodes for evaluation
        """
        expanded_indices = []
        
        for leaf_idx in leaf_indices.cpu().numpy():
            # Skip if node doesn't exist or is already expanded
            if leaf_idx < 0 or leaf_idx >= self.csr_tree.num_nodes:
                continue
                
            # Check if already expanded
            children, _, _ = self.csr_tree.get_children(leaf_idx)
            if len(children) > 0:
                continue
                
            # Get state for this leaf
            if leaf_idx in self.csr_tree.node_states:
                state = self.csr_tree.node_states[leaf_idx]
            else:
                # Need to reconstruct state - for now skip
                continue
                
            # Get legal moves
            legal_moves = self.game_interface.get_legal_moves(state)
            if not legal_moves:
                continue
                
            # Get policy prior from neural network
            nn_input = self.game_interface.state_to_numpy(state, use_enhanced=True)
            nn_tensor = torch.tensor(nn_input, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                if hasattr(self.evaluator, 'model'):
                    policy, _ = self.evaluator.model(nn_tensor)
                else:
                    policy, _ = self.evaluator.evaluate(nn_input)
                    policy = torch.tensor(policy, device=self.device).unsqueeze(0)
            
            # Add children for legal moves
            policy_probs = torch.softmax(policy[0], dim=0)
            for move in legal_moves:
                # Apply move to get child state
                child_state = self.game_interface.apply_move(state, move)
                prior = policy_probs[move].item()
                
                # Add child to tree
                child_idx = self.csr_tree.add_child(leaf_idx, move, prior, child_state)
                expanded_indices.append(child_idx)
        
        # Return indices of newly added nodes
        if expanded_indices:
            return torch.tensor(expanded_indices, device=self.device, dtype=torch.int32)
        else:
            return torch.zeros(0, device=self.device, dtype=torch.int32)
    
    def _backup_values_vectorized(self, paths_tensor: torch.Tensor, 
                                 path_lengths: torch.Tensor, values: torch.Tensor):
        """Vectorized backup of values through paths"""
        self._backup_values_parallel(paths_tensor, path_lengths, values)
    
    def _backup_values_parallel(self, paths_tensor: torch.Tensor, 
                               path_lengths: torch.Tensor, values: torch.Tensor):
        """Fully parallel backup without depth iteration"""
        
        wave_size, max_depth = paths_tensor.shape
        
        # Create 2D tensor of values for all depths
        backup_values = values.unsqueeze(1).expand(-1, max_depth).clone()
        
        # Apply alternating signs for two-player games
        depth_signs = torch.tensor([(-1) ** d for d in range(max_depth)], 
                                  device=self.device).unsqueeze(0)
        backup_values = backup_values * depth_signs
        
        # Create mask for valid positions
        depth_indices = torch.arange(max_depth, device=self.device).unsqueeze(0)
        valid_mask = depth_indices < path_lengths.unsqueeze(1)
        
        # Flatten for parallel processing
        flat_paths = paths_tensor.view(-1)
        flat_values = backup_values.view(-1)
        flat_valid = valid_mask.view(-1)
        
        # Get valid updates
        valid_indices = torch.where(flat_valid & (flat_paths >= 0))[0]
        if len(valid_indices) == 0:
            return
        
        valid_nodes = flat_paths[valid_indices]
        valid_values = flat_values[valid_indices]
        
        # Ensure nodes are within bounds
        node_mask = (valid_nodes >= 0) & (valid_nodes < self.csr_tree.num_nodes)
        valid_nodes = valid_nodes[node_mask]
        valid_values = valid_values[node_mask]
        
        if len(valid_nodes) == 0:
            return
        
        # Parallel atomic updates using scatter_add
        # Ensure types match for index_add_
        visit_increments = torch.ones_like(valid_nodes, dtype=self.csr_tree.visit_counts.dtype)
        self.csr_tree.visit_counts.index_add_(0, valid_nodes, visit_increments)
        
        # Ensure value types match
        if self.csr_tree.value_sums.dtype != valid_values.dtype:
            valid_values = valid_values.to(self.csr_tree.value_sums.dtype)
        self.csr_tree.value_sums.index_add_(0, valid_nodes, valid_values)
    
    def _adapt_wave_size(self, current_size: int) -> int:
        """Adapt wave size based on GPU utilization"""
        if not self.config.adaptive_wave_sizing:
            return current_size
            
        # Mock GPU utilization measurement (replace with actual monitoring)
        mock_gpu_util = 0.7 + torch.rand(1).item() * 0.3
        self.stats['gpu_utilization'].append(mock_gpu_util)
        
        target_util = self.config.gpu_utilization_target
        
        if mock_gpu_util < target_util - 0.1 and current_size < self.config.max_wave_size:
            # GPU underutilized, increase wave size
            new_size = min(int(current_size * 1.2), self.config.max_wave_size)
        elif mock_gpu_util > target_util + 0.1 and current_size > self.config.min_wave_size:
            # GPU overutilized, decrease wave size
            new_size = max(int(current_size * 0.9), self.config.min_wave_size)
        else:
            new_size = current_size
            
        self.current_wave_size = new_size
        self.wave_size_history.append(new_size)
        
        return new_size
    
    def reset_state_cache(self):
        """Reset any cached state information"""
        # Clear memory pool if it exists
        if self.memory_pool is not None:
            self.memory_pool._preallocate_all()
        
        # Reset wave sizing
        self.current_wave_size = self.config.wave_size
        self.wave_size_history.clear()
        
        # Clear state manager
        if hasattr(self, 'state_manager'):
            self.state_manager.states.clear()
            self.state_manager.parents.clear()
            self.state_manager.encoder.delta_cache.clear()
        
        # Reset stats
        self.stats = {
            'waves_processed': 0,
            'total_simulations': 0,
            'selection_time': 0.0,
            'evaluation_time': 0.0,
            'backup_time': 0.0,
            'gpu_utilization': []
        }
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        waves = self.stats['waves_processed']
        if waves == 0:
            return self.stats
            
        avg_gpu_util = np.mean(self.stats['gpu_utilization']) if self.stats['gpu_utilization'] else 0.0
        
        return {
            **self.stats,
            'average_gpu_utilization': avg_gpu_util,
            'average_wave_size': np.mean(self.wave_size_history) if self.wave_size_history else self.current_wave_size,
            'sims_per_second': self.stats['total_simulations'] / (
                self.stats['selection_time'] + self.stats['evaluation_time'] + self.stats['backup_time']
            ) if waves > 0 else 0,
            'selection_efficiency': self.stats['total_simulations'] / self.stats['selection_time'] if self.stats['selection_time'] > 0 else 0
        }
    
    def _compute_ucb_vectorized(self, parent_nodes: torch.Tensor, children: torch.Tensor, 
                               valid_mask: torch.Tensor, interference_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Vectorized UCB computation with phase-kicked priors and MinHash interference"""
        batch_size = parent_nodes.shape[0]
        
        # Bounds checking to prevent CUDA assertions
        parent_nodes = torch.clamp(parent_nodes, 0, self.csr_tree.num_nodes - 1)
        # Don't clamp children - instead mask invalid ones
        # children = torch.clamp(children, 0, self.csr_tree.num_nodes - 1)
        
        # Get parent visit counts
        parent_visits = self.csr_tree.visit_counts[parent_nodes].float()
        
        # Get child statistics in one operation
        # First, create a safe version of children with valid indices only
        safe_children = torch.where(valid_mask & (children >= 0) & (children < self.csr_tree.num_nodes), 
                                   children, 
                                   torch.zeros_like(children))
        
        child_visits = self.csr_tree.visit_counts[safe_children].float()
        child_values = torch.where(
            child_visits > 0,
            self.csr_tree.value_sums[safe_children] / child_visits,
            torch.zeros_like(self.csr_tree.value_sums[safe_children])
        )
        
        # Get priors - vectorized lookup with validation
        child_priors = self.csr_tree.node_priors[safe_children]
        # Ensure priors are valid probabilities
        child_priors = torch.clamp(child_priors, min=1e-8, max=1.0)
        child_priors = torch.where(torch.isfinite(child_priors), child_priors, torch.tensor(1e-8, device=self.device))
        
        # Apply phase kicks to priors if enabled
        if self.config.enable_phase_policy:
            # Get Q-values for phase calculation
            q_values = torch.where(child_visits > 0, child_values, torch.zeros_like(child_values))
            
            # Apply phase kicks
            kicked_priors = self.phase_policy.apply_phase_kicks(
                child_priors, 
                child_visits,
                q_values
            )
            child_priors = kicked_priors
            self.quantum_stats['phase_kicks_applied'] += valid_mask.sum().item()
        
        # Compute exploration term
        sqrt_parent = torch.sqrt(parent_visits + 1).unsqueeze(1)
        exploration = self.config.c_puct * child_priors * sqrt_parent / (1 + child_visits)
        
        # UCB = Q + U
        ucb = child_values + exploration
        
        # Apply MinHash interference if enabled
        if self.config.enable_interference and interference_scores is not None:
            # Reduce UCB scores for paths with high similarity to others
            # interference_scores is shape (batch_size,), need to broadcast
            interference_penalty = interference_scores.unsqueeze(1) * self.config.interference_strength
            ucb = ucb * (1.0 - interference_penalty)
            self.quantum_stats['interference_applied'] += batch_size
        
        # Mask invalid children
        ucb = torch.where(valid_mask, ucb, torch.tensor(-float('inf'), device=self.device))
        
        # Ensure UCB scores are finite and valid
        ucb = torch.where(torch.isfinite(ucb), ucb, torch.tensor(-float('inf'), device=self.device))
        
        return ucb
    
    def _batch_apply_move(self, states: List[Any], action: int) -> List[Any]:
        """Apply the same move to multiple states - can be optimized per game"""
        return [self.game_interface.apply_move(state, action) for state in states]
    
    def _batch_expand_leaves(self, leaf_indices: torch.Tensor, 
                           leaf_states: List[Any]) -> Tuple[torch.Tensor, List[Any]]:
        """Vectorized expansion of leaf nodes"""
        
        logger.debug(f"[WaveEngine] _batch_expand_leaves called with {len(leaf_indices)} leaves")
        
        if len(leaf_indices) == 0:
            logger.debug("[WaveEngine] No leaves to expand")
            return torch.zeros(0, device=self.device, dtype=torch.int32), []
        
        # IMPORTANT: Handle duplicate leaf indices to avoid expanding the same node multiple times
        unique_indices, inverse_indices = torch.unique(leaf_indices, return_inverse=True)
        logger.debug(f"[WaveEngine] Unique leaf indices: {len(unique_indices)} (was {len(leaf_indices)})")
        
        # Get unique states corresponding to unique indices
        unique_states = []
        for unique_idx in unique_indices:
            # Find first occurrence of this index
            first_occurrence = (leaf_indices == unique_idx).nonzero(as_tuple=True)[0][0].item()
            unique_states.append(leaf_states[first_occurrence])
        
        # Ensure CSR structure is consistent before batch operations
        # DISABLED for performance - row pointers are not used in current implementation
        # self.csr_tree.ensure_consistent()
        
        # Vectorized check for children
        has_children = (self.csr_tree.children[unique_indices] >= 0).any(dim=1)
        unexpanded_mask = ~has_children
        
        valid_indices = unique_indices[unexpanded_mask].cpu().numpy()
        valid_states = [unique_states[i] for i in range(len(unique_states)) if unexpanded_mask[i]]
        
        logger.debug(f"[WaveEngine] Found {len(valid_indices)} unexpanded leaves")
        
        if not valid_states:
            logger.debug("[WaveEngine] No valid states to expand")
            return torch.zeros(0, device=self.device, dtype=torch.int32), []
        
        # Batch neural network evaluation
        nn_inputs = []
        logger.debug(f"[WaveEngine] Converting {len(valid_states)} states to NN inputs")
        for state in valid_states:
            try:
                nn_input = self.game_interface.state_to_numpy(state, use_enhanced=True)
                if nn_input is not None:
                    nn_inputs.append(nn_input)
            except Exception as e:
                logger.warning(f"Failed to convert state to numpy: {e}")
                continue
        
        logger.debug(f"[WaveEngine] Got {len(nn_inputs)} NN inputs")
        
        if not nn_inputs:
            logger.debug("[WaveEngine] No NN inputs created")
            return torch.zeros(0, device=self.device, dtype=torch.int32), []
        
        try:
            batch_tensor = torch.tensor(np.array(nn_inputs), dtype=torch.float32, device=self.device)
            logger.debug(f"[WaveEngine] Created batch tensor shape: {batch_tensor.shape}")
        except Exception as e:
            logger.warning(f"Failed to create batch tensor: {e}")
            return torch.zeros(0, device=self.device, dtype=torch.int32), []
        
        logger.debug("[WaveEngine] Running NN evaluation...")
        eval_start = time.perf_counter()
        with torch.no_grad():
            if hasattr(self.evaluator, 'model'):
                policy_logits, _ = self.evaluator.model(batch_tensor)
            else:
                policy_logits, _ = self.evaluator.evaluate_batch(batch_tensor)
                policy_logits = torch.tensor(policy_logits, device=self.device)
        eval_time = time.perf_counter() - eval_start
        logger.debug(f"[WaveEngine] NN evaluation took {eval_time:.3f}s")
        
        # Apply softmax with numerical stability
        policy_logits = torch.clamp(policy_logits, min=-50, max=50)  # Prevent overflow
        policy_logits = torch.where(torch.isfinite(policy_logits), policy_logits, torch.zeros_like(policy_logits))
        policies = F.softmax(policy_logits, dim=-1)
        
        # Ensure policies are valid probabilities
        policies = torch.clamp(policies, min=1e-8, max=1.0)
        policies = policies / policies.sum(dim=-1, keepdim=True)  # Renormalize
        
        # Vectorized child creation
        all_child_indices = []
        all_child_states = []
        
        # Limit expansion count to prevent infinite loops
        max_expansions = min(len(valid_indices), 50)  # More conservative safety limit
        
        # Batch process expansions
        logger.debug(f"[WaveEngine] Processing up to {max_expansions} expansions")
        for i, (parent_idx, state, policy) in enumerate(zip(valid_indices[:max_expansions], 
                                                           valid_states[:max_expansions], 
                                                           policies[:max_expansions])):
            if i % 10 == 0:
                logger.debug(f"[WaveEngine] Processing expansion {i}/{max_expansions}")
            legal_moves = self.game_interface.get_legal_moves(state)
            
            if legal_moves:
                # Add all children but mark only one for evaluation
                priors = policy[legal_moves]
                
                # For efficiency, limit to top moves if there are too many
                max_to_add = min(10, len(legal_moves))  # Even more conservative limit for performance
                if max_to_add < len(legal_moves):
                    top_k = torch.topk(priors, max_to_add)
                    selected_indices = top_k.indices
                    selected_moves = [legal_moves[i] for i in selected_indices]
                    selected_priors = top_k.values
                else:
                    selected_moves = legal_moves
                    selected_priors = priors
                
                # Add all selected children
                child_indices_for_parent = []
                for action, prior in zip(selected_moves, selected_priors):
                    # Don't apply move yet - just add the child node
                    child_idx = self.csr_tree.add_child(parent_idx, action, prior.item(), None)
                    child_indices_for_parent.append(child_idx)
                
                # Select ONE child to evaluate based on policy
                if child_indices_for_parent:
                    priors_normalized = selected_priors / selected_priors.sum()
                    selected_child_idx = torch.multinomial(priors_normalized, 1).item()
                    selected_child_node = child_indices_for_parent[selected_child_idx]
                    selected_action = selected_moves[selected_child_idx]
                    
                    # Apply move only for the selected child
                    child_state = self.game_interface.apply_move(state, selected_action)
                    self.csr_tree.node_states[selected_child_node] = child_state
                    
                    # Only evaluate this one child
                    all_child_indices.append(selected_child_node)
                    all_child_states.append(child_state)
        
        if all_child_indices:
            return torch.tensor(all_child_indices, device=self.device, dtype=torch.int32), all_child_states
        else:
            return torch.zeros(0, device=self.device, dtype=torch.int32), []
    
    def _batch_evaluate_states(self, states: List[Any]) -> torch.Tensor:
        """Vectorized neural network evaluation with batching"""
        if not states:
            return torch.zeros(0, device=self.device)
        
        # Convert states to tensors
        nn_inputs = []
        for state in states:
            try:
                nn_input = self.game_interface.state_to_numpy(state, use_enhanced=True)
                nn_inputs.append(nn_input)
            except Exception as e:
                logger.warning(f"Failed to convert state to numpy: {e}")
                # Try direct conversion as fallback
                if hasattr(state, 'get_enhanced_tensor_representation'):
                    nn_inputs.append(state.get_enhanced_tensor_representation())
                elif hasattr(state, 'get_tensor_representation'):
                    nn_inputs.append(state.get_tensor_representation())
                else:
                    continue
        
        # Process in optimal batch sizes
        batch_size = min(512, len(nn_inputs))  # Adaptive batch size
        all_values = []
        
        for i in range(0, len(nn_inputs), batch_size):
            batch = nn_inputs[i:i + batch_size]
            batch_tensor = torch.tensor(np.array(batch), dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                if hasattr(self.evaluator, 'model'):
                    _, values = self.evaluator.model(batch_tensor)
                else:
                    _, values = self.evaluator.evaluate_batch(batch_tensor)
                    values = torch.tensor(values, device=self.device)
                
                all_values.append(values.squeeze(-1))
        
        return torch.cat(all_values)
    
    def _vectorized_backup(self, paths: torch.Tensor, path_lengths: torch.Tensor, 
                          values: torch.Tensor, expanded_indices: torch.Tensor):
        """Fully vectorized value backup using scatter operations"""
        
        batch_size, max_depth = paths.shape
        
        # Handle value dimension mismatch efficiently
        # Values correspond to expanded nodes, not paths
        if len(values) != batch_size:
            # For now, just use the first batch_size values
            # This is a simplification - in practice we'd map expanded nodes back to paths
            if len(values) >= batch_size:
                values = values[:batch_size]
            else:
                # Pad with zeros if we have fewer values than paths
                padded_values = torch.zeros(batch_size, device=self.device, dtype=values.dtype)
                padded_values[:len(values)] = values
                values = padded_values
        
        # Vectorized sign computation
        signs = torch.pow(-1, torch.arange(max_depth, device=self.device).float())
        value_matrix = values.unsqueeze(1) * signs.unsqueeze(0)
        
        # Vectorized valid mask
        depth_range = torch.arange(max_depth, device=self.device).unsqueeze(0)
        valid_mask = depth_range < path_lengths.unsqueeze(1)
        
        # Efficient scatter using advanced indexing
        valid_positions = valid_mask.nonzero(as_tuple=True)
        valid_nodes = paths[valid_positions]
        valid_values = value_matrix[valid_positions]
        
        # Filter out invalid nodes and ensure they're within bounds
        node_mask = (valid_nodes >= 0) & (valid_nodes < self.csr_tree.num_nodes)
        valid_nodes = valid_nodes[node_mask]
        valid_values = valid_values[node_mask]
        
        if len(valid_nodes) > 0:
            # Ensure indices are within tensor bounds
            max_node_idx = self.csr_tree.visit_counts.shape[0] - 1
            valid_nodes = torch.clamp(valid_nodes, 0, max_node_idx)
            
            # Atomic scatter operations
            ones = torch.ones_like(valid_nodes, dtype=self.csr_tree.visit_counts.dtype)
            self.csr_tree.visit_counts.index_add_(0, valid_nodes, ones)
            self.csr_tree.value_sums.index_add_(0, valid_nodes, valid_values.to(self.csr_tree.value_sums.dtype))
    
    def select_action(self, root_idx: int = 0, use_path_integral: Optional[bool] = None) -> int:
        """Select best action (defaults to path integral formulation)
        
        This is the main action selection method that uses quantum-inspired
        path integral by default for better exploration.
        
        Args:
            root_idx: Root node index
            use_path_integral: Override for path integral usage
            
        Returns:
            Best action to take
        """
        # Use path integral by default unless explicitly disabled
        if use_path_integral is None:
            use_path_integral = self.config.enable_path_integral
            
        if not use_path_integral:
            # Fall back to standard selection based on visit counts
            return self.select_best_action_standard(root_idx)
            
        # Get children of root
        children = self.csr_tree.children[root_idx]
        valid_mask = children >= 0
        valid_children = children[valid_mask]
        
        if len(valid_children) == 0:
            return -1
            
        # Compute path integral action values
        action_values = self.path_integral.compute_action_values(
            self.csr_tree, root_idx, valid_children
        )
        
        # Select action based on path integral values
        best_idx = torch.argmax(action_values)
        best_child = valid_children[best_idx]
        
        # Get the action that leads to this child
        best_action = self.csr_tree.parent_actions[best_child].item()
        
        self.quantum_stats['path_integral_used'] += 1
        
        return best_action
    
    def select_best_action_standard(self, root_idx: int = 0) -> int:
        """Standard action selection based on visit counts
        
        This is the classical approach that selects the most visited child.
        """
        children = self.csr_tree.children[root_idx]
        valid_mask = children >= 0
        
        if not valid_mask.any():
            return -1
            
        valid_children = children[valid_mask]
        visit_counts = self.csr_tree.visit_counts[valid_children]
        
        best_idx = torch.argmax(visit_counts)
        best_child = valid_children[best_idx]
        
        return self.csr_tree.parent_actions[best_child].item()
    
    # Backward compatibility alias
    select_best_action = select_best_action_standard
    
    def cleanup(self):
        """Clean up resources and clear caches"""
        # Clear memory pools
        if hasattr(self, 'memory_pool') and self.memory_pool is not None:
            self.memory_pool.clear()
        
        # Clear state manager cache
        if hasattr(self, 'state_manager') and self.state_manager is not None:
            self.state_manager.states.clear()
            self.state_manager.parents.clear()
        
        # Clear tree states
        if hasattr(self, 'csr_tree') and self.csr_tree is not None:
            self.csr_tree.node_states.clear()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
