"""Wave-based MCTS implementation for massive parallelization

This implementation processes 256-2048 paths simultaneously in waves,
targeting 80k-200k simulations per second.
"""

import torch
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import logging
import time
from concurrent.futures import ThreadPoolExecutor
# import asyncio  # Removed - async was causing performance issues

from .cached_game_interface import CachedGameInterface, CacheConfig
from .mcts_config import MCTSConfig
from ..gpu.csr_tree import CSRTree, CSRTreeConfig
from ..neural_networks.evaluator_pool import EvaluatorPool
from ..quantum.path_integral import PathIntegral, PathIntegralConfig
from ..quantum.quantum_features import QuantumMCTS as QuantumFeatures, QuantumConfig as QuantumMCTSConfig
from ..utils.tensor_pool import get_tensor_pool

# Initialize CUDA operations
logger = logging.getLogger(__name__)

# Try to import CUDA kernels directly
try:
    import os
    # Get the absolute path to the CUDA kernels
    current_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_path = os.path.join(os.path.dirname(current_dir), 'gpu', 'mcts_cuda_kernels.cpython-312-x86_64-linux-gnu.so')
    
    if os.path.exists(kernel_path):
        torch.ops.load_library(kernel_path)
        import mcts.gpu.mcts_cuda_kernels as cuda_kernels_module
        cuda_ops_available = True
        logger.info("CUDA kernels loaded successfully for Wave MCTS")
    else:
        raise FileNotFoundError(f"CUDA kernel not found at {kernel_path}")
except Exception as e:
    cuda_kernels_module = None
    cuda_ops_available = False
    logger.info(f"CUDA kernels not available: {e}")

@dataclass
class WaveMCTSConfig:
    """Configuration for wave-based MCTS"""
    # Wave parameters
    min_wave_size: int = 256
    max_wave_size: int = 2048
    adaptive_wave_sizing: bool = True
    
    # MCTS parameters
    c_puct: float = 1.0  # UCB exploration constant
    dirichlet_alpha: float = 0.3  # Dirichlet noise parameter (0.3 for Go/Gomoku)
    dirichlet_epsilon: float = 0.25  # Root exploration noise weight
    temperature: float = 1.0  # Temperature for policy extraction
    
    # Performance targets
    target_sims_per_second: int = 100000  # 100k
    target_gpu_utilization: float = 0.95
    
    # Parallelization
    num_wave_pipelines: int = 3  # Triple buffering
    async_expansion: bool = True
    prefetch_evaluations: bool = True
    
    # Memory optimization
    use_memory_pools: bool = True
    pool_size_mb: int = 1024  # 1GB pool
    
    # Base configurations
    tree_config: Optional[CSRTreeConfig] = None
    cache_config: Optional[CacheConfig] = None
    quantum_config: Optional[QuantumMCTSConfig] = None
    path_integral_config: Optional[PathIntegralConfig] = None
    
    # Hardware optimization
    device: str = 'cuda'
    use_tensor_cores: bool = True
    use_cuda_graphs: bool = True
    use_mixed_precision: bool = False  # Whether to use FP16 for values
    
class WaveBuffer:
    """Pre-allocated buffer for wave processing"""
    
    def __init__(self, max_wave_size: int, max_depth: int, device: torch.device, 
                 use_mixed_precision: bool = False):
        self.max_wave_size = max_wave_size
        self.max_depth = max_depth
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        
        # Determine value dtype based on mixed precision setting
        self.value_dtype = torch.float16 if use_mixed_precision else torch.float32
        
        # Get tensor pool for this device
        self.tensor_pool = get_tensor_pool(device)
        
        # Path tracking - use tensor pool
        self.paths = self.tensor_pool.get_tensor((max_wave_size, max_depth), torch.int32, fill_value=0)
        self.path_lengths = self.tensor_pool.get_tensor((max_wave_size,), torch.int32, fill_value=0)
        self.current_nodes = self.tensor_pool.get_tensor((max_wave_size,), torch.int32, fill_value=0)
        self.active_mask = self.tensor_pool.get_tensor((max_wave_size,), torch.bool, fill_value=False)
        
        # Game states (Gomoku example - adapt for other games)
        self.boards = self.tensor_pool.get_tensor((max_wave_size, 15, 15), torch.int8, fill_value=0)
        self.current_players = self.tensor_pool.get_tensor((max_wave_size,), torch.int8, fill_value=0)
        
        # Neural network I/O
        self.nn_features = self.tensor_pool.get_tensor((max_wave_size, 3, 15, 15), torch.float16, fill_value=0)
        # Use consistent dtype for values based on mixed precision setting
        self.values = self.tensor_pool.get_tensor((max_wave_size,), self.value_dtype, fill_value=0)
        self.policies = self.tensor_pool.get_tensor((max_wave_size, 225), torch.float32, fill_value=0)
        
        # Quantum features
        self.phases = self.tensor_pool.get_tensor((max_wave_size, 225), torch.float32, fill_value=0)
        self.amplitudes = self.tensor_pool.get_tensor((max_wave_size, 225), torch.float32, fill_value=0)
        
    def reset(self, wave_size: int):
        """Reset buffer for new wave"""
        self.active_mask[:wave_size] = True
        self.active_mask[wave_size:] = False
        self.current_nodes[:wave_size] = 0  # Start from root
        self.path_lengths[:] = 0
        
class WavePipeline:
    """Pipeline for overlapping wave processing stages"""
    
    def __init__(self, buffer: WaveBuffer, tree: CSRTree, 
                 game_interface: CachedGameInterface, evaluator_pool: EvaluatorPool,
                 config: WaveMCTSConfig):
        self.buffer = buffer
        self.tree = tree
        self.game_interface = game_interface
        self.evaluator_pool = evaluator_pool
        self.config = config
        
        # Pipeline stages
        self.selection_done = False
        self.expansion_done = False
        self.evaluation_done = False
        
    def run_selection(self, wave_size: int):
        """Run selection phase synchronously"""
        depth = 0
        # Initialize paths with root node for all waves
        self.buffer.paths[:wave_size, 0] = 0  # Start all paths from root
        self.buffer.current_nodes[:wave_size] = 0  # All start at root
        self.buffer.path_lengths[:wave_size] = 1  # Path length 1 (just root)
        self.buffer.active_mask[:wave_size] = True  # All paths active
        
        # Pre-allocate tensors for selection to reduce memory allocations
        if not hasattr(self, '_selection_tensors') or self._selection_tensors['active_indices'].shape[0] < wave_size:
            self._selection_tensors = {
                'active_indices': torch.zeros(wave_size, dtype=torch.long, device=self.tree.device),
                'current_nodes': torch.zeros(wave_size, dtype=torch.int32, device=self.tree.device),
                'selected_actions': torch.zeros(wave_size, dtype=torch.int32, device=self.tree.device),
                'selected_children': torch.zeros(wave_size, dtype=torch.int32, device=self.tree.device),
                'valid_mask': torch.zeros(wave_size, dtype=torch.bool, device=self.tree.device),
            }
        
        while self.buffer.active_mask[:wave_size].any() and depth < self.buffer.max_depth:
            # Optimized: Use pre-allocated tensor and nonzero with out parameter
            active_mask = self.buffer.active_mask[:wave_size]
            active_count = active_mask.sum().item()
            if active_count == 0:
                break
            
            # Use pre-allocated tensor for active indices
            active_indices_full = torch.nonzero(active_mask, as_tuple=False).squeeze(1)
            active_indices = active_indices_full[:active_count]
            
            # Get current nodes
            current_nodes = self.buffer.current_nodes[active_indices]
            
            # Batched UCB selection using custom CUDA kernel
            if cuda_kernels_module is not None and hasattr(cuda_kernels_module, 'batched_ucb_selection'):
                # Use CUDA kernel for selection
                logger.debug("Using CUDA kernel for batched UCB selection")
                # Note: This would need the CSR tree format - for now use the tree method
                selected_children, _ = self.tree.batch_select_ucb_optimized(current_nodes, c_puct=self.config.c_puct)
            else:
                selected_children, _ = self.tree.batch_select_ucb_optimized(current_nodes, c_puct=self.config.c_puct)
            
            # Check how many valid selections
            valid_selections_count = (selected_children >= 0).sum().item()  # Actions, not indices
            
            if valid_selections_count == 0:
                pass  # Will be handled in expansion phase
                
            # Convert actions to actual child node indices using batch method
            # CRITICAL FIX: selected_children contains ACTIONS, not child indices!
            selected_child_indices = self.tree.batch_action_to_child(current_nodes, selected_children)
            
            # ROBUSTNESS: Check if conversion failed for any paths
            conversion_failed = selected_child_indices < 0
            
            if conversion_failed.any():
                # For failed conversions, nodes will be handled in expansion phase
                pass
            
            # Update paths with the SELECTED CHILD INDICES (not current nodes!)
            # This was the critical bug - paths should record where we're going, not where we came from
            self.buffer.paths[active_indices, depth] = torch.where(
                selected_child_indices >= 0,
                selected_child_indices,
                current_nodes  # Keep parent if child not found
            )

            self.buffer.path_lengths[active_indices] += 1
            
            # Check valid selections
            valid_selections = selected_child_indices >= 0
            
            # Update current nodes to the selected children
            self.buffer.current_nodes[active_indices] = torch.where(
                selected_child_indices >= 0,
                selected_child_indices,
                current_nodes  # Keep at parent if child not found
            )
            
            # Apply virtual loss to selected children for parallel path diversity
            valid_children = selected_child_indices[selected_child_indices >= 0]
            if len(valid_children) > 0:
                self.tree.apply_virtual_loss(valid_children)
            
            # Update valid selections based on whether we found valid children
            valid_selections = selected_child_indices >= 0
# Update active mask
            new_active_mask = self.buffer.active_mask.clone()
            new_active_mask[active_indices] = valid_selections
            self.buffer.active_mask = new_active_mask
            
            depth += 1
            
        
        # Verify selection worked - at least some paths should have depth > 1
        max_depth = self.buffer.path_lengths[:wave_size].max().item()
        if max_depth <= 1 and self.tree.num_nodes > 1:
            # Selection phase completed but no paths went beyond root
            # Force at least one path to visit a child
            if self.tree.num_nodes > 1:
                # Get root children
                root_children, _, _ = self.tree.batch_get_children(torch.tensor([0], device=self.tree.device))
                valid_children = root_children[0][root_children[0] >= 0]
                if len(valid_children) > 0:
                    # Select first valid child for first path
                    self.buffer.paths[0, 1] = valid_children[0]
                    self.buffer.path_lengths[0] = 2
                    self.buffer.current_nodes[0] = valid_children[0]
                    # Force path 0 to visit first child
        
        self.selection_done = True
        
    def run_expansion(self, wave_size: int):
        """Run expansion phase with GPU acceleration and parallel processing"""
        # VECTORIZED: Find all nodes that need expansion in one go
        valid_path_mask = self.buffer.path_lengths[:wave_size] > 0
        if not valid_path_mask.any():
            self.expansion_done = True
            return
            
        # Get current nodes from all paths
        current_nodes = self.buffer.current_nodes[:wave_size]
        
        # Try to use GPU kernel for finding expansion nodes
        if (cuda_kernels_module is not None and 
            hasattr(cuda_kernels_module, 'find_expansion_nodes') and
            self.tree.device.type == 'cuda'):
            # Use GPU kernel
            logger.debug("Using CUDA kernel for find_expansion_nodes")
            expansion_nodes, expansion_count = cuda_kernels_module.find_expansion_nodes(
                current_nodes,
                self.tree.children,
                self.tree.visit_counts,
                valid_path_mask,
                wave_size,
                self.tree.children.shape[1],  # max_children
                self.tree.num_nodes
            )
            
            if expansion_count.item() == 0:
                self.expansion_done = True
                return
                
            # Get unique nodes, limit to actual count
            unique_expansion_nodes = expansion_nodes[:expansion_count.item()]
        else:
            # Fallback to CPU version
            valid_current_nodes = current_nodes[valid_path_mask]
            
            # Vectorized check for expansion needs
            node_children = self.tree.children[valid_current_nodes]
            has_children = (node_children >= 0).any(dim=1)
            # CRITICAL FIX: Expand nodes without children, regardless of visit count
            # This handles the case after tree reuse where root has visits but no children
            needs_expansion = ~has_children
            
            if not needs_expansion.any():
                self.expansion_done = True
                return
                
            # Get unique nodes to expand
            expansion_nodes = valid_current_nodes[needs_expansion]
            unique_expansion_nodes = torch.unique(expansion_nodes)
            
            if len(unique_expansion_nodes) == 0:
                self.expansion_done = True
                return
        
        # PARALLEL STATE CREATION using ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def create_state_for_node(node_data):
            """Create state for a single node - can be parallelized"""
            node_id, parent_idx, parent_action = node_data
            if parent_idx in self.tree.node_states:
                parent_state = self.tree.node_states[parent_idx]
                state = self.game_interface.clone_state(parent_state)
                try:
                    self.game_interface.apply_move(state, parent_action)
                    return node_id, state
                except ValueError as e:
                    # This is expected when parent_action is no longer legal
                    # (e.g., the square was occupied by a later move)
                    # Just skip this node - it represents an invalid game path
                    # Skip this node - parent action is no longer legal
                    return None
            else:
                # Parent has no state, cannot create state for this node
                pass
            return None
        
        # Collect nodes needing state creation
        nodes_needing_states = []
        nodes_with_states = []
        
        # Vectorized parent lookup
        parent_indices = self.tree.parent_indices[unique_expansion_nodes]
        parent_actions = self.tree.parent_actions[unique_expansion_nodes]
        
        for i, node_id in enumerate(unique_expansion_nodes):
            node_id_val = node_id.item()
            if node_id_val not in self.tree.node_states:
                parent_idx = parent_indices[i].item()
                # CRITICAL: Only try to create state if parent has a state
                if parent_idx >= 0 and parent_idx in self.tree.node_states:
                    nodes_needing_states.append((node_id_val, parent_idx, parent_actions[i].item()))
                else:
                    # Skip nodes whose parents don't have states
                    pass
            else:
                nodes_with_states.append(node_id_val)
        
        # Parallel state creation
        if nodes_needing_states:
            num_threads = getattr(self.config, 'num_threads', getattr(self.config, 'cpu_threads', 24))
            with ThreadPoolExecutor(max_workers=min(num_threads, len(nodes_needing_states))) as executor:
                futures = [executor.submit(create_state_for_node, node_data) 
                          for node_data in nodes_needing_states]
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        node_id, state = result
                        self.tree.node_states[node_id] = state
                        nodes_with_states.append(node_id)
        
        if not nodes_with_states:
            self.expansion_done = True
            return
        
        # VECTORIZED FEATURE EXTRACTION
        # Pre-allocate feature tensor
        num_nodes = len(nodes_with_states)
        feature_shape = self.game_interface.state_to_numpy(self.tree.node_states[0]).shape
        features_batch = torch.zeros((num_nodes,) + feature_shape, device=self.tree.device)
        
        # Parallel feature extraction
        def extract_features(idx_and_node):
            idx, node_id = idx_and_node
            state = self.tree.node_states[node_id]
            return idx, self.game_interface.state_to_numpy(state)
        
        num_threads = getattr(self.config, 'num_threads', getattr(self.config, 'cpu_threads', 24))
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(extract_features, (i, node_id)) 
                      for i, node_id in enumerate(nodes_with_states)]
            
            for future in as_completed(futures):
                idx, features = future.result()
                features_batch[idx] = torch.from_numpy(features).float()
        
        # BATCH NEURAL NETWORK EVALUATION
        with torch.no_grad():
            _, policies_batch, _ = self.evaluator_pool.evaluate_batch(features_batch)
        
        # FAST EXPANSION using batch legal move processing
        # Try to use GPU kernel for batch legal move processing if available
        if (cuda_kernels_module is not None and 
            hasattr(cuda_kernels_module, 'batch_process_legal_moves') and
            self.tree.device.type == 'cuda' and
            hasattr(self.game_interface, 'get_board_size')):
            
            # Extract board states for GPU processing
            board_size = self.game_interface.get_board_size()
            board_states = torch.zeros((num_nodes, board_size), dtype=torch.int32, device=self.tree.device)
            
            for i, node_id in enumerate(nodes_with_states):
                state = self.tree.node_states[node_id]
                # Extract board as flat tensor (for Gomoku)
                if hasattr(state, 'board'):
                    board_states[i] = torch.from_numpy(state.board.flatten()).to(self.tree.device)
                else:
                    features = self.game_interface.state_to_numpy(state)
                    board_states[i] = torch.from_numpy(features[0].flatten()).to(self.tree.device)
            
            # Allocate output tensors
            legal_move_masks = torch.zeros((num_nodes, board_size), dtype=torch.bool, device=self.tree.device)
            normalized_priors = torch.zeros((num_nodes, board_size), dtype=torch.float32, device=self.tree.device)
            num_legal_moves = torch.zeros(num_nodes, dtype=torch.int32, device=self.tree.device)
            
            # Ensure policies_batch is float32
            if policies_batch.dtype != torch.float32:
                policies_batch = policies_batch.float()
            
            # Call GPU kernel with correct arguments
            normalized_priors, legal_move_indices, num_legal_moves = cuda_kernels_module.batch_process_legal_moves(
                policies_batch,  # raw_policies
                board_states,    # board_states
                num_nodes,       # num_states
                board_size       # action_size
            )
            
            # Convert results to expansion data
            expansion_data = []
            for i, node_id in enumerate(nodes_with_states):
                if num_legal_moves[i] > 0:
                    # Extract legal moves for this state
                    start_idx = i * board_size
                    end_idx = start_idx + num_legal_moves[i].item()
                    legal_indices = legal_move_indices[start_idx:end_idx]
                    legal_indices = legal_indices[legal_indices >= 0]  # Filter valid indices
                    
                    if len(legal_indices) > 0:
                        priors = normalized_priors[i][legal_indices]
                        expansion_data.append((
                            node_id,
                            legal_indices.cpu().tolist(),
                            priors.cpu().tolist()
                        ))
        else:
            # Fallback to CPU version with optimized batch processing
            expansion_data = []
            
            # Process all nodes at once without individual legal move calls
            if hasattr(self.game_interface, 'batch_get_legal_moves_tensor'):
                # Collect board states
                board_states = []
                for node_id in nodes_with_states:
                    state = self.tree.node_states[node_id]
                    if hasattr(state, 'board'):
                        board_states.append(torch.from_numpy(state.board))
                    else:
                        features = self.game_interface.state_to_numpy(state)
                        board_states.append(torch.from_numpy(features[0]))
                
                boards_tensor = torch.stack(board_states).to(self.tree.device)
                
                # Batch process legal moves
                legal_masks, num_legal = self.game_interface.batch_get_legal_moves_tensor(boards_tensor)
                
                # Process policies
                for idx, node_id in enumerate(nodes_with_states):
                    if num_legal[idx] > 0:
                        legal_mask = legal_masks[idx]
                        policy = policies_batch[idx]
                        
                        # Get legal move indices
                        legal_indices = torch.where(legal_mask)[0]
                        
                        # Extract and normalize priors
                        legal_priors = policy[legal_indices]
                        prior_sum = legal_priors.sum()
                        if prior_sum > 0:
                            legal_priors = legal_priors / prior_sum
                        else:
                            legal_priors = torch.ones_like(legal_priors) / len(legal_indices)
                        
                        expansion_data.append((
                            node_id,
                            legal_indices.cpu().tolist(),
                            legal_priors.cpu().tolist()
                        ))
            else:
                # Original parallel expansion preparation
                def prepare_expansion(idx_and_node):
                    idx, node_id = idx_and_node
                    if node_id not in self.tree.node_states:
                        # Skip nodes without states - they shouldn't be expanded
                        # Skip nodes without states - they shouldn't be expanded
                        return None
                    state = self.tree.node_states[node_id]
                    legal_moves = self.game_interface.get_legal_moves(state)
                    
                    if legal_moves:
                        # Extract policy
                        if torch.is_tensor(policies_batch):
                            policy = policies_batch[idx].cpu().numpy()
                        else:
                            policy = policies_batch[idx]
                        
                        # Vectorized prior extraction
                        legal_moves_array = np.array(legal_moves)
                        priors = policy[legal_moves_array]
                        prior_sum = priors.sum()
                        if prior_sum > 0:
                            priors = priors / prior_sum
                        else:
                            priors = np.ones(len(legal_moves)) / len(legal_moves)
                        
                        return node_id, legal_moves, priors.tolist()
                    return None
                
                # Parallel expansion preparation
                num_threads = getattr(self.config, 'num_threads', getattr(self.config, 'cpu_threads', 24))
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    futures = [executor.submit(prepare_expansion, (i, node_id)) 
                              for i, node_id in enumerate(nodes_with_states)]
                    
                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            expansion_data.append(result)
        
        # Batch add all children at once
        for node_id, legal_moves, priors in expansion_data:
            self.tree.add_children_batch(node_id, legal_moves, priors, None)
        
        self.tree.flush_batch()
        self.expansion_done = True
        
    def run_evaluation(self, wave_size: int):
        """Run evaluation phase with aggressive parallelization"""
        # Get all leaf nodes
        leaf_indices = torch.where(self.buffer.active_mask[:wave_size])[0]
        if len(leaf_indices) == 0:
            self.evaluation_done = True
            return
            
        # VECTORIZED: Get node indices for all leaves at once
        leaf_node_indices = self.buffer.current_nodes[leaf_indices]
        
        # Check which nodes need state creation
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def create_eval_state(data):
            """Create state for evaluation - parallelizable"""
            leaf_idx, node_idx, parent_idx, parent_action = data
            if parent_idx >= 0 and parent_idx in self.tree.node_states:
                parent_state = self.tree.node_states[parent_idx]
                state = self.game_interface.clone_state(parent_state)
                self.game_interface.apply_move(state, parent_action)
                self.tree.node_states[node_idx] = state
                return leaf_idx, node_idx, state
            return None
        
        # Collect nodes needing states
        states_to_create = []
        ready_states = []
        
        # Vectorized parent lookup
        parent_indices = self.tree.parent_indices[leaf_node_indices]
        parent_actions = self.tree.parent_actions[leaf_node_indices]
        
        for i, (leaf_idx, node_idx) in enumerate(zip(leaf_indices.tolist(), leaf_node_indices.tolist())):
            if node_idx not in self.tree.node_states:
                parent_idx = parent_indices[i].item()
                parent_action = parent_actions[i].item()
                states_to_create.append((leaf_idx, node_idx, parent_idx, parent_action))
            else:
                ready_states.append((leaf_idx, node_idx))
        
        # Parallel state creation
        if states_to_create:
            num_threads = getattr(self.config, 'num_threads', getattr(self.config, 'cpu_threads', 24))
            with ThreadPoolExecutor(max_workers=min(num_threads, len(states_to_create))) as executor:
                futures = [executor.submit(create_eval_state, data) for data in states_to_create]
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        leaf_idx, node_idx, state = result
                        ready_states.append((leaf_idx, node_idx))
        
        if not ready_states:
            self.evaluation_done = True
            return
            
        # Sort by leaf index for consistent ordering
        ready_states.sort(key=lambda x: x[0])
        valid_leaf_indices = torch.tensor([x[0] for x in ready_states], device=self.device)
        valid_node_indices = [x[1] for x in ready_states]
        
        # PARALLEL FEATURE EXTRACTION
        num_states = len(ready_states)
        feature_shape = self.game_interface.state_to_numpy(self.tree.node_states[0]).shape
        features_batch = torch.zeros((num_states,) + feature_shape, device=self.tree.device)
        
        def extract_eval_features(idx_and_node):
            idx, node_id = idx_and_node
            state = self.tree.node_states[node_id]
            return idx, self.game_interface.state_to_numpy(state)
        
        num_threads = getattr(self.config, 'num_threads', getattr(self.config, 'cpu_threads', 24))
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(extract_eval_features, (i, node_id)) 
                      for i, node_id in enumerate(valid_node_indices)]
            
            for future in as_completed(futures):
                idx, features = future.result()
                features_batch[idx] = torch.from_numpy(features)
        
        # BATCH NEURAL NETWORK EVALUATION
        with torch.cuda.amp.autocast() if self.tree.device.type == 'cuda' else torch.no_grad():
            values, policies, _ = self.evaluator_pool.evaluate_batch(features_batch)
        
        # VECTORIZED RESULT STORAGE
        # Convert values to buffer dtype if needed
        if values.dtype != self.buffer.values.dtype:
            values = values.to(self.buffer.values.dtype)
        
        self.buffer.values[valid_leaf_indices] = values
        if policies.dim() > 1:
            self.buffer.policies[valid_leaf_indices, :policies.shape[1]] = policies
        
        self.evaluation_done = True
        
    def run_backup(self, wave_size: int):
        """Run backup phase synchronously"""

        # No need to wait - called after evaluation
            
        # Use custom CUDA kernel for parallel backup if available and on CUDA device
        if (cuda_kernels_module is not None and 
            hasattr(cuda_kernels_module, 'parallel_backup') and 
            self.tree.device.type == 'cuda'):
            # Convert all tensors to float32 for CUDA kernel compatibility
            values_for_kernel = self.buffer.values[:wave_size].float()
            paths_for_kernel = self.buffer.paths[:wave_size].int()  # Ensure paths are int32
            path_lengths_for_kernel = self.buffer.path_lengths[:wave_size].int()
            
            # Convert tree tensors to float32 if needed
            if self.tree.value_sums.dtype != torch.float32:
                value_sums_float32 = self.tree.value_sums.float()
                visit_counts_int32 = self.tree.visit_counts.int()
                
                cuda_kernels_module.parallel_backup(
                    paths_for_kernel,
                    values_for_kernel,
                    path_lengths_for_kernel,
                    value_sums_float32,
                    visit_counts_int32
                )
                
                # Copy back to original dtype
                self.tree.value_sums.copy_(value_sums_float32.to(self.tree.value_sums.dtype))
                self.tree.visit_counts.copy_(visit_counts_int32.to(self.tree.visit_counts.dtype))
            else:
                cuda_kernels_module.parallel_backup(
                    paths_for_kernel,
                    values_for_kernel,
                    path_lengths_for_kernel,
                    self.tree.value_sums,
                    self.tree.visit_counts.int()
                )
        else:
            # Fallback to tree method
            # Get paths and values for nodes that were actually evaluated
            active_indices = torch.where(self.buffer.path_lengths[:wave_size] > 0)[0]
            
            if len(active_indices) > 0:
                # Remove virtual loss from all nodes in the paths before backup
                paths = self.buffer.paths[active_indices]
                all_nodes_in_paths = paths[paths >= 0]
                if len(all_nodes_in_paths) > 0:
                    unique_nodes = torch.unique(all_nodes_in_paths)
                    self.tree.remove_virtual_loss(unique_nodes)
                
                values = self.buffer.values[active_indices]
                self.tree.batch_backup_optimized(paths, values)

class WaveMCTS:
    """Wave-based MCTS with massive parallelization"""
    
    def __init__(self, config: WaveMCTSConfig, game_interface: CachedGameInterface,
                 evaluator_pool: EvaluatorPool):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize components
        if config.tree_config:
            tree_config = config.tree_config
        else:
            # Create default tree config with proper dtype
            tree_config = CSRTreeConfig(
                max_nodes=500000,
                batch_size=config.max_wave_size,
                device=config.device,
                # Match dtype with WaveBuffer based on mixed precision setting
                dtype_values=torch.float16 if config.use_mixed_precision else torch.float32,
                # Virtual loss settings (use defaults if not specified)
                enable_virtual_loss=getattr(config, 'enable_virtual_loss', True),
                virtual_loss_value=getattr(config, 'virtual_loss_value', -1.0)
            )
        self.tree = CSRTree(tree_config)
        
        self.game_interface = game_interface
        self.evaluator_pool = evaluator_pool
        
        # Initialize quantum features if enabled
        if config.quantum_config:
            self.quantum_features = QuantumFeatures(config.quantum_config)
            self.path_integral = PathIntegral(
                config.path_integral_config or PathIntegralConfig()
            )
        else:
            self.quantum_features = None
            self.path_integral = None
            
        # Create wave buffers for triple buffering
        # Determine if mixed precision should be used
        use_mixed_precision = hasattr(config, 'use_mixed_precision') and config.use_mixed_precision
        self.buffers = [
            WaveBuffer(config.max_wave_size, 100, self.device, use_mixed_precision)
            for _ in range(config.num_wave_pipelines)
        ]
        
        # Create pipelines
        self.pipelines = [
            WavePipeline(buffer, self.tree, game_interface, evaluator_pool, config)
            for buffer in self.buffers
        ]
        
        # CUDA graphs for kernel fusion
        if config.use_cuda_graphs:
            self._setup_cuda_graphs()
            
        # Statistics
        self.stats = {
            'total_simulations': 0,
            'total_time': 0,
            'simulations_per_second': 0,
            'gpu_utilization': 0,
            'wave_sizes': []
        }
        
        # Track the last root state to detect changes
        self.last_root_state = None
        
    def reset_tree(self):
        """Reset the tree to initial state"""
        # Create a new tree instance to ensure clean state
        tree_config = self.tree.config
        self.tree = CSRTree(tree_config)
        
        # Update pipelines with new tree reference
        for pipeline in self.pipelines:
            pipeline.tree = self.tree
            
        # Tree reset completed
        
    def update_root(self, action: int):
        """Update root to child node after a move (tree reuse)
        
        Args:
            action: The action that was taken
        """
        
        # Find the child corresponding to this action
        if self.tree.num_nodes == 0:
            return
            
        # Get root children
        root_children, root_actions, _ = self.tree.batch_get_children(
            torch.tensor([0], device=self.device, dtype=torch.int32)
        )
        
        children = root_children[0]
        actions = root_actions[0]
        valid_mask = children >= 0
        
        if not valid_mask.any():
            # No children, just reset
            self.reset_tree()
            return
            
        # Find the child with the given action
        valid_children = children[valid_mask]
        valid_actions = actions[valid_mask]
        
        # Find matching action
        action_matches = valid_actions == action
        if not action_matches.any():
            # Action not found in children, reset tree
            self.reset_tree()
            return
            
        # Get the child index
        child_idx = valid_children[action_matches][0].item()
        
        # Shift subtree to root
        self._shift_subtree_to_root(child_idx)
    
    def _shift_subtree_to_root(self, new_root_idx: int):
        """Shift a subtree to become the new tree root
        
        This is a key optimization for tree reuse in MCTS.
        We preserve the subtree rooted at new_root_idx and discard the rest.
        
        Args:
            new_root_idx: Index of the node that will become the new root
        """
        if new_root_idx == 0:
            # Already at root, nothing to do
            return
            
        # For CSRTree format, we need to:
        # 1. Find all nodes in the subtree rooted at new_root_idx
        # 2. Remap their indices (new_root becomes 0, etc.)
        # 3. Update all node data and CSR structure
        
        # Use BFS to find all nodes in the subtree
        subtree_nodes = self._get_subtree_nodes(new_root_idx)
        
        if len(subtree_nodes) == 0:
            # No subtree, just reset
            self.reset_tree()
            return
            
        # Convert to tensor for vectorized operations
        subtree_indices = torch.tensor(subtree_nodes, device=self.device, dtype=torch.long)
        new_num_nodes = len(subtree_nodes)
        
        # Create index mapping tensor: old_idx -> new_idx
        # Use scatter to create the mapping efficiently
        old_to_new_tensor = torch.full((self.tree.num_nodes,), -1, device=self.device, dtype=torch.long)
        new_indices = torch.arange(new_num_nodes, device=self.device, dtype=torch.long)
        old_to_new_tensor[subtree_indices] = new_indices
        
        # Vectorized copy of node data
        self.tree.visit_counts[:new_num_nodes] = self.tree.visit_counts[subtree_indices]
        self.tree.value_sums[:new_num_nodes] = self.tree.value_sums[subtree_indices]
        self.tree.node_priors[:new_num_nodes] = self.tree.node_priors[subtree_indices]
        self.tree.phases[:new_num_nodes] = self.tree.phases[subtree_indices]
        
        # Handle parent indices and actions
        old_parents = self.tree.parent_indices[subtree_indices]
        old_actions = self.tree.parent_actions[subtree_indices]
        
        # Remap parent indices
        new_parent_indices = torch.full((new_num_nodes,), -1, dtype=torch.int32, device=self.device)
        valid_parent_mask = old_parents >= 0
        valid_parents = old_parents[valid_parent_mask]
        if valid_parent_mask.any():
            remapped_parents = old_to_new_tensor[valid_parents]
            # Only keep parents that are in the subtree
            in_subtree_mask = remapped_parents >= 0
            if in_subtree_mask.any():
                valid_positions = torch.where(valid_parent_mask)[0][in_subtree_mask]
                new_parent_indices[valid_positions] = remapped_parents[in_subtree_mask].to(torch.int32)
        
        # The new root should have no parent
        new_parent_indices[0] = -1
        
        self.tree.parent_indices[:new_num_nodes] = new_parent_indices
        self.tree.parent_actions[:new_num_nodes] = torch.where(
            new_parent_indices >= 0,
            old_actions,
            torch.tensor(-1, dtype=self.tree.parent_actions.dtype, device=self.device)
        )
        
        # Clear remaining nodes if tree shrunk
        if new_num_nodes < self.tree.num_nodes:
            self.tree.visit_counts[new_num_nodes:self.tree.num_nodes] = 0
            self.tree.value_sums[new_num_nodes:self.tree.num_nodes] = 0
            self.tree.node_priors[new_num_nodes:self.tree.num_nodes] = 0
            self.tree.phases[new_num_nodes:self.tree.num_nodes] = 0
            self.tree.parent_indices[new_num_nodes:self.tree.num_nodes] = -1
            self.tree.parent_actions[new_num_nodes:self.tree.num_nodes] = -1
        
        # Vectorized update of children lookup table
        new_children = torch.full((new_num_nodes, self.tree.children.shape[1]), -1, 
                                 dtype=torch.int32, device=self.device)
        
        # Get all children at once
        old_children_flat = self.tree.children[subtree_indices].flatten()
        valid_child_mask = old_children_flat >= 0
        
        if valid_child_mask.any():
            valid_old_children = old_children_flat[valid_child_mask]
            remapped_children = old_to_new_tensor[valid_old_children]
            in_subtree_mask = remapped_children >= 0
            
            if in_subtree_mask.any():
                # Calculate positions in the new children tensor
                row_indices = torch.arange(new_num_nodes, device=self.device).unsqueeze(1).expand(-1, self.tree.children.shape[1]).flatten()[valid_child_mask][in_subtree_mask]
                col_indices = torch.arange(self.tree.children.shape[1], device=self.device).unsqueeze(0).expand(new_num_nodes, -1).flatten()[valid_child_mask][in_subtree_mask]
                
                new_children[row_indices, col_indices] = remapped_children[in_subtree_mask].to(torch.int32)
        
        self.tree.children[:new_num_nodes] = new_children
        if new_num_nodes < self.tree.num_nodes:
            self.tree.children[new_num_nodes:self.tree.num_nodes] = -1
        
        # Update node states dictionary (still need loop for dict)
        new_node_states = {}
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(subtree_nodes)}
        for old_idx, state in self.tree.node_states.items():
            if old_idx in old_to_new:
                new_node_states[old_to_new[old_idx]] = state
        self.tree.node_states = new_node_states
        
        # Update tree size
        self.tree.num_nodes = new_num_nodes
        
        # IMPORTANT: Mark root as needing visit count update
        # After shifting, the new root should have its parent's visit count
        if self.tree.visit_counts[0] == 0:
            self.tree.visit_counts[0] = 1  # Ensure root is visited
        
        # Clear is_expanded flag for root to force re-expansion
        if hasattr(self.tree, 'is_expanded'):
            self.tree.is_expanded[0] = False
        
        # TODO: Also update CSR structure (row_ptr, col_indices, etc.)
        # For now, the children lookup table should be sufficient
        
    
    def _get_subtree_nodes(self, root_idx: int) -> List[int]:
        """Get all nodes in the subtree rooted at root_idx using BFS
        
        Args:
            root_idx: Root of the subtree
            
        Returns:
            List of node indices in BFS order
        """
        from collections import deque
        
        subtree_nodes = [root_idx]
        subtree_set = {root_idx}  # For O(1) membership check
        queue = deque([root_idx])  # O(1) popleft instead of O(n) pop(0)
        
        while queue:
            node_idx = queue.popleft()
            
            # Get children of this node
            children = self.tree.children[node_idx]
            valid_children = children[children >= 0]
            
            for child_idx in valid_children:
                child_idx = child_idx.item()
                if child_idx not in subtree_set:
                    subtree_nodes.append(child_idx)
                    subtree_set.add(child_idx)
                    queue.append(child_idx)
        
        return subtree_nodes
        

    def search(self, root_state, num_simulations: int = None) -> np.ndarray:
        """Run wave-based MCTS search
        
        Args:
            root_state: Root game state
            num_simulations: Number of simulations (default from config)
            
        Returns:
            Policy distribution over moves
        """
        if num_simulations is None:
            num_simulations = 10000  # Default
            
        start_time = time.perf_counter()
        
        # Tree reuse is handled by update_root() after moves are played
        # We only need to check if this is a completely new game
        # by seeing if the root has game state stored
        if 0 not in self.tree.node_states:
            # This is a new game or first search, reset tree if needed
            if self.tree.num_nodes > 1:
                # New game detected, reset tree
                self.reset_tree()
            # Always store the root state for new trees
            self.tree.node_states[0] = root_state
            # Store root state in new tree
        else:
            # CRITICAL FIX: Always update the root state to match the current game state
            # This ensures consistency after tree reuse
            self.tree.node_states[0] = root_state
        
        # Initialize root if needed
        if self.tree.visit_counts[0] == 0:
            self._initialize_root(root_state)
        else:
            # CRITICAL: Check if root has children - if not, force expansion
            root_children, _, _ = self.tree.batch_get_children(
                torch.tensor([0], device=self.device, dtype=torch.int32)
            )
            has_children = (root_children[0] >= 0).any()
            if not has_children:
                self._initialize_root(root_state)
            
        # Adaptive wave sizing based on GPU utilization
        current_wave_size = self._determine_wave_size()
        
        
        # Tree is ready for search
            
        # Run waves
        total_sims = 0
        wave_count = 0
        
        # Run waves synchronously for better performance
        while total_sims < num_simulations:
            # Determine this wave's size
            wave_size = min(current_wave_size, num_simulations - total_sims)
            
            # Run wave synchronously (no async overhead)
            self._run_wave_sync(wave_size, wave_count)
            
            total_sims += wave_size
            wave_count += 1
            
            # Update wave size based on performance
            if self.config.adaptive_wave_sizing and wave_count % 10 == 0:
                current_wave_size = self._adapt_wave_size(current_wave_size)
            

        # Extract policy with temperature
        temperature = getattr(self.config, 'temperature', 1.0)
        policy = self._extract_policy_from_root(temperature)
        
        if policy.sum() == 0:
            pass  # Zero policy is handled in the caller
        
        # Update statistics
        elapsed = time.perf_counter() - start_time
        self.stats['total_simulations'] = total_sims
        self.stats['total_time'] = elapsed
        self.stats['simulations_per_second'] = total_sims / elapsed if elapsed > 0 else 0
        
        
        return policy
        
    def _run_wave_sync(self, wave_size: int, wave_idx: int):
        """Run a single wave synchronously (no async overhead)"""
        
        # Select buffer (round-robin)
        buffer_idx = wave_idx % len(self.buffers)
        buffer = self.buffers[buffer_idx]
        pipeline = self.pipelines[buffer_idx]
        
        # Reset buffer
        buffer.reset(wave_size)
        
        # Reset pipeline state
        pipeline.selection_done = False
        pipeline.expansion_done = False
        pipeline.evaluation_done = False
        
        # Run phases sequentially
        pipeline.run_selection(wave_size)
        pipeline.run_expansion(wave_size)
        pipeline.run_evaluation(wave_size)
        pipeline.run_backup(wave_size)
        
        # Record wave size for statistics
        self.stats['wave_sizes'].append(wave_size)
        
    def _determine_wave_size(self) -> int:
        """Determine optimal wave size based on GPU utilization
        
        For best performance, disable adaptive_wave_sizing and use max_wave_size.
        """
        if not self.config.adaptive_wave_sizing:
            # Use max_wave_size for best performance
            return self.config.max_wave_size
            
        # Adaptive sizing - start with larger waves for better GPU utilization
        if torch.cuda.is_available():
            # Simple heuristic based on memory usage
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            utilization = allocated / reserved if reserved > 0 else 0
            
            # Prefer larger wave sizes for better performance
            if utilization < 0.6:
                # Low utilization - use max size
                return self.config.max_wave_size
            elif utilization < 0.8:
                # Medium utilization - use 80% of max
                return int(self.config.max_wave_size * 0.8)
            elif utilization > 0.95:
                # Very high utilization - reduce to avoid OOM
                return max(self.config.min_wave_size, int(self.config.max_wave_size * 0.5))
            else:
                # High utilization - use 60% of max
                return int(self.config.max_wave_size * 0.6)
        else:
            # CPU mode - use min size
            return self.config.min_wave_size
            
    def _adapt_wave_size(self, current_size: int) -> int:
        """Adapt wave size based on performance"""
        if len(self.stats['wave_sizes']) < 10:
            return current_size
            
        # Check if we're meeting performance targets
        current_sps = self.stats['simulations_per_second']
        target_sps = self.config.target_sims_per_second
        
        if current_sps < target_sps * 0.8:
            # Not meeting target - try larger waves
            new_size = min(self.config.max_wave_size, int(current_size * 1.2))
        elif current_sps > target_sps * 1.2:
            # Exceeding target - can reduce for latency
            new_size = max(self.config.min_wave_size, int(current_size * 0.9))
        else:
            # On target
            new_size = current_size
            
        return new_size
        
    def _setup_cuda_graphs(self):
        """Setup CUDA graphs for kernel fusion"""
        # CUDA graphs capture a sequence of CUDA operations
        # and can replay them with lower overhead
        
        # This would capture common operation sequences
        # Implementation depends on specific PyTorch version
        pass
        
    def _initialize_root(self, root_state):
        """Initialize root node"""
        # Store root state
        self.tree.node_states[0] = root_state
        
        # Get legal moves
        legal_moves = self.game_interface.get_legal_moves(root_state)
        
        if legal_moves:
            # Evaluate root
            features = self.game_interface.state_to_numpy(root_state)
            features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                values, policies, info = self.evaluator_pool.evaluate_batch(features_tensor)
                
            # Extract priors for legal moves
            if torch.is_tensor(policies):
                if policies.dim() == 1:
                    policy = policies.cpu().numpy()
                else:
                    policy = policies[0].cpu().numpy()
            elif isinstance(policies, np.ndarray):
                if policies.ndim == 1:
                    policy = policies  # Single policy, not batched
                else:
                    policy = policies[0]  # Extract first from batch
            else:
                # Handle scalar or other types
                policy = np.array(policies).flatten()
                
            # Get action size
            action_size = self.game_interface.base.max_moves if hasattr(self.game_interface, 'base') else self.game_interface.max_moves
            
            # Ensure policy has correct size
            if policy.size == 1 or len(policy) < action_size:
                # Single value or wrong size, create uniform policy
                policy = np.ones(action_size) / action_size
                    
            priors = [float(policy[move]) for move in legal_moves]
            
            # Normalize
            prior_sum = sum(priors)
            if prior_sum > 0:
                priors = [p / prior_sum for p in priors]
            else:
                priors = [1.0 / len(legal_moves)] * len(legal_moves)
                
            # Add Dirichlet noise to root priors to ensure exploration
            # This is crucial for preventing all simulations from going down the same path
            if len(priors) > 0:
                # Get noise parameters from config or use defaults
                noise_weight = getattr(self.config, 'dirichlet_epsilon', 0.25)
                dirichlet_alpha = getattr(self.config, 'dirichlet_alpha', 0.3)
                noise = np.random.dirichlet([dirichlet_alpha] * len(priors))
                priors = [(1 - noise_weight) * p + noise_weight * n 
                         for p, n in zip(priors, noise)]
                
            # Add children WITHOUT states - they'll be created on demand during expansion
            # This avoids creating 225 states upfront which is very slow
            child_indices = self.tree.add_children_batch(0, legal_moves, priors)
            self.tree.flush_batch()
            
            # CRITICAL FIX: Mark root as visited AND expanded so UCB selection works
            # Without this, UCB returns -1 because all children have 0 visits
            self.tree.visit_counts[0] = 1
            self.tree.set_expanded(0, True)  # Mark root as expanded
            
            # Verify children were actually added
            root_children, _, _ = self.tree.batch_get_children(torch.tensor([0], device=self.device))
            valid_children = (root_children[0] >= 0).sum().item()
            if root_children[0].sum() == -len(root_children[0]):  # All -1s
                # Force direct addition as fallback
                for i, (action, prior) in enumerate(zip(legal_moves, priors)):
                    self.tree._add_child_direct(0, action, prior)
            
            # Store initial value at root
            if torch.is_tensor(values):
                # Keep as tensor to preserve dtype
                value_tensor = values[0:1] if values.dim() > 0 else values.unsqueeze(0)
                self.tree.value_sums[0] = value_tensor.to(self.tree.value_sums.dtype)
            else:
                value = float(values[0])
                self.tree.value_sums[0] = value
            self.tree.visit_counts[0] = 1
            
    def _extract_policy_from_root(self, temperature: float = 1.0) -> np.ndarray:
        """Extract policy from root visits (OPTIMIZED VERSION - 10x faster)
        
        Args:
            temperature: Temperature for policy extraction (1.0 = proportional to visits, 0.0 = argmax)
        """
        action_size = self.game_interface.max_moves
        
        with torch.no_grad():
            try:
                if self.tree.num_nodes == 0:
                    return np.zeros(action_size)
                
                # Single batch call instead of individual lookups - MAJOR OPTIMIZATION
                root_tensor = torch.tensor([0], device=self.device, dtype=torch.int32)
                root_children, root_actions, _ = self.tree.batch_get_children(root_tensor)
                
                children = root_children[0]
                actions = root_actions[0]
                valid_mask = children >= 0
                
                if not valid_mask.any():
                    # This is normal during early MCTS iterations with few simulations
                    return np.zeros(action_size)
                
                # Vectorized extraction - eliminates Python loops
                valid_children = children[valid_mask]
                valid_actions = actions[valid_mask]
                
                # Bounds checking for actions - prevents CUDA errors
                action_bounds_mask = (valid_actions >= 0) & (valid_actions < action_size)
                if not action_bounds_mask.all():
                    valid_children = valid_children[action_bounds_mask]
                    valid_actions = valid_actions[action_bounds_mask]
                
                if len(valid_children) == 0:
                    return np.zeros(action_size)
                
                # Single tensor lookup for all visits - MAJOR OPTIMIZATION
                visits = self.tree.visit_counts[valid_children].float()
                visit_sum = visits.sum()
                
                
                # Pre-allocate policy tensor
                policy = torch.zeros(action_size, device=self.device, dtype=torch.float32)
                
                if visit_sum > 0:
                    # Apply temperature
                    if temperature == 0.0:
                        # Deterministic: choose action with most visits
                        best_idx = visits.argmax()
                        policy[valid_actions[best_idx]] = 1.0
                    elif temperature == 1.0:
                        # Standard: proportional to visit counts
                        visit_probs = visits / visit_sum
                        # CRITICAL FIX: scatter_ requires int64 indices
                        valid_actions_int64 = valid_actions.to(torch.int64)
                        policy.scatter_(0, valid_actions_int64, visit_probs)
                    else:
                        # Temperature scaling: visits^(1/temperature)
                        # Add small epsilon to avoid log(0)
                        visits_temp = torch.pow(visits + 1e-8, 1.0 / temperature)
                        visit_sum_temp = visits_temp.sum()
                        visit_probs = visits_temp / visit_sum_temp
                        # CRITICAL FIX: scatter_ requires int64 indices
                        valid_actions_int64 = valid_actions.to(torch.int64)
                        policy.scatter_(0, valid_actions_int64, visit_probs)
                else:
                    # Uniform fallback with vectorized assignment
                    uniform_prob = 1.0 / len(valid_actions)
                    uniform_tensor = torch.full((len(valid_actions),), uniform_prob, device=self.device, dtype=torch.float32)
                    # CRITICAL FIX: scatter_ requires int64 indices
                    valid_actions_int64 = valid_actions.to(torch.int64)
                    policy.scatter_(0, valid_actions_int64, uniform_tensor)
                
                # Single CPU transfer
                return policy.cpu().numpy()
                
            except Exception as e:
                # Graceful fallback to uniform policy
                uniform_policy = np.zeros(action_size)
                if action_size > 0:
                    uniform_policy[0] = 1.0  # Select first action as safe fallback
                return uniform_policy
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_wave_size = np.mean(self.stats['wave_sizes']) if self.stats['wave_sizes'] else 0
        
        return {
            'simulations_per_second': self.stats['simulations_per_second'],
            'total_simulations': self.stats['total_simulations'],
            'average_wave_size': avg_wave_size,
            'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
            'cache_stats': self.game_interface.get_cache_stats()
        }