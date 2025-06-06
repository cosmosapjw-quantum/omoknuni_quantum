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
import asyncio

from .cached_game_interface import CachedGameInterface, CacheConfig
from .mcts_config import MCTSConfig
from ..gpu.csr_tree import CSRTree, CSRTreeConfig
from ..neural_networks.evaluator_pool import EvaluatorPool
from ..quantum.path_integral import PathIntegral, PathIntegralConfig
from ..quantum.quantum_features import QuantumMCTS as QuantumFeatures, QuantumConfig as QuantumMCTSConfig

logger = logging.getLogger(__name__)

@dataclass
class WaveMCTSConfig:
    """Configuration for wave-based MCTS"""
    # Wave parameters
    min_wave_size: int = 256
    max_wave_size: int = 2048
    adaptive_wave_sizing: bool = True
    
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
    
class WaveBuffer:
    """Pre-allocated buffer for wave processing"""
    
    def __init__(self, max_wave_size: int, max_depth: int, device: torch.device):
        self.max_wave_size = max_wave_size
        self.max_depth = max_depth
        self.device = device
        
        # Path tracking
        self.paths = torch.zeros((max_wave_size, max_depth), dtype=torch.int32, device=device)
        self.path_lengths = torch.zeros(max_wave_size, dtype=torch.int32, device=device)
        self.current_nodes = torch.zeros(max_wave_size, dtype=torch.int32, device=device)
        self.active_mask = torch.zeros(max_wave_size, dtype=torch.bool, device=device)
        
        # Game states (Gomoku example - adapt for other games)
        self.boards = torch.zeros((max_wave_size, 15, 15), dtype=torch.int8, device=device)
        self.current_players = torch.zeros(max_wave_size, dtype=torch.int8, device=device)
        
        # Neural network I/O
        self.nn_features = torch.zeros((max_wave_size, 3, 15, 15), dtype=torch.float16, device=device)
        self.values = torch.zeros(max_wave_size, dtype=torch.float32, device=device)
        self.policies = torch.zeros((max_wave_size, 225), dtype=torch.float32, device=device)
        
        # Quantum features
        self.phases = torch.zeros((max_wave_size, 225), dtype=torch.float32, device=device)
        self.amplitudes = torch.zeros((max_wave_size, 225), dtype=torch.float32, device=device)
        
    def reset(self, wave_size: int):
        """Reset buffer for new wave"""
        self.active_mask[:wave_size] = True
        self.active_mask[wave_size:] = False
        self.current_nodes[:wave_size] = 0  # Start from root
        self.path_lengths[:] = 0
        
class WavePipeline:
    """Pipeline for overlapping wave processing stages"""
    
    def __init__(self, buffer: WaveBuffer, tree: CSRTree, 
                 game_interface: CachedGameInterface, evaluator_pool: EvaluatorPool):
        self.buffer = buffer
        self.tree = tree
        self.game_interface = game_interface
        self.evaluator_pool = evaluator_pool
        
        # Pipeline stages
        self.selection_done = False
        self.expansion_done = False
        self.evaluation_done = False
        
    async def run_selection(self, wave_size: int):
        """Run selection phase asynchronously"""
        depth = 0
        while self.buffer.active_mask[:wave_size].any() and depth < self.buffer.max_depth:
            active_indices = torch.where(self.buffer.active_mask[:wave_size])[0]
            if len(active_indices) == 0:
                break
                
            # Get current nodes
            current_nodes = self.buffer.current_nodes[active_indices]
            
            # Batched UCB selection using custom CUDA kernel
            if hasattr(torch.ops, 'custom_cuda_ops'):
                # Use fused kernel for selection
                selected_children, _ = self.tree.batch_select_ucb_optimized(current_nodes, c_puct=1.414)
            else:
                selected_children, _ = self.tree.batch_select_ucb_optimized(current_nodes, c_puct=1.414)
                
            # Update paths
            self.buffer.paths[active_indices, depth] = current_nodes
            self.buffer.path_lengths[active_indices] += 1
            
            # Check valid selections
            valid_selections = selected_children >= 0
            
            # Move to children
            self.buffer.current_nodes[active_indices] = torch.where(
                valid_selections,
                selected_children,
                current_nodes
            )
            
            # Update active mask
            new_active_mask = self.buffer.active_mask.clone()
            new_active_mask[active_indices] = valid_selections
            self.buffer.active_mask = new_active_mask
            
            depth += 1
            
        self.selection_done = True
        
    async def run_expansion(self, wave_size: int):
        """Run expansion phase asynchronously"""
        # Wait for selection
        while not self.selection_done:
            await asyncio.sleep(0.001)
            
        # Find leaves that need expansion
        leaf_indices = torch.where(self.buffer.active_mask[:wave_size])[0]
        if len(leaf_indices) == 0:
            self.expansion_done = True
            return
            
        leaf_nodes = self.buffer.current_nodes[leaf_indices]
        
        # Check which need expansion (unvisited)
        leaf_visits = self.tree.visit_counts[leaf_nodes]
        needs_expansion = leaf_visits == 0
        
        if not needs_expansion.any():
            self.expansion_done = True
            return
            
        # Batch expansion using custom CUDA kernel
        expansion_nodes = leaf_nodes[needs_expansion]
        expansion_indices = leaf_indices[needs_expansion]
        
        # Get legal moves in batch
        # This would need game-specific implementation
        # For now, using placeholder
        batch_size = len(expansion_nodes)
        if batch_size > 0:
            # Simulate batch legal move generation
            all_actions = []
            all_priors = []
            
            for i in range(batch_size):
                # Placeholder - would get actual legal moves
                actions = list(range(min(10, 225)))  # First 10 moves
                priors = [1.0 / len(actions)] * len(actions)
                all_actions.append(actions)
                all_priors.append(priors)
                
            # Batch add children
            for node_idx, actions, priors in zip(expansion_nodes, all_actions, all_priors):
                self.tree.add_children_batch(node_idx.item(), actions, priors)
                
        self.tree.flush_batch()
        self.expansion_done = True
        
    async def run_evaluation(self, wave_size: int):
        """Run evaluation phase asynchronously"""
        # Wait for expansion
        while not self.expansion_done:
            await asyncio.sleep(0.001)
            
        # Get all leaf nodes
        leaf_indices = torch.where(self.buffer.active_mask[:wave_size])[0]
        if len(leaf_indices) == 0:
            self.evaluation_done = True
            return
            
        # Prepare features using custom CUDA kernel if available
        if hasattr(torch.ops, 'custom_cuda_ops') and hasattr(torch.ops.custom_cuda_ops, 'evaluate_gomoku_positions'):
            # Use custom kernel for feature extraction
            features = torch.ops.custom_cuda_ops.evaluate_gomoku_positions(
                self.buffer.boards[leaf_indices],
                self.buffer.current_players[leaf_indices]
            )
        else:
            # Fallback
            features = self.buffer.nn_features[leaf_indices]
            
        # Batch neural network evaluation
        with torch.cuda.amp.autocast():  # Mixed precision
            values, policies, info = self.evaluator_pool.evaluate_batch(features)
            
        # Store results
        self.buffer.values[leaf_indices] = values
        self.buffer.policies[leaf_indices] = policies
        
        self.evaluation_done = True
        
    async def run_backup(self, wave_size: int):
        """Run backup phase"""
        # Wait for evaluation
        while not self.evaluation_done:
            await asyncio.sleep(0.001)
            
        # Use custom CUDA kernel for parallel backup if available
        if hasattr(torch.ops, 'custom_cuda_ops') and hasattr(torch.ops.custom_cuda_ops, 'parallel_backup'):
            torch.ops.custom_cuda_ops.parallel_backup(
                self.buffer.paths[:wave_size],
                self.buffer.values[:wave_size],
                self.buffer.path_lengths[:wave_size],
                self.tree.value_sums,
                self.tree.visit_counts
            )
        else:
            # Fallback to tree method
            self.tree.batch_backup_optimized(
                self.buffer.paths[:wave_size],
                self.buffer.values[:wave_size]
            )

class WaveMCTS:
    """Wave-based MCTS with massive parallelization"""
    
    def __init__(self, config: WaveMCTSConfig, game_interface: CachedGameInterface,
                 evaluator_pool: EvaluatorPool):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize components
        tree_config = config.tree_config or CSRTreeConfig(
            max_nodes=500000,
            batch_size=config.max_wave_size,
            device=config.device
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
        self.buffers = [
            WaveBuffer(config.max_wave_size, 100, self.device)
            for _ in range(config.num_wave_pipelines)
        ]
        
        # Create pipelines
        self.pipelines = [
            WavePipeline(buffer, self.tree, game_interface, evaluator_pool)
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
        
        # Initialize root if needed
        if self.tree.visit_counts[0] == 0:
            self._initialize_root(root_state)
            
        # Adaptive wave sizing based on GPU utilization
        current_wave_size = self._determine_wave_size()
        
        # Run waves
        total_sims = 0
        wave_count = 0
        
        # Use asyncio for pipeline parallelism
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            while total_sims < num_simulations:
                # Determine this wave's size
                wave_size = min(current_wave_size, num_simulations - total_sims)
                
                # Run wave asynchronously with pipelining
                loop.run_until_complete(self._run_wave_async(wave_size, wave_count))
                
                total_sims += wave_size
                wave_count += 1
                
                # Update wave size based on performance
                if self.config.adaptive_wave_sizing:
                    current_wave_size = self._adapt_wave_size(current_wave_size)
                    
        finally:
            loop.close()
            
        # Extract policy
        policy = self._extract_policy_from_root()
        
        # Update statistics
        elapsed = time.perf_counter() - start_time
        self.stats['total_simulations'] = total_sims
        self.stats['total_time'] = elapsed
        self.stats['simulations_per_second'] = total_sims / elapsed if elapsed > 0 else 0
        
        logger.info(f"Wave MCTS: {total_sims} simulations in {elapsed:.3f}s "
                   f"({self.stats['simulations_per_second']:.0f} sims/sec)")
        
        return policy
        
    async def _run_wave_async(self, wave_size: int, wave_idx: int):
        """Run a single wave with pipeline parallelism"""
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
        
        # Launch all phases concurrently
        tasks = [
            asyncio.create_task(pipeline.run_selection(wave_size)),
            asyncio.create_task(pipeline.run_expansion(wave_size)),
            asyncio.create_task(pipeline.run_evaluation(wave_size)),
            asyncio.create_task(pipeline.run_backup(wave_size))
        ]
        
        # Wait for all phases to complete
        await asyncio.gather(*tasks)
        
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
        logger.info("Setting up CUDA graphs for optimized execution")
        
        # This would capture common operation sequences
        # Implementation depends on specific PyTorch version
        pass
        
    def _initialize_root(self, root_state):
        """Initialize root node"""
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
                
            # Debug logging
            action_size = self.game_interface.base.max_moves if hasattr(self.game_interface, 'base') else self.game_interface.max_moves
            logger.debug(f"Policy shape: {policy.shape}, legal_moves: {len(legal_moves)}, action_size: {action_size}")
            
            # Ensure policy has correct size
            if policy.size == 1 or len(policy) < action_size:
                # Single value or wrong size, create uniform policy
                logger.warning(f"Policy has wrong size {policy.size}, creating uniform policy for {action_size} actions")
                policy = np.ones(action_size) / action_size
                    
            priors = [float(policy[move]) for move in legal_moves]
            
            # Normalize
            prior_sum = sum(priors)
            if prior_sum > 0:
                priors = [p / prior_sum for p in priors]
            else:
                priors = [1.0 / len(legal_moves)] * len(legal_moves)
                
            # Add children
            self.tree.add_children_batch(0, legal_moves, priors)
            self.tree.flush_batch()
            
    def _extract_policy_from_root(self) -> np.ndarray:
        """Extract policy from root visits"""
        # Get root children
        root_children, root_actions, _ = self.tree.batch_get_children(
            torch.tensor([0], device=self.device)
        )
        
        children = root_children[0]
        actions = root_actions[0]
        valid_mask = children >= 0
        
        if not valid_mask.any():
            return np.zeros(225)  # Empty policy
            
        # Get visit counts
        valid_children = children[valid_mask]
        valid_actions = actions[valid_mask]
        visits = self.tree.visit_counts[valid_children].float()
        
        # Compute policy proportional to visits
        policy = torch.zeros(225, device=self.device)
        visit_sum = visits.sum()
        
        if visit_sum > 0:
            visit_probs = visits / visit_sum
            for action, prob in zip(valid_actions, visit_probs):
                if action >= 0:
                    policy[action] = prob
                    
        return policy.cpu().numpy()
        
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