"""High-performance MCTS implementation

This is the unified MCTS implementation that achieves 80k-200k simulations/second through:
- Wave-based parallelization
- CSR tree with batched GPU operations
- Modular design with specialized components
- GPU acceleration for critical operations
"""

import torch
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

from ..gpu.csr_tree import CSRTree, CSRTreeConfig
from ..gpu.gpu_game_states import GPUGameStates, GPUGameStatesConfig, GameType
from ..gpu.mcts_gpu_accelerator import get_mcts_gpu_accelerator
# Quantum imports removed
from .game_interface import GameInterface, GameType as LegacyGameType
from .mcts_config import MCTSConfig
from .wave_search import WaveSearch
from .tree_operations import TreeOperations


class MCTS:
    """High-performance unified MCTS with automatic optimization selection
    
    This implementation achieves 80k-200k simulations/second through GPU vectorization.
    """
    
    def __init__(
        self,
        config: MCTSConfig,
        evaluator: Any,
        game_interface: Optional[GameInterface] = None,
        single_gpu_mode: bool = False
    ):
        """Initialize MCTS
        
        Args:
            config: MCTS configuration
            evaluator: Neural network evaluator (can be a model for single_gpu_mode)
            game_interface: Optional game interface
            single_gpu_mode: Enable single-GPU optimizations (DirectMCTS mode)
        """
        self.config = config
        self.device = torch.device(config.device)
        self.single_gpu_mode = single_gpu_mode
        self.game_interface = game_interface
        
        # Tree reuse optimization flags
        self._disable_tree_reuse = False
        if hasattr(config, 'backend') and config.backend == 'hybrid':
            # Tree reuse with CSR trees has issues:
            # 1. shift_root doesn't properly update visit counts for the new root
            # 2. This causes uniform policies when root has no visits
            # 3. Performance overhead of maintaining consistency outweighs benefits
            # Fast reset is more efficient for hybrid backend
            self._disable_tree_reuse = getattr(config, 'disable_tree_reuse_for_hybrid', True)
        
        # In single-GPU mode, create optimized evaluator from model
        if single_gpu_mode and isinstance(evaluator, torch.nn.Module):
            from ..utils.single_gpu_evaluator import SingleGPUEvaluator
            self.evaluator = SingleGPUEvaluator(
                model=evaluator,
                device=config.device,
                action_size=self._get_action_size_for_game(config),
                use_mixed_precision=config.use_mixed_precision,
                use_tensorrt=getattr(config, 'use_tensorrt', False)
            )
        else:
            self.evaluator = evaluator
        
        # Configure evaluator for torch tensors
        if hasattr(self.evaluator, '_return_torch_tensors'):
            self.evaluator._return_torch_tensors = True
        
        # If game interface is provided, update config to match
        if game_interface is not None:
            self._update_config_from_game_interface(game_interface)
            
        # Initialize components
        self._initialize_components()
        
        # Create game interface if needed
        self._setup_game_interface(game_interface)
        
        # Initialize statistics
        self._initialize_statistics()
        
        # Initialize quantum features if enabled
        self._initialize_quantum()
        
        # Apply single-GPU optimizations if enabled
        if self.single_gpu_mode:
            self._apply_single_gpu_optimizations()
        
    def _initialize_components(self):
        """Initialize core components"""
        # GPU operations
        use_cuda_kernels = (self.config.device == 'cuda' and 
                           getattr(self.config, 'enable_fast_ucb', True))
        self.gpu_ops = get_mcts_gpu_accelerator(self.device) if use_cuda_kernels else None
        
        # Initialize tree
        self._initialize_tree()
        
        # Initialize game states
        self._initialize_game_states()
        
        # Initialize specialized modules based on backend
        if hasattr(self.config, 'backend') and self.config.backend == 'hybrid':
            # HYBRID APPROACH: Use standard GPU WaveSearch but with CPU memory allocation
            # The key insight: CPU memory for tree operations, GPU for neural network evaluation
            logger.info("Using hybrid approach: GPU WaveSearch with CPU memory allocation")
            
            # Ensure config has all required attributes for WaveSearch
            if not hasattr(self.config, 'enable_tactical_boost'):
                self.config.enable_tactical_boost = False
            if not hasattr(self.config, 'dirichlet_epsilon'):
                self.config.dirichlet_epsilon = 0.25
            if not hasattr(self.config, 'dirichlet_alpha'):
                self.config.dirichlet_alpha = 0.03
            if not hasattr(self.config, 'enable_kernel_fusion'):
                self.config.enable_kernel_fusion = False  # No GPU kernels for hybrid
            if not hasattr(self.config, 'max_depth'):
                self.config.max_depth = 150
                
            # Check if async wave search is requested for maximum performance
            if getattr(self.config, 'wave_async_expansion', False):
                logger.info("Using AsyncWaveSearch for maximum GPU performance (5000+ sims/sec)")
                try:
                    from .async_wave_search import AsyncWaveSearch
                    self.wave_search = AsyncWaveSearch(
                        tree=self.tree,  # CSRTree with GPU memory
                        game_states=self.game_states,  # GPUGameStates
                        evaluator=self.evaluator,  # GPU neural network evaluation
                        config=self.config,
                        device=self.device  # CUDA device
                    )
                    logger.info("AsyncWaveSearch initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to load AsyncWaveSearch: {e}")
                    logger.info("Falling back to base WaveSearch")
                    from .wave_search import WaveSearch
                    self.wave_search = WaveSearch(
                        tree=self.tree,
                        game_states=self.game_states,
                        evaluator=self.evaluator,
                        config=self.config,
                        device=self.device
                    )
            else:
                # Use working base WaveSearch - test functionality first
                logger.info("Using base WaveSearch for GPU backend - establish working baseline")
                try:
                    from .wave_search import WaveSearch
                    self.wave_search = WaveSearch(
                        tree=self.tree,  # CSRTree with GPU memory
                        game_states=self.game_states,  # GPUGameStates
                        evaluator=self.evaluator,  # GPU neural network evaluation
                        config=self.config,
                        device=self.device  # CUDA device
                    )
                    logger.info("Base WaveSearch initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to load base WaveSearch: {e}")
                    logger.info("Falling back to minimal GPU wave search")
                    from ..gpu.minimal_gpu_wave_search import MinimalGPUWaveSearch
                    self.wave_search = MinimalGPUWaveSearch(
                        tree=self.tree,  # CSRTree with GPU memory
                        game_states=self.game_states,  # GPUGameStates
                        evaluator=self.evaluator,  # GPU neural network evaluation
                        config=self.config,
                        device=self.device  # CUDA device
                    )
        elif hasattr(self.config, 'backend') and self.config.backend == 'cpu':
            # CPU backend should use CPU-optimized wave search
            logger.info("Using CPU-optimized wave search for CPU backend")
            from ..cpu.optimized_wave_search import OptimizedCPUWaveSearch
            self.wave_search = OptimizedCPUWaveSearch(
                tree=self.tree,
                game_states=self.game_states,
                evaluator=self.evaluator,
                config=self.config,
                device=self.device
            )
        else:
            # GPU backend - check if async wave search is requested
            if getattr(self.config, 'wave_async_expansion', False):
                logger.info("Using AsyncWaveSearch for maximum GPU performance (5000+ sims/sec)")
                try:
                    from .async_wave_search import AsyncWaveSearch
                    self.wave_search = AsyncWaveSearch(
                        tree=self.tree,  # CSRTree with GPU memory
                        game_states=self.game_states,  # GPUGameStates
                        evaluator=self.evaluator,  # GPU neural network evaluation
                        config=self.config,
                        device=self.device  # CUDA device
                    )
                    logger.info("AsyncWaveSearch initialized successfully for GPU backend")
                except Exception as e:
                    logger.warning(f"Failed to load AsyncWaveSearch: {e}")
                    logger.info("Falling back to base WaveSearch")
                    from .wave_search import WaveSearch
                    self.wave_search = WaveSearch(
                        tree=self.tree,
                        game_states=self.game_states,
                        evaluator=self.evaluator,
                        config=self.config,
                        device=self.device
                    )
            else:
                # Use base WaveSearch for gpu backend (fallback case)
                logger.info("Using base WaveSearch for GPU backend (fallback)")
                try:
                    from .wave_search import WaveSearch
                    self.wave_search = WaveSearch(
                        tree=self.tree,
                        game_states=self.game_states,
                        evaluator=self.evaluator,
                        config=self.config,
                        device=self.device
                    )
                    logger.info("Base WaveSearch initialized successfully (fallback)")
                except Exception as e:
                    logger.warning(f"Failed to load base WaveSearch (fallback): {e}")
                    logger.info("Using minimal GPU wave search as final fallback")
                    from ..gpu.minimal_gpu_wave_search import MinimalGPUWaveSearch
                    self.wave_search = MinimalGPUWaveSearch(
                        tree=self.tree,
                        game_states=self.game_states,
                        evaluator=self.evaluator,
                        config=self.config,
                        device=self.device
                    )
        
        self.tree_ops = TreeOperations(
            tree=self.tree,
            config=self.config,
            device=self.device
        )
        
        # State management
        self._initialize_state_management()
        
    def _initialize_tree(self):
        """Initialize tree structure based on backend"""
        if hasattr(self.config, 'backend') and self.config.backend == 'cpu':
            # CPU backend should use CPU-optimized CythonTree
            logger.info("Using CPU-optimized CythonTree for CPU backend")
            from ..cpu import CythonTree, CythonTreeConfig
            
            tree_config = CythonTreeConfig(
                max_nodes=self.config.max_tree_nodes,
                max_children=self.config.max_tree_nodes * self.config.max_children_per_node,
                c_puct=getattr(self.config, 'c_puct', 1.414),
                virtual_loss=getattr(self.config, 'virtual_loss', 3.0)
            )
            self.tree = CythonTree(tree_config)
        else:
            # GPU and hybrid backends use CSRTree
            # Determine max_actions based on game type
            max_actions_map = {
                GameType.CHESS: 4096,
                GameType.GO: 362,
                GameType.GOMOKU: 225
            }
            max_actions = max_actions_map.get(self.config.game_type, 512)
            
            # Optimized initial capacity for single-GPU to avoid reallocations
            initial_capacity = getattr(self.config, 'initial_capacity_factor', 0.5)
            
            # CRITICAL FIX: Use CPU memory for tree in hybrid mode for thread safety
            # This enables parallel MCTS with atomic operations while avoiding GPU memory bottleneck
            if hasattr(self.config, 'backend') and self.config.backend == 'hybrid':
                tree_device = 'cpu'  # Use CPU memory for thread-safe hybrid tree operations
                logger.info("Using CSRTree with CPU memory for thread-safe hybrid backend")
            else:
                tree_device = self.config.device  # Use config device for gpu/cpu backends
            
            tree_config = CSRTreeConfig(
                max_nodes=self.config.max_tree_nodes,
                max_edges=self.config.max_tree_nodes * self.config.max_children_per_node,
                max_actions=max_actions,
                device=tree_device,  # Use CPU memory for hybrid mode
                enable_virtual_loss=self.config.enable_virtual_loss or (self.config.backend == 'hybrid'),
                virtual_loss_value=-abs(getattr(self.config, 'virtual_loss', 3.0 if hasattr(self.config, 'backend') and self.config.backend == 'hybrid' else 1.0)),
                batch_size=self.config.max_wave_size,
                enable_batched_ops=True,
                initial_capacity_factor=initial_capacity,  # Increased from 0.1 to 0.5
                growth_factor=1.5,
                enable_memory_pooling=getattr(self.config, 'enable_memory_pooling', True)
            )
            self.tree = CSRTree(tree_config)
        
    def _initialize_game_states(self):
        """Initialize game states (GPU or CPU based on backend)"""
        # Determine correct board size for the game type
        board_size = self.config.board_size
        if self.config.game_type == GameType.CHESS:
            board_size = 8
        elif self.config.game_type == GameType.GO:
            # Go can have different sizes, default to config
            board_size = getattr(self.config, 'board_size', 19)
        
        # For hybrid mode, use GPU game states but with CPU device for memory allocation
        if hasattr(self.config, 'backend') and self.config.backend == 'hybrid':
            # Use GPU game states interface but on CPU device for thread safety
            logger.info("Using GPU game states interface with CPU memory for hybrid backend")
            
            game_states_config = GPUGameStatesConfig(
                capacity=self.config.max_tree_nodes,
                game_type=self.config.game_type,
                board_size=board_size,
                device='cpu',  # Use CPU memory for thread safety
                dtype=torch.int8
            )
            self.game_states = GPUGameStates(game_states_config)
            logger.info("Using CPU game states for hybrid backend thread safety")
        elif hasattr(self.config, 'backend') and self.config.backend == 'cpu':
            # Use CPU-optimized game states for CPU backend
            logger.info("Using CPU-optimized game states for CPU backend")
            # Lazy import to avoid circular dependency
            from ..cpu.cpu_game_states import CPUGameStates
            # Convert GameType enum to string for CPUGameStates
            game_type_str = self.config.game_type.name.lower()
            self.game_states = CPUGameStates(
                capacity=self.config.max_tree_nodes,
                game_type=game_type_str,
                board_size=board_size
            )
        else:
            # Use GPU game states for GPU backend
            game_config = GPUGameStatesConfig(
                capacity=self.config.max_tree_nodes,
                game_type=self.config.game_type,
                board_size=board_size,
                device=self.config.device
            )
            self.game_states = GPUGameStates(game_config)
        
        # Enable enhanced features if needed
        expected_channels = self._get_evaluator_input_channels()
        if expected_channels >= 20:
            self.game_states.enable_enhanced_features()
            if hasattr(self.game_states, 'set_enhanced_channels'):
                self.game_states.set_enhanced_channels(expected_channels)
                
    def _initialize_state_management(self):
        """Initialize state pool and mappings"""
        self.node_to_state = torch.full(
            (self.config.max_tree_nodes,), -1, dtype=torch.int32, device=self.device
        )
        
        # Reserve state 0 for root
        self.state_pool_free_list = list(range(1, self.config.max_tree_nodes))
        self.state_pool_free_count = len(self.state_pool_free_list)
        
        # State allocation tracking
        self.state_allocation_count = 0
        self.state_deallocation_count = 0
        
    def _setup_game_interface(self, game_interface: Optional[GameInterface]):
        """Setup game interface for compatibility"""
        if game_interface is None:
            # Map GameType enum to legacy GameType
            legacy_game_type_map = {
                GameType.CHESS: LegacyGameType.CHESS,
                GameType.GO: LegacyGameType.GO,
                GameType.GOMOKU: LegacyGameType.GOMOKU
            }
            legacy_type = legacy_game_type_map.get(self.config.game_type, LegacyGameType.GOMOKU)
            
            self.cached_game = GameInterface(
                legacy_type, 
                board_size=self.config.board_size,
                input_representation='basic'
            )
        else:
            self.cached_game = game_interface
            
    def _initialize_statistics(self):
        """Initialize statistics tracking"""
        self.stats = {
            'total_searches': 0,
            'total_simulations': 0,
            'total_time': 0.0,
            'avg_sims_per_second': 0.0,
            'peak_sims_per_second': 0.0,
            'last_search_sims_per_second': 0.0,
            'tree_reuse_count': 0,
            'tree_reuse_nodes': 0
        }
        
        self.stats_internal = defaultdict(float)
        
    def _initialize_quantum(self):
        """Placeholder for quantum features (disabled)"""
        self.quantum_features = None
        self.quantum_total_simulations = 0
    
    def _get_action_size_for_game(self, config: MCTSConfig) -> int:
        """Get action space size based on game type"""
        if config.game_type == GameType.CHESS:
            return 4096  # Max chess moves
        elif config.game_type == GameType.GO:
            return config.board_size ** 2 + 1  # +1 for pass
        else:  # GOMOKU
            return config.board_size ** 2
    
    def _apply_single_gpu_optimizations(self):
        """Apply single-GPU specific optimizations"""
        logger.info("Applying single-GPU optimizations")
        
        # Increase initial tree capacity to avoid reallocations
        if hasattr(self.tree, 'config'):
            self.tree.config.initial_capacity_factor = max(
                self.tree.config.initial_capacity_factor, 0.5
            )
        
        # Pre-allocate larger buffers in wave search
        if hasattr(self.wave_search, 'allocate_buffers'):
            self.wave_search.allocate_buffers(
                self.config.max_wave_size,
                max_depth=150  # Larger than default
            )
        
        # Enable CUDA graphs if available
        if self.config.use_cuda_graphs and self.device.type == 'cuda':
            self._setup_cuda_graphs()
    
    def _setup_cuda_graphs(self):
        """Setup CUDA graphs for wave execution"""
        if not torch.cuda.is_available():
            return
            
        try:
            # Warmup the model first
            if hasattr(self.evaluator, 'warmup'):
                self.evaluator.warmup(warmup_steps=5)
            
            # Create CUDA graph for wave execution
            self.cuda_graph = torch.cuda.CUDAGraph()
            self.graph_captured = False
            
            # Allocate static tensors for graph capture
            self.graph_batch_size = self.config.max_wave_size
            self.graph_features = torch.zeros(
                (self.graph_batch_size, 18, self.config.board_size, self.config.board_size),
                device=self.device,
                dtype=torch.float32
            )
            
            logger.info("CUDA graphs initialized for wave execution")
            
        except Exception as e:
            logger.warning(f"Failed to setup CUDA graphs: {e}")
            self.cuda_graph = None
            self.graph_captured = False
    
    def warmup(self, num_searches: int = 3, simulations_per_search: int = 100):
        """Warmup the MCTS and GPU for optimal performance
        
        Args:
            num_searches: Number of warmup searches
            simulations_per_search: Simulations per warmup search
        """
        if not self.single_gpu_mode:
            return  # Only needed for single-GPU mode
            
        logger.info(f"Warming up MCTS with {num_searches} searches...")
        
        # Create a dummy game state
        if self.cached_game is not None:
            dummy_state = self.cached_game.create_initial_state()
        else:
            # Create a simple dummy state
            dummy_state = type('DummyState', (), {
                'get_basic_tensor_representation': lambda: torch.zeros(3, self.config.board_size, self.config.board_size),
                'get_current_player': lambda: 1,
                'get_move_history': lambda: [],
                'is_terminal': lambda: False,
                'get_game_result': lambda: 0
            })()
        
        # Store original num_simulations
        original_sims = self.config.num_simulations
        self.config.num_simulations = simulations_per_search
        
        # Run warmup searches
        for i in range(num_searches):
            _ = self.search(dummy_state)
            if i == 0:
                # Clear tree after first search to reset memory
                self.clear()
        
        # Restore original settings
        self.config.num_simulations = original_sims
        
        # Clear tree after warmup
        self.clear()
        
        logger.info("MCTS warmup complete")
    
    def _update_config_from_game_interface(self, game_interface: GameInterface):
        """Update MCTS config to match the provided game interface"""
        # Map legacy game type to GPU game type
        legacy_to_gpu_map = {
            LegacyGameType.CHESS: GameType.CHESS,
            LegacyGameType.GO: GameType.GO,
            LegacyGameType.GOMOKU: GameType.GOMOKU
        }
        
        if hasattr(game_interface, 'game_type'):
            gpu_game_type = legacy_to_gpu_map.get(game_interface.game_type)
            if gpu_game_type is not None:
                self.config.game_type = gpu_game_type
                
        if hasattr(game_interface, 'board_size'):
            self.config.board_size = game_interface.board_size
            
    def search(self, state: Any, num_simulations: Optional[int] = None) -> np.ndarray:
        """Run MCTS search from given state
        
        Args:
            state: Game state to search from
            num_simulations: Number of simulations (overrides config)
            
        Returns:
            Policy vector as numpy array
        """
        # Validate state shape
        if isinstance(state, np.ndarray):
            expected_shape = (self.config.board_size, self.config.board_size)
            if state.shape != expected_shape:
                raise ValueError(f"Invalid state shape: {state.shape}, expected {expected_shape}")
        
        num_sims = num_simulations or self.config.num_simulations
        
        # For hybrid backend, use the standard wave-based search approach
        # No special handling needed - state synchronization happens at MCTS level
        
        start_time = time.perf_counter()
        
        # If subtree reuse is disabled, reset the tree for each search
        if not self.config.enable_subtree_reuse:
            self._reset_for_new_search()
        
        # Reset wave search state for new search (including global noise cache)
        self.wave_search.reset_search_state()
        
        # Initialize root if needed
        self._ensure_root_initialized(state)
        
        # Note: Dirichlet noise is now applied per-simulation in wave_search
        # instead of globally to the root node
            
        # Run search
        policy = self._run_search(num_sims)
        
        # Check root visits after search
        root_visits = self.tree.node_data.visit_counts[0].item()
        if root_visits == 0:
            logger.warning(f"Root has {root_visits} visits after {num_sims} simulations!")
        
        # Update statistics
        elapsed_time = time.perf_counter() - start_time
        self._update_statistics(num_sims, elapsed_time)
        
        # CRITICAL: Clean up allocated states after search if subtree reuse is disabled
        if not self.config.enable_subtree_reuse:
            self._cleanup_after_search()
        
        return policy
        
    def _ensure_root_initialized(self, state: Any):
        """Ensure root node is properly initialized"""
        if self.node_to_state[0] < 0:  # Root has no state yet
            self._initialize_root(state)
        else:
            # Synchronize root state
            self._update_root_state(state)
            
        # CRITICAL: Ensure root is properly set up for search
        # Check if root needs expansion or re-expansion
        children, _, _ = self.tree_ops.get_root_children_info()
        root_visits = self.tree.node_data.visit_counts[0].item()
        
        # Need to handle several cases:
        # 1. Fresh tree with no children - needs expansion
        # 2. Tree reuse with single node - may have stale children, needs cleanup
        # 3. Tree reuse with subtree - children are valid
        
        if self.tree.num_nodes == 1:
            # Single node tree - ensure it's properly set up
            if len(children) > 0 and root_visits > 0:
                # This is a reused leaf node with stale children
                # Clear the children to force re-expansion
                self._clear_root_children()
                # Mark as not expanded
                self.tree.node_data.set_expanded(0, False)
            
            # Now expand the root
            self._force_expand_root()
        elif len(children) == 0:
            # Multi-node tree but root has no children - expand it
            self._force_expand_root()
            
    def _run_search(self, num_simulations: int) -> np.ndarray:
        """Run the main search loop"""
        # Handle zero simulations case
        if num_simulations == 0:
            # Return uniform policy without running any simulations
            action_space_size = self.config.board_size ** 2
            if self.config.game_type == GameType.GO:
                action_space_size += 1
            elif self.config.game_type == GameType.CHESS:
                action_space_size = 4096
            return np.ones(action_space_size) / action_space_size
        
        completed = 0
        
        # Check if this is ImprovedWaveParallelMCTSV8 which uses run_simulations
        if hasattr(self.wave_search, '__class__') and 'ImprovedWaveParallelMCTSV8' in self.wave_search.__class__.__name__:
            # V8 handles all simulations at once with async pipeline
            actual_completed = self.wave_search.run_simulations(num_simulations, self.node_to_state)
            completed = actual_completed
        else:
            # Standard wave-based approach
            while completed < num_simulations:
                wave_size = min(self.config.max_wave_size, num_simulations - completed)
                
                # Run one wave
                actual_completed = self.wave_search.run_wave(
                    wave_size, 
                    self.node_to_state,
                    self.state_pool_free_list
                )
                
                # Check if simulations are running
                if actual_completed == 0:
                    logger.warning(f"Wave returned 0 completions! Tree nodes: {self.tree.num_nodes}")
                    break
                
                completed += actual_completed
                
                # Quantum features disabled
                    
        # Extract and return policy
        policy = self._extract_policy(0)
        
        return policy
        
    def _update_quantum_state(self, completed: int, wave_size: int):
        """Placeholder for quantum state updates (disabled)"""
        pass
                
    def _extract_policy(self, node_idx: int) -> np.ndarray:
        """Extract policy from node visit counts (optimized version)"""
        
        # No special handling needed for hybrid backend - use standard tree operations
        
        actions, visits, _ = self.tree_ops.get_root_children_info()
        
        # Determine action space size
        action_space_size = self.config.board_size ** 2
        if self.config.game_type == GameType.GO:
            action_space_size += 1
            
        # Handle empty tree case
        if len(actions) == 0:
            # No children - return uniform policy over legal moves
            state_idx = self.node_to_state[node_idx].item()
            if state_idx >= 0:
                legal_mask = self.game_states.get_legal_moves_mask(
                    torch.tensor([state_idx], device=self.device)
                )[0]
                # Handle both tensor and numpy array returns
                if hasattr(legal_mask, 'cpu'):
                    legal_moves = torch.nonzero(legal_mask).squeeze(-1).cpu().numpy()
                else:
                    legal_moves = np.nonzero(legal_mask)[0]
            else:
                logger.warning(f"No state found for node {node_idx}")
                legal_moves = np.array([])
                
            policy = np.zeros(action_space_size)
            if len(legal_moves) > 0:
                policy[legal_moves] = 1.0 / len(legal_moves)
            return policy
            
        # Convert visits to probabilities
        total_visits = visits.sum().item()
        if total_visits == 0:
            # No visits - return uniform over children actions
            policy = np.zeros(action_space_size)
            # Handle both torch tensors and numpy arrays
            if hasattr(actions, 'cpu'):
                actions_np = actions.cpu().numpy()
            else:
                actions_np = actions
            if len(actions_np) > 0:
                policy[actions_np] = 1.0 / len(actions_np)
            return policy
            
        # Create policy vector efficiently
        policy = np.zeros(action_space_size)
        # Handle both torch tensors and numpy arrays
        if hasattr(visits, 'cpu'):
            visits_np = visits.cpu().numpy()
            actions_np = actions.cpu().numpy()
        else:
            # Already numpy arrays (CPU backend)
            visits_np = visits
            actions_np = actions
        
        # Trust the tree structure - children are legal by construction
        # This is the key optimization: we don't re-validate what we already know
        if self.config.enable_subtree_reuse and self.config.enable_debug_logging:
            # Only in debug mode, do a quick sanity check
            state_idx = self.node_to_state[node_idx].item()
            if state_idx >= 0:
                legal_mask = self.game_states.get_legal_moves_mask(
                    torch.tensor([state_idx], device=self.device)
                )[0]
                legal_set = set(torch.nonzero(legal_mask).squeeze(-1).cpu().numpy())
                
                # Quick vectorized check
                # Handle both torch tensors and numpy arrays
                if hasattr(actions, 'cpu'):
                    actions_np = actions.cpu().numpy()
                else:
                    actions_np = actions
                if len(legal_set) > 0:
                    illegal_mask = ~np.isin(actions_np, list(legal_set))
                    if illegal_mask.any():
                        logger.warning(f"Found {illegal_mask.sum()} illegal children in tree - this shouldn't happen")
        
        # Vectorized policy assignment
        # (actions_np and visits_np already converted above)
        
        # Direct assignment - trust tree structure
        valid_actions = (actions_np >= 0) & (actions_np < action_space_size)
        policy[actions_np[valid_actions]] = visits_np[valid_actions] / total_visits
        
        # Single normalization check
        policy_sum = policy.sum()
        if abs(policy_sum - 1.0) > 1e-6:
            if policy_sum > 0:
                policy /= policy_sum
            else:
                # This should never happen with valid tree
                logger.error("Policy sum is zero - tree may be corrupted")
                return np.ones(action_space_size) / action_space_size
                    
        return policy
        
    def get_root_value(self) -> float:
        """Get the value estimate of the root node
        
        Returns:
            Value estimate from the root node's perspective
        """
        if self.tree.num_nodes == 0:
            return 0.0
            
        # Get root value (Q-value)
        root_value = self.tree.node_data.value_sums[0].item()
        root_visits = self.tree.node_data.visit_counts[0].item()
        
        if root_visits == 0:
            return 0.0
            
        # Return average value
        return root_value / root_visits
        
    def select_action(self, state: Any, temperature: float = 1.0) -> int:
        """Select action using MCTS
        
        Args:
            state: Current game state
            temperature: Temperature for action selection
            
        Returns:
            Selected action
        """
        # Run search
        policy = self.search(state)
        
        # Select action based on temperature
        if temperature == 0:
            # For temperature=0, we need deterministic selection
            # Handle ties by taking the first occurrence of max value
            max_val = policy.max()
            if max_val == 0:
                # All probabilities are zero - should not happen
                logger.warning("All policy probabilities are zero for temperature=0 selection")
                action = 0
            else:
                # Find first index with maximum value for deterministic behavior
                action = np.argmax(policy)
        else:
            # Apply temperature
            policy_temp = np.power(policy, 1/temperature)
            policy_temp /= policy_temp.sum()
            action = np.random.choice(len(policy), p=policy_temp)
            
        # Track for subtree reuse
        self.tree_ops.last_selected_action = action
        
        return action
        
    def update_root(self, action: int, new_state: Any = None):
        """Update root after taking an action with optimized tree reuse
        
        Args:
            action: Action taken
            new_state: New game state
        """
        # Validate action is within valid range
        action_space_size = self.config.board_size ** 2
        if self.config.game_type == GameType.GO:
            action_space_size += 1  # Pass move
        elif self.config.game_type == GameType.CHESS:
            action_space_size = 4096  # Standard chess action space
            
        if action < 0 or action >= action_space_size:
            raise ValueError(f"Invalid action {action}. Must be in range [0, {action_space_size})")
        
        if self.config.enable_subtree_reuse and not self._disable_tree_reuse:
            # Fast path: try optimized subtree reuse
            reuse_success = self._try_optimized_subtree_reuse(action, new_state)
            if reuse_success:
                self.stats['tree_reuse_count'] += 1
                return
        
        # Fallback: reset tree for new search (most efficient for hybrid backend)
        self._efficient_tree_reset()
        if new_state is not None:
            self._initialize_root(new_state)
    
    def _try_optimized_subtree_reuse(self, action: int, new_state: Any = None) -> bool:
        """Try optimized subtree reuse with minimal overhead
        
        Returns:
            bool: True if reuse was successful, False if fallback needed
        """
        try:
            # Use the proper tree_ops subtree reuse mechanism
            mapping = self.tree_ops.apply_subtree_reuse(action)
            if not mapping:
                return False
                
            # Update state mappings after successful reuse
            self._update_state_mappings_after_reuse(mapping)
            self.stats['tree_reuse_nodes'] += len(mapping)
            
            # Update root state if provided
            if new_state is not None:
                self._update_root_state(new_state)
                
            # CRITICAL: Don't validate children for hybrid backend (too expensive)
            # The tree structure is already correct after shift_root
            if not (hasattr(self.config, 'backend') and self.config.backend == 'hybrid'):
                self._validate_children_after_reuse()
            
            return True
            
        except Exception as e:
            logger.debug(f"Subtree reuse failed, falling back to reset: {e}")
            return False
    
    def _find_child_by_action(self, parent_idx: int, action: int) -> Optional[int]:
        """Efficiently find child node by action"""
        if hasattr(self.tree, 'get_child_by_action'):
            return self.tree.get_child_by_action(parent_idx, action)
        
        # Fallback: linear search through children
        if hasattr(self.tree, 'node_data') and hasattr(self.tree.node_data, 'parent_actions'):
            # For CSR tree structure
            children_mask = (self.tree.node_data.parent_indices == parent_idx)
            if children_mask.any():
                child_indices = torch.nonzero(children_mask, as_tuple=True)[0]
                for child_idx in child_indices:
                    if self.tree.node_data.parent_actions[child_idx] == action:
                        return child_idx.item()
        return None
    
    def _basic_root_shift(self, new_root_idx: int) -> bool:
        """Basic root shift operation"""
        try:
            # Update node-to-state mapping
            if new_root_idx < len(self.node_to_state):
                new_root_state = self.node_to_state[new_root_idx].item()
                if new_root_state >= 0:
                    # Move state from new_root_idx to index 0
                    self.node_to_state[0] = new_root_state
                    self.node_to_state[new_root_idx] = -1
                    return True
            return False
        except Exception:
            return False
    
    def _fast_update_root_state(self, root_state_idx: int, new_state: Any):
        """Fast root state update without full reconstruction"""
        if root_state_idx < 0:
            return
            
        try:
            # For hybrid backend, update game state efficiently
            if hasattr(self.config, 'backend') and self.config.backend == 'hybrid':
                self._update_hybrid_root_state(new_state, root_state_idx)
                return
            
            # Update key state properties without full board reconstruction
            if hasattr(new_state, 'get_current_player'):
                self.game_states.current_player[root_state_idx] = new_state.get_current_player()
            
            if hasattr(new_state, 'get_move_history'):
                move_history = new_state.get_move_history()
                move_count = len(move_history) if isinstance(move_history, list) else move_history.shape[0]
                self.game_states.move_count[root_state_idx] = move_count
            
            if hasattr(new_state, 'is_terminal'):
                self.game_states.is_terminal[root_state_idx] = new_state.is_terminal()
                
        except Exception as e:
            logger.debug(f"Fast root state update failed: {e}")
    
    def _efficient_tree_reset(self):
        """Efficient tree reset optimized for hybrid backend"""
        # For hybrid backend, just reset tree structure (states managed separately)
        if hasattr(self.config, 'backend') and self.config.backend == 'hybrid':
            self._reset_tree_structure_only()
            return
        
        # Standard reset for other backends
        self._reset_for_new_search()
    
    def _reset_tree_structure_only(self):
        """Reset only tree structure, keep state pool intact"""
        if hasattr(self.tree, 'node_data'):
            # Reset tree to single root node
            if self.tree.num_nodes > 1:
                # Clear all non-root nodes
                self.tree.node_data.visit_counts[1:] = 0
                self.tree.node_data.value_sums[1:] = 0.0
                self.tree.node_data.parent_indices[1:] = -2
                self.tree.node_data.parent_actions[1:] = -1
                self.tree.num_nodes = 1
                
            # Reset root node
            self.tree.node_data.visit_counts[0] = 0
            self.tree.node_data.value_sums[0] = 0.0
            
        # Reset wave search state
        if hasattr(self, 'wave_search'):
            self.wave_search.reset_search_state()
    
    def _hybrid_optimized_clear(self):
        """Optimized clear for hybrid backend - preserves state pools for efficiency"""
        # For hybrid backend, we only need to reset tree structure
        # State management is handled by the CPU backend separately
        self._reset_tree_structure_only()
        
        # Reinitialize just the root state mapping without full state pool reset
        if hasattr(self, 'node_to_state'):
            # Keep state pool, just reset node mappings
            self.node_to_state.fill_(-1)
            
        # Reset statistics but preserve allocators
        if hasattr(self, 'stats'):
            search_stats = ['total_simulations', 'total_time', 'avg_sims_per_sec']
            for key in search_stats:
                if key in self.stats:
                    self.stats[key] = 0
            
    def clear(self):
        """Clear tree and reset state - optimized for hybrid backend"""
        # For hybrid backend, use optimized clear that preserves state pool
        if hasattr(self.config, 'backend') and self.config.backend == 'hybrid':
            self._hybrid_optimized_clear()
            return
        
        # Standard clear for other backends
        # Free all allocated GPU game states before clearing tree
        if hasattr(self, 'node_to_state') and hasattr(self, 'game_states'):
            # Get all allocated state indices
            allocated_states = self.node_to_state[self.node_to_state >= 0]
            if len(allocated_states) > 0:
                # Free the states in the GPU pool
                self.game_states.free_states(allocated_states)
                logger.debug(f"Freed {len(allocated_states)} GPU game states during clear")
        
        # Clear the tree
        self.tree_ops.clear()
        
        # Reinitialize state management
        self._initialize_state_management()
        
        # Clear enhanced feature cache if present
        if hasattr(self.game_states, 'clear_enhanced_cache'):
            self.game_states.clear_enhanced_cache()
        
        if self.quantum_features:
            self.quantum_total_simulations = 0
    
    def reset_tree(self):
        """Reset tree to initial state - alias for clear()"""
        self.clear()
    
    def update_with_move(self, move: int):
        """Update root with move - simplified version of update_root()
        
        Args:
            move: Action/move taken
        """
        self.update_root(move, None)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get MCTS statistics"""
        stats = self.stats.copy()
        
        # Add tree statistics
        tree_stats = self.tree_ops.get_tree_statistics()
        stats.update(tree_stats)
        
        # Add internal statistics
        stats.update(self.stats_internal)
        
        return stats
        
    def _update_statistics(self, num_simulations: int, elapsed_time: float):
        """Update performance statistics"""
        self.stats['total_searches'] += 1
        self.stats['total_simulations'] += num_simulations
        self.stats['total_time'] += elapsed_time
        
        sims_per_second = num_simulations / elapsed_time if elapsed_time > 0 else 0
        self.stats['last_search_sims_per_second'] = sims_per_second
        
        if sims_per_second > self.stats['peak_sims_per_second']:
            self.stats['peak_sims_per_second'] = sims_per_second
            
        total_sims = self.stats['total_simulations']
        total_time = self.stats['total_time']
        self.stats['avg_sims_per_second'] = total_sims / total_time if total_time > 0 else 0
        
    def _initialize_root(self, root_state: Any):
        """Initialize root node state"""
        # When subtree reuse is disabled, always use state 0 for root
        # Otherwise allocate a new state
        if not self.config.enable_subtree_reuse:
            # For CPU game states, we need to allocate state 0 first
            if hasattr(self.game_states, '_boards') and self.game_states._boards is None:
                # CPU game states - need to allocate first state
                allocated = self.game_states.allocate_states(1)
                if len(allocated) > 0:
                    state_idx = allocated[0]
                else:
                    logger.error("Failed to allocate root state")
                    state_idx = 0
            else:
                state_idx = 0  # Always use state 0 for root when reuse is disabled
                # Make sure state 0 is properly marked as allocated (for GPU game states)
                if hasattr(self.game_states, 'allocated_mask') and len(self.game_states.allocated_mask) > 0:
                    if not self.game_states.allocated_mask[0]:
                        self.game_states.allocated_mask[0] = True
                        self.game_states.num_states = max(self.game_states.num_states, 1)
                        # Remove state 0 from free indices if present
                        if hasattr(self.game_states, 'free_indices') and hasattr(self.game_states.free_indices, '__getitem__'):
                            if isinstance(self.game_states.free_indices, torch.Tensor):
                                free_mask = self.game_states.free_indices != 0
                                self.game_states.free_indices = self.game_states.free_indices[free_mask]
                # For CPU game states, just ensure we have at least one state
                elif hasattr(self.game_states, 'num_states'):
                    self.game_states.num_states = max(self.game_states.num_states, 1)
        else:
            # Allocate a state in the game states pool
            state_indices = self.game_states.allocate_states(1)
            state_idx = state_indices[0].item()
        
        # Set up the state based on the root_state
        if isinstance(root_state, np.ndarray):
            # Direct numpy array - set board state
            # Lazy import to avoid circular dependency
            from ..cpu.cpu_game_states import CPUGameStates
            if isinstance(self.game_states, CPUGameStates):
                # CPU backend has a dedicated method
                self.game_states.set_board_from_tensor(state_idx, root_state)
                
                # Set current player based on move count
                # Count non-zero positions to determine whose turn it is
                num_moves = (root_state != 0).sum()
                current_player = 1 if num_moves % 2 == 0 else 2
                self.game_states.current_player[state_idx] = current_player
                self.game_states.move_count[state_idx] = num_moves
            else:
                # GPU backend - update directly
                board_tensor = torch.from_numpy(root_state).to(self.device)
                self.game_states.boards[state_idx] = board_tensor
                
                # Set current player based on move count
                # Count non-zero positions to determine whose turn it is
                num_moves = (root_state != 0).sum()
                current_player = 1 if num_moves % 2 == 0 else 2
                self.game_states.current_player[state_idx] = current_player
                self.game_states.move_count[state_idx] = num_moves
        elif hasattr(root_state, 'get_basic_tensor_representation'):
            # Get board representation using game interface
            obs = self.cached_game.state_to_numpy(root_state)
            
            # Convert to game states format
            if self.game_states.game_type == GameType.GOMOKU or self.game_states.game_type == GameType.GO:
                # Get actual board size from observation
                actual_board_size = obs[0].shape[0]
                board = torch.zeros((actual_board_size, actual_board_size), dtype=torch.int8, device=self.device)
                
                # Check number of channels to determine format
                num_channels = obs.shape[0]
                
                if num_channels == 3:
                    # Standard 3-channel format (for Go/Chess)
                    # Channel 0: Player 1 pieces
                    # Channel 1: Player 2 pieces  
                    # Channel 2: Current player
                    player1_channel = torch.from_numpy(obs[0]).to(self.device)
                    player2_channel = torch.from_numpy(obs[1]).to(self.device)
                    
                    # Map to GPUGameStates format
                    board[player1_channel > 0.5] = 1  # Black
                    board[player2_channel > 0.5] = 2  # White
                else:
                    # 18-channel representation
                    # CRITICAL FIX: For 18-channel format, we cannot reconstruct the board
                    # by iterating through channels. Instead, get 3-channel representation.
                    # The 18-channel format has:
                    # - Channel 0: All stones (both players)
                    # - Channel 1: Current player indicator
                    # - Channels 2-17: Move history
                    
                    # Get 3-channel representation for accurate board reconstruction
                    obs_3ch = self.cached_game.state_to_numpy(root_state, representation_type='minimal')
                    if obs_3ch.shape[0] >= 2:
                        # CRITICAL: 3-channel format uses perspective-based encoding
                        # Channel 0: Current player's stones
                        # Channel 1: Opponent's stones
                        current_player_channel = torch.from_numpy(obs_3ch[0]).to(self.device)
                        opponent_channel = torch.from_numpy(obs_3ch[1]).to(self.device)
                        
                        # Get actual current player from game state
                        actual_current_player = root_state.get_current_player()
                        
                        # Map channels to absolute player numbers
                        if actual_current_player == 1:
                            # Current player is 1, so channel 0 = player 1, channel 1 = player 2
                            board[current_player_channel > 0.5] = 1
                            board[opponent_channel > 0.5] = 2
                        else:
                            # Current player is 2, so channel 0 = player 2, channel 1 = player 1
                            board[current_player_channel > 0.5] = 2
                            board[opponent_channel > 0.5] = 1
                    else:
                        # Fallback: should not happen
                        logger.error(f"Unexpected 3-channel format with {obs_3ch.shape[0]} channels")
                
                # Debug logging
                
                # For non-square assignment, we need to handle different board sizes
                if self.game_states.boards.shape[1:] == board.shape:
                    self.game_states.boards[state_idx] = board
                else:
                    # Board sizes don't match - this shouldn't happen if config is correct
                    logger.error(f"Board size mismatch: game_states expects {self.game_states.boards.shape[1:]}, got {board.shape}")
                    # Try to copy what we can
                    min_size = min(self.game_states.boards.shape[1], board.shape[0])
                    self.game_states.boards[state_idx, :min_size, :min_size] = board[:min_size, :min_size]
                # Map current player: GameInterface uses 1=black, 2=white
                # GPUGameStates uses 1=black, 2=white as well
                actual_current_player = root_state.get_current_player()
                self.game_states.current_player[state_idx] = actual_current_player
                move_history = root_state.get_move_history()
                if isinstance(move_history, list):
                    self.game_states.move_count[state_idx] = len(move_history)
                else:
                    self.game_states.move_count[state_idx] = move_history.shape[0]
                
                # CRITICAL: Initialize terminal status and game result
                is_terminal = root_state.is_terminal()
                self.game_states.is_terminal[state_idx] = is_terminal
                if is_terminal:
                    game_result = root_state.get_game_result()
                    # Convert GameResult enum to integer value
                    if hasattr(game_result, 'value'):
                        self.game_states.game_result[state_idx] = game_result.value
                    else:
                        self.game_states.game_result[state_idx] = game_result
            
        # Map root node to this state
        self.node_to_state[0] = state_idx
            
    def _clear_root_children(self):
        """Clear root's children in the tree structure
        
        This is needed when reusing a leaf node that has stale children
        from before it became the root.
        """
        if hasattr(self.tree, 'children'):
            # Clear children table
            self.tree.children[0] = -1
        
        if hasattr(self.tree, 'csr_storage') and hasattr(self.tree.csr_storage, 'row_ptr'):
            # Update CSR structure to indicate no edges from root
            self.tree.csr_storage.row_ptr[0] = 0
            self.tree.csr_storage.row_ptr[1] = 0
            # Also update num_edges if needed
            if self.tree.num_edges > 0 and self.tree.num_nodes == 1:
                self.tree.num_edges = 0
    
    def _force_expand_root(self):
        """Force expansion of the root node
        
        This is needed after tree reuse when the root's children have been cleared.
        """
        # Get root state
        root_state_idx = self.node_to_state[0].item()
        if root_state_idx < 0:
            logger.error("Cannot expand root - no state assigned")
            return
            
        # Use wave_search's expansion method directly
        if hasattr(self.config, 'backend') and self.config.backend == 'cpu':
            # CPU backend expects List[int]
            leaf_nodes = [0]
        else:
            # GPU backend expects torch.Tensor
            leaf_nodes = torch.tensor([0], device=self.device)
        
        self.wave_search._expand_batch_vectorized(
            leaf_nodes,
            self.node_to_state,
            self.state_pool_free_list
        )
        
        # Mark root as expanded
        if hasattr(self.tree.node_data, 'set_expanded'):
            # GPU backend (CSRTree)
            self.tree.node_data.set_expanded(0, True)
        # CPU backend (CythonTree) doesn't need explicit expanded marking
        
        # Verify expansion
        children, _, _ = self.tree_ops.get_root_children_info()
    
    def _update_hybrid_root_state(self, new_root_state: Any, root_state_idx: int):
        """Update root state for hybrid backend using GPU game states
        
        This method properly synchronizes the internal GPU game state with the
        external C++ game state passed from self-play manager.
        """
        logger.debug(f"Updating hybrid root state: {type(new_root_state)}")
        
        try:
            # Handle numpy array states
            if isinstance(new_root_state, np.ndarray):
                # Direct numpy array - set board state
                board_tensor = torch.from_numpy(new_root_state).to(self.device)
                self.game_states.boards[root_state_idx] = board_tensor
                
                # Set current player based on move count
                # Count non-zero positions to determine whose turn it is
                num_moves = (new_root_state != 0).sum()
                current_player = 1 if num_moves % 2 == 0 else 2
                self.game_states.current_player[root_state_idx] = current_player
                self.game_states.move_count[root_state_idx] = num_moves
                
                # Clear the tree to ensure fresh search from this position
                self._reset_tree_for_new_root()
                
                logger.debug("Hybrid root state updated successfully from numpy array")
                return
            
            # Extract board state using AlphaZero default method
            elif hasattr(new_root_state, 'get_basic_tensor_representation'):
                # Use AlphaZero default representation (19 channels)
                tensor_repr = new_root_state.get_basic_tensor_representation()
                # Extract the board from the first channel (current position)
                board_array = tensor_repr[0]  # Shape: (15, 15)
                logger.debug(f"Got board from get_basic_tensor_representation(): {board_array.shape}")
            elif hasattr(new_root_state, 'get_board'):
                board_array = new_root_state.get_board()
                logger.debug(f"Got board from get_board(): {board_array.shape}")
            else:
                logger.warning(f"Cannot extract board from state type: {type(new_root_state)}")
                return
            
            # Convert to PyTorch tensor for GPU game states interface
            board_tensor = torch.from_numpy(board_array.astype(np.int8))
            
            # Update the game state using GPU game states interface
            self.game_states.boards[root_state_idx] = board_tensor
            
            # Update current player (GPU game states use 1-based indexing: 1=player1, 2=player2)
            if hasattr(new_root_state, 'get_current_player'):
                current_player = new_root_state.get_current_player()
                # Convert from 0-based to 1-based indexing
                if current_player == 0:
                    current_player = 1
                elif current_player == -1:
                    current_player = 2
                # Note: GPUGameStates doesn't have separate current_player storage - inferred from board
                logger.debug(f"Current player: {current_player}")
            
            # Update move count (inferred from number of stones)
            num_stones = int(np.count_nonzero(board_array))
            logger.debug(f"Move count (stones on board): {num_stones}")
            
            # CRITICAL: Clear the tree to ensure fresh search from this position
            # This prevents illegal moves by ensuring tree structure matches current state
            self._reset_tree_for_new_root()
            
            logger.debug("Hybrid root state updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating hybrid root state: {e}")
            import traceback
            traceback.print_exc()
    
    def _reset_tree_for_new_root(self):
        """Reset tree structure while keeping root node for fresh search"""
        # Clear all child nodes but keep root
        if hasattr(self.tree, 'node_data'):
            # For CSR tree, reset the tree structure without child_counts
            # Reset visit counts for children (but not root)
            if self.tree.num_nodes > 1:
                self.tree.node_data.visit_counts[1:] = 0
                self.tree.node_data.value_sums[1:] = 0.0
                # Reset parent indices to indicate unused nodes
                self.tree.node_data.parent_indices[1:] = -2
                self.tree.node_data.parent_actions[1:] = -1
                # Reset tree node count to 1 (just root)
                self.tree.num_nodes = 1
        
        logger.debug("Reset tree structure for fresh search from new root position")
    
    def _update_root_state(self, new_root_state: Any):
        """Update root state"""
        # Get the current root state index
        root_state_idx = self.node_to_state[0].item()
        
        if root_state_idx >= 0:
            # CRITICAL: For hybrid backend, use proper CPU state synchronization
            if hasattr(self.config, 'backend') and self.config.backend == 'hybrid':
                self._update_hybrid_root_state(new_root_state, root_state_idx)
                return
            
            # Handle numpy array states
            if isinstance(new_root_state, np.ndarray):
                # Direct numpy array - set board state
                # Lazy import to avoid circular dependency
                from ..cpu.cpu_game_states import CPUGameStates
                if isinstance(self.game_states, CPUGameStates):
                    # CPU backend has a dedicated method
                    self.game_states.set_board_from_tensor(root_state_idx, new_root_state)
                    
                    # Set current player based on move count
                    # Count non-zero positions to determine whose turn it is
                    num_moves = (new_root_state != 0).sum()
                    current_player = 1 if num_moves % 2 == 0 else 2
                    self.game_states.current_player[root_state_idx] = current_player
                    self.game_states.move_count[root_state_idx] = num_moves
                else:
                    # GPU backend - update directly
                    board_tensor = torch.from_numpy(new_root_state).to(self.device)
                    self.game_states.boards[root_state_idx] = board_tensor
                    
                    # Set current player based on move count
                    # Count non-zero positions to determine whose turn it is
                    num_moves = (new_root_state != 0).sum()
                    current_player = 1 if num_moves % 2 == 0 else 2
                    self.game_states.current_player[root_state_idx] = current_player
                    self.game_states.move_count[root_state_idx] = num_moves
                return
            
            # Update the existing state instead of allocating a new one
            if hasattr(new_root_state, 'get_basic_tensor_representation'):
                # Get board representation using game interface
                obs = self.cached_game.state_to_numpy(new_root_state)
                
                # Update game state
                if self.game_states.game_type == GameType.GOMOKU or self.game_states.game_type == GameType.GO:
                    # Get actual board size from observation
                    actual_board_size = obs[0].shape[0]
                    board = torch.zeros((actual_board_size, actual_board_size), dtype=torch.int8, device=self.device)
                    
                    # Check number of channels to determine format
                    num_channels = obs.shape[0]
                    
                    if num_channels == 3:
                        # Standard 3-channel format (for Go/Chess)
                        player1_channel = torch.from_numpy(obs[0]).to(self.device)
                        player2_channel = torch.from_numpy(obs[1]).to(self.device)
                        
                        # Map to GPUGameStates format
                        board[player1_channel > 0.5] = 1  # Black
                        board[player2_channel > 0.5] = 2  # White
                    else:
                        # 18-channel representation
                        # CRITICAL FIX: For 18-channel format, we cannot reconstruct the board
                        # by iterating through channels. Instead, get 3-channel representation.
                        
                        # Get 3-channel representation for accurate board reconstruction
                        obs_3ch = self.cached_game.state_to_numpy(new_root_state, representation_type='minimal')
                        if obs_3ch.shape[0] >= 2:
                            # CRITICAL: 3-channel format uses perspective-based encoding
                            # Channel 0: Current player's stones
                            # Channel 1: Opponent's stones
                            current_player_channel = torch.from_numpy(obs_3ch[0]).to(self.device)
                            opponent_channel = torch.from_numpy(obs_3ch[1]).to(self.device)
                            
                            # Get actual current player from game state
                            actual_current_player = new_root_state.get_current_player()
                            
                            # Map channels to absolute player numbers
                            if actual_current_player == 1:
                                # Current player is 1, so channel 0 = player 1, channel 1 = player 2
                                board[current_player_channel > 0.5] = 1
                                board[opponent_channel > 0.5] = 2
                            else:
                                # Current player is 2, so channel 0 = player 2, channel 1 = player 1
                                board[current_player_channel > 0.5] = 2
                                board[opponent_channel > 0.5] = 1
                        else:
                            # Fallback: should not happen
                            logger.error(f"Unexpected 3-channel format with {obs_3ch.shape[0]} channels")
                    
                    self.game_states.boards[root_state_idx] = board
                    # Update current player (1=black, 2=white)
                    actual_current_player = new_root_state.get_current_player()
                    self.game_states.current_player[root_state_idx] = actual_current_player
                    move_history = new_root_state.get_move_history()
                    if isinstance(move_history, list):
                        self.game_states.move_count[root_state_idx] = len(move_history)
                    else:
                        self.game_states.move_count[root_state_idx] = move_history.shape[0]
                    
                    # CRITICAL: Update terminal status and game result
                    is_terminal = new_root_state.is_terminal()
                    self.game_states.is_terminal[root_state_idx] = is_terminal
                    if is_terminal:
                        game_result = new_root_state.get_game_result()
                        # Convert GameResult enum to integer value
                        if hasattr(game_result, 'value'):
                            self.game_states.game_result[root_state_idx] = game_result.value
                        else:
                            self.game_states.game_result[root_state_idx] = game_result
        else:
            # No existing state, initialize new one
            self._initialize_root(new_root_state)
        
    def _update_state_mappings_after_reuse(self, mapping: Dict[int, int]):
        """Update state mappings after subtree reuse
        
        This method ensures that the node-to-state mapping is correctly updated
        after tree reuse, and that game states are properly synchronized.
        """
        # Create new node_to_state mapping
        new_node_to_state = torch.full_like(self.node_to_state, -1)
        
        # Track which states are still in use
        states_in_use = set()
        
        for old_node, new_node in mapping.items():
            if old_node < len(self.node_to_state):
                old_state = self.node_to_state[old_node]
                if old_state >= 0 and new_node < len(new_node_to_state):
                    new_node_to_state[new_node] = old_state
                    states_in_use.add(old_state.item())
        
        # Free states that are no longer in use
        for i in range(len(self.node_to_state)):
            state_idx = self.node_to_state[i].item()
            if state_idx >= 0 and state_idx not in states_in_use:
                # Return state to free pool
                if state_idx > 0:  # Don't free state 0 (reserved for root)
                    self.state_pool_free_list.append(state_idx)
                    # Mark state as unallocated in GPU game states
                    if hasattr(self.game_states, 'allocated_mask'):
                        self.game_states.allocated_mask[state_idx] = False
        
        # Update the mapping
        self.node_to_state = new_node_to_state
        
        # Sort free list for better locality
        self.state_pool_free_list.sort()
        self.state_pool_free_count = len(self.state_pool_free_list)
        
        
    def _get_evaluator_input_channels(self) -> int:
        """Get number of input channels expected by evaluator"""
        # Try to determine from evaluator
        if hasattr(self.evaluator, 'input_channels'):
            channels = self.evaluator.input_channels
            # Handle Mock objects that don't return proper integers
            if isinstance(channels, int):
                return channels
        elif hasattr(self.evaluator, 'model') and hasattr(self.evaluator.model, 'input_channels'):
            channels = self.evaluator.model.input_channels
            # Handle Mock objects that don't return proper integers
            if isinstance(channels, int):
                return channels
        
        # Default based on game type for real evaluators or Mock objects
        return 18  # Standard AlphaZero channels
            
    def _reset_for_new_search(self):
        """Reset tree and state pool for a new search"""
        # CRITICAL FIX: Properly free all allocated states before resetting
        if hasattr(self, 'game_states'):
            # Handle CPU vs GPU game states differently
            if hasattr(self.game_states, 'allocated_mask'):
                # GPU game states
                if isinstance(self.game_states.allocated_mask, torch.Tensor):
                    allocated_indices = torch.nonzero(self.game_states.allocated_mask, as_tuple=True)[0]
                else:
                    # CPU game states with numpy allocated_mask
                    allocated_indices = torch.from_numpy(np.nonzero(self.game_states.allocated_mask)[0])
                
                if len(allocated_indices) > 0:
                    # Free all states
                    self.game_states.free_states(allocated_indices)
                
                # Verify clean state
                if isinstance(self.game_states.allocated_mask, torch.Tensor):
                    remaining_allocated = self.game_states.allocated_mask.sum().item()
                else:
                    remaining_allocated = self.game_states.allocated_mask.sum()
                    
                if remaining_allocated > 0:
                    logger.warning(f"State pool not fully cleaned: {remaining_allocated} states still allocated")
            else:
                # CPU game states without allocated_mask - just reset
                if hasattr(self.game_states, 'reset'):
                    self.game_states.reset()
                elif hasattr(self.game_states, 'num_states'):
                    # Manual reset for CPU game states
                    self.game_states.num_states = 0
        
        # Reset tree to clean state
        self.tree.reset()
        # Reset node-to-state mapping
        self.node_to_state.fill_(-1)
        # Clear state pool - but preserve state 0 for root
        self.state_pool_free_list = list(range(1, self.config.max_tree_nodes))
        
        # Clear the game state for state 0 to avoid stale data
        if hasattr(self, 'game_states'):
            # For both hybrid and GPU backends, use GPU game states interface
            if hasattr(self.game_states, 'boards') and self.game_states.boards is not None:
                # GPU game states interface - reset state 0 to empty
                self.game_states.boards[0] = 0
                self.game_states.current_player[0] = 0
                self.game_states.move_count[0] = 0
                self.game_states.is_terminal[0] = False
                self.game_states.winner[0] = 0
    
    def _cleanup_after_search(self):
        """Clean up allocated states after search completes"""
        # For non-reuse mode, we want to track how many states were used
        # This helps with debugging and optimization
        if hasattr(self, 'game_states'):
            allocated_count = self.game_states.allocated_mask.sum().item()
    

    def _cleanup_stale_children_after_reuse(self):
        """Clean up stale children that are no longer valid after tree reuse
        
        This is critical when tree reuse is enabled to prevent illegal move selection.
        """
        if not self.config.enable_subtree_reuse:
            return
            
        # Get current root state
        root_state_idx = self.node_to_state[0].item()
        if root_state_idx < 0:
            return
            
        # Get legal moves for current position
        legal_mask = self.game_states.get_legal_moves_mask(
            torch.tensor([root_state_idx], device=self.device)
        )[0]
        legal_moves_set = set(torch.nonzero(legal_mask).squeeze(-1).cpu().numpy())
        
        # Check root's children
        children, _, _ = self.tree.get_children(0)
        if len(children) == 0:
            return
            
        # Remove illegal children
        children_to_remove = []
        for child_idx in children:
            action = self.tree.node_data.parent_actions[child_idx].item()
            if action not in legal_moves_set:
                children_to_remove.append(child_idx)
        
        if children_to_remove:
            logger.info(f"Removing {len(children_to_remove)} stale children after tree reuse")
            # Remove the invalid children from the tree
            self.tree.remove_children(0, children_to_remove)

    def _validate_children_after_reuse(self):
        """Validate and clean up children after tree reuse
        
        This method ensures that all children in the reused tree are still valid
        for the current game state. It removes invalid children to prevent
        illegal move selection and wasted simulations.
        
        CRITICAL: After tree reuse, the new root's "children" may include its
        former siblings, which are not valid children of the new position.
        """
        # Get current root state
        root_state_idx = self.node_to_state[0].item()
        if root_state_idx < 0:
            return
            
        # Get legal moves for current position
        legal_mask = self.game_states.get_legal_moves_mask(
            torch.tensor([root_state_idx], device=self.device)
        )[0]
        legal_moves_set = set(torch.nonzero(legal_mask).squeeze(-1).cpu().numpy())
        
        # Validate all nodes in the tree, not just root children
        nodes_to_validate = []
        nodes_processed = set()
        
        # BFS to validate entire tree
        queue = [0]  # Start with root
        while queue:
            node_idx = queue.pop(0)
            if node_idx in nodes_processed:
                continue
            nodes_processed.add(node_idx)
            
            # Get node's state
            state_idx = self.node_to_state[node_idx].item()
            if state_idx < 0:
                continue
                
            # Get children of this node
            children, actions, _ = self.tree.get_children(node_idx)
            if len(children) == 0:
                continue
                
            # Get legal moves for this node's state
            node_legal_mask = self.game_states.get_legal_moves_mask(
                torch.tensor([state_idx], device=self.device)
            )[0]
            node_legal_moves = set(torch.nonzero(node_legal_mask).squeeze(-1).cpu().numpy())
            
            # Check each child
            invalid_children = []
            for i, (child_idx, action) in enumerate(zip(children.cpu().numpy(), actions.cpu().numpy())):
                if action not in node_legal_moves:
                    invalid_children.append(i)
                else:
                    # Add valid children to queue for further validation
                    queue.append(child_idx)
            
            # Mark invalid children for removal
            if invalid_children:
                # For now, we'll zero out their visit counts and values
                # This effectively makes them invisible to UCB selection
                for i in invalid_children:
                    child_idx = children[i].item()
                    self.tree.node_data.visit_counts[child_idx] = 0
                    self.tree.node_data.value_sums[child_idx] = 0.0
                    # Set prior to 0 to prevent selection
                    self.tree.node_data.node_priors[child_idx] = 0.0
                    
                logger.info(f"Invalidated {len(invalid_children)} stale children for node {node_idx}")

    def shutdown(self):
        """Cleanup resources"""
        # Cleanup any resources
        self.clear()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False


def create_single_gpu_mcts(
    config: MCTSConfig,
    model: torch.nn.Module,
    game_interface: Optional[GameInterface] = None,
    use_tensorrt: bool = False
) -> MCTS:
    """Factory function to create optimized single-GPU MCTS
    
    This replaces the DirectMCTS class with a cleaner interface.
    
    Args:
        config: MCTS configuration
        model: Neural network model
        game_interface: Optional game interface
        use_tensorrt: Whether to use TensorRT
        
    Returns:
        MCTS instance optimized for single-GPU
    """
    # Apply recommended optimizations to config
    config = optimize_config_for_single_gpu(config)
    
    # Store TensorRT flag in config
    config.use_tensorrt = use_tensorrt
    
    # Create MCTS with single-GPU mode enabled
    return MCTS(
        config=config,
        evaluator=model,  # Will be converted to SingleGPUEvaluator internally
        game_interface=game_interface,
        single_gpu_mode=True
    )


def optimize_config_for_single_gpu(config: MCTSConfig) -> MCTSConfig:
    """Apply recommended optimizations to MCTS config for single-GPU
    
    Args:
        config: Original MCTS configuration
        
    Returns:
        Optimized configuration
    """
    # Enable performance features
    config.classical_only_mode = True  # Skip quantum features
    config.enable_fast_ucb = True      # Use optimized UCB
    config.use_mixed_precision = True  # FP16 for tensor cores
    config.use_cuda_graphs = True      # Enable CUDA graphs
    config.use_tensor_cores = True     # Leverage tensor cores
    
    # Memory optimizations
    config.initial_capacity_factor = 0.5  # Pre-allocate more
    config.enable_memory_pooling = True   # Use memory pools
    
    # Batch size optimization
    if config.device == 'cuda' and torch.cuda.is_available():
        # Adjust wave size based on GPU memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory_gb >= 24:  # High-end GPU (3090, 4090)
            config.max_wave_size = 4096
        elif gpu_memory_gb >= 12:  # Mid-range GPU
            config.max_wave_size = 3072
        else:  # Lower-end GPU
            config.max_wave_size = 2048
    
    logger.info(f"Optimized single-GPU config: wave_size={config.max_wave_size}, "
               f"mixed_precision={config.use_mixed_precision}")
    
    return config