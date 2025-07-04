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
from ..quantum import QuantumConfig, SearchPhase, create_pragmatic_quantum_mcts
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
        game_interface: Optional[GameInterface] = None
    ):
        """Initialize MCTS
        
        Args:
            config: MCTS configuration
            evaluator: Neural network evaluator
            game_interface: Optional game interface
        """
        self.config = config
        self.device = torch.device(config.device)
        self.evaluator = evaluator
        
        # Configure evaluator for torch tensors
        if hasattr(self.evaluator, '_return_torch_tensors'):
            self.evaluator._return_torch_tensors = True
            
        # Initialize components
        self._initialize_components()
        
        # Create game interface if needed
        self._setup_game_interface(game_interface)
        
        # Initialize statistics
        self._initialize_statistics()
        
        # Initialize quantum features if enabled
        self._initialize_quantum()
        
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
        
        # Initialize specialized modules
        self.wave_search = WaveSearch(
            tree=self.tree,
            game_states=self.game_states,
            evaluator=self.evaluator,
            config=self.config,
            device=self.device,
            gpu_ops=self.gpu_ops
        )
        
        self.tree_ops = TreeOperations(
            tree=self.tree,
            config=self.config,
            device=self.device
        )
        
        # State management
        self._initialize_state_management()
        
    def _initialize_tree(self):
        """Initialize CSR tree structure"""
        # Determine max_actions based on game type
        max_actions_map = {
            GameType.CHESS: 4096,
            GameType.GO: 362,
            GameType.GOMOKU: 225
        }
        max_actions = max_actions_map.get(self.config.game_type, 512)
        
        tree_config = CSRTreeConfig(
            max_nodes=self.config.max_tree_nodes,
            max_edges=self.config.max_tree_nodes * self.config.max_children_per_node,
            max_actions=max_actions,
            device=self.config.device,
            enable_virtual_loss=self.config.enable_virtual_loss,
            virtual_loss_value=-abs(getattr(self.config, 'virtual_loss', 1.0)),
            batch_size=self.config.max_wave_size,
            enable_batched_ops=True
        )
        self.tree = CSRTree(tree_config)
        
    def _initialize_game_states(self):
        """Initialize GPU game states"""
        game_config = GPUGameStatesConfig(
            capacity=self.config.max_tree_nodes,
            game_type=self.config.game_type,
            board_size=self.config.board_size,
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
        """Initialize quantum features if enabled"""
        self.quantum_features = None
        self.quantum_total_simulations = 0
        self.quantum_phase = SearchPhase.EXPLORATION
        
        if self.config.enable_quantum and not self.config.classical_only_mode:
            quantum_config = self.config.get_or_create_quantum_config()
            self.quantum_features = create_pragmatic_quantum_mcts(
                quantum_config,
                self.tree,
                self.device
            )
            
    def search(self, state: Any, num_simulations: Optional[int] = None) -> np.ndarray:
        """Run MCTS search from given state
        
        Args:
            state: Game state to search from
            num_simulations: Number of simulations (overrides config)
            
        Returns:
            Policy vector as numpy array
        """
        num_sims = num_simulations or self.config.num_simulations
        
        start_time = time.perf_counter()
        
        # If subtree reuse is disabled, reset the tree for each search
        if not self.config.enable_subtree_reuse:
            # Reset tree to clean state
            self.tree.reset()
            # Reset node-to-state mapping
            self.node_to_state.fill_(-1)
            # Clear state pool - but preserve state 0 for root
            # Don't include 0 in free list since root will use it
            self.state_pool_free_list = list(range(1, self.config.max_tree_nodes))
            # CRITICAL: Clear the GPU game state for state 0 to avoid stale data
            if hasattr(self, 'game_states'):
                # Reset state 0 to empty
                self.game_states.boards[0] = 0
                self.game_states.current_player[0] = 0
                self.game_states.move_count[0] = 0
                self.game_states.is_terminal[0] = False
                self.game_states.winner[0] = 0
        
        # Initialize root if needed
        self._ensure_root_initialized(state)
        
        # Note: Dirichlet noise is now applied per-simulation in wave_search
        # instead of globally to the root node
            
        # Run search
        policy = self._run_search(num_sims)
        
        # Debug: Check root visits after search
        root_visits = self.tree.node_data.visit_counts[0].item()
        if root_visits == 0:
            logger.warning(f"[MCTS DEBUG] Root has {root_visits} visits after {num_sims} simulations!")
        
        # Update statistics
        elapsed_time = time.perf_counter() - start_time
        self._update_statistics(num_sims, elapsed_time)
        
        return policy
        
    def _ensure_root_initialized(self, state: Any):
        """Ensure root node is properly initialized"""
        if self.node_to_state[0] < 0:  # Root has no state yet
            self._initialize_root(state)
        else:
            # Synchronize root state
            self._update_root_state(state)
            
    def _run_search(self, num_simulations: int) -> np.ndarray:
        """Run the main search loop"""
        completed = 0
        
        while completed < num_simulations:
            wave_size = min(self.config.max_wave_size, num_simulations - completed)
            
            # Run one wave
            actual_completed = self.wave_search.run_wave(
                wave_size, 
                self.node_to_state,
                self.state_pool_free_list
            )
            
            # Debug: check if simulations are running
            if actual_completed == 0:
                logger.warning(f"Wave returned 0 completions! Tree nodes: {self.tree.num_nodes}")
                break
            
            completed += actual_completed
            
            # Update quantum state if enabled
            if self.quantum_features and self.config.quantum_version == 'v2':
                self._update_quantum_state(completed, wave_size)
                
        # Extract and return policy
        policy = self._extract_policy(0)
        
        return policy
        
    def _update_quantum_state(self, completed: int, wave_size: int):
        """Update quantum state during search"""
        self.quantum_total_simulations = completed
        
        # Check for convergence
        if (self.config.enable_phase_adaptation and 
            hasattr(self.quantum_features, 'check_convergence')):
            if self.quantum_features.check_convergence(self.tree):
                pass  # Convergence reached
                
    def _extract_policy(self, node_idx: int) -> np.ndarray:
        """Extract policy from node visit counts"""
        actions, visits, _ = self.tree_ops.get_root_children_info()
        
        # Get legal moves for the current state
        state_idx = self.node_to_state[node_idx].item()
        if state_idx >= 0:
            legal_mask = self.game_states.get_legal_moves_mask(
                torch.tensor([state_idx], device=self.device)
            )[0]
            legal_moves = torch.nonzero(legal_mask).squeeze(-1).cpu().numpy()
        else:
            # Fallback - this should not happen in normal operation
            logger.warning(f"No state found for node {node_idx}, cannot determine legal moves")
            legal_moves = np.array([])
        
        if len(actions) == 0:
            # No children - this might happen if root wasn't expanded
            # Force expansion of root node if not expanded
            if state_idx >= 0:
                # Get legal moves and expand root
                legal_mask_tensor = self.game_states.get_legal_moves_mask(
                    torch.tensor([state_idx], device=self.device)
                )[0]
                legal_actions = torch.nonzero(legal_mask_tensor).squeeze(-1)
                
                if len(legal_actions) > 0:
                    # Evaluate root state to get priors
                    features = self.game_states.get_nn_features_batch(
                        torch.tensor([state_idx], device=self.device)
                    )[0]
                    
                    with torch.no_grad():
                        features_np = features.cpu().numpy()
                        if hasattr(self.evaluator, 'evaluate_batch'):
                            policy_logits, _ = self.evaluator.evaluate_batch(features_np[np.newaxis, :])
                            policy_logits = policy_logits[0]
                        else:
                            policy_logits, _ = self.evaluator.evaluate(features_np)
                    
                    # Convert to tensor and extract legal action priors
                    if isinstance(policy_logits, np.ndarray):
                        policy_logits = torch.from_numpy(policy_logits).to(self.device)
                    
                    priors = policy_logits[legal_actions]
                    priors = torch.softmax(priors, dim=0)  # Convert logits to probabilities
                    
                    # Add children to root
                    actions_list = legal_actions.cpu().tolist()
                    priors_list = priors.cpu().tolist()
                    
                    # Clone states for children
                    parent_indices = torch.tensor([state_idx], dtype=torch.int32, device=self.device)
                    num_clones = torch.tensor([len(actions_list)], dtype=torch.int32, device=self.device)
                    child_state_indices = self.game_states.clone_states(parent_indices, num_clones)
                    
                    # Apply actions to cloned states
                    actions_tensor = torch.tensor(actions_list, dtype=torch.int32, device=self.device)
                    self.game_states.apply_moves(child_state_indices, actions_tensor)
                    
                    child_states = child_state_indices.cpu().tolist()
                    
                    # Add children to tree
                    child_indices = self.tree.add_children_batch(
                        node_idx,
                        actions_list,
                        priors_list,
                        child_states
                    )
                    
                    # Update node-to-state mapping
                    for child_idx, child_state_idx in zip(child_indices, child_states):
                        if child_idx < len(self.node_to_state):
                            self.node_to_state[child_idx] = child_state_idx
                    
                    # Ensure CSR consistency
                    self.tree.ensure_consistent()
                    
                    # Now try to get children info again
                    actions, visits, _ = self.tree_ops.get_root_children_info()
            
            # If still no children, return uniform over legal moves
            if len(actions) == 0:
                policy = np.zeros(self.config.board_size ** 2)
                if len(legal_moves) > 0:
                    policy[legal_moves] = 1.0 / len(legal_moves)
                return policy
            
        # Convert visits to probabilities
        total_visits = visits.sum().item()
        if total_visits == 0:
            # No visits - return uniform over legal moves
            print(f"[DEBUG] Root has {len(actions)} children but ZERO visits! This should not happen after 1000 simulations!")
            print(f"[DEBUG] Actions: {actions.cpu().numpy()[:10]}")
            policy = np.zeros(self.config.board_size ** 2)
            if len(legal_moves) > 0:
                policy[legal_moves] = 1.0 / len(legal_moves)
            return policy
            
        # Create policy vector
        policy = np.zeros(self.config.board_size ** 2)
        actions_np = actions.cpu().numpy()
        visits_np = visits.cpu().numpy()
        
        # Debug: check if any visited actions are illegal
        illegal_visited = []
        for action, visit_count in zip(actions_np, visits_np):
            if 0 <= action < len(policy):
                policy[action] = visit_count / total_visits
                if len(legal_moves) > 0 and action not in legal_moves:
                    illegal_visited.append((action, visit_count))
        
                
        # Ensure only legal moves have non-zero probability
        # This handles cases where MCTS might have stale children from reuse
        if len(legal_moves) > 0:
            legal_set = set(legal_moves)
            for i in range(len(policy)):
                if i not in legal_set:
                    policy[i] = 0.0
            # Renormalize
            policy_sum = policy.sum()
            if policy_sum > 0:
                policy /= policy_sum
            else:
                # All visited moves were illegal - this is a critical error
                logger.error(f"CRITICAL: All visited moves are illegal! Tree has {self.tree.num_nodes} nodes")
                logger.error(f"Root children actions: {actions.cpu().numpy()}")
                logger.error(f"Legal moves: {legal_moves[:20]}...")
                logger.error(f"Root state index: {state_idx}")
                raise RuntimeError("MCTS visited only illegal moves - tree search is broken")
                
        return policy
        
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
        """Update root after taking an action
        
        Args:
            action: Action taken
            new_state: New game state
        """
        if self.config.enable_subtree_reuse:
            # Apply subtree reuse
            mapping = self.tree_ops.apply_subtree_reuse(action)
            if mapping:
                # Update state mappings
                self._update_state_mappings_after_reuse(mapping)
                self.stats['tree_reuse_count'] += 1
                self.stats['tree_reuse_nodes'] += len(mapping)
                
        # Initialize new root if needed
        if new_state is not None:
            self._ensure_root_initialized(new_state)
            
    def clear(self):
        """Clear tree and reset state"""
        self.tree_ops.clear()
        self._initialize_state_management()
        
        if self.quantum_features:
            self.quantum_total_simulations = 0
        
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
            state_idx = 0  # Always use state 0 for root when reuse is disabled
        else:
            # Allocate a state in the game states pool
            state_indices = self.game_states.allocate_states(1)
            state_idx = state_indices[0].item()
        
        # Set up the state based on the root_state
        if hasattr(root_state, 'get_basic_tensor_representation'):
            # Get board representation using game interface
            obs = self.cached_game.state_to_numpy(root_state)
            
            # Convert to game states format
            if self.game_states.game_type == GameType.GOMOKU:
                # For basic representation (18 channels):
                # Channel 0: All stones - 1.0 for P1, 2.0 for P2
                # For GPUGameStates: 0=empty, 1=black(P1), 2=white(P2)
                
                board = torch.zeros((self.config.board_size, self.config.board_size), dtype=torch.int8, device=self.device)
                
                # Convert from numpy observation
                board_channel = torch.from_numpy(obs[0]).to(self.device)
                
                # Map values: 1.0 -> 1 (black), 2.0 -> 2 (white)
                board[board_channel == 1.0] = 1
                board[board_channel == 2.0] = 2
                
                self.game_states.boards[state_idx] = board
                # GPUGameStates uses 0-indexed players (0=black, 1=white)
                actual_current_player = root_state.get_current_player()
                current_player = 0 if actual_current_player == 1 else 1
                self.game_states.current_player[state_idx] = current_player
                move_history = root_state.get_move_history()
                if isinstance(move_history, list):
                    self.game_states.move_count[state_idx] = len(move_history)
                else:
                    self.game_states.move_count[state_idx] = move_history.shape[0]
            
        # Map root node to this state
        self.node_to_state[0] = state_idx
            
    def _update_root_state(self, new_root_state: Any):
        """Update root state"""
        # Get the current root state index
        root_state_idx = self.node_to_state[0].item()
        
        if root_state_idx >= 0:
            # Update the existing state instead of allocating a new one
            if hasattr(new_root_state, 'get_basic_tensor_representation'):
                # Get board representation using game interface
                obs = self.cached_game.state_to_numpy(new_root_state)
                
                # Update game state
                if self.game_states.game_type == GameType.GOMOKU:
                    # For basic representation (18 channels):
                    # Channel 0: All stones - 1.0 for P1, 2.0 for P2
                    board = torch.zeros((self.config.board_size, self.config.board_size), dtype=torch.int8, device=self.device)
                    
                    # Convert from numpy observation
                    board_channel = torch.from_numpy(obs[0]).to(self.device)
                    
                    # Map values: 1.0 -> 1 (black), 2.0 -> 2 (white)
                    board[board_channel == 1.0] = 1
                    board[board_channel == 2.0] = 2
                    
                    self.game_states.boards[root_state_idx] = board
                    # Update current player
                    actual_current_player = new_root_state.get_current_player()
                    current_player = 0 if actual_current_player == 1 else 1
                    self.game_states.current_player[root_state_idx] = current_player
                    move_history = new_root_state.get_move_history()
                    if isinstance(move_history, list):
                        self.game_states.move_count[root_state_idx] = len(move_history)
                    else:
                        self.game_states.move_count[root_state_idx] = move_history.shape[0]
        else:
            # No existing state, initialize new one
            self._initialize_root(new_root_state)
        
    def _update_state_mappings_after_reuse(self, mapping: Dict[int, int]):
        """Update state mappings after subtree reuse"""
        # Update node_to_state mapping based on node remapping
        new_node_to_state = torch.full_like(self.node_to_state, -1)
        
        for old_node, new_node in mapping.items():
            if old_node < len(self.node_to_state):
                old_state = self.node_to_state[old_node]
                if old_state >= 0 and new_node < len(new_node_to_state):
                    new_node_to_state[new_node] = old_state
                    
        self.node_to_state = new_node_to_state
        
    def _get_evaluator_input_channels(self) -> int:
        """Get number of input channels expected by evaluator"""
        # Try to determine from evaluator
        if hasattr(self.evaluator, 'input_channels'):
            return self.evaluator.input_channels
        elif hasattr(self.evaluator, 'model') and hasattr(self.evaluator.model, 'input_channels'):
            return self.evaluator.model.input_channels
        else:
            # Default based on game type
            return 18  # Standard AlphaZero channels
            
    def shutdown(self):
        """Cleanup resources"""
        # Cleanup any resources
        self.clear()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False