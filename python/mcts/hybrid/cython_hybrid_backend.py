"""Cython Hybrid Backend - Uses CythonTree for CPU operations with GPU evaluation"""

import logging
import numpy as np
import torch
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class CythonHybridBackend:
    """Hybrid backend using CythonTree for CPU operations and GPU for evaluation
    
    This backend properly implements subtree reuse to maintain tree structure
    across moves, preventing the tree reset issue that causes weak play.
    """
    
    def __init__(self, tree, game_states, evaluator, config):
        """Initialize Cython hybrid backend
        
        Args:
            tree: CythonTree instance
            game_states: GPU game states
            evaluator: Neural network evaluator
            config: MCTS configuration
        """
        self.tree = tree
        self.game_states = game_states
        self.evaluator = evaluator
        self.config = config
        
        # Use GPU batch size from config
        self.gpu_batch_size = getattr(config, 'hybrid_gpu_batch_size', 
                                      getattr(config, 'batch_size', 256))
        
        # Device
        self.device = getattr(config, 'device', 'cuda')
        
        # Allocate GPU buffers for batch processing
        if self.device == 'cuda':
            self._allocate_gpu_buffers()
        
        logger.info("CythonHybridBackend initialized")
        logger.info(f"  GPU batch size: {self.gpu_batch_size}")
        logger.info(f"  Device: {self.device}")
    
    def _allocate_gpu_buffers(self):
        """Pre-allocate GPU buffers for batch processing"""
        try:
            # Pre-allocate buffers for batch evaluation
            self.gpu_state_buffer = torch.zeros(
                (self.gpu_batch_size, 19, 15, 15),  # AlphaZero representation
                dtype=torch.float32,
                device=self.device
            )
            self.gpu_policy_buffer = torch.zeros(
                (self.gpu_batch_size, 225),  # 15x15 board
                dtype=torch.float32,
                device=self.device
            )
            self.gpu_value_buffer = torch.zeros(
                (self.gpu_batch_size,),
                dtype=torch.float32,
                device=self.device
            )
            logger.info(f"  Allocated GPU buffers for batch size {self.gpu_batch_size}")
        except Exception as e:
            logger.warning(f"Failed to allocate GPU buffers: {e}")
            # Fall back to CPU if GPU allocation fails
            self.device = 'cpu'
    
    def run_simulations(self, root_state: Any, num_simulations: int, 
                       dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25) -> np.ndarray:
        """Run MCTS simulations
        
        Args:
            root_state: Root game state
            num_simulations: Number of simulations to run
            dirichlet_alpha: Dirichlet noise alpha
            dirichlet_epsilon: Dirichlet noise epsilon
            
        Returns:
            Policy distribution over actions
        """
        # This is called by wave_search in MCTS
        # For now, we'll delegate to the tree's search method if it has one
        if hasattr(self.tree, 'search'):
            # Use tree's built-in search if available
            return self.tree.search(root_state, num_simulations)
        else:
            # Implement basic MCTS loop
            for _ in range(num_simulations):
                self._run_single_simulation(root_state)
            
            # Get policy from root
            return self._get_policy_from_root()
    
    def _run_single_simulation(self, root_state):
        """Run a single MCTS simulation"""
        # Select path from root to leaf
        path = self._select_path()
        
        # Expand leaf if needed
        if self._should_expand(path[-1]):
            self._expand_node(path[-1])
        
        # Evaluate leaf
        value = self._evaluate_leaf(path[-1])
        
        # Backup value through path
        self._backup_path(path, value)
    
    def _select_path(self) -> List[int]:
        """Select path from root to leaf using UCB"""
        path = [0]  # Start at root
        current = 0
        
        while self.tree.is_expanded(current):
            children = self.tree.get_children(current)
            if not children:
                break
            
            # Select best child using UCB
            best_child = self._select_best_child(current, children)
            path.append(best_child)
            current = best_child
        
        return path
    
    def _select_best_child(self, parent: int, children: List[int]) -> int:
        """Select best child using UCB formula"""
        # Simplified UCB selection
        best_score = -float('inf')
        best_child = children[0]
        
        parent_visits = self.tree.get_visit_count(parent)
        
        for child in children:
            visits = self.tree.get_visit_count(child)
            if visits == 0:
                return child  # Explore unvisited child first
            
            value = self.tree.get_value(child)
            exploration = self.config.c_puct * np.sqrt(parent_visits) / (1 + visits)
            score = value + exploration
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def _should_expand(self, node: int) -> bool:
        """Check if node should be expanded"""
        return not self.tree.is_expanded(node) and self.tree.get_visit_count(node) > 0
    
    def _expand_node(self, node: int):
        """Expand a node by adding children"""
        # Get legal actions for the node's state
        state = self.game_states.get_state(node)
        legal_actions = self.game_states.get_legal_actions(state)
        
        # Add children for legal actions
        for action in legal_actions:
            self.tree.add_child(node, action)
    
    def _evaluate_leaf(self, node: int) -> float:
        """Evaluate a leaf node using neural network"""
        state = self.game_states.get_state(node)
        
        # Convert state to tensor
        state_tensor = self.game_states.state_to_tensor(state)
        
        # Evaluate with neural network
        with torch.no_grad():
            if hasattr(self.evaluator, 'evaluate_single'):
                _, value = self.evaluator.evaluate_single(state_tensor)
            else:
                # Batch evaluation for single state
                policy, value = self.evaluator.evaluate([state_tensor])
                value = value[0]
        
        return float(value)
    
    def _backup_path(self, path: List[int], value: float):
        """Backup value through path"""
        for node in reversed(path):
            self.tree.update_node(node, value)
            # Flip value for opponent
            value = -value
    
    def _get_policy_from_root(self) -> np.ndarray:
        """Get policy distribution from root node"""
        policy = np.zeros(225)  # 15x15 board
        
        children = self.tree.get_children(0)  # Root is node 0
        if not children:
            # Uniform policy if no children
            return np.ones(225) / 225
        
        # Get visit counts for children
        total_visits = 0
        for child in children:
            action = self.tree.get_action(child)
            visits = self.tree.get_visit_count(child)
            policy[action] = visits
            total_visits += visits
        
        # Normalize
        if total_visits > 0:
            policy /= total_visits
        else:
            policy = np.ones(225) / 225
        
        return policy
    
    def apply_subtree_reuse(self, action: int) -> Optional[Dict[int, int]]:
        """Apply subtree reuse when moving to a child node
        
        This is critical for maintaining tree quality across moves.
        
        Args:
            action: Action taken to move to child
            
        Returns:
            Mapping of old node indices to new ones, or None if failed
        """
        try:
            # Find child node for this action
            children = self.tree.get_children(0)  # Root children
            child_node = None
            
            for child in children:
                if self.tree.get_action(child) == action:
                    child_node = child
                    break
            
            if child_node is None:
                # Child doesn't exist, can't reuse
                return None
            
            # Shift tree to make child the new root
            if hasattr(self.tree, 'shift_root'):
                mapping = self.tree.shift_root(child_node)
                return mapping
            else:
                # Tree doesn't support shift_root, return None to trigger reset
                return None
                
        except Exception as e:
            logger.debug(f"Subtree reuse failed in CythonHybridBackend: {e}")
            return None