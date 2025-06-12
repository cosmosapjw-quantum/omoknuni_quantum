"""Simplified unified MCTS for better performance

This implementation focuses on the core algorithm without over-engineering.
"""

import torch
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass 
class SimpleMCTSConfig:
    """Simple MCTS configuration"""
    num_simulations: int = 10000
    c_puct: float = 1.414
    temperature: float = 1.0
    device: str = 'cuda'
    board_size: int = 15


class SimpleMCTS:
    """Simplified MCTS with better performance"""
    
    def __init__(self, config: SimpleMCTSConfig, evaluator):
        self.config = config
        self.device = torch.device(config.device)
        self.evaluator = evaluator
        
        # Simple tree storage
        self.reset()
        
    def reset(self):
        """Reset the tree"""
        # Use simple Python dicts for tree - much faster than complex tensor operations
        self.children = {}  # node -> list of children
        self.edges = {}     # (parent, action) -> child
        self.visits = {}    # node -> visit count
        self.values = {}    # node -> total value
        self.priors = {}    # node -> prior
        self.states = {}    # node -> game state
        
        self.node_count = 0
        
    def search(self, root_state: Any, num_simulations: int) -> np.ndarray:
        """Run MCTS search"""
        start_time = time.time()
        
        # Initialize root
        if 0 not in self.states:
            self.states[0] = root_state
            self.visits[0] = 0
            self.values[0] = 0
            self.priors[0] = 1.0
            self.children[0] = []
            
        # Run simulations
        for _ in range(num_simulations):
            self._simulate(0)
            
        # Extract policy
        policy = self._get_policy(0)
        
        elapsed = time.time() - start_time
        logger.info(f"Search complete: {num_simulations} sims in {elapsed:.2f}s "
                   f"({num_simulations/elapsed:.0f} sims/s)")
        
        return policy
        
    def _simulate(self, node: int):
        """Run one simulation from node"""
        path = []
        
        # Selection phase - traverse tree
        while node in self.children and len(self.children[node]) > 0:
            path.append(node)
            action, child = self._select_child(node)
            node = child
            
        path.append(node)
        
        # Expansion phase
        if node not in self.children:
            self._expand(node)
            
        # Evaluation phase
        value = self._evaluate(node)
        
        # Backup phase
        for n in reversed(path):
            self.visits[n] = self.visits.get(n, 0) + 1
            self.values[n] = self.values.get(n, 0) + value
            value = -value  # Flip for opponent
            
    def _select_child(self, node: int) -> Tuple[int, int]:
        """Select best child using UCB"""
        children = self.children[node]
        parent_visits = self.visits.get(node, 1)
        
        best_ucb = -float('inf')
        best_action = -1
        best_child = -1
        
        for action, child in children:
            child_visits = self.visits.get(child, 0)
            child_value = self.values.get(child, 0)
            prior = self.priors.get(child, 1.0)
            
            if child_visits > 0:
                q = child_value / child_visits
            else:
                q = 0
                
            ucb = q + self.config.c_puct * prior * np.sqrt(parent_visits) / (1 + child_visits)
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action
                best_child = child
                
        return best_action, best_child
        
    def _expand(self, node: int):
        """Expand a node"""
        state = self.states[node]
        
        # Get legal actions from game state
        if hasattr(state, 'get_legal_moves'):
            legal_actions = state.get_legal_moves()
        else:
            # Assume all squares are legal for Gomoku
            legal_actions = [i for i in range(self.config.board_size ** 2) 
                           if self._is_empty(state, i)]
        
        if not legal_actions:
            self.children[node] = []
            return
            
        # Create child nodes
        children = []
        for action in legal_actions:
            self.node_count += 1
            child = self.node_count
            
            # Clone state and apply action
            child_state = self._clone_state(state)
            child_state.make_move(action)
            
            self.states[child] = child_state
            self.visits[child] = 0
            self.values[child] = 0
            self.priors[child] = 1.0 / len(legal_actions)  # Uniform prior
            self.edges[(node, action)] = child
            
            children.append((action, child))
            
        self.children[node] = children
        
    def _is_empty(self, state, pos):
        """Check if position is empty"""
        if hasattr(state, 'board'):
            row = pos // self.config.board_size
            col = pos % self.config.board_size
            return state.board[row][col] == 0
        return True
        
    def _clone_state(self, state):
        """Clone game state"""
        if hasattr(state, 'clone'):
            return state.clone()
        # For alphazero_py states
        import copy
        return copy.deepcopy(state)
        
    def _evaluate(self, node: int) -> float:
        """Evaluate a node"""
        state = self.states[node]
        
        # Check terminal
        if hasattr(state, 'is_game_over') and state.is_game_over():
            if hasattr(state, 'game_result'):
                result = state.game_result()
                # Map result to value from current player's perspective
                return -1.0 if result == 1 else 1.0 if result == -1 else 0.0
                
        # Neural network evaluation
        features = self._get_features(state)
        features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            result = self.evaluator.evaluate_batch(features_tensor)
            if len(result) == 3:
                values, policies, _ = result
            else:
                policies, values = result
                
        return values[0, 0].item()
        
    def _get_features(self, state) -> np.ndarray:
        """Get features from state"""
        if hasattr(state, 'to_numpy'):
            return state.to_numpy()
            
        # Simple features for Gomoku
        features = np.zeros((4, self.config.board_size, self.config.board_size))
        
        if hasattr(state, 'board'):
            board = np.array(state.board)
            current_player = state.current_player() if hasattr(state, 'current_player') else 1
            
            # Current player stones
            features[0] = (board == current_player)
            # Opponent stones  
            features[1] = (board == 3 - current_player)
            # Empty squares
            features[2] = (board == 0)
            # Constant plane
            features[3] = 1.0
            
        return features
        
    def _get_policy(self, node: int) -> np.ndarray:
        """Extract policy from node visits"""
        if node not in self.children or not self.children[node]:
            # No children - uniform policy
            return np.ones(self.config.board_size ** 2) / (self.config.board_size ** 2)
            
        # Create full policy array
        policy = np.zeros(self.config.board_size ** 2)
        
        # Get visits for each action
        for action, child in self.children[node]:
            visits = self.visits.get(child, 0)
            policy[action] = visits
            
        # Normalize
        total = policy.sum()
        if total > 0:
            policy = policy / total
        else:
            # Uniform if no visits
            policy = np.ones_like(policy) / len(policy)
            
        return policy