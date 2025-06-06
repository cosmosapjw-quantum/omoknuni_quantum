"""MCTS Node implementation for the Omoknuni system

This module provides the Node class which represents states in the MCTS tree.
Each node tracks visit statistics, value estimates, and parent-child relationships.
"""

from typing import Dict, Optional, Any
import numpy as np
import threading


class Node:
    """A node in the MCTS tree
    
    Attributes:
        state: The game state at this node
        parent: Parent node (None for root)
        action: Action that led to this node from parent
        prior: Prior probability from policy network
        visit_count: Number of times this node has been visited
        value_sum: Sum of all backup values (for computing average)
        children: Dictionary mapping actions to child nodes
        is_expanded: Whether this node has been expanded
        is_terminal: Whether this is a terminal game state
    """
    
    def __init__(
        self,
        state: Any,
        parent: Optional['Node'],
        action: Optional[int],
        prior: float
    ):
        """Initialize a new node
        
        Args:
            state: Game state at this node
            parent: Parent node (None for root)
            action: Action from parent that led here
            prior: Prior probability of selecting this node
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        
        # Statistics
        self.visit_count = 0
        self.value_sum = 0.0
        
        # Tree structure
        self.children: Dict[int, 'Node'] = {}
        self.is_expanded = False
        self.is_terminal = False
        
        # For quantum-inspired features (to be implemented)
        self.phase = 0.0  # Phase for phase-kicked priors
        self.minhash_signature = None  # For diversity computation
        
        # Thread safety
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        
    def value(self) -> float:
        """Get the average value of this node (thread-safe)
        
        Returns:
            Average value (Q-value) or 0 if no visits
        """
        with self._lock:
            if self.visit_count == 0:
                return 0.0
            return self.value_sum / self.visit_count
    
    def ucb_score(self, c_puct: float) -> float:
        """Calculate UCB score for node selection (thread-safe)
        
        Uses the PUCT formula:
        UCB = Q + c_puct * P * sqrt(parent_visits) / (1 + visits)
        
        Args:
            c_puct: Exploration constant
            
        Returns:
            UCB score for this node
        """
        if self.parent is None:
            return 0.0
            
        # Get visit counts atomically
        with self._lock:
            my_visits = self.visit_count
            my_value = self.value()
        
        # Get parent visit count
        with self.parent._lock:
            parent_visits = self.parent.visit_count
            
        # Exploration term
        exploration = c_puct * self.prior * np.sqrt(parent_visits) / (1 + my_visits)
        
        return my_value + exploration
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (thread-safe)
        
        Returns:
            True if node has no children or is terminal
        """
        with self._lock:
            return len(self.children) == 0 or self.is_terminal
    
    def expand(self, action_probs: Dict[int, float], child_states: Dict[int, Any]) -> None:
        """Expand this node by creating children (thread-safe)
        
        Args:
            action_probs: Dictionary mapping actions to prior probabilities
            child_states: Dictionary mapping actions to resulting states
            
        Raises:
            ValueError: If node is terminal or already expanded
        """
        with self._lock:
            if self.is_terminal:
                raise ValueError("Cannot expand terminal node")
                
            if self.is_expanded:
                # Already expanded, nothing to do (avoid race condition)
                return
                
            # Create child nodes
            # Note: We use a regular dict here, but the order doesn't matter
            # because select_child() handles tie-breaking randomly
            new_children = {}
            for action, prob in action_probs.items():
                child_state = child_states.get(action)
                child = Node(
                    state=child_state,
                    parent=self,
                    action=action,
                    prior=prob
                )
                new_children[action] = child
            
            # Update children atomically
            self.children.update(new_children)
            self.is_expanded = True
    
    def backup(self, value: float) -> None:
        """Backup value through the tree (thread-safe)
        
        Updates visit counts and value sums for this node and all ancestors.
        Values are negated at each level for adversarial games.
        
        Args:
            value: The value to backup (from the perspective of the player at this node)
        """
        node = self
        
        while node is not None:
            with node._lock:
                node.visit_count += 1
                node.value_sum += value
            
            # Negate value for parent (adversarial game)
            value = -value
            node = node.parent
    
    def select_child(self, c_puct: float) -> 'Node':
        """Select best child based on UCB scores (thread-safe)
        
        Args:
            c_puct: Exploration constant
            
        Returns:
            Child node with highest UCB score
            
        Raises:
            ValueError: If node has no children
        """
        with self._lock:
            if not self.children:
                raise ValueError("Cannot select from node with no children")
                
            # Create a snapshot of children to avoid modification during iteration
            children_snapshot = list(self.children.items())
            
        # Calculate UCB scores for all children (outside lock to reduce contention)
        children_scores = []
        for action, child in children_snapshot:
            score = child.ucb_score(c_puct)
            children_scores.append((score, action, child))
            
        # Find maximum score
        max_score = max(score for score, _, _ in children_scores)
        
        # Get all children with maximum score (for tie-breaking)
        best_children = [(action, child) for score, action, child in children_scores if score == max_score]
        
        # If multiple children have same score, choose randomly
        if len(best_children) > 1:
            # Random tie-breaking to avoid bias
            idx = np.random.randint(len(best_children))
            _, best_child = best_children[idx]
        else:
            _, best_child = best_children[0]
                
        return best_child
    
    def get_improved_policy(self, temperature: float = 1.0) -> Dict[int, float]:
        """Get improved policy based on visit counts (thread-safe)
        
        Args:
            temperature: Temperature for controlling exploration
                        0 = deterministic (argmax)
                        1 = proportional to visit counts
                        >1 = more exploration
                        
        Returns:
            Dictionary mapping actions to probabilities
        """
        with self._lock:
            if not self.children:
                return {}
                
            # Create snapshot of children and their visit counts
            children_data = [(a, c.visit_count) for a, c in self.children.items()]
            
        if temperature == 0:
            # Deterministic: choose most visited
            visits = {a: count for a, count in children_data}
            best_action = max(visits, key=visits.get)
            return {a: 1.0 if a == best_action else 0.0 for a in visits}
        
        # Proportional to visit count ^ (1/temperature)
        actions = [a for a, _ in children_data]
        visits = np.array([count for _, count in children_data])
        
        if temperature != 1.0:
            visits = np.power(visits, 1.0 / temperature)
            
        # Normalize to probabilities
        if visits.sum() > 0:
            probs = visits / visits.sum()
        else:
            probs = np.ones(len(visits)) / len(visits)
            
        return dict(zip(actions, probs))
    
    def get_action_values(self) -> Dict[int, float]:
        """Get Q-values for all child actions (thread-safe)
        
        Returns:
            Dictionary mapping actions to Q-values
        """
        with self._lock:
            # Create snapshot of children
            children_snapshot = list(self.children.items())
            
        # Get values outside lock to reduce contention
        return {
            action: child.value() 
            for action, child in children_snapshot
        }
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return (
            f"Node(action={self.action}, "
            f"visits={self.visit_count}, "
            f"value={self.value():.3f}, "
            f"prior={self.prior:.3f}, "
            f"children={len(self.children)})"
        )