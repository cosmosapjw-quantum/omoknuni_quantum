"""WaveEngine for vectorized MCTS processing

This module implements the wave-based parallel MCTS algorithm that processes
multiple paths simultaneously for high throughput on GPUs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import numpy as np
import time
from collections import defaultdict
import logging

from .node import Node
from .tree_arena import TreeArena
from .game_interface import GameInterface
from .evaluator import Evaluator

try:
    from mcts.quantum.interference import InterferenceEngine
    HAS_INTERFERENCE = True
except ImportError:
    InterferenceEngine = None
    HAS_INTERFERENCE = False

# Import GPU acceleration if available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


logger = logging.getLogger(__name__)


@dataclass
class WaveConfig:
    """Configuration for wave processing
    
    Attributes:
        min_wave_size: Minimum number of paths in a wave
        max_wave_size: Maximum number of paths in a wave
        initial_wave_size: Starting wave size
        c_puct: PUCT exploration constant
        enable_interference: Whether to use MinHash interference
        interference_threshold: Similarity threshold for interference
        enable_adaptive_sizing: Whether to adapt wave size based on GPU utilization
        max_concurrent_waves: Maximum concurrent waves for pipelining
        enable_phase_kicks: Whether to use phase-kicked priors
    """
    min_wave_size: int = 256
    max_wave_size: int = 2048
    initial_wave_size: int = 512
    c_puct: float = 1.0
    enable_interference: bool = True
    interference_threshold: float = 0.5
    enable_adaptive_sizing: bool = True
    max_concurrent_waves: int = 1
    enable_phase_kicks: bool = False
    

@dataclass
class Wave:
    """A wave of MCTS paths being processed together
    
    Attributes:
        size: Number of paths in the wave
        paths: Current node IDs for each path
        leaf_nodes: Leaf nodes reached by each path
        values: Evaluation values for backup
        phase: Current processing phase (0=selection, 1=expansion, 2=evaluation, 3=backup)
        root_id: Root node ID for this wave
        completed: Whether wave processing is complete
    """
    size: int
    paths: List[str] = field(default_factory=list)
    leaf_nodes: List[Node] = field(default_factory=list)
    values: Optional[np.ndarray] = None
    phase: int = 0
    root_id: Optional[str] = None
    completed: bool = False
    active_paths: int = 0
    
    def __post_init__(self):
        if not self.paths:
            self.paths = [None] * self.size
        if not self.leaf_nodes:
            self.leaf_nodes = [None] * self.size
        self.active_paths = self.size
        

class WaveEngine:
    """Engine for wave-based vectorized MCTS processing"""
    
    def __init__(
        self,
        game: GameInterface,
        evaluator: Evaluator,
        arena: TreeArena,
        config: WaveConfig
    ):
        """Initialize WaveEngine
        
        Args:
            game: Game interface
            evaluator: Neural network evaluator
            arena: Tree memory arena
            config: Wave configuration
        """
        self.game = game
        self.evaluator = evaluator
        self.arena = arena
        self.config = config
        
        # Wave management
        self.current_wave: Optional[Wave] = None
        self.wave_history: List[Wave] = []
        
        # Statistics
        self.stats = defaultdict(float)
        self.stats['total_waves'] = 0
        self.stats['total_simulations'] = 0
        
        # Interference engine
        if self.config.enable_interference and HAS_INTERFERENCE:
            self.interference_engine = InterferenceEngine(
                threshold=config.interference_threshold
            )
        else:
            self.interference_engine = None
            
        # Adaptive sizing
        self.current_wave_size = config.initial_wave_size
        self.gpu_utilization_history = []
        
    def create_wave(self, root_id: str, size: Optional[int] = None) -> Wave:
        """Create a new wave starting from root
        
        Args:
            root_id: Root node ID
            size: Wave size (uses adaptive sizing if None)
            
        Returns:
            New wave object
        """
        if size is None:
            size = self.current_wave_size
            
        wave = Wave(size=size, root_id=root_id)
        
        # Initialize all paths at root
        root = self.arena.get_node(root_id)
        for i in range(size):
            wave.paths[i] = root_id
            wave.leaf_nodes[i] = root
            
        return wave
        
    def process_wave(self, root_id: str, size: Optional[int] = None) -> Wave:
        """Process a complete wave through all phases
        
        Args:
            root_id: Root node ID
            size: Wave size
            
        Returns:
            Completed wave
        """
        start_time = time.time()
        
        # Create wave
        wave = self.create_wave(root_id, size)
        self.current_wave = wave
        
        # Phase 1: Selection
        phase_start = time.time()
        # Use GPU selection if available
        if HAS_TORCH and torch.cuda.is_available():
            self._run_gpu_selection_phase(wave)
        else:
            self._run_selection_phase(wave)
        self.stats['selection_time'] += time.time() - phase_start
        
        # Phase 2: Expansion
        phase_start = time.time()
        self._run_expansion_phase(wave)
        self.stats['expansion_time'] += time.time() - phase_start
        
        # Phase 3: Evaluation
        phase_start = time.time()
        values = self._run_evaluation_phase(wave)
        wave.values = values
        self.stats['evaluation_time'] += time.time() - phase_start
        
        # Phase 4: Backup
        phase_start = time.time()
        self._run_backup_phase(wave, values)
        self.stats['backup_time'] += time.time() - phase_start
        
        # Complete wave
        wave.completed = True
        wave.phase = 4
        self.wave_history.append(wave)
        self.current_wave = None
        
        # Update statistics
        self.stats['total_waves'] += 1
        self.stats['total_simulations'] += wave.active_paths
        self.stats['total_time'] += time.time() - start_time
        
        # Adaptive sizing
        if self.config.enable_adaptive_sizing:
            self._update_wave_size()
            
        return wave
        
    def _run_selection_phase(self, wave: Wave) -> None:
        """Run selection phase: traverse tree to find leaf nodes
        
        Args:
            wave: Current wave
        """
        wave.phase = 0
        root = self.arena.get_node(wave.root_id)
        
        # Continue selection until all paths reach leaves
        while True:
            # Check which paths need to continue selection
            continuing_paths = []
            for i in range(wave.size):
                if wave.leaf_nodes[i] is not None and not wave.leaf_nodes[i].is_leaf():
                    continuing_paths.append(i)
                    
            if not continuing_paths:
                break  # All paths at leaves
                
            # Select children for continuing paths
            for i in continuing_paths:
                node = wave.leaf_nodes[i]
                if node.children:
                    # Select child based on UCB
                    child = node.select_child(self.config.c_puct)
                    wave.paths[i] = self.arena.node_registry.get(id(child), str(id(child)))
                    wave.leaf_nodes[i] = child
                    
        # Apply interference if enabled
        if self.config.enable_interference and wave.active_paths > 1:
            self._apply_interference(wave)
            
    def _run_gpu_selection_phase(self, wave: Wave) -> None:
        """GPU-accelerated selection phase using batch operations
        
        Args:
            wave: Current wave
        """
        if not HAS_TORCH or torch.cuda.device_count() == 0:
            # Fallback to CPU version
            self._run_selection_phase(wave)
            return
            
        wave.phase = 0
        root = self.arena.get_node(wave.root_id)
        device = torch.device('cuda')
        
        # Initialize paths with root
        for i in range(wave.size):
            wave.leaf_nodes[i] = root
            wave.paths[i] = wave.root_id
            
        # Continue until all paths reach leaves
        while True:
            # Collect nodes that need child selection
            active_nodes = []
            active_indices = []
            
            for i in range(wave.size):
                node = wave.leaf_nodes[i]
                if node is not None and not node.is_leaf() and len(node.children) > 0:
                    active_nodes.append(node)
                    active_indices.append(i)
                    
            if not active_nodes:
                break  # All paths at leaves
                
            # Batch compute UCB scores for all children of all active nodes
            selected_children = self._batch_select_children(active_nodes, device)
            
            # Update wave with selected children
            for idx, (node_idx, child) in enumerate(zip(active_indices, selected_children)):
                wave.paths[node_idx] = self.arena.node_registry.get(id(child), str(id(child)))
                wave.leaf_nodes[node_idx] = child
                
        # Apply interference if enabled
        if self.config.enable_interference and wave.active_paths > 1:
            self._apply_interference(wave)
            
    def _batch_select_children(self, nodes: List[Node], device: torch.device) -> List[Node]:
        """Select children for multiple nodes using GPU batch operations
        
        Args:
            nodes: List of nodes needing child selection
            device: Torch device to use
            
        Returns:
            List of selected child nodes
        """
        selected = []
        
        for node in nodes:
            if not node.children:
                selected.append(node)
                continue
                
            # Prepare tensors for this node's children
            children_list = list(node.children.values())
            n_children = len(children_list)
            
            # Create tensors
            q_values = torch.zeros(n_children, device=device)
            visit_counts = torch.zeros(n_children, device=device)
            priors = torch.zeros(n_children, device=device)
            
            # Fill tensors
            for i, child in enumerate(children_list):
                q_values[i] = child.value()
                visit_counts[i] = child.visit_count
                priors[i] = child.prior
                
            # Parent visit count
            parent_visits = float(node.visit_count)
            
            # Compute UCB scores on GPU
            # UCB = Q + c_puct * prior * sqrt(parent_visits) / (1 + visits)
            exploration = self.config.c_puct * priors * torch.sqrt(torch.tensor(parent_visits, device=device)) / (1 + visit_counts)
            ucb_scores = q_values + exploration
            
            # Select best child
            best_idx = ucb_scores.argmax().item()
            selected.append(children_list[best_idx])
            
        return selected
            
    def _run_expansion_phase(self, wave: Wave) -> None:
        """Run expansion phase: expand leaf nodes
        
        Args:
            wave: Current wave
        """
        wave.phase = 1
        
        # Group nodes by state for batch evaluation
        state_groups = defaultdict(list)
        for i in range(wave.size):
            node = wave.leaf_nodes[i]
            if node is not None and not node.is_expanded and not node.is_terminal:
                state_groups[id(node)].append(i)
                
        # Expand each unique node
        for node_id, indices in state_groups.items():
            node = wave.leaf_nodes[indices[0]]
            
            # Get legal moves
            legal_moves = self.game.get_legal_moves(node.state)
            if not legal_moves:
                continue
                
            # Get policy from evaluator
            state_tensor = self.game.encode_for_nn(node.state, [])
            legal_mask = self.game.get_action_probabilities_mask(node.state)
            policy, _ = self.evaluator.evaluate(state_tensor, legal_mask)
            
            # Create action probabilities
            action_probs = {}
            for move in legal_moves:
                action_idx = self.game.move_to_action_index(move)
                action_probs[move] = float(policy[action_idx])
                
            # Apply phase kicks if enabled
            if self.config.enable_phase_kicks:
                action_probs = self._apply_phase_kicks(action_probs, node)
                
            # Create child states and check for transpositions
            child_states = {}
            child_hashes = {}
            
            for move in legal_moves:
                child_state = self.game.apply_move(node.state, move)
                child_states[move] = child_state
                
                # Get hash if available
                if hasattr(self.game, 'get_hash'):
                    child_hashes[move] = self.game.get_hash(child_state)
                    
            # Expand node (will check transpositions internally)
            node.expand(action_probs, child_states)
            
            # Add children to arena with hashes
            for move, child in node.children.items():
                child_hash = child_hashes.get(move)
                child_id = self.arena.add_node(child, child_hash)
                # If transposition found, update child reference
                if child_id != id(child):
                    # Replace with existing node
                    existing_child = self.arena.get_node(child_id)
                    if existing_child:
                        node.children[move] = existing_child
            
            # Update wave to point to children
            for idx in indices:
                if node.children:
                    # Select initial child (highest prior)
                    best_child = max(node.children.values(), key=lambda c: c.prior)
                    wave.leaf_nodes[idx] = best_child
                    
    def _run_evaluation_phase(self, wave: Wave) -> np.ndarray:
        """Run evaluation phase: evaluate leaf positions
        
        Args:
            wave: Current wave
            
        Returns:
            Array of position values
        """
        wave.phase = 2
        
        # Collect unique states to evaluate
        unique_states = {}
        state_indices = {}
        
        for i in range(wave.size):
            node = wave.leaf_nodes[i]
            if node is not None:
                state_id = id(node.state)
                if state_id not in unique_states:
                    unique_states[state_id] = node.state
                    state_indices[state_id] = []
                state_indices[state_id].append(i)
                
        # Batch evaluate unique states
        if unique_states:
            # Prepare batch
            states = []
            legal_masks = []
            
            for state in unique_states.values():
                state_tensor = self.game.encode_for_nn(state, [])
                states.append(state_tensor)
                legal_masks.append(self.game.get_action_probabilities_mask(state))
                
            states = np.stack(states)
            legal_masks = np.stack(legal_masks)
            
            # Evaluate
            _, values = self.evaluator.evaluate_batch(states, legal_masks)
            
            # Map values back to wave
            wave_values = np.zeros(wave.size)
            for i, (state_id, indices) in enumerate(state_indices.items()):
                for idx in indices:
                    wave_values[idx] = values[i]
                    
            return wave_values
        else:
            return np.zeros(wave.size)
            
    def _run_backup_phase(self, wave: Wave, values: np.ndarray) -> None:
        """Run backup phase: propagate values up the tree
        
        Args:
            wave: Current wave
            values: Evaluation values to backup
        """
        wave.phase = 3
        
        # Backup each path
        for i in range(wave.size):
            node = wave.leaf_nodes[i]
            if node is not None:
                node.backup(values[i])
                
    def _apply_interference(self, wave: Wave) -> None:
        """Apply interference to reduce path similarity
        
        Args:
            wave: Current wave
        """
        if not self.interference_engine:
            return
            
        # Extract paths as action sequences
        paths = []
        for i in range(wave.size):
            path = []
            node = wave.leaf_nodes[i]
            while node is not None and node.parent is not None:
                if node.action is not None:
                    path.append(node.action)
                node = node.parent
            paths.append(list(reversed(path)))
            
        if not paths or not any(paths):  # Skip if no valid paths
            return
            
        # Compute interference
        interference = self.interference_engine.compute_interference(paths)
        
        # Log statistics
        logger.debug(f"Average interference: {np.mean(interference):.3f}")
        
        # In full implementation, this would modify selection probabilities
        # during the selection phase to avoid redundant paths
        
        
    def _apply_phase_kicks(
        self,
        action_probs: Dict[int, float],
        node: Node
    ) -> Dict[int, float]:
        """Apply phase kicks to action probabilities
        
        Args:
            action_probs: Original action probabilities
            node: Current node
            
        Returns:
            Modified action probabilities
        """
        # Simplified implementation - full version would use
        # complex-valued probabilities with uncertainty-based phases
        
        # Add small phase based on visit count (uncertainty proxy)
        uncertainty = 1.0 / (1.0 + node.visit_count)
        
        modified_probs = {}
        for action, prob in action_probs.items():
            # Add phase-based perturbation
            phase = uncertainty * np.sin(action * 0.1)  # Deterministic phase
            modified_probs[action] = prob * (1.0 + 0.1 * phase)
            
        # Renormalize
        total = sum(modified_probs.values())
        if total > 0:
            for action in modified_probs:
                modified_probs[action] /= total
                
        return modified_probs
        
    def _update_wave_size(self) -> None:
        """Update wave size based on performance metrics"""
        # Simplified adaptive sizing
        # Full implementation would use GPU utilization metrics
        
        if self.stats['total_waves'] > 0:
            avg_time = self.stats['total_time'] / self.stats['total_waves']
            
            # Increase size if processing is fast
            if avg_time < 0.1:  # Less than 100ms per wave
                self.current_wave_size = min(
                    int(self.current_wave_size * 1.2),
                    self.config.max_wave_size
                )
            # Decrease size if processing is slow
            elif avg_time > 0.5:  # More than 500ms per wave
                self.current_wave_size = max(
                    int(self.current_wave_size * 0.8),
                    self.config.min_wave_size
                )
                
    def get_statistics(self) -> Dict[str, float]:
        """Get wave processing statistics
        
        Returns:
            Dictionary of statistics
        """
        stats = dict(self.stats)
        
        # Compute averages
        if stats['total_waves'] > 0:
            stats['average_wave_size'] = stats['total_simulations'] / stats['total_waves']
            stats['average_wave_time'] = stats['total_time'] / stats['total_waves']
        else:
            stats['average_wave_size'] = 0
            stats['average_wave_time'] = 0
            
        stats['current_wave_size'] = self.current_wave_size
        
        return stats
        
    def reset_statistics(self) -> None:
        """Reset all statistics"""
        self.stats.clear()
        self.stats['total_waves'] = 0
        self.stats['total_simulations'] = 0