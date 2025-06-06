#!/usr/bin/env python3
"""Simple test to debug MCTS issues"""

import torch
import numpy as np
from mcts.gpu.csr_tree import CSRTree, CSRTreeConfig
from mcts.gpu.optimized_wave_engine import OptimizedWaveEngine, OptimizedWaveConfig
from mcts.core.game_interface import GameInterface, GameType
from mcts.core.evaluator import Evaluator, EvaluatorConfig


class DummyEvaluator(Evaluator):
    def __init__(self):
        config = EvaluatorConfig(device='cpu', batch_size=32)
        super().__init__(config, action_size=225)
        
    def evaluate(self, state, legal_mask=None, temperature=1.0):
        policy = np.ones(self.action_size) / self.action_size
        value = 0.0
        return policy, value
    
    def evaluate_batch(self, states, legal_masks=None, temperature=1.0):
        if isinstance(states, torch.Tensor):
            batch_size = states.shape[0]
        elif isinstance(states, np.ndarray):
            batch_size = states.shape[0]
        else:
            batch_size = len(states)
        policies = np.ones((batch_size, self.action_size)) / self.action_size
        values = np.zeros(batch_size)
        return policies, values


# Create components
print("Creating game and tree...")
game = GameInterface(GameType.GOMOKU)
evaluator = DummyEvaluator()

tree_config = CSRTreeConfig(
    max_nodes=1000,
    max_edges=5000,
    device='cpu'
)
tree = CSRTree(tree_config)

wave_config = OptimizedWaveConfig(
    wave_size=4,
    max_depth=10,
    c_puct=1.0,
    device='cpu',
    enable_memory_pooling=False,
)

wave_engine = OptimizedWaveEngine(tree, wave_config, game, evaluator)

# Add root
print("\nAdding root...")
root_state = game.create_initial_state()
root_idx = tree.add_root(state=root_state)
print(f"Root index: {root_idx}")

# Check initial state
print(f"\nInitial tree state:")
print(f"  Nodes: {tree.num_nodes}")
print(f"  Edges: {tree.num_edges}")
print(f"  Children of root: {tree.children[root_idx][:10]}")

# Run one wave
print("\nRunning first wave...")
try:
    result = wave_engine.run_wave(root_state, wave_size=4)
    print(f"Wave completed successfully")
    print(f"  Nodes after wave: {tree.num_nodes}")
    print(f"  Edges after wave: {tree.num_edges}")
    
    # Check children of root
    children_indices = tree.children[root_idx]
    valid_children = children_indices[children_indices >= 0]
    print(f"  Root now has {len(valid_children)} children")
    
    if len(valid_children) > 0:
        print(f"  First few children: {valid_children[:5]}")
        print(f"  Their actions: {tree.parent_actions[valid_children[:5]]}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    
print("\nDebugging path selection...")
# Let's manually check what happens when selecting paths
current_nodes = torch.zeros(4, dtype=torch.int32)  # Start at root
children = tree.children[current_nodes]  # Get children of all roots
print(f"Children tensor shape: {children.shape}")
print(f"Valid children mask: {(children >= 0).sum(dim=1)}")

# Check if we're getting correct actions
first_child = children[0, 0].item()
if first_child >= 0:
    print(f"\nFirst child of root: {first_child}")
    print(f"Action that led to it: {tree.parent_actions[first_child]}")
    print(f"Is this action valid? {tree.parent_actions[first_child].item() in game.get_legal_moves(root_state)}")

print("\nTrying second wave...")
try:
    result = wave_engine.run_wave(root_state, wave_size=4)
    print(f"Second wave completed")
    print(f"  Nodes: {tree.num_nodes}")
    print(f"  Edges: {tree.num_edges}")
except Exception as e:
    print(f"Error in second wave: {e}")
    import traceback
    traceback.print_exc()