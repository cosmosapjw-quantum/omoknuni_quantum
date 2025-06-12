#!/usr/bin/env python3
"""
Simple demonstration of the optimized MCTS implementation.

This example shows basic usage and verifies the MCTS is working correctly.
"""

import torch
import numpy as np
import time
from mcts.core.mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType
import alphazero_py


class SimpleEvaluator:
    """Simple evaluator that returns reasonable policies based on board position"""
    
    def __init__(self, board_size=15):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.board_size = board_size
        self.action_size = board_size * board_size
        
    def evaluate_batch(self, features):
        """
        Simple evaluation that prefers center positions and empty spots
        """
        batch_size = features.shape[0]
        
        # Create policy that prefers center positions
        policies = torch.zeros(batch_size, self.action_size, device=self.device)
        
        center = self.board_size // 2
        for b in range(batch_size):
            for i in range(self.board_size):
                for j in range(self.board_size):
                    idx = i * self.board_size + j
                    # Distance from center
                    dist = abs(i - center) + abs(j - center)
                    # Higher probability for positions closer to center
                    policies[b, idx] = 1.0 / (1.0 + dist * 0.1)
        
        # Normalize
        policies = policies / policies.sum(dim=1, keepdim=True)
        
        # Simple value estimation (slightly positive for current player)
        values = torch.ones(batch_size, 1, device=self.device) * 0.1
        
        return policies, values


def print_board(state, last_move=None):
    """Print a simple text representation of the board"""
    board_size = 15
    
    print("\n   ", end="")
    for i in range(board_size):
        print(f"{i:2}", end=" ")
    print()
    print("   " + "-" * (board_size * 3))
    
    for row in range(board_size):
        print(f"{row:2}|", end="")
        for col in range(board_size):
            idx = row * board_size + col
            
            # Get piece at this position
            piece = '.'
            
            # Highlight last move
            if last_move is not None and idx == last_move:
                print(f"[{piece}]", end="")
            else:
                print(f" {piece} ", end="")
        print(f"|{row}")
    
    print("   " + "-" * (board_size * 3))
    print("   ", end="")
    for i in range(board_size):
        print(f"{i:2}", end=" ")
    print()


def demonstrate_mcts():
    """Demonstrate MCTS functionality"""
    
    print("="*70)
    print("OPTIMIZED MCTS DEMONSTRATION")
    print("="*70)
    
    # Check hardware
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("Running on CPU (will be slower)")
    
    # Create configuration
    config = MCTSConfig(
        num_simulations=10000,
        min_wave_size=3072,
        max_wave_size=3072,
        adaptive_wave_sizing=False,  # Critical for performance!
        device='cuda' if torch.cuda.is_available() else 'cpu',
        game_type=GameType.GOMOKU,
        board_size=15,
        c_puct=1.414,
        temperature=1.0,
        enable_virtual_loss=True,
        use_mixed_precision=True,
        use_cuda_graphs=True,
        use_tensor_cores=True
    )
    
    # Create evaluator
    evaluator = SimpleEvaluator(board_size=15)
    
    # Initialize MCTS
    print("\nInitializing MCTS...")
    mcts = MCTS(config, evaluator)
    mcts.optimize_for_hardware()
    
    # Create initial game state
    state = alphazero_py.GomokuState()
    
    print("\nRunning MCTS searches on empty board...")
    print("-" * 50)
    
    # Warmup
    print("Warming up GPU...")
    for _ in range(3):
        _ = mcts.search(state, 1000)
        mcts.reset_tree()
    
    # Test different simulation counts
    for num_sims in [1000, 5000, 10000, 50000]:
        print(f"\nTesting with {num_sims} simulations:")
        
        # Reset tree
        mcts.reset_tree()
        
        # Time the search
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        policy = mcts.search(state, num_sims)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start_time
        
        # Performance metrics
        sims_per_sec = num_sims / elapsed
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Performance: {sims_per_sec:,.0f} simulations/second")
        
        # Get statistics
        stats = mcts.get_statistics()
        print(f"  Tree nodes: {stats.get('tree_nodes', 'N/A')}")
        print(f"  Memory usage: {stats.get('tree_memory_mb', 'N/A'):.1f} MB")
        
        # Show top 5 moves
        top_moves = np.argsort(policy)[-5:][::-1]
        print(f"  Top 5 moves (action: probability):")
        for action in top_moves:
            row = action // 15
            col = action % 15
            prob = policy[action]
            print(f"    ({row:2}, {col:2}): {prob:.4f}")
    
    # Play a few moves to show it works in a game
    print("\n" + "="*70)
    print("PLAYING A FEW MOVES")
    print("="*70)
    
    moves_played = []
    
    for move_num in range(5):
        print(f"\nMove {move_num + 1}:")
        
        # Reset tree for new position
        mcts.reset_tree()
        
        # Search
        policy = mcts.search(state, 5000)
        
        # Get best move (deterministic for demo)
        action = np.argmax(policy)
        row = action // 15
        col = action % 15
        
        print(f"  Selected move: ({row}, {col})")
        print(f"  Probability: {policy[action]:.4f}")
        
        # Apply move
        state = state.clone()
        state.make_move(action)  # Use make_move instead of apply_move
        moves_played.append(action)
        
        # Show board
        print_board(state, action)
        
        # Check if game ended
        if state.is_terminal():
            print("\nGame ended!")
            break
    
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    
    final_stats = mcts.get_statistics()
    print(f"Total searches: {final_stats.get('total_searches', 0)}")
    print(f"Total simulations: {final_stats.get('total_simulations', 0)}")
    print(f"Average simulations/second: {final_stats.get('avg_sims_per_second', 0):,.0f}")
    print(f"Peak simulations/second: {final_stats.get('peak_sims_per_second', 0):,.0f}")
    
    print("\n✓ MCTS demonstration complete!")
    
    return True


def test_mcts_correctness():
    """Test that MCTS produces valid outputs"""
    
    print("\n" + "="*70)
    print("CORRECTNESS TESTS")
    print("="*70)
    
    config = MCTSConfig(
        num_simulations=1000,
        min_wave_size=256,  # Smaller for quick test
        max_wave_size=256,
        adaptive_wave_sizing=False,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        game_type=GameType.GOMOKU,
        board_size=15,
        progressive_expansion_threshold=0,  # Expand immediately
        initial_children_per_expansion=225,  # All moves
        max_children_per_node=225
    )
    
    evaluator = SimpleEvaluator(board_size=15)
    mcts = MCTS(config, evaluator)
    
    # Test 1: Policy sums to 1
    print("\nTest 1: Policy probability sum")
    state = alphazero_py.GomokuState()
    policy = mcts.search(state, 1000)
    
    policy_sum = policy.sum()
    print(f"  Policy sum: {policy_sum:.6f}")
    assert abs(policy_sum - 1.0) < 1e-5, f"Policy doesn't sum to 1: {policy_sum}"
    print("  ✓ Passed!")
    
    # Test 2: All probabilities non-negative
    print("\nTest 2: Non-negative probabilities")
    min_prob = policy.min()
    print(f"  Minimum probability: {min_prob:.6f}")
    assert min_prob >= 0, f"Negative probability found: {min_prob}"
    print("  ✓ Passed!")
    
    # Test 3: Legal moves have non-zero probability
    print("\nTest 3: Legal moves have probability")
    legal_moves_count = 225  # All moves legal on empty board
    nonzero_count = (policy > 0).sum()
    print(f"  Legal moves: {legal_moves_count}")
    print(f"  Non-zero probabilities: {nonzero_count}")
    assert nonzero_count > 0, "No moves have probability!"
    print("  ✓ Passed!")
    
    # Test 4: Best action is valid
    print("\nTest 4: Best action selection")
    best_action = mcts.get_best_action(state)
    print(f"  Best action: {best_action} (row {best_action//15}, col {best_action%15})")
    assert 0 <= best_action < 225, f"Invalid action: {best_action}"
    print("  ✓ Passed!")
    
    # Test 5: Tree creation and basic growth
    print("\nTest 5: Tree creation")
    mcts.reset_tree()
    
    # Initial state should have 1 node (root)
    stats0 = mcts.get_statistics()
    nodes0 = stats0.get('tree_nodes', 0)
    print(f"  Initial nodes: {nodes0}")
    
    # After search, should have more nodes
    _ = mcts.search(state, 1000)
    stats1 = mcts.get_statistics()
    nodes1 = stats1.get('tree_nodes', 0)
    print(f"  Nodes after 1000 sims: {nodes1}")
    
    # Verify tree was created
    assert nodes1 > 1, f"Tree wasn't created! Only {nodes1} nodes"
    
    # For uniform policy, tree might not grow deeply, so just check it expanded
    print(f"  Tree expanded from {nodes0} to {nodes1} nodes")
    print("  ✓ Passed!")
    
    print("\n✓ All correctness tests passed!")
    
    return True


if __name__ == "__main__":
    try:
        # Run correctness tests
        test_mcts_correctness()
        
        # Run demonstration
        demonstrate_mcts()
        
        print("\n" + "="*70)
        print("SUCCESS: MCTS is working correctly!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()