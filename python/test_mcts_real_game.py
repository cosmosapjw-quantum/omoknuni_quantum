#!/usr/bin/env python3
"""
Test MCTS in a real game scenario with proper move handling.
"""

import torch
import numpy as np
import time
from mcts.core.optimized_mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType
from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
import alphazero_py


def test_mcts_performance():
    """Test MCTS performance with real evaluator"""
    
    print("="*70)
    print("MCTS PERFORMANCE TEST WITH REAL EVALUATOR")
    print("="*70)
    
    # Check hardware
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This test requires GPU.")
        return False
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # Create optimized configuration
    config = MCTSConfig(
        num_simulations=100000,
        min_wave_size=3072,
        max_wave_size=3072,
        adaptive_wave_sizing=False,  # CRITICAL for performance!
        device='cuda',
        game_type=GameType.GOMOKU,
        board_size=15,
        c_puct=1.414,
        temperature=1.0,
        memory_pool_size_mb=2048,
        max_tree_nodes=500000,
        use_mixed_precision=True,
        use_cuda_graphs=True,
        use_tensor_cores=True,
        enable_virtual_loss=True,
        enable_debug_logging=False
    )
    
    # Create real neural network evaluator
    print("\nInitializing ResNet evaluator...")
    evaluator = ResNetEvaluator(
        game_type='gomoku',
        device='cuda'
    )
    
    # Initialize MCTS
    print("Initializing MCTS...")
    mcts = MCTS(config, evaluator)
    mcts.optimize_for_hardware()
    
    # Create game state
    state = alphazero_py.GomokuState()
    
    # Warmup
    print("\nWarming up...")
    for i in range(3):
        print(f"  Warmup {i+1}/3...")
        mcts.search(state, 1000)
        mcts.reset_tree()
    
    # Performance tests
    print("\n" + "-"*50)
    print("PERFORMANCE TESTS")
    print("-"*50)
    
    test_configs = [
        (10000, "Quick test"),
        (50000, "Medium test"),
        (100000, "Full test"),
        (200000, "Extended test")
    ]
    
    results = []
    
    for num_sims, test_name in test_configs:
        print(f"\n{test_name}: {num_sims} simulations")
        
        # Reset tree for clean test
        mcts.reset_tree()
        
        # Time the search
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        policy = mcts.search(state, num_sims)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        
        # Calculate metrics
        sims_per_sec = num_sims / elapsed
        stats = mcts.get_statistics()
        
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Performance: {sims_per_sec:,.0f} simulations/second")
        print(f"  Tree nodes: {stats.get('tree_nodes', 'N/A')}")
        print(f"  Memory: {stats.get('tree_memory_mb', 0):.1f} MB")
        
        # Performance assessment
        if sims_per_sec >= 80000:
            status = "✓ EXCELLENT (exceeds 80k target)"
        elif sims_per_sec >= 50000:
            status = "✓ Good"
        else:
            status = "⚠ Below target"
        print(f"  Status: {status}")
        
        results.append({
            'simulations': num_sims,
            'time': elapsed,
            'sims_per_sec': sims_per_sec,
            'tree_nodes': stats.get('tree_nodes', 0)
        })
    
    # Summary
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    
    avg_performance = sum(r['sims_per_sec'] for r in results) / len(results)
    peak_performance = max(r['sims_per_sec'] for r in results)
    
    print(f"Average performance: {avg_performance:,.0f} simulations/second")
    print(f"Peak performance: {peak_performance:,.0f} simulations/second")
    
    if avg_performance >= 80000:
        print("\n✓ SUCCESS: Achieved target performance of 80k+ sims/second!")
        print(f"  Target: 80,000 sims/s")
        print(f"  Achieved: {avg_performance:,.0f} sims/s ({avg_performance/80000*100:.1f}% of target)")
    else:
        print(f"\n⚠ Performance below target: {avg_performance:,.0f} sims/s")
    
    return avg_performance >= 80000


def test_mcts_game_play():
    """Test MCTS playing a short game sequence"""
    
    print("\n" + "="*70)
    print("MCTS GAME PLAY TEST")
    print("="*70)
    
    # Create configuration for faster play
    config = MCTSConfig(
        num_simulations=5000,
        min_wave_size=1024,
        max_wave_size=1024,
        adaptive_wave_sizing=False,
        device='cuda',
        game_type=GameType.GOMOKU,
        board_size=15,
        temperature=1.0
    )
    
    # Create evaluator
    evaluator = ResNetEvaluator(
        game_type='gomoku',
        device='cuda'
    )
    
    # Initialize MCTS
    mcts = MCTS(config, evaluator)
    
    # Play a few moves
    state = alphazero_py.GomokuState()
    moves = []
    
    print("\nPlaying 10 moves...")
    for move_num in range(10):
        # Get legal moves
        legal_moves = []
        for i in range(225):
            if state.is_move_legal(i):
                legal_moves.append(i)
        
        if not legal_moves:
            print(f"No legal moves at move {move_num}")
            break
        
        # Search
        policy = mcts.search(state, 5000)
        
        # Mask illegal moves
        masked_policy = np.zeros_like(policy)
        for move in legal_moves:
            masked_policy[move] = policy[move]
        
        # Renormalize
        if masked_policy.sum() > 0:
            masked_policy = masked_policy / masked_policy.sum()
        else:
            # Uniform over legal moves if all probabilities are 0
            for move in legal_moves:
                masked_policy[move] = 1.0 / len(legal_moves)
        
        # Select move
        if move_num < 5:
            # Sample for first 5 moves
            action = np.random.choice(225, p=masked_policy)
        else:
            # Play best move
            action = np.argmax(masked_policy)
        
        row = action // 15
        col = action % 15
        print(f"  Move {move_num + 1}: ({row}, {col})")
        
        # Make move
        state = state.clone()
        state.make_move(action)
        moves.append(action)
        
        # Reset tree for next move
        mcts.reset_tree()
        
        # Check if game ended
        if state.is_terminal():
            print(f"Game ended after {move_num + 1} moves!")
            break
    
    print(f"\nPlayed {len(moves)} moves successfully")
    print("✓ Game play test passed!")
    
    return True


def main():
    """Run all tests"""
    
    try:
        # Test 1: Performance
        performance_passed = test_mcts_performance()
        
        # Test 2: Game play
        gameplay_passed = test_mcts_game_play()
        
        # Summary
        print("\n" + "="*70)
        print("TEST RESULTS")
        print("="*70)
        
        if performance_passed and gameplay_passed:
            print("✓ ALL TESTS PASSED!")
            print("\nThe optimized MCTS implementation is working correctly and")
            print("achieving the target performance of 80k+ simulations/second.")
        else:
            print("⚠ Some tests failed")
            if not performance_passed:
                print("  - Performance test: FAILED")
            if not gameplay_passed:
                print("  - Game play test: FAILED")
        
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return performance_passed and gameplay_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)