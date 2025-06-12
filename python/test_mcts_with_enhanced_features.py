#!/usr/bin/env python3
"""
Test MCTS with enhanced tensor representation for realistic game play.
"""

import torch
import numpy as np
import time
from mcts.core.optimized_mcts import MCTS, MCTSConfig
from mcts.quantum.quantum_features import QuantumConfig
from mcts.gpu.gpu_game_states import GameType
from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
from mcts.core.game_interface import GameInterface, GameType as GameInterfaceType
import alphazero_py


class EnhancedEvaluatorWrapper:
    """Wrapper that converts game states to enhanced 20-channel representation"""
    
    def __init__(self, resnet_evaluator, game_interface):
        self.evaluator = resnet_evaluator
        self.game_interface = game_interface
        self.device = self.evaluator.device
        
    def evaluate_batch(self, features):
        """
        Convert features to enhanced representation and evaluate
        
        Args:
            features: Batch of game features (basic representation)
            
        Returns:
            policies, values from ResNet evaluator
        """
        # The features are already in the enhanced format from GPUGameStates
        # Just pass through to the evaluator
        return self.evaluator.evaluate_batch(features)


def test_mcts_with_enhanced_features():
    """Test MCTS performance with enhanced feature representation"""
    
    print("="*70)
    print("MCTS TEST WITH ENHANCED TENSOR REPRESENTATION")
    print("="*70)
    
    # Check hardware
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This test requires GPU.")
        return False
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # Create game interface
    game_interface = GameInterface(game_type=GameInterfaceType.GOMOKU)
    
    # Create quantum configuration for enhanced exploration
    quantum_config = QuantumConfig(
        enable_quantum=True,
        quantum_level='tree_level',  # Enable quantum tree-level features
        min_wave_size=32,           # Apply quantum for reasonable batch sizes
        optimal_wave_size=3072,     # Match classical wave size
        hbar_eff=0.05,             # Moderate quantum effects
        coupling_strength=0.1,
        interference_alpha=0.05,    # Moderate interference
        phase_kick_strength=0.1,    # Phase kick for exploration
        use_mixed_precision=True,
        fast_mode=True,            # Production optimizations
        device='cuda'
    )
    
    # Create optimized configuration with quantum features
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
        enable_debug_logging=False,
        # Quantum features
        enable_quantum=True,
        quantum_config=quantum_config
    )
    
    # Create ResNet evaluator (expects 20 channels)
    print("\nInitializing ResNet evaluator...")
    resnet_evaluator = ResNetEvaluator(
        game_type='gomoku',
        device='cuda'
    )
    # Enable torch tensor output for GPU MCTS
    resnet_evaluator._return_torch_tensors = True
    
    # Create wrapper evaluator
    evaluator = EnhancedEvaluatorWrapper(resnet_evaluator, game_interface)
    
    # Initialize MCTS
    print("Initializing MCTS...")
    mcts = MCTS(config, evaluator)
    mcts.optimize_for_hardware()
    
    # Enable enhanced features in the GPU game states
    print("Enabling enhanced 20-channel features...")
    mcts.game_states.enable_enhanced_features()
    
    # Create game state
    state = alphazero_py.GomokuState()
    
    # Get enhanced representation to verify it works
    print("\nVerifying enhanced tensor representation...")
    enhanced_tensor = game_interface.get_enhanced_tensor_representation(state)
    print(f"Enhanced tensor shape: {enhanced_tensor.shape}")
    print(f"Expected: (20, 15, 15) for Gomoku")
    assert enhanced_tensor.shape == (20, 15, 15), f"Unexpected shape: {enhanced_tensor.shape}"
    print("✓ Enhanced representation verified!")
    
    # Warmup
    print("\nWarming up...")
    for i in range(3):
        print(f"  Warmup {i+1}/3...")
        mcts.search(state, 1000)
        mcts.reset_tree()
    
    # Performance tests
    print("\n" + "-"*50)
    print("PERFORMANCE TESTS WITH ENHANCED FEATURES")
    print("-"*50)
    
    test_configs = [
        (10000, "Quick test"),
        (50000, "Medium test"),
        (100000, "Full test")
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
        
        # Show quantum statistics if available
        if hasattr(mcts, 'quantum_features') and mcts.quantum_features and hasattr(mcts.quantum_features, 'stats'):
            q_stats = mcts.quantum_features.stats
            print(f"  Quantum applications: {q_stats.get('quantum_applications', 0)}")
            print(f"  Quantum selections: {q_stats.get('total_selections', 0)}")
            print(f"  Quantum overhead: {q_stats.get('avg_overhead', 1.0):.2f}x")
        
        # Show top moves
        top_moves = np.argsort(policy)[-5:][::-1]
        print(f"  Top 5 moves:")
        for i, action in enumerate(top_moves):
            row = action // 15
            col = action % 15
            prob = policy[action]
            print(f"    {i+1}. ({row:2}, {col:2}): {prob:.4f}")
        
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
    
    # Play a short game sequence
    print("\n" + "="*70)
    print("PLAYING TEST GAME WITH ENHANCED FEATURES")
    print("="*70)
    
    # Reset for game play
    state = alphazero_py.GomokuState()
    mcts.reset_tree()
    
    moves_played = []
    print("\nPlaying 10 moves...")
    
    for move_num in range(10):
        # Search with fewer simulations for faster play
        policy = mcts.search(state, 5000)
        
        # Get legal moves
        legal_moves = []
        for i in range(225):
            if state.is_legal_move(i):
                legal_moves.append(i)
        
        if not legal_moves:
            print(f"No legal moves at move {move_num}")
            break
        
        # Mask illegal moves and renormalize
        masked_policy = np.zeros_like(policy)
        for move in legal_moves:
            masked_policy[move] = policy[move]
        
        if masked_policy.sum() > 0:
            masked_policy = masked_policy / masked_policy.sum()
        else:
            # Uniform over legal moves if needed
            for move in legal_moves:
                masked_policy[move] = 1.0 / len(legal_moves)
        
        # Select move (deterministic for test)
        action = np.argmax(masked_policy)
        row = action // 15
        col = action % 15
        
        print(f"  Move {move_num + 1}: ({row}, {col}) - Probability: {masked_policy[action]:.3f}")
        
        # Make move
        state = state.clone()
        state.make_move(action)
        moves_played.append(action)
        
        # Reset tree for next position
        mcts.reset_tree()
        
        # Check if game ended
        if state.is_terminal():
            winner = state.get_winner()
            if winner == 1:
                print(f"  Game ended: Player 1 wins!")
            elif winner == -1:
                print(f"  Game ended: Player 2 wins!")
            else:
                print(f"  Game ended: Draw!")
            break
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    avg_performance = sum(r['sims_per_sec'] for r in results) / len(results)
    peak_performance = max(r['sims_per_sec'] for r in results)
    
    print(f"Average performance: {avg_performance:,.0f} simulations/second")
    print(f"Peak performance: {peak_performance:,.0f} simulations/second")
    print(f"Moves played successfully: {len(moves_played)}")
    
    if avg_performance >= 80000:
        print("\n✓ SUCCESS: MCTS with enhanced features achieves 80k+ sims/second!")
        print(f"  Target: 80,000 sims/s")
        print(f"  Achieved: {avg_performance:,.0f} sims/s ({avg_performance/80000*100:.1f}% of target)")
    else:
        print(f"\n⚠ Performance: {avg_performance:,.0f} sims/s")
    
    print("\n✓ Enhanced feature test with quantum MCTS completed successfully!")
    
    # Final quantum summary
    if hasattr(mcts, 'quantum_features') and mcts.quantum_features:
        total_q_apps = sum(r.get('quantum_applications', 0) for r in [mcts.quantum_features.stats] if r)
        total_q_selections = sum(r.get('total_selections', 0) for r in [mcts.quantum_features.stats] if r)
        avg_q_overhead = mcts.quantum_features.stats.get('avg_overhead', 1.0)
        
        print(f"\nQuantum MCTS Summary:")
        print(f"  Total quantum applications: {total_q_apps}")
        print(f"  Total quantum selections: {total_q_selections}")
        print(f"  Average quantum overhead: {avg_q_overhead:.2f}x")
        
        if avg_q_overhead <= 2.0:
            print(f"  ✓ Quantum overhead within target (< 2.0x)")
        else:
            print(f"  ⚠ Quantum overhead above target: {avg_q_overhead:.2f}x")
    
    print("="*70)
    
    return avg_performance >= 80000


if __name__ == "__main__":
    try:
        success = test_mcts_with_enhanced_features()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)