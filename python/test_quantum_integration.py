#!/usr/bin/env python3
"""
Test quantum MCTS integration with optimized classical MCTS
"""

import torch
import numpy as np
import logging
from mcts.core.optimized_mcts import MCTS, MCTSConfig
from mcts.quantum.quantum_features import QuantumConfig
from mcts.gpu.gpu_game_states import GameType
from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
import alphazero_py

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)


def test_quantum_integration():
    """Test quantum features with optimized MCTS"""
    
    print("=" * 70)
    print("QUANTUM MCTS INTEGRATION TEST")
    print("=" * 70)
    
    # Check hardware
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This test requires GPU.")
        return False
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Create quantum configuration
    quantum_config = QuantumConfig(
        enable_quantum=True,
        quantum_level='tree_level',  # Enable quantum features
        min_wave_size=1,             # Very low threshold for testing
        optimal_wave_size=512,       # Reasonable size for test
        hbar_eff=0.05,              # Moderate quantum effects
        coupling_strength=0.1,
        interference_alpha=0.05,
        phase_kick_strength=0.1,
        use_mixed_precision=True,
        fast_mode=True,
        device='cuda'
    )
    
    # Create MCTS configuration with quantum features
    config = MCTSConfig(
        num_simulations=5000,
        min_wave_size=512,
        max_wave_size=512,
        adaptive_wave_sizing=False,
        device='cuda',
        game_type=GameType.GOMOKU,
        board_size=15,
        enable_quantum=True,         # Enable quantum features
        quantum_config=quantum_config,
        enable_debug_logging=True    # Enable debug output
    )
    
    # Create evaluator
    print("\nInitializing ResNet evaluator...")
    resnet_evaluator = ResNetEvaluator(
        game_type='gomoku',
        device='cuda'
    )
    resnet_evaluator._return_torch_tensors = True
    
    # Initialize MCTS with quantum features
    print("Initializing MCTS with quantum features...")
    mcts = MCTS(config, resnet_evaluator)
    
    # Enable enhanced features for 20-channel representation
    print("Enabling enhanced 20-channel features...")
    mcts.game_states.enable_enhanced_features()
    
    # Verify quantum features are enabled
    if mcts.quantum_features is None:
        print("ERROR: Quantum features not initialized!")
        return False
    
    print(f"✓ Quantum features initialized: {type(mcts.quantum_features)}")
    print(f"✓ Quantum config: {mcts.quantum_features.config}")
    
    # Create test game state
    state = alphazero_py.GomokuState()
    
    # Test quantum-enhanced search
    print("\nTesting quantum-enhanced search...")
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    # Warmup
    mcts.search(state, 1000)
    mcts.reset_tree()
    
    # Timed test
    torch.cuda.synchronize()
    start_time.record()
    
    policy = mcts.search(state, 5000)
    
    end_time.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_time.elapsed_time(end_time)
    elapsed_s = elapsed_ms / 1000.0
    sims_per_sec = 5000 / elapsed_s
    
    print(f"Performance: {sims_per_sec:,.0f} simulations/second")
    print(f"Time: {elapsed_s:.3f}s")
    
    # Get quantum statistics
    if hasattr(mcts.quantum_features, 'stats'):
        stats = mcts.quantum_features.stats
        print(f"\nQuantum Statistics:")
        print(f"  Quantum applications: {stats.get('quantum_applications', 0)}")
        print(f"  Total selections: {stats.get('total_selections', 0)}")
        print(f"  Low visit nodes: {stats.get('low_visit_nodes', 0)}")
        print(f"  Average overhead: {stats.get('avg_overhead', 1.0):.2f}x")
    
    # Test tree stats
    mcts_stats = mcts.get_statistics()
    print(f"\nMCTS Statistics:")
    print(f"  Tree nodes: {mcts_stats.get('tree_nodes', 'N/A')}")
    print(f"  Memory: {mcts_stats.get('tree_memory_mb', 0):.1f} MB")
    
    # Verify policy is valid
    assert policy.shape[0] == 225, f"Expected policy size 225, got {policy.shape[0]}"
    assert abs(policy.sum() - 1.0) < 1e-5, f"Policy doesn't sum to 1: {policy.sum()}"
    
    # Show top moves
    top_moves = np.argsort(policy)[-5:][::-1]
    print(f"\nTop 5 moves:")
    for i, action in enumerate(top_moves):
        row = action // 15
        col = action % 15
        prob = policy[action]
        print(f"  {i+1}. ({row:2}, {col:2}): {prob:.4f}")
    
    # Performance assessment
    overhead_target = 2.0  # Target < 2x overhead
    actual_overhead = stats.get('avg_overhead', 1.0) if hasattr(mcts.quantum_features, 'stats') else 1.0
    
    if sims_per_sec >= 20000:  # Reasonable target with quantum overhead
        performance_status = "✓ EXCELLENT"
    elif sims_per_sec >= 10000:
        performance_status = "✓ Good"
    else:
        performance_status = "⚠ Below target"
    
    if actual_overhead <= overhead_target:
        overhead_status = "✓ Within target"
    else:
        overhead_status = f"⚠ Above target ({actual_overhead:.1f}x)"
    
    print(f"\nResults:")
    print(f"  Performance: {performance_status} ({sims_per_sec:,.0f} sims/s)")
    print(f"  Quantum overhead: {overhead_status}")
    
    # Success criteria
    success = (
        sims_per_sec >= 10000 and 
        actual_overhead <= overhead_target and
        mcts.quantum_features is not None
    )
    
    if success:
        print("\n✓ SUCCESS: Quantum MCTS integration working correctly!")
    else:
        print("\n❌ FAILED: Quantum MCTS integration issues detected.")
    
    print("=" * 70)
    return success


if __name__ == "__main__":
    try:
        success = test_quantum_integration()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)