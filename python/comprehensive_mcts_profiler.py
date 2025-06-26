#!/usr/bin/env python3
"""
Comprehensive MCTS Profiler - Instruments ALL major methods to find hidden bottlenecks

This tool monkey-patches ALL major MCTS methods to track timing and identify
where the missing 99.7% of time is being spent.
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add MCTS modules to path
mcts_root = Path(__file__).parent
sys.path.insert(0, str(mcts_root))

# Global timing tracking
method_calls = defaultdict(int)
method_times = defaultdict(float)


def track_method(original_method, method_name):
    """Create a tracked version of a method"""
    def tracked_method(*args, **kwargs):
        global method_calls, method_times
        method_calls[method_name] += 1
        start_time = time.perf_counter()
        result = original_method(*args, **kwargs)
        method_times[method_name] += time.perf_counter() - start_time
        return result
    return tracked_method


def monkey_patch_all_methods():
    """Monkey patch ALL major MCTS methods for comprehensive tracking"""
    global method_calls, method_times
    
    # Reset tracking
    method_calls.clear()
    method_times.clear()
    
    # Import after path is set
    from mcts.core.mcts import MCTS
    from mcts.gpu.csr_tree import CSRTree
    from mcts.gpu.gpu_game_states import GPUGameStates
    from mcts.neural_networks.mock_evaluator import MockEvaluator
    
    print("🔧 Monkey-patching MCTS methods...")
    
    # Core MCTS methods
    MCTS._search_optimized = track_method(MCTS._search_optimized, "MCTS._search_optimized")
    MCTS._run_search_wave_vectorized = track_method(MCTS._run_search_wave_vectorized, "MCTS._run_search_wave_vectorized")
    MCTS._select_batch_vectorized = track_method(MCTS._select_batch_vectorized, "MCTS._select_batch_vectorized")
    MCTS._expand_batch_vectorized = track_method(MCTS._expand_batch_vectorized, "MCTS._expand_batch_vectorized")
    MCTS._evaluate_batch_vectorized = track_method(MCTS._evaluate_batch_vectorized, "MCTS._evaluate_batch_vectorized")
    MCTS._backup_batch_vectorized = track_method(MCTS._backup_batch_vectorized, "MCTS._backup_batch_vectorized")
    MCTS._initialize_root = track_method(MCTS._initialize_root, "MCTS._initialize_root")
    MCTS._extract_policy = track_method(MCTS._extract_policy, "MCTS._extract_policy")
    MCTS._allocate_states = track_method(MCTS._allocate_states, "MCTS._allocate_states")
    
    print("🔧 Monkey-patching CSR tree methods...")
    
    # CSR Tree methods
    CSRTree.get_children = track_method(CSRTree.get_children, "CSRTree.get_children")
    CSRTree.batch_get_children = track_method(CSRTree.batch_get_children, "CSRTree.batch_get_children")
    CSRTree.batch_select_ucb_optimized = track_method(CSRTree.batch_select_ucb_optimized, "CSRTree.batch_select_ucb_optimized")
    CSRTree.add_children_batch_gpu = track_method(CSRTree.add_children_batch_gpu, "CSRTree.add_children_batch_gpu")
    CSRTree.set_expanded = track_method(CSRTree.set_expanded, "CSRTree.set_expanded")
    
    print("🔧 Monkey-patching GPU game state methods...")
    
    # GPU Game States methods
    GPUGameStates.apply_moves = track_method(GPUGameStates.apply_moves, "GPUGameStates.apply_moves")
    GPUGameStates.get_legal_moves_mask = track_method(GPUGameStates.get_legal_moves_mask, "GPUGameStates.get_legal_moves_mask")
    GPUGameStates.get_nn_features = track_method(GPUGameStates.get_nn_features, "GPUGameStates.get_nn_features")
    
    print("🔧 Monkey-patching evaluator methods...")
    
    # Mock Evaluator methods
    MockEvaluator.evaluate_batch = track_method(MockEvaluator.evaluate_batch, "MockEvaluator.evaluate_batch")
    MockEvaluator.evaluate = track_method(MockEvaluator.evaluate, "MockEvaluator.evaluate")
    
    print("✅ All methods monkey-patched for comprehensive tracking")


def run_comprehensive_profiling():
    """Run comprehensive MCTS profiling"""
    
    from mcts.core.mcts import MCTS, MCTSConfig
    from mcts.gpu.gpu_game_states import GameType as GPUGameType
    from mcts.neural_networks.mock_evaluator import MockEvaluator
    import alphazero_py
    
    print("🔧 Creating MCTS for comprehensive profiling...")
    
    config = MCTSConfig(
        num_simulations=500,
        wave_size=64,
        min_wave_size=64,
        max_wave_size=64,
        device='cuda',
        game_type=GPUGameType.GOMOKU,
        board_size=15,
        enable_quantum=False,
        max_tree_nodes=5000,
        use_mixed_precision=False,
        use_cuda_graphs=False,
        adaptive_wave_sizing=False,
        enable_debug_logging=False,
        profile_gpu_kernels=False,
    )
    
    evaluator = MockEvaluator(game_type='gomoku', device='cuda')
    mcts = MCTS(config, evaluator)
    game_state = alphazero_py.GomokuState()
    
    print("🚀 Running MCTS search with comprehensive tracking...")
    
    # Time the search
    start_time = time.perf_counter()
    policy = mcts.search(game_state, 500)
    total_time = time.perf_counter() - start_time
    
    return total_time, 500 / total_time


def print_comprehensive_results(total_time, sims_per_sec):
    """Print comprehensive profiling results"""
    global method_calls, method_times
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MCTS PROFILING RESULTS")
    print("=" * 80)
    print(f"Total search time: {total_time:.3f}s")
    print(f"Performance: {sims_per_sec:.0f} sims/s")
    print()
    
    # Sort methods by total time
    sorted_methods = sorted(method_times.items(), key=lambda x: x[1], reverse=True)
    
    print("TOP METHODS BY TOTAL TIME:")
    print("-" * 80)
    print(f"{'Method':<50} {'Calls':<8} {'Total (s)':<10} {'Avg (ms)':<10} {'% Time':<8}")
    print("-" * 80)
    
    tracked_time = 0
    for method_name, method_time in sorted_methods[:20]:  # Top 20
        calls = method_calls[method_name]
        avg_time_ms = 1000 * method_time / calls if calls > 0 else 0
        percent = 100 * method_time / total_time
        tracked_time += method_time
        
        print(f"{method_name:<50} {calls:<8} {method_time:<10.3f} {avg_time_ms:<10.2f} {percent:<8.1f}")
    
    print("-" * 80)
    print(f"{'TRACKED TOTAL':<50} {'':<8} {tracked_time:<10.3f} {'':<10} {100*tracked_time/total_time:<8.1f}")
    untracked_time = total_time - tracked_time
    print(f"{'UNTRACKED TIME':<50} {'':<8} {untracked_time:<10.3f} {'':<10} {100*untracked_time/total_time:<8.1f}")
    
    print("\n" + "=" * 80)
    
    if untracked_time > total_time * 0.5:  # More than 50% untracked
        print("❌ MAJORITY OF TIME IS UNTRACKED!")
        print("   - The bottleneck is in methods we haven't instrumented")
        print("   - Look for:")
        print("     * torch operations (tensor.to(), tensor.item(), etc.)")
        print("     * GPU synchronization points")
        print("     * Hidden loops or expensive built-in functions")
        print("     * Memory allocation/deallocation")
    elif tracked_time > total_time * 0.8:  # Most time is tracked
        print("✅ Most time is tracked - we can identify the bottleneck")
        if sorted_methods:
            top_method = sorted_methods[0]
            print(f"🎯 Main bottleneck: {top_method[0]} ({100*top_method[1]/total_time:.1f}% of time)")
    else:
        print("⚠️  Some time is untracked - partial visibility")
    
    return {
        'total_time': total_time,
        'tracked_time': tracked_time,
        'untracked_time': untracked_time,
        'sims_per_sec': sims_per_sec,
        'method_times': dict(method_times),
        'method_calls': dict(method_calls)
    }


def main():
    """Main comprehensive profiling function"""
    print("=" * 80)
    print("COMPREHENSIVE MCTS PROFILER")
    print("=" * 80)
    
    try:
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Apply comprehensive monkey patches
        monkey_patch_all_methods()
        
        # Run profiling
        total_time, sims_per_sec = run_comprehensive_profiling()
        
        # Print results
        results = print_comprehensive_results(total_time, sims_per_sec)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        
        if results['untracked_time'] > 1.0:
            print("🎯 Focus on finding untracked expensive operations")
        elif results['sims_per_sec'] > 1000:
            print("🎯 Performance is acceptable")
        else:
            print("🎯 Continue optimizing the tracked bottlenecks")
            
    except Exception as e:
        print(f"❌ Comprehensive profiling failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()