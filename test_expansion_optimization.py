#!/usr/bin/env python3
"""Test the expansion phase optimization to achieve 3000+ sims/sec target"""

import torch
import numpy as np
import time
import sys
import os

# Set up library path
lib_path = os.path.join(os.getcwd(), 'python')
sys.path.insert(0, lib_path)

from mcts.core import MCTS, MCTSConfig
from mcts.neural_networks.resnet_model import ResNetModel, ResNetConfig
from mcts.utils.single_gpu_evaluator import SingleGPUEvaluator


def test_expansion_optimization():
    """Test the optimized expansion phase performance"""
    print("Testing Expansion Phase Optimization")
    print("="*70)
    
    print("GOAL: Reduce expansion time from 152ms to ~100ms")
    print("Expected result: 2438 → 2800-3000+ sims/sec")
    
    # Configuration with optimizations
    config = MCTSConfig(
        device='cuda',
        num_simulations=1000,
        batch_size=2048,
        wave_size=4096,
        wave_async_expansion=True,
        enable_fast_ucb=True,
        use_mixed_precision=True,
        enable_multi_stream=True,
        num_cuda_streams=4
    )
    
    # Model
    nn_config = ResNetConfig(num_blocks=10, num_filters=128)
    model = ResNetModel(
        config=nn_config,
        board_size=15,
        num_actions=225,
        game_type='gomoku'
    ).cuda().eval()
    
    evaluator = SingleGPUEvaluator(model, device='cuda', batch_size=2048)
    evaluator._return_torch_tensors = True
    
    # Create MCTS
    mcts = MCTS(config, evaluator)
    state = np.zeros((15, 15), dtype=np.int8)
    
    print("\n1. Warmup...")
    for _ in range(3):
        _ = mcts.search(state, num_simulations=200)
    
    print("\n2. Baseline test (1000 simulations)...")
    torch.cuda.synchronize()
    start = time.time()
    
    policy = mcts.search(state, num_simulations=1000)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    throughput = 1000 / elapsed
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {throughput:.0f} sims/sec")
    
    if throughput >= 3000:
        print(f"\n  ✅ SUCCESS! Achieved {throughput:.0f} sims/sec!")
        print("  🎯 TARGET REACHED!")
    elif throughput >= 2800:
        print(f"\n  🎯 CLOSE! {throughput:.0f} sims/sec (need {3000-throughput:.0f} more)")
        print("  Expansion optimization worked!")
    else:
        print(f"\n  📈 Progress: {throughput:.0f} sims/sec (from 2438 baseline)")
        improvement = (throughput / 2438 - 1) * 100
        print(f"  Improvement: {improvement:+.1f}%")
    
    # Extended test for stability
    print("\n3. Extended stability test (3 runs)...")
    results = []
    
    for i in range(3):
        print(f"  Run {i+1}/3...", end=" ", flush=True)
        
        torch.cuda.synchronize()
        start = time.time()
        
        _ = mcts.search(state, num_simulations=1000)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        throughput = 1000 / elapsed
        results.append(throughput)
        print(f"{throughput:.0f} sims/sec")
    
    avg_throughput = np.mean(results)
    std_throughput = np.std(results)
    
    print(f"\n  Average: {avg_throughput:.0f} ± {std_throughput:.0f} sims/sec")
    
    if avg_throughput >= 3000:
        print(f"\n  🏆 CONSISTENT SUCCESS! Average {avg_throughput:.0f} sims/sec!")
    
    # Memory usage check
    mem_allocated = torch.cuda.memory_allocated() / 1024**3
    mem_reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"\n  VRAM: {mem_allocated:.1f}GB allocated, {mem_reserved:.1f}GB reserved")
    
    return avg_throughput


def analyze_remaining_bottlenecks():
    """Analyze what bottlenecks remain if we don't hit 3000+"""
    print("\n\nBottleneck Analysis")
    print("="*70)
    
    print("If we still don't reach 3000+ sims/sec, the remaining bottlenecks are:")
    print("1. Tree operations (add_children_batch) - need true vectorization")
    print("2. Memory transfers - need to eliminate more .cpu() calls")
    print("3. Sequential dependencies - need multi-tree approach")
    print("4. Selection phase overhead - need GPU-based selection")
    
    print("\nNext optimization priorities:")
    print("A. Implement vectorized tree operations in CSR format")
    print("B. Move more operations to GPU (selection, UCB)")
    print("C. Consider multi-tree parallelism")
    print("D. Profile with nvidia-smi during execution")


def suggest_final_steps():
    """Suggest final steps to reach 3000+ if needed"""
    print("\n\nFinal Optimization Suggestions")
    print("="*70)
    
    print("If expansion optimization alone doesn't reach 3000+:")
    
    print("\n1. VECTORIZED TREE OPERATIONS:")
    print("   - Implement true batch add_children in C++/CUDA")
    print("   - Expected improvement: 30-50%")
    
    print("\n2. GPU-BASED SELECTION:")
    print("   - Move UCB computation to GPU")
    print("   - Eliminate CPU tree traversal")
    print("   - Expected improvement: 20-30%")
    
    print("\n3. MULTI-TREE APPROACH (Alternative):")
    print("   - Run 2-3 MCTS trees per position")
    print("   - Guaranteed 3000+ total throughput")
    print("   - May improve playing strength")
    
    print("\n4. HARDWARE UTILIZATION:")
    print("   - Current: 20% GPU, 70% VRAM")
    print("   - Target: 60%+ GPU, 80%+ VRAM")


def main():
    """Main test entry point"""
    print("Expansion Phase Optimization Test")
    print("="*70)
    
    print("Testing the quick expansion optimization patch...")
    print("This should reduce expansion time from 152ms to ~100ms")
    print("Target: 2438 → 3000+ sims/sec\n")
    
    try:
        throughput = test_expansion_optimization()
        
        if throughput >= 3000:
            print("\n🎉 MISSION ACCOMPLISHED! 3000+ sims/sec achieved!")
        else:
            analyze_remaining_bottlenecks()
            suggest_final_steps()
            
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()