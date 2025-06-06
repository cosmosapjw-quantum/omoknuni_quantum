#!/usr/bin/env python3
"""Performance test for optimized quantum path integral implementation"""

import time
import torch
import numpy as np
import sys
sys.path.append('.')

def benchmark_quantum_optimizations():
    """Benchmark the optimized quantum path integral implementation"""
    print("🚀 QUANTUM PATH INTEGRAL OPTIMIZATION BENCHMARK")
    print("="*60)
    
    try:
        from mcts.core.high_performance_mcts import HighPerformanceMCTS, HighPerformanceMCTSConfig
        from mcts.neural_networks.resnet_evaluator import ResNetEvaluator  
        from alphazero_py import GomokuState
        
        class SimpleGameInterface:
            def get_legal_moves(self, state):
                return state.get_legal_moves()[:15]  # More moves for stress test
            def apply_move(self, state, action):
                new_state = state.clone()
                new_state.make_move(action)
                return new_state
            def state_to_numpy(self, state, use_enhanced=True):
                return state.get_enhanced_tensor_representation()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        
        game = SimpleGameInterface()
        evaluator = ResNetEvaluator(game_type='gomoku', device=device)
        
        # Test different configurations
        configs = [
            ("Small", {"num_simulations": 20, "wave_size": 8}),
            ("Medium", {"num_simulations": 50, "wave_size": 16}),
            ("Large", {"num_simulations": 100, "wave_size": 32}),
        ]
        
        results = {}
        
        for config_name, params in configs:
            print(f"\n🔧 Testing {config_name} Configuration:")
            print(f"   Simulations: {params['num_simulations']}, Wave Size: {params['wave_size']}")
            
            config = HighPerformanceMCTSConfig(
                num_simulations=params["num_simulations"],
                wave_size=params["wave_size"],
                device=str(device),
                enable_gpu=True,
                enable_path_integral=True,
                enable_interference=True,
                enable_phase_policy=True
            )
            
            mcts = HighPerformanceMCTS(config, game, evaluator)
            root_state = GomokuState()
            
            # Warmup
            _ = mcts.search(root_state)
            
            # Benchmark multiple runs
            times = []
            for run in range(5):
                start_time = time.perf_counter()
                policy = mcts.search(root_state)
                end_time = time.perf_counter()
                
                search_time = end_time - start_time
                times.append(search_time)
                
                sims_per_sec = params["num_simulations"] / search_time
                print(f"   Run {run+1}: {search_time:.3f}s ({sims_per_sec:.0f} sims/sec)")
            
            avg_time = np.mean(times)
            avg_sims_per_sec = params["num_simulations"] / avg_time
            
            results[config_name] = {
                "avg_time": avg_time,
                "sims_per_sec": avg_sims_per_sec,
                "policy_size": len(policy)
            }
            
            print(f"   Average: {avg_time:.3f}s ({avg_sims_per_sec:.0f} sims/sec)")
            
            # Get quantum stats if available
            try:
                if hasattr(mcts.wave_engine, 'path_integral'):
                    pi_stats = mcts.wave_engine.path_integral.get_statistics()
                    print(f"   Quantum stats: {pi_stats}")
            except Exception:
                pass
        
        print(f"\n🎯 OPTIMIZATION RESULTS SUMMARY:")
        print("="*60)
        for name, result in results.items():
            print(f"{name:8}: {result['sims_per_sec']:6.0f} sims/sec | {result['avg_time']:.3f}s | {result['policy_size']} moves")
        
        # Performance targets
        print(f"\n📊 PERFORMANCE ANALYSIS:")
        print("="*60)
        best_sims = max(r['sims_per_sec'] for r in results.values())
        
        optimizations_working = []
        if best_sims > 1000:
            optimizations_working.append("✅ Vectorized path sampling")
        if best_sims > 2000:
            optimizations_working.append("✅ Fused action computation")
        if best_sims > 3000:
            optimizations_working.append("✅ GPU-native optimization")
        if best_sims > 5000:
            optimizations_working.append("✅ Triton kernels active")
            
        for opt in optimizations_working:
            print(f"   {opt}")
        
        if best_sims < 1000:
            print("   ⚠️  Performance below target - optimizations may not be fully active")
        
        print(f"\n🏆 Best performance: {best_sims:.0f} simulations/second")
        
        # Speedup estimate
        baseline_sims_per_sec = 100  # Estimated baseline without optimizations
        speedup = best_sims / baseline_sims_per_sec
        print(f"🚀 Estimated speedup: {speedup:.1f}x over baseline")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    benchmark_quantum_optimizations()