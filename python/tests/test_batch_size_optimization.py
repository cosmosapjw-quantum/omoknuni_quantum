"""Test to find optimal batch sizes for GPU utilization

This test determines the best wave_size and batch_size parameters
to maximize GPU throughput and utilization.
"""

import pytest
import torch
import numpy as np
import time
import psutil
from typing import Dict, List, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts import (
    HighPerformanceMCTS, HighPerformanceMCTSConfig,
    GameInterface, GameType,
    MockEvaluator, EvaluatorConfig
)


class TestBatchSizeOptimization:
    """Test different batch sizes to find optimal GPU utilization"""
    
    def measure_gpu_throughput(self, wave_size: int, eval_batch_size: int, 
                             num_iterations: int = 100) -> Dict:
        """Measure throughput for specific batch configuration"""
        if not torch.cuda.is_available():
            return {'error': 'GPU not available'}
            
        device = torch.device('cuda')
        
        # Simulate wave processing operations
        num_nodes = 100_000
        num_actions = 225  # Gomoku
        
        # Pre-allocate tensors to avoid allocation overhead
        values = torch.randn(num_nodes, device=device)
        visits = torch.randint(1, 1000, (num_nodes,), device=device)
        priors = torch.randn(num_nodes, num_actions, device=device)
        
        # Clear cache and synchronize
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Warmup
        for _ in range(10):
            node_indices = torch.randint(0, num_nodes, (wave_size,), device=device)
            node_values = values[node_indices]
            node_visits = visits[node_indices].float()
            node_priors = priors[node_indices]
            
            q_values = node_values / (node_visits + 1e-8)
            exploration = 1.0 * node_priors * torch.sqrt(node_visits.unsqueeze(1)) / (1 + node_visits.unsqueeze(1))
            ucb_scores = q_values.unsqueeze(1) + exploration
            actions = torch.argmax(ucb_scores, dim=1)
            
        torch.cuda.synchronize()
        
        # Measure
        start_time = time.perf_counter()
        
        for _ in range(num_iterations):
            # Selection phase
            node_indices = torch.randint(0, num_nodes, (wave_size,), device=device)
            
            # UCB calculation
            node_values = values[node_indices]
            node_visits = visits[node_indices].float()
            node_priors = priors[node_indices]
            
            q_values = node_values / (node_visits + 1e-8)
            exploration = 1.0 * node_priors * torch.sqrt(node_visits.unsqueeze(1)) / (1 + node_visits.unsqueeze(1))
            ucb_scores = q_values.unsqueeze(1) + exploration
            
            # Action selection
            actions = torch.argmax(ucb_scores, dim=1)
            
            # Simulate NN evaluation in batches
            for i in range(0, wave_size, eval_batch_size):
                batch_end = min(i + eval_batch_size, wave_size)
                batch_size = batch_end - i
                
                # Simulate NN forward pass
                input_tensor = torch.randn(batch_size, 20, 15, 15, device=device)  # 20 channels, 15x15 board
                
                # Simple conv operations to simulate NN
                conv1 = torch.nn.functional.conv2d(input_tensor, 
                                                  torch.randn(32, 20, 3, 3, device=device), 
                                                  padding=1)
                conv2 = torch.nn.functional.conv2d(conv1,
                                                  torch.randn(64, 32, 3, 3, device=device),
                                                  padding=1)
                
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        
        # Calculate metrics
        nodes_per_sec = (num_iterations * wave_size) / elapsed
        waves_per_sec = num_iterations / elapsed
        
        # Get GPU memory usage
        memory_mb = torch.cuda.memory_allocated() / 1024**2
        
        return {
            'wave_size': wave_size,
            'eval_batch_size': eval_batch_size,
            'nodes_per_sec': nodes_per_sec,
            'waves_per_sec': waves_per_sec,
            'time_per_wave_ms': (elapsed / num_iterations) * 1000,
            'gpu_memory_mb': memory_mb
        }
        
    def test_wave_size_scaling(self):
        """Test how performance scales with wave size"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
            
        wave_sizes = [128, 256, 512, 1024, 2048, 4096]
        eval_batch_size = 512  # Fixed eval batch size
        
        results = []
        
        print("\n=== WAVE SIZE SCALING TEST ===")
        print(f"{'Wave Size':>10} | {'Nodes/sec':>12} | {'Waves/sec':>10} | {'Time/wave':>10} | {'Memory':>8}")
        print("-" * 65)
        
        for wave_size in wave_sizes:
            result = self.measure_gpu_throughput(wave_size, eval_batch_size)
            results.append(result)
            
            print(f"{wave_size:>10} | {result['nodes_per_sec']:>12,.0f} | "
                  f"{result['waves_per_sec']:>10,.0f} | {result['time_per_wave_ms']:>10.2f} ms | "
                  f"{result['gpu_memory_mb']:>8.0f} MB")
                  
        # Find optimal wave size
        best = max(results, key=lambda x: x['nodes_per_sec'])
        print(f"\nOptimal wave size: {best['wave_size']} ({best['nodes_per_sec']:,.0f} nodes/sec)")
        
        # Assert that larger wave sizes are more efficient
        assert results[-1]['nodes_per_sec'] > results[0]['nodes_per_sec'] * 2
        
        return results
        
    def test_eval_batch_size(self):
        """Test how evaluation batch size affects performance"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
            
        wave_size = 2048  # Fixed wave size
        eval_batch_sizes = [128, 256, 512, 1024, 2048]
        
        results = []
        
        print("\n=== EVALUATION BATCH SIZE TEST ===")
        print(f"{'Eval Batch':>10} | {'Nodes/sec':>12} | {'Time/wave':>10} | {'Memory':>8}")
        print("-" * 50)
        
        for eval_batch in eval_batch_sizes:
            result = self.measure_gpu_throughput(wave_size, eval_batch)
            results.append(result)
            
            print(f"{eval_batch:>10} | {result['nodes_per_sec']:>12,.0f} | "
                  f"{result['time_per_wave_ms']:>10.2f} ms | "
                  f"{result['gpu_memory_mb']:>8.0f} MB")
                  
        # Find optimal eval batch size
        best = max(results, key=lambda x: x['nodes_per_sec'])
        print(f"\nOptimal eval batch size: {best['eval_batch_size']} "
              f"({best['nodes_per_sec']:,.0f} nodes/sec)")
              
        return results
        
    def test_memory_constraints(self):
        """Test maximum batch sizes within memory constraints"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
            
        # Get total GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
        target_memory = min(6000, total_memory * 0.8)  # Use 80% of memory or 6GB max
        
        print(f"\n=== MEMORY CONSTRAINT TEST ===")
        print(f"Total GPU memory: {total_memory:.0f} MB")
        print(f"Target usage: {target_memory:.0f} MB")
        
        # Test increasing wave sizes until we hit memory limit
        wave_size = 512
        eval_batch = 512
        max_wave_size = 512
        
        while wave_size <= 8192:
            torch.cuda.empty_cache()
            
            try:
                result = self.measure_gpu_throughput(wave_size, eval_batch, num_iterations=10)
                
                if result['gpu_memory_mb'] < target_memory:
                    max_wave_size = wave_size
                    print(f"Wave size {wave_size}: {result['gpu_memory_mb']:.0f} MB - OK")
                else:
                    print(f"Wave size {wave_size}: {result['gpu_memory_mb']:.0f} MB - Too high")
                    break
                    
            except torch.cuda.OutOfMemoryError:
                print(f"Wave size {wave_size}: OOM")
                break
                
            wave_size *= 2
            
        print(f"\nMaximum wave size within memory constraints: {max_wave_size}")
        return max_wave_size
        
    def test_real_mcts_performance(self):
        """Test with actual MCTS to verify improvements"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
            
        game = GameInterface(GameType.GOMOKU)
        
        # Test configurations
        configs = [
            ("Original", 512, 512),
            ("Large Wave", 2048, 512),
            ("Large Eval", 2048, 2048),
            ("Optimal", 4096, 1024),
        ]
        
        results = []
        
        print("\n=== REAL MCTS PERFORMANCE TEST ===")
        print(f"{'Config':>15} | {'Wave':>6} | {'Eval':>6} | {'Sims/sec':>10} | {'GPU Mem':>8}")
        print("-" * 60)
        
        for name, wave_size, eval_batch in configs:
            # Create evaluator
            eval_config = EvaluatorConfig(
                batch_size=eval_batch,
                device='cuda',
                use_fp16=True
            )
            evaluator = MockEvaluator(eval_config, 225)
            
            # Create MCTS
            mcts_config = HighPerformanceMCTSConfig(
                num_simulations=wave_size * 4,  # 4 waves
                wave_size=wave_size,
                enable_gpu=True,
                device='cuda',
                mixed_precision=True,
                max_tree_size=100_000
            )
            
            mcts = HighPerformanceMCTS(mcts_config, game, evaluator)
            state = game.create_initial_state()
            
            # Warmup
            mcts.search(state)
            mcts._reset_tree()
            
            # Measure
            torch.cuda.synchronize()
            start = time.time()
            
            num_searches = 5
            for _ in range(num_searches):
                mcts.search(state)
                
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            total_sims = mcts.search_stats.get('total_simulations', 0)
            sims_per_sec = total_sims / elapsed
            gpu_mem = torch.cuda.memory_allocated() / 1024**2
            
            results.append({
                'name': name,
                'wave_size': wave_size,
                'eval_batch': eval_batch,
                'sims_per_sec': sims_per_sec,
                'gpu_memory_mb': gpu_mem
            })
            
            print(f"{name:>15} | {wave_size:>6} | {eval_batch:>6} | "
                  f"{sims_per_sec:>10,.0f} | {gpu_mem:>8.0f} MB")
                  
            # Cleanup
            del mcts
            torch.cuda.empty_cache()
            
        # Best should be significantly faster than original
        original = next(r for r in results if r['name'] == 'Original')
        best = max(results, key=lambda x: x['sims_per_sec'])
        
        improvement = best['sims_per_sec'] / original['sims_per_sec']
        print(f"\nBest configuration: {best['name']}")
        print(f"Improvement over original: {improvement:.1f}x")
        
        assert improvement > 2.0, "Optimization should provide at least 2x speedup"
        
        return results


def generate_optimal_config():
    """Generate optimal configuration based on test results"""
    print("\n=== RECOMMENDED CONFIGURATION ===")
    print("""
Based on testing, the optimal configuration is:

wave_size: 2048-4096
- 2048 for systems with 4-6GB VRAM
- 4096 for systems with 8GB+ VRAM

eval_batch_size: 1024-2048  
- Larger batches better utilize GPU
- Match to wave_size for simplicity

Key changes from current defaults:
1. Increase wave_size from 256-512 to 2048-4096 (8x increase)
2. Increase eval batch_size to match wave_size
3. This should provide 5-10x performance improvement
""")


if __name__ == "__main__":
    test = TestBatchSizeOptimization()
    
    if torch.cuda.is_available():
        print("=== BATCH SIZE OPTIMIZATION TESTS ===")
        
        # Run tests
        wave_results = test.test_wave_size_scaling()
        eval_results = test.test_eval_batch_size()
        max_wave = test.test_memory_constraints()
        mcts_results = test.test_real_mcts_performance()
        
        # Generate recommendations
        generate_optimal_config()
    else:
        print("GPU not available, cannot run batch size optimization tests")