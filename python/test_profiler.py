#!/usr/bin/env python3
"""Test script for MCTS comprehensive profiler"""

import torch
from mcts_comprehensive_profiler import MCTSBenchmarkSuite, ProfilingConfig

def test_basic_profiling():
    """Test basic profiling functionality"""
    print("🧪 Testing MCTS Comprehensive Profiler")
    print(f"GPU Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - profiler requires GPU")
        return
    
    # Quick test configuration
    config = ProfilingConfig(
        simulation_counts=[1000, 5000],
        wave_sizes=[512, 1024],
        warmup_iterations=1,
        measurement_iterations=2,
        output_dir="profiler_test_results"
    )
    
    # Run benchmark
    benchmark = MCTSBenchmarkSuite(config)
    benchmark.run_comprehensive_benchmark()
    
    print("✅ Profiler test completed successfully!")

if __name__ == "__main__":
    test_basic_profiling()