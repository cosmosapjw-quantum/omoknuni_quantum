#!/usr/bin/env python3
"""Demo script showcasing MCTS profiler features"""

import torch
from mcts_comprehensive_profiler import (
    MCTSBenchmarkSuite, 
    ProfilingConfig, 
    InstrumentedMCTS,
    GPUProfiler,
    MemoryMonitor
)
from mcts.core.mcts import MCTSConfig
from mcts.gpu.gpu_game_states import GameType
import alphazero_py


class FastEvaluator:
    def __init__(self):
        self.device = torch.device('cuda')
    def evaluate_batch(self, features, legal_masks=None):
        b = features.shape[0]
        return torch.ones(b, 225, device=self.device)/225, torch.zeros(b, 1, device=self.device)


def demo_basic_profiling():
    """Demo 1: Basic performance profiling"""
    print("🎯 Demo 1: Basic Performance Profiling")
    print("=" * 50)
    
    config = ProfilingConfig(
        simulation_counts=[5000],
        wave_sizes=[1024, 3072],
        measurement_iterations=3,
        output_dir="demo_basic"
    )
    
    benchmark = MCTSBenchmarkSuite(config)
    benchmark.run_comprehensive_benchmark()
    print()


def demo_detailed_analysis():
    """Demo 2: Detailed phase analysis"""
    print("🔬 Demo 2: Detailed Phase Analysis")
    print("=" * 50)
    
    # Create MCTS config
    mcts_config = MCTSConfig(
        wave_size=2048,
        device='cuda',
        game_type=GameType.GOMOKU
    )
    
    # Create profiler config
    profiler_config = ProfilingConfig(memory_sample_interval=0.005)
    
    # Create instrumented MCTS
    evaluator = FastEvaluator()
    instrumented_mcts = InstrumentedMCTS(mcts_config, evaluator, profiler_config)
    
    # Run profiled search
    game_state = alphazero_py.GomokuState()
    profile = instrumented_mcts.profile_search(game_state, 10000)
    
    # Print detailed results
    print(f"Performance: {profile.simulations_per_second:,.0f} sims/s")
    print(f"Efficiency: {profile.efficiency_score:.1f} sims/s/GB")
    print(f"Peak GPU Memory: {profile.peak_gpu_memory_mb:.1f}MB")
    print(f"Bottleneck: {profile.bottleneck_phase}")
    print()
    
    if profile.phases:
        print("Phase Breakdown:")
        for phase in profile.phases:
            print(f"  {phase.name}: {phase.gpu_time_ms:.1f}ms (batch: {phase.batch_size})")
    
    print("\nRecommendations:")
    for rec in profile.recommendations[:3]:
        print(f"  • {rec}")
    print()


def demo_memory_monitoring():
    """Demo 3: Memory monitoring"""
    print("💾 Demo 3: Memory Monitoring")
    print("=" * 50)
    
    monitor = MemoryMonitor(sample_interval=0.01)
    monitor.start_monitoring()
    
    # Simulate some GPU work
    mcts_config = MCTSConfig(wave_size=1024, device='cuda')
    evaluator = FastEvaluator()
    from mcts.core.mcts import MCTS
    mcts = MCTS(mcts_config, evaluator)
    
    game_state = alphazero_py.GomokuState()
    policy = mcts.search(game_state, 2000)
    
    monitor.stop_monitoring()
    
    peak = monitor.get_peak_usage()
    avg = monitor.get_average_usage()
    
    print(f"Peak GPU Memory: {peak.gpu_allocated_mb:.1f}MB")
    print(f"Average GPU Memory: {avg.gpu_allocated_mb:.1f}MB")
    print(f"Peak CPU Usage: {peak.cpu_percent:.1f}%")
    print(f"Samples Collected: {len(monitor.snapshots)}")
    print()


def demo_gpu_timing():
    """Demo 4: GPU timing"""
    print("⏱️  Demo 4: GPU Timing")
    print("=" * 50)
    
    profiler = GPUProfiler(torch.device('cuda'))
    
    # Time some operations
    x = torch.randn(1000, 1000, device='cuda')
    
    with profiler.timer('matrix_multiply'):
        y = torch.matmul(x, x.T)
    
    with profiler.timer('element_wise'):
        z = x * x
    
    stats = profiler.get_stats()
    
    for op_name, timing in stats.items():
        print(f"{op_name}: {timing['avg_ms']:.2f}ms")
    print()


def main():
    """Run all demos"""
    print("🚀 MCTS Profiler Feature Demonstrations")
    print("=" * 60)
    print()
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - some demos may not work")
        return
    
    try:
        demo_gpu_timing()
        demo_memory_monitoring() 
        demo_detailed_analysis()
        demo_basic_profiling()
        
        print("✅ All demos completed successfully!")
        print("📁 Check output directories for detailed results")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()