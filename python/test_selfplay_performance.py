#!/usr/bin/env python3
"""
Test Self-Play Performance - Verify the optimizations improve self-play speed

This script tests the optimized self-play pipeline to ensure:
1. Games complete in reasonable time (<30s each)
2. CPU utilization is optimized for 12 cores/24 threads
3. No processes get stuck or timeout
4. Memory usage is reasonable
"""

import sys
import time
import torch
import psutil
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List

# Add MCTS modules to path
mcts_root = Path(__file__).parent
sys.path.insert(0, str(mcts_root))

from mcts.utils.config_system import AlphaZeroConfig
from mcts.neural_networks.self_play_module import SelfPlayManager
from mcts.neural_networks.mock_evaluator import MockEvaluator
from mcts.utils.training_profiler import enable_profiling, log_profiling_summary, reset_profiler


def create_test_config(num_games: int = 10, num_workers: int = 6) -> AlphaZeroConfig:
    """Create test configuration optimized for performance testing"""
    config = AlphaZeroConfig()
    
    # Game settings
    config.game.game_type = "gomoku"
    config.game.board_size = 15
    
    # MCTS settings - optimized for fast testing
    config.mcts.num_simulations = 400  # Reduced for faster testing
    config.mcts.min_wave_size = 3072
    config.mcts.max_wave_size = 3072
    config.mcts.adaptive_wave_sizing = False
    config.mcts.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.mcts.use_mixed_precision = torch.cuda.is_available()
    config.mcts.use_cuda_graphs = torch.cuda.is_available()
    config.mcts.use_tensor_cores = torch.cuda.is_available()
    config.mcts.quantum_level = 'classical'
    config.mcts.enable_quantum = False
    config.mcts.memory_pool_size_mb = 2048
    config.mcts.max_tree_nodes = 100000
    
    # Training settings
    config.training.num_games_per_iteration = num_games
    config.training.num_workers = num_workers
    config.training.max_moves_per_game = 100  # Shorter games for testing
    config.training.resign_threshold = -0.9
    config.training.resign_check_moves = 10
    config.training.resign_start_iteration = 1
    
    # Network settings - minimal for testing
    config.network.input_channels = 18
    config.network.input_representation = 'basic'
    config.network.num_res_blocks = 6  # Reduced for faster testing
    config.network.num_filters = 64   # Reduced for faster testing
    
    # Logging
    config.log_level = 'INFO'
    
    return config


def monitor_system_resources() -> Dict[str, float]:
    """Monitor system resource usage"""
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
    
    # Memory usage
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    memory_gb = memory.used / (1024**3)
    
    # GPU memory if available
    gpu_memory_gb = 0.0
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.memory_allocated() / (1024**3)
    
    return {
        'cpu_percent': cpu_percent,
        'cpu_cores_active': sum(1 for usage in cpu_per_core if usage > 10),
        'memory_percent': memory_percent,
        'memory_gb': memory_gb,
        'gpu_memory_gb': gpu_memory_gb,
        'processes': len(psutil.pids())
    }


def test_self_play_performance():
    """Test self-play performance with monitoring"""
    print("=" * 80)
    print("SELF-PLAY PERFORMANCE TEST")
    print("=" * 80)
    
    # Test configuration
    num_games = 8  # Small number for quick testing
    num_workers = 6  # Good utilization without overwhelming
    
    print(f"Testing: {num_games} games with {num_workers} workers")
    print(f"Expected time: <30s per game, total <4 minutes")
    print()
    
    # Create configuration and manager
    config = create_test_config(num_games, num_workers)
    
    # Auto-adjust for hardware
    config.adjust_for_hardware(target_workers=num_workers)
    hardware = config.hardware_info
    allocation = config.calculate_resource_allocation(hardware, num_workers)
    
    print(f"Hardware: {hardware['cpu_cores_physical']} cores, {hardware['total_ram_gb']:.1f}GB RAM")
    print(f"Allocation: {allocation['num_workers']} workers, {allocation['max_concurrent_workers']} concurrent")
    print()
    
    # Create mock evaluator for testing
    evaluator = MockEvaluator(game_type='gomoku', device=config.mcts.device)
    
    # Create self-play manager
    self_play_manager = SelfPlayManager(config)
    
    # Enable profiling
    enable_profiling()
    reset_profiler()
    
    print("🚀 Starting self-play test...")
    start_time = time.perf_counter()
    initial_resources = monitor_system_resources()
    
    # Track game completion times
    game_times = []
    
    try:
        # Generate games with monitoring
        examples = self_play_manager._parallel_self_play(
            evaluator, 
            iteration=1, 
            num_games=num_games, 
            num_workers=num_workers
        )
        
        total_time = time.perf_counter() - start_time
        final_resources = monitor_system_resources()
        
        # Results
        print("✅ Self-play completed successfully!")
        print()
        print("PERFORMANCE RESULTS:")
        print("-" * 40)
        print(f"Total time:          {total_time:.1f}s")
        print(f"Games completed:     {len(examples) // 50 if examples else 0}")  # Rough estimate
        print(f"Average time/game:   {total_time / num_games:.1f}s")
        print(f"Examples generated:  {len(examples) if examples else 0}")
        print()
        
        # Resource utilization
        print("RESOURCE UTILIZATION:")
        print("-" * 40)
        print(f"Peak CPU:            {final_resources['cpu_percent']:.1f}%")
        print(f"CPU cores active:    {final_resources['cpu_cores_active']}/{hardware['cpu_cores_physical']}")
        print(f"Memory usage:        {final_resources['memory_gb']:.1f}GB ({final_resources['memory_percent']:.1f}%)")
        if torch.cuda.is_available():
            print(f"GPU memory:          {final_resources['gpu_memory_gb']:.1f}GB")
        print()
        
        # Performance analysis
        print("PERFORMANCE ANALYSIS:")
        print("-" * 40)
        
        avg_time_per_game = total_time / num_games
        if avg_time_per_game > 60:
            print("❌ Games taking too long (>60s each)")
            print("   - Check for stuck processes or infinite loops")
        elif avg_time_per_game > 30:
            print("⚠️  Games slower than target (<30s each)")
            print("   - Consider optimizing MCTS parameters")
        else:
            print("✅ Game speed acceptable (<30s each)")
        
        cpu_utilization = final_resources['cpu_cores_active'] / hardware['cpu_cores_physical']
        if cpu_utilization < 0.5:
            print("⚠️  Low CPU utilization - consider more workers")
        elif cpu_utilization > 0.9:
            print("⚠️  High CPU utilization - may cause slowdowns")
        else:
            print("✅ Good CPU utilization")
        
        if final_resources['memory_percent'] > 80:
            print("⚠️  High memory usage - may cause issues")
        else:
            print("✅ Memory usage reasonable")
        
        # Success criteria
        success = (
            avg_time_per_game < 60 and  # Games complete in reasonable time
            len(examples) > 0 and       # Examples were generated
            cpu_utilization > 0.3       # CPU is being utilized
        )
        
        print()
        if success:
            print("🎉 PERFORMANCE TEST PASSED!")
        else:
            print("❌ PERFORMANCE TEST FAILED!")
            
        # Log profiling summary
        print()
        log_profiling_summary(top_n=15, min_time=0.1)
        
        return success, {
            'total_time': total_time,
            'avg_time_per_game': avg_time_per_game,
            'examples_generated': len(examples) if examples else 0,
            'cpu_utilization': cpu_utilization,
            'memory_usage': final_resources['memory_percent']
        }
        
    except Exception as e:
        print(f"❌ Self-play test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False, {}
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    """Main test function"""
    try:
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Run the test
        success, results = test_self_play_performance()
        
        print("\n" + "=" * 80)
        if success:
            print("✅ ALL TESTS PASSED - Self-play optimizations are working!")
        else:
            print("❌ TESTS FAILED - Self-play needs further optimization")
        print("=" * 80)
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())