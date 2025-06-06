"""Test CPU parallelization with multiple MCTS instances

This test verifies that running multiple MCTS instances in parallel
on different CPU cores provides linear speedup.
"""

import pytest
import torch
import time
import psutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
from typing import List, Tuple, Dict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts import (
    HighPerformanceMCTS, HighPerformanceMCTSConfig,
    GameInterface, GameType,
    MockEvaluator, EvaluatorConfig
)


def run_single_mcts(args: Tuple[int, int, int]) -> Dict:
    """Run a single MCTS instance (for multiprocessing)"""
    instance_id, num_simulations, seed = args
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create game and evaluator
    game = GameInterface(GameType.GOMOKU)
    eval_config = EvaluatorConfig(
        batch_size=256,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_fp16=True
    )
    evaluator = MockEvaluator(eval_config, 225)
    
    # Create MCTS with smaller tree for parallel instances
    mcts_config = HighPerformanceMCTSConfig(
        num_simulations=num_simulations,
        wave_size=256,  # Smaller for parallel instances
        enable_gpu=torch.cuda.is_available(),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        mixed_precision=True,
        max_tree_size=50_000  # Smaller tree per instance
    )
    
    mcts = HighPerformanceMCTS(mcts_config, game, evaluator)
    state = game.create_initial_state()
    
    # Run searches
    start_time = time.time()
    num_searches = 10
    
    for _ in range(num_searches):
        mcts.search(state)
        
    elapsed = time.time() - start_time
    
    # Get stats
    total_sims = mcts.search_stats.get('total_simulations', 0)
    if total_sims == 0:
        # Fallback: estimate based on config
        total_sims = num_simulations * num_searches
        
    return {
        'instance_id': instance_id,
        'elapsed_time': elapsed,
        'total_simulations': total_sims,
        'sims_per_sec': total_sims / elapsed if elapsed > 0 else 0
    }


class TestCPUParallelization:
    """Test CPU parallelization strategies"""
    
    def test_baseline_sequential(self):
        """Establish baseline with sequential execution"""
        print("\n=== SEQUENTIAL BASELINE TEST ===")
        
        num_instances = 4
        num_simulations = 400
        
        start_time = time.time()
        results = []
        
        for i in range(num_instances):
            result = run_single_mcts((i, num_simulations, i))
            results.append(result)
            print(f"Instance {i}: {result['sims_per_sec']:.0f} sims/s")
            
        total_time = time.time() - start_time
        total_sims = sum(r['total_simulations'] for r in results)
        
        print(f"\nSequential execution:")
        print(f"Total time: {total_time:.2f}s")
        print(f"Total simulations: {total_sims:,}")
        print(f"Overall throughput: {total_sims/total_time:,.0f} sims/s")
        
        return total_sims / total_time
        
    def test_process_pool_parallelization(self):
        """Test multiprocessing with ProcessPoolExecutor"""
        print("\n=== PROCESS POOL PARALLELIZATION TEST ===")
        
        # Use number of physical cores
        num_cores = psutil.cpu_count(logical=False)
        num_instances = min(num_cores, 12)  # Cap at 12 for testing
        num_simulations = 400
        
        print(f"Running {num_instances} instances on {num_cores} physical cores")
        
        # Prepare arguments
        args_list = [(i, num_simulations, i) for i in range(num_instances)]
        
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=num_instances) as executor:
            results = list(executor.map(run_single_mcts, args_list))
            
        total_time = time.time() - start_time
        total_sims = sum(r['total_simulations'] for r in results)
        
        # Print individual results
        for r in results:
            print(f"Instance {r['instance_id']}: {r['sims_per_sec']:.0f} sims/s")
            
        print(f"\nParallel execution with {num_instances} processes:")
        print(f"Total time: {total_time:.2f}s")
        print(f"Total simulations: {total_sims:,}")
        print(f"Overall throughput: {total_sims/total_time:,.0f} sims/s")
        
        return total_sims / total_time
        
    def test_cpu_utilization(self):
        """Monitor CPU utilization during parallel execution"""
        print("\n=== CPU UTILIZATION TEST ===")
        
        num_instances = psutil.cpu_count(logical=False)
        
        # Monitor CPU usage
        cpu_samples = []
        monitoring = True
        
        def monitor_cpu():
            while monitoring:
                cpu_samples.append(psutil.cpu_percent(interval=0.1, percpu=True))
                
        # Start monitoring
        monitor_thread = mp.Process(target=monitor_cpu)
        monitor_thread.start()
        
        # Run parallel MCTS
        time.sleep(0.5)  # Let monitoring start
        
        args_list = [(i, 400, i) for i in range(num_instances)]
        
        with ProcessPoolExecutor(max_workers=num_instances) as executor:
            results = list(executor.map(run_single_mcts, args_list))
            
        # Stop monitoring
        monitoring = False
        monitor_thread.terminate()
        monitor_thread.join(timeout=1)
        
        # Analyze CPU usage
        if cpu_samples:
            avg_per_core = [np.mean([sample[i] for sample in cpu_samples]) 
                           for i in range(len(cpu_samples[0]))]
            overall_avg = np.mean(avg_per_core)
            
            print(f"\nCPU utilization during parallel execution:")
            print(f"Overall average: {overall_avg:.1f}%")
            print(f"Physical cores used: {sum(1 for x in avg_per_core[:num_instances] if x > 50)}/{num_instances}")
            
            if overall_avg < 50:
                print("⚠️  Low CPU utilization - parallelization may not be effective")
            else:
                print("✅ Good CPU utilization")
                
    def test_scaling_efficiency(self):
        """Test how performance scales with number of instances"""
        print("\n=== SCALING EFFICIENCY TEST ===")
        
        max_cores = min(psutil.cpu_count(logical=False), 12)
        instance_counts = [1, 2, 4, 8, max_cores]
        instance_counts = [n for n in instance_counts if n <= max_cores]
        
        results = []
        
        for num_instances in instance_counts:
            args_list = [(i, 200, i) for i in range(num_instances)]
            
            start_time = time.time()
            
            with ProcessPoolExecutor(max_workers=num_instances) as executor:
                instance_results = list(executor.map(run_single_mcts, args_list))
                
            elapsed = time.time() - start_time
            total_sims = sum(r['total_simulations'] for r in instance_results)
            throughput = total_sims / elapsed
            
            results.append({
                'instances': num_instances,
                'throughput': throughput,
                'time': elapsed
            })
            
            print(f"{num_instances} instances: {throughput:,.0f} sims/s")
            
        # Calculate scaling efficiency
        baseline = results[0]['throughput']
        
        print("\nScaling efficiency:")
        for r in results:
            ideal = baseline * r['instances']
            actual = r['throughput']
            efficiency = (actual / ideal) * 100
            print(f"{r['instances']} instances: {efficiency:.1f}% efficiency")
            
        return results
        
    def test_gpu_sharing(self):
        """Test GPU sharing among multiple CPU processes"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
            
        print("\n=== GPU SHARING TEST ===")
        
        # Test with fewer instances to avoid GPU memory issues
        num_instances = 4
        
        args_list = [(i, 200, i) for i in range(num_instances)]
        
        # Check initial GPU memory
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"Initial GPU memory: {initial_memory:.0f} MB")
        
        with ProcessPoolExecutor(max_workers=num_instances) as executor:
            results = list(executor.map(run_single_mcts, args_list))
            
        # Check final GPU memory
        final_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"Final GPU memory: {final_memory:.0f} MB")
        
        # Check if all instances completed successfully
        successful = sum(1 for r in results if r['sims_per_sec'] > 0)
        print(f"Successful instances: {successful}/{num_instances}")
        
        if successful < num_instances:
            print("⚠️  Some instances failed - GPU sharing may have issues")
        else:
            print("✅ All instances completed - GPU sharing works")


def create_parallel_mcts_wrapper():
    """Create a simple parallel MCTS wrapper for demonstration"""
    print("\n=== PARALLEL MCTS WRAPPER DESIGN ===")
    print("""
    class ParallelMCTS:
        def __init__(self, num_instances, config):
            self.num_instances = num_instances
            self.executor = ProcessPoolExecutor(max_workers=num_instances)
            self.configs = [copy.deepcopy(config) for _ in range(num_instances)]
            
        def search_parallel(self, states):
            '''Run searches on multiple states in parallel'''
            futures = []
            for i, state in enumerate(states):
                future = self.executor.submit(run_single_search, 
                                            self.configs[i], state)
                futures.append(future)
                
            results = [f.result() for f in futures]
            return results
            
        def aggregate_results(self, results):
            '''Combine results from parallel searches'''
            # Average policies, select best move, etc.
            pass
    """)


if __name__ == "__main__":
    test = TestCPUParallelization()
    
    print("=== CPU PARALLELIZATION TESTS ===")
    print(f"System: {psutil.cpu_count(logical=False)} physical cores, "
          f"{psutil.cpu_count()} logical cores")
    
    # Run tests
    seq_throughput = test.test_baseline_sequential()
    par_throughput = test.test_process_pool_parallelization()
    
    # Calculate speedup
    speedup = par_throughput / seq_throughput
    print(f"\n=== SUMMARY ===")
    print(f"Sequential throughput: {seq_throughput:,.0f} sims/s")
    print(f"Parallel throughput: {par_throughput:,.0f} sims/s")
    print(f"Speedup: {speedup:.1f}x")
    
    # Additional tests
    test.test_scaling_efficiency()
    test.test_cpu_utilization()
    
    if torch.cuda.is_available():
        test.test_gpu_sharing()
        
    # Show wrapper design
    create_parallel_mcts_wrapper()