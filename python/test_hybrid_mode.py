#!/usr/bin/env python3
"""Test script for hybrid CPU-GPU mode

This script tests the hybrid execution mode that leverages both
the Ryzen 5900X CPU and GPU for optimal performance.
"""

import torch
import time
import logging
import numpy as np
from mcts.core.high_performance_mcts import HighPerformanceMCTS, HighPerformanceMCTSConfig
from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
from alphazero_py import GomokuState
import psutil
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleGameInterface:
    """Simple game interface for testing"""
    
    def __init__(self):
        self.board_size = 15
        self.action_space = self.board_size * self.board_size
        
    def get_legal_moves(self, state):
        """Get legal moves from state"""
        if hasattr(state, 'get_legal_moves'):
            return state.get_legal_moves()
        return list(range(self.action_space))
    
    def apply_move(self, state, action):
        """Apply move to state"""
        if hasattr(state, 'apply_move'):
            new_state = state.clone()
            new_state.apply_move(action)
            return new_state
        return state
    
    def state_to_numpy(self, state, use_enhanced=True):
        """Convert state to numpy array"""
        if hasattr(state, 'to_numpy'):
            # The C++ implementation provides the correct format
            return state.to_numpy()
        # Return dummy array with 20 channels to match ResNet expectations
        return np.zeros((20, self.board_size, self.board_size), dtype=np.float32)
    
    def get_state_shape(self):
        """Get state tensor shape"""
        return (20, self.board_size, self.board_size)


def get_cpu_info():
    """Get CPU information"""
    try:
        # Get CPU count
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        
        # Get CPU frequency
        freq = psutil.cpu_freq()
        
        return {
            'physical_cores': physical_cores,
            'logical_cores': logical_cores,
            'current_freq': freq.current if freq else 0,
            'max_freq': freq.max if freq else 0
        }
    except:
        return {
            'physical_cores': 'Unknown',
            'logical_cores': 'Unknown',
            'current_freq': 'Unknown',
            'max_freq': 'Unknown'
        }


def test_hybrid_mode():
    """Test hybrid CPU-GPU execution"""
    
    logger.info("=" * 80)
    logger.info("HYBRID CPU-GPU MODE TEST")
    logger.info("=" * 80)
    
    # System information
    cpu_info = get_cpu_info()
    logger.info("\nSystem Information:")
    logger.info(f"  CPU: Ryzen 5900X (expected)")
    logger.info(f"  Physical cores: {cpu_info['physical_cores']}")
    logger.info(f"  Logical cores: {cpu_info['logical_cores']}")
    logger.info(f"  Current frequency: {cpu_info['current_freq']:.2f} MHz")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"  GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create game interface and evaluator
    game_interface = SimpleGameInterface()
    evaluator = ResNetEvaluator(
        game_type='gomoku',
        device=device
    )
    
    # Test configurations
    configs = [
        {
            'name': 'GPU-only (baseline)',
            'enable_hybrid_mode': False,
            'num_simulations': 1600,
            'wave_size': 1024
        },
        {
            'name': 'Hybrid mode (4 CPU workers)',
            'enable_hybrid_mode': True,
            'num_cpu_workers': 4,
            'cpu_wave_size': 128,
            'num_simulations': 1600,
            'wave_size': 1024
        },
        {
            'name': 'Hybrid mode (8 CPU workers)',
            'enable_hybrid_mode': True,
            'num_cpu_workers': 8,
            'cpu_wave_size': 64,
            'num_simulations': 1600,
            'wave_size': 1024
        },
        {
            'name': 'Hybrid mode (12 CPU workers)',
            'enable_hybrid_mode': True,
            'num_cpu_workers': 12,
            'cpu_wave_size': 64,
            'num_simulations': 1600,
            'wave_size': 1024
        }
    ]
    
    results = []
    root_state = GomokuState()
    
    for config in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {config['name']}")
        logger.info(f"{'='*60}")
        
        # Create MCTS configuration
        mcts_config = HighPerformanceMCTSConfig(
            num_simulations=config['num_simulations'],
            wave_size=config['wave_size'],
            device=str(device),
            enable_gpu=True,
            enable_interference=True,
            enable_path_integral=True,
            enable_hybrid_mode=config['enable_hybrid_mode'],
            num_cpu_workers=config.get('num_cpu_workers', 4),
            cpu_wave_size=config.get('cpu_wave_size', 128)
        )
        
        # Create MCTS instance
        mcts = HighPerformanceMCTS(
            config=mcts_config,
            game_interface=game_interface,
            evaluator=evaluator
        )
        
        # Warm up
        logger.info("Warming up...")
        for _ in range(2):
            mcts.search(root_state)
        
        # Clear stats
        mcts.search_stats = {
            'total_searches': 0,
            'total_simulations': 0,
            'total_time': 0.0,
            'avg_simulations_per_second': 0.0,
        }
        
        # Benchmark
        logger.info("Running benchmark...")
        num_searches = 5
        search_times = []
        cpu_usages = []
        
        for i in range(num_searches):
            # Monitor CPU usage
            cpu_before = psutil.cpu_percent(interval=None)
            
            start = time.perf_counter()
            policy = mcts.search(root_state)
            elapsed = time.perf_counter() - start
            
            # Get CPU usage
            cpu_after = psutil.cpu_percent(interval=0.1)
            cpu_usage = (cpu_before + cpu_after) / 2
            
            search_times.append(elapsed)
            cpu_usages.append(cpu_usage)
            
            if i == 0:
                logger.info(f"  Policy contains {len(policy)} moves")
        
        # Calculate statistics
        avg_time = np.mean(search_times[1:])  # Skip first for stability
        std_time = np.std(search_times[1:])
        avg_cpu = np.mean(cpu_usages[1:])
        sims_per_sec = config['num_simulations'] / avg_time
        
        # Get performance report if hybrid mode
        if config['enable_hybrid_mode'] and hasattr(mcts, 'hybrid_executor'):
            perf_report = mcts.hybrid_executor.get_performance_report()
        else:
            perf_report = None
        
        result = {
            'name': config['name'],
            'avg_time': avg_time,
            'std_time': std_time,
            'avg_cpu_usage': avg_cpu,
            'simulations_per_second': sims_per_sec,
            'performance_report': perf_report
        }
        results.append(result)
        
        # Display results
        logger.info(f"\nResults for {config['name']}:")
        logger.info(f"  Average search time: {avg_time*1000:.1f} ± {std_time*1000:.1f} ms")
        logger.info(f"  Simulations/second: {sims_per_sec:,.0f}")
        logger.info(f"  Average CPU usage: {avg_cpu:.1f}%")
        
        if perf_report:
            logger.info(f"\nHybrid Performance Details:")
            logger.info(f"  Total simulations: {perf_report['total_simulations']}")
            logger.info(f"  CPU throughput: {perf_report['cpu_statistics']['throughput']:,.0f} sims/sec")
            logger.info(f"  GPU throughput: {perf_report['gpu_statistics']['throughput']:,.0f} sims/sec")
            logger.info(f"  Current allocation: {perf_report['allocation']['current_gpu']:.0%} GPU, "
                       f"{perf_report['allocation']['current_cpu']:.0%} CPU")
        
        # Clean up
        del mcts
        torch.cuda.empty_cache()
        time.sleep(1)  # Let system settle
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("="*80)
    
    baseline_sims = results[0]['simulations_per_second']
    
    for result in results:
        speedup = result['simulations_per_second'] / baseline_sims
        logger.info(f"\n{result['name']}:")
        logger.info(f"  Performance: {result['simulations_per_second']:,.0f} sims/sec")
        logger.info(f"  Speedup: {speedup:.2f}x")
        logger.info(f"  CPU usage: {result['avg_cpu_usage']:.1f}%")
    
    # Find best configuration
    best_result = max(results, key=lambda x: x['simulations_per_second'])
    best_speedup = best_result['simulations_per_second'] / baseline_sims
    
    logger.info(f"\nBest configuration: {best_result['name']}")
    logger.info(f"Best performance: {best_result['simulations_per_second']:,.0f} sims/sec")
    logger.info(f"Speedup over GPU-only: {best_speedup:.2f}x")
    
    # Performance assessment
    if best_result['simulations_per_second'] > 150000:
        logger.info("\n🎉 EXCELLENT! Exceeded 150k sims/sec with hybrid mode!")
    elif best_result['simulations_per_second'] > 100000:
        logger.info("\n✅ GREAT! Exceeded 100k sims/sec target with hybrid mode!")
    elif best_speedup > 1.2:
        logger.info("\n✓ Good! Hybrid mode provides significant speedup!")
    else:
        logger.info("\n⚡ Hybrid mode needs further optimization")
    
    return results


def test_cpu_workers():
    """Test CPU workers independently"""
    logger.info("\n" + "="*80)
    logger.info("CPU WORKER TEST")
    logger.info("="*80)
    
    from mcts.core.hybrid_cpu_gpu import CPUWorker, HybridConfig
    
    # Create test configuration
    config = HybridConfig(
        num_cpu_threads=1,
        cpu_wave_size=64,
        cpu_batch_size=16
    )
    
    # Create game interface
    game_interface = SimpleGameInterface()
    
    # Create CPU worker
    worker = CPUWorker(
        worker_id=0,
        config=config,
        game_interface=game_interface,
        evaluator=None  # Will use lightweight evaluator
    )
    
    # Test single wave
    root_state = GomokuState()
    
    logger.info("Testing single CPU wave...")
    start = time.perf_counter()
    result = worker.process_wave(root_state, wave_size=64)
    elapsed = time.perf_counter() - start
    
    logger.info(f"Wave completed in {elapsed*1000:.1f} ms")
    logger.info(f"Simulations: {result['wave_size']}")
    logger.info(f"Timing breakdown:")
    logger.info(f"  Selection: {result['timing']['selection']*1000:.1f} ms")
    logger.info(f"  Evaluation: {result['timing']['evaluation']*1000:.1f} ms")
    logger.info(f"  Backup: {result['timing']['backup']*1000:.1f} ms")
    
    # Test lightweight evaluator
    logger.info("\nTesting lightweight evaluator...")
    from mcts.neural_networks.lightweight_evaluator import create_cpu_evaluator
    
    evaluator = create_cpu_evaluator('lightweight', device='cpu')
    
    # Test single evaluation
    state_array = np.random.randn(20, 15, 15).astype(np.float32)
    start = time.perf_counter()
    policy, value = evaluator.evaluate(state_array)
    elapsed = time.perf_counter() - start
    
    logger.info(f"Single evaluation: {elapsed*1000:.2f} ms")
    
    # Test batch evaluation
    batch = np.random.randn(16, 20, 15, 15).astype(np.float32)
    start = time.perf_counter()
    policies, values = evaluator.evaluate_batch(batch)
    elapsed = time.perf_counter() - start
    
    logger.info(f"Batch evaluation (16): {elapsed*1000:.2f} ms")
    logger.info(f"Per-state: {elapsed*1000/16:.2f} ms")


if __name__ == "__main__":
    # Test CPU workers first
    test_cpu_workers()
    
    # Then test full hybrid mode
    results = test_hybrid_mode()