#!/usr/bin/env python3
"""Simple test for hybrid CPU-GPU mode

This script provides a quick test to verify hybrid mode is working correctly.
"""

import torch
import time
import logging
import numpy as np
from mcts.core.hybrid_cpu_gpu import CPUWorker, HybridExecutor, HybridConfig
from mcts.neural_networks.lightweight_evaluator import create_cpu_evaluator
from alphazero_py import GomokuState

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
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
            return state.to_numpy()
        return np.zeros((20, self.board_size, self.board_size), dtype=np.float32)
    
    def get_state_shape(self):
        """Get state tensor shape"""
        return (20, self.board_size, self.board_size)


def test_cpu_worker():
    """Test individual CPU worker"""
    logger.info("Testing CPU Worker...")
    
    # Create configuration
    config = HybridConfig(
        num_cpu_threads=1,
        cpu_wave_size=32,
        cpu_batch_size=8
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
    
    logger.info("Running CPU wave...")
    start = time.perf_counter()
    result = worker.process_wave(root_state, wave_size=32)
    elapsed = time.perf_counter() - start
    
    logger.info(f"✓ CPU wave completed in {elapsed*1000:.1f} ms")
    logger.info(f"  Simulations: {result['wave_size']}")
    logger.info(f"  Worker ID: {result['worker_id']}")
    
    return True


def test_lightweight_evaluator():
    """Test lightweight evaluator"""
    logger.info("Testing Lightweight Evaluator...")
    
    evaluator = create_cpu_evaluator('lightweight', device='cpu')
    
    # Test single evaluation
    state = np.random.randn(20, 15, 15).astype(np.float32)
    start = time.perf_counter()
    policy, value = evaluator.evaluate(state)
    elapsed = time.perf_counter() - start
    
    logger.info(f"✓ Single evaluation: {elapsed*1000:.2f} ms")
    logger.info(f"  Policy shape: {policy.shape}")
    logger.info(f"  Value: {value:.3f}")
    
    # Test batch evaluation
    batch = np.random.randn(8, 20, 15, 15).astype(np.float32)
    start = time.perf_counter()
    policies, values = evaluator.evaluate_batch(batch)
    elapsed = time.perf_counter() - start
    
    logger.info(f"✓ Batch evaluation (8): {elapsed*1000:.2f} ms")
    logger.info(f"  Policies shape: {policies.shape}")
    logger.info(f"  Values shape: {values.shape}")
    
    return True


def test_hybrid_config():
    """Test hybrid configuration"""
    logger.info("Testing Hybrid Configuration...")
    
    config = HybridConfig(
        num_cpu_threads=4,
        cpu_wave_size=64,
        gpu_wave_size=512,
        gpu_allocation=0.7,
        cpu_allocation=0.3
    )
    
    logger.info(f"✓ Configuration created:")
    logger.info(f"  CPU threads: {config.num_cpu_threads}")
    logger.info(f"  CPU wave size: {config.cpu_wave_size}")
    logger.info(f"  GPU wave size: {config.gpu_wave_size}")
    logger.info(f"  Allocation: {config.gpu_allocation:.0%} GPU, {config.cpu_allocation:.0%} CPU")
    
    return True


def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("HYBRID CPU-GPU MODE SIMPLE TEST")
    logger.info("=" * 60)
    
    tests = [
        ("Hybrid Configuration", test_hybrid_config),
        ("Lightweight Evaluator", test_lightweight_evaluator),
        ("CPU Worker", test_cpu_worker),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n{test_name}:")
        logger.info("-" * 40)
        
        try:
            success = test_func()
            if success:
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                failed += 1
                logger.info(f"✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"✗ {test_name} FAILED: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    
    if failed == 0:
        logger.info("🎉 All tests passed! Hybrid mode is working.")
    else:
        logger.error(f"❌ {failed} test(s) failed.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)