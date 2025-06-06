"""Test script to verify all optimizations are working correctly"""

import torch
import time
import logging
from mcts.gpu.csr_tree import CSRTree, CSRTreeConfig
from mcts.gpu.optimized_wave_engine import OptimizedWaveEngine, OptimizedWaveConfig
from mcts.core.high_performance_mcts import HighPerformanceMCTS, HighPerformanceMCTSConfig
from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
from alphazero_py import GomokuState
import numpy as np

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
        # Return dummy array
        return np.zeros((20, self.board_size, self.board_size), dtype=np.float32)


def test_optimizations():
    """Test all optimizations"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Testing on device: {device}")
    
    # Create game interface and evaluator
    game_interface = SimpleGameInterface()
    evaluator = ResNetEvaluator(
        game_type='gomoku',
        device=device
    )
    
    logger.info("\n" + "="*60)
    logger.info("TESTING MCTS OPTIMIZATIONS")
    logger.info("="*60)
    
    # Test 1: CSR Tree
    logger.info("\n1. Testing CSR Tree Format...")
    tree_config = CSRTreeConfig(max_nodes=100000, max_edges=500000, device=str(device))
    csr_tree = CSRTree(tree_config)
    
    # Add some nodes
    root = csr_tree.add_root(state=GomokuState())
    children = []
    for i in range(10):
        child = csr_tree.add_child(root, i, child_prior=0.1)
        children.append(child)
    
    # Test batch operations
    if hasattr(csr_tree, 'batch_ops') and csr_tree.batch_ops is not None:
        logger.info("✓ CSR tree with optimized batch operations enabled")
    else:
        logger.info("✗ CSR tree using fallback (optimized kernels not available)")
    
    # Test 2: Custom CUDA Kernels
    logger.info("\n2. Testing Custom CUDA Kernels...")
    from mcts.gpu.csr_gpu_kernels_optimized import get_csr_batch_operations
    
    batch_ops = get_csr_batch_operations(device)
    if batch_ops.use_custom_cuda:
        logger.info("✓ Custom CUDA kernels available and enabled")
    else:
        logger.info("✗ Custom CUDA kernels not available, using PyTorch fallback")
    
    # Test 3: CUDA Graph Capture
    logger.info("\n3. Testing CUDA Graph Capture...")
    wave_config = OptimizedWaveConfig()
    wave_config.device = str(device)
    wave_config.enable_cuda_graphs = True
    
    wave_engine = OptimizedWaveEngine(
        csr_tree=csr_tree,
        config=wave_config,
        game_interface=game_interface,
        evaluator=evaluator
    )
    
    if wave_engine.cuda_graph_optimizer is not None:
        logger.info("✓ CUDA graph capture enabled")
    else:
        logger.info("✗ CUDA graph capture not available")
    
    # Test 4: MinHash Interference
    logger.info("\n4. Testing MinHash Interference...")
    if wave_config.enable_interference:
        logger.info("✓ MinHash interference enabled")
        logger.info(f"  Interference strength: {wave_config.interference_strength}")
    else:
        logger.info("✗ MinHash interference disabled")
    
    # Test 5: Path Integral Action Selection
    logger.info("\n5. Testing Path Integral Action Selection...")
    if wave_config.enable_path_integral:
        logger.info("✓ Path integral action selection enabled (default)")
    else:
        logger.info("✗ Path integral action selection disabled")
    
    # Test 6: State Delta Encoding
    logger.info("\n6. Testing State Delta Encoding...")
    if hasattr(wave_engine, 'state_manager'):
        logger.info("✓ State delta encoding integrated")
        logger.info(f"  Max states: {wave_engine.state_manager.max_states}")
    else:
        logger.info("✗ State delta encoding not integrated")
    
    # Test 7: Performance Benchmark
    logger.info("\n7. Running Performance Benchmark...")
    
    # Create high-performance MCTS
    mcts_config = HighPerformanceMCTSConfig(
        num_simulations=800,
        wave_size=512,
        device=str(device),
        enable_gpu=True,
        enable_interference=True,
        enable_path_integral=True
    )
    
    mcts = HighPerformanceMCTS(
        config=mcts_config,
        game_interface=game_interface,
        evaluator=evaluator
    )
    
    # Warm up
    root_state = GomokuState()
    logger.info("  Warming up...")
    for _ in range(3):
        mcts.search(root_state)
    
    # Benchmark
    logger.info("  Running benchmark...")
    times = []
    for i in range(5):
        start = time.perf_counter()
        policy = mcts.search(root_state)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        if i == 0:
            logger.info(f"  Policy contains {len(policy)} moves")
    
    avg_time = np.mean(times[1:])  # Skip first for stability
    sims_per_sec = mcts_config.num_simulations / avg_time
    
    logger.info(f"\n  Average search time: {avg_time*1000:.1f}ms")
    logger.info(f"  Simulations/second: {sims_per_sec:,.0f}")
    
    # Check quantum statistics
    if hasattr(mcts.wave_engine, 'quantum_stats'):
        stats = mcts.wave_engine.quantum_stats
        logger.info(f"\n  Quantum Features:")
        logger.info(f"    Interference applied: {stats.get('interference_applied', 0)}")
        logger.info(f"    Phase kicks applied: {stats.get('phase_kicks_applied', 0)}")
        logger.info(f"    Path integral used: {stats.get('path_integral_used', 0)}")
    
    # Performance report
    logger.info("\n" + "="*60)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("="*60)
    
    optimizations = {
        "CSR Tree Format": csr_tree.use_optimized,
        "Custom CUDA Kernels": batch_ops.use_custom_cuda,
        "CUDA Graph Capture": wave_engine.cuda_graph_optimizer is not None,
        "MinHash Interference": wave_config.enable_interference,
        "Path Integral Selection": wave_config.enable_path_integral,
        "State Delta Encoding": hasattr(wave_engine, 'state_manager'),
    }
    
    enabled_count = sum(1 for v in optimizations.values() if v)
    
    for name, enabled in optimizations.items():
        status = "✓" if enabled else "✗"
        logger.info(f"{status} {name}")
    
    logger.info(f"\nOptimizations enabled: {enabled_count}/{len(optimizations)}")
    logger.info(f"Performance: {sims_per_sec:,.0f} sims/sec")
    
    if sims_per_sec > 100000:
        logger.info("\n🎉 EXCELLENT! Exceeded 100k sims/sec target!")
    elif sims_per_sec > 50000:
        logger.info("\n✅ GOOD! Phase 2 target (50k) achieved!")
    elif sims_per_sec > 10000:
        logger.info("\n✓ Phase 1 target (10k) achieved!")
    else:
        logger.info("\n⚡ Performance below targets, further optimization needed")
    
    return sims_per_sec


if __name__ == "__main__":
    sims_per_sec = test_optimizations()
    logger.info(f"\nFinal result: {sims_per_sec:,.0f} simulations/second")