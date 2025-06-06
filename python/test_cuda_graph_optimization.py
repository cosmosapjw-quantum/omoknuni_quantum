"""Test CUDA graph optimization with the vectorized MCTS

This script verifies that CUDA graph capture is working correctly
and measures the performance improvement.
"""

import torch
import time
import logging
from mcts.core.mcts_config import get_config
from mcts.gpu.csr_tree import CSRTree
from mcts.gpu.optimized_wave_engine import OptimizedWaveEngine, OptimizedWaveConfig
from mcts.gpu.cuda_graph_optimizer import CUDAGraphOptimizer, CUDAGraphConfig
from mcts.neural_networks.resnet_evaluator import create_resnet_evaluator
from alphazero_py import GomokuState
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
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
        return np.zeros((17, self.board_size, self.board_size), dtype=np.float32)


def test_cuda_graph_optimization():
    """Test CUDA graph optimization"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type != 'cuda':
        logger.warning("CUDA not available, skipping test")
        return
    
    # Create configurations
    wave_config = OptimizedWaveConfig()
    wave_config.device = str(device)
    wave_config.wave_size = 512  # Good size for testing
    wave_config.enable_cuda_graphs = True
    wave_config.enable_memory_pooling = True
    wave_config.enable_interference = False  # Disable for pure performance test
    wave_config.enable_phase_policy = False
    wave_config.enable_path_integral = False
    
    # Create CSR tree
    csr_tree = CSRTree(
        max_nodes=100000,
        max_children=225,  # 15x15 for Gomoku
        device=device
    )
    
    # Create game interface and evaluator
    game_interface = SimpleGameInterface()
    evaluator = create_resnet_evaluator(
        input_channels=17,
        board_size=15,
        num_blocks=4,
        channels=64,
        device=device
    )
    
    # Create wave engine
    wave_engine = OptimizedWaveEngine(
        csr_tree=csr_tree,
        config=wave_config,
        game_interface=game_interface,
        evaluator=evaluator
    )
    
    # Initialize root state
    root_state = GomokuState()
    csr_tree.node_states[0] = root_state
    
    # Warmup iterations (these will be captured)
    logger.info("Running warmup iterations for CUDA graph capture...")
    warmup_times = []
    
    for i in range(5):
        start = time.perf_counter()
        result = wave_engine.run_wave(root_state, wave_size=512)
        torch.cuda.synchronize()
        warmup_times.append(time.perf_counter() - start)
        logger.info(f"Warmup iteration {i+1}: {result['performance']['sims_per_second']:.0f} sims/sec")
    
    # Get CUDA graph statistics
    if wave_engine.cuda_graph_optimizer:
        stats = wave_engine.cuda_graph_optimizer.get_statistics()
        logger.info(f"\nCUDA Graph Statistics after warmup:")
        logger.info(f"  Graphs captured: {stats['graphs_captured']}")
        logger.info(f"  Cache hits: {stats['cache_hits']}")
        logger.info(f"  Cache misses: {stats['cache_misses']}")
        logger.info(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
    
    # Performance test with graphs
    logger.info("\nRunning performance test with CUDA graphs...")
    test_times = []
    
    for i in range(10):
        start = time.perf_counter()
        result = wave_engine.run_wave(root_state, wave_size=512)
        torch.cuda.synchronize()
        test_times.append(time.perf_counter() - start)
        
        if i % 5 == 0:
            logger.info(f"Test iteration {i+1}: {result['performance']['sims_per_second']:.0f} sims/sec")
    
    # Compare with graphs disabled
    logger.info("\nDisabling CUDA graphs for comparison...")
    wave_config.enable_cuda_graphs = False
    wave_engine_no_graphs = OptimizedWaveEngine(
        csr_tree=csr_tree,
        config=wave_config,
        game_interface=game_interface,
        evaluator=evaluator
    )
    
    # Reset tree state
    csr_tree.reset()
    csr_tree.node_states[0] = root_state
    
    no_graph_times = []
    for i in range(10):
        start = time.perf_counter()
        result = wave_engine_no_graphs.run_wave(root_state, wave_size=512)
        torch.cuda.synchronize()
        no_graph_times.append(time.perf_counter() - start)
        
        if i % 5 == 0:
            logger.info(f"No-graph iteration {i+1}: {result['performance']['sims_per_second']:.0f} sims/sec")
    
    # Calculate statistics
    avg_time_with_graphs = np.mean(test_times[2:])  # Skip first 2 for stability
    avg_time_without_graphs = np.mean(no_graph_times[2:])
    
    speedup = avg_time_without_graphs / avg_time_with_graphs
    sims_per_sec_with_graphs = 512 / avg_time_with_graphs
    sims_per_sec_without_graphs = 512 / avg_time_without_graphs
    
    logger.info("\n" + "="*60)
    logger.info("CUDA Graph Optimization Results:")
    logger.info("="*60)
    logger.info(f"Average time per wave WITH graphs: {avg_time_with_graphs*1000:.2f}ms")
    logger.info(f"Average time per wave WITHOUT graphs: {avg_time_without_graphs*1000:.2f}ms")
    logger.info(f"Speedup from CUDA graphs: {speedup:.2f}x")
    logger.info(f"Simulations/sec WITH graphs: {sims_per_sec_with_graphs:,.0f}")
    logger.info(f"Simulations/sec WITHOUT graphs: {sims_per_sec_without_graphs:,.0f}")
    
    # Final statistics
    if wave_engine.cuda_graph_optimizer:
        final_stats = wave_engine.cuda_graph_optimizer.get_statistics()
        logger.info(f"\nFinal CUDA Graph Statistics:")
        logger.info(f"  Total graphs captured: {final_stats['graphs_captured']}")
        logger.info(f"  Total graph replays: {final_stats['graph_replays']}")
        logger.info(f"  Final cache hit rate: {final_stats['cache_hit_rate']:.2%}")
    
    # Check if we're approaching the 100k target
    logger.info(f"\nProgress toward 100k sims/sec target: {sims_per_sec_with_graphs/100000:.1%}")
    
    return sims_per_sec_with_graphs


if __name__ == "__main__":
    sims_per_sec = test_cuda_graph_optimization()
    
    if sims_per_sec and sims_per_sec > 80000:
        logger.info("\n🎉 Excellent! Approaching the 100k sims/sec target!")
    elif sims_per_sec and sims_per_sec > 50000:
        logger.info("\n✅ Good progress! Phase 2 target (50k) achieved!")
    else:
        logger.info("\n⚡ More optimization needed to reach performance targets.")