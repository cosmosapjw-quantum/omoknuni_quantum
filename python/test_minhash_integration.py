"""Test MinHash interference integration in vectorized MCTS

This script verifies that MinHash interference is properly reducing
redundant exploration by penalizing similar paths.
"""

import torch
import numpy as np
import logging
from mcts.gpu.csr_tree import CSRTree
from mcts.gpu.optimized_wave_engine import OptimizedWaveEngine, OptimizedWaveConfig
from mcts.quantum.interference import MinHashInterference
from alphazero_py import GomokuState

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


class DummyEvaluator:
    """Dummy evaluator for testing"""
    
    def __init__(self, device):
        self.device = device
        
    def evaluate_batch(self, states):
        """Return random policy and values"""
        batch_size = states.shape[0] if isinstance(states, (np.ndarray, torch.Tensor)) else len(states)
        
        # Random policy (uniform over all actions)
        policy = np.ones((batch_size, 225)) / 225.0
        
        # Random values
        values = np.random.randn(batch_size) * 0.1
        
        return policy, values


def test_minhash_interference():
    """Test MinHash interference in MCTS"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Testing on device: {device}")
    
    # Test 1: Direct MinHash interference test
    logger.info("\n=== Test 1: Direct MinHash Interference ===")
    
    interference = MinHashInterference(device=device, strength=0.3)
    
    # Create similar paths
    paths = torch.tensor([
        [0, 1, 2, 3, 4, -1, -1, -1],  # Path 1
        [0, 1, 2, 3, 5, -1, -1, -1],  # Path 2 (similar to 1)
        [0, 1, 2, 3, 6, -1, -1, -1],  # Path 3 (similar to 1,2)
        [0, 7, 8, 9, 10, -1, -1, -1], # Path 4 (different)
        [0, 7, 8, 9, 11, -1, -1, -1], # Path 5 (similar to 4)
    ], device=device)
    
    # Compute diversity
    signatures, similarities = interference.compute_path_diversity_batch(paths, num_hashes=64)
    
    logger.info(f"Similarity matrix:\n{similarities}")
    logger.info(f"Average similarity: {similarities.mean().item():.3f}")
    
    # Test interference application
    scores = torch.tensor([1.0, 0.9, 0.8, 1.0, 0.9], device=device)
    modified_scores = interference.apply_interference(scores, similarities)
    
    logger.info(f"Original scores: {scores}")
    logger.info(f"Modified scores: {modified_scores}")
    logger.info(f"Score reduction: {((scores - modified_scores) / scores * 100)}")
    
    # Test 2: Integration with wave engine
    logger.info("\n=== Test 2: Wave Engine Integration ===")
    
    # Create configurations
    config_with_interference = OptimizedWaveConfig()
    config_with_interference.device = str(device)
    config_with_interference.wave_size = 32
    config_with_interference.enable_interference = True
    config_with_interference.interference_strength = 0.3
    config_with_interference.enable_phase_policy = False
    config_with_interference.enable_path_integral = False
    
    config_without_interference = OptimizedWaveConfig()
    config_without_interference.device = str(device)
    config_without_interference.wave_size = 32
    config_without_interference.enable_interference = False
    config_without_interference.enable_phase_policy = False
    config_without_interference.enable_path_integral = False
    
    # Create CSR trees
    tree_with = CSRTree(max_nodes=10000, max_children=225, device=device)
    tree_without = CSRTree(max_nodes=10000, max_children=225, device=device)
    
    # Create game interface and evaluator
    game_interface = SimpleGameInterface()
    evaluator = DummyEvaluator(device)
    
    # Create wave engines
    engine_with = OptimizedWaveEngine(
        csr_tree=tree_with,
        config=config_with_interference,
        game_interface=game_interface,
        evaluator=evaluator
    )
    
    engine_without = OptimizedWaveEngine(
        csr_tree=tree_without,
        config=config_without_interference,
        game_interface=game_interface,
        evaluator=evaluator
    )
    
    # Initialize root state
    root_state = GomokuState()
    tree_with.node_states[0] = root_state
    tree_without.node_states[0] = root_state
    
    # Run waves and compare
    logger.info("\nRunning wave WITH interference...")
    result_with = engine_with.run_wave(root_state, wave_size=32)
    
    logger.info("\nRunning wave WITHOUT interference...")
    result_without = engine_without.run_wave(root_state, wave_size=32)
    
    # Analyze results
    paths_with = result_with['paths_tensor']
    paths_without = result_without['paths_tensor']
    
    # Compute path diversity for both
    _, similarities_with = interference.compute_path_diversity_batch(paths_with)
    _, similarities_without = interference.compute_path_diversity_batch(paths_without)
    
    avg_sim_with = (similarities_with - torch.eye(32, device=device)).mean().item()
    avg_sim_without = (similarities_without - torch.eye(32, device=device)).mean().item()
    
    logger.info(f"\nPath diversity comparison:")
    logger.info(f"Average similarity WITH interference: {avg_sim_with:.3f}")
    logger.info(f"Average similarity WITHOUT interference: {avg_sim_without:.3f}")
    logger.info(f"Diversity improvement: {((avg_sim_without - avg_sim_with) / avg_sim_without * 100):.1f}%")
    
    # Check quantum stats
    logger.info(f"\nQuantum stats WITH interference:")
    logger.info(f"  Interference applied: {engine_with.quantum_stats['interference_applied']}")
    logger.info(f"  Diversity scores: {engine_with.quantum_stats['diversity_scores']}")
    
    # Test 3: Performance impact
    logger.info("\n=== Test 3: Performance Impact ===")
    
    import time
    
    # Reset trees
    tree_with.reset()
    tree_without.reset()
    tree_with.node_states[0] = root_state
    tree_without.node_states[0] = root_state
    
    # Time multiple waves
    n_waves = 10
    times_with = []
    times_without = []
    
    for i in range(n_waves):
        # With interference
        start = time.perf_counter()
        engine_with.run_wave(root_state, wave_size=64)
        times_with.append(time.perf_counter() - start)
        
        # Without interference
        start = time.perf_counter()
        engine_without.run_wave(root_state, wave_size=64)
        times_without.append(time.perf_counter() - start)
    
    avg_time_with = np.mean(times_with[2:])  # Skip warmup
    avg_time_without = np.mean(times_without[2:])
    overhead = (avg_time_with - avg_time_without) / avg_time_without * 100
    
    logger.info(f"\nPerformance comparison:")
    logger.info(f"Average time WITH interference: {avg_time_with*1000:.2f}ms")
    logger.info(f"Average time WITHOUT interference: {avg_time_without*1000:.2f}ms")
    logger.info(f"Overhead from interference: {overhead:.1f}%")
    
    # Success criteria
    if avg_sim_with < avg_sim_without and overhead < 20:
        logger.info("\n✅ MinHash interference successfully integrated!")
        logger.info("   - Path diversity improved")
        logger.info("   - Performance overhead acceptable")
    else:
        logger.info("\n⚠️  MinHash interference needs optimization")
        if avg_sim_with >= avg_sim_without:
            logger.info("   - Path diversity not improved")
        if overhead >= 20:
            logger.info("   - Performance overhead too high")


if __name__ == "__main__":
    test_minhash_interference()