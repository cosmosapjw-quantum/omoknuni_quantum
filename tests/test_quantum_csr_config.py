"""Test CSR configuration in quantum modules"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Direct import to avoid circular imports
import importlib.util
spec = importlib.util.spec_from_file_location(
    "mcts_integration", 
    "/home/cosmosapjw/omoknuni_quantum/python/mcts/quantum/analysis/mcts_integration.py"
)
mcts_integration = importlib.util.module_from_spec(spec)

# Mock dependencies to avoid import issues
sys.modules['python.mcts.core.mcts'] = type(sys)('mock')
sys.modules['python.mcts.core.mcts_config'] = type(sys)('mock')
sys.modules['python.mcts.core.game_interface'] = type(sys)('mock')
sys.modules['python.mcts.neural_networks.resnet_model'] = type(sys)('mock')
sys.modules['python.mcts.utils.single_gpu_evaluator'] = type(sys)('mock')
sys.modules['random_evaluator'] = type(sys)('mock')

# Create mock classes
class MockGameType:
    CHESS = 'chess'
    GO = 'go'
    GOMOKU = 'gomoku'
    
    def __init__(self, game_type):
        self.name = game_type

sys.modules['python.mcts.core.game_interface'].GameType = MockGameType

class MockMCTSConfig:
    def __init__(self):
        self.num_simulations = 1000
        self.c_puct = 1.414
        self.device = 'cpu'
        self.csr_max_actions = 225  # Default

sys.modules['python.mcts.core.mcts_config'].MCTSConfig = MockMCTSConfig


def test_csr_config_for_different_games():
    """Test that CSR max actions is correctly set for different game types"""
    
    # Test Gomoku
    gomoku_game = RealMCTSGame(
        sims_per_game=1000,
        game_type='gomoku',
        board_size=15,
        evaluator_type='random'
    )
    assert gomoku_game.mcts_config.csr_max_actions == 225, \
        f"Gomoku CSR should be 225, got {gomoku_game.mcts_config.csr_max_actions}"
    print("✓ Gomoku CSR configuration correct: 225")
    
    # Test Go
    go_game = RealMCTSGame(
        sims_per_game=1000,
        game_type='go',
        board_size=19,
        evaluator_type='random'
    )
    assert go_game.mcts_config.csr_max_actions == 361, \
        f"Go 19x19 CSR should be 361, got {go_game.mcts_config.csr_max_actions}"
    print("✓ Go CSR configuration correct: 361")
    
    # Test Chess
    chess_game = RealMCTSGame(
        sims_per_game=1000,
        game_type='chess',
        board_size=8,
        evaluator_type='random'
    )
    assert chess_game.mcts_config.csr_max_actions == 4096, \
        f"Chess CSR should be 4096, got {chess_game.mcts_config.csr_max_actions}"
    print("✓ Chess CSR configuration correct: 4096")
    
    print("\n✓ All quantum module CSR configurations are correct!")


def test_memory_scaling():
    """Test that memory settings scale appropriately with simulation count"""
    
    # Test different simulation counts
    sim_counts = [1000, 2500, 5000, 7500, 10000]
    
    for sims in sim_counts:
        game = RealMCTSGame(
            sims_per_game=sims,
            game_type='gomoku',
            board_size=15,
            evaluator_type='random'
        )
        
        print(f"\nSimulations: {sims}")
        print(f"  Memory pool: {game.mcts_config.memory_pool_size_mb} MB")
        print(f"  Max tree nodes: {game.mcts_config.max_tree_nodes:,}")
        print(f"  Max wave size: {game.mcts_config.max_wave_size}")
        print(f"  Children per expansion: {game.mcts_config.initial_children_per_expansion}")
        
        # Verify memory scales appropriately
        if sims >= 10000:
            assert game.mcts_config.memory_pool_size_mb == 4096
            assert game.mcts_config.max_tree_nodes == 8000000
        elif sims >= 7500:
            assert game.mcts_config.memory_pool_size_mb == 3584
            assert game.mcts_config.max_tree_nodes == 6000000
        elif sims >= 5000:
            assert game.mcts_config.memory_pool_size_mb == 3072
            assert game.mcts_config.max_tree_nodes == 4000000
        elif sims >= 2500:
            assert game.mcts_config.memory_pool_size_mb == 2048
            assert game.mcts_config.max_tree_nodes == 2000000
        else:
            assert game.mcts_config.memory_pool_size_mb == 1024
            assert game.mcts_config.max_tree_nodes == 1000000
    
    print("\n✓ Memory scaling configurations are correct!")


if __name__ == "__main__":
    print("Testing quantum module CSR configurations...")
    test_csr_config_for_different_games()
    test_memory_scaling()