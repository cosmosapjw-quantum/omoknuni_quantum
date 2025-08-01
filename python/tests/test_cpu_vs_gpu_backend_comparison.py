#!/usr/bin/env python3
"""
Comprehensive CPU vs GPU/Hybrid backend comparison tests

This test suite performs side-by-side comparison of CPU backend behavior
against GPU and Hybrid backends to ensure consistency and identify
any backend-specific issues.
"""

import pytest
import torch
import numpy as np
import logging
from pathlib import Path
import sys
from typing import Dict, Any, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mcts.core.mcts import MCTS
from mcts.core.mcts_config import MCTSConfig
from mcts.core.game_interface import GameInterface, GameType
from mcts.neural_networks.resnet_model import create_resnet_for_game
from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
from mcts.cpu import cpu_game_states

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BackendComparator:
    """Comprehensive backend comparison framework"""
    
    def __init__(self, game_type='gomoku', board_size=15):
        self.game_type = game_type
        self.board_size = board_size
        self.backends = ['cpu', 'gpu', 'hybrid']
        self.comparison_results = {}
        
    def create_mcts_config(self, backend: str) -> MCTSConfig:
        """Create consistent MCTS config for each backend"""
        config = MCTSConfig()
        config.num_simulations = 20  # Small for fast comparison
        config.backend = backend
        config.device = 'cuda' if backend in ['gpu', 'hybrid'] else 'cpu'
        config.max_tree_nodes = 1000
        config.initial_children_per_expansion = 5
        config.max_children_per_node = 25
        config.enable_subtree_reuse = False
        config.enable_dirichlet_noise = False
        config.classical_only_mode = True
        
        # CPU-specific configuration
        if backend == 'cpu':
            config.cpu_game_states_factory = lambda **kwargs: cpu_game_states.CPUGameStates(
                capacity=kwargs.get('capacity', 1000),
                game_type=self.game_type,
                board_size=self.board_size
            )
        
        return config
    
    def create_evaluator(self, backend: str) -> ResNetEvaluator:
        """Create consistent neural network evaluator"""
        model = create_resnet_for_game(
            game_type=self.game_type,
            input_channels=19,
            num_blocks=2,
            num_filters=32
        )
        model.eval()
        
        device = 'cuda' if backend in ['gpu', 'hybrid'] and torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            model = model.cuda()
        
        return ResNetEvaluator(
            model=model,
            game_type=self.game_type,
            device=device
        )
    
    def create_mcts(self, backend: str) -> MCTS:
        """Create MCTS instance for specified backend"""
        config = self.create_mcts_config(backend)
        evaluator = self.create_evaluator(backend)
        game_interface = GameInterface(GameType.GOMOKU, board_size=self.board_size)
        
        return MCTS(config, evaluator, game_interface)
    
    def run_single_search_comparison(self, max_moves: int = 3) -> Dict[str, Any]:
        """Compare single MCTS search across backends"""
        results = {}
        
        # Create game interface
        game_interface = GameInterface(GameType.GOMOKU, board_size=self.board_size)
        initial_state = game_interface.create_initial_state()
        
        for backend in self.backends:
            if backend in ['gpu', 'hybrid'] and not torch.cuda.is_available():
                logger.warning(f"Skipping {backend} backend - CUDA not available")
                continue
                
            try:
                logger.info(f"Testing {backend} backend...")
                mcts = self.create_mcts(backend)
                
                # Run search
                policy = mcts.search(initial_state, num_simulations=20)
                
                results[backend] = {
                    'policy': policy,
                    'policy_entropy': -np.sum(policy * np.log(policy + 1e-10)),
                    'top_move': np.argmax(policy),
                    'top_move_prob': np.max(policy),
                    'num_nonzero_moves': np.sum(policy > 0),
                    'success': True,
                    'error': None
                }
                
                logger.info(f"✓ {backend} backend completed successfully")
                
            except Exception as e:
                logger.error(f"✗ {backend} backend failed: {e}")
                results[backend] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def run_multi_move_comparison(self, num_moves: int = 5) -> Dict[str, Any]:
        """Compare multi-move games across backends"""
        results = {}
        
        # Create consistent game sequence
        game_interface = GameInterface(GameType.GOMOKU, board_size=self.board_size)
        
        for backend in self.backends:
            if backend in ['gpu', 'hybrid'] and not torch.cuda.is_available():
                continue
                
            try:
                logger.info(f"Testing {backend} backend multi-move game...")
                mcts = self.create_mcts(backend)
                
                state = game_interface.create_initial_state()
                move_sequence = []
                policy_sequence = []
                
                for move_num in range(num_moves):
                    # Get policy
                    policy = mcts.search(state, num_simulations=10)
                    top_move = np.argmax(policy)
                    
                    move_sequence.append(top_move)
                    policy_sequence.append(policy.copy())
                    
                    # Apply move
                    state = game_interface.apply_move(state, top_move)
                    
                    if game_interface.is_terminal(state):
                        break
                
                results[backend] = {
                    'move_sequence': move_sequence,
                    'policy_sequence': policy_sequence,
                    'final_state_terminal': game_interface.is_terminal(state),
                    'success': True,
                    'error': None
                }
                
                logger.info(f"✓ {backend} backend multi-move completed")
                
            except Exception as e:
                logger.error(f"✗ {backend} backend multi-move failed: {e}")
                results[backend] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def run_stress_test_comparison(self, num_searches: int = 10) -> Dict[str, Any]:
        """Compare backend behavior under stress"""
        results = {}
        
        game_interface = GameInterface(GameType.GOMOKU, board_size=self.board_size)
        initial_state = game_interface.create_initial_state()
        
        for backend in self.backends:
            if backend in ['gpu', 'hybrid'] and not torch.cuda.is_available():
                continue
                
            try:
                logger.info(f"Stress testing {backend} backend...")
                mcts = self.create_mcts(backend)
                
                policies = []
                search_times = []
                
                for i in range(num_searches):
                    import time
                    start_time = time.time()
                    
                    policy = mcts.search(initial_state, num_simulations=15)
                    
                    end_time = time.time()
                    
                    policies.append(policy)
                    search_times.append(end_time - start_time)
                
                # Analyze consistency
                policy_consistency = []
                for i in range(1, len(policies)):
                    # Compare top-5 moves consistency
                    prev_top5 = set(np.argsort(policies[i-1])[-5:])
                    curr_top5 = set(np.argsort(policies[i])[-5:])
                    overlap = len(prev_top5.intersection(curr_top5)) / 5.0
                    policy_consistency.append(overlap)
                
                results[backend] = {
                    'num_searches': num_searches,
                    'avg_search_time': np.mean(search_times),
                    'search_time_std': np.std(search_times),
                    'avg_policy_consistency': np.mean(policy_consistency),
                    'min_consistency': np.min(policy_consistency),
                    'success': True,
                    'error': None
                }
                
                logger.info(f"✓ {backend} stress test completed")
                
            except Exception as e:
                logger.error(f"✗ {backend} stress test failed: {e}")
                results[backend] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results


class TestCPUVsGPUComparison:
    """Test class for CPU vs GPU backend comparison"""
    
    @pytest.fixture
    def comparator(self):
        return BackendComparator()
    
    def test_single_search_consistency(self, comparator):
        """Test that CPU and GPU backends produce reasonable policies"""
        results = comparator.run_single_search_comparison()
        
        # Ensure at least CPU backend worked
        assert 'cpu' in results
        assert results['cpu']['success'], f"CPU backend failed: {results['cpu'].get('error')}"
        
        cpu_result = results['cpu']
        
        # Basic sanity checks for CPU
        assert len(cpu_result['policy']) == 225  # 15x15 board
        assert np.abs(np.sum(cpu_result['policy']) - 1.0) < 1e-6  # Policy sums to 1
        assert cpu_result['num_nonzero_moves'] > 0  # Some moves have non-zero probability
        assert 0 <= cpu_result['top_move'] < 225  # Valid move index
        
        # Compare with other backends if available
        for backend in ['gpu', 'hybrid']:
            if backend in results and results[backend]['success']:
                backend_result = results[backend]
                
                # Both should have same policy length
                assert len(backend_result['policy']) == len(cpu_result['policy'])
                
                # Both should have similar number of non-zero moves (within reasonable range)
                nonzero_diff = abs(backend_result['num_nonzero_moves'] - cpu_result['num_nonzero_moves'])
                assert nonzero_diff <= 50, f"Large difference in non-zero moves: CPU={cpu_result['num_nonzero_moves']}, {backend}={backend_result['num_nonzero_moves']}"
                
                # Entropy should be in similar range
                entropy_diff = abs(backend_result['policy_entropy'] - cpu_result['policy_entropy'])
                assert entropy_diff <= 2.0, f"Large entropy difference: CPU={cpu_result['policy_entropy']}, {backend}={backend_result['policy_entropy']}"
                
                logger.info(f"✓ CPU vs {backend} consistency check passed")
    
    def test_multi_move_consistency(self, comparator):
        """Test multi-move game consistency across backends"""
        results = comparator.run_multi_move_comparison(3)
        
        # Ensure CPU backend worked
        assert 'cpu' in results
        assert results['cpu']['success'], f"CPU multi-move failed: {results['cpu'].get('error')}"
        
        cpu_result = results['cpu']
        
        # Validate CPU results
        assert len(cpu_result['move_sequence']) > 0
        assert len(cpu_result['policy_sequence']) == len(cpu_result['move_sequence'])
        
        # All moves should be valid
        for move in cpu_result['move_sequence']:
            assert 0 <= move < 225
        
        # Compare with other backends
        for backend in ['gpu', 'hybrid']:
            if backend in results and results[backend]['success']:
                backend_result = results[backend]
                
                # Same number of moves (since we use deterministic top move selection)
                assert len(backend_result['move_sequence']) == len(cpu_result['move_sequence'])
                
                # Check if moves are similar (allow some variation due to numerical differences)
                move_similarity = sum(1 for a, b in zip(cpu_result['move_sequence'], backend_result['move_sequence']) if a == b)
                similarity_ratio = move_similarity / len(cpu_result['move_sequence'])
                
                # At least 50% of moves should be similar
                assert similarity_ratio >= 0.5, f"Low move similarity between CPU and {backend}: {similarity_ratio}"
                
                logger.info(f"✓ CPU vs {backend} multi-move similarity: {similarity_ratio:.2f}")
    
    def test_stress_test_comparison(self, comparator):
        """Test backend consistency under repeated searches"""
        results = comparator.run_stress_test_comparison(5)
        
        # Ensure CPU backend worked
        assert 'cpu' in results
        assert results['cpu']['success'], f"CPU stress test failed: {results['cpu'].get('error')}"
        
        cpu_result = results['cpu']
        
        # Validate CPU stress test results
        assert cpu_result['avg_search_time'] > 0
        assert cpu_result['avg_policy_consistency'] >= 0  # Can be 0 if completely random
        assert cpu_result['avg_policy_consistency'] <= 1  # Can't be more than 100%
        
        logger.info(f"CPU backend stress test - Avg time: {cpu_result['avg_search_time']:.3f}s, Consistency: {cpu_result['avg_policy_consistency']:.3f}")
        
        # Compare with other backends
        for backend in ['gpu', 'hybrid']:
            if backend in results and results[backend]['success']:
                backend_result = results[backend]
                
                # Performance comparison (allow wide range due to hardware differences)
                time_ratio = backend_result['avg_search_time'] / cpu_result['avg_search_time']
                logger.info(f"{backend} vs CPU time ratio: {time_ratio:.2f}")
                
                # Consistency should be similar
                consistency_diff = abs(backend_result['avg_policy_consistency'] - cpu_result['avg_policy_consistency'])
                assert consistency_diff <= 0.3, f"Large consistency difference: CPU={cpu_result['avg_policy_consistency']}, {backend}={backend_result['avg_policy_consistency']}"
                
                logger.info(f"✓ CPU vs {backend} stress test consistency check passed")
    
    def test_error_handling_comparison(self, comparator):
        """Test error handling consistency across backends"""
        # This test would check how backends handle edge cases
        # For now, just ensure no backend crashes unexpectedly
        
        results = comparator.run_single_search_comparison()
        
        # Count successful backends
        successful_backends = [b for b, r in results.items() if r.get('success', False)]
        
        # At least CPU should work
        assert 'cpu' in successful_backends, "CPU backend must work"
        
        # Log which backends worked
        logger.info(f"Successful backends: {successful_backends}")
        
        # If any backend failed, log the errors
        for backend, result in results.items():
            if not result.get('success', False):
                logger.warning(f"{backend} backend failed: {result.get('error', 'Unknown error')}")


def test_backend_comparison_integration():
    """Integration test for backend comparison"""
    comparator = BackendComparator()
    
    # Run a basic comparison
    results = comparator.run_single_search_comparison()
    
    # Should have at least CPU results
    assert 'cpu' in results
    assert results['cpu']['success']
    
    logger.info("✓ Backend comparison integration test passed")


if __name__ == "__main__":
    # Run tests manually if not using pytest
    pytest.main([__file__, "-v", "--tb=short"])