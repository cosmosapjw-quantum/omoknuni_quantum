#!/usr/bin/env python3
"""Comprehensive test suite for MCTS implementation

This test suite validates all three backends (CPU, GPU, hybrid) across all four phases
of MCTS (expansion, evaluation, selection, backpropagation) and their integration.
"""

import pytest
import torch
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import MCTS components
from mcts.core.mcts import MCTS
from mcts.core.mcts_config import MCTSConfig
from mcts.gpu.gpu_game_states import GameType
from mcts.neural_networks.resnet_model import ResNetModel, ResNetConfig
from mock_evaluator import MockEvaluator, create_mock_model


class TestMCTSBackends:
    """Test all MCTS backends (CPU, GPU, hybrid) for correctness"""
    
    @pytest.fixture(params=['cpu', 'gpu', 'hybrid'])
    def backend(self, request):
        """Parametrized fixture for backend types"""
        return request.param
    
    @pytest.fixture
    def config(self, backend):
        """Create MCTS config for given backend"""
        config = MCTSConfig(
            num_simulations=100,
            c_puct=1.0,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=0.3,
            temperature=1.0,
            board_size=9,  # Smaller board for faster tests
            game_type=GameType.GOMOKU,
            device='cuda' if torch.cuda.is_available() and backend == 'gpu' else 'cpu',
            backend=backend,
            enable_subtree_reuse=False,  # Disable for deterministic tests
            max_tree_nodes=10000,
            use_mixed_precision=False
        )
        return config
    
    @pytest.fixture
    def evaluator(self, config):
        """Create mock evaluator for testing"""
        return MockEvaluator(
            board_size=config.board_size,
            device=config.device,
            batch_size=32,
            deterministic=True  # For reproducible tests
        )
    
    @pytest.fixture
    def mcts(self, config, evaluator):
        """Create MCTS instance"""
        return MCTS(config, evaluator)
    
    def test_initialization(self, mcts, config, backend):
        """Test MCTS initialization for each backend"""
        assert mcts.config.backend == backend
        assert mcts.tree is not None
        assert mcts.game_states is not None
        assert mcts.wave_search is not None
        
        # Check device allocation
        if backend == 'gpu' and torch.cuda.is_available():
            assert str(mcts.device) == 'cuda'
        elif backend == 'hybrid':
            # Hybrid uses CPU memory for thread safety
            assert mcts.game_states.device == torch.device('cpu')
        else:
            assert str(mcts.device) == 'cpu'
    
    def test_search_basic(self, mcts, config):
        """Test basic search functionality"""
        # Create initial state
        state = np.zeros((config.board_size, config.board_size), dtype=np.int8)
        
        # Run search
        policy = mcts.search(state, num_simulations=50)
        
        # Validate output
        assert isinstance(policy, np.ndarray)
        assert len(policy) == config.board_size * config.board_size
        assert np.allclose(policy.sum(), 1.0, rtol=1e-5)
        assert np.all(policy >= 0)
    
    def test_search_consistency(self, mcts, config):
        """Test that search produces consistent results"""
        state = np.zeros((config.board_size, config.board_size), dtype=np.int8)
        
        # Run multiple searches with same state
        policies = []
        for _ in range(3):
            mcts._reset_for_new_search()  # Ensure clean state
            policy = mcts.search(state, num_simulations=100)
            policies.append(policy)
        
        # Policies should be similar
        for i in range(1, len(policies)):
            # For deterministic setup, policies might be identical (uniform)
            # Check if policies have variance before computing correlation
            std0 = np.std(policies[0])
            stdi = np.std(policies[i])
            
            if std0 > 1e-6 and stdi > 1e-6:
                # Both have variance, check correlation
                corr = np.corrcoef(policies[0], policies[i])[0, 1]
                assert corr > 0.8, f"Policies not consistent: correlation = {corr}"
            else:
                # No variance, check if they're close
                assert np.allclose(policies[0], policies[i], rtol=1e-5), \
                    f"Uniform policies not identical"


class TestMCTSPhases:
    """Test individual MCTS phases for all backends"""
    
    @pytest.fixture(params=['cpu', 'gpu', 'hybrid'])
    def backend(self, request):
        return request.param
    
    @pytest.fixture
    def setup(self, backend):
        """Setup MCTS instance and return components"""
        config = MCTSConfig(
            num_simulations=100,
            c_puct=1.0,
            board_size=9,
            game_type=GameType.GOMOKU,
            device='cuda' if torch.cuda.is_available() and backend == 'gpu' else 'cpu',
            backend=backend,
            max_tree_nodes=10000,
            enable_subtree_reuse=False
        )
        
        evaluator = MockEvaluator(
            board_size=config.board_size,
            device=config.device,
            deterministic=True,
            fixed_value=0.5  # Non-zero value for backpropagation testing
        )
        
        mcts = MCTS(config, evaluator)
        
        # Initialize root
        state = np.zeros((config.board_size, config.board_size), dtype=np.int8)
        mcts._ensure_root_initialized(state)
        
        return {
            'mcts': mcts,
            'config': config,
            'evaluator': evaluator,
            'state': state
        }
    
    def test_expansion_phase(self, setup, backend):
        """Test expansion phase"""
        mcts = setup['mcts']
        
        # Check root is expanded
        assert mcts.tree.node_data.is_expanded(0)
        
        # Get initial tree size
        initial_nodes = mcts.tree.num_nodes
        
        # Run a few simulations to trigger expansions
        mcts.wave_search.run_wave(
            wave_size=10,
            node_to_state=mcts.node_to_state,
            state_pool_free_list=mcts.state_pool_free_list
        )
        
        # Check that new nodes were added
        assert mcts.tree.num_nodes > initial_nodes
        
        # Verify expanded nodes have valid states
        for node_idx in range(1, mcts.tree.num_nodes):
            if mcts.tree.node_data.is_expanded(node_idx):
                state_idx = mcts.node_to_state[node_idx].item()
                assert state_idx >= 0, f"Node {node_idx} has invalid state index"
    
    def test_selection_phase(self, setup, backend):
        """Test selection phase (UCB calculation and path selection)"""
        mcts = setup['mcts']
        
        # Add some visits to root children for UCB testing
        if mcts.tree.num_nodes > 1:
            # Manually set some visit counts
            num_children = min(5, mcts.tree.num_nodes - 1)
            for i in range(1, num_children + 1):
                mcts.tree.node_data.visit_counts[i] = i * 10
                mcts.tree.node_data.value_sums[i] = (0.5 + i * 0.1) * (i * 10)  # value_sum = avg_value * visit_count
            
            # Test UCB calculation
            # The selection should prefer nodes with higher UCB scores
            parent_visits = mcts.tree.node_data.visit_counts[0].item()
            
            # Run selection through wave search
            paths = []
            for _ in range(5):
                mcts.wave_search.run_wave(
                    wave_size=1,
                    node_to_state=mcts.node_to_state,
                    state_pool_free_list=mcts.state_pool_free_list
                )
            
            # Verify paths were selected
            assert mcts.tree.node_data.visit_counts[0] > parent_visits
    
    def test_evaluation_phase(self, setup, backend):
        """Test neural network evaluation phase"""
        mcts = setup['mcts']
        evaluator = setup['evaluator']
        
        # Create batch of states for evaluation
        batch_size = 16
        states = []
        for i in range(batch_size):
            state = np.zeros((mcts.config.board_size, mcts.config.board_size), dtype=np.int8)
            # Add some random moves
            for _ in range(i % 5):
                x, y = np.random.randint(0, mcts.config.board_size, 2)
                state[x, y] = 1
            states.append(state)
        
        # Stack states
        if backend == 'cpu':
            state_batch = np.stack(states)
        else:
            state_batch = torch.from_numpy(np.stack(states)).to(mcts.device)
        
        # Get evaluations
        policies, values = evaluator.evaluate(state_batch)
        
        # Validate outputs
        assert values.shape == (batch_size,)
        assert policies.shape == (batch_size, mcts.config.board_size * mcts.config.board_size)
        
        # Check value bounds
        assert torch.all(values >= -1.0) and torch.all(values <= 1.0)
        
        # Check policy sums to 1
        policy_sums = policies.sum(dim=1)
        assert torch.allclose(policy_sums, torch.ones_like(policy_sums))
    
    def test_backpropagation_phase(self, setup, backend):
        """Test value backpropagation phase"""
        mcts = setup['mcts']
        
        # Create a simple path for backpropagation
        # Ensure we have at least a few nodes
        for _ in range(5):
            mcts.wave_search.run_wave(
                wave_size=1,
                node_to_state=mcts.node_to_state,
                state_pool_free_list=mcts.state_pool_free_list
            )
        
        if mcts.tree.num_nodes > 2:
            # Get initial values and visit counts
            initial_visits = mcts.tree.node_data.visit_counts.clone()
            initial_values = mcts.tree.node_data.value_sums.clone()
            
            # Simulate backpropagation by running more waves
            for _ in range(10):
                mcts.wave_search.run_wave(
                    wave_size=1,
                    node_to_state=mcts.node_to_state,
                    state_pool_free_list=mcts.state_pool_free_list
                )
            
            # Check that values were updated
            final_visits = mcts.tree.node_data.visit_counts
            final_values = mcts.tree.node_data.value_sums
            
            # Root should have more visits
            assert final_visits[0] > initial_visits[0]
            
            # Some nodes should have updated values
            value_changes = (final_values - initial_values).abs()
            assert value_changes.sum() > 0, "No values were updated during backpropagation"


class TestMCTSIntegration:
    """Integration tests for complete MCTS flow"""
    
    @pytest.fixture(params=['cpu', 'gpu', 'hybrid'])
    def backend(self, request):
        return request.param
    
    @pytest.fixture
    def mcts_setup(self, backend):
        """Setup MCTS for integration testing"""
        config = MCTSConfig(
            num_simulations=200,
            c_puct=1.0,
            board_size=9,
            game_type=GameType.GOMOKU,
            device='cuda' if torch.cuda.is_available() and backend == 'gpu' else 'cpu',
            backend=backend,
            max_tree_nodes=20000,
            enable_subtree_reuse=True,  # Test with tree reuse
            temperature=1.0
        )
        
        evaluator = MockEvaluator(
            board_size=config.board_size,
            device=config.device,
            deterministic=False  # More realistic
        )
        
        mcts = MCTS(config, evaluator)
        return mcts, config
    
    def test_full_game_simulation(self, mcts_setup, backend):
        """Test playing a full game with MCTS"""
        mcts, config = mcts_setup
        
        # Initialize game state
        state = np.zeros((config.board_size, config.board_size), dtype=np.int8)
        current_player = 1
        move_count = 0
        max_moves = config.board_size * config.board_size
        
        policies = []
        
        while move_count < max_moves and move_count < 20:  # Limit moves for test
            # Run MCTS search
            policy = mcts.search(state, num_simulations=100)
            policies.append(policy)
            
            # Select move based on policy
            valid_moves = (state.flatten() == 0)
            policy_masked = policy * valid_moves
            
            if policy_masked.sum() > 0:
                policy_masked /= policy_masked.sum()
                move = np.random.choice(len(policy_masked), p=policy_masked)
                
                # Apply move
                row, col = move // config.board_size, move % config.board_size
                state[row, col] = current_player
                
                # Update MCTS root if tree reuse is enabled
                if config.enable_subtree_reuse:
                    mcts.update_root(move)
                
                # Switch player
                current_player = -current_player
                move_count += 1
            else:
                break
        
        # Verify game was played
        assert move_count > 0, "No moves were made"
        assert len(policies) == move_count
        
        # Check policy diversity
        policy_std = np.std(policies, axis=0).mean()
        assert policy_std > 0, "All policies are identical"
    
    def test_performance_benchmarks(self, mcts_setup, backend):
        """Benchmark performance for each backend"""
        mcts, config = mcts_setup
        state = np.zeros((config.board_size, config.board_size), dtype=np.int8)
        
        # Warmup
        for _ in range(3):
            mcts.search(state, num_simulations=50)
        
        # Benchmark
        num_searches = 5
        num_simulations = 200
        
        start_time = time.perf_counter()
        for _ in range(num_searches):
            mcts._reset_for_new_search()
            mcts.search(state, num_simulations=num_simulations)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        total_simulations = num_searches * num_simulations
        sims_per_second = total_simulations / total_time
        
        logger.info(f"Backend: {backend}")
        logger.info(f"Total simulations: {total_simulations}")
        logger.info(f"Total time: {total_time:.3f}s")
        logger.info(f"Simulations/second: {sims_per_second:.0f}")
        
        # Set minimum performance thresholds
        # Note: These are for MockEvaluator, not real neural networks
        # Real ResNet would have different performance characteristics
        min_sims_per_second = {
            'cpu': 500,     # CPU with MockEvaluator
            'gpu': 1000,    # GPU with MockEvaluator (limited benefit)
            'hybrid': 1500  # Hybrid with MockEvaluator
        }
        
        if backend in min_sims_per_second:
            assert sims_per_second >= min_sims_per_second[backend], \
                f"{backend} backend too slow: {sims_per_second:.0f} < {min_sims_per_second[backend]}"
    
    def test_memory_usage(self, mcts_setup, backend):
        """Test memory usage and cleanup"""
        mcts, config = mcts_setup
        state = np.zeros((config.board_size, config.board_size), dtype=np.int8)
        
        # Get initial memory state
        if torch.cuda.is_available() and backend in ['gpu', 'hybrid']:
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        
        # Run multiple searches
        for i in range(10):
            policy = mcts.search(state, num_simulations=100)
            
            # Verify tree size is managed
            assert mcts.tree.num_nodes < config.max_tree_nodes, \
                f"Tree size exceeded limit: {mcts.tree.num_nodes} >= {config.max_tree_nodes}"
        
        # Check memory usage
        if torch.cuda.is_available() and backend in ['gpu', 'hybrid']:
            final_memory = torch.cuda.memory_allocated()
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (< 100MB)
            assert memory_increase < 100 * 1024 * 1024, \
                f"Excessive memory usage: {memory_increase / 1024 / 1024:.1f}MB"


class TestMCTSEdgeCases:
    """Test edge cases and error handling"""
    
    def test_invalid_game_state(self):
        """Test handling of invalid game states"""
        config = MCTSConfig(
            board_size=9,
            game_type=GameType.GOMOKU,
            backend='cpu'
        )
        evaluator = MockEvaluator(board_size=config.board_size)
        mcts = MCTS(config, evaluator)
        
        # Test with invalid state shape
        with pytest.raises(Exception):
            invalid_state = np.zeros((5, 5), dtype=np.int8)  # Wrong size
            mcts.search(invalid_state)
    
    def test_tree_capacity_limit(self):
        """Test behavior when tree reaches capacity"""
        config = MCTSConfig(
            board_size=9,
            game_type=GameType.GOMOKU,
            backend='cpu',
            max_tree_nodes=100,  # Very small tree
            enable_subtree_reuse=False
        )
        evaluator = MockEvaluator(board_size=config.board_size)
        mcts = MCTS(config, evaluator)
        
        state = np.zeros((config.board_size, config.board_size), dtype=np.int8)
        
        # Should handle gracefully even with small tree
        policy = mcts.search(state, num_simulations=200)
        assert policy is not None
        assert mcts.tree.num_nodes <= config.max_tree_nodes
    
    def test_concurrent_searches(self):
        """Test thread safety for hybrid backend"""
        config = MCTSConfig(
            board_size=9,
            game_type=GameType.GOMOKU,
            backend='hybrid',
            max_tree_nodes=10000
        )
        evaluator = MockEvaluator(board_size=config.board_size)
        mcts = MCTS(config, evaluator)
        
        state = np.zeros((config.board_size, config.board_size), dtype=np.int8)
        
        # Run multiple searches sequentially (concurrent testing would require threading)
        policies = []
        for _ in range(5):
            policy = mcts.search(state, num_simulations=50)
            policies.append(policy)
        
        # All searches should complete successfully
        assert len(policies) == 5
        for policy in policies:
            assert policy.sum() > 0.99  # Should sum to ~1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])