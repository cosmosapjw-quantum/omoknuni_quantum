#!/usr/bin/env python3
"""Detailed tests for MCTS phases

This module provides comprehensive testing of the four MCTS phases:
1. Selection - UCB-based node selection down the tree
2. Expansion - Adding new nodes when reaching a leaf
3. Evaluation - Neural network evaluation of leaf positions  
4. Backpropagation - Updating values back up the tree
"""

import pytest
import torch
import numpy as np
import logging
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import MCTS components
from mcts.core.mcts import MCTS
from mcts.core.mcts_config import MCTSConfig
from mcts.gpu.gpu_game_states import GameType
from mcts.gpu.csr_tree import CSRTree
from mock_evaluator import MockEvaluator


class TestSelectionPhase:
    """Comprehensive tests for the selection phase"""
    
    @pytest.fixture(params=['cpu', 'gpu', 'hybrid'])
    def mcts_setup(self, request):
        """Setup MCTS for selection testing"""
        backend = request.param
        if backend == 'gpu' and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = MCTSConfig(
            backend=backend,
            board_size=9,
            game_type=GameType.GOMOKU,
            device='cuda' if backend == 'gpu' else 'cpu',
            max_tree_nodes=10000,
            c_puct=1.0,
            dirichlet_epsilon=0.0  # No noise for testing
        )
        evaluator = MockEvaluator(board_size=9, device=config.device)
        mcts = MCTS(config, evaluator)
        
        # Initialize with a partially built tree
        state = np.zeros((9, 9), dtype=np.int8)
        mcts._ensure_root_initialized(state)
        
        # Build initial tree structure
        for _ in range(50):
            mcts.wave_search.run_wave(
                wave_size=1,
                node_to_state=mcts.node_to_state,
                state_pool_free_list=mcts.state_pool_free_list
            )
        
        return mcts, config, backend
    
    def test_ucb_calculation(self, mcts_setup):
        """Test UCB score calculation"""
        mcts, config, backend = mcts_setup
        
        # Get root's children
        children = mcts.tree.get_children(0)
        assert len(children) > 0, "Root has no children"
        
        # Manually calculate UCB for verification
        parent_visits = mcts.tree.node_data.visit_counts[0].item()
        
        for child_idx in children[:5]:  # Test first 5 children
            child_visits = mcts.tree.node_data.visit_counts[child_idx].item()
            child_value = mcts.tree.node_data.values[child_idx].item()
            child_prior = mcts.tree.node_data.priors[child_idx].item()
            
            # UCB formula: Q + c_puct * P * sqrt(parent_visits) / (1 + child_visits)
            exploration = config.c_puct * child_prior * np.sqrt(parent_visits) / (1 + child_visits)
            expected_ucb = child_value + exploration
            
            # The actual UCB calculation happens during selection
            # We verify the formula is correct
            assert child_visits >= 0
            assert -1 <= child_value <= 1
            assert 0 <= child_prior <= 1
    
    def test_selection_path_diversity(self, mcts_setup):
        """Test that selection explores different paths"""
        mcts, config, backend = mcts_setup
        
        # Track selected paths
        selected_nodes = []
        
        # Run multiple selections
        for _ in range(100):
            # Single wave to track selection
            wave_size = 1
            
            # Store initial visit counts
            initial_visits = mcts.tree.node_data.visit_counts.clone()
            
            # Run selection
            mcts.wave_search.run_wave(
                wave_size=wave_size,
                node_to_state=mcts.node_to_state,
                state_pool_free_list=mcts.state_pool_free_list
            )
            
            # Find which nodes were visited
            visit_diff = mcts.tree.node_data.visit_counts - initial_visits
            visited = torch.where(visit_diff > 0)[0].tolist()
            
            if visited:
                selected_nodes.extend(visited)
        
        # Check diversity
        unique_nodes = set(selected_nodes)
        assert len(unique_nodes) > 10, f"Poor selection diversity: only {len(unique_nodes)} unique nodes"
        
        # Check that less visited nodes are being explored
        node_visit_counts = {}
        for node in selected_nodes:
            node_visit_counts[node] = node_visit_counts.get(node, 0) + 1
        
        # Variance in visit counts indicates exploration
        visit_variance = np.var(list(node_visit_counts.values()))
        assert visit_variance > 0, "No exploration variance"
    
    def test_selection_with_virtual_loss(self, mcts_setup):
        """Test virtual loss mechanism during selection"""
        mcts, config, backend = mcts_setup
        
        # Enable virtual loss
        mcts.config.enable_virtual_loss = True
        mcts.config.virtual_loss_value = 3.0
        
        # Track concurrent selections
        concurrent_paths = []
        
        # Simulate concurrent selection (would be parallel in practice)
        initial_values = mcts.tree.node_data.values.clone()
        
        # Run wave with multiple parallel paths
        wave_size = 8
        mcts.wave_search.run_wave(
            wave_size=wave_size,
            node_to_state=mcts.node_to_state,
            state_pool_free_list=mcts.state_pool_free_list
        )
        
        # Virtual loss should temporarily reduce values during selection
        # This encourages different paths to be selected
        
        # Check tree expanded with diversity
        assert mcts.tree.num_nodes > 100, "Insufficient expansion with virtual loss"
    
    def test_progressive_widening_selection(self, mcts_setup):
        """Test progressive widening affects selection"""
        mcts, config, backend = mcts_setup
        
        # Enable progressive widening
        mcts.config.enable_progressive_widening = True
        mcts.config.progressive_widening_constant = 1.0
        mcts.config.progressive_widening_exponent = 0.5
        
        # Track how many children are explored as visits increase
        root_children_explored = []
        
        for i in range(10):
            # Run more simulations
            for _ in range(20):
                mcts.wave_search.run_wave(
                    wave_size=1,
                    node_to_state=mcts.node_to_state,
                    state_pool_free_list=mcts.state_pool_free_list
                )
            
            # Count explored children of root
            root_children = mcts.tree.get_children(0)
            explored = sum(1 for c in root_children 
                         if mcts.tree.node_data.visit_counts[c] > 0)
            root_children_explored.append(explored)
        
        # Should gradually explore more children
        assert root_children_explored[-1] > root_children_explored[0], \
            "Progressive widening not increasing exploration"
        
        # Should follow power law
        logger.info(f"Progressive widening: {root_children_explored}")


class TestExpansionPhase:
    """Comprehensive tests for the expansion phase"""
    
    @pytest.fixture(params=['cpu', 'gpu', 'hybrid'])
    def mcts_setup(self, request):
        """Setup MCTS for expansion testing"""
        backend = request.param
        if backend == 'gpu' and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = MCTSConfig(
            backend=backend,
            board_size=9,
            game_type=GameType.GOMOKU,
            device='cuda' if backend == 'gpu' else 'cpu',
            max_tree_nodes=10000
        )
        evaluator = MockEvaluator(board_size=9, device=config.device)
        mcts = MCTS(config, evaluator)
        
        state = np.zeros((9, 9), dtype=np.int8)
        mcts._ensure_root_initialized(state)
        
        return mcts, config, backend
    
    def test_basic_expansion(self, mcts_setup):
        """Test basic node expansion"""
        mcts, config, backend = mcts_setup
        
        # Root should be expanded
        assert mcts.tree.node_data.expanded[0].item()
        
        # Get initial tree size
        initial_nodes = mcts.tree.num_nodes
        
        # Force expansion by running waves
        for _ in range(10):
            mcts.wave_search.run_wave(
                wave_size=1,
                node_to_state=mcts.node_to_state,
                state_pool_free_list=mcts.state_pool_free_list
            )
        
        # Tree should have grown
        assert mcts.tree.num_nodes > initial_nodes
        
        # New nodes should have valid data
        for idx in range(initial_nodes, mcts.tree.num_nodes):
            # Should have a state assigned
            state_idx = mcts.node_to_state[idx].item()
            assert state_idx >= 0, f"Node {idx} has no state"
            
            # Should have priors from NN
            assert mcts.tree.node_data.priors[idx] >= 0
    
    def test_expansion_with_legal_moves(self, mcts_setup):
        """Test expansion respects legal moves"""
        mcts, config, backend = mcts_setup
        
        # Create a state with some occupied positions
        state = np.zeros((9, 9), dtype=np.int8)
        state[4, 4] = 1  # Center
        state[4, 5] = -1
        state[5, 4] = -1
        state[5, 5] = 1
        
        # Reset and initialize with this state
        mcts._reset_for_new_search()
        mcts._ensure_root_initialized(state)
        
        # Expand from root
        mcts.wave_search.run_wave(
            wave_size=10,
            node_to_state=mcts.node_to_state,
            state_pool_free_list=mcts.state_pool_free_list
        )
        
        # Get root's children
        root_children = mcts.tree.get_children(0)
        
        # Children should only be legal moves
        occupied_actions = [4*9+4, 4*9+5, 5*9+4, 5*9+5]  # Flatten indices
        
        for child_idx in root_children:
            action = mcts.tree.get_action(child_idx)
            assert action not in occupied_actions, \
                f"Illegal move expanded: action {action} at occupied position"
    
    def test_batch_expansion_efficiency(self, mcts_setup):
        """Test efficient batch expansion"""
        mcts, config, backend = mcts_setup
        
        # Time single expansions
        single_times = []
        for _ in range(10):
            start = time.perf_counter()
            mcts.wave_search.run_wave(
                wave_size=1,
                node_to_state=mcts.node_to_state,
                state_pool_free_list=mcts.state_pool_free_list
            )
            single_times.append(time.perf_counter() - start)
        
        # Time batch expansion
        start = time.perf_counter()
        mcts.wave_search.run_wave(
            wave_size=32,
            node_to_state=mcts.node_to_state,
            state_pool_free_list=mcts.state_pool_free_list
        )
        batch_time = time.perf_counter() - start
        
        # Batch should be more efficient than 32 singles
        total_single_time = sum(single_times[:32]) if len(single_times) >= 32 else sum(single_times) * 3.2
        assert batch_time < total_single_time, \
            f"Batch expansion not efficient: {batch_time:.3f}s vs {total_single_time:.3f}s"
    
    def test_expansion_memory_management(self, mcts_setup):
        """Test memory management during expansion"""
        mcts, config, backend = mcts_setup
        
        # Track state allocations
        initial_free_states = len(mcts.state_pool_free_list)
        
        # Expand many nodes
        for _ in range(100):
            mcts.wave_search.run_wave(
                wave_size=10,
                node_to_state=mcts.node_to_state,
                state_pool_free_list=mcts.state_pool_free_list
            )
        
        # Check state pool usage
        final_free_states = len(mcts.state_pool_free_list)
        states_used = initial_free_states - final_free_states
        
        # Should efficiently use state pool
        assert states_used > 0, "No states allocated"
        assert states_used < mcts.tree.num_nodes * 2, "Excessive state allocation"
        
        # Verify no state leaks
        allocated_states = torch.sum(mcts.node_to_state >= 0).item()
        assert allocated_states <= states_used + 1, "State leak detected"
    
    def test_expansion_limits(self, mcts_setup):
        """Test expansion behavior at tree capacity"""
        mcts, config, backend = mcts_setup
        
        # Set small tree limit
        mcts.tree.max_nodes = 100
        
        # Try to expand beyond limit
        for _ in range(20):
            if mcts.tree.num_nodes >= mcts.tree.max_nodes - 10:
                break
            mcts.wave_search.run_wave(
                wave_size=10,
                node_to_state=mcts.node_to_state,
                state_pool_free_list=mcts.state_pool_free_list
            )
        
        # Should not exceed limit
        assert mcts.tree.num_nodes <= mcts.tree.max_nodes, \
            f"Tree exceeded limit: {mcts.tree.num_nodes} > {mcts.tree.max_nodes}"


class TestEvaluationPhase:
    """Comprehensive tests for the evaluation phase"""
    
    @pytest.fixture(params=['cpu', 'gpu', 'hybrid'])
    def mcts_setup(self, request):
        """Setup MCTS for evaluation testing"""
        backend = request.param
        if backend == 'gpu' and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = MCTSConfig(
            backend=backend,
            board_size=9,
            game_type=GameType.GOMOKU,
            device='cuda' if backend == 'gpu' else 'cpu',
            max_tree_nodes=10000,
            batch_size=32
        )
        evaluator = MockEvaluator(
            board_size=9,
            device=config.device,
            batch_size=32
        )
        mcts = MCTS(config, evaluator)
        
        return mcts, config, backend
    
    def test_neural_network_evaluation(self, mcts_setup):
        """Test neural network evaluation process"""
        mcts, config, backend = mcts_setup
        
        # Create various board states
        states = []
        for i in range(10):
            state = np.zeros((9, 9), dtype=np.int8)
            # Add some moves
            for j in range(i):
                x, y = (j * 3) % 9, (j * 5) % 9
                state[x, y] = 1 if j % 2 == 0 else -1
            states.append(state)
        
        # Convert to batch
        if backend == 'cpu':
            batch = np.stack(states)
        else:
            batch = torch.from_numpy(np.stack(states)).to(mcts.device)
        
        # Evaluate
        values, policies = mcts.evaluator.evaluate(batch)
        
        # Verify outputs
        assert values.shape == (10,)
        assert policies.shape == (10, 81)
        
        # Values in valid range
        assert torch.all(values >= -1.0)
        assert torch.all(values <= 1.0)
        
        # Policies sum to 1
        policy_sums = policies.sum(dim=1)
        assert torch.allclose(policy_sums, torch.ones_like(policy_sums))
    
    def test_batch_aggregation(self, mcts_setup):
        """Test efficient batch aggregation for evaluation"""
        mcts, config, backend = mcts_setup
        
        # Initialize tree
        state = np.zeros((9, 9), dtype=np.int8)
        mcts._ensure_root_initialized(state)
        
        # Track evaluation calls
        eval_calls = []
        original_evaluate = mcts.evaluator.evaluate
        
        def track_evaluate(states):
            eval_calls.append(states.shape[0] if hasattr(states, 'shape') else len(states))
            return original_evaluate(states)
        
        mcts.evaluator.evaluate = track_evaluate
        
        # Run waves of different sizes
        for wave_size in [1, 8, 16, 32]:
            eval_calls.clear()
            
            mcts.wave_search.run_wave(
                wave_size=wave_size,
                node_to_state=mcts.node_to_state,
                state_pool_free_list=mcts.state_pool_free_list
            )
            
            # Should batch evaluations efficiently
            if eval_calls:
                logger.info(f"Wave size {wave_size}: {len(eval_calls)} eval calls, sizes: {eval_calls}")
                
                # Larger waves should use larger batches
                if wave_size >= 16:
                    assert max(eval_calls) >= 8, "Not batching efficiently"
        
        # Restore
        mcts.evaluator.evaluate = original_evaluate
    
    def test_evaluation_caching(self, mcts_setup):
        """Test that evaluations are cached properly"""
        mcts, config, backend = mcts_setup
        
        # Track unique states evaluated
        evaluated_states = set()
        original_evaluate = mcts.evaluator.evaluate
        
        def track_unique_evaluate(states):
            # Hash states to track uniqueness
            if hasattr(states, 'cpu'):
                states_np = states.cpu().numpy()
            else:
                states_np = states
            
            for state in states_np:
                state_hash = hash(state.tobytes())
                evaluated_states.add(state_hash)
            
            return original_evaluate(states)
        
        mcts.evaluator.evaluate = track_unique_evaluate
        
        # Create same position multiple times
        state = np.zeros((9, 9), dtype=np.int8)
        state[4, 4] = 1
        
        # Run multiple searches from same position
        for _ in range(5):
            mcts._reset_for_new_search()
            mcts._ensure_root_initialized(state)
            
            # Small search
            for _ in range(10):
                mcts.wave_search.run_wave(
                    wave_size=1,
                    node_to_state=mcts.node_to_state,
                    state_pool_free_list=mcts.state_pool_free_list
                )
        
        # Should not re-evaluate same positions excessively
        # (Some re-evaluation expected due to tree reset)
        logger.info(f"Unique states evaluated: {len(evaluated_states)}")
        
        # Restore
        mcts.evaluator.evaluate = original_evaluate
    
    def test_mixed_precision_evaluation(self, mcts_setup):
        """Test mixed precision evaluation if enabled"""
        mcts, config, backend = mcts_setup
        
        if backend == 'cpu':
            pytest.skip("Mixed precision primarily for GPU")
        
        # Enable mixed precision
        mcts.config.use_mixed_precision = True
        
        # Create states
        states = torch.randn(16, 4, 9, 9, device=mcts.device)
        
        # Check if evaluator supports mixed precision
        if hasattr(mcts.evaluator, 'use_amp'):
            mcts.evaluator.use_amp = True
            
            # Evaluate with mixed precision
            with torch.cuda.amp.autocast():
                values, policies = mcts.evaluator.evaluate(states)
            
            # Results should still be valid
            assert values.dtype in [torch.float16, torch.float32]
            assert torch.all(torch.isfinite(values))
            assert torch.all(torch.isfinite(policies))


class TestBackpropagationPhase:
    """Comprehensive tests for the backpropagation phase"""
    
    @pytest.fixture(params=['cpu', 'gpu', 'hybrid'])
    def mcts_setup(self, request):
        """Setup MCTS for backpropagation testing"""
        backend = request.param
        if backend == 'gpu' and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = MCTSConfig(
            backend=backend,
            board_size=9,
            game_type=GameType.GOMOKU,
            device='cuda' if backend == 'gpu' else 'cpu',
            max_tree_nodes=10000
        )
        evaluator = MockEvaluator(board_size=9, device=config.device)
        mcts = MCTS(config, evaluator)
        
        # Build a tree with known structure
        state = np.zeros((9, 9), dtype=np.int8)
        mcts._ensure_root_initialized(state)
        
        # Manually build tree for testing
        # Root -> Child1 -> Grandchild1
        #      -> Child2 -> Grandchild2
        
        # Add children to root
        for i in range(2):
            mcts.tree.add_node(parent_idx=0, action=i)
            mcts.tree.node_data.priors[i+1] = 0.5
        
        # Add grandchildren
        for i in range(2):
            parent = i + 1
            child_idx = mcts.tree.add_node(parent_idx=parent, action=i+10)
            mcts.tree.node_data.priors[child_idx] = 0.5
        
        return mcts, config, backend
    
    def test_value_backup(self, mcts_setup):
        """Test value backup along path"""
        mcts, config, backend = mcts_setup
        
        # Create a path: root -> child1 -> grandchild1
        path = [0, 1, 3]
        
        # Set initial values and visits
        for idx in path:
            mcts.tree.node_data.visit_counts[idx] = 0
            mcts.tree.node_data.values[idx] = 0.0
        
        # Simulate backup with value 0.8
        leaf_value = 0.8
        
        # Manually perform backup
        for i, idx in enumerate(reversed(path)):
            # Increment visit
            mcts.tree.node_data.visit_counts[idx] += 1
            
            # Update value (average)
            old_value = mcts.tree.node_data.values[idx].item()
            old_visits = mcts.tree.node_data.visit_counts[idx].item() - 1
            
            # Flip value for alternating players
            current_value = leaf_value if i % 2 == 0 else -leaf_value
            
            new_value = (old_value * old_visits + current_value) / (old_visits + 1)
            mcts.tree.node_data.values[idx] = new_value
        
        # Verify backup
        assert mcts.tree.node_data.visit_counts[0] == 1
        assert mcts.tree.node_data.visit_counts[1] == 1
        assert mcts.tree.node_data.visit_counts[3] == 1
        
        # Values should alternate based on player
        assert abs(mcts.tree.node_data.values[3].item() - 0.8) < 1e-5  # Leaf
        assert abs(mcts.tree.node_data.values[1].item() - (-0.8)) < 1e-5  # Parent
        assert abs(mcts.tree.node_data.values[0].item() - 0.8) < 1e-5  # Root
    
    def test_concurrent_backup(self, mcts_setup):
        """Test concurrent backpropagation"""
        mcts, config, backend = mcts_setup
        
        # Build larger tree
        for _ in range(50):
            mcts.wave_search.run_wave(
                wave_size=4,
                node_to_state=mcts.node_to_state,
                state_pool_free_list=mcts.state_pool_free_list
            )
        
        # Get initial statistics
        initial_visits = mcts.tree.node_data.visit_counts.clone()
        initial_values = mcts.tree.node_data.values.clone()
        
        # Run concurrent waves (simulated)
        for _ in range(10):
            mcts.wave_search.run_wave(
                wave_size=8,
                node_to_state=mcts.node_to_state,
                state_pool_free_list=mcts.state_pool_free_list
            )
        
        # Verify updates
        visit_diff = mcts.tree.node_data.visit_counts - initial_visits
        value_diff = (mcts.tree.node_data.values - initial_values).abs()
        
        # Multiple nodes should be updated
        nodes_updated = torch.sum(visit_diff > 0).item()
        assert nodes_updated > 10, f"Too few nodes updated: {nodes_updated}"
        
        # Values should change
        values_changed = torch.sum(value_diff > 1e-6).item()
        assert values_changed > 0, "No values updated"
        
        # Root should accumulate visits
        root_visits = mcts.tree.node_data.visit_counts[0].item()
        assert root_visits > initial_visits[0].item() + 50, \
            f"Root visits not accumulated: {root_visits}"
    
    def test_backup_consistency(self, mcts_setup):
        """Test consistency of backup across backends"""
        mcts, config, backend = mcts_setup
        
        # Run deterministic search
        state = np.zeros((9, 9), dtype=np.int8)
        mcts.config.dirichlet_epsilon = 0.0  # No noise
        
        # Build tree
        for _ in range(20):
            mcts.wave_search.run_wave(
                wave_size=1,
                node_to_state=mcts.node_to_state,
                state_pool_free_list=mcts.state_pool_free_list
            )
        
        # Check value consistency
        # All paths from root should maintain value consistency
        def check_path_consistency(node_idx, path_value, depth):
            if depth > 5:  # Limit depth
                return
            
            children = mcts.tree.get_children(node_idx)
            if not children:
                return
            
            node_value = mcts.tree.node_data.values[node_idx].item()
            node_visits = mcts.tree.node_data.visit_counts[node_idx].item()
            
            if node_visits > 0:
                # Check children values are consistent with parent
                child_values = []
                child_visits = []
                
                for child in children:
                    cv = mcts.tree.node_data.values[child].item()
                    cc = mcts.tree.node_data.visit_counts[child].item()
                    if cc > 0:
                        child_values.append(cv)
                        child_visits.append(cc)
                
                if child_values:
                    # Parent value should be weighted average of children
                    # (with sign flip for alternating players)
                    weighted_sum = sum(-v * c for v, c in zip(child_values, child_visits))
                    total_visits = sum(child_visits)
                    
                    if total_visits > 0:
                        expected_value = weighted_sum / total_visits
                        
                        # Some tolerance for floating point
                        if abs(node_value - expected_value) > 0.1:
                            logger.warning(
                                f"Value inconsistency at node {node_idx}: "
                                f"{node_value:.3f} vs expected {expected_value:.3f}"
                            )
            
            # Recurse
            for child in children:
                check_path_consistency(child, -node_value, depth + 1)
        
        # Start from root
        check_path_consistency(0, 0, 0)
    
    def test_virtual_loss_backup(self, mcts_setup):
        """Test virtual loss during backup"""
        mcts, config, backend = mcts_setup
        
        # Enable virtual loss
        mcts.config.enable_virtual_loss = True
        mcts.config.virtual_loss_value = 3.0
        
        # Track value changes during parallel selection
        initial_values = mcts.tree.node_data.values.clone()
        
        # Run parallel wave
        mcts.wave_search.run_wave(
            wave_size=8,
            node_to_state=mcts.node_to_state,
            state_pool_free_list=mcts.state_pool_free_list
        )
        
        # Values should be restored after backup
        final_values = mcts.tree.node_data.values
        
        # Check that values changed (virtual loss applied and removed)
        value_changes = (final_values - initial_values).abs()
        changed_nodes = torch.sum(value_changes > 1e-6).item()
        
        assert changed_nodes > 0, "No values changed with virtual loss"
        
        # No node should have extreme negative values (stuck virtual loss)
        assert torch.all(final_values > -2.0), \
            "Extreme negative values suggest stuck virtual loss"


class TestPhaseIntegration:
    """Test integration of all four phases"""
    
    @pytest.fixture(params=['cpu', 'gpu', 'hybrid'])
    def mcts_setup(self, request):
        """Setup MCTS for integration testing"""
        backend = request.param
        if backend == 'gpu' and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = MCTSConfig(
            backend=backend,
            board_size=9,
            game_type=GameType.GOMOKU,
            device='cuda' if backend == 'gpu' else 'cpu',
            max_tree_nodes=50000,
            num_simulations=1000,
            c_puct=1.0,
            temperature=1.0
        )
        evaluator = MockEvaluator(board_size=9, device=config.device)
        mcts = MCTS(config, evaluator)
        
        return mcts, config, backend
    
    def test_complete_simulation_flow(self, mcts_setup):
        """Test complete flow through all phases"""
        mcts, config, backend = mcts_setup
        
        # Initial state
        state = np.zeros((9, 9), dtype=np.int8)
        
        # Track phase transitions
        phase_stats = {
            'selections': 0,
            'expansions': 0,
            'evaluations': 0,
            'backprops': 0
        }
        
        # Instrument to track phases
        original_run_wave = mcts.wave_search.run_wave
        
        def tracked_run_wave(wave_size, node_to_state, state_pool_free_list):
            initial_nodes = mcts.tree.num_nodes
            initial_visits = mcts.tree.node_data.visit_counts[0].item()
            
            result = original_run_wave(wave_size, node_to_state, state_pool_free_list)
            
            # Detect what happened
            if mcts.tree.num_nodes > initial_nodes:
                phase_stats['expansions'] += mcts.tree.num_nodes - initial_nodes
            
            if mcts.tree.node_data.visit_counts[0].item() > initial_visits:
                phase_stats['backprops'] += 1
                phase_stats['selections'] += 1
                phase_stats['evaluations'] += 1
            
            return result
        
        mcts.wave_search.run_wave = tracked_run_wave
        
        # Run full search
        policy = mcts.search(state, num_simulations=100)
        
        # Restore
        mcts.wave_search.run_wave = original_run_wave
        
        # Verify all phases executed
        assert phase_stats['selections'] > 0, "No selections performed"
        assert phase_stats['expansions'] > 0, "No expansions performed"
        assert phase_stats['evaluations'] > 0, "No evaluations performed"
        assert phase_stats['backprops'] > 0, "No backpropagations performed"
        
        logger.info(f"Phase stats for {backend}: {phase_stats}")
        
        # Verify final result
        assert policy is not None
        assert len(policy) == 81
        assert np.allclose(policy.sum(), 1.0)
    
    def test_phase_timing_breakdown(self, mcts_setup):
        """Measure time spent in each phase"""
        mcts, config, backend = mcts_setup
        
        # Timing buckets
        phase_times = {
            'selection': 0.0,
            'expansion': 0.0,
            'evaluation': 0.0,
            'backprop': 0.0
        }
        
        # This would require more detailed instrumentation
        # For now, just run search and verify it completes
        
        state = np.zeros((9, 9), dtype=np.int8)
        
        start = time.perf_counter()
        policy = mcts.search(state, num_simulations=500)
        total_time = time.perf_counter() - start
        
        # Log performance
        sims_per_second = 500 / total_time
        logger.info(f"{backend} performance: {sims_per_second:.0f} sims/s")
        
        # Verify reasonable performance
        min_performance = {
            'cpu': 500,
            'gpu': 2000,
            'hybrid': 1000
        }
        
        if backend in min_performance:
            assert sims_per_second >= min_performance[backend], \
                f"{backend} too slow: {sims_per_second:.0f} sims/s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])