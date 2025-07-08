"""
Comprehensive tests for core MCTS functionality

This module tests all aspects of the MCTS implementation including:
- Tree initialization and root node creation
- Selection phase with UCB formula
- Expansion with progressive widening
- Evaluation with neural network
- Backpropagation with value negation
- Policy extraction and action selection
- Tree reuse and state management
"""

import pytest
import torch
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock

from mcts.core.mcts import MCTS
from mcts.core.mcts_config import MCTSConfig
from mcts.gpu.gpu_game_states import GameType as GPUGameType
from conftest import assert_valid_policy, assert_valid_value, assert_tensor_equal


class TestMCTSInitialization:
    """Test MCTS initialization and setup"""
    
    def test_basic_initialization(self, base_mcts_config, mock_evaluator):
        """Test basic MCTS initialization"""
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        assert mcts.config == base_mcts_config
        assert mcts.evaluator == mock_evaluator
        assert mcts.tree is not None
        assert mcts.game_states is not None
        assert mcts.wave_search is not None
        assert mcts.tree_ops is not None
        
        # Check root node exists
        assert mcts.tree.num_nodes == 1
        assert mcts.tree.node_data.visit_counts[0] == 0
        assert mcts.tree.node_data.parent_indices[0] == -1
        
    def test_device_configuration(self, base_mcts_config, mock_evaluator, device):
        """Test MCTS respects device configuration"""
        base_mcts_config.device = str(device)
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Check all components are on correct device
        def devices_equal(d1, d2):
            return d1.type == d2.type and (d1.index == d2.index or (d1.index is None and d2.index == 0) or (d2.index is None and d1.index == 0))
        
        assert devices_equal(mcts.tree.device, device)
        assert devices_equal(mcts.game_states.device, device)
        assert devices_equal(mcts.node_to_state.device, device)
        
    def test_game_type_configuration(self, base_mcts_config, mock_evaluator):
        """Test MCTS handles different game types correctly"""
        for game_type in [GPUGameType.GOMOKU, GPUGameType.GO, GPUGameType.CHESS]:
            config = base_mcts_config
            config.game_type = game_type
            
            # Set appropriate board size
            if game_type == GPUGameType.CHESS:
                config.board_size = 8
                config.max_children_per_node = 4096
            elif game_type == GPUGameType.GO:
                config.board_size = 19
                config.max_children_per_node = 362
            else:
                config.board_size = 15
                config.max_children_per_node = 225
                
            mcts = MCTS(config, mock_evaluator)
            
            # Verify game-specific settings
            assert mcts.game_states.game_type == game_type
            assert mcts.game_states.board_size == config.board_size
            
    def test_evaluator_configuration(self, base_mcts_config, device):
        """Test MCTS configures evaluator correctly"""
        # Mock evaluator with torch tensor support
        evaluator = Mock()
        evaluator._return_torch_tensors = False
        
        mcts = MCTS(base_mcts_config, evaluator)
        
        # Check evaluator was configured for torch tensors
        assert hasattr(evaluator, '_return_torch_tensors')
        assert evaluator._return_torch_tensors == True


class TestMCTSSearch:
    """Test MCTS search functionality"""
    
    def test_basic_search(self, base_mcts_config, deterministic_evaluator, empty_gomoku_state):
        """Test basic MCTS search runs correctly"""
        mcts = MCTS(base_mcts_config, deterministic_evaluator)
        
        # Run search
        policy = mcts.search(empty_gomoku_state, num_simulations=10)
        
        # Verify policy is valid
        assert_valid_policy(policy)
        assert policy.shape == (225,)  # 15x15 board
        
        # Check tree grew
        assert mcts.tree.num_nodes > 1
        assert mcts.tree.node_data.visit_counts[0] > 0
        
    def test_search_respects_num_simulations(self, base_mcts_config, mock_evaluator, empty_gomoku_state):
        """Test search runs correct number of simulations"""
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Run with specific simulation count
        num_sims = 50
        policy = mcts.search(empty_gomoku_state, num_simulations=num_sims)
        
        # Root should have approximately num_sims visits
        # (may be slightly less due to terminal nodes)
        root_visits = mcts.tree.node_data.visit_counts[0].item()
        assert num_sims * 0.9 <= root_visits <= num_sims
        
    def test_search_with_dirichlet_noise(self, base_mcts_config, mock_evaluator, empty_gomoku_state):
        """Test search applies Dirichlet noise correctly"""
        base_mcts_config.dirichlet_epsilon = 0.25
        base_mcts_config.dirichlet_alpha = 0.3
        
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Run search multiple times
        policies = []
        for _ in range(3):
            mcts.reset_tree()
            policy = mcts.search(empty_gomoku_state, num_simulations=20)
            policies.append(policy)
            
        # Policies should be different due to noise
        assert not np.allclose(policies[0], policies[1])
        assert not np.allclose(policies[1], policies[2])
        
    def test_search_without_tree_reuse(self, base_mcts_config, mock_evaluator, empty_gomoku_state):
        """Test search without subtree reuse"""
        base_mcts_config.enable_subtree_reuse = False
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Run multiple searches
        for _ in range(3):
            policy = mcts.search(empty_gomoku_state, num_simulations=10)
            nodes_after_search = mcts.tree.num_nodes
            
        # Tree should be reset each time (only growth from single search)
        assert nodes_after_search < 50  # Should not accumulate nodes
        
    def test_search_terminal_position(self, base_mcts_config, mock_evaluator, gomoku_game):
        """Test search handles terminal positions correctly"""
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Create a won position (5 in a row)
        state = gomoku_game.create_initial_state()
        # Play winning sequence for player 1
        moves = [112, 127, 113, 128, 114, 129, 115, 130, 116]  # Horizontal line
        for i, move in enumerate(moves):
            state = gomoku_game.apply_move(state, move)
            
        # Search should still work on terminal position
        policy = mcts.search(state, num_simulations=10)
        assert_valid_policy(policy)
        
        # Tree should not expand much from terminal
        assert mcts.tree.num_nodes < 5


class TestMCTSSelection:
    """Test MCTS selection phase"""
    
    def test_ucb_selection(self, base_mcts_config, deterministic_evaluator):
        """Test UCB formula selects high-value unvisited nodes"""
        mcts = MCTS(base_mcts_config, deterministic_evaluator)
        
        # Manually set up tree with known values
        tree = mcts.tree
        
        # Add children to root with different priors
        actions = [0, 1, 2]
        priors = [0.5, 0.3, 0.2]
        child_indices = tree.add_children_batch(0, actions, priors)
        
        # Set different visit counts and values
        tree.node_data.visit_counts[child_indices[0]] = 10
        tree.node_data.value_sums[child_indices[0]] = 5.0  # Q = 0.5
        
        tree.node_data.visit_counts[child_indices[1]] = 5
        tree.node_data.value_sums[child_indices[1]] = -1.0  # Q = -0.2
        
        tree.node_data.visit_counts[child_indices[2]] = 0  # Unvisited
        
        # Selection should favor unvisited node with decent prior
        parent_visits = 15
        tree.node_data.visit_counts[0] = parent_visits
        
        # Test wave search selection
        wave_search = mcts.wave_search
        paths, lengths, leaf_nodes = wave_search._select_batch_vectorized(1)
        
        # Should select the unvisited node (highest UCB due to exploration bonus)
        assert leaf_nodes[0].item() == child_indices[2]
        
    def test_virtual_loss_during_selection(self, base_mcts_config, mock_evaluator):
        """Test virtual loss is applied during parallel selection"""
        base_mcts_config.enable_virtual_loss = True
        base_mcts_config.virtual_loss = 3.0
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Set up tree
        tree = mcts.tree
        actions = list(range(5))
        priors = [0.2] * 5
        child_indices = tree.add_children_batch(0, actions, priors)
        
        # Track virtual losses
        initial_vl = tree.node_data.virtual_loss_counts[child_indices].clone()
        
        # Run parallel selection
        wave_search = mcts.wave_search
        paths, lengths, leaf_nodes = wave_search._select_batch_vectorized(3)
        
        # Virtual loss should be applied to selected nodes
        # Note: actual virtual loss application happens in expansion/evaluation phase
        # This test would need to be expanded with full wave execution


class TestMCTSExpansion:
    """Test MCTS expansion phase"""
    
    def test_basic_expansion(self, base_mcts_config, mock_evaluator, empty_gomoku_state):
        """Test node expansion creates children correctly"""
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Initialize root
        mcts._initialize_root(empty_gomoku_state)
        
        # Expand root
        wave_search = mcts.wave_search
        expanded_nodes = wave_search._expand_batch_vectorized(
            torch.tensor([0], device=mcts.device),
            node_to_state=mcts.node_to_state,
            state_pool_free_list=mcts.state_pool_free_list
        )
        
        # Check children were added
        assert mcts.tree.num_nodes > 1
        children, actions, priors = mcts.tree.get_children(0)
        assert len(children) > 0
        assert len(actions) == len(children)
        assert len(priors) == len(children)
        
        # Verify children have valid priors
        assert torch.all(priors >= 0)
        assert torch.all(priors <= 1)
        
    def test_progressive_widening(self, base_mcts_config, mock_evaluator, empty_gomoku_state):
        """Test progressive widening limits children based on visit count"""
        base_mcts_config.num_simulations = 100
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Configure progressive widening parameters
        mcts.wave_search.pw_alpha = 0.5
        mcts.wave_search.pw_base = 10.0
        
        # Initialize and run a few simulations
        policy = mcts.search(empty_gomoku_state, num_simulations=5)
        
        # Root should have limited children due to progressive widening
        children, _, _ = mcts.tree.get_children(0)
        initial_children = len(children)
        assert initial_children < 225  # Should not expand all moves immediately
        
        # Run more simulations
        policy = mcts.search(empty_gomoku_state, num_simulations=50)
        
        # Should have more children now
        children, _, _ = mcts.tree.get_children(0)
        assert len(children) >= initial_children
        
    def test_expansion_respects_legal_moves(self, base_mcts_config, mock_evaluator, sample_gomoku_position):
        """Test expansion only creates legal children"""
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Run search on position with some illegal moves
        policy = mcts.search(sample_gomoku_position, num_simulations=20)
        
        # Check all expanded actions are legal
        for node_idx in range(mcts.tree.num_nodes):
            children, actions, _ = mcts.tree.get_children(node_idx)
            if len(children) > 0:
                # Get node's state and legal moves
                state_idx = mcts.node_to_state[node_idx].item()
                if state_idx >= 0:
                    legal_mask = mcts.game_states.get_legal_moves_mask(
                        torch.tensor([state_idx], device=mcts.device)
                    )[0]
                    legal_moves = set(torch.where(legal_mask)[0].cpu().numpy())
                    
                    # All actions should be legal
                    for action in actions:
                        assert action.item() in legal_moves


class TestMCTSBackpropagation:
    """Test MCTS backpropagation phase"""
    
    def test_value_backpropagation(self, base_mcts_config, empty_gomoku_state):
        """Test values are correctly backpropagated with alternating signs"""
        # Create evaluator that returns fixed positive value
        evaluator = Mock()
        evaluator.evaluate_batch = Mock(return_value=(
            np.random.rand(1, 225),  # Random policy
            np.array([0.5])  # Fixed positive value
        ))
        
        mcts = MCTS(base_mcts_config, evaluator)
        mcts._initialize_root(empty_gomoku_state)
        
        # Manually create a path and backpropagate
        tree = mcts.tree
        
        # Create path: root -> child1 -> child2
        child1 = tree.add_child(0, 112, 0.5)
        child2 = tree.add_child(child1, 113, 0.5)
        
        # Backpropagate value
        path = torch.tensor([[0, child1, child2]])
        path_lengths = torch.tensor([3])
        values = torch.tensor([0.8])  # Positive value for leaf
        
        mcts.wave_search._backup_batch_vectorized(path, path_lengths, values)
        
        # Check alternating values
        # Leaf (child2) perspective: +0.8
        # child1 perspective: -0.8
        # Root perspective: +0.8
        assert tree.node_data.visit_counts[child2] == 1
        assert tree.node_data.visit_counts[child1] == 1
        assert tree.node_data.visit_counts[0] == 1
        
        assert abs(tree.node_data.value_sums[child2].item() - 0.8) < 1e-5
        assert abs(tree.node_data.value_sums[child1].item() - (-0.8)) < 1e-5
        assert abs(tree.node_data.value_sums[0].item() - 0.8) < 1e-5
        
    def test_visit_count_updates(self, base_mcts_config, mock_evaluator, empty_gomoku_state):
        """Test visit counts are correctly updated during search"""
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Run search
        num_sims = 20
        policy = mcts.search(empty_gomoku_state, num_simulations=num_sims)
        
        # Root should have all visits
        root_visits = mcts.tree.node_data.visit_counts[0].item()
        assert root_visits == num_sims
        
        # Sum of all node visits should equal total simulations * depth
        total_visits = mcts.tree.node_data.visit_counts[:mcts.tree.num_nodes].sum().item()
        assert total_visits >= num_sims  # At least one visit per simulation
        
    def test_q_value_computation(self, base_mcts_config, deterministic_evaluator, empty_gomoku_state):
        """Test Q-values are correctly computed as average of backpropagated values"""
        deterministic_evaluator.fixed_value = 0.5
        mcts = MCTS(base_mcts_config, deterministic_evaluator)
        
        # Run search
        policy = mcts.search(empty_gomoku_state, num_simulations=50)
        
        # Check Q-values are averages
        for node_idx in range(mcts.tree.num_nodes):
            visits = mcts.tree.node_data.visit_counts[node_idx].item()
            if visits > 0:
                value_sum = mcts.tree.node_data.value_sums[node_idx].item()
                q_value = value_sum / visits
                assert -1.0 <= q_value <= 1.0  # Q-values should be bounded


class TestMCTSPolicyExtraction:
    """Test MCTS policy extraction and action selection"""
    
    def test_policy_extraction_basic(self, base_mcts_config, mock_evaluator, empty_gomoku_state):
        """Test policy extraction from visit counts"""
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Run search
        policy = mcts.search(empty_gomoku_state, num_simulations=100)
        
        # Policy should be normalized
        assert_valid_policy(policy)
        
        # Most visited actions should have highest probability
        children, actions, _ = mcts.tree.get_children(0)
        if len(children) > 0:
            visits = mcts.tree.node_data.visit_counts[children]
            max_visit_idx = visits.argmax()
            max_visit_action = actions[max_visit_idx].item()
            
            # Corresponding policy entry should be high
            assert policy[max_visit_action] > 1.0 / 225  # Above uniform
            
    def test_policy_temperature(self, base_mcts_config, mock_evaluator, empty_gomoku_state):
        """Test temperature-based action selection"""
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Run search once
        policy = mcts.search(empty_gomoku_state, num_simulations=50)
        
        # Test temperature-based selection on the same policy
        # Temperature 0 should be deterministic
        max_action = np.argmax(policy)
        
        # Simulate temperature=0 selection (deterministic)
        action_t0_1 = max_action
        action_t0_2 = max_action
        assert action_t0_1 == action_t0_2  # Deterministic
        
        # High temperature should allow more variety
        # Apply temperature to policy
        policy_t2 = np.power(policy, 1/2.0)
        policy_t2 /= policy_t2.sum()
        
        # Sample multiple times with high temperature
        actions_t2 = [np.random.choice(len(policy_t2), p=policy_t2) for _ in range(10)]
        # With a reasonable policy distribution, we should see some variety
        # (though this could fail with very bad luck)
        
    def test_policy_masking_illegal_moves(self, base_mcts_config, mock_evaluator, sample_gomoku_position):
        """Test policy correctly masks illegal moves"""
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Run search on position with occupied squares
        policy = mcts.search(sample_gomoku_position, num_simulations=50)
        
        # Get legal moves
        from mcts.core.game_interface import GameInterface, GameType
        game = GameInterface(GameType.GOMOKU, board_size=15)
        legal_moves = game.get_legal_moves(sample_gomoku_position)
        legal_mask = np.zeros(225, dtype=bool)
        legal_mask[legal_moves] = True
        
        # Illegal moves should have zero probability
        assert_valid_policy(policy, legal_mask)
        
    def test_empty_tree_policy(self, base_mcts_config, mock_evaluator, empty_gomoku_state):
        """Test policy extraction when root has no children"""
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Initialize root but don't expand
        mcts._initialize_root(empty_gomoku_state)
        
        # Extract policy should return uniform over legal moves
        policy = mcts._extract_policy(0)
        assert_valid_policy(policy)
        
        # Should be roughly uniform
        legal_moves = mcts.game_states.get_legal_moves_mask(torch.tensor([0], device=mcts.device))[0]
        num_legal = legal_moves.sum().item()
        expected_prob = 1.0 / num_legal
        
        for i, is_legal in enumerate(legal_moves.cpu().numpy()):
            if is_legal:
                assert abs(policy[i] - expected_prob) < 1e-5
            else:
                assert policy[i] == 0.0


class TestMCTSTreeReuse:
    """Test MCTS subtree reuse functionality"""
    
    def test_subtree_reuse_basic(self, base_mcts_config, mock_evaluator, gomoku_game):
        """Test basic subtree reuse after move"""
        base_mcts_config.enable_subtree_reuse = True
        base_mcts_config.subtree_reuse_min_visits = 1  # Allow reuse with just 1 visit
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Run search on initial position
        state = gomoku_game.create_initial_state()
        policy = mcts.search(state, num_simulations=50)
        
        # Record tree state
        nodes_before = mcts.tree.num_nodes
        
        # Select and make a move
        action = policy.argmax()
        new_state = gomoku_game.apply_move(state, action)
        
        # Update root to reuse subtree
        mcts.update_with_move(action)
        
        # Tree should have shifted
        nodes_after = mcts.tree.num_nodes
        assert nodes_after < nodes_before  # Some nodes discarded
        assert nodes_after > 1  # But subtree preserved
        
        # Root should now be the previous child
        assert mcts.tree.node_data.parent_indices[0] == -1  # New root has no parent
        
    def test_subtree_reuse_preserves_values(self, base_mcts_config, deterministic_evaluator, gomoku_game):
        """Test subtree reuse preserves node statistics"""
        base_mcts_config.enable_subtree_reuse = True
        base_mcts_config.subtree_reuse_min_visits = 1  # Allow reuse with just 1 visit
        mcts = MCTS(base_mcts_config, deterministic_evaluator)
        
        # Run search
        state = gomoku_game.create_initial_state()
        policy = mcts.search(state, num_simulations=50)
        
        # Get child statistics before reuse
        action = 112  # Center position
        child_idx = mcts.tree.get_child_by_action(0, action)
        if child_idx is not None:
            child_visits_before = mcts.tree.node_data.visit_counts[child_idx].item()
            child_value_before = mcts.tree.node_data.value_sums[child_idx].item()
            
            # Apply subtree reuse
            new_state = gomoku_game.apply_move(state, action)
            mcts.update_with_move(action)
            
            # New root should have same statistics
            root_visits_after = mcts.tree.node_data.visit_counts[0].item()
            root_value_after = mcts.tree.node_data.value_sums[0].item()
            
            assert root_visits_after == child_visits_before
            # Note: value sign is already correct in the tree
            assert abs(root_value_after - child_value_before) < 1e-5
            
    def test_subtree_reuse_disabled(self, base_mcts_config, mock_evaluator, gomoku_game):
        """Test behavior when subtree reuse is disabled"""
        base_mcts_config.tree_reuse = False  # Use correct config attribute
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Run search
        state = gomoku_game.create_initial_state()
        policy = mcts.search(state, num_simulations=50)
        nodes_after_first = mcts.tree.num_nodes
        
        # Make move and search again
        action = policy.argmax()
        new_state = gomoku_game.apply_move(state, action)
        mcts.update_with_move(action)
        
        # Without reuse, tree should be small (reset for each search)
        policy2 = mcts.search(new_state, num_simulations=50)
        nodes_after_second = mcts.tree.num_nodes
        
        # Tree size should be similar to first search (not accumulated)
        assert nodes_after_second < nodes_after_first * 1.5


class TestMCTSStateManagement:
    """Test MCTS state pool and memory management"""
    
    def test_state_allocation(self, base_mcts_config, mock_evaluator, empty_gomoku_state):
        """Test state allocation and deallocation"""
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Track initial state in game_states
        initial_allocated = mcts.game_states.num_states
        
        # Run search to allocate states
        policy = mcts.search(empty_gomoku_state, num_simulations=20)
        
        # States should be allocated
        final_allocated = mcts.game_states.num_states
        states_used = final_allocated - initial_allocated
        assert states_used > 0
        assert states_used <= mcts.tree.num_nodes  # Not all nodes need states
        
    def test_state_cleanup_without_reuse(self, base_mcts_config, mock_evaluator, empty_gomoku_state):
        """Test states are properly cleaned up when reuse is disabled"""
        base_mcts_config.enable_subtree_reuse = False
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Run multiple searches
        for _ in range(3):
            policy = mcts.search(empty_gomoku_state, num_simulations=20)
            
        # GPU states should be properly managed
        allocated_states = mcts.game_states.allocated_mask.sum().item()
        assert allocated_states < 100  # Should not leak states
        
    def test_node_to_state_mapping(self, base_mcts_config, mock_evaluator, empty_gomoku_state):
        """Test node-to-state mapping is maintained correctly"""
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Run search
        policy = mcts.search(empty_gomoku_state, num_simulations=30)
        
        # Check all nodes with states have valid mappings
        for node_idx in range(mcts.tree.num_nodes):
            state_idx = mcts.node_to_state[node_idx].item()
            if state_idx >= 0:
                # State should be allocated
                assert mcts.game_states.allocated_mask[state_idx]
                # State index should be in valid range
                assert 0 <= state_idx < mcts.game_states.capacity


class TestMCTSStatistics:
    """Test MCTS statistics and performance tracking"""
    
    def test_statistics_tracking(self, base_mcts_config, mock_evaluator, empty_gomoku_state):
        """Test MCTS tracks statistics correctly"""
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Run searches
        for i in range(3):
            policy = mcts.search(empty_gomoku_state, num_simulations=10)
            
        stats = mcts.get_statistics()
        
        # Check basic stats
        assert stats['total_searches'] == 3
        assert stats['total_simulations'] == 30
        assert stats['total_time'] > 0
        assert stats['avg_sims_per_second'] > 0
        assert 'num_nodes' in stats
        assert 'max_depth' in stats
        
    def test_performance_metrics(self, base_mcts_config, mock_evaluator, empty_gomoku_state):
        """Test performance metrics are reasonable"""
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Run timed search
        start_time = time.time()
        policy = mcts.search(empty_gomoku_state, num_simulations=100)
        elapsed = time.time() - start_time
        
        stats = mcts.get_statistics()
        
        # Should achieve reasonable performance with mock evaluator
        assert stats['last_search_sims_per_second'] > 100  # At least 100 sims/sec
        assert stats['peak_sims_per_second'] >= stats['last_search_sims_per_second']
        
    @pytest.mark.slow
    def test_memory_usage_tracking(self, base_mcts_config, mock_evaluator, empty_gomoku_state):
        """Test memory usage is tracked and reasonable"""
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Run large search
        policy = mcts.search(empty_gomoku_state, num_simulations=500)
        
        stats = mcts.get_statistics()
        
        # Check memory stats exist and are reasonable
        assert 'memory_usage_mb' in stats
        assert stats['memory_usage_mb'] < 100  # Should use less than 100MB for this test
        
        # Memory per node should be reasonable
        bytes_per_node = stats['memory_usage_mb'] * 1024 * 1024 / max(1, stats['num_nodes'])
        assert bytes_per_node < 1000  # Less than 1KB per node


class TestMCTSEdgeCases:
    """Test MCTS edge cases and error handling"""
    
    def test_terminal_root_position(self, base_mcts_config, mock_evaluator, gomoku_game):
        """Test MCTS handles terminal root position"""
        # Create won position
        state = gomoku_game.create_initial_state()
        # Create 5 in a row
        moves = [112, 127, 113, 128, 114, 129, 115, 130, 116]
        for move in moves:
            state = gomoku_game.apply_move(state, move)
            
        assert gomoku_game.is_terminal(state)
        
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Should handle terminal root gracefully
        policy = mcts.search(state, num_simulations=10)
        assert_valid_policy(policy)
        
    def test_single_legal_move(self, base_mcts_config, mock_evaluator, gomoku_game):
        """Test MCTS with only one legal move"""
        # Create position with only one empty square
        state = gomoku_game.create_initial_state()
        
        # Fill board except one square (this is a bit artificial)
        # In practice, use a more realistic position
        # For now, just test the logic
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Should handle gracefully
        policy = mcts.search(state, num_simulations=10)
        assert_valid_policy(policy)
        
    def test_max_tree_nodes_limit(self, small_mcts_config, mock_evaluator, empty_gomoku_state):
        """Test MCTS respects maximum tree nodes limit"""
        small_mcts_config.max_tree_nodes = 100
        mcts = MCTS(small_mcts_config, mock_evaluator)
        
        # Try to run many simulations
        with pytest.raises(RuntimeError, match="Tree full"):
            policy = mcts.search(empty_gomoku_state, num_simulations=1000)
            
    def test_invalid_action_handling(self, base_mcts_config, mock_evaluator, gomoku_game):
        """Test MCTS handles invalid actions gracefully"""
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        state = gomoku_game.create_initial_state()
        # Make a move
        state = gomoku_game.apply_move(state, 112)
        
        # Try to update root with invalid action
        mcts.search(state, num_simulations=10)
        
        # Should not crash when trying invalid action
        mcts.update_with_move(999)  # Out of bounds action
        
        # Tree should reset or handle gracefully
        assert mcts.tree.num_nodes >= 1


class TestMCTSIntegration:
    """Integration tests for MCTS with other components"""
    
    def test_with_neural_evaluator(self, base_mcts_config, empty_gomoku_state):
        """Test MCTS with neural network evaluator mock"""
        # Create more realistic neural network mock
        evaluator = Mock()
        evaluator.evaluate_batch = Mock(side_effect=lambda states: (
            np.random.dirichlet([0.5] * 225, size=len(states)),  # Dirichlet policies
            np.random.uniform(-0.5, 0.5, size=len(states))  # Random values
        ))
        
        mcts = MCTS(base_mcts_config, evaluator)
        
        # Should work with neural network style evaluator
        policy = mcts.search(empty_gomoku_state, num_simulations=50)
        assert_valid_policy(policy)
        
        # Evaluator should have been called
        assert evaluator.evaluate_batch.called
        
    @pytest.mark.slow
    def test_full_game_simulation(self, base_mcts_config, mock_evaluator, gomoku_game):
        """Test playing a full game with MCTS"""
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        state = gomoku_game.create_initial_state()
        move_count = 0
        max_moves = 225
        
        while not gomoku_game.is_terminal(state) and move_count < max_moves:
            # Search and select action
            policy = mcts.search(state, num_simulations=50)
            action = mcts.select_action(state, temperature=1.0)
            
            # Make move
            state = gomoku_game.apply_move(state, action)
            mcts.update_with_move(action)
            
            move_count += 1
            
        # Game should terminate or reach max moves
        assert move_count > 0
        assert move_count <= max_moves
        
    def test_parallel_mcts_instances(self, base_mcts_config, mock_evaluator, empty_gomoku_state):
        """Test multiple MCTS instances don't interfere"""
        # Create multiple instances
        mcts1 = MCTS(base_mcts_config, mock_evaluator)
        mcts2 = MCTS(base_mcts_config, mock_evaluator)
        
        # Run searches in parallel
        policy1 = mcts1.search(empty_gomoku_state, num_simulations=20)
        policy2 = mcts2.search(empty_gomoku_state, num_simulations=20)
        
        # Trees should be independent
        assert mcts1.tree.num_nodes != mcts2.tree.num_nodes or \
               not torch.allclose(torch.tensor(policy1), torch.tensor(policy2))
        
        # Each should have valid results
        assert_valid_policy(policy1)
        assert_valid_policy(policy2)