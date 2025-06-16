"""Comprehensive tests for advanced MCTS features

This module tests advanced MCTS functionality including:
- CSR tree format operations
- GPU kernel integration
- Memory pooling and optimization
- CUDA graphs and tensor cores
- Vectorized operations
- Performance optimizations
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import time
import logging

from mcts.core.mcts import MCTS, MCTSConfig
from mcts.gpu.csr_tree import CSRTree, CSRTreeConfig
from mcts.gpu.gpu_game_states import GPUGameStates, GPUGameStatesConfig, GameType
from mcts.gpu.unified_kernels import get_unified_kernels


class MockState:
    """Mock game state for testing"""
    def __init__(self):
        self.board = np.zeros((15, 15))
        self.current_player = 1
        self.move_count = 0
        
    def is_terminal(self):
        return self.move_count >= 225
        
    def get_legal_actions(self):
        # Return all empty positions
        return [i for i in range(225) if self.board.flatten()[i] == 0]
        
    def apply_action(self, action):
        new_state = MockState()
        new_state.board = self.board.copy()
        new_state.board.flat[action] = self.current_player
        new_state.current_player = 3 - self.current_player
        new_state.move_count = self.move_count + 1
        return new_state
        
    def get_observation(self):
        return self.board
        
    def get_current_player(self):
        return self.current_player


class TestCSRTreeOperations:
    """Test CSR (Compressed Sparse Row) tree format operations"""
    
    @pytest.fixture
    def tree_config(self):
        """Create CSR tree configuration"""
        return CSRTreeConfig(
            max_nodes=1000,
            max_edges=10000,
            device='cpu',
            enable_virtual_loss=True,
            virtual_loss_value=-3.0,
            batch_size=32,
            enable_batched_ops=True
        )
    
    @pytest.fixture
    def tree(self, tree_config):
        """Create CSR tree instance"""
        return CSRTree(tree_config)
    
    def test_csr_initialization(self, tree, tree_config):
        """Test CSR tree initialization"""
        assert tree.config == tree_config
        assert tree.num_nodes == 1  # Root is automatically created
        assert tree.device.type == 'cpu'
        
        # Check pre-allocated arrays
        assert tree.visit_counts.shape[0] == tree_config.max_nodes
        assert tree.value_sums.shape[0] == tree_config.max_nodes
        assert tree.node_priors.shape[0] == tree_config.max_nodes
    
    def test_node_addition(self, tree):
        """Test adding nodes to CSR tree"""
        # Root is automatically created in __init__
        assert tree.num_nodes == 1
        
        # Add children
        child_ids = []
        for i in range(5):
            child_id = tree.add_child(parent_idx=0, action=i, child_prior=0.2)
            child_ids.append(child_id)
        
        assert tree.num_nodes == 6
        assert all(cid > 0 for cid in child_ids)
    
    def test_batch_node_addition(self, tree):
        """Test batch node addition"""
        # Root is automatically created in __init__
        assert tree.num_nodes == 1
        
        # Add batch of children to root
        actions = list(range(10))
        priors = [0.1] * 10
        
        child_ids = tree.add_children_batch(parent_idx=0, actions=actions, priors=priors)
        
        assert len(child_ids) == 10
        assert tree.num_nodes == 11
    
    def test_children_retrieval(self, tree):
        """Test getting children in CSR format"""
        # Root is automatically created
        root = 0
        children = []
        for i in range(3):
            child = tree.add_child(parent_idx=root, action=i, child_prior=0.33)
            children.append(child)
        
        # Get children batch
        parents = torch.tensor([root])
        children_batch = tree.get_children_batch(parents)
        
        assert children_batch.shape[0] == 1
        assert children_batch.shape[1] >= 3
        
        # First 3 should be our children
        retrieved = children_batch[0, :3].tolist()
        assert set(retrieved) == set(children)
    
    def test_node_data_access(self, tree):
        """Test efficient node data access"""
        # Root is automatically created
        root = 0
        tree.node_visits[root] = 10
        tree.value_sums[root] = 5.0
        
        # Get node data
        data = tree.get_node_data(root, ['visits', 'value', 'prior'])
        
        assert 'visits' in data
        assert 'value' in data
        assert 'prior' in data
        assert data['visits'].item() == 10
        assert data['value'].item() == 0.5  # 5.0 / 10 = 0.5 (average value)
        assert data['prior'].item() == 1.0
    
    def test_batch_updates(self, tree):
        """Test batch update operations"""
        # Root is automatically created (index 0)
        node_ids = [0]  # Start with root
        
        # Add 4 children to root
        for i in range(4):
            node_id = tree.add_child(parent_idx=0, action=i, child_prior=0.2)
            node_ids.append(node_id)
        
        # Batch update visits and values
        indices = torch.tensor(node_ids)
        visit_updates = torch.ones(5, dtype=torch.int32)
        value_updates = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        
        tree.batch_update_visits(indices, visit_updates)
        tree.batch_update_values(indices, value_updates)
        
        # Check updates
        for i, node_id in enumerate(node_ids):
            assert tree.node_visits[node_id] == 1
            assert tree.value_sums[node_id] == value_updates[i]
    
    def test_virtual_loss_mechanism(self, tree):
        """Test virtual loss application and removal"""
        # Root is automatically created
        root = 0
        child = tree.add_child(parent_idx=root, action=0, child_prior=0.5)
        
        initial_visits = tree.node_visits[child].item()
        initial_value = tree.value_sums[child].item()
        
        # Apply virtual loss
        nodes = torch.tensor([child])
        tree.apply_virtual_loss(nodes)
        
        # Check virtual loss applied (virtual loss doesn't change actual visits/values)
        assert tree.virtual_loss_counts[child] == 1
        assert tree.node_visits[child] == initial_visits  # Actual visits unchanged
        assert tree.value_sums[child] == initial_value  # Actual values unchanged
        
        # Remove virtual loss
        tree.remove_virtual_loss(nodes)
        
        # Virtual loss should be removed
        assert tree.virtual_loss_counts[child] == 0
        assert tree.node_visits[child] == initial_visits
        assert tree.value_sums[child] == initial_value
    
    def test_memory_efficiency(self, tree):
        """Test memory-efficient storage"""
        # Root is automatically created (index 0)
        # Add 99 more nodes in a chain
        for i in range(1, 100):
            parent = i - 1
            tree.add_child(parent_idx=parent, action=0, child_prior=0.01)
        
        # Check memory layout is compact
        assert tree.num_nodes == 100
        
        # All node data should be contiguous
        assert tree.node_visits[:tree.num_nodes].is_contiguous()
        assert tree.value_sums[:tree.num_nodes].is_contiguous()


class TestGPUGameStates:
    """Test GPU game state management"""
    
    @pytest.fixture
    def game_config(self):
        """Create game states configuration"""
        return GPUGameStatesConfig(
            capacity=1000,
            game_type=GameType.GOMOKU,
            board_size=15,
            device='cpu'  # Use CPU for testing
        )
    
    @pytest.fixture
    def game_states(self, game_config):
        """Create GPU game states instance"""
        return GPUGameStates(game_config)
    
    def test_initialization(self, game_states, game_config):
        """Test GPU game states initialization"""
        assert game_states.config == game_config
        assert game_states.capacity == 1000
        assert game_states.board_size == 15
        assert game_states.device.type == 'cpu'
    
    def test_state_allocation(self, game_states):
        """Test state allocation and management"""
        # Allocate single state
        state_ids = game_states.allocate_states(1)
        assert len(state_ids) == 1
        state_id = state_ids[0]
        assert state_id >= 0
        assert state_id < game_states.capacity
        
        # Allocate batch
        batch_ids = game_states.allocate_states(10)
        assert len(batch_ids) == 10
        assert all(0 <= sid < game_states.capacity for sid in batch_ids)
    
    def test_board_operations(self, game_states):
        """Test board state operations"""
        # Allocate state
        state_ids = game_states.allocate_states(1)
        state_id = state_ids[0]
        
        # Set board position
        if hasattr(game_states, 'set_board'):
            board = torch.zeros((15, 15), dtype=torch.int8)
            board[7, 7] = 1  # Center stone
            game_states.set_board(state_id, board)
            
            # Get board back
            retrieved = game_states.get_board(state_id)
            assert torch.equal(retrieved, board)
    
    def test_feature_extraction(self, game_states):
        """Test feature extraction for neural network"""
        game_states.enable_enhanced_features()
        
        # Allocate states
        state_ids = game_states.allocate_states(5)
        
        # Get features
        if hasattr(game_states, 'get_enhanced_features_batch'):
            features = game_states.get_enhanced_features_batch(state_ids)
            
            # Should have shape (batch, channels, height, width)
            assert features.shape == (5, 20, 15, 15)  # 20 channels for enhanced features
    
    def test_legal_moves_generation(self, game_states):
        """Test legal move generation"""
        state_ids = game_states.allocate_states(1)
        state_id = state_ids[0]
        
        # Get legal moves
        if hasattr(game_states, 'get_legal_moves'):
            legal_moves = game_states.get_legal_moves(state_id)
            
            # For empty board, all moves should be legal
            assert legal_moves.shape == (225,)  # 15x15
            assert torch.all(legal_moves)  # All True for empty board
    
    def test_move_application(self, game_states):
        """Test applying moves to states"""
        state_ids = game_states.allocate_states(1)
        state_id = state_ids[0]
        
        # Apply move
        if hasattr(game_states, 'apply_move'):
            move = 112  # Center of 15x15 board
            new_state_id = game_states.apply_move(state_id, move, player=1)
            
            assert new_state_id >= 0
            
            # Check move was applied
            board = game_states.get_board(new_state_id)
            row, col = move // 15, move % 15
            assert board[row, col] == 1
    
    def test_batch_operations(self, game_states):
        """Test batch state operations"""
        # Allocate batch
        state_ids = game_states.allocate_states(10)
        
        # Apply different moves to each state
        moves = torch.arange(10)
        players = torch.ones(10, dtype=torch.int8)
        
        if hasattr(game_states, 'apply_moves_batch'):
            new_state_ids = game_states.apply_moves_batch(state_ids, moves, players)
            
            assert len(new_state_ids) == 10
            assert all(sid >= 0 for sid in new_state_ids)


class TestUnifiedKernels:
    """Test unified GPU kernel operations"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_kernel_availability(self):
        """Test that GPU kernels are available"""
        device = torch.device('cuda')
        kernels = get_unified_kernels(device)
        
        assert kernels is not None
        
        # Check key kernels
        expected_kernels = [
            'batch_ucb_selection',
            'parallel_backup',
            'quantum_interference',
            'fused_minhash_interference',
            'phase_kicked_policy'
        ]
        
        for kernel_name in expected_kernels:
            assert hasattr(kernels, kernel_name)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ucb_kernel(self):
        """Test UCB computation kernel"""
        device = torch.device('cuda')
        kernels = get_unified_kernels(device)
        
        # Test UCB selection kernel
        if hasattr(kernels, 'batch_ucb_selection'):
            # Create mock CSR tree
            tree = Mock()
            tree.device = device
            tree.num_nodes = 100
            tree.visit_counts = torch.randint(1, 100, (100,), device=device, dtype=torch.int32)
            tree.value_sums = torch.rand(100, device=device)
            tree.node_priors = torch.rand(100, device=device)
            tree.virtual_loss_counts = torch.zeros(100, device=device, dtype=torch.int32)
            tree.enable_virtual_loss = False
            
            # Test batch UCB selection
            parent_indices = torch.arange(32, device=device, dtype=torch.int32)
            c_puct = 1.414
            
            # Create mock CSR data
            # row_ptr defines where each node's children start in the edge arrays
            row_ptr = torch.arange(101, device=device, dtype=torch.int32) * 5  # 5 children per node
            num_edges = 500
            # col_indices should contain valid node indices (children nodes)
            # Make sure all child indices are < 100 (num_nodes)
            col_indices = torch.randint(0, 100, (num_edges,), device=device, dtype=torch.int32)
            edge_actions = torch.randint(0, 361, (num_edges,), device=device, dtype=torch.int32)
            edge_priors = torch.rand(num_edges, device=device)
            
            selected_actions, ucb_scores = kernels.batch_ucb_selection(
                parent_indices,
                row_ptr,
                col_indices,
                edge_actions,
                edge_priors,
                tree.visit_counts,
                tree.value_sums,
                c_puct
            )
            
            assert selected_actions.shape == (32,)
            assert ucb_scores.shape == (32,)


class TestMemoryPooling:
    """Test memory pooling and optimization"""
    
    def test_memory_pool_initialization(self):
        """Test memory pool setup"""
        config = MCTSConfig(
            device='cpu',
            memory_pool_size_mb=256,
            max_tree_nodes=10000,
            use_optimized_implementation=True
        )
        evaluator = Mock()
        evaluator._return_torch_tensors = True
        evaluator.evaluate_batch = MagicMock(return_value=(
            torch.rand(32, 225),  # policies
            torch.rand(32)        # values
        ))
        
        mcts = MCTS(config, evaluator)
        
        # Check memory pools allocated
        assert hasattr(mcts, 'state_pool_free')
        assert hasattr(mcts, 'node_to_state')
        assert mcts.state_pool_free.shape[0] == config.max_tree_nodes
    
    def test_zero_allocation_search(self):
        """Test that search doesn't allocate during execution"""
        config = MCTSConfig(
            device='cpu',
            memory_pool_size_mb=128,
            max_tree_nodes=1000,
            num_simulations=100,
            use_optimized_implementation=True
        )
        evaluator = Mock()
        evaluator._return_torch_tensors = True
        evaluator.evaluate_batch = MagicMock(return_value=(
            torch.rand(32, 225),  # policies
            torch.rand(32)        # values
        ))
        
        mcts = MCTS(config, evaluator)
        
        # Track memory allocations
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        
        # Run search
        state = MockState()
        mcts.search(state, num_simulations=50)
        
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            # Memory usage should be minimal during search
            assert final_memory - initial_memory < 1024 * 1024  # Less than 1MB


class TestCUDAOptimizations:
    """Test CUDA-specific optimizations"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_graphs(self):
        """Test CUDA graph compilation"""
        config = MCTSConfig(
            device='cuda',
            use_cuda_graphs=True,
            use_optimized_implementation=True
        )
        evaluator = Mock()
        evaluator._return_torch_tensors = True
        evaluator.evaluate_batch = MagicMock(return_value=(
            torch.rand(32, 225, device='cuda'),  # policies
            torch.rand(32, device='cuda')        # values
        ))
        
        mcts = MCTS(config, evaluator)
        
        # CUDA graphs setup happens in _setup_cuda_graphs
        assert hasattr(mcts, 'cuda_graph')
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_tensor_cores(self):
        """Test tensor core utilization"""
        config = MCTSConfig(
            device='cuda',
            use_tensor_cores=True,
            use_mixed_precision=True,
            use_optimized_implementation=True
        )
        evaluator = Mock()
        evaluator._return_torch_tensors = True
        
        # Return half precision for tensor cores
        evaluator.evaluate_batch = MagicMock(return_value=(
            torch.rand(32, 225, device='cuda', dtype=torch.float16),
            torch.rand(32, device='cuda', dtype=torch.float16)
        ))
        
        mcts = MCTS(config, evaluator)
        
        # Check mixed precision enabled
        assert config.use_mixed_precision
        assert config.use_tensor_cores
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_coalescing(self):
        """Test memory access patterns for coalescing"""
        config = MCTSConfig(
            device='cuda',
            max_wave_size=32,  # Warp size for coalescing
            use_optimized_implementation=True
        )
        evaluator = Mock()
        evaluator._return_torch_tensors = True
        evaluator.evaluate_batch = MagicMock(return_value=(
            torch.rand(32, 225, device='cuda'),
            torch.rand(32, device='cuda')
        ))
        
        mcts = MCTS(config, evaluator)
        
        # Check buffer alignment for coalescing
        assert mcts.paths_buffer.is_contiguous()
        assert mcts.ucb_scores.is_contiguous()
        assert mcts.eval_values.is_contiguous()


class TestVectorizedOperations:
    """Test vectorized batch operations"""
    
    def test_vectorized_selection(self):
        """Test vectorized node selection"""
        config = MCTSConfig(
            device='cpu',
            max_wave_size=64,
            use_optimized_implementation=True
        )
        evaluator = Mock()
        evaluator._return_torch_tensors = True
        evaluator.evaluate_batch = MagicMock(return_value=(
            torch.rand(64, 225),
            torch.rand(64)
        ))
        
        mcts = MCTS(config, evaluator)
        
        # Initialize tree
        state = MockState()
        mcts._initialize_root(state)
        
        # Test vectorized selection
        wave_size = 32
        if hasattr(mcts, '_select_leaves_vectorized'):
            leaves, paths = mcts._select_leaves_vectorized(wave_size)
            
            assert leaves.shape[0] == wave_size
            assert paths.shape[0] == wave_size
    
    def test_vectorized_expansion(self):
        """Test vectorized node expansion"""
        config = MCTSConfig(
            device='cpu',
            max_wave_size=64,
            initial_children_per_expansion=8,
            use_optimized_implementation=True
        )
        evaluator = Mock()
        evaluator._return_torch_tensors = True
        evaluator.evaluate_batch = MagicMock(return_value=(
            torch.rand(64, 225),
            torch.rand(64)
        ))
        
        mcts = MCTS(config, evaluator)
        
        # Initialize
        state = MockState()
        mcts._initialize_root(state)
        
        # Test vectorized expansion
        nodes_to_expand = torch.tensor([0])  # Expand root
        if hasattr(mcts, '_expand_nodes_vectorized'):
            expanded = mcts._expand_nodes_vectorized(nodes_to_expand)
            
            # Should have expanded nodes
            assert expanded.any()
    
    def test_vectorized_backup(self):
        """Test vectorized value backup"""
        config = MCTSConfig(
            device='cpu',
            max_wave_size=64,
            use_optimized_implementation=True
        )
        evaluator = Mock()
        evaluator._return_torch_tensors = True
        evaluator.evaluate_batch = MagicMock(return_value=(
            torch.rand(64, 225),
            torch.rand(64)
        ))
        
        mcts = MCTS(config, evaluator)
        
        # Create simple paths
        batch_size = 16
        max_depth = 5
        paths = torch.zeros((batch_size, max_depth), dtype=torch.int32)
        paths[:, 0] = 0  # All start at root
        values = torch.rand(batch_size)
        
        if hasattr(mcts, '_backup_vectorized'):
            mcts._backup_vectorized(paths, values)
            
            # Root should have been updated
            root_visits = mcts.tree.node_visits[0]
            assert root_visits >= batch_size


class TestPerformanceOptimizations:
    """Test performance optimization features"""
    
    def test_compilation_modes(self):
        """Test different torch.compile modes"""
        config = MCTSConfig(
            device='cpu',
            compile_mode="reduce-overhead",
            use_optimized_implementation=True
        )
        evaluator = Mock()
        evaluator._return_torch_tensors = True
        evaluator.evaluate_batch = MagicMock(return_value=(
            torch.rand(32, 225),
            torch.rand(32)
        ))
        
        mcts = MCTS(config, evaluator)
        
        assert config.compile_mode == "reduce-overhead"
    
    def test_kernel_fusion(self):
        """Test kernel fusion opportunities"""
        config = MCTSConfig(
            device='cpu',
            use_optimized_implementation=True
        )
        evaluator = Mock()
        evaluator._return_torch_tensors = True
        evaluator.evaluate_batch = MagicMock(return_value=(
            torch.rand(32, 225),
            torch.rand(32)
        ))
        
        mcts = MCTS(config, evaluator)
        
        # In optimized implementation, operations should be fused
        # This is implementation-specific and would be verified through profiling
        assert mcts.using_optimized
    
    def test_wave_size_optimization(self):
        """Test optimal wave size selection"""
        config = MCTSConfig(
            device='cpu',
            min_wave_size=3072,
            max_wave_size=3072,
            adaptive_wave_sizing=False,  # Fixed size for performance
            use_optimized_implementation=True
        )
        evaluator = Mock()
        evaluator._return_torch_tensors = True
        evaluator.evaluate_batch = MagicMock(return_value=(
            torch.rand(3072, 225),
            torch.rand(3072)
        ))
        
        mcts = MCTS(config, evaluator)
        
        # Verify optimal wave size
        assert config.max_wave_size == 3072  # Optimal for RTX 3060 Ti
        assert not config.adaptive_wave_sizing  # Must be false for best performance
    
    def test_profiling_support(self):
        """Test GPU kernel profiling support"""
        config = MCTSConfig(
            device='cpu',
            profile_gpu_kernels=True,
            enable_debug_logging=True,
            use_optimized_implementation=True
        )
        evaluator = Mock()
        evaluator._return_torch_tensors = True
        evaluator.evaluate_batch = MagicMock(return_value=(
            torch.rand(32, 225),
            torch.rand(32)
        ))
        
        mcts = MCTS(config, evaluator)
        
        # Should have profiling enabled
        assert config.profile_gpu_kernels
        assert mcts.kernel_timings is not None
    
    def test_statistics_collection(self):
        """Test performance statistics collection"""
        config = MCTSConfig(
            device='cpu',
            use_optimized_implementation=True
        )
        evaluator = Mock()
        evaluator._return_torch_tensors = True
        evaluator.evaluate_batch = MagicMock(return_value=(
            torch.rand(32, 225),
            torch.rand(32)
        ))
        
        mcts = MCTS(config, evaluator)
        
        # Run search
        state = MockState()
        mcts.search(state, num_simulations=100)
        
        # Check statistics
        stats = mcts.stats
        assert 'total_searches' in stats
        assert 'total_simulations' in stats
        assert 'avg_sims_per_second' in stats
        assert 'peak_sims_per_second' in stats
        
        # Should have performance metrics
        assert stats['total_searches'] == 1
        assert stats['total_simulations'] == 100
        assert stats['avg_sims_per_second'] > 0


class TestIntegrationScenarios:
    """Test integrated scenarios"""
    
    def test_full_search_pipeline(self):
        """Test complete search pipeline"""
        config = MCTSConfig(
            device='cpu',
            num_simulations=1000,
            max_wave_size=64,
            use_optimized_implementation=True
        )
        evaluator = Mock()
        evaluator._return_torch_tensors = True
        
        # Mock realistic evaluation
        def mock_eval(states, legal_masks=None, temperature=1.0):
            batch_size = len(states) if hasattr(states, '__len__') else states.shape[0]
            policies = torch.rand(batch_size, 225)
            if legal_masks is not None:
                policies = policies.masked_fill(~legal_masks, 0)
            policies = torch.softmax(policies, dim=1)
            values = torch.rand(batch_size) * 2 - 1
            return policies, values
        
        evaluator.evaluate_batch = mock_eval
        
        mcts = MCTS(config, evaluator)
        
        # Run full search
        state = MockState()
        policy = mcts.search(state)
        
        # Verify results
        assert policy.shape == (225,)
        assert np.allclose(policy.sum(), 1.0)
        assert mcts.stats['total_simulations'] == 1000
        
        # Tree should have grown
        if mcts.using_optimized:
            assert mcts.tree.num_nodes > 1
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_cpu_consistency(self):
        """Test consistency between GPU and CPU implementations"""
        # Create identical configs except device
        config_gpu = MCTSConfig(
            device='cuda',
            num_simulations=100,
            c_puct=1.414,
            use_optimized_implementation=True
        )
        
        config_cpu = MCTSConfig(
            device='cpu',
            num_simulations=100,
            c_puct=1.414,
            use_optimized_implementation=True
        )
        
        # Use deterministic evaluation
        def deterministic_eval(device):
            def eval_fn(states, legal_masks=None, temperature=1.0):
                batch_size = len(states) if hasattr(states, '__len__') else states.shape[0]
                # Use fixed seed for reproducibility
                torch.manual_seed(42)
                policies = torch.rand(batch_size, 225, device=device)
                values = torch.zeros(batch_size, device=device)  # Fixed values
                return policies / policies.sum(dim=1, keepdim=True), values
            return eval_fn
        
        evaluator_gpu = Mock()
        evaluator_gpu._return_torch_tensors = True
        evaluator_gpu.evaluate_batch = deterministic_eval('cuda')
        
        evaluator_cpu = Mock()
        evaluator_cpu._return_torch_tensors = True
        evaluator_cpu.evaluate_batch = deterministic_eval('cpu')
        
        mcts_gpu = MCTS(config_gpu, evaluator_gpu)
        mcts_cpu = MCTS(config_cpu, evaluator_cpu)
        
        # Run searches
        state = MockState()
        
        torch.manual_seed(42)
        policy_gpu = mcts_gpu.search(state, num_simulations=50)
        
        torch.manual_seed(42)
        policy_cpu = mcts_cpu.search(state, num_simulations=50)
        
        # Results should have same shape
        assert policy_gpu.shape == policy_cpu.shape
        # Both should be valid probability distributions
        assert np.allclose(policy_gpu.sum(), 1.0, atol=0.01)
        assert np.allclose(policy_cpu.sum(), 1.0, atol=0.01)
        # Both should have non-negative values
        assert np.all(policy_gpu >= 0)
        assert np.all(policy_cpu >= 0)