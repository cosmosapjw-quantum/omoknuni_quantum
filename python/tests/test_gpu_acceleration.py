"""Tests for GPU acceleration functionality"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

# Import without torch to avoid import errors
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mcts.cuda_kernels import (
    CUDAKernels, GPUMemoryPool, BatchedTensorOperations,
    create_gpu_accelerated_evaluator, CUDA_AVAILABLE
)
from mcts.evaluator import EvaluatorConfig, GPUAcceleratedEvaluator
from mcts.mcts import MCTSConfig
from mcts.game_interface import GameInterface, GameType


class TestGPUAcceleration:
    """Test GPU acceleration features"""
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_cuda_kernels_initialization(self):
        """Test CUDA kernels initialization"""
        import torch
        
        device = torch.device('cuda')
        kernels = CUDAKernels(device)
        
        assert kernels.device == device
        assert kernels.use_triton in [True, False]  # Depends on Triton availability
        
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_batched_ucb_computation(self):
        """Test batched UCB score computation"""
        import torch
        
        kernels = CUDAKernels()
        
        # Create test data
        batch_size = 1000
        q_values = torch.rand(batch_size, device=kernels.device)
        visit_counts = torch.randint(0, 100, (batch_size,), device=kernels.device).float()
        parent_visits = torch.full((batch_size,), 1000.0, device=kernels.device)
        priors = torch.rand(batch_size, device=kernels.device)
        
        # Compute UCB scores
        start_time = time.time()
        ucb_scores = kernels.compute_batched_ucb(
            q_values, visit_counts, parent_visits, priors, c_puct=1.0
        )
        gpu_time = time.time() - start_time
        
        assert ucb_scores.shape == (batch_size,)
        assert ucb_scores.device == kernels.device
        print(f"GPU UCB computation time: {gpu_time*1000:.2f}ms for {batch_size} nodes")
        
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_memory_pool(self):
        """Test GPU memory pool"""
        import torch
        
        pool = GPUMemoryPool(initial_size=10)
        
        # Get tensors from pool
        shape = (100, 256)
        tensor1 = pool.get_tensor(shape)
        assert tensor1.shape == shape
        assert torch.all(tensor1 == 0)
        
        # Modify and return
        tensor1.fill_(1.0)
        pool.return_tensor(tensor1)
        
        # Get again - should be zeroed
        tensor2 = pool.get_tensor(shape)
        assert torch.all(tensor2 == 0)
        
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_batched_tensor_operations(self):
        """Test batched tensor operations"""
        import torch
        
        ops = BatchedTensorOperations()
        
        # Test action selection
        batch_size = 32
        action_size = 100
        policies = torch.rand(batch_size, action_size, device=ops.device)
        legal_moves = torch.ones(batch_size, action_size, dtype=torch.bool, device=ops.device)
        
        actions = ops.batch_select_actions(policies, legal_moves, temperature=1.0)
        assert actions.shape == (batch_size,)
        assert torch.all((actions >= 0) & (actions < action_size))
        
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_minhash_gpu(self):
        """Test MinHash computation on GPU"""
        import torch
        
        ops = BatchedTensorOperations()
        
        # Create test features
        n_nodes = 1000
        feature_dim = 128
        features = torch.rand(n_nodes, feature_dim, device=ops.device)
        
        # Compute signatures
        start_time = time.time()
        signatures = ops.compute_minhash_signatures(features, num_hashes=64)
        gpu_time = time.time() - start_time
        
        assert signatures.shape == (n_nodes, 64)
        print(f"GPU MinHash computation time: {gpu_time*1000:.2f}ms for {n_nodes} nodes")
        
    def test_gpu_accelerated_evaluator_mock(self):
        """Test GPU accelerated evaluator with mock model"""
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available")
            
        import torch
        import torch.nn as nn
        
        # Create mock model
        class MockModel(nn.Module):
            def __init__(self, board_size=15, action_size=225):
                super().__init__()
                self.board_size = board_size
                self.action_size = action_size
                
            def forward(self, x):
                batch_size = x.shape[0]
                # Mock policy and value outputs
                policy = torch.rand(batch_size, self.action_size, device=x.device)
                value = torch.rand(batch_size, 1, device=x.device) * 2 - 1  # [-1, 1]
                return policy, value
                
        model = MockModel()
        config = EvaluatorConfig(
            device='cuda',
            batch_size=512,
            use_fp16=True
        )
        
        # Create evaluator
        evaluator = GPUAcceleratedEvaluator(model, config, action_size=225)
        
        # Test single evaluation
        state = np.random.rand(20, 15, 15).astype(np.float32)
        policy, value = evaluator.evaluate(state)
        
        assert policy.shape == (225,)
        assert isinstance(value, (float, np.floating))
        
        # Test batch evaluation
        batch_size = 64
        states = np.random.rand(batch_size, 20, 15, 15).astype(np.float32)
        policies, values = evaluator.evaluate_batch(states)
        
        assert policies.shape == (batch_size, 225)
        assert values.shape == (batch_size,)
        
        # Check performance stats
        stats = evaluator.get_performance_stats()
        assert 'evaluations_per_second' in stats
        assert 'gpu_time_percentage' in stats
        print(f"GPU Evaluator stats: {stats}")
        
    def test_phase_kicks(self):
        """Test phase-kicked priors"""
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available")
            
        import torch
        
        kernels = CUDAKernels()
        
        # Create test data
        n_actions = 100
        priors = torch.rand(n_actions, device=kernels.device)
        priors = priors / priors.sum()  # Normalize
        visit_counts = torch.randint(0, 50, (n_actions,), device=kernels.device).float()
        q_values = torch.rand(n_actions, device=kernels.device) * 2 - 1
        
        # Apply phase kicks
        kicked_priors = kernels.apply_phase_kicks(
            priors, visit_counts, q_values,
            kick_strength=0.1, temperature=1.0
        )
        
        assert kicked_priors.shape == priors.shape
        assert torch.abs(kicked_priors.sum() - 1.0) < 1e-5  # Should sum to 1
        assert not torch.allclose(kicked_priors, priors)  # Should be different
        
    def test_mcts_with_gpu(self):
        """Test MCTS with GPU acceleration enabled"""
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available")
            
        from mcts import MCTS
        from mcts.evaluator import MockEvaluator
        
        # Create game and evaluator
        game = GameInterface(GameType.GOMOKU, board_size=9)
        evaluator = MockEvaluator(EvaluatorConfig(device='cuda'), action_size=81)
        
        # Create MCTS with GPU enabled
        config = MCTSConfig(
            num_simulations=100,
            use_gpu=True,
            gpu_batch_size=32,
            use_mixed_precision=True
        )
        
        mcts = MCTS(game, evaluator, config)
        
        # Check GPU components initialized
        assert mcts.gpu_kernels is not None
        assert mcts.gpu_ops is not None
        
        # Run search
        state = game.create_initial_state()
        root = mcts.search(state)
        
        assert root is not None
        assert root.visit_count > 0
        
        # Check GPU operations were used
        if mcts.stats['gpu_accelerated_ops'] > 0:
            print(f"GPU operations performed: {mcts.stats['gpu_accelerated_ops']}")


class TestGPUPerformance:
    """Performance benchmarks for GPU acceleration"""
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    @pytest.mark.benchmark(group="gpu")
    def test_benchmark_ucb_gpu(self, benchmark):
        """Benchmark UCB computation on GPU"""
        import torch
        
        kernels = CUDAKernels()
        
        # Large batch for benchmarking
        batch_size = 10000
        q_values = torch.rand(batch_size, device=kernels.device)
        visit_counts = torch.randint(0, 100, (batch_size,), device=kernels.device).float()
        parent_visits = torch.full((batch_size,), 1000.0, device=kernels.device)
        priors = torch.rand(batch_size, device=kernels.device)
        
        def compute_ucb():
            return kernels.compute_batched_ucb(
                q_values, visit_counts, parent_visits, priors, c_puct=1.0
            )
            
        result = benchmark(compute_ucb)
        assert result.shape == (batch_size,)
        
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")  
    @pytest.mark.benchmark(group="gpu")
    def test_benchmark_minhash_gpu(self, benchmark):
        """Benchmark MinHash on GPU"""
        import torch
        
        ops = BatchedTensorOperations()
        
        # Large dataset
        n_nodes = 5000
        feature_dim = 256
        features = torch.rand(n_nodes, feature_dim, device=ops.device)
        
        def compute_minhash():
            return ops.compute_minhash_signatures(features, num_hashes=128)
            
        result = benchmark(compute_minhash)
        assert result.shape == (n_nodes, 128)