"""Tests for CUDA kernels"""

import pytest
import torch
import numpy as np
from unittest.mock import patch

from mcts.gpu.cuda_kernels import CUDAKernels, create_cuda_kernels, CUDA_AVAILABLE


class TestCUDAKernels:
    """Test CUDA kernel implementations"""
    
    @pytest.fixture
    def cuda_kernels_cpu(self):
        """Create CUDA kernels instance for CPU"""
        return CUDAKernels(device=torch.device('cpu'))
        
    @pytest.fixture  
    def cuda_kernels_gpu(self):
        """Create CUDA kernels instance for GPU if available"""
        if CUDA_AVAILABLE:
            return CUDAKernels(device=torch.device('cuda'))
        else:
            pytest.skip("CUDA not available")
            
    def test_initialization_cpu(self, cuda_kernels_cpu):
        """Test initialization on CPU"""
        assert cuda_kernels_cpu.device.type == 'cpu'
        assert not cuda_kernels_cpu.use_triton
        
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_initialization_gpu(self, cuda_kernels_gpu):
        """Test initialization on GPU"""
        assert cuda_kernels_gpu.device.type == 'cuda'
        # Triton may or may not be available
        
    def test_compute_batched_ucb_cpu(self, cuda_kernels_cpu):
        """Test batched UCB computation on CPU"""
        batch_size = 100
        
        q_values = torch.rand(batch_size)
        visit_counts = torch.randint(0, 100, (batch_size,)).float()
        parent_visits = torch.randint(100, 1000, (batch_size,)).float()
        priors = torch.rand(batch_size)
        priors = priors / priors.sum()  # Normalize
        
        ucb_scores = cuda_kernels_cpu.compute_batched_ucb(
            q_values, visit_counts, parent_visits, priors, c_puct=1.0
        )
        
        # Check shape
        assert ucb_scores.shape == (batch_size,)
        
        # Check values are reasonable
        assert torch.all(torch.isfinite(ucb_scores))
        
        # Manual computation for verification
        exploration = priors * torch.sqrt(parent_visits) / (1 + visit_counts)
        expected = q_values + exploration
        assert torch.allclose(ucb_scores, expected, rtol=1e-5)
        
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_compute_batched_ucb_gpu(self, cuda_kernels_gpu):
        """Test batched UCB computation on GPU"""
        batch_size = 10000
        
        q_values = torch.rand(batch_size, device='cuda')
        visit_counts = torch.randint(0, 100, (batch_size,), device='cuda').float()
        parent_visits = torch.randint(100, 1000, (batch_size,), device='cuda').float()
        priors = torch.rand(batch_size, device='cuda')
        priors = priors / priors.sum()
        
        ucb_scores = cuda_kernels_gpu.compute_batched_ucb(
            q_values, visit_counts, parent_visits, priors, c_puct=2.0
        )
        
        assert ucb_scores.device.type == 'cuda'
        assert ucb_scores.shape == (batch_size,)
        assert torch.all(torch.isfinite(ucb_scores))
        
    def test_mixed_precision_backup_cpu(self, cuda_kernels_cpu):
        """Test mixed precision backup on CPU"""
        n_paths = 32
        max_depth = 20
        n_nodes = 1000
        
        values = torch.rand(n_paths)
        visit_counts = torch.randint(0, 200, (n_nodes,)).float()
        node_indices = torch.randint(-1, n_nodes, (n_paths, max_depth))
        
        q_updates, visit_updates = cuda_kernels_cpu.mixed_precision_backup(
            values, visit_counts, node_indices, threshold=100
        )
        
        assert q_updates.shape == (n_nodes,)
        assert visit_updates.shape == (n_nodes,)
        # Q-values can be negative due to negamax backup in MCTS
        assert torch.all(torch.isfinite(q_updates))
        assert torch.all(visit_updates >= 0)
        
    def test_apply_phase_kicks_cpu(self, cuda_kernels_cpu):
        """Test phase kicks on CPU"""
        n_actions = 50
        
        priors = torch.rand(n_actions)
        priors = priors / priors.sum()
        visit_counts = torch.randint(0, 100, (n_actions,)).float()
        q_values = torch.rand(n_actions) - 0.5  # Center around 0
        
        original_priors = priors.clone()
        
        kicked_priors = cuda_kernels_cpu.apply_phase_kicks(
            priors.clone(), visit_counts, q_values,
            kick_strength=0.1, temperature=1.0
        )
        
        # Check normalization
        assert torch.allclose(kicked_priors.sum(), torch.tensor(1.0), rtol=1e-5)
        
        # Check that kicks were applied (should be different)
        assert not torch.allclose(kicked_priors, original_priors)
        
        # Check all values are valid
        assert torch.all(kicked_priors >= 0)
        assert torch.all(torch.isfinite(kicked_priors))
        
    def test_parallel_minhash(self, cuda_kernels_cpu):
        """Test parallel MinHash computation"""
        batch_size = 16
        feature_dim = 100
        num_hashes = 64
        
        features = torch.rand(batch_size, feature_dim)
        
        signatures = cuda_kernels_cpu.parallel_minhash(
            features, num_hashes=num_hashes, seed=42
        )
        
        assert signatures.shape == (batch_size, num_hashes)
        assert torch.all(signatures >= 0)
        assert torch.all(torch.isfinite(signatures))
        
        # Test determinism with same seed
        signatures2 = cuda_kernels_cpu.parallel_minhash(
            features, num_hashes=num_hashes, seed=42
        )
        assert torch.allclose(signatures, signatures2)
        
        # Test different seed gives different results
        signatures3 = cuda_kernels_cpu.parallel_minhash(
            features, num_hashes=num_hashes, seed=123
        )
        assert not torch.allclose(signatures, signatures3)
        
    def test_batch_entropy(self, cuda_kernels_cpu):
        """Test batch entropy computation"""
        batch_size = 32
        num_actions = 10
        
        # Test uniform distribution (high entropy)
        uniform_probs = torch.ones(batch_size, num_actions) / num_actions
        uniform_entropy = cuda_kernels_cpu.batch_entropy(uniform_probs)
        
        expected_uniform = -np.log(1.0 / num_actions)
        assert torch.allclose(uniform_entropy, torch.tensor(expected_uniform, dtype=torch.float32), rtol=1e-4)
        
        # Test peaked distribution (low entropy)
        peaked_probs = torch.zeros(batch_size, num_actions)
        peaked_probs[:, 0] = 0.99
        peaked_probs[:, 1:] = 0.01 / (num_actions - 1)
        peaked_entropy = cuda_kernels_cpu.batch_entropy(peaked_probs)
        
        assert torch.all(peaked_entropy < uniform_entropy)
        assert torch.all(peaked_entropy >= 0)
        
    def test_vectorized_softmax_sample(self, cuda_kernels_cpu):
        """Test vectorized softmax sampling"""
        batch_size = 64
        num_actions = 20
        
        logits = torch.randn(batch_size, num_actions)
        
        # Single sample per distribution
        samples = cuda_kernels_cpu.vectorized_softmax_sample(
            logits, temperature=1.0, num_samples=1
        )
        
        assert samples.shape == (batch_size, 1)
        assert torch.all(samples >= 0)
        assert torch.all(samples < num_actions)
        
        # Multiple samples
        multi_samples = cuda_kernels_cpu.vectorized_softmax_sample(
            logits, temperature=0.5, num_samples=10
        )
        
        assert multi_samples.shape == (batch_size, 10)
        assert torch.all(multi_samples >= 0)
        assert torch.all(multi_samples < num_actions)
        
        # Test temperature effect
        hot_samples = cuda_kernels_cpu.vectorized_softmax_sample(
            logits, temperature=0.1, num_samples=100
        )
        cold_samples = cuda_kernels_cpu.vectorized_softmax_sample(
            logits, temperature=10.0, num_samples=100
        )
        
        # Hot temperature should have less diversity
        hot_unique = torch.tensor([torch.unique(hot_samples[i]).shape[0] 
                                   for i in range(batch_size)])
        cold_unique = torch.tensor([torch.unique(cold_samples[i]).shape[0] 
                                    for i in range(batch_size)])
        
        assert torch.mean(hot_unique.float()) < torch.mean(cold_unique.float())
        
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_gpu_cpu_consistency(self):
        """Test that GPU and CPU implementations give same results"""
        cpu_kernels = CUDAKernels(device=torch.device('cpu'))
        gpu_kernels = CUDAKernels(device=torch.device('cuda'))
        
        # Test data
        batch_size = 100
        q_values = torch.rand(batch_size)
        visit_counts = torch.randint(0, 100, (batch_size,)).float()
        parent_visits = torch.randint(100, 1000, (batch_size,)).float() 
        priors = torch.rand(batch_size)
        priors = priors / priors.sum()
        
        # CPU computation
        ucb_cpu = cpu_kernels.compute_batched_ucb(
            q_values, visit_counts, parent_visits, priors
        )
        
        # GPU computation
        ucb_gpu = gpu_kernels.compute_batched_ucb(
            q_values.cuda(), visit_counts.cuda(), 
            parent_visits.cuda(), priors.cuda()
        )
        
        # Compare
        assert torch.allclose(ucb_cpu, ucb_gpu.cpu(), rtol=1e-5)


class TestFactoryFunction:
    """Test factory function"""
    
    def test_create_cuda_kernels_default(self):
        """Test creating kernels with default device"""
        kernels = create_cuda_kernels()
        assert isinstance(kernels, CUDAKernels)
        
        if CUDA_AVAILABLE:
            assert kernels.device.type == 'cuda'
        else:
            assert kernels.device.type == 'cpu'
            
    def test_create_cuda_kernels_cpu(self):
        """Test creating kernels with CPU device"""
        kernels = create_cuda_kernels(torch.device('cpu'))
        assert kernels.device.type == 'cpu'
        assert not kernels.use_triton
        
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_create_cuda_kernels_gpu(self):
        """Test creating kernels with GPU device"""
        kernels = create_cuda_kernels(torch.device('cuda:0'))
        assert kernels.device.type == 'cuda'


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_batch(self):
        """Test handling of empty batches"""
        kernels = CUDAKernels(device=torch.device('cpu'))
        
        # Empty UCB computation
        q_values = torch.tensor([])
        visit_counts = torch.tensor([])
        parent_visits = torch.tensor([])
        priors = torch.tensor([])
        
        ucb_scores = kernels.compute_batched_ucb(
            q_values, visit_counts, parent_visits, priors
        )
        
        assert ucb_scores.shape == (0,)
        
    def test_single_element(self):
        """Test handling of single element batches"""
        kernels = CUDAKernels(device=torch.device('cpu'))
        
        q_values = torch.tensor([0.5])
        visit_counts = torch.tensor([10.0])
        parent_visits = torch.tensor([100.0])
        priors = torch.tensor([1.0])
        
        ucb_scores = kernels.compute_batched_ucb(
            q_values, visit_counts, parent_visits, priors
        )
        
        assert ucb_scores.shape == (1,)
        assert torch.isfinite(ucb_scores[0])
        
    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        kernels = CUDAKernels(device=torch.device('cpu'))
        
        # Test with very small values
        small_probs = torch.tensor([1e-10, 1e-9, 1e-8, 1.0 - 3e-8])
        small_probs = small_probs / small_probs.sum()
        
        entropy = kernels.batch_entropy(small_probs.unsqueeze(0))
        assert torch.all(torch.isfinite(entropy))
        assert torch.all(entropy >= 0)
        
        # Test with zero visits
        q_values = torch.rand(10)
        visit_counts = torch.zeros(10)
        parent_visits = torch.ones(10) * 100
        priors = torch.rand(10)
        priors = priors / priors.sum()
        
        ucb_scores = kernels.compute_batched_ucb(
            q_values, visit_counts, parent_visits, priors
        )
        
        assert torch.all(torch.isfinite(ucb_scores))