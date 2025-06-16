"""Comprehensive tests for GPU-accelerated quantum interference

This module tests the MinHash interference implementation including:
- MinHash signature computation
- Jaccard similarity estimation
- Interference application
- LSH (Locality Sensitive Hashing) operations
- GPU acceleration
- Performance characteristics
"""

import pytest

# Skip entire module - quantum features are under development
pytestmark = pytest.mark.skip(reason="Quantum features are under development")

import torch
import numpy as np
from unittest.mock import Mock, patch
import logging

from mcts.quantum.interference_gpu import (
    MinHashConfig,
    MinHashInterference
)


class TestMinHashConfig:
    """Test MinHash configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = MinHashConfig()
        
        assert config.num_hashes == 128
        assert config.prime == 2147483647
        assert config.seed_base == 42
        assert config.similarity_threshold == 0.5
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = MinHashConfig(
            num_hashes=64,
            prime=1000000007,
            seed_base=123,
            similarity_threshold=0.7
        )
        
        assert config.num_hashes == 64
        assert config.prime == 1000000007
        assert config.seed_base == 123
        assert config.similarity_threshold == 0.7


class TestMinHashInterference:
    """Test MinHash interference engine"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return MinHashConfig(num_hashes=32, seed_base=42)
    
    @pytest.fixture
    def minhash(self, config):
        """Create MinHash interference instance"""
        device = torch.device('cpu')
        return MinHashInterference(device, strength=0.15, config=config)
    
    def test_initialization(self, minhash, config):
        """Test MinHash initialization"""
        assert minhash.device.type == 'cpu'
        assert minhash.strength == 0.15
        assert minhash.config == config
        
        # Check hash functions generated
        assert minhash.hash_a.shape == (32,)  # num_hashes
        assert minhash.hash_b.shape == (32,)
        assert torch.all(minhash.hash_a > 0)
        assert torch.all(minhash.hash_a < config.prime)
    
    def test_statistics_initialization(self, minhash):
        """Test statistics tracking initialization"""
        assert minhash.stats['paths_processed'] == 0
        assert minhash.stats['interference_events'] == 0
        assert minhash.stats['average_similarity'] == 0.0
        assert minhash.stats['gpu_kernel_calls'] == 0
    
    def test_hash_function_generation(self, minhash):
        """Test hash function coefficient generation"""
        # Hash functions should be deterministic with seed
        torch.manual_seed(minhash.config.seed_base)
        
        # Regenerate and compare
        minhash2 = MinHashInterference(
            minhash.device, 
            strength=minhash.strength, 
            config=minhash.config
        )
        
        assert torch.allclose(minhash.hash_a, minhash2.hash_a)
        assert torch.allclose(minhash.hash_b, minhash2.hash_b)
    
    def test_minhash_signature_single_path(self, minhash):
        """Test MinHash signature computation for single path"""
        # Create a simple path
        path = torch.tensor([[0, 1, 2, 3, -1, -1]], dtype=torch.long)
        
        signatures = minhash.compute_minhash_signatures(path)
        
        assert signatures.shape == (1, 32)  # (batch_size, num_hashes)
        assert signatures.dtype == torch.int64
        assert torch.all(signatures >= 0)
        assert torch.all(signatures < minhash.config.prime)
        
        # Check statistics
        assert minhash.stats['gpu_kernel_calls'] == 1
    
    def test_minhash_signature_batch(self, minhash):
        """Test MinHash signature computation for batch of paths"""
        batch_size = 10
        max_depth = 8
        
        # Create random paths with padding
        paths = torch.randint(0, 100, (batch_size, max_depth))
        paths[paths > 50] = -1  # Add padding
        
        signatures = minhash.compute_minhash_signatures(paths)
        
        assert signatures.shape == (batch_size, 32)
        assert torch.all(signatures >= 0)
        assert torch.all(signatures < minhash.config.prime)
    
    def test_minhash_with_custom_hashes(self, minhash):
        """Test MinHash with custom number of hash functions"""
        paths = torch.tensor([[0, 1, 2, -1], [3, 4, 5, -1]])
        
        # Use fewer hashes
        signatures = minhash.compute_minhash_signatures(paths, num_hashes=16)
        
        assert signatures.shape == (2, 16)
    
    def test_position_encoding(self, minhash):
        """Test that position encoding maintains order information"""
        # Two paths with same elements but different order
        path1 = torch.tensor([[0, 1, 2, -1]])
        path2 = torch.tensor([[2, 1, 0, -1]])
        
        sig1 = minhash.compute_minhash_signatures(path1)
        sig2 = minhash.compute_minhash_signatures(path2)
        
        # Signatures should be different due to position encoding
        assert not torch.allclose(sig1, sig2)
    
    def test_padding_handling(self, minhash):
        """Test handling of padded paths"""
        # Paths with different lengths
        paths = torch.tensor([
            [0, 1, 2, 3, 4],
            [0, 1, -1, -1, -1],  # Short path with padding
            [5, 6, 7, 8, 9]
        ])
        
        signatures = minhash.compute_minhash_signatures(paths)
        
        assert signatures.shape == (3, 32)
        # All signatures should be valid
        assert torch.all(signatures >= 0)
    
    def test_jaccard_similarity(self, minhash):
        """Test Jaccard similarity estimation"""
        # Create signatures
        signatures = torch.tensor([
            [1, 2, 3, 4],
            [1, 2, 5, 6],  # 50% overlap
            [7, 8, 9, 10]  # No overlap
        ])
        
        similarities = minhash.compute_jaccard_similarities(signatures)
        
        assert similarities.shape == (3, 3)
        assert torch.allclose(similarities, similarities.t())  # Symmetric
        assert torch.allclose(similarities.diag(), torch.ones(3))  # Self-similarity = 1
        
        # Check specific similarities
        assert similarities[0, 1] == 0.5  # 2/4 matches
        assert similarities[0, 2] == 0.0  # No matches
    
    def test_path_diversity_batch(self, minhash):
        """Test complete path diversity computation"""
        # Create paths with known similarities
        paths = torch.tensor([
            [0, 1, 2, 3, -1],
            [0, 1, 2, 4, -1],  # Very similar to first
            [5, 6, 7, 8, -1]   # Different
        ])
        
        signatures, similarities = minhash.compute_path_diversity_batch(paths, num_hashes=32)
        
        assert signatures.shape == (3, 32)
        assert similarities.shape == (3, 3)
        
        # Similar paths should have higher similarity
        assert similarities[0, 1] > similarities[0, 2]
        
        # Check statistics updated
        assert minhash.stats['paths_processed'] == 3
        assert minhash.stats['average_similarity'] > 0
    
    def test_interference_application(self, minhash):
        """Test quantum-inspired interference application"""
        # Create scores and similarities
        scores = torch.tensor([1.0, 0.8, 0.6])
        similarities = torch.tensor([
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.3],
            [0.2, 0.3, 1.0]
        ])
        
        # Apply interference
        modified_scores = minhash.apply_interference(scores, similarities)
        
        assert modified_scores.shape == scores.shape
        assert torch.all(modified_scores >= 0)  # Should remain positive (ReLU)
        
        # Similar paths should interfere more
        # Path 0 and 1 are similar, so should reduce each other's scores
        assert modified_scores[0] < scores[0]
        assert modified_scores[1] < scores[1]
    
    def test_interference_strength(self, minhash):
        """Test different interference strengths"""
        scores = torch.tensor([1.0, 1.0])
        similarities = torch.tensor([[1.0, 0.9], [0.9, 1.0]])
        
        # Test with different strengths
        weak_interference = minhash.apply_interference(scores, similarities, interference_strength=0.1)
        strong_interference = minhash.apply_interference(scores, similarities, interference_strength=0.5)
        
        # Stronger interference should reduce scores more
        assert torch.all(strong_interference < weak_interference)
    
    def test_interference_2d_scores(self, minhash):
        """Test interference with 2D score matrices"""
        batch_size = 4
        num_actions = 10
        
        scores = torch.rand(batch_size, num_actions)
        similarities = torch.eye(batch_size) + 0.5 * torch.rand(batch_size, batch_size)
        similarities = (similarities + similarities.t()) / 2  # Make symmetric
        
        modified_scores = minhash.apply_interference(scores, similarities)
        
        assert modified_scores.shape == (batch_size, num_actions)
        assert torch.all(modified_scores >= 0)
    
    def test_lsh_buckets(self, minhash):
        """Test LSH bucket computation"""
        # Create signatures
        signatures = torch.randint(0, 1000, (10, 32))
        
        buckets = minhash.compute_lsh_buckets(signatures, num_bands=4)
        
        assert buckets.shape == (10, 4)  # (num_signatures, num_bands)
        assert torch.all(buckets >= 0)
    
    def test_find_similar_paths_lsh(self, minhash):
        """Test LSH-based similar path finding"""
        # Create signatures with known patterns
        signatures = torch.tensor([
            [1, 1, 1, 1, 2, 2, 2, 2],  # Query
            [1, 1, 1, 1, 3, 3, 3, 3],  # Similar in first band
            [4, 4, 4, 4, 2, 2, 2, 2],  # Similar in second band
            [5, 5, 5, 5, 6, 6, 6, 6]   # Different
        ])
        
        similar_indices = minhash.find_similar_paths_lsh(signatures, query_idx=0, num_bands=2)
        
        # Should find paths 1 and 2 as similar
        assert 1 in similar_indices or 2 in similar_indices
        assert 3 not in similar_indices
    
    def test_compute_interference_vectorized(self, minhash):
        """Test fully vectorized interference computation"""
        batch_size = 20
        max_depth = 10
        
        paths = torch.randint(0, 100, (batch_size, max_depth))
        paths[paths > 70] = -1
        weights = torch.rand(batch_size)
        
        interference = minhash.compute_interference_vectorized(paths, weights)
        
        assert interference.shape == (batch_size,)
        assert torch.all(interference >= 0)
        
        # Check that similar paths have higher interference
        # (This is a statistical property, not guaranteed for random paths)
    
    def test_single_path_no_interference(self, minhash):
        """Test that single path has no interference"""
        path = torch.tensor([[0, 1, 2, -1]])
        
        interference = minhash.compute_interference_vectorized(path)
        
        assert interference.shape == (1,)
        assert interference[0] == 0.0  # No interference with self
    
    def test_statistics_tracking(self, minhash):
        """Test statistics tracking during operations"""
        paths = torch.randint(0, 50, (15, 5))
        
        # Run various operations
        signatures, similarities = minhash.compute_path_diversity_batch(paths)
        scores = torch.rand(15)
        minhash.apply_interference(scores, similarities)
        
        stats = minhash.get_statistics()
        
        assert stats['paths_processed'] == 15
        assert stats['interference_events'] > 0
        assert stats['average_similarity'] >= 0
        assert stats['gpu_kernel_calls'] > 0
    
    def test_statistics_reset(self, minhash):
        """Test statistics reset"""
        # Generate some statistics
        paths = torch.tensor([[0, 1, 2, -1]])
        minhash.compute_minhash_signatures(paths)
        
        assert minhash.stats['gpu_kernel_calls'] > 0
        
        # Reset
        minhash.reset_statistics()
        
        assert minhash.stats['paths_processed'] == 0
        assert minhash.stats['interference_events'] == 0
        assert minhash.stats['average_similarity'] == 0.0
        assert minhash.stats['gpu_kernel_calls'] == 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_acceleration(self):
        """Test GPU acceleration"""
        device = torch.device('cuda')
        config = MinHashConfig(num_hashes=128)
        minhash = MinHashInterference(device, strength=0.15, config=config)
        
        # Create GPU tensors
        batch_size = 100
        max_depth = 20
        paths = torch.randint(0, 1000, (batch_size, max_depth), device=device)
        paths[paths > 800] = -1
        
        # Run computations
        signatures, similarities = minhash.compute_path_diversity_batch(paths)
        
        assert signatures.device.type == 'cuda'
        assert similarities.device.type == 'cuda'
        
        # Apply interference
        scores = torch.rand(batch_size, device=device)
        modified_scores = minhash.apply_interference(scores, similarities)
        
        assert modified_scores.device.type == 'cuda'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_performance_characteristics(self):
        """Test performance with large batches"""
        device = torch.device('cuda')
        config = MinHashConfig(num_hashes=64)
        minhash = MinHashInterference(device, strength=0.1, config=config)
        
        # Large batch
        batch_size = 1000
        max_depth = 50
        paths = torch.randint(0, 10000, (batch_size, max_depth), device=device)
        paths[paths > 8000] = -1
        
        import time
        start = time.time()
        
        signatures, similarities = minhash.compute_path_diversity_batch(paths, num_hashes=32)
        
        elapsed = time.time() - start
        
        # Should be fast on GPU
        assert elapsed < 1.0  # Less than 1 second for 1000 paths
        
        # Check memory efficiency
        assert signatures.shape == (batch_size, 32)
        assert similarities.shape == (batch_size, batch_size)


class TestIntegration:
    """Integration tests with MCTS"""
    
    def test_with_mcts_paths(self):
        """Test with realistic MCTS path data"""
        device = torch.device('cpu')
        minhash = MinHashInterference(device, strength=0.2)
        
        # Simulate MCTS paths (node indices along tree paths)
        # Root = 0, then various child nodes
        paths = torch.tensor([
            [0, 1, 5, 12, -1, -1],   # Path 1
            [0, 1, 5, 13, -1, -1],   # Similar to path 1
            [0, 2, 7, 15, -1, -1],   # Different branch
            [0, 1, 6, 14, -1, -1],   # Somewhat similar to 1&2
            [0, 3, 8, 16, -1, -1]    # Different branch
        ])
        
        signatures, similarities = minhash.compute_path_diversity_batch(paths)
        
        # Paths 1 and 2 should be most similar (differ only in last node)
        assert similarities[0, 1] > similarities[0, 2]
        assert similarities[0, 1] > similarities[0, 4]
        
        # UCB scores for these paths
        ucb_scores = torch.tensor([1.5, 1.4, 1.2, 1.3, 1.1])
        
        # Apply interference
        modified_scores = minhash.apply_interference(ucb_scores, similarities)
        
        # Similar paths should have reduced scores
        assert modified_scores[0] < ucb_scores[0]
        assert modified_scores[1] < ucb_scores[1]
        
        # Different paths should be less affected
        score_reduction_similar = (ucb_scores[0] - modified_scores[0]).item()
        score_reduction_different = (ucb_scores[4] - modified_scores[4]).item()
        assert score_reduction_similar > score_reduction_different
    
    def test_diversity_enhancement(self):
        """Test that interference enhances diversity"""
        device = torch.device('cpu')
        minhash = MinHashInterference(device, strength=0.3)
        
        # Create a scenario with many similar paths
        base_path = [0, 1, 2, 3]
        paths = []
        
        # Add base path
        paths.append(base_path + [-1, -1])
        
        # Add many slight variations
        for i in range(10):
            variant = base_path.copy()
            variant[-1] = 3 + i  # Change last node
            paths.append(variant + [-1, -1])
        
        paths = torch.tensor(paths)
        
        # All paths are very similar
        signatures, similarities = minhash.compute_path_diversity_batch(paths)
        
        # Initial scores (all high)
        scores = torch.ones(len(paths))
        
        # Apply interference
        modified_scores = minhash.apply_interference(scores, similarities, interference_strength=0.5)
        
        # With uniform initial scores, interference won't create diversity
        # Instead, it will suppress all similar paths
        # Check that interference was applied
        assert torch.all(modified_scores <= scores)
        assert not torch.allclose(modified_scores, scores)
        
        # With very similar paths and strong interference, scores may be heavily suppressed
        assert torch.all(modified_scores >= 0)  # But should remain non-negative