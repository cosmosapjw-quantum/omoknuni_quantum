"""Integration tests to ensure all components work together"""

import pytest
import torch
import numpy as np

from mcts.gpu.cuda_kernels import create_cuda_kernels
# cpu_gpu_hybrid functionality is now integrated
from mcts.gpu.gpu_tree_kernels import GPUTreeKernels
from mcts.utils.state_delta_encoder import StateDeltaEncoder

# Mock create_hybrid_processor for compatibility
def create_hybrid_processor(config=None):
    """Legacy compatibility function"""
    return None


class TestIntegration:
    """Test that all components work together"""
    
    def test_cuda_kernels_integration(self):
        """Test CUDA kernels can be created and used"""
        kernels = create_cuda_kernels()
        
        # Test UCB computation
        batch_size = 100
        q_values = torch.rand(batch_size)
        visit_counts = torch.randint(0, 100, (batch_size,)).float()
        parent_visits = torch.ones(batch_size) * 100
        priors = torch.rand(batch_size)
        priors = priors / priors.sum()
        
        ucb_scores = kernels.compute_batched_ucb(
            q_values, visit_counts, parent_visits, priors
        )
        
        assert ucb_scores.shape == (batch_size,)
        assert torch.all(torch.isfinite(ucb_scores))
        
    def test_hybrid_processor_integration(self):
        """Test CPU-GPU hybrid processor"""
        processor = create_hybrid_processor(n_cpu_workers=4)
        
        # Test hybrid UCB computation
        batch_size = 1000
        q_values = torch.rand(batch_size)
        visit_counts = torch.randint(0, 100, (batch_size,)).float()
        parent_visits = torch.ones(batch_size) * 100
        priors = torch.rand(batch_size)
        priors = priors / priors.sum()
        
        ucb_scores = processor.compute_batched_ucb_hybrid(
            q_values, visit_counts, parent_visits, priors
        )
        
        assert ucb_scores.shape == (batch_size,)
        assert torch.all(torch.isfinite(ucb_scores))
        
        # Get stats
        stats = processor.get_performance_stats()
        assert 'cpu_fraction' in stats
        assert 'n_cpu_workers' in stats
        
        # Cleanup
        processor.shutdown()
        
    def test_gpu_tree_kernels_integration(self):
        """Test GPU tree kernels"""
        kernels = GPUTreeKernels()
        
        # Create simple tree structure
        n_nodes = 100
        max_children = 5
        
        # Create a proper tree structure where children indices are valid
        children_tensor = torch.full((n_nodes, max_children), -1)
        # Only first 50 nodes have children
        for i in range(50):
            # Each node has 2-3 children
            n_children = min(3, n_nodes - i * 2 - 1)
            for j in range(n_children):
                child_idx = min(i * 2 + j + 1, n_nodes - 1)
                children_tensor[i, j] = child_idx
        
        q_values = torch.rand(n_nodes)
        visit_counts = torch.randint(1, 100, (n_nodes,)).float()
        priors = torch.rand(n_nodes, max_children)
        priors = priors / priors.sum(dim=1, keepdim=True)  # Normalize
        
        # Test parallel selection on first 5 nodes
        node_indices = torch.arange(5)
        selected = kernels.parallel_select_children(
            node_indices, children_tensor, q_values, 
            visit_counts, priors, c_puct=1.0
        )
        
        assert selected.shape == (5,)
        
    def test_state_delta_encoder_integration(self):
        """Test state delta encoder"""
        encoder = StateDeltaEncoder(
            state_shape=(2, 15, 15),
            checkpoint_interval=5
        )
        
        # Create sequence of states
        states = []
        base_state = torch.zeros(2, 15, 15)
        states.append(base_state)
        
        # Make incremental changes
        for i in range(10):
            new_state = states[-1].clone()
            x, y = i % 15, (i * 2) % 15
            new_state[i % 2, x, y] = 1.0
            states.append(new_state)
            
        # Encode path
        encoded = encoder.encode_path(states)
        
        # Verify we can reconstruct any state
        for i in range(len(states)):
            reconstructed = encoder.reconstruct_state(encoded, i)
            assert torch.allclose(reconstructed, states[i])
            
        # Check compression
        stats = encoder.get_compression_stats(states)
        assert stats['compression_ratio'] > 1.0
        
    def test_all_components_together(self):
        """Test that all components can work together"""
        # Create components
        cuda_kernels = create_cuda_kernels()
        hybrid_processor = create_hybrid_processor(n_cpu_workers=2)
        gpu_tree_kernels = GPUTreeKernels()
        state_encoder = StateDeltaEncoder((2, 8, 8))
        
        # Simulate MCTS-like workflow
        batch_size = 64
        
        # 1. State encoding
        states = [torch.rand(2, 8, 8) for _ in range(batch_size)]
        encoded = state_encoder.encode_path(states)
        
        # 2. UCB computation with hybrid processor
        q_values = torch.rand(batch_size)
        visit_counts = torch.randint(0, 50, (batch_size,)).float()
        parent_visits = torch.ones(batch_size) * 100
        priors = torch.rand(batch_size)
        priors = priors / priors.sum()
        
        ucb_scores = hybrid_processor.compute_batched_ucb_hybrid(
            q_values, visit_counts, parent_visits, priors
        )
        
        # 3. MinHash for diversity
        features = torch.rand(batch_size, 100)
        signatures = cuda_kernels.parallel_minhash(features, num_hashes=32)
        
        # 4. Entropy computation
        action_probs = torch.rand(batch_size, 10)
        action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)
        entropy = cuda_kernels.batch_entropy(action_probs)
        
        # Verify all outputs
        assert ucb_scores.shape == (batch_size,)
        assert signatures.shape == (batch_size, 32)
        assert entropy.shape == (batch_size,)
        assert all(torch.isfinite(t).all() for t in [ucb_scores, signatures, entropy])
        
        # Cleanup
        hybrid_processor.shutdown()
        
        print("All components working together successfully!")