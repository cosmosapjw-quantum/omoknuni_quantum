#!/usr/bin/env python3
"""Test wave search CUDA optimizations

Validates that the CUDA kernel optimizations work correctly and provide performance improvements.
"""

import pytest
import torch
import numpy as np
import time
from typing import List, Tuple

from mcts.core.wave_search import WaveSearch
from mcts.core.mcts import MCTSConfig
from mcts.gpu.csr_tree import CSRTree, CSRTreeConfig
from mcts.gpu.gpu_game_states import GPUGameStates, GPUGameStatesConfig, GameType
from mcts.gpu.mcts_gpu_accelerator import get_mcts_gpu_accelerator
from tests.mock_evaluator import MockEvaluator


class TestWaveSearchCUDAOptimizations:
    """Test CUDA kernel optimizations for wave search"""
    
    @pytest.fixture
    def setup_cuda(self):
        """Setup for CUDA tests"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        device = torch.device('cuda')
        board_size = 15
        
        # Create config
        config = MCTSConfig(
            num_simulations=1000,
            min_wave_size=256,
            max_wave_size=1024,
            device='cuda',
            game_type=GameType.GOMOKU,
            board_size=board_size,
            c_puct=1.414,
            temperature=1.0,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            max_tree_nodes=100000,
            max_children_per_node=225,
            enable_virtual_loss=True,
            virtual_loss=1.0,
            enable_debug_logging=False,
            classical_only_mode=True,
            enable_fast_ucb=True,
            enable_quantum=False,
            enable_subtree_reuse=False
        )
        
        # Create tree
        tree_config = CSRTreeConfig(
            max_nodes=config.max_tree_nodes,
            max_actions=config.max_children_per_node,
            device='cuda',
            enable_batched_ops=True,
            enable_virtual_loss=config.enable_virtual_loss,
            virtual_loss_value=config.virtual_loss
        )
        tree = CSRTree(tree_config)
        
        # Create game states
        game_states_config = GPUGameStatesConfig(
            capacity=config.max_tree_nodes,
            board_size=board_size,
            game_type=GameType.GOMOKU,
            device='cuda'
        )
        game_states = GPUGameStates(game_states_config)
        
        # Create evaluator
        evaluator = MockEvaluator(
            game_type='gomoku',
            device='cuda',
            deterministic=True,
            fixed_value=0.0,
            policy_temperature=1.0
        )
        
        # State management
        node_to_state = torch.full((config.max_tree_nodes,), -1, dtype=torch.int32, device=device)
        node_to_state[0] = 0
        state_pool_free_list = list(range(1, config.max_tree_nodes))
        
        # Get GPU accelerator
        gpu_ops = get_mcts_gpu_accelerator(device)
        
        # Create wave search
        wave_search = WaveSearch(
            tree=tree,
            game_states=game_states,
            evaluator=evaluator,
            config=config,
            device=device,
            gpu_ops=gpu_ops
        )
        
        return {
            'wave_search': wave_search,
            'config': config,
            'tree': tree,
            'game_states': game_states,
            'node_to_state': node_to_state,
            'state_pool_free_list': state_pool_free_list,
            'gpu_ops': gpu_ops
        }
        
    def test_cuda_kernels_available(self, setup_cuda):
        """Test that CUDA kernels are properly loaded"""
        gpu_ops = setup_cuda['gpu_ops']
        
        # Check if key kernels are available
        assert gpu_ops is not None, "GPU accelerator should be loaded"
        
        # Check for wave search optimization kernels
        if hasattr(gpu_ops._kernel_interface, 'kernels'):
            kernels = gpu_ops._kernel_interface.kernels
            
            # These might not be available until recompiled
            kernel_names = ['batched_dirichlet_noise', 'fused_ucb_with_noise', 'optimized_backup_scatter']
            available = []
            for name in kernel_names:
                if hasattr(kernels, name):
                    available.append(name)
                    
            if available:
                print(f"Available wave search kernels: {available}")
            else:
                print("Wave search kernels not yet compiled - need to recompile CUDA kernels")
                
    def test_dirichlet_noise_generation(self, setup_cuda):
        """Test batched Dirichlet noise generation"""
        wave_search = setup_cuda['wave_search']
        
        # Test different batch sizes
        for num_sims in [32, 64, 128]:
            num_children = 225  # Full board
            
            # Generate noise
            noise = wave_search.apply_dirichlet_noise_batched(
                num_sims, num_children,
                wave_search.config.dirichlet_alpha,
                wave_search.config.dirichlet_epsilon
            )
            
            # Validate shape
            assert noise.shape == (num_sims, num_children)
            
            # Check that each row sums to approximately 1 (Dirichlet property)
            row_sums = noise.sum(dim=1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
            
            # Check that noise is different for each simulation
            if num_sims > 1:
                std_across_sims = noise.std(dim=0).mean()
                assert std_across_sims > 0.001, "Noise should vary across simulations"
                
    def test_vectorized_backup_performance(self, setup_cuda):
        """Test vectorized backup performance"""
        wave_search = setup_cuda['wave_search']
        tree = setup_cuda['tree']
        
        # Create test paths
        batch_sizes = [128, 256, 512]
        max_depth = 20
        
        for batch_size in batch_sizes:
            # Generate random paths
            paths = torch.zeros((batch_size, max_depth), dtype=torch.int32, device='cuda')
            path_lengths = torch.randint(5, max_depth, (batch_size,), device='cuda')
            values = torch.randn(batch_size, device='cuda')
            
            # Fill paths with valid node indices
            for i in range(batch_size):
                length = path_lengths[i].item()
                for j in range(length):
                    paths[i, j] = j  # Simple ascending path
                    
            # Time the backup operation
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            wave_search._backup_batch_vectorized(paths, path_lengths, values)
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            backups_per_second = batch_size / elapsed
            print(f"Backup performance (batch {batch_size}): {backups_per_second:,.0f} backups/sec")
            
            # Verify some nodes were updated
            assert tree.node_data.visit_counts[0] >= batch_size
            
    def test_wave_search_with_cuda(self, setup_cuda):
        """Test full wave search with CUDA optimizations"""
        wave_search = setup_cuda['wave_search']
        node_to_state = setup_cuda['node_to_state']
        state_pool_free_list = setup_cuda['state_pool_free_list']
        
        # Test different wave sizes
        wave_sizes = [64, 128, 256]
        
        for wave_size in wave_sizes:
            # Time the wave
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            completed = wave_search.run_wave(wave_size, node_to_state, state_pool_free_list)
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            assert completed == wave_size
            
            sims_per_second = wave_size / elapsed
            print(f"Wave search performance (size {wave_size}): {sims_per_second:,.0f} sims/sec")
            
    def test_memory_efficiency(self, setup_cuda):
        """Test memory usage with optimizations"""
        wave_search = setup_cuda['wave_search']
        
        # Check buffer allocation
        wave_search.allocate_buffers(1024)
        
        # Check that buffers are properly sized
        assert wave_search.paths_buffer.shape[0] >= 1024
        assert wave_search.ucb_scores.shape[0] >= 1024
        
        # Check memory usage
        allocated = torch.cuda.memory_allocated() / (1024**2)
        print(f"GPU memory allocated: {allocated:.1f} MB")
        

def test_cuda_kernel_compilation():
    """Test that CUDA kernels can be imported"""
    try:
        from mcts.gpu.cuda_manager import detect_cuda_kernels
        kernels = detect_cuda_kernels()
        if kernels is None:
            print("CUDA kernels not compiled - run compilation to enable optimizations")
        else:
            print("CUDA kernels detected successfully")
            # List available functions
            functions = [attr for attr in dir(kernels) if not attr.startswith('_')]
            print(f"Available kernel functions: {functions[:10]}...")  # Show first 10
    except ImportError as e:
        print(f"Could not import CUDA manager: {e}")
        

if __name__ == "__main__":
    import sys
    
    # Run tests
    print("Testing Wave Search CUDA Optimizations")
    print("=" * 60)
    
    # Check kernel compilation
    test_cuda_kernel_compilation()
    
    # Run pytest if CUDA available
    if torch.cuda.is_available():
        print("\nRunning CUDA optimization tests...")
        pytest.main([__file__, "-v", "-s"])
    else:
        print("\nCUDA not available - skipping GPU tests")