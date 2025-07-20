"""Benchmark to measure performance improvements from optimizations"""

import pytest
import torch
import time
import numpy as np
from unittest.mock import Mock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.mcts.core.wave_search import WaveSearch
from python.mcts.gpu.csr_tree import CSRTree, CSRTreeConfig
from python.mcts.gpu.gpu_game_states import GPUGameStates, GPUGameStatesConfig, GameType
from python.mcts.core.mcts_config import MCTSConfig
from python.mcts.gpu.cuda_manager import detect_cuda_kernels


class TestOptimizationBenchmark:
    """Benchmark tests to measure optimization improvements"""
    
    def setup_method(self):
        """Setup test environment"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create MCTS config
        self.config = MCTSConfig(
            num_simulations=1000,
            c_puct=1.4,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            device=str(self.device),
            enable_virtual_loss=True,
            virtual_loss=1.0,
            max_wave_size=64,
            board_size=15,
            game_type=GameType.GOMOKU,
            max_children_per_node=225
        )
        
        # Create tree
        tree_config = CSRTreeConfig(
            max_nodes=1000000,  # Increased for benchmark
            max_edges=10000000,  # Increased for benchmark
            max_actions=225,
            device=str(self.device),
            enable_virtual_loss=True,
            virtual_loss_value=-1.0,
            batch_size=64
        )
        self.tree = CSRTree(tree_config)
        
        # Create game states
        game_config = GPUGameStatesConfig(
            capacity=100000,
            game_type=GameType.GOMOKU,
            board_size=15,
            device=str(self.device)
        )
        self.game_states = GPUGameStates(game_config)
        
        # Create mock evaluator
        self.evaluator = Mock()
        self.evaluator.evaluate_batch = Mock(return_value=(
            np.random.rand(64, 225),  # policies
            np.random.rand(64)        # values
        ))
        
        # Detect GPU ops
        self.gpu_ops = detect_cuda_kernels()
        
        # Create wave search
        self.wave_search = WaveSearch(
            tree=self.tree,
            game_states=self.game_states,
            evaluator=self.evaluator,
            config=self.config,
            device=self.device,
            gpu_ops=self.gpu_ops
        )
        
    def _setup_tree_with_nodes(self, num_children_per_node=50, depth=3):
        """Create a tree with multiple levels for testing"""
        self.tree.reset()
        root_idx = 0
        
        nodes_by_level = [[root_idx]]
        
        for level in range(depth):
            next_level = []
            for parent in nodes_by_level[level]:
                # Add children
                for i in range(num_children_per_node):
                    child_idx = self.tree.add_child(parent, action=i, child_prior=1.0/num_children_per_node)
                    self.tree.node_data.visit_counts[child_idx] = 10 + np.random.randint(0, 10)
                    self.tree.node_data.value_sums[child_idx] = 5.0 + np.random.rand()
                    next_level.append(child_idx)
            nodes_by_level.append(next_level)
            
        return nodes_by_level
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for benchmark")
    def test_virtual_loss_performance(self):
        """Benchmark virtual loss application performance"""
        # Setup tree
        nodes_by_level = self._setup_tree_with_nodes(num_children_per_node=20, depth=2)
        
        # Benchmark parallel selection with virtual loss
        num_iterations = 100
        wave_size = 32
        
        # Setup wave search state
        self.wave_search.node_to_state = torch.zeros(100000, dtype=torch.int32, device=self.device)
        
        # Mock legal moves
        self.game_states.get_legal_moves_mask = Mock(
            return_value=torch.ones((wave_size, 225), dtype=torch.bool, device=self.device)
        )
        
        # Warm up
        for _ in range(10):
            active_indices = torch.arange(wave_size, device=self.device)
            active_nodes = torch.randint(0, len(nodes_by_level[1]), (wave_size,), device=self.device, dtype=torch.int32)
            
            batch_children, batch_actions, batch_priors = self.tree.batch_get_children(active_nodes)
            valid_children_mask = batch_children >= 0
            
            self.wave_search._parallel_select_with_virtual_loss(
                active_indices, active_nodes, batch_children, batch_priors,
                valid_children_mask, depth=1
            )
        
        # Measure time
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_iterations):
            active_indices = torch.arange(wave_size, device=self.device)
            active_nodes = torch.randint(0, len(nodes_by_level[1]), (wave_size,), device=self.device, dtype=torch.int32)
            
            batch_children, batch_actions, batch_priors = self.tree.batch_get_children(active_nodes)
            valid_children_mask = batch_children >= 0
            
            selected = self.wave_search._parallel_select_with_virtual_loss(
                active_indices, active_nodes, batch_children, batch_priors,
                valid_children_mask, depth=1
            )
            
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        selections_per_second = (num_iterations * wave_size) / elapsed
        ms_per_wave = (elapsed / num_iterations) * 1000
        
        print(f"\nVirtual Loss Performance:")
        print(f"  Selections per second: {selections_per_second:,.0f}")
        print(f"  Time per wave (32 sims): {ms_per_wave:.2f}ms")
        print(f"  GPU ops available: {self.gpu_ops is not None}")
        
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for benchmark")
    def test_dirichlet_noise_performance(self):
        """Benchmark Dirichlet noise generation performance"""
        if self.gpu_ops is None or not hasattr(self.gpu_ops, 'batched_dirichlet_noise'):
            pytest.skip("CUDA kernels not available")
            
        num_iterations = 1000
        num_sims = 32
        num_actions = 225
        
        # Warm up
        for _ in range(10):
            noise = self.gpu_ops.batched_dirichlet_noise(
                num_sims, num_actions, 0.3, 1.0, self.device
            )
            
        # Benchmark CUDA kernel
        torch.cuda.synchronize()
        start_cuda = time.time()
        
        for _ in range(num_iterations):
            noise = self.gpu_ops.batched_dirichlet_noise(
                num_sims, num_actions, 0.3, 1.0, self.device
            )
            
        torch.cuda.synchronize()
        cuda_time = time.time() - start_cuda
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start_pytorch = time.time()
        
        for _ in range(num_iterations):
            noise = torch.zeros((num_sims, num_actions), device=self.device)
            for i in range(num_sims):
                noise[i] = torch.distributions.Dirichlet(
                    torch.full((num_actions,), 0.3, device=self.device)
                ).sample()
                
        torch.cuda.synchronize()
        pytorch_time = time.time() - start_pytorch
        
        speedup = pytorch_time / cuda_time
        
        print(f"\nDirichlet Noise Performance:")
        print(f"  CUDA kernel time: {cuda_time:.3f}s ({cuda_time/num_iterations*1000:.2f}ms per batch)")
        print(f"  PyTorch time: {pytorch_time:.3f}s ({pytorch_time/num_iterations*1000:.2f}ms per batch)")
        print(f"  Speedup: {speedup:.1f}x")
        
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for benchmark")
    def test_overall_wave_performance(self):
        """Benchmark overall wave search performance"""
        # Setup a more realistic tree
        self._setup_tree_with_nodes(num_children_per_node=50, depth=3)
        
        # Mock game states
        self.wave_search.node_to_state = torch.zeros(100000, dtype=torch.int32, device=self.device)
        for i in range(1000):
            self.wave_search.node_to_state[i] = i % 100
            
        # Mock legal moves
        self.game_states.get_legal_moves_mask = Mock(
            return_value=torch.ones((64, 225), dtype=torch.bool, device=self.device)
        )
        
        # Measure wave performance
        num_waves = 50
        wave_size = 64
        
        # Warm up
        for _ in range(5):
            self.wave_search._run_single_wave(torch.arange(wave_size, device=self.device))
            
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_waves):
            active_simulations = torch.ones(wave_size, dtype=torch.bool, device=self.device)
            self.wave_search._run_single_wave(torch.arange(wave_size, device=self.device))
            
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        waves_per_second = num_waves / elapsed
        sims_per_second = (num_waves * wave_size) / elapsed
        
        print(f"\nOverall Wave Performance:")
        print(f"  Waves per second: {waves_per_second:.1f}")
        print(f"  Simulations per second: {sims_per_second:,.0f}")
        print(f"  Time per wave (64 sims): {elapsed/num_waves*1000:.2f}ms")
        
        # This should be significantly faster than before optimizations
        # Target: >1000 sims/second on RTX 3060 Ti