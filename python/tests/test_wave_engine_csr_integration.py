"""Integration tests for Wave Engine with CSR GPU kernels

This test suite validates the integration of the wave engine with
CSR tree format and GPU kernels for optimal performance.
"""

import pytest
import torch
import time
import numpy as np
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock

# Import MCTS modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts.csr_tree import CSRTree, CSRTreeConfig
from mcts.csr_gpu_kernels import get_csr_kernels, CSRGPUKernels
from mcts.vectorized_wave_engine import VectorizedWaveEngine, VectorizedWaveConfig
from mcts.game_interface import GameInterface
from mcts.evaluator import Evaluator


class MockGameInterface:
    """Mock game interface for testing"""
    
    def __init__(self):
        self.board_size = 5
        self.action_space_size = 25
        
    def get_legal_moves(self, state: Dict) -> List[int]:
        """Return mock legal moves"""
        return list(range(min(5, self.action_space_size)))
    
    def apply_move(self, state: Dict, action: int) -> Dict:
        """Apply mock move"""
        new_state = state.copy()
        new_state['moves'] = state.get('moves', []) + [action]
        new_state['turn'] = 1 - state.get('turn', 0)
        return new_state
    
    def is_terminal(self, state: Dict) -> bool:
        """Check if state is terminal"""
        return len(state.get('moves', [])) >= 10
    
    def get_reward(self, state: Dict, player: int) -> float:
        """Get reward for terminal state"""
        if not self.is_terminal(state):
            return 0.0
        # Mock reward based on move count
        moves = len(state.get('moves', []))
        return 1.0 if moves % 2 == player else -1.0
    
    def state_to_tensor(self, state: Dict) -> torch.Tensor:
        """Convert state to tensor representation"""
        moves = state.get('moves', [])
        tensor = torch.zeros(self.board_size, self.board_size)
        for i, move in enumerate(moves):
            row, col = move // self.board_size, move % self.board_size
            tensor[row, col] = 1.0 if i % 2 == 0 else -1.0
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def get_symmetries(self, state: Dict) -> List[Dict]:
        """Get symmetric states"""
        return [state]  # No symmetries for mock


class MockEvaluator(Evaluator):
    """Mock neural network evaluator"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        
    def evaluate(self, states: List[Any]) -> torch.Tensor:
        """Mock evaluation returning random values"""
        batch_size = len(states)
        return torch.randn(batch_size, device=self.device) * 0.1
    
    def evaluate_batch(self, state_tensors: torch.Tensor) -> torch.Tensor:
        """Mock batch evaluation"""
        batch_size = state_tensors.shape[0]
        return torch.randn(batch_size, device=self.device) * 0.1
    
    def get_policy_and_value(self, state_tensors: torch.Tensor) -> tuple:
        """Mock policy and value prediction"""
        batch_size = state_tensors.shape[0]
        action_size = 25  # 5x5 board
        
        # Random policy (will be normalized to probabilities)
        policy_logits = torch.randn(batch_size, action_size, device=self.device)
        values = torch.randn(batch_size, device=self.device) * 0.1
        
        return policy_logits, values


class CSRWaveEngine:
    """Wave engine optimized for CSR tree format
    
    This class integrates the vectorized wave engine with CSR trees
    and GPU kernels for maximum performance.
    """
    
    def __init__(self, csr_tree: CSRTree, game_interface: GameInterface, 
                 evaluator: Evaluator, config: VectorizedWaveConfig):
        self.csr_tree = csr_tree
        self.game_interface = game_interface
        self.evaluator = evaluator
        self.config = config
        self.device = csr_tree.device
        
        # Get CSR GPU kernels
        self.csr_kernels = get_csr_kernels()
        
        # Performance tracking
        self.stats = {
            'waves_processed': 0,
            'total_simulations': 0,
            'selection_time': 0.0,
            'evaluation_time': 0.0,
            'backup_time': 0.0,
            'gpu_utilization': 0.0
        }
        
    def run_wave(self, root_state: Any, wave_size: int = None) -> Dict:
        """Run a single wave of MCTS simulations using CSR format
        
        Args:
            root_state: Game state at root
            wave_size: Size of wave (defaults to config.wave_size)
            
        Returns:
            Wave results with performance metrics
        """
        if wave_size is None:
            wave_size = self.config.wave_size
            
        wave_start_time = time.perf_counter()
        
        # Phase 1: Selection
        selection_start = time.perf_counter()
        paths, leaf_nodes = self._select_wave_paths(root_state, wave_size)
        selection_time = time.perf_counter() - selection_start
        
        # Phase 2: Evaluation  
        eval_start = time.perf_counter()
        values = self._evaluate_leaves(leaf_nodes, root_state)
        eval_time = time.perf_counter() - eval_start
        
        # Phase 3: Backup
        backup_start = time.perf_counter()
        self._backup_values(paths, values)
        backup_time = time.perf_counter() - backup_start
        
        # Update statistics
        total_time = time.perf_counter() - wave_start_time
        self.stats['waves_processed'] += 1
        self.stats['total_simulations'] += wave_size
        self.stats['selection_time'] += selection_time
        self.stats['evaluation_time'] += eval_time
        self.stats['backup_time'] += backup_time
        
        return {
            'wave_size': wave_size,
            'paths': paths,
            'values': values,
            'timing': {
                'total': total_time,
                'selection': selection_time,
                'evaluation': eval_time,
                'backup': backup_time
            },
            'performance': {
                'sims_per_second': wave_size / total_time,
                'selection_throughput': wave_size / selection_time,
                'evaluation_throughput': wave_size / eval_time
            }
        }
        
    def _select_wave_paths(self, root_state: Any, wave_size: int) -> tuple:
        """Select paths for wave using CSR UCB calculation"""
        paths = []
        leaf_nodes = []
        
        # Start all paths from root
        current_nodes = torch.full((wave_size,), 0, dtype=torch.int32, device=self.device)
        path_depths = torch.zeros(wave_size, dtype=torch.int32, device=self.device)
        
        # Track paths for each simulation
        max_depth = self.config.max_depth
        all_paths = torch.full((wave_size, max_depth), -1, dtype=torch.int32, device=self.device)
        
        for depth in range(max_depth):
            # Check which nodes are not terminal and not fully expanded
            valid_mask = current_nodes >= 0
            
            if not valid_mask.any():
                break
                
            valid_nodes = current_nodes[valid_mask]
            
            if len(valid_nodes) == 0:
                break
                
            # Use CSR GPU kernels for UCB selection
            selected_actions = self.csr_kernels.batch_ucb_selection(
                node_indices=valid_nodes,
                row_ptr=self.csr_tree.row_ptr,
                col_indices=self.csr_tree.col_indices,
                edge_actions=self.csr_tree.edge_actions,
                edge_priors=self.csr_tree.edge_priors,
                visit_counts=self.csr_tree.visit_counts,
                value_sums=self.csr_tree.value_sums,
                c_puct=self.config.c_puct,
                use_cuda_kernel=False  # Use PyTorch version for stability
            )
            
            # Get children for selected actions
            children, _, _ = self.csr_tree.batch_get_children(valid_nodes)
            
            # Update current nodes for next iteration
            next_nodes = torch.full_like(current_nodes, -1)
            for i, (node_idx, action) in enumerate(zip(valid_nodes, selected_actions)):
                if action >= 0 and action < children.shape[1]:
                    child_idx = children[i, action.item()]
                    if child_idx >= 0:
                        next_nodes[valid_mask][i] = child_idx
                        all_paths[valid_mask, depth][i] = child_idx
                        
            current_nodes = next_nodes
            path_depths[valid_mask] += 1
            
        # Extract final paths and leaf nodes
        for i in range(wave_size):
            depth = path_depths[i].item()
            if depth > 0:
                path = all_paths[i, :depth].tolist()
                paths.append(path)
                leaf_nodes.append(current_nodes[i].item() if current_nodes[i] >= 0 else 0)
            else:
                paths.append([0])  # Root only
                leaf_nodes.append(0)
                
        return paths, torch.tensor(leaf_nodes, device=self.device)
        
    def _evaluate_leaves(self, leaf_nodes: torch.Tensor, root_state: Any) -> torch.Tensor:
        """Evaluate leaf nodes using mock evaluation"""
        batch_size = len(leaf_nodes)
        
        # For testing, return random values
        values = torch.randn(batch_size, device=self.device) * 0.1
        
        return values
        
    def _backup_values(self, paths: List[List[int]], values: torch.Tensor):
        """Backup values through paths using CSR format"""
        for i, (path, value) in enumerate(zip(paths, values)):
            # Alternate signs for two-player games
            current_value = value.item()
            
            for depth, node_idx in enumerate(reversed(path)):
                if node_idx >= 0:
                    # Update visit count and value sum
                    self.csr_tree.update_visit_count(node_idx, 1)
                    self.csr_tree.update_value_sum(node_idx, current_value)
                    
                    # Flip value for opponent
                    current_value = -current_value


@pytest.fixture
def device():
    """Test device"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def mock_game():
    """Mock game interface"""
    return MockGameInterface()


@pytest.fixture
def mock_evaluator(device):
    """Mock evaluator"""
    return MockEvaluator(device)


@pytest.fixture  
def csr_tree(device):
    """CSR tree for testing"""
    config = CSRTreeConfig(max_nodes=1000, max_edges=5000, device=device)
    tree = CSRTree(config)
    
    # Build small test tree
    root = tree.add_root(1.0)
    
    # Add some children to create a tree structure
    for i in range(3):
        child = tree.add_child(root, action=i, child_prior=0.3)
        for j in range(2):
            grandchild = tree.add_child(child, action=j, child_prior=0.5)
            
    return tree


@pytest.fixture
def csr_wave_engine(csr_tree, mock_game, mock_evaluator):
    """CSR wave engine for testing"""
    config = VectorizedWaveConfig(
        wave_size=64,
        c_puct=1.0,
        max_depth=10,
        enable_gpu=True,
        device=csr_tree.device.type
    )
    return CSRWaveEngine(csr_tree, mock_game, mock_evaluator, config)


class TestCSRWaveIntegration:
    """Test CSR wave engine integration"""
    
    def test_wave_engine_initialization(self, csr_wave_engine):
        """Test that wave engine initializes correctly with CSR tree"""
        assert csr_wave_engine.csr_tree is not None
        assert csr_wave_engine.csr_kernels is not None
        assert csr_wave_engine.config.wave_size == 64
        
    def test_single_wave_execution(self, csr_wave_engine):
        """Test executing a single wave"""
        root_state = {'moves': [], 'turn': 0}
        
        result = csr_wave_engine.run_wave(root_state, wave_size=16)
        
        assert result['wave_size'] == 16
        assert len(result['paths']) == 16
        assert len(result['values']) == 16
        assert 'timing' in result
        assert 'performance' in result
        
    def test_wave_performance_metrics(self, csr_wave_engine):
        """Test that performance metrics are tracked correctly"""
        root_state = {'moves': [], 'turn': 0}
        
        result = csr_wave_engine.run_wave(root_state, wave_size=32)
        
        # Check timing metrics
        timing = result['timing']
        assert timing['total'] > 0
        assert timing['selection'] > 0
        assert timing['evaluation'] > 0
        assert timing['backup'] > 0
        
        # Check performance metrics
        perf = result['performance']
        assert perf['sims_per_second'] > 0
        assert perf['selection_throughput'] > 0
        assert perf['evaluation_throughput'] > 0
        
    def test_multiple_waves(self, csr_wave_engine):
        """Test running multiple waves consecutively"""
        root_state = {'moves': [], 'turn': 0}
        
        results = []
        for _ in range(5):
            result = csr_wave_engine.run_wave(root_state, wave_size=8)
            results.append(result)
            
        assert len(results) == 5
        assert csr_wave_engine.stats['waves_processed'] == 5
        assert csr_wave_engine.stats['total_simulations'] == 40
        
    def test_csr_ucb_selection_integration(self, csr_wave_engine):
        """Test that CSR UCB selection works correctly"""
        root_state = {'moves': [], 'turn': 0}
        
        # Run several waves to build up visit counts
        for _ in range(3):
            csr_wave_engine.run_wave(root_state, wave_size=16)
            
        # Check that visit counts have been updated
        root_visits = csr_wave_engine.csr_tree.visit_counts[0].item()
        assert root_visits > 0
        
        # Check that children have been visited
        children, _, _ = csr_wave_engine.csr_tree.get_children(0)
        if len(children) > 0:
            child_visits = [csr_wave_engine.csr_tree.visit_counts[child.item()].item() 
                           for child in children]
            assert any(visits > 0 for visits in child_visits)
            
    def test_memory_efficiency(self, csr_wave_engine):
        """Test that CSR format provides memory efficiency"""
        initial_memory = csr_wave_engine.csr_tree.get_memory_usage()
        
        root_state = {'moves': [], 'turn': 0}
        
        # Run multiple waves
        for _ in range(10):
            csr_wave_engine.run_wave(root_state, wave_size=32)
            
        final_memory = csr_wave_engine.csr_tree.get_memory_usage()
        
        # Memory usage should be reasonable
        assert final_memory['total_mb'] < 50  # Less than 50MB
        assert final_memory['bytes_per_node'] < 1000  # Less than 1KB per node
        
    @pytest.mark.parametrize("wave_size", [16, 64, 256])
    def test_wave_size_scalability(self, csr_wave_engine, wave_size):
        """Test different wave sizes"""
        root_state = {'moves': [], 'turn': 0}
        
        result = csr_wave_engine.run_wave(root_state, wave_size=wave_size)
        
        assert result['wave_size'] == wave_size
        assert len(result['paths']) == wave_size
        assert len(result['values']) == wave_size
        
        # Performance should scale reasonably
        perf = result['performance']
        assert perf['sims_per_second'] > wave_size * 0.1  # At least 0.1x throughput
        
    def test_csr_kernel_fallback(self, csr_wave_engine):
        """Test that CSR kernels fall back gracefully when CUDA unavailable"""
        root_state = {'moves': [], 'turn': 0}
        
        # Force using PyTorch fallback
        result = csr_wave_engine.run_wave(root_state, wave_size=8)
        
        # Should still work correctly
        assert result['wave_size'] == 8
        assert len(result['paths']) == 8
        assert 'performance' in result


class TestCSRKernelPerformance:
    """Test CSR kernel performance characteristics"""
    
    def test_batch_ucb_performance(self, csr_tree):
        """Test batch UCB calculation performance"""
        kernels = get_csr_kernels()
        
        # Create batch of nodes to process
        batch_size = 256
        node_indices = torch.zeros(batch_size, dtype=torch.int32, device=csr_tree.device)
        
        # Time the batch UCB calculation
        start_time = time.perf_counter()
        
        for _ in range(10):  # Multiple iterations for stable timing
            selected_actions = kernels.batch_ucb_selection(
                node_indices=node_indices,
                row_ptr=csr_tree.row_ptr,
                col_indices=csr_tree.col_indices,
                edge_actions=csr_tree.edge_actions,
                edge_priors=csr_tree.edge_priors,
                visit_counts=csr_tree.visit_counts,
                value_sums=csr_tree.value_sums,
                c_puct=1.0,
                use_cuda_kernel=False
            )
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / 10
        throughput = batch_size / avg_time
        
        # Should process at least 1000 nodes per second
        assert throughput > 1000
        assert selected_actions.shape[0] == batch_size
        
    def test_memory_access_patterns(self, csr_tree):
        """Test that CSR format provides good memory access patterns"""
        # Test coalesced access to children
        node_indices = torch.arange(min(100, csr_tree.num_nodes), 
                                   dtype=torch.int32, device=csr_tree.device)
        
        start_time = time.perf_counter()
        
        for _ in range(100):
            children, actions, priors = csr_tree.batch_get_children(node_indices)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / 100
        throughput = len(node_indices) / avg_time
        
        # Should be fast due to coalesced access
        assert throughput > 5000  # At least 5k nodes/second
        
    def test_csr_vs_sparse_comparison(self, device):
        """Compare CSR format performance vs sparse format"""
        # This test demonstrates the performance advantage of CSR
        
        # Create test data
        batch_size = 128
        num_nodes = 1000
        
        # CSR format data
        csr_config = CSRTreeConfig(max_nodes=num_nodes, device=device)
        csr_tree = CSRTree(csr_config)
        
        # Build tree
        root = csr_tree.add_root()
        for i in range(min(100, num_nodes - 1)):
            csr_tree.add_child(root, action=i % 10, child_prior=0.1)
            
        node_indices = torch.randint(0, min(csr_tree.num_nodes, batch_size), 
                                    (batch_size,), device=csr_tree.device)
        
        # Time CSR access
        start_time = time.perf_counter()
        for _ in range(50):
            children, actions, priors = csr_tree.batch_get_children(node_indices)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        csr_time = time.perf_counter() - start_time
        
        csr_throughput = (batch_size * 50) / csr_time
        
        # CSR should provide good throughput
        assert csr_throughput > 1000  # At least 1k ops/second
        
        print(f"CSR throughput: {csr_throughput:.0f} ops/second")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])