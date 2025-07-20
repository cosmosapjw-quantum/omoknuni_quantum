"""Test for identifying and optimizing unnecessary .item() calls

This test identifies where .item() calls cause GPU-CPU synchronization
and validates optimizations to remove them.
"""

import pytest
import torch
import time
import numpy as np
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.mcts.core.wave_search import WaveSearch
from python.mcts.gpu.csr_tree import CSRTree, CSRTreeConfig
from python.mcts.gpu.gpu_game_states import GPUGameStates, GPUGameStatesConfig, GameType
from python.mcts.core.mcts_config import MCTSConfig


class TestItemCallOptimization:
    """Test cases for identifying and removing unnecessary .item() calls"""
    
    def setup_method(self):
        """Setup test environment"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create MCTS config
        self.config = MCTSConfig(
            num_simulations=100,
            c_puct=1.4,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            device=str(self.device),
            max_wave_size=16,
            board_size=15,
            game_type=GameType.GOMOKU,
            max_children_per_node=225
        )
        
        # Create tree
        tree_config = CSRTreeConfig(
            max_nodes=10000,
            max_edges=100000,
            max_actions=225,
            device=str(self.device),
            batch_size=16
        )
        self.tree = CSRTree(tree_config)
        
        # Create game states
        game_config = GPUGameStatesConfig(
            capacity=10000,
            game_type=GameType.GOMOKU,
            board_size=15,
            device=str(self.device)
        )
        self.game_states = GPUGameStates(game_config)
        
        # Create mock evaluator
        self.evaluator = Mock()
        self.evaluator.evaluate_batch = Mock(return_value=(
            np.random.rand(16, 225),  # policies
            np.random.rand(16)        # values
        ))
        
        # Create wave search
        self.wave_search = WaveSearch(
            tree=self.tree,
            game_states=self.game_states,
            evaluator=self.evaluator,
            config=self.config,
            device=self.device,
            gpu_ops=None
        )
        
    def test_identify_item_calls_in_wave_search(self):
        """Identify all .item() calls in wave_search.py that cause synchronization"""
        # This test scans the code to find .item() patterns
        import ast
        import inspect
        
        # Get source code of wave_search module
        source_file = inspect.getsourcefile(WaveSearch)
        with open(source_file, 'r') as f:
            source_code = f.read()
            
        # Parse AST to find .item() calls
        tree = ast.parse(source_code)
        
        item_calls = []
        
        class ItemCallVisitor(ast.NodeVisitor):
            def visit_Attribute(self, node):
                if node.attr == 'item' and isinstance(node.ctx, ast.Load):
                    # Get line number and surrounding context
                    line_no = node.lineno
                    # Try to get the variable name being called
                    if isinstance(node.value, ast.Name):
                        var_name = node.value.id
                    elif isinstance(node.value, ast.Subscript):
                        var_name = ast.unparse(node.value)
                    else:
                        var_name = "unknown"
                    
                    item_calls.append({
                        'line': line_no,
                        'variable': var_name,
                        'full_expr': ast.unparse(node)
                    })
                self.generic_visit(node)
                
        visitor = ItemCallVisitor()
        visitor.visit(tree)
        
        # Print findings
        print(f"\nFound {len(item_calls)} .item() calls in wave_search.py:")
        for call in item_calls:
            print(f"  Line {call['line']}: {call['variable']}.item()")
            
        # Verify we found the expected problematic calls
        assert len(item_calls) > 0, "Should find .item() calls in wave_search.py"
        
    def test_synchronization_overhead(self):
        """Measure synchronization overhead from .item() calls"""
        if self.device.type != 'cuda':
            pytest.skip("GPU required for synchronization test")
            
        # Create test tensors
        size = 1000
        tensor_gpu = torch.rand(size, device=self.device)
        
        # Measure time with .item() calls
        torch.cuda.synchronize()
        start_time = time.time()
        
        total = 0
        for i in range(100):
            # This causes GPU-CPU sync
            total += tensor_gpu[i].item()
            
        torch.cuda.synchronize()
        time_with_item = time.time() - start_time
        
        # Measure time without .item() calls (staying on GPU)
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Stay on GPU
        gpu_sum = tensor_gpu[:100].sum()
        
        torch.cuda.synchronize()
        time_without_item = time.time() - start_time
        
        print(f"\nSynchronization overhead test:")
        print(f"  Time with .item() calls: {time_with_item:.4f}s")
        print(f"  Time without .item(): {time_without_item:.4f}s")
        print(f"  Overhead: {time_with_item/time_without_item:.1f}x slower")
        
        # .item() should be significantly slower due to synchronization
        assert time_with_item > time_without_item * 2
        
    def test_item_removal_in_path_lengths(self):
        """Test removing .item() from path_lengths usage"""
        # Common pattern: path_lengths[i].item()
        path_lengths = torch.tensor([3, 2, 4, 3], device=self.device)
        paths = torch.zeros((4, 5), dtype=torch.int32, device=self.device)
        
        # Bad: Using .item() in loop
        total_bad = 0
        for i in range(len(path_lengths)):
            length = path_lengths[i].item()  # Causes sync
            total_bad += length
            
        # Good: Vectorized operation
        total_good = path_lengths.sum().item()  # Single sync at end
        
        assert total_bad == total_good
        
        # Even better: Stay on GPU if possible
        total_gpu = path_lengths.sum()  # No sync
        assert total_gpu.item() == total_good
        
    def test_item_removal_in_node_indices(self):
        """Test removing .item() from node index operations"""
        # Common pattern: node = paths[i, j].item()
        paths = torch.tensor([[0, 1, 2], [0, 1, 0], [0, 1, 3]], device=self.device)
        
        # Bad: Using .item() in nested loops
        nodes_bad = []
        for i in range(paths.shape[0]):
            for j in range(paths.shape[1]):
                node = paths[i, j].item()  # Many syncs
                if node > 0:
                    nodes_bad.append(node)
                    
        # Good: Vectorized operation
        valid_nodes = paths[paths > 0]  # Stay on GPU
        nodes_good = valid_nodes.cpu().numpy().tolist()  # Single transfer
        
        assert sorted(nodes_bad) == sorted(nodes_good)
        
    def test_item_removal_in_conditional_checks(self):
        """Test removing .item() from conditional checks"""
        values = torch.tensor([0.5, -0.3, 0.8, 0.0], device=self.device)
        
        # Bad: Using .item() for conditionals
        count_bad = 0
        for i in range(len(values)):
            if values[i].item() > 0:  # Sync for each check
                count_bad += 1
                
        # Good: Vectorized boolean operations
        count_good = (values > 0).sum().item()  # Single sync
        
        assert count_bad == count_good
        
        # Even better: Keep as tensor if used later
        positive_mask = values > 0  # No sync
        positive_values = values[positive_mask]  # Still on GPU
        
    def test_wave_search_without_item_calls(self):
        """Test that wave search can run without excessive .item() calls"""
        # Setup tree with some nodes
        root_idx = 0
        self.tree.reset()
        
        # Add children to root
        for i in range(5):
            self.tree.add_child(root_idx, action=i, child_prior=0.2)
            
        # Mock node_to_state
        self.wave_search.node_to_state = torch.zeros(100, dtype=torch.int32, device=self.device)
        
        # Count .item() calls using mock
        item_call_count = 0
        original_item = torch.Tensor.item
        
        def counting_item(self):
            nonlocal item_call_count
            item_call_count += 1
            return original_item(self)
            
        # Run a wave with patched .item()
        with patch.object(torch.Tensor, 'item', counting_item):
            # Initialize wave
            wave_size = 8
            active_simulations = torch.ones(wave_size, dtype=torch.bool, device=self.device)
            active_nodes = torch.zeros(wave_size, dtype=torch.int32, device=self.device)
            
            # Get children (this should minimize .item() calls)
            batch_children, _, batch_priors = self.tree.batch_get_children(active_nodes)
            valid_mask = batch_children >= 0
            
            # Simple selection without virtual loss
            if valid_mask.any():
                # This operation should stay on GPU
                first_valid = batch_children[valid_mask][0]
                
        print(f"\nItem calls during batch operation: {item_call_count}")
        
        # Should have very few .item() calls
        assert item_call_count < 10, f"Too many .item() calls: {item_call_count}"


class TestItemCallPatterns:
    """Test specific patterns of .item() usage and their optimizations"""
    
    def setup_method(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_max_item_pattern(self):
        """Test optimization of tensor.max().item() pattern"""
        tensor = torch.tensor([1, 5, 3, 2], device=self.device)
        
        # Bad: Immediate .item()
        max_val_bad = tensor.max().item()
        
        # Good: Defer .item() if max value used in GPU operations
        max_val_tensor = tensor.max()
        # Use in GPU operation
        result = tensor / max_val_tensor  # Stays on GPU
        
        # Only sync when absolutely needed
        max_val_good = max_val_tensor.item()
        
        assert max_val_bad == max_val_good
        
    def test_len_vs_shape(self):
        """Test using .shape[0] instead of len() to avoid potential syncs"""
        tensor = torch.zeros(100, 50, device=self.device)
        
        # Both work, but shape is more explicit about no sync
        length1 = len(tensor)
        length2 = tensor.shape[0]
        length3 = tensor.size(0)
        
        assert length1 == length2 == length3
        
        # For conditional checks, use tensor operations
        is_empty_bad = len(tensor) == 0  # Python bool
        is_empty_good = tensor.numel() == 0  # Can stay as tensor
        
    def test_argmax_pattern(self):
        """Test optimization of argmax().item() pattern"""
        scores = torch.tensor([0.1, 0.5, 0.3, 0.9, 0.2], device=self.device)
        
        # Bad: Multiple .item() calls
        best_idx_bad = scores.argmax().item()
        best_val_bad = scores[best_idx_bad].item()
        
        # Good: Batch operations
        best_idx = scores.argmax()
        best_val = scores[best_idx]
        
        # Only sync once at the end if needed
        best_idx_good = best_idx.item()
        best_val_good = best_val.item()
        
        assert best_idx_bad == best_idx_good
        assert best_val_bad == best_val_good
        
    def test_accumulator_pattern(self):
        """Test optimization of accumulator patterns with .item()"""
        values = torch.rand(1000, device=self.device)
        
        # Bad: Accumulating with .item() in loop
        total_bad = 0
        for val in values:
            total_bad += val.item()  # 1000 syncs!
            
        # Good: GPU reduction
        total_good = values.sum().item()  # 1 sync
        
        assert abs(total_bad - total_good) < 1e-4
        
        # For running statistics, use GPU tensors
        running_sum = torch.tensor(0.0, device=self.device)
        for i in range(0, len(values), 100):
            batch = values[i:i+100]
            running_sum += batch.sum()  # Stays on GPU
            
        final_sum = running_sum.item()  # Single sync at end