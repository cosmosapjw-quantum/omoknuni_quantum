"""Tests for UCB selection logic

Tests cover:
- UCB formula calculation
- Single node selection
- Batch selection
- Virtual loss handling
- Temperature scaling
- Tie-breaking logic
- Edge cases
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch

from mcts.gpu.ucb_selector import UCBSelector, UCBConfig


@pytest.fixture
def ucb_config():
    """Create UCB configuration"""
    return UCBConfig(
        c_puct=1.4,
        temperature=1.0,
        enable_virtual_loss=True,
        virtual_loss_value=-1.0,
        device='cpu'
    )


@pytest.fixture
def selector(ucb_config):
    """Create UCBSelector instance"""
    return UCBSelector(ucb_config)


@pytest.fixture
def sample_node_data():
    """Create sample node data for testing"""
    return {
        'parent_visits': 100,
        'child_visits': torch.tensor([10, 20, 0, 5], dtype=torch.int32),
        'child_values': torch.tensor([5.0, 12.0, 0.0, 2.0], dtype=torch.float32),
        'child_priors': torch.tensor([0.25, 0.3, 0.3, 0.15], dtype=torch.float32)
    }


@pytest.fixture
def batch_node_data():
    """Create batch node data for testing"""
    batch_size = 4
    max_children = 5
    
    return {
        'parent_visits': torch.tensor([100, 50, 200, 0], dtype=torch.int32),
        'children_visits': torch.tensor([
            [10, 20, 0, 5, 0],      # Node 0: mixed visited/unvisited
            [0, 0, 0, 0, 0],        # Node 1: all unvisited
            [15, 15, 15, 15, 15],   # Node 2: all visited equally
            [1, 2, 3, 0, 0]         # Node 3: partially visited
        ], dtype=torch.int32),
        'children_values': torch.tensor([
            [5.0, 12.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [7.5, 7.5, 7.5, 7.5, 7.5],
            [0.5, 1.2, 1.8, 0.0, 0.0]
        ], dtype=torch.float32),
        'children_priors': torch.tensor([
            [0.2, 0.3, 0.3, 0.15, 0.05],
            [0.1, 0.2, 0.3, 0.3, 0.1],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.25, 0.25, 0.25, 0.25, 0.0]
        ], dtype=torch.float32),
        'valid_mask': torch.tensor([
            [True, True, True, True, False],
            [True, True, True, True, True],
            [True, True, True, True, True],
            [True, True, True, False, False]
        ], dtype=torch.bool)
    }


class TestUCBConfig:
    """Test UCBConfig configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = UCBConfig()
        assert config.c_puct == 1.4
        assert config.temperature == 1.0
        assert config.enable_virtual_loss == True
        assert config.virtual_loss_value == -1.0
        assert config.device == 'cuda'
        
    def test_custom_config(self):
        """Test custom configuration"""
        config = UCBConfig(
            c_puct=2.0,
            temperature=0.5,
            enable_virtual_loss=False,
            device='cpu'
        )
        assert config.c_puct == 2.0
        assert config.temperature == 0.5
        assert config.enable_virtual_loss == False
        assert config.device == 'cpu'


class TestUCBSelectorInitialization:
    """Test UCBSelector initialization"""
    
    def test_initialization(self, selector, ucb_config):
        """Test basic initialization"""
        assert selector.config == ucb_config
        assert selector.device == torch.device('cpu')
        assert selector._empty_tensors_cache is None
        
    def test_cuda_device_fallback(self):
        """Test CUDA device fallback to CPU"""
        with patch('torch.cuda.is_available', return_value=False):
            config = UCBConfig(device='cuda')
            selector = UCBSelector(config)
            assert selector.device == torch.device('cpu')


class TestSingleNodeSelection:
    """Test single node UCB selection"""
    
    def test_basic_selection(self, selector, sample_node_data):
        """Test basic UCB selection for single node"""
        best_idx = selector.select_single(
            parent_visits=sample_node_data['parent_visits'],
            child_visits=sample_node_data['child_visits'],
            child_values=sample_node_data['child_values'],
            child_priors=sample_node_data['child_priors']
        )
        
        # Should select a valid child
        assert 0 <= best_idx < len(sample_node_data['child_visits'])
        
    def test_empty_children(self, selector):
        """Test selection with no children"""
        best_idx = selector.select_single(
            parent_visits=100,
            child_visits=torch.tensor([], dtype=torch.int32),
            child_values=torch.tensor([], dtype=torch.float32),
            child_priors=torch.tensor([], dtype=torch.float32)
        )
        
        assert best_idx == -1
        
    def test_all_unvisited(self, selector):
        """Test selection when all children unvisited"""
        child_visits = torch.zeros(4, dtype=torch.int32)
        child_values = torch.zeros(4, dtype=torch.float32)
        child_priors = torch.tensor([0.1, 0.4, 0.3, 0.2], dtype=torch.float32)
        
        best_idx = selector.select_single(100, child_visits, child_values, child_priors)
        
        # Should select based on exploration term (priors)
        # Child 1 has highest prior (0.4)
        assert best_idx == 1
        
    def test_custom_c_puct(self, selector, sample_node_data):
        """Test selection with custom c_puct value"""
        # High c_puct favors exploration
        best_idx_high = selector.select_single(
            parent_visits=sample_node_data['parent_visits'],
            child_visits=sample_node_data['child_visits'],
            child_values=sample_node_data['child_values'],
            child_priors=sample_node_data['child_priors'],
            c_puct=10.0
        )
        
        # Low c_puct favors exploitation
        best_idx_low = selector.select_single(
            parent_visits=sample_node_data['parent_visits'],
            child_visits=sample_node_data['child_visits'],
            child_values=sample_node_data['child_values'],
            child_priors=sample_node_data['child_priors'],
            c_puct=0.1
        )
        
        # Results may differ based on exploration vs exploitation
        assert 0 <= best_idx_high < 4
        assert 0 <= best_idx_low < 4
        
    def test_ucb_calculation_correctness(self, selector):
        """Test correctness of UCB calculation"""
        parent_visits = 100
        child_visits = torch.tensor([10, 0], dtype=torch.int32)
        child_values = torch.tensor([5.0, 0.0], dtype=torch.float32)
        child_priors = torch.tensor([0.5, 0.5], dtype=torch.float32)
        c_puct = 1.0
        
        # Calculate expected UCB scores manually
        # Child 0: Q = 5/10 = 0.5, U = 1.0 * 0.5 * sqrt(100) / (1 + 10) = 0.454...
        # UCB_0 = 0.5 + 0.454... = 0.954...
        
        # Child 1: Q = 0, U = 1.0 * 0.5 * sqrt(100) / (1 + 0) = 5.0
        # UCB_1 = 0 + 5.0 = 5.0
        
        best_idx = selector.select_single(parent_visits, child_visits, child_values, child_priors, c_puct)
        
        # Unvisited child should have higher UCB due to exploration bonus
        assert best_idx == 1


class TestBatchSelection:
    """Test batch UCB selection"""
    
    def test_basic_batch_selection(self, selector, batch_node_data):
        """Test basic batch selection"""
        selected_indices, selected_scores = selector.select_batch(
            parent_visits=batch_node_data['parent_visits'],
            children_visits=batch_node_data['children_visits'],
            children_values=batch_node_data['children_values'],
            children_priors=batch_node_data['children_priors'],
            valid_mask=batch_node_data['valid_mask']
        )
        
        batch_size = len(batch_node_data['parent_visits'])
        assert len(selected_indices) == batch_size
        assert len(selected_scores) == batch_size
        
        # Check valid selections
        for i in range(batch_size):
            if batch_node_data['valid_mask'][i].any():
                assert 0 <= selected_indices[i] < batch_node_data['valid_mask'].shape[1]
            else:
                assert selected_indices[i] == -1
                
    def test_empty_batch(self, selector):
        """Test selection with empty batch"""
        selected_indices, selected_scores = selector.select_batch(
            parent_visits=torch.tensor([], dtype=torch.int32),
            children_visits=torch.zeros((0, 5), dtype=torch.int32),
            children_values=torch.zeros((0, 5), dtype=torch.float32),
            children_priors=torch.zeros((0, 5), dtype=torch.float32),
            valid_mask=torch.zeros((0, 5), dtype=torch.bool)
        )
        
        assert len(selected_indices) == 0
        assert len(selected_scores) == 0
        
    def test_no_valid_children(self, selector):
        """Test batch with no valid children"""
        batch_size = 3
        max_children = 4
        
        selected_indices, selected_scores = selector.select_batch(
            parent_visits=torch.ones(batch_size, dtype=torch.int32),
            children_visits=torch.zeros((batch_size, max_children), dtype=torch.int32),
            children_values=torch.zeros((batch_size, max_children), dtype=torch.float32),
            children_priors=torch.ones((batch_size, max_children), dtype=torch.float32) * 0.25,
            valid_mask=torch.zeros((batch_size, max_children), dtype=torch.bool)
        )
        
        # All should be -1 (no valid selection)
        assert torch.all(selected_indices == -1)
        assert torch.all(selected_scores == 0)
        
    def test_all_unvisited_nodes(self, selector):
        """Test batch selection with all unvisited nodes"""
        torch.manual_seed(42)  # For reproducibility
        
        batch_size = 2
        max_children = 3
        
        # Different prior distributions
        children_priors = torch.tensor([
            [0.5, 0.3, 0.2],  # Node 0: clear preference
            [0.33, 0.33, 0.34]  # Node 1: nearly uniform
        ], dtype=torch.float32)
        
        selected_indices, selected_scores = selector.select_batch(
            parent_visits=torch.ones(batch_size, dtype=torch.int32) * 100,
            children_visits=torch.zeros((batch_size, max_children), dtype=torch.int32),
            children_values=torch.zeros((batch_size, max_children), dtype=torch.float32),
            children_priors=children_priors,
            valid_mask=torch.ones((batch_size, max_children), dtype=torch.bool)
        )
        
        # Should select based on priors (stochastically)
        assert torch.all(selected_indices >= 0)
        assert torch.all(selected_indices < max_children)
        
    def test_temperature_scaling(self, selector, batch_node_data):
        """Test temperature effect on selection"""
        # Low temperature (more deterministic)
        indices_low_temp, _ = selector.select_batch(
            parent_visits=batch_node_data['parent_visits'],
            children_visits=batch_node_data['children_visits'],
            children_values=batch_node_data['children_values'],
            children_priors=batch_node_data['children_priors'],
            valid_mask=batch_node_data['valid_mask'],
            temperature=0.1
        )
        
        # High temperature (more exploration)
        indices_high_temp, _ = selector.select_batch(
            parent_visits=batch_node_data['parent_visits'],
            children_visits=batch_node_data['children_visits'],
            children_values=batch_node_data['children_values'],
            children_priors=batch_node_data['children_priors'],
            valid_mask=batch_node_data['valid_mask'],
            temperature=10.0
        )
        
        # Both should produce valid selections
        assert torch.all(indices_low_temp >= -1)
        assert torch.all(indices_high_temp >= -1)
        
    def test_parent_visits_zero(self, selector):
        """Test handling of zero parent visits"""
        selected_indices, selected_scores = selector.select_batch(
            parent_visits=torch.zeros(2, dtype=torch.int32),
            children_visits=torch.tensor([[10, 0], [5, 5]], dtype=torch.int32),
            children_values=torch.tensor([[5.0, 0.0], [2.5, 2.5]], dtype=torch.float32),
            children_priors=torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32),
            valid_mask=torch.ones((2, 2), dtype=torch.bool)
        )
        
        # Should handle gracefully (parent visits clamped to 1)
        assert torch.all(selected_indices >= 0)


class TestVirtualLoss:
    """Test virtual loss functionality"""
    
    def test_virtual_loss_application(self, selector, batch_node_data):
        """Test selection with virtual loss"""
        batch_size = 2
        max_children = 3
        
        # Create virtual loss counts
        virtual_loss_counts = torch.tensor([
            [0, 2, 0],  # Node 0: child 1 has virtual loss
            [1, 0, 0]   # Node 1: child 0 has virtual loss
        ], dtype=torch.int32)
        
        selected_indices, selected_scores = selector.select_batch_with_virtual_loss(
            parent_visits=torch.ones(batch_size, dtype=torch.int32) * 100,
            children_visits=torch.zeros((batch_size, max_children), dtype=torch.int32),
            children_values=torch.zeros((batch_size, max_children), dtype=torch.float32),
            children_priors=torch.ones((batch_size, max_children), dtype=torch.float32) / 3,
            virtual_loss_counts=virtual_loss_counts,
            valid_mask=torch.ones((batch_size, max_children), dtype=torch.bool)
        )
        
        # Virtual loss should discourage selection of those children
        # But with all unvisited, effect depends on exploration
        assert torch.all(selected_indices >= 0)
        
    def test_virtual_loss_disabled(self, batch_node_data):
        """Test with virtual loss disabled"""
        config = UCBConfig(enable_virtual_loss=False, device='cpu')
        selector = UCBSelector(config)
        
        virtual_loss_counts = torch.ones((2, 3), dtype=torch.int32) * 5
        
        # Should ignore virtual loss
        selected_indices, _ = selector.select_batch_with_virtual_loss(
            parent_visits=torch.ones(2, dtype=torch.int32) * 100,
            children_visits=torch.zeros((2, 3), dtype=torch.int32),
            children_values=torch.zeros((2, 3), dtype=torch.float32),
            children_priors=torch.ones((2, 3), dtype=torch.float32) / 3,
            virtual_loss_counts=virtual_loss_counts,
            valid_mask=torch.ones((2, 3), dtype=torch.bool)
        )
        
        assert torch.all(selected_indices >= 0)


class TestTieBreaking:
    """Test tie-breaking logic"""
    
    def test_equal_ucb_scores(self, selector):
        """Test selection when children have equal UCB scores"""
        torch.manual_seed(42)
        
        # All children visited equally with same values
        selected_indices, _ = selector.select_batch(
            parent_visits=torch.tensor([100], dtype=torch.int32),
            children_visits=torch.tensor([[10, 10, 10]], dtype=torch.int32),
            children_values=torch.tensor([[5.0, 5.0, 5.0]], dtype=torch.float32),
            children_priors=torch.tensor([[0.33, 0.33, 0.34]], dtype=torch.float32),
            valid_mask=torch.tensor([[True, True, True]], dtype=torch.bool)
        )
        
        # Should select one of them (random tie-breaking)
        assert 0 <= selected_indices[0] < 3
        
    def test_stochastic_selection_unvisited(self, selector):
        """Test stochastic selection for unvisited nodes"""
        # Run multiple times to check stochasticity
        selections = []
        
        for seed in range(10):
            torch.manual_seed(seed)
            
            selected_indices, _ = selector.select_batch(
                parent_visits=torch.tensor([100], dtype=torch.int32),
                children_visits=torch.zeros((1, 3), dtype=torch.int32),
                children_values=torch.zeros((1, 3), dtype=torch.float32),
                children_priors=torch.tensor([[0.33, 0.33, 0.34]], dtype=torch.float32),
                valid_mask=torch.ones((1, 3), dtype=torch.bool)
            )
            
            selections.append(selected_indices[0].item())
            
        # Should see some variety in selections
        unique_selections = set(selections)
        assert len(unique_selections) > 1


class TestEdgeCases:
    """Test edge cases and special scenarios"""
    
    def test_single_valid_child(self, selector):
        """Test selection with only one valid child"""
        selected_indices, _ = selector.select_batch(
            parent_visits=torch.tensor([100], dtype=torch.int32),
            children_visits=torch.tensor([[10, 0, 0]], dtype=torch.int32),
            children_values=torch.tensor([[5.0, 0.0, 0.0]], dtype=torch.float32),
            children_priors=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            valid_mask=torch.tensor([[True, False, False]], dtype=torch.bool)
        )
        
        assert selected_indices[0] == 0
        
    def test_large_batch(self, selector):
        """Test selection with large batch"""
        batch_size = 1000
        max_children = 50
        
        # Random data
        torch.manual_seed(42)
        parent_visits = torch.randint(1, 1000, (batch_size,), dtype=torch.int32)
        children_visits = torch.randint(0, 100, (batch_size, max_children), dtype=torch.int32)
        children_values = torch.randn(batch_size, max_children) * 10
        children_priors = torch.rand(batch_size, max_children)
        children_priors = children_priors / children_priors.sum(dim=1, keepdim=True)
        valid_mask = torch.rand(batch_size, max_children) > 0.2
        
        selected_indices, selected_scores = selector.select_batch(
            parent_visits=parent_visits,
            children_visits=children_visits,
            children_values=children_values,
            children_priors=children_priors,
            valid_mask=valid_mask
        )
        
        assert len(selected_indices) == batch_size
        assert len(selected_scores) == batch_size
        
        # Check valid selections
        for i in range(batch_size):
            if valid_mask[i].any():
                assert 0 <= selected_indices[i] < max_children
                assert valid_mask[i, selected_indices[i]]
            else:
                assert selected_indices[i] == -1
                
    def test_extreme_c_puct_values(self, selector, sample_node_data):
        """Test with extreme c_puct values"""
        # Very low c_puct (pure exploitation)
        best_idx = selector.select_single(
            parent_visits=sample_node_data['parent_visits'],
            child_visits=sample_node_data['child_visits'],
            child_values=sample_node_data['child_values'],
            child_priors=sample_node_data['child_priors'],
            c_puct=0.0
        )
        
        # With c_puct=0, pure exploitation should select highest Q-value
        # Calculate expected Q-values
        q_values = torch.where(
            sample_node_data['child_visits'] > 0,
            sample_node_data['child_values'] / sample_node_data['child_visits'],
            torch.zeros_like(sample_node_data['child_values'])
        )
        
        # Child Q-values: [0.5, 0.6, 0.0, 0.4]
        # With c_puct=0, should select child 1 (highest Q=0.6)
        expected_idx = q_values.argmax().item()
        assert best_idx == expected_idx, f"With c_puct=0, expected child {expected_idx} (Q={q_values[expected_idx]:.2f}) but got {best_idx} (Q={q_values[best_idx]:.2f})"
        
        # Test with high c_puct to verify exploration
        best_idx_high = selector.select_single(
            parent_visits=sample_node_data['parent_visits'],
            child_visits=sample_node_data['child_visits'],
            child_values=sample_node_data['child_values'],
            child_priors=sample_node_data['child_priors'],
            c_puct=10.0
        )
        
        # With high c_puct, unvisited node (child 2) might be selected due to exploration bonus
        # Child 2 has 0 visits but 0.3 prior, giving it high exploration value
        assert isinstance(best_idx_high, int) and 0 <= best_idx_high < len(sample_node_data['child_visits'])
        
    def test_numerical_stability(self, selector):
        """Test numerical stability with extreme values"""
        # Very large values
        selected_indices, _ = selector.select_batch(
            parent_visits=torch.tensor([1000000], dtype=torch.int32),
            children_visits=torch.tensor([[999999, 1]], dtype=torch.int32),
            children_values=torch.tensor([[500000.0, 0.5]], dtype=torch.float32),
            children_priors=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
            valid_mask=torch.ones((1, 2), dtype=torch.bool)
        )
        
        # Should handle without overflow/underflow
        assert 0 <= selected_indices[0] < 2
        
    def test_zero_temperature(self, selector):
        """Test with zero temperature (should not crash)"""
        # Temperature = 0 would cause division by zero, but should be handled
        selected_indices, _ = selector.select_batch(
            parent_visits=torch.tensor([100], dtype=torch.int32),
            children_visits=torch.tensor([[10, 20]], dtype=torch.int32),
            children_values=torch.tensor([[5.0, 10.0]], dtype=torch.float32),
            children_priors=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
            valid_mask=torch.ones((1, 2), dtype=torch.bool),
            temperature=0.0  # Edge case
        )
        
        # Should still produce valid result
        assert 0 <= selected_indices[0] < 2