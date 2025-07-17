"""
Tests for RG-enhanced backpropagation.

Following TDD principles - these tests are written before implementation.
"""
import pytest
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

# Import to-be-implemented modules (will fail initially)
try:
    from python.mcts.quantum.rg_backprop import RGBackpropagator, ScaleKernel
except ImportError:
    # Expected to fail before implementation
    RGBackpropagator = None
    ScaleKernel = None


@dataclass
class MockTreeNode:
    """Mock tree node for testing backpropagation"""
    value_sum: float = 0.0
    visit_count: int = 0
    parent: Optional['MockTreeNode'] = None
    children: Dict[str, 'MockTreeNode'] = None
    depth: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}
    
    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def add_child(self, action: str, child: 'MockTreeNode'):
        self.children[action] = child
        child.parent = self
        child.depth = self.depth + 1


class TestRGBackpropagator:
    """Test suite for RG-enhanced backpropagation"""
    
    def test_backpropagator_exists(self):
        """RGBackpropagator class should exist"""
        assert RGBackpropagator is not None, "RGBackpropagator not implemented"
    
    def test_basic_backpropagation(self):
        """Should perform basic value backpropagation"""
        if RGBackpropagator is None:
            pytest.skip("RGBackpropagator not yet implemented")
            
        # Create simple tree
        root = MockTreeNode()
        child1 = MockTreeNode()
        child2 = MockTreeNode()
        leaf = MockTreeNode()
        
        root.add_child("a", child1)
        root.add_child("b", child2)
        child1.add_child("c", leaf)
        
        backprop = RGBackpropagator()
        
        # Backpropagate value from leaf
        backprop.backpropagate(leaf, value=1.0)
        
        # Check values propagated
        assert leaf.visit_count == 1
        assert leaf.value_sum == 1.0
        assert child1.visit_count == 1
        assert child1.value_sum == 1.0
        assert root.visit_count == 1
        assert root.value_sum == 1.0
    
    def test_rg_smoothing(self):
        """RG backprop should smooth values across scales"""
        if RGBackpropagator is None:
            pytest.skip("RGBackpropagator not yet implemented")
            
        # Create tree with parent and children
        root = MockTreeNode()
        parent = MockTreeNode()
        root.add_child("parent", parent)
        
        # Create children with different values
        children = []
        for i in range(5):
            child = MockTreeNode(value_sum=i*10, visit_count=10)
            parent.add_child(f"child_{i}", child)
            children.append(child)
        
        # Add some noise to values
        children[0].value_sum = 50  # Outlier
        
        backprop = RGBackpropagator(smoothing_factor=0.3)
        
        # Store original outlier value
        outlier_before = children[0].q_value
        mean_others_before = np.mean([c.q_value for c in children[1:]])
        
        # Create a leaf under the outlier to trigger backprop
        leaf = MockTreeNode()
        children[0].add_child("leaf", leaf)
        
        # Backpropagate through the outlier
        backprop.backpropagate(leaf, value=0.0)
        
        # The outlier should move towards the mean of siblings
        outlier_after = children[0].q_value
        
        # Check that smoothing occurred
        # The outlier should be closer to the mean of others
        distance_before = abs(outlier_before - mean_others_before)
        distance_after = abs(outlier_after - mean_others_before)
        
        assert distance_after < distance_before, \
            f"Outlier should move towards mean: before={distance_before:.3f}, after={distance_after:.3f}"
    
    def test_scale_dependent_smoothing(self):
        """Smoothing should be scale-dependent"""
        if RGBackpropagator is None:
            pytest.skip("RGBackpropagator not yet implemented")
            
        # Create deeper tree
        root = MockTreeNode()
        level1 = [MockTreeNode() for _ in range(3)]
        level2 = [MockTreeNode() for _ in range(9)]
        
        # Build tree structure
        for i, node in enumerate(level1):
            root.add_child(f"l1_{i}", node)
            for j in range(3):
                child = level2[i*3 + j]
                node.add_child(f"l2_{j}", child)
        
        backprop = RGBackpropagator(smoothing_factor=0.1)
        
        # Test different scales
        leaf = level2[0]
        backprop.backpropagate(leaf, value=1.0)
        
        # Smoothing should be stronger at higher levels (closer to root)
        # This is tested indirectly through the smoothing kernel
        kernel = backprop.get_smoothing_kernel(scale=1)
        assert len(kernel) > 0, "Should have smoothing kernel"
        # For Gaussian kernel, center should be higher than edges
        center_idx = len(kernel) // 2
        assert kernel[center_idx] > kernel[0], "Kernel should peak at center"
    
    def test_scale_kernel(self):
        """Test scale-dependent smoothing kernel"""
        if ScaleKernel is None:
            pytest.skip("ScaleKernel not yet implemented")
            
        kernel = ScaleKernel(base_smoothing=0.1)
        
        # Test different scales
        k1 = kernel.get_kernel(scale=1)  # Near leaf
        k2 = kernel.get_kernel(scale=5)  # Near root
        
        # Kernel should be normalized
        assert abs(torch.sum(k1) - 1.0) < 1e-6
        assert abs(torch.sum(k2) - 1.0) < 1e-6
        
        # Higher scale should have wider kernel
        assert len(k2) >= len(k1)
    
    def test_gpu_acceleration(self):
        """GPU implementation should match CPU"""
        if RGBackpropagator is None:
            pytest.skip("RGBackpropagator not yet implemented")
            
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        # Create tree
        root = MockTreeNode()
        for i in range(10):
            root.add_child(f"child_{i}", MockTreeNode())
        
        # CPU backpropagator
        cpu_backprop = RGBackpropagator(device='cpu')
        
        # GPU backpropagator
        gpu_backprop = RGBackpropagator(device='cuda')
        
        # Both should produce same results
        leaf = list(root.children.values())[0]
        
        # Clone tree for separate tests
        import copy
        root_cpu = copy.deepcopy(root)
        root_gpu = copy.deepcopy(root)
        
        cpu_backprop.backpropagate(list(root_cpu.children.values())[0], 1.0)
        gpu_backprop.backpropagate(list(root_gpu.children.values())[0], 1.0)
        
        # Compare results
        for key in root_cpu.children:
            cpu_val = root_cpu.children[key].q_value
            gpu_val = root_gpu.children[key].q_value
            assert abs(cpu_val - gpu_val) < 1e-5
    
    def test_batch_backpropagation(self):
        """Should support batch backpropagation"""
        if RGBackpropagator is None:
            pytest.skip("RGBackpropagator not yet implemented")
            
        # Create tree with multiple leaves
        root = MockTreeNode()
        leaves = []
        
        for i in range(4):
            child = MockTreeNode()
            root.add_child(f"child_{i}", child)
            for j in range(3):
                leaf = MockTreeNode()
                child.add_child(f"leaf_{j}", leaf)
                leaves.append(leaf)
        
        backprop = RGBackpropagator()
        
        # Batch backpropagation
        values = torch.rand(len(leaves))
        backprop.backpropagate_batch(leaves, values)
        
        # All leaves should be updated
        for i, leaf in enumerate(leaves):
            assert leaf.visit_count == 1
            assert abs(leaf.value_sum - values[i].item()) < 1e-6
    
    def test_rg_flow_visualization(self):
        """Should track RG flow for analysis"""
        if RGBackpropagator is None:
            pytest.skip("RGBackpropagator not yet implemented")
            
        root = MockTreeNode()
        child = MockTreeNode()
        leaf = MockTreeNode()
        
        root.add_child("a", child)
        child.add_child("b", leaf)
        
        backprop = RGBackpropagator(track_flow=True)
        backprop.backpropagate(leaf, value=1.0)
        
        # Should have flow data
        flow_data = backprop.get_flow_data()
        assert len(flow_data) > 0, "Should track RG flow"
        
        # Flow should include scale information
        assert 'scale' in flow_data[0]
        assert 'value_before' in flow_data[0]
        assert 'value_after' in flow_data[0]
    
    def test_adaptive_smoothing(self):
        """Smoothing should adapt based on value variance"""
        if RGBackpropagator is None:
            pytest.skip("RGBackpropagator not yet implemented")
            
        # High variance case
        root_high_var = MockTreeNode()
        for i in range(5):
            child = MockTreeNode(
                value_sum=np.random.rand() * 100,
                visit_count=10
            )
            root_high_var.add_child(f"child_{i}", child)
        
        # Low variance case
        root_low_var = MockTreeNode()
        for i in range(5):
            child = MockTreeNode(
                value_sum=50 + np.random.rand(),
                visit_count=10
            )
            root_low_var.add_child(f"child_{i}", child)
        
        backprop = RGBackpropagator(adaptive_smoothing=True)
        
        # Get adaptive smoothing factors
        smooth_high = backprop.compute_adaptive_smoothing(root_high_var)
        smooth_low = backprop.compute_adaptive_smoothing(root_low_var)
        
        # Higher variance should lead to more smoothing
        assert smooth_high > smooth_low


class TestRGIntegration:
    """Test RG backpropagation integration with MCTS"""
    
    def test_rg_preserves_convergence(self):
        """RG smoothing should not prevent convergence"""
        if RGBackpropagator is None:
            pytest.skip("RGBackpropagator not yet implemented")
            
        # Simulate many backpropagations
        root = MockTreeNode()
        child_good = MockTreeNode()
        child_bad = MockTreeNode()
        
        root.add_child("good", child_good)
        root.add_child("bad", child_bad)
        
        backprop = RGBackpropagator(smoothing_factor=0.1)
        
        # Backpropagate many times
        for _ in range(1000):
            # Good action gets positive values
            backprop.backpropagate(child_good, value=1.0)
            # Bad action gets negative values
            backprop.backpropagate(child_bad, value=-1.0)
        
        # Despite smoothing, should converge to correct values
        assert child_good.q_value > 0.5
        assert child_bad.q_value < -0.5
        assert child_good.q_value > child_bad.q_value
    
    def test_rg_with_exploration(self):
        """RG should work with exploration bonuses"""
        if RGBackpropagator is None:
            pytest.skip("RGBackpropagator not yet implemented")
            
        root = MockTreeNode(visit_count=1000)
        children = []
        
        for i in range(3):
            child = MockTreeNode(
                visit_count=100 - i*30,  # Different visit counts
                value_sum=(100 - i*30) * 0.5
            )
            root.add_child(f"child_{i}", child)
            children.append(child)
        
        backprop = RGBackpropagator()
        
        # Test that RG doesn't interfere with exploration
        # by checking relative ordering is preserved
        q_before = [c.q_value for c in children]
        
        backprop.backpropagate(children[1], value=0.6)
        
        q_after = [c.q_value for c in children]
        
        # Check that smoothing doesn't completely destroy value differences
        # The updated child should have increased value
        assert q_after[1] > q_before[1], "Backpropagated value should increase"
        
        # But shouldn't completely override the structure
        value_range_before = max(q_before) - min(q_before)
        value_range_after = max(q_after) - min(q_after)
        assert value_range_after > 0, "Should maintain some value differences"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])