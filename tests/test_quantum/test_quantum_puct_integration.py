"""
Integration tests for Quantum-Augmented PUCT formula.

Tests the proper integration of quantum corrections with standard PUCT.
"""
import pytest
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

# Import quantum components
from python.mcts.quantum.temperature import TemperatureMeasurer
from python.mcts.quantum.corrections import QuantumCorrectionCalculator


@dataclass
class MockMCTSNode:
    """Mock MCTS node for integration testing"""
    state: object
    parent: Optional['MockMCTSNode'] = None
    children: Dict[int, 'MockMCTSNode'] = None
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}
    
    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    @property
    def is_leaf(self):
        return len(self.children) == 0


class StandardPUCT:
    """Standard PUCT implementation for comparison"""
    
    def __init__(self, c_puct: float = 1.0):
        self.c_puct = c_puct
    
    def compute_puct_score(self, parent: MockMCTSNode, child: MockMCTSNode) -> float:
        """Standard PUCT formula"""
        q_value = child.q_value
        
        # Exploration term: c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        exploration = self.c_puct * child.prior * np.sqrt(parent.visit_count) / (1 + child.visit_count)
        
        return q_value + exploration


class QuantumAugmentedPUCT:
    """Quantum-augmented PUCT implementation"""
    
    def __init__(self, c_puct: float = 1.0, gamma: float = 1.0, 
                 use_temperature_scaling: bool = True):
        self.c_puct = c_puct
        self.gamma = gamma
        self.use_temperature_scaling = use_temperature_scaling
        
        self.temp_measurer = TemperatureMeasurer()
        self.corrector = QuantumCorrectionCalculator(c_puct=c_puct, gamma=gamma)
    
    def compute_puct_score(self, parent: MockMCTSNode, child: MockMCTSNode,
                          temperature: Optional[float] = None) -> float:
        """Quantum-augmented PUCT formula"""
        # Standard components
        q_value = child.q_value
        exploration = self.c_puct * child.prior * np.sqrt(parent.visit_count) / (1 + child.visit_count)
        
        # Measure temperature if not provided
        if temperature is None and self.use_temperature_scaling:
            temperature = 1.0 / self.temp_measurer.measure(parent)
        else:
            temperature = temperature or 1.0
        
        # Quantum correction
        quantum_bonus = self.corrector.compute_bonus(
            q_value=q_value,
            visits=child.visit_count,
            beta=1.0 / temperature
        )
        
        return q_value + exploration + quantum_bonus


class TestQuantumPUCTIntegration:
    """Test suite for Quantum-PUCT integration"""
    
    def test_quantum_puct_vs_standard(self):
        """Quantum PUCT should enhance but not override standard PUCT"""
        # Create test tree
        parent = MockMCTSNode(state="root", visit_count=1000)
        
        # Create children with various statistics
        children = []
        for i in range(4):
            child = MockMCTSNode(
                state=f"child_{i}",
                parent=parent,
                visit_count=100 * (i + 1),
                value_sum=50 * (i + 1) * (0.8 - i * 0.1),  # Decreasing Q-values
                prior=0.25  # Uniform prior
            )
            parent.children[i] = child
            children.append(child)
        
        # Compute scores
        standard_puct = StandardPUCT(c_puct=1.0)
        quantum_puct = QuantumAugmentedPUCT(c_puct=1.0, gamma=1.0)
        
        standard_scores = []
        quantum_scores = []
        
        for child in children:
            standard_score = standard_puct.compute_puct_score(parent, child)
            quantum_score = quantum_puct.compute_puct_score(parent, child, temperature=1.0)
            
            standard_scores.append(standard_score)
            quantum_scores.append(quantum_score)
        
        # Quantum should add small positive corrections
        for std, qnt in zip(standard_scores, quantum_scores):
            assert qnt > std, "Quantum score should be larger due to positive correction"
            assert qnt - std < 0.1, "Quantum correction should be small"
        
        # Relative ordering should be mostly preserved
        standard_ranking = np.argsort(standard_scores)[::-1]
        quantum_ranking = np.argsort(quantum_scores)[::-1]
        
        # At least top 2 should be the same
        assert len(set(standard_ranking[:2]) & set(quantum_ranking[:2])) >= 1
    
    def test_quantum_correction_scaling_with_exploration(self):
        """Quantum corrections should scale appropriately with exploration term"""
        parent = MockMCTSNode(state="root", visit_count=1000)
        
        # Low visit child - high exploration, should also have high quantum correction
        low_visit_child = MockMCTSNode(
            state="low_visit",
            parent=parent,
            visit_count=10,
            value_sum=5.0,
            prior=0.5
        )
        
        # High visit child - low exploration, should have low quantum correction
        high_visit_child = MockMCTSNode(
            state="high_visit",
            parent=parent,
            visit_count=500,
            value_sum=250.0,
            prior=0.5
        )
        
        quantum_puct = QuantumAugmentedPUCT(c_puct=1.0)
        corrector = QuantumCorrectionCalculator(c_puct=1.0)
        
        # Get exploration terms
        exploration_low = 1.0 * 0.5 * np.sqrt(1000) / (1 + 10)
        exploration_high = 1.0 * 0.5 * np.sqrt(1000) / (1 + 500)
        
        # Get quantum corrections
        quantum_low = corrector.compute_bonus(q_value=0.5, visits=10)
        quantum_high = corrector.compute_bonus(q_value=0.5, visits=500)
        
        # Both should decrease with visits
        assert exploration_low > exploration_high
        assert quantum_low > quantum_high
        
        # Quantum correction should be much smaller than exploration
        assert quantum_low < exploration_low * 0.1
        assert quantum_high < exploration_high * 0.1
    
    def test_temperature_adaptive_scoring(self):
        """Temperature should affect quantum corrections appropriately"""
        parent = MockMCTSNode(state="root", visit_count=1000)
        
        # Create diverse children for temperature measurement
        for i in range(5):
            child = MockMCTSNode(
                state=f"child_{i}",
                parent=parent,
                visit_count=200 if i == 0 else 50,  # One dominant child
                value_sum=100 if i == 0 else 25,
                prior=0.2
            )
            child.score = child.q_value + 0.1 * (5 - i)  # Add score attribute
            parent.children[i] = child
        
        quantum_puct = QuantumAugmentedPUCT(c_puct=1.0, use_temperature_scaling=True)
        
        # Compute score with automatic temperature
        best_child = parent.children[0]
        score_auto_temp = quantum_puct.compute_puct_score(parent, best_child)
        
        # Compare with fixed temperatures
        score_high_temp = quantum_puct.compute_puct_score(parent, best_child, temperature=2.0)
        score_low_temp = quantum_puct.compute_puct_score(parent, best_child, temperature=0.5)
        
        # Higher temperature should give larger corrections (more exploration)
        # Note: temperature is converted to beta internally (beta = 1/T)
        assert score_high_temp > score_low_temp
    
    def test_quantum_puct_preserves_relative_ordering(self):
        """Quantum corrections should refine selections while preserving key relationships"""
        parent = MockMCTSNode(state="root", visit_count=10000)
        
        # Clear best action with high visits
        best_child = MockMCTSNode(
            state="best",
            parent=parent,
            visit_count=5000,
            value_sum=4500.0,  # Q = 0.9
            prior=0.4
        )
        parent.children[0] = best_child
        
        # Good but less explored action
        explore_child = MockMCTSNode(
            state="explore",
            parent=parent,
            visit_count=100,
            value_sum=75.0,  # Q = 0.75
            prior=0.3
        )
        parent.children[1] = explore_child
        
        # Bad action
        bad_child = MockMCTSNode(
            state="bad",
            parent=parent,
            visit_count=1000,
            value_sum=200.0,  # Q = 0.2
            prior=0.3
        )
        parent.children[2] = bad_child
        
        # Compute scores
        standard_puct = StandardPUCT(c_puct=1.0)
        quantum_puct = QuantumAugmentedPUCT(c_puct=1.0)
        
        standard_scores = {}
        quantum_scores = {}
        
        for action, child in parent.children.items():
            standard_scores[action] = standard_puct.compute_puct_score(parent, child)
            quantum_scores[action] = quantum_puct.compute_puct_score(parent, child, temperature=1.0)
        
        # Bad action should remain worst in both
        assert min(standard_scores, key=standard_scores.get) == 2
        assert min(quantum_scores, key=quantum_scores.get) == 2
        
        # Quantum corrections should be inversely related to visits
        quantum_bonuses = {
            action: quantum_scores[action] - standard_scores[action]
            for action in parent.children
        }
        
        # Lower visits should get higher bonus
        assert quantum_bonuses[1] > quantum_bonuses[0]  # explore > best
        assert quantum_bonuses[1] > quantum_bonuses[2]  # explore > bad
    
    def test_batch_quantum_scoring(self):
        """Test batch computation of quantum-augmented scores"""
        parent = MockMCTSNode(state="root", visit_count=1000)
        
        # Create many children
        n_children = 32
        for i in range(n_children):
            child = MockMCTSNode(
                state=f"child_{i}",
                parent=parent,
                visit_count=np.random.randint(10, 500),
                value_sum=np.random.randint(5, 250),
                prior=1.0 / n_children
            )
            parent.children[i] = child
        
        quantum_puct = QuantumAugmentedPUCT(c_puct=1.0)
        
        # Prepare batch data
        q_values = torch.tensor([c.q_value for c in parent.children.values()])
        visits = torch.tensor([c.visit_count for c in parent.children.values()])
        priors = torch.tensor([c.prior for c in parent.children.values()])
        
        # Compute exploration terms
        sqrt_parent = np.sqrt(parent.visit_count)
        exploration = priors * sqrt_parent / (1 + visits)
        
        # Compute quantum corrections in batch
        quantum_bonuses = quantum_puct.corrector.compute_bonus_batch(
            q_values=q_values,
            visits=visits,
            beta=1.0
        )
        
        # Total scores
        quantum_scores = q_values + quantum_puct.c_puct * exploration + quantum_bonuses.cpu()
        
        # Verify all scores are valid
        assert torch.all(torch.isfinite(quantum_scores))
        assert torch.all(quantum_bonuses >= 0)
        
        # Verify batch matches individual computation
        for i, child in enumerate(parent.children.values()):
            individual_score = quantum_puct.compute_puct_score(parent, child, temperature=1.0)
            batch_score = quantum_scores[i].item()
            assert abs(individual_score - batch_score) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])