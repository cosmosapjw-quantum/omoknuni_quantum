"""
Tests for thermodynamic analysis of MCTS.

Following TDD principles - these tests are written before implementation.
"""
import pytest
import torch
import numpy as np
from typing import List, Dict, Any

# Import to-be-implemented modules
import sys
import os

# Set library path for C++ extensions
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.environ['LD_LIBRARY_PATH'] = f"{os.path.join(project_root, 'python')}:{os.environ.get('LD_LIBRARY_PATH', '')}"

try:
    from python.mcts.quantum.phenomena import TreeSnapshot
    from python.mcts.quantum.phenomena.thermodynamics import (
        ThermodynamicsAnalyzer, ThermodynamicResult
    )
except ImportError:
    # Expected to fail before implementation
    ThermodynamicsAnalyzer = None
    ThermodynamicResult = None
    TreeSnapshot = None


def create_test_snapshots_with_temperature() -> List[TreeSnapshot]:
    """Create snapshots with varying temperature"""
    snapshots = []
    
    # Simulate cooling: temperature decreases over time
    for k in [10, 100, 1000, 10000]:
        n_nodes = 20
        
        # Temperature should decrease as ~1/sqrt(k)
        temperature = 2.0 / np.sqrt(k)
        
        # Create visit distribution at this temperature
        scores = torch.randn(n_nodes)
        visits = torch.exp(scores / temperature)
        visits = visits / visits.sum() * k  # Total visits = k
        
        # Q-values (average rewards)
        q_values = torch.randn(n_nodes) * 0.1 + 0.5
        value_sum = q_values * visits
        
        observables = {
            'visit_distribution': visits,
            'value_landscape': q_values,
            'value_sum': value_sum,
            'temperature': temperature
        }
        
        snapshot = TreeSnapshot(
            timestamp=k,
            tree_size=n_nodes,
            observables=observables
        )
        snapshots.append(snapshot)
    
    return snapshots


class TestThermodynamicsAnalyzer:
    """Test suite for thermodynamic analysis"""
    
    def test_analyzer_exists(self):
        """ThermodynamicsAnalyzer class should exist"""
        assert ThermodynamicsAnalyzer is not None, "ThermodynamicsAnalyzer not implemented"
    
    def test_basic_thermodynamics(self):
        """Should compute basic thermodynamic quantities"""
        if ThermodynamicsAnalyzer is None:
            pytest.skip("ThermodynamicsAnalyzer not yet implemented")
            
        analyzer = ThermodynamicsAnalyzer()
        snapshots = create_test_snapshots_with_temperature()
        
        results = analyzer.measure_thermodynamics(snapshots)
        
        assert len(results) == len(snapshots)
        
        # Check each result has required fields
        for result in results:
            assert 'energy' in result
            assert 'entropy' in result
            assert 'temperature' in result
            assert 'free_energy' in result
            assert 'heat_capacity' in result
    
    def test_energy_computation(self):
        """Should compute energy correctly"""
        if ThermodynamicsAnalyzer is None:
            pytest.skip("ThermodynamicsAnalyzer not yet implemented")
            
        analyzer = ThermodynamicsAnalyzer()
        
        # Simple test case
        q_values = torch.tensor([0.5, 0.3, 0.2])
        visits = torch.tensor([50.0, 30.0, 20.0])
        
        energy = analyzer.compute_energy(q_values, visits)
        
        # Energy = -<Q> = -sum(q * p) where p = visits/total
        expected = -torch.sum(q_values * visits) / visits.sum()
        assert torch.allclose(energy, expected)
    
    def test_entropy_computation(self):
        """Should compute entropy correctly"""
        if ThermodynamicsAnalyzer is None:
            pytest.skip("ThermodynamicsAnalyzer not yet implemented")
            
        analyzer = ThermodynamicsAnalyzer()
        
        # Test with known distribution
        visits = torch.tensor([60.0, 30.0, 10.0])
        entropy = analyzer.compute_entropy(visits)
        
        # Shannon entropy
        probs = visits / visits.sum()
        expected = -torch.sum(probs * torch.log(probs))
        
        assert torch.allclose(entropy, expected)
    
    def test_free_energy(self):
        """Should compute free energy F = E - TS"""
        if ThermodynamicsAnalyzer is None:
            pytest.skip("ThermodynamicsAnalyzer not yet implemented")
            
        analyzer = ThermodynamicsAnalyzer()
        
        energy = torch.tensor(-0.5)
        temperature = torch.tensor(2.0)
        entropy = torch.tensor(1.5)
        
        free_energy = analyzer.compute_free_energy(energy, temperature, entropy)
        expected = energy - temperature * entropy
        
        assert torch.allclose(free_energy, expected)
    
    def test_heat_capacity(self):
        """Should compute heat capacity dE/dT"""
        if ThermodynamicsAnalyzer is None:
            pytest.skip("ThermodynamicsAnalyzer not yet implemented")
            
        analyzer = ThermodynamicsAnalyzer()
        snapshots = create_test_snapshots_with_temperature()
        
        results = analyzer.measure_thermodynamics(snapshots)
        
        # Heat capacity should be computed between adjacent snapshots
        for i in range(1, len(results)):
            assert results[i]['heat_capacity'] != 0.0
    
    def test_partition_function(self):
        """Should compute partition function Z"""
        if ThermodynamicsAnalyzer is None:
            pytest.skip("ThermodynamicsAnalyzer not yet implemented")
            
        analyzer = ThermodynamicsAnalyzer()
        
        scores = torch.tensor([1.0, 0.5, 0.0])
        temperature = 2.0
        
        Z = analyzer.compute_partition_function(scores, temperature)
        
        # Z = sum(exp(beta * score))
        beta = 1.0 / temperature
        expected = torch.sum(torch.exp(beta * scores))
        
        assert torch.allclose(Z, expected)
    
    def test_susceptibility(self):
        """Should compute susceptibility chi = d<Q>/dc_puct"""
        if ThermodynamicsAnalyzer is None:
            pytest.skip("ThermodynamicsAnalyzer not yet implemented")
            
        analyzer = ThermodynamicsAnalyzer()
        
        # Create snapshots at different c_puct values
        snapshots_low_c = create_test_snapshots_with_temperature()
        snapshots_high_c = create_test_snapshots_with_temperature()
        
        # Mark with c_puct values
        for s in snapshots_low_c:
            s.observables['c_puct'] = 1.0
        for s in snapshots_high_c:
            s.observables['c_puct'] = 2.0
            
        susceptibility = analyzer.compute_susceptibility(
            snapshots_low_c + snapshots_high_c
        )
        
        assert 'chi' in susceptibility
        assert susceptibility['chi'] != 0.0
    
    def test_gpu_acceleration(self):
        """Should support GPU computation"""
        if ThermodynamicsAnalyzer is None:
            pytest.skip("ThermodynamicsAnalyzer not yet implemented")
            
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        analyzer = ThermodynamicsAnalyzer(device='cuda')
        
        q_values = torch.randn(100, device='cuda')
        visits = torch.rand(100, device='cuda') * 100
        
        energy = analyzer.compute_energy(q_values, visits)
        assert energy.device.type == 'cuda'


class TestThermodynamicResult:
    """Test ThermodynamicResult data structure"""
    
    def test_result_structure(self):
        """ThermodynamicResult should store thermodynamic data"""
        if ThermodynamicResult is None:
            pytest.skip("ThermodynamicResult not yet implemented")
            
        result = ThermodynamicResult(
            energy=-0.5,
            entropy=1.5,
            temperature=2.0,
            free_energy=-3.5,
            heat_capacity=0.8,
            partition_function=10.0
        )
        
        assert result.energy == -0.5
        assert result.free_energy == -3.5
    
    def test_result_export(self):
        """Should export to dictionary"""
        if ThermodynamicResult is None:
            pytest.skip("ThermodynamicResult not yet implemented")
            
        result = ThermodynamicResult(
            energy=-0.5,
            entropy=1.5,
            temperature=2.0,
            free_energy=-3.5,
            heat_capacity=0.8,
            partition_function=10.0
        )
        
        data = result.to_dict()
        assert isinstance(data, dict)
        assert data['energy'] == -0.5


class TestJarzynskiEquality:
    """Test Jarzynski equality validation"""
    
    def test_work_distribution(self):
        """Should compute work distribution"""
        if ThermodynamicsAnalyzer is None:
            pytest.skip("ThermodynamicsAnalyzer not yet implemented")
            
        analyzer = ThermodynamicsAnalyzer()
        
        # Create ensemble of trajectories
        trajectories = []
        for _ in range(100):
            # Each trajectory is a sequence of snapshots
            traj = create_test_snapshots_with_temperature()
            trajectories.append(traj)
        
        work_distribution = analyzer.compute_work_distribution(trajectories)
        
        assert len(work_distribution) == len(trajectories)
        assert all(isinstance(w, float) for w in work_distribution)
    
    def test_jarzynski_validation(self):
        """Should validate Jarzynski equality"""
        if ThermodynamicsAnalyzer is None:
            pytest.skip("ThermodynamicsAnalyzer not yet implemented")
            
        analyzer = ThermodynamicsAnalyzer()
        
        # Create trajectories
        trajectories = []
        for _ in range(1000):
            traj = create_test_snapshots_with_temperature()
            trajectories.append(traj)
        
        result = analyzer.validate_jarzynski_equality(trajectories)
        
        assert 'work_distribution' in result
        assert 'jarzynski_average' in result
        assert 'free_energy_difference' in result
        assert 'equality_satisfied' in result


class TestFluctuationTheorems:
    """Test fluctuation theorem validation"""
    
    def test_entropy_production(self):
        """Should compute entropy production"""
        if ThermodynamicsAnalyzer is None:
            pytest.skip("ThermodynamicsAnalyzer not yet implemented")
            
        analyzer = ThermodynamicsAnalyzer()
        trajectory = create_test_snapshots_with_temperature()
        
        entropy_production = analyzer.compute_entropy_production(trajectory)
        
        assert isinstance(entropy_production, float)
        # Second law: entropy production >= 0
        assert entropy_production >= 0
    
    def test_crooks_theorem(self):
        """Should validate Crooks fluctuation theorem"""
        if ThermodynamicsAnalyzer is None:
            pytest.skip("ThermodynamicsAnalyzer not yet implemented")
            
        analyzer = ThermodynamicsAnalyzer()
        
        # Need forward and reverse trajectories
        forward_trajectories = [create_test_snapshots_with_temperature() 
                              for _ in range(100)]
        
        # For testing, use same trajectories reversed
        reverse_trajectories = [list(reversed(traj)) 
                              for traj in forward_trajectories]
        
        result = analyzer.validate_crooks_theorem(
            forward_trajectories, reverse_trajectories
        )
        
        assert 'work_ratio' in result
        assert 'theorem_satisfied' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])