"""
Tests for critical phenomena and phase transitions in MCTS.

Following TDD principles - these tests are written before implementation.
"""
import pytest
import torch
import numpy as np
from typing import List, Dict, Any, Tuple

# Import to-be-implemented modules
import sys
import os

# Set library path for C++ extensions
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.environ['LD_LIBRARY_PATH'] = f"{os.path.join(project_root, 'python')}:{os.environ.get('LD_LIBRARY_PATH', '')}"

try:
    from python.mcts.quantum.phenomena.critical import (
        CriticalPhenomenaAnalyzer, CriticalPoint, ScalingResult
    )
except ImportError:
    # Expected to fail before implementation
    CriticalPhenomenaAnalyzer = None
    CriticalPoint = None
    ScalingResult = None


def create_critical_position_data(delta_q: float = 0.01) -> Dict[str, Any]:
    """Create data for a critical position (two moves with similar values)"""
    # Two top moves with very close Q-values
    q_values = torch.tensor([0.5, 0.5 - delta_q, 0.2, 0.1, 0.0])
    
    # Visit distribution will show competition
    visits = torch.tensor([450.0, 440.0, 60.0, 30.0, 20.0])
    
    return {
        'q_values': q_values,
        'visits': visits,
        'delta_q': delta_q,
        'total_visits': visits.sum()
    }


def create_non_critical_data() -> Dict[str, Any]:
    """Create data for non-critical position (clear best move)"""
    q_values = torch.tensor([0.8, 0.3, 0.2, 0.1, 0.0])
    visits = torch.tensor([800.0, 100.0, 60.0, 30.0, 10.0])
    
    return {
        'q_values': q_values,
        'visits': visits,
        'delta_q': 0.5,
        'total_visits': visits.sum()
    }


class TestCriticalPhenomenaAnalyzer:
    """Test suite for critical phenomena analysis"""
    
    def test_analyzer_exists(self):
        """CriticalPhenomenaAnalyzer class should exist"""
        assert CriticalPhenomenaAnalyzer is not None, "CriticalPhenomenaAnalyzer not implemented"
    
    def test_critical_point_detection(self):
        """Should detect critical points in game positions"""
        if CriticalPhenomenaAnalyzer is None:
            pytest.skip("CriticalPhenomenaAnalyzer not yet implemented")
            
        analyzer = CriticalPhenomenaAnalyzer()
        
        # Test critical position
        critical_data = create_critical_position_data()
        is_critical = analyzer.is_critical_position(
            critical_data['q_values'],
            critical_data['visits']
        )
        assert is_critical
        
        # Test non-critical position
        non_critical_data = create_non_critical_data()
        is_critical = analyzer.is_critical_position(
            non_critical_data['q_values'],
            non_critical_data['visits']
        )
        assert not is_critical
    
    def test_order_parameter(self):
        """Should compute order parameter m = π₁ - π₂"""
        if CriticalPhenomenaAnalyzer is None:
            pytest.skip("CriticalPhenomenaAnalyzer not yet implemented")
            
        analyzer = CriticalPhenomenaAnalyzer()
        
        visits = torch.tensor([600.0, 400.0, 0.0, 0.0])
        order_param = analyzer.compute_order_parameter(visits)
        
        # m = (600 - 400) / 1000 = 0.2
        expected = 0.2
        assert abs(order_param - expected) < 1e-6
    
    def test_susceptibility(self):
        """Should compute susceptibility χ = dm/dh"""
        if CriticalPhenomenaAnalyzer is None:
            pytest.skip("CriticalPhenomenaAnalyzer not yet implemented")
            
        analyzer = CriticalPhenomenaAnalyzer()
        
        # Create perturbed data
        visits_unbiased = torch.tensor([500.0, 500.0])
        visits_biased = torch.tensor([550.0, 450.0])
        bias = 0.1
        
        chi = analyzer.compute_susceptibility(
            visits_unbiased, visits_biased, bias
        )
        
        assert chi > 0
    
    def test_correlation_length(self):
        """Should compute correlation length ξ"""
        if CriticalPhenomenaAnalyzer is None:
            pytest.skip("CriticalPhenomenaAnalyzer not yet implemented")
            
        analyzer = CriticalPhenomenaAnalyzer()
        
        # Tree structure with depths
        visits = torch.tensor([100.0, 50.0, 50.0, 25.0, 25.0, 25.0, 25.0])
        depths = torch.tensor([0, 1, 1, 2, 2, 2, 2])
        
        xi = analyzer.compute_correlation_length(visits, depths)
        
        # Weighted average depth
        expected = torch.sum(visits * depths) / visits.sum()
        assert abs(xi - expected.item()) < 1e-6
    
    def test_finite_size_scaling(self):
        """Should analyze finite-size scaling"""
        if CriticalPhenomenaAnalyzer is None:
            pytest.skip("CriticalPhenomenaAnalyzer not yet implemented")
            
        analyzer = CriticalPhenomenaAnalyzer()
        
        # Data at different system sizes
        sizes = [100, 200, 400, 800, 1600]
        order_params = []
        susceptibilities = []
        
        for L in sizes:
            # Near critical point
            visits = torch.tensor([L/2 + 10, L/2 - 10])
            m = analyzer.compute_order_parameter(visits)
            chi = L * m**2  # Simplified susceptibility
            
            order_params.append(m)
            susceptibilities.append(chi)
        
        # Analyze scaling
        scaling_result = analyzer.analyze_finite_size_scaling(
            sizes, order_params, susceptibilities
        )
        
        assert 'beta_over_nu' in scaling_result
        assert 'gamma_over_nu' in scaling_result
        assert 'nu' in scaling_result
    
    def test_critical_exponents(self):
        """Should extract critical exponents"""
        if CriticalPhenomenaAnalyzer is None:
            pytest.skip("CriticalPhenomenaAnalyzer not yet implemented")
            
        analyzer = CriticalPhenomenaAnalyzer()
        
        # Generate scaling data
        sizes = np.array([64, 128, 256, 512, 1024])
        
        # Theoretical scaling: m ~ L^(-β/ν)
        beta_over_nu = 0.125
        m_values = 0.5 * sizes**(-beta_over_nu)
        
        # Theoretical scaling: χ ~ L^(γ/ν)  
        gamma_over_nu = 1.75
        chi_values = 2.0 * sizes**(gamma_over_nu)
        
        exponents = analyzer.extract_critical_exponents(
            sizes.tolist(),
            m_values.tolist(),
            chi_values.tolist()
        )
        
        assert abs(exponents['beta_over_nu'] - beta_over_nu) < 0.1
        assert abs(exponents['gamma_over_nu'] - gamma_over_nu) < 0.1
    
    def test_universality_class(self):
        """Should determine universality class"""
        if CriticalPhenomenaAnalyzer is None:
            pytest.skip("CriticalPhenomenaAnalyzer not yet implemented")
            
        analyzer = CriticalPhenomenaAnalyzer()
        
        # Known exponents for 2D Ising class
        exponents = {
            'beta': 0.125,
            'gamma': 1.75,
            'nu': 1.0
        }
        
        universality_class = analyzer.identify_universality_class(exponents)
        
        assert universality_class is not None
        assert 'name' in universality_class
    
    def test_data_collapse(self):
        """Should perform data collapse for scaling functions"""
        if CriticalPhenomenaAnalyzer is None:
            pytest.skip("CriticalPhenomenaAnalyzer not yet implemented")
            
        analyzer = CriticalPhenomenaAnalyzer()
        
        # Generate data that should collapse
        sizes = [100, 200, 400, 800]
        tau_values = np.linspace(-0.1, 0.1, 10)  # Distance from critical point
        
        data = []
        for L in sizes:
            for tau in tau_values:
                # Scaling form: m(L,τ) = L^(-β/ν) f(τL^(1/ν))
                scaling_var = tau * L**(1.0)  # nu = 1
                m = L**(-0.125) * np.tanh(scaling_var)  # beta/nu = 0.125
                
                data.append({
                    'L': L,
                    'tau': tau,
                    'm': m
                })
        
        # Test collapse
        collapse_result = analyzer.test_data_collapse(
            data,
            beta_over_nu=0.125,
            nu=1.0
        )
        
        # Data collapse should show improvement
        assert collapse_result['collapse_quality'] > 0.5
        assert 'scaled_x' in collapse_result
        assert 'scaled_y' in collapse_result


class TestCriticalPoint:
    """Test CriticalPoint data structure"""
    
    def test_critical_point_structure(self):
        """CriticalPoint should store critical position info"""
        if CriticalPoint is None:
            pytest.skip("CriticalPoint not yet implemented")
            
        cp = CriticalPoint(
            position_id=1,
            delta_q=0.01,
            order_parameter=0.05,
            susceptibility=10.0,
            correlation_length=5.0
        )
        
        assert cp.delta_q == 0.01
        assert cp.susceptibility == 10.0
    
    def test_critical_point_export(self):
        """Should export to dictionary"""
        if CriticalPoint is None:
            pytest.skip("CriticalPoint not yet implemented")
            
        cp = CriticalPoint(
            position_id=1,
            delta_q=0.01,
            order_parameter=0.05,
            susceptibility=10.0,
            correlation_length=5.0
        )
        
        data = cp.to_dict()
        assert data['delta_q'] == 0.01


class TestPhaseTransitions:
    """Test phase transition detection"""
    
    def test_phase_transition_detection(self):
        """Should detect phase transitions in policy evolution"""
        if CriticalPhenomenaAnalyzer is None:
            pytest.skip("CriticalPhenomenaAnalyzer not yet implemented")
            
        analyzer = CriticalPhenomenaAnalyzer()
        
        # Simulate policy evolution with phase transition
        entropy_trajectory = [
            2.0, 1.9, 1.8, 1.7,  # Gradual decrease
            1.0, 0.5, 0.3,       # Sudden drop (phase transition)
            0.2, 0.2, 0.2        # Stable
        ]
        
        transitions = analyzer.detect_phase_transitions(entropy_trajectory)
        
        assert len(transitions) >= 1
        # Transition should be around index 4-5
        assert 3 <= transitions[0] <= 6
    
    def test_phase_diagram(self):
        """Should construct phase diagram"""
        if CriticalPhenomenaAnalyzer is None:
            pytest.skip("CriticalPhenomenaAnalyzer not yet implemented")
            
        analyzer = CriticalPhenomenaAnalyzer()
        
        # Data points in (c_puct, complexity) space
        phase_data = []
        
        for c_puct in [0.1, 0.5, 1.0, 2.0, 5.0]:
            for complexity in [1, 2, 3, 4, 5]:
                # Simulate different phases
                if c_puct < 1.0:
                    phase = 'exploration'
                elif c_puct > 2.0:
                    phase = 'exploitation'
                else:
                    phase = 'critical'
                    
                phase_data.append({
                    'c_puct': c_puct,
                    'complexity': complexity,
                    'phase': phase,
                    'order_parameter': c_puct / (1 + complexity)
                })
        
        phase_diagram = analyzer.construct_phase_diagram(phase_data)
        
        assert 'boundaries' in phase_diagram
        assert 'phases' in phase_diagram


if __name__ == "__main__":
    pytest.main([__file__, "-v"])