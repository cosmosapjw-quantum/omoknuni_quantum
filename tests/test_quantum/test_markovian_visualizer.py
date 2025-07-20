#!/usr/bin/env python3
"""
Test suite for Markovian validation visualization.

Following TDD principles - tests written before implementation.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
from typing import Dict, List

# Import modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'python/mcts/quantum'))
    from markovian_visualizer import MarkovianVisualizer
except ImportError as e:
    print(f"Import error: {e}")
    MarkovianVisualizer = None


def create_test_validation_results() -> Dict:
    """Create test validation results for visualization"""
    return {
        'autocorrelation': {
            'c1': 0.02,
            'c1_ci': (0.01, 0.03),
            'tau_c': 2.5,
            'fit_quality': 0.95,
            'correlation_function': np.exp(-np.arange(11) / 2.5),
            'lags': np.arange(11)
        },
        'markov_test': {
            'js_divergences': {2: 0.005, 3: 0.003},
            'markovian': True,
            'transition_matrices': {
                1: np.random.rand(5, 5),
                2: np.random.rand(25, 5)
            }
        },
        'analytical_comparison': {
            'c1_measured': 0.02,
            'c1_predicted': 0.018,
            'c1_ratio': 1.11,
            'n_visits_range': np.arange(10, 1000, 10),
            'c1_scaling': 1.0 / np.arange(10, 1000, 10)
        },
        'raw_data': {
            'all_correlations': [np.random.normal(0.02, 0.005, 100) for _ in range(50)],
            'all_tau_c': np.random.gamma(2.5, 0.5, 50),
            'game_lengths': np.random.randint(100, 1000, 50)
        }
    }


class TestMarkovianVisualizer:
    """Test suite for Markovian validation visualizer"""
    
    def test_visualizer_exists(self):
        """MarkovianVisualizer class should exist"""
        assert MarkovianVisualizer is not None, "MarkovianVisualizer not implemented"
    
    def test_plot_autocorrelation_function(self):
        """Should plot autocorrelation function with exponential fit"""
        if MarkovianVisualizer is None:
            pytest.skip("MarkovianVisualizer not yet implemented")
            
        visualizer = MarkovianVisualizer()
        results = create_test_validation_results()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = visualizer.plot_autocorrelation_function(
                results, save_path=os.path.join(tmpdir, "autocorr.png")
            )
            
            assert os.path.exists(save_path)
            assert save_path.endswith('.png')
            
        # Check that figure was created
        assert len(plt.get_fignums()) > 0
        plt.close('all')
    
    def test_plot_markov_property_test(self):
        """Should visualize Markov property test results"""
        if MarkovianVisualizer is None:
            pytest.skip("MarkovianVisualizer not yet implemented")
            
        visualizer = MarkovianVisualizer()
        results = create_test_validation_results()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = visualizer.plot_markov_property_test(
                results, save_path=os.path.join(tmpdir, "markov_test.png")
            )
            
            assert os.path.exists(save_path)
            
        plt.close('all')
    
    def test_plot_analytical_comparison(self):
        """Should plot analytical vs measured correlations"""
        if MarkovianVisualizer is None:
            pytest.skip("MarkovianVisualizer not yet implemented")
            
        visualizer = MarkovianVisualizer()
        results = create_test_validation_results()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = visualizer.plot_analytical_comparison(
                results, save_path=os.path.join(tmpdir, "analytical.png")
            )
            
            assert os.path.exists(save_path)
            
        plt.close('all')
    
    def test_plot_correlation_distribution(self):
        """Should plot distribution of correlations across games"""
        if MarkovianVisualizer is None:
            pytest.skip("MarkovianVisualizer not yet implemented")
            
        visualizer = MarkovianVisualizer()
        results = create_test_validation_results()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = visualizer.plot_correlation_distribution(
                results, save_path=os.path.join(tmpdir, "corr_dist.png")
            )
            
            assert os.path.exists(save_path)
            
        plt.close('all')
    
    def test_plot_transition_matrix_heatmap(self):
        """Should plot transition matrix as heatmap"""
        if MarkovianVisualizer is None:
            pytest.skip("MarkovianVisualizer not yet implemented")
            
        visualizer = MarkovianVisualizer()
        results = create_test_validation_results()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = visualizer.plot_transition_matrices(
                results, save_path=os.path.join(tmpdir, "transitions.png")
            )
            
            assert os.path.exists(save_path)
            
        plt.close('all')
    
    def test_generate_full_report(self):
        """Should generate comprehensive report with all plots"""
        if MarkovianVisualizer is None:
            pytest.skip("MarkovianVisualizer not yet implemented")
            
        visualizer = MarkovianVisualizer()
        results = create_test_validation_results()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            report_paths = visualizer.generate_full_report(
                results, output_dir=tmpdir
            )
            
            # Should create multiple figures
            assert len(report_paths) >= 4
            assert all(os.path.exists(p) for p in report_paths.values())
            
            # Check for summary figure
            assert 'summary' in report_paths
            
        plt.close('all')
    
    def test_plot_timescale_separation(self):
        """Should visualize timescale separation"""
        if MarkovianVisualizer is None:
            pytest.skip("MarkovianVisualizer not yet implemented")
            
        visualizer = MarkovianVisualizer()
        results = create_test_validation_results()
        
        # Add timescale data
        results['timescales'] = {
            'tau_env': 1.0,
            'tau_sys': 100.0,
            'separation_ratio': 100.0
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = visualizer.plot_timescale_separation(
                results, save_path=os.path.join(tmpdir, "timescales.png")
            )
            
            assert os.path.exists(save_path)
            
        plt.close('all')
    
    def test_interactive_mode(self):
        """Should support interactive mode without saving"""
        if MarkovianVisualizer is None:
            pytest.skip("MarkovianVisualizer not yet implemented")
            
        visualizer = MarkovianVisualizer()
        results = create_test_validation_results()
        
        # Should not save when save_path is None
        fig = visualizer.plot_autocorrelation_function(
            results, save_path=None, show=False
        )
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        plt.close('all')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])