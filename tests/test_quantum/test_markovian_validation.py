#!/usr/bin/env python3
"""
Test suite for Markovian approximation validation and analytical justification.

This module tests:
1. Autocorrelation of value fluctuations (Test 1)
2. Direct test of Markov property (Test 2)
3. Analytical predictions (C(1) ~ 1/N, timescale separation)
"""

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import to-be-implemented modules (will fail initially per TDD)
try:
    # Add quantum module to path for direct import
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'python/mcts/quantum'))
    from markovian_validation import (
        MarkovianValidator,
        AutocorrelationAnalyzer,
        MarkovPropertyTester,
        AnalyticalPredictions
    )
except ImportError as e:
    # Expected to fail before implementation
    print(f"Import error: {e}")
    MarkovianValidator = None
    AutocorrelationAnalyzer = None
    MarkovPropertyTester = None
    AnalyticalPredictions = None


class TestMarkovianValidatorExists:
    """Test that the MarkovianValidator class exists"""
    
    def test_validator_class_exists(self):
        """MarkovianValidator class should exist"""
        assert MarkovianValidator is not None, "MarkovianValidator class not implemented"


class TestAutocorrelationAnalyzer:
    """Test suite for autocorrelation analysis of value fluctuations"""
    
    def test_analyzer_exists(self):
        """AutocorrelationAnalyzer should exist"""
        assert AutocorrelationAnalyzer is not None, "AutocorrelationAnalyzer not implemented"
    
    def test_compute_autocorrelation_single_lag(self):
        """Should compute C(1) for single lag"""
        if AutocorrelationAnalyzer is None:
            pytest.skip("AutocorrelationAnalyzer not yet implemented")
        
        # Create test data with known correlation
        n_simulations = 1000
        values = np.random.normal(0, 1, n_simulations)
        # Add small correlation
        for i in range(1, n_simulations):
            values[i] += 0.1 * values[i-1]
        
        analyzer = AutocorrelationAnalyzer()
        c1 = analyzer.compute_autocorrelation(values, lag=1)
        
        assert isinstance(c1, float), "C(1) should be a float"
        assert -1 <= c1 <= 1, "Correlation should be between -1 and 1"
        assert c1 > 0, "Should detect positive correlation"
    
    def test_compute_autocorrelation_multiple_lags(self):
        """Should compute C(τ) for multiple lags"""
        if AutocorrelationAnalyzer is None:
            pytest.skip("AutocorrelationAnalyzer not yet implemented")
        
        values = np.random.normal(0, 1, 1000)
        analyzer = AutocorrelationAnalyzer()
        
        correlations = analyzer.compute_autocorrelation_function(values, max_lag=10)
        
        assert len(correlations) == 11, "Should return correlations for lags 0 to 10"
        assert correlations[0] == pytest.approx(1.0), "C(0) should be 1"
        assert all(abs(c) <= 1 for c in correlations), "All correlations should be |C| <= 1"
    
    def test_fit_exponential_decay(self):
        """Should fit exponential decay to correlation function"""
        if AutocorrelationAnalyzer is None:
            pytest.skip("AutocorrelationAnalyzer not yet implemented")
        
        # Create synthetic exponential decay
        lags = np.arange(0, 10)
        tau_c_true = 2.5
        correlations = np.exp(-lags / tau_c_true)
        
        analyzer = AutocorrelationAnalyzer()
        tau_c_fit, r_squared = analyzer.fit_exponential_decay(correlations)
        
        assert abs(tau_c_fit - tau_c_true) < 0.5, f"Should recover true τ_c within 0.5"
        assert r_squared > 0.95, "Should have good fit for exponential data"
    
    def test_bootstrap_confidence_intervals(self):
        """Should compute bootstrap confidence intervals"""
        if AutocorrelationAnalyzer is None:
            pytest.skip("AutocorrelationAnalyzer not yet implemented")
        
        # Multiple runs of same experiment
        n_runs = 100
        n_simulations = 500
        all_values = [np.random.normal(0, 1, n_simulations) for _ in range(n_runs)]
        
        analyzer = AutocorrelationAnalyzer()
        c1_mean, c1_lower, c1_upper = analyzer.bootstrap_correlation(
            all_values, lag=1, n_bootstrap=1000, confidence=0.95
        )
        
        assert c1_lower < c1_mean < c1_upper, "Mean should be within CI"
        assert c1_upper - c1_lower < 0.2, "CI should be reasonably tight with 100 runs"


class TestMarkovPropertyTester:
    """Test suite for direct Markov property testing"""
    
    def test_tester_exists(self):
        """MarkovPropertyTester should exist"""
        assert MarkovPropertyTester is not None, "MarkovPropertyTester not implemented"
    
    def test_state_discretization(self):
        """Should discretize continuous states appropriately"""
        if MarkovPropertyTester is None:
            pytest.skip("MarkovPropertyTester not yet implemented")
        
        # Create continuous states (N, Q, σ²)
        states = np.array([
            [10, 0.5, 0.1],
            [11, 0.52, 0.09],
            [50, 0.8, 0.02],
            [51, 0.81, 0.02]
        ])
        
        tester = MarkovPropertyTester()
        discrete_states = tester.discretize_states(states, n_bins=10)
        
        assert discrete_states.shape == states.shape[:1], "Should return state indices"
        assert discrete_states.dtype == np.int32, "Should return integer indices"
        assert len(np.unique(discrete_states)) <= 2, "Similar states should map to same bin"
    
    def test_compute_transition_probabilities(self):
        """Should compute empirical transition probabilities"""
        if MarkovPropertyTester is None:
            pytest.skip("MarkovPropertyTester not yet implemented")
        
        # Create state sequences from multiple runs
        n_runs = 100
        sequence_length = 50
        n_states = 5
        
        sequences = []
        for _ in range(n_runs):
            # Generate Markov chain
            seq = [np.random.randint(n_states)]
            for _ in range(sequence_length - 1):
                # Simple transition matrix
                seq.append((seq[-1] + np.random.randint(2)) % n_states)
            sequences.append(np.array(seq))
        
        tester = MarkovPropertyTester()
        
        # First-order transitions P(S_{k+1} | S_k)
        p_first = tester.compute_transition_matrix(sequences, order=1)
        assert p_first.shape == (n_states, n_states), "Should be n_states x n_states"
        assert np.allclose(p_first.sum(axis=1), 1.0), "Rows should sum to 1"
        
        # Second-order transitions P(S_{k+1} | S_k, S_{k-1})
        p_second = tester.compute_transition_matrix(sequences, order=2)
        assert p_second.shape[0] == n_states ** 2, "Should have n_states² history states"
    
    def test_jensen_shannon_divergence(self):
        """Should compute JS divergence between distributions"""
        if MarkovPropertyTester is None:
            pytest.skip("MarkovPropertyTester not yet implemented")
        
        # Create two similar distributions
        p = np.array([0.3, 0.5, 0.2])
        q = np.array([0.25, 0.55, 0.2])
        
        tester = MarkovPropertyTester()
        js_div = tester.jensen_shannon_divergence(p, q)
        
        assert 0 <= js_div <= 1, "JS divergence should be in [0, 1]"
        assert js_div < 0.1, "Similar distributions should have small divergence"
        
        # Test symmetry
        js_div_rev = tester.jensen_shannon_divergence(q, p)
        assert abs(js_div - js_div_rev) < 1e-10, "JS divergence should be symmetric"
    
    def test_markov_property_test(self):
        """Should test Markov property on MCTS-like data"""
        if MarkovPropertyTester is None:
            pytest.skip("MarkovPropertyTester not yet implemented")
        
        # Generate MCTS-like state trajectories
        n_runs = 500
        n_simulations = 200
        
        trajectories = []
        for _ in range(n_runs):
            # Start with initial state
            N, Q, var = 1, 0.5, 0.25
            trajectory = []
            
            for _ in range(n_simulations):
                # MCTS-like update
                value = np.random.normal(Q, np.sqrt(var))
                N += 1
                Q = ((N-1) * Q + value) / N
                var = var * 0.99  # Variance decay
                
                trajectory.append([N, Q, var])
            
            trajectories.append(np.array(trajectory))
        
        tester = MarkovPropertyTester()
        results = tester.test_markov_property(trajectories, max_order=3)
        
        assert 'js_divergences' in results
        assert results['js_divergences'][2] < 0.01, "First-order memory should be negligible"
        assert results['js_divergences'][3] <= results['js_divergences'][2], "No higher-order memory"


class TestAnalyticalPredictions:
    """Test analytical predictions for correlation and timescales"""
    
    def test_predictions_exists(self):
        """AnalyticalPredictions should exist"""
        assert AnalyticalPredictions is not None, "AnalyticalPredictions not implemented"
    
    def test_predict_c1_correlation(self):
        """Should predict C(1) ~ 1/N analytically"""
        if AnalyticalPredictions is None:
            pytest.skip("AnalyticalPredictions not yet implemented")
        
        predictor = AnalyticalPredictions()
        
        # Test parameters
        n_visits = 100
        beta = 1.0
        sigma_q = 0.2
        
        c1_predicted = predictor.predict_c1(
            n_visits=n_visits,
            beta=beta,
            sigma_q=sigma_q
        )
        
        assert isinstance(c1_predicted, float)
        assert 0 < c1_predicted < 0.1, "C(1) should be small for N=100"
        assert abs(c1_predicted - 1/n_visits) < 0.02, "Should scale approximately as 1/N"
    
    def test_predict_correlation_decay(self):
        """Should predict exponential decay C(m) ~ (1/N)^m"""
        if AnalyticalPredictions is None:
            pytest.skip("AnalyticalPredictions not yet implemented")
        
        predictor = AnalyticalPredictions()
        
        n_visits = 50
        correlations = [predictor.predict_cm(m, n_visits) for m in range(5)]
        
        # Check exponential decay
        ratios = [correlations[i+1] / correlations[i] for i in range(4)]
        assert all(abs(r - 1/n_visits) < 0.02 for r in ratios), "Should decay as (1/N)^m"
    
    def test_timescale_separation(self):
        """Should compute timescale separation ratio"""
        if AnalyticalPredictions is None:
            pytest.skip("AnalyticalPredictions not yet implemented")
        
        predictor = AnalyticalPredictions()
        
        n_visits = 100
        tau_env, tau_sys = predictor.compute_timescales(n_visits)
        
        assert tau_env == pytest.approx(1, abs=0.5), "Environment timescale should be O(1)"
        assert tau_sys == pytest.approx(n_visits, rel=0.3), "System timescale should be O(N)"
        assert tau_sys / tau_env > 10, "Should have clear timescale separation for N=100"


class TestMarkovianValidatorIntegration:
    """Integration tests for complete validation pipeline"""
    
    def test_full_validation_pipeline(self):
        """Should run complete validation on MCTS data"""
        if MarkovianValidator is None:
            pytest.skip("MarkovianValidator not yet implemented")
        
        # Create synthetic MCTS data
        n_games = 10
        n_simulations = 500
        
        game_data = []
        for _ in range(n_games):
            # Create more realistic MCTS-like data with weak correlations
            values = []
            q_values = []
            
            # Initial state
            q_true = 0.5 + np.random.normal(0, 0.05)  # True value
            
            for i in range(n_simulations):
                # MCTS-like value generation with decreasing noise
                noise_scale = 0.1 / np.sqrt(i + 1)  # Variance decreases with visits
                value = q_true + np.random.normal(0, noise_scale)
                values.append(value)
                
                # Update Q-value estimate (running average)
                if i == 0:
                    q_values.append(value)
                else:
                    q_new = ((i) * q_values[-1] + value) / (i + 1)
                    q_values.append(q_new)
            
            game_data.append({
                'values': np.array(values),
                'visit_counts': np.arange(1, n_simulations + 1),
                'q_values': np.array(q_values)
            })
        
        validator = MarkovianValidator()
        results = validator.validate(game_data)
        
        assert 'autocorrelation' in results
        assert 'markov_test' in results
        assert 'analytical_comparison' in results
        
        # Check autocorrelation results
        assert results['autocorrelation']['tau_c'] < 20, "Correlation time should be finite"
        assert abs(results['autocorrelation']['c1']) < 0.5, "C(1) should be moderate"
        
        # Check Markov test results
        assert results['markov_test']['js_divergence_order2'] < 0.05, "Should be approximately Markovian"
        
        # Check analytical comparison exists and is reasonable
        assert 'c1_measured' in results['analytical_comparison']
        assert 'c1_predicted' in results['analytical_comparison']
        assert 'c1_ratio' in results['analytical_comparison']
        assert results['analytical_comparison']['c1_predicted'] > 0, "Predicted C(1) should be positive"


if __name__ == "__main__":
    # Pretty output for manual testing
    print("\n" + "="*60)
    print("Testing Markovian Validation")
    print("="*60 + "\n")
    
    # Run tests with simple output
    test_classes = [
        TestMarkovianValidatorExists,
        TestAutocorrelationAnalyzer,
        TestMarkovPropertyTester,
        TestAnalyticalPredictions,
        TestMarkovianValidatorIntegration
    ]
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        test_obj = test_class()
        for method_name in dir(test_obj):
            if method_name.startswith('test_'):
                try:
                    method = getattr(test_obj, method_name)
                    method()
                    print(f"  ✓ {method_name}")
                except AssertionError as e:
                    print(f"  ✗ {method_name}: {e}")
                except Exception as e:
                    print(f"  ✗ {method_name}: {type(e).__name__}: {e}")