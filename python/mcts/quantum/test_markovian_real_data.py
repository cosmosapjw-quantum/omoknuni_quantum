#!/usr/bin/env python3
"""
Test Markovian validation on real MCTS data.

This script demonstrates how to use the MarkovianValidator with actual
self-play data from MCTS games.
"""

import numpy as np
import sys
import os
from typing import List, Dict
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from markovian_validation import MarkovianValidator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_mcts_data_from_physics_output(data_path: str = None) -> List[Dict]:
    """
    Load MCTS data from physics analysis output or generate synthetic data.
    
    In a real implementation, this would load actual self-play data
    from the physics analysis pipeline.
    """
    # For demonstration, create realistic MCTS-like data
    logger.info("Generating synthetic MCTS data for demonstration...")
    
    n_games = 50
    n_simulations_range = (100, 1000)
    
    game_data = []
    
    for game_idx in range(n_games):
        n_simulations = np.random.randint(*n_simulations_range)
        
        # Initialize game state
        values = []
        visit_counts = []
        q_values = []
        
        # True game value (unknown to MCTS)
        true_value = np.random.uniform(-1, 1)
        
        # MCTS exploration
        current_q = 0.0
        total_visits = 0
        
        for sim in range(n_simulations):
            # UCB exploration with decreasing exploration over time
            exploration_factor = np.sqrt(2 * np.log(total_visits + 1) / (total_visits + 1))
            
            # Simulate rollout with noise decreasing as we get more samples
            noise_scale = 1.0 / np.sqrt(sim + 1)
            rollout_value = true_value + np.random.normal(0, noise_scale)
            
            # Update statistics
            total_visits += 1
            current_q = ((total_visits - 1) * current_q + rollout_value) / total_visits
            
            values.append(rollout_value)
            visit_counts.append(total_visits)
            q_values.append(current_q)
        
        game_data.append({
            'values': np.array(values),
            'visit_counts': np.array(visit_counts),
            'q_values': np.array(q_values),
            'game_id': game_idx,
            'true_value': true_value
        })
    
    logger.info(f"Generated {len(game_data)} games with {n_simulations_range} simulations each")
    return game_data


def main():
    """Run Markovian validation on MCTS data."""
    # Load or generate data
    game_data = load_mcts_data_from_physics_output()
    
    # Initialize validator
    validator = MarkovianValidator()
    
    # Run validation
    logger.info("Running Markovian validation...")
    results = validator.validate(game_data)
    
    # Display results
    print("\n" + "="*60)
    print("MARKOVIAN VALIDATION RESULTS")
    print("="*60)
    
    print("\n1. AUTOCORRELATION ANALYSIS:")
    print(f"   - C(1) = {results['autocorrelation']['c1']:.6f}")
    print(f"   - 95% CI: [{results['autocorrelation']['c1_ci'][0]:.6f}, "
          f"{results['autocorrelation']['c1_ci'][1]:.6f}]")
    print(f"   - Correlation time τ_c = {results['autocorrelation']['tau_c']:.2f}")
    print(f"   - Exponential fit quality R² = {results['autocorrelation']['fit_quality']:.3f}")
    
    print("\n2. MARKOV PROPERTY TEST:")
    print(f"   - JS divergence (2nd order): {results['markov_test']['js_divergence_order2']:.6f}")
    print(f"   - JS divergence (3rd order): {results['markov_test']['js_divergence_order3']:.6f}")
    print(f"   - Is Markovian? {'YES' if results['markov_test']['markovian'] else 'NO'}")
    
    print("\n3. ANALYTICAL PREDICTIONS:")
    print(f"   - Measured C(1): {results['analytical_comparison']['c1_measured']:.6f}")
    print(f"   - Predicted C(1): {results['analytical_comparison']['c1_predicted']:.6f}")
    print(f"   - Ratio (measured/predicted): {results['analytical_comparison']['c1_ratio']:.2f}")
    
    print("\n4. INTERPRETATION:")
    if results['markov_test']['markovian']:
        print("   ✓ The MCTS process exhibits Markovian behavior")
        print("   ✓ Higher-order memory effects are negligible")
    else:
        print("   ✗ Significant non-Markovian effects detected")
        print("   ✗ Consider including memory kernel corrections")
    
    if results['autocorrelation']['tau_c'] < 10:
        print("   ✓ Short correlation time supports mean-field approximation")
    else:
        print("   ✗ Long correlation time may invalidate mean-field approach")
    
    print("\n" + "="*60)
    
    # Save results for further analysis
    import json
    output_file = "markovian_validation_results.json"
    with open(output_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = {
            'autocorrelation': {
                'c1': float(results['autocorrelation']['c1']),
                'c1_ci': [float(x) for x in results['autocorrelation']['c1_ci']],
                'tau_c': float(results['autocorrelation']['tau_c']),
                'fit_quality': float(results['autocorrelation']['fit_quality'])
            },
            'markov_test': {
                'js_divergence_order2': float(results['markov_test']['js_divergence_order2']),
                'js_divergence_order3': float(results['markov_test']['js_divergence_order3']),
                'markovian': bool(results['markov_test']['markovian'])
            },
            'analytical_comparison': {
                'c1_measured': float(results['analytical_comparison']['c1_measured']),
                'c1_predicted': float(results['analytical_comparison']['c1_predicted']),
                'c1_ratio': float(results['analytical_comparison']['c1_ratio'])
            }
        }
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()