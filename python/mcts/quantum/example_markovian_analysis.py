#!/usr/bin/env python3
"""
Complete example of Markovian validation with visualizations.

This script demonstrates the full pipeline:
1. Generate or load MCTS data
2. Run Markovian validation
3. Generate comprehensive visualizations
4. Produce analysis report
"""

import numpy as np
import sys
import os
from typing import List, Dict
import logging
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from markovian_validation import MarkovianValidator
from markovian_visualizer import MarkovianVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_mcts_data(n_games: int = 100, 
                      min_simulations: int = 100,
                      max_simulations: int = 1000) -> List[Dict]:
    """
    Generate synthetic MCTS data that mimics real game behavior.
    
    This creates data with:
    - Decreasing value variance as visits increase (UCB exploration)
    - Weak temporal correlations (1/N scaling)
    - Realistic Q-value convergence
    """
    logger.info(f"Generating {n_games} synthetic MCTS games...")
    
    game_data = []
    
    for game_idx in range(n_games):
        n_simulations = np.random.randint(min_simulations, max_simulations)
        
        # Game parameters
        true_value = np.random.uniform(-0.5, 0.5)  # True minimax value
        exploration_const = np.sqrt(2)  # UCB constant
        initial_uncertainty = 0.5
        
        # Initialize arrays
        values = np.zeros(n_simulations)
        visit_counts = np.arange(1, n_simulations + 1)
        q_values = np.zeros(n_simulations)
        
        # Simulate MCTS with UCB
        running_sum = 0.0
        
        for t in range(n_simulations):
            # UCB exploration bonus decreases over time
            exploration_bonus = exploration_const * np.sqrt(np.log(t + 1) / (t + 1))
            
            # Value noise decreases with visits (convergence)
            noise_scale = initial_uncertainty / np.sqrt(t + 1)
            
            # Simulate rollout value
            rollout_value = true_value + np.random.normal(0, noise_scale)
            
            # Small temporal correlation (violates perfect Markov property)
            if t > 0:
                correlation_strength = 1.0 / (t + 1)  # Decays as 1/N
                rollout_value += correlation_strength * (values[t-1] - true_value)
            
            values[t] = rollout_value
            
            # Update Q-value (running average)
            running_sum += rollout_value
            q_values[t] = running_sum / (t + 1)
        
        game_data.append({
            'game_id': game_idx,
            'values': values,
            'visit_counts': visit_counts,
            'q_values': q_values,
            'true_value': true_value,
            'final_q': q_values[-1],
            'convergence_error': abs(q_values[-1] - true_value)
        })
    
    logger.info(f"Generated {len(game_data)} games with {min_simulations}-{max_simulations} simulations each")
    return game_data


def print_summary_statistics(game_data: List[Dict]):
    """Print summary statistics of the game data"""
    print("\n" + "="*60)
    print("MCTS DATA SUMMARY")
    print("="*60)
    
    n_games = len(game_data)
    avg_simulations = np.mean([len(g['values']) for g in game_data])
    convergence_errors = [g['convergence_error'] for g in game_data]
    
    print(f"Number of games: {n_games}")
    print(f"Average simulations per game: {avg_simulations:.0f}")
    print(f"Mean convergence error: {np.mean(convergence_errors):.4f}")
    print(f"Std convergence error: {np.std(convergence_errors):.4f}")
    print("="*60 + "\n")


def main():
    """Run complete Markovian analysis with visualizations"""
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"markovian_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Created output directory: {output_dir}")
    
    # Step 1: Generate or load MCTS data
    game_data = generate_mcts_data(n_games=100)
    print_summary_statistics(game_data)
    
    # Step 2: Run Markovian validation
    logger.info("Running Markovian validation...")
    validator = MarkovianValidator()
    results = validator.validate(game_data)
    
    # Print key results
    print("\n" + "="*60)
    print("MARKOVIAN VALIDATION RESULTS")
    print("="*60)
    
    print("\n1. AUTOCORRELATION ANALYSIS:")
    print(f"   C(1) = {results['autocorrelation']['c1']:.6f}")
    print(f"   95% CI: [{results['autocorrelation']['c1_ci'][0]:.6f}, "
          f"{results['autocorrelation']['c1_ci'][1]:.6f}]")
    print(f"   Correlation time τ_c = {results['autocorrelation']['tau_c']:.2f}")
    print(f"   Exponential fit R² = {results['autocorrelation']['fit_quality']:.3f}")
    
    print("\n2. MARKOV PROPERTY TEST:")
    js_divs = results['markov_test']['js_divergences']
    for order, div in sorted(js_divs.items()):
        print(f"   JS divergence (order {order}): {div:.6f}")
    print(f"   Is Markovian? {'YES' if results['markov_test']['markovian'] else 'NO'}")
    
    print("\n3. ANALYTICAL PREDICTIONS:")
    print(f"   Measured C(1): {results['analytical_comparison']['c1_measured']:.6f}")
    print(f"   Predicted C(1): {results['analytical_comparison']['c1_predicted']:.6f}")
    print(f"   Ratio: {results['analytical_comparison']['c1_ratio']:.2f}")
    
    print("\n4. TIMESCALE SEPARATION:")
    timescales = results.get('timescales', {})
    if timescales:
        print(f"   τ_env = {timescales['tau_env']:.1f}")
        print(f"   τ_sys = {timescales['tau_sys']:.1f}")
        print(f"   Separation ratio = {timescales['separation_ratio']:.1f}")
    
    print("="*60 + "\n")
    
    # Step 3: Generate visualizations
    logger.info("Generating visualizations...")
    visualizer = MarkovianVisualizer(output_dir=output_dir)
    
    # Generate all plots
    plot_paths = visualizer.generate_full_report(results, output_dir=output_dir)
    
    logger.info(f"Generated {len(plot_paths)} visualizations:")
    for name, path in plot_paths.items():
        logger.info(f"  - {name}: {path}")
    
    # Step 4: Save results to JSON
    json_results = convert_results_to_json(results)
    json_path = os.path.join(output_dir, 'markovian_validation_results.json')
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Saved numerical results to {json_path}")
    
    # Step 5: Generate text report
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    generate_text_report(results, game_data, report_path)
    
    logger.info(f"Generated text report: {report_path}")
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}/")
    print("\nKey findings:")
    
    # Final interpretation
    if results['markov_test']['markovian']:
        print("✓ The MCTS process is effectively Markovian")
    else:
        print("✗ Significant non-Markovian effects detected")
    
    if results['autocorrelation']['tau_c'] < 5:
        print("✓ Short correlation time supports theoretical approximations")
    else:
        print("✗ Long correlations may require corrections")
    
    if 0.5 < results['analytical_comparison']['c1_ratio'] < 2.0:
        print("✓ Good agreement between theory and measurement")
    else:
        print("✗ Theory deviates from empirical results")


def convert_results_to_json(results: Dict) -> Dict:
    """Convert numpy arrays to lists for JSON serialization"""
    def convert_value(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        elif isinstance(v, (np.floating, np.float32, np.float64)):
            return float(v)
        elif isinstance(v, (np.integer, np.int32, np.int64)):
            return int(v)
        elif isinstance(v, (np.bool_, bool)):
            return bool(v)
        elif isinstance(v, dict):
            return {k: convert_value(val) for k, val in v.items()}
        elif isinstance(v, list):
            return [convert_value(item) for item in v]
        elif isinstance(v, tuple):
            return tuple(convert_value(item) for item in v)
        else:
            return v
    
    return convert_value(results)


def generate_text_report(results: Dict, game_data: List[Dict], output_path: str):
    """Generate detailed text report of the analysis"""
    with open(output_path, 'w') as f:
        f.write("MARKOVIAN VALIDATION ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of games analyzed: {len(game_data)}\n")
        f.write(f"Total simulations: {sum(len(g['values']) for g in game_data)}\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-"*40 + "\n")
        
        # Key findings
        markovian = results['markov_test']['markovian']
        tau_c = results['autocorrelation']['tau_c']
        c1_ratio = results['analytical_comparison']['c1_ratio']
        
        if markovian and tau_c < 5 and 0.5 < c1_ratio < 2.0:
            f.write("The MCTS implementation exhibits Markovian behavior with good\n")
            f.write("theoretical agreement. The mean-field approximation is valid.\n\n")
        else:
            f.write("Deviations from ideal Markovian behavior detected:\n")
            if not markovian:
                f.write("  - Non-Markovian memory effects present\n")
            if tau_c >= 5:
                f.write("  - Long correlation times observed\n")
            if not (0.5 < c1_ratio < 2.0):
                f.write("  - Theory-experiment mismatch\n")
            f.write("\n")
        
        f.write("DETAILED RESULTS\n")
        f.write("-"*40 + "\n\n")
        
        # Autocorrelation
        f.write("1. Temporal Correlations:\n")
        f.write(f"   - One-step correlation C(1) = {results['autocorrelation']['c1']:.6f}\n")
        f.write(f"   - 95% confidence interval: [{results['autocorrelation']['c1_ci'][0]:.6f}, "
                f"{results['autocorrelation']['c1_ci'][1]:.6f}]\n")
        f.write(f"   - Correlation decay time τ_c = {tau_c:.3f}\n")
        f.write(f"   - Exponential fit quality R² = {results['autocorrelation']['fit_quality']:.3f}\n\n")
        
        # Markov property
        f.write("2. Markov Property Test:\n")
        js_divs = results['markov_test']['js_divergences']
        for order in sorted(js_divs.keys()):
            f.write(f"   - JS divergence (order {order}): {js_divs[order]:.6f}\n")
        f.write(f"   - Markovian threshold: 0.01\n")
        f.write(f"   - Classification: {'MARKOVIAN' if markovian else 'NON-MARKOVIAN'}\n\n")
        
        # Analytical comparison
        f.write("3. Theory vs Experiment:\n")
        comp = results['analytical_comparison']
        f.write(f"   - Measured C(1): {comp['c1_measured']:.6f}\n")
        f.write(f"   - Predicted C(1): {comp['c1_predicted']:.6f}\n")
        f.write(f"   - Ratio (measured/predicted): {comp['c1_ratio']:.3f}\n")
        f.write(f"   - Average visit count: {comp['avg_n']:.1f}\n\n")
        
        # Timescales
        if 'timescales' in results:
            f.write("4. Timescale Analysis:\n")
            ts = results['timescales']
            f.write(f"   - Environment timescale τ_env = {ts['tau_env']:.1f}\n")
            f.write(f"   - System timescale τ_sys = {ts['tau_sys']:.1f}\n")
            f.write(f"   - Separation ratio = {ts['separation_ratio']:.1f}\n")
            f.write(f"   - Interpretation: {'Strong separation' if ts['separation_ratio'] > 10 else 'Weak separation'}\n\n")
        
        f.write("RECOMMENDATIONS\n")
        f.write("-"*40 + "\n")
        
        if markovian and tau_c < 5:
            f.write("1. The Markovian approximation is valid for this system.\n")
            f.write("2. Standard MCTS theory applies without corrections.\n")
            f.write("3. Continue using mean-field theoretical framework.\n")
        else:
            f.write("1. Consider non-Markovian corrections to improve accuracy.\n")
            f.write("2. Investigate sources of temporal correlations.\n")
            f.write("3. May need to include memory kernel in theoretical treatment.\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("END OF REPORT\n")


if __name__ == "__main__":
    main()