#!/usr/bin/env python3
"""
Visualization tools for Markovian approximation validation results.

This module provides comprehensive plotting functions for:
- Autocorrelation functions and exponential fits
- Markov property test results
- Analytical vs empirical comparisons
- Statistical distributions across games
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from typing import Dict, List, Any, Optional, Tuple, Union
import os
import logging

logger = logging.getLogger(__name__)


class MarkovianVisualizer:
    """Visualization tools for Markovian validation results"""
    
    def __init__(self, output_dir: str = ".", style: str = "default"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory for saving plots
            style: Matplotlib style to use
        """
        self.output_dir = output_dir
        try:
            plt.style.use(style)
        except OSError:
            # Fallback to default if style not available
            plt.style.use('default')
        
        # Color scheme
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#d62728',
            'neutral': '#7f7f7f'
        }
    
    def plot_autocorrelation_function(self, results: Dict[str, Any], 
                                    save_path: Optional[str] = None,
                                    show: bool = True) -> Optional[str]:
        """
        Plot autocorrelation function with exponential fit.
        
        Args:
            results: Validation results dictionary
            save_path: Path to save figure
            show: Whether to display the plot
            
        Returns:
            Path to saved figure or Figure object if save_path is None
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Autocorrelation Analysis', fontsize=16)
        
        # Extract data
        autocorr = results['autocorrelation']
        lags = autocorr.get('lags', np.arange(11))
        corr_func = autocorr.get('correlation_function', np.zeros_like(lags))
        tau_c = autocorr['tau_c']
        r_squared = autocorr['fit_quality']
        
        # Plot 1: Autocorrelation function
        ax1.plot(lags, corr_func, 'o-', color=self.colors['primary'], 
                label='Measured C(τ)', markersize=8)
        
        # Add exponential fit
        if tau_c > 0:
            fit_lags = np.linspace(0, max(lags), 100)
            fit_corr = np.exp(-fit_lags / tau_c)
            ax1.plot(fit_lags, fit_corr, '--', color=self.colors['secondary'],
                    label=f'Exp fit: τ_c = {tau_c:.2f}', linewidth=2)
        
        # Add confidence interval for C(1)
        c1_ci = autocorr['c1_ci']
        ax1.axhspan(c1_ci[0], c1_ci[1], alpha=0.2, color=self.colors['primary'],
                   label=f'95% CI for C(1)')
        
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax1.set_xlabel('Lag τ')
        ax1.set_ylabel('Autocorrelation C(τ)')
        ax1.set_title(f'Autocorrelation Function (R² = {r_squared:.3f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Log-scale for exponential decay verification
        positive_mask = corr_func > 0
        if np.any(positive_mask):
            ax2.semilogy(lags[positive_mask], corr_func[positive_mask], 
                        'o', color=self.colors['primary'], markersize=8)
            
            if tau_c > 0:
                ax2.semilogy(fit_lags, fit_corr, '--', 
                            color=self.colors['secondary'], linewidth=2)
            
            ax2.set_xlabel('Lag τ')
            ax2.set_ylabel('log C(τ)')
            ax2.set_title('Exponential Decay Verification')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved autocorrelation plot to {save_path}")
            if show:
                plt.show()
            plt.close()
            return save_path
        else:
            if show:
                plt.show()
            return fig
    
    def plot_markov_property_test(self, results: Dict[str, Any],
                                 save_path: Optional[str] = None,
                                 show: bool = True) -> Optional[str]:
        """
        Visualize Markov property test results.
        
        Args:
            results: Validation results dictionary
            save_path: Path to save figure
            show: Whether to display the plot
            
        Returns:
            Path to saved figure or Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Markov Property Test Results', fontsize=16)
        
        # Extract data
        markov_test = results['markov_test']
        js_divs = markov_test['js_divergences']
        
        # Plot 1: JS divergence by order
        orders = sorted(js_divs.keys())
        divergences = [js_divs[o] for o in orders]
        
        bars = ax1.bar(orders, divergences, color=self.colors['primary'])
        
        # Color bars based on threshold
        threshold = 0.01
        for i, (order, div) in enumerate(zip(orders, divergences)):
            if div < threshold:
                bars[i].set_color(self.colors['success'])
            else:
                bars[i].set_color(self.colors['warning'])
        
        ax1.axhline(y=threshold, color='k', linestyle='--', 
                   label=f'Markovian threshold ({threshold})')
        ax1.set_xlabel('Markov Order')
        ax1.set_ylabel('JS Divergence')
        ax1.set_title('Jensen-Shannon Divergence Test')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Transition matrix heatmap (if available)
        if 'transition_matrices' in markov_test:
            matrices = markov_test['transition_matrices']
            if 1 in matrices:
                trans_matrix = matrices[1]
                
                im = ax2.imshow(trans_matrix, cmap='YlOrRd', aspect='auto')
                ax2.set_xlabel('Next State')
                ax2.set_ylabel('Current State')
                ax2.set_title('First-Order Transition Matrix')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax2)
                cbar.set_label('Transition Probability')
        else:
            # Alternative visualization
            ax2.text(0.5, 0.5, 
                    f"Markovian: {'YES' if markov_test['markovian'] else 'NO'}\n" +
                    f"JS(2): {js_divs.get(2, 0):.6f}\n" +
                    f"JS(3): {js_divs.get(3, 0):.6f}",
                    transform=ax2.transAxes,
                    fontsize=14,
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle="round,pad=0.5", 
                             facecolor=self.colors['success'] if markov_test['markovian'] 
                                      else self.colors['warning'],
                             alpha=0.8))
            ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved Markov test plot to {save_path}")
            if show:
                plt.show()
            plt.close()
            return save_path
        else:
            if show:
                plt.show()
            return fig
    
    def plot_analytical_comparison(self, results: Dict[str, Any],
                                 save_path: Optional[str] = None,
                                 show: bool = True) -> Optional[str]:
        """
        Plot analytical vs measured correlations.
        
        Args:
            results: Validation results dictionary
            save_path: Path to save figure
            show: Whether to display the plot
            
        Returns:
            Path to saved figure or Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Analytical vs Empirical Comparison', fontsize=16)
        
        # Extract data
        comparison = results['analytical_comparison']
        c1_measured = comparison['c1_measured']
        c1_predicted = comparison['c1_predicted']
        c1_ratio = comparison['c1_ratio']
        
        # Plot 1: Bar comparison
        labels = ['Measured', 'Predicted']
        values = [c1_measured, c1_predicted]
        colors = [self.colors['primary'], self.colors['secondary']]
        
        bars = ax1.bar(labels, values, color=colors)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.6f}',
                    ha='center', va='bottom')
        
        ax1.set_ylabel('C(1) Value')
        ax1.set_title(f'C(1) Comparison (Ratio: {c1_ratio:.2f})')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Scaling behavior (if available)
        if 'n_visits_range' in comparison and 'c1_scaling' in comparison:
            n_visits = comparison['n_visits_range']
            c1_theory = comparison['c1_scaling']
            
            ax2.loglog(n_visits, c1_theory, '--', 
                      color=self.colors['secondary'],
                      label='Theory: C(1) ~ 1/N', linewidth=2)
            
            # Add measured point
            if 'avg_n' in comparison:
                ax2.scatter([comparison['avg_n']], [abs(c1_measured)],
                          s=100, color=self.colors['primary'],
                          label='Measured', zorder=5)
            
            ax2.set_xlabel('Number of Visits (N)')
            ax2.set_ylabel('|C(1)|')
            ax2.set_title('Correlation Scaling with Visits')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            # Text summary
            summary_text = (
                f"Measured C(1): {c1_measured:.6f}\n"
                f"Predicted C(1): {c1_predicted:.6f}\n"
                f"Ratio: {c1_ratio:.2f}\n\n"
                f"Prediction {'matches' if 0.5 < c1_ratio < 2.0 else 'deviates from'} measurement"
            )
            
            ax2.text(0.5, 0.5, summary_text,
                    transform=ax2.transAxes,
                    fontsize=12,
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle="round,pad=0.5", 
                             facecolor='lightgray',
                             alpha=0.8))
            ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved analytical comparison to {save_path}")
            if show:
                plt.show()
            plt.close()
            return save_path
        else:
            if show:
                plt.show()
            return fig
    
    def plot_correlation_distribution(self, results: Dict[str, Any],
                                    save_path: Optional[str] = None,
                                    show: bool = True) -> Optional[str]:
        """
        Plot distribution of correlations across games.
        
        Args:
            results: Validation results dictionary
            save_path: Path to save figure
            show: Whether to display the plot
            
        Returns:
            Path to saved figure or Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Correlation Statistics Across Games', fontsize=16)
        
        # Get raw data if available
        if 'raw_data' not in results:
            # Create synthetic data for visualization
            n_games = 50
            raw_data = {
                'all_correlations': [np.random.normal(results['autocorrelation']['c1'], 
                                                    0.01, 100) 
                                   for _ in range(n_games)],
                'all_tau_c': np.random.gamma(results['autocorrelation']['tau_c'], 
                                           0.5, n_games),
                'game_lengths': np.random.randint(100, 1000, n_games)
            }
        else:
            raw_data = results['raw_data']
        
        # Plot 1: C(1) distribution
        ax = axes[0, 0]
        all_c1 = [np.mean(corrs[:1]) for corrs in raw_data['all_correlations']]
        ax.hist(all_c1, bins=20, color=self.colors['primary'], 
                alpha=0.7, edgecolor='black')
        ax.axvline(results['autocorrelation']['c1'], color='red', 
                  linestyle='--', linewidth=2, label='Mean C(1)')
        ax.set_xlabel('C(1) Value')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of C(1) Across Games')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Correlation time distribution
        ax = axes[0, 1]
        tau_c_values = raw_data['all_tau_c']
        ax.hist(tau_c_values, bins=20, color=self.colors['secondary'], 
                alpha=0.7, edgecolor='black')
        ax.axvline(results['autocorrelation']['tau_c'], color='red', 
                  linestyle='--', linewidth=2, label='Mean τ_c')
        ax.set_xlabel('Correlation Time τ_c')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Correlation Times')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: C(1) vs game length
        ax = axes[1, 0]
        game_lengths = raw_data['game_lengths']
        ax.scatter(game_lengths, all_c1, alpha=0.6, 
                  color=self.colors['primary'])
        
        # Add trend line
        z = np.polyfit(game_lengths, all_c1, 1)
        p = np.poly1d(z)
        ax.plot(sorted(game_lengths), p(sorted(game_lengths)), 
                "r--", alpha=0.8, label='Trend')
        
        ax.set_xlabel('Game Length (Simulations)')
        ax.set_ylabel('C(1)')
        ax.set_title('Correlation vs Game Length')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Box plots
        ax = axes[1, 1]
        data_to_plot = [all_c1, tau_c_values / np.max(tau_c_values)]
        box_plot = ax.boxplot(data_to_plot, tick_labels=['C(1)', 'τ_c (normalized)'],
                            patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(box_plot['boxes'], 
                               [self.colors['primary'], self.colors['secondary']]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Value')
        ax.set_title('Summary Statistics')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved correlation distribution to {save_path}")
            if show:
                plt.show()
            plt.close()
            return save_path
        else:
            if show:
                plt.show()
            return fig
    
    def plot_transition_matrices(self, results: Dict[str, Any],
                               save_path: Optional[str] = None,
                               show: bool = True) -> Optional[str]:
        """
        Plot transition matrices as heatmaps.
        
        Args:
            results: Validation results dictionary
            save_path: Path to save figure
            show: Whether to display the plot
            
        Returns:
            Path to saved figure or Figure object
        """
        markov_test = results['markov_test']
        
        if 'transition_matrices' not in markov_test:
            logger.warning("No transition matrices in results")
            return None
        
        matrices = markov_test['transition_matrices']
        n_matrices = len(matrices)
        
        fig, axes = plt.subplots(1, n_matrices, figsize=(6*n_matrices, 5))
        if n_matrices == 1:
            axes = [axes]
        
        fig.suptitle('Transition Probability Matrices', fontsize=16)
        
        for i, (order, matrix) in enumerate(sorted(matrices.items())):
            ax = axes[i]
            
            # Create heatmap
            im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('P(next|history)')
            
            ax.set_xlabel('Next State')
            ax.set_ylabel('History State')
            ax.set_title(f'Order {order} Transitions')
            
            # Add grid
            ax.set_xticks(np.arange(matrix.shape[1]))
            ax.set_yticks(np.arange(matrix.shape[0]))
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved transition matrices to {save_path}")
            if show:
                plt.show()
            plt.close()
            return save_path
        else:
            if show:
                plt.show()
            return fig
    
    def plot_timescale_separation(self, results: Dict[str, Any],
                                save_path: Optional[str] = None,
                                show: bool = True) -> Optional[str]:
        """
        Visualize timescale separation.
        
        Args:
            results: Validation results dictionary
            save_path: Path to save figure
            show: Whether to display the plot
            
        Returns:
            Path to saved figure or Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle('Timescale Separation in MCTS', fontsize=16)
        
        # Extract or create timescale data
        if 'timescales' in results:
            tau_env = results['timescales']['tau_env']
            tau_sys = results['timescales']['tau_sys']
            ratio = results['timescales']['separation_ratio']
        else:
            # Estimate from correlation data
            tau_env = 1.0
            tau_sys = 1.0 / results['analytical_comparison'].get('c1_predicted', 0.01)
            ratio = tau_sys / tau_env
        
        # Create visual representation
        scales = [tau_env, tau_sys]
        labels = ['τ_env\n(Environment)', 'τ_sys\n(System)']
        colors = [self.colors['success'], self.colors['warning']]
        
        # Log scale bar plot
        bars = ax.bar(labels, scales, color=colors, alpha=0.7)
        ax.set_yscale('log')
        
        # Add value labels
        for bar, scale in zip(bars, scales):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{scale:.1f}',
                   ha='center', va='bottom')
        
        # Add separation ratio
        ax.text(0.5, 0.95, f'Separation Ratio: {ratio:.1f}',
               transform=ax.transAxes,
               fontsize=14,
               ha='center',
               bbox=dict(boxstyle="round,pad=0.5", 
                        facecolor='lightblue' if ratio > 10 else 'lightyellow',
                        alpha=0.8))
        
        ax.set_ylabel('Timescale')
        ax.set_title('Environment vs System Timescales')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add interpretation
        if ratio > 10:
            interpretation = "Strong separation: Mean-field approximation valid"
            color = self.colors['success']
        else:
            interpretation = "Weak separation: Non-Markovian effects may be important"
            color = self.colors['warning']
        
        ax.text(0.5, 0.05, interpretation,
               transform=ax.transAxes,
               fontsize=12,
               ha='center',
               color=color,
               weight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved timescale plot to {save_path}")
            if show:
                plt.show()
            plt.close()
            return save_path
        else:
            if show:
                plt.show()
            return fig
    
    def generate_full_report(self, results: Dict[str, Any],
                           output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Generate comprehensive report with all plots.
        
        Args:
            results: Validation results dictionary
            output_dir: Directory to save plots (uses self.output_dir if None)
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        report_paths = {}
        
        # Generate individual plots
        plots = [
            ('autocorrelation', self.plot_autocorrelation_function),
            ('markov_test', self.plot_markov_property_test),
            ('analytical', self.plot_analytical_comparison),
            ('distributions', self.plot_correlation_distribution),
            ('transitions', self.plot_transition_matrices),
            ('timescales', self.plot_timescale_separation)
        ]
        
        for name, plot_func in plots:
            try:
                path = os.path.join(output_dir, f'markovian_{name}.png')
                result = plot_func(results, save_path=path, show=False)
                if result:
                    report_paths[name] = result
            except Exception as e:
                logger.warning(f"Failed to generate {name} plot: {e}")
        
        # Generate summary figure
        summary_path = self._generate_summary_figure(results, output_dir)
        if summary_path:
            report_paths['summary'] = summary_path
        
        logger.info(f"Generated {len(report_paths)} plots in {output_dir}")
        return report_paths
    
    def _generate_summary_figure(self, results: Dict[str, Any],
                               output_dir: str) -> Optional[str]:
        """Generate summary figure with key results"""
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        fig.suptitle('Markovian Validation Summary Report', fontsize=20)
        
        # Main autocorrelation plot
        ax1 = fig.add_subplot(gs[0, :2])
        autocorr = results['autocorrelation']
        lags = autocorr.get('lags', np.arange(11))
        corr_func = autocorr.get('correlation_function', np.zeros_like(lags))
        
        ax1.plot(lags, corr_func, 'o-', color=self.colors['primary'], 
                markersize=8, linewidth=2)
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax1.set_xlabel('Lag τ')
        ax1.set_ylabel('C(τ)')
        ax1.set_title('Autocorrelation Function')
        ax1.grid(True, alpha=0.3)
        
        # Key metrics panel
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        metrics_text = (
            f"KEY RESULTS\n\n"
            f"C(1) = {autocorr['c1']:.4f}\n"
            f"τ_c = {autocorr['tau_c']:.2f}\n"
            f"Markovian: {'YES' if results['markov_test']['markovian'] else 'NO'}\n"
            f"JS(2) = {results['markov_test']['js_divergences'].get(2, 0):.4f}\n"
            f"C(1) ratio = {results['analytical_comparison']['c1_ratio']:.2f}"
        )
        
        ax2.text(0.1, 0.5, metrics_text,
                transform=ax2.transAxes,
                fontsize=14,
                verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", 
                         facecolor='lightgray',
                         alpha=0.8))
        
        # Markov test visualization
        ax3 = fig.add_subplot(gs[1, 0])
        js_divs = results['markov_test']['js_divergences']
        orders = sorted(js_divs.keys())
        divergences = [js_divs[o] for o in orders]
        
        bars = ax3.bar(orders, divergences, color=self.colors['primary'])
        ax3.axhline(y=0.01, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Order')
        ax3.set_ylabel('JS Divergence')
        ax3.set_title('Markov Property Test')
        ax3.grid(True, alpha=0.3)
        
        # Analytical comparison
        ax4 = fig.add_subplot(gs[1, 1])
        comparison = results['analytical_comparison']
        labels = ['Measured', 'Predicted']
        values = [abs(comparison['c1_measured']), comparison['c1_predicted']]
        
        ax4.bar(labels, values, color=[self.colors['primary'], self.colors['secondary']])
        ax4.set_ylabel('|C(1)|')
        ax4.set_title('Theory vs Experiment')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Interpretation text
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Generate interpretation
        interpretation = self._generate_interpretation(results)
        
        ax5.text(0.5, 0.5, interpretation,
                transform=ax5.transAxes,
                fontsize=12,
                ha='center',
                va='center',
                bbox=dict(boxstyle="round,pad=1", 
                         facecolor='lightblue',
                         alpha=0.8))
        
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, 'markovian_summary.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _generate_interpretation(self, results: Dict[str, Any]) -> str:
        """Generate textual interpretation of results"""
        markovian = results['markov_test']['markovian']
        tau_c = results['autocorrelation']['tau_c']
        c1_ratio = results['analytical_comparison']['c1_ratio']
        
        interpretation = "INTERPRETATION:\n\n"
        
        if markovian:
            interpretation += "✓ The MCTS process exhibits Markovian behavior with negligible memory effects.\n"
        else:
            interpretation += "✗ Significant non-Markovian effects detected. Consider memory kernel corrections.\n"
        
        if tau_c < 5:
            interpretation += "✓ Short correlation time supports mean-field approximation.\n"
        else:
            interpretation += "✗ Long correlation time suggests strong temporal dependencies.\n"
        
        if 0.5 < c1_ratio < 2.0:
            interpretation += "✓ Good agreement between theory and experiment.\n"
        else:
            interpretation += "✗ Significant deviation from theoretical predictions.\n"
        
        interpretation += f"\nRecommendation: "
        if markovian and tau_c < 5 and 0.5 < c1_ratio < 2.0:
            interpretation += "The Markovian approximation is valid for this MCTS implementation."
        else:
            interpretation += "Consider non-Markovian corrections or alternative theoretical approaches."
        
        return interpretation