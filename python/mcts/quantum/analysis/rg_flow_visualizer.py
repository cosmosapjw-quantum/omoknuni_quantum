"""
RG Flow visualization for MCTS analysis.

Creates visual representations of renormalization group flow
through the MCTS tree, showing how information coarse-grains
from leaves to root.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RGFlowVisualizer:
    """Creates visualizations of RG flow in MCTS"""
    
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        
    def plot_rg_flow_trajectories(self, flow_data: Dict[str, Any], 
                                 save_path: Optional[str] = None) -> str:
        """
        Plot RG flow trajectories in parameter space.
        
        Args:
            flow_data: RG flow analysis results
            save_path: Optional path to save figure
            
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Renormalization Group Flow in MCTS', fontsize=16)
        
        # Plot 1: Q-value evolution along RG flow
        ax = axes[0, 0]
        trajectories = flow_data.get('trajectories', [])
        
        for i, traj in enumerate(trajectories[:10]):  # Show first 10
            flow_points = traj.get('flow_points', [])
            if flow_points:
                scales = [p['scale'] for p in flow_points]
                q_means = [p['q_mean'] for p in flow_points]
                ax.plot(scales, q_means, 'o-', alpha=0.6, label=f'Game {i+1}')
        
        ax.set_xlabel('RG Scale (tree depth)')
        ax.set_ylabel('Mean Q-value')
        ax.set_title('Q-value Evolution Under RG Flow')
        ax.grid(True, alpha=0.3)
        if len(trajectories) <= 5:
            ax.legend()
        
        # Plot 2: Variance flow (running coupling)
        ax = axes[0, 1]
        for i, traj in enumerate(trajectories[:10]):
            flow_points = traj.get('flow_points', [])
            if flow_points:
                scales = [p['scale'] for p in flow_points]
                variances = [p['q_variance'] for p in flow_points]
                ax.semilogy(scales, variances, 'o-', alpha=0.6)
        
        ax.set_xlabel('RG Scale (tree depth)')
        ax.set_ylabel('Q-value Variance (log scale)')
        ax.set_title('Running Coupling Evolution')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Beta function
        ax = axes[1, 0]
        ensemble = flow_data.get('ensemble_analysis', {})
        ensemble_flow = ensemble.get('ensemble_flow', [])
        
        if len(ensemble_flow) >= 2:
            # Calculate beta function from ensemble average
            scales = [p['scale'] for p in ensemble_flow]
            variances = [p['q_variance'] for p in ensemble_flow]
            
            beta_function = []
            for i in range(1, len(scales)):
                if scales[i] != scales[i-1]:
                    beta = (variances[i] - variances[i-1]) / (scales[i] - scales[i-1])
                    beta_function.append(beta)
                
            if beta_function:
                ax.plot(scales[1:], beta_function, 'ro-', markersize=8, linewidth=2)
                ax.axhline(0, color='k', linestyle='--', alpha=0.5)
                ax.set_xlabel('RG Scale')
                ax.set_ylabel('β(g) = dg/d(scale)')
                ax.set_title('RG Beta Function')
                ax.grid(True, alpha=0.3)
                
                # Mark fixed points
                fixed_points = flow_data.get('fixed_points', [])
                for fp in fixed_points:
                    ax.axvline(fp['scale'], color='red', linestyle=':', alpha=0.7)
                    ax.text(fp['scale'], ax.get_ylim()[1]*0.9, 
                           f"{fp['type']}", rotation=90, va='top')
        
        # Plot 4: Flow diagram in 2D parameter space
        ax = axes[1, 1]
        self._plot_2d_flow_diagram(ax, trajectories)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/rg_flow_analysis.png"
        
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return save_path
    
    def _plot_2d_flow_diagram(self, ax, trajectories: List[Dict[str, Any]]):
        """Create 2D flow diagram in (variance, entropy) space"""
        # Extract flow in 2D parameter space
        for traj in trajectories[:5]:  # Show first 5 clearly
            flow_points = traj.get('flow_points', [])
            if len(flow_points) >= 2:
                variances = [p['q_variance'] for p in flow_points]
                entropies = [p['visit_entropy'] for p in flow_points]
                
                # Plot trajectory
                ax.plot(variances, entropies, 'o-', alpha=0.5, markersize=4)
                
                # Add flow arrows
                for i in range(len(variances)-1):
                    dx = variances[i+1] - variances[i]
                    dy = entropies[i+1] - entropies[i]
                    
                    if abs(dx) + abs(dy) > 0.01:  # Skip tiny movements
                        arrow = FancyArrowPatch(
                            (variances[i], entropies[i]),
                            (variances[i] + 0.8*dx, entropies[i] + 0.8*dy),
                            arrowstyle='->', mutation_scale=15,
                            alpha=0.3, color='black'
                        )
                        ax.add_patch(arrow)
        
        ax.set_xlabel('Q-value Variance')
        ax.set_ylabel('Visit Entropy')
        ax.set_title('RG Flow in Parameter Space')
        ax.grid(True, alpha=0.3)
    
    def plot_fixed_point_analysis(self, flow_data: Dict[str, Any],
                                 save_path: Optional[str] = None) -> str:
        """
        Detailed analysis of RG fixed points.
        
        Args:
            flow_data: RG flow analysis results
            save_path: Optional path to save figure
            
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('RG Fixed Point Analysis', fontsize=16)
        
        # Plot 1: Fixed point distribution
        ax = axes[0]
        fixed_points = flow_data.get('fixed_points', [])
        
        if fixed_points:
            scales = [fp['mean_scale'] for fp in fixed_points]
            couplings = [fp['mean_coupling'] for fp in fixed_points]
            counts = [fp['count'] for fp in fixed_points]
            stabilities = [fp['stability'] for fp in fixed_points]
            
            # Size by count, color by stability
            scatter = ax.scatter(scales, couplings, s=np.array(counts)*50,
                               c=stabilities, cmap='RdYlBu', alpha=0.7,
                               edgecolors='black', linewidth=1)
            
            ax.set_xlabel('RG Scale')
            ax.set_ylabel('Coupling (Q-variance)')
            ax.set_title('Fixed Point Locations')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Stability (fraction stable)')
        
        # Plot 2: Universality analysis
        ax = axes[1]
        universality = flow_data.get('ensemble_analysis', {}).get('universality_measure', {})
        
        if universality and 'mean_scaling_exponent' in universality:
            # Show distribution of scaling exponents
            exponent = universality['mean_scaling_exponent']
            std = universality['std_scaling_exponent']
            n_traj = universality['n_valid_trajectories']
            
            if not np.isnan(exponent):
                # Create gaussian distribution visualization
                x = np.linspace(exponent - 3*std, exponent + 3*std, 100)
                y = np.exp(-(x - exponent)**2 / (2*std**2)) / (std * np.sqrt(2*np.pi))
                
                ax.fill_between(x, y, alpha=0.5, label=f'n={n_traj} trajectories')
                ax.axvline(exponent, color='red', linestyle='--', 
                          label=f'Mean = {exponent:.3f}')
                
                # Compare to known universality classes
                known_exponents = {
                    'Mean Field': -0.5,
                    '2D Ising': -1.0,
                    'Gaussian': 0.0
                }
                
                for name, value in known_exponents.items():
                    ax.axvline(value, color='gray', linestyle=':', alpha=0.5)
                    ax.text(value, ax.get_ylim()[1]*0.9, name, 
                           rotation=90, va='top', ha='right')
                
                ax.set_xlabel('Scaling Exponent')
                ax.set_ylabel('Probability Density')
                ax.set_title('Universality Class Analysis')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/rg_fixed_points.png"
        
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return save_path
    
    def plot_scale_dependent_observables(self, flow_data: Dict[str, Any],
                                       save_path: Optional[str] = None) -> str:
        """
        Plot how various observables change with RG scale.
        
        Args:
            flow_data: RG flow analysis results
            save_path: Optional path to save figure
            
        Returns:
            Path to saved figure
        """
        ensemble = flow_data.get('ensemble_analysis', {})
        ensemble_flow = ensemble.get('ensemble_flow', [])
        
        if not ensemble_flow:
            logger.warning("No ensemble flow data to plot")
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Scale-Dependent Observables', fontsize=16)
        
        scales = [p['scale'] for p in ensemble_flow]
        
        # Plot 1: Mean Q-value with error bars
        ax = axes[0, 0]
        q_means = [p['q_mean'] for p in ensemble_flow]
        q_stds = [p['q_mean_std'] for p in ensemble_flow]
        
        ax.errorbar(scales, q_means, yerr=q_stds, fmt='o-', capsize=5)
        ax.set_xlabel('RG Scale')
        ax.set_ylabel('Mean Q-value')
        ax.set_title('Q-value Renormalization')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Variance (running coupling)
        ax = axes[0, 1]
        variances = [p['q_variance'] for p in ensemble_flow]
        var_stds = [p['q_variance_std'] for p in ensemble_flow]
        
        ax.errorbar(scales, variances, yerr=var_stds, fmt='o-', capsize=5, color='red')
        ax.set_xlabel('RG Scale')
        ax.set_ylabel('Q-value Variance')
        ax.set_title('Running Coupling g(scale)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Entropy evolution
        ax = axes[1, 0]
        entropies = [p['entropy'] for p in ensemble_flow]
        ent_stds = [p['entropy_std'] for p in ensemble_flow]
        
        ax.errorbar(scales, entropies, yerr=ent_stds, fmt='o-', capsize=5, color='green')
        ax.set_xlabel('RG Scale')
        ax.set_ylabel('Visit Entropy')
        ax.set_title('Information Loss Under RG')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Participation ratio
        ax = axes[1, 1]
        participation = [p['participation_ratio'] for p in ensemble_flow]
        
        ax.plot(scales, participation, 'o-', color='purple')
        ax.set_xlabel('RG Scale')
        ax.set_ylabel('Participation Ratio')
        ax.set_title('Effective Degrees of Freedom')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/rg_scale_observables.png"
        
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return save_path