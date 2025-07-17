"""
Honest plotting functions that clearly distinguish between theoretical 
expectations and actual measurements from MCTS analysis.

This module replaces synthetic data visualizations with honest plots
that either show measured data or clearly label theoretical references.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from pathlib import Path


class HonestPlotter:
    """Creates honest visualizations distinguishing theory from measurements"""
    
    def __init__(self, results: Dict[str, Any], output_dir: Path):
        self.results = results
        self.output_dir = output_dir
        
    def add_measurement_label(self, ax):
        """Add clear label for measured data"""
        ax.text(0.05, 0.95, 'MEASURED', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
               fontsize=10, weight='bold', verticalalignment='top')
    
    def add_theory_label(self, ax):
        """Add clear label for theoretical curves"""
        ax.text(0.05, 0.95, 'THEORETICAL', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
               fontsize=10, weight='bold', verticalalignment='top')
    
    def add_no_data_label(self, ax, reason='Random Evaluator'):
        """Add label when no valid data exists"""
        ax.text(0.5, 0.5, f'NO MEASUREMENTS\n({reason})', 
               transform=ax.transAxes, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
               fontsize=12, weight='bold')
    
    def plot_temperature_extraction(self, ax):
        """Plot temperature extraction results"""
        temp_data = self.results.get('authentic_measurements', {}).get('temperatures', [])
        
        if temp_data and len(temp_data) > 0:
            # Plot measured temperatures
            temps = [t['temperature'] for t in temp_data if not np.isnan(t['temperature'])]
            errors = [t['error'] for t in temp_data if not np.isnan(t['temperature'])]
            
            if temps:
                ax.hist(temps, bins=min(30, len(temps)//2), alpha=0.7, 
                       edgecolor='black', color='blue')
                ax.axvline(np.mean(temps), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(temps):.3f}')
                ax.set_xlabel('Temperature (measured from π(a) ∝ exp(βQ(a)))')
                ax.set_ylabel('Count')
                ax.set_title('Authentic Temperature Measurements')
                ax.legend()
                self.add_measurement_label(ax)
            else:
                ax.text(0.5, 0.5, 'All temperature fits failed\n(Poor R² or random Q-values)', 
                       transform=ax.transAxes, ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7),
                       fontsize=12, weight='bold')
                ax.set_title('Temperature Extraction Failed')
        else:
            self.add_no_data_label(ax, 'No temperature data')
            ax.set_title('Temperature Measurements')
    
    def plot_temperature_scaling(self, ax):
        """Plot temperature vs visits scaling"""
        temp_data = self.results.get('authentic_measurements', {}).get('temperatures', [])
        
        if temp_data and len(temp_data) > 0:
            # Extract valid temperature measurements
            valid_data = [(t['temperature'], t['n_visits']) 
                         for t in temp_data 
                         if not np.isnan(t['temperature']) and 'n_visits' in t]
            
            if len(valid_data) > 3:
                temps, visits = zip(*valid_data)
                ax.scatter(visits, temps, alpha=0.6, s=30)
                
                # Try to fit scaling relation
                log_visits = np.log(visits)
                log_temps = np.log(temps)
                
                try:
                    p = np.polyfit(log_visits, log_temps, 1)
                    r_squared = np.corrcoef(log_visits, log_temps)[0, 1]**2
                    
                    if r_squared > 0.1:  # Only show if some correlation
                        x_fit = np.linspace(min(visits), max(visits), 100)
                        y_fit = np.exp(p[1]) * (x_fit ** p[0])
                        ax.plot(x_fit, y_fit, 'r--', alpha=0.5, 
                               label=f'T ∝ N^{p[0]:.3f} (R² = {r_squared:.3f})')
                    
                    ax.set_xlabel('N (visits)')
                    ax.set_ylabel('Temperature')
                    ax.set_title(f'Temperature Scaling (R² = {r_squared:.3f})')
                    ax.legend()
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    self.add_measurement_label(ax)
                except:
                    ax.set_xlabel('N (visits)')
                    ax.set_ylabel('Temperature')
                    ax.set_title('Temperature Scaling (Fit failed)')
                    self.add_measurement_label(ax)
            else:
                self.add_no_data_label(ax, 'Insufficient valid data')
                ax.set_title('Temperature Scaling')
        else:
            self.add_no_data_label(ax)
            ax.set_title('Temperature Scaling')
    
    def plot_phase_transitions(self, ax):
        """Plot phase transition measurements"""
        transition_data = self.results.get('statistical_mechanics', {}).get('phase_transitions', [])
        
        if transition_data and len(transition_data) > 0:
            # Plot measured phase transitions
            temps = [t['temperature'] for t in transition_data]
            order_params = [t['order_parameter'] for t in transition_data]
            
            ax.scatter(temps, order_params, c='blue', s=50, alpha=0.7, label='Measured')
            
            # Fit if enough points
            if len(temps) >= 5:
                sorted_data = sorted(zip(temps, order_params))
                temps_sorted = [t for t, _ in sorted_data]
                order_sorted = [o for _, o in sorted_data]
                ax.plot(temps_sorted, order_sorted, 'b-', alpha=0.5, label='Fitted')
            
            ax.set_xlabel('Temperature')
            ax.set_ylabel('Order Parameter')
            ax.set_title('Phase Transitions')
            ax.legend()
            ax.grid(True, alpha=0.3)
            self.add_measurement_label(ax)
        else:
            # Show theoretical reference but clearly labeled
            temps = np.linspace(0.5, 2.0, 100)
            Tc = 1.0
            beta = 0.125
            order_param = np.where(temps < Tc, np.abs(Tc - temps)**beta, 0)
            
            ax.plot(temps, order_param, 'k--', linewidth=1, alpha=0.5, label='2D Ising theory')
            ax.axvline(x=Tc, color='r', linestyle=':', alpha=0.5, label=f'Tc = {Tc}')
            
            ax.set_xlabel('Temperature')
            ax.set_ylabel('Order Parameter')
            ax.set_title('Phase Transitions\n(Theoretical Reference Only)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            self.add_theory_label(ax)
            self.add_no_data_label(ax)
    
    def plot_critical_exponents(self, ax):
        """Plot critical exponents"""
        exponents = self.results.get('statistical_mechanics', {}).get('critical_exponents', {})
        
        if exponents and not all(np.isnan(v) for v in exponents.values() if isinstance(v, (int, float))):
            # Plot measured exponents vs theoretical values
            theoretical = {
                '2D Ising': {'β/ν': 0.125, 'γ/ν': 1.75},
                '3D Ising': {'β/ν': 0.518, 'γ/ν': 1.963},
                'Mean Field': {'β/ν': 0.5, 'γ/ν': 1.0}
            }
            
            classes = list(theoretical.keys())
            if 'beta_over_nu' in exponents and 'gamma_over_nu' in exponents:
                classes.append('MCTS (measured)')
                theoretical['MCTS (measured)'] = {
                    'β/ν': exponents['beta_over_nu'],
                    'γ/ν': exponents['gamma_over_nu']
                }
            
            x = np.arange(len(classes))
            beta_vals = [theoretical[c]['β/ν'] for c in classes]
            gamma_vals = [theoretical[c]['γ/ν'] for c in classes]
            
            width = 0.35
            ax.bar(x - width/2, beta_vals, width, label='β/ν', alpha=0.7)
            ax.bar(x + width/2, gamma_vals, width, label='γ/ν', alpha=0.7)
            
            ax.set_xlabel('Universality Class')
            ax.set_ylabel('Critical Exponent Value')
            ax.set_title('Critical Exponents')
            ax.set_xticks(x)
            ax.set_xticklabels(classes, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # Mark measured values
            if 'MCTS (measured)' in classes:
                self.add_measurement_label(ax)
            else:
                self.add_theory_label(ax)
        else:
            ax.text(0.5, 0.5, 'Critical exponents: NaN\n(Need multiple system sizes)', 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7),
                   fontsize=12, weight='bold')
            ax.set_title('Critical Exponents')
    
    def plot_rg_flow(self, ax):
        """Plot RG flow if measured"""
        rg_data = self.results.get('rg_flow', {})
        
        if rg_data and 'flow_trajectories' in rg_data:
            # Plot measured RG flow
            trajectories = rg_data['flow_trajectories']
            
            for i, traj in enumerate(trajectories[:5]):  # Show first 5
                scales = [p['scale'] for p in traj]
                q_means = [p['q_mean'] for p in traj]
                
                ax.plot(scales, q_means, alpha=0.6, label=f'Trajectory {i+1}')
                
                # Add flow arrows
                for j in range(len(scales)-1):
                    ax.annotate('', xy=(scales[j+1], q_means[j+1]), 
                               xytext=(scales[j], q_means[j]),
                               arrowprops=dict(arrowstyle='->', alpha=0.3))
            
            ax.set_xlabel('RG Scale log₂(k)')
            ax.set_ylabel('Mean Q-value')
            ax.set_title('RG Flow Trajectories')
            ax.legend()
            ax.grid(True, alpha=0.3)
            self.add_measurement_label(ax)
        else:
            self.add_no_data_label(ax, 'No RG flow data')
            ax.set_title('RG Flow Analysis')
    
    def plot_quantum_darwinism(self, ax):
        """Plot quantum darwinism redundancy"""
        qd_data = self.results.get('quantum_darwinism', {})
        
        if qd_data and 'redundancy_curve' in qd_data:
            # Plot measured redundancy
            curve = qd_data['redundancy_curve']
            fractions = [p['fraction'] for p in curve]
            redundancies = [p['redundancy'] for p in curve]
            
            ax.semilogx(fractions, redundancies, 'bo-', alpha=0.7)
            ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Full redundancy')
            
            ax.set_xlabel('Environment Fragment Size |E_f|')
            ax.set_ylabel('Information Redundancy')
            ax.set_title('Quantum Darwinism: Information Proliferation')
            ax.legend()
            ax.grid(True, alpha=0.3)
            self.add_measurement_label(ax)
        else:
            self.add_no_data_label(ax, 'No redundancy data')
            ax.set_title('Quantum Darwinism')
    
    def create_honest_summary_plot(self) -> str:
        """Create the main honest summary plot"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('MCTS Physics Analysis: Measurements vs Theory', fontsize=16)
        
        # Temperature measurements
        self.plot_temperature_extraction(axes[0, 0])
        self.plot_temperature_scaling(axes[0, 1])
        
        # Phase transitions and critical phenomena
        self.plot_phase_transitions(axes[0, 2])
        self.plot_critical_exponents(axes[1, 0])
        
        # RG flow
        self.plot_rg_flow(axes[1, 1])
        
        # Quantum phenomena
        self.plot_quantum_darwinism(axes[1, 2])
        
        # Information-to-work conversion
        self.plot_information_work_conversion(axes[2, 0])
        
        # Thermodynamic evolution
        self.plot_thermodynamic_evolution(axes[2, 1])
        
        # Measurement quality summary
        self.plot_measurement_quality(axes[2, 2])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'honest_physics_summary.png'
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return str(plot_path)
    
    def plot_information_work_conversion(self, ax):
        """Plot information-to-work conversion"""
        # This would need actual measurements from the system
        # For now, show that it's not implemented
        ax.text(0.5, 0.5, 'Information-Work Conversion\nNot yet implemented', 
               transform=ax.transAxes, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
               fontsize=12, weight='bold')
        ax.set_title('Information-Work Conversion')
    
    def plot_thermodynamic_evolution(self, ax):
        """Plot thermodynamic evolution"""
        # This would need actual measurements from the system
        # For now, show that it's not implemented
        ax.text(0.5, 0.5, 'Thermodynamic Evolution\nNot yet implemented', 
               transform=ax.transAxes, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
               fontsize=12, weight='bold')
        ax.set_title('Thermodynamic Evolution')
    
    def plot_measurement_quality(self, ax):
        """Plot measurement quality summary"""
        # Count valid measurements
        temp_data = self.results.get('authentic_measurements', {}).get('temperatures', [])
        valid_temps = sum(1 for t in temp_data if not np.isnan(t['temperature']))
        
        critical_points = self.results.get('summary', {}).get('n_critical_points', 0)
        
        # Create quality summary
        quality_metrics = {
            'Valid Temperatures': valid_temps,
            'Critical Points': critical_points,
            'Thermodynamic States': len(self.results.get('statistical_mechanics', {}).get('thermodynamics', [])),
            'Decoherence Trajectories': len(self.results.get('quantum_phenomena', {}).get('decoherence', [])),
            'Entanglement Measurements': len(self.results.get('quantum_phenomena', {}).get('entanglement', []))
        }
        
        metrics = list(quality_metrics.keys())
        values = list(quality_metrics.values())
        
        ax.barh(metrics, values, alpha=0.7)
        ax.set_xlabel('Number of Measurements')
        ax.set_title('MEASUREMENT QUALITY\n' + '='*20)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add total count
        total = sum(values)
        ax.text(0.95, 0.95, f'Total: {total}', transform=ax.transAxes,
               ha='right', va='top', fontsize=12, weight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))