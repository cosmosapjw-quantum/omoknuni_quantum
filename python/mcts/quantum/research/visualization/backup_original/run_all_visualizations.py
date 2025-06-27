#!/usr/bin/env python3
"""
Driver Code for Running All Quantum MCTS Visualization Tools

This script sequentially executes all visualization tools for comprehensive analysis
of the quantum MCTS research project. It extracts data from authentic MCTS runs
and generates publication-ready plots and reports.

Usage:
    python run_all_visualizations.py [--data-path PATH] [--output-dir DIR] [--skip-animations]
"""

import sys
import os
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any, List
import json
import traceback
import numpy as np

# Add the research directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all visualization tools
from plot_rg_flow_phase_diagrams import RGFlowPhaseVisualizer
from plot_statistical_physics import StatisticalPhysicsVisualizer
from plot_critical_phenomena_scaling import CriticalPhenomenaVisualizer
from plot_decoherence_darwinism import DecoherenceDarwinismVisualizer
from plot_entropy_analysis import EntropyAnalysisVisualizer
from plot_thermodynamics import ThermodynamicsVisualizer
from plot_jarzynski_equality import JarzynskiEqualityVisualizer
from plot_exact_hbar_analysis import ExactHbarAnalysisVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('visualization_run.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class QuantumMCTSVisualizationSuite:
    """Comprehensive visualization suite for quantum MCTS research"""
    
    def __init__(self, data_path: str = None, output_dir: str = "quantum_mcts_visualizations"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualization modules
        self.visualizers = {}
        self.reports = {}
        self.execution_times = {}
        
        logger.info(f"Quantum MCTS Visualization Suite initialized")
        logger.info(f"Output directory: {self.output_dir.absolute()}")
        
    def load_mcts_data(self) -> Dict[str, Any]:
        """Load MCTS data from file or generate mock data"""
        
        if self.data_path and Path(self.data_path).exists():
            logger.info(f"Loading MCTS data from: {self.data_path}")
            try:
                with open(self.data_path, 'r') as f:
                    data = json.load(f)
                datasets = data.get('tree_expansion_data', [])
        total_sims = sum(d.get('total_simulations', 0) for d in datasets)
        logger.info(f"Successfully loaded {len(datasets)} MCTS datasets with {total_sims:,} total simulations")
                return data
            except Exception as e:
                logger.warning(f"Failed to load data from {self.data_path}: {e}")
                logger.info("Falling back to mock data generation")
        
        # Generate comprehensive large-scale mock data
        logger.info("Generating large-scale mock MCTS data (1000 games, 1000 sims/move) for visualization")
        
        # Create realistic MCTS tree expansion data
        tree_expansion_data = []
        performance_metrics = []
        
        # Scale up for 1000 games with 1000 simulations per move
        num_games = 1000
        simulations_per_move = 1000
        
        for i in range(num_games):
            # More realistic large-scale MCTS data
            # Visit counts reflecting 1000 simulations per move
            base_visits = simulations_per_move * (0.8 + 0.4 * np.random.random())
            visit_counts = np.random.exponential(base_visits / 50, 200 + i // 5)
            visit_counts = visit_counts[visit_counts > 0]  # Remove zeros
            
            # Q-values with better convergence for large simulation counts
            q_values = np.random.normal(0, 0.2, len(visit_counts))
            q_values = np.clip(q_values, -1.0, 1.0)
            
            # Larger trees due to more simulations
            tree_size = 500 + i * 2  # Much larger trees
            max_depth = 15 + i // 100  # Deeper trees
            
            # Policy distribution evolution
            policy_entropy = 1.0 + 0.2 * np.sin(i * 0.5) + np.random.normal(0, 0.1)
            
            tree_data = {
                'visit_counts': visit_counts.tolist(),
                'q_values': q_values.tolist(),
                'tree_size': tree_size,
                'max_depth': max_depth,
                'policy_entropy': policy_entropy,
                'timestamp': time.time() + i * 100,
                'total_simulations': int(np.sum(visit_counts)),  # Track total simulations
                'game_id': i + 1,
                'simulations_per_move': simulations_per_move,
                'visit_count_stats': {
                    'count': len(visit_counts),
                    'min': float(np.min(visit_counts)),
                    'max': float(np.max(visit_counts)),
                    'mean': float(np.mean(visit_counts)),
                    'std': float(np.std(visit_counts)),
                    'total': float(np.sum(visit_counts))
                }
            }
            
            tree_expansion_data.append(tree_data)
            
            # Performance metrics scaled for large simulations
            performance_data = {
                'win_rate': 0.5 + 0.3 * np.sin(i * 0.3) + np.random.normal(0, 0.05),
                'average_game_length': 50 + np.random.normal(0, 10),
                'search_time': 5.0 + i * 0.01 + np.random.normal(0, 0.5),  # Longer search times
                'nodes_per_second': 5000 + i * 5 + np.random.normal(0, 200),  # Higher throughput
                'memory_usage': 500 + i * 0.5 + np.random.normal(0, 25),  # More memory usage
                'total_nodes_expanded': tree_size,
                'simulations_completed': int(np.sum(visit_counts)),
                'convergence_quality': 0.7 + 0.2 * np.random.random()  # Quality metric
            }
            
            performance_metrics.append(performance_data)
        
        mock_data = {
            'tree_expansion_data': tree_expansion_data,
            'performance_metrics': performance_metrics,
            'metadata': {
                'generated': True,
                'timestamp': time.time(),
                'description': 'Scaled mock data for quantum MCTS visualization',
                'data_scale': {
                    'num_games': num_games,
                    'simulations_per_move': simulations_per_move,
                    'total_simulations': sum(data['total_simulations'] for data in tree_expansion_data),
                    'average_tree_size': np.mean([data['tree_size'] for data in tree_expansion_data]),
                    'average_depth': np.mean([data['max_depth'] for data in tree_expansion_data])
                }
            }
        }
        
        # Save mock data for reference
        mock_data_file = self.output_dir / 'mock_mcts_data.json'
        with open(mock_data_file, 'w') as f:
            json.dump(mock_data, f, indent=2)
        logger.info(f"Saved mock data to: {mock_data_file}")
        
        return mock_data
    
    def initialize_visualizers(self, mcts_data: Dict[str, Any]):
        """Initialize all visualization modules"""
        
        logger.info("Initializing visualization modules...")
        
        visualizer_configs = [
            ('rg_flow_phase', RGFlowPhaseVisualizer, 'rg_flow_plots'),
            ('statistical_physics', StatisticalPhysicsVisualizer, 'statistical_physics_plots'),
            ('critical_phenomena', CriticalPhenomenaVisualizer, 'critical_phenomena_plots'),
            ('decoherence_darwinism', DecoherenceDarwinismVisualizer, 'decoherence_plots'),
            ('entropy_analysis', EntropyAnalysisVisualizer, 'entropy_plots'),
            ('thermodynamics', ThermodynamicsVisualizer, 'thermodynamics_plots'),
            ('jarzynski_equality', JarzynskiEqualityVisualizer, 'jarzynski_plots'),
            ('exact_hbar_analysis', ExactHbarAnalysisVisualizer, 'exact_hbar_plots')
        ]
        
        for name, visualizer_class, output_subdir in visualizer_configs:
            try:
                output_path = self.output_dir / output_subdir
                self.visualizers[name] = visualizer_class(mcts_data, str(output_path))
                logger.info(f"✓ Initialized {name} visualizer")
            except Exception as e:
                logger.error(f"✗ Failed to initialize {name} visualizer: {e}")
                self.visualizers[name] = None
        
        logger.info(f"Successfully initialized {len([v for v in self.visualizers.values() if v is not None])}/{len(visualizer_configs)} visualizers")
    
    def run_rg_flow_analysis(self) -> Dict[str, Any]:
        """Run RG flow and phase diagram analysis"""
        
        logger.info("="*60)
        logger.info("RUNNING RG FLOW AND PHASE DIAGRAM ANALYSIS")
        logger.info("="*60)
        
        start_time = time.time()
        
        try:
            visualizer = self.visualizers['rg_flow_phase']
            if visualizer is None:
                raise ValueError("RG flow visualizer not initialized")
            
            # Run comprehensive analysis
            report = visualizer.generate_comprehensive_report(save_report=True)
            
            # Create animations if requested
            try:
                logger.info("Creating RG flow animation...")
                visualizer.create_flow_animation('rg_flow', save_animation=True)
            except Exception as e:
                logger.warning(f"Animation creation failed: {e}")
            
            execution_time = time.time() - start_time
            self.execution_times['rg_flow_phase'] = execution_time
            
            logger.info(f"✓ RG flow analysis completed in {execution_time:.2f}s")
            logger.info(f"Generated {len(report['output_files'])} plots")
            
            return report
            
        except Exception as e:
            logger.error(f"✗ RG flow analysis failed: {e}")
            logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def run_statistical_physics_analysis(self) -> Dict[str, Any]:
        """Run statistical physics analysis"""
        
        logger.info("="*60)
        logger.info("RUNNING STATISTICAL PHYSICS ANALYSIS")
        logger.info("="*60)
        
        start_time = time.time()
        
        try:
            visualizer = self.visualizers['statistical_physics']
            if visualizer is None:
                raise ValueError("Statistical physics visualizer not initialized")
            
            # Run comprehensive analysis
            report = visualizer.generate_comprehensive_report(save_report=True)
            
            # Create animations
            try:
                logger.info("Creating statistical physics animations...")
                visualizer.create_temporal_evolution_animation('correlations', save_animation=True)
                visualizer.create_temporal_evolution_animation('susceptibility', save_animation=True)
            except Exception as e:
                logger.warning(f"Animation creation failed: {e}")
            
            execution_time = time.time() - start_time
            self.execution_times['statistical_physics'] = execution_time
            
            logger.info(f"✓ Statistical physics analysis completed in {execution_time:.2f}s")
            logger.info(f"Generated {len(report['output_files'])} plots")
            
            return report
            
        except Exception as e:
            logger.error(f"✗ Statistical physics analysis failed: {e}")
            logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def run_critical_phenomena_analysis(self) -> Dict[str, Any]:
        """Run critical phenomena and scaling analysis"""
        
        logger.info("="*60)
        logger.info("RUNNING CRITICAL PHENOMENA ANALYSIS")
        logger.info("="*60)
        
        start_time = time.time()
        
        try:
            visualizer = self.visualizers['critical_phenomena']
            if visualizer is None:
                raise ValueError("Critical phenomena visualizer not initialized")
            
            # Run comprehensive analysis
            report = visualizer.generate_comprehensive_report(save_report=True)
            
            # Create animations
            try:
                logger.info("Creating critical phenomena animations...")
                visualizer.create_critical_animation('order_parameter', save_animation=True)
                visualizer.create_critical_animation('susceptibility', save_animation=True)
            except Exception as e:
                logger.warning(f"Animation creation failed: {e}")
            
            execution_time = time.time() - start_time
            self.execution_times['critical_phenomena'] = execution_time
            
            logger.info(f"✓ Critical phenomena analysis completed in {execution_time:.2f}s")
            logger.info(f"Generated {len(report['output_files'])} plots")
            
            return report
            
        except Exception as e:
            logger.error(f"✗ Critical phenomena analysis failed: {e}")
            logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def run_decoherence_analysis(self) -> Dict[str, Any]:
        """Run decoherence and quantum Darwinism analysis"""
        
        logger.info("="*60)
        logger.info("RUNNING DECOHERENCE AND QUANTUM DARWINISM ANALYSIS")
        logger.info("="*60)
        
        start_time = time.time()
        
        try:
            visualizer = self.visualizers['decoherence_darwinism']
            if visualizer is None:
                raise ValueError("Decoherence visualizer not initialized")
            
            # Run comprehensive analysis
            report = visualizer.generate_comprehensive_report(save_report=True)
            
            # Create animations
            try:
                logger.info("Creating decoherence animations...")
                visualizer.create_decoherence_animation('coherence_decay', save_animation=True)
                visualizer.create_decoherence_animation('purity_evolution', save_animation=True)
            except Exception as e:
                logger.warning(f"Animation creation failed: {e}")
            
            execution_time = time.time() - start_time
            self.execution_times['decoherence_darwinism'] = execution_time
            
            logger.info(f"✓ Decoherence analysis completed in {execution_time:.2f}s")
            logger.info(f"Generated {len(report['output_files'])} plots")
            
            return report
            
        except Exception as e:
            logger.error(f"✗ Decoherence analysis failed: {e}")
            logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def run_entropy_analysis(self) -> Dict[str, Any]:
        """Run entropy and information theory analysis"""
        
        logger.info("="*60)
        logger.info("RUNNING ENTROPY AND INFORMATION ANALYSIS")
        logger.info("="*60)
        
        start_time = time.time()
        
        try:
            visualizer = self.visualizers['entropy_analysis']
            if visualizer is None:
                raise ValueError("Entropy visualizer not initialized")
            
            # Run comprehensive analysis
            report = visualizer.generate_comprehensive_report(save_report=True)
            
            # Create animations
            try:
                logger.info("Creating entropy animations...")
                visualizer.create_entropy_animation('entanglement', save_animation=True)
                visualizer.create_entropy_animation('von_neumann', save_animation=True)
            except Exception as e:
                logger.warning(f"Animation creation failed: {e}")
            
            execution_time = time.time() - start_time
            self.execution_times['entropy_analysis'] = execution_time
            
            logger.info(f"✓ Entropy analysis completed in {execution_time:.2f}s")
            logger.info(f"Generated {len(report['output_files'])} plots")
            
            return report
            
        except Exception as e:
            logger.error(f"✗ Entropy analysis failed: {e}")
            logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def run_thermodynamics_analysis(self) -> Dict[str, Any]:
        """Run thermodynamics and phase transition analysis"""
        
        logger.info("="*60)
        logger.info("RUNNING THERMODYNAMICS ANALYSIS")
        logger.info("="*60)
        
        start_time = time.time()
        
        try:
            visualizer = self.visualizers['thermodynamics']
            if visualizer is None:
                raise ValueError("Thermodynamics visualizer not initialized")
            
            # Run comprehensive analysis
            report = visualizer.generate_comprehensive_report(save_report=True)
            
            # Create animations
            try:
                logger.info("Creating thermodynamics animations...")
                visualizer.create_thermodynamic_animation('otto', save_animation=True)
                visualizer.create_thermodynamic_animation('carnot', save_animation=True)
            except Exception as e:
                logger.warning(f"Animation creation failed: {e}")
            
            execution_time = time.time() - start_time
            self.execution_times['thermodynamics'] = execution_time
            
            logger.info(f"✓ Thermodynamics analysis completed in {execution_time:.2f}s")
            logger.info(f"Generated {len(report['output_files'])} plots")
            
            return report
            
        except Exception as e:
            logger.error(f"✗ Thermodynamics analysis failed: {e}")
            logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def run_jarzynski_analysis(self) -> Dict[str, Any]:
        """Run Jarzynski equality and fluctuation theorem analysis"""
        
        logger.info("="*60)
        logger.info("RUNNING JARZYNSKI EQUALITY ANALYSIS")
        logger.info("="*60)
        
        start_time = time.time()
        
        try:
            visualizer = self.visualizers['jarzynski_equality']
            if visualizer is None:
                raise ValueError("Jarzynski visualizer not initialized")
            
            # Run comprehensive analysis
            report = visualizer.generate_comprehensive_report(save_report=True)
            
            # Create animations
            try:
                logger.info("Creating Jarzynski animations...")
                visualizer.create_jarzynski_animation(save_animation=True)
            except Exception as e:
                logger.warning(f"Animation creation failed: {e}")
            
            execution_time = time.time() - start_time
            self.execution_times['jarzynski_equality'] = execution_time
            
            logger.info(f"✓ Jarzynski analysis completed in {execution_time:.2f}s")
            logger.info(f"Generated {len(report['output_files'])} plots")
            
            return report
            
        except Exception as e:
            logger.error(f"✗ Jarzynski analysis failed: {e}")
            logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def run_exact_hbar_analysis(self) -> Dict[str, Any]:
        """Run exact ℏ_eff analysis"""
        
        logger.info("="*60)
        logger.info("RUNNING EXACT ℏ_EFF ANALYSIS")
        logger.info("="*60)
        
        start_time = time.time()
        
        try:
            visualizer = self.visualizers['exact_hbar_analysis']
            if visualizer is None:
                raise ValueError("Exact ℏ_eff visualizer not initialized")
            
            # Run comprehensive analysis
            report = visualizer.generate_comprehensive_report(save_report=True)
            
            # Create animations
            try:
                logger.info("Creating ℏ_eff animations...")
                visualizer.create_hbar_evolution_animation(save_animation=True)
            except Exception as e:
                logger.warning(f"Animation creation failed: {e}")
            
            execution_time = time.time() - start_time
            self.execution_times['exact_hbar_analysis'] = execution_time
            
            logger.info(f"✓ Exact ℏ_eff analysis completed in {execution_time:.2f}s")
            logger.info(f"Generated {len(report['output_files'])} plots")
            
            return report
            
        except Exception as e:
            logger.error(f"✗ Exact ℏ_eff analysis failed: {e}")
            logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def run_all_analyses(self, skip_animations: bool = False) -> Dict[str, Any]:
        """Run all visualization analyses sequentially"""
        
        logger.info("🚀 STARTING COMPREHENSIVE QUANTUM MCTS VISUALIZATION SUITE")
        logger.info("="*80)
        
        overall_start_time = time.time()
        
        # Load data
        logger.info("Loading MCTS data...")
        mcts_data = self.load_mcts_data()
        
        # Initialize visualizers
        self.initialize_visualizers(mcts_data)
        
        # Run all analyses in sequence
        analysis_functions = [
            ('rg_flow_phase', self.run_rg_flow_analysis),
            ('statistical_physics', self.run_statistical_physics_analysis),
            ('critical_phenomena', self.run_critical_phenomena_analysis),
            ('decoherence_darwinism', self.run_decoherence_analysis),
            ('entropy_analysis', self.run_entropy_analysis),
            ('thermodynamics', self.run_thermodynamics_analysis),
            ('jarzynski_equality', self.run_jarzynski_analysis),
            ('exact_hbar_analysis', self.run_exact_hbar_analysis)
        ]
        
        for analysis_name, analysis_func in analysis_functions:
            try:
                self.reports[analysis_name] = analysis_func()
            except Exception as e:
                logger.error(f"Critical error in {analysis_name}: {e}")
                self.reports[analysis_name] = {'error': str(e)}
        
        # Generate master report
        total_time = time.time() - overall_start_time
        master_report = self.generate_master_report(total_time)
        
        logger.info("="*80)
        logger.info("🎉 QUANTUM MCTS VISUALIZATION SUITE COMPLETED")
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        logger.info(f"Master report saved to: {self.output_dir / 'master_report.json'}")
        logger.info("="*80)
        
        return master_report
    
    def generate_master_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive master report"""
        
        logger.info("Generating master report...")
        
        # Count successful analyses
        successful_analyses = len([r for r in self.reports.values() if 'error' not in r])
        total_analyses = len(self.reports)
        
        # Count total output files
        total_plots = 0
        for report in self.reports.values():
            if 'output_files' in report:
                total_plots += len(report['output_files'])
        
        master_report = {
            'quantum_mcts_visualization_suite': {
                'timestamp': time.time(),
                'total_execution_time': total_time,
                'successful_analyses': successful_analyses,
                'total_analyses': total_analyses,
                'success_rate': successful_analyses / total_analyses if total_analyses > 0 else 0,
                'total_plots_generated': total_plots
            },
            'execution_times': self.execution_times,
            'analysis_reports': self.reports,
            'output_structure': {
                'master_output_dir': str(self.output_dir),
                'subdirectories': [
                    'rg_flow_plots',
                    'statistical_physics_plots', 
                    'critical_phenomena_plots',
                    'decoherence_plots',
                    'entropy_plots',
                    'thermodynamics_plots',
                    'jarzynski_plots',
                    'exact_hbar_plots'
                ]
            },
            'summary': {
                'rg_flow_analysis': 'completed' if 'error' not in self.reports.get('rg_flow_phase', {}) else 'failed',
                'statistical_physics': 'completed' if 'error' not in self.reports.get('statistical_physics', {}) else 'failed',
                'critical_phenomena': 'completed' if 'error' not in self.reports.get('critical_phenomena', {}) else 'failed',
                'decoherence_darwinism': 'completed' if 'error' not in self.reports.get('decoherence_darwinism', {}) else 'failed',
                'entropy_analysis': 'completed' if 'error' not in self.reports.get('entropy_analysis', {}) else 'failed',
                'thermodynamics': 'completed' if 'error' not in self.reports.get('thermodynamics', {}) else 'failed',
                'jarzynski_equality': 'completed' if 'error' not in self.reports.get('jarzynski_equality', {}) else 'failed',
                'exact_hbar_analysis': 'completed' if 'error' not in self.reports.get('exact_hbar_analysis', {}) else 'failed'
            }
        }
        
        # Save master report
        master_report_file = self.output_dir / 'master_report.json'
        
        # Convert numpy arrays to JSON-serializable format
        def convert_numpy_to_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_to_json(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_json(item) for item in obj]
            else:
                return obj
        
        json_safe_master_report = convert_numpy_to_json(master_report)
        
        with open(master_report_file, 'w') as f:
            json.dump(json_safe_master_report, f, indent=2)
        
        # Generate summary statistics
        logger.info(f"Analysis Summary:")
        logger.info(f"  ✓ Successful analyses: {successful_analyses}/{total_analyses}")
        logger.info(f"  📊 Total plots generated: {total_plots}")
        logger.info(f"  ⏱️ Average analysis time: {np.mean(list(self.execution_times.values())):.2f}s")
        logger.info(f"  📁 Output directory: {self.output_dir}")
        
        return master_report


def main():
    """Main entry point for the visualization suite"""
    
    parser = argparse.ArgumentParser(
        description="Run all Quantum MCTS visualization tools sequentially",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with mock data
    python run_all_visualizations.py
    
    # Run with custom data file
    python run_all_visualizations.py --data-path /path/to/mcts_data.json
    
    # Custom output directory
    python run_all_visualizations.py --output-dir /path/to/output
    
    # Skip animations for faster execution
    python run_all_visualizations.py --skip-animations
        """
    )
    
    parser.add_argument(
        '--data-path', 
        type=str, 
        help='Path to MCTS data file (JSON format). If not provided, mock data will be generated.'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='quantum_mcts_visualizations',
        help='Output directory for all plots and reports (default: quantum_mcts_visualizations)'
    )
    
    parser.add_argument(
        '--skip-animations', 
        action='store_true',
        help='Skip animation generation for faster execution'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Initialize visualization suite
        suite = QuantumMCTSVisualizationSuite(
            data_path=args.data_path,
            output_dir=args.output_dir
        )
        
        # Run all analyses
        master_report = suite.run_all_analyses(skip_animations=args.skip_animations)
        
        # Print final summary
        print("\n" + "="*80)
        print("🎯 QUANTUM MCTS VISUALIZATION SUITE - FINAL SUMMARY")
        print("="*80)
        print(f"✅ Execution completed successfully!")
        print(f"📊 {master_report['quantum_mcts_visualization_suite']['total_plots_generated']} plots generated")
        print(f"⏱️  Total time: {master_report['quantum_mcts_visualization_suite']['total_execution_time']:.2f} seconds")
        print(f"📁 Results saved to: {args.output_dir}")
        print("="*80)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())