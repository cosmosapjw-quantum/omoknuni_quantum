"""
Run complete physics analysis on MCTS self-play data.

This script uses ALL validated physics modules:
- Statistical mechanics: Thermodynamics, Critical phenomena, FDT
- Quantum phenomena: Decoherence, Tunneling, Entanglement
- Authentic measurements (no predetermined formulas)
"""
import logging
import argparse
from pathlib import Path
# Always use absolute imports since we're being called from project root
import sys
from pathlib import Path

# Add the analysis directory to path
analysis_dir = Path(__file__).parent
sys.path.insert(0, str(analysis_dir))

from auto_generator import CompleteAutoDataGenerator, CompleteGeneratorConfig
from ensemble_analyzer_complete import CompleteEnsembleConfig
from cpuct_phase_transition_validator import CpuctPhaseConfig, run_cpuct_phase_validation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Run complete MCTS physics analysis using all modules'
    )
    parser.add_argument(
        '--preset', 
        choices=['quick', 'standard', 'comprehensive', 'deep', 'overnight'],
        default='standard',
        help='Analysis preset'
    )
    parser.add_argument(
        '--games',
        type=int,
        help='Override number of games to generate'
    )
    parser.add_argument(
        '--sims',
        type=int,
        help='Override simulations per game'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./complete_mcts_physics',
        help='Output directory'
    )
    parser.add_argument(
        '--game-type',
        choices=['gomoku', 'go', 'chess'],
        default='gomoku',
        help='Game type to analyze'
    )
    parser.add_argument(
        '--disable-modules',
        nargs='+',
        choices=['thermo', 'critical', 'fdt', 'decoherence', 'tunneling', 'entanglement'],
        help='Disable specific physics modules'
    )
    parser.add_argument(
        '--c-puct-sweep',
        action='store_true',
        help='Enable c_puct parameter sweep for phase transition validation'
    )
    parser.add_argument(
        '--c-puct-range',
        type=float,
        nargs=2,
        default=[0.5, 3.0],
        help='c_puct range for parameter sweep (default: 0.5 3.0)'
    )
    parser.add_argument(
        '--c-puct-steps',
        type=int,
        default=10,
        help='Number of c_puct values to test (default: 10)'
    )
    parser.add_argument(
        '--run-cpuct-validation',
        action='store_true',
        help='Run dedicated c_puct phase transition validation'
    )
    parser.add_argument(
        '--evaluator-type',
        choices=['resnet', 'random', 'fast_random'],
        default='resnet',
        help='Type of neural network evaluator (default: resnet)'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    if args.preset == 'quick':
        config = CompleteGeneratorConfig.quick_preset()
    elif args.preset == 'standard':
        config = CompleteGeneratorConfig.standard_preset()
    elif args.preset == 'comprehensive':
        config = CompleteGeneratorConfig.comprehensive_preset()
    elif args.preset == 'deep':
        config = CompleteGeneratorConfig.deep_physics_preset()
    else:  # overnight
        config = CompleteGeneratorConfig.overnight_preset()
    
    # Override parameters if specified
    if args.games:
        config.target_games = args.games
    if args.sims:
        config.sims_per_game = args.sims
    
    config.output_dir = args.output
    config.game_type = args.game_type
    config.evaluator_type = args.evaluator_type
    
    # Configure c_puct parameter sweep
    if args.c_puct_sweep:
        config.enable_parameter_sweep = True
        config.cpuct_range = tuple(args.c_puct_range)
        config.sweep_resolution = args.c_puct_steps
    
    # Configure complete ensemble analysis
    ensemble_config = CompleteEnsembleConfig(
        measure_authentic_physics=True,
        analyze_thermodynamics='thermo' not in (args.disable_modules or []),
        analyze_critical_phenomena='critical' not in (args.disable_modules or []),
        analyze_fluctuation_dissipation='fdt' not in (args.disable_modules or []),
        analyze_decoherence='decoherence' not in (args.disable_modules or []),
        analyze_tunneling='tunneling' not in (args.disable_modules or []),
        analyze_entanglement='entanglement' not in (args.disable_modules or []),
        min_games_for_statistics=10,
        bootstrap_samples=1000,
        confidence_level=0.95,
        save_raw_measurements=True,
        save_validated_results=True,
        output_dir=Path(args.output) / 'complete_physics'
    )
    
    config.ensemble_config = ensemble_config
    
    # Check if dedicated c_puct validation is requested
    if args.run_cpuct_validation:
        logger.info("="*70)
        logger.info("DEDICATED c_puct PHASE TRANSITION VALIDATION")
        logger.info("="*70)
        
        # Configure c_puct validation
        cpuct_config = CpuctPhaseConfig(
            cpuct_range=tuple(args.c_puct_range),
            cpuct_resolution=args.c_puct_steps,
            games_per_point=min(50, config.target_games // args.c_puct_steps),
            sims_per_game=config.sims_per_game,
            game_type=config.game_type,
            output_dir=str(Path(args.output) / 'cpuct_validation')
        )
        
        logger.info(f"c_puct validation configuration:")
        logger.info(f"  c_puct range: {cpuct_config.cpuct_range}")
        logger.info(f"  Resolution: {cpuct_config.cpuct_resolution} points")
        logger.info(f"  Games per point: {cpuct_config.games_per_point}")
        logger.info(f"  Total games: {cpuct_config.cpuct_resolution * cpuct_config.games_per_point}")
        logger.info(f"  Simulations per game: {cpuct_config.sims_per_game}")
        logger.info("")
        
        # Run c_puct validation
        try:
            cpuct_results = run_cpuct_phase_validation(cpuct_config)
            
            logger.info("="*70)
            logger.info("c_puct VALIDATION RESULTS")
            logger.info("="*70)
            
            if 'summary' in cpuct_results:
                summary = cpuct_results['summary']
                logger.info(f"Phase transitions detected: {summary['n_transitions']}")
                
                if summary['critical_points']:
                    logger.info("Critical c_puct values:")
                    for i, cp in enumerate(summary['critical_points']):
                        logger.info(f"  Transition {i+1}: c_puct = {cp:.3f}")
                
                logger.info(f"Average confidence: {summary['average_confidence']:.3f}")
                logger.info(f"Total games analyzed: {summary['total_games_analyzed']}")
            
            logger.info("="*70)
            return cpuct_results
            
        except Exception as e:
            logger.error(f"c_puct validation failed: {e}")
            return {'error': str(e)}
    
    logger.info("="*70)
    logger.info("COMPLETE MCTS PHYSICS ANALYSIS")
    logger.info("="*70)
    logger.info("Configuration:")
    logger.info(f"  Games: {config.target_games}")
    logger.info(f"  Simulations per game: {config.sims_per_game}")
    logger.info(f"  Game type: {config.game_type}")
    logger.info(f"  Evaluator type: {config.evaluator_type}")
    logger.info(f"  Output directory: {config.output_dir}")
    logger.info(f"  Parameter sweep enabled: {config.enable_parameter_sweep}")
    if config.enable_parameter_sweep:
        logger.info(f"  c_puct range: {config.cpuct_range}")
        logger.info(f"  c_puct resolution: {config.sweep_resolution}")
        logger.info(f"  Estimated total simulations: {config.target_games * config.sims_per_game * config.sweep_resolution:,}")
    logger.info("")
    logger.info("Physics modules enabled:")
    logger.info("  Statistical Mechanics:")
    logger.info(f"    • Thermodynamics: {ensemble_config.analyze_thermodynamics}")
    logger.info(f"    • Critical phenomena: {ensemble_config.analyze_critical_phenomena}")
    logger.info(f"    • Fluctuation-dissipation: {ensemble_config.analyze_fluctuation_dissipation}")
    logger.info("  Quantum Phenomena:")
    logger.info(f"    • Decoherence: {ensemble_config.analyze_decoherence}")
    logger.info(f"    • Tunneling: {ensemble_config.analyze_tunneling}")
    logger.info(f"    • Entanglement: {ensemble_config.analyze_entanglement}")
    logger.info("  Advanced Quantum Phenomena:")
    logger.info(f"    • Gauge-invariant policy: {ensemble_config.analyze_gauge_policy}")
    logger.info(f"    • Quantum error correction: {ensemble_config.analyze_quantum_error_correction}")
    logger.info(f"    • Topological analysis: {ensemble_config.analyze_topological}")
    logger.info("")
    logger.info("Key features:")
    logger.info("  ✓ Authentic physics extraction (no predetermined formulas)")
    logger.info("  ✓ Cross-validation between modules")
    logger.info("  ✓ Comprehensive statistical analysis")
    logger.info("  ✓ All validated physics modules included")
    logger.info("="*70)
    
    # Create and run generator
    generator = CompleteAutoDataGenerator(config)
    
    try:
        results = generator.run()
        
        # Log key findings
        if 'complete_physics_analysis' in results and results['complete_physics_analysis']:
            analysis = results['complete_physics_analysis']
            
            logger.info("")
            logger.info("="*70)
            logger.info("KEY PHYSICS DISCOVERIES")
            logger.info("="*70)
            
            # Summary statistics
            if 'summary' in analysis:
                summary = analysis['summary']
                
                if 'temperature' in summary:
                    temp = summary['temperature']
                    logger.info(f"Temperature:")
                    logger.info(f"  • Mean: {temp['mean']:.3f} ± {temp['std']:.3f}")
                    logger.info(f"  • Measurements: {temp['n_measurements']}")
                    logger.info(f"  • Method: Fitted π(a) ∝ exp(β·Q(a)) to visits")
                
                if 'n_critical_points' in summary:
                    logger.info(f"\nCritical phenomena:")
                    logger.info(f"  • Critical points found: {summary['n_critical_points']}")
                
                if 'tunneling' in summary:
                    tunn = summary['tunneling']
                    logger.info(f"\nQuantum tunneling:")
                    logger.info(f"  • Events detected: {tunn['n_events']}")
                    logger.info(f"  • Average barrier height: {tunn['avg_barrier_height']:.2f}")
                
                # Advanced quantum phenomena
                if 'gauge_policy' in summary:
                    gauge = summary['gauge_policy']
                    logger.info(f"\nGauge-invariant policy:")
                    logger.info(f"  • Avg Wilson loops: {gauge['avg_wilson_loops']:.3f}")
                    logger.info(f"  • Gauge invariance: {gauge['gauge_invariance']:.3f}")
                    logger.info(f"  • Policy stability: {gauge['policy_stability']:.3f}")
                
                if 'quantum_error_correction' in summary:
                    qec = summary['quantum_error_correction']
                    logger.info(f"\nQuantum error correction:")
                    logger.info(f"  • Avg error rate: {qec['avg_error_rate']:.3f}")
                    logger.info(f"  • Correction success: {qec['correction_success']:.3f}")
                    logger.info(f"  • Value redundancy: {qec['redundancy']:.3f}")
                
                if 'topological_analysis' in summary:
                    topo = summary['topological_analysis']
                    logger.info(f"\nTopological analysis:")
                    logger.info(f"  • Avg critical points: {topo['avg_critical_points']:.1f}")
                    logger.info(f"  • Topological complexity: {topo['complexity']:.3f}")
                    logger.info(f"  • Feature persistence: {topo['persistence']:.3f}")
            
            # Cross-validations
            if 'cross_validations' in analysis:
                logger.info("\nCross-module validations:")
                for key, val in analysis['cross_validations'].items():
                    if isinstance(val, dict):
                        if 'consistent' in val:
                            status = '✓ Consistent' if val['consistent'] else '✗ Inconsistent'
                            logger.info(f"  • {key}: {status}")
                        if 'ks_test_p_value' in val:
                            logger.info(f"    - KS test p-value: {val['ks_test_p_value']:.3f}")
            
            # Statistical mechanics validations
            if 'statistical_mechanics' in analysis:
                stat_mech = analysis['statistical_mechanics']
                
                logger.info("\nStatistical mechanics validations:")
                
                if 'jarzynski' in stat_mech:
                    jarzynski = stat_mech['jarzynski']
                    status = '✓' if jarzynski['satisfied'] else '✗'
                    logger.info(f"  • Jarzynski equality: {status}")
                    logger.info(f"    - Average work: {jarzynski['avg_work']:.3f}")
                    logger.info(f"    - Free energy estimate: {jarzynski['jarzynski_estimate']:.3f}")
                
                if 'onsager_reciprocity' in stat_mech:
                    status = '✓' if stat_mech['onsager_reciprocity'] else '✗'
                    logger.info(f"  • Onsager reciprocity: {status}")
            
            logger.info("="*70)
            
    except KeyboardInterrupt:
        logger.info("\nAnalysis interrupted by user")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == '__main__':
    main()