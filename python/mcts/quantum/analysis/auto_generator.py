"""
Enhanced MCTS data generator with complete physics analysis.

Uses ALL validated physics modules:
- Statistical mechanics (thermodynamics, critical phenomena, FDT)
- Quantum phenomena (decoherence, tunneling, entanglement)
- Authentic measurements (no predetermined formulas)
"""
import os
import json
import time
import signal
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc
from tqdm import tqdm

# Disable torch multiprocessing file sharing to avoid file descriptor issues
torch.multiprocessing.set_sharing_strategy('file_system')

# Import memory management
try:
    from memory_manager import (
        MemoryConfig, OvernightAnalysisManager, configure_for_overnight_analysis,
        MemoryMonitor, CheckpointManager, CompressedDataStorage
    )
    MEMORY_MANAGER_AVAILABLE = True
except ImportError:
    MEMORY_MANAGER_AVAILABLE = False
    MemoryConfig = None
    OvernightAnalysisManager = None
    configure_for_overnight_analysis = None
    MemoryMonitor = None
    CheckpointManager = None
    CompressedDataStorage = None

# Use absolute imports for analysis modules
from dynamics_extractor import MCTSDynamicsExtractor, ExtractionConfig, DynamicsData
from ensemble_analyzer_complete import CompleteEnsembleAnalyzer, CompleteEnsembleConfig
from mcts_integration import create_mcts_game_generator
from budget_settings import BudgetCalculator

logger = logging.getLogger(__name__)

# Reduce verbosity of MCTS integration logging
logging.getLogger('mcts_integration').setLevel(logging.WARNING)
logging.getLogger('random_evaluator').setLevel(logging.WARNING)
logging.getLogger('python.mcts.gpu.cuda_manager').setLevel(logging.WARNING)


@dataclass
class CompleteGeneratorConfig:
    """Configuration for complete physics generation and analysis"""
    
    # Game generation
    target_games: int = 100
    sims_per_game: int = 5000
    game_type: str = 'gomoku'
    board_size: int = 15
    model_path: Optional[str] = None
    evaluator_type: str = 'resnet'  # 'resnet', 'random', 'fast_random'
    
    # Data extraction
    extract_data: bool = True
    extraction_config: Optional[ExtractionConfig] = None
    
    # Complete physics analysis
    perform_complete_analysis: bool = True
    ensemble_config: Optional[CompleteEnsembleConfig] = None
    
    # Parameter sweeps
    enable_parameter_sweep: bool = False
    temperature_range: Tuple[float, float] = (0.1, 2.0)
    cpuct_range: Tuple[float, float] = (0.5, 3.0)
    sweep_resolution: int = 10
    
    # Output
    output_dir: str = './mcts_complete_analysis'
    save_raw_data: bool = True
    save_analysis_results: bool = True
    
    # Performance
    parallel_games: bool = False
    batch_size: int = 10
    max_workers: int = 1
    
    # Resource management
    manage_resources: bool = True
    memory_cleanup_interval: int = 50
    
    # Progress
    progress_reporting: bool = True
    save_checkpoint_interval: int = 25
    
    def __post_init__(self):
        """Initialize default values"""
        if self.extraction_config is None:
            self.extraction_config = ExtractionConfig(
                extract_q_values=True,
                extract_visits=True,
                extract_value_landscape=True,
                extract_quantum_phenomena=True
            )
            
        if self.ensemble_config is None:
            self.ensemble_config = CompleteEnsembleConfig(
                # Enable all physics analyses
                measure_authentic_physics=True,
                analyze_thermodynamics=True,
                analyze_critical_phenomena=True,
                analyze_fluctuation_dissipation=True,
                analyze_decoherence=True,
                analyze_tunneling=True,
                analyze_entanglement=True
            )
            
    @classmethod
    def quick_preset(cls) -> 'CompleteGeneratorConfig':
        """Quick analysis preset (10 games with random evaluator)"""
        return cls(
            target_games=10,
            sims_per_game=1000,
            save_raw_data=False,
            memory_cleanup_interval=5,
            evaluator_type='random'  # Use random evaluator for quick testing
        )
        
    @classmethod
    def standard_preset(cls) -> 'CompleteGeneratorConfig':
        """Standard analysis preset (50 games)"""
        return cls(
            target_games=50,
            sims_per_game=5000,
            save_checkpoint_interval=10
        )
        
    @classmethod
    def comprehensive_preset(cls) -> 'CompleteGeneratorConfig':
        """Comprehensive analysis with all physics modules"""
        config = cls(
            target_games=100,
            sims_per_game=10000,
            save_checkpoint_interval=20
        )
        
        # Ensure all analyses are enabled
        config.ensemble_config = CompleteEnsembleConfig(
            measure_authentic_physics=True,
            analyze_thermodynamics=True,
            analyze_critical_phenomena=True,
            analyze_fluctuation_dissipation=True,
            analyze_decoherence=True,
            analyze_tunneling=True,
            analyze_entanglement=True,
            min_games_for_statistics=20,
            bootstrap_samples=1000
        )
        
        return config
        
    @classmethod
    def deep_physics_preset(cls) -> 'CompleteGeneratorConfig':
        """Deep physics analysis with parameter sweeps"""
        return cls(
            target_games=200,
            sims_per_game=10000,
            enable_parameter_sweep=True,
            sweep_resolution=15,
            save_checkpoint_interval=25
        )
    
    @classmethod
    def overnight_preset(cls) -> 'CompleteGeneratorConfig':
        """Overnight analysis preset for rigorous statistical analysis"""
        config = cls(
            target_games=1000,  # 10x more games for statistical rigor
            sims_per_game=5000,  # Optimized for hardware constraints
            enable_parameter_sweep=True,
            cpuct_range=(0.3, 4.0),  # Wider range for phase transition validation
            temperature_range=(0.5, 1.5),  # Temperature range for sweep
            sweep_resolution=10,  # 10x10 = 100 points total
            save_checkpoint_interval=50,
            memory_cleanup_interval=50,  # Less frequent cleanup for performance
            parallel_games=False,  # Single worker constraint
            batch_size=1,          # Single game at a time
            max_workers=1          # Single worker constraint
        )
        
        # Enhanced ensemble configuration for overnight analysis
        config.ensemble_config = CompleteEnsembleConfig(
            measure_authentic_physics=True,
            analyze_thermodynamics=True,
            analyze_critical_phenomena=True,
            analyze_fluctuation_dissipation=True,
            analyze_decoherence=True,
            analyze_tunneling=True,
            analyze_entanglement=True,
            analyze_tensor_networks=True,
            analyze_holographic_bounds=True,
            min_games_for_statistics=50,  # Higher threshold for statistical validity
            bootstrap_samples=2000,  # More bootstrap samples
            confidence_level=0.99,  # Higher confidence level
            save_raw_measurements=True,
            save_validated_results=True
        )
        
        return config


class CompleteAutoDataGenerator:
    """
    Complete data generator with all physics analyses.
    
    Features:
    - Uses ALL validated physics modules (statistical + quantum)
    - Authentic measurements (no predetermined formulas)
    - Cross-validation between different physics modules
    - Parameter sweeps for phase diagrams
    """
    
    def __init__(self, config: CompleteGeneratorConfig):
        """Initialize complete generator"""
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.extractor = MCTSDynamicsExtractor(
            config=config.extraction_config,
            n_workers=1
        )
        
        self.ensemble_analyzer = CompleteEnsembleAnalyzer(
            config=config.ensemble_config
        )
        
        # Progress tracking
        self.games_completed = 0
        self.start_time = time.time()
        self.should_stop = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Complete physics generator initialized for {config.target_games} games")
        logger.info("Using ALL validated physics modules:")
        logger.info("  - Statistical: Thermodynamics, Critical, FDT")
        logger.info("  - Quantum: Decoherence, Tunneling, Entanglement")
        logger.info("  - Authentic measurements (no predetermined formulas)")
        
    def _signal_handler(self, signum, frame):
        """Handle interruption gracefully"""
        logger.info(f"Received signal {signum}, stopping gracefully...")
        self.should_stop = True
        
    def generate_parameter_sweep(self) -> Dict[str, List[Any]]:
        """Generate games with parameter variations"""
        logger.info("Starting parameter sweep generation")
        
        sweep_data = {
            'temperatures': [],
            'cpuct_values': [],
            'games': [],
            'dynamics': []
        }
        
        # Temperature variations
        temps = np.linspace(
            self.config.temperature_range[0],
            self.config.temperature_range[1],
            self.config.sweep_resolution
        )
        
        # c_puct variations
        cpucts = np.linspace(
            self.config.cpuct_range[0],
            self.config.cpuct_range[1],
            self.config.sweep_resolution
        )
        
        # Calculate total points and games per point
        total_points = len(temps) * len(cpucts)
        games_per_point = max(1, self.config.target_games // total_points)
        
        logger.info(f"Parameter sweep: {len(temps)} temperatures × {len(cpucts)} c_puct values = {total_points} points")
        logger.info(f"Games per point: {games_per_point} (total target: {self.config.target_games})")
        
        # Create a SINGLE generator instance to reuse
        logger.info("Creating single MCTS generator instance for entire sweep...")
        self._sweep_generator = create_mcts_game_generator(
            sims_per_game=self.config.sims_per_game,
            game_type=self.config.game_type,
            board_size=self.config.board_size,
            model_path=self.config.model_path,
            evaluator_type=self.config.evaluator_type
        )
        
        # Create progress bar for parameter sweep
        with tqdm(total=total_points, 
                 desc="Parameter sweep", 
                 unit="point",
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} points [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            
            for temp in temps:
                for cpuct in cpucts:
                    if self.should_stop:
                        break
                        
                    # Generate games at this parameter point (using shared generator)
                    point_games = self._generate_games_with_params(
                        n_games=games_per_point,
                        temperature=temp,
                        cpuct=cpuct,
                        generator=self._sweep_generator
                    )
                    
                    # Extract dynamics
                    if self.config.extract_data:
                        point_dynamics = self.extractor.extract_batch(point_games)
                    else:
                        point_dynamics = []
                        
                    # Store results
                    for game, dynamics in zip(point_games, point_dynamics):
                        sweep_data['temperatures'].append(temp)
                        sweep_data['cpuct_values'].append(cpuct)
                        sweep_data['games'].append(game)
                        sweep_data['dynamics'].append(dynamics)
                        
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'T': f"{temp:.2f}",
                        'c_puct': f"{cpuct:.2f}",
                        'games/point': games_per_point,
                        'total_games': len(sweep_data['games'])
                    })
                
        return sweep_data
        
    def _generate_games_with_params(self, n_games: int, 
                                   temperature: float, 
                                   cpuct: float,
                                   generator=None) -> List[Any]:
        """Generate games with specific parameters"""
        games = []
        
        # Temporarily suppress logging for cleaner output
        prev_log_levels = {}
        loggers_to_suppress = [
            'mcts_integration',
            'random_evaluator', 
            'python.mcts.gpu.cuda_manager',
            'python.mcts.core.mcts'
        ]
        
        for logger_name in loggers_to_suppress:
            logger_obj = logging.getLogger(logger_name)
            prev_log_levels[logger_name] = logger_obj.level
            logger_obj.setLevel(logging.ERROR)
        
        try:
            # Use provided generator or create new one
            if generator is None:
                generator = create_mcts_game_generator(
                    sims_per_game=self.config.sims_per_game,
                    game_type=self.config.game_type,
                    board_size=self.config.board_size,
                    model_path=self.config.model_path,
                    evaluator_type=self.config.evaluator_type
                )
        
            # Modify MCTS config (just update the c_puct value)
            generator.mcts_config.c_puct = cpuct
            
            # Generate games (no progress bar for small n_games to avoid clutter)
            for _ in range(n_games):
                if self.should_stop:
                    break
                    
                # Temperature is applied during action selection
                game_data = generator.generate_game(temperature_threshold=int(30 * temperature))
                games.append(game_data)
                
        finally:
            # Restore original log levels
            for logger_name, level in prev_log_levels.items():
                logging.getLogger(logger_name).setLevel(level)
                
        return games
        
    def run(self) -> Dict[str, Any]:
        """
        Run complete generation and physics analysis pipeline.
        
        Returns:
            Comprehensive results including all physics analyses
        """
        logger.info("="*60)
        logger.info("COMPLETE MCTS PHYSICS ANALYSIS")
        logger.info("="*60)
        logger.info("Modules enabled:")
        config = self.config.ensemble_config
        logger.info(f"  Authentic physics: {config.measure_authentic_physics}")
        logger.info(f"  Thermodynamics: {config.analyze_thermodynamics}")
        logger.info(f"  Critical phenomena: {config.analyze_critical_phenomena}")
        logger.info(f"  Fluctuation-dissipation: {config.analyze_fluctuation_dissipation}")
        logger.info(f"  Decoherence: {config.analyze_decoherence}")
        logger.info(f"  Tunneling: {config.analyze_tunneling}")
        logger.info(f"  Entanglement: {config.analyze_entanglement}")
        logger.info("="*60)
        
        try:
            # Step 1: Generate data
            if self.config.enable_parameter_sweep:
                logger.info("Performing parameter sweep")
                sweep_data = self.generate_parameter_sweep()
                all_dynamics = sweep_data['dynamics']
                metadata = {
                    'type': 'parameter_sweep',
                    'temperatures': sweep_data['temperatures'],
                    'cpuct_values': sweep_data['cpuct_values']
                }
            else:
                logger.info(f"Generating {self.config.target_games} standard games")
                games = self._generate_standard_games()
                
                # Extract dynamics
                if self.config.extract_data:
                    all_dynamics = self.extractor.extract_batch(games)
                else:
                    all_dynamics = []
                    
                metadata = {
                    'type': 'standard',
                    'n_games': len(games)
                }
                
            # Step 2: Save raw data if requested
            if self.config.save_raw_data and all_dynamics:
                self._save_dynamics_data(all_dynamics)
                
            # Step 3: Perform complete physics analysis
            if self.config.perform_complete_analysis and all_dynamics:
                logger.info("")
                logger.info("="*60)
                logger.info("Starting complete physics analysis")
                logger.info("="*60)
                
                analysis_results = self.ensemble_analyzer.analyze_ensemble(all_dynamics)
                
                # Add metadata
                analysis_results['generation_metadata'] = metadata
                
                # Save analysis results
                if self.config.save_analysis_results:
                    self._save_analysis_results(analysis_results)
                    
            else:
                analysis_results = {}
                
            # Step 4: Create final results
            results = {
                'n_games_generated': len(all_dynamics),
                'total_time': time.time() - self.start_time,
                'complete_physics_analysis': analysis_results,
                'output_directory': str(self.output_dir),
                'config': asdict(self.config)
            }
            
            # Log summary
            self._log_results_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            self._save_checkpoint()
            raise
            
    def _generate_standard_games(self) -> List[Any]:
        """Generate standard games without parameter variation"""
        games = []
        
        # Create MCTS generator
        generator = create_mcts_game_generator(
            sims_per_game=self.config.sims_per_game,
            game_type=self.config.game_type,
            board_size=self.config.board_size,
            model_path=self.config.model_path,
            evaluator_type=self.config.evaluator_type
        )
        
        # Use tqdm for progress tracking
        with tqdm(total=self.config.target_games, 
                 desc="Generating games", 
                 unit="game",
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} games [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            
            for i in range(self.config.target_games):
                if self.should_stop:
                    break
                    
                # Generate game
                start_time = time.time()
                game = generator.generate_game()
                game_time = time.time() - start_time
                
                games.append(game)
                self.games_completed += 1
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'sims/game': f"{self.config.sims_per_game:,}",
                    'game_time': f"{game_time:.1f}s",
                    'avg_rate': f"{self.games_completed/(time.time() - self.start_time):.2f} games/sec"
                })
                
                # Checkpoint saving
                if i % self.config.save_checkpoint_interval == 0:
                    self._save_checkpoint()
                    
                # Memory cleanup
                if i % self.config.memory_cleanup_interval == 0:
                    self._cleanup_memory()
                    
        return games
        
    def _save_dynamics_data(self, dynamics_list: List[DynamicsData]):
        """Save extracted dynamics data"""
        data_dir = self.output_dir / "dynamics_data"
        data_dir.mkdir(exist_ok=True)
        
        for i, dynamics in enumerate(dynamics_list):
            path = data_dir / f"dynamics_{i:04d}.npz"
            dynamics.save_compressed(path)
            
        logger.info(f"Saved {len(dynamics_list)} dynamics datasets")
        
    def _save_analysis_results(self, results: Dict[str, Any]):
        """Save complete analysis results"""
        # Save numerical results
        results_path = self.output_dir / "complete_physics_results.json"
        
        # Convert numpy arrays and paths to JSON-serializable format
        json_safe_results = self._make_json_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(json_safe_results, f, indent=2)
            
        logger.info(f"Saved complete physics results to {results_path}")
        
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and paths to JSON-serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, '__dict__'):
            # Handle dataclasses
            return {k: self._make_json_serializable(v) for k, v in obj.__dict__.items()}
        else:
            return obj
            
    def _report_progress(self, last_game_time: float):
        """Report generation progress"""
        elapsed = time.time() - self.start_time
        games_remaining = self.config.target_games - self.games_completed
        eta = games_remaining * last_game_time
        
        progress_pct = (self.games_completed / self.config.target_games) * 100
        
        logger.info(
            f"Progress: {self.games_completed}/{self.config.target_games} "
            f"({progress_pct:.1f}%) - ETA: {eta/60:.1f}min - "
            f"Rate: {self.games_completed/elapsed:.2f} games/sec"
        )
        
    def _save_checkpoint(self):
        """Save progress checkpoint"""
        checkpoint = {
            'games_completed': self.games_completed,
            'elapsed_time': time.time() - self.start_time,
            'config': self._make_json_serializable(asdict(self.config))
        }
        
        checkpoint_path = self.output_dir / "checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
            
    def _cleanup_memory(self):
        """Clean up memory periodically"""
        if self.config.manage_resources:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    def _log_results_summary(self, results: Dict[str, Any]):
        """Log comprehensive summary of results"""
        logger.info("")
        logger.info("="*60)
        logger.info("COMPLETE PHYSICS ANALYSIS SUMMARY")
        logger.info("="*60)
        logger.info(f"Games analyzed: {results['n_games_generated']}")
        logger.info(f"Total time: {results['total_time']/60:.1f} minutes")
        
        if 'complete_physics_analysis' in results and results['complete_physics_analysis']:
            analysis = results['complete_physics_analysis']
            
            # Authentic measurements
            if 'authentic_measurements' in analysis:
                auth = analysis['authentic_measurements']
                if 'temperatures' in auth:
                    logger.info(f"\nAuthentic temperature measurements: {len(auth['temperatures'])}")
                if 'temperature_scaling' in auth and 'best_model' in auth['temperature_scaling']:
                    scaling = auth['temperature_scaling']
                    logger.info(f"Temperature scaling: {scaling['best_model']} "
                              f"(R² = {scaling['best_model_details']['r_squared']:.3f})")
            
            # Statistical mechanics
            if 'statistical_mechanics' in analysis:
                stat_mech = analysis['statistical_mechanics']
                
                if 'jarzynski' in stat_mech:
                    jarzynski = stat_mech['jarzynski']
                    logger.info(f"\nJarzynski equality: {'✓' if jarzynski['satisfied'] else '✗'}")
                    
                if 'critical_exponents' in stat_mech:
                    exps = stat_mech['critical_exponents']
                    logger.info(f"Critical exponents: β/ν={exps['beta_over_nu']:.3f}, "
                              f"γ/ν={exps['gamma_over_nu']:.3f}")
                    
                if 'onsager_reciprocity' in stat_mech:
                    logger.info(f"Onsager reciprocity: {'✓' if stat_mech['onsager_reciprocity'] else '✗'}")
            
            # Quantum phenomena
            if 'quantum_phenomena' in analysis:
                quantum = analysis['quantum_phenomena']
                
                if 'decoherence' in quantum and quantum['decoherence']:
                    logger.info(f"\nDecoherence trajectories: {len(quantum['decoherence'])}")
                    
                if 'tunneling_events' in quantum:
                    logger.info(f"Tunneling events: {len(quantum['tunneling_events'])}")
                    
                if 'entanglement' in quantum and quantum['entanglement']:
                    logger.info(f"Entanglement measurements: {len(quantum['entanglement'])}")
            
            # Cross-validations
            if 'cross_validations' in analysis:
                logger.info("\nCross-validations:")
                for key, val in analysis['cross_validations'].items():
                    if isinstance(val, dict) and 'consistent' in val:
                        logger.info(f"  {key}: {'✓' if val['consistent'] else '✗'}")
            
            # Plots
            if 'plots' in analysis:
                logger.info(f"\nGenerated {len(analysis['plots'])} comprehensive plots")
                
        logger.info("="*60)


def create_complete_generator(preset: str = 'standard') -> CompleteAutoDataGenerator:
    """
    Create complete physics generator with preset configuration.
    
    Args:
        preset: Configuration preset ('quick', 'standard', 'comprehensive', 'deep')
        
    Returns:
        Configured generator instance
    """
    if preset == 'quick':
        config = CompleteGeneratorConfig.quick_preset()
    elif preset == 'standard':
        config = CompleteGeneratorConfig.standard_preset()
    elif preset == 'comprehensive':
        config = CompleteGeneratorConfig.comprehensive_preset()
    elif preset == 'deep':
        config = CompleteGeneratorConfig.deep_physics_preset()
    else:
        raise ValueError(f"Unknown preset: {preset}")
        
    return CompleteAutoDataGenerator(config)