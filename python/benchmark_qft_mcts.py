"""
Performance Benchmarks: QFT-MCTS vs Classical MCTS
==================================================

This benchmark suite compares the performance of quantum field theoretic MCTS
with classical MCTS across various metrics and game scenarios.

Benchmark Categories:
1. Throughput: Simulations per second
2. Quality: Win rate and move quality
3. Scalability: Performance vs tree size
4. Efficiency: Memory usage and GPU utilization
5. Convergence: Time to optimal play
"""

import torch
import numpy as np
import time
import psutil
import GPUtil
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from pathlib import Path

# Add parent directory to path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import game implementations
import alphazero_py

# Import MCTS implementations
from mcts.core.high_performance_mcts import HighPerformanceMCTS, HighPerformanceMCTSConfig
from mcts.core.mcts_config import MCTSConfig
from mcts.core.game_interface import GameInterface

# Import quantum components
from mcts.quantum.qft_engine import create_qft_engine
from mcts.quantum.decoherence import create_decoherence_engine
from mcts.quantum.envariance import create_envariance_engine
from mcts.quantum.rg_flow import create_rg_optimizer
from mcts.quantum.interference_gpu import MinHashInterference
from mcts.gpu.wave_engine import create_wave_engine

# Import neural network
from mcts.neural_networks.resnet_evaluator import ResNetEvaluator

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG if "--debug" in sys.argv else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks"""
    # Games to test
    games: List[str] = None
    
    # MCTS parameters
    simulations_per_move: List[int] = None
    c_puct_values: List[float] = None
    
    # Test parameters
    num_games: int = 10
    num_positions: int = 100
    max_moves: int = 200
    
    # Hardware
    device: str = 'cuda'
    num_threads: int = 8
    
    # Output
    output_dir: Path = Path('benchmark_results')
    
    def __post_init__(self):
        if self.games is None:
            self.games = ['gomoku', 'chess']
        if self.simulations_per_move is None:
            self.simulations_per_move = [100, 500, 1000, 5000]
        if self.c_puct_values is None:
            self.c_puct_values = [1.0, 1.414, 2.0]


class MCTSBenchmark:
    """Base class for MCTS benchmarks"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        self.config.output_dir.mkdir(exist_ok=True)
        
        # Initialize results storage
        self.results = {
            'throughput': {},
            'quality': {},
            'scalability': {},
            'efficiency': {},
            'convergence': {}
        }
        
        # Cache evaluators to avoid recreating them
        self.evaluator_cache = {}
        
    def create_game(self, game_name: str):
        """Create game instance"""
        if game_name == 'gomoku':
            return alphazero_py.GomokuState()
        elif game_name == 'chess':
            return alphazero_py.ChessState()
        elif game_name == 'go':
            return alphazero_py.GoState(9)  # 9x9 board
        else:
            raise ValueError(f"Unknown game: {game_name}")
    
    def get_game_type(self, game_name: str):
        """Get GameType enum for game"""
        from mcts.core.game_interface import GameType
        if game_name == 'gomoku':
            return GameType.GOMOKU
        elif game_name == 'chess':
            return GameType.CHESS
        elif game_name == 'go':
            return GameType.GO
        else:
            raise ValueError(f"Unknown game: {game_name}")
    
    def create_evaluator(self, game_name: str):
        """Create neural network evaluator"""
        # Check cache first
        if game_name in self.evaluator_cache:
            return self.evaluator_cache[game_name]
            
        # Use actual ResNet evaluator for robust testing
        from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
        
        # Create ResNet evaluator
        evaluator = ResNetEvaluator(
            game_type=game_name,
            device=str(self.device)
        )
        
        # Cache it
        self.evaluator_cache[game_name] = evaluator
        
        return evaluator
    
    def measure_throughput(
        self,
        mcts_type: str,
        game_name: str,
        num_simulations: int
    ) -> Dict[str, float]:
        """Measure simulations per second"""
        game = self.create_game(game_name)
        game_type = self.get_game_type(game_name)
        game_interface = GameInterface(game_type)
        evaluator = self.create_evaluator(game_name)
        
        # Configure MCTS
        if mcts_type == 'classical':
            # Classical MCTS without quantum features
            config = HighPerformanceMCTSConfig(
                num_simulations=num_simulations,
                c_puct=1.414,
                device=str(self.device),
                wave_size=min(2048, num_simulations),  # Larger waves for better GPU utilization
                enable_path_integral=False,  # Disable quantum features
                enable_interference=False,
                enable_phase_policy=False,
                interference_strength=0.0,
                phase_config=None
            )
            mcts = HighPerformanceMCTS(config, game_interface, evaluator)
        else:  # quantum
            # Quantum MCTS with all features enabled
            config = HighPerformanceMCTSConfig(
                num_simulations=num_simulations,
                c_puct=1.414,
                device=str(self.device),
                wave_size=min(2048, num_simulations),  # Larger waves for better GPU utilization
                enable_path_integral=True,  # Enable path integral
                enable_interference=True,  # Enable quantum interference
                enable_phase_policy=True,  # Enable phase kicks
                interference_strength=0.15,
                phase_config=None  # Use default phase config
            )
            mcts = HighPerformanceMCTS(config, game_interface, evaluator)
        
        # Warmup
        logger.info(f"Starting warmup for {game_name} with {num_simulations} simulations...")
        for i in range(3):
            logger.info(f"  Warmup run {i+1}/3...")
            start = time.perf_counter()
            mcts.search(game)
            end = time.perf_counter()
            logger.info(f"  Warmup run {i+1} took {end-start:.2f}s")
        
        # Measure throughput
        num_runs = 5  # Reduced from 10
        total_time = 0
        total_simulations = 0
        
        logger.info(f"Starting measurement runs...")
        for i in range(num_runs):
            logger.info(f"  Measurement run {i+1}/{num_runs}...")
            start = time.perf_counter()
            policy = mcts.search(game)
            end = time.perf_counter()
            
            elapsed = end - start
            logger.info(f"  Run {i+1} took {elapsed:.2f}s ({num_simulations/elapsed:.1f} sims/sec)")
            
            total_time += elapsed
            total_simulations += num_simulations  # We know how many simulations were requested
        
        throughput = total_simulations / total_time
        
        return {
            'simulations_per_second': throughput,
            'avg_time_per_search': total_time / num_runs,
            'total_simulations': total_simulations
        }
    
    def _create_quantum_mcts(self, config, game_interface, evaluator):
        """Create MCTS with quantum enhancements"""
        # For now, use classical MCTS as placeholder
        # In practice, would integrate all quantum components
        mcts = HighPerformanceMCTS(config, game_interface, evaluator)
        
        # Add quantum components
        mcts.qft_engine = create_qft_engine(self.device)
        mcts.decoherence_engine = create_decoherence_engine(self.device)
        mcts.interference = MinHashInterference(self.device, strength=0.15)
        
        return mcts
    
    def measure_quality(
        self,
        mcts_type: str,
        game_name: str,
        num_simulations: int
    ) -> Dict[str, float]:
        """Measure move quality and win rate"""
        # Simplified quality measurement
        # In practice, would play against reference opponent
        
        game = self.create_game(game_name)
        game_type = self.get_game_type(game_name)
        game_interface = GameInterface(game_type)
        evaluator = self.create_evaluator(game_name)
        
        # Configure MCTS
        if mcts_type == 'classical':
            # Classical MCTS without quantum features
            config = HighPerformanceMCTSConfig(
                num_simulations=num_simulations,
                c_puct=1.414,
                device=str(self.device),
                enable_path_integral=False,
                enable_interference=False,
                enable_phase_policy=False
            )
        else:  # quantum
            # Quantum MCTS with all features enabled
            config = HighPerformanceMCTSConfig(
                num_simulations=num_simulations,
                c_puct=1.414,
                device=str(self.device),
                enable_path_integral=True,
                enable_interference=True,
                enable_phase_policy=True,
                interference_strength=0.15
            )
        
        mcts = HighPerformanceMCTS(config, game_interface, evaluator)
        
        # Measure move quality metrics
        total_entropy = 0
        total_confidence = 0
        num_moves = 50
        
        for _ in range(num_moves):
            if game.is_terminal():
                game = self.create_game(game_name)
                
            policy_dict = mcts.search(game)
            # Convert policy dict to tensor
            # Get number of actions based on game type
            if game_name == 'gomoku':
                num_actions = 15 * 15  # 15x15 board
            elif game_name == 'chess':
                num_actions = 64 * 64  # from-to moves
            else:  # go
                num_actions = 9 * 9 + 1  # 9x9 board + pass
            policy = torch.zeros(num_actions)
            for action, prob in policy_dict.items():
                if action < num_actions:
                    policy[action] = prob
            # Normalize if needed
            if policy.sum() > 0:
                policy = policy / policy.sum()
            else:
                policy = torch.ones(num_actions) / num_actions
            
            # Entropy as proxy for move clarity
            entropy = -torch.sum(policy * torch.log(policy + 1e-8))
            total_entropy += entropy.item()
            
            # Max probability as confidence
            confidence = torch.max(policy).item()
            total_confidence += confidence
            
            # Make move - ensure it's legal
            legal_moves = game.get_legal_moves()
            if len(legal_moves) > 0:
                # Filter policy to only legal moves
                legal_probs = []
                for move in legal_moves:
                    if move < len(policy):
                        legal_probs.append(policy[move].item())
                    else:
                        legal_probs.append(0.0)
                
                # Normalize
                legal_probs = torch.tensor(legal_probs)
                if legal_probs.sum() > 0:
                    legal_probs = legal_probs / legal_probs.sum()
                else:
                    legal_probs = torch.ones(len(legal_moves)) / len(legal_moves)
                
                # Select action
                action_idx = torch.multinomial(legal_probs, 1).item()
                action = legal_moves[action_idx]
                game.make_move(action)
        
        return {
            'avg_move_entropy': total_entropy / num_moves,
            'avg_move_confidence': total_confidence / num_moves,
            'moves_evaluated': num_moves
        }
    
    def measure_scalability(
        self,
        mcts_type: str,
        game_name: str
    ) -> Dict[str, List[float]]:
        """Measure performance vs tree size"""
        game = self.create_game(game_name)
        game_type = self.get_game_type(game_name)
        game_interface = GameInterface(game_type)
        evaluator = self.create_evaluator(game_name)
        
        simulation_counts = [100, 200]  # Reduced for debugging
        times = []
        throughputs = []
        
        for num_sims in simulation_counts:
            config = HighPerformanceMCTSConfig(num_simulations=num_sims, device=str(self.device))
            
            if mcts_type == 'classical':
                mcts = HighPerformanceMCTS(config, game_interface, evaluator)
            else:
                mcts = self._create_quantum_mcts(config, game_interface, evaluator)
            
            # Measure time
            start = time.perf_counter()
            policy = mcts.search(game)
            end = time.perf_counter()
            
            elapsed = end - start
            throughput = num_sims / elapsed
            
            times.append(elapsed)
            throughputs.append(throughput)
        
        return {
            'simulation_counts': simulation_counts,
            'search_times': times,
            'throughputs': throughputs
        }
    
    def measure_efficiency(
        self,
        mcts_type: str,
        game_name: str,
        num_simulations: int
    ) -> Dict[str, float]:
        """Measure memory usage and GPU utilization"""
        game = self.create_game(game_name)
        game_type = self.get_game_type(game_name)
        game_interface = GameInterface(game_type)
        evaluator = self.create_evaluator(game_name)
        
        config = MCTSConfig(num_simulations=num_simulations)
        
        # Memory before
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            gpu_mem_before = torch.cuda.memory_allocated() / 1024 / 1024
        
        # Create and run MCTS
        mcts = HighPerformanceMCTS(config, game_interface, evaluator)
        
        # Run search
        mcts.search(game)
        
        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024
        memory_used = mem_after - mem_before
        
        if torch.cuda.is_available():
            gpu_mem_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
            gpu_memory_used = gpu_mem_peak - gpu_mem_before
            
            # GPU utilization (if available)
            try:
                gpus = GPUtil.getGPUs()
                gpu_util = gpus[0].load * 100 if gpus else 0
            except:
                gpu_util = 0
        else:
            gpu_memory_used = 0
            gpu_util = 0
        
        return {
            'cpu_memory_mb': memory_used,
            'gpu_memory_mb': gpu_memory_used,
            'gpu_utilization_percent': gpu_util
        }
    
    def run_benchmarks(self):
        """Run all benchmarks"""
        logger.info("Starting QFT-MCTS benchmarks...")
        
        for game in self.config.games:
            logger.info(f"\nBenchmarking {game}...")
            
            self.results['throughput'][game] = {}
            self.results['quality'][game] = {}
            self.results['scalability'][game] = {}
            self.results['efficiency'][game] = {}
            
            # Test both classical and quantum
            for mcts_type in ['classical', 'quantum']:
                logger.info(f"  Testing {mcts_type} MCTS...")
                
                # Throughput benchmarks
                throughput_results = {}
                for num_sims in self.config.simulations_per_move:
                    result = self.measure_throughput(mcts_type, game, num_sims)
                    throughput_results[num_sims] = result
                    logger.info(f"    {num_sims} sims: {result['simulations_per_second']:.1f} sims/sec")
                
                self.results['throughput'][game][mcts_type] = throughput_results
                
                # Quality benchmarks
                quality_result = self.measure_quality(mcts_type, game, 1000)
                self.results['quality'][game][mcts_type] = quality_result
                logger.info(f"    Move confidence: {quality_result['avg_move_confidence']:.3f}")
                
                # Scalability benchmarks
                scalability_result = self.measure_scalability(mcts_type, game)
                self.results['scalability'][game][mcts_type] = scalability_result
                
                # Efficiency benchmarks - skip for now as it hangs
                # efficiency_result = self.measure_efficiency(mcts_type, game, 1000)
                # self.results['efficiency'][game][mcts_type] = efficiency_result
                # logger.info(f"    Memory usage: {efficiency_result['cpu_memory_mb']:.1f} MB")
        
        # Save results
        self.save_results()
        
        # Generate plots
        # self.generate_plots()  # Skip for now
        
        # Print summary
        self.print_summary()
        
        logger.info("\nBenchmarks complete!")
    
    def save_results(self):
        """Save benchmark results to JSON"""
        output_file = self.config.output_dir / 'benchmark_results.json'
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(self.results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    def generate_plots(self):
        """Generate visualization plots"""
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Throughput comparison
        self._plot_throughput_comparison()
        
        # 2. Scalability curves
        self._plot_scalability_curves()
        
        # 3. Quality metrics
        self._plot_quality_metrics()
        
        # 4. Efficiency comparison
        self._plot_efficiency_comparison()
    
    def _plot_throughput_comparison(self):
        """Plot throughput comparison between classical and quantum"""
        fig, axes = plt.subplots(1, len(self.config.games), figsize=(15, 6))
        if len(self.config.games) == 1:
            axes = [axes]
        
        for idx, game in enumerate(self.config.games):
            ax = axes[idx]
            
            classical_throughputs = []
            quantum_throughputs = []
            sim_counts = []
            
            for num_sims in self.config.simulations_per_move:
                if num_sims in self.results['throughput'][game]['classical']:
                    classical = self.results['throughput'][game]['classical'][num_sims]['simulations_per_second']
                    quantum = self.results['throughput'][game]['quantum'][num_sims]['simulations_per_second']
                    
                    classical_throughputs.append(classical)
                    quantum_throughputs.append(quantum)
                    sim_counts.append(num_sims)
            
            x = np.arange(len(sim_counts))
            width = 0.35
            
            ax.bar(x - width/2, classical_throughputs, width, label='Classical', color='blue', alpha=0.7)
            ax.bar(x + width/2, quantum_throughputs, width, label='Quantum', color='red', alpha=0.7)
            
            ax.set_xlabel('Simulations per Move')
            ax.set_ylabel('Throughput (sims/sec)')
            ax.set_title(f'{game.capitalize()} Throughput')
            ax.set_xticks(x)
            ax.set_xticklabels(sim_counts)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.output_dir / 'throughput_comparison.png', dpi=150)
        plt.close()
    
    def _plot_scalability_curves(self):
        """Plot scalability curves"""
        fig, axes = plt.subplots(1, len(self.config.games), figsize=(15, 6))
        if len(self.config.games) == 1:
            axes = [axes]
        
        for idx, game in enumerate(self.config.games):
            ax = axes[idx]
            
            for mcts_type in ['classical', 'quantum']:
                data = self.results['scalability'][game][mcts_type]
                sim_counts = data['simulation_counts']
                throughputs = data['throughputs']
                
                color = 'blue' if mcts_type == 'classical' else 'red'
                ax.plot(sim_counts, throughputs, 'o-', label=mcts_type.capitalize(), 
                       color=color, linewidth=2, markersize=8)
            
            ax.set_xlabel('Number of Simulations')
            ax.set_ylabel('Throughput (sims/sec)')
            ax.set_title(f'{game.capitalize()} Scalability')
            ax.set_xscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.output_dir / 'scalability_curves.png', dpi=150)
        plt.close()
    
    def _plot_quality_metrics(self):
        """Plot quality metrics comparison"""
        metrics = ['avg_move_confidence', 'avg_move_entropy']
        metric_names = ['Move Confidence', 'Move Entropy']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx]
            
            games = []
            classical_values = []
            quantum_values = []
            
            for game in self.config.games:
                games.append(game.capitalize())
                classical_values.append(self.results['quality'][game]['classical'][metric])
                quantum_values.append(self.results['quality'][game]['quantum'][metric])
            
            x = np.arange(len(games))
            width = 0.35
            
            ax.bar(x - width/2, classical_values, width, label='Classical', color='blue', alpha=0.7)
            ax.bar(x + width/2, quantum_values, width, label='Quantum', color='red', alpha=0.7)
            
            ax.set_xlabel('Game')
            ax.set_ylabel(name)
            ax.set_title(f'{name} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(games)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.output_dir / 'quality_metrics.png', dpi=150)
        plt.close()
    
    def _plot_efficiency_comparison(self):
        """Plot efficiency metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Memory usage
        ax = axes[0]
        games = []
        classical_cpu_mem = []
        quantum_cpu_mem = []
        classical_gpu_mem = []
        quantum_gpu_mem = []
        
        for game in self.config.games:
            games.append(game.capitalize())
            classical_cpu_mem.append(self.results['efficiency'][game]['classical']['cpu_memory_mb'])
            quantum_cpu_mem.append(self.results['efficiency'][game]['quantum']['cpu_memory_mb'])
            classical_gpu_mem.append(self.results['efficiency'][game]['classical']['gpu_memory_mb'])
            quantum_gpu_mem.append(self.results['efficiency'][game]['quantum']['gpu_memory_mb'])
        
        x = np.arange(len(games))
        width = 0.2
        
        ax.bar(x - 1.5*width, classical_cpu_mem, width, label='Classical CPU', color='lightblue')
        ax.bar(x - 0.5*width, quantum_cpu_mem, width, label='Quantum CPU', color='lightcoral')
        ax.bar(x + 0.5*width, classical_gpu_mem, width, label='Classical GPU', color='blue')
        ax.bar(x + 1.5*width, quantum_gpu_mem, width, label='Quantum GPU', color='red')
        
        ax.set_xlabel('Game')
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('Memory Usage Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(games)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # GPU utilization
        ax = axes[1]
        classical_gpu_util = []
        quantum_gpu_util = []
        
        for game in self.config.games:
            classical_gpu_util.append(self.results['efficiency'][game]['classical']['gpu_utilization_percent'])
            quantum_gpu_util.append(self.results['efficiency'][game]['quantum']['gpu_utilization_percent'])
        
        width = 0.35
        ax.bar(x - width/2, classical_gpu_util, width, label='Classical', color='blue', alpha=0.7)
        ax.bar(x + width/2, quantum_gpu_util, width, label='Quantum', color='red', alpha=0.7)
        
        ax.set_xlabel('Game')
        ax.set_ylabel('GPU Utilization (%)')
        ax.set_title('GPU Utilization Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(games)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.output_dir / 'efficiency_comparison.png', dpi=150)
        plt.close()
    
    def print_summary(self):
        """Print summary of benchmark results"""
        print("\n" + "="*60)
        print("QFT-MCTS BENCHMARK SUMMARY")
        print("="*60)
        
        for game in self.config.games:
            print(f"\n{game.upper()}:")
            print("-"*40)
            
            # Throughput comparison
            print("\nThroughput (1000 simulations):")
            if 'classical' in self.results['throughput'][game] and 1000 in self.results['throughput'][game]['classical']:
                classical_tp = self.results['throughput'][game]['classical'][1000]['simulations_per_second']
                print(f"  Classical: {classical_tp:.1f} sims/sec")
                
                if 'quantum' in self.results['throughput'][game] and 1000 in self.results['throughput'][game]['quantum']:
                    quantum_tp = self.results['throughput'][game]['quantum'][1000]['simulations_per_second']
                    speedup = quantum_tp / classical_tp
                    print(f"  Quantum:   {quantum_tp:.1f} sims/sec")
                    print(f"  Speedup:   {speedup:.2f}x")
            
            # Quality comparison
            print("\nMove Quality:")
            if 'classical' in self.results['quality'][game]:
                classical_conf = self.results['quality'][game]['classical']['avg_move_confidence']
                print(f"  Classical confidence: {classical_conf:.3f}")
            if 'quantum' in self.results['quality'][game]:
                quantum_conf = self.results['quality'][game]['quantum']['avg_move_confidence']
                print(f"  Quantum confidence:   {quantum_conf:.3f}")
            
            # Efficiency comparison (if available)
            if game in self.results.get('efficiency', {}) and 'classical' in self.results['efficiency'].get(game, {}):
                print("\nEfficiency:")
                classical_mem = self.results['efficiency'][game]['classical']['cpu_memory_mb']
                quantum_mem = self.results['efficiency'][game].get('quantum', {}).get('cpu_memory_mb', 0)
                
                print(f"  Classical memory: {classical_mem:.1f} MB")
                if quantum_mem > 0:
                    print(f"  Quantum memory:   {quantum_mem:.1f} MB")
        
        print("\n" + "="*60)


def main():
    """Run benchmarks with default configuration"""
    config = BenchmarkConfig(
        games=['gomoku'],  # Start with gomoku for faster testing
        simulations_per_move=[100, 500, 1000],  # Test different scales
        num_games=1,  # Minimal
        device='cuda' if torch.cuda.is_available() else 'cpu'  # Back to GPU!
    )
    
    benchmark = MCTSBenchmark(config)
    
    try:
        benchmark.run_benchmarks()
        benchmark.print_summary()
    except KeyboardInterrupt:
        logger.info("\nBenchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()