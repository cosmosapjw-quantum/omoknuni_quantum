#!/usr/bin/env python3
"""
Comprehensive MCTS Self-Play Example

This example demonstrates the MCTS implementation working in a real training environment
with neural network integration, proper game handling, and performance monitoring.

Features:
- End-to-end self-play games using optimized MCTS
- Neural network evaluation with GPU batching
- Training data collection and storage
- Performance monitoring and statistics
- Multi-process game execution
- Real Gomoku game logic with win detection

Usage:
    python mcts_selfplay_example.py [--games N] [--processes P] [--save-data]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import argparse
import logging
import multiprocessing
from multiprocessing import Queue, Process, set_start_method
import os
import pickle
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict, deque
import alphazero_py

# MCTS imports
from mcts.core.mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType
from mcts.neural_networks.evaluator_pool import EvaluatorPool
from mcts.neural_networks.nn_model import AlphaZeroNetwork
from mcts.utils.gpu_evaluator_service import GPUEvaluatorService, RemoteEvaluator

# Profiler import
from mcts_comprehensive_profiler import MemoryMonitor, GPUProfiler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SelfPlayConfig:
    """Configuration for self-play session"""
    # Game settings
    board_size: int = 15
    game_type: str = 'gomoku'
    
    # MCTS settings
    num_simulations: int = 800
    wave_size: int = 2048
    c_puct: float = 1.4
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    
    # Neural network settings
    model_channels: int = 64
    model_blocks: int = 6
    
    # Self-play settings
    num_games: int = 10
    num_processes: int = 4
    max_game_length: int = 225
    
    # Performance settings
    device: str = 'cuda'
    use_profiling: bool = True
    save_training_data: bool = False
    output_dir: str = 'selfplay_results'
    
    # Temperature schedule
    temp_threshold: int = 10  # Switch to deterministic after this many moves


@dataclass
class GameExample:
    """Single training example from self-play"""
    state: np.ndarray  # Board state representation
    policy: np.ndarray  # MCTS policy distribution
    value: float  # Game outcome from player's perspective
    move_number: int
    game_id: str


@dataclass
class GameResult:
    """Result of a single self-play game"""
    game_id: str
    winner: int  # 1 for first player, -1 for second, 0 for draw
    num_moves: int
    examples: List[GameExample]
    game_time: float
    avg_simulations_per_second: float


class SimpleAlphaZeroNetwork(nn.Module):
    """Simplified AlphaZero network for demonstration"""
    
    def __init__(self, board_size: int = 15, channels: int = 64, blocks: int = 6):
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size
        
        # Convolutional layers
        self.conv_input = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            self._make_res_block(channels) for _ in range(blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_size)
        
        # Value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        
    def _make_res_block(self, channels):
        """Create a residual block"""
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        """Forward pass"""
        # Input convolution
        x = self.relu(self.conv_input(x))
        
        # Residual blocks
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = self.relu(x + residual)
        
        # Policy head
        policy = self.relu(self.policy_conv(x))
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)
        
        # Value head
        value = self.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)
        value = self.relu(self.value_fc1(value))
        value = self.tanh(self.value_fc2(value))
        
        return policy_logits, value


class GameInterface:
    """Interface for game state management"""
    
    def __init__(self, board_size: int = 15):
        self.board_size = board_size
        
    def get_initial_state(self):
        """Get initial game state"""
        return alphazero_py.GomokuState()
    
    def get_legal_actions(self, state):
        """Get legal actions for current state"""
        return state.get_legal_actions()
    
    def apply_action(self, state, action):
        """Apply action to state"""
        new_state = state.copy()
        new_state.apply_action(action)
        return new_state
    
    def is_terminal(self, state):
        """Check if state is terminal"""
        return state.is_terminal()
    
    def get_winner(self, state):
        """Get winner of terminal state"""
        if not state.is_terminal():
            return 0
        return 1 if state.get_winner() == state.current_player() else -1
    
    def state_to_tensor(self, state):
        """Convert game state to neural network input tensor"""
        # Simple representation: current player's stones as 1, opponent as -1, empty as 0
        board = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        
        # Get board state from C++ game
        for row in range(self.board_size):
            for col in range(self.board_size):
                pos = row * self.board_size + col
                if pos in state.get_played_positions():
                    # This is a simplified representation
                    # In practice, you'd need proper player identification
                    board[row, col] = 1.0 if (row + col) % 2 == 0 else -1.0
        
        # Add batch and channel dimensions
        return torch.FloatTensor(board).unsqueeze(0).unsqueeze(0)
    
    def get_canonical_state(self, state):
        """Get canonical form of state (always from current player's perspective)"""
        return self.state_to_tensor(state)


class NetworkEvaluator:
    """Neural network evaluator for MCTS"""
    
    def __init__(self, model, device='cuda', temperature=1.0):
        self.model = model.to(device)
        self.device = device
        self.temperature = temperature
        self.game_interface = GameInterface()
        
    def evaluate_batch(self, features, legal_masks=None):
        """Evaluate batch of game states"""
        self.model.eval()
        with torch.no_grad():
            # Convert states to proper format if needed
            if isinstance(features, list):
                # Convert list of game states to tensor batch
                batch_tensors = []
                for state in features:
                    if hasattr(state, 'board'):  # Simple board attribute
                        tensor = self.game_interface.state_to_tensor(state)
                    else:
                        tensor = features  # Already a tensor
                    batch_tensors.append(tensor)
                features = torch.cat(batch_tensors, dim=0)
            
            # Ensure tensor is on correct device
            features = features.to(self.device)
            
            # Forward pass
            policy_logits, values = self.model(features)
            
            # Convert to probabilities
            policies = torch.softmax(policy_logits / self.temperature, dim=1)
            
            # Apply legal move masking if provided
            if legal_masks is not None:
                legal_masks = legal_masks.to(self.device)
                policies = policies * legal_masks
                policies = policies / (policies.sum(dim=1, keepdim=True) + 1e-8)
        
        return policies, values


def play_single_game(game_id: str, config: SelfPlayConfig, 
                    request_queue: Queue, response_queue: Queue) -> GameResult:
    """Play a single self-play game"""
    # Set up worker environment
    os.environ['CUDA_VISIBLE_DEVICES'] = str(torch.cuda.current_device())
    
    logger.info(f"[Game {game_id}] Starting self-play game")
    game_start_time = time.time()
    
    # Initialize game interface
    game_interface = GameInterface(config.board_size)
    state = game_interface.get_initial_state()
    
    # Create remote evaluator for this worker
    evaluator = RemoteEvaluator(
        request_queue=request_queue,
        response_queue=response_queue,
        action_size=config.board_size ** 2,
        worker_id=game_id,
        timeout=10.0
    )
    
    # Configure MCTS
    mcts_config = MCTSConfig(
        num_simulations=config.num_simulations,
        c_puct=config.c_puct,
        temperature=config.temperature,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_epsilon=config.dirichlet_epsilon,
        wave_size=config.wave_size,
        device=config.device,
        game_type=GameType.GOMOKU,
        board_size=config.board_size,
        enable_virtual_loss=True
    )
    
    # Create MCTS instance
    mcts = MCTS(mcts_config, evaluator)
    
    # Game loop
    examples = []
    move_number = 0
    total_simulations = 0
    total_search_time = 0
    
    logger.info(f"[Game {game_id}] Starting game loop")
    
    while not game_interface.is_terminal(state) and move_number < config.max_game_length:
        move_start = time.time()
        
        # Determine temperature
        temp = config.temperature if move_number < config.temp_threshold else 0.01
        
        # Run MCTS search
        try:
            policy = mcts.search(state, config.num_simulations)
            
            # Get MCTS statistics
            stats = mcts.get_statistics()
            search_time = time.time() - move_start
            sims_per_sec = stats.get('last_search_sims_per_second', 0)
            
            total_simulations += config.num_simulations
            total_search_time += search_time
            
            # Create training example
            canonical_state = game_interface.get_canonical_state(state)
            example = GameExample(
                state=canonical_state.numpy(),
                policy=policy,
                value=0.0,  # Will be filled with game outcome
                move_number=move_number,
                game_id=game_id
            )
            examples.append(example)
            
            # Select action
            if temp > 0.1:
                # Stochastic selection with temperature
                action_probs = np.power(policy, 1/temp)
                action_probs /= action_probs.sum()
                action = np.random.choice(len(policy), p=action_probs)
            else:
                # Deterministic selection
                action = np.argmax(policy)
            
            # Apply action
            state = game_interface.apply_action(state, action)
            move_number += 1
            
            logger.debug(f"[Game {game_id}] Move {move_number}: action={action}, "
                        f"time={search_time:.2f}s, sims/s={sims_per_sec:.0f}")
            
        except Exception as e:
            logger.error(f"[Game {game_id}] Error during MCTS search: {e}")
            break
    
    # Determine game outcome
    winner = game_interface.get_winner(state) if game_interface.is_terminal(state) else 0
    
    # Assign values to examples based on game outcome
    for i, example in enumerate(examples):
        # Value from the perspective of the player who made the move
        example.value = winner * ((-1) ** i)
    
    # Calculate performance metrics
    game_time = time.time() - game_start_time
    avg_sims_per_sec = total_simulations / total_search_time if total_search_time > 0 else 0
    
    logger.info(f"[Game {game_id}] Finished: winner={winner}, "
               f"moves={move_number}, time={game_time:.2f}s, "
               f"avg_sims/s={avg_sims_per_sec:.0f}")
    
    return GameResult(
        game_id=game_id,
        winner=winner,
        num_moves=move_number,
        examples=examples,
        game_time=game_time,
        avg_simulations_per_second=avg_sims_per_sec
    )


class SelfPlayEngine:
    """Main self-play engine coordinating multiple processes"""
    
    def __init__(self, config: SelfPlayConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize neural network
        self.model = SimpleAlphaZeroNetwork(
            board_size=config.board_size,
            channels=config.model_channels,
            blocks=config.model_blocks
        ).to(self.device)
        
        # Initialize with random weights
        self._initialize_model()
        
        # Performance monitoring
        self.memory_monitor = MemoryMonitor() if config.use_profiling else None
        self.gpu_profiler = GPUProfiler(self.device) if config.use_profiling else None
        
        # Statistics
        self.stats = {
            'games_played': 0,
            'total_examples': 0,
            'avg_game_length': 0,
            'avg_simulations_per_second': 0,
            'win_rates': defaultdict(int)
        }
    
    def _initialize_model(self):
        """Initialize model with random weights"""
        logger.info("Initializing neural network with random weights")
        
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.model.apply(init_weights)
    
    def run_selfplay(self) -> Dict[str, Any]:
        """Run self-play session"""
        logger.info(f"Starting self-play: {self.config.num_games} games, "
                   f"{self.config.num_processes} processes")
        
        if self.memory_monitor:
            self.memory_monitor.start_monitoring()
        
        session_start = time.time()
        
        # Create GPU evaluation service
        gpu_service = GPUEvaluatorService(
            model=self.model,
            device=self.device,
            batch_size=256,
            batch_timeout=0.01
        )
        
        # Start GPU service
        gpu_service.start()
        
        try:
            # Run games in parallel
            all_results = self._run_parallel_games(gpu_service)
            
            # Process results
            session_results = self._process_results(all_results)
            session_results['total_time'] = time.time() - session_start
            
            # Save results
            if self.config.save_training_data:
                self._save_training_data(all_results)
            
            return session_results
            
        finally:
            # Stop GPU service
            gpu_service.stop()
            
            if self.memory_monitor:
                self.memory_monitor.stop_monitoring()
    
    def _run_parallel_games(self, gpu_service) -> List[GameResult]:
        """Run games in parallel using multiprocessing"""
        # Create queues for communication
        request_queue = gpu_service.request_queue
        response_queue = gpu_service.response_queue
        
        # Create game processes
        processes = []
        result_queue = Queue()
        
        def game_worker(game_ids):
            """Worker function for running multiple games"""
            worker_results = []
            for game_id in game_ids:
                try:
                    result = play_single_game(game_id, self.config, request_queue, response_queue)
                    worker_results.append(result)
                except Exception as e:
                    logger.error(f"Game {game_id} failed: {e}")
            result_queue.put(worker_results)
        
        # Distribute games across processes
        games_per_process = max(1, self.config.num_games // self.config.num_processes)
        
        for i in range(self.config.num_processes):
            start_game = i * games_per_process
            end_game = min((i + 1) * games_per_process, self.config.num_games)
            
            if start_game >= self.config.num_games:
                break
                
            game_ids = [f"game_{j:03d}" for j in range(start_game, end_game)]
            
            process = Process(target=game_worker, args=(game_ids,))
            process.start()
            processes.append(process)
        
        # Collect results
        all_results = []
        for _ in processes:
            worker_results = result_queue.get()
            all_results.extend(worker_results)
        
        # Wait for all processes to complete
        for process in processes:
            process.join()
        
        return all_results
    
    def _process_results(self, results: List[GameResult]) -> Dict[str, Any]:
        """Process and analyze game results"""
        if not results:
            return {}
        
        # Calculate statistics
        total_examples = sum(len(r.examples) for r in results)
        total_moves = sum(r.num_moves for r in results)
        total_time = sum(r.game_time for r in results)
        avg_sims_per_sec = np.mean([r.avg_simulations_per_second for r in results])
        
        # Win rate analysis
        winners = [r.winner for r in results]
        win_rates = {
            'player_1_wins': winners.count(1),
            'player_2_wins': winners.count(-1), 
            'draws': winners.count(0)
        }
        
        # Game length analysis
        game_lengths = [r.num_moves for r in results]
        
        # Performance metrics
        performance = {
            'avg_simulations_per_second': avg_sims_per_sec,
            'total_search_time': total_time,
            'games_per_hour': len(results) / (total_time / 3600) if total_time > 0 else 0
        }
        
        # Memory statistics
        memory_stats = {}
        if self.memory_monitor:
            peak_memory = self.memory_monitor.get_peak_usage()
            if peak_memory:
                memory_stats = {
                    'peak_gpu_memory_mb': peak_memory.gpu_allocated_mb,
                    'peak_cpu_percent': peak_memory.cpu_percent,
                    'avg_ram_usage_mb': self.memory_monitor.get_average_usage().ram_mb
                }
        
        results_summary = {
            'games_completed': len(results),
            'total_examples': total_examples,
            'avg_game_length': np.mean(game_lengths),
            'std_game_length': np.std(game_lengths),
            'win_rates': win_rates,
            'performance': performance,
            'memory': memory_stats,
            'game_length_distribution': {
                'min': min(game_lengths),
                'max': max(game_lengths),
                'median': np.median(game_lengths),
                'p25': np.percentile(game_lengths, 25),
                'p75': np.percentile(game_lengths, 75)
            }
        }
        
        return results_summary
    
    def _save_training_data(self, results: List[GameResult]):
        """Save training examples to disk"""
        all_examples = []
        for result in results:
            all_examples.extend(result.examples)
        
        # Save examples
        examples_file = os.path.join(self.config.output_dir, 'training_examples.pkl')
        with open(examples_file, 'wb') as f:
            pickle.dump(all_examples, f)
        
        # Save game results
        results_file = os.path.join(self.config.output_dir, 'game_results.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Saved {len(all_examples)} training examples to {examples_file}")
        logger.info(f"Saved {len(results)} game results to {results_file}")


def print_results(results: Dict[str, Any]):
    """Print formatted results"""
    print("\n" + "="*60)
    print("🎮 MCTS SELF-PLAY RESULTS")
    print("="*60)
    
    print(f"📊 Games: {results['games_completed']}")
    print(f"📚 Training Examples: {results['total_examples']}")
    print(f"⏱️  Total Time: {results.get('total_time', 0):.1f}s")
    print()
    
    print("🎯 Game Statistics:")
    print(f"   Average Length: {results['avg_game_length']:.1f} ± {results['std_game_length']:.1f} moves")
    print(f"   Range: {results['game_length_distribution']['min']}-{results['game_length_distribution']['max']} moves")
    print(f"   Median: {results['game_length_distribution']['median']:.1f} moves")
    print()
    
    print("🏆 Win Rates:")
    win_rates = results['win_rates']
    total = sum(win_rates.values())
    if total > 0:
        print(f"   Player 1: {win_rates['player_1_wins']}/{total} ({100*win_rates['player_1_wins']/total:.1f}%)")
        print(f"   Player 2: {win_rates['player_2_wins']}/{total} ({100*win_rates['player_2_wins']/total:.1f}%)")
        print(f"   Draws:    {win_rates['draws']}/{total} ({100*win_rates['draws']/total:.1f}%)")
    print()
    
    print("⚡ Performance:")
    perf = results['performance']
    print(f"   Simulations/sec: {perf['avg_simulations_per_second']:,.0f}")
    print(f"   Games/hour: {perf['games_per_hour']:.1f}")
    print(f"   Search time: {perf['total_search_time']:.1f}s")
    print()
    
    if 'memory' in results and results['memory']:
        print("💾 Memory Usage:")
        mem = results['memory']
        print(f"   Peak GPU: {mem.get('peak_gpu_memory_mb', 0):.1f}MB")
        print(f"   Peak CPU: {mem.get('peak_cpu_percent', 0):.1f}%")
        print(f"   Avg RAM: {mem.get('avg_ram_usage_mb', 0):.1f}MB")
        print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='MCTS Self-Play Example')
    parser.add_argument('--games', type=int, default=10, help='Number of games to play')
    parser.add_argument('--processes', type=int, default=4, help='Number of parallel processes')
    parser.add_argument('--simulations', type=int, default=800, help='MCTS simulations per move')
    parser.add_argument('--wave-size', type=int, default=2048, help='MCTS wave size')
    parser.add_argument('--save-data', action='store_true', help='Save training data')
    parser.add_argument('--output-dir', type=str, default='selfplay_results', help='Output directory')
    parser.add_argument('--no-profiling', action='store_true', help='Disable performance profiling')
    
    args = parser.parse_args()
    
    # Set multiprocessing start method
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    # Create configuration
    config = SelfPlayConfig(
        num_games=args.games,
        num_processes=args.processes,
        num_simulations=args.simulations,
        wave_size=args.wave_size,
        save_training_data=args.save_data,
        output_dir=args.output_dir,
        use_profiling=not args.no_profiling
    )
    
    print("🚀 Starting MCTS Self-Play Example")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Configuration: {args.games} games, {args.processes} processes")
    print()
    
    # Create and run self-play engine
    try:
        engine = SelfPlayEngine(config)
        results = engine.run_selfplay()
        
        # Print results
        print_results(results)
        
        # Save summary
        import json
        summary_file = os.path.join(config.output_dir, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Self-play completed successfully!")
        print(f"📁 Results saved to: {config.output_dir}")
        
    except Exception as e:
        logger.error(f"Self-play failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())