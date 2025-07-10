"""Self-play manager for AlphaZero training

This module handles self-play data generation including:
- Single-GPU game execution
- MCTS integration
- Training example collection
- Game metrics tracking
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Type, Tuple
import numpy as np
from tqdm import tqdm

from mcts.neural_networks.replay_buffer import GameExample
from mcts.core.mcts import MCTS
from mcts.core.mcts_config import MCTSConfig
from mcts.core.game_interface import GameInterface
from mcts.utils.config_system import AlphaZeroConfig
from mcts.utils.direct_gpu_evaluator import DirectGPUEvaluator


logger = logging.getLogger(__name__)


@dataclass
class SelfPlayConfig:
    """Configuration for self-play"""
    num_games_per_iteration: int = 100
    mcts_simulations: int = 800
    temperature: float = 1.0
    temperature_threshold: int = 30
    resign_threshold: float = -0.98
    enable_resign: bool = True
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    c_puct: float = 1.5
    enable_progress_bar: bool = True
    batch_size: int = 512  # Optimized for GPU batching
    device: str = 'cuda'  # GPU device for single-GPU execution


class SelfPlayGame:
    """Manages a single self-play game"""
    
    def __init__(self, game, mcts, config: SelfPlayConfig, game_id: str):
        """Initialize self-play game
        
        Args:
            game: Game instance
            mcts: Shared MCTS instance
            config: Self-play configuration
            game_id: Unique game identifier
        """
        self.game = game
        self.mcts = mcts  # Use shared MCTS instance
        self.config = config
        self.game_id = game_id
        self.examples = []
        self.current_player = 1
        self.move_count = 0
        
        # Metrics tracking
        self.resigned = False
        self.policy_entropies = []
        self.value_predictions = []
        self.illegal_move_attempts = 0
        
        # Check if game has game_interface attribute (GameWrapper case)
        if hasattr(game, 'game_interface'):
            self.game_interface = game.game_interface
        else:
            # Regular game case - create GameInterface
            self.game_interface = GameInterface(game)
    
    def play_single_move(self):
        """Play a single move in the game"""
        # Get current state
        state = self.game.get_state()
        # GameWrapper.get_current_player() doesn't take parameters
        self.current_player = self.game.get_current_player()
        
        # Run MCTS search to get policy
        action_probs = self.mcts.search(state)
        
        
        # Calculate and store policy entropy
        # Entropy = -sum(p * log(p)) for p > 0
        entropy = -np.sum(action_probs[action_probs > 0] * np.log(action_probs[action_probs > 0] + 1e-10))
        self.policy_entropies.append(entropy)
        
        # Log low entropy situations which might indicate overconfidence
        # Low entropy situations tracked internally
        
        # Get value prediction from MCTS (if available)
        if hasattr(self.mcts, 'tree') and hasattr(self.mcts.tree, 'get_node_value'):
            value_pred = self.mcts.tree.get_node_value(0)  # Root node value
            self.value_predictions.append((value_pred, self.current_player))
            # Value prediction tracked
        else:
            value_pred = 0.0  # Neutral value when no prediction available (don't resign)
            # No value prediction available
        
        # Store training example with numpy array state
        # Convert state to numpy array for training compatibility
        state_array = self.game_interface.state_to_numpy(state)
        self.examples.append({
            'state': state_array,
            'policy': action_probs.copy(),
            'player': self.current_player,
            'move_number': self.move_count,
            'game_move_number': self.move_count  # Keep track of actual game move for win calculation
        })
        
        # Select action based on temperature
        temperature = self._get_temperature()
        if temperature > 0:
            # Sample from distribution
            action = np.random.choice(len(action_probs), p=action_probs)
        else:
            # Select best action
            action = np.argmax(action_probs)
        
        # Validate action is legal
        valid_actions = self.game.get_valid_actions()
        # valid_actions is a list of legal move indices, not a boolean mask
        if action not in valid_actions:
            self.illegal_move_attempts += 1
            logger.error(f"Illegal move attempted in game {self.game_id}: action {action}, valid_actions: {valid_actions[:10]}...")
            # Find a legal action instead
            if len(valid_actions) == 0:
                logger.error(f"No legal actions available in game {self.game_id}!")
                raise RuntimeError("No legal actions available")
            action = np.random.choice(valid_actions)
        
        self.game.make_action(action)
        self.move_count += 1
        
        # Check for resignation
        if self.config.enable_resign and self.move_count > 10:
            # Use value_pred from above (now always defined)
            if value_pred < self.config.resign_threshold:
                self.resigned = True
                # Player resigned based on value threshold
                return True  # Resign
            else:
                logger.debug(f"Game {self.game_id}: Move {self.move_count}, Player {self.current_player}, "
                            f"value_pred={value_pred:.4f} >= threshold={self.config.resign_threshold}, continuing...")
        
        return False
    
    def play_game(self) -> Tuple[List[GameExample], Dict[str, Any]]:
        """Play a complete game and return training examples and metrics
        
        Returns:
            Tuple of (game_examples, metrics_dict)
        """
        resigned = False
        
        while not self.game.is_terminal():
            resigned = self.play_single_move()
            if resigned:
                break
        
        # Get final outcome
        if resigned:
            # Current player resigned, so they lost
            # If player 1 resigned, player 2 wins (winner = -1)
            # If player 2 resigned, player 1 wins (winner = 1)
            # current_player is now 1-based (1 or 2)
            winner = -1 if self.current_player == 1 else 1
            pass  # Player resigned
        else:
            winner = self.game.get_winner()
            pass  # Game ended naturally
            
        # Game termination details tracked internally
        
        # Convert examples to GameExample objects with correct values
        game_examples = []
        for i, example in enumerate(self.examples):
            # Value is from the perspective of the player who made the move
            # winner is 1 for P1 win, -1 for P2 win, 0 for draw
            # Players are numbered 1 and 2
            if example['player'] == 1:
                value = winner  # P1 gets +1 if they won, -1 if they lost
            else:  # player == 2
                value = -winner  # P2 gets +1 if they won (winner=-1), -1 if they lost (winner=1)
            
            # Value assignment tracked internally
            
            game_examples.append(GameExample(
                state=example['state'],
                policy=example['policy'],
                value=float(value),
                game_id=self.game_id,
                move_number=example['move_number']
            ))
        
        # Calculate value accuracy for this game
        value_accuracies = []
        if self.value_predictions:
            for value_pred, player in self.value_predictions:
                # Actual value from player's perspective
                # Players are numbered 1 and 2, winner is 1/-1/0
                if player == 1:
                    actual_value = winner
                else:  # player == 2
                    actual_value = -winner
                # Accuracy is 1 if prediction has same sign as actual, 0 otherwise
                accuracy = 1.0 if (value_pred > 0) == (actual_value > 0) else 0.0
                value_accuracies.append(accuracy)
        
        # Prepare metrics for this game
        game_metrics = {
            'resigned': self.resigned,
            'policy_entropies': self.policy_entropies,
            'value_accuracies': value_accuracies,
            'illegal_move_attempts': self.illegal_move_attempts,
            'game_length': len(self.examples),
            'winner': winner
        }
        
        return game_examples, game_metrics
    
    def _get_temperature(self) -> float:
        """Get temperature for current move"""
        if self.move_count < self.config.temperature_threshold:
            return self.config.temperature
        return 0.0


class SelfPlayManager:
    """Manages self-play data generation"""
    
    def __init__(self, config: AlphaZeroConfig, game_class: Type, evaluator,
                 game_config: Optional[Dict[str, Any]] = None):
        """Initialize self-play manager
        
        Args:
            config: AlphaZero configuration
            game_class: Game class to instantiate
            evaluator: Neural network evaluator
            game_config: Configuration for game creation (for GameInterface wrapper)
        """
        self.config = config
        self.game_class = game_class
        self.evaluator = evaluator
        self.game_config = game_config
        
        # Metrics collection
        self.collected_metrics = []
        
        # Create self-play config from AlphaZero config
        self.self_play_config = SelfPlayConfig(
            num_games_per_iteration=config.training.num_games_per_iteration,
            mcts_simulations=config.mcts.num_simulations,
            temperature=config.mcts.temperature,
            temperature_threshold=config.mcts.temperature_threshold,
            resign_threshold=getattr(config.mcts, 'resign_threshold', -0.98),
            enable_resign=getattr(config.mcts, 'enable_resign', True),
            dirichlet_alpha=config.mcts.dirichlet_alpha,
            dirichlet_epsilon=config.mcts.dirichlet_epsilon,
            c_puct=config.mcts.c_puct,
            batch_size=getattr(config.mcts, 'batch_size', 512),
            device=config.mcts.device  # Use device from MCTS config
        )
        
        # Create direct GPU evaluator if evaluator is a model
        if hasattr(evaluator, 'model'):
            self.gpu_evaluator = DirectGPUEvaluator(
                model=evaluator.model,
                device=self.self_play_config.device,
                action_size=config.game.board_size ** 2,
                batch_size=config.resources.inference_batch_size,
                use_mixed_precision=getattr(config.mcts, 'use_mixed_precision', True)
            )
        else:
            # Evaluator is already a direct evaluator
            self.gpu_evaluator = evaluator
    
    def generate_self_play_data(self) -> List[GameExample]:
        """Generate self-play training data using single GPU
        
        Returns:
            List of training examples
        """
        logger.info(f"Generating {self.self_play_config.num_games_per_iteration} self-play games on {self.self_play_config.device}")
        
        # Always use single-GPU execution
        return self._generate_single_gpu()
    
    def _generate_single_gpu(self) -> List[GameExample]:
        """Generate self-play data using single GPU"""
        all_examples = []
        self.collected_metrics = []
        
        # Create shared MCTS instance with optimized config
        mcts_config = MCTSConfig(
            num_simulations=self.self_play_config.mcts_simulations,
            c_puct=self.self_play_config.c_puct,
            dirichlet_alpha=self.self_play_config.dirichlet_alpha,
            dirichlet_epsilon=self.self_play_config.dirichlet_epsilon,
            tree_batch_size=512,  # Optimized batch size for GPU
            device=self.self_play_config.device,
            enable_fast_ucb=True,
            use_cuda_graphs=True,
            use_mixed_precision=True,
            enable_subtree_reuse=False,  # Disable to prevent memory issues
            max_tree_nodes=self.config.mcts.max_tree_nodes,
            memory_pool_size_mb=self.config.mcts.memory_pool_size_mb
        )
        
        # Create game interface for MCTS (use first game as template)
        template_game = self._create_game_instance()
        if hasattr(template_game, 'game_interface'):
            game_interface = template_game.game_interface
        else:
            game_interface = GameInterface(template_game)
        
        # Create shared MCTS instance
        shared_mcts = MCTS(
            config=mcts_config,
            evaluator=self.gpu_evaluator,
            game_interface=game_interface
        )
        
        # Progress bar
        if self.self_play_config.enable_progress_bar:
            pbar = tqdm(
                total=self.self_play_config.num_games_per_iteration,
                desc="Self-play games (GPU)",
                leave=False
            )
        
        for game_idx in range(self.self_play_config.num_games_per_iteration):
            # Create new game instance
            game = self._create_game_instance()
            
            # Reset MCTS tree for new game
            shared_mcts.reset_tree()
            
            # Play game with shared MCTS
            sp_game = SelfPlayGame(
                game=game,
                mcts=shared_mcts,
                config=self.self_play_config,
                game_id=f"game_{game_idx}"
            )
            
            examples, metrics = sp_game.play_game()
            all_examples.extend(examples)
            self.collected_metrics.append(metrics)
            
            if self.self_play_config.enable_progress_bar:
                pbar.update(1)
        
        if self.self_play_config.enable_progress_bar:
            pbar.close()
        
        # Log GPU evaluator statistics
        if hasattr(self.gpu_evaluator, 'get_statistics'):
            stats = self.gpu_evaluator.get_statistics()
            logger.info(f"GPU Evaluator Stats: {stats}")
        
        return all_examples
    
    def _generate_multi_process_deprecated(self) -> List[GameExample]:
        """Generate self-play data using multiple processes with GPU service"""
        # Import GPU service components
        from mcts.utils.gpu_evaluator_service import GPUEvaluatorService
        from mcts.utils.optimized_remote_evaluator import OptimizedRemoteEvaluator
        
        # Set multiprocessing start method for CUDA
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
        
        # Create GPU service in main process with configurable settings
        gpu_service = GPUEvaluatorService(
            model=self.evaluator.model,
            device=self.evaluator.device,
            batch_size=self.config.resources.inference_batch_size,
            batch_timeout=self.self_play_config.gpu_batch_timeout,
            workload_type="inference"
        )
        
        # Start GPU service
        gpu_service.start()
        logger.info("GPU service started for self-play")
        
        try:
            # Get request queue
            request_queue = gpu_service.get_request_queue()
            
            # Calculate work distribution
            games_per_worker = self.self_play_config.num_games_per_iteration // self.self_play_config.num_workers
            remainder = self.self_play_config.num_games_per_iteration % self.self_play_config.num_workers
            
            work_distribution = []
            for i in range(self.self_play_config.num_workers):
                num_games = games_per_worker + (1 if i < remainder else 0)
                if num_games > 0:
                    work_distribution.append((i, num_games))
            
            # Create result queue
            result_queue = mp.Queue()
            
            # Start worker processes
            processes = []
            for worker_id, num_games in work_distribution:
                # Create response queue for this worker
                response_queue = gpu_service.create_worker_queue(worker_id)
                
                # Start worker process
                p = mp.Process(
                    target=_worker_process_with_gpu_service,
                    args=(
                        worker_id,
                        num_games,
                        self.game_config,
                        self.self_play_config,
                        request_queue,
                        response_queue,
                        result_queue,
                        self.config.game.board_size ** 2
                    )
                )
                p.start()
                processes.append((p, worker_id))
            
            # Collect results
            all_examples = []
            workers_completed = 0
            
            if self.self_play_config.enable_progress_bar:
                pbar = tqdm(
                    total=self.self_play_config.num_games_per_iteration,
                    desc="Self-play games",
                    leave=False
                )
            
            while workers_completed < len(processes):
                try:
                    msg_type, data = result_queue.get(timeout=5.0)
                    
                    if msg_type == 'progress':
                        # Progress update
                        if self.self_play_config.enable_progress_bar:
                            pbar.update(1)
                            
                    elif msg_type == 'results':
                        # Results from worker
                        all_examples.extend(data)
                        workers_completed += 1
                        
                    elif msg_type == 'error':
                        # Error from worker
                        logger.error(f"Worker error: {data}")
                        workers_completed += 1
                        
                except:
                    # Check if processes are still alive
                    for p, worker_id in processes:
                        if not p.is_alive() and p.exitcode != 0:
                            logger.error(f"Worker {worker_id} died with exit code {p.exitcode}")
                            workers_completed += 1
            
            # Wait for all processes
            for p, worker_id in processes:
                p.join()
                gpu_service.cleanup_worker_queue(worker_id)
                
            if self.self_play_config.enable_progress_bar:
                pbar.close()
                
            return all_examples
            
        finally:
            # Always stop GPU service
            gpu_service.stop()
            logger.info("GPU service stopped")
    
    def _create_game_instance(self):
        """Create a game instance, handling both regular games and GameInterface wrapper"""
        if self.game_config:
            # For GameInterface wrapper, we need to create it with config
            from mcts.core.game_interface import GameInterface, GameType
            
            game_config = self.game_config  # Capture in local variable
            
            class GameWrapper:
                def __init__(self):
                    game_type = GameType[game_config['game_type'].upper()]
                    self.game_interface = GameInterface(
                        game_type,
                        board_size=game_config['board_size'],
                        input_representation=game_config['input_representation']
                    )
                    self.state = self.game_interface.create_initial_state()
                    self.action_size = game_config['board_size'] * game_config['board_size']
                    
                def get_state(self):
                    return self.state
                    
                def get_current_player(self):
                    return self.game_interface.get_current_player(self.state)
                    
                def get_valid_actions(self):
                    return self.game_interface.get_legal_moves(self.state)
                    
                def is_terminal(self):
                    return self.game_interface.is_terminal(self.state)
                    
                def make_action(self, action):
                    self.state = self.game_interface.get_next_state(self.state, action)
                    
                def get_winner(self):
                    return self.game_interface.get_winner(self.state)
                    
                def get_action_size(self):
                    return self.action_size
            
            return GameWrapper()
        else:
            # Regular game class
            return self.game_class()
    
    def _play_games_worker(self, worker_id: int, num_games: int) -> List[GameExample]:
        """Worker function for parallel self-play
        
        Args:
            worker_id: Worker identifier
            num_games: Number of games to play
            
        Returns:
            List of training examples
        """
        examples = []
        
        for game_idx in range(num_games):
            # Create new game instance
            game = self._create_game_instance()
            
            # Play game
            sp_game = SelfPlayGame(
                game=game,
                evaluator=self.evaluator,
                config=self.self_play_config,
                game_id=f"worker_{worker_id}_game_{game_idx}"
            )
            
            game_examples = sp_game.play_game()
            examples.extend(game_examples)
        
        return examples
    
    def collect_game_metrics(self, examples: List[GameExample]) -> Dict[str, Any]:
        """Collect metrics from self-play games
        
        Args:
            examples: List of training examples
            
        Returns:
            Dictionary of metrics
        """
        if not examples:
            return {
                'total_games': 0,
                'total_moves': 0,
                'avg_game_length': 0,
                'min_game_length': 0,
                'max_game_length': 0,
                'unique_games': 0,
                'player1_wins': 0,
                'player2_wins': 0,
                'draws': 0,
                'resignation_count': 0,
                'avg_entropy': 0,
                'value_accuracy': 0,
                'illegal_move_attempts': 0
            }
        
        # Group examples by game
        games = {}
        for ex in examples:
            game_id = ex.game_id
            if game_id not in games:
                games[game_id] = []
            games[game_id].append(ex)
        
        # Calculate metrics from collected_metrics if available
        if hasattr(self, 'collected_metrics') and self.collected_metrics:
            # Aggregate metrics from individual games
            resignation_count = sum(1 for m in self.collected_metrics if m['resigned'])
            
            # Collect all policy entropies
            all_entropies = []
            for m in self.collected_metrics:
                all_entropies.extend(m['policy_entropies'])
            avg_entropy = np.mean(all_entropies) if all_entropies else 0
            
            # Collect all value accuracies
            all_value_accs = []
            for m in self.collected_metrics:
                all_value_accs.extend(m['value_accuracies'])
            value_accuracy = np.mean(all_value_accs) if all_value_accs else 0
            
            # Sum illegal move attempts
            illegal_move_attempts = sum(m['illegal_move_attempts'] for m in self.collected_metrics)
        else:
            # Fallback if metrics weren't collected
            resignation_count = 0
            avg_entropy = 0
            value_accuracy = 0
            illegal_move_attempts = 0
        
        # Calculate game lengths
        game_lengths = [len(game_examples) for game_examples in games.values()]
        
        # Count outcomes
        player1_wins = 0
        player2_wins = 0
        draws = 0
        
        # Use collected metrics if available for accurate win counting
        if hasattr(self, 'collected_metrics') and self.collected_metrics:
            for i, game_examples in enumerate(games.values()):
                if i < len(self.collected_metrics):
                    # Use the actual winner from metrics
                    winner = self.collected_metrics[i]['winner']
                    game_id = game_examples[0].game_id if game_examples else f"game_{i}"
                    
                    if winner == 0:  # Draw
                        draws += 1
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Game {game_id}: Draw (winner={winner})")
                    elif winner == 1:  # Player 1 (BLACK) wins
                        player1_wins += 1
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Game {game_id}: Player 1 wins (winner={winner})")
                    elif winner == -1:  # Player 2 (WHITE) wins  
                        player2_wins += 1
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Game {game_id}: Player 2 wins (winner={winner})")
                    else:
                        logger.warning(f"Game {game_id}: Unknown winner value {winner}")
                        draws += 1
        else:
            # Fallback to old logic if metrics not available
            for game_examples in games.values():
                # Get final outcome from last example
                final_value = game_examples[-1].value
                
                if abs(final_value) < 0.01:  # Draw threshold
                    draws += 1
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Game {game_examples[0].game_id}: Draw (final_value={final_value})")
                elif final_value > 0:
                    # Final player won - Player 1 starts, so even moves are Player 1, odd moves are Player 2
                    final_move_number = game_examples[-1].move_number
                    if final_move_number % 2 == 0:
                        # Even move number = Player 1 made this move and won
                        player1_wins += 1
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Game {game_examples[0].game_id}: Player 1 wins "
                                       f"(final_move={final_move_number}, final_value={final_value})")
                    else:
                        # Odd move number = Player 2 made this move and won
                        player2_wins += 1
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Game {game_examples[0].game_id}: Player 2 wins "
                                       f"(final_move={final_move_number}, final_value={final_value})")
                else:
                    # Final player lost - the other player won
                    final_move_number = game_examples[-1].move_number
                    if final_move_number % 2 == 0:
                        # Player 1 made the final move and lost, so Player 2 won
                        player2_wins += 1
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Game {game_examples[0].game_id}: Player 2 wins "
                                       f"(P1 lost, final_move={final_move_number}, final_value={final_value})")
                    else:
                        # Player 2 made the final move and lost, so Player 1 won
                        player1_wins += 1
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Game {game_examples[0].game_id}: Player 1 wins "
                                       f"(P2 lost, final_move={final_move_number}, final_value={final_value})")
        
        return {
            'total_games': len(games),
            'total_moves': len(examples),
            'avg_game_length': np.mean(game_lengths) if game_lengths else 0,
            'min_game_length': min(game_lengths) if game_lengths else 0,
            'max_game_length': max(game_lengths) if game_lengths else 0,
            'unique_games': len(games),
            'player1_wins': player1_wins,
            'player2_wins': player2_wins,
            'draws': draws,
            'win_rate_p1': player1_wins / len(games) if games else 0,
            'win_rate_p2': player2_wins / len(games) if games else 0,
            'draw_rate': draws / len(games) if games else 0,
            'resignation_count': resignation_count,
            'avg_entropy': avg_entropy,
            'value_accuracy': value_accuracy,
            'illegal_move_attempts': illegal_move_attempts
        }


def _worker_process_with_gpu_service_deprecated(worker_id: int,
                                     num_games: int,
                                     game_config: Dict[str, Any],
                                     self_play_config: SelfPlayConfig,
                                     request_queue,
                                     response_queue,
                                     result_queue,
                                     action_size: int):
    """Worker process for self-play with GPU service"""
    try:
        # Import here to avoid issues in main process
        from mcts.utils.optimized_remote_evaluator import OptimizedRemoteEvaluator
        from mcts.core.game_interface import GameInterface, GameType
        
        # Create evaluator that communicates with GPU service with configurable settings
        evaluator = OptimizedRemoteEvaluator(
            request_queue=request_queue,
            response_queue=response_queue,
            action_size=action_size,
            worker_id=worker_id,
            batch_timeout=self_play_config.worker_batch_timeout,
            enable_coordination=True,
            max_coordination_batch_size=self_play_config.max_coordination_batch_size
        )
        
        # Play games
        all_examples = []
        for game_idx in range(num_games):
            # Create game instance
            game_type = GameType[game_config['game_type'].upper()]
            
            class GameWrapper:
                def __init__(self):
                    self.game_interface = GameInterface(
                        game_type,
                        board_size=game_config['board_size'],
                        input_representation=game_config['input_representation']
                    )
                    self.state = self.game_interface.create_initial_state()
                    self.action_size = game_config['board_size'] * game_config['board_size']
                    
                def get_state(self):
                    return self.state
                    
                def get_current_player(self):
                    return self.game_interface.get_current_player(self.state)
                    
                def get_valid_actions(self):
                    return self.game_interface.get_legal_moves(self.state)
                    
                def is_terminal(self):
                    return self.game_interface.is_terminal(self.state)
                    
                def make_action(self, action):
                    self.state = self.game_interface.get_next_state(self.state, action)
                    
                def get_winner(self):
                    return self.game_interface.get_winner(self.state)
                    
                def get_action_size(self):
                    return self.action_size
            
            game = GameWrapper()
            
            # Play game using SelfPlayGame
            sp_game = SelfPlayGame(
                game=game,
                evaluator=evaluator,
                config=self_play_config,
                game_id=f"worker_{worker_id}_game_{game_idx}"
            )
            
            examples, metrics = sp_game.play_game()
            all_examples.extend(examples)
            
            # Send progress update
            result_queue.put(('progress', len(examples)))
        
        # Send final results
        result_queue.put(('results', all_examples))
        
    except Exception as e:
        logger.error(f"Worker {worker_id} error: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put(('error', str(e)))