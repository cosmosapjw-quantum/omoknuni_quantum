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
from mcts.utils.single_gpu_evaluator import SingleGPUEvaluator


logger = logging.getLogger(__name__)


# Module-level worker function for multiprocessing
def _worker_generate_games(worker_id: int, num_games: int, config: Any, 
                           self_play_config, game_config) -> Tuple[List[GameExample], List[Dict]]:
    """Worker function to generate games in parallel"""
    # Import here to avoid pickling issues
    import numpy as np
    import torch
    from mcts.core.mcts import MCTS
    from mcts.core.mcts_config import MCTSConfig
    from mcts.utils.single_gpu_evaluator import SingleGPUEvaluator
    
    # Set random seed for this worker
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)
    
    worker_examples = []
    worker_metrics = []
    
    # Create evaluator for this worker
    # For testing, create a dummy model
    from mcts.neural_networks.resnet_model import ResNetModel, ResNetConfig
    
    # Convert config to ResNetConfig
    resnet_config = ResNetConfig()
    for key, value in vars(config.network).items():
        if hasattr(resnet_config, key):
            setattr(resnet_config, key, value)
    
    # Handle game_config as either object or dict
    if isinstance(game_config, dict):
        board_size = game_config['board_size']
        game_type = game_config['game_type']
    else:
        board_size = game_config.board_size
        game_type = game_config.game_type
    
    # Calculate action size based on game type
    if game_type == 'gomoku':
        action_size = board_size * board_size  # 15*15 = 225
    elif game_type == 'go':
        action_size = board_size * board_size + 1  # 19*19 + 1 = 362 (including pass)
    elif game_type == 'chess':
        action_size = 4096  # Upper bound for chess moves
    else:
        action_size = board_size * board_size  # Default
    
    # Create model
    model = ResNetModel(
        config=resnet_config,
        board_size=board_size,
        num_actions=action_size,
        game_type=game_type
    )
    
    # Create evaluator - always use CPU for workers to avoid CUDA fork issues
    evaluator = SingleGPUEvaluator(
        model=model,
        device='cpu',  # Force CPU for multiprocessing workers
        action_size=action_size,
        batch_size=self_play_config.batch_size,
        use_mixed_precision=False  # Disable for CPU
    )
    
    # Create MCTS config - force CPU for workers
    backend = getattr(config.mcts, 'backend', 'cpu')
    mcts_config = MCTSConfig(
        num_simulations=self_play_config.mcts_simulations,
        c_puct=self_play_config.c_puct,
        dirichlet_alpha=self_play_config.dirichlet_alpha,
        dirichlet_epsilon=self_play_config.dirichlet_epsilon,
        tree_batch_size=128 if backend == 'cpu' else 512,
        device='cpu',  # Force CPU for multiprocessing workers
        backend=backend,
        game_type=game_type,
        board_size=board_size,
        wave_size=getattr(config.mcts, 'wave_size', None),
        enable_fast_ucb=True,
        use_cuda_graphs=False,  # Disable for multiprocessing
        use_mixed_precision=False,  # Disable for multiprocessing
        enable_subtree_reuse=False,
        max_tree_nodes=config.mcts.max_tree_nodes,
        memory_pool_size_mb=config.mcts.memory_pool_size_mb
    )
    
    # Generate games
    for game_idx in range(num_games):
        # Create new game instance
        if game_type == 'chess':
            import chess
            game = chess.Board()
        elif game_type == 'go':
            import alphazero_py as az
            game = az.GoState(board_size)
        else:  # gomoku
            import alphazero_py as az
            game = az.GomokuState(board_size)
        
        # Create MCTS instance based on backend
        if backend == 'cpu':
            from mcts.cpu.cpu_mcts_wrapper import create_cpu_optimized_mcts
            mcts = create_cpu_optimized_mcts(mcts_config, evaluator, None)
        elif backend == 'hybrid':
            from mcts.hybrid import create_hybrid_mcts
            mcts = create_hybrid_mcts(mcts_config, evaluator, None)
        else:
            mcts = MCTS(
                config=mcts_config,
                evaluator=evaluator
            )
        
        # Play game  
        # Create self-play game instance
        sp_game = SelfPlayGame(
            game=game,
            mcts=mcts,
            config=self_play_config,
            game_id=f"worker{worker_id}_game{game_idx}"
        )
        
        examples, metrics = sp_game.play_game()
        worker_examples.extend(examples)
        worker_metrics.append(metrics)
        
        # Log progress
        if (game_idx + 1) % 10 == 0:
            logger.info(f"Worker {worker_id}: Completed {game_idx + 1}/{num_games} games")
    
    return worker_examples, worker_metrics


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
    num_workers: int = 4  # Number of parallel workers
    cpu_threads_per_worker: int = 1  # CPU threads per worker
    use_mcts_q_values: bool = False  # Use Q-values from MCTS as training targets
    q_value_weight: float = 0.8  # Weight for mixing Q-values with game outcomes


class SelfPlayGame:
    """Manages a single self-play game"""
    
    def __init__(self, game, mcts, config: SelfPlayConfig, game_id: str):
        """Initialize self-play game
        
        Args:
            game: Game instance
            mcts: MCTS instance or dict mapping player to MCTS
            config: Self-play configuration
            game_id: Unique game identifier
        """
        self.game = game
        self.mcts = mcts  # Can be single MCTS or dict of player -> MCTS
        self.config = config
        self.game_id = game_id
        self.is_multi_mcts = isinstance(mcts, dict)
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
            # For raw C++ game objects, we'll use them directly
            # The game interface is only used for type detection which we can skip
            self.game_interface = None
    
    def play_single_move(self):
        """Play a single move in the game"""
        # Get current state
        if hasattr(self.game, 'get_state'):
            state = self.game.get_state()
        else:
            # For C++ games, they are the state themselves
            state = self.game
        
        # Get current player
        if hasattr(self.game, 'get_current_player'):
            self.current_player = self.game.get_current_player()
        else:
            # For C++ games, use the current_player property
            self.current_player = self.game.current_player()
        
        # Run MCTS search to get policy
        if self.is_multi_mcts:
            # Use appropriate MCTS for current player (0-indexed)
            player_idx = self.current_player - 1
            current_mcts = self.mcts[player_idx]
            action_probs = current_mcts.search(state)
        else:
            action_probs = self.mcts.search(state)
        
        
        # Calculate and store policy entropy
        # Entropy = -sum(p * log(p)) for p > 0
        # Add small epsilon to avoid log(0) and ensure non-negative entropy
        valid_probs = action_probs[action_probs > 1e-10]
        if len(valid_probs) > 0:
            entropy = -np.sum(valid_probs * np.log(valid_probs))
        else:
            entropy = 0.0
        self.policy_entropies.append(entropy)
        
        # Log low entropy situations which might indicate overconfidence
        # Low entropy situations tracked internally
        
        # Get value prediction from MCTS (if available)
        # CRITICAL FIX: The Q-value from MCTS is from the perspective of the player
        # who is ABOUT TO MOVE (current_player). This is the value of the position
        # AFTER the search, which represents how good the position is for current_player.
        if self.is_multi_mcts:
            # Get value from current player's MCTS
            player_idx = self.current_player - 1
            current_mcts = self.mcts[player_idx]
            if hasattr(current_mcts, 'get_root_value'):
                value_pred = current_mcts.get_root_value()
                # Store with current player for proper perspective tracking
                self.value_predictions.append((value_pred, self.current_player))
            else:
                value_pred = 0.0
        else:
            if hasattr(self.mcts, 'get_root_value'):
                value_pred = self.mcts.get_root_value()
                # Store with current player for proper perspective tracking
                self.value_predictions.append((value_pred, self.current_player))
            else:
                value_pred = 0.0  # Neutral value when no prediction available
        
        # Store training example with numpy array state
        # Convert state to numpy array for training compatibility
        if self.game_interface:
            state_array = self.game_interface.state_to_numpy(state)
        else:
            # For C++ games, the state is already a tensor/array
            state_array = state if hasattr(state, 'numpy') else np.array(state)
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
        
        
        # FIXED: MCTS should now properly filter illegal moves with CPU game states fix
        # Trust MCTS to only return legal actions
        
        # Make action
        if hasattr(self.game, 'make_action'):
            self.game.make_action(action)
        else:
            # For C++ games, use make_move
            self.game.make_move(action)
        self.move_count += 1
        
        # Update MCTS tree to move to the new root
        # CRITICAL FIX: Pass the new game state to update_root when subtree reuse is disabled
        # This ensures MCTS maintains the correct game position
        # Get updated state
        if hasattr(self.game, 'get_state'):
            new_state = self.game.get_state()
        else:
            new_state = self.game
            
        if self.is_multi_mcts:
            # Update both players' MCTS trees
            for player_idx in self.mcts:
                self.mcts[player_idx].update_root(action, new_state)
        else:
            # Update single MCTS tree
            self.mcts.update_root(action, new_state)
        
        # Debug logging for early game termination investigation
        is_terminal = self.game.is_terminal() if hasattr(self.game, 'is_terminal') else self.game.is_terminal()
        if is_terminal and self.move_count <= 20:
            winner = self.game.get_winner() if hasattr(self.game, 'get_winner') else self.game.get_winner()
            logger.warning(f"Game {self.game_id} ended early at move {self.move_count}, winner: {winner}, action: {action}")
        
        # Check for resignation
        if self.config.enable_resign and self.move_count > 10:
            # Use value_pred from above (now always defined)
            if value_pred < self.config.resign_threshold:
                self.resigned = True
                # Player resigned based on value threshold
                logger.info(f"Game {self.game_id} resigned: value_pred={value_pred:.4f} < threshold={self.config.resign_threshold}")
                return True  # Resign
            else:
                logger.debug(f"Game {self.game_id}: Move {self.move_count}, Player {self.current_player}, "
                            f"value_pred={value_pred:.4f} >= threshold={self.config.resign_threshold}, continuing...")
        else:
            logger.debug(f"Game {self.game_id}: Resignation check: enable_resign={self.config.enable_resign}, move_count={self.move_count}, value_pred={value_pred:.4f}")
        
        return False
    
    def play_game(self) -> Tuple[List[GameExample], Dict[str, Any]]:
        """Play a complete game and return training examples and metrics
        
        Returns:
            Tuple of (game_examples, metrics_dict)
        """
        resigned = False
        
        # Main game loop
        while True:
            is_terminal = self.game.is_terminal() if hasattr(self.game, 'is_terminal') else self.game.is_terminal()
            if is_terminal:
                break
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
            winner = self.game.get_winner() if hasattr(self.game, 'get_winner') else self.game.get_winner()
            pass  # Game ended naturally
            
        # Game termination details tracked internally
        
        # Convert examples to GameExample objects with correct values
        game_examples = []
        
        # Check if we should use Q-values from MCTS
        use_q_values = getattr(self.config, 'use_mcts_q_values', False)
        q_value_weight = getattr(self.config, 'q_value_weight', 0.8)
        
        for i, example in enumerate(self.examples):
            # Value is from the perspective of the player who made the move
            # winner is 1 for P1 win, -1 for P2 win, 0 for draw
            # Players are numbered 1 and 2
            if example['player'] == 1:
                game_outcome_value = winner  # P1 gets +1 if they won, -1 if they lost
            else:  # player == 2
                game_outcome_value = -winner  # P2 gets +1 if they won (winner=-1), -1 if they lost (winner=1)
            
            # Use Q-value if available and enabled
            if use_q_values and i < len(self.value_predictions):
                q_value, q_player = self.value_predictions[i]
                # CRITICAL: Verify the Q-value is from the same player
                if q_player != example['player']:
                    logger.error(f"Player mismatch in Q-value assignment: example player={example['player']}, Q-value player={q_player}")
                # The Q-value is already from the correct player's perspective
                # Mix Q-value with game outcome
                value = q_value_weight * q_value + (1 - q_value_weight) * game_outcome_value
            else:
                value = game_outcome_value
            
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
                # Special case for draws: check if prediction is close to 0
                if actual_value == 0:
                    accuracy = 1.0 if abs(value_pred) < 0.1 else 0.0
                else:
                    accuracy = 1.0 if (value_pred * actual_value) > 0 else 0.0
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
                 game_config: Optional[Dict[str, Any]] = None, opponent_buffer=None):
        """Initialize self-play manager
        
        Args:
            config: AlphaZero configuration
            game_class: Game class to instantiate
            evaluator: Neural network evaluator
            game_config: Optional game configuration
            opponent_buffer: Optional opponent buffer for population-based training
        """
        self.config = config
        self.game_class = game_class
        self.opponent_buffer = opponent_buffer
        self.evaluator = evaluator
        self.game_config = game_config if game_config else self.config.game
        
        # Metrics collection
        self.collected_metrics = []
        
        # Create self-play config from AlphaZero config
        enable_resign_value = getattr(config.training, 'enable_resign', True)
        logger.info(f"SelfPlayManager: enable_resign from config = {enable_resign_value}")
        
        self.self_play_config = SelfPlayConfig(
            num_games_per_iteration=config.training.num_games_per_iteration,
            mcts_simulations=config.mcts.num_simulations,
            temperature=config.mcts.temperature,
            temperature_threshold=config.mcts.temperature_threshold,
            resign_threshold=getattr(config.training, 'resign_threshold', -0.98),
            enable_resign=enable_resign_value,
            dirichlet_alpha=config.mcts.dirichlet_alpha,
            dirichlet_epsilon=config.mcts.dirichlet_epsilon,
            c_puct=config.mcts.c_puct,
            batch_size=getattr(config.mcts, 'batch_size', 512),
            device=config.mcts.device,  # Use device from MCTS config
            use_mcts_q_values=getattr(config.training, 'use_mcts_q_values', False),
            q_value_weight=getattr(config.training, 'q_value_weight', 0.8),
            num_workers=getattr(config.resources, 'num_workers', 4)  # Number of workers for parallel self-play
        )
        
        # Create direct GPU evaluator if evaluator is a model
        if hasattr(evaluator, 'model'):
            self.gpu_evaluator = SingleGPUEvaluator(
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
        """Generate self-play training data
        
        Returns:
            List of training examples
        """
        logger.info(f"Generating {self.self_play_config.num_games_per_iteration} self-play games on {self.self_play_config.device}")
        
        # Determine backend from config
        backend = getattr(self.config.mcts, 'backend', 'gpu')
        
        # Only use parallel generation for pure CPU mode
        # Hybrid and GPU modes use CUDA which doesn't work with multiprocessing fork
        if backend == 'cpu' and self.self_play_config.num_workers > 1:
            logger.info(f"Using parallel self-play with {self.self_play_config.num_workers} workers for {backend} backend")
            return self._generate_parallel_cpu()
        else:
            # Use sequential generation for GPU/hybrid modes to avoid CUDA multiprocessing errors
            logger.info(f"Using sequential self-play for {backend} backend")
            return self._generate_single_gpu()
    
    def _generate_parallel_cpu(self) -> List[GameExample]:
        """Generate self-play data using parallel workers for CPU/hybrid backends"""
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        all_examples = []
        self.collected_metrics = []
        
        # Calculate work distribution
        games_per_worker = self.self_play_config.num_games_per_iteration // self.self_play_config.num_workers
        remainder = self.self_play_config.num_games_per_iteration % self.self_play_config.num_workers
        
        work_distribution = []
        for i in range(self.self_play_config.num_workers):
            num_games = games_per_worker + (1 if i < remainder else 0)
            if num_games > 0:
                work_distribution.append((i, num_games))
        
        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=self.self_play_config.num_workers) as executor:
            # Submit all worker tasks
            futures = []
            for worker_id, num_games in work_distribution:
                future = executor.submit(
                    self._worker_generate_games,
                    worker_id,
                    num_games,
                    self.config,
                    self.self_play_config,
                    self.game_config
                )
                futures.append(future)
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    worker_examples, worker_metrics = future.result()
                    all_examples.extend(worker_examples)
                    self.collected_metrics.extend(worker_metrics)
                    logger.info(f"Worker completed with {len(worker_examples)} examples")
                except Exception as e:
                    logger.error(f"Worker failed with error: {e}")
                    raise
        
        logger.info(f"Generated {len(all_examples)} total training examples from parallel workers")
        return all_examples
    
    def _generate_parallel_cpu(self) -> List[GameExample]:
        """Generate self-play data using parallel workers for CPU/hybrid backends"""
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        all_examples = []
        self.collected_metrics = []
        
        # Calculate work distribution
        games_per_worker = self.self_play_config.num_games_per_iteration // self.self_play_config.num_workers
        remainder = self.self_play_config.num_games_per_iteration % self.self_play_config.num_workers
        
        work_distribution = []
        for i in range(self.self_play_config.num_workers):
            num_games = games_per_worker + (1 if i < remainder else 0)
            if num_games > 0:
                work_distribution.append((i, num_games))
        
        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=self.self_play_config.num_workers) as executor:
            # Submit all worker tasks
            futures = []
            for worker_id, num_games in work_distribution:
                # Convert game_config to dict if it's an object
                if hasattr(self.game_config, '__dict__'):
                    game_config_dict = {
                        'game_type': self.game_config.game_type,
                        'board_size': self.game_config.board_size
                    }
                else:
                    game_config_dict = self.game_config
                    
                future = executor.submit(
                    _worker_generate_games,
                    worker_id,
                    num_games,
                    self.config,
                    self.self_play_config,
                    game_config_dict
                )
                futures.append(future)
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    worker_examples, worker_metrics = future.result()
                    all_examples.extend(worker_examples)
                    self.collected_metrics.extend(worker_metrics)
                    logger.info(f"Worker completed with {len(worker_examples)} examples")
                except Exception as e:
                    logger.error(f"Worker failed with error: {e}")
                    raise
        
        logger.info(f"Generated {len(all_examples)} total training examples from parallel workers")
        return all_examples
    
    def _generate_single_gpu(self) -> List[GameExample]:
        """Generate self-play data using single GPU"""
        all_examples = []
        self.collected_metrics = []
        
        # Create shared MCTS instance with optimized config
        backend = getattr(self.config.mcts, 'backend', 'gpu')
        mcts_config = MCTSConfig(
            num_simulations=self.self_play_config.mcts_simulations,
            c_puct=self.self_play_config.c_puct,
            dirichlet_alpha=self.self_play_config.dirichlet_alpha,
            dirichlet_epsilon=self.self_play_config.dirichlet_epsilon,
            tree_batch_size=128 if backend == 'cpu' else 512,  # Optimized batch size
            device=self.self_play_config.device,
            backend=backend,  # Use backend from config
            wave_size=getattr(self.config.mcts, 'wave_size', None),  # Pass wave size
            enable_fast_ucb=True,
            use_cuda_graphs=backend == 'gpu',  # Only for GPU
            use_mixed_precision=backend == 'gpu',  # Only for GPU
            enable_subtree_reuse=False,  # Disable to prevent memory issues
            max_tree_nodes=self.config.mcts.max_tree_nodes,
            memory_pool_size_mb=self.config.mcts.memory_pool_size_mb,
            # Enable parallel MCTS for hybrid backend
            use_parallel_mcts=backend == 'hybrid' and getattr(self.config.mcts, 'use_parallel_mcts', True),
            num_mcts_workers=getattr(self.config.mcts, 'num_mcts_workers', 4),
            worker_batch_size=getattr(self.config.mcts, 'worker_batch_size', 32)
        )
        
        # Create game interface for MCTS (use first game as template)
        template_game = self._create_game_instance()
        if hasattr(template_game, 'game_interface'):
            game_interface = template_game.game_interface
        else:
            game_interface = GameInterface(template_game)
        
        # Create shared MCTS instance based on backend
        if backend == 'cpu':
            from mcts.cpu.cpu_mcts_wrapper import create_cpu_optimized_mcts
            shared_mcts = create_cpu_optimized_mcts(mcts_config, self.gpu_evaluator, game_interface)
        elif backend == 'hybrid':
            from mcts.hybrid import create_hybrid_mcts
            shared_mcts = create_hybrid_mcts(mcts_config, self.gpu_evaluator, game_interface)
        else:
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
            
            # Decide if we should use an opponent from the buffer
            use_opponent = False
            opponent_evaluator = None
            
            if self.opponent_buffer and game_idx % 4 == 0:  # Use opponent 25% of the time
                opponent_evaluator = self.opponent_buffer.get_random_opponent()
                if opponent_evaluator:
                    use_opponent = True
                    
            # Create MCTS instance(s)
            if use_opponent:
                # Create separate MCTS for each player based on backend
                if backend == 'cpu':
                    from mcts.cpu.cpu_mcts_wrapper import create_cpu_optimized_mcts
                    mcts_current = create_cpu_optimized_mcts(mcts_config, self.gpu_evaluator, game_interface)
                    mcts_opponent = create_cpu_optimized_mcts(mcts_config, opponent_evaluator, game_interface)
                elif backend == 'hybrid':
                    from mcts.hybrid import create_hybrid_mcts
                    mcts_current = create_hybrid_mcts(mcts_config, self.gpu_evaluator, game_interface)
                    mcts_opponent = create_hybrid_mcts(mcts_config, opponent_evaluator, game_interface)
                else:
                    mcts_current = MCTS(
                        config=mcts_config,
                        evaluator=self.gpu_evaluator,
                        game_interface=game_interface
                    )
                    mcts_opponent = MCTS(
                        config=mcts_config,
                        evaluator=opponent_evaluator,
                        game_interface=game_interface
                    )
                # Randomly assign who plays first
                if np.random.random() < 0.5:
                    player_mcts = {0: mcts_current, 1: mcts_opponent}
                    using_opponent_as = "player2"
                else:
                    player_mcts = {0: mcts_opponent, 1: mcts_current}
                    using_opponent_as = "player1"
            else:
                # Use shared MCTS for both players
                shared_mcts.reset_tree()
                player_mcts = shared_mcts
                using_opponent_as = None
            
            # Play game
            sp_game = SelfPlayGame(
                game=game,
                mcts=player_mcts,
                config=self.self_play_config,
                game_id=f"game_{game_idx}"
            )
            
            examples, metrics = sp_game.play_game()
            
            # Mark examples if playing against opponent
            if use_opponent:
                metrics['vs_opponent'] = True
                metrics['opponent_as'] = using_opponent_as
                
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