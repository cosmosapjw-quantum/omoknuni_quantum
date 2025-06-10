"""Self-play module for training data generation

This module handles parallel self-play game generation with progress tracking.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import torch

logger = logging.getLogger(__name__)


@dataclass
class SelfPlayConfig:
    """Configuration for self-play generation"""
    num_games: int = 100
    num_workers: int = 4
    max_moves_per_game: int = 500
    temperature_threshold: int = 30
    use_progress_bar: bool = True
    debug_mode: bool = False


class SelfPlayManager:
    """Manages self-play data generation"""
    
    def __init__(self, config):
        from mcts.core.game_interface import GameInterface, GameType
        
        self.config = config
        self.game_type = GameType[config.game.game_type.upper()]
        self.game_interface = GameInterface(
            self.game_type,
            board_size=config.game.board_size
        )
    
    def generate_games(self, model: torch.nn.Module, iteration: int,
                      num_games: Optional[int] = None,
                      num_workers: Optional[int] = None) -> List[Any]:
        """Generate self-play games with progress tracking
        
        Args:
            model: Neural network model
            iteration: Current training iteration
            num_games: Number of games to generate (overrides config)
            num_workers: Number of parallel workers (overrides config)
            
        Returns:
            List of game examples
        """
        num_games = num_games or self.config.training.num_games_per_iteration
        num_workers = num_workers or self.config.training.num_workers
        
        if num_workers <= 1:
            return self._sequential_self_play(model, iteration, num_games)
        else:
            return self._parallel_self_play(model, iteration, num_games, num_workers)
    
    def _sequential_self_play(self, model: torch.nn.Module, iteration: int,
                             num_games: int) -> List[Any]:
        """Generate games sequentially with progress bar"""
        from .unified_training_pipeline import GameExample
        from mcts.core.evaluator import AlphaZeroEvaluator
        
        examples = []
        
        # Create evaluator
        evaluator = AlphaZeroEvaluator(
            model=model,
            device=self.config.mcts.device
        )
        
        # Progress bar setup
        disable_progress = logger.level > logging.INFO
        
        with tqdm(total=num_games, desc="Self-play games", unit="game",
                 disable=disable_progress) as pbar:
            for game_idx in range(num_games):
                game_examples = self._play_single_game(
                    model, evaluator, game_idx, iteration
                )
                examples.extend(game_examples)
                pbar.update(1)
        
        return examples
    
    def _parallel_self_play(self, model: torch.nn.Module, iteration: int,
                           num_games: int, num_workers: int) -> List[Any]:
        """Generate games in parallel with progress tracking"""
        from .unified_training_pipeline import GameExample
        
        examples = []
        model_state_dict = model.state_dict()
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all games
            futures = []
            for game_idx in range(num_games):
                future = executor.submit(
                    _play_game_worker,
                    self.config,
                    model_state_dict,
                    game_idx,
                    iteration
                )
                futures.append(future)
            
            # Collect results with progress bar
            disable_progress = logger.level > logging.INFO
            
            with tqdm(total=len(futures), desc="Self-play games", unit="game",
                     disable=disable_progress) as pbar:
                for future in as_completed(futures):
                    try:
                        game_examples = future.result()
                        examples.extend(game_examples)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Self-play game failed: {e}")
                        if self.config.log_level == "DEBUG":
                            import traceback
                            logger.debug(traceback.format_exc())
                        pbar.update(1)
        
        return examples
    
    def _play_single_game(self, model: torch.nn.Module, evaluator: Any,
                         game_idx: int, iteration: int) -> List[Any]:
        """Play a single self-play game"""
        from .unified_training_pipeline import GameExample
        
        # Create MCTS
        mcts = self._create_mcts(evaluator)
        
        # Play game
        examples = []
        state = self.game_interface.create_initial_state()
        game_id = f"iter{iteration}_game{game_idx}"
        
        for move_num in range(self.config.training.max_moves_per_game):
            # Get action probabilities
            temperature = 1.0 if move_num < self.config.mcts.temperature_threshold else 0.0
            
            # Get full policy for storage
            policy = mcts.get_action_probabilities(state, temperature=temperature)
            
            # Get only valid actions and their probabilities for sampling
            valid_actions, valid_probs = mcts.get_valid_actions_and_probabilities(state, temperature=temperature)
            
            if not valid_actions:
                # This should never happen in a properly implemented game
                raise ValueError(f"No valid actions available at move {move_num}")
            
            # Sample action from valid moves only
            action = np.random.choice(valid_actions, p=valid_probs)
            
            # Double-check the action is actually legal
            if not state.is_legal_move(action):
                logger.error(f"MCTS returned illegal action {action} at move {move_num}")
                logger.error(f"Valid actions according to MCTS: {valid_actions[:10]}...")
                legal_moves = self.game_interface.get_legal_moves(state)
                logger.error(f"Actual legal moves: {legal_moves[:10]}... (total {len(legal_moves)})")
                # Try to find a legal action
                legal_valid = [a for a in valid_actions if a in legal_moves]
                if legal_valid:
                    action = np.random.choice(legal_valid)
                    logger.warning(f"Using alternative legal action: {action}")
                else:
                    raise ValueError(f"No legal actions available from MCTS at move {move_num}")
            
            # Store example
            canonical_state = self.game_interface.get_canonical_form(state)
            examples.append(GameExample(
                state=canonical_state,
                policy=policy,
                value=0,
                game_id=game_id,
                move_number=move_num
            ))
            
            # Apply action
            try:
                state = self.game_interface.get_next_state(state, action)
            except ValueError as e:
                # Log more context when illegal move happens
                logger.error(f"Illegal move at move {move_num}: action={action}")
                logger.error(f"Valid actions were: {valid_actions[:10]}... (total {len(valid_actions)})")
                logger.error(f"Game terminal before move: {self.game_interface.is_terminal(state)}")
                raise
            
            # Update MCTS tree root for tree reuse
            mcts.update_root(action)
            
            # Check terminal
            if self.game_interface.is_terminal(state):
                outcome = self.game_interface.get_value(state)
                
                # Update values based on game outcome
                for i, example in enumerate(examples):
                    example.value = outcome * ((-1) ** (i % 2))
                break
        
        return examples
    
    def _create_mcts(self, evaluator: Any):
        """Create MCTS instance with quantum features if enabled"""
        from mcts.core.mcts import MCTS, MCTSConfig
        from mcts.quantum.quantum_features import create_quantum_mcts
        from mcts.utils.config_system import QuantumLevel
        
        mcts_config = MCTSConfig(
            num_simulations=self.config.mcts.num_simulations,
            c_puct=self.config.mcts.c_puct,
            temperature=1.0,
            device=self.config.mcts.device,
            game_type=self.game_type,
            min_wave_size=self.config.mcts.min_wave_size,
            max_wave_size=self.config.mcts.max_wave_size,
            adaptive_wave_sizing=self.config.mcts.adaptive_wave_sizing
        )
        
        # Create MCTS with quantum features if enabled
        if self.config.mcts.enable_quantum and self.config.mcts.quantum_level != QuantumLevel.CLASSICAL:
            # Create standard MCTS first
            base_mcts = MCTS(mcts_config, evaluator)
            # Wrap with quantum features
            mcts = create_quantum_mcts(
                enable_quantum=True,
                quantum_level=self.config.mcts.quantum_level.value,
                mcts=base_mcts
            )
        else:
            mcts = MCTS(mcts_config, evaluator)
        
        return mcts


def _play_game_worker(config, model_state_dict: Dict,
                     game_idx: int, iteration: int) -> List[Any]:
    """Worker function for parallel self-play"""
    from .unified_training_pipeline import GameExample
    from mcts.core.game_interface import GameInterface, GameType
    from mcts.neural_networks.nn_model import create_model
    from mcts.core.mcts import MCTS, MCTSConfig
    from mcts.quantum.quantum_features import create_quantum_mcts
    from mcts.utils.config_system import QuantumLevel
    
    # Set up logging level for worker
    logging.basicConfig(level=getattr(logging, config.log_level))
    
    # Create game interface
    game_type = GameType[config.game.game_type.upper()]
    game_interface = GameInterface(game_type, config.game.board_size)
    
    # Get action space size
    initial_state = game_interface.create_initial_state()
    action_size = game_interface.get_action_space_size(initial_state)
    
    # Recreate model
    model = create_model(
        game_type=config.game.game_type,
        input_height=config.game.board_size,
        input_width=config.game.board_size,
        num_actions=action_size,
        input_channels=config.network.input_channels,
        num_res_blocks=config.network.num_res_blocks,
        num_filters=config.network.num_filters
    )
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Create evaluator
    from mcts.core.evaluator import AlphaZeroEvaluator
    evaluator = AlphaZeroEvaluator(
        model=model,
        device=config.mcts.device
    )
    
    # Create MCTS
    mcts_config = MCTSConfig(
        num_simulations=config.mcts.num_simulations,
        c_puct=config.mcts.c_puct,
        temperature=1.0,
        device=config.mcts.device,
        game_type=game_type,
        min_wave_size=config.mcts.min_wave_size,
        max_wave_size=config.mcts.max_wave_size,
        adaptive_wave_sizing=config.mcts.adaptive_wave_sizing
    )
    
    # Create MCTS with quantum features if enabled
    if config.mcts.enable_quantum and config.mcts.quantum_level != QuantumLevel.CLASSICAL:
        # Create standard MCTS first
        base_mcts = MCTS(mcts_config, evaluator)
        # Wrap with quantum features
        mcts = create_quantum_mcts(
            enable_quantum=True,
            quantum_level=config.mcts.quantum_level.value,
            mcts=base_mcts
        )
    else:
        mcts = MCTS(mcts_config, evaluator)
    
    # Play game
    examples = []
    state = game_interface.create_initial_state()
    game_id = f"iter{iteration}_game{game_idx}"
    
    for move_num in range(config.training.max_moves_per_game):
        # Get action probabilities
        temperature = 1.0 if move_num < config.mcts.temperature_threshold else 0.0
        
        # Get full policy for storage
        policy = mcts.get_action_probabilities(state, temperature=temperature)
        
        # Get only valid actions and their probabilities for sampling
        valid_actions, valid_probs = mcts.get_valid_actions_and_probabilities(state, temperature=temperature)
        
        if not valid_actions:
            # This should never happen in a properly implemented game
            raise ValueError(f"No valid actions available at move {move_num}")
        
        # Sample action from valid moves only
        action = np.random.choice(valid_actions, p=valid_probs)
        
        # Store example
        canonical_state = game_interface.get_canonical_form(state)
        examples.append(GameExample(
            state=canonical_state,
            policy=policy,
            value=0,
            game_id=game_id,
            move_number=move_num
        ))
        
        # Apply action
        state = game_interface.get_next_state(state, action)
        
        # Update MCTS tree root for tree reuse
        mcts.update_root(action)
        
        # Check terminal
        if game_interface.is_terminal(state):
            outcome = game_interface.get_value(state)
            
            # Update values
            for i, example in enumerate(examples):
                example.value = outcome * ((-1) ** (i % 2))
            break
    
    return examples