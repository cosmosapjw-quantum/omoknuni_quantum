"""Self-play training pipeline for AlphaZero MCTS

This module implements the training pipeline that runs self-play games,
collects training data, and trains the neural network.
"""

import os
import time
import logging
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from collections import deque
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# Avoid circular import - import at runtime
# from mcts.core.high_performance_mcts import HighPerformanceMCTS, HighPerformanceMCTSConfig
from mcts.core.game_interface import GameInterface, GameType
from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
from .nn_model import AlphaZeroNetwork, ModelConfig, create_model


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training pipeline
    
    Attributes:
        game_type: Type of game ('chess', 'go', 'gomoku')
        batch_size: Training batch size
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        checkpoint_interval: Save checkpoint every N iterations
        window_size: Size of replay buffer
        num_epochs: Epochs per training iteration
        num_games_per_iteration: Self-play games per iteration
        num_workers: Parallel self-play workers
        max_moves_per_game: Maximum moves before draw
        temperature_threshold: Move number to switch from exploration to exploitation
        dirichlet_alpha: Dirichlet noise parameter
        dirichlet_epsilon: Weight of noise at root
        mcts_simulations: MCTS simulations per move
        c_puct: PUCT exploration constant
        save_dir: Directory for checkpoints
        device: Training device ('cuda' or 'cpu')
        mixed_precision: Use mixed precision training
        gradient_accumulation_steps: Steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping
    """
    game_type: str = 'gomoku'
    batch_size: int = 512
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    checkpoint_interval: int = 100
    window_size: int = 500000
    num_epochs: int = 10
    num_games_per_iteration: int = 100
    num_workers: int = 4
    max_moves_per_game: int = 500
    temperature_threshold: int = 30
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    mcts_simulations: int = 800
    c_puct: float = 1.0
    save_dir: str = 'checkpoints'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision: bool = False
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 5.0


@dataclass
class GameExample:
    """Single training example from self-play
    
    Attributes:
        state: Game state as numpy array
        policy: Target policy (action probabilities)
        value: Target value (-1 to 1)
        game_id: Unique game identifier
        move_number: Move number in game
    """
    state: np.ndarray
    policy: np.ndarray
    value: float
    game_id: str
    move_number: int


class ReplayBuffer(Dataset):
    """Replay buffer for training examples"""
    
    def __init__(self, max_size: int = 500000):
        self.buffer = deque(maxlen=max_size)
        
    def add_game(self, examples: List[GameExample]):
        """Add examples from a game to buffer"""
        self.buffer.extend(examples)
        
    def __len__(self):
        return len(self.buffer)
        
    def __getitem__(self, idx):
        example = self.buffer[idx]
        return (
            torch.from_numpy(example.state).float(),
            torch.from_numpy(example.policy).float(),
            torch.tensor(example.value).float()
        )
        
    def save(self, path: str):
        """Save buffer to disk"""
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)
            
    def load(self, path: str):
        """Load buffer from disk"""
        with open(path, 'rb') as f:
            self.buffer = deque(pickle.load(f), maxlen=self.buffer.maxlen)


def play_self_play_game(
    game: Any,  # Can be either GameInterface or StatefulGame wrapper
    mcts: Any,  # HighPerformanceMCTS (avoid circular import)
    config: TrainingConfig,
    game_id: str
) -> List[GameExample]:
    """Play a single self-play game
    
    Args:
        game: Game interface (stateful wrapper)
        mcts: MCTS instance
        config: Training configuration
        game_id: Unique game identifier
        
    Returns:
        List of training examples from the game
    """
    examples = []
    game.reset()
    move_number = 0
    
    while not game.is_terminal() and move_number < config.max_moves_per_game:
        # Get current state for NN
        state_tensor = game.get_nn_input()
        
        # Run MCTS (it returns a policy dict)
        policy = mcts.search(game.state)
        
        # Get action space size
        if hasattr(game, 'action_space_size'):
            action_size = game.action_space_size
        else:
            # For Gomoku, it's 15x15 = 225
            action_size = 225
        
        # Convert to probability array
        action_probs = np.zeros(action_size)
        total = 0.0
        for action, prob in policy.items():
            action_probs[action] = prob
            total += prob
        
        # Normalize if needed
        if total > 0:
            action_probs = action_probs / total
        else:
            # Uniform over legal moves
            legal_moves = game.get_legal_moves()
            if len(legal_moves) > 0:
                for move in legal_moves:
                    action_probs[move] = 1.0 / len(legal_moves)
            else:
                # No legal moves - shouldn't happen
                break
        
        # Temperature-based sampling
        if move_number < config.temperature_threshold:
            # Sample from distribution
            # Make sure probabilities sum to 1
            if action_probs.sum() > 0:
                action_probs = action_probs / action_probs.sum()
                action = np.random.choice(len(action_probs), p=action_probs)
            else:
                # Fallback to first legal move
                legal_moves = game.get_legal_moves()
                action = legal_moves[0] if legal_moves else 0
        else:
            # Choose best action
            action = np.argmax(action_probs)
        
        # Store example
        examples.append(GameExample(
            state=state_tensor,
            policy=action_probs,
            value=0.0,  # Will be filled later
            game_id=game_id,
            move_number=move_number
        ))
        
        # Make move
        game.make_move(action)
        move_number += 1
    
    # Get final reward and update values
    if game.is_terminal():
        reward = game.get_reward()
        # Propagate reward backwards, alternating signs for two players
        for i in range(len(examples)):
            # Player 1 moves on even indices, Player 2 on odd indices
            if i % 2 == 0:
                examples[i] = GameExample(
                    state=examples[i].state,
                    policy=examples[i].policy,
                    value=reward,
                    game_id=examples[i].game_id,
                    move_number=examples[i].move_number
                )
            else:
                examples[i] = GameExample(
                    state=examples[i].state,
                    policy=examples[i].policy,
                    value=-reward,
                    game_id=examples[i].game_id,
                    move_number=examples[i].move_number
                )
    
    return examples


def run_self_play_worker(
    worker_id: int,
    game_class: type,
    model_path: str,
    config: TrainingConfig,
    num_games: int,
    result_queue: mp.Queue
):
    """Worker process for self-play
    
    Args:
        worker_id: Worker identifier
        game_class: Game class to instantiate
        model_path: Path to neural network model
        config: Training configuration
        num_games: Number of games to play
        result_queue: Queue for results
    """
    try:
        # Create game and evaluator
        game = game_class()
        
        # Load model with weights_only=True for security
        # First check if model exists
        if not os.path.exists(model_path):
            logger.error(f"Worker {worker_id}: Model path {model_path} does not exist")
            result_queue.put((worker_id, None))
            return
            
        # First try loading with weights_only=True (for state_dict)
        try:
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            # Create model instance based on game configuration
            board_size = game.board_size if hasattr(game, 'board_size') else 15
            model_config = ModelConfig(
                input_channels=20,  # 20 channels as per encoding
                input_height=board_size,
                input_width=board_size,
                num_actions=game.action_space_size if hasattr(game, 'action_space_size') else board_size*board_size,
                num_res_blocks=10,
                num_filters=256
            )
            model = create_model(game_type=config.game_type, 
                               input_channels=model_config.input_channels,
                               input_height=model_config.input_height,
                               input_width=model_config.input_width,
                               num_actions=model_config.num_actions,
                               num_res_blocks=model_config.num_res_blocks,
                               num_filters=model_config.num_filters)
            model.load_state_dict(state_dict)
            logger.info(f"Worker {worker_id}: Loaded model state dict successfully")
        except Exception as e:
            # Fallback: try loading full model (less secure, for older checkpoints)
            logger.warning(f"Worker {worker_id}: Loading state dict failed ({e}), trying full model load")
            try:
                loaded = torch.load(model_path, map_location='cpu', weights_only=False)
                if isinstance(loaded, dict) and 'state_dict' in loaded:
                    # It's a checkpoint with state_dict
                    # Create model first
                    board_size = game.board_size if hasattr(game, 'board_size') else 15
                    model = create_model(game_type=config.game_type, 
                                       input_channels=20,
                                       input_height=board_size,
                                       input_width=board_size,
                                       num_actions=game.action_space_size if hasattr(game, 'action_space_size') else board_size*board_size,
                                       num_res_blocks=10,
                                       num_filters=256)
                    model.load_state_dict(loaded['state_dict'])
                elif hasattr(loaded, 'state_dict'):
                    # It's a full model object
                    model = loaded
                else:
                    # Try to load it as a state dict directly
                    board_size = game.board_size if hasattr(game, 'board_size') else 15
                    model = create_model(game_type=config.game_type, 
                                       input_channels=20,
                                       input_height=board_size,
                                       input_width=board_size,
                                       num_actions=game.action_space_size if hasattr(game, 'action_space_size') else board_size*board_size,
                                       num_res_blocks=10,
                                       num_filters=256)
                    model.load_state_dict(loaded)
                logger.warning(f"Worker {worker_id}: Loaded model successfully.")
            except Exception as e2:
                logger.error(f"Worker {worker_id}: Failed to load model: {e2}")
                result_queue.put((worker_id, None))
                return
        
        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        logger.info(f"Worker {worker_id}: Model loaded on {device}")
        
        # Create evaluator with the model
        from mcts.core.evaluator import Evaluator
        
        class SimpleEvaluator(Evaluator):
            def __init__(self, model):
                self.model = model
                self.model.eval()
                self.device = next(model.parameters()).device
                
            def evaluate(self, state):
                with torch.no_grad():
                    tensor = torch.tensor(state.get_tensor_representation(), dtype=torch.float32).unsqueeze(0)
                    tensor = tensor.to(self.device)
                    policy, value = self.model(tensor)
                    return policy.squeeze(0).cpu().numpy(), value.cpu().item()
                    
            def evaluate_batch(self, states):
                with torch.no_grad():
                    # Handle both tensor input and list of states
                    if isinstance(states, torch.Tensor):
                        batch = states.to(self.device)
                    else:
                        tensors = [torch.tensor(s.get_tensor_representation(), dtype=torch.float32) for s in states]
                        batch = torch.stack(tensors).to(self.device)
                    policies, values = self.model(batch)
                    return policies.cpu().numpy(), values.squeeze(-1).cpu().numpy()
        
        evaluator = SimpleEvaluator(model)
        
        # Create MCTS (import at runtime to avoid circular imports)
        from mcts.core.high_performance_mcts import HighPerformanceMCTS, HighPerformanceMCTSConfig
        
        mcts_config = HighPerformanceMCTSConfig(
            num_simulations=config.mcts_simulations,
            c_puct=config.c_puct,
            temperature=1.0,  # Default temperature for exploration
            enable_gpu=True,  # Enable GPU for better performance
            device='cuda' if torch.cuda.is_available() else 'cpu',
            mixed_precision=True  # Use mixed precision for faster GPU computation
        )
        mcts = HighPerformanceMCTS(mcts_config, game, evaluator)
        
        # Play games
        all_examples = []
        for i in range(num_games):
            game_id = f"worker_{worker_id}_game_{i}"
            
            # Create a stateful game wrapper
            class StatefulGame:
                def __init__(self, interface):
                    self.interface = interface
                    self.state = interface.create_initial_state()
                    
                def reset(self):
                    self.state = self.interface.create_initial_state()
                    
                def is_terminal(self):
                    return self.interface.is_terminal(self.state)
                    
                def get_nn_input(self):
                    return self.interface.state_to_numpy(self.state, use_enhanced=True)
                    
                def get_legal_moves(self):
                    return self.interface.get_legal_moves(self.state)
                    
                def make_move(self, move):
                    self.state = self.interface.apply_move(self.state, move)
                    
                def get_reward(self):
                    if not self.interface.is_terminal(self.state):
                        return 0.0
                    
                    winner = self.interface.get_winner(self.state)
                    current_player = self.interface.get_current_player(self.state)
                    
                    if winner == 0:  # Draw
                        return 0.0
                    elif winner == 1:  # Player 1 wins
                        return 1.0 if current_player == 0 else -1.0
                    else:  # Player 2 wins
                        return -1.0 if current_player == 0 else 1.0
                    
                @property
                def action_space_size(self):
                    return self.interface.get_action_space_size(self.state)
            
            stateful_game = StatefulGame(game)
            examples = play_self_play_game(stateful_game, mcts, config, game_id)
            all_examples.extend(examples)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Worker {worker_id}: Completed {i+1}/{num_games} games")
                
        result_queue.put((worker_id, all_examples))
        
    except Exception as e:
        logger.error(f"Worker {worker_id} failed: {str(e)}")
        logger.error(traceback.format_exc())
        result_queue.put((worker_id, None))


class TrainingPipeline:
    """Main training pipeline for AlphaZero"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.iteration = 0
        self.replay_buffer = ReplayBuffer(config.window_size)
        
        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)
        
        # Initialize model with explicit configuration
        # Get game instance to determine board size
        game_type_enum = GameType[config.game_type.upper()]
        game = GameInterface(game_type_enum)
        board_size = game.board_size
        
        self.model = create_model(
            game_type=config.game_type,
            input_channels=20,  # Standard encoding
            input_height=board_size,
            input_width=board_size,
            num_actions=game.action_space_size if hasattr(game, 'action_space_size') else board_size*board_size,
            num_res_blocks=10,
            num_filters=256,
            dropout_rate=0.1  # Add some dropout for training
        )
        self.model = self.model.to(config.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=50,
            gamma=0.1
        )
        
        # Loss functions
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        
        # Mixed precision scaler
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            
    def generate_self_play_data(self, game_class: type) -> List[GameExample]:
        """Generate self-play data in parallel
        
        Args:
            game_class: Game class for self-play
            
        Returns:
            List of training examples
        """
        logger.info(f"Starting self-play generation with {self.config.num_workers} workers")
        
        # Save current model state dict for workers (more secure than saving the whole model)
        model_path = os.path.join(self.config.save_dir, 'current_model.pt')
        torch.save(self.model.state_dict(), model_path)
        
        # Calculate games per worker
        games_per_worker = self.config.num_games_per_iteration // self.config.num_workers
        extra_games = self.config.num_games_per_iteration % self.config.num_workers
        
        # Create result queue
        result_queue = mp.Queue()
        
        # Start workers
        processes = []
        for i in range(self.config.num_workers):
            num_games = games_per_worker + (1 if i < extra_games else 0)
            
            p = mp.Process(
                target=run_self_play_worker,
                args=(i, game_class, model_path, self.config, num_games, result_queue)
            )
            p.start()
            processes.append(p)
            
        # Collect results
        all_examples = []
        for _ in range(self.config.num_workers):
            worker_id, examples = result_queue.get()
            if examples is not None:
                all_examples.extend(examples)
                logger.info(f"Collected {len(examples)} examples from worker {worker_id}")
            else:
                logger.error(f"Worker {worker_id} failed")
                
        # Wait for all processes to finish
        for p in processes:
            p.join()
            
        logger.info(f"Generated {len(all_examples)} training examples")
        return all_examples
        
    def train_network(self):
        """Train the neural network on replay buffer"""
        if len(self.replay_buffer) < self.config.batch_size:
            logger.warning("Not enough data in replay buffer")
            return
            
        logger.info(f"Training network on {len(self.replay_buffer)} examples")
        
        # Create data loader
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Training metrics
        policy_losses = []
        value_losses = []
        total_losses = []
        
        # Train for configured epochs
        self.model.train()
        for epoch in range(self.config.num_epochs):
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_total_loss = 0
            num_batches = 0
            
            for batch_idx, (states, target_policies, target_values) in enumerate(dataloader):
                # Move to device
                states = states.to(self.config.device)
                target_policies = target_policies.to(self.config.device)
                target_values = target_values.to(self.config.device).unsqueeze(1)
                
                # Mixed precision training
                if self.config.mixed_precision:
                    with torch.cuda.amp.autocast():
                        policy_logits, values = self.model(states)
                        policy_loss = self.policy_loss_fn(policy_logits, target_policies)
                        value_loss = self.value_loss_fn(values, target_values)
                        total_loss = policy_loss + value_loss
                else:
                    policy_logits, values = self.model(states)
                    policy_loss = self.policy_loss_fn(policy_logits, target_policies)
                    value_loss = self.value_loss_fn(values, target_values)
                    total_loss = policy_loss + value_loss
                    
                # Gradient accumulation
                total_loss = total_loss / self.config.gradient_accumulation_steps
                
                if self.config.mixed_precision:
                    self.scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
                    
                # Update weights
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                        clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.optimizer.step()
                        
                    self.optimizer.zero_grad()
                    
                # Track losses
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_total_loss += total_loss.item()
                num_batches += 1
                
            # Average losses
            avg_policy_loss = epoch_policy_loss / num_batches
            avg_value_loss = epoch_value_loss / num_batches
            avg_total_loss = epoch_total_loss / num_batches
            
            policy_losses.append(avg_policy_loss)
            value_losses.append(avg_value_loss)
            total_losses.append(avg_total_loss)
            
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Policy Loss: {avg_policy_loss:.4f}, "
                f"Value Loss: {avg_value_loss:.4f}, "
                f"Total Loss: {avg_total_loss:.4f}"
            )
            
        # Update learning rate
        self.scheduler.step()
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'total_loss': np.mean(total_losses)
        }
        
    def save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint = {
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        checkpoint_path = os.path.join(
            self.config.save_dir,
            f'checkpoint_iter_{self.iteration}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Also save replay buffer
        buffer_path = os.path.join(
            self.config.save_dir,
            f'replay_buffer_iter_{self.iteration}.pkl'
        )
        self.replay_buffer.save(buffer_path)
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        # Load checkpoint - need to handle config object specially
        try:
            # First try with weights_only=True (won't work if checkpoint contains config)
            checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=True)
        except Exception as e:
            # If it fails due to config object, use safe_globals to allow TrainingConfig
            if "TrainingConfig" in str(e):
                # Allow TrainingConfig to be loaded (it's safe as it only contains parameters)
                with torch.serialization.safe_globals([TrainingConfig]):
                    checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=True)
            else:
                # For other errors or older checkpoints, fall back to unsafe loading
                logger.warning(f"Loading checkpoint with weights_only=False: {e}")
                checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=False)
        
        self.iteration = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from iteration {self.iteration}")
        
    def run_training_loop(self, game_class: type, num_iterations: int):
        """Run the main training loop
        
        Args:
            game_class: Game class for self-play
            num_iterations: Number of training iterations
        """
        logger.info(f"Starting training for {num_iterations} iterations")
        
        for i in range(num_iterations):
            self.iteration += 1
            logger.info(f"\n=== Training Iteration {self.iteration} ===")
            
            # Generate self-play data
            start_time = time.time()
            examples = self.generate_self_play_data(game_class)
            self.replay_buffer.add_game(examples)
            self_play_time = time.time() - start_time
            
            logger.info(f"Self-play took {self_play_time:.2f} seconds")
            
            # Train network
            start_time = time.time()
            train_metrics = self.train_network()
            train_time = time.time() - start_time
            
            logger.info(f"Training took {train_time:.2f} seconds")
            logger.info(f"Metrics: {train_metrics}")
            
            # Save checkpoint
            if self.iteration % self.config.checkpoint_interval == 0:
                self.save_checkpoint()
                
            # Log progress
            logger.info(
                f"Iteration {self.iteration} complete. "
                f"Replay buffer size: {len(self.replay_buffer)}"
            )


def create_training_pipeline(config: Optional[TrainingConfig] = None) -> TrainingPipeline:
    """Create a training pipeline with optional config
    
    Args:
        config: Training configuration (uses default if None)
        
    Returns:
        TrainingPipeline instance
    """
    if config is None:
        config = TrainingConfig()
        
    return TrainingPipeline(config)