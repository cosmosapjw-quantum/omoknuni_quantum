"""Unified AlphaZero training pipeline with enhanced features

This module provides a consolidated training pipeline that combines:
- Self-play data generation with progress tracking
- Neural network training with mixed precision
- Arena evaluation and ELO tracking
- Quantum MCTS integration
- Comprehensive configuration management
"""

import os
import time
import logging
import pickle
import shutil
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Union
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import only essential components at module level
from mcts.utils.config_system import (
    AlphaZeroConfig, QuantumLevel, create_default_config,
    merge_configs
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class GameExample:
    """Training example from self-play"""
    state: np.ndarray
    policy: np.ndarray
    value: float
    game_id: str = ""
    move_number: int = 0
    
    def __post_init__(self):
        # Ensure arrays are numpy arrays
        if not isinstance(self.state, np.ndarray):
            self.state = np.array(self.state)
        if not isinstance(self.policy, np.ndarray):
            self.policy = np.array(self.policy)


class ReplayBuffer(Dataset):
    """Experience replay buffer for training examples"""
    
    def __init__(self, max_size: int = 500000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add(self, examples: List[GameExample]):
        """Add examples to buffer"""
        self.buffer.extend(examples)
    
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, idx):
        example = self.buffer[idx]
        return (
            torch.FloatTensor(example.state),
            torch.FloatTensor(example.policy),
            torch.FloatTensor([example.value])
        )
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
    
    def save(self, path: str):
        """Save buffer to disk"""
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)
    
    def load(self, path: str):
        """Load buffer from disk"""
        with open(path, 'rb') as f:
            examples = pickle.load(f)
            self.buffer = deque(examples, maxlen=self.max_size)


class UnifiedTrainingPipeline:
    """Unified training pipeline for AlphaZero"""
    
    def __init__(self, config: AlphaZeroConfig, resume_from: Optional[str] = None):
        """Initialize the unified training pipeline
        
        Args:
            config: Complete AlphaZero configuration
            resume_from: Path to checkpoint to resume from
        """
        # Import here to avoid circular imports
        from mcts.core.game_interface import GameInterface, GameType
        
        self.config = config
        self.iteration = 0
        self.best_model_iteration = 0
        self.elo_tracker = None
        
        # Setup directories
        self._setup_directories()
        
        # Setup logging
        self._setup_logging()
        
        # Save configuration
        config_path = self.experiment_dir / "config.yaml"
        config.save(str(config_path))
        
        # Initialize game interface
        self.game_type = GameType[config.game.game_type.upper()]
        self.game_interface = GameInterface(
            self.game_type,
            board_size=config.game.board_size
        )
        
        # Initialize components
        self.replay_buffer = ReplayBuffer(config.training.window_size)
        self.model = self._create_model()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Loss functions
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        
        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda') if config.training.mixed_precision and torch.cuda.is_available() else None
        
        # Arena for evaluation (always create, but use based on evaluation_interval)
        self.arena = self._create_arena()
        # ELO tracker will be created when needed
        self.elo_tracker = None
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
    
    def _setup_directories(self):
        """Create necessary directories"""
        self.experiment_dir = Path(self.config.experiment_name)
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.best_model_dir = self.experiment_dir / "best_models"
        self.arena_log_dir = self.experiment_dir / self.config.arena.arena_log_dir
        self.data_dir = self.experiment_dir / self.config.training.data_dir
        
        for dir_path in [self.checkpoint_dir, self.best_model_dir, 
                         self.arena_log_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Configure logging based on config"""
        level = getattr(logging, self.config.log_level)
        
        # Only show INFO and above for production, DEBUG for debug mode
        if level == logging.DEBUG:
            logging.basicConfig(
                level=level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            # Reduce verbosity for production
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            # Suppress verbose library logs
            logging.getLogger('torch').setLevel(logging.WARNING)
            logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    def _create_model(self):
        """Create neural network model"""
        from mcts.neural_networks.nn_model import create_model
        
        # Get network config
        network_config = self.config.network
        
        # Get action space size from initial state
        initial_state = self.game_interface.create_initial_state()
        action_size = self.game_interface.get_action_space_size(initial_state)
        
        return create_model(
            game_type=self.config.game.game_type,
            input_height=self.config.game.board_size,
            input_width=self.config.game.board_size,
            num_actions=action_size,
            input_channels=network_config.input_channels,
            num_res_blocks=network_config.num_res_blocks,
            num_filters=network_config.num_filters,
            value_head_hidden_size=network_config.value_head_hidden_size
        )
    
    def _create_optimizer(self):
        """Create optimizer"""
        return optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.training.learning_rate_schedule == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.lr_decay_steps,
                gamma=self.config.training.lr_decay_rate
            )
        elif self.config.training.learning_rate_schedule == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.lr_decay_steps,
                eta_min=self.config.training.min_learning_rate
            )
        elif self.config.training.learning_rate_schedule == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.config.training.lr_decay_rate
            )
        else:
            # No scheduler
            return None
    
    def _create_arena(self):
        """Create arena for model evaluation"""
        from mcts.neural_networks.arena_module import ArenaManager, ArenaConfig
        
        arena_config = ArenaConfig(
            num_games=self.config.arena.num_games,
            win_threshold=self.config.arena.win_threshold,
            num_workers=self.config.arena.num_workers,
            mcts_simulations=self.config.arena.mcts_simulations,
            c_puct=self.config.arena.c_puct,
            temperature=self.config.arena.temperature,
            temperature_threshold=0,
            device=self.config.mcts.device
        )
        
        return ArenaManager(self.config, arena_config)
    
    def train(self, num_iterations: int):
        """Run the complete training pipeline
        
        Args:
            num_iterations: Number of training iterations
        """
        logger.info(f"Starting training for {num_iterations} iterations")
        
        # Use trange for iteration progress
        for i in trange(num_iterations, desc="Training iterations", 
                       initial=self.iteration, unit="iter"):
            self.iteration += 1
            iteration_start = time.time()
            
            # Phase 1: Self-play data generation
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {self.iteration}: Generating self-play data")
            self_play_examples = self.generate_self_play_data()
            
            # Add to replay buffer
            self.replay_buffer.add(self_play_examples)
            logger.info(f"Replay buffer size: {len(self.replay_buffer)}")
            
            # Phase 2: Neural network training
            logger.info(f"Iteration {self.iteration}: Training neural network")
            train_stats = self.train_neural_network()
            
            # Phase 3: Arena evaluation (if enabled)
            accepted = True
            if self.iteration % self.config.training.validation_interval == 0:
                logger.info(f"Iteration {self.iteration}: Evaluating model in arena")
                accepted = self.evaluate_model()
            
            # Save checkpoint
            if self.iteration % self.config.training.checkpoint_interval == 0:
                self.save_checkpoint()
            
            # Log iteration summary
            iteration_time = time.time() - iteration_start
            logger.info(f"Iteration {self.iteration} completed in {iteration_time:.1f}s")
            logger.info(f"Training stats: {train_stats}")
            
        logger.info("Training completed!")
        
        # Run final tournament if requested (disabled by default)
        # self.run_final_tournament()
    
    def generate_self_play_data(self) -> List[GameExample]:
        """Generate self-play training data with progress tracking"""
        from mcts.core.evaluator import AlphaZeroEvaluator
        
        examples = []
        
        # Create evaluator for self-play
        evaluator = AlphaZeroEvaluator(
            model=self.model,
            device=self.config.mcts.device
        )
        
        # Use multiprocessing for parallel self-play
        if self.config.training.num_workers > 1:
            examples = self._parallel_self_play(evaluator)
        else:
            # Single-threaded self-play with progress bar
            for game_idx in trange(self.config.training.num_games_per_iteration,
                                  desc="Self-play games", unit="game"):
                game_examples = self._play_single_game(evaluator, game_idx)
                examples.extend(game_examples)
        
        return examples
    
    def _parallel_self_play(self, evaluator) -> List[GameExample]:
        """Run self-play games in parallel"""
        from mcts.utils.safe_multiprocessing import serialize_state_dict_for_multiprocessing
        
        examples = []
        
        # Serialize model state dict for safe multiprocessing
        logger.info("Serializing model state dict for multiprocessing...")
        model_state_dict = serialize_state_dict_for_multiprocessing(self.model.state_dict())
        
        with ProcessPoolExecutor(max_workers=self.config.training.num_workers) as executor:
            # Submit all games
            futures = []
            for game_idx in range(self.config.training.num_games_per_iteration):
                future = executor.submit(
                    play_self_play_game_wrapper,
                    self.config,
                    model_state_dict,
                    game_idx,
                    self.iteration
                )
                futures.append(future)
            
            # Collect results with progress bar
            with tqdm(total=len(futures), desc="Self-play games", unit="game") as pbar:
                for future in as_completed(futures):
                    try:
                        game_examples = future.result()
                        examples.extend(game_examples)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Self-play game failed: {e}")
                        pbar.update(1)
        
        return examples
    
    def _play_single_game(self, evaluator, game_idx: int) -> List[GameExample]:
        """Play a single self-play game"""
        from mcts.core.mcts import MCTS, MCTSConfig
        from mcts.quantum.quantum_features import create_quantum_mcts
        
        # Create MCTS with quantum features if enabled
        mcts_config = MCTSConfig(
            num_simulations=self.config.mcts.num_simulations,
            c_puct=self.config.mcts.c_puct,
            temperature=1.0,
            dirichlet_alpha=self.config.mcts.dirichlet_alpha,
            dirichlet_epsilon=self.config.mcts.dirichlet_epsilon,
            device=self.config.mcts.device,
            game_type=self.game_type,
            min_wave_size=getattr(self.config.mcts, 'min_wave_size', self.config.mcts.wave_min_size),
            max_wave_size=getattr(self.config.mcts, 'max_wave_size', self.config.mcts.wave_max_size),
            adaptive_wave_sizing=getattr(self.config.mcts, 'adaptive_wave_sizing', self.config.mcts.wave_adaptive_sizing)
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
        
        # Play game
        examples = []
        state = self.game_interface.create_initial_state()
        game_id = f"iter{self.iteration}_game{game_idx}"
        
        for move_num in range(self.config.training.max_moves_per_game):
            # Get action probabilities from MCTS
            temperature = 1.0 if move_num < self.config.mcts.temperature_threshold else 0.0
            
            # Get full policy for storage
            policy = mcts.get_action_probabilities(state, temperature=temperature)
            
            
            # Get only valid actions and their probabilities for sampling
            try:
                valid_actions, valid_probs = mcts.get_valid_actions_and_probabilities(state, temperature=temperature)
            except AttributeError as e:
                # Fallback: get legal moves from game interface
                legal_moves = self.game_interface.get_legal_moves(state)
                
                # Filter policy to only legal moves
                legal_probs = policy[legal_moves]
                legal_probs = legal_probs / legal_probs.sum()
                valid_actions = legal_moves
                valid_probs = legal_probs
            
            if not valid_actions:
                # This should never happen in a properly implemented game
                raise ValueError(f"No valid actions available at move {move_num}")
            
            
            # Sample action from valid moves only
            action = np.random.choice(valid_actions, p=valid_probs)
            
            
            # Store example (from current player's perspective)
            canonical_state = self.game_interface.get_canonical_form(state)
            examples.append(GameExample(
                state=canonical_state,
                policy=policy,
                value=0,  # Will be filled with game outcome
                game_id=game_id,
                move_number=move_num
            ))
            
            # Apply action
            state = self.game_interface.get_next_state(state, action)
            
            # Update MCTS tree root for tree reuse
            mcts.update_root(action)
            
            # Check terminal
            if self.game_interface.is_terminal(state):
                # Get game outcome
                outcome = self.game_interface.get_value(state)
                
                # Update values in examples
                for i, example in enumerate(examples):
                    # Alternate perspective for each move
                    example.value = outcome * ((-1) ** (i % 2))
                
                break
        
        return examples
    
    def train_neural_network(self) -> Dict[str, float]:
        """Train the neural network on replay buffer"""
        if len(self.replay_buffer) < self.config.training.batch_size:
            return {"loss": 0, "policy_loss": 0, "value_loss": 0}
        
        # Create data loader
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0,  # Use 0 to avoid multiprocessing issues
            pin_memory=True
        )
        
        # Training stats
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0
        
        # Train for specified epochs
        self.model.train()
        for epoch in range(self.config.training.num_epochs):
            epoch_desc = f"Training epoch {epoch+1}/{self.config.training.num_epochs}"
            
            # Use tqdm for batch progress
            with tqdm(dataloader, desc=epoch_desc, unit="batch", 
                     disable=logger.level > logging.INFO) as pbar:
                for batch_idx, (states, target_policies, target_values) in enumerate(pbar):
                    # Move to device
                    states = states.to(self.config.mcts.device)
                    target_policies = target_policies.to(self.config.mcts.device)
                    target_values = target_values.to(self.config.mcts.device).squeeze()
                    
                    # Mixed precision training
                    if self.config.training.mixed_precision:
                        with torch.cuda.amp.autocast():
                            # Forward pass
                            pred_policies, pred_values = self.model(states)
                            pred_values = pred_values.squeeze()
                            
                            # Calculate losses
                            policy_loss = self.policy_loss_fn(pred_policies, target_policies)
                            value_loss = self.value_loss_fn(pred_values, target_values)
                            loss = policy_loss + value_loss
                        
                        # Backward pass with scaling
                        self.scaler.scale(loss).backward()
                        
                        if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                            # Gradient clipping
                            self.scaler.unscale_(self.optimizer)
                            clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                            
                            # Optimizer step
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()
                    else:
                        # Standard training
                        pred_policies, pred_values = self.model(states)
                        pred_values = pred_values.squeeze()
                        
                        policy_loss = self.policy_loss_fn(pred_policies, target_policies)
                        value_loss = self.value_loss_fn(pred_values, target_values)
                        loss = policy_loss + value_loss
                        
                        loss.backward()
                        
                        if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                            clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                    
                    # Update stats
                    total_loss += loss.item()
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    num_batches += 1
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'p_loss': f'{policy_loss.item():.4f}',
                        'v_loss': f'{value_loss.item():.4f}'
                    })
        
        # Update learning rate
        if self.scheduler:
            self.scheduler.step()
        
        # Return average stats
        stats = {
            "loss": total_loss / num_batches,
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
        }
        
        if self.scheduler:
            stats["lr"] = self.scheduler.get_last_lr()[0]
        else:
            stats["lr"] = self.config.training.learning_rate
            
        return stats
    
    def evaluate_model(self) -> bool:
        """Evaluate current model against best model in arena"""
        if not self.arena or not self.best_model_iteration:
            # First model, automatically accept
            self._save_best_model()
            return True
        
        # Load best model
        best_model_path = self.best_model_dir / f"model_iter_{self.best_model_iteration}.pt"
        best_model = self._create_model()
        best_model.load_state_dict(torch.load(best_model_path))
        best_model.eval()
        
        # Run arena match
        logger.info(f"Running arena: Current (iter {self.iteration}) vs Best (iter {self.best_model_iteration})")
        
        wins, draws, losses = self.arena.compare_models(
            self.model, best_model,
            model1_name=f"iter_{self.iteration}",
            model2_name=f"iter_{self.best_model_iteration}"
        )
        
        win_rate = wins / (wins + draws + losses)
        logger.info(f"Arena results: {wins}W-{draws}D-{losses}L (win rate: {win_rate:.2%})")
        
        # Update ELO ratings
        if self.elo_tracker:
            self.elo_tracker.update_ratings(
                f"iter_{self.iteration}",
                f"iter_{self.best_model_iteration}",
                wins, draws, losses
            )
            
            current_elo = self.elo_tracker.get_rating(f"iter_{self.iteration}")
            best_elo = self.elo_tracker.get_rating(f"iter_{self.best_model_iteration}")
            logger.info(f"ELO ratings: Current={current_elo:.1f}, Best={best_elo:.1f}")
        else:
            # Create ELO tracker if needed
            from mcts.neural_networks.arena_module import ELOTracker
            self.elo_tracker = ELOTracker(k_factor=self.config.arena.elo_k_factor)
            self.elo_tracker.update_ratings(
                f"iter_{self.iteration}",
                f"iter_{self.best_model_iteration}",
                wins, draws, losses
            )
            
            current_elo = self.elo_tracker.get_rating(f"iter_{self.iteration}")
            best_elo = self.elo_tracker.get_rating(f"iter_{self.best_model_iteration}")
        
        # Accept if win rate exceeds threshold
        accepted = win_rate >= self.config.arena.win_threshold
        if accepted:
            logger.info(f"✓ New model accepted as best!")
            self._save_best_model()
        else:
            logger.info(f"✗ New model rejected, keeping previous best")
        
        # Save arena results
        self._save_arena_results({
            "iteration": self.iteration,
            "vs_iteration": self.best_model_iteration,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": win_rate,
            "accepted": accepted,
            "current_elo": current_elo if self.elo_tracker else None,
            "best_elo": best_elo if self.elo_tracker else None
        })
        
        return accepted
    
    def _save_best_model(self):
        """Save current model as best"""
        self.best_model_iteration = self.iteration
        model_path = self.best_model_dir / f"model_iter_{self.iteration}.pt"
        torch.save(self.model.state_dict(), model_path)
        
        # Also save as 'best_model.pt' for easy access
        best_path = self.best_model_dir / "best_model.pt"
        torch.save(self.model.state_dict(), best_path)
    
    def _save_arena_results(self, results: Dict):
        """Save arena evaluation results"""
        results_file = self.arena_log_dir / "arena_results.json"
        
        # Load existing results
        if results_file.exists():
            with open(results_file, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = []
        
        # Add timestamp
        results["timestamp"] = datetime.now().isoformat()
        all_results.append(results)
        
        # Save updated results
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    def save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint = {
            "iteration": self.iteration,
            "best_model_iteration": self.best_model_iteration,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config
        }
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        if self.elo_tracker:
            checkpoint["elo_ratings"] = self.elo_tracker.ratings
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_iter_{self.iteration}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save replay buffer separately (can be large)
        buffer_path = self.data_dir / f"replay_buffer_iter_{self.iteration}.pkl"
        self.replay_buffer.save(str(buffer_path))
        
        logger.info(f"Saved checkpoint at iteration {self.iteration}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        
        self.iteration = checkpoint["iteration"]
        self.best_model_iteration = checkpoint["best_model_iteration"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        if self.elo_tracker and "elo_ratings" in checkpoint:
            self.elo_tracker.ratings = checkpoint["elo_ratings"]
        
        # Try to load replay buffer
        buffer_path = self.data_dir / f"replay_buffer_iter_{self.iteration}.pkl"
        if buffer_path.exists():
            self.replay_buffer.load(str(buffer_path))
            logger.info(f"Loaded replay buffer with {len(self.replay_buffer)} examples")
    
    def run_final_tournament(self):
        """Run tournament between all saved models"""
        logger.info("Running final tournament between all models")
        
        # Find all saved models
        model_files = sorted(self.best_model_dir.glob("model_iter_*.pt"))
        if len(model_files) < 2:
            return
        
        # Load models
        models = {}
        for model_file in model_files[-10:]:  # Last 10 models
            iteration = int(model_file.stem.split('_')[-1])
            model = self._create_model()
            model.load_state_dict(torch.load(model_file))
            model.eval()
            models[f"iter_{iteration}"] = model
        
        # Run round-robin tournament
        from itertools import combinations
        results = {}
        
        model_pairs = list(combinations(models.keys(), 2))
        with tqdm(total=len(model_pairs), desc="Tournament matches") as pbar:
            for model1_name, model2_name in model_pairs:
                wins, draws, losses = self.arena.compare_models(
                    models[model1_name], models[model2_name],
                    model1_name=model1_name,
                    model2_name=model2_name
                )
                
                results[(model1_name, model2_name)] = (wins, draws, losses)
                pbar.update(1)
        
        # Calculate final standings
        standings = {}
        for model_name in models:
            total_wins = 0
            total_games = 0
            
            for (m1, m2), (w, d, l) in results.items():
                if m1 == model_name:
                    total_wins += w
                    total_games += w + d + l
                elif m2 == model_name:
                    total_wins += l
                    total_games += w + d + l
            
            standings[model_name] = {
                "wins": total_wins,
                "games": total_games,
                "win_rate": total_wins / total_games if total_games > 0 else 0
            }
        
        # Sort by win rate
        sorted_standings = sorted(
            standings.items(),
            key=lambda x: x[1]["win_rate"],
            reverse=True
        )
        
        # Print results
        logger.info("\nFinal Tournament Results:")
        logger.info("-" * 50)
        for rank, (model_name, stats) in enumerate(sorted_standings, 1):
            logger.info(
                f"{rank}. {model_name}: "
                f"{stats['wins']}/{stats['games']} "
                f"({stats['win_rate']:.1%} win rate)"
            )
        
        # Save tournament results
        tournament_results = {
            "timestamp": datetime.now().isoformat(),
            "models": list(models.keys()),
            "matches": [
                {
                    "model1": m1,
                    "model2": m2,
                    "wins": w,
                    "draws": d,
                    "losses": l
                }
                for (m1, m2), (w, d, l) in results.items()
            ],
            "standings": sorted_standings
        }
        
        results_file = self.arena_log_dir / "tournament_results.json"
        with open(results_file, 'w') as f:
            json.dump(tournament_results, f, indent=2)


# Wrapper function for multiprocessing
def play_self_play_game_wrapper(config: AlphaZeroConfig, model_state_dict: Dict,
                               game_idx: int, iteration: int) -> List[GameExample]:
    """Wrapper function for parallel self-play"""
    import os
    import sys
    import logging
    
    # Set up logging for debugging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    
    # Disable CUDA in worker processes
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Import here to avoid circular imports in multiprocessing
    from mcts.core.mcts import MCTS, MCTSConfig
    from mcts.core.evaluator import AlphaZeroEvaluator
    from mcts.core.game_interface import GameInterface, GameType
    from mcts.neural_networks.nn_model import create_model
    from mcts.utils.safe_multiprocessing import deserialize_state_dict_from_multiprocessing
    
    # Deserialize model state dict
    model_state_dict = deserialize_state_dict_from_multiprocessing(model_state_dict)
    
    # Force CPU device for workers
    import torch
    device = torch.device('cpu')
    
    # Recreate model
    game_type = GameType[config.game.game_type.upper()]
    game_interface = GameInterface(game_type, config.game.board_size)
    
    # Get action space size
    initial_state = game_interface.create_initial_state()
    action_size = game_interface.get_action_space_size(initial_state)
    
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
    model = model.to(device)
    model.eval()
    
    # Create evaluator
    from mcts.core.evaluator import AlphaZeroEvaluator
    evaluator = AlphaZeroEvaluator(
        model=model,
        device=str(device)  # Use CPU device in workers
    )
    
    # Create MCTS
    mcts_config = MCTSConfig(
        num_simulations=config.mcts.num_simulations,
        c_puct=config.mcts.c_puct,
        temperature=1.0,
        dirichlet_alpha=config.mcts.dirichlet_alpha,
        dirichlet_epsilon=config.mcts.dirichlet_epsilon,
        device=str(device),  # Use CPU device in workers
        game_type=game_type,
        # Use smaller wave sizes for CPU
        min_wave_size=32,
        max_wave_size=64,
        adaptive_wave_sizing=False,
        # Reduce memory usage on CPU
        memory_pool_size_mb=128,
        max_tree_nodes=50000
    )
    
    
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
            for i, example in enumerate(examples):
                example.value = outcome * ((-1) ** (i % 2))
            break
    
    return examples


if __name__ == "__main__":
    import argparse
    import sys
    
    # Set multiprocessing start method for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Unified AlphaZero Training Pipeline")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--game", type=str, default="gomoku", choices=["chess", "go", "gomoku"],
                        help="Game type (if not using config file)")
    parser.add_argument("--iterations", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--experiment", type=str, help="Experiment name")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = AlphaZeroConfig.load(args.config)
        if args.experiment:
            config.experiment_name = args.experiment
    else:
        # Create default config
        config = create_default_config(game_type=args.game)
        config.experiment_name = args.experiment or f"{args.game}_unified_training"
        
        # Set some defaults
        config.training.num_workers = 4
        config.training.num_games_per_iteration = 100
        config.arena.enabled = True
        config.arena.evaluation_interval = 10
    
    # Create and run pipeline
    pipeline = UnifiedTrainingPipeline(config, resume_from=args.resume)
    
    try:
        pipeline.train(num_iterations=args.iterations)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        pipeline.save_checkpoint()
        print("Checkpoint saved.")
        sys.exit(0)