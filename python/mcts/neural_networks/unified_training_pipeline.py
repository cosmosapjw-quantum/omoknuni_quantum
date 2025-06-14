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
import time

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
        
        # Loss functions (will be moved to device automatically when needed)
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
        # Save training data in the parent directory (outside python folder)
        parent_dir = Path(__file__).parent.parent.parent.parent  # Go up to omoknuni_quantum
        self.experiment_dir = parent_dir / "experiments" / self.config.experiment_name
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
        
        model = create_model(
            game_type=self.config.game.game_type,
            input_height=self.config.game.board_size,
            input_width=self.config.game.board_size,
            num_actions=action_size,
            input_channels=network_config.input_channels,  # Configurable input channels
            num_res_blocks=network_config.num_res_blocks,
            num_filters=network_config.num_filters,
            value_head_hidden_size=network_config.value_head_hidden_size
        )
        
        # Move model to device
        model = model.to(self.config.mcts.device)
        
        # Mixed precision training uses FP32 model with FP16 computations
        # PyTorch's autocast handles the conversions automatically
        
        # Debug: Verify model is on correct device
        logger.info(f"Model device: {next(model.parameters()).device}")
        logger.info(f"Target device: {self.config.mcts.device}")
        
        return model
    
    def _get_action_size(self):
        """Get action space size for the game"""
        initial_state = self.game_interface.create_initial_state()
        return self.game_interface.get_action_space_size(initial_state)
    
    def _create_optimizer(self):
        """Create optimizer based on configuration"""
        training_config = self.config.training
        
        if training_config.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=training_config.learning_rate,
                betas=(training_config.adam_beta1, training_config.adam_beta2),
                eps=training_config.adam_epsilon,
                weight_decay=training_config.weight_decay
            )
        elif training_config.optimizer.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=training_config.learning_rate,
                betas=(training_config.adam_beta1, training_config.adam_beta2),
                eps=training_config.adam_epsilon,
                weight_decay=training_config.weight_decay
            )
        elif training_config.optimizer.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=training_config.learning_rate,
                momentum=training_config.sgd_momentum,
                nesterov=training_config.sgd_nesterov,
                weight_decay=training_config.weight_decay
            )
        else:
            logger.warning(f"Unknown optimizer {training_config.optimizer}, defaulting to Adam")
            return optim.Adam(
                self.model.parameters(),
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay
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
            temperature_threshold=self.config.mcts.temperature_threshold,  # Use configurable threshold
            timeout_seconds=int(self.config.arena.time_limit_seconds) if self.config.arena.time_limit_seconds else 300,
            device=self.config.mcts.device,
            save_game_records=self.config.arena.save_game_records
        )
        
        return ArenaManager(self.config, arena_config)
    
    def train(self, num_iterations: int):
        """Run the complete training pipeline
        
        Args:
            num_iterations: Number of training iterations
        """
        logger.info(f"Starting training for {num_iterations} iterations")
        
        # Use trange for iteration progress with clean formatting
        print()  # Initial newline for clean start
        for i in trange(num_iterations, desc="Training iterations", 
                       initial=self.iteration, unit="iter", position=0, leave=True):
            self.iteration += 1
            iteration_start = time.time()
            
            # Print clean header with proper spacing
            tqdm.write(f"\n\n{'='*80}")
            tqdm.write(f"ITERATION {self.iteration}")
            tqdm.write(f"{'='*80}\n")
            
            # Phase 1: Self-play data generation
            tqdm.write("[1/4] Generating self-play data...")
            self_play_start = time.time()
            self_play_examples = self.generate_self_play_data()
            self_play_time = time.time() - self_play_start
            
            # Add to replay buffer
            self.replay_buffer.add(self_play_examples)
            tqdm.write(f"      Generated {len(self_play_examples)} examples in {self_play_time:.1f}s")
            tqdm.write(f"      Replay buffer size: {len(self.replay_buffer)}")
            
            # Phase 2: Neural network training  
            tqdm.write("\n[2/4] Training neural network...")
            train_start = time.time()
            train_stats = self.train_neural_network()
            train_time = time.time() - train_start
            tqdm.write(f"      Training completed in {train_time:.1f}s")
            tqdm.write(f"      Loss: {train_stats['loss']:.4f}, P-Loss: {train_stats['policy_loss']:.4f}, V-Loss: {train_stats['value_loss']:.4f}")
            
            # Phase 3: Arena evaluation (every epoch for better tracking)
            tqdm.write("\n[3/4] Arena evaluation...")
            arena_start = time.time()
            accepted = self.evaluate_model_with_elo()
            arena_time = time.time() - arena_start
            tqdm.write(f"      Arena completed in {arena_time:.1f}s")
            
            # Phase 4: Save checkpoint if needed
            if self.iteration % self.config.training.checkpoint_interval == 0:
                tqdm.write("\n[4/4] Saving checkpoint...")
                self.save_checkpoint()
                tqdm.write("      Checkpoint saved.")
            
            # Summary
            iteration_time = time.time() - iteration_start
            tqdm.write(f"\n{'='*80}")
            tqdm.write(f"Iteration {self.iteration} Summary:")
            tqdm.write(f"  Total time: {iteration_time:.1f}s")
            tqdm.write(f"  Self-play: {self_play_time:.1f}s, Training: {train_time:.1f}s, Arena: {arena_time:.1f}s")
            tqdm.write(f"  Model accepted: {accepted}")
            tqdm.write(f"{'='*80}")
            
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
        
        # Configure evaluator to return torch tensors for GPU operations
        evaluator._return_torch_tensors = True
        
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
        """Run self-play games in parallel using GPU service architecture"""
        import multiprocessing as mp
        from .self_play_module import SelfPlayManager
        
        # Ensure spawn method for CUDA compatibility
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
        
        # Use self-play manager
        self_play_manager = SelfPlayManager(self.config)
        examples = self_play_manager.generate_games(
            self.model, 
            self.iteration,
            num_games=self.config.training.num_games_per_iteration,
            num_workers=self.config.training.num_workers
        )
        
        return examples
    
    def _create_quantum_config(self):
        """Create quantum configuration from MCTS config"""
        from mcts.quantum.quantum_features import QuantumConfig
        
        return QuantumConfig(
            quantum_level=self.config.mcts.quantum_level.value,
            enable_quantum=self.config.mcts.enable_quantum,
            min_wave_size=self.config.mcts.min_wave_size,
            optimal_wave_size=self.config.mcts.max_wave_size,
            hbar_eff=self.config.mcts.quantum_coupling,
            phase_kick_strength=self.config.mcts.phase_kick_strength,
            interference_alpha=self.config.mcts.interference_alpha,
            coupling_strength=self.config.mcts.quantum_coupling,
            temperature=self.config.mcts.quantum_temperature,
            decoherence_rate=self.config.mcts.decoherence_rate,
            fast_mode=True,  # Enable fast mode for training
            device=self.config.mcts.device,
            use_mixed_precision=self.config.mcts.use_mixed_precision
        )
    
    def _play_single_game(self, evaluator, game_idx: int) -> List[GameExample]:
        """Play a single self-play game"""
        from mcts.core.mcts import MCTS, MCTSConfig
        from mcts.quantum.quantum_features import create_quantum_mcts
        from mcts.gpu.gpu_game_states import GameType as GPUGameType
        
        # Convert game_interface.GameType to gpu_game_states.GameType
        game_type_mapping = {
            'CHESS': GPUGameType.CHESS,
            'GO': GPUGameType.GO,
            'GOMOKU': GPUGameType.GOMOKU
        }
        gpu_game_type = game_type_mapping[self.game_type.name]
        
        # Create MCTS configuration from full config
        mcts_config = MCTSConfig(
            # Core MCTS parameters
            num_simulations=self.config.mcts.num_simulations,
            c_puct=self.config.mcts.c_puct,
            temperature=self.config.mcts.temperature,  # Will be adjusted during play
            dirichlet_alpha=self.config.mcts.dirichlet_alpha,
            dirichlet_epsilon=self.config.mcts.dirichlet_epsilon,
            
            # Performance and wave parameters
            min_wave_size=self.config.mcts.min_wave_size,
            max_wave_size=self.config.mcts.max_wave_size,
            adaptive_wave_sizing=self.config.mcts.adaptive_wave_sizing,
            
            # Memory and optimization
            memory_pool_size_mb=self.config.mcts.memory_pool_size_mb,
            max_tree_nodes=self.config.mcts.max_tree_nodes,
            use_mixed_precision=self.config.mcts.use_mixed_precision,
            use_cuda_graphs=self.config.mcts.use_cuda_graphs,
            use_tensor_cores=self.config.mcts.use_tensor_cores,
            
            # Game and device configuration
            device=self.config.mcts.device,
            game_type=gpu_game_type,
            board_size=self.config.game.board_size,
            
            # Enable quantum features if configured
            enable_quantum=self.config.mcts.enable_quantum,
            quantum_config=self._create_quantum_config() if self.config.mcts.enable_quantum else None,
            
            # Virtual loss for parallel exploration
            enable_virtual_loss=True,
            virtual_loss=self.config.mcts.virtual_loss
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
            # AlphaZero-style temperature annealing
            # Use exploration temperature for first N moves, then switch to deterministic
            if move_num < self.config.mcts.temperature_threshold:
                # Exploration phase - use stochastic temperature
                mcts.config.temperature = 1.0
            else:
                # Exploitation phase - deterministic play
                mcts.config.temperature = 0.0
            
            # Search with MCTS
            policy = mcts.search(state, num_simulations=self.config.mcts.num_simulations)
            
            # Convert to numpy if needed
            if isinstance(policy, torch.Tensor):
                policy = policy.cpu().numpy()
            
            # Select move based on policy
            # Note: MCTS already applies temperature scaling to visit counts using
            # the AlphaZero formula: policy[a] ∝ visits[a]^(1/temperature)
            if mcts.config.temperature == 0:
                # Deterministic - select highest probability move
                action = np.argmax(policy)
            else:
                # Stochastic - sample from the temperature-scaled distribution
                # Ensure we have a valid probability distribution
                if np.sum(policy) > 0:
                    policy = policy / np.sum(policy)  # Normalize
                    action = np.random.choice(len(policy), p=policy)
                else:
                    # Fallback to uniform random if all probabilities are zero
                    legal_moves = self.game_interface.get_legal_moves(state)
                    if not legal_moves:
                        raise ValueError(f"No legal actions available at move {move_num}")
                    action = np.random.choice(legal_moves)
            
            
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
            
            # Reset MCTS tree for next search (no tree reuse for better exploration)
            mcts.reset_tree()
            
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
        
        # Ensure model is on correct device (it may have been moved during self-play)
        self.model = self.model.to(self.config.mcts.device)
        
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
            # Use tqdm for batch progress with position parameter to avoid overlap
            with tqdm(dataloader, desc=epoch_desc, unit="batch", 
                     position=1, leave=False) as pbar:
                for batch_idx, (states, target_policies, target_values) in enumerate(pbar):
                    # Move to device
                    states = states.to(self.config.mcts.device)
                    target_policies = target_policies.to(self.config.mcts.device)
                    target_values = target_values.to(self.config.mcts.device).squeeze()
                    
                    # Mixed precision training
                    if self.config.training.mixed_precision:
                        with torch.amp.autocast('cuda', dtype=torch.float16):
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
    
    def evaluate_model_with_elo(self) -> bool:
        """3-way ELO evaluation: Random vs Best vs Current model"""
        from mcts.core.evaluator import RandomEvaluator
        
        # Initialize ELO tracker if needed
        if not self.elo_tracker:
            from mcts.neural_networks.arena_module import ELOTracker
            self.elo_tracker = ELOTracker(
                k_factor=self.config.arena.elo_k_factor,
                initial_rating=self.config.arena.elo_initial_rating
            )
            # Set random model as anchor with rating 0
            self.elo_tracker.ratings["random"] = self.config.arena.elo_anchor_rating
        
        # Create random evaluator
        from mcts.core.evaluator import EvaluatorConfig
        eval_config = EvaluatorConfig(
            device=self.config.mcts.device,
            use_fp16=self.config.mcts.use_mixed_precision
        )
        random_evaluator = RandomEvaluator(
            eval_config, 
            self._get_action_size()
        )
        
        results_summary = []
        accepted = True
        
        # Match 1: Current vs Random
        tqdm.write("      Current vs Random...")
        wins_vs_random, draws_vs_random, losses_vs_random = self.arena.compare_models(
            self.model, random_evaluator,
            model1_name=f"iter_{self.iteration}",
            model2_name="random",
            silent=True
        )
        win_rate_vs_random = wins_vs_random / (wins_vs_random + draws_vs_random + losses_vs_random)
        tqdm.write(f"      Result: {wins_vs_random}W-{draws_vs_random}D-{losses_vs_random}L ({win_rate_vs_random:.1%})")
        
        # Update ELO
        self.elo_tracker.update_ratings(
            f"iter_{self.iteration}", "random",
            wins_vs_random, draws_vs_random, losses_vs_random
        )
        
        # Match 2: Best vs Random (if we have a best model)
        if self.best_model_iteration:
            tqdm.write("      Best vs Random...")
            best_model_path = self.best_model_dir / f"model_iter_{self.best_model_iteration}.pt"
            best_model = self._create_model()
            best_model.load_state_dict(torch.load(best_model_path))
            best_model.eval()
            
            wins_best_random, draws_best_random, losses_best_random = self.arena.compare_models(
                best_model, random_evaluator,
                model1_name=f"iter_{self.best_model_iteration}",
                model2_name="random",
                silent=True
            )
            win_rate_best_random = wins_best_random / (wins_best_random + draws_best_random + losses_best_random)
            tqdm.write(f"      Result: {wins_best_random}W-{draws_best_random}D-{losses_best_random}L ({win_rate_best_random:.1%})")
            
            self.elo_tracker.update_ratings(
                f"iter_{self.best_model_iteration}", "random",
                wins_best_random, draws_best_random, losses_best_random
            )
            
            # Match 3: Current vs Best
            tqdm.write("      Current vs Best...")
            wins_vs_best, draws_vs_best, losses_vs_best = self.arena.compare_models(
                self.model, best_model,
                model1_name=f"iter_{self.iteration}",
                model2_name=f"iter_{self.best_model_iteration}",
                silent=True
            )
            win_rate_vs_best = wins_vs_best / (wins_vs_best + draws_vs_best + losses_vs_best)
            tqdm.write(f"      Result: {wins_vs_best}W-{draws_vs_best}D-{losses_vs_best}L ({win_rate_vs_best:.1%})")
            
            self.elo_tracker.update_ratings(
                f"iter_{self.iteration}", f"iter_{self.best_model_iteration}",
                wins_vs_best, draws_vs_best, losses_vs_best
            )
            
            # Check if current model should be accepted
            accepted = win_rate_vs_best >= self.config.arena.win_threshold
        else:
            # First model - check against random baseline
            accepted = win_rate_vs_random >= self.config.arena.min_win_rate_vs_random
        
        # Get final ELO ratings
        current_elo = self.elo_tracker.get_rating(f"iter_{self.iteration}")
        random_elo = self.elo_tracker.get_rating("random")
        
        tqdm.write(f"\n      ELO Ratings:")
        tqdm.write(f"        Random: {random_elo:.1f} (anchor)")
        tqdm.write(f"        Current (iter {self.iteration}): {current_elo:.1f}")
        
        # Save detailed results
        results = {
            "iteration": self.iteration,
            "vs_random": {
                "wins": wins_vs_random,
                "draws": draws_vs_random,
                "losses": losses_vs_random,
                "win_rate": win_rate_vs_random
            },
            "elo_ratings": {
                "random": random_elo,
                "current": current_elo
            },
            "accepted": accepted
        }
        
        if self.best_model_iteration:
            best_elo = self.elo_tracker.get_rating(f"iter_{self.best_model_iteration}")
            tqdm.write(f"        Best (iter {self.best_model_iteration}): {best_elo:.1f}")
            
            # Add vs_best results only if we have them
            results["vs_best"] = {
                "wins": wins_vs_best,
                "draws": draws_vs_best,
                "losses": losses_vs_best,
                "win_rate": win_rate_vs_best
            }
            results["elo_ratings"]["best"] = best_elo
        
        # Save as best model if accepted
        if accepted:
            self._save_best_model()
            tqdm.write(f"\n      ✓ Model accepted as new best!")
        else:
            tqdm.write(f"\n      ✗ Model rejected")
            
        self._save_arena_results(results)
        
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
        # Ensure model is on correct device after loading checkpoint
        self.model = self.model.to(self.config.mcts.device)
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


# Old wrapper function removed - now using GPU service architecture from self_play_module
# The old play_self_play_game_wrapper function has been removed.
# Workers now use RemoteEvaluator to communicate with GPU service for neural network evaluation
# while using CPU for tree operations to avoid GPU memory exhaustion.


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