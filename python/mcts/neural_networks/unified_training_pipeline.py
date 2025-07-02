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
        # Ensure arrays are numpy arrays with positive strides
        if not isinstance(self.state, np.ndarray):
            self.state = np.ascontiguousarray(self.state)
        else:
            self.state = np.ascontiguousarray(self.state)
        if not isinstance(self.policy, np.ndarray):
            self.policy = np.ascontiguousarray(self.policy)
        else:
            self.policy = np.ascontiguousarray(self.policy)


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
        # Ensure arrays have positive strides by making copies if needed
        state = np.ascontiguousarray(example.state)
        policy = np.ascontiguousarray(example.policy)
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(policy),
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
        input_representation = getattr(config.network, 'input_representation', 'enhanced')
        self.game_interface = GameInterface(
            self.game_type,
            board_size=config.game.board_size,
            input_representation=input_representation
        )
        
        # Initialize components
        self.replay_buffer = ReplayBuffer(config.training.window_size)
        self.model = self._create_model()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Loss functions (will be moved to device automatically when needed)
        # For AlphaZero, we need to use KL divergence for policy loss since targets are distributions
        self.policy_loss_fn = self._policy_loss_kl_div
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
            logger.info(f"Resumed from checkpoint: iteration {self.iteration}")
        else:
            logger.info("Starting fresh training")
    
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
        from mcts.neural_networks.resnet_model import create_resnet_for_game
        
        # Get network config
        network_config = self.config.network
        
        # Get action space size from initial state
        initial_state = self.game_interface.create_initial_state()
        action_size = self.game_interface.get_action_space_size(initial_state)
        
        model = create_resnet_for_game(
            game_type=self.config.game.game_type,
            input_channels=network_config.input_channels,  # Configurable input channels
            num_blocks=network_config.num_res_blocks,
            num_filters=network_config.num_filters
        )
        
        # Move model to device
        model = model.to(self.config.mcts.device)
        
        # Mixed precision training uses FP32 model with FP16 computations
        # PyTorch's autocast handles the conversions automatically
        
        # Debug: Verify model is on correct device
        logger.debug(f"Model device: {next(model.parameters()).device}")
        logger.debug(f"Target device: {self.config.mcts.device}")
        
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
    
    def _policy_loss_kl_div(self, pred_log_probs, target_probs):
        """Calculate KL divergence loss for policy
        
        AlphaZero uses the MCTS visit counts as a probability distribution target.
        We need to calculate the cross-entropy between the network's output and this distribution.
        
        Args:
            pred_log_probs: Network output (already log probabilities) (batch_size, action_size)
            target_probs: Target probability distributions from MCTS (batch_size, action_size)
            
        Returns:
            KL divergence loss
        """
        # The network already outputs log probabilities, so we don't need to apply log_softmax
        # Calculate cross-entropy: -sum(target * log(pred))
        # This is equivalent to KL divergence when target sums to 1
        loss = -(target_probs * pred_log_probs).sum(dim=1).mean()
        
        return loss
    
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
        # Calculate actual iterations to run
        start_iteration = self.iteration
        # When resuming, num_iterations should be the total target, not additional iterations
        target_iteration = num_iterations
        remaining_iterations = target_iteration - start_iteration
        
        if remaining_iterations <= 0:
            logger.info(f"Already completed {start_iteration} iterations out of {num_iterations}. Training complete.")
            return
        
        logger.info(f"Training from iteration {start_iteration + 1} to {target_iteration}")
        if start_iteration > 0:
            logger.info(f"Resuming from previous checkpoint at iteration {start_iteration}")
        
        # Use trange for iteration progress with clean formatting
        print()  # Initial newline for clean start
        
        # Create progress bar that shows correct progress when resuming
        pbar = trange(target_iteration, desc="Training iterations", 
                     initial=start_iteration, unit="iter", position=0, leave=True)
        
        # Pre-compile CUDA kernels to avoid JIT overhead during training
        if self.config.mcts.device == 'cuda' and start_iteration == 0:
            logger.info("🔧 Pre-compiling CUDA kernels...")
            self._precompile_cuda_kernels()
        
        for _ in range(remaining_iterations):
            self.iteration += 1
            pbar.update(1)
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
            
            # Apply data augmentation if enabled
            if self.config.training.augment_data:
                augmented_examples = self._augment_training_data(self_play_examples)
                tqdm.write(f"      Applied data augmentation: {len(self_play_examples)} → {len(augmented_examples)} examples")
                self.replay_buffer.add(augmented_examples)
            else:
                self.replay_buffer.add(self_play_examples)
            
            final_examples = len(augmented_examples) if self.config.training.augment_data else len(self_play_examples)
            tqdm.write(f"      Generated {final_examples} examples in {self_play_time:.1f}s")
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
            
            # Phase 4: Save checkpoint every iteration
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
            
        # Log profiling summary if enabled
        try:
            from mcts.utils.training_profiler import get_profiler
            if get_profiler().enabled:
                tqdm.write("\n" + "=" * 80)
                tqdm.write("TRAINING PROFILING SUMMARY")
                tqdm.write("=" * 80)
                log_profiling_summary(top_n=30, min_time=0.1)
        except ImportError:
            # Profiler not available, skip logging
            pass
        
        logger.info("Training completed!")
        
        # Run final tournament if requested (disabled by default)
        # self.run_final_tournament()
    
    def generate_self_play_data(self) -> List[GameExample]:
        """Generate self-play training data with progress tracking"""
        from mcts.core.evaluator import AlphaZeroEvaluator
        
        examples = []
        
        # Initialize comprehensive game quality metrics
        self.game_metrics = {
            'game_lengths': [],
            'policy_entropies': [],
            'value_trajectories': [],
            'outcomes': {'player1_wins': 0, 'player2_wins': 0, 'draws': 0},
            'resignation_count': 0,
            'illegal_move_attempts': 0,
            'avg_branching_factor': [],
            'move_time_ms': [],
            'final_positions': [],
            'value_accuracy': []  # Compare predicted vs actual outcome
        }
        
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
        
        # Log comprehensive game quality report
        self._log_game_quality_metrics()
        
        return examples
    
    def _log_game_quality_metrics(self):
        """Log comprehensive game quality metrics"""
        if not self.game_metrics['game_lengths']:
            logger.warning("No game metrics collected!")
            return
            
        import numpy as np
        
        # Calculate statistics
        avg_length = np.mean(self.game_metrics['game_lengths'])
        std_length = np.std(self.game_metrics['game_lengths'])
        min_length = np.min(self.game_metrics['game_lengths'])
        max_length = np.max(self.game_metrics['game_lengths'])
        
        total_games = sum(self.game_metrics['outcomes'].values())
        
        # Calculate win rates
        p1_win_rate = self.game_metrics['outcomes']['player1_wins'] / max(total_games, 1)
        p2_win_rate = self.game_metrics['outcomes']['player2_wins'] / max(total_games, 1)
        draw_rate = self.game_metrics['outcomes']['draws'] / max(total_games, 1)
        
        # Policy entropy analysis
        if self.game_metrics['policy_entropies']:
            avg_entropy = np.mean(self.game_metrics['policy_entropies'])
            entropy_trend = 'decreasing' if len(self.game_metrics['policy_entropies']) > 10 and \
                           np.mean(self.game_metrics['policy_entropies'][-10:]) < \
                           np.mean(self.game_metrics['policy_entropies'][:10]) else 'stable'
        else:
            avg_entropy = 0
            entropy_trend = 'unknown'
        
        # Value accuracy
        if self.game_metrics['value_accuracy']:
            value_acc = np.mean(self.game_metrics['value_accuracy'])
        else:
            value_acc = 0
        
        logger.info("="*80)
        logger.info("SELF-PLAY GAME QUALITY METRICS")
        logger.info("="*80)
        logger.info(f"Total games played: {total_games}")
        logger.info(f"Game length: avg={avg_length:.1f}, std={std_length:.1f}, min={min_length}, max={max_length}")
        logger.info(f"Outcomes: P1 wins={p1_win_rate:.1%}, P2 wins={p2_win_rate:.1%}, Draws={draw_rate:.1%}")
        logger.info(f"Resignations: {self.game_metrics['resignation_count']} ({self.game_metrics['resignation_count']/max(total_games,1):.1%})")
        logger.info(f"Average policy entropy: {avg_entropy:.3f} (trend: {entropy_trend})")
        logger.info(f"Value prediction accuracy: {value_acc:.1%}")
        logger.info(f"Illegal move attempts: {self.game_metrics['illegal_move_attempts']}")
        
        if self.game_metrics['avg_branching_factor']:
            logger.info(f"Average branching factor: {np.mean(self.game_metrics['avg_branching_factor']):.1f}")
        
        # Check for potential issues
        if avg_length < 20:
            logger.warning("⚠️  Games are ending too quickly! Check game logic or increase exploration.")
        if p1_win_rate > 0.65 or p2_win_rate > 0.65:
            logger.warning("⚠️  High win rate imbalance detected! Check for first-player advantage.")
        if avg_entropy < 0.5:
            logger.warning("⚠️  Low policy entropy! Model may be overly deterministic.")
        if self.game_metrics['illegal_move_attempts'] > 0:
            logger.error("❌ Illegal moves attempted! Check MCTS action selection.")
        
        logger.info("="*80)
    
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
        from mcts.gpu.gpu_game_states import GameType as GPUGameType
        
        # Only import quantum features if quantum is enabled
        if getattr(self.config.mcts, 'enable_quantum', False):
            try:
                from mcts.quantum.quantum_features import create_quantum_mcts
            except ImportError:
                # Quantum features not available, disable quantum
                pass
        
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
        
        # Track game-specific metrics
        game_value_trajectory = []
        game_policy_entropies = []
        game_start_time = time.time()
        
        for move_num in range(self.config.training.max_moves_per_game):
            # Enhanced opening diversity: Force extra exploration for first 3 moves
            if move_num < 3:
                # CRITICAL: Extra high temperature + noise for opening diversity
                mcts.config.temperature = max(2.0, self.config.mcts.temperature)
                mcts.config.dirichlet_epsilon = min(0.8, self.config.mcts.dirichlet_epsilon * 2.0)
            elif move_num < self.config.mcts.temperature_threshold:
                # Normal exploration phase
                mcts.config.temperature = self.config.mcts.temperature
                mcts.config.dirichlet_epsilon = self.config.mcts.dirichlet_epsilon
            else:
                # Exploitation phase - lower temperature but not fully deterministic
                mcts.config.temperature = getattr(self.config.mcts, 'temperature_final', 0.1)
                mcts.config.dirichlet_epsilon = self.config.mcts.dirichlet_epsilon * 0.5
            
            # Track search time
            search_start = time.time()
            
            # Search with MCTS
            policy = mcts.search(state, num_simulations=self.config.mcts.num_simulations)
            
            # Track search time
            search_time_ms = (time.time() - search_start) * 1000
            if hasattr(self, 'game_metrics'):
                self.game_metrics['move_time_ms'].append(search_time_ms)
            
            # Convert to numpy if needed
            if isinstance(policy, torch.Tensor):
                policy = policy.cpu().numpy()
            
            # Calculate policy entropy for diversity analysis
            policy_entropy = -np.sum(policy * np.log(policy + 1e-8))
            game_policy_entropies.append(policy_entropy)
            
            # Track root value from MCTS
            root_value = mcts.get_root_value() if hasattr(mcts, 'get_root_value') else 0
            game_value_trajectory.append(root_value)
            
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
                
                # Collect final game metrics
                if hasattr(self, 'game_metrics'):
                    # Record game length
                    self.game_metrics['game_lengths'].append(move_num + 1)
                    
                    # Record outcome
                    if outcome == 1:
                        self.game_metrics['outcomes']['player1_wins'] += 1
                    elif outcome == -1:
                        self.game_metrics['outcomes']['player2_wins'] += 1
                    else:
                        self.game_metrics['outcomes']['draws'] += 1
                    
                    # Record policy entropies
                    self.game_metrics['policy_entropies'].extend(game_policy_entropies)
                    
                    # Record value trajectory
                    self.game_metrics['value_trajectories'].append(game_value_trajectory)
                    
                    # Calculate value prediction accuracy
                    if game_value_trajectory:
                        # Check if final value predictions matched actual outcome
                        final_predictions = game_value_trajectory[-min(5, len(game_value_trajectory)):]
                        for pred in final_predictions:
                            # Account for perspective alternation
                            correct = (pred > 0 and outcome == 1) or (pred < 0 and outcome == -1)
                            self.game_metrics['value_accuracy'].append(float(correct))
                
                break
        
        return examples
    
    def _augment_training_data(self, examples: List[GameExample]) -> List[GameExample]:
        """Apply data augmentation to training examples using board symmetries"""
        augmented_examples = []
        
        for example in examples:
            # Get board and policy from the example
            board = example.state
            policy = example.policy
            
            # Apply symmetries (rotations and reflections)
            symmetries = self.game_interface.get_symmetries(board, policy)
            
            for aug_board, aug_policy in symmetries:
                # Create augmented example
                augmented_example = GameExample(
                    state=aug_board,
                    policy=aug_policy,
                    value=example.value,  # Value doesn't change with symmetry
                    game_id=example.game_id + f"_sym{len(augmented_examples)}",
                    move_number=example.move_number
                )
                augmented_examples.append(augmented_example)
        
        return augmented_examples
    
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
        import gc
        
        # Initialize ELO tracker if needed
        if not self.elo_tracker:
            from mcts.neural_networks.arena_module import ELOTracker
            self.elo_tracker = ELOTracker(
                k_factor=self.config.arena.elo_k_factor,
                initial_rating=self.config.arena.elo_initial_rating
            )
            # Set random model as anchor with rating 0
            self.elo_tracker.ratings["random"] = self.config.arena.elo_anchor_rating
            
            # Configure enhanced ELO features based on config
            if hasattr(self.config.arena, 'elo_enable_deflation'):
                self.elo_tracker.use_deflation = self.config.arena.elo_enable_deflation
            if hasattr(self.config.arena, 'elo_deflation_factor'):
                self.elo_tracker.deflation_factor = self.config.arena.elo_deflation_factor
                
            logger.info(f"ELO Tracker initialized with enhanced features:")
            logger.info(f"  K-factor: {self.elo_tracker.k_factor}")
            logger.info(f"  Initial rating: {self.elo_tracker.initial_rating}")
            logger.info(f"  Deflation enabled: {self.elo_tracker.use_deflation}")
            logger.info(f"  Uncertainty tracking: enabled")
            logger.info(f"  Validation metrics: enabled")
        
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
        
        # Clear GPU cache before starting arena battles
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Implement CORRECT ELO inheritance: always inherit from previous iteration
        current_key = f"iter_{self.iteration}"
        if current_key not in self.elo_tracker.ratings:
            if self.iteration == 1:
                # First model starts at 0 (same as random)
                initial_elo = 0.0
                logger.info(f"First model (iter_1) starting at ELO {initial_elo:.1f} (same as random)")
            else:
                # ALWAYS inherit from previous iteration model, not best model
                # This is the correct approach - new models are trained from previous iteration
                previous_key = f"iter_{self.iteration - 1}"
                previous_elo = self.elo_tracker.get_rating(previous_key)
                if previous_elo is not None:
                    initial_elo = previous_elo
                    logger.info(f"Model iter_{self.iteration} inheriting ELO {initial_elo:.1f} from previous {previous_key}")
                else:
                    # Fallback to initial rating if previous doesn't exist
                    initial_elo = self.config.arena.elo_initial_rating
                    logger.info(f"Model iter_{self.iteration} starting at initial ELO {initial_elo:.1f} (previous not found)")
            
            self.elo_tracker.ratings[current_key] = initial_elo
            # Store initial ELO for potential adjustment
            self._initial_inherited_elo = initial_elo
        
        # Match 1: Current vs Random (using adaptive logic like best model)
        current_model_elo = self.elo_tracker.get_rating(f"iter_{self.iteration}")
        
        # Use adaptive logic if enabled in config
        if self.config.arena.enable_adaptive_random_matches:
            should_current_play_random = self.elo_tracker.should_play_vs_random(
                self.iteration, current_model_elo
            )
        else:
            # Always play random matches if adaptive logic is disabled
            should_current_play_random = True
        
        wins_vs_random = 0
        draws_vs_random = 0
        losses_vs_random = 0
        win_rate_vs_random = 0.0
        
        if should_current_play_random:
            tqdm.write("      Current vs Random (adaptive check)...")
            wins_vs_random, draws_vs_random, losses_vs_random = self.arena.compare_models(
                self.model, random_evaluator,
                model1_name=f"iter_{self.iteration}",
                model2_name="random",
                silent=False
            )
            win_rate_vs_random = wins_vs_random / (wins_vs_random + draws_vs_random + losses_vs_random)
            tqdm.write(f"      Result: {wins_vs_random}W-{draws_vs_random}D-{losses_vs_random}L ({win_rate_vs_random:.1%})")
            
            # Update ELO
            self.elo_tracker.update_ratings(
                f"iter_{self.iteration}", "random",
                wins_vs_random, draws_vs_random, losses_vs_random
            )
        else:
            tqdm.write("      Current vs Random: Skipped (adaptive criteria)")
            # For first model or when forced, still run the match
            if self.iteration == 1 or not self.best_model_iteration:
                tqdm.write("      Current vs Random (forced for first model)...")
                wins_vs_random, draws_vs_random, losses_vs_random = self.arena.compare_models(
                    self.model, random_evaluator,
                    model1_name=f"iter_{self.iteration}",
                    model2_name="random",
                    silent=False
                )
                win_rate_vs_random = wins_vs_random / (wins_vs_random + draws_vs_random + losses_vs_random)
                tqdm.write(f"      Result: {wins_vs_random}W-{draws_vs_random}D-{losses_vs_random}L ({win_rate_vs_random:.1%})")
                
                # Update ELO
                self.elo_tracker.update_ratings(
                    f"iter_{self.iteration}", "random",
                    wins_vs_random, draws_vs_random, losses_vs_random
                )
        
        # Clean up after first match
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Match 2: Current vs Previous Model (for ELO calibration)
        wins_vs_previous = 0
        draws_vs_previous = 0
        losses_vs_previous = 0
        win_rate_vs_previous = 0.0
        
        if self.iteration > 1 and self.config.arena.enable_current_vs_previous:
            # Check if previous model exists
            previous_model_path = self.experiment_dir / "checkpoints" / f"checkpoint_iter_{self.iteration - 1}.pt"
            
            if previous_model_path.exists():
                tqdm.write(f"      Current vs Previous (iter_{self.iteration - 1})...")
                
                # Move current model to CPU temporarily to free GPU memory
                current_device = next(self.model.parameters()).device
                self.model.cpu()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Load previous model
                previous_model = self._create_model()
                checkpoint = torch.load(previous_model_path, map_location=self.config.mcts.device, weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    previous_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    previous_model.load_state_dict(checkpoint)
                previous_model.eval()
                
                # Move current model back to device for comparison
                self.model.to(current_device)
                
                # Run arena match
                wins_vs_previous, draws_vs_previous, losses_vs_previous = self.arena.compare_models(
                    self.model, previous_model,
                    model1_name=f"iter_{self.iteration}",
                    model2_name=f"iter_{self.iteration - 1}",
                    silent=False
                )
                win_rate_vs_previous = wins_vs_previous / (wins_vs_previous + draws_vs_previous + losses_vs_previous)
                tqdm.write(f"      Result: {wins_vs_previous}W-{draws_vs_previous}D-{losses_vs_previous}L ({win_rate_vs_previous:.1%})")
                
                # Update ELO based on actual performance vs previous model
                self.elo_tracker.update_ratings(
                    f"iter_{self.iteration}", f"iter_{self.iteration - 1}",
                    wins_vs_previous, draws_vs_previous, losses_vs_previous
                )
                
                # Clean up previous model
                del previous_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                tqdm.write(f"      Current vs Previous: Previous model iter_{self.iteration - 1} not found")
        elif self.iteration > 1:
            tqdm.write("      Current vs Previous: Disabled in config")
        else:
            tqdm.write("      Current vs Previous: Skipped (first model)")
        
        # Match 3: Best vs Random (if we have a best model)
        if self.best_model_iteration:
            # First, move current model to CPU temporarily to free GPU memory
            current_device = next(self.model.parameters()).device
            self.model.cpu()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Use sophisticated adaptive logic to determine if best model should play random
            best_model_key = f"iter_{self.best_model_iteration}"
            best_model_elo = self.elo_tracker.get_rating(best_model_key)
            
            # Check if best model should play against random using adaptive criteria
            should_best_play_random = self.elo_tracker.should_play_vs_random(
                self.best_model_iteration, best_model_elo
            )
            
            if should_best_play_random:
                tqdm.write(f"      Best vs Random (adaptive check)...")
                best_model_path = self.best_model_dir / f"model_iter_{self.best_model_iteration}.pt"
                best_model = self._create_model()
                
                # Load best model directly to GPU
                checkpoint = torch.load(best_model_path, map_location=self.config.mcts.device, weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    best_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    best_model.load_state_dict(checkpoint)
                best_model.eval()
                
                wins_best_random, draws_best_random, losses_best_random = self.arena.compare_models(
                    best_model, random_evaluator,
                    model1_name=f"iter_{self.best_model_iteration}",
                    model2_name="random",
                    silent=False
                )
                win_rate_best_random = wins_best_random / (wins_best_random + draws_best_random + losses_best_random)
                tqdm.write(f"      Result: {wins_best_random}W-{draws_best_random}D-{losses_best_random}L ({win_rate_best_random:.1%})")
                
                self.elo_tracker.update_ratings(
                    f"iter_{self.best_model_iteration}", "random",
                    wins_best_random, draws_best_random, losses_best_random
                )
            else:
                # Skip re-evaluation based on adaptive criteria
                tqdm.write(f"      Best model (iter {self.best_model_iteration}, ELO {best_model_elo:.1f}) skipping random match")
                tqdm.write(f"        Reason: Adaptive criteria (iteration={self.best_model_iteration}, ELO={best_model_elo:.1f})")
                best_model_path = self.best_model_dir / f"model_iter_{self.best_model_iteration}.pt"
                best_model = self._create_model()
                
                # Still need to load the model for current vs best comparison
                checkpoint = torch.load(best_model_path, map_location=self.config.mcts.device, weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    best_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    best_model.load_state_dict(checkpoint)
                best_model.eval()
            
            # Move best model to CPU before loading current model back
            best_model.cpu()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Move current model back to GPU
            self.model.to(current_device)
            
            # Match 4: Current vs Best
            tqdm.write("      Current vs Best...")
            
            # Move best model back to GPU for the match
            best_model.to(self.config.mcts.device)
            
            try:
                wins_vs_best, draws_vs_best, losses_vs_best = self.arena.compare_models(
                    self.model, best_model,
                    model1_name=f"iter_{self.iteration}",
                    model2_name=f"iter_{self.best_model_iteration}",
                    silent=False
                )
            except RuntimeError as e:
                if "CUDA error" in str(e):
                    logger.error(f"CUDA error during arena evaluation: {e}")
                    # Treat CUDA errors as draws to continue training
                    logger.warning("Treating all games as draws due to CUDA error")
                    wins_vs_best = 0
                    draws_vs_best = self.config.arena.num_games
                    losses_vs_best = 0
                else:
                    raise
            win_rate_vs_best = wins_vs_best / (wins_vs_best + draws_vs_best + losses_vs_best)
            tqdm.write(f"      Result: {wins_vs_best}W-{draws_vs_best}D-{losses_vs_best}L ({win_rate_vs_best:.1%})")
            
            # Update ratings with best model protection
            self.elo_tracker.update_ratings(
                f"iter_{self.iteration}", f"iter_{self.best_model_iteration}",
                wins_vs_best, draws_vs_best, losses_vs_best,
                protect_best_elo=True,
                best_player=f"iter_{self.best_model_iteration}"
            )
            
            # Store wins vs best for ELO adjustment if needed
            self._last_wins_vs_best = wins_vs_best
            self._last_games_vs_best = wins_vs_best + draws_vs_best + losses_vs_best
            
            # Check if current model should be accepted
            accepted = win_rate_vs_best >= self.config.arena.win_threshold
            
            # CRITICAL: Delete best model to free GPU memory
            del best_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            # First model - check against random baseline
            accepted = win_rate_vs_random >= self.config.arena.min_win_rate_vs_random
        
        # Get final ELO ratings
        current_elo = self.elo_tracker.get_rating(f"iter_{self.iteration}")
        random_elo = self.elo_tracker.get_rating("random")
        
        # Get enhanced ELO information
        current_rating, current_uncertainty = self.elo_tracker.get_rating_with_uncertainty(f"iter_{self.iteration}")
        current_games = self.elo_tracker.game_counts.get(f"iter_{self.iteration}", 0)
        
        tqdm.write(f"\n      ELO Ratings:")
        tqdm.write(f"        Random: {random_elo:.1f} (anchor)")
        tqdm.write(f"        Current (iter {self.iteration}): {current_elo:.1f} ±{current_uncertainty:.1f} ({current_games} games)")
        
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
        
        # Add vs_previous results if we have them
        if self.iteration > 1:
            results["vs_previous"] = {
                "wins": wins_vs_previous,
                "draws": draws_vs_previous,
                "losses": losses_vs_previous,
                "win_rate": win_rate_vs_previous
            }
            # Add previous model ELO if it exists
            previous_elo = self.elo_tracker.get_rating(f"iter_{self.iteration - 1}")
            if previous_elo is not None:
                results["elo_ratings"]["previous"] = previous_elo
                tqdm.write(f"        Previous (iter {self.iteration - 1}): {previous_elo:.1f}")
        
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
            
            # ELO consistency check and automatic adjustment
            if self.config.arena.enable_elo_consistency_checks and current_elo > best_elo and win_rate_vs_best < 0.5:
                logger.warning(f"ELO inconsistency detected: Current model (ELO {current_elo:.1f}) has higher ELO than best model (ELO {best_elo:.1f}) "
                              f"but only achieved {win_rate_vs_best:.1%} win rate against best model.")
                
                if self.config.arena.enable_elo_auto_adjustment:
                    # Calculate appropriate ELO adjustment
                    # If current model can't beat best model, its ELO should be lower than best model
                    # Adjust current model ELO to be slightly below best model ELO
                    elo_gap = current_elo - best_elo
                    
                    # Set current ELO to best ELO minus a penalty based on performance
                    performance_penalty = (0.5 - win_rate_vs_best) * 100  # Scale penalty by how much worse than 50%
                    adjusted_current_elo = best_elo - performance_penalty
                    
                    # Ensure the adjustment is reasonable (not too extreme)
                    max_adjustment = min(200, elo_gap * 1.5)  # Cap the adjustment
                    if (current_elo - adjusted_current_elo) > max_adjustment:
                        adjusted_current_elo = current_elo - max_adjustment
                        performance_penalty = current_elo - adjusted_current_elo
                    
                    # Apply the adjustment
                    old_current_elo = current_elo
                    self.elo_tracker.ratings[f"iter_{self.iteration}"] = adjusted_current_elo
                    current_elo = adjusted_current_elo  # Update local variable for display
                    
                    tqdm.write(f"        🔧 ELO Adjustment Applied:")
                    tqdm.write(f"           Current model: {old_current_elo:.1f} → {adjusted_current_elo:.1f} (Δ{adjusted_current_elo - old_current_elo:+.1f})")
                    tqdm.write(f"           Best model: {best_elo:.1f} (unchanged)")
                    tqdm.write(f"           Reason: Current won only {win_rate_vs_best:.1%} vs best, penalty = {performance_penalty:.1f}")
                    
                    logger.info(f"ELO adjustment applied: Current model ELO {old_current_elo:.1f} → {adjusted_current_elo:.1f} "
                               f"(penalty: {performance_penalty:.1f} based on {win_rate_vs_best:.1%} win rate vs best)")
                    
                    # Update the results with the corrected ELO
                    results["elo_ratings"]["current"] = adjusted_current_elo
                    results["elo_adjustment"] = {
                        "applied": True,
                        "old_elo": old_current_elo,
                        "new_elo": adjusted_current_elo,
                        "penalty": performance_penalty,
                        "reason": f"Win rate vs best: {win_rate_vs_best:.1%}",
                        "max_adjustment_capped": (current_elo - adjusted_current_elo) >= max_adjustment
                    }
                else:
                    # Just log the inconsistency without adjusting
                    tqdm.write(f"        ⚠️  ELO inconsistency: Current ELO {current_elo:.1f} > Best ELO {best_elo:.1f} but current only won {win_rate_vs_best:.1%} vs best")
                    tqdm.write(f"           Auto-adjustment disabled in config")
                    
                    results["elo_adjustment"] = {
                        "applied": False,
                        "detected_inconsistency": True,
                        "reason": "Auto-adjustment disabled in configuration"
                    }
            else:
                # No inconsistency detected or checks disabled
                results["elo_adjustment"] = {
                    "applied": False,
                    "detected_inconsistency": False,
                    "reason": "No ELO inconsistency detected"
                }
        else:
            # No best model yet - first model case
            results["elo_adjustment"] = {
                "applied": False,
                "detected_inconsistency": False,
                "reason": "No best model exists yet (first model)"
            }
        
        # Save as best model if accepted
        if accepted:
            # Ensure the new best model has a higher ELO than the previous best
            if self.best_model_iteration:
                old_best_elo = self.elo_tracker.get_rating(f"iter_{self.best_model_iteration}")
                current_elo = self.elo_tracker.get_rating(f"iter_{self.iteration}")
                if current_elo <= old_best_elo:
                    logger.warning(f"New best model has lower ELO ({current_elo:.1f}) than previous best ({old_best_elo:.1f})")
            
            self._save_best_model()
            # Get the final ELO after potential adjustment
            final_elo = self.elo_tracker.get_rating(f"iter_{self.iteration}")
            tqdm.write(f"\n      ✓ Model accepted as new best! (ELO: {final_elo:.1f})")
        else:
            tqdm.write(f"\n      ✗ Model rejected")
        
        # Log ELO health report every 10 iterations or if inflation is suspected
        if self.iteration % 10 == 0 or self.iteration < 5:
            try:
                health_report = self.elo_tracker.get_health_report()
                
                # Check for warnings
                inflation_indicators = health_report.get("inflation_indicators", {})
                if inflation_indicators.get("inflation_detected", False):
                    tqdm.write("\n      ⚠️  ELO INFLATION WARNING:")
                    tqdm.write(f"        Growth rate: {inflation_indicators.get('avg_growth_per_iteration', 0):.1f} ELO/iteration")
                    tqdm.write(f"        Total growth: {inflation_indicators.get('total_elo_growth', 0):.1f} ELO")
                
                # Show validation metrics if available
                validation = health_report.get("validation_metrics", {})
                if "prediction_accuracy" in validation:
                    tqdm.write(f"\n      ELO System Health:")
                    tqdm.write(f"        Prediction accuracy: {validation['prediction_accuracy']:.1%}")
                    if validation.get("rating_inflation_detected", False):
                        tqdm.write(f"        ⚠️  {validation.get('recommendation', '')}")
                
                # Log detailed summary to debug logs every 20 iterations
                if self.iteration % 20 == 0:
                    logger.debug("Generating detailed ELO health report...")
                    self.elo_tracker.log_detailed_summary()
                    
            except Exception as e:
                logger.warning(f"Could not generate ELO health report: {e}")
            
        self._save_arena_results(results)
        
        return accepted
    
    def _calculate_win_rate_vs_best(self) -> float:
        """Calculate win rate of current model vs best model from arena results"""
        if hasattr(self, '_last_wins_vs_best') and hasattr(self, '_last_games_vs_best'):
            if self._last_games_vs_best > 0:
                # We have data from the current vs best match
                wins = getattr(self, '_last_wins_vs_best', 0)
                games = getattr(self, '_last_games_vs_best', 1)
                return wins / games
        
        return 0.55  # Default slightly above 50% since we beat the best
    
    def _save_best_model(self):
        """Save current model as best"""
        # CRITICAL FIX: When a model becomes the new best (by beating the current best in arena),
        # its ELO MUST be higher than the previous best, regardless of inheritance or match calculations
        if self.elo_tracker and self.best_model_iteration:
            old_best_elo = self.elo_tracker.get_rating(f"iter_{self.best_model_iteration}")
            current_elo = self.elo_tracker.get_rating(f"iter_{self.iteration}")
            
            # ALWAYS ensure new best has higher ELO than previous best
            # This is a fundamental requirement: better models must have higher ratings
            if current_elo <= old_best_elo:
                # Calculate appropriate ELO boost based on performance
                win_rate_vs_best = self._calculate_win_rate_vs_best()
                if win_rate_vs_best >= 0.75:
                    min_gain = 25.0  # Strong victory: substantial boost
                elif win_rate_vs_best >= 0.60:
                    min_gain = 15.0  # Solid victory: good boost
                else:
                    min_gain = 10.0  # Close victory: modest boost
                
                # Set new ELO to old best + gain
                adjusted_elo = old_best_elo + min_gain
                
                # Log the adjustment (this should be expected, not an error)
                logger.warning(f"ELO inconsistency detected: Beat best model but ELO {current_elo:.1f} <= {old_best_elo:.1f}")
                logger.info(f"Adjusting ELO for new best model:")
                logger.info(f"  Current ELO from matches: {current_elo:.1f}")
                logger.info(f"  Previous best ELO: {old_best_elo:.1f}")
                logger.info(f"  Win rate vs best: {win_rate_vs_best:.1%}")
                logger.info(f"  Applied ELO boost: +{min_gain:.1f}")
                logger.info(f"  Final adjusted ELO: {adjusted_elo:.1f}")
                
                # Update the rating in the tracker
                self.elo_tracker.ratings[f"iter_{self.iteration}"] = adjusted_elo
                current_elo = adjusted_elo
            else:
                # Current ELO is already higher - ensure minimum gain
                min_acceptable_elo = old_best_elo + 5.0
                if current_elo < min_acceptable_elo:
                    adjusted_elo = min_acceptable_elo
                    logger.info(f"Ensuring minimum ELO gain for new best: {current_elo:.1f} -> {adjusted_elo:.1f}")
                    self.elo_tracker.ratings[f"iter_{self.iteration}"] = adjusted_elo
                    current_elo = adjusted_elo
            
            # Log the progression
            elo_gain = current_elo - old_best_elo
            logger.info(f"Best model progression: iter {self.best_model_iteration} (ELO: {old_best_elo:.1f}) -> iter {self.iteration} (ELO: {current_elo:.1f})")
            logger.info(f"Net ELO gain: {elo_gain:+.1f}")
            
            # Final safety check
            if elo_gain <= 0:
                logger.error(f"CRITICAL ERROR: New best model still has ELO gain of {elo_gain:+.1f}")
                logger.error(f"This should never happen after the adjustment logic above!")
                # Emergency fix
                emergency_elo = old_best_elo + 10.0
                logger.error(f"Emergency ELO adjustment: {current_elo:.1f} -> {emergency_elo:.1f}")
                self.elo_tracker.ratings[f"iter_{self.iteration}"] = emergency_elo
        
        self.best_model_iteration = self.iteration
        model_path = self.best_model_dir / f"model_iter_{self.iteration}.pt"
        
        # Save model with metadata if available
        if hasattr(self.model, 'metadata') and self.model.metadata:
            # Update metadata before saving
            self.model.metadata.training_steps = self.iteration
            if hasattr(self, 'elo_tracker') and self.elo_tracker:
                current_elo = self.elo_tracker.get_rating(f"iter_{self.iteration}")
                if current_elo is not None:
                    self.model.metadata.elo_rating = current_elo
            
            # Save with metadata
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'metadata': self.model.metadata.to_dict()
            }
            torch.save(checkpoint, model_path)
            
            # Also save as 'best_model.pt' for easy access
            best_path = self.best_model_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
        else:
            # Fallback to saving just state dict
            torch.save(self.model.state_dict(), model_path)
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
        
        # Create symlink to latest checkpoint for easy resuming (before cleanup)
        latest_link = self.checkpoint_dir / "latest_checkpoint.pt"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(checkpoint_path.name)
        
        # Clean up old replay buffers to save disk space (after symlink creation)
        self._cleanup_old_replay_buffers()
        
        # Clean up old checkpoints to save disk space (after symlink creation)
        self._cleanup_old_checkpoints()
        
        # Also save a resume info file
        resume_info = {
            "iteration": self.iteration,
            "checkpoint_path": str(checkpoint_path),
            "buffer_path": str(buffer_path),
            "timestamp": datetime.now().isoformat()
        }
        with open(self.checkpoint_dir / "resume_info.json", "w") as f:
            json.dump(resume_info, f, indent=2)
        
        logger.info(f"Saved checkpoint at iteration {self.iteration}")
    
    def _cleanup_old_replay_buffers(self, keep_last_n: int = 40):
        """Clean up old replay buffers to save disk space
        
        Args:
            keep_last_n: Number of recent replay buffers to keep (default: 40)
        """
        try:
            # Find all replay buffer files
            replay_files = list(self.data_dir.glob("replay_buffer_iter_*.pkl"))
            
            if len(replay_files) <= keep_last_n:
                return  # Nothing to clean up
            
            # Sort by iteration number
            def get_iteration_number(filename):
                try:
                    return int(filename.stem.split('_')[-1])
                except (ValueError, IndexError):
                    return 0
            
            replay_files.sort(key=get_iteration_number)
            
            # Files to remove (keep only the last keep_last_n files)
            files_to_remove = replay_files[:-keep_last_n]
            
            if files_to_remove:
                total_size_freed = 0
                for file_path in files_to_remove:
                    if file_path.exists():
                        file_size = file_path.stat().st_size
                        total_size_freed += file_size
                        file_path.unlink()
                
                # Log cleanup info
                size_freed_gb = total_size_freed / (1024**3)
                logger.info(f"Cleaned up {len(files_to_remove)} old replay buffers, "
                           f"freed {size_freed_gb:.1f} GB of disk space")
                           
        except Exception as e:
            logger.warning(f"Failed to cleanup old replay buffers: {e}")
    
    def _cleanup_old_checkpoints(self, keep_last_n: int = 20):
        """Clean up old checkpoint files to save disk space
        
        Args:
            keep_last_n: Number of recent checkpoints to keep (default: 20)
        """
        try:
            # Find all checkpoint files
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_iter_*.pt"))
            
            if len(checkpoint_files) <= keep_last_n:
                return  # Nothing to clean up
            
            # Sort by iteration number
            def get_iteration_number(filename):
                try:
                    return int(filename.stem.split('_')[-1])
                except (ValueError, IndexError):
                    return 0
            
            checkpoint_files.sort(key=get_iteration_number)
            
            # Files to remove (keep only the last keep_last_n files)
            files_to_remove = checkpoint_files[:-keep_last_n]
            
            # Additional safety: never remove the current iteration's checkpoint
            current_checkpoint = self.checkpoint_dir / f"checkpoint_iter_{self.iteration}.pt"
            files_to_remove = [f for f in files_to_remove if f != current_checkpoint]
            
            if files_to_remove:
                total_size_freed = 0
                for file_path in files_to_remove:
                    if file_path.exists():
                        file_size = file_path.stat().st_size
                        total_size_freed += file_size
                        file_path.unlink()
                
                # Log cleanup info
                size_freed_mb = total_size_freed / (1024**2)
                logger.info(f"Cleaned up {len(files_to_remove)} old checkpoints, "
                           f"freed {size_freed_mb:.1f} MB of disk space")
                           
        except Exception as e:
            logger.warning(f"Failed to cleanup old checkpoints: {e}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        
        # Convert to absolute path if relative
        if not checkpoint_path.is_absolute():
            # Try to resolve relative to the project root (where train.py is)
            import os
            project_root = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent.parent
            checkpoint_path = project_root / checkpoint_path
        
        # Handle both direct checkpoint path and checkpoint directory
        if checkpoint_path.is_dir():
            # Find latest checkpoint in directory
            checkpoints = sorted(checkpoint_path.glob("checkpoint_iter_*.pt"))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {checkpoint_path}")
            checkpoint_path = checkpoints[-1]
            logger.info(f"Found latest checkpoint: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.config.mcts.device, weights_only=False)
        
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
        
        if "elo_ratings" in checkpoint:
            # Create ELO tracker if needed
            if not self.elo_tracker:
                from mcts.neural_networks.arena_module import ELOTracker
                self.elo_tracker = ELOTracker(
                    k_factor=self.config.arena.elo_k_factor,
                    initial_rating=self.config.arena.elo_initial_rating
                )
            self.elo_tracker.ratings = checkpoint["elo_ratings"]
            logger.info(f"Loaded ELO ratings for {len(checkpoint['elo_ratings'])} players")
        
        # Try to load replay buffer
        buffer_path = self.data_dir / f"replay_buffer_iter_{self.iteration}.pkl"
        if buffer_path.exists():
            self.replay_buffer.load(str(buffer_path))
            logger.info(f"Loaded replay buffer with {len(self.replay_buffer)} examples")
        else:
            # Try to find most recent replay buffer
            buffer_files = sorted(self.data_dir.glob("replay_buffer_iter_*.pkl"))
            if buffer_files:
                latest_buffer = buffer_files[-1]
                self.replay_buffer.load(str(latest_buffer))
                logger.info(f"Loaded most recent replay buffer from {latest_buffer.name} with {len(self.replay_buffer)} examples")
            else:
                logger.warning("No replay buffer found, starting with empty buffer")
    
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
            checkpoint = torch.load(model_file, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
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
    
    def _precompile_cuda_kernels(self):
        """Pre-compile CUDA kernels to avoid JIT overhead during training"""
        try:
            from mcts.gpu.unified_kernels import get_unified_kernels
            from mcts.core.mcts import MCTS, MCTSConfig
            from mcts.neural_networks.mock_evaluator import MockEvaluator
            import torch
            
            # Force kernel loading
            logger.info("  📦 Loading unified kernels...")
            kernels = get_unified_kernels(torch.device(self.config.mcts.device))
            
            # Create minimal MCTS instance to trigger kernel compilation
            logger.info("  ⚙️  Initializing MCTS system...")
            mcts_config = MCTSConfig(
                num_simulations=10,  # Minimal simulations
                wave_size=100,       # Small wave size
                device=self.config.mcts.device,
                enable_quantum=False  # Keep it simple
            )
            
            evaluator = MockEvaluator()
            mcts = MCTS(mcts_config, evaluator)
            
            # Run a minimal search to trigger any lazy compilation
            logger.info("  🔥 Warming up kernels...")
            import alphazero_py
            game_state = alphazero_py.GomokuState()
            mcts.search(game_state, 5)  # Very small search to trigger compilation
            
            logger.debug("CUDA kernels pre-compiled successfully!")
            
        except Exception as e:
            logger.warning(f"Kernel pre-compilation failed: {e}")
            logger.info("  📝 Continuing with PyTorch fallback...")


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