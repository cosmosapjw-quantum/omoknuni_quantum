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
import time

# Import only essential components at module level
from mcts.utils.config_system import (
    AlphaZeroConfig, create_default_config,
    merge_configs
)
from mcts.neural_networks.replay_buffer import ReplayBuffer, GameExample

# Configure logging
logger = logging.getLogger(__name__)


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
        
        # Arena for evaluation with ELO tracking
        self.arena = self._create_arena()
        
        # Initialize metrics recorder
        from mcts.utils.training_metrics import TrainingMetricsRecorder
        metrics_dir = self.experiment_dir / "metrics"
        self.metrics_recorder = TrainingMetricsRecorder(
            save_dir=metrics_dir,
            window_size=100,
            auto_save_interval=config.log.checkpoint_frequency
        )
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
        # Try to find the project root by looking for a marker file
        current_path = Path(__file__).resolve()
        parent_dir = current_path.parent.parent.parent.parent  # Default: Go up to omoknuni_quantum
        
        # If we can't create directories there (e.g., in tests), use a temp directory
        test_marker = parent_dir / "experiments"
        try:
            test_marker.mkdir(exist_ok=True, parents=True)
            self.experiment_dir = parent_dir / "experiments" / self.config.experiment_name
        except (PermissionError, OSError):
            # Fall back to temp directory for tests
            import tempfile
            self.experiment_dir = Path(tempfile.gettempdir()) / "omoknuni_experiments" / self.config.experiment_name
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.best_model_dir = self.experiment_dir / "best_models"
        self.arena_log_dir = self.experiment_dir / self.config.arena.arena_log_dir
        self.data_dir = self.experiment_dir / self.config.training.data_dir
        
        for dir_path in [self.checkpoint_dir, self.best_model_dir, 
                         self.arena_log_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Configure logging with both console and file output"""
        import logging.handlers
        from datetime import datetime
        
        level = getattr(logging, self.config.log_level)
        
        # Create logs directory
        log_dir = self.experiment_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Create timestamp for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{timestamp}.log"
        
        # Clear any existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # Set up formatter
        if level == logging.DEBUG:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        # Console handler (for terminal output)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler with rotation (captures everything to file)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, 
            maxBytes=50*1024*1024,  # 50MB per file
            backupCount=5,          # Keep 5 backup files
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # Always capture all levels to file
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Set root logger level
        root_logger.setLevel(min(level, logging.DEBUG))
        
        # Suppress verbose library logs for console but keep in file
        if level != logging.DEBUG:
            logging.getLogger('torch').setLevel(logging.WARNING)
            logging.getLogger('matplotlib').setLevel(logging.WARNING)
        
        # Log the setup
        logger.info(f"Logging configured - Console: {logging.getLevelName(level)}, File: {log_file}")
        logger.info(f"Experiment directory: {self.experiment_dir}")
        
        # Store log file path for reference
        self.log_file = log_file
        
        # Set up tqdm to use logging for progress bars
        import tqdm
        tqdm.tqdm.write = lambda s, file=None, end="\n", nolock=False: logger.info(s)
        
        # Set up stdout/stderr capture for any print statements
        self._setup_stdout_capture(log_file)
    
    def _setup_stdout_capture(self, log_file):
        """Set up capturing of stdout/stderr to log file"""
        import sys
        
        class TeeStream:
            """Stream that writes to both original stream and log file"""
            def __init__(self, original_stream, log_file):
                self.original_stream = original_stream
                self.log_file = log_file
                
            def write(self, data):
                # Write to original stream (terminal)
                self.original_stream.write(data)
                self.original_stream.flush()
                
                # Write to log file
                try:
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write(data)
                        f.flush()
                except Exception:
                    pass  # Don't break if log writing fails
                    
            def flush(self):
                self.original_stream.flush()
                
            def __getattr__(self, name):
                return getattr(self.original_stream, name)
        
        # Store original streams
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        
        # Replace with tee streams
        sys.stdout = TeeStream(sys.stdout, log_file)
        sys.stderr = TeeStream(sys.stderr, log_file)
        
    def _restore_stdout_capture(self):
        """Restore original stdout/stderr streams"""
        import sys
        if hasattr(self, '_original_stdout'):
            sys.stdout = self._original_stdout
        if hasattr(self, '_original_stderr'):
            sys.stderr = self._original_stderr
    
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
        from mcts.neural_networks.arena_manager import ArenaManager, ArenaConfig
        
        arena_config = ArenaConfig(
            num_games=self.config.arena.num_games,
            win_threshold=self.config.arena.win_threshold,
            mcts_simulations=self.config.arena.mcts_simulations,
            c_puct=self.config.arena.c_puct,
            temperature=self.config.arena.temperature,
            temperature_threshold=self.config.mcts.temperature_threshold,  # Use configurable threshold
            timeout_seconds=int(self.config.arena.time_limit_seconds) if self.config.arena.time_limit_seconds else 300,
            device=self.config.mcts.device,
            save_game_records=self.config.arena.save_game_records
        )
        
        return ArenaManager(self.config, arena_config, self.game_interface)
    
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
        
        # Create progress bar that shows correct progress when resuming
        pbar = trange(target_iteration, desc="Training iterations", 
                     initial=start_iteration, unit="iter", position=0, leave=True)
        
        # Warm up CUDA kernels (detection only, compilation handled by setup.py)
        if self.config.mcts.device == 'cuda' and start_iteration == 0:
            logger.info("🔧 Warming up CUDA kernels...")
            self._warmup_cuda_kernels()
        
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
            
            # Print metrics summary every 5 iterations or when model is accepted
            if self.iteration % 5 == 0 or accepted:
                tqdm.write("")  # Empty line
                self.metrics_recorder.print_summary(detailed=(self.iteration % 10 == 0))
            
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
        
        # Save final metrics
        self.metrics_recorder.save()
        
        # Generate visualization if matplotlib is available
        try:
            from mcts.utils.training_metrics import MetricsVisualizer
            visualizer = MetricsVisualizer(self.metrics_recorder)
            plot_path = self.experiment_dir / "training_curves.png"
            visualizer.plot_training_curves(save_path=plot_path)
            logger.info(f"Training curves saved to {plot_path}")
        except ImportError:
            logger.info("Matplotlib not available, skipping visualization")
        
        # Print final metrics summary
        logger.info("\nFinal Training Metrics Summary:")
        self.metrics_recorder.print_summary(detailed=True)
        
        # Run final tournament if requested (disabled by default)
        # self.run_final_tournament()
    
    def generate_self_play_data(self) -> List[GameExample]:
        """Generate self-play training data using SelfPlayManager"""
        from mcts.core.evaluator import AlphaZeroEvaluator
        from mcts.neural_networks.self_play_manager import SelfPlayManager
        
        # Create evaluator for self-play
        evaluator = AlphaZeroEvaluator(
            model=self.model,
            device=self.config.mcts.device
        )
        
        # Configure evaluator to return torch tensors
        evaluator._return_torch_tensors = True
        
        # Create game configuration
        game_config = {
            'game_type': self.config.game.game_type,
            'board_size': self.config.game.board_size,
            'input_representation': getattr(self.config.network, 'input_representation', 'enhanced')
        }
        
        # Use SelfPlayManager with game configuration
        self_play_manager = SelfPlayManager(
            config=self.config,
            game_class=None,  # We'll use game_config instead
            evaluator=evaluator,
            game_config=game_config
        )
        
        # Generate self-play data (will use GPU service for multi-worker)
        examples = self_play_manager.generate_self_play_data()
        
        # Collect metrics
        self.game_metrics = self_play_manager.collect_game_metrics(examples)
        
        # Log game quality metrics
        self._log_game_quality_metrics()
        
        return examples
    
    
    def _log_game_quality_metrics(self):
        """Log comprehensive game quality metrics"""
        if not self.game_metrics or self.game_metrics.get('total_games', 0) == 0:
            logger.warning("No game metrics collected!")
            return
            
        import numpy as np
        import time
        
        # Use the metrics from collect_game_metrics
        total_games = self.game_metrics.get('total_games', 0)
        avg_length = self.game_metrics.get('avg_game_length', 0)
        min_length = self.game_metrics.get('min_game_length', 0)
        max_length = self.game_metrics.get('max_game_length', 0)
        
        # Calculate std_length (approximation since we don't have the raw data)
        std_length = (max_length - min_length) / 4.0 if total_games > 1 else 0
        
        # Get win rates
        p1_win_rate = self.game_metrics.get('win_rate_p1', 0)
        p2_win_rate = self.game_metrics.get('win_rate_p2', 0)
        draw_rate = self.game_metrics.get('draw_rate', 0)
        
        # Get the new metrics from collect_game_metrics
        resignation_count = self.game_metrics.get('resignation_count', 0)
        avg_entropy = self.game_metrics.get('avg_entropy', 0)
        value_acc = self.game_metrics.get('value_accuracy', 0)
        illegal_move_attempts = self.game_metrics.get('illegal_move_attempts', 0)
        
        # Entropy trend would require historical data, so keep as unknown for now
        entropy_trend = 'unknown'
        
        logger.info("="*80)
        logger.info("SELF-PLAY GAME QUALITY METRICS")
        logger.info("="*80)
        logger.info(f"Total games played: {total_games}")
        logger.info(f"Game length: avg={avg_length:.1f}, std={std_length:.1f}, min={min_length}, max={max_length}")
        logger.info(f"Outcomes: P1 wins={p1_win_rate:.1%}, P2 wins={p2_win_rate:.1%}, Draws={draw_rate:.1%}")
        logger.info(f"Resignations: {resignation_count} ({resignation_count/max(total_games,1):.1%})")
        logger.info(f"Average policy entropy: {avg_entropy:.3f} (trend: {entropy_trend})")
        logger.info(f"Value prediction accuracy: {value_acc:.1%}")
        if illegal_move_attempts > 0:
            logger.error(f"❌ Illegal moves attempted: {illegal_move_attempts}")
        else:
            logger.info(f"Illegal move attempts: {illegal_move_attempts}")
        
        # Check for potential issues
        if avg_length < 20:
            logger.warning("⚠️  Games are ending too quickly! Check game logic or increase exploration.")
        if p1_win_rate > 0.65 or p2_win_rate > 0.65:
            logger.warning("⚠️  High win rate imbalance detected! Check for first-player advantage.")
        if avg_entropy < 0.5:
            logger.warning("⚠️  Low policy entropy! Model may be overly deterministic.")
        
        logger.info("="*80)
        
        # Record self-play metrics
        self.metrics_recorder.record_self_play_metrics(
            iteration=self.iteration,
            games_per_second=total_games / 1.0,  # Default to 1 second if we don't have timing
            avg_game_length=avg_length,
            policy_entropy=avg_entropy,
            mcts_value_accuracy=value_acc,
            p1_win_rate=p1_win_rate,
            p2_win_rate=p2_win_rate,
            draw_rate=draw_rate,
            resignation_rate=resignation_count / max(total_games, 1)
        )
    
    
    
    def _augment_training_data(self, examples: List[GameExample]) -> List[GameExample]:
        """Apply data augmentation to training examples using board symmetries"""
        augmented_examples = []
        
        for example in examples:
            # Get board and policy from the example
            # State is already a numpy array
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
        """Train the neural network using TrainingManager"""
        from mcts.neural_networks.training_manager import TrainingManager
        
        if len(self.replay_buffer) < self.config.training.batch_size:
            return {"loss": 0, "policy_loss": 0, "value_loss": 0}
        
        # Create training manager
        training_manager = TrainingManager(
            config=self.config,
            model=self.model
        )
        
        # Use the same optimizer and scheduler as the pipeline
        training_manager.optimizer = self.optimizer
        training_manager.scheduler = self.scheduler
        training_manager.scaler = self.scaler
        
        # Train the model
        results = training_manager.train(self.replay_buffer)
        
        # Record training metrics
        self.metrics_recorder.record_training_step(
            iteration=self.iteration,
            epoch=self.config.training.num_epochs - 1,
            policy_loss=results['final_policy_loss'],
            value_loss=results['final_value_loss'],
            total_loss=results['final_loss'],
            learning_rate=self.optimizer.param_groups[0]['lr']
        )
        
        # Return stats in expected format
        stats = {
            "loss": results['final_loss'],
            "policy_loss": results['final_policy_loss'],
            "value_loss": results['final_value_loss'],
            "lr": self.optimizer.param_groups[0]['lr']
        }
            
        return stats
    
    def evaluate_model_with_elo(self) -> bool:
        """Evaluate model using ArenaManager with ELO tracking"""
        from mcts.core.evaluator import RandomEvaluator, AlphaZeroEvaluator, EvaluatorConfig
        import gc
        
        # Initialize ELO tracker in arena if needed
        if not self.arena.elo_system:
            logger.error("Arena ELO system not initialized!")
            return False
            
        # Create evaluators
        eval_config = EvaluatorConfig(device=self.config.mcts.device)
        random_evaluator = RandomEvaluator(self._get_action_size(), eval_config)
        current_evaluator = AlphaZeroEvaluator(
            model=self.model,
            device=self.config.mcts.device
        )
        
        # Clear GPU cache before arena battles
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Setup ELO inheritance for new model
        current_key = f"iter_{self.iteration}"
        if self.iteration > 1:
            # Inherit ELO from previous iteration
            self.arena.inherit_elo(current_key, f"iter_{self.iteration - 1}")
        else:
            # First model starts at anchor rating
            self.arena.elo_system.ratings[current_key] = self.arena.elo_system.anchor_rating
        
        # Match 1: Current vs Random
        current_model_elo = self.arena.elo_system.get_rating(current_key)
        should_play_random = self.arena.elo_system.should_play_vs_random(
            self.iteration, current_model_elo
        )
        
        win_rate_vs_random = 0.0
        if should_play_random or self.iteration == 1:
            tqdm.write("      Current vs Random...")
            results_vs_random = self.arena.evaluate_models(
                current_evaluator, random_evaluator,
                current_key, "random"
            )
            win_rate_vs_random = results_vs_random['win_rate']
            tqdm.write(f"      Result: {results_vs_random['model1_wins']}W-{results_vs_random['draws']}D-{results_vs_random['model2_wins']}L ({win_rate_vs_random:.1%})")
        
        # Match 2: Current vs Previous Model (for ELO calibration)
        if self.iteration > 1 and getattr(self.config.arena, 'enable_current_vs_previous', True):
            previous_model_path = self.checkpoint_dir / f"checkpoint_iter_{self.iteration - 1}.pt"
            results_vs_previous = self.arena.evaluate_with_previous(
                current_evaluator, current_key, previous_model_path, self.iteration
            )
            if results_vs_previous:
                tqdm.write(f"      Current vs Previous: {results_vs_previous['model1_wins']}W-{results_vs_previous['draws']}D-{results_vs_previous['model2_wins']}L ({results_vs_previous['win_rate']:.1%})")
        
        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Match 2: Current vs Best (if we have a best model)
        win_rate_vs_best = None
        accepted = False
        
        if self.best_model_iteration:
            # Evaluate against best model
            tqdm.write("      Current vs Best...")
            best_model_path = self.best_model_dir / f"model_iter_{self.best_model_iteration}.pt"
            best_model = self._create_model()
            
            # Load best model
            checkpoint = torch.load(best_model_path, map_location=self.config.mcts.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                best_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                best_model.load_state_dict(checkpoint)
            best_model.eval()
            
            # Create evaluator for best model
            best_evaluator = AlphaZeroEvaluator(
                model=best_model,
                device=self.config.mcts.device
            )
            
            # Run arena match
            results_vs_best = self.arena.evaluate_models(
                current_evaluator, best_evaluator,
                current_key, f"iter_{self.best_model_iteration}"
            )
            win_rate_vs_best = results_vs_best['win_rate']
            tqdm.write(f"      Result: {results_vs_best['model1_wins']}W-{results_vs_best['draws']}D-{results_vs_best['model2_wins']}L ({win_rate_vs_best:.1%})")
            
            # Store win rate for ELO adjustment if model is accepted
            self._last_win_rate_vs_best = win_rate_vs_best
            
            # Check acceptance
            accepted = win_rate_vs_best >= self.config.arena.win_threshold
            
            # Clean up
            del best_model, best_evaluator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            # First model - check against random baseline with logarithmic scheduling
            dynamic_threshold = self.arena.get_dynamic_win_rate_threshold(self.iteration)
            accepted = win_rate_vs_random >= dynamic_threshold
            tqdm.write(f"      Dynamic threshold: {dynamic_threshold:.1%} (iteration {self.iteration})")
        
        # Get final ELO ratings
        current_elo = self.arena.elo_system.get_rating(current_key)
        tqdm.write(f"\n      ELO Rating: {current_elo:.1f}")
        
        # Save as best model if accepted
        if accepted:
            self._save_best_model()
            tqdm.write(f"\n      ✓ Model accepted as new best!")
        else:
            tqdm.write(f"\n      ✗ Model rejected")
        
        # Record evaluation metrics
        self.metrics_recorder.record_evaluation(
            iteration=self.iteration,
            win_rate=win_rate_vs_best if win_rate_vs_best else win_rate_vs_random,
            elo_rating=current_elo,
            elo_change=0,  # Simplified
            vs_random_win_rate=win_rate_vs_random,
            vs_best_win_rate=win_rate_vs_best,
            model_accepted=accepted
        )
        
        return accepted
    
    
    def _save_best_model(self):
        """Save current model as best"""
        # Store win rate vs best for ELO adjustment
        win_rate_vs_best = getattr(self, '_last_win_rate_vs_best', None)
        
        # Adjust ELO if needed to ensure monotonic improvement
        if self.best_model_iteration and win_rate_vs_best is not None:
            current_key = f"iter_{self.iteration}"
            best_key = f"iter_{self.best_model_iteration}"
            self.arena.adjust_elo_for_new_best(current_key, best_key, win_rate_vs_best)
        
        self.best_model_iteration = self.iteration
        model_path = self.best_model_dir / f"model_iter_{self.iteration}.pt"
        
        # Save model state dict
        torch.save(self.model.state_dict(), model_path)
        
        # Also save as 'best_model.pt' for easy access
        best_path = self.best_model_dir / "best_model.pt"
        torch.save(self.model.state_dict(), best_path)
        
        logger.info(f"Saved best model: iteration {self.iteration}")
    
    def _save_arena_results(self, results: Dict):
        """Save arena evaluation results"""
        results_file = self.arena_log_dir / "arena_results.json"
        
        # Convert numpy/torch types to Python native types
        def convert_to_native(obj):
            """Recursively convert numpy/torch types to native Python types"""
            import numpy as np
            
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            elif isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
                return float(obj)
            elif hasattr(obj, 'item'):  # Handles torch tensors and numpy scalars
                return obj.item()
            else:
                return obj
        
        # Convert results to native types
        results = convert_to_native(results)
        
        # Load existing results
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    all_results = json.load(f)
            except json.JSONDecodeError as e:
                logger.warning(f"Corrupted arena results file, creating backup and starting fresh: {e}")
                # Create backup of corrupted file
                backup_file = results_file.with_suffix('.json.corrupted')
                import shutil
                shutil.copy(results_file, backup_file)
                all_results = []
        else:
            all_results = []
        
        # Add timestamp
        results["timestamp"] = datetime.now().isoformat()
        
        # Track best model ELO progression
        if all_results and results.get("accepted", False):
            # This is a new best model - verify it has higher ELO than the previous best
            current_model_elo = results.get("elo_ratings", {}).get("current", 0)
            
            # Find the most recent accepted model (previous best)
            prev_best_elo = None
            prev_best_iter = None
            for prev_result in reversed(all_results):
                if prev_result.get("accepted", False):
                    # Get the ELO of the previous best model at the time it was accepted
                    prev_best_elo = prev_result.get("elo_ratings", {}).get("current", 0)
                    prev_best_iter = prev_result.get("iteration", "unknown")
                    break
            
            if prev_best_elo is not None:
                elo_gain = current_model_elo - prev_best_elo
                if elo_gain <= 0:
                    # This should never happen with our enforcement logic
                    logger.error(f"CRITICAL: New best model (iter {results.get('iteration', 'unknown')}, ELO {current_model_elo:.1f}) "
                                f"has lower or equal ELO than previous best model (iter {prev_best_iter}, ELO {prev_best_elo:.1f})")
                    logger.error("This violates the fundamental requirement that new best models must have higher ELO.")
                else:
                    logger.info(f"Best model ELO progression confirmed: {prev_best_elo:.1f} -> {current_model_elo:.1f} (+{elo_gain:.1f})")
        
        all_results.append(results)
        
        # Save updated results
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    def save_checkpoint(self):
        """Save training checkpoint using CheckpointManager"""
        from mcts.neural_networks.checkpoint_manager import CheckpointManager
        
        # Create checkpoint manager
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            config=self.config
        )
        
        # Prepare metadata - include scheduler and scaler state in metadata
        metadata = {
            "best_model_iteration": self.best_model_iteration,
            "elo_ratings": self.arena.elo_system.ratings if self.arena.elo_system else {},
            "config": self.config,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None
        }
        
        # Save checkpoint
        checkpoint_path = checkpoint_manager.save_checkpoint(
            iteration=self.iteration,
            model=self.model,
            optimizer=self.optimizer,
            replay_buffer=self.replay_buffer,
            metadata=metadata
        )
        
        logger.info(f"Saved checkpoint at iteration {self.iteration}")
    
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint using CheckpointManager"""
        from mcts.neural_networks.checkpoint_manager import CheckpointManager
        
        checkpoint_path = Path(checkpoint_path)
        
        # Convert to absolute path if relative
        if not checkpoint_path.is_absolute():
            # Try to resolve relative to the project root (where train.py is)
            import os
            project_root = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent.parent
            checkpoint_path = project_root / checkpoint_path
        
        # Create checkpoint manager
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            config=self.config
        )
        
        # Load checkpoint
        checkpoint_data = checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            replay_buffer=self.replay_buffer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.config.mcts.device
        )
        
        # Update state from checkpoint
        self.iteration = checkpoint_data["iteration"]
        self.best_model_iteration = checkpoint_data["metadata"]["best_model_iteration"]
        
        # Load ELO ratings if available
        if "elo_ratings" in checkpoint_data["metadata"]:
            self.arena.elo_system.ratings = checkpoint_data["metadata"]["elo_ratings"]
            logger.info(f"Loaded ELO ratings for {len(self.arena.elo_system.ratings)} players")
    
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
        with tqdm(total=len(model_pairs), desc="Tournament matches", leave=False) as pbar:
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
    
    def _warmup_cuda_kernels(self):
        """Warm up CUDA kernels and pre-compile TensorRT if enabled"""
        try:
            from mcts.gpu.mcts_gpu_accelerator import get_mcts_gpu_accelerator
            from mcts.core.mcts import MCTS, MCTSConfig
            from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
            import torch
            
            # Detect and load pre-compiled kernels
            logger.info("  📦 Loading unified kernels...")
            kernels = get_mcts_gpu_accelerator(torch.device(self.config.mcts.device))
            
            # Pre-compile TensorRT if enabled
            if self.config.mcts.use_tensorrt:
                self._precompile_tensorrt_engine()
            
            # Create minimal MCTS instance for kernel warmup
            logger.info("  ⚙️  Initializing MCTS system...")
            mcts_config = MCTSConfig(
                num_simulations=10,  # Minimal simulations
                wave_size=100,       # Small wave size
                device=self.config.mcts.device,
                enable_quantum=False  # Keep it simple
            )
            
            # Use actual model for warmup instead of mock  
            evaluator = ResNetEvaluator(
                model=self.model,
                device=self.config.mcts.device
            )
            mcts = MCTS(mcts_config, evaluator)
            
            # Run a minimal search for kernel warmup (not compilation)
            logger.info("  🔥 Warming up kernels...")
            game_state = self.game_interface.create_initial_state()
            mcts.search(game_state, 5)  # Very small search for warmup
            
            logger.debug("CUDA kernels warmed up successfully!")
            
        except Exception as e:
            logger.warning(f"Kernel warmup failed: {e}")
            logger.info("  📝 Continuing with PyTorch fallback...")
    
    def _precompile_tensorrt_engine(self):
        """Pre-compile TensorRT engine in main process before worker spawning"""
        logger.info("  🚀 Pre-compiling TensorRT engine...")
        
        try:
            from mcts.utils.tensorrt_manager import get_tensorrt_manager
            from mcts.neural_networks.tensorrt_converter import HAS_TENSORRT
            
            if not HAS_TENSORRT:
                logger.warning("  ⚠️  TensorRT not available, skipping compilation")
                return
            
            # Get TensorRT manager with custom cache directory if specified
            cache_dir = getattr(self.config.mcts, 'tensorrt_engine_cache_dir', None)
            manager = get_tensorrt_manager(cache_dir=cache_dir)
            
            # Get input shape based on game configuration
            initial_state = self.game_interface.create_initial_state()
            state_array = self.game_interface.state_to_numpy(initial_state)
            input_shape = state_array.shape  # Should be (C, H, W)
            
            # Determine batch sizes to optimize for
            max_batch_size = getattr(self.config.mcts, 'tensorrt_max_batch_size', 2048)
            batch_sizes = [1, 8, 32, 64, 128, 512, 1024, 2048]  # Common batch sizes
            if hasattr(self.config.training, 'batch_size'):
                batch_sizes.append(self.config.training.batch_size)
            if max_batch_size not in batch_sizes:
                batch_sizes.append(max_batch_size)
            batch_sizes = [bs for bs in batch_sizes if bs <= max_batch_size]
            
            logger.info(f"  📏 Input shape: {input_shape}")
            logger.info(f"  📊 Optimizing for batch sizes: {batch_sizes}")
            
            # Get workspace size from config (convert MB to bytes)
            workspace_size_mb = getattr(self.config.mcts, 'tensorrt_workspace_size', 2048)
            workspace_size = workspace_size_mb << 20  # Convert MB to bytes
            
            # Convert model to TensorRT
            start_time = time.time()
            trt_model = manager.get_or_convert_model(
                pytorch_model=self.model,
                input_shape=input_shape,
                batch_size=max_batch_size,
                fp16=self.config.mcts.tensorrt_fp16,
                workspace_size=workspace_size,
                worker_id=0  # Main process
            )
            
            if trt_model is not None:
                conversion_time = time.time() - start_time
                logger.info(f"  ✅ TensorRT engine compiled successfully in {conversion_time:.2f}s")
                
                # Store engine path for workers to use
                model_hash = manager.get_model_hash(self.model, input_shape)
                self.tensorrt_engine_path = manager.get_cached_engine_path(model_hash)
                logger.info(f"  💾 Engine cached at: {self.tensorrt_engine_path}")
                
                # Verify engine can be loaded
                if self.tensorrt_engine_path.exists():
                    logger.info("  ✅ Engine file verified")
                else:
                    logger.warning("  ⚠️  Engine file not found after compilation")
            else:
                logger.warning("  ⚠️  TensorRT compilation failed, workers will use PyTorch")
                
        except Exception as e:
            logger.error(f"  ❌ TensorRT pre-compilation error: {e}")
            logger.info("  📝 Workers will fall back to PyTorch")


# Old wrapper function removed - now using GPU service architecture from self_play_module
# The old play_self_play_game_wrapper function has been removed.
# Workers now use RemoteEvaluator to communicate with GPU service for neural network evaluation
# while using CPU for tree operations to avoid GPU memory exhaustion.


if __name__ == "__main__":
    import argparse
    import sys
    
    # Single-GPU mode - no multiprocessing needed
    
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
        # Single-GPU mode - no workers needed
        config.training.num_games_per_iteration = 100
        config.arena.enabled = True
        config.arena.evaluation_interval = 10
    
    # Create and run pipeline
    pipeline = UnifiedTrainingPipeline(config, resume_from=args.resume)
    
    try:
        pipeline.train(num_iterations=args.iterations)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted. Saving checkpoint...")
        pipeline.save_checkpoint()
        logger.info("Checkpoint saved.")
        sys.exit(0)