"""Self-play module for training data generation

This module handles parallel self-play game generation with progress tracking.
"""

import logging
import numpy as np
import time
import queue
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import torch

# Core MCTS components
from mcts.core.game_interface import GameInterface, GameType
from mcts.core.evaluator import AlphaZeroEvaluator
from mcts.core.mcts import MCTS, MCTSConfig
from mcts.utils.gpu_evaluator_service import GPUEvaluatorService
# Quantum imports removed
from .unified_training_pipeline import GameExample

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
        self.config = config
        self.game_type = GameType[config.game.game_type.upper()]
        input_representation = getattr(config.network, 'input_representation', 'basic')
        self.game_interface = GameInterface(
        self.game_type,
        board_size=config.game.board_size,
        input_representation=input_representation
        )
    
    def generate_games(self, model: torch.nn.Module, iteration: int,
                  num_games: Optional[int] = None,
                  num_workers: Optional[int] = None,
                  tensorrt_engine_path: Optional[str] = None) -> List[Any]:
        """Generate self-play games with progress tracking
        
        Args:
        model: Neural network model
        iteration: Current training iteration
        num_games: Number of games to generate (overrides config)
        num_workers: Number of parallel workers (overrides config)
        tensorrt_engine_path: Path to pre-compiled TensorRT engine
        
        Returns:
        List of game examples
        """
        num_games = num_games or self.config.training.num_games_per_iteration
        num_workers = num_workers or self.config.training.num_workers
        
        if num_workers <= 1:
            return self._sequential_self_play(model, iteration, num_games)
        else:
            return self._parallel_self_play(model, iteration, num_games, num_workers, tensorrt_engine_path)
    
    def _sequential_self_play(self, model: torch.nn.Module, iteration: int,
                         num_games: int) -> List[Any]:
        """Generate games sequentially with progress bar"""
        examples = []
        
        # Create evaluator
        evaluator = AlphaZeroEvaluator(
        model=model,
        device=self.config.mcts.device
        )
        
        # Progress bar setup with proper positioning
        disable_progress = logger.level > logging.INFO
        
        with tqdm(total=num_games, desc="Self-play games", unit="game",
             disable=disable_progress, position=2, leave=False) as pbar:
            for game_idx in range(num_games):
                game_examples = self._play_single_game(
                    model, evaluator, game_idx, iteration
                )
                examples.extend(game_examples)
                pbar.update(1)
        
        return examples
    
    def _parallel_self_play(self, model: torch.nn.Module, iteration: int,
                       num_games: int, num_workers: int,
                       tensorrt_engine_path: Optional[str] = None) -> List[Any]:
        """Generate games in parallel with GPU evaluation service"""
        import multiprocessing as mp
        
        # Get resource allocation from config (already adjusted for hardware)
        allocation = getattr(self.config, '_resource_allocation', None)
        if allocation is None:
            # Fallback: calculate allocation if not already done
            hardware = self.config.detect_hardware()
            allocation = self.config.calculate_resource_allocation(hardware, num_workers)
            self.config._resource_allocation = allocation
        
        logger.debug(f"Starting parallel self-play with {num_workers} workers")
        
        # Set up multiprocessing start method for CUDA safety
        try:
            mp.set_start_method('spawn', force=True)
            logger.debug("Set multiprocessing start method to 'spawn' for CUDA safety")
        except RuntimeError as e:
            logger.debug(f"Multiprocessing start method already set: {e}")
        
        examples = []
        
        # Create GPU evaluator service in main process with hardware-optimized settings
        gpu_service = GPUEvaluatorService(
            model=model,
            device=self.config.mcts.device,
            workload_type="throughput",  # Optimize for maximum throughput
            use_tensorrt=getattr(self.config.mcts, 'use_tensorrt', True),  # Use config or default to True
            tensorrt_fp16=getattr(self.config.mcts, 'tensorrt_fp16', True),  # Use config or default to True
            tensorrt_engine_path=tensorrt_engine_path  # Pass pre-compiled engine path
        )
        
        # Start the service
        gpu_service.start()
        logger.debug(f"[SELF-PLAY] GPU evaluation service started on device: {self.config.mcts.device}")
        
        # Create coordinated batch response queue for worker_id=-1 (used by batch coordinator)
        coordinated_response_queue = gpu_service.create_worker_queue(-1)
        logger.debug(f"[SELF-PLAY] Created coordinated batch response queue for cross-worker batching")
        
        try:
            # Get the request queue
            request_queue = gpu_service.get_request_queue()
            
            # Get action space size
            initial_state = self.game_interface.create_initial_state()
            action_size = self.game_interface.get_action_space_size(initial_state)
            
            logger.debug(f"[SELF-PLAY] Starting {num_workers} worker processes")
            
            # Use Process directly instead of ProcessPoolExecutor
            processes = []
            result_queues = []
            
            # Use hardware-aware concurrent worker limit
            max_concurrent = allocation['max_concurrent_workers']
            games_per_batch = max_concurrent
            
            logger.debug(f"[SELF-PLAY] Processing games in batches of {games_per_batch}")
            
            # Progress bar setup
            disable_progress = logger.level > logging.INFO
            
            with tqdm(total=num_games, desc="Self-play games", unit="game",
                     disable=disable_progress, position=2, leave=False) as pbar:
                
                for batch_start in range(0, num_games, games_per_batch):
                    batch_end = min(batch_start + games_per_batch, num_games)
                    batch_processes = []
                    batch_queues = []
                    
                    # Start processes for this batch
                    for game_idx in range(batch_start, batch_end):
                        # Create result queue for this game
                        result_queue = mp.Queue()
                        result_queues.append(result_queue)
                        batch_queues.append(result_queue)
                        
                        # Create worker-specific response queue
                        response_queue = gpu_service.create_worker_queue(game_idx)
                        
                        # Create process with hardware-aware resource allocation
                        p = mp.Process(
                            target=_play_game_worker_wrapper,
                            args=(self.config, request_queue, response_queue,
                                  action_size, game_idx, iteration, result_queue,
                                  allocation)
                        )
                        p.start()
                        processes.append(p)
                        batch_processes.append(p)
                        logger.debug(f"[SELF-PLAY] Started process for game {game_idx}")
                
                    # Collect results from this batch as they complete
                    logger.debug(f"[SELF-PLAY] Collecting results from batch of {len(batch_processes)} games...")
                    
                    for idx, (p, q) in enumerate(zip(batch_processes, batch_queues)):
                        game_idx = batch_start + idx
                        try:
                            # First check if process is still alive
                            if p.is_alive():
                                # Wait for result with timeout (increased for complex searches)
                                game_examples = q.get(timeout=600)  # 10 minute timeout for complex MCTS with initialization
                            else:
                                # Process already finished, get result immediately
                                game_examples = q.get(timeout=10)
                            
                            if game_examples:
                                examples.extend(game_examples)
                                logger.debug(f"[SELF-PLAY] Collected {len(game_examples)} examples from game {game_idx}")
                            else:
                                logger.warning(f"[SELF-PLAY] Game {game_idx} returned empty examples")
                                
                        except queue.Empty:
                            logger.error(f"[SELF-PLAY] Timeout (600s) waiting for result from game {game_idx}")
                            if p.is_alive():
                                logger.warning(f"[SELF-PLAY] Terminating stuck process for game {game_idx}")
                                p.terminate()
                                # Add diagnostic info
                                logger.error(f"[SELF-PLAY] Process stuck - likely CPU/GPU sync issue or MCTS infinite loop")
                                
                        except Exception as e:
                            logger.error(f"[SELF-PLAY] Failed to get result from game {game_idx}: {e}")
                            import traceback
                            traceback.print_exc()
                            
                        finally:
                            # Ensure process is terminated
                            if p.is_alive():
                                p.terminate()
                            p.join(timeout=5)
                            if p.is_alive():  # Still alive after join
                                logger.error(f"[SELF-PLAY] Force killing stuck process {game_idx}")
                                p.kill()
                            pbar.update(1)
        
                        
        finally:
            # Stop the GPU service
            gpu_service.stop()
            logger.debug("[SELF-PLAY] GPU evaluation service stopped")
            # Progress bar closed automatically by context manager
        
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
            
            # Check resignation condition
            root_value = mcts.get_root_value()
            
            # Initialize resignation tracking if not exists
            if not hasattr(mcts, 'recent_values'):
                mcts.recent_values = []
            
            # Track recent values from current player's perspective
            mcts.recent_values.append(root_value)
            # Keep only the last N moves
            mcts.recent_values = mcts.recent_values[-self.config.training.resign_check_moves:]
            
            # FIXED: Improved resignation logic with adaptive thresholds and randomness
            if iteration >= self.config.training.resign_start_iteration:
                if len(mcts.recent_values) >= self.config.training.resign_check_moves:
                    # Calculate adaptive threshold that becomes more aggressive over time
                    decay_factor = self.config.training.resign_threshold_decay ** max(0, iteration - self.config.training.resign_start_iteration)
                    adaptive_threshold = self.config.training.resign_threshold * decay_factor
                    
                    # Add randomness to prevent uniform behavior (±resign_randomness)
                    import random
                    randomness = self.config.training.resign_randomness
                    random_threshold = adaptive_threshold + random.uniform(-randomness, randomness)
                    
                    # Minimal resignation logging
                    if move_num % 20 == 0 and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Resignation check at move {move_num}: threshold={random_threshold:.3f}")
                    
                    if all(v < random_threshold for v in mcts.recent_values):
                        # FIXED: Consistent value assignment for resignation
                        # When current player resigns, the other player wins
                        current_player = self.game_interface.get_current_player(state)
                        winner = 2 if current_player == 0 else 1  # Other player wins
                        
                        # ENHANCED: Comprehensive resignation logging and validation
                        game_length = len(examples)
                        self._log_game_completion(game_id, "resignation", winner, game_length, move_num, examples)
                        self._assign_values_consistently(examples, winner)
                        
                        logger.debug(f"Resigned with threshold {random_threshold:.3f} at iteration {iteration}")
                        break
            
            # Select move based on policy - MCTS should ensure only legal moves
            # Note: MCTS already applies temperature scaling to visit counts using
            # the AlphaZero formula: policy[a] ∝ visits[a]^(1/temperature)
            
            if mcts.config.temperature == 0:
                # Deterministic - select highest probability action
                action = np.argmax(policy)
            else:
                # Stochastic - sample from the temperature-scaled distribution
                if np.sum(policy) > 0:
                    normalized_policy = policy / np.sum(policy)
                    action = np.random.choice(len(policy), p=normalized_policy)
                else:
                    # Fallback to uniform random
                    action = np.random.choice(len(policy))
            
            # Store example with the full policy (before move selection)
            canonical_state = self.game_interface.get_canonical_form(state)
            examples.append(GameExample(
                state=canonical_state,
                policy=policy,
                value=0,
                game_id=game_id,
                move_number=move_num
            ))
            
            # Apply action to game interface state first to get new state
            try:
                state = self.game_interface.get_next_state(state, action)
                
                # CRITICAL FIX: Update MCTS tree root with new state to keep states synchronized
                # This ensures MCTS internal state matches game interface state for next search
                mcts.update_root(action, state)
            except ValueError as e:
                # CRITICAL: If action is illegal, MCTS and game interface are out of sync!
                logger.error(f"State synchronization error at move {move_num}")
                logger.error(f"MCTS selected illegal action: {action}")
                logger.error(f"This indicates MCTS internal state != game interface state")
                logger.error(f"Policy for this action: {policy[action]}")
                logger.error(f"Game terminal before move: {self.game_interface.is_terminal(state)}")
                raise
            
            # Check terminal
            if self.game_interface.is_terminal(state):
                # FIXED: Consistent value assignment for natural termination
                winner = self.game_interface.get_winner(state)
                
                # ENHANCED: Comprehensive game completion logging and validation
                game_length = len(examples)
                self._log_game_completion(game_id, "natural", winner, game_length, move_num, examples)
                self._assign_values_consistently(examples, winner)
                
                break
        
        return examples
    
    def _assign_values_consistently(self, examples: List, winner: int):
        """
        Assign values to training examples consistently from each player's perspective.
        
        This fixes the critical perspective inconsistency bug by ensuring that:
        1. Each example gets value from the perspective of the player who made that move
        2. Value is +1 if that player won, -1 if they lost, 0 for draw
        3. Logic is identical regardless of resignation vs natural termination
        
        Args:
        examples: List of training examples to update
        winner: Game winner (0=draw, 1=player1, 2=player2)
        """
        for i, example in enumerate(examples):
            # Determine which player made this move (0-indexed moves)
            player_who_made_move = (i % 2) + 1  # 1=Player1, 2=Player2
            
            # Assign value from that player's perspective
            if winner == 0:  # Draw
                example.value = 0.0
            elif player_who_made_move == winner:
                example.value = 1.0   # This player won
            else:
                example.value = -1.0  # This player lost
                
            # Minimal logging for value assignment
            if logger.isEnabledFor(logging.DEBUG) and i < 3:  # Only log first 3 moves
                logger.debug(f"  Move {i}: P{player_who_made_move} → value={example.value}")
    
    def _log_game_completion(self, game_id: str, completion_type: str, winner: int, 
                       game_length: int, final_move_num: int, examples: List):
        """
        Essential game completion logging and sanity checks.
        
        Performs critical validation to detect systematic bias issues while
        maintaining minimal logging overhead for production use.
        """
        # Basic completion logging
        outcome_desc = f"P{winner} wins" if winner > 0 else "Draw"
        logger.info(f"Game {game_id}: {outcome_desc} via {completion_type} ({game_length} moves)")
        
        # Critical sanity checks only
        sanity_issues = []
        
        # Check for uniform game length (systematic bug indicator)
        if game_length == 14:
            sanity_issues.append("Game ended at exactly 14 moves")
        
        # Check value assignment consistency (sample check)
        if examples and len(examples) >= 4:
            p1_sample = examples[0].value  # P1's first move value
            p2_sample = examples[1].value  # P2's first move value
            
            # Basic consistency check
            if completion_type == "resignation":
                if winner == 2 and p1_sample > 0:  # P1 resigned but has positive value
                    sanity_issues.append("Inconsistent resignation values")
                elif winner == 1 and p2_sample > 0:  # P2 resigned but has positive value
                    sanity_issues.append("Inconsistent resignation values")
        
        # Log any critical issues
        if sanity_issues:
            logger.warning(f"Sanity check issues in {game_id}: {', '.join(sanity_issues)}")
        
        # Minimal additional info for resignations
        if completion_type == "resignation":
            resigning_player = 1 if winner == 2 else 2
            logger.debug(f"P{resigning_player} resigned, P{winner} wins")
    
    def _create_quantum_config(self):
        """Placeholder for quantum config (disabled)"""
        return None

    def _create_mcts(self, evaluator: Any, is_sequential: bool = True):
        """Create MCTS instance with quantum features if enabled
        
        Args:
            evaluator: Neural network evaluator
            is_sequential: True for sequential mode, False for worker mode
        """
        # Determine if CUDA kernels should be enabled
        cuda_available = torch.cuda.is_available() and os.environ.get('DISABLE_CUDA_KERNELS', '0') != '1'
        
        # Use CUDA for sequential mode if available
        if is_sequential and cuda_available:
            device = 'cuda'
        else:
            device = 'cpu'
        
        # Create MCTS configuration using unified config system
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
            
            # Memory and optimization
            memory_pool_size_mb=max(512, self.config.mcts.memory_pool_size_mb // 4),
            max_tree_nodes=min(100000, self.config.mcts.max_tree_nodes // 2),  # Cap at 100k for stability
            use_mixed_precision=cuda_available and self.config.mcts.use_mixed_precision,
            use_cuda_graphs=cuda_available and self.config.mcts.use_cuda_graphs,
            use_tensor_cores=cuda_available and self.config.mcts.use_tensor_cores,
            
            # Game and device configuration  
            device=device,
            game_type=self.game_type,
            board_size=self.config.game.board_size,
        
        # Quantum features disabled
        enable_quantum=False,
        
        # Virtual loss for parallel exploration
        enable_virtual_loss=True,
        virtual_loss=self.config.mcts.virtual_loss,
        # Debug options
        enable_debug_logging=(self.config.log_level == "DEBUG"),
        profile_gpu_kernels=False  # Don't profile in main process MCTS
        )
        
        # Create classical MCTS
        mcts = MCTS(mcts_config, evaluator)
        
        return mcts


# This function has been removed - self-play now uses GPU evaluation service
# See _play_game_worker_with_gpu_service for the current implementation


def _play_game_worker_wrapper(config, request_queue, response_queue, action_size: int,
                         game_idx: int, iteration: int, result_queue,
                         allocation) -> None:
    """Wrapper that puts results in a queue instead of returning them"""
    try:
        results = _play_game_worker_with_gpu_service(
            config, request_queue, response_queue, action_size, game_idx, iteration,
            allocation
        )
        result_queue.put(results)
    except Exception as e:
        import sys
        print(f"[WORKER {game_idx}] Failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        result_queue.put([])  # Empty list on error


def _play_game_worker_with_gpu_service(config, request_queue, response_queue, action_size: int,
                                  game_idx: int, iteration: int,
                                  allocation) -> List[Any]:
    """Worker function for parallel self-play using GPU evaluation service"""
    # Phase 2.1: Fix CUDA multiprocessing warnings with comprehensive solution
    import os
    import sys
    from pathlib import Path
    
    # Set up logging first before any other imports
    import logging
    logging.basicConfig(level=getattr(logging, config.log_level))
    logger = logging.getLogger(__name__)
    
    # Add project path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    # CRITICAL: Only disable CUDA for multi-worker mode
    # Single worker mode can safely use CUDA kernels
    import os
    import sys
    
    # CUDA kernel control for workers
    # We want to use CUDA kernels for performance, but prevent GPU memory corruption
    if game_idx > 0:  # Multi-worker mode (worker 1, 2, 3...)
        # Don't hide GPUs completely - we need them for CUDA kernels
        # Just ensure workers don't try to initialize TensorRT or allocate large GPU memory
        os.environ['WORKER_INDEX'] = str(game_idx)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # All workers use same GPU
        # Don't disable CUDA kernels - we want to use them!
        # os.environ['DISABLE_CUDA_KERNELS'] = '1'  # REMOVED - allow CUDA kernels
    else:  # Single worker mode (worker 0) or sequential mode
        # Main worker/process
        os.environ['WORKER_INDEX'] = '0'
        # Allow full GPU access
        pass
    
    try:
        # Now safe to import torch and other modules
        import torch
        
        # CRITICAL: Prevent TensorRT initialization in workers
        # This is what causes memory corruption when multiple processes access GPU
        os.environ['DISABLE_TENSORRT_IN_WORKERS'] = '1'
        
        # Check CUDA availability based on worker mode
        if game_idx > 0:  # Multi-worker mode
            # Force disable CUDA in torch to prevent any GPU operations
            torch.cuda.is_available = lambda: False
            torch.cuda.device_count = lambda: 0
            torch.cuda.current_device = lambda: 0
            torch.cuda.get_device_name = lambda x=0: "CPU"
            torch.cuda.get_device_properties = lambda x=0: None
            worker_device = 'cpu'
            cuda_available = False
            logger.debug(f"[WORKER {game_idx}] Multi-worker mode - CUDA disabled, NN eval via GPU service")
        else:  # Single worker mode
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                worker_device = 'cuda'
                logger.debug(f"[WORKER {game_idx}] Single worker mode - CUDA enabled for MCTS kernels")
            else:
                worker_device = 'cpu'
                logger.debug(f"[WORKER {game_idx}] Single worker mode - CUDA not available, using CPU")
        
        from .unified_training_pipeline import GameExample
        from mcts.core.game_interface import GameInterface, GameType
        # Use optimized MCTS directly as requested
        from mcts.core.mcts import MCTS, MCTSConfig
        from mcts.gpu.gpu_game_states import GameType as GPUGameType
        # Quantum imports removed
        from mcts.utils.gpu_evaluator_service import RemoteEvaluator
        # Use optimized evaluator for performance improvements
        from mcts.utils.optimized_remote_evaluator import OptimizedRemoteEvaluator
        
        # Logger set up earlier
        
        logger.debug(f"[WORKER {game_idx}] Worker started - MCTS on CPU, NN evaluation via GPU service")
        
        # Create game interface
        game_type = GameType[config.game.game_type.upper()]
        input_representation = getattr(config.network, 'input_representation', 'basic')
        game_interface = GameInterface(game_type, config.game.board_size, input_representation=input_representation)
        
        # DEBUG: Comprehensive game interface validation in worker
        initial_state = game_interface.create_initial_state()
        action_space_size = game_interface.get_action_space_size(initial_state)
        legal_moves = game_interface.get_legal_moves(initial_state)
        
        logger.debug(f"[WORKER {game_idx} DEBUG] Game interface created:")
        logger.debug(f"[WORKER {game_idx} DEBUG]   Game type: {game_type}")
        logger.debug(f"[WORKER {game_idx} DEBUG]   Board size: {config.game.board_size}")
        logger.debug(f"[WORKER {game_idx} DEBUG]   Action space: {action_space_size}")
        logger.debug(f"[WORKER {game_idx} DEBUG]   Legal moves: {len(legal_moves)}")
        logger.debug(f"[WORKER {game_idx} DEBUG]   Legal range: [{min(legal_moves)}, {max(legal_moves)}]")
        logger.debug(f"[WORKER {game_idx} DEBUG]   Action 200 legal: {200 in legal_moves}")
        
        # Test action 200 specifically in worker
        try:
            test_state = game_interface.apply_move(initial_state, 200)
            logger.debug(f"[WORKER {game_idx} DEBUG] Action 200 test: SUCCESS")
        except Exception as e:
            logger.error(f"[WORKER {game_idx} DEBUG] Action 200 test: FAILED - {e}")
        
        # Create optimized remote evaluator for GPU service communication
        batch_timeout = getattr(config.mcts, 'gpu_batch_timeout', 0.1)  # 100ms timeout (reasonable for coordination)
        
        # Use optimized evaluator with intelligent coordination
        # Coordinator now auto-detects multiprocessing context and disables coordination safely
        remote_evaluator = OptimizedRemoteEvaluator(
            request_queue=request_queue,
            response_queue=response_queue, 
            action_size=action_size,
            worker_id=game_idx, 
            batch_timeout=batch_timeout,
            enable_coordination=True,  # ENABLED: coordinator handles multiprocessing detection
            max_coordination_batch_size=64
        )
        # Removed verbose logging - using optimized evaluator
        
        # Wrap evaluator to return torch tensors for optimized MCTS
        class TensorEvaluator:
            def __init__(self, evaluator, device):
                self.evaluator = evaluator
                self.device = device
                self._return_torch_tensors = True  # Signal that we return tensors
            
            def evaluate(self, state, legal_mask=None, temperature=1.0):
                policy, value = self.evaluator.evaluate(state, legal_mask, temperature)
                # Convert numpy to torch tensors
                policy_tensor = torch.from_numpy(policy).float().to(self.device)
                value_tensor = torch.tensor(value, dtype=torch.float32, device=self.device)
                return policy_tensor, value_tensor
            
            def evaluate_batch(self, states, legal_masks=None, temperature=1.0):
                # Convert torch tensors to numpy for the remote evaluator
                if isinstance(states, torch.Tensor):
                    states = states.cpu().numpy()
                if legal_masks is not None and isinstance(legal_masks, torch.Tensor):
                    legal_masks = legal_masks.cpu().numpy()
                
                # Call remote evaluator's batch method
                policies, values = self.evaluator.evaluate_batch(states, legal_masks, temperature)
                
                # Convert numpy to torch tensors
                policies_tensor = torch.from_numpy(policies).float().to(self.device)
                values_tensor = torch.from_numpy(values).float().to(self.device)
                return policies_tensor, values_tensor
        
        # Use the same device as MCTS for tensor operations
        # Neural network evaluation still happens in main process via GPU service
        evaluator = TensorEvaluator(remote_evaluator, worker_device)
        # Removed verbose logging - tensor evaluator created
        
        # Memory debugging removed - fix confirmed working
        
        # Create MCTS
        # CRITICAL FIX: Workers must use CPU for tree operations to avoid GPU memory exhaustion
        # Only neural network evaluation goes through the GPU service
        # The issue was that each worker was creating 600K nodes × 256 children × 4 bytes = 614MB per worker
        # With 8 workers, this consumed 4.9GB GPU memory just for tree storage
        
        # Game type conversion happens silently
        
        # FIXED: Each worker needs full tree nodes for independent MCTS searches
        # Workers don't share state pools - each runs independent tree searches
        num_workers = allocation.get('num_workers', 1)
        
        # Each worker gets the full configured tree size, not a fraction
        # This is critical because workers run independent MCTS searches
        nodes_per_worker = config.mcts.max_tree_nodes
        
        # Cap nodes based on device to avoid GPU state pool exhaustion
        if worker_device == 'cuda':
            # GPU has limited state pool capacity
            nodes_per_worker = min(nodes_per_worker, 100000)
        else:
            # CPU can handle more nodes
            nodes_per_worker = max(nodes_per_worker, 500000)
        
        # Worker device already set above based on CUDA availability
        # MCTS kernels can safely use GPU in worker processes
        
        # Clean logging removed - using CPU for tree operations
        
        mcts_config = MCTSConfig(
        num_simulations=config.mcts.num_simulations,
        c_puct=config.mcts.c_puct,
        temperature=1.0,  # Will be adjusted during play
        # CRITICAL FIX: Force CPU usage for tree operations in workers
        device=worker_device,
        game_type=GPUGameType[game_type.name],  # Convert to GPU game type enum
        # Use benchmarked optimal wave size
        min_wave_size=config.mcts.min_wave_size,
        max_wave_size=config.mcts.max_wave_size,
        # use_optimized_implementation=True,  # This parameter doesn't exist in MCTSConfig
        # Dynamic memory pool based on hardware allocation
        memory_pool_size_mb=min(allocation.get('memory_per_worker_mb', 512) // 2, 512),
        use_mixed_precision=cuda_available and config.mcts.use_mixed_precision,
        use_cuda_graphs=cuda_available and config.mcts.use_cuda_graphs,
        use_tensor_cores=cuda_available and config.mcts.use_tensor_cores,
        # FIXED: Ensure sufficient tree size to prevent state pool exhaustion
        max_tree_nodes=nodes_per_worker,
        dirichlet_epsilon=0.25,  # Add exploration noise to root
        dirichlet_alpha=0.3,  # Dirichlet noise parameter
        board_size=config.game.board_size,  # Add missing board_size parameter
        # Quantum features disabled
        enable_quantum=False,
        # Debug options - disable for workers
        enable_debug_logging=False,
        profile_gpu_kernels=False
        )
        
        # Create classical MCTS
        mcts = MCTS(mcts_config, evaluator)
        
        # Clean logging - only essential information
        logger.debug(f"[WORKER {game_idx}] MCTS configured: {nodes_per_worker} nodes, {config.mcts.num_simulations} simulations, device: {mcts_config.device}")
        
        # DEBUG: Final validation before starting game loop
        test_state = game_interface.create_initial_state()
        test_legal_moves = game_interface.get_legal_moves(test_state)
        logger.debug(f"[WORKER {game_idx} DEBUG] Pre-game validation:")
        logger.debug(f"[WORKER {game_idx} DEBUG]   Action 200 legal before game: {200 in test_legal_moves}")
        logger.debug(f"[WORKER {game_idx} DEBUG]   Game interface board size: {game_interface.board_size}")
        logger.debug(f"[WORKER {game_idx} DEBUG]   Config board size: {config.game.board_size}")
        
        # Verify state pool availability
        if hasattr(mcts, 'state_pool_free'):
            total_states = len(mcts.state_pool_free)
            free_states = mcts.state_pool_free.sum().item()
            logger.debug(f"[WORKER {game_idx}] State pool: {free_states}/{total_states} free states")
        
        # Initialize MCTS internals if needed
        if hasattr(mcts, 'initialize') and callable(mcts.initialize):
            try:
                mcts.initialize()
                logger.debug(f"[WORKER {game_idx}] MCTS initialized successfully")
            except Exception as e:
                logger.warning(f"[WORKER {game_idx}] Failed to initialize MCTS: {e}")
        
        # State pool verification removed - fix confirmed working
        
        # Optimize MCTS for hardware (enables GPU kernels, mixed precision, etc.)
        if hasattr(mcts, 'optimize_for_hardware'):
            try:
                mcts.optimize_for_hardware()
                logger.debug(f"[WORKER {game_idx}] MCTS optimized for hardware")
            except Exception as e:
                logger.warning(f"[WORKER {game_idx}] Failed to optimize MCTS: {e}")
        # Clean logging - MCTS configured successfully
        
        # CUDA kernels should be available for high-performance MCTS
        examples = []
        state = game_interface.create_initial_state()
        game_id = f"iter{iteration}_game{game_idx}"
        
        # DEBUG: Validate initial game state
        initial_legal_moves = game_interface.get_legal_moves(state)
        logger.debug(f"[WORKER {game_idx} DEBUG] Game started:")
        logger.debug(f"[WORKER {game_idx} DEBUG]   Initial legal moves: {len(initial_legal_moves)}")
        logger.debug(f"[WORKER {game_idx} DEBUG]   Action 200 legal at start: {200 in initial_legal_moves}")
        logger.debug(f"[WORKER {game_idx} DEBUG]   Initial state type: {type(state)}")
        
        start_time = time.time()
        for move_num in range(config.training.max_moves_per_game):
            # Minimal progress monitoring - only for debugging
            if move_num > 0 and move_num % 50 == 0:  # Log every 50 moves
                elapsed = time.time() - start_time
                logger.debug(f"[WORKER {game_idx}] Move {move_num}, elapsed: {elapsed:.1f}s")
            
            # AlphaZero-style temperature annealing
            # Use exploration temperature for first N moves, then switch to deterministic
            if move_num < config.mcts.temperature_threshold:
                # Exploration phase - use stochastic temperature
                mcts.config.temperature = 1.0
            else:
                # Exploitation phase - deterministic play
                mcts.config.temperature = 0.0
            
            # Search with MCTS
            try:
                search_start = time.time()
                policy = mcts.search(state, num_simulations=config.mcts.num_simulations)
                search_time = time.time() - search_start
                
                # Log timing for performance analysis
                if move_num <= 2 or move_num % 50 == 0:
                    logger.debug(f"Worker {game_idx} Move {move_num}: {config.mcts.num_simulations} sims "
                                f"in {search_time*1000:.1f}ms ({config.mcts.num_simulations/search_time:.1f} sims/s)")
                
            except Exception as e:
                logger.error(f"[WORKER {game_idx}] MCTS search failed at move {move_num}: {e}")
                raise
            
            # Convert to numpy if needed
            if isinstance(policy, torch.Tensor):
                policy = policy.cpu().numpy()
            
            # Check resignation condition
            root_value = mcts.get_root_value()
            
            # Initialize resignation tracking if not exists
            if not hasattr(mcts, 'recent_values'):
                mcts.recent_values = []
            
            # Track recent values from current player's perspective
            mcts.recent_values.append(root_value)
            # Keep only the last N moves
            mcts.recent_values = mcts.recent_values[-config.training.resign_check_moves:]
            
            # Check if we should resign (only after resign_start_iteration)
            if iteration >= config.training.resign_start_iteration:
                if len(mcts.recent_values) >= config.training.resign_check_moves:
                    # Log the resignation check for debugging
                    if move_num % 10 == 0:  # Log every 10 moves to avoid spam
                        logger.debug(f"Resignation check at move {move_num}: recent values {[f'{v:.3f}' for v in mcts.recent_values]}, threshold {config.training.resign_threshold}")
                    
                    if all(v < config.training.resign_threshold for v in mcts.recent_values):
                        # Resign - current player loses
                        outcome = -1  # Loss for current player
                        
                        # Update all examples with the game outcome
                        for i, example in enumerate(examples):
                            # Alternate values based on player perspective
                            example.value = outcome * ((-1) ** (i % 2))
                        
                        logger.debug(f"Game {game_id} resigned at move {move_num} with root value {root_value:.3f} (threshold: {config.training.resign_threshold})")
                        break
            
            # Select move based on policy - MCTS should ensure only legal moves
            # Note: MCTS already applies temperature scaling to visit counts using
            # the AlphaZero formula: policy[a] ∝ visits[a]^(1/temperature)
            
            if mcts.config.temperature == 0:
                # Deterministic - select highest probability action
                action = np.argmax(policy)
            else:
                # Stochastic - sample from the temperature-scaled distribution
                if np.sum(policy) > 0:
                    normalized_policy = policy / np.sum(policy)
                    action = np.random.choice(len(policy), p=normalized_policy)
                else:
                    # Fallback to uniform random
                    action = np.random.choice(len(policy))
            
            # Store example with the full policy (before move selection)
            canonical_state = game_interface.get_canonical_form(state)
            examples.append(GameExample(
                state=canonical_state,
                policy=policy,
                value=0,
                game_id=game_id,
                move_number=move_num
            ))
            
            # Apply action to game interface state first to get new state
            try:
                current_legal_moves = game_interface.get_legal_moves(state)
                logger.debug(f"[WORKER {game_idx} DEBUG] About to apply action {action}:")
                logger.debug(f"[WORKER {game_idx} DEBUG]   Move number: {move_num}")
                logger.debug(f"[WORKER {game_idx} DEBUG]   Current legal moves: {len(current_legal_moves)}")
                logger.debug(f"[WORKER {game_idx} DEBUG]   Action in legal moves: {action in current_legal_moves}")
                logger.debug(f"[WORKER {game_idx} DEBUG]   Action value: {action} (type: {type(action)})")
                if hasattr(state, 'board'):
                    logger.debug(f"[WORKER {game_idx} DEBUG]   Board occupied: {np.count_nonzero(state.board) if hasattr(state.board, 'shape') else 'unknown'}")
                
                state = game_interface.get_next_state(state, action)
                
                # CRITICAL FIX: Update MCTS tree root with new state to keep states synchronized
                # This ensures MCTS internal state matches game interface state for next search
                mcts.update_root(action, state)
                logger.debug(f"[WORKER {game_idx} DEBUG] Action {action} applied successfully - states now synchronized")
            except ValueError as e:
                # CRITICAL: If action is illegal, MCTS and game interface are out of sync!
                logger.error(f"[WORKER {game_idx} CRITICAL] State synchronization error at move {move_num}")
                logger.error(f"[WORKER {game_idx} ERROR] MCTS selected illegal action: {action}")
                logger.error(f"[WORKER {game_idx} ERROR] This indicates MCTS internal state != game interface state")
                logger.error(f"[WORKER {game_idx} ERROR] Policy for this action: {policy[action] if action < len(policy) else 'OUT_OF_BOUNDS'}")
                logger.error(f"[WORKER {game_idx} ERROR] Policy length: {len(policy)}")
                logger.error(f"[WORKER {game_idx} ERROR] Action type: {type(action)}")
                logger.error(f"[WORKER {game_idx} ERROR] Game terminal before move: {game_interface.is_terminal(state)}")
                logger.error(f"[WORKER {game_idx} ERROR] Board size: {game_interface.board_size}")
                logger.error(f"[WORKER {game_idx} ERROR] Expected action range: [0, {game_interface.get_action_space_size(state)-1}]")
                current_legal_moves = game_interface.get_legal_moves(state)
                logger.error(f"[WORKER {game_idx} ERROR] Current legal moves count: {len(current_legal_moves)}")
                logger.error(f"[WORKER {game_idx} ERROR] Legal moves range: [{min(current_legal_moves) if current_legal_moves else 'N/A'}, {max(current_legal_moves) if current_legal_moves else 'N/A'}]")
                raise
            
            # Check terminal
            if game_interface.is_terminal(state):
                outcome = game_interface.get_value(state)
                
                # Update values
                for i, example in enumerate(examples):
                    example.value = outcome * ((-1) ** (i % 2))
                break
        
        total_time = time.time() - start_time
        moves_per_sec = len(examples) / total_time if total_time > 0 else 0
        total_sims = len(examples) * config.mcts.num_simulations
        overall_sims_per_sec = total_sims / total_time if total_time > 0 else 0
        
        logger.debug(f"Worker {game_idx} completed: {len(examples)} moves in {total_time:.1f}s "
                f"({moves_per_sec:.1f} moves/s, {overall_sims_per_sec:.1f} sims/s overall)")
        
        # Clean up any GPU memory if needed
        
        return examples
        
    except Exception as e:
        # Logger already set up
        logger.error(f"[WORKER {game_idx}] Failed with error: {str(e)}")
        logger.error(f"[WORKER {game_idx}] CUDA available: {torch.cuda.is_available()}")
        logger.error(f"[WORKER {game_idx}] Device: {config.mcts.device}")
        logger.error(f"[WORKER {game_idx}] Worker device: {mcts_config.device if 'mcts_config' in locals() else 'not set'}")
        import traceback
        logger.error(traceback.format_exc())
        return []  # Return empty list instead of raising to prevent crash

