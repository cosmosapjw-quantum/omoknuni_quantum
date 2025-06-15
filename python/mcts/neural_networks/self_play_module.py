"""Self-play module for training data generation

This module handles parallel self-play game generation with progress tracking.
"""

import logging
import numpy as np
import time
import queue
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
                           num_games: int, num_workers: int) -> List[Any]:
        """Generate games in parallel with GPU evaluation service"""
        from .unified_training_pipeline import GameExample
        from mcts.utils.gpu_evaluator_service import GPUEvaluatorService
        import multiprocessing as mp
        
        # Get resource allocation from config (already adjusted for hardware)
        allocation = getattr(self.config, '_resource_allocation', None)
        if allocation is None:
            # Fallback: calculate allocation if not already done
            hardware = self.config.detect_hardware()
            allocation = self.config.calculate_resource_allocation(hardware, num_workers)
            self.config._resource_allocation = allocation
        
        logger.info(f"[SELF-PLAY] Starting parallel self-play with {num_workers} workers")
        
        # Set up multiprocessing start method
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            # Already set, that's fine
            pass
        
        examples = []
        
        # Create GPU evaluator service in main process
        gpu_service = GPUEvaluatorService(
            model=model,
            device=self.config.mcts.device,
            batch_size=256,
            batch_timeout=0.01
        )
        
        # Start the service
        gpu_service.start()
        logger.debug(f"[SELF-PLAY] GPU evaluation service started on device: {self.config.mcts.device}")
        
        try:
            # Get the request and response queues for workers
            request_queue, response_queue = gpu_service.get_queues()
            
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
                                # Wait for result with timeout
                                game_examples = q.get(timeout=300)  # 5 minute timeout
                            else:
                                # Process already finished, get result immediately
                                game_examples = q.get(timeout=10)
                            
                            if game_examples:
                                examples.extend(game_examples)
                                logger.debug(f"[SELF-PLAY] Collected {len(game_examples)} examples from game {game_idx}")
                            else:
                                logger.warning(f"[SELF-PLAY] Game {game_idx} returned empty examples")
                                
                        except queue.Empty:
                            logger.error(f"[SELF-PLAY] Timeout waiting for result from game {game_idx}")
                            if p.is_alive():
                                logger.warning(f"[SELF-PLAY] Terminating stuck process for game {game_idx}")
                                p.terminate()
                                
                        except Exception as e:
                            logger.error(f"[SELF-PLAY] Failed to get result from game {game_idx}: {e}")
                            import traceback
                            traceback.print_exc()
                            
                        finally:
                            # Ensure process is terminated
                            if p.is_alive():
                                p.terminate()
                            p.join(timeout=5)
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
                    policy = policy / np.sum(policy)  # Normalize to ensure valid probabilities
                    action = np.random.choice(len(policy), p=policy)
                else:
                    # Fallback to uniform random if all probabilities are zero
                    legal_moves = self.game_interface.get_legal_moves(state)
                    if not legal_moves:
                        raise ValueError(f"No legal actions available at move {move_num}")
                    action = np.random.choice(legal_moves)
            
            # Double-check the action is actually legal
            if not self.game_interface.is_move_legal(state, action):
                logger.error(f"MCTS returned illegal action {action} at move {move_num}")
                legal_moves = self.game_interface.get_legal_moves(state)
                logger.error(f"Legal moves: {legal_moves[:10]}... (total {len(legal_moves)})")
                logger.error(f"Policy values: {policy[legal_moves[:10]]}")
                # Try to select best legal action from policy
                legal_policies = [(a, policy[a]) for a in legal_moves]
                legal_policies.sort(key=lambda x: x[1], reverse=True)
                action = legal_policies[0][0]
                logger.warning(f"Using best legal action: {action} with policy value {policy[action]}")
            
            # Store example with the full policy (before move selection)
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
                logger.error(f"Policy for this action: {policy[action]}")
                logger.error(f"Game terminal before move: {self.game_interface.is_terminal(state)}")
                raise
            
            # Reset MCTS tree for next search (no tree reuse for better exploration)
            mcts.reset_tree()
            
            # Check terminal
            if self.game_interface.is_terminal(state):
                outcome = self.game_interface.get_value(state)
                
                # Update values based on game outcome
                for i, example in enumerate(examples):
                    example.value = outcome * ((-1) ** (i % 2))
                break
        
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

    def _create_mcts(self, evaluator: Any):
        """Create MCTS instance with quantum features if enabled"""
        # Use unified MCTS module
        from mcts.core.mcts import MCTS, MCTSConfig
        from mcts.quantum.quantum_features import create_quantum_mcts
        from mcts.utils.config_system import QuantumLevel
        
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
            adaptive_wave_sizing=self.config.mcts.adaptive_wave_sizing,
            
            # Memory and optimization (reduced for workers)
            memory_pool_size_mb=max(512, self.config.mcts.memory_pool_size_mb // 4),  # Reduced for workers
            max_tree_nodes=self.config.mcts.max_tree_nodes // 2,  # Reduced for workers
            use_mixed_precision=self.config.mcts.use_mixed_precision,
            use_cuda_graphs=self.config.mcts.use_cuda_graphs,
            use_tensor_cores=self.config.mcts.use_tensor_cores,
            
            # Game and device configuration  
            device=self.config.mcts.device,
            game_type=self.game_type,
            board_size=self.config.game.board_size,
            
            # Enable quantum features if configured
            enable_quantum=self.config.mcts.enable_quantum,
            quantum_config=self._create_quantum_config() if self.config.mcts.enable_quantum else None,
            
            # Virtual loss for parallel exploration
            enable_virtual_loss=True,
            virtual_loss=self.config.mcts.virtual_loss,
            # Debug options
            enable_debug_logging=(self.config.log_level == "DEBUG"),
            profile_gpu_kernels=False  # Don't profile in main process MCTS
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
    import os
    import sys
    
    # Worker startup (minimal output)
    
    # Workers CAN use GPU for tree operations (custom CUDA kernels)
    # Only neural network evaluation goes through the service
    # Don't disable CUDA entirely - just avoid creating models on GPU in workers
    
    try:
        # Import torch first to check CUDA state
        import torch
        # Torch imported, CUDA availability checked
        print(f"[WORKER {game_idx}] CUDA device count: {torch.cuda.device_count()}", file=sys.stderr)
        
        from .unified_training_pipeline import GameExample
        from mcts.core.game_interface import GameInterface, GameType
        from mcts.neural_networks.nn_model import create_model
        # Use optimized MCTS directly as requested
        from mcts.core.optimized_mcts import MCTS, MCTSConfig
        from mcts.quantum.quantum_features import create_quantum_mcts
        from mcts.utils.config_system import QuantumLevel
        
        # Set up logging level for worker
        logging.basicConfig(level=getattr(logging, config.log_level))
        logger = logging.getLogger(__name__)
        
        logger.debug(f"[WORKER {game_idx}] Worker started - device: {config.mcts.device}, model params: {len(model_state_dict)}")
        
        # Deserialize the state dict from numpy format
        from mcts.utils.safe_multiprocessing import deserialize_state_dict_from_multiprocessing
        
        logger.debug(f"[WORKER {game_idx}] Deserializing state dict from numpy format...")
        model_state_dict = deserialize_state_dict_from_multiprocessing(model_state_dict)
        logger.debug(f"[WORKER {game_idx}] Deserialized {len(model_state_dict)} parameters")
        
        # Force CPU for workers to avoid CUDA multiprocessing issues
        # Workers can still use CPU effectively for self-play
        device = torch.device('cpu')
        logger.debug(f"[WORKER {game_idx}] Using CPU device for all operations")
        
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
        
        # Load state dict - already contains CPU tensors from deserialization
        model.load_state_dict(model_state_dict)
        
        # Move model to the correct device
        model = model.to(device)
        model.eval()
    
        # Create evaluator
        from mcts.core.evaluator import AlphaZeroEvaluator
        evaluator = AlphaZeroEvaluator(
        model=model,
        device=str(device)  # Convert device object to string
        )
        
        # Create MCTS
        mcts_config = MCTSConfig(
        num_simulations=config.mcts.num_simulations,
        c_puct=config.mcts.c_puct,
        temperature=1.0,  # Will be adjusted during play
        device=str(device),  # Convert device object to string
        game_type=game_type,
        min_wave_size=config.mcts.min_wave_size,
        max_wave_size=config.mcts.max_wave_size,
        adaptive_wave_sizing=config.mcts.adaptive_wave_sizing,
        dirichlet_epsilon=0.25,  # Add exploration noise to root
        dirichlet_alpha=0.3,  # Dirichlet noise parameter
        enable_debug_logging=False
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
            # AlphaZero-style temperature annealing
            # Use exploration temperature for first N moves, then switch to deterministic
            if move_num < config.mcts.temperature_threshold:
                # Exploration phase - use stochastic temperature
                mcts.config.temperature = 1.0
            else:
                # Exploitation phase - deterministic play
                mcts.config.temperature = 0.0
            
            # Search with MCTS
            policy = mcts.search(state, num_simulations=config.mcts.num_simulations)
            
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
                    policy = policy / np.sum(policy)  # Normalize to ensure valid probabilities
                    action = np.random.choice(len(policy), p=policy)
                else:
                    # Fallback to uniform random if all probabilities are zero
                    legal_moves = game_interface.get_legal_moves(state)
                    if not legal_moves:
                        raise ValueError(f"No valid actions available at move {move_num}")
                    action = np.random.choice(legal_moves)
            
            # Store example with the full policy (before move selection)
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
            
            # Reset MCTS tree for next search (no tree reuse for better exploration)
            mcts.reset_tree()
            
            # Check terminal
            if game_interface.is_terminal(state):
                outcome = game_interface.get_value(state)
                
                # Update values
                for i, example in enumerate(examples):
                    example.value = outcome * ((-1) ** (i % 2))
                break
        
        return examples
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Worker {game_idx} failed with error: {str(e)}")
        logger.error(f"Config device: {config.mcts.device}, Worker device: {device if 'device' in locals() else 'not set'}")
        import traceback
        logger.error(traceback.format_exc())
        raise


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
    import os
    import sys
    
    # Worker startup with hardware allocation
    
    # CRITICAL: Limit CUDA memory for workers to prevent OOM
    # Workers use GPU for MCTS tree operations but not for models
    max_split_mb = min(512, allocation.get('gpu_memory_per_worker_mb', 512)) if allocation.get('gpu_memory_per_worker_mb', 0) > 0 else 512
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{max_split_mb}'
    
    try:
        import torch
        # Torch imported, CUDA availability checked
        
        # Set memory fraction for worker processes based on allocation
        if torch.cuda.is_available() and allocation.get('gpu_memory_fraction', 0) > 0:
            # Use the calculated fraction for this worker
            worker_fraction = allocation['gpu_memory_fraction'] / allocation['num_workers']
            torch.cuda.set_per_process_memory_fraction(worker_fraction)
            # CUDA memory fraction set silently
        
        from .unified_training_pipeline import GameExample
        from mcts.core.game_interface import GameInterface, GameType
        # Use optimized MCTS directly as requested
        from mcts.core.optimized_mcts import MCTS, MCTSConfig
        from mcts.gpu.gpu_game_states import GameType as GPUGameType
        from mcts.quantum.quantum_features import create_quantum_mcts
        from mcts.utils.config_system import QuantumLevel
        from mcts.utils.gpu_evaluator_service import RemoteEvaluator
        
        # Set up logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        logger = logging.getLogger(__name__)
        
        logger.debug(f"[WORKER {game_idx}] Worker started - using GPU service for evaluations")
        
        # Debug CUDA availability
        logger.debug(f"[WORKER {game_idx}] CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}")
        
        # Create game interface
        game_type = GameType[config.game.game_type.upper()]
        game_interface = GameInterface(game_type, config.game.board_size)
        
        # Create remote evaluator that sends requests to GPU service
        remote_evaluator = RemoteEvaluator(request_queue, response_queue, action_size, worker_id=game_idx)
        
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
                policies, values = self.evaluator.evaluate_batch(states, legal_masks, temperature)
                # Convert numpy to torch tensors
                policies_tensor = torch.from_numpy(policies).float().to(self.device)
                values_tensor = torch.from_numpy(values).float().to(self.device)
                return policies_tensor, values_tensor
        
        # Determine device for tensors
        tensor_device = 'cuda' if (torch.cuda.is_available() and allocation.get('use_gpu_for_workers', True)) else 'cpu'
        evaluator = TensorEvaluator(remote_evaluator, tensor_device)
        logger.debug(f"[WORKER {game_idx}] Created tensor evaluator wrapper")
        
        # Create MCTS
        # Workers CAN use GPU for tree operations (custom CUDA kernels)
        # Only neural network evaluation goes through the service
            
        # Game type conversion happens silently
        
        mcts_config = MCTSConfig(
            num_simulations=config.mcts.num_simulations,
            c_puct=config.mcts.c_puct,
            temperature=1.0,  # Will be adjusted during play
            # Use hardware-aware device selection
            device='cuda' if (torch.cuda.is_available() and allocation.get('use_gpu_for_workers', True)) else 'cpu',
            game_type=GPUGameType[game_type.name],  # Convert to GPU game type enum
            min_wave_size=config.mcts.min_wave_size,
            max_wave_size=config.mcts.max_wave_size,
            adaptive_wave_sizing=config.mcts.adaptive_wave_sizing,
            # use_optimized_implementation=True,  # This parameter doesn't exist in MCTSConfig
            # Dynamic memory pool based on hardware allocation
            memory_pool_size_mb=min(allocation.get('memory_per_worker_mb', 512) // 2, 512),
            use_mixed_precision=True,
            use_cuda_graphs=False,  # Disable CUDA graphs in workers to save memory
            use_tensor_cores=True,
            # Dynamic tree size based on available memory
            max_tree_nodes=min(100000, config.mcts.max_tree_nodes // allocation.get('num_workers', 1)),
            dirichlet_epsilon=0.25,  # Add exploration noise to root
            dirichlet_alpha=0.3,  # Dirichlet noise parameter
            board_size=config.game.board_size,  # Add missing board_size parameter
            # Quantum features
            enable_quantum=config.mcts.enable_quantum,
            quantum_config=None,  # Will be created if needed
            # Debug options - disable for workers
            enable_debug_logging=False,
            profile_gpu_kernels=False
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
        
        # Initialize MCTS internals if needed
        if hasattr(mcts, 'initialize') and callable(mcts.initialize):
            try:
                mcts.initialize()
                logger.debug(f"[WORKER {game_idx}] MCTS initialized successfully")
            except Exception as e:
                logger.warning(f"[WORKER {game_idx}] Failed to initialize MCTS: {e}")
        
        # Optimize MCTS for hardware (enables GPU kernels, mixed precision, etc.)
        if hasattr(mcts, 'optimize_for_hardware'):
            try:
                mcts.optimize_for_hardware()
                logger.debug(f"[WORKER {game_idx}] MCTS optimized for hardware")
            except Exception as e:
                logger.warning(f"[WORKER {game_idx}] Failed to optimize MCTS: {e}")
        logger.debug(f"[WORKER {game_idx}] MCTS configured - device: {mcts_config.device}, memory: {mcts_config.memory_pool_size_mb}MB")
        
        # CUDA kernel checks happen silently
        
        # Play game
        examples = []
        state = game_interface.create_initial_state()
        game_id = f"iter{iteration}_game{game_idx}"
        
        for move_num in range(config.training.max_moves_per_game):
            # AlphaZero-style temperature annealing
            # Use exploration temperature for first N moves, then switch to deterministic
            if move_num < config.mcts.temperature_threshold:
                # Exploration phase - use stochastic temperature
                mcts.config.temperature = 1.0
            else:
                # Exploitation phase - deterministic play
                mcts.config.temperature = 0.0
            
            # Search with MCTS
            policy = mcts.search(state, num_simulations=config.mcts.num_simulations)
            
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
                    policy = policy / np.sum(policy)  # Normalize to ensure valid probabilities
                    action = np.random.choice(len(policy), p=policy)
                else:
                    # Fallback to uniform random if all probabilities are zero
                    legal_moves = game_interface.get_legal_moves(state)
                    if not legal_moves:
                        raise ValueError(f"No legal actions available at move {move_num}")
                    action = np.random.choice(legal_moves)
            
            # Store example with the full policy (before move selection)
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
            
            # Reset MCTS tree for next search (no tree reuse for better exploration)
            mcts.reset_tree()
            
            # Check terminal
            if game_interface.is_terminal(state):
                outcome = game_interface.get_value(state)
                
                # Update values
                for i, example in enumerate(examples):
                    example.value = outcome * ((-1) ** (i % 2))
                break
        
        logger.debug(f"[WORKER {game_idx}] Game completed with {len(examples)} moves")
        
        # Clean up GPU memory before returning
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return examples
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"[WORKER {game_idx}] Failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


# Remove duplicate function definition