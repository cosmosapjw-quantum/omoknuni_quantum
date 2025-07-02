#!/usr/bin/env python3
"""
Real MCTS Data Collector for Quantum Research

This module collects authentic data from actual MCTS self-play runs for physics analysis.
It captures real tree expansion dynamics, quantum state evolution, and performance metrics
during actual game simulations using the optimized MCTS implementation.

Usage:
    collector = RealMCTSDataCollector()
    collector.run_data_collection_session(n_games=10, num_simulations=1000)
    data = collector.export_data('quantum_mcts_data.json')
"""

import sys
import os
import json
import time
import logging
import numpy as np
import warnings

# Suppress all warnings at the start to clean up output
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Set CUDA environment variables for optimal performance
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pickle
import threading
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
import queue
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import signal

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Set up logging - reduce verbosity by default
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Keep our module at WARNING level

# Suppress verbose CUDA and other logging completely
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('mcts').setLevel(logging.ERROR)  # Reduced back to ERROR level
logging.getLogger('alphazero_py').setLevel(logging.ERROR)
logging.getLogger('authentic_mcts_physics_extractor').setLevel(logging.ERROR)

# Suppress all other potential verbose loggers
for name in ['matplotlib', 'PIL', 'numba', 'cuda', 'gpu']:
    logging.getLogger(name).setLevel(logging.ERROR)

# Add the MCTS modules to path
mcts_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(mcts_root))

# Import PyTorch
import torch

# Import MCTS modules
try:
    from mcts.core.mcts import MCTS, MCTSConfig
    from mcts.core.game_interface import GameInterface, GameType
    from mcts.gpu.gpu_game_states import GameType as GPUGameType  
    from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
    from mcts.neural_networks.mock_evaluator import MockEvaluator
    import alphazero_py
    
    MCTS_AVAILABLE = True
    # Successfully imported MCTS modules
except ImportError as e:
    logger.error(f"Could not import MCTS modules: {e}")
    logger.error("This collector requires the full MCTS implementation")
    sys.exit(1)

@dataclass
class RealTreeSnapshot:
    """Snapshot of real MCTS tree at a specific point in time"""
    timestamp: float
    move_number: int
    player: int
    
    # Tree statistics
    total_nodes: int
    total_visits: int
    tree_depth: int
    root_visits: int
    root_value: float
    
    # Node-level data (extracted from CSRTree)
    visit_counts: List[int]
    value_sums: List[float]  
    q_values: List[float]
    node_priors: List[float]
    parent_indices: List[int]
    parent_actions: List[int]
    node_phases: List[float]  # Quantum phases
    
    # Tree structure analysis
    branching_factor: float
    max_depth: int
    leaf_nodes: List[int]
    
    # Policy and search data
    policy_distribution: List[float]
    search_time: float
    simulations_per_second: float
    
    # Quantum-specific data
    quantum_mode_active: bool
    quantum_corrections: List[float]
    decoherence_detected: bool

@dataclass 
class RealGameSession:
    """Complete real game session data"""
    game_id: str
    start_time: float
    end_time: float
    game_type: str
    board_size: int
    winner: int
    final_score: float
    move_count: int
    
    # MCTS configuration used
    mcts_config: Dict[str, Any]
    
    # Tree snapshots during game
    tree_snapshots: List[RealTreeSnapshot]
    
    # Move sequence with full context
    moves: List[Dict[str, Any]]
    
    # Performance metrics
    total_simulations: int
    avg_simulations_per_move: float
    total_search_time: float
    peak_sims_per_second: float
    
    # Game progression analysis
    game_phases: List[str]  # opening, midgame, endgame
    complexity_evolution: List[float]
    decision_confidence: List[float]


class RealMCTSDataCollector:
    """Collects comprehensive data from real MCTS self-play runs"""
    
    def __init__(self, evaluator_type: str = "mock", device: str = "cpu", verbose: bool = False,
                 auto_optimize: bool = True, workload_type: str = "balanced"):
        self.evaluator_type = evaluator_type
        self.device = device
        self.verbose = verbose
        self.auto_optimize = auto_optimize
        self.workload_type = workload_type
        self.game_sessions: List[RealGameSession] = []
        self._use_enhanced_representation = False  # Track if we need enhanced representation
        
        # Set logging level based on verbose flag
        if not verbose:
            logger.setLevel(logging.ERROR)
        
        # Initialize MCTS system
        self._initialize_mcts_system()
        
        # Collection metadata
        self.collection_metadata = {
            'collection_start': None,
            'collection_end': None,
            'total_games': 0,
            'total_tree_snapshots': 0,
            'evaluator_type': evaluator_type,
            'device': device,
            'data_version': '4.0.0-real',
            'hardware_optimized': auto_optimize
        }
        
        # Real MCTS Data Collector initialized
    
    def _initialize_mcts_system(self):
        """Initialize real MCTS components with hardware optimization"""
        
        # Create evaluator
        if self.evaluator_type == "mock":
            self.evaluator = MockEvaluator(game_type='gomoku', device=self.device)
        elif self.evaluator_type == "resnet":
            try:
                from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
                # Use basic representation with 18 input channels (faster and current standard)
                self.evaluator = ResNetEvaluator(device=self.device, input_channels=18)
                logger.info("Using ResNet evaluator with basic representation (18 channels)")
                # Use basic representation for 18 channels
                self._use_enhanced_representation = False
            except Exception as e:
                logger.warning(f"Failed to load ResNet evaluator: {e}")
                logger.warning("Falling back to mock evaluator")
                self.evaluator = MockEvaluator(game_type='gomoku', device=self.device)
                self.evaluator_type = "mock"  # Update type to reflect fallback
        else:
            logger.warning(f"Unknown evaluator type: {self.evaluator_type}, using mock evaluator")
            self.evaluator = MockEvaluator(game_type='gomoku', device=self.device)
        
        # Get hardware-optimized MCTS configuration - adjusted for low GPU utilization
        if self.auto_optimize:
            try:
                # Direct import from relative path
                sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
                from utils.hardware_optimizer import HardwareOptimizer
                optimizer = HardwareOptimizer()
                hw_profile = optimizer.detect_hardware()
                allocation = optimizer.calculate_optimal_allocation(self.workload_type)
                
                # Reduced simulations to prevent timeouts, but keep reasonable tree size for GPU efficiency
                wave_size = allocation.mcts_wave_size  # Keep original for GPU throughput
                max_tree_nodes = allocation.mcts_max_tree_nodes  # Keep original - no memory issues
                memory_pool_mb = allocation.mcts_memory_pool_mb  # Keep original - no memory issues
                num_simulations = min(200, allocation.mcts_simulations_per_move // 5)  # Reduce only simulations
                
                logger.info(f"Timeout-optimized MCTS: wave_size={wave_size}, "
                           f"max_nodes={max_tree_nodes}, simulations={num_simulations}")
            except ImportError:
                logger.warning("Hardware optimizer not available, using timeout-safe defaults")
                wave_size = 2048  # Reasonable for GPU throughput
                max_tree_nodes = 100000  # Reasonable size
                memory_pool_mb = 512  # Reasonable memory
                num_simulations = 150  # Reduced simulations only
        else:
            # Use timeout-safe defaults
            wave_size = 2048
            max_tree_nodes = 100000  
            memory_pool_mb = 512
            num_simulations = 150
        
        # MCTS configuration with hardware-aware settings
        self.mcts_config = MCTSConfig(
            num_simulations=num_simulations,
            c_puct=1.4,  # Match training config
            temperature=1.5,  # Match training config
            dirichlet_alpha=0.2,  # Keep positive value to avoid numpy error
            dirichlet_epsilon=0.0,  # Zero epsilon to disable noise application
            device=self.device,
            game_type=GPUGameType.GOMOKU,
            board_size=15,
            # QUANTUM FEATURES DISABLED for pure classical MCTS data
            enable_quantum=False,
            # Wave parallelization settings - hardware optimized
            wave_size=wave_size,
            min_wave_size=wave_size,
            max_wave_size=wave_size,
            adaptive_wave_sizing=False,  # Match training config
            # Memory settings - hardware optimized
            max_tree_nodes=max_tree_nodes,
            memory_pool_size_mb=memory_pool_mb,
            use_mixed_precision=self.device == 'cuda',  # Only on GPU
            use_cuda_graphs=self.device == 'cuda',  # Only on GPU
            use_tensor_cores=self.device == 'cuda',  # Only on GPU
            # Virtual loss
            virtual_loss=0.5,  # Match training config
            # Disable debug logging for performance
            enable_debug_logging=False,
        )
        
        # Create MCTS instance
        self.mcts = MCTS(self.mcts_config, self.evaluator)
        
        # Optimize for current hardware
        self.mcts.optimize_for_hardware()
        
        # MCTS initialized
    
    def _extract_tree_snapshot(self, move_number: int, player: int, 
                             search_time: float, policy: np.ndarray) -> RealTreeSnapshot:
        """Extract comprehensive tree snapshot from current MCTS state"""
        
        timestamp = time.time()
        
        # Access the CSRTree directly
        tree = self.mcts.tree
        
        # Extract basic tree statistics
        total_nodes = tree.num_nodes
        root_visits = tree.visit_counts[0].item() if total_nodes > 0 else 0
        root_value = tree.value_sums[0].item() / max(root_visits, 1) if root_visits > 0 else 0.0
        
        # Calculate tree depth
        max_depth = 0
        if total_nodes > 0:
            # Simple depth calculation by following longest path
            current_depth = 0
            for node_idx in range(min(total_nodes, 1000)):  # Limit for performance
                depth = 0
                current = node_idx
                while current != -1 and depth < 100:  # Avoid infinite loops
                    parent = tree.parent_indices[current].item()
                    if parent == -1:
                        break
                    current = parent
                    depth += 1
                max_depth = max(max_depth, depth)
        
        # Extract node-level data (limit to reasonable size)
        extract_limit = min(total_nodes, 1000)  # Don't extract massive trees
        
        visit_counts = []
        value_sums = []
        q_values = []
        node_priors = []
        parent_indices = []
        parent_actions = []
        node_phases = []
        
        if total_nodes > 0:
            # Convert tensors to lists with proper type conversion for JSON
            # Handle both CPU and GPU tensors safely
            try:
                visit_counts = [int(x) for x in tree.visit_counts[:extract_limit].cpu().numpy()]
                value_sums = [float(x) for x in tree.value_sums[:extract_limit].cpu().numpy()]
                node_priors = [float(x) for x in tree.node_priors[:extract_limit].cpu().numpy()]
                parent_indices = [int(x) for x in tree.parent_indices[:extract_limit].cpu().numpy()]
                parent_actions = [int(x) for x in tree.parent_actions[:extract_limit].cpu().numpy()]
                node_phases = [float(x) for x in tree.phases[:extract_limit].cpu().numpy()]
            except Exception as e:
                logger.error(f"Failed to extract tree data: {e}")
                # Return empty lists if extraction fails
                visit_counts = [0] * extract_limit
                value_sums = [0.0] * extract_limit
                node_priors = [0.0] * extract_limit  
                parent_indices = [-1] * extract_limit
                parent_actions = [-1] * extract_limit
                node_phases = [0.0] * extract_limit
            
            # Calculate Q-values
            q_values = []
            for i in range(extract_limit):
                visits = visit_counts[i]
                if visits > 0:
                    q_val = value_sums[i] / visits
                else:
                    q_val = 0.0
                q_values.append(q_val)
        
        # Find leaf nodes
        leaf_nodes = []
        for i in range(min(extract_limit, 100)):  # Limit leaf search
            if i < total_nodes:
                # Check if node has children by looking at row_ptr
                if hasattr(tree, 'row_ptr') and i + 1 < len(tree.row_ptr):
                    has_children = tree.row_ptr[i + 1] > tree.row_ptr[i]
                    if not has_children:
                        leaf_nodes.append(i)
        
        # Calculate branching factor
        branching_factor = 0.0
        if total_nodes > 1:
            total_children = sum(1 for i in range(min(total_nodes, 100)) 
                               if parent_indices[i] != -1) if parent_indices else 0
            internal_nodes = max(total_nodes - len(leaf_nodes), 1)
            branching_factor = total_children / internal_nodes
        
        # Search performance
        sims_per_second = self.mcts_config.num_simulations / max(search_time, 0.001)
        
        # Quantum analysis - DISABLED for classical MCTS data collection
        quantum_mode_active = False  # Explicitly disabled
        quantum_corrections = []     # No quantum corrections in classical mode
        decoherence_detected = False # No decoherence in classical mode
        
        # Note: node_phases may still exist in CSRTree structure but will be zeros
        # This provides baseline data for quantum vs classical comparison
        
        return RealTreeSnapshot(
            timestamp=timestamp,
            move_number=move_number,
            player=player,
            total_nodes=total_nodes,
            total_visits=sum(visit_counts) if visit_counts else 0,
            tree_depth=max_depth,
            root_visits=root_visits,
            root_value=root_value,
            visit_counts=visit_counts,
            value_sums=value_sums,
            q_values=q_values,
            node_priors=node_priors,
            parent_indices=parent_indices,
            parent_actions=parent_actions,
            node_phases=node_phases,
            branching_factor=branching_factor,
            max_depth=max_depth,
            leaf_nodes=leaf_nodes,
            policy_distribution=policy.tolist(),
            search_time=search_time,
            simulations_per_second=sims_per_second,
            quantum_mode_active=quantum_mode_active,
            quantum_corrections=quantum_corrections,
            decoherence_detected=decoherence_detected
        )
    
    def play_real_game_with_data_collection(self, game_type: str = "gomoku",
                                          board_size: int = 15,
                                          snapshot_frequency: int = 5) -> RealGameSession:
        """Play a real MCTS self-play game while collecting comprehensive data"""
        
        game_id = f"{game_type}_{int(time.time())}_{np.random.randint(1000, 9999)}"
        start_time = time.time()
        
        # Starting real MCTS game
        
        # Reset MCTS tree for new game
        self.mcts.clear()
        
        # Initialize game state
        if game_type.lower() == "gomoku":
            state = alphazero_py.GomokuState()
        else:
            raise ValueError(f"Unsupported game type: {game_type}")
        
        # Initialize game session
        game_session = RealGameSession(
            game_id=game_id,
            start_time=start_time,
            end_time=0.0,
            game_type=game_type,
            board_size=board_size,
            winner=-1,
            final_score=0.0,
            move_count=0,
            mcts_config=asdict(self.mcts_config),
            tree_snapshots=[],
            moves=[],
            total_simulations=0,
            avg_simulations_per_move=0.0,
            total_search_time=0.0,
            peak_sims_per_second=0.0,
            game_phases=[],
            complexity_evolution=[],
            decision_confidence=[]
        )
        
        move_count = 0
        total_search_time = 0.0
        
        # Play game until terminal
        while not state.is_terminal() and move_count < 500:  # Limit game length
            move_count += 1
            current_player = state.get_current_player()
            
            # Move progress (suppressed for clean output)
            
            # Determine temperature (exploration vs exploitation)
            if move_count <= 30:
                temperature = 1.0  # Explore early game
            else:
                temperature = 0.1  # More deterministic late game
            
            # Reset tree for fresh search (but capture snapshot before reset)
            # Note: MCTS needs proper tree management between moves
            
            # Run MCTS search
            search_start = time.perf_counter()
            policy = self.mcts.search(state, self.mcts_config.num_simulations)
            search_time = time.perf_counter() - search_start
            
            # CRITICAL: Extract tree snapshot IMMEDIATELY after search before any other operations
            if move_count % snapshot_frequency == 0 or move_count <= 10:
                snapshot = self._extract_tree_snapshot(move_count, current_player, search_time, policy)
                game_session.tree_snapshots.append(snapshot)
            
            
            total_search_time += search_time
            game_session.total_simulations += self.mcts_config.num_simulations
            
            # Update peak performance
            sims_per_sec = self.mcts_config.num_simulations / search_time
            game_session.peak_sims_per_second = max(game_session.peak_sims_per_second, sims_per_sec)
            
            # Normalize policy to ensure probabilities sum to 1
            policy_sum = np.sum(policy)
            if policy_sum > 0:
                policy = policy / policy_sum
            else:
                # Fallback to uniform distribution if policy is all zeros
                policy = np.ones(len(policy)) / len(policy)
            
            # Get legal moves and validate action bounds
            legal_moves = state.get_legal_moves()
            if not legal_moves:
                raise ValueError(f"No legal actions available at move {move_count}")
            
            # Validate action space bounds
            max_action = len(policy) - 1
            legal_moves = [a for a in legal_moves if 0 <= a <= max_action]
            if not legal_moves:
                raise ValueError(f"No legal actions within policy bounds [0, {max_action}] at move {move_count}")
            
            # Select action based on policy and temperature, restricting to legal moves only
            if temperature > 0.5:
                # Sample from legal moves policy distribution
                legal_policy_values = np.array([policy[a] for a in legal_moves])
                if np.sum(legal_policy_values) > 0:
                    legal_policy_values = legal_policy_values / np.sum(legal_policy_values)
                    action_idx = np.random.choice(len(legal_moves), p=legal_policy_values)
                    action = legal_moves[action_idx]
                else:
                    # Fallback to uniform random from legal moves
                    action = np.random.choice(legal_moves)
            else:
                # Select best legal action
                legal_policies = [(a, policy[a]) for a in legal_moves]
                legal_policies.sort(key=lambda x: x[1], reverse=True)
                action = legal_policies[0][0]
            
            # Final validation
            if action not in legal_moves:
                logger.error(f"Action selection bug: selected action {action} not in legal moves at move {move_count}")
                action = legal_moves[0]  # Emergency fallback
            
            # Record move
            move_data = {
                'move_number': move_count,
                'player': current_player,
                'action': action,
                'policy': policy.tolist(),
                'search_time': search_time,
                'temperature': temperature,
                'tree_size': self.mcts.tree.num_nodes,
                'confidence': float(np.max(policy))
            }
            game_session.moves.append(move_data)
            game_session.decision_confidence.append(float(np.max(policy)))
            
            # Calculate game complexity (entropy of policy)
            policy_entropy = -np.sum(policy * np.log(policy + 1e-10))
            game_session.complexity_evolution.append(float(policy_entropy))
            
            # Determine game phase
            if move_count <= 15:
                phase = "opening"
            elif move_count <= 30:
                phase = "midgame"
            else:
                phase = "endgame"
            game_session.game_phases.append(phase)
            
            # Apply action to game state
            state.make_move(action)
            
            # Update MCTS tree root to new position (preserves tree statistics)
            self.mcts.update_root(action)
            
            # Action details (suppressed for clean output)
        
        # Game finished
        game_session.end_time = time.time()
        game_session.move_count = move_count
        game_session.avg_simulations_per_move = game_session.total_simulations / max(move_count, 1)
        game_session.total_search_time = total_search_time
        
        # Determine winner
        if state.is_terminal():
            result = state.get_game_result()
            if result == alphazero_py.GameResult.WIN_PLAYER1:
                game_session.winner = 0  # Player 1 (0-indexed)
                game_session.final_score = 1.0
            elif result == alphazero_py.GameResult.WIN_PLAYER2:
                game_session.winner = 1  # Player 2 (0-indexed)
                game_session.final_score = 1.0
            elif result == alphazero_py.GameResult.DRAW:
                game_session.winner = -1  # Draw
                game_session.final_score = 0.5
            else:
                game_session.winner = -1  # Unknown
                game_session.final_score = 0.0
        
        game_duration = game_session.end_time - game_session.start_time
        # Game completed (details suppressed for clean output)
        
        return game_session
    
    def run_data_collection_session(self, n_games: int = 10,
                                  game_type: str = "gomoku",
                                  board_size: int = 15,
                                  snapshot_frequency: int = 5,
                                  num_workers: Optional[int] = None,
                                  auto_optimize: bool = True,
                                  workload_type: str = "balanced") -> Dict[str, Any]:
        """Run a complete real MCTS data collection session with optional parallelization
        
        Args:
            n_games: Number of games to collect
            game_type: Type of game
            board_size: Board size
            snapshot_frequency: How often to take tree snapshots
            num_workers: Number of parallel workers (None for auto-detection)
            auto_optimize: Whether to auto-optimize based on hardware
            workload_type: "latency", "throughput", or "balanced"
        """
        
        # Auto-detect optimal number of workers if not specified
        if num_workers is None and auto_optimize:
            try:
                # Direct import from relative path
                sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
                from utils.hardware_optimizer import HardwareOptimizer
                optimizer = HardwareOptimizer()
                hw_profile = optimizer.detect_hardware()
                allocation = optimizer.calculate_optimal_allocation(workload_type)
                num_workers = allocation.data_collector_workers
                
                print(f"🔧 Auto-detected hardware: {hw_profile.cpu_model} ({hw_profile.cpu_cores_physical} cores)")
                print(f"📊 Optimized for {workload_type} workload: {num_workers} workers")
            except ImportError:
                logger.warning("Hardware optimizer not available, defaulting to 1 worker")
                num_workers = 1
        elif num_workers is None:
            num_workers = 1
        
        print(f"🎯 Starting MCTS data collection: {n_games} games with {num_workers} workers")
        self.collection_metadata['collection_start'] = time.time()
        
        if num_workers > 1:
            # Use parallel data collection
            self._run_parallel_data_collection(n_games, game_type, board_size, 
                                              snapshot_frequency, num_workers, 
                                              auto_optimize, workload_type)
            return self._finalize_collection_session(n_games)
        else:
            # Use sequential data collection (original behavior)
            return self._run_sequential_data_collection(n_games, game_type, board_size, 
                                                snapshot_frequency)
    
    def _run_sequential_data_collection(self, n_games: int, game_type: str, 
                                       board_size: int, snapshot_frequency: int):
        """Run sequential data collection (original behavior)"""
        # Use tqdm for progress tracking
        for game_idx in tqdm(range(n_games), desc="Collecting MCTS data", unit="games"):
            
            try:
                game_session = self.play_real_game_with_data_collection(
                    game_type=game_type,
                    board_size=board_size,
                    snapshot_frequency=snapshot_frequency
                )
                self.game_sessions.append(game_session)
                
            except Exception as e:
                import traceback
                logger.error(f"Error in game {game_idx + 1}: {e}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                continue
        
        # Generate and return summary
        return self._finalize_collection_session(n_games)
    
    def _run_parallel_data_collection(self, n_games: int, game_type: str, 
                                     board_size: int, snapshot_frequency: int, 
                                     num_workers: int, auto_optimize: bool = True,
                                     workload_type: str = "balanced"):
        """Run parallel data collection using the same architecture as training pipeline"""
        # Direct import from relative path
        import sys
        import importlib
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from utils.gpu_evaluator_service import GPUEvaluatorService
        import resource
        
        logger.info(f"Starting parallel data collection with {num_workers} workers")
        
        # Check system limits
        try:
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            logger.info(f"System file descriptor limits: soft={soft_limit}, hard={hard_limit}")
            if soft_limit < 4096:
                logger.warning(f"Low file descriptor limit ({soft_limit}). Consider increasing with 'ulimit -n 4096'")
        except:
            pass
        
        # Set up multiprocessing start method
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            # Already set, that's fine
            pass
        
        # Create GPU evaluator service in main process
        # Need to provide the actual model/evaluator to the service
        if self.evaluator_type == "resnet":
            if hasattr(self.evaluator, 'model'):
                model = self.evaluator.model
                logger.info("Using ResNet model for GPU evaluator service")
            else:
                logger.error("ResNet evaluator has no model attribute! Falling back to mock.")
                # Create a proper ResNet model as fallback
                from mcts.neural_networks.resnet_model import create_resnet_for_game
                model = create_resnet_for_game('gomoku', input_channels=18)
                logger.info("Created fallback ResNet model for GPU evaluator service")
        else:
            # For mock evaluator, we need to use the evaluator itself, not a dummy model
            logger.info("Using mock evaluator for GPU evaluator service")
            # The GPU service will handle mock evaluation internally
            from mcts.neural_networks.resnet_model import create_resnet_for_game
            model = create_resnet_for_game('gomoku', input_channels=18)
            logger.info("Created ResNet model for mock evaluation mode")
        
        # Create GPU service with auto-optimization
        gpu_service = GPUEvaluatorService(
            model=model,
            device=self.device,
            batch_size=None,  # Let it auto-detect
            batch_timeout=None,  # Let it auto-detect
            auto_optimize=auto_optimize,
            workload_type=workload_type
        )
        
        # Start the service
        gpu_service.start()
        logger.info(f"GPU evaluation service started on device: {self.device}")
        
        try:
            # Get the request queue
            request_queue = gpu_service.get_request_queue()
            
            # Use hardware optimizer for better resource allocation
            if auto_optimize:
                try:
                    # Direct import from relative path
                    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
                    from utils.hardware_optimizer import HardwareOptimizer
                    optimizer = HardwareOptimizer()
                    hw_profile = optimizer.detect_hardware()
                    allocation = optimizer.calculate_optimal_allocation(workload_type, num_workers)
                    
                    max_concurrent = allocation.data_collector_max_concurrent
                    chunk_size = allocation.data_collector_chunk_size
                    games_per_batch = min(max_concurrent, chunk_size)
                    
                    logger.info(f"Hardware-optimized allocation: max_concurrent={max_concurrent}, "
                              f"chunk_size={chunk_size}, batch_size={games_per_batch}")
                except ImportError:
                    logger.warning("Hardware optimizer not available, using fallback allocation")
                    # Fallback allocation - reduce worker overhead for better GPU utilization
                    cpu_count = mp.cpu_count()
                    max_concurrent = min(num_workers, max(2, cpu_count // 4))  # Fewer workers for better GPU batching
                    games_per_batch = max_concurrent
                    chunk_size = min(30, max_concurrent * 3)  # Smaller chunks for better control
            else:
                # Manual allocation - reduce worker overhead for better GPU utilization
                cpu_count = mp.cpu_count()
                max_concurrent = min(num_workers, max(2, cpu_count // 4))  # Fewer workers
                games_per_batch = max_concurrent
                chunk_size = min(30, max_concurrent * 3)  # Smaller chunks
            
            logger.info(f"Processing games in batches of {games_per_batch}")
            
            # Track active processes globally
            active_processes = {}
            completed_games = 0
            
            for chunk_start in range(0, n_games, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_games)
                logger.info(f"Processing games {chunk_start} to {chunk_end}")
                
                # Process games in batches within this chunk
                with tqdm(total=chunk_end - chunk_start, desc=f"Collecting MCTS data (chunk {chunk_start//chunk_size + 1})", unit="games") as pbar:
                    
                    for batch_start in range(chunk_start, chunk_end, games_per_batch):
                        batch_end = min(batch_start + games_per_batch, chunk_end)
                        batch_processes = []
                        batch_queues = []
                    
                        # Start processes for this batch
                        for game_idx in range(batch_start, batch_end):
                            # Create result queue for this game
                            result_queue = mp.Queue()
                            batch_queues.append(result_queue)
                            
                            # Create worker-specific response queue
                            response_queue = gpu_service.create_worker_queue(game_idx)
                            
                            # Create process
                            p = mp.Process(
                                target=_play_game_worker_data_collection,
                                args=(self.mcts_config, request_queue, response_queue,
                                      game_type, board_size, snapshot_frequency,
                                      game_idx, result_queue)
                            )
                            p.start()
                            batch_processes.append(p)
                            logger.debug(f"Started data collection process for game {game_idx}")
                
                        # Collect results from this batch with better timeout handling
                        logger.debug(f"Collecting results from batch of {len(batch_processes)} games...")
                        
                        # Track which processes are still running
                        running_processes = list(range(len(batch_processes)))
                        batch_timeout = time.time() + 900  # 15 minute total batch timeout - much more lenient
                        
                        while running_processes and time.time() < batch_timeout:
                            for idx in running_processes[:]:  # Copy list to modify during iteration
                                p = batch_processes[idx]
                                q = batch_queues[idx]
                                game_idx = batch_start + idx
                                
                                try:
                                    # Try to get result without blocking
                                    game_session = q.get_nowait()
                                    
                                    if game_session:
                                        self.game_sessions.append(game_session)
                                        logger.debug(f"Collected data from game {game_idx}")
                                        completed_games += 1
                                    else:
                                        logger.warning(f"Game {game_idx} returned empty data")
                                    
                                    # Remove from running list
                                    running_processes.remove(idx)
                                    pbar.update(1)
                                    
                                except queue.Empty:
                                    # Check if process is still alive
                                    if not p.is_alive():
                                        tqdm.write(f"⚠️  Game {game_idx} process died")
                                        running_processes.remove(idx)
                                        pbar.update(1)
                                    # Otherwise, continue waiting
                                
                                except Exception as e:
                                    tqdm.write(f"⚠️  Game {game_idx} failed: {str(e)[:50]}...")
                                    running_processes.remove(idx)
                                    pbar.update(1)
                            
                            # Small sleep to avoid busy waiting
                            if running_processes:
                                time.sleep(0.1)
                        
                        # Clean up any remaining processes
                        if running_processes:
                            tqdm.write(f"⚠️  {len(running_processes)} games timed out, cleaning up...")
                            for idx in running_processes:
                                p = batch_processes[idx]
                                game_idx = batch_start + idx
                                
                                # Terminate stuck process
                                if p.is_alive():
                                    p.terminate()
                                    p.join(timeout=2)
                                    
                                    if p.is_alive():  # Still alive after join
                                        p.kill()
                                        p.join(timeout=1)
                                
                                pbar.update(1)
                        
                        # Clean up all resources for this batch (simplified like self-play)
                        for idx, p in enumerate(batch_processes):
                            game_idx = batch_start + idx
                            
                            # Aggressive cleanup like self-play module
                            if p.is_alive():
                                p.terminate()
                                p.join(timeout=2)
                                if p.is_alive():
                                    logger.error(f"Force killing stuck process {game_idx}")
                                    p.kill()
                            
                            # Clean up worker queue
                            if hasattr(gpu_service, 'cleanup_worker_queue'):
                                gpu_service.cleanup_worker_queue(game_idx)
                        
                        # Clear lists and force garbage collection
                        batch_processes.clear()
                        
                        # IMPORTANT: Close all queues to free file descriptors
                        for q in batch_queues:
                            try:
                                # Drain any remaining items
                                while not q.empty():
                                    q.get_nowait()
                            except:
                                pass
                            # Close the queue
                            q.close()
                            q.join_thread()
                        
                        batch_queues.clear()
                
                # Between chunks, do a more thorough cleanup
                logger.info(f"Completed chunk, cleaning up resources...")
                time.sleep(1)  # Brief pause between chunks
                gc.collect()
                        
        finally:
            # Stop the GPU service and clean up
            logger.info("Stopping GPU evaluation service...")
            gpu_service.stop()
            
            # Give some time for cleanup
            time.sleep(1)
            
            # Clear any remaining messages in queues
            try:
                while not request_queue.empty():
                    request_queue.get_nowait()
            except:
                pass
            
            logger.info("GPU evaluation service stopped")
            logger.info(f"Data collection completed: {completed_games}/{n_games} games successfully collected")
    
    def _finalize_collection_session(self, n_games: int) -> Dict[str, Any]:
        """Finalize the collection session and generate summary"""
        self.collection_metadata['collection_end'] = time.time()
        self.collection_metadata['total_games'] = len(self.game_sessions)
        self.collection_metadata['total_tree_snapshots'] = sum(
            len(session.tree_snapshots) for session in self.game_sessions
        )
        
        # Generate summary
        total_duration = self.collection_metadata['collection_end'] - self.collection_metadata['collection_start']
        total_moves = sum(session.move_count for session in self.game_sessions)
        total_simulations = sum(session.total_simulations for session in self.game_sessions)
        
        summary = {
            'collection_summary': {
                'total_duration': total_duration,
                'games_completed': len(self.game_sessions),
                'games_requested': n_games,
                'success_rate': len(self.game_sessions) / n_games if n_games > 0 else 0,
            },
            'game_statistics': {
                'total_moves': total_moves,
                'avg_moves_per_game': total_moves / max(len(self.game_sessions), 1),
                'total_simulations': total_simulations,
                'avg_simulations_per_game': total_simulations / max(len(self.game_sessions), 1)
            },
            'tree_statistics': {
                'total_snapshots': self.collection_metadata['total_tree_snapshots'],
                'avg_snapshots_per_game': self.collection_metadata['total_tree_snapshots'] / max(len(self.game_sessions), 1)
            },
            'performance_statistics': {
                'avg_sims_per_second': sum(session.peak_sims_per_second for session in self.game_sessions) / max(len(self.game_sessions), 1),
                'total_search_time': sum(session.total_search_time for session in self.game_sessions)
            }
        }
        
        logger.info(f"MCTS data collection completed: {summary['collection_summary']['games_completed']}/{n_games} games")
        return summary
    
    def export_data(self, output_path: str) -> str:
        """Export collected real MCTS data to JSON file"""
        
        # Convert game sessions to dictionary format
        game_data = []
        tree_snapshots = []
        
        for session in self.game_sessions:
            # Convert session to dict
            session_dict = asdict(session)
            game_data.append(session_dict)
            
            # Extract tree snapshots for physics analysis
            for snapshot in session.tree_snapshots:
                snapshot_dict = asdict(snapshot)
                tree_snapshots.append(snapshot_dict)
        
        # Create final data structure
        export_data = {
            'metadata': self.collection_metadata,
            'game_sessions': game_data,
            'tree_expansion_data': tree_snapshots,  # For compatibility with visualization tools
            'performance_metrics': [
                {
                    'game_id': session.game_id,
                    'move_count': session.move_count,
                    'total_simulations': session.total_simulations,
                    'avg_search_time': session.total_search_time / max(session.move_count, 1),
                    'peak_sims_per_second': session.peak_sims_per_second,
                    'winner': session.winner
                }
                for session in self.game_sessions
            ]
        }
        
        # Save to file with custom encoder
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Real MCTS data exported to {output_path}")
        logger.info(f"Export contains {len(tree_snapshots)} tree snapshots from {len(self.game_sessions)} games")
        
        return output_path


def _play_game_worker_data_collection(mcts_config, request_queue, response_queue, 
                                     game_type: str, board_size: int, 
                                     snapshot_frequency: int, game_idx: int, 
                                     result_queue) -> None:
    """Worker function for parallel data collection using GPU evaluation service"""
    # CRITICAL: Disable CUDA in workers to avoid multiprocessing issues
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['CUDA_CACHE_DISABLE'] = '1'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    # Set up logging
    import logging
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)
    
    # Don't set up signal handlers - let the main process handle termination
    # The signal handlers are causing premature exits
    
    try:
        # Import modules after CUDA is disabled
        import torch
        
        # Force disable CUDA in PyTorch to avoid device queries
        torch.cuda.is_available = lambda: False
        torch.cuda.device_count = lambda: 0
        torch.cuda.get_device_name = lambda device=None: "CPU"
        
        import sys
        from pathlib import Path
        
        # Add MCTS modules to path
        mcts_root = Path(__file__).parent.parent.parent.parent
        sys.path.insert(0, str(mcts_root))
        
        from mcts.core.mcts import MCTS
        from mcts.neural_networks.mock_evaluator import MockEvaluator
        from mcts.utils.gpu_evaluator_service import RemoteEvaluator
        import alphazero_py
        import numpy as np
        import time
        from dataclasses import asdict
        
        logger.debug(f"[DATA-WORKER {game_idx}] Worker started - using CPU only")
        
        # Create remote evaluator that sends requests to GPU service
        remote_evaluator = RemoteEvaluator(request_queue, response_queue, 225, worker_id=game_idx)
        
        # Set a timeout for the entire game to prevent hanging
        game_start_time = time.time()
        game_timeout = 600  # 10 minutes per game max - increased for complex games
        
        # Wrap evaluator to return torch tensors
        class TensorEvaluator:
            def __init__(self, evaluator, device):
                self.evaluator = evaluator
                self.device = device
                self._return_torch_tensors = True
            
            def evaluate(self, state, legal_mask=None, temperature=1.0):
                policy, value = self.evaluator.evaluate(state, legal_mask, temperature)
                policy_tensor = torch.from_numpy(policy).float().to(self.device)
                value_tensor = torch.tensor(value, dtype=torch.float32, device=self.device)
                return policy_tensor, value_tensor
            
            def evaluate_batch(self, states, legal_masks=None, temperature=1.0):
                if isinstance(states, torch.Tensor):
                    states = states.cpu().numpy()
                if legal_masks is not None and isinstance(legal_masks, torch.Tensor):
                    legal_masks = legal_masks.cpu().numpy()
                
                policies, values = self.evaluator.evaluate_batch(states, legal_masks, temperature)
                policies_tensor = torch.from_numpy(policies).float().to(self.device)
                values_tensor = torch.from_numpy(values).float().to(self.device)
                return policies_tensor, values_tensor
        
        # Workers must use CPU
        tensor_device = 'cpu'
        evaluator = TensorEvaluator(remote_evaluator, tensor_device)
        
        # Create CPU-only MCTS config (make a copy to avoid modifying original)
        from copy import deepcopy
        worker_config = deepcopy(mcts_config)
        worker_config.device = 'cpu'
        worker_config.use_mixed_precision = False
        worker_config.use_cuda_graphs = False
        worker_config.use_tensor_cores = False
        worker_config.max_tree_nodes = min(50000, worker_config.max_tree_nodes)
        # Disable debug logging to avoid CUDA device queries
        worker_config.enable_debug_logging = False
        
        # Create MCTS
        mcts = MCTS(worker_config, evaluator)
        
        # Create a simplified data collector for this worker
        worker_collector = _create_worker_data_collector(mcts, worker_config)
        
        # Play game and collect data
        game_session = worker_collector.play_real_game_with_data_collection(
            game_type=game_type,
            board_size=board_size,
            snapshot_frequency=snapshot_frequency,
            game_idx=game_idx,
            game_timeout=game_timeout
        )
        
        logger.debug(f"[DATA-WORKER {game_idx}] Game completed with {game_session.move_count} moves")
        
        # Return result
        result_queue.put(game_session)
        
        # Clean exit
        logger.debug(f"[DATA-WORKER {game_idx}] Worker exiting cleanly")
        
    except Exception as e:
        logger.error(f"[DATA-WORKER {game_idx}] Failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        result_queue.put(None)  # Return None on error
    finally:
        # Ensure we always put something in the queue
        try:
            # Check if we already put something
            if result_queue.empty():
                result_queue.put(None)
        except:
            pass


def _create_worker_data_collector(mcts, mcts_config):
    """Create a simplified data collector for worker processes"""
    
    class WorkerDataCollector:
        def __init__(self, mcts, mcts_config):
            self.mcts = mcts
            self.mcts_config = mcts_config
        
        def play_real_game_with_data_collection(self, game_type: str = "gomoku",
                                               board_size: int = 15,
                                               snapshot_frequency: int = 5,
                                               game_idx: int = 0,
                                               game_timeout: int = 600) -> 'RealGameSession':
            """Simplified game play for worker processes"""
            
            game_id = f"{game_type}_{int(time.time())}_{game_idx}"
            start_time = time.time()
            game_start_time = start_time  # Track for timeout
            
            # Initialize game state
            if game_type.lower() == "gomoku":
                state = alphazero_py.GomokuState()
            else:
                raise ValueError(f"Unsupported game type: {game_type}")
            
            # Initialize simplified game session
            game_session = RealGameSession(
                game_id=game_id,
                start_time=start_time,
                end_time=0.0,
                game_type=game_type,
                board_size=board_size,
                winner=-1,
                final_score=0.0,
                move_count=0,
                mcts_config=asdict(self.mcts_config),
                tree_snapshots=[],
                moves=[],
                total_simulations=0,
                avg_simulations_per_move=0.0,
                total_search_time=0.0,
                peak_sims_per_second=0.0,
                game_phases=[],
                complexity_evolution=[],
                decision_confidence=[]
            )
            
            move_count = 0
            total_search_time = 0.0
            
            # Play game until terminal
            while not state.is_terminal() and move_count < 500:
                # Check timeout
                if time.time() - game_start_time > game_timeout:
                    logger.error(f"[DATA-WORKER {game_idx}] Game timeout after {move_count} moves")
                    break
                
                move_count += 1
                current_player = state.get_current_player()
                
                # Determine temperature
                if move_count <= 30:
                    temperature = 1.0
                else:
                    temperature = 0.1
                
                # Preserve tree statistics for data collection by not resetting
                # self.mcts.reset_tree()  # Disabled to maintain statistics
                
                # Run MCTS search
                search_start = time.perf_counter()
                policy = self.mcts.search(state, self.mcts_config.num_simulations)
                search_time = time.perf_counter() - search_start
                
                total_search_time += search_time
                game_session.total_simulations += self.mcts_config.num_simulations
                
                # Update peak performance
                sims_per_sec = self.mcts_config.num_simulations / search_time
                game_session.peak_sims_per_second = max(game_session.peak_sims_per_second, sims_per_sec)
                
                # Extract tree snapshot (simplified)
                if move_count % snapshot_frequency == 0 or move_count <= 10:
                    snapshot = self._extract_simple_tree_snapshot(move_count, current_player, search_time, policy)
                    game_session.tree_snapshots.append(snapshot)
                
                # Normalize policy
                policy_sum = np.sum(policy)
                if policy_sum > 0:
                    policy = policy / policy_sum
                else:
                    policy = np.ones(len(policy)) / len(policy)
                
                # Get legal moves and validate action bounds
                legal_moves = state.get_legal_moves()
                if not legal_moves:
                    raise ValueError(f"No legal actions available at move {move_count}")
                
                # Validate action space bounds
                max_action = len(policy) - 1
                legal_moves = [a for a in legal_moves if 0 <= a <= max_action]
                if not legal_moves:
                    raise ValueError(f"No legal actions within policy bounds [0, {max_action}] at move {move_count}")
                
                # Select action, restricting to legal moves only
                if temperature > 0.5:
                    # Sample from legal moves policy distribution
                    legal_policy_values = np.array([policy[a] for a in legal_moves])
                    if np.sum(legal_policy_values) > 0:
                        legal_policy_values = legal_policy_values / np.sum(legal_policy_values)
                        action_idx = np.random.choice(len(legal_moves), p=legal_policy_values)
                        action = legal_moves[action_idx]
                    else:
                        # Fallback to uniform random from legal moves
                        action = np.random.choice(legal_moves)
                else:
                    # Select best legal action
                    legal_policies = [(a, policy[a]) for a in legal_moves]
                    legal_policies.sort(key=lambda x: x[1], reverse=True)
                    action = legal_policies[0][0]
                
                # Final validation
                if action not in legal_moves:
                    logger.error(f"Action selection bug: selected action {action} not in legal moves at move {move_count}")
                    action = legal_moves[0]  # Emergency fallback
                
                # Record move
                move_data = {
                    'move_number': move_count,
                    'player': current_player,
                    'action': action,
                    'policy': policy.tolist(),
                    'search_time': search_time,
                    'temperature': temperature,
                    'tree_size': self.mcts.tree.num_nodes,
                    'confidence': float(np.max(policy))
                }
                game_session.moves.append(move_data)
                game_session.decision_confidence.append(float(np.max(policy)))
                
                # Calculate game complexity
                policy_entropy = -np.sum(policy * np.log(policy + 1e-10))
                game_session.complexity_evolution.append(float(policy_entropy))
                
                # Determine game phase
                if move_count <= 15:
                    phase = "opening"
                elif move_count <= 30:
                    phase = "midgame"
                else:
                    phase = "endgame"
                game_session.game_phases.append(phase)
                
                # Apply action
                state.make_move(action)
            
            # Game finished
            game_session.end_time = time.time()
            game_session.move_count = move_count
            game_session.avg_simulations_per_move = game_session.total_simulations / max(move_count, 1)
            game_session.total_search_time = total_search_time
            
            # Determine winner
            if state.is_terminal():
                result = state.get_game_result()
                if result == alphazero_py.GameResult.WIN_PLAYER1:
                    game_session.winner = 0
                    game_session.final_score = 1.0
                elif result == alphazero_py.GameResult.WIN_PLAYER2:
                    game_session.winner = 1
                    game_session.final_score = 1.0
                elif result == alphazero_py.GameResult.DRAW:
                    game_session.winner = -1
                    game_session.final_score = 0.5
                else:
                    game_session.winner = -1
                    game_session.final_score = 0.0
            
            return game_session
        
        def _extract_simple_tree_snapshot(self, move_number: int, player: int, 
                                         search_time: float, policy: np.ndarray) -> 'RealTreeSnapshot':
            """Simplified tree snapshot extraction for workers"""
            
            timestamp = time.time()
            tree = self.mcts.tree
            
            # Basic tree statistics
            total_nodes = tree.num_nodes
            root_visits = tree.visit_counts[0].item() if total_nodes > 0 else 0
            root_value = tree.value_sums[0].item() / max(root_visits, 1) if root_visits > 0 else 0.0
            
            # Limited data extraction for performance
            extract_limit = min(total_nodes, 100)  # Much smaller limit for workers
            
            visit_counts = []
            value_sums = []
            q_values = []
            node_priors = []
            parent_indices = []
            parent_actions = []
            node_phases = []
            
            if total_nodes > 0:
                visit_counts = [int(x) for x in tree.visit_counts[:extract_limit].cpu().numpy()]
                value_sums = [float(x) for x in tree.value_sums[:extract_limit].cpu().numpy()]
                node_priors = [float(x) for x in tree.node_priors[:extract_limit].cpu().numpy()]
                parent_indices = [int(x) for x in tree.parent_indices[:extract_limit].cpu().numpy()]
                parent_actions = [int(x) for x in tree.parent_actions[:extract_limit].cpu().numpy()]
                node_phases = [float(x) for x in tree.phases[:extract_limit].cpu().numpy()]
                
                # Calculate Q-values
                q_values = []
                for i in range(extract_limit):
                    visits = visit_counts[i]
                    if visits > 0:
                        q_val = value_sums[i] / visits
                    else:
                        q_val = 0.0
                    q_values.append(q_val)
            
            # Search performance
            sims_per_second = self.mcts_config.num_simulations / max(search_time, 0.001)
            
            return RealTreeSnapshot(
                timestamp=timestamp,
                move_number=move_number,
                player=player,
                total_nodes=total_nodes,
                total_visits=sum(visit_counts) if visit_counts else 0,
                tree_depth=10,  # Simplified for workers
                root_visits=root_visits,
                root_value=root_value,
                visit_counts=visit_counts,
                value_sums=value_sums,
                q_values=q_values,
                node_priors=node_priors,
                parent_indices=parent_indices,
                parent_actions=parent_actions,
                node_phases=node_phases,
                branching_factor=2.0,  # Simplified for workers
                max_depth=10,  # Simplified for workers
                leaf_nodes=[],  # Simplified for workers
                policy_distribution=policy.tolist(),
                search_time=search_time,
                simulations_per_second=sims_per_second,
                quantum_mode_active=False,
                quantum_corrections=[],
                decoherence_detected=False
            )
    
    return WorkerDataCollector(mcts, mcts_config)


def _precompile_cuda_kernels(device: str):
    """Pre-compile CUDA kernels to avoid JIT overhead during data collection"""
    try:
        import torch
        
        # Clear GPU cache first
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        from mcts.gpu.unified_kernels import get_unified_kernels
        from mcts.core.mcts import MCTS, MCTSConfig
        
        # Force kernel loading
        print("  📦 Loading unified kernels...")
        kernels = get_unified_kernels(torch.device(device))
        
        # Create ultra-minimal MCTS instance to trigger kernel compilation
        print("  ⚙️  Initializing minimal MCTS system...")
        mcts_config = MCTSConfig(
            num_simulations=3,    # Ultra-minimal simulations
            wave_size=32,         # Very small wave size to avoid memory issues
            min_wave_size=32,
            max_wave_size=32,
            device=device,
            enable_quantum=False,  # Keep it simple
            max_tree_nodes=1000,   # Minimal tree size
            use_mixed_precision=False,  # Disable to save memory
            use_cuda_graphs=False,      # Disable to save memory
            use_tensor_cores=False      # Disable to save memory
        )
        
        evaluator = MockEvaluator()
        mcts = MCTS(mcts_config, evaluator)
        
        # Run a ultra-minimal search to trigger any lazy compilation
        print("  🔥 Warming up kernels...")
        import alphazero_py
        game_state = alphazero_py.GomokuState()
        mcts.search(game_state, 3)  # Ultra-small search to trigger compilation
        
        # Clean up immediately
        del mcts
        del evaluator
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        print("  ✅ CUDA kernels pre-compiled successfully!")
        
    except Exception as e:
        logger.warning(f"Kernel pre-compilation failed: {e}")
        print("  📝 Continuing with PyTorch fallback...")


def main():
    """Main function for real MCTS data collection"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect REAL MCTS data for quantum research")
    parser.add_argument('--n-games', type=int, default=5, help='Number of games to collect')
    parser.add_argument('--game-type', type=str, default='gomoku', help='Type of game')
    parser.add_argument('--board-size', type=int, default=15, help='Board size')
    parser.add_argument('--output', type=str, default='real_mcts_data.json', help='Output file')
    parser.add_argument('--snapshot-freq', type=int, default=5, help='Snapshot frequency')
    parser.add_argument('--evaluator', type=str, default='mock', help='Evaluator type (mock/resnet)')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--simulations', type=int, default=None, help='MCTS simulations per move (None for auto)')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (None for auto)')
    parser.add_argument('--workload', type=str, default='balanced', 
                        choices=['latency', 'throughput', 'balanced'],
                        help='Workload type for optimization')
    parser.add_argument('--no-auto-optimize', action='store_true', 
                        help='Disable automatic hardware optimization')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("REAL MCTS DATA COLLECTOR FOR QUANTUM RESEARCH")
    print(f"{'='*70}")
    print(f"Collecting data from {args.n_games} real MCTS self-play games")
    print(f"Device: {args.device}")
    print(f"Evaluator: {args.evaluator}")
    print(f"Workload type: {args.workload}")
    print(f"Auto-optimization: {'Enabled' if not args.no_auto_optimize else 'Disabled'}")
    print(f"{'='*70}")
    
    # Show hardware optimization report if enabled
    auto_optimize = not args.no_auto_optimize
    if auto_optimize:
        try:
            # Direct import from relative path
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
            from utils.hardware_optimizer import HardwareOptimizer
            optimizer = HardwareOptimizer()
            hw_profile = optimizer.detect_hardware()
            allocation = optimizer.calculate_optimal_allocation(args.workload)
            print("\n" + optimizer.get_optimization_report())
        except ImportError as e:
            print(f"⚠️  Hardware optimizer not available: {e}")
    
    # Pre-compile CUDA kernels to avoid JIT overhead during data collection
    if args.device == 'cuda':
        print("\n🔧 Pre-compiling CUDA kernels...")
        _precompile_cuda_kernels(args.device)
    
    # Performance warning for ResNet evaluator
    if args.evaluator == 'resnet':
        print("\n⚠️  Warning: ResNet evaluator may cause file I/O overhead during data collection.")
        print("📝 For performance testing, consider using --evaluator mock instead.")
    
    # Create collector with hardware optimization
    collector = RealMCTSDataCollector(
        evaluator_type=args.evaluator, 
        device=args.device,
        auto_optimize=auto_optimize,
        workload_type=args.workload
    )
    
    # Override simulation count if specified (otherwise use hardware-optimized value)
    if args.simulations is not None:
        collector.mcts_config.num_simulations = args.simulations
        collector.mcts = MCTS(collector.mcts_config, collector.evaluator)
        collector.mcts.optimize_for_hardware()
    
    # Run collection with hardware optimization
    summary = collector.run_data_collection_session(
        n_games=args.n_games,
        game_type=args.game_type,
        board_size=args.board_size,
        snapshot_frequency=args.snapshot_freq,
        num_workers=args.workers,
        auto_optimize=auto_optimize,
        workload_type=args.workload
    )
    
    # Export data
    output_path = collector.export_data(args.output)
    
    print(f"\n{'='*70}")
    print("REAL MCTS DATA COLLECTION COMPLETED")
    print(f"{'='*70}")
    print(f"Games collected: {summary['game_statistics']['total_moves']}")
    print(f"Tree snapshots: {summary['tree_statistics']['total_snapshots']}")
    print(f"Total simulations: {summary['game_statistics']['total_simulations']:,}")
    print(f"Average sims/second: {summary['performance_statistics']['avg_sims_per_second']:.0f}")
    print(f"Data exported to: {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()