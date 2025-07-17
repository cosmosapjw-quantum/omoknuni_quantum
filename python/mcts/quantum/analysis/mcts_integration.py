"""
MCTS integration for genuine self-play data generation.

Provides real MCTS self-play games for quantum phenomena analysis.
"""
import torch
import numpy as np
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# MCTS imports
try:
    from ...core.mcts import MCTS
    from ...core.mcts_config import MCTSConfig
    from ...core.game_interface import GameInterface, GameType
    from ...neural_networks.resnet_model import create_resnet_for_game
    from ...utils.single_gpu_evaluator import SingleGPUEvaluator
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
    from python.mcts.core.mcts import MCTS
    from python.mcts.core.mcts_config import MCTSConfig
    from python.mcts.core.game_interface import GameInterface, GameType
    from python.mcts.neural_networks.resnet_model import create_resnet_for_game
    from python.mcts.utils.single_gpu_evaluator import SingleGPUEvaluator

# Random evaluator import
try:
    from .random_evaluator import create_random_evaluator
except ImportError:
    from random_evaluator import create_random_evaluator

logger = logging.getLogger(__name__)


@dataclass
class MCTSGameData:
    """Data from a single MCTS self-play game"""
    trajectory: List[Dict[str, Any]]
    winner: int
    game_length: int
    total_simulations: int
    
    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Get game trajectory for analysis"""
        return self.trajectory


class RealMCTSGame:
    """
    Real MCTS self-play game generator.
    
    This replaces the mock game generator with actual MCTS self-play.
    """
    
    def __init__(self, 
                 sims_per_game: int,
                 game_type: str = 'gomoku',
                 board_size: int = 15,
                 model_path: Optional[str] = None,
                 device: str = None,
                 evaluator_type: str = 'resnet'):
        """
        Initialize MCTS game generator.
        
        Args:
            sims_per_game: Number of MCTS simulations per move
            game_type: Type of game ('gomoku', 'go', 'chess')
            board_size: Board size for board games
            model_path: Path to neural network model (optional)
            device: Device to use ('cuda' or 'cpu')
            evaluator_type: Type of evaluator ('resnet', 'random', 'fast_random')
        """
        self.sims_per_game = sims_per_game
        self.game_type = GameType(game_type)  # Convert string to enum
        self.board_size = board_size
        self.evaluator_type = evaluator_type
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Initialize MCTS components
        self._setup_mcts(model_path)
        
    def _setup_mcts(self, model_path: Optional[str]):
        """Setup MCTS with optimized configuration"""
        # Create MCTS configuration
        self.mcts_config = MCTSConfig()
        self.mcts_config.num_simulations = self.sims_per_game
        self.mcts_config.c_puct = 1.414
        self.mcts_config.device = self.device
        
        # Set CSR max actions based on game type
        if self.game_type == GameType.CHESS:
            self.mcts_config.csr_max_actions = 4096
        elif self.game_type == GameType.GO:
            self.mcts_config.csr_max_actions = 361  # 19x19
        elif self.game_type == GameType.GOMOKU:
            self.mcts_config.csr_max_actions = 225  # 15x15
        else:
            # Default to board_size squared for unknown games
            self.mcts_config.csr_max_actions = self.board_size * self.board_size
        
        # Scale memory settings based on simulation count
        # Optimized for RTX 3060 Ti with 8GB VRAM
        if self.sims_per_game >= 10000:
            # Deep/Overnight preset: 10K simulations (reduced from 25K)
            # Balanced for comprehensive physics while respecting hardware limits
            self.mcts_config.max_wave_size = 6144    # Reduced for stability
            self.mcts_config.min_wave_size = 1024
            self.mcts_config.wave_num_pipelines = 4  # Moderate parallelism
            self.mcts_config.batch_size = 768
            self.mcts_config.inference_batch_size = 768
            
            # Memory settings for 10K simulations on 8GB GPU
            self.mcts_config.memory_pool_size_mb = 4096  # 4GB pool
            self.mcts_config.max_tree_nodes = 8000000    # 8M nodes
            self.mcts_config.initial_capacity_factor = 0.1  # Start small
            
            # Conservative tree expansion for 10K sims
            self.mcts_config.initial_children_per_expansion = 3
            self.mcts_config.max_children_per_node = 20
            
        elif self.sims_per_game >= 7500:
            # Comprehensive preset: 7.5K simulations (sweet spot for physics)
            # Captures all major phase transitions with good performance
            self.mcts_config.max_wave_size = 7936    # Optimal for GPU
            self.mcts_config.min_wave_size = 1024
            self.mcts_config.wave_num_pipelines = 5
            self.mcts_config.wave_adaptive_sizing = True
            self.mcts_config.batch_size = 896
            self.mcts_config.inference_batch_size = 896
            
            # Memory for 7.5K simulations
            self.mcts_config.memory_pool_size_mb = 3584  # 3.5GB
            self.mcts_config.max_tree_nodes = 6000000    # 6M nodes
            self.mcts_config.initial_capacity_factor = 0.1
            
            # Moderate tree expansion
            self.mcts_config.initial_children_per_expansion = 4
            self.mcts_config.max_children_per_node = 25
            
        elif self.sims_per_game >= 5000:
            # Standard preset: 5K simulations (minimum for good physics)
            # Peak performance while capturing decoherence
            self.mcts_config.max_wave_size = 7936    # Peak efficiency
            self.mcts_config.min_wave_size = 1024
            self.mcts_config.wave_num_pipelines = 5
            self.mcts_config.wave_adaptive_sizing = True
            self.mcts_config.batch_size = 1024
            self.mcts_config.inference_batch_size = 1024
            
            # Memory for 5K simulations
            self.mcts_config.memory_pool_size_mb = 3072  # 3GB
            self.mcts_config.max_tree_nodes = 4000000  # 4M nodes
            self.mcts_config.initial_capacity_factor = 0.1
            
            # Balanced tree expansion
            self.mcts_config.initial_children_per_expansion = 5
            self.mcts_config.max_children_per_node = 30
            
        elif self.sims_per_game >= 2500:
            # Quick preset: 2.5K simulations (fast exploration)
            # Captures early decoherence with fast runtime
            self.mcts_config.max_wave_size = 4096    # Smaller waves for quick runs
            self.mcts_config.min_wave_size = 512
            self.mcts_config.wave_num_pipelines = 4
            self.mcts_config.wave_adaptive_sizing = True
            self.mcts_config.batch_size = 512
            self.mcts_config.inference_batch_size = 512
            
            # Memory for 2.5K simulations
            self.mcts_config.memory_pool_size_mb = 2048  # 2GB
            self.mcts_config.max_tree_nodes = 2000000    # 2M nodes
            self.mcts_config.initial_capacity_factor = 0.1
            
            # Aggressive tree expansion for exploration
            self.mcts_config.initial_children_per_expansion = 8
            self.mcts_config.max_children_per_node = 50
            
        else:
            # Ultra-quick preset: 1K simulations (minimal viable)
            # For rapid testing and high-temperature physics only
            self.mcts_config.max_wave_size = 2048    # Small waves
            self.mcts_config.min_wave_size = 256
            self.mcts_config.wave_num_pipelines = 3
            self.mcts_config.wave_adaptive_sizing = True
            self.mcts_config.batch_size = 256
            self.mcts_config.inference_batch_size = 256
            
            # Minimal memory for 1K simulations
            self.mcts_config.memory_pool_size_mb = 1024  # 1GB
            self.mcts_config.max_tree_nodes = 1000000    # 1M nodes
            self.mcts_config.initial_capacity_factor = 0.2  # Start larger
            
            # Wide tree for exploration phase
            self.mcts_config.initial_children_per_expansion = 10
            self.mcts_config.max_children_per_node = 60
            
        # GPU settings (from benchmark - disabling some features improved performance)
        if hasattr(self.mcts_config, 'use_mixed_precision'):
            self.mcts_config.use_mixed_precision = False  # Disabled for stability
        if hasattr(self.mcts_config, 'use_cuda_graphs'):
            self.mcts_config.use_cuda_graphs = False  # Disabled for flexibility
        if hasattr(self.mcts_config, 'enable_kernel_fusion'):
            self.mcts_config.enable_kernel_fusion = False  # Not beneficial
        
        # Timeouts
        if hasattr(self.mcts_config, 'gpu_batch_timeout'):
            self.mcts_config.gpu_batch_timeout = 0.034  # Optimized timeout
        
        # Standard settings (from benchmark)
        self.mcts_config.enable_virtual_loss = True
        self.mcts_config.virtual_loss = 1.0
        self.mcts_config.enable_fast_ucb = True
        # Disable subtree reuse for high simulation counts to save memory
        if self.sims_per_game >= 10000:
            self.mcts_config.enable_subtree_reuse = False  # Disabled to save memory
        else:
            self.mcts_config.enable_subtree_reuse = True  # Enabled for performance
        self.mcts_config.classical_only_mode = True    # Skip quantum features
        
        # Enable tree pruning for memory management - more aggressive for higher sim counts
        if hasattr(self.mcts_config, 'enable_tree_pruning'):
            self.mcts_config.enable_tree_pruning = True
        if hasattr(self.mcts_config, 'tree_pruning_threshold'):
            if self.sims_per_game >= 10000:
                self.mcts_config.tree_pruning_threshold = 0.6  # Prune when 60% full (very aggressive)
            elif self.sims_per_game >= 5000:
                self.mcts_config.tree_pruning_threshold = 0.75  # Prune when 75% full (aggressive)
            else:
                self.mcts_config.tree_pruning_threshold = 0.85  # Prune when 85% full (normal)
        if hasattr(self.mcts_config, 'pruning_frequency'):
            if self.sims_per_game >= 10000:
                self.mcts_config.pruning_frequency = 100  # Prune every 100 simulations (very frequent)
            elif self.sims_per_game >= 5000:
                self.mcts_config.pruning_frequency = 150  # Prune every 150 simulations (frequent)
            else:
                self.mcts_config.pruning_frequency = 200  # Prune every 200 simulations (normal)
        
        # Create game interface
        self.game_interface = GameInterface(
            self.game_type, 
            board_size=self.board_size, 
            input_representation='basic'
        )
        
        # Create evaluator based on type
        if self.evaluator_type in ['random', 'fast_random']:
            # Create random evaluator
            logger.info(f"Creating {self.evaluator_type} evaluator")
            action_space_size = self.board_size * self.board_size  # For board games
            self.evaluator = create_random_evaluator(
                evaluator_type=self.evaluator_type,
                action_space_size=action_space_size,
                device=self.device
            )
            self.model = None  # No model needed for random evaluator
        else:
            # Create or load neural network model for ResNet evaluator
            if model_path:
                # Load existing model
                logger.info(f"Loading ResNet model from {model_path}")
                self.model = torch.load(model_path)
            else:
                # Create default model
                logger.info("Creating default ResNet neural network model")
                self.model = create_resnet_for_game(
                    game_type=self.game_type.name.lower(),
                    input_channels=18,  # Basic representation
                    num_blocks=10,
                    num_filters=128
                )
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Create ResNet evaluator
            self.evaluator = SingleGPUEvaluator(self.model, device=self.device)
        
        # Log memory configuration for debugging
        logger.info(f"MCTS Memory Configuration for {self.sims_per_game} simulations:")
        logger.info(f"  Memory pool size: {self.mcts_config.memory_pool_size_mb} MB")
        logger.info(f"  Max tree nodes: {self.mcts_config.max_tree_nodes:,}")
        logger.info(f"  Initial capacity factor: {self.mcts_config.initial_capacity_factor}")
        logger.info(f"  Max wave size: {self.mcts_config.max_wave_size}")
        logger.info(f"  Children per expansion: {self.mcts_config.initial_children_per_expansion}")
        
        # Clear GPU cache before MCTS creation to ensure maximum available memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Create MCTS instance
        self.mcts = MCTS(self.mcts_config, self.evaluator, self.game_interface)
        
    def generate_game(self, temperature_threshold: int = 30) -> MCTSGameData:
        """
        Generate a single self-play game.
        
        Args:
            temperature_threshold: Move number after which to play deterministically
            
        Returns:
            MCTSGameData with game trajectory and MCTS tree references
        """
        # Initialize game state
        state = self.game_interface.create_initial_state()
        trajectory = []
        move_count = 0
        total_simulations = 0
        
        # Play until terminal
        while not self.game_interface.is_terminal(state):
            move_count += 1
            
            # Run MCTS search with tree reset for fresh search
            self.mcts.reset_tree()
            
            # Skip memory cleanup during gameplay - it's too expensive!
            # torch.cuda.empty_cache() can take 100-200ms!
            
            # However, for physics analysis with high simulation counts, 
            # we need to be more careful about memory management
            if self.sims_per_game >= 5000 and move_count % 10 == 0:
                # Every 10 moves, check if we're getting close to capacity
                if hasattr(self.mcts, 'tree') and hasattr(self.mcts.tree, 'num_nodes'):
                    capacity_used = self.mcts.tree.num_nodes / self.mcts_config.max_tree_nodes
                    if capacity_used > 0.9:
                        logger.warning(f"Tree capacity at {capacity_used:.1%} after {move_count} moves")
                        # Force aggressive pruning if available
                        if hasattr(self.mcts, 'prune_tree'):
                            self.mcts.prune_tree(threshold=0.5)  # Prune half the tree
                
            policy = self.mcts.search(state, num_simulations=self.mcts_config.num_simulations)
            
            # Get legal moves and create masked policy
            legal_moves = self.game_interface.get_legal_moves(state)
            masked_policy = np.zeros(len(policy))
            for move in legal_moves:
                masked_policy[move] = policy[move]
            
            # Renormalize
            if masked_policy.sum() > 0:
                masked_policy = masked_policy / masked_policy.sum()
            else:
                for move in legal_moves:
                    masked_policy[move] = 1.0 / len(legal_moves)
            
            # Determine temperature
            temperature = 1.0 if move_count <= temperature_threshold else 0.0
            
            # Select action
            if temperature > 0 and len(legal_moves) > 1:
                action = np.random.choice(len(masked_policy), p=masked_policy)
            else:
                legal_values = [(move, masked_policy[move]) for move in legal_moves]
                action = max(legal_values, key=lambda x: x[1])[0]
            
            # Extract tree data immediately after search
            # Since MCTS doesn't expose tree internals, we'll extract what we can
            tree_stats = self.mcts.get_statistics() if hasattr(self.mcts, 'get_statistics') else {}
            
            # Create trajectory entry with all needed data
            trajectory_entry = {
                'position_id': move_count,
                'move_number': move_count,
                'state': state.clone() if hasattr(state, 'clone') else state,
                'policy': masked_policy.tolist(),
                'action': action,
                'temperature': temperature,
                'player': self.game_interface.get_current_player(state),
                'legal_moves': legal_moves,
                # Add placeholder data that dynamics_extractor expects
                'q_values': masked_policy.tolist(),  # Use policy as proxy for Q-values
                'visits': [int(p * self.mcts_config.num_simulations) for p in masked_policy],
                'total_visits': self.mcts_config.num_simulations,
                'depth': tree_stats.get('max_depth', 0),
                'timestamp': move_count,
                'evaluator_type': self.evaluator_type  # Add evaluator type for physics analysis
            }
            
            trajectory.append(trajectory_entry)
            total_simulations += self.mcts_config.num_simulations
            
            # Apply move
            state = self.game_interface.apply_move(state, action)
        
        # Game ended - get winner
        winner = self.game_interface.get_winner(state)
        
        return MCTSGameData(
            trajectory=trajectory,
            winner=winner,
            game_length=move_count,
            total_simulations=total_simulations
        )
    
    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Generate game and return trajectory (for compatibility)"""
        game_data = self.generate_game()
        return game_data.get_trajectory()


def create_mcts_game_generator(sims_per_game: int, 
                             game_type: str = 'gomoku',
                             board_size: int = 15,
                             model_path: Optional[str] = None,
                             evaluator_type: str = 'resnet') -> RealMCTSGame:
    """
    Factory function to create MCTS game generator.
    
    Args:
        sims_per_game: Number of simulations per move
        game_type: Game type string ('gomoku', 'go', 'chess')
        board_size: Board size
        model_path: Optional path to trained model
        evaluator_type: Type of evaluator ('resnet', 'random', 'fast_random')
        
    Returns:
        RealMCTSGame instance
    """
    return RealMCTSGame(
        sims_per_game=sims_per_game,
        game_type=game_type,  # Pass string directly, constructor handles conversion
        board_size=board_size,
        model_path=model_path,
        evaluator_type=evaluator_type
    )