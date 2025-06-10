"""Comprehensive configuration system for AlphaZero training pipeline

This module provides YAML-based configuration for all components including
MCTS, neural networks, training, arena, and quantum features.
"""

import os
import yaml
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union
from enum import Enum

logger = logging.getLogger(__name__)


class QuantumLevel(Enum):
    """Quantum computation level for MCTS"""
    CLASSICAL = "classical"      # No quantum features
    TREE_LEVEL = "tree_level"    # Tree-level quantum corrections
    ONE_LOOP = "one_loop"        # One-loop quantum corrections


@dataclass
class MCTSFullConfig:
    """Complete MCTS configuration with all parameters exposed"""
    # Core MCTS parameters
    num_simulations: int = 800
    c_puct: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_threshold: int = 30
    temperature_final: float = 0.1
    
    # Performance parameters
    min_wave_size: int = 256
    max_wave_size: int = 3072
    adaptive_wave_sizing: bool = False
    batch_size: int = 256
    virtual_loss: float = 3.0
    
    # Memory and optimization
    memory_pool_size_mb: int = 2048
    max_tree_nodes: int = 500000
    tree_reuse: bool = True
    tree_reuse_fraction: float = 0.5
    
    # GPU optimization
    use_cuda_graphs: bool = True
    use_mixed_precision: bool = True
    use_tensor_cores: bool = True
    compile_mode: str = "reduce-overhead"  # torch.compile mode
    
    # Quantum features
    quantum_level: QuantumLevel = QuantumLevel.CLASSICAL
    enable_quantum: bool = False
    
    # Quantum physics parameters
    quantum_coupling: float = 0.1
    quantum_temperature: float = 1.0
    decoherence_rate: float = 0.01
    measurement_noise: float = 0.0
    
    # Path integral parameters
    path_integral_steps: int = 10
    path_integral_beta: float = 1.0
    use_wick_rotation: bool = True
    
    # Interference parameters
    interference_alpha: float = 0.05
    interference_method: str = "minhash"  # minhash, phase_kick, cosine
    minhash_size: int = 64
    phase_kick_strength: float = 0.1
    
    # Wave MCTS specific
    wave_min_size: int = 256
    wave_max_size: int = 2048
    wave_adaptive_sizing: bool = True
    wave_target_sims_per_second: int = 100000
    wave_target_gpu_utilization: float = 0.95
    wave_num_pipelines: int = 3
    wave_async_expansion: bool = True
    wave_prefetch_evaluations: bool = True
    wave_memory_pool_mb: int = 1024
    
    # CSR Tree parameters
    csr_max_actions: int = 10
    csr_use_sparse_operations: bool = True
    
    # Interference thresholds (for quantum kernels)
    interference_threshold: float = 0.1
    constructive_interference_factor: float = 0.1
    destructive_interference_factor: float = 0.05
    
    # Device configuration
    device: str = "cuda"
    num_threads: int = 4
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling enums"""
        data = asdict(self)
        data['quantum_level'] = self.quantum_level.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCTSFullConfig':
        """Create from dictionary, handling enums"""
        if 'quantum_level' in data and isinstance(data['quantum_level'], str):
            data['quantum_level'] = QuantumLevel(data['quantum_level'])
        return cls(**data)


@dataclass
class NeuralNetworkConfig:
    """Neural network architecture and training configuration"""
    # Architecture
    model_type: str = "resnet"  # resnet, simple, lightweight
    input_channels: int = 20
    num_res_blocks: int = 10
    num_filters: int = 256
    value_head_hidden_size: int = 256
    policy_head_filters: int = 2
    
    # Regularization
    dropout_rate: float = 0.1
    batch_norm: bool = True
    batch_norm_momentum: float = 0.997
    l2_regularization: float = 1e-4
    
    # Activation
    activation: str = "relu"  # relu, leaky_relu, elu, gelu
    leaky_relu_alpha: float = 0.01
    
    # Initialization
    weight_init: str = "he_normal"  # he_normal, xavier_normal, orthogonal
    bias_init: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeuralNetworkConfig':
        return cls(**data)


@dataclass
class TrainingFullConfig:
    """Complete training configuration"""
    # Basic training
    batch_size: int = 512
    learning_rate: float = 0.01
    learning_rate_schedule: str = "step"  # step, cosine, exponential, none
    lr_decay_steps: int = 50
    lr_decay_rate: float = 0.1
    min_learning_rate: float = 1e-5
    
    # Optimization
    optimizer: str = "adam"  # adam, sgd, adamw, lamb
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    sgd_momentum: float = 0.9
    sgd_nesterov: bool = True
    weight_decay: float = 1e-4
    
    # Gradient handling
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 5.0
    gradient_clip_value: Optional[float] = None
    
    # Tournament settings for final evaluation
    final_tournament_enabled: bool = True
    final_tournament_model_selection_step: int = 10  # Select every Nth model
    final_tournament_max_models: int = 10  # Maximum models in tournament
    
    # Training loop
    num_epochs: int = 10
    checkpoint_interval: int = 100
    validation_interval: int = 10
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 1e-4
    
    # Self-play
    num_games_per_iteration: int = 100
    num_workers: int = 4
    games_per_worker: int = 25
    max_moves_per_game: int = 500
    resign_threshold: float = -0.95
    resign_check_moves: int = 10
    
    # Data handling
    window_size: int = 500000
    sample_weight_by_game_length: bool = True
    augment_data: bool = True
    shuffle_buffer_size: int = 10000
    dataloader_workers: int = 4
    pin_memory: bool = True
    
    # Mixed precision
    mixed_precision: bool = True
    amp_opt_level: str = "O1"  # O0, O1, O2, O3
    loss_scale: str = "dynamic"  # dynamic, static
    static_loss_scale: float = 1.0
    
    # Evaluation
    eval_temperature: float = 0.1
    eval_num_games: int = 20
    
    # Paths
    save_dir: str = "checkpoints"
    tensorboard_dir: str = "runs"
    data_dir: str = "self_play_data"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingFullConfig':
        return cls(**data)


@dataclass
class ArenaFullConfig:
    """Complete arena configuration"""
    # Battle settings
    num_games: int = 40
    num_workers: int = 4
    games_per_worker: int = 10
    win_threshold: float = 0.55
    statistical_significance: bool = True
    confidence_level: float = 0.95
    
    # Game settings
    temperature: float = 0.1
    mcts_simulations: int = 400
    c_puct: float = 1.0
    max_moves: int = 500
    time_limit_seconds: Optional[float] = None
    randomize_start_player: bool = True  # Random vs alternating start positions
    
    # ELO settings
    elo_k_factor: float = 32.0
    elo_initial_rating: float = 1500.0
    elo_anchor_rating: float = 0.0  # Random policy anchor
    update_elo: bool = True
    
    # Random policy evaluation
    eval_vs_random_interval: int = 10  # Every N iterations
    eval_vs_random_games: int = 20
    min_win_rate_vs_random: float = 0.95  # Sanity check
    
    # Tournament settings
    tournament_rounds: int = 1  # Round-robin rounds
    tournament_games_per_pair: int = 10
    
    # Data saving
    save_game_records: bool = False
    save_arena_logs: bool = True
    arena_log_dir: str = "arena_logs"
    elo_save_path: str = "elo_ratings.json"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArenaFullConfig':
        return cls(**data)


@dataclass
class GameConfig:
    """Game-specific configuration"""
    game_type: str = "gomoku"  # chess, go, gomoku
    board_size: int = 15  # For Go and Gomoku
    
    # Go-specific
    go_komi: float = 7.5
    go_rules: str = "chinese"  # chinese, japanese, tromp_taylor
    go_superko: bool = True
    
    # Gomoku-specific
    gomoku_use_renju: bool = False
    gomoku_use_omok: bool = False
    gomoku_use_pro_long_opening: bool = False
    
    # Chess-specific
    chess_960: bool = False
    chess_starting_fen: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameConfig':
        return cls(**data)


@dataclass
class AlphaZeroConfig:
    """Master configuration containing all components"""
    game: GameConfig = field(default_factory=GameConfig)
    mcts: MCTSFullConfig = field(default_factory=MCTSFullConfig)
    network: NeuralNetworkConfig = field(default_factory=NeuralNetworkConfig)
    training: TrainingFullConfig = field(default_factory=TrainingFullConfig)
    arena: ArenaFullConfig = field(default_factory=ArenaFullConfig)
    
    # Global settings
    experiment_name: str = "alphazero_experiment"
    seed: int = 42
    log_level: str = "INFO"
    num_iterations: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire config to dictionary"""
        return {
            'game': self.game.to_dict(),
            'mcts': self.mcts.to_dict(),
            'network': self.network.to_dict(),
            'training': self.training.to_dict(),
            'arena': self.arena.to_dict(),
            'experiment_name': self.experiment_name,
            'seed': self.seed,
            'log_level': self.log_level,
            'num_iterations': self.num_iterations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlphaZeroConfig':
        """Create from dictionary"""
        return cls(
            game=GameConfig.from_dict(data.get('game', {})),
            mcts=MCTSFullConfig.from_dict(data.get('mcts', {})),
            network=NeuralNetworkConfig.from_dict(data.get('network', {})),
            training=TrainingFullConfig.from_dict(data.get('training', {})),
            arena=ArenaFullConfig.from_dict(data.get('arena', {})),
            experiment_name=data.get('experiment_name', 'alphazero_experiment'),
            seed=data.get('seed', 42),
            log_level=data.get('log_level', 'INFO'),
            num_iterations=data.get('num_iterations', 1000)
        )
    
    def save(self, path: str):
        """Save configuration to YAML file"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'AlphaZeroConfig':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        config = cls.from_dict(data)
        logger.info(f"Configuration loaded from {path}")
        return config
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings"""
        warnings = []
        
        # MCTS validation
        if self.mcts.max_wave_size < self.mcts.min_wave_size:
            warnings.append("MCTS max_wave_size < min_wave_size")
        
        if self.mcts.adaptive_wave_sizing and self.mcts.max_wave_size > 1024:
            warnings.append("Large wave sizes with adaptive sizing may cause OOM")
        
        if self.mcts.quantum_level != QuantumLevel.CLASSICAL and not self.mcts.enable_quantum:
            warnings.append("Quantum level set but quantum not enabled")
        
        # Training validation
        if self.training.num_workers > self.training.num_games_per_iteration:
            warnings.append("More workers than games per iteration")
        
        if self.training.batch_size > self.training.window_size:
            warnings.append("Batch size larger than replay buffer")
        
        # Arena validation
        if self.arena.num_workers > self.arena.num_games:
            warnings.append("More arena workers than games")
        
        if self.arena.win_threshold < 0.5:
            warnings.append("Win threshold below 50% - new models will always be rejected")
        
        # Game validation
        if self.game.game_type == "go" and self.game.board_size not in [9, 13, 19]:
            warnings.append(f"Non-standard Go board size: {self.game.board_size}")
        
        if self.game.game_type == "gomoku" and self.game.board_size < 15:
            warnings.append(f"Gomoku board size {self.game.board_size} may be too small")
        
        return warnings


def create_default_config(game_type: str = "gomoku") -> AlphaZeroConfig:
    """Create default configuration for a game type
    
    Args:
        game_type: Type of game (chess, go, gomoku)
        
    Returns:
        Default configuration
    """
    config = AlphaZeroConfig()
    config.game.game_type = game_type
    
    # Game-specific defaults
    if game_type == "chess":
        config.game.board_size = 8
        config.mcts.dirichlet_alpha = 0.3
        config.network.num_res_blocks = 20
        config.network.num_filters = 256
    elif game_type == "go":
        config.game.board_size = 19
        config.game.go_komi = 7.5
        config.mcts.dirichlet_alpha = 0.03  # Lower for larger board
        config.network.num_res_blocks = 20
        config.network.num_filters = 256
    elif game_type == "gomoku":
        config.game.board_size = 15
        config.mcts.dirichlet_alpha = 0.15
        config.network.num_res_blocks = 10
        config.network.num_filters = 128
    
    return config


def merge_configs(base: AlphaZeroConfig, override: Dict[str, Any]) -> AlphaZeroConfig:
    """Merge override dictionary into base configuration
    
    Args:
        base: Base configuration
        override: Dictionary with overrides (can be nested)
        
    Returns:
        Merged configuration
    """
    # Convert base to dict, merge, and recreate
    base_dict = base.to_dict()
    
    def deep_merge(d1: Dict, d2: Dict) -> Dict:
        """Recursively merge dictionaries"""
        result = d1.copy()
        for key, value in d2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    merged_dict = deep_merge(base_dict, override)
    return AlphaZeroConfig.from_dict(merged_dict)