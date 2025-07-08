"""Comprehensive configuration system for AlphaZero training pipeline

This module provides YAML-based configuration for all components including
MCTS, neural networks, training, arena, and quantum features.
"""

import os
import yaml
import logging
import psutil
import platform
import multiprocessing
from dataclasses import dataclass, field, asdict, fields
from typing import Dict, Any, Optional, List, Union, get_origin, get_args
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import torch for GPU detection
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


def convert_field_value(value: Any, field_type: type) -> Any:
    """Convert a value to the appropriate type based on field annotation"""
    # Handle None values
    if value is None:
        return None
        
    # Handle string 'null' values from YAML
    if isinstance(value, str) and value.lower() in ('null', 'none', '~'):
        return None
    
    # Get the origin and args for generic types (Optional, Union, etc)
    origin = get_origin(field_type)
    args = get_args(field_type)
    
    # Handle Optional[T] which is Union[T, None]
    if origin is Union:
        # For Optional[T], try to convert to the non-None type
        non_none_types = [arg for arg in args if arg is not type(None)]
        if non_none_types:
            return convert_field_value(value, non_none_types[0])
    
    # Handle basic types
    if field_type == float:
        return float(value)
    elif field_type == int:
        return int(value)
    elif field_type == bool:
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        else:
            return bool(value)
    elif field_type == QuantumLevel:
        if isinstance(value, str):
            return QuantumLevel(value)
        else:
            return value
    else:
        return value


class QuantumLevel(Enum):
    """Quantum computation level (disabled)"""
    CLASSICAL = "classical"      # Classical only


@dataclass
class HardwareInfo:
    """Hardware information for auto-detection and optimization"""
    cpu_count: int
    cpu_freq_mhz: float
    total_memory_gb: float
    available_memory_gb: float
    
    # GPU info
    has_gpu: bool = False
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    gpu_compute_capability: tuple = (0, 0)
    cuda_version: str = ""
    
    # System info
    os_name: str = ""
    python_version: str = ""


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
    batch_size: int = 256
    virtual_loss: float = 1.0
    enable_virtual_loss: bool = True
    enable_fast_ucb: bool = True
    classical_only_mode: bool = True
    
    # Memory and optimization
    memory_pool_size_mb: int = 2048
    max_tree_nodes: int = 500000
    tree_reuse: bool = True
    
    # GPU optimization
    use_cuda_graphs: bool = True
    use_mixed_precision: bool = True
    use_tensor_cores: bool = True
    
    # TensorRT acceleration
    use_tensorrt: bool = True
    tensorrt_fp16: bool = True
    tensorrt_fallback: bool = True
    tensorrt_workspace_size: int = 2048  # MB - workspace size for optimization
    tensorrt_int8: bool = False  # INT8 quantization (requires calibration)
    tensorrt_max_batch_size: int = 512  # Maximum batch size to optimize for
    tensorrt_engine_cache_dir: Optional[str] = None  # Custom cache directory
    
    # Quantum features disabled
    enable_quantum: bool = False
    
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
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCTSFullConfig':
        """Create from dictionary, handling enums and type conversion"""
        # Convert string values to proper types
        converted_data = {}
        for field in fields(cls):
            field_name = field.name
            if field_name in data:
                value = data[field_name]
                converted_data[field_name] = convert_field_value(value, field.type)
            
        return cls(**converted_data)


@dataclass
class NeuralNetworkConfig:
    """Neural network architecture and training configuration"""
    # Architecture
    model_type: str = "resnet"  # resnet, simple, lightweight
    input_channels: int = 18
    input_representation: str = "basic"  # basic, standard, enhanced, compact
    num_res_blocks: int = 10
    num_filters: int = 256
    fc_hidden_size: int = 256  # Hidden size for fully connected layers
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
    
    def __post_init__(self):
        """Validate neural network configuration"""
        if self.num_res_blocks < 0:
            raise ValueError(f"num_res_blocks must be non-negative, got {self.num_res_blocks}")
        if self.num_filters <= 0:
            raise ValueError(f"num_filters must be positive, got {self.num_filters}")
        if self.input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {self.input_channels}")
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeuralNetworkConfig':
        # Convert string values to proper types
        converted_data = {}
        for field in fields(cls):
            field_name = field.name
            if field_name in data:
                value = data[field_name]
                converted_data[field_name] = convert_field_value(value, field.type)
            
        return cls(**converted_data)


@dataclass
class TrainingFullConfig:
    """Complete training configuration"""
    # Basic training
    batch_size: int = 512  # Default batch size
    learning_rate: float = 0.01
    learning_rate_schedule: str = "step"  # step, cosine, exponential, none
    lr_schedule: str = "step"  # Alias for learning_rate_schedule
    lr_decay_steps: int = 50
    lr_decay_rate: float = 0.1
    min_learning_rate: float = 1e-5
    device: str = "cuda"  # cuda or cpu
    
    # Training iterations
    num_iterations: int = 100
    checkpoint_interval: int = 10
    
    # Optimization
    optimizer: str = "adam"  # adam, sgd, adamw, lamb
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    sgd_momentum: float = 0.9
    momentum: float = 0.9  # Alias for sgd_momentum
    sgd_nesterov: bool = True
    weight_decay: float = 1e-4
    
    # Gradient handling
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 5.0
    gradient_clip_norm: float = 10.0  # Alias for max_grad_norm
    gradient_clip_value: Optional[float] = None
    
    # Tournament settings for final evaluation
    final_tournament_enabled: bool = True
    final_tournament_model_selection_step: int = 10  # Select every Nth model
    final_tournament_max_models: int = 10  # Maximum models in tournament
    
    # Training loop
    num_epochs: int = 10
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 1e-4
    
    # Self-play
    num_games_per_iteration: int = 100
    num_workers: int = 4
    games_per_worker: int = 25
    max_moves_per_game: int = 500
    temperature_threshold: int = 30  # Move threshold for temperature reduction
    resign_threshold: float = -0.98  # FIXED: More conservative threshold (was -0.95)
    resign_check_moves: int = 10
    resign_start_iteration: int = 20  # FIXED: Start resignation later (was 10)
    resign_threshold_decay: float = 0.995  # Gradually make resignation more aggressive
    resign_randomness: float = 0.1  # Add randomness to prevent uniform behavior
    
    # Data handling
    window_size: int = 500000
    augment_data: bool = True
    shuffle_buffer_size: int = 10000
    dataloader_workers: int = 4
    pin_memory: bool = True
    
    # Mixed precision
    mixed_precision: bool = True
    amp_opt_level: str = "O1"  # O0, O1, O2, O3
    loss_scale: str = "dynamic"  # dynamic, static
    static_loss_scale: float = 1.0
    
    # Paths
    save_dir: str = "checkpoints"
    tensorboard_dir: str = "runs"
    data_dir: str = "self_play_data"
    
    # Distributed training
    distributed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingFullConfig':
        # Convert string values to proper types
        converted_data = {}
        for field in fields(cls):
            field_name = field.name
            if field_name in data:
                value = data[field_name]
                converted_data[field_name] = convert_field_value(value, field.type)
        
        # Auto-calculate games_per_worker if not provided
        if 'games_per_worker' not in converted_data and 'num_games_per_iteration' in converted_data and 'num_workers' in converted_data:
            num_games = converted_data['num_games_per_iteration']
            num_workers = converted_data['num_workers']
            converted_data['games_per_worker'] = (num_games + num_workers - 1) // num_workers  # Round up
            
        return cls(**converted_data)


@dataclass
class ArenaFullConfig:
    """Complete arena configuration"""
    # Battle settings
    num_games: int = 40
    num_workers: int = 4
    games_per_worker: int = 10
    win_threshold: float = 0.55
    update_threshold: float = 0.51  # Threshold for updating best model
    statistical_significance: bool = True
    confidence_level: float = 0.95
    
    # Game settings
    temperature: float = 0.0  # Deterministic play for arena evaluation
    mcts_simulations: int = 400
    c_puct: float = 1.0
    max_moves: int = 500
    time_limit_seconds: Optional[float] = None
    randomize_start_player: bool = True  # Random vs alternating start positions
    
    # ELO settings
    elo_k_factor: float = 32.0
    elo_initial_rating: float = 1500.0
    initial_elo: float = 1500.0  # Alias for elo_initial_rating
    elo_anchor_rating: float = 0.0  # Random policy anchor
    random_elo_anchor: float = 0.0  # Alias for elo_anchor_rating
    update_elo: bool = True
    play_vs_random_interval: int = 10  # How often to play vs random
    
    # Random policy evaluation
    min_win_rate_vs_random: float = 0.95  # Sanity check for first model
    
    # Tournament settings
    tournament_rounds: int = 1  # Round-robin rounds
    tournament_games_per_pair: int = 10
    
    # Data saving
    save_game_records: bool = False
    save_arena_logs: bool = True
    arena_log_dir: str = "arena_logs"
    elo_save_path: str = "elo_ratings.json"
    
    # New arena features
    enable_current_vs_previous: bool = True  # Enable current vs previous model arena matches
    enable_adaptive_random_matches: bool = True  # Use adaptive logic for random matches
    enable_elo_consistency_checks: bool = True  # Check for ELO inconsistencies
    enable_elo_auto_adjustment: bool = True  # Automatically adjust ELO when inconsistencies are detected
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArenaFullConfig':
        # Convert string values to proper types
        converted_data = {}
        for field in fields(cls):
            field_name = field.name
            if field_name in data:
                value = data[field_name]
                converted_data[field_name] = convert_field_value(value, field.type)
        
        # Auto-calculate games_per_worker if not provided
        if 'games_per_worker' not in converted_data and 'num_games' in converted_data and 'num_workers' in converted_data:
            num_games = converted_data['num_games']
            num_workers = converted_data['num_workers']
            converted_data['games_per_worker'] = (num_games + num_workers - 1) // num_workers  # Round up
            
        return cls(**converted_data)


@dataclass
class GameConfig:
    """Game-specific configuration"""
    game_type: str = "gomoku"  # chess, go, gomoku
    board_size: int = 15  # For Go and Gomoku
    win_length: int = 5  # For Gomoku
    use_symmetries: bool = True
    
    # Go-specific
    go_komi: float = 7.5
    go_rules: str = "chinese"  # chinese, japanese, tromp_taylor
    go_superko: bool = True
    komi: float = 7.5  # Alias for go_komi
    
    # Gomoku-specific
    gomoku_use_renju: bool = False
    gomoku_use_omok: bool = False
    gomoku_use_pro_long_opening: bool = False
    
    # Chess-specific
    chess_960: bool = False
    chess_starting_fen: Optional[str] = None
    
    def __post_init__(self):
        """Adjust board size based on game type and validate"""
        # Validate game type
        valid_game_types = ['chess', 'go', 'gomoku']
        if self.game_type not in valid_game_types:
            raise ValueError(f"Invalid game type: {self.game_type}. Must be one of {valid_game_types}")
            
        # Adjust board size based on game type
        if self.game_type == 'chess' and self.board_size == 15:
            self.board_size = 8
            
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameConfig':
        # Convert string values to proper types
        converted_data = {}
        for field in fields(cls):
            field_name = field.name
            if field_name in data:
                value = data[field_name]
                converted_data[field_name] = convert_field_value(value, field.type)
            
        return cls(**converted_data)


@dataclass
class LogConfig:
    """Logging configuration with comprehensive settings"""
    # Directory settings
    log_dir: str = "logs"
    tensorboard_dir: str = "tensorboard"
    checkpoint_dir: str = "checkpoints"
    file_path: str = "logs/training.log"
    
    # Logging levels
    level: str = "INFO"  # Main log level
    console_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    file_level: str = "DEBUG"
    console: bool = True
    
    # Tensorboard settings
    tensorboard: bool = True
    tensorboard_flush_secs: int = 30
    tensorboard_histogram_freq: int = 10
    
    # Checkpoint settings
    checkpoint_frequency: int = 10  # Save every N iterations
    checkpoint_keep_last: int = 5  # Keep last N checkpoints
    checkpoint_keep_best: int = 3  # Keep best N checkpoints by ELO
    
    # Metrics logging
    log_metrics: bool = True
    metrics_frequency: int = 1  # Log metrics every N games
    detailed_metrics: bool = False  # Log per-move metrics
    
    # Game logging
    save_game_records: bool = True
    game_record_format: str = "pgn"  # pgn, sgf, or custom
    save_failed_games: bool = True  # Save games that ended in errors
    
    # Performance logging
    profile_performance: bool = False
    profile_frequency: int = 100  # Profile every N games
    
    def __post_init__(self):
        """Validate logging configuration"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        if self.console_level not in valid_levels:
            raise ValueError(f"Invalid console_level: {self.console_level}")
        if self.file_level not in valid_levels:
            raise ValueError(f"Invalid file_level: {self.file_level}")
            
        valid_formats = ['pgn', 'sgf', 'custom']
        if self.game_record_format not in valid_formats:
            raise ValueError(f"Invalid game_record_format: {self.game_record_format}")
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResourceConfig:
    """Resource management configuration with hardware optimization"""
    # Worker settings
    num_workers: int = 4  # Number of parallel workers
    max_cpu_workers: int = 4  # Maximum CPU workers
    num_data_workers: int = 2  # Number of data loading workers
    num_actors: int = 8  # Number of self-play actors
    num_evaluators: int = 2  # Number of neural network evaluators
    
    # CPU settings
    cpu_threads_per_worker: int = 1
    cpu_affinity: bool = False  # Pin workers to specific CPU cores
    
    # Memory settings
    memory_limit_gb: Optional[float] = None  # None = auto-detect
    max_gpu_memory_gb: float = 8.0  # Maximum GPU memory to use
    tree_memory_fraction: float = 0.4  # Fraction of memory for MCTS trees
    buffer_memory_fraction: float = 0.3  # Fraction for replay buffers
    pin_memory: bool = True  # Pin memory for faster GPU transfer
    cache_size: int = 1000  # Cache size for frequently accessed data
    
    # GPU settings
    num_gpus: int = 1  # Number of GPUs to use
    gpu_memory_fraction: float = 0.9  # Fraction of GPU memory to use
    gpu_allow_growth: bool = True  # Allow GPU memory to grow dynamically
    mixed_precision: bool = True  # Use mixed precision training
    
    # Batch settings
    inference_batch_size: int = 256  # Batch size for NN inference
    training_batch_queue_size: int = 10  # Queue size for training batches
    batch_queue_size: int = 10  # Alias for training_batch_queue_size
    
    # Queue settings
    self_play_queue_size: int = 100  # Max games in self-play queue
    training_queue_size: int = 50  # Max samples in training queue
    
    # Performance settings
    enable_profiling: bool = False  # Enable performance profiling
    profile_kernel_launches: bool = False
    
    # Resource limits
    max_tree_nodes_per_worker: int = 800000
    max_game_length: int = 500
    
    def __post_init__(self):
        """Validate and auto-detect resources"""
        import multiprocessing
        
        # Auto-detect number of workers if not set
        if self.num_workers <= 0:
            self.num_workers = max(1, multiprocessing.cpu_count() - 2)
            
        # Validate memory fractions
        total_fraction = self.tree_memory_fraction + self.buffer_memory_fraction
        if total_fraction > 0.95:
            logger.warning(f"Memory fractions sum to {total_fraction}, may cause OOM")
            
        # Auto-detect memory limit if not set
        if self.memory_limit_gb is None:
            if psutil is not None:
                total_memory_gb = psutil.virtual_memory().total / (1024**3)
                self.memory_limit_gb = total_memory_gb * 0.8  # Use 80% of total
            else:
                self.memory_limit_gb = 8.0  # Conservative default
                
        # Ensure batch size is reasonable
        if self.inference_batch_size > 1024:
            logger.warning(f"Large inference_batch_size: {self.inference_batch_size}")
    
    def get_memory_budget(self) -> Dict[str, float]:
        """Calculate memory budget for different components"""
        total_gb = self.memory_limit_gb or 8.0
        return {
            'tree_gb': total_gb * self.tree_memory_fraction,
            'buffer_gb': total_gb * self.buffer_memory_fraction,
            'other_gb': total_gb * (1 - self.tree_memory_fraction - self.buffer_memory_fraction)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AlphaZeroConfig:
    """Master configuration containing all components"""
    
    def __post_init__(self):
        """Post-initialization processing"""
        # This is called after the dataclass __init__
        pass
    
    def __init__(self, **kwargs):
        """Custom initialization to handle nested attribute assignment"""
        # Extract game config parameters first
        game_kwargs = {}
        if 'game_type' in kwargs:
            game_kwargs['game_type'] = kwargs.pop('game_type')
        
        # Extract other nested config parameters
        network_kwargs = {}
        if 'num_res_blocks' in kwargs:
            network_kwargs['num_res_blocks'] = kwargs.pop('num_res_blocks')
            
        training_kwargs = {}
        if 'batch_size' in kwargs:
            training_kwargs['batch_size'] = kwargs.pop('batch_size')
        
        # Initialize all fields with defaults and extracted parameters
        self.game = GameConfig(**game_kwargs)
        self.mcts = MCTSFullConfig()
        self.network = NeuralNetworkConfig(**network_kwargs)
        self.training = TrainingFullConfig(**training_kwargs)
        self.arena = ArenaFullConfig()
        self.log = LogConfig()
        self.resources = ResourceConfig()
        self.experiment_name = "default"
        self.checkpoint_dir = "checkpoints"
        self.tensorboard_dir = "runs"
        self.save_interval = 10
        self.seed = 42
        self.log_level = "INFO"
        self.num_iterations = 1000
        self.auto_detect_hardware = True
        self.hardware_info = None
            
        # Set remaining kwargs as direct attributes
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Try to set on nested configs
                for config_name in ['game', 'mcts', 'network', 'training', 'arena']:
                    config = getattr(self, config_name)
                    if hasattr(config, key):
                        setattr(config, key, value)
                        break
                else:
                    logger.warning(f"Unknown config parameter: {key}")
                    
        # Call post_init
        self.__post_init__()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire config to dictionary"""
        return {
            'game': self.game.to_dict(),
            'mcts': self.mcts.to_dict(),
            'network': self.network.to_dict(),
            'training': self.training.to_dict(),
            'arena': self.arena.to_dict(),
            'log': self.log.to_dict(),
            'resources': self.resources.to_dict(),
            'experiment_name': self.experiment_name,
            'checkpoint_dir': self.checkpoint_dir,
            'tensorboard_dir': self.tensorboard_dir,
            'save_interval': self.save_interval,
            'seed': self.seed,
            'log_level': self.log_level,
            'num_iterations': self.num_iterations
        }
    
    def __str__(self) -> str:
        """String representation of configuration"""
        lines = [
            "AlphaZeroConfig(",
            f"  game_type: {self.game.game_type}",
            f"  board_size: {self.game.board_size}",
            f"  num_res_blocks: {self.network.num_res_blocks}",
            f"  num_filters: {self.network.num_filters}",
            f"  batch_size: {self.training.batch_size}",
            f"  learning_rate: {self.training.learning_rate}",
            f"  num_simulations: {self.mcts.num_simulations}",
            f"  c_puct: {self.mcts.c_puct}",
            f"  experiment_name: {self.experiment_name}",
            ")"
        ]
        return "\n".join(lines)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlphaZeroConfig':
        """Create from dictionary"""
        return cls(
            game=GameConfig.from_dict(data.get('game', {})),
            mcts=MCTSFullConfig.from_dict(data.get('mcts', {})),
            network=NeuralNetworkConfig.from_dict(data.get('network', {})),
            training=TrainingFullConfig.from_dict(data.get('training', {})),
            arena=ArenaFullConfig.from_dict(data.get('arena', {})),
            experiment_name=data.get('experiment_name', 'default'),
            checkpoint_dir=data.get('checkpoint_dir', 'checkpoints'),
            tensorboard_dir=data.get('tensorboard_dir', 'runs'),
            save_interval=data.get('save_interval', 10),
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
    def load(cls, path: str, auto_adjust_hardware: bool = True) -> 'AlphaZeroConfig':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        config = cls.from_dict(data)
        logger.info(f"Configuration loaded from {path}")
        
        # Store explicitly set YAML values to preserve them during hardware adjustment
        if auto_adjust_hardware and config.auto_detect_hardware:
            # Store original YAML values that should not be overridden
            yaml_overrides = {}
            if 'mcts' in data:
                mcts_data = data['mcts']
                if 'max_tree_nodes' in mcts_data:
                    yaml_overrides['max_tree_nodes'] = mcts_data['max_tree_nodes']
                if 'memory_pool_size_mb' in mcts_data:
                    yaml_overrides['memory_pool_size_mb'] = mcts_data['memory_pool_size_mb']
                if 'wave_size' in mcts_data:
                    yaml_overrides['wave_size'] = mcts_data['wave_size']
                if 'min_wave_size' in mcts_data:
                    yaml_overrides['min_wave_size'] = mcts_data['min_wave_size']
                if 'max_wave_size' in mcts_data:
                    yaml_overrides['max_wave_size'] = mcts_data['max_wave_size']
            
            # Also preserve training batch size if explicitly set
            if 'training' in data:
                training_data = data['training']
                if 'batch_size' in training_data:
                    yaml_overrides['batch_size'] = training_data['batch_size']
            
            config.adjust_for_hardware(yaml_overrides)
        
        return config
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings"""
        warnings = []
        
        # Critical validation errors - raise ValueError immediately
        
        # Game validation - critical errors
        if self.game.game_type == "gomoku" and self.game.board_size > 25:
            raise ValueError(f"Invalid board size {self.game.board_size} for gomoku - maximum is 25")
        
        if self.game.board_size <= 0:
            raise ValueError(f"Invalid board size {self.game.board_size} - must be positive")
        
        # Network validation - critical errors  
        if self.network.num_res_blocks <= 0:
            raise ValueError(f"Invalid number of res blocks {self.network.num_res_blocks} - must be positive")
        
        # Training validation - critical errors
        if self.training.batch_size <= 0:
            raise ValueError(f"Invalid batch size {self.training.batch_size} - must be positive")
        
        if self.training.learning_rate <= 0:
            raise ValueError(f"Invalid learning rate {self.training.learning_rate} - must be positive")
        
        # MCTS validation - critical errors
        if self.mcts.num_simulations <= 0:
            raise ValueError(f"Invalid number of simulations {self.mcts.num_simulations} - must be positive")
        
        if self.mcts.c_puct <= 0:
            raise ValueError(f"Invalid c_puct {self.mcts.c_puct} - must be positive")
        
        # Non-critical validation - warnings only
        
        # MCTS validation
        if self.mcts.max_wave_size < self.mcts.min_wave_size:
            warnings.append("MCTS max_wave_size < min_wave_size")
        
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
        
        # Game validation - warnings only
        if self.game.game_type == "go" and self.game.board_size not in [9, 13, 19]:
            warnings.append(f"Non-standard Go board size: {self.game.board_size}")
        
        if self.game.game_type == "gomoku" and self.game.board_size < 15:
            warnings.append(f"Gomoku board size {self.game.board_size} may be too small")
        
        return warnings
    
    def detect_hardware(self) -> Dict[str, Any]:
        """Detect system hardware specifications"""
        # CPU information
        cpu_count = os.cpu_count() or 1
        cpu_cores_physical = psutil.cpu_count(logical=False) or cpu_count // 2
        cpu_threads = psutil.cpu_count(logical=True) or cpu_count
        
        # Memory information
        mem = psutil.virtual_memory()
        total_ram_gb = mem.total / (1024**3)
        available_ram_gb = mem.available / (1024**3)
        
        # GPU information
        # Lazy import torch to avoid importing it in worker processes
        gpu_count = 0
        gpu_memory_mb = 0
        gpu_name = "None"
        
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_count = torch.cuda.device_count()
                gpu_props = torch.cuda.get_device_properties(0)
                gpu_name = gpu_props.name
                gpu_memory_mb = gpu_props.total_memory // (1024*1024)
        except ImportError:
            gpu_available = False
        
        hardware_info = {
            'cpu_count': cpu_count,
            'cpu_cores_physical': cpu_cores_physical,
            'cpu_threads': cpu_threads,
            'total_ram_gb': total_ram_gb,
            'available_ram_gb': available_ram_gb,
            'gpu_available': gpu_available,
            'gpu_count': gpu_count,
            'gpu_name': gpu_name,
            'gpu_memory_mb': gpu_memory_mb,
            'os_name': f"{platform.system()} {platform.release()}",
            'python_version': platform.python_version()
        }
        
        self.hardware_info = hardware_info
        return hardware_info
    
    def calculate_resource_allocation(self, hardware: Dict[str, Any], 
                                    target_workers: Optional[int] = None) -> Dict[str, Any]:
        """Calculate optimal resource allocation based on hardware"""
        # Worker allocation - optimize for CPU utilization
        if target_workers is None:
            if hardware['gpu_available']:
                # Use more workers to better utilize CPU while GPU handles NN
                num_workers = min(
                    hardware['cpu_cores_physical'],  # Physical cores
                    hardware['cpu_threads'] - 2,      # Leave 2 threads for system
                    16  # Reasonable upper limit
                )
            else:
                num_workers = min(hardware['cpu_cores_physical'] // 2, 4)
        else:
            num_workers = target_workers
        
        # Memory calculations
        if hardware['gpu_available']:
            gpu_reserved_mb = 2048
            gpu_available_mb = hardware['gpu_memory_mb'] - gpu_reserved_mb
            gpu_memory_per_worker_mb = max(128, gpu_available_mb // num_workers)
            max_concurrent_by_gpu = gpu_available_mb // 256
            use_gpu_for_workers = num_workers <= 4
            gpu_memory_fraction = min(0.8, gpu_memory_per_worker_mb * num_workers / hardware['gpu_memory_mb'])
        else:
            gpu_memory_per_worker_mb = 0
            max_concurrent_by_gpu = num_workers
            use_gpu_for_workers = False
            gpu_memory_fraction = 0.0
        
        # RAM calculations - more efficient for CPU-based workers
        ram_reserved_gb = 4
        ram_available_gb = max(1, hardware['total_ram_gb'] - ram_reserved_gb)
        ram_per_worker_gb = 0.5  # Reduced from 1.5GB since workers use CPU
        max_concurrent_by_ram = int(ram_available_gb / ram_per_worker_gb)
        
        # Final limits - allow all workers to run concurrently for better throughput
        # Workers use CPU for MCTS while GPU service handles neural network
        cpu_cores_limit = hardware['cpu_threads'] - 2  # Leave 2 threads for system
        max_concurrent_workers = min(num_workers, max_concurrent_by_gpu, max_concurrent_by_ram, cpu_cores_limit)
        memory_per_worker_mb = min(2048, int(hardware['total_ram_gb'] * 1024 - 4096) // num_workers)
        
        return {
            'num_workers': num_workers,
            'max_concurrent_workers': max_concurrent_workers,
            'memory_per_worker_mb': memory_per_worker_mb,
            'gpu_memory_per_worker_mb': gpu_memory_per_worker_mb,
            'gpu_memory_fraction': gpu_memory_fraction,
            'use_gpu_for_workers': use_gpu_for_workers
        }
    
    def adjust_for_hardware(self, yaml_overrides: Optional[Dict[str, Any]] = None, target_workers: Optional[int] = None):
        """Adjust configuration based on detected hardware, respecting YAML overrides"""
        if yaml_overrides is None:
            yaml_overrides = {}
            
        hardware = self.detect_hardware()
        allocation = self.calculate_resource_allocation(hardware, target_workers or self.training.num_workers)
        
        logger.info(f"Detected hardware: {hardware['cpu_cores_physical']} CPU cores, "
                   f"{hardware['total_ram_gb']:.1f}GB RAM")
        if hardware['gpu_available']:
            logger.info(f"GPU: {hardware['gpu_name']} ({hardware['gpu_memory_mb']}MB)")
        
        # Apply adjustments
        self.training.num_workers = allocation['num_workers']
        self.training.dataloader_workers = min(hardware['cpu_threads'] // 4, 8)
        self.training.pin_memory = hardware['gpu_available'] and hardware['total_ram_gb'] >= 16
        
        # Adjust MCTS settings based on GPU, but respect YAML overrides
        if hardware['gpu_available']:
            if hardware['gpu_memory_mb'] >= 7500:  # 8GB GPUs (accounting for system overhead)
                # Only adjust if not explicitly set in YAML
                if 'memory_pool_size_mb' not in yaml_overrides:
                    self.mcts.memory_pool_size_mb = 2048
                if 'max_tree_nodes' not in yaml_overrides:
                    self.mcts.max_tree_nodes = 500000
                if 'batch_size' not in yaml_overrides:
                    self.training.batch_size = 512
            elif hardware['gpu_memory_mb'] >= 6144:
                if 'memory_pool_size_mb' not in yaml_overrides:
                    self.mcts.memory_pool_size_mb = 1536
                if 'max_tree_nodes' not in yaml_overrides:
                    self.mcts.max_tree_nodes = 300000
                if 'batch_size' not in yaml_overrides:
                    self.training.batch_size = 256
            else:
                if 'memory_pool_size_mb' not in yaml_overrides:
                    self.mcts.memory_pool_size_mb = 1024
                if 'max_tree_nodes' not in yaml_overrides:
                    self.mcts.max_tree_nodes = 200000
                if 'batch_size' not in yaml_overrides:
                    self.training.batch_size = 128
                
            # Log what values are being preserved from YAML
            if yaml_overrides:
                preserved_values = []
                for key, value in yaml_overrides.items():
                    preserved_values.append(f"{key}={value}")
                logger.info(f"Preserving YAML values: {', '.join(preserved_values)}")
        else:
            # CPU-only mode - but this should not apply to workers since they get their own MCTSConfig
            # Workers should never hit this path since they use pre-configured MCTSConfig
            self.mcts.memory_pool_size_mb = 512
            self.mcts.max_tree_nodes = 100000
            self.training.batch_size = 64
        
        # Store allocation info for workers
        self._resource_allocation = allocation
        
        logger.info(f"Adjusted config: {allocation['num_workers']} workers, "
                   f"{self.mcts.memory_pool_size_mb}MB memory pool")


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


def merge_configs(base: Union[AlphaZeroConfig, Dict[str, Any]], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge override dictionary into base configuration
    
    Args:
        base: Base configuration (can be AlphaZeroConfig object or dict)
        override: Dictionary with overrides (can be nested)
        
    Returns:
        Merged configuration as dictionary
    """
    # Convert base to dict if it's an AlphaZeroConfig object
    if hasattr(base, 'to_dict'):
        base_dict = base.to_dict()
    else:
        base_dict = base
    
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
    
    return merged_dict