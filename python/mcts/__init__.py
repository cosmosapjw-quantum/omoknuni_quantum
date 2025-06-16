"""
Omoknuni MCTS - Quantum-inspired Monte Carlo Tree Search

This package implements vectorized MCTS with quantum-inspired diversity mechanisms
for achieving high-performance game AI on consumer hardware.
"""

__version__ = "0.2.0"
__author__ = "Omoknuni Team"

# Core components
from .core import (
    GameInterface, GameType,
    Evaluator, MockEvaluator, RandomEvaluator, EvaluatorConfig, AlphaZeroEvaluator,
    MCTS, MCTSConfig,
)

# Neural network components
from .neural_networks import (
    # Framework
    ModelRegistry, ModelLoader, BaseGameModel, ModelMetadata,
    ModelEnsemble, AdapterWrapper, MixedPrecisionWrapper,
    create_model_from_config, load_model_for_game,
    # AlphaZero model
    AlphaZeroNetwork, ModelConfig, ResidualBlock, PolicyHead, ValueHead,
    # ResNet
    ResNetModel, ResNetConfig, create_resnet_for_game,
    ResNetEvaluator, create_evaluator_for_game,
    create_chess_evaluator, create_go_evaluator, create_gomoku_evaluator,
)

# GPU acceleration components
from .gpu import (
    CSRTree, CSRTreeConfig,
    get_csr_kernels, CSRGPUKernels, CSRBatchOperations,
    CUDAKernels, OptimizedCUDAKernels, GPUTreeKernels
)

# Quantum-inspired components (optional)
try:
    from .quantum import (
        QuantumConfig,
        create_quantum_mcts,
        QuantumLevel,
        PathIntegral, PathIntegralConfig,
        DecoherenceEngine, DecoherenceConfig,
        MinHashInterference, MinHashConfig
    )
    __all_quantum = [
        "QuantumConfig",
        "create_quantum_mcts",
        "QuantumLevel",
        "PathIntegral", "PathIntegralConfig",
        "DecoherenceEngine", "DecoherenceConfig",
        "MinHashInterference", "MinHashConfig"
    ]
except ImportError:
    __all_quantum = []

# Utility components
from .utils import (
    ConfigManager,
    OptimizedConfig,
    AlphaZeroConfig,
    MCTSFullConfig,
    QuantumLevel,
    create_default_config,
    merge_configs
)

# No legacy aliases needed - MCTS is now the main class

__all__ = [
    # Core
    "GameInterface", "GameType",
    "Evaluator", "MockEvaluator", "RandomEvaluator", "EvaluatorConfig", "AlphaZeroEvaluator",
    "MCTS", "MCTSConfig",
    
    # Neural Networks
    "ModelRegistry", "ModelLoader", "BaseGameModel", "ModelMetadata",
    "ModelEnsemble", "AdapterWrapper", "MixedPrecisionWrapper",
    "create_model_from_config", "load_model_for_game",
    "AlphaZeroNetwork", "ModelConfig", "ResidualBlock", "PolicyHead", "ValueHead",
    "ResNetModel", "ResNetConfig", "create_resnet_for_game",
    "ResNetEvaluator", "create_evaluator_for_game",
    "create_chess_evaluator", "create_go_evaluator", "create_gomoku_evaluator",
    "UnifiedTrainingPipeline", "GameExample",
    "SelfPlayManager", "SelfPlayConfig",
    "ArenaManager", "ArenaConfig", "ELOTracker",
    
    # GPU
    "CSRTree", "CSRTreeConfig",
    "get_csr_kernels", "CSRGPUKernels", "CSRBatchOperations",
    "CUDAKernels", "OptimizedCUDAKernels", "GPUTreeKernels",
    
    # Utils
    "ConfigManager",
    "OptimizedConfig",
    "AlphaZeroConfig",
    "MCTSFullConfig",
    "QuantumLevel",
    "create_default_config",
    "merge_configs",
] + __all_quantum