"""
Omoknuni MCTS - Quantum-inspired Monte Carlo Tree Search

This package implements vectorized MCTS with quantum-inspired diversity mechanisms
for achieving high-performance game AI on consumer hardware.
"""

__version__ = "0.2.0"
__author__ = "Omoknuni Team"

# Core components
from .core import (
    Node, TreeArena, MemoryConfig,
    GameInterface, GameType,
    Evaluator, MockEvaluator, EvaluatorConfig,
    BatchGameOps, BatchGameOpsConfig, TensorGameState,
    MCTS, MCTSConfig,
    WaveMCTS, WaveMCTSConfig,
    CachedGameInterface, CacheConfig,
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
    # Training
    TrainingPipeline, TrainingConfig, create_training_pipeline
)

# GPU acceleration components
from .gpu import (
    CSRTree, CSRTreeConfig,
    GPUOptimizer, AsyncEvaluator, StateMemoryPool,
    get_csr_kernels, CSRGPUKernels, CSRBatchOperations,
    CUDAKernels, OptimizedCUDAKernels, GPUTreeKernels
)

# Quantum-inspired components (optional)
try:
    from .quantum import (
        InterferenceEngine,
        PhaseKickedPolicy, PhaseConfig,
        PathIntegral, PathIntegralConfig
    )
    __all_quantum = [
        "InterferenceEngine",
        "PhaseKickedPolicy", "PhaseConfig",
        "PathIntegral", "PathIntegralConfig"
    ]
except ImportError:
    __all_quantum = []

# Utility components
from .utils import (
    ConfigManager,
    ResourceMonitor,
    StateDeltaEncoder,
    compute_gomoku_attack_defense_scores,
    compute_attack_defense_scores,
    evaluate_position
)

# No legacy aliases needed - MCTS is now the main class

__all__ = [
    # Core
    "Node", "TreeArena", "MemoryConfig",
    "GameInterface", "GameType",
    "Evaluator", "MockEvaluator", "EvaluatorConfig",
    "BatchGameOps", "BatchGameOpsConfig", "TensorGameState",
    "MCTS", "MCTSConfig",
    "WaveMCTS", "WaveMCTSConfig",
    "CachedGameInterface", "CacheConfig",
    
    # Neural Networks
    "ModelRegistry", "ModelLoader", "BaseGameModel", "ModelMetadata",
    "ModelEnsemble", "AdapterWrapper", "MixedPrecisionWrapper",
    "create_model_from_config", "load_model_for_game",
    "AlphaZeroNetwork", "ModelConfig", "ResidualBlock", "PolicyHead", "ValueHead",
    "ResNetModel", "ResNetConfig", "create_resnet_for_game",
    "ResNetEvaluator", "create_evaluator_for_game",
    "create_chess_evaluator", "create_go_evaluator", "create_gomoku_evaluator",
    "TrainingPipeline", "TrainingConfig", "create_training_pipeline",
    
    # GPU
    "CSRTree", "CSRTreeConfig",
    "GPUOptimizer", "AsyncEvaluator", "StateMemoryPool",
    "get_csr_kernels", "CSRGPUKernels", "CSRBatchOperations",
    "CUDAKernels", "OptimizedCUDAKernels", "GPUTreeKernels",
    
    # Utils
    "ConfigManager",
    "ResourceMonitor",
    "StateDeltaEncoder",
    "compute_gomoku_attack_defense_scores",
    "compute_attack_defense_scores",
    "evaluate_position",
] + __all_quantum