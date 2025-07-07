"""
Omoknuni MCTS - Quantum-inspired Monte Carlo Tree Search

This package implements vectorized MCTS with quantum-inspired diversity mechanisms
for achieving high-performance game AI on consumer hardware.
"""

__version__ = "0.2.0"
__author__ = "Omoknuni Team"

# Setup CUDA environment before any torch imports
try:
    from . import cuda_setup
except ImportError:
    pass  # cuda_setup is optional

# Core components
from .core import (
    GameInterface, GameType,
    Evaluator, EvaluatorConfig, AlphaZeroEvaluator,
    MCTS, MCTSConfig,
)

# Neural network components
from .neural_networks import (
    # Framework
    ModelRegistry, ModelLoader, BaseGameModel, ModelMetadata,
    ModelEnsemble, AdapterWrapper, MixedPrecisionWrapper,
    create_model_from_config, load_model_for_game,
    # ResNet
    ResNetModel, ResNetConfig, create_resnet_for_game,
    ResNetEvaluator, create_evaluator_for_game,
    create_chess_evaluator, create_go_evaluator, create_gomoku_evaluator,
)

# GPU acceleration components
from .gpu import (
    CSRTree, CSRTreeConfig,
    CSRBatchOperations,
    GPUTreeKernels
)

# Quantum components removed
__all_quantum = []

# Utility components
from .utils import (
    AlphaZeroConfig,
    MCTSFullConfig,
    # QuantumLevel removed,
    HardwareInfo,
    create_default_config,
    merge_configs
)

# No legacy aliases needed - MCTS is now the main class

__all__ = [
    # Core
    "GameInterface", "GameType",
    "Evaluator", "EvaluatorConfig", "AlphaZeroEvaluator",
    "MCTS", "MCTSConfig",
    
    # Neural Networks
    "ModelRegistry", "ModelLoader", "BaseGameModel", "ModelMetadata",
    "ModelEnsemble", "AdapterWrapper", "MixedPrecisionWrapper",
    "create_model_from_config", "load_model_for_game",
    "ResNetModel", "ResNetConfig", "create_resnet_for_game",
    "ResNetEvaluator", "create_evaluator_for_game",
    "create_chess_evaluator", "create_go_evaluator", "create_gomoku_evaluator",
    "UnifiedTrainingPipeline", "GameExample",
    "SelfPlayManager", "SelfPlayConfig",
    "ArenaManager", "ArenaConfig", "ELOTracker",
    
    # GPU
    "CSRTree", "CSRTreeConfig",
    "CSRBatchOperations",
    "GPUTreeKernels",
    
    # Utils
    "AlphaZeroConfig",
    "MCTSFullConfig",
    "QuantumLevel",
    "HardwareInfo",
    "create_default_config",
    "merge_configs",
] + __all_quantum