"""Neural network models and evaluators"""

from .nn_framework import (
    ModelRegistry, ModelLoader, BaseGameModel, ModelMetadata,
    ModelEnsemble, AdapterWrapper, MixedPrecisionWrapper,
    create_model_from_config, load_model_for_game,
    # Autocast utilities (moved from utils for better organization)
    safe_autocast, get_device_type
)
# Removed old nn_model imports - use resnet_model instead
from .resnet_model import ResNetModel, ResNetConfig, create_resnet_for_game
from .resnet_evaluator import (
    ResNetEvaluator, create_evaluator_for_game,
    create_chess_evaluator, create_go_evaluator, create_gomoku_evaluator
)
# MockEvaluator moved to tests directory - import from tests.mock_evaluator if needed

# Import unified modules last to avoid circular imports
# These are imported after the basic modules to prevent circular dependencies

__all__ = [
    # Framework
    "ModelRegistry",
    "ModelLoader",
    "BaseGameModel",
    "ModelMetadata",
    "ModelEnsemble",
    "AdapterWrapper",
    "MixedPrecisionWrapper",
    "create_model_from_config",
    "load_model_for_game",
    # Autocast utilities
    "safe_autocast",
    "get_device_type",
    # ResNet
    "ResNetModel",
    "ResNetConfig",
    "create_resnet_for_game",
    "ResNetEvaluator",
    "create_evaluator_for_game",
    "create_chess_evaluator",
    "create_go_evaluator",
    "create_gomoku_evaluator",
    # Unified training modules
    "UnifiedTrainingPipeline",
    "GameExample",
    "SelfPlayManager",
    "SelfPlayConfig",
    "ArenaManager",
    "ArenaConfig",
    "ELOTracker",
    # AlphaZero evaluator
    "AlphaZeroEvaluator",
    # MockEvaluator removed from production - available in tests if needed
]

# Import unified modules after __all__ to avoid circular imports
try:
    from .unified_training_pipeline import UnifiedTrainingPipeline, GameExample
    from .self_play_module import SelfPlayManager, SelfPlayConfig
    from .arena_module import ArenaManager, ArenaConfig, ELOTracker
except ImportError:
    # These modules require additional dependencies
    pass
    
# Import AlphaZeroEvaluator from core module
try:
    from ..core.evaluator import AlphaZeroEvaluator
except ImportError:
    # May not be available in all configurations
    pass