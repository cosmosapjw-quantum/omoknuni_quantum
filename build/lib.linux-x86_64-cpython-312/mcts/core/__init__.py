"""Core MCTS components"""

# Import non-torch modules immediately
from .game_interface import GameInterface, GameType

# Lazy imports for torch-dependent modules
_evaluator_module = None
_mcts_module = None

def _lazy_import_evaluator():
    """Lazy import evaluator module to avoid importing torch in workers"""
    global _evaluator_module
    if _evaluator_module is None:
        from . import evaluator as _evaluator_module
    return _evaluator_module

def _lazy_import_mcts():
    """Lazy import mcts module to avoid importing torch in workers"""
    global _mcts_module
    if _mcts_module is None:
        from . import mcts as _mcts_module
    return _mcts_module

# Define lazy attribute access
def __getattr__(name):
    # List of attributes from evaluator module
    evaluator_attrs = ['Evaluator', 'MockEvaluator', 'EvaluatorConfig', 
                       'AlphaZeroEvaluator', 'RandomEvaluator']
    
    # List of attributes from mcts module
    mcts_attrs = ['MCTS', 'MCTSConfig']
    
    if name in evaluator_attrs:
        module = _lazy_import_evaluator()
        return getattr(module, name)
    elif name in mcts_attrs:
        module = _lazy_import_mcts()
        return getattr(module, name)
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "GameInterface",
    "GameType",
    "Evaluator",
    "MockEvaluator",
    "RandomEvaluator",
    "EvaluatorConfig",
    "AlphaZeroEvaluator",
    "MCTS",
    "MCTSConfig",
]