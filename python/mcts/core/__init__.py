"""Core MCTS components"""

# Import non-torch modules immediately
from .game_interface import GameInterface, GameType

# Lazy imports for torch-dependent modules
_evaluator_module = None
_mcts_module = None
_mcts_config_module = None
_wave_search_module = None
_tree_operations_module = None

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

def _lazy_import_mcts_config():
    """Lazy import mcts_config module to avoid importing torch in workers"""
    global _mcts_config_module
    if _mcts_config_module is None:
        from . import mcts_config as _mcts_config_module
    return _mcts_config_module

def _lazy_import_wave_search():
    """Lazy import wave_search module to avoid importing torch in workers"""
    global _wave_search_module
    if _wave_search_module is None:
        from . import wave_search as _wave_search_module
    return _wave_search_module

def _lazy_import_tree_operations():
    """Lazy import tree_operations module to avoid importing torch in workers"""
    global _tree_operations_module
    if _tree_operations_module is None:
        from . import tree_operations as _tree_operations_module
    return _tree_operations_module

# Define lazy attribute access
def __getattr__(name):
    # List of attributes from evaluator module
    evaluator_attrs = ['Evaluator', 'EvaluatorConfig', 
                       'AlphaZeroEvaluator']
    
    # List of attributes from mcts module
    mcts_attrs = ['MCTS']
    
    # List of attributes from mcts_config module
    mcts_config_attrs = ['MCTSConfig']
    
    # List of attributes from wave_search module
    wave_search_attrs = ['WaveSearch']
    
    # List of attributes from tree_operations module
    tree_operations_attrs = ['TreeOperations']
    
    if name in evaluator_attrs:
        module = _lazy_import_evaluator()
        return getattr(module, name)
    elif name in mcts_attrs:
        module = _lazy_import_mcts()
        return getattr(module, name)
    elif name in mcts_config_attrs:
        module = _lazy_import_mcts_config()
        return getattr(module, name)
    elif name in wave_search_attrs:
        module = _lazy_import_wave_search()
        return getattr(module, name)
    elif name in tree_operations_attrs:
        module = _lazy_import_tree_operations()
        return getattr(module, name)
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "GameInterface",
    "GameType",
    "Evaluator",
    "EvaluatorConfig",
    "AlphaZeroEvaluator",
    "MCTS",
    "MCTSConfig",
    "WaveSearch",
    "TreeOperations",
]