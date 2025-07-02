"""Utility components"""

# Import non-torch modules first
from .config_manager import ConfigManager, OptimizedConfig
from .config_system import AlphaZeroConfig, MCTSFullConfig, QuantumLevel, create_default_config, merge_configs

# Worker init must be importable without torch
from .worker_init import init_worker_process, verify_cuda_disabled, get_cpu_device, ensure_cpu_tensor, create_cpu_only_config

# Hardware optimization (uses psutil but not torch)
from .hardware_optimizer import HardwareOptimizer, HardwareProfile, OptimalResourceAllocation, optimize_for_hardware

# Lazy imports for torch-dependent modules to avoid importing torch in workers
def _lazy_import_torch_modules():
    """Import torch-dependent modules only when needed"""
    global serialize_state_dict_for_multiprocessing, deserialize_state_dict_from_multiprocessing
    global GPUEvaluatorService, RemoteEvaluator
    
    from .safe_multiprocessing import serialize_state_dict_for_multiprocessing, deserialize_state_dict_from_multiprocessing
    # Force fresh import to get updated GPUEvaluatorService
    import importlib
    gpu_module = importlib.import_module('.gpu_evaluator_service', __name__)
    GPUEvaluatorService = gpu_module.GPUEvaluatorService
    RemoteEvaluator = gpu_module.RemoteEvaluator

# Initialize as None - will be imported on first use
serialize_state_dict_for_multiprocessing = None
deserialize_state_dict_from_multiprocessing = None
GPUEvaluatorService = None
RemoteEvaluator = None

# Override __getattr__ to lazy load torch modules
def __getattr__(name):
    if name in ['serialize_state_dict_for_multiprocessing', 'deserialize_state_dict_from_multiprocessing', 
                'GPUEvaluatorService', 'RemoteEvaluator']:
        if globals()[name] is None:
            _lazy_import_torch_modules()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ConfigManager",
    "OptimizedConfig",
    "AlphaZeroConfig",
    "MCTSFullConfig",
    "QuantumLevel",
    "create_default_config",
    "merge_configs",
    "serialize_state_dict_for_multiprocessing",
    "deserialize_state_dict_from_multiprocessing",
    "GPUEvaluatorService",
    "RemoteEvaluator",
    "init_worker_process",
    "verify_cuda_disabled",
    "get_cpu_device",
    "ensure_cpu_tensor",
    "create_cpu_only_config",
    "HardwareOptimizer",
    "HardwareProfile",
    "OptimalResourceAllocation",
    "optimize_for_hardware",
]