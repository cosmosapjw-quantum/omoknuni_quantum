"""Safe model loading utilities for multiprocessing environments"""

import torch
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def safe_create_model(game_type: str, **kwargs):
    """Safely create a model in multiprocessing context
    
    This function ensures proper initialization and avoids memory corruption
    issues that can occur with PyTorch models in spawned processes.
    """
    # Ensure we're in a clean state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Workers can use CUDA for tree operations
    # Only disable CUDA if explicitly requested
    # if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Import here to ensure clean import in worker
    from mcts.neural_networks.nn_model import create_model
    
    # Create model with no_grad to avoid autograd issues
    with torch.no_grad():
        model = create_model(game_type=game_type, **kwargs)
    
    # Ensure model is in eval mode
    model.eval()
    
    # Force CPU mode
    model = model.cpu()
    
    return model


def safe_load_state_dict(model, state_dict: Dict[str, Any], strict: bool = True):
    """Safely load state dict into model
    
    This ensures proper handling of state dict loading in multiprocessing context.
    """
    # Ensure all tensors are on CPU
    cpu_state_dict = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value) and value.is_cuda:
            cpu_state_dict[key] = value.cpu()
        else:
            cpu_state_dict[key] = value
    
    # Load with no_grad to avoid autograd issues
    with torch.no_grad():
        model.load_state_dict(cpu_state_dict, strict=strict)
    
    return model


def worker_create_and_load_model(serialized_state_dict: Dict[str, Any], 
                                 game_type: str, 
                                 model_kwargs: Dict[str, Any],
                                 worker_id: int):
    """Worker function that safely creates and loads a model"""
    import traceback
    
    try:
        logger.debug(f"[Worker {worker_id}] Starting safe model creation")
        
        # Ensure clean environment
        # Workers can use CUDA for tree operations
        # os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['OMP_NUM_THREADS'] = '1'  # Prevent OpenMP issues
        
        # Import modules in worker
        import torch
        torch.set_num_threads(1)  # Prevent threading issues in worker
        
        from mcts.utils.safe_multiprocessing import deserialize_state_dict_from_multiprocessing
        
        logger.debug(f"[Worker {worker_id}] Creating model")
        model = safe_create_model(game_type=game_type, **model_kwargs)
        
        logger.debug(f"[Worker {worker_id}] Deserializing state dict")
        state_dict = deserialize_state_dict_from_multiprocessing(serialized_state_dict)
        
        logger.debug(f"[Worker {worker_id}] Loading state dict")
        model = safe_load_state_dict(model, state_dict)
        
        # Test forward pass
        logger.debug(f"[Worker {worker_id}] Testing forward pass")
        input_channels = model_kwargs.get('input_channels', 20)
        input_height = model_kwargs.get('input_height', 9)
        input_width = model_kwargs.get('input_width', 9)
        
        dummy_input = torch.randn(1, input_channels, input_height, input_width)
        with torch.no_grad():
            policy, value = model(dummy_input)
        
        result = f"Worker {worker_id}: Model loaded successfully, policy shape={policy.shape}, value shape={value.shape}"
        logger.debug(f"[Worker {worker_id}] Success: {result}")
        return result
        
    except Exception as e:
        error_msg = f"Worker {worker_id} failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return error_msg