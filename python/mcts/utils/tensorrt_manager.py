"""TensorRT Model Manager for Safe Multiprocessing

This module provides a centralized TensorRT model manager that ensures
TensorRT conversion happens only once and models are safely shared across
multiple worker processes.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import logging
import time
import pickle
import tempfile
import os
import fcntl
import hashlib
import threading

logger = logging.getLogger(__name__)

# Global TensorRT model cache
_TENSORRT_MODELS: Dict[str, Any] = {}
_TENSORRT_LOCK = threading.Lock()


class TensorRTManager:
    """Manages TensorRT models for safe multiprocessing"""
    
    # Class-level flags to track if messages have been logged
    _cache_message_logged = False
    _conversion_message_logged = False
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize TensorRT manager
        
        Args:
            cache_dir: Directory to cache TensorRT engines
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # Use system temp directory with app-specific folder
            self.cache_dir = Path(tempfile.gettempdir()) / "omoknuni_tensorrt_cache"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Lock file for multiprocess synchronization
        self.lock_file = self.cache_dir / ".tensorrt_lock"
        
    def get_model_hash(self, model: nn.Module, input_shape: Tuple[int, ...]) -> str:
        """Generate unique hash for model architecture and weights
        
        Args:
            model: PyTorch model
            input_shape: Input shape for the model
            
        Returns:
            Unique hash string
        """
        # Create hash from model architecture and input shape
        hasher = hashlib.sha256()
        
        # Add model class name
        hasher.update(model.__class__.__name__.encode())
        
        # Add input shape
        hasher.update(str(input_shape).encode())
        
        # Add model architecture info
        if hasattr(model, 'num_blocks'):
            hasher.update(f"blocks_{model.num_blocks}".encode())
        if hasattr(model, 'num_filters'):
            hasher.update(f"filters_{model.num_filters}".encode())
        if hasattr(model, 'board_size'):
            hasher.update(f"board_{model.board_size}".encode())
        if hasattr(model, 'action_size'):
            hasher.update(f"action_{model.action_size}".encode())
        
        # Add model state dict structure (keys and shapes)
        state_dict = model.state_dict()
        for key in sorted(state_dict.keys()):
            hasher.update(key.encode())
            hasher.update(str(state_dict[key].shape).encode())
        
        return hasher.hexdigest()[:16]  # Use first 16 chars
    
    def get_cached_engine_path(self, model_hash: str) -> Path:
        """Get path to cached TensorRT engine
        
        Args:
            model_hash: Unique model hash
            
        Returns:
            Path to cached engine file
        """
        return self.cache_dir / f"engine_{model_hash}.trt"
    
    def get_or_convert_model(self, 
                           pytorch_model: nn.Module,
                           input_shape: Tuple[int, ...],
                           batch_size: int = 1,
                           fp16: bool = True,
                           workspace_size: int = 1 << 30,
                           worker_id: int = 0) -> Optional[Any]:
        """Get TensorRT model, converting if necessary
        
        This method ensures thread-safe and process-safe TensorRT conversion.
        Only the first worker performs conversion, others wait and load the result.
        
        Args:
            pytorch_model: PyTorch model to convert
            input_shape: Input shape (C, H, W)
            batch_size: Batch size for optimization
            fp16: Enable FP16 precision
            workspace_size: TensorRT workspace size
            worker_id: Worker process ID
            
        Returns:
            TensorRT model or None if conversion fails
        """
        # Get unique model identifier
        model_hash = self.get_model_hash(pytorch_model, input_shape)
        
        # Check in-memory cache first
        with _TENSORRT_LOCK:
            if model_hash in _TENSORRT_MODELS:
                if not TensorRTManager._cache_message_logged:
                    logger.info(f"Worker {worker_id}: Using cached TensorRT model")
                    TensorRTManager._cache_message_logged = True
                else:
                    logger.debug(f"Worker {worker_id}: Using cached TensorRT model")
                return _TENSORRT_MODELS[model_hash]
        
        # Check file cache
        engine_path = self.get_cached_engine_path(model_hash)
        
        # Use file locking for multi-process synchronization
        lock_handle = open(self.lock_file, 'a+')
        
        try:
            # Try to acquire exclusive lock (only one process converts)
            try:
                fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                is_converter = True
                logger.info(f"Worker {worker_id}: Acquired lock for TensorRT conversion")
            except IOError:
                # Another process is converting, wait for it
                is_converter = False
                logger.info(f"Worker {worker_id}: Waiting for TensorRT conversion by another worker...")
                fcntl.flock(lock_handle, fcntl.LOCK_SH)  # Wait for shared lock
            if is_converter and not engine_path.exists():
                # This worker performs the conversion
                logger.info(f"Worker {worker_id}: Starting TensorRT conversion...")
                
                try:
                    # Import here to avoid issues if TensorRT not installed
                    from mcts.neural_networks.tensorrt_converter import TensorRTConverter
                    
                    converter = TensorRTConverter(
                        workspace_size=workspace_size,
                        fp16_mode=fp16,
                        max_batch_size=batch_size
                    )
                    
                    start_time = time.time()
                    
                    # Convert model
                    trt_model = converter.convert_pytorch_to_tensorrt(
                        pytorch_model,
                        input_shape,
                        output_path=str(engine_path)
                    )
                    
                    conversion_time = time.time() - start_time
                    logger.info(f"Worker {worker_id}: TensorRT conversion completed in {conversion_time:.2f}s")
                    
                    # Cache in memory
                    with _TENSORRT_LOCK:
                        _TENSORRT_MODELS[model_hash] = trt_model
                    
                    return trt_model
                    
                except Exception as e:
                    logger.error(f"Worker {worker_id}: TensorRT conversion failed: {e}")
                    # Clean up failed conversion
                    if engine_path.exists():
                        engine_path.unlink()
                    return None
            
            else:
                # Load existing engine
                if engine_path.exists():
                    if not TensorRTManager._cache_message_logged:
                        logger.info(f"Worker {worker_id}: Loading TensorRT engine from cache...")
                        TensorRTManager._cache_message_logged = True
                    else:
                        logger.debug(f"Worker {worker_id}: Loading TensorRT engine from cache...")
                    
                    try:
                        # Import here to avoid issues if TensorRT not installed
                        from mcts.neural_networks.tensorrt_converter import TensorRTModel
                        
                        # Load pre-converted engine
                        trt_model = TensorRTModel.load_engine(
                            str(engine_path),
                            input_shape=input_shape,
                            max_batch_size=batch_size
                        )
                        
                        # Cache in memory
                        with _TENSORRT_LOCK:
                            _TENSORRT_MODELS[model_hash] = trt_model
                        
                        logger.debug(f"Worker {worker_id}: Successfully loaded TensorRT engine")
                        return trt_model
                        
                    except Exception as e:
                        logger.error(f"Worker {worker_id}: Failed to load TensorRT engine: {e}")
                        return None
                else:
                    logger.error(f"Worker {worker_id}: TensorRT engine not found after conversion")
                    return None
                    
        finally:
            # Release lock and close file
            fcntl.flock(lock_handle, fcntl.LOCK_UN)
            lock_handle.close()
    
    def clear_cache(self):
        """Clear TensorRT engine cache"""
        # Clear memory cache
        with _TENSORRT_LOCK:
            _TENSORRT_MODELS.clear()
        
        # Clear file cache
        for engine_file in self.cache_dir.glob("engine_*.trt"):
            try:
                engine_file.unlink()
                logger.info(f"Removed cached engine: {engine_file}")
            except Exception as e:
                logger.warning(f"Failed to remove {engine_file}: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached engines
        
        Returns:
            Dictionary with cache statistics
        """
        engine_files = list(self.cache_dir.glob("engine_*.trt"))
        total_size = sum(f.stat().st_size for f in engine_files)
        
        return {
            'cache_dir': str(self.cache_dir),
            'num_engines': len(engine_files),
            'total_size_mb': total_size / (1024 * 1024),
            'in_memory_models': len(_TENSORRT_MODELS),
            'engine_files': [f.name for f in engine_files]
        }


# Global instance
_tensorrt_manager: Optional[TensorRTManager] = None


def get_tensorrt_manager(cache_dir: Optional[str] = None) -> TensorRTManager:
    """Get global TensorRT manager instance
    
    Args:
        cache_dir: Optional cache directory
        
    Returns:
        TensorRT manager instance
    """
    global _tensorrt_manager
    
    if _tensorrt_manager is None:
        _tensorrt_manager = TensorRTManager(cache_dir)
    
    return _tensorrt_manager