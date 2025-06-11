"""Pickle debugging utilities to trace CUDA tensor serialization"""

import pickle
import torch
import io
import logging
from typing import Any, Set, Tuple

logger = logging.getLogger(__name__)


class CUDAPickleTracer:
    """Traces and detects CUDA tensors during pickling"""
    
    def __init__(self):
        self.cuda_objects = []
        self.visited = set()
    
    def trace_object(self, obj: Any, path: str = "root") -> None:
        """Recursively trace object for CUDA tensors"""
        # Avoid circular references
        obj_id = id(obj)
        if obj_id in self.visited:
            return
        self.visited.add(obj_id)
        
        # Check if it's a tensor
        if torch.is_tensor(obj):
            if obj.is_cuda:
                self.cuda_objects.append({
                    'path': path,
                    'type': type(obj).__name__,
                    'shape': tuple(obj.shape),
                    'device': str(obj.device),
                    'dtype': str(obj.dtype)
                })
                logger.warning(f"[PICKLE_TRACE] Found CUDA tensor at {path}: shape={obj.shape}, device={obj.device}")
        
        # Recursively check containers
        elif isinstance(obj, dict):
            for k, v in obj.items():
                self.trace_object(v, f"{path}[{repr(k)}]")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                self.trace_object(v, f"{path}[{i}]")
        elif hasattr(obj, '__dict__'):
            # Check object attributes
            for attr, value in obj.__dict__.items():
                self.trace_object(value, f"{path}.{attr}")
    
    def get_report(self) -> str:
        """Get a report of found CUDA objects"""
        if not self.cuda_objects:
            return "No CUDA tensors found"
        
        report = f"Found {len(self.cuda_objects)} CUDA tensors:\n"
        for obj in self.cuda_objects:
            report += f"  - {obj['path']}: {obj['type']} {obj['shape']} on {obj['device']}\n"
        return report


class DebugPickler(pickle.Pickler):
    """Custom pickler that logs what's being pickled"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cuda_tensors_found = []
        
    def persistent_id(self, obj):
        """Called for every object during pickling"""
        if torch.is_tensor(obj) and obj.is_cuda:
            self.cuda_tensors_found.append({
                'type': type(obj).__name__,
                'shape': tuple(obj.shape),
                'device': str(obj.device),
                'id': id(obj)
            })
            logger.error(f"[PICKLE_DEBUG] Attempting to pickle CUDA tensor: {obj.shape} on {obj.device}")
        return None


def safe_pickle_dumps(obj: Any) -> bytes:
    """Safely pickle an object with CUDA detection"""
    tracer = CUDAPickleTracer()
    tracer.trace_object(obj)
    
    if tracer.cuda_objects:
        logger.error(f"[PICKLE_DEBUG] Pre-pickle CUDA detection:\n{tracer.get_report()}")
        raise RuntimeError(f"Cannot pickle object with CUDA tensors:\n{tracer.get_report()}")
    
    # Use custom pickler
    buffer = io.BytesIO()
    pickler = DebugPickler(buffer, protocol=pickle.HIGHEST_PROTOCOL)
    
    try:
        pickler.dump(obj)
        
        if pickler.cuda_tensors_found:
            logger.error(f"[PICKLE_DEBUG] CUDA tensors found during pickling: {pickler.cuda_tensors_found}")
            raise RuntimeError("CUDA tensors detected during pickling")
            
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"[PICKLE_DEBUG] Pickle failed: {e}")
        raise


def analyze_pickle_data(data: bytes) -> None:
    """Analyze pickled data for CUDA references"""
    import pickletools
    
    logger.info("[PICKLE_DEBUG] Analyzing pickle data...")
    
    # Look for CUDA-related strings in the pickle
    cuda_indicators = [b'cuda', b'CudaTensor', b'gpu', b'device']
    for indicator in cuda_indicators:
        if indicator in data:
            logger.warning(f"[PICKLE_DEBUG] Found '{indicator.decode()}' in pickle data")
    
    # Use pickletools to analyze
    try:
        ops = list(pickletools.dis(data))
        for op in ops:
            if 'cuda' in str(op).lower():
                logger.warning(f"[PICKLE_DEBUG] CUDA reference in pickle op: {op}")
    except Exception as e:
        logger.error(f"[PICKLE_DEBUG] Failed to analyze pickle: {e}")


def test_object_picklability(obj: Any, name: str = "object") -> bool:
    """Test if an object can be safely pickled without CUDA tensors"""
    logger.info(f"[PICKLE_DEBUG] Testing picklability of {name}")
    
    try:
        # First trace for CUDA tensors
        tracer = CUDAPickleTracer()
        tracer.trace_object(obj, name)
        
        if tracer.cuda_objects:
            logger.error(f"[PICKLE_DEBUG] {name} contains CUDA tensors:\n{tracer.get_report()}")
            return False
        
        # Try to pickle
        data = safe_pickle_dumps(obj)
        logger.info(f"[PICKLE_DEBUG] {name} can be pickled: {len(data)} bytes")
        
        # Analyze the pickle
        analyze_pickle_data(data)
        
        # Try to unpickle
        unpickled = pickle.loads(data)
        logger.info(f"[PICKLE_DEBUG] {name} unpickled successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"[PICKLE_DEBUG] {name} pickling failed: {e}")
        return False