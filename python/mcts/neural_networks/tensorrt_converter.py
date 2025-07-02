"""TensorRT Model Converter for AlphaZero ResNet

This module provides utilities to convert PyTorch ResNet models to TensorRT
for accelerated inference.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
import logging
import tempfile
import warnings

# TensorRT imports with error handling
try:
    import tensorrt as trt
    import torch2trt
    from torch2trt import torch2trt as convert_to_trt
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False
    warnings.warn("TensorRT not available. Install with: pip install tensorrt torch2trt")

logger = logging.getLogger(__name__)


class TensorRTConverter:
    """Converts PyTorch ResNet models to TensorRT format"""
    
    def __init__(self, 
                 workspace_size: int = 1 << 30,  # 1GB workspace
                 fp16_mode: bool = True,
                 int8_mode: bool = False,
                 max_batch_size: int = 512,
                 calibration_data: Optional[np.ndarray] = None):
        """
        Initialize TensorRT converter
        
        Args:
            workspace_size: Maximum workspace size for TensorRT optimization
            fp16_mode: Enable FP16 precision (recommended for RTX GPUs)
            int8_mode: Enable INT8 precision (requires calibration)
            max_batch_size: Maximum batch size to optimize for
            calibration_data: Calibration data for INT8 mode
        """
        if not HAS_TENSORRT:
            raise RuntimeError("TensorRT not available. Please install tensorrt and torch2trt")
            
        self.workspace_size = workspace_size
        self.fp16_mode = fp16_mode
        self.int8_mode = int8_mode
        self.max_batch_size = max_batch_size
        self.calibration_data = calibration_data
        
        # TensorRT logger
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        
    def convert_pytorch_to_tensorrt(self, 
                                   pytorch_model: nn.Module,
                                   input_shape: Tuple[int, ...],
                                   output_path: Optional[str] = None) -> 'TensorRTModel':
        """
        Convert PyTorch model to TensorRT
        
        Args:
            pytorch_model: PyTorch model to convert
            input_shape: Input shape (channels, height, width)
            output_path: Path to save TensorRT engine (optional)
            
        Returns:
            TensorRTModel wrapper
        """
        pytorch_model.eval()
        device = next(pytorch_model.parameters()).device
        
        # Create example input
        dummy_input = torch.randn(1, *input_shape).to(device)
        
        logger.info(f"Converting model to TensorRT with input shape: {input_shape}")
        logger.info(f"FP16 mode: {self.fp16_mode}, INT8 mode: {self.int8_mode}")
        
        # Convert using torch2trt with suppressed warnings
        with torch.no_grad():
            # Create a custom logger to suppress TensorRT warnings
            warning_logger = trt.Logger(trt.Logger.ERROR)
            
            trt_model = convert_to_trt(
                pytorch_model,
                [dummy_input],
                fp16_mode=self.fp16_mode,
                int8_mode=self.int8_mode,
                max_batch_size=self.max_batch_size,
                max_workspace_size=self.workspace_size,
                log_level=trt.Logger.WARNING  # Keep at WARNING level for important messages
            )
        
        logger.info("TensorRT conversion completed successfully")
        
        # Save engine if path provided
        if output_path:
            self.save_engine(trt_model, output_path)
            
        return TensorRTModel(trt_model, input_shape, self.max_batch_size)
    
    def convert_with_dynamic_shapes(self,
                                   pytorch_model: nn.Module,
                                   min_shape: Tuple[int, ...],
                                   opt_shape: Tuple[int, ...],
                                   max_shape: Tuple[int, ...]) -> 'TensorRTModel':
        """
        Convert with dynamic batch sizes for more flexibility
        
        Args:
            pytorch_model: PyTorch model to convert
            min_shape: Minimum input shape (with batch)
            opt_shape: Optimal input shape (with batch) 
            max_shape: Maximum input shape (with batch)
            
        Returns:
            TensorRTModel with dynamic shapes
        """
        # This requires more advanced TensorRT APIs
        # For now, we'll use fixed shapes and log a warning
        logger.warning("Dynamic shapes not fully implemented, using optimal shape")
        return self.convert_pytorch_to_tensorrt(
            pytorch_model, 
            opt_shape[1:],  # Remove batch dimension
            None
        )
    
    def save_engine(self, trt_model, path: str):
        """Save TensorRT engine to file"""
        engine_bytes = trt_model.engine.serialize()
        with open(path, 'wb') as f:
            f.write(engine_bytes)
        logger.info(f"TensorRT engine saved to {path}")
    
    @staticmethod
    def load_engine(path: str, max_batch_size: int = 512) -> 'TensorRTModel':
        """Load TensorRT engine from file"""
        if not HAS_TENSORRT:
            raise RuntimeError("TensorRT not available")
            
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
            
        if engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {path}")
            
        # Create execution context
        context = engine.create_execution_context()
        
        # Infer input shape from engine
        input_shape = engine.get_binding_shape(0)[1:]  # Remove batch dimension
        
        logger.info(f"Loaded TensorRT engine from {path}")
        return TensorRTModel(engine, input_shape, max_batch_size, context)


class TensorRTModel:
    """Wrapper for TensorRT model inference"""
    
    def __init__(self, 
                 trt_model_or_engine,
                 input_shape: Tuple[int, ...],
                 max_batch_size: int,
                 context=None,
                 cuda_stream=None):
        """
        Initialize TensorRT model wrapper
        
        Args:
            trt_model_or_engine: torch2trt model or TensorRT engine
            input_shape: Input shape without batch dimension
            max_batch_size: Maximum batch size
            context: Execution context (for loaded engines)
            cuda_stream: CUDA stream for async execution (optional)
        """
        self.input_shape = input_shape
        self.max_batch_size = max_batch_size
        self.cuda_stream = cuda_stream or torch.cuda.current_stream()
        
        if hasattr(trt_model_or_engine, 'engine'):
            # torch2trt model
            self.trt_model = trt_model_or_engine
            self.engine = trt_model_or_engine.engine
            self.context = trt_model_or_engine.context
        else:
            # Raw TensorRT engine
            self.trt_model = None
            self.engine = trt_model_or_engine
            self.context = context or trt_model_or_engine.create_execution_context()
    
    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (policy_logits, value)
        """
        if self.trt_model is not None:
            # Use torch2trt model (handles everything)
            return self.trt_model(x)
        else:
            # Manual TensorRT inference
            return self._manual_inference(x)
    
    def _manual_inference(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Manual TensorRT inference for loaded engines"""
        batch_size = x.shape[0]
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds maximum {self.max_batch_size}")
        
        # Prepare bindings
        bindings = []
        outputs = []
        
        # Input binding
        x_ptr = x.contiguous().data_ptr()
        bindings.append(x_ptr)
        
        # Output bindings (assuming 2 outputs: policy and value)
        # You may need to adjust based on your model
        policy_shape = (batch_size, 225)  # For 15x15 Gomoku
        value_shape = (batch_size, 1)
        
        policy_output = torch.empty(policy_shape, dtype=torch.float32, device=x.device)
        value_output = torch.empty(value_shape, dtype=torch.float32, device=x.device)
        
        bindings.append(policy_output.data_ptr())
        bindings.append(value_output.data_ptr())
        
        # Run inference using non-default CUDA stream
        # Use execute_async_v2 with specified CUDA stream to avoid synchronization warnings
        success = self.context.execute_async_v2(
            bindings, 
            stream_handle=self.cuda_stream.cuda_stream
        )
        
        if not success:
            # Fallback to synchronous execution if async fails
            self.context.execute_v2(bindings)
        
        # Synchronize the stream to ensure completion
        self.cuda_stream.synchronize()
        
        return policy_output, value_output
    
    def benchmark(self, num_runs: int = 100, batch_size: int = 32) -> Dict[str, float]:
        """
        Benchmark TensorRT model performance
        
        Args:
            num_runs: Number of inference runs
            batch_size: Batch size for benchmarking
            
        Returns:
            Dictionary with timing statistics
        """
        device = torch.device('cuda')
        dummy_input = torch.randn(batch_size, *self.input_shape).to(device)
        
        # Warmup
        for _ in range(10):
            _ = self(dummy_input)
        
        torch.cuda.synchronize()
        
        # Benchmark
        import time
        times = []
        
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self(dummy_input)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        times = np.array(times)
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'throughput': batch_size / (np.mean(times) / 1000),  # items/sec
            'batch_size': batch_size
        }


def optimize_for_hardware(pytorch_model: nn.Module,
                         input_shape: Tuple[int, ...],
                         target_gpu: str = 'auto') -> TensorRTModel:
    """
    Automatically optimize model for target hardware
    
    Args:
        pytorch_model: PyTorch model to optimize
        input_shape: Input shape (C, H, W)
        target_gpu: Target GPU or 'auto' to detect
        
    Returns:
        Optimized TensorRT model
    """
    if target_gpu == 'auto':
        # Detect GPU
        gpu_name = torch.cuda.get_device_name(0).lower()
        if 'rtx' in gpu_name or 'a100' in gpu_name or 'v100' in gpu_name:
            fp16_mode = True
            logger.info(f"Detected {gpu_name}, enabling FP16 mode")
        else:
            fp16_mode = False
            logger.info(f"Detected {gpu_name}, using FP32 mode")
    else:
        fp16_mode = 'rtx' in target_gpu.lower() or 'a100' in target_gpu.lower()
    
    converter = TensorRTConverter(
        fp16_mode=fp16_mode,
        int8_mode=False,  # INT8 requires calibration
        max_batch_size=1024  # Increased to support larger training batches
    )
    
    return converter.convert_pytorch_to_tensorrt(pytorch_model, input_shape)