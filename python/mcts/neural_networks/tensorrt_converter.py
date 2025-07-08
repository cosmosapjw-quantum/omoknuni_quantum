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
    
    # Class-level flag to track if messages have been logged
    _load_message_logged = False
    
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
        self.max_batch_size = max_batch_size if max_batch_size is not None else 512
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
        
        # Determine game type from input shape
        channels, height, width = input_shape
        if height == 15 and width == 15:
            game_type = "gomoku"
            policy_size = 225
        elif height == 19 and width == 19:
            game_type = "go"
            policy_size = 362  # 19x19 + 1 pass
        elif height == 8 and width == 8:
            game_type = "chess"
            policy_size = 4096  # All possible chess moves
        else:
            game_type = f"custom_{height}x{width}"
            policy_size = height * width
            
        logger.info(f"Converting {game_type} model (policy_size={policy_size})")
        
        # Create example inputs for multiple batch sizes to enable dynamic batching
        # Include the actual batch sizes we use during training
        batch_sizes_to_optimize = [1, 8, 32, 64, 128, 256, 512]
        example_inputs = []
        
        for bs in batch_sizes_to_optimize:
            if bs <= self.max_batch_size:
                example_inputs.append(torch.randn(bs, *input_shape).to(device))
        
        logger.debug(f"Converting model to TensorRT with input shape: {input_shape}")
        logger.debug(f"FP16 mode: {self.fp16_mode}, INT8 mode: {self.int8_mode}")
        logger.debug(f"Optimizing for batch sizes: {[x.shape[0] for x in example_inputs]}")
        
        # Convert using torch2trt with suppressed warnings
        with torch.no_grad():
            # Temporarily redirect TensorRT logs to suppress warnings
            import os
            import sys
            from contextlib import redirect_stderr
            from io import StringIO
            
            # Capture TensorRT warnings
            captured_output = StringIO()
            
            with redirect_stderr(captured_output):
                # Also suppress stdout warnings
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                
                try:
                    # torch2trt expects specific parameter names
                    trt_model = convert_to_trt(
                        pytorch_model,
                        [example_inputs[-1]],  # Use largest batch size for better optimization
                        fp16_mode=self.fp16_mode,
                        int8_mode=self.int8_mode,
                        max_batch_size=self.max_batch_size,
                        max_workspace_size=self.workspace_size,
                        log_level=trt.Logger.ERROR  # Suppress all but errors
                    )
                finally:
                    # Restore stdout
                    sys.stdout = old_stdout
        
        logger.debug("TensorRT conversion completed successfully")
        
        # Create TensorRT model wrapper
        tensorrt_model = TensorRTModel(trt_model, input_shape, self.max_batch_size)
        
        # Save engine if path provided
        if output_path:
            tensorrt_model.save_engine(output_path)
            
        return tensorrt_model
    
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
        if hasattr(trt_model, 'save_engine'):
            # If it's a TensorRTModel instance, use its save method
            trt_model.save_engine(path)
        else:
            # Otherwise handle raw engine
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
        
        # Get number of bindings to determine output shapes
        num_bindings = 0
        binding_names = []
        
        # Handle different TensorRT API versions
        if hasattr(engine, 'num_io_tensors'):
            # TensorRT 8.5+ API
            num_bindings = engine.num_io_tensors
            for i in range(num_bindings):
                binding_names.append(engine.get_tensor_name(i))
        elif hasattr(engine, 'num_bindings'):
            # Older TensorRT API
            num_bindings = engine.num_bindings
            for i in range(num_bindings):
                binding_names.append(engine.get_binding_name(i))
        else:
            # Fallback - assume standard 3 bindings (input, policy, value)
            num_bindings = 3
            binding_names = ['input', 'policy', 'value']
        
        # Get input shape - handle different API versions
        try:
            if hasattr(engine, 'get_tensor_shape'):
                # New API
                input_shape = list(engine.get_tensor_shape(binding_names[0]))
            else:
                # Old API
                input_shape = list(engine.get_binding_shape(0))
        except Exception as e:
            logger.warning(f"Could not infer input shape from engine: {e}")
            # Default shape for Gomoku
            input_shape = [-1, 18, 15, 15]
        
        # Remove batch dimension if present
        if len(input_shape) == 4:
            # Remove the batch dimension (first dimension)
            input_shape = tuple(input_shape[1:])
        elif len(input_shape) > 4:
            # Something is wrong
            logger.warning(f"Unexpected input shape dimensions: {input_shape}")
            input_shape = tuple(input_shape[-3:])  # Take last 3 dimensions
        else:
            input_shape = tuple(input_shape)
        
        if not TensorRTConverter._load_message_logged:
            logger.info(f"Loaded TensorRT engine from {path} with input shape {input_shape}")
            logger.info(f"Engine has {num_bindings} bindings: {binding_names}")
            TensorRTConverter._load_message_logged = True
        else:
            logger.debug(f"Loaded TensorRT engine from {path} with input shape {input_shape}")
            logger.debug(f"Engine has {num_bindings} bindings: {binding_names}")
        
        # Log warning if binding names don't match expected format
        if binding_names and len(binding_names) >= 3:
            if not any(name.startswith('input') for name in binding_names[:1]):
                logger.warning(f"First binding '{binding_names[0]}' doesn't start with 'input' - may need adjustment")
            if not any(name.startswith('output') for name in binding_names[1:]):
                logger.warning(f"Output bindings {binding_names[1:]} don't start with 'output' - may need adjustment")
        
        # Create a CUDA stream for TensorRT execution
        cuda_stream = torch.cuda.Stream()
        
        return TensorRTModel(engine, input_shape, max_batch_size, context, cuda_stream=cuda_stream, binding_names=binding_names)


class TensorRTModel:
    """Wrapper for TensorRT model inference"""
    
    def __init__(self, 
                 trt_model_or_engine,
                 input_shape: Tuple[int, ...],
                 max_batch_size: int,
                 context=None,
                 cuda_stream=None,
                 binding_names=None):
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
        # Create a dedicated CUDA stream for TensorRT execution if not provided
        self.cuda_stream = cuda_stream if cuda_stream is not None else torch.cuda.Stream()
        self.binding_names = binding_names or ['input', 'policy', 'value']
        
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
            
        # Determine output shapes based on game type (inferred from input shape)
        if len(self.input_shape) == 3:
            channels, height, width = self.input_shape
        elif len(self.input_shape) == 4:
            # Has batch dimension, skip it
            _, channels, height, width = self.input_shape
        else:
            # Unexpected shape
            logger.warning(f"Unexpected input shape: {self.input_shape}")
            # Default to Gomoku
            channels, height, width = 18, 15, 15
            
        # Determine policy size based on board dimensions
        if height == 15 and width == 15:
            # Gomoku
            self.policy_size = 225
            self.game_type = "gomoku"
        elif height == 19 and width == 19:
            # Go (19x19 + 1 pass move)
            self.policy_size = 362
            self.game_type = "go"
        elif height == 8 and width == 8:
            # Chess (64 squares * 64 destinations = 4096 possible moves)
            self.policy_size = 4096
            self.game_type = "chess"
        elif height == 9 and width == 9:
            # Small board (for testing)
            self.policy_size = 81
            self.game_type = "small_board"
        else:
            # Custom board size
            self.policy_size = height * width
            self.game_type = f"custom_{height}x{width}"
            
        logger.debug(f"TensorRT Model initialized for {self.game_type} with policy size {self.policy_size}")
    
    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (policy_logits, value)
        """
        if self.trt_model is not None:
            # Use torch2trt model with custom stream
            with torch.cuda.stream(self.cuda_stream):
                result = self.trt_model(x)
            return result
        else:
            # Manual TensorRT inference
            return self._manual_inference(x)
    
    def _manual_inference(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Manual TensorRT inference for loaded engines"""
        batch_size = x.shape[0]
        if self.max_batch_size is not None and batch_size > self.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds maximum {self.max_batch_size}")
        
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Create output tensors based on detected policy size
        policy_shape = (batch_size, self.policy_size)
        value_shape = (batch_size, 1)
        
        policy_output = torch.empty(policy_shape, dtype=torch.float32, device=x.device)
        value_output = torch.empty(value_shape, dtype=torch.float32, device=x.device)
        
        # Skip dynamic shape setting since our engine has fixed batch dimensions
        # The engine was compiled with specific batch sizes and cannot be changed at runtime
        
        # Prepare bindings based on API version
        if hasattr(self.context, 'set_tensor_address'):
            # New TensorRT API (8.5+)
            try:
                # Use the actual binding names from the engine
                # TensorRT often uses names like 'input_0', 'output_0', 'output_1'
                # We need to use the exact names the engine expects
                if len(self.binding_names) >= 3:
                    input_name = self.binding_names[0]
                    policy_name = self.binding_names[1]
                    value_name = self.binding_names[2]
                else:
                    # Fallback to common TensorRT naming convention
                    input_name = 'input_0' if 'input_0' in self.binding_names else self.binding_names[0]
                    policy_name = 'output_0' if 'output_0' in self.binding_names else self.binding_names[1] if len(self.binding_names) > 1 else 'output_0'
                    value_name = 'output_1' if 'output_1' in self.binding_names else self.binding_names[2] if len(self.binding_names) > 2 else 'output_1'
                
                # Set input shape first (required for v3 API)
                if hasattr(self.context, 'set_input_shape'):
                    self.context.set_input_shape(input_name, x.shape)
                
                # Set addresses for all tensors
                self.context.set_tensor_address(input_name, x.data_ptr())
                self.context.set_tensor_address(policy_name, policy_output.data_ptr())
                self.context.set_tensor_address(value_name, value_output.data_ptr())
                
                # Execute asynchronously
                if hasattr(self.context, 'execute_async_v3'):
                    # Get the CUDA stream handle
                    # PyTorch CUDA stream objects have a cuda_stream property that returns the handle
                    stream_handle = self.cuda_stream.cuda_stream
                    
                    success = self.context.execute_async_v3(stream_handle=stream_handle)
                    if not success:
                        raise RuntimeError("TensorRT execute_async_v3 failed")
                else:
                    # Fallback to v2
                    success = self.context.execute_v2([x.data_ptr(), policy_output.data_ptr(), value_output.data_ptr()])
                    if not success:
                        raise RuntimeError("TensorRT execute_v2 failed")
                    
            except Exception as e:
                logger.error(f"TensorRT execution failed with new API: {e}")
                # Fall back to old API
                return self._manual_inference_old_api(x, policy_output, value_output)
        else:
            # Old API fallback
            return self._manual_inference_old_api(x, policy_output, value_output)
        
        # Synchronize the stream
        if hasattr(self.cuda_stream, 'synchronize'):
            self.cuda_stream.synchronize()
        else:
            torch.cuda.synchronize()
        
        return policy_output, value_output
    
    def _manual_inference_old_api(self, x: torch.Tensor, policy_output: torch.Tensor, value_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fallback for older TensorRT API"""
        # Prepare bindings list
        bindings = []
        bindings.append(x.data_ptr())
        bindings.append(policy_output.data_ptr())
        bindings.append(value_output.data_ptr())
        
        # Execute
        if hasattr(self.context, 'execute_async_v2'):
            # Async execution - use the proper CUDA stream handle
            stream_handle = self.cuda_stream.cuda_stream
            success = self.context.execute_async_v2(bindings, stream_handle=stream_handle)
            if hasattr(self.cuda_stream, 'synchronize'):
                self.cuda_stream.synchronize()
            else:
                torch.cuda.synchronize()
        else:
            # Sync execution
            success = self.context.execute_v2(bindings)
            
        if not success:
            raise RuntimeError("TensorRT execution failed")
            
        return policy_output, value_output
    
    def forward_batch(self, states: torch.Tensor, legal_masks: Optional[torch.Tensor] = None, 
                     temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GPU-native batch forward compatible with ResNet interface
        
        Args:
            states: Batch of game states (already on GPU)
            legal_masks: Optional legal action masks
            temperature: Temperature for policy (applied internally)
            
        Returns:
            Tuple of (policies, values) tensors
        """
        # Run inference
        policy_logits, values = self(states)
        
        # Apply temperature if needed
        if temperature != 1.0:
            policy_logits = policy_logits / temperature
        
        # Convert to probabilities
        policies = torch.softmax(policy_logits, dim=1)
        
        # Apply legal masks if provided
        if legal_masks is not None:
            policies = policies * legal_masks.float()
            policies = policies / policies.sum(dim=1, keepdim=True)
        
        return policies, values
    
    @classmethod
    def load_engine(cls, engine_path: str, input_shape: Tuple[int, ...] = (18, 15, 15), 
                   max_batch_size: int = 512) -> 'TensorRTModel':
        """Load TensorRT engine from file
        
        Args:
            engine_path: Path to saved TensorRT engine
            input_shape: Input shape (C, H, W)
            max_batch_size: Maximum batch size
            
        Returns:
            TensorRTModel instance
        """
        import tensorrt as trt
        
        # Create TensorRT runtime
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        
        # Load engine from file
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        engine = runtime.deserialize_cuda_engine(engine_data)
        if engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")
        
        # Create execution context
        context = engine.create_execution_context()
        
        # Create CUDA stream
        cuda_stream = torch.cuda.Stream()
        
        # Get binding names from engine
        binding_names = []
        if hasattr(engine, 'num_io_tensors'):
            # TensorRT 8.5+ API
            for i in range(engine.num_io_tensors):
                binding_names.append(engine.get_tensor_name(i))
        elif hasattr(engine, 'num_bindings'):
            # Older TensorRT API
            for i in range(engine.num_bindings):
                binding_names.append(engine.get_binding_name(i))
        else:
            # Fallback
            binding_names = ['input_0', 'output_0', 'output_1']
            
        # Create wrapper using loaded engine
        return cls(
            trt_model_or_engine=engine,
            input_shape=input_shape,
            max_batch_size=max_batch_size,
            context=context,
            cuda_stream=cuda_stream,
            binding_names=binding_names
        )
    
    def save_engine(self, output_path: str):
        """Save TensorRT engine to file
        
        Args:
            output_path: Path to save engine
        """
        if hasattr(self.trt_model, 'engine'):
            # torch2trt model
            engine = self.trt_model.engine
        elif hasattr(self, 'engine'):
            # Direct engine
            engine = self.engine
        else:
            raise RuntimeError("No TensorRT engine found to save")
        
        # Serialize engine
        engine_bytes = engine.serialize()
        
        # Write to file
        with open(output_path, 'wb') as f:
            f.write(engine_bytes)
        
        logger.info(f"Saved TensorRT engine to {output_path}")
    
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
            logger.debug(f"Detected {gpu_name}, enabling FP16 mode")
        else:
            fp16_mode = False
            logger.debug(f"Detected {gpu_name}, using FP32 mode")
    else:
        fp16_mode = 'rtx' in target_gpu.lower() or 'a100' in target_gpu.lower()
    
    converter = TensorRTConverter(
        fp16_mode=fp16_mode,
        int8_mode=False,  # INT8 requires calibration
        max_batch_size=1024  # Increased to support larger training batches
    )
    
    return converter.convert_pytorch_to_tensorrt(pytorch_model, input_shape)