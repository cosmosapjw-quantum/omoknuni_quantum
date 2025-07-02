"""TensorRT-Accelerated Neural Network Evaluator

This module provides a TensorRT-accelerated evaluator that can be used as a drop-in
replacement for ResNetEvaluator with significantly improved inference performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union, List
from pathlib import Path
import logging
import warnings
import time

from mcts.core.evaluator import Evaluator, EvaluatorConfig
from .nn_framework import ModelLoader, BaseGameModel, ModelMetadata
from .resnet_evaluator import ResNetEvaluator
from .tensorrt_converter import TensorRTConverter, TensorRTModel, optimize_for_hardware, HAS_TENSORRT

logger = logging.getLogger(__name__)


class TensorRTEvaluator(Evaluator):
    """TensorRT-accelerated neural network evaluator for MCTS"""
    
    def __init__(
        self,
        model: Optional[BaseGameModel] = None,
        config: Optional[EvaluatorConfig] = None,
        game_type: str = 'gomoku',
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        network_config: Optional[Any] = None,
        tensorrt_engine_path: Optional[str] = None,
        fp16_mode: bool = True,
        int8_mode: bool = False,
        max_batch_size: int = 512,
        fallback_to_pytorch: bool = True,
        cache_engine: bool = True
    ):
        """
        Initialize TensorRT evaluator
        
        Args:
            model: Pre-loaded PyTorch model (optional)
            config: Evaluator configuration
            game_type: Type of game
            checkpoint_path: Path to PyTorch model checkpoint
            device: Device to run on
            network_config: Neural network architecture config
            tensorrt_engine_path: Path to pre-built TensorRT engine
            fp16_mode: Enable FP16 precision
            int8_mode: Enable INT8 precision (requires calibration)
            max_batch_size: Maximum batch size for optimization
            fallback_to_pytorch: Fall back to PyTorch if TensorRT fails
            cache_engine: Cache converted TensorRT engines
        """
        # Initialize with default action size for gomoku, will be updated after model loading
        action_size = 225 if game_type == 'gomoku' else 362 if game_type == 'go' else 4096
        super().__init__(config, action_size)
        
        if not HAS_TENSORRT:
            if fallback_to_pytorch:
                warnings.warn("TensorRT not available, falling back to PyTorch")
                self._fallback_evaluator = ResNetEvaluator(
                    model, config, game_type, checkpoint_path, 
                    device, network_config
                )
                self.use_tensorrt = False
                return
            else:
                raise RuntimeError("TensorRT not available and fallback disabled")
        
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT requires CUDA")
            
        self.game_type = game_type
        self.fp16_mode = fp16_mode
        self.int8_mode = int8_mode
        self.max_batch_size = max_batch_size
        self.cache_engine = cache_engine
        self.use_tensorrt = True
        
        # Load or create TensorRT model
        if tensorrt_engine_path and Path(tensorrt_engine_path).exists():
            # Load pre-built engine
            logger.info(f"Loading TensorRT engine from {tensorrt_engine_path}")
            self.trt_model = TensorRTConverter.load_engine(
                tensorrt_engine_path, 
                max_batch_size
            )
            self._setup_from_engine()
        else:
            # Convert from PyTorch model
            self._convert_from_pytorch(
                model, checkpoint_path, network_config, 
                tensorrt_engine_path
            )
    
    def _convert_from_pytorch(self, 
                             model: Optional[BaseGameModel],
                             checkpoint_path: Optional[str],
                             network_config: Optional[Any],
                             engine_save_path: Optional[str]):
        """Convert PyTorch model to TensorRT"""
        # First load PyTorch model using ResNetEvaluator
        temp_evaluator = ResNetEvaluator(
            model=model,
            config=self.config,
            game_type=self.game_type,
            checkpoint_path=checkpoint_path,
            device=str(self.device),
            network_config=network_config
        )
        
        self.pytorch_model = temp_evaluator.model
        self.action_size = temp_evaluator.action_size
        self.board_size = temp_evaluator.board_size
        
        # Get input shape from model
        if hasattr(self.pytorch_model, 'metadata'):
            input_channels = self.pytorch_model.metadata.input_channels
        else:
            input_channels = 18  # Default for Gomoku
            
        input_shape = (input_channels, self.board_size, self.board_size)
        
        # Check for cached engine
        if self.cache_engine and checkpoint_path:
            cache_path = self._get_engine_cache_path(checkpoint_path)
            if cache_path.exists():
                logger.info(f"Loading cached TensorRT engine from {cache_path}")
                try:
                    self.trt_model = TensorRTConverter.load_engine(
                        str(cache_path), 
                        self.max_batch_size
                    )
                    return
                except Exception as e:
                    logger.warning(f"Failed to load cached engine: {e}, converting again")
        
        # Convert to TensorRT
        logger.info("Converting PyTorch model to TensorRT...")
        start_time = time.time()
        
        if self.fp16_mode or self.int8_mode:
            # Use manual converter for advanced options
            converter = TensorRTConverter(
                fp16_mode=self.fp16_mode,
                int8_mode=self.int8_mode,
                max_batch_size=self.max_batch_size
            )
            self.trt_model = converter.convert_pytorch_to_tensorrt(
                self.pytorch_model,
                input_shape,
                engine_save_path
            )
        else:
            # Use automatic optimization
            self.trt_model = optimize_for_hardware(
                self.pytorch_model,
                input_shape
            )
        
        conversion_time = time.time() - start_time
        logger.info(f"TensorRT conversion completed in {conversion_time:.2f}s")
        
        # Cache engine if requested
        if self.cache_engine and checkpoint_path and engine_save_path is None:
            cache_path = self._get_engine_cache_path(checkpoint_path)
            logger.info(f"Caching TensorRT engine to {cache_path}")
            converter.save_engine(self.trt_model, str(cache_path))
    
    def _setup_from_engine(self):
        """Setup evaluator from loaded TensorRT engine"""
        # Infer game properties from engine
        # This is simplified - you may need to store metadata with the engine
        if self.game_type == 'gomoku':
            self.board_size = 15
            self.action_size = 225
        elif self.game_type == 'go':
            self.board_size = 19
            self.action_size = 362
        else:
            raise ValueError(f"Unknown game type: {self.game_type}")
    
    def _get_engine_cache_path(self, checkpoint_path: str) -> Path:
        """Get cache path for TensorRT engine"""
        checkpoint_path = Path(checkpoint_path)
        precision_suffix = 'fp16' if self.fp16_mode else 'fp32'
        if self.int8_mode:
            precision_suffix = 'int8'
        
        engine_name = f"{checkpoint_path.stem}_trt_{precision_suffix}.engine"
        return checkpoint_path.parent / engine_name
    
    def evaluate(self, 
                 state: Union[np.ndarray, torch.Tensor],
                 legal_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 temperature: float = 1.0) -> Tuple[np.ndarray, float]:
        """
        Evaluate a single game state using TensorRT
        
        Args:
            state: Game state tensor
            legal_mask: Mask of legal actions
            temperature: Temperature for policy (applied externally)
            
        Returns:
            Tuple of (policy, value)
        """
        if not self.use_tensorrt:
            return self._fallback_evaluator.evaluate(state, legal_mask, temperature)
        
        # Convert to tensor if needed
        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().to(self.device)
        else:
            state_tensor = state.float().to(self.device)
        
        # Add batch dimension if needed
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)
        
        # Run TensorRT inference
        with torch.no_grad():
            policy_logits, value = self.trt_model(state_tensor)
        
        # Apply softmax to get probabilities
        if self.trt_model.trt_model is None:
            # Manual inference doesn't apply log_softmax
            policy = F.softmax(policy_logits, dim=1)
        else:
            # torch2trt model returns log probabilities
            policy = policy_logits.exp()
        
        # Apply legal mask if provided
        if legal_mask is not None:
            if isinstance(legal_mask, np.ndarray):
                legal_mask_tensor = torch.from_numpy(legal_mask).bool().to(self.device)
            else:
                legal_mask_tensor = legal_mask.bool().to(self.device)
                
            if legal_mask_tensor.dim() == 1:
                legal_mask_tensor = legal_mask_tensor.unsqueeze(0)
                
            policy = policy * legal_mask_tensor.float()
            policy = policy / policy.sum(dim=1, keepdim=True)
        
        # Convert to numpy
        policy_np = policy[0].cpu().numpy()
        value_np = value[0].item()
        
        return policy_np, value_np
    
    def evaluate_batch(self,
                      states: Union[np.ndarray, torch.Tensor],
                      legal_masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
                      temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate a batch of game states using TensorRT
        
        Args:
            states: Batch of game states
            legal_masks: Batch of legal action masks
            temperature: Temperature for policy
            
        Returns:
            Tuple of (policies, values)
        """
        if not self.use_tensorrt:
            return self._fallback_evaluator.evaluate_batch(states, legal_masks, temperature)
        
        batch_size = states.shape[0]
        if batch_size > self.max_batch_size:
            # Process in chunks
            policies = []
            values = []
            
            for i in range(0, batch_size, self.max_batch_size):
                end = min(i + self.max_batch_size, batch_size)
                chunk_states = states[i:end]
                chunk_masks = legal_masks[i:end] if legal_masks is not None else None
                
                chunk_policies, chunk_values = self._evaluate_batch_chunk(
                    chunk_states, chunk_masks, temperature
                )
                
                policies.append(chunk_policies)
                values.append(chunk_values)
            
            return np.vstack(policies), np.concatenate(values)
        else:
            return self._evaluate_batch_chunk(states, legal_masks, temperature)
    
    def _evaluate_batch_chunk(self,
                             states: Union[np.ndarray, torch.Tensor],
                             legal_masks: Optional[Union[np.ndarray, torch.Tensor]],
                             temperature: float) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a single chunk that fits in max_batch_size"""
        # Convert to tensor
        if isinstance(states, np.ndarray):
            states_tensor = torch.from_numpy(states).float().to(self.device)
        else:
            states_tensor = states.float().to(self.device)
        
        # Run TensorRT inference
        with torch.no_grad():
            policy_logits, values = self.trt_model(states_tensor)
        
        # Apply softmax
        if self.trt_model.trt_model is None:
            policies = F.softmax(policy_logits, dim=1)
        else:
            policies = policy_logits.exp()
        
        # Apply legal masks if provided
        if legal_masks is not None:
            if isinstance(legal_masks, np.ndarray):
                legal_masks_tensor = torch.from_numpy(legal_masks).bool().to(self.device)
            else:
                legal_masks_tensor = legal_masks.bool().to(self.device)
            
            policies = policies * legal_masks_tensor.float()
            policies = policies / policies.sum(dim=1, keepdim=True)
        
        # Convert to numpy
        policies_np = policies.cpu().numpy()
        values_np = values.cpu().numpy().squeeze(-1)
        
        return policies_np, values_np
    
    def forward_batch(self,
                     states: torch.Tensor,
                     legal_masks: Optional[torch.Tensor] = None,
                     temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        PyTorch-compatible batch forward (returns tensors)
        
        This method is for compatibility with GPU service optimization
        """
        if not self.use_tensorrt:
            return self._fallback_evaluator.model.forward_batch(
                states, legal_masks, temperature
            )
        
        # Direct TensorRT inference
        with torch.no_grad():
            policy_logits, values = self.trt_model(states)
        
        # Apply softmax
        if self.trt_model.trt_model is None:
            policies = F.softmax(policy_logits, dim=1)
        else:
            policies = policy_logits.exp()
        
        # Apply legal masks
        if legal_masks is not None:
            policies = policies * legal_masks.float()
            policies = policies / policies.sum(dim=1, keepdim=True)
        
        return policies, values
    
    def benchmark_vs_pytorch(self, 
                            num_runs: int = 100, 
                            batch_sizes: List[int] = [1, 8, 32, 64, 128, 256]) -> Dict[str, Any]:
        """
        Benchmark TensorRT vs PyTorch performance
        
        Args:
            num_runs: Number of runs per batch size
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary with benchmark results
        """
        if not hasattr(self, 'pytorch_model'):
            logger.warning("PyTorch model not available for comparison")
            return {}
        
        results = {
            'tensorrt': {},
            'pytorch': {},
            'speedup': {}
        }
        
        for batch_size in batch_sizes:
            if batch_size > self.max_batch_size:
                continue
                
            # Benchmark TensorRT
            trt_stats = self.trt_model.benchmark(num_runs, batch_size)
            results['tensorrt'][batch_size] = trt_stats
            
            # Benchmark PyTorch
            pytorch_stats = self._benchmark_pytorch(
                self.pytorch_model, num_runs, batch_size
            )
            results['pytorch'][batch_size] = pytorch_stats
            
            # Calculate speedup
            speedup = pytorch_stats['mean_ms'] / trt_stats['mean_ms']
            results['speedup'][batch_size] = speedup
            
            logger.info(f"Batch size {batch_size}: "
                       f"PyTorch {pytorch_stats['mean_ms']:.2f}ms, "
                       f"TensorRT {trt_stats['mean_ms']:.2f}ms, "
                       f"Speedup: {speedup:.2f}x")
        
        return results
    
    def _benchmark_pytorch(self, model: nn.Module, num_runs: int, batch_size: int) -> Dict[str, float]:
        """Benchmark PyTorch model"""
        device = next(model.parameters()).device
        input_shape = self.trt_model.input_shape
        dummy_input = torch.randn(batch_size, *input_shape).to(device)
        
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(dummy_input)
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)
        
        times = np.array(times)
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'throughput': batch_size / (np.mean(times) / 1000),
            'batch_size': batch_size
        }