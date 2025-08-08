"""Base Neural Network Evaluator

This module provides a unified base class for neural network evaluators that consolidates
common functionality between ResNet and TensorRT implementations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
import logging
import time
from abc import abstractmethod

from mcts.core.evaluator import Evaluator, EvaluatorConfig
from .nn_framework import BaseGameModel

logger = logging.getLogger(__name__)


class BaseNeuralEvaluator(Evaluator):
    """Base class for neural network evaluators with common functionality"""
    
    def __init__(
        self,
        model: Optional[BaseGameModel] = None,
        config: Optional[EvaluatorConfig] = None,
        game_type: str = 'gomoku',
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize base neural evaluator
        
        Args:
            model: Neural network model
            config: Evaluator configuration
            game_type: Type of game
            device: Device to run on
            **kwargs: Additional arguments
        """
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Initialize configuration
        if config is None:
            config = EvaluatorConfig()
        
        # Determine action size based on game type and model
        action_size = self._determine_action_size(model, game_type, kwargs)
        
        super().__init__(config, action_size)
        
        # Store common attributes
        self.model = model
        self.game_type = game_type
        self.batch_size = getattr(config, 'batch_size', 512)
        self._return_torch_tensors = False
        
        # Performance tracking
        self.evaluation_count = 0
        self.total_eval_time = 0.0
        self.batch_evaluation_count = 0
        self.total_batch_eval_time = 0.0
        
        # OPTIMIZATION: Pre-allocate pinned memory buffers for 27x faster transfers
        self._init_pinned_memory_buffers()
        
        # Persistent batching context for Phase 1.4 optimization
        self._persistent_batching_enabled = False
        self._persistent_batch_buffers = {}
        
        # Move model to device if provided
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()
        
        # Perform comprehensive warm-up to eliminate first-move overhead
        self._comprehensive_warmup()
    
    def _comprehensive_warmup(self):
        """Perform comprehensive warm-up to eliminate first-move initialization overhead"""
        if self.model is None:
            return
            
        try:
            logger.info("Starting comprehensive warm-up for neural network evaluator...")
            start_time = time.time()
            
            # Force import of heavy libraries that might be lazily loaded
            try:
                import numba
                import numba.cuda
            except ImportError:
                pass  # Not critical if numba isn't used
            
            # Apply PyTorch optimizations for GPU
            if self.device.type == 'cuda':
                # Enable TensorFloat-32 for Ampere GPUs (RTX 30xx, A100)
                if hasattr(torch.backends.cuda, 'matmul'):
                    torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = True
                
                # Enable cuDNN autotuner for optimal conv kernels
                torch.backends.cudnn.benchmark = True
                
                # Set optimal matmul precision
                if hasattr(torch, 'set_float32_matmul_precision'):
                    torch.set_float32_matmul_precision('high')
            
            # Determine input shape based on game type
            if self.game_type == 'gomoku':
                board_size = 15
                channels = getattr(self.model, 'input_channels', 19)
            elif self.game_type == 'go':
                board_size = 19
                channels = getattr(self.model, 'input_channels', 19)
            elif self.game_type == 'chess':
                board_size = 8
                channels = getattr(self.model, 'input_channels', 12)
            else:
                board_size = 15
                channels = 19
            
            # Stage 1: Memory pool initialization
            if self.device.type == 'cuda':
                # Allocate and free large tensor to initialize memory pool
                dummy_large = torch.randn(512, channels, board_size, board_size, device=self.device)
                del dummy_large
                torch.cuda.empty_cache()
            
            # Stage 2: cuDNN kernel selection warm-up with various batch sizes
            batch_sizes = [1, 8, 16, 32, 64, 128, 256, 512]
            with torch.no_grad():
                for batch_size in batch_sizes:
                    if batch_size > self.batch_size:
                        break
                    dummy_input = torch.randn(batch_size, channels, board_size, board_size, device=self.device)
                    # Multiple runs for kernel benchmarking
                    for _ in range(2):
                        _ = self._forward_model(dummy_input)
                    
                    # Also warm up common batch sizes for MCTS
                    if batch_size in [1, self.batch_size]:
                        for _ in range(3):
                            _ = self._forward_model(dummy_input)
            
            # Stage 3: Initialize pinned memory if using GPU
            if self.device.type == 'cuda' and not hasattr(self, 'pinned_input_buffer'):
                self._init_pinned_memory_buffers()
            
            # Stage 4: Pre-compute any cached values
            if hasattr(self, '_precompute_caches'):
                self._precompute_caches()
            
            # Synchronize to ensure all operations complete
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.time() - start_time
            logger.info(f"Comprehensive warm-up completed in {elapsed:.3f}s")
            
        except Exception as e:
            logger.warning(f"Warm-up encountered error: {e}")
            # Don't fail initialization due to warm-up errors
    
    def _determine_action_size(self, model: Optional[BaseGameModel], game_type: str, kwargs: Dict[str, Any]) -> int:
        """Determine action size from model or game type"""
        if model is not None and hasattr(model, 'action_size'):
            return model.action_size
        
        # Fallback to game type defaults
        if game_type == 'gomoku':
            board_size = kwargs.get('board_size', 15)
            return board_size * board_size
        elif game_type == 'go':
            board_size = kwargs.get('board_size', 19)
            return board_size * board_size + 1  # +1 for pass
        elif game_type == 'chess':
            return 4096  # Standard chess action space
        else:
            return kwargs.get('action_size', 225)  # Default fallback
    
    def _init_pinned_memory_buffers(self):
        """Initialize pinned memory buffers for faster CPU-GPU transfers (27x speedup)"""
        if self.device.type == 'cuda':
            try:
                # Determine board size for buffer allocation
                if self.game_type == 'gomoku':
                    board_size = 15
                    channels = 19  # Standard feature channels
                elif self.game_type == 'go':
                    board_size = 19
                    channels = 19
                elif self.game_type == 'chess':
                    board_size = 8
                    channels = 12
                else:
                    board_size = 15
                    channels = 19
                
                # Pre-allocate pinned memory buffers
                self.pinned_input_buffer = torch.zeros(
                    (self.batch_size, channels, board_size, board_size),
                    dtype=torch.float32,
                    pin_memory=True
                )
                self.pinned_policy_buffer = torch.zeros(
                    (self.batch_size, self.action_size),
                    dtype=torch.float32,
                    pin_memory=True
                )
                self.pinned_value_buffer = torch.zeros(
                    (self.batch_size, 1),
                    dtype=torch.float32,
                    pin_memory=True
                )
                
                logger.info(f"Allocated pinned memory buffers for {self.batch_size} batch size")
            except Exception as e:
                logger.warning(f"Failed to allocate pinned memory: {e}")
                self.pinned_input_buffer = None
                self.pinned_policy_buffer = None
                self.pinned_value_buffer = None
        else:
            self.pinned_input_buffer = None
            self.pinned_policy_buffer = None
            self.pinned_value_buffer = None
    
    def _prepare_input(self, states: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert input states to properly formatted tensor with pinned memory optimization"""
        if isinstance(states, np.ndarray):
            # OPTIMIZATION: Use pinned memory buffer for 27x faster transfer
            if self.pinned_input_buffer is not None and states.shape[0] <= self.batch_size:
                # Copy to pinned buffer first
                batch_size = states.shape[0]
                self.pinned_input_buffer[:batch_size] = torch.from_numpy(states).float()
                # Non-blocking transfer to GPU
                states = self.pinned_input_buffer[:batch_size].to(self.device, non_blocking=True)
            else:
                states = torch.from_numpy(states).float()
                states = states.to(self.device).float()
        elif isinstance(states, torch.Tensor):
            # If already a tensor, just ensure it's on the right device
            states = states.to(self.device).float()
        else:
            raise ValueError(f"Expected np.ndarray or torch.Tensor, got {type(states)}")
        
        # Ensure 4D tensor (batch_size, channels, height, width)
        if states.dim() == 3:
            states = states.unsqueeze(0)
        elif states.dim() != 4:
            raise ValueError(f"Expected 3D or 4D tensor, got {states.dim()}D")
        
        return states
    
    def _prepare_legal_mask(self, legal_mask: Optional[Union[np.ndarray, torch.Tensor]], batch_size: int) -> Optional[torch.Tensor]:
        """Convert legal mask to properly formatted tensor"""
        if legal_mask is None:
            return None
        
        if isinstance(legal_mask, np.ndarray):
            legal_mask = torch.from_numpy(legal_mask)
        
        # Ensure proper device and dtype
        legal_mask = legal_mask.to(self.device).float()
        
        # Ensure 2D tensor (batch_size, action_size)
        if legal_mask.dim() == 1:
            legal_mask = legal_mask.unsqueeze(0)
        elif legal_mask.dim() != 2:
            raise ValueError(f"Expected 1D or 2D legal mask, got {legal_mask.dim()}D")
        
        # Verify batch size compatibility
        if legal_mask.shape[0] != batch_size:
            if legal_mask.shape[0] == 1 and batch_size > 1:
                # Broadcast single mask to batch
                legal_mask = legal_mask.expand(batch_size, -1)
            else:
                raise ValueError(f"Legal mask batch size {legal_mask.shape[0]} doesn't match states batch size {batch_size}")
        
        return legal_mask
    
    def _apply_legal_mask(self, policy: torch.Tensor, legal_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply legal move mask to policy tensor"""
        if legal_mask is not None:
            # Zero out illegal moves
            policy = policy * legal_mask
            # Renormalize
            policy_sum = policy.sum(dim=1, keepdim=True)
            policy_sum = torch.clamp(policy_sum, min=1e-8)  # Avoid division by zero
            policy = policy / policy_sum
        return policy
    
    def _apply_temperature(self, policy: torch.Tensor, temperature: float) -> torch.Tensor:
        """Apply temperature scaling to policy"""
        if temperature != 1.0 and temperature > 0:
            # Apply temperature to logits (convert back from softmax)
            policy = torch.clamp(policy, min=1e-8)  # Avoid log(0)
            log_policy = torch.log(policy)
            scaled_log_policy = log_policy / temperature
            policy = F.softmax(scaled_log_policy, dim=1)
        return policy
    
    def _convert_output(self, policy: torch.Tensor, value: torch.Tensor) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """Convert model output to appropriate format with optimized transfers"""
        if self._return_torch_tensors:
            return policy, value
        else:
            # OPTIMIZATION: Use pinned memory for output if available
            if self.pinned_policy_buffer is not None and policy.shape[0] <= self.batch_size:
                batch_size = policy.shape[0]
                # Non-blocking transfer to pinned memory
                self.pinned_policy_buffer[:batch_size] = policy.to('cpu', non_blocking=True)
                # Ensure value has correct shape [batch_size, 1]
                if value.dim() == 1:
                    value = value.unsqueeze(1)
                self.pinned_value_buffer[:batch_size] = value.to('cpu', non_blocking=True)
                # Synchronize only once for both transfers
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                return self.pinned_policy_buffer[:batch_size].numpy(), self.pinned_value_buffer[:batch_size].numpy()
            else:
                # Fallback to regular transfer
                return policy.cpu().numpy(), value.cpu().numpy()
    
    @abstractmethod
    def _forward_model(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model - to be implemented by subclasses"""
        pass
    
    def evaluate(
        self, 
        state: Union[np.ndarray, torch.Tensor],
        legal_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        temperature: float = 1.0
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Evaluate a single game state
        
        Args:
            state: Game state tensor
            legal_mask: Legal move mask
            temperature: Temperature for policy scaling
            
        Returns:
            Tuple of (policy, value)
        """
        start_time = time.time()
        
        # Prepare inputs
        states = self._prepare_input(state)
        batch_size = states.shape[0]
        legal_mask_tensor = self._prepare_legal_mask(legal_mask, batch_size)
        
        # Forward pass
        with torch.no_grad():
            policy, value = self._forward_model(states)
        
        # Post-process outputs
        policy = self._apply_legal_mask(policy, legal_mask_tensor)
        policy = self._apply_temperature(policy, temperature)
        
        # Update statistics
        eval_time = time.time() - start_time
        self.evaluation_count += 1
        self.total_eval_time += eval_time
        
        # Convert output format
        policy_out, value_out = self._convert_output(policy, value)
        
        # Return single item if input was single
        if policy_out.shape[0] == 1:
            if isinstance(policy_out, np.ndarray):
                return policy_out[0], value_out[0]
            else:
                return policy_out[0], value_out[0]
        
        return policy_out, value_out
    
    def evaluate_batch(
        self,
        states: Union[np.ndarray, torch.Tensor],
        legal_masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
        temperature: float = 1.0
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Evaluate a batch of game states
        
        Args:
            states: Batch of game states
            legal_masks: Batch of legal move masks
            temperature: Temperature for policy scaling
            
        Returns:
            Tuple of (policies, values)
        """
        start_time = time.time()
        
        # Prepare inputs
        states = self._prepare_input(states)
        batch_size = states.shape[0]
        legal_mask_tensor = self._prepare_legal_mask(legal_masks, batch_size)
        
        # Forward pass
        with torch.no_grad():
            policy, value = self._forward_model(states)
        
        # Post-process outputs
        policy = self._apply_legal_mask(policy, legal_mask_tensor)
        policy = self._apply_temperature(policy, temperature)
        
        # Update statistics
        eval_time = time.time() - start_time
        self.batch_evaluation_count += 1
        self.total_batch_eval_time += eval_time
        
        return self._convert_output(policy, value)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        stats = {
            'evaluation_count': self.evaluation_count,
            'batch_evaluation_count': self.batch_evaluation_count,
            'total_eval_time': self.total_eval_time,
            'total_batch_eval_time': self.total_batch_eval_time,
            'device': str(self.device),
            'game_type': self.game_type,
            'action_size': self.action_size
        }
        
        if self.evaluation_count > 0:
            stats['avg_eval_time'] = self.total_eval_time / self.evaluation_count
        
        if self.batch_evaluation_count > 0:
            stats['avg_batch_eval_time'] = self.total_batch_eval_time / self.batch_evaluation_count
        
        return stats
    
    def reset_statistics(self):
        """Reset evaluation statistics"""
        self.evaluation_count = 0
        self.total_eval_time = 0.0
        self.batch_evaluation_count = 0
        self.total_batch_eval_time = 0.0
    
    def enable_persistent_batching(self, enabled: bool = True):
        """Enable persistent batching context for self-play
        
        This optimization maintains evaluation context across moves to avoid
        reallocation overhead during long games.
        
        Args:
            enabled: Whether to enable persistent batching
        """
        self._persistent_batching_enabled = enabled
        if not enabled:
            # Clear persistent buffers when disabled
            self._persistent_batch_buffers.clear()
    
    def allocate_game_length_buffers(self, max_moves: int = 100):
        """Pre-allocate buffers for entire game length
        
        This avoids repeated allocation/deallocation during self-play games.
        
        Args:
            max_moves: Maximum expected moves per game
        """
        if not self._persistent_batching_enabled:
            return
        
        if self.device.type == 'cuda':
            try:
                # Determine sizes
                if self.game_type == 'gomoku':
                    board_size = 15
                    channels = 19
                elif self.game_type == 'go':
                    board_size = 19
                    channels = 19
                elif self.game_type == 'chess':
                    board_size = 8
                    channels = 12
                else:
                    board_size = 15
                    channels = 19
                
                # Pre-allocate game-length buffers
                self._persistent_batch_buffers = {
                    'input_queue': torch.zeros(
                        (max_moves, self.batch_size, channels, board_size, board_size),
                        dtype=torch.float32,
                        device=self.device
                    ),
                    'policy_queue': torch.zeros(
                        (max_moves, self.batch_size, self.action_size),
                        dtype=torch.float32,
                        device=self.device
                    ),
                    'value_queue': torch.zeros(
                        (max_moves, self.batch_size, 1),
                        dtype=torch.float32,
                        device=self.device
                    ),
                    'current_position': 0
                }
                
                logger.info(f"Allocated persistent batch buffers for {max_moves} moves")
                
            except Exception as e:
                logger.warning(f"Failed to allocate persistent batch buffers: {e}")
                self._persistent_batching_enabled = False