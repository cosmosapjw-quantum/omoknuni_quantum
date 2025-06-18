"""Neural network evaluator interface for MCTS

This module provides the interface for neural network evaluation and
a mock implementation for testing without a trained model.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
import numpy
import numpy as np
import time
from collections import OrderedDict

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

# Import GPU kernels if available
try:
    from ..gpu.cuda_kernels import OptimizedCUDAKernels
    HAS_GPU_KERNELS = True
    CUDA_AVAILABLE = torch.cuda.is_available() if HAS_TORCH else False
except ImportError:
    HAS_GPU_KERNELS = False
    CUDA_AVAILABLE = False

# Import GPU components if available
try:
    from ..gpu.gpu_optimizer import create_gpu_accelerated_evaluator, BatchedTensorOperations, GPUMemoryPool
    HAS_GPU_COMPONENTS = True
except ImportError:
    create_gpu_accelerated_evaluator = None
    BatchedTensorOperations = None
    GPUMemoryPool = None
    HAS_GPU_COMPONENTS = False


@dataclass
class EvaluatorConfig:
    """Configuration for neural network evaluator
    
    Attributes:
        batch_size: Maximum batch size for evaluation
        device: Device to run on ('cpu' or 'cuda')
        num_channels: Number of channels in ResNet
        num_blocks: Number of ResNet blocks
        use_fp16: Whether to use mixed precision
        cache_size: Maximum number of positions to cache
    """
    batch_size: int = 512
    device: str = 'cuda' if HAS_TORCH and torch.cuda.is_available() else 'cpu'
    num_channels: int = 256
    num_blocks: int = 20
    use_fp16: bool = False
    cache_size: int = 10000


class Evaluator(ABC):
    """Abstract base class for neural network evaluators"""
    
    def __init__(self, config: EvaluatorConfig, action_size: int):
        """Initialize evaluator
        
        Args:
            config: Evaluator configuration
            action_size: Size of action space
        """
        self.config = config
        self.action_size = action_size
        self.cache_enabled = False
        self.cache: OrderedDict[bytes, Tuple[np.ndarray, float]] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        
    @abstractmethod
    def evaluate(
        self, 
        state: np.ndarray,
        legal_mask: Optional[np.ndarray] = None,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """Evaluate a single position
        
        Args:
            state: Board state array of shape (channels, height, width)
            legal_mask: Boolean mask for legal actions
            temperature: Temperature for policy scaling
            
        Returns:
            policy: Probability distribution over actions
            value: Position evaluation in [-1, 1]
        """
        pass
        
    @abstractmethod
    def evaluate_batch(
        self,
        states: np.ndarray,
        legal_masks: Optional[np.ndarray] = None,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch of positions
        
        Args:
            states: Batch of states (batch, channels, height, width)
            legal_masks: Batch of legal move masks (batch, action_size)
            temperature: Temperature for policy scaling
            
        Returns:
            policies: Batch of policy distributions (batch, action_size)
            values: Batch of position values (batch,)
        """
        pass
        
    def enable_cache(self, max_size: int = 10000) -> None:
        """Enable position caching
        
        Args:
            max_size: Maximum number of positions to cache
        """
        self.cache_enabled = True
        self.config.cache_size = max_size
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        
    def clear_cache(self) -> None:
        """Clear the position cache"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        
    def warmup(self, num_iterations: int = 10) -> None:
        """Warmup the evaluator with dummy evaluations
        
        Args:
            num_iterations: Number of warmup iterations
        """
        # Create dummy states
        dummy_state = np.random.rand(20, 19, 19).astype(np.float32)
        dummy_batch = np.random.rand(32, 20, 19, 19).astype(np.float32)
        
        # Run evaluations
        for _ in range(num_iterations):
            self.evaluate(dummy_state)
            self.evaluate_batch(dummy_batch)
            

class RandomEvaluator(Evaluator):
    """Pure random evaluator for baseline comparisons
    
    Returns uniform random policy and random value estimates.
    Used as ELO anchor with rating 0.
    """
    
    def __init__(self, config: EvaluatorConfig, action_size: int):
        """Initialize random evaluator"""
        super().__init__(config, action_size)
        self.eval_count = 0
        
    def evaluate(
        self, 
        state: np.ndarray,
        legal_mask: Optional[np.ndarray] = None,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """Return uniform random policy over legal moves"""
        # Create uniform policy
        policy = np.ones(self.action_size, dtype=np.float32) / self.action_size
        
        # Apply legal move mask if provided
        if legal_mask is not None:
            policy[~legal_mask] = 0
            legal_sum = policy.sum()
            if legal_sum > 0:
                policy = policy / legal_sum
            else:
                # No legal moves - shouldn't happen
                policy = np.ones(self.action_size) / self.action_size
        
        # Random value between -1 and 1
        value = np.random.uniform(-1, 1)
        
        self.eval_count += 1
        return policy, value
    
    def evaluate_batch(
        self,
        states: np.ndarray,
        legal_masks: Optional[np.ndarray] = None,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate batch with random policies"""
        batch_size = states.shape[0]
        policies = np.ones((batch_size, self.action_size), dtype=np.float32) / self.action_size
        
        # Apply legal masks if provided
        if legal_masks is not None:
            for i in range(batch_size):
                policies[i][~legal_masks[i]] = 0
                legal_sum = policies[i].sum()
                if legal_sum > 0:
                    policies[i] = policies[i] / legal_sum
        
        # Random values
        values = np.random.uniform(-1, 1, size=batch_size)
        
        self.eval_count += batch_size
        return policies, values


class MockEvaluator(Evaluator):
    """Mock evaluator for testing without a trained model
    
    Generates plausible policy and value outputs for testing MCTS.
    """
    
    def __init__(self, config: EvaluatorConfig, action_size: int):
        super().__init__(config, action_size)
        self.eval_count = 0
        
    def evaluate(
        self,
        state: np.ndarray,
        legal_mask: Optional[np.ndarray] = None,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """Evaluate a single position with mock outputs"""
        # Check cache if enabled
        if self.cache_enabled:
            cache_key = state.tobytes()
            if cache_key in self.cache:
                self.cache_hits += 1
                self.cache.move_to_end(cache_key)  # LRU update
                return self.cache[cache_key]
            self.cache_misses += 1
            
        # Generate mock policy
        # Convert torch tensor to numpy if needed
        if HAS_TORCH and torch.is_tensor(state):
            state_np = state.cpu().numpy()
        else:
            state_np = state
            
        # Use position-dependent but deterministic base values
        # Ensure we're using numpy's sum, not torch's
        base_logits = np.array(state_np).sum(axis=(0, 1)) % 7 - 3.5
        
        # Reshape to action space
        if len(base_logits) < self.action_size:
            # Pad with zeros
            logits = np.zeros(self.action_size)
            logits[:len(base_logits)] = base_logits.flatten()
        else:
            # Take first action_size elements
            logits = base_logits.flatten()[:self.action_size]
            
        # Add some noise for diversity
        noise = np.random.randn(self.action_size) * 0.1
        logits += noise
        
        # Apply temperature
        if temperature != 1.0:
            logits /= temperature
            
        # Apply legal move mask if provided
        if legal_mask is not None:
            logits[~legal_mask] = -1e9
            
        # Softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))
        policy = exp_logits / exp_logits.sum()
        
        # Generate mock value
        # Use center control and material balance as heuristic
        # Make sure we're working with numpy arrays
        if HAS_TORCH and torch.is_tensor(state):
            state_arr = state.cpu().numpy()
        else:
            state_arr = np.array(state)
            
        center_control = np.sum(state_arr[:, 7:12, 7:12]) / (5 * 5 * state_arr.shape[0])
        
        # Avoid division by zero for material balance
        total_material = np.sum(state_arr)
        if total_material > 0:
            material_balance = (np.sum(state_arr[:state_arr.shape[0]//2]) - 
                              np.sum(state_arr[state_arr.shape[0]//2:])) / total_material
        else:
            material_balance = 0.0
        
        value = float(np.tanh(center_control + material_balance + np.random.randn() * 0.1))
        
        # Cache result if enabled
        if self.cache_enabled:
            self.cache[cache_key] = (policy, value)
            # Evict oldest if cache full
            if len(self.cache) > self.config.cache_size:
                self.cache.popitem(last=False)
                
        self.eval_count += 1
        return policy, value
        
    def evaluate_batch(
        self,
        states: np.ndarray,
        legal_masks: Optional[np.ndarray] = None,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch of positions"""
        batch_size = states.shape[0]
        
        # Check if we should use GPU batch processing
        if self.config.device == 'cuda' and batch_size > 16:
            try:
                import torch
                device = torch.device('cuda')
                
                # Convert to tensors
                states_tensor = torch.from_numpy(states).float().to(device)
                
                # Generate batch policies on GPU
                dtype = torch.float16 if self.config.use_fp16 else torch.float32
                batch_logits = torch.randn(batch_size, self.action_size, device=device, dtype=dtype)
                
                # Apply temperature
                if temperature != 1.0:
                    batch_logits = batch_logits / temperature
                    
                # Apply legal masks if provided
                if legal_masks is not None:
                    masks_tensor = torch.from_numpy(legal_masks).bool().to(device)
                    batch_logits[~masks_tensor] = -1e9
                    
                # Softmax to get policies
                policies_tensor = torch.softmax(batch_logits, dim=-1)
                
                # Generate values
                values_tensor = torch.tanh(torch.randn(batch_size, device=device, dtype=dtype))
                
                # Convert back to numpy
                policies = policies_tensor.cpu().numpy()
                values = values_tensor.cpu().numpy()
                
                self.eval_count += batch_size
                return policies, values
                
            except Exception:
                pass  # Fall through to CPU version
        
        # CPU fallback
        policies = np.zeros((batch_size, self.action_size))
        values = np.zeros(batch_size)
        
        # Process each position
        for i in range(batch_size):
            legal_mask = legal_masks[i] if legal_masks is not None else None
            policies[i], values[i] = self.evaluate(
                states[i], legal_mask, temperature
            )
            
        return policies, values
        

class GPUAcceleratedEvaluator(Evaluator):
    """GPU-accelerated neural network evaluator
    
    Uses custom CUDA kernels and optimizations for high-throughput evaluation.
    """
    
    def __init__(self, model, config: EvaluatorConfig, action_size: int):
        """Initialize GPU-accelerated evaluator
        
        Args:
            model: Neural network model
            config: Evaluator configuration
            action_size: Size of action space
        """
        super().__init__(config, action_size)
        
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for GPU acceleration")
            
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")
            
        # Optimize model for GPU inference
        if create_gpu_accelerated_evaluator and HAS_GPU_COMPONENTS:
            self.model = create_gpu_accelerated_evaluator(model, config)
        else:
            self.model = model
            self.model.to(config.device)
            self.model.eval()
        
        # Initialize GPU components
        self.device = torch.device(config.device)
        self.gpu_ops = BatchedTensorOperations(self.device) if BatchedTensorOperations and HAS_GPU_KERNELS else None
        self.memory_pool = GPUMemoryPool(device=self.device) if GPUMemoryPool and HAS_GPU_KERNELS else None
        
        # Batch processing queue
        self.batch_queue = []
        self.batch_size = config.batch_size
        
        # Performance tracking
        self.gpu_time = 0.0
        self.cpu_time = 0.0
        self.eval_count = 0
        
    def evaluate(
        self,
        state: np.ndarray,
        legal_mask: Optional[np.ndarray] = None,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """Evaluate single position using GPU acceleration"""
        # Convert to batch format
        states = np.expand_dims(state, axis=0)
        masks = np.expand_dims(legal_mask, axis=0) if legal_mask is not None else None
        
        # Evaluate batch
        policies, values = self.evaluate_batch(states, masks, temperature)
        
        return policies[0], values[0]
        
    def evaluate_batch(
        self,
        states: np.ndarray,
        legal_masks: Optional[np.ndarray] = None,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate batch of positions with GPU acceleration"""
        start_time = time.time()
        
        # Convert to tensors
        if self.gpu_ops:
            # Use optimized encoding
            state_tensor = self.gpu_ops.batch_encode_states(list(states))
        else:
            # Standard conversion
            state_tensor = torch.from_numpy(states).float().to(self.device)
            
        # Get tensor from memory pool if available
        if self.memory_pool:
            output_shape = (states.shape[0], self.action_size)
            policy_tensor = self.memory_pool.get_tensor(output_shape)
            value_tensor = self.memory_pool.get_tensor((states.shape[0], 1))
        
        # GPU inference
        gpu_start = time.time()
        with torch.no_grad():
            if hasattr(self.model, '__call__'):
                # Direct model call
                policy_logits, values = self.model(state_tensor)
            else:
                # Standard forward pass
                policy_logits, values = self.model.forward(state_tensor)
                
        # Synchronize GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        self.gpu_time += time.time() - gpu_start
        
        # Apply temperature and legal moves
        if temperature != 1.0:
            policy_logits = policy_logits / temperature
            
        if legal_masks is not None:
            mask_tensor = torch.from_numpy(legal_masks).bool().to(self.device)
            policy_logits = policy_logits.masked_fill(~mask_tensor, -float('inf'))
            
        # Softmax to get probabilities
        policies = F.softmax(policy_logits, dim=1)
        
        # Convert back to numpy
        policies_np = policies.cpu().numpy()
        values_np = values.cpu().numpy().squeeze(1)
        
        # Return tensors to pool
        if self.memory_pool and 'policy_tensor' in locals():
            self.memory_pool.return_tensor(policy_tensor)
            self.memory_pool.return_tensor(value_tensor)
            
        self.cpu_time += time.time() - start_time - (time.time() - gpu_start)
        self.eval_count += states.shape[0]
        
        return policies_np, values_np
        
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        total_time = self.gpu_time + self.cpu_time
        if total_time > 0:
            return {
                'evaluations_per_second': self.eval_count / total_time,
                'gpu_time_percentage': 100 * self.gpu_time / total_time,
                'cpu_time_percentage': 100 * self.cpu_time / total_time,
                'total_evaluations': self.eval_count,
                'average_gpu_time_ms': 1000 * self.gpu_time / max(1, self.eval_count),
                'average_cpu_time_ms': 1000 * self.cpu_time / max(1, self.eval_count),
            }
        return {}
        

class BatchedEvaluator(Evaluator):
    """Wrapper that batches single evaluations for efficiency"""
    
    def __init__(self, base_evaluator: Evaluator, max_batch_size: int = 512):
        """Initialize batched evaluator
        
        Args:
            base_evaluator: The underlying evaluator to wrap
            max_batch_size: Maximum batch size
        """
        super().__init__(base_evaluator.config, base_evaluator.action_size)
        self.base_evaluator = base_evaluator
        self.max_batch_size = max_batch_size
        self.pending_states: List[np.ndarray] = []
        self.pending_masks: List[Optional[np.ndarray]] = []
        self.pending_futures: List[Dict] = []
        
    def evaluate(
        self,
        state: np.ndarray,
        legal_mask: Optional[np.ndarray] = None,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """Add to batch and evaluate when full"""
        # For now, just pass through to base evaluator
        # In a real implementation, this would batch requests
        return self.base_evaluator.evaluate(state, legal_mask, temperature)
        
    def evaluate_batch(
        self,
        states: np.ndarray,
        legal_masks: Optional[np.ndarray] = None,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pass through to base evaluator"""
        return self.base_evaluator.evaluate_batch(states, legal_masks, temperature)
        
    def flush(self) -> None:
        """Force evaluation of any pending states"""
        if self.pending_states:
            # Process pending batch
            states = np.stack(self.pending_states)
            masks = np.stack(self.pending_masks) if self.pending_masks[0] is not None else None
            
            policies, values = self.base_evaluator.evaluate_batch(states, masks)
            
            # Clear pending
            self.pending_states.clear()
            self.pending_masks.clear()
            self.pending_futures.clear()


class AlphaZeroEvaluator(Evaluator):
    """Evaluator for AlphaZeroNetwork models that works with MCTS"""
    
    def __init__(self, model, config: Optional[EvaluatorConfig] = None,
                 device: Optional[str] = None, action_size: Optional[int] = None):
        """Initialize AlphaZero evaluator
        
        Args:
            model: AlphaZeroNetwork model (torch.nn.Module)
            config: Evaluator configuration
            device: Device to run on
            action_size: Size of action space (will try to infer if not provided)
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for AlphaZeroEvaluator")
            
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create config if not provided
        if config is None:
            config = EvaluatorConfig(device=device)
        
        # Get action size from model if not provided
        if action_size is None:
            # Try to infer from model's policy head
            if hasattr(model, 'policy_head'):
                # Check for different possible final layer names
                if hasattr(model.policy_head, 'fc'):
                    action_size = model.policy_head.fc.out_features
                elif hasattr(model.policy_head, 'fc2'):
                    action_size = model.policy_head.fc2.out_features
                elif hasattr(model.policy_head, 'final_fc'):
                    action_size = model.policy_head.final_fc.out_features
            elif hasattr(model, 'config') and hasattr(model.config, 'num_actions'):
                action_size = model.config.num_actions
            elif hasattr(model, 'num_actions'):
                action_size = model.num_actions
            elif hasattr(model, 'metadata') and hasattr(model.metadata, 'num_actions'):
                action_size = model.metadata.num_actions
            else:
                raise ValueError("Cannot infer action_size from model. Please provide it explicitly.")
        
        # Initialize base class
        super().__init__(config, action_size)
        
        # Set up model
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        
        # Performance tracking
        self.eval_count = 0
        self.total_time = 0.0
    
    def evaluate(self, state: np.ndarray, legal_mask: Optional[np.ndarray] = None,
                temperature: float = 1.0) -> Tuple[np.ndarray, float]:
        """Evaluate a single position
        
        Args:
            state: Board state array
            legal_mask: Boolean mask for legal actions
            temperature: Temperature for policy
            
        Returns:
            policy: Probability distribution over actions
            value: Position evaluation [-1, 1]
        """
        # Add batch dimension
        state_batch = np.expand_dims(state, axis=0)
        legal_mask_batch = np.expand_dims(legal_mask, axis=0) if legal_mask is not None else None
        
        # Evaluate batch
        policies, values = self.evaluate_batch(state_batch, legal_mask_batch, temperature)
        
        return policies[0], values[0]
    
    def evaluate_batch(self, states: np.ndarray,
                      legal_masks: Optional[np.ndarray] = None,
                      temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch of positions
        
        Args:
            states: Batch of states (batch, channels, height, width)
            legal_masks: Optional legal move masks
            temperature: Temperature for policy
            
        Returns:
            policies: Policy distributions (batch, action_size)
            values: Position evaluations (batch,)
        """
        with torch.no_grad():
            # Convert to tensor if needed
            if isinstance(states, np.ndarray):
                states_tensor = torch.FloatTensor(states).to(self.device)
            else:
                states_tensor = states.to(self.device)
            
            # Forward pass through model
            log_policies, values = self.model(states_tensor)
            
            # Convert log probabilities to probabilities
            policies = torch.softmax(log_policies, dim=1)
            
            # Apply legal mask if provided
            if legal_masks is not None:
                if isinstance(legal_masks, np.ndarray):
                    masks = torch.FloatTensor(legal_masks).to(self.device)
                else:
                    masks = legal_masks.to(self.device)
                
                # Mask illegal moves
                policies = policies * masks
                
                # Renormalize
                policy_sums = policies.sum(dim=1, keepdim=True)
                policies = policies / (policy_sums + 1e-8)
            
            # Apply temperature if needed
            if temperature != 1.0 and temperature > 0:
                # Apply temperature to log probabilities for numerical stability
                log_policies = torch.log(policies + 1e-8) / temperature
                policies = torch.softmax(log_policies, dim=1)
            
            # Convert to numpy (keep on GPU if requested)
            if hasattr(self, '_return_torch_tensors') and self._return_torch_tensors:
                # Return torch tensors for GPU-based MCTS
                values = torch.tanh(values.squeeze(-1))  # Ensure values are in [-1, 1] range
                self.eval_count += len(states)
                return policies, values
            else:
                # Convert to numpy for compatibility
                policies_np = policies.cpu().numpy()
                values_np = values.cpu().numpy().squeeze(-1)
                
                # Ensure values are in [-1, 1] range
                values_np = np.tanh(values_np)
                
                self.eval_count += len(states)
                
                return policies_np, values_np
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save(self.model.state_dict(), path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
    
    def shutdown(self):
        """Cleanup resources"""
        # Clear CUDA cache if using GPU
        if hasattr(self, 'device') and self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.shutdown()
        except:
            pass