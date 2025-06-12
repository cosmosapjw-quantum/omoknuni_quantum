"""Registry for CUDA operations to provide consistent access"""

import torch
import logging
from typing import Optional, Callable, Dict

logger = logging.getLogger(__name__)

class CUDAOpsRegistry:
    """Registry for CUDA operations with fallback to CPU"""
    
    _instance = None
    _cuda_ops: Optional[object] = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Don't initialize immediately - wait for first use
        pass
    
    def _initialize_cuda_ops(self):
        """Initialize CUDA operations"""
        if not torch.cuda.is_available():
            logger.info("CUDA not available, will use CPU fallbacks")
            return
            
        try:
            # Try to load the compiled CUDA kernels
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            kernel_path = os.path.join(current_dir, 'mcts_cuda_kernels.cpython-312-x86_64-linux-gnu.so')
            
            if not os.path.exists(kernel_path):
                raise FileNotFoundError(f"CUDA kernel not found at {kernel_path}")
                
            torch.ops.load_library(kernel_path)
            
            # Import the module directly
            import mcts.gpu.mcts_cuda_kernels as cuda_kernels
            self._cuda_ops = cuda_kernels
            
            # Create namespace objects that can hold our operations
            class MCTSCUDAKernels:
                def __init__(self, cuda_module):
                    # Directly assign functions as attributes
                    self.find_expansion_nodes = cuda_module.find_expansion_nodes
                    self.batch_process_legal_moves = cuda_module.batch_process_legal_moves
                    self.batched_ucb_selection = cuda_module.batched_ucb_selection
                    self.parallel_backup = cuda_module.parallel_backup
                    self.batched_add_children = cuda_module.batched_add_children
                    self.evaluate_gomoku_positions = cuda_module.evaluate_gomoku_positions
                    self.fused_minhash_interference = cuda_module.fused_minhash_interference
                    self.phase_kicked_policy = cuda_module.phase_kicked_policy
                    self.quantum_interference = cuda_module.quantum_interference
            
            class CustomCUDAOps:
                def __init__(self, cuda_module):
                    # For backward compatibility
                    self.parallel_backup = cuda_module.parallel_backup
            
            # Register namespaces
            if not hasattr(torch.ops, 'mcts_cuda_kernels'):
                torch.ops.mcts_cuda_kernels = MCTSCUDAKernels(cuda_kernels)
                
            if not hasattr(torch.ops, 'custom_cuda_ops'):
                torch.ops.custom_cuda_ops = CustomCUDAOps(cuda_kernels)
            
            logger.info("CUDA operations loaded and registered successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load CUDA operations: {e}")
            logger.info("Will use CPU fallbacks")
    
    def _ensure_initialized(self):
        """Ensure CUDA operations are initialized"""
        if not self._initialized:
            self._initialize_cuda_ops()
            self._initialized = True
    
    def has_cuda_ops(self) -> bool:
        """Check if CUDA operations are available"""
        self._ensure_initialized()
        return self._cuda_ops is not None
    
    def get_op(self, op_name: str) -> Optional[Callable]:
        """Get a CUDA operation by name"""
        self._ensure_initialized()
        if self._cuda_ops and hasattr(self._cuda_ops, op_name):
            return getattr(self._cuda_ops, op_name)
        return None
    
    def find_expansion_nodes(self, *args, **kwargs):
        """Find nodes needing expansion"""
        op = self.get_op('find_expansion_nodes')
        if op:
            return op(*args, **kwargs)
        else:
            # Fallback implementation
            raise NotImplementedError("CPU fallback for find_expansion_nodes not implemented")
    
    def batch_process_legal_moves(self, *args, **kwargs):
        """Batch process legal moves"""
        op = self.get_op('batch_process_legal_moves')
        if op:
            return op(*args, **kwargs)
        else:
            raise NotImplementedError("CPU fallback for batch_process_legal_moves not implemented")
    
    def batched_ucb_selection(self, *args, **kwargs):
        """Batched UCB selection"""
        op = self.get_op('batched_ucb_selection')
        if op:
            return op(*args, **kwargs)
        else:
            raise NotImplementedError("CPU fallback for batched_ucb_selection not implemented")
    
    def parallel_backup(self, *args, **kwargs):
        """Parallel backup"""
        op = self.get_op('parallel_backup')
        if op:
            return op(*args, **kwargs)
        else:
            raise NotImplementedError("CPU fallback for parallel_backup not implemented")


# Global instance
cuda_ops = CUDAOpsRegistry()

def initialize_cuda_ops():
    """Initialize CUDA operations - call this early in your application"""
    cuda_ops._ensure_initialized()
    return cuda_ops.has_cuda_ops()