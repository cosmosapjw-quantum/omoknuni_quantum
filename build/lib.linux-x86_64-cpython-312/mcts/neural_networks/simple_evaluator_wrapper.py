"""Simple wrapper to make a single evaluator work like an evaluator pool"""

import torch
import numpy as np
from typing import Union, Tuple, Optional, Dict, Any


class SimpleEvaluatorWrapper:
    """Wraps a single evaluator to provide evaluator pool interface"""
    
    def __init__(self, evaluator, device='cuda'):
        self.evaluator = evaluator
        self.device = torch.device(device)
        
    def evaluate_batch(
        self, 
        states: Union[np.ndarray, torch.Tensor],
        legal_masks: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Evaluate batch of states
        
        Args:
            states: Batch of game states
            legal_masks: Optional legal move masks
            
        Returns:
            Tuple of (values, policies, info)
        """
        policies, values = self.evaluator.evaluate_batch(states, legal_masks)
        
        # Convert to tensors if needed
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, device=self.device)
        if not isinstance(policies, torch.Tensor):
            policies = torch.tensor(policies, device=self.device)
        
        # Create dummy info dict to match EvaluatorPool interface
        info = {
            'weights': torch.ones(1, device=self.device),
            'diversity': torch.zeros(1, device=self.device),
            'confidence': torch.ones(1, device=self.device)
        }
        
        # Return in correct order: values, policies, info
        return values, policies, info
        
    def shutdown(self):
        """Shutdown the evaluator wrapper"""
        # If the wrapped evaluator has a shutdown method, call it
        if hasattr(self.evaluator, 'shutdown'):
            self.evaluator.shutdown()
        
        # Clear any CUDA memory if using GPU
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.shutdown()
        except:
            pass
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluator statistics"""
        return {
            'evaluations': getattr(self.evaluator, 'num_evaluations', 0),
            'batch_size': getattr(self.evaluator, 'batch_size', 1)
        }