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
        
        # Create dummy info dict to match EvaluatorPool interface
        info = {
            'weights': torch.ones(1, device=self.device),
            'diversity': torch.zeros(1, device=self.device),
            'confidence': torch.ones(1, device=self.device)
        }
        
        return values, policies, info
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluator statistics"""
        return {
            'evaluations': getattr(self.evaluator, 'num_evaluations', 0),
            'batch_size': getattr(self.evaluator, 'batch_size', 1)
        }