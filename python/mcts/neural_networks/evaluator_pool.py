"""EvaluatorPool with meta-weighting ensemble

This module implements an evaluator pool that manages multiple neural networks
and combines their predictions using meta-weighting. This provides:
- Improved robustness through ensemble predictions
- Dynamic weighting based on confidence and accuracy
- Efficient batch processing across multiple models
- Support for different model architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import logging
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor

from ..core.evaluator import Evaluator, EvaluatorConfig

logger = logging.getLogger(__name__)


@dataclass 
class MetaWeightConfig:
    """Configuration for meta-weighting"""
    # Weight initialization
    initial_weight: float = 1.0
    
    # Weight update parameters
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0001
    
    # Confidence estimation
    use_uncertainty: bool = True
    temperature: float = 1.0
    
    # History tracking
    history_window: int = 100
    min_samples_for_update: int = 10
    
    # Ensemble parameters
    diversity_bonus: float = 0.1
    agreement_penalty: float = 0.05


class MetaWeightingModule(nn.Module):
    """Neural network for computing meta-weights"""
    
    def __init__(self, num_models: int, feature_dim: int = 32):
        super().__init__()
        
        self.num_models = num_models
        
        # Feature extraction for each model's prediction
        self.feature_net = nn.Sequential(
            nn.Linear(3, 16),  # (value, policy_entropy, confidence)
            nn.ReLU(),
            nn.Linear(16, feature_dim)
        )
        
        # Attention mechanism for weighting
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Final weight computation
        self.weight_net = nn.Sequential(
            nn.Linear(feature_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()  # Ensure positive weights
        )
        
    def forward(
        self,
        model_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute meta-weights for model predictions
        
        Args:
            model_features: (batch_size, num_models, 3) features
            
        Returns:
            Weights: (batch_size, num_models)
        """
        batch_size = model_features.shape[0]
        
        # Extract features for each model
        features = self.feature_net(model_features)  # (batch, num_models, feature_dim)
        
        # Apply self-attention to capture model relationships
        attended, _ = self.attention(features, features, features)
        
        # Compute weights
        weights = self.weight_net(attended).squeeze(-1)  # (batch, num_models)
        
        # Normalize weights
        weights = F.softmax(weights, dim=-1)
        
        return weights


class EvaluatorPool:
    """Pool of evaluators with meta-weighting ensemble
    
    This class manages multiple neural network evaluators and combines
    their predictions using learned meta-weights based on confidence,
    diversity, and historical performance.
    """
    
    def __init__(
        self,
        evaluators: List[Evaluator],
        meta_config: Optional[MetaWeightConfig] = None,
        device: Union[str, torch.device] = 'cuda'
    ):
        """Initialize evaluator pool
        
        Args:
            evaluators: List of evaluator instances
            meta_config: Meta-weighting configuration
            device: Device for computation
        """
        if len(evaluators) == 0:
            raise ValueError("Need at least one evaluator")
            
        self.evaluators = evaluators
        self.num_models = len(evaluators)
        self.config = meta_config or MetaWeightConfig()
        
        if isinstance(device, str):
            device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # Initialize meta-weighting module
        self.meta_weight_module = MetaWeightingModule(self.num_models).to(device)
        self.meta_optimizer = torch.optim.Adam(
            self.meta_weight_module.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Performance tracking
        self.performance_history = {
            i: deque(maxlen=self.config.history_window)
            for i in range(self.num_models)
        }
        
        # Current weights
        self.current_weights = torch.ones(self.num_models, device=device) / self.num_models
        
        # Thread pool for parallel evaluation
        self.executor = ThreadPoolExecutor(max_workers=self.num_models)
        
        # Statistics
        self.stats = {
            'evaluations': 0,
            'weight_updates': 0,
            'avg_diversity': 0.0,
            'avg_confidence': 0.0
        }
        
    def evaluate_batch(
        self,
        states: Union[List, torch.Tensor],
        use_meta_weights: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Evaluate a batch of states using ensemble
        
        Args:
            states: Batch of game states
            use_meta_weights: Whether to use learned weights
            
        Returns:
            Tuple of:
                - values: (batch_size,) tensor
                - policies: (batch_size, action_space) tensor
                - info: Dictionary with additional information
        """
        batch_size = len(states) if isinstance(states, list) else states.shape[0]
        
        # Parallel evaluation across models
        futures = []
        for i, evaluator in enumerate(self.evaluators):
            future = self.executor.submit(self._evaluate_single, evaluator, states)
            futures.append(future)
        
        # Collect results
        model_values = []
        model_policies = []
        model_features = []
        
        for i, future in enumerate(futures):
            value, policy, features = future.result()
            model_values.append(value)
            model_policies.append(policy)
            model_features.append(features)
        
        # Stack results
        model_values = torch.stack(model_values, dim=1)  # (batch, num_models)
        model_policies = torch.stack(model_policies, dim=1)  # (batch, num_models, actions)
        model_features = torch.stack(model_features, dim=1)  # (batch, num_models, 3)
        
        # Compute weights
        if use_meta_weights and self.num_models > 1:
            with torch.no_grad():
                weights = self.meta_weight_module(model_features)
        else:
            weights = self.current_weights.unsqueeze(0).expand(batch_size, -1)
        
        # Weighted ensemble
        ensemble_values = (model_values * weights).sum(dim=1)
        ensemble_policies = (model_policies * weights.unsqueeze(-1)).sum(dim=1)
        
        # Compute diversity and confidence metrics
        diversity = self._compute_diversity(model_policies)
        confidence = self._compute_confidence(model_values, model_policies)
        
        # Update statistics
        self.stats['evaluations'] += batch_size
        self.stats['avg_diversity'] = 0.9 * self.stats['avg_diversity'] + 0.1 * diversity.mean().item()
        self.stats['avg_confidence'] = 0.9 * self.stats['avg_confidence'] + 0.1 * confidence.mean().item()
        
        info = {
            'weights': weights,
            'diversity': diversity,
            'confidence': confidence,
            'individual_values': model_values,
            'individual_policies': model_policies
        }
        
        return ensemble_values, ensemble_policies, info
    
    def _evaluate_single(
        self,
        evaluator: Evaluator,
        states: Union[List, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate states with a single model
        
        Returns:
            Tuple of (values, policies, features)
        """
        # Get predictions
        values, policies = evaluator.evaluate_batch(states)
        
        # Ensure tensors
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, device=self.device)
        if not isinstance(policies, torch.Tensor):
            policies = torch.tensor(policies, device=self.device)
        
        # Move to device if needed
        values = values.to(self.device)
        policies = policies.to(self.device)
        
        # Compute features for meta-weighting
        policy_entropy = -(policies * torch.log(policies + 1e-8)).sum(dim=-1)
        confidence = 1.0 / (1.0 + policy_entropy)
        
        features = torch.stack([
            values,
            policy_entropy,
            confidence
        ], dim=-1)
        
        return values, policies, features
    
    def _compute_diversity(self, model_policies: torch.Tensor) -> torch.Tensor:
        """Compute diversity among model predictions
        
        Args:
            model_policies: (batch, num_models, actions)
            
        Returns:
            Diversity scores: (batch,)
        """
        # Compute pairwise KL divergences
        batch_size = model_policies.shape[0]
        diversity = torch.zeros(batch_size, device=self.device)
        
        for i in range(self.num_models):
            for j in range(i + 1, self.num_models):
                # KL divergence between models i and j
                kl_ij = F.kl_div(
                    torch.log(model_policies[:, i] + 1e-8),
                    model_policies[:, j],
                    reduction='none'
                ).sum(dim=-1)
                
                kl_ji = F.kl_div(
                    torch.log(model_policies[:, j] + 1e-8),
                    model_policies[:, i],
                    reduction='none'
                ).sum(dim=-1)
                
                # Symmetric KL
                diversity += (kl_ij + kl_ji) / 2
        
        # Normalize by number of pairs
        num_pairs = self.num_models * (self.num_models - 1) / 2
        diversity /= num_pairs
        
        return diversity
    
    def _compute_confidence(
        self,
        model_values: torch.Tensor,
        model_policies: torch.Tensor
    ) -> torch.Tensor:
        """Compute confidence scores for predictions
        
        Args:
            model_values: (batch, num_models)
            model_policies: (batch, num_models, actions)
            
        Returns:
            Confidence scores: (batch,)
        """
        # Value agreement (low variance = high confidence)
        value_var = model_values.var(dim=1)
        value_confidence = 1.0 / (1.0 + value_var)
        
        # Policy agreement (low entropy of mean policy = high confidence)
        mean_policy = model_policies.mean(dim=1)
        policy_entropy = -(mean_policy * torch.log(mean_policy + 1e-8)).sum(dim=-1)
        policy_confidence = 1.0 / (1.0 + policy_entropy)
        
        # Combined confidence
        confidence = 0.5 * value_confidence + 0.5 * policy_confidence
        
        return confidence
    
    def update_weights(
        self,
        states: torch.Tensor,
        true_values: torch.Tensor,
        true_outcomes: Optional[torch.Tensor] = None
    ):
        """Update meta-weights based on prediction accuracy
        
        Args:
            states: States that were evaluated
            true_values: True values (e.g., from MCTS)
            true_outcomes: Optional game outcomes for terminal states
        """
        # Get individual model predictions
        values, policies, info = self.evaluate_batch(states, use_meta_weights=False)
        individual_values = info['individual_values']
        
        # Compute losses for each model
        model_losses = []
        for i in range(self.num_models):
            loss = F.mse_loss(individual_values[:, i], true_values, reduction='none')
            model_losses.append(loss.mean())
            
            # Track performance
            self.performance_history[i].append(loss.mean().item())
        
        # Update meta-weights if enough history
        if all(len(hist) >= self.config.min_samples_for_update 
               for hist in self.performance_history.values()):
            
            # Compute features for meta-weighting
            # Create 3 features: value, policy entropy (dummy), confidence (dummy)
            individual_values = info['individual_values']
            batch_size, num_models = individual_values.shape
            model_features = torch.zeros(batch_size, num_models, 3, device=individual_values.device)
            model_features[:, :, 0] = individual_values  # Value predictions
            model_features[:, :, 1] = 0.5  # Dummy policy entropy
            model_features[:, :, 2] = 1.0  # Dummy confidence
            
            # Forward pass
            self.meta_weight_module.train()
            predicted_weights = self.meta_weight_module(model_features)
            
            # Ensemble prediction with current weights
            ensemble_pred = (individual_values * predicted_weights).sum(dim=1)
            
            # Meta loss: how well does the weighted ensemble perform?
            meta_loss = F.mse_loss(ensemble_pred, true_values)
            
            # Add diversity regularization
            diversity = self._compute_diversity(info['individual_policies'])
            diversity_bonus = -self.config.diversity_bonus * diversity.mean()
            
            # Total loss
            total_loss = meta_loss + diversity_bonus
            
            # Update
            self.meta_optimizer.zero_grad()
            total_loss.backward()
            self.meta_optimizer.step()
            
            # Update current weights
            with torch.no_grad():
                self.current_weights = predicted_weights.mean(dim=0)
            
            self.stats['weight_updates'] += 1
    
    def add_evaluator(self, evaluator: Evaluator):
        """Add a new evaluator to the pool"""
        self.evaluators.append(evaluator)
        self.num_models += 1
        
        # Reinitialize meta-weighting module
        old_state = self.meta_weight_module.state_dict()
        self.meta_weight_module = MetaWeightingModule(self.num_models).to(self.device)
        
        # Try to preserve some weights
        try:
            self.meta_weight_module.load_state_dict(old_state, strict=False)
        except:
            logger.warning("Could not preserve meta-weights after adding evaluator")
        
        # Reset optimizer
        self.meta_optimizer = torch.optim.Adam(
            self.meta_weight_module.parameters(),
            lr=self.config.learning_rate
        )
        
        # Add performance tracking
        self.performance_history[self.num_models - 1] = deque(
            maxlen=self.config.history_window
        )
        
        # Reset weights
        self.current_weights = torch.ones(self.num_models, device=self.device) / self.num_models
    
    def get_statistics(self) -> Dict:
        """Get pool statistics"""
        stats = dict(self.stats)
        
        # Add per-model statistics
        for i in range(self.num_models):
            if len(self.performance_history[i]) > 0:
                stats[f'model_{i}_avg_loss'] = np.mean(list(self.performance_history[i]))
            stats[f'model_{i}_weight'] = self.current_weights[i].item()
        
        return stats
    
    def save_state(self, path: str):
        """Save pool state including meta-weights"""
        state = {
            'meta_weights': self.meta_weight_module.state_dict(),
            'optimizer': self.meta_optimizer.state_dict(),
            'current_weights': self.current_weights,
            'performance_history': {
                i: list(hist) for i, hist in self.performance_history.items()
            },
            'stats': self.stats
        }
        torch.save(state, path)
    
    def load_state(self, path: str):
        """Load pool state"""
        state = torch.load(path, map_location=self.device, weights_only=False)
        
        self.meta_weight_module.load_state_dict(state['meta_weights'])
        self.meta_optimizer.load_state_dict(state['optimizer'])
        self.current_weights = state['current_weights']
        
        # Restore history
        for i, hist in state['performance_history'].items():
            self.performance_history[int(i)] = deque(hist, maxlen=self.config.history_window)
        
        self.stats = state['stats']
    
    def shutdown(self):
        """Shutdown the evaluator pool and clean up resources"""
        logger.info("Shutting down evaluator pool...")
        
        # Shutdown the thread pool executor
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
            logger.debug("Thread pool executor shut down")
        
        # Clear any GPU memory if possible
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
    def __del__(self):
        """Cleanup when object is garbage collected"""
        try:
            self.shutdown()
        except Exception as e:
            logger.warning(f"Error during evaluator pool cleanup: {e}")
    
    def __enter__(self):
        """Context manager support"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.shutdown()
        return False