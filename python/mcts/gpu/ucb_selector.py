"""UCB selection logic for MCTS tree

This module handles UCB (Upper Confidence Bound) selection,
separating selection algorithms from tree structure.
"""

import torch
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class UCBConfig:
    """Configuration for UCB selection"""
    c_puct: float = 1.4
    temperature: float = 1.0
    enable_virtual_loss: bool = True
    virtual_loss_value: float = -1.0
    device: str = 'cuda'


class UCBSelector:
    """Handles UCB-based action selection for MCTS
    
    Implements various UCB formulas including:
    - Standard UCB with exploration/exploitation tradeoff
    - Virtual loss for parallelization
    - Temperature-based exploration
    - Quantum-inspired variations
    """
    
    def __init__(self, config: UCBConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Cache for empty tensors
        self._empty_tensors_cache = None
        
    def select_single(self, parent_visits: int, child_visits: torch.Tensor,
                     child_values: torch.Tensor, child_priors: torch.Tensor,
                     c_puct: Optional[float] = None) -> int:
        """Select best child for a single node using UCB formula"""
        if len(child_visits) == 0:
            return -1
            
        c_puct = c_puct or self.config.c_puct
        
        # Calculate Q-values
        q_values = torch.where(
            child_visits > 0,
            child_values / child_visits,
            torch.zeros_like(child_values)
        )
        
        # Calculate exploration term
        sqrt_parent = torch.sqrt(torch.tensor(parent_visits, dtype=torch.float32, device=self.device))
        exploration = c_puct * child_priors * sqrt_parent / (1 + child_visits.float())
        
        # UCB scores
        ucb_scores = q_values + exploration
        
        # Select best
        return ucb_scores.argmax().item()
        
    def select_batch(self, parent_visits: torch.Tensor, 
                    children_visits: torch.Tensor,
                    children_values: torch.Tensor,
                    children_priors: torch.Tensor,
                    valid_mask: torch.Tensor,
                    c_puct: Optional[float] = None,
                    temperature: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch UCB selection for multiple nodes
        
        Args:
            parent_visits: (batch_size,) parent visit counts
            children_visits: (batch_size, max_children) child visit counts
            children_values: (batch_size, max_children) child value sums
            children_priors: (batch_size, max_children) child priors
            valid_mask: (batch_size, max_children) mask for valid children
            c_puct: UCB exploration constant
            temperature: Temperature for exploration
            
        Returns:
            selected_indices: (batch_size,) selected child indices
            ucb_scores: (batch_size,) UCB scores of selected children
        """
        batch_size = len(parent_visits)
        c_puct = c_puct or self.config.c_puct
        temperature = temperature or self.config.temperature
        
        # Initialize outputs
        selected_indices = torch.full((batch_size,), -1, dtype=torch.int32, device=self.device)
        selected_scores = torch.zeros(batch_size, device=self.device)
        
        # Early return for empty batch
        if batch_size == 0:
            return selected_indices, selected_scores
            
        # Ensure parent visits are at least 1
        parent_visits = torch.maximum(parent_visits, torch.ones_like(parent_visits))
        
        # Calculate Q-values
        q_values = torch.zeros_like(children_values)
        visited_mask = children_visits > 0
        if visited_mask.any():
            q_values[visited_mask] = children_values[visited_mask] / children_visits[visited_mask]
            
        # Calculate exploration term
        sqrt_parent = torch.sqrt(parent_visits.float()).unsqueeze(1)
        exploration = c_puct * children_priors * sqrt_parent / (1 + children_visits.float())
        
        # UCB scores
        ucb_scores = q_values + exploration
        
        # Apply temperature if needed
        if temperature != 1.0 and temperature > 0:
            ucb_scores = ucb_scores / temperature
            
        # Mask invalid children
        ucb_scores[~valid_mask] = -float('inf')
        
        # Select best children
        self._select_best_with_ties(
            ucb_scores, children_visits, children_priors, valid_mask,
            selected_indices, selected_scores
        )
        
        return selected_indices, selected_scores
        
    def _select_best_with_ties(self, ucb_scores: torch.Tensor,
                              children_visits: torch.Tensor,
                              children_priors: torch.Tensor,
                              valid_mask: torch.Tensor,
                              selected_indices: torch.Tensor,
                              selected_scores: torch.Tensor):
        """Select best children with proper tie-breaking
        
        This handles:
        - Stochastic selection for unvisited nodes
        - Random tie-breaking for equal UCB scores
        - Vectorized processing for efficiency
        """
        batch_size = ucb_scores.shape[0]
        
        # Check which nodes have valid children
        has_valid_children = valid_mask.any(dim=1)
        
        if not has_valid_children.any():
            return
            
        # Process nodes with valid children
        valid_nodes_mask = has_valid_children
        valid_ucb_scores = ucb_scores[valid_nodes_mask]
        valid_visits = children_visits[valid_nodes_mask]
        valid_priors = children_priors[valid_nodes_mask]
        valid_child_masks = valid_mask[valid_nodes_mask]
        
        # Check for all-unvisited nodes
        masked_visits = torch.where(valid_child_masks, valid_visits, torch.tensor(1, device=self.device))
        all_unvisited = (masked_visits == 0).all(dim=1)
        
        # Handle all-unvisited nodes with prior-based selection
        if all_unvisited.any():
            self._select_from_priors(
                valid_priors[all_unvisited],
                valid_child_masks[all_unvisited],
                selected_indices,
                selected_scores,
                valid_nodes_mask,
                all_unvisited
            )
            
        # Handle mixed visited/unvisited nodes with UCB
        mixed_mask = ~all_unvisited
        if mixed_mask.any():
            self._select_by_ucb(
                valid_ucb_scores[mixed_mask],
                valid_child_masks[mixed_mask],
                selected_indices,
                selected_scores,
                valid_nodes_mask,
                mixed_mask
            )
            
    def _select_from_priors(self, priors: torch.Tensor, masks: torch.Tensor,
                           selected_indices: torch.Tensor, selected_scores: torch.Tensor,
                           valid_nodes_mask: torch.Tensor, sub_mask: torch.Tensor):
        """Select from unvisited nodes based on priors"""
        # Add epsilon and normalize
        safe_priors = torch.where(masks, priors + 1e-6, torch.tensor(0.0, device=self.device))
        prior_sums = safe_priors.sum(dim=1, keepdim=True)
        normalized_priors = safe_priors / (prior_sums + 1e-8)
        
        # Use Gumbel-max trick for efficient sampling
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(normalized_priors) + 1e-8) + 1e-8)
        gumbel_scores = torch.where(masks,
                                   torch.log(normalized_priors + 1e-8) + gumbel_noise,
                                   torch.tensor(-float('inf'), device=self.device))
        selected = gumbel_scores.argmax(dim=1)
        
        # Update results
        node_indices = torch.where(valid_nodes_mask)[0][sub_mask]
        selected_indices[node_indices] = selected.to(torch.int32)
        
        # Set scores based on priors
        batch_indices = torch.arange(len(selected), device=self.device)
        selected_scores[node_indices] = priors[batch_indices, selected]
        
    def _select_by_ucb(self, ucb_scores: torch.Tensor, masks: torch.Tensor,
                      selected_indices: torch.Tensor, selected_scores: torch.Tensor,
                      valid_nodes_mask: torch.Tensor, sub_mask: torch.Tensor):
        """Select by UCB scores with tie-breaking"""
        # Find max UCB per node
        max_ucb_values = ucb_scores.max(dim=1, keepdim=True)[0]
        
        # Find ties within epsilon
        epsilon = 1e-8
        is_max = (ucb_scores >= max_ucb_values - epsilon) & masks
        
        # Random tie-breaking
        tie_break_noise = torch.rand_like(ucb_scores)
        tie_break_scores = torch.where(is_max, tie_break_noise, torch.tensor(-1.0, device=self.device))
        selected = tie_break_scores.argmax(dim=1)
        
        # Update results
        node_indices = torch.where(valid_nodes_mask)[0][sub_mask]
        selected_indices[node_indices] = selected.to(torch.int32)
        
        # Set UCB scores
        batch_indices = torch.arange(len(selected), device=self.device)
        selected_scores[node_indices] = ucb_scores[batch_indices, selected]
        
    def select_batch_with_virtual_loss(self,
                                     parent_visits: torch.Tensor,
                                     children_visits: torch.Tensor,
                                     children_values: torch.Tensor,
                                     children_priors: torch.Tensor,
                                     virtual_loss_counts: torch.Tensor,
                                     valid_mask: torch.Tensor,
                                     c_puct: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch UCB selection with virtual loss consideration"""
        if not self.config.enable_virtual_loss:
            return self.select_batch(
                parent_visits, children_visits, children_values,
                children_priors, valid_mask, c_puct
            )
            
        # Adjust visits and values with virtual loss
        effective_visits = children_visits + virtual_loss_counts
        effective_values = children_values + virtual_loss_counts.float() * self.config.virtual_loss_value
        
        return self.select_batch(
            parent_visits, effective_visits, effective_values,
            children_priors, valid_mask, c_puct
        )