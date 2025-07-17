"""
Quantum-Inspired MCTS Implementation.

This module provides the main interface for quantum-augmented MCTS,
integrating temperature measurement, quantum corrections, and other
quantum-inspired enhancements.
"""
import torch
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .temperature import TemperatureMeasurer, BatchTemperatureMeasurer
from .corrections import QuantumCorrectionCalculator


@dataclass
class QuantumMCTSConfig:
    """Configuration for quantum-inspired MCTS enhancements"""
    # Standard MCTS parameters
    c_puct: float = 1.0
    
    # Quantum enhancement parameters
    enable_quantum_corrections: bool = True
    correction_gamma: float = 1.0
    correction_scale: float = 0.1
    
    # Temperature adaptation
    enable_temperature_adaptation: bool = True
    temperature_min_visits: int = 10
    
    # Batch processing
    batch_size: int = 32
    
    # Device configuration
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class QuantumMCTS:
    """
    Quantum-inspired enhancements for Monte Carlo Tree Search.
    
    This class provides quantum-inspired modifications to standard MCTS:
    1. Emergent temperature measurement from visit distributions
    2. One-loop quantum corrections to action selection
    3. GPU-accelerated batch processing
    
    The quantum corrections are based on path integral formulation
    and favor actions with lower curvature in the value landscape.
    """
    
    def __init__(self, config: Optional[QuantumMCTSConfig] = None):
        """
        Initialize quantum MCTS with given configuration.
        
        Args:
            config: Configuration object, uses defaults if None
        """
        self.config = config or QuantumMCTSConfig()
        
        # Initialize components
        self.temp_measurer = TemperatureMeasurer(gpu_device=self.config.device)
        self.batch_temp_measurer = BatchTemperatureMeasurer(
            gpu_device=self.config.device
        )
        self.corrector = QuantumCorrectionCalculator(
            c_puct=self.config.c_puct,
            gamma=self.config.correction_gamma,
            device=self.config.device
        )
        
        # Cache for temperature measurements
        self._temperature_cache: Dict[int, float] = {}
    
    def compute_augmented_puct_score(self, parent_node, child_node, 
                                   temperature: Optional[float] = None) -> float:
        """
        Compute quantum-augmented PUCT score for a single action.
        
        Args:
            parent_node: Parent MCTS node
            child_node: Child node (action) to score
            temperature: Optional temperature, measured if not provided
            
        Returns:
            Augmented PUCT score including quantum corrections
        """
        # Standard PUCT components
        q_value = child_node.q_value if hasattr(child_node, 'q_value') else (
            child_node.value_sum / max(1, child_node.visit_count)
        )
        
        # Exploration term
        prior = child_node.prior if hasattr(child_node, 'prior') else 0.0
        exploration = (self.config.c_puct * prior * 
                      torch.sqrt(torch.tensor(parent_node.visit_count)) / 
                      (1 + child_node.visit_count))
        
        score = q_value + exploration.item()
        
        # Add quantum corrections if enabled
        if self.config.enable_quantum_corrections:
            # Measure or use provided temperature
            if temperature is None and self.config.enable_temperature_adaptation:
                temperature = self._get_temperature(parent_node)
            else:
                temperature = temperature or 1.0
            
            # Compute quantum bonus
            quantum_bonus = self.corrector.compute_bonus(
                q_value=q_value,
                visits=child_node.visit_count,
                beta=1.0 / temperature
            )
            
            score += quantum_bonus * self.config.correction_scale
        
        return score
    
    def compute_augmented_puct_scores_batch(self, parent_node, 
                                          temperature: Optional[float] = None) -> torch.Tensor:
        """
        Compute quantum-augmented PUCT scores for all children in batch.
        
        Args:
            parent_node: Parent MCTS node with children
            temperature: Optional temperature, measured if not provided
            
        Returns:
            Tensor of augmented scores for all children
        """
        children = list(parent_node.children.values())
        if not children:
            return torch.tensor([])
        
        # Extract data for batch processing
        q_values = torch.tensor([
            c.q_value if hasattr(c, 'q_value') else c.value_sum / max(1, c.visit_count)
            for c in children
        ], device=self.config.device)
        
        visits = torch.tensor([c.visit_count for c in children], 
                            device=self.config.device)
        
        priors = torch.tensor([
            c.prior if hasattr(c, 'prior') else 0.0 for c in children
        ], device=self.config.device)
        
        # Standard PUCT exploration terms
        sqrt_parent = torch.sqrt(torch.tensor(parent_node.visit_count, 
                                            device=self.config.device))
        exploration = self.config.c_puct * priors * sqrt_parent / (1 + visits)
        
        scores = q_values + exploration
        
        # Add quantum corrections if enabled
        if self.config.enable_quantum_corrections:
            if temperature is None and self.config.enable_temperature_adaptation:
                temperature = self._get_temperature(parent_node)
            else:
                temperature = temperature or 1.0
            
            quantum_bonuses = self.corrector.compute_bonus_batch(
                q_values=q_values,
                visits=visits,
                beta=1.0 / temperature
            )
            
            scores += quantum_bonuses * self.config.correction_scale
        
        return scores
    
    def _get_temperature(self, node) -> float:
        """
        Get temperature for a node, using cache if available.
        
        Args:
            node: MCTS node
            
        Returns:
            Temperature (1/beta)
        """
        node_id = id(node)
        
        if node_id not in self._temperature_cache:
            beta = self.temp_measurer.measure(
                node, 
                min_visits=self.config.temperature_min_visits
            )
            # Convert beta to temperature
            self._temperature_cache[node_id] = 1.0 / beta if beta != float('inf') else 10.0
        
        return self._temperature_cache[node_id]
    
    def clear_temperature_cache(self):
        """Clear the temperature cache"""
        self._temperature_cache.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about quantum enhancements.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'temperature_cache_size': len(self._temperature_cache),
            'average_cached_temperature': (
                sum(self._temperature_cache.values()) / len(self._temperature_cache)
                if self._temperature_cache else 0.0
            ),
            'device': self.config.device,
            'quantum_corrections_enabled': self.config.enable_quantum_corrections,
            'temperature_adaptation_enabled': self.config.enable_temperature_adaptation
        }