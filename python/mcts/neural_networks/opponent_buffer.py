"""Opponent buffer for population-based training

This module implements a buffer of historical model checkpoints
to provide diverse opponents during self-play, preventing strategy
collapse and improving robustness.
"""

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from mcts.core.evaluator import AlphaZeroEvaluator
from mcts.utils.config_system import AlphaZeroConfig


logger = logging.getLogger(__name__)


@dataclass
class OpponentEntry:
    """Entry in the opponent buffer"""
    model_path: Path
    iteration: int
    elo_rating: float
    win_rate: float
    timestamp: float
    metadata: Dict[str, any]


class OpponentBuffer:
    """Manages a buffer of historical opponents for diverse training
    
    This helps prevent strategy collapse by maintaining a population
    of models with different playing styles and strengths.
    """
    
    def __init__(self, config: AlphaZeroConfig, buffer_size: int = 10):
        """Initialize opponent buffer
        
        Args:
            config: AlphaZero configuration
            buffer_size: Maximum number of opponents to keep
        """
        self.config = config
        self.buffer_size = buffer_size
        self.opponents: List[OpponentEntry] = []
        self.device = torch.device(config.mcts.device)
        
        # Selection strategy parameters
        self.selection_temperature = getattr(config.training, 'opponent_selection_temperature', 1.0)
        self.use_elo_weighting = getattr(config.training, 'opponent_use_elo_weighting', True)
        self.min_elo_difference = getattr(config.training, 'opponent_min_elo_difference', 50.0)
        
    def add_opponent(self, model: nn.Module, iteration: int, elo_rating: float, 
                     win_rate: float, save_dir: Path, metadata: Optional[Dict] = None):
        """Add a new opponent to the buffer
        
        Args:
            model: The model to save
            iteration: Training iteration
            elo_rating: Model's ELO rating
            win_rate: Model's win rate against previous best
            save_dir: Directory to save model checkpoint
            metadata: Additional metadata to store
        """
        # Create unique filename
        timestamp = time.time()
        model_filename = f"opponent_iter{iteration}_elo{int(elo_rating)}_{int(timestamp)}.pt"
        model_path = save_dir / model_filename
        
        # Save model state
        torch.save({
            'model_state_dict': model.state_dict(),
            'iteration': iteration,
            'elo_rating': elo_rating,
            'win_rate': win_rate,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }, model_path)
        
        # Create opponent entry
        entry = OpponentEntry(
            model_path=model_path,
            iteration=iteration,
            elo_rating=elo_rating,
            win_rate=win_rate,
            timestamp=timestamp,
            metadata=metadata or {}
        )
        
        # Add to buffer
        self.opponents.append(entry)
        
        # Remove oldest opponents if buffer is full
        if len(self.opponents) > self.buffer_size:
            # Sort by a combination of recency and ELO
            # Keep diverse opponents - not just the strongest
            self._prune_buffer()
        
        logger.info(f"Added opponent from iteration {iteration} with ELO {elo_rating:.0f} to buffer")
        
    def _prune_buffer(self):
        """Prune buffer to maintain diversity"""
        if len(self.opponents) <= self.buffer_size:
            return
            
        # Keep the most recent model
        recent = self.opponents[-1]
        
        # Sort remaining by ELO
        others = sorted(self.opponents[:-1], key=lambda x: x.elo_rating)
        
        # Keep models at different ELO levels for diversity
        keep_indices = []
        
        # Always keep the weakest and strongest
        if len(others) > 0:
            keep_indices.append(0)  # Weakest
            if len(others) > 1:
                keep_indices.append(len(others) - 1)  # Strongest
        
        # Keep models spaced by ELO
        if len(others) > 2:
            num_to_keep = self.buffer_size - 2  # -2 for recent and one we already kept
            if num_to_keep > 0:
                # Select evenly spaced models by ELO
                step = len(others) // num_to_keep
                for i in range(1, num_to_keep):
                    idx = min(i * step, len(others) - 1)
                    if idx not in keep_indices:
                        keep_indices.append(idx)
        
        # Create new opponent list
        new_opponents = [others[i] for i in sorted(keep_indices)]
        new_opponents.append(recent)
        
        # Remove old model files
        for opponent in self.opponents:
            if opponent not in new_opponents and opponent.model_path.exists():
                opponent.model_path.unlink()
                
        self.opponents = new_opponents
        
    def get_random_opponent(self) -> Optional[AlphaZeroEvaluator]:
        """Get a random opponent from the buffer
        
        Returns:
            AlphaZeroEvaluator for the selected opponent, or None if buffer is empty
        """
        if not self.opponents:
            return None
            
        # Select opponent based on strategy
        if self.use_elo_weighting and len(self.opponents) > 1:
            # Weight selection by ELO difference
            weights = []
            current_elo = self.opponents[-1].elo_rating  # Most recent model's ELO
            
            for opponent in self.opponents[:-1]:  # Exclude current model
                elo_diff = abs(current_elo - opponent.elo_rating)
                # Prefer opponents with similar strength
                weight = 1.0 / (1.0 + elo_diff / self.min_elo_difference)
                weights.append(weight ** (1.0 / self.selection_temperature))
                
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                selected_idx = random.choices(range(len(self.opponents) - 1), weights=weights)[0]
                selected = self.opponents[selected_idx]
            else:
                # Fallback to uniform random
                selected = random.choice(self.opponents[:-1])
        else:
            # Uniform random selection
            selected = random.choice(self.opponents)
            
        # Load model
        return self._load_opponent(selected)
        
    def get_opponent_by_elo(self, target_elo: float) -> Optional[AlphaZeroEvaluator]:
        """Get opponent closest to target ELO rating
        
        Args:
            target_elo: Target ELO rating
            
        Returns:
            AlphaZeroEvaluator for the selected opponent, or None if buffer is empty
        """
        if not self.opponents:
            return None
            
        # Find closest ELO
        selected = min(self.opponents, key=lambda x: abs(x.elo_rating - target_elo))
        return self._load_opponent(selected)
        
    def _load_opponent(self, entry: OpponentEntry) -> AlphaZeroEvaluator:
        """Load opponent model from checkpoint
        
        Args:
            entry: Opponent entry to load
            
        Returns:
            AlphaZeroEvaluator with loaded model
        """
        # Create model architecture (same as current)
        from mcts.neural_networks.resnet_model import create_resnet_for_game
        
        # Get model configuration from current config
        model = create_resnet_for_game(
            game_type=self.config.game.game_type,
            input_channels=self.config.network.input_channels,
            num_blocks=self.config.network.num_res_blocks,
            num_filters=self.config.network.num_filters
        )
        model = model.to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(entry.model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create evaluator
        evaluator = AlphaZeroEvaluator(
            model=model,
            device=self.device
        )
        
        logger.info(f"Loaded opponent from iteration {entry.iteration} with ELO {entry.elo_rating:.0f}")
        
        return evaluator
        
    def get_self_play_opponents(self, num_opponents: int = 1) -> List[Optional[AlphaZeroEvaluator]]:
        """Get multiple opponents for self-play
        
        Args:
            num_opponents: Number of opponents to return
            
        Returns:
            List of evaluators (may contain None if not enough opponents)
        """
        if not self.opponents:
            return [None] * num_opponents
            
        opponents = []
        for _ in range(num_opponents):
            opponent = self.get_random_opponent()
            opponents.append(opponent)
            
        return opponents
        
    def get_buffer_stats(self) -> Dict[str, any]:
        """Get statistics about the opponent buffer
        
        Returns:
            Dictionary with buffer statistics
        """
        if not self.opponents:
            return {
                'num_opponents': 0,
                'elo_range': (0, 0),
                'iteration_range': (0, 0),
                'average_elo': 0
            }
            
        elos = [o.elo_rating for o in self.opponents]
        iterations = [o.iteration for o in self.opponents]
        
        return {
            'num_opponents': len(self.opponents),
            'elo_range': (min(elos), max(elos)),
            'iteration_range': (min(iterations), max(iterations)),
            'average_elo': sum(elos) / len(elos),
            'elo_diversity': max(elos) - min(elos) if len(elos) > 1 else 0
        }
        
    def clear(self):
        """Clear the opponent buffer and remove all saved models"""
        for opponent in self.opponents:
            if opponent.model_path.exists():
                opponent.model_path.unlink()
        self.opponents.clear()
        logger.info("Cleared opponent buffer")